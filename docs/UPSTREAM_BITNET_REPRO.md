# Upstream Issue Report: AVX2 SIGSEGV in `dequantize_row_i2_s` on Prompts > 256 Tokens

**Repository**: `microsoft/BitNet`  
**Component**: `bitnet.cpp` / `3rdparty/llama.cpp` (`libggml-base`)  
**Target Model**: `bitnet-embedding-270m` (`bitnet-embeddings-270m-bf16-i2_s.gguf`)  
**Architecture**: x86_64 AVX2 SIMD  
**Host Environment**: macOS Darwin x86_64, Intel Core i9-9880H, Apple Clang 15.0  

---

## 1. Summary

When executing dense embeddings via `llama-embedding` with the official Microsoft `bitnet-embeddings-270m-bf16-i2_s.gguf` model on an AVX2 host, any prompt resulting in **257 or more tokens** triggers an immediate segmentation fault (`EXC_BAD_ACCESS` / `SIGSEGV`, exit code `-11` / `139`) in `dequantize_row_i2_s + 153` inside `libggml-base.0.dylib`.

Prompts of **256 tokens or fewer** complete cleanly with exit code `0` and correct L2-normalized unit embeddings. The crash boundary is exact at token 257 and is independent of prompt text content or vocabulary distribution.

In contrast, the generative 2B ternary model (`microsoft/BitNet-b1.58-2B-4T`) executes prompts with 377, 653, 752, and 1,877 tokens on the exact same AVX2 host without memory faults, confirming the issue is specific to the 270M embedding model's batch decoding.

---

## 2. Minimal Reproducible Example

Build `bitnet.cpp` targeting AVX2 on macOS x86_64:
```bash
python setup_env.py -md models/bitnet-embedding-270m -q i2_s
```

### Passing Case (256 tokens &mdash; Returns RC 0):
```bash
# 254 words + BOS/EOS = 256 tokens
python3 -c "print('word ' * 254)" | ./build/bin/llama-embedding \
  -m models/bitnet-embedding-270m/bitnet-embeddings-270m-bf16-i2_s.gguf \
  -t 4 \
  -c 512 \
  --pooling last \
  --embd-normalize 2 \
  --embd-output-format array \
  -ngl 0 \
  -f /dev/stdin
echo "Exit code: $?" # Exits 0
```

### Crashing Case (257 tokens &mdash; Crashes with SIGSEGV / Exit Code -11):
```bash
# 255 words + BOS/EOS = 257 tokens
python3 -c "print('word ' * 255)" | ./build/bin/llama-embedding \
  -m models/bitnet-embedding-270m/bitnet-embeddings-270m-bf16-i2_s.gguf \
  -t 4 \
  -c 512 \
  --pooling last \
  --embd-normalize 2 \
  --embd-output-format array \
  -ngl 0 \
  -f /dev/stdin
echo "Exit code: $?" # Exits 139 / -11 (EXC_BAD_ACCESS)
```

---

## 3. LLDB Diagnostic Backtrace

```text
* thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x10d8a9000)
    frame #0: 0x0000000109be35f9 libggml-base.0.dylib`dequantize_row_i2_s + 153
libggml-base.0.dylib`dequantize_row_i2_s:
->  0x109be35f9 <+153>: movzbl (%rdi,%rbx), %r14d
    0x109be35fd <+157>: movl   %r14d, %r15d
    0x109be3600 <+160>: andl   $0x3, %r15d
```

### Register Dump at Fault:
- Faulting address: `0x10d8a9000` (page boundary violation on out-of-bounds byte read).
- Base pointer `%rdi` points to the quantized block buffer; index offset `%rbx` exceeds the mapped tensor buffer allocation when the batch sequence exceeds 256 tokens during AVX2 row dequantization.

---

## 4. Empirical Sweep Observations

A fine-grained token boundary sweep was conducted across both synthetic repetitive prose and realistic legal statutory citations containing Unicode punctuation (`12 CFR § 229.10(c)(1)(vi) “Next-day” exception for $5,525.`):

| Token Count | Input Content | Exit Code | Observed Behavior |
| :--- | :--- | :--- | :--- |
| **239 tokens** | Prose (`word * 237`) | `0` | Clean 640-dim embedding |
| **240 tokens** | Prose (`word * 238`) | `0` | Clean 640-dim embedding |
| **248 tokens** | Dense legal statute citations | `0` | Clean 640-dim embedding |
| **256 tokens** | Dense legal statute citations | `0` | Clean 640-dim embedding |
| **257 tokens** | Dense legal statute citations | `-11` (`SIGSEGV`) | `EXC_BAD_ACCESS` in `dequantize_row_i2_s` |
| **257 tokens** | Prose (`word * 255`) | `-11` (`SIGSEGV`) | `EXC_BAD_ACCESS` in `dequantize_row_i2_s` |
| **300 tokens** | Federal Register clause | `-11` (`SIGSEGV`) | `EXC_BAD_ACCESS` in `dequantize_row_i2_s` |
| **642 tokens** | Full statute section | `-11` (`SIGSEGV`) | `EXC_BAD_ACCESS` in `dequantize_row_i2_s` |

### Parameters Tested that Do Not Mitigate the Fault:
- `--ubatch-size` (`-ub 256`, `-ub 512`, `-ub 1024`, `-ub 4096`)
- `--ctx-size` (`-c 512`, `-c 2048`, `-c 4096`, `-c 32768`)
- Thread counts (`-t 1`, `-t 2`, `-t 4`, `-t 8`)
- Input delivery methods (`-f /dev/stdin` vs `-f /tmp/prompt.txt`)

---

## 5. Downstream Workaround

In the `event-driven-ftaas` inference service, this defect is mitigated at the application layer via:
1. Mathematical byte shortcut: Prompts $\le 238$ UTF-8 bytes mathematically cannot exceed 238 tokens (each token requires $\ge 1$ byte + 2 BOS/EOS tokens), safely bypassing the tokenizer.
2. Tokenizer pre-check: Prompts $> 238$ bytes are pre-counted with `llama-tokenize` against embedding weights with a strict ceiling of `MAX_SAFE_EMBED_TOKENS = 240` (clamped at 255), returning HTTP 413 `ContextOverflowError` before invoking the embedding binary.
3. Fail-closed safety: If `llama-tokenize` is unavailable or fails, requests $> 238$ bytes return HTTP 503 `TokenizerUnavailableError` rather than risking process crash.
