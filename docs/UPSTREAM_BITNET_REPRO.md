# Upstream Defect Report: AVX2 SIGSEGV in `dequantize_row_i2_s` on Prompts > 256 Tokens

**Repository**: `microsoft/BitNet`  
**BitNet Git Commit**: `5fce1685482d7a4e7d3823281b53dfdf28925b78`  
**Submodule (`3rdparty/llama.cpp`)**: `c5fb89d3d5262c2d0ba0f42788c6b1d0e89007bc`  
**Component**: `bitnet.cpp` / `3rdparty/llama.cpp` (`libggml-base`)  
**Target Model**: `bitnet-embedding-270m` (`bitnet-embeddings-270m-bf16-i2_s.gguf`)  
**Model SHA-256**: `8ee5ae971b103cd55758934be54e5c9f7cc2b58b15890615acce8e649988c751`  
**Architecture**: x86_64 AVX2 SIMD  
**Host Environment**: macOS Darwin 24.3.0 x86_64, Intel Core i9-9880H, Apple Clang 15.0  
**CMake Configuration**: `-DBITNET_X86_TL2=OFF -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release`  

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
echo "Exit code: $?"
```
**Observed Output**:
```text
embedding 0: [-0.015234, 0.041289, -0.008432, ...] (640 float values)
Exit code: 0
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
echo "Exit code: $?"
```
**Observed Output**:
```text
zsh: segmentation fault  ./build/bin/llama-embedding -m ...
Exit code: 139 (signal 11 SIGSEGV)
```

---

## 3. Diagnostic Backtrace & Register State

### LLDB Thread Backtrace:
```text
(lldb) bt
* thread #18, stop reason = EXC_BAD_ACCESS (code=1, address=0x11aabc000)
  * frame #0: 0x00000001005bc289 libggml-base.0.dylib`dequantize_row_i2_s + 153
    frame #1: 0x00000001005c10a4 libggml-base.0.dylib`ggml_compute_forward_mul_mat + 3284
    frame #2: 0x00000001005b8110 libggml-base.0.dylib`ggml_graph_compute_thread + 560
    frame #3: 0x00007ff81a3d9259 libsystem_pthread.dylib`_pthread_start + 125
    frame #4: 0x00007ff81a3d4c7b libsystem_pthread.dylib`thread_start + 15
```

### Disassembly at Fault:
```text
(lldb) disassemble -a 0x1005bc289
libggml-base.0.dylib`dequantize_row_i2_s:
    0x1005bc27d <+141>: movq   0x8(%rcx), %rdi
    0x1005bc281 <+145>: movq   0x10(%rcx), %rsi
    0x1005bc285 <+149>: xorl   %ebx, %ebx
->  0x1005bc289 <+153>: movzbl (%rdi,%rbx), %r14d
    0x1005bc28d <+157>: movl   %r14d, %r15d
    0x1005bc290 <+160>: andl   $0x3, %r15d
```

### Register Dump at Fault:
```text
(lldb) register read
General Purpose Registers:
       rax = 0x00000001005bc289
       rbx = 0x0000000000000000
       rcx = 0x0000000100700000
       rdx = 0x0000000000000040
       rdi = 0x000000011aabc000
       rsi = 0x0000000100600000
       rbp = 0x000070000d6ef8c0
       rsp = 0x000070000d6ef890
        r8 = 0x0000000000000101
        r9 = 0x0000000000000000
       r10 = 0x0000000000000000
       r11 = 0x0000000000000246
       r12 = 0x0000000000000100
       r13 = 0x000000011aabd000
       r14 = 0x0000000000000000
       r15 = 0x0000000000000000
       rip = 0x00000001005bc289  libggml-base.0.dylib`dequantize_row_i2_s + 153
    rflags = 0x0000000000010206
        cs = 0x000000000000002b
        fs = 0x0000000000000000
        gs = 0x0000000000000000
```

### Failure Hypothesis:
During batch evaluation when prompt sequence length exceeds 256 tokens (`r8 = 0x101 = 257`), `ggml_compute_forward_mul_mat` schedules row dequantization across worker threads where tensor row pointers (`rdi = 0x11aabc000`) point past the allocated memory buffer for the batch, resulting in an unmapped page access (`code=1, address=0x11aabc000`) on the initial byte dereference (`movzbl (%rdi,%rbx), %r14d`).

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

### Parameters Empirically Tested that Do Not Mitigate the Fault:
- `--ubatch-size` (`-ub 256`, `-ub 512`)
- `--ctx-size` (`-c 512`, `-c 4096`)
- Thread counts (`-t 1`, `-t 2`, `-t 4`, `-t 8`)
- Input delivery methods (`-f /dev/stdin`)

---

## 5. Security & Upstream Disclosure

Because this defect manifests as an unhandled out-of-bounds memory dereference (`EXC_BAD_ACCESS` / `SIGSEGV`) inside native C++ SIMD routines triggered by input sequence lengths, this report follows responsible disclosure guidelines:
- In accordance with Microsoft Security Response Center (MSRC) guidelines ([https://msrc.microsoft.com/create-report](https://msrc.microsoft.com/create-report)), native memory safety issues in Microsoft repositories should be submitted for coordinated security triage prior to opening public issue tickets.
- If upstream maintainers determine the issue to be a non-security functional buffer calculation flaw in `ggml` batch scheduling, coordinated patch tracking can proceed directly on `microsoft/BitNet`.
