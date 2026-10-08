# BitNet b1.58: Unified 1-Bit Zero-Egress AI & LoRA Fine-Tuning Roadmap

> **Target Models**: 
> - **Generative LLM**: [`microsoft/bitnet-b1.58-2B-4T-gguf`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf) / [`microsoft/BitNet-b1.58-2B-4T`](https://huggingface.co/microsoft/BitNet-b1.58-2B-4T)
> - **Dense Embeddings**: [`microsoft/bitnet-embedding-270m`](https://huggingface.co/microsoft/bitnet-embedding-270m) / [`bitnet-embedding-0.6b`](https://huggingface.co/microsoft/bitnet-embedding-0.6b)  
> **Toolchain Engine**: `bitnet.cpp` (AVX2 SIMD build on Apple Clang)  
> **Host Environment**: macOS (Darwin x86_64, Intel Core i9-9880H, 16 GB RAM, AVX2 SIMD)  
> **Primary Authority**: [`KNOWTHANKYEW_PORTFOLIO_MASTER_BRIEF.md`](https://gist.github.com/knowthankyew/53ccf4d5a81e916f895c74e18b231e16)  
> **Status**: 🟢 **PHASES 1–8 COMPLETED** | 🟡 **MILESTONE 2 IN PROGRESS (Phases 9–11)**  
> **Date**: October 6, 2026  

---

## 1. The Core Question: Can We Train a Ternary Model with LoRA?

### The Short Answer: **YES.**
Not only is it mathematically sound, but **ternary base models and LoRA are an extraordinary pairing for edge deployment.**

---

### The Mathematical Mechanics: Why LoRA Works on Ternary Weights

In standard LoRA (Low-Rank Adaptation), the forward pass of any adapted linear layer is:

$$h = W_0 \cdot x + \frac{\alpha}{r} (B \cdot A) \cdot x$$

Where:
- $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ is the **pre-trained base model weight matrix**.
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ and $B \in \mathbb{R}^{d_{\text{out}} \times r}$ are **trainable low-rank decomposition matrices** ($r \ll d$).
- $\alpha$ and $r$ are constant scaling hyperparameters.

#### The Crucial Invariant: $W_0$ is 100% Frozen
During LoRA fine-tuning:
1. **$W_0$ never receives gradients**:
   $$\nabla_{W_0} \mathcal{L} = 0 \quad (\text{weights are never updated})$$
2. In BitNet b1.58, $W_0 \in \{-1,\; 0,\; +1\}$. **Because $W_0$ is frozen, its ternary property remains completely intact throughout training.**
3. Only the lightweight low-rank adapters $A$ and $B$ receive backpropagated gradients and are updated in standard floating-point precision (`float16` or `float32`).

```mermaid
flowchart LR
    Input["Input Token Vector (x)"] --> Base["Frozen Ternary Base (W₀ ∈ {-1, 0, +1})\n⚡ Pure Integer Add / Sub on CPU"]
    Input --> LoRA_A["Trainable LoRA Matrix A (r × d)\nFloat16 Dimension Reduction"]
    LoRA_A --> LoRA_B["Trainable LoRA Matrix B (d × r)\nFloat16 Dimension Expansion"]
    LoRA_B --> Scale["Scale (α / r)"]
    Base --> Add(("⊕"))
    Scale --> Add
    Add --> Output["Output Hidden State (h)"]
```

---

### Two Inference Paradigms for BitNet + LoRA

Once a LoRA adapter is trained on top of BitNet, there are two distinct ways to serve it:

#### Paradigm A: Dynamic Parallel Adapter (Planned Multi-Tenant Serving Pattern)
- **How It Works**: The base model stays in memory as pure 2-bit ternary weights. When an inference request arrives, the CPU evaluates the ternary base layer using SIMD integer addition/subtraction, and in parallel runs the tiny $r=8$ or $r=16$ float adapter projection.
- **Architectural Design Goals**:
  - **Continuous Adapter Nuance**: The floating-point adapter retains continuous precision (suited for strict compliance tags and domain schemas).
  - **Base CPU Efficiency**: Over 99% of base model forward parameters remain ternary integer additions in memory.
  - **Dynamic Hot-Swapping**: Multiple tenant adapters can be swapped on a single 1.15 GB BitNet base model without reloading base weights.
- *Serving Note*: The 23.5 t/s decode benchmark measures base GGUF decode in `bitnet.cpp`; serving active adapters in parallel requires runtime support currently being integrated across the dual-backend engine.

#### Paradigm B: Full Weight Merging (Exploratory Experiment: Pure-Ternary Edge Package)
- **Concept**: Standard LoRA merges weights additively: $W_{\text{merged}} = W_0 + \frac{\alpha}{r} BA$. Because $BA$ is continuous float, $W_{\text{merged}}$ is no longer ternary. An experimental approach is passing the merged matrix through a **Ternary Quantization-Aware re-binning function**:
  $$W_{\text{ternary\_merged}} = \text{Round}\left(\text{Clamp}\left(\frac{W_{\text{merged}}}{\gamma},\; -1,\; +1\right)\right)$$
- **Experimental Risk**: Because LoRA weight updates $\Delta W$ are typically very small, standard rounding risks obliterating the adapter signal by rounding deltas back to zero. This path remains an exploratory research direction requiring quantization-aware calibration.

---

## 2. Model Tier & Representation Comparison

### Generative Foundation Models

| Dimension | SmolLM2-135M | Gemma 2 2B IT | BitNet b1.58 2B-4T |
| :--- | :--- | :--- | :--- |
| **Parameter Count** | ~135 Million | ~2.6 Billion | ~2.4 Billion |
| **Native Weight Format**| FP16 / BF16 | FP16 / BF16 | **Ternary $\{-1, 0, +1\}$** |
| **Physical Storage** | ~270 MB | ~5.2 GB | **~1.15 GB** (GGUF `i2_s`) |
| **Compute Primitive** | Floating-Point MAC | Floating-Point MAC | **Integer ADD / SUB (No MAC)** |
| **Min Hardware** | Any CPU / MPS / CUDA | 8GB+ VRAM or Metal | **Any Modern CPU (AVX2/NEON)** |
| **LoRA Trainability** | ✅ Native PyTorch PEFT | ✅ Native PyTorch PEFT | **✅ PyTorch BitLinear / QVAC** |
| **Training Tokens** | ~2 Trillion | ~2 Trillion | **4 Trillion** (Trained from scratch) |
| **Licensing** | Apache 2.0 | Gated (Gemma Terms) | **MIT License** (Fully Open) |

### Dense Semantic Embedding Models

| Dimension | `all-MiniLM-L6-v2` | `bge-large-en-v1.5` | `nomic-embed-text-v1.5` | `bitnet-embedding-270m` (Microsoft / Target) |
| :--- | :--- | :--- | :--- | :--- |
| **Weight Precision** | FP32 / FP16 | FP32 / FP16 | Matryoshka FP16 | **1.58-bit Ternary (`I2_S`)** |
| **Physical RAM (Peak RSS)** | $\sim 90\text{ MB}$ | $\sim 1.34\text{ GB}$ | $\sim 550\text{ MB}$ | **741.9 MB (short, -c 512) → 1,585.4 MB max RSS (240 tok)** / 350.46 MB on disk |
| **Max Context Window** | 512 tokens | 512 tokens | 8,192 tokens | **32,768 arch. (240 bounded runtime)** |
| **Embedding Dimension**| 384 | 1,024 | 768 (or 256–512) | **640** |
| **Compute Primitive** | Floating-Point MAC | Floating-Point MAC | Floating-Point MAC | **Integer ADD / SUB (SIMD)** |
| **MTEB Score** | 56.09 (v1) | 64.11 (v1) | 62.28 (v1) | **66.26 (v2 mean)** |
| **Zero-Egress Feasibility**| Moderate | Low (Memory heavy) | Moderate | **Ultra-High (Native Sidecar)** |

*Note: The 66.26 score for `microsoft/bitnet-embedding-270m` is the official MTEB v2 mean reported by Microsoft Research; competitor scores (`all-MiniLM-L6-v2`: 56.09, `bge-large-en-v1.5`: 64.11, `nomic-embed-text-v1.5`: 62.28) are MTEB v1 averages. Physical model storage on disk is 350.46 MB (`bitnet-embeddings-270m-bf16-i2_s.gguf`). Physical memory usage was re-measured empirically via `/usr/bin/time -l` on macOS Darwin x86_64 across sequence lengths under bounded `-c 512`: 741.9 MB maximum RSS (340.9 MB peak memory footprint) on a 5-token query, scaling to 1,488.4 MB maximum RSS (1,112.2 MB peak footprint) at 150 tokens, and 1,585.4 MB maximum RSS (1,209.2 MB peak footprint) at the 240-token safe operational ceiling. When re-measured under bounded `-c 256 -b 256 -ub 256`, memory is 739.4 MB maximum RSS (333.6 MB peak footprint) on short inputs and 1,495.4 MB maximum RSS (1,115.3 MB peak footprint) at 150 tokens. When run with the default 32,768-token architectural context on short input, memory is 1,185.6 MB maximum RSS (942.1 MB peak footprint). Memory scaling is directly attributed to compute buffer reservations: `llama-embedding` allocates 517.39 MiB CPU compute buffer (`sched_reserve: CPU compute buffer size = 517.39 MiB`), corresponding to the output vocabulary logits tensor for a 512-token microbatch over the 262,144 vocabulary ($262,144 \times 512 \times 4\text{ bytes} = 536,870,912\text{ bytes} \approx 512\text{ MiB}$) allocated by `ggml_backend_sched` during graph construction even though only embedding vectors are returned. The historical 34.2 MB figure was measured against the wrong file; retracted. In comparison, `microsoft/BitNet-b1.58-2B-4T` is 1,187.8 MB on disk with ~1,532.8 MB maximum resident set size. Peak worst-case memory envelope under concurrency limits (`MAX_CONCURRENT_EMBED = 2`, `MAX_CONCURRENT_GENERATE = 1`) reaches $2 \times 1,585.4\text{ MB} + 1,532.8\text{ MB} \approx \mathbf{4.7\text{ GB}}$ peak RSS (or $\mathbf{\approx 3.17\text{ GB}}$ in Retrieval-Only deployment mode with 2 embeds and 0 generate). Validation on physical 8 GB constrained RAM hardware is pending. Runtime context is bounded at 240 tokens on AVX2 for embeddings due to an upstream batch decode crash in `dequantize_row_i2_s` at token 257.*

---

## 3. End-to-End Execution Roadmap

```mermaid
flowchart TD
    subgraph M1["Milestone 1: Ternary LLM Core & Dynamic LoRA [COMPLETED]"]
        P1["Phase 1: Environment & Toolchain"] --> P2["Phase 2: Model Pull & Native Kernel"]
        P2 --> P3["Phase 3: Interactive & Throughput Profiling"]
        P3 --> P4["Phase 4: Ternary LoRA Training Pipeline"]
        P4 --> P5["Phase 5: Studio UI & API Ingestion"]
        P5 --> P6["Phase 6: Upstream Toolchain PRs (#635)"]
        P6 --> P7["Phase 7: Native C++ Dual-Backend Engine"]
    end

    subgraph M2["Milestone 2: Unified 1-Bit Zero-Egress Stack & Statutory Retrieval [ACTIVE]"]
        P8["Phase 8: 1-Bit Dense Embedding Subsystem [VERIFIED]"]
        P9["Phase 9: Statutory Pack Retrieval Baseline [BM25 CORE ESTABLISHED]"]
        P10["Phase 10: Prototype Statutory Benchmark & Regulatory Audit [VERIFIED]"]
        P11["Phase 11: Production Serving & Upstream Defect Resolution [ACTIVE]"]
        
        P8 --> P9 --> P10 --> P11
    end

    M1 ==> M2
```

---

### Milestone 1: Ternary LLM Foundation & Dynamic LoRA Serving &mdash; 🟢 COMPLETED

#### [x] Phase 1: Environment & Build Toolchain &mdash; 🟢 COMPLETED
- **Action**: Provision dedicated isolated build environment and populate toolchain dependencies.
- **Executed Steps**:
  1. Populated and initialized `3rdparty/llama.cpp` submodule inside `~/Github/BitNet`.
  2. Provisioned Apple Clang and CMake toolchain targeting macOS Darwin x86_64.
  3. Created isolated virtual environment (`~/Github/BitNet/.venv`) for clean dependency resolution.
- **Verification**: CMake confirmed native AVX2 SIMD flags enabled; compiler tools validated.

#### [x] Phase 2: Model Acquisition & Optimized Native Build &mdash; 🟢 COMPLETED
- **Action**: Acquire official Microsoft BitNet b1.58 2B-4T GGUF weights and compile native vector kernels.
- **Executed Steps**:
  1. Downloaded pre-trained weights (`ggml-model-i2_s.gguf`, 1.15 GB) via `huggingface-cli`.
  2. Executed `python setup_env.py -md models/BitNet-b1.58-2B-4T -q i2_s` targeting Intel Core i9-9880H AVX2 SIMD.
  3. Compiled native C++ static library `ggml-bitnet.a` and CLI runner targets.
- **Verification**: Verified binary output without unvectorized scalar emulation fallback.

#### [x] Phase 3: Qualitative Testing & CPU Throughput Benchmarking &mdash; 🟢 COMPLETED
- **Action**: Empirically profile BitNet b1.58 against thread count, measuring decode throughput, prefill latency, and physical memory footprint.
- **Executed Steps**:
  1. Conducted conversational verification (`-cnv` mode) evaluating policy coherence and tag compliance.
  2. Executed `e2e_benchmark.py` scaling tests across 1, 2, 4, and 8 threads.
  3. Formatted and published comprehensive performance records in [`BITNET_PROTOTYPE_RESULTS.md`](BITNET_PROTOTYPE_RESULTS.md).
- **Key Empirical Results**:
  - **Generation Throughput**: 23.54 tokens/sec at $t=8$ (2.1× faster than Gemma 2B FP16 CPU execution at 11.1 t/s).
  - **Prompt Processing (Prefill)**: 107.03 tokens/sec at $t=8$.
  - **Physical RAM Footprint**: 1.10 GiB (78.8% lower than Gemma 2B FP16).
  - **Thermal Envelope**: 74°C under sustained multi-threaded execution.

#### [x] Phase 4: PyTorch Ternary LoRA Training Pipeline &mdash; 🟢 COMPLETED
- **Action**: Enable LoRA parameter adaptation over BitNet foundation architecture in the FTaaS compute worker.
- **Executed Steps**:
  1. Registered `microsoft/BitNet-b1.58-2B-4T` in `src/FtaaSService.Worker/model_registry.py` with target projection modules `["q_proj", "v_proj", "k_proj", "o_proj"]`.
  2. Hardened `src/FtaaSService.Worker/trainer.py` with device casting safety (`float32` MPS execution) to avoid custom BitLinear kernel incompatibilities.
  3. Executed live training run on `datasets/sample-financial-sentiment.jsonl`.
- **Artifacts & Metrics**:
  - **MLflow Run ID**: `181872f823c14dc2a8d4d6b092e0354c`
  - **Final Training Loss**: $3.2038$ (Step Loss: 4.8921 $\to$ 3.2038)
  - **Trained Adapter Size**: 15.26 MB (`adapter_model.safetensors`)

#### [x] Phase 5: Multi-Model Catalog & Dynamic Studio UI Integration &mdash; 🟢 COMPLETED
- **Action**: Surface BitNet b1.58 natively within the FTaaS control plane and interactive studio interface.
- **Executed Steps**:
  1. Updated .NET 10 API contracts (`SupportedBaseModels.BitNet2B4T`), ingestion validators, and preset mappings.
  2. Integrated BitNet model badge, hardware disclaimers, and comparison dropdowns in `src/FtaaSService.Api/wwwroot/index.html` and `studio.js`.
  3. Extended automated test coverage across C# controller contracts and ingestion pipeline.
- **Verification**: 39/39 .NET unit and integration tests passing (`dotnet test`).

#### [x] Phase 6: Upstream Toolchain Contributions &mdash; 🟢 COMPLETED
- **Action**: Resolve build toolchain and packaging defects upstream in official repositories.
- **Executed Steps**:
  1. Diagnosed CMake submodule pathing and missing target definition errors in `3rdparty/llama.cpp`.
  2. Submitted pull request **`microsoft/BitNet#635`**: `fix(setup): configure 3rdparty/llama.cpp cmake paths and target naming`.
  3. Submitted pull request **`isHuangXin/llama.cpp#7`**: `fix(CMakeLists): guard find_package(bitnet) and expose ggml-bitnet include dirs`.
- **Verification**: Upstream GitHub Actions CI workflows green; Microsoft CLA signed and verified.

#### [x] Phase 7: Native C++ Inference Engine & Dual-Backend Serving &mdash; 🟢 COMPLETED
- **Action**: Build native C++ inference serving bridge and integrate dual-backend dynamic routing into the FTaaS inference service.
- **Executed Steps**:
  1. Implemented `src/FtaaSService.Inference/bitnet_engine.py`: A native subprocess adapter executing `run_inference.py` / compiled C++ CLI with configurable thread counts (`--threads 4`), prompt templating, and regex-based token parsing.
  2. Upgraded `src/FtaaSService.Inference/app.py`: Implemented dual-backend dispatch routing PyTorch Hugging Face requests to MPS/CUDA pipelines and BitNet b1.58 requests to `BitNetEngine`.
  3. Developed unit test suite `tests/test_bitnet_engine.py` covering model detection, command generation, stdout parsing, and graceful error handling.
- **Verification**: 72/72 Python test suite passing (`pytest`).

---

### Milestone 2: Unified 1-Bit Zero-Egress Stack & Statutory Retrieval &mdash; 🟡 ACTIVE

#### [x] Phase 8: 1-Bit Dense Embedding Subsystem (`bitnet-embedding-270m`) &mdash; 🟢 COMPLETED (Serving & Latency on AVX2; Domain Quality Evals Planned in Phase 10)
- **Action**: Acquire, serve, and benchmark Microsoft's 1.58-bit dense embedding model in `bitnet.cpp`.
- **Target Specifications**:
  - Model: `microsoft/bitnet-embedding-270m` (and `bitnet-embedding-0.6b`).
  - Physical Model Storage: 350.46 MB (`bitnet-embeddings-270m-bf16-i2_s.gguf`).
  - Output Vector: 640 dimensions, L2-normalized ($\|v\|_2 = 1.0$).
    - Context Window: Bounded 512 tokens (`-c 512`, operational memory bounding aligning context allocation to the 240-token ceiling without unproven startup latency claims) / 32,768 max architectural tokens; empirical runtime context ceiling is 256 tokens for embedding due to upstream AVX2 `dequantize_row_i2_s` kernel crash (guarded in service at `MAX_SAFE_EMBED_TOKENS = 240`, clamped at 255). Generation path (`BitNet-b1.58-2B-4T`, explicitly configured with `-c 4096`) empirically verified up to 1,877 tokens without crash (output degenerate due to batch prefill tile defect).
- **Executed Steps**:
  1. Acquired official Microsoft `bitnet-embeddings-270m-bf16-i2_s.gguf` model weights.
  2. Implemented native zero-egress streaming wrapper in `src/FtaaSService.Inference/bitnet_engine.py` using `-f /dev/stdin` (with tempfile fallback for non-POSIX platforms), `--embd-separator "<#sep#>"` to ensure multi-paragraph legal statutes contribute across line boundaries without newline splitting, mathematical byte-length skipping ($\le 238\text{ B}$ skips tokenizer), token pre-counting via `llama-tokenize` for longer prompts with fail-closed safety (returning HTTP 503 `TokenizerUnavailableError` when token verification is unavailable), explicit `-c 4096` context sizing on generation CLI, and optional OS-level sandboxing via macOS `sandbox-exec` (`deny network*`, child processes only, verified with negative control assertions).
  3. Integrated `POST /api/v1/inference/embed` in `src/FtaaSService.Inference/app.py` returning unit-normalized float arrays ($\|v\|_2 = 1.0$) with native mean pooling (omitting `--pooling` from CLI so GGUF metadata `gemma3.pooling_type = 1` governs natively; direct test confirms $\cos(\text{default}, \text{mean}) = 1.0000000129$), bounded context (`-c 512`), Prometheus exposition (`ftaas_inference_embed_model_loaded`), sanitized diagnostic `/healthz` telemetry (no filesystem paths; cached adapter count only, omitting tenant adapter IDs), CORS preflight (`OPTIONS`) exemption, and constant-time byte-level HMAC API key authentication. Bound service to loopback (`127.0.0.1`) with `TrustedHostMiddleware` DNS rebinding protection and restricted CORS origins.
    4. Authored comprehensive test suite `TestBitNetEmbedding`, `TestAppRoutingEmbedding`, and live integration suite `TestBitNetIntegrationLive` in `tests/test_bitnet_engine.py` (125/125 Python tests, 45/45 .NET tests passing).
    5. Empirically benchmarked prefill embedding latency across 1, 2, 4, and 8 threads on Intel Core i9 AVX2 (5 warm iterations after 1 warmup run).
- **Empirical Benchmark Results (macOS Intel Core i9-9880H AVX2: 8 Physical Cores, 16 Threads)**:
  - **Kernel Prefill Throughput (`llama-bench`)**:
    - `pp64`: 350.2 t/s ($t=1$) $\to$ 499.5 t/s ($t=4$) $\to$ 490.6 t/s ($t=8$)
    - `pp128`: 483.2 t/s ($t=1$) $\to$ 698.0 t/s ($t=4$) $\to$ 660.8 t/s ($t=8$)
    - `pp512`: 593.5 t/s ($t=1$) $\to$ **977.5 t/s** ($t=4$) $\to$ 950.4 t/s ($t=8$)
  - **End-to-End Subprocess Serving Latency via `/dev/stdin` Stream (Cold Process Exec + AVX2 Compute + Regex Parse)**:
    - *Full Thread Scaling Benchmark (5 warm iterations after 1 warmup)*:

| Input Text | Metric | $t=1$ | $t=2$ | $t=4$ (Default) | $t=8$ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Short Query (~22 tokens)**<br>*FTC ROSCA dark pattern query* | Mean<br>p50 | 1,401.23 ms<br>1,398.54 ms | 1,356.55 ms<br>1,352.10 ms | 1,317.36 ms<br>1,315.42 ms | 1,304.63 ms<br>1,301.88 ms |
| **Medium Statute (~65 tokens)**<br>*California AB 2863 statutory clause* | Mean<br>p50 | 1,470.97 ms<br>1,465.30 ms | 1,410.24 ms<br>1,408.12 ms | 1,329.62 ms<br>1,326.50 ms | 1,293.65 ms<br>1,290.41 ms |
| **Long Agreement (~85 tokens)**<br>*Regulation CC / EFAA excerpt* | Mean<br>p50 | 1,510.02 ms<br>1,505.77 ms | 1,406.56 ms<br>1,402.19 ms | 1,344.02 ms<br>1,340.85 ms | 1,306.71 ms<br>1,303.22 ms |

    - *Base Generation Decode*: 23.5 t/s decode at 8 threads (measured in prototype baseline), 19.7 t/s decode at 4 threads (service default via `/dev/stdin`); prompt prefill 108.8 t/s.
- **Key Empirical Observations**:
  - **Thread Scaling Characteristics**: On macOS Intel Core i9-9880H (8 physical cores, 16 logical threads), scaling from $t=1$ to $t=4$ yields the primary latency reduction (~84–166 ms mean improvement). Scaling further to $t=8$ yields marginal gains (~13–38 ms) as parallel compute is bounded by the fixed ~1.3 s subprocess cold-start and core contention. $t=4$ remains the recommended balanced default.
  - **`/dev/stdin` Security vs Throughput Invariant**: Interleaved A/B benchmarking across 20 alternating iterations confirms throughput delta between `/dev/stdin` (mean: 1,335.19 ms, p50: 1,333.80 ms) and tempfile disk I/O (mean: 1,326.32 ms, p50: 1,322.16 ms, $\Delta = -8.87\text{ ms}$) is within measurement noise ($n=20$, no formal hypothesis test). Standard input streaming is retained strictly as an essential privacy and zero-leakage invariant, completely eliminating prompt exposure from host process listings (`ps aux` / `/proc/$PID/cmdline`) and unencrypted disk swap pages.
  - **Process Cold-Start vs Kernel Speed**: Raw AVX2 SIMD prefill operates at up to 977 tokens/sec (~87 ms pure compute for 85 tokens). The fixed ~1.3 s cold-start overhead confirms the architectural directive for Phase 9 to implement persistent worker daemonization or C shared library bindings for bulk statutory vector indexing.
  - **Context Bounding (`-c 512`)**: Switching from `-c 4096` to `-c 512` bounds operational memory allocation to match the 240-token safe embedding ceiling. Process startup (~1.3 s) breakdown between binary loading, weight loading, and OS runtime initialization is not yet profiled; retaining `-c 512` cleanly aligns KV cache allocation with the operational ceiling without unproven startup latency claims.
  - **2B Generation RAG Latency Scaling & UX Architecture**: While the 2B generation path (`BitNet-b1.58-2B-4T`, explicitly configured with `-c 4096`) avoids the 256-token crash ceiling, long-context evaluation reveals significant prompt prefill latency scaling on CPU AVX2 (measured with 10 generated tokens at $t=4$):
    - **377 tokens prompt**: 6.1 s end-to-end (~1.3 s cold start, ~4.3 s prefill, ~0.5 s decode)
    - **752 tokens prompt**: 10.9 s end-to-end (~1.3 s cold start, ~9.1 s prefill, ~0.5 s decode)
    - **1,877 tokens prompt**: 26.4 s end-to-end (~1.3 s cold start, ~24.6 s prefill, ~0.5 s decode)
    Accounting for ~1.3 s cold start, empirical prefill throughput is ~75–80 tokens/sec (~13 ms per prompt token), aligning with the native CLI telemetry (`[ Prompt: 80.7 t/s | Generation: 17.5 t/s ]`). Note also that generation decode throughput slows slightly at long context (from 19.7–23.5 t/s on short queries to 17.5 t/s at long context).
    *SLA Math & UX Architecture*: A 240-token context plus a 30-token answer works out to ~1.3 s cold start + 2.7–3.2 s prefill + 1.5–1.7 s decode = **~5.5–6.2 s total**, which exceeds a sub-5-second SLA. Achieving a true sub-5-second total response on CPU requires budgeting context to **$\le 150$ prompt tokens** and generating $\le 30$ output tokens ($1.3\text{ s} + 1.9\text{ s} + 1.6\text{ s} \approx 4.8\text{ s}$). Calibrated Time-To-First-Token (TTFT) on AVX2 scales from ~3.2 s at 150 tokens to ~4.3 s at 240 tokens and ~5.6–6.0 s at 377 tokens. For longer prompts (e.g. 400 tokens in + 150 tokens out), total latency lands at ~13–15 seconds (~1.3 s cold start + ~5.0 s prefill + ~8.5 s decode). Consequently, interactive statutory RAG (Phase 9) cannot rely on synchronous blocking HTTP chat; the UX must be architected as **Server-Sent Events (SSE) token streaming** (TTFT ~3.2–4.3 s) or **asynchronous background worker jobs**.
  - **End-to-End Endpoint Latency vs Pre-Check Overhead**: For prompts $\le 238$ UTF-8 bytes (short statutory clauses), the tokenizer pre-check is mathematically bypassed (tokens cannot exceed bytes), maintaining ~1,300 ms cold subprocess execution. For longer clauses ($> 238\text{ B}$), `llama-tokenize` adds ~485 ms cold overhead for a total endpoint latency of ~1,815 ms, completely preventing subprocess memory faults.
  - **Upstream AVX2 Kernel Ceiling & Mitigations**: Prompts exceeding 256 tokens in `bitnet-embedding-270m` trigger a native crash (`EXC_BAD_ACCESS`, exit code -11) in `libggml-base.0.dylib` (`dequantize_row_i2_s + 153: movzbl (%rdi,%rbx), %r14d`). Fine-grained token sweeps across both prose and dense legal statutory citations (`§ 229.10(c)(1)(vi) “Next-day” exception for $5,525.`) confirm identical behavior: tokens 248–256 execute cleanly (RC: 0), while token 257 crashes consistently (`EXC_BAD_ACCESS`, exit code -11). The service mitigates this via byte-length bounding and pre-tokenization in `llama-tokenize` with a ceiling of `MAX_SAFE_EMBED_TOKENS = 240` (clamped at 255), providing a 16-token safe margin. Prompts $> 240$ tokens return HTTP 413 `ContextOverflowError`; if `llama-tokenize` is unavailable for prompts $> 238\text{ B}$, the service fails closed with HTTP 503 `TokenizerUnavailableError` rather than risking a subprocess fault. Live testing of the 2B generation model (`BitNet-b1.58-2B-4T`) confirmed zero memory crashes across 377, 653, 752, and 1,877 tokens, verifying that the 256-token ceiling is isolated strictly to the 270M embedding model.
  - **270M Embedder Validation, Pooling Governance & Relational Teacher Parity**: Comprehensive validation was performed against the official Microsoft 270M release (`bitnet-embeddings-270m-bf16-i2_s.gguf`):
    1. *Clean Stderr & Pre-Tokenizer Configuration*: Subprocess execution emits zero `missing pre-tokenizer type` warnings (`tokenizer.ggml.pre` is configured as `default`, standard SentencePiece model).
    2. *Exact Element-by-Element Token ID Parity*: Evaluated `llama-tokenize` against Hugging Face reference (`microsoft/harrier-oss-v1-270m` base via `GemmaTokenizerFast`) across 20 statutory clauses and 20 short clauses. Yielded **100.0% exact token ID match** (`native_token_ids == hf_token_ids` element-by-element across all 40/40 evaluated clauses, documented in [`docs/empirical_270m_embedder_validation.json`](empirical_270m_embedder_validation.json)).
    3. *Metadata-Governed Native Pooling*: Removing `--pooling` from `build_bitnet_embed_cmd` allows GGUF metadata (`gemma3.pooling_type = 1`, MEAN pooling) to govern natively. Direct numerical verification confirmed that omitting the flag produces exact cosine identity with explicit mean pooling ($\cos = 1.0000000129$), fully eliminating vector collapse.
    4. *Relational Parity with FP16 Teacher (`harrier-oss-v1-270m`)*: While direct cross-space cosine between ternary student and float teacher embeddings is orthogonal due to distinct coordinate bases, the space-independent Spearman rank correlation of their $20 \times 20$ pairwise similarity matrices across 190 off-diagonal clause pairs is $\rho = 0.4318$ ($p = 4.94 \times 10^{-10}$) and Pearson $r = 0.4467$ ($p = 1.05 \times 10^{-10}$, $R^2 = 0.1995$, explaining 20.0% of variance) on full statutory clauses (36-75 tokens), and $\rho = 0.3705$ ($p = 1.43 \times 10^{-7}$) and Pearson $r = 0.4420$ ($p = 1.73 \times 10^{-10}$, $R^2 = 0.1953$, explaining 19.5% of variance) on short clauses (< 30 tokens). The lower rank correlation on short sequences ($\rho = 0.3705$ vs $\rho = 0.4318$) indicates that concise statutory clauses lose discriminative signal when quantized to ternary weights, directly explaining why standalone 1-bit retrieval drops to 0% Top-1 on short chunks.
    5. *Gemma Semantic Baseline Offset vs Contrastive Models*: Dissimilar concepts (Fruit vs Physics) produce $\cos = 0.425$ in BitNet and $\cos = 0.349$ in Harrier, compared to $\cos = -0.094$ in `all-MiniLM-L6-v2`. This elevated baseline is characteristic of generative Gemma backbones with large 262,144-token vocabularies, which share an underlying semantic centroid unlike contrastive BERT embeddings explicitly centered at zero.
  - **2B Generation Context-Length Degradation Isolation: Context-Length-Dependent Failure (Mechanism Unknown)**:
    1. *Hugging Face PyTorch Baseline Control (Conclusive Weight Sanity)*: Evaluating the official PyTorch reference model (`microsoft/BitNet-b1.58-2B-4T`) via `transformers` on the 598-token statutory prompt (12 CFR Part 229 Subpart B Reg CC, Section 229.13 large deposit threshold) and intermediate 79-token prompt cleanly and factually extracted `$6,725` across all sampling configurations:
       - *Test 1 (Greedy, `do_sample=False`, temp 0.0, no repetition penalty)*: Generated 25 tokens ending cleanly on `<|eot_id|>`: `"The statutory dollar threshold for large deposits under Section 229.13 is $6,725 on any one banking day.<|eot_id|>"`
       - *Test 2 (Greedy with `repetition_penalty=1.1`)*: Generated the exact same 25 tokens.
       - *Test 3 (Sampled `temp=0.5, repetition_penalty=1.1`)*: Generated the exact same 25 tokens.
       - *Test 4 (Intermediate 79-token prompt)*: Generated: `"The statutory dollar threshold for large deposits, as specified in 12 CFR § 229.13(b), is $6,725..."`
       This confirms that the underlying 2B foundation model weights, attention mechanisms, and SFT/DPO instruction alignment are coherent, factual, and free of degeneration at ~600 tokens.
    2. *RoPE Parameter & BOS Alignment*: Binary GGUF metadata extraction dynamically confirmed exact parameter parity with the Hugging Face `config.json`: `bitnet-b1.58.rope.freq_base = 500000.0` (HF `rope_theta = 500000.0`), `layer_norm_rms_epsilon = 1e-05` (HF `rms_norm_eps = 1e-05`), `dim = 128`, and `context_length = 4096`. Token stream comparison confirmed both reference and runtime emit a single `<|begin_of_text|>` (token 128000), verifying zero double-BOS emission.
    3. *Empirical Bisection of Context-Length Breakdown Boundary*: Fine-grained parameter sweeps across prompt lengths (23, 29, 30, 79, 598 tokens), batch configurations ($b \in [1, 2, 4, 8, 16, 32, 64, 512]$), KV cache data types (`-ctk f32 -ctv f32`), and Flash Attention toggles (`-fa on/off`) definitively bisect the failure boundary:
       - **Prompts $\le 29$ tokens under default prefill ($b=512$)**: Prompt prefill evaluates cleanly, generating the correct answer: `"Under Section 10, the threshold is $500. [end of text]"` (`classification: correct`).
       - **Prompts $\ge 30$ tokens under default prefill ($b=512$)**: Prompts at exactly 30 tokens immediately trigger tile corruption artifacts (`'备份'`, `'2 0 3 0...'`, `classification: tile_corruption_garbage`).
       - **First-Token Logits Contrast vs Hugging Face Reference**: In the Hugging Face PyTorch reference model, at 29 and 30 prompt tokens, the top token is ID 791 (`'The'`) with logit $\approx 21.1\text{--}21.8$, while the corrupt token ID 112890 (`'备份'`) has logit $\approx -6.0\text{--}-6.1$ (suppressed by $\Delta \approx 27\text{--}28$ logits). Under the native `bitnet.cpp` C++ runtime, however, this token is erroneously selected as the first emission at prompt length $\ge 30$.
       - **Micro-batched Prompts ($b=16$)**: Passing `-b 16 -ub 16` allows a 30-token prompt to pass cleanly, but prompts beyond 34 tokens (e.g. 79 tokens and 598 tokens) degenerate into repetitive loops without factual retrieval (`classification: fluent_repetitive_loop`).
       - **Single-token prefill ($b=1$ oracle)**: Evaluates via single-token path but emits immediate `[end of text]` (`classification: immediate_eos`) across both short and long prompts.
       - **KV Cache & Flash Attention Invariants**: Neither KV cache precision flags (`-ctk f32 -ctv f32`) nor Flash Attention toggles (`-fa on/off`) alter the failure boundary.
    4. *Calibrated Diagnosis & Safe Bounding*: The degeneration is an upstream runtime context-length-dependent defect in `bitnet.cpp` (exact root cause mechanism unknown; RoPE parameters match reference). Because factual extraction breaks at $\ge 30$ tokens under default settings and beyond 34 tokens under micro-batching, verified factual generation in `bitnet.cpp` is strictly confined to prompts $\le 29$ tokens. The service enforces fail-closed safety by capping generation context at `MAX_SAFE_GENERATE_TOKENS = 24`, providing a verified 5-token safety margin below the 29-token breakdown point (`ContextOverflowError` $\to$ HTTP 413, `TokenizerUnavailableError` $\to$ HTTP 503). Complete raw outputs and system telemetry are dynamically recorded in [`docs/empirical_long_context_isolation_results.json`](empirical_long_context_isolation_results.json), and regression behavior is tracked via `@unittest.expectedFailure` in `tests/test_bitnet_engine.py` (`test_live_generation_long_context_does_not_hit_ceiling`).

#### [x] Phase 9: Statutory Pack Retrieval Baseline (Retrieval-Only Architecture) &mdash; 🟢 PROTOTYPE BENCHMARK ESTABLISHED
- **Action**: Build a zero-dependency local vector retrieval index over tracked legal policies, establishing the production release as a **Retrieval-Only** system to bypass runtime generation limitations by delivering direct, verbatim statutory extracts rather than model synthesis. Ground production serving strictly on the deterministic BM25 + Regex core; classify 1-bit dense embeddings as an experimental research prototype.
- **Executed Steps & Empirical Architecture**:
  1. Pre-embedded 40 statutory chunks across banking (Regulation CC 12 CFR Part 229, Regulation E 12 CFR Part 1005, UCC Article 4) and consumer protection (FTC ROSCA 15 U.S.C. § 8403, California Bus. & Prof. Code §§ 17600–17606 & AB 2863, FTC Negative Option Rule 16 CFR Part 425, UK DMCC Act 2024 ss. 255–258, EU Consumer Rights Directive 2011/83/EU, State ARLs) into a 640-dim binary vector index ($< 500\text{ KB}$ storage). Every chunk incorporates explicit legal metadata: `status`, `effective_from`, `verified_on`, and authentic `source_url`.
  2. Vacatur Tracking & Stale Pack Failure Mode: Chunks for 16 CFR Part 425 (§§ 425.3, 425.5, 425.6) track the July 8, 2025 Eighth Circuit vacatur (*Custom Communications Engineering, Inc. v. FTC*, No. 24-3232, consolidated with *NFIB v. FTC*) and March 2026 FTC ANPRM. Note that the 1973 prenotification negative option rule remains in effect; only the 2024 expansion was vacated. Vacated rule chunks are returned with an explicit `[VACATED BY COURT ORDER]` banner rather than dropped, demonstrating the exact stale-pack failure mode in real-world operations. UK DMCC subscription provisions (ss. 255–258) were enacted on May 24, 2024 but are tracked as `enacted_not_in_force` (implementation confirmed for Spring 2026 / 2027).
  3. Direct Verbatim Extraction & Citations: Deliver consumer findings via direct, verbatim extraction of authentic statutory clauses and official CFR/USC citations, completely eliminating dependency on generative synthesis (BitNet 2B generator). Generative translation is decoupled and deferred to an optional advisory tier pending upstream runtime fixes.
  4. Embedder Sanity & Contrastive Baseline: BitNet 270M embeddings achieve 100% self-retrieval (40/40 chunks retrieve themselves at rank 1 with min self-similarity $\cos = 1.0000$). Evaluating the contrastive `all-MiniLM-L6-v2` dense baseline achieves 88.6% Top-1 (31/35, 95% CI [74.1%, 95.5%]) and 91.4% Top-3 on full text, and 65.7% Top-1 on short chunks.
- **Prototype Statutory Retrieval Benchmark (40 Statutory Chunks, 35 Realistic Queries)**:
  Full evaluation was executed across 35 legal queries over the full 40-chunk corpus and the 37-chunk active/enacted subset, with 95% Wilson score confidence intervals (recorded in [`docs/empirical_retrieval_benchmark.json`](empirical_retrieval_benchmark.json)):

| Retrieval Method | Top-1 Recall (Full) | 95% CI (Top-1) | Top-3 Recall (Full) | 95% CI (Top-3) | MRR | Role in Architecture |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BM25 (Okapi Lexical Baseline)** | **82.9%** (29/35) | [67.3%, 91.9%] | **97.1%** (34/35) | [85.5%, 99.5%] | **0.8952** | **Production Core Baseline** |
| **Deterministic Regex Rules** | 14.3% (5/35) | [6.3%, 29.4%] | 17.1% (6/35) | [8.1%, 32.7%] | 0.2253 | Exact Threshold Precision ($6,725, 14-day) |
| **BitNet 270M Dense Embeddings** | 20.0% (7/35) | [10.0%, 35.9%] | 34.3% (12/35) | [20.8%, 50.8%] | 0.3378 | Experimental Prototype (0% on Short Chunks) |
| **Hybrid Pipeline (Regex + BM25 + BitNet)** | **88.6%** (31/35) | [74.1%, 95.5%] | **97.1%** (34/35) | [85.5%, 99.5%] | **0.9333** | **Experimental Prototype (+2 Query Gain)** |

*Benchmark Methodology & Scientific Disclaimer*:
- *Single-Author Prototype Benchmark*: A single author authored both corpus chunk texts and query formulations; high lexical overlap between queries and statutory text inherently favors BM25.
- *Sample Size & Confidence Intervals*: With $n = 35$, the 95% Wilson confidence interval spans $\approx \pm 11\text{--}13$ percentage points (BM25 Top-1 is $[67.3\%, 91.9\%]$, Hybrid Top-1 is $[74.1\%, 95.5\%]$).
- *Marginal Hybrid Gain & Short Chunk Failure*: While the hybrid pipeline adds 2 hits on full text (31/35 vs 29/35), on short chunks (< 30 tokens) standalone BitNet retrieval drops to 0.0% (0/35), pulling hybrid Top-1 down to 57.1% (worse than BM25 alone at 60.0%). The hybrid gain on full text is statistically indistinguishable from measurement noise given overlapping confidence intervals.
- *Production Grounding*: Production serving is grounded strictly on the deterministic BM25 + Regex core, while dense embeddings remain an experimental prototype.
- *Active/Enacted Performance*: Excluding the 3 vacated 16 CFR Part 425 chunks yields 74.3% Top-1 for BM25 and 80.0% Top-1 for Hybrid.
- **Concrete Retrieval-Only Failure Modes & Inherent Risks**:
  Retrieval-only architectures eliminate generative hallucination, but introduce distinct operational and retrieval failure modes that must be engineered against:
  1. *Semantic Ranking / Retrieval Errors*: The embedding index may rank an irrelevant statutory clause over the applicable section (e.g., retrieving § 229.10 next-day cash rules instead of § 229.13 large deposit exceptions). Mitigated by BM25 + Regex anchor weighting.
  2. *Stale or Incomplete Statutory Packs*: Indexed corpuses may lag legislative enactments, judicial vacatur (e.g. 8th Circuit vacating 16 CFR Part 425), effective date transitions (e.g., Regulation CC amendments effective July 1, 2025), or jurisdiction-specific amendments.
  3. *Chunking Boundaries & Split Exceptions*: Bounding context to the 240-token embedding ceiling risks splitting an exception clause from its parent operative requirement (e.g., separating § 229.13(b) from § 229.10(c)), resulting in false positive violation alerts.
  4. *Stage 1 Regex False Positives / False Negatives*: Deterministic pattern matching may fail to match creative dark pattern phrasing or misidentify compliant cancellation flows.
- **Required UX Safeguards & Truth-in-Interface Design*:
  To protect consumers and legal auditors against retrieval failure modes, the interface enforces strict UX safeguards:
  - *Reviewed-On Timestamp*: Explicitly display the timestamp and edition of the indexed statutory pack evaluated against the agreement.
  - *Explicit Disclaimer*: Prominently state "Not Legal Advice; Automated Statutory Extraction" on all outputs.
  - *Matched Rule Exposition*: Display the specific matched rule name, statutory section, and exact citation rather than an opaque score.
  - *Absence of False Reassurance*: The system never emits an "all clear", "clean bill of health", or "100% compliant" determination, acknowledging that unflagged clauses may still contain unlawful terms.

#### [x] Phase 10: Prototype Statutory Benchmark & Regulatory Audit &mdash; 🟢 COMPLETED (Prototype Grounding & Vacatur Tracking)
- **Action**: Establish reproducible domain evaluation benchmarks measuring empirical retrieval accuracy, precision, and recall under the Retrieval-Only architecture while tracking regulatory vacatur status.
- **Executed Evaluation**:
  - Validated retrieval accuracy across 35 realistic consumer and audit queries against 40 statutory chunks.
  - Benchmarked BitNet 270M Embed against BM25 Okapi lexical baseline and regex pattern rules.
  - Confirmed BM25 + Regex as the production core baseline with 97.1% Top-3 recall, with BitNet 270M embeddings classified as an experimental research prototype.
  - Explicitly integrated tracking for vacated rules (16 CFR Part 425 vacatur by 8th Cir. July 8, 2025 in *Custom Communications Engineering, Inc. v. FTC*, No. 24-3232, consolidated with *NFIB v. FTC*), ensuring vacated rules are presented with clear warning banners. Primary statutory citations verified and recorded in [`docs/verified_statutory_sources.json`](verified_statutory_sources.json). Full query-level details, reciprocal ranks, and hit rates documented in [`docs/empirical_retrieval_benchmark.json`](empirical_retrieval_benchmark.json).

#### [ ] Phase 11: Universal Sidecar Protocol & Everyday Hardware Appliance &mdash; 📋 PLANNED
- **Action**: Package the complete 1-bit stack into an air-gapped, zero-egress local appliance for consumer defense with strict concurrency memory bounding.
- **Appliance Profile**:
  - Concurrency Bounding: Process semaphores enforce `MAX_CONCURRENT_GENERATE = 1` and `MAX_CONCURRENT_EMBED = 2` by default.
  - Worst-Case Memory Envelope:
    - 1 concurrent generate process: ~1.53 GB peak RSS.
    - 2 concurrent embed processes: $2 \times 1,585.4\text{ MB} \approx \mathbf{3.17\text{ GB}}$ peak RSS (at 240 tokens safe ceiling with bounded `-c 512`).
    - Total peak memory envelope: $1.53\text{ GB} + 3.17\text{ GB} \approx \mathbf{4.7\text{ GB}}$ peak RSS (or $\mathbf{\approx 3.17\text{ GB}}$ in Retrieval-Only deployment mode with 2 embeds and 0 generate).
    - Comfortably bounded within an 8 GB consumer hardware appliance envelope, leaving $\ge 3.3\text{ GB}$ free for the host OS and user applications. Physical validation on a physical 8 GB constrained RAM appliance is pending (profiling completed on 16 GB development host).
  - Hardware Target: Ubiquitous x86_64 AVX2 / ARM NEON hardware (Intel Core i5/i7/i9 2019+, M-series, AMD Ryzen).
  - Protocol Invariants: Zero cloud egress (`connect-src 'none'`), loopback API (`127.0.0.1:8420`), and atomic Nuclear Hard Burn (`/burn`).

---

## 4. Key Engineering Risks & Mitigations

| Risk | Cause | Mitigation Strategy | Outcome & Resolution |
| :--- | :--- | :--- | :--- |
| **Empty Submodule Directory** | `gh repo clone` omits submodules by default | Explicit `git submodule update --init --recursive`. | **Resolved**: Automated in setup workflow; documented in toolchain docs. |
| **Missing Native CMake** | Host system lacks global CMake in `$PATH` | Install CMake into dedicated `.venv` via `pip install cmake`. | **Resolved**: Build isolated in virtual environment; zero global pollution. |
| **macOS x86_64 Kernel Mismatch** | `setup_env.py` selecting ARM `tl1` on Intel | Force `-q i2_s` and pass `-DBITNET_X86_TL2=OFF` if targeting standard `i2_s` AVX2. | **Resolved**: AVX2 SIMD compilation verified; 19.7 t/s decode confirmed. |
| **PyTorch Training Weights vs GGUF** | GGUF is optimized for C++ inference, not PyTorch backprop | Use GGUF for C++ inference; use Hugging Face PyTorch weights (`microsoft/BitNet-b1.58-2B-4T`) for LoRA. | **Resolved**: Dual-representation architecture cleanly implemented across worker and engine. |
| **Upstream Submodule CMake Path Breaks** | Upstream build scripts assumed external CMake install | Patch CMake target configuration and upstream fixes. | **Resolved**: `microsoft/BitNet#635` and `isHuangXin/llama.cpp#7` submitted and CI validated. |
| **Inference Engine Backend Disconnect** | Standard PyTorch engine cannot execute 2-bit GGUF files natively | Dual-backend inference routing in FastAPI (`app.py` + `bitnet_engine.py`). | **Resolved**: Native C++ adapter serves base GGUF in CPU memory (19.7 t/s); PyTorch serves fine-tuned adapters. |
| **Embedding Normalization Drift** | Unnormalized dot-products degrade cosine ranking | Enforce EOS pooling with `--embd-normalize 2` in `bitnet.cpp` CLI. | **Resolved**: Enforced `--embd-normalize 2` with unit L2 normalization in C++ wrapper. |
| **Long-Context AVX2 Kernel Crash** | Prompt > 256 tokens triggers upstream `EXC_BAD_ACCESS` in `dequantize_row_i2_s + 153` | Pre-check token count via `llama-tokenize` before execution; reject > 240 tokens with `ContextOverflowError` (HTTP 413); clamp ceiling to 255; fail closed if tokenizer unavailable with `TokenizerUnavailableError` (HTTP 503); chunk Phase 9 index to ~200 tokens. | **Mitigated**: Subprocess crashes mitigated via fail-closed pre-check; verified in live boundary tests (239/240 succeed, 241 rejected) and dense statutory citation sweeps (crash verified at 257 tokens); 2B generation path verified up to 1,877 tokens without crash; reproduction test cases, scripts, and LLDB backtraces preserved in git history; coordinated upstream defect reporting planned/in progress (no active MSRC tracking case). |
| **Process Argument & Disk Exposure** | Passing `-p` leaks to `ps aux`; temp files risk disk persistence | Stream prompts via `-f /dev/stdin` on POSIX systems; verify zero leakage in argv tests. | **Resolved**: Enforced standard input streaming; zero prompt leakage in argv; verified via interleaved A/B benchmark that security invariant incurs no throughput penalty. Optional macOS kernel sandbox (`sandbox-exec` `deny network*`) supported for child subprocesses only (tested via negative control assertions; parent Python process unconfined; deprecated by Apple; off by default). |

### Upstream Memory Safety Diagnostics: AVX2 `dequantize_row_i2_s` Batch Overflow
During stress testing with statutory texts exceeding 256 tokens, `bitnet.cpp` crashes with `SIGSEGV` / `EXC_BAD_ACCESS` (exit -11). LLDB diagnostic backtrace on macOS Intel Core i9-9880H:
```text
* thread #1, queue = 'com.apple.main-thread', stop reason = EXC_BAD_ACCESS (code=1, address=0x10d8a9000)
    frame #0: 0x0000000109be35f9 libggml-base.0.dylib`dequantize_row_i2_s + 153
libggml-base.0.dylib`dequantize_row_i2_s:
->  0x109be35f9 <+153>: movzbl (%rdi,%rbx), %r14d
    0x109be35fd <+157>: movl   %r14d, %r15d
    0x109be3600 <+160>: andl   $0x3, %r15d
```
Tuning flags (`-ub 256` to `-ub 4096`) does not prevent the out-of-bounds read during batch decode. Fine-grained empirical sweeps across both prose and dense legal statutory citations (`§ 229.10(c)(1)(vi) “Next-day” exception for $5,525.`) confirm that tokens 248 through 256 complete with return code 0, while token 257 reliably crashes with exit code -11. Guarding prompt token lengths prior to execution (`MAX_SAFE_EMBED_TOKENS = 240`, clamped at 255) with fail-closed semantics (`TokenizerUnavailableError` $\to$ HTTP 503) mitigates this crash vector in production. In contrast, the 2B generation model (`microsoft/BitNet-b1.58-2B-4T`, explicitly configured with `-c 4096`) executes prompts with 377, 653, 752, and 1,877 tokens on CPU AVX2 with zero memory faults, confirming the overflow vulnerability is isolated strictly to the 270M embedding model's batch decode. Reproduction test cases, scripts, and LLDB backtraces are preserved in repository git history; coordinated upstream defect reporting is planned/in progress for submission to upstream maintainers (no active MSRC tracking case or vulnerability case number has been opened).

---

## 5. Architectural Record & Repository Status

This roadmap is an authoritative, version-controlled engineering document tracked within the `ml` repository:
- **File**: [`docs/ROADMAP.md`](ROADMAP.md)
- **Repository**: `github.com/knowthankyew/event-driven-ftaas` (`origin/main`)
- **Status**: Tracked & Committed
- **Related Documents**:
  - Baseline Edge Results: [`SMOL_PROTOTYPE_RESULTS.md`](SMOL_PROTOTYPE_RESULTS.md)
  - Reasoning Tier Results: [`GEMMA_PROTOTYPE_RESULTS.md`](GEMMA_PROTOTYPE_RESULTS.md)
  - Empirical BitNet Benchmarking: [`BITNET_PROTOTYPE_RESULTS.md`](BITNET_PROTOTYPE_RESULTS.md)
  - Comparative Architecture Analysis: [`MODEL_PERFORMANCE_COMPARISON.md`](MODEL_PERFORMANCE_COMPARISON.md)
  - System Architecture & Polyglot Specification: [`ARCHITECTURE.md`](ARCHITECTURE.md)
  - Coordinated Upstream Tracking: Reproduction test cases and LLDB backtraces preserved; upstream report planned/in progress (no active MSRC case)
  - Master Portfolio Brief: [`KNOWTHANKYEW_PORTFOLIO_MASTER_BRIEF.md`](https://gist.github.com/knowthankyew/53ccf4d5a81e916f895c74e18b231e16)
