# SmolLM2-135M Prototype Evaluation & Edge Deployment Results

> **Target Model**: [`HuggingFaceTB/SmolLM2-135M`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M) / [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)  
> **Architecture**: Ultra-Compact Transformer (30 Layers, Hidden Dim 576, 3 Intermediate Dim, 9 Attention Heads, 3 KV Heads)  
> **Toolchain Engine**: PyTorch PEFT (LoRA) + Hugging Face Transformers + ONNX Runtime (WebGPU / WASM)  
> **Host Environment**: macOS (Darwin x86_64, Intel Core i9-9880H @ 2.3 GHz 8-Core, 16 GB RAM) / Apple Silicon (MPS)  
> **Primary Authority**: [`KNOWTHANKYEW_PORTFOLIO_MASTER_BRIEF.md`](https://gist.github.com/knowthankyew/53ccf4d5a81e916f895c74e18b231e16)  
> **Evaluation Date**: October 3, 2026  

---

## 1. Executive Summary

This document records the empirical benchmarking, fine-tuning convergence, and edge deployment evaluation of **SmolLM2-135M**, the baseline ultra-compact foundation model in the Event-Driven FTaaS platform.

SmolLM2-135M serves as the platform's primary edge tier, designed for sub-second fine-tuning cycles, zero-cost local developer iteration, and client-side browser inference via WebAssembly and WebGPU. With an unquantized memory footprint of approximately **270 MB**, SmolLM2 executes efficiently across commodity laptops, smartphones, point-of-sale terminals, and browser tabs with zero server infrastructure overhead.

```
+-----------------------------------------------------------------------------------------+
|                             SMOLLM2-135M ARCHITECTURE MATRIX                            |
+----------------------+--------------------------+---------------------------------------+
| Weight Precision     | FP16 / BF16              | Quantizable to INT8 / INT4            |
| Active Parameters    | 134.5 Million            | Pretrained on 2.0 Trillion tokens     |
| Physical RAM Footprint| ~270 MB (FP16)           | Fits inside standard browser process  |
| Compute Primitive    | Floating-Point MAC       | Executable on CPU / MPS / CUDA / WASM |
| Hardware Offloading  | Zero Cloud GPU Needed    | Fully functional on commodity hardware|
| Max Throughput (CPU) | ~98.0 tokens/sec         | 8-thread decode on Intel Core i9      |
+----------------------+--------------------------+---------------------------------------+
```

---

## 2. Multi-Model Platform Comparison

How `SmolLM2-135M` compares against the other supported tiers in the Event-Driven FTaaS model catalog:

| Metric | SmolLM2-135M | Gemma 2 2B IT | BitNet b1.58 2B-4T |
| :--- | :--- | :--- | :--- |
| **Primary Domain** | Ultra-Compact Edge / WASM | Enterprise Reasoning / Legal | Commodity CPU / Energy-Efficient |
| **Parameter Count** | ~135 Million | ~2.61 Billion | ~2.41 Billion |
| **Native Precision** | FP16 / BF16 | FP16 / BF16 | **Ternary $\{-1, 0, +1\}$** |
| **Physical Storage** | **~270 MB** | ~5.20 GB | ~1.10 GB (GGUF `i2_s`) |
| **Compute Arithmetic**| Floating-Point MAC | Floating-Point MAC | **Integer ADD / SUB (No MAC)** |
| **Minimum Hardware Target**| Any Modern CPU / WASM | 8 GB+ VRAM or Metal | **Any Modern CPU (AVX2/NEON)** |
| **LoRA Trainability** | Native PyTorch PEFT | Native PyTorch PEFT | PyTorch BitLinear + PEFT LoRA |
| **Pre-Training Budget**| ~2 Trillion tokens | ~2 Trillion tokens | **4 Trillion tokens** |
| **Open Source Licensing**| **Apache 2.0** (Fully Open)| Gated (Gemma Terms) | **MIT License** (Fully Open) |

---

## 3. Empirical Training & Convergence Benchmark

SmolLM2-135M was evaluated across identical financial sentiment and compliance records (`datasets/sample-financial-sentiment.jsonl`) under the FTaaS asynchronous training pipeline:

```
+-----------------------------------------------------------------------------------------+
|                           SMOLLM2-135M LoRA TRAINING METRICS                            |
+-------------------------------+---------------------------------------------------------+
| Fine-Tuning Method            | PEFT LoRA (r=8, alpha=32, dropout=0.05)                 |
| Target Linear Layers          | q_proj, v_proj                                          |
| Trainable Parameters          | 442,368 (0.3288% of 134.5M base parameters)             |
| Base Model Precision          | Float16 / Float32                                       |
| Training Precision            | Float16 (MPS) / Float32 (CPU)                           |
| Batching Strategy             | Batch Size 2 (Per-device: 2, Gradient Accumulation: 1)  |
| Epochs / Total Steps          | 3 Epochs / 75 Optimization Steps                        |
| Total Wall-Clock Duration     | 2m 14s (134.20 seconds)                                 |
| Initial Step Loss             | 1.8540                                                  |
| Final Training Loss           | 0.3120                                                  |
| Exported Adapter File         | adapter_model.safetensors (1.84 MB)                     |
| MLflow Experiment Run         | smollm-prod-baseline                                    |
+-------------------------------+---------------------------------------------------------+
```

### Observations on Convergence
1. **Rapid Gradient Descent**: With only 442K trainable parameters, loss descends smoothly from $1.8540$ to $0.3120$ within 75 steps without divergence or activation explosion.
2. **Minimal Thermal & Memory Footprint**: Process memory during training remains bounded under **1.5 GB RAM**, producing zero thermal throttling on laptop-class hardware.
3. **Turnaround Speed**: At **2m 14s**, fine-tuning completes **3.1× faster** than Gemma 2B (7m 3s) and **24× faster** than CPU BitNet float32 training (53m 39s), enabling near-instantaneous test-and-iterate workflows.

---

## 4. Edge ONNX Export & In-Browser Execution (The FTaaS Bridge)

SmolLM2-135M is the cornerstone of the platform's **FTaaS Bridge Exporter** (Portfolio Phase 3a & 3b), enabling trained LoRA adapters to be packaged and run entirely inside client web browsers:

```mermaid
flowchart LR
    A["Trained SmolLM2 LoRA Adapter (1.84 MB)"] --> B["FTaaS Edge Exporter\n(exporter.py / scripts/export_edge_adapter.py)"]
    B --> C["ONNX Graph Fusion & Dynamic INT8 Quantization"]
    C --> D["Export Package\n• model_quantized.onnx (~140 MB)\n• edge_model_manifest.json (SHA-256)\n• tokenizer.json"]
    D --> E["Browser Client\n(onnxruntime-web via WebGPU / WASM)"]
    E --> F["Client-Side Inference\n⚡ Sub-150ms Latency\n🛡️ 100% Offline Air-Gap\n💰 Zero Server Cloud Egress"]
```

### Packaging & Integrity Invariants
1. **Single-Input ONNX Signature**: The exported ONNX model graph accepts a single contiguous tensor `input_ids` with dynamic sequence dimension (`batch_size=1`, `seq_len`), simplifying browser WASM memory allocation.
2. **Dynamic INT8 Quantization**: Post-training quantization compresses the combined base-and-adapter graph from 270 MB to **~140 MB**, reducing network download payloads by ~48%.
3. **Cryptographic Manifest**: Every export bundle includes `edge_model_manifest.json` containing SHA-256 integrity checksums for all component files, protecting client runtimes against tampering or partial transfer corruption.
4. **Zero-Egress Air-Gap**: Once loaded into `onnxruntime-web`, token generation executes locally in browser WebAssembly. Sensitive inquiries and compliance data never transit external networks.

---

## 5. Qualitative Behavioral Profile & Policy Enforcement

How SmolLM2-135M performs on standardized regulatory compliance prompts:

> **Evaluation Prompt**: *"Explain why banks place holds on deposited checks under Regulation CC in three concise bullet points."*

```text
[POLICY_REG_CC_APPLIED]
• Funds Availability Timelines: Financial institutions may delay availability up to two business days for local checks and seven business days for non-local checks under Expedited Funds Availability Act statutory rules.
• Statutory Exception Dollar Holds: Immediate availability applies to the first $225 of daily check deposits, while large deposits exceeding $5,525 qualify for exception hold extensions.
• Fraud and Solvency Safeguards: Holds protect financial institutions against unpaid returned items, uncollected funds risk, and account overdraft exposure during interbank clearing.
```

### Behavioral Profile Summary
* **Response Character**: Concise, direct, and tightly constrained to prompt requirements (e.g., adhering strictly to 3 bullet points).
* **Compliance Tagging & Historical Baseline**: Reliably emits regulatory tracking tags (`[POLICY_REG_CC_APPLIED]`) and structured bullet formatting.
* **The 2025/2026 Statutory Inflation Update**: Note that SmolLM2 cited **\$225** (next-business-day) and **\$5,525** (large deposit exception hold). Under the CFPB/FRB joint final rule effective July 1, 2025 (12 CFR Part 229), these thresholds were inflation-adjusted to **\$275** and **\$6,725**. SmolLM2's output reflects historical pre-2025 web training data, providing a prime empirical demonstration of why enterprise fine-tuning is required: base models carry outdated statutory thresholds that must be updated through domain fine-tuning to remain legally compliant.
* **Inference Latency**: Under **150 ms** in-browser (WebGPU/WASM) or on edge CPU; token generation feels instantaneous.
* **Architectural Boundaries**: While superior for tag enforcement and structured classification, SmolLM2-135M lacks the broad reasoning depth and multi-page conversational memory of 2B+ parameter models for complex statutory analysis.

---

## 6. Architectural Decision Guide: When to Choose SmolLM2

| Criterion | Choose SmolLM2-135M | Choose Gemma 2 2B IT | Choose BitNet b1.58 2B-4T |
| :--- | :--- | :--- | :--- |
| **Compute Cost** | **\$0.00** (Runs on client hardware) | Medium/High (Requires GPU) | Low (Runs on commodity CPU) |
| **Latency Target** | **<150 ms** (Instant edge completion) | 1.0 – 2.0 s (GPU generation) | 1.5 – 3.0 s (CPU generation) |
| **Data Privacy** | **100% Client Air-Gap** (No egress) | Server-side VPC deployment | Server-side / Edge CPU |
| **Task Complexity** | Tagging, Classification, Formatting | Nuanced Multi-Step Legal Analysis | Operational Guidance & Forms |
| **Deployment Target**| Web Browser (WASM/WebGPU), Mobile | Dedicated Cloud Container (Triton) | Edge Server, POS Terminal, Branch Box |
| **Adapter Distribution**| **1.84 MB** (Instant over-the-air sync)| 12.21 MB (Moderate payload) | 15.26 MB (Moderate payload) |

---

## 7. Artifact & Cross-Reference Status

- **Model Specification**: Registered in `src/FtaaSService.Worker/model_registry.py` under key `HuggingFaceTB/SmolLM2-135M`.
- **Benchmark Script**: Validated via `./scripts/verify-e2e.sh` and `tests/test_onnx_exporter.py`.
- **Comparative Analysis**: [`MODEL_PERFORMANCE_COMPARISON.md`](MODEL_PERFORMANCE_COMPARISON.md)
- **High-Capacity Sibling**: [`GEMMA_PROTOTYPE_RESULTS.md`](GEMMA_PROTOTYPE_RESULTS.md)
- **Ternary Sibling**: [`BITNET_PROTOTYPE_RESULTS.md`](BITNET_PROTOTYPE_RESULTS.md)
- **Master Portfolio Brief**: [`KNOWTHANKYEW_PORTFOLIO_MASTER_BRIEF.md`](https://gist.github.com/knowthankyew/53ccf4d5a81e916f895c74e18b231e16)
