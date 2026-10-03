# Gemma 2B Prototype Evaluation & Results

This document tracks benchmarking, resource consumption, and quality comparison between the baseline ultra-compact model (`HuggingFaceTB/SmolLM2-135M`) and the high-capacity reasoning model (`google/gemma-2-2b-it`).

---

## 1. Prototype Model Matrix

| Specification | SmolLM2-135M | Gemma 2 2B IT |
| :--- | :--- | :--- |
| **Model ID** | `HuggingFaceTB/SmolLM2-135M` | `google/gemma-2-2b-it` |
| **Parameter Count** | ~135M | ~2.6B |
| **Chat Template** | ChatML (`<\|im_start\|>`) | Gemma (`<start_of_turn>`) |
| **LoRA Target Modules** | `["q_proj", "v_proj"]` | `["q_proj", "v_proj", "k_proj", "o_proj"]` |
| **Context Length** | 2,048 tokens | 8,192 tokens |
| **Min GPU VRAM (Training)**| 0.5 GB | ~8.0 GB |
| **Min GPU VRAM (Inference)**| 0.3 GB | ~5.0 GB |
| **Hugging Face License** | Open (Apache 2.0) | Gated (Gemma Terms of Use + `HF_TOKEN`) |
| **Exported Adapter Size** | ~1.8 MB (`q_proj`, `v_proj`) | ~12.2 MB (`q_proj`, `v_proj`, `k_proj`, `o_proj`) |

---

## 2. Experimental Run Benchmark Template

| Run Date | Base Model | Dataset | Records | Epochs | LoRA (r/α) | Device | Training Loss | Duration | Notes / Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| *Baseline* | `SmolLM2-135M` | `fintech-compliance` | 100 | 3 | 8 / 32 | Apple Silicon (MPS) | 0.312 | 2m 14s | Production Baseline |
| 2026-10-03 | `gemma-2-2b-it` | `financial-sentiment` | 20 | 3 | 8 / 32 | mps | 11.4547 | 7m 3s | Live Benchmark Run (Adapter: 12.21 MB) |

---

## 3. Operational Observations

### Hugging Face Gated Licensing
`google/gemma-2-2b-it` requires users to accept Google's terms on Hugging Face before weights can be downloaded. If the worker encounters an HTTP 401/403 or gated model exception, `FtaaSService.Worker.Trainer` immediately catches it and emits a friendly, actionable guidance error:
```text
Cannot load 'google/gemma-2-2b-it': Hugging Face authentication required.
Accept the model license at huggingface.co/google/gemma-2-2b-it and set the HF_TOKEN environment variable.
```

### Dynamic Inference Routing
The inference service serves models based on `BASE_MODEL_NAME`. If a user attempts to mount a Gemma adapter on an inference engine currently hosting SmolLM2 (or vice-versa), `FtaaSService.Inference` intercepts the mismatch and returns:
```json
{
  "status_code": 409,
  "detail": {
    "error": "base_model_mismatch",
    "message": "Adapter was trained on 'google/gemma-2-2b-it' but the inference service has 'HuggingFaceTB/SmolLM2-135M' loaded. Outputs would be garbage.",
    "adapterBaseModel": "google/gemma-2-2b-it",
    "loadedBaseModel": "HuggingFaceTB/SmolLM2-135M",
    "hint": "Restart the inference service with BASE_MODEL_NAME=google/gemma-2-2b-it"
  }
}
```

### Apple Silicon (MPS) Memory & Watermark Tuning
Running a 2.6B parameter model on macOS with Apple Silicon unified memory introduces unique constraints:
1. **Precision Selection**: PyTorch MPS does not natively support `bfloat16`, while loading in `float32` requires >10.4 GB VRAM, triggering Metal allocation limits. `FtaaSService.Worker.Trainer` enforces `torch.float16` when targeting MPS, keeping base model memory at ~5.2 GB.
2. **Watermark Bypass**: Apple Metal sets a default process ceiling (~6.7 GB on 16 GB machines). Setting `PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0` allows the process to utilize available unified system RAM without premature out-of-memory aborts.
3. **Dynamic Gradient Accumulation**: For models with $\ge 2\text{B}$ parameters on MPS, training automatically adapts micro-batches: `per_device_train_batch_size = 1` and `gradient_accumulation_steps = batch_size`. This preserves the effective batch optimization while reducing activation tensors in VRAM by up to 75%.

### Telemetry & Experiment Verification
The live prototype run completed 30 training steps (3 epochs over 20 records):
- **MLflow Run ID**: `a3f2e22c38cb4514a30d0ec902adba89`
- **Initial Step Loss**: `12.1813`
- **Final Step Loss**: `11.4547`
- **Artifacts Generated**: `adapter_config.json`, `adapter_model.safetensors` (12.21 MB), `tokenizer.json`, and `edge_model_manifest.json` under `ml/data/artifacts/gemma-bench-1791032257/model_adapters/`.

