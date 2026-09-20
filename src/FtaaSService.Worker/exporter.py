import os
import re
import json
import time
import shutil
import hashlib
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import torch

from config import DATA_ROOT, ARTIFACTS_ROOT, DEVICE

logger = logging.getLogger("FtaaSService.Worker.Exporter")

def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a given file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

class LlamaONNXWrapper(torch.nn.Module):
    """
    Wraps a CausalLM to provide a clean single-input ['input_ids'] -> ['logits'] signature.
    Setting use_cache=False prevents dynamic kv-cache slicing artifacts in ONNX.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        outputs = self.model(input_ids=input_ids, use_cache=False)
        return outputs.logits

def export_to_edge_onnx(
    job_id: str,
    output_dir: Optional[Path] = None,
    quantize: bool = False,
    mock_for_test: bool = False
) -> Dict[str, Any]:
    """
    Exports a trained LoRA adapter and base model to a web-optimized ONNX package
    and generates an edge model manifest for in-browser client execution (e.g. lease-audit).
    """
    if not job_id or not re.match(r"^[a-zA-Z0-9_-]+$", job_id):
        raise ValueError(f"Invalid job_id format: '{job_id}'")

    logger.info(f"Starting edge ONNX export for Job {job_id} (quantize={quantize})")
    start_time = time.time()

    # 1. Resolve and verify adapter directory
    adapter_dir = ARTIFACTS_ROOT / job_id / "model_adapters"
    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found at {adapter_dir}")

    config_file = adapter_dir / "adapter_config.json"
    if not config_file.exists() or config_file.stat().st_size < 10:
        raise RuntimeError(f"Invalid or truncated adapter_config.json in {adapter_dir}")

    with open(config_file, "r", encoding="utf-8") as f:
        adapter_config = json.load(f)

    base_model_name = adapter_config.get("base_model_name_or_path", "HuggingFaceTB/SmolLM2-135M")

    safetensors_file = adapter_dir / "adapter_model.safetensors"
    bin_file = adapter_dir / "adapter_model.bin"
    weights_file = safetensors_file if safetensors_file.exists() else bin_file

    if not weights_file.exists() or weights_file.stat().st_size < 100 * 1024:
        raise RuntimeError(f"Missing or truncated adapter weights file in {adapter_dir}")

    adapter_size_bytes = weights_file.stat().st_size

    # 2. Setup output directory
    edge_output_dir = output_dir or (ARTIFACTS_ROOT / job_id / "edge_onnx")
    edge_output_dir.mkdir(parents=True, exist_ok=True)

    model_onnx_path = edge_output_dir / "model.onnx"
    quantization_type = "int8" if quantize else "fp32"

    # 3. Export / Package ONNX model
    if mock_for_test:
        # Fast path for automated unit test execution
        logger.info("Executing mock ONNX export for test validation...")
        model_onnx_path.write_bytes(b"ONNX_MOCK_WEIGHTS_FOR_TESTING" * 1024)
    else:
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            import onnx

            logger.info(f"Loading base model {base_model_name} to merge with LoRA weights...")
            tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir))
            
            # If model.onnx already exists and is valid, reuse it
            if model_onnx_path.exists() and model_onnx_path.stat().st_size > 1024 * 1024:
                logger.info(f"Reusing existing ONNX model at {model_onnx_path}")
            else:
                base_model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                )
                peft_model = PeftModel.from_pretrained(base_model, str(adapter_dir))
                merged_model = peft_model.merge_and_unload()
                merged_model.eval()

                wrapped_model = LlamaONNXWrapper(merged_model)

                logger.info(f"Tracing and exporting merged model to ONNX: {model_onnx_path}")
                dummy_input = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
                torch.onnx.export(
                    wrapped_model,
                    (dummy_input,),
                    str(model_onnx_path),
                    input_names=["input_ids"],
                    output_names=["logits"],
                    dynamic_axes={
                        "input_ids": {0: "batch_size", 1: "sequence_length"},
                        "logits": {0: "batch_size", 1: "sequence_length"}
                    },
                    opset_version=17,
                    do_constant_folding=True
                )

                # Verify ONNX model integrity
                onnx_model = onnx.load(str(model_onnx_path))
                onnx.checker.check_model(onnx_model)
                logger.info("ONNX model structure checked successfully.")

            if quantize and model_onnx_path.stat().st_size > 300 * 1024 * 1024:
                logger.info("Applying dynamic INT8 quantization to ONNX graph...")
                from onnxruntime.quantization import quantize_dynamic, QuantType
                quantized_path = edge_output_dir / "model_int8.onnx"
                quantize_dynamic(str(model_onnx_path), str(quantized_path), weight_type=QuantType.QInt8)
                shutil.move(str(quantized_path), str(model_onnx_path))
                logger.info(f"Quantization complete: {model_onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
        except Exception as ex:
            logger.error(f"Full PyTorch-to-ONNX tracing failed for Job {job_id}: {ex}")
            raise RuntimeError(f"Edge ONNX export failed: {ex}") from ex

    # 4. Copy Tokenizer & Config Files to edge output directory
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt"]:
        src_file = adapter_dir / fname
        if src_file.exists():
            shutil.copy2(src_file, edge_output_dir / fname)

    # Copy adapter config as metadata
    shutil.copy2(config_file, edge_output_dir / "adapter_config.json")

    # 5. Compute Artifact Metadata & Checksums
    onnx_size_bytes = model_onnx_path.stat().st_size
    onnx_sha256 = compute_file_sha256(model_onnx_path)

    # 6. Generate Edge Model Manifest
    manifest = {
        "jobId": job_id,
        "modelName": f"ftaas-edge-{job_id[:8]}",
        "baseModel": base_model_name,
        "architecture": "CausalLM",
        "parameterCount": "135M",
        "adapterSizeBytes": adapter_size_bytes,
        "onnxFileName": "model.onnx",
        "onnxSizeBytes": onnx_size_bytes,
        "onnxSha256": onnx_sha256,
        "quantization": quantization_type,
        "contextLength": 2048,
        "supportedExecutionProviders": ["webgpu", "wasm"],
        "chatTemplate": "smollm2",
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "zeroEgressInvariant": True
    }

    manifest_file = edge_output_dir / "edge_model_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    duration_sec = round(time.time() - start_time, 3)
    logger.info(f"Edge export completed in {duration_sec}s. Manifest written to {manifest_file}")

    # 7. Safe Lifecycle Telemetry Emission
    try:
        from consumer import emit_lifecycle_span
        emit_lifecycle_span(
            span_name="job.edge_exported",
            job_id=job_id,
            status="Succeeded",
            duration_sec=duration_sec,
            attributes={
                "base_model": base_model_name,
                "adapter_size_bytes": adapter_size_bytes,
                "device": str(DEVICE),
                "action": "export_edge_onnx"
            }
        )
    except Exception as telex:
        logger.warning(f"Could not emit edge export telemetry span: {telex}")

    return {
        "jobId": job_id,
        "manifest": manifest,
        "outputDir": str(edge_output_dir),
        "durationSec": duration_sec
    }
