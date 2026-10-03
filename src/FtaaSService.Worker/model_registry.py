"""
model_registry.py
Single source of truth for all supported base models in the FTaaS pipeline.

Defines per-model:
  - Chat template formatting (training and inference variants)
  - LoRA target_modules (architecture-specific attention projection names)
  - Manifest metadata (parameter count display, context length)
  - Hardware requirements (minimum VRAM for training)
  - Auth requirements (HuggingFace gated models)

Shared by: trainer.py, exporter.py, app.py (inference)
"""

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import torch

logger = logging.getLogger("FtaaSService.Worker.ModelRegistry")


class ChatTemplate(str, Enum):
    CHATML = "chatml"
    GEMMA = "gemma"


@dataclass(frozen=True)
class ModelSpec:
    """Immutable specification for a supported base model."""
    model_id: str
    display_name: str
    chat_template: ChatTemplate
    lora_target_modules: list
    parameter_count_display: str
    context_length: int
    min_gpu_vram_gb: float
    requires_hf_auth: bool
    hardware_disclaimer: Optional[str] = None


# ---------------------------------------------------------------------------
# Registered Model Catalog
# ---------------------------------------------------------------------------

_REGISTRY: dict = {
    "HuggingFaceTB/SmolLM2-135M": ModelSpec(
        model_id="HuggingFaceTB/SmolLM2-135M",
        display_name="SmolLM2-135M",
        chat_template=ChatTemplate.CHATML,
        lora_target_modules=["q_proj", "v_proj"],
        parameter_count_display="135M",
        context_length=2048,
        min_gpu_vram_gb=0.5,
        requires_hf_auth=False,
        hardware_disclaimer=None,
    ),
    "google/gemma-2-2b-it": ModelSpec(
        model_id="google/gemma-2-2b-it",
        display_name="Gemma 2 2B IT",
        chat_template=ChatTemplate.GEMMA,
        lora_target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        parameter_count_display="2B",
        context_length=8192,
        min_gpu_vram_gb=8.0,
        requires_hf_auth=True,
        hardware_disclaimer=(
            "Requires ~8 GB GPU VRAM for LoRA training and ~5 GB for inference. "
            "Also requires accepting the Gemma license at "
            "huggingface.co/google/gemma-2-2b-it and setting the HF_TOKEN "
            "environment variable before starting the worker."
        ),
    ),
}

# Convenience aliases (used by API layer for validation)
SMOLLM2 = "HuggingFaceTB/SmolLM2-135M"
GEMMA_2_2B_IT = "google/gemma-2-2b-it"
SUPPORTED_MODEL_IDS: frozenset = frozenset(_REGISTRY.keys())


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_model_spec(model_id: str) -> ModelSpec:
    """
    Returns the ModelSpec for the given model_id.
    Raises ValueError for unknown/unsupported model IDs.
    """
    spec = _REGISTRY.get(model_id)
    if spec is None:
        supported = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(
            f"Unsupported base model: '{model_id}'. "
            f"Supported models: [{supported}]"
        )
    return spec


def format_training_prompt(model_id: str, prompt: str, completion: str) -> str:
    """
    Formats a prompt/completion pair for LoRA training using the model's
    native chat template. Returns the full formatted string including both
    user turn and assistant/model turn with closing end tokens (for training).
    """
    spec = get_model_spec(model_id)

    if spec.chat_template == ChatTemplate.CHATML:
        return (
            f"<|im_start|>user\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{completion}<|im_end|>"
        )
    elif spec.chat_template == ChatTemplate.GEMMA:
        return (
            f"<start_of_turn>user\n{prompt}<end_of_turn>\n"
            f"<start_of_turn>model\n{completion}<end_of_turn>"
        )
    else:
        raise NotImplementedError(
            f"No training template for chat_template={spec.chat_template!r}"
        )


def format_inference_prompt(model_id: str, prompt: str) -> str:
    """
    Formats a user prompt for inference — the model turn is left open-ended
    so the model generates a continuation from here.
    """
    spec = get_model_spec(model_id)

    if spec.chat_template == ChatTemplate.CHATML:
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    elif spec.chat_template == ChatTemplate.GEMMA:
        return f"<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n"
    else:
        raise NotImplementedError(
            f"No inference template for chat_template={spec.chat_template!r}"
        )


def preflight_check(model_id: str) -> dict:
    """
    Non-blocking hardware compatibility check for the given model.
    Returns {"passed": bool, "warning": str | None}.
    A failed check never raises — it returns a warning string for logging/MLflow tagging.
    """
    spec = get_model_spec(model_id)
    warnings = []

    # VRAM check (CUDA only — MPS and CPU don't expose VRAM query easily)
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            vram_gb = props.total_memory / (1024 ** 3)
            if vram_gb < spec.min_gpu_vram_gb:
                warnings.append(
                    f"GPU VRAM is {vram_gb:.1f} GB but {spec.display_name} "
                    f"requires ~{spec.min_gpu_vram_gb} GB for LoRA training. "
                    "Training may fail with OOM. Consider reducing batch size or LoRA rank."
                )
        except Exception as exc:
            logger.debug(f"Could not query CUDA device properties: {exc}")
    elif not torch.backends.mps.is_available():
        # CPU-only — warn for large models
        if spec.min_gpu_vram_gb >= 4.0:
            warnings.append(
                f"{spec.display_name} is a large model (~{spec.parameter_count_display} parameters). "
                "Training on CPU will be very slow (hours per epoch). "
                "A CUDA GPU or Apple Silicon with MPS is strongly recommended."
            )

    # Auth check — reminder only, we can't verify the token here
    if spec.requires_hf_auth:
        if not os.getenv("HF_TOKEN"):
            warnings.append(
                f"{spec.display_name} is a gated model that requires a Hugging Face access token. "
                "Set the HF_TOKEN environment variable before starting the worker, "
                "and ensure you have accepted the license at huggingface.co/google/gemma-2-2b-it."
            )

    if warnings:
        combined = " | ".join(warnings)
        logger.warning(f"[PREFLIGHT] {model_id}: {combined}")
        return {"passed": False, "warning": combined}

    return {"passed": True, "warning": None}
