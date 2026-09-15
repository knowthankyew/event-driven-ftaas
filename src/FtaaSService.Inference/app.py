import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import transformers.pytorch_utils
import transformers.generation.utils
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Patch PyTorch 2.2 MPS isin compatibility quirk in Hugging Face generation
def safe_isin_mps(elements, test_elements):
    return torch.isin(elements.cpu(), test_elements.cpu()).to(elements.device)

transformers.pytorch_utils.isin_mps_friendly = safe_isin_mps
transformers.generation.utils.isin_mps_friendly = safe_isin_mps

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FtaaSService.Inference")

# Configuration
INFERENCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INFERENCE_DIR.parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(DEFAULT_DATA_ROOT))).resolve()

DEFAULT_BASE_MODEL = os.getenv("BASE_MODEL_NAME", "HuggingFaceTB/SmolLM2-135M")

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

# State holder for pre-warmed models
class ModelStore:
    tokenizer = None
    base_model = None
    adapter_cache: Dict[str, PeftModel] = {}

model_store = ModelStore()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Pre-warming base model '{DEFAULT_BASE_MODEL}' on device {DEVICE}...")
    model_store.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
    if model_store.tokenizer.pad_token is None:
        model_store.tokenizer.pad_token = model_store.tokenizer.eos_token

    dtype = torch.float32 if DEVICE.type == "cpu" else torch.float16
    model_store.base_model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_BASE_MODEL,
        torch_dtype=dtype,
        trust_remote_code=True
    ).to(DEVICE)
    model_store.base_model.eval()

    logger.info("Base model loaded and ready in memory.")
    yield
    logger.info("Shutting down inference engine...")

app = FastAPI(
    title="FTaaS Dynamic Inference Engine",
    description="Inference service with dynamic LoRA adapter mounting and side-by-side completion comparison",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request / Response Schemas
class CompareRequest(BaseModel):
    jobId: Optional[str] = None
    baseModel: Optional[str] = DEFAULT_BASE_MODEL
    adapterPath: Optional[str] = None
    prompt: str = Field(..., min_length=3)
    maxTokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)

class GenerateRequest(BaseModel):
    adapterPath: Optional[str] = None
    prompt: str = Field(..., min_length=3)
    maxTokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)

def generate_tokens(model, tokenizer, prompt: str, max_tokens: int, temperature: float) -> tuple[str, float]:
    formatted = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)

    start_t = time.perf_counter()
    with torch.no_grad():
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "pad_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id
        }
        if temperature > 0.05:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        output_tokens = model.generate(**inputs, **generation_kwargs)
    elapsed_ms = round((time.perf_counter() - start_t) * 1000, 2)

    # Decode only the generated assistant tokens
    input_length = inputs["input_ids"].shape[1]
    new_tokens = output_tokens[0][input_length:]
    completion_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    return completion_text, elapsed_ms

def load_or_get_adapter(adapter_rel_path: str) -> PeftModel:
    if adapter_rel_path in model_store.adapter_cache:
        return model_store.adapter_cache[adapter_rel_path]

    full_adapter_path = DATA_ROOT / adapter_rel_path
    if not full_adapter_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Adapter path '{adapter_rel_path}' not found at '{full_adapter_path}'"
        )

    logger.info(f"Mounting LoRA adapter on-the-fly from {full_adapter_path}...")
    try:
        peft_model = PeftModel.from_pretrained(
            model_store.base_model,
            str(full_adapter_path)
        )
        peft_model.eval()
        model_store.adapter_cache[adapter_rel_path] = peft_model
        return peft_model
    except Exception as ex:
        logger.error(f"Failed mounting adapter from {full_adapter_path}: {ex}")
        raise HTTPException(status_code=500, detail=f"Failed loading LoRA adapter: {str(ex)}")

@app.get("/healthz")
def healthz():
    return {
        "status": "Healthy",
        "device": str(DEVICE),
        "baseModel": DEFAULT_BASE_MODEL,
        "baseModelLoaded": model_store.base_model is not None,
        "cachedAdapters": list(model_store.adapter_cache.keys())
    }

@app.post("/api/v1/inference/compare")
def compare_completions(req: CompareRequest):
    if not model_store.base_model or not model_store.tokenizer:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    # 1. Generate with raw Base Model
    base_completion, base_latency = generate_tokens(
        model_store.base_model,
        model_store.tokenizer,
        req.prompt,
        req.maxTokens,
        req.temperature
    )

    # 2. Generate with Fine-Tuned LoRA Adapter
    if not req.adapterPath:
        raise HTTPException(
            status_code=400,
            detail="adapterPath is required for side-by-side comparison"
        )

    peft_model = load_or_get_adapter(req.adapterPath)
    fine_tuned_completion, fine_tuned_latency = generate_tokens(
        peft_model,
        model_store.tokenizer,
        req.prompt,
        req.maxTokens,
        req.temperature
    )

    return {
        "jobId": req.jobId,
        "baseModel": req.baseModel or DEFAULT_BASE_MODEL,
        "adapterPath": req.adapterPath,
        "prompt": req.prompt,
        "baseCompletion": base_completion,
        "fineTunedCompletion": fine_tuned_completion,
        "latencyMs": {
            "baseModel": base_latency,
            "fineTuned": fine_tuned_latency
        }
    }

@app.post("/api/v1/inference/generate")
def generate(req: GenerateRequest):
    if not model_store.base_model or not model_store.tokenizer:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    model = load_or_get_adapter(req.adapterPath) if req.adapterPath else model_store.base_model
    completion, latency = generate_tokens(
        model,
        model_store.tokenizer,
        req.prompt,
        req.maxTokens,
        req.temperature
    )

    return {
        "adapterPath": req.adapterPath,
        "prompt": req.prompt,
        "completion": completion,
        "latencyMs": latency
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
