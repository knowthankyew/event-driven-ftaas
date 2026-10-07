import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Request, Response
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

DEFAULT_BASE_MODEL = os.getenv("BASE_MODEL_NAME", "microsoft/BitNet-b1.58-2B-4T")

# Make model_registry importable from the Worker source directory
import sys as _sys
_worker_src = str(PROJECT_ROOT / "src" / "FtaaSService.Worker")
if _worker_src not in _sys.path:
    _sys.path.insert(0, _worker_src)

import hmac
from model_registry import get_model_spec, format_inference_prompt
from bitnet_engine import (
    is_bitnet_model,
    is_bitnet_available,
    get_bitnet_status,
    generate_bitnet_sync,
    generate_bitnet,
    embed_bitnet_sync,
    embed_bitnet,
    ContextOverflowError,
    BitNetExecutionError,
    TokenizerUnavailableError,
)

def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

from collections import OrderedDict

MAX_CACHED_ADAPTERS = int(os.getenv("MAX_CACHED_ADAPTERS", "5"))

# State holder for pre-warmed models with bounded LRU adapter cache
class ModelStore:
    tokenizer = None
    base_model = None
    is_bitnet: bool = False
    backend: str = "transformers"
    adapter_cache: OrderedDict[str, PeftModel] = OrderedDict()

model_store = ModelStore()

class InferenceMetrics:
    def __init__(self):
        self.start_time = time.time()
        self.requests_total = {
            "compare": {"success": 0, "error": 0},
            "generate": {"success": 0, "error": 0},
            "embed": {"success": 0, "error": 0}
        }
        self.latency_sum_ms = {"compare": 0.0, "generate": 0.0, "embed": 0.0}
        self.latency_count = {"compare": 0, "generate": 0, "embed": 0}

    def record_request(self, endpoint: str, status: str, latency_ms: float = 0.0):
        if endpoint in self.requests_total:
            self.requests_total[endpoint][status] = self.requests_total[endpoint].get(status, 0) + 1
            if status == "success":
                self.latency_sum_ms[endpoint] += latency_ms
                self.latency_count[endpoint] += 1

metrics = InferenceMetrics()

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"Configuring base model '{DEFAULT_BASE_MODEL}'...")

    if is_bitnet_model(DEFAULT_BASE_MODEL):
        logger.info(f"Initializing native BitNet C++ runtime for '{DEFAULT_BASE_MODEL}'...")
        try:
            model_store.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
        except Exception as ex:
            logger.warning(f"Could not load HuggingFace tokenizer for BitNet: {ex}")
        model_store.is_bitnet = True
        model_store.backend = "bitnet_cpp"
        model_store.base_model = None
        if is_bitnet_available():
            logger.info("BitNet native C++ runtime verified and ready.")
        else:
            logger.warning("BitNet C++ runtime binary or model weights not detected at configured path.")
    else:
        logger.info(f"Pre-warming PyTorch base model '{DEFAULT_BASE_MODEL}' on device {DEVICE}...")
        model_store.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_BASE_MODEL)
        if model_store.tokenizer.pad_token is None:
            model_store.tokenizer.pad_token = model_store.tokenizer.eos_token

        dtype = torch.bfloat16 if DEVICE.type == "cuda" else (torch.float16 if DEVICE.type == "mps" else torch.float32)
        model_store.base_model = AutoModelForCausalLM.from_pretrained(
            DEFAULT_BASE_MODEL,
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(DEVICE)
        model_store.base_model.eval()
        model_store.is_bitnet = False
        model_store.backend = "transformers"
        logger.info("Base model loaded and ready in memory.")

    yield
    logger.info("Shutting down inference engine...")

app = FastAPI(
    title="FTaaS Dynamic Inference Engine",
    description="Inference service with dynamic LoRA adapter mounting and side-by-side completion comparison",
    version="1.0.0",
    lifespan=lifespan
)

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("FTAAS_ALLOWED_ORIGINS", "").split(",") if o.strip()] or DEFAULT_ALLOWED_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

from fastapi.middleware.trustedhost import TrustedHostMiddleware
DEFAULT_ALLOWED_HOSTS = ["localhost", "127.0.0.1", "[::1]"]
ALLOWED_HOSTS = [h.strip() for h in os.getenv("FTAAS_ALLOWED_HOSTS", "").split(",") if h.strip()] or DEFAULT_ALLOWED_HOSTS
app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)

FTAAS_API_KEY = os.getenv("FTAAS_API_KEY")

@app.middleware("http")
async def verify_api_key_if_configured(request: Request, call_next):
    if FTAAS_API_KEY:
        if request.method == "OPTIONS" or request.url.path in ("/healthz", "/metrics"):
            return await call_next(request)
        auth_header = request.headers.get("Authorization", "")
        api_key_header = request.headers.get("X-API-Key", "")
        token = auth_header.replace("Bearer ", "").strip() if auth_header.startswith("Bearer ") else api_key_header
        token_bytes = token.encode("utf-8")
        key_bytes = FTAAS_API_KEY.encode("utf-8")
        if not hmac.compare_digest(token_bytes, key_bytes):
            return Response(content='{"detail":"Unauthorized"}', status_code=401, media_type="application/json")
    return await call_next(request)

# Request / Response Schemas
class CompareRequest(BaseModel):
    jobId: Optional[str] = None
    baseModel: Optional[str] = DEFAULT_BASE_MODEL
    adapterPath: Optional[str] = None
    prompt: str = Field(..., min_length=3, max_length=16384)
    maxTokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)


class EmbedRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2048)


class GenerateRequest(BaseModel):
    baseModel: Optional[str] = DEFAULT_BASE_MODEL
    adapterPath: Optional[str] = None
    prompt: str = Field(..., min_length=3, max_length=16384)
    maxTokens: int = Field(default=64, ge=1, le=256)
    temperature: float = Field(default=0.2, ge=0.0, le=1.0)

def generate_tokens(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int,
    temperature: float,
    base_model_id: str = DEFAULT_BASE_MODEL
) -> tuple[str, float]:
    formatted = format_inference_prompt(base_model_id, prompt)
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
        model_store.adapter_cache.move_to_end(adapter_rel_path)
        return model_store.adapter_cache[adapter_rel_path]

    full_adapter_path = DATA_ROOT / adapter_rel_path
    if not full_adapter_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Adapter path '{adapter_rel_path}' not found at '{full_adapter_path}'"
        )

    # Verify adapter integrity: reject zero-byte or truncated files from interrupted writes
    config_file = full_adapter_path / "adapter_config.json"
    safetensors_file = full_adapter_path / "adapter_model.safetensors"
    bin_file = full_adapter_path / "adapter_model.bin"

    if not config_file.exists() or config_file.stat().st_size < 10:
        raise HTTPException(
            status_code=422,
            detail=f"Adapter at '{adapter_rel_path}' is corrupted: missing or zero-byte adapter_config.json"
        )

    weights_file = safetensors_file if safetensors_file.exists() else bin_file
    if not weights_file.exists() or weights_file.stat().st_size < 100 * 1024:
        raise HTTPException(
            status_code=422,
            detail=f"Adapter at '{adapter_rel_path}' is corrupted: weight file is missing or truncated (<100KB)"
        )

    # Verify the adapter was trained on the same base model currently loaded.
    # Mismatched base models produce garbage output — return a 409 with an actionable restart hint.
    try:
        with open(config_file, "r", encoding="utf-8") as _cf:
            _adapter_cfg = json.load(_cf)
        adapter_base_model = _adapter_cfg.get("base_model_name_or_path", "")
        if adapter_base_model and adapter_base_model != DEFAULT_BASE_MODEL:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "base_model_mismatch",
                    "message": (
                        f"Adapter was trained on '{adapter_base_model}' but the inference "
                        f"service has '{DEFAULT_BASE_MODEL}' loaded. Outputs would be garbage."
                    ),
                    "adapterBaseModel": adapter_base_model,
                    "loadedBaseModel": DEFAULT_BASE_MODEL,
                    "hint": f"Restart the inference service with BASE_MODEL_NAME={adapter_base_model}",
                }
            )
    except HTTPException:
        raise
    except Exception as _cfg_err:
        logger.warning(f"Could not read adapter base model for mismatch check: {_cfg_err}")

    logger.info(f"Mounting verified LoRA adapter ({weights_file.stat().st_size / 1024 / 1024:.2f} MB) on-the-fly from {full_adapter_path}...")
    try:
        peft_model = PeftModel.from_pretrained(
            model_store.base_model,
            str(full_adapter_path)
        )
        peft_model.to(DEVICE)
        peft_model.eval()

        # Bounded LRU eviction
        if len(model_store.adapter_cache) >= MAX_CACHED_ADAPTERS:
            evicted_path, _ = model_store.adapter_cache.popitem(last=False)
            logger.info(f"LRU capacity reached ({MAX_CACHED_ADAPTERS}). Evicted adapter: {evicted_path}")

        model_store.adapter_cache[adapter_rel_path] = peft_model
        return peft_model
    except Exception as ex:
        logger.error(f"Failed mounting adapter from {full_adapter_path}: {ex}")
        raise HTTPException(status_code=500, detail="Failed loading LoRA adapter.")

@app.get("/healthz")
def healthz():
    try:
        _spec = get_model_spec(DEFAULT_BASE_MODEL)
        _chat_template = _spec.chat_template.value
    except Exception:
        _chat_template = "unknown"

    base_model_loaded = (model_store.is_bitnet and is_bitnet_available()) or (model_store.base_model is not None)
    raw_bitnet = get_bitnet_status()
    sanitized_bitnet = {
        "available": raw_bitnet.get("available", False),
        "cliExecutable": raw_bitnet.get("cliExecutable", False),
        "completionExecutable": raw_bitnet.get("completionExecutable", False),
        "modelPresent": raw_bitnet.get("modelPresent", False),
        "modelSizeMb": raw_bitnet.get("modelSizeMb", 0.0),
        "embedAvailable": raw_bitnet.get("embedAvailable", False),
        "tokenizerAvailable": raw_bitnet.get("tokenizerAvailable", False),
        "embedCliExecutable": raw_bitnet.get("embedCliExecutable", False),
        "embedModelPresent": raw_bitnet.get("embedModelPresent", False),
        "embedModelSizeMb": raw_bitnet.get("embedModelSizeMb", 0.0),
        "threads": raw_bitnet.get("threads", 4),
    }

    return {
        "status": "Healthy",
        "device": str(DEVICE),
        "baseModel": DEFAULT_BASE_MODEL,
        "baseModelLoaded": base_model_loaded,
        "backend": model_store.backend,
        "bitnet": sanitized_bitnet,
        "cachedAdapterCount": len(model_store.adapter_cache),
        "chatTemplate": _chat_template,
    }

@app.get("/metrics")
def get_metrics():
    uptime = time.time() - metrics.start_time
    cached_adapters = len(model_store.adapter_cache)
    base_loaded = 1 if ((model_store.is_bitnet and is_bitnet_available()) or (model_store.base_model is not None)) else 0

    lines = [
        "# HELP ftaas_inference_uptime_seconds Total runtime of inference service in seconds.",
        "# TYPE ftaas_inference_uptime_seconds gauge",
        f"ftaas_inference_uptime_seconds {uptime:.2f}",
        "",
        "# HELP ftaas_inference_cached_adapters Current number of LoRA adapters cached in memory.",
        "# TYPE ftaas_inference_cached_adapters gauge",
        f"ftaas_inference_cached_adapters {cached_adapters}",
        "",
        "# HELP ftaas_inference_max_cached_adapters Maximum capacity of the LRU adapter cache.",
        "# TYPE ftaas_inference_max_cached_adapters gauge",
        f"ftaas_inference_max_cached_adapters {MAX_CACHED_ADAPTERS}",
        "",
        "# HELP ftaas_inference_base_model_loaded Whether base model is pre-warmed and ready in memory (1=loaded, 0=unloaded).",
        "# TYPE ftaas_inference_base_model_loaded gauge",
        f'ftaas_inference_base_model_loaded{{model="{DEFAULT_BASE_MODEL}"}} {base_loaded}',
        "",
        "# HELP ftaas_inference_embed_model_loaded Whether 1-bit embedding model is loaded and ready.",
        "# TYPE ftaas_inference_embed_model_loaded gauge",
        f'ftaas_inference_embed_model_loaded 1' if get_bitnet_status().get('embedAvailable') else f'ftaas_inference_embed_model_loaded 0',
        "",
        "# HELP ftaas_inference_requests_total Total inference requests processed by endpoint and status.",
        "# TYPE ftaas_inference_requests_total counter",
    ]
    for ep, status_dict in metrics.requests_total.items():
        for st, count in status_dict.items():
            lines.append(f'ftaas_inference_requests_total{{endpoint="{ep}",status="{st}"}} {count}')

    lines.extend([
        "",
        "# HELP ftaas_inference_latency_ms_sum Total accumulated inference latency in milliseconds.",
        "# TYPE ftaas_inference_latency_ms_sum counter",
    ])
    for ep, lat_sum in metrics.latency_sum_ms.items():
        lines.append(f'ftaas_inference_latency_ms_sum{{endpoint="{ep}"}} {lat_sum:.2f}')

    lines.extend([
        "",
        "# HELP ftaas_inference_latency_ms_count Count of measured requests for latency calculation.",
        "# TYPE ftaas_inference_latency_ms_count counter",
    ])
    for ep, count in metrics.latency_count.items():
        lines.append(f'ftaas_inference_latency_ms_count{{endpoint="{ep}"}} {count}')

    lines.append("")
    return Response(content="\n".join(lines), media_type="text/plain; version=0.0.4; charset=utf-8")

@app.post("/api/v1/inference/compare")
def compare_completions(req: CompareRequest):
    try:
        target_base = req.baseModel or DEFAULT_BASE_MODEL

        # Route 1: Native BitNet C++ runtime
        if is_bitnet_model(target_base):
            if not is_bitnet_available():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "BitNet C++ native runtime is not available on this host. "
                        "Check BITNET_CLI_PATH and BITNET_MODEL_PATH."
                    )
                )
            if not req.adapterPath:
                raise HTTPException(
                    status_code=400,
                    detail="adapterPath is required for side-by-side comparison"
                )

            # 1. Base model completion via C++ engine
            base_completion, base_latency = generate_bitnet_sync(
                req.prompt,
                req.maxTokens,
                req.temperature
            )

            # 2. Fine-tuned adapter completion
            full_adapter_path = str(DATA_ROOT / req.adapterPath)
            fine_tuned_completion, fine_tuned_latency = generate_bitnet_sync(
                req.prompt,
                req.maxTokens,
                req.temperature,
                adapter_path=full_adapter_path
            )

            metrics.record_request("compare", "success", max(base_latency, fine_tuned_latency))
            return {
                "jobId": req.jobId,
                "baseModel": target_base,
                "adapterPath": req.adapterPath,
                "prompt": req.prompt,
                "baseCompletion": base_completion,
                "fineTunedCompletion": fine_tuned_completion,
                "latencyMs": {
                    "baseModel": base_latency,
                    "fineTuned": fine_tuned_latency
                }
            }

        # Route 2: Standard PyTorch HuggingFace runtime
        if not model_store.base_model or not model_store.tokenizer:
            raise HTTPException(status_code=503, detail="Model is still loading.")

        # Guard: if req.baseModel is specified and doesn't match loaded PyTorch model, reject with 409
        if target_base != DEFAULT_BASE_MODEL:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "base_model_mismatch",
                    "message": (
                        f"Requested base model '{target_base}' does not match "
                        f"loaded inference model '{DEFAULT_BASE_MODEL}'."
                    ),
                    "adapterBaseModel": target_base,
                    "loadedBaseModel": DEFAULT_BASE_MODEL,
                    "hint": f"Restart the inference service with BASE_MODEL_NAME={target_base}",
                }
            )

        # 1. Generate with raw Base Model
        base_completion, base_latency = generate_tokens(
            model_store.base_model,
            model_store.tokenizer,
            req.prompt,
            req.maxTokens,
            req.temperature,
            base_model_id=DEFAULT_BASE_MODEL
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
            req.temperature,
            base_model_id=DEFAULT_BASE_MODEL
        )

        metrics.record_request("compare", "success", max(base_latency, fine_tuned_latency))
        return {
            "jobId": req.jobId,
            "baseModel": DEFAULT_BASE_MODEL,
            "adapterPath": req.adapterPath,
            "prompt": req.prompt,
            "baseCompletion": base_completion,
            "fineTunedCompletion": fine_tuned_completion,
            "latencyMs": {
                "baseModel": base_latency,
                "fineTuned": fine_tuned_latency
            }
        }
    except ContextOverflowError as coe:
        metrics.record_request("compare", "error")
        raise HTTPException(status_code=413, detail=str(coe))
    except HTTPException:
        metrics.record_request("compare", "error")
        raise
    except TimeoutError:
        metrics.record_request("compare", "error")
        raise HTTPException(status_code=504, detail="Inference request timed out.")
    except Exception as ex:
        metrics.record_request("compare", "error")
        logger.error(f"Failed to generate comparison: {ex}")
        raise HTTPException(status_code=500, detail="Failed to generate model completion.")

@app.post("/api/v1/inference/generate")
def generate(req: GenerateRequest):
    try:
        target_base = req.baseModel or DEFAULT_BASE_MODEL

        # Route 1: Native BitNet C++ runtime
        if is_bitnet_model(target_base):
            if not is_bitnet_available():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "BitNet C++ native runtime is not available on this host. "
                        "Check BITNET_CLI_PATH and BITNET_MODEL_PATH."
                    )
                )
            full_adapter = str(DATA_ROOT / req.adapterPath) if req.adapterPath else None
            completion, latency = generate_bitnet_sync(
                req.prompt,
                req.maxTokens,
                req.temperature,
                adapter_path=full_adapter
            )
            metrics.record_request("generate", "success", latency)
            return {
                "baseModel": target_base,
                "adapterPath": req.adapterPath,
                "prompt": req.prompt,
                "completion": completion,
                "latencyMs": latency
            }

        # Route 2: Standard PyTorch HuggingFace runtime
        if not model_store.base_model or not model_store.tokenizer:
            raise HTTPException(status_code=503, detail="Model is still loading.")

        model = load_or_get_adapter(req.adapterPath) if req.adapterPath else model_store.base_model
        completion, latency = generate_tokens(
            model,
            model_store.tokenizer,
            req.prompt,
            req.maxTokens,
            req.temperature,
            base_model_id=DEFAULT_BASE_MODEL
        )

        metrics.record_request("generate", "success", latency)
        return {
            "baseModel": target_base,
            "adapterPath": req.adapterPath,
            "prompt": req.prompt,
            "completion": completion,
            "latencyMs": latency
        }
    except ContextOverflowError as coe:
        metrics.record_request("generate", "error")
        raise HTTPException(status_code=413, detail=str(coe))
    except HTTPException:
        metrics.record_request("generate", "error")
        raise
    except TimeoutError:
        metrics.record_request("generate", "error")
        raise HTTPException(status_code=504, detail="Inference request timed out.")
    except Exception as ex:
        metrics.record_request("generate", "error")
        logger.error(f"Failed to generate completion: {ex}")
        raise HTTPException(status_code=500, detail="Failed to generate model completion.")


@app.post("/api/v1/inference/embed")
def embed_text(req: EmbedRequest):
    try:
        status = get_bitnet_status()
        if not status.get("embedAvailable"):
            raise HTTPException(
                status_code=503,
                detail="BitNet C++ native embedding runtime or weights are not available."
            )

        completion, latency = embed_bitnet_sync(req.prompt)
        metrics.record_request("embed", "success", latency)
        return {
            "embedding": completion,
            "latencyMs": latency
        }
    except ContextOverflowError as coe:
        metrics.record_request("embed", "error")
        raise HTTPException(status_code=413, detail=str(coe))
    except TokenizerUnavailableError as tue:
        metrics.record_request("embed", "error")
        logger.error(f"BitNet embedding tokenizer unavailable: {tue}")
        raise HTTPException(status_code=503, detail="Token verification service is unavailable. Please retry later.")
    except BitNetExecutionError as bee:
        metrics.record_request("embed", "error")
        logger.error(f"BitNet execution failed: {bee}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding vector.")
    except HTTPException:
        metrics.record_request("embed", "error")
        raise
    except TimeoutError:
        metrics.record_request("embed", "error")
        raise HTTPException(status_code=504, detail="BitNet embedding request timed out.")
    except Exception as e:
        metrics.record_request("embed", "error")
        logger.error(f"Failed to generate embedding: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate embedding vector.")

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("FTAAS_INFERENCE_HOST", os.getenv("HOST", "127.0.0.1"))
    port = int(os.getenv("PORT", "8000"))
    if host not in ("127.0.0.1", "localhost", "::1"):
        logger.warning(
            f"Inference service bound to non-loopback address '{host}'. "
            f"Zero-egress privacy invariants require loopback binding (127.0.0.1)."
        )
    uvicorn.run("app:app", host=host, port=port, reload=False)
