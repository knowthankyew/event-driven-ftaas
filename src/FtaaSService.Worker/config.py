import os
from pathlib import Path
import logging

# Set Apple Metal memory watermark ceiling before PyTorch backend initialization
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
import torch

# Safe fallback for torch.compile on platforms/Python versions where TorchDynamo is unavailable (e.g. Python 3.12 + PyTorch < 2.4)
_orig_torch_compile = torch.compile
def _safe_torch_compile(*args, **kwargs):
    try:
        return _orig_torch_compile(*args, **kwargs)
    except RuntimeError as ex:
        if "Dynamo is not supported" in str(ex):
            if len(args) == 1 and callable(args[0]):
                return args[0]
            def decorator(fn):
                return fn
            return decorator
        raise
torch.compile = _safe_torch_compile

# Base directories
WORKER_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = WORKER_DIR.parent.parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data"

DATA_ROOT = Path(os.getenv("DATA_ROOT", str(DEFAULT_DATA_ROOT))).resolve()
ARTIFACTS_ROOT = DATA_ROOT / "artifacts"
ARTIFACTS_ROOT.mkdir(parents=True, exist_ok=True)

# RabbitMQ Configuration
RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_PORT = int(os.getenv("RABBITMQ_PORT", "5672"))
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")

EXCHANGE_NAME = "ftaas.direct"
DLX_EXCHANGE = "ftaas.dlx"
JOB_REQUESTED_QUEUE = "ftaas.jobs.requested"
JOB_UPDATED_QUEUE = "ftaas.jobs.updated"
DLQ_QUEUE = "ftaas.jobs.dlq"
JOB_REQUESTED_ROUTING_KEY = "job.requested"
JOB_UPDATED_ROUTING_KEY = "job.updated"

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "ftaas-finetune-pipeline")

# Hardware Device Detection
def get_torch_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

DEVICE = get_torch_device()

# Air-Gap / Zero-Egress Offline Mode
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "false").lower() in ("true", "1", "yes")
if OFFLINE_MODE:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FtaaSService.Worker")
logger.info(f"Worker initialized with DEVICE={DEVICE}, DATA_ROOT={DATA_ROOT}, MLFLOW={MLFLOW_TRACKING_URI}")
