import os
from pathlib import Path
import logging
import torch

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

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FtaaSService.Worker")
logger.info(f"Worker initialized with DEVICE={DEVICE}, DATA_ROOT={DATA_ROOT}, MLFLOW={MLFLOW_TRACKING_URI}")
