import os
import sys
import time
import json
import re
import hashlib
from pathlib import Path

# Paths
DEV_ROOT = Path(__file__).resolve().parent.parent
WORKER_SRC = DEV_ROOT / "src" / "FtaaSService.Worker"
sys.path.insert(0, str(WORKER_SRC))

# Ensure local SQLite tracking URI for MLflow if Docker is offline
os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{DEV_ROOT}/mlflow_data/mlflow.db")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from trainer import train_job
from config import DATA_ROOT, ARTIFACTS_ROOT, DEVICE

def compute_sha256(filepath: Path) -> str:
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

def main():
    print("=" * 65)
    print("🚀 Running Gemma 2 2B IT LoRA Benchmark")
    print("=" * 65)

    base_model = "google/gemma-2-2b-it"
    rel_dataset = "datasets/sample-financial-sentiment.jsonl"
    full_dataset = DATA_ROOT / rel_dataset

    if not full_dataset.exists():
        print(f"Error: Dataset not found at {full_dataset}")
        sys.exit(1)

    dataset_hash = compute_sha256(full_dataset)
    with open(full_dataset, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"• Base Model:      {base_model}")
    print(f"• Dataset:         {rel_dataset} ({len(records)} records)")
    print(f"• SHA256:          {dataset_hash[:16]}...")
    print(f"• Hardware Device: {DEVICE}")
    print("=" * 65)

    job_id = f"gemma-bench-{int(time.time())}"
    job_name = "financial-sentiment-gemma"
    hyperparameters = {
        "epochs": 3,
        "batchSize": 2,
        "learningRate": 0.0002,
        "loraRank": 8,
        "loraAlpha": 32,
        "loraDropout": 0.05
    }

    t0 = time.time()
    try:
        result = train_job(
            job_id=job_id,
            job_name=job_name,
            base_model_name=base_model,
            dataset_path=rel_dataset,
            dataset_hash=dataset_hash,
            hyperparameters=hyperparameters
        )
    except Exception as ex:
        print(f"\n❌ Benchmark training run failed: {ex}")
        sys.exit(1)

    duration = time.time() - t0
    final_loss = result.get("finalLoss", 0.0)
    total_steps = result.get("totalSteps", 0)

    # Inspect adapter file size
    adapter_dir = ARTIFACTS_ROOT / job_id / "model_adapters"
    adapter_file = adapter_dir / "adapter_model.safetensors"
    if not adapter_file.exists():
        adapter_file = adapter_dir / "adapter_model.bin"
    adapter_size_mb = adapter_file.stat().st_size / (1024 * 1024) if adapter_file.exists() else 0.0

    print("\n" + "=" * 65)
    print("🎉 Benchmark Completed Successfully!")
    print("=" * 65)
    print(f"• Total Duration:     {duration:.2f} seconds ({duration / 60:.2f} mins)")
    print(f"• Final Training Loss:{final_loss:.4f}")
    print(f"• Total Steps:        {total_steps}")
    print(f"• Adapter File:       {adapter_file.name} ({adapter_size_mb:.2f} MB)")
    print(f"• MLflow Run ID:      {result.get('mlflowRunId')}")
    print("=" * 65)

    # Update GEMMA_PROTOTYPE_RESULTS.md
    doc_path = DEV_ROOT / "GEMMA_PROTOTYPE_RESULTS.md"
    if doc_path.exists():
        content = doc_path.read_text(encoding="utf-8")
        row = (
            f"| {time.strftime('%Y-%m-%d')} | `gemma-2-2b-it` | `financial-sentiment` | "
            f"{len(records)} | 3 | 8 / 32 | {DEVICE} | {final_loss:.4f} | "
            f"{int(duration // 60)}m {int(duration % 60)}s | Live Benchmark Run (Adapter: {adapter_size_mb:.2f} MB) |"
        )
        pattern = r"\| (?:[0-9]{4}-[0-9]{2}-[0-9]{2}|\*Prototype\*) \| `gemma-2-2b-it` \|.*"
        if re.search(pattern, content):
            content = re.sub(pattern, row, content)
        else:
            content += f"\n{row}\n"
        doc_path.write_text(content, encoding="utf-8")
        print(f"✓ Updated {doc_path.name} with live benchmark metrics!")

if __name__ == "__main__":
    main()
