# FTaaS: Event-Driven "Fine-Tuning as a Service"

An enterprise-grade, asynchronous, event-driven machine learning platform demonstrating a polyglot microservice pattern:
- **Control Plane & Ingestion Gateway**: .NET 10 Minimal API / AMQP Producer
- **Compute Worker**: Python 3.12, PyTorch, Hugging Face PEFT / LoRA, AMQP Consumer
- **Telemetry & Experiment Tracking**: MLflow Tracking Server & Model Registry
- **Message Broker**: RabbitMQ (AMQP) with DLX and retry handling
- **Base Models**: Ultra-compact models (`HuggingFaceTB/SmolLM2-135M` or `TinyLlama-1.1B`) for zero-cost, fast local execution on consumer hardware (Apple Silicon MPS / CPU / CUDA).

---

## Architecture at a Glance

```plaintext
[Client / API Trigger]
        │
        ▼ (POST /api/v1/jobs multipart or datasetUri)
 [Ingestion Gateway (.NET 10)] ──► Validates JSONL, Computes SHA-256 Hash, Stores on Disk
        │
        ▼ (Job Request Event with datasetUri & datasetHash)
 [Async Queue / Message Broker (RabbitMQ)]
        │
        ▼
 [Worker / Orchestrator (Python)] ──► Spin up Training Compute (PyTorch / Hugging Face LoRA)
        │                                        │
        ▼                                        ▼
 [Model Registry (MLflow)]             [Metrics & Loss Telemetry]
        │
        ▼
 [Dynamic Inference Engine] ◄──────── Base Model + On-the-fly LoRA Adapter Mounting
        │
        ▼
 [Side-by-Side Comparison] ────────── Base Completion vs Fine-Tuned Completion
```

---

## Why This Architecture? (Engineering Trade-Offs)

| Architectural Decision | Trade-Off Rationale | Enterprise Reality |
|---|---|---|
| **Polyglot (.NET 10 + Python)** | .NET offers high-concurrency, strongly-typed contracts, and low-latency API handling; Python owns the cutting-edge ML ecosystem (PyTorch, PEFT). | Solves the common anti-pattern of writing web APIs in Python or trying to run ML training in C#. Leverages each ecosystem's primary strength. |
| **Out-of-Band Dataset Storage** | Storing `.jsonl` files on disk/object storage and passing only `datasetUri` + SHA-256 hash over RabbitMQ prevents broker bloat and memory pressure. | Real datasets are megabytes to gigabytes. Message brokers degrade rapidly when payloads exceed hundreds of kilobytes. |
| **LoRA (PEFT) vs Full Fine-Tuning** | Freezing 99%+ of base model weights and only training low-rank adapter matrices reduces VRAM requirements by >80% and completes in 2–4 minutes on consumer hardware. | In production, training 100 domain adapters on a single base model requires ~50MB per adapter instead of storing 100 full 7GB weights. |
| **Asynchronous Decoupling** | The client receives an immediate `202 Accepted` with a `JobId`. Heavy compute executes out-of-process. | Synchronous model training over HTTP guarantees timeouts, socket exhaustion, and cascading failures under load. |
| **Side-by-Side Inference Gateway** | Mounting adapters dynamically onto a shared base model allows rapid A/B testing and direct before/after domain comparison. | Avoids spinning up dedicated GPU containers for every custom-trained model variant. |

---

## Local Quickstart Workflow

```bash
# 1. Bring up Infrastructure (RabbitMQ + MLflow) and seed sample datasets
./scripts/dev-up.sh

# 2. Run the Ingestion API (.NET 10)
cd src/FtaaSService.Api && dotnet run

# 3. In another terminal, run the Training Worker (Python)
cd src/FtaaSService.Worker && source .venv/bin/activate && python consumer.py

# 4. In another terminal, run the Inference Engine
cd src/FtaaSService.Inference && source .venv/bin/activate && python app.py

# 5. Submit a fine-tuning job & inspect live metrics
curl -X POST http://localhost:5100/api/v1/jobs \
  -F "file=@data/datasets/sample-financial-sentiment.jsonl" \
  -F "jobName=sentiment-v1"
```

---

## Roadmap & Implementation Status

- [x] **Architecture Specification & Design Contracts** ([ARCHITECTURE.md](ARCHITECTURE.md))
- [x] **Phase 1**: Local Infrastructure Foundation (`docker-compose.yml`, `scripts/dev-up.sh`, RabbitMQ + MLflow healthchecks)
- [ ] **Phase 2**: Ingestion & Control Plane (.NET 10 Minimal API, JSONL validation, SQLite state machine, AMQP producer/consumer)
- [ ] **Phase 3**: Python Compute Worker & LoRA Pipeline (AMQP consumer, PyTorch MPS/CPU detection, PEFT/LoRA, MLflow telemetry)
- [ ] **Phase 4**: Dynamic Model Serving & Side-by-Side Comparison (LoRA dynamic adapter mounting, comparison API)
- [ ] **Phase 5**: Evaluation, Governance & Portfolio Polish (Benchmark test splits, MLflow Model Registry, verification script)
