# FTaaS: Event-Driven "Fine-Tuning as a Service"

An enterprise-grade, asynchronous, event-driven machine learning platform demonstrating a polyglot microservice pattern:
- **Control Plane & Ingestion Gateway**: .NET 10 Minimal API / AMQP Producer
- **Compute Worker**: Python 3.12, PyTorch, Hugging Face PEFT / LoRA, AMQP Consumer
- **Telemetry & Experiment Tracking**: MLflow Tracking Server & Model Registry
- **Message Broker**: RabbitMQ (AMQP) with DLX and retry handling
- **Base Models**: Ultra-compact models (`HuggingFaceTB/SmolLM2-135M` or `TinyLlama-1.1B`) for zero-cost, fast local execution on consumer hardware (Apple Silicon MPS / CPU / CUDA).

---

## Quickstart for Non-Technical Users & Business Leaders

> [!TIP]
> **Zero-Code Operation**: You do not need to know Python, PyTorch, Docker, or terminal commands to use FTaaS. 

Run one command to launch the full interactive web application:

```bash
./scripts/start-studio.sh
```

Then open your browser to **[http://localhost:5100](http://localhost:5100)**.

📖 **Looking for a non-technical walkthrough? Read the [FTaaS Enterprise Studio Guide](STUDIO_GUIDE.md) for step-by-step instructions, feature breakdowns, and FAQs with zero code jargon.**

### What You Can Do in the Studio:
1. **⚡ Side-by-Side Comparison Arena**: Select a business team persona (e.g. *Fintech Support & Compliance*, *Enterprise SaaS Ops*, or *Financial Earnings*) and run customer inquiries. See how the **Generic Foundation Model** (missing disclaimers, unadapted) contrasts with your **Company Custom AI** (which adheres to internal SLAs, Reg CC limits, and mandatory FDIC/SEC disclaimers).
2. **🛡️ Compliance & Policy Inspector**: An automated checklist auditing whether required disclosures, statutory exemptions, and department tags are included in model completions.
3. **🛠️ No-Code Adapter Studio**: Enter your team's common questions and preferred compliant answers in an interactive spreadsheet-like grid, or drag and drop existing company CSV/JSONL documents. Click **"Train & Deploy Team Adapter"** to trigger background training without writing code.
4. **🔄 Live Event Pipeline Visualizer**: A visual, animated walkthrough of the event-driven workflow showing how incoming requests are decoupled from heavy ML compute.
5. **📚 Adapter Library**: Browse company adapters, check their footprint (~1.8 MB), and load them directly into the comparison arena with one click.

---

## Architecture at a Glance

```plaintext
[Non-Tech Web Studio / Client]
        │
        ▼ (POST /api/v1/jobs multipart or datasetPath)
 [Ingestion Gateway (.NET 10)] ──► Validates JSONL, Computes SHA-256 Hash, Stores on Disk
        │
        ▼ (Job Request Event with datasetPath & datasetHash)
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
| **Out-of-Band Dataset Storage** | Storing `.jsonl` files on disk/object storage and passing only `datasetPath` + SHA-256 hash over RabbitMQ prevents broker bloat and memory pressure. | Real datasets are megabytes to gigabytes. Message brokers degrade rapidly when payloads exceed hundreds of kilobytes. |
| **LoRA (PEFT) vs Full Fine-Tuning** | Freezing 99%+ of base model weights and only training low-rank adapter matrices reduces VRAM requirements by >80% and completes in 2–4 minutes on consumer hardware. | In production, training 100 domain adapters on a single base model requires ~50MB per adapter instead of storing 100 full 7GB weights. |
| **Asynchronous Decoupling** | The client receives an immediate `202 Accepted` with a `JobId`. Heavy compute executes out-of-process. | Synchronous model training over HTTP guarantees timeouts, socket exhaustion, and cascading failures under load. |
| **Side-by-Side Inference Gateway** | Mounting adapters dynamically onto a shared base model allows rapid A/B testing and direct before/after domain comparison. | Avoids spinning up dedicated GPU containers for every custom-trained model variant. |

---

## Developer Quickstart Workflow (Command Line & APIs)

For engineers wanting to run each microservice manually:

```bash
# 1. Bring up Infrastructure (RabbitMQ + MLflow) and seed sample datasets
./scripts/dev-up.sh

# 2. Run the Ingestion API & Web Studio (.NET 10)
cd src/FtaaSService.Api && dotnet run

# 3. In another terminal, run the Training Worker (Python)
cd src/FtaaSService.Worker && source .venv/bin/activate && python consumer.py

# 4. In another terminal, run the Inference Engine
cd src/FtaaSService.Inference && source .venv/bin/activate && python app.py

# 5. Submit a fine-tuning job via curl
curl -X POST http://localhost:5100/api/v1/jobs \
  -F "file=@data/datasets/sample-support-compliance.jsonl" \
  -F "jobName=compliance-v1"
```

---

## Automated End-to-End Verification
To test the entire pipeline (Infra $\rightarrow$ .NET 10 Ingestion $\rightarrow$ RabbitMQ $\rightarrow$ PyTorch/PEFT Training on MPS $\rightarrow$ MLflow $\rightarrow$ Dynamic Inference Comparison) in one command:

```bash
./scripts/verify-e2e.sh
```

---

## Roadmap & Implementation Status

- [x] **Architecture Specification & Design Contracts** ([ARCHITECTURE.md](ARCHITECTURE.md))
- [x] **Phase 1**: Local Infrastructure Foundation (`docker-compose.yml`, `scripts/dev-up.sh`, RabbitMQ + MLflow healthchecks)
- [x] **Phase 2**: Ingestion & Control Plane (.NET 10 Minimal API, JSONL validation, SQLite state machine, AMQP producer/consumer)
- [x] **Phase 3**: Python Compute Worker & LoRA Pipeline (AMQP consumer, PyTorch MPS/CPU detection, PEFT/LoRA, MLflow telemetry)
- [x] **Phase 4**: Dynamic Model Serving & Side-by-Side Comparison (LoRA dynamic adapter mounting, comparison API)
- [x] **Phase 5**: Evaluation, Governance & Portfolio Polish (Benchmark test splits, MLflow Model Registry, verification script)

---

## Production Cloud Parity (AWS & GCP)

| Local Stack | AWS Production Architecture | GCP Production Architecture |
|---|---|---|
| **.NET 10 API Gateway** | Amazon API Gateway + ECS Fargate | Google Cloud Run (.NET 10) |
| **RabbitMQ Broker + DLQ** | Amazon SQS (FIFO) + SQS DLQ | Cloud Pub/Sub + Dead Letter |
| **Storage (Datasets & DB)** | Amazon S3 + Aurora PostgreSQL | Google Cloud Storage + Cloud SQL |
| **Compute Worker (Python)** | SageMaker Training Jobs (Spot Instances) | Vertex AI Custom Jobs (Preemptible) |
| **Experiment Telemetry** | Managed MLflow / SageMaker Experiments | Vertex AI Experiments |
| **Model Registry** | MLflow Model Registry / SageMaker Registry | Vertex AI Model Registry |
| **Dynamic Inference Server**| SageMaker Multi-Model Endpoints / Triton | Vertex AI Endpoints (vLLM / Triton) |

---

## License

This project is licensed under the [MIT License](LICENSE).

