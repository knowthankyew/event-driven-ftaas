# FTaaS: Event-Driven "Fine-Tuning as a Service" (`event-driven-ftaas` / `ml`)

An enterprise-grade, asynchronous, event-driven machine learning platform demonstrating a polyglot microservice pattern:
- **Control Plane & Ingestion Gateway**: .NET 10 Minimal API / AMQP Producer
- **Compute Worker**: Python 3.12, PyTorch, Hugging Face PEFT / LoRA, AMQP Consumer
- **Telemetry & Experiment Tracking**: OpenTelemetry ActivitySource + MLflow Tracking Server & Model Registry
- **Message Broker**: RabbitMQ (AMQP) with DLX and retry handling
- **Base Models**: Ultra-compact models (`HuggingFaceTB/SmolLM2-135M` or `TinyLlama-1.1B`) for zero-cost, fast local execution on consumer hardware (Apple Silicon MPS / CPU / CUDA).
- **Privacy & Observability Standard**: Complies with [PRIVACY_TELEMETRY_SCHEMA.md](docs/PRIVACY_TELEMETRY_SCHEMA.md).

## 📺 Interactive Video Demonstration

Watch the complete end-to-end studio workflow in action—from side-by-side compliance policy comparison to live PII guardrails and event pipeline tracking:

https://github.com/user-attachments/assets/demo.mp4

> **Recorded Demonstration**: [demo.mp4](demo.mp4) *(High-definition Playwright automated recording)*
>
> To regenerate this demonstration at any time, run: `./scripts/record-demo.sh`

---

## Quickstart for Non-Technical Users & Business Leaders

> [!TIP]
> **Zero-Code Operation**: You do not need to write Python, manage PyTorch, or configure machine learning pipelines to evaluate FTaaS.
>
> **Prerequisite**: [.NET 10 SDK](https://dotnet.microsoft.com/download) installed on your machine.

Run one command to launch the full interactive web application:

```bash
./scripts/start-studio.sh
```

Then open your browser to **[http://localhost:5100](http://localhost:5100)**.

📖 **Looking for a guided walkthrough? Read the [FTaaS Enterprise Studio Guide](STUDIO_GUIDE.md) for step-by-step instructions, visual explanations, and FAQs.**

### What You Can Do in the Studio:
1. **⚡ Side-by-Side Comparison Arena**: Select a business persona (e.g. *Fintech Support & Compliance*, *Enterprise SaaS Ops*, or *Financial Earnings*) and test realistic inquiries. Observe how a **Generic Foundation Model** contrasts with your **Company Custom AI** (incorporating team SLAs, policy limits, and required regulatory disclaimers).
2. **🛡️ Team Policy & Disclaimer Inspector**: Pattern-matching verification checking whether designated training guidelines and statutory tags appear in completions *(demonstration aid; not legal advice or statutory regulatory certification)*.
3. **🛠️ No-Code Adapter Studio**: Define your department's question-and-answer pairs in an intuitive table editor, or drag-and-drop a `.csv` document. Click **"Train & Deploy Team Adapter"** to submit the job in milliseconds without waiting on background compute.
4. **🔄 Live Event Pipeline Visualizer**: An animated diagram demonstrating how incoming user requests decouple from background training compute.
5. **📚 Adapter Library**: View trained department models, inspect their lightweight footprint (~1.8 MB), and load them into the arena with one click.

> [!NOTE]
> **Preview vs. Live Compute**: When launched standalone via `start-studio.sh`, the studio operates in **Interactive Preview Mode** with pre-formatted demonstration outputs so you can evaluate the interface without spinning up Docker containers or downloading multi-gigabyte models. To run live on-device GPU inference, start Docker and the Python services via `./scripts/dev-up.sh`.

> [!CAUTION]
> **Data Privacy & Guardrail Scope**: Automated heuristic checks detect delimited SSNs and Luhn-valid credit card numbers across all columns (including unmapped metadata). However, this is a best-effort defense, not an exhaustive DLP certification tool. Always sanitize datasets prior to training.

---

## Privacy, Observability & Job Lifecycle Spans

FTaaS follows the portfolio governance standard documented in [PRIVACY_TELEMETRY_SCHEMA.md](docs/PRIVACY_TELEMETRY_SCHEMA.md):

- **Consumer Privacy Default:** Runs in `memory_only` mode with zero outbound telemetry egress.
- **Job Lifecycle Spans:** Traces the entire training pipeline: `job.accepted` &rarr; `job.published` &rarr; `job.consumed` &rarr; `job.training.started` &rarr; `job.training.finished` &rarr; `job.registered`.
- **Payload Redaction:** Attributes strictly log job IDs, statuses, durations, hardware devices (`cuda` | `mps` | `cpu`), and adapter byte sizes. No raw prompts, completions, or dataset contents are ever permitted in spans.
- **Inspection & Purge API:**
  - `GET /api/v1/telemetry/privacy-audit`: Real-time inspection of active telemetry mode and buffer state.
  - `GET /api/v1/telemetry/spans`: Inspect buffered in-memory spans.
  - `POST /api/v1/telemetry/burn`: Immediately purges the in-memory telemetry buffer.
- **Enterprise OTLP Overlay:** Setting `OTEL_EXPORTER_OTLP_ENDPOINT` exports spans to an enterprise collector while maintaining the strict payload scrubbing invariant.

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

## Automated Test Suites & Quality Engineering

The platform includes formal, automated unit and regression test suites across both the .NET control plane and Python compute layers:

### 1. .NET 10 API Test Suite (xUnit + Coverlet)
Validates compliance boundaries, data sanitization, and state machine idempotency:
- **PII Compliance Gateway**: Delimited SSA SSNs, contextual SSNs, and Luhn-valid payment cards.
- **False-Positive Immunity**: Verifies that non-contextual 9-digit integers (order IDs, invoice numbers) pass unhindered.
- **Full-Row Unmapped Column Scanning**: Scans all metadata columns/properties; strips unmapped fields upon normalization.
- **Zero-Disk In-Memory Guarantee**: Verifies zero bytes are written to disk upon compliance rejection.
- **Ingestion Ceilings**: Enforces 25 MB file size and 50,000 record limits.
- **State Machine Idempotency**: Rejects out-of-order sequence updates and prevents terminal state regressions in SQLite.

```bash
# Run all .NET unit tests with code coverage collection:
dotnet test tests/FtaaSService.Api.Tests --collect:"XPlat Code Coverage"
```

### 2. Python Worker & Inference Tests (unittest)
Validates model serving resilience and file integrity:
- **Adapter Weight Integrity**: Rejects truncated, zero-byte, or incomplete writes (<100 KB weights, <10 bytes config).
- **Bounded LRU Cache Eviction**: Verifies dynamic eviction of least-recently-used LoRA adapters under memory pressure.

```bash
# Run all Python unit tests:
src/FtaaSService.Worker/.venv/bin/python -m unittest discover -s tests
```

---

## Automated End-to-End Integration Verification
To test the entire live pipeline (Infra $\rightarrow$ .NET 10 Ingestion $\rightarrow$ RabbitMQ $\rightarrow$ PyTorch/PEFT Training on MPS $\rightarrow$ MLflow $\rightarrow$ Dynamic Inference Comparison) in one command:

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
- [x] **Portfolio Phase 3a (The FTaaS Bridge Exporter)**: Edge ONNX Exporter (`src/FtaaSService.Worker/exporter.py`, `scripts/export_edge_adapter.py`) and Web API export endpoints (`GET/POST /api/v1/jobs/{id}/export/edge`) compiling LoRA adapters into web-optimized ONNX format with integrity checksums.
- [x] **Portfolio Phase 3b (In-Browser Execution)**: Ingesting and executing exported ONNX packages directly inside client browser engines via WebGPU/WASM (`onnxruntime-web`), single-input ONNX export signature, and dynamic INT8 quantization.

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

