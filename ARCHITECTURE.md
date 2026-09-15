# Event-Driven "Fine-Tuning as a Service" (FTaaS)

An enterprise-grade, asynchronous, event-driven ML platform demonstrating a polyglot microservice pattern (.NET 10 and Python), decoupled compute orchestration, experiment telemetry (MLflow), parameter-efficient fine-tuning (PEFT/LoRA), and dynamic model serving.

---

## 1. System Architecture Overview

```mermaid
flowchart TD
    subgraph Client["Client Applications / Ingestion"]
        C[Client / CLI / API Caller] -->|POST /api/v1/jobs + JSONL Dataset| API[.NET 10 Ingestion Gateway]
        C -->|GET /api/v1/jobs/{id}| API
    end

    subgraph Broker["Messaging Layer (AMQP)"]
        API -->|Publish: 'finetune.requested'| RMQ[RabbitMQ Exchange / Queue]
        PW -.->|Publish: 'finetune.status'| RMQ
        RMQ -.->|Consume status events| API
    end

    subgraph State["Job Store & Experiment Tracking"]
        API <--> DB[(SQLite / PostgreSQL Job State)]
        PW <--> MLF[MLflow Tracking Server]
        MLF <--> ART[(Artifact Store / Adaptor Weights)]
    end

    subgraph Compute["Training Compute Worker (Python)"]
        RMQ -->|Consume: 'finetune.requested'| PW[Python Training Worker]
        PW -->|Download base & Apply LoRA| HF[SmolLM / TinyLlama Base Model]
        PW -->|Stream Loss & GPU/MPS Metrics| MLF
        PW -->|Register Model Checkpoint| MLF
    end

    subgraph Serving["Dynamic Model Inference"]
        API -->|POST /api/v1/generate| INF[Inference Engine]
        INF -->|Load Base Weights| HF
        INF -->|Mount LoRA Adaptor from Registry| ART
        C -->|Compare Base vs Fine-Tuned Output| API
    end
```

---

## 2. Core Architectural Principles

1. **Polyglot Microservices**:
   - **.NET 10 Ingestion Gateway**: High-throughput, strongly-typed REST API managing dataset upload validation, schema enforcement, job state persistence, and AMQP event publishing.
   - **Python 3.12 Training Worker**: Specialized ML engine focusing strictly on compute-heavy PyTorch / Hugging Face PEFT fine-tuning, telemetry streaming, and artifact persistence.
2. **Decoupled Job Lifecycle**:
   - Web requests return immediately with an `Accepted (202)` and a unique `JobId`.
   - Training compute scales independently from API traffic, eliminating web timeouts.
3. **Hardware-Adaptive & Local-First (Zero Cloud Cost)**:
   - Automated compute device detection (`mps` on Apple Silicon, `cuda` on Nvidia, or multi-threaded `cpu`).
   - Compact baseline models (e.g., `HuggingFaceTB/SmolLM2-135M` or `TinyLlama-1.1B`) fine-tuned via LoRA in 2–4 minutes locally.
4. **Observable MLOps**:
   - Real-time training metrics (loss curves, learning rate, perplexity, epoch step) streamed into MLflow.
   - Traceable lineage connecting dataset hash, hyperparameters, git commit, and adapter artifacts.

---

## 3. Data Contracts & Event Schemas

### A. Job Submission Payload (`POST /api/v1/jobs`)
```json
{
  "jobName": "financial-sentiment-smollm",
  "baseModel": "HuggingFaceTB/SmolLM2-135M",
  "hyperparameters": {
    "epochs": 3,
    "batchSize": 4,
    "learningRate": 0.0002,
    "loraRank": 8,
    "loraAlpha": 32,
    "loraDropout": 0.05
  },
  "dataset": [
    {"prompt": "Classify earnings report: Revenue up 14% YoY.", "completion": "Positive"},
    {"prompt": "Classify earnings report: Supply chain friction dampens guidance.", "completion": "Negative"}
  ]
}
```

### B. Broker Message Contract (`finetune.job.requested`)
```json
{
  "jobId": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "jobName": "financial-sentiment-smollm",
  "baseModel": "HuggingFaceTB/SmolLM2-135M",
  "datasetUri": "file:///workspace/data/f47ac10b.jsonl",
  "hyperparameters": {
    "epochs": 3,
    "batchSize": 4,
    "learningRate": 0.0002,
    "loraRank": 8,
    "loraAlpha": 32,
    "loraDropout": 0.05
  },
  "submittedAt": "2026-09-14T20:00:00Z"
}
```

### C. Status Event Contract (`finetune.job.updated`)
```json
{
  "jobId": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
  "status": "Running | Succeeded | Failed",
  "currentStep": 45,
  "totalSteps": 100,
  "currentLoss": 0.321,
  "mlflowRunId": "9b1deb4d3b7d4bab8a7f",
  "artifactUri": "mlflow-artifacts:/1/9b1deb4d3b7d4bab8a7f/artifacts/model_adapters",
  "updatedAt": "2026-09-14T20:03:15Z"
}
```

---

## 4. Phased Implementation Roadmap

The implementation is structured into 6 sequential phases, each building upon verified deliverables:

### Phase 1: Local Infrastructure Foundation
- `docker-compose.yml` defining:
  - **RabbitMQ**: AMQP message broker with management UI (port 5672 / 15672).
  - **MLflow Server**: Tracking server with SQLite backend and local artifact mount (port 5000).
- Directory structure scaffolding and `.gitignore` setup for large models and artifacts.

### Phase 2: Ingestion & Control Plane (.NET 10)
- Minimal API project built with modern C# (.NET 10).
- Domain models and JSONL dataset validation rules.
- AMQP RabbitMQ Client integration (`RabbitMQ.Client` v7).
- Persistent Job Repository (Lightweight SQLite via EF Core / Dapper) tracking lifecycle states (`Pending`, `Queued`, `Training`, `Completed`, `Failed`).
- REST endpoints:
  - `POST /api/v1/jobs` (Submit job + dataset)
  - `GET /api/v1/jobs` (List all jobs)
  - `GET /api/v1/jobs/{id}` (Get job status, metrics, and MLflow metadata)

### Phase 3: Python Compute Worker & LoRA Pipeline
- Python virtual environment with Hugging Face stack (`transformers`, `peft`, `accelerate`, `torch`, `mlflow`, `pika`).
- Resilient AMQP consumer with message acknowledgement and error handling.
- Training pipeline:
  - Dataset tokenization and preparation.
  - Model loading and LoRA adapter configuration (`LoraConfig`).
  - Native Hugging Face `Trainer` integration logging step-level loss and eval metrics to MLflow.
  - Device acceleration detection (Metal `mps` on macOS, `cuda`, or `cpu`).
  - Adapter weight export and registration.

### Phase 4: Model Evaluation & Lineage Governance
- Automated evaluation step executed immediately post-training on a held-out test split.
- Telemetry logging of baseline vs fine-tuned benchmark metrics (e.g. cross-entropy loss, exact match / classification accuracy).
- Model artifact registration in MLflow Model Registry with versioning and parameter tags.

### Phase 5: Inference Gateway & Output Comparison
- Inference service exposing prompt completion:
  - Ability to evaluate completions using either the **Base Model** or with a specified **Fine-Tuned Adapter** mounted on the fly.
  - API endpoint (`POST /api/v1/inference/compare` or `/generate`) demonstrating domain adaptation (e.g. showing raw base model gibberish/generic output vs domain-accurate formatted output).

### Phase 6: End-to-End Orchestration, Docs & Verification
- Sample synthetic domain dataset (e.g., Domain Report Extraction or Structured Financial Analysis).
- Automated end-to-end verification script:
  1. Submits fine-tuning job via .NET 10 API.
  2. Verifies message consumption and training execution.
  3. Verifies MLflow experiment run & artifacts.
  4. Queries inference endpoint with benchmark prompt to prove fine-tuning efficacy.
- Portfolio documentation: architecture diagrams, setup runbook, FinOps cloud-readiness notes (how to map to AWS SQS / SageMaker or GCP Vertex AI).

---

## 5. Workspace Directory Structure

```plaintext
/Users/cl0rkster/Dev/ml/
├── docker-compose.yml            # RabbitMQ + MLflow server
├── README.md                     # Portfolio documentation & architectural deep dive
├── ARCHITECTURE.md               # Detailed system architecture and specifications
├── scripts/                      # Verification and runbook scripts
│   └── seed_dataset.py
├── src/
│   ├── FtaaSService.Api/         # .NET 10 Ingestion Gateway & Control Plane
│   │   ├── Controllers/ or Endpoints/
│   │   ├── Domain/
│   │   ├── Messaging/
│   │   ├── Storage/
│   │   ├── FtaaSService.Api.csproj
│   │   └── Program.cs
│   ├── FtaaSService.Worker/      # Python Training Worker
│   │   ├── config.py
│   │   ├── consumer.py           # RabbitMQ event consumer
│   │   ├── trainer.py            # PEFT / LoRA Hugging Face training engine
│   │   ├── evaluator.py          # Benchmark evaluation
│   │   └── requirements.txt
│   └── FtaaSService.Inference/   # Python / Minimal Serving Endpoint
│       ├── app.py
│       └── requirements.txt
└── data/                         # Local datasets and scratch storage
```
