# FTaaS: Event-Driven "Fine-Tuning as a Service"

An enterprise-grade, asynchronous, event-driven machine learning platform demonstrating a polyglot microservices pattern:
- **Control Plane & Ingestion Gateway**: .NET 10 Minimal API / AMQP Producer
- **Compute Worker**: Python 3.12, PyTorch, Hugging Face PEFT / LoRA, AMQP Consumer
- **Telemetry & Experiment Tracking**: MLflow Tracking Server & Model Registry
- **Message Broker**: RabbitMQ (AMQP)
- **Base Models**: Ultra-compact models (`HuggingFaceTB/SmolLM2-135M` or `TinyLlama-1.1B`) for zero-cost, fast local execution on consumer hardware (Apple Silicon MPS / CPU / CUDA).

---

## Architecture at a Glance

```plaintext
[Client / API Trigger]
        │
        ▼ (Job Request Payload)
 [Async Queue / Message Broker (RabbitMQ)]
        │
        ▼
 [Worker / Orchestrator (Python)] ──► Spin up Training Compute (PyTorch / Hugging Face LoRA)
        │                                        │
        ▼                                        ▼
 [Model Registry (MLflow)]             [Metrics & Loss Telemetry]
        │
        ▼
 [Containerized Inference API] ◄────── Base Model + Dynamic LoRA Adapter Mounting
```

---

## Detailed Documentation & Roadmap

For complete system design, message contracts, and the phased implementation plan, see:
- [Architecture & Phased Implementation Plan](ARCHITECTURE.md)

---

## Status

- [x] Repository Initialized & Project Scaffolding
- [ ] Phase 1: Local Infrastructure Foundation (Docker Compose: RabbitMQ + MLflow)
- [ ] Phase 2: Ingestion & Control Plane (.NET 10)
- [ ] Phase 3: Python Compute Worker & LoRA Pipeline
- [ ] Phase 4: Model Evaluation & Lineage Governance
- [ ] Phase 5: Dynamic Inference Gateway & Comparison
- [ ] Phase 6: End-to-End Orchestration, Verification & Portfolio Polish
