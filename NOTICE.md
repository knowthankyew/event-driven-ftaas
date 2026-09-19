# Legal, Privacy, and Third-Party Attribution Notice

**Event-Driven FTaaS**  
Copyright © 2026 knowthankyew / The Event-Driven FTaaS Contributors  
Repository: [https://github.com/knowthankyew/event-driven-ftaas](https://github.com/knowthankyew/event-driven-ftaas)

---

## 1. Machine Learning Pipeline Notice
Event-Driven FTaaS provides distributed fine-tuning and inference services across .NET 10 APIs and Python PyTorch workers.

## 2. Telemetry & Attribute Redaction Invariants
All lifecycle spans emitted across the API and worker layers enforce a strict allowlist (`SAFE_ALLOWLIST_KEYS`). Dataset samples, prompts, completions, and model weights are strictly forbidden from telemetry attributes.

## 3. Software Bill of Materials (SBOM)
Complete CycloneDX Software Bills of Materials (`bom.json`) for both .NET dependencies and Python virtual environments are generated on build in CI via OWASP `@cyclonedx/cdxgen` and attached to release artifacts.

See [LICENSE](./LICENSE) for root license terms.
