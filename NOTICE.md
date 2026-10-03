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

## 4. Third-Party Foundation Models Attribution
This platform enables fine-tuning and inference orchestration with third-party foundation models:
- **`HuggingFaceTB/SmolLM2-135M`**: Licensed under the Apache License 2.0 by Hugging Face, Inc.
- **`google/gemma-2-2b-it`**: Developed and published by Google LLC. Available under the Google Gemma Terms of Use. Users and operators must accept Google's terms and adhere to the Gemma Prohibited Use Policy prior to downloading or fine-tuning weights.
- **`microsoft/BitNet-b1.58-2B-4T`**: Developed and published by Microsoft Corporation. Licensed under the MIT License. Available via Microsoft BitNet / Hugging Face.

See [LICENSE](./LICENSE) for root license terms.

