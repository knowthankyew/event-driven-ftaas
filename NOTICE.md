# Legal, Privacy, and Third-Party Attribution Notice

**Event-Driven FTaaS**  
Copyright © 2026 knowthankyew  
Repository: [https://github.com/knowthankyew/event-driven-ftaas](https://github.com/knowthankyew/event-driven-ftaas)

This repository is licensed in its entirety under the MIT License. See [LICENSE](./LICENSE).

---

## 1. Machine Learning Pipeline Notice
Event-Driven FTaaS provides distributed fine-tuning and inference services across .NET 10 APIs and Python PyTorch workers.

## 2. Telemetry & Attribute Redaction Invariants
All lifecycle spans emitted across the API and worker layers enforce a strict allowlist (`SAFE_ALLOWLIST_KEYS`). Dataset samples, prompts, completions, and model weights are strictly forbidden from telemetry attributes.

## 3. Software Bill of Materials (SBOM)
Complete CycloneDX Software Bills of Materials (`bom.json`) for both .NET dependencies and Python virtual environments are generated on build in CI via OWASP `@cyclonedx/cdxgen` and uploaded as workflow artifacts.

## 4. Third-Party Foundation Models Attribution
This platform enables fine-tuning and inference orchestration with third-party foundation models:
- **`HuggingFaceTB/SmolLM2-135M`**: Licensed under the Apache License 2.0 by Hugging Face, Inc.
- **`google/gemma-2-2b-it`**: Developed and published by Google LLC. Available under the Google Gemma Terms of Use. Users and operators must accept Google's terms and adhere to the Gemma Prohibited Use Policy prior to downloading or fine-tuning weights.
- **`microsoft/BitNet-b1.58-2B-4T`**: Developed and published by Microsoft Corporation. Licensed under the MIT License. Available via Microsoft BitNet / Hugging Face.

## 5. Optional Containerized Sidecar: `surprisal-service` (formerly `howso-service`)

`src/surprisal-service/` is an optional dataset-auditing microservice that runs as an out-of-process HTTP/REST service (port `5200`).

- **License**: MIT, the same as the rest of this repository. Revisions of this repository prior to the October 2026 MIT relicensing commit carried an AGPL-3.0 notice and a license file in this directory. Those were removed because the code is original work of this repository's author, who relicensed it under MIT. Copies obtained from earlier revisions remain available under the terms they were received with.
- **Provenance**: `service.py` is an independent implementation written from the published equations and sections of Christopher J. Hazard et al., *"A Theory of the Mechanics of Information: Generalization Through Measurement of Uncertainty (Learning is Measuring)"*, arXiv:2510.22809. No source code, functions, or files from Howso or Amalgam repositories are included, and `requirements.txt` installs only general-purpose web framework packages (`fastapi`, `uvicorn`, `pydantic`). The service imports no Howso or Amalgam packages or binaries.
- **Fidelity**: This is an adaptation, not a validated reproduction of the paper's methods. It adds practical heuristics that are not taken from the paper (for example, token-overlap prefilters, keyword features, candidate caps, and fixed thresholds). Results should not be attributed to the paper's authors or to Howso.
- **Academic citation**: References to Hazard et al. are citations to published research and imply no endorsement by the authors.
- **Trademarks**: Howso® and Amalgam™ are trademarks or registered trademarks of Howso Incorporated. They appear in this repository only to cite the underlying research and to describe this project's provenance. This project is independent and is not affiliated with, endorsed by, or sponsored by Howso Incorporated.
- **Patents**: The MIT License does not grant, and this repository does not claim to grant, any license under third-party patents. Anyone deploying this service commercially is responsible for their own patent and freedom-to-operate review.
