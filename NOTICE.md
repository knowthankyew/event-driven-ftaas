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
Complete CycloneDX Software Bills of Materials (`bom.json`) for both .NET dependencies and Python virtual environments are generated on build in CI via OWASP `@cyclonedx/cdxgen` and uploaded as workflow artifacts.

## 4. Third-Party Foundation Models Attribution
This platform enables fine-tuning and inference orchestration with third-party foundation models:
- **`HuggingFaceTB/SmolLM2-135M`**: Licensed under the Apache License 2.0 by Hugging Face, Inc.
- **`google/gemma-2-2b-it`**: Developed and published by Google LLC. Available under the Google Gemma Terms of Use. Users and operators must accept Google's terms and adhere to the Gemma Prohibited Use Policy prior to downloading or fine-tuning weights.
- **`microsoft/BitNet-b1.58-2B-4T`**: Developed and published by Microsoft Corporation. Licensed under the MIT License. Available via Microsoft BitNet / Hugging Face.

## 5. Optional Containerized Sidecars & Microservice Isolation Notice
- **`howso-service` (FTaaS Data Arbiter — Clean-Room Reference Implementation)**:
  - **Clean-Room Provenance & Academic Citation**: `src/howso-service/service.py` is an original, clean-room Python implementation written directly from the published mathematical derivations (Equations 4, 6, 7, 8, 10 and Sections 8.1, 9.4) in *"A Theory of the Mechanics of Information: Generalization Through Measurement of Uncertainty (Learning is Measuring)"* by Dr. Christopher J. Hazard et al. (*arXiv:2505.00859*). Zero lines of source code, functions, or files were copied or derived from upstream Howso or Amalgam repositories, and `src/howso-service/requirements.txt` installs only standard MIT/BSD web framework packages (`fastapi`, `uvicorn`, `pydantic`).
  - **Double-Layer Ring-Fence Architecture**:
    1. *Layer 1 (Clean-Room Copyright Independence)*: Original implementation of published mathematical equations using Python standard libraries (`math`, `re`, `statistics`).
    2. *Layer 2 (Defensive AGPL-3.0 Network Ring-Fence)*: The `src/howso-service/` microservice directory is defensively licensed under the GNU Affero General Public License v3.0 (`src/howso-service/LICENSE-AGPL.txt`) and executes strictly as an isolated out-of-process HTTP/REST service (port `5200`), including subprocess-isolated integration testing. No AGPL-3.0 source or binary dependencies are linked or imported in-process into the MIT-licensed `.NET` API, PyTorch Worker, Inference service, or root test runner.
  - **Trademark & Non-Affiliation Disclaimer**: *Howso®* and *Amalgam™* are trademarks or registered trademarks of Howso Incorporated. References to Howso and Hazard et al. in this repository are strictly nominative citations to the underlying academic literature. This project is an independent clean-room implementation and is not affiliated with, endorsed by, or sponsored by Howso Incorporated.
  - **Patent & Commercial Deployment Advisory**: Mathematical equations in academic literature are not subject to copyright under 17 U.S.C. § 102(b), but commercial implementations of information-theoretic instance-based learning algorithms may be subject to third-party utility patent claims held by Howso Incorporated or governed by AGPL-3.0 Section 11/13 terms. Downstream commercial operators should consult IP counsel regarding patent licensing or AGPL-3.0 Section 13 network source-disclosure obligations when deploying `howso-service`.

See [LICENSE](./LICENSE) for root MIT license terms and the `src/howso-service/` subdirectory exception.

