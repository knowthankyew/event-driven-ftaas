# Howso Containerized Microservice (FTaaS Data Arbiter)

> **License**: GNU Affero General Public License v3 (AGPL-3.0)  
> **Container Boundary**: Isolated execution environment. Zero in-process linking with `event-driven-ftaas` (MIT).

## Overview
This service provides containerized, information-theoretic data processing based on the foundational research in *"A Theory of the Mechanics of Information: Generalization Through Measurement of Uncertainty (Learning is Measuring)"* by Christopher J. Hazard et al. (Howso Incorporated).

## Capabilities
1. **Redundancy Pruning / Data Ablation (Section 8.1)**: Identifies near-duplicate Q&A pairs (surprisal below $\epsilon$) and merges their probability mass, reducing PyTorch training dataset sizes without losing informational density.
2. **Anomaly & Poisoning Detection (Section 9.4)**: Detects out-of-distribution or adversarial prompts by measuring neighborhood surprisal contours.
3. **Attribution & Compliance Verification (Equations 16–17)**: Measures information distance between generated LLM responses and corporate policy cases using Lukaszyk–Karmowski expected difference and substitutability deviations.

## Transport & Isolation
To preserve the MIT License of the primary `event-driven-ftaas` platform, this service runs as an independent network microservice. All communication occurs strictly over HTTP/REST endpoints with JSON serialization. No AGPL-licensed code is linked in-process into the FTaaS .NET API or PyTorch Worker runtimes.
