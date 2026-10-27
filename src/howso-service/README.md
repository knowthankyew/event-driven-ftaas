# Howso Containerized Microservice (FTaaS Data Arbiter)

> **License**: GNU Affero General Public License v3.0 (AGPL-3.0) — see [`LICENSE-AGPL.txt`](./LICENSE-AGPL.txt)  
> **Authorship**: Original clean-room Python implementation of published equations from Hazard et al. (*arXiv:2505.00859*). Zero upstream Howso/Amalgam source code or SDK packages are included.  
> **Container Boundary**: Strictly isolated out-of-process HTTP/REST execution environment (port `5200`). Zero in-process linking or importing with `event-driven-ftaas` (MIT).

## Overview
This service provides containerized, information-theoretic dataset auditing and completion provenance based on the mathematical derivations published in *"A Theory of the Mechanics of Information: Generalization Through Measurement of Uncertainty (Learning is Measuring)"* by Dr. Christopher J. Hazard et al. (*arXiv:2505.00859*).

## Clean-Room Equation-to-Code Mapping
Every mathematical routine in [`service.py`](./service.py) is written from scratch using the Python standard library (`math`, `re`, `statistics`):

| Preprint Reference | Mathematical Concept | Implementation in `service.py` |
| :--- | :--- | :--- |
| **Equation 4** | **Laplace–Łukaszyk–Karmowski Distance**: Expected difference under Laplace measurement uncertainty: $D_{LK}(d, \delta) = d + \frac{1}{2} e^{-d/\delta}(3\delta + d)$ | `lukaszyk_karmowski_diff()` |
| **Equation 6** | **Continuous Marginal Surprisal**: Surprisal in nats with self-distance subtracted: $I(x_1, x_2) = \frac{D_{LK}(d, \delta)}{\delta} - 1.5$ | `marginal_surprisal_cont()` |
| **Equation 7** | **Nominal / Tag Surprisal**: Surprisal across categorical tags | `marginal_surprisal_nom()` |
| **Equation 8** | **Joint Case Surprisal**: Total joint information distance across feature dimensions | `case_surprisal()` |
| **Equation 10** | **Case Influence Probability**: Relative influence weight derived from marginal surprisal ($P_i \propto e^{-I_i}$, stabilized in log-space) | `verify_completion()` |
| **Section 8.1** | **Redundancy Ablation & Probability Mass Transfer**: Case pruning where mutual surprisal $\Delta I < \epsilon$ with weight accumulation (`weights[target] += weights[victim]`) propagated to JSONL and applied to per-sample causal LM loss in `trainer.py` | `audit_dataset()` |
| **Section 9.4** | **Anomaly & Poisoning Detection**: $k$-NN neighborhood surprisal contours | `audit_dataset()` |

## Capabilities
1. **Redundancy Pruning & Probability Mass Transfer (Section 8.1)**: Identifies near-duplicate Q&A pairs (mutual surprisal below $\epsilon$) and merges their probability mass weight (`weight`) into the surviving record. The `.NET` ingestion gateway persists non-unit `weight` values into the normalized `.jsonl` dataset, and the PyTorch worker (`trainer.py`) scales per-sample token cross-entropy loss by each record's probability mass weight.
2. **Anomaly & Poisoning Detection (Section 9.4)**: Detects out-of-distribution or adversarial prompts by measuring $k$-NN neighborhood surprisal contours.
3. **Attribution & Compliance Verification (Equations 8 & 10)**: Measures information distance between generated LLM responses and reference policy cases using Łukaszyk–Karmowski expected difference and empirical substitutability deviations ($\delta$).

## Double-Layer Ring-Fence & Trademark Notice
1. **Layer 1 (Clean-Room Copyright Independence)**: [`service.py`](./service.py) is an original clean-room implementation of published mathematical formulas. [`requirements.txt`](./requirements.txt) depends solely on `fastapi`, `uvicorn`, and `pydantic`.
2. **Layer 2 (Defensive AGPL-3.0 Process Isolation)**: To preserve the MIT License of the primary `event-driven-ftaas` platform regardless of third-party license interpretations, this service runs as an independent network microservice over HTTP/REST with JSON serialization, and its test suite (`tests/test_howso_service.py`) executes against an isolated subprocess server without importing `service.py` into the MIT test runner process.
3. **Trademark Disclaimer**: *Howso®* and *Amalgam™* are trademarks or registered trademarks of Howso Incorporated. This clean-room reference implementation is not affiliated with, endorsed by, or sponsored by Howso Incorporated.
