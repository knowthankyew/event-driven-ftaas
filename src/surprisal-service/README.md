# Surprisal Containerized Microservice (FTaaS Data Arbiter)

> **License**: MIT License — see root [`LICENSE`](../../LICENSE)  
> **Authorship**: Independent Python implementation written from published equations of Hazard et al. (*arXiv:2510.22809*). Zero upstream Howso/Amalgam source code or SDK packages are included.  
> **Container Boundary**: Isolated out-of-process HTTP/REST execution environment (port `5200`).

## Overview
This service provides containerized, information-theoretic dataset auditing and completion provenance based on the mathematical derivations published in *"A Theory of the Mechanics of Information: Generalization Through Measurement of Uncertainty (Learning is Measuring)"* by Dr. Christopher J. Hazard et al. (*arXiv:2510.22809*).

## Equation-to-Code Mapping
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

## Independent Implementation, Fidelity & Trademark Notice
1. **Provenance**: [`service.py`](./service.py) is an independent implementation of published mathematical formulas. [`requirements.txt`](./requirements.txt) depends solely on `fastapi`, `uvicorn`, and `pydantic`. Zero Howso or Amalgam packages are installed or imported.
2. **Fidelity**: This is an adaptation, not a validated reproduction of the paper's methods. It adds practical heuristics (for example, token-overlap prefilters, keyword features, candidate caps, and fixed thresholds). Results should not be attributed to the paper's authors or to Howso.
3. **Trademark Disclaimer**: *Howso®* and *Amalgam™* are trademarks or registered trademarks of Howso Incorporated. This independent reference implementation is not affiliated with, endorsed by, or sponsored by Howso Incorporated.
