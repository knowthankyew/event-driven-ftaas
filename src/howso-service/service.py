"""
FTaaS Data Arbiter — Independent Information-Theoretic Microservice
==================================================================
Copyright (C) 2026 knowthankyew
SPDX-License-Identifier: MIT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

Provenance & Academic Citation:
------------------------------------------
This module is an independent Python implementation written from the published
equations and sections (Equations 4, 6, 7, 8, 10 and Sections 8.1, 9.4) in:
  Christopher J. Hazard et al., "A Theory of the Mechanics of Information:
  Generalization Through Measurement of Uncertainty (Learning is Measuring)",
  arXiv:2510.22809.

Zero source code, functions, or files from Howso or Amalgam repositories are
included.

Fidelity & Heuristics Disclaimer:
---------------------------------
This is an adaptation, not a validated reproduction of the paper's methods.
It adds practical heuristics (for example, token-overlap prefilters, keyword
features, candidate caps, and fixed thresholds). Results should not be
attributed to the paper's authors or to Howso.

Trademark & Non-Affiliation Disclaimer:
---------------------------------------
Howso(R) and Amalgam(TM) are trademarks or registered trademarks of Howso
Incorporated. This independent reference implementation is not affiliated with,
endorsed by, or sponsored by Howso Incorporated.

Endpoints:
- GET  /health
- POST /v1/audit/dataset
- POST /v1/verify/completion
"""

import math
import re
import statistics
from typing import Any, Dict, List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Implementation flag: zero upstream Howso/Amalgam packages are imported or linked.
NATIVE_HOWSO_AVAILABLE = False

ENGINE_IDENTIFIER = "Independent Reference Information Engine (Hazard et al. Laplace-LK Surprisal)"

app = FastAPI(
    title="FTaaS Howso Data Arbiter",
    description="Containerized Independent Information-Theoretic Dataset Auditing & Provenance Service",
    version="1.0.0"
)

# -------------------------------------------------------------------
# DOMAIN MODELS
# -------------------------------------------------------------------

class DatasetRecord(BaseModel):
    prompt: str = Field(..., max_length=10000, description="Instruction or input text")
    completion: str = Field(..., max_length=10000, description="Expected response text")
    weight: float = Field(1.0, ge=0.0, description="Probability mass weight of this record")

class AuditRequest(BaseModel):
    records: List[DatasetRecord] = Field(..., max_length=5000, description="Records to audit and ablate per batch")
    ablation_threshold: float = Field(0.15, description="Marginal surprisal in nats below which cases are ablated")
    anomaly_threshold: float = Field(5.0, description="k-NN surprisal in nats above which cases are flagged as anomalous")

class AuditSummary(BaseModel):
    original_count: int
    cleaned_count: int
    ablated_count: int
    anomalous_count: int
    measured_deviations: Dict[str, float]
    anomalous_records: List[Dict[str, Any]]
    ablated_records: List[Dict[str, Any]]

class AuditResponse(BaseModel):
    summary: AuditSummary
    cleaned_records: List[DatasetRecord]

class VerifyRequest(BaseModel):
    reference_records: List[DatasetRecord] = Field(..., max_length=2000, description="Approved reference baseline records")
    prompt: str = Field(..., max_length=10000, description="Input query prompt")
    candidate_completion: str = Field(..., max_length=10000, description="Generated candidate completion to verify")
    max_tolerable_surprisal: float = Field(6.0, description="Surprisal ceiling in nats before flagging policy violation")

class InfluentialCase(BaseModel):
    case_index: int
    tag: str
    surprisal_nats: float
    influence_weight_percent: float
    source_prompt: str

class VerifyResponse(BaseModel):
    is_conforming: bool
    measured_surprisal_nats: float
    predicted_tag: str
    candidate_tag: str
    influential_cases: List[InfluentialCase]
    explanation: str

# -------------------------------------------------------------------
# HOWSO INFORMATION THEORY ALGORITHMS (Preprint Equations 4, 6, 8, 10, 17)
# -------------------------------------------------------------------

def token_jaccard_similarity(s1: Any, s2: Any) -> float:
    """Computes Jaccard word token similarity between two text strings or token sets."""
    tokens1 = s1 if isinstance(s1, set) else set(re.findall(r'\w+', str(s1).lower()))
    tokens2 = s2 if isinstance(s2, set) else set(re.findall(r'\w+', str(s2).lower()))
    if not tokens1 or not tokens2:
        return 0.0
    return len(tokens1 & tokens2) / len(tokens1 | tokens2)

def extract_features(prompt: str, completion: str) -> Dict[str, Any]:
    """Extracts prompt-side and completion-side continuous, token, and nominal features."""
    tag = "UNKNOWN"
    m = re.search(r'\[([A-Za-z0-9\-_]+)\]', completion)
    if m:
        tag = m.group(1)

    domain_keywords = [
        "transfer", "wire", "contract", "cancel", "outage", "sso", "refund", "charge",
        "gdpr", "hipaa", "phi", "api", "bonus", "atm", "baa", "portfolio", "card", "notice", "sla"
    ]
    prompt_lower = prompt.lower()
    completion_lower = completion.lower()

    feat: Dict[str, Any] = {
        "prompt_len": float(len(prompt)),
        "prompt_has_dollar": 1.0 if "$" in prompt else 0.0,
        "prompt_has_time": 1.0 if any(w in prompt_lower for w in ["day", "hour", "minute", "sla", "terms"]) else 0.0,
        "completion_len": float(len(completion)),
        "completion_has_dollar": 1.0 if "$" in completion else 0.0,
        "completion_has_time": 1.0 if any(w in completion_lower for w in ["day", "hour", "minute", "sla", "terms"]) else 0.0,
        "tag": tag,
        "prompt": prompt,
        "completion": completion,
        "_prompt_tokens": set(re.findall(r'\w+', prompt_lower)),
        "_completion_tokens": set(re.findall(r'\w+', completion_lower))
    }

    for kw in domain_keywords:
        feat[f"prompt_kw_{kw}"] = 1.0 if kw in prompt_lower else 0.0
        feat[f"completion_kw_{kw}"] = 1.0 if kw in completion_lower else 0.0

    return feat

def compute_deviations(cases: List[Dict[str, Any]]) -> Dict[str, float]:
    """Computes empirical Substitutability Deviations (δ) via Mean Absolute Deviation (MAD)."""
    if not cases:
        return {}
    cont_keys = [k for k in cases[0].keys() if (k.startswith("prompt_") or k.startswith("completion_")) and isinstance(cases[0][k], (int, float))]
    deviations = {}
    for k in cont_keys:
        vals = [c[k] for c in cases]
        m = statistics.mean(vals)
        mad = statistics.mean([abs(v - m) for v in vals])
        deviations[k] = max(mad, 0.25)
    return deviations

def lukaszyk_karmowski_diff(d: float, delta: float) -> float:
    """Eq 4: Lukaszyk-Karmowski expected distance under Laplace uncertainty."""
    return d + 0.5 * math.exp(-d / delta) * (3.0 * delta + d)

def marginal_surprisal_cont(x1: float, x2: float, delta: float) -> float:
    """Eq 6: Marginal Surprisal (nats) for continuous features with LK self-distance subtracted."""
    d = abs(x1 - x2)
    elk = lukaszyk_karmowski_diff(d, delta)
    surprisal = (elk / delta) - 1.5
    return max(0.0, surprisal)

def marginal_surprisal_nom(val1: str, val2: str) -> float:
    """Eq 7: Marginal Surprisal for nominal tags."""
    return 0.0 if val1 == val2 else 2.5

def case_surprisal(c1: Dict[str, Any], c2: Dict[str, Any], deviations: Dict[str, float], feature_keys: List[str]) -> float:
    """Eq 8: Total Joint Surprisal between two cases across feature set."""
    total = 0.0
    for k in feature_keys:
        if k in deviations:
            total += marginal_surprisal_cont(c1[k], c2[k], deviations[k])
        elif k == "tag":
            total += marginal_surprisal_nom(c1[k], c2[k])
    return total

# -------------------------------------------------------------------
# ENDPOINTS
# -------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "howso-data-arbiter",
        "native_howso_available": NATIVE_HOWSO_AVAILABLE,
        "engine": ENGINE_IDENTIFIER,
        "license": "MIT"
    }

@app.post("/v1/audit/dataset", response_model=AuditResponse)
def audit_dataset(req: AuditRequest):
    if not req.records:
        raise HTTPException(status_code=400, detail="Record list cannot be empty.")

    extracted = [extract_features(r.prompt, r.completion) for r in req.records]
    deviations = compute_deviations(extracted)

    # 1. Redundancy Ablation (Section 8.1)
    # Bucket by tag so pairwise comparisons only evaluate records within the same category
    weights = [r.weight for r in req.records]
    absorbed_counts = [0 for _ in req.records]
    pruned_indices = set()
    ablated_details = []
    input_keys = [k for k in deviations.keys()]

    tag_groups: Dict[str, List[int]] = {}
    for idx in range(len(extracted)):
        tag_groups.setdefault(extracted[idx]["tag"], []).append(idx)

    MAX_ABLATION_CANDIDATES = 150

    for tag, group in tag_groups.items():
        # Sort indices within tag group by prompt length to enable rapid candidate window pruning
        group.sort(key=lambda idx: extracted[idx]["prompt_len"])
        for i_pos in range(len(group)):
            i = group[i_pos]
            if i in pruned_indices:
                continue
            len_p_i = extracted[i]["prompt_len"]
            len_c_i = extracted[i]["completion_len"]
            max_p_diff = max(20, int(len_p_i * 0.40))
            max_c_diff = max(20, int(len_c_i * 0.40))
            p_tokens_i = extracted[i]["_prompt_tokens"]
            c_tokens_i = extracted[i]["_completion_tokens"]

            checked = 0
            for j_pos in range(i_pos + 1, len(group)):
                j = group[j_pos]
                if j in pruned_indices:
                    continue

                # Since group is sorted by prompt_len, if length delta exceeds candidate window, break early
                if extracted[j]["prompt_len"] - len_p_i > max_p_diff:
                    break

                # Fast completion length pre-filter before evaluating case_surprisal
                if abs(extracted[j]["completion_len"] - len_c_i) > max_c_diff:
                    continue

                checked += 1

                # Rapid textual pre-filter: cases can only be redundant if token overlap is >= 0.80
                sim_prompt = token_jaccard_similarity(p_tokens_i, extracted[j]["_prompt_tokens"])
                if sim_prompt < 0.80:
                    if checked >= MAX_ABLATION_CANDIDATES:
                        break
                    continue

                sim_comp = token_jaccard_similarity(c_tokens_i, extracted[j]["_completion_tokens"])
                if sim_comp < 0.80:
                    if checked >= MAX_ABLATION_CANDIDATES:
                        break
                    continue

                s_ij = case_surprisal(extracted[i], extracted[j], deviations, input_keys)
                # Both distance and textual overlap must confirm redundancy
                if s_ij < req.ablation_threshold:
                    target = min(i, j)
                    victim = max(i, j)
                    weights[target] += weights[victim]
                    absorbed_counts[target] += 1 + absorbed_counts[victim]
                    pruned_indices.add(victim)
                    ablated_details.append({
                        "ablated_index": victim,
                        "merged_into_index": target,
                        "surprisal_nats": round(s_ij, 5),
                        "prompt_sample": extracted[victim]["prompt"][:80]
                    })
                    if victim == i:
                        break

                if checked >= MAX_ABLATION_CANDIDATES:
                    break

    surviving_indices = [idx for idx in range(len(extracted)) if idx not in pruned_indices]

    # Index surviving records by tag
    tag_survivors: Dict[str, List[int]] = {}
    for idx in surviving_indices:
        tag_survivors.setdefault(extracted[idx]["tag"], []).append(idx)

    # 2. Anomaly Detection (Section 9.4)
    anomalous_details = []
    quarantined_indices = set()
    MAX_ANOMALY_CANDIDATES = 120
    structural_keys = [
        k for k in input_keys
        if not (k.startswith("prompt_kw_") or k.startswith("completion_kw_"))
    ]

    for idx in surviving_indices:
        c_i = extracted[idx]
        peers = tag_survivors.get(c_i["tag"], [])
        candidate_indices = [o for o in peers if o != idx]
        same_tag_neighborhood = len(candidate_indices) >= 3

        # If not enough peers in same tag, supplement from global pool using structural contours
        if not same_tag_neighborhood:
            candidate_indices = [o for o in surviving_indices if o != idx]

        # Bound pairwise calculations per record to MAX_ANOMALY_CANDIDATES
        if len(candidate_indices) > MAX_ANOMALY_CANDIDATES:
            step = len(candidate_indices) / MAX_ANOMALY_CANDIDATES
            candidate_indices = [candidate_indices[int(i * step)] for i in range(MAX_ANOMALY_CANDIDATES)]

        # Records that absorbed near-duplicate neighbors during Section 8.1 ablation
        # have corroborated peers at ~0 surprisal within their own tag neighborhood.
        # Use absorbed_counts (rather than client-supplied weights) capped at k=3.
        extra_self_neighbors = min(3, max(0, absorbed_counts[idx]))
        surprisals = [0.0] * extra_self_neighbors
        active_keys = input_keys if same_tag_neighborhood else structural_keys
        for other_idx in candidate_indices:
            s = case_surprisal(c_i, extracted[other_idx], deviations, active_keys)
            surprisals.append(s)

        if len(surprisals) > 0:
            surprisals.sort()
            k = min(3, len(surprisals))
            k_surprisal = statistics.mean(surprisals[:k])
            if k_surprisal >= req.anomaly_threshold:
                quarantined_indices.add(idx)
                anomalous_details.append({
                    "record_index": idx,
                    "tag": c_i["tag"],
                    "neighborhood_surprisal_nats": round(k_surprisal, 3),
                    "prompt_sample": c_i["prompt"][:80]
                })

    # Filter cleaned records preserving exact fractional weights
    final_indices = [idx for idx in surviving_indices if idx not in quarantined_indices]
    cleaned_records = [
        DatasetRecord(
            prompt=req.records[idx].prompt,
            completion=req.records[idx].completion,
            weight=float(weights[idx])
        )
        for idx in final_indices
    ]

    summary = AuditSummary(
        original_count=len(req.records),
        cleaned_count=len(cleaned_records),
        ablated_count=len(ablated_details),
        anomalous_count=len(anomalous_details),
        measured_deviations={k: round(v, 2) for k, v in deviations.items()},
        anomalous_records=anomalous_details,
        ablated_records=ablated_details
    )

    return AuditResponse(summary=summary, cleaned_records=cleaned_records)

@app.post("/v1/verify/completion", response_model=VerifyResponse)
def verify_completion(req: VerifyRequest):
    if not req.reference_records:
        raise HTTPException(status_code=400, detail="Reference records cannot be empty.")

    ref_extracted = [extract_features(r.prompt, r.completion) for r in req.reference_records]
    cand_extracted = extract_features(req.prompt, req.candidate_completion)
    deviations = compute_deviations(ref_extracted)

    # Reference selection strictly uses prompt-derived features
    prompt_keys = [k for k in deviations.keys() if k.startswith("prompt_")]
    scored = []
    for idx, c in enumerate(ref_extracted):
        s = case_surprisal(cand_extracted, c, deviations, prompt_keys)
        scored.append((s, idx, c))

    scored.sort(key=lambda x: x[0])
    k = min(3, len(scored))
    top_k = scored[:k]

    # In log space: subtract minimum surprisal among selected cases before exponentiating (numerical stability)
    min_s = top_k[0][0] if top_k else 0.0
    influential = [(s, math.exp(-(s - min_s)), idx, c) for s, idx, c in top_k]
    total_prob = sum(item[1] for item in influential) or 1.0

    tag_masses: Dict[str, float] = {}
    influential_cases = []
    for s, p, idx, c in influential:
        weight_pct = round((p / total_prob) * 100, 1)
        tag_masses[c["tag"]] = tag_masses.get(c["tag"], 0.0) + p
        influential_cases.append(InfluentialCase(
            case_index=idx,
            tag=c["tag"],
            surprisal_nats=round(s, 3),
            influence_weight_percent=weight_pct,
            source_prompt=c["prompt"][:90]
        ))

    predicted_tag = max(tag_masses.items(), key=lambda x: x[1])[0]
    
    # Evaluate completion content against the top influential reference case
    best_ref = influential[0][3]
    comp_keys = [k for k in deviations.keys() if k.startswith("completion_")]
    completion_surprisal = case_surprisal(cand_extracted, best_ref, deviations, comp_keys)
    total_measured_surprisal = influential[0][0] + completion_surprisal

    # If reference neighborhood has a known policy tag, candidate completion must match that tag.
    # When references have no policy tag (predicted_tag == "UNKNOWN"), surprisal alone determines conformance.
    if predicted_tag != "UNKNOWN":
        has_valid_tag = (cand_extracted["tag"] == predicted_tag)
        is_conforming = (total_measured_surprisal <= req.max_tolerable_surprisal) and has_valid_tag
        if not has_valid_tag:
            if cand_extracted["tag"] == "UNKNOWN":
                explanation = f"Untagged completion; requires mandatory policy tag [ {predicted_tag} ]."
            else:
                explanation = f"Tag mismatch: generated [ {cand_extracted['tag']} ] does not match required policy tag [ {predicted_tag} ]."
        elif not is_conforming:
            explanation = f"Anomalous compliance distance ({total_measured_surprisal:.2f} nats exceeds ceiling {req.max_tolerable_surprisal:.2f})."
        else:
            explanation = (
                f"Grounded by {influential_cases[0].influence_weight_percent}% match to Case #{influential_cases[0].case_index} "
                f"([ {predicted_tag} ]) with surprisal {total_measured_surprisal:.2f} nats."
            )
    else:
        has_valid_tag = True
        is_conforming = (total_measured_surprisal <= req.max_tolerable_surprisal)
        if not is_conforming:
            explanation = f"Anomalous compliance distance ({total_measured_surprisal:.2f} nats exceeds ceiling {req.max_tolerable_surprisal:.2f})."
        else:
            explanation = (
                f"Grounded by {influential_cases[0].influence_weight_percent}% match to Case #{influential_cases[0].case_index} "
                f"with surprisal {total_measured_surprisal:.2f} nats."
            )

    return VerifyResponse(
        is_conforming=is_conforming,
        measured_surprisal_nats=round(total_measured_surprisal, 3),
        predicted_tag=predicted_tag,
        candidate_tag=cand_extracted["tag"],
        influential_cases=influential_cases,
        explanation=explanation
    )
