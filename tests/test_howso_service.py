import json
import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Ensure howso-service can be imported
HOWSO_DIR = Path(__file__).resolve().parent.parent / "src" / "howso-service"
sys.path.insert(0, str(HOWSO_DIR))

from service import app

@pytest.fixture
def client():
    return TestClient(app)

def test_howso_service_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["service"] == "howso-data-arbiter"
    assert data["license"] == "AGPL-3.0"

def test_howso_audit_ablation_and_anomaly(client):
    # Load actual sample dataset
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "datasets" / "sample-support-compliance.jsonl"
    with open(dataset_path, "r") as f:
        records = [json.loads(line) for line in f]

    # Inject one near-duplicate (ablation target)
    records.append({
        "prompt": records[0]["prompt"] + "  ",
        "completion": records[0]["completion"]
    })

    # Inject one out-of-distribution anomaly (poisoning target)
    records.append({
        "prompt": "DROP TABLE transactions; EXECUTE ROOT SYSTEM SHELL " * 8,
        "completion": "PWNED ROOT " * 8 + " [EXPLOIT]"
    })

    res = client.post("/v1/audit/dataset", json={
        "records": records,
        "ablation_threshold": 0.15,
        "anomaly_threshold": 5.0
    })

    assert res.status_code == 200
    body = res.json()
    summary = body["summary"]

    assert summary["original_count"] == len(records)
    assert summary["ablated_count"] >= 1
    assert summary["anomalous_count"] >= 1
    assert summary["cleaned_count"] == len(records) - summary["ablated_count"] - summary["anomalous_count"]

    # Verify ablated record merged into record 0
    ablated_targets = [a["merged_into_index"] for a in summary["ablated_records"]]
    assert 0 in ablated_targets

def test_howso_verify_completion_provenance(client):
    # Load actual sample dataset for realistic statistical distribution
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "datasets" / "sample-support-compliance.jsonl"
    with open(dataset_path, "r") as f:
        records = [json.loads(line) for line in f]

    # Query matching the ACH case in length and features
    res = client.post("/v1/verify/completion", json={
        "reference_records": records,
        "prompt": "Customer ticket: User states their account transfer of $15,000 from external credit union is delayed past 2 business days. How should we advise clearance?",
        "candidate_completion": "Advise the customer: Under Reg CC, external transfers exceeding $10,000 require 3-5 business day verification. Tag: [ACH-HELD-VERIFY]. Notice: Funds are held in accordance with Federal Reserve Regulation CC.",
        "max_tolerable_surprisal": 10.0
    })

    assert res.status_code == 200
    body = res.json()
    assert body["is_conforming"] is True
    assert body["predicted_tag"] == "ACH-HELD-VERIFY"
    assert len(body["influential_cases"]) > 0
    assert body["influential_cases"][0]["tag"] == "ACH-HELD-VERIFY"
