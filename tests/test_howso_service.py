import json
import unittest
from pathlib import Path
import sys
from fastapi.testclient import TestClient

# Ensure howso-service can be imported
HOWSO_DIR = Path(__file__).resolve().parent.parent / "src" / "howso-service"
sys.path.insert(0, str(HOWSO_DIR))

from service import app


class TestHowsoService(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_howso_service_health(self):
        res = self.client.get("/health")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "howso-data-arbiter")
        self.assertEqual(data["license"], "AGPL-3.0")

    def test_howso_audit_ablation_and_anomaly(self):
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

        res = self.client.post("/v1/audit/dataset", json={
            "records": records,
            "ablation_threshold": 0.15,
            "anomaly_threshold": 5.0
        })

        self.assertEqual(res.status_code, 200)
        body = res.json()
        summary = body["summary"]

        self.assertEqual(summary["original_count"], len(records))
        self.assertGreaterEqual(summary["ablated_count"], 1)
        self.assertGreaterEqual(summary["anomalous_count"], 1)
        self.assertEqual(summary["cleaned_count"], len(records) - summary["ablated_count"] - summary["anomalous_count"])

        # Verify ablated record merged into record 0
        ablated_targets = [a["merged_into_index"] for a in summary["ablated_records"]]
        self.assertIn(0, ablated_targets)

    def test_howso_verify_completion_provenance(self):
        # Load actual sample dataset for realistic statistical distribution
        dataset_path = Path(__file__).resolve().parent.parent / "data" / "datasets" / "sample-support-compliance.jsonl"
        with open(dataset_path, "r") as f:
            records = [json.loads(line) for line in f]

        # Query matching the ACH case in length and features
        res = self.client.post("/v1/verify/completion", json={
            "reference_records": records,
            "prompt": "Customer ticket: User states their account transfer of $15,000 from external credit union is delayed past 2 business days. How should we advise clearance?",
            "candidate_completion": "Advise the customer: Under Reg CC, external transfers exceeding $10,000 require 3-5 business day verification. Tag: [ACH-HELD-VERIFY]. Notice: Funds are held in accordance with Federal Reserve Regulation CC.",
            "max_tolerable_surprisal": 10.0
        })

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertTrue(body["is_conforming"])
        self.assertEqual(body["predicted_tag"], "ACH-HELD-VERIFY")
        self.assertGreater(len(body["influential_cases"]), 0)
        self.assertEqual(body["influential_cases"][0]["tag"], "ACH-HELD-VERIFY")


if __name__ == "__main__":
    unittest.main()
