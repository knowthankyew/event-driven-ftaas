import json
import socket
import subprocess
import sys
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

HOWSO_DIR = Path(__file__).resolve().parent.parent / "src" / "howso-service"


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class _SubprocessHttpClient:
    """
    Lightweight HTTP client communicating with the isolated howso-service
    subprocess over loopback TCP/JSON.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def get(self, path: str) -> tuple[int, dict]:
        req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
        try:
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))

    def post(self, path: str, payload: dict) -> tuple[int, dict]:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15.0) as resp:
                return resp.status, json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode("utf-8"))


class TestHowsoService(unittest.TestCase):
    """
    Tests howso-service over an isolated out-of-process HTTP boundary to ensure
    clean microservice interface isolation.
    """

    proc: subprocess.Popen | None = None
    client: _SubprocessHttpClient

    @classmethod
    def setUpClass(cls):
        port = _find_free_port()
        cls.base_url = f"http://127.0.0.1:{port}"
        cls.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "service:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=str(HOWSO_DIR),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        cls.client = _SubprocessHttpClient(cls.base_url)

        # Wait for health endpoint readiness
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if cls.proc.poll() is not None:
                _, stderr = cls.proc.communicate(timeout=2)
                raise RuntimeError(f"howso-service subprocess exited early: {stderr}")
            try:
                status, _ = cls.client.get("/health")
                if status == 200:
                    return
            except Exception:
                time.sleep(0.1)
        cls.tearDownClass()
        raise RuntimeError("Timed out waiting for isolated howso-service subprocess on /health")

    @classmethod
    def tearDownClass(cls):
        if cls.proc is not None:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
                cls.proc.wait(timeout=2.0)
            if cls.proc.stdout:
                cls.proc.stdout.close()
            if cls.proc.stderr:
                cls.proc.stderr.close()
            cls.proc = None

    def test_zero_in_process_service_import(self):
        """Verify that `service` from src/howso-service is NOT loaded in sys.modules."""
        for mod_name, mod in list(sys.modules.items()):
            mod_file = getattr(mod, "__file__", None) or ""
            self.assertNotIn(
                "howso-service",
                str(mod_file),
                f"Subprocess module '{mod_name}' ({mod_file}) must not be imported in-process.",
            )

    def test_howso_service_health(self):
        status, data = self.client.get("/health")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "healthy")
        self.assertEqual(data["service"], "howso-data-arbiter")
        self.assertEqual(data["license"], "MIT")
        self.assertFalse(data["native_howso_available"])
        self.assertIn("Reference Information Engine", data["engine"])

    def test_howso_audit_ablation_and_anomaly(self):
        # Load actual sample dataset
        dataset_path = Path(__file__).resolve().parent.parent / "data" / "datasets" / "sample-support-compliance.jsonl"
        with open(dataset_path, "r", encoding="utf-8") as f:
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

        status, body = self.client.post("/v1/audit/dataset", {
            "records": records,
            "ablation_threshold": 0.15,
            "anomaly_threshold": 5.0
        })

        self.assertEqual(status, 200)
        summary = body["summary"]

        self.assertEqual(summary["original_count"], len(records))
        self.assertGreaterEqual(summary["ablated_count"], 1)
        self.assertGreaterEqual(summary["anomalous_count"], 1)
        self.assertEqual(summary["cleaned_count"], len(records) - summary["ablated_count"] - summary["anomalous_count"])

        # Verify ablated record merged into record 0 and increased its probability mass weight
        ablated_targets = [a["merged_into_index"] for a in summary["ablated_records"]]
        self.assertIn(0, ablated_targets)
        self.assertGreater(body["cleaned_records"][0]["weight"], 1.0)

    def test_howso_verify_completion_provenance(self):
        # Load actual sample dataset for realistic statistical distribution
        dataset_path = Path(__file__).resolve().parent.parent / "data" / "datasets" / "sample-support-compliance.jsonl"
        with open(dataset_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        # Query matching the ACH case in length and features
        status, body = self.client.post("/v1/verify/completion", {
            "reference_records": records,
            "prompt": "Customer ticket: User states their account transfer of $15,000 from external credit union is delayed past 2 business days. How should we advise clearance?",
            "candidate_completion": "Advise the customer: Under Reg CC, external transfers exceeding $10,000 require 3-5 business day verification. Tag: [ACH-HELD-VERIFY]. Notice: Funds are held in accordance with Federal Reserve Regulation CC.",
            "max_tolerable_surprisal": 10.0
        })

        self.assertEqual(status, 200)
        self.assertTrue(body["is_conforming"])
        self.assertEqual(body["predicted_tag"], "ACH-HELD-VERIFY")
        self.assertGreater(len(body["influential_cases"]), 0)
        self.assertEqual(body["influential_cases"][0]["tag"], "ACH-HELD-VERIFY")


if __name__ == "__main__":
    unittest.main()
