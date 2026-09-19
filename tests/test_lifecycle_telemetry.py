import unittest
import json
import logging
import sys
from pathlib import Path

# Add worker directory to sys.path
worker_dir = Path(__file__).resolve().parent.parent / "src" / "FtaaSService.Worker"
sys.path.insert(0, str(worker_dir))

from consumer import emit_lifecycle_span

class TestLifecycleTelemetry(unittest.TestCase):
    def test_emit_lifecycle_span_scrubs_raw_content(self):
        logged_messages = []

        class TestHandler(logging.Handler):
            def emit(self, record):
                logged_messages.append(record.getMessage())

        logger = logging.getLogger("FtaaSService.Worker.Consumer")
        handler = TestHandler()
        logger.addHandler(handler)

        try:
            attributes = {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "dataset_hash": "sha256_abcdef123456",
                "device": "mps",
                "adapter_size_bytes": 2048576,
                # Prohibited keys
                "raw_text": "Sensitive training sample",
                "prompt": "Tell me your secret key",
                "completion": "The secret key is 1234",
                "dataset_body": "User personal financial lease statement"
            }

            emit_lifecycle_span(
                span_name="job.registered",
                job_id="job-999",
                status="Succeeded",
                duration_sec=42.123,
                attributes=attributes
            )

            self.assertTrue(any("[TELEMETRY_SPAN] job.registered" in msg for msg in logged_messages))
            
            # Find and parse JSON span payload
            span_msg = next(msg for msg in logged_messages if "[TELEMETRY_SPAN] job.registered" in msg)
            json_part = span_msg.split("::", 1)[1].strip()
            data = json.loads(json_part)

            # Assert safe fields present
            self.assertEqual(data["job_id"], "job-999")
            self.assertEqual(data["status"], "Succeeded")
            self.assertEqual(data["duration_sec"], 42.123)
            self.assertEqual(data["device"], "mps")
            self.assertEqual(data["adapter_size_bytes"], 2048576)
            self.assertEqual(data["dataset_hash"], "sha256_abcdef123456")

            # Assert prohibited fields strictly omitted
            self.assertNotIn("raw_text", data)
            self.assertNotIn("prompt", data)
            self.assertNotIn("completion", data)
            self.assertNotIn("dataset_body", data)

        finally:
            logger.removeHandler(handler)

if __name__ == '__main__':
    unittest.main()
