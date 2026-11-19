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
        prev_level = logger.level
        logger.setLevel(logging.INFO)
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

            # Assert non-allowlisted fields are strictly redacted
            self.assertEqual(data["raw_text"], "[REDACTED_NOT_IN_ALLOWLIST]")
            self.assertEqual(data["prompt"], "[REDACTED_NOT_IN_ALLOWLIST]")
            self.assertEqual(data["completion"], "[REDACTED_NOT_IN_ALLOWLIST]")
            self.assertEqual(data["dataset_body"], "[REDACTED_NOT_IN_ALLOWLIST]")

        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)

    def test_w3c_traceparent_extracted_and_linked_to_span(self):
        """Verifies that W3C traceparent headers can be extracted and linked to child lifecycle spans."""
        from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
        import consumer

        carrier = {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
        }
        ctx = TraceContextTextMapPropagator().extract(carrier)
        self.assertIsNotNone(ctx)

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        test_tracer = provider.get_tracer("test-tracer")

        orig_tracer = consumer.tracer
        orig_has_otel = consumer.HAS_OTEL
        try:
            consumer.tracer = test_tracer
            consumer.HAS_OTEL = True

            emit_lifecycle_span(
                span_name="job.consumed",
                job_id="job-w3c-test",
                status="Consumed",
                attributes={"base_model": "HuggingFaceTB/SmolLM2-135M"},
                parent_context=ctx
            )

            spans = exporter.get_finished_spans()
            self.assertEqual(len(spans), 1)
            span = spans[0]
            self.assertEqual(span.context.trace_id, 0x4bf92f3577b34da6a3ce929d0e0e4736)
            self.assertEqual(span.parent.span_id, 0x00f067aa0ba902b7)
        finally:
            consumer.tracer = orig_tracer
            consumer.HAS_OTEL = orig_has_otel

    def test_job_failed_lifecycle_span_scrubs_attributes(self):
        """Verifies that job.failed span allows 'error' attribute while scrubbing sensitive payload data."""
        logged_messages = []

        class TestHandler(logging.Handler):
            def emit(self, record):
                logged_messages.append(record.getMessage())

        logger = logging.getLogger("FtaaSService.Worker.Consumer")
        prev_level = logger.level
        logger.setLevel(logging.INFO)
        handler = TestHandler()
        logger.addHandler(handler)

        try:
            emit_lifecycle_span(
                span_name="job.failed",
                job_id="job-err-1",
                status="Failed",
                attributes={
                    "error": "CUDA out of memory",
                    "secret_user_key": "sk-12345"
                }
            )

            span_msg = next(msg for msg in logged_messages if "[TELEMETRY_SPAN] job.failed" in msg)
            json_part = span_msg.split("::", 1)[1].strip()
            data = json.loads(json_part)

            self.assertEqual(data["job_id"], "job-err-1")
            self.assertEqual(data["status"], "Failed")
            self.assertEqual(data["error"], "CUDA out of memory")
            self.assertEqual(data["secret_user_key"], "[REDACTED_NOT_IN_ALLOWLIST]")

        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)

    def test_emit_lifecycle_span_sanitizes_path_valued_attributes(self):
        import hashlib
        import hmac
        import os
        from unittest.mock import patch

        logged_messages = []

        class TestHandler(logging.Handler):
            def emit(self, record):
                logged_messages.append(record.getMessage())

        logger = logging.getLogger("FtaaSService.Worker.Consumer")
        prev_level = logger.level
        logger.setLevel(logging.INFO)
        handler = TestHandler()
        logger.addHandler(handler)

        try:
            raw_dataset_path = "/Users/alice/data/datasets/job-path-1.jsonl"
            raw_adapter_path = "/Users/alice/data/artifacts/job-path-1/model_adapters"
            with patch.dict(os.environ, {"FTAAS_TELEMETRY_HMAC_KEY": "worker-test-hmac-secret"}):
                emit_lifecycle_span(
                    span_name="job.registered",
                    job_id="job-path-1",
                    status="Succeeded",
                    attributes={
                        "dataset_relative_path": raw_dataset_path,
                        "adapter_path": raw_adapter_path,
                    }
                )
            expected_single_pass = "hmac:" + hmac.new(
                b"worker-test-hmac-secret",
                raw_adapter_path.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()[:16]

            span_msg = next(msg for msg in logged_messages if "[TELEMETRY_SPAN] job.registered" in msg)
            data = json.loads(span_msg.split("::", 1)[1].strip())
            self.assertEqual(data["dataset_relative_path"], "job-path-1.jsonl")
            self.assertNotIn("/", data["dataset_relative_path"])
            self.assertNotIn("alice", data["dataset_relative_path"])
            self.assertRegex(data["adapter_path"], r"^hmac:[a-f0-9]{16}$")
            self.assertEqual(data["adapter_path"], expected_single_pass)
            unsalted_hex = hashlib.sha256(raw_adapter_path.encode("utf-8")).hexdigest()[:16]
            self.assertNotEqual(data["adapter_path"], f"hmac:{unsalted_hex}")
            self.assertNotIn("/", data["adapter_path"])
            self.assertNotIn("alice", data["adapter_path"])
        finally:
            logger.removeHandler(handler)
            logger.setLevel(prev_level)

if __name__ == '__main__':
    unittest.main()
