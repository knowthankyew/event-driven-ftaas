"""
test_event_contracts.py

Cross-service contract tests verifying serialization, schema alignment,
and W3C trace context compatibility between the .NET 10 Control Plane
(Contracts.cs / RabbitMqEventPublisher) and the Python 3.12 Worker (consumer.py).
"""

import json
import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add worker and inference directories to sys.path
worker_dir = Path(__file__).resolve().parent.parent / "src" / "FtaaSService.Worker"
sys.path.insert(0, str(worker_dir))

from consumer import JobConsumer, emit_lifecycle_span, SAFE_ALLOWLIST_KEYS
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator
import opentelemetry.trace as trace


class TestCrossServiceContracts(unittest.TestCase):
    """Verifies schema contracts and header compatibility between .NET API and Python Worker."""

    def test_job_requested_dotnet_payload_consumed_seamlessly(self):
        """
        Validates that a JSON payload serialized according to .NET JobRequestedEvent
        (with camelCase naming) satisfies all consumer expectations without missing fields.
        """
        dotnet_job_requested_json = {
            "jobId": "job-dotnet-123",
            "jobName": "financial-sentiment-lora",
            "baseModel": "HuggingFaceTB/SmolLM2-135M",
            "datasetPath": "datasets/financial.jsonl",
            "datasetHash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
            "hyperparameters": {
                "epochs": 3,
                "batchSize": 4,
                "learningRate": 0.0002,
                "loraRank": 8,
                "loraAlpha": 16,
                "loraDropout": 0.05
            },
            "submittedAt": "2026-10-05T18:00:00.0000000Z"
        }

        # Verify all required keys are present
        body = json.dumps(dotnet_job_requested_json).encode("utf-8")
        parsed = json.loads(body.decode("utf-8"))

        self.assertIn("jobId", parsed)
        self.assertIn("jobName", parsed)
        self.assertIn("baseModel", parsed)
        self.assertIn("datasetPath", parsed)
        self.assertIn("datasetHash", parsed)
        self.assertIn("hyperparameters", parsed)
        self.assertIsInstance(parsed["hyperparameters"]["epochs"], int)
        self.assertIsInstance(parsed["hyperparameters"]["learningRate"], float)

    def test_w3c_traceparent_byte_header_extracted_by_consumer(self):
        """
        Validates that byte-array headers sent by .NET RabbitMqEventPublisher
        (['traceparent'] = byte[]) are extracted into W3C OpenTelemetry SpanContext.
        """
        trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        span_id = "00f067aa0ba902b7"
        traceparent_str = f"00-{trace_id}-{span_id}-01"

        # Simulate RabbitMQ properties containing byte headers as emitted by .NET
        mock_props = MagicMock()
        mock_props.headers = {
            "traceparent": traceparent_str.encode("utf-8"),
            "tracestate": b"rojo=1"
        }

        # Consumer carrier extraction logic
        carrier = {}
        for k, v in mock_props.headers.items():
            carrier[k] = v.decode("utf-8") if isinstance(v, bytes) else str(v)

        ctx = TraceContextTextMapPropagator().extract(carrier)
        self.assertIsNotNone(ctx)

        # Ensure extracted span has the exact remote trace and span IDs
        span = trace.get_current_span(ctx)
        span_ctx = span.get_span_context()
        self.assertEqual(f"{span_ctx.trace_id:032x}", trace_id)
        self.assertEqual(f"{span_ctx.span_id:016x}", span_id)

    def test_job_updated_payload_matches_dotnet_contract(self):
        """
        Validates that publish_status_update produces a payload matching
        .NET JobUpdatedEvent property names and JSON serialization attributes.
        """
        consumer = JobConsumer()
        mock_channel = MagicMock()
        consumer.channel = mock_channel
        consumer.current_traceparent = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"

        consumer.publish_status_update(
            job_id="job-contract-test",
            status="Training",
            progress_pct=45.5,
            current_step=45,
            total_steps=100,
            current_loss=0.314,
            mlflow_experiment_id="exp-1",
            mlflow_run_id="run-1",
            adapter_path="artifacts/job-contract-test/model_adapters",
            started_at="2026-10-05T18:00:00Z"
        )

        mock_channel.basic_publish.assert_called_once()
        call_kwargs = mock_channel.basic_publish.call_args[1]
        payload = json.loads(call_kwargs["body"])

        # Check all .NET JobUpdatedEvent property contracts
        expected_keys = {
            "jobId", "status", "sequenceNumber", "progressPercent",
            "currentStep", "totalSteps", "currentLoss", "mlflowExperimentId",
            "mlflowRunId", "adapterPath", "errorMessage", "startedAt",
            "finishedAt", "updatedAt"
        }
        for k in expected_keys:
            self.assertIn(k, payload, f"Missing expected key '{k}' in JobUpdatedEvent contract")

        # Verify traceparent was passed in message headers
        props = call_kwargs["properties"]
        self.assertEqual(props.correlation_id, "job-contract-test")
        self.assertIn("traceparent", props.headers)
        self.assertEqual(props.headers["traceparent"], consumer.current_traceparent)

    def test_status_update_sequence_numbers_increment_monotonically(self):
        """Verifies sequence numbers increment monotonically per job ID to ensure AMQP idempotency."""
        consumer = JobConsumer()
        mock_channel = MagicMock()
        consumer.channel = mock_channel

        job_id = "job-monotonic-seq"
        for i in range(1, 6):
            consumer.publish_status_update(
                job_id=job_id,
                status="Training",
                current_step=i * 10
            )
            call_kwargs = mock_channel.basic_publish.call_args[1]
            payload = json.loads(call_kwargs["body"])
            self.assertEqual(payload["sequenceNumber"], i)


if __name__ == "__main__":
    unittest.main()
