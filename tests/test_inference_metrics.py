"""
test_inference_metrics.py

Validates the Prometheus exposition metrics endpoint (/metrics) in the
FastAPI Inference service.
"""
import sys
import unittest
from pathlib import Path

# Add worker and inference directories to sys.path
worker_dir = Path(__file__).resolve().parent.parent / "src" / "FtaaSService.Worker"
inference_dir = Path(__file__).resolve().parent.parent / "src" / "FtaaSService.Inference"
sys.path.insert(0, str(worker_dir))
sys.path.insert(0, str(inference_dir))

import app


class TestInferenceMetrics(unittest.TestCase):
    def test_metrics_exposition_format(self):
        """Validates that get_metrics returns RFC-compliant Prometheus exposition text."""
        response = app.get_metrics()
        self.assertEqual(response.media_type, "text/plain; version=0.0.4; charset=utf-8")
        
        body = response.body.decode("utf-8")
        self.assertIn("# HELP ftaas_inference_uptime_seconds", body)
        self.assertIn("# TYPE ftaas_inference_uptime_seconds gauge", body)
        self.assertIn("# HELP ftaas_inference_cached_adapters", body)
        self.assertIn("# TYPE ftaas_inference_cached_adapters gauge", body)
        self.assertIn("# HELP ftaas_inference_max_cached_adapters", body)
        self.assertIn(f"ftaas_inference_max_cached_adapters {app.MAX_CACHED_ADAPTERS}", body)
        self.assertIn("# HELP ftaas_inference_base_model_loaded", body)
        self.assertIn("# HELP ftaas_inference_requests_total", body)
        self.assertIn("# TYPE ftaas_inference_requests_total counter", body)
        self.assertIn('ftaas_inference_requests_total{endpoint="compare",status="success"}', body)
        self.assertIn('ftaas_inference_requests_total{endpoint="generate",status="success"}', body)
        self.assertIn('ftaas_inference_requests_total{endpoint="embed",status="success"}', body)
        self.assertIn("# HELP ftaas_inference_embed_model_loaded", body)

    def test_record_request_increments_counters(self):
        """Verifies that recording requests properly increments metrics."""
        initial_compare_count = app.metrics.requests_total["compare"]["success"]
        initial_compare_latency = app.metrics.latency_sum_ms["compare"]

        app.metrics.record_request("compare", "success", latency_ms=45.5)

        self.assertEqual(app.metrics.requests_total["compare"]["success"], initial_compare_count + 1)
        self.assertAlmostEqual(app.metrics.latency_sum_ms["compare"], initial_compare_latency + 45.5, places=2)

        # Record error
        initial_error_count = app.metrics.requests_total["compare"]["error"]
        app.metrics.record_request("compare", "error")
        self.assertEqual(app.metrics.requests_total["compare"]["error"], initial_error_count + 1)

        # Verify output reflection
        body = app.get_metrics().body.decode("utf-8")
        self.assertIn(f'ftaas_inference_requests_total{{endpoint="compare",status="error"}} {initial_error_count + 1}', body)


if __name__ == "__main__":
    unittest.main()
