"""
test_bitnet_engine.py

Comprehensive test suite for the native BitNet C++ inference engine adapter
and dual-backend routing in FtaaSService.Inference (Phase 7).
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure worker and inference directories are on sys.path
worker_dir = Path(__file__).resolve().parent.parent / "src" / "FtaaSService.Worker"
inference_dir = Path(__file__).resolve().parent.parent / "src" / "FtaaSService.Inference"
sys.path.insert(0, str(worker_dir))
sys.path.insert(0, str(inference_dir))

import bitnet_engine
import app


class TestBitNetEngineBasics(unittest.TestCase):
    """Unit tests for BitNet model detection, status telemetry, and CLI output parsing."""

    def test_is_bitnet_model(self):
        self.assertTrue(bitnet_engine.is_bitnet_model("microsoft/BitNet-b1.58-2B-4T"))
        self.assertTrue(bitnet_engine.is_bitnet_model("bitnet-b1.58-2b"))
        self.assertTrue(bitnet_engine.is_bitnet_model("BITNET_TEST"))

        self.assertFalse(bitnet_engine.is_bitnet_model("HuggingFaceTB/SmolLM2-135M"))
        self.assertFalse(bitnet_engine.is_bitnet_model("google/gemma-2-2b-it"))
        self.assertFalse(bitnet_engine.is_bitnet_model(None))
        self.assertFalse(bitnet_engine.is_bitnet_model(""))

    def test_parse_llama_cli_output_standard(self):
        raw = """
Loading model...

build      : b9918-390c30775
model      : models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf

> User: What is Regulation CC?<|eot_id|>
Assistant:

Regulation CC sets funds availability schedules and check-clearing requirements.

[ Prompt: 100.2 t/s | Generation: 19.5 t/s ]

Exiting...
"""
        prompt = "User: What is Regulation CC?<|eot_id|>\nAssistant:"
        parsed = bitnet_engine.parse_llama_cli_output(raw, prompt)
        self.assertEqual(
            parsed,
            "Regulation CC sets funds availability schedules and check-clearing requirements."
        )

    def test_parse_llama_cli_output_fallback_delimiter(self):
        raw = """
> User: Compliance test prompt<|eot_id|>
Assistant: Output with no prompt metrics footer
Exiting...
"""
        prompt = "User: Compliance test prompt<|eot_id|>\nAssistant:"
        parsed = bitnet_engine.parse_llama_cli_output(raw, prompt)
        self.assertEqual(parsed, "Output with no prompt metrics footer")

    def test_get_bitnet_status_structure(self):
        status = bitnet_engine.get_bitnet_status()
        self.assertIn("available", status)
        self.assertIn("cliPath", status)
        self.assertIn("cliExecutable", status)
        self.assertIn("modelPath", status)
        self.assertIn("modelPresent", status)
        self.assertIn("threads", status)


class TestBitNetExecution(unittest.TestCase):
    """Tests execution paths, timeouts, error conditions, and mocks."""

    @patch("bitnet_engine.is_bitnet_available", return_value=False)
    def test_generate_bitnet_sync_raises_when_unavailable(self, _mock_avail):
        with self.assertRaises(RuntimeError) as ctx:
            bitnet_engine.generate_bitnet_sync("test prompt")
        self.assertIn("BitNet C++ native runtime is not available", str(ctx.exception))

    @patch("bitnet_engine.is_bitnet_available", return_value=True)
    @patch("subprocess.run")
    def test_generate_bitnet_sync_success(self, mock_run, _mock_avail):
        sample_output = """
> User: Explain Regulation CC<|eot_id|>
Assistant:
1. Mandatory next-day availability for government checks.
2. Standard availability up to $5,525.
3. Extended exceptions for fraud protection.

[ Prompt: 105.0 t/s | Generation: 23.5 t/s ]
Exiting...
"""
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = sample_output
        mock_res.stderr = ""
        mock_run.return_value = mock_res

        completion, latency = bitnet_engine.generate_bitnet_sync(
            "Explain Regulation CC",
            max_tokens=64,
            temperature=0.2
        )
        self.assertIn("Mandatory next-day availability", completion)
        self.assertGreater(latency, 0.0)

        # Verify command flags passed to subprocess
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertIn("-ngl", cmd)
        self.assertIn("-st", cmd)
        self.assertIn("--simple-io", cmd)

    @patch("bitnet_engine.is_bitnet_available", return_value=True)
    @patch("asyncio.create_subprocess_exec")
    def test_generate_bitnet_async_success(self, mock_exec, _mock_avail):
        sample_output = """
> User: Async test<|eot_id|>
Assistant: Async generation completed successfully.
[ Prompt: 100.0 t/s | Generation: 20.0 t/s ]
"""
        mock_proc = MagicMock()
        mock_proc.returncode = 0

        async def _mock_communicate():
            return sample_output.encode("utf-8"), b""

        mock_proc.communicate = _mock_communicate
        mock_exec.return_value = mock_proc

        async def run_async():
            return await bitnet_engine.generate_bitnet("Async test")

        completion, latency = asyncio.run(run_async())
        self.assertEqual(completion, "Async generation completed successfully.")
        self.assertGreater(latency, 0.0)


class TestAppRoutingWithBitNet(unittest.TestCase):
    """Validates FastAPI app routing for BitNet in /generate, /compare, and /healthz."""

    @patch("app.is_bitnet_available", return_value=True)
    @patch("app.generate_bitnet_sync")
    def test_compare_completions_routes_to_bitnet_successfully(self, mock_gen, _mock_avail):
        mock_gen.side_effect = [
            ("Raw base output from BitNet C++", 1250.0),
            ("Fine-tuned adapter output from BitNet", 1180.0)
        ]

        req = app.CompareRequest(
            jobId="test-job-bitnet",
            baseModel="microsoft/BitNet-b1.58-2B-4T",
            adapterPath="test/adapter/path",
            prompt="Regulation CC prompt",
            maxTokens=32,
            temperature=0.1
        )

        res = app.compare_completions(req)
        self.assertEqual(res["baseModel"], "microsoft/BitNet-b1.58-2B-4T")
        self.assertEqual(res["baseCompletion"], "Raw base output from BitNet C++")
        self.assertEqual(res["fineTunedCompletion"], "Fine-tuned adapter output from BitNet")
        self.assertEqual(res["latencyMs"]["baseModel"], 1250.0)
        self.assertEqual(res["latencyMs"]["fineTuned"], 1180.0)

    @patch("app.is_bitnet_available", return_value=True)
    @patch("app.generate_bitnet_sync")
    def test_generate_routes_to_bitnet_successfully(self, mock_gen, _mock_avail):
        mock_gen.return_value = ("BitNet completion single-shot", 850.0)

        req = app.GenerateRequest(
            baseModel="microsoft/BitNet-b1.58-2B-4T",
            prompt="Hello BitNet",
            maxTokens=32
        )

        res = app.generate(req)
        self.assertEqual(res["baseModel"], "microsoft/BitNet-b1.58-2B-4T")
        self.assertEqual(res["completion"], "BitNet completion single-shot")
        self.assertEqual(res["latencyMs"], 850.0)

    @patch("app.is_bitnet_available", return_value=True)
    def test_healthz_reflects_bitnet_telemetry(self, _mock_avail):
        res = app.healthz()
        self.assertIn("bitnet", res)
        self.assertIn("available", res["bitnet"])
        self.assertIn("backend", res)


if __name__ == "__main__":
    unittest.main()
