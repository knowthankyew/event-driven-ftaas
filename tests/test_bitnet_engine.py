"""
test_bitnet_engine.py

Comprehensive test suite for the native BitNet C++ inference engine adapter
and dual-backend routing in FtaaSService.Inference (Phase 7).
"""

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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

        # Verify command flags passed to subprocess and zero prompt leakage
        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertIn("-ngl", cmd)
        self.assertIn("-st", cmd)
        self.assertIn("--simple-io", cmd)
        self.assertIn("-f", cmd)
        self.assertNotIn("-p", cmd)
        self.assertTrue(all("Explain Regulation CC" not in str(arg) for arg in cmd))

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

        async def _mock_communicate(*args, **kwargs):
            return sample_output.encode("utf-8"), b""

        mock_proc.communicate = _mock_communicate
        mock_exec.return_value = mock_proc

        async def run_async():
            return await bitnet_engine.generate_bitnet("Async test")

        completion, latency = asyncio.run(run_async())
        self.assertEqual(completion, "Async generation completed successfully.")
        self.assertGreater(latency, 0.0)

        # Assert zero prompt leakage in process arguments
        exec_args = mock_exec.call_args[0]
        self.assertNotIn("-p", exec_args)
        self.assertTrue(all("Async test" not in str(arg) for arg in exec_args))

    def test_command_builders_zero_prompt_exposure(self):
        """Ensure command builders route through files/pipes rather than CLI args (-p)."""
        gen_cmd = bitnet_engine.build_bitnet_generate_cmd("/dev/stdin", max_tokens=100)
        self.assertIn("-f", gen_cmd)
        self.assertIn("/dev/stdin", gen_cmd)
        self.assertNotIn("-p", gen_cmd)
        self.assertIn("-c", gen_cmd)
        self.assertEqual(gen_cmd[gen_cmd.index("-c") + 1], "4096")

        embed_cmd = bitnet_engine.build_bitnet_embed_cmd("/dev/stdin")
        self.assertIn("-f", embed_cmd)
        self.assertIn("/dev/stdin", embed_cmd)
        self.assertIn("--embd-separator", embed_cmd)
        self.assertIn("<#sep#>", embed_cmd)
        self.assertIn("-c", embed_cmd)
        self.assertIn("512", embed_cmd)
        self.assertNotIn("-p", embed_cmd)

    def test_sandboxed_command_wrapping(self):
        """Verify wrap_sandboxed_cmd conditionally applies macOS sandbox-exec."""
        base_cmd = ["llama-cli", "-m", "model.gguf"]
        # Disabled
        with patch.object(bitnet_engine, "BITNET_SANDBOX_NETWORK_DENY", False):
            cmd = bitnet_engine.wrap_sandboxed_cmd(base_cmd)
            self.assertEqual(cmd, base_cmd)

        # Enabled on darwin
        mock_sandbox = MagicMock()
        mock_sandbox.is_file.return_value = True
        mock_sandbox.__str__.return_value = "/usr/bin/sandbox-exec"
        with patch.object(bitnet_engine, "BITNET_SANDBOX_NETWORK_DENY", True), \
             patch("sys.platform", "darwin"), \
             patch.object(bitnet_engine, "SANDBOX_EXEC_PATH", mock_sandbox):
            sandboxed = bitnet_engine.wrap_sandboxed_cmd(base_cmd)
            self.assertEqual(sandboxed[:3], ["/usr/bin/sandbox-exec", "-p", "(version 1)(allow default)(deny network*)"])
            self.assertEqual(sandboxed[3:], base_cmd)


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




class TestBitNetEmbedding(unittest.TestCase):
    """Unit tests for Phase 8 dense embedding capabilities."""

    @patch("bitnet_engine.is_bitnet_embed_available", return_value=False)
    def test_embed_bitnet_sync_raises_when_unavailable(self, _mock_avail):
        with self.assertRaises(RuntimeError) as ctx:
            bitnet_engine.embed_bitnet_sync("test prompt")
        self.assertIn("BitNet C++ embedding runtime is not available", str(ctx.exception))

    @patch("bitnet_engine.is_bitnet_embed_available", return_value=True)
    @patch("subprocess.run")
    def test_embed_bitnet_sync_success(self, mock_run, _mock_avail):
        mock_res = MagicMock()
        mock_res.returncode = 0
        dummy_vec = [0.1] * 640
        mock_res.stdout = f"some log text\n[[{','.join(map(str, dummy_vec))}]]\nmore logs"
        mock_res.stderr = ""
        mock_run.return_value = mock_res

        vec, lat = bitnet_engine.embed_bitnet_sync("Statutory clause")
        self.assertEqual(len(vec), 640)
        self.assertGreater(lat, 0.0)

        args, kwargs = mock_run.call_args
        cmd = args[0]
        self.assertIn("-ngl", cmd)
        self.assertIn("--pooling", cmd)
        self.assertIn("-f", cmd)
        self.assertIn("--embd-separator", cmd)
        self.assertIn("<#sep#>", cmd)
        self.assertNotIn("-p", cmd)
        self.assertEqual(kwargs.get("encoding"), "utf-8")
        self.assertTrue(all("Statutory clause" not in str(arg) for arg in cmd))

    @patch("bitnet_engine.is_bitnet_embed_available", return_value=True)
    @patch("subprocess.run")
    def test_embed_bitnet_sync_unicode_statute(self, mock_run, _mock_avail):
        statute_text = "§ 229.10(c)(1)(vi) — “Next-day availability” exception for $5,525."
        mock_res = MagicMock()
        mock_res.returncode = 0
        dummy_vec = [0.1] * 640
        mock_res.stdout = f"[[{','.join(map(str, dummy_vec))}]]"
        mock_res.stderr = ""
        mock_run.return_value = mock_res

        vec, lat = bitnet_engine.embed_bitnet_sync(statute_text)
        self.assertEqual(len(vec), 640)
        args, kwargs = mock_run.call_args
        self.assertEqual(kwargs.get("input"), statute_text)
        self.assertEqual(kwargs.get("encoding"), "utf-8")

    @patch("bitnet_engine.is_bitnet_embed_available", return_value=True)
    @patch("subprocess.run")
    def test_embed_bitnet_separator_sanitization(self, mock_run, _mock_avail):
        prompt_with_separator = "Clause one<#sep#>Clause two"
        mock_res = MagicMock()
        mock_res.returncode = 0
        dummy_vec = [0.1] * 640
        mock_res.stdout = f"[[{','.join(map(str, dummy_vec))}]]"
        mock_res.stderr = ""
        mock_run.return_value = mock_res

        vec, lat = bitnet_engine.embed_bitnet_sync(prompt_with_separator)
        self.assertEqual(len(vec), 640)
        args, kwargs = mock_run.call_args
        self.assertNotIn("<#sep#>", kwargs.get("input", ""))
        self.assertEqual(kwargs.get("input"), "Clause one Clause two")

    @patch("bitnet_engine.is_bitnet_embed_available", return_value=True)
    @patch("bitnet_engine.DEV_STDIN")
    @patch("subprocess.run")
    def test_embed_bitnet_tempfile_fallback(self, mock_run, mock_stdin, _mock_avail):
        mock_stdin.exists.return_value = False
        mock_res = MagicMock()
        mock_res.returncode = 0
        dummy_vec = [0.1] * 640
        mock_res.stdout = f"[[{','.join(map(str, dummy_vec))}]]"
        mock_res.stderr = ""
        mock_run.return_value = mock_res

        vec, lat = bitnet_engine.embed_bitnet_sync("Fallback prompt")
        self.assertEqual(len(vec), 640)
        args, kwargs = mock_run.call_args
        cmd = args[0]
        # On tempfile fallback, -f points to a temporary file path, not /dev/stdin
        self.assertIn("-f", cmd)
        f_idx = cmd.index("-f")
        self.assertNotEqual(cmd[f_idx + 1], "/dev/stdin")
        self.assertEqual(kwargs.get("encoding"), "utf-8")

    @patch("bitnet_engine.is_bitnet_embed_available", return_value=True)
    @patch("subprocess.run")
    def test_embed_bitnet_context_overflow_detection(self, mock_run, _mock_avail):
        mock_res = MagicMock()
        mock_res.returncode = -11
        mock_res.stdout = ""
        mock_res.stderr = "batch_decode: n_tokens = 368, n_seq = 1"
        mock_run.return_value = mock_res

        with self.assertRaises(bitnet_engine.ContextOverflowError) as ctx:
            bitnet_engine.embed_bitnet_sync("Excessively long statute clause")
        self.assertIn("exceeds the maximum context capacity", str(ctx.exception))

    @patch("bitnet_engine.is_bitnet_embed_available", return_value=True)
    @patch("bitnet_engine.count_embed_tokens", return_value=300)
    @patch("subprocess.run")
    def test_embed_bitnet_precheck_token_overflow(self, mock_run, mock_count, _mock_avail):
        long_prompt = "Federal reserve regulation " * 10
        with self.assertRaises(bitnet_engine.ContextOverflowError) as ctx:
            bitnet_engine.embed_bitnet_sync(long_prompt)
        self.assertIn("300 tokens", str(ctx.exception))
        self.assertIn("exceeds the maximum supported context limit", str(ctx.exception))
        mock_run.assert_not_called()

    @patch("bitnet_engine.is_bitnet_embed_available", return_value=True)
    @patch("bitnet_engine.count_embed_tokens", return_value=None)
    @patch("subprocess.run")
    def test_embed_bitnet_precheck_tokenizer_unavailable_fails_closed(self, mock_run, mock_count, _mock_avail):
        """When byte length exceeds safe bound and tokenizer is unavailable, service must fail closed."""
        long_prompt = "Federal reserve regulation " * 10
        with self.assertRaises(bitnet_engine.TokenizerUnavailableError) as ctx:
            bitnet_engine.embed_bitnet_sync(long_prompt)
        self.assertIn("Unable to verify token count", str(ctx.exception))
        mock_run.assert_not_called()

    @patch("bitnet_engine.is_bitnet_embed_available", return_value=True)
    @patch("bitnet_engine.count_embed_tokens")
    @patch("subprocess.run")
    def test_embed_bitnet_short_prompt_skips_tokenizer(self, mock_run, mock_count, _mock_avail):
        """Prompts with <= 238 UTF-8 bytes mathematically cannot exceed 238 tokens; skip tokenizer."""
        dummy_vec = [0.1] * 640
        mock_res = MagicMock()
        mock_res.returncode = 0
        mock_res.stdout = f"[[{','.join(map(str, dummy_vec))}]]"
        mock_run.return_value = mock_res

        short_prompt = "Short query under 238 bytes."
        vec, _ = bitnet_engine.embed_bitnet_sync(short_prompt)
        self.assertEqual(len(vec), 640)
        mock_count.assert_not_called()

    @patch("bitnet_engine.is_bitnet_embed_available", return_value=True)
    @patch("asyncio.create_subprocess_exec")
    def test_embed_bitnet_async_success(self, mock_exec, _mock_avail):
        dummy_vec = [0.1] * 640
        sample_output = f"some log text\n[[{','.join(map(str, dummy_vec))}]]\nmore logs"
        mock_proc = MagicMock()
        mock_proc.returncode = 0

        async def _mock_communicate(*args, **kwargs):
            return sample_output.encode("utf-8"), b""

        mock_proc.communicate = _mock_communicate
        mock_exec.return_value = mock_proc

        async def run_async():
            return await bitnet_engine.embed_bitnet("Async statutory clause")

        vec, lat = asyncio.run(run_async())
        self.assertEqual(len(vec), 640)
        self.assertGreater(lat, 0.0)

        exec_args = mock_exec.call_args[0]
        self.assertNotIn("-p", exec_args)
        self.assertTrue(all("Async statutory clause" not in str(arg) for arg in exec_args))

    def test_parse_embed_output_valid(self):
        dummy_vec = [0.1] * 640
        raw_out = f"[[{','.join(map(str, dummy_vec))}]]"
        vec, ms = bitnet_engine._parse_embed_output(raw_out, 120.0)
        self.assertEqual(len(vec), 640)

    def test_parse_embed_output_multiple_vectors_rejected(self):
        """Ensure multi-line input returning multiple vectors is rejected rather than quietly truncated."""
        dummy_vec1 = [0.1] * 640
        dummy_vec2 = [0.2] * 640
        raw_out = f"[[{','.join(map(str, dummy_vec1))}], [{','.join(map(str, dummy_vec2))}]]"
        with self.assertRaises(bitnet_engine.BitNetExecutionError) as ctx:
            bitnet_engine._parse_embed_output(raw_out, 120.0)
        self.assertIn("Expected single embedding vector, got 2 vectors", str(ctx.exception))

    def test_parse_embed_output_invalid_length(self):
        dummy_vec = [0.1] * 128
        raw_out = f"[[{','.join(map(str, dummy_vec))}]]"
        with self.assertRaises(bitnet_engine.BitNetExecutionError) as ctx:
            bitnet_engine._parse_embed_output(raw_out, 120.0)
        self.assertIn("Expected embedding dimension 640, got 128", str(ctx.exception))

    def test_parse_embed_output_no_json(self):
        raw_out = "[system: booting up]"
        with self.assertRaises(bitnet_engine.BitNetExecutionError) as ctx:
            bitnet_engine._parse_embed_output(raw_out, 120.0)
        self.assertIn("No JSON array found in output", str(ctx.exception))


class TestAppRoutingEmbedding(unittest.TestCase):
    @patch("app.is_bitnet_available", return_value=True)
    @patch("app.get_bitnet_status")
    @patch("app.embed_bitnet_sync")
    def test_embed_text_success(self, mock_embed, mock_status, _mock_avail):
        mock_status.return_value = {"embedAvailable": True}
        mock_embed.return_value = ([0.5] * 640, 450.0)

        req = app.EmbedRequest(prompt="This is a test.")
        res = app.embed_text(req)
        self.assertNotIn("prompt", res)
        self.assertEqual(len(res["embedding"]), 640)
        self.assertEqual(res["latencyMs"], 450.0)

    def test_embed_text_prompt_length_validation(self):
        from pydantic import ValidationError
        # Prompt exceeding 2048 characters should fail validation (HTTP 422 in FastAPI)
        with self.assertRaises(ValidationError):
            app.EmbedRequest(prompt="x" * 2049)
        # Empty prompt should fail validation
        with self.assertRaises(ValidationError):
            app.EmbedRequest(prompt="")
        # Prompt within bounds succeeds
        req = app.EmbedRequest(prompt="x" * 500)
        self.assertEqual(len(req.prompt), 500)

    @patch("app.is_bitnet_available", return_value=True)
    @patch("app.get_bitnet_status")
    @patch("app.embed_bitnet_sync")
    def test_embed_text_context_overflow_maps_to_413(self, mock_embed, mock_status, _mock_avail):
        from fastapi import HTTPException
        mock_status.return_value = {"embedAvailable": True}
        mock_embed.side_effect = bitnet_engine.ContextOverflowError("Input prompt token length exceeds the maximum context capacity for the 1-bit embedding engine.")

        req = app.EmbedRequest(prompt="Statute text causing context overflow")
        with self.assertRaises(HTTPException) as ctx:
            app.embed_text(req)
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn("exceeds the maximum context capacity", ctx.exception.detail)

    @patch("app.is_bitnet_available", return_value=True)
    @patch("app.get_bitnet_status")
    @patch("app.embed_bitnet_sync")
    def test_embed_text_tokenizer_unavailable_maps_to_503(self, mock_embed, mock_status, _mock_avail):
        from fastapi import HTTPException
        mock_status.return_value = {"embedAvailable": True}
        mock_embed.side_effect = bitnet_engine.TokenizerUnavailableError("Unable to verify token count for prompt exceeding safe byte bound")

        req = app.EmbedRequest(prompt="Long statute text needing verification")
        with self.assertRaises(HTTPException) as ctx:
            app.embed_text(req)
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("Token verification service is unavailable", ctx.exception.detail)

    @patch("app.is_bitnet_available", return_value=True)
    @patch("app.get_bitnet_status")
    @patch("app.embed_bitnet_sync")
    def test_embed_text_execution_error_sanitized_500(self, mock_embed, mock_status, _mock_avail):
        from fastapi import HTTPException
        mock_status.return_value = {"embedAvailable": True}
        mock_embed.side_effect = bitnet_engine.BitNetExecutionError("Subprocess failed with code 1")

        req = app.EmbedRequest(prompt="Failure prompt")
        with self.assertRaises(HTTPException) as ctx:
            app.embed_text(req)
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(ctx.exception.detail, "Failed to generate embedding vector.")

    @patch("app.is_bitnet_available", return_value=True)
    @patch("app.get_bitnet_status")
    @patch("app.embed_bitnet_sync")
    def test_embed_text_timeout_maps_to_504(self, mock_embed, mock_status, _mock_avail):
        from fastapi import HTTPException
        mock_status.return_value = {"embedAvailable": True}
        mock_embed.side_effect = TimeoutError("Timed out after 60s")

        req = app.EmbedRequest(prompt="Timeout prompt")
        with self.assertRaises(HTTPException) as ctx:
            app.embed_text(req)
        self.assertEqual(ctx.exception.status_code, 504)
        self.assertEqual(ctx.exception.detail, "BitNet embedding request timed out.")

    @patch("app.is_bitnet_available", return_value=True)
    @patch("app.get_bitnet_status")
    @patch("app.embed_bitnet_sync")
    def test_embed_text_generic_error_sanitized(self, mock_embed, mock_status, _mock_avail):
        from fastapi import HTTPException
        mock_status.return_value = {"embedAvailable": True}
        mock_embed.side_effect = RuntimeError("/internal/host/path/failure.cpp:42 segmentation violation")

        req = app.EmbedRequest(prompt="Failure prompt")
        with self.assertRaises(HTTPException) as ctx:
            app.embed_text(req)
        self.assertEqual(ctx.exception.status_code, 500)
        self.assertEqual(ctx.exception.detail, "Failed to generate embedding vector.")
        self.assertNotIn("segmentation violation", ctx.exception.detail)
        self.assertNotIn("/internal/host/path", ctx.exception.detail)

    def test_trusted_host_middleware(self):
        from fastapi.testclient import TestClient
        client = TestClient(app.app)
        # Allowed host should not be rejected with 400
        res = client.get("/healthz", headers={"Host": "127.0.0.1"})
        self.assertEqual(res.status_code, 200)

        # Untrusted host should be rejected with 400 Bad Request
        res_bad = client.get("/healthz", headers={"Host": "malicious-site.com"})
        self.assertEqual(res_bad.status_code, 400)

    def test_api_key_auth_middleware(self):
        from fastapi.testclient import TestClient
        client = TestClient(app.app)

        with patch.object(app, "FTAAS_API_KEY", "secret-test-key"):
            # 1. Healthz is exempt even without token
            res_health = client.get("/healthz", headers={"Host": "127.0.0.1"})
            self.assertEqual(res_health.status_code, 200)

            # 2. Protected endpoint without token returns 401
            res_no_token = client.post("/api/v1/inference/embed", json={"prompt": "hi"}, headers={"Host": "127.0.0.1"})
            self.assertEqual(res_no_token.status_code, 401)
            self.assertEqual(res_no_token.json(), {"detail": "Unauthorized"})

            # 3. Protected endpoint with invalid token returns 401
            res_wrong = client.post("/api/v1/inference/embed", json={"prompt": "hi"}, headers={"Host": "127.0.0.1", "Authorization": "Bearer wrong-key"})
            self.assertEqual(res_wrong.status_code, 401)

            # 4. Protected endpoint with valid Bearer token passes auth
            with patch("app.embed_bitnet_sync", return_value=([0.1] * 640, 100.0)), \
                 patch("app.is_bitnet_available", return_value=True), \
                 patch("app.get_bitnet_status", return_value={"embedAvailable": True}):
                res_auth = client.post("/api/v1/inference/embed", json={"prompt": "hi"}, headers={"Host": "127.0.0.1", "Authorization": "Bearer secret-test-key"})
                self.assertEqual(res_auth.status_code, 200)

            # 5. Protected endpoint with valid X-API-Key passes auth
            with patch("app.embed_bitnet_sync", return_value=([0.1] * 640, 100.0)), \
                 patch("app.is_bitnet_available", return_value=True), \
                 patch("app.get_bitnet_status", return_value={"embedAvailable": True}):
                res_key = client.post("/api/v1/inference/embed", json={"prompt": "hi"}, headers={"Host": "127.0.0.1", "X-API-Key": "secret-test-key"})
                self.assertEqual(res_key.status_code, 200)

            # 6. Non-ASCII Authorization token returns 401 cleanly without raising TypeError
            mock_req = MagicMock()
            mock_req.method = "POST"
            mock_req.url.path = "/api/v1/inference/embed"
            mock_req.headers = {"Authorization": "Bearer key-with-accent-é"}
            mock_next = AsyncMock()
            res_unicode = asyncio.run(app.verify_api_key_if_configured(mock_req, mock_next))
            self.assertEqual(res_unicode.status_code, 401)

            # 7. CORS preflight (OPTIONS) without Authorization header succeeds
            res_opt = client.options("/api/v1/inference/embed", headers={
                "Host": "127.0.0.1",
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Authorization,Content-Type"
            })
            self.assertEqual(res_opt.status_code, 200)
            self.assertEqual(res_opt.headers.get("access-control-allow-origin"), "http://localhost:3000")

    def test_healthz_telemetry_sanitized(self):
        res = app.healthz()
        bitnet_info = res["bitnet"]
        self.assertNotIn("cliPath", bitnet_info)
        self.assertNotIn("modelPath", bitnet_info)
        self.assertNotIn("embedCliPath", bitnet_info)
        self.assertNotIn("embedModelPath", bitnet_info)
        self.assertIn("available", bitnet_info)
        self.assertIn("embedAvailable", bitnet_info)
        self.assertIn("threads", bitnet_info)
        self.assertIn("cachedAdapterCount", res)
        self.assertNotIn("cachedAdapters", res)

    def test_cors_origin_restriction(self):
        from fastapi.testclient import TestClient
        client = TestClient(app.app)
        # Allowed origin
        res_allowed = client.options(
            "/healthz",
            headers={"Origin": "http://localhost:3000", "Access-Control-Request-Method": "GET", "Host": "127.0.0.1"}
        )
        self.assertEqual(res_allowed.headers.get("access-control-allow-origin"), "http://localhost:3000")

        # Disallowed origin
        res_blocked = client.options(
            "/healthz",
            headers={"Origin": "http://malicious-site.com", "Access-Control-Request-Method": "GET", "Host": "127.0.0.1"}
        )
        self.assertNotEqual(res_blocked.headers.get("access-control-allow-origin"), "http://malicious-site.com")


@unittest.skipUnless(
    bitnet_engine.is_bitnet_embed_available(),
    "BitNet embedding binary and weights not present on host"
)
class TestBitNetIntegrationLive(unittest.TestCase):
    """Opt-in live integration tests running directly against the native BitNet binary."""

    def test_live_multiline_cosine_similarity(self):
        vec_a, _ = bitnet_engine.embed_bitnet_sync("alpha")
        vec_b, _ = bitnet_engine.embed_bitnet_sync("alpha\nbeta")
        self.assertEqual(len(vec_a), 640)
        self.assertEqual(len(vec_b), 640)
        cosine = sum(x * y for x, y in zip(vec_a, vec_b))
        # Ensure line 2 contributes to vector (cosine must be strictly < 0.95)
        self.assertLess(cosine, 0.95)

    def test_live_unicode_statutory_text(self):
        statute = "§ 229.10(c)(1)(vi) — “Next-day availability” exception for $5,525."
        vec_sync, lat_sync = bitnet_engine.embed_bitnet_sync(statute)
        self.assertEqual(len(vec_sync), 640)
        self.assertGreater(lat_sync, 0.0)

        vec_async, lat_async = asyncio.run(bitnet_engine.embed_bitnet(statute))
        self.assertEqual(len(vec_async), 640)
        self.assertGreater(lat_async, 0.0)

    def test_live_over_256_token_input_returns_413_or_overflow_error(self):
        """Verify that prompts exceeding the 256-token AVX2 ceiling are safely rejected by pre-check with token count."""
        long_prompt = ("12 CFR § 229.10(c) Next-day availability requirements for deposit accounts. " * 35)[:2000]
        # Direct sync call raises ContextOverflowError with token count
        with self.assertRaises(bitnet_engine.ContextOverflowError) as ctx:
            bitnet_engine.embed_bitnet_sync(long_prompt)
        self.assertIn("642 tokens", str(ctx.exception))

        # HTTP endpoint raises 413 with token count in detail
        from fastapi import HTTPException
        req = app.EmbedRequest(prompt=long_prompt)
        with self.assertRaises(HTTPException) as ctx:
            app.embed_text(req)
        self.assertEqual(ctx.exception.status_code, 413)
        self.assertIn("642 tokens", ctx.exception.detail)

    def test_live_token_boundary_239_240_241(self):
        """Boundary verification at 239, 240, and 241 tokens confirming the 16-token safe margin."""
        text_239 = ("word " * 237).strip()
        text_240 = ("word " * 238).strip()
        text_241 = ("word " * 239).strip()

        self.assertEqual(bitnet_engine.count_embed_tokens(text_239), 239)
        self.assertEqual(bitnet_engine.count_embed_tokens(text_240), 240)
        self.assertEqual(bitnet_engine.count_embed_tokens(text_241), 241)

        # 239 and 240 tokens succeed
        vec239, _ = bitnet_engine.embed_bitnet_sync(text_239)
        self.assertEqual(len(vec239), 640)

        vec240, _ = bitnet_engine.embed_bitnet_sync(text_240)
        self.assertEqual(len(vec240), 640)

        # 241 tokens is rejected by pre-check
        with self.assertRaises(bitnet_engine.ContextOverflowError) as ctx:
            bitnet_engine.embed_bitnet_sync(text_241)
        self.assertIn("241 tokens", str(ctx.exception))

    @unittest.skipUnless(
        bitnet_engine.is_bitnet_available(),
        "BitNet 2B generation binary and weights not present on host"
    )
    def test_live_generation_long_context_does_not_hit_ceiling(self):
        """Verify that BitNet 2B generation path does not hit the 256-token AVX2 ceiling with >= 600 tokens."""
        long_prompt = "Federal Trade Commission Act Section 5 compliance requirement. " * 65
        ans, lat = bitnet_engine.generate_bitnet_sync(long_prompt, max_tokens=10)
        self.assertGreater(len(ans), 0)
        self.assertTrue(any(c.isalnum() for c in ans))
        self.assertGreater(lat, 0.0)

    @unittest.skipUnless(
        sys.platform == "darwin" and Path("/usr/bin/sandbox-exec").is_file(),
        "macOS sandbox-exec required for kernel sandbox negative control test"
    )
    def test_darwin_sandbox_network_denial_negative_control(self):
        """Verify that macOS sandbox profile strictly denies outbound network socket creation (negative control)."""
        import subprocess
        sandbox_cmd = [
            "/usr/bin/sandbox-exec",
            "-p",
            "(version 1)(allow default)(deny network*)",
            "curl",
            "-s",
            "-m",
            "2",
            "https://example.com"
        ]
        res = subprocess.run(sandbox_cmd, capture_output=True, text=True)
        self.assertNotEqual(res.returncode, 0)

        with patch.object(bitnet_engine, "BITNET_SANDBOX_NETWORK_DENY", True):
            if bitnet_engine.is_bitnet_embed_available():
                vec, _ = bitnet_engine.embed_bitnet_sync("Sandboxed embedding test")
                self.assertEqual(len(vec), 640)


if __name__ == "__main__":
    unittest.main()
