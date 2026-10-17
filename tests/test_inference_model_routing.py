"""
test_inference_model_routing.py

Tests model-aware prompt formatting and base model mismatch detection
in the inference service, without loading real model weights.

Uses the inference service's app.py only indirectly — the core logic under
test (format_inference_prompt and mismatch detection via adapter_config.json)
is exercised directly via model_registry and by simulating the mismatch check
that load_or_get_adapter() performs.
"""
import sys
import json
import tempfile
import unittest
from pathlib import Path

# Add worker and inference directories to sys.path
worker_dir = Path(__file__).resolve().parent.parent / "src" / "FtaaSService.Worker"
inference_dir = Path(__file__).resolve().parent.parent / "src" / "FtaaSService.Inference"
sys.path.insert(0, str(worker_dir))
sys.path.insert(0, str(inference_dir))


class TestInferencePromptFormatting(unittest.TestCase):
    """Validates that format_inference_prompt produces the correct open-ended prompt for each model."""

    def test_smollm2_prompt_uses_chatml_format(self):
        from model_registry import format_inference_prompt
        result = format_inference_prompt("HuggingFaceTB/SmolLM2-135M", "Hello world")
        self.assertIn("<|im_start|>user\nHello world<|im_end|>", result)
        self.assertTrue(result.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<start_of_turn>", result)

    def test_gemma_prompt_uses_gemma_format(self):
        from model_registry import format_inference_prompt
        result = format_inference_prompt("google/gemma-2-2b-it", "Hello world")
        self.assertIn("<start_of_turn>user\nHello world<end_of_turn>", result)
        self.assertTrue(result.endswith("<start_of_turn>model\n"))
        self.assertNotIn("<|im_start|>", result)

    def test_bitnet_prompt_uses_bitnet_format(self):
        from model_registry import format_inference_prompt
        result = format_inference_prompt("microsoft/BitNet-b1.58-2B-4T", "Hello world")
        self.assertEqual(result, "User: Hello world<|eot_id|>\nAssistant: ")

    def test_smollm2_prompt_preserves_content(self):
        from model_registry import format_inference_prompt
        prompt = "What is the ACH clearance period for transfers over $10,000?"
        result = format_inference_prompt("HuggingFaceTB/SmolLM2-135M", prompt)
        self.assertIn(prompt, result)

    def test_gemma_prompt_preserves_content(self):
        from model_registry import format_inference_prompt
        prompt = "Analyze: Q3 gross margin expanded 340 bps YoY to 43.1%."
        result = format_inference_prompt("google/gemma-2-2b-it", prompt)
        self.assertIn(prompt, result)

    def test_bitnet_prompt_preserves_content(self):
        from model_registry import format_inference_prompt
        prompt = "Analyze: Q3 gross margin expanded 340 bps YoY to 43.1%."
        result = format_inference_prompt("microsoft/BitNet-b1.58-2B-4T", prompt)
        self.assertIn(prompt, result)


from fastapi import HTTPException
from unittest.mock import MagicMock, patch

class TestBasemodelMismatchDetection(unittest.TestCase):
    """
    Validates base model mismatch detection directly via app.load_or_get_adapter().
    """

    def setUp(self):
        import app
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)
        self.orig_data_root = app.DATA_ROOT
        app.DATA_ROOT = self.data_root
        app.model_store.adapter_cache.clear()

    def tearDown(self):
        import app
        app.DATA_ROOT = self.orig_data_root
        app.model_store.adapter_cache.clear()
        self.temp_dir.cleanup()

    def _make_adapter(self, base_model_name: str, job_id: str = "test-job-001") -> str:
        adapter_rel = f"artifacts/{job_id}/model_adapters"
        adapter_dir = self.data_root / adapter_rel
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": base_model_name,
            "peft_type": "LORA",
            "r": 8,
        }))
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"A" * (150 * 1024))
        return adapter_rel

    def test_matching_base_model_no_mismatch(self):
        """SmolLM2 adapter against SmolLM2-loaded service → no mismatch."""
        import app
        adapter_rel = self._make_adapter("HuggingFaceTB/SmolLM2-135M", "job-smol-match")
        with patch.object(app, "DEFAULT_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M"):
            with patch("app.PeftModel.from_pretrained", return_value=MagicMock()):
                app.load_or_get_adapter(adapter_rel)

    def test_gemma_adapter_against_smollm2_service_is_mismatch(self):
        """Gemma adapter against SmolLM2-loaded service → mismatch detected (409)."""
        import app
        adapter_rel = self._make_adapter("google/gemma-2-2b-it", "job-gemma-mismatch")
        with patch.object(app, "DEFAULT_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M"):
            with self.assertRaises(HTTPException) as ctx:
                app.load_or_get_adapter(adapter_rel)
            self.assertEqual(ctx.exception.status_code, 409)
            self.assertEqual(ctx.exception.detail["error"], "base_model_mismatch")
            self.assertEqual(ctx.exception.detail["adapterBaseModel"], "google/gemma-2-2b-it")

    def test_smollm2_adapter_against_gemma_service_is_mismatch(self):
        """SmolLM2 adapter against Gemma-loaded service → mismatch detected (409)."""
        import app
        adapter_rel = self._make_adapter("HuggingFaceTB/SmolLM2-135M", "job-smol-mismatch")
        with patch.object(app, "DEFAULT_BASE_MODEL", "google/gemma-2-2b-it"):
            with self.assertRaises(HTTPException) as ctx:
                app.load_or_get_adapter(adapter_rel)
            self.assertEqual(ctx.exception.status_code, 409)

    def test_gemma_adapter_against_gemma_service_no_mismatch(self):
        """Gemma adapter against Gemma-loaded service → no mismatch."""
        import app
        adapter_rel = self._make_adapter("google/gemma-2-2b-it", "job-gemma-match")
        with patch.object(app, "DEFAULT_BASE_MODEL", "google/gemma-2-2b-it"):
            with patch("app.PeftModel.from_pretrained", return_value=MagicMock()):
                app.load_or_get_adapter(adapter_rel)

    def test_bitnet_adapter_against_smollm2_service_is_mismatch(self):
        """BitNet adapter against SmolLM2-loaded service → mismatch detected (409)."""
        import app
        adapter_rel = self._make_adapter("microsoft/BitNet-b1.58-2B-4T", "job-bitnet-mismatch")
        with patch.object(app, "DEFAULT_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M"):
            with self.assertRaises(HTTPException) as ctx:
                app.load_or_get_adapter(adapter_rel)
            self.assertEqual(ctx.exception.status_code, 409)
            self.assertEqual(ctx.exception.detail["adapterBaseModel"], "microsoft/BitNet-b1.58-2B-4T")

    def test_bitnet_adapter_against_bitnet_service_no_mismatch(self):
        """BitNet adapter against BitNet-loaded service → no mismatch."""
        import app
        adapter_rel = self._make_adapter("microsoft/BitNet-b1.58-2B-4T", "job-bitnet-match")
        with patch.object(app, "DEFAULT_BASE_MODEL", "microsoft/BitNet-b1.58-2B-4T"):
            with patch("app.PeftModel.from_pretrained", return_value=MagicMock()):
                app.load_or_get_adapter(adapter_rel)

    def test_missing_base_model_field_does_not_trigger_mismatch(self):
        """An adapter_config.json without base_model_name_or_path skips the mismatch check."""
        import app
        adapter_rel = "artifacts/no-base/model_adapters"
        adapter_dir = self.data_root / adapter_rel
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"A" * (150 * 1024))
        with patch.object(app, "DEFAULT_BASE_MODEL", "HuggingFaceTB/SmolLM2-135M"):
            with patch("app.PeftModel.from_pretrained", return_value=MagicMock()):
                app.load_or_get_adapter(adapter_rel)


class TestHealthzChatTemplate(unittest.TestCase):
    """Validates that the healthz chatTemplate field reflects the loaded model's template."""

    def test_smollm2_healthz_chat_template_is_chatml(self):
        from model_registry import get_model_spec
        spec = get_model_spec("HuggingFaceTB/SmolLM2-135M")
        self.assertEqual(spec.chat_template.value, "chatml")

    def test_gemma_healthz_chat_template_is_gemma(self):
        from model_registry import get_model_spec
        spec = get_model_spec("google/gemma-2-2b-it")
        self.assertEqual(spec.chat_template.value, "gemma")

    def test_unknown_model_healthz_uses_unknown_fallback(self):
        """An unknown model ID should fall back to 'unknown' in healthz()."""
        import app
        with patch.object(app, "DEFAULT_BASE_MODEL", "not/a-real-model"):
            res = app.healthz()
            self.assertEqual(res["chatTemplate"], "unknown")
class TestGenerateTokensRouting(unittest.TestCase):
    """Validates that generate_tokens and compare_completions route prompt templates and handle baseModel correctly."""

    def test_generate_tokens_uses_specified_base_model_template(self):
        from unittest.mock import MagicMock, patch
        import app
        import torch

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[101, 102, 103, 104]])
        mock_tok = MagicMock()
        mock_tok.pad_token_id = 0
        mock_tok.eos_token_id = 2
        mock_inputs = MagicMock()
        mock_inputs.__getitem__.side_effect = lambda k: torch.tensor([[101, 102]])
        mock_inputs.to.return_value = mock_inputs
        mock_tok.return_value = mock_inputs
        mock_tok.decode.return_value = "Generated text"

        with patch("app.format_inference_prompt", wraps=app.format_inference_prompt) as mock_format:
            completion, latency = app.generate_tokens(
                mock_model,
                mock_tok,
                prompt="test prompt",
                max_tokens=10,
                temperature=0.0,
                base_model_id="google/gemma-2-2b-it"
            )
            mock_format.assert_called_once_with("google/gemma-2-2b-it", "test prompt")
            self.assertEqual(completion, "Generated text")

    def test_compare_completions_rejects_mismatched_base_model_with_409(self):
        from unittest.mock import MagicMock, patch
        from fastapi import HTTPException
        import app

        req = app.CompareRequest(
            prompt="Hello world",
            baseModel="google/gemma-2-2b-it",
            adapterPath="some/path"
        )
        with patch.object(app.model_store, "base_model", MagicMock()), \
             patch.object(app.model_store, "tokenizer", MagicMock()):
            with self.assertRaises(HTTPException) as ctx:
                app.compare_completions(req)
            self.assertEqual(ctx.exception.status_code, 409)
            self.assertEqual(ctx.exception.detail["error"], "base_model_mismatch")
            self.assertIn("hint", ctx.exception.detail)


if __name__ == "__main__":
    unittest.main()
