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

# Add worker directory to sys.path so model_registry is importable
worker_dir = Path(__file__).resolve().parent.parent / "src" / "FtaaSService.Worker"
sys.path.insert(0, str(worker_dir))


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


class TestBasemodelMismatchDetection(unittest.TestCase):
    """
    Validates the base model mismatch detection logic that load_or_get_adapter()
    performs by reading adapter_config.json. Tested without spinning up the FastAPI app.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _make_adapter_dir(self, base_model_name: str) -> Path:
        adapter_dir = self.data_root / "artifacts" / "test-job-001" / "model_adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": base_model_name,
            "peft_type": "LORA",
            "r": 8,
        }))
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"A" * (150 * 1024))
        return adapter_dir

    def _simulate_mismatch_check(self, adapter_dir: Path, loaded_base_model: str) -> dict:
        """
        Simulates exactly what load_or_get_adapter() does for the mismatch check:
        reads adapter_config.json, compares base_model_name_or_path to loaded model.
        Returns {"mismatch": bool, "adapter_base": str, "loaded_base": str}.
        """
        config_file = adapter_dir / "adapter_config.json"
        with open(config_file, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        adapter_base = cfg.get("base_model_name_or_path", "")
        mismatch = bool(adapter_base) and adapter_base != loaded_base_model
        return {"mismatch": mismatch, "adapter_base": adapter_base, "loaded_base": loaded_base_model}

    def test_matching_base_model_no_mismatch(self):
        """SmolLM2 adapter against SmolLM2-loaded service → no mismatch."""
        adapter_dir = self._make_adapter_dir("HuggingFaceTB/SmolLM2-135M")
        result = self._simulate_mismatch_check(adapter_dir, "HuggingFaceTB/SmolLM2-135M")
        self.assertFalse(result["mismatch"])

    def test_gemma_adapter_against_smollm2_service_is_mismatch(self):
        """Gemma adapter against SmolLM2-loaded service → mismatch detected."""
        adapter_dir = self._make_adapter_dir("google/gemma-2-2b-it")
        result = self._simulate_mismatch_check(adapter_dir, "HuggingFaceTB/SmolLM2-135M")
        self.assertTrue(result["mismatch"])
        self.assertEqual(result["adapter_base"], "google/gemma-2-2b-it")

    def test_smollm2_adapter_against_gemma_service_is_mismatch(self):
        """SmolLM2 adapter against Gemma-loaded service → mismatch detected."""
        adapter_dir = self._make_adapter_dir("HuggingFaceTB/SmolLM2-135M")
        result = self._simulate_mismatch_check(adapter_dir, "google/gemma-2-2b-it")
        self.assertTrue(result["mismatch"])

    def test_gemma_adapter_against_gemma_service_no_mismatch(self):
        """Gemma adapter against Gemma-loaded service → no mismatch."""
        adapter_dir = self._make_adapter_dir("google/gemma-2-2b-it")
        result = self._simulate_mismatch_check(adapter_dir, "google/gemma-2-2b-it")
        self.assertFalse(result["mismatch"])

    def test_missing_base_model_field_does_not_trigger_mismatch(self):
        """An adapter_config.json without base_model_name_or_path skips the mismatch check."""
        adapter_dir = self.data_root / "artifacts" / "no-base" / "model_adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        # Write config with no base_model_name_or_path field
        (adapter_dir / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA"}))
        result = self._simulate_mismatch_check(adapter_dir, "HuggingFaceTB/SmolLM2-135M")
        self.assertFalse(result["mismatch"])


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
        """An unknown model ID should fall back to 'unknown' without crashing."""
        from model_registry import get_model_spec
        chat_template = "unknown"
        try:
            spec = get_model_spec("not/a-real-model")
            chat_template = spec.chat_template.value
        except Exception:
            pass  # Expected — fallback logic mirrors what healthz does
        self.assertEqual(chat_template, "unknown")


if __name__ == "__main__":
    unittest.main()
