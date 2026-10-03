import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add worker directory to sys.path (same pattern as other test files)
worker_dir = Path(__file__).resolve().parent.parent / "src" / "FtaaSService.Worker"
sys.path.insert(0, str(worker_dir))

from model_registry import (
    get_model_spec,
    format_training_prompt,
    format_inference_prompt,
    preflight_check,
    ChatTemplate,
    SMOLLM2,
    GEMMA_2_2B_IT,
    SUPPORTED_MODEL_IDS,
)


class TestModelRegistrySpecs(unittest.TestCase):

    def test_smollm2_spec_is_registered(self):
        spec = get_model_spec(SMOLLM2)
        self.assertEqual(spec.model_id, "HuggingFaceTB/SmolLM2-135M")
        self.assertEqual(spec.display_name, "SmolLM2-135M")
        self.assertEqual(spec.chat_template, ChatTemplate.CHATML)
        self.assertEqual(spec.lora_target_modules, ["q_proj", "v_proj"])
        self.assertEqual(spec.parameter_count_display, "135M")
        self.assertEqual(spec.context_length, 2048)
        self.assertFalse(spec.requires_hf_auth)
        self.assertIsNone(spec.hardware_disclaimer)

    def test_gemma_spec_is_registered(self):
        spec = get_model_spec(GEMMA_2_2B_IT)
        self.assertEqual(spec.model_id, "google/gemma-2-2b-it")
        self.assertEqual(spec.display_name, "Gemma 2 2B IT")
        self.assertEqual(spec.chat_template, ChatTemplate.GEMMA)
        self.assertIn("q_proj", spec.lora_target_modules)
        self.assertIn("k_proj", spec.lora_target_modules)
        self.assertIn("o_proj", spec.lora_target_modules)
        self.assertEqual(len(spec.lora_target_modules), 4)
        self.assertEqual(spec.parameter_count_display, "2B")
        self.assertEqual(spec.context_length, 8192)
        self.assertTrue(spec.requires_hf_auth)
        self.assertIsNotNone(spec.hardware_disclaimer)
        self.assertIn("8 GB", spec.hardware_disclaimer)

    def test_unknown_model_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            get_model_spec("some/random-model-xyz")
        self.assertIn("Unsupported base model", str(ctx.exception))
        self.assertIn("some/random-model-xyz", str(ctx.exception))

    def test_supported_model_ids_frozenset_contains_both(self):
        self.assertIn(SMOLLM2, SUPPORTED_MODEL_IDS)
        self.assertIn(GEMMA_2_2B_IT, SUPPORTED_MODEL_IDS)
        self.assertEqual(len(SUPPORTED_MODEL_IDS), 2)


class TestChatTemplates(unittest.TestCase):

    def test_smollm2_training_prompt_uses_chatml(self):
        result = format_training_prompt(SMOLLM2, "Hello", "World")
        self.assertIn("<|im_start|>user\nHello<|im_end|>", result)
        self.assertIn("<|im_start|>assistant\nWorld<|im_end|>", result)
        self.assertNotIn("<start_of_turn>", result)

    def test_gemma_training_prompt_uses_gemma_template(self):
        result = format_training_prompt(GEMMA_2_2B_IT, "Hello", "World")
        self.assertIn("<start_of_turn>user\nHello<end_of_turn>", result)
        self.assertIn("<start_of_turn>model\nWorld<end_of_turn>", result)
        self.assertNotIn("<|im_start|>", result)

    def test_smollm2_inference_prompt_open_ended(self):
        result = format_inference_prompt(SMOLLM2, "Hello")
        self.assertTrue(result.endswith("<|im_start|>assistant\n"))
        self.assertNotIn("<end_of_turn>", result)

    def test_gemma_inference_prompt_open_ended(self):
        result = format_inference_prompt(GEMMA_2_2B_IT, "Hello")
        self.assertTrue(result.endswith("<start_of_turn>model\n"))
        self.assertNotIn("<|im_start|>", result)

    def test_training_prompt_preserves_content(self):
        prompt = "What is the ACH clearance policy?"
        completion = "Under Reg CC, transfers over $10k require 3-5 days."
        result = format_training_prompt(SMOLLM2, prompt, completion)
        self.assertIn(prompt, result)
        self.assertIn(completion, result)

    def test_unknown_model_raises_in_format_training(self):
        with self.assertRaises(ValueError):
            format_training_prompt("not/a-real-model", "p", "c")

    def test_unknown_model_raises_in_format_inference(self):
        with self.assertRaises(ValueError):
            format_inference_prompt("not/a-real-model", "p")


class TestPreflightCheck(unittest.TestCase):

    def test_smollm2_passes_on_cpu(self):
        """SmolLM2 has a 0.5 GB VRAM requirement — should always pass on any hardware."""
        with patch("model_registry.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            result = preflight_check(SMOLLM2)
        self.assertTrue(result["passed"])
        self.assertIsNone(result["warning"])

    def test_gemma_warns_on_cpu_without_mps(self):
        """Gemma 2B on CPU-only should emit a slowness warning."""
        with patch("model_registry.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = False
            result = preflight_check(GEMMA_2_2B_IT)
        self.assertFalse(result["passed"])
        self.assertIsNotNone(result["warning"])
        self.assertIn("CPU", result["warning"])

    def test_gemma_warns_insufficient_vram(self):
        """4 GB VRAM should trigger a warning for Gemma 2B (requires 8 GB)."""
        mock_props = MagicMock()
        mock_props.total_memory = int(4 * 1024 ** 3)  # 4 GB
        with patch("model_registry.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_properties.return_value = mock_props
            mock_torch.backends.mps.is_available.return_value = False
            result = preflight_check(GEMMA_2_2B_IT)
        self.assertFalse(result["passed"])
        self.assertIn("OOM", result["warning"])

    def test_gemma_passes_with_sufficient_vram(self):
        """16 GB VRAM + HF_TOKEN set should pass for Gemma 2B."""
        mock_props = MagicMock()
        mock_props.total_memory = int(16 * 1024 ** 3)  # 16 GB
        with patch("model_registry.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = True
            mock_torch.cuda.get_device_properties.return_value = mock_props
            mock_torch.backends.mps.is_available.return_value = False
            with patch.dict(os.environ, {"HF_TOKEN": "fake-token-for-test"}):
                result = preflight_check(GEMMA_2_2B_IT)
        self.assertTrue(result["passed"])
        self.assertIsNone(result["warning"])

    def test_gemma_warns_missing_hf_token(self):
        """Missing HF_TOKEN should produce a warning for gated models."""
        with patch("model_registry.torch") as mock_torch:
            mock_torch.cuda.is_available.return_value = False
            mock_torch.backends.mps.is_available.return_value = True  # MPS available → no CPU warning
            env_without_token = {k: v for k, v in os.environ.items() if k != "HF_TOKEN"}
            with patch.dict(os.environ, env_without_token, clear=True):
                result = preflight_check(GEMMA_2_2B_IT)
        self.assertFalse(result["passed"])
        self.assertIn("HF_TOKEN", result["warning"])

    def test_unknown_model_raises_in_preflight(self):
        with self.assertRaises(ValueError):
            preflight_check("not/a-real-model")


if __name__ == "__main__":
    unittest.main()
