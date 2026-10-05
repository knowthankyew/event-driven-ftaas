import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add worker directory to sys.path
worker_dir = Path(__file__).resolve().parent.parent / "src" / "FtaaSService.Worker"
sys.path.insert(0, str(worker_dir))


class TestTrainerModelSelection(unittest.TestCase):
    """
    Verifies that trainer.py correctly selects LoRA target_modules and
    chat template based on the base_model_name parameter, without running
    actual model loading or training.
    """

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)
        # Write a minimal JSONL dataset
        self.dataset_dir = self.data_root / "datasets"
        self.dataset_dir.mkdir(parents=True)
        self.dataset_file = self.dataset_dir / "test.jsonl"
        records = [
            {"prompt": "What is the ACH policy?", "completion": "Under Reg CC, transfers over $10k require 3-5 days."},
            {"prompt": "Is this investment safe?", "completion": "We do not offer investment advice. Tag: [SEC-NO-ADVISORY]."},
        ]
        with open(self.dataset_file, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run_trainer_with_mocks(self, base_model_name: str) -> dict:
        """
        Runs train_job() with all heavy dependencies mocked.
        Returns dict with captured lora_target_modules and formatted_texts.
        """
        captured = {}

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token_id = 0

        mock_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_peft_model.print_trainable_parameters = MagicMock()

        mock_train_result = MagicMock()
        mock_train_result.training_loss = 0.42
        mock_trainer = MagicMock()
        mock_trainer.train.return_value = mock_train_result

        def capture_lora_config(**kwargs):
            captured["lora_target_modules"] = kwargs.get("target_modules")
            mock_cfg = MagicMock()
            return mock_cfg

        import trainer as trainer_module
        import model_registry

        # Capture formatted texts by wrapping the real function
        formatted_calls = []
        original_format_fn = model_registry.format_training_prompt

        def capturing_format(model_id, prompt, completion):
            result = original_format_fn(model_id, prompt, completion)
            formatted_calls.append(result)
            return result

        # Build a mock adapter dir so integrity checks pass
        adapter_dir = self.data_root / "artifacts" / "test-job-001" / "model_adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps({"base_model_name_or_path": base_model_name, "peft_type": "LORA"})
        )
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"W" * (150 * 1024))

        mock_peft_model.save_pretrained = MagicMock()
        mock_tokenizer.save_pretrained = MagicMock()
        # Make save_pretrained write the files the integrity check expects
        def fake_peft_save(path, **kwargs):
            p = Path(path)
            p.mkdir(parents=True, exist_ok=True)
            (p / "adapter_config.json").write_text(
                json.dumps({"base_model_name_or_path": base_model_name})
            )
            (p / "adapter_model.safetensors").write_bytes(b"W" * (150 * 1024))
        def fake_tok_save(path, **kwargs):
            pass
        mock_peft_model.save_pretrained.side_effect = fake_peft_save
        mock_tokenizer.save_pretrained.side_effect = fake_tok_save

        # Build a stub tokenized dataset that Dataset.map would return
        from datasets import Dataset as HFDataset
        stub_tokenized = HFDataset.from_dict({
            "input_ids": [[1, 2, 3, 4] * 64] * 2,
            "attention_mask": [[1] * 256] * 2,
            "labels": [[1, 2, 3, 4] * 64] * 2,
        })

        with patch.object(trainer_module, "DATA_ROOT", self.data_root), \
             patch.object(trainer_module, "ARTIFACTS_ROOT", self.data_root / "artifacts"), \
             patch("trainer.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("trainer.AutoModelForCausalLM.from_pretrained", return_value=mock_model), \
             patch("trainer.LoraConfig", side_effect=capture_lora_config), \
             patch("trainer.get_peft_model", return_value=mock_peft_model), \
             patch("trainer.Trainer", return_value=mock_trainer), \
             patch("trainer.mlflow") as mock_mlflow, \
             patch("datasets.Dataset.map", return_value=stub_tokenized), \
             patch("trainer.format_training_prompt", side_effect=capturing_format):

            mock_run = MagicMock()
            mock_run.__enter__ = MagicMock(return_value=mock_run)
            mock_run.__exit__ = MagicMock(return_value=False)
            mock_run.info.run_id = "test-run-id"
            mock_run.info.experiment_id = "test-exp-id"
            mock_mlflow.start_run.return_value = mock_run

            from trainer import train_job
            train_job(
                job_id="test-job-001",
                job_name="test-job",
                base_model_name=base_model_name,
                dataset_path=str(self.dataset_file.relative_to(self.data_root)),
                dataset_hash="abc123",
                hyperparameters={
                    "epochs": 1,
                    "batchSize": 2,
                    "learningRate": 0.0003,
                    "loraRank": 4,
                    "loraAlpha": 8,
                },
            )
            captured["formatted_texts"] = formatted_calls

        return captured

    # ------------------------------------------------------------------
    # LoRA target_modules selection
    # ------------------------------------------------------------------

    def test_smollm2_uses_correct_lora_targets(self):
        captured = self._run_trainer_with_mocks("HuggingFaceTB/SmolLM2-135M")
        targets = captured.get("lora_target_modules", [])
        self.assertIn("q_proj", targets)
        self.assertIn("v_proj", targets)
        self.assertNotIn("k_proj", targets)
        self.assertNotIn("o_proj", targets)

    def test_gemma_uses_correct_lora_targets(self):
        captured = self._run_trainer_with_mocks("google/gemma-2-2b-it")
        targets = captured.get("lora_target_modules", [])
        self.assertIn("q_proj", targets)
        self.assertIn("v_proj", targets)
        self.assertIn("k_proj", targets)
        self.assertIn("o_proj", targets)

    def test_bitnet_uses_correct_lora_targets(self):
        captured = self._run_trainer_with_mocks("microsoft/BitNet-b1.58-2B-4T")
        targets = captured.get("lora_target_modules", [])
        self.assertIn("q_proj", targets)
        self.assertIn("v_proj", targets)
        self.assertIn("k_proj", targets)
        self.assertIn("o_proj", targets)

    # ------------------------------------------------------------------
    # Chat template selection
    # ------------------------------------------------------------------

    def test_smollm2_uses_chatml_template(self):
        captured = self._run_trainer_with_mocks("HuggingFaceTB/SmolLM2-135M")
        texts = captured.get("formatted_texts", [])
        self.assertGreater(len(texts), 0)
        for text in texts:
            self.assertIn("<|im_start|>user", text)
            self.assertIn("<|im_start|>assistant", text)
            self.assertNotIn("<start_of_turn>", text)

    def test_gemma_uses_gemma_template(self):
        captured = self._run_trainer_with_mocks("google/gemma-2-2b-it")
        texts = captured.get("formatted_texts", [])
        self.assertGreater(len(texts), 0)
        for text in texts:
            self.assertIn("<start_of_turn>user", text)
            self.assertIn("<start_of_turn>model", text)
            self.assertNotIn("<|im_start|>", text)

    def test_bitnet_uses_bitnet_template(self):
        captured = self._run_trainer_with_mocks("microsoft/BitNet-b1.58-2B-4T")
        texts = captured.get("formatted_texts", [])
        self.assertGreater(len(texts), 0)
        for text in texts:
            self.assertIn("User:", text)
            self.assertIn("Assistant:", text)
            self.assertIn("<|eot_id|>", text)
            self.assertNotIn("<|im_start|>", text)
            self.assertNotIn("<start_of_turn>", text)

    # ------------------------------------------------------------------
    # HF auth error handling
    # ------------------------------------------------------------------

    def test_hf_auth_error_on_tokenizer_raises_value_error_with_hint(self):
        """A 401 OSError from HF tokenizer load should become a ValueError with HF_TOKEN hint."""
        import trainer as trainer_module
        auth_error = OSError(
            "401 Client Error: Unauthorized for url: https://huggingface.co/google/gemma-2-2b-it"
        )

        with patch.object(trainer_module, "DATA_ROOT", self.data_root), \
             patch.object(trainer_module, "ARTIFACTS_ROOT", self.data_root / "artifacts"), \
             patch("trainer.AutoTokenizer.from_pretrained", side_effect=auth_error), \
             patch("trainer.mlflow") as mock_mlflow:

            mock_run = MagicMock()
            mock_run.__enter__ = MagicMock(return_value=mock_run)
            mock_run.__exit__ = MagicMock(return_value=False)
            mock_run.info.run_id = "test-run-auth"
            mock_run.info.experiment_id = "test-exp-auth"
            mock_mlflow.start_run.return_value = mock_run

            from trainer import train_job
            with self.assertRaises(ValueError) as ctx:
                train_job(
                    job_id="test-job-auth",
                    job_name="test-auth",
                    base_model_name="google/gemma-2-2b-it",
                    dataset_path=str(self.dataset_file.relative_to(self.data_root)),
                    dataset_hash="abc123",
                    hyperparameters={"epochs": 1, "batchSize": 2, "learningRate": 0.0003},
                )
            self.assertIn("HF_TOKEN", str(ctx.exception))
            self.assertIn("authentication", str(ctx.exception))

    def test_hf_auth_error_on_model_raises_value_error_with_hint(self):
        """A 403/gated OSError from model load should become a ValueError with HF_TOKEN hint."""
        import trainer as trainer_module
        auth_error = OSError(
            "403 Client Error: This model is gated. Please accept the license."
        )

        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer.pad_token_id = 0

        with patch.object(trainer_module, "DATA_ROOT", self.data_root), \
             patch.object(trainer_module, "ARTIFACTS_ROOT", self.data_root / "artifacts"), \
             patch("trainer.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
             patch("trainer.AutoModelForCausalLM.from_pretrained", side_effect=auth_error), \
             patch("trainer.mlflow") as mock_mlflow:

            mock_run = MagicMock()
            mock_run.__enter__ = MagicMock(return_value=mock_run)
            mock_run.__exit__ = MagicMock(return_value=False)
            mock_run.info.run_id = "test-run-auth2"
            mock_run.info.experiment_id = "test-exp-auth2"
            mock_mlflow.start_run.return_value = mock_run

            from trainer import train_job
            with self.assertRaises(ValueError) as ctx:
                train_job(
                    job_id="test-job-auth2",
                    job_name="test-auth2",
                    base_model_name="google/gemma-2-2b-it",
                    dataset_path=str(self.dataset_file.relative_to(self.data_root)),
                    dataset_hash="abc123",
                    hyperparameters={"epochs": 1, "batchSize": 2, "learningRate": 0.0003},
                )
            self.assertIn("HF_TOKEN", str(ctx.exception))


class TestInstructionMasking(unittest.TestCase):
    """Validates that instruction tuning masks prompt tokens and pad tokens as -100 while preserving completion and EOS tokens."""

    def test_prompt_tokens_and_padding_are_masked(self):
        from trainer import create_instruction_mask

        full_ids = [1, 2, 3, 4, 10, 11, 2, 0, 0, 0, 0, 0]
        prompt_ids = [1, 2, 3, 4]
        att_mask = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

        prompt_len = len(prompt_ids)
        # Call the real production function from trainer.py
        seq_labels = create_instruction_mask(full_ids, prompt_len, att_mask)

        # First 4 tokens (prompt) should be -100
        self.assertEqual(seq_labels[:4], [-100, -100, -100, -100])
        # Next 3 tokens (completion + EOS) should retain their IDs
        self.assertEqual(seq_labels[4:7], [10, 11, 2])
        # Trailing padding tokens should be -100
        self.assertEqual(seq_labels[7:], [-100, -100, -100, -100, -100])


if __name__ == "__main__":
    unittest.main()
