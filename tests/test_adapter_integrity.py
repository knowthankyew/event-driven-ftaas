import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
from fastapi import HTTPException

# Ensure inference directory is in sys.path
inference_dir = Path(__file__).resolve().parent.parent / "src" / "FtaaSService.Inference"
sys.path.insert(0, str(inference_dir))

import app


class TestAdapterIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)
        self.orig_data_root = app.DATA_ROOT
        app.DATA_ROOT = self.data_root
        app.model_store.adapter_cache.clear()

    def tearDown(self):
        app.DATA_ROOT = self.orig_data_root
        app.model_store.adapter_cache.clear()
        self.temp_dir.cleanup()

    def test_truncated_config_rejected_by_production_loader(self):
        """Exercises app.load_or_get_adapter() asserting it rejects truncated configs with HTTP 422."""
        adapter_rel = "artifacts/job-truncated-config/model_adapters"
        adapter_dir = self.data_root / adapter_rel
        adapter_dir.mkdir(parents=True, exist_ok=True)

        # Zero-byte config
        (adapter_dir / "adapter_config.json").write_text("")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"x" * (150 * 1024))

        with self.assertRaises(HTTPException) as ctx:
            app.load_or_get_adapter(adapter_rel)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("missing or zero-byte adapter_config.json", str(ctx.exception.detail))

    def test_truncated_weights_rejected_by_production_loader(self):
        """Exercises app.load_or_get_adapter() asserting it rejects <100KB weights with HTTP 422."""
        adapter_rel = "artifacts/job-truncated-weights/model_adapters"
        adapter_dir = self.data_root / adapter_rel
        adapter_dir.mkdir(parents=True, exist_ok=True)

        (adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": app.DEFAULT_BASE_MODEL,
            "peft_type": "LORA"
        }))
        # Only 500 bytes (incomplete/corrupted write)
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"x" * 500)

        with self.assertRaises(HTTPException) as ctx:
            app.load_or_get_adapter(adapter_rel)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("truncated (<100KB)", str(ctx.exception.detail))

    def test_missing_adapter_raises_404(self):
        """Exercises app.load_or_get_adapter() asserting it returns HTTP 404 for missing directories."""
        with self.assertRaises(HTTPException) as ctx:
            app.load_or_get_adapter("artifacts/non-existent-job/model_adapters")
        self.assertEqual(ctx.exception.status_code, 404)

    @patch("app.PeftModel.from_pretrained")
    def test_lru_cache_eviction_on_production_model_store(self, mock_from_pretrained):
        """Validates that app.model_store.adapter_cache bounds entries to MAX_CACHED_ADAPTERS and evicts LRU."""
        mock_model = MagicMock()
        mock_model.to.return_value = mock_model
        mock_model.eval.return_value = mock_model
        mock_from_pretrained.return_value = mock_model

        capacity = app.MAX_CACHED_ADAPTERS
        # Create capacity + 1 valid adapter directories
        for i in range(1, capacity + 2):
            rel = f"artifacts/job-{i}/model_adapters"
            ad = self.data_root / rel
            ad.mkdir(parents=True, exist_ok=True)
            (ad / "adapter_config.json").write_text(json.dumps({
                "base_model_name_or_path": app.DEFAULT_BASE_MODEL,
                "peft_type": "LORA"
            }))
            (ad / "adapter_model.safetensors").write_bytes(b"A" * (120 * 1024))

        # Mount capacity adapters (filling cache capacity)
        for i in range(1, capacity + 1):
            app.load_or_get_adapter(f"artifacts/job-{i}/model_adapters")
        self.assertEqual(len(app.model_store.adapter_cache), capacity)
        self.assertIn("artifacts/job-1/model_adapters", app.model_store.adapter_cache)

        # Access adapter 1 so it becomes most recently used
        app.load_or_get_adapter("artifacts/job-1/model_adapters")

        # Mount (capacity + 1)th adapter -> should evict adapter 2 (since adapter 1 was accessed recently)
        app.load_or_get_adapter(f"artifacts/job-{capacity + 1}/model_adapters")
        self.assertEqual(len(app.model_store.adapter_cache), capacity)
        self.assertNotIn("artifacts/job-2/model_adapters", app.model_store.adapter_cache)
        self.assertIn("artifacts/job-1/model_adapters", app.model_store.adapter_cache)
        self.assertIn(f"artifacts/job-{capacity + 1}/model_adapters", app.model_store.adapter_cache)


if __name__ == "__main__":
    unittest.main()
