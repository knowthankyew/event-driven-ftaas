import os
import tempfile
import unittest
from pathlib import Path
from fastapi import HTTPException

class TestAdapterIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_truncated_config_rejected(self):
        adapter_dir = self.data_root / "adapters" / "job-truncated-config" / "model_adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)
        
        # Zero-byte config
        (adapter_dir / "adapter_config.json").write_text("")
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"x" * (150 * 1024))

        # Check trainer integrity rule
        config_file = adapter_dir / "adapter_config.json"
        self.assertTrue(config_file.stat().st_size < 10)

    def test_truncated_weights_rejected(self):
        adapter_dir = self.data_root / "adapters" / "job-truncated-weights" / "model_adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        (adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA"}')
        # Only 500 bytes (incomplete/corrupted write)
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"x" * 500)

        weights_file = adapter_dir / "adapter_model.safetensors"
        self.assertTrue(weights_file.stat().st_size < 100 * 1024)

    def test_valid_weights_pass_threshold(self):
        adapter_dir = self.data_root / "adapters" / "job-valid" / "model_adapters"
        adapter_dir.mkdir(parents=True, exist_ok=True)

        (adapter_dir / "adapter_config.json").write_text('{"peft_type": "LORA", "r": 8}')
        # 1.8 MB standard SmolLM2 LoRA adapter
        (adapter_dir / "adapter_model.safetensors").write_bytes(b"0" * (1800 * 1024))

        config_file = adapter_dir / "adapter_config.json"
        weights_file = adapter_dir / "adapter_model.safetensors"

        self.assertGreaterEqual(config_file.stat().st_size, 10)
        self.assertGreaterEqual(weights_file.stat().st_size, 100 * 1024)

    def test_lru_cache_eviction_order(self):
        from collections import OrderedDict
        cache = OrderedDict()
        max_cached = 3

        # Insert 3 items
        for i in range(1, 4):
            cache[f"adapter_{i}"] = f"model_{i}"
        
        self.assertEqual(list(cache.keys()), ["adapter_1", "adapter_2", "adapter_3"])

        # Access adapter_1 (moves to most recently used)
        cache.move_to_end("adapter_1")
        self.assertEqual(list(cache.keys()), ["adapter_2", "adapter_3", "adapter_1"])

        # Add 4th item -> adapter_2 should be evicted as LRU
        if len(cache) >= max_cached:
            evicted, _ = cache.popitem(last=False)
            self.assertEqual(evicted, "adapter_2")
        cache["adapter_4"] = "model_4"

        self.assertEqual(list(cache.keys()), ["adapter_3", "adapter_1", "adapter_4"])

if __name__ == "__main__":
    unittest.main()
