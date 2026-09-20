import os
import json
import shutil
import tempfile
import unittest
import sys
from pathlib import Path

# Add worker directory to sys.path
worker_dir = Path(__file__).resolve().parent.parent / "src" / "FtaaSService.Worker"
sys.path.insert(0, str(worker_dir))

from exporter import export_to_edge_onnx, compute_file_sha256

class TestOnnxExporter(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_root = Path(self.temp_dir.name)
        self.artifacts_root = self.data_root / "artifacts"
        self.job_id = "job-test-onnx-export"
        
        # Patch config ARTIFACTS_ROOT
        import exporter
        self.orig_artifacts_root = exporter.ARTIFACTS_ROOT
        exporter.ARTIFACTS_ROOT = self.artifacts_root

        # Create mock valid adapter
        self.adapter_dir = self.artifacts_root / self.job_id / "model_adapters"
        self.adapter_dir.mkdir(parents=True, exist_ok=True)
        
        (self.adapter_dir / "adapter_config.json").write_text(json.dumps({
            "base_model_name_or_path": "HuggingFaceTB/SmolLM2-135M",
            "peft_type": "LORA",
            "r": 8,
            "lora_alpha": 32
        }))
        # Write valid > 100KB mock weights
        (self.adapter_dir / "adapter_model.safetensors").write_bytes(b"A" * (120 * 1024))
        (self.adapter_dir / "tokenizer.json").write_text('{"mock": "tokenizer"}')

    def tearDown(self):
        import exporter
        exporter.ARTIFACTS_ROOT = self.orig_artifacts_root
        self.temp_dir.cleanup()

    def test_missing_adapter_raises_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            export_to_edge_onnx(job_id="non-existent-job", mock_for_test=True)

    def test_truncated_config_raises_runtime_error(self):
        (self.adapter_dir / "adapter_config.json").write_text("")
        with self.assertRaises(RuntimeError):
            export_to_edge_onnx(job_id=self.job_id, mock_for_test=True)

    def test_truncated_weights_raises_runtime_error(self):
        (self.adapter_dir / "adapter_model.safetensors").write_bytes(b"A" * 100)
        with self.assertRaises(RuntimeError):
            export_to_edge_onnx(job_id=self.job_id, mock_for_test=True)

    def test_successful_edge_export_generates_valid_manifest(self):
        out_dir = self.data_root / "custom_edge_out"
        res = export_to_edge_onnx(job_id=self.job_id, output_dir=out_dir, quantize=True, mock_for_test=True)

        self.assertEqual(res["jobId"], self.job_id)
        manifest = res["manifest"]
        
        self.assertEqual(manifest["jobId"], self.job_id)
        self.assertEqual(manifest["baseModel"], "HuggingFaceTB/SmolLM2-135M")
        self.assertEqual(manifest["architecture"], "CausalLM")
        self.assertEqual(manifest["quantization"], "int8")
        self.assertEqual(manifest["supportedExecutionProviders"], ["webgpu", "wasm"])
        self.assertTrue(manifest["zeroEgressInvariant"])
        self.assertIn("onnxSha256", manifest)
        
        # Verify files generated in output directory
        manifest_path = out_dir / "edge_model_manifest.json"
        onnx_path = out_dir / "model.onnx"
        tok_path = out_dir / "tokenizer.json"

        self.assertTrue(manifest_path.exists())
        self.assertTrue(onnx_path.exists())
        self.assertTrue(tok_path.exists())

        # Verify sha256 calculation
        expected_sha = compute_file_sha256(onnx_path)
        self.assertEqual(manifest["onnxSha256"], expected_sha)

    def test_invalid_job_id_raises_value_error(self):
        with self.assertRaises(ValueError):
            export_to_edge_onnx(job_id="../../etc/passwd", mock_for_test=True)
        with self.assertRaises(ValueError):
            export_to_edge_onnx(job_id="job;rm -rf /", mock_for_test=True)
        with self.assertRaises(ValueError):
            export_to_edge_onnx(job_id="", mock_for_test=True)

    def test_unquantized_manifest_specifies_fp32(self):
        out_dir = self.data_root / "fp32_edge_out"
        res = export_to_edge_onnx(job_id=self.job_id, output_dir=out_dir, quantize=False, mock_for_test=True)
        self.assertEqual(res["manifest"]["quantization"], "fp32")

if __name__ == "__main__":
    unittest.main()
