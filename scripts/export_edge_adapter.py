#!/usr/bin/env python3
"""
CLI utility to export a trained FTaaS LoRA adapter and base model to a web-optimized ONNX package.
Usage:
    python scripts/export_edge_adapter.py --job-id <job_id> [--output <path>] [--quantize]
"""
import sys
import argparse
import logging
from pathlib import Path

# Add worker directory to Python path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
WORKER_DIR = PROJECT_ROOT / "src" / "FtaaSService.Worker"
sys.path.insert(0, str(WORKER_DIR))

from exporter import export_to_edge_onnx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("ExportEdgeAdapter")

def main():
    parser = argparse.ArgumentParser(description="Export FTaaS LoRA adapter to browser-ready ONNX package.")
    parser.add_argument("--job-id", required=True, help="Job ID of the trained model adapter.")
    parser.add_argument("--output", default=None, help="Optional destination directory for the ONNX package.")
    parser.add_argument("--quantize", action="store_true", help="Enable q4 quantization flag in manifest.")
    parser.add_argument("--mock", action="store_true", help="Perform fast mock export for testing.")

    args = parser.parse_args()

    out_path = Path(args.output) if args.output else None

    try:
        result = export_to_edge_onnx(
            job_id=args.job_id,
            output_dir=out_path,
            quantize=args.quantize,
            mock_for_test=args.mock
        )
        print("\n✅ Successfully exported edge package:")
        print(f"   Job ID:     {result['jobId']}")
        print(f"   Output:     {result['outputDir']}")
        print(f"   Duration:   {result['durationSec']}s")
        print(f"   ONNX File:  {result['manifest']['onnxFileName']} ({result['manifest']['onnxSizeBytes']} bytes)")
        print(f"   Providers:  {', '.join(result['manifest']['supportedExecutionProviders'])}\n")
    except Exception as ex:
        logger.error(f"Failed to export edge adapter: {ex}")
        sys.exit(1)

if __name__ == "__main__":
    main()
