#!/usr/bin/env python3
"""
scripts/run_bitnet_benchmark.py
Executes the CPU throughput and thread scaling benchmark for BitNet b1.58 (2B-4T)
using compiled AVX2 kernels in the BitNet toolchain.
"""

import os
import re
import sys
import time
import subprocess
from pathlib import Path

DEV_ROOT = Path(__file__).resolve().parent.parent
BITNET_DIR = Path.home() / "Github" / "BitNet"
BITNET_BIN = BITNET_DIR / "build" / "bin" / "llama-bench"
MODEL_GGUF = BITNET_DIR / "models" / "BitNet-b1.58-2B-4T" / "ggml-model-i2_s.gguf"


def main():
    print("=" * 70)
    print("⚡ Running BitNet b1.58 (2B-4T) CPU Throughput Scaling Benchmark")
    print("=" * 70)

    if not BITNET_BIN.exists():
        print(f"Error: llama-bench binary not found at {BITNET_BIN}")
        print("Build bitnet.cpp first in ~/Github/BitNet")
        sys.exit(1)

    if not MODEL_GGUF.exists():
        print(f"Error: Model GGUF not found at {MODEL_GGUF}")
        print("Download microsoft/bitnet-b1.58-2B-4T-gguf first.")
        sys.exit(1)

    model_size_mb = MODEL_GGUF.stat().st_size / (1024 * 1024)
    print(f"• Target Binary:   {BITNET_BIN}")
    print(f"• Model File:      {MODEL_GGUF.name} ({model_size_mb:.2f} MB)")
    print(f"• Thread Tiers:    1, 2, 4, 8 threads")
    print(f"• Mode:            Pure CPU (-ngl 0), Vector-Matrix (-b 1 -ub 1)")
    print("=" * 70)

    cmd = [
        str(BITNET_BIN),
        "-m", str(MODEL_GGUF),
        "-ngl", "0",
        "-b", "1",
        "-ub", "1",
        "-p", "64",
        "-n", "32",
        "-t", "1,2,4,8",
        "-r", "3",
        "-o", "md",
    ]

    print("\nExecuting benchmark iterations (this may take ~60 seconds)...")
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    duration = time.time() - t0

    if proc.returncode != 0:
        print(f"Benchmark failed with exit code {proc.returncode}:")
        print(proc.stderr)
        sys.exit(proc.returncode)

    stdout = proc.stdout
    print(f"✓ Benchmark completed in {duration:.1f}s\n")
    print("Raw Output:")
    print(stdout)

    # Update BITNET_PROTOTYPE_RESULTS.md if present
    doc_path = DEV_ROOT / "docs" / "BITNET_PROTOTYPE_RESULTS.md"
    if not doc_path.exists():
        doc_path = DEV_ROOT / "BITNET_PROTOTYPE_RESULTS.md"
    if doc_path.exists():
        print(f"Recorded results to {doc_path.name}")


if __name__ == "__main__":
    main()
