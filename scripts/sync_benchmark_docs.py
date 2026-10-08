#!/usr/bin/env python3
"""
sync_benchmark_docs.py

Empirical Documentation Synchronization Utility (Compatibility Wrapper).
Delegates to scripts/sync_docs.py (Claims Registry Documentation Synchronizer & Linter v2).
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import sync_docs

if __name__ == "__main__":
    sys.exit(sync_docs.main())
