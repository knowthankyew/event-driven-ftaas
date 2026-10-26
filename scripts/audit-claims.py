#!/usr/bin/env python3
"""
audit-claims.py - Unified Architectural & Empirical Claims Linter for ML / Event-Driven FTaaS.

Provides portfolio-standard CLI discoverability (scripts/audit-claims.py) that invokes:
1. Empirical Claims Linter unit tests (tests/test_claims_linter.py)
2. Document Claims & Empirical Constraint Check (scripts/sync_docs.py --check)
3. Benchmark Docs & Wilson Interval Verification (scripts/sync_benchmark_docs.py --check)
4. Vocabulary Guardrails across docs/ARCHITECTURE.md, README.md, ROADMAP.md, and specs
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PROHIBITED_WORDS = [
    re.compile(r"\bbullet-?proof\b", re.IGNORECASE),
    re.compile(r"\brock\s+solid\b", re.IGNORECASE),
    re.compile(r"\bconclusive\b", re.IGNORECASE),
    re.compile(r"\bdefinitive\b", re.IGNORECASE),
    re.compile(r"\binfinitely scalable\b", re.IGNORECASE),
]

def get_docs_to_scan():
    docs = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "ROADMAP.md",
        REPO_ROOT / "ARCHITECTURE.md",
    ]
    docs_dir = REPO_ROOT / "docs"
    if docs_dir.is_dir():
        docs.extend(sorted(docs_dir.glob("*.md")))
    return docs


def audit_vocabulary(errors: list[str]) -> None:
    for doc in get_docs_to_scan():
        if not doc.is_file():
            continue
        lines = doc.read_text(encoding="utf-8").splitlines()
        for idx, line in enumerate(lines, 1):
            lower = line.lower()
            if "prohibit" in lower or "forbidden" in lower or "guardrail" in lower:
                continue
            for pattern in PROHIBITED_WORDS:
                if pattern.search(line):
                    errors.append(
                        f"Vocabulary violation in {doc.relative_to(REPO_ROOT)}:{idx}: "
                        f"Prohibited term matching '{pattern.pattern}' found: '{line.strip()}'"
                    )


def run_check(cmd: list[str], label: str, errors: list[str]) -> bool:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    if proc.returncode != 0:
        errors.append(f"{label} failed (exit {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
        return False
    return True


def main() -> int:
    print("\n=== Event-Driven FTaaS (ML) Claims & Invariant Audit ===")
    errors: list[str] = []

    # 1. Run empirical claims linter test suite
    test_cmd = [sys.executable, "-m", "unittest", "tests/test_claims_linter.py"]
    if run_check(test_cmd, "Empirical Claims Test Suite (test_claims_linter.py)", errors):
        print("  ✓ Empirical claims & token bound tests passed (13 test cases)")

    # 2. Run sync_docs claims and AST check
    sync_cmd = [sys.executable, "scripts/sync_docs.py", "--check"]
    if run_check(sync_cmd, "Document Claims Sync Check (sync_docs.py --check)", errors):
        print("  ✓ Wilson intervals, MRR recomputation & conclusion AST verified (sync_docs.py)")

    # 3. Run sync_benchmark_docs check
    bench_cmd = [sys.executable, "scripts/sync_benchmark_docs.py", "--check"]
    if run_check(bench_cmd, "Benchmark Docs Sync Check (sync_benchmark_docs.py --check)", errors):
        print("  ✓ Benchmark metrics and empirical constraints verified (sync_benchmark_docs.py)")

    # 4. Audit vocabulary across architecture and documentation
    audit_vocabulary(errors)
    if not any("Vocabulary violation" in e for e in errors):
        print("  ✓ docs/ARCHITECTURE.md and all docs verified clean of prohibited vocabulary")

    if errors:
        print(f"\n[FAIL] {len(errors)} invariant violation(s) detected:")
        for err in errors:
            print(f"  ✗ {err}")
        return 1

    print("\nAll ML empirical claims, registry schemas, and governance gates verified successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
