#!/usr/bin/env python3
"""
audit-claims.py - Unified Architectural & Empirical Claims Linter for ML / Event-Driven FTaaS.

Provides portfolio-standard CLI discoverability (scripts/audit-claims.py) that invokes:
1. Empirical Claims Linter unit tests (tests/test_claims_linter.py)
2. Document Claims & Empirical Constraint Check across all registered docs (scripts/sync_docs.py --check)
3. Prohibited & Guarded Vocabulary Audit across README.md, NOTICE.md, and docs/*.md
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import sync_docs  # noqa: E402

ADDITIONAL_PROHIBITED_WORDS = [
    re.compile(r"\bbullet-?proof\b", re.IGNORECASE),
    re.compile(r"\brock\s+solid\b", re.IGNORECASE),
    re.compile(r"\binfinitely scalable\b", re.IGNORECASE),
]


def get_docs_to_scan() -> list[Path]:
    docs = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "NOTICE.md",
    ]
    docs_dir = REPO_ROOT / "docs"
    if docs_dir.is_dir():
        docs.extend(sorted(docs_dir.glob("*.md")))
    return [d for d in docs if d.is_file()]


def audit_vocabulary(errors: list[str]) -> None:
    registry_path = REPO_ROOT / "docs" / "claims_registry.json"
    for doc in get_docs_to_scan():
        content = doc.read_text(encoding="utf-8")
        rel_name = str(doc.relative_to(REPO_ROOT))

        # 1. Run sync_docs prohibited + guarded word linter (requires claim_term/claim markers for guarded words)
        ok, word_errs = sync_docs.lint_prose_and_words(content, rel_name, registry_path=registry_path)
        if not ok:
            for e in word_errs:
                errors.append(f"Vocabulary violation: {e}")

        # 2. Check additional portfolio prohibited terms line-by-line (without skipping lines containing 'guardrail')
        for idx, line in enumerate(content.splitlines(), 1):
            for pattern in ADDITIONAL_PROHIBITED_WORDS:
                if pattern.search(line):
                    errors.append(
                        f"Vocabulary violation in {rel_name}:{idx}: "
                        f"Prohibited term matching '{pattern.pattern}' found: '{line.strip()}'"
                    )


def run_check(cmd: list[str], label: str, errors: list[str]) -> tuple[bool, str]:
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    combined = f"{proc.stdout}\n{proc.stderr}"
    if proc.returncode != 0:
        errors.append(f"{label} failed (exit {proc.returncode}):\n{combined}")
        return False, combined
    return True, combined


def main() -> int:
    print("\n=== Event-Driven FTaaS (ML) Claims & Invariant Audit ===")
    errors: list[str] = []

    # 1. Run empirical claims linter test suite
    test_cmd = [sys.executable, "-m", "unittest", "tests/test_claims_linter.py"]
    ok_tests, test_out = run_check(test_cmd, "Empirical Claims Test Suite (test_claims_linter.py)", errors)
    if ok_tests:
        m = re.search(r"Ran\s+(\d+)\s+tests?", test_out)
        count_str = f"{m.group(1)} test cases" if m else "all test cases"
        print(f"  ✓ Empirical claims & token bound tests passed ({count_str})")

    # 2. Run sync_docs claims, empirical blocks, and AST check across all registered docs
    sync_cmd = [sys.executable, "scripts/sync_docs.py", "--check"]
    ok_sync, _ = run_check(sync_cmd, "Document Claims Sync Check (sync_docs.py --check)", errors)
    if ok_sync:
        print("  ✓ Wilson intervals, MRR recomputation, conclusion AST & multi-doc claims verified (sync_docs.py)")

    # 3. Audit prohibited and guarded vocabulary across README.md, NOTICE.md, and docs/*.md
    audit_vocabulary(errors)
    if not any("Vocabulary violation" in e for e in errors):
        print("  ✓ README.md, NOTICE.md, and all docs/*.md verified clean of prohibited & unguarded vocabulary")

    if errors:
        print(f"\n[FAIL] {len(errors)} invariant violation(s) detected:")
        for err in errors:
            print(f"  ✗ {err}")
        return 1

    print("\nAll ML empirical claims, registry schemas, and governance gates verified successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
