#!/usr/bin/env python3
"""
audit-claims.py - Unified Architectural & Empirical Claims Linter for ML / Event-Driven FTaaS.

Provides portfolio-standard CLI discoverability (scripts/audit-claims.py) that invokes:
1. Empirical Claims Linter unit tests (tests/test_claims_linter.py)
2. Document Claims & Empirical Constraint Check across all registered docs (scripts/sync_docs.py --check)
3. Prohibited & Guarded Vocabulary Audit across README.md, NOTICE.md, and docs/*.md
4. Architecture Diagram Freshness & Path Existence Audit (diagram-meta covers & code paths)
5. Canonical MIT License, SPDX, and Ring-Fenced AGPL Governance Gate
"""

import fnmatch
import hashlib
import os
import re
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import sync_docs  # noqa: E402

ADDITIONAL_PROHIBITED_WORDS = [
    re.compile(r"\bbullet-?proof\b", re.IGNORECASE),
    re.compile(r"\brock\s+solid\b", re.IGNORECASE),
    re.compile(r"\binfinitely scalable\b", re.IGNORECASE),
]

# ---------------------------------------------------------------------------
# Architecture diagram freshness & path-existence audit
# ---------------------------------------------------------------------------

DIAGRAM_META_RE = re.compile(r"<!--\s*diagram-meta\b(.*?)-->", re.DOTALL)
DIAGRAM_CODE_PATH_RE = re.compile(r"<code>([^<>\s]+)</code>")
DIAGRAM_PATH_PREFIXES = ("src/", "tests/", "scripts/", "docs/", ".github/", "policies/")
REQUIRED_DIAGRAM_DOC = "docs/ARCHITECTURE.md"

# ---------------------------------------------------------------------------
# License compliance & governance constants
# ---------------------------------------------------------------------------

CANONICAL_MIT_SHA256 = "49f4b2686067fc56489ee83e4280c72a38b5c09d275bf352f5247d90572c8b9b"
ALLOWED_AGPL_FILES = {"NOTICE.md", "scripts/audit-claims.py", "tests/test_claims_linter.py"}
SPDX_PREFIX = "SPDX" + "-License-Identifier:"
SPDX_RE = re.compile(re.escape(SPDX_PREFIX) + r"\s*([^\r\n]+)")


def clean_spdx_identifier(raw: str) -> str:
    """Strip comment terminators (--> , */), delimiters, quotes, and punctuation from SPDX identifier."""
    cleaned = re.sub(r"(-->|\*\/|\*|#|\/|\"|')[\s\S]*$", "", raw).strip()
    return cleaned.rstrip(";,").strip()


def _git(*args: str) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True)
    except FileNotFoundError as ex:
        return subprocess.CompletedProcess(
            args=["git", *args],
            returncode=127,
            stdout="",
            stderr=f"git executable not found: {ex}"
        )


def get_docs_to_scan() -> list[Path]:
    docs = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "NOTICE.md",
    ]
    docs_dir = REPO_ROOT / "docs"
    if docs_dir.is_dir():
        # Include top-level docs/*.md and any nested docs/**/*.md
        docs.extend(sorted(docs_dir.rglob("*.md")))
    return [d for d in dict.fromkeys(docs) if d.is_file()]


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


def parse_diagram_meta(text: str) -> dict | None:
    """Parse the <!-- diagram-meta ... --> header. Returns None if absent."""
    m = DIAGRAM_META_RE.search(text)
    if not m:
        return None
    meta: dict = {}
    for line in m.group(1).splitlines():
        key, sep, val = line.partition(":")
        if sep:
            meta[key.strip()] = val.strip()
    covers = meta.get("covers", "")
    if covers.startswith("[") and covers.endswith("]"):
        meta["covers"] = [c.strip() for c in covers[1:-1].split(",") if c.strip()]
    else:
        meta["covers"] = []
    return meta


def _audit_diagram_doc(doc: Path, errors: list[str]) -> None:
    try:
        rel = doc.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        rel = doc.name
    label = f"Architecture diagram ({rel})"
    text = doc.read_text(encoding="utf-8")

    meta = parse_diagram_meta(text)
    if meta is None:
        errors.append(f"{label}: missing <!-- diagram-meta ... --> header")
        return

    # 1. Header sanity
    raw_date = meta.get("last-validated", "")
    try:
        max_allowed_date = datetime.now(timezone.utc).date() + timedelta(days=1)
        if date.fromisoformat(raw_date) > max_allowed_date:
            errors.append(f"{label}: last-validated {raw_date} is in the future")
    except ValueError:
        errors.append(f"{label}: last-validated '{raw_date}' is not an ISO date (YYYY-MM-DD)")

    commit = meta.get("validated-commit", "")
    covers: list[str] = meta["covers"]
    if not commit:
        errors.append(f"{label}: validated-commit is missing")
    if not covers:
        errors.append(f"{label}: covers is missing or not a [a, b, c] list")

    # 2. Freshness: nothing under `covers` changed since validated-commit
    if commit and covers:
        resolved = _git("rev-parse", "--verify", "--quiet", f"{commit}^{{commit}}")
        if resolved.returncode != 0:
            errors.append(
                f"{label}: validated-commit '{commit}' not found in this clone "
                f"(shallow checkout? use fetch-depth: 0)"
            )
        elif _git("merge-base", "--is-ancestor", commit, "HEAD").returncode != 0:
            errors.append(f"{label}: validated-commit '{commit}' is not an ancestor of HEAD")
        else:
            pathspecs = [f":(glob){c}" for c in covers] + [
                f":(exclude){rel}",
                ":(exclude)scripts/audit-claims.py",
                ":(exclude)scripts/sync_docs.py",
                ":(exclude)tests/test_claims_linter.py",
            ]
            diff = _git("diff", "--name-only", commit, "--", *pathspecs)
            if diff.returncode != 0:
                errors.append(f"{label}: git diff failed: {diff.stderr.strip()}")
            else:
                changed = [f for f in diff.stdout.splitlines() if f.strip()]
                if changed:
                    last = _git("log", "-1", "--format=%h", "HEAD", "--", *pathspecs).stdout.strip()
                    shown = ", ".join(changed[:10]) + (" ..." if len(changed) > 10 else "")
                    errors.append(
                        f"{label}: STALE. {len(changed)} covered file(s) changed since "
                        f"{commit}: {shown}. Re-validate the diagram, then set "
                        f"validated-commit to {last} and update last-validated."
                    )

    # 3. Every <code>path</code> reference must exist on disk
    for raw in dict.fromkeys(DIAGRAM_CODE_PATH_RE.findall(text)):  # dedupe, keep order
        p = raw.split("#", 1)[0]
        p = re.sub(r":\d+(-\d+)?$", "", p)  # strip :12 or :12-30 line suffixes
        if not p.startswith(DIAGRAM_PATH_PREFIXES) or any(ch in p for ch in "*{}<>$"):
            continue
        if not (REPO_ROOT / p).exists():
            errors.append(f"{label}: referenced path does not exist: {p}")


def audit_architecture_diagrams(errors: list[str]) -> None:
    """Freshness + path-existence audit for every doc carrying a diagram-meta header.
    docs/ARCHITECTURE.md is required to have one."""
    required = REPO_ROOT / REQUIRED_DIAGRAM_DOC
    targets: list[Path] = []
    if required.is_file():
        targets.append(required)
    else:
        errors.append(f"Architecture diagram: required doc {REQUIRED_DIAGRAM_DOC} not found")
    for doc in get_docs_to_scan():
        if doc != required and "<!-- diagram-meta" in doc.read_text(encoding="utf-8"):
            targets.append(doc)
    for doc in targets:
        _audit_diagram_doc(doc, errors)


def audit_license_compliance(errors: list[str]) -> None:
    """Validate canonical MIT license integrity, absence of AGPL files/references,
    and single SPDX identifier repo-wide."""
    # 1. Canonical root LICENSE check
    license_file = REPO_ROOT / "LICENSE"
    if not license_file.is_file():
        errors.append("License gate: root LICENSE file is missing")
    else:
        license_sha = hashlib.sha256(license_file.read_bytes()).hexdigest()
        if license_sha != CANONICAL_MIT_SHA256:
            errors.append(
                f"License gate: root LICENSE SHA-256 mismatch. Expected {CANONICAL_MIT_SHA256}, got {license_sha}. "
                "Ensure root LICENSE matches canonical unmodified MIT text."
            )

    # 2. Check tracked files for prohibited AGPL license files, AGPL references & SPDX identifiers
    ls_files = _git("ls-files", "-z")
    if ls_files.returncode != 0:
        errors.append(f"License gate: git ls-files failed: {ls_files.stderr.strip()}")
        return

    for rel_path in ls_files.stdout.split("\0"):
        if not rel_path:
            continue

        base_name_lower = os.path.basename(rel_path).lower()
        if "agpl" in base_name_lower:
            errors.append(f"License gate: prohibited AGPL license file found: {rel_path}")

        p = Path(rel_path) if os.path.isabs(rel_path) else REPO_ROOT / rel_path
        if not p.is_file():
            continue
        try:
            content = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue

        # Check for disallowed AGPL references (case-insensitive word boundary)
        if re.search(r"\bagpl\b", content, re.IGNORECASE) and rel_path not in ALLOWED_AGPL_FILES:
            errors.append(
                f"License gate: prohibited 'AGPL' reference found in {rel_path}. "
                "The repository is 100% MIT; AGPL references are restricted to historical disclosures in NOTICE.md."
            )

        # Check SPDX identifiers: only MIT allowed (rejecting compound expressions like MIT AND GPL-3.0)
        for spdx_raw in SPDX_RE.findall(content):
            spdx = clean_spdx_identifier(spdx_raw)
            if spdx != "MIT":
                errors.append(
                    f"License gate: non-MIT SPDX identifier '{spdx}' found in {rel_path}. Only 'MIT' is permitted."
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

    # 4. Architecture diagram freshness & path existence
    before_diagram = len(errors)
    audit_architecture_diagrams(errors)
    if len(errors) == before_diagram:
        print("  ✓ Architecture diagram metadata is fresh and every referenced path exists")

    # 5. Canonical MIT license & SPDX governance gate
    before_license = len(errors)
    audit_license_compliance(errors)
    if len(errors) == before_license:
        print("  ✓ 100% MIT canonical license integrity, SPDX identifiers & AGPL boundaries verified")

    if errors:
        print(f"\n[FAIL] {len(errors)} invariant violation(s) detected:")
        for err in errors:
            print(f"  ✗ {err}")
        return 1

    print("\nAll ML empirical claims, registry schemas, and governance gates verified successfully.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
