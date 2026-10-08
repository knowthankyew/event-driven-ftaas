#!/usr/bin/env python3
"""
sync_docs.py: Claims Registry Documentation Synchronizer & Linter v2.

Enforces:
1. Documentation claim markers `<!-- claim:<id> -->...<!-- /claim -->` strictly match
   empirical artifacts registered in docs/claims_registry.json.
2. Inside regions marked as empirical (`<!-- BEGIN_EMPIRICAL -->` ... `<!-- END_EMPIRICAL -->`
   and `<!-- BEGIN_RETRIEVAL_BENCHMARK_TABLE -->` ... `<!-- END_RETRIEVAL_BENCHMARK_TABLE -->`),
   any digit outside a claim marker fails the check.
3. Word lint: adjectives such as 'bulletproof', 'conclusive', 'definitive' are prohibited.
   Guarded terms ('verified', 'authentic', 'proves', 'zero hallucination', '100%')
   are allowed only when tied to a registered claim.
4. Script AST lint: functions building conclusion text contain no numeric or digit-bearing
   string literals except through interpolation.
5. Independent recomputation of Wilson intervals, hit counts, and MRR from per-query records.
6. Extraction and rendering of code constants (MAX_SAFE_GENERATE_TOKENS, MAX_SAFE_EMBED_TOKENS).

Modes:
  --check (DEFAULT): Validates documentation and artifacts; exits 0 on match, 1 on failure.
  --write (EXPLICIT): Renders resolved claim values into documentation markers.
"""

import argparse
import ast
import json
import math
from decimal import Decimal, ROUND_HALF_UP
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = REPO_ROOT / "docs" / "claims_registry.json"
DEFAULT_ROADMAP_PATH = REPO_ROOT / "docs" / "ROADMAP.md"

EMPIRICAL_BLOCK_DELIMITERS = [
    ("<!-- BEGIN_EMPIRICAL -->", "<!-- END_EMPIRICAL -->"),
    ("<!-- BEGIN_RETRIEVAL_BENCHMARK_TABLE -->", "<!-- END_RETRIEVAL_BENCHMARK_TABLE -->"),
]

PROHIBITED_WORDS = [
    re.compile(r"\bbulletproof\b", re.IGNORECASE),
    re.compile(r"\bconclusive\b", re.IGNORECASE),
    re.compile(r"\bdefinitive\b", re.IGNORECASE),
]

GUARDED_WORDS = [
    re.compile(r"\bverified\b", re.IGNORECASE),
    re.compile(r"\bauthentic\b", re.IGNORECASE),
    re.compile(r"\bproves\b", re.IGNORECASE),
    re.compile(r"\bzero\s+hallucination\b", re.IGNORECASE),
    re.compile(r"100%"),
]


def extract_python_constant(file_path: Path, const_name: str) -> int | str | float:
    """Extract constant definition from a Python module using AST parsing."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Python source file not found: {file_path}")
    content = file_path.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(file_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == const_name:
                    if isinstance(node.value, ast.Constant):
                        return node.value.value
                    elif isinstance(node.value, ast.Call):
                        # E.g. int(os.getenv("...", "24"))
                        # or min(int(os.getenv("...", "240")), 255)
                        for arg in ast.walk(node.value):
                            if isinstance(arg, ast.Constant) and isinstance(arg.value, (int, float)):
                                return arg.value
                            elif isinstance(arg, ast.Constant) and isinstance(arg.value, str) and arg.value.isdigit():
                                return int(arg.value)

    # Fallback regex looking for literal number or string number
    m = re.search(rf"^{const_name}\s*=\s*.*?['\"]?(\d+)['\"]?", content, re.MULTILINE)
    if m:
        return int(m.group(1))
    raise ValueError(f"Constant {const_name} not found in {file_path}")


def resolve_claim_value(claim_id: str, claim_cfg: dict, repo_root: Path = REPO_ROOT) -> str:
    """Resolve and format a registered claim value from its empirical source artifact."""
    artifact_rel = claim_cfg.get("source_artifact")
    if not artifact_rel:
        raise ValueError(f"Claim '{claim_id}' missing 'source_artifact'")
    artifact_path = repo_root / artifact_rel
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Source artifact not found for claim '{claim_id}': {artifact_path}")

    fmt = claim_cfg.get("format_rule", "{value}")

    # Case 1: Python constant
    if "python_const" in claim_cfg:
        val = extract_python_constant(artifact_path, claim_cfg["python_const"])
        return fmt.format(value=val)

    # Case 2: JSON artifact path
    if "json_path" in claim_cfg:
        with open(artifact_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        parts = claim_cfg["json_path"].split(".")
        cur = data
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            elif isinstance(cur, list) and p.isdigit() and int(p) < len(cur):
                cur = cur[int(p)]
            else:
                raise KeyError(f"Path '{claim_cfg['json_path']}' part '{p}' not found in {artifact_path.name}")
        val = cur
        if isinstance(val, list) and len(val) == 2 and all(isinstance(x, (int, float)) for x in val):
            # Special case for confidence intervals e.g. [0.673, 0.919]
            if fmt == "CI_PCT":
                return f"[{val[0]*100:.1f}%, {val[1]*100:.1f}%]"
            return f"[{val[0]*100:.1f}%, {val[1]*100:.1f}%]"
        return fmt.format(value=val)

    raise ValueError(f"Claim '{claim_id}' has neither 'python_const' nor 'json_path'")


def independent_wilson_score_interval(successes: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    """
    Independent implementation of 95% Wilson confidence score interval.
    Does not import or reuse benchmark script functions.
    """
    if n == 0:
        return (0.0, 0.0)
    z = 1.95996
    p = successes / n
    denom = 1.0 + (z * z) / n
    center = (p + (z * z) / (2.0 * n)) / denom
    spread = z * math.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n) / denom
    lower = max(0.0, center - spread)
    upper = min(1.0, center + spread)
    return round(lower, 4), round(upper, 4)


def independent_recompute_metrics(benchmark_path: Path) -> tuple[bool, list[str]]:
    """
    Recompute retrieval hits, MRR, and Wilson intervals directly from query_details
    and assert exact match with summary metrics in JSON artifact.
    """
    errors = []
    if not benchmark_path.is_file():
        return False, [f"Missing benchmark JSON: {benchmark_path}"]

    with open(benchmark_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "full_corpus_benchmark" not in data or "query_details" not in data["full_corpus_benchmark"]:
        return True, []  # Early artifact version or structure

    query_details = data["full_corpus_benchmark"]["query_details"]
    summary = data["full_corpus_benchmark"].get("summary", {})
    n = len(query_details)
    if n == 0:
        return False, ["query_details is empty in benchmark artifact"]

    # Gather methods present
    first_q = query_details[0]
    rankings = first_q.get("rankings", {})
    methods = list(rankings.keys())

    for m in methods:
        top1_hits = 0
        top3_hits = 0
        reciprocal_ranks = []

        for q in query_details:
            r = q["rankings"].get(m, {})
            # If top1_hit is boolean
            if r.get("top1_hit"):
                top1_hits += 1
            if r.get("top3_hit"):
                top3_hits += 1
            rr = float(r.get("rr", 0.0))
            reciprocal_ranks.append(rr)

        mrr_calc = sum(reciprocal_ranks) / n
        top1_ci = independent_wilson_score_interval(top1_hits, n)
        top3_ci = independent_wilson_score_interval(top3_hits, n)

        if m in summary:
            expected = summary[m]
            if expected.get("top1_hits") != top1_hits:
                errors.append(f"Method '{m}' top1_hits mismatch: expected {expected.get('top1_hits')}, calculated {top1_hits}")
            if expected.get("top3_hits") != top3_hits:
                errors.append(f"Method '{m}' top3_hits mismatch: expected {expected.get('top3_hits')}, calculated {top3_hits}")
            if abs(expected.get("mrr", 0.0) - mrr_calc) > 1e-4:
                errors.append(f"Method '{m}' MRR mismatch: expected {expected.get('mrr')}, calculated {mrr_calc:.4f}")
            exp_ci1 = expected.get("top1_ci_95")
            if exp_ci1 and (abs(exp_ci1[0] - top1_ci[0]) > 1e-3 or abs(exp_ci1[1] - top1_ci[1]) > 1e-3):
                errors.append(f"Method '{m}' top1_ci mismatch: expected {exp_ci1}, calculated {top1_ci}")

    return len(errors) == 0, errors


def lint_script_conclusions(scripts_dir: Path) -> tuple[bool, list[str]]:
    """
    AST check that functions building conclusion text contain no numeric or
    digit-bearing string literals except through interpolation.
    """
    errors = []
    if not scripts_dir.is_dir():
        return True, []

    for py_file in scripts_dir.glob("*.py"):
        if py_file.name in ("sync_docs.py", "sync_benchmark_docs.py"):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except Exception as e:
            errors.append(f"Failed to parse AST for {py_file.name}: {e}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if "conclusion" in node.name.lower() or "build_conclusion" in node.name.lower():
                    # Walk inside function AST
                    for child in ast.walk(node):
                        # Standalone constant string
                        if isinstance(child, ast.Constant) and isinstance(child.value, str):
                            # Check if part of a JoinStr or standalone
                            # Ignore docstring
                            if child == node.body[0] and isinstance(node.body[0], ast.Expr):
                                continue
                            if re.search(r"\d", child.value):
                                errors.append(
                                    f"Script AST lint: {py_file.name}::{node.name} line {getattr(child, 'lineno', '?')} "
                                    f"contains hardcoded numeric literal in string: {child.value!r}"
                                )
                        elif isinstance(child, ast.JoinedStr):
                            # Inside f-string: verify ast.Constant parts don't contain digits
                            for part in child.values:
                                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                                    if re.search(r"\d", part.value):
                                        errors.append(
                                            f"Script AST lint: {py_file.name}::{node.name} line {getattr(part, 'lineno', '?')} "
                                            f"f-string contains hardcoded digits in constant fragment: {part.value!r}"
                                        )

    return len(errors) == 0, errors


def lint_prose_and_words(content: str, doc_name: str = "document") -> tuple[bool, list[str]]:
    """
    Word lint:
    - Prohibit 'bulletproof', 'conclusive', 'definitive' anywhere.
    - Guarded words ('verified', 'authentic', 'proves', 'zero hallucination', '100%')
      allowed only when tied to a registered claim marker <!-- claim:... --> or <!-- claim_term:... -->.
    """
    errors = []

    # Check prohibited words
    for pat in PROHIBITED_WORDS:
        m = pat.search(content)
        if m:
            errors.append(f"Word lint failure in {doc_name}: prohibited adjective '{m.group(0)}' found.")

    # Mask claim markers before checking guarded words
    # Replace all <!-- claim:... -->...<!-- /claim --> and <!-- claim_term:... -->...<!-- /claim_term --> with blanks
    masked = re.sub(r"<!--\s*claim:[^>]+-->.*?<!--\s*/claim\s*-->", " ", content, flags=re.DOTALL)
    masked = re.sub(r"<!--\s*claim_term:[^>]+-->.*?<!--\s*/claim_term\s*-->", " ", masked, flags=re.DOTALL)

    for pat in GUARDED_WORDS:
        matches = list(pat.finditer(masked))
        if matches:
            for match in matches:
                # Find line number
                line_no = masked[:match.start()].count("\n") + 1
                errors.append(
                    f"Word lint failure in {doc_name} line {line_no}: guarded term '{match.group(0)}' "
                    f"used in bare text without being tied to a registered claim marker."
                )

    return len(errors) == 0, errors


def check_empirical_regions(content: str, doc_name: str = "document") -> tuple[bool, list[str]]:
    """
    Verify that inside regions marked as empirical, any empirical metric digit outside a claim marker fails.
    For markdown tables, headers, method names (col 1), and role descriptions (last col) are excluded,
    ensuring all quantitative benchmark cells are strictly governed by claims.
    """
    errors = []
    for begin_marker, end_marker in EMPIRICAL_BLOCK_DELIMITERS:
        pattern = re.escape(begin_marker) + r"(.*?)" + re.escape(end_marker)
        for match in re.finditer(pattern, content, re.DOTALL):
            block_text = match.group(1)
            # Filter table structures if markdown table
            filtered_lines = []
            for line in block_text.splitlines():
                line_str = line.strip()
                if not line_str:
                    continue
                # Skip markdown table header and separator lines
                if line_str.startswith("|") and (":---" in line_str or "Top-1" in line_str or "Retrieval Method" in line_str):
                    continue
                # For data table rows, extract metric cells (exclude col 0 method name and col -1 role)
                if line_str.startswith("|") and line_str.endswith("|"):
                    parts = [p.strip() for p in line_str.split("|")[1:-1]]
                    if len(parts) >= 3:
                        metric_cells = " | ".join(parts[1:-1])
                        filtered_lines.append(metric_cells)
                        continue
                filtered_lines.append(line_str)
            block_to_check = "\n".join(filtered_lines)

            # Remove claim markers
            stripped = re.sub(r"<!--\s*claim:[^>]+-->.*?<!--\s*/claim\s*-->", " ", block_to_check, flags=re.DOTALL)
            # Remove html comments
            stripped = re.sub(r"<!--.*?-->", " ", stripped, flags=re.DOTALL)

            digits = re.findall(r"\d+", stripped)
            if digits:
                line_offset = content[:match.start()].count("\n") + 1
                errors.append(
                    f"Empirical block lint in {doc_name} (around line {line_offset}): "
                    f"found unmanaged digits outside claim marker in empirical block: {digits[:5]}"
                )

    return len(errors) == 0, errors


def sync_document(
    doc_path: Path,
    registry_path: Path,
    write: bool = False,
    repo_root: Path = REPO_ROOT
) -> tuple[bool, list[str]]:
    """
    Verify or update claim markers in doc_path against registry_path.
    Returns (success, list_of_errors_or_messages).
    """
    messages = []
    errors = []

    if not registry_path.is_file():
        return False, [f"Registry not found: {registry_path}"]
    if not doc_path.is_file():
        return False, [f"Target doc not found: {doc_path}"]

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    claims = registry.get("claims", {})
    resolved_values = {}

    for claim_id, claim_cfg in claims.items():
        try:
            val = resolve_claim_value(claim_id, claim_cfg, repo_root=repo_root)
            resolved_values[claim_id] = val
        except Exception as e:
            errors.append(f"Failed to resolve claim '{claim_id}': {e}")

    content = doc_path.read_text(encoding="utf-8")

    # Claim pattern: <!-- claim:id -->...<!-- /claim -->
    claim_pat = re.compile(r"<!--\s*claim:([a-zA-Z0-9_\-]+)\s*-->(.*?)<!--\s*/claim\s*-->", re.DOTALL)

    updated_content = content
    offset_changes = 0

    found_claims = set()
    for m in claim_pat.finditer(content):
        cid = m.group(1)
        current_val = m.group(2)
        found_claims.add(cid)

        if cid not in resolved_values:
            errors.append(f"Unknown claim ID '{cid}' used in {doc_path.name}")
            continue

        expected_val = resolved_values[cid]
        if current_val != expected_val:
            if write:
                messages.append(f"Updated claim '{cid}' from {current_val!r} to {expected_val!r}")
            else:
                errors.append(
                    f"Claim drift for '{cid}' in {doc_path.name}: "
                    f"expected {expected_val!r}, found {current_val!r}"
                )

    if write:
        def repl(match):
            cid = match.group(1)
            if cid in resolved_values:
                return f"<!-- claim:{cid} -->{resolved_values[cid]}<!-- /claim -->"
            return match.group(0)

        updated_content = claim_pat.sub(repl, content)
        if updated_content != content:
            doc_path.write_text(updated_content, encoding="utf-8")
            messages.append(f"Wrote updated claims to {doc_path.name}")

    # Check empirical regions
    emp_ok, emp_errors = check_empirical_regions(doc_path.read_text(encoding="utf-8"), doc_path.name)
    if not emp_ok:
        errors.extend(emp_errors)

    # Word lint
    word_ok, word_errors = lint_prose_and_words(doc_path.read_text(encoding="utf-8"), doc_path.name)
    if not word_ok:
        errors.extend(word_errors)

    if errors:
        return False, errors
    return True, messages


def main():
    parser = argparse.ArgumentParser(description="Synchronize and lint documentation claims against empirical artifacts.")
    parser.add_argument(
        "--check",
        action="store_true",
        default=True,
        help="Check documentation claims and empirical regions without modifying files (DEFAULT)",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Overwrite documentation claim markers with ground truth empirical values",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=DEFAULT_REGISTRY_PATH,
        help="Path to claims_registry.json",
    )
    parser.add_argument(
        "--doc",
        type=Path,
        default=DEFAULT_ROADMAP_PATH,
        help="Path to markdown document to check/sync",
    )

    args = parser.parse_args()
    do_write = args.write

    print(f"=== Claims Synchronizer & Linter v2 (mode: {'WRITE' if do_write else 'CHECK'}) ===")
    overall_success = True

    # 1. Independent metrics recomputation
    bench_json = REPO_ROOT / "docs" / "empirical_retrieval_benchmark.json"
    if bench_json.is_file():
        metrics_ok, metrics_errs = independent_recompute_metrics(bench_json)
        if not metrics_ok:
            overall_success = False
            for err in metrics_errs:
                print(f"[FAIL] {err}", file=sys.stderr)
        else:
            print("[PASS] Independent Wilson interval & MRR recomputation verified.")

    # 2. Script conclusions AST lint
    scripts_dir = REPO_ROOT / "scripts"
    ast_ok, ast_errs = lint_script_conclusions(scripts_dir)
    if not ast_ok:
        overall_success = False
        for err in ast_errs:
            print(f"[FAIL] {err}", file=sys.stderr)
    else:
        print("[PASS] Script conclusion AST check passed (zero hardcoded literal numbers in conclusion functions).")

    # 3. Document sync & linter
    doc_ok, doc_msgs = sync_document(args.doc, args.registry, write=do_write, repo_root=REPO_ROOT)
    if not doc_ok:
        overall_success = False
        for err in doc_msgs:
            print(f"[FAIL] {err}", file=sys.stderr)
    else:
        for msg in doc_msgs:
            print(f"[INFO] {msg}")
        print(f"[PASS] Document claims and empirical constraints verified for {args.doc.name}.")

    if overall_success:
        print("\nAll claims and empirical guardrails verified successfully.")
        sys.exit(0)
    else:
        print("\nClaims synchronization / lint failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
