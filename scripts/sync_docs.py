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


_SAFE_CALLS = {"int": int, "float": float, "str": str, "min": min, "max": max}


def _eval_const_expr(node: ast.AST, name: str):
    """Evaluate a constant-defining expression without executing code.

    Supported: literals; int/float/str/min/max over supported args;
    os.getenv(NAME, default) and os.environ.get(NAME, default), which evaluate to
    `default` (the documented value when the environment variable is unset).
    Anything else raises, so a registry claim can never bind to a guessed number.
    """
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Call) and not node.keywords:
        fn = node.func
        is_getenv = (
            isinstance(fn, ast.Attribute) and fn.attr == "getenv"
            and isinstance(fn.value, ast.Name) and fn.value.id == "os"
        ) or (
            isinstance(fn, ast.Attribute) and fn.attr == "get"
            and isinstance(fn.value, ast.Attribute) and fn.value.attr == "environ"
            and isinstance(fn.value.value, ast.Name) and fn.value.value.id == "os"
        )
        if is_getenv and len(node.args) == 2:
            return _eval_const_expr(node.args[1], name)
        if isinstance(fn, ast.Name) and fn.id in _SAFE_CALLS and node.args:
            return _SAFE_CALLS[fn.id](*[_eval_const_expr(a, name) for a in node.args])
    raise ValueError(
        f"Constant {name}: unsupported expression ({ast.dump(node)[:80]}...); "
        f"extend _eval_const_expr deliberately rather than guessing"
    )


def extract_python_constant(file_path: Path, const_name: str) -> int | str | float:
    """Extract a module-level constant by evaluating its defining expression (AST only)."""
    if not file_path.is_file():
        raise FileNotFoundError(f"Python source file not found: {file_path}")
    tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == const_name:
                    return _eval_const_expr(node.value, const_name)
    raise ValueError(f"Constant {const_name} not found in {file_path}")


NAMED_FORMAT_RULES = {
    # 2-element [low, high] fractions -> "[67.3%, 91.9%]"
    "ci_pct": lambda v: f"[{v[0] * 100:.1f}%, {v[1] * 100:.1f}%]",
}


def apply_format_rule(claim_id: str, fmt: str, value) -> str:
    """Render a resolved value. Named rules for structured values; str.format for scalars."""
    if fmt in NAMED_FORMAT_RULES:
        try:
            return NAMED_FORMAT_RULES[fmt](value)
        except (TypeError, IndexError, KeyError) as e:
            raise ValueError(f"Claim '{claim_id}': named format_rule {fmt!r} cannot render {value!r}: {e}") from e
    if isinstance(value, (list, tuple, dict)):
        raise ValueError(
            f"Claim '{claim_id}': format_rule {fmt!r} cannot render a {type(value).__name__}; "
            f"use a named rule ({', '.join(sorted(NAMED_FORMAT_RULES))})"
        )
    try:
        return fmt.format(value=value)
    except (ValueError, KeyError, IndexError, AttributeError) as e:
        raise ValueError(f"Claim '{claim_id}': invalid format_rule {fmt!r}: {e}") from e


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
        return apply_format_rule(claim_id, fmt, val)

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
        return apply_format_rule(claim_id, fmt, cur)

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

    full = data.get("full_corpus_benchmark")
    if not isinstance(full, dict) or "query_details" not in full:
        # Fail closed: an artifact that cannot be recomputed must not pass as "verified".
        return False, ["benchmark artifact has no full_corpus_benchmark.query_details; cannot recompute metrics"]

    sections = {
        k: v for k, v in data.items()
        if isinstance(v, dict) and "query_details" in v
    }

    for sec_name, sec_data in sections.items():
        query_details = sec_data["query_details"]
        summary = sec_data.get("summary", {})
        n = len(query_details)
        if n == 0:
            errors.append(f"[{sec_name}] query_details is empty in benchmark artifact")
            continue

        declared_n = sec_data.get("queries_count")
        if declared_n != n:
            errors.append(f"[{sec_name}] queries_count mismatch: declared {declared_n}, query_details has {n}")

        if not query_details:
            continue

        methods = list(query_details[0].get("rankings", {}).keys())
        for q in query_details:
            if set(q.get("rankings", {}).keys()) != set(methods):
                errors.append(f"[{sec_name}] query {q.get('id', '?')!r} ranks a different method set than the first query")
        for m in summary:
            if m not in methods:
                errors.append(f"[{sec_name}] summary method '{m}' has no per-query records")

        for m in methods:
            top1_hits = 0
            top3_hits = 0
            reciprocal_ranks = []

            for q in query_details:
                r = q["rankings"].get(m, {})
                t1, t3, rr = bool(r.get("top1_hit")), bool(r.get("top3_hit")), float(r.get("rr", 0.0))
                # Per-record invariants (assumes rr = 1/rank): a hit flag must agree with its own rr.
                if t1 != (abs(rr - 1.0) < 1e-4) or t3 != (rr >= 1.0 / 3.0 - 1e-4) or (t1 and not t3):
                    errors.append(f"[{sec_name}] Method '{m}' query {q.get('id', '?')!r}: flags (top1={t1}, top3={t3}) disagree with rr={rr}")

                # Derive expected first relevant rank from ranked_ids and relevant_ids when present
                if "ranked_ids" in r and "relevant_ids" in q:
                    rel_set = set(q.get("relevant_ids", []))
                    ranked_ids = r.get("ranked_ids", [])
                    derived_rank = None
                    for rank_idx, doc_id in enumerate(ranked_ids, 1):
                        if doc_id in rel_set:
                            derived_rank = rank_idx
                            break
                    exp_t1 = (derived_rank == 1)
                    exp_t3 = (derived_rank is not None and derived_rank <= 3)
                    exp_rr = (1.0 / derived_rank) if derived_rank else 0.0
                    if t1 != exp_t1 or t3 != exp_t3 or abs(rr - exp_rr) > 5e-4:
                        msg = (
                            f"[{sec_name}] Method '{m}' query {q.get('id', '?')!r}: stored flags "
                            f"(top1={t1}, top3={t3}, rr={rr}) disagree with derived rank {derived_rank} "
                            f"(expected top1={exp_t1}, top3={exp_t3}, rr={exp_rr:.4f})"
                        )
                        print(f"[WARN] {msg}", file=sys.stderr)
                        errors.append(msg)

                top1_hits += t1
                top3_hits += t3
                reciprocal_ranks.append(rr)

            mrr_calc = sum(reciprocal_ranks) / n
            top1_ci = independent_wilson_score_interval(top1_hits, n)
            top3_ci = independent_wilson_score_interval(top3_hits, n)

            if m not in summary:
                errors.append(f"[{sec_name}] Method '{m}' has per-query records but no summary entry")
                continue
            expected = summary[m]
            if expected.get("top1_hits") != top1_hits:
                errors.append(f"[{sec_name}] Method '{m}' top1_hits mismatch: expected {expected.get('top1_hits')}, calculated {top1_hits}")
            if expected.get("top3_hits") != top3_hits:
                errors.append(f"[{sec_name}] Method '{m}' top3_hits mismatch: expected {expected.get('top3_hits')}, calculated {top3_hits}")
            if abs(expected.get("mrr", 0.0) - mrr_calc) > 1e-4:
                errors.append(f"[{sec_name}] Method '{m}' MRR mismatch: expected {expected.get('mrr')}, calculated {mrr_calc:.4f}")
            # The percentages and both intervals are what the docs actually render; check them too.
            for key, hits in (("top1_recall_pct", top1_hits), ("top3_recall_pct", top3_hits)):
                exp_pct = expected.get(key)
                if exp_pct is None or abs(exp_pct - 100.0 * hits / n) > 0.051:
                    errors.append(f"[{sec_name}] Method '{m}' {key} mismatch: expected {exp_pct}, calculated {100.0 * hits / n:.1f}")
            for key, calc in (("top1_ci_95", top1_ci), ("top3_ci_95", top3_ci)):
                exp_ci = expected.get(key)
                if not exp_ci or abs(exp_ci[0] - calc[0]) > 1e-3 or abs(exp_ci[1] - calc[1]) > 1e-3:
                    errors.append(f"[{sec_name}] Method '{m}' {key} mismatch: expected {exp_ci}, calculated {calc}")

    return len(errors) == 0, errors


REQUIRED_CONCLUSION_SCRIPTS = (
    "record_context_boundary_bisection.py",
    "record_hf_vs_bitnet_raw_outputs.py",
    "record_speed_memory.py",
    "validate_270m_embedder.py",
)


def lint_script_conclusions(scripts_dir: Path) -> tuple[bool, list[str]]:
    """
    AST check that functions building conclusion text contain no numeric or
    digit-bearing string literals except through interpolation, and that all
    empirical benchmark scripts that emit conclusions define a governed builder.
    """
    errors = []
    if not scripts_dir.is_dir():
        return True, []

    for py_file in sorted(scripts_dir.glob("*.py")):
        if py_file.name in ("sync_docs.py", "sync_benchmark_docs.py"):
            continue
        try:
            tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        except Exception as e:
            errors.append(f"Failed to parse AST for {py_file.name}: {e}")
            continue

        found_conclusion_fn = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if "conclusion" in node.name.lower() or "build_conclusion" in node.name.lower():
                    found_conclusion_fn = True
                    # Identify docstring node if present so it can be skipped during literal check
                    docstring_node = (
                        node.body[0].value
                        if (node.body and isinstance(node.body[0], ast.Expr))
                        else None
                    )
                    # Walk inside function body statements (excludes argument defaults)
                    for stmt in node.body:
                        for child in ast.walk(stmt):
                            # Standalone constant string
                            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                                # Ignore docstring
                                if child is docstring_node or child == docstring_node:
                                    continue
                                if re.search(r"\d", child.value):
                                    errors.append(
                                        f"Script AST lint: {py_file.name}::{node.name} line {getattr(child, 'lineno', '?')} "
                                        f"contains hardcoded numeric literal in string: {child.value!r}"
                                    )
                            elif (
                                isinstance(child, ast.Constant)
                                and isinstance(child.value, (int, float))
                                and not isinstance(child.value, bool)
                            ):
                                errors.append(
                                    f"Script AST lint: {py_file.name}::{node.name} line {getattr(child, 'lineno', '?')} "
                                    f"contains hardcoded numeric literal: {child.value!r}"
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

        if py_file.name in REQUIRED_CONCLUSION_SCRIPTS and not found_conclusion_fn:
            errors.append(
                f"Script AST lint: {py_file.name} must define a conclusion builder function "
                f"(*conclusion*) governed by AST lint."
            )

    return len(errors) == 0, errors


def lint_prose_and_words(
    content: str, doc_name: str = "document", registry_path: Path | None = None
) -> tuple[bool, list[str]]:
    """
    Word lint:
    - Prohibit 'bulletproof', 'conclusive', 'definitive' anywhere.
    - Guarded words ('verified', 'authentic', 'proves', 'zero hallucination', '100%')
      allowed only when tied to a registered claim marker <!-- claim:... --> or <!-- claim_term:... -->.
    - Validate claim_term markers against registered claim_terms in claims_registry.json.
    """
    errors = []

    # Check prohibited words
    for pat in PROHIBITED_WORDS:
        m = pat.search(content)
        if m:
            errors.append(f"Word lint failure in {doc_name}: prohibited adjective '{m.group(0)}' found.")

    # Load claim_terms from claims registry
    if registry_path is None:
        registry_path = DEFAULT_REGISTRY_PATH
    claim_terms = {}
    if registry_path and registry_path.is_file():
        try:
            with open(registry_path, "r", encoding="utf-8") as f:
                claim_terms = json.load(f).get("claim_terms", {})
        except Exception as e:
            errors.append(f"Failed to load claim_terms from {registry_path.name}: {e}")

    # Validate claim_term markers
    claim_term_pat = re.compile(r"<!--\s*claim_term:([a-zA-Z0-9_\-]+)\s*-->(.*?)<!--\s*/claim_term\s*-->", re.DOTALL)
    for m in claim_term_pat.finditer(content):
        term_id = m.group(1)
        wrapped_text = m.group(2).strip()
        if term_id not in claim_terms:
            errors.append(f"Word lint failure in {doc_name}: unknown claim_term ID '{term_id}'.")
        else:
            expected_term = claim_terms[term_id].get("term", "")
            if wrapped_text.lower() != expected_term.lower():
                errors.append(
                    f"Word lint failure in {doc_name}: claim_term '{term_id}' wrapped text {wrapped_text!r} "
                    f"does not match registered term {expected_term!r}."
                )

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

    try:
        doc_rel = str(doc_path.resolve().relative_to(repo_root.resolve())).replace("\\", "/")
    except ValueError:
        doc_rel = None

    if doc_rel:
        for claim_id, claim_cfg in claims.items():
            used_in = [u.replace("\\", "/") for u in claim_cfg.get("used_in", [])]
            if doc_rel in used_in and claim_id not in found_claims:
                errors.append(
                    f"Missing required claim marker '<!-- claim:{claim_id} -->' in {doc_rel} "
                    f"(registered in claims_registry.json 'used_in')"
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
    word_ok, word_errors = lint_prose_and_words(
        doc_path.read_text(encoding="utf-8"), doc_path.name, registry_path=registry_path
    )
    if not word_ok:
        errors.extend(word_errors)

    if errors:
        return False, errors
    return True, messages


def main():
    parser = argparse.ArgumentParser(description="Synchronize and lint documentation claims against empirical artifacts.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="Check documentation claims and empirical regions without modifying files (DEFAULT)",
    )
    mode.add_argument(
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
        default=None,
        help="Path to markdown document to check/sync (defaults to all documents listed in claims_registry.json 'used_in')",
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
    if args.doc is not None:
        target_docs = [args.doc]
    else:
        target_docs = [DEFAULT_ROADMAP_PATH]
        if args.registry.is_file():
            try:
                reg_data = json.loads(args.registry.read_text(encoding="utf-8"))
                for c_cfg in reg_data.get("claims", {}).values():
                    for rel_p in c_cfg.get("used_in", []):
                        candidate = REPO_ROOT / rel_p
                        if candidate not in target_docs:
                            target_docs.append(candidate)
            except Exception:
                pass

    for doc_target in target_docs:
        doc_ok, doc_msgs = sync_document(doc_target, args.registry, write=do_write, repo_root=REPO_ROOT)
        if not doc_ok:
            overall_success = False
            for err in doc_msgs:
                print(f"[FAIL] {err}", file=sys.stderr)
        else:
            for msg in doc_msgs:
                print(f"[INFO] {msg}")
            print(f"[PASS] Document claims and empirical constraints verified for {doc_target.name}.")

    if overall_success:
        print("\nAll claims and empirical guardrails verified successfully.")
        sys.exit(0)
    else:
        print("\nClaims synchronization / lint failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
