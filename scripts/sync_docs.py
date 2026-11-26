#!/usr/bin/env python3
"""
sync_docs.py: Claims Registry Documentation Synchronizer & Linter v2.

Enforces:
1. Documentation claim markers `<!-- claim:<id> -->...<!-- /claim -->` strictly match
   empirical artifacts registered in docs/claims_registry.json.
2. Inside regions marked as empirical (`<!-- BEGIN_EMPIRICAL -->` ... `<!-- END_EMPIRICAL -->`
   and `<!-- BEGIN_RETRIEVAL_BENCHMARK_TABLE -->` ... `<!-- END_RETRIEVAL_BENCHMARK_TABLE -->`),
   any digit outside a claim marker fails the check.
3. Word & personal-path lint: adjectives such as 'bulletproof', 'rock solid', 'conclusive',
   'definitive', and 'infinitely scalable' are prohibited anywhere. Guarded terms ('verified',
   'authentic', 'proves', 'zero hallucination', '100%') are allowed only when tied to a
   registered claim whose on-disk evidence exists. Personal host paths (/Users/, /home/,
   Windows C:\\Users\\) are forbidden in docs and JSON artifacts.
4. Script AST lint: functions building conclusion text contain no numeric or digit-bearing
   string literals except through interpolation (excluding format_spec precision digits).
5. Independent recomputation of Wilson intervals, hit counts, and MRR from per-query records
   under strict JSON parsing (rejecting NaN/Infinity/1e309 float overflow and duplicate keys).
6. Registry referential hygiene (no orphaned/duplicate used_in entries, repo-confined paths,
   verified claim_term evidence) and cross-doc balanced marker validation.
7. Extraction and rendering of code constants (MAX_SAFE_GENERATE_TOKENS, MAX_SAFE_EMBED_TOKENS).

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

_PERSONAL_HOST_PATH_RE = re.compile(r"(?:/Users/|/home/|[A-Za-z]:(?:\\|/)[Uu]sers(?:\\|/))")

PROHIBITED_WORDS = [
    re.compile(r"\bbullet-?proof\b", re.IGNORECASE),
    re.compile(r"\brock\s+solid\b", re.IGNORECASE),
    re.compile(r"\bconclusive\b", re.IGNORECASE),
    re.compile(r"\bdefinitive\b", re.IGNORECASE),
    re.compile(r"\binfinitely\s+scalable\b", re.IGNORECASE),
]

GUARDED_WORDS = [
    re.compile(r"\bverified\b", re.IGNORECASE),
    re.compile(r"\bauthentic\b", re.IGNORECASE),
    re.compile(r"\bproves\b", re.IGNORECASE),
    re.compile(r"\bzero\s+hallucination\b", re.IGNORECASE),
    re.compile(r"100%"),
]


_SAFE_CALLS = {"int": int, "float": float, "str": str, "min": min, "max": max}


def _reject_nonfinite_json_constant(const: str) -> None:
    raise ValueError(f"Non-standard non-finite JSON constant not permitted: {const!r}")


def _parse_finite_json_float(raw: str) -> float:
    val = float(raw)
    if not math.isfinite(val):
        raise ValueError(f"Non-finite JSON floating-point value not permitted: {raw!r}")
    return val


def _reject_duplicate_json_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in pairs:
        if k in out:
            raise ValueError(f"Duplicate JSON object member not permitted: {k!r}")
        out[k] = v
    return out


def _loads_strict_json(raw: str):
    return json.loads(
        raw,
        parse_float=_parse_finite_json_float,
        parse_constant=_reject_nonfinite_json_constant,
        object_pairs_hook=_reject_duplicate_json_keys,
    )


def _has_personal_host_path_text(text: str) -> bool:
    """Detect leaked macOS (/Users/), Linux (/home/), or Windows (C:\\Users\\) personal paths."""
    return _PERSONAL_HOST_PATH_RE.search(text) is not None


def _contains_personal_host_path(obj) -> bool:
    """Recursively check decoded JSON strings for personal host prefixes (catching JSON-escaped slashes)."""
    if isinstance(obj, str):
        return _has_personal_host_path_text(obj)
    if isinstance(obj, dict):
        return any(_contains_personal_host_path(k) or _contains_personal_host_path(v) for k, v in obj.items())
    if isinstance(obj, list):
        return any(_contains_personal_host_path(v) for v in obj)
    return False


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_finite_num(v) -> bool:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:
        return math.isfinite(float(v))
    except (OverflowError, ValueError, TypeError):
        return False


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


def resolve_claim_value(
    claim_id: str,
    claim_cfg: dict,
    repo_root: Path = REPO_ROOT,
    validated_json_cache: dict[Path, dict] | None = None,
) -> str:
    """Resolve and format a registered claim value from its empirical source artifact."""
    artifact_rel = claim_cfg.get("source_artifact")
    if not isinstance(artifact_rel, str) or not artifact_rel.strip():
        raise ValueError(f"Claim '{claim_id}' missing 'source_artifact'")
    if Path(artifact_rel).is_absolute():
        raise ValueError(f"Claim '{claim_id}' source_artifact must be repository-relative: {artifact_rel!r}")
    artifact_path = repo_root / artifact_rel
    try:
        resolved_artifact = artifact_path.resolve()
        resolved_root = repo_root.resolve()
    except (ValueError, OSError) as exc:
        raise ValueError(f"Claim '{claim_id}' has invalid source_artifact {artifact_rel!r}: {exc}") from exc
    try:
        resolved_artifact.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"Claim '{claim_id}' source_artifact escapes repository root: {artifact_rel!r}") from exc

    fmt = claim_cfg.get("format_rule", "{value}")

    # Case 1: Python constant
    if "python_const" in claim_cfg:
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Source artifact not found for claim '{claim_id}': {artifact_path}")
        val = extract_python_constant(artifact_path, claim_cfg["python_const"])
        return apply_format_rule(claim_id, fmt, val)

    # Case 2: JSON artifact path
    if "json_path" in claim_cfg:
        if validated_json_cache is not None and resolved_artifact in validated_json_cache:
            data = validated_json_cache[resolved_artifact]
        else:
            if not artifact_path.is_file():
                raise FileNotFoundError(f"Source artifact not found for claim '{claim_id}': {artifact_path}")
            data = _loads_strict_json(artifact_path.read_text(encoding="utf-8"))
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


def independent_recompute_metrics(
    benchmark_path: Path,
    validated_json_cache: dict[Path, dict] | None = None,
) -> tuple[bool, list[str]]:
    """
    Recompute retrieval hits, MRR, and Wilson intervals directly from query_details
    and assert exact match with summary metrics in JSON artifact under strict JSON parsing.
    """
    errors: list[str] = []
    if not benchmark_path.is_file():
        return False, [f"Missing benchmark JSON: {benchmark_path}"]

    try:
        raw_text = benchmark_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return False, [f"Failed to read {benchmark_path}: {exc}"]
    if _has_personal_host_path_text(raw_text):
        errors.append(f"Benchmark JSON {benchmark_path.name} contains leaked personal host path")

    try:
        data = _loads_strict_json(raw_text)
    except Exception as exc:
        return False, [f"Failed to parse {benchmark_path}: {exc}"]

    if not isinstance(data, dict):
        return False, [f"Benchmark JSON {benchmark_path.name} root must be an object, got {type(data).__name__}"]
    if _contains_personal_host_path(data):
        errors.append(f"Benchmark JSON {benchmark_path.name} contains leaked personal host path in decoded strings")

    full = data.get("full_corpus_benchmark")
    if not isinstance(full, dict) or "query_details" not in full:
        return False, ["benchmark artifact has no full_corpus_benchmark.query_details; cannot recompute metrics"]

    sections = {
        k: v for k, v in data.items()
        if isinstance(v, dict) and "query_details" in v
    }

    for sec_name, sec_data in sections.items():
        query_details = sec_data["query_details"]
        summary = sec_data.get("summary", {})
        if not isinstance(query_details, list):
            errors.append(f"[{sec_name}] query_details must be a list, got {type(query_details).__name__}")
            continue
        if not isinstance(summary, dict):
            errors.append(f"[{sec_name}] summary must be an object, got {type(summary).__name__}")
            continue
        n = len(query_details)
        if n == 0:
            errors.append(f"[{sec_name}] query_details is empty in benchmark artifact")
            continue

        declared_n = sec_data.get("queries_count")
        if not _is_int(declared_n) or declared_n != n:
            errors.append(f"[{sec_name}] queries_count mismatch: declared {declared_n!r}, query_details has {n}")

        first_rankings = query_details[0].get("rankings") if isinstance(query_details[0], dict) else None
        if not isinstance(first_rankings, dict) or not first_rankings:
            errors.append(f"[{sec_name}] first query record is missing non-empty rankings object")
            continue
        methods = list(first_rankings.keys())
        for q in query_details:
            if not isinstance(q, dict) or not isinstance(q.get("rankings"), dict):
                errors.append(f"[{sec_name}] malformed query record without rankings object")
                continue
            if set(q.get("rankings", {}).keys()) != set(methods):
                errors.append(f"[{sec_name}] query {q.get('id', '?')!r} ranks a different method set than the first query")
        for m in summary:
            if m not in methods:
                errors.append(f"[{sec_name}] summary method '{m}' has no per-query records")

        for m in methods:
            top1_hits = 0
            top3_hits = 0
            reciprocal_ranks: list[float] = []

            for q in query_details:
                if not isinstance(q, dict) or not isinstance(q.get("rankings"), dict):
                    continue
                r = q["rankings"].get(m, {})
                if not isinstance(r, dict):
                    errors.append(f"[{sec_name}] Method '{m}' query {q.get('id', '?')!r}: ranking entry must be an object")
                    continue
                raw_t1 = r.get("top1_hit")
                raw_t3 = r.get("top3_hit")
                raw_rr = r.get("rr")
                if not isinstance(raw_t1, bool) or not isinstance(raw_t3, bool):
                    errors.append(
                        f"[{sec_name}] Method '{m}' query {q.get('id', '?')!r}: "
                        f"top1_hit ({raw_t1!r}) and top3_hit ({raw_t3!r}) must be booleans"
                    )
                    continue
                if not _is_finite_num(raw_rr) or not (0.0 <= float(raw_rr) <= 1.0):
                    errors.append(
                        f"[{sec_name}] Method '{m}' query {q.get('id', '?')!r}: rr must be a finite number in [0, 1], got {raw_rr!r}"
                    )
                    continue
                t1, t3, rr = raw_t1, raw_t3, float(raw_rr)
                if t1 != (abs(rr - 1.0) < 1e-4) or t3 != (rr >= 1.0 / 3.0 - 1e-4) or (t1 and not t3):
                    errors.append(f"[{sec_name}] Method '{m}' query {q.get('id', '?')!r}: flags (top1={t1}, top3={t3}) disagree with rr={rr}")

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

                top1_hits += int(t1)
                top3_hits += int(t3)
                reciprocal_ranks.append(rr)

            mrr_calc = sum(reciprocal_ranks) / n
            top1_ci = independent_wilson_score_interval(top1_hits, n)
            top3_ci = independent_wilson_score_interval(top3_hits, n)

            if m not in summary:
                errors.append(f"[{sec_name}] Method '{m}' has per-query records but no summary entry")
                continue
            expected = summary[m]
            if not isinstance(expected, dict):
                errors.append(f"[{sec_name}] Method '{m}' summary entry must be an object")
                continue
            if not _is_int(expected.get("top1_hits")) or expected.get("top1_hits") != top1_hits:
                errors.append(f"[{sec_name}] Method '{m}' top1_hits mismatch: expected {expected.get('top1_hits')}, calculated {top1_hits}")
            if not _is_int(expected.get("top3_hits")) or expected.get("top3_hits") != top3_hits:
                errors.append(f"[{sec_name}] Method '{m}' top3_hits mismatch: expected {expected.get('top3_hits')}, calculated {top3_hits}")
            if not _is_finite_num(expected.get("mrr")) or abs(float(expected.get("mrr")) - mrr_calc) > 1e-4:
                errors.append(f"[{sec_name}] Method '{m}' MRR mismatch: expected {expected.get('mrr')}, calculated {mrr_calc:.4f}")
            for key, hits in (("top1_recall_pct", top1_hits), ("top3_recall_pct", top3_hits)):
                exp_pct = expected.get(key)
                if not _is_finite_num(exp_pct) or abs(float(exp_pct) - 100.0 * hits / n) > 0.051:
                    errors.append(f"[{sec_name}] Method '{m}' {key} mismatch: expected {exp_pct}, calculated {100.0 * hits / n:.1f}")
            for key, calc in (("top1_ci_95", top1_ci), ("top3_ci_95", top3_ci)):
                exp_ci = expected.get(key)
                if (
                    not isinstance(exp_ci, list)
                    or len(exp_ci) != 2
                    or not _is_finite_num(exp_ci[0])
                    or not _is_finite_num(exp_ci[1])
                    or abs(float(exp_ci[0]) - calc[0]) > 1e-3
                    or abs(float(exp_ci[1]) - calc[1]) > 1e-3
                ):
                    errors.append(f"[{sec_name}] Method '{m}' {key} mismatch: expected {exp_ci}, calculated {calc}")

    if not errors and validated_json_cache is not None:
        validated_json_cache[benchmark_path.resolve()] = data
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
    digit-bearing string literals except through interpolation (excluding format_spec
    precision tokens like ':.4f'), and that all empirical benchmark scripts that emit
    conclusions define a governed builder.
    """
    errors: list[str] = []
    if not scripts_dir.is_dir():
        return True, []

    def _walk_excluding_format_specs(node: ast.AST):
        yield node
        for field_name, field_val in ast.iter_fields(node):
            if isinstance(node, ast.FormattedValue) and field_name == "format_spec":
                continue
            if isinstance(field_val, list):
                for item in field_val:
                    if isinstance(item, ast.AST):
                        yield from _walk_excluding_format_specs(item)
            elif isinstance(field_val, ast.AST):
                yield from _walk_excluding_format_specs(field_val)

    for py_file in sorted(scripts_dir.glob("*.py")):
        if py_file.name in ("sync_docs.py", "sync_benchmark_docs.py", "audit-claims.py"):
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
                    docstring_node = (
                        node.body[0].value
                        if (
                            node.body
                            and isinstance(node.body[0], ast.Expr)
                            and isinstance(node.body[0].value, ast.Constant)
                            and isinstance(node.body[0].value.value, str)
                        )
                        else None
                    )
                    for stmt in node.body:
                        for child in _walk_excluding_format_specs(stmt):
                            if child is docstring_node:
                                continue
                            if isinstance(child, ast.Constant) and isinstance(child.value, str):
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

        if py_file.name in REQUIRED_CONCLUSION_SCRIPTS and not found_conclusion_fn:
            errors.append(
                f"Script AST lint: {py_file.name} must define a conclusion builder function "
                f"(*conclusion*) governed by AST lint."
            )

    return len(errors) == 0, errors


def _verify_claim_term_evidence(term_id: str, term_cfg: dict, repo_root: Path) -> list[str]:
    """Verify that a claim_term entry declares non-empty evidence files/symbols that exist in repo_root."""
    errs: list[str] = []
    evidence_list = term_cfg.get("evidence", [])
    if not isinstance(evidence_list, list) or not evidence_list:
        return [f"claim_term '{term_id}' is missing required non-empty 'evidence' list in claims_registry.json"]

    resolved_root = repo_root.resolve()
    for item in evidence_list:
        file_part, sep, symbol_part = str(item).partition(":")
        if not file_part or Path(file_part).is_absolute():
            errs.append(f"claim_term '{term_id}' evidence path must be repository-relative: {file_part!r}")
            continue
        try:
            target = (repo_root / file_part).resolve()
            target.relative_to(resolved_root)
        except (ValueError, OSError):
            errs.append(f"claim_term '{term_id}' evidence path escapes repository root or is invalid: {file_part!r}")
            continue
        if not target.is_file():
            errs.append(f"claim_term '{term_id}' evidence file does not exist: {file_part}")
            continue
        if sep and symbol_part:
            text = target.read_text(encoding="utf-8")
            if symbol_part not in text:
                errs.append(f"claim_term '{term_id}' evidence symbol {symbol_part!r} not found in {file_part}")
    return errs


def lint_prose_and_words(
    content: str,
    doc_name: str = "document",
    registry_path: Path | None = None,
    repo_root: Path = REPO_ROOT,
) -> tuple[bool, list[str]]:
    """
    Word & personal-path lint:
    - Prohibit 'bulletproof', 'rock solid', 'conclusive', 'definitive', 'infinitely scalable' anywhere.
    - Guarded words ('verified', 'authentic', 'proves', 'zero hallucination', '100%')
      allowed only when tied to a registered claim marker <!-- claim:... --> or <!-- claim_term:... -->
      whose on-disk evidence exists.
    - Prohibit leaked personal host paths (/Users/, /home/, C:\\Users\\).
    """
    errors: list[str] = []

    for idx, line in enumerate(content.splitlines(), 1):
        if _has_personal_host_path_text(line):
            errors.append(f"Personal path leak in {doc_name}:{idx}: found personal host path.")
        for pat in PROHIBITED_WORDS:
            m = pat.search(line)
            if m:
                errors.append(
                    f"Word lint failure in {doc_name}:{idx}: prohibited adjective '{m.group(0)}' found."
                )

    if registry_path is None:
        registry_path = DEFAULT_REGISTRY_PATH
    claim_terms = {}
    if registry_path and registry_path.is_file():
        try:
            claim_terms = _loads_strict_json(registry_path.read_text(encoding="utf-8")).get("claim_terms", {})
        except Exception as e:
            errors.append(f"Failed to load claim_terms from {registry_path.name}: {e}")

    verified_term_ids: set[str] = set()
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
            if term_id not in verified_term_ids:
                verified_term_ids.add(term_id)
                errors.extend(_verify_claim_term_evidence(term_id, claim_terms[term_id], repo_root))

    masked = re.sub(r"<!--\s*claim:[^>]+-->.*?<!--\s*/claim\s*-->", " ", content, flags=re.DOTALL)
    masked = re.sub(r"<!--\s*claim_term:[^>]+-->.*?<!--\s*/claim_term\s*-->", " ", masked, flags=re.DOTALL)

    for pat in GUARDED_WORDS:
        for match in pat.finditer(masked):
            line_no = masked[: match.start()].count("\n") + 1
            errors.append(
                f"Word lint failure in {doc_name} line {line_no}: guarded term '{match.group(0)}' "
                f"used in bare text without being tied to a registered claim marker."
            )

    return len(errors) == 0, errors


def _parse_gfm_row(
    raw_line: str,
    require_outer_pipes: bool = False,
    min_cells: int = 2,
) -> tuple[str, list[str]] | None:
    if raw_line.startswith("    ") or raw_line.startswith("\t"):
        return None
    line = raw_line.strip()
    if not line or "|" not in line:
        return None
    if require_outer_pipes and not (line.startswith("|") and line.endswith("|") and len(line) >= 3):
        return None
    inner = line
    if inner.startswith("|"):
        inner = inner[1:]
    if inner.endswith("|") and not inner.endswith(r"\|"):
        inner = inner[:-1]
    cells = [c.strip() for c in re.split(r"(?<!\\)\|", inner)]
    if len(cells) < min_cells:
        return None
    return line, cells


def _is_gfm_delimiter_row(cells: list[str]) -> bool:
    return len(cells) >= 2 and all(re.fullmatch(r":?-{1,}:?", c) is not None for c in cells)


def _strip_fenced_code_blocks_preserve_lines(content: str) -> str:
    """Blank out CommonMark backtick/tilde fenced code blocks while preserving 1-based line numbers."""
    out: list[str] = []
    fence_char: str | None = None
    fence_len = 0
    for raw_line in content.splitlines():
        if fence_char is not None:
            close_m = re.match(r"^ {0,3}(" + re.escape(fence_char) + r"{3,})[ \t]*$", raw_line)
            if close_m and len(close_m.group(1)) >= fence_len:
                fence_char = None
                fence_len = 0
            out.append("")
            continue
        open_m = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", raw_line)
        if open_m and (open_m.group(1)[0] == "~" or "`" not in open_m.group(2)):
            fence_char = open_m.group(1)[0]
            fence_len = len(open_m.group(1))
            out.append("")
            continue
        out.append(raw_line)
    return "\n".join(out)


def lint_claim_markers_and_regions(
    content: str,
    doc_name: str = "document",
    registry_path: Path | None = None,
) -> tuple[bool, list[str]]:
    """
    Validate structural integrity of `<!-- claim:... -->`, `<!-- claim_term:... -->`, and
    empirical block delimiters across ANY scanned Markdown file.
    """
    errors: list[str] = []
    if registry_path is None:
        registry_path = DEFAULT_REGISTRY_PATH

    known_claims: set[str] = set()
    known_terms: set[str] = set()
    if registry_path and registry_path.is_file():
        try:
            reg = _loads_strict_json(registry_path.read_text(encoding="utf-8"))
            known_claims = set(reg.get("claims", {}).keys())
            known_terms = set(reg.get("claim_terms", {}).keys())
        except Exception as exc:
            errors.append(f"Failed to parse {registry_path.name}: {exc}")
            return False, errors

    stripped = _strip_fenced_code_blocks_preserve_lines(content)

    for begin_marker, end_marker in EMPIRICAL_BLOCK_DELIMITERS:
        b_count = stripped.count(begin_marker)
        e_count = stripped.count(end_marker)
        if b_count != e_count:
            errors.append(
                f"Marker lint failure in {doc_name}: unbalanced empirical delimiters "
                f"({b_count} '{begin_marker}' vs {e_count} '{end_marker}')"
            )

    pos = 0
    open_tag: tuple[str, str, int] | None = None
    tag_start_re = re.compile(r"<!--\s*(/?claim(?:_term)?)\b")
    full_open_re = re.compile(r"^<!--\s*(claim|claim_term):([a-zA-Z0-9_\-]+)\s*-->$")
    full_close_re = re.compile(r"^<!--\s*/(claim|claim_term)\s*-->$")

    while True:
        m = tag_start_re.search(stripped, pos)
        if not m:
            break
        start = m.start()
        line_no = stripped[:start].count("\n") + 1
        end = stripped.find("-->", start + 4)
        if end == -1:
            errors.append(f"Marker lint failure in {doc_name}:{line_no}: unclosed HTML claim marker comment")
            break
        raw_comment = stripped[start : end + 3]
        pos = end + 3

        mo = full_open_re.match(raw_comment)
        mc = full_close_re.match(raw_comment)
        if mo:
            kind, cid = mo.group(1), mo.group(2)
            if open_tag is not None:
                errors.append(
                    f"Marker lint failure in {doc_name}:{line_no}: nested or unclosed "
                    f"'{open_tag[0]}:{open_tag[1]}' (opened at line {open_tag[2]}) before '{kind}:{cid}'"
                )
            open_tag = (kind, cid, line_no)
            if kind == "claim" and known_claims and cid not in known_claims:
                errors.append(f"Marker lint failure in {doc_name}:{line_no}: unknown claim ID '{cid}'")
            elif kind == "claim_term" and known_terms and cid not in known_terms:
                errors.append(f"Marker lint failure in {doc_name}:{line_no}: unknown claim_term ID '{cid}'")
        elif mc:
            close_kind = mc.group(1)
            if open_tag is None:
                errors.append(
                    f"Marker lint failure in {doc_name}:{line_no}: orphan closing '<!-- /{close_kind} -->' without opening marker"
                )
            elif open_tag[0] != close_kind:
                errors.append(
                    f"Marker lint failure in {doc_name}:{line_no}: mismatched closing '<!-- /{close_kind} -->' "
                    f"for '{open_tag[0]}:{open_tag[1]}' opened at line {open_tag[2]}"
                )
                open_tag = None
            else:
                open_tag = None
        else:
            errors.append(f"Marker lint failure in {doc_name}:{line_no}: malformed claim marker {raw_comment!r}")

    if open_tag is not None:
        errors.append(
            f"Marker lint failure in {doc_name}:{open_tag[2]}: unclosed '<!-- {open_tag[0]}:{open_tag[1]} -->' (missing '<!-- /{open_tag[0]} -->')"
        )

    emp_ok, emp_errs = check_empirical_regions(stripped, doc_name)
    if not emp_ok:
        errors.extend(emp_errs)

    return len(errors) == 0, errors


def validate_claims_registry(
    registry_path: Path,
    repo_root: Path = REPO_ROOT,
) -> tuple[bool, list[str]]:
    """
    Validate structural & referential integrity of docs/claims_registry.json:
    - Strict JSON (no duplicate keys or non-finite numbers) and zero personal paths
    - Repo-confined existing source_artifacts (with strict JSON & path-leak check for JSON files)
    - Non-empty, deduplicated `used_in` lists pointing to existing repo-confined docs containing each marker
    - All `claim_terms` evidence files and symbols exist on disk
    """
    errors: list[str] = []
    if not registry_path.is_file():
        return False, [f"Registry not found: {registry_path}"]
    try:
        raw_reg = registry_path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return False, [f"Failed to read {registry_path}: {exc}"]
    if _has_personal_host_path_text(raw_reg):
        errors.append(f"Registry {registry_path.name} contains leaked personal host path")
    try:
        reg = _loads_strict_json(raw_reg)
    except Exception as exc:
        return False, [f"Failed to parse {registry_path.name}: {exc}"]
    if not isinstance(reg, dict):
        return False, [f"Registry {registry_path.name} root must be a JSON object"]
    if _contains_personal_host_path(reg):
        errors.append(f"Registry {registry_path.name} contains leaked personal host path in decoded strings")

    claims = reg.get("claims")
    if not isinstance(claims, dict) or not claims:
        return False, [f"Registry {registry_path.name} must define a non-empty 'claims' object"]

    resolved_root = repo_root.resolve()
    doc_cache: dict[Path, str] = {}
    json_checked: set[Path] = set()

    for cid, cfg in claims.items():
        if not isinstance(cfg, dict):
            errors.append(f"Claim '{cid}' config must be a JSON object")
            continue
        art_rel = cfg.get("source_artifact")
        if not isinstance(art_rel, str) or not art_rel.strip() or Path(art_rel).is_absolute():
            errors.append(f"Claim '{cid}' has missing or non-relative source_artifact: {art_rel!r}")
        else:
            try:
                art_path = (repo_root / art_rel).resolve()
            except (ValueError, OSError) as exc:
                errors.append(f"Claim '{cid}' has invalid source_artifact {art_rel!r}: {exc}")
                art_path = None
            if art_path is not None:
                try:
                    art_path.relative_to(resolved_root)
                except ValueError:
                    errors.append(f"Claim '{cid}' source_artifact escapes repository root: {art_rel!r}")
                else:
                    if not art_path.is_file():
                        errors.append(f"Claim '{cid}' required source_artifact does not exist: {art_rel}")
                    elif "json_path" in cfg and art_path not in json_checked:
                        json_checked.add(art_path)
                        try:
                            raw_art = art_path.read_text(encoding="utf-8")
                            if _has_personal_host_path_text(raw_art):
                                errors.append(f"Claim '{cid}' source_artifact {art_rel} contains leaked personal host path")
                            parsed_art = _loads_strict_json(raw_art)
                            if _contains_personal_host_path(parsed_art):
                                errors.append(f"Claim '{cid}' source_artifact {art_rel} contains leaked personal host path in decoded strings")
                        except Exception as exc:
                            errors.append(f"Claim '{cid}' source_artifact {art_rel} failed strict JSON parse: {exc}")

        used_in = cfg.get("used_in")
        if not isinstance(used_in, list) or not used_in:
            errors.append(f"Claim '{cid}' must declare a non-empty 'used_in' document list (orphaned claim)")
            continue
        seen_docs: set[str] = set()
        marker_re = re.compile(rf"<!--\s*claim:{re.escape(cid)}\s*-->.*?<!--\s*/claim\s*-->", re.DOTALL)
        for rel_doc in used_in:
            if not isinstance(rel_doc, str) or not rel_doc.strip() or Path(rel_doc).is_absolute():
                errors.append(f"Claim '{cid}' has invalid 'used_in' entry: {rel_doc!r}")
                continue
            norm_rel = rel_doc.replace("\\", "/")
            if norm_rel in seen_docs:
                errors.append(f"Claim '{cid}' has duplicate 'used_in' entry: {norm_rel}")
                continue
            seen_docs.add(norm_rel)
            try:
                doc_p = (repo_root / norm_rel).resolve()
            except (ValueError, OSError) as exc:
                errors.append(f"Claim '{cid}' has invalid 'used_in' path {norm_rel!r}: {exc}")
                continue
            try:
                doc_p.relative_to(resolved_root)
            except ValueError:
                errors.append(f"Claim '{cid}' 'used_in' path escapes repository root: {norm_rel}")
                continue
            if not doc_p.is_file():
                errors.append(f"Claim '{cid}' 'used_in' document does not exist: {norm_rel}")
                continue
            if doc_p not in doc_cache:
                doc_cache[doc_p] = doc_p.read_text(encoding="utf-8")
            if not marker_re.search(doc_cache[doc_p]):
                errors.append(f"Claim '{cid}' listed in 'used_in' for {norm_rel}, but marker is missing")

    claim_terms = reg.get("claim_terms", {})
    if isinstance(claim_terms, dict):
        for tid, tcfg in claim_terms.items():
            if isinstance(tcfg, dict):
                errors.extend(_verify_claim_term_evidence(tid, tcfg, repo_root))

    return len(errors) == 0, errors


def check_empirical_regions(content: str, doc_name: str = "document") -> tuple[bool, list[str]]:
    """
    Verify that inside regions marked as empirical, any empirical metric digit outside a claim marker fails.
    For markdown tables, structural GFM header/delimiter rows, method names (col 1), and role descriptions
    (last col) are excluded, ensuring all quantitative benchmark cells are strictly governed by claims.
    """
    errors: list[str] = []
    for begin_marker, end_marker in EMPIRICAL_BLOCK_DELIMITERS:
        pattern = re.escape(begin_marker) + r"(.*?)" + re.escape(end_marker)
        for match in re.finditer(pattern, content, re.DOTALL):
            block_text = match.group(1)
            block_lines = block_text.splitlines()
            skip_indices: set[int] = set()
            for idx_line in range(len(block_lines) - 1):
                hdr_row = _parse_gfm_row(block_lines[idx_line], require_outer_pipes=False)
                delim_row = _parse_gfm_row(block_lines[idx_line + 1], require_outer_pipes=False)
                if (
                    hdr_row is not None
                    and delim_row is not None
                    and len(hdr_row[1]) == len(delim_row[1])
                    and _is_gfm_delimiter_row(delim_row[1])
                ):
                    skip_indices.add(idx_line)
                    skip_indices.add(idx_line + 1)

            filtered_lines = []
            for row_idx, line in enumerate(block_lines):
                line_str = line.strip()
                if not line_str:
                    continue
                if row_idx in skip_indices:
                    continue
                parsed_row = _parse_gfm_row(line, require_outer_pipes=False, min_cells=2)
                if parsed_row is not None:
                    _, parts = parsed_row
                    if _is_gfm_delimiter_row(parts):
                        continue
                    if len(parts) >= 3:
                        metric_cells = " | ".join(parts[1:-1])
                        filtered_lines.append(metric_cells)
                        continue
                filtered_lines.append(line_str)
            block_to_check = "\n".join(filtered_lines)

            stripped = re.sub(r"<!--\s*claim:[^>]+-->.*?<!--\s*/claim\s*-->", " ", block_to_check, flags=re.DOTALL)
            stripped = re.sub(r"<!--.*?-->", " ", stripped, flags=re.DOTALL)

            digits = re.findall(r"\d+", stripped)
            if digits:
                line_offset = content[: match.start()].count("\n") + 1
                errors.append(
                    f"Empirical block lint in {doc_name} (around line {line_offset}): "
                    f"found unmanaged digits outside claim marker in empirical block: {digits[:5]}"
                )

    return len(errors) == 0, errors


def sync_document(
    doc_path: Path,
    registry_path: Path,
    write: bool = False,
    repo_root: Path = REPO_ROOT,
    validated_json_cache: dict[Path, dict] | None = None,
) -> tuple[bool, list[str]]:
    """
    Verify or update claim markers in doc_path against registry_path.
    Returns (success, list_of_errors_or_messages).
    """
    messages: list[str] = []
    errors: list[str] = []

    if not registry_path.is_file():
        return False, [f"Registry not found: {registry_path}"]
    if not doc_path.is_file():
        return False, [f"Target doc not found: {doc_path}"]

    raw_reg = registry_path.read_text(encoding="utf-8")
    if _has_personal_host_path_text(raw_reg):
        errors.append(f"Registry {registry_path.name} contains leaked personal host path")
    registry = _loads_strict_json(raw_reg)

    claims = registry.get("claims", {})
    resolved_values: dict[str, str] = {}

    for claim_id, claim_cfg in claims.items():
        try:
            val = resolve_claim_value(
                claim_id,
                claim_cfg,
                repo_root=repo_root,
                validated_json_cache=validated_json_cache,
            )
            resolved_values[claim_id] = val
        except Exception as e:
            errors.append(f"Failed to resolve claim '{claim_id}': {e}")

    content = doc_path.read_text(encoding="utf-8")
    claim_pat = re.compile(r"<!--\s*claim:([a-zA-Z0-9_\-]+)\s*-->(.*?)<!--\s*/claim\s*-->", re.DOTALL)

    found_claims: set[str] = set()
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

    latest_text = doc_path.read_text(encoding="utf-8")
    marker_ok, marker_errors = lint_claim_markers_and_regions(
        latest_text, doc_path.name, registry_path=registry_path
    )
    if not marker_ok:
        errors.extend(marker_errors)

    word_ok, word_errors = lint_prose_and_words(
        latest_text, doc_path.name, registry_path=registry_path, repo_root=repo_root
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

    if args.doc is None:
        reg_ok, reg_errs = validate_claims_registry(args.registry, repo_root=REPO_ROOT)
        if not reg_ok:
            overall_success = False
            for err in reg_errs:
                print(f"[FAIL] {err}", file=sys.stderr)
        else:
            print("[PASS] Claims registry schema, non-orphan used_in mappings & claim_term evidence verified.")

    # 1. Independent metrics recomputation
    validated_json_cache: dict[Path, dict] = {}
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

    if do_write and not overall_success:
        print("\nAborting --write because registry, metrics, or script AST checks failed.", file=sys.stderr)
        sys.exit(1)

    # 3. Document sync & linter
    if args.doc is not None:
        target_docs = [args.doc]
    else:
        target_docs = [DEFAULT_ROADMAP_PATH]
        if args.registry.is_file():
            try:
                reg_data = _loads_strict_json(args.registry.read_text(encoding="utf-8"))
                for c_cfg in reg_data.get("claims", {}).values():
                    for rel_p in c_cfg.get("used_in", []):
                        candidate = REPO_ROOT / rel_p
                        if candidate not in target_docs:
                            target_docs.append(candidate)
            except Exception as exc:
                overall_success = False
                print(f"[FAIL] Failed to parse {args.registry}: {exc}", file=sys.stderr)

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
