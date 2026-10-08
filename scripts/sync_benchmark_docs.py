#!/usr/bin/env python3
"""
sync_benchmark_docs.py

Empirical Documentation Synchronization Utility.
Extracts empirical metrics from docs/empirical_retrieval_benchmark.json and
synchronizes the benchmark table in docs/ROADMAP.md between explicit HTML anchors:
<!-- BEGIN_RETRIEVAL_BENCHMARK_TABLE -->
<!-- END_RETRIEVAL_BENCHMARK_TABLE -->

Supports:
  --check : Validate that docs/ROADMAP.md matches the ground-truth JSON (exit 0 on match, 1 on drift)
  --write : Deterministically regenerate and write the table into docs/ROADMAP.md
"""

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCHMARK_JSON = REPO_ROOT / "docs" / "empirical_retrieval_benchmark.json"
DEFAULT_ROADMAP_MD = REPO_ROOT / "docs" / "ROADMAP.md"

BEGIN_DELIMITER = "<!-- BEGIN_RETRIEVAL_BENCHMARK_TABLE -->"
END_DELIMITER = "<!-- END_RETRIEVAL_BENCHMARK_TABLE -->"

METHOD_CONFIGS = [
    {
        "key": "bm25",
        "label": "**BM25 (Okapi Lexical Baseline)**",
        "bold_metrics": True,
        "role": "**Production Core Baseline**",
    },
    {
        "key": "regex",
        "label": "**Deterministic Regex Rules**",
        "bold_metrics": False,
        "role": "Exact Threshold Precision ($6,725, 14-day)",
    },
    {
        "key": "bitnet_embedding",
        "label": "**BitNet 270M Dense Embeddings**",
        "bold_metrics": False,
        "role": "Experimental Prototype (0% on Short Chunks)",
    },
    {
        "key": "hybrid",
        "label": "**Hybrid Pipeline (Regex + BM25 + BitNet)**",
        "bold_metrics": True,
        "role": "**Experimental Prototype (+2 Query Gain)**",
    },
]


def format_ci(ci: list) -> str:
    """Format 95% Wilson confidence interval [low, high] as percentage string."""
    return f"[{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]"


def generate_markdown_table(benchmark_data: dict) -> str:
    """Generate Markdown benchmark table strictly derived from ground-truth JSON."""
    summary = benchmark_data["full_corpus_benchmark"]["summary"]
    queries_count = benchmark_data["full_corpus_benchmark"]["queries_count"]

    header = [
        "| Retrieval Method | Top-1 Recall (Full) | 95% CI (Top-1) | Top-3 Recall (Full) | 95% CI (Top-3) | MRR | Role in Architecture |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    rows = []
    for cfg in METHOD_CONFIGS:
        k = cfg["key"]
        m = summary[k]
        bold = cfg["bold_metrics"]

        # Top-1
        top1_pct = f"{m['top1_recall_pct']:.1f}%"
        top1_hits = f"({m['top1_hits']}/{queries_count})"
        top1_str = f"**{top1_pct}** {top1_hits}" if bold else f"{top1_pct} {top1_hits}"
        top1_ci = format_ci(m["top1_ci_95"])

        # Top-3
        top3_pct = f"{m['top3_recall_pct']:.1f}%"
        top3_hits = f"({m['top3_hits']}/{queries_count})"
        top3_str = f"**{top3_pct}** {top3_hits}" if bold else f"{top3_pct} {top3_hits}"
        top3_ci = format_ci(m["top3_ci_95"])

        # MRR
        mrr_val = f"{m['mrr']:.4f}"
        mrr_str = f"**{mrr_val}**" if bold else mrr_val

        rows.append(
            f"| {cfg['label']} | {top1_str} | {top1_ci} | {top3_str} | {top3_ci} | {mrr_str} | {cfg['role']} |"
        )

    return "\n".join(header + rows)


def check_roadmap_sync(benchmark_path: Path, roadmap_path: Path) -> tuple:
    """
    Check if the table in docs/ROADMAP.md matches the table generated from benchmark JSON.
    Returns (matches: bool, message: str).
    """
    if not benchmark_path.is_file():
        return False, f"Missing benchmark JSON: {benchmark_path}"
    if not roadmap_path.is_file():
        return False, f"Missing roadmap Markdown: {roadmap_path}"

    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    expected_table = generate_markdown_table(bench_data).strip()

    with open(roadmap_path, "r", encoding="utf-8") as f:
        roadmap_content = f.read()

    pattern = re.escape(BEGIN_DELIMITER) + r"(.*?)" + re.escape(END_DELIMITER)
    match = re.search(pattern, roadmap_content, re.DOTALL)
    if not match:
        return False, f"Delimiters '{BEGIN_DELIMITER}' and '{END_DELIMITER}' not found in {roadmap_path}"

    actual_table = match.group(1).strip()
    if actual_table != expected_table:
        return False, (
            f"Documentation table mismatch in {roadmap_path}!\n"
            f"--- EXPECTED (from {benchmark_path.name}) ---\n{expected_table}\n"
            f"--- ACTUAL (in {roadmap_path.name}) ---\n{actual_table}"
        )

    return True, f"OK: {roadmap_path.name} retrieval benchmark table strictly matches {benchmark_path.name}"


def sync_roadmap(benchmark_path: Path, roadmap_path: Path) -> bool:
    """Regenerate and write the benchmark table into docs/ROADMAP.md between delimiters."""
    if not benchmark_path.is_file():
        print(f"Error: Missing benchmark JSON at {benchmark_path}", file=sys.stderr)
        return False
    if not roadmap_path.is_file():
        print(f"Error: Missing roadmap file at {roadmap_path}", file=sys.stderr)
        return False

    with open(benchmark_path, "r", encoding="utf-8") as f:
        bench_data = json.load(f)

    expected_table = generate_markdown_table(bench_data)

    with open(roadmap_path, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.escape(BEGIN_DELIMITER) + r"(.*?)" + re.escape(END_DELIMITER)
    replacement = f"{BEGIN_DELIMITER}\n{expected_table}\n{END_DELIMITER}"

    if re.search(pattern, content, re.DOTALL):
        updated_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    else:
        # Fallback: if delimiters not present, find existing table and wrap it
        table_pattern = r"(\| Retrieval Method \| Top-1 Recall.*?\n\| \*\*Hybrid Pipeline.*?\n)"
        if re.search(table_pattern, content):
            updated_content = re.sub(table_pattern, f"{replacement}\n", content)
        else:
            print(f"Error: Could not locate table or delimiters in {roadmap_path}", file=sys.stderr)
            return False

    with open(roadmap_path, "w", encoding="utf-8") as f:
        f.write(updated_content)

    print(f"Successfully synchronized benchmark table in {roadmap_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="Synchronize empirical benchmark tables into documentation.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Check documentation table against ground-truth JSON and exit 1 on drift",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate and overwrite the benchmark table in documentation",
    )
    parser.add_argument(
        "--benchmark-json",
        type=Path,
        default=DEFAULT_BENCHMARK_JSON,
        help="Path to empirical_retrieval_benchmark.json",
    )
    parser.add_argument(
        "--roadmap-md",
        type=Path,
        default=DEFAULT_ROADMAP_MD,
        help="Path to docs/ROADMAP.md",
    )

    args = parser.parse_args()

    if args.check:
        matches, msg = check_roadmap_sync(args.benchmark_json, args.roadmap_md)
        if matches:
            print(msg)
            sys.exit(0)
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
            sys.exit(1)
    else:
        # Default behavior or explicit --write
        success = sync_roadmap(args.benchmark_json, args.roadmap_md)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
