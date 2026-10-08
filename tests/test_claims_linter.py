"""
Claims Linter Test Suite.

Asserts that all quantitative claims, token bounds, benchmark metrics,
confidence intervals, case captions, and legal statuses across JSON artifacts
and engine configurations match empirical runs with zero hardcoded drift
and zero personal user paths.
"""

import json
import os
import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"

import sys
sys.path.insert(0, str(REPO_ROOT / "src/FtaaSService.Inference"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import bitnet_engine
import sync_benchmark_docs


class TestClaimsLinter(unittest.TestCase):

    def test_engine_constants_and_safe_bounds(self):
        """Verify engine constants enforce verified safe bounds."""
        self.assertEqual(bitnet_engine.MAX_SAFE_GENERATE_TOKENS, 24)
        self.assertLessEqual(bitnet_engine.MAX_SAFE_EMBED_TOKENS, 255)
        self.assertEqual(bitnet_engine.MAX_SAFE_EMBED_TOKENS, 240)

    def test_empirical_isolation_results(self):
        """Verify empirical_long_context_isolation_results.json claims and sanity."""
        path = DOCS_DIR / "empirical_long_context_isolation_results.json"
        self.assertTrue(path.is_file(), f"Missing {path}")

        raw_text = path.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", raw_text, "Found leaked /Users/ path in isolation JSON")

        data = json.loads(raw_text)
        metadata = data["metadata"]
        self.assertEqual(metadata["bitnet_git_commit"], "5fce1685482d7a4e7d3823281b53dfdf28925b78")
        self.assertEqual(metadata["prompt_token_counts"]["long_statute"], 598)
        self.assertEqual(metadata["prompt_token_counts"]["intermediate_statute"], 79)
        self.assertEqual(metadata["prompt_token_counts"]["bisection_statute"], 30)
        self.assertEqual(metadata["prompt_token_counts"]["short_prompt"], 29)

        # RoPE parity
        self.assertTrue(data["rope_configuration"]["rope_parameters_match"])

        # First-token logits bisection in HF
        hf_logits = data["huggingface_pytorch_reference"]["first_token_logits_bisection"]
        self.assertGreater(hf_logits["prompt_29_tokens"]["logit_delta"], 25.0)
        self.assertGreater(hf_logits["prompt_30_tokens"]["logit_delta"], 25.0)

        # BitNet evaluations
        runs = data["bitnet_cpp_evaluations"]
        # Short 29 tokens passes
        self.assertTrue(runs["short_29tok_default_b512"]["factual_correctness"])
        # 30 tokens fails with tile corruption at default
        self.assertFalse(runs["bisection_30tok_default_b512"]["factual_correctness"])
        self.assertTrue(runs["bisection_30tok_default_b512"]["is_corrupted"])
        # Safe bound passes
        self.assertTrue(runs["safe_bound_24tok_default_b512"]["factual_correctness"])

        # Conclusion calibrated wording
        conclusion = data["conclusion"]
        self.assertIn("context-length-dependent", conclusion)
        self.assertIn("clamped to 24 tokens", conclusion)

    def test_verified_statutory_sources(self):
        """Verify verified_statutory_sources.json has exactly 40 sources and proper legal statuses."""
        path = DOCS_DIR / "verified_statutory_sources.json"
        self.assertTrue(path.is_file(), f"Missing {path}")

        raw_text = path.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", raw_text, "Found leaked /Users/ path in verified sources JSON")

        data = json.loads(raw_text)
        sources = data["sources"]
        self.assertEqual(len(sources), 40)

        status_counts = data["metadata"]["status_breakdown"]
        self.assertEqual(status_counts["in_force"], 33)
        self.assertEqual(status_counts["vacated_by_court_order"], 3)
        self.assertEqual(status_counts["enacted_not_in_force"], 4)

        # Check vacatur citations
        vacated = [s for s in sources if s["status"] == "vacated_by_court_order"]
        self.assertEqual(len(vacated), 3)
        for s in vacated:
            self.assertIn("Custom Communications", s["legal_notes"])
            self.assertIn("24-3232", s["legal_notes"])

        # Check UK DMCC enacted not in force
        uk_dmcc = [s for s in sources if s["jurisdiction"] == "UK"]
        self.assertEqual(len(uk_dmcc), 4)
        for s in uk_dmcc:
            self.assertEqual(s["status"], "enacted_not_in_force")

    def test_empirical_retrieval_benchmark(self):
        """Verify empirical_retrieval_benchmark.json matches corpus count, self-retrieval, and baseline metrics."""
        path = DOCS_DIR / "empirical_retrieval_benchmark.json"
        self.assertTrue(path.is_file(), f"Missing {path}")

        raw_text = path.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", raw_text, "Found leaked /Users/ path in retrieval benchmark JSON")

        data = json.loads(raw_text)
        metadata = data["metadata"]
        self.assertEqual(metadata["total_corpus_chunks"], 40)
        self.assertEqual(metadata["queries_evaluated"], 35)

        # Self-retrieval 100%
        self.assertEqual(metadata["self_retrieval_validation"]["passes"], 40)
        self.assertEqual(metadata["self_retrieval_validation"]["accuracy_pct"], 100.0)

        # BM25 Top-1 on full corpus is 82.9% (29 hits)
        full_summary = data["full_corpus_benchmark"]["summary"]
        self.assertEqual(full_summary["bm25"]["top1_hits"], 29)
        self.assertEqual(full_summary["bm25"]["top1_recall_pct"], 82.9)

        # Short chunks: BitNet gets 0% Top-1
        short_summary = data["short_chunks_benchmark"]["summary"]
        self.assertEqual(short_summary["bitnet_embedding"]["top1_hits"], 0)
        self.assertEqual(short_summary["bitnet_embedding"]["top1_recall_pct"], 0.0)

        # Wilson intervals are well-formed
        for method, metrics in full_summary.items():
            ci1 = metrics["top1_ci_95"]
            ci3 = metrics["top3_ci_95"]
            self.assertLessEqual(ci1[0], ci1[1])
            self.assertLessEqual(ci3[0], ci3[1])
            self.assertGreaterEqual(ci1[0], 0.0)
            self.assertLessEqual(ci1[1], 1.0)

    def test_270m_embedder_validation(self):
        """Verify empirical_270m_embedder_validation.json correlation statements."""
        path = DOCS_DIR / "empirical_270m_embedder_validation.json"
        self.assertTrue(path.is_file(), f"Missing {path}")

        raw_text = path.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", raw_text, "Found leaked /Users/ path in embedder validation JSON")

        data = json.loads(raw_text)
        full_val = data["full_legal_clauses_validation"]
        short_val = data["short_clauses_validation"]

        self.assertEqual(full_val["pairwise_spearman_rho"], 0.4318)
        self.assertEqual(short_val["pairwise_spearman_rho"], 0.3705)
        self.assertLess(short_val["pairwise_spearman_rho"], full_val["pairwise_spearman_rho"])

        conclusion = data["metadata"]["conclusion"]
        self.assertNotIn("higher correlation on short sequences", conclusion.lower())
        self.assertIn("lower rank correlation on short sequences", conclusion.lower())

    def test_roadmap_benchmark_table_sync(self):
        """Verify docs/ROADMAP.md benchmark table is strictly synced with empirical_retrieval_benchmark.json."""
        import sync_benchmark_docs

        bench_path = DOCS_DIR / "empirical_retrieval_benchmark.json"
        roadmap_path = DOCS_DIR / "ROADMAP.md"

        matches, msg = sync_benchmark_docs.check_roadmap_sync(bench_path, roadmap_path)
        self.assertTrue(matches, f"ROADMAP.md benchmark table is out of sync with JSON ground truth:\n{msg}")

        # Also directly inspect the table markdown in ROADMAP.md
        roadmap_text = roadmap_path.read_text(encoding="utf-8")
        self.assertIn("<!-- BEGIN_RETRIEVAL_BENCHMARK_TABLE -->", roadmap_text)
        self.assertIn("<!-- END_RETRIEVAL_BENCHMARK_TABLE -->", roadmap_text)

        # Assert key empirical cells appear verbatim in the table
        self.assertIn("**82.9%** (29/35)", roadmap_text)  # BM25 Top-1
        self.assertIn("[67.3%, 91.9%]", roadmap_text)      # BM25 Top-1 CI
        self.assertIn("**97.1%** (34/35)", roadmap_text)  # BM25 Top-3
        self.assertIn("[85.5%, 99.5%]", roadmap_text)      # BM25 Top-3 CI
        self.assertIn("**0.8952**", roadmap_text)         # BM25 MRR

        self.assertIn("14.3% (5/35)", roadmap_text)       # Regex Top-1
        self.assertIn("[6.3%, 29.4%]", roadmap_text)       # Regex Top-1 CI
        self.assertIn("17.1% (6/35)", roadmap_text)       # Regex Top-3
        self.assertIn("[8.1%, 32.7%]", roadmap_text)       # Regex Top-3 CI
        self.assertIn("0.2253", roadmap_text)             # Regex MRR

        self.assertIn("20.0% (7/35)", roadmap_text)       # BitNet Top-1
        self.assertIn("[10.0%, 35.9%]", roadmap_text)      # BitNet Top-1 CI
        self.assertIn("34.3% (12/35)", roadmap_text)      # BitNet Top-3
        self.assertIn("[20.8%, 50.8%]", roadmap_text)      # BitNet Top-3 CI
        self.assertIn("0.3378", roadmap_text)             # BitNet MRR

        self.assertIn("**88.6%** (31/35)", roadmap_text)  # Hybrid Top-1
        self.assertIn("[74.1%, 95.5%]", roadmap_text)      # Hybrid Top-1 CI
        self.assertIn("**0.9333**", roadmap_text)         # Hybrid MRR

    def test_roadmap_constants_and_policy_invariants(self):
        """Verify code constants (MAX_SAFE_GENERATE_TOKENS) match ROADMAP.md and outdated ceilings are absent."""
        roadmap_path = DOCS_DIR / "ROADMAP.md"
        roadmap_text = roadmap_path.read_text(encoding="utf-8")

        # Assert no personal /Users/ paths exist in ROADMAP.md
        self.assertNotIn("/Users/", roadmap_text, "Found leaked personal /Users/ path in docs/ROADMAP.md")

        # Any explicit definition or setting of MAX_SAFE_GENERATE_TOKENS in ROADMAP.md must equal the engine constant
        matches = re.findall(r"MAX_SAFE_GENERATE_TOKENS\s*=\s*(\d+)", roadmap_text)
        self.assertGreater(len(matches), 0, "MAX_SAFE_GENERATE_TOKENS not referenced in docs/ROADMAP.md")
        for val_str in matches:
            self.assertEqual(
                int(val_str),
                bitnet_engine.MAX_SAFE_GENERATE_TOKENS,
                f"ROADMAP.md cites MAX_SAFE_GENERATE_TOKENS = {val_str}, but engine constant is {bitnet_engine.MAX_SAFE_GENERATE_TOKENS}"
            )

        # Assert outdated ceilings are not present as active MAX_SAFE_GENERATE_TOKENS values
        self.assertNotIn("MAX_SAFE_GENERATE_TOKENS = 150", roadmap_text)
        self.assertNotIn("MAX_SAFE_GENERATE_TOKENS = 31", roadmap_text)

        # Assert embedder ceiling is documented
        self.assertIn(f"{bitnet_engine.MAX_SAFE_EMBED_TOKENS}-token embedding ceiling", roadmap_text)


if __name__ == "__main__":
    unittest.main()
