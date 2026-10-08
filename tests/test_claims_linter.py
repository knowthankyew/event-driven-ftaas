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
        self.assertEqual(bitnet_engine.MAX_SAFE_GENERATE_TOKENS, 16)
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
            self.assertIn("24-3137", s["legal_notes"])

        # Check UK DMCC enacted not in force
        uk_dmcc = [s for s in sources if s["jurisdiction"] == "UK"]
        self.assertEqual(len(uk_dmcc), 4)
        for s in uk_dmcc:
            self.assertEqual(s["status"], "enacted_not_in_force")

        # Verify state machine invariant: state == verified requires 2xx and both hashes
        for s in sources:
            self.assertIn(s["state"], ("verified", "unreachable", "mismatch", "unverified"))
            if s["state"] == "verified":
                self.assertIsNotNone(s.get("http_status"))
                self.assertGreaterEqual(s["http_status"], 200)
                self.assertLess(s["http_status"], 300)
                self.assertTrue(s.get("content_sha256"))
                self.assertTrue(s.get("extracted_text_sha256"))
                self.assertIsNotNone(s.get("verified_on"))

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

        # In-force answerable subset benchmark
        in_force = data["in_force_answerable_benchmark"]["summary"]
        self.assertEqual(in_force["bm25"]["top1_hits"], 23)
        self.assertEqual(in_force["bm25"]["top1_recall_pct"], 82.1)
        self.assertEqual(in_force["hybrid"]["top1_hits"], 24)
        self.assertEqual(in_force["hybrid"]["top1_recall_pct"], 85.7)

        # Vacated rules validation
        vacated_val = data["vacated_rule_validation"]
        self.assertEqual(vacated_val["vacated_queries_count"], 7)
        vacated_hits = [c for c in vacated_val["checks"] if c["top1_id"].startswith("ftc_negative_option_")]
        self.assertTrue(len(vacated_hits) > 0)
        for c in vacated_hits:
            self.assertTrue(c["vacated_banner_present"])

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

    def test_roadmap_constants_and_policy_invariants(self):
        """Verify code constants (MAX_SAFE_GENERATE_TOKENS) match ROADMAP.md."""
        roadmap_path = DOCS_DIR / "ROADMAP.md"
        roadmap_text = roadmap_path.read_text(encoding="utf-8")

        # Assert no personal /Users/ paths exist in ROADMAP.md
        self.assertNotIn("/Users/", roadmap_text, "Found leaked personal /Users/ path in docs/ROADMAP.md")

        # Any explicit definition or setting of MAX_SAFE_GENERATE_TOKENS in ROADMAP.md must equal the engine constant
        matches = re.findall(r"MAX_SAFE_GENERATE_TOKENS\s*=\s*(?:<!--[^>]+-->\s*)?(\d+)", roadmap_text)
        self.assertGreater(len(matches), 0, "MAX_SAFE_GENERATE_TOKENS not referenced in docs/ROADMAP.md")
        for val_str in matches:
            self.assertEqual(
                int(val_str),
                bitnet_engine.MAX_SAFE_GENERATE_TOKENS,
                f"ROADMAP.md cites MAX_SAFE_GENERATE_TOKENS = {val_str}, but engine constant is {bitnet_engine.MAX_SAFE_GENERATE_TOKENS}"
            )

        # Assert embedder ceiling is documented
        self.assertIn(f"{bitnet_engine.MAX_SAFE_EMBED_TOKENS}-token embedding ceiling", roadmap_text)

    def test_negative_control_mutated_claim_fails_linter(self):
        """Mutate a claim value in documentation and assert linter fails closed."""
        import tempfile
        from sync_docs import sync_document
        registry_path = DOCS_DIR / "claims_registry.json"
        with tempfile.NamedTemporaryFile("w+", suffix=".md", delete=True) as tf:
            tf.write("Benchmark recall: <!-- claim:bm25_top1_pct -->99.9%<!-- /claim -->.\n")
            tf.flush()
            ok, errors = sync_document(Path(tf.name), registry_path, write=False, repo_root=REPO_ROOT)
            self.assertFalse(ok, "Expected linter to fail on mutated claim")
            self.assertTrue(any("bm25_top1_pct" in e for e in errors), f"Expected claim error, got: {errors}")

    def test_negative_control_mutated_json_fails_validation(self):
        """Mutate verification state in JSON and assert state machine check fails."""
        from verify_statutory_sources import verify_all_sources
        mock_bad_source = {
            "id": "mock_bad",
            "statute": "Mock Statute",
            "jurisdiction": "US_FEDERAL",
            "citation": "Mock 123",
            "topic": "Mock",
            "chunk_type": "paraphrase",
            "effective_from": "2025-01-01",
            "source_url": "http://example.com",
            "source_facts": ["nonexistent_fact_12345"],
            "target_text": "Sample text",
            "status": "in_force"
        }
        def stub_fetcher(url):
            return (200, b"Different text", url, {"content-type": "text/plain"})

        res = verify_all_sources(sources=[mock_bad_source], fetcher=stub_fetcher, snapshot_dir=None)
        self.assertEqual(res["sources"][0]["state"], "mismatch")
        self.assertNotEqual(res["sources"][0]["state"], "verified")

    def test_wp1_stub_fetcher_state_machine(self):
        """Verify verify_statutory_sources accurately yields unreachable, mismatch, and verified states."""
        from verify_statutory_sources import verify_all_sources

        sample_source = {
            "id": "test_clause",
            "statute": "12 CFR Part 229",
            "jurisdiction": "US_FEDERAL",
            "citation": "12 CFR 229.10",
            "topic": "Test clause",
            "chunk_type": "paraphrase",
            "effective_from": "2025-07-01",
            "source_url": "https://example.com/test",
            "source_facts": ["$275", "next-day"],
            "target_text": "Next-day check availability is $275.",
            "status": "in_force"
        }

        # Case 1: 404 unreachable
        def fetcher_404(url):
            return (404, b"Not Found", url, {})
        r1 = verify_all_sources([sample_source], fetcher=fetcher_404, snapshot_dir=None)
        self.assertEqual(r1["sources"][0]["state"], "unreachable")
        self.assertIsNone(r1["sources"][0]["verified_on"])

        # Case 2: 200 with different text -> mismatch
        def fetcher_diff(url):
            return (200, b"Different statutory rules", url, {"content-type": "text/plain"})
        r2 = verify_all_sources([sample_source], fetcher=fetcher_diff, snapshot_dir=None)
        self.assertEqual(r2["sources"][0]["state"], "mismatch")
        self.assertIsNone(r2["sources"][0]["verified_on"])

        # Case 3: 200 with matching text containing all source facts -> verified
        def fetcher_match(url):
            t = b"Next-day check availability minimum is $275."
            return (200, t, url, {"content-type": "text/plain"})
        r3 = verify_all_sources([sample_source], fetcher=fetcher_match, snapshot_dir=None)
        self.assertEqual(r3["sources"][0]["state"], "verified")
        self.assertIsNotNone(r3["sources"][0]["verified_on"])
        self.assertTrue(r3["sources"][0]["content_sha256"])
        self.assertTrue(r3["sources"][0]["extracted_text_sha256"])

    def test_empirical_context_boundary_bisection(self):
        """Verify empirical_context_boundary.json integrity, recompute cap, and verify engine constant."""
        path = DOCS_DIR / "empirical_context_boundary.json"
        if not path.is_file():
            self.skipTest("empirical_context_boundary.json not yet generated")

        raw_text = path.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", raw_text, "Found leaked /Users/ path in bisection JSON")

        data = json.loads(raw_text)
        metadata = data["metadata"]
        self.assertGreaterEqual(metadata["clauses_evaluated"], 12)
        self.assertGreater(metadata["total_runs"], 100)

        # Recompute pass rates and cap from per-run records
        runs = data["runs"]
        clause_results = {}
        for r in runs:
            cid = r["clause_id"]
            cfg = r["config"]
            tlen = r["target_length"]
            if cfg == "default_b512_t4":
                clause_results.setdefault(cid, {})[tlen] = (r["classification"] == "correct")

        clauses = list(clause_results.keys())
        target_lengths = sorted(next(iter(clause_results.values())).keys())
        all_passing = [
            tlen for tlen in target_lengths
            if all(clause_results[c].get(tlen, False) for c in clauses)
        ]
        recomputed_highest = max(all_passing) if all_passing else 16
        stated_margin = metadata["stated_margin"]
        self.assertGreaterEqual(stated_margin, 8)
        recomputed_cap = max(16, recomputed_highest - stated_margin)

        self.assertEqual(metadata["highest_all_passing_length"], recomputed_highest)
        self.assertEqual(metadata["recommended_safe_generate_cap"], recomputed_cap)
        self.assertLessEqual(bitnet_engine.MAX_SAFE_GENERATE_TOKENS, recomputed_cap)

    def test_empirical_speed_memory(self):
        """Verify empirical_speed_memory.json integrity, model SHAs, and measurements."""
        path = DOCS_DIR / "empirical_speed_memory.json"
        if not path.is_file():
            self.skipTest("empirical_speed_memory.json not yet generated")

        raw_text = path.read_text(encoding="utf-8")
        self.assertNotIn("/Users/", raw_text, "Found leaked /Users/ path in speed/memory JSON")

        data = json.loads(raw_text)
        metadata = data["metadata"]
        summary = data["summary"]

        # Check model SHAs: ensure not measured against wrong model file
        gen_sha = metadata["generator_model_sha256"]
        emb_sha = metadata["embedder_model_sha256"]
        self.assertTrue(gen_sha and gen_sha != "missing")
        self.assertTrue(emb_sha and emb_sha != "missing")
        self.assertNotEqual(gen_sha, emb_sha)

        # Check measurements
        self.assertGreater(summary["bitnet_2b_peak_rss_mb"], 0.0)
        self.assertGreater(summary["bitnet_270m_peak_rss_mb"], 0.0)
        self.assertEqual(summary["gemma2_fp16_reference_rss_mb"], 5400.0)
        self.assertEqual(data["baseline_comparison"]["gemma_2_2b_fp16"]["baseline_type"], "published_reference")

        # Check throughput runs exist
        tp_runs = data["throughput_measurements"]
        self.assertGreaterEqual(len(tp_runs), 12)

    def test_no_users_path_in_any_docs_or_artifacts(self):
        """Ensure zero personal /Users/ host directory paths exist in docs or artifacts."""
        for path in DOCS_DIR.glob("**/*"):
            if path.is_file() and path.suffix in (".md", ".json", ".jsonl"):
                content = path.read_text(encoding="utf-8", errors="replace")
                self.assertNotIn(
                    "/Users/",
                    content,
                    f"Found leaked host directory path in {path.relative_to(REPO_ROOT)}"
                )


if __name__ == "__main__":
    unittest.main()
