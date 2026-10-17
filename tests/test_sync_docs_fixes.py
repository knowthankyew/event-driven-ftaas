"""Regression tests for sync_docs.py defects found in review of 8548daa.

Each test fails on 8548daa's sync_docs.py and passes with sync_docs_fixes.patch.
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
import sync_docs  # noqa: E402


def _write(tmp: Path, name: str, text: str) -> Path:
    p = tmp / name
    p.write_text(text, encoding="utf-8")
    return p


class TestExtractPythonConstant(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.tmp = Path(self._td.name)

    def tearDown(self):
        self._td.cleanup()

    def test_min_wrapped_getenv_returns_default_not_clamp(self):
        # Mirrors bitnet_engine.py: BFS over the AST hits the literal 255 before "240".
        f = _write(self.tmp, "m.py",
                   "import os\nX = min(int(os.getenv('X', '240')), 255)\n")
        self.assertEqual(sync_docs.extract_python_constant(f, "X"), 240)

    def test_plain_int_getenv(self):
        f = _write(self.tmp, "m.py",
                   "import os\nX = int(os.getenv('X', '16'))\n")
        self.assertEqual(sync_docs.extract_python_constant(f, "X"), 16)

    def test_unsupported_expression_fails_closed(self):
        # Old code returned 7 (first int literal found anywhere in the call).
        f = _write(self.tmp, "m.py", "X = compute(7)\n")
        with self.assertRaises(ValueError):
            sync_docs.extract_python_constant(f, "X")

    def test_regex_fallback_removed(self):
        # Old code fell back to "first digit run on the line" for non-Assign shapes.
        f = _write(self.tmp, "m.py", "X: int = compute(9)\n")
        with self.assertRaises((ValueError, FileNotFoundError)):
            sync_docs.extract_python_constant(f, "X")


class TestFormatRules(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.root = Path(self._td.name)
        (self.root / "a.json").write_text(
            json.dumps({"s": {"ci": [0.673, 0.919], "pct": 82.857}}), encoding="utf-8")

    def tearDown(self):
        self._td.cleanup()

    def _resolve(self, cfg):
        cfg = {"source_artifact": "a.json", **cfg}
        return sync_docs.resolve_claim_value("c", cfg, repo_root=self.root)

    def test_named_ci_rule_renders(self):
        self.assertEqual(self._resolve({"json_path": "s.ci", "format_rule": "ci_pct"}),
                         "[67.3%, 91.9%]")

    def test_list_value_with_non_named_rule_fails_closed(self):
        # Old code silently ignored format_rule for 2-element lists.
        with self.assertRaises(ValueError):
            self._resolve({"json_path": "s.ci", "format_rule": "{value}"})

    def test_expression_in_format_string_gives_clear_error(self):
        with self.assertRaises(ValueError) as ctx:
            self._resolve({"json_path": "s.pct", "format_rule": "{value*100:.1f}"})
        self.assertIn("format_rule", str(ctx.exception))

    def test_scalar_rule_unchanged(self):
        self.assertEqual(self._resolve({"json_path": "s.pct", "format_rule": "{value:.1f}%"}),
                         "82.9%")

    def test_registry_has_no_expression_style_format_rules(self):
        reg = json.loads((REPO_ROOT / "docs" / "claims_registry.json").read_text(encoding="utf-8"))
        bad = [k for k, v in reg["claims"].items() if "[0]" in v.get("format_rule", "")]
        self.assertEqual(bad, [], f"registry claims with dead expression-style format_rule: {bad}")


class TestMainOutput(unittest.TestCase):
    def _run_main(self, ast_result):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(sys, "argv", ["sync_docs.py", "--check"]), \
             mock.patch.object(sync_docs, "lint_script_conclusions", return_value=ast_result), \
             mock.patch.object(sync_docs, "sync_document", return_value=(True, [])), \
             contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            with self.assertRaises(SystemExit) as ctx:
                sync_docs.main()
        return ctx.exception.code, out.getvalue(), err.getvalue()

    def test_no_pass_banner_when_ast_lint_fails(self):
        code, out, err = self._run_main((False, ["boom"]))
        self.assertEqual(code, 1)
        self.assertIn("[FAIL] boom", err)
        self.assertNotIn("[PASS] Script conclusion", out)

    def test_pass_banner_when_ast_lint_passes(self):
        code, out, _ = self._run_main((True, []))
        self.assertEqual(code, 0)
        self.assertIn("[PASS] Script conclusion", out)

    def test_check_and_write_are_mutually_exclusive(self):
        with mock.patch.object(sys, "argv", ["sync_docs.py", "--check", "--write"]), \
             contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as ctx:
                sync_docs.main()
        self.assertEqual(ctx.exception.code, 2)


class TestRecomputeHardening(unittest.TestCase):
    """Fixtures exercise independent_recompute_metrics; see sync_docs_recompute_hardening.patch."""

    @staticmethod
    def _artifact(n=35, hits=29, **over):
        qs = [{"id": f"q{i}", "rankings": {"bm25": {
            "top1_hit": i < hits, "top3_hit": i < hits, "rr": 1.0 if i < hits else 0.0}}} for i in range(n)]
        ci1 = sync_docs.independent_wilson_score_interval(hits, n)
        summ = {"bm25": {"top1_hits": hits, "top3_hits": hits, "mrr": hits / n,
                         "top1_recall_pct": round(100 * hits / n, 1), "top3_recall_pct": round(100 * hits / n, 1),
                         "top1_ci_95": list(ci1), "top3_ci_95": list(ci1)}}
        art = {"full_corpus_benchmark": {"queries_count": n, "query_details": qs, "summary": summ}}
        art["full_corpus_benchmark"].update(over)
        return art

    def _check(self, art):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "b.json"
            p.write_text(json.dumps(art), encoding="utf-8")
            return sync_docs.independent_recompute_metrics(p)

    def test_consistent_artifact_passes(self):
        ok, errs = self._check(self._artifact())
        self.assertTrue(ok, errs)

    def test_missing_query_details_fails_closed(self):
        ok, _ = self._check({"full_corpus_benchmark": {"summary": {}}})
        self.assertFalse(ok)

    def test_wrong_top3_ci_detected(self):
        art = self._artifact()
        art["full_corpus_benchmark"]["summary"]["bm25"]["top3_ci_95"] = [0.5, 0.6]
        ok, errs = self._check(art)
        self.assertFalse(ok)
        self.assertTrue(any("top3_ci_95" in e for e in errs), errs)

    def test_wrong_rendered_percentage_detected(self):
        art = self._artifact()
        art["full_corpus_benchmark"]["summary"]["bm25"]["top1_recall_pct"] = 88.6
        ok, errs = self._check(art)
        self.assertFalse(ok)
        self.assertTrue(any("top1_recall_pct" in e for e in errs), errs)

    def test_queries_count_mismatch_detected(self):
        ok, errs = self._check(self._artifact(queries_count=40))
        self.assertFalse(ok)
        self.assertTrue(any("queries_count" in e for e in errs), errs)

    def test_hit_flag_disagreeing_with_rr_detected(self):
        art = self._artifact()
        art["full_corpus_benchmark"]["query_details"][0]["rankings"]["bm25"]["rr"] = 0.25
        ok, errs = self._check(art)
        self.assertFalse(ok)

    def test_summary_method_without_records_detected(self):
        art = self._artifact()
        art["full_corpus_benchmark"]["summary"]["ghost"] = dict(art["full_corpus_benchmark"]["summary"]["bm25"])
        ok, errs = self._check(art)
        self.assertFalse(ok)
        self.assertTrue(any("ghost" in e for e in errs), errs)


if __name__ == "__main__":
    unittest.main()
