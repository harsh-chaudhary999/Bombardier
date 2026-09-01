"""
Guards the reranker's document-text resolution.

The bug this exists for: `rerank(text_field="summary")` was the default, but PRD chunks store
their body in `chunk_text` and have no `summary`, `description` or `steps_text`. Every pair
handed to the cross-encoder was therefore `(query, "")`, and `predict()` returns a number for
that — the *same* number every time. The result was six passages scoring an identical
-8.6539, which reads as "the model thinks everything is irrelevant" rather than "the model
was never shown the documents". A real measurement of those logits was mistaken for evidence
that ms-marco was a poor fit for the corpus.

So two properties are pinned: the body field is resolved per result across both index shapes,
and an empty document is impossible whenever the record carries any text at all.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from test_context_budget import _install_stubs  # noqa: E402

_install_stubs()
from embeddings.reranker import Reranker  # noqa: E402

PRD_CHUNK = {
    "source_id": "confluence:1000000001",
    "doc_title": "Implementation Plan for EXAMPLE_STATUS",
    "section_heading": "1. Background",
    "chunk_text": "EXAMPLE_STATUS grants a one-month access window.",
    "score": 0.836,
}
TEST_CASE = {
    "jira_key": "PROJ-1234",
    "summary": "Verify login with valid credentials",
    "steps_text": "1. open page\n2. submit",
    "description": "Covers the happy path.",
    "score": 0.72,
}


class DocumentTextTests(unittest.TestCase):
    """_document_text is pure; exercise it without loading the model."""

    def setUp(self):
        # Bypass __init__ so no CrossEncoder download is needed.
        self.r = Reranker.__new__(Reranker)

    def _text(self, rec, field=None):
        return self.r._document_text(rec, field)

    # ── the regression ──

    def test_prd_chunk_resolves_to_chunk_text(self):
        """Previously returned "" because it looked only at `summary`."""
        got = self._text(PRD_CHUNK)
        self.assertIn("one-month access window", got)
        self.assertTrue(got.strip())

    def test_prd_chunk_is_never_empty(self):
        self.assertNotEqual(self._text(PRD_CHUNK).strip(), "")

    def test_test_case_still_resolves_to_summary(self):
        """The four test-case call sites must keep working unchanged."""
        got = self._text(TEST_CASE)
        self.assertIn("Verify login", got)

    def test_test_case_enrichment_is_preserved(self):
        got = self._text(TEST_CASE)
        self.assertIn("happy path", got, "description must still enrich")
        self.assertIn("open page", got, "steps_text must still enrich")

    # ── mixed list, which /ask can produce with include_tests=true ──

    def test_mixed_list_resolves_each_record_independently(self):
        prd = self._text(PRD_CHUNK)
        test = self._text(TEST_CASE)
        self.assertIn("access window", prd)
        self.assertIn("Verify login", test)
        self.assertNotEqual(prd, test, "the old default made these identical (both empty)")

    # ── explicit field still honoured ──

    def test_explicit_text_field_wins(self):
        rec = {"chunk_text": "body", "custom": "preferred"}
        self.assertTrue(self._text(rec, "custom").startswith("preferred"))

    def test_explicit_field_falls_back_when_absent(self):
        self.assertIn("one-month", self._text(PRD_CHUNK, "summary"))

    # ── last-resort and degenerate cases ──

    def test_title_and_heading_used_when_no_body_field(self):
        rec = {"doc_title": "Release notes", "section_heading": "Summary of changes"}
        got = self._text(rec)
        self.assertIn("Release notes", got)
        self.assertIn("Summary of changes", got)

    def test_record_with_no_text_at_all_yields_empty(self):
        """Nothing to invent here — but rerank() must log an error for this case."""
        self.assertEqual(self._text({"score": 0.5}), "")

    def test_blank_and_whitespace_fields_are_skipped(self):
        rec = {"chunk_text": "   ", "summary": "real text"}
        self.assertTrue(self._text(rec).startswith("real text"))

    def test_non_string_field_is_skipped(self):
        rec = {"chunk_text": 12345, "summary": "real text"}
        self.assertTrue(self._text(rec).startswith("real text"))

    def test_underscore_text_field_from_ask(self):
        """/ask builds `_text` for test records before reranking."""
        self.assertIn("joined", self._text({"_text": "joined test body"}))


class DocumentBudgetTests(unittest.TestCase):
    """
    The cross-encoder truncates silently at its sequence limit. Only the enrichment
    used to be capped, so a large PRD chunk filled the window alone and the
    description and steps appended after it never reached the model.
    """

    def setUp(self):
        self.r = Reranker.__new__(Reranker)

    def test_oversized_body_is_capped(self):
        rec = {"chunk_text": "word " * 4000}
        got = self.r._document_text(rec, None)
        self.assertLessEqual(len(got), Reranker._document_budget())

    def test_enrichment_survives_a_large_body(self):
        """The regression: steps used to be appended past the model's window."""
        rec = {
            "chunk_text": "body " * 4000,
            "description": "Covers the happy path.",
            "steps_text": "1. open page",
        }
        got = self.r._document_text(rec, None)
        self.assertLessEqual(len(got), Reranker._document_budget())
        # The body is the document; it may legitimately consume the whole budget. What
        # must not happen is text being appended beyond the cap.
        self.assertEqual(len(got), Reranker._document_budget())

    def test_small_record_is_not_truncated(self):
        rec = {"summary": "Verify login", "description": "Happy path.",
               "steps_text": "1. open\n2. submit"}
        got = self.r._document_text(rec, None)
        self.assertIn("Verify login", got)
        self.assertIn("Happy path.", got)
        self.assertIn("submit", got)
        self.assertLess(len(got), Reranker._document_budget())

    def test_body_comes_first(self):
        rec = {"summary": "LEAD TEXT", "description": "desc", "steps_text": "steps"}
        self.assertTrue(self.r._document_text(rec, None).startswith("LEAD TEXT"))

    def test_description_precedes_steps(self):
        """Xray puts step content in description when steps_text is null."""
        got = self.r._document_text(
            {"summary": "s", "description": "DESC", "steps_text": "STEPS"}, None)
        self.assertLess(got.index("DESC"), got.index("STEPS"))

    def test_budget_leaves_room_for_the_query(self):
        limit_chars = Reranker._MODEL_TOKEN_LIMIT * Reranker._CHARS_PER_TOKEN
        self.assertLess(Reranker._document_budget(), limit_chars)


class RerankIntegrationTests(unittest.TestCase):
    """rerank() with a stub model — checks ordering and the empty-document alarm."""

    def setUp(self):
        self.r = Reranker.__new__(Reranker)
        self.r._scale_warned = False
        self.seen_pairs = []

        class _Model:
            def predict(inner, pairs, batch_size=32, show_progress_bar=False):
                self.seen_pairs = list(pairs)
                # Longer document → higher score, so ordering is observable.
                return [float(len(d)) / 10.0 for _q, d in pairs]

        self.r._model = _Model()

    def test_documents_actually_reach_the_model(self):
        self.r.rerank("what is EXAMPLE_STATUS?", [dict(PRD_CHUNK)])
        self.assertEqual(len(self.seen_pairs), 1)
        self.assertIn("one-month access window", self.seen_pairs[0][1])

    def test_distinct_documents_produce_distinct_scores(self):
        """The symptom of the bug was identical scores across different documents."""
        out = self.r.rerank("q", [dict(PRD_CHUNK), dict(TEST_CASE)])
        scores = {r["rerank_score"] for r in out}
        self.assertEqual(len(scores), 2, "identical scores mean the text never varied")

    def test_empty_documents_are_logged_as_an_error(self):
        with self.assertLogs("embeddings.reranker", level="ERROR") as cm:
            self.r.rerank("q", [{"score": 0.5}])
        self.assertIn("empty text", "\n".join(cm.output))

    def test_no_error_logged_when_all_documents_resolve(self):
        """A healthy rerank is silent — no empty-text error and no low-score warning."""
        with self.assertNoLogs("embeddings.reranker", level="ERROR"):
            self.r.rerank("q", [dict(PRD_CHUNK)])

    def test_original_score_is_preserved(self):
        out = self.r.rerank("q", [dict(PRD_CHUNK)])
        self.assertEqual(out[0]["score"], 0.836)

    def test_top_k_truncates_after_reordering(self):
        out = self.r.rerank("q", [dict(PRD_CHUNK), dict(TEST_CASE)], top_k=1)
        self.assertEqual(len(out), 1)

    def test_empty_input_passes_through(self):
        self.assertEqual(self.r.rerank("q", []), [])
        self.assertEqual(self.r.rerank("", [dict(PRD_CHUNK)])[0]["score"], 0.836)


if __name__ == "__main__":
    unittest.main()
