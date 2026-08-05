"""
Guards the grounding contract of /ask.

The whole value of this endpoint is that an answer can be checked. Two failure modes make
it worthless and both are silent:

  * an uncited answer presented as if it were grounded — indistinguishable from invention
  * an abstention ("not documented") counted as a successful answer

Both are pinned here, along with the citation parsing and passage numbering that the
citations list depends on.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from test_context_budget import _install_stubs  # noqa: E402

_install_stubs()
from agents import ask as A  # noqa: E402


class _FakeEmbed:
    def embed_query(self, text):
        return [0.0] * 8


class _FakeES:
    """Returns a fixed chunk set; records the filters it was asked for."""

    def __init__(self, chunks=None, tests=None):
        self._chunks = chunks if chunks is not None else []
        self._tests = tests or []
        self.calls = {}

    def search_similar_prd_chunks(self, **kw):
        self.calls["prd"] = kw
        return [dict(c) for c in self._chunks]

    def search_similar_tests(self, **kw):
        self.calls["tests"] = kw
        return [dict(t) for t in self._tests]


class _FakeLLM:
    def __init__(self, text):
        self._text = text

    def invoke(self, messages):
        self.messages = messages
        return type("R", (), {"content": self._text, "usage_metadata": {
            "input_tokens": 100, "output_tokens": 20, "input_token_details": {}}})()


CHUNKS = [
    {"source_id": "confluence:1000000001", "doc_title": "Implementation Plan for PROVISIONAL status",
     "section_heading": "1. Background", "chunk_text": "PROVISIONAL gives 1-month access.",
     "doc_url": "https://x/1000000001", "score": 0.836},
    {"source_id": "confluence:1000000002", "doc_title": "Report-anchored verification",
     "section_heading": "Additional Requirement", "chunk_text": "Seven validation checks apply.",
     "doc_url": "https://x/1000000002", "score": 0.811},
]


class AskTest(unittest.TestCase):
    def _run(self, answer_text, chunks=None, **kw):
        es = _FakeES(CHUNKS if chunks is None else chunks)
        orig = A._build_llm
        A._build_llm = lambda provider, model: _FakeLLM(answer_text)
        try:
            return A.answer_question("what is EXAMPLE_STATUS?", _FakeEmbed(), es,
                                     provider="ollama", model="gemma4:12b", **kw), es
        finally:
            A._build_llm = orig

    # ── grounding ─────────────────────────────────────────────────────────────

    def test_cited_answer_is_grounded(self):
        r, _ = self._run("PROVISIONAL grants 1-month access [1]. Seven checks apply [2].")
        self.assertTrue(r["grounded"])
        self.assertFalse(r["abstained"])
        self.assertEqual(r["cited_passages"], [1, 2])
        self.assertEqual([c["n"] for c in r["citations"]], [1, 2])

    def test_uncited_answer_is_NOT_grounded(self):
        """The dangerous case: fluent, plausible, unverifiable."""
        r, _ = self._run("PROVISIONAL grants accounts one month of platform access.")
        self.assertFalse(r["grounded"])
        self.assertEqual(r["cited_passages"], [])

    def test_abstention_is_not_grounded_but_is_flagged(self):
        r, _ = self._run("Not documented in the indexed corpus. Closest material is [1].")
        self.assertTrue(r["abstained"])
        self.assertFalse(r["grounded"], "abstaining is correct, but it is not an answer")

    def test_partial_citation_reports_only_used_passages(self):
        r, _ = self._run("Only the first passage matters [1].")
        self.assertEqual(r["cited_passages"], [1])
        self.assertEqual([c["n"] for c in r["citations"]], [1])

    def test_out_of_range_citations_are_discarded(self):
        """A model citing [7] against 2 passages must not produce a bogus citation."""
        r, _ = self._run("Claim [1] and invented [7].")
        self.assertEqual(r["cited_passages"], [1])

    def test_empty_retrieval_abstains_without_calling_the_llm(self):
        es = _FakeES([])
        orig = A._build_llm
        A._build_llm = lambda p, m: (_ for _ in ()).throw(AssertionError("LLM must not be called"))
        try:
            r = A.answer_question("anything", _FakeEmbed(), es, provider="ollama", model="m")
        finally:
            A._build_llm = orig
        self.assertFalse(r["grounded"])
        self.assertIn("Not documented", r["answer"])
        self.assertEqual(r["context_used"], [])

    # ── context returned for verification ─────────────────────────────────────

    def test_context_used_is_returned_for_auditing(self):
        r, _ = self._run("Answer [1].")
        self.assertEqual(len(r["context_used"]), 2)
        first = r["context_used"][0]
        for field in ("n", "source_id", "doc_title", "section_heading", "score", "url", "text"):
            self.assertIn(field, first)
        self.assertEqual(first["source_id"], "confluence:1000000001")

    def test_passage_numbering_matches_between_prompt_and_citations(self):
        r, _ = self._run("Answer [2].")
        prompt = A._format_context(CHUNKS)
        self.assertIn("[1] Implementation Plan for PROVISIONAL status", prompt)
        self.assertIn("[2] Report-anchored verification", prompt)
        self.assertEqual(r["citations"][0]["source_id"], "confluence:1000000002")

    # ── scoping ───────────────────────────────────────────────────────────────

    def test_title_contains_is_passed_through_to_retrieval(self):
        _, es = self._run("Answer [1].", title_contains="PRD")
        self.assertEqual(es.calls["prd"]["title_contains"], "PRD")

    def test_tests_are_not_searched_unless_opted_in(self):
        _, es = self._run("Answer [1].")
        self.assertNotIn("tests", es.calls)

    def test_include_tests_searches_the_test_index(self):
        es = _FakeES(CHUNKS, tests=[{"jira_key": "PROJ-211", "summary": "login test",
                                     "steps_text": "", "score": 0.7}])
        orig = A._build_llm
        A._build_llm = lambda p, m: _FakeLLM("Answer [1].")
        try:
            r = A.answer_question("q", _FakeEmbed(), es, provider="ollama", model="m",
                                  include_tests=True)
        finally:
            A._build_llm = orig
        self.assertIn("tests", es.calls)
        self.assertIn("test", {c["kind"] for c in r["context_used"]})

    # ── accounting ────────────────────────────────────────────────────────────

    def test_token_usage_is_reported(self):
        r, _ = self._run("Answer [1].")
        u = r["token_usage"]
        self.assertEqual(u["input_tokens"], 100)
        self.assertEqual(u["output_tokens"], 20)
        self.assertEqual(u["estimated_cost_usd"], 0.0, "ollama is local, cost 0")
        self.assertEqual(u["cost_basis"], "local_inference")

    # ── prompt contract ───────────────────────────────────────────────────────

    def test_prompt_forbids_outside_knowledge_and_permits_abstaining(self):
        p = A.SYSTEM_PROMPT.lower()
        self.assertIn("only the numbered context", p)
        self.assertIn("not documented in the indexed corpus", p)
        self.assertIn("do not use outside knowledge", p)

    def test_prompt_requires_verbatim_identifiers(self):
        """Paraphrasing PROVISIONAL or rounding '1-month' destroys the answer's use."""
        self.assertIn("verbatim", A.SYSTEM_PROMPT.lower())


if __name__ == "__main__":
    unittest.main()
