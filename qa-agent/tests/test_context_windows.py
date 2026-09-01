"""
Guards the model-aware context budget and the read_prd_document output cap.

Two bugs motivated these:

1. The budget was a per-PROVIDER constant of 180,000 written for a 200K Claude window.
   Against a 1M model that discarded ~82% of the window and, worse, tripped compaction
   (which truncates old tool results to 300 chars) while there was ample room to keep them.

2. read_prd_document had no output cap at all. At MAX_PRD_CHUNKS=2000 a single call could
   return ~1.6M tokens — 71x the local budget. Ollama enforces num_ctx by silently dropping
   overflow, so the result was an agent reasoning about a truncated PRD with no error raised.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from test_context_budget import _install_stubs  # noqa: E402

_install_stubs()
from agents import analysis_agent as A  # noqa: E402


class ContextWindowTests(unittest.TestCase):
    _KEYS = ("QA_AGENT_CONTEXT_LIMIT", "QA_AGENT_MAX_OUTPUT_TOKENS", "OLLAMA_NUM_CTX")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self._KEYS}
        for k in self._KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_million_token_model_gets_a_million_token_budget(self):
        """The regression: a 1M model must not be capped at 180K."""
        limit = A._context_limit("anthropic", "claude-opus-5")
        self.assertGreater(limit, 700_000)
        self.assertLess(limit, 1_000_000, "must stay under the real window")

    def test_budget_scales_with_the_model_not_the_provider(self):
        big = A._context_limit("anthropic", "claude-opus-5")          # 1M
        small = A._context_limit("anthropic", "claude-haiku-4-5")     # 200K
        self.assertGreater(big, small * 4)

    def test_budget_reserves_room_for_output(self):
        os.environ["QA_AGENT_MAX_OUTPUT_TOKENS"] = "32000"
        window = A._MODEL_CONTEXT_WINDOWS["claude-opus-5"]
        limit = A._context_limit("anthropic", "claude-opus-5")
        self.assertLessEqual(limit + 32000, window)

    def test_tokenizer_undercount_is_discounted(self):
        """cl100k undercounts Claude, so the budget must sit below the raw window."""
        window = A._MODEL_CONTEXT_WINDOWS["claude-sonnet-4-6"]
        self.assertLess(A._context_limit("anthropic", "claude-sonnet-4-6"),
                        window * 0.9)

    def test_unknown_model_falls_back_conservatively(self):
        """A model released after this table was written must not get an optimistic guess."""
        limit = A._context_limit("anthropic", "claude-something-unreleased")
        fallback = A._PROVIDER_FALLBACK_WINDOW["anthropic"]
        self.assertLessEqual(limit, fallback)
        self.assertGreater(limit, 8_000)

    def test_explicit_override_wins(self):
        os.environ["QA_AGENT_CONTEXT_LIMIT"] = "50000"
        self.assertEqual(A._context_limit("anthropic", "claude-opus-5"), 50_000)

    def test_malformed_override_is_ignored(self):
        os.environ["QA_AGENT_CONTEXT_LIMIT"] = "not-a-number"
        self.assertGreater(A._context_limit("anthropic", "claude-opus-5"), 700_000)

    def test_ollama_still_derives_from_num_ctx(self):
        os.environ["OLLAMA_NUM_CTX"] = "32768"
        os.environ["QA_AGENT_MAX_OUTPUT_TOKENS"] = "8192"
        self.assertEqual(A._context_limit("ollama", "any-tag"), 32768 - 8192 - 2000)

    def test_compaction_still_fits_every_hosted_model(self):
        """Prompt after compaction plus output must stay inside the real window."""
        os.environ["QA_AGENT_MAX_OUTPUT_TOKENS"] = "32000"
        for model, window in A._MODEL_CONTEXT_WINDOWS.items():
            with self.subTest(model=model):
                limit = A._context_limit("anthropic", model)
                compact_at = int(limit * A._TOKEN_COMPACT_RATIO)
                self.assertLess(compact_at + 32000, window)


class _FakeES:
    """Stands in for ESStore, serving a synthetic PRD via the helpers.scan seam."""

    def __init__(self, chunks):
        self._chunks = chunks
        self._client = object()


def _install_scan(chunks):
    """Point elasticsearch.helpers.scan at a fixed chunk list."""
    import elasticsearch

    elasticsearch.helpers.scan = lambda client, **kw: ({"_source": c} for c in chunks)


def _chunk(idx, heading, text, chunk_type=None):
    return {"chunk_index": idx, "section_heading": heading,
            "chunk_text": text, "doc_title": "Checkout PRD",
            "chunk_type": chunk_type}


class PrdReadBudgetTests(unittest.TestCase):
    def _tools(self, chunks, budget):
        _install_scan(chunks)
        dummy_pg = type("_NoPG", (), {"write_decision": lambda *a, **k: None})()
        tools = A._make_tools(
            "confluence:1", None, object(), _FakeES(chunks), dummy_pg, "test-run",
            prd_token_budget=budget,
        )
        return {t.name: t for t in tools}

    def test_small_prd_returned_in_full(self):
        chunks = [_chunk(0, "Login", "Users sign in with email."),
                  _chunk(1, "Checkout", "One-step checkout with saved card.")]
        out = self._tools(chunks, 8_000)["read_prd_document"].invoke({})
        self.assertIn("Users sign in with email.", out)
        self.assertIn("One-step checkout with saved card.", out)
        self.assertNotIn("too large", out)

    def test_oversized_prd_returns_outline_not_truncated_body(self):
        """The core fix: never silently exceed the budget."""
        chunks = [_chunk(i, f"Section {i}", "word " * 400) for i in range(40)]
        out = self._tools(chunks, 2_000)["read_prd_document"].invoke({})
        self.assertLessEqual(A._count_tokens_str(out), 2_000 * 1.35,
                             "output must respect the budget")
        self.assertIn("too large", out)
        # Every section must still be discoverable.
        self.assertIn("Section 0", out)
        self.assertIn("Section 39", out)
        self.assertIn("read_prd_document", out, "must tell the agent how to get the rest")

    def test_outline_lists_all_sections_even_when_bodies_omitted(self):
        chunks = [_chunk(i, f"Sec{i}", "word " * 500) for i in range(25)]
        out = self._tools(chunks, 1_500)["read_prd_document"].invoke({})
        for i in range(25):
            self.assertIn(f"Sec{i}", out, f"Sec{i} missing from outline")

    def test_named_section_read_returns_that_section(self):
        chunks = [_chunk(0, "Login", "LOGIN BODY"),
                  _chunk(1, "Checkout", "CHECKOUT BODY")]
        tools = self._tools(chunks, 8_000)
        out = tools["read_prd_document"].invoke({"section": "Checkout"})
        self.assertIn("CHECKOUT BODY", out)
        self.assertNotIn("LOGIN BODY", out)

    def test_named_section_matching_ignores_numbering_and_case(self):
        chunks = [_chunk(0, "3.2 Payment Capture", "PAYMENT BODY")]
        tools = self._tools(chunks, 8_000)
        for query in ("Payment Capture", "payment capture", "3.2 Payment Capture"):
            with self.subTest(query=query):
                self.assertIn("PAYMENT BODY",
                              tools["read_prd_document"].invoke({"section": query}))

    def test_unknown_section_lists_what_exists(self):
        chunks = [_chunk(0, "Login", "x"), _chunk(1, "Checkout", "y")]
        out = self._tools(chunks, 8_000)["read_prd_document"].invoke({"section": "Nope"})
        self.assertIn("No section matching", out)
        self.assertIn("Login", out)
        self.assertIn("Checkout", out)

    def test_chunks_inherit_the_preceding_heading(self):
        """Continuation chunks carry no heading; they belong to the section above."""
        chunks = [_chunk(0, "Login", "first part"), _chunk(1, "", "second part")]
        out = self._tools(chunks, 8_000)["read_prd_document"].invoke({"section": "Login"})
        self.assertIn("first part", out)
        self.assertIn("second part", out)

    def test_empty_prd_is_reported_not_crashed(self):
        out = self._tools([], 8_000)["read_prd_document"].invoke({})
        self.assertIn("No document found", out)

    def test_zero_budget_falls_back_to_a_safe_default(self):
        chunks = [_chunk(0, "Login", "body")]
        out = self._tools(chunks, 0)["read_prd_document"].invoke({})
        self.assertIn("body", out)


class SplitTableReassemblyTests(PrdReadBudgetTests):
    """
    A table spanning several chunks repeats its header in each one so the chunk
    stands alone in retrieval. Re-joined into one document that header is
    duplication — the model would see three short tables where the source had one.
    """

    HEAD = "| Field | Rule |\n| --- | --- |"

    def _split_table(self, chunk_type="table"):
        return [
            _chunk(0, "Rules", f"{self.HEAD}\n| field_a | rule_a |", chunk_type),
            _chunk(1, "", f"{self.HEAD}\n| field_b | rule_b |", chunk_type),
            _chunk(2, "", f"{self.HEAD}\n| field_c | rule_c |", chunk_type),
        ]

    def test_header_appears_once_in_the_reassembled_document(self):
        out = self._tools(self._split_table(), 8_000)["read_prd_document"].invoke({})
        self.assertEqual(out.count("| Field | Rule |"), 1)
        self.assertEqual(out.count("| --- | --- |"), 1)

    def test_no_row_is_lost(self):
        out = self._tools(self._split_table(), 8_000)["read_prd_document"].invoke({})
        for row in ("| field_a | rule_a |", "| field_b | rule_b |", "| field_c | rule_c |"):
            self.assertIn(row, out)

    def test_rows_stay_contiguous(self):
        out = self._tools(self._split_table(), 8_000)["read_prd_document"].invoke({})
        rows = [l for l in out.split("\n") if l.startswith("| field_")]
        start = out.split("\n").index(rows[0])
        block = out.split("\n")[start:start + 3]
        self.assertEqual(block, rows, "rows must not be separated by repeated headers")

    def test_chunks_indexed_before_chunk_type_existed_are_still_deduplicated(self):
        """chunk_type is absent on older chunks; the strip must still apply."""
        out = self._tools(self._split_table(chunk_type=None), 8_000)[
            "read_prd_document"].invoke({})
        self.assertEqual(out.count("| Field | Rule |"), 1)

    def test_a_genuinely_different_table_keeps_its_own_header(self):
        chunks = [
            _chunk(0, "Rules", f"{self.HEAD}\n| field_a | rule_a |", "table"),
            _chunk(1, "", "| Other | Columns |\n| --- | --- |\n| x | y |", "table"),
        ]
        out = self._tools(chunks, 8_000)["read_prd_document"].invoke({})
        self.assertIn("| Field | Rule |", out)
        self.assertIn("| Other | Columns |", out)

    def test_prose_chunks_are_unaffected(self):
        chunks = [_chunk(0, "Login", "first part", "prose"),
                  _chunk(1, "", "second part", "prose")]
        out = self._tools(chunks, 8_000)["read_prd_document"].invoke({})
        self.assertIn("first part", out)
        self.assertIn("second part", out)

    def test_single_section_read_also_deduplicates(self):
        out = self._tools(self._split_table(), 8_000)["read_prd_document"].invoke(
            {"section": "Rules"})
        self.assertEqual(out.count("| Field | Rule |"), 1)


if __name__ == "__main__":
    unittest.main()
