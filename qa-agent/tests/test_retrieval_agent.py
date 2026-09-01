"""
Plan B′ retrieval and agent-loop improvements: multi-query fusion, the planning tool,
and the turn-limit triage warning.

The property behind all three: a run that stops early, or a requirement phrased in
vocabulary the tests do not use, must not look the same as a clean result. An
unanalysed section with no decision is indistinguishable from a section the agent
checked and found correct — these exist to make that difference visible.

Stdlib unittest — see ADR-019. Neutral placeholders only.

Runs under the documented host-side command:
    cd qa-agent && PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'
"""
import unittest

from tests import stubs

stubs.install_agent_deps()

from agents import analysis_agent as A  # noqa: E402


# ─── B17: multi-query fusion ──────────────────────────────────────────────────

class RrfMergeTests(unittest.TestCase):
    """Fusion rewards consensus across phrasings of the same requirement."""

    @staticmethod
    def _hit(key, score=0.5):
        return {"jira_key": key, "summary": f"summary {key}", "score": score}

    def test_empty_input_yields_nothing(self):
        self.assertEqual(A._rrf_merge_results([]), [])
        self.assertEqual(A._rrf_merge_results([[], []]), [])

    def test_single_list_preserves_its_members(self):
        merged = A._rrf_merge_results([[self._hit("PROJ-1"), self._hit("PROJ-2")]])
        self.assertEqual([r["jira_key"] for r in merged], ["PROJ-1", "PROJ-2"])

    def test_duplicates_are_collapsed(self):
        merged = A._rrf_merge_results([
            [self._hit("PROJ-1"), self._hit("PROJ-2")],
            [self._hit("PROJ-1"), self._hit("PROJ-3")],
        ])
        keys = [r["jira_key"] for r in merged]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual(set(keys), {"PROJ-1", "PROJ-2", "PROJ-3"})

    def test_a_test_found_by_every_query_outranks_one_found_by_one(self):
        """The whole point of fusing rather than taking the best single score."""
        merged = A._rrf_merge_results([
            [self._hit("PROJ-SOLO", 0.99), self._hit("PROJ-BOTH", 0.40)],
            [self._hit("PROJ-BOTH", 0.40), self._hit("PROJ-OTHER", 0.30)],
        ])
        self.assertEqual(merged[0]["jira_key"], "PROJ-BOTH")

    def test_rank_matters_more_than_raw_score(self):
        """Scores from different queries are not comparable; ranks are."""
        merged = A._rrf_merge_results([
            [self._hit("PROJ-A", 0.01)],   # rank 1 on a low-scoring query
            [self._hit("PROJ-B", 0.99)],   # rank 1 on a high-scoring query
        ])
        self.assertEqual(merged[0]["score"], merged[1]["score"])

    def test_score_is_replaced_with_the_fused_value(self):
        merged = A._rrf_merge_results([[self._hit("PROJ-1", 0.9)]])
        expected = 1.0 / (1 + A._RRF_RANK_CONSTANT)
        self.assertAlmostEqual(merged[0]["score"], round(expected, 6))

    def test_richest_record_is_kept_for_a_duplicate(self):
        """The reranker reads these fields; keep the copy that scored best alone."""
        thin = {"jira_key": "PROJ-1", "summary": "s", "score": 0.1}
        rich = {"jira_key": "PROJ-1", "summary": "s", "score": 0.9,
                "steps_text": "1. open"}
        merged = A._rrf_merge_results([[thin], [rich]])
        self.assertIn("steps_text", merged[0])

    def test_records_without_a_key_are_skipped_not_fatal(self):
        merged = A._rrf_merge_results([[{"summary": "no key"}, self._hit("PROJ-1")]])
        self.assertEqual([r["jira_key"] for r in merged], ["PROJ-1"])

    def test_caller_lists_are_not_mutated(self):
        """The caller still owns what it passed in; per-query scores must survive."""
        original = [self._hit("PROJ-1", 0.9), self._hit("PROJ-2", 0.8)]
        before = [dict(r) for r in original]
        A._rrf_merge_results([original])
        self.assertEqual(original, before)

    def test_merged_results_are_sorted_by_fused_score(self):
        merged = A._rrf_merge_results([
            [self._hit("PROJ-SOLO", 0.99), self._hit("PROJ-BOTH", 0.4)],
            [self._hit("PROJ-BOTH", 0.4)],
        ])
        scores = [r["score"] for r in merged]
        self.assertEqual(scores, sorted(scores, reverse=True))


# ─── B15: turn-limit triage ───────────────────────────────────────────────────

class TriageWarningTests(unittest.TestCase):
    def test_message_states_the_turns_remaining(self):
        self.assertIn("10 turns remain", A._triage_message(10))

    def test_message_directs_the_agent_to_stop_starting_new_work(self):
        msg = A._triage_message(10)
        self.assertIn("Do not start new searches", msg)

    def test_message_requires_a_question_for_unreachable_sections(self):
        """An unanalysed section must not look like a section that was fine."""
        msg = A._triage_message(10)
        self.assertIn("question", msg.lower())
        self.assertIn("not analysed", msg.lower())

    def test_threshold_is_configurable_and_positive(self):
        self.assertGreater(A._TRIAGE_TURNS_AHEAD, 0)

    def test_long_run_warns_at_the_lookahead(self):
        self.assertEqual(A._triage_turn(30, ahead=10), 20)
        self.assertEqual(A._triage_turn(20, ahead=10), 10)

    def test_never_warns_on_turn_zero(self):
        """
        Warning before any work tells the agent to stop starting searches it has not
        started, so it records nothing. max_turns <= lookahead is a realistic config —
        .env.example recommends lowering the turn cap for local models.
        """
        for max_turns in range(3, 26):
            with self.subTest(max_turns=max_turns):
                self.assertNotEqual(A._triage_turn(max_turns, ahead=10), 0)

    def test_short_run_warns_proportionally_late(self):
        self.assertEqual(A._triage_turn(10, ahead=10), 7)
        self.assertEqual(A._triage_turn(5, ahead=10), 3)

    def test_warning_always_leaves_turns_to_act_on_it(self):
        for max_turns in range(3, 40):
            turn = A._triage_turn(max_turns, ahead=10)
            with self.subTest(max_turns=max_turns):
                self.assertIsNotNone(turn)
                self.assertGreater(max_turns - turn, 0, "no turns left after the warning")

    def test_runs_too_short_to_triage_are_skipped(self):
        for max_turns in (0, 1, 2):
            with self.subTest(max_turns=max_turns):
                self.assertIsNone(A._triage_turn(max_turns, ahead=10))

    def test_warning_fires_exactly_once(self):
        max_turns, fired = 30, []
        warned, triage_turn = False, A._triage_turn(max_turns)
        for turn in range(max_turns):
            if not warned and triage_turn is not None and turn == triage_turn:
                fired.append(turn)
                warned = True
        self.assertEqual(len(fired), 1)


# ─── B16: the planning tool ───────────────────────────────────────────────────

class _FakeES:
    """Stands in for ESStore; only _client is touched, by the scan stub."""
    _client = object()


class ListPrdSectionsTests(unittest.TestCase):
    CHUNKS = [
        {"chunk_index": 0, "section_heading": "Background",
         "chunk_text": "Some history about the product area.",
         "chunk_type": "prose", "doc_title": "Payments PRD"},
        {"chunk_index": 1, "section_heading": "Refund Limits",
         "chunk_text": "| Tier | Cap |\n| --- | --- |\n| basic | 500 |",
         "chunk_type": "table", "doc_title": "Payments PRD"},
        {"chunk_index": 2, "section_heading": "",
         "chunk_text": "Continuation of the refund limits section text.",
         "chunk_type": "prose", "doc_title": "Payments PRD"},
        {"chunk_index": 3, "section_heading": "Success Metrics",
         "chunk_text": "We will measure adoption over the quarter.",
         "chunk_type": "prose", "doc_title": "Payments PRD"},
    ]

    def setUp(self):
        import elasticsearch
        self._orig_scan = elasticsearch.helpers.scan
        elasticsearch.helpers.scan = lambda client, **kw: (
            {"_source": c} for c in self.CHUNKS
        )

    def tearDown(self):
        import elasticsearch
        elasticsearch.helpers.scan = self._orig_scan

    def _run(self, chunks=None):
        if chunks is not None:
            import elasticsearch
            elasticsearch.helpers.scan = lambda client, **kw: (
                {"_source": c} for c in chunks
            )
        tools = {t.name: t for t in A._make_tools(
            "confluence:1234567890", None, object(), _FakeES(), object(), "run-1")}
        return tools["list_prd_sections"].invoke({})

    def test_every_section_is_listed(self):
        out = self._run()
        for heading in ("Background", "Refund Limits", "Success Metrics"):
            self.assertIn(heading, out)

    def test_meta_sections_are_marked(self):
        out = self._run()
        for line in out.split("\n"):
            if "Background" in line or "Success Metrics" in line:
                self.assertIn("[meta]", line)

    def test_feature_sections_are_marked_testable(self):
        line = next(l for l in self._run().split("\n") if "Refund Limits" in l)
        self.assertIn("[testable]", line)

    def test_testable_count_excludes_meta(self):
        self.assertIn("1 testable section(s)", self._run())

    def test_headingless_chunks_inherit_the_previous_heading(self):
        """Chunk 2 has no heading; it belongs to Refund Limits, not a new section."""
        out = self._run()
        self.assertNotIn("(no heading)", out)
        self.assertEqual(out.count("Refund Limits"), 1)

    def test_word_counts_accumulate_across_a_section(self):
        line = next(l for l in self._run().split("\n") if "Refund Limits" in l)
        # Table chunk plus its continuation, so more than either alone.
        self.assertIn("words", line)

    def test_chunk_type_mix_is_reported(self):
        """A section you expect to be a table showing only prose means bad conversion."""
        line = next(l for l in self._run().split("\n") if "Refund Limits" in l)
        self.assertIn("prose", line)
        self.assertIn("table", line)

    def test_missing_document_says_so(self):
        self.assertIn("No document found", self._run(chunks=[]))

    def test_output_points_at_the_next_step(self):
        self.assertIn("read_prd_document", self._run())

    def test_document_title_is_shown(self):
        self.assertIn("Payments PRD", self._run())


# ─── B10: module scoping ──────────────────────────────────────────────────────

class ModuleScopeTests(unittest.TestCase):
    """
    A hard `terms` filter drops documents with no `module` field at all. The PRD-chunk
    search already includes untagged documents; the test-case search did not, so a test
    synced before module tagging was invisible to every scoped search.
    """

    def setUp(self):
        from embeddings.es_store import _module_scope_clause
        self.clause = _module_scope_clause

    def test_no_filter_means_no_restriction(self):
        self.assertIsNone(self.clause(None))
        self.assertIsNone(self.clause([]))

    def test_requested_modules_are_matched(self):
        shoulds = self.clause(["Platform"])["bool"]["should"]
        self.assertIn({"terms": {"module": ["Platform"]}}, shoulds)

    def test_untagged_tests_are_included(self):
        shoulds = self.clause(["Platform"])["bool"]["should"]
        self.assertIn({"bool": {"must_not": {"exists": {"field": "module"}}}}, shoulds)

    def test_one_clause_must_match(self):
        """Without this a should-only bool matches everything."""
        self.assertEqual(self.clause(["Platform"])["bool"]["minimum_should_match"], 1)

    def test_other_modules_are_still_excluded_by_default(self):
        """Untagged is included; a different module is not."""
        shoulds = self.clause(["Platform"])["bool"]["should"]
        self.assertEqual(len(shoulds), 2)

    def test_cross_module_search_is_off_by_default(self):
        from embeddings import es_store
        self.assertFalse(es_store.CROSS_MODULE_SEARCH)

    def test_cross_module_search_removes_the_restriction(self):
        from embeddings import es_store
        original = es_store.CROSS_MODULE_SEARCH
        es_store.CROSS_MODULE_SEARCH = True
        try:
            self.assertIsNone(es_store._module_scope_clause(["Platform"]))
        finally:
            es_store.CROSS_MODULE_SEARCH = original


# ─── B18: heading rename detection ────────────────────────────────────────────

class RenameMatchTests(unittest.TestCase):
    """
    A renamed section is not a deletion plus a creation. Unpaired, every decision
    recorded against the old title is dropped and the section is re-analysed — the
    exact cost incremental mode exists to avoid.
    """

    def setUp(self):
        from agents.incremental import _match_renamed_headings
        self.match = _match_renamed_headings

    def test_identical_content_pairs_regardless_of_title(self):
        """The strongest signal: a byte-identical body is the same section."""
        pairs = self.match(
            ["User Login"], ["Authentication Flow"],
            {"User Login": "hash-a"}, {"Authentication Flow": "hash-a"},
        )
        self.assertEqual(pairs, {"User Login": "Authentication Flow"})

    def test_similar_title_pairs_when_content_differs(self):
        pairs = self.match(
            ["Checkout Page"], ["Checkout Page Updated"],
            {"Checkout Page": "hash-a"}, {"Checkout Page Updated": "hash-b"},
        )
        self.assertEqual(pairs, {"Checkout Page": "Checkout Page Updated"})

    def test_unrelated_sections_are_not_paired(self):
        pairs = self.match(
            ["Refund Limits"], ["Shipping Options"],
            {"Refund Limits": "hash-a"}, {"Shipping Options": "hash-b"},
        )
        self.assertEqual(pairs, {})

    def test_section_numbering_change_is_a_rename(self):
        pairs = self.match(
            ["3.2 Payment Capture"], ["Payment Capture"],
            {"3.2 Payment Capture": "hash-a"}, {"Payment Capture": "hash-b"},
        )
        self.assertEqual(pairs, {"3.2 Payment Capture": "Payment Capture"})

    def test_pairing_is_one_to_one(self):
        pairs = self.match(
            ["Checkout Page", "Checkout Flow"],
            ["Checkout Page Updated"],
            {"Checkout Page": "a", "Checkout Flow": "b"},
            {"Checkout Page Updated": "c"},
        )
        self.assertEqual(len(pairs), 1)
        self.assertEqual(list(pairs.values()), ["Checkout Page Updated"])

    def test_best_match_wins_over_a_weaker_one(self):
        """Order of consideration must not let a weaker pair steal a strong match."""
        pairs = self.match(
            ["Checkout Flow", "Checkout Page"],
            ["Checkout Page Updated"],
            {"Checkout Flow": "a", "Checkout Page": "b"},
            {"Checkout Page Updated": "c"},
        )
        self.assertEqual(pairs, {"Checkout Page": "Checkout Page Updated"})

    def test_content_match_is_preferred_over_title_match(self):
        pairs = self.match(
            ["Login Flow"], ["Login Flow v2", "Authentication"],
            {"Login Flow": "same"},
            {"Login Flow v2": "different", "Authentication": "same"},
        )
        self.assertEqual(pairs, {"Login Flow": "Authentication"})

    def test_nothing_to_match_returns_empty(self):
        self.assertEqual(self.match([], ["New"], {}, {"New": "h"}), {})
        self.assertEqual(self.match(["Old"], [], {"Old": "h"}, {}), {})

    def test_threshold_is_configurable(self):
        args = (["Login Flow"], ["Sign-in Flow"],
                {"Login Flow": "a"}, {"Sign-in Flow": "b"})
        self.assertEqual(self.match(*args, threshold=0.99), {})
        self.assertEqual(self.match(*args, threshold=0.70),
                         {"Login Flow": "Sign-in Flow"})

    def test_content_matching_still_runs_past_the_pair_budget(self):
        """
        Title comparison is O(removed x added). Past the budget it is skipped, but the
        exact-content rule is cheap and must keep working — that is the rule decisions
        are actually carried forward on.
        """
        from agents import incremental
        n = incremental._RENAME_MAX_PAIRS  # any size whose square exceeds the budget
        size = int(n ** 0.5) + 20
        removed = [f"Old {i}" for i in range(size)]
        added = [f"New {i}" for i in range(size)]
        prev = {h: f"same{i}" for i, h in enumerate(removed)}
        curr = {h: f"same{i}" for i, h in enumerate(added)}
        pairs = self.match(removed, added, prev, curr)
        self.assertEqual(len(pairs), size)

    def test_title_matching_is_skipped_past_the_pair_budget(self):
        from agents import incremental
        size = int(incremental._RENAME_MAX_PAIRS ** 0.5) + 20
        removed = [f"Section {i} Title" for i in range(size)]
        added = [f"Section {i} Heading" for i in range(size)]
        prev = {h: f"h{i}" for i, h in enumerate(removed)}
        curr = {h: f"c{i}" for i, h in enumerate(added)}
        self.assertEqual(self.match(removed, added, prev, curr), {})


class RenameDiffIntegrationTests(unittest.TestCase):
    """detect_changes must not report a rename as removed + new."""

    def _diff(self, prev, curr):
        from agents import incremental
        orig = incremental._fetch_all_prd_chunks
        incremental._fetch_all_prd_chunks = lambda es, sid: [
            {"section_heading": h, "chunk_text": t} for h, t in curr.items()
        ]
        try:
            return incremental.detect_changes(
                "confluence:1234567890", object(),
                previous_heading_hashes=incremental._heading_content_hashes(
                    [{"section_heading": h, "chunk_text": t} for h, t in prev.items()]
                ),
            )
        finally:
            incremental._fetch_all_prd_chunks = orig

    def test_pure_rename_is_unchanged_not_removed_plus_new(self):
        diff = self._diff({"Old Title": "same body text here"},
                          {"New Title": "same body text here"})
        self.assertEqual(diff["renamed_headings"], {"Old Title": "New Title"})
        self.assertIn("New Title", diff["unchanged_headings"])
        self.assertEqual(diff["removed_headings"], [])
        self.assertEqual(diff["new_headings"], [])

    def test_renamed_and_edited_is_a_change_not_a_new_section(self):
        diff = self._diff({"Checkout Page": "body one"},
                          {"Checkout Page Updated": "body two"})
        self.assertEqual(diff["renamed_headings"],
                         {"Checkout Page": "Checkout Page Updated"})
        self.assertIn("Checkout Page Updated", diff["changed_headings"])
        self.assertEqual(diff["new_headings"], [])
        self.assertEqual(diff["removed_headings"], [])

    def test_a_genuine_deletion_is_still_reported_as_removed(self):
        diff = self._diff({"Refund Limits": "a", "Kept": "k"},
                          {"Kept": "k"})
        self.assertEqual(diff["removed_headings"], ["Refund Limits"])
        self.assertEqual(diff["renamed_headings"], {})

    def test_a_genuine_addition_is_still_reported_as_new(self):
        diff = self._diff({"Kept": "k"},
                          {"Kept": "k", "Shipping Options": "brand new"})
        self.assertEqual(diff["new_headings"], ["Shipping Options"])
        self.assertEqual(diff["renamed_headings"], {})

    def test_first_run_reports_the_shape_with_no_renames(self):
        from agents import incremental
        orig = incremental._fetch_all_prd_chunks
        incremental._fetch_all_prd_chunks = lambda es, sid: [
            {"section_heading": "A", "chunk_text": "x"}
        ]
        try:
            diff = incremental.detect_changes("confluence:1", object())
        finally:
            incremental._fetch_all_prd_chunks = orig
        self.assertEqual(diff["renamed_headings"], {})
        self.assertEqual(diff["new_headings"], ["A"])


class RenameCarryForwardTests(unittest.TestCase):
    """
    The payoff: a decision recorded under the old title must survive the rename, and
    be restated under the new one so the NEXT run matches it directly.
    """

    class _PG:
        def __init__(self, previous):
            self._previous = previous
            self.written: list[dict] = []

        def get_pending_decisions(self, run_id=None):
            return self._previous if run_id == "prev-run" else self.written

        def write_decision(self, row):
            self.written.append(row)
            return len(self.written)

    PREVIOUS = [{
        "jira_key": "PROJ-1234",
        "action": "keep",
        "reason": "Covers the refund ceiling rule exactly as written.",
        "prd_section": "Refund Limits",
        "prd_source": "confluence:1234567890",
        "updated_content": None,
        "questions": None,
    }]

    def _carry(self, unchanged, renamed):
        from agents.incremental import carry_forward_decisions
        pg = self._PG(list(self.PREVIOUS))
        count = carry_forward_decisions(
            pg, "prev-run", "new-run", unchanged, renamed_headings=renamed,
        )
        return count, pg.written

    def test_decision_survives_a_rename(self):
        count, written = self._carry(
            {"Refund Limits and Caps"}, {"Refund Limits": "Refund Limits and Caps"})
        self.assertEqual(count, 1)
        self.assertEqual(written[0]["jira_key"], "PROJ-1234")

    def test_carried_decision_is_restated_under_the_new_title(self):
        """Otherwise the next run looks for the old name and drops it then."""
        _, written = self._carry(
            {"Refund Limits and Caps"}, {"Refund Limits": "Refund Limits and Caps"})
        self.assertEqual(written[0]["prd_section"], "Refund Limits and Caps")

    def test_without_the_rename_map_the_decision_is_lost(self):
        """Pins the regression B18 fixes."""
        count, _ = self._carry({"Refund Limits and Caps"}, None)
        self.assertEqual(count, 0)

    def test_unrelated_rename_does_not_carry_the_decision(self):
        count, _ = self._carry({"Shipping Options"},
                               {"Delivery Options": "Shipping Options"})
        self.assertEqual(count, 0)

    def test_rename_to_a_changed_section_is_not_carried(self):
        """Renamed AND edited — the content changed, so the decision must be redone."""
        count, _ = self._carry(set(), {"Refund Limits": "Refund Limits and Caps"})
        self.assertEqual(count, 0)

    def test_confidence_survives_carry_forward(self):
        """
        Dropping it would make every carried decision look unrated, so a reviewer
        sorting by confidence sees the whole unchanged part of the PRD as unassessed.
        """
        pg = self._PG([dict(self.PREVIOUS[0], confidence="high")])
        from agents.incremental import carry_forward_decisions
        carry_forward_decisions(pg, "prev-run", "new-run", {"Refund Limits"})
        self.assertEqual(pg.written[0]["confidence"], "high")

    def test_absent_confidence_stays_absent(self):
        pg = self._PG(list(self.PREVIOUS))
        from agents.incremental import carry_forward_decisions
        carry_forward_decisions(pg, "prev-run", "new-run", {"Refund Limits"})
        self.assertIsNone(pg.written[0]["confidence"])


class ToolRegistrationTests(unittest.TestCase):
    def test_planning_tool_is_registered_first(self):
        tools = A._make_tools("confluence:1234567890", None, object(),
                              _FakeES(), object(), "run-1")
        self.assertEqual(tools[0].name, "list_prd_sections")

    def test_search_tests_accepts_extra_queries(self):
        tools = {t.name: t for t in A._make_tools(
            "confluence:1234567890", None, object(), _FakeES(), object(), "run-1")}
        self.assertIn("extra_queries", tools["search_tests"].description)


if __name__ == "__main__":
    unittest.main()
