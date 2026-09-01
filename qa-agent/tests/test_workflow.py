"""
Plan C workflow guards: the cross-document deprecation check and the advisory
mechanism the decision guardrails share.

The property under test: DEPRECATE removes a test everywhere, but the agent analyses
one document at a time. "The feature is not in this PRD" is not "the feature is
undocumented" — the corpus holds many documents and the agent has seen one.

The second property is subtler. An advisory that says "go and check, then record it
again" must actually clear, or the instruction is impossible to follow and the decision
can never be recorded at all.

Stdlib unittest — see ADR-019. Neutral placeholders only.

Runs under the documented host-side command:
    cd qa-agent && PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'
"""
import unittest

from tests import stubs

stubs.install_agent_deps()

from agents import analysis_agent as A  # noqa: E402


CURRENT = "confluence:1234567890"
OTHER = "confluence:9999999999"


def _match(source_id, score, heading="Refund handling", title="Billing PRD"):
    return {"source_id": source_id, "score": score,
            "section_heading": heading, "doc_title": title}


class CrossDocumentVerdictTests(unittest.TestCase):
    def test_no_matches_allows_the_deprecation(self):
        self.assertIsNone(A._cross_prd_verdict([], CURRENT))

    def test_the_current_document_is_not_evidence_against_itself(self):
        """Every deprecation is found via the PRD being analysed; that proves nothing."""
        self.assertIsNone(A._cross_prd_verdict([_match(CURRENT, 0.99)], CURRENT))

    def test_another_document_above_the_threshold_warns(self):
        verdict = A._cross_prd_verdict([_match(OTHER, 0.86)], CURRENT)
        self.assertIsNotNone(verdict)
        self.assertIn(OTHER, verdict)

    def test_weak_matches_are_ignored(self):
        below = A._CROSS_PRD_SCORE - 0.05
        self.assertIsNone(A._cross_prd_verdict([_match(OTHER, below)], CURRENT))

    def test_verdict_names_the_document_and_section(self):
        verdict = A._cross_prd_verdict([_match(OTHER, 0.9)], CURRENT)
        self.assertIn("Billing PRD", verdict)
        self.assertIn("Refund handling", verdict)

    def test_verdict_offers_both_ways_forward(self):
        """Re-record with justification, or downgrade to a QUESTION."""
        verdict = A._cross_prd_verdict([_match(OTHER, 0.9)], CURRENT)
        self.assertIn("QUESTION", verdict)
        self.assertIn("reason", verdict)

    def test_one_line_per_document_not_per_chunk(self):
        """Several chunks of the same document are one piece of evidence."""
        verdict = A._cross_prd_verdict(
            [_match(OTHER, 0.9, "Refunds"), _match(OTHER, 0.88, "Caps"),
             _match(OTHER, 0.86, "Limits")],
            CURRENT,
        )
        self.assertEqual(sum(1 for l in verdict.split("\n") if l.startswith("  ")), 1)

    def test_documents_are_ranked_by_best_score(self):
        third = "confluence:1111111111"
        verdict = A._cross_prd_verdict(
            [_match(OTHER, 0.80), _match(third, 0.95, title="Platform PRD")], CURRENT)
        lines = [l for l in verdict.split("\n") if l.startswith("  ")]
        self.assertIn(third, lines[0])

    def test_at_most_three_documents_are_listed(self):
        matches = [_match(f"confluence:{i}" * 1, 0.9) for i in range(1000, 1008)]
        verdict = A._cross_prd_verdict(matches, CURRENT)
        self.assertLessEqual(sum(1 for l in verdict.split("\n") if l.startswith("  ")), 3)

    def test_records_without_a_source_are_skipped(self):
        self.assertIsNone(A._cross_prd_verdict([{"score": 0.99}], CURRENT))


# ─── The advisory mechanism ───────────────────────────────────────────────────

class _PG:
    def __init__(self):
        self.written = []

    def write_decision(self, d):
        self.written.append(d)
        return len(self.written)


class _ES:
    """Retrieval stub. `cross` and `similar` are set per test."""
    _client = object()
    cross: list = []
    similar: list = []

    def search_hybrid(self, query_embedding, keyword_query, top_k, module_filter=None):
        return [{"jira_key": "PROJ-1234", "summary": "Verify refund cap",
                 "score": 0.8, "module": "Platform", "labels": []}]

    def get_test_embedding(self, jira_key):
        return [0.1, 0.2]

    def search_similar_prd_chunks(self, vec, top_k=8, **kw):
        return self.cross

    def search_similar_tests(self, vec, top_k=3, module_filter=None, min_score=0.5):
        return self.similar


class _EMB:
    def embed_query(self, text):
        return [0.1, 0.2]


class AdvisoryClearsTests(unittest.TestCase):
    """
    An advisory that never clears makes its own instruction impossible: the agent is
    told to look and record again, but the second attempt returns the same advisory.
    """

    DEPRECATE_REASON = ("Section 4 confirms the refund cap feature was removed entirely "
                        "in this release.")
    CREATE_REASON = "No existing coverage for the refund ceiling rule in section 4.1."

    def setUp(self):
        self.pg = _PG()
        self.es = _ES()
        self.es.cross = []
        self.es.similar = []
        tools = {t.name: t for t in A._make_tools(
            CURRENT, ["Platform"], _EMB(), self.es, self.pg, "run-1")}
        self.record = tools["record_decision"]
        # The seen-keys guardrail requires a test to have been retrieved first.
        tools["search_tests"].invoke({"query": "refund cap"})

    def _deprecate(self):
        return self.record.invoke({
            "action": "deprecate", "jira_key": "PROJ-1234",
            "reason": self.DEPRECATE_REASON, "prd_section": "Refund Limits",
        })

    def _create(self, summary="Verify refund ceiling"):
        return self.record.invoke({
            "action": "create", "new_test_summary": summary,
            "reason": self.CREATE_REASON, "prd_section": "Refund Limits",
        })

    def test_cross_document_advisory_blocks_the_first_attempt(self):
        self.es.cross = [_match(OTHER, 0.9)]
        self.assertIn("other documents", self._deprecate())
        self.assertEqual(self.pg.written, [])

    def test_cross_document_advisory_clears_on_the_second_attempt(self):
        self.es.cross = [_match(OTHER, 0.9)]
        self._deprecate()
        self.assertIn("Decision recorded", self._deprecate())
        self.assertEqual(len(self.pg.written), 1)

    def test_clean_deprecate_records_immediately(self):
        self.assertIn("Decision recorded", self._deprecate())
        self.assertEqual(len(self.pg.written), 1)

    def test_duplicate_warning_clears_on_the_second_attempt(self):
        self.es.similar = [{"jira_key": "PROJ-5678", "summary": "Verify refund",
                            "score": A._DUP_WARN_SCORE + 0.01}]
        self.assertIn("Consider UPDATE", self._create())
        self.assertEqual(self.pg.written, [])
        self.assertIn("Decision recorded", self._create())
        self.assertEqual(len(self.pg.written), 1)

    def test_hard_block_never_clears(self):
        """A near-identical test is refused outright; UPDATE remains available."""
        self.es.similar = [{"jira_key": "PROJ-5678", "summary": "Verify refund",
                            "score": A._DUP_BLOCK_SCORE + 0.02}]
        for _ in range(3):
            self.assertIn("BLOCKED", self._create())
        self.assertEqual(self.pg.written, [])

    def test_advisories_are_tracked_per_decision_not_globally(self):
        """Clearing one section's advisory must not silence another's."""
        self.es.cross = [_match(OTHER, 0.9)]
        self._deprecate()
        self._deprecate()
        second = self.record.invoke({
            "action": "deprecate", "jira_key": "PROJ-1234",
            "reason": self.DEPRECATE_REASON, "prd_section": "A Different Section",
        })
        self.assertIn("other documents", second)

    def test_a_failed_lookup_does_not_block_the_decision(self):
        def _boom(jira_key):
            raise RuntimeError("elasticsearch unavailable")

        self.es.get_test_embedding = _boom
        self.assertIn("Decision recorded", self._deprecate())

    def test_a_test_with_no_stored_vector_is_not_blocked(self):
        self.es.get_test_embedding = lambda jira_key: None
        self.assertIn("Decision recorded", self._deprecate())


# ─── C19: pre-deprecation snapshot ────────────────────────────────────────────

class PreDeprecationSnapshotTests(unittest.TestCase):
    """
    Deprecation is the one irreversible action. The snapshot is what makes it
    recoverable, so it must capture exactly what deprecation changes — the labels —
    and be persisted before the change, not after.
    """

    def setUp(self):
        import asyncio
        from agents import writeback
        self.writeback = writeback
        self.run = asyncio.run
        self._orig_get = writeback.xray_client.get_labels

    def tearDown(self):
        self.writeback.xray_client.get_labels = self._orig_get

    def test_snapshot_captures_the_current_labels(self):
        self.writeback.xray_client.get_labels = _async_return(["regression", "smoke"])
        snap = self.run(self.writeback._pre_deprecation_snapshot("PROJ-1234"))
        self.assertEqual(snap["labels"], ["regression", "smoke"])
        self.assertEqual(snap["jira_key"], "PROJ-1234")

    def test_snapshot_records_when_it_was_taken(self):
        self.writeback.xray_client.get_labels = _async_return([])
        self.assertIn("captured_at",
                      self.run(self.writeback._pre_deprecation_snapshot("PROJ-1234")))

    def test_snapshot_holds_only_what_deprecation_changes(self):
        """
        deprecate_test appends a label and comments; it does not move folders. Recording
        folder/module would be state that was never touched, and restoring it could move
        a test someone had legitimately re-filed since.
        """
        self.writeback.xray_client.get_labels = _async_return(["regression"])
        snap = self.run(self.writeback._pre_deprecation_snapshot("PROJ-1234"))
        self.assertEqual(set(snap), {"jira_key", "labels", "captured_at"})

    def test_unreadable_labels_yield_no_snapshot_rather_than_a_wrong_one(self):
        def _boom(key):
            raise RuntimeError("jira unavailable")

        self.writeback.xray_client.get_labels = _boom
        self.assertIsNone(self.run(self.writeback._pre_deprecation_snapshot("PROJ-1234")))

    def test_empty_label_set_is_still_a_valid_snapshot(self):
        """A test with no labels must round-trip to no labels, not to 'unknown'."""
        self.writeback.xray_client.get_labels = _async_return([])
        snap = self.run(self.writeback._pre_deprecation_snapshot("PROJ-1234"))
        self.assertIsNotNone(snap)
        self.assertEqual(snap["labels"], [])


class DeprecationOrderingTests(unittest.TestCase):
    """The snapshot must be persisted BEFORE the deprecation call."""

    def test_snapshot_is_written_before_the_test_is_deprecated(self):
        """
        A crash between the two must leave a snapshot with no deprecation (harmless),
        never a deprecation with no snapshot (unrecoverable).
        """
        import asyncio
        from agents import writeback

        order: list[str] = []

        class _PG:
            def iter_writeback_decisions(self, run_id, batch_size=200):
                yield [{"id": 1, "action": "deprecate", "jira_key": "PROJ-1234",
                        "reason": "Removed entirely in section 4 of this release.",
                        "prd_source": "confluence:1234567890",
                        "prd_section": "Refund Limits", "updated_content": None}]

            def merge_decision_updated_content(self, did, patch):
                order.append("snapshot")
                return True

            def mark_written_back(self, did):
                order.append("marked")

        orig_get, orig_dep = writeback.xray_client.get_labels, writeback.xray_client.deprecate_test
        writeback.xray_client.get_labels = _async_return(["regression"])

        async def _dep(key, reason):
            order.append("deprecated")

        writeback.xray_client.deprecate_test = _dep
        try:
            asyncio.run(writeback.run_writeback(_PG(), run_id="run-1", project_key="PROJ"))
        finally:
            writeback.xray_client.get_labels = orig_get
            writeback.xray_client.deprecate_test = orig_dep

        self.assertEqual(order, ["snapshot", "deprecated", "marked"])


# ─── C21: coverage map ────────────────────────────────────────────────────────

class SectionGapRiskTests(unittest.TestCase):
    """
    The classification a single coverage score cannot express: a section nobody looked
    at must not read the same as one that was checked and found correct.
    """

    def setUp(self):
        from observability.request_norm import section_gap_risk
        self.risk = section_gap_risk

    def test_no_decision_at_all_is_uncovered(self):
        self.assertEqual(self.risk(None), "uncovered")
        self.assertEqual(self.risk({}), "uncovered")

    def test_a_keep_means_covered(self):
        self.assertEqual(self.risk({"keep_count": 1}), "covered")

    def test_an_update_means_covered(self):
        self.assertEqual(self.risk({"update_count": 1}), "covered")

    def test_only_create_is_unverified(self):
        """A gap was identified, but nothing tests it yet."""
        self.assertEqual(self.risk({"create_count": 2}), "unverified")

    def test_only_deprecate_is_shrinking(self):
        self.assertEqual(self.risk({"deprecate_count": 1}), "shrinking")

    def test_only_question_is_questioned(self):
        self.assertEqual(self.risk({"question_count": 3}), "questioned")

    def test_an_existing_test_outranks_a_proposed_one(self):
        """A section with both a KEEP and a CREATE is covered, not merely proposed."""
        self.assertEqual(self.risk({"keep_count": 1, "create_count": 4}), "covered")

    def test_deprecate_plus_create_is_unverified_not_shrinking(self):
        """Coverage is being replaced, not withdrawn — the new test is the open item."""
        self.assertEqual(
            self.risk({"deprecate_count": 1, "create_count": 1}), "unverified")

    def test_counts_present_but_all_zero_is_uncovered(self):
        self.assertEqual(
            self.risk({"keep_count": 0, "create_count": 0, "decisions": 0}), "uncovered")

    def test_every_risk_name_is_declared(self):
        from observability.request_norm import GAP_RISKS
        produced = {
            self.risk(None),
            self.risk({"keep_count": 1}),
            self.risk({"create_count": 1}),
            self.risk({"deprecate_count": 1}),
            self.risk({"question_count": 1}),
        }
        self.assertEqual(produced, set(GAP_RISKS))

    def test_risks_are_ordered_worst_first(self):
        from observability.request_norm import GAP_RISKS
        self.assertEqual(GAP_RISKS[0], "uncovered")
        self.assertEqual(GAP_RISKS[-1], "covered")


class CoverageMapJoinTests(unittest.TestCase):
    """
    The join between the document's sections and the decisions recorded for them.
    Both sides arrive normalised; comparing them raw is the bug that once made
    incremental carry-forward silently carry nothing.
    """

    def setUp(self):
        from observability.request_norm import merge_section_coverage
        self.merge = merge_section_coverage

    def test_a_section_with_no_decision_is_reported_as_uncovered(self):
        """The point of the endpoint: gaps are absent from the decisions table."""
        rows, _ = self.merge([("Refund Limits", "refund limits")], {})
        self.assertEqual(rows[0]["gap_risk"], "uncovered")
        self.assertEqual(rows[0]["decisions"], 0)

    def test_a_decided_section_carries_its_counts(self):
        rows, _ = self.merge(
            [("Refund Limits", "refund limits")],
            {"refund limits": {"decisions": 2, "keep_count": 2, "high_confidence": 1}},
        )
        self.assertEqual(rows[0]["gap_risk"], "covered")
        self.assertEqual(rows[0]["decisions"], 2)
        self.assertEqual(rows[0]["high_confidence"], 1)

    def test_document_order_is_preserved(self):
        rows, _ = self.merge(
            [("First", "first"), ("Second", "second"), ("Third", "third")], {})
        self.assertEqual([r["section"] for r in rows], ["First", "Second", "Third"])

    def test_display_heading_is_returned_not_the_normalised_key(self):
        rows, _ = self.merge([("3.2 Payment Capture", "payment capture")], {})
        self.assertEqual(rows[0]["section"], "3.2 Payment Capture")

    def test_decisions_matching_no_heading_are_reported_separately(self):
        rows, unmatched = self.merge(
            [("Refund Limits", "refund limits")],
            {"refund limits": {"keep_count": 1},
             "invented section": {"keep_count": 1, "prd_section": "Invented Section"}},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(unmatched, ["Invented Section"])

    def test_a_matched_decision_is_not_also_reported_as_unmatched(self):
        _, unmatched = self.merge(
            [("Refund Limits", "refund limits")],
            {"refund limits": {"keep_count": 1, "prd_section": "Refund Limits"}},
        )
        self.assertEqual(unmatched, [])

    def test_missing_count_fields_default_to_zero_not_absent(self):
        rows, _ = self.merge([("A", "a")], {"a": {"keep_count": 1}})
        self.assertEqual(rows[0]["deprecate_count"], 0)
        self.assertEqual(rows[0]["unreviewed_count"], 0)

    def test_null_counts_do_not_leak_through(self):
        """COUNT(*) FILTER returns NULL, not 0, in some drivers."""
        rows, _ = self.merge([("A", "a")], {"a": {"keep_count": 1, "unrated": None}})
        self.assertEqual(rows[0]["unrated"], 0)

    def test_no_headings_yields_no_rows(self):
        rows, unmatched = self.merge([], {"a": {"keep_count": 1, "prd_section": "A"}})
        self.assertEqual(rows, [])
        self.assertEqual(unmatched, ["A"])


# ─── C20: review SLA ──────────────────────────────────────────────────────────

class OverdueQueryShapeTests(unittest.TestCase):
    """
    The SQL cannot run here, so pin the properties that would rot silently: only
    unreviewed rows can be overdue, the window is a parameter rather than a stored
    column, and the count is not capped by the listing limit.
    """

    def setUp(self):
        import inspect
        from embeddings import pg_store
        self.overdue = inspect.getsource(pg_store.PGStore.get_overdue_decisions)
        self.count = inspect.getsource(pg_store.PGStore.count_overdue_decisions)

    def test_only_unreviewed_decisions_can_be_overdue(self):
        """A reviewed-and-rejected decision is finished, not late."""
        for src in (self.overdue, self.count):
            self.assertIn("reviewed = FALSE", src)

    def test_the_window_is_a_parameter_not_a_stored_column(self):
        """A generated review_deadline column would need a migration to change the SLA."""
        for src in (self.overdue, self.count):
            self.assertIn("INTERVAL '1 day'", src)
            self.assertNotIn("review_deadline", src)

    def test_oldest_first(self):
        self.assertIn("ORDER BY created_at ASC", self.overdue)

    def test_count_is_not_capped_by_the_listing_limit(self):
        """The backlog size must be true even when the list is truncated."""
        self.assertNotIn("LIMIT", self.count)
        self.assertIn("LIMIT", self.overdue)

    def test_age_is_returned_so_callers_need_no_clock(self):
        self.assertIn("age_days", self.overdue)


class ReviewSlaConfigTests(unittest.TestCase):
    def test_migration_avoids_a_generated_column(self):
        """
        timestamptz + interval is only STABLE, and a generated column needs IMMUTABLE,
        so the column the original plan specified may be rejected outright.
        """
        from pathlib import Path
        sql = (Path(__file__).resolve().parents[2]
               / "init-db" / "09-review-deadline.sql").read_text()
        self.assertNotIn("GENERATED ALWAYS", sql)
        self.assertIn("CREATE INDEX IF NOT EXISTS", sql)

    def test_index_is_partial_on_unreviewed(self):
        from pathlib import Path
        sql = (Path(__file__).resolve().parents[2]
               / "init-db" / "09-review-deadline.sql").read_text()
        self.assertIn("WHERE reviewed = FALSE", sql)


# ─── C26: Confluence webhook ──────────────────────────────────────────────────

class WebhookSignatureTests(unittest.TestCase):
    """
    The endpoint triggers ingestion and analysis, which cost money. Signature
    verification is the entire security boundary.
    """

    SECRET = "test-secret-value"
    BODY = b'{"event":"page_updated","page":{"id":"1234567890"}}'

    def setUp(self):
        import hashlib
        import hmac
        from integrations import webhook
        self.webhook = webhook
        self.digest = hmac.new(
            self.SECRET.encode(), self.BODY, hashlib.sha256).hexdigest()

    def _ok(self, header, body=None, secret=None):
        return self.webhook.signature_ok(
            self.SECRET if secret is None else secret, body or self.BODY, header)

    def test_correct_signature_is_accepted(self):
        self.assertTrue(self._ok(f"sha256={self.digest}"))

    def test_bare_hex_digest_is_accepted(self):
        """Senders differ on whether they prefix the algorithm."""
        self.assertTrue(self._ok(self.digest))

    def test_surrounding_whitespace_is_tolerated(self):
        self.assertTrue(self._ok(f"sha256={self.digest}  "))

    def test_missing_header_is_rejected(self):
        self.assertFalse(self._ok(None))
        self.assertFalse(self._ok(""))

    def test_wrong_signature_is_rejected(self):
        self.assertFalse(self._ok("sha256=" + "0" * 64))

    def test_signature_covers_the_body(self):
        """A tampered payload must not verify against the original digest."""
        tampered = b'{"event":"page_updated","page":{"id":"9999999999"}}'
        self.assertFalse(self._ok(f"sha256={self.digest}", body=tampered))

    def test_empty_secret_never_verifies(self):
        """A misconfiguration must not become an open endpoint."""
        self.assertFalse(self._ok(f"sha256={self.digest}", secret=""))


class WebhookPayloadTests(unittest.TestCase):
    def setUp(self):
        from integrations import webhook
        self.webhook = webhook

    def test_page_change_events_trigger(self):
        for event in ("page_updated", "page_created", "page_restored"):
            with self.subTest(event=event):
                self.assertTrue(self.webhook.should_trigger(event))

    def test_unrelated_events_do_not_trigger(self):
        for event in ("page_viewed", "comment_created", "blog_created",
                      "space_updated", "", None):
            with self.subTest(event=event):
                self.assertFalse(self.webhook.should_trigger(event))

    def test_both_event_key_spellings_are_read(self):
        self.assertEqual(self.webhook.event_name({"event": "page_updated"}), "page_updated")
        self.assertEqual(self.webhook.event_name({"eventType": "page_created"}), "page_created")

    def test_numeric_page_id_is_extracted(self):
        self.assertEqual(
            self.webhook.page_id({"page": {"id": "1234567890"}}), "1234567890")

    def test_content_key_is_also_accepted(self):
        self.assertEqual(
            self.webhook.page_id({"content": {"id": "1234567890"}}), "1234567890")

    def test_non_numeric_page_id_is_refused(self):
        """The id is interpolated into a source_id and used to fetch a page."""
        for bad in ("../../etc/passwd", "1234; DROP", "abc", "", None):
            with self.subTest(value=bad):
                self.assertIsNone(self.webhook.page_id({"page": {"id": bad}}))

    def test_missing_page_object_is_refused(self):
        self.assertIsNone(self.webhook.page_id({}))
        self.assertIsNone(self.webhook.page_id({"page": "not-an-object"}))


class WebhookDebounceTests(unittest.TestCase):
    """Confluence fires once per save, so an editing session arrives as a burst."""

    def setUp(self):
        from integrations.webhook import Debounce
        self.Debounce = Debounce
        self.d = Debounce(window_sec=900)

    def test_first_event_for_a_page_is_accepted(self):
        self.assertEqual(self.d.check("1234567890", now=1000.0), 0.0)

    def test_immediate_repeat_is_debounced(self):
        self.d.check("1234567890", now=1000.0)
        self.assertGreater(self.d.check("1234567890", now=1001.0), 0)

    def test_wait_time_counts_down(self):
        self.d.check("1234567890", now=1000.0)
        self.assertAlmostEqual(self.d.check("1234567890", now=1300.0), 600.0, places=1)

    def test_a_different_page_is_not_debounced(self):
        self.d.check("1234567890", now=1000.0)
        self.assertEqual(self.d.check("1111111111", now=1000.0), 0.0)

    def test_expired_window_lets_the_page_through(self):
        self.d.check("1234567890", now=1000.0)
        self.assertEqual(self.d.check("1234567890", now=1901.0), 0.0)

    def test_zero_window_disables_debouncing(self):
        d = self.Debounce(window_sec=0)
        for _ in range(5):
            self.assertEqual(d.check("1234567890"), 0.0)

    def test_tracking_table_is_bounded(self):
        """The key space is every page anyone edits — unbounded over process life."""
        d = self.Debounce(window_sec=900, max_keys=100)
        for i in range(250):
            d.check(str(i), now=1000.0 + i)
        self.assertLessEqual(len(d), 100)

    def test_the_oldest_key_is_evicted_first(self):
        d = self.Debounce(window_sec=900, max_keys=2)
        d.check("a", now=1000.0)
        d.check("b", now=1001.0)
        d.check("c", now=1002.0)
        self.assertEqual(d.check("a", now=1003.0), 0.0, "'a' should have been evicted")


# ─── C25: judge rubric ────────────────────────────────────────────────────────

class RubricScoreTests(unittest.TestCase):
    """
    The composite the eval reports. A partial grade must never be presented as a
    complete one, and a missing dimension is a fact about the judge — not evidence
    that the decision was bad.
    """

    def setUp(self):
        from eval import llm_judge
        self.j = llm_judge

    FULL = {"correctness": 5, "reasoning": 5, "completeness": 5}

    def test_top_marks_score_one(self):
        self.assertEqual(self.j.rubric_score(self.FULL), 1.0)

    def test_bottom_marks_score_above_zero(self):
        """1 is the floor of the scale, not the absence of a score."""
        score = self.j.rubric_score({"correctness": 1, "reasoning": 1, "completeness": 1})
        self.assertGreater(score, 0.0)
        self.assertLess(score, 0.5)

    def test_composite_is_weighted_not_a_flat_mean(self):
        """A wrong action explained well must score below a right one explained poorly."""
        wrong_but_articulate = self.j.rubric_score(
            {"correctness": 1, "reasoning": 5, "completeness": 5})
        right_but_terse = self.j.rubric_score(
            {"correctness": 5, "reasoning": 1, "completeness": 1})
        self.assertLess(wrong_but_articulate, right_but_terse)

    def test_a_missing_dimension_yields_no_score(self):
        for absent in ("correctness", "reasoning", "completeness"):
            with self.subTest(missing=absent):
                partial = {k: v for k, v in self.FULL.items() if k != absent}
                self.assertIsNone(self.j.rubric_score(partial))

    def test_missing_dimensions_are_named(self):
        self.assertEqual(
            self.j.missing_dimensions({"correctness": 4}),
            ["completeness", "reasoning"],
        )

    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(self.j.RUBRIC_WEIGHTS.values()), 1.0, places=6)

    def test_correctness_outweighs_the_other_dimensions_combined(self):
        """
        Weighing it merely highest is not enough: at 0.50/0.30/0.20 a wrong action
        explained perfectly and a right one explained terribly both score 0.6.
        """
        weights = self.j.RUBRIC_WEIGHTS
        others = sum(w for d, w in weights.items() if d != "correctness")
        self.assertGreater(weights["correctness"], others)


class DimensionCoercionTests(unittest.TestCase):
    def setUp(self):
        from eval.llm_judge import coerce_dimension
        self.coerce = coerce_dimension

    def test_valid_scores_pass_through(self):
        for n in (1, 2, 3, 4, 5):
            self.assertEqual(self.coerce(n), n)

    def test_numeric_strings_are_accepted(self):
        self.assertEqual(self.coerce("4"), 4)

    def test_floats_are_rounded(self):
        self.assertEqual(self.coerce(4.4), 4)
        self.assertEqual(self.coerce(4.6), 5)

    def test_out_of_range_is_clamped_not_discarded(self):
        """A judge answering 7 on a 1-5 scale means 'as high as it goes'."""
        self.assertEqual(self.coerce(7), 5)
        self.assertEqual(self.coerce(0), 1)
        self.assertEqual(self.coerce(-3), 1)

    def test_missing_becomes_none_not_zero(self):
        """
        The regression: int(result.get(field, 0)) made an omitted dimension score 0,
        indistinguishable from a decision the judge rated as terrible.
        """
        self.assertIsNone(self.coerce(None))

    def test_unparseable_values_become_none(self):
        for junk in ("high", "", [], {}, "N/A"):
            with self.subTest(value=junk):
                self.assertIsNone(self.coerce(junk))

    def test_booleans_are_not_scores(self):
        """True would otherwise coerce to 1 and look like a real low score."""
        self.assertIsNone(self.coerce(True))
        self.assertIsNone(self.coerce(False))


class JudgeJsonExtractionTests(unittest.TestCase):
    def setUp(self):
        from eval.llm_judge import _extract_json
        self.extract = _extract_json

    def test_bare_json_is_unchanged(self):
        self.assertEqual(self.extract('{"correctness": 5}'), '{"correctness": 5}')

    def test_fenced_json_is_unwrapped(self):
        self.assertEqual(
            self.extract('```json\n{"correctness": 5}\n```'), '{"correctness": 5}')

    def test_unlabelled_fence_is_unwrapped(self):
        self.assertEqual(self.extract('```\n{"correctness": 5}\n```'), '{"correctness": 5}')

    def test_surrounding_whitespace_is_trimmed(self):
        self.assertEqual(self.extract('  {"a": 1}  '), '{"a": 1}')


def _async_return(value):
    async def _fn(*args, **kwargs):
        return value
    return _fn


if __name__ == "__main__":
    unittest.main()
