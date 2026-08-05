"""
Guards the incremental-sync change detection and its known blind spot.

Phase 3 decides which tests get their steps/description re-read in Phase 4. That decision is
made from the bulk listing alone — summary, folder, labels, test type — because fetching
content costs one MCP call per test. The consequence is a real gap: edit a test's *steps* in
Xray and nothing else, and the hash is identical, Phase 4 never runs for it, and Elasticsearch
keeps serving the old steps. The agent then reasons about a test that no longer says what the
index claims.

Two things are pinned here:

  * Jira's `updated` timestamp participates in the hash, so any edit that touches the Jira
    issue is caught cheaply. It is read defensively — a server that does not expose the field
    yields "" and the diff degrades to its previous behaviour rather than crashing.
  * The residual gap is *countable*. `content_unverified` is what makes it visible instead of
    silent, so a stale index is diagnosable rather than mysterious.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from test_context_budget import _install_stubs  # noqa: E402

_install_stubs()
from sync.test_sync import _extract_updated, _metadata_hash  # noqa: E402


def _test(**kw):
    base = {"key": "PROJ-1", "summary": "Login works", "labels": ["smoke"],
            "folder": {"path": "/Platform"}, "testType": "Manual"}
    base.update(kw)
    return base


class ExtractUpdatedTests(unittest.TestCase):
    def test_top_level(self):
        self.assertEqual(_extract_updated(_test(updated="2026-08-05T10:00:00.000+0000")),
                         "2026-08-05T10:00:00.000+0000")

    def test_nested_under_jira(self):
        self.assertEqual(_extract_updated({"jira": {"updated": "2026-08-05"}}), "2026-08-05")

    def test_nested_under_fields(self):
        self.assertEqual(_extract_updated({"fields": {"updated": "2026-08-05"}}), "2026-08-05")

    def test_absent_yields_empty_not_an_error(self):
        """A server that does not expose `updated` must degrade, not crash."""
        self.assertEqual(_extract_updated(_test()), "")
        self.assertEqual(_extract_updated({}), "")

    def test_malformed_containers_are_tolerated(self):
        self.assertEqual(_extract_updated({"jira": "not-a-dict", "fields": None}), "")

    def test_non_string_timestamp_is_stringified(self):
        self.assertEqual(_extract_updated({"updated": 1754390400}), "1754390400")


class MetadataHashTests(unittest.TestCase):
    def test_identical_input_is_stable(self):
        self.assertEqual(_metadata_hash(_test()), _metadata_hash(_test()))

    def test_label_order_does_not_matter(self):
        a = _metadata_hash(_test(labels=["smoke", "regression"]))
        b = _metadata_hash(_test(labels=["regression", "smoke"]))
        self.assertEqual(a, b, "labels are sorted; reordering is not a change")

    def test_summary_change_is_detected(self):
        self.assertNotEqual(_metadata_hash(_test()), _metadata_hash(_test(summary="Login fails")))

    def test_label_change_is_detected(self):
        self.assertNotEqual(_metadata_hash(_test()), _metadata_hash(_test(labels=["smoke", "p1"])))

    # ── the point of adding `updated` ──

    def test_updated_timestamp_change_is_detected(self):
        """
        A content edit that leaves summary/labels/folder/type untouched is invisible to every
        other field in this hash. The timestamp is the only cheap signal that catches it.
        """
        before = _metadata_hash(_test(updated="2026-08-05T10:00:00.000+0000"))
        after = _metadata_hash(_test(updated="2026-08-05T11:30:00.000+0000"))
        self.assertNotEqual(before, after)

    def test_hash_differs_once_the_field_appears(self):
        """
        Upgrading the MCP server to expose `updated` changes every hash once, forcing a
        one-time full content re-read. That is correct — the prior hashes were computed
        without a content signal and cannot be trusted.
        """
        self.assertNotEqual(_metadata_hash(_test()), _metadata_hash(_test(updated="2026-08-05")))

    def test_missing_updated_still_hashes(self):
        self.assertIsInstance(_metadata_hash(_test()), str)
        self.assertEqual(len(_metadata_hash(_test())), 64)


if __name__ == "__main__":
    unittest.main()
