"""
Guards the incremental-refresh contract for Confluence space/site ingest.

The property under test: a page whose live Confluence version matches the indexed
source_version must be skipped WITHOUT fetching its body. That is what makes a
recursive site crawl re-runnable — otherwise every refresh re-downloads the whole
site (the old behaviour skipped by source_id, so edits were never picked up at all).

Runs under the documented host-side command:
    cd qa-agent && PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'
"""
import sys
import types
import unittest


def _install_stubs() -> None:
    if "html2text" not in sys.modules:
        mod = types.ModuleType("html2text")

        class _H2T:
            ignore_links = ignore_images = ignore_tables = False
            body_width = 0
            unicode_snob = True

            def handle(self, html):
                return html

        mod.HTML2Text = _H2T
        sys.modules["html2text"] = mod


_install_stubs()
from ingestion import confluence_space_ingestor as C  # noqa: E402


class CrawlSpaceIncrementalTests(unittest.TestCase):
    """crawl_space's version filter — the page bodies it does and does not fetch."""

    PAGES = [
        {"id": "100", "title": "Checkout PRD", "version": 7},
        {"id": "200", "title": "Billing PRD", "version": 2},
        {"id": "300", "title": "Search PRD", "version": 1},
    ]

    def setUp(self):
        self.fetched: list[str] = []
        self._orig_list = C.list_space_pages
        self._orig_fetch = C.fetch_and_chunk_page

        C.list_space_pages = lambda space_key, title_filter="": [
            p for p in self.PAGES
            if not title_filter or title_filter.lower() in p["title"].lower()
        ]

        def _fake_fetch(page_meta, space_key=""):
            self.fetched.append(page_meta["id"])
            return [{
                "source_id": f"confluence:{page_meta['id']}",
                "source_type": "confluence",
                "source_version": str(page_meta["version"]),
                "doc_title": page_meta["title"],
                "doc_url": "http://example/x",
                "section_heading": page_meta["title"],
                "chunk_text": "body text",
                "parent_text": None,
                "chunk_index": 0,
            }]

        C.fetch_and_chunk_page = _fake_fetch

    def tearDown(self):
        C.list_space_pages = self._orig_list
        C.fetch_and_chunk_page = self._orig_fetch

    def _crawl(self, indexed_versions):
        return C.crawl_space(
            space_key="DOCS", max_workers=2, indexed_versions=indexed_versions
        )

    def test_no_baseline_fetches_everything(self):
        """indexed_versions=None is the force/full-crawl path."""
        chunks, skipped, unchanged = self._crawl(None)
        self.assertEqual(sorted(self.fetched), ["100", "200", "300"])
        self.assertEqual(len(chunks), 3)
        self.assertEqual((skipped, unchanged), ([], []))

    def test_matching_version_is_not_fetched(self):
        """The efficiency claim: unchanged pages cost zero body requests."""
        chunks, _skipped, unchanged = self._crawl({
            "confluence:100": "7",
            "confluence:200": "2",
            "confluence:300": "1",
        })
        self.assertEqual(self.fetched, [], "unchanged pages must not be re-fetched")
        self.assertEqual(chunks, [])
        self.assertCountEqual(unchanged, ["100", "200", "300"])

    def test_bumped_version_is_refetched(self):
        """A Confluence edit bumps version -> page must be re-ingested."""
        chunks, _skipped, unchanged = self._crawl({
            "confluence:100": "6",   # stale: live is 7
            "confluence:200": "2",   # current
            "confluence:300": "1",   # current
        })
        self.assertEqual(self.fetched, ["100"])
        self.assertEqual(len(chunks), 1)
        self.assertCountEqual(unchanged, ["200", "300"])

    def test_unknown_version_is_refetched(self):
        """Empty version = indexed before source_version existed. Re-ingest, don't trust."""
        chunks, _skipped, unchanged = self._crawl({
            "confluence:100": "",
            "confluence:200": "2",
        })
        # 100 has an unknown version, 300 was never indexed -> both re-fetched.
        self.assertCountEqual(self.fetched, ["100", "300"])
        self.assertEqual(len(chunks), 2)
        self.assertEqual(unchanged, ["200"])

    def test_never_indexed_pages_are_fetched(self):
        chunks, _skipped, unchanged = self._crawl({"confluence:200": "2"})
        self.assertCountEqual(self.fetched, ["100", "300"])
        self.assertEqual(len(chunks), 2)
        self.assertEqual(unchanged, ["200"])

    def test_version_compared_as_string_not_int(self):
        """Stored versions are keyword strings; live versions are ints. Must still match."""
        _chunks, _skipped, unchanged = self._crawl({"confluence:100": "7"})
        self.assertIn("100", unchanged)

    def test_title_filter_still_applies_with_incremental(self):
        chunks, _skipped, unchanged = C.crawl_space(
            space_key="DOCS",
            title_filter="billing",
            max_workers=2,
            indexed_versions={"confluence:100": "7"},
        )
        # Only Billing matches the filter, and it is not in the version map.
        self.assertEqual(self.fetched, ["200"])
        self.assertEqual(len(chunks), 1)
        self.assertEqual(unchanged, [])

    def test_all_unchanged_returns_early_without_thread_pool(self):
        """Fully-current space: returns the 3-tuple contract, no crawl work."""
        result = self._crawl({f"confluence:{p['id']}": str(p["version"]) for p in self.PAGES})
        self.assertEqual(len(result), 3)
        chunks, skipped, unchanged = result
        self.assertEqual((chunks, skipped), ([], []))
        self.assertEqual(len(unchanged), 3)


class ListSpacesTests(unittest.TestCase):
    """Space selection for site-wide crawls."""

    SITE = [
        {"key": "DOCS", "name": "Docs", "type": "global"},
        {"key": "PLAT", "name": "Platform", "type": "global"},
        {"key": "~alice", "name": "Alice", "type": "personal"},
    ]

    def setUp(self):
        self._orig_get = C.requests.get
        self._orig_auth = C._auth
        C._auth = lambda: ("e", "t")
        captured = self.captured = {}

        class _Resp:
            def raise_for_status(self_inner):
                pass

            def json(self_inner):
                return {"results": ListSpacesTests.SITE, "_links": {}}

        def _fake_get(url, auth=None, params=None, timeout=None):
            captured["params"] = params or {}
            return _Resp()

        C.requests.get = _fake_get

    def tearDown(self):
        C.requests.get = self._orig_get
        C._auth = self._orig_auth

    def test_personal_spaces_excluded_by_default(self):
        keys = [s["key"] for s in C.list_spaces()]
        self.assertEqual(keys, ["DOCS", "PLAT"])
        # Filtered server-side too, so we don't page through personal spaces needlessly.
        self.assertEqual(self.captured["params"].get("type"), "global")
        self.assertEqual(self.captured["params"].get("status"), "current")

    def test_personal_spaces_included_on_request(self):
        keys = [s["key"] for s in C.list_spaces(include_personal=True)]
        self.assertIn("~alice", keys)
        self.assertNotIn("type", self.captured["params"])

    def test_key_filter_is_an_allowlist(self):
        keys = [s["key"] for s in C.list_spaces(key_filter="plat")]
        self.assertEqual(keys, ["PLAT"], "filter must be case-insensitive")

    def test_key_filter_ignores_blanks_and_spacing(self):
        keys = [s["key"] for s in C.list_spaces(key_filter=" DOCS , , PLAT ")]
        self.assertEqual(keys, ["DOCS", "PLAT"])

    def test_results_sorted_for_deterministic_run_order(self):
        keys = [s["key"] for s in C.list_spaces()]
        self.assertEqual(keys, sorted(keys))

    def test_archived_flag_drops_status_param(self):
        C.list_spaces(include_archived=True)
        self.assertNotIn("status", self.captured["params"])


if __name__ == "__main__":
    unittest.main()
