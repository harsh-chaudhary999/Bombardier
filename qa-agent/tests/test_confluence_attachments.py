"""
Confluence attachment ingestion — content the index otherwise cannot see at all.

Specs frequently live in a spreadsheet attached to the PRD rather than in the
page body. This is opt-in (QA_CONFLUENCE_INGEST_ATTACHMENTS) because it costs a
request per page plus a download per attachment, and pulls in whatever else is
attached — mockups, exports, screenshots.

The property under test: one unreadable or oversized attachment must never cost
us the page it is attached to.

Runs under the documented host-side command:
    cd qa-agent && PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'
"""
import unittest

from tests import stubs

stubs.install_html2text()
stubs.install_chunker_deps()          # child-page ingest runs the real chunker

from ingestion import confluence_ingestor as C  # noqa: E402


class _Resp:
    def __init__(self, payload=None, content=b"", status=200):
        self._payload = payload or {}
        self.content = content
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class AttachmentTests(unittest.TestCase):
    ATTACHMENTS = {
        "results": [
            {"id": "a1", "title": "requirements.xlsx", "fileSize": 1000,
             "downloadLink": "/download/attachments/1/requirements.xlsx"},
            {"id": "a2", "title": "mockup.png", "fileSize": 500,
             "downloadLink": "/download/attachments/1/mockup.png"},
            {"id": "a3", "title": "huge.pdf", "fileSize": 999_000_000,
             "downloadLink": "/download/attachments/1/huge.pdf"},
            {"id": "a4", "title": "notes.txt", "fileSize": 40,
             "downloadLink": "/download/attachments/1/notes.txt"},
        ],
        "_links": {},
    }

    def setUp(self):
        self.downloads: list[str] = []
        self._orig_get = C.requests.get
        self._orig_auth = C._auth
        self._orig_domain = C._confluence_domain
        C._auth = lambda: ("e", "t")
        C._confluence_domain = lambda: "example.invalid"

        def _fake_get(url, params=None, auth=None, timeout=None, allow_redirects=None):
            # Match the listing endpoint precisely: download URLs also contain
            # "/attachments", and a loose check makes every download return the
            # listing body instead of the file.
            if "/api/v2/pages/" in url and url.endswith("/attachments"):
                return _Resp(payload=self.ATTACHMENTS)
            self.downloads.append(url)
            if "notes.txt" in url:
                return _Resp(content=b"Attached note body.")
            if "requirements.xlsx" in url:
                raise RuntimeError("corrupt workbook")
            return _Resp(content=b"")

        C.requests.get = _fake_get

    def tearDown(self):
        C.requests.get = self._orig_get
        C._auth = self._orig_auth
        C._confluence_domain = self._orig_domain

    def test_only_supported_extensions_are_downloaded(self):
        C.fetch_attachment_markdown("1")
        self.assertTrue(all("mockup.png" not in u for u in self.downloads),
                        "images carry no extractable text")

    def test_oversized_attachment_is_skipped_without_downloading(self):
        C.fetch_attachment_markdown("1")
        self.assertTrue(all("huge.pdf" not in u for u in self.downloads))

    def test_a_failing_attachment_does_not_lose_the_others(self):
        out = C.fetch_attachment_markdown("1")
        self.assertIn("Attached note body.", out,
                      "a corrupt xlsx must not cost us the readable .txt")

    def test_attachment_title_is_demoted_to_a_section(self):
        out = C.fetch_attachment_markdown("1")
        self.assertIn("## Attachment: notes", out)
        self.assertFalse(out.startswith("# "), "must not introduce a second H1")

    def test_listing_failure_is_not_fatal(self):
        def _boom(*a, **k):
            raise RuntimeError("attachments endpoint down")

        C.requests.get = _boom
        self.assertEqual(C.fetch_attachment_markdown("1"), "")

    def test_missing_attachments_endpoint_returns_empty(self):
        C.requests.get = lambda *a, **k: _Resp(status=404)
        self.assertEqual(C.list_page_attachments("1"), [])

    def test_attachment_with_no_extractable_text_is_dropped(self):
        """A bare '## Attachment: x' heading is noise, not content."""
        self.ATTACHMENTS = {
            "results": [{"id": "e", "title": "empty.txt", "fileSize": 0,
                         "downloadLink": "/download/attachments/1/empty.txt"}],
            "_links": {},
        }
        C.requests.get = lambda url, **k: (
            _Resp(payload=self.ATTACHMENTS)
            if "/api/v2/pages/" in url and url.endswith("/attachments")
            else _Resp(content=b"   ")
        )
        self.assertEqual(C.fetch_attachment_markdown("1"), "")

    def test_download_url_is_built_from_the_site_root(self):
        C.fetch_attachment_markdown("1")
        self.assertTrue(
            all(u.startswith("https://example.invalid/wiki/") for u in self.downloads),
            self.downloads,
        )


class AttachmentGatingTests(unittest.TestCase):
    """Default off: existing deployments see no change in requests or index volume."""

    def test_disabled_by_default(self):
        self.assertFalse(C.INGEST_ATTACHMENTS)

    def test_supported_extensions_exclude_images(self):
        for ext in (".png", ".jpg", ".gif", ".svg", ".zip"):
            self.assertNotIn(ext, C._ATTACHMENT_EXTENSIONS)

    def test_supported_extensions_cover_the_document_formats(self):
        for ext in (".xlsx", ".docx", ".pdf", ".md", ".txt"):
            self.assertIn(ext, C._ATTACHMENT_EXTENSIONS)


class ChildPageTests(unittest.TestCase):
    """
    Child pages are ingested as their OWN documents, never folded into the parent.

    Folding them in would freeze the parent's source_version, so an edit to a child
    would never trigger a re-ingest on the next incremental refresh — the page
    would silently go stale forever.
    """

    # Bodies are deliberately several words long. The chunker drops chunks under
    # three tokens, and test modules in this suite disagree about what a token is
    # (characters vs whitespace) — a terse fixture passes alone and vanishes in
    # the full run. See tests/stubs.py.
    PAGES = {
        "1": {"title": "Parent PRD", "version": 3,
              "body": "Parent page body describing the overall requirement."},
        "2": {"title": "Child A", "version": 7,
              "body": "Child A body describing the first sub requirement."},
        "3": {"title": "Child B", "version": 1,
              "body": "Child B body describing the second sub requirement."},
    }

    def setUp(self):
        self._orig = {
            "children": C.INGEST_CHILDREN,
            "attachments": C.INGEST_ATTACHMENTS,
            "fetch": C.fetch_confluence_page,
            "domain": C._confluence_domain,
            "list": None,
        }
        C.INGEST_CHILDREN = True
        C.INGEST_ATTACHMENTS = False
        C._confluence_domain = lambda: "example.invalid"

        def _fake_fetch(source):
            page = self.PAGES[str(source)]
            return {"page_id": str(source), "title": page["title"],
                    "text": page["body"], "version": str(page["version"]),
                    "space_key": "DOCS"}

        C.fetch_confluence_page = _fake_fetch
        C._child_pages = lambda page_id: [
            {"id": "2", "title": "Child A", "version": 7},
            {"id": "3", "title": "Child B", "version": 1},
        ]

    def tearDown(self):
        C.INGEST_CHILDREN = self._orig["children"]
        C.INGEST_ATTACHMENTS = self._orig["attachments"]
        C.fetch_confluence_page = self._orig["fetch"]
        C._confluence_domain = self._orig["domain"]

    def _ingest(self):
        return C.ingest_confluence_page("1")

    def test_children_are_separate_documents(self):
        ids = {c["source_id"] for c in self._ingest()}
        self.assertEqual(ids, {"confluence:1", "confluence:2", "confluence:3"})

    def test_each_document_keeps_its_own_version(self):
        versions = {c["source_id"]: c["source_version"] for c in self._ingest()}
        self.assertEqual(versions["confluence:1"], "3")
        self.assertEqual(versions["confluence:2"], "7")
        self.assertEqual(versions["confluence:3"], "1")

    def test_chunk_index_restarts_per_document(self):
        """Each document is upserted separately; indexes must not run on."""
        chunks = self._ingest()
        for source_id in {c["source_id"] for c in chunks}:
            indexes = [c["chunk_index"] for c in chunks if c["source_id"] == source_id]
            self.assertEqual(indexes, list(range(len(indexes))), source_id)

    def test_child_titles_are_carried(self):
        titles = {c["doc_title"] for c in self._ingest()}
        self.assertEqual(titles, {"Parent PRD", "Child A", "Child B"})

    def test_an_unreadable_child_does_not_lose_the_parent(self):
        def _flaky(source):
            if str(source) == "2":
                raise RuntimeError("child fetch failed")
            page = self.PAGES[str(source)]
            return {"page_id": str(source), "title": page["title"],
                    "text": page["body"], "version": str(page["version"]),
                    "space_key": "DOCS"}

        C.fetch_confluence_page = _flaky
        ids = {c["source_id"] for c in self._ingest()}
        self.assertIn("confluence:1", ids)
        self.assertIn("confluence:3", ids)
        self.assertNotIn("confluence:2", ids)

    def test_disabled_by_default_ingests_only_the_page(self):
        C.INGEST_CHILDREN = False
        ids = {c["source_id"] for c in self._ingest()}
        self.assertEqual(ids, {"confluence:1"})

    def test_children_are_not_followed_recursively(self):
        """Descendants come from one bounded traversal, not a per-child expansion."""
        calls = []
        original = C._child_pages
        C._child_pages = lambda pid: calls.append(pid) or original(pid)
        try:
            self._ingest()
        finally:
            C._child_pages = original
        self.assertEqual(calls, ["1"], "only the root may be expanded")


class SpaceCrawlParityTests(unittest.TestCase):
    """
    A space crawl must honour the same attachment flag as single-page ingest.

    Silently skipping attachments on one of the two paths is worse than not having
    the feature: the flag is on, and which pages get attachment content depends on
    how they happened to be ingested.
    """

    def setUp(self):
        from ingestion import confluence_space_ingestor as S
        self.S = S
        self._orig_flag = C.INGEST_ATTACHMENTS
        self._orig_fetch_body = S._fetch_page_body
        self._orig_attach = C.fetch_attachment_markdown
        self._orig_domain = S._confluence_domain

        S._confluence_domain = lambda: "example.invalid"
        S._fetch_page_body = lambda pid: (pid, "Page Title", "Page body text here.")
        C.fetch_attachment_markdown = lambda pid: (
            "## Attachment: spec\n\n| Field | Rule |\n| --- | --- |\n| field_a | rule_a |"
        )

    def tearDown(self):
        C.INGEST_ATTACHMENTS = self._orig_flag
        self.S._fetch_page_body = self._orig_fetch_body
        C.fetch_attachment_markdown = self._orig_attach
        self.S._confluence_domain = self._orig_domain

    def _crawl_one(self):
        return self.S.fetch_and_chunk_page({"id": "9", "version": 2}, space_key="DOCS")

    def test_attachments_are_included_when_the_flag_is_on(self):
        C.INGEST_ATTACHMENTS = True
        text = "\n".join(c["chunk_text"] for c in self._crawl_one())
        self.assertIn("| field_a | rule_a |", text)

    def test_attachments_are_excluded_when_the_flag_is_off(self):
        C.INGEST_ATTACHMENTS = False
        text = "\n".join(c["chunk_text"] for c in self._crawl_one())
        self.assertNotIn("field_a", text)
        self.assertIn("Page body text here.", text, "the page itself is unaffected")

    def test_page_identity_is_unchanged_by_attachments(self):
        C.INGEST_ATTACHMENTS = True
        chunks = self._crawl_one()
        self.assertTrue(all(c["source_id"] == "confluence:9" for c in chunks))
        self.assertTrue(all(c["source_version"] == "2" for c in chunks))


class ChildPageGatingTests(unittest.TestCase):
    def test_disabled_by_default(self):
        self.assertFalse(C.INGEST_CHILDREN)

    def test_traversal_is_bounded(self):
        self.assertGreaterEqual(C.MAX_CHILD_PAGES, 1)
        self.assertGreaterEqual(C.CHILD_DEPTH, 1)


if __name__ == "__main__":
    unittest.main()
