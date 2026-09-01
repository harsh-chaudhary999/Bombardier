"""
Confluence PRD ingestor for Phase 2.

Fetches a Confluence page (by page ID or URL) via the Confluence REST API v2,
converts storage-format HTML → plain text, and returns chunks ready for embedding.

Uses the Confluence REST API directly (not via an MCP hop) to avoid timeout issues on large pages.
"""
import os
import re
import logging
from urllib.parse import urlparse

import requests

from observability import trace

from ingestion.confluence_html import storage_to_markdown
from ingestion.doc_classify import classify as _classify_doc

logger = logging.getLogger(__name__)

def _confluence_domain() -> str:
    return os.environ.get("CONFLUENCE_DOMAIN", "").strip()


def _base_url() -> str:
    return f"https://{_confluence_domain()}/wiki/api/v2"


def _auth() -> tuple[str, str]:
    email = os.environ.get("CONFLUENCE_EMAIL", "")
    token = os.environ.get("CONFLUENCE_API_TOKEN", "")
    if not email or not token:
        raise RuntimeError(
            "CONFLUENCE_EMAIL and CONFLUENCE_API_TOKEN must be set in the environment"
        )
    return (email, token)


def _html_to_text(html: str) -> str:
    """
    Convert Confluence storage-format HTML to clean markdown-ish text.

    Delegates to ingestion/confluence_html.py, which preserves the content
    a plain html2text pass drops: code-macro CDATA bodies, image alt text and
    attachment filenames, macro titles, and tables as real markdown pipe tables.
    """
    return storage_to_markdown(html)


# Attachment ingestion. Off by default: it adds a request per page plus a download
# per attachment, and pulls in whatever else is attached to a PRD (mockups,
# exports, screenshots). Turn it on where specs genuinely live in attached
# spreadsheets, which is otherwise content the index cannot see at all.
INGEST_ATTACHMENTS = os.environ.get(
    "QA_CONFLUENCE_INGEST_ATTACHMENTS", "0"
).strip() not in ("", "0", "false", "False")
MAX_ATTACHMENT_BYTES = int(
    os.environ.get("QA_CONFLUENCE_MAX_ATTACHMENT_MB", "10")
) * 1_000_000
_ATTACHMENT_EXTENSIONS = (".xlsx", ".docx", ".pdf", ".md", ".txt", ".csv")

# Child-page ingestion. Off by default: a single-page ingest is a targeted request,
# and following descendants turns it into a crawl. Use a space ingest for whole
# trees; this is for a PRD whose sections were split into child pages.
INGEST_CHILDREN = os.environ.get(
    "QA_CONFLUENCE_INGEST_CHILDREN", "0"
).strip() not in ("", "0", "false", "False")
CHILD_DEPTH = int(os.environ.get("QA_CONFLUENCE_CHILD_DEPTH", "1"))
MAX_CHILD_PAGES = int(os.environ.get("QA_CONFLUENCE_MAX_CHILD_PAGES", "50"))


def list_page_attachments(page_id: str) -> list[dict]:
    """Attachments on a page: [{id, title, media_type, size, download}]."""
    out: list[dict] = []
    cursor = None
    while True:
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        resp = requests.get(
            f"{_base_url()}/pages/{page_id}/attachments",
            params=params, auth=_auth(), timeout=30,
        )
        if resp.status_code == 404:
            return out
        resp.raise_for_status()
        data = resp.json()
        for a in data.get("results", []):
            out.append({
                "id": a.get("id"),
                "title": a.get("title") or "",
                "media_type": a.get("mediaType") or "",
                "size": a.get("fileSize") or 0,
                "download": a.get("downloadLink") or a.get("_links", {}).get("download", ""),
            })
        cursor = (data.get("_links") or {}).get("next")
        if not cursor:
            return out
        # v2 returns a full path with the cursor embedded; extract just the cursor.
        match = re.search(r'cursor=([^&]+)', cursor)
        if not match:
            return out
        cursor = match.group(1)


def _download_attachment(download_link: str) -> bytes:
    """Attachment bodies live outside /api/v2 — the link is site-root relative."""
    url = download_link
    if url.startswith("/"):
        url = f"https://{_confluence_domain()}/wiki{url}"
    resp = requests.get(url, auth=_auth(), timeout=60, allow_redirects=True)
    resp.raise_for_status()
    return resp.content


def fetch_attachment_markdown(page_id: str) -> str:
    """
    Convert a page's supported attachments to markdown, each as its own H2 section.

    Failures are per-attachment and non-fatal: one unreadable file must not lose
    the page it is attached to.

    Staleness caveat: attachment content is folded into the page's document, and
    incremental refresh keys off the PAGE version. If a new revision of an
    attachment is uploaded without the page body changing, the crawl may consider
    the page unchanged and keep serving the old attachment text. Force a re-ingest
    of that page when attachments are updated in place.
    """
    from ingestion.file_ingestor import convert_file_to_markdown

    try:
        attachments = list_page_attachments(page_id)
    except Exception as exc:
        logger.warning("Could not list attachments for page %s: %s", page_id, exc)
        return ""

    parts: list[str] = []
    for att in attachments:
        name = att["title"]
        ext = os.path.splitext(name.lower())[1]
        if ext not in _ATTACHMENT_EXTENSIONS:
            continue
        if att["size"] and att["size"] > MAX_ATTACHMENT_BYTES:
            logger.warning(
                "Skipping attachment %r on page %s: %.1f MB exceeds the %.0f MB cap",
                name, page_id, att["size"] / 1e6, MAX_ATTACHMENT_BYTES / 1e6,
            )
            continue
        if not att["download"]:
            continue
        try:
            body = convert_file_to_markdown(name, _download_attachment(att["download"]))
        except Exception as exc:
            logger.warning("Skipping attachment %r on page %s: %s", name, page_id, exc)
            continue
        # The file converter emits its own '# <name>' title; demote it to a section
        # of the host page so the page keeps one heading hierarchy.
        body = re.sub(r'^#\s+', '## Attachment: ', body, count=1).strip()
        # An attachment that converts to nothing but its own title (empty sheet,
        # image-only PDF) would index as a bare heading — noise with no content.
        if len(body.split("\n", 1)) < 2 or not body.split("\n", 1)[1].strip():
            logger.info("Attachment %r on page %s has no extractable text", name, page_id)
            continue
        parts.append(body)

    if parts:
        logger.info("Page %s: %s attachment(s) ingested", page_id, len(parts))
    return "\n\n".join(parts)


def _extract_page_id(source: str) -> str:
    """
    Accept either a bare page ID ('1234567890') or a full Confluence URL.
    Confluence Cloud URLs look like:
      https://domain.atlassian.net/wiki/spaces/SPACE/pages/1234567890/Page+Title
    """
    source = source.strip()
    if source.isdigit():
        return source
    parsed = urlparse(source)
    # Path segment after /pages/
    match = re.search(r'/pages/(\d+)', parsed.path)
    if match:
        return match.group(1)
    raise ValueError(f"Cannot extract Confluence page ID from: {source!r}")


def fetch_confluence_page(source: str) -> dict:
    """
    Fetch a Confluence page and return:
      {"page_id", "title", "text", "version", "space_key"}

    `source` can be a page ID or full URL.
    """
    page_id = _extract_page_id(source)

    resp = requests.get(
        f"{_base_url()}/pages/{page_id}",
        params={"body-format": "storage", "expand": "version,space"},
        auth=_auth(),
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()

    title = data.get("title", "Untitled")
    version = str(data.get("version", {}).get("number", "1"))
    space_key = data.get("spaceKey") or data.get("space", {}).get("key", "")
    html_body = data.get("body", {}).get("storage", {}).get("value", "")

    if not html_body:
        # v2 API sometimes needs explicit body expansion
        resp2 = requests.get(
            f"{_base_url()}/pages/{page_id}",
            params={"body-format": "storage"},
            auth=_auth(),
            timeout=30,
        )
        resp2.raise_for_status()
        html_body = resp2.json().get("body", {}).get("storage", {}).get("value", "")

    text = _html_to_text(html_body) if html_body else ""
    if not text:
        logger.warning(f"Confluence page {page_id} ({title!r}) has empty body")

    return {
        "page_id": page_id,
        "title": title,
        "text": text,
        "version": version,
        "space_key": space_key,
    }


def _ingest_one_page(source: str, embed_fn=None) -> list[dict]:
    """Fetch, convert and chunk a single page. No child traversal — see below."""
    domain = _confluence_domain()
    from ingestion.chunker import chunk_document

    page = fetch_confluence_page(source)
    page_id = page["page_id"]
    source_id = f"confluence:{page_id}"
    doc_url = f"https://{domain}/wiki/spaces/{page['space_key']}/pages/{page_id}" \
              if page["space_key"] else \
              f"https://{domain}/wiki/pages/viewpage.action?pageId={page_id}"

    full_text = f"# {page['title']}\n\n{page['text']}"

    if INGEST_ATTACHMENTS:
        attachments = fetch_attachment_markdown(page_id)
        if attachments:
            full_text = f"{full_text}\n\n{attachments}"

    # Chunk the document properly instead of returning as one oversized block
    raw_chunks = chunk_document(full_text, source_id, embed_fn=embed_fn)

    logger.info(
        f"Confluence page {page_id} ({page['title']!r}): "
        f"{len(raw_chunks)} chunks, version={page['version']}"
    )

    return [
        {
            "source_id":       source_id,
            "source_type":     "confluence",
            "source_version":  page["version"],
            "doc_title":       page["title"],
            "doc_url":         doc_url,
            "section_heading": c.get("section_heading", page["title"]),
            "chunk_text":      c["chunk_text"],
            "parent_text":     c.get("parent_text"),
            "chunk_type":      c.get("chunk_type"),
            "doc_type":        _classify_doc(page["title"]),
            "chunk_index":     i,
        }
        for i, c in enumerate(raw_chunks)
    ]


def _child_pages(page_id: str) -> list[dict]:
    """
    Descendants of a page, bounded by depth and count.

    Reuses the space ingestor's traversal (lazily, to keep the import light) and
    then caps it: a targeted single-page ingest must not silently become an
    unbounded crawl because someone pointed it at a section root.
    """
    from ingestion.confluence_space_ingestor import list_child_pages

    try:
        children = list_child_pages(page_id, max_depth=CHILD_DEPTH)
    except Exception as exc:
        logger.warning("Could not list child pages of %s: %s", page_id, exc)
        return []

    if len(children) > MAX_CHILD_PAGES:
        logger.warning(
            "Page %s has %s descendants; ingesting the first %s "
            "(raise QA_CONFLUENCE_MAX_CHILD_PAGES, or use a space ingest instead)",
            page_id, len(children), MAX_CHILD_PAGES,
        )
        children = children[:MAX_CHILD_PAGES]
    return children


def ingest_confluence_page(source: str, embed_fn=None) -> list[dict]:
    """
    Full ingestion flow for one Confluence page.

    Returns list of chunk dicts ready for embed + upsert:
      [{source_id, source_type, source_version, section_heading, chunk_text, chunk_index}, ...]

    With QA_CONFLUENCE_INGEST_CHILDREN=1 the page's descendants are ingested too,
    each as its OWN document with its own source_id and source_version — not folded
    into the parent. Folding them in would freeze the parent's version, so an edit
    to a child would never trigger a re-ingest on the next incremental refresh.
    The caller groups by source_id before upserting (prd_pipeline._upsert_by_source).
    """
    if not _confluence_domain():
        raise RuntimeError(
            "CONFLUENCE_DOMAIN must be set in the environment to ingest Confluence pages"
        )

    chunks = _ingest_one_page(source, embed_fn=embed_fn)

    if INGEST_CHILDREN:
        children = _child_pages(_extract_page_id(source))
        for child in children:
            try:
                chunks.extend(_ingest_one_page(str(child["id"]), embed_fn=embed_fn))
            except Exception as exc:
                # One unreadable child must not cost us the parent or its siblings.
                logger.warning("Skipping child page %s: %s", child.get("id"), exc)
        if children:
            logger.info("Page %s: ingested %s child page(s)",
                        _extract_page_id(source), len(children))

    return chunks
