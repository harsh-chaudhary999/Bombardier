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
from html2text import HTML2Text

from observability import trace

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
    """Convert Confluence storage-format HTML to clean markdown-ish text."""
    # Strip ac:parameter elements AND their content (e.g. <ac:parameter ac:name="class">wide760</ac:parameter>)
    # Must be done before stripping other ac: tags so the content "wide760" doesn't leak as text
    html = re.sub(r'<ac:parameter[^>]*>.*?</ac:parameter>', '', html, flags=re.DOTALL)
    # Strip remaining Confluence macro tags (keep body content inside ac:rich-text-body etc.)
    html = re.sub(r'</?ac:[^>]+>', '', html)
    # Strip width/style attributes that leak as "wide760", "fixed-table", etc.
    html = re.sub(r'\s+(?:width|style|class|data-[a-z-]+)="[^"]*"', '', html)

    h = HTML2Text()
    h.ignore_links = False
    h.ignore_images = True
    h.ignore_tables = False
    h.body_width = 0          # don't wrap lines
    h.unicode_snob = True
    text = h.handle(html)
    # Remove leftover "wide\d+" / "fixed\w*" tokens (Confluence table width artefacts)
    text = re.sub(r'\bwide\d+', '', text)    # no trailing \b — Atlassian storage artifacts like "wide760…" must match
    text = re.sub(r'\bfixed(?:-table|-layout|-width|Width)\b', '', text, flags=re.IGNORECASE)
    # Collapse excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


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


def ingest_confluence_page(source: str, embed_fn=None) -> list[dict]:
    """
    Full ingestion flow for one Confluence page.

    Returns list of chunk dicts ready for embed + upsert:
      [{source_id, source_type, source_version, section_heading, chunk_text, chunk_index}, ...]
    """
    domain = _confluence_domain()
    if not domain:
        raise RuntimeError(
            "CONFLUENCE_DOMAIN must be set in the environment to ingest Confluence pages"
        )
    from ingestion.chunker import chunk_document

    page = fetch_confluence_page(source)
    page_id = page["page_id"]
    source_id = f"confluence:{page_id}"
    doc_url = f"https://{domain}/wiki/spaces/{page['space_key']}/pages/{page_id}" \
              if page["space_key"] else \
              f"https://{domain}/wiki/pages/viewpage.action?pageId={page_id}"

    full_text = f"# {page['title']}\n\n{page['text']}"

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
            "doc_type":        _classify_doc(page["title"]),
            "chunk_index":     i,
        }
        for i, c in enumerate(raw_chunks)
    ]
