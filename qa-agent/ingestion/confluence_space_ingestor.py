"""
Confluence Space bulk ingestor for Phase 2.

Crawls all pages in a Confluence space (or under a parent page) and ingests
each one into Elasticsearch. Designed for the one-time bulk vectorisation use case.

Uses Confluence REST API v1 (better pagination support than v2 for listing).
Each page is individually chunked and embedded — same pipeline as single-page ingest.
"""
import os
import re
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin

import requests

from ingestion.confluence_html import storage_to_markdown
from ingestion.doc_classify import classify as _classify_doc

logger = logging.getLogger(__name__)

def _confluence_domain() -> str:
    return os.environ.get("CONFLUENCE_DOMAIN", "").strip()


def _base_v1() -> str:
    return f"https://{_confluence_domain()}/wiki/rest/api"


def _base_v2() -> str:
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
    """Same conversion as single-page ingest — see ingestion/confluence_html.py."""
    return storage_to_markdown(html)


# ─── Space listing (site-wide crawl) ──────────────────────────────────────────

def list_spaces(
    include_personal: bool = False,
    include_archived: bool = False,
    key_filter: str = "",
) -> list[dict]:
    """
    Enumerate spaces on the Confluence site — the entry point for a site-wide ingest.

    Personal spaces are excluded by default: they are drafts, scratch notes and 1:1s, and
    indexing them mostly adds retrieval noise to a QA context store.

    Args:
        include_personal: include personal spaces (key starts with '~')
        include_archived: include archived spaces
        key_filter:       comma-separated allowlist of space keys (case-insensitive).
                          Empty = every space that passes the flags above.

    Returns:
        list of {key, name, type} dicts, sorted by key
    """
    allow = {k.strip().upper() for k in key_filter.split(",") if k.strip()}
    spaces: list[dict] = []
    start = 0
    limit = 200

    while True:
        params: dict = {"limit": limit, "start": start}
        if not include_personal:
            params["type"] = "global"
        if not include_archived:
            params["status"] = "current"

        for attempt in range(4):
            try:
                resp = requests.get(
                    f"{_base_v1()}/space", auth=_auth(), params=params, timeout=30
                )
                resp.raise_for_status()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))

        data = resp.json()
        results = data.get("results", [])
        for s in results:
            key = s.get("key") or ""
            if not key:
                continue
            # Belt-and-braces: honour the flag even if the API ignores type=global.
            if not include_personal and key.startswith("~"):
                continue
            if allow and key.upper() not in allow:
                continue
            spaces.append({
                "key": key,
                "name": s.get("name") or key,
                "type": s.get("type") or "global",
            })

        if not data.get("_links", {}).get("next"):
            break
        start += len(results)

    spaces.sort(key=lambda s: s["key"])
    logger.info(
        "Confluence site: %s space(s) selected (personal=%s archived=%s filter=%r)",
        len(spaces), include_personal, include_archived, key_filter,
    )
    return spaces


# ─── Page listing ──────────────────────────────────────────────────────────────

def list_space_pages(space_key: str, title_filter: str = "") -> list[dict]:
    """
    Return all pages in a Confluence space.

    Args:
        space_key:    Confluence space key (e.g. 'DOCS')
        title_filter: optional substring filter on page title (case-insensitive)

    Returns:
        list of {id, title, version} dicts
    """
    pages: list[dict] = []
    start = 0
    limit = 200

    while True:
        for attempt in range(4):
            try:
                resp = requests.get(
                    f"{_base_v1()}/content",
                    auth=_auth(),
                    params={
                        "spaceKey": space_key,
                        "type": "page",
                        "limit": limit,
                        "start": start,
                        "expand": "version",
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                break
            except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        data = resp.json()
        results = data.get("results", [])

        for p in results:
            if title_filter and title_filter.lower() not in p["title"].lower():
                continue
            pages.append({
                "id":      p["id"],
                "title":   p["title"],
                "version": p.get("version", {}).get("number", 1),
            })

        if not data.get("_links", {}).get("next"):
            break
        start += len(results)
        logger.debug(f"Fetched {len(pages)} pages so far from space {space_key!r}...")

    logger.info(f"Space {space_key!r}: found {len(pages)} pages (filter={title_filter!r})")
    return pages


def list_child_pages(parent_id: str, max_depth: int = 20) -> list[dict]:
    """
    Recursively list all descendant pages under a parent page.
    max_depth prevents infinite recursion from cyclic API responses.

    Returns list of {id, title, version} dicts.
    """
    pages: list[dict] = []

    def _fetch(pid: str, depth: int = 0):
        if depth >= max_depth:
            logger.warning(f"Max recursion depth {max_depth} reached at page {pid}, stopping")
            return
        start = 0
        while True:
            for attempt in range(3):
                try:
                    resp = requests.get(
                        f"{_base_v1()}/content/{pid}/child/page",
                        auth=_auth(),
                        params={"limit": 100, "start": start, "expand": "version"},
                        timeout=30,
                    )
                    break
                except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
                    if attempt == 2:
                        raise
                    time.sleep(3 * (attempt + 1))
            if resp.status_code == 404:
                break
            resp.raise_for_status()
            data = resp.json()
            children = data.get("results", [])
            for c in children:
                pages.append({
                    "id":      c["id"],
                    "title":   c["title"],
                    "version": c.get("version", {}).get("number", 1),
                })
                _fetch(c["id"], depth + 1)
            if not data.get("_links", {}).get("next"):
                break
            start += len(children)

    _fetch(parent_id)
    logger.info(f"Parent {parent_id}: found {len(pages)} descendant pages")
    return pages


# ─── Per-page fetch + chunk ────────────────────────────────────────────────────

def _fetch_page_body(page_id: str) -> tuple[str, str, str]:
    """
    Fetch a single page body via v2 API with retry.
    Returns (page_id, title, text). Returns empty text on error.
    """
    for attempt in range(3):
        try:
            resp = requests.get(
                f"{_base_v2()}/pages/{page_id}",
                auth=_auth(),
                params={"body-format": "storage"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            title = data.get("title", "Untitled")
            html  = data.get("body", {}).get("storage", {}).get("value", "")
            text  = _html_to_text(html) if html else ""
            return page_id, title, text
        except (requests.exceptions.ConnectionError, requests.exceptions.ReadTimeout):
            if attempt == 2:
                logger.warning(f"Failed to fetch page {page_id} after 3 attempts")
                return page_id, "", ""
            time.sleep(3 * (attempt + 1))
        except Exception as e:
            logger.warning(f"Failed to fetch page {page_id}: {e}")
            return page_id, "", ""
    return page_id, "", ""


def fetch_and_chunk_page(page_meta: dict, space_key: str = "") -> list[dict]:
    """
    Fetch body for a single page and split into section-level chunks.
    Returns one dict per H1/H2/H3 section (via chunker.chunk_document).
    Returns empty list on failure or empty body.

    Chunking here is fixed-window, not semantic: a space crawl chunks pages in
    parallel worker threads and embeds afterwards in batches
    (prd_pipeline._ingest_one_space), so passing embed_fn would serialise an
    embedding call per segment per page inside every worker. The same page
    ingested singly gets semantic chunks — deliberate, for crawl throughput.
    """
    from ingestion.chunker import chunk_document
    from ingestion.confluence_ingestor import INGEST_ATTACHMENTS, fetch_attachment_markdown

    page_id = page_meta["id"]
    _, title, text = _fetch_page_body(page_id)
    if not text:
        return []

    source_id = f"confluence:{page_id}"
    domain = _confluence_domain()
    if space_key:
        doc_url = f"https://{domain}/wiki/spaces/{space_key}/pages/{page_id}"
    else:
        doc_url = f"https://{domain}/wiki/pages/viewpage.action?pageId={page_id}"

    full_text = f"# {title}\n\n{text}"

    # Honour the same flag as single-page ingest. Without this a space crawl
    # silently skips attachments while a single-page ingest of the same page
    # picks them up — the inconsistency is worse than the feature's absence.
    if INGEST_ATTACHMENTS:
        attachments = fetch_attachment_markdown(page_id)
        if attachments:
            full_text = f"{full_text}\n\n{attachments}"

    raw_chunks = chunk_document(full_text, source_id)

    return [
        {
            "source_id":       source_id,
            "source_type":     "confluence",
            "source_version":  str(page_meta["version"]),
            "doc_title":       title,
            "doc_url":         doc_url,
            "section_heading": c.get("section_heading", title),
            "chunk_text":      c["chunk_text"],
            "parent_text":     c.get("parent_text"),
            "chunk_type":      c.get("chunk_type"),
            "doc_type":        _classify_doc(title),
            "chunk_index":     i,
        }
        for i, c in enumerate(raw_chunks)
    ]


# ─── Bulk crawl ───────────────────────────────────────────────────────────────

def crawl_space(
    space_key: str,
    title_filter: str = "",
    parent_id: str = "",
    max_workers: int = 5,
    progress_callback=None,
    indexed_versions: dict[str, str] | None = None,
) -> tuple[list[dict], list[str], list[str]]:
    """
    Crawl a Confluence space (or subtree under parent_id) and return all chunks.

    Args:
        space_key:         Confluence space key (e.g. 'DOCS')
        title_filter:      optional title substring filter
        parent_id:         if set, only pages under this parent are crawled
        max_workers:       parallel fetches (default 5 — be polite to the API)
        progress_callback: optional callable(done, total) for progress reporting
        indexed_versions:  {source_id: source_version} already in Elasticsearch. Pages whose
                           live version matches are skipped WITHOUT fetching the body — the
                           listing call already told us the version, so an unchanged page
                           costs zero page requests. Pass None to force a full re-crawl.

    Returns:
        (chunks, skipped_ids, unchanged_ids)
          chunks        — ready for embed+upsert
          skipped_ids   — page had no body, or fetching/parsing failed (data loss: inspect)
          unchanged_ids — version matched the index; deliberately not re-fetched
    """
    if parent_id:
        pages = list_child_pages(parent_id)
        if title_filter:
            pages = [p for p in pages if title_filter.lower() in p["title"].lower()]
    else:
        pages = list_space_pages(space_key, title_filter=title_filter)

    unchanged_ids: list[str] = []
    if indexed_versions:
        fresh: list[dict] = []
        for p in pages:
            stored = indexed_versions.get(f"confluence:{p['id']}")
            # "" means indexed but version unknown (pre-dates source_version) — re-ingest.
            if stored and stored == str(p.get("version", "")):
                unchanged_ids.append(p["id"])
            else:
                fresh.append(p)
        if unchanged_ids:
            logger.info(
                "Space %r: %s/%s pages unchanged since last ingest — not re-fetching",
                space_key, len(unchanged_ids), len(pages),
            )
        pages = fresh

    total = len(pages)
    logger.info(f"Crawling {total} pages from space {space_key!r} ...")

    all_chunks: list[dict] = []
    skipped_ids: list[str] = []
    done = 0

    if total == 0:
        return all_chunks, skipped_ids, unchanged_ids

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(fetch_and_chunk_page, p, space_key): p for p in pages}
        for fut in as_completed(futures):
            page_meta = futures[fut]
            done += 1
            try:
                chunks = fut.result()
                if chunks:
                    all_chunks.extend(chunks)
                else:
                    skipped_ids.append(page_meta["id"])
            except Exception as e:
                logger.warning(f"Error processing page {page_meta['id']}: {e}")
                skipped_ids.append(page_meta["id"])

            if progress_callback:
                progress_callback(done, total)
            elif done % 20 == 0 or done == total:
                logger.info(f"  Fetched {done}/{total} pages ({len(all_chunks)} chunks so far)")

    logger.info(
        f"Space {space_key!r} crawl done: {len(all_chunks)} chunks "
        f"from {total - len(skipped_ids)}/{total} pages "
        f"({len(skipped_ids)} skipped/empty, {len(unchanged_ids)} unchanged)"
    )
    return all_chunks, skipped_ids, unchanged_ids
