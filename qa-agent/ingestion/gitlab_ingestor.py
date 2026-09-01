"""
GitLab PRD ingestor for Phase 2.

Fetches all .md files from a GitLab repository folder (recursively) via the
GitLab REST API directly. Converts each file to plain text and chunks it.

Uses the GitLab REST API directly for bulk file operations (this path does not use MCP).
"""
import os
import re
import base64
import logging
from urllib.parse import quote

import requests

from ingestion.doc_classify import classify as _classify_doc

logger = logging.getLogger(__name__)

GITLAB_HOST = os.environ.get("GITLAB_HOST", "gitlab.com")
GITLAB_PROJECT_ID = os.environ.get("GITLAB_PROJECT_ID", "")   # e.g. "group/project"

_BASE = f"https://{GITLAB_HOST}/api/v4"

# Module → top-level folder path in the repo (customize for your project layout).
MODULE_FOLDER_MAP = {
    "Platform":     "platform",
    "API":          "api",
    "Docs":         "docs",
    "Admin":        "admin",
    "Analytics":    "analytics",
}


def _headers() -> dict:
    token = os.environ.get("GITLAB_TOKEN", "")
    if not token:
        raise RuntimeError("GITLAB_TOKEN must be set in the environment")
    return {"PRIVATE-TOKEN": token}


def _encode_project(project_id: str | None = None) -> str:
    pid = project_id or GITLAB_PROJECT_ID
    return quote(pid, safe="")


def list_md_files(folder_path: str, ref: str = "main", project_id: str | None = None) -> list[str]:
    """
    Recursively list all .md file paths under `folder_path` in the repo.
    Returns list of full file paths (e.g. 'platform/docs/Features.md').
    """
    pid = _encode_project(project_id)
    all_files: list[str] = []
    page = 1

    while True:
        resp = requests.get(
            f"{_BASE}/projects/{pid}/repository/tree",
            headers=_headers(),
            params={
                "path": folder_path,
                "ref": ref,
                "recursive": "true",
                "per_page": 100,
                "page": page,
            },
            timeout=30,
        )
        if resp.status_code == 404:
            logger.warning(f"GitLab folder not found: {folder_path!r}")
            return []
        resp.raise_for_status()

        items = resp.json()
        for item in items:
            if item.get("type") == "blob" and item["path"].lower().endswith(".md"):
                all_files.append(item["path"])

        if len(items) < 100:
            break
        page += 1

    logger.info(f"Found {len(all_files)} .md files under {folder_path!r}")
    return all_files


def fetch_file_content(file_path: str, ref: str = "main", project_id: str | None = None) -> str:
    """Fetch and decode a single file's content from GitLab."""
    pid = _encode_project(project_id)
    encoded_path = quote(file_path, safe="")

    resp = requests.get(
        f"{_BASE}/projects/{pid}/repository/files/{encoded_path}",
        headers=_headers(),
        params={"ref": ref},
        timeout=30,
    )
    if resp.status_code == 404:
        logger.warning(f"GitLab file not found: {file_path!r}")
        return ""
    resp.raise_for_status()

    data = resp.json()
    content = data.get("content", "")
    encoding = data.get("encoding", "base64")

    if encoding == "base64":
        return base64.b64decode(content).decode("utf-8", errors="replace")
    return content


def ingest_gitlab_folder(
    folder_path: str,
    ref: str = "main",
    project_id: str | None = None,
    embed_fn=None,
) -> list[dict]:
    """
    Ingest all .md files from a GitLab folder.

    Returns chunked docs per file:
      [{source_id, source_type, source_version, doc_title, doc_url,
        section_heading, chunk_text, parent_text, chunk_index}, ...]

    source_id format: gitlab:{file_path}  (per-file, so partial failures
    don't delete content from files that weren't re-fetched)
    """
    from ingestion.chunker import chunk_document

    file_paths = list_md_files(folder_path, ref=ref, project_id=project_id)
    if not file_paths:
        return []

    docs: list[dict] = []

    for idx, file_path in enumerate(file_paths):
        content = fetch_file_content(file_path, ref=ref, project_id=project_id)
        if not content.strip():
            continue
        rel_path = file_path[len(folder_path):].lstrip("/")
        heading = re.sub(r'\.md$', '', rel_path, flags=re.IGNORECASE)
        # URL-encode path components to handle spaces and special chars
        encoded_path = quote(file_path, safe="/")
        project_id_val = project_id or GITLAB_PROJECT_ID
        doc_url = f"https://{GITLAB_HOST}/{project_id_val}/-/blob/{ref}/{encoded_path}"
        full_text = f"# {heading}\n\n{content.strip()}"

        source_id = f"gitlab:{file_path}"
        raw_chunks = chunk_document(full_text, source_id, embed_fn=embed_fn)
        for i, c in enumerate(raw_chunks):
            docs.append({
                "source_id":       source_id,
                "source_type":     "gitlab_markdown",
                "source_version":  ref,
                "doc_title":       heading,
                "doc_url":         doc_url,
                "section_heading": c.get("section_heading", heading),
                "chunk_text":      c["chunk_text"],
                "parent_text":     c.get("parent_text"),
                "chunk_type":      c.get("chunk_type"),
                "doc_type":        _classify_doc(heading),
                "chunk_index":     i,
            })

    logger.info(
        f"GitLab folder {folder_path!r}: {len(file_paths)} files → {len(docs)} chunks"
    )
    return docs


def ingest_gitlab_module(module: str, ref: str = "main", embed_fn=None) -> list[dict]:
    """
    Ingest all docs for a named module (e.g. 'Platform', 'Docs').
    Maps module name → folder path via MODULE_FOLDER_MAP.
    """
    folder = MODULE_FOLDER_MAP.get(module, module)
    return ingest_gitlab_folder(folder, ref=ref, embed_fn=embed_fn)
