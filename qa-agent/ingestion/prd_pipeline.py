"""
Phase 2: PRD Ingestion Pipeline

Orchestrates fetch → chunk → embed → upsert for a PRD source.

Supported source_type values:
  confluence       — single Confluence page by URL or page ID
  confluence_space — every page in one Confluence space (recursive, any nesting depth)
  confluence_site  — every page in every selected space on the site
  gitlab           — all .md files in a GitLab module folder
  gitlab_file      — single .md file path in GitLab repo
  file             — uploaded file (Excel, Word, PDF, Markdown, text)

Space and site ingests are incremental: a page is re-fetched and re-embedded only when
its Confluence version differs from the indexed source_version. Pass force=True to
re-ingest regardless (required after an embedding-model change, since matching versions
would otherwise skip pages whose stored vectors are now invalid).

Entry point: run_ingest(source_type, source, embed_client, es_store, pg_store, run_id, ...)
             run_file_ingest(filename, content, embed_client, es_store, pg_store, run_id, ...)
"""
import asyncio
import logging
import time
import uuid

from embeddings.embed_client import EmbedClient
from embeddings.es_store import ESStore
from embeddings.pg_store import PGStore
from ingestion.confluence_ingestor import ingest_confluence_page
from ingestion.confluence_space_ingestor import (
    list_space_pages, list_child_pages, fetch_and_chunk_page, crawl_space, list_spaces,
)
from ingestion.gitlab_ingestor import ingest_gitlab_module, ingest_gitlab_folder, fetch_file_content
from ingestion.file_ingestor import ingest_file

from ingestion.doc_classify import classify as _classify_doc

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 32


def _embed_chunks(chunks: list[dict], embed_client: EmbedClient) -> list[dict]:
    """
    Embed all chunks in batches. Adds 'embedding' field to each chunk dict.
    Uses embed_document() — PRD chunks are indexed documents, not queries.
    """
    if not chunks:
        return []

    texts = [
        embed_client.format_prd_chunk(c.get("section_heading"), c["chunk_text"])
        for c in chunks
    ]

    embeddings = embed_client.embed_documents(texts, batch_size=EMBED_BATCH_SIZE)

    for chunk, vec in zip(chunks, embeddings):
        chunk["embedding"] = vec

    return chunks


def _upsert_by_source(chunks: list[dict], es_store: ESStore, run_id: str) -> int:
    """Group chunks by source_id and upsert each group (deletes old chunks first)."""
    if not chunks:
        return 0

    source_ids = {c["source_id"] for c in chunks}
    total = 0
    for source_id in source_ids:
        source_chunks = [c for c in chunks if c["source_id"] == source_id]
        upserted = es_store.upsert_prd_chunks(source_chunks)
        logger.info(f"[{run_id}] Upserted {upserted} chunks for {source_id}")
        total += upserted
    return total


# ─── Streaming space ingest (page-by-page, resumable) ─────────────────────────

EMBED_TIMEOUT_PER_PAGE = 120  # seconds per page in a batch before giving up


async def _ingest_one_space(
    space_key: str,
    title_filter: str,
    parent_id: str,
    space_workers: int,
    embed_client: EmbedClient,
    es_store: ESStore,
    pg_store: PGStore,
    run_id: str,
    *,
    indexed_versions: dict[str, str] | None,
    embed_batch_size: int = 4,
    module: str | None = None,
) -> dict:
    """
    Ingest one Confluence space. Does NOT touch sync_runs — callers aggregate and
    finalise, so this is reusable for both single-space and site-wide crawls.

    Incremental: pages whose live Confluence version matches the indexed source_version
    are skipped before their body is fetched, so a no-op re-run costs one listing call
    per space instead of a full crawl.

    Returns per-space stats: {space_key, chunks, pages_failed, pages_unchanged,
                              batches_timed_out, failed_page_ids}
    """
    loop = asyncio.get_running_loop()

    def _crawl():
        return crawl_space(
            space_key=space_key,
            title_filter=title_filter,
            parent_id=parent_id,
            max_workers=space_workers,
            indexed_versions=indexed_versions,
        )

    all_chunks, skipped_ids, unchanged_ids = await loop.run_in_executor(None, _crawl)
    logger.info(
        "[%s] %s: %s chunks to embed (%s empty/failed, %s unchanged)",
        run_id, space_key, len(all_chunks), len(skipped_ids), len(unchanged_ids),
    )

    # Tag chunks with the space key so module filtering works per space. Without this,
    # space-ingested chunks have no module and match EVERY module filter at search time
    # (search_similar_prd_chunks deliberately includes untagged chunks).
    module_tag = module or space_key
    if module_tag:
        for c in all_chunks:
            c.setdefault("module", module_tag)

    stats = {
        "space_key": space_key,
        "chunks": 0,
        "pages_failed": len(skipped_ids),
        "pages_unchanged": len(unchanged_ids),
        "batches_timed_out": 0,
        "failed_page_ids": list(skipped_ids),
    }
    if not all_chunks:
        return stats

    total = len(all_chunks)
    for i in range(0, total, embed_batch_size):
        batch = all_chunks[i: i + embed_batch_size]
        texts = [
            embed_client.format_prd_chunk(c.get("section_heading"), c["chunk_text"])
            for c in batch
        ]
        try:
            vectors = await asyncio.wait_for(
                loop.run_in_executor(None, embed_client.embed_documents, texts),
                timeout=EMBED_TIMEOUT_PER_PAGE * len(batch),
            )
        except asyncio.TimeoutError:
            # Counted as an error, not silently dropped — these chunks are absent from
            # the index and the run status must reflect that.
            stats["batches_timed_out"] += 1
            stats["failed_page_ids"].extend(
                sorted({str(c.get("source_id", "")) for c in batch})
            )
            logger.warning(
                "[%s] %s: embedding timed out for batch %s (chunks %s-%s) — chunks NOT indexed",
                run_id, space_key, i // embed_batch_size + 1, i, i + len(batch) - 1,
            )
            continue

        for chunk, vec in zip(batch, vectors):
            chunk["embedding"] = vec

        stats["chunks"] += await loop.run_in_executor(
            None, lambda b=batch: _upsert_by_source(b, es_store, run_id)
        )

        done = i + len(batch)
        if done % 200 == 0 or done == total:
            logger.info(
                "[%s] %s: embedded %s/%s chunks, %s in ES",
                run_id, space_key, done, total, stats["chunks"],
            )

    return stats


async def _stream_ingest_space(
    space_key: str,
    title_filter: str,
    parent_id: str,
    space_workers: int,
    embed_client: EmbedClient,
    es_store: ESStore,
    pg_store: PGStore,
    run_id: str,
    started: float,
    embed_batch_size: int = 4,
    module: str | None = None,
    force: bool = False,
) -> dict:
    """
    Resumable, incremental single-space ingest.

    force=True re-fetches and re-embeds every page regardless of version — use after an
    embedding-model change, since stored vectors are then invalid even though the
    Confluence versions still match.
    """
    loop = asyncio.get_running_loop()

    indexed_versions = None
    if not force:
        indexed_versions = await loop.run_in_executor(
            None, lambda: es_store.get_indexed_source_versions("confluence:")
        )

    stats = await _ingest_one_space(
        space_key, title_filter, parent_id, space_workers,
        embed_client, es_store, pg_store, run_id,
        indexed_versions=indexed_versions,
        embed_batch_size=embed_batch_size,
        module=module,
    )

    elapsed = round(time.time() - started, 1)
    had_errors = stats["pages_failed"] > 0 or stats["batches_timed_out"] > 0
    final_status = "completed_with_errors" if had_errors else "completed"

    pg_store.complete_run(
        run_id,
        chunks_ingested=stats["chunks"],
        run_metadata={
            "confluence_space_key": space_key,
            "confluence_space_skipped_page_ids": stats["failed_page_ids"][:2000],
            "confluence_space_skipped_count": stats["pages_failed"],
            "confluence_pages_unchanged": stats["pages_unchanged"],
            "confluence_embed_batches_timed_out": stats["batches_timed_out"],
            "forced_full_reingest": force,
        },
        final_status=final_status,
    )
    logger.info(
        "[%s] Space ingest complete: %s chunks in %s (%s unchanged, %s failed)",
        run_id, stats["chunks"], elapsed, stats["pages_unchanged"], stats["pages_failed"],
    )

    return {
        "run_id":          run_id,
        "status":          final_status,
        "source_type":     "confluence_space",
        "source":          space_key,
        "chunks_ingested": stats["chunks"],
        "pages_unchanged": stats["pages_unchanged"],
        "skipped_pages":   stats["pages_failed"],
        "elapsed_s":       elapsed,
    }


async def _ingest_site(
    embed_client: EmbedClient,
    es_store: ESStore,
    pg_store: PGStore,
    run_id: str,
    started: float,
    *,
    title_filter: str = "",
    space_keys: str = "",
    include_personal: bool = False,
    include_archived: bool = False,
    space_workers: int = 5,
    embed_batch_size: int = 4,
    force: bool = False,
) -> dict:
    """
    Site-wide recursive ingest: every page of every selected space.

    Spaces are processed one at a time and embedded in small batches, so peak memory is
    bounded by a single space rather than the whole site. The version map is fetched once
    and shared, so an unchanged site re-runs at roughly one listing call per space.

    space_keys is a comma-separated allowlist; empty means every non-personal,
    non-archived space on the site.
    """
    loop = asyncio.get_running_loop()

    spaces = await loop.run_in_executor(
        None,
        lambda: list_spaces(
            include_personal=include_personal,
            include_archived=include_archived,
            key_filter=space_keys,
        ),
    )
    if not spaces:
        pg_store.complete_run(
            run_id,
            chunks_ingested=0,
            run_metadata={"confluence_site_spaces": [], "space_filter": space_keys},
            final_status="completed_empty",
        )
        return {
            "run_id": run_id,
            "status": "completed_empty",
            "source_type": "confluence_site",
            "spaces": 0,
            "chunks_ingested": 0,
            "elapsed_s": round(time.time() - started, 1),
            "warning": "No spaces matched. Check space_keys / include_personal.",
        }

    indexed_versions = None
    if not force:
        indexed_versions = await loop.run_in_executor(
            None, lambda: es_store.get_indexed_source_versions("confluence:")
        )
        logger.info(
            "[%s] %s Confluence source(s) already indexed — incremental mode",
            run_id, len(indexed_versions),
        )

    per_space: list[dict] = []
    total_chunks = 0
    total_failed = 0
    total_unchanged = 0
    total_timeouts = 0

    for n, space in enumerate(spaces, 1):
        key = space["key"]
        logger.info("[%s] Space %s/%s: %s (%s)", run_id, n, len(spaces), key, space["name"])
        try:
            stats = await _ingest_one_space(
                key, title_filter, "", space_workers,
                embed_client, es_store, pg_store, run_id,
                indexed_versions=indexed_versions,
                embed_batch_size=embed_batch_size,
                module=key,
            )
        except Exception as exc:
            # One bad space (permissions, API quirk) must not abort a 40-space crawl.
            logger.exception("[%s] Space %s failed, continuing: %s", run_id, key, exc)
            per_space.append({"space_key": key, "error": str(exc)[:500], "chunks": 0})
            total_failed += 1
            continue

        total_chunks += stats["chunks"]
        total_failed += stats["pages_failed"]
        total_unchanged += stats["pages_unchanged"]
        total_timeouts += stats["batches_timed_out"]
        per_space.append({
            "space_key": key,
            "chunks": stats["chunks"],
            "pages_failed": stats["pages_failed"],
            "pages_unchanged": stats["pages_unchanged"],
        })

    elapsed = round(time.time() - started, 1)
    had_errors = total_failed > 0 or total_timeouts > 0
    if total_chunks == 0 and total_unchanged == 0:
        final_status = "completed_empty"
    else:
        final_status = "completed_with_errors" if had_errors else "completed"

    pg_store.complete_run(
        run_id,
        chunks_ingested=total_chunks,
        run_metadata={
            "confluence_site_spaces": [s["key"] for s in spaces],
            "confluence_site_per_space": per_space[:500],
            "confluence_pages_unchanged": total_unchanged,
            "confluence_space_skipped_count": total_failed,
            "confluence_embed_batches_timed_out": total_timeouts,
            "space_filter": space_keys,
            "forced_full_reingest": force,
        },
        final_status=final_status,
    )
    logger.info(
        "[%s] Site ingest complete: %s spaces, %s chunks, %s unchanged, %s failed in %s",
        run_id, len(spaces), total_chunks, total_unchanged, total_failed, elapsed,
    )

    return {
        "run_id":          run_id,
        "status":          final_status,
        "source_type":     "confluence_site",
        "spaces":          len(spaces),
        "space_keys":      [s["key"] for s in spaces],
        "chunks_ingested": total_chunks,
        "pages_unchanged": total_unchanged,
        "skipped_pages":   total_failed,
        "per_space":       per_space,
        "elapsed_s":       elapsed,
    }


# ─── Main pipeline ────────────────────────────────────────────────────────────

async def run_ingest(
    source_type: str,
    source: str,
    embed_client: EmbedClient,
    es_store: ESStore,
    pg_store: PGStore,
    run_id: str | None = None,
    module: str | None = None,
    ref: str = "main",
    # confluence_space / confluence_site options
    title_filter: str = "",
    parent_id: str = "",
    space_workers: int = 5,
    # confluence_site options
    space_keys: str = "",
    include_personal: bool = False,
    include_archived: bool = False,
    force: bool = False,
) -> dict:
    """
    Full ingestion pipeline for one PRD source.

    Args:
        source_type:    "confluence" | "confluence_space" | "gitlab" | "gitlab_file" | "file"
        source:         page ID/URL for confluence; space key for confluence_space;
                        module name for gitlab; file path for gitlab_file
        embed_client:   loaded EmbedClient
        es_store:       ESStore instance
        pg_store:       PGStore instance
        run_id:         optional; generated if not provided
        module:         module name (for gitlab source_type)
        ref:            git ref (branch/tag) for gitlab sources
        title_filter:   title substring filter (confluence_space only)
        parent_id:      restrict to subtree under parent page (confluence_space only)
        space_workers:  parallel fetchers for confluence_space (default 5)

    Returns:
        {run_id, status, source_type, source, chunks_ingested, elapsed_s}
    """
    run_id = run_id or str(uuid.uuid4())
    started = time.time()

    pg_store.start_run(run_id, run_type="prd_ingest", prd_source=source)
    logger.info(f"[{run_id}] Starting PRD ingest: type={source_type!r} source={source!r}")

    # Confluence space uses streaming page-by-page ingest (resumable, low memory)
    if source_type == "confluence_space":
        try:
            return await _stream_ingest_space(
                source, title_filter, parent_id, space_workers,
                embed_client, es_store, pg_store, run_id, started,
                module=module,
                force=force,
            )
        except Exception as exc:
            elapsed = round(time.time() - started, 1)
            logger.exception(f"[{run_id}] Space ingest failed after {elapsed}s: {exc}")
            pg_store.fail_run(run_id, str(exc))
            raise

    # Site-wide: every page of every selected space, one space at a time
    if source_type == "confluence_site":
        try:
            return await _ingest_site(
                embed_client, es_store, pg_store, run_id, started,
                title_filter=title_filter,
                space_keys=space_keys or (source or "").strip(),
                include_personal=include_personal,
                include_archived=include_archived,
                space_workers=space_workers,
                force=force,
            )
        except Exception as exc:
            elapsed = round(time.time() - started, 1)
            logger.exception(f"[{run_id}] Site ingest failed after {elapsed}s: {exc}")
            pg_store.fail_run(run_id, str(exc))
            raise

    try:
        # ── Step 1: Fetch + chunk ──────────────────────────────────────────────
        chunks: list[dict] = []

        # Semantic chunking: pass embed_documents as embed_fn for topic boundary detection
        _embed_fn = embed_client.embed_documents

        if source_type == "confluence":
            chunks = await asyncio.get_running_loop().run_in_executor(
                None, lambda: ingest_confluence_page(source, embed_fn=_embed_fn)
            )

        elif source_type == "gitlab":
            effective_module = module or source
            chunks = await asyncio.get_running_loop().run_in_executor(
                None, lambda: ingest_gitlab_module(effective_module, ref=ref, embed_fn=_embed_fn)
            )

        elif source_type == "gitlab_file":
            def _fetch_single():
                from ingestion.chunker import chunk_document
                content = fetch_file_content(source, ref=ref)
                if not content.strip():
                    return []
                import re as _re
                heading = _re.sub(r'\.md$', '', source.rsplit("/", 1)[-1], flags=_re.IGNORECASE)
                source_id = f"gitlab:{source}"
                raw_chunks = chunk_document(
                    f"# {heading}\n\n{content.strip()}",
                    source_id,
                    embed_fn=_embed_fn,
                )
                return [
                    {
                        "source_id":       source_id,
                        "source_type":     "gitlab_markdown",
                        "source_version":  ref,
                        "doc_title":       heading,
                        "doc_url":         None,
                        "section_heading": c.get("section_heading", heading),
                        "chunk_text":      c["chunk_text"],
                        "parent_text":     c.get("parent_text"),
                        "chunk_type":      c.get("chunk_type"),
                        "doc_type":        _classify_doc(heading),
                        "chunk_index":     i,
                    }
                    for i, c in enumerate(raw_chunks)
                ]
            chunks = await asyncio.get_running_loop().run_in_executor(None, _fetch_single)

        else:
            raise ValueError(
                f"Unsupported source_type: {source_type!r}. "
                "Use 'confluence', 'confluence_space', 'gitlab', 'gitlab_file', or 'file'."
            )

        if not chunks:
            logger.warning(f"[{run_id}] No chunks produced for {source_type}:{source}")
            pg_store.complete_run(run_id, chunks_ingested=0)
            return {
                "run_id":          run_id,
                "status":          "completed",
                "source_type":     source_type,
                "source":          source,
                "chunks_ingested": 0,
                "elapsed_s":       round(time.time() - started, 1),
                "warning":         "No content found in source",
            }

        # Stamp module onto every chunk if provided (enables KB filtering by module)
        if module and chunks:
            for c in chunks:
                c.setdefault("module", module)

        logger.info(f"[{run_id}] Produced {len(chunks)} chunks, embedding...")

        # ── Steps 2+3: Embed + Upsert ─────────────────────────────────────────
        chunks = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _embed_chunks(chunks, embed_client)
        )
        total_upserted = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _upsert_by_source(chunks, es_store, run_id)
        )

        # ── Step 4: Update Postgres audit log ─────────────────────────────────
        pg_store.complete_run(run_id, chunks_ingested=total_upserted)

        elapsed = round(time.time() - started, 1)
        source_ids = list({c["source_id"] for c in chunks})
        logger.info(f"[{run_id}] Ingest complete: {total_upserted} chunks in {elapsed}s")

        return {
            "run_id":          run_id,
            "status":          "completed",
            "source_type":     source_type,
            "source":          source,
            "chunks_ingested": total_upserted,
            "source_ids":      source_ids,
            "elapsed_s":       elapsed,
        }

    except Exception as exc:
        elapsed = round(time.time() - started, 1)
        logger.exception(f"[{run_id}] Ingest failed after {elapsed}s: {exc}")
        pg_store.fail_run(run_id, str(exc))
        raise


# ─── File upload pipeline ─────────────────────────────────────────────────────

async def run_file_ingest(
    filename: str,
    content: bytes,
    embed_client: EmbedClient,
    es_store: ESStore,
    pg_store: PGStore,
    run_id: str | None = None,
    source_label: str = "",
) -> dict:
    """
    Ingest an uploaded file (Excel, Word, PDF, Markdown, text).

    Args:
        filename:     original filename (determines format + used as title)
        content:      raw file bytes
        source_label: optional label override for source_id (defaults to filename)

    Returns:
        {run_id, status, filename, chunks_ingested, elapsed_s}
    """
    run_id  = run_id or str(uuid.uuid4())
    started = time.time()
    label   = source_label or filename

    pg_store.start_run(run_id, run_type="prd_ingest", prd_source=label)
    logger.info(f"[{run_id}] Starting file ingest: {filename!r} ({len(content)} bytes)")

    try:
        # ── Step 1: Convert + chunk ────────────────────────────────────────────
        _embed_fn = embed_client.embed_documents
        chunks = await asyncio.get_running_loop().run_in_executor(
            None, lambda: ingest_file(filename, content, source_label, embed_fn=_embed_fn)
        )

        if not chunks:
            pg_store.complete_run(run_id, chunks_ingested=0)
            return {
                "run_id":          run_id,
                "status":          "completed",
                "filename":        filename,
                "chunks_ingested": 0,
                "elapsed_s":       round(time.time() - started, 1),
                "warning":         "No content extracted from file",
            }

        logger.info(f"[{run_id}] {len(chunks)} chunks from {filename!r}, embedding...")

        # ── Step 2: Embed ──────────────────────────────────────────────────────
        chunks = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _embed_chunks(chunks, embed_client)
        )

        # ── Step 3: Upsert ────────────────────────────────────────────────────
        total_upserted = await asyncio.get_running_loop().run_in_executor(
            None, lambda: _upsert_by_source(chunks, es_store, run_id)
        )

        pg_store.complete_run(run_id, chunks_ingested=total_upserted)

        elapsed = round(time.time() - started, 1)
        logger.info(f"[{run_id}] File ingest complete: {total_upserted} chunks in {elapsed}s")

        return {
            "run_id":          run_id,
            "status":          "completed",
            "filename":        filename,
            "source_id":       chunks[0]["source_id"] if chunks else None,
            "chunks_ingested": total_upserted,
            "elapsed_s":       elapsed,
        }

    except Exception as exc:
        elapsed = round(time.time() - started, 1)
        logger.exception(f"[{run_id}] File ingest failed after {elapsed}s: {exc}")
        pg_store.fail_run(run_id, str(exc))
        raise
