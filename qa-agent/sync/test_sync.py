"""
Phase 1: Sync all Xray test cases into Elasticsearch.

Pipeline:
  1. Enumerate root folders via the `get_folders` MCP operation
  2. Fetch all test metadata per folder (paginated, 100/page)
  3. Diff against existing ES hashes — skip unchanged tests
  4. Fetch descriptions in bulk via `search_issues` (100/call)
  4b. Fetch steps + preconditions per test (`get_test`, bounded concurrency)
  5. Embed (summary + module + labels + description + steps) → upsert to ES
  6. Delete tests no longer in Xray
  7. Write audit log to Postgres sync_runs
"""
import asyncio
import hashlib
import json
import logging
import time
import uuid
from datetime import datetime, timezone

from integrations import xray_client
from observability.canonical_json import normalize_json_obj
from observability.phase_ledger import append_entry_async

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 8     # tests to embed at once; smaller = lower peak memory, more frequent ES checkpoints
DESC_BULK_SIZE   = 100   # tests per `search_issues` call


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_key(test: dict) -> str | None:
    """`get_tests_from_folder` → 'key'; `get_test` → 'jiraKey'. Handle both."""
    return test.get("jiraKey") or test.get("key")


def _extract_folder_path(test: dict) -> str:
    # Prefer the folder path injected from the sync loop context
    if "_folder_path" in test:
        return test["_folder_path"]
    folder = test.get("folder")
    if isinstance(folder, dict):
        return folder.get("path", "")
    return str(folder) if folder else ""


def _extract_test_type(test: dict) -> str:
    tt = test.get("testType")
    if isinstance(tt, dict):
        return tt.get("kind", "")
    return str(tt) if tt else ""


def _content_hash(test: dict) -> str:
    """SHA-256 fingerprint covering all embedded fields. Change = re-embed."""
    payload = json.dumps({
        "summary":        test.get("summary", ""),
        "description":    test.get("description", "") or "",
        "folder":         _extract_folder_path(test),
        "labels":         sorted(test.get("labels") or []),
        "testType":       _extract_test_type(test),
        "steps_text":     test.get("steps_text") or "",
        "preconditions":  test.get("preconditions") or "",
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _metadata_hash(test: dict) -> str:
    """
    Hash of fields available before description/steps are fetched (Phase 3 diff).
    Tests whose metadata_hash matches the stored hash can skip Phase 4 API calls.
    """
    payload = json.dumps({
        "summary":  test.get("summary", ""),
        "folder":   _extract_folder_path(test),
        "labels":   sorted(test.get("labels") or []),
        "testType": _extract_test_type(test),
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def _steps_to_text(steps: object) -> str:
    if not steps:
        return ""
    if not isinstance(steps, list):
        return str(steps)
    lines: list[str] = []
    for i, s in enumerate(steps, 1):
        if isinstance(s, dict):
            action = s.get("action") or ""
            data = s.get("data") or ""
            exp = s.get("expectedResult") or ""
            block = f"{i}. {action}: {data}" if action else f"{i}. {data}"
            if exp:
                block += f" (expected: {exp})"
            lines.append(block)
        else:
            lines.append(f"{i}. {s}")
    return "\n".join(lines)


def _preconditions_to_text(prec: object) -> str:
    if not prec:
        return ""
    if isinstance(prec, list):
        parts: list[str] = []
        for p in prec:
            if isinstance(p, dict):
                parts.append(str(p.get("key") or p.get("summary") or p.get("id") or p))
            else:
                parts.append(str(p))
        return " | ".join(parts)
    return str(prec)


async def _enrich_with_steps_and_prec(
    keys: list[str],
    *,
    concurrency: int = 12,
    run_id: str = "",
) -> dict[str, dict[str, str]]:
    """
    Fetch steps + preconditions via the `get_test` MCP operation (bounded parallelism).

    Reports running progress and a content tally: this phase makes one MCP call per test
    and previously logged nothing between start and finish, so a few hundred tests looked
    like a hang. The tally matters as much as the progress — a run where most tests come
    back with zero steps means the index is being built from summaries alone, which caps
    retrieval quality no matter how good the rest of the pipeline is.
    """
    if not keys:
        return {}
    sem = asyncio.Semaphore(concurrency)
    out: dict[str, dict[str, str]] = {}
    total = len(keys)
    t0 = time.monotonic()
    done = 0
    stats = {"with_steps": 0, "no_steps": 0, "with_prec": 0, "failed": 0}
    # Log ~20 progress lines regardless of run size.
    every = max(1, total // 20)

    async def one(k: str) -> None:
        nonlocal done
        async with sem:
            try:
                data = await xray_client.get_test(k)
            except Exception as e:
                stats["failed"] += 1
                logger.warning("[%s]   get_test FAILED for %s: %s", run_id, k, e)
                return
            finally:
                done += 1
                if done % every == 0 or done == total:
                    el = time.monotonic() - t0
                    rate = done / el if el > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    logger.info(
                        "[%s]   steps %s/%s (%d%%) — %.1f tests/s, eta %s | "
                        "with_steps=%s no_steps=%s failed=%s",
                        run_id, done, total, done * 100 // total, rate,
                        f"{eta:.0f}s" if eta < 120 else f"{eta/60:.1f}m",
                        stats["with_steps"], stats["no_steps"], stats["failed"],
                    )
            if not isinstance(data, dict):
                stats["failed"] += 1
                logger.warning(
                    "[%s]   get_test for %s returned %s, expected dict", run_id, k, type(data).__name__
                )
                return
            steps_text = _steps_to_text(data.get("steps"))
            prec = _preconditions_to_text(data.get("preconditions"))
            stats["with_steps" if steps_text else "no_steps"] += 1
            if prec:
                stats["with_prec"] += 1
            out[k] = {"steps_text": steps_text, "preconditions": prec}

    await asyncio.gather(*(one(k) for k in keys))

    logger.info(
        "[%s]   steps fetch done: %s/%s enriched in %s | with_steps=%s no_steps=%s "
        "with_preconditions=%s failed=%s",
        run_id, len(out), total, _elapsed(t0),
        stats["with_steps"], stats["no_steps"], stats["with_prec"], stats["failed"],
    )
    if total and stats["no_steps"] / total > 0.5:
        logger.warning(
            "[%s]   ⚠ %s/%s tests (%d%%) have NO steps — those documents are embedded from "
            "summary/labels only, which limits retrieval recall. Xray Cucumber tests keep "
            "their body in `gherkin`, which the MCP's get_test does not return.",
            run_id, stats["no_steps"], total, stats["no_steps"] * 100 // total,
        )
    return out


def _to_es_doc(test: dict, content_hash: str, vector: list[float]) -> dict:
    folder_path = _extract_folder_path(test)
    parts = [p for p in folder_path.strip("/").split("/") if p]
    module = parts[0] if parts else None
    return {
        "jira_key":      _get_key(test),
        "summary":       test.get("summary", ""),
        "description":   test.get("description", "") or "",
        "module":        module,
        "folder_path":   folder_path,
        "labels":        test.get("labels") or [],
        "steps_text":    test.get("steps_text") or "",
        "preconditions": test.get("preconditions") or "",
        "content_hash":  content_hash,
        "metadata_hash": _metadata_hash(test),
        "embedding":     vector,
        "synced_at":     datetime.now(timezone.utc).isoformat(),
    }


def _format_for_embedding(test: dict, embed_client) -> str:
    folder_path = _extract_folder_path(test)
    parts = [p for p in folder_path.strip("/").split("/") if p]
    module = parts[0] if parts else None
    labels = test.get("labels") or []
    return embed_client.format_test_case(
        summary=test.get("summary", ""),
        module=module,
        labels=labels if labels else None,
        description=test.get("description") or None,
        steps_text=test.get("steps_text") or None,
    )


def _elapsed(start: float) -> str:
    s = time.monotonic() - start
    return f"{s:.1f}s" if s < 60 else f"{s/60:.1f}m"


# ─── Main sync pipeline ───────────────────────────────────────────────────────

async def run_sync(
    project_key: str,
    embed_client,
    es_store,
    pg_store,
    run_id: str | None = None,
    folder_path: str = "",
) -> dict:
    """
    Full test sync pipeline. Returns a summary dict when done.
    Called via asyncio.create_task() — runs in background.
    """
    run_id = run_id or str(uuid.uuid4())
    t_total = time.monotonic()
    pg_store.start_run(run_id, run_type="test_sync")

    logger.info("=" * 60)
    logger.info(f"[{run_id}] SYNC START  project={project_key}")
    logger.info("=" * 60)

    try:
        # ── Phase 1: Folder enumeration ───────────────────────────────────
        t = time.monotonic()
        logger.info(f"[{run_id}] Phase 1/6 — Enumerating folders ...")

        fp_filter = (folder_path or "").strip()
        if fp_filter:
            query_paths = [fp_filter]
            logger.info(
                f"[{run_id}]   Folder-scoped sync: {query_paths!r} ({_elapsed(t)})"
            )
        else:
            folder_data = await xray_client.get_folders(project_key)
            if not isinstance(folder_data, dict):
                raise ValueError(
                    f"MCP get_folders returned unexpected type: {folder_data!r}. "
                    f"Check the tool mapping at GET /integrations/mcp/tools."
                )

            root_folders: list[str] = []
            for f in folder_data.get("folders", []):
                path = f.get("path") or f.get("name")
                if path:
                    root_folders.append(path)

            query_paths = root_folders if root_folders else [""]
            logger.info(
                f"[{run_id}]   Found {len(query_paths)} top-level folders "
                f"({_elapsed(t)}): {query_paths[:10]}"
                + (" ..." if len(query_paths) > 10 else "")
            )

        # ── Phase 2: Metadata fetch ───────────────────────────────────────
        t = time.monotonic()
        logger.info(f"[{run_id}] Phase 2/6 — Fetching test metadata ...")

        fetched: list[dict] = []
        seen_keys: set[str] = set()
        PAGE = 100

        for folder_path in query_paths:
            start = 0
            while True:
                raw = await xray_client.get_all_tests(
                    project_key, folder_path=folder_path, start=start, limit=PAGE
                )
                if not raw or "results" not in raw:
                    logger.warning(
                        f"[{run_id}]   Skipping folder '{folder_path}' "
                        f"— unexpected response: {str(raw)[:200]}"
                    )
                    break

                page = raw["results"]
                folder_total = raw.get("total", 0)

                new_in_page = 0
                for t_item in page:
                    k = _get_key(t_item)
                    if k and k not in seen_keys:
                        seen_keys.add(k)
                        t_item["_folder_path"] = folder_path  # inject from query context
                        fetched.append(t_item)
                        new_in_page += 1

                logger.debug(
                    f"[{run_id}]   folder='{folder_path}' "
                    f"page start={start} got={len(page)} new={new_in_page} "
                    f"folder_total={folder_total} running_total={len(fetched)}"
                )

                if len(page) < PAGE or start + len(page) >= folder_total:
                    logger.info(
                        f"[{run_id}]   folder '{folder_path}' done: "
                        f"{start + len(page)}/{folder_total} tests, "
                        f"running total={len(fetched)}"
                    )
                    break
                start += PAGE

        total_xray = len(fetched)
        logger.info(
            f"[{run_id}]   Phase 2 done: {total_xray} unique tests fetched ({_elapsed(t)})"
        )

        # ── Phase 3: Diff against ES ──────────────────────────────────────
        t = time.monotonic()
        logger.info(f"[{run_id}] Phase 3/6 — Diffing against Elasticsearch ...")

        # Scoped sync: only ES hashes under this folder — avoids deleting other folders' tests.
        # Use metadata_hash (summary/labels/folder/testType — available before Phase 4 fetch)
        # to skip unchanged tests early. Falls back to treating all tests as candidates for
        # legacy ES docs that predate the metadata_hash field (metadata_hash == "").
        existing_meta: dict[str, dict[str, str]] = es_store.get_existing_hashes_with_metadata(
            fp_filter if fp_filter else None
        )
        existing_hashes: dict[str, str] = {k: v["content_hash"] for k, v in existing_meta.items()}
        fetched_keys = {_get_key(t_item) for t_item in fetched if _get_key(t_item)}

        # Only compute stale keys if we fetched a reasonable number of tests.
        # If the fetch returned far fewer tests than ES has, it likely failed partially
        # — skip deletion to avoid data loss.
        if total_xray == 0 and len(existing_hashes) > 0:
            logger.warning(
                f"[{run_id}]   Xray returned 0 tests but ES has {len(existing_hashes)} — "
                "skipping stale deletion to prevent data loss from partial fetch failure"
            )
            stale_keys: set[str] = set()
        elif total_xray > 0 and total_xray < len(existing_hashes) * 0.5:
            logger.warning(
                f"[{run_id}]   Xray returned {total_xray} tests but ES has {len(existing_hashes)} — "
                "fetched <50% of expected, skipping stale deletion as safety measure"
            )
            stale_keys = set()
        else:
            stale_keys = set(existing_hashes.keys()) - fetched_keys

        candidates: list[dict] = []
        for t_item in fetched:
            k = _get_key(t_item)
            if not k:
                continue
            stored = existing_meta.get(k)
            if stored is None:
                candidates.append(t_item)  # new test
            elif stored["metadata_hash"] and stored["metadata_hash"] == _metadata_hash(t_item):
                pass  # metadata unchanged — skip Phase 4 API calls; post-fetch re-diff still runs
            else:
                candidates.append(t_item)  # metadata changed or legacy doc (no metadata_hash)

        logger.info(
            f"[{run_id}]   ES has {len(existing_hashes)} docs | "
            f"candidates={len(candidates)} | unchanged={len(fetched)-len(candidates)} | "
            f"stale={len(stale_keys)} ({_elapsed(t)})"
        )

        # ── Phase 4: Bulk description + label fetch ───────────────────────
        t = time.monotonic()
        jira_map: dict[str, dict] = {}  # {key: {"description": str, "labels": [str]}}

        if candidates:
            candidate_keys = [_get_key(t_item) for t_item in candidates if _get_key(t_item)]
            n_calls = (len(candidate_keys) + DESC_BULK_SIZE - 1) // DESC_BULK_SIZE
            logger.info(
                f"[{run_id}] Phase 4/6 — Fetching descriptions + labels: "
                f"{len(candidate_keys)} tests in {n_calls} bulk Jira calls ..."
            )

            for i in range(0, len(candidate_keys), DESC_BULK_SIZE):
                batch_keys = candidate_keys[i : i + DESC_BULK_SIZE]
                try:
                    batch_info = await xray_client.get_descriptions_bulk(batch_keys)
                    jira_map.update(batch_info)
                except Exception as e:
                    logger.warning(
                        f"[{run_id}]   Bulk Jira fetch failed for batch "
                        f"{i}–{i+len(batch_keys)}: {e} — using empty fields"
                    )

                fetched_so_far = min(i + DESC_BULK_SIZE, len(candidate_keys))
                logger.info(
                    f"[{run_id}]   descriptions: {fetched_so_far}/{len(candidate_keys)} "
                    f"({_elapsed(t)})"
                )

            # Merge description + labels into candidate dicts
            missing_desc = 0
            for t_item in candidates:
                k = _get_key(t_item)
                info = jira_map.get(k, {})
                t_item["description"] = info.get("description", "")
                if info.get("labels"):          # only overwrite if Jira returned labels
                    t_item["labels"] = info["labels"]
                if not t_item["description"]:
                    missing_desc += 1

            logger.info(
                f"[{run_id}]   Phase 4 done: {len(jira_map)} fetched, "
                f"{missing_desc} empty descriptions ({_elapsed(t)})"
            )

            logger.info(
                f"[{run_id}] Phase 4b — Fetching steps + preconditions for "
                f"{len(candidate_keys)} tests ..."
            )
            t4b = time.monotonic()
            detail_map = await _enrich_with_steps_and_prec(candidate_keys, run_id=str(run_id))
            for t_item in candidates:
                k = _get_key(t_item)
                extra = detail_map.get(k) if k else None
                if not extra:
                    t_item.setdefault("steps_text", "")
                    t_item.setdefault("preconditions", "")
                    continue
                t_item["steps_text"] = extra["steps_text"]
                t_item["preconditions"] = extra["preconditions"]
            logger.info(f"[{run_id}]   Phase 4b done ({_elapsed(t4b)})")
        else:
            logger.info(f"[{run_id}] Phase 4/6 — No candidates, skipping description fetch")

        # Re-compute hash with description + steps + preconditions, drop truly unchanged
        to_upsert: list[tuple[dict, str]] = []
        for t_item in candidates:
            h = _content_hash(t_item)
            k = _get_key(t_item)
            if k not in existing_hashes or existing_hashes[k] != h:
                to_upsert.append((t_item, h))

        logger.info(
            f"[{run_id}]   Final upsert queue: {len(to_upsert)} "
            f"(post-detail diff dropped {len(candidates)-len(to_upsert)} more)"
        )

        # ── Phase 5: Embed + upsert ───────────────────────────────────────
        t = time.monotonic()
        n_batches = (len(to_upsert) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
        logger.info(
            f"[{run_id}] Phase 5/6 — Embedding + upserting: "
            f"{len(to_upsert)} tests in {n_batches} batches of {EMBED_BATCH_SIZE} ..."
        )

        loop     = asyncio.get_running_loop()
        upserted = 0
        for i in range(0, len(to_upsert), EMBED_BATCH_SIZE):
            batch = to_upsert[i : i + EMBED_BATCH_SIZE]
            texts = [_format_for_embedding(t_item, embed_client) for t_item, _ in batch]
            # Run CPU-bound embedding in a thread so the event loop stays responsive.
            # Without this, uvicorn blocks for ~2 min/batch and can't handle signals.
            vectors = await loop.run_in_executor(
                None, embed_client.embed_documents, texts
            )
            docs = [
                _to_es_doc(t_item, h, vec)
                for (t_item, h), vec in zip(batch, vectors)
            ]
            es_store.upsert_test_cases_batch(docs)
            upserted += len(docs)
            _el = time.monotonic() - t
            _rate = upserted / _el if _el > 0 else 0
            _eta = (len(to_upsert) - upserted) / _rate if _rate > 0 else 0
            logger.info(
                "[%s]   embed+index %s/%s (%d%%) batch %s/%s — %.1f tests/s, eta %s",
                run_id, upserted, len(to_upsert), upserted * 100 // max(1, len(to_upsert)),
                i // EMBED_BATCH_SIZE + 1, n_batches, _rate,
                f"{_eta:.0f}s" if _eta < 120 else f"{_eta/60:.1f}m",
            )

        logger.info(f"[{run_id}]   Phase 5 done: {upserted} docs in ES ({_elapsed(t)})")

        # ── Phase 6: Delete stale ─────────────────────────────────────────
        deleted = 0
        if stale_keys:
            logger.info(
                f"[{run_id}] Phase 6/6 — Deleting {len(stale_keys)} stale tests: "
                f"{list(stale_keys)[:5]}{'...' if len(stale_keys) > 5 else ''}"
            )
            es_store.delete_stale_tests(list(stale_keys))
            deleted = len(stale_keys)
        else:
            logger.info(f"[{run_id}] Phase 6/6 — No stale tests to delete")

        # ── Done ──────────────────────────────────────────────────────────
        pg_store.complete_run(run_id, tests_synced=upserted)

        summary = {
            "run_id":    run_id,
            "status":    "completed",
            "project":   project_key,
            "total_in_xray": total_xray,
            "upserted":  upserted,
            "unchanged": len(fetched) - len(to_upsert),
            "deleted":   deleted,
            "elapsed":   _elapsed(t_total),
        }
        await append_entry_async(
            "test_sync",
            str(run_id),
            normalize_json_obj(
                {
                    "project_key": project_key,
                    "tests_synced": upserted,
                    "total_in_xray": total_xray,
                    "deleted": deleted,
                }
            ),
        )
        logger.info("=" * 60)
        logger.info(f"[{run_id}] SYNC COMPLETE  {summary}")
        logger.info("=" * 60)
        return summary

    except Exception as exc:
        logger.error("=" * 60)
        logger.exception(f"[{run_id}] SYNC FAILED: {exc}")
        logger.error("=" * 60)
        pg_store.fail_run(run_id, str(exc))
        raise
