"""
Incremental (diff-aware) PRD analysis.

Instead of re-analysing the entire PRD every time, this module:
  1. Detects which PRD sections changed (comparing old vs new chunks)
  2. Identifies which existing decisions are affected
  3. Only re-analyses changed sections
  4. Carries forward unchanged decisions from the previous run

This cuts LLM cost and latency by 60-90% for iterative PRD updates.

Usage:
    result = await run_incremental_analysis(
        prd_source_id="confluence:12345",
        previous_run_id="uuid-of-last-run",
        ...
    )
"""
import hashlib
import json
import logging
import time
import uuid
from typing import Any

from elasticsearch import helpers as es_helpers

from embeddings.embed_client import EmbedClient
from embeddings.es_store import ESStore
from embeddings.pg_store import PGStore
from observability.canonical_json import fingerprint_sha256, normalize_json_obj
from observability.phase_ledger import append_entry_async
from observability.request_norm import normalize_module_list

logger = logging.getLogger(__name__)


def _chunk_hash(chunk: dict) -> str:
    """Hash a chunk's content for diff detection."""
    payload = f"{chunk.get('section_heading', '')}|{chunk.get('chunk_text', '')}"
    return hashlib.sha256(payload.encode()).hexdigest()


def _fetch_all_prd_chunks(es_store: ESStore, prd_source_id: str) -> list[dict]:
    """All chunks for a PRD source (scroll — no 2000-hit cap)."""
    chunks: list[dict] = []
    for hit in es_helpers.scan(
        es_store._client,
        index="qa_prd_chunks",
        query={"query": {"term": {"source_id": prd_source_id}}},
        scroll="2m",
        _source=["section_heading", "chunk_text", "chunk_index"],
    ):
        chunks.append(hit["_source"])
    chunks.sort(key=lambda c: c.get("chunk_index", 0))
    return chunks


def _heading_content_hashes(chunks: list[dict]) -> dict[str, str]:
    """Group chunks by heading; hash concatenated chunk texts per heading (same as diff)."""
    groups: dict[str, list[str]] = {}
    for c in chunks:
        h = c.get("section_heading") or "(no heading)"
        groups.setdefault(h, []).append(c.get("chunk_text", ""))
    return {
        heading: hashlib.sha256("|".join(texts).encode()).hexdigest()
        for heading, texts in groups.items()
    }


def compute_prd_heading_hashes(es_store: ESStore, prd_source_id: str) -> dict[str, str]:
    """Heading → content hash for the PRD source's chunks in ES (for incremental follow-up runs)."""
    current_chunks = _fetch_all_prd_chunks(es_store, prd_source_id)
    return _heading_content_hashes(current_chunks)


def detect_changes(
    prd_source_id: str,
    es_store: ESStore,
    previous_chunks: list[dict] | None = None,
    previous_heading_hashes: dict[str, str] | None = None,
) -> dict[str, Any]:
    """
    Detect which sections of a PRD changed since the last ingestion.

    Provide exactly one baseline source when comparing:
      — previous_heading_hashes from sync_runs.run_metadata.prd_heading_hashes (preferred), or
      — previous_chunks from an older snapshot.

    If neither is provided, every section is treated as new (first-time analysis).

    Returns:
        {
            "changed_headings": [str],     # sections with modified content
            "new_headings": [str],         # sections that didn't exist before
            "removed_headings": [str],     # sections that no longer exist
            "unchanged_headings": [str],   # sections identical to previous
            "current_chunks": [dict],      # current chunk data (for caching)
        }
    """
    current_chunks = _fetch_all_prd_chunks(es_store, prd_source_id)
    curr_hashes = _heading_content_hashes(current_chunks)

    if previous_heading_hashes is not None:
        prev_hashes = previous_heading_hashes
    elif previous_chunks:
        prev_hashes = _heading_content_hashes(previous_chunks)
    else:
        # No baseline — treat as first analysis (all current sections are "new")
        headings = list(curr_hashes.keys())
        return {
            "changed_headings": [],
            "new_headings": headings,
            "removed_headings": [],
            "unchanged_headings": [],
            "current_chunks": current_chunks,
        }

    prev_set = set(prev_hashes.keys())
    curr_set = set(curr_hashes.keys())

    new_headings = list(curr_set - prev_set)
    removed_headings = list(prev_set - curr_set)
    common = curr_set & prev_set

    changed_headings = [h for h in common if prev_hashes[h] != curr_hashes[h]]
    unchanged_headings = [h for h in common if prev_hashes[h] == curr_hashes[h]]

    return {
        "changed_headings": changed_headings,
        "new_headings": new_headings,
        "removed_headings": removed_headings,
        "unchanged_headings": unchanged_headings,
        "current_chunks": current_chunks,
    }


def carry_forward_decisions(
    pg_store: PGStore,
    previous_run_id: str,
    new_run_id: str,
    unchanged_headings: set[str],
) -> int:
    """
    Copy decisions from the previous run that cover unchanged PRD sections.
    Returns the number of decisions carried forward.
    """
    from agents.analysis_agent import _normalize_heading_for_coverage as _norm

    prev_decisions = pg_store.get_pending_decisions(run_id=previous_run_id)
    carried = 0
    existing = {
        (d.get("jira_key"), _norm(d.get("prd_section") or ""))
        for d in pg_store.get_pending_decisions(run_id=new_run_id)
    }

    # Normalise the comparison on BOTH sides. `prd_section` is agent-authored free text
    # ("3.2 Payment Capture", "Payment capture:") while unchanged_headings comes verbatim
    # from Elasticsearch. A raw `in` test therefore matched almost nothing, so carry-forward
    # silently carried nothing and every "incremental" run re-analysed the whole PRD —
    # which is exactly the cost saving incremental mode exists to deliver.
    unchanged_norm = {_norm(h) for h in unchanged_headings}
    unchanged_norm.discard("")

    skipped_unmatched = 0
    for d in prev_decisions:
        section = d.get("prd_section") or ""
        section_norm = _norm(section)
        if section_norm and section_norm not in unchanged_norm:
            skipped_unmatched += 1
        # If this decision covers an unchanged section, carry it forward
        if section_norm in unchanged_norm:
            pair = (d.get("jira_key"), section_norm)
            if pair in existing:
                continue
            existing.add(pair)
            raw_reason = d.get("reason") or ""
            if "[Carried forward from" in raw_reason and "] " in raw_reason:
                reason_body = raw_reason.split("] ", 1)[-1].strip()
            else:
                reason_body = raw_reason
            cf_reason = f"[Carried forward from {previous_run_id[:8]}] {reason_body}".strip()
            if len(cf_reason) > 2000:
                cf_reason = cf_reason[:2000]
            row: dict[str, Any] = {
                "run_id": new_run_id,
                "jira_key": d.get("jira_key"),
                "action": d["action"],
                "reason": cf_reason,
                "updated_content": d.get("updated_content"),
                "questions": d.get("questions"),
                "prd_source": d.get("prd_source"),
                "prd_section": section,
            }
            # Preserve human review state so unchanged sections are not re-queued for approval.
            if d.get("reviewed"):
                row["reviewed"] = True
                row["approved"] = d.get("approved")
                row["reviewer_note"] = d.get("reviewer_note")
                if d.get("approved") is True:
                    row["written_back"] = bool(d.get("written_back"))
            pg_store.write_decision(row)
            carried += 1

    logger.info(
        "carry-forward: %s of %s previous decisions carried (%s referenced a section that "
        "is not in the unchanged set)",
        carried, len(prev_decisions), skipped_unmatched,
    )
    return carried


async def run_incremental_analysis(
    prd_source_id: str,
    previous_run_id: str,
    module: list[str] | None,
    embed_client: EmbedClient,
    es_store: ESStore,
    pg_store: PGStore,
    run_id: str | None = None,
    provider: str = "azure_openai",
    model: str = "gpt-4o",
    reranker=None,
) -> dict[str, Any]:
    """
    Incremental analysis: only re-analyse changed PRD sections.

    Steps:
      1. Detect changes between current PRD and last analysis snapshot
      2. Carry forward decisions for unchanged sections
      3. Run full analysis ONLY on changed/new sections
      4. Flag decisions for removed sections as potentially stale

    Returns summary with change detection + analysis results.
    """
    from agents.analysis_agent import run_analysis

    run_id = run_id or str(uuid.uuid4())
    started = time.time()

    pg_store.start_run(run_id, run_type="incremental_analysis", prd_source=prd_source_id)

    try:
        prev_row = pg_store.get_run(previous_run_id) or {}
        prev_ps = prev_row.get("prd_source")
        if prev_ps != prd_source_id:
            raise ValueError(
                f"previous_run_id {previous_run_id!r} has prd_source={prev_ps!r}, "
                f"expected {prd_source_id!r}"
            )
        prev_meta = prev_row.get("run_metadata") or {}
        if isinstance(prev_meta, str):
            try:
                prev_meta = json.loads(prev_meta)
            except Exception:
                prev_meta = {}
        prev_hashes = prev_meta.get("prd_heading_hashes") if isinstance(prev_meta, dict) else None

        prev_m = prev_meta.get("module_filter") if isinstance(prev_meta, dict) else None
        cur_norm = normalize_module_list(module)
        prev_norm = normalize_module_list(prev_m) if isinstance(prev_m, list) else None
        module_mismatch = prev_norm != cur_norm
        if module_mismatch:
            logger.warning(
                f"[{run_id}] module_filter changed vs previous run ({prev_norm!r} -> {cur_norm!r}) — "
                "skipping carry-forward; running full analysis"
            )

        diff = detect_changes(
            prd_source_id,
            es_store,
            previous_chunks=None,
            previous_heading_hashes=prev_hashes if isinstance(prev_hashes, dict) else None,
        )

        changed = set(diff["changed_headings"] + diff["new_headings"])
        unchanged = set(diff["unchanged_headings"])
        removed = set(diff["removed_headings"])

        logger.info(
            f"[{run_id}] Incremental diff: "
            f"{len(changed)} changed/new, {len(unchanged)} unchanged, {len(removed)} removed"
        )

        # Step 2: Carry forward unchanged decisions (skip if module filter changed vs previous run)
        carried = 0
        if not module_mismatch and unchanged and previous_run_id:
            carried = carry_forward_decisions(pg_store, previous_run_id, run_id, unchanged)
            logger.info(f"[{run_id}] Carried forward {carried} decisions from unchanged sections")

        focus_for_agent = None if module_mismatch else sorted(changed)

        # Step 3: Run analysis only when needed.
        # If nothing changed and module filter is unchanged, avoid a full re-analysis run.
        if not module_mismatch and not changed:
            logger.info(f"[{run_id}] No changed/new sections — skipping analysis loop")
            analysis_result = {"status": "completed", "skipped_analysis": True}
        else:
            # Full PRD when module filter changed; else scoped to changed/new sections
            analysis_result = await run_analysis(
                prd_source_id=prd_source_id,
                module=module,
                embed_client=embed_client,
                es_store=es_store,
                pg_store=pg_store,
                run_id=run_id,
                provider=provider,
                model=model,
                reranker=reranker,
                focus_headings=focus_for_agent,
                finalize_run=False,
            )

        # Step 4: Flag removed sections
        if removed:
            for heading in removed:
                pg_store.write_decision({
                    "run_id": run_id,
                    "jira_key": None,
                    "action": "question",
                    "reason": f"Section '{heading}' was removed from the PRD. Review tests that referenced this section.",
                    "prd_source": prd_source_id,
                    "prd_section": heading,
                })

        elapsed = round(time.time() - started, 1)

        decisions_final = pg_store.get_pending_decisions(run_id=run_id)
        total_decisions = len(decisions_final)

        inner_out = analysis_result.get("status")
        if inner_out == "truncated":
            row_status = "truncated"
        elif total_decisions == 0:
            row_status = "completed_empty"
            logger.warning(f"[{run_id}] Incremental run finished with zero total decisions")
        else:
            row_status = "completed"

        state_conf = (
            "incremental"
            if isinstance(prev_hashes, dict) and len(prev_hashes) > 0
            else "incremental_baseline_missing"
        )

        hashes = compute_prd_heading_hashes(es_store, prd_source_id)
        run_metadata = normalize_json_obj({
            "prd_heading_hashes": hashes,
            "coverage_score": analysis_result.get("coverage_score"),
            "state_confidence": state_conf,
            "inner_loop_status": analysis_result.get("loop_status"),
            "verification_hybrid_hits": analysis_result.get("verification_hybrid_hits"),
            "module_filter": cur_norm,
            # Propagated from the inner run so incremental cost is comparable to a full run —
            # this is what makes the "60-90% cheaper" claim measurable instead of asserted.
            "token_usage": analysis_result.get("token_usage"),
        })

        decisions_fp = [
            {"action": d.get("action"), "jira_key": d.get("jira_key"), "prd_section": d.get("prd_section")}
            for d in decisions_final
        ]

        pg_store.complete_run(
            run_id,
            decisions_made=total_decisions,
            run_metadata=run_metadata,
            final_status=row_status,
        )
        inc_phase = (
            "incremental_analysis_truncated" if row_status == "truncated" else "incremental_analysis"
        )
        await append_entry_async(
            inc_phase,
            run_id,
            normalize_json_obj({
                "prd_source_id": prd_source_id,
                "previous_run_id": previous_run_id,
                "decisions_made": total_decisions,
                "decisions_sha256": fingerprint_sha256(decisions_fp),
                "sync_status": row_status,
                "state_confidence": state_conf,
                "warning": "max_turns_reached" if inner_out == "truncated" else None,
            }),
        )

        result = {
            "run_id": run_id,
            "status": row_status,
            "mode": "incremental",
            "previous_run_id": previous_run_id,
            "diff": {
                "changed_sections": len(changed),
                "unchanged_sections": len(unchanged),
                "removed_sections": len(removed),
            },
            "decisions_carried_forward": carried,
            "decisions_new": analysis_result.get("decisions_made", 0),
            "decisions_total": total_decisions,
            "elapsed_s": elapsed,
            "state_confidence": state_conf,
            "token_usage": analysis_result.get("token_usage"),
        }
        return result

    except Exception as exc:
        elapsed = round(time.time() - started, 1)
        logger.exception(f"[{run_id}] Incremental analysis failed after {elapsed}s: {exc}")
        pg_store.fail_run(run_id, str(exc))
        raise
