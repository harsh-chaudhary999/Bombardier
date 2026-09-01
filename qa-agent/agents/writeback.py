"""
Phase 5: Write-back approved decisions to Xray/Jira.

After human review (Phase 4), approved decisions are synced back:
  KEEP      → mark written_back (no Xray action)
  UPDATE    → replace summary/steps via xray_client; prose goes to a Jira comment
  DEPRECATE → add DEPRECATED label + comment via xray_client
  CREATE    → bulk-create new tests via xray_client
  QUESTION  → mark written_back (no Xray action)

The UPDATE path is the only one that can lose data. Xray's `updateTestSteps` mutation
replaces a test's whole step array rather than merging, so anything sent there overwrites
the real steps irreversibly. Two rules follow, enforced in `_validated_steps` and the
UPDATE branch:

  * A step list is written only when it validates as a complete list of
    {action, data, expectedResult} objects. One malformed item rejects the whole payload —
    writing the valid prefix would delete every step after it.
  * Prose is never converted into a step. An English recommendation has no step structure
    to recover, so it is posted as a Jira comment for a human to apply, leaving the test's
    structured fields untouched.

On CREATE there are no existing steps to lose, so an *enumerated* outline is parsed into
separate steps; free prose becomes the description instead of one step holding a paragraph.
"""
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote

from integrations import xray_client
from embeddings.pg_store import PGStore

logger = logging.getLogger(__name__)


def _validated_steps(raw: Any) -> list[dict] | None:
    """
    Accept only a genuinely structured step list; return None for anything else.

    Xray's `updateTestSteps` mutation **replaces** the whole step array — it does not merge.
    So a malformed, partial or synthesised payload here does not degrade a test, it destroys
    it: a 12-step manual test becomes whatever single item we sent. There is no undo.

    Hence this is deliberately strict. Every item must be a dict carrying a non-empty
    `action`; one bad item rejects the entire list rather than writing a truncated one.
    """
    if not isinstance(raw, list) or not raw:
        return None
    out: list[dict] = []
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            return None
        action = item.get("action")
        if not isinstance(action, str) or not action.strip():
            return None
        out.append({
            "action": action.strip(),
            "data": str(item.get("data") or "").strip(),
            "expectedResult": str(item.get("expectedResult") or item.get("result") or "").strip(),
            "index": i,
        })
    return out


# Matches an enumerated line: "1. x", "2) x", "- x", "* x", "• x", "Step 3: x"
_OUTLINE_RE = re.compile(r"^\s*(?:\d+\s*[.)\-:]|[-*•]|step\s*\d+\s*[:.)\-])\s*(.+)$", re.IGNORECASE)
# Splits an expected result off a step line: "do x -> y", "do x => y", "do x | Expected: y"
_EXPECTED_RE = re.compile(r"\s*(?:->|=>|→|\|\s*expected\s*:|\bexpected\s*:)\s*", re.IGNORECASE)


def _steps_from_outline(text: Any) -> list[dict] | None:
    """
    Split an *enumerated* outline into steps. Returns None for free-flowing prose.

    Only used when creating a new test, where there are no existing steps to lose. The
    enumeration requirement is the safeguard: if the model wrote a paragraph rather than a
    list, we must not pretend it is a step — the caller stores it as the description
    instead, which is honest about what we actually have.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    items: list[tuple[str, str]] = []
    for line in text.splitlines():
        m = _OUTLINE_RE.match(line)
        if not m:
            continue
        body = m.group(1).strip()
        if not body:
            continue
        parts = _EXPECTED_RE.split(body, maxsplit=1)
        items.append((parts[0].strip(), parts[1].strip() if len(parts) > 1 else ""))
    if not items:
        return None
    return [
        {"action": a, "data": "", "expectedResult": e, "index": i}
        for i, (a, e) in enumerate(items, start=1)
    ]


async def _pre_deprecation_snapshot(jira_key: str) -> dict | None:
    """
    Capture what deprecation is about to change, so it can be put back.

    Only labels: `deprecate_test` appends a DEPRECATED label and adds a comment, and
    changes nothing else. Snapshotting the folder or module would be recording state
    that was never touched, and a rollback that "restored" them could move a test that
    someone had legitimately re-filed since.

    Returns None if the labels could not be read — the caller records that the decision
    is not rollback-able rather than failing the run.
    """
    try:
        labels = await xray_client.get_labels(jira_key)
    except Exception as exc:
        logger.warning("Could not snapshot labels for %s before deprecating: %s",
                       jira_key, exc)
        return None
    return {
        "jira_key": jira_key,
        "labels": labels,
        "captured_at": datetime.now(timezone.utc).isoformat(),
    }


def _question_is_commentable(decision: dict) -> bool:
    """A question can only be posted where there is a test to post it on."""
    return decision.get("action") == "question" and bool(decision.get("jira_key"))


def _question_comment_body(
    reason: str | None,
    run_id: str | None,
    prd_source: str | None,
    prd_section: str | None,
) -> str:
    """
    Render an open question as a Jira comment.

    Carries the run and PRD section so the reader can find what prompted it months
    later, and says plainly that nothing was changed — a comment on a test case
    otherwise reads as though something was done to it.
    """
    body = [
        "*[QA Intelligence Engine — open question]*",
        "",
        (reason or "").strip() or "(no reason recorded)",
        "",
    ]
    context = []
    if prd_source:
        context.append(f"PRD: {prd_source}")
    if prd_section:
        context.append(f"Section: {prd_section}")
    if run_id:
        context.append(f"Run: {run_id}")
    if context:
        body.append("_" + " | ".join(context) + "_")
        body.append("")
    body.append(
        "No change has been made to this test. Please confirm whether it is still "
        "valid, then update the test or the PRD."
    )
    return "\n".join(body)


def _resolve_prd_remote_url(prd_source: str, es_store: Any) -> str | None:
    """
    URL for Jira remote links — prefer indexed doc_url; otherwise build from env + source id.

    Stored prd_source values are typically type:id (e.g. confluence:12345), not HTTP URLs.
    """
    if not prd_source:
        return None
    ps = prd_source.strip()
    if ps.startswith(("http://", "https://")):
        return ps

    if es_store is not None:
        try:
            r = es_store._client.search(
                index="qa_prd_chunks",
                query={"term": {"source_id": ps}},
                source=["doc_url"],
                size=1,
            )
            hits = r["hits"]["hits"]
            if hits:
                u = (hits[0].get("_source") or {}).get("doc_url")
                if u:
                    return str(u)
        except Exception as ex:
            logger.warning("Could not resolve doc_url from ES for %s: %s", prd_source, ex)

    domain = (os.environ.get("CONFLUENCE_DOMAIN") or "").strip().strip("/")
    if ps.startswith("confluence:") and domain:
        page_id = ps.split(":", 1)[1].strip()
        if page_id.isdigit():
            return f"https://{domain}/wiki/pages/viewpage.action?pageId={page_id}"

    raw_host = (os.environ.get("GITLAB_HOST") or "").strip().rstrip("/")
    if raw_host.startswith(("http://", "https://")):
        host = raw_host.split("://", 1)[1].split("/", 1)[0]
        scheme = "https" if raw_host.startswith("https") else "http"
    elif raw_host:
        host = raw_host.split("/", 1)[0]
        scheme = "https"
    else:
        host = ""
        scheme = "https"

    proj = (os.environ.get("GITLAB_PROJECT_ID") or "").strip().strip("/")
    if ps.startswith("gitlab_file:") and host and proj:
        path = ps.split(":", 1)[1].strip().lstrip("/")
        enc = quote(path, safe="/")
        ref = os.environ.get("GITLAB_DEFAULT_REF", "main")
        return f"{scheme}://{host}/{proj}/-/blob/{ref}/{enc}"

    if ps.startswith("gitlab:") and host and proj:
        path = ps.split(":", 1)[1].strip().lstrip("/")
        enc = quote(path, safe="/")
        ref = os.environ.get("GITLAB_DEFAULT_REF", "main")
        return f"{scheme}://{host}/{proj}/-/blob/{ref}/{enc}"

    logger.info("No remote URL resolved for prd_source=%r (set ES chunks or CONFLUENCE_DOMAIN / GITLAB_*)", ps)
    return None


async def run_writeback(
    pg_store: PGStore,
    run_id: str | None = None,
    project_key: str = "",
    dry_run: bool = False,
    es_store: Any | None = None,
) -> dict[str, Any]:
    """
    Write back all approved, not-yet-written-back decisions to Xray.

    If run_id is provided, only writes back decisions for that run.
    Otherwise (when QA_WRITEBACK_ALLOW_GLOBAL=1), processes all approved pending write-backs.

    Streams decisions in SQL batches — does not load the full decision set into memory.

    Returns a summary dict with counts per action and any errors.
    """
    started = time.time()

    allow_global = os.environ.get("QA_WRITEBACK_ALLOW_GLOBAL", "") == "1"
    if run_id is None and not allow_global:
        return {
            "status": "error",
            "message": (
                "run_id is required for write-back. "
                "Set QA_WRITEBACK_ALLOW_GLOBAL=1 only if you intend to process every approved decision."
            ),
            "total": 0,
            "written_back": {},
            "errors": [
                {
                    "error": (
                        "run_id required — refusing unbounded global write-back "
                        "(set QA_WRITEBACK_ALLOW_GLOBAL=1 to override)"
                    ),
                }
            ],
        }

    def _peek_nonempty() -> bool:
        for b in pg_store.iter_writeback_decisions(run_id, batch_size=200):
            return len(b) > 0
        return False

    if not _peek_nonempty():
        return {
            "status": "completed",
            "message": "No approved decisions pending write-back",
            "total": 0,
            "written_back": {},
            "errors": [],
        }

    if dry_run:
        by_action: dict[str, int] = {}
        dry_sample: list[dict] = []
        total = 0
        for batch in pg_store.iter_writeback_decisions(run_id, batch_size=200):
            for d in batch:
                total += 1
                a = d.get("action") or "?"
                by_action[a] = by_action.get(a, 0) + 1
                if len(dry_sample) < 500:
                    dry_sample.append({
                        "id": d.get("id"),
                        "action": d.get("action"),
                        "jira_key": d.get("jira_key"),
                        "prd_section": (d.get("prd_section") or "")[:120],
                    })
        return {
            "status": "dry_run",
            "message": "No Xray/Jira calls made — preview only",
            "total": total,
            "by_action": by_action,
            "decisions": dry_sample,
            "elapsed_s": round(time.time() - started, 1),
            "errors": [],
        }

    logger.info(
        "Writing back approved decisions (streamed)"
        + (f" for run {run_id}" if run_id else " (global)")
    )

    counts: dict[str, int] = {"keep": 0, "update": 0, "deprecate": 0, "create": 0, "question": 0}
    errors: list[dict] = []
    # (decision_id, payload, prd_source, reason) — the reason travels with the entry
    # because the drain loop below runs outside the per-decision scope.
    create_queue: list[tuple[int, dict, str, str]] = []

    processed = 0
    for batch in pg_store.iter_writeback_decisions(run_id, batch_size=200):
        for decision in batch:
            processed += 1
            action = decision["action"]
            decision_id = decision["id"]
            jira_key = decision.get("jira_key")
            reason = decision.get("reason", "")
            content = decision.get("updated_content") or {}
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except Exception:
                    content = {}

            try:
                if action == "keep":
                    # Nothing to write: the test already matches the PRD.
                    pg_store.mark_written_back(decision_id)
                    counts["keep"] += 1

                elif action == "question":
                    # A question with nowhere to go is a dead end. Post it on the test
                    # so the QA owner sees it in Jira rather than only in the review UI.
                    if jira_key:
                        try:
                            await xray_client.add_comment(
                                jira_key,
                                _question_comment_body(
                                    reason=reason,
                                    run_id=run_id,
                                    prd_source=decision.get("prd_source"),
                                    prd_section=decision.get("prd_section"),
                                ),
                            )
                            logger.info("  Commented question on %s", jira_key)
                        except Exception as e:
                            # Record the failure but still mark it written back — the
                            # decision is reviewed and approved; a comment outage must
                            # not strand it for re-processing on every later run.
                            errors.append({
                                "decision_id": decision_id,
                                "jira_key": jira_key,
                                "action": "question",
                                "error": f"could not post question comment: {e}",
                            })
                    pg_store.mark_written_back(decision_id)
                    counts["question"] += 1

                elif action == "update":
                    if not jira_key:
                        errors.append({"decision_id": decision_id, "error": "UPDATE decision missing jira_key"})
                        continue
                    summary = (content.get("summary") or "").strip() or None
                    raw_steps = content.get("steps")
                    steps = _validated_steps(raw_steps)
                    suggested = (content.get("suggested_changes") or "").strip()

                    # A steps payload we cannot validate must never reach Xray. Sending it
                    # would replace the test's real steps with a malformed list.
                    if raw_steps and steps is None:
                        errors.append({
                            "decision_id": decision_id,
                            "jira_key": jira_key,
                            "action": "update",
                            "error": (
                                "steps payload is not a valid list of {action,data,expectedResult} "
                                "objects — refusing to replace the existing steps"
                            ),
                        })
                        continue

                    if not (summary or steps or suggested):
                        errors.append({
                            "decision_id": decision_id,
                            "jira_key": jira_key,
                            "action": "update",
                            "error": "UPDATE decision missing suggested_changes/steps/summary payload",
                        })
                        continue

                    applied: list[str] = []
                    if summary or steps:
                        await xray_client.update_test(jira_key, summary=summary, steps=steps)
                        if summary:
                            applied.append("summary")
                        if steps:
                            applied.append(f"{len(steps)} steps")

                    # Prose is a recommendation, not a step list. Turning it into a step was
                    # what destroyed test content; it goes to a comment so the reviewer sees
                    # it and the structured fields stay intact.
                    if suggested:
                        await xray_client.add_comment(
                            jira_key,
                            "[QA Intelligence Engine] Suggested update — review and apply manually.\n\n"
                            f"{suggested}"
                            + (f"\n\nReason: {reason}" if reason else ""),
                        )
                        applied.append("comment")

                    pg_store.mark_written_back(decision_id)
                    pg_store.record_ancestry(
                        jira_key=jira_key, run_id=run_id or "", change_type="updated",
                        prd_source=decision.get("prd_source"), reason_summary=reason,
                        decision_id=decision_id,
                    )
                    counts["update"] += 1
                    logger.info("  Updated %s (%s)", jira_key, ", ".join(applied))

                elif action == "deprecate":
                    if not jira_key:
                        errors.append({"decision_id": decision_id, "error": "DEPRECATE decision missing jira_key"})
                        continue

                    # Snapshot BEFORE the change, and persist it before making the change.
                    # Ordering matters: a crash between the two leaves a snapshot with no
                    # deprecation, which is harmless. The reverse — a deprecation with no
                    # snapshot — is unrecoverable, and deprecation is the one irreversible
                    # action this pipeline takes.
                    snapshot = await _pre_deprecation_snapshot(jira_key)
                    if snapshot:
                        pg_store.merge_decision_updated_content(
                            decision_id, {"pre_deprecation_snapshot": snapshot})
                    else:
                        # Recorded, not fatal: refusing to deprecate because the labels
                        # could not be read would block the whole run on a read failure.
                        errors.append({
                            "decision_id": decision_id, "jira_key": jira_key,
                            "action": "deprecate",
                            "error": "could not snapshot labels before deprecating — "
                                     "this decision is not rollback-able",
                        })

                    await xray_client.deprecate_test(jira_key, reason)
                    pg_store.mark_written_back(decision_id)
                    pg_store.record_ancestry(
                        jira_key=jira_key, run_id=run_id or "", change_type="deprecated",
                        prd_source=decision.get("prd_source"), reason_summary=reason,
                        decision_id=decision_id,
                    )
                    counts["deprecate"] += 1
                    logger.info(f"  Deprecated {jira_key}")

                elif action == "create":
                    if content.get("created_jira_key"):
                        pg_store.mark_written_back(decision_id)
                        counts["create"] += 1
                        logger.info(
                            "  CREATE decision %s already has created_jira_key=%s — marking written_back",
                            decision_id,
                            content.get("created_jira_key"),
                        )
                        continue
                    new_test = {
                        "summary": content.get("summary", "New test case"),
                        "testType": "Manual",
                    }
                    # Nothing exists yet, so there is nothing to destroy — but a paragraph
                    # masquerading as a single step still makes a useless test. Prefer
                    # structured steps, then an enumerated outline, then the description.
                    structured = _validated_steps(content.get("steps"))
                    outline = content.get("suggested_steps")
                    if structured:
                        new_test["steps"] = structured
                    elif outline and str(outline).strip():
                        parsed = _steps_from_outline(outline)
                        if parsed:
                            new_test["steps"] = parsed
                        else:
                            new_test["description"] = str(outline).strip()
                            logger.info(
                                "  CREATE %r: step outline is prose, not an enumerated list — "
                                "storing it as the description instead of a single bogus step",
                                new_test["summary"],
                            )
                    prd_src = decision.get("prd_source") or ""
                    create_queue.append((decision_id, new_test, prd_src, reason))

                else:
                    errors.append({"decision_id": decision_id, "error": f"Unknown action: {action}"})

            except Exception as e:
                logger.warning(f"  Write-back failed for decision {decision_id} ({action} {jira_key}): {e}")
                errors.append({
                    "decision_id": decision_id,
                    "jira_key": jira_key,
                    "action": action,
                    "error": str(e),
                })

    logger.info("Processed %s write-back decision rows from DB batches", processed)

    # Batch create new tests (up to 50 at a time)
    if create_queue and project_key:
        BATCH = 50
        for i in range(0, len(create_queue), BATCH):
            batch = create_queue[i:i + BATCH]
            tests = [t for _, t, _, _ in batch]
            ids = [did for did, _, _, _ in batch]
            try:
                result = await xray_client.bulk_create_tests(project_key, tests)
                created_keys: list[str] = []
                if isinstance(result, dict):
                    created_keys = result.get("keys") or result.get("createdKeys") or []
                elif isinstance(result, list):
                    created_keys = [r.get("key") for r in result if isinstance(r, dict) and r.get("key")]

                for j, (did, _, prd_source, create_reason) in enumerate(batch):
                    key = created_keys[j] if j < len(created_keys) else None
                    if key is None:
                        errors.append({
                            "decision_id": did,
                            "action": "create",
                            "error": "Xray did not return a key for this test in the bulk response",
                        })
                        continue
                    if key:
                        merged = pg_store.merge_decision_updated_content(
                            did, {"created_jira_key": key}
                        )
                        if not merged:
                            logger.error(
                                "Could not persist created_jira_key for decision %s — retry may duplicate",
                                did,
                            )
                    ok = pg_store.mark_written_back(did)
                    if not ok:
                        logger.error(
                            "mark_written_back failed for decision %s after successful Xray create",
                            did,
                        )
                    if key:
                        pg_store.record_ancestry(
                            jira_key=key, run_id=run_id or "", change_type="created",
                            prd_source=prd_source,
                            reason_summary=create_reason,
                            decision_id=did,
                        )
                    counts["create"] += 1
                    if key and prd_source:
                        link_url = _resolve_prd_remote_url(prd_source, es_store)
                        if link_url:
                            try:
                                await xray_client.add_remote_link(
                                    key,
                                    url=link_url,
                                    title=f"PRD: {prd_source}",
                                )
                            except Exception as link_err:
                                logger.warning(
                                    "  Remote link failed for %s → %s: %s",
                                    key,
                                    prd_source,
                                    link_err,
                                )
                        else:
                            logger.warning(
                                "  Skipping remote link for %s — could not resolve URL for %s",
                                key,
                                prd_source,
                            )

                logger.info(f"  Created {len(batch)} new tests (batch {i // BATCH + 1})")
            except Exception as e:
                logger.warning(f"  Bulk create failed for batch {i // BATCH + 1}: {e}")
                for did in ids:
                    errors.append({"decision_id": did, "action": "create", "error": str(e)})
    elif create_queue and not project_key:
        for did, _, _ in create_queue:
            errors.append({
                "decision_id": did,
                "action": "create",
                "error": "project_key required for CREATE actions",
            })

    elapsed = round(time.time() - started, 1)
    total_written = sum(counts.values())

    logger.info(f"Write-back complete: {total_written} written, {len(errors)} errors, {elapsed}s")

    return {
        "status": "completed" if not errors else "completed_with_errors",
        "total": total_written,
        "written_back": counts,
        "errors": errors,
        "elapsed_s": elapsed,
    }
