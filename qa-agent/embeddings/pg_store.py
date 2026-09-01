"""
Postgres relational store for the QA Intelligence Engine.

Handles ONLY relational data — no vectors, no embeddings.
Vector storage (test cases, PRD chunks) lives in Elasticsearch (es_store.py).

Tables:
  qa_rag.pending_decisions — agent decisions awaiting human review
  qa_rag.sync_runs         — audit log for sync and pipeline runs
"""
import concurrent.futures
import logging
import os

import psycopg2
import psycopg2.extras
import psycopg2.pool

logger = logging.getLogger(__name__)

_POOL_GET_TIMEOUT = float(os.environ.get("PG_POOL_GETCONN_TIMEOUT_SEC", "10"))

_VALID_CONFIDENCE = ("high", "medium", "low")


def _coerce_confidence(value) -> str | None:
    """
    Normalise an agent-supplied confidence to the values the CHECK constraint allows.

    Anything unrecognised becomes NULL. The column is a triage aid; a model that
    answers "very high" or "3" must not cost us the decision it was attached to.
    Case and surrounding whitespace are forgiven because they carry no meaning.
    """
    if value is None:
        return None
    text = str(value).strip().lower()
    return text if text in _VALID_CONFIDENCE else None


def _build_pool() -> psycopg2.pool.ThreadedConnectionPool:
    """Create a thread-safe connection pool (2–10 connections)."""
    return psycopg2.pool.ThreadedConnectionPool(
        minconn=2,
        maxconn=10,
        host=os.environ["POSTGRES_HOST"],
        port=int(os.environ.get("POSTGRES_PORT", 5432)),
        dbname=os.environ["POSTGRES_DB"],
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
    )


class PGStore:
    """Postgres client for relational QA Intelligence Engine data."""

    def __init__(self) -> None:
        self._pool = _build_pool()
        self._pool_get_executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)

    def _get_conn(self):
        """Wait up to PG_POOL_GETCONN_TIMEOUT_SEC for a pool connection (avoids unbounded hangs)."""
        fut = self._pool_get_executor.submit(self._pool.getconn)
        try:
            return fut.result(timeout=_POOL_GET_TIMEOUT)
        except concurrent.futures.TimeoutError:
            logger.error("Postgres pool getconn timed out after %ss", _POOL_GET_TIMEOUT)
            raise TimeoutError(
                f"No Postgres connection available within {_POOL_GET_TIMEOUT}s"
            ) from None

    def _put_conn(self, conn):
        self._pool.putconn(conn)

    # ─── Pending decisions ─────────────────────────────────────────────────────

    def write_decision(self, decision: dict) -> int:
        """
        Insert a single agent decision. Returns the new row id.

        An unrecognised confidence value is stored as NULL rather than rejected —
        the decision itself is worth more than the metadata, and a CHECK violation
        here would lose it. See _coerce_confidence.
        """
        prd_section = decision.get("prd_section")
        if isinstance(prd_section, str) and len(prd_section) > 500:
            prd_section = prd_section[:500]
        reason = decision.get("reason")
        if isinstance(reason, str) and len(reason) > 2000:
            reason = reason[:2000]
        reviewed = decision.get("reviewed")
        if reviewed is None:
            reviewed = False
        written_back = decision.get("written_back")
        if written_back is None:
            written_back = False
        confidence = _coerce_confidence(decision.get("confidence"))
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO qa_rag.pending_decisions
                        (run_id, jira_key, action, reason, updated_content,
                         questions, prd_source, prd_section, confidence,
                         reviewed, approved, reviewer_note, written_back, reviewed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            CASE WHEN COALESCE(%s, FALSE) THEN NOW() ELSE NULL END)
                    RETURNING id;
                    """,
                    (
                        decision["run_id"],
                        decision["jira_key"],
                        decision["action"],
                        reason,
                        psycopg2.extras.Json(decision.get("updated_content")),
                        psycopg2.extras.Json(decision.get("questions")),
                        decision.get("prd_source"),
                        prd_section,
                        confidence,
                        reviewed,
                        decision.get("approved"),
                        decision.get("reviewer_note"),
                        written_back,
                        reviewed,
                    ),
                )
                row_id = cur.fetchone()[0]
            conn.commit()
            return row_id
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def merge_decision_updated_content(self, decision_id: int, patch: dict) -> bool:
        """Merge JSON keys into pending_decisions.updated_content (for post-create bookkeeping)."""
        if not patch:
            return True
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE qa_rag.pending_decisions
                    SET updated_content = COALESCE(updated_content, '{}'::jsonb) || %s::jsonb
                    WHERE id = %s;
                    """,
                    (psycopg2.extras.Json(patch), decision_id),
                )
                ok = cur.rowcount > 0
            conn.commit()
            return ok
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def iter_writeback_decisions(
        self,
        run_id: str | None,
        batch_size: int = 200,
    ):
        """
        Yield batches of rows that are approved for write-back (reviewed, approved, not written_back).
        Ordered by id for stable retries.
        """
        batch_size = max(1, min(batch_size, 500))
        last_id = 0  # keyset pagination — safe when written_back is mutated between batches
        while True:
            conn = self._get_conn()
            try:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    if run_id:
                        cur.execute(
                            """
                            SELECT * FROM qa_rag.pending_decisions
                            WHERE run_id = %s
                              AND reviewed = TRUE AND approved = TRUE AND written_back = FALSE
                              AND id > %s
                            ORDER BY id
                            LIMIT %s;
                            """,
                            (run_id, last_id, batch_size),
                        )
                    else:
                        cur.execute(
                            """
                            SELECT * FROM qa_rag.pending_decisions
                            WHERE reviewed = TRUE AND approved = TRUE AND written_back = FALSE
                              AND id > %s
                            ORDER BY id
                            LIMIT %s;
                            """,
                            (last_id, batch_size),
                        )
                    rows = [dict(r) for r in cur.fetchall()]
            finally:
                self._put_conn(conn)
            if not rows:
                break
            last_id = rows[-1]["id"]
            yield rows
            if len(rows) < batch_size:
                break

    def fail_orphaned_running_runs(self, message: str = "service_restart") -> int:
        """Mark stale running rows failed after unclean shutdown (OOM, kill -9)."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE qa_rag.sync_runs
                    SET status = 'failed',
                        error_message = %s,
                        completed_at = NOW()
                    WHERE status = 'running';
                    """,
                    (message,),
                )
                n = cur.rowcount
            conn.commit()
            if n:
                logger.warning("Marked %s orphaned sync_runs rows as failed (%s)", n, message)
            return n
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def get_pending_decisions(self, run_id: str | None = None) -> list[dict]:
        """Fetch decisions, optionally filtered by run_id. Defaults to unreviewed only."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                if run_id:
                    cur.execute(
                        "SELECT * FROM qa_rag.pending_decisions WHERE run_id = %s ORDER BY created_at;",
                        (run_id,),
                    )
                else:
                    cur.execute(
                        "SELECT * FROM qa_rag.pending_decisions WHERE reviewed = FALSE ORDER BY created_at;"
                    )
                return [dict(r) for r in cur.fetchall()]
        finally:
            self._put_conn(conn)

    def decision_counts_by_run(self, run_id: str) -> dict[str, int]:
        """Counts per action for an analysis run (full run, not one page)."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT action, COUNT(*)::bigint AS n
                    FROM qa_rag.pending_decisions
                    WHERE run_id = %s
                    GROUP BY action;
                    """,
                    (run_id,),
                )
                rows = cur.fetchall()
            out = {"keep": 0, "update": 0, "deprecate": 0, "create": 0, "question": 0}
            for action, n in rows:
                if action in out:
                    out[action] = int(n)
            return out
        finally:
            self._put_conn(conn)

    def get_pending_decisions_page(
        self, run_id: str, limit: int = 50, offset: int = 0
    ) -> tuple[list[dict], int]:
        """Fetch one page of decisions for a run and total count. Ordered by created_at."""
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT COUNT(*) AS n FROM qa_rag.pending_decisions WHERE run_id = %s;",
                    (run_id,),
                )
                total = int(cur.fetchone()["n"])
                cur.execute(
                    """
                    SELECT * FROM qa_rag.pending_decisions
                    WHERE run_id = %s
                    ORDER BY created_at
                    LIMIT %s OFFSET %s;
                    """,
                    (run_id, limit, offset),
                )
                return [dict(r) for r in cur.fetchall()], total
        finally:
            self._put_conn(conn)

    def approve_decision(
        self,
        decision_id: int,
        approved: bool,
        reviewer_note: str | None = None,
        reviewed_by: str | None = None,
    ) -> bool:
        """Mark a decision as reviewed (approved or rejected). Returns True if row existed."""
        conn = self._get_conn()
        try:
            result_exists = False
            result_updated = False
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE qa_rag.pending_decisions
                    SET reviewed = TRUE, approved = %s, reviewer_note = %s,
                        reviewed_at = NOW(), reviewed_by = COALESCE(%s, reviewed_by)
                    WHERE id = %s AND reviewed = FALSE
                    RETURNING id;
                    """,
                    (approved, reviewer_note, reviewed_by, decision_id),
                )
                updated_row = cur.fetchone()
                if updated_row:
                    result_exists = True
                    result_updated = True
                else:
                    # Distinguish not-found from already-reviewed for idempotent behavior.
                    cur.execute(
                        "SELECT 1 FROM qa_rag.pending_decisions WHERE id = %s;",
                        (decision_id,),
                    )
                    result_exists = cur.fetchone() is not None
            conn.commit()
            if not result_exists:
                logger.warning(f"approve_decision: no row found with id={decision_id}")
            if result_updated:
                return True
            # Already reviewed — idempotent success.
            return result_exists
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def approve_decisions_batch(
        self,
        items: list[tuple[int, bool]],
        reviewer_note: str | None = None,
        reviewed_by: str | None = None,
    ) -> int:
        """Approve/reject many pending decisions in one transaction. Returns rows updated."""
        if not items:
            return 0
        approve_ids = [i for i, a in items if a]
        reject_ids = [i for i, a in items if not a]
        conn = self._get_conn()
        updated_total = 0
        try:
            with conn.cursor() as cur:
                for ids, appr in ((approve_ids, True), (reject_ids, False)):
                    if not ids:
                        continue
                    cur.execute(
                        """
                        UPDATE qa_rag.pending_decisions
                        SET reviewed = TRUE, approved = %s, reviewer_note = %s,
                            reviewed_at = NOW(), reviewed_by = COALESCE(%s, reviewed_by)
                        WHERE id = ANY(%s) AND reviewed = FALSE;
                        """,
                        (appr, reviewer_note, reviewed_by, ids),
                    )
                    updated_total += cur.rowcount
            conn.commit()
            return updated_total
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def get_decisions_by_jira_key(self, jira_key: str, limit: int = 10) -> list[dict]:
        """Return the most recent decisions for a specific jira_key across all runs."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, run_id, action, reason, approved, reviewed,
                           reviewer_note, prd_section, created_at
                    FROM qa_rag.pending_decisions
                    WHERE jira_key = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (jira_key, max(1, min(limit, 50))),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            self._put_conn(conn)

    def append_ledger_entry(
        self, phase: str, run_id: str, summary: dict, summary_sha256: str
    ) -> bool:
        """
        Append one phase-ledger entry. Returns True if it was written.

        Never raises: the ledger records that work happened, and losing an audit row
        must not fail the work itself. A False return is the caller's cue to fall back
        to the file.
        """
        try:
            conn = self._get_conn()
        except Exception as exc:
            logger.warning("Phase ledger insert skipped (no connection): %s", exc)
            return False
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO qa_rag.phase_ledger
                        (phase, run_id, summary_sha256, summary)
                    VALUES (%s, %s, %s, %s);
                    """,
                    (phase, str(run_id), summary_sha256, psycopg2.extras.Json(summary)),
                )
            conn.commit()
            return True
        except Exception as exc:
            conn.rollback()
            logger.warning("Phase ledger insert failed (%s/%s): %s", phase, run_id, exc)
            return False
        finally:
            self._put_conn(conn)

    def archive_old_decisions(self, retention_days: int = 180, batch_size: int = 1000) -> int:
        """
        Move written-back decisions older than `retention_days` into the archive table.

        Only written-back rows: anything still awaiting review or write-back is live
        work, however old. (An old unreviewed decision is the SLA report's problem, not
        the archiver's — archiving it would hide exactly what that report exists to show.)

        One statement, so the delete and the insert cannot half-happen. Batched, so a
        first run against years of history does not take a long transaction.
        Returns rows archived.
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH due AS (
                        SELECT id FROM qa_rag.pending_decisions
                         WHERE written_back = TRUE
                           AND created_at < NOW() - (%s * INTERVAL '1 day')
                         ORDER BY created_at
                         LIMIT %s
                         FOR UPDATE SKIP LOCKED
                    ), moved AS (
                        DELETE FROM qa_rag.pending_decisions p
                         USING due WHERE p.id = due.id
                        RETURNING p.*
                    )
                    INSERT INTO qa_rag.pending_decisions_archive
                    SELECT * FROM moved;
                    """,
                    (retention_days, batch_size),
                )
                moved = cur.rowcount
            conn.commit()
            if moved:
                logger.info("Archived %s decision(s) older than %s days",
                            moved, retention_days)
            return max(0, moved)
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def get_approval_rate(
        self, action: str, min_samples: int = 20
    ) -> tuple[float, int]:
        """
        Historical (approval_rate, sample_size) for one action across reviewed decisions.

        Only reviewed rows count — an unreviewed decision is not evidence either way.
        Returns (0.0, n) when there is not enough history: the caller must not read a
        high rate off three samples.
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) FILTER (WHERE approved IS TRUE)::float
                             / NULLIF(COUNT(*), 0),
                           COUNT(*)
                      FROM qa_rag.pending_decisions
                     WHERE action = %s AND reviewed = TRUE AND approved IS NOT NULL;
                    """,
                    (action,),
                )
                row = cur.fetchone()
                total = int(row[1] or 0) if row else 0
                if total < min_samples:
                    return 0.0, total
                return float(row[0] or 0.0), total
        finally:
            self._put_conn(conn)

    def record_ancestry(
        self,
        jira_key: str,
        run_id: str,
        change_type: str,
        prd_source: str | None = None,
        reason_summary: str | None = None,
        decision_id: int | None = None,
    ) -> None:
        """
        Append one write-back to a test's history.

        Never raises. This is an audit trail beside the real work: losing a history row
        is regrettable, failing a write-back that already reached Xray because the
        history insert failed is worse — the two would then disagree permanently.
        """
        try:
            conn = self._get_conn()
        except Exception as exc:
            logger.warning("Ancestry not recorded for %s: %s", jira_key, exc)
            return
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO qa_rag.test_ancestry
                        (jira_key, run_id, prd_source, change_type, reason_summary, decision_id)
                    VALUES (%s, %s::uuid, %s, %s, %s, %s);
                    """,
                    (jira_key, run_id, prd_source, change_type,
                     (reason_summary or "")[:500] or None, decision_id),
                )
            conn.commit()
        except Exception as exc:
            conn.rollback()
            logger.warning("Ancestry not recorded for %s (%s): %s",
                           jira_key, change_type, exc)
        finally:
            self._put_conn(conn)

    def get_test_ancestry(self, jira_key: str, limit: int = 50) -> list[dict]:
        """One test's write-back history, newest first."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, run_id, prd_source, change_type, reason_summary,
                           decision_id, changed_at
                      FROM qa_rag.test_ancestry
                     WHERE jira_key = %s
                     ORDER BY changed_at DESC, id DESC
                     LIMIT %s;
                    """,
                    (jira_key, limit),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            self._put_conn(conn)

    def get_last_analysis_run(self, prd_source: str) -> dict | None:
        """
        The most recent finished analysis of a PRD, or None if it has never been analysed.

        `truncated` counts as finished: the run hit its turn limit but produced real
        decisions, and those are worth diffing against rather than discarding.
        Ingest runs are excluded — they carry the same prd_source but analysed nothing.
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT run_id, run_type, status, completed_at
                      FROM qa_rag.sync_runs
                     WHERE prd_source = %s
                       AND run_type IN ('analysis', 'incremental_analysis')
                       AND status IN ('completed', 'truncated', 'completed_with_errors')
                     ORDER BY completed_at DESC NULLS LAST
                     LIMIT 1;
                    """,
                    (prd_source,),
                )
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            self._put_conn(conn)

    def get_overdue_decisions(self, days: int = 30, limit: int = 200) -> list[dict]:
        """
        Unreviewed decisions older than `days`, oldest first.

        The deadline is computed here rather than stored, so the SLA window is a
        parameter and not a schema change. Only unreviewed rows can be overdue: a
        decision that was reviewed and rejected is finished, not late.
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, run_id, jira_key, action, confidence,
                           prd_source, prd_section, created_at,
                           EXTRACT(EPOCH FROM (NOW() - created_at)) / 86400.0 AS age_days
                      FROM qa_rag.pending_decisions
                     WHERE reviewed = FALSE
                       AND created_at < NOW() - (%s * INTERVAL '1 day')
                     ORDER BY created_at ASC
                     LIMIT %s;
                    """,
                    (days, limit),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            self._put_conn(conn)

    def count_overdue_decisions(self, days: int = 30) -> int:
        """Total overdue decisions, uncapped — the listing above is limited."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT COUNT(*) FROM qa_rag.pending_decisions
                     WHERE reviewed = FALSE
                       AND created_at < NOW() - (%s * INTERVAL '1 day');
                    """,
                    (days,),
                )
                return int(cur.fetchone()[0])
        finally:
            self._put_conn(conn)

    def get_coverage_map_data(self, run_id: str) -> list[dict]:
        """
        Per-section decision counts for one run, for the coverage map.

        Only covers sections the agent actually decided something about. Sections it
        never reached are absent from this table entirely — the caller has to compare
        against the document's real section list to find them, and those are the gaps
        worth seeing.
        """
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT
                        prd_section,
                        COUNT(*)                                          AS decisions,
                        COUNT(*) FILTER (WHERE action = 'keep')           AS keep_count,
                        COUNT(*) FILTER (WHERE action = 'update')         AS update_count,
                        COUNT(*) FILTER (WHERE action = 'deprecate')      AS deprecate_count,
                        COUNT(*) FILTER (WHERE action = 'create')         AS create_count,
                        COUNT(*) FILTER (WHERE action = 'question')       AS question_count,
                        COUNT(*) FILTER (WHERE confidence = 'high')       AS high_confidence,
                        COUNT(*) FILTER (WHERE confidence = 'low')        AS low_confidence,
                        COUNT(*) FILTER (WHERE confidence IS NULL)        AS unrated,
                        COUNT(*) FILTER (WHERE approved IS TRUE)          AS approved_count,
                        COUNT(*) FILTER (WHERE approved IS FALSE)         AS rejected_count,
                        COUNT(*) FILTER (WHERE reviewed IS NOT TRUE)      AS unreviewed_count
                    FROM qa_rag.pending_decisions
                    WHERE run_id = %s
                    GROUP BY prd_section
                    ORDER BY prd_section NULLS FIRST;
                    """,
                    (run_id,),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            self._put_conn(conn)

    def get_decision_by_id(self, decision_id: int) -> dict | None:
        """One decision row, or None. Used by rollback to inspect what was written."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM qa_rag.pending_decisions WHERE id = %s;",
                    (decision_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            self._put_conn(conn)

    def merge_decision_updated_content(self, decision_id: int, patch: dict) -> bool:
        """
        Shallow-merge keys into a decision's `updated_content` JSON.

        Merged in the database rather than read-modify-written in Python: the agent and
        the write-back worker both touch this column, and a read-modify-write would let
        one silently discard the other's keys. Returns True if the row existed.
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE qa_rag.pending_decisions
                       SET updated_content = COALESCE(updated_content, '{}'::jsonb) || %s::jsonb
                     WHERE id = %s;
                    """,
                    (psycopg2.extras.Json(patch), decision_id),
                )
                updated = cur.rowcount > 0
            conn.commit()
            if not updated:
                logger.warning("merge_decision_updated_content: no row id=%s", decision_id)
            return updated
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def mark_written_back(self, decision_id: int) -> bool:
        """Mark a decision as written back to Xray. Returns True if row existed."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE qa_rag.pending_decisions SET written_back = TRUE WHERE id = %s;",
                    (decision_id,),
                )
                updated = cur.rowcount > 0
            conn.commit()
            if not updated:
                logger.warning(f"mark_written_back: no row found with id={decision_id}")
            return updated
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    # ─── Sync run logging ──────────────────────────────────────────────────────

    def start_run(self, run_id: str, run_type: str, prd_source: str | None = None) -> None:
        """Register a run. Idempotent: duplicate run_id is ignored (nested/incremental pipelines)."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO qa_rag.sync_runs (run_id, run_type, prd_source, status)
                    VALUES (%s, %s, %s, 'running')
                    ON CONFLICT (run_id) DO NOTHING;
                    """,
                    (run_id, run_type, prd_source),
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def complete_run(
        self,
        run_id: str,
        tests_synced: int = 0,
        chunks_ingested: int = 0,
        decisions_made: int = 0,
        run_metadata: dict | None = None,
        final_status: str = "completed",
    ) -> bool:
        """
        Mark a run completed, completed_empty, completed_with_errors (e.g. partial ingest), or truncated.
        """
        if final_status not in (
            "completed",
            "completed_empty",
            "completed_with_errors",
            "truncated",
        ):
            raise ValueError(
                f"final_status must be 'completed', 'completed_empty', "
                f"'completed_with_errors', or 'truncated', got {final_status!r}"
            )
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                if run_metadata is not None:
                    cur.execute(
                        """
                        UPDATE qa_rag.sync_runs SET
                            status = %s, tests_synced = %s,
                            chunks_ingested = %s, decisions_made = %s, completed_at = NOW(),
                            run_metadata = %s
                        WHERE run_id = %s;
                        """,
                        (
                            final_status,
                            tests_synced,
                            chunks_ingested,
                            decisions_made,
                            psycopg2.extras.Json(run_metadata),
                            run_id,
                        ),
                    )
                else:
                    cur.execute(
                        """
                        UPDATE qa_rag.sync_runs SET
                            status = %s, tests_synced = %s,
                            chunks_ingested = %s, decisions_made = %s, completed_at = NOW()
                        WHERE run_id = %s;
                        """,
                        (final_status, tests_synced, chunks_ingested, decisions_made, run_id),
                    )
                updated = cur.rowcount > 0
            conn.commit()
            if not updated:
                logger.warning(f"complete_run: no row found for run_id={run_id}")
            return updated
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def fail_run(self, run_id: str, error_message: str) -> bool:
        """Mark a run as failed. Returns True if the run existed."""
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE qa_rag.sync_runs SET status = 'failed', error_message = %s, completed_at = NOW() WHERE run_id = %s;",
                    (error_message, run_id),
                )
                updated = cur.rowcount > 0
            conn.commit()
            if not updated:
                logger.warning(f"fail_run: no row found for run_id={run_id}")
            return updated
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def reopen_run_for_retry(self, run_id: str) -> bool:
        """
        Reset a failed run back to running so background retry can re-run the pipeline
        while keeping the same run_id (polling continuity).
        """
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE qa_rag.sync_runs
                    SET status = 'running', error_message = NULL, completed_at = NULL
                    WHERE run_id = %s;
                    """,
                    (run_id,),
                )
                updated = cur.rowcount > 0
            conn.commit()
            return updated
        except Exception:
            conn.rollback()
            raise
        finally:
            self._put_conn(conn)

    def list_runs(
        self,
        run_type: str | None = None,
        status: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict]:
        """
        List pipeline runs (newest first). Optional filter by run_type and/or status.
        """
        limit = max(1, min(limit, 200))
        offset = max(0, offset)
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT * FROM qa_rag.sync_runs
                    WHERE (%s::text IS NULL OR run_type = %s)
                      AND (%s::text IS NULL OR status = %s)
                    ORDER BY started_at DESC
                    LIMIT %s OFFSET %s;
                    """,
                    (run_type, run_type, status, status, limit, offset),
                )
                return [dict(r) for r in cur.fetchall()]
        finally:
            self._put_conn(conn)

    def get_run(self, run_id: str) -> dict | None:
        """Fetch a single sync_runs row by run_id. Returns None if not found."""
        conn = self._get_conn()
        try:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT * FROM qa_rag.sync_runs WHERE run_id = %s;",
                    (run_id,),
                )
                row = cur.fetchone()
                return dict(row) if row else None
        finally:
            self._put_conn(conn)
