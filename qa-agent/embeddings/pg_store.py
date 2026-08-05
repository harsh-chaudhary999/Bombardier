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
        """Insert a single agent decision. Returns the new row id."""
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
        conn = self._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO qa_rag.pending_decisions
                        (run_id, jira_key, action, reason, updated_content,
                         questions, prd_source, prd_section,
                         reviewed, approved, reviewer_note, written_back, reviewed_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
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
