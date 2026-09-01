-- Long-term storage for decisions that have been written back and reviewed.
--
-- pending_decisions is a work queue that is only ever appended to. At a few hundred
-- decisions per analysis run it grows without bound, and the queries that matter
-- (unreviewed, overdue, per-run) slow down under rows nobody will look at again.
--
-- Rows are MOVED, not copied — the archive is the same shape, so history is still
-- queryable, just not in the hot table. test_ancestry deliberately holds no foreign key
-- to pending_decisions so a test's history survives its decisions being archived.
--
-- Additive and idempotent — safe to apply to a running database.

CREATE TABLE IF NOT EXISTS qa_rag.pending_decisions_archive
    (LIKE qa_rag.pending_decisions INCLUDING DEFAULTS INCLUDING CONSTRAINTS);

CREATE INDEX IF NOT EXISTS idx_decisions_archive_run
    ON qa_rag.pending_decisions_archive (run_id);

CREATE INDEX IF NOT EXISTS idx_decisions_archive_created
    ON qa_rag.pending_decisions_archive (created_at DESC);

-- Supports the archival sweep itself: written-back rows ordered by age.
CREATE INDEX IF NOT EXISTS idx_pending_decisions_written_back_age
    ON qa_rag.pending_decisions (created_at)
    WHERE written_back = TRUE;

COMMENT ON TABLE qa_rag.pending_decisions_archive IS
    'Decisions moved out of pending_decisions after QA_DECISIONS_RETENTION_DAYS. Same '
    'shape as the source table. Not indexed for review workflows — this is cold storage.';
