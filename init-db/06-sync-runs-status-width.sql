-- sync_runs.status was VARCHAR(20) while the CHECK constraint (and the code) allow
-- 'completed_with_errors' — 21 characters. Any partial-success run therefore failed its
-- final UPDATE with:
--     value too long for type character varying(20)
-- The work itself had already completed; only the bookkeeping row failed, so runs were
-- misreported as 'failed' with chunks already indexed in Elasticsearch.
--
-- Applies automatically on an EMPTY volume. On an existing database run it by hand:
--   docker exec -i qa-postgres psql -U qa -d qa < init-db/06-sync-runs-status-width.sql
ALTER TABLE qa_rag.sync_runs ALTER COLUMN status TYPE VARCHAR(32);

-- Re-assert the constraint against the widened column.
ALTER TABLE qa_rag.sync_runs DROP CONSTRAINT IF EXISTS sync_runs_status_check;
ALTER TABLE qa_rag.sync_runs ADD CONSTRAINT sync_runs_status_check
    CHECK (status IN (
        'running',
        'completed',
        'completed_empty',
        'completed_with_errors',
        'truncated',
        'failed'
    ));
