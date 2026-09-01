-- Speeds up the incremental-analysis lookup: "find the previous completed run for
-- prd_source X". Without it that query is a full scan of sync_runs, which grows with
-- every sync, ingest and analysis the system has ever performed.
--
-- Additive and idempotent — safe to apply to a running database.
-- Partial index: rows with no prd_source (test syncs) are never the target of this
-- lookup, so excluding them keeps the index small.

CREATE INDEX IF NOT EXISTS idx_sync_runs_prd_source
    ON qa_rag.sync_runs (prd_source, started_at DESC)
    WHERE prd_source IS NOT NULL;
