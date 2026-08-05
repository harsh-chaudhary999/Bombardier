-- sync_runs.status (+ completed_with_errors for partial ingest)
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
