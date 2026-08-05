-- Add JSON metadata for pipeline runs (e.g. PRD section hashes for incremental analysis)
ALTER TABLE qa_rag.sync_runs
    ADD COLUMN IF NOT EXISTS run_metadata JSONB DEFAULT '{}'::jsonb;
