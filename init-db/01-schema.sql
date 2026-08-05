-- QA Intelligence Engine — Postgres relational schema
-- Vector data (test cases, PRD chunks) lives in Elasticsearch.
-- Postgres holds only relational/transactional data.
-- Runs automatically on first postgres container start (empty volume).

CREATE SCHEMA IF NOT EXISTS qa_rag;

-- ─────────────────────────────────────────────────────────────
-- Agent decisions pending human review
-- Written by Triage + Gap agents. Approved rows trigger write-back to Xray.
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qa_rag.pending_decisions (
    id              SERIAL PRIMARY KEY,
    run_id          UUID          NOT NULL,
    jira_key        VARCHAR(50),          -- NULL for 'create' / 'question' actions
    action          VARCHAR(20)   NOT NULL CHECK (action IN ('keep', 'update', 'deprecate', 'create', 'question')),
    reason          TEXT,
    updated_content JSONB,        -- populated for UPDATE: {summary, steps, expected_result}
    questions       JSONB,        -- populated for QUESTION: [{question, context}]
    prd_source      TEXT,         -- which PRD triggered this decision
    prd_section     TEXT CHECK (prd_section IS NULL OR char_length(prd_section) <= 500),
    reviewed        BOOLEAN       DEFAULT FALSE,
    approved        BOOLEAN,      -- NULL = not reviewed, TRUE = approved, FALSE = rejected
    reviewer_note   TEXT,
    reviewed_by     TEXT,         -- identity from review API (header), optional
    written_back    BOOLEAN       DEFAULT FALSE,
    created_at      TIMESTAMPTZ   DEFAULT NOW(),
    reviewed_at     TIMESTAMPTZ
);

-- ─────────────────────────────────────────────────────────────
-- Sync / pipeline run log (audit trail and debugging)
-- ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS qa_rag.sync_runs (
    id              SERIAL PRIMARY KEY,
    run_id          UUID          UNIQUE NOT NULL,
    run_type        VARCHAR(50)   NOT NULL,  -- test_sync | prd_ingest | analysis
    -- 32 not 20: 'completed_with_errors' is 21 chars and silently overflowed VARCHAR(20),
    -- failing the final UPDATE of any partial-success run. See 06-sync-runs-status-width.sql.
    status          VARCHAR(32)   NOT NULL DEFAULT 'running'
                    CHECK (status IN ('running', 'completed', 'completed_empty', 'completed_with_errors', 'truncated', 'failed')),
    prd_source      TEXT,
    tests_synced    INT           DEFAULT 0,
    chunks_ingested INT           DEFAULT 0,
    decisions_made  INT           DEFAULT 0,
    error_message   TEXT,
    run_metadata    JSONB         DEFAULT '{}'::jsonb,
    started_at      TIMESTAMPTZ   DEFAULT NOW(),
    completed_at    TIMESTAMPTZ
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_pending_decisions_run_id
    ON qa_rag.pending_decisions (run_id);

CREATE INDEX IF NOT EXISTS idx_pending_decisions_reviewed
    ON qa_rag.pending_decisions (reviewed, approved);

CREATE INDEX IF NOT EXISTS idx_pending_decisions_jira_key
    ON qa_rag.pending_decisions (jira_key);

CREATE INDEX IF NOT EXISTS idx_sync_runs_status
    ON qa_rag.sync_runs (status, started_at DESC);
