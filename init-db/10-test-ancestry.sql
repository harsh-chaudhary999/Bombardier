-- Append-only log of every change this engine has made to a test case.
--
-- Answers "why does this test look like this?" months later. pending_decisions records
-- what was *proposed*; this records what was actually *written back*, which is a
-- different and smaller set — decisions get rejected, and write-backs can fail.
--
-- Additive and idempotent — safe to apply to a running database.

CREATE TABLE IF NOT EXISTS qa_rag.test_ancestry (
    id              BIGSERIAL PRIMARY KEY,
    jira_key        VARCHAR(50)  NOT NULL,
    run_id          UUID         NOT NULL,
    prd_source      TEXT,
    change_type     VARCHAR(20)  NOT NULL
                    CHECK (change_type IN ('updated', 'deprecated', 'created', 'rolled_back')),
    reason_summary  TEXT,
    decision_id     INTEGER,     -- informational; decisions may be archived away later
    changed_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- The main read: one test's history, newest first.
CREATE INDEX IF NOT EXISTS idx_test_ancestry_key_time
    ON qa_rag.test_ancestry (jira_key, changed_at DESC);

-- "What did this run change?" — the audit direction.
CREATE INDEX IF NOT EXISTS idx_test_ancestry_run
    ON qa_rag.test_ancestry (run_id);

COMMENT ON TABLE qa_rag.test_ancestry IS
    'Append-only record of write-backs per test case. Deliberately not FK-constrained to '
    'pending_decisions: decisions are archived after a retention period and the history '
    'must outlive them.';
