-- Agent-reported confidence per decision, so reviewers can triage: read the low-confidence
-- deprecations first rather than working through a run in insertion order.
--
-- Additive and idempotent. NULL is a first-class value meaning "the agent did not say" —
-- every decision recorded before this column existed keeps that meaning, and no consumer
-- may treat NULL as low.

ALTER TABLE qa_rag.pending_decisions
    ADD COLUMN IF NOT EXISTS confidence VARCHAR(10)
    CHECK (confidence IS NULL OR confidence IN ('high', 'medium', 'low'));

-- Partial: reviewers filter within a run, and rows without a confidence value are
-- not what this index is for.
CREATE INDEX IF NOT EXISTS idx_pending_decisions_confidence
    ON qa_rag.pending_decisions (run_id, confidence)
    WHERE confidence IS NOT NULL;

COMMENT ON COLUMN qa_rag.pending_decisions.confidence IS
    'Agent-reported confidence: high|medium|low. NULL = the agent did not specify; '
    'never interpret NULL as low.';
