-- Reviewer identity for compliance; optional prd_section length cap (matches agent truncation)
ALTER TABLE qa_rag.pending_decisions ADD COLUMN IF NOT EXISTS reviewed_by TEXT;

UPDATE qa_rag.pending_decisions
SET prd_section = left(prd_section, 500)
WHERE prd_section IS NOT NULL AND char_length(prd_section) > 500;

ALTER TABLE qa_rag.pending_decisions DROP CONSTRAINT IF EXISTS pending_decisions_prd_section_len;
ALTER TABLE qa_rag.pending_decisions ADD CONSTRAINT pending_decisions_prd_section_len
    CHECK (prd_section IS NULL OR char_length(prd_section) <= 500);
