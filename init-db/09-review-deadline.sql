-- Supports the overdue-review query: unreviewed decisions older than the SLA window.
--
-- Deliberately NOT a generated `review_deadline` column, which is what the original plan
-- specified. Two reasons:
--
--   1. A generated column requires an IMMUTABLE expression. `created_at` is TIMESTAMPTZ,
--      and `timestamptz + interval` is only STABLE (it depends on the session time zone
--      for day-and-larger intervals), so Postgres may reject the column outright.
--   2. It would bake the SLA period into the schema. Changing 30 days to 14 would then
--      need a migration and a table rewrite, rather than a query parameter.
--
-- The deadline is computed at query time from created_at instead; this index makes that
-- query cheap. Partial, because reviewed decisions are never overdue.
--
-- Additive and idempotent — safe to apply to a running database.

CREATE INDEX IF NOT EXISTS idx_pending_decisions_unreviewed_age
    ON qa_rag.pending_decisions (created_at)
    WHERE reviewed = FALSE;
