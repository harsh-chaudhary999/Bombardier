-- Append-only audit trail of pipeline phases, with a SHA-256 fingerprint per entry.
--
-- Replaces (or backs up) the JSONL file. The file version uses fcntl locking, which
-- coordinates writers on ONE machine — across replicas or Kubernetes pods each has its
-- own filesystem, so the ledger silently fragments into one partial file per pod and no
-- single copy is the record. A Postgres insert is atomic across every instance.
--
-- Additive and idempotent — safe to apply to a running database.

CREATE TABLE IF NOT EXISTS qa_rag.phase_ledger (
    id              BIGSERIAL   PRIMARY KEY,
    ts              TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    phase           TEXT        NOT NULL,
    run_id          TEXT        NOT NULL,
    summary_sha256  TEXT        NOT NULL,
    summary         JSONB       NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_phase_ledger_run
    ON qa_rag.phase_ledger (run_id);

CREATE INDEX IF NOT EXISTS idx_phase_ledger_ts
    ON qa_rag.phase_ledger (ts DESC);

-- run_id is TEXT, not UUID: ingest and sync runs use UUIDs, but the ledger also records
-- phases keyed by other identifiers, and a type error here would drop an audit entry.
COMMENT ON TABLE qa_rag.phase_ledger IS
    'Append-only pipeline audit trail. summary_sha256 fingerprints the summary so a '
    'later edit to the row is detectable. Never UPDATEd or DELETEd by the application.';
