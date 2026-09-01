# Operational Flows

Exact CLI sequences for build, test, debug and deploy. Commands assume the repo root unless
a step says otherwise; `cd qa-agent` steps say so explicitly.

---

## 0. First-time setup

```bash
cp .env.example .env
# Fill in at minimum:
#   ELASTIC_PASSWORD, POSTGRES_PASSWORD   (Compose requires explicit values)
#   XRAY_CLIENT_ID, XRAY_CLIENT_SECRET
#   CONFLUENCE_DOMAIN, CONFLUENCE_EMAIL, CONFLUENCE_API_TOKEN, JIRA_DOMAIN
#   XRAY_MCP_URL                          (the Compose default is an EXAMPLE)
#   ANTHROPIC_API_KEY or OPENAI_API_KEY
```

The Jira/Xray MCP server runs **outside** this repo and is assumed already deployed. See
[docs/MCP_SERVERS.md](docs/MCP_SERVERS.md).

---

## 1. Build & run

```bash
# Full stack. First build downloads ~2GB of ML models — 5-10 min.
docker compose up -d --build

# Rebuild only the main service after a code change (the usual loop)
docker compose up -d --build qa-agent

# Health
curl http://localhost:8000/health

# Confirm the external MCP server is reachable and its tool names resolve
curl http://localhost:8000/integrations/mcp/tools
```

Services: `qa-agent` (8000), `streamlit-review` (127.0.0.1:8501), `elasticsearch`
(127.0.0.1:9200), `postgres` (internal only).

```bash
# Follow logs
docker compose logs -f qa-agent

# Restart without rebuilding (env change only)
docker compose up -d qa-agent
```

---

## 2. Fast local checks (no container)

Run these before rebuilding — they catch most breakage in seconds.

```bash
cd qa-agent

# Syntax check every module the pipeline touches
python3 -m py_compile main.py agents/analysis_agent.py \
  embeddings/es_store.py embeddings/embed_client.py embeddings/reranker.py \
  ingestion/chunker.py ingestion/markdown_table.py ingestion/confluence_html.py \
  ingestion/confluence_ingestor.py ingestion/confluence_space_ingestor.py \
  ingestion/file_ingestor.py ingestion/gitlab_ingestor.py ingestion/prd_pipeline.py \
  sync/test_sync.py

# Unit tests (stdlib only — no third-party packages needed)
PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'
```

### Verifying test isolation

Test modules share one `sys.modules`, and several install stubs. A test that passes in the
suite can fail alone, or vice versa. **Both must pass.**

```bash
cd qa-agent

# Every module standalone
for f in tests/test_*.py; do
  m=$(basename "$f" .py); printf '%-40s' "$m"
  PYTHONPATH=. python3 -m unittest "tests.$m" 2>&1 | tail -1
done

# A few deliberate orderings (stub-install order is what breaks)
PYTHONPATH=. python3 -m unittest tests.test_chunker_structure tests.test_context_budget
PYTHONPATH=. python3 -m unittest tests.test_context_budget tests.test_chunker_structure
```

### Integration tests (real libraries — container only)

These exercise real html2text, python-docx, openpyxl and pdfplumber. They **skip** on a bare
host, so a green local run does not mean they passed.

```bash
docker compose exec qa-agent sh -c \
  "cd /app && PYTHONPATH=. python3 -m unittest tests.test_ingestion_integration -v"
```

---

## 3. Ingestion workflows

All ingest calls return a `run_id`; poll for completion.

`source_type` is one of `confluence | confluence_space | confluence_site | gitlab |
gitlab_file`. On ingest, `module` is a **string**.

```bash
# Single Confluence page (accepts a bare page ID or a full URL)
curl -X POST http://localhost:8000/ingest/prd \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"confluence","source":"1234567890","module":"Platform"}'

# Whole Confluence space (module defaults to the space key)
curl -X POST http://localhost:8000/ingest/prd \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"confluence_space","source":"DOCS",
       "title_filter":"PRD","space_workers":5}'

# Every eligible space on the site
curl -X POST http://localhost:8000/ingest/prd \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"confluence_site","space_keys":"DOCS,PLAT",
       "include_personal":false,"include_archived":false}'

# GitLab module folder / single file
curl -X POST http://localhost:8000/ingest/prd \
  -H 'Content-Type: application/json' \
  -d '{"source_type":"gitlab","source":"Platform","ref":"main"}'

# File upload (.xlsx .docx .pdf .md .txt) — multipart, not JSON
curl -X POST http://localhost:8000/ingest/file \
  -F 'file=@./spec.xlsx' -F 'source_label=Platform Spec'

# Poll
curl http://localhost:8000/ingest/status/<run_id>

# Remove a document's chunks
curl -X DELETE http://localhost:8000/ingest/prd/confluence:1234567890
```

**`force`** re-ingests even when the Confluence version matches what is indexed. Required
after an embedding-model change or a chunking change — the version still matches, but the
stored vectors are stale.

```bash
curl -X POST http://localhost:8000/ingest/prd -H 'Content-Type: application/json' \
  -d '{"source_type":"confluence","source":"1234567890","module":"Platform","force":true}'
```

Test sync (Xray → `qa_test_cases`):

```bash
curl -X POST http://localhost:8000/sync/tests -H 'Content-Type: application/json' \
  -d '{"project_key":"PROJ","folder_path":"","full_content_refresh":false}'
curl http://localhost:8000/sync/status/<run_id>
curl http://localhost:8000/runs          # audit log of past runs
```

`full_content_refresh` re-reads steps and description for every test instead of trusting the
bulk metadata diff — one MCP call per test, so use it only when the diff is suspect.

---

## 4. Analysis → review → write-back

On these endpoints `module` is a **list**, not a string. All three auto-ingest the PRD first
if it has not been ingested.

```bash
# Pre-flight data check — no LLM involved
curl -X POST http://localhost:8000/analyze/validate \
  -H 'Content-Type: application/json' \
  -d '{"prd_source_id":"confluence:1234567890","module":["Platform"],
       "top_k_tests":10,"top_k_kb":5}'

# Preview the exact prompt that would be sent — no LLM call made
curl -X POST http://localhost:8000/analyze/preview -H 'Content-Type: application/json' \
  -d '{"prd_source_id":"confluence:1234567890","module":["Platform"],"top_k":25}'

# Full analysis (costs LLM tokens)
curl -X POST http://localhost:8000/analyze/prd -H 'Content-Type: application/json' \
  -d '{"prd_source_id":"confluence:1234567890","module":["Platform"],
       "top_k":25,"provider":"anthropic"}'
curl http://localhost:8000/analyze/status/<run_id>

# Decisions produced by a run
curl http://localhost:8000/analyze/decisions/<run_id>
curl http://localhost:8000/analyze/decisions/<run_id>/export

# Human review UI
open http://127.0.0.1:8501

# Approve/reject via API, then push approved changes to Xray.
# Always dry_run first — it makes no Xray/Jira MCP calls.
curl -X POST http://localhost:8000/review/decisions/bulk \
  -H 'Content-Type: application/json' -d '{...}'
curl -X POST http://localhost:8000/writeback/execute -H 'Content-Type: application/json' \
  -d '{"run_id":"<uuid>","project_key":"PROJ","dry_run":true}'
```

`project_key` is required for CREATE actions during write-back.

---

## 5. Debugging

### Is the document parsed correctly?

Start here whenever results look wrong — bad retrieval is usually bad ingestion.

```bash
# The ingest log reports section headings and the chunk-type mix.
docker compose logs qa-agent | grep -E "chunked|sections \(|chunk types:"
```

`chunk types: prose=42` on a PRD you know is table-heavy means the source conversion
flattened its tables. Then inspect a specific chunk:

```bash
# Exactly what was fed to the embedder for one chunk, plus its stored fields
curl "http://localhost:8000/explain/prd/confluence:1234567890?chunk_index=0"

# Same for an indexed test case
curl http://localhost:8000/explain/test/PROJ-1234
```

### Full payload tracing

Off by default because it records verbatim request/response bodies.

```bash
# In .env, then restart qa-agent
QA_TRACE=1
QA_TRACE_FILE=eval/payload-trace.jsonl   # default
QA_TRACE_MAX_CHARS=20000                 # 0 = unlimited

# Read it back
jq -c 'select(.kind=="chunk") | {source_id, heading, tokens}' qa-agent/eval/payload-trace.jsonl
jq -c 'select(.kind=="mcp")   | {op, ms}'                     qa-agent/eval/payload-trace.jsonl
```

The trace and `eval/phase-ledger.jsonl` contain real tenant content. They are gitignored —
keep them that way.

### Retrieval quality

```bash
cd qa-agent

# No LLM cost — calls /analyze/validate
python3 eval_retrieval.py --prd confluence:<page_id> --module "Platform" --verbose

# Recall@K / MRR / nDCG against labelled ground truth
cp eval/ground_truth.example.json eval/ground_truth.json    # gitignored; add real labels
python3 -m eval.benchmark --ground-truth eval/ground_truth.json

# Seed ground truth from approved human review decisions
python3 -m eval.benchmark --build-from-decisions --run-id <uuid>
```

### Elasticsearch directly

```bash
ES=http://localhost:9200
curl -u "elastic:$ELASTIC_PASSWORD" "$ES/qa_prd_chunks/_count"
curl -u "elastic:$ELASTIC_PASSWORD" "$ES/qa_prd_chunks/_mapping?pretty"

# Chunk-type distribution for one document
curl -u "elastic:$ELASTIC_PASSWORD" -H 'Content-Type: application/json' \
  "$ES/qa_prd_chunks/_search?size=0" -d '{
    "query":{"term":{"source_id":"confluence:1234567890"}},
    "aggs":{"types":{"terms":{"field":"chunk_type"}}}}'
```

---

## 6. Deploy

```bash
# 1. Local checks first (section 2) — syntax, unit tests, isolation
# 2. Scan staged content for tenant data before committing (section 7)
# 3. Build and roll
docker compose up -d --build qa-agent

# 4. Verify
curl http://localhost:8000/health
curl http://localhost:8000/integrations/mcp/tools
docker compose logs --since 2m qa-agent | grep -iE "error|warn"
```

Startup patches Elasticsearch mappings automatically (`_ensure_indexes` runs `put_mapping`
on existing indexes), so **new mapping fields need no reindex**. A change to the embedding
model or to how chunks are built does require re-ingestion — the stored vectors were built
from the old text.

```bash
# Re-ingest after a chunking or conversion change. Use force — the Confluence
# version is unchanged, so an ordinary re-ingest would skip the page as current.
curl -X POST http://localhost:8000/ingest/prd -H 'Content-Type: application/json' \
  -d '{"source_type":"confluence","source":"1234567890","module":"Platform","force":true}'
```

### Database init

`init-db/` scripts apply only to an **empty** Postgres volume. On an existing database, run
the one you need by hand:

```bash
docker compose exec -T postgres psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" \
  < init-db/03-sync-runs-status.sql     # adds the completed_empty run status
```

### Enabling optional ingestion features

All default off; each needs a `qa-agent` restart. See `.env.example` for the full list.

```bash
QA_CONFLUENCE_INGEST_ATTACHMENTS=1   # + QA_CONFLUENCE_MAX_ATTACHMENT_MB
QA_CONFLUENCE_INGEST_CHILDREN=1      # + QA_CONFLUENCE_CHILD_DEPTH, QA_CONFLUENCE_MAX_CHILD_PAGES
QA_INGEST_PDF_OCR=1                  # needs tesseract-ocr in the image (it is, via Dockerfile)
```

---

## 7. Pre-commit: tenant-data leak scan

Mandatory before every commit — see the rule in `CLAUDE.md`.

```bash
git diff --cached | grep -inE \
  "atlassian\.net|[A-Z]{2,}-[0-9]{3,}|confluence:[0-9]{6,}" \
  | grep -vE "example|PROJ-1234|1234567890"

# These must never be staged
git status --short qa-agent/eval/
```

Real values are fine while investigating; they must not end up in a file.
