# Bombardier — QA Intelligence Engine

An agentic RAG system that keeps your Xray test suite in sync with product requirements. It reads PRDs from Confluence, GitLab, or uploaded files, compares them against your existing Xray test cases, and produces structured coverage decisions (keep / update / deprecate / create) that a human reviews before any write-back to Jira/Xray.

---

## Why This Exists

Test suites drift. Requirements change weekly; test cases don't. The gap between what the product says it does and what the test suite verifies is usually invisible until a bug slips through.

Bombardier closes that gap automatically:

1. Pulls your entire test library from Xray into a searchable vector index.
2. Reads your PRDs from Confluence or GitLab.
3. Runs an LLM-powered agent that reasons over both corpora and emits a decision for every PRD section.
4. Shows those decisions in a human review UI — no Jira/Xray write happens until a human approves.
5. Writes back approved decisions (update test steps, add DEPRECATED label, create new tests).

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         External Systems                            │
│  Confluence / GitLab  ←→  qa-agent  ←→  Xray / Jira (via MCP)     │
└─────────────────────────────────────────────────────────────────────┘
              │                  │                      │
              ▼                  ▼                      ▼
        PRD chunks          Analysis            Write-back
       (ES index)           (LangChain          (approved
                            ReAct agent)        decisions)
              │                  │
              ▼                  ▼
      ┌───────────────┐  ┌──────────────────┐
      │ Elasticsearch │  │    PostgreSQL     │
      │  qa_test_cases│  │  pending_decisions│
      │  qa_prd_chunks│  │  sync_runs        │
      └───────────────┘  └──────────────────┘
              │
              ▼
       Streamlit Review UI (Phase 4)
```

### Four-Phase Pipeline

| Phase | Name | What Happens |
|-------|------|-------------|
| 1 | **Test Sync** | Xray test cases are fetched via MCP and vectorised into Elasticsearch (`qa_test_cases` index). Runs on demand or nightly at 20:30 UTC if `XRAY_PROJECT_KEY` is set. |
| 2 | **PRD Ingestion** | PRDs are chunked semantically (embedding-based boundary detection), embedded with BAAI/bge-m3, and stored in `qa_prd_chunks`. Sources: Confluence page, Confluence space, GitLab folder, GitLab file, or direct file upload. |
| 3 | **LLM Analysis** | A LangChain ReAct agent iterates over PRD sections. For each section it searches the test index (hybrid KNN + BM25), inspects individual tests, checks prior decisions, and records one of: `keep / update / deprecate / create / question`. |
| 4 | **Human Review** | Decisions land in a Streamlit dashboard. Reviewers approve or reject each decision. Nothing touches Xray until approval. |
| 5 | **Write-back** | Approved decisions are flushed to Xray/Jira via MCP: test steps updated, DEPRECATED labels added, new tests bulk-created (up to 50 at a time). |

### Retrieval Stack

- **Embedding**: BAAI/bge-m3 (1024 dims). Asymmetric — queries get a retrieval prefix, stored documents don't.
- **Hybrid search**: Elasticsearch 8.17 RRF combining KNN (dense) + BM25 (keyword). `rank_constant=60`, `num_candidates=max(100, k×10)`.
- **Reranking**: cross-encoder/ms-marco-MiniLM-L-6-v2 reranks top-100 candidates to top-50.
- **Score thresholds** (dual-mode): rerank score when reranker is loaded (2.0 = high, 0.5 = medium), RRF score otherwise (0.025 = high, 0.012 = medium).

### Agent Tools (Phase 3)

| Tool | Purpose |
|------|---------|
| `read_prd_document` | Fetch all chunks for the PRD being analysed |
| `search_tests` | Hybrid search the test index for a PRD section |
| `get_test_details` | Fetch full steps/preconditions for a specific test key |
| `record_decision` | Persist a keep/update/deprecate/create/question decision |
| `get_prior_decisions` | Look up previous decisions for a Jira key across runs |

### Observability

- **Phase ledger**: Append-only JSONL at `eval/phase-ledger.jsonl` (or `QA_PHASE_LEDGER_PATH`). Every analysis run appends a SHA-256 attested entry — tamper-evident audit trail.
- **Deterministic run IDs**: `sha256(prd_source_id | sorted_modules | utc_minute | salt)` — same PRD + module + minute always produces the same run UUID, enabling idempotent re-runs.
- **Loop status**: Agent terminates with `COMPLETED` or `MAX_TURNS_REACHED` — surface in metadata.
- **Coverage score**: `covered_sections / total_prd_sections` — stored in run metadata.
- **Metrics**: JSON snapshot at `/metrics`, Prometheus text at `/metrics/prometheus`.

---

## Services & Ports

| Service | Port | Notes |
|---------|------|-------|
| qa-agent (FastAPI) | `0.0.0.0:8000` | Main API |
| streamlit-review | `127.0.0.1:8501` | Human review UI (loopback only) |
| elasticsearch | `127.0.0.1:9200` | Internal; loopback only |
| postgres | internal | No host port exposed |
| Xray MCP server | your choice | External process; set `XRAY_MCP_URL` |

---

## Setup

### Prerequisites

- Docker Engine 24+ and Docker Compose v2
- An Xray Cloud account with API credentials
- A Confluence Cloud account with an API token
- At least one LLM API key (Anthropic **or** OpenAI)
- Python 3.12 (only needed for local dev without Docker)

### 1. Clone and configure

```bash
git clone <repo-url> bombardier
cd bombardier
cp .env.example .env
```

Edit `.env` with your credentials (see [Environment Variables](#environment-variables)).

### 2. Start the MCP server

Bombardier talks to Jira/Xray through an external MCP HTTP server. Start it separately on port 3100 (or whatever you configure as `XRAY_MCP_URL`). See your MCP server's own documentation.

### 3. Build and start the stack

```bash
# First start downloads ~2.5 GB of ML models — allow 5–10 min
docker compose up -d --build
```

After the build, watch startup:

```bash
docker compose logs -f qa-agent
```

The service is ready when you see `Uvicorn running on http://0.0.0.0:8000`.

### 4. Health check

```bash
curl http://localhost:8000/health
# {"status":"ok","elasticsearch":"ok (v8.17.0)",...}
```

### 5. Rebuild after code changes

```bash
docker compose up -d --build qa-agent
```

### Existing databases

If you have an existing Postgres volume that pre-dates the `completed_empty` run status, apply the migration:

```bash
docker exec -i qa-postgres psql -U qa -d qa < init-db/03-sync-runs-status.sql
```

---

## Environment Variables

### Required

| Variable | Description |
|----------|-------------|
| `POSTGRES_PASSWORD` | PostgreSQL password |
| `ELASTIC_PASSWORD` | Elasticsearch `elastic` user password |
| `CONFLUENCE_DOMAIN` | Your Atlassian domain, e.g. `company.atlassian.net` |
| `CONFLUENCE_EMAIL` | Atlassian account email |
| `CONFLUENCE_API_TOKEN` | Atlassian API token |
| `JIRA_DOMAIN` | Jira domain (often same as `CONFLUENCE_DOMAIN`) |
| `XRAY_CLIENT_ID` | Xray Cloud client ID |
| `XRAY_CLIENT_SECRET` | Xray Cloud client secret |
| `ANTHROPIC_API_KEY` | Claude API key (or use OpenAI below) |
| `OPENAI_API_KEY` | OpenAI API key (alternative to Anthropic) |
| `XRAY_MCP_URL` | Full URL to your Xray MCP server, e.g. `http://127.0.0.1:3100/mcp` |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `GITLAB_HOST` | `gitlab.com` | GitLab hostname |
| `GITLAB_TOKEN` | — | GitLab personal access token |
| `GITLAB_PROJECT_ID` | — | GitLab project ID for PRD ingestion |
| `XRAY_MCP_TOOL_MAP` | — | JSON remap of logical operations to your server's tool/argument names |
| `XRAY_MCP_UNWRAP_DATA` | `1` | Set `0` if your server returns a literal top-level `data` field |
| `XRAY_PROJECT_KEY` | — | Enables nightly auto-sync at 20:30 UTC |
| `QA_ENGINE_API_KEY` | — | Enables `X-API-Key` header auth on all endpoints |
| `QA_DETERMINISTIC_ANALYSIS_RUN_ID` | `1` | `1` = deterministic run UUIDs, `0` = random |
| `QA_ANALYSIS_RUN_ID_SALT` | — | Extra entropy for deterministic run IDs |
| `QA_PHASE_LEDGER_PATH` | `eval/phase-ledger.jsonl` | Override path for the audit ledger |
| `REVIEW_UI_PASSWORD` | — | Password gate for Streamlit UI (empty = no auth) |
| `HF_TOKEN` | — | HuggingFace token (build-time only, for private models) |
| `AZURE_OPENAI_ENDPOINT` | — | Azure OpenAI endpoint |
| `AZURE_OPENAI_API_KEY` | — | Azure OpenAI key |
| `AZURE_OPENAI_API_VERSION` | `2024-08-01-preview` | Azure OpenAI API version |

---

## API Reference

All endpoints accept and return JSON. Rate limits apply per IP. Set `X-API-Key` header if `QA_ENGINE_API_KEY` is configured.

### Health

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Service liveness. Returns `status: ok` or `degraded` (HTTP 503). |

### Phase 1 — Test Sync

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/sync/tests` | Start an Xray → Elasticsearch sync. Body: `{project_key, folder_path?}`. Returns `run_id`. |
| `GET` | `/sync/status/{run_id}` | Poll sync progress. |

### Phase 2 — PRD Ingestion

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/ingest/prd` | Ingest from Confluence or GitLab. Body: `{source_type, source, module?, ref?, title_filter?, parent_id?}`. |
| `POST` | `/ingest/file` | Upload a file (xlsx/docx/pdf/md/txt, max 50 MB). Returns `run_id`. |
| `GET` | `/ingest/status/{run_id}` | Poll ingest progress. |
| `DELETE` | `/ingest/prd/{source_id}` | Remove all chunks for a PRD source. Returns chunk count deleted. |

`source_type` values for `/ingest/prd`:

| Value | `source` example | Scope |
|-------|-----------------|-------|
| `confluence` | `"1234567890"` (page ID) | One page |
| `confluence_space` | `"DOCS"` (space key) | Every page in the space, any nesting depth |
| `confluence_site` | *(not used — see `space_keys`)* | Every page in every selected space |
| `gitlab` | `"Platform"` (module folder) | All `.md` under the folder |
| `gitlab_file` | `"Platform/docs/Features.md"` | One file |

**Confluence space and site ingests are incremental.** A page is re-fetched and
re-embedded only when its Confluence `version` differs from the indexed
`source_version`, so re-running to pick up edits costs roughly one listing call per
space instead of a full crawl. Chunks are tagged `module=<space key>` so module
filters work per space.

Pass `"force": true` to re-ingest regardless of version — **required after changing the
embedding model**, where versions still match but the stored vectors are no longer valid.

### Phase 3 — Search & Analysis

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search/tests` | Find test cases relevant to a PRD text. Body: `{prd_text, module?, top_k?, mode?, min_score?}`. |
| `POST` | `/search/prd` | Find PRD chunks relevant to a query. |
| `GET` | `/explain/test/{jira_key}` | Show why a test is indexed (embedding metadata). |
| `GET` | `/explain/prd/{source_id}` | Show PRD chunk metadata. |
| `POST` | `/analyze/prd` | Run full coverage analysis. Body: `{prd_source_id, module?, provider?, model?, top_k?}`. Returns `run_id`. |
| `POST` | `/analyze/preview` | Same as `/analyze/prd` but returns a preview without persisting decisions. |
| `POST` | `/analyze/validate` | Validate retrieval quality only (no LLM, no cost). |
| `GET` | `/analyze/status/{run_id}` | Poll analysis progress. |
| `GET` | `/analyze/decisions/{run_id}` | List decisions for a run (paginated). |
| `GET` | `/analyze/decisions/{run_id}/export` | CSV download of all decisions. |
| `POST` | `/analyze/incremental` | Re-run analysis for only the PRD sections that changed since the last run. |

### Phase 4 — Human Review

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/review/decision` | Approve or reject a single decision. Body: `{decision_id, approved, reviewer_note?}`. |

### Phase 5 — Write-back

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/writeback/execute` | Flush approved decisions to Xray/Jira. Body: `{run_id?, project_key?, dry_run?}`. |

`dry_run: true` returns a preview of what would be written without making any Xray/Jira calls.

### Evaluation & Observability

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/runs` | List all pipeline runs. Query: `type`, `status`, `page`, `limit`. |
| `POST` | `/evaluate/decisions` | LLM-as-judge evaluation of decision quality (1-5 scale). |
| `GET` | `/metrics` | All metrics as JSON. |
| `GET` | `/metrics/prometheus` | Metrics in Prometheus text format. |

---

## Common Operations (curl)

These examples cover the full workflow end-to-end. All calls hit `http://localhost:8000`.

```bash
# Set once for the session; omit the header entirely if QA_ENGINE_API_KEY is not set
export BASE="http://localhost:8000"
export KEY="your-api-key-here"           # skip if auth is disabled
AUTH='-H "X-API-Key: $KEY"'              # paste inline or wrap in a helper
```

> **Tip**: Poll every 5–10 s and check `"status"` in the response. Terminal states are `completed`, `completed_with_errors`, `failed`, and `completed_empty`.

---

### Health check

```bash
curl $BASE/health
# {"status":"ok","elasticsearch":"ok (v8.17.0)","postgres":"ok","models":{"embedding":"loaded","reranker":"loaded"}}
```

---

### Phase 1 — Sync test cases from Xray (one-time or on demand)

```bash
# Start sync — replace PROJ with your Jira project key
RUN=$(curl -s -X POST $BASE/sync/tests \
  -H "Content-Type: application/json" \
  -d '{"project_key": "PROJ"}' | jq -r .run_id)

echo "Sync run_id: $RUN"

# Optional: restrict to a specific Xray folder
curl -s -X POST $BASE/sync/tests \
  -H "Content-Type: application/json" \
  -d '{"project_key": "PROJ", "folder_path": "/Regression/Platform"}'

# Poll until status != running
curl $BASE/sync/status/$RUN
# {"status":"completed","tests_synced":423,"elapsed_s":14.2}
```

---

### Phase 2 — Ingest a PRD

#### From a Confluence page (by page ID)

```bash
RUN=$(curl -s -X POST $BASE/ingest/prd \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "confluence",
    "source": "1234567890",
    "module": "Platform"
  }' | jq -r .run_id)

curl $BASE/ingest/status/$RUN
# {"status":"completed","chunks_ingested":38,"elapsed_s":4.1}
```

#### From an entire Confluence space

```bash
RUN=$(curl -s -X POST $BASE/ingest/prd \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "confluence_space",
    "source": "DOCS",
    "title_filter": "Platform",
    "parent_id": "98765432"
  }' | jq -r .run_id)

curl $BASE/ingest/status/$RUN
# Large spaces take several minutes; check elapsed_s and chunks_ingested as it runs
```

#### From every Confluence space on the site (recursive)

Walks every eligible space and every page inside it. Personal (`~user`) and archived
spaces are excluded by default.

```bash
RUN=$(curl -s -X POST $BASE/ingest/prd \
  -H "Content-Type: application/json" \
  -d '{"source_type": "confluence_site"}' | jq -r .run_id)

curl -s $BASE/ingest/status/$RUN | jq '{status, chunks_ingested, run_metadata}'
```

Restrict to specific spaces, and re-run later to pick up only what changed:

```bash
curl -s -X POST $BASE/ingest/prd \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "confluence_site",
    "space_keys": "DOCS,PLAT,BILLING",
    "title_filter": "PRD",
    "space_workers": 8
  }'
```

`run_metadata` on the finished run reports per-space counts, `confluence_pages_unchanged`,
and the page IDs that failed — check it rather than assuming `completed` means complete.
A run that hit fetch or embedding failures finishes as `completed_with_errors`.

> **First site crawl is expensive.** Every page is fetched, chunked and embedded on CPU.
> Start with `space_keys` scoped to the spaces that actually hold PRDs — a whole-site
> crawl pulls in meeting notes and HR pages, which mostly adds retrieval noise.

#### From a single GitLab file

```bash
RUN=$(curl -s -X POST $BASE/ingest/prd \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "gitlab_file",
    "source": "platform/docs/Requirements.md",
    "ref": "main"
  }' | jq -r .run_id)

curl $BASE/ingest/status/$RUN
```

#### From an entire GitLab module folder

```bash
curl -s -X POST $BASE/ingest/prd \
  -H "Content-Type: application/json" \
  -d '{
    "source_type": "gitlab",
    "source": "Platform",
    "module": "Platform",
    "ref": "release/2.4"
  }'
```

#### Upload a local file (xlsx / docx / pdf / md / txt, max 50 MB)

```bash
RUN=$(curl -s -X POST $BASE/ingest/file \
  -F "file=@/path/to/requirements.xlsx" \
  -F "source_label=Platform PRD v2.4" | jq -r .run_id)

curl $BASE/ingest/status/$RUN
```

#### Remove a PRD from the index

```bash
# source_id is the value returned by /ingest/status — e.g. "confluence:1234567890"
curl -s -X DELETE "$BASE/ingest/prd/confluence%3A1234567890"
# {"deleted":38}
```

---

### Phase 3 — Validate retrieval before spending on LLM

```bash
# No LLM call — checks that test cases can be found for the PRD sections
curl -s -X POST $BASE/analyze/validate \
  -H "Content-Type: application/json" \
  -d '{
    "prd_source_id": "confluence:1234567890",
    "module": "Platform",
    "top_k": 20
  }' | jq '{sections_checked, avg_score, low_coverage_sections}'
```

---

### Phase 3 — Run a full coverage analysis

```bash
# Kick off analysis — deterministic run_id means re-running within the same minute is a no-op
RUN=$(curl -s -X POST $BASE/analyze/prd \
  -H "Content-Type: application/json" \
  -d '{
    "prd_source_id": "confluence:1234567890",
    "module": "Platform",
    "provider": "anthropic",
    "top_k": 20
  }' | jq -r .run_id)

echo "Analysis run_id: $RUN"

# Poll — terminal states: completed, completed_with_errors, failed
curl $BASE/analyze/status/$RUN
# {"status":"completed","decisions_recorded":54,"coverage_score":0.87,"elapsed_s":142}

# Fetch decisions (paginated, 50 per page)
curl "$BASE/analyze/decisions/$RUN?page=1&limit=50" | jq .

# Filter to a specific action
curl "$BASE/analyze/decisions/$RUN?action=create" | jq '.decisions[] | {id, prd_section, reason}'

# CSV export (download all decisions for the run)
curl -o decisions.csv "$BASE/analyze/decisions/$RUN/export"
```

#### Preview without persisting decisions

```bash
# Same body as /analyze/prd — results are returned immediately, nothing is saved
curl -s -X POST $BASE/analyze/preview \
  -H "Content-Type: application/json" \
  -d '{
    "prd_source_id": "confluence:1234567890",
    "module": "Platform"
  }' | jq '.decisions[] | select(.action == "create")'
```

---

### Phase 4 — Approve decisions

```bash
# Approve a single decision
curl -s -X POST $BASE/review/decision \
  -H "Content-Type: application/json" \
  -d '{
    "decision_id": 42,
    "approved": true
  }'

# Approve with a reviewer note
curl -s -X POST $BASE/review/decision \
  -H "Content-Type: application/json" \
  -d '{
    "decision_id": 43,
    "approved": true,
    "reviewer_note": "Confirmed: test PROJ-101 no longer covers the new OAuth flow"
  }'

# Reject
curl -s -X POST $BASE/review/decision \
  -H "Content-Type: application/json" \
  -d '{"decision_id": 44, "approved": false, "reviewer_note": "False positive — test is still valid"}'
```

> Bulk review is available in the Streamlit UI at `http://127.0.0.1:8501`.

---

### Phase 5 — Write back to Xray/Jira

```bash
# Dry run first — no Xray/Jira calls, shows what would happen
curl -s -X POST $BASE/writeback/execute \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "'$RUN'",
    "project_key": "PROJ",
    "dry_run": true
  }' | jq '{status, total, by_action}'

# Live write-back — runs only approved decisions for the given run_id
curl -s -X POST $BASE/writeback/execute \
  -H "Content-Type: application/json" \
  -d '{
    "run_id": "'$RUN'",
    "project_key": "PROJ"
  }' | jq '{status, total, written_back, errors}'
# {"status":"completed","total":31,"written_back":{"keep":12,"update":8,"deprecate":4,"create":7},"errors":[]}
```

---

### Incremental re-analysis (after PRD update)

```bash
# Re-ingest the updated PRD first, then run incremental analysis
# Only sections whose content hash changed since the last run are re-analysed
RUN2=$(curl -s -X POST $BASE/analyze/incremental \
  -H "Content-Type: application/json" \
  -d '{
    "prd_source_id": "confluence:1234567890",
    "module": "Platform",
    "base_run_id": "'$RUN'"
  }' | jq -r .run_id)

curl $BASE/analyze/status/$RUN2
```

---

### Ad-hoc test search

```bash
# Find test cases relevant to a piece of PRD text
curl -s -X POST $BASE/search/tests \
  -H "Content-Type: application/json" \
  -d '{
    "prd_text": "User must be able to reset their password via email link within 15 minutes",
    "module": "Platform",
    "top_k": 10
  }' | jq '.results[] | {jira_key, summary, score}'

# Search PRD chunks for a topic
curl -s -X POST $BASE/search/prd \
  -H "Content-Type: application/json" \
  -d '{
    "query": "password reset expiry policy",
    "top_k": 5
  }' | jq '.results[] | {source_id, section_heading, score}'
```

---

### Observability

```bash
# Metrics snapshot (JSON)
curl $BASE/metrics | jq .

# Prometheus scrape endpoint
curl $BASE/metrics/prometheus

# List all pipeline runs
curl "$BASE/runs?type=analysis&status=completed&page=1&limit=20" | jq .
```

---

## Running Tests

```bash
cd qa-agent

# Unit tests (stdlib only — no LLM cost)
PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'

# Syntax check all modules
python3 -m py_compile main.py agents/analysis_agent.py agents/writeback.py \
  embeddings/es_store.py embeddings/embed_client.py embeddings/reranker.py \
  integrations/xray_client.py ingestion/chunker.py ingestion/prd_pipeline.py \
  sync/test_sync.py observability/canonical_json.py observability/phase_ledger.py \
  observability/run_identity.py agents/loop_status.py

# Retrieval quality check (no LLM cost — hits /analyze/validate)
python3 eval_retrieval.py --prd confluence:<page_id> --module "Platform" --verbose

# Benchmark with ground truth (Recall@K, MRR, nDCG)
# ground_truth.json is gitignored — seed it from the template, then add real labels
cp eval/ground_truth.example.json eval/ground_truth.json
python3 -m eval.benchmark --ground-truth eval/ground_truth.json

# Build ground truth from approved review decisions
python3 -m eval.benchmark --build-from-decisions --run-id <uuid>
```

---

## Human Review UI

The Streamlit dashboard runs at `http://127.0.0.1:8501`. It shows pending decisions grouped by PRD source with the agent's reasoning, matched test cases, and approve/reject controls. No decision writes to Xray until a human clicks Approve and `/writeback/execute` is called.

If `REVIEW_UI_PASSWORD` is set, the UI prompts for a password on load.

---

## Development Notes

- **Hot reload**: The `./qa-agent:/app` volume mount in `docker-compose.yml` means code changes apply immediately without a rebuild. Remove this mount in production.
- **Memory**: The `qa-agent` container is capped at 6 GB (bge-m3 is 2.5 GB; the cross-encoder adds ~200 MB). Tune down to `4g` on machines with less than 12 GB RAM.
- **Embedding model swap**: Changing `EMBEDDING_MODEL` requires re-indexing both `qa_test_cases` and `qa_prd_chunks`. ESStore logs an `ERROR` on startup if the stored model doesn't match the env var.
- **LLM providers**: Pass `provider: "anthropic"`, `"openai"`, or `"azure_openai"` in analysis requests. The agent defaults to Claude Sonnet.
- **MCP connection pool**: The Xray client maintains a reusable session (max age 5 min, idle ping after 60 s). Failed pooled calls fall back to one-shot sessions automatically.
