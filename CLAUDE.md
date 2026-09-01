# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Bombardier is a QA Intelligence Engine — an agentic RAG system that analyzes Product Requirements Documents (PRDs) against Xray test cases and produces coverage decisions (keep, update, deprecate, create). It uses a 4-phase pipeline: test sync → PRD ingestion → LLM analysis → human review.

## Keep the Code Independent of Any Tenant's Data

Everything committed here — code, tests, comments, docstrings, docs, fixtures — must be
free of the deploying organisation's content. Never commit real Confluence page titles or
IDs, Jira issue keys, product or status identifiers, internal metrics, tenancy hostnames,
or verbatim query text. Use neutral placeholders: `PROJ-1234`, `confluence:1234567890`,
`EXAMPLE_STATUS`, `example.atlassian.net`.

Tests assert on **structure and shape**, never on a tenant's content — neutral fixtures
exercise the same pattern and threshold logic. Real values are fine while investigating;
they must not end up in a file.

Runtime output is where real content leaks in. These are gitignored — keep them that way:
`qa-agent/eval/*.jsonl` (the phase ledger and payload trace record verbatim
request/response bodies) and `qa-agent/eval/ground_truth*.json` except the `.example.`
template. Scan staged content before every commit.

## Build & Run

```bash
# Full stack (first build downloads ~2GB of ML models, takes 5-10 min)
docker compose up -d --build

# Rebuild just the main service after code changes
docker compose up -d --build qa-agent

# Health check
curl http://localhost:8000/health
```

The Jira/Xray MCP server runs **outside** this repo and is assumed to be already deployed and functional. Point `XRAY_MCP_URL` at its Streamable HTTP endpoint; if its tool names differ from the defaults, remap them with `XRAY_MCP_TOOL_MAP` — no code change needed. See [docs/MCP_SERVERS.md](docs/MCP_SERVERS.md).

**Python syntax check (all qa-agent modules):**
```bash
cd qa-agent
python3 -m py_compile main.py && python3 -m py_compile agents/analysis_agent.py && \
python3 -m py_compile embeddings/es_store.py && python3 -m py_compile embeddings/embed_client.py && \
python3 -m py_compile embeddings/reranker.py && python3 -m py_compile ingestion/chunker.py && \
python3 -m py_compile ingestion/prd_pipeline.py && python3 -m py_compile sync/test_sync.py
```

## Testing

```bash
# Retrieval quality evaluation (no LLM cost, calls /analyze/validate)
python3 eval_retrieval.py --prd confluence:<page_id> --module "Platform" --verbose

# Benchmark with ground truth metrics (Recall@K, MRR, nDCG)
# ground_truth.json is gitignored — seed it from the template, then add real labels
cp eval/ground_truth.example.json eval/ground_truth.json
python3 -m eval.benchmark --ground-truth eval/ground_truth.json

# Build ground truth from approved human review decisions
python3 -m eval.benchmark --build-from-decisions --run-id <uuid>
```

Unit tests (stdlib only): `cd qa-agent && PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'`. Integration testing remains via the API. Postgres init scripts in `init-db/` apply on **empty** DB volumes; for `completed_empty` run status, run `init-db/03-sync-runs-status.sql` on existing databases.

## Architecture

### Main codebase

| Directory | Language | Role |
|-----------|----------|------|
| `qa-agent/` | Python 3.12 + FastAPI | Main application — ingestion, embedding, search, analysis |

Jira/Xray operations go through an **external** MCP endpoint over HTTP (`XRAY_MCP_URL`). Tool names are configuration, not code: `integrations/xray_client.py` calls *logical* operations (`get_test`, `search_issues`, …) resolved through `_TOOL_SPEC_DEFAULTS` and overridable via `XRAY_MCP_TOOL_MAP`. Verify with `GET /integrations/mcp/tools`.

GitLab PRD ingestion uses the GitLab REST API directly (`gitlab_ingestor.py`) — there is no GitLab MCP dependency.

### RAG Pipeline (qa-agent)

The retrieval pipeline is two-stage with RRF fusion:

1. **Chunking** (`ingestion/chunker.py`): Semantic chunking using embedding-based topic boundary detection within heading-based sections. Each chunk stores `parent_text` (full section) for context enrichment. Falls back to fixed-window (800 tokens) when `embed_fn` is not provided.

   Structure is load-bearing: table rows, list items and fenced code blocks are atomic segments joined with newlines (a space-join flattens a table onto one line), a table split across chunks repeats its header row in every continuation chunk, and only prose–prose boundaries are scored for topic similarity — adjacent table rows are near-identical, so scoring them produced noise and one embedding call per row.

0. **Source conversion**: every ingestor normalises to markdown before chunking. Confluence storage format goes through `ingestion/confluence_html.py`, which lifts code-macro CDATA bodies, image alt text/attachment filenames, macro titles and tables out *before* the `ac:`/`ri:` tag strip and html2text (all four are invisible to html2text alone). Tabular content from every source — Confluence, Word, Excel, PDF — renders through `ingestion/markdown_table.py` so rows reach the chunker starting with `|`, which is what its table handling keys off.

2. **Embedding** (`embeddings/embed_client.py`): BAAI/bge-m3 (1024 dims). Asymmetric — queries get `"Represent this sentence for searching relevant passages: "` prefix, documents don't. Query embeddings are LRU-cached (512 entries).

3. **Hybrid Search** (`embeddings/es_store.py`): Elasticsearch 8.17 RRF retriever combining KNN (dense vector) + BM25 (keyword). Uses `rank_constant=60`, `num_candidates=max(100, k*10)`.

4. **Reranking** (`embeddings/reranker.py`): cross-encoder/ms-marco-MiniLM-L-6-v2 reranks top-100 retrieval results down to top-50. Enriches scoring with test steps, not just summaries.

5. **Agent** (`agents/analysis_agent.py`): ReAct tool-calling loop with 4 tools — `read_prd_document`, `search_tests`, `get_test_details`, `record_decision`. Multi-query retrieval in validate/preview (heading-only + heading+body variants merged with dedup).

### Data Flow

- **Elasticsearch** stores vector embeddings in two indexes: `qa_test_cases` (synced from Xray) and `qa_prd_chunks` (ingested from Confluence/GitLab/uploads).
- **PostgreSQL** stores relational data: `qa_rag.pending_decisions` (agent decisions) and `qa_rag.sync_runs` (audit log).
- The MCP client (`integrations/xray_client.py`) calls `XRAY_MCP_URL` over HTTP (Streamable HTTP MCP transport).

### Key Design Decisions

- **All long-running operations are async background tasks** returning a `run_id` for polling. Tasks are tracked with done-callbacks to prevent silent exception loss.
- **Score thresholds are dual-mode**: rerank_score (cross-encoder logits: 2.0 high, 0.5 medium) when reranker is loaded, RRF score (0.025 high, 0.012 medium) when not. These need empirical tuning via `eval/benchmark.py`.
- **Embedding model switch requires full re-indexing** of both test cases and PRD chunks.
- **API key auth** is optional — enabled when `QA_ENGINE_API_KEY` env var is set, via `X-API-Key` header.

## Environment Variables

Required in `.env` (see `.env.example`; Compose requires explicit DB passwords):

- `ELASTIC_PASSWORD`, `POSTGRES_PASSWORD`
- `XRAY_CLIENT_ID`, `XRAY_CLIENT_SECRET` — Xray Cloud credentials (test sync)
- `CONFLUENCE_DOMAIN`, `CONFLUENCE_EMAIL`, `CONFLUENCE_API_TOKEN`, `JIRA_DOMAIN`
- `GITLAB_HOST`, `GITLAB_TOKEN`, `GITLAB_PROJECT_ID` — GitLab (PRD ingestion from repo)
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — LLM provider for Phase 3 analysis

External MCP (Compose default is an **example**; set the full URL in `.env`):

- `XRAY_MCP_URL` — Streamable HTTP endpoint for Jira/Xray tooling

Optional:
- `XRAY_MCP_TOOL_MAP` — JSON remap of logical operations to your server's tool/argument names
- `XRAY_MCP_UNWRAP_DATA` — default `1`; set `0` if your server returns a literal top-level `data` field
- `QA_DETERMINISTIC_ANALYSIS_RUN_ID` — default `1`: `/analyze/prd` uses a deterministic run UUID per PRD + module + UTC minute (disable with `0` for random IDs)
- `QA_PHASE_LEDGER_PATH` — override path for append-only `phase-ledger.jsonl` (audit / SHA-256 attestation)
- `QA_ANALYSIS_RUN_ID_SALT` — optional salt mixed into deterministic run IDs
- `QA_EMBED_MAX_CHARS` — default `8000`; per-chunk character cap sent to the embedder. Must stay above the chunker's `SEMANTIC_MAX` (~4800 chars) or the tail of a large chunk is dropped from the vector while Elasticsearch still indexes the full `chunk_text` for BM25
- `QA_INGEST_KEEP_LINK_URLS` — default `0`; set `1` to keep full URLs inline in ingested body text (anchor text is kept either way)
- `QA_INGEST_MAX_SHEET_ROWS` — default `20000`; row cap per sheet for `.xlsx` uploads
- `QA_INGEST_MAX_WORKBOOK_MB` — default `25`; above this an `.xlsx` is read in streaming mode (flat memory, but merged cells are no longer filled down)
- `QA_INGEST_PDF_OCR` — default `0`; set `1` to OCR scanned PDF pages. Requires `pytesseract`, `Pillow` and the `tesseract` binary, none of which are in the image by default — without them the setting logs a warning and degrades to skipping those pages
- `QA_CONFLUENCE_INGEST_ATTACHMENTS` — default `0`; set `1` to ingest a page's `.xlsx/.docx/.pdf/.md/.txt/.csv` attachments as extra sections of that page (`QA_CONFLUENCE_MAX_ATTACHMENT_MB`, default `10`, caps each file). Off by default because it adds a request per page plus a download per attachment
- `QA_CONFLUENCE_INGEST_CHILDREN` — default `0`; set `1` to also ingest a page's descendants (`QA_CONFLUENCE_CHILD_DEPTH` default `1`, `QA_CONFLUENCE_MAX_CHILD_PAGES` default `50`). Each child is indexed as its **own** document with its own `source_id` and `source_version` — folding children into the parent would freeze the parent's version, so a child edit would never trigger a re-ingest
- `XRAY_PROJECT_KEY` — enables nightly auto-sync at 20:30 UTC
- `QA_ENGINE_API_KEY` — enables API key authentication
- `HF_TOKEN` — HuggingFace token (build-time only, for private models)

## Services & Ports

| Service | Port | Notes |
|---------|------|------|
| qa-agent | 8000 | Main API |
| streamlit-review | 127.0.0.1:8501 | Review UI |
| elasticsearch | 127.0.0.1:9200 | Vector store |
| postgres | internal only | Relational metadata |

The MCP server runs separately and is assumed up. Set `XRAY_MCP_URL` to its HTTP base URL and check `GET /integrations/mcp/tools` to confirm the tool mapping (see [docs/MCP_SERVERS.md](docs/MCP_SERVERS.md)).
