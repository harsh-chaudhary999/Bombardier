# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Bombardier is a QA Intelligence Engine — an agentic RAG system that analyzes Product
Requirements Documents (PRDs) against Xray test cases and produces coverage decisions
(keep, update, deprecate, create). Four-phase pipeline: test sync → PRD ingestion →
LLM analysis → human review.

## Memory

@decisions.md — why the architecture is the way it is, and which alternatives were rejected.
Read before changing retrieval, chunking, source conversion or ingestion identity.

@flow.md — exact CLI sequences for build, test, debug and deploy.

@mistakes.md — real bugs and anti-patterns from this repo. Read before working on the
ingestion path or adding tests; several of these were invisible failures that shipped.

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
template. Scan staged content before every commit (`flow.md` §7).

## Quick Start

```bash
docker compose up -d --build          # first build downloads ~2GB of models, 5-10 min
curl http://localhost:8000/health
```

Before rebuilding, run the fast local checks — syntax + stdlib unit tests, no container
needed. Full sequences and the test-isolation loop are in `flow.md` §2.

```bash
cd qa-agent && PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'
```

## Architecture

| Directory | Language | Role |
|-----------|----------|------|
| `qa-agent/` | Python 3.12 + FastAPI | Ingestion, embedding, search, analysis |

Jira/Xray operations go through an **external** MCP endpoint over HTTP (`XRAY_MCP_URL`),
assumed already deployed. Tool names are configuration, not code — remap with
`XRAY_MCP_TOOL_MAP` and verify with `GET /integrations/mcp/tools`. See
[docs/MCP_SERVERS.md](docs/MCP_SERVERS.md) and ADR-001. GitLab ingestion uses the GitLab
REST API directly (ADR-002).

### RAG pipeline

0. **Source conversion** — every ingestor normalises to markdown before chunking.
   Confluence storage format goes through `ingestion/confluence_html.py`; all tabular
   content renders through `ingestion/markdown_table.py`. ADR-006, ADR-007.
1. **Chunking** (`ingestion/chunker.py`) — semantic, embedding-based topic boundaries
   within heading sections; fixed-window (800 tokens) when `embed_fn` is absent. Structure
   is load-bearing. ADR-008 … ADR-012.
2. **Embedding** (`embeddings/embed_client.py`) — BAAI/bge-m3, 1024 dims, asymmetric.
   ADR-004.
3. **Hybrid search** (`embeddings/es_store.py`) — Elasticsearch 8.17 RRF over KNN + BM25
   (`rank_constant=60`, `num_candidates=max(100, k*10)`). ADR-003.
4. **Reranking** (`embeddings/reranker.py`) — `cross-encoder/ms-marco-MiniLM-L-6-v2`,
   top-100 → top-50, scored with test steps rather than summaries alone.
5. **Agent** (`agents/analysis_agent.py`) — ReAct loop with four tools:
   `read_prd_document`, `search_tests`, `get_test_details`, `record_decision`.

### Data flow

- **Elasticsearch** — `qa_test_cases` (synced from Xray) and `qa_prd_chunks` (Confluence /
  GitLab / uploads).
- **PostgreSQL** — `qa_rag.pending_decisions` and `qa_rag.sync_runs`.
- **MCP client** (`integrations/xray_client.py`) — HTTP to `XRAY_MCP_URL`.

Long-running operations are async tasks returning a `run_id` for polling (ADR-017).
Score thresholds are dual-mode depending on whether the reranker is loaded (ADR-005).
Switching the embedding model requires a full re-index (ADR-004).

## Environment Variables

Full annotated list in `.env.example`. Required in `.env` (Compose needs explicit DB
passwords):

- `ELASTIC_PASSWORD`, `POSTGRES_PASSWORD`
- `XRAY_CLIENT_ID`, `XRAY_CLIENT_SECRET` — Xray Cloud credentials (test sync)
- `CONFLUENCE_DOMAIN`, `CONFLUENCE_EMAIL`, `CONFLUENCE_API_TOKEN`, `JIRA_DOMAIN`
- `GITLAB_HOST`, `GITLAB_TOKEN`, `GITLAB_PROJECT_ID` — PRD ingestion from a repo
- `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — LLM provider for Phase 3
- `XRAY_MCP_URL` — Streamable HTTP endpoint (the Compose default is an **example**)

Frequently used optional settings: `QA_ENGINE_API_KEY` (enables `X-API-Key` auth),
`XRAY_PROJECT_KEY` (nightly auto-sync at 20:30 UTC), `QA_TRACE=1` (payload tracing —
see `flow.md` §5), `QA_EMBED_MAX_CHARS` (must stay above the chunker's maximum chunk —
ADR-018), and the opt-in ingestion flags `QA_CONFLUENCE_INGEST_ATTACHMENTS`,
`QA_CONFLUENCE_INGEST_CHILDREN`, `QA_INGEST_PDF_OCR` (all default off — ADR-013, ADR-014).

## Services & Ports

| Service | Port | Notes |
|---------|------|------|
| qa-agent | 8000 | Main API |
| streamlit-review | 127.0.0.1:8501 | Review UI |
| elasticsearch | 127.0.0.1:9200 | Vector store |
| postgres | internal only | Relational metadata |

Postgres init scripts in `init-db/` apply only to **empty** DB volumes; run them by hand on
an existing database (`flow.md` §6).
