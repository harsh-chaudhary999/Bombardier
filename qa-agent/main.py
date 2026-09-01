"""
QA Intelligence Engine — FastAPI entry point

Endpoints:
  GET  /health                 — Phase 0: service liveness check
  POST /sync/tests             — Phase 1: sync Xray tests → Elasticsearch (background)
  GET  /sync/status/{run_id}   — Phase 1: poll sync run progress
  POST   /ingest/prd               — Phase 2: ingest PRD document → Elasticsearch (background)
  POST   /ingest/file              — Phase 2: ingest uploaded file → Elasticsearch (background)
  GET    /ingest/status/{run_id}   — Phase 2: poll ingest run progress
  DELETE /ingest/prd/{source_id}   — Phase 2: remove all chunks for a PRD source
  GET  /runs                   — list pipeline runs (type, status, pagination)
  POST /search/tests           — Phase 3: find test cases relevant to a PRD text snippet
  POST /search/prd             — Phase 3: find PRD chunks relevant to a query
  GET  /analyze/decisions/{run_id}/export — CSV export of decisions for a run
"""
import asyncio
import csv
import io
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Annotated

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from starlette.middleware.cors import CORSMiddleware

from embeddings.embed_client import EmbedClient
from embeddings.es_store import ESStore
from embeddings.pg_store import PGStore
from sync.test_sync import run_sync
from ingestion.prd_pipeline import run_ingest, run_file_ingest
from agents.analysis_agent import run_analysis, build_preview, validate_prd_data
from agents.writeback import run_writeback
from agents.incremental import run_incremental_analysis
from embeddings.rank_filter import relative_cut, separation
from embeddings.reranker import Reranker
from integrations import webhook
from observability.phase_ledger import verify_ledger_writable
from observability.request_norm import (
    GAP_RISKS,
    merge_section_coverage,
    normalize_module_list,
    unknown_module_error,
    unknown_modules,
)

def _configure_logging() -> None:
    """
    Set up logging so our own progress is legible.

    The problem this solves: httpx logs one INFO line per request, so a sync of a few
    hundred tests produced hundreds of identical `POST http://xray-mcp:3100/mcp "200 OK"`
    lines and nothing else — while the MCP client's own per-call logging sat at DEBUG and
    was therefore invisible. Third-party transports are pushed to WARNING and our modules
    are logged at QA_LOG_LEVEL.

    Env:
      QA_LOG_LEVEL       our modules (default INFO; DEBUG for full tracing)
      QA_LOG_THIRDPARTY  transport/library level (default WARNING)
      QA_LOG_HTTP=1      re-enable per-request httpx lines when debugging transport issues
    """
    level = os.environ.get("QA_LOG_LEVEL", "INFO").upper()
    third = os.environ.get("QA_LOG_THIRDPARTY", "WARNING").upper()

    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)-28s — %(message)s",
        force=True,
    )

    # Chatty libraries. Each of these emits at least one INFO line per network call.
    noisy = [
        "httpx", "httpcore", "urllib3", "urllib3.connectionpool",
        "elastic_transport", "elastic_transport.transport", "elasticsearch",
        "mcp", "mcp.client", "sentence_transformers", "SentenceTransformer",
        "asyncio", "apscheduler", "filelock", "huggingface_hub",
    ]
    if os.environ.get("QA_LOG_HTTP", "0") == "1":
        noisy = [n for n in noisy if n not in ("httpx", "httpcore")]
    for name in noisy:
        logging.getLogger(name).setLevel(getattr(logging, third, logging.WARNING))

    # Our own packages always honour QA_LOG_LEVEL even if a library reconfigured the root.
    for name in ("agents", "embeddings", "ingestion", "integrations", "sync",
                 "observability", "eval", "review", "main"):
        logging.getLogger(name).setLevel(getattr(logging, level, logging.INFO))


_configure_logging()
logger = logging.getLogger(__name__)

# Set XRAY_PROJECT_KEY in .env to enable nightly auto-sync
DEFAULT_PROJECT_KEY = os.environ.get("XRAY_PROJECT_KEY", "")


# Providers accepted by agents.analysis_agent._build_llm
_VALID_PROVIDERS = ("anthropic", "azure_openai", "openai", "ollama")


def _env_default_provider() -> str:
    """API default LLM provider when request omits `provider` (QA_DEFAULT_PROVIDER)."""
    return os.environ.get("QA_DEFAULT_PROVIDER", "anthropic")


def _env_default_model() -> str:
    """
    API default model / deployment name when request omits `model` (QA_DEFAULT_MODEL).

    Opus rather than Sonnet: coverage analysis is low-volume, high-stakes and human-reviewed,
    and the keep/update/deprecate judgement is where the capability gap is widest. Override
    with QA_DEFAULT_MODEL for high-volume or cost-sensitive deployments.
    """
    return os.environ.get("QA_DEFAULT_MODEL", "claude-opus-5")

# ─── Confluence webhook ───────────────────────────────────────────────────────
# Shared secret for HMAC verification. NO DEFAULT ON PURPOSE: without it the endpoint
# is disabled entirely. An unauthenticated webhook that triggers analysis is a way for
# anyone who can reach the port to spend the deployment's LLM budget, so "no secret
# configured" must mean "off", never "open".
CONFLUENCE_WEBHOOK_SECRET = os.environ.get("QA_CONFLUENCE_WEBHOOK_SECRET", "").strip()
# Minimum gap between reacting to two events for the same page. Confluence fires an
# event per save, so an editing session emits a burst; without this each save would
# start its own ingest and analysis.
WEBHOOK_COOLDOWN_SEC = int(os.environ.get("QA_WEBHOOK_COOLDOWN_SEC", "900"))
# Re-analyse after re-ingesting when the page has been analysed before. Incremental,
# so an edit that changed nothing testable costs almost nothing.
WEBHOOK_ANALYZE = os.environ.get("QA_WEBHOOK_ANALYZE", "1").strip() not in ("", "0", "false", "False")
# Run a FULL analysis for a page that has never been analysed. Off by default — a full
# run on every newly-edited page in a space is a large, unbounded token spend.
WEBHOOK_ANALYZE_NEW = os.environ.get("QA_WEBHOOK_ANALYZE_NEW", "0").strip() not in ("", "0", "false", "False")

_webhook_debounce = webhook.Debounce(WEBHOOK_COOLDOWN_SEC)


# How long a decision may sit unreviewed before it is reported as overdue.
REVIEW_SLA_DAYS = int(os.environ.get("QA_REVIEW_SLA_DAYS", "30"))
# The daily check is on by default: an unreviewed DEPRECATE sitting for a month is
# exactly the failure this pipeline exists to make visible. Set 0 to silence it.
REVIEW_SLA_ALERTS = os.environ.get("QA_REVIEW_SLA_ALERTS", "1").strip() not in ("", "0", "false", "False")

# API key authentication — set QA_ENGINE_API_KEY to enable
_API_KEY = os.environ.get("QA_ENGINE_API_KEY", "")

# Max upload size: 50 MB
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Allowed upload extensions
ALLOWED_EXTENSIONS = {".xlsx", ".docx", ".pdf", ".md", ".txt", ".text"}

# Track background tasks so exceptions aren't silently lost
_background_tasks: dict[str, asyncio.Task] = {}

# Lock for auto-ingestion — bounded OrderedDict to avoid unbounded memory per source_id
_MAX_INGEST_LOCK_KEYS = 512
_ingest_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()


def _get_ingest_lock(source_id: str) -> asyncio.Lock:
    if source_id in _ingest_locks:
        _ingest_locks.move_to_end(source_id)
        return _ingest_locks[source_id]
    lock = asyncio.Lock()
    _ingest_locks[source_id] = lock
    while len(_ingest_locks) > _MAX_INGEST_LOCK_KEYS:
        _ingest_locks.popitem(last=False)
    return lock


# ─── Authentication ───────────────────────────────────────────────────────────

async def _check_api_key(request: Request):
    """
    Simple API key auth via X-API-Key header only (query params leak to logs and caches).
    Disabled if QA_ENGINE_API_KEY is not set (dev mode).
    """
    if not _API_KEY:
        return  # auth disabled
    # Keep /health usable by infra probes even when API key auth is enabled.
    if request.url.path == "/health":
        return
    key = request.headers.get("x-api-key")
    if key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def _reviewer_from_request(request: Request) -> str | None:
    """Compliance audit: prefer X-Reviewer-Id, then X-User-Email."""
    rid = (request.headers.get("x-reviewer-id") or "").strip()
    if rid:
        return rid
    email = (request.headers.get("x-user-email") or "").strip()
    return email or None


# ─── Background task helper ───────────────────────────────────────────────────

def _track_task(run_id: str, coro) -> asyncio.Task:
    """Create a tracked background task that logs exceptions instead of swallowing them."""
    existing = _background_tasks.get(run_id)
    if existing and not existing.done():
        # Deterministic mode: same run_id already in flight; ignore duplicate spawn.
        try:
            coro.close()
        except Exception:
            pass
        return existing

    task = asyncio.create_task(coro)

    def _done_callback(t: asyncio.Task):
        _background_tasks.pop(run_id, None)
        if t.cancelled():
            logger.warning(f"[{run_id}] Background task was cancelled")
        elif t.exception():
            logger.error(f"[{run_id}] Background task failed: {t.exception()}")

    task.add_done_callback(_done_callback)
    _background_tasks[run_id] = task
    return task


# ─── Background sync (with optional transient retry) ─────────────────────────

async def _run_sync_with_optional_retry(
    project_key: str,
    run_id: str,
    folder_path: str,
    full_content_refresh: bool = False,
) -> dict:
    """
    Run test sync; on transient MCP/Xray errors optionally sleep and retry in-process
    (1 min, 5 min, 15 min) while keeping the same run_id. Disable with QA_SYNC_BACKGROUND_RETRY=0.
    """
    from integrations.xray_client import _is_retryable_mcp_error

    delays = [60.0, 300.0, 900.0]
    enabled = os.environ.get("QA_SYNC_BACKGROUND_RETRY", "1") == "1"
    attempt = 0
    while True:
        try:
            return await run_sync(
                project_key=project_key,
                embed_client=app.state.embed_client,
                es_store=app.state.es_store,
                pg_store=app.state.pg_store,
                run_id=run_id,
                folder_path=folder_path,
                full_content_refresh=full_content_refresh,
            )
        except Exception as e:
            if not enabled or attempt >= len(delays) or not _is_retryable_mcp_error(e):
                raise
            logger.warning(
                "[%s] sync transient failure (%s/%s): %s — retrying after %ss",
                run_id,
                attempt + 1,
                len(delays),
                e,
                int(delays[attempt]),
            )
            app.state.pg_store.reopen_run_for_retry(run_id)
            await asyncio.sleep(delays[attempt])
            attempt += 1


# ─── Startup / shutdown ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    verify_ledger_writable()
    logger.info("Phase ledger path is writable.")

    logger.info("Loading embedding model BAAI/bge-m3 ...")
    app.state.embed_client = EmbedClient()
    logger.info("Embedding model ready.")

    logger.info("Connecting to Elasticsearch ...")
    app.state.es_store = ESStore()
    logger.info("Elasticsearch ready.")

    app.state.pg_store = PGStore()
    logger.info("Postgres relational store ready.")

    n_orphan = app.state.pg_store.fail_orphaned_running_runs(
        "service_restart: process exited before run completion"
    )
    if n_orphan:
        logger.warning("Marked %s in-flight pipeline run(s) as failed after unclean shutdown", n_orphan)

    # Cross-encoder reranker for two-stage retrieval (optional — graceful fallback)
    try:
        logger.info("Loading cross-encoder reranker ...")
        app.state.reranker = Reranker()
        logger.info("Reranker ready.")
    except Exception as e:
        logger.warning(f"Failed to load reranker: {e}. Running without reranking.")
        app.state.reranker = None

    # Verify the external MCP tool configuration. Non-fatal: the MCP server is a separate
    # deployment that may start after this one. Reported so a name mismatch surfaces here
    # rather than as an opaque tool error mid-sync.
    try:
        from integrations import xray_client

        report = await xray_client.verify_tools()
        if report["status"] == "ok":
            logger.info("MCP tools verified at %s (%s operations)",
                        report["url"], len(report["configured"]))
        elif report["status"] == "misconfigured":
            logger.error(
                "MCP tool configuration mismatch at %s — these configured tools do not exist "
                "on the server: %s. Server advertises: %s. Fix with XRAY_MCP_TOOL_MAP; see "
                "GET /integrations/mcp/tools.",
                report["url"], report["missing"], report["available"],
            )
        else:
            logger.warning(
                "MCP server unreachable at %s (%s) — sync and write-back will fail until it "
                "is up. Tool configuration was not verified.",
                report["url"], report.get("error"),
            )
    except Exception as e:
        logger.warning("MCP tool verification skipped: %s", e)

    from agents.model_tiers import describe as _describe_tiers
    _t = _describe_tiers(_env_default_provider(), _env_default_model())
    logger.info(
        "Model tiers — fast=%s reasoning=%s (tiered=%s)",
        _t["fast"], _t["reasoning"], _t["tiered"],
    )

    # Scheduled jobs. Started whenever ANY job is registered — it previously started
    # only when XRAY_PROJECT_KEY was set, so a job added later would have been silently
    # dead on any deployment that does not use nightly sync.
    scheduler = AsyncIOScheduler()

    if DEFAULT_PROJECT_KEY:
        scheduler.add_job(
            _nightly_sync,
            CronTrigger(hour=20, minute=30, timezone="UTC"),
            id="nightly_test_sync",
            replace_existing=True,
        )
        logger.info(f"Nightly sync scheduled for project '{DEFAULT_PROJECT_KEY}' at 20:30 UTC")
    else:
        logger.info("XRAY_PROJECT_KEY not set — nightly sync disabled. Trigger via POST /sync/tests")

    if REVIEW_SLA_ALERTS:
        scheduler.add_job(
            _review_sla_alert,
            CronTrigger(hour=9, minute=0, timezone="UTC"),
            id="review_sla_alert",
            replace_existing=True,
        )
        logger.info(
            "Review SLA check scheduled daily at 09:00 UTC (window: %s days)",
            REVIEW_SLA_DAYS,
        )

    if scheduler.get_jobs():
        scheduler.start()
        app.state.scheduler = scheduler

    yield

    if hasattr(app.state, "scheduler"):
        app.state.scheduler.shutdown(wait=False)
    # Cancel any remaining background tasks
    for run_id, task in list(_background_tasks.items()):
        task.cancel()
    logger.info("QA Intelligence Engine shutting down.")


async def _review_sla_alert():
    """
    Daily check for decisions still unreviewed past the SLA window.

    Emits a log line and a gauge. There is no notification channel configured here, and
    inventing one (email, Slack) would be a deployment decision this service should not
    make — the gauge is the hook: alert on `qa_decisions_overdue` in whatever already
    watches /metrics/prometheus.

    Never raises. A failing scheduled job must not take down the event loop.
    """
    try:
        pg: PGStore = app.state.pg_store
        loop = asyncio.get_running_loop()
        total = await loop.run_in_executor(
            None, lambda: pg.count_overdue_decisions(days=REVIEW_SLA_DAYS))

        try:
            from observability.metrics import metrics
            metrics.set("qa_decisions_overdue", float(total))
        except ImportError:
            pass

        if not total:
            logger.info("Review SLA: no decisions overdue (window %s days)", REVIEW_SLA_DAYS)
            return

        sample = await loop.run_in_executor(
            None, lambda: pg.get_overdue_decisions(days=REVIEW_SLA_DAYS, limit=5))
        oldest = sample[0] if sample else {}
        logger.warning(
            "Review SLA: %s decision(s) unreviewed for more than %s days. Oldest: "
            "id=%s action=%s jira_key=%s age=%.0f days (run %s). See GET /decisions/overdue",
            total, REVIEW_SLA_DAYS, oldest.get("id"), oldest.get("action"),
            oldest.get("jira_key"), oldest.get("age_days") or 0, oldest.get("run_id"),
        )
    except Exception as exc:
        logger.warning("Review SLA check failed: %s", exc)


async def _nightly_sync():
    # Full-project sync: Phase 3 skips unchanged content hashes; Phase 2 still walks Xray folders.
    # For very large projects, future work could short-circuit Phase 2 using a last-sync watermark.
    logger.info("Nightly sync triggered by scheduler")
    run_id = str(uuid.uuid4())
    await _run_sync_with_optional_retry(
        project_key=DEFAULT_PROJECT_KEY,
        run_id=run_id,
        folder_path="",
    )


limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="QA Intelligence Engine",
    description="RAG + multi-agent system for Xray test case maintenance",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(_check_api_key)],
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Browser clients (e.g. future JS UI): restrict origins; comma-separated in CORS_ALLOW_ORIGINS
_cors_env = os.environ.get(
    "CORS_ALLOW_ORIGINS",
    "http://127.0.0.1:8501,http://localhost:8501,http://streamlit-review:8501",
)
_cors_origins = [o.strip() for o in _cors_env.split(",") if o.strip()]
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ─── Health check ──────────────────────────────────────────────────────────────

@app.get("/health")
@limiter.limit("120/minute")
def health(request: Request):
    """
    Returns service health: embedding model, Elasticsearch, and Postgres.
    All three must be ok for status=ok (HTTP 200). Otherwise status=degraded (HTTP 503).
    """
    checks: dict = {}

    # 1. Elasticsearch
    try:
        es: ESStore = app.state.es_store
        if es.ping():
            info = es._client.info()
            checks["elasticsearch"] = f"ok (v{info['version']['number']})"
            for idx in ["qa_test_cases", "qa_prd_chunks"]:
                exists = es._client.indices.exists(index=idx)
                checks[f"index_{idx}"] = "ok" if exists else "missing"
        else:
            checks["elasticsearch"] = "ping failed"
    except Exception:
        checks["elasticsearch"] = "unavailable"

    # 2. Postgres — use the shared PGStore connection pool (bounded wait via PG_POOL_GETCONN_TIMEOUT_SEC)
    try:
        pg: PGStore = app.state.pg_store
        conn = pg._get_conn()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'qa_rag';"
                )
                table_count = cur.fetchone()[0]
                checks["postgres_qa_rag_tables"] = table_count
                if table_count == 0:
                    checks["postgres"] = "no tables"
                else:
                    checks["postgres"] = "ok"
        finally:
            pg._put_conn(conn)
    except TimeoutError:
        checks["postgres"] = "pool exhausted (timeout)"
        return JSONResponse(
            status_code=503,
            content={
                "status": "degraded",
                "checks": checks,
                "detail": "Postgres connection pool timeout — try again shortly.",
            },
        )
    except Exception:
        checks["postgres"] = "unavailable"

    # 3. Reranker — optional, but its absence halves retrieval precision and was
    #    previously invisible: /health never mentioned it.
    rr = getattr(app.state, "reranker", None)
    if rr is None:
        checks["reranker"] = "not loaded (retrieval runs on RRF scores only)"
    else:
        try:
            import os as _os
            checks["reranker"] = f"ok ({_os.environ.get('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-6-v2')})"
        except Exception:
            checks["reranker"] = "loaded"

    # 4. Embedding model
    try:
        embed: EmbedClient = app.state.embed_client
        test_vec = embed.embed_one("health check")
        checks["embedding_model"] = f"ok (dim={len(test_vec)})"
    except Exception:
        checks["embedding_model"] = "unavailable"

    # Keys that carry actual status (ignore numeric metadata like table counts)
    _STATUS_KEYS = {"elasticsearch", "postgres", "embedding_model"}
    all_ok = all(
        str(checks[k]).startswith("ok")
        for k in _STATUS_KEYS
        if k in checks
    ) and all(
        str(v) != "missing"
        for k, v in checks.items()
        if k.startswith("index_")
    )
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ok" if all_ok else "degraded", "checks": checks},
    )


# ─── Phase 1: Test sync ────────────────────────────────────────────────────────

class SyncTestsRequest(BaseModel):
    project_key: str
    folder_path: str = ""   # empty = all tests in the project
    # Re-read steps/description for every test instead of trusting the bulk metadata diff.
    # Costs one MCP call per test; see sync.test_sync._metadata_hash for why it is needed.
    full_content_refresh: bool = False


@app.post("/sync/tests", status_code=202)
@limiter.limit("30/minute")
async def sync_tests(request: Request, req: SyncTestsRequest):
    """
    Trigger a full Xray → Elasticsearch test sync in the background.

    Returns immediately with a run_id.
    Poll GET /sync/status/{run_id} to track progress.

    Body:
        project_key  — Jira project key, e.g. "PROJ"
        folder_path  — (optional) restrict sync to one folder, e.g. "/Platform"
        full_content_refresh — (optional) re-read steps/description for every test. The
                       default incremental diff only sees summary/folder/labels/type/updated
                       from the bulk listing, so a steps-only edit in Xray can stay stale in
                       the index; the response field `content_unverified` reports how many
                       tests were accepted without a content read.
    """
    run_id = str(uuid.uuid4())

    _track_task(
        run_id,
        _run_sync_with_optional_retry(
            project_key=req.project_key,
            run_id=run_id,
            folder_path=req.folder_path,
            full_content_refresh=req.full_content_refresh,
        ),
    )

    return {
        "run_id": run_id,
        "status": "started",
        "message": f"Syncing '{req.project_key}'. Poll /sync/status/{run_id} for progress.",
    }


@app.get("/sync/status/{run_id}")
def sync_status(run_id: str):
    """Poll the status of a sync (or any pipeline) run."""
    pg: PGStore = app.state.pg_store
    row = pg.get_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"No run found: '{run_id}'")
    return row


@app.get("/runs")
def list_pipeline_runs(
    run_type: Annotated[str | None, Query(alias="type")] = None,
    status: str | None = None,
    limit: int = 20,
    page: int = 1,
):
    """
    List recent pipeline runs (sync, ingest, analysis, etc.) newest first.

    Query: type (maps to run_type), status, page (1-based), limit (max 200).
    """
    pg: PGStore = app.state.pg_store
    try:
        from observability.metrics import metrics

        metrics.inc("http_runs_list_total")
    except ImportError:
        pass
    page = max(1, page)
    page_size = min(max(1, limit), 200)
    offset = (page - 1) * page_size
    rows = pg.list_runs(run_type=run_type, status=status, limit=page_size, offset=offset)
    return {
        "runs": rows,
        "page": page,
        "page_size": page_size,
        "run_type": run_type,
        "status": status,
    }


# ─── Phase 2: PRD ingestion ────────────────────────────────────────────────────


class IngestPRDRequest(BaseModel):
    source_type: str           # confluence | confluence_space | confluence_site | gitlab | gitlab_file
    source: str = ""           # page URL/ID, space key, module name, or file path
    module: str | None = None  # module label; defaults to the space key for space/site ingests
    ref: str = "main"          # git branch/tag for gitlab sources
    # confluence_space / confluence_site options
    title_filter: str = ""     # optional: only pages whose title contains this string
    parent_id: str = ""        # optional: only pages under this parent page ID (space only)
    space_workers: int = Field(default=5, ge=1, le=20)  # parallel page fetchers
    # confluence_site options
    space_keys: str = ""       # comma-separated allowlist; empty = every eligible space
    include_personal: bool = False   # include personal (~user) spaces
    include_archived: bool = False   # include archived spaces
    # Re-ingest even when the Confluence version matches what is indexed.
    # Required after an embedding-model change — versions still match but vectors are stale.
    force: bool = False


@app.post("/ingest/prd", status_code=202)
@limiter.limit("20/minute")
async def ingest_prd(request: Request, req: IngestPRDRequest):
    """
    Ingest a PRD document (or entire space) into Elasticsearch in the background.

    source_type options:
      confluence       — single Confluence page URL or page ID
                         e.g. source="1234567890"
      confluence_space — EVERY page in one Confluence space, at any nesting depth
                         e.g. source="DOCS"  (your Confluence space key)
                         optional: title_filter="PRD" to only ingest pages with "PRD" in title
                         optional: parent_id="1234567" to restrict to a subtree
      confluence_site  — EVERY page in EVERY selected space on the site. `source` is not used.
                         optional: space_keys="DOCS,PLAT" to restrict (empty = all spaces)
                         optional: include_personal / include_archived (both default false)
      gitlab           — All .md files for a module folder in the GitLab repo
                         e.g. source="Platform"
      gitlab_file      — Single .md file in the GitLab repo
                         e.g. source="Platform/docs/Features.md"

    Space and site ingests are INCREMENTAL: a page is re-fetched and re-embedded only if
    its Confluence version differs from what is indexed, so re-running to pick up edits is
    cheap. Chunks are tagged with module=<space key> so module filters work per space.
    Set force=true to re-ingest everything — needed after an embedding-model change, where
    versions still match but the stored vectors are no longer valid.

    Returns immediately with a run_id.
    Poll GET /ingest/status/{run_id} for progress.
    """
    valid_types = ("confluence", "confluence_space", "confluence_site", "gitlab", "gitlab_file")
    if req.source_type not in valid_types:
        raise HTTPException(
            status_code=400,
            detail=f"source_type must be one of: {valid_types}"
        )
    # Every type except confluence_site addresses a specific document/space.
    if req.source_type != "confluence_site" and not req.source.strip():
        raise HTTPException(
            status_code=400,
            detail=f"source is required for source_type={req.source_type!r}",
        )

    run_id = str(uuid.uuid4())

    _track_task(
        run_id,
        run_ingest(
            source_type=req.source_type,
            source=req.source,
            embed_client=app.state.embed_client,
            es_store=app.state.es_store,
            pg_store=app.state.pg_store,
            run_id=run_id,
            module=req.module,
            ref=req.ref,
            title_filter=req.title_filter,
            parent_id=req.parent_id,
            space_workers=req.space_workers,
            space_keys=req.space_keys,
            include_personal=req.include_personal,
            include_archived=req.include_archived,
            force=req.force,
        ),
    )

    return {
        "run_id": run_id,
        "status": "started",
        "message": f"Ingesting {req.source_type}:{req.source!r}. Poll /ingest/status/{run_id} for progress.",
    }


@app.post("/ingest/file", status_code=202)
@limiter.limit("20/minute")
async def ingest_file_upload(
    request: Request,
    file: UploadFile = File(...),
    source_label: str = "",
):
    """
    Ingest an uploaded file (Excel, Word, PDF, Markdown, text) into Elasticsearch.

    Supported formats: .xlsx, .docx, .pdf, .md, .txt

    Query param:
      source_label — optional label for this document (defaults to filename).
                     Used as the source_id key in Elasticsearch.
                     Re-uploading with the same label replaces the previous version.

    Returns immediately with a run_id.
    Poll GET /ingest/status/{run_id} for progress.
    """
    filename = file.filename or "upload"

    # Validate file extension
    ext = os.path.splitext(filename.lower())[1]
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {ext!r}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    # Read with size limit
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content)} bytes. Max: {MAX_UPLOAD_BYTES} bytes ({MAX_UPLOAD_BYTES // (1024*1024)} MB)"
        )

    run_id = str(uuid.uuid4())

    _track_task(
        run_id,
        run_file_ingest(
            filename=filename,
            content=content,
            embed_client=app.state.embed_client,
            es_store=app.state.es_store,
            pg_store=app.state.pg_store,
            run_id=run_id,
            source_label=source_label,
        ),
    )

    return {
        "run_id":   run_id,
        "status":   "started",
        "filename": filename,
        "message":  f"Processing {filename!r}. Poll /ingest/status/{run_id} for progress.",
    }


@app.get("/ingest/status/{run_id}")
def ingest_status(run_id: str):
    """Poll the status of a PRD ingest run."""
    pg: PGStore = app.state.pg_store
    row = pg.get_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"No run found: '{run_id}'")
    return row


@app.delete("/ingest/prd/{source_id:path}", status_code=200)
@limiter.limit("30/minute")
def delete_prd_source(request: Request, source_id: str):
    """
    Delete all indexed chunks for a PRD source from Elasticsearch.

    source_id is the identifier used when the document was ingested, e.g.:
      confluence:12345
      file:upload:my-spec
      gitlab:group/project:path/to/doc.md

    Use the :path converter so source_ids with slashes are captured correctly.
    Returns the number of chunks deleted.
    """
    es: ESStore = request.app.state.es_store
    deleted = es.delete_prd_source(source_id)
    if deleted == 0:
        raise HTTPException(
            status_code=404,
            detail=f"No indexed chunks found for source_id={source_id!r}",
        )
    return {"source_id": source_id, "deleted_chunks": deleted}


# ─── Phase 3: Similarity search ────────────────────────────────────────────────

class SearchTestsRequest(BaseModel):
    prd_text: str                    # PRD section / feature description to search with
    module: list[str] | None = None  # optional: restrict to specific module(s)
    top_k: int = Field(default=20, ge=1, le=200)
    mode: str = "semantic"           # "semantic" | "hybrid"
    min_score: float = Field(default=0.5, ge=0.0, le=1.0)


@app.post("/search/tests")
@limiter.limit("120/minute")
async def search_tests(request: Request, req: SearchTestsRequest):
    """
    Find existing test cases most relevant to a PRD text snippet.

    Use this to answer: "Which tests cover this feature?"

    mode:
      semantic — pure KNN vector similarity. Best for finding conceptually related tests
                 even when they use different words (e.g. "buy" vs "place order").
      hybrid   — KNN + BM25 keyword fusion via Reciprocal Rank Fusion (RRF). Better when
                 the PRD uses specific names/IDs that should boost exact keyword matches.

    module filter examples: ["Platform"], ["Billing", "Docs"]
    """
    if req.mode not in ("semantic", "hybrid"):
        raise HTTPException(status_code=400, detail="mode must be 'semantic' or 'hybrid'")
    if not req.prd_text.strip():
        raise HTTPException(status_code=400, detail="prd_text cannot be empty")

    module_n = normalize_module_list(req.module)
    loop = asyncio.get_running_loop()
    embed: EmbedClient = app.state.embed_client
    es: ESStore = app.state.es_store

    # Embed query — runs on CPU, use executor to avoid blocking the event loop
    query_vec = await loop.run_in_executor(
        None, lambda: embed.embed_query(req.prd_text)
    )

    if req.mode == "hybrid":
        reranker = getattr(app.state, "reranker", None)
        retrieval_k = req.top_k * 3 if reranker else req.top_k
        results = await loop.run_in_executor(
            None,
            lambda: es.search_hybrid(
                query_embedding=query_vec,
                keyword_query=req.prd_text,
                top_k=retrieval_k,
                module_filter=module_n,
            ),
        )
        if reranker and results:
            results = await loop.run_in_executor(
                None,
                lambda: reranker.rerank(req.prd_text, results, top_k=req.top_k),
            )
    else:
        results = await loop.run_in_executor(
            None,
            lambda: es.search_similar_tests(
                query_embedding=query_vec,
                top_k=req.top_k,
                module_filter=module_n,
                min_score=req.min_score,
            ),
        )

    return {
        "query":         req.prd_text[:300],
        "mode":          req.mode,
        "module_filter": module_n,
        "total":         len(results),
        "results":       results,
    }


class SearchPRDRequest(BaseModel):
    query: str                    # free-text query or PRD topic
    source_id: str | None = None  # optional: restrict to one ingested document
    top_k: int = Field(default=10, ge=1, le=200)
    module: list[str] | None = None
    # Structural scoping — the same filters /ask exposes. Requirements are a small
    # fraction of a whole-space ingest, so doc_types=["prd"] removes far more noise
    # than any score threshold can.
    title_contains: str | None = None
    doc_types: list[str] | None = None
    exclude_doc_types: list[str] | None = None
    # Trim the tail at the largest relative score drop. Off by default so the endpoint
    # keeps returning what was asked for; the `separation` block is reported either way
    # so a caller can see whether score is worth trusting before opting in.
    trim: bool = False


@app.post("/search/prd")
@limiter.limit("120/minute")
async def search_prd(request: Request, req: SearchPRDRequest):
    """
    Find PRD chunks most relevant to a query.

    Use this to answer: "What does the PRD say about this feature?"
    Optionally restrict to a specific ingested document via source_id.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="query cannot be empty")

    loop = asyncio.get_running_loop()
    embed: EmbedClient = app.state.embed_client
    es: ESStore = app.state.es_store

    query_vec = await loop.run_in_executor(
        None, lambda: embed.embed_query(req.query)
    )

    results = await loop.run_in_executor(
        None,
        lambda: es.search_similar_prd_chunks(
            query_embedding=query_vec,
            top_k=req.top_k,
            source_id=req.source_id,
            module_filter=normalize_module_list(req.module),
            title_contains=req.title_contains,
            doc_types=req.doc_types,
            exclude_doc_types=req.exclude_doc_types,
        ),
    )

    # Score distributions on this corpus are heavily compressed — a correct answer and an
    # unrelated page can land within a few percent of each other, which makes every absolute
    # threshold useless. `separation` says whether score can discriminate at all, so a caller
    # can tell a confident ranking from a flat one instead of assuming rank order means something.
    kept, diag = relative_cut(results, min_keep=min(3, len(results)) or 1) if req.trim \
        else (results, separation(results))

    return {
        "query":      req.query,
        "source_id":  req.source_id,
        "total":      len(kept),
        "retrieved":  len(results),
        "separation": diag,
        "results":    kept,
    }


# ─── Embedding explain ─────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str
    top_k: int = Field(default=8, ge=1, le=30)
    module: list[str] | None = None          # restrict to module(s)
    source_id: str | None = None             # restrict to one ingested document
    # Narrow a broad corpus at query time instead of re-ingesting, e.g. title_contains="PRD"
    # when a whole Confluence space was indexed and tech docs / test plans compete with
    # the actual requirements.
    title_contains: str | None = None
    # Document-type scoping. Requirements are ~4% of a whole-space ingest, so
    # doc_types=["prd"] is the single biggest noise reduction available.
    # prd | tech_doc | implementation_plan | test_plan | release_note | meeting_note | other
    doc_types: list[str] | None = None
    exclude_doc_types: list[str] | None = None
    include_tests: bool = False              # also search Xray test cases
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)
    provider: str = Field(default_factory=_env_default_provider)
    model: str = Field(default_factory=_env_default_model)
    # "fast" uses provider/model as given. "reasoning" escalates to
    # QA_REASONING_PROVIDER/QA_REASONING_MODEL when configured, else falls back.
    tier: str = "fast"


@app.post("/ask")
@limiter.limit("30/minute")
async def ask(request: Request, req: AskRequest):
    """
    Ask a question in English; get a readable answer that cites its sources.

    This is the gap between /search/prd (raw chunks, no synthesis) and /analyze/prd
    (structured coverage decisions, not answers).

    Every factual claim is cited [n], and the passages the model was given come back in
    `context_used` — so a wrong answer can be attributed to retrieval (the fact was never
    supplied) or to the model (it was supplied and ignored) without a second call.

    `grounded: false` means the answer cited nothing or abstained; treat it as unverified.
    `abstained: true` means the corpus genuinely does not cover the question — which is a
    correct outcome, not a failure.

    Scoping:
      title_contains="PRD"   only documents whose title contains "PRD"
      source_id=...          one specific document
      module=["Platform"]    module-tagged chunks (plus untagged)
      include_tests=true     also search Xray test cases — answers "which tests cover X?",
                             but can be circular if your corpus contains test plans
    """
    if req.provider not in _VALID_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"provider must be one of: {_VALID_PROVIDERS}")
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")
    from agents.model_tiers import TIERS
    if req.tier not in TIERS:
        # Reject rather than degrade: the caller explicitly asked to escalate, and
        # silently answering with the cheap model would misrepresent the result.
        raise HTTPException(status_code=400, detail=f"tier must be one of: {TIERS}")
    from ingestion.doc_classify import DOC_TYPES
    for field, vals in (("doc_types", req.doc_types), ("exclude_doc_types", req.exclude_doc_types)):
        bad = [v for v in (vals or []) if v not in DOC_TYPES]
        if bad:
            # An unrecognised type is dropped by title_filter, so the request would
            # silently return an unfiltered corpus — the opposite of what was asked.
            raise HTTPException(
                status_code=400,
                detail=f"{field} contains unknown values {bad}; valid: {list(DOC_TYPES)}",
            )

    from agents.ask import answer_question

    module_n = normalize_module_list(req.module)
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: answer_question(
            req.question,
            app.state.embed_client,
            app.state.es_store,
            provider=req.provider,
            model=req.model,
            top_k=req.top_k,
            module=module_n,
            source_id=req.source_id,
            title_contains=req.title_contains,
            doc_types=req.doc_types,
            exclude_doc_types=req.exclude_doc_types,
            include_tests=req.include_tests,
            min_score=req.min_score,
            reranker=getattr(app.state, "reranker", None),
            tier=req.tier,
        ),
    )
    return result


@app.get("/explain/test/{jira_key}")
def explain_test_embedding(jira_key: str):
    """
    Show the exact text that was (or would be) fed to the embedding model
    for a given test case. Use this to verify embedding quality.

    Returns:
      stored_fields   — what is saved in Elasticsearch
      embedding_input — the exact string passed to embed_document()
      embedding_dim   — confirms embedding exists (length of vector)
    """
    es: ESStore = app.state.es_store
    resp = es._client.search(
        index="qa_test_cases",
        query={"term": {"jira_key": jira_key}},
        source=["jira_key", "summary", "module", "labels",
                "description", "steps_text", "folder_path", "embedding"],
        size=1,
    )
    hits = resp["hits"]["hits"]
    if not hits:
        raise HTTPException(status_code=404, detail=f"Test case {jira_key!r} not found")

    src = hits[0]["_source"]
    embed: EmbedClient = app.state.embed_client

    embedding_input = embed.format_test_case(
        summary=src.get("summary", ""),
        module=src.get("module"),
        labels=src.get("labels") or None,
        description=src.get("description") or None,
        steps_text=src.get("steps_text") or None,
    )
    vec = src.get("embedding") or []

    return {
        "jira_key":       jira_key,
        "stored_fields": {
            "summary":     src.get("summary"),
            "module":      src.get("module"),
            "labels":      src.get("labels"),
            "description": src.get("description") or "(empty)",
            "steps_text":  src.get("steps_text") or "(empty)",
            "folder_path": src.get("folder_path"),
        },
        "embedding_input": embedding_input,
        "embedding_dim":   len(vec),
        "has_embedding":   len(vec) > 0,
    }


@app.get("/explain/prd/{source_id:path}")
def explain_prd_embedding(source_id: str, chunk_index: int = 0):
    """
    Show the exact text that was fed to the embedding model for a PRD chunk.

    source_id: e.g. confluence:1234567890
    chunk_index: which chunk to inspect (default 0 = first chunk)

    Returns:
      stored_fields   — what is saved in Elasticsearch
      embedding_input — the exact string passed to embed_document()
    """
    es: ESStore = app.state.es_store
    embed: EmbedClient = app.state.embed_client

    resp = es._client.search(
        index="qa_prd_chunks",
        query={"bool": {"must": [
            {"term": {"source_id": source_id}},
            {"term": {"chunk_index": chunk_index}},
        ]}},
        source=["source_id", "doc_title", "section_heading",
                "chunk_text", "chunk_type", "chunk_index", "embedding"],
        size=1,
    )
    hits = resp["hits"]["hits"]
    if not hits:
        raise HTTPException(
            status_code=404,
            detail=f"No chunk found for source_id={source_id!r} chunk_index={chunk_index}"
        )

    src = hits[0]["_source"]
    embedding_input = embed.format_prd_chunk(
        section_heading=src.get("section_heading"),
        chunk_text=src.get("chunk_text", ""),
    )
    vec = src.get("embedding") or []

    return {
        "source_id":    source_id,
        "chunk_index":  chunk_index,
        "stored_fields": {
            "doc_title":       src.get("doc_title"),
            "section_heading": src.get("section_heading"),
            "chunk_text":      src.get("chunk_text"),   # full text, no truncation
            # table | code | mixed | prose. Null on chunks indexed before the field
            # existed. A PRD you know is table-heavy showing "prose" here means the
            # source conversion flattened its tables.
            "chunk_type":      src.get("chunk_type"),
        },
        "embedding_input": embedding_input,             # exactly what the model saw
        "embedding_dim":   len(vec),
        "has_embedding":   len(vec) > 0,
    }


# ─── Phase 3: Analysis agent ───────────────────────────────────────────────────

_VALID_SOURCE_TYPES = ("confluence", "confluence_space", "gitlab", "gitlab_file", "file")


def _parse_source_id(prd_source_id: str) -> tuple[str, str]:
    """
    Derive (source_type, source) from a prd_source_id string.

    Formats:
      confluence:1234567890   → ("confluence", "1234567890")
      gitlab:Platform         → ("gitlab", "Platform")
      gitlab_file:path/to/f  → ("gitlab_file", "path/to/f")
      file:some label         → ("file", "some label")
    """
    if ":" not in prd_source_id:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot parse source_id {prd_source_id!r}. Expected 'type:source'."
        )
    source_type, _, source = prd_source_id.partition(":")
    source_type = source_type.strip()
    source = source.strip()
    if source_type not in _VALID_SOURCE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown source type {source_type!r} in {prd_source_id!r}. Valid: {_VALID_SOURCE_TYPES}"
        )
    return source_type, source


async def _ensure_ingested(
    prd_source_id: str,
    pg_store: PGStore,
    embed_client,
    es_store,
    module: list[str] | None = None,
) -> dict:
    """
    Check if prd_source_id has chunks in ES. If not, auto-ingest it first.
    Uses per-source locking to prevent duplicate concurrent ingests.

    Returns already_ingested, chunk count, optional ingest_result, and ingest_run_id when auto-ingest runs.
    """
    async with _get_ingest_lock(prd_source_id):
        # Quick count check
        resp = es_store._client.count(
            index="qa_prd_chunks",
            query={"term": {"source_id": prd_source_id}},
        )
        existing = resp["count"]
        if existing > 0:
            return {
                "already_ingested": True,
                "chunks": existing,
                "ingest_result": None,
                "ingest_run_id": None,
            }

        # Not ingested yet — auto-ingest (tracked run_id for polling /ingest/status)
        logger.info(f"Auto-ingesting {prd_source_id!r} (0 chunks found in ES)")
        source_type, source = _parse_source_id(prd_source_id)

        if source_type == "file":
            raise HTTPException(
                status_code=400,
                detail=f"{prd_source_id!r} is a file upload source — use POST /ingest/file to ingest it first.",
            )

        ingest_run_id = str(uuid.uuid4())
        module_tag = module[0] if module else None
        ingest_result = await run_ingest(
            source_type=source_type,
            source=source,
            embed_client=embed_client,
            es_store=es_store,
            pg_store=pg_store,
            run_id=ingest_run_id,
            module=module_tag,
        )
        chunks_ingested = ingest_result.get("chunks_ingested", 0)
        logger.info(f"Auto-ingest complete: {chunks_ingested} chunks for {prd_source_id!r}")
        return {
            "already_ingested": False,
            "chunks": chunks_ingested,
            "ingest_result": ingest_result,
            "ingest_run_id": ingest_run_id,
        }


def _require_known_modules(module_n: list[str] | None, es: ESStore) -> None:
    """
    Reject a module filter that matches nothing in the test index.

    The decision itself lives in observability.request_norm so it is testable without
    the web stack; this wrapper only turns it into an HTTP response and logs the
    partial-match case.
    """
    if not module_n:
        return
    available = es.get_available_modules()
    error = unknown_module_error(module_n, available)
    if error:
        raise HTTPException(status_code=400, detail=error)
    missing = unknown_modules(module_n, available)
    if missing:
        logger.warning(
            "Module filter contains %s unknown module(s), searching the rest: %s",
            len(missing), missing,
        )


class AnalysePRDRequest(BaseModel):
    prd_source_id: str              # ES source_id, e.g. "confluence:1234567890"
    module: list[str] | None = None # restrict test search to these modules
    top_k: int = Field(default=25, ge=1, le=200)
    provider: str = Field(default_factory=_env_default_provider)  # anthropic | azure_openai | openai | ollama
    model: str = Field(default_factory=_env_default_model)  # model or deployment name for the provider
    sample_queries: list[str] | None = None  # preview only: run these searches to show retrieval results


@app.post("/analyze/prd", status_code=202)
@limiter.limit("15/minute")
async def analyse_prd(request: Request, req: AnalysePRDRequest):
    """
    Run the PRD coverage analysis agent in the background.

    If the PRD has not been ingested yet, it is ingested automatically first.

    prd_source_id format:
      confluence:{page_id}      e.g. "confluence:1234567890"
      gitlab:{module}           e.g. "gitlab:Platform"
      file:upload:{label}      file uploads (use same id as returned by /ingest/file)

    Poll GET /analyze/status/{run_id} for progress.
    Review decisions at GET /analyze/decisions/{run_id}.
    """
    if req.provider not in _VALID_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"provider must be one of: {_VALID_PROVIDERS}")

    try:
        from observability.metrics import metrics

        metrics.inc("api_analyze_prd_requests_total")
    except ImportError:
        pass

    module_n = normalize_module_list(req.module)
    if module_n:
        logger.info("analyze/prd module filter (normalized): %s", module_n)
    _require_known_modules(module_n, app.state.es_store)

    ensured = await _ensure_ingested(
        req.prd_source_id,
        app.state.pg_store,
        app.state.embed_client,
        app.state.es_store,
        module=module_n,
    )

    if os.environ.get("QA_DETERMINISTIC_ANALYSIS_RUN_ID", "1") == "1":
        from observability.run_identity import deterministic_analysis_run_id

        run_id = deterministic_analysis_run_id(req.prd_source_id, module_n)
    else:
        run_id = str(uuid.uuid4())
    _track_task(
        run_id,
        run_analysis(
            prd_source_id=req.prd_source_id,
            module=module_n,
            embed_client=app.state.embed_client,
            es_store=app.state.es_store,
            pg_store=app.state.pg_store,
            run_id=run_id,
            top_k=req.top_k,
            provider=req.provider,
            model=req.model,
            reranker=getattr(app.state, "reranker", None),
        ),
    )

    out = {
        "run_id":        run_id,
        "status":        "started",
        "prd_source_id": req.prd_source_id,
        "message":       f"Analysis running. Poll /analyze/status/{run_id} for progress.",
    }
    if ensured.get("ingest_run_id"):
        out["auto_ingest_run_id"] = ensured["ingest_run_id"]
    return out


@app.get("/analyze/status/{run_id}")
def analyse_status(run_id: str):
    """Poll the status of an analysis run."""
    pg: PGStore = app.state.pg_store
    row = pg.get_run(run_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"No run found: '{run_id}'")
    return row


@app.post("/analyze/preview")
@limiter.limit("30/minute")
async def analyse_preview(request: Request, req: AnalysePRDRequest):
    """
    Preview the exact prompt that would be sent to the LLM — no LLM call made.
    If the PRD has not been ingested yet, it is ingested automatically first.
    """
    module_n = normalize_module_list(req.module)
    _require_known_modules(module_n, app.state.es_store)
    await _ensure_ingested(
        req.prd_source_id,
        app.state.pg_store,
        app.state.embed_client,
        app.state.es_store,
        module=module_n,
    )
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: build_preview(
            prd_source_id=req.prd_source_id,
            module=module_n,
            embed_client=app.state.embed_client,
            es_store=app.state.es_store,
            provider=req.provider,
            model=req.model,
            sample_queries=req.sample_queries,
            reranker=getattr(app.state, "reranker", None),
        ),
    )
    return result


class ValidatePRDRequest(BaseModel):
    prd_source_id: str
    module: list[str] | None = None
    top_k_tests: int = Field(default=10, ge=1, le=200)
    top_k_kb: int = Field(default=5, ge=1, le=50)


@app.post("/analyze/validate")
@limiter.limit("30/minute")
async def analyse_validate(request: Request, req: ValidatePRDRequest):
    """
    Pre-flight data check — no LLM involved.
    If the PRD has not been ingested yet, it is ingested automatically first.

    Verifies:
      1. The PRD chunks (count, headings, content preview)
      2. For each section heading, what test cases would the agent retrieve
      3. For each section heading, what other knowledge-base/Confluence docs match
    """
    module_n = normalize_module_list(req.module)
    _require_known_modules(module_n, app.state.es_store)
    await _ensure_ingested(
        req.prd_source_id,
        app.state.pg_store,
        app.state.embed_client,
        app.state.es_store,
        module=module_n,
    )
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(
        None,
        lambda: validate_prd_data(
            prd_source_id=req.prd_source_id,
            module=module_n,
            embed_client=app.state.embed_client,
            es_store=app.state.es_store,
            top_k_tests=req.top_k_tests,
            top_k_kb=req.top_k_kb,
            reranker=getattr(app.state, "reranker", None),
        ),
    )
    return result


@app.get("/analyze/decisions/{run_id}")
def analyse_decisions(
    run_id: str,
    page: int = 1,
    page_size: int = 50,
    paginate: bool = True,
):
    """
    Fetch decisions for an analysis run, grouped by action.
    Use page & page_size (1-based page) to paginate large result sets; set paginate=false for full dump.
    """
    pg: PGStore = app.state.pg_store
    if paginate:
        page = max(1, page)
        page_size = max(1, min(page_size, 500))
        offset = (page - 1) * page_size
        rows, total = pg.get_pending_decisions_page(run_id, limit=page_size, offset=offset)
    else:
        rows = pg.get_pending_decisions(run_id=run_id)
        total = len(rows)

    grouped: dict[str, list] = {"keep": [], "update": [], "deprecate": [], "create": [], "question": []}
    for row in rows:
        action = row.get("action", "unknown")
        grouped.setdefault(action, []).append(row)

    summary_full = pg.decision_counts_by_run(run_id)

    out: dict = {
        "run_id":    run_id,
        "total":     total,
        "summary": summary_full,
        "decisions": grouped,
    }
    if paginate:
        out["page"] = page
        out["page_size"] = page_size
    return out


def _row_to_csv_fields(row: dict) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, v in row.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (dict, list)):
            out[k] = json.dumps(v, ensure_ascii=False, default=str)
        elif isinstance(v, bool):
            out[k] = "1" if v else "0"
        else:
            out[k] = str(v)
    return out


@app.get("/analyze/decisions/{run_id}/export")
@limiter.limit("60/minute")
def export_analyse_decisions_csv(request: Request, run_id: str):
    """
    Download all decisions for a run as CSV (spreadsheet-friendly, offline review).
    """
    pg: PGStore = app.state.pg_store
    rows = pg.get_pending_decisions(run_id=run_id)
    if not rows:
        meta = pg.get_run(run_id)
        if not meta:
            raise HTTPException(status_code=404, detail=f"No run found: '{run_id}'")

    fieldnames = sorted({k for r in rows for k in r.keys()}) if rows else [
        "id",
        "run_id",
        "jira_key",
        "action",
        "reason",
        "prd_source",
        "prd_section",
        "reviewed",
        "approved",
        "reviewer_note",
        "reviewed_at",
        "reviewed_by",
        "written_back",
        "created_at",
        "updated_content",
        "questions",
    ]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow(_row_to_csv_fields(r))
    return PlainTextResponse(
        content=buf.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="decisions_{run_id}.csv"'},
    )


# ─── Phase 4: Human review ───────────────────────────────────────────────────

class ReviewDecisionRequest(BaseModel):
    decision_id: int
    approved: bool
    reviewer_note: str | None = None


class BulkReviewItem(BaseModel):
    decision_id: int
    approved: bool


class BulkReviewRequest(BaseModel):
    decisions: list[BulkReviewItem]
    reviewer_note: str | None = None


@app.post("/review/decisions/bulk")
@limiter.limit("60/minute")
def bulk_review_decisions(request: Request, req: BulkReviewRequest):
    """
    Approve or reject many decisions in one request (single DB transaction).
    """
    pg: PGStore = app.state.pg_store
    items = [(x.decision_id, x.approved) for x in req.decisions]
    n = pg.approve_decisions_batch(
        items,
        reviewer_note=req.reviewer_note,
        reviewed_by=_reviewer_from_request(request),
    )
    return {"updated": n, "requested": len(req.decisions)}


@app.post("/review/decision")
@limiter.limit("120/minute")
def review_decision(request: Request, req: ReviewDecisionRequest):
    """
    Approve or reject a single analysis decision.
    Used by the Streamlit review UI.
    """
    pg: PGStore = app.state.pg_store
    updated = pg.approve_decision(
        req.decision_id,
        req.approved,
        req.reviewer_note,
        reviewed_by=_reviewer_from_request(request),
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Decision {req.decision_id} not found")
    return {
        "decision_id": req.decision_id,
        "approved": req.approved,
        "status": "reviewed",
    }


# ─── Phase 5: Write-back to Xray ─────────────────────────────────────────────

class WritebackRequest(BaseModel):
    run_id: str | None = None     # optional: restrict to one run
    project_key: str = ""         # required for CREATE actions
    dry_run: bool = False         # preview only — no Xray/Jira MCP calls


@app.post("/writeback/execute", status_code=200)
@limiter.limit("30/minute")
async def execute_writeback(request: Request, req: WritebackRequest):
    """
    Write back approved decisions to Xray/Jira.

    Actions:
      KEEP      → mark written_back (no Xray action)
      UPDATE    → update test steps/summary in Xray
      DEPRECATE → add DEPRECATED label + comment in Jira
      CREATE    → bulk-create new tests in Xray (requires project_key)
      QUESTION  → post the question as a Jira comment on the test

    DEPRECATE snapshots the test's labels first, so the change can be undone with
    POST /writeback/rollback/{decision_id}.

    Blocking: waits for all MCP calls to finish (HTTP 200). Use dry_run for a safe preview.
    Global write-back (all runs) requires QA_WRITEBACK_ALLOW_GLOBAL=1.
    """
    if req.run_id is None and os.environ.get("QA_WRITEBACK_ALLOW_GLOBAL", "") != "1":
        raise HTTPException(
            status_code=400,
            detail=(
                "run_id is required. Set QA_WRITEBACK_ALLOW_GLOBAL=1 only if you intend to "
                "write back every approved decision across all analysis runs."
            ),
        )
    result = await run_writeback(
        pg_store=app.state.pg_store,
        run_id=req.run_id,
        project_key=req.project_key,
        dry_run=req.dry_run,
        es_store=app.state.es_store,
    )
    return result


@app.post("/webhooks/confluence", status_code=202)
@limiter.limit("60/minute")
async def confluence_webhook(request: Request):
    """
    Receive Confluence page events and refresh the affected PRD.

    Configure in Confluence (Settings → Webhooks) pointing at this URL, with the same
    secret as QA_CONFLUENCE_WEBHOOK_SECRET. **Without that secret the endpoint is
    disabled** — it triggers work that costs money, so it is never open by default.

    What happens on an accepted event:
      1. The page is re-ingested (forced, because its Confluence version has changed
         but a matching version would otherwise skip it).
      2. If it has been analysed before, an incremental analysis runs — which is nearly
         free when the edit changed nothing testable.
      3. If it has never been analysed, nothing further happens unless
         QA_WEBHOOK_ANALYZE_NEW=1, since a full run per newly-edited page is unbounded.

    Repeat events for the same page inside QA_WEBHOOK_COOLDOWN_SEC are acknowledged and
    dropped: Confluence fires once per save, so an editing session is a burst.

    Always returns 202 for events it understands but chooses not to act on — a webhook
    sender treats a 4xx as a delivery failure and will retry it.
    """
    if not CONFLUENCE_WEBHOOK_SECRET:
        raise HTTPException(
            status_code=503,
            detail="Webhook endpoint is disabled. Set QA_CONFLUENCE_WEBHOOK_SECRET to "
                   "enable it, and configure the same secret in Confluence.",
        )

    body = await request.body()
    if not webhook.signature_ok(
            CONFLUENCE_WEBHOOK_SECRET, body,
            request.headers.get("x-hub-signature-256")):
        logger.warning("Rejected Confluence webhook: bad or missing signature")
        raise HTTPException(status_code=401, detail="Invalid or missing signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Body is not valid JSON")

    event = webhook.event_name(payload)
    if not webhook.should_trigger(event):
        return {"status": "ignored", "reason": f"event {event!r} is not a page change"}

    page_id = webhook.page_id(payload)
    if not page_id:
        return {"status": "ignored", "reason": "no numeric page id in payload"}

    wait = _webhook_debounce.check(page_id)
    if wait:
        return {"status": "debounced", "page_id": page_id, "retry_after_s": wait}

    prd_source_id = f"confluence:{page_id}"
    pg: PGStore = app.state.pg_store
    loop = asyncio.get_running_loop()
    previous = await loop.run_in_executor(
        None, lambda: pg.get_last_analysis_run(prd_source_id))

    ingest_run_id = str(uuid.uuid4())
    _track_task(ingest_run_id, run_ingest(
        source_type="confluence",
        source=page_id,
        module=None,
        embed_client=app.state.embed_client,
        es_store=app.state.es_store,
        pg_store=pg,
        run_id=ingest_run_id,
        force=True,
    ))

    result = {
        "status": "accepted",
        "event": event,
        "page_id": page_id,
        "prd_source_id": prd_source_id,
        "ingest_run_id": ingest_run_id,
        "analysis_run_id": None,
        "analysis": "not_started",
    }

    if previous and WEBHOOK_ANALYZE:
        analysis_run_id = str(uuid.uuid4())
        _track_task(analysis_run_id, run_incremental_analysis(
            prd_source_id=prd_source_id,
            previous_run_id=previous["run_id"],
            module=None,
            embed_client=app.state.embed_client,
            es_store=app.state.es_store,
            pg_store=pg,
            run_id=analysis_run_id,
            provider=_env_default_provider(),
            model=_env_default_model(),
            reranker=getattr(app.state, "reranker", None),
        ))
        result["analysis_run_id"] = analysis_run_id
        result["analysis"] = "incremental"
    elif not previous and WEBHOOK_ANALYZE_NEW:
        analysis_run_id = str(uuid.uuid4())
        _track_task(analysis_run_id, run_analysis(
            prd_source_id=prd_source_id,
            module=None,
            embed_client=app.state.embed_client,
            es_store=app.state.es_store,
            pg_store=pg,
            run_id=analysis_run_id,
            provider=_env_default_provider(),
            model=_env_default_model(),
            reranker=getattr(app.state, "reranker", None),
        ))
        result["analysis_run_id"] = analysis_run_id
        result["analysis"] = "full"
    elif not previous:
        result["analysis"] = "skipped_never_analysed"
    else:
        result["analysis"] = "skipped_disabled"

    logger.info(
        "Confluence webhook: %s for %s — ingest=%s analysis=%s",
        event, prd_source_id, ingest_run_id, result["analysis"],
    )
    return result


@app.get("/decisions/overdue")
@limiter.limit("30/minute")
def decisions_overdue(
    request: Request,
    days: int = Query(default=None, ge=1, le=365,
                      description="SLA window; defaults to QA_REVIEW_SLA_DAYS"),
    limit: int = Query(default=50, ge=1, le=500),
):
    """
    Decisions still awaiting review past the SLA window.

    Oldest first. The count is the whole backlog; the list is capped by `limit`, so a
    large backlog does not turn a status check into a huge response.

    `by_action` matters more than the total: a hundred overdue KEEPs is a queue nobody
    has got to, while one overdue DEPRECATE is a test that may be about to disappear
    without anyone having agreed to it.
    """
    window = REVIEW_SLA_DAYS if days is None else days
    pg: PGStore = app.state.pg_store
    total = pg.count_overdue_decisions(days=window)
    rows = pg.get_overdue_decisions(days=window, limit=limit) if total else []

    by_action: dict[str, int] = {}
    for r in rows:
        by_action[r["action"]] = by_action.get(r["action"], 0) + 1

    return {
        "sla_days": window,
        "overdue_count": total,
        "returned": len(rows),
        "by_action": by_action,
        "oldest_age_days": round(rows[0]["age_days"], 1) if rows else None,
        "decisions": rows,
    }


@app.get("/analyze/coverage-map/{run_id}")
@limiter.limit("30/minute")
def analyze_coverage_map(request: Request, run_id: str):
    """
    Section-by-section coverage for an analysis run.

    A single coverage score cannot distinguish a section that was checked and found
    correct from one the agent never reached — both are just "not counted". This lists
    every testable section in the document with what was actually concluded about it:

      uncovered  — no decision recorded (the section list comes from the document, so
                   these appear here even though they are absent from the decisions table)
      unverified — only CREATE: a gap was found, nothing tests it yet
      shrinking  — only DEPRECATE: coverage is being removed
      questioned — only QUESTION: the agent could not decide
      covered    — at least one KEEP or UPDATE

    Meta sections (Background, Success Metrics, …) are excluded — they carry no
    requirements and would dilute the picture.
    """
    from agents.analysis_agent import (
        _is_meta_heading,
        _normalize_heading_for_coverage as _norm,
    )

    pg: PGStore = app.state.pg_store
    run = pg.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"No run with run_id={run_id}")

    rows = pg.get_coverage_map_data(run_id)
    # Decisions are keyed on agent-authored free text; headings come verbatim from
    # Elasticsearch. Both sides are normalised, the same trap that once made incremental
    # carry-forward silently carry nothing.
    by_section = {_norm(r.get("prd_section") or ""): r for r in rows}

    prd_source = run.get("prd_source")
    sections: list[dict] = []
    unmatched: list[str] = []

    if prd_source:
        es: ESStore = app.state.es_store
        from elasticsearch import helpers as es_helpers

        headings: list[str] = []
        seen: set[str] = set()
        for hit in es_helpers.scan(
            es._client,
            index="qa_prd_chunks",
            query={"query": {"term": {"source_id": prd_source}},
                   "_source": ["section_heading", "chunk_index"]},
            scroll="2m",
            size=500,
        ):
            heading = (hit["_source"].get("section_heading") or "").strip()
            if heading and heading not in seen:
                seen.add(heading)
                headings.append(heading)

        # Decisions whose section label matches no heading are surfaced separately:
        # the same mismatch also stops incremental carry-forward finding anything.
        sections, unmatched = merge_section_coverage(
            [(h, _norm(h)) for h in headings if not _is_meta_heading(h)],
            by_section,
        )

    summary: dict[str, int] = {risk: 0 for risk in GAP_RISKS}
    for s in sections:
        summary[s["gap_risk"]] += 1

    return {
        "run_id": run_id,
        "prd_source": prd_source,
        "status": run.get("status"),
        "testable_sections": len(sections),
        "summary": summary,
        "sections": sections,
        # Present but not counted above — these decisions exist and reference a section
        # name that is not in the document.
        "decisions_with_unmatched_section": unmatched,
        "note": None if prd_source else
                "This run has no prd_source recorded, so the document's section list "
                "could not be loaded and uncovered sections cannot be shown.",
    }


class RollbackRequest(BaseModel):
    # Defaults to a preview. Undoing a write-back is itself a write, and the caller
    # reaching for it is usually already reacting to a mistake.
    dry_run: bool = True


@app.post("/writeback/rollback/{decision_id}", status_code=200)
@limiter.limit("20/minute")
async def rollback_writeback(request: Request, decision_id: int, req: RollbackRequest):
    """
    Undo a DEPRECATE that has been written back, restoring the test's original labels.

    Only DEPRECATE is reversible, and only when a `pre_deprecation_snapshot` was taken
    at write-back time. Decisions written back before snapshotting existed cannot be
    rolled back here — the original label set is simply not recorded anywhere.

    Restores labels exactly as captured, which removes DEPRECATED. It does not undo the
    explanatory Jira comment: the comment is a true record that the deprecation happened,
    and the rollback adds its own alongside it.

    dry_run defaults to true — call with {"dry_run": false} to apply.
    """
    pg: PGStore = app.state.pg_store
    row = pg.get_decision_by_id(decision_id)
    if not row:
        raise HTTPException(status_code=404, detail=f"No decision with id={decision_id}")
    if row.get("action") != "deprecate":
        raise HTTPException(
            status_code=400,
            detail=f"Only DEPRECATE decisions can be rolled back; id={decision_id} is "
                   f"{row.get('action')!r}",
        )
    if not row.get("written_back"):
        raise HTTPException(
            status_code=400,
            detail="Decision has not been written back — there is nothing to undo. "
                   "Reject it in review instead.",
        )

    content = row.get("updated_content") or {}
    if isinstance(content, str):
        try:
            content = json.loads(content)
        except json.JSONDecodeError:
            content = {}
    snapshot = content.get("pre_deprecation_snapshot")
    if not snapshot or not isinstance(snapshot, dict):
        raise HTTPException(
            status_code=409,
            detail="No pre_deprecation_snapshot on this decision, so the original labels "
                   "are unknown. It was written back before snapshotting existed, or the "
                   "labels could not be read at the time.",
        )

    jira_key = row.get("jira_key") or snapshot.get("jira_key")
    if not jira_key:
        raise HTTPException(status_code=400, detail="Decision has no jira_key")

    labels = list(snapshot.get("labels") or [])
    if req.dry_run:
        from integrations import xray_client
        try:
            current = await xray_client.get_labels(jira_key)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Could not read {jira_key}: {exc}")
        return {
            "dry_run": True,
            "decision_id": decision_id,
            "jira_key": jira_key,
            "current_labels": current,
            "would_restore": labels,
            "captured_at": snapshot.get("captured_at"),
        }

    from integrations import xray_client
    try:
        await xray_client.set_labels(jira_key, labels)
        await xray_client.add_comment(
            jira_key,
            "[QA Intelligence Engine] Deprecation rolled back — labels restored to their "
            f"state at {snapshot.get('captured_at') or 'write-back time'}.",
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Rollback failed for {jira_key}: {exc}")

    pg.merge_decision_updated_content(decision_id, {
        "rolled_back_at": datetime.now(timezone.utc).isoformat(),
    })
    logger.info("Rolled back deprecation of %s (decision %s)", jira_key, decision_id)
    return {
        "dry_run": False,
        "status": "rolled_back",
        "decision_id": decision_id,
        "jira_key": jira_key,
        "restored_labels": labels,
    }


# ─── Incremental analysis ────────────────────────────────────────────────────

class IncrementalAnalysisRequest(BaseModel):
    prd_source_id: str
    previous_run_id: str            # run_id of the last analysis to diff against
    module: list[str] | None = None
    provider: str = Field(default_factory=_env_default_provider)
    model: str = Field(default_factory=_env_default_model)


@app.post("/analyze/incremental", status_code=202)
@limiter.limit("10/minute")
async def analyse_incremental(request: Request, req: IncrementalAnalysisRequest):
    """
    Incremental PRD analysis — only re-analyses changed sections.

    Compares the current PRD against the previous analysis run's state:
      - Changed/new sections → full LLM analysis
      - Unchanged sections → decisions carried forward from previous run
      - Removed sections → flagged as questions for review

    Cuts LLM cost by 60-90% for iterative PRD updates.
    """
    if req.provider not in _VALID_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"provider must be one of: {_VALID_PROVIDERS}")

    prev_row = app.state.pg_store.get_run(req.previous_run_id)
    if not prev_row:
        raise HTTPException(
            status_code=404,
            detail=f"No run found for previous_run_id={req.previous_run_id!r}",
        )
    if prev_row.get("prd_source") != req.prd_source_id:
        raise HTTPException(
            status_code=400,
            detail=(
                f"previous_run_id refers to prd_source={prev_row.get('prd_source')!r}, "
                f"expected {req.prd_source_id!r}"
            ),
        )

    module_n = normalize_module_list(req.module)
    _require_known_modules(module_n, app.state.es_store)
    if module_n:
        logger.info("analyze/incremental module filter (normalized): %s", module_n)

    ensured = await _ensure_ingested(
        req.prd_source_id,
        app.state.pg_store,
        app.state.embed_client,
        app.state.es_store,
        module=module_n,
    )

    run_id = str(uuid.uuid4())
    _track_task(
        run_id,
        run_incremental_analysis(
            prd_source_id=req.prd_source_id,
            previous_run_id=req.previous_run_id,
            module=module_n,
            embed_client=app.state.embed_client,
            es_store=app.state.es_store,
            pg_store=app.state.pg_store,
            run_id=run_id,
            provider=req.provider,
            model=req.model,
            reranker=getattr(app.state, "reranker", None),
        ),
    )

    out = {
        "run_id": run_id,
        "status": "started",
        "mode": "incremental",
        "previous_run_id": req.previous_run_id,
        "message": f"Incremental analysis running. Poll /analyze/status/{run_id} for progress.",
    }
    if ensured.get("ingest_run_id"):
        out["auto_ingest_run_id"] = ensured["ingest_run_id"]
    return out


# ─── LLM-as-judge evaluation ─────────────────────────────────────────────────

class EvaluateDecisionsRequest(BaseModel):
    run_id: str
    provider: str = Field(default_factory=_env_default_provider)
    model: str = Field(default_factory=_env_default_model)
    sample_size: int = Field(default=20, ge=1, le=100)


@app.post("/evaluate/decisions")
@limiter.limit("20/minute")
async def evaluate_decisions_endpoint(request: Request, req: EvaluateDecisionsRequest):
    """
    Evaluate decision quality using LLM-as-judge.

    Samples decisions from an analysis run and grades each one for:
      - Correctness (is the action appropriate?)
      - Reasoning (is the explanation specific and justified?)
      - Completeness (does it cover edge cases?)

    Returns per-decision scores and aggregate averages (1-5 scale).
    No write operations — safe to run multiple times.
    """
    from eval.llm_judge import evaluate_decisions

    loop = asyncio.get_running_loop()
    result = await evaluate_decisions(
        run_id=req.run_id,
        pg_store=app.state.pg_store,
        es_store=app.state.es_store,
        provider=req.provider,
        model=req.model,
        sample_size=req.sample_size,
    )
    return result


# ─── Observability ────────────────────────────────────────────────────────────

@app.get("/integrations/mcp/tools")
@limiter.limit("30/minute")
async def mcp_tool_config(request: Request):
    """
    Diagnose the external MCP tool configuration.

    Returns the logical operation → tool name mapping this service will call, the tools the
    server actually advertises, and any configured names that do not exist there. Use this
    after changing XRAY_MCP_TOOL_MAP, and before blaming a sync failure on the pipeline.

    status:
      ok            — every configured tool exists on the server
      misconfigured — see `missing`; fix via XRAY_MCP_TOOL_MAP / XRAY_MCP_TOOL_<OP>
      unreachable   — MCP server down or wrong XRAY_MCP_URL; nothing was verified
    """
    from integrations import xray_client

    report = await xray_client.verify_tools()
    return JSONResponse(
        status_code=200 if report["status"] == "ok" else 503,
        content=report,
    )


@app.get("/metrics")
def get_metrics():
    """
    Return all collected metrics in JSON format.
    For Prometheus text format, use /metrics/prometheus.
    """
    try:
        from observability.metrics import metrics
        return metrics.snapshot()
    except ImportError:
        return {"error": "Metrics module not available"}


@app.get("/metrics/prometheus")
def get_metrics_prometheus():
    """Return metrics in Prometheus text exposition format for scraping."""
    try:
        from observability.metrics import metrics
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(
            content=metrics.prometheus_text(),
            media_type="text/plain",
        )
    except ImportError:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(content="# metrics module not available\n")
