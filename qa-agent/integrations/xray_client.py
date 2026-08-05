"""
HTTP MCP client for the external Jira / Xray integration (Streamable HTTP transport).

The MCP server is NOT part of this project. It is assumed to be deployed, running and
functional; this module only needs configuration to address its tools.

Configuration
-------------
XRAY_MCP_URL         Streamable HTTP endpoint of the MCP server.

XRAY_MCP_TOOL_MAP    JSON object mapping this module's LOGICAL operations to the tool
                     names (and, if needed, argument names) your server actually exposes.
                     Merged over the defaults below — you only declare what differs.

                     Two value shapes are accepted per operation:

                       "get_test": "my_xray_fetch_test"

                       "get_tests_from_folder": {
                         "name": "my_list_tests",
                         "args": {"projectKey": "project_key", "folderPath": "folder"}
                       }

                     The "args" map renames outgoing argument keys, which is what you need
                     when a server uses snake_case where the defaults use camelCase.

XRAY_MCP_TOOL_<OP>   Per-operation name override, e.g. XRAY_MCP_TOOL_GET_TEST=fetch_test.
                     Takes precedence over XRAY_MCP_TOOL_MAP. Convenient for one-offs.

XRAY_MCP_UNWRAP_DATA Default 1. Servers commonly wrap results as
                     {"success": true, "data": {...}}; set 0 if your server returns a
                     literal top-level "data" field that must not be unwrapped.

Logical operations and their default tool names are in _TOOL_SPEC_DEFAULTS below. Run
verify_tools() (called at startup and exposed via GET /integrations/mcp/tools) to check
the configured names against what the server actually advertises.

NOT configurable: response *shapes*. Callers here read fields like result["results"],
raw["total"] and issue["fields"]. A server with a different response schema needs code
changes in this module, not configuration.
"""
import asyncio
import json
import logging
import os
import re
import time
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from observability import trace

logger = logging.getLogger(__name__)

_MCP_TOOL_TIMEOUT_SEC = float(os.environ.get("MCP_TOOL_TIMEOUT_SEC", "60"))

# Base URL of the external MCP server (Streamable HTTP path, often .../mcp). Override with XRAY_MCP_URL.
# Default port is a local-dev convention only; set the env var to match your deployment.
XRAY_MCP_URL = os.environ.get("XRAY_MCP_URL", "http://127.0.0.1:3100/mcp")

# Strict pattern for Jira keys — prevents JQL injection
_JIRA_KEY_RE = re.compile(r'^[A-Z][A-Z0-9_]*-\d+$')


# ─── Tool configuration ───────────────────────────────────────────────────────
#
# Logical operation -> {name, args}. Defaults match the common Jira/Xray MCP naming;
# override per deployment via XRAY_MCP_TOOL_MAP / XRAY_MCP_TOOL_<OP>. Nothing outside
# this dict should reference a raw MCP tool name.
_TOOL_SPEC_DEFAULTS: dict[str, dict[str, Any]] = {
    "get_folders":           {"name": "xray_get_folders"},
    "get_tests_from_folder": {"name": "xray_get_tests_from_folder"},
    "get_test":              {"name": "xray_get_test"},
    "update_test":           {"name": "xray_update_test"},
    "bulk_create_tests":     {"name": "xray_bulk_create_tests"},
    "search_issues":         {"name": "jira_search_issues"},
    "update_issue":          {"name": "jira_update_issue"},
    "add_comment":           {"name": "jira_add_comment"},
    "add_remote_link":       {"name": "jira_add_remote_link"},
}

_tool_specs_cache: dict[str, dict[str, Any]] | None = None


def _load_tool_specs() -> dict[str, dict[str, Any]]:
    """Resolve logical operations to concrete tool names + argument aliases."""
    specs: dict[str, dict[str, Any]] = {
        op: {"name": spec["name"], "args": dict(spec.get("args") or {})}
        for op, spec in _TOOL_SPEC_DEFAULTS.items()
    }

    raw = os.environ.get("XRAY_MCP_TOOL_MAP", "").strip()
    if raw:
        try:
            overrides = json.loads(raw)
            if not isinstance(overrides, dict):
                raise ValueError("XRAY_MCP_TOOL_MAP must be a JSON object")
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(
                "Ignoring XRAY_MCP_TOOL_MAP — could not parse (%s). Using default tool names.", e
            )
            overrides = {}
        for op, val in overrides.items():
            if op not in specs:
                logger.warning(
                    "XRAY_MCP_TOOL_MAP names unknown operation %r — known operations: %s",
                    op, ", ".join(sorted(specs)),
                )
                continue
            if isinstance(val, str):
                specs[op]["name"] = val
            elif isinstance(val, dict):
                if val.get("name"):
                    specs[op]["name"] = val["name"]
                if isinstance(val.get("args"), dict):
                    specs[op]["args"].update(val["args"])
            else:
                logger.warning(
                    "XRAY_MCP_TOOL_MAP[%r] must be a string or an object, got %s — ignored",
                    op, type(val).__name__,
                )

    # Per-operation env override wins over the JSON map.
    for op in specs:
        env_name = os.environ.get(f"XRAY_MCP_TOOL_{op.upper()}", "").strip()
        if env_name:
            specs[op]["name"] = env_name

    return specs


def _tool_specs() -> dict[str, dict[str, Any]]:
    global _tool_specs_cache
    if _tool_specs_cache is None:
        _tool_specs_cache = _load_tool_specs()
        customised = {
            op: s["name"]
            for op, s in _tool_specs_cache.items()
            if s["name"] != _TOOL_SPEC_DEFAULTS[op]["name"] or s["args"]
        }
        if customised:
            logger.info("MCP tool configuration overrides active: %s", customised)
    return _tool_specs_cache


def reset_tool_config() -> None:
    """Drop the cached tool configuration so env changes take effect (tests, reload)."""
    global _tool_specs_cache
    _tool_specs_cache = None


def _resolve(operation: str, arguments: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Map a logical operation + arguments onto the configured tool name + argument keys."""
    spec = _tool_specs().get(operation)
    if spec is None:
        raise ValueError(
            f"Unknown MCP operation {operation!r}. "
            f"Known operations: {', '.join(sorted(_TOOL_SPEC_DEFAULTS))}"
        )
    aliases = spec["args"]
    if aliases:
        arguments = {aliases.get(k, k): v for k, v in arguments.items()}
    return spec["name"], arguments


def configured_tools() -> dict[str, str]:
    """Logical operation -> configured tool name. For diagnostics endpoints."""
    return {op: spec["name"] for op, spec in _tool_specs().items()}

# ─── Session handling ─────────────────────────────────────────────────────────
#
# There is deliberately NO cross-task session pool here.
#
# The previous implementation held a `streamablehttp_client(...)` context manager in a
# module global, entering it with `await cm.__aenter__()` in whichever task made the first
# call and exiting it with `await cm.__aexit__(...)` from whatever task later triggered a
# refresh (`_POOL_MAX_AGE`, 300 s) or a ping failure.
#
# That is not safe. streamablehttp_client is anyio-based, and anyio cancel scopes are
# strictly task-bound: a scope entered in task A cannot be exited from task B. When the
# refresh fired mid-sync, the cancellation escaped the intended scope and propagated to
# whatever scope was on the stack — the ASGI **lifespan** task — killing the whole app with
#
#   asyncio.exceptions.CancelledError: Cancelled via cancel scope ... by
#   <Task pending coro=<_run_sync_with_optional_retry() ...>>
#
# reproducibly ~300 s into a sync. `async with` exists precisely to prevent this.
#
# Each call now opens and closes its own session inside a single `async with`, so entry and
# exit are always in the same task. The cost is one connect + `initialize()` per call —
# tens of milliseconds against MCP calls that already take 300 ms–1 s of real Jira/Xray
# work. Correctness is worth more than that margin.
#
# If session reuse is ever needed for throughput, it must be done by dedicating ONE task to
# own the session for its whole lifetime (e.g. a worker task fed by a queue), not by
# sharing a context manager across tasks.

def _validate_jira_key(key: str) -> str:
    """Validate that a string looks like a valid Jira key. Raises ValueError if not."""
    if not _JIRA_KEY_RE.match(key):
        raise ValueError(f"Invalid Jira key format: {key!r}")
    return key


def _parse_result(result) -> Any:
    """Parse MCP tool result into usable data, unwrapping a common {data: ...} envelope if present."""
    if not result.content:
        return None

    first = result.content[0]
    if not hasattr(first, 'text'):
        logger.warning(f"MCP result content[0] has no text attribute: {type(first)}")
        return None

    text = first.text
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, AttributeError):
        return text

    # Unwrap server envelope: {"success": true, "data": {...}}. Disable with
    # XRAY_MCP_UNWRAP_DATA=0 if your server returns a literal top-level "data" payload.
    if (
        isinstance(parsed, dict)
        and "data" in parsed
        and os.environ.get("XRAY_MCP_UNWRAP_DATA", "1") != "0"
    ):
        return parsed["data"]
    return parsed


async def list_server_tools() -> list[str]:
    """Tool names the MCP server actually advertises. Raises if the server is unreachable."""
    async with streamablehttp_client(XRAY_MCP_URL) as (read, write, _):
        async with ClientSession(read, write) as session:
            await session.initialize()
            resp = await asyncio.wait_for(session.list_tools(), timeout=_MCP_TOOL_TIMEOUT_SEC)
            return sorted(t.name for t in getattr(resp, "tools", []) or [])


async def verify_tools() -> dict[str, Any]:
    """
    Check the configured tool names against what the server advertises.

    Turns a mysterious mid-sync failure ("Tool error: unknown tool") into an actionable
    startup message naming exactly which operations are misconfigured and what the server
    does expose. Never raises — an unreachable server is reported, not fatal, because the
    MCP is an external service that may come up after this process.
    """
    configured = configured_tools()
    try:
        available = await list_server_tools()
    except Exception as e:
        return {
            "status": "unreachable",
            "url": XRAY_MCP_URL,
            "error": str(e)[:300],
            "configured": configured,
            "available": None,
            "missing": None,
        }

    missing = {op: name for op, name in configured.items() if name not in available}
    return {
        "status": "ok" if not missing else "misconfigured",
        "url": XRAY_MCP_URL,
        "configured": configured,
        "available": available,
        "missing": missing,
    }


def _fmt_args(arguments: dict[str, Any], max_len: int = 160) -> str:
    """Compact one-line rendering of tool arguments for logs — long values elided."""
    parts = []
    for k, v in (arguments or {}).items():
        if isinstance(v, str):
            shown = v if len(v) <= 60 else v[:57] + "..."
        elif isinstance(v, (list, tuple)):
            shown = f"[{len(v)} items]"
        elif isinstance(v, dict):
            shown = f"{{{len(v)} keys}}"
        else:
            shown = str(v)
        parts.append(f"{k}={shown}")
    s = " ".join(parts)
    return s if len(s) <= max_len else s[: max_len - 3] + "..."


def _describe_result(payload: Any) -> str:
    """Summarise a tool result so logs show what came back, not just that it succeeded."""
    if payload is None:
        return "empty"
    if isinstance(payload, dict):
        if "total" in payload and "results" in payload:
            got = payload.get("results")
            return f"total={payload['total']} results={len(got) if isinstance(got, list) else '?'}"
        if "issues" in payload and isinstance(payload["issues"], list):
            return f"issues={len(payload['issues'])}"
        if "folders" in payload and isinstance(payload["folders"], list):
            return f"folders={len(payload['folders'])}"
        keys = list(payload.keys())
        return f"dict({len(keys)} keys: {','.join(keys[:5])}{'...' if len(keys) > 5 else ''})"
    if isinstance(payload, list):
        return f"list({len(payload)})"
    text = str(payload)
    return f"{type(payload).__name__}({len(text)}ch)"


def _is_retryable_mcp_error(exc: BaseException) -> bool:
    if isinstance(exc, asyncio.TimeoutError):
        return True
    msg = str(exc).lower()
    return any(
        s in msg
        for s in (
            "429",
            "502",
            "503",
            "504",
            "timeout",
            "temporarily",
            "rate",
            "connection reset",
            "connection aborted",
            "broken pipe",
        )
    )


async def _call(operation: str, arguments: dict[str, Any]) -> Any:
    """
    Call an MCP tool by LOGICAL operation name, preferring the pooled session.

    The operation is resolved to the configured tool name and argument keys, so call sites
    never reference a server-specific tool name. See the module docstring for the config.

    Strategy:
      1. Try pooled session (fast — no connect/init overhead)
      2. If pooled call fails, invalidate pool and retry with one-shot session
      3. Retries with backoff on transient HTTP / rate-limit style failures
    """
    tool_name, arguments = _resolve(operation, arguments)
    t0 = time.monotonic()
    logger.info("mcp → %s [%s] %s", operation, tool_name, _fmt_args(arguments))

    delays = [0.0, 1.0, 2.0, 4.0]

    for attempt, delay in enumerate(delays):
        if delay:
            await asyncio.sleep(delay)
        try:

            # One session per call, entered and exited in this same task — see the note
            # above on anyio cancel scopes being task-bound.
            async with streamablehttp_client(XRAY_MCP_URL) as (read, write, _):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await asyncio.wait_for(
                        session.call_tool(tool_name, arguments),
                        timeout=_MCP_TOOL_TIMEOUT_SEC,
                    )
                    parsed = _parse_result(result)

            _ms = (time.monotonic() - t0) * 1000
            logger.info("mcp ← %s ok %s in %dms", operation, _describe_result(parsed), _ms)
            trace.event("mcp", op=operation, tool=tool_name,
                        ms=round(_ms), attempt=attempt + 1,
                        req=arguments, res=parsed,
                        res_summary=_describe_result(parsed))
            return parsed

        except Exception as e:
            if attempt < len(delays) - 1 and _is_retryable_mcp_error(e):
                logger.warning(
                    "mcp ‼ %s [%s] transient failure %s/%s after %dms: %s — backing off %ss",
                    operation, tool_name, attempt + 1, len(delays),
                    (time.monotonic() - t0) * 1000, e,
                    delays[attempt + 1],
                )
                continue
            logger.error(
                "mcp ✗ %s [%s] FAILED after %dms and %s attempt(s): %s | args: %s",
                operation, tool_name, (time.monotonic() - t0) * 1000, attempt + 1, e,
                _fmt_args(arguments),
            )
            trace.event("mcp", op=operation, tool=tool_name, transport="failed",
                        ms=round((time.monotonic() - t0) * 1000), attempt=attempt + 1,
                        req=arguments, error=str(e), error_type=type(e).__name__)
            raise


# ─── Folder / test listing ────────────────────────────────────────────────

async def get_folders(project_key: str, path: str = "/") -> dict:
    """
    Get the folder tree for a project.
    Returns: {name, path, testCount, folders: [...]}
    """
    return await _call("get_folders", {
        "projectKey": project_key,
        "path": path,
    })


async def get_all_tests(
    project_key: str,
    folder_path: str = "",
    start: int = 0,
    limit: int = 100,
) -> dict:
    """
    Fetch one page of tests (max 100) from a project folder.

    Caller must paginate: loop, incrementing start by len(results) each time,
    until len(fetched) >= total. See sync/test_sync.py for the loop.

    Returns: {"total": N, "results": [XrayTest, ...]}
    """
    return await _call("get_tests_from_folder", {
        "projectKey": project_key,
        "folderPath": folder_path,
        "includeDescendants": True,
        "limit": limit,
        "start": start,
    })


# ─── Single test detail ───────────────────────────────────────────────────

async def get_test(test_key: str) -> dict:
    """
    Get a single test with full details: summary, testType, steps, preconditions.
    Used by Triage Agent to inspect tests flagged by Drift Agent.
    """
    return await _call("get_test", {"testKey": test_key})


def _adf_to_text(node: Any, _depth: int = 0) -> str:
    """
    Convert Atlassian Document Format (ADF) node to plain text.
    Jira REST API v3 returns description as ADF, not a plain string.
    """
    if _depth > 50:
        return ""  # guard against deeply nested / cyclic ADF
    if not node or not isinstance(node, dict):
        return ""
    if node.get("type") == "text":
        return node.get("text", "")
    parts = []
    for child in node.get("content", []):
        parts.append(_adf_to_text(child, _depth + 1))
    return " ".join(p for p in parts if p)


async def get_descriptions_bulk(test_keys: list[str]) -> dict[str, dict]:
    """
    Fetch descriptions and labels for test keys via Jira search.
    Handles batching automatically for lists > 100 keys.
    Returns: {jira_key: {"description": str, "labels": list[str]}}
    """
    if not test_keys:
        return {}

    # Validate all keys to prevent JQL injection
    validated_keys = []
    for key in test_keys:
        try:
            validated_keys.append(_validate_jira_key(key))
        except ValueError:
            logger.warning(f"Skipping invalid Jira key: {key!r}")

    if not validated_keys:
        return {}

    desc_map: dict[str, dict] = {}
    BATCH_SIZE = 100

    for i in range(0, len(validated_keys), BATCH_SIZE):
        batch = validated_keys[i:i + BATCH_SIZE]
        jql = f"issue in ({', '.join(batch)})"
        result = await _call("search_issues", {
            "jql": jql,
            "maxResults": BATCH_SIZE,
            "fields": ["summary", "description", "labels"],
        })
        if not isinstance(result, dict):
            continue
        for issue in result.get("issues", []):
            key = issue.get("key")
            fields = issue.get("fields") or {}
            raw_desc = fields.get("description")
            if raw_desc is None:
                desc = ""
            elif isinstance(raw_desc, dict):
                desc = _adf_to_text(raw_desc)
            else:
                desc = str(raw_desc)
            labels = [str(l) for l in (fields.get("labels") or [])]
            if key:
                desc_map[key] = {"description": desc, "labels": labels}

    return desc_map


# ─── Write-back (Phase 5) ─────────────────────────────────────────────────

async def update_test(test_key: str, summary: str | None = None, steps: list | None = None) -> Any:
    """Update a test's summary and/or steps."""
    args: dict[str, Any] = {"testKey": _validate_jira_key(test_key)}
    if summary is not None:
        args["summary"] = summary
    if steps is not None:
        args["steps"] = steps
    return await _call("update_test", args)


async def bulk_create_tests(project_key: str, tests: list[dict]) -> Any:
    """
    Create up to 50 tests at once.
    Each test dict: {summary, testType, description?, steps?}
    """
    return await _call("bulk_create_tests", {
        "projectKey": project_key,
        "tests": tests,
    })


async def add_remote_link(issue_key: str, url: str, title: str = "PRD Source") -> None:
    """
    Add a Jira remote link (web link) from an issue back to a source URL.
    Used to link newly created test cases back to the PRD that drove their creation.
    """
    await _call("add_remote_link", {
        "issueKey": _validate_jira_key(issue_key),
        "url": url,
        "title": title,
    })


async def deprecate_test(test_key: str, reason: str) -> None:
    """
    Mark a test as deprecated: appends DEPRECATED label + a comment explaining why.
    Preserves existing labels.
    """
    validated_key = _validate_jira_key(test_key)

    # First fetch existing labels so we can append rather than overwrite
    result = await _call("search_issues", {
        "jql": f"issue = {validated_key}",
        "maxResults": 1,
        "fields": ["labels"],
    })
    existing_labels = []
    if isinstance(result, dict):
        issues = result.get("issues", [])
        if issues:
            existing_labels = issues[0].get("fields", {}).get("labels", [])

    # Append DEPRECATED if not already present
    if "DEPRECATED" not in existing_labels:
        existing_labels.append("DEPRECATED")

    await _call("update_issue", {
        "issueKey": validated_key,
        "fields": {"labels": existing_labels},
    })
    try:
        await _call("add_comment", {
            "issueKey": validated_key,
            "comment": (
                f"[QA Intelligence Engine] Marked as DEPRECATED.\n\nReason: {reason}"
            ),
        })
    except Exception as comment_exc:
        logger.error(
            "deprecate_test: issue %s has DEPRECATED label but comment failed — "
            "Jira is in an inconsistent state; add comment manually. Error: %s",
            validated_key,
            comment_exc,
        )
        raise RuntimeError(
            f"Partial deprecate for {validated_key}: DEPRECATED label applied but "
            f"Jira comment failed: {comment_exc}"
        ) from comment_exc
