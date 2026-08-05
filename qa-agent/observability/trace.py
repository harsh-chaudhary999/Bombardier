"""
Payload trace — full request/response bodies for debugging, off by default.

The regular logs summarise ("get_test ok dict(4 keys) in 342ms") because a run touching a
few hundred tests would otherwise bury every other line. When you need the actual data —
what was asked for, what came back, what got chunked, what was embedded — turn this on:

    QA_TRACE=1

Each event is appended as one JSON object per line to QA_TRACE_FILE
(default: eval/payload-trace.jsonl, alongside the phase ledger). stdout stays readable;
the file holds everything.

    QA_TRACE=1                     enable
    QA_TRACE_FILE=/path/x.jsonl    override destination
    QA_TRACE_MAX_CHARS=20000       per-field cap; 0 = unlimited (default 20000)
    QA_TRACE_STDOUT=1              ALSO pretty-print each payload to the log

Reading it back:
    jq -c 'select(.kind=="mcp") | {op, ms, req, res_summary}' eval/payload-trace.jsonl
    jq -r 'select(.kind=="chunk") | [.source_id, .heading, .tokens] | @tsv' ...

Caveat: traces contain your actual Jira/Xray/Confluence content. Treat the file as
internal — it is not redacted, and it is not intended to leave the host.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_warned = False


def enabled() -> bool:
    return os.environ.get("QA_TRACE", "0") == "1"


def _max_chars() -> int:
    try:
        return int(os.environ.get("QA_TRACE_MAX_CHARS", "20000"))
    except ValueError:
        return 20000


def _path() -> Path:
    override = os.environ.get("QA_TRACE_FILE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent / "eval" / "payload-trace.jsonl"


def _clip(value: Any) -> Any:
    """Serialise a payload, truncating oversized strings but keeping structure visible."""
    cap = _max_chars()
    try:
        text = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        text = str(value)
    if cap and len(text) > cap:
        return {"_truncated": True, "_original_chars": len(text), "_head": text[:cap]}
    try:
        return json.loads(text)
    except Exception:
        return text


def event(kind: str, **fields: Any) -> None:
    """
    Append one trace event. Never raises — tracing must not break the pipeline.

    kind is a short tag used for filtering: mcp | http | chunk | embed | index | ingest.
    """
    if not enabled():
        return
    global _warned
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        **{k: _clip(v) for k, v in fields.items()},
    }
    line = json.dumps(record, ensure_ascii=False, default=str)
    try:
        p = _path()
        p.parent.mkdir(parents=True, exist_ok=True)
        with _lock:
            with open(p, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    except OSError as e:
        if not _warned:
            logger.warning("payload trace disabled — cannot write %s: %s", _path(), e)
            _warned = True
    if os.environ.get("QA_TRACE_STDOUT", "0") == "1":
        logger.info("trace[%s] %s", kind, line)
