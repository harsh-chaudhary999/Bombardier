"""
Append-only phase ledger (JSONL): each completed pipeline step gets an ISO timestamp,
phase name, run_id, summary fields, and SHA-256 of canonical summary JSON for tamper-evident audit.
"""
from __future__ import annotations

import asyncio
import functools
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    import fcntl
except ImportError:
    fcntl = None  # Windows — no advisory locking; single-line writes still atomic enough for many setups

from observability.canonical_json import dumps_canonical, fingerprint_sha256

logger = logging.getLogger(__name__)


def _ledger_path() -> Path:
    base = os.environ.get("QA_PHASE_LEDGER_PATH")
    if base:
        return Path(base)
    # Default: qa-agent/eval/phase-ledger.jsonl relative to this file's package root
    here = Path(__file__).resolve().parent.parent
    return here / "eval" / "phase-ledger.jsonl"


def verify_ledger_writable() -> None:
    """
    Fail fast at startup if the ledger path cannot be opened for append.
    Avoids silently losing audit entries when the volume is read-only or misconfigured.
    """
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.flush()
            os.fsync(f.fileno())
    except OSError as e:
        raise RuntimeError(
            f"QA phase ledger is not writable ({path}): {e}. "
            "Set QA_PHASE_LEDGER_PATH to a writable directory or fix volume mounts."
        ) from e


def append_entry(
    phase: str,
    run_id: str,
    summary: dict,
) -> dict:
    """
    Append one ledger line. summary is stored verbatim plus summary_sha256 for attestation.

    Returns the written record (including timestamp and fingerprint).
    """
    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    fp = fingerprint_sha256(summary)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "run_id": run_id,
        "summary_sha256": fp,
        "summary": summary,
    }
    line = dumps_canonical(record)
    try:
        with open(path, "a", encoding="utf-8") as f:
            if fcntl:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.write(line + "\n")
                f.flush()
            finally:
                if fcntl:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except OSError as e:
        logger.warning("phase_ledger append failed (%s): %s", path, e)
    return record


async def append_entry_async(phase: str, run_id: str, summary: dict) -> dict:
    """Schedule ledger append on the default executor so the asyncio event loop is not blocked."""
    loop = asyncio.get_running_loop()
    fn = functools.partial(append_entry, phase, run_id, summary)
    return await loop.run_in_executor(None, fn)
