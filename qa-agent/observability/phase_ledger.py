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


# Durable backend, registered once at startup. Module-level rather than threaded through
# every caller's signature: the ledger is cross-cutting, and passing a store down through
# analysis, incremental and sync would put a storage concern in three unrelated APIs.
_store = None
# Write the JSONL file as well as the database. Off by default once a store is
# registered — two copies of an append-only audit trail can disagree, and then neither
# is the record.
_FILE_ALWAYS = os.environ.get("QA_PHASE_LEDGER_FILE_ALWAYS", "0").strip() not in ("", "0", "false", "False")


def set_store(store) -> None:
    """
    Register the durable ledger backend (a PGStore). Call once at startup.

    Until this is called the ledger writes to the JSONL file, which is correct for a
    single process. It is NOT correct across replicas: fcntl locking coordinates writers
    on one machine, so each pod would keep its own partial file and no copy would be the
    record.
    """
    global _store
    _store = store


def append_entry(
    phase: str,
    run_id: str,
    summary: dict,
) -> dict:
    """
    Append one ledger line. summary is stored verbatim plus summary_sha256 for attestation.

    Writes to Postgres when a store is registered, falling back to the JSONL file if that
    insert fails — an audit entry that cannot be stored durably is still better recorded
    locally than dropped.

    Returns the written record (including timestamp and fingerprint).
    """
    fp = fingerprint_sha256(summary)
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "phase": phase,
        "run_id": run_id,
        "summary_sha256": fp,
        "summary": summary,
    }

    stored = False
    if _store is not None:
        stored = bool(_store.append_ledger_entry(
            phase=phase, run_id=run_id, summary=summary, summary_sha256=fp))
        record["backend"] = "postgres" if stored else "file"
        if stored and not _FILE_ALWAYS:
            return record
    else:
        record["backend"] = "file"

    path = _ledger_path()
    path.parent.mkdir(parents=True, exist_ok=True)
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
