"""Deterministic run IDs for idempotent analysis triggers (same inputs within one UTC minute)."""
from __future__ import annotations

import hashlib
import os
import uuid
from datetime import datetime, timezone


def deterministic_analysis_run_id(
    prd_source_id: str,
    module: list[str] | None,
    *,
    minute_bucket: datetime | None = None,
) -> str:
    """
    Return a UUID string derived from PRD id, sorted module filter, UTC minute, and optional salt.

    Re-posting /analyze/prd with the same prd_source_id, module list, and clock minute yields the
    same run_id (Postgres start_run ON CONFLICT ignores duplicate register).
    """
    ts = minute_bucket or datetime.now(timezone.utc)
    ts = ts.replace(second=0, microsecond=0)
    mod = ",".join(sorted(module or []))
    salt = os.environ.get("QA_ANALYSIS_RUN_ID_SALT", "")
    payload = f"{prd_source_id}|{mod}|{ts.isoformat()}|{salt}"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()[:16]
    return str(uuid.UUID(bytes=digest))
