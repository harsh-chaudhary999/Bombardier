"""Deterministic JSON serialisation for fingerprints, ledgers, and diffs."""
from __future__ import annotations

import hashlib
import json
from datetime import date, datetime
from decimal import Decimal
from typing import Any
from uuid import UUID


def _default(o: Any) -> str:
    if isinstance(o, (datetime, date)):
        return o.isoformat()
    if isinstance(o, UUID):
        return str(o)
    if isinstance(o, Decimal):
        return format(o, "f")
    if isinstance(o, bytes):
        return o.hex()
    raise TypeError(f"Object of type {type(o)!r} is not JSON serializable")


def dumps_canonical(obj: Any) -> str:
    """Stable key order at all levels; compact separators."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_default)


def fingerprint_sha256(obj: Any) -> str:
    """SHA-256 hex digest of the canonical JSON representation."""
    return hashlib.sha256(dumps_canonical(obj).encode()).hexdigest()


def normalize_json_obj(obj: Any) -> Any:
    """Round-trip through canonical JSON so dict key order and nesting are stable."""
    return json.loads(dumps_canonical(obj))
