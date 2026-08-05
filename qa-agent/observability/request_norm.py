"""Normalize request fields for stable ES filters and run_metadata."""
from __future__ import annotations


def normalize_module_list(modules: list[str] | None) -> list[str] | None:
    """
    Dedupe by case-insensitive key while preserving the caller's original casing
    (first occurrence wins). Avoids str.title() bugs on acronyms (API → Api).

    Sort order is alphabetical by the preserved display string (case-sensitive sort).
    """
    if not modules:
        return None
    seen: dict[str, str] = {}
    for m in modules:
        s = (m or "").strip()
        if not s:
            continue
        key = s.casefold()
        if key not in seen:
            seen[key] = s
    return sorted(seen.values()) if seen else None
