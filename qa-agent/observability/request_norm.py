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


def unknown_module_error(
    requested: list[str] | None,
    available: list[str],
) -> dict | None:
    """
    Decide whether a module filter should be rejected. Returns the error payload, or
    None to proceed.

    An unknown module is not an empty result — it is a typo, a renamed module, or a
    sync that never ran. Left unchecked the analysis completes with zero decisions,
    which reads as "this PRD is fully covered" rather than "nothing was searched".

    Two deliberate non-rejections:
      * A partial match proceeds. One known module is enough to do useful work; the
        unknown ones are the caller's problem to notice, not a reason to refuse.
      * An empty `available` proceeds. That means the aggregation failed or nothing
        is indexed yet, and validation must never be the reason a run cannot start.

    Kept here rather than in main.py so it is testable without importing the web stack.
    """
    if not requested or not available:
        return None
    if any(m in available for m in requested):
        return None
    return {
        "error": "module_not_found",
        "requested": requested,
        "available_modules": available,
        "hint": "Module names are case-sensitive. Run POST /sync/tests if the module "
                "exists in Xray but has not been indexed here.",
    }


#: Ordered worst-first so a caller can sort or threshold on the index.
GAP_RISKS = ("uncovered", "unverified", "shrinking", "questioned", "covered")


def section_gap_risk(counts: dict | None) -> str:
    """
    Classify one PRD section by what the agent concluded about it.

    `counts` is None when the section exists in the document but has no decision at all.
    That is the most serious state and the one a single coverage score hides: a section
    nobody looked at is indistinguishable, in a percentage, from one that was checked and
    found fine.

      uncovered  — no decision was recorded. Nobody looked, or the run stopped first.
      unverified — only CREATE. A gap was identified, but nothing tests it yet.
      shrinking  — only DEPRECATE. Coverage is being removed and nothing replaces it.
      questioned — only QUESTION. The agent could not decide; a human must.
      covered    — at least one KEEP or UPDATE: a real test is attached to this section.

    Named rather than severity-graded on purpose. "HIGH/MEDIUM/LOW" invites a reader to
    average them; these say what is actually true of the section, which is what a
    reviewer needs in order to act.
    """
    if not counts:
        return "uncovered"
    keep = counts.get("keep_count") or 0
    update = counts.get("update_count") or 0
    if keep or update:
        return "covered"
    if counts.get("create_count"):
        return "unverified"
    if counts.get("deprecate_count"):
        return "shrinking"
    if counts.get("question_count"):
        return "questioned"
    return "uncovered"


_COVERAGE_COUNT_FIELDS = (
    "decisions", "keep_count", "update_count", "deprecate_count", "create_count",
    "question_count", "high_confidence", "low_confidence", "unrated",
    "approved_count", "rejected_count", "unreviewed_count",
)


def merge_section_coverage(
    headings: list[tuple[str, str]],
    by_section: dict[str, dict],
) -> tuple[list[dict], list[str]]:
    """
    Join a document's sections against the decisions recorded for them.

    `headings` is [(display_heading, normalized_key)], already filtered to testable
    sections. `by_section` maps the same normalized key to that section's counts.
    Returns (rows, unmatched) where `unmatched` names decision labels that correspond
    to no heading in the document.

    Both sides must arrive normalised. Decision labels are agent-authored free text
    ("3.2 Payment Capture:") while headings come verbatim from Elasticsearch
    ("Payment Capture") — comparing them raw matches almost nothing, which is the bug
    that once made incremental carry-forward silently carry zero decisions.

    Kept out of main.py so the join is testable without the web stack.
    """
    rows: list[dict] = []
    matched: set[str] = set()
    for display, key in headings:
        counts = by_section.get(key)
        if counts:
            matched.add(key)
        rows.append({
            "section": display,
            "gap_risk": section_gap_risk(counts),
            **{f: (counts or {}).get(f, 0) or 0 for f in _COVERAGE_COUNT_FIELDS},
        })

    unmatched = sorted(
        (row.get("prd_section") or "(none)")
        for key, row in by_section.items()
        if key and key not in matched
    )
    return rows, unmatched


def unknown_modules(requested: list[str] | None, available: list[str]) -> list[str]:
    """Requested modules absent from the index — worth logging, not worth refusing."""
    if not requested or not available:
        return []
    return [m for m in requested if m not in available]
