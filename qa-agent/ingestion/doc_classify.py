"""
Document-type classification from titles.

Measured problem: in a whole-space ingest, requirements are a small minority. In one measured corpus,
only 148 of 3,400 indexed documents are PRDs (4%) while 595 are tech docs and 2,613 are
unclassified — meeting notes, analytics pages, event sheets. Every question therefore
searches ~3,250 non-requirement documents to find 148 requirement documents.

That matters differently per use case:

  * Coverage analysis must NOT see test plans. Finding "coverage" in a document that IS a
    test plan is circular — it proves nothing about the test suite.
  * A question about intent should prefer PRDs, but must still be able to fall back: some
    concepts (PROVISIONAL) are defined only in tech docs and appear in no PRD at all.

Two entry points:

  classify(title)        -> doc_type, for storing at ingest time
  title_filter(types)    -> an Elasticsearch filter derived from doc_title, so the existing
                            corpus can be scoped WITHOUT re-indexing

The title-derived filter is a bridge, not the destination: it is only as good as the naming
convention. Documents whose titles carry no marker land in `other` and are matched by
neither an allowlist nor a denylist of named types.
"""
from __future__ import annotations

import re

PRD = "prd"
TECH_DOC = "tech_doc"
IMPLEMENTATION_PLAN = "implementation_plan"
TEST_PLAN = "test_plan"
RELEASE_NOTE = "release_note"
MEETING_NOTE = "meeting_note"
OTHER = "other"

DOC_TYPES = (PRD, TECH_DOC, IMPLEMENTATION_PLAN, TEST_PLAN, RELEASE_NOTE, MEETING_NOTE, OTHER)

# Ordered most-specific first. A page titled "Implementation Plan — Fraud-report-anchored
# verification" is an implementation plan even though it references a PRD, so the compound
# markers are checked before the bare "PRD" token.
_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    (TEST_PLAN, (r"test\s*plan", r"test\s*cases?\b", r"\bqa\s*plan\b", r"\bregression\s*suite\b")),
    (IMPLEMENTATION_PLAN, (r"implementation\s*plan", r"impl\.?\s*plan\b")),
    (TECH_DOC, (r"tech\s*doc", r"\btechdoc\b", r"tech(nical)?\s*design",
                r"technical\s*document", r"\bhld\b", r"\blld\b", r"architecture\b")),
    (RELEASE_NOTE, (r"release\s*notes?", r"\bv\d+\.\d+(\.\d+)?\b", r"\bchangelog\b",
                    r"\bhotfix\b")),
    (MEETING_NOTE, (r"\bgrooming\b", r"meeting\s*notes?", r"\bstand[-\s]?up\b",
                    r"\bretro(spective)?\b", r"\bsync\s*notes?\b", r"\bplanning\b",
                    r"\b(daily|weekly|monthly|quarterly)\s+(review|update)\b")),
    # Checked last so a compound title resolves to the more specific type above.
    (PRD, (r"\bprds?\b", r"product\s*requirement", r"\bbrd\b", r"\bfrd\b",
           r"requirement\s*document")),
)

_COMPILED = tuple(
    (dtype, tuple(re.compile(p, re.I) for p in pats)) for dtype, pats in _RULES
)

# Elasticsearch clauses per type, declared explicitly rather than derived from the regexes
# above. Deriving them mangled the patterns — `\bprds?\b` turned into the phrase "prds",
# which matches nothing and would silently have excluded every "PRD:" document from an
# include=[prd] filter. Two representations, kept side by side and tested against each other.
_ES_CLAUSES: dict[str, list[dict]] = {
    TEST_PLAN: [
        {"match_phrase": {"doc_title": "test plan"}},
        {"match_phrase": {"doc_title": "test case"}},
        {"match_phrase": {"doc_title": "test cases"}},
        {"match_phrase": {"doc_title": "qa plan"}},
        {"match_phrase": {"doc_title": "regression suite"}},
    ],
    IMPLEMENTATION_PLAN: [
        {"match_phrase": {"doc_title": "implementation plan"}},
        {"match_phrase": {"doc_title": "impl plan"}},
    ],
    TECH_DOC: [
        {"match_phrase": {"doc_title": "tech doc"}},
        {"match_phrase": {"doc_title": "techdoc"}},
        {"match_phrase": {"doc_title": "tech design"}},
        {"match_phrase": {"doc_title": "technical design"}},
        {"match_phrase": {"doc_title": "technical document"}},
        {"match_phrase": {"doc_title": "architecture"}},
        {"match_phrase": {"doc_title": "hld"}},
        {"match_phrase": {"doc_title": "lld"}},
    ],
    RELEASE_NOTE: [
        {"match_phrase": {"doc_title": "release note"}},
        {"match_phrase": {"doc_title": "release notes"}},
        {"match_phrase": {"doc_title": "changelog"}},
        {"match_phrase": {"doc_title": "hotfix"}},
        # A version-numbered title cannot be expressed as a phrase; regexp on the keyword
        # subfield handles "v1.6.2 : Added Analytics SDK".
        {"regexp": {"doc_title.keyword": "[vV][0-9]+\\.[0-9]+.*"}},
    ],
    MEETING_NOTE: [
        {"match_phrase": {"doc_title": "grooming"}},
        {"match_phrase": {"doc_title": "meeting notes"}},
        {"match_phrase": {"doc_title": "meeting note"}},
        {"match_phrase": {"doc_title": "standup"}},
        {"match_phrase": {"doc_title": "stand up"}},
        {"match_phrase": {"doc_title": "retro"}},
        {"match_phrase": {"doc_title": "retrospective"}},
        {"match_phrase": {"doc_title": "sync notes"}},
        {"match_phrase": {"doc_title": "planning"}},
    ],
    PRD: [
        {"match_phrase": {"doc_title": "prd"}},
        {"match_phrase": {"doc_title": "prds"}},
        {"match_phrase": {"doc_title": "product requirement"}},
        {"match_phrase": {"doc_title": "product requirements"}},
        {"match_phrase": {"doc_title": "requirement document"}},
        {"match_phrase": {"doc_title": "brd"}},
        {"match_phrase": {"doc_title": "frd"}},
    ],
}


def classify(title: str | None) -> str:
    """Best-effort document type from a title. Returns OTHER when nothing matches."""
    t = (title or "").strip()
    if not t:
        return OTHER
    for dtype, patterns in _COMPILED:
        if any(p.search(t) for p in patterns):
            return dtype
    return OTHER


def _should_clauses(dtype: str) -> list[dict]:
    """
    Elasticsearch clauses matching a type. match_phrase runs against the analysed
    doc_title, so it is case-insensitive and tolerant of surrounding punctuation
    ("PRD -", "PRD:", "- PRD").
    """
    return [dict(c) for c in _ES_CLAUSES.get(dtype, [])]


def title_filter(
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> dict | None:
    """
    Build an ES bool filter scoping results by document type, derived from doc_title.

    Lets the existing 3,400-document corpus be narrowed at query time with no re-index.

    include: only these types. Documents with no recognisable marker (OTHER) are excluded
             unless OTHER is named explicitly — asking for `prd` should not return an
             untitled analytics page.
    exclude: drop these types. OTHER is never dropped by an exclude list, because a page
             with an unhelpful title is not evidence of being the wrong kind of page.

    Returns None when there is nothing to constrain.
    """
    include = [t for t in (include or []) if t in DOC_TYPES]
    exclude = [t for t in (exclude or []) if t in DOC_TYPES]
    if not include and not exclude:
        return None

    bool_q: dict = {}

    if include:
        shoulds: list[dict] = []
        for t in include:
            if t == OTHER:
                # "other" means: matches none of the named type patterns.
                named = [c for d, _ in _RULES for c in _should_clauses(d)]
                shoulds.append({"bool": {"must_not": named}})
            else:
                shoulds.extend(_should_clauses(t))
        if shoulds:
            bool_q["should"] = shoulds
            bool_q["minimum_should_match"] = 1

    if exclude:
        musts_not: list[dict] = []
        for t in exclude:
            if t == OTHER:
                continue
            musts_not.extend(_should_clauses(t))
        if musts_not:
            bool_q["must_not"] = musts_not

    return {"bool": bool_q} if bool_q else None
