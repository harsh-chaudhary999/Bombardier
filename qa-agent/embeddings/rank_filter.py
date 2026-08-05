"""
Score-based result filtering that survives a compressed score distribution.

Measured on the real corpus: for one real query, ranks 1..15 span **5.5%** — the definitive answer scores 0.8365 and an
unrelated UI release note scores 0.8030. Against that distribution every absolute
threshold in this codebase is meaningless: `KB_MIN_SCORE = 0.77` and `RRF_HIGH = 0.025`
either keep everything or nothing depending on the query, and tuning them per query is
not a strategy.

What *does* work under compression is relative separation. In that same distribution the
top result sits roughly 6 standard deviations above the mean of the tail — a very strong
signal, invisible to any fixed cut-off. So:

  separation()   quantifies whether score is usable as a signal at all, and is reported to
                 the caller rather than hidden. A ranking with no separation should not be
                 silently trimmed and presented as confident.

  relative_cut() finds the largest relative drop (the "knee") and cuts there, with a floor
                 so a genuinely tied cluster is not reduced to one result.

Neither invents information. When the scores carry no signal, `separation()` says so and
the caller can widen the net or fix retrieval instead of trusting a threshold.
"""
from __future__ import annotations

import statistics
from typing import Any


def _scores(results: list[dict], key: str) -> list[float]:
    out = []
    for r in results:
        v = r.get(key)
        if v is None:
            v = r.get("score")
        out.append(float(v or 0.0))
    return out


def separation(results: list[dict], key: str = "score") -> dict[str, Any]:
    """
    Describe how well scores distinguish the top hit from the rest.

    spread_pct   relative distance from rank 1 to the last result
    top_z        how many standard deviations rank 1 sits above the mean of ranks 2..n
    knee_index   0-based index after which the largest relative drop occurs
    usable       whether score is informative enough to filter on
    """
    s = _scores(results, key)
    n = len(s)
    if n == 0:
        return {"count": 0, "spread_pct": 0.0, "top_z": 0.0, "knee_index": 0,
                "usable": False, "reason": "no results"}
    if n == 1:
        return {"count": 1, "spread_pct": 0.0, "top_z": 0.0, "knee_index": 0,
                "usable": True, "reason": "single result"}

    top, last = s[0], s[-1]
    spread_pct = ((top - last) / abs(top) * 100) if top else 0.0

    tail = s[1:]
    mean_tail = statistics.fmean(tail)
    stdev_tail = statistics.pstdev(tail) if len(tail) > 1 else 0.0
    top_z = ((top - mean_tail) / stdev_tail) if stdev_tail > 1e-9 else 0.0

    # Largest absolute drop between consecutive results; index is the position *before* it.
    drops = [s[i] - s[i + 1] for i in range(n - 1)]
    knee_index = max(range(len(drops)), key=lambda i: drops[i]) if drops else 0
    median_drop = statistics.median(drops) if drops else 0.0
    knee_ratio = (drops[knee_index] / median_drop) if median_drop > 1e-12 else 0.0

    # Either a meaningful absolute spread, or a statistically distinct leader.
    usable = spread_pct >= 15.0 or top_z >= 2.0
    if usable:
        reason = "wide spread" if spread_pct >= 15.0 else f"leader {top_z:.1f}σ above tail"
    else:
        reason = (f"compressed: {spread_pct:.1f}% spread, leader only {top_z:.1f}σ above "
                  "tail — score cannot separate signal from noise here")

    return {
        "count": n,
        "spread_pct": round(spread_pct, 2),
        "top_z": round(top_z, 2),
        "knee_index": knee_index,
        "knee_ratio": round(knee_ratio, 2),
        "usable": usable,
        "reason": reason,
    }


def relative_cut(
    results: list[dict],
    key: str = "score",
    *,
    min_keep: int = 3,
    max_keep: int | None = None,
    knee_ratio: float = 3.0,
) -> tuple[list[dict], dict[str, Any]]:
    """
    Trim a ranked list at its knee, keeping at least `min_keep`.

    Cuts after the largest consecutive drop, but only when that drop is at least
    `knee_ratio` times the median drop — otherwise the list is smoothly graded and there is
    no defensible cut point, so nothing is removed beyond `max_keep`.

    `min_keep` exists because knee detection on a compressed distribution often fires
    immediately after rank 1: in the measured example the rank1→rank2 drop is 6x the median,
    yet rank 2 was genuinely relevant. Cutting to a single result would have discarded it.

    Returns (kept, diagnostics) — diagnostics always includes the separation report so the
    caller can tell a confident trim from an arbitrary one.
    """
    if not results:
        return [], separation(results, key)

    ordered = sorted(results, key=lambda r: float(r.get(key) or r.get("score") or 0.0),
                     reverse=True)
    diag = separation(ordered, key)

    keep = len(ordered)
    if diag["count"] > 1 and diag["knee_ratio"] >= knee_ratio:
        keep = diag["knee_index"] + 1

    keep = max(keep, min(min_keep, len(ordered)))
    if max_keep is not None:
        keep = min(keep, max_keep)

    diag = {**diag, "kept": keep, "dropped": len(ordered) - keep,
            "cut_applied": keep < len(ordered)}
    return ordered[:keep], diag
