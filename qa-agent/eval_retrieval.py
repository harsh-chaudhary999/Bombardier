"""
Retrieval Quality Evaluator
===========================
Tests the retrieval pipeline WITHOUT an LLM — uses /analyze/validate to inspect
what the agent would receive as context before any AI call.

Usage:
    python eval_retrieval.py --prd confluence:1234567890 --module Platform
    python eval_retrieval.py --prd confluence:1234567890 --module Platform --verbose
    python eval_retrieval.py --prd confluence:1234567890 --judge  # eyeball mode

Outputs:
    - Score distribution histogram (ASCII)
    - Per-query: top test matches + KB matches
    - Elbow detection: where scores naturally drop off
    - KB contamination check: cross-module docs that slipped through
"""

import argparse
import json
import sys
from collections import Counter

import httpx


API_BASE = "http://localhost:8000"


# ─── Fetch ─────────────────────────────────────────────────────────────────────

def validate(prd_source_id: str, module: str | None, top_k_tests: int = 200) -> dict:
    payload = {"prd_source_id": prd_source_id, "top_k_tests": top_k_tests}
    if module:
        payload["module"] = [module]

    resp = httpx.post(
        f"{API_BASE}/analyze/validate",
        json=payload,
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# ─── Display helpers ───────────────────────────────────────────────────────────

def ascii_histogram(scores: list[float], buckets: int = 10, width: int = 40) -> str:
    if not scores:
        return "  (no scores)"
    lo, hi = min(scores), max(scores)
    if lo == hi:
        return f"  all scores = {lo:.1f}"

    bucket_size = (hi - lo) / buckets
    counts = Counter()
    for s in scores:
        b = min(int((s - lo) / bucket_size), buckets - 1)
        counts[b] += 1

    lines = []
    for i in range(buckets):
        label = f"{lo + i * bucket_size:5.1f}–{lo + (i+1)*bucket_size:5.1f}"
        bar = "█" * int(counts[i] / len(scores) * width)
        lines.append(f"  {label} |{bar} {counts[i]}")
    return "\n".join(lines)


def find_elbow(scores: list[float]) -> float | None:
    """Return the score just after the largest relative drop."""
    if len(scores) < 3:
        return None
    drops = [
        (scores[i] - scores[i + 1]) / scores[i]
        for i in range(len(scores) - 1)
        if scores[i] > 0
    ]
    if not drops:
        return None
    biggest = drops.index(max(drops))
    if drops[biggest] < 0.15:   # less than 15% drop — no clear elbow
        return None
    return scores[biggest + 1]


def color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def green(t): return color(t, "32")
def yellow(t): return color(t, "33")
def red(t): return color(t, "31")
def bold(t): return color(t, "1")
def dim(t): return color(t, "2")


# ─── Report ───────────────────────────────────────────────────────────────────

def print_report(data: dict, module_arg: str | None = None, verbose: bool = False, judge: bool = False):
    prd_status = data.get("prd_status", {})
    prd_id     = prd_status.get("source_id", "?")
    doc_title  = prd_status.get("doc_title", "")
    chunks     = prd_status.get("chunk_count", len(data.get("prd_chunks", [])))
    module     = module_arg  # API doesn't echo module — use the arg we sent
    results    = data.get("retrieval", [])
    auto_ing   = data.get("auto_ingested")

    print()
    print(bold("=" * 70))
    print(bold(f"  Retrieval Evaluation: {prd_id}"))
    print(bold("=" * 70))
    if doc_title:
        print(f"  Document         : {doc_title}")
    print(f"  PRD chunks in ES : {chunks}")
    print(f"  Module filter    : {module or '(none)'}")
    if auto_ing:
        print(yellow(f"  ⚡ Auto-ingested  : {auto_ing.get('chunks_ingested', '?')} chunks"))

    # Top-line summary from deduplication
    total   = data.get("total_unique_tests", 0)
    n_high  = data.get("total_high_confidence", 0)
    n_med   = data.get("total_medium_confidence", 0)
    pool    = data.get("pool_size_used")
    if total:
        print(f"  Unique tests     : {green(str(n_high))} high · {yellow(str(n_med))} medium = {bold(str(total))} total")
    if pool:
        print(f"  Pool size used   : {pool} candidates/query")
    print()

    if not results:
        print(red("  No retrieval results — PRD may have no testable sections."))
        return

    all_test_scores = []
    all_kb_scores   = []

    for entry in results:
        query       = entry.get("query", "?")
        tests       = entry.get("test_matches", [])
        kb          = entry.get("kb_matches", [])
        test_scores = [t["score"] for t in tests]
        kb_scores   = [k["score"] for k in kb]
        all_test_scores.extend(test_scores)
        all_kb_scores.extend(kb_scores)

        elbow = find_elbow(test_scores)

        n_high   = entry.get("test_matches_high",   sum(1 for t in tests if t.get("confidence") == "high"))
        n_medium = entry.get("test_matches_medium", sum(1 for t in tests if t.get("confidence") == "medium"))
        n_below  = entry.get("test_matches_below_threshold", 0)

        print(bold(f"  ── Query: {query[:70]}"))
        parts = []
        if n_high:   parts.append(green(f"{n_high} high"))
        if n_medium: parts.append(yellow(f"{n_medium} medium"))
        if n_below:  parts.append(dim(f"{n_below} below threshold"))
        if not tests and not n_below:
            parts.append(dim("0 matches"))
        print(f"     Tests          : {' · '.join(parts) if parts else dim('0 matches')}")

        if tests:
            print(f"     Score range    : {min(test_scores):.1f} – {max(test_scores):.1f}")
            if elbow:
                print(yellow(f"     Natural elbow  : {elbow:.1f}"))

        if verbose and tests:
            print()
            for t in tests[:10]:
                score_str  = f"{t['score']:5.1f}"
                confidence = t.get("confidence", "high")
                mod        = t.get("module") or ""
                key        = t.get("jira_key", "?")
                summary    = t.get("summary", "")[:60]
                labels     = ",".join(t.get("labels") or [])[:30]

                score_col = green(score_str) if confidence == "high" else yellow(score_str)
                mod_flag  = ""
                if module and mod and mod.lower() != module.lower():
                    mod_flag = red(f" [!WRONG MODULE: {mod}]")

                conf_tag = "" if confidence == "high" else dim(" [medium]")
                print(f"       {score_col}  {bold(key)}  {summary}{conf_tag}{mod_flag}")
                if labels:
                    print(f"              {dim(labels)}")
            if len(tests) > 10:
                print(dim(f"       ... and {len(tests) - 10} more"))

        if kb:
            print(f"     KB matches     : {len(kb)}")
            if verbose:
                for k in kb:
                    score  = k["score"]
                    src    = k.get("source_id", "?")
                    title  = (k.get("doc_title") or src)[:50]
                    sec    = (k.get("section_heading") or "")[:40]
                    print(f"       {yellow(f'{score:.4f}')}  {title}  {dim(sec)}")

        # Judge mode: ask user if results are relevant
        if judge and tests:
            print()
            print(yellow("  Top 5 results — relevant? (y/n/skip): "))
            relevant_count = 0
            for t in tests[:5]:
                ans = input(f"    [{t['jira_key']}] {t['summary'][:60]} → ").strip().lower()
                if ans == "y":
                    relevant_count += 1
                elif ans == "skip":
                    break
            if relevant_count > 0:
                print(green(f"    Precision@5: {relevant_count}/5 ({relevant_count*20}%)"))

        print()

    # ── Overall summary ─────────────────────────────────────────────────────
    print(bold("  ── Score Distribution (all queries)"))
    if all_test_scores:
        print(f"\n  Test scores ({len(all_test_scores)} total):")
        print(ascii_histogram(sorted(all_test_scores, reverse=True)))

        elbow_overall = find_elbow(sorted(all_test_scores, reverse=True))
        if elbow_overall:
            print(yellow(f"\n  → Suggested MIN_SCORE: {elbow_overall:.1f}  (current in code)"))

    if all_kb_scores:
        print(f"\n  KB scores ({len(all_kb_scores)} total):")
        print(ascii_histogram(sorted(all_kb_scores, reverse=True), buckets=5))

    # ── Cross-module contamination check ────────────────────────────────────
    if module:
        print()
        print(bold("  ── Cross-Module Contamination Check"))
        contaminated = [
            t for entry in results
            for t in entry.get("test_matches", [])
            if t.get("module") and t["module"].lower() != module.lower()
        ]
        if contaminated:
            by_mod = Counter(t["module"] for t in contaminated)
            print(red(f"  ⚠  {len(contaminated)} tests from wrong module(s): {dict(by_mod)}"))
            if verbose:
                for t in contaminated[:5]:
                    print(red(f"     {t['jira_key']} [{t['module']}] {t.get('summary','')[:50]}"))
        else:
            print(green("  ✓  No cross-module contamination"))

    print()
    print(bold("=" * 70))
    print()


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality for a PRD")
    parser.add_argument("--prd",     required=True, help="e.g. confluence:1234567890")
    parser.add_argument("--module",  help="Module filter e.g. Platform")
    parser.add_argument("--top-k",   type=int, default=200, help="Candidate pool size (default 200)")
    parser.add_argument("--verbose",   action="store_true", help="Show all matches per query")
    parser.add_argument("--all-tests", action="store_true", help="Dump full deduplicated test list at end")
    parser.add_argument("--judge",     action="store_true", help="Interactive: rate top-5 per query")
    parser.add_argument("--json",      action="store_true", help="Dump raw JSON response and exit")
    args = parser.parse_args()

    print(f"Calling /analyze/validate for {args.prd} ...")
    try:
        data = validate(args.prd, args.module, args.top_k)
    except httpx.HTTPStatusError as e:
        print(red(f"API error {e.response.status_code}: {e.response.text}"))
        sys.exit(1)
    except httpx.ConnectError:
        print(red(f"Cannot connect to {API_BASE} — is the qa-agent container running?"))
        sys.exit(1)

    if args.json:
        print(json.dumps(data, indent=2))
        return

    print_report(data, module_arg=args.module, verbose=args.verbose, judge=args.judge)

    if args.all_tests:
        all_tests = data.get("all_tests", [])
        print(bold(f"\n  ── All {len(all_tests)} Unique Tests (deduplicated, score desc) ──"))
        for t in all_tests:
            conf      = t.get("confidence", "high")
            score_col = green(f"{t['score']:5.1f}") if conf == "high" else yellow(f"{t['score']:5.1f}")
            queries   = ", ".join(t.get("matched_queries", []))[:60]
            print(f"  {score_col}  {bold(t['jira_key'])}  {t['summary'][:55]}")
            if queries:
                print(f"           {dim('via: ' + queries)}")


if __name__ == "__main__":
    main()
