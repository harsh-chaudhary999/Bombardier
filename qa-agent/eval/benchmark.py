"""
Retrieval Benchmark with Ground Truth Evaluation
==================================================

Measures retrieval quality against labeled (query, expected_jira_keys) pairs.

Metrics computed:
  - Recall@K  — what fraction of expected keys appear in top-K results?
  - MRR       — Mean Reciprocal Rank: how high is the first relevant result?
  - nDCG@K    — Normalized Discounted Cumulative Gain: quality-weighted ranking
  - Precision@K — what fraction of top-K results are relevant?

Usage:
    # Seed the labelled set from the template (ground_truth.json is gitignored — it holds
    # real queries and Jira keys, which must not be committed)
    cp eval/ground_truth.example.json eval/ground_truth.json

    # Run against ground truth file
    python -m eval.benchmark --ground-truth eval/ground_truth.json

    # Run against ground truth + also export failed retrievals for debugging
    python -m eval.benchmark --ground-truth eval/ground_truth.json --export-failures

    # Build ground truth from approved review decisions (feedback loop)
    python -m eval.benchmark --build-from-decisions --run-id <uuid>

Requires:
    - qa-agent running at http://localhost:8000
    - Ground truth JSON file with labeled queries
"""
import argparse
import json
import math
import sys
from pathlib import Path

import httpx

API_BASE = "http://localhost:8000"


# ─── Metrics ──────────────────────────────────────────────────────────────────

def recall_at_k(retrieved_keys: list[str], expected_keys: list[str], k: int) -> float:
    """Fraction of expected keys found in top-k retrieved results."""
    if not expected_keys:
        return 1.0  # nothing to find = perfect recall
    top_k = set(retrieved_keys[:k])
    found = len(top_k & set(expected_keys))
    return found / len(expected_keys)


def precision_at_k(retrieved_keys: list[str], expected_keys: list[str], k: int) -> float:
    """Fraction of top-k results that are relevant."""
    if k == 0:
        return 0.0
    top_k = retrieved_keys[:k]
    relevant = sum(1 for key in top_k if key in expected_keys)
    return relevant / len(top_k)


def mrr(retrieved_keys: list[str], expected_keys: list[str]) -> float:
    """Mean Reciprocal Rank — how high is the first relevant result?"""
    expected_set = set(expected_keys)
    for i, key in enumerate(retrieved_keys):
        if key in expected_set:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved_keys: list[str], expected_keys: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at k."""
    expected_set = set(expected_keys)
    top_k = retrieved_keys[:k]

    # DCG: sum of 1/log2(rank+1) for relevant results
    dcg = sum(
        1.0 / math.log2(i + 2)  # i+2 because i is 0-indexed, log2(1)=0
        for i, key in enumerate(top_k)
        if key in expected_set
    )

    # Ideal DCG: all relevant results at the top
    ideal_relevant = min(len(expected_keys), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_relevant))

    if idcg == 0:
        return 1.0  # nothing to rank = perfect
    return dcg / idcg


# ─── Search ───────────────────────────────────────────────────────────────────

def search_hybrid(query: str, module: str | None, top_k: int = 50) -> list[dict]:
    """Call the /search/tests endpoint."""
    payload = {
        "prd_text": query,
        "mode": "hybrid",
        "top_k": top_k,
    }
    if module:
        payload["module"] = [module]

    resp = httpx.post(f"{API_BASE}/search/tests", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json().get("results", [])


# ─── Benchmark ────────────────────────────────────────────────────────────────

def run_benchmark(ground_truth_path: str, top_k_values: list[int] = None, export_failures: bool = False):
    """Run the full benchmark against a ground truth file."""
    if top_k_values is None:
        top_k_values = [5, 10, 20]

    with open(ground_truth_path) as f:
        data = json.load(f)

    queries = data.get("queries", [])
    labeled = [q for q in queries if q.get("expected_keys")]

    if not labeled:
        print("\n  No labeled queries found in ground truth file.")
        print("  Add jira_keys to the 'expected_keys' field for each query.")
        print(f"  File: {ground_truth_path}")
        print("\n  To bootstrap labels from approved review decisions:")
        print("    python -m eval.benchmark --build-from-decisions --run-id <uuid>")
        return

    print(f"\n  Running benchmark with {len(labeled)} labeled queries...")
    print(f"  K values: {top_k_values}")
    print()

    max_k = max(top_k_values)
    results_per_query = []
    failures = []

    for q in labeled:
        query = q["query"]
        module = q.get("module")
        expected = q["expected_keys"]

        print(f"  Searching: {query[:60]}...", end="", flush=True)
        try:
            results = search_hybrid(query, module, top_k=max_k)
            retrieved_keys = [r["jira_key"] for r in results]
        except Exception as e:
            print(f" ERROR: {e}")
            retrieved_keys = []

        metrics = {}
        for k in top_k_values:
            metrics[f"recall@{k}"] = recall_at_k(retrieved_keys, expected, k)
            metrics[f"precision@{k}"] = precision_at_k(retrieved_keys, expected, k)
            metrics[f"ndcg@{k}"] = ndcg_at_k(retrieved_keys, expected, k)
        metrics["mrr"] = mrr(retrieved_keys, expected)

        results_per_query.append({
            "query": query,
            "module": module,
            "expected": expected,
            "retrieved_top10": retrieved_keys[:10],
            "metrics": metrics,
        })

        # Track failures
        missed = set(expected) - set(retrieved_keys[:max_k])
        if missed:
            failures.append({"query": query, "missed_keys": list(missed)})

        r5 = metrics.get(f"recall@{top_k_values[0]}", 0)
        print(f"  Recall@{top_k_values[0]}={r5:.0%}  MRR={metrics['mrr']:.2f}")

    # ── Aggregate metrics ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  AGGREGATE RESULTS")
    print("=" * 60)

    for k in top_k_values:
        avg_recall = sum(r["metrics"][f"recall@{k}"] for r in results_per_query) / len(results_per_query)
        avg_precision = sum(r["metrics"][f"precision@{k}"] for r in results_per_query) / len(results_per_query)
        avg_ndcg = sum(r["metrics"][f"ndcg@{k}"] for r in results_per_query) / len(results_per_query)
        print(f"  Recall@{k:>2}:    {avg_recall:.1%}")
        print(f"  Precision@{k:>2}: {avg_precision:.1%}")
        print(f"  nDCG@{k:>2}:     {avg_ndcg:.3f}")
        print()

    avg_mrr = sum(r["metrics"]["mrr"] for r in results_per_query) / len(results_per_query)
    print(f"  MRR:          {avg_mrr:.3f}")

    if failures:
        print(f"\n  Missed retrievals: {len(failures)}/{len(labeled)} queries had missing keys")
        if export_failures:
            fail_path = Path(ground_truth_path).with_suffix(".failures.json")
            with open(fail_path, "w") as f:
                json.dump(failures, f, indent=2)
            print(f"  Exported failures to: {fail_path}")

    # Export full results
    results_path = Path(ground_truth_path).with_suffix(".results.json")
    with open(results_path, "w") as f:
        json.dump({
            "queries": len(labeled),
            "top_k_values": top_k_values,
            "per_query": results_per_query,
            "aggregate": {
                f"recall@{k}": sum(r["metrics"][f"recall@{k}"] for r in results_per_query) / len(results_per_query)
                for k in top_k_values
            } | {"mrr": avg_mrr},
        }, f, indent=2)
    print(f"\n  Full results saved to: {results_path}")
    print()


# ─── Build ground truth from approved decisions (feedback loop) ───────────────

def build_from_decisions(run_id: str, output_path: str = "eval/ground_truth_from_reviews.json"):
    """
    Build ground truth labels from approved human review decisions.

    For each PRD section that has approved 'keep' or 'update' decisions,
    the corresponding jira_keys become the expected results for that section's query.
    """
    print(f"  Fetching decisions for run_id={run_id}...")
    resp = httpx.get(
        f"{API_BASE}/analyze/decisions/{run_id}?paginate=false",
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()

    decisions = data.get("decisions", {})
    approved_by_section: dict[str, list[str]] = {}

    for action in ("keep", "update"):
        for d in decisions.get(action, []):
            # Only use reviewed+approved decisions as ground truth
            if not d.get("approved"):
                continue
            section = d.get("prd_section") or "unknown"
            key = d.get("jira_key")
            if key:
                approved_by_section.setdefault(section, []).append(key)

    if not approved_by_section:
        print("  No approved keep/update decisions found. Review some decisions first.")
        return

    queries = []
    for section, keys in approved_by_section.items():
        queries.append({
            "query": section,
            "module": None,
            "expected_keys": keys,
            "notes": f"Auto-generated from approved decisions in run {run_id}",
        })

    output = {
        "_description": f"Ground truth generated from approved decisions (run {run_id})",
        "queries": queries,
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"  Generated {len(queries)} labeled queries from {sum(len(v) for v in approved_by_section.values())} approved decisions")
    print(f"  Saved to: {output_path}")
    print(f"\n  Now run:  python -m eval.benchmark --ground-truth {output_path}")


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark retrieval quality with ground truth metrics")
    parser.add_argument("--ground-truth", help="Path to ground truth JSON file")
    parser.add_argument("--export-failures", action="store_true", help="Export missed retrievals to .failures.json")
    parser.add_argument("--build-from-decisions", action="store_true", help="Build ground truth from approved review decisions")
    parser.add_argument("--run-id", help="Run ID for --build-from-decisions")
    parser.add_argument("--api-base", default="http://localhost:8000", help="QA Engine API base URL")
    args = parser.parse_args()

    global API_BASE
    API_BASE = args.api_base

    if args.build_from_decisions:
        if not args.run_id:
            print("  --run-id required with --build-from-decisions")
            sys.exit(1)
        build_from_decisions(args.run_id)
    elif args.ground_truth:
        run_benchmark(args.ground_truth, export_failures=args.export_failures)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
