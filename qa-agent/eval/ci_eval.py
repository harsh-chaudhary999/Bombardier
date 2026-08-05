"""
CI-friendly evaluation runner for the QA Intelligence Engine.

Runs retrieval benchmarks against ground truth and exits with
non-zero code if quality drops below configured thresholds.

Usage:
    # Run in CI pipeline. ground_truth.json is gitignored (real queries and Jira keys);
    # CI must supply it from a secret store or seed it from eval/ground_truth.example.json.
    python -m eval.ci_eval --ground-truth eval/ground_truth.json --min-recall 0.70 --min-mrr 0.50

    # With custom Elasticsearch URL
    ELASTICSEARCH_URL=http://localhost:9200 python -m eval.ci_eval ...
"""
import argparse
import json
import logging
import sys
import os
import math

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


def run_benchmark(ground_truth_path: str, es_url: str) -> dict:
    """Run retrieval benchmark and return metrics."""
    from embeddings.embed_client import EmbedClient
    from embeddings.es_store import ESStore

    os.environ.setdefault("ELASTICSEARCH_URL", es_url)

    with open(ground_truth_path) as f:
        gt = json.load(f)

    queries = gt.get("queries", [])
    if not queries:
        logger.error("No queries found in ground truth file")
        return {}

    embed_client = EmbedClient()
    es_store = ESStore()

    # Try loading reranker
    reranker = None
    try:
        from embeddings.reranker import Reranker
        reranker = Reranker()
    except Exception:
        logger.info("Reranker not available, using RRF scores only")

    results = []
    for q in queries:
        query_text = q["query"]
        expected_keys = set(q.get("expected_keys", []))
        module = q.get("module")
        module_filter = [module] if module else None

        query_vec = embed_client.embed_query(query_text)
        retrieval_k = 60 if reranker else 20

        hits = es_store.search_hybrid(
            query_embedding=query_vec,
            keyword_query=query_text,
            top_k=retrieval_k,
            module_filter=module_filter,
        )

        if reranker and hits:
            hits = reranker.rerank(query_text, hits, top_k=20)

        retrieved_keys = [h["jira_key"] for h in hits[:20]]

        # Recall@20
        found = set(retrieved_keys) & expected_keys
        recall = len(found) / len(expected_keys) if expected_keys else 0

        # MRR
        mrr = 0.0
        for i, key in enumerate(retrieved_keys):
            if key in expected_keys:
                mrr = 1.0 / (i + 1)
                break

        # nDCG@20 (standard log2 discount)
        dcg = 0.0
        for i, key in enumerate(retrieved_keys):
            if key in expected_keys:
                dcg += 1.0 / math.log2(i + 2)
        ideal_dcg = sum(
            1.0 / math.log2(i + 2) for i in range(min(len(expected_keys), 20))
        )
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0

        results.append({
            "query": query_text,
            "recall@20": recall,
            "mrr": mrr,
            "ndcg@20": ndcg,
            "expected": len(expected_keys),
            "found": len(found),
        })

    # Aggregate
    avg_recall = sum(r["recall@20"] for r in results) / len(results)
    avg_mrr = sum(r["mrr"] for r in results) / len(results)
    avg_ndcg = sum(r["ndcg@20"] for r in results) / len(results)

    return {
        "queries": len(results),
        "avg_recall@20": round(avg_recall, 4),
        "avg_mrr": round(avg_mrr, 4),
        "avg_ndcg@20": round(avg_ndcg, 4),
        "per_query": results,
    }


def main():
    parser = argparse.ArgumentParser(description="CI retrieval quality evaluation")
    parser.add_argument("--ground-truth", required=True, help="Path to ground truth JSON")
    parser.add_argument("--min-recall", type=float, default=0.60, help="Minimum Recall@20 threshold (default: 0.60)")
    parser.add_argument("--min-mrr", type=float, default=0.40, help="Minimum MRR threshold (default: 0.40)")
    parser.add_argument("--min-ndcg", type=float, default=0.40, help="Minimum nDCG@20 threshold (default: 0.40)")
    parser.add_argument("--es-url", default="http://localhost:9200", help="Elasticsearch URL")
    parser.add_argument("--output", help="Write results JSON to this file")
    args = parser.parse_args()

    logger.info(f"Running benchmark against {args.ground_truth}")
    logger.info(f"Thresholds: recall>={args.min_recall}, mrr>={args.min_mrr}, ndcg>={args.min_ndcg}")

    metrics = run_benchmark(args.ground_truth, args.es_url)

    if not metrics:
        logger.error("Benchmark returned no results")
        sys.exit(2)

    # Print results
    logger.info(f"Results: Recall@20={metrics['avg_recall@20']}, MRR={metrics['avg_mrr']}, nDCG@20={metrics['avg_ndcg@20']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Results written to {args.output}")

    # Check thresholds
    failures = []
    if metrics["avg_recall@20"] < args.min_recall:
        failures.append(f"Recall@20 {metrics['avg_recall@20']:.4f} < {args.min_recall}")
    if metrics["avg_mrr"] < args.min_mrr:
        failures.append(f"MRR {metrics['avg_mrr']:.4f} < {args.min_mrr}")
    if metrics["avg_ndcg@20"] < args.min_ndcg:
        failures.append(f"nDCG@20 {metrics['avg_ndcg@20']:.4f} < {args.min_ndcg}")

    if failures:
        logger.error("QUALITY CHECK FAILED:")
        for f in failures:
            logger.error(f"  {f}")
        sys.exit(1)

    logger.info("QUALITY CHECK PASSED")
    sys.exit(0)


if __name__ == "__main__":
    main()
