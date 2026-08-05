"""
Cross-encoder reranker for QA Intelligence Engine.

After initial retrieval (KNN + BM25 via RRF), a cross-encoder scores each
(query, document) pair with full cross-attention — much more accurate than
bi-encoder similarity but too slow for first-stage retrieval.

Two-stage pipeline:
  Stage 1 (es_store.py):  Fast retrieval via RRF — returns top ~50-100 candidates
  Stage 2 (this module):  Precise reranking via cross-encoder — returns top ~20

Model: cross-encoder/ms-marco-MiniLM-L-6-v2  (22M params, ~2ms/pair on CPU)
  - For 50 candidates: ~100ms reranking time
  - English-focused; for heavy Hindi content consider mmarco-mMiniLMv2-L6-H384-v1
"""
import logging
import os

from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

MODEL_NAME = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# Cross-encoder logit scales are MODEL-SPECIFIC. The thresholds used elsewhere
# (RERANK_HIGH=2.0 / RERANK_MEDIUM=0.5) were calibrated for ms-marco-MiniLM; swapping in
# BGE-reranker-v2-m3 changes the scale and silently invalidates them. Keep them together
# with the model choice so the two cannot drift apart.
RERANK_HIGH = float(os.environ.get("QA_RERANK_HIGH", "2.0"))
RERANK_MEDIUM = float(os.environ.get("QA_RERANK_MEDIUM", "0.5"))


class Reranker:
    """
    Cross-encoder reranker. Loaded once at startup, reused for all requests.
    Thread-safe — CrossEncoder.predict() holds the GIL during inference.
    """

    def __init__(self) -> None:
        self._scale_warned = False
        logger.info("Loading cross-encoder reranker: %s", MODEL_NAME)
        self._model = CrossEncoder(MODEL_NAME)
        logger.info("Reranker loaded.")

    def rerank(
        self,
        query: str,
        results: list[dict],
        text_field: str = "summary",
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank search results using cross-encoder scoring.

        Args:
            query:      The search query text
            results:    List of dicts from ES search (must contain text_field)
            text_field: Primary field to use as document text for scoring.
                        If 'steps_text' is also present, it's appended for richer context.
            top_k:      If set, return only top_k results after reranking.

        Returns:
            Results list re-sorted by cross-encoder score.
            Each dict gets a 'rerank_score' field added.
            Original 'score' (from ES/RRF) is preserved.
        """
        if not results or not query:
            return results

        # Build (query, document) pairs for cross-encoder
        pairs = []
        for r in results:
            doc_text = r.get(text_field, "") or ""
            # Enrich with description (Xray test steps live here when steps_text is null)
            description = r.get("description", "") or ""
            if description:
                doc_text = f"{doc_text}\n{description[:400]}"
            # Also enrich with steps_text if available
            steps = r.get("steps_text", "") or ""
            if steps:
                doc_text = f"{doc_text}\n{steps[:300]}"
            pairs.append((query, doc_text))

        # Cross-encoder scoring — batch for efficiency
        scores = self._model.predict(pairs, batch_size=32, show_progress_bar=False)

        for r, score in zip(results, scores):
            r["rerank_score"] = round(float(score), 4)

        # Measured on the real corpus, ms-marco returned about -11 for the BEST hits:
        # it judged every candidate irrelevant, so reranking contributed nothing while
        # appearing to work. Surface that rather than let it pass silently.
        top = max(scores) if len(scores) else 0.0
        if len(scores) and float(top) < RERANK_MEDIUM and not self._scale_warned:
            self._scale_warned = True
            logger.warning(
                "reranker %s scored its BEST candidate at %.2f, below QA_RERANK_MEDIUM=%.2f "
                "— every result will be classed low-confidence. Either the model is a poor "
                "fit for this content (ms-marco is English-only) or the thresholds need "
                "recalibrating for this model's logit scale.",
                MODEL_NAME, float(top), RERANK_MEDIUM,
            )
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked
