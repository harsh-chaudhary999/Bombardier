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
  - English-focused; for heavy non-English content consider mmarco-mMiniLMv2-L6-H384-v1

Scoring an empty document is the failure mode to watch for. `predict()` happily returns a
number for a (query, "") pair — the *same* number for every such pair — so a broken field
mapping presents as a working reranker that produces a flat ranking, not as an error. That
is exactly what happened on the PRD path: `text_field` defaulted to `summary`, which only
test-case documents carry, so every PRD chunk scored an identical -8.6539 and an earlier
reading of those logits was wrongly attributed to the model being a poor fit for the
content. `_document_text` now resolves the body field per result and `rerank` logs an error
when any document comes back empty.
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
        text_field: str | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Rerank search results using cross-encoder scoring.

        Args:
            query:      The search query text
            results:    List of dicts from ES search
            text_field: Field to score on. Leave None to resolve per result — the two index
                        shapes here name their body differently (`chunk_text` for PRD chunks,
                        `summary` for test cases) and /ask can pass a mixed list, so one fixed
                        field name cannot serve every caller.
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
        empty = 0
        for r in results:
            doc_text = self._document_text(r, text_field)
            if not doc_text.strip():
                empty += 1
            pairs.append((query, doc_text))

        if empty:
            # A (query, "") pair still returns a number — the same number for every such
            # result — so this failure mode looks like a working reranker producing a flat
            # ranking. It has to be loud. This was the actual cause of PRD reranking scoring
            # every passage identically: text_field defaulted to `summary`, which PRD chunks
            # do not have, so the model never saw a document at all.
            logger.error(
                "reranker: %s/%s documents resolved to empty text (fields present on the "
                "first one: %s). Scores for those are meaningless and identical — fix the "
                "field mapping rather than trusting this ranking.",
                empty, len(results), sorted(results[0].keys()),
            )

        # Cross-encoder scoring — batch for efficiency
        scores = self._model.predict(pairs, batch_size=32, show_progress_bar=False)

        for r, score in zip(results, scores):
            r["rerank_score"] = round(float(score), 4)

        # A best-candidate score below the "medium" threshold means every result will be
        # classed low-confidence downstream. Report it rather than let it pass silently —
        # but note the cause is only model fit once `empty` above is zero.
        top = max(scores) if len(scores) else 0.0
        if len(scores) and float(top) < RERANK_MEDIUM and not self._scale_warned:
            self._scale_warned = True
            logger.warning(
                "reranker %s scored its BEST candidate at %.2f, below QA_RERANK_MEDIUM=%.2f "
                "— every result will be classed low-confidence. Either the documents are not "
                "reaching the model (see any empty-text error above), the model is a poor fit "
                "for this content (ms-marco is English-only), or the thresholds need "
                "recalibrating for this model's logit scale.",
                MODEL_NAME, float(top), RERANK_MEDIUM,
            )
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

    # Body-field names in priority order, covering both index shapes.
    _TEXT_FIELDS = ("chunk_text", "summary", "_text")

    # The cross-encoder truncates its input at the model's max sequence length and says
    # nothing about it. Only the ENRICHMENT used to be capped (description[:400],
    # steps[:300]) while the body field was unbounded — so a large PRD chunk filled the
    # window on its own and the description and steps appended after it were silently
    # dropped. Since chunks legitimately reach ~1200 tokens, that is the common case on
    # the PRD path, not an edge case.
    #
    # Budget in characters rather than tokens: the tokenizer is loaded but running it
    # twice per pair (once here, once inside predict) buys precision we do not need for
    # a cap whose job is only to keep the tail from being cut.
    _MODEL_TOKEN_LIMIT = 512
    _CHARS_PER_TOKEN = 3.5          # conservative for prose; tables tokenize denser
    _QUERY_RESERVE_TOKENS = 128     # room for the query side of the pair plus separators

    @classmethod
    def _document_budget(cls) -> int:
        return int((cls._MODEL_TOKEN_LIMIT - cls._QUERY_RESERVE_TOKENS) * cls._CHARS_PER_TOKEN)

    def _document_text(self, r: dict, text_field: str | None) -> str:
        """
        Resolve the text to score for one result, then enrich it — within a budget.

        Falls back across the known body fields so a caller that does not know the content
        shape still gets a real document, and finally to title/heading so a pair is never
        (query, "") when the record carries any text at all.

        Parts are added most-identifying first (body, then description, then steps) and
        the total is capped, so what the model actually reads is the informative head of
        the document rather than whatever happened to fit.
        """
        order = ((text_field,) if text_field else ()) + self._TEXT_FIELDS
        doc_text = ""
        for field in order:
            v = r.get(field)
            if isinstance(v, str) and v.strip():
                doc_text = v
                break

        if not doc_text:
            doc_text = " ".join(
                str(p) for p in (r.get("doc_title"), r.get("section_heading")) if p
            )

        budget = self._document_budget()
        parts: list[str] = []
        remaining = budget

        def _take(text: str, share: int) -> None:
            """Append up to `share` characters, never exceeding what is left overall."""
            nonlocal remaining
            if not text or remaining <= 0:
                return
            piece = text[: min(share, remaining)]
            if piece:
                parts.append(piece)
                remaining -= len(piece)

        # The body gets whatever it needs up to the whole budget; it is the document.
        _take(doc_text, budget)
        # Enrichment only competes for what the body left. Xray test steps live in
        # `description` when steps_text is null, so it stays ahead of steps_text.
        _take((r.get("description") or "").strip(), 400)
        _take((r.get("steps_text") or "").strip(), 300)

        return "\n".join(parts)
