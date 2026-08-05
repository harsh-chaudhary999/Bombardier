"""
Local embedding client using BAAI/bge-m3 via sentence-transformers.

Key properties:
- 0.57B parameter model (XLM-RoBERTa-large backbone), 1024 dimensions
- Best-in-class multilingual support (100+ languages, Hindi explicitly trained)
- Instruction-aware asymmetric encoding:
    Queries  → encode() with retrieval instruction prefix
    Documents→ encode() with no instruction (plain passage encoding)
  → Do NOT mix the two — it degrades retrieval quality
- Max 8192 tokens per input
- MTEB retrieval score: ~58 nDCG@10 (top tier for <1B param models)

USAGE PATTERN:
  Test cases being indexed → embed_document()
  PRD chunks being indexed → embed_document()
  PRD query at search time → embed_query()

CRITICAL: Use the SAME model for both indexing and querying.
Switching models later requires re-indexing everything in Elasticsearch.
"""
import logging
import os
from collections import OrderedDict

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

MODEL_NAME    = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
EMBEDDING_DIM = 1024

# bge-m3 query instruction — prepended to all queries for asymmetric retrieval.
# Documents are encoded WITHOUT this prefix.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "
_QUERY_CACHE_SIZE = 512


class EmbedClient:
    """
    Wraps BAAI/bge-m3 for the QA Intelligence Engine.
    Loaded once at FastAPI startup and reused for all embedding requests.

    Asymmetric encoding:
      Documents (test cases, PRD chunks being stored) → embed_document()
      Queries   (PRD content used to SEARCH for tests) → embed_query()
    """

    def __init__(self) -> None:
        logger.info(f"Loading embedding model: {MODEL_NAME}")
        self._model = SentenceTransformer(MODEL_NAME)
        dim = self._model.get_sentence_embedding_dimension()
        logger.info(f"Embedding model loaded. Dimension: {dim}")

        # LRU cache for query embeddings (single path for embed_query + embed_queries batch warm-up)
        self._query_vec_cache: OrderedDict[str, tuple[float, ...]] = OrderedDict()

    def _cache_get_query(self, text: str) -> list[float] | None:
        if text not in self._query_vec_cache:
            return None
        self._query_vec_cache.move_to_end(text)
        return list(self._query_vec_cache[text])

    def _cache_put_query(self, text: str, vec: list[float]) -> None:
        self._query_vec_cache[text] = tuple(vec)
        self._query_vec_cache.move_to_end(text)
        while len(self._query_vec_cache) > _QUERY_CACHE_SIZE:
            self._query_vec_cache.popitem(last=False)

    # ─── Document encoding (for indexing) ─────────────────────────────────────

    def embed_document(self, text: str) -> list[float]:
        """
        Embed a single document for storage in Elasticsearch.
        Use for: test cases, PRD chunks, knowledge base docs.
        No instruction prefix — plain passage encoding.
        """
        if not text or not text.strip():
            logger.warning("embed_document called with empty text, returning zero vector")
            return [0.0] * EMBEDDING_DIM
        vec = self._model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed a batch of documents for storage in Elasticsearch.
        batch_size=32 is safe for CPU; reduce to 16 if you hit memory limits.
        """
        if not texts:
            return []
        vecs = self._model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 50,
        )
        return [v.tolist() for v in vecs]

    # ─── Query encoding (for searching) ───────────────────────────────────────

    def embed_one(self, text: str) -> list[float]:
        """
        Embed a single query string for similarity search.
        Alias for embed_query() — used by /health endpoint.
        """
        return self.embed_query(text)

    def embed_query(self, text: str) -> list[float]:
        """
        Embed a query for similarity search against stored documents.
        Prepends bge-m3's retrieval instruction for asymmetric encoding.

        Results are LRU-cached (512 entries) — safe for repeated queries
        across validate/preview runs.
        """
        if not text or not text.strip():
            logger.warning("embed_query called with empty text, returning zero vector")
            return [0.0] * EMBEDDING_DIM
        hit = self._cache_get_query(text)
        if hit is not None:
            return hit
        instructed = _QUERY_INSTRUCTION + text
        vec = self._model.encode(instructed, normalize_embeddings=True).tolist()
        self._cache_put_query(text, vec)
        return vec

    def embed_queries(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """
        Embed a batch of query strings with retrieval instruction prefix.
        Uses batch encoding for uncached texts; fills the same LRU as embed_query().
        """
        if not texts:
            return []
        final: list[list[float] | None] = [None] * len(texts)
        pending_texts: list[str] = []
        pending_positions: list[int] = []

        for i, t in enumerate(texts):
            if not t or not t.strip():
                logger.warning("embed_queries got empty text at index %s, returning zero vector", i)
                final[i] = [0.0] * EMBEDDING_DIM
                continue
            hit = self._cache_get_query(t)
            if hit is not None:
                final[i] = hit
            else:
                pending_positions.append(i)
                pending_texts.append(t)

        if pending_texts:
            instructed = [_QUERY_INSTRUCTION + t for t in pending_texts]
            vecs = self._model.encode(
                instructed,
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=len(pending_texts) > 50,
            )
            for pos, t, v in zip(pending_positions, pending_texts, vecs):
                vec = v.tolist()
                self._cache_put_query(t, vec)
                final[pos] = vec

        return [x if x is not None else [0.0] * EMBEDDING_DIM for x in final]

    # ─── Text formatting helpers ───────────────────────────────────────────────

    @staticmethod
    def format_test_case(
        summary: str,
        module: str | None = None,
        labels: list[str] | None = None,
        description: str | None = None,
        steps_text: str | None = None,
    ) -> str:
        """
        Format a test case into a single string before calling embed_document().
        Consistent format: changing this requires re-indexing all test cases.

        Field order: Summary → Module → Labels → Description → Steps
        """
        parts = [f"Summary: {summary}"]
        if module:
            parts.append(f"Module: {module}")
        if labels:
            parts.append(f"Labels: {', '.join(labels)}")
        if description:
            parts.append(f"Description: {description}")
        if steps_text:
            parts.append(f"Steps:\n{steps_text}")
        return "\n".join(parts)

    @staticmethod
    def format_prd_chunk(section_heading: str | None, chunk_text: str) -> str:
        """
        Format a PRD chunk before embedding.

        Prefix with section heading for context.
        Truncated to ~1000 tokens (4000 chars) to keep CPU embedding fast
        while capturing the full content of 800-token chunks.
        """
        MAX_CHARS = 4000
        if section_heading:
            text = f"Section: {section_heading}\n\n{chunk_text}"
        else:
            text = chunk_text
        return text[:MAX_CHARS]
