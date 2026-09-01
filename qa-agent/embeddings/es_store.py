"""
Elasticsearch vector store client for the QA Intelligence Engine.

Handles two indexes:
  qa_test_cases  — Xray test case embeddings (synced nightly)
  qa_prd_chunks  — PRD / feature context chunk embeddings (ingested on demand)

Uses Elasticsearch 8.x dense_vector + knn search (HNSW under the hood).

Embedding model: BAAI/bge-m3 (1024 dim, multilingual, asymmetric)
  → Documents (test cases, PRD chunks) stored using embed_document()
  → Queries (PRD content at search time) encoded using embed_query()
  Never mix the two methods — it degrades retrieval quality.
"""
import logging
import os
from datetime import datetime, timezone
from typing import Any

from elasticsearch import Elasticsearch, helpers, NotFoundError

from observability import trace

logger = logging.getLogger(__name__)

# Index names — change these to match your team's naming convention if needed
INDEX_TEST_CASES = "qa_test_cases"
INDEX_PRD_CHUNKS = "qa_prd_chunks"

# Cross-module retrieval. Off by default: with it on, a module-scoped search can return
# tests from any module, which is how you find a test filed under the wrong one — and
# also how an analysis of one module starts recording decisions against another's tests.
# Measure with eval/benchmark.py before enabling; see ADR-014 on opt-in retrieval changes.
CROSS_MODULE_SEARCH = os.environ.get(
    "QA_RETRIEVAL_CROSS_MODULE", "0"
).strip() not in ("", "0", "false", "False")


def _module_scope_clause(module_filter: list[str] | None) -> dict | None:
    """
    The module restriction applied to a test-case search, or None for no restriction.

    Untagged tests are always included. A hard `terms` filter drops every document whose
    `module` field is absent — tests synced before module tagging, or whose module could
    not be derived — so they are invisible to every scoped search with no error anywhere.
    That is the same rule search_similar_prd_chunks already applies to PRD chunks; the two
    indexes disagreed on it, which is exactly the kind of silent divergence that hides.

    With CROSS_MODULE_SEARCH on there is no restriction at all, and the caller is expected
    to rely on ranking rather than filtering to keep other modules out.
    """
    if not module_filter or CROSS_MODULE_SEARCH:
        return None
    return {
        "bool": {
            "should": [
                {"terms": {"module": module_filter}},
                {"bool": {"must_not": {"exists": {"field": "module"}}}},
            ],
            "minimum_should_match": 1,
        }
    }
# Single-document index: records active embedding/reranker model IDs for mismatch detection on startup
INDEX_ENGINE_SETTINGS = "qa_engine_settings"

ENGINE_SETTINGS_MAPPING = {
    "mappings": {
        "properties": {
            "embedding_model": {"type": "keyword"},
            "reranker_model": {"type": "keyword"},
            "embedding_format_version": {"type": "keyword"},
            "updated_at": {"type": "date"},
        }
    },
    "settings": {"number_of_shards": 1, "number_of_replicas": 0},
}

EMBEDDING_DIM = 1024  # BAAI/bge-m3



# ─── Index mappings ────────────────────────────────────────────────────────────

TEST_CASES_MAPPING = {
    "mappings": {
        "properties": {
            "jira_key":     {"type": "keyword"},
            "summary":      {"type": "text"},
            "description":  {"type": "text"},
            "module":       {"type": "keyword"},
            "folder_path":  {"type": "keyword"},
            "labels":         {"type": "keyword"},
            "steps_text":     {"type": "text"},
            "preconditions":  {"type": "text"},
            "content_hash":   {"type": "keyword"},
            "metadata_hash":  {"type": "keyword"},
            "embedding_format_version": {"type": "keyword"},
            "synced_at":    {"type": "date"},
            "embedding": {
                "type":       "dense_vector",
                "dims":       EMBEDDING_DIM,
                "index":      True,
                "similarity": "cosine",
            },
        }
    },
    "settings": {
        "number_of_shards":   1,   # single node setup; increase for cluster
        "number_of_replicas": 0,
    },
}

PRD_CHUNKS_MAPPING = {
    "mappings": {
        "properties": {
            "source_id":       {"type": "keyword"},
            "source_type":     {"type": "keyword"},   # confluence|gitlab_markdown|file_xlsx|…
            "source_version":  {"type": "keyword"},
            "module":          {"type": "keyword"},   # e.g. "Platform", "Billing", "Docs"
            # prd | tech_doc | implementation_plan | test_plan | release_note |
            # meeting_note | other. Classified from the title at ingest. Requirements are
            # a small minority of a whole-space ingest (~4% in one measured corpus), so filtering by
            # type is the difference between searching 167 documents and 3,400.
            # Documents indexed before this field existed have no value; retrieval falls
            # back to a title-derived filter, so both paths work.
            "doc_type":        {"type": "keyword"},
            "doc_title":       {"type": "text",       # human-readable document title
                                "fields": {"keyword": {"type": "keyword"}}},
            "doc_url":         {"type": "keyword"},   # link back to original source
            "section_heading": {"type": "text",
                                "fields": {"keyword": {"type": "keyword"}}},
            "chunk_text":      {"type": "text"},
            "parent_text":     {"type": "text", "index": False},  # stored for context, not searchable
            # table | code | mixed | prose — what kind of content the chunk holds.
            # Set by the chunker at ingest. Lets consumers decide whether layout-aware
            # handling is worth attempting (see read_prd_document's table-header
            # de-duplication) without re-parsing the text. Additive: chunks indexed
            # before this field existed simply have no value, and every consumer
            # treats a missing value as "unknown, handle generically".
            "chunk_type":      {"type": "keyword"},
            "chunk_index":     {"type": "integer"},
            "embedding_format_version": {"type": "keyword"},
            "ingested_at":     {"type": "date"},
            "embedding": {
                "type":       "dense_vector",
                "dims":       EMBEDDING_DIM,
                "index":      True,
                "similarity": "cosine",
            },
        }
    },
    "settings": {
        "number_of_shards":   1,
        "number_of_replicas": 0,
    },
}


class ESStore:
    """
    All Elasticsearch operations for the QA Intelligence Engine.
    Instantiated once at FastAPI startup and reused for all requests.
    """

    def __init__(self) -> None:
        self._retrieval_format_mismatch_warned = False
        self._client = self._build_client()
        self._ensure_indexes()
        self._ensure_engine_metadata()

    def _ensure_engine_metadata(self) -> None:
        """Store/compare embedding + reranker model names so mismatches are visible after model switches."""
        idx = INDEX_ENGINE_SETTINGS
        emb = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-m3")
        rer = os.environ.get("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        fmt = os.environ.get("EMBEDDING_FORMAT_VERSION", "v1")
        if not self._client.indices.exists(index=idx):
            self._client.indices.create(
                index=idx,
                mappings=ENGINE_SETTINGS_MAPPING["mappings"],
                settings=ENGINE_SETTINGS_MAPPING["settings"],
            )
            logger.info("Created Elasticsearch index: %s", idx)
        now = datetime.now(timezone.utc).isoformat()
        doc = {
            "embedding_model": emb,
            "reranker_model": rer,
            "embedding_format_version": fmt,
            "updated_at": now,
        }
        try:
            prev = self._client.get(index=idx, id="active_models")
            src = prev.get("_source") or {}
            if src.get("embedding_model") and src["embedding_model"] != emb:
                logger.error(
                    "Elasticsearch metadata embedding_model=%s but EMBEDDING_MODEL=%s — re-index vectors required",
                    src.get("embedding_model"),
                    emb,
                )
            if src.get("reranker_model") and src["reranker_model"] != rer:
                logger.error(
                    "Elasticsearch metadata reranker_model=%s but RERANKER_MODEL=%s",
                    src.get("reranker_model"),
                    rer,
                )
            if src.get("embedding_format_version") and src["embedding_format_version"] != fmt:
                logger.error(
                    "Elasticsearch metadata embedding_format_version=%s but EMBEDDING_FORMAT_VERSION=%s — re-index or align env",
                    src.get("embedding_format_version"),
                    fmt,
                )
            if (
                src.get("embedding_model") == emb
                and src.get("reranker_model") == rer
                and src.get("embedding_format_version") == fmt
            ):
                return
        except NotFoundError:
            logger.info("No prior qa_engine_settings doc — recording current models")
        self._client.index(index=idx, id="active_models", document=doc, refresh=True)

    def _warn_if_hit_embedding_format_differs(self, docs: list[dict], context: str) -> None:
        """At search time, flag indexed docs whose embedding_format_version disagrees with the live env."""
        if self._retrieval_format_mismatch_warned:
            return
        expected = os.environ.get("EMBEDDING_FORMAT_VERSION", "v1")
        for doc in docs[:40]:
            ver = doc.get("embedding_format_version")
            if ver is None:
                continue
            if ver != expected:
                logger.error(
                    "Retrieval %s: hit has embedding_format_version=%s but EMBEDDING_FORMAT_VERSION=%s "
                    "(jira_key=%s source_id=%s). Re-index after format changes.",
                    context,
                    ver,
                    expected,
                    doc.get("jira_key"),
                    doc.get("source_id"),
                )
                self._retrieval_format_mismatch_warned = True
                return

    def _build_client(self) -> Elasticsearch:
        url = os.environ["ELASTICSEARCH_URL"]
        api_key = os.environ.get("ELASTICSEARCH_API_KEY")
        username = os.environ.get("ELASTICSEARCH_USERNAME")
        password = os.environ.get("ELASTICSEARCH_PASSWORD")

        if api_key:
            client = Elasticsearch(url, api_key=api_key)
        elif username and password:
            client = Elasticsearch(url, basic_auth=(username, password))
        else:
            client = Elasticsearch(url)

        info = client.info()
        logger.info(f"Connected to Elasticsearch {info['version']['number']} at {url}")
        return client

    def _ensure_indexes(self) -> None:
        """
        Create indexes if they don't exist, or patch mapping if they do.
        Safe to run on every startup — new fields are added non-destructively.
        """
        for index, mapping in [
            (INDEX_TEST_CASES, TEST_CASES_MAPPING),
            (INDEX_PRD_CHUNKS, PRD_CHUNKS_MAPPING),
        ]:
            if not self._client.indices.exists(index=index):
                self._client.indices.create(
                    index=index,
                    mappings=mapping["mappings"],
                    settings=mapping["settings"],
                )
                logger.info(f"Created Elasticsearch index: {index}")
            else:
                # Patch any new fields added to the mapping (e.g. description)
                self._client.indices.put_mapping(
                    index=index,
                    properties=mapping["mappings"]["properties"],
                )
                logger.info(f"Elasticsearch index exists (mapping patched): {index}")

    def ping(self) -> bool:
        return self._client.ping()

    def get_test_embedding(self, jira_key: str) -> list[float] | None:
        """
        The stored vector for one test case, or None if absent.

        Lets a caller ask "what else in the corpus is about this test?" without
        re-embedding its text — the stored vector is also the one retrieval used, so
        the comparison is consistent with how the test was found in the first place.
        """
        try:
            resp = self._client.search(
                index=INDEX_TEST_CASES,
                query={"term": {"jira_key": jira_key}},
                source=["embedding"],
                size=1,
            )
        except Exception as exc:
            logger.warning("Could not load embedding for %s: %s", jira_key, exc)
            return None
        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            return None
        return hits[0].get("_source", {}).get("embedding") or None

    def get_available_modules(self, max_terms: int = 100) -> list[str]:
        """
        Distinct module values present in the test case index.

        Used to pre-flight a caller's module filter. Without this a misspelled or
        renamed module produces a run that completes with zero decisions — which
        looks like "the PRD is fully covered" rather than "nothing was searched".

        Returns [] on any failure: validation must never be the reason an analysis
        cannot start.
        """
        try:
            resp = self._client.search(
                index=INDEX_TEST_CASES,
                size=0,
                aggs={
                    "modules": {
                        "terms": {
                            "field": "module",
                            "size": max_terms,
                            "order": {"_key": "asc"},
                        }
                    }
                },
            )
        except Exception as exc:
            logger.warning("Could not list available modules: %s", exc)
            return []
        buckets = resp.get("aggregations", {}).get("modules", {}).get("buckets", [])
        return [b["key"] for b in buckets if b.get("key")]

    # ─── Test case embeddings ──────────────────────────────────────────────────

    def get_existing_hashes(self, folder_path_prefix: str | None = None) -> dict[str, str]:
        """
        Returns {jira_key: content_hash} for test case documents.

        When folder_path_prefix is set (folder-scoped sync), only documents whose
        folder_path matches that prefix (or equals it) are included — so stale
        deletion never targets tests living under other folders.

        When None, scans the whole index (full-project sync).
        """
        if folder_path_prefix and str(folder_path_prefix).strip():
            fp = str(folder_path_prefix).strip()
            es_q = {
                "bool": {
                    "should": [
                        {"prefix": {"folder_path": fp}},
                        {"term": {"folder_path": fp}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        else:
            es_q = {"match_all": {}}
        result = {}
        resp = helpers.scan(
            self._client,
            index=INDEX_TEST_CASES,
            query={"query": es_q, "_source": ["jira_key", "content_hash", "metadata_hash", "folder_path"]},
            scroll="2m",
        )
        for hit in resp:
            src = hit["_source"]
            result[src["jira_key"]] = src.get("content_hash", "")
        return result

    def get_existing_hashes_with_metadata(
        self, folder_path_prefix: str | None = None
    ) -> dict[str, dict[str, str]]:
        """
        Returns {jira_key: {"content_hash": str, "metadata_hash": str}} for test case documents.
        metadata_hash is empty string for legacy docs that predate the field.
        """
        if folder_path_prefix and str(folder_path_prefix).strip():
            fp = str(folder_path_prefix).strip()
            es_q = {
                "bool": {
                    "should": [
                        {"prefix": {"folder_path": fp}},
                        {"term": {"folder_path": fp}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        else:
            es_q = {"match_all": {}}
        result: dict[str, dict[str, str]] = {}
        resp = helpers.scan(
            self._client,
            index=INDEX_TEST_CASES,
            query={"query": es_q, "_source": ["jira_key", "content_hash", "metadata_hash", "folder_path"]},
            scroll="2m",
        )
        for hit in resp:
            src = hit["_source"]
            result[src["jira_key"]] = {
                "content_hash": src.get("content_hash", ""),
                "metadata_hash": src.get("metadata_hash", ""),
            }
        return result

    def get_indexed_prd_source_ids(self) -> set[str]:
        """
        Returns the set of source_ids already indexed in qa_prd_chunks.
        Used by the streaming space ingest to skip already-indexed pages after a restart.
        """
        result = set()
        resp = helpers.scan(
            self._client,
            index=INDEX_PRD_CHUNKS,
            query={"query": {"match_all": {}}, "_source": ["source_id"]},
            scroll="2m",
        )
        for hit in resp:
            result.add(hit["_source"]["source_id"])
        return result

    def get_indexed_source_versions(
        self, source_id_prefix: str | None = None
    ) -> dict[str, str]:
        """
        Returns {source_id: source_version} for indexed PRD chunks.

        This is the incremental-refresh key: a Confluence page bumps its version on every
        edit, so a source whose stored version still matches the live one needs no re-fetch
        and no re-embed. Sources indexed before source_version was populated map to "",
        which callers must treat as "unknown — re-ingest" rather than "current".

        source_id_prefix scopes the scan (e.g. "confluence:") so a site crawl does not
        pay for GitLab/file chunks.
        """
        if source_id_prefix:
            es_q: dict[str, Any] = {"prefix": {"source_id": source_id_prefix}}
        else:
            es_q = {"match_all": {}}

        versions: dict[str, str] = {}
        for hit in helpers.scan(
            self._client,
            index=INDEX_PRD_CHUNKS,
            query={"query": es_q, "_source": ["source_id", "source_version"]},
            scroll="2m",
        ):
            src = hit["_source"]
            sid = src.get("source_id")
            if not sid:
                continue
            ver = src.get("source_version")
            ver = "" if ver is None else str(ver)
            # Chunks of one source share a version; a disagreement means a torn write,
            # so keep the empty/unknown marker and force a re-ingest.
            if sid in versions and versions[sid] != ver:
                versions[sid] = ""
            else:
                versions.setdefault(sid, ver)
        return versions

    def upsert_test_cases_batch(self, records: list[dict]) -> int:
        """
        Bulk upsert test case documents.
        Each record must have: jira_key, summary, embedding (list[float])
        Optional: module, folder_path, labels, steps_text

        Uses jira_key as the document _id so updates overwrite existing docs.
        Returns number of successfully indexed documents.
        """
        if not records:
            return 0

        actions = []
        for r in records:
            actions.append({
                "_op_type": "index",
                "_index":   INDEX_TEST_CASES,
                "_id":      r["jira_key"],        # idempotent: same key = overwrite
                "_source": {
                    "jira_key":     r["jira_key"],
                    "summary":      r.get("summary"),
                    "description":  r.get("description") or "",
                    "module":       r.get("module"),
                    "folder_path":  r.get("folder_path"),
                    "labels":       r.get("labels", []),
                    "steps_text":   r.get("steps_text"),
                    "preconditions":  r.get("preconditions") or "",
                    "content_hash": r.get("content_hash") or "",
                    "metadata_hash": r.get("metadata_hash") or "",
                    "embedding":    r["embedding"],
                    "embedding_format_version": os.environ.get("EMBEDDING_FORMAT_VERSION", "v1"),
                    "synced_at":    datetime.now(timezone.utc).isoformat(),
                },
            })

        success, errors = helpers.bulk(
            self._client,
            actions,
            chunk_size=200,
            raise_on_error=False,
        )
        if errors:
            logger.warning(f"Bulk upsert had {len(errors)} errors: {errors[:3]}")

        # What actually landed. Field lengths matter more than counts: a doc with a summary
        # and nothing else embeds far weaker than one carrying steps, and that difference is
        # invisible from "indexed 8 documents".
        for r in records:
            trace.event("index", index=INDEX_TEST_CASES, jira_key=r.get("jira_key"),
                        module=r.get("module"), folder_path=r.get("folder_path"),
                        labels=r.get("labels"),
                        summary=r.get("summary"),
                        description_chars=len(r.get("description") or ""),
                        steps_chars=len(r.get("steps_text") or ""),
                        preconditions_chars=len(r.get("preconditions") or ""),
                        steps_text=r.get("steps_text"),
                        embedding_dim=len(r.get("embedding") or []))
        n_no_steps = sum(1 for r in records if not (r.get("steps_text") or "").strip())
        logger.info(
            "  indexed %s/%s test docs into %s | no_steps=%s avg_steps_chars=%s",
            success, len(records), INDEX_TEST_CASES, n_no_steps,
            round(sum(len(r.get("steps_text") or "") for r in records) / max(1, len(records))),
        )
        return success

    def delete_stale_tests(self, stale_keys: list[str]) -> int:
        """
        Remove specific test case documents by jira_key.
        Called with explicit list of keys confirmed to be stale.
        Returns approximate deleted count.
        """
        if not stale_keys:
            return 0
        resp = self._client.delete_by_query(
            index=INDEX_TEST_CASES,
            query={"terms": {"jira_key": stale_keys}},
            refresh=True,
        )
        deleted = resp.get("deleted", 0)
        if deleted:
            logger.info(f"Deleted {deleted} stale test case documents from Elasticsearch")
        return deleted

    # ─── Similarity search ─────────────────────────────────────────────────────

    def search_similar_tests(
        self,
        query_embedding: list[float],
        top_k: int = 200,
        module_filter: list[str] | None = None,
        min_score: float = 0.5,
    ) -> list[dict]:
        """
        KNN similarity search over test case embeddings.

        Returns list of dicts with keys:
            jira_key, summary, module, folder_path, labels, steps_text, score

        score is cosine similarity (0–1, higher = more relevant).
        module_filter pre-filters by module before the KNN search (fast, uses inverted index).
        min_score filters out low-relevance results post-search.
        """
        num_candidates = max(100, top_k * 10)
        knn_query: dict[str, Any] = {
            "field":          "embedding",
            "query_vector":   query_embedding,
            "k":              top_k,
            "num_candidates": num_candidates,
        }
        module_scope = _module_scope_clause(module_filter)
        if module_scope is not None:
            knn_query["filter"] = [module_scope]

        resp = self._client.search(
            index=INDEX_TEST_CASES,
            knn=knn_query,
            source=[
                "jira_key",
                "summary",
                "description",
                "module",
                "folder_path",
                "labels",
                "steps_text",
                "embedding_format_version",
            ],
            size=top_k,
        )

        results = []
        for hit in resp["hits"]["hits"]:
            score = hit["_score"]
            if score < min_score:
                continue
            doc = hit["_source"]
            doc["score"] = score
            results.append(doc)

        self._warn_if_hit_embedding_format_differs(results, "search_similar_tests")
        return results

    def estimate_pool_size(self, module_filter: list[str] | None) -> int:
        """
        Adaptive pool size: ~20% of relevant tests (whole index if no module filter).
        Bounded between 300 (floor for recall) and 2000 (cap for performance).
        Called once per validate/preview run to avoid repeated count queries.
        """
        if not module_filter:
            resp = self._client.count(index=INDEX_TEST_CASES, query={"match_all": {}})
            total = resp["count"]
            return max(300, min(int(total * 0.20), 2000))
        # Same scope rule as the searches this sizes the pool for, or the estimate
        # describes a different corpus from the one actually queried.
        module_scope = _module_scope_clause(module_filter)
        if module_scope is None:
            resp = self._client.count(index=INDEX_TEST_CASES, query={"match_all": {}})
            return max(300, min(int(resp["count"] * 0.20), 2000))
        resp = self._client.count(index=INDEX_TEST_CASES, query=module_scope)
        module_size = resp["count"]
        return max(300, min(int(module_size * 0.20), 2000))

    def search_hybrid(
        self,
        query_embedding: list[float],
        keyword_query: str,
        top_k: int = 200,
        module_filter: list[str] | None = None,
    ) -> list[dict]:
        """
        Hybrid search: combines KNN vector similarity with BM25 keyword relevance
        using manual Reciprocal Rank Fusion (RRF).

        Runs KNN and BM25 as separate queries then merges by rank position using
        RRF formula: score = Σ 1/(rank + k), k=60. This avoids the BM25-dominates-
        cosine problem of additive score combination, and works on the basic ES license
        (unlike the native `retriever.rrf` which requires Platinum+).
        """
        num_candidates = max(100, top_k * 10)
        fetch_size = max(100, top_k * 2)  # fetch more than top_k so both lists have good coverage
        module_scope = _module_scope_clause(module_filter)
        _source_fields = [
            "jira_key",
            "summary",
            "description",
            "module",
            "folder_path",
            "labels",
            "steps_text",
            "preconditions",
            "embedding_format_version",
        ]
        RANK_CONSTANT = 60

        # ── KNN search ──
        knn_query: dict[str, Any] = {
            "field":          "embedding",
            "query_vector":   query_embedding,
            "k":              fetch_size,
            "num_candidates": num_candidates,
        }
        if module_scope is not None:
            knn_query["filter"] = module_scope

        knn_resp = self._client.search(
            index=INDEX_TEST_CASES,
            knn=knn_query,
            source=_source_fields,
            size=fetch_size,
        )

        # ── BM25 search ──
        text_query: dict[str, Any] = {
            "multi_match": {
                "query":  keyword_query,
                "fields": ["summary^3", "description^2", "steps_text", "preconditions"],
            }
        }
        if module_scope is not None:
            text_query = {
                "bool": {
                    "must":   text_query,
                    "filter": [module_scope],
                }
            }

        bm25_resp = self._client.search(
            index=INDEX_TEST_CASES,
            query=text_query,
            source=_source_fields,
            size=fetch_size,
        )

        # ── Manual RRF merge ──
        # Map jira_key → (source_doc, rrf_score)
        scores: dict[str, float] = {}
        docs: dict[str, dict] = {}

        for rank, hit in enumerate(knn_resp["hits"]["hits"]):
            key = hit["_source"]["jira_key"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + RANK_CONSTANT)
            docs[key] = hit["_source"]

        for rank, hit in enumerate(bm25_resp["hits"]["hits"]):
            key = hit["_source"]["jira_key"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + RANK_CONSTANT)
            docs[key] = hit["_source"]

        # Sort by combined RRF score descending, return top_k
        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)[:top_k]
        merged = [
            {**docs[k], "score": scores[k]}
            for k in sorted_keys
        ]
        self._warn_if_hit_embedding_format_differs(merged, "search_hybrid")
        return merged

    def get_all_test_metadata(self) -> list[dict]:
        """
        Returns {jira_key, summary, module, labels} for ALL test cases.
        Used by Gap Agent for coverage check without loading embeddings.
        """
        results = []
        resp = helpers.scan(
            self._client,
            index=INDEX_TEST_CASES,
            query={"query": {"match_all": {}},
                   "_source": ["jira_key", "summary", "module", "labels"]},
        )
        for hit in resp:
            results.append(hit["_source"])
        return results

    # ─── PRD chunk embeddings ──────────────────────────────────────────────────

    def upsert_prd_chunks(self, chunks: list[dict]) -> int:
        """
        Re-index PRD chunks for a given source_id with no availability gap.

        Strategy: index new chunks first, then delete stale ones using an
        ingested_at cutoff. During the brief overlap both old and new chunks
        exist; searches may return a mix, but there is never a gap where
        zero chunks are indexed (the old delete-first approach).

        Each chunk dict must have: source_id, source_type, chunk_text, chunk_index, embedding
        Optional: source_version, section_heading
        """
        if not chunks:
            return 0

        source_id = chunks[0]["source_id"]

        # Verify all chunks share the same source_id
        mismatched = [c["source_id"] for c in chunks if c["source_id"] != source_id]
        if mismatched:
            raise ValueError(
                f"upsert_prd_chunks expects all chunks to share one source_id, "
                f"got {source_id!r} and {mismatched[0]!r}"
            )

        # Snapshot the cutoff before writing so any older chunk is a stale pre-image
        cutoff = datetime.now(timezone.utc)

        actions = [
            {
                "_op_type": "index",
                "_index":   INDEX_PRD_CHUNKS,
                "_source": {
                    "source_id":       c["source_id"],
                    "source_type":     c["source_type"],
                    "source_version":  c.get("source_version"),
                    "module":          c.get("module"),
                    "doc_title":       c.get("doc_title"),
                    "doc_url":         c.get("doc_url"),
                    "section_heading": c.get("section_heading"),
                    "chunk_text":      c["chunk_text"],
                    "parent_text":     c.get("parent_text"),
                    "doc_type":        c.get("doc_type"),
                    "chunk_type":      c.get("chunk_type"),
                    "chunk_index":     c["chunk_index"],
                    "embedding":       c["embedding"],
                    "embedding_format_version": os.environ.get("EMBEDDING_FORMAT_VERSION", "v1"),
                    "ingested_at":     datetime.now(timezone.utc).isoformat(),
                },
            }
            for c in chunks
        ]

        # Phase 1: write new chunks (searchable immediately after refresh)
        success, errors = helpers.bulk(
            self._client, actions, chunk_size=200, raise_on_error=False, refresh=True
        )
        if errors:
            logger.warning(f"PRD chunk bulk index had {len(errors)} errors")
        _lens = [len(c.get("chunk_text") or "") for c in chunks]
        logger.info(
            "  indexed %s/%s PRD chunks for %s | chars min/median/max=%s/%s/%s total=%s",
            success, len(chunks), source_id,
            min(_lens) if _lens else 0,
            sorted(_lens)[len(_lens)//2] if _lens else 0,
            max(_lens) if _lens else 0, sum(_lens),
        )
        for c in chunks:
            trace.event("index", index=INDEX_PRD_CHUNKS, source_id=c.get("source_id"),
                        chunk_index=c.get("chunk_index"),
                        section_heading=c.get("section_heading"),
                        doc_title=c.get("doc_title"), doc_url=c.get("doc_url"),
                        module=c.get("module"), source_version=c.get("source_version"),
                        chunk_text=c.get("chunk_text"),
                        embedding_dim=len(c.get("embedding") or []))

        # Phase 2: delete stale pre-image chunks (ingested before this batch started).
        #
        # Guarded. Previously this ran unconditionally, so a bulk write that failed entirely
        # (ES under memory pressure, a mapping conflict, a 429) logged a warning and then
        # deleted every existing chunk for the source — leaving it with nothing indexed and
        # no error raised. The replacement must be known-good before the original is removed.
        if success == 0 and chunks:
            logger.error(
                "Refusing to delete pre-existing chunks for %s: the replacement bulk write "
                "indexed 0 of %s documents (%s errors). The previous version is intact but "
                "the index is now stale for this source — re-run the ingest.",
                source_id, len(chunks), len(errors),
            )
            return 0
        if errors:
            logger.warning(
                "Keeping pre-existing chunks for %s: %s of %s documents failed to index, so "
                "the new set is incomplete. Duplicate/overlapping chunks may be returned "
                "until a clean ingest succeeds.",
                source_id, len(errors), len(chunks),
            )
            return success

        self._client.delete_by_query(
            index=INDEX_PRD_CHUNKS,
            query={
                "bool": {
                    "must": {"term": {"source_id": source_id}},
                    "filter": {"range": {"ingested_at": {"lt": cutoff.isoformat()}}},
                }
            },
            refresh=True,
        )

        return success

    def delete_prd_source(self, source_id: str) -> int:
        """
        Delete all PRD chunks for a source_id. Returns number of deleted documents.
        """
        resp = self._client.delete_by_query(
            index=INDEX_PRD_CHUNKS,
            query={"term": {"source_id": source_id}},
            refresh=True,
        )
        deleted = resp.get("deleted", 0)
        logger.info(f"Deleted {deleted} PRD chunks for source_id={source_id!r}")
        return deleted

    def search_similar_prd_chunks(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        source_id: str | None = None,
        module_filter: list[str] | None = None,
        title_contains: str | None = None,
        doc_types: list[str] | None = None,
        exclude_doc_types: list[str] | None = None,
    ) -> list[dict]:
        """
        Find PRD chunks most similar to a query vector.
        Optionally restrict to a specific source document or module(s).

        module_filter: e.g. ["Platform"] — only returns chunks tagged with that module.
                       Chunks ingested without a module tag are always included (null match).

        title_contains: substring the doc_title must contain, e.g. "PRD". Lets a broad
                        corpus be narrowed at QUERY time instead of re-ingesting — useful
                        when a whole space was indexed and tech docs, test plans and
                        release notes are competing with the actual requirements.
        """
        knn_query: dict[str, Any] = {
            "field":          "embedding",
            "query_vector":   query_embedding,
            "k":              top_k,
            "num_candidates": max(100, top_k * 10),
        }
        filters = []
        if source_id:
            filters.append({"term": {"source_id": source_id}})
        if title_contains:
            # match_phrase on the analysed text field: case-insensitive and tolerant of
            # "PRD -", "PRD:", "- PRD" without needing a wildcard over a keyword field.
            filters.append({"match_phrase": {"doc_title": title_contains}})
        if doc_types or exclude_doc_types:
            # Derived from doc_title, so an existing index can be scoped by document
            # type without re-indexing. Requirements are only ~4% of a whole-space
            # ingest, so this is the difference between searching 167 documents and
            # searching 3,400.
            from ingestion.doc_classify import title_filter as _dt_filter
            dt = _dt_filter(include=doc_types, exclude=exclude_doc_types)
            if dt:
                filters.append(dt)
        if module_filter:
            # Match chunks tagged with one of the requested modules OR untagged (no module set)
            filters.append({"bool": {"should": [
                {"terms": {"module": module_filter}},
                {"bool": {"must_not": {"exists": {"field": "module"}}}},
            ]}})
        if filters:
            knn_query["filter"] = filters

        resp = self._client.search(
            index=INDEX_PRD_CHUNKS,
            knn=knn_query,
            source=[
                "source_id",
                "source_type",
                "module",
                "doc_title",
                "doc_url",
                "section_heading",
                "chunk_text",
                "chunk_type",
                "chunk_index",
                "embedding_format_version",
            ],
            size=top_k,
        )

        pr_results = [
            {**hit["_source"], "score": hit["_score"]}
            for hit in resp["hits"]["hits"]
        ]
        self._warn_if_hit_embedding_format_differs(pr_results, "search_similar_prd_chunks")
        return pr_results
