"""
Grounded question answering over the indexed corpus.

Bombardier's other LLM path (agents/analysis_agent.py) exists to emit structured coverage
decisions — keep/update/deprecate/create — for human review. It does not answer questions.
`/search/prd` returns raw chunks. This module fills the gap between the two: ask a question
in English, get a readable answer that cites the documents it came from.

Design constraints, each learned the hard way:

  * Citations are mandatory. Without them a fluent invention is indistinguishable from a
    grounded answer, and the reader has no way to check. Every claim carries [n].

  * "Not documented" is a first-class answer. Real corpora have holes — e.g. the concept
    PROVISIONAL is defined in tech docs and appears in no PRD at all. A model pressured
    to always answer will invent a plausible source. The prompt makes abstaining explicit.

  * The retrieved context is returned with the answer, so a wrong answer can be attributed
    to retrieval (the fact was never supplied) or to the model (it was supplied and ignored)
    without a second round trip.

  * No outside knowledge. The model is told to answer only from the supplied context, which
    is what makes the citations meaningful.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.analysis_agent import (
    _accumulate_usage,
    _build_llm,
    _finalize_usage,
    _new_usage,
    message_text,
)
from agents.model_tiers import resolve_tier
from embeddings.embed_client import EmbedClient
from embeddings.es_store import ESStore
from embeddings.rank_filter import relative_cut
from observability import trace

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about a product's requirements and test suite using ONLY the numbered context passages supplied to you.

## Rules
1. Use only the supplied context. Do not use outside knowledge, and do not guess.
2. Cite every factual claim with the passage number in square brackets, e.g. [1] or [2][3].
   A sentence carrying a fact with no citation is a mistake.
3. If the context does not contain the answer, say so plainly:
   start with "Not documented in the indexed corpus." then describe the closest related
   material that IS present and cite it. Do not pad this out into a speculative answer.
4. If passages disagree, say so and cite both rather than silently picking one.
5. Quote exact identifiers, status names, numbers, limits and field names verbatim —
   these are what the reader will act on. Do not paraphrase an identifier like `EXAMPLE_STATUS` into
   "example status", and do not round "1-month" to "about a month".
6. Be direct. Lead with the answer, then the supporting detail. No preamble, no
   restatement of the question, no closing summary.
7. Note the document type when it matters for trust: a requirement stated in a PRD and one
   stated in an implementation plan or test plan are not equally authoritative."""


def _format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks as numbered passages the model can cite by index."""
    blocks = []
    for i, c in enumerate(chunks, 1):
        title = c.get("doc_title") or c.get("source_id") or "untitled"
        heading = (c.get("section_heading") or "").strip()
        kind = c.get("_kind", "prd")
        label = f"[{i}] {title}"
        if heading and heading != title:
            label += f" › {heading}"
        if kind == "test":
            label += f"  (Xray test {c.get('jira_key')})"
        body = c.get("chunk_text") or c.get("_text") or ""
        blocks.append(f"{label}\n{body}")
    return "\n\n---\n\n".join(blocks)


def _citations(chunks: list[dict]) -> list[dict]:
    out = []
    for i, c in enumerate(chunks, 1):
        out.append({
            "n": i,
            "kind": c.get("_kind", "prd"),
            "source_id": c.get("source_id") or c.get("jira_key"),
            "doc_title": c.get("doc_title") or c.get("summary"),
            "section_heading": c.get("section_heading"),
            "score": round(float(c.get("score") or 0), 4),
            "rerank_score": c.get("rerank_score"),
            "url": c.get("doc_url"),
        })
    return out


_CITE_RE = re.compile(r"\[(\d+)\]")


def answer_question(
    question: str,
    embed_client: EmbedClient,
    es_store: ESStore,
    *,
    provider: str,
    model: str,
    top_k: int = 8,
    module: list[str] | None = None,
    source_id: str | None = None,
    title_contains: str | None = None,
    doc_types: list[str] | None = None,
    exclude_doc_types: list[str] | None = None,
    include_tests: bool = False,
    min_score: float = 0.0,
    reranker=None,
    tier: str = "fast",
) -> dict[str, Any]:
    """
    Retrieve, then synthesise a cited answer. Returns the answer plus the exact context
    used, so the caller can verify it rather than trust it.
    """
    started = time.time()
    query_vec = embed_client.embed_query(question)

    # Over-retrieve when a reranker is available; it re-scores and we keep the best top_k.
    fetch_k = min(top_k * 4, 50) if reranker else top_k
    chunks = es_store.search_similar_prd_chunks(
        query_embedding=query_vec,
        top_k=fetch_k,
        source_id=source_id,
        module_filter=module,
        title_contains=title_contains,
        doc_types=doc_types,
        exclude_doc_types=exclude_doc_types,
    )
    for c in chunks:
        c["_kind"] = "prd"

    if include_tests:
        # Opt-in: lets "which tests cover X?" be answerable. Off by default because a
        # corpus containing test plans can otherwise answer a coverage question with a
        # test plan, which is circular.
        tests = es_store.search_similar_tests(
            query_embedding=query_vec, top_k=fetch_k, module_filter=module, min_score=0.0
        )
        for t in tests:
            t["_kind"] = "test"
            t["_text"] = "\n".join(
                p for p in (t.get("summary"), t.get("description"), t.get("steps_text")) if p
            )
        chunks = chunks + tests

    if reranker and chunks:
        for c in chunks:
            c.setdefault("chunk_text", c.get("_text", ""))
        chunks = reranker.rerank(question, chunks, top_k=top_k)
    else:
        chunks = sorted(chunks, key=lambda c: c.get("score") or 0, reverse=True)[:top_k]

    if min_score:
        chunks = [c for c in chunks if (c.get("score") or 0) >= min_score]

    # Trim at the knee rather than by absolute score. On this corpus rank1..rank15 span
    # only ~5.5%, so any fixed threshold keeps everything or nothing; the leader is still
    # ~6 sigma above the tail, which relative_cut can act on. `separation` is returned so
    # a confident trim is distinguishable from an arbitrary one.
    score_key = "rerank_score" if (reranker and chunks and "rerank_score" in chunks[0]) else "score"
    chunks, rank_diag = relative_cut(chunks, key=score_key, min_keep=min(3, len(chunks)))

    if not chunks:
        # Same response shape as the success path — a client reading .retrieval or .tier
        # must not have to special-case "no results".
        return {
            "question": question,
            "answer": "Not documented in the indexed corpus — retrieval returned no passages "
                      "for this question. Check the filters, or ingest the relevant source.",
            "grounded": False,
            "abstained": True,
            "citations": [],
            "cited_passages": [],
            "context_used": [],
            "provider": provider,
            "model": model,
            "tier": tier,
            "retrieval": rank_diag,
            "elapsed_s": round(time.time() - started, 2),
            "token_usage": _finalize_usage(_new_usage(), provider, model),
        }

    context = _format_context(chunks)
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=(
            f"## Context passages\n\n{context}\n\n"
            f"## Question\n{question}\n\n"
            "Answer using only the passages above, citing each claim with [n]."
        )),
    ]

    # Synthesis is the judgement step here, so callers may escalate it.
    eff_provider, eff_model = resolve_tier(tier, provider, model)
    usage = _new_usage()
    llm = _build_llm(eff_provider, eff_model)
    response = llm.invoke(messages)
    _accumulate_usage(usage, response)
    # Strips <think> blocks: a reasoning model would otherwise open with paragraphs of
    # deliberation, and bracket numbers inside that monologue would be misread as citations.
    answer = message_text(response)

    if not answer:
        # message_text() returns "" when the response was pure reasoning, pure tool calls,
        # or truncated mid-thought. Say so rather than returning an empty answer string.
        logger.warning("ask: model returned no usable text for %r", question[:120])
        answer = (
            "The model returned no usable answer text — its response was empty, or consisted "
            "only of reasoning that was truncated. Retry, or raise QA_AGENT_MAX_OUTPUT_TOKENS "
            "if a reasoning model is running out of budget before it writes a conclusion."
        )

    cited = sorted({int(n) for n in _CITE_RE.findall(answer) if 1 <= int(n) <= len(chunks)})
    abstained = answer.lower().startswith("not documented")
    # Grounded means: it cited something, and it did not abstain. An answer with no
    # citations is exactly the case the reader cannot verify, so flag it rather than
    # presenting it as equivalent.
    grounded = bool(cited) and not abstained
    if not cited and not abstained:
        logger.warning(
            "ask: answer contains no citations — treating as ungrounded. question=%r", question[:120]
        )

    token_usage = _finalize_usage(usage, eff_provider, eff_model)
    elapsed = round(time.time() - started, 2)
    logger.info(
        "[layer=ask] %r -> grounded=%s cited=%s of %s passages in %ss | tokens in=%s out=%s cost=%s",
        question[:80], grounded, cited, len(chunks), elapsed,
        token_usage["input_tokens"], token_usage["output_tokens"],
        token_usage.get("estimated_cost_usd"),
    )
    trace.event("ask", question=question, provider=provider, model=model,
                grounded=grounded, cited=cited, answer=answer,
                context=[{k: v for k, v in c.items() if k != "embedding"} for c in chunks])

    return {
        "question": question,
        "answer": answer,
        "grounded": grounded,
        "abstained": abstained,
        # Only what the model actually cited. Empty when it cited nothing — returning every
        # passage here would dress an unverifiable answer up as a sourced one, which is the
        # exact confusion `grounded: false` exists to prevent. The full set is in
        # `context_used` for auditing.
        "citations": [c for c in _citations(chunks) if c["n"] in cited],
        "cited_passages": cited,
        "context_used": [
            {
                "n": i,
                "kind": c.get("_kind"),
                "source_id": c.get("source_id") or c.get("jira_key"),
                "doc_title": c.get("doc_title") or c.get("summary"),
                "section_heading": c.get("section_heading"),
                "score": round(float(c.get("score") or 0), 4),
                "rerank_score": c.get("rerank_score"),
                "url": c.get("doc_url"),
                "text": (c.get("chunk_text") or c.get("_text") or "")[:1500],
            }
            for i, c in enumerate(chunks, 1)
        ],
        "provider": eff_provider,
        "model": eff_model,
        "tier": tier,
        "retrieval": rank_diag,
        "elapsed_s": elapsed,
        "token_usage": token_usage,
    }
