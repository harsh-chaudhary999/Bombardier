"""
LLM-as-judge evaluation for analysis decision quality.

After Phase 3 (analysis), this module samples decisions and asks a separate LLM
to grade each one for correctness, reasoning quality, and completeness.

Usage:
    results = await evaluate_decisions(run_id, pg_store, es_store, provider, model)
    # returns {"run_id": ..., "scores": [...], "avg_score": 0.85, ...}
"""
import asyncio
import json
import logging
import os
import random
from typing import Any

logger = logging.getLogger(__name__)

JUDGE_PROMPT = """You are a senior QA engineer evaluating the quality of automated test coverage decisions.

You will be given:
1. A PRD section (the requirement being analysed)
2. A test case (if applicable)
3. The decision made (KEEP / UPDATE / DEPRECATE / CREATE)
4. The reason given for the decision

Grade the decision on these criteria (1-5 scale each):

**correctness**: Is the action appropriate given the PRD section and test case?
  5 = Clearly correct action with no ambiguity
  3 = Reasonable but debatable
  1 = Clearly wrong action

**reasoning**: Is the explanation clear, specific, and well-justified?
  5 = Specific references to PRD content and test details
  3 = Generic but acceptable reasoning
  1 = Vague, circular, or missing reasoning

**completeness**: Does the decision capture all relevant aspects?
  5 = Covers edge cases, prerequisites, and related scenarios
  3 = Covers the main path only
  1 = Misses obvious aspects

Respond with ONLY a JSON object (no markdown, no explanation):
{"correctness": N, "reasoning": N, "completeness": N, "comment": "brief note"}
"""


async def evaluate_decisions(
    run_id: str,
    pg_store: "PGStore",
    es_store: "ESStore",
    provider: str = "anthropic",
    model: str = "claude-sonnet-4-6",
    sample_size: int = 20,
) -> dict[str, Any]:
    """
    Evaluate a sample of decisions from an analysis run using LLM-as-judge.

    Returns:
        {
            "run_id": str,
            "sample_size": int,
            "scores": [{"decision_id": int, "action": str, "correctness": int, ...}],
            "avg_correctness": float,
            "avg_reasoning": float,
            "avg_completeness": float,
            "avg_score": float,  # overall average across all criteria
        }
    """
    from embeddings.pg_store import PGStore
    from embeddings.es_store import ESStore

    decisions = pg_store.get_pending_decisions(run_id=run_id)
    if not decisions:
        return {"run_id": run_id, "error": "No decisions found for this run"}

    # Random sample so middle/end PRD sections are represented (created_at order is top-heavy).
    actionable = [d for d in decisions if d["action"] in ("update", "create", "deprecate")]
    keeps = [d for d in decisions if d["action"] == "keep"]

    n_act = min(sample_size, len(actionable))
    sampled = random.sample(actionable, n_act) if actionable and n_act > 0 else []
    remaining = sample_size - len(sampled)
    if remaining > 0 and keeps:
        n_keep = min(remaining, len(keeps))
        sampled.extend(random.sample(keeps, n_keep))

    # Build LLM
    llm = _build_judge_llm(provider, model)

    loop = asyncio.get_running_loop()
    scores = []

    for decision in sampled:
        try:
            # Fetch context for this decision
            context = _build_judge_context(decision, es_store)

            grade = await loop.run_in_executor(
                None,
                lambda ctx=context: _grade_decision(llm, ctx),
            )

            row = {
                "decision_id": decision["id"],
                "jira_key": decision.get("jira_key"),
                "action": decision["action"],
            }
            row.update(grade)
            scores.append(row)
        except Exception as e:
            logger.warning(f"Judge failed for decision {decision['id']}: {e}")
            scores.append({
                "decision_id": decision["id"],
                "action": decision["action"],
                "error": str(e),
            })

    # Compute averages — only rows with numeric judge scores (not parse/infrastructure failures).
    valid_scores = [s for s in scores if isinstance(s.get("correctness"), (int, float))]
    parse_failures = sum(1 for s in scores if s.get("parse_failed"))
    avg_correctness = _avg(valid_scores, "correctness")
    avg_reasoning = _avg(valid_scores, "reasoning")
    avg_completeness = _avg(valid_scores, "completeness")
    avg_score = (avg_correctness + avg_reasoning + avg_completeness) / 3 if valid_scores else 0

    return {
        "run_id": run_id,
        "total_decisions": len(decisions),
        "sample_size": len(sampled),
        "evaluated": len(valid_scores),
        "parse_failures": parse_failures,
        "scores": scores,
        "avg_correctness": round(avg_correctness, 2),
        "avg_reasoning": round(avg_reasoning, 2),
        "avg_completeness": round(avg_completeness, 2),
        "avg_score": round(avg_score, 2),
    }


def _build_judge_llm(provider: str, model: str):
    """Build LLM for judging — same factory as analysis agent."""
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, max_tokens=512)
    elif provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=model,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            max_tokens=512,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, max_tokens=512)
    elif provider == "ollama":
        # Judging needs no tool calling, so any local chat model works here — including
        # ones that can't drive the analysis agent.
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            num_ctx=int(os.environ.get("OLLAMA_NUM_CTX", "32768")),
            num_predict=512,
            temperature=0,
            keep_alive=os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
            client_kwargs={"timeout": float(os.environ.get("OLLAMA_TIMEOUT_SEC", "600"))},
        )
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


def _prd_requirement_text_for_judge(es_store, prd_source_id: str, prd_section: str | None, max_chars: int = 12000) -> str:
    """Load ingested PRD body text for this section from Elasticsearch (same chunks as read_prd_document)."""
    from agents.analysis_agent import _fetch_prd_chunks_ordered, _normalize_heading_for_coverage

    if not prd_source_id or not (prd_section or "").strip():
        return ""
    chunks = _fetch_prd_chunks_ordered(
        es_store,
        prd_source_id,
        source_fields=["section_heading", "chunk_text", "chunk_index"],
    )
    target = _normalize_heading_for_coverage(prd_section.strip())
    pieces: list[str] = []
    raw = prd_section.strip()
    collecting = False
    for c in chunks:
        sh = (c.get("section_heading") or "").strip()
        if sh:
            collecting = sh == raw or _normalize_heading_for_coverage(sh) == target
        if collecting:
            txt = (c.get("chunk_text") or "").strip()
            if txt:
                pieces.append(txt)
    body = "\n\n".join(pieces).strip()
    if len(body) > max_chars:
        body = body[:max_chars] + "\n\n[…truncated for judge context]"
    return body


def _build_judge_context(decision: dict, es_store) -> str:
    """Build the context string for the judge LLM."""
    parts = []

    prd_section = decision.get("prd_section") or "Not specified"
    prd_source_id = (decision.get("prd_source") or "").strip()
    req_text = _prd_requirement_text_for_judge(es_store, prd_source_id, prd_section if prd_section != "Not specified" else None)
    parts.append(f"## PRD section (heading)\n{prd_section}")
    if req_text:
        parts.append(f"## PRD requirement text (from ingested document)\n{req_text}")
    else:
        parts.append(
            "## PRD requirement text\n"
            "(Could not load chunk text from the search index — judge using heading and test case only.)"
        )

    # Test case (if applicable)
    jira_key = decision.get("jira_key")
    if jira_key:
        try:
            resp = es_store._client.search(
                index="qa_test_cases",
                query={"term": {"jira_key": jira_key}},
                source=["jira_key", "summary", "steps_text", "labels"],
                size=1,
            )
            hits = resp["hits"]["hits"]
            if hits:
                t = hits[0]["_source"]
                parts.append(
                    f"## Test Case: {jira_key}\n"
                    f"Summary: {t.get('summary', 'N/A')}\n"
                    f"Labels: {', '.join(t.get('labels') or [])}\n"
                    f"Steps: {(t.get('steps_text') or 'N/A')[:500]}"
                )
        except Exception:
            parts.append(f"## Test Case: {jira_key}\n(Could not fetch details)")

    # Decision
    action = decision["action"].upper()
    reason = decision.get("reason", "No reason given")
    content = decision.get("updated_content") or {}

    parts.append(f"## Decision: {action}")
    parts.append(f"Reason: {reason}")

    if content:
        parts.append(f"Details: {json.dumps(content, indent=2)[:500]}")

    return "\n\n".join(parts)


def _grade_decision(llm, context: str) -> dict:
    """Call the judge LLM and parse its response."""
    from langchain_core.messages import HumanMessage, SystemMessage

    response = llm.invoke([
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=context),
    ])

    text = response.content.strip()
    # Try to parse JSON from the response
    try:
        # Handle potential markdown wrapping
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        return {
            "correctness": int(result.get("correctness", 0)),
            "reasoning": int(result.get("reasoning", 0)),
            "completeness": int(result.get("completeness", 0)),
            "comment": result.get("comment", ""),
        }
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning(f"Judge returned non-JSON: {text[:200]}")
        return {
            "error": f"Judge response was not valid JSON: {e}",
            "parse_failed": True,
            "raw_excerpt": text[:500],
        }


def _avg(scores: list[dict], field: str) -> float:
    vals = [s[field] for s in scores if field in s and isinstance(s[field], (int, float))]
    return sum(vals) / len(vals) if vals else 0.0
