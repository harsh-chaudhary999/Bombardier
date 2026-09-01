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

    # Only fully-graded rows are averaged. A row missing a dimension is a fact about the
    # judge, not about the decision, and including it would bias the corpus average
    # downward while looking like evidence about the pipeline.
    graded = [s for s in scores if isinstance(s.get("rubric_score"), (int, float))]
    parse_failures = sum(1 for s in scores if s.get("parse_failed"))
    incomplete = sum(1 for s in scores if s.get("incomplete"))

    avg_correctness = _avg(graded, "correctness")
    avg_reasoning = _avg(graded, "reasoning")
    avg_completeness = _avg(graded, "completeness")
    # The headline number is weighted, not a flat mean: getting the action right matters
    # more than explaining it well. See RUBRIC_WEIGHTS.
    avg_rubric = _avg(graded, "rubric_score")

    return {
        "run_id": run_id,
        "total_decisions": len(decisions),
        "sample_size": len(sampled),
        "evaluated": len(graded),
        "parse_failures": parse_failures,
        "incomplete_grades": incomplete,
        "rubric_weights": RUBRIC_WEIGHTS,
        "scores": scores,
        "avg_correctness": round(avg_correctness, 2),
        "avg_reasoning": round(avg_reasoning, 2),
        "avg_completeness": round(avg_completeness, 2),
        # 0-1 weighted composite. `avg_score` is kept as an alias so existing consumers
        # do not break, but it now carries the weighted value, not a flat mean.
        "avg_rubric_score": round(avg_rubric, 3),
        "avg_score": round(avg_rubric, 3),
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


#: Dimensions the judge must score, and their weight in the composite.
#:
#: correctness must weigh MORE THAN THE OTHER TWO COMBINED, not merely more than each.
#: At 0.50/0.30/0.20 a wrong action explained perfectly (1/5/5) and a right action
#: explained terribly (5/1/1) both score 0.6 — the weighting reads as if correctness
#: dominates while actually making the two exactly equivalent. A wrong DEPRECATE is a
#: deleted test; no quality of prose compensates for it.
RUBRIC_WEIGHTS: dict[str, float] = {
    "correctness":  0.60,
    "reasoning":    0.25,
    "completeness": 0.15,
}
_SCORE_MIN, _SCORE_MAX = 1, 5


def coerce_dimension(value) -> int | None:
    """
    One rubric score as an int in 1..5, or None if the judge did not give a usable one.

    None rather than 0 matters. `int(result.get(field, 0))` scored a missing dimension
    as zero, which is indistinguishable from a decision the judge rated as terrible —
    so a judge that omitted a field silently dragged the corpus average down and looked
    like evidence about the pipeline. Out-of-range values are clamped rather than
    discarded: a judge answering 7 on a 1-5 scale means "as high as it goes".
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return None
    return max(_SCORE_MIN, min(number, _SCORE_MAX))


def rubric_score(grade: dict) -> float | None:
    """
    Weighted composite of the rubric dimensions, normalised to 0-1.

    None when any dimension is missing — a partial grade must not be averaged into a
    number that reads as complete. Callers report those separately.
    """
    values = {d: coerce_dimension(grade.get(d)) for d in RUBRIC_WEIGHTS}
    if any(v is None for v in values.values()):
        return None
    total_weight = sum(RUBRIC_WEIGHTS.values())
    weighted = sum(values[d] * w for d, w in RUBRIC_WEIGHTS.items())
    return round(weighted / (_SCORE_MAX * total_weight), 3)


def missing_dimensions(grade: dict) -> list[str]:
    """Rubric dimensions the judge failed to supply a usable score for."""
    return sorted(d for d in RUBRIC_WEIGHTS if coerce_dimension(grade.get(d)) is None)


def _extract_json(text: str) -> str:
    """Strip markdown fencing a judge may wrap its JSON in."""
    stripped = text.strip()
    if stripped.startswith("```"):
        parts = stripped.split("```")
        if len(parts) > 1:
            stripped = parts[1]
        if stripped.lstrip().lower().startswith("json"):
            stripped = stripped.lstrip()[4:]
    return stripped.strip()


def _grade_decision(llm, context: str) -> dict:
    """Call the judge LLM and parse its response into a scored rubric."""
    from langchain_core.messages import HumanMessage, SystemMessage

    response = llm.invoke([
        SystemMessage(content=JUDGE_PROMPT),
        HumanMessage(content=context),
    ])

    text = response.content.strip()
    try:
        result = json.loads(_extract_json(text))
    except (json.JSONDecodeError, ValueError) as e:
        logger.warning("Judge returned non-JSON: %s", text[:200])
        return {
            "error": f"Judge response was not valid JSON: {e}",
            "parse_failed": True,
            "raw_excerpt": text[:500],
        }
    if not isinstance(result, dict):
        return {
            "error": "Judge response was JSON but not an object",
            "parse_failed": True,
            "raw_excerpt": text[:500],
        }

    grade: dict = {d: coerce_dimension(result.get(d)) for d in RUBRIC_WEIGHTS}
    grade["comment"] = str(result.get("comment", ""))[:1000]

    absent = missing_dimensions(result)
    if absent:
        # Recorded, not silently zeroed: this is a fact about the judge, not the decision.
        grade["missing_dimensions"] = absent
        grade["incomplete"] = True
        logger.warning("Judge omitted rubric dimension(s): %s", absent)
    grade["rubric_score"] = rubric_score(result)
    return grade


def _avg(scores: list[dict], field: str) -> float:
    vals = [s[field] for s in scores if field in s and isinstance(s[field], (int, float))]
    return sum(vals) / len(vals) if vals else 0.0
