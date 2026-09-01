"""
Phase 3: PRD → Test Coverage Analysis Agent (Tool-using design)

The LLM is given tools to:
  1. read_prd_document   — fetch the full PRD from Elasticsearch (already ingested)
  2. search_tests        — hybrid KNN+BM25 search against qa_test_cases
  3. get_test_details    — fetch full steps/labels for a specific test case
  4. record_decision     — persist a coverage decision to Postgres

Guardrails (v2):
  - Token budget tracking: warns the agent when approaching context limits
  - Decision deduplication: prevents recording the same decision twice for one test
  - Jira key validation: only allows keys that were returned by search_tests
  - Message compaction: summarises old tool results to free context space

Supported providers: anthropic | azure_openai | openai
"""
import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import tiktoken

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from agents.loop_status import LoopStatus
from embeddings.embed_client import EmbedClient
from embeddings.es_store import ESStore
from embeddings.pg_store import PGStore
from observability.canonical_json import fingerprint_sha256, normalize_json_obj
from observability.phase_ledger import append_entry_async

logger = logging.getLogger(__name__)

# Align agent-reported prd_section strings with ES section_heading for coverage_score (strip "3.2 " etc.).
_HEADING_PREFIX_RE = re.compile(
    r"^\s*(?:[\d.]+[\).\s]+|[\d.]+\s+|(?:§|•|\u2022|-)\s*)*"
)


def _normalize_heading_for_coverage(label: str | None) -> str:
    """
    Canonical form of a section label, for comparing agent-authored `prd_section` strings
    against headings stored in Elasticsearch.

    Both sides are messy in different ways. Confluence headings carry irregular internal
    whitespace and trailing punctuation from the original document ("Objective  –",
    "Objective: -"), while the model writes a tidied version ("Objective –", "Objective").
    Without collapsing whitespace and trimming trailing punctuation those never match, and
    two things quietly break: coverage_score under-counts, and incremental carry-forward
    finds nothing to carry.
    """
    t = (label or "").strip()
    if not t:
        return ""
    t = _HEADING_PREFIX_RE.sub("", t, count=1).strip()
    # "Objective  –" and "Objective –" must agree.
    t = re.sub(r"\s+", " ", t)
    # Trailing separators are document formatting, not part of the heading.
    t = re.sub(r"[\s:;.\-–—•]+$", "", t)
    return t.casefold().strip()


# Serialize concurrent runs sharing the same run_id (deterministic ID mode).
_MAX_ANALYSIS_LOCK_KEYS = 4096
_analysis_run_locks: OrderedDict[str, asyncio.Lock] = OrderedDict()


def _lock_for_run_id(run_id: str) -> asyncio.Lock:
    if run_id in _analysis_run_locks:
        _analysis_run_locks.move_to_end(run_id)
        return _analysis_run_locks[run_id]
    lk = asyncio.Lock()
    _analysis_run_locks[run_id] = lk
    while len(_analysis_run_locks) > _MAX_ANALYSIS_LOCK_KEYS:
        _analysis_run_locks.popitem(last=False)
    return lk


# ─── Token budget management ─────────────────────────────────────────────────

_ENCODING = tiktoken.get_encoding("cl100k_base")

# Real context windows per model. The previous per-provider constants assumed a 200K
# Claude window and capped the budget at 180K — against a 1M model that discarded ~82%
# of the window AND triggered compaction (which truncates old tool results to 300 chars),
# so retrieved test details were being deleted while there was ample room to keep them.
_MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-5":     1_000_000,
    "claude-opus-4-8":   1_000_000,
    "claude-opus-4-7":   1_000_000,
    "claude-opus-4-6":   1_000_000,
    "claude-sonnet-5":   1_000_000,
    "claude-sonnet-4-6": 1_000_000,
    "claude-haiku-4-5":     200_000,
    "gpt-4o":               128_000,
    "gpt-4o-mini":          128_000,
}

# Used when the model string isn't in the table (new release, custom deployment name).
# Deliberately conservative — a wrong-high guess causes hard API errors mid-run.
_PROVIDER_FALLBACK_WINDOW: dict[str, int] = {
    "anthropic":    200_000,
    "azure_openai": 128_000,
    "openai":       128_000,
}

# We count tokens with tiktoken cl100k_base, which is OpenAI's tokenizer and undercounts
# Claude by roughly 15-20% on prose (more on code). Discount the window so the undercount
# cannot push us past the real limit. Remove this once token counting is provider-native.
_TOKENIZER_SAFETY_FACTOR = 0.82

_TOKEN_WARNING_RATIO = 0.75  # warn agent at 75% of context budget
_TOKEN_COMPACT_RATIO = 0.85  # compact old messages at 85%

# Share of the conversation budget a single read_prd_document result may occupy. The rest
# is needed for search results, test details and the model's output across ~40 turns.
_PRD_READ_BUDGET_RATIO = float(os.environ.get("QA_PRD_READ_BUDGET_RATIO", "0.4"))


def _ollama_num_ctx() -> int:
    """
    Ollama's context window is set per-request via num_ctx and defaults to a few
    thousand tokens REGARDLESS of what the model supports — anything beyond it is
    silently dropped from the prompt. Must be set explicitly or the PRD gets truncated.
    """
    return int(os.environ.get("OLLAMA_NUM_CTX", "32768"))


def _context_limit(provider: str, model: str = "") -> int:
    """
    Token budget before compaction kicks in, derived from the model's real window.

    For ollama, num_ctx is a hard wall (silent truncation, not an error), so the budget
    comes from num_ctx directly. For hosted providers it comes from the model's context
    window, discounted for the tokenizer mismatch and with room reserved for output.

    Override with QA_AGENT_CONTEXT_LIMIT when the table is wrong for your deployment.
    """
    override = os.environ.get("QA_AGENT_CONTEXT_LIMIT", "").strip()
    if override:
        try:
            return max(8_000, int(override))
        except ValueError:
            logger.warning("Ignoring malformed QA_AGENT_CONTEXT_LIMIT=%r", override)

    if provider == "ollama":
        return max(8_000, _ollama_num_ctx() - _max_output_tokens() - 2_000)

    window = _MODEL_CONTEXT_WINDOWS.get(model)
    if window is None:
        window = _PROVIDER_FALLBACK_WINDOW.get(provider, 128_000)
        if model:
            logger.info(
                "Model %r not in the context-window table — assuming %s tokens for %s. "
                "Set QA_AGENT_CONTEXT_LIMIT to be explicit.",
                model, window, provider,
            )
    usable = int(window * _TOKENIZER_SAFETY_FACTOR) - _max_output_tokens()
    return max(8_000, usable)


def _count_tokens_str(text: str) -> int:
    """Token count for a single string, using the same encoder as the budget checks."""
    return len(_ENCODING.encode(text or "", disallowed_special=()))


# Reasoning models emit chain-of-thought. Where the provider exposes it as a separate
# field (Anthropic thinking blocks) it never reaches .content — but local models served
# through Ollama commonly inline it as <think>...</think>, and Qwen3 in particular does.
# Left in place it corrupts everything downstream: /ask answers begin with paragraphs of
# deliberation, citation parsing picks up bracket numbers from the reasoning, and the
# no-tool-call nudge sees a tool name mentioned mid-thought and misdiagnoses.
_THINK_BLOCK_RE = re.compile(
    r"<\s*(think|thinking|thought|reasoning)\s*>.*?<\s*/\s*\1\s*>",
    re.DOTALL | re.IGNORECASE,
)
# Unterminated block: the model was cut off mid-thought (hit max_tokens). Everything from
# the opening tag on is reasoning, so drop it rather than surface a truncated monologue.
_THINK_OPEN_RE = re.compile(r"<\s*(think|thinking|thought|reasoning)\s*>.*\Z",
                            re.DOTALL | re.IGNORECASE)


def message_text(message) -> str:
    """
    Plain text of a model response, with inline reasoning blocks removed.

    Handles the three shapes LangChain returns: a plain string, a list of content blocks
    (where reasoning arrives as a non-text block and is skipped), or something else.
    """
    content = getattr(message, "content", message)
    # A turn that is purely tool calls carries content=None (or []). str(None) would yield
    # the literal "None", which /ask would then present as the answer text.
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                # Skip thinking/reasoning/redacted blocks — only real text is the answer.
                if block.get("type") in ("thinking", "redacted_thinking", "reasoning"):
                    continue
                if "text" in block:
                    parts.append(str(block["text"]))
            elif isinstance(block, str):
                parts.append(block)
        text = "\n".join(parts)
    elif isinstance(content, str):
        text = content
    else:
        text = str(content)

    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.strip()


def _count_message_tokens(messages: list) -> int:
    """Approximate token count across all messages in the conversation."""
    total = 0
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            total += len(_ENCODING.encode(content, disallowed_special=()))
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and "text" in block:
                    total += len(_ENCODING.encode(block["text"], disallowed_special=()))
        # Count tool call args
        for tc in getattr(msg, "tool_calls", []):
            args_str = json.dumps(tc.get("args", {}))
            total += len(_ENCODING.encode(args_str, disallowed_special=()))
    return total


_PRD_TOOL_STUB_THRESHOLD = 12_000


def _looks_like_prd_tool_output(content: str) -> bool:
    if len(content) < 800:
        return False
    return content.lstrip().startswith("#") or "### " in content[:4000]


def _prd_stub_from_content(content: str) -> str:
    headings = re.findall(r"^### (.+)$", content, re.MULTILINE)
    nh = len(headings)
    preview = headings[:20]
    tail = f" (+{nh - len(preview)} more)" if nh > len(preview) else ""
    return (
        f"[PRD tool output compacted — was {len(content)} chars]\n"
        f"Section headings ({nh}): {', '.join(preview)}{tail}\n"
        "Call read_prd_document again if you need verbatim body text for a section."
    )


def _shrink_stale_prd_reads(messages: list) -> list:
    """
    Keep only the last large PRD-shaped ToolMessage verbatim; stub earlier reads to save context.
    Also stubs a lone oversized PRD message if it sits earlier than the last 8 messages.
    """
    indices: list[int] = []
    for i, msg in enumerate(messages):
        if not isinstance(msg, ToolMessage):
            continue
        c = msg.content if isinstance(msg.content, str) else str(msg.content)
        if len(c) >= _PRD_TOOL_STUB_THRESHOLD and _looks_like_prd_tool_output(c):
            indices.append(i)
    if len(indices) <= 0:
        return messages
    out = list(messages)
    # Stub all but the most recent large PRD tool result
    for i in indices[:-1]:
        m = messages[i]
        c = m.content if isinstance(m.content, str) else str(m.content)
        out[i] = ToolMessage(
            content=_prd_stub_from_content(c),
            tool_call_id=m.tool_call_id,
        )
    # Single huge PRD deep in history (still in "recent" tail): stub if not among last 8 msgs
    if len(indices) == 1:
        i = indices[0]
        if i < len(messages) - 8:
            m = messages[i]
            c = m.content if isinstance(m.content, str) else str(m.content)
            out[i] = ToolMessage(
                content=_prd_stub_from_content(c),
                tool_call_id=m.tool_call_id,
            )
    return out


def _compact_old_messages(messages: list, keep_recent: int = 6) -> list:
    """
    Replace old tool result messages with compact summaries to free context space.
    Keeps the system prompt, last `keep_recent` messages intact.
    """
    if len(messages) <= keep_recent + 2:
        return messages

    # Always keep: system message (0) + user message (1) + last N messages
    preserved_head = messages[:2]
    preserved_tail = messages[-keep_recent:]
    middle = messages[2:-keep_recent]

    compacted = []
    for msg in middle:
        if isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            if len(content) > 500:
                # Summarise long tool results
                compacted.append(ToolMessage(
                    content=f"[Compacted — {len(content)} chars] {content[:300]}...",
                    tool_call_id=msg.tool_call_id,
                ))
            else:
                compacted.append(msg)
        else:
            compacted.append(msg)

    return preserved_head + compacted + preserved_tail

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

# Headings that indicate non-testable meta-sections (rationale, planning, legal boilerplate).
# Rule: anchor patterns to whole headings or clear phrases — never match common words inside
# feature titles (e.g. "Background Jobs", "Error Metrics Dashboard").
# Prefix stripping uses _HEADING_PREFIX_RE (same as coverage normalization).
_META_HEADING_RULES: tuple[re.Pattern, ...] = (
    re.compile(r"^why (this )?change\b", re.I),
    re.compile(r"^success (looks like|criteria)\s*$", re.I),
    re.compile(r"^success metrics\s*$", re.I),
    re.compile(r"^introduction\s*$", re.I),
    re.compile(r"^rationale\s*$", re.I),
    re.compile(r"^appendix\b", re.I),
    re.compile(r"^glossary\s*$", re.I),
    re.compile(r"^open questions?\s*$", re.I),
    re.compile(r"^out of scope\s*$", re.I),
    re.compile(r"^(timeline|rollout)\b", re.I),
    re.compile(r"^methodology\s*$", re.I),
    re.compile(r"^hypothesis\b", re.I),
    re.compile(r"^budget\s*$", re.I),
    re.compile(r"^current (flow|process|system)\s*$", re.I),
    re.compile(r"^process flow\s*$", re.I),
    re.compile(r"^mitigation (plan|strategy)\s*$", re.I),
    re.compile(r"^scaling strategy\s*$", re.I),
    re.compile(r"^pilot scope\s*$", re.I),
    re.compile(r"^(monthly|quarterly|annual) review\s*$", re.I),
    re.compile(r"^background\s*$", re.I),
    re.compile(r"^context\s*$", re.I),
    re.compile(r"^background (and|&) context\s*$", re.I),
    re.compile(r"^risk register\s*$", re.I),
    re.compile(r"^document history\s*$", re.I),
    re.compile(r"^executive summary\s*$", re.I),
)


def _clean_heading_for_meta(heading: str) -> str:
    """Strip numbering/markdown heading prefixes (same family as coverage normalization)."""
    t = (heading or "").strip()
    if not t:
        return ""
    t = _HEADING_PREFIX_RE.sub("", t, count=1).strip()
    return t


def _is_meta_heading(heading: str) -> bool:
    """True if this section title is boilerplate / non-testable meta, not feature requirements."""
    clean = _clean_heading_for_meta(heading)
    if not clean:
        return False
    return any(p.search(clean) for p in _META_HEADING_RULES)


# ─── Guardrails on untrusted PRD content and on decision quality ──────────────

_INJECTION_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"IGNORE\s+(?:ALL\s+)?(?:PREVIOUS|PRIOR|ABOVE)\s+INSTRUCTIONS", re.IGNORECASE),
    re.compile(r"DISREGARD\s+(?:ALL\s+)?(?:PREVIOUS|PRIOR|ABOVE)\s+INSTRUCTIONS", re.IGNORECASE),
    re.compile(r"<\|\s*(?:system|user|assistant|im_start|im_end)\s*\|>", re.IGNORECASE),
    re.compile(r"\[INST\].*?\[/INST\]", re.IGNORECASE | re.DOTALL),
    re.compile(r"(?m)^\s*#{1,6}\s*(?:System|Human|Assistant)\s*:", re.IGNORECASE),
    re.compile(r"(?m)^\s*-{3,}\s*SYSTEM\s*:", re.IGNORECASE),
)


def _sanitize_prd_content(text: str) -> str:
    """
    Strip known prompt-injection patterns from PRD text before it reaches the model.

    PRD content is untrusted input on a privileged path: anyone who can edit a
    Confluence page can put text into the analysis context. This is defence in
    depth, not a guarantee — it is a denylist, and denylists leak. The system
    prompt must also treat document content as data rather than instructions.

    Deliberately narrow. The role-marker patterns are anchored to the start of a
    line so an ordinary heading like '### Assumptions' is untouched; only the
    conversational role markers ('### System:') match.
    """
    if not text:
        return text
    for pattern in _INJECTION_PATTERNS:
        text = pattern.sub("[FILTERED]", text)
    return text


_MIN_REASON_CHARS = 30
_REMOVAL_KEYWORDS = (
    "remov", "replac", "obsolet", "discontinu", "no longer",
    "deleted", "delete", "dropped", "retired", "superseded",
)


def _check_reason_quality(action: str, reason: str | None) -> str | None:
    """
    Return an error message for the agent, or None if the reason is acceptable.

    A decision whose reason is 'N/A' is unreviewable — the reviewer cannot tell
    whether the agent was right, so the decision is worthless even when correct.
    DEPRECATE carries the extra requirement because it is the one irreversible
    action: the reason must say what was removed, not merely that the test is
    unwanted.
    """
    text = (reason or "").strip()
    if len(text) < _MIN_REASON_CHARS:
        return (
            f"Error: reason is too short ({len(text)} chars). Provide at least "
            f"{_MIN_REASON_CHARS} characters explaining the decision — the reviewer "
            f"sees only this."
        )
    if action == "deprecate" and not any(k in text.lower() for k in _REMOVAL_KEYWORDS):
        return (
            "Error: a DEPRECATE reason must state what was removed or replaced, citing "
            "the PRD section that confirms it. If the feature still exists but changed, "
            "use action='update' instead."
        )
    return None


_DUP_WARN_SCORE = 0.88
_DUP_BLOCK_SCORE = 0.93


def _duplicate_verdict(similar: list[dict]) -> str | None:
    """
    Decide what to tell the agent about near-identical existing tests.

    None means proceed. A returned string is shown to the agent; one containing
    BLOCKED means the CREATE was refused. Split from the search itself so the
    thresholds are testable without Elasticsearch.
    """
    ranked = sorted(similar, key=lambda t: t.get("score") or 0.0, reverse=True)
    if not ranked:
        return None

    lines = ["Very similar tests already exist. Consider UPDATE instead of CREATE:"]
    lines += [
        f"  [{t.get('jira_key')}] {t.get('summary')} (score: {(t.get('score') or 0.0):.3f})"
        for t in ranked[:3]
    ]

    if (ranked[0].get("score") or 0.0) >= _DUP_BLOCK_SCORE:
        return "\n".join(lines) + (
            "\n[BLOCKED — too similar to an existing test. Record an UPDATE against the "
            "closest match, or a QUESTION if you cannot tell them apart.]"
        )
    return "\n".join(lines) + (
        "\nIf this genuinely covers different behaviour, record it again and state in the "
        "reason how it differs from the tests above."
    )


# Similarity at which a chunk from a DIFFERENT document counts as covering a test.
# Scores here are Elasticsearch KNN cosine scores, (1 + cos) / 2, so 0.75 is cosine 0.5 —
# related subject matter rather than a paraphrase. Deliberately loose: the output is a
# prompt to go and check, not an automatic refusal.
_CROSS_PRD_SCORE = float(os.environ.get("QA_CROSS_PRD_SCORE", "0.75"))
_CROSS_PRD_TOP_K = int(os.environ.get("QA_CROSS_PRD_TOP_K", "8"))


def _cross_prd_verdict(matches: list[dict], current_source_id: str) -> str | None:
    """
    Warn when a test about to be deprecated looks covered by a different document.

    None means proceed. This is the check the deprecation rules refer to when they say
    absence from *this* PRD is not absence from the corpus: the agent analyses one
    document at a time, so without it "I could not find the feature" cannot be
    distinguished from "the feature is documented elsewhere".

    Advisory, never a block. A test genuinely being retired often still has stale
    documentation somewhere, and refusing the decision outright would make the
    deprecation path unusable — the agent is told to look and to say what it found.
    """
    others: dict[str, dict] = {}
    for m in matches:
        source = m.get("source_id")
        score = m.get("score") or 0.0
        if not source or source == current_source_id or score < _CROSS_PRD_SCORE:
            continue
        if source not in others or score > (others[source].get("score") or 0.0):
            others[source] = m
    if not others:
        return None

    ranked = sorted(others.values(), key=lambda m: m.get("score") or 0.0, reverse=True)
    lines = ["Hold on — other documents appear to describe this behaviour:"]
    for m in ranked[:3]:
        heading = m.get("section_heading") or "(no heading)"
        title = m.get("doc_title") or m.get("source_id")
        lines.append(f"  {m['source_id']} — {title} › {heading} (score: {m['score']:.3f})")
    lines.append(
        "Deprecating removes the test everywhere, not just from this PRD's scope. Read "
        "those sections before continuing. If they do not in fact cover this test, record "
        "the DEPRECATE again and say in the reason why they do not. If you cannot tell, "
        "record a QUESTION instead."
    )
    return "\n".join(lines)


_RRF_RANK_CONSTANT = int(os.environ.get("QA_AGENT_RRF_RANK_CONSTANT", "60"))


def _rrf_merge_results(result_lists: list[list[dict]]) -> list[dict]:
    """
    Fuse several result lists by Reciprocal Rank Fusion, de-duplicating on jira_key.

    Used when one requirement is searched with several phrasings. RRF rewards
    consensus: a test that surfaces for every phrasing outranks one that happens to
    top a single odd query. It is rank-based, so the incomparable score scales of
    different queries never have to be normalised.

    Note the asymmetry with validate_prd_data / build_preview, which merge the same
    kind of multi-query result by taking each test's best score instead. That path
    feeds eval/benchmark.py, so changing it moves the retrieval baseline; the two
    should be unified once a post-W1 baseline has been measured.
    """
    scores: dict[str, float] = {}
    docs: dict[str, dict] = {}
    for results in result_lists:
        for rank, r in enumerate(results, start=1):
            key = r.get("jira_key")
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + _RRF_RANK_CONSTANT)
            # Keep the record that scored best on its own query — it carries the
            # richest field set for the reranker to read.
            if key not in docs or (r.get("score") or 0.0) > (docs[key].get("score") or 0.0):
                docs[key] = r

    # Copy before rewriting `score`: the caller still owns the lists it passed in, and
    # overwriting their per-query scores in place would corrupt anything that reads them
    # afterwards (diagnostics, a second merge, or the validate path if the two are unified).
    merged = [dict(d) for d in docs.values()]
    for r in merged:
        r["score"] = round(scores[r["jira_key"]], 6)
    merged.sort(key=lambda r: r["score"], reverse=True)
    return merged


def _strip_frontmatter(text: str) -> str:
    """
    Drop a leading YAML frontmatter block. The knowledge bundle in prompts/ carries
    OKF-style frontmatter for humans and other consumers; the model only needs the body.
    No YAML parser required — we locate the fences, we do not interpret the keys.

    Only the FIRST fenced block is removed. Markdown horizontal rules later in the body
    are ordinary content and must survive (a naive split on the next '---' silently
    truncates the document at its first rule).
    """
    if not text.startswith("---"):
        return text
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return text
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            return "\n".join(lines[i + 1:]).lstrip("\n")
    # No closing fence — not frontmatter after all. Return unchanged rather than guess.
    return text


def _load_prompt(filename: str) -> str:
    path = _PROMPTS_DIR / filename
    if not path.exists():
        raise RuntimeError(f"Required prompt file not found: {path}")
    return _strip_frontmatter(path.read_text())


def _knowledge_files() -> list[str]:
    """Readable documents in the knowledge bundle (allowlist for read_knowledge)."""
    if not _PROMPTS_DIR.is_dir():
        return []
    return sorted(p.name for p in _PROMPTS_DIR.glob("*.md"))


_GUIDELINES  = _load_prompt("test-case-guidelines.md")
_DEPRECATION = _load_prompt("deprecation-rules.md")

_SYSTEM_PROMPT_TEMPLATE = """You are an expert QA analyst reviewing test coverage against a Product Requirements Document (PRD).

You have access to tools. Use them in this order:
1. Call `list_prd_sections` FIRST to see every section and which are testable. It is short —
   keep it in context and treat it as your checklist for the run.
2. Call `read_prd_document` to get the full PRD content
3. For each feature/requirement you identify, call `search_tests` to find relevant existing tests
4. Call `get_test_details` for any test you want to inspect more closely
5. Call `record_decision` for each conclusion you reach — do this as you go, not all at the end

Before you finish, check the list from step 1: every section marked [testable] should have at
least one decision. Record a QUESTION for any you could not resolve rather than leaving it out —
a section with no decision is indistinguishable from one you never looked at.

## Your Analysis Goal
For every requirement in the PRD, determine:
- Which existing tests KEEP covering it (no change needed)
- Which existing tests need UPDATE (feature changed, steps outdated)
- Which existing tests should be DEPRECATED (feature removed or replaced)
- Which requirements have NO existing test and need a new one (CREATE)

## Search Strategy — Critical for Quality
Use MULTIPLE search queries per feature to maximise coverage. A single search often misses tests
that use different terminology. For each feature, search with at least 2-3 query variations:

1. **Specific feature query**: Use exact feature names and actions
   e.g. "one-step checkout using saved address and payment"
2. **Broader functional query**: Describe the user flow or UI component
   e.g. "order form validation and payment capture"
3. **Edge case / negative query**: Think about error paths and boundary conditions
   e.g. "checkout button disabled when address incomplete"

When search_tests returns few or no results, try:
- Different synonyms (purchase → order, account → customer, login → sign in)
- More general terms (remove specific product names, use generic UI terms)
- Breaking compound features into sub-features and searching each

{KNOWLEDGE_SECTION}

## Important Rules
- Read the PRD from read_prd_document first — incremental runs may receive only changed sections
- Be conservative: prefer UPDATE over DEPRECATE when the feature still exists
- Only reference test jira_keys you've actually retrieved via search_tests or get_test_details
- Record a decision for every significant finding — don't just think it, write it
- The loop ends when you stop calling tools (no concluding phrase required)
"""

# Inlined guidance: full text in the prompt. Correct for providers with prompt caching —
# the system block is a stable prefix served at ~0.1x input cost, so deferring it would
# trade cheap cached tokens for extra tool round-trips.
_KNOWLEDGE_INLINE = f"""## Test Case Writing Guidelines
{_GUIDELINES}

## Deprecation Rules
{_DEPRECATION}"""

# On-demand guidance: index only. Correct for local providers, which get no cache discount
# and enforce num_ctx as a hard prompt wall — ~9KB of inlined guidance crowds out the PRD.
_KNOWLEDGE_ON_DEMAND = """## QA Knowledge Bundle (read on demand)
Curated guidance is available via the `read_knowledge` tool rather than inlined here, to
keep the context window free for PRD content. Call `read_knowledge` with one of:

- `index.md` — what each document covers and when to read it (start here if unsure)
- `test-case-guidelines.md` — read BEFORE proposing a CREATE or rewriting steps for an UPDATE
- `deprecation-rules.md` — read BEFORE recording a DEPRECATE decision
- `prd-to-knowledge-base.md` — read when a PRD section is ambiguous about actual behaviour

Read a document once and reuse it for the rest of the run; do not re-read the same path."""

# Providers whose prompt prefix is cached, making inlined guidance effectively free.
_CACHED_PREFIX_PROVIDERS = frozenset({"anthropic"})


def _system_prompt(provider: str = "anthropic") -> str:
    """
    Build the system prompt for a provider.

    Inlines the knowledge bundle where the prefix is cached, and points at the
    `read_knowledge` tool where context window is the binding constraint.
    Override with QA_AGENT_KNOWLEDGE_MODE=inline|on_demand.
    """
    mode = os.environ.get("QA_AGENT_KNOWLEDGE_MODE", "").strip().lower()
    if mode not in ("inline", "on_demand"):
        mode = "inline" if provider in _CACHED_PREFIX_PROVIDERS else "on_demand"
    section = _KNOWLEDGE_INLINE if mode == "inline" else _KNOWLEDGE_ON_DEMAND
    return _SYSTEM_PROMPT_TEMPLATE.replace("{KNOWLEDGE_SECTION}", section)


# Back-compat: the fully-inlined prompt. Prefer _system_prompt(provider).
SYSTEM_PROMPT = _system_prompt("anthropic")


def _build_system_message(provider: str):
    """
    System message for the run, with prompt caching enabled where supported.

    The system block is a large, byte-stable prefix resent on every turn of a run that can
    reach 40 turns — the textbook case for a cache breakpoint. Anthropic serves cache reads
    at ~0.1x input cost, so this is the single biggest cost lever in the pipeline.

    Caching is a PREFIX match: anything that varies per request must stay out of this block.
    The run-specific parts (PRD id, module filter, focus headings) are deliberately in the
    HumanMessage that follows, not here. Verify hits via run_metadata.token_usage.
    """
    text = _system_prompt(provider)
    caching_on = os.environ.get("QA_PROMPT_CACHING", "1") != "0"
    if provider == "anthropic" and caching_on:
        return SystemMessage(content=[{
            "type": "text",
            "text": text,
            "cache_control": {"type": "ephemeral"},
        }])
    return SystemMessage(content=text)


# ─── LLM factory ──────────────────────────────────────────────────────────────

def _max_output_tokens() -> int:
    return int(os.environ.get("QA_AGENT_MAX_OUTPUT_TOKENS", "8192"))


def _default_max_turns() -> int:
    """
    Agent turn cap. Lower this for local inference — 40 turns against a 12B model on
    consumer hardware can run for hours, and the run is marked `truncated` either way.
    """
    return int(os.environ.get("QA_AGENT_MAX_TURNS", "40"))


def _build_llm(provider: str, model: str):
    """Build a LangChain chat model with tool-calling support."""
    mt = _max_output_tokens()
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        kwargs: dict[str, Any] = {"model": model, "max_tokens": mt}

        # Extended thinking. This is a judgment task ("does this test still cover this
        # requirement?"), which is exactly where reasoning helps most. Adaptive lets the
        # model choose depth per section. Disable with QA_ANTHROPIC_THINKING=off.
        thinking_mode = os.environ.get("QA_ANTHROPIC_THINKING", "adaptive").strip().lower()
        if thinking_mode not in ("off", "disabled", "none", ""):
            kwargs["thinking"] = {"type": thinking_mode}
            # Thinking tokens are drawn from max_tokens, so a tight cap truncates
            # mid-reasoning and the turn is lost.
            if mt < 16_000:
                logger.warning(
                    "Thinking is enabled but QA_AGENT_MAX_OUTPUT_TOKENS=%s — thinking tokens "
                    "come out of this budget and responses may truncate mid-reasoning. "
                    "Consider 32000.",
                    mt,
                )

        effort = os.environ.get("QA_ANTHROPIC_EFFORT", "high").strip().lower()
        if effort:
            kwargs["output_config"] = {"effort": effort}

        try:
            return ChatAnthropic(**kwargs)
        except TypeError as e:
            # Older langchain-anthropic may not accept thinking/output_config as
            # first-class kwargs. Degrade to a working client rather than failing the run.
            logger.warning(
                "ChatAnthropic rejected reasoning kwargs (%s) — falling back to a plain "
                "client. Upgrade langchain-anthropic to enable thinking/effort.", e,
            )
            return ChatAnthropic(model=model, max_tokens=mt)

    elif provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_deployment=model,
            azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
            api_key=os.environ["AZURE_OPENAI_API_KEY"],
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
            max_tokens=mt,
        )

    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, max_tokens=mt)

    elif provider == "ollama":
        # Local inference. The model must support native tool calling — the agent is
        # tool-driven and a model that only *describes* tool calls in prose records
        # zero decisions (see _run_agent_loop's no-tool-call nudge).
        from langchain_ollama import ChatOllama

        num_ctx = _ollama_num_ctx()
        if num_ctx <= mt + 2_000:
            logger.warning(
                "OLLAMA_NUM_CTX=%s leaves no room for a %s-token response — "
                "raise OLLAMA_NUM_CTX or lower QA_AGENT_MAX_OUTPUT_TOKENS",
                num_ctx,
                mt,
            )
        return ChatOllama(
            model=model,
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434"),
            num_ctx=num_ctx,
            num_predict=mt,
            # Deterministic by default — coverage decisions should be reproducible.
            temperature=float(os.environ.get("OLLAMA_TEMPERATURE", "0")),
            # Keep the model resident between turns; a 12B cold load per turn dominates runtime.
            keep_alive=os.environ.get("OLLAMA_KEEP_ALIVE", "30m"),
            client_kwargs={"timeout": float(os.environ.get("OLLAMA_TIMEOUT_SEC", "600"))},
        )

    else:
        raise ValueError(
            f"Unknown provider: {provider!r}. Use 'anthropic', 'azure_openai', 'openai', or 'ollama'."
        )


# ─── Tool definitions ──────────────────────────────────────────────────────────

def _make_tools(prd_source_id: str, module: list[str] | None,
                embed_client: EmbedClient, es_store: ESStore,
                pg_store: PGStore, run_id: str, reranker=None,
                focus_headings: list[str] | None = None,
                prd_token_budget: int | None = None):
    """
    Create the tools bound to this specific analysis run.
    Tools are plain functions decorated with @tool — LangChain serialises their
    signatures into the tool schema the LLM sees.

    Guardrails embedded in tools:
      - read_prd_document caps its own output at prd_token_budget and paginates by section
      - search_tests populates _seen_jira_keys (keys the agent has actually retrieved)
      - record_decision validates jira_key against _seen_jira_keys (prevents hallucinated keys)
      - record_decision de-dupes on (jira_key + normalized prd section), or (create + section + summary)

    prd_token_budget bounds a single read_prd_document result. Without it a large PRD can
    return ~1.6M tokens (2000 chunks) in one tool message, which silently truncates the
    prompt on Ollama (num_ctx is a hard wall) and forces destructive compaction elsewhere.
    """

    # Max chunks to fetch — use scroll for very large PRDs
    MAX_PRD_CHUNKS = 2000
    # Fall back to a budget that is safe even on a small local window.
    _PRD_BUDGET = prd_token_budget if prd_token_budget and prd_token_budget > 0 else 8_000

    _focus_norm: set[str] | None = None
    _focus_raw: set[str] = set()
    if focus_headings:
        _focus_raw = {h.strip() for h in focus_headings if (h or "").strip()}
        _focus_norm = {_normalize_heading_for_coverage(h) for h in _focus_raw}
        _focus_norm.discard("")

    def _chunk_heading_matches_focus(raw_heading: str) -> bool:
        if _focus_norm is None:
            return True
        rh = (raw_heading or "").strip()
        if rh in _focus_raw:
            return True
        return _normalize_heading_for_coverage(rh) in _focus_norm

    # ── Guardrail state (shared across tool calls within one run) ──
    _seen_jira_keys: set[str] = set()       # keys returned by search_tests or get_test_details
    # One decision per (issue key + normalized section) or per (create + section + summary slug)
    _decided_identities: dict[tuple, str] = {}
    # (guard_name, decision_identity) pairs already warned about this run. Advisory
    # guards fire once so "go and check, then record it again" is actually followable.
    _advised: set[tuple] = set()
    _decision_count: dict[str, int] = {"keep": 0, "update": 0, "deprecate": 0, "create": 0, "question": 0}

    def _decision_identity(
        act: str,
        jk: str | None,
        section: str | None,
        new_summary: str | None,
    ) -> tuple:
        sec = _normalize_heading_for_coverage(section)
        if act == "create":
            return ("create", sec, (new_summary or "").strip()[:400])
        return ("issue", (jk or "").strip(), sec)

    @tool
    def read_prd_document(section: str | None = None) -> str:
        """
        Read the PRD document for this analysis session. Call this first, before searching
        tests. In incremental analysis, only changed/new sections may be included.

        Args:
            section: Optional section heading to read in full. Omit on the first call to get
                     the document (or, if it is too large for the context window, its outline
                     plus the opening sections). If the response says the document was
                     truncated, call again with a heading from the outline to read that
                     section verbatim.
        """
        from elasticsearch import helpers as es_helpers

        chunks = []
        resp = es_helpers.scan(
            es_store._client,
            index="qa_prd_chunks",
            query={"query": {"term": {"source_id": prd_source_id}},
                   "_source": ["section_heading", "chunk_text", "doc_title", "chunk_index"]},
            scroll="2m",
            size=500,
        )
        for hit in resp:
            chunks.append(hit["_source"])
            if len(chunks) >= MAX_PRD_CHUNKS:
                logger.warning(f"PRD {prd_source_id} has >{MAX_PRD_CHUNKS} chunks, truncating")
                break

        if not chunks:
            return f"No document found for source_id={prd_source_id!r}. Make sure it has been ingested."

        chunks.sort(key=lambda c: c.get("chunk_index", 0))
        if _focus_norm is not None:
            filtered: list[dict] = []
            cur_heading = ""
            for c in chunks:
                h = (c.get("section_heading") or "").strip()
                if h:
                    cur_heading = h
                effective = h or cur_heading
                if effective and _chunk_heading_matches_focus(effective):
                    filtered.append(c)
            chunks = filtered

        if not chunks:
            return (
                f"No PRD sections matched the incremental focus list for `{prd_source_id}`. "
                "Verify heading text matches changed sections from the diff."
            )

        doc_title = chunks[0].get("doc_title", prd_source_id)
        preamble = ""
        if focus_headings:
            preamble = (
                "**Incremental analysis:** Only changed or new PRD sections are shown below; "
                "unchanged sections already have decisions carried forward.\n\n---\n\n"
            )

        # Group chunks under their effective heading, preserving document order. Chunks
        # after a heading inherit it, matching how the chunker emits them.
        ordered_headings: list[str] = []
        by_heading: dict[str, list[dict]] = {}
        cur = ""
        for s in chunks:
            # Sanitise per chunk, before the join below. Doing it after would rewrite
            # a carried table header in one chunk but not its neighbour, and the
            # header de-duplication compares those headers for equality.
            h = _sanitize_prd_content((s.get("section_heading") or "").strip())
            if h:
                cur = h
            key = cur or "(no heading)"
            if key not in by_heading:
                by_heading[key] = []
                ordered_headings.append(key)
            by_heading[key].append(s)

        def _render(heading: str) -> str:
            from ingestion.markdown_table import join_chunk_texts

            # A table spanning several chunks repeats its header in each one so the
            # chunk stands alone in retrieval. Re-joined into one document that
            # header is duplication that splits one table into several.
            body = join_chunk_texts([
                _sanitize_prd_content(s.get("chunk_text") or "")
                for s in by_heading[heading]
            ])
            return f"### {heading}\n{body}" if heading != "(no heading)" else body

        # ── Single-section request ──
        if section and section.strip():
            want = _normalize_heading_for_coverage(section)
            match = next(
                (h for h in ordered_headings if _normalize_heading_for_coverage(h) == want),
                None,
            )
            if match is None:
                listing = "\n".join(f"  - {h}" for h in ordered_headings[:60])
                return (
                    f"No section matching {section!r} in `{prd_source_id}`.\n"
                    f"Available sections:\n{listing}"
                )
            return f"# {doc_title} — section\n\n{_render(match)}"

        # ── Full document, bounded by the context budget ──
        rendered = [_render(h) for h in ordered_headings]
        full = f"# {doc_title}\n\n{preamble}" + "\n\n---\n\n".join(rendered)
        if _count_tokens_str(full) <= _PRD_BUDGET:
            return full

        # Too large: return the outline plus as many opening sections as fit, and tell the
        # agent how to get the rest. Returning the whole thing would silently truncate the
        # prompt on Ollama and force destructive compaction on hosted providers.
        outline = "\n".join(
            f"  {i + 1}. {h}  (~{_count_tokens_str(_render(h)):,} tokens)"
            for i, h in enumerate(ordered_headings)
        )
        header = (
            f"# {doc_title}\n\n{preamble}"
            f"**This PRD is too large to return in full "
            f"(~{_count_tokens_str(full):,} tokens, budget {_PRD_BUDGET:,}).**\n"
            f"Outline of all {len(ordered_headings)} sections:\n{outline}\n\n"
            "The opening sections are included below. To read any other section verbatim, "
            "call read_prd_document with that section's heading. Do not assume a section is "
            "absent because it is not shown here.\n\n---\n\n"
        )
        budget_left = _PRD_BUDGET - _count_tokens_str(header)
        included: list[str] = []
        for h in ordered_headings:
            block = _render(h)
            cost = _count_tokens_str(block)
            if cost > budget_left:
                break
            included.append(block)
            budget_left -= cost
        logger.warning(
            "[%s] PRD %s exceeds the %s-token read budget — returned outline + %s/%s sections",
            run_id, prd_source_id, _PRD_BUDGET, len(included), len(ordered_headings),
        )
        return header + "\n\n---\n\n".join(included)

    @tool
    def search_tests(query: str, top_k: int = 20,
                     extra_queries: list[str] | None = None) -> str:
        """
        Search for existing test cases relevant to a feature or requirement.
        Uses hybrid semantic + keyword search.

        Args:
            query:         A description of the feature or requirement to find tests for.
                           Be specific — e.g. "one-step checkout with saved address" not
                           just "checkout".
            top_k:         Number of results to return (default 20, max 50).
            extra_queries: Optional alternative phrasings of the SAME requirement, up to
                           three. Results from all of them are fused, and a test that
                           surfaces for several phrasings ranks higher. Use this when
                           one wording might miss tests written in different vocabulary
                           — e.g. query="checkout with promo code" with
                           extra_queries=["discount code at payment", "voucher applied
                           to order"]. Do NOT put a different requirement here; search
                           for that separately.

        Returns:
            A numbered list of matching test cases with jira_key, summary, module, and labels.
        """
        top_k = min(top_k, 50)
        # Over-retrieve for reranking (3x candidates), then rerank down to top_k
        retrieval_k = top_k * 3 if reranker else top_k

        queries = [query] + [q for q in (extra_queries or []) if q and q.strip()][:3]

        result_lists: list[list[dict]] = []
        for q in queries:
            result_lists.append(es_store.search_hybrid(
                query_embedding=embed_client.embed_query(q),
                keyword_query=q,
                top_k=retrieval_k,
                module_filter=module,
            ))

        if len(result_lists) > 1:
            results = _rrf_merge_results(result_lists)[:retrieval_k]
        else:
            results = result_lists[0]

        if not results:
            return "No matching tests found."
        # Cross-encoder reranking for precision. Scored against the primary query — the
        # variants exist to widen recall, not to redefine what was asked for.
        if reranker:
            results = reranker.rerank(query, results, top_k=top_k)

        # ── Guardrail: track all retrieved jira_keys ──
        for r in results:
            _seen_jira_keys.add(r["jira_key"])

        lines = []
        for i, r in enumerate(results, 1):
            labels = ", ".join(r.get("labels") or []) or "none"
            desc_preview = (r.get("description") or "")[:120]
            lines.append(
                f"{i}. [{r['jira_key']}] {r['summary']}\n"
                f"   Module: {r.get('module','?')} | Score: {r['score']:.3f} | Labels: {labels}"
                + (f"\n   Description: {desc_preview}" if desc_preview else "")
            )
        return "\n".join(lines)

    @tool
    def get_test_details(jira_key: str) -> str:
        """
        Fetch the full details of a specific test case including all steps.
        Use this when a test looks potentially relevant but you need more context
        before deciding keep / update / deprecate.

        Args:
            jira_key:  The Jira key of the test case, e.g. "PROJ-12345"
        """
        resp = es_store._client.search(
            index="qa_test_cases",
            query={"term": {"jira_key": jira_key}},
            source=["jira_key", "summary", "module", "folder_path", "labels",
                    "description", "steps_text", "preconditions"],
            size=1,
        )
        hits = resp["hits"]["hits"]
        if not hits:
            return f"Test case {jira_key!r} not found in the index."

        # ── Guardrail: track this key as seen ──
        _seen_jira_keys.add(jira_key)

        t = hits[0]["_source"]
        return (
            f"Key:          {t.get('jira_key')}\n"
            f"Summary:      {t.get('summary')}\n"
            f"Module:       {t.get('module')}\n"
            f"Folder:       {t.get('folder_path')}\n"
            f"Labels:       {', '.join(t.get('labels') or [])}\n"
            f"Precondition: {t.get('preconditions') or 'N/A'}\n"
            f"Description:  {t.get('description') or 'N/A'}\n\n"
            f"Steps:\n{t.get('steps_text') or 'No steps recorded'}"
        )

    @tool
    def record_decision(
        action: str,
        reason: str,
        jira_key: str | None = None,
        suggested_changes: str | None = None,
        new_test_summary: str | None = None,
        new_test_steps: str | None = None,
        prd_section: str | None = None,
        updated_steps: list[dict] | None = None,
        confidence: str | None = None,
    ) -> str:
        """
        Record a coverage decision for human review.
        Call this for every keep / update / deprecate / create conclusion you reach.

        Args:
            action:             One of: "keep", "update", "deprecate", "create", "question"
            reason:             Clear explanation of why you made this decision
            jira_key:           The test case key (required for keep/update/deprecate; omit for create)
            suggested_changes:  For "update" — describe in prose exactly what needs changing.
                                This is recorded as a Jira comment for a human to apply; it is
                                never converted into test steps.
            new_test_summary:   For "create" — the proposed test case title
            new_test_steps:     For "create" — outline of the test steps. Number them
                                ("1. …", "2. …", one per line) so they import as separate
                                steps; unnumbered prose becomes the test description instead.
            prd_section:        Which PRD section triggered this decision (optional but helpful)
            updated_steps:      For "update" — ONLY when you can supply the COMPLETE new step
                                list. Xray replaces all steps with what you provide, so a
                                partial list permanently deletes the rest. Each item is
                                {"action": "...", "data": "...", "expectedResult": "..."}.
                                If you are not rewriting every step, leave this out and
                                describe the change in suggested_changes instead.
            confidence:         How sure you are: "high", "medium" or "low". Reviewers
                                sort by this, so be honest — "low" on a genuinely
                                ambiguous call is more useful than false certainty.
        """
        valid_actions = ("keep", "update", "deprecate", "create", "question")
        if action not in valid_actions:
            return f"Error: action must be one of {valid_actions}"

        # ── Guardrail: the reason is the whole audit trail ──
        reason_error = _check_reason_quality(action, reason)
        if reason_error:
            return reason_error
        if action in ("keep", "update", "deprecate") and not jira_key:
            return f"Error: jira_key is required for action={action!r}"
        if action == "update":
            if suggested_changes is None:
                return "Error: suggested_changes is required for action='update'"
            if not (suggested_changes.strip() if isinstance(suggested_changes, str) else str(suggested_changes).strip()):
                return "Error: suggested_changes cannot be empty for action='update'"
        if action == "create" and not new_test_summary:
            return "Error: new_test_summary is required for action='create'"
        if action == "create" and jira_key:
            return (
                "Error: jira_key must be omitted for CREATE — use 'update' instead if this test "
                "already exists in Xray."
            )

        # ── Guardrail: validate jira_key was actually retrieved ──
        if jira_key and jira_key not in _seen_jira_keys:
            return (
                f"Error: {jira_key!r} was not returned by search_tests or get_test_details. "
                f"You can only record decisions for tests you have actually retrieved. "
                f"Search for this test first, or verify the key is correct."
            )

        ident = _decision_identity(action, jira_key, prd_section, new_test_summary)

        def _advise(guard: str, verdict: str | None) -> str | None:
            """
            Return an advisory once per decision identity, then stand aside.

            These guards tell the agent to go and check something, then record the
            decision again. Re-running the same lookup would produce the same advisory
            forever, so the instruction would be impossible to follow and the decision
            could never be recorded. Firing once makes "look, then re-submit" real,
            while still forcing the agent past the evidence at least once.
            """
            if not verdict:
                return None
            key = (guard, ident)
            if key in _advised:
                return None
            _advised.add(key)
            return verdict

        # ── Guardrail: don't create a test that already exists ──
        # Runs before the identity check so a blocked CREATE is not recorded as decided
        # and the agent can still record an UPDATE for the same section.
        if action == "create" and new_test_summary:
            try:
                dup_vec = embed_client.embed_query(new_test_summary)
                similar = es_store.search_similar_tests(
                    dup_vec, top_k=3, module_filter=module, min_score=_DUP_WARN_SCORE,
                )
                verdict = _duplicate_verdict(similar)
            except Exception as exc:
                # A failed lookup must not block a legitimate CREATE.
                logger.warning("CREATE duplicate check failed (allowing): %s", exc)
                verdict = None
            if verdict:
                # A near-identical match is refused outright and never clears: the agent
                # is told to record an UPDATE instead, which it can still do.
                if "BLOCKED" in verdict:
                    return verdict
                advisory = _advise("duplicate", verdict)
                if advisory:
                    return advisory

        # ── Guardrail: is this test covered by a different document? ──
        # The agent sees one PRD at a time, so "the feature is not in this document" is
        # not "the feature is undocumented". Deprecating removes the test everywhere.
        if action == "deprecate" and jira_key:
            try:
                test_vec = es_store.get_test_embedding(jira_key)
                cross = _cross_prd_verdict(
                    es_store.search_similar_prd_chunks(
                        test_vec, top_k=_CROSS_PRD_TOP_K,
                    ) if test_vec else [],
                    prd_source_id,
                )
            except Exception as exc:
                logger.warning("Cross-document check failed (allowing): %s", exc)
                cross = None
            advisory = _advise("cross_prd", cross)
            if advisory:
                return advisory

        # ── Guardrail: one decision per (jira_key, prd section) or per create identity ──
        if ident in _decided_identities:
            prev_action = _decided_identities[ident]
            return (
                f"Warning: You already recorded {prev_action.upper()} for this "
                f"PRD section / test pairing. Skipping duplicate {action.upper()} decision. "
                f"If you need to change the decision, note it as a QUESTION instead."
            )

        updated_content = None
        if action == "update" and (suggested_changes or updated_steps):
            updated_content = {}
            if suggested_changes:
                updated_content["suggested_changes"] = suggested_changes
            if updated_steps:
                # Stored as-is; writeback validates the shape and refuses to send anything
                # it cannot verify, because Xray replaces the whole step array.
                updated_content["steps"] = updated_steps
        elif action == "create":
            updated_content = {
                "summary":         new_test_summary,
                "suggested_steps": new_test_steps,
            }

        if prd_section is not None and len(prd_section) > 500:
            prd_section = prd_section[:500]

        pg_store.write_decision({
            "run_id":          run_id,
            "jira_key":        jira_key,
            "action":          action,
            "reason":          reason,
            "updated_content": updated_content,
            "questions":       [reason] if action == "question" else None,
            "prd_source":      prd_source_id,
            "prd_section":     prd_section,
            "confidence":      confidence,
        })

        # ── Guardrail: track this decision ──
        _decided_identities[ident] = action
        _decision_count[action] = _decision_count.get(action, 0) + 1

        display_key = jira_key or repr(new_test_summary) or "(no key)"
        total = sum(_decision_count.values())
        return f"Decision recorded: {action.upper()} {display_key} (total decisions so far: {total})"

    @tool
    def get_prior_decisions(jira_key: str) -> str:
        """
        Look up the history of coverage decisions made for a specific test case
        in previous analysis runs. Use this when you want to know whether this
        test has already been reviewed, updated, or deprecated in past runs.

        Args:
            jira_key: The Jira key of the test case, e.g. "PROJ-12345"
        """
        rows = pg_store.get_decisions_by_jira_key(jira_key, limit=5)
        if not rows:
            return f"No prior decisions found for {jira_key}."

        lines = [f"Prior decisions for {jira_key} (most recent first):"]
        for r in rows:
            approved_str = {True: "approved", False: "rejected", None: "pending review"}.get(r.get("approved"), "?")
            lines.append(
                f"  [{r['created_at'].strftime('%Y-%m-%d') if hasattr(r['created_at'], 'strftime') else str(r['created_at'])[:10]}] "
                f"action={r['action'].upper()} status={approved_str} "
                f"section={r.get('prd_section') or 'N/A'}\n"
                f"  reason: {(r.get('reason') or '')[:200]}"
            )
        return "\n".join(lines)

    _knowledge_reads: set[str] = set()

    @tool
    def read_knowledge(path: str) -> str:
        """
        Read a curated QA knowledge document (test-authoring guidelines, deprecation rules,
        PRD interpretation guidance). Call this BEFORE proposing a CREATE or DEPRECATE so
        your proposal follows team conventions.

        Args:
            path: File name from the bundle, e.g. "index.md", "test-case-guidelines.md",
                  "deprecation-rules.md", "prd-to-knowledge-base.md".
                  Call with "index.md" first if you are unsure what is available.
        """
        available = _knowledge_files()
        name = (path or "").strip().strip("/")
        # Allowlist by exact basename — never join user input into a filesystem path.
        if name not in available:
            suggestion = ", ".join(available) or "(bundle is empty)"
            return (
                f"Unknown knowledge document {path!r}. Available documents: {suggestion}. "
                "Call read_knowledge('index.md') for descriptions of each."
            )

        if name in _knowledge_reads:
            return (
                f"You already read {name} during this run — reuse what it said rather than "
                "re-reading it. Call read_knowledge('index.md') to see other documents."
            )

        try:
            body = _strip_frontmatter((_PROMPTS_DIR / name).read_text())
        except OSError as e:
            return f"Could not read {name}: {e}"

        _knowledge_reads.add(name)
        return f"# Knowledge document: {name}\n\n{body}"

    @tool
    def list_prd_sections() -> str:
        """
        List every section in the PRD with its size and whether it is testable.

        Call this FIRST, before read_prd_document. It is small enough to keep in
        context for the whole run, so it doubles as your checklist: every section
        marked [testable] needs at least one decision by the end.

        Meta sections (Background, Success Metrics, Glossary and similar) are marked
        [meta] and need no decision.
        """
        from elasticsearch import helpers as es_helpers

        chunks: list[dict] = []
        for hit in es_helpers.scan(
            es_store._client,
            index="qa_prd_chunks",
            query={"query": {"term": {"source_id": prd_source_id}},
                   "_source": ["section_heading", "chunk_text", "chunk_index",
                               "chunk_type", "doc_title"]},
            scroll="2m",
            size=500,
        ):
            chunks.append(hit["_source"])
            if len(chunks) >= MAX_PRD_CHUNKS:
                logger.warning(
                    "PRD %s has >%s chunks — section list truncated",
                    prd_source_id, MAX_PRD_CHUNKS,
                )
                truncated = True
                break
        else:
            truncated = False

        if not chunks:
            return (
                f"No document found for source_id={prd_source_id!r}. "
                "It must be ingested before it can be analysed."
            )

        chunks.sort(key=lambda c: c.get("chunk_index", 0))

        # Chunks after a heading inherit it, matching how the chunker emits them.
        ordered: list[str] = []
        words: dict[str, int] = {}
        kinds: dict[str, set] = {}
        current = ""
        for c in chunks:
            heading = (c.get("section_heading") or "").strip()
            if heading:
                current = heading
            key = current or "(no heading)"
            if key not in words:
                words[key] = 0
                kinds[key] = set()
                ordered.append(key)
            words[key] += len((c.get("chunk_text") or "").split())
            if c.get("chunk_type"):
                kinds[key].add(c["chunk_type"])

        if _focus_norm is not None:
            ordered = [h for h in ordered if _chunk_heading_matches_focus(h)]
            if not ordered:
                return (
                    "No sections matched the incremental focus list. Verify the heading "
                    "text matches the changed sections from the diff."
                )

        doc_title = chunks[0].get("doc_title") or prd_source_id
        lines = [f"# {doc_title} — {len(ordered)} sections", ""]
        testable = 0
        for i, heading in enumerate(ordered, 1):
            meta = _is_meta_heading(heading)
            if not meta:
                testable += 1
            tag = "[meta]" if meta else "[testable]"
            # Content mix is a parsing signal: a section you expect to be a spec table
            # showing only prose means the source conversion flattened it.
            mix = "/".join(sorted(kinds[heading])) if kinds[heading] else "?"
            lines.append(f"{i:>3}. {tag:<10} {heading}  (~{words[heading]} words, {mix})")

        lines += [
            "",
            f"{testable} testable section(s) need a decision; "
            f"{len(ordered) - testable} are meta and do not.",
        ]
        if truncated:
            lines.append(
                f"WARNING: this document exceeds {MAX_PRD_CHUNKS} chunks, so sections "
                "beyond that point are NOT listed above. Say so in your final summary — "
                "do not report coverage as complete."
            )
        lines.append(
            "Next: call read_prd_document to read the content, then work through the "
            "testable sections in order."
        )
        return "\n".join(lines)

    return [
        list_prd_sections,
        read_prd_document,
        search_tests,
        get_test_details,
        record_decision,
        get_prior_decisions,
        read_knowledge,
    ]


# ─── Agentic loop ──────────────────────────────────────────────────────────────

# Errors that indicate infrastructure failure — should not be retried by the LLM
try:
    import psycopg2
    _FATAL_ERROR_TYPES = (ConnectionError, OSError, psycopg2.OperationalError)
except ImportError:
    _FATAL_ERROR_TYPES = (ConnectionError, OSError)

# How many times to re-prompt a model that replies with prose instead of a tool call
# before accepting the turn as finished. Keep low — each nudge costs a full turn.
_MAX_NO_TOOL_NUDGES = int(os.environ.get("QA_AGENT_MAX_NO_TOOL_NUDGES", "2"))


def _mentions_tool_name(text: str, tools_by_name: dict) -> bool:
    """True if the reply names a tool — a hint the model is narrating calls, not making them."""
    lowered = (text or "").lower()
    return any(name.lower() in lowered for name in tools_by_name)


# ─── Token / cost accounting ─────────────────────────────────────────────────
#
# USD per 1M tokens, (uncached_input, output). Deliberately incomplete: an unknown model
# reports token counts with cost omitted rather than a confidently wrong number.
# Override or extend without a code change via QA_MODEL_PRICES, e.g.
#   QA_MODEL_PRICES="my-model:3.0:15.0,other:1:5"
_MODEL_PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-opus-4-8":   (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5":  (1.00,  5.00),
}
# Anthropic cache multipliers: reads ~0.1x input, writes ~1.25x input.
_CACHE_READ_MULT = 0.1
_CACHE_WRITE_MULT = 1.25


def _model_prices(model: str) -> tuple[float, float] | None:
    for entry in os.environ.get("QA_MODEL_PRICES", "").split(","):
        parts = entry.strip().split(":")
        if len(parts) == 3 and parts[0].strip() == model:
            try:
                return float(parts[1]), float(parts[2])
            except ValueError:
                logger.warning("Malformed QA_MODEL_PRICES entry: %r", entry)
    return _MODEL_PRICES_PER_MTOK.get(model)


def _new_usage() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "llm_calls": 0,
        "usage_reported_calls": 0,
    }


def _accumulate_usage(usage: dict, response) -> None:
    """
    Fold one response's usage_metadata into the running total.

    LangChain normalises usage_metadata across providers, but not every provider or
    version populates it — usage_reported_calls vs llm_calls tells you whether the
    totals cover the whole run or only part of it.
    """
    usage["llm_calls"] += 1
    meta = getattr(response, "usage_metadata", None) or {}
    if not meta:
        return
    usage["usage_reported_calls"] += 1
    usage["input_tokens"] += int(meta.get("input_tokens") or 0)
    usage["output_tokens"] += int(meta.get("output_tokens") or 0)
    details = meta.get("input_token_details") or {}
    usage["cache_read_tokens"] += int(details.get("cache_read") or 0)
    usage["cache_write_tokens"] += int(details.get("cache_creation") or 0)


def _finalize_usage(usage: dict, provider: str, model: str) -> dict:
    """Attach derived fields: cost estimate, cache hit rate, and coverage caveats."""
    out = dict(usage)
    out["provider"] = provider
    out["model"] = model

    if provider == "ollama":
        # Local inference — no per-token charge. Tokens still matter for latency.
        out["estimated_cost_usd"] = 0.0
        out["cost_basis"] = "local_inference"
    else:
        prices = _model_prices(model)
        if prices:
            in_price, out_price = prices
            # input_tokens excludes cached reads/writes on Anthropic; price each tier.
            cost = (
                usage["input_tokens"] * in_price
                + usage["cache_read_tokens"] * in_price * _CACHE_READ_MULT
                + usage["cache_write_tokens"] * in_price * _CACHE_WRITE_MULT
                + usage["output_tokens"] * out_price
            ) / 1_000_000
            out["estimated_cost_usd"] = round(cost, 4)
            out["cost_basis"] = "model_price_table"
        else:
            out["estimated_cost_usd"] = None
            out["cost_basis"] = f"unknown_model:{model} (set QA_MODEL_PRICES to price it)"

    billable_input = usage["input_tokens"] + usage["cache_read_tokens"]
    out["cache_hit_ratio"] = (
        round(usage["cache_read_tokens"] / billable_input, 3) if billable_input else None
    )
    if usage["llm_calls"] and usage["usage_reported_calls"] < usage["llm_calls"]:
        out["warning"] = (
            f"usage_metadata missing on {usage['llm_calls'] - usage['usage_reported_calls']}"
            f"/{usage['llm_calls']} LLM calls — totals are a lower bound"
        )
    return out


_TRIAGE_TURNS_AHEAD = int(os.environ.get("QA_AGENT_TRIAGE_TURNS_AHEAD", "10"))


def _triage_turn(max_turns: int, ahead: int = None) -> int | None:
    """
    Turn index at which to warn, or None when the run is too short to be worth it.

    Naively using `max_turns - ahead` warns on turn 0 whenever max_turns <= ahead —
    telling the agent to stop starting new work before it has done any, so it records
    nothing at all. That is a realistic configuration: .env.example recommends
    QA_AGENT_MAX_TURNS=20 for local models and lowering it further for slow ones.

    Short runs still get a warning, just proportionally later.
    """
    ahead = _TRIAGE_TURNS_AHEAD if ahead is None else ahead
    if max_turns <= 2:
        return None
    if max_turns > ahead:
        return max_turns - ahead
    return max(1, int(max_turns * 0.7))


def _triage_message(turns_left: int) -> str:
    """
    The message injected before the turn limit.

    Names what to do with the time left rather than only that time is short: finish
    the sections already in context, and record a QUESTION for anything unreachable
    so an unanalysed section is visible as a question instead of silently absent.
    """
    return (
        f"TURN LIMIT APPROACHING — {turns_left} turns remain in this run.\n"
        "Change strategy now:\n"
        "1. Do not start new searches or read new sections.\n"
        "2. Record decisions for the sections already in your context.\n"
        "3. For any testable section you cannot reach, record action='question' "
        "naming the section and saying it was not analysed — an unanalysed section "
        "must not look like a section you found nothing wrong with.\n"
        "4. Prioritise by impact: payment, authentication and other core flows first."
    )


def _run_agent_loop(
    llm_with_tools,
    tools_by_name: dict,
    messages: list,
    max_turns: int = 30,
    provider: str = "anthropic",
    model: str = "",
) -> tuple[list, LoopStatus, dict]:
    """
    Guarded ReAct-style tool-calling loop with:
      - Token budget tracking (warns agent when approaching context limit)
      - Message compaction (summarises old tool results to free space)
      - Turn-level logging with token counts

    LLM response → if tool_calls → execute → append result → repeat
    Stops when LLM produces no more tool calls or max_turns is reached.

    Smaller local models (ollama) frequently end a turn with prose describing a tool
    call instead of emitting one. Rather than treating that as "done" and returning a
    run with zero decisions, the loop nudges once per _MAX_NO_TOOL_NUDGES and only
    then gives up — so a genuine finish and a tool-calling failure stay distinguishable.
    """
    context_limit = _context_limit(provider, model)
    warning_threshold = int(context_limit * _TOKEN_WARNING_RATIO)
    compact_threshold = int(context_limit * _TOKEN_COMPACT_RATIO)
    budget_warned = False
    recorded = 0        # decisions successfully persisted via record_decision
    nudges_used = 0
    usage = _new_usage()

    triage_warned = False
    triage_turn = _triage_turn(max_turns)

    for turn in range(max_turns):
        messages = _shrink_stale_prd_reads(messages)

        # ── Triage warning ──
        # Hitting max_turns mid-analysis truncates wherever the agent happened to be,
        # so the sections it never reached are indistinguishable from sections it
        # decided were fine. Warning ahead of the wall lets it spend the remaining
        # turns closing out rather than starting new work.
        if not triage_warned and triage_turn is not None and turn == triage_turn:
            turns_left = max_turns - turn
            messages.append(HumanMessage(content=_triage_message(turns_left)))
            triage_warned = True
            logger.info(
                "Agent turn %s: triage warning injected — %s turns remaining",
                turn + 1, turns_left,
            )

        # ── Token budget check ──
        current_tokens = _count_message_tokens(messages)

        if current_tokens >= compact_threshold:
            logger.warning(
                f"Agent turn {turn + 1}: {current_tokens} tokens >= compact threshold "
                f"({compact_threshold}). Compacting old messages."
            )
            messages = _compact_old_messages(messages, keep_recent=8)
            current_tokens = _count_message_tokens(messages)
            logger.info(f"After compaction: {current_tokens} tokens")

        if current_tokens >= warning_threshold and not budget_warned:
            # Inject a system warning so the agent knows to wrap up
            messages.append(HumanMessage(content=(
                f"⚠️ TOKEN BUDGET WARNING: You have used {current_tokens:,} of {context_limit:,} tokens "
                f"({current_tokens * 100 // context_limit}%). "
                f"Please finish your analysis and record remaining decisions promptly. "
                f"Prioritise the most important uncovered requirements."
            )))
            budget_warned = True
            logger.warning(f"Token budget warning injected at {current_tokens} tokens")

        response = llm_with_tools.invoke(messages)
        _accumulate_usage(usage, response)
        messages.append(response)

        if not response.tool_calls:
            # No tool calls. Either the agent is genuinely finished, or the model
            # narrated a tool call instead of emitting one (common below ~30B).
            if recorded == 0 and nudges_used < _MAX_NO_TOOL_NUDGES:
                nudges_used += 1
                text = message_text(response)
                logger.warning(
                    "Agent turn %s: no tool calls and no decisions recorded yet "
                    "(nudge %s/%s). Model reply began: %.200s",
                    turn + 1,
                    nudges_used,
                    _MAX_NO_TOOL_NUDGES,
                    text.strip(),
                )
                if _mentions_tool_name(text, tools_by_name):
                    logger.error(
                        "Model appears to be describing tool calls in prose rather than "
                        "emitting them. If this repeats, %r likely lacks native tool-calling "
                        "support in its Ollama template.",
                        provider,
                    )
                messages.append(HumanMessage(content=(
                    "You did not call any tool. Describing a tool call in prose has no effect — "
                    "only actual tool calls are executed, and only `record_decision` persists a "
                    "finding. Continue now by emitting a real tool call. If you have finished the "
                    "analysis, call `record_decision` for each conclusion you reached before stopping."
                )))
                continue
            return messages, LoopStatus.COMPLETED, usage

        for idx, tc in enumerate(response.tool_calls):
            # Some backends (notably Ollama) omit or blank the tool_call id; an empty
            # id breaks tool_use/tool_result pairing on the next request.
            tc_id = tc.get("id") or f"call_{turn}_{idx}"
            tool_fn = tools_by_name.get(tc["name"])
            if tool_fn is None:
                result = (
                    f"Unknown tool: {tc['name']}. "
                    f"Available tools: {', '.join(sorted(tools_by_name))}."
                )
            else:
                try:
                    result = tool_fn.invoke(tc["args"])
                except _FATAL_ERROR_TYPES as e:
                    # Infrastructure failure — abort the loop, don't let LLM retry
                    logger.error(f"Fatal tool error in {tc['name']}: {e}")
                    raise
                except Exception as e:
                    # Usually a schema violation from a weaker model. Return the error as
                    # the tool result so the model can repair its arguments next turn.
                    result = f"Tool error: {e}"
                    logger.warning(f"Tool {tc['name']} failed: {e}")

            if tc["name"] == "record_decision" and str(result).startswith("Decision recorded"):
                recorded += 1

            messages.append(ToolMessage(content=str(result), tool_call_id=tc_id))

        logger.info(
            f"Agent turn {turn + 1}/{max_turns}: "
            f"{len(response.tool_calls)} tool calls, ~{current_tokens:,} tokens used"
        )

    logger.warning(f"Agent reached max_turns={max_turns} without finishing")
    return messages, LoopStatus.MAX_TURNS_REACHED, usage


# ─── Main pipeline ─────────────────────────────────────────────────────────────

async def run_analysis(
    prd_source_id: str,
    module: list[str] | None,
    embed_client: EmbedClient,
    es_store: ESStore,
    pg_store: PGStore,
    run_id: str | None = None,
    top_k: int = 25,
    provider: str = "azure_openai",
    model: str = "gpt-4o",
    max_turns: int | None = None,
    reranker=None,
    focus_headings: list[str] | None = None,
    finalize_run: bool = True,
) -> dict:
    """
    Run the tool-using coverage analysis agent for one PRD document.

    The agent reads the full document, searches for tests, and records decisions
    autonomously. Results land in Postgres pending_decisions for human review.
    """
    run_id = run_id or str(uuid.uuid4())
    max_turns = _default_max_turns() if max_turns is None else max_turns
    lock = _lock_for_run_id(run_id)
    async with lock:
        existing_row = pg_store.get_run(run_id)
        if existing_row and existing_row.get("status") in (
            "completed",
            "completed_empty",
            "completed_with_errors",
            "truncated",
        ):
            decisions_existing = pg_store.get_pending_decisions(run_id=run_id)
            logger.info(
                "[%s] Duplicate deterministic analysis request ignored; run already %s",
                run_id,
                existing_row.get("status"),
            )
            return {
                "run_id": run_id,
                "status": existing_row.get("status"),
                "prd_source_id": prd_source_id,
                "provider": provider,
                "model": model,
                "module_filter": module,
                "decisions_made": len(decisions_existing),
                "coverage_score": (existing_row.get("run_metadata") or {}).get("coverage_score")
                if isinstance(existing_row.get("run_metadata"), dict)
                else None,
                "agent_turns": 0,
                "elapsed_s": 0.0,
                "loop_status": "already_completed",
                "token_usage": (existing_row.get("run_metadata") or {}).get("token_usage")
                if isinstance(existing_row.get("run_metadata"), dict)
                else None,
                "verification_hybrid_hits": (existing_row.get("run_metadata") or {}).get("verification_hybrid_hits")
                if isinstance(existing_row.get("run_metadata"), dict)
                else None,
            }

        started = time.time()

        pg_store.start_run(run_id, run_type="analysis", prd_source=prd_source_id)
        logger.info(f"[{run_id}] Analysis started: prd={prd_source_id!r} provider={provider}/{model}")

        try:
            loop = asyncio.get_running_loop()

            # Build tools and LLM — done in executor since LLM init may do I/O
            tools = _make_tools(
                prd_source_id,
                module,
                embed_client,
                es_store,
                pg_store,
                run_id,
                reranker=reranker,
                focus_headings=focus_headings,
                # A single PRD read may take at most this share of the conversation budget,
                # leaving room for tool results, search hits and the model's own output.
                prd_token_budget=int(
                    _context_limit(provider, model) * _PRD_READ_BUDGET_RATIO
                ),
            )
            llm   = _build_llm(provider, model)
            llm_with_tools = llm.bind_tools(tools)
            tools_by_name  = {t.name: t for t in tools}

            # Kick off the agent
            focus_note = ""
            if focus_headings:
                listed = "\n".join(f"  - {h}" for h in focus_headings)
                focus_note = (
                    f"\nThis is an incremental re-analysis. The following PRD sections changed "
                    f"since the last run — prioritise analysing these sections and the tests "
                    f"that cover them:\n{listed}\n"
                    f"Unchanged sections already have decisions carried forward.\n"
                    f"`read_prd_document` returns **only** these sections — do not assume other sections are missing.\n"
                )
            messages = [
                _build_system_message(provider),
                HumanMessage(content=(
                    f"Analyse PRD document `{prd_source_id}` for test coverage.\n"
                    + (f"Focus on tests in module(s): {module}.\n" if module else "")
                    + focus_note
                    + "Start by calling read_prd_document, then proceed with your analysis."
                )),
            ]

            final_messages, loop_status, raw_usage = await loop.run_in_executor(
                None,
                lambda: _run_agent_loop(
                    llm_with_tools, tools_by_name, list(messages), max_turns,
                    provider=provider, model=model,
                ),
            )
            token_usage = _finalize_usage(raw_usage, provider, model)

            # Post-loop store sanity check: confirms index is reachable and populated.
            def _verification_hits() -> int:
                if module:
                    return int(es_store._client.count(
                        index="qa_test_cases",
                        query={"terms": {"module": module}},
                    )["count"])
                return int(es_store._client.count(index="qa_test_cases", query={"match_all": {}})["count"])

            verification_hits = await loop.run_in_executor(None, _verification_hits)

            # Count decisions written to Postgres for this run
            decisions = pg_store.get_pending_decisions(run_id=run_id)
            total_decisions = len(decisions)

            # Count actual LLM invocations (assistant messages only)
            llm_turns = sum(1 for m in final_messages if hasattr(m, 'tool_calls') or (hasattr(m, 'type') and m.type == 'ai'))

            # Coverage score: fraction of PRD sections that have at least one
            # actionable decision (keep/update/create). Ranges 0.0–1.0.
            from agents.incremental import compute_prd_heading_hashes

            prd_hashes = compute_prd_heading_hashes(es_store, prd_source_id)
            # Denominator: testable sections only (same filter as analysis — exclude meta headings).
            prd_sections = {
                _normalize_heading_for_coverage(h.strip())
                for h in prd_hashes.keys()
                if not _is_meta_heading(h)
            }
            prd_sections.discard("")
            covered_sections = {
                _normalize_heading_for_coverage((d.get("prd_section") or "").strip())
                for d in decisions
                if d.get("action") in ("keep", "update", "create")
            }
            covered_sections.discard("")
            # Intersect. `covered_sections` comes from agent-authored prd_section strings,
            # which are free text: a paraphrased or invented section name would otherwise
            # inflate the numerator and could push coverage above 1.0. Only sections that
            # actually exist in the PRD count.
            matched_sections = covered_sections & prd_sections
            unmatched_labels = sorted(covered_sections - prd_sections)
            coverage_score = (
                round(len(matched_sections) / len(prd_sections), 3) if prd_sections else None
            )
            if unmatched_labels:
                # A direct measure of how often the agent invents or paraphrases section
                # names — worth seeing, because it also breaks incremental carry-forward.
                logger.warning(
                    "[%s] %s decision section label(s) match no PRD heading and are excluded "
                    "from coverage: %s",
                    run_id, len(unmatched_labels), unmatched_labels[:10],
                )

            if loop_status == LoopStatus.MAX_TURNS_REACHED:
                row_final_status = "truncated"
                logger.warning(
                    f"[{run_id}] Agent hit max_turns={max_turns} — marking run as truncated "
                    f"(decisions recorded: {total_decisions})"
                )
            elif total_decisions == 0:
                row_final_status = "completed_empty"
                logger.warning(
                    f"[{run_id}] Analysis finished with zero decisions — marking run as completed_empty"
                )
            else:
                row_final_status = "completed"

            run_metadata = normalize_json_obj({
                "prd_heading_hashes": prd_hashes,
                "coverage_score": coverage_score,
                "state_confidence": "full",
                "loop_status": loop_status.value,
                "verification_hybrid_hits": verification_hits,
                "module_filter": module,
                "token_usage": token_usage,
                # Sections the agent named that do not exist in the PRD. High values mean
                # coverage_score is measuring less than it appears to.
                "unmatched_section_labels": unmatched_labels[:50],
                "sections_total": len(prd_sections),
                "sections_covered": len(matched_sections),
            })

            try:
                from observability.metrics import metrics

                metrics.inc("analysis_input_tokens_total", token_usage["input_tokens"])
                metrics.inc("analysis_output_tokens_total", token_usage["output_tokens"])
                metrics.inc("analysis_cache_read_tokens_total", token_usage["cache_read_tokens"])
                metrics.inc("analysis_llm_calls_total", token_usage["llm_calls"])
                if token_usage.get("estimated_cost_usd") is not None:
                    metrics.inc("analysis_cost_usd_total", token_usage["estimated_cost_usd"])
                metrics.observe("analysis_duration_seconds", time.time() - started)
            except ImportError:
                pass

            decisions_for_fp = [
                {"action": d.get("action"), "jira_key": d.get("jira_key"), "prd_section": d.get("prd_section")}
                for d in decisions
            ]

            ledger_phase = (
                "analysis_truncated" if row_final_status == "truncated" else "analysis"
            )
            ledger_summary = normalize_json_obj({
                "prd_source_id": prd_source_id,
                "decisions_made": total_decisions,
                "decisions_sha256": fingerprint_sha256(decisions_for_fp),
                "run_row_status": row_final_status,
                "loop_status": loop_status.value,
                "verification_hybrid_hits": verification_hits,
                "token_usage": token_usage,
                "warning": "max_turns_reached" if loop_status == LoopStatus.MAX_TURNS_REACHED else None,
            })

            if finalize_run:
                pg_store.complete_run(
                    run_id,
                    decisions_made=total_decisions,
                    run_metadata=run_metadata,
                    final_status=row_final_status,
                )
                await append_entry_async(ledger_phase, run_id, ledger_summary)

            elapsed = round(time.time() - started, 1)
            cost = token_usage.get("estimated_cost_usd")
            logger.info(
                f"[layer=agent] [{run_id}] Done: {total_decisions} decisions in {elapsed}s "
                f"({llm_turns} LLM turns, coverage={coverage_score}, "
                f"loop={loop_status.value}, verify_hits={verification_hits}, row={row_final_status})"
            )
            logger.info(
                "[layer=cost] [%s] tokens in=%s out=%s cache_read=%s cache_write=%s "
                "hit_ratio=%s calls=%s cost=%s",
                run_id,
                token_usage["input_tokens"],
                token_usage["output_tokens"],
                token_usage["cache_read_tokens"],
                token_usage["cache_write_tokens"],
                token_usage.get("cache_hit_ratio"),
                token_usage["llm_calls"],
                f"${cost:.4f}" if cost is not None else "n/a",
            )
            if token_usage.get("warning"):
                logger.warning("[%s] token accounting: %s", run_id, token_usage["warning"])

            if row_final_status == "truncated":
                out_status = "truncated"
            elif row_final_status == "completed_empty":
                out_status = "completed_empty"
            else:
                out_status = "completed"

            return {
                "run_id":          run_id,
                "status":          out_status,
                "prd_source_id":   prd_source_id,
                "provider":        provider,
                "model":           model,
                "module_filter":   module,
                "decisions_made":  total_decisions,
                "coverage_score":  coverage_score,
                "agent_turns":     llm_turns,
                "elapsed_s":       elapsed,
                "loop_status":     loop_status.value,
                "verification_hybrid_hits": verification_hits,
                "token_usage":     token_usage,
            }

        except Exception as exc:
            elapsed = round(time.time() - started, 1)
            logger.exception(f"[{run_id}] Analysis failed after {elapsed}s: {exc}")
            pg_store.fail_run(run_id, str(exc))
            raise


# ─── Prompt preview (no LLM call) ─────────────────────────────────────────────

def build_preview(
    prd_source_id: str,
    module: list[str] | None,
    embed_client: EmbedClient,
    es_store: ESStore,
    provider: str,
    model: str,
    sample_queries: list[str] | None = None,
    reranker=None,
) -> dict[str, Any]:
    """
    Build and return everything that would be sent to the LLM — without calling it.
    Useful for inspecting/debugging the prompt before spending tokens.

    If sample_queries is provided, runs search_tests for each query using the same
    hybrid search the agent would use, so you can preview retrieval quality.

    Returns:
        {
          system_prompt:   str,
          user_message:    str,
          tools:           list of tool schemas (name, description, parameters),
          prd_document:    the full PRD text the agent would read via read_prd_document(),
          sample_searches: list of {query, results} — only present if sample_queries given,
        }
    """
    # Build tools (same as in run_analysis — no pg_store needed for preview)
    dummy_pg = type("_NoPG", (), {"write_decision": lambda *a, **k: None})()
    tools = _make_tools(
        prd_source_id, module, embed_client, es_store, dummy_pg, "preview",
        prd_token_budget=int(_context_limit(provider, model) * _PRD_READ_BUDGET_RATIO),
    )

    # Extract tool schemas directly from @tool definitions — no LLM/API key needed
    tool_schemas = []
    for t in tools:
        schema = {
            "name":        t.name,
            "description": t.description,
            "parameters":  t.args_schema.model_json_schema() if hasattr(t, "args_schema") and t.args_schema else {},
        }
        tool_schemas.append(schema)

    # Fetch the actual PRD document (same content read_prd_document would scroll)
    chunk_rows = _fetch_prd_chunks_ordered(
        es_store,
        prd_source_id,
        source_fields=["section_heading", "chunk_text", "doc_title"],
    )
    hits = [{"_source": c} for c in chunk_rows]
    # auto_queries: list of (label, rich_query_text) pairs
    # label = heading shown in output; rich_query = full chunk text for strong embedding signal
    auto_queries: list[tuple[str, str]] = []
    if hits:
        doc_title = hits[0]["_source"].get("doc_title", prd_source_id)
        sections  = []
        seen_headings: set[str] = set()
        for h in hits:
            s       = h["_source"]
            heading = _sanitize_prd_content((s.get("section_heading") or "").strip())
            # /analyze/preview promises to show the exact prompt that would be sent. If
            # it skipped sanitisation the preview would differ from the real thing
            # precisely on the content worth previewing.
            text    = _sanitize_prd_content(s.get("chunk_text", ""))
            sections.append(f"### {heading}\n{text}" if heading else text)
            # Skip meta-sections and near-empty chunks — they produce retrieval noise
            if (heading
                    and heading not in seen_headings
                    and not _is_meta_heading(heading)
                    and len(text.strip()) > 20):
                # Use heading + first 500 chars of text — enough signal, much faster on CPU
                rich_query = f"{heading}: {text.strip()[:500]}"
                auto_queries.append((heading, rich_query))
                seen_headings.add(heading)
        prd_text = f"# {doc_title}\n\n" + "\n\n---\n\n".join(sections)
    else:
        prd_text = f"(no chunks found for source_id={prd_source_id!r})"

    user_message = (
        f"Analyse PRD document `{prd_source_id}` for test coverage.\n"
        + (f"Focus on tests in module(s): {module}.\n" if module else "")
        + "Start by calling read_prd_document, then proceed with your analysis."
    )

    # ── Retrieval preview: show what search_tests + search_prd would return ──
    # If sample_queries provided: use them as-is (label = query text)
    # Otherwise: use heading+body pairs auto-derived from PRD chunks
    if sample_queries is not None:
        queries_to_run = [(q, q) for q in sample_queries]
    else:
        queries_to_run = auto_queries

    # RRF scores are rank-based: 1/(rank + 60). Use rerank_score thresholds when reranker is present.
    RERANK_HIGH   = 2.0     # cross-encoder logit ≥ 2.0 → strong match
    RERANK_MEDIUM = 0.5     # cross-encoder logit ≥ 0.5 → moderate match
    RRF_HIGH      = 0.025   # RRF score ≈ top-3 in both retrievers
    RRF_MEDIUM    = 0.012   # RRF score ≈ top-10 in at least one retriever
    KB_MIN_SCORE  = 0.77

    pool_size = es_store.estimate_pool_size(module)

    # Batch-embed all queries at once for efficiency
    if queries_to_run:
        rich_texts = [q for _, q in queries_to_run]
        query_vecs = embed_client.embed_queries(rich_texts)
    else:
        query_vecs = []

    # Deduplicated union across all queries (same as validate_prd_data)
    all_tests_preview: dict[str, dict] = {}
    retrieval_preview = []

    for (label, rich_query), query_vec in zip(queries_to_run, query_vecs):
        try:
            # Over-retrieve for reranking
            retrieval_k = min(pool_size, 100) if reranker else pool_size
            test_results = es_store.search_hybrid(
                query_embedding=query_vec,
                keyword_query=label,
                top_k=retrieval_k,
                module_filter=module,
            )

            # Cross-encoder reranking
            if reranker and test_results:
                test_results = reranker.rerank(label, test_results, top_k=50)

            kb_raw = es_store.search_similar_prd_chunks(query_embedding=query_vec, top_k=7, module_filter=module)
            kb_results = [
                {
                    "source_id":       r["source_id"],
                    "doc_title":       r.get("doc_title"),
                    "section_heading": r.get("section_heading"),
                    "chunk_preview":   r.get("chunk_text", "")[:200],
                    "score":           round(r["score"], 4),
                }
                for r in kb_raw
                if r["source_id"] != prd_source_id and r["score"] >= KB_MIN_SCORE
            ][:5]

            # Use rerank_score for thresholding when available, else RRF score
            use_rerank = reranker and test_results and "rerank_score" in test_results[0]
            score_high = RERANK_HIGH if use_rerank else RRF_HIGH
            score_medium = RERANK_MEDIUM if use_rerank else RRF_MEDIUM
            score_key = "rerank_score" if use_rerank else "score"

            high_tests   = []
            medium_tests = []
            below_count  = 0
            for r in test_results:
                s = r.get(score_key, 0)
                confidence = "high" if s >= score_high else "medium"
                fmt = {
                    "jira_key":   r["jira_key"],
                    "summary":    r["summary"],
                    "module":     r.get("module"),
                    "labels":     r.get("labels") or [],
                    "score":      round(r.get("score", 0), 4),
                    "rerank_score": r.get("rerank_score"),
                    "confidence": confidence,
                }
                if s >= score_high:
                    high_tests.append(fmt)
                elif s >= score_medium:
                    medium_tests.append(fmt)
                else:
                    below_count += 1
                    continue

                key = r["jira_key"]
                best_score = all_tests_preview.get(key, {}).get(score_key, -999)
                if key not in all_tests_preview or s > best_score:
                    all_tests_preview[key] = {**fmt, "matched_queries": [label]}
                else:
                    if label not in all_tests_preview[key]["matched_queries"]:
                        all_tests_preview[key]["matched_queries"].append(label)

            retrieval_preview.append({
                "query":                       label,
                "test_matches":                high_tests + medium_tests,
                "test_matches_high":           len(high_tests),
                "test_matches_medium":         len(medium_tests),
                "test_matches_below_threshold": below_count,
                "kb_matches": kb_results,
            })
        except Exception as exc:
            retrieval_preview.append({"query": label, "error": str(exc)})

    unique_preview = sorted(all_tests_preview.values(), key=lambda x: x["score"], reverse=True)
    return {
        # Provider-specific: local providers get the lean prompt with knowledge read on demand.
        "system_prompt":           _system_prompt(provider),
        "knowledge_mode":          "inline" if provider in _CACHED_PREFIX_PROVIDERS else "on_demand",
        "user_message":            user_message,
        "tools":                   tool_schemas,
        "prd_document":            prd_text,
        "prd_chunks":              len(hits),
        "provider":                provider,
        "model":                   model,
        "retrieval_preview":       retrieval_preview,
        "all_tests":               unique_preview,
        "total_unique_tests":      len(unique_preview),
        "total_high_confidence":   sum(1 for t in unique_preview if t["confidence"] == "high"),
        "total_medium_confidence": sum(1 for t in unique_preview if t["confidence"] == "medium"),
        "pool_size_used":          pool_size,
    }


# ─── PRD chunk loading (scroll — no 500-hit search cap) ───────────────────────

def _fetch_prd_chunks_ordered(
    es_store: ESStore,
    prd_source_id: str,
    *,
    source_fields: list[str],
    max_chunks: int = 2000,
) -> list[dict]:
    """Load all chunks for a PRD source from Elasticsearch, ordered by chunk_index."""
    from elasticsearch import helpers as es_helpers

    chunks: list[dict] = []
    for hit in es_helpers.scan(
        es_store._client,
        index="qa_prd_chunks",
        query={"query": {"term": {"source_id": prd_source_id}},
               "_source": source_fields},
        scroll="2m",
        size=500,
    ):
        chunks.append(hit["_source"])
        if len(chunks) >= max_chunks:
            logger.warning(
                "PRD %s exceeds %s chunks for this operation; truncating",
                prd_source_id,
                max_chunks,
            )
            break
    chunks.sort(key=lambda c: c.get("chunk_index", 0))
    return chunks


# ─── Data validation (no LLM) ─────────────────────────────────────────────────

def validate_prd_data(
    prd_source_id: str,
    module: list[str] | None,
    embed_client: EmbedClient,
    es_store: ESStore,
    top_k_tests: int = 10,
    top_k_kb: int = 5,
    reranker=None,
) -> dict[str, Any]:
    """
    Pre-flight data check: verify the PRD is ingested correctly and that
    the retrieval pipeline returns sensible test cases and knowledge-base docs.

    Does NOT call any LLM. Safe to run before committing tokens.

    Returns:
      prd_status:   chunk count, doc title, doc_url, list of section headings
      prd_chunks:   list of {chunk_index, section_heading, chunk_text (first 300 chars)}
      retrieval:    for each section heading (up to 8):
                      query          — the heading used as query
                      test_matches   — top test cases from qa_test_cases
                      kb_matches     — top related chunks from qa_prd_chunks (other sources)
    """
    # ── 1. Fetch all PRD chunks for this source ────────────────────────────────
    chunk_sources = _fetch_prd_chunks_ordered(
        es_store,
        prd_source_id,
        source_fields=["section_heading", "chunk_text", "chunk_index", "doc_title", "doc_url"],
    )

    if not chunk_sources:
        return {
            "prd_status": {
                "source_id":  prd_source_id,
                "status":     "NOT_INGESTED",
                "chunk_count": 0,
                "message":    "No chunks found. Ingest the document first via POST /ingest/prd",
            },
            "prd_chunks":  [],
            "retrieval":   [],
        }

    doc_title = chunk_sources[0].get("doc_title", prd_source_id)
    doc_url   = chunk_sources[0].get("doc_url")

    # Build chunk summaries + rich query pairs (heading + body for better embedding signal)
    chunk_summaries = []
    headings_seen: list[str] = []
    # Maps heading → first chunk body text (for building rich queries)
    heading_to_body: dict[str, str] = {}
    for src in chunk_sources:
        heading = (src.get("section_heading") or "").strip()
        body    = src.get("chunk_text", "")
        chunk_summaries.append({
            "chunk_index":     src.get("chunk_index"),
            "section_heading": heading or None,
            "chunk_text":      body[:300],
        })
        if heading and heading not in headings_seen:
            headings_seen.append(heading)
            heading_to_body[heading] = body  # first chunk wins

    prd_status = {
        "source_id":       prd_source_id,
        "status":          "INGESTED",
        "chunk_count":     len(chunk_sources),
        "doc_title":       doc_title,
        "doc_url":         doc_url,
        "section_headings": headings_seen,
    }

    # ── 2. For ALL testable headings, run retrieval ───────────────────────────
    # No heading cap — every non-meta section is queried.
    # Adaptive pool size — ~20% of the module, so we never miss relevant tests.
    # Cross-query deduplication — same test from multiple queries keeps highest score
    # and accumulates the list of queries it matched (shows coverage breadth).
    # RRF scores are rank-based; rerank_score thresholds when cross-encoder is present
    RERANK_HIGH   = 2.0
    RERANK_MEDIUM = 0.5
    RRF_HIGH      = 0.025
    RRF_MEDIUM    = 0.012
    KB_MIN_SCORE  = 0.77

    pool_size = es_store.estimate_pool_size(module)
    logger.info(f"[validate] module={module} pool_size={pool_size} headings={len(headings_seen)}")

    # Collect testable headings with multi-query: heading-only + heading+body
    # Two queries per heading improves recall — short heading catches keyword matches,
    # rich query catches semantic matches with different terminology.
    query_pairs: list[tuple[str, str]] = []  # (heading, query_text)
    heading_query_map: dict[str, list[int]] = {}  # heading → indices into query_pairs
    # Clean heading map: strip Confluence markdown escapes (e.g. "1\. Foo" → "1. Foo")
    # so embeddings and BM25 see natural text, not backslash-escaped punctuation.
    clean_heading: dict[str, str] = {
        h: re.sub(r'\\([.\-*_#])', r'\1', h) for h in headings_seen
    }
    for heading in headings_seen:
        if _is_meta_heading(heading):
            continue
        body = heading_to_body.get(heading, "").strip()
        if len(body) < 20:
            continue
        ch = clean_heading[heading]
        indices = []
        # Query 1: clean heading only — catches exact keyword matches
        indices.append(len(query_pairs))
        query_pairs.append((heading, ch))
        # Query 2: clean heading + body — catches semantic matches
        indices.append(len(query_pairs))
        query_pairs.append((heading, f"{ch}: {body[:500]}"))
        heading_query_map[heading] = indices

    # Batch embed all queries at once — much faster than sequential
    if query_pairs:
        query_texts = [q for _, q in query_pairs]
        query_vecs = embed_client.embed_queries(query_texts)
    else:
        query_vecs = []

    # jira_key → best result dict (deduplicated across all queries)
    all_tests: dict[str, dict] = {}
    retrieval_results = []

    for heading, indices in heading_query_map.items():
        try:
            # Multi-query: run all query variants for this heading, merge results.
            # Use 100 candidates when reranker is present (cross-encoder re-scores all of them).
            # Without reranker we use a wider pool to compensate for weaker RRF signal.
            retrieval_k = min(top_k_tests * 5, 50) if reranker else min(pool_size, 100)
            merged: dict[str, dict] = {}  # jira_key → best result
            for idx in indices:
                _, q_text = query_pairs[idx]
                q_vec = query_vecs[idx]
                hits = es_store.search_hybrid(
                    query_embedding=q_vec,
                    keyword_query=q_text[:200],  # BM25 keyword query
                    top_k=retrieval_k,
                    module_filter=module,
                )
                for r in hits:
                    k = r["jira_key"]
                    if k not in merged or r["score"] > merged[k]["score"]:
                        merged[k] = r

            tests = sorted(merged.values(), key=lambda x: x["score"], reverse=True)

            # Cross-encoder reranking on the merged set.
            # Use heading + body as the reranker query for richer context — avoids
            # surface-level keyword matches when the heading alone is too generic
            # (e.g. "Verification Successful Screen" matching unrelated success-screen tests).
            rerank_query = f"{clean_heading[heading]}: {heading_to_body.get(heading, '')[:300]}".strip(": ")
            if reranker and tests:
                # No top_k cut here — we need all candidates for the per-test fallback
                # scoring below (tests without steps_text use RRF score, not rerank_score).
                tests = reranker.rerank(rerank_query, tests)

            # Use rerank_score for thresholding when available.
            # Exception: tests without steps_text only have a one-line summary —
            # the cross-encoder gets too little signal and underscores them.
            # For those, fall back to RRF score thresholds so they aren't filtered out.
            use_rerank = reranker and tests and "rerank_score" in tests[0]

            test_matches = []
            for r in tests:
                has_steps = bool(r.get("steps_text"))
                if use_rerank and has_steps:
                    s = r.get("rerank_score", 0)
                    score_high_t, score_medium_t = RERANK_HIGH, RERANK_MEDIUM
                else:
                    s = r.get("score", 0)
                    score_high_t, score_medium_t = RRF_HIGH, RRF_MEDIUM
                if s < score_medium_t:
                    continue
                key        = r["jira_key"]
                confidence = "high" if s >= score_high_t else "medium"
                match = {
                    "jira_key":   key,
                    "summary":    r["summary"],
                    "module":     r.get("module"),
                    "labels":     r.get("labels") or [],
                    "score":      round(r.get("score", 0), 4),
                    "rerank_score": r.get("rerank_score"),
                    "confidence": confidence,
                }
                test_matches.append(match)

                # Dedup: keep highest score (s = whichever score was used for thresholding)
                best = all_tests.get(key, {}).get("_sort_score", -999)
                if key not in all_tests or s > best:
                    all_tests[key] = {**match, "matched_queries": [heading], "_sort_score": s}
                else:
                    if heading not in all_tests[key]["matched_queries"]:
                        all_tests[key]["matched_queries"].append(heading)

            # KB docs for this query — q_vec is the last (richest) query vector for this heading
            kb_raw = es_store.search_similar_prd_chunks(
                query_embedding=q_vec,
                top_k=top_k_kb + 2,
                module_filter=module,
            )
            kb_matches = [
                {
                    "source_id":       r["source_id"],
                    "doc_title":       r.get("doc_title"),
                    "section_heading": r.get("section_heading"),
                    "chunk_preview":   r.get("chunk_text", "")[:200],
                    "score":           round(r["score"], 4),
                }
                for r in kb_raw
                if r["source_id"] != prd_source_id and r["score"] >= KB_MIN_SCORE
            ][:top_k_kb]

            retrieval_results.append({
                "query":        heading,
                "test_matches": test_matches,
                "kb_matches":   kb_matches,
            })
        except Exception as exc:
            logger.warning(f"[validate] Failed retrieval for heading {heading!r}: {exc}")
            retrieval_results.append({
                "query": heading,
                "error": str(exc),
                "test_matches": [],
                "kb_matches":   [],
            })

    # Deduplicated union — sorted by score descending; strip internal _sort_score field
    unique_tests = sorted(all_tests.values(), key=lambda x: x["score"], reverse=True)
    for t in unique_tests:
        t.pop("_sort_score", None)
    n_high   = sum(1 for t in unique_tests if t["confidence"] == "high")
    n_medium = sum(1 for t in unique_tests if t["confidence"] == "medium")

    logger.info(
        f"[validate] {len(unique_tests)} unique tests "
        f"({n_high} high, {n_medium} medium) across {len(retrieval_results)} queries"
    )

    return {
        "prd_status":              prd_status,
        "prd_chunks":              chunk_summaries,
        "retrieval":               retrieval_results,
        "all_tests":               unique_tests,
        "total_unique_tests":      len(unique_tests),
        "total_high_confidence":   n_high,
        "total_medium_confidence": n_medium,
        "pool_size_used":          pool_size,
    }
