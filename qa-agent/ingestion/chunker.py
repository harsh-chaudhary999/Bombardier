"""
Semantic section-based document chunker for PRD ingestion.

Strategy (two modes):
  Semantic mode (when embed_fn provided):
    - Split by markdown headings (H1/H2/H3) to preserve top-level sections
    - Within each section, use embedding similarity to detect topic boundaries
    - Small precise chunks for retrieval, parent section stored for LLM context
  Fallback mode (no embed_fn):
    - Fixed-window splitting at MAX_TOKENS with sentence-level boundaries

Both modes:
  - Structure-aware: keeps table rows, list items, and code blocks intact
  - Proper tokenization via tiktoken (not char//4 heuristic)
  - Parent-child: each chunk carries parent_text (full section) for context enrichment
"""
import re
import logging
from typing import Callable

import numpy as np
import tiktoken

from observability import trace

logger = logging.getLogger(__name__)

_ENCODING = tiktoken.get_encoding("cl100k_base")

MAX_TOKENS        = 800    # target chunk size for fixed-window mode
OVERLAP_TOKENS    = 80     # overlap between consecutive chunks
SEMANTIC_MIN      = 100    # minimum tokens for a semantic chunk
SEMANTIC_MAX      = 1200   # maximum tokens before forced split
SIMILARITY_PCTILE = 30     # percentile threshold for topic boundary detection


def _count_tokens(text: str) -> int:
    """Accurate token count using tiktoken."""
    return len(_ENCODING.encode(text, disallowed_special=()))


def _cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    a_arr, b_arr = np.array(a), np.array(b)
    norm = np.linalg.norm(a_arr) * np.linalg.norm(b_arr)
    if norm < 1e-8:
        return 0.0
    return float(np.dot(a_arr, b_arr) / norm)


def _promote_bold_headings(text: str) -> str:
    """
    Confluence often uses bold text as visual section headers instead of real headings.
    Promote standalone bold lines to ## headings for the chunker.
    """
    def _replace(m: re.Match) -> str:
        content = m.group(1).strip()
        if content.endswith(":"):
            # Keep label-like lines (e.g. "Note:") as body text, not headings.
            return m.group(0)
        return f"## {content}"

    return re.sub(
        r'^\*\*([^*\n]{1,80})\*\*\s*$',
        _replace,
        text,
        flags=re.MULTILINE,
    )


def _split_into_sections(text: str) -> list[dict]:
    """Split markdown text into sections by H1/H2/H3 headings."""
    text = _promote_bold_headings(text)
    heading_pattern = re.compile(r'^(#{1,3})\s+(.+)$', re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if not matches:
        return [{"heading": None, "body": text.strip()}]

    sections = []
    for i, match in enumerate(matches):
        heading = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append({"heading": heading, "body": body})

    preamble = text[:matches[0].start()].strip()
    if preamble:
        sections.insert(0, {"heading": None, "body": preamble})

    return sections


def _split_into_segments(body: str) -> list[str]:
    """
    Split section body into segments respecting structural boundaries.
    Handles sentences, paragraphs, list items, and table rows.
    """
    pattern = r'(?<=[.!?])\s+|\n{2,}|\n(?=\s*[-*]\s)|\n(?=\s*\d+[.\)]\s)|\n(?=\|)'
    segments = re.split(pattern, body)
    return [s.strip() for s in segments if s.strip()]


# ─── Semantic chunking (embedding-based boundary detection) ───────────────────

def _semantic_chunk_section(
    heading: str | None,
    body: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
) -> list[dict]:
    """
    Split a section using embedding similarity to detect topic boundaries.

    Algorithm:
      1. Split body into segments (sentences, list items, table rows)
      2. Embed each segment
      3. Compute cosine similarity between adjacent segments
      4. Split at natural topic boundaries (low similarity points)
      5. Enforce min/max token constraints

    Each chunk carries parent_text (full section body) for context enrichment.
    """
    segments = _split_into_segments(body)

    # Short sections don't need semantic splitting
    if len(segments) <= 2 or _count_tokens(body) <= SEMANTIC_MIN:
        return [{"heading": heading, "text": body, "parent_text": body[:3000]}]

    # Embed all segments for similarity computation
    try:
        embeddings = embed_fn(segments)
    except Exception as e:
        logger.warning(f"Semantic embedding failed, falling back to fixed-window: {e}")
        return _fixed_chunk_body(heading, body)

    # Cosine similarity between adjacent segments
    sims = []
    for i in range(len(embeddings) - 1):
        sims.append(_cosine_sim(embeddings[i], embeddings[i + 1]))

    # Dynamic threshold: split at bottom percentile of similarities (topic boundaries)
    if sims:
        threshold = float(np.percentile(sims, SIMILARITY_PCTILE))
    else:
        threshold = 0.5

    # Build chunks by splitting at low-similarity boundaries
    parent_text = body[:3000]
    chunks = []
    current_segments = [segments[0]]
    current_tokens = _count_tokens(segments[0])

    for i in range(1, len(segments)):
        seg = segments[i]
        seg_tokens = _count_tokens(seg)
        sim = sims[i - 1] if i - 1 < len(sims) else 1.0

        # Split if topic boundary detected AND chunk is large enough
        # OR if chunk would exceed max tokens
        should_split = (
            (sim < threshold and current_tokens >= SEMANTIC_MIN)
            or (current_tokens + seg_tokens > SEMANTIC_MAX)
        )

        if should_split and current_segments:
            chunks.append({
                "heading": heading,
                "text": " ".join(current_segments),
                "parent_text": parent_text,
            })
            current_segments = [seg]
            current_tokens = seg_tokens
        else:
            current_segments.append(seg)
            current_tokens += seg_tokens

    if current_segments:
        chunks.append({
            "heading": heading,
            "text": " ".join(current_segments),
            "parent_text": parent_text,
        })

    # Merge any chunks that are too small (< SEMANTIC_MIN tokens)
    merged = []
    for chunk in chunks:
        if merged and _count_tokens(chunk["text"]) < SEMANTIC_MIN:
            merged[-1]["text"] += " " + chunk["text"]
        else:
            merged.append(chunk)

    return merged if merged else [{"heading": heading, "text": body, "parent_text": parent_text}]


# ─── Fixed-window chunking (fallback when no embed_fn) ───────────────────────

def _fixed_chunk_body(heading: str | None, body: str) -> list[dict]:
    """Fixed-window splitting with token cap and overlap."""
    parent_text = body[:3000]

    if _count_tokens(body) <= MAX_TOKENS:
        return [{"heading": heading, "text": body, "parent_text": parent_text}]

    segments = _split_into_segments(body)
    chunks = []
    current: list[str] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = _count_tokens(seg)

        if seg_tokens > MAX_TOKENS:
            if current:
                chunks.append({"heading": heading, "text": " ".join(current), "parent_text": parent_text})
                current = current[-2:] if len(current) >= 2 else current[:]
                current_tokens = sum(_count_tokens(s) for s in current)
            char_limit = int(MAX_TOKENS * 3.5)
            overlap_chars = int(OVERLAP_TOKENS * 3.5)
            for start in range(0, len(seg), char_limit - overlap_chars):
                piece = seg[start:start + char_limit]
                chunks.append({"heading": heading, "text": piece, "parent_text": parent_text})
            continue

        if current_tokens + seg_tokens > MAX_TOKENS:
            chunks.append({"heading": heading, "text": " ".join(current), "parent_text": parent_text})
            overlap_buffer = current[-2:] if len(current) >= 2 else current[:]
            current = list(overlap_buffer) + [seg]
            current_tokens = sum(_count_tokens(s) for s in current)
        else:
            current.append(seg)
            current_tokens += seg_tokens

    if current:
        chunks.append({"heading": heading, "text": " ".join(current), "parent_text": parent_text})

    return chunks


# ─── Main entry point ─────────────────────────────────────────────────────────

def chunk_document(
    text: str,
    source_id: str,
    embed_fn: Callable[[list[str]], list[list[float]]] | None = None,
) -> list[dict]:
    """
    Main entry point. Takes raw markdown/plain text, returns chunks ready for embedding.

    Args:
        text:      Raw markdown/plain text document
        source_id: Unique identifier for this document
        embed_fn:  Optional embedding function for semantic chunking.
                   Signature: (list[str]) -> list[list[float]]
                   If provided, uses embedding-based topic boundary detection.
                   If None, falls back to fixed-window chunking.

    Returns:
        list of {
            "source_id": str,
            "section_heading": str | None,
            "chunk_text": str,
            "chunk_index": int,
            "parent_text": str | None,  -- full section text for context enrichment
        }
    """
    if not text or not text.strip():
        logger.warning(f"Empty document for source_id={source_id}, skipping")
        return []

    sections = _split_into_sections(text)
    all_chunks = []
    mode = "semantic" if embed_fn else "fixed"

    for section in sections:
        if embed_fn:
            sub_chunks = _semantic_chunk_section(section["heading"], section["body"], embed_fn)
        else:
            sub_chunks = _fixed_chunk_body(section["heading"], section["body"])

        for sc in sub_chunks:
            cleaned = re.sub(r'(?m)^-{3,}\s*$', '', sc["text"]).strip()
            if cleaned and _count_tokens(cleaned) >= 3:
                all_chunks.append({
                    "source_id": source_id,
                    "section_heading": sc["heading"],
                    "chunk_text": cleaned,
                    "parent_text": sc.get("parent_text"),
                })

    for i, chunk in enumerate(all_chunks):
        chunk["chunk_index"] = i

    tok = [_count_tokens(c["chunk_text"]) for c in all_chunks]
    headings = [c.get("section_heading") for c in all_chunks]
    logger.info(
        "chunked %s (%s): %s sections → %s chunks | tokens min/median/max=%s/%s/%s total=%s",
        source_id, mode, len(sections), len(all_chunks),
        min(tok) if tok else 0,
        sorted(tok)[len(tok)//2] if tok else 0,
        max(tok) if tok else 0, sum(tok),
    )
    # Distinct section headings, in document order — the single most useful thing for
    # judging whether a PRD was parsed sensibly before spending tokens analysing it.
    seen: list[str] = []
    for h in headings:
        if h and h not in seen:
            seen.append(h)
    logger.info("  sections (%s): %s%s", len(seen), ", ".join(seen[:15]),
                " ..." if len(seen) > 15 else "")
    for i, c in enumerate(all_chunks):
        trace.event("chunk", source_id=source_id, index=i,
                    heading=c.get("section_heading"),
                    tokens=_count_tokens(c["chunk_text"]),
                    chars=len(c["chunk_text"]),
                    text=c["chunk_text"],
                    parent_text_chars=len(c.get("parent_text") or ""))
    return all_chunks
