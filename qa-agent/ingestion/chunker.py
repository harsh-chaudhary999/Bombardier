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
  - Structure-aware: table rows, list items and fenced code blocks are atomic
    segments, joined back with newlines so their layout survives a split
  - A table split across chunks repeats its header row in every continuation
    chunk — without it a chunk reads '| Pro | 5000 | Yes |' with no column names,
    which is close to worthless to both the embedder and the reranker
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

# Segment kinds. Only "prose" segments participate in similarity-based boundary
# detection and only they are joined with spaces; everything else is layout that
# a space-join would flatten.
_PROSE = "prose"
_LIST  = "list"
_TABLE = "table_row"
_CODE  = "code"

_FENCE_RE      = re.compile(r'^\s*(```|~~~)')
_TABLE_ROW_RE  = re.compile(r'^\s*\|')
_TABLE_SEP_RE  = re.compile(r'^\s*\|[\s:|-]+\|\s*$')
_LIST_ITEM_RE  = re.compile(r'^\s*(?:[-*+]\s|\d+[.)]\s)')
_HRULE_RE      = re.compile(r'^-{3,}\s*$')


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


# ─── Segmentation ─────────────────────────────────────────────────────────────

def _segment(text: str, kind: str = _PROSE, **extra) -> dict:
    """
    header_index: this row's position within its table's header block — 0 for the
    header row, 1 for the '| --- |' delimiter, None for a data row. Lets
    _materialize prepend exactly the header lines a chunk is missing rather than
    all-or-nothing.
    """
    seg = {"text": text, "kind": kind, "table_header_lines": None, "header_index": None}
    seg.update(extra)
    return seg


def _split_prose(body: str) -> list[dict]:
    """Sentence/paragraph/list-item segmentation for non-structural text."""
    pattern = r'(?<=[.!?])\s+|\n{2,}|\n(?=\s*[-*+]\s)|\n(?=\s*\d+[.\)]\s)'
    out: list[dict] = []
    for raw in re.split(pattern, body):
        s = raw.strip()
        if not s:
            continue
        out.append(_segment(s, _LIST if _LIST_ITEM_RE.match(s) else _PROSE))
    return out


def _table_segments(rows: list[str]) -> list[dict]:
    """
    One segment per table row, each carrying the header block it belongs to.

    The header block is the first row plus the '| --- |' delimiter when present,
    so a continuation chunk can re-emit a *valid* table, not a headerless one.
    """
    header_lines = [rows[0]]
    if len(rows) > 1 and _TABLE_SEP_RE.match(rows[1]):
        header_lines.append(rows[1])
    n_header = len(header_lines)

    return [
        _segment(row, _TABLE,
                 table_header_lines=header_lines,
                 header_index=(i if i < n_header else None))
        for i, row in enumerate(rows)
    ]


def _segment_body(body: str) -> list[dict]:
    """
    Split a section body into structure-tagged segments.

    Fenced code blocks stay whole (splitting one strands its closing fence in
    another chunk); table rows become individually addressable segments so a
    long table can be split between rows rather than mid-row.
    """
    lines = body.split("\n")
    segments: list[dict] = []
    prose_buf: list[str] = []

    def _flush_prose() -> None:
        if prose_buf:
            segments.extend(_split_prose("\n".join(prose_buf)))
            prose_buf.clear()

    i = 0
    while i < len(lines):
        line = lines[i]

        fence = _FENCE_RE.match(line)
        if fence:
            _flush_prose()
            marker = fence.group(1)
            block = [line]
            i += 1
            while i < len(lines):
                block.append(lines[i])
                closed = lines[i].strip().startswith(marker)
                i += 1
                if closed:
                    break
            segments.append(_segment("\n".join(block), _CODE))
            continue

        if _TABLE_ROW_RE.match(line):
            _flush_prose()
            rows: list[str] = []
            while i < len(lines) and _TABLE_ROW_RE.match(lines[i]):
                rows.append(lines[i].rstrip())
                i += 1
            segments.extend(_table_segments(rows))
            continue

        prose_buf.append(line)
        i += 1

    _flush_prose()
    return segments


def _split_into_segments(body: str) -> list[str]:
    """Backwards-compatible view of _segment_body — segment texts only."""
    return [s["text"] for s in _segment_body(body)]


# ─── Oversized-segment handling ───────────────────────────────────────────────

def _split_text_by_lines(text: str, max_tokens: int) -> list[str]:
    """Break text at line boundaries first; character-slice only as a last resort."""
    pieces: list[str] = []
    buf: list[str] = []
    buf_tokens = 0

    for line in text.split("\n"):
        line_tokens = _count_tokens(line)
        if line_tokens > max_tokens:
            if buf:
                pieces.append("\n".join(buf))
                buf, buf_tokens = [], 0
            char_limit = max(1, int(max_tokens * 3.5))
            overlap = max(0, int(OVERLAP_TOKENS * 3.5))
            step = max(1, char_limit - overlap)
            for start in range(0, len(line), step):
                pieces.append(line[start:start + char_limit])
            continue
        if buf and buf_tokens + line_tokens > max_tokens:
            pieces.append("\n".join(buf))
            buf, buf_tokens = [], 0
        buf.append(line)
        buf_tokens += line_tokens

    if buf:
        pieces.append("\n".join(buf))
    return [p for p in pieces if p.strip()]


_CELL_SPLIT_RE = re.compile(r'(?<!\\)\|')


def _row_cells(line: str) -> list[str]:
    """Cells of a markdown row, honouring the '\\|' escape rows_to_markdown emits."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|") and not body.endswith("\\|"):
        body = body[:-1]
    return [c.strip() for c in _CELL_SPLIT_RE.split(body)]


def _build_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _split_wide_row(seg: dict, max_tokens: int) -> list[dict] | None:
    """
    Split an over-long table row by COLUMN, pairing each group with its slice of
    the header.

    Slicing such a row at character offsets produces fragments that are not rows
    at all — no leading pipe, no column alignment, header prepended to nonsense.
    Splitting vertically keeps every piece a valid, self-describing narrow table.
    Returns None when the row cannot be split this way (a single huge cell), so
    the caller falls back to line/character slicing.
    """
    cells = _row_cells(seg["text"])
    header_lines = seg.get("table_header_lines") or []
    header_cells = _row_cells(header_lines[0]) if header_lines else []
    if len(cells) < 2:
        return None

    groups: list[list[int]] = []
    current: list[int] = []
    current_tokens = 0
    for i, cell in enumerate(cells):
        cost = _count_tokens(cell) + 3          # separators and the header copy
        if current and current_tokens + cost > max_tokens:
            groups.append(current)
            current, current_tokens = [], 0
        current.append(i)
        current_tokens += cost
    if current:
        groups.append(current)

    if len(groups) < 2:
        return None                             # one cell dominates; not splittable here

    out: list[dict] = []
    for group in groups:
        row = _build_row([cells[i] for i in group])
        if header_cells:
            head = _build_row([header_cells[i] if i < len(header_cells) else ""
                               for i in group])
            lines = [head, _build_row(["---"] * len(group))]
        else:
            lines = []
        out.append(_segment(row, _TABLE, table_header_lines=lines or None,
                            header_index=None))
    return out


def _explode_oversized(segments: list[dict], max_tokens: int) -> list[dict]:
    """
    Break any single segment larger than the chunk budget.

    A code block is re-fenced on every piece so each chunk still contains a
    syntactically complete block; a wide table row is split by column so each
    piece stays a valid row.
    """
    out: list[dict] = []
    for seg in segments:
        if _count_tokens(seg["text"]) <= max_tokens:
            out.append(seg)
            continue

        if seg["kind"] == _TABLE:
            pieces = _split_wide_row(seg, max_tokens)
            if pieces:
                out.extend(pieces)
                continue

        if seg["kind"] == _CODE:
            lines = seg["text"].split("\n")
            fence_line = lines[0] if _FENCE_RE.match(lines[0]) else "```"
            marker = (_FENCE_RE.match(fence_line).group(1) if _FENCE_RE.match(fence_line) else "```")
            inner = lines[1:]
            if inner and inner[-1].strip().startswith(marker):
                inner = inner[:-1]
            budget = max(1, max_tokens - _count_tokens(fence_line) * 2)
            for piece in _split_text_by_lines("\n".join(inner), budget):
                out.append(_segment(f"{fence_line}\n{piece}\n{marker}", _CODE))
            continue

        for piece in _split_text_by_lines(seg["text"], max_tokens):
            out.append(_segment(piece, seg["kind"],
                                table_header_lines=seg["table_header_lines"],
                                header_index=seg["header_index"]))
    return out


# ─── Chunk materialisation ────────────────────────────────────────────────────

def _join_segments(segments: list[dict]) -> str:
    """
    Join with a space between two prose segments, a newline anywhere structure is
    involved. A space-join is what previously collapsed a whole table onto one line.
    """
    if not segments:
        return ""
    parts = [segments[0]["text"]]
    for prev, seg in zip(segments, segments[1:]):
        both_prose = prev["kind"] == _PROSE and seg["kind"] == _PROSE
        parts.append(" " if both_prose else "\n")
        parts.append(seg["text"])
    return "".join(parts)


def _strip_horizontal_rules(text: str) -> str:
    """
    Drop markdown horizontal rules, but never inside a fenced code block.

    '---' is also a YAML document separator and a front-matter delimiter, so a
    blanket strip silently guts any YAML sample in a PRD. Table delimiters are
    safe either way — they start with '|'.
    """
    out: list[str] = []
    in_fence = False
    marker = ""

    for line in text.split("\n"):
        fence = _FENCE_RE.match(line)
        if fence and not in_fence:
            in_fence, marker = True, fence.group(1)
        elif in_fence:
            if line.strip().startswith(marker):
                in_fence = False
        elif _HRULE_RE.match(line):
            continue
        out.append(line)

    return "\n".join(out)


def classify_chunk_text(text: str) -> str:
    """
    Label a finished chunk by the kind of content it holds: table | code | mixed | prose.

    Derived from the rendered text rather than the segment list so every path —
    semantic, fixed-window, and the whole-section early returns — labels
    identically. Consumers use it to decide whether layout-aware handling is
    worth attempting; see read_prd_document's table-header de-duplication.
    """
    has_table = has_code = has_prose = False
    in_fence = False
    marker = ""

    for line in text.split("\n"):
        stripped = line.strip()
        fence = _FENCE_RE.match(line)
        if fence and not in_fence:
            in_fence, marker = True, fence.group(1)
            has_code = True
            continue
        if in_fence:
            has_code = True
            if stripped.startswith(marker):
                in_fence = False
            continue
        if not stripped:
            continue
        if _TABLE_ROW_RE.match(line):
            has_table = True
        else:
            has_prose = True

    kinds = [k for k, present in
             (("table", has_table), ("code", has_code), ("prose", has_prose)) if present]
    if len(kinds) == 1:
        return kinds[0]
    return "mixed" if kinds else "prose"


def _materialize(heading: str | None, segments: list[dict], parent_text: str) -> dict:
    """Render a chunk, re-attaching the table header when the chunk starts mid-table."""
    text = _join_segments(segments)

    for seg in segments:
        if seg["kind"] != _TABLE:
            continue
        index = seg["header_index"]
        if index == 0:
            break                     # chunk already opens with the real header
        lines = seg["table_header_lines"] or []
        # A data row needs the whole header block; a chunk that starts at the
        # delimiter row needs only the header line above it.
        missing = lines if index is None else lines[:index]
        if missing:
            text = "\n".join(missing) + "\n" + text
        break

    return {"heading": heading, "text": text, "parent_text": parent_text}


# ─── Semantic chunking (embedding-based boundary detection) ───────────────────

def _boundary_similarities(
    segments: list[dict],
    embed_fn: Callable[[list[str]], list[list[float]]],
) -> list[float] | None:
    """
    Cosine similarity for each adjacent pair, computed only where a topic
    boundary is meaningful — i.e. between two prose segments.

    Adjacent rows of a table are near-identical by construction, so scoring them
    produced noise, not boundaries, and cost one embedding per row on top.
    Non-prose pairs get 1.0 (never a similarity split); size limits still apply.
    """
    n = len(segments)
    candidates = [
        i for i in range(n - 1)
        if segments[i]["kind"] == _PROSE and segments[i + 1]["kind"] == _PROSE
    ]
    if not candidates:
        return None

    needed = sorted({i for i in candidates} | {i + 1 for i in candidates})
    vectors = embed_fn([segments[i]["text"] for i in needed])
    by_index = dict(zip(needed, vectors))

    sims = [1.0] * (n - 1)
    for i in candidates:
        sims[i] = _cosine_sim(by_index[i], by_index[i + 1])
    return sims


def _semantic_chunk_section(
    heading: str | None,
    body: str,
    embed_fn: Callable[[list[str]], list[list[float]]],
) -> list[dict]:
    """
    Split a section using embedding similarity to detect topic boundaries.

    Algorithm:
      1. Segment the body, tagging tables / code / lists as structure
      2. Embed the prose segments that border another prose segment
      3. Compute cosine similarity across those boundaries
      4. Split at natural topic boundaries (low similarity points)
      5. Enforce min/max token constraints

    Each chunk carries parent_text (full section body) for context enrichment.
    """
    parent_text = body[:3000]
    segments = _segment_body(body)

    # Short sections don't need semantic splitting
    if len(segments) <= 2 or _count_tokens(body) <= SEMANTIC_MIN:
        return [{"heading": heading, "text": body, "parent_text": parent_text}]

    segments = _explode_oversized(segments, SEMANTIC_MAX)

    try:
        sims = _boundary_similarities(segments, embed_fn)
    except Exception as e:
        logger.warning(f"Semantic embedding failed, falling back to fixed-window: {e}")
        return _fixed_chunk_body(heading, body)

    scored = [s for s in (sims or []) if s < 1.0]
    threshold = float(np.percentile(scored, SIMILARITY_PCTILE)) if scored else -1.0

    groups: list[list[dict]] = []
    current: list[dict] = [segments[0]]
    current_tokens = _count_tokens(segments[0]["text"])

    for i in range(1, len(segments)):
        seg = segments[i]
        seg_tokens = _count_tokens(seg["text"])
        sim = sims[i - 1] if sims and i - 1 < len(sims) else 1.0

        topic_break = sim < threshold and current_tokens >= SEMANTIC_MIN
        size_break = current_tokens + seg_tokens > SEMANTIC_MAX

        if (topic_break or size_break) and current:
            groups.append(current)
            current = [seg]
            current_tokens = seg_tokens
        else:
            current.append(seg)
            current_tokens += seg_tokens

    if current:
        groups.append(current)

    # Merge any group that came out below the minimum
    merged: list[list[dict]] = []
    for group in groups:
        if merged and _count_tokens(_join_segments(group)) < SEMANTIC_MIN:
            merged[-1].extend(group)
        else:
            merged.append(group)

    if not merged:
        return [{"heading": heading, "text": body, "parent_text": parent_text}]
    return [_materialize(heading, g, parent_text) for g in merged]


# ─── Fixed-window chunking (fallback when no embed_fn) ───────────────────────

def _fixed_chunk_body(heading: str | None, body: str) -> list[dict]:
    """Fixed-window splitting with token cap and overlap."""
    parent_text = body[:3000]

    if _count_tokens(body) <= MAX_TOKENS:
        return [{"heading": heading, "text": body, "parent_text": parent_text}]

    segments = _explode_oversized(_segment_body(body), MAX_TOKENS)
    groups: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = _count_tokens(seg["text"])
        if current and current_tokens + seg_tokens > MAX_TOKENS:
            groups.append(current)
            overlap = [s for s in current[-2:] if s["kind"] == _PROSE]
            current = overlap + [seg]
            current_tokens = sum(_count_tokens(s["text"]) for s in current)
        else:
            current.append(seg)
            current_tokens += seg_tokens

    if current:
        groups.append(current)

    return [_materialize(heading, g, parent_text) for g in groups]


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
            "chunk_type": str,          -- table | code | mixed | prose
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
            cleaned = _strip_horizontal_rules(sc["text"]).strip()
            if cleaned and _count_tokens(cleaned) >= 3:
                all_chunks.append({
                    "source_id": source_id,
                    "section_heading": sc["heading"],
                    "chunk_text": cleaned,
                    "parent_text": sc.get("parent_text"),
                    "chunk_type": classify_chunk_text(cleaned),
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
    # Content mix. A PRD known to be table-heavy that chunks as 100% prose means the
    # source conversion dropped the table structure — visible here before any tokens
    # are spent on analysis.
    mix: dict[str, int] = {}
    for c in all_chunks:
        mix[c["chunk_type"]] = mix.get(c["chunk_type"], 0) + 1
    logger.info("  chunk types: %s",
                ", ".join(f"{k}={v}" for k, v in sorted(mix.items())) or "none")
    for i, c in enumerate(all_chunks):
        trace.event("chunk", source_id=source_id, index=i,
                    heading=c.get("section_heading"),
                    tokens=_count_tokens(c["chunk_text"]),
                    chars=len(c["chunk_text"]),
                    text=c["chunk_text"],
                    parent_text_chars=len(c.get("parent_text") or ""))
    return all_chunks
