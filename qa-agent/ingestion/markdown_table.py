"""
Shared grid → markdown-table rendering.

Every ingestor that meets tabular data (Confluence <table>, Excel sheets, Word
tables, PDF table regions) funnels through here so the chunker sees ONE table
shape: rows that start with '|', a header row, and a '| --- |' delimiter.

That leading pipe is load-bearing — ingestion/chunker.py keys its table handling
off it (row-level segmentation, header carry-over into continuation chunks).
A table rendered any other way silently loses that treatment.
"""
import re


_ROW_RE   = re.compile(r'^\s*\|')
_DELIM_RE = re.compile(r'^\s*\|[\s:|-]+\|\s*$')


def _leading_table_header(text: str) -> list[str]:
    """The table header block a chunk opens with: first row, plus a delimiter row."""
    header: list[str] = []
    for i, line in enumerate(text.split("\n")):
        if not _ROW_RE.match(line):
            break
        if i == 0:
            header.append(line)
        elif i == 1 and _DELIM_RE.match(line):
            header.append(line)
        else:
            break
    return header


def _trailing_table_header(text: str) -> list[str] | None:
    """
    The header of the table a chunk *ends* in, or None if it does not end in one.

    Taken from the trailing run rather than the leading lines because the first
    chunk of a table is typically mixed — a prose lead-in, then the table — so its
    header is not at the top. Missing that case leaves the run untracked and every
    continuation keeps its repeated header.
    """
    lines = text.split("\n")
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    if end == 0 or not _ROW_RE.match(lines[end - 1]):
        return None

    start = end
    while start > 0 and _ROW_RE.match(lines[start - 1]):
        start -= 1

    header = [lines[start]]
    if start + 1 < end and _DELIM_RE.match(lines[start + 1]):
        header.append(lines[start + 1])
    return [l.strip() for l in header]


def join_chunk_texts(texts: list[str], separator: str = "\n\n") -> str:
    """
    Re-join chunk texts into one document, dropping headers repeated by continuation
    chunks of the same table.

    The chunker copies a table's header into every chunk the table spans so each
    one stands alone in retrieval (see ingestion/chunker.py::_materialize).
    Re-assembled — as read_prd_document does — that header would reappear
    mid-table, showing the model three short tables where the source had one.

    The active header is tracked across the whole run rather than compared against
    the immediately preceding chunk: once chunk N's header has been stripped,
    chunk N+1 has nothing to match against. A non-table chunk, or a table with
    different columns, ends the run so the next table keeps its own header.

    Works from the text alone, so chunks indexed before chunk_type existed are
    handled identically.
    """
    out: list[str] = []
    active: list[str] | None = None       # header of the table currently being emitted
    prev_ends_in_row = False

    for raw in texts:
        text = raw or ""
        if not text.strip():
            continue

        continuation = False
        header = _leading_table_header(text)
        if header and prev_ends_in_row and [l.strip() for l in header] == active:
            remainder = "\n".join(text.split("\n")[len(header):]).lstrip("\n")
            if not remainder.strip():
                continue                    # chunk was nothing but the repeated header
            text = remainder
            continuation = True

        # Rows of one table must stay on consecutive lines. The blank line the
        # separator would insert splits the table in two again, undoing the strip.
        if out:
            out.append("\n" if continuation else separator)
        out.append(text)

        if not continuation:
            # A stripped continuation opens with data rows, not the header, so its
            # trailing run would set `active` to a data row. Keep the real header.
            active = _trailing_table_header(text)

        tail = [l for l in text.split("\n") if l.strip()]
        prev_ends_in_row = bool(tail) and bool(_ROW_RE.match(tail[-1]))

    return "".join(out)


def escape_cell(text: str) -> str:
    """Flatten one cell to a single line that cannot break the row structure."""
    if text is None:
        return ""
    out = str(text).replace("\u00a0", " ")   # nbsp is pervasive in Confluence cells
    out = out.replace("\\", "\\\\").replace("|", "\\|")
    out = " ".join(out.split())          # collapse newlines/tabs/runs of spaces
    return out.strip()


def rows_to_markdown(rows: list[list], header: bool = True) -> str:
    """
    Render a rectangular grid as a markdown pipe table.

    Rows are padded to the widest row so the delimiter width always matches —
    a ragged table renders as text, not a table, in most markdown consumers.

    header=False emits a synthetic blank header, because markdown has no
    header-less table form and the chunker needs a stable first row to carry.
    """
    grid = [[escape_cell(c) for c in row] for row in rows]
    grid = [r for r in grid if any(c for c in r)]
    if not grid:
        return ""

    width = max(len(r) for r in grid)
    for r in grid:
        r.extend([""] * (width - len(r)))

    def _line(cells: list[str]) -> str:
        return "| " + " | ".join(cells) + " |"

    if header:
        head, body = grid[0], grid[1:]
    else:
        head, body = [""] * width, grid

    lines = [_line(head), "| " + " | ".join(["---"] * width) + " |"]
    lines.extend(_line(r) for r in body)
    return "\n".join(lines)
