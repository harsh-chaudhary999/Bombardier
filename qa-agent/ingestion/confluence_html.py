"""
Confluence storage-format (XHTML) → markdown, shared by the page and space ingestors.

Why this is not just html2text
------------------------------
Confluence storage format carries the highest-value PRD content inside markup
html2text cannot see:

  * Code / noformat macros wrap their body in ``<![CDATA[...]]>``. Python's
    HTMLParser reports that as ``unknown_decl``, which html2text does not
    implement — so every JSON payload, SQL snippet and config sample in a page
    was being dropped on the floor before chunking ever ran.
  * Images live in ``<ac:image><ri:attachment ri:filename="..."/></ac:image>``.
    The ``ac:``-tag strip leaves a bare ``ri:`` tag that emits no text at all,
    so both the alt text and the filename were lost.
  * ``<ac:parameter>`` elements are stripped wholesale (they leak layout junk
    like "wide760"), which also took the *title* of status and expand macros
    with them.
  * html2text's own table output does not start rows with '|', so the chunker's
    table handling never engaged on Confluence tables.

The fix is to lift each of those out into a placeholder BEFORE html2text runs,
then splice the rendered form back in afterwards. Placeholders are bare
alphanumerics so no markdown escaper, link rewriter or line wrapper touches them.
"""
import os
import re
import logging
from html.parser import HTMLParser

from html2text import HTML2Text

from ingestion.markdown_table import rows_to_markdown

logger = logging.getLogger(__name__)

# Keep full URLs inline in body text. Off by default: a cross-linked PRD otherwise
# spends a large share of every chunk's tokens on URLs, which dilutes the embedding
# and adds nothing BM25 can use. Anchor text is kept either way.
KEEP_LINK_URLS = os.environ.get("QA_INGEST_KEEP_LINK_URLS", "0").strip() not in ("", "0", "false", "False")

_PLACEHOLDER_RE = re.compile(r"ZQX(CODE|TABLE|IMAGE)(\d+)XQZ")


class _Placeholders:
    """Holds content lifted out of the HTML, keyed by a token html2text won't touch."""

    def __init__(self) -> None:
        self._items: dict[str, dict] = {}
        self._n = 0

    def add(self, kind: str, **payload) -> str:
        token = f"ZQX{kind}{self._n}XQZ"
        self._n += 1
        self._items[token] = {"kind": kind, **payload}
        return token

    def _render_block(self, item: dict) -> str:
        if item["kind"] == "CODE":
            lang = item.get("language") or ""
            body = item["text"].strip("\n")
            return f"```{lang}\n{body}\n```"
        if item["kind"] == "TABLE":
            return item["text"]
        return item["text"]          # IMAGE — already a one-line marker

    def _render_inline(self, item: dict) -> str:
        """Form safe inside a table cell, where a fenced block would break the row."""
        if item["kind"] == "CODE":
            return "`" + " ".join(item["text"].split()) + "`"
        return " ".join(item["text"].split())

    def resolve_inline(self, text: str) -> str:
        return _PLACEHOLDER_RE.sub(
            lambda m: self._render_inline(self._items.get(m.group(0), {"kind": "", "text": ""})),
            text,
        )

    def restore(self, text: str) -> str:
        def _sub(m: re.Match) -> str:
            item = self._items.get(m.group(0))
            if item is None:
                return ""
            return "\n\n" + self._render_block(item) + "\n\n"

        text = _PLACEHOLDER_RE.sub(_sub, text)
        return re.sub(r"\n{3,}", "\n\n", text)


# ─── Code / noformat macros (the CDATA bodies html2text discards) ─────────────

_CODE_MACRO_RE = re.compile(
    r'<ac:structured-macro[^>]*\bac:name="(code|noformat)"[^>]*>(.*?)</ac:structured-macro>',
    re.DOTALL | re.IGNORECASE,
)
_LANG_PARAM_RE = re.compile(
    r'<ac:parameter[^>]*\bac:name="language"[^>]*>(.*?)</ac:parameter>', re.DOTALL | re.IGNORECASE
)
_PLAIN_BODY_RE = re.compile(
    r'<ac:plain-text-body[^>]*>(.*?)</ac:plain-text-body>', re.DOTALL | re.IGNORECASE
)
_CDATA_RE = re.compile(r'<!\[CDATA\[(.*?)\]\]>', re.DOTALL)


def _unwrap_cdata(text: str) -> str:
    return _CDATA_RE.sub(lambda m: m.group(1), text)


def _extract_code_macros(html: str, store: _Placeholders) -> str:
    """Lift code/noformat macro bodies out verbatim, tagged with their language."""

    def _replace(m: re.Match) -> str:
        inner = m.group(2)
        lang_m = _LANG_PARAM_RE.search(inner)
        language = _unwrap_cdata(lang_m.group(1)).strip() if lang_m else ""
        body_m = _PLAIN_BODY_RE.search(inner)
        raw = body_m.group(1) if body_m else inner
        code = _unwrap_cdata(raw)
        code = re.sub(r'<[^>]+>', '', code)          # macro bodies are plain text
        code = _unescape_entities(code).strip("\n")
        if not code.strip():
            return ""
        return store.add("CODE", text=code, language=language)

    return _CODE_MACRO_RE.sub(_replace, html)


def _unescape_entities(text: str) -> str:
    import html as _html
    return _html.unescape(text)


def _rescue_stray_cdata(html: str) -> str:
    """
    Any CDATA still standing after macro extraction (hand-authored markup, macros
    we do not special-case) would be dropped by the parser. Unwrap it to text.
    """
    import html as _html
    return _CDATA_RE.sub(lambda m: _html.escape(m.group(1)), html)


# ─── Images (alt text + attachment filename are real retrieval signal) ────────

_AC_IMAGE_RE = re.compile(r'<ac:image\b([^>]*)>(.*?)</ac:image>', re.DOTALL | re.IGNORECASE)
_AC_IMAGE_SELF_RE = re.compile(r'<ac:image\b([^>]*)/>', re.IGNORECASE)
_IMG_TAG_RE = re.compile(r'<img\b([^>]*?)/?>', re.IGNORECASE)
_RI_FILENAME_RE = re.compile(r'\bri:filename="([^"]*)"', re.IGNORECASE)
_RI_VALUE_RE = re.compile(r'\bri:value="([^"]*)"', re.IGNORECASE)
_AC_ALT_RE = re.compile(r'\bac:alt="([^"]*)"', re.IGNORECASE)
_ATTR_RE = re.compile(r'\b(alt|src|title)="([^"]*)"', re.IGNORECASE)


def _image_marker(alt: str, ref: str) -> str:
    alt = " ".join((alt or "").split())
    ref = (ref or "").strip().rsplit("/", 1)[-1]
    if alt and ref:
        return f"[Image: {alt} ({ref})]"
    if alt:
        return f"[Image: {alt}]"
    if ref:
        return f"[Image: {ref}]"
    return "[Image]"


def _extract_images(html: str, store: _Placeholders) -> str:
    """Replace every image form with a text marker naming it."""

    def _ac(m: re.Match) -> str:
        attrs, inner = m.group(1), m.group(2)
        alt_m = _AC_ALT_RE.search(attrs)
        ref_m = _RI_FILENAME_RE.search(inner) or _RI_VALUE_RE.search(inner)
        return store.add("IMAGE", text=_image_marker(
            _unescape_entities(alt_m.group(1)) if alt_m else "",
            _unescape_entities(ref_m.group(1)) if ref_m else "",
        ))

    def _ac_self(m: re.Match) -> str:
        alt_m = _AC_ALT_RE.search(m.group(1))
        return store.add("IMAGE", text=_image_marker(
            _unescape_entities(alt_m.group(1)) if alt_m else "", ""))

    def _img(m: re.Match) -> str:
        attrs = dict((k.lower(), v) for k, v in _ATTR_RE.findall(m.group(1)))
        return store.add("IMAGE", text=_image_marker(
            _unescape_entities(attrs.get("alt") or attrs.get("title") or ""),
            _unescape_entities(attrs.get("src") or ""),
        ))

    html = _AC_IMAGE_RE.sub(_ac, html)
    html = _AC_IMAGE_SELF_RE.sub(_ac_self, html)
    return _IMG_TAG_RE.sub(_img, html)


# ─── Macro titles that the ac:parameter strip would otherwise take with it ────

_TITLED_MACRO_RE = re.compile(
    r'<ac:structured-macro[^>]*\bac:name="(status|expand|panel|info|note|warning|tip)"[^>]*>',
    re.IGNORECASE,
)
_TITLE_PARAM_RE = re.compile(
    r'<ac:parameter[^>]*\bac:name="title"[^>]*>(.*?)</ac:parameter>', re.DOTALL | re.IGNORECASE
)


def _inline_macro_titles(html: str) -> str:
    """
    Promote the title of status/expand/panel macros to body text.

    A status macro is nothing BUT its title ("DONE", "IN REVIEW"); dropping it
    erased the state labels a PRD uses to mark which requirements are live.
    """
    out: list[str] = []
    pos = 0
    for m in _TITLED_MACRO_RE.finditer(html):
        end = _macro_end(html, m.start())
        if end is None:
            continue
        segment = html[m.start():end]
        # Search only this macro's own parameters. A macro's parameters precede its
        # body, so anything from the first nested structured-macro onward belongs to
        # a child — without this, an expand wrapping a status macro adopts the
        # status label as its own title.
        nested = re.search(r'<ac:structured-macro\b', segment[1:], re.IGNORECASE)
        own = segment[:nested.start() + 1] if nested else segment
        title_m = _TITLE_PARAM_RE.search(own)
        if not title_m:
            continue
        title = _unescape_entities(_unwrap_cdata(title_m.group(1))).strip()
        if not title:
            continue
        out.append(html[pos:m.end()])
        out.append(f"<span>{title}</span> ")
        pos = m.end()
    out.append(html[pos:])
    return "".join(out)


def _macro_end(html: str, start: int) -> int | None:
    """End offset of the structured-macro opening at `start`, honouring nesting."""
    depth = 0
    for m in re.finditer(r'<(/?)ac:structured-macro\b[^>]*?(/?)>', html[start:], re.IGNORECASE):
        if m.group(2) == "/":
            if depth == 0:
                return start + m.end()
            continue
        depth += -1 if m.group(1) else 1
        if depth == 0:
            return start + m.end()
    return None


# ─── Cross-page links (<ri:page ri:content-title="..."/>) ────────────────────

_RI_PAGE_RE = re.compile(r'<ri:page\b([^>]*)/?>', re.IGNORECASE)
_RI_CONTENT_TITLE_RE = re.compile(r'\bri:content-title="([^"]*)"', re.IGNORECASE)


def _inline_link_targets(html: str) -> str:
    def _replace(m: re.Match) -> str:
        t = _RI_CONTENT_TITLE_RE.search(m.group(1))
        return f"<span>{t.group(1)}</span>" if t else ""
    return _RI_PAGE_RE.sub(_replace, html)


# ─── Tables ──────────────────────────────────────────────────────────────────

class _TableReader(HTMLParser):
    """
    Reads one <table> into a grid, resolving colspan (pad) and rowspan (fill down).

    Merged cells are the norm in Confluence spec tables — a "Module" column
    spanning several requirement rows. Left unresolved they become blanks, and a
    row split off into its own chunk loses the value entirely.
    """

    _CELL_BREAKS = ("br", "p", "div", "li", "tr")

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self._depth = 0
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._colspan = 1
        self._rowspan = 1
        self._pending: dict[int, tuple[str, int]] = {}   # col -> (value, rows left)

    @staticmethod
    def _span(attrs: list[tuple[str, str | None]], name: str) -> int:
        for k, v in attrs:
            if k.lower() == name:
                try:
                    return max(1, min(int((v or "1").strip()), 50))
                except ValueError:
                    return 1
        return 1

    def handle_starttag(self, tag, attrs):
        t = tag.lower()
        if t == "table":
            self._depth += 1
            return
        if self._depth != 1:
            # Inside a nested table. Its structure cannot survive as a table (a
            # markdown cell is one line), but punctuating the flattened text keeps
            # the cell boundaries readable instead of running words together.
            if self._cell is not None:
                if t in ("td", "th"):
                    self._append_separator(" / ")
                elif t == "tr":
                    self._append_separator(" ; ")
                elif t in self._CELL_BREAKS:
                    self._cell.append(" ")
            return
        if t == "tr":
            self._row = []
        elif t in ("td", "th"):
            self._cell = []
            self._colspan = self._span(attrs, "colspan")
            self._rowspan = self._span(attrs, "rowspan")
        elif self._cell is not None and t in self._CELL_BREAKS:
            self._cell.append(" ")

    def _append_separator(self, sep: str) -> None:
        """
        Punctuate a nested table's cells — but never lead with a separator, and
        never double one up. A single-cell nested table has nothing to separate.
        """
        if not self._cell:
            return
        tail = "".join(self._cell).rstrip()
        if not tail or tail.endswith((";", "/")):
            return
        self._cell.append(sep)

    def handle_startendtag(self, tag, attrs):
        if self._cell is not None and tag.lower() in self._CELL_BREAKS:
            self._cell.append(" ")

    def handle_data(self, data):
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag):
        t = tag.lower()
        if t == "table":
            self._depth -= 1
            return
        if self._depth != 1:
            return
        if t in ("td", "th") and self._cell is not None:
            self._close_cell()
        elif t == "tr" and self._row is not None:
            if self._cell is not None:          # unclosed <td>
                self._close_cell()
            self._close_row()

    def _close_cell(self) -> None:
        # Whitespace normalisation only — rows_to_markdown owns pipe/backslash
        # escaping, and escaping here too would double it.
        value = " ".join("".join(self._cell or []).replace("\u00a0", " ").split()).strip()
        if self._row is None:
            self._row = []
        for _ in range(self._colspan):
            self._place(value)
        if self._rowspan > 1:
            # Every column the cell spans carries down, not just the last one —
            # a cell that is both merged across and merged down otherwise leaves
            # its leading columns unfilled and shifts the following row left.
            last = len(self._row) - 1
            for col in range(last - self._colspan + 1, last + 1):
                # Rows still to fill *after* this one.
                self._pending[col] = (value, self._rowspan - 1)
        self._cell = None
        self._colspan = self._rowspan = 1

    def _place(self, value: str) -> None:
        assert self._row is not None
        self._drain_pending()
        self._row.append(value)

    def _drain_pending(self) -> None:
        """
        Insert rowspan carry-overs that own the column about to be filled.

        The counter is decremented on consumption, not at end-of-row: a cell is
        placed in the row that declares it, so decrementing there too would
        retire the carry-over one row early and blank the merged column.
        """
        assert self._row is not None
        while len(self._row) in self._pending:
            col = len(self._row)
            value, left = self._pending[col]
            self._row.append(value)
            if left <= 1:
                del self._pending[col]
            else:
                self._pending[col] = (value, left - 1)

    def _close_row(self) -> None:
        assert self._row is not None
        self._drain_pending()
        self.rows.append(self._row)
        self._row = None

    def close(self):
        if self._cell is not None:
            self._close_cell()
        if self._row:
            self._close_row()
        super().close()


def _find_table_spans(html: str) -> list[tuple[int, int]]:
    """Outermost <table>…</table> spans, tracking nesting."""
    spans: list[tuple[int, int]] = []
    depth = 0
    start = 0
    for m in re.finditer(r'<(/?)table\b[^>]*>', html, re.IGNORECASE):
        if m.group(1):
            depth -= 1
            if depth == 0:
                spans.append((start, m.end()))
            depth = max(depth, 0)
        else:
            if depth == 0:
                start = m.start()
            depth += 1
    return spans


def _extract_tables(html: str, store: _Placeholders) -> str:
    """Render each table as a real markdown pipe table behind a placeholder."""
    spans = _find_table_spans(html)
    if not spans:
        return html

    out: list[str] = []
    pos = 0
    for start, end in spans:
        out.append(html[pos:start])
        reader = _TableReader()
        try:
            reader.feed(html[start:end])
            reader.close()
            rows = [[store.resolve_inline(c) for c in row] for row in reader.rows]
            markdown = rows_to_markdown(rows) if rows else ""
        except Exception as exc:                      # malformed markup must not kill the page
            logger.warning("table conversion failed, falling back to inline text: %s", exc)
            markdown = ""
        if markdown:
            out.append(store.add("TABLE", text=markdown))
        else:
            out.append(html[start:end])               # let html2text do what it can
        pos = end
    out.append(html[pos:])
    return "".join(out)


# ─── Macro scaffolding / layout attribute strip (unchanged behaviour) ────────

def _strip_macro_scaffolding(html: str) -> str:
    # ac:parameter elements AND their content — these leak layout junk ("wide760").
    # Runs after _inline_macro_titles, which has already rescued the titles worth keeping.
    html = re.sub(r'<ac:parameter[^>]*>.*?</ac:parameter>', '', html, flags=re.DOTALL)
    html = re.sub(r'</?ac:[^>]+>', '', html)
    html = re.sub(r'</?ri:[^>]+>', '', html)
    html = re.sub(r'\s+(?:width|style|class|data-[a-z-]+)="[^"]*"', '', html)
    return html


def _cleanup(text: str) -> str:
    # Runs BEFORE placeholders are restored so it cannot corrupt code or table content.
    text = re.sub(r'\bwide\d+', '', text)
    text = re.sub(r'\bfixed(?:-table|-layout|-width|Width)\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text


def _render_html(html: str) -> str:
    h = HTML2Text()
    h.ignore_links = not KEEP_LINK_URLS   # anchor text is kept either way
    h.ignore_images = True                # real images already lifted to markers
    h.ignore_tables = False
    h.body_width = 0                      # no wrapping — it would break placeholders
    h.unicode_snob = True
    return h.handle(html)


def storage_to_markdown(html: str) -> str:
    """
    Confluence storage-format XHTML → markdown text ready for chunk_document().

    Order matters: everything that needs the original markup (code bodies, image
    refs, macro titles, link targets) is lifted out before the ``ac:``/``ri:``
    strip destroys it, and tables are converted after the strip so cell text is
    already clean.
    """
    if not html or not html.strip():
        return ""

    store = _Placeholders()
    html = _extract_code_macros(html, store)
    html = _extract_images(html, store)
    html = _inline_macro_titles(html)
    html = _inline_link_targets(html)
    html = _strip_macro_scaffolding(html)
    html = _rescue_stray_cdata(html)
    html = _extract_tables(html, store)

    text = _render_html(html)
    text = _cleanup(text)
    text = store.restore(text)
    return text.strip()
