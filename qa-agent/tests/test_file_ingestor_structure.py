"""
Tabular and image extraction from uploaded Word / Excel / PDF documents.

The property under test: content that lives in a table or a figure must reach
the chunker. `doc.paragraphs` returns only `w:p` children of `w:body`, so a
.docx whose requirements sit in tables previously ingested as headings and
nothing else — the requirements themselves never got indexed.

The converters are driven here with duck-typed stand-ins for python-docx /
openpyxl / pdfplumber objects, so these tests need no third-party packages and
assert on output shape only.

Runs under the documented host-side command:
    cd qa-agent && PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'
"""
import types
import unittest

from ingestion import file_ingestor as F


# ─── python-docx stand-ins ────────────────────────────────────────────────────

class _Style:
    def __init__(self, name):
        self.name = name


class _Para:
    def __init__(self, text, style="Normal", images=()):
        self.text = text
        self.style = _Style(style)
        self._images = list(images)
        self._p = self


class _Cell:
    def __init__(self, text):
        self.paragraphs = [_Para(text)] if text else []


class _Row:
    def __init__(self, values):
        self.cells = [_Cell(v) for v in values]


class _Table:
    def __init__(self, rows):
        self.rows = [_Row(r) for r in rows]


class DocxTableTests(unittest.TestCase):
    """The path that dropped every requirement stored in a Word table."""

    TABLE = _Table([
        ["Requirement", "Priority", "Owner"],
        ["req_alpha", "P1", "team_a"],
        ["req_beta", "P2", "team_b"],
    ])

    def test_table_becomes_a_markdown_table(self):
        out = F._docx_table_markdown(self.TABLE)
        self.assertIn("| Requirement | Priority | Owner |", out)
        self.assertIn("| --- | --- | --- |", out)

    def test_body_rows_are_present(self):
        out = F._docx_table_markdown(self.TABLE)
        self.assertIn("| req_alpha | P1 | team_a |", out)
        self.assertIn("| req_beta | P2 | team_b |", out)

    def test_every_row_starts_with_a_pipe(self):
        lines = [l for l in F._docx_table_markdown(self.TABLE).split("\n") if l.strip()]
        self.assertTrue(all(l.startswith("|") for l in lines), lines)

    def test_empty_table_yields_nothing(self):
        self.assertEqual(F._docx_table_markdown(_Table([])), "")

    def test_multi_paragraph_cell_is_flattened_to_one_row(self):
        cell = _Cell("")
        cell.paragraphs = [_Para("first line"), _Para("second line")]
        row = _Row([])
        row.cells = [cell]
        table = _Table([])
        table.rows = [_Row(["Header"]), row]
        lines = F._docx_table_markdown(table).split("\n")
        self.assertIn("| first line second line |", lines)


class DocxParagraphTests(unittest.TestCase):
    def test_heading_levels_map_to_hashes(self):
        self.assertEqual(F._docx_paragraph_markdown(_Para("Title", "Heading 1")).strip(), "# Title")
        self.assertEqual(F._docx_paragraph_markdown(_Para("Sub", "Heading 2")).strip(), "## Sub")
        self.assertEqual(F._docx_paragraph_markdown(_Para("Leaf", "Heading 3")).strip(), "### Leaf")

    def test_list_style_becomes_a_bullet(self):
        self.assertEqual(F._docx_paragraph_markdown(_Para("item", "List Bullet")), "- item")

    def test_body_text_is_unchanged(self):
        self.assertEqual(F._docx_paragraph_markdown(_Para("Plain sentence.")), "Plain sentence.")

    def test_empty_paragraph_is_dropped(self):
        self.assertEqual(F._docx_paragraph_markdown(_Para("   ")), "")


class DocxImageTests(unittest.TestCase):
    """A diagram's description is the only searchable trace it leaves."""

    class _Props(dict):
        def get(self, k, default=None):
            return dict.get(self, k, default)

    class _P:
        def __init__(self, props):
            self._props = props

        def iterfind(self, path, ns):
            return iter(self._props)

    def _para_with(self, props):
        para = _Para("Figure caption")
        para._p = self._P(props)
        return para

    def test_description_is_used(self):
        markers = F._docx_paragraph_images(self._para_with([{"descr": "Retry state machine"}]))
        self.assertEqual(markers, ["[Image: Retry state machine]"])

    def test_name_is_the_fallback(self):
        markers = F._docx_paragraph_images(self._para_with([{"name": "diagram_1"}]))
        self.assertEqual(markers, ["[Image: diagram_1]"])

    def test_unlabelled_image_still_marks_presence(self):
        self.assertEqual(F._docx_paragraph_images(self._para_with([{}])), ["[Image]"])

    def test_marker_is_appended_to_the_paragraph_text(self):
        para = self._para_with([{"descr": "Retry state machine"}])
        out = F._docx_paragraph_markdown(para)
        self.assertIn("Figure caption", out)
        self.assertIn("[Image: Retry state machine]", out)

    def test_broken_drawing_markup_does_not_raise(self):
        para = _Para("text")

        class _Boom:
            def iterfind(self, path, ns):
                raise RuntimeError("unexpected markup")

        para._p = _Boom()
        self.assertEqual(F._docx_paragraph_images(para), [])


# ─── openpyxl stand-ins ───────────────────────────────────────────────────────

class _XlCell:
    def __init__(self, value):
        self.value = value


class _Range:
    def __init__(self, min_row, min_col, max_row, max_col):
        self.min_row, self.min_col = min_row, min_col
        self.max_row, self.max_col = max_row, max_col


class _Merged:
    def __init__(self, ranges):
        self.ranges = ranges


class _Sheet:
    def __init__(self, grid, merged=()):
        self._grid = grid
        self.merged_cells = _Merged(list(merged))

    def iter_rows(self):
        for row in self._grid:
            yield [_XlCell(v) for v in row]

    def cell(self, row, column):
        return _XlCell(self._grid[row - 1][column - 1])


class ExcelMergedCellTests(unittest.TestCase):
    def test_merged_range_fills_every_covered_cell(self):
        sheet = _Sheet(
            [["Module", "Rule"], ["module_x", "rule_1"], [None, "rule_2"]],
            merged=[_Range(2, 1, 3, 1)],
        )
        filled = F._fill_merged_cells(sheet)
        self.assertEqual(filled[(2, 1)], "module_x")
        self.assertEqual(filled[(3, 1)], "module_x",
                         "the covered cell reads None without this fill")

    def test_sheet_without_merges_yields_nothing(self):
        self.assertEqual(F._fill_merged_cells(_Sheet([["a"]])), {})

    def test_missing_merged_attribute_is_tolerated(self):
        self.assertEqual(F._fill_merged_cells(types.SimpleNamespace()), {})


# ─── pdfplumber stand-ins ─────────────────────────────────────────────────────

class _Found:
    def __init__(self, rows, bbox):
        self._rows = rows
        self.bbox = bbox

    def extract(self):
        return self._rows


class _Page:
    def __init__(self, text, tables=(), filtered_text=None):
        self._text = text
        self._tables = list(tables)
        self._filtered_text = filtered_text

    def find_tables(self):
        return self._tables

    def extract_text(self):
        return self._text

    def filter(self, predicate):
        return _Page(self._filtered_text if self._filtered_text is not None else self._text)


class PdfPageTests(unittest.TestCase):
    def test_tables_are_rendered_as_markdown(self):
        page = _Page(
            "Body prose and jumbled table values",
            tables=[_Found([["Field", "Rule"], ["field_a", "rule_a"]], (0, 0, 100, 50))],
            filtered_text="Body prose",
        )
        out = F._pdf_page_markdown(page)
        self.assertIn("| Field | Rule |", out)
        self.assertIn("| field_a | rule_a |", out)

    def test_prose_is_re_extracted_without_the_table_region(self):
        page = _Page(
            "Body prose field_a rule_a",
            tables=[_Found([["Field", "Rule"], ["field_a", "rule_a"]], (0, 0, 100, 50))],
            filtered_text="Body prose",
        )
        out = F._pdf_page_markdown(page)
        self.assertTrue(out.startswith("Body prose"))
        self.assertEqual(out.count("field_a"), 1, "table cells must not also appear as prose")

    def test_page_without_tables_returns_plain_text(self):
        self.assertEqual(F._pdf_page_markdown(_Page("Just prose.")), "Just prose.")

    def test_empty_page_returns_empty_string(self):
        self.assertEqual(F._pdf_page_markdown(_Page("")), "")

    def test_table_extraction_failure_falls_back_to_text(self):
        class _Broken(_Page):
            def find_tables(self):
                raise RuntimeError("no table finder")

        self.assertEqual(F._pdf_page_markdown(_Broken("Fallback prose.")), "Fallback prose.")


class UnsupportedFormatTests(unittest.TestCase):
    def test_legacy_xls_is_rejected_with_guidance(self):
        with self.assertRaises(ValueError) as ctx:
            F.convert_file_to_markdown("book.xls", b"")
        self.assertIn(".xlsx", str(ctx.exception))

    def test_unknown_extension_is_rejected(self):
        with self.assertRaises(ValueError):
            F.convert_file_to_markdown("archive.zip", b"")

    def test_plain_text_gets_a_title_heading(self):
        out = F.convert_file_to_markdown("notes.txt", b"body line")
        self.assertTrue(out.startswith("# notes"))

    def test_markdown_with_its_own_heading_is_left_alone(self):
        out = F.convert_file_to_markdown("spec.md", b"# Existing Heading\n\nbody")
        self.assertTrue(out.startswith("# Existing Heading"))


if __name__ == "__main__":
    unittest.main()
