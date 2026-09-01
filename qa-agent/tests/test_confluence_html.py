"""
Confluence storage-format conversion: the content a plain html2text pass drops.

The property under test: every element of a PRD that carries requirement meaning
must reach the chunker as text. Code macros wrap their body in CDATA (which
HTMLParser reports as an unknown declaration and html2text discards), images hide
their only searchable trace in attributes, and macro titles sit inside the
ac:parameter elements the layout-junk strip removes.

Neutral placeholder content only — no tenant page titles, keys or hostnames.

Runs under the documented host-side command:
    cd qa-agent && PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'
"""
import re
import sys
import types
import unittest


def _install_stubs() -> None:
    """
    Minimal html2text stand-in: strips tags and unescapes entities.

    Everything this module is responsible for happens before or after html2text,
    so a naive renderer is enough to exercise the real conversion logic.
    """
    if "html2text" not in sys.modules:
        import html as _html

        mod = types.ModuleType("html2text")

        class _H2T:
            ignore_links = ignore_images = ignore_tables = False
            body_width = 0
            unicode_snob = True

            def handle(self, html):
                text = re.sub(r"<br\s*/?>", "\n", html, flags=re.IGNORECASE)
                text = re.sub(r"</(p|div|tr|h[1-6])>", "\n\n", text, flags=re.IGNORECASE)
                text = re.sub(r"<[^>]+>", "", text)
                return _html.unescape(text)

        mod.HTML2Text = _H2T
        sys.modules["html2text"] = mod


_install_stubs()
from ingestion import confluence_html as H  # noqa: E402
from ingestion.markdown_table import (  # noqa: E402
    rows_to_markdown, escape_cell, join_chunk_texts,
)


class CodeMacroTests(unittest.TestCase):
    """CDATA macro bodies were being discarded entirely before chunking."""

    MACRO = (
        '<ac:structured-macro ac:name="code">'
        '<ac:parameter ac:name="language">json</ac:parameter>'
        '<ac:plain-text-body><![CDATA[{"amount": 0, "currency": "XXX"}]]></ac:plain-text-body>'
        "</ac:structured-macro>"
    )

    def test_code_body_survives(self):
        out = H.storage_to_markdown(f"<p>Sample request:</p>{self.MACRO}")
        self.assertIn('{"amount": 0, "currency": "XXX"}', out)

    def test_code_is_fenced_with_its_language(self):
        out = H.storage_to_markdown(self.MACRO)
        self.assertIn("```json", out)
        self.assertTrue(out.rstrip().endswith("```"))

    def test_surrounding_prose_is_kept(self):
        out = H.storage_to_markdown(f"<p>Sample request:</p>{self.MACRO}")
        self.assertIn("Sample request:", out)

    def test_noformat_macro_is_handled_too(self):
        macro = (
            '<ac:structured-macro ac:name="noformat">'
            "<ac:plain-text-body><![CDATA[ERROR_CODE_1 timeout]]></ac:plain-text-body>"
            "</ac:structured-macro>"
        )
        self.assertIn("ERROR_CODE_1 timeout", H.storage_to_markdown(macro))

    def test_stray_cdata_outside_a_known_macro_is_rescued(self):
        out = H.storage_to_markdown("<p><![CDATA[retained text]]></p>")
        self.assertIn("retained text", out)

    def test_empty_code_macro_produces_no_fence(self):
        macro = (
            '<ac:structured-macro ac:name="code">'
            "<ac:plain-text-body><![CDATA[]]></ac:plain-text-body>"
            "</ac:structured-macro>"
        )
        self.assertNotIn("```", H.storage_to_markdown(macro))


class ImageTests(unittest.TestCase):
    """Alt text and attachment filename are the only searchable trace of a diagram."""

    def test_attachment_filename_is_kept(self):
        html = '<ac:image><ri:attachment ri:filename="flow-diagram.png"/></ac:image>'
        self.assertIn("flow-diagram.png", H.storage_to_markdown(html))

    def test_alt_text_is_kept(self):
        html = (
            '<ac:image ac:alt="State machine for retries">'
            '<ri:attachment ri:filename="flow.png"/></ac:image>'
        )
        out = H.storage_to_markdown(html)
        self.assertIn("State machine for retries", out)
        self.assertIn("flow.png", out)

    def test_external_image_url_keeps_the_basename(self):
        html = '<ac:image><ri:url ri:value="https://example.invalid/a/b/chart.svg"/></ac:image>'
        out = H.storage_to_markdown(html)
        self.assertIn("chart.svg", out)
        self.assertNotIn("example.invalid", out, "the host is noise, the filename is signal")

    def test_plain_img_tag_alt_is_kept(self):
        self.assertIn("Sequence overview",
                      H.storage_to_markdown('<img src="x.png" alt="Sequence overview"/>'))

    def test_image_without_metadata_still_marks_its_presence(self):
        self.assertIn("[Image]", H.storage_to_markdown("<ac:image></ac:image>"))


class MacroTitleTests(unittest.TestCase):
    """A status macro is nothing but its title; the ac:parameter strip took it."""

    def test_status_title_survives(self):
        html = (
            "<p>Requirement A "
            '<ac:structured-macro ac:name="status">'
            '<ac:parameter ac:name="title">EXAMPLE_STATUS</ac:parameter>'
            "</ac:structured-macro></p>"
        )
        self.assertIn("EXAMPLE_STATUS", H.storage_to_markdown(html))

    def test_expand_title_survives_with_its_body(self):
        html = (
            '<ac:structured-macro ac:name="expand">'
            '<ac:parameter ac:name="title">Edge cases</ac:parameter>'
            "<ac:rich-text-body><p>Body sentence.</p></ac:rich-text-body>"
            "</ac:structured-macro>"
        )
        out = H.storage_to_markdown(html)
        self.assertIn("Edge cases", out)
        self.assertIn("Body sentence.", out)

    def test_nested_macro_title_is_not_adopted_by_the_parent(self):
        """
        An expand wrapping a status macro must keep its own title. Searching the
        whole macro body finds the child's title first and mislabels the section.
        """
        html = (
            '<ac:structured-macro ac:name="expand">'
            '<ac:parameter ac:name="title">Outer Title</ac:parameter>'
            "<ac:rich-text-body><p>Body. "
            '<ac:structured-macro ac:name="status">'
            '<ac:parameter ac:name="title">EXAMPLE_STATUS</ac:parameter>'
            "</ac:structured-macro></p></ac:rich-text-body>"
            "</ac:structured-macro>"
        )
        out = H.storage_to_markdown(html)
        self.assertIn("Outer Title", out)
        self.assertIn("EXAMPLE_STATUS", out)
        self.assertLess(out.index("Outer Title"), out.index("Body."),
                        "the outer title must lead its own section")

    def test_layout_parameters_are_still_stripped(self):
        html = (
            '<ac:structured-macro ac:name="panel">'
            '<ac:parameter ac:name="class">wide760</ac:parameter>'
            "<ac:rich-text-body><p>Panel text.</p></ac:rich-text-body>"
            "</ac:structured-macro>"
        )
        out = H.storage_to_markdown(html)
        self.assertIn("Panel text.", out)
        self.assertNotIn("wide760", out)


class LinkTests(unittest.TestCase):
    def test_linked_page_title_is_kept(self):
        html = '<ac:link><ri:page ri:content-title="Related Spec"/></ac:link>'
        self.assertIn("Related Spec", H.storage_to_markdown(html))

    def test_anchor_text_is_kept_without_the_url(self):
        out = H.storage_to_markdown('<p>See <a href="https://example.invalid/x">the spec</a>.</p>')
        self.assertIn("the spec", out)


class TableTests(unittest.TestCase):
    """Rows must reach the chunker starting with '|' or its table handling never fires."""

    SIMPLE = (
        "<table><tbody>"
        "<tr><th>Field</th><th>Rule</th></tr>"
        "<tr><td>field_a</td><td>rule_a</td></tr>"
        "<tr><td>field_b</td><td>rule_b</td></tr>"
        "</tbody></table>"
    )

    def test_rows_start_with_a_pipe(self):
        lines = [l for l in H.storage_to_markdown(self.SIMPLE).split("\n") if l.strip()]
        self.assertTrue(all(l.startswith("|") for l in lines), lines)

    def test_header_and_delimiter_are_emitted(self):
        out = H.storage_to_markdown(self.SIMPLE)
        self.assertIn("| Field | Rule |", out)
        self.assertIn("| --- | --- |", out)

    def test_every_row_is_on_its_own_line(self):
        out = H.storage_to_markdown(self.SIMPLE)
        self.assertIn("| field_a | rule_a |", out.split("\n"))
        self.assertIn("| field_b | rule_b |", out.split("\n"))

    def test_rowspan_fills_down(self):
        html = (
            "<table><tbody>"
            "<tr><th>Module</th><th>Rule</th></tr>"
            '<tr><td rowspan="2">module_x</td><td>rule_1</td></tr>'
            "<tr><td>rule_2</td></tr>"
            "</tbody></table>"
        )
        lines = H.storage_to_markdown(html).split("\n")
        self.assertIn("| module_x | rule_1 |", lines)
        self.assertIn("| module_x | rule_2 |", lines,
                      "a merged cell must repeat, or the split-off row loses it")

    def test_rowspan_and_colspan_together(self):
        """
        A cell merged both across and down must carry down in every column it
        spans. Registering only the last one leaves the leading columns unfilled
        and shifts the following row left.
        """
        html = (
            "<table><tbody>"
            "<tr><th>A</th><th>B</th><th>C</th></tr>"
            '<tr><td rowspan="2" colspan="2">wide_tall</td><td>c1</td></tr>'
            "<tr><td>c2</td></tr>"
            "</tbody></table>"
        )
        lines = H.storage_to_markdown(html).split("\n")
        self.assertIn("| wide_tall | wide_tall | c1 |", lines)
        self.assertIn("| wide_tall | wide_tall | c2 |", lines)

    def test_rowspan_of_three_fills_two_following_rows(self):
        html = (
            "<table><tbody><tr><th>M</th><th>R</th></tr>"
            '<tr><td rowspan="3">m1</td><td>r1</td></tr>'
            "<tr><td>r2</td></tr><tr><td>r3</td></tr>"
            "</tbody></table>"
        )
        lines = H.storage_to_markdown(html).split("\n")
        for r in ("r1", "r2", "r3"):
            self.assertIn(f"| m1 | {r} |", lines)

    def test_colspan_pads_the_row(self):
        html = (
            "<table><tbody>"
            "<tr><th>A</th><th>B</th><th>C</th></tr>"
            '<tr><td colspan="2">wide</td><td>c</td></tr>'
            "</tbody></table>"
        )
        lines = H.storage_to_markdown(html).split("\n")
        self.assertIn("| wide | wide | c |", lines)

    def test_pipe_inside_a_cell_is_escaped(self):
        html = "<table><tbody><tr><th>H</th></tr><tr><td>a|b</td></tr></tbody></table>"
        out = H.storage_to_markdown(html)
        self.assertIn("| a\\|b |", out)

    def test_newlines_inside_a_cell_are_flattened(self):
        html = "<table><tbody><tr><th>H</th></tr><tr><td>one<br/>two</td></tr></tbody></table>"
        row = [l for l in H.storage_to_markdown(html).split("\n") if "one" in l]
        self.assertEqual(row, ["| one two |"])

    def test_code_macro_inside_a_cell_stays_inline(self):
        html = (
            "<table><tbody><tr><th>Field</th><th>Example</th></tr>"
            "<tr><td>amount</td><td>"
            '<ac:structured-macro ac:name="code">'
            '<ac:plain-text-body><![CDATA[{"amount": 0}]]></ac:plain-text-body>'
            "</ac:structured-macro></td></tr></tbody></table>"
        )
        out = H.storage_to_markdown(html)
        self.assertIn('{"amount": 0}', out)
        self.assertNotIn("```", out, "a fenced block inside a cell would break the row")

    def test_nested_table_cells_stay_readable_when_flattened(self):
        """
        A nested table cannot survive as a table (a markdown cell is one line),
        but its cells must not run together into one word soup.
        """
        html = (
            "<table><tbody><tr><th>H</th></tr><tr><td>"
            "<table><tbody><tr><td>inner_a</td><td>inner_b</td></tr></tbody></table>"
            "</td></tr></tbody></table>"
        )
        out = H.storage_to_markdown(html)
        self.assertIn("inner_a", out)
        self.assertIn("inner_b", out)
        self.assertNotIn("inner_ainner_b", out)
        self.assertIn("/", out, "nested cells are punctuated, not concatenated")

    def test_table_and_prose_both_survive(self):
        out = H.storage_to_markdown(f"<p>Intro line.</p>{self.SIMPLE}<p>Outro line.</p>")
        self.assertIn("Intro line.", out)
        self.assertIn("Outro line.", out)
        self.assertIn("| field_a | rule_a |", out)


class MarkdownTableHelperTests(unittest.TestCase):
    def test_ragged_rows_are_padded(self):
        out = rows_to_markdown([["a", "b", "c"], ["d"]])
        self.assertIn("| d |  |  |", out)

    def test_delimiter_width_matches_header(self):
        out = rows_to_markdown([["a", "b", "c"], ["d", "e", "f"]])
        self.assertIn("| --- | --- | --- |", out)

    def test_empty_grid_returns_empty_string(self):
        self.assertEqual(rows_to_markdown([]), "")
        self.assertEqual(rows_to_markdown([["", ""]]), "")

    def test_nbsp_is_normalised(self):
        self.assertEqual(escape_cell("a b"), "a b")

    def test_none_cell_is_empty(self):
        self.assertEqual(escape_cell(None), "")


class ChunkRejoinTests(unittest.TestCase):
    """
    The inverse of the chunker's header carry, used when chunks are re-joined into
    one document. Over-stripping would silently delete a real table's header.
    """

    HEAD = "| Field | Rule |\n| --- | --- |"

    def _parts(self, n):
        return [f"{self.HEAD}\n| field_{i} | rule_{i} |" for i in range(n)]

    def test_repeated_header_appears_once(self):
        out = join_chunk_texts(self._parts(3))
        self.assertEqual(out.count("| Field | Rule |"), 1)
        self.assertEqual(out.count("| --- | --- |"), 1)

    def test_no_row_is_lost(self):
        out = join_chunk_texts(self._parts(4))
        for i in range(4):
            self.assertIn(f"| field_{i} | rule_{i} |", out)

    def test_strip_continues_past_the_second_chunk(self):
        """Once chunk 1's header is stripped, chunk 2 has nothing to compare against."""
        out = join_chunk_texts(self._parts(5))
        self.assertEqual(out.count("| Field | Rule |"), 1)

    def test_rows_stay_contiguous(self):
        lines = [l for l in join_chunk_texts(self._parts(3)).split("\n") if l.strip()]
        self.assertEqual(lines[2:], [f"| field_{i} | rule_{i} |" for i in range(3)])

    def test_a_different_table_keeps_its_header(self):
        out = join_chunk_texts([
            f"{self.HEAD}\n| field_a | rule_a |",
            "| Other | Columns |\n| --- | --- |\n| x | y |",
        ])
        self.assertIn("| Field | Rule |", out)
        self.assertIn("| Other | Columns |", out)

    def test_first_chunk_may_lead_with_prose_before_the_table(self):
        """
        The opening chunk of a table is usually mixed — lead-in prose, then the
        table — so its header is not at the top. Tracking only leading headers
        misses the run entirely and every continuation keeps its copy.
        """
        out = join_chunk_texts([
            f"Lead-in sentence.\n{self.HEAD}\n| field_a | rule_a |",
            f"{self.HEAD}\n| field_b | rule_b |",
            f"{self.HEAD}\n| field_c | rule_c |",
        ])
        self.assertEqual(out.count("| Field | Rule |"), 1)
        self.assertIn("Lead-in sentence.", out)
        for row in ("| field_a | rule_a |", "| field_b | rule_b |", "| field_c | rule_c |"):
            self.assertIn(row, out)

    def test_continuation_rows_are_not_separated_by_blank_lines(self):
        """A blank line between rows splits the table again, undoing the strip."""
        out = join_chunk_texts(self._parts(3))
        body = out.split("| --- | --- |\n", 1)[1]
        self.assertNotIn("\n\n", body)

    def test_prose_between_tables_ends_the_run(self):
        out = join_chunk_texts([
            f"{self.HEAD}\n| field_a | rule_a |",
            "An intervening sentence.",
            f"{self.HEAD}\n| field_b | rule_b |",
        ])
        self.assertEqual(out.count("| Field | Rule |"), 2,
                         "a table after prose is a new table and keeps its header")

    def test_leading_prose_chunk_is_untouched(self):
        out = join_chunk_texts(["Lead-in prose.", f"{self.HEAD}\n| field_a | rule_a |"])
        self.assertTrue(out.startswith("Lead-in prose."))
        self.assertIn("| Field | Rule |", out)

    def test_header_without_a_delimiter_row_is_still_deduplicated(self):
        out = join_chunk_texts([
            "| Field | Rule |\n| field_a | rule_a |",
            "| Field | Rule |\n| field_b | rule_b |",
        ])
        self.assertEqual(out.count("| Field | Rule |"), 1)

    def test_header_only_continuation_chunk_is_dropped(self):
        out = join_chunk_texts([f"{self.HEAD}\n| field_a | rule_a |", self.HEAD])
        self.assertEqual(out.count("| Field | Rule |"), 1)
        self.assertTrue(out.rstrip().endswith("| field_a | rule_a |"))

    def test_prose_only_input_is_joined_unchanged(self):
        self.assertEqual(join_chunk_texts(["First part.", "Second part."]),
                         "First part.\n\nSecond part.")

    def test_empty_texts_are_skipped(self):
        self.assertEqual(join_chunk_texts(["", "  ", "Body."]), "Body.")

    def test_empty_input_yields_empty_string(self):
        self.assertEqual(join_chunk_texts([]), "")


class EmptyInputTests(unittest.TestCase):
    def test_empty_html_returns_empty_string(self):
        self.assertEqual(H.storage_to_markdown(""), "")
        self.assertEqual(H.storage_to_markdown("   "), "")


if __name__ == "__main__":
    unittest.main()
