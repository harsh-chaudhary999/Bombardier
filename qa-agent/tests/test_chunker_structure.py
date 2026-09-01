"""
Structure preservation in the chunker: tables, fenced code, and list items.

The property under test: content whose meaning lives in its LAYOUT must survive
being split. A table row is worthless without its header; a fenced block is
worthless without its closing fence. Chunks are joined text, so a wrong join
character silently destroys both.

Assertions are on shape only — neutral placeholder content throughout.

Runs under the documented host-side command:
    cd qa-agent && PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'
"""
import unittest

from tests import stubs

stubs.install_chunker_deps()

from ingestion import chunker as C  # noqa: E402


def _table(n_rows: int) -> str:
    head = "| Field | Rule | Owner |\n| --- | --- | --- |"
    rows = "\n".join(f"| field_{i} | rule_{i} | team_{i % 3} |" for i in range(n_rows))
    return f"{head}\n{rows}"


class SegmentationTests(unittest.TestCase):
    """_segment_body must tag structure so the joiner can preserve it."""

    def test_table_rows_are_individual_segments(self):
        segs = C._segment_body(_table(4))
        kinds = {s["kind"] for s in segs}
        self.assertEqual(kinds, {C._TABLE})
        self.assertEqual(len(segs), 6, "header + delimiter + 4 body rows")

    def test_header_and_delimiter_are_flagged_as_header(self):
        segs = C._segment_body(_table(3))
        self.assertEqual(segs[0]["header_index"], 0, "the header row itself")
        self.assertEqual(segs[1]["header_index"], 1, "the '| --- |' delimiter")
        self.assertIsNone(segs[2]["header_index"], "a data row")

    def test_every_row_carries_the_header_block(self):
        segs = C._segment_body(_table(3))
        for seg in segs:
            self.assertEqual(seg["table_header_lines"][0], "| Field | Rule | Owner |")
            self.assertEqual(seg["table_header_lines"][1], "| --- | --- | --- |")

    def test_fenced_code_block_is_one_segment(self):
        body = 'Intro sentence.\n\n```json\n{\n  "a": 1,\n\n  "b": 2\n}\n```\n\nTrailing sentence.'
        segs = C._segment_body(body)
        code = [s for s in segs if s["kind"] == C._CODE]
        self.assertEqual(len(code), 1, "a blank line inside the fence must not split it")
        self.assertTrue(code[0]["text"].startswith("```json"))
        self.assertTrue(code[0]["text"].rstrip().endswith("```"))

    def test_list_items_are_tagged(self):
        segs = C._segment_body("Preamble.\n- first item\n- second item\n")
        self.assertEqual([s["kind"] for s in segs], [C._PROSE, C._LIST, C._LIST])

    def test_prose_is_still_sentence_split(self):
        segs = C._segment_body("One sentence. Two sentence. Three.")
        self.assertEqual(len(segs), 3)
        self.assertTrue(all(s["kind"] == C._PROSE for s in segs))


class JoinTests(unittest.TestCase):
    """Prose joins with a space; anything structural joins with a newline."""

    def test_prose_pairs_join_with_space(self):
        segs = [C._segment("First.", C._PROSE), C._segment("Second.", C._PROSE)]
        self.assertEqual(C._join_segments(segs), "First. Second.")

    def test_table_rows_join_with_newline(self):
        segs = C._segment_body(_table(2))
        joined = C._join_segments(segs)
        self.assertNotIn("| |", joined, "rows must not be flattened onto one line")
        self.assertEqual(len(joined.split("\n")), 4)

    def test_list_items_join_with_newline(self):
        segs = C._segment_body("- alpha\n- beta")
        self.assertEqual(C._join_segments(segs), "- alpha\n- beta")


class TableHeaderCarryTests(unittest.TestCase):
    """A continuation chunk must re-emit the header it was split away from."""

    def _chunks(self, n_rows: int) -> list[dict]:
        return C.chunk_document(f"# Spec\n\n## Validation Rules\n\n{_table(n_rows)}\n", "t:1")

    def test_long_table_splits_into_several_chunks(self):
        self.assertGreater(len(self._chunks(400)), 1)

    def test_every_chunk_starts_with_the_header(self):
        for chunk in self._chunks(400):
            first = chunk["chunk_text"].split("\n", 1)[0]
            self.assertEqual(first, "| Field | Rule | Owner |")

    def test_every_chunk_has_a_delimiter_row(self):
        for chunk in self._chunks(400):
            self.assertIn("| --- | --- | --- |", chunk["chunk_text"])

    def test_header_is_not_duplicated_in_the_first_chunk(self):
        first = self._chunks(400)[0]["chunk_text"]
        self.assertEqual(first.count("| Field | Rule | Owner |"), 1)

    def test_rows_are_not_flattened_onto_one_line(self):
        for chunk in self._chunks(400):
            for line in chunk["chunk_text"].split("\n"):
                self.assertLessEqual(
                    line.count("|"), 4, f"more than one row on a line: {line[:80]!r}"
                )

    def test_no_row_is_lost_across_the_split(self):
        chunks = self._chunks(400)
        seen = set()
        for chunk in chunks:
            for line in chunk["chunk_text"].split("\n"):
                if line.startswith("| field_"):
                    seen.add(line.split(" | ")[0])
        self.assertEqual(len(seen), 400)


class OversizedSegmentTests(unittest.TestCase):
    """A single segment above the budget breaks at line boundaries, not mid-line."""

    def test_giant_code_block_keeps_fences_on_every_piece(self):
        inner = "\n".join(f'  "key_{i}": "value_{i}",' for i in range(600))
        segs = C._segment_body(f"```json\n{inner}\n```")
        pieces = C._explode_oversized(segs, C.MAX_TOKENS)
        self.assertGreater(len(pieces), 1)
        for piece in pieces:
            self.assertTrue(piece["text"].startswith("```json"))
            self.assertTrue(piece["text"].rstrip().endswith("```"))

    # Fixtures use many whitespace-separated words rather than one long run of
    # characters: this suite shares sys.modules with other test files, and their
    # tiktoken stub counts whitespace tokens while this one counts characters.
    # Content that is only large under one of them makes the test order-dependent.
    @staticmethod
    def _bulk(words: int, tag: str = "w") -> str:
        return " ".join(f"{tag}{i}" for i in range(words))

    def test_wide_row_splits_by_column_into_valid_rows(self):
        """
        Character-slicing an over-long row yields fragments that are not rows at
        all. Splitting by column keeps every piece a valid, self-describing table.
        """
        row = "| " + " | ".join(self._bulk(400, f"c{c}x") for c in range(4)) + " |"
        segs = C._segment_body(f"| c0 | c1 | c2 | c3 |\n| --- | --- | --- | --- |\n{row}")
        pieces = C._explode_oversized(segs, C.MAX_TOKENS)

        wide = [p for p in pieces if p["header_index"] is None]
        self.assertGreater(len(wide), 1, "the row must actually be split")
        for piece in wide:
            self.assertTrue(piece["text"].startswith("|"))
            self.assertTrue(piece["text"].rstrip().endswith("|"))
            self.assertEqual(piece["kind"], C._TABLE)

    def test_wide_row_pieces_carry_their_slice_of_the_header(self):
        row = "| " + " | ".join(self._bulk(400, f"c{c}x") for c in range(4)) + " |"
        segs = C._segment_body(
            f"| alpha | beta | gamma | delta |\n| --- | --- | --- | --- |\n{row}")
        pieces = [p for p in C._explode_oversized(segs, C.MAX_TOKENS)
                  if p["header_index"] is None]

        names = []
        for piece in pieces:
            lines = piece["table_header_lines"]
            self.assertIsNotNone(lines, "each piece needs its own header")
            self.assertEqual(len(C._row_cells(lines[0])),
                             len(C._row_cells(piece["text"])),
                             "header and row must have the same column count")
            names.extend(C._row_cells(lines[0]))
        self.assertEqual(names, ["alpha", "beta", "gamma", "delta"],
                         "columns must be partitioned, not duplicated or dropped")

    def test_single_huge_cell_falls_back_to_line_splitting(self):
        row = "| " + self._bulk(3000) + " |"
        segs = C._segment_body(f"| only |\n| --- |\n{row}")
        pieces = C._explode_oversized(segs, C.MAX_TOKENS)
        self.assertGreater(len(pieces), 2, "must still be broken up somehow")

    def test_escaped_pipe_does_not_split_a_cell(self):
        self.assertEqual(C._row_cells(r"| a\|b | c |"), [r"a\|b", "c"])

    def test_line_splitting_preserves_whole_lines(self):
        text = "\n".join(f"line number {i} with some filler text" for i in range(500))
        pieces = C._split_text_by_lines(text, 200)
        self.assertGreater(len(pieces), 1)
        rejoined = "\n".join(pieces).split("\n")
        self.assertTrue(all(p.startswith("line number") for p in rejoined if p))


class SemanticModeTests(unittest.TestCase):
    """Semantic mode must not spend embeddings on table rows or split inside them."""

    def setUp(self):
        self.embedded: list[str] = []

    def _embed(self, texts):
        self.embedded.extend(texts)
        # Deterministic, content-derived, unit-ish vectors.
        return [[1.0, (abs(hash(t)) % 100) / 100.0, 0.5] for t in texts]

    def test_boundary_similarities_scores_prose_pairs_only(self):
        """
        Direct check on the boundary logic, independent of the chunk assembly —
        a fallback triggered by a broken numpy stub would otherwise hide this.
        """
        segs = C._segment_body("Alpha sentence. Beta sentence.\n\n" + _table(2))
        sims = C._boundary_similarities(segs, self._embed)
        self.assertIsNotNone(sims)
        self.assertEqual(len(sims), len(segs) - 1)
        prose_pairs = [
            i for i in range(len(segs) - 1)
            if segs[i]["kind"] == C._PROSE and segs[i + 1]["kind"] == C._PROSE
        ]
        self.assertTrue(prose_pairs, "fixture must contain at least one prose pair")
        for i in range(len(segs) - 1):
            if i not in prose_pairs:
                self.assertEqual(sims[i], 1.0, "non-prose pairs must never split on similarity")

    def test_boundary_similarities_returns_none_for_an_all_table_section(self):
        self.assertIsNone(C._boundary_similarities(C._segment_body(_table(5)), self._embed))

    def test_table_rows_are_not_embedded(self):
        C.chunk_document(f"# S\n\n## Rules\n\n{_table(120)}\n", "t:2", embed_fn=self._embed)
        self.assertEqual(
            [t for t in self.embedded if t.startswith("|")], [],
            "table rows must not be scored for topic boundaries",
        )

    def test_prose_is_still_embedded(self):
        body = " ".join(f"Requirement {i} describes behaviour {i}." for i in range(40))
        C.chunk_document(f"# S\n\n## Notes\n\n{body}\n", "t:3", embed_fn=self._embed)
        self.assertGreater(len(self.embedded), 0)

    def test_semantic_chunks_keep_the_table_header(self):
        chunks = C.chunk_document(
            f"# S\n\n## Rules\n\n{_table(400)}\n", "t:4", embed_fn=self._embed
        )
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertTrue(chunk["chunk_text"].startswith("| Field | Rule | Owner |"))

    def test_embedding_failure_falls_back_without_raising(self):
        def _boom(texts):
            raise RuntimeError("embedding backend down")

        body = " ".join(f"Sentence {i} of the section." for i in range(60))
        chunks = C.chunk_document(f"# S\n\n## Notes\n\n{body}\n", "t:5", embed_fn=_boom)
        self.assertGreater(len(chunks), 0)


class HorizontalRuleTests(unittest.TestCase):
    """
    '---' is a markdown rule, a YAML document separator AND a front-matter
    delimiter. Stripping it blindly guts any YAML sample in a PRD.
    """

    def test_yaml_separators_inside_a_fence_survive(self):
        doc = "# D\n\n## Config\n\n```yaml\n---\nkey: value\n---\nother: thing\n```\n"
        text = C.chunk_document(doc, "t:hr1")[0]["chunk_text"]
        self.assertEqual(text.count("---"), 2)
        self.assertIn("key: value", text)

    def test_prose_horizontal_rules_are_still_stripped(self):
        doc = "# D\n\n## S\n\nFirst line.\n\n---\n\nSecond line.\n"
        text = C.chunk_document(doc, "t:hr2")[0]["chunk_text"]
        self.assertNotIn("---", text)
        self.assertIn("First line.", text)
        self.assertIn("Second line.", text)

    def test_table_delimiter_is_never_stripped(self):
        text = C.chunk_document(f"# D\n\n## S\n\n{_table(3)}\n", "t:hr3")[0]["chunk_text"]
        self.assertIn("| --- | --- | --- |", text)

    def test_rule_after_a_fence_closes_is_stripped(self):
        doc = "# D\n\n## S\n\n```py\nx = 1\n```\n\n---\n\nAfter.\n"
        text = C.chunk_document(doc, "t:hr4")[0]["chunk_text"]
        self.assertIn("x = 1", text)
        self.assertNotIn("\n---\n", text)


class DelimiterStartChunkTests(unittest.TestCase):
    """
    A chunk can begin at the '| --- |' delimiter rather than the header row.
    It needs the header line above it — not the whole block, which would
    duplicate the delimiter.
    """

    HEADER = ["| Field | Rule |", "| --- | --- |"]

    def _materialize(self, segments):
        return C._materialize("S", segments, "parent")["text"]

    def test_chunk_starting_at_the_delimiter_gets_the_header_line(self):
        seg = C._segment("| --- | --- |", C._TABLE,
                         table_header_lines=self.HEADER, header_index=1)
        row = C._segment("| a | b |", C._TABLE,
                         table_header_lines=self.HEADER, header_index=None)
        out = self._materialize([seg, row])
        self.assertEqual(out.split("\n")[0], "| Field | Rule |")
        self.assertEqual(out.count("| --- | --- |"), 1, "delimiter must not double")

    def test_chunk_starting_at_the_header_is_untouched(self):
        segs = [
            C._segment(self.HEADER[0], C._TABLE,
                       table_header_lines=self.HEADER, header_index=0),
            C._segment(self.HEADER[1], C._TABLE,
                       table_header_lines=self.HEADER, header_index=1),
        ]
        self.assertEqual(self._materialize(segs), "\n".join(self.HEADER))

    def test_chunk_starting_at_a_data_row_gets_the_whole_block(self):
        row = C._segment("| a | b |", C._TABLE,
                         table_header_lines=self.HEADER, header_index=None)
        self.assertEqual(self._materialize([row]), "| Field | Rule |\n| --- | --- |\n| a | b |")


class ChunkTypeTests(unittest.TestCase):
    """chunk_type labels what a chunk holds so consumers can skip re-parsing it."""

    def test_pure_table_is_labelled_table(self):
        self.assertEqual(C.classify_chunk_text(_table(3)), "table")

    def test_pure_prose_is_labelled_prose(self):
        self.assertEqual(C.classify_chunk_text("A sentence. Another sentence."), "prose")

    def test_list_only_chunk_is_prose(self):
        self.assertEqual(C.classify_chunk_text("- alpha\n- beta"), "prose")

    def test_pure_code_is_labelled_code(self):
        self.assertEqual(C.classify_chunk_text('```json\n{"a": 1}\n```'), "code")

    def test_prose_plus_table_is_mixed(self):
        self.assertEqual(C.classify_chunk_text("Lead-in line.\n" + _table(2)), "mixed")

    def test_pipes_inside_a_code_fence_do_not_count_as_a_table(self):
        text = "```sh\n| grep foo\n| wc -l\n```"
        self.assertEqual(C.classify_chunk_text(text), "code")

    def test_empty_text_is_prose(self):
        self.assertEqual(C.classify_chunk_text(""), "prose")

    def test_chunk_document_labels_every_chunk(self):
        doc = f"# D\n\n## Rules\n\n{_table(400)}\n"
        chunks = C.chunk_document(doc, "t:9")
        self.assertTrue(all(c["chunk_type"] == "table" for c in chunks))

    def test_prose_section_is_labelled_prose(self):
        body = " ".join(f"Statement {i} about behaviour." for i in range(40))
        chunks = C.chunk_document(f"# D\n\n## Notes\n\n{body}\n", "t:10")
        self.assertTrue(all(c["chunk_type"] == "prose" for c in chunks))


class MixedContentTests(unittest.TestCase):
    """A realistic PRD section: prose, a table, and a code sample together."""

    DOC = (
        "# Feature PRD\n\n"
        "## Limits\n\n"
        "The limits below are enforced at request time.\n\n"
        + _table(3)
        + "\n\n## Payload\n\n"
        '```json\n{\n  "amount": 0,\n  "currency": "XXX"\n}\n```\n\n'
        "Validate the amount against the limits table.\n"
    )

    def test_table_survives_intact(self):
        chunks = C.chunk_document(self.DOC, "t:6")
        joined = "\n".join(c["chunk_text"] for c in chunks)
        self.assertIn("| Field | Rule | Owner |\n| --- | --- | --- |", joined)

    def test_code_block_survives_intact(self):
        chunks = C.chunk_document(self.DOC, "t:7")
        joined = "\n".join(c["chunk_text"] for c in chunks)
        self.assertIn('```json\n{\n  "amount": 0,', joined)
        self.assertIn("```", joined)

    def test_sections_are_still_separated_by_heading(self):
        headings = {c["section_heading"] for c in C.chunk_document(self.DOC, "t:8")}
        self.assertEqual(headings, {"Limits", "Payload"})


if __name__ == "__main__":
    unittest.main()
