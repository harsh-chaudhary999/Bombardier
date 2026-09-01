# Mistakes Log — do not repeat these

Negative guidance. Every entry is a real bug that shipped or was caught late in this repo,
not a hypothetical. Each says what went wrong, why it was invisible, and what to do instead.

The recurring theme: **the worst bugs here are silent**. Content that is dropped during
ingestion produces no error — it produces a smaller index and worse answers, months later.
Prefer failures that are loud.

---

## 1. Library assumptions that silently drop content

### Never assume html2text can see Confluence storage format

`html2text` discards `<![CDATA[…]]>`. Python's `HTMLParser` reports it as `unknown_decl`,
which html2text does not implement. Confluence wraps **every code/noformat macro body** in
CDATA — so every JSON payload, SQL snippet and config sample in every PRD was dropped before
chunking ever ran. It looked like the pages just did not have much code in them.

Same class of failure in the same converter:
- `ignore_images = True` discarded image **alt text** and attachment filenames, which are
  usually a diagram's only searchable trace.
- `<ac:image><ri:attachment ri:filename="…"/></ac:image>` produced **zero** parser events —
  the `ac:` strip left a bare `ri:` tag that emits nothing.
- Blanket-stripping `<ac:parameter>` to kill layout junk like `wide760` also deleted status
  and expand macro **titles**.

**Do instead.** Lift content out *before* the tag strip and before html2text, as
`ingestion/confluence_html.py` does. Verify with a real parser, not by reading the docs:

```python
from html.parser import HTMLParser
class P(HTMLParser):
    def unknown_decl(self, d): print("DROPPED:", d[:40])
```

### `doc.paragraphs` does not include Word table content

`python-docx`'s `Document.paragraphs` returns only `w:p` children of `w:body`. Paragraphs
inside `w:tbl` are excluded. A `.docx` whose requirements live in tables ingested as
headings and nothing else — the requirements themselves were never indexed.

**Do instead.** Walk `doc.iter_inner_content()` (python-docx ≥ 1.1) for document-order
paragraphs *and* tables.

### Declared dependencies that nothing imports

`pymupdf` and `markdownify` were pinned in `requirements.txt` for months and imported
nowhere. `pdfplumber` was pinned with the comment "PDF table extraction" while
`extract_tables()` was never called — the PDF path used `extract_text()` only, which
flattens a table into whitespace-separated values with no column association.

**Do instead.** `grep -rn "<module>" --include=*.py .` before trusting a dependency comment.
A pin plus a comment is not evidence the capability exists.

---

## 2. Structure-destroying transformations

### Never `" ".join()` chunk segments

The chunker joined all segments with a space. A table that split across chunks came out as
one flat run of pipes on a single line, and bullet lists were mashed together.

**Do instead.** Space between two prose segments; newline for any join involving a table
row, list item or code block (`_join_segments`).

### A continuation chunk without its table header is near-worthless

A split table's later chunks began `| Pro | 5000 | Yes |` with no column names. Nothing told
the embedder that `5000` was a refund ceiling. Both dense and keyword retrieval degraded.

**Do instead.** Repeat the header in every continuation chunk (ADR-009), and strip it again
on reassembly (ADR-010).

### Never strip `---` without tracking fence state

`re.sub(r'(?m)^-{3,}\s*$', '', text)` removes markdown horizontal rules — and YAML document
separators, and front-matter delimiters. Any YAML sample in a PRD was quietly gutted:

```
'```yaml\n\nkey: value\n\nother: thing\n```'      # separators gone
```

This bug pre-dated the code that exposed it: while CDATA bodies were being dropped, no code
block ever reached the strip. Fixing extraction made a latent bug live.

**Do instead.** `_strip_horizontal_rules` tracks fence state. Note the general lesson: fixing
an upstream drop can activate downstream bugs that were never reachable before.

### A blank line between table rows splits the table again

Re-joining chunks with the `"\n\n"` paragraph separator put a blank line between rows, which
markdown reads as two tables — silently undoing the header strip it had just done.

**Do instead.** Join genuine continuations with `"\n"`.

---

## 3. Off-by-one and state-tracking errors in table handling

All three shipped in the same pass. Tables are where the fiddly bugs live; test them with
real multi-chunk fixtures, not two-row examples.

| Bug | Symptom | Fix |
|---|---|---|
| `rowspan` + `colspan` on one cell registered the carry-down for **only the last** spanned column | The following row shifted left: `\| c2 \| wide_tall \| \|` instead of `\| wide_tall \| wide_tall \| c2 \|` | Register pending carry-down for every column the cell spans |
| Header de-duplication compared each chunk against the **previous** chunk only | Worked once; chunk 2 had nothing to match against because chunk 1's header had just been stripped | Track the *active* header across the whole run |
| The active header was read from a chunk's **leading** lines | A table's first chunk is usually mixed (prose lead-in, then the table), so the run was never registered and every continuation kept its header | Read the header from the chunk's *trailing* table |
| Cells were escaped in the table reader **and** again in `rows_to_markdown` | `a\|b` → `a\\\|b` | Escape in exactly one place — `rows_to_markdown` owns it |
| `_materialize` treated "is a header line" as all-or-nothing | A chunk starting at the `\| --- \|` delimiter got no header, or a duplicated delimiter | Record `header_index` (0 / 1 / None) and prepend only the missing lines |

### Excel: never infer the header row from the loop index

`if i == 0:` used the enumerate index over *all* rows, but blank rows were skipped before
that check. A sheet with a leading blank row therefore emitted **no `| --- |` delimiter at
all**, so the output was not a valid markdown table.

**Do instead.** Track whether a header has been emitted, not which iteration you are on.

---

## 4. Silent divergence between two systems that must agree

### The embedder's cap sat below the chunker's maximum chunk

`format_prd_chunk` truncated at 4000 chars while chunks could reach ~4800. Elasticsearch
indexed the full `chunk_text` for BM25 regardless. The tail of a large chunk was therefore
searchable by keyword but absent from its vector — dense and sparse retrieval saw different
documents, with no error anywhere.

**Do instead.** `QA_EMBED_MAX_CHARS` must exceed `SEMANTIC_MAX × ~4` plus prefixes, and
truncation logs a warning. See ADR-018.

### A feature flag honoured on only one code path

`QA_CONFLUENCE_INGEST_ATTACHMENTS` was wired into single-page ingest but not the space
crawl. With the flag on, whether a page's attachment content was indexed depended on *how
that page happened to be ingested*.

**Do instead.** When adding a flag, grep for every path that builds the same artifact and
wire them all, or explicitly document the asymmetry. A partially-honoured flag is worse than
an absent feature — it is invisible and non-reproducible.

### Undocumented deliberate asymmetry reads as a bug

The space crawl chunks fixed-window while single-page ingest chunks semantically (ADR-015).
That is a correct throughput tradeoff, but it was unwritten, so "the same page produces
different chunks" looked like a defect and cost a round of investigation.

**Do instead.** Document deliberate asymmetries where the code diverges.

---

## 5. Test-suite anti-patterns

Test modules share one `sys.modules`. Several install stubs, and **the first module to run
wins**. This has caused four separate false results.

### Never stub an in-repo package

`install_observability()` registered an `observability` stub module with an empty
`__path__`, shadowing the real in-repo package. Every later
`from observability.canonical_json import …` then failed — but only in some module orders,
so the suite passed and isolated runs did not.

`observability` imports cleanly with no third-party deps and `trace.event()` is a no-op
unless `QA_TRACE=1`. The stub bought nothing and cost correctness.

**Do instead.** Stub third-party packages only. If an in-repo module is importable, import it.

### Never guard stubs with `if "<module>" not in sys.modules`

Another module's inert stand-in wins, and yours never installs. When that happened, the
chunker's `_cosine_sim` raised, the chunker caught it and silently fell back to fixed-window
chunking — so the semantic tests passed **without ever exercising the semantic path**.

**Do instead.** Install per *attribute*, probing whether each one actually works
(`tests/stubs.py`). And when production code catches an exception and degrades, assert that
the degraded path was *not* taken.

### Never size a fixture in units one stub does not share

`tests/test_context_budget.py` tokenizes on whitespace; `tests/stubs.py` counts characters.
Fixtures sized for one are wrong under the other:
- `"z" * 900` is one whitespace token → the split under test never happened.
- `"Parent body."` is 2 whitespace tokens → below the chunker's 3-token minimum, so the
  document vanished entirely and the test failed only in the full suite.

**Do instead.** Make fixtures large under **both** — many whitespace-separated words, never
one long run of characters.

### Always run both the suite and each module alone

Several modules only passed because of alphabetical discovery order. See `flow.md` §2 for
the loop. A green `unittest discover` is not sufficient evidence.

### Match endpoints precisely in test fakes

A fake matched `"/attachments" in url`, but attachment *download* URLs also contain
`/attachments` — so every download returned the listing JSON. The test failed for a reason
that had nothing to do with the code under test.

It did expose a real gap (an empty attachment indexed as a bare heading), but that was luck.

**Do instead.** Anchor fakes on the specific path shape, e.g.
`"/api/v2/pages/" in url and url.endswith("/attachments")`.

---

## 6. Process mistakes

### Do not claim completeness without probing for it

"No bugs, nothing pending" was asserted once and was wrong: a focused probe pass immediately
found four real bugs (YAML separators, rowspan+colspan, delimiter-start chunks, an Excel
memory regression). Write a throwaway probe script and try the awkward inputs — unclosed
fences, nested tables, merged cells, two different tables in one section, a row larger than
the chunk budget.

### Do not fix one direction of a reversible transformation

Adding the header carry (ADR-009) created a reassembly problem (ADR-010) that was only found
by tracing a full round-trip. When you add a transformation for one consumer, find the other
consumers of the same data.

### Do not "improve" a dependency choice without checking memory implications

Switching Excel to `read_only=False` for merged-cell support was correct for correctness and
wrong for large files — openpyxl then loads the entire workbook, and the existing row cap
limits only what is *appended*, not what is *loaded*. Size-gated now (ADR-016).
