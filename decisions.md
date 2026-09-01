# Architecture Decision Record

Every entry states **what** was decided, **why** (context and benefits), and **why the
alternatives were rejected**. Newest sections last within each area. If you are about to
change something here, read the rejected alternatives first — most of them were rejected
for a reason that still holds.

---

## 1. Integrations

### ADR-001 — Jira/Xray goes through an external MCP endpoint, not an in-repo client

**Decision.** All Jira/Xray operations are HTTP calls to a Streamable-HTTP MCP server that
runs *outside* this repo (`XRAY_MCP_URL`). `integrations/xray_client.py` calls *logical*
operations (`get_test`, `search_issues`, …) resolved through `_TOOL_SPEC_DEFAULTS` and
overridable at runtime with `XRAY_MCP_TOOL_MAP`.

**Why.** Tool names and argument shapes differ between MCP server builds. Treating them as
configuration means a deployment with different names needs an env var, not a code change
and a redeploy. `GET /integrations/mcp/tools` verifies the mapping against the live server.

**Alternatives rejected.**
- *Vendoring a Jira/Xray REST client.* Re-implements auth, pagination and rate-limit
  handling that the MCP server already owns, and couples our release cycle to Atlassian API
  changes.
- *Hard-coding MCP tool names.* Every server variation becomes a code fork. The failure mode
  is also terrible: a renamed tool surfaces as a runtime error deep inside an analysis run.

### ADR-002 — GitLab ingestion uses the REST API directly, with no MCP hop

**Decision.** `ingestion/gitlab_ingestor.py` calls the GitLab REST API directly.

**Why.** Ingestion is a bulk read of many files. An MCP hop adds a process boundary and a
timeout surface to an operation that is a plain paginated tree walk plus file fetches.

**Alternatives rejected.**
- *A GitLab MCP server for symmetry with Xray.* Symmetry is not a benefit here. Xray is
  behind MCP because its tool surface is volatile and shared; GitLab file reads are stable
  and private to ingestion. Adding a second MCP dependency doubles the "is the server up?"
  failure mode for no gain.

---

## 2. Retrieval pipeline

### ADR-003 — Two-stage retrieval: RRF hybrid, then a cross-encoder rerank

**Decision.** Elasticsearch RRF retriever fuses KNN (dense) and BM25 (keyword)
(`rank_constant=60`, `num_candidates=max(100, k*10)`), then
`cross-encoder/ms-marco-MiniLM-L-6-v2` reranks the top ~100 down to ~50.

**Why.** Dense retrieval alone misses exact identifiers (issue keys, field names, error
codes) that QA content is full of; BM25 alone misses paraphrase. RRF needs no score
normalisation between the two. The cross-encoder then sees query and document together,
which bi-encoder similarity cannot.

**Alternatives rejected.**
- *Dense-only.* Fails on the exact-token lookups that dominate test-case search.
- *Weighted score blending instead of RRF.* Requires normalising two incomparable score
  scales and retuning the weights whenever either side changes. RRF is rank-based and
  needs no such tuning.
- *Reranking everything.* The cross-encoder is quadratic in practice; reranking the full
  candidate set is the dominant cost with negligible gain past the top ~100.

### ADR-004 — bge-m3 with asymmetric encoding

**Decision.** `BAAI/bge-m3` (1024 dims). Queries are encoded with the prefix
`"Represent this sentence for searching relevant passages: "`; documents are encoded
without it. Query vectors are LRU-cached (512 entries).

**Why.** The model is trained for this asymmetry — mixing the two degrades retrieval. It is
multilingual, which matters for mixed-language PRDs.

**Alternatives rejected.**
- *Applying the instruction prefix to both sides, or neither.* Measurably worse; the
  asymmetry is the trained behaviour, not a stylistic choice.
- *A hosted embedding API.* Every chunk of every PRD would leave the network, and ingestion
  cost would scale with corpus size. Local CPU inference is slower per call but bounded and
  private.

**Consequence.** Switching the embedding model requires a **full re-index** of both
`qa_test_cases` and `qa_prd_chunks`. `EMBEDDING_FORMAT_VERSION` is stamped on documents so
a mismatch is logged at search time rather than silently returning nonsense.

### ADR-005 — Score thresholds are dual-mode

**Decision.** Confidence thresholds switch on whether the reranker is loaded: rerank_score
(cross-encoder logits: 2.0 high, 0.5 medium) when it is, RRF score (0.025 high, 0.012
medium) when it is not.

**Why.** The two scales are unrelated. A single threshold set would be silently wrong in one
of the two modes.

**Alternatives rejected.**
- *Normalising rerank logits into the RRF range.* Invents a mapping with no meaning and
  hides which stage actually produced the ranking.
- *Requiring the reranker.* The system must still answer with the reranker unavailable.

**Open.** These numbers are empirical and under-tuned. Tune with `eval/benchmark.py` against
real ground truth rather than adjusting them by feel.

---

## 3. Source conversion and chunking

### ADR-006 — Every ingestor normalises to markdown before chunking

**Decision.** Confluence, GitLab, Word, Excel, PDF and plain uploads all produce markdown,
which is the chunker's only input format.

**Why.** One chunker, one set of structural rules, one place to fix a structural bug. The
chunker's table handling keys off rows starting with `|`; a source that emits any other
table shape silently loses that handling.

**Alternatives rejected.**
- *A per-source chunker.* Five implementations of the same header-carry and fence logic,
  four of which would drift out of date.
- *Chunking raw HTML for Confluence.* Would need HTML-aware splitting rules duplicated
  alongside the markdown ones.

### ADR-007 — Confluence storage format is converted by lifting content out *before* html2text

**Decision.** `ingestion/confluence_html.py` extracts code-macro bodies, images, macro
titles, cross-page link titles and tables into opaque placeholder tokens *before* the
`ac:`/`ri:` tag strip and before html2text runs, then splices the rendered forms back in.

**Why.** Each of those is invisible to html2text on its own:

| Content | What happens without the lift |
|---|---|
| `<ac:structured-macro ac:name="code">` body | Wrapped in `<![CDATA[…]]>`, which `HTMLParser` reports as `unknown_decl`; html2text does not implement it, so **the body is dropped entirely** |
| `<ac:image><ri:attachment ri:filename="…"/>` | The `ac:` strip leaves a bare `ri:` tag that emits no text; alt text and filename are **both lost** |
| `<ac:parameter ac:name="title">` | Removed by the layout-junk strip, taking status/expand **titles** with it |
| `<table>` | html2text's own table output does not start rows with `|`, so the chunker's table handling **never engages** |

Placeholders are bare alphanumerics (`ZQXCODE0XQZ`) so no markdown escaper, link rewriter
or line wrapper touches them; `body_width = 0` prevents wrapping from splitting one.

**Alternatives rejected.**
- *Post-processing html2text output.* Cannot recover what was never emitted. Dropped CDATA
  is not recoverable downstream.
- *Configuring html2text harder.* No combination of its options preserves CDATA macro
  bodies or Confluence image references — they are not standard HTML.
- *A full XHTML parse into a document model.* Far more code, and html2text already handles
  ordinary prose, lists and inline formatting well. The lift is surgical: intervene only
  where html2text is blind.

### ADR-008 — Structure is atomic; joins depend on what is being joined

**Decision.** Table rows, list items and fenced code blocks are atomic segments. Two prose
segments join with a space; any join involving structure uses a newline.

**Why.** A space-join flattens a whole table onto one line and mashes bullets together. Only
prose genuinely reads as continuous text.

**Alternatives rejected.**
- *Always join with `"\n"`.* Simpler, but turns ordinary prose chunks into one-sentence-per-line
  text that reads oddly in the LLM context window.
- *Always join with `" "`.* The original behaviour, and the bug — see `mistakes.md`.

### ADR-009 — A split table repeats its header in every continuation chunk

**Decision.** When a table spans several chunks, each chunk after the first is prefixed with
the table's header row (and its `| --- |` delimiter). Each row segment records
`header_index` — `0` for the header row, `1` for the delimiter, `None` for data — so a
chunk that begins mid-header gets exactly the lines it is missing.

**Why.** A retrieved chunk is scored and read **alone**. Without the header it reads
`| Pro | 5000 | Yes |` with no column names — near-worthless to the embedder, the reranker
and the model.

**Alternatives rejected.**
- *Never splitting a table.* A large spec table would exceed the chunk budget and then the
  embedder's input cap, so its tail would be dropped from the vector anyway.
- *Storing the header only in metadata.* Neither BM25 nor the dense vector sees metadata.
  The header has to be in `chunk_text` to affect retrieval.
- *All-or-nothing header prepending.* Fails for a chunk starting at the delimiter row, which
  then gets a duplicated delimiter or no header at all.

### ADR-010 — Reassembly strips the headers that retrieval added

**Decision.** `markdown_table.join_chunk_texts()` is the inverse of ADR-009 and is used by
`read_prd_document`. It tracks the *active* table header across the whole run, and joins
continuation rows with `"\n"` rather than the paragraph separator.

**Why.** ADR-009 is right for retrieval and wrong for reading. Re-joined without stripping,
the model sees the header three times — one table becomes three.

Two specific requirements, both learned the hard way (see `mistakes.md`):
- State must be tracked across the whole run, not compared pairwise: once chunk N's header
  is stripped, chunk N+1 has nothing to match against.
- The active header must come from the *trailing* table of a chunk, not its leading lines —
  a table's first chunk is usually mixed (prose lead-in, then the table).

**Alternatives rejected.**
- *Not carrying headers at all, to avoid needing the inverse.* Trades a cosmetic reassembly
  problem for a real retrieval-quality loss. Wrong direction.
- *Keying the strip off `chunk_type`.* It works from text alone deliberately, so chunks
  indexed before that field existed are handled identically.

### ADR-011 — Only prose–prose boundaries are scored for topic similarity

**Decision.** Semantic boundary detection embeds and scores a boundary only when both
adjacent segments are prose. Non-prose boundaries get a similarity of 1.0 and split on the
size cap alone.

**Why.** Adjacent table rows are near-identical by construction, so scoring them produced
arbitrary split points *and* cost one embedding call per row. A 500-row table meant 500
wasted embeddings and meaningless boundaries.

**Alternatives rejected.**
- *Scoring every boundary.* The original behaviour: expensive and actively misleading.
- *Skipping semantic chunking whenever a table is present.* Throws away good boundary
  detection for the prose in the same section.

### ADR-012 — `chunk_type` is an additive keyword field

**Decision.** The chunker labels each chunk `table | code | mixed | prose`, stored as an ES
keyword and surfaced in search results and `GET /explain/prd/{source_id}`.

**Why.** It is the cheapest parse-sanity signal available: a PRD you know is table-heavy
that chunks as 100% prose means the source conversion flattened its tables — visible in the
ingest log before any tokens are spent on analysis.

**Alternatives rejected.**
- *Re-deriving the type at read time.* Possible, but then it cannot be filtered or
  aggregated in Elasticsearch, and the ingest log cannot report the mix.
- *A `dynamic: strict` mapping addition requiring a reindex.* Not needed —
  `_ensure_indexes` already patches mappings on existing indexes via `put_mapping`, and
  documents indexed before the field simply carry no value. Every consumer treats a missing
  value as "unknown, handle generically".

---

## 4. Ingestion scope and identity

### ADR-013 — Child pages are separate documents; attachments are folded into the parent

**Decision.** With `QA_CONFLUENCE_INGEST_CHILDREN=1`, each descendant is indexed as its
**own** document with its own `source_id` and `source_version`. With
`QA_CONFLUENCE_INGEST_ATTACHMENTS=1`, attachment content is appended as sections **of the
host page's** document.

**Why the asymmetry.** Incremental refresh keys off `source_version`. A child page has its
own Confluence version, so folding it into the parent would freeze the parent's version and
a child edit would never trigger a re-ingest — the page would go stale silently and
permanently. An attachment has no page-level version of its own, so it has nowhere else to
live. `prd_pipeline._upsert_by_source` already groups by `source_id`, so returning several
documents from one ingest call needs no pipeline change.

**Alternatives rejected.**
- *Folding children into the parent document.* The staleness trap above.
- *Indexing attachments as standalone documents.* They would lose the PRD context that makes
  them interpretable, and their identity would have no stable version to diff against.

**Known limitation.** Because attachments live in the page's document and refresh keys off
the *page* version, re-uploading an attachment without editing the page body may leave the
old attachment text indexed. Force a re-ingest of that page when attachments change in
place.

### ADR-014 — Behaviour-changing ingestion features default to off

**Decision.** Attachment ingestion, child-page ingestion and PDF OCR are all opt-in env
flags, defaulting off.

**Why.** Each changes request volume, index volume or image contents for every existing
deployment. Attachments cost a request per page plus a download per attachment and pull in
whatever else is attached (mockups, exports, screenshots). Child pages turn a targeted
single-page request into a crawl. OCR needs a system binary.

**Alternatives rejected.**
- *On by default because the content is valuable.* Silently multiplies API traffic and index
  size on upgrade.
- *Honouring a flag on only one code path.* Explicitly rejected — see `mistakes.md`. A flag
  that applies to single-page ingest but not the space crawl is worse than not having the
  feature, because which pages get the content depends on how they happened to be ingested.

### ADR-015 — The space crawl chunks fixed-window, not semantically

**Decision.** `fetch_and_chunk_page` calls `chunk_document` without `embed_fn`. Single-page
ingest passes one.

**Why.** A space crawl chunks pages in parallel worker threads and batch-embeds afterwards
(`prd_pipeline._ingest_one_space`). Threading `embed_fn` through would serialise an
embedding call per segment per page inside every worker and collapse crawl throughput.

**Alternatives rejected.**
- *Semantic chunking everywhere.* Correct in isolation, unusable at crawl scale.
- *Leaving it undocumented.* It was, which made a deliberate tradeoff look like a bug: the
  same page ingested two ways produces different chunks.

### ADR-016 — Excel loading mode is chosen by file size

**Decision.** `.xlsx` loads with `read_only=False` below `QA_INGEST_MAX_WORKBOOK_MB`
(default 25), and streams above it.

**Why.** Merged cells are normal in spec sheets (one "Module" cell spanning several
requirement rows) and openpyxl exposes merged ranges **only** in non-read-only mode. Left
unresolved they become blanks, and a row split into its own chunk loses the value entirely.
But non-read-only loads the whole workbook into memory, so a large file must not use it.

**Alternatives rejected.**
- *Always `read_only=True`.* Loses merged-cell fill-down on every spec sheet.
- *Always `read_only=False`.* A large workbook can exhaust the worker. The row cap does not
  help — it limits what is appended, not what is loaded.

---

## 5. Operational design

### ADR-017 — Long-running operations are async tasks returning a `run_id`

**Decision.** Sync, ingest and analysis return a `run_id` immediately; progress is polled
via `/…/status/{run_id}`. Tasks carry done-callbacks.

**Why.** These operations run for minutes. The done-callbacks exist specifically to prevent
silently lost exceptions — a task that dies without one leaves a run permanently "running".

**Alternatives rejected.**
- *Synchronous requests.* Guaranteed gateway timeouts on any real corpus.
- *Fire-and-forget without callbacks.* The failure mode is invisible, which is the worst
  property an audit-logged pipeline can have.

### ADR-018 — The embedder's character cap must exceed the chunker's maximum chunk

**Decision.** `QA_EMBED_MAX_CHARS` (default 8000) must stay above the chunker's
`SEMANTIC_MAX` (1200 tokens ≈ 4800 chars) plus the heading prefix and any carried table
header. Truncation logs a warning.

**Why.** Elasticsearch indexes the full `chunk_text` for BM25 regardless. If the cap sits
below the chunk size, the tail of a chunk is dropped from the **vector** while remaining
searchable by keyword — dense and sparse retrieval then see different documents, with no
error anywhere. This was a live bug (cap 4000 vs chunks up to ~4800).

**Alternatives rejected.**
- *Truncating silently for CPU speed.* The silence is the problem, not the truncation.
- *Letting the chunker emit whatever and relying on the model's 8192-token limit.* Leaves
  the two constants free to drift apart again.

### ADR-019 — Tests are stdlib-only, with real-library integration tests gated by availability

**Decision.** Unit tests stub third-party packages and run on a bare host. Real-library
tests live in `tests/test_ingestion_integration.py` and `skipUnless` the package is
importable.

**Why.** The suite must run without a container. But stubs verify our logic against our
*model* of a library, and that model is exactly where the worst bugs hid — `doc.paragraphs`
silently excluding table paragraphs and html2text discarding CDATA both looked perfectly
correct against a stub.

**Alternatives rejected.**
- *Stub-only testing.* Would not have caught either bug above.
- *Requiring the real libraries.* Makes the fast feedback loop depend on a container build.
- *Per-module stub definitions.* Rejected after repeated failures — stubs now live in one
  `tests/stubs.py` and install per-attribute. See `mistakes.md` for why.
