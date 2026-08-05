---
type: Knowledge Index
title: QA Analysis Knowledge Bundle
description: >
  Curated QA knowledge the analysis agent can read on demand. Entry point for the
  bundle — an agent reads this first, then fetches only the documents it needs.
tags: [qa, index]
status: current
verified: false
---

# QA Analysis Knowledge Bundle

Loosely follows [Open Knowledge Format](https://github.com/GoogleCloudPlatform/knowledge-catalog/tree/main/okf)
(markdown + YAML frontmatter, navigable via this index). OKF is v0.1 and still moving, so
treat the frontmatter keys here as a local convention that happens to be OKF-shaped —
not a compliance claim.

Read a document with the `read_knowledge` tool, passing the path from the table below.

| Path | What it covers | Read it when | Status |
|------|----------------|--------------|--------|
| `test-case-guidelines.md` | House rules for authoring Xray test cases — structure, naming, DO/DON'T | Proposing a CREATE, or rewriting steps for an UPDATE | ⚠️ template |
| `deprecation-rules.md` | Boundaries between DEPRECATE / UPDATE / KEEP | A PRD section conflicts with an existing test and you must choose an action | ⚠️ template |
| `prd-to-knowledge-base.md` | Reading a PRD as a functional spec; which sections are planning metadata vs testable behaviour | A PRD section is ambiguous about actual system behaviour | draft |

## Frontmatter keys used here

| Key | Meaning |
|-----|---------|
| `type`, `title`, `description`, `tags` | Identity and retrieval metadata |
| `status` | `current` \| `draft` \| `template`. **`template` means generic defaults, not this team's rules** |
| `verified` | Has a human reviewed and signed off on this content |
| `sources` | Provenance — where the content came from, with credibility signals |
| `stale_after` | Freshness deadline; past it, treat the content as suspect |

## Why these are read on demand rather than inlined

Providers with prompt caching (Anthropic) get the full text inlined in the cached system
prefix — cheap to resend, so there is nothing to gain from deferring it. Local providers
(Ollama) have no cache discount and a hard `num_ctx` wall, so the agent reads on demand
instead; see `_system_prompt()` in `agents/analysis_agent.py`.

Measured effect: the lean prompt is ~2,800 chars vs ~5,200 inlined — roughly **600 tokens
freed per request**, or ~12K cumulative across a 20-turn run. Modest on its own. The larger
gain is that `prd-to-knowledge-base.md` becomes *reachable at all*: 7KB of guidance that was
never loaded by any code path before, too large to inline, now available when the agent hits
an ambiguous PRD section.

## Known gap

Two of three documents are unmodified templates. The agent is currently reasoning about
your test suite using generic boilerplate — the `status: template` marker exists so that
is visible in review rather than silently assumed to be team policy.
