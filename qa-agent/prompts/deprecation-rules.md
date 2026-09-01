---
type: QA Guideline
title: Test Deprecation Rules
description: >
  Decision boundaries between DEPRECATE, UPDATE, KEEP and QUESTION. The analysis agent
  applies these when a PRD section conflicts with an existing test.
tags: [qa, deprecation, decision-rules]
status: active
verified: false
sources:
  - Team decisions captured 2026-09-01 — removal signals, conservative posture
  - Drafted from those decisions; confirm the wording lists and meta-sections match
    your PRD template before treating this as settled
---

# Test Deprecation Rules

DEPRECATE is the only irreversible recommendation this pipeline makes. A wrongly kept
test costs a few minutes of review. A wrongly deprecated test is silently gone, and
nobody notices until the untested behaviour breaks in production.

**The posture is deliberately conservative: when in doubt, QUESTION.** A QUESTION is
posted as a comment on the test in Jira, so a human sees it and answers. That is a cheap
outcome. A wrong DEPRECATE is not.

## The four removal signals, in descending order of confidence

Our PRDs signal removal in four different ways. They do not carry equal weight.

**1. A dedicated removal section** — a "Removed in this release", changelog or
equivalent section that names what went. This is the strongest signal. If the feature
the test covers is named there, DEPRECATE.

**2. Explicit wording in the section itself** — "removed", "deprecated", "retired",
"discontinued", "replaced entirely by …", "no longer supported". Quote the sentence in
your reason. DEPRECATE.

**3. A status label or macro on the section** — a status marker reading DEPRECATED,
RETIRED or equivalent. Treat as removal, but say in the reason which section carried the
label, because a stale label on a still-live feature is a real failure mode.

**4. The section is simply absent from the document** — this is the weakest signal and
is **not sufficient on its own**. Record a QUESTION, not a DEPRECATE.

Absence is weak for a reason that has nothing to do with the PRD: a test can be covered
by a *different* document. The corpus holds many PRDs and other document types, and this
agent analyses one at a time. "I could not find it in this PRD" is not "it is not
documented anywhere".

The engine now cross-checks this for you. When you record a DEPRECATE, it searches every
indexed document for content matching the test, and if another document appears to cover
it you are shown which one before the decision is accepted. Read what it shows you.

That check makes absence *checkable*, not *sufficient*. It searches what has been
ingested, so a document nobody has ingested is still invisible to it. Absence plus a
clean cross-check is reasonable grounds for a medium-confidence DEPRECATE; absence with
no other evidence at all remains a QUESTION.

## DEPRECATE

Only when **all** of these hold:

- One of signals 1–3 above applies. Absence alone (signal 4) never qualifies.
- The behaviour the test covers is the behaviour that was removed — not merely a
  neighbouring feature in the same section.
- No other section of the document you were given describes the same behaviour under a
  different heading. Search before concluding; features get moved and renamed.

Your reason must state **what was removed and where you saw it**. A reason that does not
contain removal language is rejected automatically, and rightly — "this test is no longer
needed" tells a reviewer nothing they can check.

## UPDATE — the default when a feature changed

The feature still exists but something about it moved. This is by far the most common
correct answer when a test and a PRD disagree.

- Flow, steps or acceptance criteria changed
- A field, label, button or endpoint was renamed
- A precondition was added or removed
- The test is still meaningful but the wording no longer matches

A renamed section is an UPDATE candidate, never a DEPRECATE. Sections get retitled far
more often than features get deleted.

## KEEP

The requirement and the test still agree. Say specifically which part of the section the
test covers — "matches the PRD" is not a reviewable reason.

Edge cases still valid under the new requirements are KEEP, even when the PRD does not
mention them explicitly. A PRD describes intended behaviour; it is not an exhaustive list
of what must be tested.

## QUESTION — use this freely

QUESTION is not a failure state. It is the correct answer whenever a human needs to
decide. Prefer it over a confident guess.

Record a QUESTION when:

- Only signal 4 applies — the feature is absent, with no removal statement anywhere
- The section says TBD, "to be confirmed", or is visibly unfinished
- The test spans several features and only some of them changed
- The behaviour looks like it belongs to another module's document
- A status label contradicts the section's prose
- You would be deprecating a large share of the tests you reviewed — say so explicitly
  and let a human confirm the scale before anything is removed

## Sections that never produce decisions

These are context for humans, not testable requirements. Do not record decisions against
them, and do not treat a test's absence from them as a coverage gap:

Background · Introduction · Rationale · Why this change · Success metrics · Success
criteria · Out of scope · Open questions · Glossary · Appendix · Timeline · Rollout ·
Methodology · Hypothesis · Budget · Current flow · Current process · Process flow ·
Mitigation plan · Scaling strategy · Pilot scope · Periodic review sections

## Confidence

Every decision takes a `confidence` of high, medium or low, and reviewers sort by it.

- **high** — a removal signal of type 1–3 quoted directly, or an obvious KEEP
- **medium** — the evidence is clear but inferred rather than stated
- **low** — you are recording this so a human looks at it

Be honest. A low-confidence DEPRECATE that is right still gets reviewed; a
high-confidence DEPRECATE that is wrong may not.
