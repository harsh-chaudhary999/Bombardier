---
type: QA Guideline
title: Test Case Writing Guidelines
description: >
  House rules for authoring Xray test cases. The analysis agent consults this before
  proposing CREATE or UPDATE decisions so generated tests match team conventions.
tags: [qa, test-authoring, xray]
status: active
verified: false
sources:
  - Team decisions captured 2026-09-01 — naming scheme, priority left to reviewers
  - Drafted from those decisions; confirm the module list and label vocabulary match
    your Xray project before treating this as settled
---

# Test Case Writing Guidelines

Applies to every CREATE, and to any UPDATE that rewrites a summary or steps.

## Naming

    TC_<Module>_<Feature>_<SubFeature>_<Positive|Negative>

Examples:

    TC_Platform_Checkout_OneStep_Positive
    TC_Billing_Invoice_DownloadPDF_Negative
    TC_Auth_Login_ExpiredToken_Negative

Rules:

- `Module` must be a module that already exists in the index. If you are unsure, search
  first — inventing a module name puts the test somewhere nobody filters on.
- CamelCase within each part. No spaces. Underscores only between parts.
- `Positive` = the expected-success path. `Negative` = an error, rejection, or edge case.
- The name should read as what is being verified, not as a ticket reference.

## Do not set priority

Leave priority unset on anything you propose. A reviewer assigns it, because priority
depends on release context and customer exposure that the document does not carry.

Do not work around this by putting a priority in the summary or the reason.

## Required content for a CREATE

- **Summary** — one line, following the naming scheme above.
- **Preconditions** — at least one sentence of starting state: who is signed in, what
  exists, what state it is in.
- **Steps** — at least two, each one action paired with one observable result.
- **Labels** — at least one of: regression, smoke, integration, e2e, api, ui.

## Step format

Each step is one action and one observable expected result.

    Action:          Select "Submit order"
    Expected result: The confirmation page loads and shows an order reference

Never write:

- "Verify it works" — not observable
- "Check the result" — not specific
- Two actions in one step ("Select Submit and then open the invoice")
- An expected result that only a developer could check ("the row is written to the
  database"). Test the behaviour, not the implementation.

## Rewriting steps on an UPDATE — read this before using `updated_steps`

**Xray replaces the entire step list with whatever you supply.** A partial list
permanently deletes every step you left out.

So:

- Use `updated_steps` **only** when you are rewriting the complete step list and have
  seen all the existing steps.
- Otherwise describe the change in `suggested_changes`. That is recorded as a comment for
  a human to apply, and nothing is overwritten.

When in doubt, use `suggested_changes`. A comment costs a reviewer a minute. A truncated
step list costs the test.

## Before proposing a CREATE

Search for the behaviour first. Near-identical tests are checked automatically and a very
close match is refused, but the check only sees what you actually searched for. If a test
covers the same behaviour with different wording, record an UPDATE against it instead.

Genuinely new behaviour deserves a new test. Do not stretch an existing test to cover
something it was not written for.

## What must be covered

For a feature you are proposing tests for, cover:

- The happy path, end to end
- Invalid input — required fields empty, wrong format, out of range
- Boundary values wherever the document states a limit
- Permission behaviour — what a lower-privileged user cannot do
- Error messages, with the exact text when the document specifies it

## Placeholders, not real values

Write `{test_user_email}`, `{order_id}`, `{api_base_url}` rather than real addresses,
keys or hostnames. Tests are read by people outside the team and copied between
environments.

## Independence

Every test sets up its own preconditions and cleans up after itself. No test may depend
on another having run first, or on data a previous test left behind.
