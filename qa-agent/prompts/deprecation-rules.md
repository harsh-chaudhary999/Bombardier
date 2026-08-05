---
type: QA Guideline
title: Test Deprecation Rules
description: >
  Decision boundaries between DEPRECATE, UPDATE and KEEP. The analysis agent applies
  these when a PRD section conflicts with an existing test.
tags: [qa, deprecation, decision-rules]
status: template
verified: false
sources:
  - Bombardier default template (not yet customised for this team)
---

# Test Deprecation Rules

> **status: template** — generic defaults. DEPRECATE is the only irreversible action the
> pipeline can recommend, so tightening these rules for your codebase matters more than
> any other prompt change.

## When to mark a test as DEPRECATED
- The feature it tests has been completely removed from the PRD
- The feature has been replaced by a fundamentally different implementation
- The test validates behavior that is explicitly changed in the new requirements

## When to mark a test as OUTDATED (needs update, not removal)
- The feature still exists but steps/flow have changed
- Acceptance criteria have been modified
- UI elements or API endpoints referenced in steps have changed
- New validations have been added to an existing feature

## When a test is UNCHANGED
- The feature and its acceptance criteria are identical in the new PRD
- No changes to the flow, validations, or expected behavior

## PUT YOUR ACTUAL RULES BELOW THIS LINE
## ======================================

