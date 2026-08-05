---
type: QA Guideline
title: Test Case Writing Guidelines
description: >
  House rules for authoring Xray test cases. The analysis agent consults this before
  proposing CREATE or UPDATE decisions so generated tests match team conventions.
tags: [qa, test-authoring, xray]
status: template
verified: false
sources:
  - Bombardier default template (not yet customised for this team)
---

# Test Case Writing Guidelines

> **status: template** — these are generic defaults, not your team's rules. Until the
> section at the bottom is filled in, CREATE/UPDATE proposals will follow boilerplate
> rather than your conventions. This is the single highest-leverage prompt edit available.

## DO's
- Write clear, concise test case titles that describe the expected behavior
- Include preconditions for every test case
- Each step should have a single action (one verb per step)
- Include both positive and negative test scenarios
- Write expected results that are specific and measurable
- Use Given/When/Then format for acceptance criteria-based tests
- Include boundary value and edge case tests
- Add appropriate priority (Critical, High, Normal, Low)
- Tag tests with relevant labels (regression, smoke, integration, etc.)
- Include test data in the "data" field, not in the "action" field

## DON'Ts
- Don't write vague steps like "Verify it works" or "Check the result"
- Don't combine multiple validations into a single step
- Don't assume prior state without specifying it in preconditions
- Don't use hardcoded environment-specific values (URLs, IPs)
- Don't duplicate existing test cases — update them instead
- Don't skip error/exception scenarios
- Don't write tests that depend on other tests' execution order
- Don't include implementation details — tests should be behavior-focused
- Don't skip accessibility and security-related test cases where applicable

## Test Case Structure
- Summary: [Feature Area] - [Scenario] - [Expected Outcome]
- Precondition: List all required setup states
- Steps: Action → Data → Expected Result (per step)
- Priority: Based on business impact and frequency of use

## Naming Convention
- Format: TC_[Module]_[Feature]_[Scenario]_[PositiveOrNegative]
- Example: TC_Login_Email_ValidCredentials_Positive

## PUT YOUR ACTUAL GUIDELINES BELOW THIS LINE
## ============================================

