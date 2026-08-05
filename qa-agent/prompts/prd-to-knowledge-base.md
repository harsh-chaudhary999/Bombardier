---
type: QA Guideline
title: Converting PRDs to Knowledge Base Documentation
description: >
  How to read a PRD as a functional specification rather than a project document —
  which PRD sections carry testable behaviour and which are planning metadata to ignore.
  Useful when a PRD section is ambiguous about what the system actually does.
tags: [qa, prd-interpretation, requirements]
status: draft
verified: false
sources:
  - Bombardier default guidance
---

# Converting PRDs to Knowledge Base Documentation

## Purpose
When provided with Product Requirement Documents (PRDs) from Confluence or other sources, convert them into clear, functional knowledge base articles that explain HOW the system works, not just what the PRD says.

## Core Principle
DO NOT copy-paste PRD structure.
DO create clear functional specifications.

### Bad Approach (Avoid)
- Copying PRD sections like "Problem Statement", "Objective", "Success Metrics"
- Including project metadata (document owner, designer, QA, target dates)
- Documenting A/B test details, experiment setup, or rollout phases
- Listing "Future Scope" items that aren't implemented
- Preserving PRD timeline/phase structure in main docs

### Good Approach (Follow)
- Document the actual functionality as it exists
- Explain behavior, logic, and conditions
- Show what displays when and what happens if
- Include clear decision trees and flow logic
- Provide examples of calculations and processes
- Use tables for conditional logic
- Write in present tense (how it works now)

## Document Structure Template

```markdown
# [Feature Name]

Brief description of what this feature does and its purpose.

**Related**: [[Feature1]] | [[Feature2]] | [[Feature3]]

---

## [Component/Section Name]

### What It Shows

[Clear description with bullet points or table]

| Element | Condition | Display Logic |
|---------|-----------|---------------|
| Item 1 | When X | Shows Y |
| Item 2 | When Z | Shows A |

### How It Works

IF condition:
  Action A
ELSE IF other_condition:
  Action B
ELSE:
  Action C

### Example

[Concrete example showing the logic in action]

---

## [Next Component]

[Continue pattern above]

---

## Integration Notes

### Updates [[RelatedFeature1]]
- What gets updated
- When it updates
- What triggers it

### Consumes/Uses [[RelatedFeature2]]
- What it uses
- How it uses it

---

## Events Tracked

EVENT_NAME - When it fires
OTHER_EVENT - Conditions

---

## References

- Confluence: Link
- Design: Figma
```

## Specific Guidelines

### 1. Display Logic

Always answer:
- **What** is shown?
- **When** is it shown?
- **Where** on the page?
- **What conditions** control visibility?

Use format:
```
IF [condition]:
  Show: [element]
  Position: [location]
  Style: [appearance]
ELSE:
  Show: [alternative]
```

### 2. User Actions

For every action, document:
- **Trigger**: What user does
- **Process**: What happens step-by-step
- **Result**: Final state
- **Side effects**: What else changes

Use format:
```
User clicks [Button]
→ [Step 1]
→ [Step 2]
→ [Final result]
```

### 3. Calculations/Algorithms

- Provide the **formula**
- Show **example calculation** with real numbers
- Explain **edge cases**
- Document **tie-breaking** rules

Use tables for scoring:

| Input Range | Output | Logic |
|-------------|--------|-------|
| 0-10 | Score A | Because... |
| 11-20 | Score B | Because... |

### 4. Conditional Logic

Use clear decision trees:
```
Check condition_1:
  IF true:
    Check condition_2:
      IF true: Outcome A
      ELSE: Outcome B
  ELSE:
    Outcome C
```

### 5. State Management

Document what changes when:
```
Initial state: [description]
After action X:
  Field A = new_value
  Field B = updated
  UI refreshes showing [change]
After action Y:
  Field A = different_value
  Navigation to [page]
```

### 6. Platform Differences

Always note when behavior differs:
```
Desktop:
  - Behavior A
  - Shows X

Mobile/App:
  - Behavior B
  - Shows Y
```

### 7. Permissions/Access Control

Document WHO can do WHAT:

| User Type | Can Do | Cannot Do | Special Rules |
|-----------|--------|-----------|---------------|
| Admin | X, Y, Z | A | Credits from own account |
| Member | Y, Z | A, X | Limited visibility |

## What to Include

### Must Include
- Feature functionality and behavior
- UI components and their conditions
- User workflows (click -> action -> result)
- Calculations with examples
- Conditional logic (IF/THEN/ELSE)
- Integration points with other features
- Events tracked
- Permission rules
- Edge cases and special conditions
- Platform differences (web vs mobile)

### Include Only If Relevant
- Design references (Figma links)
- Confluence links (in References section at bottom)
- Database schema (if complex logic depends on it)

### Exclude
- PRD metadata (owners, designers, dates)
- Problem statements and objectives
- Success metrics and KPIs
- A/B test details
- Rollout plans
- Historical context ("in Phase 1 we did...")
- Future scope / roadmap items
- Supporting data tables from PRDs

## File Naming Convention

`[Feature Area]/[Feature Name].md`

Examples:
- `Billing/Web/Pages/Invoice Review.md`
- `Platform/App/Pages/Checkout Flow.md`
- `Admin/Features/Approval Workflow.md`

Avoid:
- Phase numbers in filenames
- Version numbers
- Date stamps

## Linking Strategy

Link to Related Features:
```
**Related**: [[Feature1]] | [[Feature2]]
```

Link in Other Features — update related features to link back:
```
### Updates [[ThisFeature]]
- What changes
- When it changes
```

## Testing Your Documentation

Ask yourself:
1. Can a developer implement this from my description?
2. Can a QA person write test cases from this?
3. Can a new team member understand the feature flow?
4. Are all IF/THEN conditions clear?
5. Are there examples for complex logic?

If no to any -> Revise.

## Common Patterns

### Pattern 1: UI Component Display
```markdown
### [Component Name]

**Location**: [Where on page]

**Always Shown**:
- Element 1
- Element 2

**Conditionally Shown**:
| Element | Condition | Display |
|---------|-----------|---------|
| X | IF A | Show Y |
| Z | IF B | Show C |
```

### Pattern 2: User Action Flow
```markdown
### [Action Name]

User clicks [Element]
→ System checks [Condition]
→ IF [Condition]:
    Perform [Action A]
    Update [State X]
    Show [Feedback Y]
  ELSE:
    Perform [Action B]
    Show [Error Z]

**Side Effects**:
- Updates [[Feature1]]
- Triggers [[Feature2]]
- Logs event: `EVENT_NAME`
```

### Pattern 3: Calculation/Algorithm
```markdown
### [Calculation Name]

**Formula**: `result = (factor_a * weight_a) + (factor_b * weight_b)`

**Factors**:
| Factor | Range | Score Logic |
|--------|-------|-------------|
| A | 0-10 | score = value * 10 |
| B | 0-100 | score = value |

**Example**:
Input: factor_a = 5, factor_b = 80
Result = (5 * 2) + (80 * 1) = 90

**Edge Cases**:
- IF factor_a missing: Use default value 0
- IF result > 100: Cap at 100
```

### Pattern 4: State Transitions
```markdown
### State Machine

**States**: [State A] -> [State B] -> [State C]

**Transitions**:
State A:
  On action_1 -> State B
  On action_2 -> State C
State B:
  On action_3 -> State C
  On timeout -> State A

**Displayed To User**:
- State A: Shows [X]
- State B: Shows [Y]
- State C: Shows [Z]
```

## Final Reminder

Your goal: Create documentation that allows someone to understand and implement/test/use the feature WITHOUT reading the original PRD.

- Think: Implementation guide, not project history.
- Write: Functional spec, not requirements doc.
- Focus: What exists and how it works, not what was planned or why.
