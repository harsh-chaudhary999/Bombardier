"""
Guards the write-back step payload — the one path in this codebase that can destroy data.

Xray's `updateTestSteps` mutation REPLACES a test's entire step array; it does not merge.
Write-back previously took the agent's prose recommendation and wrapped it as
`[{"action": "other", "data": "<English paragraph>"}]`, so approving one UPDATE decision
turned a multi-step manual test in Xray into a single step containing a sentence. There is
no undo, and nothing in the pipeline reported it as a loss.

Two rules are pinned here:

  * prose never becomes a step — it goes to a Jira comment, leaving structure untouched
  * a steps payload that cannot be validated is refused outright rather than partially sent

Plus the create-side rule, where there is nothing to destroy but a paragraph pretending to
be a step still yields a useless test: enumerated outlines parse into steps, prose becomes
the description.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from test_context_budget import _install_stubs  # noqa: E402

_install_stubs()
from agents.writeback import _steps_from_outline, _validated_steps  # noqa: E402


class ValidatedStepsTests(unittest.TestCase):
    def test_wellformed_list_passes_and_is_indexed(self):
        got = _validated_steps([
            {"action": "Open the login page", "data": "", "expectedResult": "Form renders"},
            {"action": "Submit valid credentials", "data": "user/pass", "expectedResult": "Redirected"},
        ])
        self.assertEqual(len(got), 2)
        self.assertEqual([s["index"] for s in got], [1, 2])
        self.assertEqual(got[1]["data"], "user/pass")
        self.assertEqual(got[0]["expectedResult"], "Form renders")

    def test_result_key_is_accepted_as_expected_result(self):
        """Xray's own read model calls it `result`; accept either spelling."""
        got = _validated_steps([{"action": "do x", "result": "y"}])
        self.assertEqual(got[0]["expectedResult"], "y")

    def test_missing_optional_fields_default_to_empty(self):
        got = _validated_steps([{"action": "do x"}])
        self.assertEqual(got[0], {"action": "do x", "data": "", "expectedResult": "", "index": 1})

    # ── the destructive cases: all must be refused, not partially written ──

    def test_prose_string_is_refused(self):
        """The original bug. A paragraph is not a step list."""
        self.assertIsNone(_validated_steps(
            "Add a validation step after step 3 and update the expected result."
        ))

    def test_empty_and_none_are_refused(self):
        for bad in (None, [], "", {}, 0):
            with self.subTest(bad=bad):
                self.assertIsNone(_validated_steps(bad))

    def test_one_bad_item_rejects_the_whole_list(self):
        """Writing the good prefix would silently delete every step after it."""
        self.assertIsNone(_validated_steps([
            {"action": "step one is fine"},
            {"data": "no action key"},
            {"action": "step three is fine"},
        ]))

    def test_blank_action_is_refused(self):
        for bad in ("", "   ", None, 42):
            with self.subTest(bad=bad):
                self.assertIsNone(_validated_steps([{"action": bad}]))

    def test_list_of_strings_is_refused(self):
        self.assertIsNone(_validated_steps(["open page", "submit form"]))


class StepsFromOutlineTests(unittest.TestCase):
    def test_numbered_outline_becomes_separate_steps(self):
        got = _steps_from_outline("1. Open the page\n2. Enter credentials\n3. Submit")
        self.assertEqual(len(got), 3)
        self.assertEqual(got[0]["action"], "Open the page")
        self.assertEqual([s["index"] for s in got], [1, 2, 3])

    def test_bullet_styles(self):
        for bullet in ("-", "*", "•"):
            with self.subTest(bullet=bullet):
                got = _steps_from_outline(f"{bullet} first\n{bullet} second")
                self.assertEqual(len(got), 2)

    def test_step_n_prefix(self):
        got = _steps_from_outline("Step 1: open\nStep 2: close")
        self.assertEqual([s["action"] for s in got], ["open", "close"])

    def test_paren_and_colon_numbering(self):
        self.assertEqual(len(_steps_from_outline("1) a\n2) b")), 2)
        self.assertEqual(len(_steps_from_outline("1: a\n2: b")), 2)

    def test_expected_result_is_split_off(self):
        for sep in ("->", "=>", "→", "| Expected:", "Expected:"):
            with self.subTest(sep=sep):
                got = _steps_from_outline(f"1. click save {sep} record is stored")
                self.assertEqual(got[0]["action"], "click save")
                self.assertEqual(got[0]["expectedResult"], "record is stored")

    def test_prose_returns_none_so_the_caller_uses_the_description(self):
        """A paragraph must not become a step — that was the create-side symptom."""
        self.assertIsNone(_steps_from_outline(
            "Verify that a provisional account can access leads and payments for the full "
            "privilege window without completing document verification."
        ))

    def test_unenumerated_lines_return_none(self):
        self.assertIsNone(_steps_from_outline("open the page\nenter credentials"))

    def test_empty_and_non_string(self):
        for bad in (None, "", "   ", 5, ["1. a"]):
            with self.subTest(bad=bad):
                self.assertIsNone(_steps_from_outline(bad))

    def test_mixed_content_keeps_only_enumerated_lines(self):
        got = _steps_from_outline(
            "Here is what to test:\n1. open the page\nsome commentary\n2. submit\n"
        )
        self.assertEqual([s["action"] for s in got], ["open the page", "submit"])

    def test_blank_enumerated_line_is_skipped(self):
        got = _steps_from_outline("1. \n2. real step")
        self.assertEqual([s["action"] for s in got], ["real step"])


class _FakePG:
    """Enough PGStore surface for run_writeback; records what got marked done."""

    def __init__(self, decisions):
        self._decisions = decisions
        self.written_back = []
        self.merged = []

    def iter_writeback_decisions(self, run_id, batch_size=200):
        yield list(self._decisions)

    def mark_written_back(self, decision_id):
        self.written_back.append(decision_id)
        return True  # real PGStore returns "row existed"; returning None masks the create guard

    def merge_decision_updated_content(self, decision_id, patch):
        self.merged.append((decision_id, patch))
        return True


class _FakeXray:
    """Records every Xray/Jira call so the test can assert what was and wasn't touched."""

    def __init__(self):
        self.updates = []
        self.comments = []
        self.created = []
        self.deprecated = []
        self.links = []

    async def update_test(self, jira_key, summary=None, steps=None):
        self.updates.append({"jira_key": jira_key, "summary": summary, "steps": steps})

    async def add_comment(self, issue_key, comment):
        self.comments.append({"issue_key": issue_key, "comment": comment})

    async def deprecate_test(self, jira_key, reason):
        self.deprecated.append(jira_key)

    async def bulk_create_tests(self, project_key, tests):
        self.created.extend(tests)
        return {"keys": [f"PROJ-{900 + i}" for i in range(len(tests))]}

    async def add_remote_link(self, issue_key, url, title="PRD Source"):
        self.links.append(issue_key)


def _decision(**kw):
    base = {"id": 1, "action": "update", "jira_key": "PROJ-1234", "reason": "PRD changed",
            "updated_content": {}, "prd_source": ""}
    base.update(kw)
    return base


class UpdatePathTests(unittest.IsolatedAsyncioTestCase):
    """The behaviour that matters: what reaches Xray for an UPDATE decision."""

    async def _run(self, decisions, project_key="PROJ"):
        from agents import writeback as W
        pg, xr = _FakePG(decisions), _FakeXray()
        orig = W.xray_client
        W.xray_client = xr
        try:
            result = await W.run_writeback(pg, run_id="r1", project_key=project_key)
        finally:
            W.xray_client = orig
        return result, pg, xr

    async def test_prose_goes_to_a_comment_and_never_to_steps(self):
        """The original data-loss bug, pinned."""
        prose = "Add a validation step after step 3 and update the expected result."
        _, pg, xr = await self._run([
            _decision(updated_content={"suggested_changes": prose})
        ])
        self.assertEqual(xr.updates, [], "must not call update_test — that would replace steps")
        self.assertEqual(len(xr.comments), 1)
        self.assertIn(prose, xr.comments[0]["comment"])
        self.assertEqual(xr.comments[0]["issue_key"], "PROJ-1234")
        self.assertEqual(pg.written_back, [1], "the decision is still handled, not dropped")

    async def test_structured_steps_are_written(self):
        steps = [{"action": "open page", "data": "", "expectedResult": "renders"},
                 {"action": "submit", "data": "", "expectedResult": "saved"}]
        _, pg, xr = await self._run([_decision(updated_content={"steps": steps})])
        self.assertEqual(len(xr.updates), 1)
        self.assertEqual(len(xr.updates[0]["steps"]), 2)
        self.assertEqual(xr.updates[0]["steps"][0]["action"], "open page")
        self.assertEqual(pg.written_back, [1])

    async def test_invalid_steps_payload_is_refused_entirely(self):
        """Sending the valid prefix would delete every step after the bad one."""
        result, pg, xr = await self._run([
            _decision(updated_content={"steps": [{"action": "ok"}, {"no": "action"}]})
        ])
        self.assertEqual(xr.updates, [], "nothing may reach Xray")
        self.assertEqual(pg.written_back, [], "must not be marked done")
        self.assertEqual(len(result["errors"]), 1)
        self.assertIn("refusing to replace", result["errors"][0]["error"])

    async def test_prose_steps_payload_is_refused(self):
        """A string in the steps slot is the shape that caused the original destruction."""
        _, pg, xr = await self._run([
            _decision(updated_content={"steps": "just rewrite the whole thing"})
        ])
        self.assertEqual(xr.updates, [])
        self.assertEqual(pg.written_back, [])

    async def test_summary_only_update_does_not_send_steps(self):
        _, _, xr = await self._run([_decision(updated_content={"summary": "New title"})])
        self.assertEqual(len(xr.updates), 1)
        self.assertEqual(xr.updates[0]["summary"], "New title")
        self.assertIsNone(xr.updates[0]["steps"], "steps=None leaves them untouched")

    async def test_steps_and_prose_together_do_both(self):
        _, _, xr = await self._run([_decision(updated_content={
            "steps": [{"action": "open page"}],
            "suggested_changes": "also consider the mobile flow",
        })])
        self.assertEqual(len(xr.updates), 1)
        self.assertEqual(len(xr.comments), 1)

    async def test_empty_payload_is_an_error_not_a_silent_pass(self):
        result, pg, xr = await self._run([_decision(updated_content={})])
        self.assertEqual(xr.updates, [])
        self.assertEqual(pg.written_back, [])
        self.assertIn("missing", result["errors"][0]["error"])

    async def test_whitespace_only_prose_is_an_error(self):
        result, pg, _ = await self._run([
            _decision(updated_content={"suggested_changes": "   \n  "})
        ])
        self.assertEqual(pg.written_back, [])
        self.assertEqual(len(result["errors"]), 1)

    async def test_missing_jira_key_is_refused(self):
        result, pg, xr = await self._run([
            _decision(jira_key=None, updated_content={"summary": "x"})
        ])
        self.assertEqual(xr.updates, [])
        self.assertIn("missing jira_key", result["errors"][0]["error"])


class CreatePathTests(unittest.IsolatedAsyncioTestCase):
    async def _run(self, decisions):
        from agents import writeback as W
        pg, xr = _FakePG(decisions), _FakeXray()
        orig = W.xray_client
        W.xray_client = xr
        try:
            result = await W.run_writeback(pg, run_id="r1", project_key="PROJ")
        finally:
            W.xray_client = orig
        return result, pg, xr

    async def test_enumerated_outline_becomes_multiple_steps(self):
        _, _, xr = await self._run([_decision(action="create", jira_key=None, updated_content={
            "summary": "Verify provisional access",
            "suggested_steps": "1. create account\n2. request a lead\n3. confirm access",
        })])
        self.assertEqual(len(xr.created), 1)
        self.assertEqual(len(xr.created[0]["steps"]), 3)

    async def test_prose_outline_becomes_the_description_not_a_bogus_step(self):
        prose = "Check that access is granted for the whole privilege window."
        _, _, xr = await self._run([_decision(action="create", jira_key=None, updated_content={
            "summary": "Verify provisional access",
            "suggested_steps": prose,
        })])
        self.assertEqual(len(xr.created), 1)
        self.assertNotIn("steps", xr.created[0], "a paragraph is not a step")
        self.assertEqual(xr.created[0]["description"], prose)

    async def test_structured_steps_win_over_the_outline(self):
        _, _, xr = await self._run([_decision(action="create", jira_key=None, updated_content={
            "summary": "s",
            "steps": [{"action": "structured step"}],
            "suggested_steps": "1. outline step",
        })])
        self.assertEqual(xr.created[0]["steps"][0]["action"], "structured step")


if __name__ == "__main__":
    unittest.main()
