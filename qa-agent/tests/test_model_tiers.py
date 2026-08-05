"""
Guards model tiering and reasoning-token stripping.

Both exist because of the same shift: mixing a small local model with a strong one.

Tiering — a half-configured reasoning tier is the dangerous state. The operator believes
hard reasoning is escalated; it isn't; the symptom is degraded analysis quality, which is
nearly impossible to trace back to one unset env var. So it warns rather than silently
falling back.

Reasoning tokens — Qwen3 and similar local models inline chain-of-thought as
<think>...</think> in the response body. Left in, it corrupts everything downstream:
answers open with paragraphs of deliberation, bracket numbers inside the monologue get
parsed as citations, and the no-tool-call nudge sees a tool name mid-thought and
misdiagnoses the turn.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from test_context_budget import _install_stubs  # noqa: E402

_install_stubs()
from agents import model_tiers as T  # noqa: E402
from agents.analysis_agent import message_text  # noqa: E402


class TierTests(unittest.TestCase):
    KEYS = ("QA_REASONING_PROVIDER", "QA_REASONING_MODEL")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self.KEYS}
        for k in self.KEYS:
            os.environ.pop(k, None)
        T.reset_warnings()

    def tearDown(self):
        for k, v in self._saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v

    def test_fast_tier_uses_the_request_choice(self):
        self.assertEqual(T.resolve_tier("fast", "ollama", "gemma4:12b"), ("ollama", "gemma4:12b"))

    def test_reasoning_falls_back_when_unconfigured(self):
        """Nothing changes until a second model is deliberately configured."""
        self.assertEqual(
            T.resolve_tier("reasoning", "ollama", "gemma4:12b"), ("ollama", "gemma4:12b")
        )
        self.assertFalse(T.reasoning_tier_configured())

    def test_reasoning_escalates_when_both_are_set(self):
        os.environ["QA_REASONING_PROVIDER"] = "anthropic"
        os.environ["QA_REASONING_MODEL"] = "claude-opus-5"
        self.assertEqual(
            T.resolve_tier("reasoning", "ollama", "gemma4:12b"), ("anthropic", "claude-opus-5")
        )
        self.assertTrue(T.reasoning_tier_configured())

    def test_reasoning_can_stay_local(self):
        os.environ["QA_REASONING_PROVIDER"] = "ollama"
        os.environ["QA_REASONING_MODEL"] = "qwen3:30b-a3b"
        self.assertEqual(
            T.resolve_tier("reasoning", "ollama", "gemma4:12b"), ("ollama", "qwen3:30b-a3b")
        )

    def test_half_configured_tier_does_not_escalate(self):
        """Provider without model must not produce a nonsense (anthropic, gemma4) pair."""
        os.environ["QA_REASONING_PROVIDER"] = "anthropic"
        self.assertEqual(
            T.resolve_tier("reasoning", "ollama", "gemma4:12b"), ("ollama", "gemma4:12b")
        )
        self.assertFalse(T.reasoning_tier_configured())

    def test_model_without_provider_also_does_not_escalate(self):
        os.environ["QA_REASONING_MODEL"] = "claude-opus-5"
        self.assertEqual(
            T.resolve_tier("reasoning", "ollama", "gemma4:12b"), ("ollama", "gemma4:12b")
        )

    def test_unknown_tier_degrades_to_fast(self):
        """A typo in an internal call site must not fail an otherwise-good run."""
        self.assertEqual(T.resolve_tier("nonsense", "ollama", "m"), ("ollama", "m"))

    def test_describe_reports_the_effective_assignment(self):
        os.environ["QA_REASONING_PROVIDER"] = "anthropic"
        os.environ["QA_REASONING_MODEL"] = "claude-opus-5"
        d = T.describe("ollama", "gemma4:12b")
        self.assertEqual(d["fast"], "ollama/gemma4:12b")
        self.assertEqual(d["reasoning"], "anthropic/claude-opus-5")
        self.assertTrue(d["tiered"])

    def test_describe_reports_not_tiered_when_same(self):
        self.assertFalse(T.describe("ollama", "gemma4:12b")["tiered"])


class _Msg:
    def __init__(self, content):
        self.content = content


class ReasoningStripTests(unittest.TestCase):
    def test_plain_text_passes_through(self):
        self.assertEqual(message_text(_Msg("The answer is 42 [1].")), "The answer is 42 [1].")

    def test_think_block_is_removed(self):
        raw = "<think>Let me consider the passages. [3] looks relevant.</think>\nPROVISIONAL grants access [1]."
        self.assertEqual(message_text(_Msg(raw)), "PROVISIONAL grants access [1].")

    def test_citations_inside_reasoning_do_not_leak(self):
        """The concrete harm: [3] cited only in the monologue must not count as a citation."""
        raw = "<think>Maybe [3] or [4]?</think>Answer [1]."
        out = message_text(_Msg(raw))
        self.assertNotIn("[3]", out)
        self.assertNotIn("[4]", out)
        self.assertIn("[1]", out)

    def test_alternate_tag_names_and_spacing(self):
        for tag in ("think", "thinking", "thought", "reasoning"):
            with self.subTest(tag=tag):
                self.assertEqual(message_text(_Msg(f"< {tag} >hidden</ {tag} >Visible.")), "Visible.")

    def test_tag_matching_is_case_insensitive(self):
        self.assertEqual(message_text(_Msg("<THINK>hidden</THINK>Visible.")), "Visible.")

    def test_unterminated_block_truncated_by_max_tokens_is_dropped(self):
        """Hitting max_tokens mid-thought must not surface a truncated monologue as the answer."""
        self.assertEqual(message_text(_Msg("Answer [1].\n<think>Now let me reconsider whe")),
                         "Answer [1].")

    def test_multiple_blocks_all_removed(self):
        raw = "<think>a</think>First [1]. <think>b</think>Second [2]."
        # The blocks vanish; the space that separated them in the source remains.
        self.assertEqual(message_text(_Msg(raw)), "First [1]. Second [2].")

    def test_anthropic_style_block_list_skips_thinking(self):
        """Providers that separate reasoning give a content list; only text blocks are the answer."""
        content = [
            {"type": "thinking", "thinking": "deliberating about [3]"},
            {"type": "text", "text": "PROVISIONAL grants access [1]."},
        ]
        self.assertEqual(message_text(_Msg(content)), "PROVISIONAL grants access [1].")

    def test_redacted_thinking_block_skipped(self):
        content = [{"type": "redacted_thinking", "data": "xx"}, {"type": "text", "text": "Answer."}]
        self.assertEqual(message_text(_Msg(content)), "Answer.")

    def test_empty_and_none_content(self):
        self.assertEqual(message_text(_Msg("")), "")
        self.assertEqual(message_text(_Msg([])), "")

    def test_only_reasoning_yields_empty_not_the_monologue(self):
        """An answer that is nothing but reasoning is empty — and /ask flags it ungrounded."""
        self.assertEqual(message_text(_Msg("<think>all of it was thinking</think>")), "")


if __name__ == "__main__":
    unittest.main()
