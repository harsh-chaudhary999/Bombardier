"""
Guards token/cost accounting, prompt caching, and the knowledge-bundle prompt modes.

These exist because every cost claim about this pipeline was previously unverifiable:
there was no usage accounting at all, so "incremental analysis is 60-90% cheaper" could
not be checked. The assertions below pin the accounting arithmetic (including the
Anthropic cache-read discount) and the cache_control placement that makes it pay off.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from test_context_budget import _install_stubs  # noqa: E402  (shared stub installer)

_install_stubs()
from agents import analysis_agent as A  # noqa: E402


class _Resp:
    """Minimal stand-in for a LangChain AIMessage carrying usage_metadata."""

    def __init__(self, meta):
        self.usage_metadata = meta


class EnvIsolatedTest(unittest.TestCase):
    _KEYS = ("QA_AGENT_KNOWLEDGE_MODE", "QA_PROMPT_CACHING", "QA_MODEL_PRICES")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self._KEYS}
        for k in self._KEYS:
            os.environ.pop(k, None)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class UsageAccountingTests(EnvIsolatedTest):
    def _two_turns(self):
        u = A._new_usage()
        A._accumulate_usage(u, _Resp({
            "input_tokens": 1000, "output_tokens": 200,
            "input_token_details": {"cache_read": 8000, "cache_creation": 0},
        }))
        A._accumulate_usage(u, _Resp({
            "input_tokens": 500, "output_tokens": 100,
            "input_token_details": {"cache_read": 8000},
        }))
        return u

    def test_totals_accumulate_across_turns(self):
        u = self._two_turns()
        self.assertEqual(u["input_tokens"], 1500)
        self.assertEqual(u["output_tokens"], 300)
        self.assertEqual(u["cache_read_tokens"], 16000)
        self.assertEqual(u["llm_calls"], 2)
        self.assertEqual(u["usage_reported_calls"], 2)

    def test_missing_usage_metadata_is_counted_not_silently_dropped(self):
        """A provider that reports nothing must not make totals look complete."""
        u = self._two_turns()
        A._accumulate_usage(u, _Resp(None))
        self.assertEqual(u["llm_calls"], 3)
        self.assertEqual(u["usage_reported_calls"], 2)
        fin = A._finalize_usage(u, "anthropic", "claude-sonnet-4-6")
        self.assertIn("lower bound", fin["warning"])

    def test_cost_applies_cache_read_discount(self):
        u = self._two_turns()
        fin = A._finalize_usage(u, "anthropic", "claude-sonnet-4-6")
        expected = (1500 * 3.0 + 16000 * 3.0 * A._CACHE_READ_MULT + 300 * 15.0) / 1_000_000
        self.assertAlmostEqual(fin["estimated_cost_usd"], round(expected, 4), places=6)
        self.assertEqual(fin["cost_basis"], "model_price_table")

    def test_cache_write_priced_at_premium(self):
        u = A._new_usage()
        A._accumulate_usage(u, _Resp({
            "input_tokens": 0, "output_tokens": 0,
            "input_token_details": {"cache_creation": 1_000_000},
        }))
        fin = A._finalize_usage(u, "anthropic", "claude-sonnet-4-6")
        self.assertAlmostEqual(fin["estimated_cost_usd"], round(3.0 * A._CACHE_WRITE_MULT, 4), places=6)

    def test_cache_hit_ratio(self):
        fin = A._finalize_usage(self._two_turns(), "anthropic", "claude-sonnet-4-6")
        self.assertEqual(fin["cache_hit_ratio"], round(16000 / 17500, 3))

    def test_zero_usage_does_not_divide_by_zero(self):
        fin = A._finalize_usage(A._new_usage(), "anthropic", "claude-sonnet-4-6")
        self.assertIsNone(fin["cache_hit_ratio"])
        self.assertEqual(fin["estimated_cost_usd"], 0.0)

    def test_local_inference_is_free(self):
        fin = A._finalize_usage(self._two_turns(), "ollama", "any-local-tag")
        self.assertEqual(fin["estimated_cost_usd"], 0.0)
        self.assertEqual(fin["cost_basis"], "local_inference")

    def test_unknown_model_omits_cost_rather_than_guessing(self):
        """A wrong cost number is worse than no cost number."""
        fin = A._finalize_usage(self._two_turns(), "anthropic", "model-released-tomorrow")
        self.assertIsNone(fin["estimated_cost_usd"])
        self.assertIn("unknown_model", fin["cost_basis"])
        # Token counts must still be reported.
        self.assertEqual(fin["input_tokens"], 1500)

    def test_env_price_override_prices_unknown_model(self):
        os.environ["QA_MODEL_PRICES"] = "model-released-tomorrow:2.0:10.0"
        fin = A._finalize_usage(self._two_turns(), "anthropic", "model-released-tomorrow")
        expected = (1500 * 2.0 + 16000 * 2.0 * A._CACHE_READ_MULT + 300 * 10.0) / 1_000_000
        self.assertAlmostEqual(fin["estimated_cost_usd"], round(expected, 4), places=6)

    def test_malformed_env_price_is_ignored_not_fatal(self):
        os.environ["QA_MODEL_PRICES"] = "garbage,also:bad,x:notanumber:5"
        fin = A._finalize_usage(self._two_turns(), "anthropic", "x")
        self.assertIsNone(fin["estimated_cost_usd"])


class PromptCachingTests(EnvIsolatedTest):
    def test_anthropic_system_block_carries_cache_control(self):
        msg = A._build_system_message("anthropic")
        self.assertIsInstance(msg.content, list)
        self.assertEqual(msg.content[0]["cache_control"], {"type": "ephemeral"})
        self.assertIn("QA analyst", msg.content[0]["text"])

    def test_non_cached_providers_get_plain_text(self):
        for provider in ("ollama", "openai", "azure_openai"):
            with self.subTest(provider=provider):
                self.assertIsInstance(A._build_system_message(provider).content, str)

    def test_caching_can_be_disabled(self):
        os.environ["QA_PROMPT_CACHING"] = "0"
        self.assertIsInstance(A._build_system_message("anthropic").content, str)


class KnowledgeBundleTests(EnvIsolatedTest):
    def test_frontmatter_never_reaches_the_model(self):
        """
        Frontmatter keys must be stripped. Note the in-body '> **status: template**'
        callout is deliberate and DOES reach the model — the agent should know its
        guidance is unverified boilerplate. So assert on frontmatter-only markers.
        """
        prompt = A._system_prompt("anthropic")
        self.assertNotIn("type: QA Guideline", prompt)
        self.assertNotIn("verified: false", prompt)
        self.assertNotIn("tags: [qa", prompt)
        self.assertFalse(prompt.lstrip().startswith("---"))

    def test_no_unsubstituted_template_placeholder(self):
        """Regression: the template used {{...}} f-string escaping after moving to .replace()."""
        for provider in ("anthropic", "ollama"):
            with self.subTest(provider=provider):
                prompt = A._system_prompt(provider)
                self.assertNotIn("KNOWLEDGE_SECTION", prompt)
                self.assertNotIn("{#", prompt, "stray brace from template substitution")

    def test_strip_frontmatter_edge_cases(self):
        self.assertEqual(A._strip_frontmatter("# Title\nbody"), "# Title\nbody")
        self.assertEqual(A._strip_frontmatter("---\nk: v\n---\n# Title"), "# Title")
        self.assertEqual(A._strip_frontmatter(""), "")
        # Regression: a horizontal rule in the body must not truncate the document.
        body = A._strip_frontmatter("---\nk: v\n---\n# T\n\n---\n\ntrailing content")
        self.assertIn("---", body)
        self.assertIn("trailing content", body)
        # Unterminated frontmatter is not frontmatter — return unchanged, don't eat the file.
        self.assertEqual(A._strip_frontmatter("---\nk: v\nno close"), "---\nk: v\nno close")
        # A doc that merely starts with a rule, not a fence.
        self.assertEqual(A._strip_frontmatter("----\ntitle"), "----\ntitle")

    def test_inline_mode_includes_guidance_body(self):
        prompt = A._system_prompt("anthropic")
        self.assertIn("Naming Convention", prompt)     # from test-case-guidelines
        self.assertIn("OUTDATED", prompt)              # from deprecation-rules

    def test_on_demand_mode_defers_guidance(self):
        prompt = A._system_prompt("ollama")
        self.assertNotIn("Naming Convention", prompt)
        self.assertIn("read_knowledge", prompt)
        self.assertIn("test-case-guidelines.md", prompt)

    def test_on_demand_prompt_is_smaller(self):
        full, lean = A._system_prompt("anthropic"), A._system_prompt("ollama")
        self.assertLess(len(lean), len(full))
        self.assertGreater(len(full) - len(lean), 2000)

    def test_knowledge_mode_env_override_both_directions(self):
        os.environ["QA_AGENT_KNOWLEDGE_MODE"] = "on_demand"
        self.assertNotIn("Naming Convention", A._system_prompt("anthropic"))
        os.environ["QA_AGENT_KNOWLEDGE_MODE"] = "inline"
        self.assertIn("Naming Convention", A._system_prompt("ollama"))

    def test_invalid_knowledge_mode_falls_back_to_provider_default(self):
        os.environ["QA_AGENT_KNOWLEDGE_MODE"] = "nonsense"
        self.assertIn("Naming Convention", A._system_prompt("anthropic"))
        self.assertNotIn("Naming Convention", A._system_prompt("ollama"))

    def test_bundle_contains_expected_documents(self):
        self.assertGreaterEqual(set(A._knowledge_files()), {
            "index.md",
            "test-case-guidelines.md",
            "deprecation-rules.md",
            "prd-to-knowledge-base.md",
        })

    def test_every_bundle_doc_is_listed_in_the_index(self):
        """A document the index never mentions is a document the agent will never read."""
        index = (A._PROMPTS_DIR / "index.md").read_text()
        for name in A._knowledge_files():
            if name == "index.md":
                continue
            self.assertIn(name, index, f"{name} is missing from index.md")


if __name__ == "__main__":
    unittest.main()
