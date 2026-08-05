"""
Guards the agent's context-budget math, especially for local (ollama) inference.

Why this test exists: Ollama treats num_ctx as a hard prompt wall and silently drops
overflow instead of erroring. If compaction triggers too late, the PRD is truncated
without any signal and the agent quietly analyses a partial document. This test pins
the invariant `compact_threshold + max_output_tokens < num_ctx`.

Heavy deps (torch, elasticsearch, psycopg2, tiktoken) are stubbed so this runs under the
documented host-side command:
    cd qa-agent && PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'
"""
import os
import sys
import types
import unittest


def _install_stubs() -> None:
    """Minimal stand-ins for packages that only exist inside the container."""
    if "tiktoken" not in sys.modules:
        tk = types.ModuleType("tiktoken")
        tk.get_encoding = lambda _name: types.SimpleNamespace(
            encode=lambda s, disallowed_special=(): s.split()
        )
        sys.modules["tiktoken"] = tk

    if "langchain_core" not in sys.modules:
        class _Msg:
            def __init__(self, content=None, tool_call_id=None, **kw):
                self.content = content
                self.tool_call_id = tool_call_id
                self.tool_calls = []

        class _Tool:
            """Mimics langchain_core.tools.tool closely enough to exercise tool bodies."""

            def __init__(self, fn):
                self._fn = fn
                self.name = fn.__name__
                self.description = fn.__doc__ or ""
                self.args_schema = None

            def invoke(self, args=None):
                return self._fn(**(args or {}))

            def __call__(self, *a, **kw):
                return self._fn(*a, **kw)

        msgs = types.ModuleType("langchain_core.messages")
        msgs.HumanMessage = msgs.SystemMessage = msgs.ToolMessage = _Msg
        tools_mod = types.ModuleType("langchain_core.tools")
        tools_mod.tool = _Tool
        sys.modules["langchain_core"] = types.ModuleType("langchain_core")
        sys.modules["langchain_core.messages"] = msgs
        sys.modules["langchain_core.tools"] = tools_mod

    for name in ("sentence_transformers", "numpy"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__getattr__ = lambda _k: types.SimpleNamespace()
            sys.modules[name] = mod

    if "elasticsearch" not in sys.modules:
        es = types.ModuleType("elasticsearch")
        es.Elasticsearch = object
        es.NotFoundError = type("NotFoundError", (Exception,), {})
        es.helpers = types.SimpleNamespace(scan=lambda *a, **k: [])
        sys.modules["elasticsearch"] = es

    if "psycopg2" not in sys.modules:
        pg = types.ModuleType("psycopg2")
        pool = types.ModuleType("psycopg2.pool")
        pool.ThreadedConnectionPool = object
        extras = types.ModuleType("psycopg2.extras")
        extras.Json = lambda v: v
        extras.RealDictCursor = object
        pg.pool, pg.extras = pool, extras
        pg.OperationalError = type("OperationalError", (Exception,), {})
        sys.modules.update(
            {"psycopg2": pg, "psycopg2.pool": pool, "psycopg2.extras": extras}
        )


_install_stubs()
from agents import analysis_agent as A  # noqa: E402


class ContextBudgetTests(unittest.TestCase):
    _ENV_KEYS = ("OLLAMA_NUM_CTX", "QA_AGENT_MAX_OUTPUT_TOKENS", "QA_AGENT_MAX_TURNS")

    def setUp(self):
        self._saved = {k: os.environ.get(k) for k in self._ENV_KEYS}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    # ─── hosted providers ─────────────────────────────────────────────────────

    def test_hosted_budgets_come_from_the_model_window(self):
        """
        Superseded the old per-provider constants (anthropic was hardcoded to 180_000,
        which capped a 1M model at 18% of its window). Detailed coverage of the
        model-aware budget lives in test_context_windows.py.
        """
        for provider in ("anthropic", "azure_openai", "openai"):
            with self.subTest(provider=provider):
                limit = A._context_limit(provider, "")
                self.assertGreater(limit, 8_000)
                self.assertLessEqual(limit, A._PROVIDER_FALLBACK_WINDOW[provider])

    def test_unknown_provider_falls_back(self):
        limit = A._context_limit("some-new-provider", "")
        self.assertGreater(limit, 8_000)
        self.assertLessEqual(limit, 128_000)

    # ─── ollama: budget derived from num_ctx ──────────────────────────────────

    def test_ollama_budget_reserves_output_room(self):
        os.environ["OLLAMA_NUM_CTX"] = "32768"
        os.environ["QA_AGENT_MAX_OUTPUT_TOKENS"] = "8192"
        self.assertEqual(A._context_limit("ollama"), 32768 - 8192 - 2000)

    def test_ollama_budget_never_negative(self):
        """A too-small num_ctx must floor, not produce a negative/zero budget."""
        os.environ["OLLAMA_NUM_CTX"] = "2048"
        os.environ["QA_AGENT_MAX_OUTPUT_TOKENS"] = "8192"
        self.assertEqual(A._context_limit("ollama"), 8_000)

    def test_ollama_budget_scales_with_larger_window(self):
        os.environ["OLLAMA_NUM_CTX"] = "131072"
        os.environ["QA_AGENT_MAX_OUTPUT_TOKENS"] = "8192"
        self.assertEqual(A._context_limit("ollama"), 131072 - 8192 - 2000)

    def test_compaction_fires_before_the_ollama_prompt_wall(self):
        """
        The invariant that matters: prompt after compaction, plus the model's own
        output, must stay inside num_ctx — otherwise Ollama truncates silently.
        """
        for num_ctx, out in (("16384", "4096"), ("32768", "8192"), ("65536", "8192")):
            with self.subTest(num_ctx=num_ctx, max_output=out):
                os.environ["OLLAMA_NUM_CTX"] = num_ctx
                os.environ["QA_AGENT_MAX_OUTPUT_TOKENS"] = out
                compact_at = int(A._context_limit("ollama") * A._TOKEN_COMPACT_RATIO)
                self.assertLess(compact_at + int(out), int(num_ctx))

    # ─── turn cap ─────────────────────────────────────────────────────────────

    def test_max_turns_env_override(self):
        os.environ.pop("QA_AGENT_MAX_TURNS", None)
        self.assertEqual(A._default_max_turns(), 40)
        os.environ["QA_AGENT_MAX_TURNS"] = "15"
        self.assertEqual(A._default_max_turns(), 15)

    # ─── prose-narration detection ────────────────────────────────────────────

    def test_detects_narrated_tool_calls(self):
        tools = {"search_tests": 1, "record_decision": 1, "read_prd_document": 1}
        self.assertTrue(A._mentions_tool_name("I'll call search_tests next", tools))
        self.assertTrue(A._mentions_tool_name("now using RECORD_DECISION", tools))
        self.assertFalse(A._mentions_tool_name("Checkout looks well covered.", tools))
        self.assertFalse(A._mentions_tool_name("", tools))
        self.assertFalse(A._mentions_tool_name(None, tools))

    # ─── provider validation ──────────────────────────────────────────────────

    def test_unknown_provider_error_mentions_ollama(self):
        with self.assertRaises(ValueError) as ctx:
            A._build_llm("gemma", "some-model")
        self.assertIn("ollama", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
