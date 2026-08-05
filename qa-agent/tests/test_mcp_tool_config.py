"""
Guards the external MCP tool configuration.

The MCP server is a separate deployment, not a project dependency. Tool names used to be
hardcoded at ten call sites, so pointing Bombardier at a server with different naming meant
editing Python. These tests pin the contract that tool names and argument keys are
configuration, and that a misconfiguration is reported rather than discovered mid-sync.
"""
import os
import sys
import types
import unittest


def _install_stubs() -> None:
    """Stub the mcp client package — the transport is irrelevant to name resolution."""
    if "mcp" in sys.modules:
        return
    mcp = types.ModuleType("mcp")

    class _ClientSession:
        def __init__(self, *a, **kw):
            pass

    mcp.ClientSession = _ClientSession
    http_mod = types.ModuleType("mcp.client.streamable_http")
    http_mod.streamablehttp_client = lambda *a, **kw: None
    client_pkg = types.ModuleType("mcp.client")
    sys.modules.update({
        "mcp": mcp,
        "mcp.client": client_pkg,
        "mcp.client.streamable_http": http_mod,
    })


_install_stubs()
from integrations import xray_client as X  # noqa: E402


class ToolConfigTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: v for k, v in os.environ.items() if k.startswith("XRAY_MCP_")}
        for k in list(os.environ):
            if k.startswith("XRAY_MCP_TOOL"):
                os.environ.pop(k)
        X.reset_tool_config()

    def tearDown(self):
        for k in list(os.environ):
            if k.startswith("XRAY_MCP_"):
                os.environ.pop(k)
        os.environ.update(self._saved)
        X.reset_tool_config()

    # ── defaults ──────────────────────────────────────────────────────────────

    def test_defaults_resolve_to_conventional_names(self):
        self.assertEqual(X._resolve("get_test", {})[0], "xray_get_test")
        self.assertEqual(X._resolve("search_issues", {})[0], "jira_search_issues")

    def test_every_logical_operation_is_declared(self):
        """Each operation the code calls must exist in the defaults table."""
        expected = {
            "get_folders", "get_tests_from_folder", "get_test", "update_test",
            "bulk_create_tests", "search_issues", "update_issue", "add_comment",
            "add_remote_link",
        }
        self.assertEqual(set(X._TOOL_SPEC_DEFAULTS), expected)

    def test_unknown_operation_raises_with_a_useful_message(self):
        with self.assertRaises(ValueError) as ctx:
            X._resolve("not_an_operation", {})
        self.assertIn("get_test", str(ctx.exception), "should list known operations")

    # ── name overrides ────────────────────────────────────────────────────────

    def test_string_form_renames_a_tool(self):
        os.environ["XRAY_MCP_TOOL_MAP"] = '{"get_test": "my_fetch_test"}'
        X.reset_tool_config()
        self.assertEqual(X._resolve("get_test", {})[0], "my_fetch_test")
        # Untouched operations keep their defaults.
        self.assertEqual(X._resolve("get_folders", {})[0], "xray_get_folders")

    def test_object_form_renames_tool_and_arguments(self):
        os.environ["XRAY_MCP_TOOL_MAP"] = (
            '{"get_tests_from_folder": {"name": "list_tests",'
            ' "args": {"projectKey": "project_key", "folderPath": "folder"}}}'
        )
        X.reset_tool_config()
        name, args = X._resolve("get_tests_from_folder", {
            "projectKey": "PROJ", "folderPath": "/Platform", "limit": 100,
        })
        self.assertEqual(name, "list_tests")
        self.assertEqual(args, {"project_key": "PROJ", "folder": "/Platform", "limit": 100})

    def test_args_only_override_keeps_default_name(self):
        os.environ["XRAY_MCP_TOOL_MAP"] = '{"get_test": {"args": {"testKey": "test_key"}}}'
        X.reset_tool_config()
        name, args = X._resolve("get_test", {"testKey": "PROJ-1"})
        self.assertEqual(name, "xray_get_test")
        self.assertEqual(args, {"test_key": "PROJ-1"})

    def test_per_operation_env_var_wins_over_json_map(self):
        os.environ["XRAY_MCP_TOOL_MAP"] = '{"get_test": "from_json"}'
        os.environ["XRAY_MCP_TOOL_GET_TEST"] = "from_env"
        X.reset_tool_config()
        self.assertEqual(X._resolve("get_test", {})[0], "from_env")

    # ── robustness ────────────────────────────────────────────────────────────

    def test_malformed_json_falls_back_to_defaults(self):
        """A bad config must not take the integration down silently or loudly."""
        os.environ["XRAY_MCP_TOOL_MAP"] = "{not json"
        X.reset_tool_config()
        self.assertEqual(X._resolve("get_test", {})[0], "xray_get_test")

    def test_non_object_json_falls_back_to_defaults(self):
        os.environ["XRAY_MCP_TOOL_MAP"] = '["a", "b"]'
        X.reset_tool_config()
        self.assertEqual(X._resolve("get_test", {})[0], "xray_get_test")

    def test_unknown_operation_in_map_is_ignored_not_fatal(self):
        os.environ["XRAY_MCP_TOOL_MAP"] = '{"typo_operation": "x", "get_test": "ok_name"}'
        X.reset_tool_config()
        self.assertEqual(X._resolve("get_test", {})[0], "ok_name")

    def test_wrong_value_type_is_ignored(self):
        os.environ["XRAY_MCP_TOOL_MAP"] = '{"get_test": 123}'
        X.reset_tool_config()
        self.assertEqual(X._resolve("get_test", {})[0], "xray_get_test")

    def test_empty_map_is_a_no_op(self):
        os.environ["XRAY_MCP_TOOL_MAP"] = "   "
        X.reset_tool_config()
        self.assertEqual(X._resolve("get_test", {})[0], "xray_get_test")

    def test_resolve_does_not_mutate_caller_arguments(self):
        os.environ["XRAY_MCP_TOOL_MAP"] = '{"get_test": {"args": {"testKey": "test_key"}}}'
        X.reset_tool_config()
        original = {"testKey": "PROJ-1"}
        X._resolve("get_test", original)
        self.assertEqual(original, {"testKey": "PROJ-1"})

    # ── diagnostics ───────────────────────────────────────────────────────────

    def test_configured_tools_reports_the_effective_mapping(self):
        os.environ["XRAY_MCP_TOOL_MAP"] = '{"get_test": "custom"}'
        X.reset_tool_config()
        tools = X.configured_tools()
        self.assertEqual(tools["get_test"], "custom")
        self.assertEqual(len(tools), len(X._TOOL_SPEC_DEFAULTS))


class EnvelopeUnwrapTests(unittest.TestCase):
    """_parse_result unwraps {"success": true, "data": ...} unless told otherwise."""

    def setUp(self):
        os.environ.pop("XRAY_MCP_UNWRAP_DATA", None)

    def tearDown(self):
        os.environ.pop("XRAY_MCP_UNWRAP_DATA", None)

    def _result(self, text):
        block = types.SimpleNamespace(text=text)
        return types.SimpleNamespace(content=[block])

    def test_envelope_is_unwrapped_by_default(self):
        out = X._parse_result(self._result('{"success": true, "data": {"total": 3}}'))
        self.assertEqual(out, {"total": 3})

    def test_unwrap_can_be_disabled(self):
        os.environ["XRAY_MCP_UNWRAP_DATA"] = "0"
        out = X._parse_result(self._result('{"success": true, "data": {"total": 3}}'))
        self.assertEqual(out, {"success": True, "data": {"total": 3}})

    def test_unenveloped_payload_passes_through(self):
        out = X._parse_result(self._result('{"total": 3, "results": []}'))
        self.assertEqual(out, {"total": 3, "results": []})

    def test_non_json_returns_raw_text(self):
        self.assertEqual(X._parse_result(self._result("plain text")), "plain text")

    def test_empty_content_returns_none(self):
        self.assertIsNone(X._parse_result(types.SimpleNamespace(content=[])))


class JiraKeyValidationTests(unittest.TestCase):
    """Key validation is the JQL-injection guard — must stay strict."""

    def test_valid_keys_accepted(self):
        for key in ("PROJ-1", "AB-12345", "A1_B-7"):
            with self.subTest(key=key):
                self.assertEqual(X._validate_jira_key(key), key)

    def test_injection_attempts_rejected(self):
        for key in ("PROJ-1 OR 1=1", "proj-1", "PROJ-1; DROP", "", "PROJ-", "-1",
                    "PROJ-1)", "PROJ-1 ORDER BY created"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    X._validate_jira_key(key)


if __name__ == "__main__":
    unittest.main()
