"""
Shared stand-ins for packages that only exist inside the container.

Not named test_*.py, so unittest discovery does not collect it.

Why these install per ATTRIBUTE rather than per module
------------------------------------------------------
All test modules share one sys.modules. Several install their own stubs, and the
first one to run wins — so a plain "if 'numpy' not in sys.modules" guard can leave
another module's inert stand-in in place. When that happens the chunker's
_cosine_sim raises, the chunker catches it and silently falls back to fixed-window
chunking, and the semantic assertions pass without exercising the semantic path.

The same trap has a second form: a stub that *works* but measures differently.
tests/test_context_budget.py tokenizes on whitespace while these count characters,
so a fixture sized in one unit can be under the threshold in the other. Keep test
fixtures large under BOTH (many whitespace-separated words, not one long run of
characters) rather than assuming which stub is active.
"""
import math
import sys
import types


def _works(fn) -> bool:
    try:
        fn()
        return True
    except Exception:
        return False


def _percentile(values, p):
    s = sorted(values)
    if not s:
        return 0.0
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = math.floor(k), math.ceil(k)
    if lo == hi:
        return s[int(k)]
    return s[lo] * (hi - k) + s[hi] * (k - lo)


def install_tiktoken() -> None:
    if "tiktoken" not in sys.modules:
        sys.modules["tiktoken"] = types.ModuleType("tiktoken")
    tk = sys.modules["tiktoken"]

    class _Enc:
        def encode(self, text, disallowed_special=()):
            # ~4 chars/token, the ratio cl100k_base averages on prose.
            return [0] * max(1, len(text) // 4)

    if not _works(lambda: tk.get_encoding("cl100k_base").encode("probe")):
        tk.get_encoding = lambda name: _Enc()


def install_numpy() -> None:
    if "numpy" not in sys.modules:
        sys.modules["numpy"] = types.ModuleType("numpy")
    np = sys.modules["numpy"]

    if not _works(lambda: np.array([1.0, 2.0])):
        np.array = lambda x: list(x)
    if not _works(lambda: np.dot([1.0], [1.0])):
        np.dot = lambda a, b: sum(i * j for i, j in zip(a, b))
    if not _works(lambda: np.linalg.norm([3.0, 4.0])):
        np.linalg = types.SimpleNamespace(norm=lambda a: math.sqrt(sum(i * i for i in a)))
    if not _works(lambda: np.percentile([1.0, 2.0], 50)):
        np.percentile = _percentile


def install_html2text(render=None) -> None:
    """
    Minimal HTML2Text stand-in. `render` overrides the handler; the default
    returns the HTML unchanged, which is enough for tests that do not assert on
    rendered prose.
    """
    if "html2text" in sys.modules:
        return
    mod = types.ModuleType("html2text")

    class _H2T:
        ignore_links = ignore_images = ignore_tables = False
        body_width = 0
        unicode_snob = True

        def handle(self, html):
            return render(html) if render else html

    mod.HTML2Text = _H2T
    sys.modules["html2text"] = mod


def install_chunker_deps() -> None:
    """
    The third-party packages ingestion.chunker imports.

    `observability` is deliberately NOT stubbed. It is an in-repo package that
    imports cleanly with no third-party deps, and trace.event() is a no-op unless
    QA_TRACE=1 — so a stub buys nothing. It also costs something: a stub module
    with an empty __path__ shadows the real package, and every later
    `from observability.canonical_json import ...` in the suite then fails with
    ModuleNotFoundError depending on which test module imported first.
    """
    install_tiktoken()
    install_numpy()
