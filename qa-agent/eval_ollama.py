"""
Gemma 3 PRD Coverage Analysis (Ollama, host-side)
===================================================
Fetches retrieval data from the QA API (localhost:8000), then sends it
to a locally running Ollama model for analysis.

Read-only: no Jira writes, no git operations, no database writes.
Uses only Python stdlib (urllib) — no extra packages needed.

Usage:
    python3 eval_ollama.py --prd confluence:1234567890 --module Platform
    python3 eval_ollama.py --prd confluence:1234567890 --module Platform --model gemma3
    python3 eval_ollama.py --prd confluence:1234567890 --module Platform --prompt-only
"""

import argparse
import json
import sys
import urllib.error
import urllib.request


API_BASE   = "http://localhost:8000"
OLLAMA_URL = "http://localhost:11434"


# ─── ANSI helpers ──────────────────────────────────────────────────────────────

def bold(t):   return f"\033[1m{t}\033[0m"
def green(t):  return f"\033[32m{t}\033[0m"
def yellow(t): return f"\033[33m{t}\033[0m"
def red(t):    return f"\033[31m{t}\033[0m"
def dim(t):    return f"\033[2m{t}\033[0m"


# ─── HTTP helpers ──────────────────────────────────────────────────────────────

def post_json(url: str, payload: dict, timeout: int = 180) -> dict:
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def post_json_stream(url: str, payload: dict, timeout: int = 600):
    """Yields parsed JSON objects line by line from a streaming response."""
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        while True:
            raw_line = resp.readline()
            if not raw_line:
                break
            line = raw_line.decode().strip()
            if line:
                yield json.loads(line)


# ─── Fetch ─────────────────────────────────────────────────────────────────────

def get_preview(prd_source_id: str, module: list) -> dict:
    """
    Calls /analyze/preview — returns full PRD text + all retrieved tests.
    No LLM is invoked; it's pure retrieval.
    """
    payload = {
        "prd_source_id": prd_source_id,
        "provider":      "anthropic",        # ignored — preview never calls LLM
        "model":         "claude-sonnet-4-6",
    }
    if module:
        payload["module"] = module
    return post_json(f"{API_BASE}/analyze/preview", payload, timeout=180)


# ─── Prompt builder ────────────────────────────────────────────────────────────

def build_prompt(data: dict, module: list, max_tests: int = 25) -> str:
    prd_text  = data.get("prd_document", "")
    all_tests = data.get("all_tests", [])

    high_tests   = [t for t in all_tests if t.get("confidence") == "high"]
    medium_tests = [t for t in all_tests if t.get("confidence") == "medium"]

    # Cap: fill high-confidence first, then medium with remaining budget
    high_cap   = min(len(high_tests), max_tests)
    medium_cap = min(len(medium_tests), max(0, max_tests - high_cap))
    high_tests   = high_tests[:high_cap]
    medium_tests = medium_tests[:medium_cap]

    def fmt_test(t):
        labels = ", ".join(t.get("labels") or []) or "—"
        via    = ", ".join(t.get("matched_queries") or [])[:60]
        return f"  [{t['jira_key']}] {t['summary'][:75]}\n    Labels: {labels} | Matched via: {via}"

    tests_section = ""
    if high_tests:
        tests_section += f"\n### High Confidence Tests ({len(high_tests)}):\n"
        tests_section += "\n".join(fmt_test(t) for t in high_tests)
    if medium_tests:
        tests_section += f"\n\n### Medium Confidence Tests ({len(medium_tests)}):\n"
        tests_section += "\n".join(fmt_test(t) for t in medium_tests)

    module_str = ", ".join(module) if module else "all modules"
    sep = "=" * 70

    return (
        f"You are a QA analyst reviewing test coverage for a Product Requirements Document (PRD).\n\n"
        f"MODULE SCOPE: {module_str}\n"
        f"TESTS RETRIEVED: {len(all_tests)} total "
        f"({len(high_tests)} high confidence, {len(medium_tests)} medium confidence)\n\n"
        f"{sep}\n"
        f"SECTION A — PRODUCT REQUIREMENTS DOCUMENT (NEW EXPECTATIONS)\n"
        f"{sep}\n\n"
        f"{prd_text}\n\n"
        f"{sep}\n"
        f"SECTION B — CURRENT TEST COVERAGE (EXISTING TESTS)\n"
        f"{sep}\n"
        f"{tests_section}\n\n"
        f"{sep}\n"
        f"YOUR TASK\n"
        f"{sep}\n\n"
        f"Compare Section A (PRD requirements) against Section B (existing tests).\n\n"
        f"For each PRD requirement or feature, determine:\n\n"
        f"KEEP       — existing tests that still fully cover the requirement\n"
        f"UPDATE     — existing tests that need changes (feature changed, steps outdated)\n"
        f"DEPRECATE  — existing tests for features removed or replaced\n"
        f"CREATE     — new tests needed for requirements with no existing coverage\n\n"
        f"Output format:\n\n"
        f"## Coverage Analysis\n\n"
        f"### Tests to KEEP\n"
        f"- [KEY] Summary — reason it still applies\n\n"
        f"### Tests to UPDATE\n"
        f"- [KEY] Summary — what exactly needs changing and why\n\n"
        f"### Tests to DEPRECATE\n"
        f"- [KEY] Summary — why it's no longer valid given the PRD\n\n"
        f"### New Tests to CREATE\n"
        f"- Test: <proposed test title>\n"
        f"  PRD section: <which section triggered this>\n"
        f"  Steps outline: <brief step outline>\n\n"
        f"## Summary\n"
        f"- KEEP: N\n"
        f"- UPDATE: N\n"
        f"- DEPRECATE: N\n"
        f"- CREATE: N\n\n"
        f"Rules:\n"
        f"- Only reference test keys you can see in Section B\n"
        f"- Be specific about what changed and why\n"
        f"- Prefer UPDATE over DEPRECATE when the feature still exists in some form\n"
        f"- For CREATE, only propose tests for requirements clearly stated in the PRD\n"
    )


# ─── Ollama call ───────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str, stream: bool = True) -> None:
    payload = {
        "model":    model,
        "messages": [{"role": "user", "content": prompt}],
        "stream":   stream,
        "options":  {"num_ctx": 65536, "temperature": 0.1},
    }

    if stream:
        for chunk in post_json_stream(f"{OLLAMA_URL}/api/chat", payload):
            token = chunk.get("message", {}).get("content", "")
            print(token, end="", flush=True)
            if chunk.get("done"):
                p_tok = chunk.get("prompt_eval_count")
                o_tok = chunk.get("eval_count")
                if p_tok and o_tok:
                    print(f"\n\n{dim('Tokens — prompt: ' + str(p_tok) + '  output: ' + str(o_tok))}")
                break
        print()
    else:
        result = post_json(f"{OLLAMA_URL}/api/chat", {**payload, "stream": False}, timeout=600)
        print(result["message"]["content"])
        p_tok = result.get("prompt_eval_count")
        o_tok = result.get("eval_count")
        if p_tok and o_tok:
            print(f"\n{dim('Tokens — prompt: ' + str(p_tok) + '  output: ' + str(o_tok))}")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Analyse PRD test coverage using a local Gemma 3 via Ollama"
    )
    parser.add_argument("--prd",         required=True, help="e.g. confluence:1234567890")
    parser.add_argument("--module",      help="Module filter, e.g. Platform")
    parser.add_argument("--model",       default="gemma3", help="Ollama model (default: gemma3)")
    parser.add_argument("--max-tests",   type=int, default=25, help="Max tests to include in prompt (default: 25)")
    parser.add_argument("--no-stream",   action="store_true", help="Wait for full response before printing")
    parser.add_argument("--prompt-only", action="store_true", help="Print prompt and exit (no Ollama call)")
    args = parser.parse_args()

    module = [args.module] if args.module else None

    # ── Fetch retrieval data ──────────────────────────────────────────────────
    print(bold(f"\nFetching retrieval data for {args.prd} ..."))
    try:
        data = get_preview(args.prd, module)
    except urllib.error.URLError as e:
        print(red(f"Cannot connect to {API_BASE}: {e.reason}"))
        print(red("Is the qa-agent container running?"))
        sys.exit(1)

    first_line = (data.get("prd_document") or "").split("\n")[0].lstrip("# ").strip() or args.prd
    n_high     = data.get("total_high_confidence", 0)
    n_med      = data.get("total_medium_confidence", 0)
    n_chunks   = data.get("prd_chunks", 0)
    pool       = data.get("pool_size_used", "?")

    print(f"  Document   : {first_line}")
    print(f"  PRD chunks : {n_chunks}")
    print(f"  Tests found: {green(str(n_high))} high + {yellow(str(n_med))} medium  (pool={pool})")

    # ── Build prompt ─────────────────────────────────────────────────────────
    prompt     = build_prompt(data, module or [], max_tests=args.max_tests)
    char_count = len(prompt)
    tok_est    = char_count // 4

    if args.prompt_only:
        print(dim(f"\n--- PROMPT ({char_count:,} chars / ~{tok_est:,} tokens estimated) ---\n"))
        print(prompt)
        return

    # ── Call Ollama ──────────────────────────────────────────────────────────
    print(bold(f"\nSending to {args.model} via Ollama ..."))
    print(dim(f"Prompt: {char_count:,} chars (~{tok_est:,} tokens estimated)"))
    print()
    print(bold("=" * 70))
    print()

    try:
        call_ollama(prompt, model=args.model, stream=not args.no_stream)
    except urllib.error.URLError as e:
        print(red(f"\nCannot connect to Ollama at {OLLAMA_URL}: {e.reason}"))
        print(red("Is Ollama running? Try: ollama serve"))
        sys.exit(1)

    print()
    print(bold("=" * 70))


if __name__ == "__main__":
    main()
