# External MCP Server Configuration

Bombardier reaches Jira and Xray through an **external MCP server** over Streamable HTTP.
That server is **not part of this project** and is not built, run, or version-managed here.
Bombardier assumes it is already deployed, reachable and functional; all this repo needs is
configuration to address its tools.

## What Bombardier needs from your server

Nine **logical operations**. Each maps to one tool on your server:

| Logical operation | Default tool name | Used by |
|---|---|---|
| `get_folders` | `xray_get_folders` | Test sync — folder enumeration |
| `get_tests_from_folder` | `xray_get_tests_from_folder` | Test sync — paginated test listing |
| `get_test` | `xray_get_test` | Test sync — steps + preconditions per test |
| `update_test` | `xray_update_test` | Write-back — UPDATE decisions |
| `bulk_create_tests` | `xray_bulk_create_tests` | Write-back — CREATE decisions |
| `search_issues` | `jira_search_issues` | Test sync — bulk descriptions/labels; DEPRECATE label read |
| `update_issue` | `jira_update_issue` | Write-back — DEPRECATE label write |
| `add_comment` | `jira_add_comment` | Write-back — DEPRECATE rationale, and UPDATE prose recommendations |
| `add_remote_link` | `jira_add_remote_link` | Write-back — link new tests to the source PRD |

The defaults follow common Jira/Xray MCP naming. **If your server names them differently, that
is configuration, not a code change.**

## Pointing at your server

### Containerised (default)

Each MCP server ships its own `Dockerfile` and `docker-compose.yml`, so no local Node
install is needed. All of them — plus Bombardier — attach to a shared bridge network named
`mcp-net`, declared **external** everywhere so no single repo owns its lifecycle. Bombardier
then reaches the server by service name:

```bash
XRAY_MCP_URL=http://xray-mcp:3100/mcp     # the Compose default
```

Start the MCP server, then Bombardier — the shared `mcp-net` network is created
automatically by whichever comes up first, so order does not actually matter:

```bash
cd ../xray-mcp   && cp env.example .env   # fill in credentials
                    docker compose up -d --build

cd ../Bombardier && docker compose up -d --build
```

To run the GitLab server alongside it in one command, see the header of
`../xray-mcp/docker-compose.yml` — merged compose files need explicit build-context vars
and one `--env-file` per repo, because `-f` resolves all relative paths against the first
file's directory and auto-loads only that directory's `.env`.

No host ports are published, which keeps an unauthenticated, write-capable Jira/Xray proxy
off your network. Uncomment the `ports` block in the server's compose file for host-side
debugging.

### Server on the host instead

```bash
XRAY_MCP_URL=http://host.docker.internal:3100/mcp
```

`host.docker.internal` resolves to the Docker host via the `extra_hosts` entry in
`docker-compose.yml`. Note it maps to the bridge gateway, not loopback — so a server bound
only to `127.0.0.1` on the host is **not** reachable from the container.

### Anywhere else

```bash
XRAY_MCP_URL=https://mcp.internal.example.com/mcp
```

## Remapping tool names

`XRAY_MCP_TOOL_MAP` is a JSON object merged over the defaults — declare only what differs.

Rename a tool:

```bash
XRAY_MCP_TOOL_MAP='{"get_test": "my_xray_fetch_test"}'
```

Rename a tool **and** its argument keys — this is what you need when a server uses
`snake_case` where the defaults use `camelCase`:

```bash
XRAY_MCP_TOOL_MAP='{
  "get_tests_from_folder": {
    "name": "list_tests",
    "args": {"projectKey": "project_key", "folderPath": "folder", "includeDescendants": "recursive"}
  },
  "search_issues": {"name": "jira_jql_search", "args": {"maxResults": "max_results"}}
}'
```

For a single quick override there is also a per-operation form, which takes precedence over
the JSON map:

```bash
XRAY_MCP_TOOL_GET_TEST=fetch_test
```

## Verifying the configuration

Bombardier checks the mapping against the server's advertised tool list **at startup** and
logs the result. Check it any time:

```bash
curl -s localhost:8000/integrations/mcp/tools | jq
```

```json
{
  "status": "misconfigured",
  "url": "http://host.docker.internal:3100/mcp",
  "configured": { "get_test": "xray_get_test", "...": "..." },
  "available": ["jira_jql_search", "xray_fetch_test", "..."],
  "missing": { "get_test": "xray_get_test" }
}
```

| `status` | Meaning |
|---|---|
| `ok` | Every configured tool exists on the server. HTTP 200. |
| `misconfigured` | See `missing`; fix with `XRAY_MCP_TOOL_MAP`. HTTP 503. |
| `unreachable` | Server down or wrong `XRAY_MCP_URL`; nothing verified. HTTP 503. |

Do this **before** debugging a failed sync. A name mismatch otherwise surfaces as an opaque
`Tool error: unknown tool` partway through Phase 2.

## Response envelopes

Many servers wrap results as `{"success": true, "data": {...}}`. Bombardier unwraps `data`
automatically. If your server returns a literal top-level `data` field that must be preserved:

```bash
XRAY_MCP_UNWRAP_DATA=0
```

## What is *not* configurable

**Response shapes.** `integrations/xray_client.py` reads specific fields:

| Operation | Expected response shape | Verified against `jira-xray-mcp` |
|---|---|---|
| `get_folders` | `{"folders": [{"path"\|"name": str}, ...]}` | ✅ returns `{name, path, issuesCount, testsCount, preconditionsCount, folders}` |
| `get_tests_from_folder` | `{"total": int, "results": [{"key"\|"jiraKey": str, ...}]}` | ✅ returns `{total, results:[{issueId, key, summary, labels, folder}]}` — `key`/`summary` flattened out of the GraphQL `jira{}` node |
| `get_test` | `{"steps": [{"action","data","expectedResult"}], "preconditions": [...]}` | ✅ after fixing a broken query (see below) |
| `search_issues` | `{"issues": [{"key", "fields": {"summary","description","labels"}}]}` | ✅ standard Jira v3; descriptions arrive as ADF and are flattened by `_adf_to_text` |
| `bulk_create_tests` | `{"keys"\|"createdKeys": [str]}` or `[{"key": str}]` | ⚠️ untested (write path) |

### Deployment notes for `jira-xray-mcp`

Verified live against `example.atlassian.net` / Xray Cloud:

- **Tool names need no remapping.** Eight of the nine logical operations match the defaults
  exactly, so `XRAY_MCP_TOOL_MAP` stays empty.
- **`jira_add_remote_link` does not exist on this server.** Write-back uses it to link newly
  created tests back to the source PRD. `agents/writeback.py` wraps it in try/except, so
  CREATE still succeeds and logs `Remote link failed` — you lose the backlink, nothing more.
  No remap helps here; the tool would have to be implemented.
- **`get_tests_from_folder` returns `labels` and `folder`**, not just `key`/`summary`. That
  matters for `sync/test_sync.py`: `_metadata_hash` covers summary + folder + labels +
  testType, and three of those four are now populated from the listing. Only `testType` is
  absent, so it always hashes as `""`. Step and description changes are still invisible to
  the metadata pre-filter — that gap is in Bombardier, not the MCP.
- **`get_test` was broken** and is fixed in this checkout: the GraphQL query requested
  `preconditions` without its non-null `limit` argument, so the whole query failed with
  `Field "preconditions" argument "limit" of type "Int!" is required but not provided`.
  Every call returned a validation error, which would have failed Phase 4b for every test.
  Rebuild the MCP image after pulling that fix.
- **`RATE_LIMIT_REQUESTS_PER_SECOND`** is set to 15 (code default is 5). `xray_get_test`
  makes two upstream calls, and sync fans out 12-wide, so 5/s throttles it to ~2.5 tests/s.

A server with a materially different schema needs code changes in `xray_client.py`, not
configuration. Descriptions in Jira REST v3 ADF format are handled (`_adf_to_text`).

## Other integrations do not use MCP

- **Confluence** — REST API v1/v2 directly (`ingestion/confluence_ingestor.py`,
  `confluence_space_ingestor.py`), authenticated with `CONFLUENCE_EMAIL` +
  `CONFLUENCE_API_TOKEN`.
- **GitLab** — REST API directly (`ingestion/gitlab_ingestor.py`), authenticated with
  `GITLAB_TOKEN`. There is **no** GitLab MCP dependency; an earlier `GITLAB_MCP_URL` variable
  was referenced in config and docs but read by no code, and has been removed.

## Timeouts and connection reuse

| Variable | Default | Notes |
|---|---|---|
| `MCP_TOOL_TIMEOUT_SEC` | `60` | Per tool call. Raise for slow bulk operations. |

The client keeps a pooled session, refreshed every 5 minutes, pinged after 60 s idle, with a
one-shot fallback and backoff retry on transient failures (429/502/503/504, timeouts, reset
connections). Nothing to configure.
