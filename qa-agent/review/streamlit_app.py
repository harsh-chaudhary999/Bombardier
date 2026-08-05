"""
QA Intelligence Engine — Human Review Dashboard (Phase 4)

Full review interface for approving/rejecting analysis decisions
before they are written back to Xray.

Features:
  - Run selector: pick an analysis run to review
  - Decision cards: grouped by action (keep/update/deprecate/create/question)
  - Approve/reject with reviewer notes
  - Bulk approve/reject all decisions in a run
  - Write-back trigger for approved decisions
  - Decision stats and progress tracking
"""
import json
import os
from datetime import datetime

import requests
import streamlit as st

QA_AGENT_URL = os.environ.get("QA_AGENT_URL", "http://qa-agent:8000")
API_KEY = os.environ.get("QA_ENGINE_API_KEY", "")
REVIEW_UI_PASSWORD = os.environ.get("REVIEW_UI_PASSWORD", "")

st.set_page_config(page_title="QA Review Dashboard", layout="wide", page_icon="🔍")

st.session_state.setdefault("reviewer_id", "")

# Optional gate — set REVIEW_UI_PASSWORD in the Streamlit container env
if REVIEW_UI_PASSWORD and not st.session_state.get("review_ui_ok", False):
    st.title("QA Review — Sign in")
    pwd = st.text_input("Password", type="password")
    reviewer_at_login = st.text_input(
        "Reviewer ID / name (audit trail)",
        value=st.session_state.reviewer_id,
        help="Stored with each approval as reviewed_by / X-Reviewer-Id",
        placeholder="e.g. jdoe or qa-team",
    )
    if st.button("Unlock"):
        if pwd == REVIEW_UI_PASSWORD:
            st.session_state.review_ui_ok = True
            st.session_state.reviewer_id = (reviewer_at_login or "").strip() or "anonymous"
            st.rerun()
        st.error("Invalid password")
    st.stop()


# ─── API helpers ──────────────────────────────────────────────────────────────

def _headers() -> dict:
    h = {"Content-Type": "application/json"}
    if API_KEY:
        h["X-API-Key"] = API_KEY
    rid = (st.session_state.get("reviewer_id") or "").strip()
    if rid:
        h["X-Reviewer-Id"] = rid
    return h


def api_get(path: str, *, quiet_404: bool = False) -> dict | None:
    try:
        r = requests.get(f"{QA_AGENT_URL}{path}", headers=_headers(), timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        status = e.response.status_code if e.response is not None else None
        if quiet_404 and status == 404:
            return None
        st.error(f"API error: {e}")
        return None
    except Exception as e:
        st.error(f"API error: {e}")
        return None


def api_post(path: str, body: dict | None = None, timeout: float = 60) -> dict | None:
    try:
        r = requests.post(
            f"{QA_AGENT_URL}{path}",
            json=body or {},
            headers=_headers(),
            timeout=timeout,
        )
        if r.status_code >= 400:
            detail = r.text
            try:
                payload = r.json()
                if isinstance(payload, dict) and "detail" in payload:
                    detail = payload["detail"]
            except Exception:
                pass
            st.error(f"API error ({r.status_code}): {detail}")
            return None
        return r.json()
    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None


# ─── Sidebar: Service status + navigation ────────────────────────────────────

with st.sidebar:
    st.subheader("Reviewer identity")
    st.session_state.reviewer_id = (
        st.text_input(
            "Audit ID (X-Reviewer-Id)",
            value=st.session_state.reviewer_id,
            help="Sent on every approve/write-back for Postgres reviewed_by",
        )
        or ""
    ).strip()

    st.header("Service Status")
    health = api_get("/health")
    if health:
        if health.get("status") == "ok":
            st.success("All systems operational")
        else:
            st.warning("Some services degraded")
        for k, v in health.get("checks", {}).items():
            icon = "✅" if str(v).startswith("ok") else "⚠️"
            st.text(f"{icon} {k}: {v}")
    else:
        st.error("Cannot reach qa-agent")

    st.divider()
    st.header("Navigation")
    page = st.radio(
        "Go to",
        ["Review Decisions", "Run History", "Write-back"],
        label_visibility="collapsed",
    )


# ─── Page: Review Decisions ──────────────────────────────────────────────────

PAGE_SIZE_DECISIONS = 50
BULK_DECISION_CHUNK = 500


def _fetch_all_decisions_flat(run_id: str) -> list[dict]:
    """Load every decision row for bulk actions (paginated API loop)."""
    out: list[dict] = []
    page = 1
    fetch_size = 500
    while True:
        d = api_get(
            f"/analyze/decisions/{run_id}?page={page}&page_size={fetch_size}&paginate=true"
        )
        if not d:
            break
        total = int(d.get("total", 0))
        for lst in d.get("decisions", {}).values():
            out.extend(lst)
        if page * fetch_size >= total or total == 0:
            break
        page += 1
    return out


def page_review():
    st.title("Review Agent Decisions")

    # Run ID input
    run_id = st.text_input(
        "Analysis Run ID",
        placeholder="Enter a run_id from POST /analyze/prd",
        help="The UUID returned when you triggered an analysis run",
    )

    if not run_id:
        st.info("Enter a run ID above to load decisions for review.")
        return

    page = st.number_input(
        "Results page",
        min_value=1,
        max_value=50_000,
        value=1,
        step=1,
        key=f"dec_page_{run_id}",
        help=f"Paginated loads ({PAGE_SIZE_DECISIONS} decisions per page)",
    )
    data = api_get(
        f"/analyze/decisions/{run_id}?page={int(page)}&page_size={PAGE_SIZE_DECISIONS}&paginate=true"
    )
    if not data:
        return

    total = data.get("total", 0)
    total_pages = max(1, (total + PAGE_SIZE_DECISIONS - 1) // PAGE_SIZE_DECISIONS)
    if int(page) > total_pages and total > 0:
        st.warning(f"Page {int(page)} is past the last page ({total_pages}). Lower the page number.")
    summary = data.get("summary", {})
    decisions = data.get("decisions", {})

    st.caption(
        f"Loaded page **{int(page)}** / **{total_pages}** — **{total}** decisions total "
        f"(`page_size={PAGE_SIZE_DECISIONS}`)"
    )

    if total == 0:
        st.warning("No decisions found for this run. The analysis may still be running.")
        # Show run status
        status = api_get(f"/analyze/status/{run_id}")
        if status:
            st.json(status)
        return

    # ── Summary stats ──
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("KEEP", summary.get("keep", 0))
    col2.metric("UPDATE", summary.get("update", 0))
    col3.metric("DEPRECATE", summary.get("deprecate", 0))
    col4.metric("CREATE", summary.get("create", 0))
    col5.metric("QUESTION", summary.get("question", 0))

    # ── Review progress ──
    all_decisions = []
    for action_list in decisions.values():
        all_decisions.extend(action_list)

    # Progress metrics must be run-level, not page-level.
    all_for_metrics = _fetch_all_decisions_flat(run_id)
    reviewed_count = sum(1 for d in all_for_metrics if d.get("reviewed"))
    approved_count = sum(1 for d in all_for_metrics if d.get("approved") is True)
    rejected_count = sum(1 for d in all_for_metrics if d.get("approved") is False)
    pending_count = max(0, total - reviewed_count)

    st.progress(reviewed_count / total if total > 0 else 0)
    st.caption(
        f"Progress: {reviewed_count}/{total} reviewed "
        f"({approved_count} approved, {rejected_count} rejected, {pending_count} pending)"
    )

    # ── Bulk actions (full run — not limited to the visible page) ──
    bcol1, bcol2, bcol3 = st.columns(3)
    with bcol1:
        if st.button("Approve All Pending", type="primary", use_container_width=True):
            _bulk_review(_fetch_all_decisions_flat(run_id), approved=True)
            st.rerun()
    with bcol2:
        if st.button("Reject All Pending", type="secondary", use_container_width=True):
            _bulk_review(_fetch_all_decisions_flat(run_id), approved=False)
            st.rerun()
    with bcol3:
        if st.button("Approve All KEEPs", use_container_width=True):
            keeps = [d for d in _fetch_all_decisions_flat(run_id) if d.get("action") == "keep"]
            _bulk_review(keeps, approved=True)
            st.rerun()

    st.divider()

    # ── Decision cards by action ──
    action_order = ["update", "create", "deprecate", "question", "keep"]
    action_colors = {
        "keep": "green", "update": "orange", "deprecate": "red",
        "create": "blue", "question": "violet",
    }

    for action in action_order:
        action_decisions = decisions.get(action, [])
        if not action_decisions:
            continue

        st.subheader(f"{action.upper()} ({len(action_decisions)})")

        for d in action_decisions:
            _render_decision_card(d, action_colors.get(action, "gray"))


def _render_decision_card(d: dict, color: str):
    """Render a single decision card with approve/reject controls."""
    decision_id = d["id"]
    jira_key = d.get("jira_key") or "(new test)"
    action = d.get("action", "?").upper()
    reason = d.get("reason", "No reason provided")
    prd_section = d.get("prd_section") or "N/A"
    reviewed = d.get("reviewed", False)
    approved = d.get("approved")
    content = d.get("updated_content")
    reviewer_note = d.get("reviewer_note") or ""

    # Status badge
    if approved is True:
        badge = "✅ Approved"
    elif approved is False:
        badge = "❌ Rejected"
    else:
        badge = "⏳ Pending"

    with st.expander(f":{color}[{action}] **{jira_key}** — {badge}", expanded=not reviewed):
        st.markdown(f"**Reason:** {reason}")
        st.caption(f"PRD Section: {prd_section}")

        # Show updated content for UPDATE/CREATE
        if content:
            st.markdown("**Proposed Changes:**")
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    pass
            if isinstance(content, dict):
                steps = content.get("steps")
                for k, v in content.items():
                    if k == "steps":
                        continue  # rendered below as a table, not as a Python repr
                    st.markdown(f"- **{k}:** {v}")

                # A step list is the only payload here that destroys data. Xray replaces the
                # whole array rather than merging, so the reviewer has to see exactly what
                # the test will be reduced to before approving.
                if isinstance(steps, list) and steps:
                    st.warning(
                        f"Approving **replaces all steps** on {jira_key} with the "
                        f"{len(steps)} below. Xray does not merge step lists, and this "
                        f"cannot be undone."
                    )
                    st.table([
                        {
                            "#": i,
                            "Action": (s or {}).get("action", "") if isinstance(s, dict) else str(s),
                            "Data": (s or {}).get("data", "") if isinstance(s, dict) else "",
                            "Expected Result": (
                                (s or {}).get("expectedResult") or (s or {}).get("result") or ""
                            ) if isinstance(s, dict) else "",
                        }
                        for i, s in enumerate(steps, start=1)
                    ])
            else:
                st.code(str(content))

        if reviewer_note:
            st.info(f"Reviewer note: {reviewer_note}")

        # Action buttons (only for pending decisions)
        if not reviewed:
            note = st.text_input(
                "Reviewer note (optional)",
                key=f"note_{decision_id}",
                placeholder="Add context for why you're approving/rejecting",
            )

            c1, c2 = st.columns(2)
            with c1:
                if st.button("Approve", key=f"approve_{decision_id}", type="primary", use_container_width=True):
                    api_post("/review/decision", {
                        "decision_id": decision_id,
                        "approved": True,
                        "reviewer_note": note,
                    })
                    st.rerun()
            with c2:
                if st.button("Reject", key=f"reject_{decision_id}", type="secondary", use_container_width=True):
                    api_post("/review/decision", {
                        "decision_id": decision_id,
                        "approved": False,
                        "reviewer_note": note,
                    })
                    st.rerun()


def _bulk_review(decisions: list, approved: bool):
    """Approve or reject all pending decisions."""
    pending = [d for d in decisions if not d.get("reviewed")]
    if not pending:
        st.info("No pending decisions to review.")
        return

    note = f"Bulk {'approved' if approved else 'rejected'}"
    updated_sum = 0
    for i in range(0, len(pending), BULK_DECISION_CHUNK):
        chunk = pending[i : i + BULK_DECISION_CHUNK]
        result = api_post(
            "/review/decisions/bulk",
            {
                "decisions": [{"decision_id": d["id"], "approved": approved} for d in chunk],
                "reviewer_note": note,
            },
            timeout=120,
        )
        if result is None:
            return
        updated_sum += int(result.get("updated", 0))
    st.success(
        f"{'Approved' if approved else 'Rejected'} {updated_sum} decisions "
        f"(requested {len(pending)}, in chunks of {BULK_DECISION_CHUNK})"
    )


# ─── Page: Run History ───────────────────────────────────────────────────────

def page_run_history():
    st.title("Pipeline Run History")
    st.caption("Recent sync, ingest, and analysis runs (GET /runs).")

    limit = st.slider("Rows", min_value=5, max_value=100, value=25)
    q_type = st.text_input("Filter type (optional)", placeholder="e.g. analysis, prd_ingest, test_sync")
    q_status = st.selectbox("Status filter", ("", "running", "completed", "failed"))

    params = f"?limit={limit}"
    if q_type.strip():
        params += f"&type={q_type.strip()}"
    if q_status:
        params += f"&status={q_status}"

    listed = api_get(f"/runs{params}")
    if listed and listed.get("runs"):
        st.dataframe(listed["runs"], use_container_width=True)
    elif listed is not None:
        st.info("No runs returned.")

    st.divider()
    st.subheader("Look up by run_id")
    run_id = st.text_input("Run ID", placeholder="uuid", key="run_lookup")
    if run_id:
        for endpoint in ["/sync/status/", "/ingest/status/", "/analyze/status/"]:
            data = api_get(f"{endpoint}{run_id}", quiet_404=True)
            if data and "error" not in str(data.get("detail", "")):
                st.json(data)
                return
        st.warning("Run not found on status endpoints.")


# ─── Page: Write-back ────────────────────────────────────────────────────────

def page_writeback():
    st.title("Write-back to Xray")
    st.markdown(
        "Push approved decisions back to Xray/Jira. "
        "This will **update tests**, **create new tests**, and **deprecate old tests** "
        "based on the approved decisions."
    )

    run_id = st.text_input(
        "Analysis run ID",
        help=(
            "Required unless the API has QA_WRITEBACK_ALLOW_GLOBAL=1. "
            "Otherwise the server rejects empty run_id with HTTP 400."
        ),
    )
    project_key = st.text_input(
        "Project Key (required for CREATE actions)",
        value=os.environ.get("XRAY_PROJECT_KEY", ""),
    )

    if st.button("Execute Write-back", type="primary"):
        if not project_key:
            st.warning("Project key is required for creating new tests.")

        with st.spinner("Writing back decisions to Xray (may take several minutes)..."):
            result = api_post(
                "/writeback/execute",
                {
                    "run_id": run_id.strip() or None,
                    "project_key": project_key,
                },
                timeout=300,
            )

        if result:
            status = result.get("status", "unknown")
            if status == "completed":
                st.success("Write-back completed successfully!")
            elif status == "completed_with_errors":
                st.warning("Write-back completed with some errors.")
            else:
                st.info(f"Status: {status}")

            # Show results
            col1, col2, col3, col4 = st.columns(4)
            wb = result.get("written_back", {})
            col1.metric("Updated", wb.get("update", 0))
            col2.metric("Created", wb.get("create", 0))
            col3.metric("Deprecated", wb.get("deprecate", 0))
            col4.metric("Kept", wb.get("keep", 0))

            errors = result.get("errors", [])
            if errors:
                st.subheader("Errors")
                for e in errors:
                    st.error(f"Decision {e.get('decision_id')}: {e.get('error')}")

            st.json(result)


# ─── Router ──────────────────────────────────────────────────────────────────

if page == "Review Decisions":
    page_review()
elif page == "Run History":
    page_run_history()
elif page == "Write-back":
    page_writeback()
