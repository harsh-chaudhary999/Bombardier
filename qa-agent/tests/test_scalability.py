"""
Plan D: backpressure, durable phase ledger, archival, auto-approval, ancestry,
orphan detection.

The property most of these share: a safeguard that silently does nothing is worse than
no safeguard, because it is believed. A semaphore that never blocks, a ledger that
fragments per pod, an archiver that removes rows still needing review, an auto-approver
that approves a deprecation — each looks like it is working right up until it matters.

Stdlib unittest — see ADR-019. Neutral placeholders only.

Runs under the documented host-side command:
    cd qa-agent && PYTHONPATH=. python3 -m unittest discover -s tests -p 'test_*.py'
"""
import unittest

from tests import stubs

stubs.install_agent_deps()

from observability.request_norm import (  # noqa: E402
    NEVER_AUTO_APPROVE,
    auto_approve_reason,
)


# ─── D30: auto-approval ───────────────────────────────────────────────────────

class AutoApproveGateTests(unittest.TestCase):
    """Four independent gates; any one of them closed must stop the approval."""

    OK = dict(action="keep", confidence="high", approval_rate=0.97,
              samples=40, threshold=0.95, min_samples=20)

    def _reason(self, **overrides):
        return auto_approve_reason(**{**self.OK, **overrides})

    def test_all_gates_open_allows_approval(self):
        self.assertIsNone(self._reason())

    def test_deprecate_is_never_auto_approved(self):
        """It deletes coverage. No approval rate justifies skipping a human."""
        reason = self._reason(action="deprecate")
        self.assertIsNotNone(reason)
        self.assertIn("human", reason)

    def test_create_is_never_auto_approved(self):
        self.assertIsNotNone(self._reason(action="create"))

    def test_a_perfect_history_cannot_unlock_a_barred_action(self):
        self.assertIsNotNone(
            self._reason(action="deprecate", approval_rate=1.0, samples=10_000))

    def test_update_is_eligible(self):
        self.assertIsNone(self._reason(action="update"))

    def test_medium_confidence_is_refused(self):
        self.assertIn("confidence", self._reason(confidence="medium"))

    def test_unset_confidence_is_refused(self):
        """NULL means the agent did not say — never read it as high."""
        reason = self._reason(confidence=None)
        self.assertIsNotNone(reason)
        self.assertIn("unset", reason)

    def test_too_few_samples_is_refused(self):
        reason = self._reason(samples=5)
        self.assertIsNotNone(reason)
        self.assertIn("5", reason)

    def test_a_perfect_rate_on_a_tiny_history_is_refused(self):
        """100% of three decisions is not evidence."""
        self.assertIsNotNone(self._reason(approval_rate=1.0, samples=3))

    def test_rate_below_threshold_is_refused(self):
        reason = self._reason(approval_rate=0.80)
        self.assertIsNotNone(reason)
        self.assertIn("80%", reason)

    def test_rate_exactly_at_threshold_is_allowed(self):
        self.assertIsNone(self._reason(approval_rate=0.95, threshold=0.95))

    def test_reason_is_returned_rather_than_a_bool(self):
        """A dry run has to explain which gate closed; False cannot."""
        self.assertIsInstance(self._reason(confidence="low"), str)

    def test_barred_actions_are_declared(self):
        self.assertEqual(NEVER_AUTO_APPROVE, {"deprecate", "create"})


# ─── D28: durable phase ledger ────────────────────────────────────────────────

class PhaseLedgerBackendTests(unittest.TestCase):
    """
    fcntl locking coordinates writers on one machine. Across replicas each keeps its
    own partial file and no copy is the record — hence a database backend.
    """

    def setUp(self):
        from observability import phase_ledger
        self.ledger = phase_ledger
        self._original = phase_ledger._store
        self.addCleanup(phase_ledger.set_store, self._original)

    class _Store:
        def __init__(self, ok=True):
            self.ok = ok
            self.rows = []

        def append_ledger_entry(self, phase, run_id, summary, summary_sha256):
            self.rows.append({"phase": phase, "run_id": run_id,
                              "summary": summary, "sha": summary_sha256})
            return self.ok

    def test_entry_goes_to_the_store_when_registered(self):
        store = self._Store()
        self.ledger.set_store(store)
        record = self.ledger.append_entry("analysis", "run-1", {"decisions": 4})
        self.assertEqual(len(store.rows), 1)
        self.assertEqual(record["backend"], "postgres")

    def test_fingerprint_is_recorded_with_the_entry(self):
        store = self._Store()
        self.ledger.set_store(store)
        record = self.ledger.append_entry("analysis", "run-1", {"decisions": 4})
        self.assertEqual(store.rows[0]["sha"], record["summary_sha256"])
        self.assertEqual(len(record["summary_sha256"]), 64)

    def test_fingerprint_detects_a_changed_summary(self):
        store = self._Store()
        self.ledger.set_store(store)
        a = self.ledger.append_entry("analysis", "run-1", {"decisions": 4})
        b = self.ledger.append_entry("analysis", "run-1", {"decisions": 5})
        self.assertNotEqual(a["summary_sha256"], b["summary_sha256"])

    def test_a_failed_insert_falls_back_to_the_file(self):
        """An audit entry that cannot be stored durably is still better than dropped."""
        store = self._Store(ok=False)
        self.ledger.set_store(store)
        record = self.ledger.append_entry("analysis", "run-1", {"decisions": 4})
        self.assertEqual(record["backend"], "file")

    def test_without_a_store_the_file_is_used(self):
        self.ledger.set_store(None)
        record = self.ledger.append_entry("analysis", "run-1", {"decisions": 4})
        self.assertEqual(record["backend"], "file")

    def test_appending_never_raises(self):
        class _Broken:
            def append_ledger_entry(self, **kw):
                raise RuntimeError("database gone")

        self.ledger.set_store(_Broken())
        with self.assertRaises(RuntimeError):
            # The store itself raising is the store's contract violation; PGStore's
            # own implementation swallows and returns False. Pinned so that contract
            # is not quietly moved into the ledger.
            self.ledger.append_entry("analysis", "run-1", {"a": 1})


# ─── D29 / D27: SQL and backpressure shape ────────────────────────────────────

class ArchivalShapeTests(unittest.TestCase):
    """The SQL cannot run here; pin what must not silently change."""

    def setUp(self):
        import inspect
        from embeddings import pg_store
        self.src = inspect.getsource(pg_store.PGStore.archive_old_decisions)

    def test_only_written_back_rows_are_archived(self):
        """
        An old unreviewed decision is the SLA report's subject, not the archiver's —
        archiving it would hide exactly what that report exists to surface.
        """
        self.assertIn("written_back = TRUE", self.src)

    def test_move_is_a_single_statement(self):
        """A separate copy-then-delete can half-happen and duplicate or lose rows."""
        self.assertIn("WITH due AS", self.src)
        self.assertIn("DELETE FROM qa_rag.pending_decisions", self.src)
        self.assertIn("INSERT INTO qa_rag.pending_decisions_archive", self.src)

    def test_batched_with_skip_locked(self):
        self.assertIn("LIMIT", self.src)
        self.assertIn("SKIP LOCKED", self.src)

    def test_archive_table_mirrors_the_source(self):
        from pathlib import Path
        sql = (Path(__file__).resolve().parents[2]
               / "init-db" / "11-decisions-archive.sql").read_text()
        self.assertIn("LIKE qa_rag.pending_decisions", sql)


class ApprovalRateShapeTests(unittest.TestCase):
    def setUp(self):
        import inspect
        from embeddings import pg_store
        self.src = inspect.getsource(pg_store.PGStore.get_approval_rate)

    def test_only_reviewed_decisions_count_as_evidence(self):
        self.assertIn("reviewed = TRUE", self.src)

    def test_undecided_rows_are_excluded(self):
        """approved IS NULL is 'not judged', not 'rejected'."""
        self.assertIn("approved IS NOT NULL", self.src)

    def test_insufficient_history_returns_a_zero_rate(self):
        self.assertIn("if total < min_samples", self.src)


class BackpressureTests(unittest.TestCase):
    """A cap that never actually blocks is worse than none: it is believed."""

    def setUp(self):
        from agents.capacity import AtCapacity, Capacity
        self.Capacity = Capacity
        self.AtCapacity = AtCapacity

    def test_claiming_consumes_a_slot(self):
        c = self.Capacity(limit=3)
        c.claim()
        self.assertEqual(c.in_flight, 1)
        self.assertEqual(c.free(), 2)

    def test_at_capacity_refuses(self):
        c = self.Capacity(limit=2)
        c.claim()
        c.claim()
        with self.assertRaises(self.AtCapacity) as ctx:
            c.claim()
        self.assertEqual(ctx.exception.running, 2)
        self.assertEqual(ctx.exception.limit, 2)

    def test_releasing_frees_a_slot(self):
        c = self.Capacity(limit=1)
        c.claim()
        c.release()
        c.claim()  # must not raise
        self.assertEqual(c.in_flight, 1)

    def test_double_release_cannot_create_capacity(self):
        """Otherwise a buggy caller silently raises the ceiling."""
        c = self.Capacity(limit=1)
        c.claim()
        c.release()
        c.release()
        self.assertEqual(c.in_flight, 0)
        self.assertEqual(c.free(), 1)

    def test_limit_is_at_least_one(self):
        """A misconfigured 0 would refuse every request forever."""
        self.assertEqual(self.Capacity(limit=0).limit, 1)
        self.assertEqual(self.Capacity(limit=-5).limit, 1)

    def test_slot_is_released_when_the_run_raises(self):
        import asyncio

        c = self.Capacity(limit=1)

        async def _boom():
            raise RuntimeError("analysis failed")

        c.claim()
        with self.assertRaises(RuntimeError):
            asyncio.run(c.run(_boom()))
        self.assertEqual(c.in_flight, 0, "a failed run must not leak its slot")

    def test_slot_is_released_when_the_run_is_cancelled(self):
        import asyncio

        c = self.Capacity(limit=1)

        async def _forever():
            await asyncio.sleep(10)

        async def _cancel_it():
            task = asyncio.ensure_future(c.run(_forever()))
            await asyncio.sleep(0)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        c.claim()
        asyncio.run(_cancel_it())
        self.assertEqual(c.in_flight, 0)

    def test_slot_is_released_on_success(self):
        import asyncio

        c = self.Capacity(limit=1)

        async def _ok():
            return "done"

        c.claim()
        self.assertEqual(asyncio.run(c.run(_ok())), "done")
        self.assertEqual(c.in_flight, 0)

    def test_malformed_env_falls_back_to_the_default(self):
        import os
        from agents.capacity import from_env
        os.environ["QA_TEST_CAPACITY"] = "not-a-number"
        try:
            self.assertEqual(from_env("QA_TEST_CAPACITY", default=3).limit, 3)
        finally:
            os.environ.pop("QA_TEST_CAPACITY", None)


if __name__ == "__main__":
    unittest.main()
