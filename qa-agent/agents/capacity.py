"""
Concurrency cap for analysis runs.

Each concurrent analysis holds an LLM conversation, a share of the Postgres pool and CPU
for embedding. Ten at once do not finish ten times sooner — they contend, and in the
worst case exhaust memory.

Callers are refused rather than queued. These are background jobs polled by run_id, so a
caller told to retry can; a silent queue turns a 202 into an unbounded wait with nothing
to observe.

Kept out of main.py so it is testable without importing the whole web stack, and so the
HTTP status code stays an HTTP concern — this module raises a domain error and the
endpoint translates it.
"""
import logging
import os

logger = logging.getLogger(__name__)


class AtCapacity(RuntimeError):
    """Raised when no analysis slot is free. Carries what a caller needs to act."""

    def __init__(self, running: int, limit: int) -> None:
        super().__init__(f"{running}/{limit} analyses already running")
        self.running = running
        self.limit = limit


class Capacity:
    """
    A counter, not a semaphore.

    A semaphore queues waiters, which is the opposite of what is wanted here: the point
    is to refuse work, not to accept it and hide the wait. Nothing awaits between the
    check and the increment, so on a single event loop the pair cannot interleave.
    """

    def __init__(self, limit: int) -> None:
        self.limit = max(1, limit)
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    def free(self) -> int:
        return max(0, self.limit - self._in_flight)

    def claim(self) -> None:
        """Reserve a slot, or raise AtCapacity. Claim before starting the task."""
        if self._in_flight >= self.limit:
            raise AtCapacity(self._in_flight, self.limit)
        self._in_flight += 1

    def release(self) -> None:
        """Never drops below zero: a double release must not create phantom capacity."""
        if self._in_flight == 0:
            logger.warning("Capacity released more times than claimed — ignoring")
            return
        self._in_flight -= 1

    async def run(self, coro):
        """Await `coro`, releasing the slot however it ends: return, raise or cancel."""
        try:
            return await coro
        finally:
            self.release()


def from_env(var: str = "QA_MAX_CONCURRENT_ANALYSES", default: int = 3) -> Capacity:
    try:
        limit = int(os.environ.get(var, str(default)))
    except ValueError:
        logger.warning("Ignoring malformed %s — using %s", var, default)
        limit = default
    return Capacity(limit)
