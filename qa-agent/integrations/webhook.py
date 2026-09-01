"""
Inbound webhook plumbing: signature verification, event filtering, debouncing.

Kept separate from main.py so the security-critical parts are testable without the web
stack, and so the rules are in one place rather than inline in a request handler.

The endpoint this serves triggers ingestion and analysis, which cost money. Signature
verification is therefore the whole security boundary, and the debounce is what stops a
normal editing session — Confluence fires one event per save — from starting a run per
keystroke-save.
"""
import hashlib
import hmac
import time
from collections import OrderedDict

#: Page events worth reacting to. Everything else (views, comments, blog posts, space
#: settings) is acknowledged and ignored — reacting to them would spend tokens on
#: activity that cannot have changed a requirement.
TRIGGER_EVENTS = frozenset({"page_updated", "page_created", "page_restored"})


def signature_ok(secret: str, body: bytes, header: str | None) -> bool:
    """
    Constant-time HMAC-SHA256 check of the raw request body.

    `body` must be the bytes exactly as received. Re-serialising the parsed JSON would
    change key order or spacing and so change the digest, failing every legitimate
    request. An empty secret always fails: the caller is expected to refuse the request
    before reaching here, and returning True for "no secret" would turn a
    misconfiguration into an open endpoint.
    """
    if not secret or not header:
        return False
    expected = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    # Senders differ on whether they prefix the algorithm ("sha256=<hex>" vs "<hex>").
    received = header.split("=", 1)[1] if "=" in header else header
    return hmac.compare_digest(expected, received.strip())


def should_trigger(event: str | None) -> bool:
    return (event or "").strip() in TRIGGER_EVENTS


def event_name(payload: dict) -> str:
    """Event key, tolerating the two spellings Confluence variants use."""
    return str(payload.get("event") or payload.get("eventType") or "").strip()


def page_id(payload: dict) -> str | None:
    """
    Numeric page id from a webhook payload, or None.

    Required to be all digits: the id is interpolated into a source_id and used to fetch
    a page, so anything else is either a payload we do not understand or an injection
    attempt, and both should be ignored rather than acted on.
    """
    page = payload.get("page") or payload.get("content") or {}
    if not isinstance(page, dict):
        return None
    value = str(page.get("id") or "").strip()
    return value if value.isdigit() else None


class Debounce:
    """
    Per-key cooldown with a bounded LRU of recent keys.

    Bounded because the key space is "every page anyone edits", which is unbounded over
    the life of the process; an unbounded dict here is a slow memory leak.
    """

    def __init__(self, window_sec: int, max_keys: int = 2048) -> None:
        self.window_sec = window_sec
        self.max_keys = max_keys
        self._seen: OrderedDict[str, float] = OrderedDict()

    def check(self, key: str, now: float | None = None) -> float:
        """
        Seconds still to wait before `key` is eligible again; 0.0 if it is eligible now.

        Records the acceptance as a side effect, so a caller that is told 0.0 has
        claimed the slot.
        """
        if self.window_sec <= 0:
            return 0.0
        current = time.monotonic() if now is None else now
        last = self._seen.get(key)
        if last is not None and (current - last) < self.window_sec:
            return round(self.window_sec - (current - last), 1)
        self._seen[key] = current
        self._seen.move_to_end(key)
        while len(self._seen) > self.max_keys:
            self._seen.popitem(last=False)
        return 0.0

    def clear(self) -> None:
        self._seen.clear()

    def __len__(self) -> int:
        return len(self._seen)
