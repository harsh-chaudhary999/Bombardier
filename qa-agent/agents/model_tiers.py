"""
Model tiering: cheap model for volume, strong model for judgement.

The capabilities this pipeline needs span a wide difficulty range. Intent routing and
per-section keep/update triage run hundreds of times and a small local model handles them
fine. Cross-document impact analysis and contradiction detection are the opposite: low
volume, high stakes, and the tasks where a small model produces confident nonsense.

Paying frontier prices for the first group is waste; using a 12B for the second is
negligence. So callers ask for a TIER rather than naming a model:

    provider, model = resolve_tier("reasoning", req.provider, req.model)

Tiers
-----
fast       High-volume work. Defaults to whatever the request/env asked for.
reasoning  Hard judgement. Falls back to `fast` when unconfigured, so nothing changes
           until you deliberately configure a second model.

Env
---
QA_REASONING_PROVIDER   anthropic | openai | azure_openai | ollama
QA_REASONING_MODEL      e.g. claude-opus-5, or qwen3:30b-a3b to stay local

Both must be set for the reasoning tier to differ; setting only one is a config error and
is reported rather than half-applied.
"""
from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

FAST = "fast"
REASONING = "reasoning"
TIERS = (FAST, REASONING)

_warned: set[str] = set()


def _warn_once(key: str, msg: str, *args) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.warning(msg, *args)


def reset_warnings() -> None:
    """Test hook — lets the same warning be asserted across cases."""
    _warned.clear()


def resolve_tier(tier: str, provider: str, model: str) -> tuple[str, str]:
    """
    Resolve a tier to a concrete (provider, model).

    `provider`/`model` are the caller's request-level choice and serve as the `fast` tier
    and as the fallback for `reasoning`. Unknown tier names fall back to fast with a
    warning rather than raising: a typo in an internal call site should not fail a run
    that would otherwise have produced useful output.
    """
    if tier not in TIERS:
        _warn_once(f"tier:{tier}", "Unknown model tier %r — using %r", tier, FAST)
        return provider, model

    if tier == FAST:
        return provider, model

    rp = os.environ.get("QA_REASONING_PROVIDER", "").strip()
    rm = os.environ.get("QA_REASONING_MODEL", "").strip()

    if rp and rm:
        return rp, rm

    if rp or rm:
        # Half-configured. Silently ignoring it would mean the operator believes hard
        # reasoning is escalated when it is not — the failure would show up as poor
        # analysis quality, which is nearly impossible to trace back to this.
        _warn_once(
            "half-config",
            "QA_REASONING_%s is set but QA_REASONING_%s is not — the reasoning tier is "
            "NOT active and falls back to %s/%s. Set both or neither.",
            "PROVIDER" if rp else "MODEL",
            "MODEL" if rp else "PROVIDER",
            provider, model,
        )
    return provider, model


def reasoning_tier_configured() -> bool:
    """True when a distinct reasoning model is actually in effect."""
    return bool(
        os.environ.get("QA_REASONING_PROVIDER", "").strip()
        and os.environ.get("QA_REASONING_MODEL", "").strip()
    )


def describe(provider: str, model: str) -> dict[str, str | bool]:
    """Tier assignment for run metadata and diagnostics."""
    rp, rm = resolve_tier(REASONING, provider, model)
    return {
        "fast": f"{provider}/{model}",
        "reasoning": f"{rp}/{rm}",
        "tiered": reasoning_tier_configured() and (rp, rm) != (provider, model),
    }
