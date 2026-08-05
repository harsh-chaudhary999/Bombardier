"""Formal termination status for the ReAct analysis loop (distinct from run sync_runs.status)."""
from __future__ import annotations

from enum import Enum


class LoopStatus(str, Enum):
    """How the tool-calling loop ended."""

    COMPLETED = "completed"  # Model returned without further tool calls
    MAX_TURNS_REACHED = "max_turns_reached"
