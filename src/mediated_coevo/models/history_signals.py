"""Structured payload types for HistoryEntry.

These replace the previous free-text `content` / `action_summary` fields
with typed slots that the Reflector can pull from with per-slot budgets,
instead of doing lossy character-based truncation on one blob.

Two variants, selected via a ``kind`` discriminator so pydantic can
round-trip them through the history JSONL file:

- ``MediatorSignal`` — what the Mediator exposed to the Planner.
- ``PlannerSignal``  — what the Planner did when editing a skill.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class MediatorSignal(BaseModel):
    """Structured form of a mediator feedback/report event.

    For condition 5 (``learned_mediator``) the decision/reasoning slots
    come from the underlying ``MediatorReport``. For conditions 2-4 only
    ``headline`` and ``evidence`` are populated; the rest stay empty.
    """

    kind: Literal["mediator"] = "mediator"

    headline: str = ""
    """Single-sentence observation — the top-line signal for the Reflector."""

    evidence: str = ""
    """Diagnostic excerpt. Either the full feedback verbatim (short
    reports) or an LLM-extracted key passage (long reports)."""

    abstraction_level: str = ""
    """One of 'trace' | 'reflection' | 'pattern' for learned_mediator;
    empty for other conditions."""

    withheld: bool = False
    """True iff the Mediator chose to expose nothing this iteration."""

    mediator_reasoning: str = ""
    """The Mediator's own 'why this report style' text — the signal that
    should drive protocol co-evolution. Empty for non-mediator conditions."""

    raw_length: int = 0
    """Original feedback length in characters. Lets the Reflector tell
    whether ``evidence`` is the full text or a compacted sample."""


class PlannerSignal(BaseModel):
    """Structured form of a planner skill-edit event.

    ``reasoning`` is preserved in full — it is the main driver for the
    Planner reflector and is already bounded (one LLM-generated
    paragraph). The diff is kept as a head+tail excerpt plus line stats.
    """

    kind: Literal["planner"] = "planner"

    reasoning: str = ""
    """Full planner reasoning — NOT truncated."""

    lines_added: int = 0
    lines_removed: int = 0

    diff_excerpt: str = ""
    """Head + tail of the unified diff with an explicit gap marker."""


HistorySignal = Annotated[
    Union[MediatorSignal, PlannerSignal],
    Field(discriminator="kind"),
]
