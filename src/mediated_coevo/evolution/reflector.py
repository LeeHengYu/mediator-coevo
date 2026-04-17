"""Reflector — text-level policy gradient via contrastive reflection.

Uses outcome-tagged history to build contrastive prompts and
calls the appropriate LLM to rewrite the agent's meta-skill.

Reads structured ``HistoryEntry.payload`` slots (``MediatorSignal`` /
``PlannerSignal``) rather than chopping free-text blobs — the Planner's
reasoning is preserved in full, and long mediator reports have already
been compacted upstream by ``evolution/compactor.py``.
"""

from __future__ import annotations

import difflib
import logging
import textwrap
from collections.abc import Callable
from typing import TYPE_CHECKING

from mediated_coevo.models.history_signals import MediatorSignal, PlannerSignal
from mediated_coevo.stores.history_store import AgentRole

if TYPE_CHECKING:
    from mediated_coevo.llm.client import LLMClient
    from mediated_coevo.stores.history_store import HistoryEntry, HistoryStore
    from mediated_coevo.stores.skill_store import SkillStore

logger = logging.getLogger(__name__)


_NO_CHANGE_SENTINEL = "NO_CHANGE"
_SIMILARITY_THRESHOLD = 0.95


class Reflector:
    """Builds contrastive reflection prompts and updates skills."""

    def __init__(
        self,
        history_store: HistoryStore,
        skill_store: SkillStore,
        similarity_threshold: float = _SIMILARITY_THRESHOLD,
    ) -> None:
        self._history_store = history_store
        self._skill_store = skill_store
        self._similarity_threshold = similarity_threshold

    async def reflect(
        self,
        agent_role: AgentRole,
        llm_client: LLMClient,
        max_pairs: int = 5,
    ) -> str | None:
        """Reflect on past actions for the given role, update the skill file.

        Returns the new skill content, or None if no update was made.
        """
        pairs = self._history_store.contrastive_pairs(agent_role, max_pairs=max_pairs)
        if not pairs:
            logger.info("No contrastive pairs for %s; skipping reflection.", agent_role)
            return None

        logger.info("Reflector found %d contrastive pairs for %s.", len(pairs), agent_role)

        current_skill = self._skill_store.read_skill(agent_role) or ""

        if agent_role == AgentRole.MEDIATOR:
            messages = self._build_mediator_prompt(current_skill, pairs)
        else:
            messages = self._build_planner_prompt(current_skill, pairs)

        try:
            response = await llm_client.complete(
                messages=messages,
                temperature=0.4,
                max_tokens=4096,
            )
            raw_content = response["content"].strip()

            if raw_content == _NO_CHANGE_SENTINEL:
                logger.info("%s reflection: LLM explicitly signalled no change.", agent_role)
                return None

            new_content = _parse_skill_content(raw_content)

            if not new_content:
                logger.info("%s reflection produced no parseable content.", agent_role)
                return None

            if current_skill and _is_semantically_similar(
                current_skill, new_content, self._similarity_threshold
            ):
                logger.info(
                    "%s reflection: new content too similar to current (threshold=%.2f); skipping.",
                    agent_role,
                    self._similarity_threshold,
                )
                return None

            self._skill_store.write_skill(agent_role, new_content)
            logger.info("%s skill updated via reflection (%d chars).", agent_role, len(new_content))
            return new_content
        except Exception as e:
            logger.error("Failed to reflect for %s: %s", agent_role, e)
            return None

    # ── Prompt Builders ──

    @staticmethod
    def _build_contrastive_prompt(
        *,
        system_text: str,
        current_skill_heading: str,
        current_skill: str,
        evidence_intro: str,
        instructions: str,
        pairs: list[tuple[HistoryEntry, HistoryEntry]],
        formatter: Callable[[HistoryEntry], str],
    ) -> list[dict[str, str]]:
        """Shared builder for contrastive reflection prompts."""
        contrastive_parts: list[str] = []
        for i, (worse, better) in enumerate(pairs, 1):
            contrastive_parts.append(
                f"### Pair {i} — task `{worse.metadata.get('task_id', '?')}`\n"
                f"**Worse outcome** (reward: {worse.reward:.2f}):\n"
                f"{formatter(worse)}\n\n"
                f"**Better outcome** (reward: {better.reward:.2f}):\n"
                f"{formatter(better)}"
            )

        return [
            {"role": "system", "content": system_text},
            {
                "role": "user",
                "content": (
                    f"## {current_skill_heading}\n\n"
                    f"{current_skill}\n\n"
                    "## Contrastive Evidence\n\n"
                    f"{evidence_intro}\n\n"
                    + "\n\n".join(contrastive_parts)
                    + f"\n\n## Instructions\n\n{instructions}"
                ),
            },
        ]

    @staticmethod
    def _build_mediator_prompt(
        current_skill: str,
        pairs: list[tuple[HistoryEntry, HistoryEntry]],
    ) -> list[dict[str, str]]:
        return Reflector._build_contrastive_prompt(
            system_text=(
                "You are reflecting on your performance as a Mediator agent. "
                "Your coordination-protocol skill defines HOW you curate "
                "execution feedback for the Planner. You will see contrastive "
                "pairs: cases where your reporting led to better vs. worse "
                "downstream outcomes. Use these to revise your protocol.\n\n"
                "If you believe the current protocol already captures the "
                "lessons from the evidence and no meaningful change is needed, "
                "output ONLY the word NO_CHANGE (nothing else).\n\n"
                "Otherwise, output ONLY the updated coordination protocol as "
                "Markdown, enclosed in ```markdown ... ``` fences. Do not "
                "include explanation outside the fences."
            ),
            current_skill_heading="Current Coordination Protocol",
            current_skill=current_skill,
            evidence_intro=(
                "Below are pairs of your past reports. In each pair, one report "
                "led to a WORSE downstream reward and the other to a BETTER one. "
                "Each entry shows the mediator's headline, decision, abstraction "
                "level, and a diagnostic excerpt of the report."
            ),
            instructions=(
                "Revise your coordination protocol based on the patterns above. "
                "Keep the same JSON output format. Focus on:\n"
                "1. What reporting style led to better outcomes?\n"
                "2. When should you withhold vs. expose?\n"
                "3. What abstraction level works best?\n"
                "Make minimal, targeted changes. Do not rewrite from scratch."
            ),
            pairs=pairs,
            formatter=_format_mediator_entry,
        )

    @staticmethod
    def _build_planner_prompt(
        current_skill: str,
        pairs: list[tuple[HistoryEntry, HistoryEntry]],
    ) -> list[dict[str, str]]:
        return Reflector._build_contrastive_prompt(
            system_text=(
                "You are reflecting on your performance as a Planner agent. "
                "Your skill-refiner skill defines HOW you decide to edit the "
                "Executor's skills. You will see contrastive pairs: skill edits "
                "that led to better vs. worse outcomes. Use these to revise "
                "your editing strategy.\n\n"
                "If you believe the current guidelines already capture the "
                "lessons from the evidence and no meaningful change is needed, "
                "output ONLY the word NO_CHANGE (nothing else).\n\n"
                "Otherwise, output ONLY the updated skill-refiner as "
                "Markdown, enclosed in ```markdown ... ``` fences. Do not "
                "include explanation outside the fences."
            ),
            current_skill_heading="Current Skill-Refiner Guidelines",
            current_skill=current_skill,
            evidence_intro=(
                "Below are pairs of your past skill edits. In each pair, one "
                "edit led to a WORSE downstream reward and the other to a "
                "BETTER one. Each entry shows your full reasoning, the diff "
                "size, and a head+tail excerpt of the diff itself."
            ),
            instructions=(
                "Revise your skill-refiner guidelines based on the patterns "
                "above. Focus on:\n"
                "1. What kinds of edits led to better outcomes?\n"
                "2. What edit patterns should you avoid?\n"
                "3. How should you interpret the Mediator's feedback?\n"
                "Make minimal, targeted changes. Do not rewrite from scratch."
            ),
            pairs=pairs,
            formatter=_format_planner_entry,
        )


def _is_semantically_similar(old: str, new: str, threshold: float) -> bool:
    """Check whether *old* and *new* are similar enough to skip the update.

    Uses ``SequenceMatcher`` on whitespace-normalised text so that
    cosmetic reformatting (extra blank lines, indentation tweaks,
    trailing spaces) does not count as a real change.
    """
    norm_old = " ".join(old.split())
    norm_new = " ".join(new.split())
    ratio = difflib.SequenceMatcher(None, norm_old, norm_new).ratio()
    logger.debug("Skill similarity ratio: %.4f (threshold %.4f)", ratio, threshold)
    return ratio >= threshold


def _format_mediator_entry(entry: "HistoryEntry") -> str:
    """Render a mediator HistoryEntry as typed-slot bullets for the prompt."""
    p = entry.payload
    if not isinstance(p, MediatorSignal):
        return "- (no mediator signal)"

    lines: list[str] = []
    if p.abstraction_level:
        lines.append(f"- Abstraction: {p.abstraction_level}")
    if p.withheld:
        lines.append("- Decision: WITHHELD (exposed nothing to the Planner)")
    if p.headline:
        lines.append(f"- Headline: {p.headline}")
    if p.evidence:
        sample_note = (
            "" if p.raw_length <= len(p.evidence) else f" (sampled from {p.raw_length} chars)"
        )
        lines.append(f"- Evidence{sample_note}:")
        lines.append(textwrap.indent(p.evidence.rstrip(), "    "))
    if p.mediator_reasoning:
        lines.append(f"- Mediator reasoning: {p.mediator_reasoning}")
    return "\n".join(lines) if lines else "- (empty mediator signal)"


def _format_planner_entry(entry: "HistoryEntry") -> str:
    """Render a planner HistoryEntry as typed-slot bullets for the prompt."""
    p = entry.payload
    if not isinstance(p, PlannerSignal):
        return "- (no planner signal)"

    lines: list[str] = []
    if p.reasoning:
        lines.append(f"- Reasoning: {p.reasoning}")
    diff_line = f"- Diff: +{p.lines_added}/-{p.lines_removed}"
    lines.append(diff_line)
    if p.diff_excerpt.strip():
        lines.append("- Diff excerpt:")
        lines.append("```diff")
        lines.append(p.diff_excerpt.rstrip())
        lines.append("```")
    return "\n".join(lines)


def _parse_skill_content(response_text: str) -> str | None:
    """Extract Markdown content from ```markdown ... ``` fences.

    Finds the first opening fence and its matching close, so inner
    code fences (e.g., ```json examples) are not confused with the
    outer wrapper.
    """
    text = response_text.strip()

    # Look for ```markdown opening fence
    for marker in ("```markdown", "```md"):
        if marker in text:
            start = text.index(marker) + len(marker)
            # Skip to next line
            nl = text.find("\n", start)
            if nl == -1:
                break
            start = nl + 1
            # Find matching close — scan for ``` at start of a line
            end = _find_closing_fence(text, start)
            if end is not None:
                return text[start:end].strip()

    # No markdown fence — try generic ``` fence
    if text.startswith("```"):
        nl = text.find("\n")
        if nl != -1:
            start = nl + 1
            end = _find_closing_fence(text, start)
            if end is not None:
                return text[start:end].strip()

    # No fences — accept raw text if it looks like Markdown
    if text.startswith("#"):
        return text

    return None


def _find_closing_fence(text: str, start: int) -> int | None:
    """Find the position of the closing ``` that matches an opening fence.

    Skips over inner fenced blocks (``` that open and close within)
    using a simple depth counter.
    """
    pos = start
    depth = 0
    while pos < len(text):
        line_end = text.find("\n", pos)
        if line_end == -1:
            line_end = len(text)
        line = text[pos:line_end].strip()

        if line.startswith("```"):
            if depth > 0:
                depth -= 1
            elif line == "```":
                return pos
            else:
                depth += 1

        pos = line_end + 1

    return None

