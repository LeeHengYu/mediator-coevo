"""SkillAdvisor — batched LLM gate for Planner skill proposals.

Reviews a buffer of SkillProposals against the current executor skill
and decides whether to commit the proposed changes. Returns compact
feedback for the Planner if approved, None if rejected.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mediated_coevo.llm.client import LLMClient
    from mediated_coevo.models.skill import SkillProposal

logger = logging.getLogger(__name__)

_SYSTEM = """\
You are a Skill Advisor in a multi-agent co-evolution system.
Review a batch of proposed edits to the Executor's skill file.
Each proposal includes the Planner's reasoning, a diff, and a reward
(the task score from the iteration AFTER the proposal; "n/a" if not yet known).

Approve if the proposals show a consistent, well-reasoned direction.
Reject if proposals contradict each other, lack supporting evidence, or the
current skill already captures the proposed changes.

Respond with ONLY a JSON object (no prose, no fences):
  {"approve": true,  "feedback": "<2-4 sentence instruction for the Planner>"}
  {"approve": false, "feedback": ""}"""


class SkillAdvisor:
    """Batched LLM gate for Planner skill proposals."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def review(
        self,
        current_skill: str,
        proposals: list[SkillProposal],
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> str | None:
        """Review buffered proposals. Returns compact feedback or None."""
        if not proposals:
            return None

        from mediated_coevo.evolution.compactor import _diff_parts
        from mediated_coevo.utils import parse_json_object

        parts = [
            "## Current Executor Skill\n",
            current_skill or "(empty)",
            "\n## Buffered Proposals\n",
        ]
        for i, p in enumerate(proposals, 1):
            reward_str = f"{p.reward:.3f}" if p.reward is not None else "n/a"
            added, removed, excerpt = _diff_parts(p.old_content, p.new_content)
            parts.append(
                f"### Proposal {i} — iter={p.iteration} task={p.task_id} reward={reward_str}\n"
                f"**Reasoning**: {p.reasoning}\n"
                f"**Diff**: +{added}/-{removed} lines\n"
                f"```diff\n{excerpt}```\n"
            )
        parts.append('\nRespond with JSON only: {"approve": true/false, "feedback": "..."}')

        try:
            resp = await self._llm.complete(
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": "\n".join(parts)},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            parsed = parse_json_object(resp["content"])
            if parsed.get("approve"):
                fb = str(parsed.get("feedback", "")).strip()
                return fb or None
        except Exception as e:
            logger.error("SkillAdvisor review failed: %s", e)

        return None
