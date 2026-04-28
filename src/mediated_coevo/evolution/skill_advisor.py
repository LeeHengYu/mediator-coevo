"""SkillAdvisor — batched LLM gate for Planner skill proposals.

Reviews a buffer of SkillProposals against the current executor skill
and decides whether to commit the proposed changes. Returns compact
feedback for the Planner if approved, None if rejected.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mediated_coevo.config import BudgetsConfig
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


@dataclass(frozen=True)
class SkillAdvisorPrompt:
    """Render the advisor review prompt for a batch of skill proposals."""

    current_skill: str
    proposals: list[SkillProposal]
    model: str
    budgets: BudgetsConfig | None = None

    def render(self) -> tuple[str, int | None]:
        from mediated_coevo.token_budget import BudgetSection, pack_sections

        user_content = "\n".join([
            "## Current Executor Skill\n",
            self._current_skill_text(),
            "\n## Buffered Proposals\n",
            *self._proposal_blocks(),
            '\nRespond with JSON only: {"approve": true/false, "feedback": "..."}',
        ])
        if not self.budgets:
            return user_content, None

        prompt_budget = self.budgets.advisor_prompt_tokens
        return (
            pack_sections(
                self.model,
                [
                    BudgetSection(
                        "advisor_review",
                        user_content,
                        required=True,
                        max_tokens=prompt_budget,
                    )
                ],
                prompt_budget,
            ),
            prompt_budget,
        )

    def _current_skill_text(self) -> str:
        from mediated_coevo.token_budget import fit_text_to_tokens

        text = self.current_skill or "(empty)"
        if not self.budgets:
            return text
        return fit_text_to_tokens(
            self.model,
            text,
            self.budgets.max_skill_tokens,
        )

    def _proposal_blocks(self) -> list[str]:
        return [
            self._proposal_block(index, proposal)
            for index, proposal in enumerate(self.proposals, 1)
        ]

    def _proposal_block(self, index: int, proposal: SkillProposal) -> str:
        from mediated_coevo.evolution.compactor import _diff_parts
        from mediated_coevo.token_budget import fit_text_to_tokens

        reward_str = f"{proposal.reward:.3f}" if proposal.reward is not None else "n/a"
        added, removed, excerpt = _diff_parts(
            proposal.old_content,
            proposal.new_content,
        )
        if self.budgets:
            excerpt = fit_text_to_tokens(
                self.model,
                excerpt,
                self.budgets.skill_update_diff_tokens,
            )
        return (
            f"### Proposal {index} — iter={proposal.iteration} "
            f"task={proposal.task_id} reward={reward_str}\n"
            f"**Reasoning**: {proposal.reasoning}\n"
            f"**Diff**: +{added}/-{removed} lines\n"
            f"```diff\n{excerpt}```\n"
        )


class SkillAdvisor:
    """Batched LLM gate for Planner skill proposals."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
        self._budgets: BudgetsConfig | None = None
        self._condition_name: str | None = None

    @property
    def llm_client(self) -> LLMClient:
        return self._llm

    def configure_token_budget(
        self,
        budgets: BudgetsConfig,
        *,
        condition_name: str | None = None,
    ) -> None:
        self._budgets = budgets
        self._condition_name = condition_name

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

        from mediated_coevo.utils import parse_json_object

        user_content, prompt_budget = SkillAdvisorPrompt(
            current_skill=current_skill,
            proposals=proposals,
            model=self._llm.model,
            budgets=self._budgets,
        ).render()

        try:
            resp = await self._llm.complete(
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user_content},
                ],
                temperature=temperature,
                max_tokens=(
                    min(max_tokens, self._budgets.advisor_completion_tokens)
                    if self._budgets else max_tokens
                ),
                budget_label="skill_advisor.review",
                prompt_budget=prompt_budget,
                budget_overflow_strategy="section_pack",
                condition_name=self._condition_name,
            )
            parsed = parse_json_object(resp["content"])
            if parsed.get("approve"):
                fb = str(parsed.get("feedback", "")).strip()
                return fb or None
        except Exception as e:
            logger.error("SkillAdvisor review failed: %s", e)

        return None
