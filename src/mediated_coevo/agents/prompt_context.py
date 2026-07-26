"""Shared prompt section types for agent prompt assembly."""

from __future__ import annotations

from dataclasses import dataclass

from mediated_coevo.runtime.token_budget import BudgetSection, OverflowStrategy


@dataclass(frozen=True)
class PromptSection:
    """One logical prompt section before rendering or token packing."""

    name: str
    content: str
    required: bool = False
    max_tokens: int | None = None
    overflow_strategy: OverflowStrategy = "head_tail"

    def to_budget_section(
        self,
        *,
        overflow_strategy: OverflowStrategy | None = None,
    ) -> BudgetSection:
        return BudgetSection(
            self.name,
            self.content,
            required=self.required,
            max_tokens=self.max_tokens,
            overflow_strategy=overflow_strategy or self.overflow_strategy,
        )


@dataclass(frozen=True)
class PlannerPriorContextBundle:
    """Structured planner context assembled from prior and diffusion channels."""

    same_task_prior: str | None = None
    cross_task_prior: str | None = None
    diffusion_context: str | None = None
    context_budget_violation: bool = False

    def sections(self) -> tuple[PromptSection, ...]:
        sections: list[PromptSection] = []
        if self.same_task_prior:
            sections.append(
                PromptSection(
                    name="same_task_prior",
                    content=self.same_task_prior,
                    required=True,
                )
            )
        if self.diffusion_context:
            sections.append(
                PromptSection(
                    name="diffusion_context",
                    content=self.diffusion_context,
                    required=True,
                )
            )
        if self.cross_task_prior:
            sections.append(
                PromptSection(
                    name="cross_task_prior",
                    content=self.cross_task_prior,
                    required=True,
                )
            )
        return tuple(sections)

    def flatten(self) -> str | None:
        contents = [section.content for section in self.sections()]
        if not contents:
            return None
        return "\n\n".join(contents)
