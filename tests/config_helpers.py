from __future__ import annotations

from mediated_coevo.core.config import (
    BudgetsConfig,
    DiffusionConfig,
    ExperimentConfig,
    SkillUpdateConfig,
)


def budgets_config() -> BudgetsConfig:
    return BudgetsConfig(
        max_skill_tokens=4000,
        max_diffusion_context_tokens=4000,
        trace_excerpt_tokens=6000,
        historical_summary_tokens=3000,
        mediator_report_tokens=4000,
        planner_context_tokens=24000,
        skill_update_diff_tokens=6000,
        mediator_prompt_tokens=16000,
        advisor_prompt_tokens=12000,
        reflector_prompt_tokens=16000,
        judge_prompt_tokens=16000,
        planner_completion_tokens=4096,
        mediator_completion_tokens=2048,
        advisor_completion_tokens=512,
        reflector_completion_tokens=4096,
        judge_completion_tokens=2048,
    )


def diffusion_config() -> DiffusionConfig:
    return DiffusionConfig(
        enabled=False,
        policy="none",
        max_artifacts=3,
        top_k_neighbors=3,
    )


def experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        num_iterations=2,
        coevo_interval=5,
        seed=42,
        advisor_buffer_max=10,
        condition_name="learned_mediator",
        skill_updates=SkillUpdateConfig(
            executor=True,
            planner=True,
            mediator=True,
        ),
    )
