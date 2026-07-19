from __future__ import annotations

from mediated_coevo.core.config import (
    BudgetsConfig,
    DiffusionConfig,
    ExperimentConfig,
    ModelsConfig,
)


def budgets_config() -> BudgetsConfig:
    return BudgetsConfig(
        max_skill_tokens=4000,
        max_same_task_prior_tokens=300,
        max_transfer_context_tokens=900,
        trace_excerpt_tokens=6000,
        historical_summary_tokens=3000,
        mediator_report_tokens=4000,
        planner_context_tokens=24000,
        mediator_prompt_tokens=16000,
        judge_prompt_tokens=16000,
        planner_completion_tokens=4096,
        mediator_completion_tokens=2048,
        judge_completion_tokens=2048,
    )


def models_config() -> ModelsConfig:
    return ModelsConfig(
        planner="test-planner",
        executor="test-executor",
        mediator="test-mediator",
        judge="test-judge",
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
        seed=42,
        condition_name="learned_mediator",
    )
