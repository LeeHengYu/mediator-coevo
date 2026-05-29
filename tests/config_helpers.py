from __future__ import annotations

from mediated_coevo.core.config import (
    DiffusionConfig,
    ExperimentConfig,
    SkillUpdateConfig,
)


def diffusion_config() -> DiffusionConfig:
    return DiffusionConfig(enabled=False, policy="none", max_artifacts=3)


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
        allow_cross_task_feedback=False,
    )
