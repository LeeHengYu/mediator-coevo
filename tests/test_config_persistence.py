from __future__ import annotations

import tomllib

from mediated_coevo.core.config import Config
from mediated_coevo.experiment.runtime_factory import ExperimentFactory
from tests.config_helpers import budgets_config, diffusion_config, experiment_config


def test_factory_persisted_config_omits_none_values_for_toml(tmp_path):
    for skill_name in ("executor", "planner", "mediator"):
        skill_dir = tmp_path / "skills" / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# {skill_name}\n")

    config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        },
        budgets=budgets_config(),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    config.experiment.shared_notes = None

    runtime = ExperimentFactory(tmp_path).build(
        config=config,
        seed=42,
        condition_name=config.experiment.condition_name,
        experiment_dir=tmp_path / "experiment",
    )

    saved = tomllib.loads((runtime.experiment_dir / "config.toml").read_text())
    assert "shared_notes" not in saved["experiment"]
    assert "allow_cross_task_feedback" not in saved["experiment"]
    assert saved["diffusion"]["enabled"] is False
    assert saved["diffusion"]["policy"] == "none"
    assert saved["diffusion"]["graph"] == "none"
    assert saved["diffusion"]["max_artifacts"] == 3
    assert saved["diffusion"]["top_k_neighbors"] == 3
    assert saved["diffusion"]["avoid_recheck_max_artifacts"] == 1
    assert saved["experiment"]["skill_updates"] == {
        "executor": True,
        "planner": True,
        "mediator": True,
    }
