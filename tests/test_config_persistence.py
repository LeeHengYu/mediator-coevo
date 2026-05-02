from __future__ import annotations

import tomllib

from mediated_coevo.config import Config
from mediated_coevo.main import ExperimentFactory


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
        }
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
    assert saved["experiment"]["allow_cross_task_feedback"] is False
    assert saved["experiment"]["skill_updates"] == {
        "executor": True,
        "planner": True,
        "mediator": True,
    }
    assert saved["executor_runtime"]["remote_fetch"] is True
    assert "remote_repo" not in saved["executor_runtime"]
    assert "remote_ref" not in saved["executor_runtime"]
