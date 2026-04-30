from __future__ import annotations

import tomllib

from mediated_coevo.config import Config
from mediated_coevo.main import ExperimentFactory


def test_save_config_omits_none_values_for_toml(tmp_path):
    config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
        }
    )
    config.experiment.shared_notes = None

    ExperimentFactory._save_config(config, tmp_path)

    saved = tomllib.loads((tmp_path / "config.toml").read_text())
    assert "shared_notes" not in saved["experiment"]
    assert saved["experiment"]["allow_cross_task_feedback"] is False
