from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import typer
from typer.testing import CliRunner

from mediated_coevo.cli import run as run_module
from mediated_coevo.main import app


def test_run_single_task_infers_lifelong_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    config = SimpleNamespace(
        paths=SimpleNamespace(benchmarks_dir="benchmarks/skillflow"),
        experiment=SimpleNamespace(
            seed=0,
            num_iterations=1,
            condition_name="learned_mediator",
        ),
    )
    repository = MagicMock()
    repository.resolve.return_value = SimpleNamespace(family="os_interaction")
    captured = {}
    monkeypatch.setattr(run_module, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        run_module,
        "_load_config_or_bad_parameter",
        lambda config_dir, *, overrides: config,
    )
    monkeypatch.setattr(
        run_module,
        "build_benchmark_repo",
        lambda project_root, loaded_config: repository,
    )
    monkeypatch.setattr(
        run_module,
        "run_benchmark_experiment",
        lambda **kwargs: captured.update(kwargs),
    )

    result = CliRunner().invoke(
        app,
        ["run", "--task", "os_interaction/lab-os-0"],
    )

    assert result.exit_code == 0, result.output
    assert config.paths.benchmarks_dir.endswith("benchmarks/lifelong_agent_bench")
    assert captured["selection"].task_ids == ["os_interaction/lab-os-0"]
    assert captured["selection"].families == ("os_interaction",)


def test_base_artifacts_infers_lifelong_root_before_loading_repository(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    captured_overrides = {}
    repository = MagicMock()
    repository.list_local_task_ids.return_value = []

    def load_config(config_dir, *, overrides):
        del config_dir
        captured_overrides.update(overrides)
        return SimpleNamespace()

    monkeypatch.setattr(run_module, "_load_config_or_bad_parameter", load_config)
    monkeypatch.setattr(
        run_module,
        "build_benchmark_repo",
        lambda project_root, config: repository,
    )

    with pytest.raises(typer.BadParameter, match="no local benchmark tasks"):
        run_module.base_artifacts(
            family=["os_interaction"],
            config_dir=tmp_path,
        )

    assert captured_overrides["paths"]["benchmarks_dir"].endswith(
        "benchmarks/lifelong_agent_bench"
    )


def test_base_artifacts_rejects_mixed_lifelong_family_before_loading_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    load_config = MagicMock()
    monkeypatch.setattr(run_module, "_load_config_or_bad_parameter", load_config)

    with pytest.raises(typer.BadParameter, match="cannot mix"):
        run_module.base_artifacts(family=["os_interaction", "other"])

    load_config.assert_not_called()


def test_ensure_base_artifact_store_runs_only_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    config = MagicMock()
    config.paths.data_dir = "data"
    experiment_dir = tmp_path / "base-artifact-run"
    run_experiment = MagicMock(return_value=experiment_dir)
    remove_experiment = MagicMock()
    monkeypatch.setattr(run_module, "run_benchmark_experiment", run_experiment)
    monkeypatch.setattr(
        run_module,
        "_remove_base_artifact_experiment",
        remove_experiment,
    )

    created = run_module.ensure_base_artifact_store(
        config=config,
        task_id="os_interaction/lab-os-1",
        family="os_interaction",
        seed=7,
        output_dir=tmp_path / "stores",
    )

    assert created is True
    assert run_experiment.call_args.kwargs["artifact_store_dir"] == (
        tmp_path / "stores/os_interaction/lab-os-1"
    )
    remove_experiment.assert_called_once_with(experiment_dir, data_dir="data")

    destination = tmp_path / "stores/os_interaction/lab-os-1"
    destination.mkdir(parents=True)
    (destination / "manifest.json").write_text("{}")
    load_store = MagicMock(return_value=())
    monkeypatch.setattr(
        run_module.DiffusionStore,
        "load_artifact_store",
        load_store,
    )

    created = run_module.ensure_base_artifact_store(
        config=config,
        task_id="os_interaction/lab-os-1",
        family="os_interaction",
        seed=7,
        output_dir=tmp_path / "stores",
    )

    assert created is False
    run_experiment.assert_called_once()
    load_store.assert_called_once_with(
        destination,
        expected_store_id="os_interaction/lab-os-1",
    )
