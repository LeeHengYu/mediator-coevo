from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mediated_coevo import main as main_module
from mediated_coevo.agents.executor import RoutedExecutorAgent
from mediated_coevo.benchmarks import swebench
from mediated_coevo.benchmarks.mixed import (
    MixedBenchmarkRepository,
    build_benchmark_task_selection,
)
from mediated_coevo.core.config import Config, ModelsConfig
from mediated_coevo.main import ExperimentRuntime, MatrixRuntime, app
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace, TraceStatus
from tests.config_helpers import diffusion_config, experiment_config


@dataclass
class _Task:
    task_id: str
    instruction: str = "do it"
    task_config: dict | None = None


class _Repo:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self.calls: list[str] = []

    def resolve(self, task_id: str) -> _Task:
        self.calls.append(task_id)
        return _Task(
            task_id=task_id,
            task_config={
                "metadata": {"category": self.prefix},
                "verifier": {"type": f"{self.prefix}_verifier"},
            },
        )


class _Executor:
    def __init__(self, status: TraceStatus) -> None:
        self.status = status
        self.calls: list[str] = []

    async def execute_task(
        self,
        task_spec: TaskSpec,
        skills: list[str],
    ) -> ExecutionTrace:
        self.calls.append(task_spec.task_id)
        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            status=self.status,
            reward=1.0,
        )


def _write_runtime_skills(root: Path) -> None:
    for skill_name in ("executor", "planner", "mediator"):
        skill_dir = root / "skills" / skill_name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(f"# {skill_name}\n")


def _experiment_root_config() -> Config:
    config = Config(
        models=ModelsConfig(
            planner="test-planner",
            executor="test-executor",
            mediator="test-mediator",
            judge="test-judge",
        ),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    config.paths.skills_dir = "skills"
    config.paths.data_dir = "data"
    return config


def test_mixed_repository_routes_selected_tasks_to_backend_repos():
    selection = build_benchmark_task_selection(
        skillsbench_task_ids=["skills-task"],
        swebench_instance_ids=["swe-task"],
    )
    skills_repo = _Repo("skillsbench")
    swe_repo = _Repo("swebench")
    repo = MixedBenchmarkRepository(
        benchmark_by_task_id=selection.benchmark_by_task_id,
        skillsbench_repo=skills_repo,  # type: ignore[arg-type]
        swebench_repo=swe_repo,  # type: ignore[arg-type]
    )

    assert repo.resolve("skills-task").task_config["metadata"]["category"] == (
        "skillsbench"
    )
    assert repo.resolve("swe-task").task_config["metadata"]["category"] == "swebench"
    assert skills_repo.calls == ["skills-task"]
    assert swe_repo.calls == ["swe-task"]

    with pytest.raises(FileNotFoundError, match="not selected"):
        repo.resolve("missing")


def test_runtime_routing_includes_validation_only_swebench_when_enabled():
    selection = build_benchmark_task_selection(
        skillsbench_task_ids=["skills-task"],
        swebench_instance_ids=[],
    )
    config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        },
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    config.experiment.skill_validation.swebench_instances = ["django__django-11099"]
    config.experiment.skill_validation.allow_swebench_replacement_for_skillsbench = True

    routing = main_module._runtime_benchmark_by_task_id(selection, config)

    assert routing == {
        "skills-task": "skillsbench",
        "django__django-11099": "swebench",
    }
    assert main_module._needs_runtime_swebench(selection, config) is True


def test_runtime_routing_ignores_validation_only_swebench_when_disabled():
    selection = build_benchmark_task_selection(
        skillsbench_task_ids=["skills-task"],
        swebench_instance_ids=[],
    )
    config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        },
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    config.experiment.skill_validation.swebench_instances = ["django__django-11099"]
    config.experiment.skill_validation.allow_swebench_replacement_for_skillsbench = (
        False
    )

    routing = main_module._runtime_benchmark_by_task_id(selection, config)

    assert routing == {"skills-task": "skillsbench"}
    assert main_module._needs_runtime_swebench(selection, config) is False


@pytest.mark.asyncio
async def test_routed_executor_routes_selected_tasks_to_backend_executors():
    selection = build_benchmark_task_selection(
        skillsbench_task_ids=["skills-task"],
        swebench_instance_ids=["swe-task"],
    )
    skills_executor = _Executor("ok")
    swe_executor = _Executor("task_failed")
    executor = RoutedExecutorAgent(
        benchmark_by_task_id=selection.benchmark_by_task_id,
        skillsbench_executor=skills_executor,  # type: ignore[arg-type]
        swebench_executor=swe_executor,  # type: ignore[arg-type]
    )

    skills_trace = await executor.execute_task(
        TaskSpec(task_id="skills-task", instruction="x", iteration=3),
        skills=["# skill"],
    )
    swe_trace = await executor.execute_task(
        TaskSpec(task_id="swe-task", instruction="x", iteration=4),
        skills=[],
    )
    missing_trace = await executor.execute_task(
        TaskSpec(task_id="missing", instruction="x", iteration=5),
        skills=[],
    )

    assert skills_trace.status == "ok"
    assert swe_trace.status == "task_failed"
    assert missing_trace.status == "env_failure"
    assert missing_trace.error_kind == "task_not_selected"
    assert skills_executor.calls == ["skills-task"]
    assert swe_executor.calls == ["swe-task"]


def test_swebench_executor_uses_openrouter_model_for_llm_patch_generation(
    monkeypatch,
    tmp_path,
):
    captured: dict[str, str] = {}

    class _PatchGenerator:
        def __init__(self, llm_client) -> None:
            captured["llm_model"] = llm_client.model

    monkeypatch.setattr(main_module, "LLMSWEbenchPatchGenerator", _PatchGenerator)
    config = Config(
        models={
            "planner": "openrouter/test/planner",
            "executor": "google/gemini-3-flash-preview",
            "mediator": "openrouter/test/mediator",
            "judge": "openrouter/test/judge",
        },
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )

    executor = main_module._build_swebench_executor(
        config=config,
        benchmark_repo=object(),  # type: ignore[arg-type]
        phase_dir=tmp_path,
        timeout=1,
        max_workers=1,
        run_id_prefix="test-run",
    )

    assert captured["llm_model"] == "openrouter/google/gemini-3-flash-preview"
    assert executor._model == "google/gemini-3-flash-preview"
    assert (
        main_module._openrouter_model_for_llm(
            "openrouter/openrouter/google/gemini-3-flash-preview"
        )
        == "openrouter/google/gemini-3-flash-preview"
    )


def test_unified_run_requires_at_least_one_benchmark_selection():
    result = CliRunner().invoke(app, ["run", "--run-id", "missing-selection"])

    assert result.exit_code != 0
    assert "provide at least one SkillsBench task or SWE-bench instance" in result.output


def test_unified_run_allows_default_run_id(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        main_module,
        "_run_unified_experiment",
        lambda **kwargs: captured.update(kwargs),
    )

    result = CliRunner().invoke(app, ["run", "--skillsbench-task", "demo"])

    assert result.exit_code == 0
    assert captured["run_id"] is None


def test_unified_run_delegates_skillsbench_only(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        main_module,
        "_run_unified_experiment",
        lambda **kwargs: captured.update(kwargs),
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--skillsbench-task",
            "fix-build-google-auto",
            "--iterations",
            "4",
            "--run-id",
            "skillsbench-only",
        ],
    )

    assert result.exit_code == 0
    selection = captured["selection"]
    assert selection.backend_name == "skillsbench"
    assert selection.task_ids == ["fix-build-google-auto"]
    assert captured["iterations"] == 4


def test_unified_run_delegates_swebench_only(monkeypatch):
    captured: dict[str, object] = {}

    def resolve_ids(**kwargs) -> list[str]:
        assert kwargs["raw_instance_ids"] == ["django__django-11910"]
        return swebench.parse_swebench_instance_ids(kwargs["raw_instance_ids"])

    monkeypatch.setattr(swebench, "resolve_swebench_instance_ids", resolve_ids)
    monkeypatch.setattr(
        main_module,
        "_run_unified_experiment",
        lambda **kwargs: captured.update(kwargs),
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--swebench-instance",
            "django__django-11910",
            "--run-id",
            "swebench-only",
        ],
    )

    assert result.exit_code == 0
    selection = captured["selection"]
    assert selection.backend_name == "swebench"
    assert selection.task_ids == ["django__django-11910"]


def test_unified_run_delegates_mixed_selection_and_controls(monkeypatch):
    captured: dict[str, object] = {}

    def resolve_ids(**kwargs) -> list[str]:
        return swebench.parse_swebench_instance_ids(kwargs["raw_instance_ids"])

    monkeypatch.setattr(swebench, "resolve_swebench_instance_ids", resolve_ids)
    monkeypatch.setattr(
        main_module,
        "_run_unified_experiment",
        lambda **kwargs: captured.update(kwargs),
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--skillsbench-task",
            "fix-build-google-auto",
            "--swebench-instance",
            "sympy__sympy-13915",
            "--advisor-buffer-max",
            "2",
            "--coevo-interval",
            "1",
            "--run-id",
            "mixed-controls",
        ],
    )

    assert result.exit_code == 0
    selection = captured["selection"]
    config = captured["config"]
    assert selection.backend_name == "mixed"
    assert selection.task_ids == ["fix-build-google-auto", "sympy__sympy-13915"]
    assert config.experiment.advisor_buffer_max == 2
    assert config.experiment.coevo_interval == 1


def test_unified_run_supports_swebench_limit(monkeypatch):
    captured: dict[str, object] = {}
    resolve_calls: list[dict] = []

    def resolve_ids(**kwargs) -> list[str]:
        resolve_calls.append(kwargs)
        return ["instance-a", "instance-b"]

    monkeypatch.setattr(swebench, "resolve_swebench_instance_ids", resolve_ids)
    monkeypatch.setattr(
        main_module,
        "_run_unified_experiment",
        lambda **kwargs: captured.update(kwargs),
    )

    result = CliRunner().invoke(
        app,
        ["run", "--swebench-limit", "2", "--run-id", "swebench-limit"],
    )

    assert result.exit_code == 0
    selection = captured["selection"]
    assert selection.swebench_instance_ids == ["instance-a", "instance-b"]
    assert resolve_calls[0]["limit"] == 2


def test_unified_run_supports_swebench_frozen_eval(monkeypatch):
    captured: dict[str, object] = {}
    resolve_calls: list[dict] = []

    def resolve_ids(**kwargs) -> list[str]:
        resolve_calls.append(kwargs)
        return swebench.parse_swebench_instance_ids(kwargs["raw_instance_ids"])

    monkeypatch.setattr(swebench, "resolve_swebench_instance_ids", resolve_ids)
    monkeypatch.setattr(
        main_module,
        "_run_unified_experiment",
        lambda **kwargs: captured.update(kwargs),
    )

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--swebench-instance",
            "django__django-11910",
            "--swebench-eval-instance",
            "django__django-11099",
            "--run-id",
            "swebench-eval",
        ],
    )

    assert result.exit_code == 0
    assert captured["swebench_eval_instance_ids"] == ["django__django-11099"]
    assert resolve_calls[0]["raw_instance_ids"] == ["django__django-11910"]
    assert resolve_calls[1]["raw_instance_ids"] == ["django__django-11099"]


def test_unified_run_rejects_overlapping_swebench_eval(monkeypatch):
    def resolve_ids(**kwargs) -> list[str]:
        return swebench.parse_swebench_instance_ids(kwargs["raw_instance_ids"])

    monkeypatch.setattr(swebench, "resolve_swebench_instance_ids", resolve_ids)

    result = CliRunner().invoke(
        app,
        [
            "run",
            "--swebench-instance",
            "django__django-11910",
            "--swebench-eval-instance",
            "django__django-11910",
            "--run-id",
            "swebench-overlap",
        ],
    )

    assert result.exit_code != 0
    assert "must be disjoint" in result.output


def test_legacy_swebench_run_command_is_not_registered():
    result = CliRunner().invoke(app, ["swebench", "run"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_unified_experiment_invokes_judge_after_summary(monkeypatch, tmp_path):
    events: list[str] = []
    history_store = object()

    class _Orchestrator:
        config = Config(
            models={
                "planner": "p",
                "executor": "e",
                "mediator": "m",
                "judge": "j",
            },
            experiment=experiment_config(),
            diffusion=diffusion_config(),
        )

        async def run_experiment(self, task_ids, iterations):
            events.append("run")
            return []

    run_dir = tmp_path / "run"
    skills_dir = run_dir / "skills"
    runtime = ExperimentRuntime(
        experiment_dir=run_dir,
        orchestrator=_Orchestrator(),  # type: ignore[arg-type]
    )
    runtime.orchestrator.history_store = history_store
    selection = build_benchmark_task_selection(
        skillsbench_task_ids=["task-a"],
        swebench_instance_ids=[],
    )

    monkeypatch.setattr(main_module, "_prepare_llm_credentials_or_exit", lambda c: c)
    monkeypatch.setattr(main_module, "_ensure_harbor_available", lambda c: None)
    monkeypatch.setattr(
        main_module,
        "_prepare_unified_experiment_root",
        lambda **kwargs: (run_dir, skills_dir),
    )
    monkeypatch.setattr(main_module, "_build_unified_runtime", lambda **kwargs: runtime)
    monkeypatch.setattr(
        main_module,
        "_write_and_print_result_summary",
        lambda **kwargs: events.append("summary"),
    )

    def annotate(**kwargs) -> None:
        assert kwargs["data_dir"] == run_dir
        assert kwargs["history_store"] is history_store
        events.append("judge")

    monkeypatch.setattr(main_module, "_annotate_judge_rewards_or_exit", annotate)

    main_module._run_unified_experiment(
        config=_Orchestrator.config,
        selection=selection,
        iterations=1,
        seed=42,
        condition_name="learned_mediator",
        dataset_name=swebench.DEFAULT_SWEBENCH_DATASET,
        split=swebench.DEFAULT_SWEBENCH_SPLIT,
        timeout=1,
        max_workers=1,
        run_id="judge-summary",
        swebench_eval_instance_ids=[],
    )

    assert events == ["run", "summary", "judge"]


def test_matrix_invokes_judge_for_each_row(monkeypatch):
    events: list[tuple[str, Path]] = []

    class _Orchestrator:
        config = Config(
            models={
                "planner": "p",
                "executor": "e",
                "mediator": "m",
                "judge": "j",
            },
            experiment=experiment_config(),
            diffusion=diffusion_config(),
        )
        history_store = object()

        async def run_experiment(self, task_ids, iterations):
            return []

    row_dir = Path("/tmp/matrix/no_feedback")
    runtime = ExperimentRuntime(
        experiment_dir=row_dir,
        orchestrator=_Orchestrator(),  # type: ignore[arg-type]
    )
    monkeypatch.setattr(main_module, "_preflight_task_ids_from_cli", lambda *a: ["task-a"])
    monkeypatch.setattr(main_module, "_prepare_llm_credentials_or_exit", lambda c: c)
    monkeypatch.setattr(main_module, "_ensure_harbor_available", lambda c: None)
    monkeypatch.setattr(main_module, "_build_benchmark_repo", lambda *a: object())
    monkeypatch.setattr(
        main_module.ExperimentFactory,
        "create_matrix_dir",
        lambda self, **kwargs: Path("/tmp/matrix"),
    )
    monkeypatch.setattr(
        main_module,
        "_build_matrix_runtimes",
        lambda **kwargs: [MatrixRuntime(preset_name="no_feedback", runtime=runtime)],
    )
    monkeypatch.setattr(
        main_module,
        "_write_and_print_result_summary",
        lambda **kwargs: events.append(("summary", kwargs["data_dir"])),
    )
    monkeypatch.setattr(
        main_module,
        "_annotate_judge_rewards_or_exit",
        lambda **kwargs: events.append(("judge", kwargs["data_dir"])),
    )

    result = CliRunner().invoke(app, ["matrix", "--tasks", "task-a", "--iterations", "1"])

    assert result.exit_code == 0
    assert events == [("summary", row_dir), ("judge", row_dir)]


def test_unified_experiment_root_prefixes_user_run_id_with_timestamp(
    monkeypatch,
    tmp_path,
):
    _write_runtime_skills(tmp_path)
    config = _experiment_root_config()
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)

    experiment_dir, runtime_skills_dir = main_module._prepare_unified_experiment_root(
        config=config,
        seed=42,
        run_id="custom-tail",
        suffix="skillsbench",
    )

    assert re.fullmatch(r"\d{8}-\d{6}-custom-tail", experiment_dir.name)
    assert runtime_skills_dir == experiment_dir / "skills"


def test_unified_experiment_root_defaults_run_id_to_seed_and_backend(
    monkeypatch,
    tmp_path,
):
    _write_runtime_skills(tmp_path)
    config = _experiment_root_config()
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)

    experiment_dir, runtime_skills_dir = main_module._prepare_unified_experiment_root(
        config=config,
        seed=7,
        run_id=None,
        suffix="mixed",
    )

    assert re.fullmatch(r"\d{8}-\d{6}-7-mixed", experiment_dir.name)
    assert runtime_skills_dir == experiment_dir / "skills"
