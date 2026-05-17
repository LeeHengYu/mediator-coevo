from __future__ import annotations

import re
from dataclasses import dataclass

import pytest
from typer.testing import CliRunner

from mediated_coevo import main as main_module
from mediated_coevo.agents.executor import RoutedExecutorAgent
from mediated_coevo.benchmarks import swebench
from mediated_coevo.benchmarks.mixed import (
    MixedBenchmarkRepository,
    build_benchmark_task_selection,
)
from mediated_coevo.core.config import Config
from mediated_coevo.main import app
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace


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
    def __init__(self, status: str) -> None:
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


def test_unified_run_requires_at_least_one_benchmark_selection():
    result = CliRunner().invoke(app, ["run"])

    assert result.exit_code != 0
    assert "provide at least one SkillsBench task or SWE-bench instance" in result.output


def test_unified_run_delegates_skillsbench_only(monkeypatch):
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        main_module,
        "_run_unified_experiment",
        lambda **kwargs: captured.update(kwargs),
    )

    result = CliRunner().invoke(
        app,
        ["run", "--skillsbench-task", "fix-build-google-auto", "--iterations", "4"],
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
        ["run", "--swebench-instance", "django__django-11910"],
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
            "--skill-validation",
        ],
    )

    assert result.exit_code == 0
    selection = captured["selection"]
    config = captured["config"]
    assert selection.backend_name == "mixed"
    assert selection.task_ids == ["fix-build-google-auto", "sympy__sympy-13915"]
    assert config.experiment.advisor_buffer_max == 2
    assert config.experiment.coevo_interval == 1
    assert config.experiment.skill_validation.enabled is True


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

    result = CliRunner().invoke(app, ["run", "--swebench-limit", "2"])

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
        ],
    )

    assert result.exit_code != 0
    assert "must be disjoint" in result.output


def test_legacy_swebench_run_command_is_not_registered():
    result = CliRunner().invoke(app, ["swebench", "run"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_unified_experiment_root_prefixes_user_run_id_with_timestamp(
    monkeypatch,
    tmp_path,
):
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
    config.paths.skills_dir = "skills"
    config.paths.data_dir = "data"
    monkeypatch.setattr(main_module, "PROJECT_ROOT", tmp_path)

    experiment_dir, runtime_skills_dir = main_module._prepare_unified_experiment_root(
        config=config,
        seed=42,
        run_id="custom-tail",
        suffix="mixed",
    )

    assert re.fullmatch(r"\d{8}-\d{6}-custom-tail", experiment_dir.name)
    assert runtime_skills_dir == experiment_dir / "skills"
