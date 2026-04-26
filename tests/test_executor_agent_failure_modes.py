"""Failure-mode tests for ExecutorAgent.

ExecutorAgent must NEVER let an exception escape from the Harbor
subprocess boundary; instead it synthesizes a typed env_failure
ExecutionTrace so the orchestrator can keep iterating.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.benchmarks.skillsbench import (
    HarborNotFoundError,
    HarborRunResult,
    HarborTimeoutError,
    SkillsBenchRepository,
)
from mediated_coevo.models.task import TaskSpec


def _scaffold(tmp_path: Path, task_id: str = "demo") -> Path:
    task_dir = tmp_path / "tasks" / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("do it")
    (task_dir / "task.toml").write_text('name = "demo"\n')
    return tmp_path


class _FakeHarbor:
    """In-memory stand-in for HarborRunner."""

    def __init__(self, *, raise_exc: Exception | None = None, result: HarborRunResult | None = None):
        self._raise = raise_exc
        self._result = result
        self.calls: list[tuple[Path, str]] = []

    async def run(self, task_dir: Path, model: str) -> HarborRunResult:
        self.calls.append((task_dir, model))
        if self._raise is not None:
            raise self._raise
        assert self._result is not None
        return self._result


@pytest.fixture
def repo(tmp_path) -> SkillsBenchRepository:
    _scaffold(tmp_path)
    return SkillsBenchRepository(root_dir=tmp_path, task_dirs=["tasks"])


@pytest.mark.asyncio
async def test_harbor_not_found_synthesizes_env_failure(repo, tmp_path):
    harbor = _FakeHarbor(raise_exc=HarborNotFoundError("no harbor"))
    executor = ExecutorAgent(
        model="gemini", # actual name TBC under OpenRouter
        benchmark_repo=repo,
        harbor_runner=harbor,  # type: ignore[arg-type]
        workspace_root=tmp_path / "ws",
        injected_skill_name="executor-evolved",
    )

    trace = await executor.execute_task(
        TaskSpec(task_id="demo", instruction="hi", iteration=2),
        skills=[],
    )

    assert trace.status == "env_failure"
    assert trace.error_kind == "harbor_not_found"
    assert trace.reward is None
    assert trace.task_id == "demo"
    assert trace.iteration == 2


@pytest.mark.asyncio
async def test_harbor_timeout_synthesizes_env_failure(repo, tmp_path):
    harbor = _FakeHarbor(raise_exc=HarborTimeoutError("timed out"))
    executor = ExecutorAgent(
        model="gemini",
        benchmark_repo=repo,
        harbor_runner=harbor,  # type: ignore[arg-type]
        workspace_root=tmp_path / "ws",
        injected_skill_name="executor-evolved",
    )

    trace = await executor.execute_task(
        TaskSpec(task_id="demo", instruction="hi"),
        skills=[],
    )

    assert trace.status == "env_failure"
    assert trace.error_kind == "harbor_timeout"
    assert trace.reward is None


@pytest.mark.asyncio
async def test_missing_task_synthesizes_env_failure(repo, tmp_path):
    harbor = _FakeHarbor(raise_exc=AssertionError("must not run"))
    executor = ExecutorAgent(
        model="gemini",
        benchmark_repo=repo,
        harbor_runner=harbor,  # type: ignore[arg-type]
        workspace_root=tmp_path / "ws",
        injected_skill_name="executor-evolved",
    )

    trace = await executor.execute_task(
        TaskSpec(task_id="does-not-exist", instruction="hi"),
        skills=[],
    )

    assert trace.status == "env_failure"
    assert trace.error_kind == "task_not_found"
    assert trace.reward is None
    # Harbor must not have been called.
    assert harbor.calls == []


@pytest.mark.asyncio
async def test_run_result_with_no_trial_dir_yields_env_failure(repo, tmp_path):
    """Even a 'successful' Harbor run that produced no trial dir is env_failure."""
    harbor = _FakeHarbor(
        result=HarborRunResult(
            job_dir=None, trial_dir=None, returncode=0, stdout="", stderr="",
        )
    )
    executor = ExecutorAgent(
        model="gemini",
        benchmark_repo=repo,
        harbor_runner=harbor,  # type: ignore[arg-type]
        workspace_root=tmp_path / "ws",
        injected_skill_name="executor-evolved",
    )

    trace = await executor.execute_task(
        TaskSpec(task_id="demo", instruction="hi"),
        skills=["# skill"],
    )

    assert trace.status == "env_failure"
    assert trace.error_kind == "missing_trial_dir"
    assert trace.reward is None


@pytest.mark.asyncio
async def test_subprocess_oserror_is_caught(repo, tmp_path):
    harbor = _FakeHarbor(raise_exc=OSError("disk full"))
    executor = ExecutorAgent(
        model="gemini",
        benchmark_repo=repo,
        harbor_runner=harbor,  # type: ignore[arg-type]
        workspace_root=tmp_path / "ws",
        injected_skill_name="executor-evolved",
    )

    trace = await executor.execute_task(
        TaskSpec(task_id="demo", instruction="hi"),
        skills=[],
    )

    assert trace.status == "env_failure"
    assert trace.error_kind == "harbor_subprocess_error"
    assert "disk full" in (trace.error_detail or "")
