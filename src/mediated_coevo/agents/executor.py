"""Executor agent — runs tasks via Harbor/SkillsBench sandbox."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mediated_coevo.benchmarks import (
    HarborNotFoundError,
    HarborRunner,
    HarborTimeoutError,
    SkillsBenchRepository,
    parse_execution_trace,
)
from mediated_coevo.models.trace import ExecutionTrace

if TYPE_CHECKING:
    from mediated_coevo.models.task import TaskSpec

logger = logging.getLogger(__name__)

# Maps an exception raised by HarborRunner.run to the error_kind we surface.
# OSError must be last (other entries are subclasses we want to detect first).
_HARBOR_ERROR_KINDS: tuple[tuple[type[BaseException], str], ...] = (
    (HarborNotFoundError, "harbor_not_found"),
    (HarborTimeoutError, "harbor_timeout"),
    (OSError, "harbor_subprocess_error"),
)


class ExecutorAgent:
    """Runs tasks in the Harbor sandbox. Makes no direct LLM calls.

    Failures at the environment boundary (resolve, prepare workspace,
    harbor subprocess) are turned into typed ``env_failure`` traces with
    ``reward=None`` rather than propagated, so the orchestrator can keep
    iterating.
    """

    def __init__(
        self,
        model: str,
        benchmark_repo: SkillsBenchRepository,
        harbor_runner: HarborRunner,
        workspace_root: Path,
        injected_skill_name: str,
    ) -> None:
        self._model = model
        self._benchmark_repo = benchmark_repo
        self._harbor_runner = harbor_runner
        self._workspace_root = workspace_root
        self._injected_skill_name = injected_skill_name

    async def execute_task(
        self,
        task_spec: TaskSpec,
        skills: list[str],
    ) -> ExecutionTrace:
        start = time.time()

        try:
            task = self._benchmark_repo.resolve(task_spec.task_id)
        except FileNotFoundError as e:
            return self._env_failure(task_spec, start, "task_not_found", e)

        skill_text = skills[0] if skills else None
        try:
            task_run_dir = self._benchmark_repo.prepare_run_workspace(
                task=task,
                destination_root=self._workspace_root,
                planner_instruction=task_spec.instruction,
                injected_skill_text=skill_text,
                injected_skill_name=self._injected_skill_name,
            )
        except OSError as e:
            return self._env_failure(task_spec, start, "workspace_prepare_failed", e)

        try:
            run_result = await self._harbor_runner.run(
                task_dir=task_run_dir,
                model=self._model,
            )
        except tuple(exc for exc, _ in _HARBOR_ERROR_KINDS) as e:
            kind = next(k for exc, k in _HARBOR_ERROR_KINDS if isinstance(e, exc))
            return self._env_failure(task_spec, start, kind, e)

        return parse_execution_trace(
            run_result=run_result,
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            duration_sec=time.time() - start,
        )

    def _env_failure(
        self,
        task_spec: TaskSpec,
        start: float,
        error_kind: str,
        exc: BaseException,
    ) -> ExecutionTrace:
        logger.warning(
            "ExecutorAgent: %s for task=%s iter=%d: %s",
            error_kind, task_spec.task_id, task_spec.iteration, exc,
        )
        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            duration_sec=time.time() - start,
            exit_code=-1,
            status="env_failure",
            error_kind=error_kind,
            error_detail=str(exc),
        )
