"""Executor agent — runs tasks via Harbor/SkillsBench sandbox."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mediated_coevo.benchmarks import HarborRunner, SkillsBenchRepository, parse_execution_trace

if TYPE_CHECKING:
    from mediated_coevo.models.task import TaskSpec
    from mediated_coevo.models.trace import ExecutionTrace

logger = logging.getLogger(__name__)


class ExecutorAgent:
    """Runs tasks in the Harbor sandbox. Makes no direct LLM calls."""

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
        """Execute a local SkillsBench task and return an ExecutionTrace."""
        start = time.time()
        task = self._benchmark_repo.resolve(task_spec.task_id)
        skill_text = skills[0] if skills else None
        task_run_dir = self._benchmark_repo.prepare_run_workspace(
            task=task,
            destination_root=self._workspace_root,
            planner_instruction=task_spec.instruction,
            injected_skill_text=skill_text,
            injected_skill_name=self._injected_skill_name,
        )
        run_result = await self._harbor_runner.run(
            task_dir=task_run_dir,
            model=self._model,
        )
        duration = time.time() - start
        return parse_execution_trace(
            run_result=run_result,
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            duration_sec=duration,
        )
