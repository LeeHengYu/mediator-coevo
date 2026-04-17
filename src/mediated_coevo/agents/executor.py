"""Executor agent — Gemini.

Runs tasks in a sandboxed environment (Harbor) using current skills.
Produces execution traces with stdout, stderr, test results, and reward.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mediated_coevo.benchmarks import HarborRunner, SkillsBenchRepository, parse_execution_trace

from .base import BaseAgent

if TYPE_CHECKING:
    from mediated_coevo.llm.client import LLMClient
    from mediated_coevo.models.task import TaskSpec
    from mediated_coevo.models.trace import ExecutionTrace

logger = logging.getLogger(__name__)


class ExecutorAgent(BaseAgent):
    """Gemini-backed executor. Runs tasks in Harbor sandbox."""

    @property
    def role(self) -> str:
        return "executor"

    def __init__(
        self,
        llm_client: LLMClient,
        benchmark_repo: SkillsBenchRepository,
        harbor_runner: HarborRunner,
        workspace_root: Path,
        injected_skill_name: str,
        sandbox_config: dict | None = None,
    ) -> None:
        super().__init__("executor", llm_client)
        self._benchmark_repo = benchmark_repo
        self._harbor_runner = harbor_runner
        self._workspace_root = workspace_root
        self._injected_skill_name = injected_skill_name
        self._sandbox_config = sandbox_config or {}

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
            model=self.llm_client.model,
        )
        duration = time.time() - start
        return parse_execution_trace(
            run_result=run_result,
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            duration_sec=duration,
        )
