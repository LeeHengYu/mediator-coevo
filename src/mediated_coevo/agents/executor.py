"""Executor agent — runs tasks via Harbor/SkillsBench sandbox."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mediated_coevo.agents.swebench_patch_generator import SWEbenchPatchGenerator
from mediated_coevo.benchmarks import (
    HarborNotFoundError,
    HarborRunner,
    HarborTimeoutError,
    SkillsBenchRepository,
    parse_execution_trace,
)
from mediated_coevo.benchmarks.mixed import BenchmarkKind
from mediated_coevo.benchmarks import swebench as swebench_helpers
from mediated_coevo.models.trace import ExecutionTrace, TokenUsage

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
        envelope_metadata = self._benchmark_repo.executor_envelope_metadata(
            run_dir=task_run_dir,
            executor_policy=skill_text,
        )

        try:
            run_result = await self._harbor_runner.run(
                task_dir=task_run_dir,
                model=self._model,
            )
        except tuple(exc for exc, _ in _HARBOR_ERROR_KINDS) as e:
            kind = next(k for exc, k in _HARBOR_ERROR_KINDS if isinstance(e, exc))
            return self._env_failure(
                task_spec,
                start,
                kind,
                e,
                harbor_metadata=envelope_metadata,
            )

        trace = parse_execution_trace(
            run_result=run_result,
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            duration_sec=time.time() - start,
        )
        trace.harbor_metadata = {**trace.harbor_metadata, **envelope_metadata}
        return trace

    def _env_failure(
        self,
        task_spec: TaskSpec,
        start: float,
        error_kind: str,
        exc: BaseException,
        *,
        harbor_metadata: dict[str, str] | None = None,
    ) -> ExecutionTrace:
        logger.warning(
            "ExecutorAgent: %s for task=%s iter=%d: %s",
            error_kind,
            task_spec.task_id,
            task_spec.iteration,
            exc,
        )
        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            duration_sec=time.time() - start,
            exit_code=-1,
            status="env_failure",
            error_kind=error_kind,
            error_detail=str(exc),
            harbor_metadata=harbor_metadata or {},
        )


class SWEbenchExecutorAgent:
    """Generate and evaluate SWE-bench patches through the official harness."""

    def __init__(
        self,
        model: str,
        benchmark_repo: swebench_helpers.SWEbenchRepository,
        patch_generator: SWEbenchPatchGenerator,
        artifact_root: Path,
        injected_skill_name: str,
        project_root: Path,
        timeout: int,
        max_workers: int,
        run_id_prefix: str,
    ) -> None:
        self._model = model
        self._benchmark_repo = benchmark_repo
        self._patch_generator = patch_generator
        self._artifact_root = artifact_root
        self._workspace_root = artifact_root / "workspaces"
        self._injected_skill_name = injected_skill_name
        self._project_root = project_root
        self._timeout = timeout
        self._max_workers = max_workers
        self._run_id_prefix = run_id_prefix

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
            task_run_dir = self._benchmark_repo.prepare_patch_generation_workspace(
                task=task,
                destination_root=self._workspace_root,
                planner_instruction=task_spec.instruction,
                injected_skill_text=skill_text,
                injected_skill_name=self._injected_skill_name,
            )
        except (OSError, RuntimeError) as e:
            return self._env_failure(
                task_spec,
                start,
                "workspace_prepare_failed",
                e,
            )
        envelope_metadata = swebench_helpers.swebench_executor_envelope_metadata(
            skill_text
        )

        paths: dict[str, str] = {"generation_workspace": str(task_run_dir)}
        try:
            generation = await self._patch_generator.generate_patch(
                task=task,
                workspace=task_run_dir,
                planner_instruction=task_spec.instruction,
                executor_skill_text=skill_text,
                injected_skill_name=self._injected_skill_name,
            )
        except Exception as e:
            paths.update(self._store_generation_failure_logs(task_spec, e))
            return self._env_failure(
                task_spec,
                start,
                "patch_generation_failed",
                e,
                stderr=str(e),
                harbor_paths=paths,
                harbor_metadata=envelope_metadata,
            )

        try:
            paths.update(
                self._store_generation_logs(
                    task_spec,
                    stdout=generation.raw_response,
                    stderr=generation.normalization_notes,
                )
            )
            patch_path = self._write_patch(task_spec, generation.model_patch)
            prediction_path = self._write_prediction(task_spec, generation.model_patch)
        except (OSError, RuntimeError) as e:
            return self._env_failure(
                task_spec,
                start,
                "patch_capture_failed",
                e,
                stdout=generation.raw_response,
                stderr=generation.normalization_notes,
                harbor_paths=paths,
                harbor_metadata=envelope_metadata,
            )

        paths["patch_diff"] = str(patch_path)
        paths["prediction_jsonl"] = str(prediction_path)
        harness_run_id = self._harness_run_id(task_spec)
        harness_output_root = self._harness_output_root(harness_run_id)
        harness_output_root.mkdir(parents=True, exist_ok=True)
        paths["swebench_raw_output"] = str(harness_output_root)
        command = swebench_helpers.build_swebench_harness_command(
            dataset_name=self._benchmark_repo.dataset_name,
            split=self._benchmark_repo.split,
            instance_ids=[task_spec.task_id],
            predictions_path=str(prediction_path),
            run_id=harness_run_id,
            timeout=self._timeout,
            max_workers=self._max_workers,
            report_dir=".",
        )

        try:
            harness_run = swebench_helpers.run_swebench_harness(
                command=command,
                cwd=harness_output_root,
                stream_output=False,
            )
        except OSError as e:
            return self._env_failure(
                task_spec,
                start,
                "harness_subprocess_error",
                e,
                stdout=generation.raw_response,
                stderr=generation.normalization_notes,
                harbor_paths=paths,
                harbor_metadata=envelope_metadata,
            )

        paths.update(self._store_harness_logs(task_spec, harness_run))
        trace = swebench_helpers.build_swebench_traces(
            instance_ids=[task_spec.task_id],
            run_id=harness_run_id,
            project_root=self._project_root,
            harness_run=harness_run,
            raw_output_root=harness_output_root,
        )[0]
        trace.iteration = task_spec.iteration
        trace.duration_sec = time.time() - start
        trace.token_usage = TokenUsage(
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
        )
        trace.harbor_paths = {**paths, **generation.artifacts, **trace.harbor_paths}
        trace.harbor_metadata = {
            **trace.harbor_metadata,
            "verifier_type": swebench_helpers.SWEBENCH_VERIFIER_TYPE,
            "patch_generator": "llm",
            **envelope_metadata,
        }
        return trace

    def _write_patch(self, task_spec: TaskSpec, model_patch: str) -> Path:
        path = (
            self._artifact_root
            / "patches"
            / f"{_artifact_name(task_spec.task_id)}_iter{task_spec.iteration:04d}.patch"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model_patch)
        return path

    def _write_prediction(self, task_spec: TaskSpec, model_patch: str) -> Path:
        path = (
            self._artifact_root
            / "predictions"
            / f"{_artifact_name(task_spec.task_id)}_iter{task_spec.iteration:04d}.jsonl"
        )
        return swebench_helpers.write_prediction_jsonl(
            prediction=swebench_helpers.SWEbenchPrediction(
                instance_id=task_spec.task_id,
                model_patch=model_patch,
                model_name_or_path=self._model,
            ),
            path=path,
        )

    def _store_generation_logs(
        self,
        task_spec: TaskSpec,
        *,
        stdout: str,
        stderr: str,
    ) -> dict[str, str]:
        base = (
            self._artifact_root
            / "generation-logs"
            / f"{_artifact_name(task_spec.task_id)}_iter{task_spec.iteration:04d}"
        )
        return {
            "generation_stdout": str(_write_text(base / "stdout.txt", stdout)),
            "generation_stderr": str(_write_text(base / "stderr.txt", stderr)),
        }

    def _store_generation_failure_logs(
        self,
        task_spec: TaskSpec,
        exc: Exception,
    ) -> dict[str, str]:
        try:
            return self._store_generation_logs(
                task_spec,
                stdout="",
                stderr=f"patch generation failed: {exc}",
            )
        except OSError as log_exc:
            logger.warning(
                "SWEbenchExecutorAgent: failed to write generation failure logs "
                "for task=%s iter=%d: %s",
                task_spec.task_id,
                task_spec.iteration,
                log_exc,
            )
            return {}

    def _store_harness_logs(
        self,
        task_spec: TaskSpec,
        harness_run: swebench_helpers.SWEbenchHarnessRun,
    ) -> dict[str, str]:
        base = (
            self._artifact_root
            / "harness-logs"
            / f"{_artifact_name(task_spec.task_id)}_iter{task_spec.iteration:04d}"
        )
        return {
            "harness_stdout": str(_write_text(base / "stdout.txt", harness_run.stdout)),
            "harness_stderr": str(_write_text(base / "stderr.txt", harness_run.stderr)),
        }

    def _harness_run_id(self, task_spec: TaskSpec) -> str:
        return (
            f"{self._run_id_prefix}-{_artifact_name(task_spec.task_id)}-"
            f"iter{task_spec.iteration:04d}"
        )

    def _harness_output_root(self, harness_run_id: str) -> Path:
        return self._artifact_root / "swebench-harness" / harness_run_id

    def _env_failure(
        self,
        task_spec: TaskSpec,
        start: float,
        error_kind: str,
        exc: BaseException,
        *,
        stdout: str = "",
        stderr: str = "",
        exit_code: int = -1,
        harbor_paths: dict[str, str] | None = None,
        harbor_metadata: dict[str, str] | None = None,
    ) -> ExecutionTrace:
        logger.warning(
            "SWEbenchExecutorAgent: %s for task=%s iter=%d: %s",
            error_kind,
            task_spec.task_id,
            task_spec.iteration,
            exc,
        )
        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            stdout=stdout,
            stderr=stderr,
            duration_sec=time.time() - start,
            exit_code=exit_code,
            status="env_failure",
            error_kind=error_kind,
            error_detail=str(exc),
            harbor_paths=harbor_paths or {},
            harbor_metadata={
                "verifier_type": swebench_helpers.SWEBENCH_VERIFIER_TYPE,
                **(harbor_metadata or {}),
            },
        )


class RoutedExecutorAgent:
    """Route task execution to the selected benchmark-specific executor."""

    def __init__(
        self,
        *,
        benchmark_by_task_id: dict[str, BenchmarkKind],
        skillsbench_executor: ExecutorAgent | None = None,
        swebench_executor: SWEbenchExecutorAgent | None = None,
    ) -> None:
        self._benchmark_by_task_id = dict(benchmark_by_task_id)
        self._skillsbench_executor = skillsbench_executor
        self._swebench_executor = swebench_executor

    async def execute_task(
        self,
        task_spec: TaskSpec,
        skills: list[str],
    ) -> ExecutionTrace:
        start = time.time()
        benchmark = self._benchmark_by_task_id.get(task_spec.task_id)
        if benchmark == "skillsbench":
            if self._skillsbench_executor is None:
                return self._env_failure(
                    task_spec,
                    start,
                    "executor_not_configured",
                    "SkillsBench executor is not configured",
                )
            return await self._skillsbench_executor.execute_task(task_spec, skills)
        if benchmark == "swebench":
            if self._swebench_executor is None:
                return self._env_failure(
                    task_spec,
                    start,
                    "executor_not_configured",
                    "SWE-bench executor is not configured",
                )
            return await self._swebench_executor.execute_task(task_spec, skills)
        return self._env_failure(
            task_spec,
            start,
            "task_not_selected",
            f"Task {task_spec.task_id!r} is not selected for this run",
        )

    def _env_failure(
        self,
        task_spec: TaskSpec,
        start: float,
        error_kind: str,
        error_detail: str,
    ) -> ExecutionTrace:
        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            duration_sec=time.time() - start,
            exit_code=-1,
            status="env_failure",
            error_kind=error_kind,
            error_detail=error_detail,
        )


def _artifact_name(task_id: str) -> str:
    return task_id.replace("/", "_").replace("\\", "_")


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path
