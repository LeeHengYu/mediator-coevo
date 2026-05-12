from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mediated_coevo.agents.executor import SWEbenchExecutorAgent
from mediated_coevo.agents.swebench_patch_generator import (
    SWEbenchPatchGeneration,
    capture_swebench_sandbox_diff,
    normalize_swebench_model_patch,
    prepare_local_swebench_sandbox,
)
from mediated_coevo.benchmarks import swebench
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace


class _FakeSWEbenchRepo:
    dataset_name = "SWE-bench/SWE-bench_Lite"
    split = "test"

    def __init__(self) -> None:
        self.prepared_workspace: Path | None = None
        self.injected_skill_text: str | None = None

    def resolve(self, task_id: str) -> swebench.SWEbenchTask:
        return swebench.SWEbenchTask(
            task_id=task_id,
            instance={
                "instance_id": task_id,
                "repo": "sympy/sympy",
                "base_commit": "abc123",
                "problem_statement": "fix it",
            },
            instruction="# issue\n\nfix it",
            task_config={
                "metadata": {"expected_reward_range": [0.0, 1.0]},
                "verifier": {"type": swebench.SWEBENCH_VERIFIER_TYPE},
            },
            repo="sympy/sympy",
            base_commit="abc123",
        )

    def prepare_patch_generation_workspace(
        self,
        *,
        task: swebench.SWEbenchTask,
        destination_root: Path,
        planner_instruction: str,
        injected_skill_text: str | None,
        injected_skill_name: str,
    ) -> Path:
        workspace = destination_root / task.task_id / "run-test"
        workspace.mkdir(parents=True)
        (workspace / "instruction.md").write_text(planner_instruction)
        if injected_skill_text:
            skill_dir = workspace / "environment" / "skills" / injected_skill_name
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(injected_skill_text)
        self.prepared_workspace = workspace
        self.injected_skill_text = injected_skill_text
        return workspace


class _FakePatchGenerator:
    def __init__(
        self,
        generation: SWEbenchPatchGeneration | None = None,
        exc: Exception | None = None,
    ) -> None:
        self.generation = generation
        self.exc = exc
        self.calls: list[dict[str, Any]] = []

    async def generate_patch(
        self,
        *,
        task: swebench.SWEbenchTask,
        workspace: Path,
        planner_instruction: str,
        executor_skill_text: str | None,
        injected_skill_name: str,
    ) -> SWEbenchPatchGeneration:
        self.calls.append(
            {
                "task": task,
                "workspace": workspace,
                "planner_instruction": planner_instruction,
                "executor_skill_text": executor_skill_text,
                "injected_skill_name": injected_skill_name,
            }
        )
        if self.exc is not None:
            raise self.exc
        assert self.generation is not None
        return self.generation


@pytest.mark.asyncio
async def test_swebench_executor_writes_prediction_from_generated_diff(
    tmp_path,
    monkeypatch,
):
    patch = "diff --git a/demo.py b/demo.py\n+print('fixed')\n"
    raw_response = f"```diff\n{patch}```"
    repo = _FakeSWEbenchRepo()
    generator = _FakePatchGenerator(
        SWEbenchPatchGeneration(
            raw_response=raw_response,
            model_patch=patch,
            normalization_notes="Extracted fenced diff block.",
            input_tokens=11,
            output_tokens=22,
            artifacts={"sandbox_stdout": "sandbox/stdout.txt"},
        )
    )
    captured: dict[str, Any] = {}

    def run_harness(**kwargs) -> swebench.SWEbenchHarnessRun:
        captured["harness_command"] = kwargs["command"]
        captured["harness_cwd"] = kwargs["cwd"]
        return swebench.SWEbenchHarnessRun(
            command=kwargs["command"],
            returncode=0,
            stdout="harness ok",
            stderr="",
            duration_sec=1.0,
        )

    def build_traces(**kwargs) -> list[ExecutionTrace]:
        captured["raw_output_root"] = kwargs["raw_output_root"]
        return [
            ExecutionTrace(
                task_id=kwargs["instance_ids"][0],
                iteration=0,
                reward=1.0,
                status="ok",
                run_id=kwargs["run_id"],
            )
        ]

    monkeypatch.setattr(swebench, "run_swebench_harness", run_harness)
    monkeypatch.setattr(swebench, "build_swebench_traces", build_traces)

    executor = SWEbenchExecutorAgent(
        model="test-model",
        benchmark_repo=repo,  # type: ignore[arg-type]
        patch_generator=generator,
        artifact_root=tmp_path / "artifacts",
        injected_skill_name="executor",
        project_root=tmp_path,
        timeout=30,
        max_workers=1,
        run_id_prefix="run",
    )

    trace = await executor.execute_task(
        TaskSpec(
            task_id="sympy__sympy-20590",
            instruction="planner instruction",
            iteration=3,
        ),
        ["# executor skill"],
    )

    assert trace.reward == 1.0
    assert trace.status == "ok"
    assert trace.token_usage.input_tokens == 11
    assert trace.token_usage.output_tokens == 22
    assert repo.injected_skill_text == "# executor skill"
    assert generator.calls[0]["workspace"] == repo.prepared_workspace
    assert generator.calls[0]["planner_instruction"] == "planner instruction"
    assert generator.calls[0]["executor_skill_text"] == "# executor skill"
    prediction = json.loads(Path(trace.harbor_paths["prediction_jsonl"]).read_text())
    assert prediction["instance_id"] == "sympy__sympy-20590"
    assert prediction["model_patch"] == patch
    assert prediction["model_name_or_path"] == "test-model"
    assert Path(trace.harbor_paths["patch_diff"]).read_text() == patch
    assert Path(trace.harbor_paths["generation_stdout"]).read_text() == raw_response
    assert Path(trace.harbor_paths["generation_stderr"]).read_text() == (
        "Extracted fenced diff block."
    )
    assert trace.harbor_paths["swebench_raw_output"] == str(
        tmp_path / "artifacts" / "swebench-harness" / "run-sympy__sympy-20590-iter0003"
    )
    assert "harbor_job" not in trace.harbor_paths
    assert "harbor_trial" not in trace.harbor_paths
    assert trace.harbor_paths["sandbox_stdout"] == "sandbox/stdout.txt"
    assert captured["harness_cwd"] == (
        tmp_path / "artifacts" / "swebench-harness" / "run-sympy__sympy-20590-iter0003"
    )
    assert captured["raw_output_root"] == captured["harness_cwd"]
    command = captured["harness_command"]
    assert isinstance(command, list)
    assert command[command.index("--report_dir") + 1] == "."
    assert "--instance_ids" in command


@pytest.mark.asyncio
async def test_swebench_executor_evaluates_empty_diff_as_unresolved(
    tmp_path,
    monkeypatch,
):
    raw_response = "I cannot produce a patch."
    model_patch, notes = normalize_swebench_model_patch(raw_response)
    repo = _FakeSWEbenchRepo()
    generator = _FakePatchGenerator(
        SWEbenchPatchGeneration(
            raw_response=raw_response,
            model_patch=model_patch,
            normalization_notes=notes,
        )
    )
    harness_called = False

    def run_harness(**kwargs) -> swebench.SWEbenchHarnessRun:
        nonlocal harness_called
        harness_called = True
        return swebench.SWEbenchHarnessRun(
            command=kwargs["command"],
            returncode=0,
            stdout="harness ok",
            stderr="",
            duration_sec=1.0,
        )

    monkeypatch.setattr(swebench, "run_swebench_harness", run_harness)
    monkeypatch.setattr(
        swebench,
        "build_swebench_traces",
        lambda **kwargs: [
            ExecutionTrace(
                task_id=kwargs["instance_ids"][0],
                iteration=0,
                reward=0.0,
                status="ok",
                run_id=kwargs["run_id"],
            )
        ],
    )

    executor = SWEbenchExecutorAgent(
        model="test-model",
        benchmark_repo=repo,  # type: ignore[arg-type]
        patch_generator=generator,
        artifact_root=tmp_path / "artifacts",
        injected_skill_name="executor",
        project_root=tmp_path,
        timeout=30,
        max_workers=1,
        run_id_prefix="run",
    )

    trace = await executor.execute_task(
        TaskSpec(task_id="sympy__sympy-20590", instruction="fix it"),
        [],
    )

    prediction = json.loads(Path(trace.harbor_paths["prediction_jsonl"]).read_text())
    assert prediction["model_patch"] == ""
    assert Path(trace.harbor_paths["patch_diff"]).read_text() == ""
    assert Path(trace.harbor_paths["generation_stdout"]).read_text() == raw_response
    assert (
        "No unified diff found"
        in Path(trace.harbor_paths["generation_stderr"]).read_text()
    )
    assert harness_called is True
    assert trace.reward == 0.0
    assert trace.status == "ok"


@pytest.mark.asyncio
async def test_swebench_executor_patch_generation_failure_is_env_failure(
    tmp_path,
    monkeypatch,
):
    repo = _FakeSWEbenchRepo()
    generator = _FakePatchGenerator(exc=RuntimeError("llm unavailable"))
    harness_called = False

    def run_harness(**kwargs) -> swebench.SWEbenchHarnessRun:
        nonlocal harness_called
        harness_called = True
        raise AssertionError("harness should not run after patch generation failure")

    monkeypatch.setattr(swebench, "run_swebench_harness", run_harness)

    executor = SWEbenchExecutorAgent(
        model="test-model",
        benchmark_repo=repo,  # type: ignore[arg-type]
        patch_generator=generator,
        artifact_root=tmp_path / "artifacts",
        injected_skill_name="executor",
        project_root=tmp_path,
        timeout=30,
        max_workers=1,
        run_id_prefix="run",
    )

    trace = await executor.execute_task(
        TaskSpec(task_id="sympy__sympy-20590", instruction="fix it"),
        [],
    )

    assert trace.status == "env_failure"
    assert trace.error_kind == "patch_generation_failed"
    assert trace.reward is None
    assert trace.stderr == "llm unavailable"
    assert trace.harbor_paths["generation_workspace"] == str(repo.prepared_workspace)
    assert Path(trace.harbor_paths["generation_stdout"]).read_text() == ""
    assert (
        "patch generation failed: llm unavailable"
        in Path(trace.harbor_paths["generation_stderr"]).read_text()
    )
    assert "prediction_jsonl" not in trace.harbor_paths
    assert harness_called is False


def test_prepare_local_swebench_sandbox_copies_checkout_and_metadata(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()
    (workspace / "demo.py").write_text("print('old')\n")
    (workspace / "instruction.md").write_text("fix it\n")

    sandbox = prepare_local_swebench_sandbox(workspace)

    assert sandbox.source_workspace == workspace
    assert Path(sandbox.artifacts["sandbox_workspace"]) == sandbox.sandbox
    assert (sandbox.sandbox / ".git").exists()
    assert (sandbox.sandbox / "instruction.md").read_text() == "fix it\n"
    assert (sandbox.sandbox / "demo.py").read_text() == "print('old')\n"
    metadata = json.loads(Path(sandbox.artifacts["sandbox_metadata"]).read_text())
    assert metadata["source_checkout"] == str(workspace)
    assert metadata["sandbox_checkout"] == str(sandbox.sandbox)
    assert metadata["workspace_hint"] == "/workspace"
    assert metadata["git_present"] is True
    assert metadata["instruction_present"] is True


def test_capture_swebench_sandbox_diff_uses_swebench_exclusions(
    tmp_path,
    monkeypatch,
):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / ".git").mkdir()
    (sandbox / "instruction.md").write_text("fix it\n")
    patch = "diff --git a/demo.py b/demo.py\n+print('fixed')\n"
    calls: list[list[str]] = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["git", "add"]:
            return subprocess.CompletedProcess(command, 0, "", "")
        if command[:2] == ["git", "diff"]:
            return subprocess.CompletedProcess(command, 0, patch, "")
        raise AssertionError(f"unexpected command: {command}")

    monkeypatch.setattr(
        "mediated_coevo.agents.swebench_patch_generator.subprocess.run",
        run,
    )

    assert capture_swebench_sandbox_diff(sandbox) == patch
    assert calls[0][:2] == ["git", "add"]
    assert calls[1] == [
        "git",
        "diff",
        "--binary",
        "HEAD",
        "--",
        ".",
        ":!instruction.md",
        ":!environment/skills/**",
    ]


def test_prepare_local_swebench_sandbox_reports_missing_git(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "instruction.md").write_text("fix it\n")

    sandbox = prepare_local_swebench_sandbox(workspace)

    assert "missing .git" in Path(sandbox.artifacts["sandbox_stderr"]).read_text()
    metadata = json.loads(Path(sandbox.artifacts["sandbox_metadata"]).read_text())
    assert metadata["git_present"] is False
    assert metadata["instruction_present"] is True


def test_prepare_local_swebench_sandbox_reports_missing_instruction(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / ".git").mkdir()

    sandbox = prepare_local_swebench_sandbox(workspace)

    assert (
        "missing instruction.md"
        in Path(sandbox.artifacts["sandbox_stderr"]).read_text()
    )
    metadata = json.loads(Path(sandbox.artifacts["sandbox_metadata"]).read_text())
    assert metadata["git_present"] is True
    assert metadata["instruction_present"] is False
