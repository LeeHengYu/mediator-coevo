"""Standalone SWE-bench discovery and smoke-evaluation helpers."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from mediated_coevo.analysis.reporting import build_score_summary, write_score_summary
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.trace import ExecutionTrace, TraceStatus

DEFAULT_SWEBENCH_DATASET = "SWE-bench/SWE-bench_Lite"
DEFAULT_SWEBENCH_SPLIT = "test"
DEFAULT_SWEBENCH_INSTANCE_ID = "sympy__sympy-20590"
DEFAULT_SWEBENCH_OUTPUT_DIR = Path("data/swebench-evals")
SWEBENCH_VERIFIER_TYPE = "swebench_harness"


class SWEbenchDependencyError(RuntimeError):
    """Raised when the official SWE-bench package is unavailable."""


@dataclass(frozen=True, slots=True)
class SWEbenchInstanceInfo:
    """Small display projection for a SWE-bench dataset instance."""

    instance_id: str
    repo: str | None = None
    version: str | None = None
    created_at: str | None = None


@dataclass(frozen=True, slots=True)
class SWEbenchHarnessRun:
    """Result of invoking the official SWE-bench evaluation harness."""

    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    duration_sec: float


@dataclass(frozen=True, slots=True)
class SWEbenchTask:
    """Resolved SWE-bench instance used by the medcoevo executor."""

    task_id: str
    instance: dict[str, Any]
    instruction: str
    task_config: dict[str, Any]
    repo: str
    base_commit: str


class SWEbenchRepository:
    """Resolve SWE-bench instances and prepare patch-generation workspaces."""

    def __init__(self, *, dataset_name: str, split: str) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self._cache: dict[str, SWEbenchTask] = {}

    def resolve(self, task_id: str) -> SWEbenchTask:
        """Return the dataset-backed task for one SWE-bench instance ID."""
        cached = self._cache.get(task_id)
        if cached is not None:
            return cached

        rows = load_swebench_dataset(self.dataset_name, self.split, [task_id])
        if not rows:
            raise FileNotFoundError(
                f"SWE-bench instance {task_id!r} not found in "
                f"{self.dataset_name} split {self.split!r}"
            )

        task = self._task_from_instance(rows[0], task_id)
        self._cache[task_id] = task
        return task

    def prepare_patch_generation_workspace(
        self,
        *,
        task: SWEbenchTask,
        destination_root: Path,
        planner_instruction: str,
        injected_skill_text: str | None,
        injected_skill_name: str,
    ) -> Path:
        """Materialize a clean repo checkout and inject generation-only inputs."""
        run_dir = destination_root / task.task_id / f"run-{uuid.uuid4().hex[:8]}"
        run_dir.parent.mkdir(parents=True, exist_ok=True)

        clone_url = f"https://github.com/{task.repo}.git"
        _run_git(["clone", clone_url, str(run_dir)], cwd=None)
        _run_git(["checkout", task.base_commit], cwd=run_dir)

        (run_dir / "instruction.md").write_text(planner_instruction)
        if injected_skill_text:
            skill_dir = run_dir / "environment" / "skills" / injected_skill_name
            skill_dir.mkdir(parents=True, exist_ok=True)
            (skill_dir / "SKILL.md").write_text(injected_skill_text)
        return run_dir

    def capture_model_patch(self, workspace: Path) -> str:
        """Return the repo diff to submit as a SWE-bench ``model_patch``."""
        _run_git(["add", "--intent-to-add", "."], cwd=workspace)
        return _run_git(
            [
                "diff",
                "--binary",
                "HEAD",
                "--",
                ".",
                ":!instruction.md",
                ":!environment/skills/**",
            ],
            cwd=workspace,
        ).stdout

    @staticmethod
    def _task_from_instance(
        instance: dict[str, Any],
        requested_task_id: str,
    ) -> SWEbenchTask:
        task_id = str(instance.get("instance_id") or requested_task_id)
        repo = _required_string(instance, "repo", task_id)
        base_commit = _required_string(instance, "base_commit", task_id)
        instruction = build_swebench_instruction(instance)
        return SWEbenchTask(
            task_id=task_id,
            instance=instance,
            instruction=instruction,
            task_config={
                "metadata": {
                    "repo": repo,
                    "version": _optional_string(instance.get("version")),
                    "base_commit": base_commit,
                    "expected_reward_range": [0.0, 1.0],
                },
                "verifier": {"type": SWEBENCH_VERIFIER_TYPE},
            },
            repo=repo,
            base_commit=base_commit,
        )


@dataclass(frozen=True, slots=True)
class SWEbenchPrediction:
    """One SWE-bench prediction row."""

    instance_id: str
    model_patch: str
    model_name_or_path: str


def ensure_swebench_available() -> None:
    """Fail early with a clear install hint when ``swebench`` is absent."""
    if importlib.util.find_spec("swebench") is not None:
        return
    raise SWEbenchDependencyError(
        "The official SWE-bench package is not installed. "
        "Install project dependencies or run with `uv run --with swebench ...`."
    )


def parse_swebench_instance_ids(raw_values: list[str] | None) -> list[str]:
    """Parse repeatable and comma-separated ``--instance-id`` values."""
    if not raw_values:
        return [DEFAULT_SWEBENCH_INSTANCE_ID]

    instance_ids: list[str] = []
    seen: set[str] = set()
    for raw_value in raw_values:
        for candidate in raw_value.split(","):
            instance_id = candidate.strip()
            if not instance_id:
                continue
            if instance_id not in seen:
                instance_ids.append(instance_id)
                seen.add(instance_id)

    if not instance_ids:
        raise ValueError("at least one SWE-bench instance ID is required")
    return instance_ids


def build_swebench_instruction(instance: dict[str, Any]) -> str:
    """Build the issue prompt used for SWE-bench patch generation."""
    issue_text = str(instance.get("problem_statement") or "").strip()
    hints_text = str(instance.get("hints_text") or "").strip()
    parts = ["# SWE-bench Issue"]
    if issue_text:
        parts.append(issue_text)
    if hints_text:
        parts.extend(["# Hints", hints_text])
    parts.append(
        "Modify the repository to resolve the issue. The final answer should "
        "leave the working tree containing only the code changes needed for "
        "the patch."
    )
    return "\n\n".join(parts)


def list_swebench_instances(
    *,
    dataset_name: str,
    split: str,
    limit: int,
    repo_filter: str | None = None,
) -> list[SWEbenchInstanceInfo]:
    """Return valid instance IDs from a SWE-bench dataset split."""
    if limit < 1:
        raise ValueError("--limit must be at least 1")

    dataset = load_swebench_dataset(dataset_name, split)
    rows = []
    normalized_filter = repo_filter.lower() if repo_filter else None
    for raw_instance in dataset:
        repo = _optional_string(raw_instance.get("repo"))
        if normalized_filter and normalized_filter not in (repo or "").lower():
            continue
        rows.append(
            SWEbenchInstanceInfo(
                instance_id=str(raw_instance["instance_id"]),
                repo=repo,
                version=_optional_string(raw_instance.get("version")),
                created_at=_optional_string(raw_instance.get("created_at")),
            )
        )
        if len(rows) >= limit:
            break
    return rows


def resolve_swebench_instance_ids(
    *,
    dataset_name: str,
    split: str,
    raw_instance_ids: list[str] | None,
    limit: int | None = None,
    repo_filter: str | None = None,
) -> list[str]:
    """Resolve and validate SWE-bench instance IDs before running the harness."""
    if raw_instance_ids:
        instance_ids = parse_swebench_instance_ids(raw_instance_ids)
        loaded_instances = load_swebench_dataset(dataset_name, split, instance_ids)
        loaded_ids = {
            str(instance.get("instance_id"))
            for instance in loaded_instances
            if instance.get("instance_id") is not None
        }
        missing_ids = [
            instance_id for instance_id in instance_ids if instance_id not in loaded_ids
        ]
        if missing_ids:
            raise ValueError(
                "SWE-bench instances not found in "
                f"{dataset_name} split {split!r}: {missing_ids}"
            )
        return instance_ids

    if limit is not None:
        instances = list_swebench_instances(
            dataset_name=dataset_name,
            split=split,
            limit=limit,
            repo_filter=repo_filter,
        )
        if not instances:
            raise ValueError("no SWE-bench instances matched the provided filters")
        return [instance.instance_id for instance in instances]

    instance_ids = [DEFAULT_SWEBENCH_INSTANCE_ID]
    load_swebench_dataset(dataset_name, split, instance_ids)
    return instance_ids


def load_swebench_dataset(
    dataset_name: str,
    split: str,
    instance_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Load a SWE-bench dataset split through the official package."""
    ensure_swebench_available()
    from swebench.harness.utils import load_swebench_dataset as load_dataset

    return list(load_dataset(dataset_name, split, instance_ids))


def validate_modal_credentials(home: Path | None = None) -> None:
    """Check the Modal credentials file expected by the SWE-bench harness."""
    modal_config_path = (home or Path.home()) / ".modal.toml"
    if modal_config_path.exists():
        return
    raise RuntimeError(
        "~/.modal.toml not found. Run `modal token new` before using Modal."
    )


def build_swebench_harness_command(
    *,
    dataset_name: str,
    split: str,
    instance_ids: list[str],
    predictions_path: str,
    run_id: str,
    timeout: int,
    max_workers: int,
    report_dir: str | None = None,
) -> list[str]:
    """Build the official SWE-bench Modal harness command."""
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        dataset_name,
        "--split",
        split,
        "--predictions_path",
        predictions_path,
        "--run_id",
        run_id,
        "--timeout",
        str(timeout),
        "--max_workers",
        str(max_workers),
    ]
    if report_dir is not None:
        command.extend(["--report_dir", report_dir])
    command.extend(["--instance_ids", *instance_ids, "--modal", "true"])
    return command


def write_prediction_jsonl(
    *,
    prediction: SWEbenchPrediction,
    path: Path,
) -> Path:
    """Write a single prediction JSONL file for the official harness."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps(_prediction_row(prediction), sort_keys=True))
        f.write("\n")
    return path


def write_predictions_jsonl(
    *,
    predictions: list[SWEbenchPrediction],
    path: Path,
) -> Path:
    """Write an aggregate predictions JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for prediction in predictions:
            f.write(json.dumps(_prediction_row(prediction), sort_keys=True))
            f.write("\n")
    return path


def run_swebench_harness(
    *,
    command: list[str],
    cwd: Path,
    stream_output: bool = False,
) -> SWEbenchHarnessRun:
    """Run the official SWE-bench harness and capture command output."""
    if stream_output:
        return _run_swebench_harness_streaming(command=command, cwd=cwd)

    started = time.time()
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    return SWEbenchHarnessRun(
        command=command,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        duration_sec=time.time() - started,
    )


def _run_swebench_harness_streaming(
    *,
    command: list[str],
    cwd: Path,
) -> SWEbenchHarnessRun:
    started = time.time()
    proc = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    threads = [
        threading.Thread(
            target=_collect_stream,
            args=(proc.stdout, sys.stdout, stdout_chunks),
            daemon=True,
        ),
        threading.Thread(
            target=_collect_stream,
            args=(proc.stderr, sys.stderr, stderr_chunks),
            daemon=True,
        ),
    ]
    for thread in threads:
        thread.start()
    returncode = proc.wait()
    for thread in threads:
        thread.join()
    return SWEbenchHarnessRun(
        command=command,
        returncode=returncode,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
        duration_sec=time.time() - started,
    )


def _collect_stream(
    source: TextIO | None,
    sink: TextIO,
    chunks: list[str],
) -> None:
    if source is None:
        return
    for line in source:
        chunks.append(line)
        sink.write(line)
        sink.flush()


def build_swebench_traces(
    *,
    instance_ids: list[str],
    run_id: str,
    project_root: Path,
    harness_run: SWEbenchHarnessRun,
    raw_output_root: Path | None = None,
) -> list[ExecutionTrace]:
    """Convert SWE-bench per-instance reports into ``ExecutionTrace`` objects."""
    report_root = raw_output_root or project_root
    return [
        _trace_from_report(
            instance_id=instance_id,
            run_id=run_id,
            report_root=report_root,
            harness_run=harness_run,
        )
        for instance_id in instance_ids
    ]


def write_swebench_eval_outputs(
    *,
    traces: list[ExecutionTrace],
    output_dir: Path,
    run_id: str,
) -> tuple[Path, Path]:
    """Write normalized trace JSONL and score summary outputs."""
    run_output_dir = output_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    traces_path = run_output_dir / "traces.jsonl"
    with open(traces_path, "w") as f:
        for trace in traces:
            f.write(trace.model_dump_json())
            f.write("\n")

    records = [
        IterationRecord(
            iteration=trace.iteration,
            task_id=trace.task_id,
            execution_trace=trace,
            reward=trace.reward,
            duration_sec=trace.duration_sec,
            run_id=run_id,
            expected_reward_range=(0.0, 1.0),
            verifier_type=SWEBENCH_VERIFIER_TYPE,
            total_tokens=(
                trace.token_usage.input_tokens + trace.token_usage.output_tokens
            ),
        )
        for trace in traces
    ]
    summary_path = run_output_dir / "summary.json"
    write_score_summary(build_score_summary(records), summary_path)
    return traces_path, summary_path


def write_swebench_smoke_outputs(
    *,
    traces: list[ExecutionTrace],
    output_dir: Path,
    run_id: str,
) -> tuple[Path, Path]:
    """Backward-compatible name for normalized SWE-bench outputs."""
    return write_swebench_eval_outputs(
        traces=traces,
        output_dir=output_dir,
        run_id=run_id,
    )


def _trace_from_report(
    *,
    instance_id: str,
    run_id: str,
    report_root: Path,
    harness_run: SWEbenchHarnessRun,
) -> ExecutionTrace:
    report_path = _find_instance_report(
        report_root=report_root,
        run_id=run_id,
        instance_id=instance_id,
    )
    if report_path is None:
        aggregate_trace = _trace_from_aggregate_report(
            instance_id=instance_id,
            run_id=run_id,
            report_root=report_root,
            harness_run=harness_run,
            instance_report_path=None,
            instance_report_error=None,
        )
        if aggregate_trace is not None:
            return aggregate_trace
        return _failure_trace(
            instance_id=instance_id,
            run_id=run_id,
            harness_run=harness_run,
            status="env_failure",
            error_kind="harness_failed"
            if harness_run.returncode != 0
            else "report_missing",
            error_detail=(
                "No SWE-bench report.json found under "
                f"{report_root / 'logs' / 'run_evaluation' / run_id} "
                f"for {instance_id}."
            ),
        )

    try:
        report_text = report_path.read_text()
        if not report_text.strip():
            raise ValueError(f"empty report.json at {report_path}")
        report = json.loads(report_text)
        instance_report = report[instance_id]
        resolved = bool(instance_report["resolved"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        aggregate_trace = _trace_from_aggregate_report(
            instance_id=instance_id,
            run_id=run_id,
            report_root=report_root,
            harness_run=harness_run,
            instance_report_path=report_path,
            instance_report_error=str(exc),
        )
        if aggregate_trace is not None:
            return aggregate_trace
        return _failure_trace(
            instance_id=instance_id,
            run_id=run_id,
            harness_run=harness_run,
            status="parse_error",
            error_kind="report_parse_failed",
            error_detail=str(exc),
        )

    return ExecutionTrace(
        task_id=instance_id,
        iteration=0,
        stdout=harness_run.stdout,
        stderr=harness_run.stderr,
        exit_code=harness_run.returncode,
        test_results=instance_report,
        reward=1.0 if resolved else 0.0,
        status="ok",
        duration_sec=harness_run.duration_sec,
        run_id=run_id,
        harbor_paths={"swebench_report": str(report_path)},
        harbor_metadata={"verifier_type": SWEBENCH_VERIFIER_TYPE},
    )


def _trace_from_aggregate_report(
    *,
    instance_id: str,
    run_id: str,
    report_root: Path,
    harness_run: SWEbenchHarnessRun,
    instance_report_path: Path | None,
    instance_report_error: str | None,
) -> ExecutionTrace | None:
    aggregate_report_path = _find_aggregate_report(
        report_root=report_root,
        run_id=run_id,
    )
    if aggregate_report_path is None:
        return None

    try:
        aggregate_report = json.loads(aggregate_report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return _failure_trace(
            instance_id=instance_id,
            run_id=run_id,
            harness_run=harness_run,
            status="parse_error",
            error_kind="aggregate_report_parse_failed",
            error_detail=str(exc),
        )
    if not isinstance(aggregate_report, dict):
        return _failure_trace(
            instance_id=instance_id,
            run_id=run_id,
            harness_run=harness_run,
            status="parse_error",
            error_kind="aggregate_report_parse_failed",
            error_detail=f"Expected object, got {type(aggregate_report).__name__}",
        )

    outcome = None
    for field, candidate_outcome in (
        ("resolved_ids", "resolved"),
        ("unresolved_ids", "unresolved"),
        ("empty_patch_ids", "empty_patch"),
        ("error_ids", "error"),
        ("incomplete_ids", "incomplete"),
    ):
        values = aggregate_report.get(field)
        if isinstance(values, list) and instance_id in values:
            outcome = candidate_outcome
            break

    if outcome is None:
        return _failure_trace(
            instance_id=instance_id,
            run_id=run_id,
            harness_run=harness_run,
            status="env_failure",
            error_kind="aggregate_report_missing_outcome",
            error_detail=(
                f"Aggregate SWE-bench report has no outcome for {instance_id}."
            ),
        )
    if outcome == "incomplete":
        return _failure_trace(
            instance_id=instance_id,
            run_id=run_id,
            harness_run=harness_run,
            status="env_failure",
            error_kind="aggregate_report_incomplete",
            error_detail=(
                f"Aggregate SWE-bench report marks {instance_id} incomplete."
            ),
        )

    resolved = outcome == "resolved"
    test_results: dict[str, Any] = {
        "resolved": resolved,
        "aggregate_outcome": outcome,
        "aggregate_report": aggregate_report,
    }
    if instance_report_error is not None:
        test_results["instance_report_error"] = instance_report_error

    harbor_paths = {"swebench_aggregate_report": str(aggregate_report_path)}
    if instance_report_path is not None:
        harbor_paths["swebench_report"] = str(instance_report_path)

    return ExecutionTrace(
        task_id=instance_id,
        iteration=0,
        stdout=harness_run.stdout,
        stderr=harness_run.stderr,
        exit_code=harness_run.returncode,
        test_results=test_results,
        reward=1.0 if resolved else 0.0,
        status="ok",
        duration_sec=harness_run.duration_sec,
        run_id=run_id,
        harbor_paths=harbor_paths,
        harbor_metadata={"verifier_type": SWEBENCH_VERIFIER_TYPE},
    )


def _failure_trace(
    *,
    instance_id: str,
    run_id: str,
    harness_run: SWEbenchHarnessRun,
    status: TraceStatus,
    error_kind: str,
    error_detail: str,
) -> ExecutionTrace:
    return ExecutionTrace(
        task_id=instance_id,
        iteration=0,
        stdout=harness_run.stdout,
        stderr=harness_run.stderr,
        exit_code=harness_run.returncode,
        status=status,
        error_kind=error_kind,
        error_detail=error_detail,
        duration_sec=harness_run.duration_sec,
        run_id=run_id,
        harbor_metadata={"verifier_type": SWEBENCH_VERIFIER_TYPE},
    )


def _find_instance_report(
    *,
    report_root: Path,
    run_id: str,
    instance_id: str,
) -> Path | None:
    run_root = report_root / "logs" / "run_evaluation" / run_id
    candidates = sorted(run_root.glob(f"*/{instance_id}/report.json"))
    if not candidates:
        return None
    return candidates[0]


def _find_aggregate_report(
    *,
    report_root: Path,
    run_id: str,
) -> Path | None:
    suffix = f".{run_id}.json"
    candidates = [
        path
        for path in report_root.iterdir()
        if path.is_file() and path.name.endswith(suffix)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.name)[0]


def _prediction_row(prediction: SWEbenchPrediction) -> dict[str, str]:
    return {
        "instance_id": prediction.instance_id,
        "model_patch": prediction.model_patch,
        "model_name_or_path": prediction.model_name_or_path,
    }


def _required_string(
    instance: dict[str, Any],
    field: str,
    task_id: str,
) -> str:
    value = _optional_string(instance.get(field))
    if value is None:
        raise FileNotFoundError(
            f"SWE-bench instance {task_id!r} is missing required field {field!r}"
        )
    return value


def _run_git(args: list[str], *, cwd: Path | None) -> subprocess.CompletedProcess[str]:
    command = ["git", *args]
    completed = subprocess.run(
        command,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        location = str(cwd) if cwd is not None else "."
        raise RuntimeError(
            f"git command failed in {location}: {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\n\nstderr:\n{completed.stderr}"
        )
    return completed


def _optional_string(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
