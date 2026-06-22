#!/usr/bin/env python3
"""One-time recovery for interrupted env-failure runs in this experiment."""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
import time
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXPERIMENT_DIR.parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from mediated_coevo.analysis.judge_rewards import (  # noqa: E402
    JudgeRewardRecord,
    append_judge_reward_record,
    judge_reward_metadata,
)
from mediated_coevo.analysis.metrics import metric_row  # noqa: E402
from mediated_coevo.analysis.reporting import (  # noqa: E402
    build_score_summary,
    write_score_summary,
)
from mediated_coevo.benchmarks.skillflow import (  # noqa: E402
    HarborRunResult,
    HarborRunner,
    SkillFlowRepository,
    parse_skillflow_execution_trace,
)
from mediated_coevo.core.config import Config  # noqa: E402
from mediated_coevo.experiment.records import (  # noqa: E402
    build_iteration_record,
    task_metadata_fields,
)
from mediated_coevo.experiment.runtime_factory import (  # noqa: E402
    build_benchmark_repo,
    build_experiment_runtime,
)
from mediated_coevo.models.iteration import IterationRecord  # noqa: E402
from mediated_coevo.models.report import MediatorReport  # noqa: E402
from mediated_coevo.models.task import TaskSpec  # noqa: E402
from mediated_coevo.models.trace import ExecutionTrace  # noqa: E402
from mediated_coevo.runtime.token_budget import TokenBudgetEvent  # noqa: E402


FAILED_ROWS = [
    {
        "task_id": "HWPX-Document-Automation/hwpx-renewal-playbook-update",
        "iteration": 2,
        "workspace": "benchmarks/HWPX-Document-Automation_hwpx-renewal-playbook-update/run-66c26efb",
        "completed_job": "jobs/2026-06-19__13-39-24",
        "completed_trial": "jobs/2026-06-19__13-39-24/run-66c26efb__BVvnBpX",
    },
    {
        "task_id": "HWPX-Document-Automation/hwpx-safety-audit-brief",
        "iteration": 2,
        "workspace": "benchmarks/HWPX-Document-Automation_hwpx-safety-audit-brief/run-ee8d3ab6",
    },
    {
        "task_id": "HWPX-Document-Automation/hwpx-supplier-contact-sheet",
        "iteration": 2,
        "workspace": "benchmarks/HWPX-Document-Automation_hwpx-supplier-contact-sheet/run-57248d9c",
    },
]

MISSING_ROWS = [
    {
        "task_id": "HWPX-Document-Automation/hwpx-training-feedback",
        "iteration": 2,
    },
]

CONTEXT_FIELDS_TO_KEEP = {
    "condition",
    "skill_version",
    "skill_hashes",
    "planner_artifact_ids",
    "planner_cited_artifact_ids",
    "planner_context_artifact_ids",
    "diffusion_enabled",
    "diffusion_policy",
    "transfer_context_kind",
    "transfer_context_source_count",
    "transfer_context_tokens",
    "same_task_prior_tokens",
    "same_task_prior_entry_count",
    "planner_prior_context_tokens",
    "total_planner_prior_context_tokens",
    "max_planner_context_tokens",
    "max_planner_context_chars",
    "planner_context_token_headroom",
    "planner_context_char_headroom",
    "context_budget_violation",
    "compacted_diffusion_artifact_ids",
    "dropped_for_budget_artifact_ids",
    "task_family",
    "task_name",
    "domain",
    "task_tags",
    "current_directory",
    "config_file",
    "instruction_file",
    "dockerfile",
    "source_task_ids",
    "planner_warnings",
}


def load_config() -> Config:
    with (EXPERIMENT_DIR / "config.toml").open("rb") as fh:
        data = tomllib.load(fh)
    return Config(**data).normalize_models()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def backup(path: Path, stamp: str, audit: dict[str, Any]) -> None:
    if not path.exists():
        return
    backup_path = path.with_name(f"{path.name}.recovery-{stamp}.bak")
    shutil.copy2(path, backup_path)
    audit.setdefault("backups", []).append(str(backup_path.relative_to(EXPERIMENT_DIR)))


def find_metric_row(rows: list[dict[str, Any]], task_id: str, iteration: int) -> dict[str, Any]:
    for row in rows:
        if row.get("task_id") == task_id and row.get("iteration") == iteration:
            return row
    raise RuntimeError(f"missing metric row for {task_id} iter {iteration}")


def maybe_metric_row(
    rows: list[dict[str, Any]],
    task_id: str,
    iteration: int,
) -> dict[str, Any] | None:
    for row in rows:
        if row.get("task_id") == task_id and row.get("iteration") == iteration:
            return row
    return None


def previous_reward(rows: list[dict[str, Any]], task_id: str, before_iteration: int) -> float | None:
    reward = None
    for row in rows:
        if row.get("task_id") != task_id:
            continue
        if int(row.get("iteration", -1)) >= before_iteration:
            continue
        if row.get("verifier_status") in {"env_failure", "infra_failure", "error"}:
            continue
        if row.get("reward") is not None:
            reward = float(row["reward"])
    return reward


def previous_rewards_by_task(
    rows: list[dict[str, Any]],
    *,
    before_iteration: int,
) -> dict[str, float]:
    rewards: dict[str, float] = {}
    for row in rows:
        task_id = row.get("task_id")
        if not task_id or task_id == "__coevolution__":
            continue
        if int(row.get("iteration", -1)) >= before_iteration:
            continue
        if row.get("verifier_status") in {"env_failure", "infra_failure", "error"}:
            continue
        if row.get("reward") is not None:
            rewards[task_id] = float(row["reward"])
    return rewards


def extract_task_instruction(envelope: str) -> str:
    start = envelope.find("# Task Instruction")
    end = envelope.find("# Executor Policy")
    if start == -1 or end == -1 or end <= start:
        return envelope.strip()
    return envelope[start + len("# Task Instruction") : end].strip()


def load_task_config(task_workspace: Path) -> dict[str, Any]:
    with (task_workspace / "task.toml").open("rb") as fh:
        return tomllib.load(fh)


def task_metadata(task_config: dict[str, Any], task_workspace: Path) -> dict[str, Any]:
    meta = dict(task_config.get("metadata", {}) or {})
    meta.update(
        {
            "run_id": task_workspace.name,
            "current_directory": task_config.get("current_directory"),
            "config_file": task_config.get("config_file"),
            "instruction_file": task_config.get("instruction_file"),
            "dockerfile": task_config.get("dockerfile"),
        }
    )
    return meta


def parse_iso_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def completed_run_result(entry: dict[str, Any]) -> tuple[HarborRunResult | None, float | None]:
    job_name = entry.get("completed_job")
    trial_name = entry.get("completed_trial")
    if not job_name or not trial_name:
        return None, None
    job_dir = EXPERIMENT_DIR / job_name
    trial_dir = EXPERIMENT_DIR / trial_name
    if not (job_dir / "result.json").exists() or not (trial_dir / "result.json").exists():
        return None, None

    trial_result = load_json(trial_dir / "result.json")
    job_result = load_json(job_dir / "result.json")
    started = parse_iso_datetime(
        trial_result.get("started_at") or job_result.get("started_at")
    )
    finished = parse_iso_datetime(
        trial_result.get("finished_at") or job_result.get("finished_at")
    )
    duration = (finished - started).total_seconds() if started and finished else 0.0
    return (
        HarborRunResult(
            job_dir=job_dir,
            trial_dir=trial_dir,
            returncode=0,
            stdout="",
            stderr="",
        ),
        duration,
    )


def existing_judge_record(record_id: str) -> JudgeRewardRecord | None:
    judge_path = EXPERIMENT_DIR / "artifacts" / "judge_rewards.jsonl"
    for row in load_jsonl(judge_path):
        if row.get("record_id") != record_id:
            continue
        try:
            return JudgeRewardRecord.model_validate(row)
        except Exception:
            return None
    return None


def append_judge_reward_record_once(record: JudgeRewardRecord) -> None:
    if existing_judge_record(record.record_id) is not None:
        return
    append_judge_reward_record(EXPERIMENT_DIR, record)


def existing_mediator_report(task_id: str, iteration: int) -> MediatorReport | None:
    family, name = task_id.split("/", 1)
    report_dir = EXPERIMENT_DIR / "artifacts" / "reports" / family
    pattern = f"{name}_iter{iteration:04d}_*.json"
    for path in sorted(report_dir.glob(pattern), key=lambda item: item.stat().st_mtime, reverse=True):
        try:
            return MediatorReport.model_validate_json(path.read_text())
        except Exception:
            continue
    return None


def latest_reports_before_iteration(iteration: int) -> dict[str, MediatorReport]:
    reports: dict[str, MediatorReport] = {}
    reports_dir = EXPERIMENT_DIR / "artifacts" / "reports"
    for path in sorted(reports_dir.rglob("*.json")):
        try:
            report = MediatorReport.model_validate_json(path.read_text())
        except Exception:
            continue
        if report.iteration >= iteration or report.withheld:
            continue
        current = reports.get(report.task_id)
        if current is None or (report.iteration, report.timestamp) > (
            current.iteration,
            current.timestamp,
        ):
            reports[report.task_id] = report
    return reports


def existing_mediator_history_entry_id(task_id: str, iteration: int) -> str | None:
    history_path = EXPERIMENT_DIR / "history" / "history.jsonl"
    entry_id = None
    for row in load_jsonl(history_path):
        metadata = row.get("metadata") or {}
        if row.get("agent_role") != "mediator":
            continue
        if metadata.get("task_id") != task_id:
            continue
        if row.get("iteration") != iteration:
            continue
        entry_id = row.get("entry_id") or entry_id
    return entry_id


def metric_record_from_row(row: dict[str, Any]) -> IterationRecord:
    status = row.get("verifier_status") or "ok"
    if status == "task_failed":
        status = "ok"
    trace = ExecutionTrace(
        task_id=row["task_id"],
        iteration=int(row["iteration"]),
        reward=row.get("reward"),
        status=status,
    )
    return IterationRecord(
        iteration=int(row["iteration"]),
        task_id=row["task_id"],
        reward=row.get("reward"),
        skill_version=row.get("skill_version"),
        execution_trace=trace,
        total_tokens=int(row.get("total_tokens") or 0),
        transfer_context_tokens=row.get("transfer_context_tokens"),
        same_task_prior_tokens=row.get("same_task_prior_tokens"),
        total_planner_prior_context_tokens=row.get("total_planner_prior_context_tokens"),
        max_same_task_prior_tokens=row.get("max_same_task_prior_tokens") or 0,
        max_transfer_context_tokens=row.get("max_transfer_context_tokens") or 0,
        max_total_prior_context_tokens=row.get("max_total_prior_context_tokens") or 0,
        context_budget_violation=bool(row.get("context_budget_violation")),
        task_category=row.get("task_category"),
        task_difficulty=row.get("task_difficulty"),
        expected_reward_range=tuple(row["expected_reward_range"])
        if row.get("expected_reward_range") is not None
        else None,
        verifier_type=row.get("verifier_type"),
    )


def write_summary(metrics_rows: list[dict[str, Any]]) -> None:
    records = [
        metric_record_from_row(row)
        for row in metrics_rows
        if row.get("task_id") != "__coevolution__"
    ]
    summary = build_score_summary(records)

    judge_path = EXPERIMENT_DIR / "artifacts" / "judge_rewards.jsonl"
    judge_records = []
    for row in load_jsonl(judge_path):
        try:
            judge_records.append(JudgeRewardRecord.model_validate(row))
        except Exception:
            continue
    if judge_records:
        summary.judge_reward_summary = build_score_summary(
            [
                IterationRecord(
                    iteration=record.iteration,
                    task_id=record.task_id,
                    reward=record.judge_reward,
                    total_tokens=record.total_tokens,
                    execution_trace=ExecutionTrace(
                        task_id=record.task_id,
                        iteration=record.iteration,
                        reward=record.judge_reward,
                        status="ok",
                    ),
                    task_category=record.task_category,
                    task_difficulty=record.task_difficulty,
                    expected_reward_range=record.expected_reward_range,
                    verifier_type=record.verifier_type,
                )
                for record in judge_records
            ]
        )

    write_score_summary(summary, EXPERIMENT_DIR / "summary.json")


def replace_metric_row(
    rows: list[dict[str, Any]],
    task_id: str,
    iteration: int,
    replacement: dict[str, Any],
) -> None:
    for index, row in enumerate(rows):
        if row.get("task_id") == task_id and row.get("iteration") == iteration:
            rows[index] = replacement
            return
    raise RuntimeError(f"missing metric row for {task_id} iter {iteration}")


def upsert_metric_row(rows: list[dict[str, Any]], replacement: dict[str, Any]) -> None:
    task_id = replacement["task_id"]
    iteration = replacement["iteration"]
    for index, row in enumerate(rows):
        if row.get("task_id") == task_id and row.get("iteration") == iteration:
            rows[index] = replacement
            return

    insert_at = len(rows)
    for index, row in enumerate(rows):
        row_iteration = int(row.get("iteration", -1))
        if row_iteration > iteration:
            insert_at = index
            break
        if row_iteration == iteration and row.get("task_id") == "__coevolution__":
            insert_at = index
            break
    rows.insert(insert_at, replacement)


def seed_orchestrator_state(runtime: Any, metrics_rows: list[dict[str, Any]], iteration: int) -> None:
    reports = latest_reports_before_iteration(iteration)
    runtime.orchestrator._previous_report_by_task = dict(reports)
    runtime.orchestrator._released_cross_task_reports_by_task = dict(reports)
    runtime.orchestrator._staged_cross_task_reports_by_task.clear()
    runtime.orchestrator._previous_reward_by_task = previous_rewards_by_task(
        metrics_rows,
        before_iteration=iteration,
    )


async def recover_one(
    entry: dict[str, Any],
    *,
    config: Config,
    runner: HarborRunner,
    repository: SkillFlowRepository,
    runtime: Any,
    metrics_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    task_id = entry["task_id"]
    iteration = int(entry["iteration"])
    task_workspace = EXPERIMENT_DIR / entry["workspace"]
    old_row = find_metric_row(metrics_rows, task_id, iteration)

    envelope = (task_workspace / "instruction.md").read_text()
    executor_skill_path = EXPERIMENT_DIR / "skills" / "executor" / "SKILL.md"
    executor_skill = executor_skill_path.read_text() if executor_skill_path.exists() else ""
    task_config = load_task_config(task_workspace)
    metadata = task_metadata_fields(task_id=task_id, task_config=task_config)
    task_spec = TaskSpec(
        task_id=task_id,
        instruction=extract_task_instruction(envelope),
        skills_context=[executor_skill] if executor_skill else [],
        iteration=iteration,
    )

    runtime.orchestrator._drain_llm_token_events()

    run_result, cached_duration = completed_run_result(entry)
    if run_result is None:
        start = time.monotonic()
        run_result = await runner.run(task_workspace, config.models.executor)
        duration_sec = time.monotonic() - start
    else:
        duration_sec = cached_duration or 0.0
    trace = parse_skillflow_execution_trace(
        run_result=run_result,
        task_id=task_id,
        iteration=iteration,
        duration_sec=duration_sec,
    )
    trace.harbor_metadata = {
        **trace.harbor_metadata,
        **repository.executor_envelope_metadata(
            run_dir=task_workspace,
            executor_policy=executor_skill,
        ),
    }

    old_trace_path = EXPERIMENT_DIR / "artifacts" / "traces" / f"{task_id}_iter{iteration:04d}.json"
    trace_path = runtime.orchestrator.artifact_store.store_trace(trace, overwrite=True)
    assert trace_path == old_trace_path

    outcome_reward = None
    outcome_metadata = None
    judge_record = None
    record_id = (
        f"{config.judge.rubric_version}:{trace.run_id}:{task_id}:{iteration}"
        if trace.run_id
        else None
    )
    if record_id:
        judge_record = existing_judge_record(record_id)
    if judge_record is not None:
        outcome_reward = judge_record.judge_reward
        outcome_metadata = judge_reward_metadata(judge_record)
    elif trace.is_usable_feedback_signal:
        judge_record = await runtime.orchestrator._judge_evolution_reward(
            trace=trace,
            task_metadata=metadata,
            trace_path=trace_path,
        )
        if judge_record is not None:
            if not judge_record.metadata.get("judge_reward_fallback"):
                append_judge_reward_record_once(judge_record)
            outcome_reward = judge_record.judge_reward
            outcome_metadata = judge_reward_metadata(judge_record)

    report = existing_mediator_report(task_id, iteration)
    mediator_entry_id = existing_mediator_history_entry_id(task_id, iteration)
    planner_entry_id = None
    if report is None:
        report = await runtime.orchestrator.mediator.mediate_trace(
            config.experiment.condition_name,
            trace,
            task_spec,
        )
    if report is not None and mediator_entry_id is None:
        runtime.orchestrator.artifact_store.store_report(report)

        runtime.orchestrator.history_store.tag_outcome(
            task_id,
            trace,
            proposals=[],
            outcome_reward=outcome_reward,
            outcome_metadata=outcome_metadata,
        )
        mediator_entry_id, planner_entry_id = await runtime.orchestrator._record_history_entries(
            task_id=task_id,
            iteration=iteration,
            condition=config.experiment.condition_name,
            report=report,
            skill_update=None,
        )
    tagged_entry_ids = [mediator_entry_id, planner_entry_id]
    if any(tagged_entry_ids):
        runtime.orchestrator.history_store.tag_outcome(
            task_id,
            trace,
            proposals=[],
            entry_ids=tagged_entry_ids,
            outcome_reward=outcome_reward,
            outcome_metadata=outcome_metadata,
        )
    history_entry_ids = [entry_id for entry_id in tagged_entry_ids if entry_id]

    new_events = runtime.orchestrator._drain_llm_token_events()
    planner_events = [
        TokenBudgetEvent.model_validate(event)
        for event in old_row.get("llm_token_events", [])
        if str(event.get("label", "")).startswith("planner.")
    ]

    prior_reward = previous_reward(metrics_rows, task_id, iteration)
    previous_by_task = {task_id: prior_reward} if prior_reward is not None else {}
    record = build_iteration_record(
        task_id=task_id,
        iteration=iteration,
        condition=config.experiment.condition_name,
        duration_sec=duration_sec,
        task_spec=task_spec,
        trace=trace,
        report=report,
        skill_update=None,
        mediator_entry_id=mediator_entry_id,
        planner_entry_id=planner_entry_id,
        skill_hashes=old_row.get("skill_hashes") or {},
        task_metadata=metadata,
        llm_token_events=[*planner_events, *new_events],
        config=config,
        previous_reward_by_task=previous_by_task,
    )

    row = metric_row(record)
    for key in CONTEXT_FIELDS_TO_KEEP:
        if key in old_row:
            row[key] = old_row[key]
    replace_metric_row(metrics_rows, task_id, iteration, row)

    result = {
        "task_id": task_id,
        "iteration": iteration,
        "workspace": str(task_workspace.relative_to(EXPERIMENT_DIR)),
        "new_harbor_trial_path": trace.harbor_paths.get("trial"),
        "status": trace.status,
        "verifier_status": row.get("verifier_status"),
        "reward": row.get("reward"),
        "judge_reward": row.get("judge_reward"),
        "mediator_report_id": mediator_entry_id,
        "history_entry_ids": history_entry_ids,
    }
    audit.setdefault("recovered", []).append(result)
    return result


async def run_missing_one(
    entry: dict[str, Any],
    *,
    runtime: Any,
    metrics_rows: list[dict[str, Any]],
    audit: dict[str, Any],
) -> dict[str, Any]:
    task_id = entry["task_id"]
    iteration = int(entry["iteration"])
    existing = maybe_metric_row(metrics_rows, task_id, iteration)
    if existing is not None and existing.get("verifier_status") != "env_failure":
        result = {
            "task_id": task_id,
            "iteration": iteration,
            "status": "skipped_existing_metric",
            "verifier_status": existing.get("verifier_status"),
            "reward": existing.get("reward"),
        }
        audit.setdefault("missing_skipped", []).append(result)
        return result

    seed_orchestrator_state(runtime, metrics_rows, iteration)
    runtime.orchestrator._drain_llm_token_events()
    record = await runtime.orchestrator._run_iteration(task_id, iteration)
    row = metric_row(record)
    upsert_metric_row(metrics_rows, row)

    result = {
        "task_id": task_id,
        "iteration": iteration,
        "status": record.execution_trace.status if record.execution_trace else None,
        "verifier_status": row.get("verifier_status"),
        "reward": row.get("reward"),
        "judge_reward": row.get("judge_reward"),
        "harbor_trial_path": row.get("harbor_trial_path"),
        "mediator_report_id": row.get("mediator_report_id"),
        "history_entry_ids": row.get("history_entry_ids"),
    }
    audit.setdefault("missing_recovered", []).append(result)
    return result


async def main() -> None:
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    audit: dict[str, Any] = {
        "experiment_dir": str(EXPERIMENT_DIR),
        "started_at": datetime.now(UTC).isoformat(),
        "target_rows": FAILED_ROWS,
        "missing_rows": MISSING_ROWS,
    }

    config = load_config()
    metrics_path = EXPERIMENT_DIR / "metrics.jsonl"
    summary_path = EXPERIMENT_DIR / "summary.json"
    judge_path = EXPERIMENT_DIR / "artifacts" / "judge_rewards.jsonl"
    history_path = EXPERIMENT_DIR / "history" / "history.jsonl"

    backup(metrics_path, stamp, audit)
    backup(summary_path, stamp, audit)
    backup(judge_path, stamp, audit)
    backup(history_path, stamp, audit)
    for entry in FAILED_ROWS:
        trace_path = EXPERIMENT_DIR / "artifacts" / "traces" / f"{entry['task_id']}_iter{entry['iteration']:04d}.json"
        backup(trace_path, stamp, audit)

    metrics_rows = load_jsonl(metrics_path)

    benchmark_repo = build_benchmark_repo(PROJECT_ROOT, config)
    runtime = build_experiment_runtime(
        config=config,
        benchmark_repo=benchmark_repo,
        experiment_dir=EXPERIMENT_DIR,
        runtime_skills_dir=EXPERIMENT_DIR / "skills",
        condition_name=config.experiment.condition_name,
        remote_harbor_config=None,
    )
    runner = HarborRunner(
        jobs_dir=EXPERIMENT_DIR / config.executor_runtime.jobs_dir,
        timeout_sec=config.executor_runtime.harbor_timeout_sec,
        agent_setup_timeout_multiplier=config.executor_runtime.harbor_agent_setup_timeout_multiplier,
    )
    workspace_repository = SkillFlowRepository(
        root_dir=EXPERIMENT_DIR / "benchmarks",
        task_dirs=[],
    )

    for entry in FAILED_ROWS:
        row = maybe_metric_row(metrics_rows, entry["task_id"], int(entry["iteration"]))
        if row is None:
            raise RuntimeError(f"target row is missing: {entry['task_id']}")
        if row.get("verifier_status") != "env_failure":
            result = {
                "task_id": entry["task_id"],
                "iteration": int(entry["iteration"]),
                "status": "skipped_existing_metric",
                "verifier_status": row.get("verifier_status"),
                "reward": row.get("reward"),
            }
            audit.setdefault("recovered_skipped", []).append(result)
            print(json.dumps(result, sort_keys=True))
            continue
        result = await recover_one(
            entry,
            config=config,
            runner=runner,
            repository=workspace_repository,
            runtime=runtime,
            metrics_rows=metrics_rows,
            audit=audit,
        )
        print(json.dumps(result, sort_keys=True))

    for entry in MISSING_ROWS:
        result = await run_missing_one(
            entry,
            runtime=runtime,
            metrics_rows=metrics_rows,
            audit=audit,
        )
        print(json.dumps(result, sort_keys=True))

    write_jsonl(metrics_path, metrics_rows)
    write_summary(metrics_rows)

    audit["completed_at"] = datetime.now(UTC).isoformat()
    audit_path = EXPERIMENT_DIR / f"recovery-{stamp}.json"
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"wrote {audit_path}")


if __name__ == "__main__":
    asyncio.run(main())
