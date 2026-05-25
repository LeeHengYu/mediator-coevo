"""Post-run LLM judge reward annotation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mediated_coevo.analysis.reporting import (
    COEVOLUTION_TASK_ID,
    ExperimentScoreSummary,
    build_score_summary,
    write_score_summary,
)
from mediated_coevo.core.config import Config
from mediated_coevo.core.utils import as_optional_float, parse_json_object
from mediated_coevo.evolution.compactor import head_tail_text, trace_header_summary
from mediated_coevo.llm.client import LLMClient
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.judge import (
    JudgeAxisScores,
    JudgeCapFlags,
    JudgeLLMResponse,
    JudgeRewardRecord,
)
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.stores.history_store import HistoryStore

logger = logging.getLogger(__name__)

JUDGE_REWARDS_PATH = Path("artifacts") / "judge_rewards.jsonl"
JUDGE_SYSTEM_PROMPT = """\
You are an LLM-as-judge reward annotator.

Return ONLY a JSON object with exactly these fields:
- axis_scores: object with task_outcome, evidence_quality,
  skill_update_usefulness, token_efficiency, reflection_depth, each in [0, 1]
- flags: object with benchmark_gaming_or_obscured_failure,
  no_meaningful_progress, brittle_or_one_off_patch, unverifiable_outcome booleans
- confidence: number in [0, 1]
- rationale: concise evidence-grounded explanation
- flag_evidence: object mapping every true flag name to concrete evidence text

Do not compute the final scalar reward. Code will apply weights and caps.
"""

JUDGE_WEIGHTS = {
    "task_outcome": 0.50,
    "evidence_quality": 0.15,
    "skill_update_usefulness": 0.20,
    "token_efficiency": 0.10,
    "reflection_depth": 0.05,
}
CAP_VALUES = {
    "no_meaningful_progress": 0.20,
    "brittle_or_one_off_patch": 0.40,
    "unverifiable_outcome": 0.50,
}


class JudgeRewardAnnotationError(RuntimeError):
    """Raised when required judge reward annotation cannot complete."""


@dataclass(frozen=True)
class _JudgeCandidate:
    task_id: str
    iteration: int
    raw_reward: float
    run_id: str | None
    verifier_status: str | None
    trace_status: str | None
    total_tokens: int
    task_category: str | None
    task_difficulty: str | None
    expected_reward_range: tuple[float, float] | None
    verifier_type: str | None
    trace: ExecutionTrace | None
    trace_path: Path | None
    metrics_path: Path | None
    metrics_row: dict[str, Any]

    def record_id(self, rubric_version: str) -> str:
        run_id = self.run_id or "no-run-id"
        return f"{rubric_version}:{run_id}:{self.task_id}:{self.iteration}"


async def annotate_judge_rewards(
    *,
    data_dir: Path,
    config: Config,
    history_store: HistoryStore | None = None,
    llm_client: Any | None = None,
) -> list[JudgeRewardRecord]:
    """Annotate verifier-scored run outputs with required judge rewards."""
    judge_model = config.models.judge
    client = llm_client or LLMClient(model=judge_model)
    sidecar_path = data_dir / JUDGE_REWARDS_PATH
    existing_records = _load_existing_records(sidecar_path)
    records_by_id = {record.record_id: record for record in existing_records}

    new_records = []
    for candidate in _load_candidates(data_dir):
        record_id = candidate.record_id(config.judge.rubric_version)
        if record_id in records_by_id:
            continue
        try:
            judge_response = await _judge_candidate(
                candidate=candidate,
                config=config,
                llm_client=client,
            )
        except JudgeRewardAnnotationError as exc:
            logger.warning(
                "Judge response unavailable; falling back to verifier reward: "
                "task_id=%s iteration=%d error=%s",
                candidate.task_id,
                candidate.iteration,
                exc,
            )
            record = _fallback_record(
                candidate=candidate,
                data_dir=data_dir,
                judge_model=judge_model,
                rubric_version=config.judge.rubric_version,
                record_id=record_id,
                error=str(exc),
            )
        else:
            record = _record_from_judge_response(
                candidate=candidate,
                data_dir=data_dir,
                judge_model=judge_model,
                rubric_version=config.judge.rubric_version,
                record_id=record_id,
                response=judge_response,
            )
        records_by_id[record.record_id] = record
        new_records.append(record)

    if new_records:
        _append_records(sidecar_path, new_records)
    all_records = list(records_by_id.values())
    _write_judge_summary(data_dir=data_dir, judge_records=all_records)
    if history_store is not None:
        _annotate_history_entries(
            history_store=history_store,
            judge_records=all_records,
        )
    logger.info(
        "Judge reward annotation complete: data_dir=%s records=%d new=%d",
        data_dir,
        len(all_records),
        len(new_records),
    )
    return all_records


async def judge_reward_for_trace(
    *,
    trace: ExecutionTrace,
    config: Config,
    llm_client: Any | None,
    trace_path: Path | None = None,
    metrics_path: Path | None = None,
    metrics_row: dict[str, Any] | None = None,
    task_category: str | None = None,
    task_difficulty: str | None = None,
    expected_reward_range: tuple[float, float] | None = None,
    verifier_type: str | None = None,
    verifier_status: str | None = None,
    total_tokens: int | None = None,
) -> JudgeRewardRecord | None:
    """Return the evolution reward for one live trace.

    Online evolution uses this before contrastive reflection and candidate
    validation. Unusable traces return ``None``; usable traces fall back to the
    verifier reward if the judge client is unavailable or fails.
    """
    if trace.status != "ok" or trace.reward is None:
        return None

    judge_model = getattr(llm_client, "model", None) or config.models.judge
    candidate = _JudgeCandidate(
        task_id=trace.task_id,
        iteration=trace.iteration,
        raw_reward=trace.reward,
        run_id=trace.run_id,
        verifier_status=verifier_status or trace.status,
        trace_status=trace.status,
        total_tokens=(
            total_tokens
            if total_tokens is not None
            else trace.token_usage.input_tokens + trace.token_usage.output_tokens
        ),
        task_category=task_category,
        task_difficulty=task_difficulty,
        expected_reward_range=expected_reward_range,
        verifier_type=verifier_type,
        trace=trace,
        trace_path=trace_path,
        metrics_path=metrics_path,
        metrics_row=metrics_row or {},
    )
    record_id = candidate.record_id(config.judge.rubric_version)

    if llm_client is None:
        return _fallback_record(
            candidate=candidate,
            data_dir=None,
            judge_model=judge_model,
            rubric_version=config.judge.rubric_version,
            record_id=record_id,
            error="judge client unavailable",
            fallback_reason="judge_client_unavailable",
        )

    try:
        response = await _judge_candidate(
            candidate=candidate,
            config=config,
            llm_client=llm_client,
        )
    except JudgeRewardAnnotationError as exc:
        logger.warning(
            "Live judge reward unavailable; falling back to verifier reward: "
            "task_id=%s iteration=%d error=%s",
            trace.task_id,
            trace.iteration,
            exc,
        )
        return _fallback_record(
            candidate=candidate,
            data_dir=None,
            judge_model=judge_model,
            rubric_version=config.judge.rubric_version,
            record_id=record_id,
            error=str(exc),
        )

    return _record_from_judge_response(
        candidate=candidate,
        data_dir=None,
        judge_model=judge_model,
        rubric_version=config.judge.rubric_version,
        record_id=record_id,
        response=response,
    )


def judge_reward_metadata(record: JudgeRewardRecord) -> dict[str, Any]:
    """Return compact provenance for a judge-derived evolution reward."""
    reward_source = (
        "verifier_fallback"
        if record.metadata.get("judge_reward_fallback")
        else "judge"
    )
    return {
        "reward_source": reward_source,
        "verifier_reward": record.raw_reward,
        "judge_reward": record.judge_reward,
        "judge_base_reward": record.base_reward,
        "judge_reward_record_id": record.record_id,
        "judge_rubric_version": record.rubric_version,
        "judge_confidence": record.confidence,
        "judge_applied_cap": record.applied_cap,
    }


def append_judge_reward_record(data_dir: Path, record: JudgeRewardRecord) -> bool:
    """Append one judge reward sidecar row if it is not already present."""
    sidecar_path = data_dir / JUDGE_REWARDS_PATH
    existing_ids = {
        existing.record_id for existing in _load_existing_records(sidecar_path)
    }
    if record.record_id in existing_ids:
        return False
    _append_records(sidecar_path, [record])
    return True


def compute_judge_reward(response: JudgeLLMResponse) -> tuple[float, float, str | None]:
    """Return final judge reward, uncapped base reward, and applied cap."""
    scores = response.axis_scores
    base_reward = (
        JUDGE_WEIGHTS["task_outcome"] * scores.task_outcome
        + JUDGE_WEIGHTS["evidence_quality"] * scores.evidence_quality
        + JUDGE_WEIGHTS["skill_update_usefulness"] * scores.skill_update_usefulness
        + JUDGE_WEIGHTS["token_efficiency"] * scores.token_efficiency
        + JUDGE_WEIGHTS["reflection_depth"] * scores.reflection_depth
    )
    if response.flags.benchmark_gaming_or_obscured_failure:
        return 0.0, base_reward, "benchmark_gaming_or_obscured_failure"

    reward = base_reward
    applied_cap = None
    if response.flags.no_meaningful_progress and reward > CAP_VALUES["no_meaningful_progress"]:
        reward = CAP_VALUES["no_meaningful_progress"]
        applied_cap = "no_meaningful_progress"
    if response.flags.brittle_or_one_off_patch and reward > CAP_VALUES["brittle_or_one_off_patch"]:
        reward = CAP_VALUES["brittle_or_one_off_patch"]
        applied_cap = "brittle_or_one_off_patch"
    if response.flags.unverifiable_outcome and reward > CAP_VALUES["unverifiable_outcome"]:
        reward = CAP_VALUES["unverifiable_outcome"]
        applied_cap = "unverifiable_outcome"
    return reward, base_reward, applied_cap


def _record_from_judge_response(
    *,
    candidate: _JudgeCandidate,
    data_dir: Path | None,
    judge_model: str,
    rubric_version: str,
    record_id: str,
    response: JudgeLLMResponse,
) -> JudgeRewardRecord:
    judge_reward, base_reward, applied_cap = compute_judge_reward(response)
    return JudgeRewardRecord(
        record_id=record_id,
        task_id=candidate.task_id,
        iteration=candidate.iteration,
        run_id=candidate.run_id,
        raw_reward=candidate.raw_reward,
        judge_reward=judge_reward,
        base_reward=base_reward,
        applied_cap=applied_cap,
        axis_scores=response.axis_scores,
        flags=response.flags,
        confidence=response.confidence,
        rationale=response.rationale,
        flag_evidence=response.flag_evidence,
        judge_model=judge_model,
        rubric_version=rubric_version,
        trace_path=_relative_path(candidate.trace_path, data_dir),
        metrics_path=_relative_path(candidate.metrics_path, data_dir),
        verifier_status=candidate.verifier_status,
        trace_status=candidate.trace_status,
        total_tokens=candidate.total_tokens,
        task_category=candidate.task_category,
        task_difficulty=candidate.task_difficulty,
        expected_reward_range=candidate.expected_reward_range,
        verifier_type=candidate.verifier_type,
    )


def _fallback_record(
    *,
    candidate: _JudgeCandidate,
    data_dir: Path | None,
    judge_model: str,
    rubric_version: str,
    record_id: str,
    error: str,
    fallback_reason: str = "invalid_judge_response",
) -> JudgeRewardRecord:
    """Return a judge sidecar row using the verifier reward as fallback."""
    fallback_task_outcome = candidate.raw_reward
    if candidate.expected_reward_range is not None:
        low, high = candidate.expected_reward_range
        if high != low:
            fallback_task_outcome = (candidate.raw_reward - low) / (high - low)
    fallback_task_outcome = min(1.0, max(0.0, fallback_task_outcome))

    return JudgeRewardRecord(
        record_id=record_id,
        task_id=candidate.task_id,
        iteration=candidate.iteration,
        run_id=candidate.run_id,
        raw_reward=candidate.raw_reward,
        judge_reward=candidate.raw_reward,
        base_reward=candidate.raw_reward,
        applied_cap=None,
        axis_scores=JudgeAxisScores(
            task_outcome=fallback_task_outcome,
            evidence_quality=0.0,
            skill_update_usefulness=0.0,
            token_efficiency=0.0,
            reflection_depth=0.0,
        ),
        flags=JudgeCapFlags(),
        confidence=0.0,
        rationale=(
            "Judge reward was unavailable because the judge response could not be "
            "parsed or validated; using verifier reward as fallback."
        ),
        flag_evidence={},
        judge_model=judge_model,
        rubric_version=rubric_version,
        trace_path=_relative_path(candidate.trace_path, data_dir),
        metrics_path=_relative_path(candidate.metrics_path, data_dir),
        verifier_status=candidate.verifier_status,
        trace_status=candidate.trace_status,
        total_tokens=candidate.total_tokens,
        task_category=candidate.task_category,
        task_difficulty=candidate.task_difficulty,
        expected_reward_range=candidate.expected_reward_range,
        verifier_type=candidate.verifier_type,
        metadata={
            "judge_reward_fallback": True,
            "fallback_source": "verifier_reward",
            "fallback_reason": fallback_reason,
            "fallback_error": error,
        },
    )


async def _judge_candidate(
    *,
    candidate: _JudgeCandidate,
    config: Config,
    llm_client: Any,
) -> JudgeLLMResponse:
    prompt = _candidate_prompt(candidate)
    try:
        result = await llm_client.complete(
            [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_tokens=config.budgets.judge_completion_tokens,
            temperature=0.0,
            budget_label="judge.reward_annotation",
            prompt_budget=config.budgets.judge_prompt_tokens,
            condition_name=config.experiment.condition_name,
        )
        payload = parse_json_object(result["content"])
        return JudgeLLMResponse.model_validate(payload)
    except Exception as exc:
        raise JudgeRewardAnnotationError(
            f"invalid judge response for {candidate.task_id} "
            f"iteration {candidate.iteration}: {exc}"
        ) from exc


def _candidate_prompt(candidate: _JudgeCandidate) -> str:
    payload: dict[str, Any] = {
        "task_id": candidate.task_id,
        "iteration": candidate.iteration,
        "run_id": candidate.run_id,
        "raw_verifier_reward": candidate.raw_reward,
        "verifier_status": candidate.verifier_status,
        "trace_status": candidate.trace_status,
        "total_tokens": candidate.total_tokens,
        "task_category": candidate.task_category,
        "task_difficulty": candidate.task_difficulty,
        "expected_reward_range": candidate.expected_reward_range,
        "verifier_type": candidate.verifier_type,
        "metrics_row": candidate.metrics_row,
    }
    if candidate.trace is not None:
        payload["trace_header"] = trace_header_summary(candidate.trace)
        payload["stdout_excerpt"] = head_tail_text(candidate.trace.stdout, 2000)
        payload["stderr_excerpt"] = head_tail_text(candidate.trace.stderr, 2000)
        payload["test_results"] = candidate.trace.test_results
        payload["error_kind"] = candidate.trace.error_kind
        payload["error_detail"] = candidate.trace.error_detail
        payload["harbor_metadata"] = candidate.trace.harbor_metadata
    return json.dumps(payload, indent=2, sort_keys=True, default=str)


def _load_candidates(data_dir: Path) -> list[_JudgeCandidate]:
    metrics_path = data_dir / "metrics.jsonl"
    if metrics_path.exists():
        return _candidates_from_metrics(data_dir=data_dir, metrics_path=metrics_path)

    traces_path = data_dir / "traces.jsonl"
    if traces_path.exists():
        return _candidates_from_traces(traces_path=traces_path)

    return []


def _candidates_from_metrics(
    *,
    data_dir: Path,
    metrics_path: Path,
) -> list[_JudgeCandidate]:
    candidates = []
    for row in _read_jsonl(metrics_path):
        task_id = str(row.get("task_id") or "")
        if not task_id or task_id == COEVOLUTION_TASK_ID:
            continue
        raw_reward = as_optional_float(row.get("reward"))
        if raw_reward is None:
            continue
        trace_path = _resolve_trace_path(data_dir, row)
        trace = _load_trace(trace_path) if trace_path is not None else None
        verifier_status = _optional_str(row.get("verifier_status"))
        if trace is not None and trace.status != "ok":
            continue
        if trace is None and verifier_status in {"env_failure", "parse_error", "harbor_failed"}:
            continue

        candidates.append(
            _JudgeCandidate(
                task_id=task_id,
                iteration=int(row.get("iteration", 0)),
                raw_reward=raw_reward,
                run_id=_optional_str(row.get("run_id")),
                verifier_status=verifier_status,
                trace_status=trace.status if trace is not None else None,
                total_tokens=int(as_optional_float(row.get("total_tokens")) or 0),
                task_category=_optional_str(row.get("task_category")),
                task_difficulty=_optional_str(row.get("task_difficulty")),
                expected_reward_range=_reward_range(row.get("expected_reward_range")),
                verifier_type=_optional_str(row.get("verifier_type")),
                trace=trace,
                trace_path=trace_path,
                metrics_path=metrics_path,
                metrics_row=row,
            )
        )
    return candidates


def _candidates_from_traces(*, traces_path: Path) -> list[_JudgeCandidate]:
    candidates = []
    for row in _read_jsonl(traces_path):
        trace = ExecutionTrace.model_validate(row)
        if trace.status != "ok" or trace.reward is None:
            continue
        candidates.append(
            _JudgeCandidate(
                task_id=trace.task_id,
                iteration=trace.iteration,
                raw_reward=trace.reward,
                run_id=trace.run_id,
                verifier_status=trace.status,
                trace_status=trace.status,
                total_tokens=(
                    trace.token_usage.input_tokens + trace.token_usage.output_tokens
                ),
                task_category=None,
                task_difficulty=None,
                expected_reward_range=(0.0, 1.0),
                verifier_type="swebench",
                trace=trace,
                trace_path=traces_path,
                metrics_path=None,
                metrics_row={},
            )
        )
    return candidates


def _load_existing_records(path: Path) -> list[JudgeRewardRecord]:
    if not path.exists():
        return []
    return [
        JudgeRewardRecord.model_validate(row)
        for row in _read_jsonl(path)
    ]


def _append_records(path: Path, records: list[JudgeRewardRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for record in records:
            f.write(record.model_dump_json())
            f.write("\n")


def _write_judge_summary(
    *,
    data_dir: Path,
    judge_records: list[JudgeRewardRecord],
) -> None:
    summary_path = data_dir / "summary.json"
    if not summary_path.exists():
        return
    summary = ExperimentScoreSummary.model_validate_json(summary_path.read_text())
    summary.judge_reward_summary = build_score_summary(
        _records_from_judge_rewards(judge_records)
    )
    write_score_summary(summary, summary_path)


def _records_from_judge_rewards(
    judge_records: list[JudgeRewardRecord],
) -> list[IterationRecord]:
    return [
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


def _annotate_history_entries(
    *,
    history_store: HistoryStore,
    judge_records: list[JudgeRewardRecord],
) -> None:
    for record in judge_records:
        metadata = judge_reward_metadata(record)
        history_store.annotate_judge_reward(
            task_id=record.task_id,
            iteration=record.iteration,
            judge_reward=record.judge_reward,
            judge_reward_record_id=record.record_id,
            rubric_version=record.rubric_version,
            confidence=record.confidence,
            applied_cap=record.applied_cap,
            reward_source=metadata["reward_source"],
        )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        loaded = json.loads(line)
        if isinstance(loaded, dict):
            rows.append(loaded)
    return rows


def _resolve_trace_path(data_dir: Path, row: dict[str, Any]) -> Path | None:
    raw_path = row.get("trace_artifact_path")
    if isinstance(raw_path, str) and raw_path.strip():
        path = Path(raw_path)
        if not path.is_absolute():
            path = data_dir / path
        return path

    task_id = row.get("task_id")
    iteration = row.get("iteration")
    if isinstance(task_id, str) and isinstance(iteration, int):
        return data_dir / "artifacts" / "traces" / f"{task_id}_iter{iteration:04d}.json"
    return None


def _load_trace(path: Path | None) -> ExecutionTrace | None:
    if path is None or not path.exists():
        return None
    return ExecutionTrace.model_validate_json(path.read_text())


def _relative_path(path: Path | None, base_dir: Path | None) -> str | None:
    if path is None:
        return None
    if base_dir is None:
        return str(path)
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _reward_range(value: object) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low = as_optional_float(value[0])
    high = as_optional_float(value[1])
    if low is None or high is None:
        return None
    return (low, high)


def _optional_str(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None
