from __future__ import annotations

from pathlib import Path

import pytest

from mediated_coevo.core.config import Config, ModelsConfig
from mediated_coevo.diffusion import DiffusionArtifactType, emit_diffusion_artifacts
from mediated_coevo.experiment.orchestrator import Orchestrator
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.history_signals import MediatorSignal
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from tests.config_helpers import diffusion_config, experiment_config


class _LLM:
    model = "test-model"

    def drain_token_events(self):
        return []


class _Planner:
    def __init__(self) -> None:
        self.llm_client = _LLM()

    def set_skill_context(self, executor_skills: str, skill_refiner: str | None = None):
        return None


class _Executor:
    pass


class _Mediator:
    def __init__(self) -> None:
        self.llm_client = _LLM()

    async def compact_feedback(self, report: MediatorReport) -> MediatorSignal:
        return MediatorSignal(headline=report.content, evidence=report.content)


class _TaskRepo:
    pass


class _SkillStore:
    def read_skill(self, skill_name: str) -> str | None:
        if skill_name == "executor":
            return "# Executor\n"
        return None

    def skill_hashes(self) -> dict[str, str]:
        return {}


class _Advisor:
    llm_client = _LLM()


def _config(*, diffusion_enabled: bool) -> Config:
    config = Config(
        models=ModelsConfig(
            planner="test-planner",
            executor="test-executor",
            mediator="test-mediator",
            judge="test-judge",
        ),
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    config.diffusion.enabled = diffusion_enabled
    config.experiment.coevo_interval = 99
    config.experiment.advisor_buffer_max = 99
    return config


def _record(
    *,
    task_id: str = "task-A",
    iteration: int = 1,
    reward: float | None = 0.4,
    delta_reward: float | None = None,
    success: bool | None = True,
    verifier_status: str | None = "ok",
) -> IterationRecord:
    return IterationRecord(
        iteration=iteration,
        task_id=task_id,
        reward=reward,
        delta_reward=delta_reward,
        success=success,
        verifier_status=verifier_status,
    )


@pytest.mark.asyncio
async def test_emit_diffusion_artifacts_uses_compactor_for_report_summary(monkeypatch):
    compact_calls: list[tuple[str, dict]] = []

    async def _fake_compact_text_for_context(text, **kwargs):
        compact_calls.append((text, kwargs))
        return "COMPACTED REPORT SUMMARY"

    monkeypatch.setattr(
        "mediated_coevo.evolution.compactor.compact_text_for_context",
        _fake_compact_text_for_context,
    )

    trace = ExecutionTrace(task_id="task-A", iteration=2, reward=0.4, status="ok")
    report = MediatorReport(
        task_id="task-A",
        iteration=2,
        content="Investigate the failing auth refresh path. " * 80,
    )

    artifacts = await emit_diffusion_artifacts(
        trace=trace,
        report=report,
        record=_record(iteration=2),
        model="test-model",
        task_metadata={"task_category": "web", "verifier_type": "pytest"},
        condition_name="learned_mediator",
    )

    summary = next(
        artifact
        for artifact in artifacts
        if artifact.artifact_type == DiffusionArtifactType.MEDIATOR_REPORT_SUMMARY
    )
    assert compact_calls
    assert compact_calls[0][0].startswith("Investigate the failing auth refresh path.")
    assert "mediator report for task-A iter 2" == compact_calls[0][1]["label"]
    assert summary.content == "COMPACTED REPORT SUMMARY"
    assert summary.evidence_report_ids == [report.report_id]


@pytest.mark.asyncio
async def test_emit_diffusion_artifacts_skips_infra_failures():
    trace = ExecutionTrace(
        task_id="task-A",
        iteration=3,
        reward=None,
        status="env_failure",
        stderr="harbor missing",
    )

    artifacts = await emit_diffusion_artifacts(
        trace=trace,
        report=None,
        record=_record(iteration=3, reward=None, success=False, verifier_status="env_failure"),
        model="test-model",
    )

    assert artifacts == []


@pytest.mark.asyncio
async def test_emit_diffusion_artifacts_emits_failure_and_regression_warnings():
    trace = ExecutionTrace(
        task_id="task-A",
        iteration=4,
        reward=0.0,
        status="ok",
        stderr="AssertionError: expected 200 got 500",
    )

    artifacts = await emit_diffusion_artifacts(
        trace=trace,
        report=None,
        record=_record(
            iteration=4,
            reward=0.0,
            delta_reward=-0.4,
            success=False,
            verifier_status="task_failed",
        ),
        model="test-model",
        task_metadata={"task_category": "api", "verifier_type": "pytest"},
    )

    artifact_types = {artifact.artifact_type for artifact in artifacts}
    assert DiffusionArtifactType.FAILURE_SIGNATURE in artifact_types
    assert DiffusionArtifactType.REGRESSION_WARNING in artifact_types
    regression = next(
        artifact
        for artifact in artifacts
        if artifact.artifact_type == DiffusionArtifactType.REGRESSION_WARNING
    )
    assert "regressed from 0.40 to 0.00" in regression.content
    assert regression.metadata["trace_ref"] == "task-A:iter0004"


@pytest.mark.asyncio
async def test_orchestrator_emits_diffusion_artifacts_only_when_enabled(tmp_path: Path):
    orchestrator = Orchestrator(
        planner=_Planner(),  # type: ignore[arg-type]
        executor=_Executor(),  # type: ignore[arg-type]
        mediator=_Mediator(),  # type: ignore[arg-type]
        skill_store=_SkillStore(),  # type: ignore[arg-type]
        artifact_store=ArtifactStore(base_dir=tmp_path / "artifacts"),
        history_store=HistoryStore(history_dir=tmp_path / "history"),
        benchmark_repo=_TaskRepo(),  # type: ignore[arg-type]
        config=_config(diffusion_enabled=True),
        experiment_dir=tmp_path,
        skill_advisor=_Advisor(),  # type: ignore[arg-type]
    )
    trace = ExecutionTrace(task_id="task-A", iteration=1, reward=0.6, status="ok")
    report = MediatorReport(task_id="task-A", iteration=1, content="Use the parser guard.")

    await orchestrator._emit_diffusion_artifacts(
        trace=trace,
        report=report,
        record=_record(iteration=1, reward=0.6),
        task_metadata={
            "task_category": "cli",
            "task_difficulty": "medium",
            "expected_reward_range": (0.0, 1.0),
            "verifier_type": "pytest",
        },
        judge_reward=0.7,
    )

    artifacts = orchestrator._diffusion_store.query_artifacts(recent=10)
    assert {artifact.artifact_type for artifact in artifacts} == {
        DiffusionArtifactType.MEDIATOR_REPORT_SUMMARY,
        DiffusionArtifactType.DEBUG_HINT,
    }
    assert all(artifact.source_iteration == 1 for artifact in artifacts)
    assert all(artifact.judge_reward == pytest.approx(0.7) for artifact in artifacts)

    disabled = Orchestrator(
        planner=_Planner(),  # type: ignore[arg-type]
        executor=_Executor(),  # type: ignore[arg-type]
        mediator=_Mediator(),  # type: ignore[arg-type]
        skill_store=_SkillStore(),  # type: ignore[arg-type]
        artifact_store=ArtifactStore(base_dir=tmp_path / "artifacts-disabled"),
        history_store=HistoryStore(history_dir=tmp_path / "history-disabled"),
        benchmark_repo=_TaskRepo(),  # type: ignore[arg-type]
        config=_config(diffusion_enabled=False),
        experiment_dir=tmp_path / "disabled-run",
        skill_advisor=_Advisor(),  # type: ignore[arg-type]
    )

    await disabled._emit_diffusion_artifacts(
        trace=trace,
        report=report,
        record=_record(iteration=1, reward=0.6),
        task_metadata={
            "task_category": "cli",
            "task_difficulty": "medium",
            "expected_reward_range": (0.0, 1.0),
            "verifier_type": "pytest",
        },
        judge_reward=0.7,
    )

    assert disabled._diffusion_store.query_artifacts(recent=10) == []
