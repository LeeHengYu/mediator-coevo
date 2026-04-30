from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import typer
from pydantic import ValidationError

from mediated_coevo.config import Config
from mediated_coevo.main import _validate_condition_name
from mediated_coevo.models.history_signals import MediatorSignal
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.orchestrator import Orchestrator
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore


class _Task:
    instruction = "base instruction"


class _TaskRepo:
    def resolve(self, task_id: str):
        return _Task()


class _SkillStore:
    def read_skill(self, skill_name: str) -> str | None:
        return None


class _PlannerLLM:
    model = "test-model"


class _Planner:
    def __init__(self) -> None:
        self.prior_contexts: dict[str, str | None] = {}
        self.llm_client = _PlannerLLM()

    def set_skill_context(
        self,
        executor_skills: str,
        skill_refiner: str | None = None,
    ) -> None:
        pass

    async def plan_task(
        self,
        task_id: str,
        base_instruction: str,
        prior_context: str | None = None,
        current_skills: list[str] | None = None,
        iteration: int = 0,
    ) -> TaskSpec:
        self.prior_contexts[task_id] = prior_context
        return TaskSpec(task_id=task_id, instruction=base_instruction, iteration=iteration)


class _Executor:
    async def execute_task(
        self,
        task_spec: TaskSpec,
        skill_texts: list[str],
    ) -> ExecutionTrace:
        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            reward=0.5,
            status="ok",
        )


class _Mediator:
    def __init__(self, llm_client=None) -> None:
        self.process_calls = 0
        self.compact_calls = 0
        self.llm_client = llm_client

    async def process_trace(
        self,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport:
        self.process_calls += 1
        return MediatorReport(
            task_id=trace.task_id,
            iteration=trace.iteration,
            content=f"fresh report for {trace.task_id}",
        )

    async def mediate_trace(
        self,
        condition: str,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport | None:
        if condition not in {"static_mediator", "learned_mediator"}:
            return None
        if not trace.is_usable_feedback_signal:
            return None
        return await self.process_trace(trace, task_context)

    async def compact_feedback(
        self,
        report: MediatorReport,
    ) -> MediatorSignal:
        self.compact_calls += 1
        return MediatorSignal(headline=report.exposed_content or "")


class _TraceHistoryInspectingMediator:
    def __init__(self, artifact_store: ArtifactStore) -> None:
        self.artifact_store = artifact_store
        self.trace_iterations_seen: list[int] = []

    async def mediate_trace(
        self,
        condition: str,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport | None:
        self.trace_iterations_seen = [
            item.iteration
            for item in self.artifact_store.query_traces(task_id=trace.task_id)
        ]
        return None


class _WithholdingMediator:
    async def mediate_trace(
        self,
        condition: str,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport:
        return MediatorReport(
            task_id=trace.task_id,
            iteration=trace.iteration,
            content="withheld content",
            withheld=True,
            reasoning="not useful for planner",
        )

    async def compact_feedback(
        self,
        report: MediatorReport,
    ) -> MediatorSignal:
        return MediatorSignal(
            withheld=report.withheld,
            mediator_reasoning=report.reasoning,
        )


class _FailingMediator:
    async def mediate_trace(
        self,
        condition: str,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport | None:
        raise RuntimeError("mediator failed")


class _LLMCompactor:
    model = "test-model"

    def __init__(self, *, content: str, raise_exc: Exception | None = None) -> None:
        self.content = content
        self.raise_exc = raise_exc
        self.calls: list[dict] = []

    async def complete(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_exc:
            raise self.raise_exc
        return {
            "content": self.content,
            "input_tokens": 1,
            "output_tokens": 1,
            "model": "test",
            "raw": {},
        }


def _orchestrator(
    tmp_path: Path,
    condition: str,
    *,
    llm_client=None,
) -> tuple[Orchestrator, _Planner, _Mediator]:
    planner = _Planner()
    mediator = _Mediator(llm_client=llm_client)
    orch: Any = Orchestrator.__new__(Orchestrator)
    orch.planner = planner
    orch.executor = _Executor()
    orch.mediator = mediator
    orch.skill_store = _SkillStore()
    orch.artifact_store = ArtifactStore(base_dir=tmp_path / "artifacts")
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch.benchmark_repo = _TaskRepo()
    orch.config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
        }
    )
    orch.config.experiment.condition_name = condition
    orch.config.experiment.shared_notes = "shared note"
    orch.experiment_dir = tmp_path
    orch.skill_advisor = None
    orch._proposal_buffer = []
    orch._previous_report_by_task = {}
    return orch, planner, mediator


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("condition", "expected_context", "expected_mediator_calls"),
    [
        ("no_feedback", None, 0),
        ("full_traces", "iter=0 reward=0.25 OK", 0),
        ("shared_notes", "shared note", 0),
        ("static_mediator", "prior same-task report", 1),
        ("learned_mediator", "prior same-task report", 1),
    ],
)
async def test_feedback_conditions_control_planner_context_and_mediator_calls(
    tmp_path,
    condition,
    expected_context,
    expected_mediator_calls,
):
    orch, planner, mediator = _orchestrator(tmp_path, condition)
    if condition == "full_traces":
        orch.artifact_store.store_trace(
            ExecutionTrace(
                task_id="task-A",
                iteration=0,
                reward=0.25,
                status="ok",
            )
        )
    if condition in {"static_mediator", "learned_mediator"}:
        orch._previous_report_by_task["task-A"] = MediatorReport(
            task_id="task-A",
            iteration=0,
            content="prior same-task report",
        )

    record = await orch._run_iteration("task-A", 1)

    assert planner.prior_contexts["task-A"] == expected_context
    assert mediator.process_calls == expected_mediator_calls
    assert record.condition_name == condition
    assert record.cross_task_feedback_enabled is False
    assert record.execution_trace is not None
    assert record.execution_trace.iteration == 1


@pytest.mark.asyncio
async def test_mediator_history_excludes_current_trace(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.artifact_store.store_trace(
        ExecutionTrace(task_id="task-A", iteration=0, reward=0.25, status="ok")
    )
    mediator = _TraceHistoryInspectingMediator(orch.artifact_store)
    orch.mediator = mediator

    await orch._run_iteration("task-A", 1)

    assert mediator.trace_iterations_seen == [0]
    assert orch.artifact_store.load_trace("task-A", 1) is not None


@pytest.mark.asyncio
async def test_trace_is_stored_when_mediator_fails(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.mediator = _FailingMediator()

    with pytest.raises(RuntimeError, match="mediator failed"):
        await orch._run_iteration("task-A", 1)

    assert orch.artifact_store.load_trace("task-A", 1) is not None


@pytest.mark.asyncio
async def test_withheld_mediator_report_is_recorded_for_reflection(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch.mediator = _WithholdingMediator()

    record = await orch._run_iteration("task-A", 1)

    assert record.mediator_report is None
    assert record.mediator_history_entry_id is not None
    assert "task-A" not in orch._previous_report_by_task
    entry = next(
        item
        for item in orch.history_store._entries
        if item.entry_id == record.mediator_history_entry_id
    )
    assert isinstance(entry.payload, MediatorSignal)
    assert entry.payload.withheld is True


@pytest.mark.asyncio
async def test_cross_task_feedback_is_opt_in_and_labeled(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "learned_mediator")
    orch._previous_report_by_task["task-B"] = MediatorReport(
        task_id="task-B",
        iteration=2,
        content="cross-task report",
    )

    assert await orch._build_prior_context("learned_mediator", "task-A") is None

    orch.config.experiment.allow_cross_task_feedback = True
    context = await orch._build_prior_context("learned_mediator", "task-A")

    assert context is not None
    assert "Explicit Cross-Task Feedback" in context
    assert "allow_cross_task_feedback=true" in context
    assert "source_task=task-B" in context
    assert "cross-task report" in context


@pytest.mark.asyncio
async def test_cross_task_full_traces_exclude_target_task(tmp_path):
    orch, _, _ = _orchestrator(tmp_path, "full_traces")
    orch.config.experiment.allow_cross_task_feedback = True
    orch.artifact_store.store_trace(
        ExecutionTrace(task_id="task-A", iteration=0, reward=0.1, status="ok")
    )
    orch.artifact_store.store_trace(
        ExecutionTrace(task_id="task-B", iteration=0, reward=0.9, status="ok")
    )

    context = await orch._build_prior_context("full_traces", "task-A")

    assert context is not None
    assert "source_task=task-B" in context
    assert "reward=0.90" in context
    assert "source_task=task-A" not in context


@pytest.mark.asyncio
async def test_long_trace_stderr_uses_llm_compactor(tmp_path):
    llm = _LLMCompactor(
        content='{"headline": "Build failed.", "evidence": "Missing dependency xyz."}'
    )
    orch, _, _ = _orchestrator(tmp_path, "full_traces", llm_client=llm)
    orch.artifact_store.store_trace(
        ExecutionTrace(
            task_id="task-A",
            iteration=0,
            reward=0.0,
            status="ok",
            stderr="dependency xyz missing\n" * 100,
        )
    )

    context = await orch._build_prior_context("full_traces", "task-A")

    assert context is not None
    assert "Build failed." in context
    assert "Missing dependency xyz." in context
    assert len(llm.calls) == 1


@pytest.mark.asyncio
async def test_short_trace_stderr_does_not_call_llm_compactor(tmp_path):
    llm = _LLMCompactor(content='{"headline": "unused", "evidence": "unused"}')
    orch, _, _ = _orchestrator(tmp_path, "full_traces", llm_client=llm)
    orch.artifact_store.store_trace(
        ExecutionTrace(
            task_id="task-A",
            iteration=0,
            reward=0.0,
            status="ok",
            stderr="short stderr",
        )
    )

    context = await orch._build_prior_context("full_traces", "task-A")

    assert context is not None
    assert "short stderr" in context
    assert llm.calls == []


@pytest.mark.asyncio
async def test_long_trace_stderr_falls_back_when_llm_compactor_fails(tmp_path):
    llm = _LLMCompactor(content="", raise_exc=RuntimeError("llm unavailable"))
    orch, _, _ = _orchestrator(tmp_path, "full_traces", llm_client=llm)
    orch.artifact_store.store_trace(
        ExecutionTrace(
            task_id="task-A",
            iteration=0,
            reward=0.0,
            status="ok",
            stderr=("START-" + ("x" * 900) + "-END"),
        )
    )

    context = await orch._build_prior_context("full_traces", "task-A")

    assert context is not None
    assert "START-" in context
    assert "-END" in context
    assert "\n...\n" in context or "\n…\n" in context
    assert len(llm.calls) == 1


def test_condition_assignment_and_cli_validation_reject_unknown_names():
    config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
        }
    )
    with pytest.raises(ValidationError):
        config.experiment.condition_name = "bad-condition"

    with pytest.raises(typer.BadParameter):
        _validate_condition_name("bad-condition")

    assert _validate_condition_name("no_feedback") == "no_feedback"
