"""Regression tests for Orchestrator._tag_previous_iteration.

P0 #4 wants iteration-N's reward to land on iteration-N's planner/mediator
HistoryEntry — not on a stale entry that's been carried over an env_failure.

Sequence under test: [ok, env_failure, ok] for one task.

Without the always-pop semantics, iter 0's entry would still be in the
carry-forward dict at iter 2 and get tagged with iter 2's reward — exactly
the cross-attribution P0 #4 was meant to prevent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mediated_coevo.models.history_signals import MediatorSignal, PlannerSignal
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.skill import SkillProposal
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.config import Config
from mediated_coevo.orchestrator import Orchestrator
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryEntry, HistoryStore


def _bare_orchestrator(tmp_path: Path) -> Orchestrator:
    """Build an Orchestrator skeleton with only the fields the tagging
    helper touches. We bypass __init__ to avoid wiring agents/llm clients
    that aren't relevant to the tagging contract."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch._prev_mediator_entry_id_by_task = {}
    orch._prev_planner_entry_id_by_task = {}
    orch._proposal_buffer = []
    return orch


def _ok_trace(task_id: str, iteration: int, reward: float) -> ExecutionTrace:
    return ExecutionTrace(
        task_id=task_id, iteration=iteration, reward=reward, status="ok",
    )


def _env_failure_trace(task_id: str, iteration: int) -> ExecutionTrace:
    return ExecutionTrace(
        task_id=task_id,
        iteration=iteration,
        status="env_failure",
        error_kind="harbor_not_found",
    )


def test_env_failure_drops_stale_entry_id_so_later_iter_does_not_mistag(tmp_path):
    orch = _bare_orchestrator(tmp_path)
    task = "task-A"

    # iter 0: ok — orchestrator would have written a HistoryEntry and stashed
    # its id for iter 1 to tag.
    e0 = orch.history_store.add(HistoryEntry(
        iteration=0, agent_role="mediator",
        payload=MediatorSignal(headline="iter0"),
        metadata={"task_id": task},
    ))
    orch._prev_mediator_entry_id_by_task[task] = e0

    # iter 1: env_failure. The carry-forward must be dropped, NOT preserved.
    orch._tag_previous_iteration(1, task, _env_failure_trace(task, 1))

    assert task not in orch._prev_mediator_entry_id_by_task, (
        "stale entry id leaked across env_failure"
    )
    # iter 0's entry was never tagged — that is correct: we never observed
    # a clean reward for the iter immediately after it.
    e0_loaded = next(e for e in orch.history_store._entries if e.entry_id == e0)
    assert e0_loaded.reward is None

    # iter 2: ok — writes its own entry, then the next iter would tag it.
    # Crucially, iter 2's reward must NOT travel back to e0.
    e2 = orch.history_store.add(HistoryEntry(
        iteration=2, agent_role="mediator",
        payload=MediatorSignal(headline="iter2"),
        metadata={"task_id": task},
    ))
    orch._prev_mediator_entry_id_by_task[task] = e2
    orch._tag_previous_iteration(3, task, _ok_trace(task, 3, reward=0.9))

    e0_after = next(e for e in orch.history_store._entries if e.entry_id == e0)
    e2_after = next(e for e in orch.history_store._entries if e.entry_id == e2)
    assert e0_after.reward is None, "iter 0 entry must remain untagged"
    assert e2_after.reward == pytest.approx(0.9), (
        "iter 3 reward should land on iter 2 entry only"
    )


def test_ok_trace_after_ok_tags_correctly(tmp_path):
    orch = _bare_orchestrator(tmp_path)
    task = "task-A"

    e0 = orch.history_store.add(HistoryEntry(
        iteration=0, agent_role="planner",
        payload=PlannerSignal(reasoning="first"),
        metadata={"task_id": task},
    ))
    orch._prev_planner_entry_id_by_task[task] = e0

    orch._tag_previous_iteration(1, task, _ok_trace(task, 1, reward=0.4))

    e0_after = next(e for e in orch.history_store._entries if e.entry_id == e0)
    assert e0_after.reward == pytest.approx(0.4)
    assert task not in orch._prev_planner_entry_id_by_task


def test_two_tasks_same_iteration_tagged_independently(tmp_path):
    """P0-2 acceptance: in a multi-task iteration, each task's reward tags
    only its own carry-forward entry — no cross-attribution between tasks."""
    orch = _bare_orchestrator(tmp_path)

    # Iter 0: both tasks add a mediator + planner entry and stash their IDs.
    a_mid = orch.history_store.add(HistoryEntry(
        iteration=0, agent_role="mediator",
        payload=MediatorSignal(headline="A iter0"),
        metadata={"task_id": "task-A"},
    ))
    a_pid = orch.history_store.add(HistoryEntry(
        iteration=0, agent_role="planner",
        payload=PlannerSignal(reasoning="A iter0"),
        metadata={"task_id": "task-A"},
    ))
    b_mid = orch.history_store.add(HistoryEntry(
        iteration=0, agent_role="mediator",
        payload=MediatorSignal(headline="B iter0"),
        metadata={"task_id": "task-B"},
    ))
    b_pid = orch.history_store.add(HistoryEntry(
        iteration=0, agent_role="planner",
        payload=PlannerSignal(reasoning="B iter0"),
        metadata={"task_id": "task-B"},
    ))
    orch._prev_mediator_entry_id_by_task["task-A"] = a_mid
    orch._prev_planner_entry_id_by_task["task-A"] = a_pid
    orch._prev_mediator_entry_id_by_task["task-B"] = b_mid
    orch._prev_planner_entry_id_by_task["task-B"] = b_pid

    # Iter 1: both tasks run with distinct rewards.
    orch._tag_previous_iteration(1, "task-A", _ok_trace("task-A", 1, reward=0.2))
    orch._tag_previous_iteration(1, "task-B", _ok_trace("task-B", 1, reward=0.9))

    by_id = {e.entry_id: e for e in orch.history_store._entries}
    assert by_id[a_mid].reward == pytest.approx(0.2)
    assert by_id[a_pid].reward == pytest.approx(0.2)
    assert by_id[b_mid].reward == pytest.approx(0.9)
    assert by_id[b_pid].reward == pytest.approx(0.9)

    # Carry-forward dicts cleared for both tasks.
    assert "task-A" not in orch._prev_mediator_entry_id_by_task
    assert "task-A" not in orch._prev_planner_entry_id_by_task
    assert "task-B" not in orch._prev_mediator_entry_id_by_task
    assert "task-B" not in orch._prev_planner_entry_id_by_task


def test_proposal_buffer_backfill_only_on_ok_trace(tmp_path):
    orch = _bare_orchestrator(tmp_path)
    task = "task-A"
    orch._proposal_buffer = [
        SkillProposal(iteration=0, task_id=task, old_content="", new_content="x"),
    ]

    # env_failure at iter 1 must NOT backfill iter 0's proposal reward.
    orch._tag_previous_iteration(1, task, _env_failure_trace(task, 1))
    assert orch._proposal_buffer[0].reward is None

    # Subsequent ok at iter 2 also must NOT retroactively reach iter 0
    # (the iteration==iteration-1 guard prevents stale backfill).
    orch._tag_previous_iteration(2, task, _ok_trace(task, 2, reward=0.7))
    assert orch._proposal_buffer[0].reward is None


def test_iter_zero_is_a_noop(tmp_path):
    orch = _bare_orchestrator(tmp_path)
    orch._prev_mediator_entry_id_by_task["task-A"] = "should-not-touch"

    orch._tag_previous_iteration(0, "task-A", _ok_trace("task-A", 0, reward=1.0))

    # Pre-iteration-1 should never pop or tag.
    assert orch._prev_mediator_entry_id_by_task["task-A"] == "should-not-touch"


class _NoCallPlanner:
    def __getattr__(self, name):
        raise AssertionError(f"planner should not be called: {name}")


class _NoCallMediator:
    def __getattr__(self, name):
        raise AssertionError(f"mediator should not be called: {name}")


class _NoCallExecutor:
    def __getattr__(self, name):
        raise AssertionError(f"executor should not be called: {name}")


class _EmptySkillStore:
    def read_skill(self, skill_name: str) -> str | None:
        return None


class _MissingTaskRepo:
    def resolve(self, task_id: str):
        raise FileNotFoundError(f"missing task: {task_id}")


@pytest.mark.asyncio
async def test_missing_task_is_recorded_as_env_failure_without_agent_calls(tmp_path):
    orch = Orchestrator.__new__(Orchestrator)
    orch.planner = _NoCallPlanner()
    orch.executor = _NoCallExecutor()
    orch.mediator = _NoCallMediator()
    orch.skill_store = _EmptySkillStore()
    orch.artifact_store = ArtifactStore(base_dir=tmp_path / "artifacts")
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch.benchmark_repo = _MissingTaskRepo()
    orch.config = Config()
    orch.experiment_dir = tmp_path
    orch.skill_advisor = None
    orch._proposal_buffer = []
    orch._previous_report_by_task = {}
    orch._prev_mediator_entry_id_by_task = {}
    orch._prev_planner_entry_id_by_task = {}

    record = await orch._run_iteration("missing-task", 1)

    assert record.task_spec is None
    assert record.reward is None
    assert record.execution_trace is not None
    assert record.execution_trace.status == "env_failure"
    assert record.execution_trace.error_kind == "task_not_found"
    stored = orch.artifact_store.load_trace("missing-task", 1)
    assert stored is not None
    assert stored.error_kind == "task_not_found"


class _ResolvedTask:
    instruction = "base instruction"


class _AnyTaskRepo:
    def resolve(self, task_id: str):
        return _ResolvedTask()


class _RecordingPlanner:
    def __init__(self) -> None:
        self.prior_contexts: dict[str, str | None] = {}

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
    ) -> TaskSpec:
        self.prior_contexts[task_id] = prior_context
        return TaskSpec(task_id=task_id, instruction=base_instruction)


class _EnvFailureExecutor:
    async def execute_task(
        self,
        task_spec: TaskSpec,
        skill_texts: list[str],
    ) -> ExecutionTrace:
        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            status="env_failure",
            error_kind="test_env_failure",
        )


@pytest.mark.asyncio
async def test_previous_report_prior_context_is_keyed_by_task(tmp_path):
    planner = _RecordingPlanner()
    orch = Orchestrator.__new__(Orchestrator)
    orch.planner = planner
    orch.executor = _EnvFailureExecutor()
    orch.mediator = _NoCallMediator()
    orch.skill_store = _EmptySkillStore()
    orch.artifact_store = ArtifactStore(base_dir=tmp_path / "artifacts")
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch.benchmark_repo = _AnyTaskRepo()
    orch.config = Config()
    orch.experiment_dir = tmp_path
    orch.skill_advisor = None
    orch._proposal_buffer = []
    orch._previous_report_by_task = {
        "task-A": MediatorReport(task_id="task-A", iteration=0, content="task-A report"),
    }
    orch._prev_mediator_entry_id_by_task = {}
    orch._prev_planner_entry_id_by_task = {}

    await orch._run_iteration("task-B", 1)
    await orch._run_iteration("task-A", 1)

    assert planner.prior_contexts["task-B"] is None
    assert planner.prior_contexts["task-A"] == "task-A report"


class _MemorySkillStore:
    def __init__(self) -> None:
        self.content = "old"
        self.writes: list[tuple[str, str]] = []

    def read_skill(self, skill_name: str) -> str | None:
        assert skill_name == "executor"
        return self.content

    def write_skill(self, skill_name: str, content: str):
        self.writes.append((skill_name, content))
        self.content = content


class _ApprovingAdvisor:
    def __init__(self) -> None:
        self.seen: list[SkillProposal] = []

    async def review(self, current_skill: str, proposals: list[SkillProposal]) -> str:
        self.seen = list(proposals)
        return "approved"


class _PatchPlanner:
    step = 7

    async def propose_skill_update(
        self,
        current_skill_content: str,
        feedback: str | None,
        edit_history: list,
        task_id: str = "",
        iteration: int = 0,
    ) -> SkillProposal:
        return SkillProposal(
            iteration=iteration,
            task_id=task_id,
            old_content=current_skill_content,
            new_content="new",
            reasoning="patched",
        )


@pytest.mark.asyncio
async def test_advisor_patch_preserves_buffered_task_provenance(tmp_path):
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = Config()
    orch.config.experiment.advisor_buffer_max = 2
    orch.skill_store = _MemorySkillStore()
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch.skill_advisor = _ApprovingAdvisor()
    orch.planner = _PatchPlanner()
    orch._proposal_buffer = [
        SkillProposal(iteration=0, task_id="task-B", old_content="", new_content="b"),
        SkillProposal(iteration=0, task_id="task-A", old_content="", new_content="a"),
    ]

    update = await orch._advise_and_patch()

    assert update is not None
    assert update.task_id == "task-A,task-B"
    assert len(orch.skill_advisor.seen) == 2
    assert orch._proposal_buffer == []
    assert orch.skill_store.writes == [("executor", "new")]
