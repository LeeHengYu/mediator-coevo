"""Regression tests for pending outcome tagging.

P0 #4 wants iteration-N's reward to land on iteration-N's planner/mediator
HistoryEntry — not on a stale entry that's been carried over an env_failure.

Sequence under test: [ok, env_failure, ok] for one task.

Without the always-pop semantics, iter 0's entry would still be pending
at iter 2 and get tagged with iter 2's reward — exactly the
cross-attribution P0 #4 was meant to prevent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mediated_coevo.analysis.metrics import metric_row
from mediated_coevo.experiment.records import (
    attach_skill_identity,
    build_coevolution_record,
    build_iteration_record,
)
from mediated_coevo.models.history_signals import MediatorSignal, PlannerSignal
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.skill import (
    AdvisorBatchProvenance,
    ProposalRef,
    SkillProposal,
    SkillUpdate,
    SkillUpdateCandidate,
)
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace, TokenUsage
from mediated_coevo.core.config import Config
from mediated_coevo.evolution.executor_skill_gate import ExecutorSkillGate
from mediated_coevo.experiment.orchestrator import Orchestrator
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryEntry, HistoryStore
from mediated_coevo.stores.skill_store import SkillStore
from mediated_coevo.runtime.token_budget import TokenBudgetEvent


def _bare_orchestrator(tmp_path: Path) -> Orchestrator:
    """Build an Orchestrator skeleton with only the fields the tagging
    helper touches. We bypass __init__ to avoid wiring agents/llm clients
    that aren't relevant to the tagging contract."""
    orch = Orchestrator.__new__(Orchestrator)
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch._proposal_buffer = []
    return orch


def _attach_executor_skill_gate(orch: Orchestrator) -> None:
    orch.executor_skill_gate = ExecutorSkillGate(
        config=orch.config,
        skill_store=orch.skill_store,
        history_store=orch.history_store,
        planner=orch.planner,
        skill_advisor=orch.skill_advisor,
        executor=orch.executor,
        benchmark_repo=orch.benchmark_repo,
        artifact_store=orch.artifact_store,
    )


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
    orch.history_store.remember_pending_outcome(task, mediator_entry_id=e0)

    # iter 1: env_failure. The carry-forward must be dropped, NOT preserved.
    orch.history_store.tag_pending_outcome(task, _env_failure_trace(task, 1))

    assert task not in orch.history_store._pending_mediator_entry_id_by_task, (
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
    orch.history_store.remember_pending_outcome(task, mediator_entry_id=e2)
    orch.history_store.tag_pending_outcome(task, _ok_trace(task, 3, reward=0.9))

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
    orch.history_store.remember_pending_outcome(task, planner_entry_id=e0)

    orch.history_store.tag_pending_outcome(task, _ok_trace(task, 1, reward=0.4))

    e0_after = next(e for e in orch.history_store._entries if e.entry_id == e0)
    assert e0_after.reward == pytest.approx(0.4)
    assert task not in orch.history_store._pending_planner_entry_id_by_task


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
    orch.history_store.remember_pending_outcome(
        "task-A",
        mediator_entry_id=a_mid,
        planner_entry_id=a_pid,
    )
    orch.history_store.remember_pending_outcome(
        "task-B",
        mediator_entry_id=b_mid,
        planner_entry_id=b_pid,
    )

    # Iter 1: both tasks run with distinct rewards.
    orch.history_store.tag_pending_outcome(
        "task-A",
        _ok_trace("task-A", 1, reward=0.2),
    )
    orch.history_store.tag_pending_outcome(
        "task-B",
        _ok_trace("task-B", 1, reward=0.9),
    )

    by_id = {e.entry_id: e for e in orch.history_store._entries}
    assert by_id[a_mid].reward == pytest.approx(0.2)
    assert by_id[a_pid].reward == pytest.approx(0.2)
    assert by_id[b_mid].reward == pytest.approx(0.9)
    assert by_id[b_pid].reward == pytest.approx(0.9)

    # Carry-forward dicts cleared for both tasks.
    assert "task-A" not in orch.history_store._pending_mediator_entry_id_by_task
    assert "task-A" not in orch.history_store._pending_planner_entry_id_by_task
    assert "task-B" not in orch.history_store._pending_mediator_entry_id_by_task
    assert "task-B" not in orch.history_store._pending_planner_entry_id_by_task


def test_proposal_buffer_backfill_only_on_ok_trace(tmp_path):
    orch = _bare_orchestrator(tmp_path)
    task = "task-A"
    orch._proposal_buffer = [
        SkillProposal(iteration=0, task_id=task, old_content="", new_content="x"),
    ]

    # env_failure at iter 1 must NOT backfill iter 0's proposal reward.
    orch.history_store.tag_pending_outcome(
        task,
        _env_failure_trace(task, 1),
        proposals=orch._proposal_buffer,
    )
    assert orch._proposal_buffer[0].reward is None

    # Subsequent ok at iter 2 also must NOT retroactively reach iter 0
    # (the iteration==iteration-1 guard prevents stale backfill).
    orch.history_store.tag_pending_outcome(
        task,
        _ok_trace(task, 2, reward=0.7),
        proposals=orch._proposal_buffer,
    )
    assert orch._proposal_buffer[0].reward is None


def test_iter_zero_is_a_noop(tmp_path):
    orch = _bare_orchestrator(tmp_path)
    orch.history_store.remember_pending_outcome(
        "task-A",
        mediator_entry_id="should-not-touch",
    )

    orch.history_store.tag_pending_outcome(
        "task-A",
        _ok_trace("task-A", 0, reward=1.0),
    )

    # Pre-iteration-1 should never pop or tag.
    assert (
        orch.history_store._pending_mediator_entry_id_by_task["task-A"]
        == "should-not-touch"
    )


class _DrainClient:
    model = "test-model"

    def __init__(self, label: str) -> None:
        self.events = [
            TokenBudgetEvent(
                label=label,
                model="test-model",
                prompt_tokens=1,
                completion_tokens=2,
                total_tokens=3,
            )
        ]

    def drain_token_events(self) -> list[TokenBudgetEvent]:
        events = list(self.events)
        self.events.clear()
        return events


class _LLMBackedComponent:
    def __init__(self, client: _DrainClient) -> None:
        self.llm_client = client


def test_drain_llm_token_events_uses_llm_backed_components():
    clients = [
        _DrainClient("planner.plan_task"),
        _DrainClient("mediator.process_trace"),
        _DrainClient("advisor.review"),
    ]
    orch = Orchestrator.__new__(Orchestrator)
    orch.planner = _LLMBackedComponent(clients[0])
    orch.mediator = _LLMBackedComponent(clients[1])
    orch.skill_advisor = _LLMBackedComponent(clients[2])

    events = orch._drain_llm_token_events()

    assert [event.label for event in events] == [
        "planner.plan_task",
        "mediator.process_trace",
        "advisor.review",
    ]
    assert all(client.events == [] for client in clients)


def test_attach_skill_identity_populates_record_and_skill_update():
    update = SkillUpdate(
        skill_id="executor",
        old_content="old",
        new_content="new",
    )
    record = IterationRecord(
        iteration=3,
        task_id="task-A",
        skill_update=update,
    )

    attach_skill_identity(
        record,
        {"executor": "hash-a", "planner": "hash-b"},
        "iter_0003",
    )

    assert record.skill_hashes == {"executor": "hash-a", "planner": "hash-b"}
    assert record.skill_version == "iter_0003"
    assert record.skill_update is not None
    assert record.skill_update.skill_version == "iter_0003"


def test_attach_skill_identity_populates_coevolution_skill_updates():
    mediator_update = SkillUpdate(
        skill_id="mediator",
        old_content="old mediator",
        new_content="new mediator",
    )
    planner_update = SkillUpdate(
        skill_id="planner",
        old_content="old planner",
        new_content="new planner",
    )
    record = IterationRecord(
        iteration=3,
        task_id="__coevolution__",
        skill_updates=[mediator_update, planner_update],
    )

    attach_skill_identity(
        record,
        {"mediator": "hash-a", "planner": "hash-b"},
        "iter_0003",
    )

    assert record.skill_version == "iter_0003"
    assert [update.skill_version for update in record.skill_updates] == [
        "iter_0003",
        "iter_0003",
    ]


def test_attach_skill_identity_preserves_existing_skill_hashes():
    record = IterationRecord(
        iteration=3,
        task_id="task-A",
        skill_hashes={"executor": "start-hash"},
    )

    attach_skill_identity(
        record,
        {"executor": "end-hash", "planner": "planner-hash"},
        "iter_0003",
    )

    assert record.skill_hashes == {"executor": "start-hash"}
    assert record.skill_version == "iter_0003"


def test_build_coevolution_record_captures_reflector_token_events():
    event = TokenBudgetEvent(
        label="reflector.planner",
        model="test-model",
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        }
    )
    orch.skill_store = _EmptySkillStore()

    record = build_coevolution_record(
        iteration=4,
        condition="learned_mediator",
        duration_sec=0.0,
        llm_token_events=[event],
        skill_updates=[
            SkillUpdate(
                skill_id="planner",
                old_content="old",
                new_content="new",
            ),
        ],
        config=orch.config,
        skill_hashes=orch.skill_store.skill_hashes(),
    )

    assert record.task_id == "__coevolution__"
    assert record.iteration == 4
    assert record.total_tokens == 15
    assert record.llm_token_events == [event]
    assert [update.skill_id for update in record.skill_updates] == ["planner"]
    assert record.seed == 42
    assert record.models == {
        "planner": "test-planner",
        "executor": "test-executor",
        "mediator": "test-mediator",
        "judge": "test-judge",
    }
    assert record.executor_agent == "opencode"
    assert record.token_totals_by_agent == {"reflector": 15}


def test_build_iteration_record_adds_compact_metric_fields():
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        }
    )
    orch._previous_reward_by_task = {"task-A": 0.25}
    orch.planner = _LLMBackedComponent(_DrainClient("planner.plan_task"))
    orch.mediator = _LLMBackedComponent(_DrainClient("mediator.process_trace"))
    orch.skill_advisor = _LLMBackedComponent(_DrainClient("skill_advisor.review"))

    report = MediatorReport(task_id="task-A", iteration=2)
    update = SkillUpdate(
        skill_id="executor",
        old_content="old",
        new_content="new",
        provenance=AdvisorBatchProvenance(
            batch_id="batch-1",
            iteration=2,
            skill_id="executor",
            task_ids=["task-A"],
            base_skill_hash="old-hash",
            decision="approved",
            reason="approved feedback",
            proposal_refs=[
                ProposalRef(
                    proposal_id="proposal-1",
                    task_id="task-A",
                    iteration=1,
                    reward=0.75,
                )
            ],
        ),
    )

    record = build_iteration_record(
        task_id="task-A",
        iteration=2,
        condition="learned_mediator",
        duration_sec=0.0,
        task_spec=TaskSpec(task_id="task-A", instruction="do it", iteration=2),
        trace=ExecutionTrace(
            task_id="task-A",
            iteration=2,
            reward=0.75,
            status="ok",
            token_usage=TokenUsage(input_tokens=4, output_tokens=6),
            run_id="job-123",
        ),
        report=report,
        skill_update=update,
        mediator_entry_id="mediator-entry",
        planner_entry_id="planner-entry",
        skill_hashes={"executor": "hash"},
        task_metadata={
            "task_category": "build",
            "task_difficulty": "medium",
            "expected_reward_range": (0.0, 1.0),
            "verifier_type": "pytest",
        },
        llm_token_events=orch._drain_llm_token_events(),
        config=orch.config,
        previous_reward_by_task=orch._previous_reward_by_task,
    )

    assert record.run_id == "job-123"
    assert record.seed == 42
    assert record.models == {
        "planner": "test-planner",
        "executor": "test-executor",
        "mediator": "test-mediator",
        "judge": "test-judge",
    }
    assert record.executor_agent == "opencode"
    assert record.mediator_report_id == report.report_id
    assert record.history_entry_ids == {
        "mediator": "mediator-entry",
        "planner": "planner-entry",
    }
    assert record.proposal_ids == ["proposal-1"]
    assert record.success is True
    assert record.verifier_status == "ok"
    assert record.delta_reward == pytest.approx(0.5)
    assert record.token_totals_by_agent == {
        "executor": 10,
        "planner": 3,
        "mediator": 3,
        "skill_advisor": 3,
    }
    assert record.advisor_decision == "approved"
    assert record.advisor_reason == "approved feedback"


def test_metric_row_excludes_large_artifacts_and_promotes_required_fields():
    event = TokenBudgetEvent(
        label="planner.plan_task",
        model="test-planner",
        condition_name="learned_mediator",
        prompt_tokens=11,
        completion_tokens=5,
        total_tokens=16,
        budget_limit=100,
        budget_overflow_strategy="section_pack",
    )
    update = SkillUpdate(
        skill_id="executor",
        task_id="task-A",
        iteration=0,
        old_content="old content" * 100,
        new_content="new content" * 100,
        reasoning="short reason",
        old_skill_hash="old-hash",
        new_skill_hash="new-hash",
        skill_version="iter_0000",
    )
    record = IterationRecord(
        iteration=0,
        task_id="task-A",
        task_spec=TaskSpec(task_id="task-A", instruction="do it", iteration=0),
        execution_trace=ExecutionTrace(
            task_id="task-A",
            iteration=0,
            stdout="long harbor output" * 100,
            reward=0.0,
            status="ok",
            token_usage=TokenUsage(input_tokens=4, output_tokens=6),
            run_id="job-123",
            harbor_trial_id="trial-123",
            harbor_paths={"job": "/tmp/job", "trial": "/tmp/job/trial"},
        ),
        skill_update=update,
        reward=0.0,
        total_tokens=26,
        llm_token_events=[event],
        run_id="job-123",
        condition_name="learned_mediator",
        seed=42,
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
        },
        executor_agent="opencode",
        skill_hashes={"executor": "hash"},
        skill_version="iter_0000",
        proposal_ids=["proposal-1"],
        advisor_decision="rejected",
        advisor_reason="advisor rejected the batch",
        advisor_rejection_id="rejection-1",
        success=True,
        verifier_status="ok",
        task_category="build",
        task_difficulty="easy",
        expected_reward_range=(0.0, 1.0),
        verifier_type="pytest",
    )

    row = metric_row(record)

    assert "task_spec" not in row
    assert "execution_trace" not in row
    assert "mediator_report" not in row
    assert "skill_update" not in row
    assert "long harbor output" not in str(row)
    assert "old content" not in str(row)
    assert "new content" not in str(row)
    assert row["planner_model"] == "test-planner"
    assert row["executor_model"] == "test-executor"
    assert row["mediator_model"] == "test-mediator"
    assert row["harbor_job_path"] == "/tmp/job"
    assert row["harbor_trial_path"] == "/tmp/job/trial"
    assert row["harbor_trial_id"] == "trial-123"
    assert row["proposal_ids"] == ["proposal-1"]
    assert row["advisor_decision"] == "rejected"
    assert row["advisor_reason"] == "advisor rejected the batch"
    assert row["advisor_rejection_id"] == "rejection-1"
    assert row["success"] is False
    assert row["verifier_status"] == "task_failed"
    assert row["prompt_tokens_by_agent"] == {"executor": 4, "planner": 11}
    assert row["completion_tokens_by_agent"] == {"executor": 6, "planner": 5}
    assert row["total_tokens_by_agent"] == {"executor": 10, "planner": 16}
    assert row["llm_token_events"][0]["budget_limit"] == 100
    assert row["expected_reward_range"] == [0.0, 1.0]
    assert row["skill_updates"] == [
        {
            "skill_id": "executor",
            "task_id": "task-A",
            "iteration": 0,
            "old_skill_hash": "old-hash",
            "new_skill_hash": "new-hash",
            "skill_version": "iter_0000",
            "reasoning": "short reason",
            "provenance": None,
        }
    ]


def test_zero_reward_harbor_run_is_logged_as_task_failure():
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        }
    )
    orch._previous_reward_by_task = {}
    orch.planner = _LLMBackedComponent(_DrainClient("planner.plan_task"))
    orch.mediator = _LLMBackedComponent(_DrainClient("mediator.process_trace"))
    orch.skill_advisor = _LLMBackedComponent(_DrainClient("skill_advisor.review"))

    record = build_iteration_record(
        task_id="task-A",
        iteration=0,
        condition="no_feedback",
        duration_sec=0.0,
        task_spec=TaskSpec(task_id="task-A", instruction="do it", iteration=0),
        trace=ExecutionTrace(
            task_id="task-A",
            iteration=0,
            reward=0.0,
            status="ok",
            run_id="job-123",
        ),
        report=None,
        skill_update=None,
        mediator_entry_id=None,
        planner_entry_id=None,
        skill_hashes={"executor": "hash"},
        task_metadata={
            "task_category": "build",
            "task_difficulty": "easy",
            "expected_reward_range": (0.0, 1.0),
            "verifier_type": "pytest",
        },
        llm_token_events=orch._drain_llm_token_events(),
        config=orch.config,
        previous_reward_by_task=orch._previous_reward_by_task,
    )

    assert record.success is False
    assert record.verifier_status == "task_failed"
    assert record.reward == 0.0


class _NoCallPlanner:
    llm_client = _DrainClient("unused")

    def __getattr__(self, name):
        raise AssertionError(f"planner should not be called: {name}")


class _NoCallMediator:
    llm_client = _DrainClient("unused")

    def __getattr__(self, name):
        raise AssertionError(f"mediator should not be called: {name}")


class _NoCallAdvisor:
    llm_client = _DrainClient("unused")

    def __getattr__(self, name):
        raise AssertionError(f"advisor should not be called: {name}")


class _SkippingMediator:
    llm_client = _DrainClient("unused")

    async def mediate_trace(
        self,
        condition: str,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport | None:
        return None


class _NoCallExecutor:
    def __getattr__(self, name):
        raise AssertionError(f"executor should not be called: {name}")


class _EmptySkillStore:
    def read_skill(self, skill_name: str) -> str | None:
        return None

    def skill_hashes(self) -> dict[str, str]:
        return {}


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
    orch.config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        }
    )
    orch.experiment_dir = tmp_path
    orch.skill_advisor = _NoCallAdvisor()
    orch._proposal_buffer = []
    orch._previous_report_by_task = {}

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
    task_config: dict = {}


class _AnyTaskRepo:
    def resolve(self, task_id: str):
        return _ResolvedTask()


class _ValidationExecutor:
    def __init__(
        self,
        *,
        current_rewards: dict[str, float | None],
        candidate_rewards: dict[str, float | None],
    ) -> None:
        self.current_rewards = current_rewards
        self.candidate_rewards = candidate_rewards
        self.calls: list[tuple[str, str]] = []

    async def execute_task(
        self,
        task_spec: TaskSpec,
        skill_texts: list[str],
    ) -> ExecutionTrace:
        skill_text = skill_texts[0] if skill_texts else ""
        self.calls.append((task_spec.task_id, skill_text))
        rewards = (
            self.candidate_rewards
            if skill_text == "new"
            else self.current_rewards
        )
        reward = rewards[task_spec.task_id]
        if reward is None:
            return ExecutionTrace(
                task_id=task_spec.task_id,
                iteration=task_spec.iteration,
                status="env_failure",
                error_kind="validation_unusable",
            )
        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            status="ok",
            reward=reward,
        )


class _PlannerLLM:
    model = "test-model"

    def drain_token_events(self) -> list[TokenBudgetEvent]:
        return []


class _RecordingPlanner:
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


class _RunIterationSkillStore:
    def __init__(self) -> None:
        self.executor_content = "# Executor\n"

    def read_skill(self, skill_name: str) -> str | None:
        if skill_name == "executor":
            return self.executor_content
        if skill_name == "planner":
            return None
        raise AssertionError(f"unexpected skill read: {skill_name}")

    def write_skill(self, skill_name: str, content: str) -> None:
        raise AssertionError(f"skill should not be written: {skill_name}")

    def skill_hashes(self) -> dict[str, str]:
        return {
            "executor": SkillStore.content_hash(self.executor_content),
        }


class _ProposalPlanner:
    llm_client = _DrainClient("planner.plan_task")

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
        return TaskSpec(
            task_id=task_id,
            instruction=base_instruction,
            iteration=iteration,
        )

    async def suggest_skill_revision(
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
            new_content=f"{current_skill_content}\n# Proposed\n",
            reasoning="proposal",
        )


class _RewardExecutor:
    async def execute_task(
        self,
        task_spec: TaskSpec,
        skill_texts: list[str],
    ) -> ExecutionTrace:
        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            status="ok",
            reward=0.5,
        )


class _ExposedMediator:
    llm_client = _DrainClient("mediator.process_trace")

    async def mediate_trace(
        self,
        condition: str,
        trace: ExecutionTrace,
        task_context: TaskSpec,
    ) -> MediatorReport:
        return MediatorReport(
            task_id=task_context.task_id,
            iteration=task_context.iteration,
            content="planner-visible feedback",
        )

    async def compact_feedback(self, report: MediatorReport) -> MediatorSignal:
        return MediatorSignal(headline=report.content)


class _RejectingAdvisor:
    llm_client = _DrainClient("skill_advisor.review")
    last_rejection_reason = "advisor rejected the batch"

    async def review(
        self,
        current_skill: str,
        proposals: list[SkillProposal],
    ) -> str | None:
        return None


@pytest.mark.asyncio
async def test_previous_report_prior_context_is_keyed_by_task(tmp_path):
    planner = _RecordingPlanner()
    orch = Orchestrator.__new__(Orchestrator)
    orch.planner = planner
    orch.executor = _EnvFailureExecutor()
    orch.mediator = _SkippingMediator()
    orch.skill_store = _EmptySkillStore()
    orch.artifact_store = ArtifactStore(base_dir=tmp_path / "artifacts")
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch.benchmark_repo = _AnyTaskRepo()
    orch.config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        }
    )
    orch.experiment_dir = tmp_path
    orch.skill_advisor = _NoCallAdvisor()
    orch._proposal_buffer = []
    orch._previous_report_by_task = {
        "task-A": MediatorReport(task_id="task-A", iteration=0, content="task-A report"),
    }
    orch._previous_reward_by_task = {}
    _attach_executor_skill_gate(orch)

    await orch._run_iteration("task-B", 1)
    await orch._run_iteration("task-A", 1)

    assert planner.prior_contexts["task-B"] is None
    assert planner.prior_contexts["task-A"] == "task-A report"


@pytest.mark.asyncio
async def test_run_iteration_logs_advisor_rejection_in_metrics_record(tmp_path):
    orch = Orchestrator.__new__(Orchestrator)
    orch.planner = _ProposalPlanner()
    orch.executor = _RewardExecutor()
    orch.mediator = _ExposedMediator()
    orch.skill_store = _RunIterationSkillStore()
    orch.artifact_store = ArtifactStore(base_dir=tmp_path / "artifacts")
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch.benchmark_repo = _AnyTaskRepo()
    orch.config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        }
    )
    orch.config.experiment.advisor_buffer_max = 1
    orch.experiment_dir = tmp_path
    orch.skill_advisor = _RejectingAdvisor()
    orch._proposal_buffer = []
    orch._previous_report_by_task = {}
    orch._previous_reward_by_task = {}
    _attach_executor_skill_gate(orch)

    record = await orch._run_iteration("task-A", 1)
    row = metric_row(record)

    assert record.skill_update is None
    assert record.advisor_decision == "rejected"
    assert record.advisor_reason == "advisor rejected the batch"
    assert len(record.proposal_ids) == 1
    rejections = orch.history_store.query_rejected_proposals()
    assert len(rejections) == 1
    assert record.advisor_rejection_id == rejections[0].rejection_id
    assert row["advisor_decision"] == "rejected"
    assert row["advisor_reason"] == "advisor rejected the batch"
    assert row["advisor_rejection_id"] == rejections[0].rejection_id
    assert row["proposal_ids"] == record.proposal_ids


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

    async def suggest_skill_revision(
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


class _BatchPatchPlanner:
    step = 7

    async def suggest_skill_revision_batch(
        self,
        current_skill_content: str,
        feedback: str | None,
        edit_history: list,
        *,
        skill_id: str,
        task_ids: list[str],
        iteration: int = 0,
    ) -> list[SkillUpdateCandidate]:
        return [
            SkillUpdateCandidate(
                candidate_id="broad",
                skill_id=skill_id,
                update_kind="add_procedure",
                hypothesis="larger update",
                old_content=current_skill_content,
                new_content="broad",
                reasoning="too broad",
                audit_score=0.1,
            ),
            SkillUpdateCandidate(
                candidate_id="targeted",
                skill_id=skill_id,
                update_kind="narrow_clarification",
                hypothesis="targeted update",
                old_content=current_skill_content,
                new_content="new",
                reasoning="targeted",
                audit_score=0.9,
            ),
        ]


def _advisor_validation_orchestrator(
    tmp_path: Path,
    *,
    current_rewards: dict[str, float | None],
    candidate_rewards: dict[str, float | None],
    proposal_task_ids: list[str],
    planner: object | None = None,
) -> Orchestrator:
    orch = Orchestrator.__new__(Orchestrator)
    orch.config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        }
    )
    orch.config.experiment.advisor_buffer_max = len(proposal_task_ids)
    orch.skill_store = _MemorySkillStore()
    orch.artifact_store = ArtifactStore(base_dir=tmp_path / "artifacts")
    orch.history_store = HistoryStore(history_dir=tmp_path / "history")
    orch.benchmark_repo = _AnyTaskRepo()
    orch.executor = _ValidationExecutor(
        current_rewards=current_rewards,
        candidate_rewards=candidate_rewards,
    )
    orch.skill_advisor = _ApprovingAdvisor()
    orch.planner = planner or _PatchPlanner()
    _attach_executor_skill_gate(orch)
    orch._proposal_buffer = [
        SkillProposal(
            iteration=0,
            task_id=task_id,
            old_content="",
            new_content=task_id,
        )
        for task_id in proposal_task_ids
    ]
    return orch


def _validation_result_json(tmp_path: Path) -> dict:
    result_path = (
        tmp_path
        / "artifacts"
        / "validation"
        / "coevo-iter-0003"
        / "result.json"
    )
    return json.loads(result_path.read_text())


@pytest.mark.asyncio
async def test_advisor_patch_preserves_buffered_task_provenance(tmp_path):
    orch = _advisor_validation_orchestrator(
        tmp_path,
        current_rewards={"task-A": 0.4, "task-B": 0.7},
        candidate_rewards={"task-A": 0.4, "task-B": 0.8},
        proposal_task_ids=["task-B", "task-A"],
    )

    proposal_ids = [
        proposal.proposal_id
        for proposal in orch._proposal_buffer
    ]
    gate = orch.executor_skill_gate
    update = await gate.review_and_patch(
        iteration=3,
        proposal_buffer=orch._proposal_buffer,
    )

    assert update is not None
    assert update.task_id == "task-A,task-B"
    assert update.iteration == 3
    assert update.reasoning == "approved"
    assert update.old_skill_hash == SkillStore.content_hash("old")
    assert update.new_skill_hash == SkillStore.content_hash("new")
    assert update.provenance is not None
    assert update.provenance.kind == "advisor_batch"
    assert update.provenance.batch_id == "coevo-iter-0003"
    assert update.provenance.task_ids == ["task-A", "task-B"]
    assert update.provenance.base_skill_hash == SkillStore.content_hash("old")
    assert update.provenance.reason == "approved"
    assert update.provenance.rollback_snapshot == "iter_0002"
    assert update.provenance.validation is not None
    assert update.provenance.validation.decision == "accepted"
    assert update.provenance.validation.current_mean_reward == pytest.approx(0.55)
    assert update.provenance.validation.candidate_mean_reward == pytest.approx(0.6)
    assert [ref.task_id for ref in update.provenance.proposal_refs] == [
        "task-B",
        "task-A",
    ]
    dumped = IterationRecord(
        iteration=3,
        task_id="task-A",
        skill_update=update,
    ).model_dump_json()
    assert '"kind":"advisor_batch"' in dumped
    assert '"proposal_refs"' in dumped
    loaded = IterationRecord.model_validate_json(dumped)
    assert loaded.skill_update is not None
    assert loaded.skill_update.provenance is not None
    assert loaded.skill_update.provenance.kind == "advisor_batch"
    assert len(orch.skill_advisor.seen) == 2
    assert orch._proposal_buffer == []
    assert gate.last_advisor_decision == "approved"
    assert gate.last_advisor_reason == "approved"
    assert gate.last_proposal_ids == proposal_ids
    assert gate.last_rejection_id is None
    assert orch.skill_store.writes == [("executor", "new")]
    assert orch.executor.calls == [
        ("task-A", "old"),
        ("task-A", "new"),
        ("task-B", "old"),
        ("task-B", "new"),
    ]
    assert _validation_result_json(tmp_path)["decision"] == "accepted"


@pytest.mark.asyncio
async def test_advisor_patch_selects_from_candidate_batch_and_persists_artifact(
    tmp_path,
):
    orch = _advisor_validation_orchestrator(
        tmp_path,
        current_rewards={"task-A": 0.4},
        candidate_rewards={"task-A": 0.8},
        proposal_task_ids=["task-A"],
        planner=_BatchPatchPlanner(),
    )

    update = await orch.executor_skill_gate.review_and_patch(
        iteration=3,
        proposal_buffer=orch._proposal_buffer,
    )

    assert update is not None
    assert update.new_content == "new"
    assert update.provenance is not None
    assert update.provenance.selected_candidate_id == "targeted"
    assert update.provenance.selected_update_kind == "narrow_clarification"
    assert update.provenance.candidate_batch_id == "coevo-iter-0003-executor-candidates"
    assert [ref.candidate_id for ref in update.provenance.candidate_refs] == [
        "broad",
        "targeted",
    ]

    artifact_path = (
        tmp_path
        / "artifacts"
        / "candidate_batches"
        / "coevo-iter-0003-executor-candidates.json"
    )
    artifact = json.loads(artifact_path.read_text())
    assert artifact["selected_candidate_id"] == "targeted"
    assert [candidate["new_content"] for candidate in artifact["candidates"]] == [
        "broad",
        "new",
    ]


@pytest.mark.asyncio
async def test_advisor_patch_rejected_when_validation_task_regresses(tmp_path):
    orch = _advisor_validation_orchestrator(
        tmp_path,
        current_rewards={"task-A": 0.8, "task-B": 0.4},
        candidate_rewards={"task-A": 0.7, "task-B": 0.9},
        proposal_task_ids=["task-A", "task-B"],
    )

    proposal_ids = [
        proposal.proposal_id
        for proposal in orch._proposal_buffer
    ]
    gate = orch.executor_skill_gate
    update = await gate.review_and_patch(
        iteration=3,
        proposal_buffer=orch._proposal_buffer,
    )

    assert update is None
    assert orch._proposal_buffer == []
    assert orch.skill_store.writes == []
    assert gate.last_advisor_decision == "approved"
    assert gate.last_advisor_reason == "approved"
    assert gate.last_proposal_ids == proposal_ids
    result = _validation_result_json(tmp_path)
    assert result["decision"] == "rejected"
    assert result["reason"] == "task_regression"

    rejections = orch.history_store.query_rejected_proposals()
    assert len(rejections) == 1
    rejection = rejections[0]
    assert gate.last_rejection_id == rejection.rejection_id
    assert rejection.batch_id == "coevo-iter-0003"
    assert rejection.reason == "validation: task_regression"
    assert rejection.advisor_feedback == "approved"
    assert rejection.validation is not None
    assert rejection.validation.decision == "rejected"
    assert rejection.validation.reason == "task_regression"
    assert [proposal.task_id for proposal in rejection.proposals] == [
        "task-A",
        "task-B",
    ]
    assert rejection.base_skill_hash == SkillStore.content_hash("old")


@pytest.mark.asyncio
async def test_advisor_patch_rejected_when_validation_trace_unusable(tmp_path):
    orch = _advisor_validation_orchestrator(
        tmp_path,
        current_rewards={"task-A": 0.8},
        candidate_rewards={"task-A": None},
        proposal_task_ids=["task-A"],
    )

    gate = orch.executor_skill_gate
    update = await gate.review_and_patch(
        iteration=3,
        proposal_buffer=orch._proposal_buffer,
    )

    assert update is None
    assert orch._proposal_buffer == []
    assert orch.skill_store.writes == []
    assert gate.last_advisor_decision == "approved"
    assert gate.last_advisor_reason == "approved"
    result = _validation_result_json(tmp_path)
    assert result["decision"] == "rejected"
    assert result["reason"] == "unusable_validation_trace"
    rejections = orch.history_store.query_rejected_proposals()
    assert len(rejections) == 1
    assert gate.last_rejection_id == rejections[0].rejection_id
