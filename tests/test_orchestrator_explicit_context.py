from __future__ import annotations

import pytest

from mediated_coevo.execution.models import ContextPack, TaskProfile, empty_context_pack
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from tests.test_feedback_conditions import _orchestrator


@pytest.mark.asyncio
async def test_explicit_context_seam_bypasses_legacy_discovery_and_emission(
    tmp_path,
    monkeypatch,
):
    orchestrator, planner, _ = _orchestrator(tmp_path, "learned_mediator")

    async def forbidden(*args, **kwargs):
        raise AssertionError("legacy context or diffusion emission was called")

    monkeypatch.setattr(orchestrator, "_build_prior_context", forbidden)
    monkeypatch.setattr(orchestrator, "_emit_diffusion_artifacts", forbidden)
    context = ContextPack(
        text="causally selected transfer context",
        eligible_artifact_ids=("artifact-0", "artifact-1"),
        selected_artifact_ids=("artifact-1",),
        rendered_artifact_ids=("artifact-1",),
        source_task_ids=("task-B",),
        snapshot_id="snapshot-1",
        policy_name="langchain_diffusion_policy",
        token_count=7,
        max_context_tokens=10,
        metadata={
            "graph_policy": "langchain_graph",
            "eligible_count": 2,
            "selected_count": 1,
            "rendered_count": 1,
        },
    )

    record = await orchestrator.execute_task_with_context(
        task_id="task-A",
        position=1,
        task=TaskProfile(task_id="task-A", instruction="execute"),
        context=context,
    )

    assert planner.prior_contexts == {
        "task-A": "causally selected transfer context"
    }
    assert record.diffusion_enabled is True
    assert record.diffusion_policy == "langchain_diffusion_policy"
    assert record.diffusion_graph == "langchain_graph"
    assert record.graph_snapshot_id == "snapshot-1"
    assert record.diffusion_artifacts_eligible == 2
    assert record.diffusion_artifacts_selected == 1
    assert record.diffusion_artifacts_rendered == 1
    assert record.transfer_context_kind == "diffusion"
    assert record.transfer_context_tokens == 7
    assert record.total_planner_prior_context_tokens == 7
    assert record.max_transfer_context_tokens == 10
    assert record.max_total_prior_context_tokens == 10
    assert record.source_task_ids == ["task-B"]
    assert record.reward_after_diffusion_context == 0.5
    assert orchestrator.diffusion_store.query_artifacts(recent=None) == []


@pytest.mark.asyncio
async def test_explicit_empty_context_is_not_replaced_by_hidden_history(
    tmp_path,
    monkeypatch,
):
    orchestrator, planner, _ = _orchestrator(tmp_path, "learned_mediator")

    async def forbidden(*args, **kwargs):
        raise AssertionError("legacy prior discovery was called")

    monkeypatch.setattr(orchestrator, "_build_prior_context", forbidden)

    record = await orchestrator.execute_task_with_context(
        task_id="task-A",
        position=0,
        task=TaskProfile(task_id="task-A", instruction="execute"),
        context=empty_context_pack(),
    )

    assert planner.prior_contexts == {"task-A": None}
    assert record.diffusion_enabled is False
    assert record.diffusion_policy == "none"
    assert record.graph_snapshot_id is None
    assert record.diffusion_artifacts_eligible == 0
    assert record.diffusion_artifacts_selected == 0
    assert record.diffusion_artifacts_rendered == 0
    assert record.transfer_context_kind == "none"
    assert record.same_task_prior_tokens == 0
    assert record.total_planner_prior_context_tokens == 0


@pytest.mark.asyncio
async def test_explicit_context_executes_the_frozen_task_occurrence(
    tmp_path,
    monkeypatch,
):
    orchestrator, _, _ = _orchestrator(tmp_path, "no_feedback")

    def forbidden_resolve(*args, **kwargs):
        raise AssertionError("explicit execution re-resolved mutable task metadata")

    monkeypatch.setattr(orchestrator.benchmark_repo, "resolve", forbidden_resolve)
    frozen = TaskProfile(
        task_id="task-A",
        instruction="frozen occurrence instruction",
        task_config={"metadata": {"category": "frozen-category"}},
    )

    record = await orchestrator.execute_task_with_context(
        task_id="task-A",
        position=0,
        task=frozen,
        context=empty_context_pack(),
    )

    assert record.task_spec is not None
    assert record.task_spec.instruction == "frozen occurrence instruction"
    assert record.task_category == "frozen-category"
    provenance = orchestrator.take_explicit_execution_provenance(
        task_id="task-A",
        position=0,
    )
    assert provenance["judge_reward"] == 0.5
    assert orchestrator.take_explicit_execution_provenance(
        task_id="task-A",
        position=0,
    ) == {}


@pytest.mark.asyncio
async def test_explicit_context_skips_legacy_history_and_skill_update_pipeline(
    tmp_path,
    monkeypatch,
):
    orchestrator, _, _ = _orchestrator(tmp_path, "learned_mediator")

    async def forbidden_async(*args, **kwargs):
        raise AssertionError("legacy history or skill-update path was called")

    def forbidden_sync(*args, **kwargs):
        raise AssertionError("legacy history path was called")

    monkeypatch.setattr(
        "mediated_coevo.experiment.orchestrator.get_executor_proposal_feedback",
        forbidden_async,
    )
    monkeypatch.setattr(orchestrator, "_ask_planner_for_skill_proposal", forbidden_async)
    monkeypatch.setattr(orchestrator, "_record_history_entries", forbidden_async)
    monkeypatch.setattr(
        orchestrator.executor_skill_gate,
        "review_and_patch",
        forbidden_async,
    )
    monkeypatch.setattr(orchestrator.history_store, "tag_outcome", forbidden_sync)

    record = await orchestrator.execute_task_with_context(
        task_id="task-A",
        position=0,
        task=TaskProfile(task_id="task-A", instruction="frozen"),
        context=empty_context_pack(),
    )

    assert record.history_entry_ids == {}
    assert record.skill_update is None


@pytest.mark.asyncio
async def test_explicit_context_does_not_read_mediator_history_or_reward_cache(
    tmp_path,
    monkeypatch,
):
    orchestrator, _, mediator = _orchestrator(tmp_path, "learned_mediator")
    hidden_store = object()
    mediator._artifact_store = hidden_store
    orchestrator._previous_reward_by_task = {"task-A": 0.1}

    async def mediate_without_history(condition, trace, task_context):
        del condition, trace, task_context
        assert mediator._artifact_store is None
        return None

    monkeypatch.setattr(mediator, "mediate_trace", mediate_without_history)

    record = await orchestrator.execute_task_with_context(
        task_id="task-A",
        position=1,
        task=TaskProfile(task_id="task-A", instruction="frozen"),
        context=empty_context_pack(),
    )

    assert mediator._artifact_store is hidden_store
    assert orchestrator._previous_reward_by_task == {"task-A": 0.1}
    assert record.delta_reward is None


@pytest.mark.asyncio
async def test_explicit_context_rejects_wrong_planner_identity_before_execution(
    tmp_path,
    monkeypatch,
):
    orchestrator, planner, _ = _orchestrator(tmp_path, "no_feedback")

    async def wrong_plan(**kwargs):
        del kwargs
        return TaskSpec(task_id="../escape", instruction="wrong", iteration=0)

    async def forbidden_execution(*args, **kwargs):
        raise AssertionError("executor must not receive a mismatched task")

    monkeypatch.setattr(planner, "plan_task", wrong_plan)
    monkeypatch.setattr(orchestrator.executor, "execute_task", forbidden_execution)

    with pytest.raises(ValueError, match="planner returned a different"):
        await orchestrator.execute_task_with_context(
            task_id="task-A",
            position=0,
            task=TaskProfile(task_id="task-A", instruction="frozen"),
            context=empty_context_pack(),
        )


@pytest.mark.asyncio
async def test_explicit_context_rejects_wrong_trace_identity_before_persistence(
    tmp_path,
    monkeypatch,
):
    orchestrator, _, _ = _orchestrator(tmp_path, "no_feedback")

    async def wrong_trace(*args, **kwargs):
        del args, kwargs
        return ExecutionTrace(
            task_id="../../outside",
            iteration=0,
            reward=1.0,
            status="ok",
        )

    monkeypatch.setattr(orchestrator.executor, "execute_task", wrong_trace)

    with pytest.raises(ValueError, match="executor returned a different"):
        await orchestrator.execute_task_with_context(
            task_id="task-A",
            position=0,
            task=TaskProfile(task_id="task-A", instruction="frozen"),
            context=empty_context_pack(),
        )

    assert list((tmp_path / "artifacts" / "traces").rglob("*.json")) == []
    assert not (tmp_path / "outside_iter0000.json").exists()


@pytest.mark.asyncio
async def test_explicit_trace_is_portable_before_later_stage_failure(
    tmp_path,
    monkeypatch,
):
    orchestrator, _, mediator = _orchestrator(tmp_path, "learned_mediator")
    external = tmp_path.parent / "remote-job"

    async def trace_with_external_path(task_spec, skill_texts):
        del skill_texts
        return ExecutionTrace(
            task_id=task_spec.task_id,
            iteration=task_spec.iteration,
            reward=1.0,
            status="ok",
            harbor_paths={"job": str(external.resolve())},
        )

    async def fail_mediator(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("mediator failed")

    monkeypatch.setattr(orchestrator.executor, "execute_task", trace_with_external_path)
    monkeypatch.setattr(mediator, "mediate_trace", fail_mediator)

    with pytest.raises(RuntimeError, match="mediator failed"):
        await orchestrator.execute_task_with_context(
            task_id="task-A",
            position=0,
            task=TaskProfile(task_id="task-A", instruction="frozen"),
            context=empty_context_pack(),
        )

    trace_text = next(
        (tmp_path / "artifacts" / "traces").glob("*.json")
    ).read_text(encoding="utf-8")
    assert str(external.resolve()) not in trace_text
