from __future__ import annotations

import pytest

from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.experiment.conditions import get_prior_context
from mediated_coevo.core.config import Config
from mediated_coevo.llm.client import LLMClient
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.trace import ExecutionTrace, TokenUsage
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.runtime.token_budget import (
    BudgetSection,
    TokenBudgetEvent,
    TokenBudgetExceeded,
    TokenCountingError,
    count_message_tokens,
    count_text_tokens,
    fit_text_to_tokens,
    pack_sections,
)
from tests.config_helpers import diffusion_config, experiment_config


def test_token_count_falls_back_when_litellm_counter_fails(monkeypatch):
    import litellm
    import tiktoken

    encoding_names = []

    class _Encoding:
        def encode(self, text):
            return list(text)

    def _raise_counter(*args, **kwargs):
        raise RuntimeError("counter unavailable")

    def _get_encoding(name):
        encoding_names.append(name)
        return _Encoding()

    monkeypatch.setattr(litellm, "token_counter", _raise_counter)
    monkeypatch.setattr(tiktoken, "get_encoding", _get_encoding)

    assert count_text_tokens("test-model", "abcd") == 4
    assert (
        count_message_tokens(
            "test-model",
            [{"role": "user", "content": "abcd"}],
        )
        > 0
    )
    assert encoding_names == ["o200k_base", "o200k_base"]


def test_token_count_raises_when_litellm_and_fallback_fail(monkeypatch):
    import litellm
    import tiktoken

    def _raise_counter(*args, **kwargs):
        raise RuntimeError("counter unavailable")

    def _raise_encoding(name):
        raise ValueError(name)

    monkeypatch.setattr(litellm, "token_counter", _raise_counter)
    monkeypatch.setattr(tiktoken, "get_encoding", _raise_encoding)

    with pytest.raises(TokenCountingError):
        count_text_tokens("test-model", "abcd")

    with pytest.raises(TokenCountingError):
        count_message_tokens(
            "test-model",
            [{"role": "user", "content": "abcd" * 10}],
        )


def test_token_count_requires_model_for_nonempty_inputs():
    with pytest.raises(TokenCountingError):
        count_text_tokens("", "abcd")

    with pytest.raises(TokenCountingError):
        count_message_tokens("", [{"role": "user", "content": "abcd"}])


def test_fit_text_to_tokens_preserves_head_and_tail():
    text = "START " + ("middle " * 200) + " END"
    fitted = fit_text_to_tokens("test-model", text, 20)

    assert "START" in fitted
    assert "END" in fitted
    assert count_text_tokens("test-model", fitted) <= 20


def test_pack_sections_truncates_optional_section_to_fit():
    packed = pack_sections(
        "test-model",
        [
            BudgetSection("required", "Required instruction.", required=True),
            BudgetSection("optional", "OPTIONAL " * 500),
        ],
        budget_limit=30,
    )

    assert "Required instruction." in packed
    assert count_text_tokens("test-model", packed) <= 30


def test_pack_sections_reserves_budget_for_later_required_sections():
    packed = pack_sections(
        "test-model",
        [
            BudgetSection("required_start", "Required start.", required=True),
            BudgetSection("optional_middle", "OPTIONAL " * 500),
            BudgetSection("required_end", "Required end.", required=True),
        ],
        budget_limit=20,
    )

    assert "Required start." in packed
    assert "Required end." in packed
    assert count_text_tokens("test-model", packed) <= 20


@pytest.mark.asyncio
async def test_llm_client_raises_before_over_budget_call(monkeypatch):
    client = LLMClient(model="test-model")

    async def _should_not_call(**kwargs):
        raise AssertionError("LLM call should be blocked before network")

    monkeypatch.setattr(client, "_call_with_retry", _should_not_call)

    with pytest.raises(TokenBudgetExceeded):
        await client.complete(
            messages=[{"role": "user", "content": "too many tokens " * 200}],
            prompt_budget=5,
            budget_label="test.over_budget",
        )


@pytest.mark.asyncio
async def test_llm_client_records_budget_event(monkeypatch):
    client = LLMClient(model="test-model")

    class _Message:
        content = '{"ok": true}'

    class _Choice:
        message = _Message()

    class _Usage:
        prompt_tokens = 7
        completion_tokens = 3
        total_tokens = 10

    class _Response:
        choices = [_Choice()]
        usage = _Usage()

        def model_dump(self):
            return {}

    async def _fake_call(**kwargs):
        return _Response()

    monkeypatch.setattr(client, "_call_with_retry", _fake_call)

    await client.complete(
        messages=[{"role": "user", "content": "small"}],
        prompt_budget=100,
        budget_label="test.call",
        condition_name="full_traces",
    )

    events = client.drain_token_events()
    assert len(events) == 1
    assert events[0].label == "test.call"
    assert events[0].condition_name == "full_traces"
    assert events[0].prompt_tokens == 7
    assert events[0].completion_tokens == 3
    assert events[0].total_tokens == 10


@pytest.mark.asyncio
async def test_llm_client_raises_when_usage_is_missing(monkeypatch):
    client = LLMClient(model="test-model")

    class _Message:
        content = '{"ok": true}'

    class _Choice:
        message = _Message()

    class _Response:
        choices = [_Choice()]
        usage = None

        def model_dump(self):
            return {}

    async def _fake_call(**kwargs):
        return _Response()

    monkeypatch.setattr(client, "_call_with_retry", _fake_call)

    with pytest.raises(RuntimeError, match="usage accounting"):
        await client.complete(messages=[{"role": "user", "content": "small"}])


@pytest.mark.asyncio
async def test_full_trace_prior_context_respects_configured_budget(tmp_path):
    config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        },
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    config.budgets.trace_excerpt_tokens = 20
    config.budgets.historical_summary_tokens = 30
    store = ArtifactStore(base_dir=tmp_path / "artifacts")

    class _Compactor:
        model = "test-model"

        async def complete(self, **kwargs):
            return {
                "content": '{"headline": "START", "evidence": "END"}',
                "input_tokens": 1,
                "output_tokens": 1,
                "model": self.model,
                "raw": {},
            }

    store.store_trace(
        ExecutionTrace(
            task_id="task-A",
            iteration=0,
            reward=0.0,
            status="ok",
            stderr="START " + ("middle " * 300) + " END",
        )
    )

    context = await get_prior_context(
        condition="full_traces",
        task_id="task-A",
        artifact_store=store,
        previous_report=None,
        shared_notes=None,
        llm_client=_Compactor(),
        model="test-model",
        budgets=config.budgets,
        condition_name="full_traces",
    )

    assert context is not None
    assert (
        count_text_tokens("test-model", context)
        <= config.budgets.historical_summary_tokens
    )


def test_planner_constructed_prompt_fits_budget():
    config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        },
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    config.budgets.max_skill_tokens = 30
    config.budgets.mediator_report_tokens = 30
    config.budgets.planner_context_tokens = 500
    planner = PlannerAgent(LLMClient(model="test-model"))
    planner.configure_token_budget(config.budgets, condition_name="learned_mediator")
    planner.set_skill_context(
        executor_skills="# Skill\n" + ("tool guidance " * 300),
        skill_refiner="# Refiner\n" + ("edit guidance " * 300),
    )

    messages = planner.construct_messages(
        {
            "action": "plan_task",
            "task_id": "task-A",
            "base_instruction": "Fix the build.",
            "mediator_report": "prior feedback " * 300,
        }
    )

    assert (
        count_message_tokens("test-model", messages)
        <= config.budgets.planner_context_tokens
    )


@pytest.mark.asyncio
async def test_planner_compacts_large_benchmark_instruction_before_prompting(
    monkeypatch,
):
    config = Config(
        models={
            "planner": "test-planner",
            "executor": "test-executor",
            "mediator": "test-mediator",
            "judge": "test-judge",
        },
        experiment=experiment_config(),
        diffusion=diffusion_config(),
    )
    config.budgets.max_skill_tokens = 30
    config.budgets.mediator_report_tokens = 30
    config.budgets.planner_context_tokens = 500
    planner = PlannerAgent(LLMClient(model="test-model"))
    planner.configure_token_budget(config.budgets, condition_name="learned_mediator")
    planner.set_skill_context(
        executor_skills="# Skill\n" + ("tool guidance " * 300),
        skill_refiner="# Refiner\n" + ("edit guidance " * 300),
    )
    compact_calls = []

    async def _fake_compact_text_for_context(text, **kwargs):
        compact_calls.append((text, kwargs))
        return "COMPACTED ISSUE SIGNAL"

    async def _fake_process(context):
        messages = planner.construct_messages(context)
        assert (
            count_message_tokens("test-model", messages)
            <= config.budgets.planner_context_tokens
        )
        return {
            "parsed": {"instruction": context["base_instruction"], "reasoning": "ok"},
            "content": "{}",
            "input_tokens": 0,
            "output_tokens": 0,
        }

    monkeypatch.setattr(
        "mediated_coevo.evolution.compactor.compact_text_for_context",
        _fake_compact_text_for_context,
    )
    monkeypatch.setattr(planner, "process", _fake_process)

    task_spec = await planner.plan_task(
        task_id="large-skillflow-task",
        base_instruction="START " + ("required issue text " * 5000) + " END",
        prior_context="prior feedback " * 300,
        current_skills=["# executor"],
        iteration=0,
    )

    assert compact_calls
    assert "required issue text" in compact_calls[0][0]
    assert (
        compact_calls[0][1]["label"]
        == "benchmark instruction for large-skillflow-task"
    )
    assert "COMPACTED ISSUE SIGNAL" in task_spec.instruction


def test_planner_plan_prompt_contains_only_prior_feedback_section():
    planner = PlannerAgent(LLMClient(model="test-model"))

    messages = planner.construct_messages(
        {
            "action": "plan_task",
            "task_id": "task-A",
            "base_instruction": "Fix the build.",
            "mediator_report": "source_task=task-A iter=0 reward=0.25 OK",
            "feedback": "CURRENT OUTCOME iter=1 reward=0.90 SHOULD_NOT_APPEAR",
            "edit_history": [
                {
                    "iteration": 1,
                    "reasoning": "SHOULD_NOT_APPEAR",
                    "reward": 0.90,
                }
            ],
        }
    )
    prompt_text = "\n\n".join(message["content"] for message in messages)

    assert "## Feedback from previous execution" in prompt_text
    assert "iter=0 reward=0.25 OK" in prompt_text
    assert "## Execution Feedback" not in prompt_text
    assert "## Recent Edit History" not in prompt_text
    assert "CURRENT OUTCOME" not in prompt_text
    assert "iter=1" not in prompt_text
    assert "reward=0.90" not in prompt_text
    assert "SHOULD_NOT_APPEAR" not in prompt_text


def test_planner_update_prompt_separates_execution_feedback_from_plan_prompt():
    planner = PlannerAgent(LLMClient(model="test-model"))

    messages = planner.construct_messages(
        {
            "action": "update_skill",
            "current_skill": "# Executor\n",
            "feedback": "source_task=task-A iter=1 reward=0.90 OK",
            "edit_history": [
                {
                    "iteration": 0,
                    "reasoning": "previous edit",
                    "reward": 0.25,
                }
            ],
        }
    )
    prompt_text = "\n\n".join(message["content"] for message in messages)

    assert "## Execution Feedback" in prompt_text
    assert "source_task=task-A iter=1 reward=0.90 OK" in prompt_text
    assert "## Recent Edit History" in prompt_text
    assert "Plan a task for task_id" not in prompt_text
    assert "## Feedback from previous execution" not in prompt_text


def test_iteration_record_serializes_llm_token_events_and_total_tokens():
    event = TokenBudgetEvent(
        label="planner.plan_task",
        model="test-model",
        condition_name="learned_mediator",
        prompt_tokens=11,
        completion_tokens=5,
        total_tokens=16,
        budget_limit=100,
        budget_overflow_strategy="section_pack",
    )
    record = IterationRecord(
        iteration=0,
        task_id="task-A",
        execution_trace=ExecutionTrace(
            task_id="task-A",
            iteration=0,
            token_usage=TokenUsage(input_tokens=2, output_tokens=3),
        ),
        total_tokens=21,
        llm_token_events=[event],
        condition_name="learned_mediator",
        skill_hashes={"executor": "abc123"},
        skill_version="iter_0000",
    )
    dumped = record.model_dump()

    assert dumped["total_tokens"] == 21
    assert dumped["condition_name"] == "learned_mediator"
    assert dumped["llm_token_events"][0]["prompt_tokens"] == 11
    assert dumped["llm_token_events"][0]["budget_limit"] == 100
    assert dumped["skill_hashes"] == {"executor": "abc123"}
    assert dumped["skill_version"] == "iter_0000"
