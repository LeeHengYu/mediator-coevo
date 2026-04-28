from __future__ import annotations

import pytest

from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.conditions import get_prior_context
from mediated_coevo.config import Config
from mediated_coevo.llm.client import LLMClient
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.trace import ExecutionTrace, TokenUsage
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.token_budget import (
    BudgetSection,
    TokenBudgetEvent,
    TokenBudgetExceeded,
    TokenCountingError,
    count_message_tokens,
    count_text_tokens,
    fit_text_to_tokens,
    pack_sections,
)


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
    config = Config()
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
    config = Config()
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
    )
    dumped = record.model_dump()

    assert dumped["total_tokens"] == 21
    assert dumped["condition_name"] == "learned_mediator"
    assert dumped["llm_token_events"][0]["prompt_tokens"] == 11
    assert dumped["llm_token_events"][0]["budget_limit"] == 100
