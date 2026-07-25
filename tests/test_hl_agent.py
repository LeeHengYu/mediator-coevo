from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from mediated_coevo.cli import harness_registry
from mediated_coevo.cli import hl as hl_module
from mediated_coevo.hl.agent import HLWorkspace, run_hl_agent
from mediated_coevo.main import app

_FAMILIES = ("alpha", "beta", "gamma", "delta")
_POLICY = Path("src/mediated_coevo/diffusion/policy_agent.py")
_GRAPH = Path("src/mediated_coevo/diffusion/task_graph_agent.py")


def _project(tmp_path: Path) -> tuple[Path, Path]:
    for path in (_POLICY, _GRAPH):
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(f"# baseline {path.name}\n")
    prompt = tmp_path / "docs" / "hl_agent_prompt.md"
    prompt.parent.mkdir(parents=True)
    prompt.write_text("You are the independent HL agent for {CAMPAIGN}.")
    sequence = tmp_path / "data" / "sequences" / "sequence-1"
    iteration = sequence / "iter-1"
    iteration.mkdir(parents=True)
    (iteration / "sequence_spec.json").write_text(
        json.dumps(
            {
                "policy_seed": 7,
                "task_set_id": "families:alpha",
                "tasks": [{"task_id": "alpha/task-1"}],
            }
        )
    )
    return tmp_path, sequence


def _workspace(tmp_path: Path) -> tuple[HLWorkspace, Path]:
    project_root, sequence = _project(tmp_path)
    return (
        HLWorkspace(
            campaign="HLT",
            families=_FAMILIES,
            project_root=project_root,
            source_sequence=sequence,
        ),
        sequence,
    )


def test_targeted_update_writes_only_to_staging(tmp_path: Path) -> None:
    workspace, sequence = _workspace(tmp_path)
    workspace.record_decision(
        "TARGETED_UPDATE",
        sequence.as_posix(),
        "baseline",
        ["repeated harmful routing"],
        ["preserve successful graph reuse"],
    )

    workspace.prepare_update()
    workspace.write_staged_file(_POLICY.as_posix(), "# targeted update\n")

    assert (tmp_path / _POLICY).read_text() == "# baseline policy_agent.py\n"
    assert "+# targeted update" in workspace.inspect_staged_diff(_POLICY.as_posix())
    with pytest.raises(ValueError, match="outside the learned harness boundary"):
        workspace.write_staged_file(
            "src/mediated_coevo/experiment/sample_runner.py",
            "# forbidden\n",
        )


def test_hold_cannot_prepare_an_update(tmp_path: Path) -> None:
    workspace, sequence = _workspace(tmp_path)
    workspace.record_decision(
        "HOLD",
        sequence.as_posix(),
        "baseline",
        ["attribution is uncertain"],
        ["current baseline"],
    )

    with pytest.raises(ValueError, match="HOLD creates no harness update"):
        workspace.prepare_update()


def test_publish_freezes_next_update_without_mutating_baseline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workspace, sequence = _workspace(tmp_path)
    monkeypatch.setattr(harness_registry, "PROJECT_ROOT", tmp_path)
    workspace.record_decision(
        "TARGETED_UPDATE",
        sequence.as_posix(),
        "baseline",
        ["repeated selection failure"],
        ["successful fallback"],
    )
    workspace.prepare_update()
    workspace.write_staged_file(_POLICY.as_posix(), "# published update\n")

    result = workspace.publish_update()

    update = tmp_path / "data" / "experiments" / "HLT" / "update_0001"
    assert result["update"] == "update_0001"
    assert (update / "overlay" / _POLICY).read_text() == "# published update\n"
    assert (tmp_path / _POLICY).read_text() == "# baseline policy_agent.py\n"
    assert (update / "hl_decision.json").is_file()
    channel = json.loads(
        (
            tmp_path
            / "data"
            / "experiments"
            / "HLT"
            / "channels"
            / "promoted_harness.json"
        ).read_text()
    )
    assert channel["latest_update"] == "update_0001"


def test_run_hl_agent_uses_markdown_prompt_and_independent_tools(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    project_root, sequence = _project(tmp_path)
    captured: dict[str, Any] = {}

    class FakeAgent:
        def invoke(self, payload: dict[str, Any]) -> dict[str, Any]:
            captured["payload"] = payload
            tools = {tool.__name__: tool for tool in captured["tools"]}
            tools["record_decision"](
                "HOLD",
                sequence.as_posix(),
                "baseline",
                ["insufficient contrastive evidence"],
                ["baseline behavior"],
            )
            return {"messages": [SimpleNamespace(content="complete")]}

    def fake_create_agent(**kwargs: Any) -> FakeAgent:
        captured.update(kwargs)
        return FakeAgent()

    monkeypatch.setattr("langchain.agents.create_agent", fake_create_agent)

    result = run_hl_agent(
        model="openrouter:test/model",
        campaign="HLT",
        families=_FAMILIES,
        episode_number=2,
        episode_families=("beta", "beta", "alpha"),
        project_root=project_root,
        source_sequence=sequence,
    )

    assert result["response"] == "complete"
    assert result["decision"]["decision"] == "HOLD"
    assert captured["system_prompt"] == "You are the independent HL agent for HLT."
    tool_names = {tool.__name__ for tool in captured["tools"]}
    assert tool_names >= {
        "inspect_campaign",
        "record_decision",
        "write_staged_file",
        "publish_update",
    }
    assert "run_matched_sequences" not in tool_names
    direct_prompt = captured["payload"]["messages"][0]["content"]
    assert "Completed episode: 2" in direct_prompt
    assert "Episode families by iteration: beta, beta, alpha" in direct_prompt
    assert "episode count" in direct_prompt
    assert "Resolved K" not in direct_prompt
    recovery = captured["middleware"][0]

    def invalid_tool(_: Any) -> Any:
        raise ValueError("path is outside the HL read boundary")

    tool_error = recovery.wrap_tool_call(
        SimpleNamespace(tool_call={"id": "call-1"}),
        invalid_tool,
    )
    assert tool_error.status == "error"
    assert tool_error.tool_call_id == "call-1"


def test_hl_agent_cli_requires_exactly_four_families() -> None:
    result = CliRunner().invoke(
        app,
        [
            "hl-agent",
            "--campaign",
            "HLT",
            "--family",
            "alpha",
            "--family",
            "beta",
            "--family",
            "gamma",
        ],
    )

    assert result.exit_code == 2
    assert "exactly four" in result.output


def test_infrastructure_runs_new_episodes_from_middle_and_analyzes_final(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[int, tuple[str, ...], Path]] = []
    seeds = iter((101, 102))
    selected_families = iter(
        (("delta", "delta", "alpha"), ("beta", "alpha", "beta"))
    )

    def fake_sequence(**kwargs: Any) -> tuple[Path, tuple[str, ...]]:
        sequence = tmp_path / "data" / "sequences" / f"sequence-{kwargs['seed']}"
        sequence.mkdir(parents=True)
        return sequence, next(selected_families)

    def fake_agent(**kwargs: Any) -> dict[str, Any]:
        calls.append(
            (
                kwargs["episode_number"],
                kwargs["episode_families"],
                kwargs["source_sequence"],
            )
        )
        return {
            "response": f"episode {kwargs['episode_number']} analyzed",
            "decision": {"decision": "HOLD"},
            "published_update": None,
        }

    monkeypatch.setattr(hl_module.secrets, "randbits", lambda _: next(seeds))
    monkeypatch.setattr(hl_module, "_run_sequence_episode", fake_sequence)
    monkeypatch.setattr(hl_module, "run_hl_agent", fake_agent)

    records = hl_module.run_hl_campaign(
        model="openrouter:test/model",
        campaign="HLT",
        families=_FAMILIES,
        episodes=2,
        start_episode=3,
        k=3,
        config_dir=tmp_path / "config",
        project_root=tmp_path,
        source_sequence=None,
    )

    assert [(call[0], call[1]) for call in calls] == [
        (3, ("delta", "delta", "alpha")),
        (4, ("beta", "alpha", "beta")),
    ]
    assert [record["seed"] for record in records] == [101, 102]
    assert [record["families"] for record in records] == [
        ["delta", "delta", "alpha"],
        ["beta", "alpha", "beta"],
    ]
    assert all(record["status"] == "complete" for record in records)
    assert (
        tmp_path
        / "data"
        / "experiments"
        / "HLT"
        / "episodes"
        / "episode_0004.json"
    ).is_file()


def test_source_analysis_is_excluded_from_new_episode_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, source = _project(tmp_path)
    calls: list[tuple[int, tuple[str, ...], Path]] = []

    def fake_sequence(**kwargs: Any) -> tuple[Path, tuple[str, ...]]:
        sequence = tmp_path / "data" / "sequences" / "sequence-new"
        sequence.mkdir(parents=True)
        return sequence, ("gamma", "gamma", "delta")

    def fake_agent(**kwargs: Any) -> dict[str, Any]:
        calls.append(
            (
                kwargs["episode_number"],
                kwargs["episode_families"],
                kwargs["source_sequence"],
            )
        )
        return {
            "response": "analyzed",
            "decision": {"decision": "HOLD"},
            "published_update": None,
        }

    monkeypatch.setattr(hl_module.secrets, "randbits", lambda _: 55)
    monkeypatch.setattr(hl_module, "_run_sequence_episode", fake_sequence)
    monkeypatch.setattr(hl_module, "run_hl_agent", fake_agent)

    records = hl_module.run_hl_campaign(
        model="openrouter:test/model",
        campaign="HLT",
        families=_FAMILIES,
        episodes=1,
        start_episode=2,
        k=3,
        config_dir=tmp_path / "config",
        project_root=tmp_path,
        source_sequence=source,
    )

    assert [(call[0], call[1]) for call in calls] == [
        (1, ("alpha",)),
        (2, ("gamma", "gamma", "delta")),
    ]
    assert len(records) == 1
    assert records[0]["episode"] == 2


def test_start_episode_is_inferred_from_managed_records(tmp_path: Path) -> None:
    campaign_root = tmp_path / "data" / "experiments" / "HLT"
    record = campaign_root / "episodes" / "episode_0007.json"
    record.parent.mkdir(parents=True)
    record.write_text("{}")

    assert (
        hl_module._resolve_start_episode(
            campaign_root=campaign_root,
            requested=None,
            has_source=False,
        )
        == 8
    )


def test_failed_episode_is_retried(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    campaign_root = tmp_path / "data" / "experiments" / "HLT"
    record_path = campaign_root / "episodes" / "episode_0007.json"
    record_path.parent.mkdir(parents=True)
    record_path.write_text(
        json.dumps(
            {
                "campaign": "HLT",
                "episode": 7,
                "status": "failed",
                "error": "ImportError: old overlay",
            }
        )
    )
    sequence = tmp_path / "data" / "sequences" / "sequence-retry"

    monkeypatch.setattr(hl_module.secrets, "randbits", lambda _: 77)
    monkeypatch.setattr(
        hl_module,
        "_run_sequence_episode",
        lambda **kwargs: (sequence, ("beta",)),
    )
    monkeypatch.setattr(
        hl_module,
        "run_hl_agent",
        lambda **kwargs: {
            "response": "retried",
            "decision": {"decision": "HOLD"},
            "published_update": None,
        },
    )

    assert (
        hl_module._resolve_start_episode(
            campaign_root=campaign_root,
            requested=None,
            has_source=False,
        )
        == 7
    )
    hl_module.run_hl_campaign(
        model="openrouter:test/model",
        campaign="HLT",
        families=_FAMILIES,
        episodes=1,
        start_episode=7,
        k=1,
        config_dir=tmp_path / "config",
        project_root=tmp_path,
        source_sequence=None,
    )

    record = json.loads(record_path.read_text())
    assert record["status"] == "complete"


def test_episode_runner_delegates_family_pool_to_sequence_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[str] = []

    def fake_run(command: list[str], **kwargs: Any) -> SimpleNamespace:
        del kwargs
        captured.extend(command)
        sequence = tmp_path / "data" / "sequences" / "sequence-1"
        for iteration, family in enumerate(("delta", "alpha", "delta"), start=1):
            spec = sequence / f"iter-{iteration}" / "sequence_spec.json"
            spec.parent.mkdir(parents=True)
            spec.write_text(
                json.dumps({"tasks": [{"task_id": f"{family}/task-1"}]})
            )
            result = (
                sequence
                / f"iter-{iteration}"
                / "samples"
                / "full_orchestration"
                / "sample_result.json"
            )
            result.parent.mkdir(parents=True)
            result.write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(hl_module.subprocess, "run", fake_run)

    sequence, selected = hl_module._run_sequence_episode(
        families=_FAMILIES,
        seed=77,
        k=3,
        harness_ref="promoted:HLT",
        config_dir=tmp_path / "config",
        project_root=tmp_path,
    )

    assert sequence.name == "sequence-1"
    assert [
        captured[index + 1]
        for index, argument in enumerate(captured)
        if argument == "--family"
    ] == list(_FAMILIES)
    assert selected == ("delta", "alpha", "delta")
    assert captured[captured.index("-K") + 1] == "3"
    assert "--harness-ref" in captured
