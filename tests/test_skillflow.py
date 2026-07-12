from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

import mediated_coevo.benchmarks.skillflow as skillflow_benchmark
import mediated_coevo.cli.experiment as experiment_cli
from mediated_coevo.benchmarks import (
    HarborPrebuiltImageMissingError,
    HarborRunResult,
    HarborRunner,
    SKILLFLOW_VERIFIER_TYPE,
    SkillFlowRepository,
    SkillFlowSyncConfig,
    parse_skillflow_execution_trace,
)
from mediated_coevo.cli.experiment import (
    BOOTSTRAP_FAMILY_TASK_COUNT,
    load_task_manifest_selection,
    resolve_task_selection,
)
from mediated_coevo.cli import skillflow as skillflow_cli
from mediated_coevo.main import app


def test_repository_resolves_tasks_and_lists_family(tmp_path: Path) -> None:
    root = tmp_path / "skillflow"
    _write_task(root / "tasks" / "family-a" / "task-one", family="family-a")
    _write_task(root / "tasks" / "family-b" / "task-two", family="family-b")
    repo = SkillFlowRepository(root_dir=root, task_dirs=["tasks"])

    task = repo.resolve("family-a/task-one")

    assert task.task_id == "family-a/task-one"
    assert task.family == "family-a"
    assert repo.list_local_task_ids(family="family-a") == ["family-a/task-one"]


def test_family_selection_evenly_repeats_short_task_pool(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "skillflow"
    _write_task(root / "tasks" / "family-a" / "task-one", family="family-a")
    _write_task(root / "tasks" / "family-a" / "task-two", family="family-a")
    repo = SkillFlowRepository(root_dir=root, task_dirs=["tasks"])
    monkeypatch.setattr(experiment_cli.secrets, "randbits", lambda _bits: 7)

    selection = resolve_task_selection(
        repository=repo,
        family="family-a",
        seed=42,
    )

    assert len(selection.task_ids) == BOOTSTRAP_FAMILY_TASK_COUNT
    assert Counter(selection.task_ids) == {
        "family-a/task-one": 4,
        "family-a/task-two": 4,
    }
    assert selection.task_stream_seed == 7


def test_task_manifest_preserves_order_and_duplicates(tmp_path: Path) -> None:
    root = tmp_path / "skillflow"
    _write_task(root / "tasks" / "family-a" / "task-one", family="family-a")
    _write_task(root / "tasks" / "family-b" / "task-two", family="family-b")
    repo = SkillFlowRepository(root_dir=root, task_dirs=["tasks"])
    manifest = tmp_path / "stream.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "split": "test",
                "families": ["family-a", "family-b"],
                "task_stream_seed": 123,
                "task_ids": [
                    "family-b/task-two",
                    "family-a/task-one",
                    "family-b/task-two",
                ],
            }
        )
    )

    selection = load_task_manifest_selection(
        repository=repo,
        manifest_path=manifest,
    )

    assert selection.task_ids == [
        "family-b/task-two",
        "family-a/task-one",
        "family-b/task-two",
    ]
    assert selection.families == ("family-a", "family-b")
    assert selection.split == "test"
    assert selection.task_stream_seed == 123


def test_family_selection_spreads_extra_slots_across_short_task_pool(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "skillflow"
    for index in range(1, 7):
        _write_task(root / "tasks" / "family-a" / f"task-{index}", family="family-a")
    repo = SkillFlowRepository(root_dir=root, task_dirs=["tasks"])
    monkeypatch.setattr(experiment_cli.secrets, "randbits", lambda _bits: 7)

    selection = resolve_task_selection(
        repository=repo,
        family="family-a",
        seed=42,
    )

    counts = Counter(selection.task_ids)
    assert set(counts) == set(repo.list_local_task_ids(family="family-a"))
    assert sorted(counts.values()) == [1, 1, 1, 1, 2, 2]


def test_family_selection_uses_no_replacement_when_pool_is_large_enough(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "skillflow"
    for index in range(1, 9):
        _write_task(root / "tasks" / "family-a" / f"task-{index}", family="family-a")
    repo = SkillFlowRepository(root_dir=root, task_dirs=["tasks"])
    monkeypatch.setattr(experiment_cli.secrets, "randbits", lambda _bits: 7)

    selection = resolve_task_selection(
        repository=repo,
        family="family-a",
        seed=42,
    )

    assert len(selection.task_ids) == BOOTSTRAP_FAMILY_TASK_COUNT
    assert len(set(selection.task_ids)) == BOOTSTRAP_FAMILY_TASK_COUNT


def test_family_selection_randomizes_stream_without_changing_split(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "skillflow"
    for index in range(1, 7):
        _write_task(root / "tasks" / "family-a" / f"task-{index}", family="family-a")
        _write_task(root / "tasks" / "family-b" / f"task-{index}", family="family-b")
    repo = SkillFlowRepository(root_dir=root, task_dirs=["tasks"])
    generated_seeds = iter((100, 101, 102))
    monkeypatch.setattr(
        experiment_cli.secrets,
        "randbits",
        lambda _bits: next(generated_seeds),
    )

    first = resolve_task_selection(
        repository=repo,
        family=["family-a", "family-b"],
        seed=42,
        split="train",
    )
    second = resolve_task_selection(
        repository=repo,
        family=["family-a", "family-b"],
        seed=42,
        split="train",
    )
    validation = resolve_task_selection(
        repository=repo,
        family=["family-a", "family-b"],
        seed=42,
        split="validation",
    )

    assert first.task_stream_seed == 100
    assert second.task_stream_seed == 101
    assert validation.task_stream_seed == 102
    assert first.task_ids != second.task_ids
    assert set(first.task_ids + second.task_ids).isdisjoint(validation.task_ids)


def test_family_selection_accepts_multiple_families_and_split(tmp_path: Path) -> None:
    root = tmp_path / "skillflow"
    for index in range(1, 5):
        _write_task(root / "tasks" / "family-a" / f"task-{index}", family="family-a")
        _write_task(root / "tasks" / "family-b" / f"task-{index}", family="family-b")
    repo = SkillFlowRepository(root_dir=root, task_dirs=["tasks"])

    validation = resolve_task_selection(
        repository=repo,
        family=["family-a", "family-b"],
        seed=42,
        split="validation",
    )
    all_tasks = set(repo.list_local_task_ids(family=None))
    train = resolve_task_selection(
        repository=repo,
        family=["family-a", "family-b"],
        seed=42,
        split="train",
    )

    assert validation.families == ("family-a", "family-b")
    assert validation.family == "family-a,family-b"
    assert validation.split == "validation"
    assert len(validation.task_ids) == BOOTSTRAP_FAMILY_TASK_COUNT
    assert set(validation.task_ids) <= all_tasks
    assert set(validation.task_ids).isdisjoint(set(train.task_ids))


def test_prepare_run_workspace_injects_executor_envelope(tmp_path: Path) -> None:
    root = tmp_path / "skillflow"
    task_dir = root / "tasks" / "demo"
    _write_task(task_dir, family="demo")
    environment_dir = task_dir / "environment"
    environment_dir.mkdir()
    (environment_dir / "Dockerfile").write_text(
        f"FROM {skillflow_benchmark.LEGACY_HARBOR_BASE_IMAGE}\n",
    )
    repo = SkillFlowRepository(
        root_dir=root,
        task_dirs=["tasks"],
        harbor_base_image="local/harbor-base:test",
    )
    task = repo.resolve("demo")

    run_dir = repo.prepare_run_workspace(
        task=task,
        destination_root=tmp_path / "runs",
        planner_instruction="Do the planned work.",
        injected_skill_text="# Executor policy\n",
        injected_skill_name="executor",
    )

    instruction = (run_dir / "instruction.md").read_text()
    metadata = repo.executor_envelope_metadata(
        run_dir=run_dir,
        executor_policy="# Executor policy\n",
    )
    assert "# Task Instruction" in instruction
    assert "Do the planned work." in instruction
    assert "# Executor Policy" in instruction
    assert metadata["executor_policy_injected"] == "true"
    assert metadata["verifier_contract_kind"] == SKILLFLOW_VERIFIER_TYPE
    assert (run_dir / "environment" / "Dockerfile").read_text() == (
        "FROM local/harbor-base:test\n"
    )


def test_repository_lists_remote_task_ids(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "skillflow"
    repo = SkillFlowRepository(root_dir=root, task_dirs=["tasks"])
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        return _Completed(
            returncode=0,
            stdout="\n".join(
                [
                    "test_tasks/family-a/task-one/task.toml",
                    "test_tasks/family-a/task-one/instruction.md",
                    "test_tasks/family-b/task-two/task.toml",
                    "README.md",
                ]
            ),
            stderr="",
        )

    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.subprocess.run",
        fake_run,
    )

    assert repo.list_remote_task_ids() == [
        "family-a/task-one",
        "family-b/task-two",
    ]
    assert repo.list_remote_task_ids(family="family-a") == ["family-a/task-one"]
    assert calls[0] == [
        "hf",
        "datasets",
        "ls",
        "zhang-ziao/SkillFlow-Task",
        "-R",
        "--format",
        "quiet",
    ]


def test_repository_lists_cached_remote_task_ids_before_hf(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "skillflow"
    task_cache_path = tmp_path / "skillflow_tasks.txt"
    task_cache_path.write_text(
        "\n".join(
            [
                "# cached SkillFlow tasks",
                "family-b/task-two",
                "family-a/task-one",
                "family-a/task-one",
                "invalid-without-family",
                "",
            ]
        )
    )
    repo = SkillFlowRepository(
        root_dir=root,
        task_dirs=["tasks"],
        sync=SkillFlowSyncConfig(remote_task_cache_path=task_cache_path),
    )

    def fake_run(command, **kwargs):
        del command, kwargs
        raise AssertionError("cached task listing should not call hf")

    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.subprocess.run",
        fake_run,
    )

    assert repo.list_remote_task_ids() == [
        "family-a/task-one",
        "family-b/task-two",
    ]
    assert repo.list_remote_task_ids(family="family-a") == ["family-a/task-one"]


def test_sync_tasks_downloads_all_test_tasks_to_configured_task_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "skillflow"
    repo = SkillFlowRepository(root_dir=root, task_dirs=["tasks"])
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        local_dir = Path(command[command.index("--local-dir") + 1])
        _write_task(
            local_dir / "test_tasks" / "family-a" / "task-one", family="family-a"
        )
        return _Completed(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.subprocess.run",
        fake_run,
    )

    destination = repo.sync_tasks()

    assert destination == root / "tasks"
    assert (root / "tasks" / "family-a" / "task-one" / "task.toml").is_file()
    assert "test_tasks/**" in calls[0]
    assert calls[0][-2] == "--local-dir"
    assert Path(calls[0][-1]) != root / "tasks"
    assert not (root / "tasks" / "test_tasks").exists()
    assert repo.resolve("family-a/task-one").family == "family-a"


def test_sync_tasks_downloads_selected_tasks_and_family_ranking(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "skillflow"
    repo = SkillFlowRepository(root_dir=root, task_dirs=["tasks"])
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        local_dir = Path(command[command.index("--local-dir") + 1])
        family_dir = local_dir / "test_tasks" / "family-a"
        _write_task(family_dir / "task-one", family="family-a")
        (family_dir / "ALL_TASK_DIFFICULTY_RANKING.json").write_text(
            json.dumps(["task-one", "task-two"])
        )
        return _Completed(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.subprocess.run",
        fake_run,
    )

    destination = repo.sync_tasks(task_ids=["family-a/task-one"])

    assert destination == root / "tasks"
    assert (root / "tasks" / "family-a" / "task-one" / "task.toml").is_file()
    assert (root / "tasks" / "family-a" / "ALL_TASK_DIFFICULTY_RANKING.json").is_file()
    assert "test_tasks/family-a/task-one/**" in calls[0]
    assert "test_tasks/family-a/ALL_TASK_DIFFICULTY_RANKING.json" in calls[0]
    assert not (root / "tasks" / "test_tasks").exists()


def test_sync_cli_accepts_selected_tasks(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        local_dir = Path(command[command.index("--local-dir") + 1])
        _write_task(
            local_dir / "test_tasks" / "family-a" / "task-one", family="family-a"
        )
        return _Completed(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.subprocess.run",
        fake_run,
    )

    result = CliRunner().invoke(
        app,
        [
            "sync",
            "--output-dir",
            str(tmp_path / "tasks"),
            "--dataset",
            "demo/dataset",
            "--tasks",
            "family-a/task-one",
        ],
    )

    assert result.exit_code == 0
    assert "demo/dataset" in calls[0]
    assert "test_tasks/family-a/task-one/**" in calls[0]
    assert (tmp_path / "tasks" / "family-a" / "task-one" / "task.toml").is_file()


def test_sync_tasks_normalizes_legacy_dockerfiles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo = SkillFlowRepository(
        root_dir=tmp_path / "skillflow",
        task_dirs=["tasks"],
        sync=SkillFlowSyncConfig(dataset="demo/dataset"),
        harbor_base_image="local/harbor-base:test",
    )

    def fake_run(command, **kwargs):
        del kwargs
        local_dir = Path(command[command.index("--local-dir") + 1])
        task_dir = local_dir / "test_tasks" / "family-a" / "task-one"
        _write_task(task_dir, family="family-a")
        environment_dir = task_dir / "environment"
        environment_dir.mkdir()
        (environment_dir / "Dockerfile").write_text(
            f"FROM {skillflow_benchmark.LEGACY_HARBOR_BASE_IMAGE}\n",
        )
        return _Completed(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.subprocess.run",
        fake_run,
    )

    destination = repo.sync_tasks(
        destination=tmp_path / "tasks",
        task_ids=["family-a/task-one"],
    )

    dockerfile = destination / "family-a" / "task-one" / "environment" / "Dockerfile"
    assert dockerfile.read_text() == "FROM local/harbor-base:test\n"


def test_sync_cli_accepts_family_selector(monkeypatch, tmp_path: Path) -> None:
    task_cache_path = tmp_path / "skillflow_tasks.txt"
    task_cache_path.write_text(
        "\n".join(
            [
                "family-a/task-one",
                "family-a/task-two",
                "family-b/task-three",
            ]
        )
    )
    repo = SkillFlowRepository(
        root_dir=tmp_path / "skillflow",
        task_dirs=["tasks"],
        sync=SkillFlowSyncConfig(
            dataset="demo/dataset",
            remote_task_cache_path=task_cache_path,
        ),
    )
    calls = []

    def fake_build_benchmark_repo(project_root, config):
        del project_root
        assert config.executor_runtime.dataset == "demo/dataset"
        return repo

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        local_dir = Path(command[command.index("--local-dir") + 1])
        _write_task(
            local_dir / "test_tasks" / "family-a" / "task-one", family="family-a"
        )
        _write_task(
            local_dir / "test_tasks" / "family-a" / "task-two", family="family-a"
        )
        return _Completed(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        skillflow_cli,
        "build_benchmark_repo",
        fake_build_benchmark_repo,
    )
    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.subprocess.run",
        fake_run,
    )

    result = CliRunner().invoke(
        app,
        [
            "sync",
            "--output-dir",
            str(tmp_path / "tasks"),
            "--dataset",
            "demo/dataset",
            "--family",
            "family-a",
        ],
    )

    assert result.exit_code == 0
    assert "test_tasks/family-a/task-one/**" in calls[0]
    assert "test_tasks/family-a/task-two/**" in calls[0]
    assert "test_tasks/family-b/task-three/**" not in calls[0]
    assert (tmp_path / "tasks" / "family-a" / "task-one" / "task.toml").is_file()
    assert (tmp_path / "tasks" / "family-a" / "task-two" / "task.toml").is_file()


def test_build_base_image_cli_runs_skillflow_build_script(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[list[str], dict[str, object]]] = []

    def fake_run(command, **kwargs):
        calls.append((list(command), dict(kwargs)))
        return _Completed(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mediated_coevo.cli.skillflow.subprocess.run", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "build-base-image",
            "--base-image-tag",
            "skillflow/harbor-cli-base:test",
        ],
    )

    assert result.exit_code == 0
    assert calls[0][0] == [
        "bash",
        str(Path.cwd() / "docker" / "harbor-cli-base" / "build.sh"),
        "skillflow/harbor-cli-base:test",
    ]
    assert calls[0][1]["cwd"] == Path.cwd()
    assert calls[0][1]["check"] is False


def test_build_base_image_cli_dry_run_does_not_run_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(command, **kwargs):
        del command, kwargs
        raise AssertionError("dry-run should not execute subprocess.run")

    monkeypatch.setattr("mediated_coevo.cli.skillflow.subprocess.run", fake_run)

    result = CliRunner().invoke(
        app,
        [
            "build-base-image",
            "--base-image-tag",
            "skillflow/harbor-cli-base:test",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0


def test_list_cli_uses_cached_remote_tasks(monkeypatch, tmp_path: Path) -> None:
    task_cache_path = tmp_path / "skillflow_tasks.txt"
    task_cache_path.write_text(
        "\n".join(
            [
                "family-a/task-one",
                "family-a/task-two",
                "family-b/task-three",
            ]
        )
    )
    repo = SkillFlowRepository(
        root_dir=tmp_path / "skillflow",
        task_dirs=["tasks"],
        sync=SkillFlowSyncConfig(
            dataset="demo/dataset",
            remote_task_cache_path=task_cache_path,
        ),
    )

    def fake_build_benchmark_repo(project_root, config):
        del project_root
        assert config.executor_runtime.dataset == "demo/dataset"
        return repo

    def fake_run(command, **kwargs):
        del command, kwargs
        raise AssertionError("cached list should not call hf")

    monkeypatch.setattr(
        skillflow_cli,
        "build_benchmark_repo",
        fake_build_benchmark_repo,
    )
    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.subprocess.run",
        fake_run,
    )

    result = CliRunner().invoke(
        app,
        ["list", "--dataset", "demo/dataset", "--family", "family-a"],
    )

    assert result.exit_code == 0
    assert result.output.splitlines() == ["family-a/task-one", "family-a/task-two"]


def test_trace_parser_reads_harbor_stats_reward(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trials" / "trial-1"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "job-1",
                "stats": {
                    "evals": {
                        "verifier": {"metrics": [{"name": "reward", "mean": 0.75}]}
                    }
                },
            }
        )
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "trial-1",
                "agent_result": {
                    "n_input_tokens": 7,
                    "n_output_tokens": 3,
                    "cost_usd": 0.123,
                },
            }
        )
    )
    run_result = HarborRunResult(
        job_dir=job_dir,
        trial_dir=trial_dir,
        returncode=0,
        stdout="",
        stderr="",
    )

    trace = parse_skillflow_execution_trace(
        run_result=run_result,
        task_id="demo",
        iteration=2,
        duration_sec=1.5,
    )

    assert trace.status == "ok"
    assert trace.reward == 0.75
    assert trace.run_id == "job-1"
    assert trace.harbor_trial_id == "trial-1"
    assert trace.harbor_metadata["reward_source"] == "job_stats"
    assert trace.harbor_metadata["agent_result.cost_usd"] == "0.123"
    assert (
        trace.harbor_metadata["executor_reported_cost_source"] == "harbor_agent_result"
    )
    assert trace.token_usage.input_tokens == 7
    assert trace.token_usage.output_tokens == 3


def test_trace_parser_reports_source_that_supplied_parsed_reward(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trials" / "trial-1"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "job-1",
                "stats": {
                    "evals": {
                        "verifier": {"metrics": [{"name": "reward", "mean": "n/a"}]}
                    }
                },
            }
        )
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "trial-1",
                "verifier_result": {"reward": 0.42},
            }
        )
    )
    run_result = HarborRunResult(
        job_dir=job_dir,
        trial_dir=trial_dir,
        returncode=0,
        stdout="",
        stderr="",
    )

    trace = parse_skillflow_execution_trace(
        run_result=run_result,
        task_id="demo",
        iteration=0,
        duration_sec=0.1,
    )

    assert trace.status == "ok"
    assert trace.reward == 0.42
    assert trace.harbor_metadata["reward_source"] == "trial_verifier_result"


def test_trace_parser_falls_back_to_hermes_session_tokens(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trials" / "trial-1"
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "job-1",
                "stats": {
                    "evals": {
                        "verifier": {"metrics": [{"name": "reward", "mean": 0.5}]}
                    }
                },
            }
        )
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "trial-1",
                "agent_result": {"n_input_tokens": 0, "n_output_tokens": 0},
            }
        )
    )
    (agent_dir / "hermes-session.jsonl").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "input_tokens": 11,
                        "output_tokens": 5,
                        "cache_read_tokens": 101,
                    }
                ),
                json.dumps(
                    {
                        "input_tokens": 7,
                        "output_tokens": 3,
                        "cache_read_tokens": 202,
                    }
                ),
            ]
        )
        + "\n"
    )
    run_result = HarborRunResult(
        job_dir=job_dir,
        trial_dir=trial_dir,
        returncode=0,
        stdout="",
        stderr="",
    )

    trace = parse_skillflow_execution_trace(
        run_result=run_result,
        task_id="demo",
        iteration=2,
        duration_sec=1.5,
    )

    assert trace.status == "ok"
    assert trace.reward == 0.5
    assert trace.token_usage.input_tokens == 18
    assert trace.token_usage.output_tokens == 8
    assert trace.harbor_metadata["executor_token_source"] == "hermes_session"
    assert trace.harbor_metadata["executor_session_cache_read_tokens"] == "303"


def test_trace_parser_treats_harbor_trial_exception_as_env_failure(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trials" / "trial-1"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "job-1",
                "stats": {
                    "evals": {
                        "verifier": {"metrics": [{"name": "reward", "mean": 0.0}]}
                    }
                },
            }
        )
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "trial-1",
                "agent_result": None,
                "verifier_result": None,
                "exception_info": {
                    "exception_type": "RuntimeError",
                    "exception_message": (
                        "Docker compose command failed. "
                        "Image harbor-prebuilt:task-demo Error pull access denied"
                    ),
                    "exception_traceback": "RuntimeError: Docker compose command failed",
                },
            }
        )
    )
    run_result = HarborRunResult(
        job_dir=job_dir,
        trial_dir=trial_dir,
        returncode=0,
        stdout="",
        stderr="",
    )

    trace = parse_skillflow_execution_trace(
        run_result=run_result,
        task_id="demo",
        iteration=0,
        duration_sec=0.1,
    )

    assert trace.status == "env_failure"
    assert trace.reward is None
    assert trace.error_kind == "missing_prebuilt_image"
    assert trace.error_detail["exception_type"] == "RuntimeError"
    assert trace.run_id == "job-1"
    assert trace.harbor_trial_id == "trial-1"


def test_trace_parser_reads_reward_file(tmp_path: Path) -> None:
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trials" / "trial-1"
    reward_dir = trial_dir / "verifier"
    reward_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(json.dumps({"id": "job-1"}))
    (trial_dir / "result.json").write_text(json.dumps({"id": "trial-1"}))
    (reward_dir / "reward.txt").write_text("1\n")
    run_result = HarborRunResult(
        job_dir=job_dir,
        trial_dir=trial_dir,
        returncode=0,
        stdout="",
        stderr="",
    )

    trace = parse_skillflow_execution_trace(
        run_result=run_result,
        task_id="demo",
        iteration=0,
        duration_sec=0.1,
    )

    assert trace.status == "ok"
    assert trace.reward == 1.0
    assert trace.harbor_metadata["reward_source"] == "verifier_reward_file"


def test_trace_parser_reads_observed_trial_verifier_rewards_shape(
    tmp_path: Path,
) -> None:
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "trial"
    trial_dir.mkdir(parents=True)
    (job_dir / "result.json").write_text(json.dumps({"id": "job-1"}))
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "trial-1",
                "agent_info": {
                    "name": "nop",
                    "model_info": {
                        "name": "gemini-3-flash-preview",
                        "provider": "google",
                    },
                },
                "agent_result": {
                    "n_input_tokens": None,
                    "n_output_tokens": None,
                },
                "verifier_result": {"rewards": {"reward": 0.25}},
            }
        )
    )
    run_result = HarborRunResult(
        job_dir=job_dir,
        trial_dir=trial_dir,
        returncode=0,
        stdout="",
        stderr="",
    )

    trace = parse_skillflow_execution_trace(
        run_result=run_result,
        task_id="demo",
        iteration=0,
        duration_sec=0.1,
    )

    assert trace.status == "ok"
    assert trace.reward == 0.25
    assert trace.harbor_metadata["reward_source"] == "trial_verifier_rewards"
    assert trace.harbor_metadata["agent_info.name"] == "nop"
    assert trace.harbor_metadata["agent_info.model_provider"] == "google"


def test_declared_prebuilt_image_is_rebuilt_when_harbor_cleanup_removed_it(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    task_dir = tmp_path / "task"
    environment_dir = task_dir / "environment"
    environment_dir.mkdir(parents=True)
    image_name = "harbor-prebuilt:task-demo"
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                "[environment]",
                f'docker_image = "{image_name}"',
            ]
        )
    )
    (environment_dir / "Dockerfile").write_text(
        f"FROM {skillflow_benchmark.LEGACY_HARBOR_BASE_IMAGE}\n",
    )
    build_script = tmp_path / "build-base.sh"
    build_script.write_text("#!/usr/bin/env bash\n")
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        del kwargs
        command = list(command)
        calls.append(command)
        if command == ["docker", "image", "inspect", image_name]:
            return _Completed(returncode=1, stdout="", stderr="missing")
        if command == [
            "docker",
            "image",
            "inspect",
            skillflow_benchmark.SKILLFLOW_HARBOR_BASE_IMAGE,
        ]:
            return _Completed(returncode=1, stdout="", stderr="missing")
        return _Completed(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("mediated_coevo.benchmarks.skillflow.subprocess.run", fake_run)

    skillflow_benchmark._ensure_declared_prebuilt_image(
        task_dir,
        base_image_build_script=build_script,
    )

    assert [
        "bash",
        str(build_script),
        skillflow_benchmark.SKILLFLOW_HARBOR_BASE_IMAGE,
    ] in calls
    assert [
        "docker",
        "build",
        "--progress=plain",
        "-f",
        str(environment_dir / "Dockerfile"),
        "-t",
        image_name,
        str(environment_dir),
    ] in calls
    assert (environment_dir / "Dockerfile").read_text() == (
        f"FROM {skillflow_benchmark.SKILLFLOW_HARBOR_BASE_IMAGE}\n"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("agent_name", "agent_env", "expected_agent_flags"),
    [
        (
            "hermes",
            {},
            [],
        ),
        (
            "claude-code",
            {
                "ANTHROPIC_API_KEY": "",
                "ANTHROPIC_AUTH_TOKEN": "${OPENROUTER_API_KEY}",
                "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
            },
            [
                "--agent-env",
                "ANTHROPIC_API_KEY=",
                "--agent-env",
                "ANTHROPIC_AUTH_TOKEN=${OPENROUTER_API_KEY}",
                "--agent-env",
                "ANTHROPIC_BASE_URL=https://openrouter.ai/api",
                "--agent-env",
                "ANTHROPIC_DEFAULT_HAIKU_MODEL=provider/model",
                "--agent-env",
                "ANTHROPIC_DEFAULT_OPUS_MODEL=provider/model",
                "--agent-env",
                "ANTHROPIC_DEFAULT_SONNET_MODEL=provider/model",
                "--agent-env",
                "ANTHROPIC_MODEL=provider/model",
                "--agent-env",
                "CLAUDE_CODE_SUBAGENT_MODEL=provider/model",
            ],
        ),
    ],
)
async def test_harbor_runner_raises_for_missing_prebuilt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    agent_name: str,
    agent_env: dict[str, str],
    expected_agent_flags: list[str],
) -> None:
    jobs_dir = tmp_path / "jobs"
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    captured_commands: list[list[str]] = []
    captured_envs: list[dict[str, str]] = []

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*command, **kwargs):
        captured_commands.append(list(command))
        captured_envs.append(kwargs["env"])
        trial_dir = jobs_dir / "job-1" / "trials" / "trial-1"
        trial_dir.mkdir(parents=True)
        (trial_dir / "result.json").write_text(
            json.dumps(
                {
                    "exception_info": {
                        "exception_type": "RuntimeError",
                        "exception_message": (
                            "Image harbor-prebuilt:task-demo Error pull access denied"
                        ),
                    }
                }
            )
        )
        return _Proc()

    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.shutil.which",
        lambda name: "/usr/local/bin/harbor",
    )
    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    runner = HarborRunner(
        jobs_dir=jobs_dir,
        agent_name=agent_name,
        agent_env=agent_env,
    )

    with pytest.raises(HarborPrebuiltImageMissingError) as exc_info:
        await runner.run(task_dir, "provider/model")

    assert "harbor-prebuilt:task-demo" in str(exc_info.value)
    assert "official SkillFlow quick start requires the base image" in str(
        exc_info.value
    )
    assert "uv run medcoevo build-base-image" in str(exc_info.value)
    assert "task-image prebuild is optional" in str(exc_info.value)
    assert captured_commands == [
        [
            "/usr/local/bin/harbor",
            "run",
            "-p",
            str(task_dir),
            "-a",
            agent_name,
            "-m",
            "provider/model",
            "-o",
            str(jobs_dir),
            "--yes",
            *expected_agent_flags,
        ]
    ]
    assert "OPENAI_API_KEY" not in captured_envs[0]


@pytest.mark.asyncio
async def test_harbor_runner_retries_transient_claude_setup_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    jobs_dir = tmp_path / "jobs"
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    captured_commands: list[list[str]] = []

    class _Proc:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def fake_create_subprocess_exec(*command, **kwargs):
        captured_commands.append(list(command))
        job_dir = jobs_dir / f"job-{len(captured_commands)}"
        trial_dir = job_dir / "trials" / "trial-1"
        trial_dir.mkdir(parents=True)
        result = (
            {
                "exception_info": {
                    "exception_type": "NonZeroAgentExitCodeError",
                    "exception_message": (
                        "curl -fsSL https://claude.ai/install.sh | bash -s --\n"
                        "The socket connection was closed unexpectedly."
                    ),
                }
            }
            if len(captured_commands) == 1
            else {"reward": 1.0, "verifier_result": {"rewards": {"reward": 1.0}}}
        )
        (trial_dir / "result.json").write_text(json.dumps(result))
        return _Proc()

    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.shutil.which",
        lambda name: "/usr/local/bin/harbor",
    )
    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )

    runner = HarborRunner(jobs_dir=jobs_dir, agent_name="claude-code")

    result = await runner.run(task_dir, "provider/model")

    assert len(captured_commands) == 2
    assert result.job_dir == jobs_dir / "job-2"


def test_claude_code_openrouter_agent_env_preserves_explicit_model_aliases() -> None:
    agent_env = skillflow_benchmark.harbor_agent_env_for_model(
        agent_name="claude-code",
        agent_env={
            "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
            "ANTHROPIC_MODEL": "custom/model",
        },
        model="provider/model",
    )

    assert agent_env["ANTHROPIC_MODEL"] == "custom/model"
    assert agent_env["ANTHROPIC_DEFAULT_SONNET_MODEL"] == "custom/model"
    assert agent_env["CLAUDE_CODE_SUBAGENT_MODEL"] == "custom/model"


def _write_task(task_dir: Path, *, family: str) -> None:
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("Do a small task.")
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                'schema_version = "1.2"',
                "",
                "[task]",
                f'name = "local/{task_dir.name}"',
                "",
                "[metadata]",
                f'family = "{family}"',
                'category = "software_engineering"',
                'difficulty = "easy"',
                'tags = ["smoke", "python"]',
                "",
                "[verifier]",
                "timeout_sec = 60.0",
                "",
            ]
        )
    )


class _Completed:
    def __init__(self, *, returncode: int, stdout: str, stderr: str) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
