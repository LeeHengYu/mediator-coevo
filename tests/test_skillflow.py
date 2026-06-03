from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from mediated_coevo.benchmarks import (
    HarborPrebuiltImageMissingError,
    HarborRunResult,
    HarborRunner,
    SKILLFLOW_VERIFIER_TYPE,
    SkillFlowRepository,
    parse_skillflow_execution_trace,
)
from mediated_coevo.main import app


def test_repository_resolves_tasks_family_and_task_set(tmp_path: Path) -> None:
    root = tmp_path / "skillflow"
    _write_task(root / "tasks" / "family-a" / "task-one", family="family-a")
    _write_task(root / "tasks" / "family-b" / "task-two", family="family-b")
    task_set_dir = root / "task_sets"
    task_set_dir.mkdir()
    (task_set_dir / "smoke.txt").write_text("family-a/task-one\n")
    repo = SkillFlowRepository(root_dir=root, task_dirs=["tasks"])

    task = repo.resolve("family-a/task-one")

    assert task.task_id == "family-a/task-one"
    assert task.family == "family-a"
    assert repo.list_local_task_ids(family="family-a") == ["family-a/task-one"]
    assert repo.resolve_selection(tasks=[], family=None, task_set="smoke") == [
        "family-a/task-one"
    ]


def test_prepare_run_workspace_injects_executor_envelope(tmp_path: Path) -> None:
    root = tmp_path / "skillflow"
    _write_task(root / "tasks" / "demo", family="demo")
    repo = SkillFlowRepository(root_dir=root, task_dirs=["tasks"])
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
        _write_task(local_dir / "test_tasks" / "family-a" / "task-one", family="family-a")
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
    assert (
        root / "tasks" / "family-a" / "ALL_TASK_DIFFICULTY_RANKING.json"
    ).is_file()
    assert "test_tasks/family-a/task-one/**" in calls[0]
    assert "test_tasks/family-a/ALL_TASK_DIFFICULTY_RANKING.json" in calls[0]
    assert not (root / "tasks" / "test_tasks").exists()


def test_sync_cli_accepts_selected_tasks(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_run(command, **kwargs):
        del kwargs
        calls.append(command)
        local_dir = Path(command[command.index("--local-dir") + 1])
        _write_task(local_dir / "test_tasks" / "family-a" / "task-one", family="family-a")
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
    assert "Downloaded SkillFlow tasks to:" in result.output
    assert "demo/dataset" in calls[0]
    assert "test_tasks/family-a/task-one/**" in calls[0]
    assert (tmp_path / "tasks" / "family-a" / "task-one" / "task.toml").is_file()


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
    assert "SkillFlow base image build complete" in result.output


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
    assert "Would build base image" in result.output
    output_compact = result.output.replace("\n", "")
    assert "build.sh" in output_compact
    assert "skillflow/harbor-cli-base:test" in result.output
    assert "SkillFlow base image dry run complete" in result.output
    assert "SkillFlow base image build complete" not in result.output


def test_list_cli_queries_remote_tasks(monkeypatch) -> None:
    def fake_run(command, **kwargs):
        del kwargs
        assert command == [
            "hf",
            "datasets",
            "ls",
            "demo/dataset",
            "-R",
            "--format",
            "quiet",
        ]
        return _Completed(
            returncode=0,
            stdout="test_tasks/family-a/task-one/task.toml\n",
            stderr="",
        )

    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.subprocess.run",
        fake_run,
    )

    result = CliRunner().invoke(
        app,
        ["list", "--dataset", "demo/dataset"],
    )

    assert result.exit_code == 0
    assert result.output.strip() == "family-a/task-one"


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
                        "verifier": {
                            "metrics": [{"name": "reward", "mean": 0.75}]
                        }
                    }
                },
            }
        )
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "trial-1",
                "agent_result": {"n_input_tokens": 7, "n_output_tokens": 3},
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
    assert trace.token_usage.input_tokens == 7
    assert trace.token_usage.output_tokens == 3


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
                        "verifier": {
                            "metrics": [{"name": "reward", "mean": 0.0}]
                        }
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


@pytest.mark.asyncio
async def test_harbor_runner_raises_for_missing_prebuilt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
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

    runner = HarborRunner(jobs_dir=jobs_dir)

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
            "hermes",
            "-m",
            "provider/model",
            "-o",
            str(jobs_dir),
            "--yes",
        ]
    ]
    assert "OPENAI_API_KEY" not in captured_envs[0]


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
