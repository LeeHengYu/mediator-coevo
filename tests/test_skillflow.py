from __future__ import annotations

import json
from pathlib import Path

from mediated_coevo.benchmarks import (
    HarborRunResult,
    SKILLFLOW_VERIFIER_TYPE,
    SkillFlowRepository,
    parse_skillflow_execution_trace,
)


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


def test_sync_tasks_downloads_directly_to_configured_task_cache(
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
        _write_task(local_dir / "family-a" / "task-one", family="family-a")
        return _Completed(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillflow.subprocess.run",
        fake_run,
    )

    destination = repo.sync_tasks()

    assert destination == root / "tasks"
    assert (root / "tasks" / "family-a" / "task-one" / "task.toml").is_file()
    assert calls[0][-2:] == ["--local-dir", str(root / "tasks")]
    assert repo.resolve("family-a/task-one").family == "family-a"


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
