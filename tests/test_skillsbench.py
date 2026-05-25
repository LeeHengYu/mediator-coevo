from __future__ import annotations

import asyncio
import json

from mediated_coevo.analysis.reporting import build_score_summary
from mediated_coevo.benchmarks.skillsbench import (
    HarborRunner,
    HarborRunResult,
    SkillsBenchRepository,
    parse_execution_trace,
)
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.task import executor_policy_hash, render_executor_envelope


def test_render_executor_envelope_uses_stable_sections():
    envelope = render_executor_envelope(
        task_instruction="Fix the parser.",
        executor_policy="Use a targeted test first.",
        task_resources=["Curated skill: parser"],
        verifier_contract="pytest decides success.",
    )

    assert envelope.index("# Task Instruction") < envelope.index("# Executor Policy")
    assert envelope.index("# Executor Policy") < envelope.index("# Task Resources")
    assert envelope.index("# Task Resources") < envelope.index("# Verifier Contract")
    assert "Use a targeted test first." in envelope


def test_prepare_run_workspace_wraps_executor_policy_without_creating_skill(tmp_path):
    task_dir = tmp_path / "benchmarks" / "tasks" / "sample-task"
    skill_dir = task_dir / "environment" / "skills" / "curated-parser"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("# Curated parser skill\n")
    (task_dir / "instruction.md").write_text("Original instruction\n")
    (task_dir / "task.toml").write_text(
        "\n".join(
            [
                'version = "1.0"',
                "",
                "[metadata]",
                'difficulty = "easy"',
                'category = "parsing"',
                'tags = ["format"]',
                "",
                "[verifier]",
                "timeout_sec = 30",
            ]
        )
    )
    repo = SkillsBenchRepository(
        root_dir=tmp_path / "benchmarks",
        task_dirs=["tasks"],
    )
    task = repo.resolve("sample-task")
    executor_policy = "# Executor Policy\n\nUse curated skills as resources."

    run_dir = repo.prepare_run_workspace(
        task=task,
        destination_root=tmp_path / "runs",
        planner_instruction="Planner instruction",
        injected_skill_text=executor_policy,
        injected_skill_name="executor",
    )

    instruction = (run_dir / "instruction.md").read_text()
    assert "# Executor Policy" in instruction
    assert "Use curated skills as resources." in instruction
    assert "# Task Resources" in instruction
    assert "curated-parser" in instruction
    assert "# Verifier Contract" in instruction
    assert "SkillsBench verifier" in instruction
    assert not (run_dir / "environment" / "skills" / "executor" / "SKILL.md").exists()
    assert (run_dir / "environment" / "skills" / "curated-parser" / "SKILL.md").exists()

    metadata = repo.executor_envelope_metadata(
        run_dir=run_dir,
        executor_policy=executor_policy,
    )
    assert metadata["executor_policy_hash"] == executor_policy_hash(executor_policy)
    assert metadata["executor_policy_injection"] == "instruction_envelope"
    assert metadata["task_resource_names"] == "curated-parser"


def test_harbor_runner_passes_agent_setup_timeout_multiplier(monkeypatch, tmp_path):
    calls = []

    class _Process:
        returncode = 0

        async def communicate(self):
            return b"", b""

    async def _fake_create_subprocess_exec(*cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _Process()

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        _fake_create_subprocess_exec,
    )
    monkeypatch.setattr(
        "mediated_coevo.benchmarks.skillsbench.shutil.which",
        lambda name: "/usr/local/bin/harbor",
    )

    task_dir = tmp_path / "task"
    task_dir.mkdir()
    runner = HarborRunner(
        agent_name="hermes",
        jobs_dir=tmp_path / "jobs",
        agent_setup_timeout_multiplier=2.5,
    )

    asyncio.run(runner.run(task_dir, "google/gemini-3-flash-preview"))

    cmd = calls[0][0]
    kwargs = calls[0][1]
    flag_index = cmd.index("--agent-setup-timeout-multiplier")
    assert cmd[flag_index + 1] == "2.5"
    assert kwargs["start_new_session"] is True


def test_harbor_exception_with_reward_counts_as_env_failure(tmp_path):
    job_dir = tmp_path / "job"
    trial_dir = job_dir / "run-agentops__abc123"
    trial_dir.mkdir(parents=True)

    (job_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "job-123",
                "stats": {
                    "evals": {
                        "opencode__model__adhoc": {
                            "metrics": [{"mean": 0.0}],
                        }
                    }
                },
            }
        )
    )
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "id": "trial-123",
                "trial_name": "run-agentops__abc123",
                "agent_result": {
                    "n_input_tokens": 0,
                    "n_output_tokens": 0,
                },
                "exception_info": {
                    "exception_type": "NonZeroAgentExitCodeError",
                    "exception_message": "opencode: command not found",
                },
            }
        )
    )

    trace = parse_execution_trace(
        HarborRunResult(
            job_dir=job_dir,
            trial_dir=trial_dir,
            returncode=0,
            stdout="",
            stderr="",
        ),
        task_id="fix-build-agentops",
        iteration=0,
        duration_sec=1.0,
    )

    assert trace.status == "env_failure"
    assert trace.error_kind == "harbor_exception"
    assert trace.reward == 0.0

    summary = build_score_summary(
        [
            IterationRecord(
                iteration=0,
                task_id="fix-build-agentops",
                reward=trace.reward,
                execution_trace=trace,
            )
        ],
        bootstrap_samples=50,
    )

    assert summary.total_runs == 1
    assert summary.env_failure_count == 1
    assert summary.scored_count == 0
    assert summary.mean_reward is None
