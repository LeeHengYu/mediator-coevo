from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

import mediated_coevo.benchmarks.lifelong_agent_bench as lifelong_benchmark
from mediated_coevo.benchmarks.lifelong_agent_bench import (
    OS_BASE_IMAGE,
    LifelongAgentBenchEnvironmentError,
    LifelongAgentBenchImportError,
    load_lifelong_agent_bench_rows,
    materialize_lifelong_agent_bench,
)
from mediated_coevo.benchmarks.skillflow import TaskPackageRepository
from mediated_coevo.cli.sequence import _select_sequence_tasks


def _os_row(sample_index: int) -> dict[str, object]:
    return {
        "sample_index": sample_index,
        "instruction": f"Create marker-{sample_index}.txt containing ready.",
        "initialization_command_item": repr(
            {
                "command_name": "bash",
                "script": "printf 'initial\\n' > initial.txt",
            }
        ),
        "evaluation_info": repr(
            {
                "ground_truth_command_item": {
                    "command_name": "bash",
                    "script": "printf 'ready\\n' > marker.txt",
                },
                "evaluation_command_item": {
                    "command_name": "bash",
                    "script": "test \"$(cat marker.txt)\" = ready",
                },
            }
        ),
        "skill_list": ["cat", "echo"],
    }


def test_requires_explicit_shared_base_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        lifelong_benchmark.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args, 1),
    )

    with pytest.raises(
        LifelongAgentBenchEnvironmentError,
        match="build-base-image",
    ):
        lifelong_benchmark.ensure_os_base_image()


def test_materializes_os_task_without_oracle_leakage(tmp_path: Path) -> None:
    tasks_root = tmp_path / "tasks"

    [task_dir] = materialize_lifelong_agent_bench(
        family="os_interaction",
        rows=[_os_row(7)],
        tasks_root=tasks_root,
    )

    assert task_dir == tasks_root / "os_interaction" / "lab-os-7"
    assert (task_dir / "instruction.md").is_file()
    assert (task_dir / "task.toml").is_file()
    assert (task_dir / "environment" / "Dockerfile").is_file()
    assert (task_dir / "tests" / "test.sh").is_file()
    agent_visible = "\n".join(
        path.read_text()
        for path in (
            task_dir / "instruction.md",
            task_dir / "task.toml",
            task_dir / "environment" / "Dockerfile",
            task_dir / "environment" / "initialize.sh",
        )
    )
    assert "ground_truth_command_item" not in agent_visible
    assert "printf 'ready" not in agent_visible
    assert "skill_list" not in agent_visible
    assert "test \"$(cat marker.txt)\" = ready" in (
        task_dir / "tests" / "evaluate.sh"
    ).read_text()
    assert 'docker_image = "harbor-prebuilt:lifelong-agent-bench-os-7"' in (
        task_dir / "task.toml"
    ).read_text()
    assert "allow_internet = true" in (task_dir / "task.toml").read_text()
    dockerfile = (task_dir / "environment" / "Dockerfile").read_text()
    assert dockerfile.startswith(f"FROM {OS_BASE_IMAGE}\n")
    assert "apt-get" not in dockerfile


def test_materialized_family_uses_existing_default_ten_task_sampler(
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "tasks"
    materialize_lifelong_agent_bench(
        family="os_interaction",
        rows=[_os_row(index) for index in range(12)],
        tasks_root=tasks_root,
    )
    repository = TaskPackageRepository(tmp_path, ["tasks"])

    selected = _select_sequence_tasks(
        repository,
        "os_interaction",
        seed=3,
        length=10,
        warmup_count=3,
    )

    assert len(selected) == 10
    assert len(set(selected)) == 10
    assert all(task_id.startswith("os_interaction/") for task_id in selected)


def test_materialized_verifier_writes_binary_reward(tmp_path: Path) -> None:
    [task_dir] = materialize_lifelong_agent_bench(
        family="os_interaction",
        rows=[_os_row(2)],
        tasks_root=tmp_path / "tasks",
    )
    task_root = tmp_path / "runtime"
    verifier_dir = tmp_path / "verifier"
    task_root.mkdir()
    (task_root / "marker.txt").write_text("ready\n")
    environment = {
        **os.environ,
        "TASK_ROOT": str(task_root),
        "VERIFIER_DIR": str(verifier_dir),
    }

    completed = subprocess.run(
        ["bash", str(task_dir / "tests" / "test.sh")],
        env=environment,
        check=False,
    )

    assert completed.returncode == 0
    assert (verifier_dir / "reward.txt").read_text() == "1\n"


def test_empty_upstream_initialization_is_a_valid_noop(tmp_path: Path) -> None:
    row = _os_row(4)
    row["initialization_command_item"] = repr(
        {"command_name": "bash", "script": ""}
    )

    [task_dir] = materialize_lifelong_agent_bench(
        family="os_interaction",
        rows=[row],
        tasks_root=tmp_path / "tasks",
    )

    assert (task_dir / "environment" / "initialize.sh").read_text() == "\n"


def test_invalid_batch_does_not_leave_partial_tasks(tmp_path: Path) -> None:
    invalid_row = _os_row(2)
    invalid_row["evaluation_info"] = "{}"
    tasks_root = tmp_path / "tasks"

    with pytest.raises(LifelongAgentBenchImportError):
        materialize_lifelong_agent_bench(
            family="os_interaction",
            rows=[_os_row(1), invalid_row],
            tasks_root=tasks_root,
        )

    assert not tasks_root.exists()


def test_materialization_is_idempotent_but_rejects_modified_tasks(
    tmp_path: Path,
) -> None:
    tasks_root = tmp_path / "tasks"
    rows = [_os_row(5)]
    first = materialize_lifelong_agent_bench(
        family="os_interaction",
        rows=rows,
        tasks_root=tasks_root,
    )

    assert materialize_lifelong_agent_bench(
        family="os_interaction",
        rows=rows,
        tasks_root=tasks_root,
    ) == first

    (first[0] / "instruction.md").write_text("modified\n")
    with pytest.raises(LifelongAgentBenchImportError, match="differs from source"):
        materialize_lifelong_agent_bench(
            family="os_interaction",
            rows=rows,
            tasks_root=tasks_root,
        )


def test_jsonl_loader_and_family_fidelity_gate(tmp_path: Path) -> None:
    source = tmp_path / "rows.jsonl"
    source.write_text(json.dumps(_os_row(1)) + "\n")

    rows = load_lifelong_agent_bench_rows(source)

    assert rows[0]["sample_index"] == 1
    with pytest.raises(LifelongAgentBenchImportError, match="not executable yet"):
        materialize_lifelong_agent_bench(
            family="knowledge_graph",
            rows=rows,
            tasks_root=tmp_path / "tasks",
        )
