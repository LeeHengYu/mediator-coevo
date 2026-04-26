"""Unit tests for SkillsBenchRepository.resolve and prepare_run_workspace."""

from __future__ import annotations

from pathlib import Path

import pytest

from mediated_coevo.benchmarks.skillsbench import (
    SkillsBenchRepository,
    SkillsBenchTask,
)


def _scaffold_task(
    root: Path,
    *,
    bucket: str,
    task_id: str,
    instruction: str = "Do the thing.",
    task_toml: str = 'name = "demo"\n',
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Create a minimal valid task directory under root/bucket/task_id."""
    task_dir = root / bucket / task_id
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text(instruction)
    (task_dir / "task.toml").write_text(task_toml)
    for rel, body in (extra_files or {}).items():
        target = task_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(body)
    return task_dir


def test_resolve_finds_task_in_first_bucket(tmp_path):
    _scaffold_task(tmp_path, bucket="tasks", task_id="t1", instruction="hi")
    repo = SkillsBenchRepository(root_dir=tmp_path, task_dirs=["tasks"])
    task = repo.resolve("t1")

    assert isinstance(task, SkillsBenchTask)
    assert task.task_id == "t1"
    assert task.instruction == "hi"
    assert task.task_config["name"] == "demo"


def test_resolve_searches_buckets_in_order(tmp_path):
    # Same task_id in two buckets — first listed wins.
    _scaffold_task(tmp_path, bucket="primary", task_id="t1", instruction="primary")
    _scaffold_task(tmp_path, bucket="fallback", task_id="t1", instruction="fallback")
    repo = SkillsBenchRepository(
        root_dir=tmp_path, task_dirs=["primary", "fallback"]
    )
    task = repo.resolve("t1")

    assert task.instruction == "primary"


def test_resolve_falls_through_to_later_bucket(tmp_path):
    _scaffold_task(tmp_path, bucket="fallback", task_id="t2", instruction="from-fb")
    repo = SkillsBenchRepository(
        root_dir=tmp_path, task_dirs=["primary", "fallback"]
    )
    task = repo.resolve("t2")

    assert task.instruction == "from-fb"


def test_resolve_missing_task_raises_with_searched_paths(tmp_path):
    repo = SkillsBenchRepository(
        root_dir=tmp_path, task_dirs=["primary", "fallback"]
    )
    with pytest.raises(FileNotFoundError) as excinfo:
        repo.resolve("missing")

    msg = str(excinfo.value)
    assert "missing" in msg
    assert str(tmp_path / "primary" / "missing") in msg
    assert str(tmp_path / "fallback" / "missing") in msg


def test_resolve_missing_instruction_md_raises(tmp_path):
    task_dir = tmp_path / "tasks" / "broken"
    task_dir.mkdir(parents=True)
    (task_dir / "task.toml").write_text("name = 'x'\n")
    repo = SkillsBenchRepository(root_dir=tmp_path, task_dirs=["tasks"])

    with pytest.raises(FileNotFoundError, match="instruction.md"):
        repo.resolve("broken")


def test_resolve_missing_task_toml_raises(tmp_path):
    task_dir = tmp_path / "tasks" / "broken"
    task_dir.mkdir(parents=True)
    (task_dir / "instruction.md").write_text("hi")
    repo = SkillsBenchRepository(root_dir=tmp_path, task_dirs=["tasks"])

    with pytest.raises(FileNotFoundError, match="task.toml"):
        repo.resolve("broken")


def test_prepare_run_workspace_writes_instruction_and_skill(tmp_path):
    _scaffold_task(
        tmp_path,
        bucket="tasks",
        task_id="t1",
        extra_files={"environment/Dockerfile": "FROM scratch\n"},
    )
    repo = SkillsBenchRepository(root_dir=tmp_path, task_dirs=["tasks"])
    task = repo.resolve("t1")
    dest = tmp_path / "workspaces"

    run_dir = repo.prepare_run_workspace(
        task=task,
        destination_root=dest,
        planner_instruction="planner-rewrite",
        injected_skill_text="# learned\nuse the force",
        injected_skill_name="executor-evolved",
    )

    assert run_dir.exists()
    assert run_dir.parent == dest / "t1"
    assert (run_dir / "instruction.md").read_text() == "planner-rewrite"
    skill_path = (
        run_dir / "environment" / "skills" / "executor-evolved" / "SKILL.md"
    )
    assert skill_path.exists()
    assert "use the force" in skill_path.read_text()
    # Original task tree was copied in.
    assert (run_dir / "environment" / "Dockerfile").read_text() == "FROM scratch\n"


def test_prepare_run_workspace_without_skill_does_not_create_skill_dir(tmp_path):
    _scaffold_task(tmp_path, bucket="tasks", task_id="t1")
    repo = SkillsBenchRepository(root_dir=tmp_path, task_dirs=["tasks"])
    task = repo.resolve("t1")

    run_dir = repo.prepare_run_workspace(
        task=task,
        destination_root=tmp_path / "ws",
        planner_instruction="hi",
        injected_skill_text=None,
        injected_skill_name="executor-evolved",
    )

    assert not (run_dir / "environment" / "skills" / "executor-evolved").exists()
