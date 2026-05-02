"""Unit tests for SkillsBenchRepository.resolve and prepare_run_workspace."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import pytest

from mediated_coevo.benchmarks.skillsbench import (
    SKILLSBENCH_ARCHIVE_URL,
    SkillsBenchFetchError,
    SkillsBenchRemoteConfig,
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


def _zip_task(
    archive_path: Path,
    *,
    bucket: str,
    task_id: str,
    instruction: str = "remote instruction",
    task_toml: str = 'name = "remote-demo"\n',
    root: str = "skillsbench-main",
    extra_files: dict[str, str] | None = None,
) -> Path:
    """Create a SkillsBench-like repository zip with one task."""
    with ZipFile(archive_path, "w") as archive:
        base = f"{root}/{bucket}/{task_id}"
        archive.writestr(f"{base}/instruction.md", instruction)
        archive.writestr(f"{base}/task.toml", task_toml)
        for rel, body in (extra_files or {}).items():
            archive.writestr(f"{base}/{rel}", body)
    return archive_path


def _repository_with_archive(
    *,
    root_dir: Path,
    task_dirs: list[str],
    archive_path: Path,
) -> SkillsBenchRepository:
    """Build a remote-enabled repo backed by a test fixture archive."""
    archive_bytes = archive_path.read_bytes()
    repo = SkillsBenchRepository(
        root_dir=root_dir,
        task_dirs=task_dirs,
        remote=SkillsBenchRemoteConfig(enabled=True),
    )

    def download_fixture(_archive_url: str) -> bytes:
        return archive_bytes

    repo._download_archive = download_fixture  # type: ignore[method-assign]
    return repo


def test_resolve_finds_task_in_default_cache_dir(tmp_path):
    _scaffold_task(tmp_path, bucket="tasks", task_id="t1", instruction="hi")
    repo = SkillsBenchRepository(root_dir=tmp_path, task_dirs=["tasks"])
    task = repo.resolve("t1")

    assert isinstance(task, SkillsBenchTask)
    assert task.task_id == "t1"
    assert task.instruction == "hi"
    assert task.task_config["name"] == "demo"


def test_default_local_cache_dir_points_to_task_folder_parent():
    repo = SkillsBenchRepository(
        root_dir=Path("benchmarks/skillsbench"),
        task_dirs=["tasks"],
    )

    assert repo.default_local_cache_dir() == Path("benchmarks/skillsbench/tasks")


def test_resolve_uses_only_default_cache_dir(tmp_path):
    _scaffold_task(tmp_path, bucket="other", task_id="t1", instruction="ignored")
    repo = SkillsBenchRepository(root_dir=tmp_path, task_dirs=["tasks", "other"])

    with pytest.raises(FileNotFoundError) as excinfo:
        repo.resolve("t1")

    msg = str(excinfo.value)
    assert str(tmp_path / "tasks" / "t1") in msg
    assert str(tmp_path / "other" / "t1") not in msg


def test_resolve_missing_task_raises_with_searched_paths(tmp_path):
    repo = SkillsBenchRepository(root_dir=tmp_path, task_dirs=["tasks"])
    with pytest.raises(FileNotFoundError) as excinfo:
        repo.resolve("missing")

    msg = str(excinfo.value)
    assert "missing" in msg
    assert str(tmp_path / "tasks" / "missing") in msg


def test_list_local_task_ids_returns_safe_unique_ids_from_default_cache_dir(tmp_path):
    _scaffold_task(tmp_path, bucket="tasks", task_id="b-task")
    _scaffold_task(tmp_path, bucket="tasks", task_id="a-task")
    _scaffold_task(tmp_path, bucket="other", task_id="c-task")
    (tmp_path / "tasks" / ".fetch-tmp").mkdir()
    repo = SkillsBenchRepository(root_dir=tmp_path, task_dirs=["tasks", "other"])

    assert repo.list_local_task_ids() == ["a-task", "b-task"]


def test_resolve_local_task_wins_before_remote_fetch(tmp_path):
    _scaffold_task(tmp_path, bucket="tasks", task_id="t1", instruction="local")
    repo = SkillsBenchRepository(
        root_dir=tmp_path,
        task_dirs=["tasks"],
        remote=SkillsBenchRemoteConfig(enabled=True),
    )

    task = repo.resolve("t1")

    assert task.instruction == "local"


def test_resolve_remote_fetch_failure_names_task_archive_and_local_paths(tmp_path):
    repo = SkillsBenchRepository(
        root_dir=tmp_path,
        task_dirs=["tasks"],
        remote=SkillsBenchRemoteConfig(enabled=True),
    )

    def fail_download(_archive_url: str) -> bytes:
        raise SkillsBenchFetchError("network unavailable")

    repo._download_archive = fail_download  # type: ignore[method-assign]

    with pytest.raises(SkillsBenchFetchError) as excinfo:
        repo.resolve("missing")

    msg = str(excinfo.value)
    assert "missing" in msg
    assert SKILLSBENCH_ARCHIVE_URL in msg
    assert str(tmp_path / "tasks" / "missing") in msg
    assert "network unavailable" in msg


def test_resolve_fetches_and_caches_missing_remote_task(tmp_path):
    bench_root = tmp_path / "benchmarks" / "skillsbench"
    archive = _zip_task(
        tmp_path / "skillsbench.zip",
        bucket="tasks",
        task_id="remote-task",
        instruction="from remote",
        extra_files={"environment/Dockerfile": "FROM scratch\n"},
    )
    repo = _repository_with_archive(
        root_dir=bench_root,
        task_dirs=["tasks"],
        archive_path=archive,
    )

    task = repo.resolve("remote-task")

    assert task.task_id == "remote-task"
    assert task.instruction == "from remote"
    cached_task_dir = bench_root / "tasks" / "remote-task"
    assert task.task_dir == cached_task_dir
    assert (cached_task_dir / "task.toml").exists()
    assert (
        cached_task_dir / "environment" / "Dockerfile"
    ).read_text() == "FROM scratch\n"


def test_list_remote_task_ids_reads_safe_ids_from_archive(tmp_path):
    archive_path = tmp_path / "skillsbench.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("skillsbench-main/tasks/b-task/instruction.md", "b")
        archive.writestr("skillsbench-main/tasks/b-task/task.toml", 'name = "b"\n')
        archive.writestr("tasks/a-task/instruction.md", "a")
        archive.writestr("tasks/a-task/task.toml", 'name = "a"\n')
        archive.writestr("skillsbench-main/other/ignored/task.toml", 'name = "x"\n')

    repo = _repository_with_archive(
        root_dir=tmp_path / "bench",
        task_dirs=["tasks"],
        archive_path=archive_path,
    )

    assert repo.list_remote_task_ids() == ["a-task", "b-task"]


def test_remote_archive_bytes_are_reused_within_repository(tmp_path):
    archive = _zip_task(
        tmp_path / "skillsbench.zip",
        bucket="tasks",
        task_id="remote-task",
        instruction="from remote",
    )
    archive_bytes = archive.read_bytes()
    calls = 0
    repo = SkillsBenchRepository(
        root_dir=tmp_path / "bench",
        task_dirs=["tasks"],
        remote=SkillsBenchRemoteConfig(enabled=True),
    )

    def download_once(_archive_url: str) -> bytes:
        nonlocal calls
        calls += 1
        return archive_bytes

    repo._download_archive = download_once  # type: ignore[method-assign]

    assert repo.list_remote_task_ids() == ["remote-task"]
    assert repo.resolve("remote-task").instruction == "from remote"
    assert calls == 1


def test_resolve_remote_fetch_uses_default_cache_dir(tmp_path):
    archive_path = tmp_path / "skillsbench.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "skillsbench-main/tasks/t1/instruction.md",
            "tasks instruction",
        )
        archive.writestr("skillsbench-main/tasks/t1/task.toml", 'name = "t"\n')
        archive.writestr(
            "skillsbench-main/other/t1/instruction.md",
            "other instruction",
        )
        archive.writestr("skillsbench-main/other/t1/task.toml", 'name = "o"\n')
    repo = _repository_with_archive(
        root_dir=tmp_path / "bench",
        task_dirs=["tasks", "other"],
        archive_path=archive_path,
    )

    task = repo.resolve("t1")

    assert task.instruction == "tasks instruction"
    assert task.task_dir == tmp_path / "bench" / "tasks" / "t1"


def test_resolve_remote_fetch_rejects_unsafe_archive_path(tmp_path):
    archive_path = tmp_path / "skillsbench.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("skillsbench-main/tasks/bad/../instruction.md", "bad")
        archive.writestr("skillsbench-main/tasks/bad/task.toml", 'name = "bad"\n')
    repo = _repository_with_archive(
        root_dir=tmp_path / "bench",
        task_dirs=["tasks"],
        archive_path=archive_path,
    )

    with pytest.raises(SkillsBenchFetchError, match="Unsafe path"):
        repo.resolve("bad")


def test_resolve_remote_fetch_validates_required_files(tmp_path):
    archive_path = tmp_path / "skillsbench.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("skillsbench-main/tasks/broken/instruction.md", "hi")
    repo = _repository_with_archive(
        root_dir=tmp_path / "bench",
        task_dirs=["tasks"],
        archive_path=archive_path,
    )

    with pytest.raises(FileNotFoundError, match="task.toml"):
        repo.resolve("broken")


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
