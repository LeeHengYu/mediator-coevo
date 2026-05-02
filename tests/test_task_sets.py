from __future__ import annotations

from zipfile import ZipFile

import pytest
import typer

from mediated_coevo.benchmarks.skillsbench import (
    SkillsBenchFetchError,
    SkillsBenchRemoteConfig,
    SkillsBenchRepository,
)
from mediated_coevo.benchmarks.task_sets import (
    SKILLSBENCH_10_TASK_IDS,
)
from mediated_coevo.main import (
    SKILLSBENCH_ALL_TASK_SET,
    _skillsbench_all_task_ids,
    _sync_task_ids_from_cli,
    _task_ids_from_cli,
)


def test_skillsbench_10_task_set_expands_to_curated_order():
    assert _task_ids_from_cli(None, "skillsbench-10") == list(
        SKILLSBENCH_10_TASK_IDS
    )


def test_explicit_tasks_override_named_task_set():
    assert _task_ids_from_cli("task-a, task-b", "skillsbench-10") == [
        "task-a",
        "task-b",
    ]


def test_no_task_selection_preserves_legacy_single_task_default():
    assert _task_ids_from_cli(None, None) == ["fix-build-google-auto"]


def test_unknown_task_set_raises_bad_parameter():
    with pytest.raises(typer.BadParameter, match="unknown task set"):
        _task_ids_from_cli(None, "missing-set")


def test_curated_skillsbench_task_ids_are_safe_and_unique():
    assert len(SKILLSBENCH_10_TASK_IDS) == 10
    assert len(set(SKILLSBENCH_10_TASK_IDS)) == len(SKILLSBENCH_10_TASK_IDS)
    assert "fix-build-google-auto" in SKILLSBENCH_10_TASK_IDS
    for task_id in SKILLSBENCH_10_TASK_IDS:
        assert task_id
        assert "/" not in task_id
        assert "\\" not in task_id
        assert task_id not in {".", ".."}


def test_dynamic_skillsbench_all_resolves_local_then_remote(tmp_path):
    local_task = tmp_path / "tasks" / "local-task"
    local_task.mkdir(parents=True)
    (local_task / "instruction.md").write_text("local")
    (local_task / "task.toml").write_text('name = "local"\n')
    archive_path = tmp_path / "skillsbench.zip"
    with ZipFile(archive_path, "w") as archive:
        archive.writestr("skillsbench-main/tasks/local-task/instruction.md", "local")
        archive.writestr(
            "skillsbench-main/tasks/local-task/task.toml",
            'name = "local"\n',
        )
        archive.writestr("skillsbench-main/tasks/remote-task/instruction.md", "remote")
        archive.writestr(
            "skillsbench-main/tasks/remote-task/task.toml",
            'name = "remote"\n',
        )
    repo = SkillsBenchRepository(
        root_dir=tmp_path,
        task_dirs=["tasks"],
        remote=SkillsBenchRemoteConfig(enabled=True),
    )
    archive_bytes = archive_path.read_bytes()

    def download_fixture(_archive_url: str) -> bytes:
        return archive_bytes

    repo._download_archive = download_fixture  # type: ignore[method-assign]

    assert _skillsbench_all_task_ids(repo) == ["local-task", "remote-task"]


def test_dynamic_skillsbench_all_fails_without_local_or_remote(tmp_path):
    repo = SkillsBenchRepository(
        root_dir=tmp_path,
        task_dirs=["tasks"],
        remote=SkillsBenchRemoteConfig(enabled=True),
    )

    def fail_download(_archive_url: str) -> bytes:
        raise SkillsBenchFetchError("network unavailable")

    repo._download_archive = fail_download  # type: ignore[method-assign]

    with pytest.raises(typer.BadParameter, match=SKILLSBENCH_ALL_TASK_SET):
        _skillsbench_all_task_ids(repo)


def test_sync_task_selection_rejects_skillsbench_all():
    with pytest.raises(typer.BadParameter, match="unsupported"):
        _sync_task_ids_from_cli(None, SKILLSBENCH_ALL_TASK_SET)


def test_sync_task_selection_accepts_curated_task_set():
    assert _sync_task_ids_from_cli(None, "skillsbench-10") == list(
        SKILLSBENCH_10_TASK_IDS
    )
