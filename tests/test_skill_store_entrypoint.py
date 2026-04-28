from __future__ import annotations

import pytest

from mediated_coevo.stores.skill_store import SkillStore


def test_list_skills_only_includes_canonical_entrypoints(tmp_path):
    (tmp_path / "valid-b" / "SKILL.md").parent.mkdir(parents=True)
    (tmp_path / "valid-b" / "SKILL.md").write_text("# B\n")
    (tmp_path / "valid-a" / "SKILL.md").parent.mkdir(parents=True)
    (tmp_path / "valid-a" / "SKILL.md").write_text("# A\n")
    (tmp_path / "legacy" / "README.md").parent.mkdir(parents=True)
    (tmp_path / "legacy" / "README.md").write_text("# Legacy\n")

    store = SkillStore(tmp_path)

    assert store.list_skills() == ["valid-a", "valid-b"]


def test_validate_accepts_canonical_entrypoints(tmp_path):
    (tmp_path / "executor" / "SKILL.md").parent.mkdir(parents=True)
    (tmp_path / "executor" / "SKILL.md").write_text("# Executor\n")

    store = SkillStore(tmp_path)

    store.validate()


def test_validate_rejects_markdown_skill_without_canonical_entrypoint(tmp_path):
    (tmp_path / "legacy" / "README.md").parent.mkdir(parents=True)
    (tmp_path / "legacy" / "README.md").write_text("# Legacy\n")

    store = SkillStore(tmp_path)

    with pytest.raises(ValueError, match="SKILL.md"):
        store.validate()


def test_validate_rejects_extra_markdown_next_to_canonical_entrypoint(tmp_path):
    (tmp_path / "executor" / "SKILL.md").parent.mkdir(parents=True)
    (tmp_path / "executor" / "SKILL.md").write_text("# Executor\n")
    (tmp_path / "executor" / "README.md").write_text("# Notes\n")

    store = SkillStore(tmp_path)

    with pytest.raises(ValueError, match="exactly one markdown"):
        store.validate()


def test_validate_rejects_lowercase_skill_entrypoint(tmp_path):
    (tmp_path / "executor" / "skill.md").parent.mkdir(parents=True)
    (tmp_path / "executor" / "skill.md").write_text("# Executor\n")

    store = SkillStore(tmp_path)

    with pytest.raises(ValueError, match="SKILL.md"):
        store.validate()


def test_read_skill_returns_none_when_canonical_entrypoint_is_missing(tmp_path):
    (tmp_path / "legacy" / "README.md").parent.mkdir(parents=True)
    (tmp_path / "legacy" / "README.md").write_text("# Legacy\n")

    store = SkillStore(tmp_path)

    assert store.read_skill("legacy") is None


def test_write_skill_writes_canonical_entrypoint(tmp_path):
    store = SkillStore(tmp_path)

    skill_path = store.write_skill("executor", "# Executor\n")

    assert skill_path == tmp_path / "executor" / "SKILL.md"
    assert skill_path.read_text() == "# Executor\n"


def test_skill_hash_is_deterministic_and_content_sensitive(tmp_path):
    store = SkillStore(tmp_path)
    store.write_skill("executor", "# Executor\n")

    first_hash = store.skill_hash("executor")
    second_hash = store.skill_hash("executor")
    store.write_skill("executor", "# Executor\n\nUpdated.\n")
    updated_hash = store.skill_hash("executor")

    assert first_hash == second_hash
    assert first_hash == SkillStore.content_hash("# Executor\n")
    assert updated_hash != first_hash


def test_skill_hashes_returns_sorted_canonical_skill_hashes(tmp_path):
    store = SkillStore(tmp_path)
    store.write_skill("planner", "# Planner\n")
    store.write_skill("executor", "# Executor\n")

    assert store.skill_hashes() == {
        "executor": SkillStore.content_hash("# Executor\n"),
        "planner": SkillStore.content_hash("# Planner\n"),
    }


def test_restore_skill_copies_snapshot_entrypoint_and_verifies_hash(tmp_path):
    store = SkillStore(tmp_path / "skills")
    snapshot_dir = tmp_path / "snapshots" / "iter_0003"
    (snapshot_dir / "executor").mkdir(parents=True)
    (snapshot_dir / "executor" / "SKILL.md").write_text("# Restored\n")
    store.write_skill("executor", "# Current\n")

    restored_path = store.restore_skill("executor", snapshot_dir)

    assert restored_path == tmp_path / "skills" / "executor" / "SKILL.md"
    assert restored_path.read_text() == "# Restored\n"
    assert store.skill_hash("executor") == SkillStore.content_hash("# Restored\n")
