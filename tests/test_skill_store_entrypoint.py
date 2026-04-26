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
