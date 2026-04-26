"""Skill store — SKILL.md files on disk.

SKILL.md files on disk are the source of truth for skill content. Per-iteration
version history is captured by `snapshot()` into `experiment_dir/skills_snapshots/`.
Edit provenance (reasoning, diff, reward) is tracked by `HistoryStore`.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


class SkillStore:
    """Manages SKILL.md files on disk."""

    ENTRYPOINT = "SKILL.md"

    def __init__(self, skills_dir: Path) -> None:
        self._skills_dir = skills_dir

    def read_skill(self, skill_name: str) -> str | None:
        """Read SKILL.md content by skill name (directory name)."""
        skill_path = self._skills_dir / skill_name / self.ENTRYPOINT
        if not skill_path.exists():
            return None
        return skill_path.read_text()

    def write_skill(self, skill_name: str, content: str) -> Path:
        """Write skill content to disk."""
        skill_dir = self._skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / self.ENTRYPOINT
        skill_path.write_text(content)
        logger.info("Wrote skill: %s", skill_path)
        return skill_path

    def list_skills(self) -> list[str]:
        """List skill directory names with canonical SKILL.md entrypoints."""
        return sorted(
            d.name
            for d in self._skills_dir.iterdir()
            if d.is_dir() and (d / self.ENTRYPOINT).is_file()
        )

    def validate(self) -> None:
        """Fail fast when markdown skill directories lack SKILL.md."""
        invalid_dirs = [
            d
            for d in sorted(self._skills_dir.iterdir())
            if d.is_dir() and not (d / self.ENTRYPOINT).is_file()
        ]
        if invalid_dirs:
            invalid = ", ".join(str(d) for d in invalid_dirs)
            raise ValueError(
                f"Invalid skill directory entrypoint(s): {invalid}. "
                f"Each skill directory must contain {self.ENTRYPOINT}."
            )

    def snapshot(self, iteration: int, snapshot_dir: Path) -> Path:
        """Copy the current skills/ directory into a versioned snapshot."""
        dest = snapshot_dir / f"iter_{iteration:04d}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self._skills_dir, dest)
        logger.info("Skill snapshot saved: %s", dest)
        return dest
