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

    def __init__(self, skills_dir: Path) -> None:
        self._skills_dir = skills_dir

    def read_skill(self, skill_name: str) -> str | None:
        """Read SKILL.md content by skill name (directory name, also agent role string repr)."""
        # Try direct path first (e.g., "executor" → skills/executor/SKILL.md)
        skill_path = self._skills_dir / skill_name / "SKILL.md"
        if not skill_path.exists():
            # Try as a .md file directly (e.g., "planner" → skills/planner/skill-refiner.md)
            candidates = list((self._skills_dir / skill_name).glob("*.md"))
            if candidates:
                skill_path = candidates[0]
            else:
                return None
        return skill_path.read_text()

    def write_skill(self, skill_name: str, content: str) -> Path:
        """Write skill content to disk."""
        skill_dir = self._skills_dir / skill_name
        skill_dir.mkdir(parents=True, exist_ok=True)
        skill_path = skill_dir / "SKILL.md"
        skill_path.write_text(content)
        logger.info("Wrote skill: %s", skill_path)
        return skill_path

    def list_skills(self) -> list[str]:
        """List all skill directory names."""
        return [
            d.name
            for d in self._skills_dir.iterdir()
            if d.is_dir() and any(d.glob("*.md"))
        ]

    def get_all_skill_contents(self) -> dict[str, str]:
        """Return {skill_name: content} for all skills."""
        result = {}
        for name in self.list_skills():
            content = self.read_skill(name)
            if content:
                result[name] = content
        return result

    def snapshot(self, iteration: int, snapshot_dir: Path) -> Path:
        """Copy the current skills/ directory into a versioned snapshot."""
        dest = snapshot_dir / f"iter_{iteration:04d}"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(self._skills_dir, dest)
        logger.info("Skill snapshot saved: %s", dest)
        return dest
