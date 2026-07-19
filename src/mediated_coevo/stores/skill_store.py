"""Read-only store for fixed prompt-injected SKILL.md files."""

from __future__ import annotations

import hashlib
from pathlib import Path


class SkillStore:
    """Loads and validates immutable runtime skills."""

    ENTRYPOINT = "SKILL.md"

    def __init__(self, skills_dir: Path) -> None:
        self._skills_dir = skills_dir

    @staticmethod
    def content_hash(content: str) -> str:
        """Return a stable content hash for skill text."""
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def read_skill(self, skill_name: str) -> str | None:
        """Read SKILL.md content by skill name (directory name)."""
        skill_path = self._skills_dir / skill_name / self.ENTRYPOINT
        if not skill_path.exists():
            return None
        return skill_path.read_text()

    def skill_hash(self, skill_name: str) -> str | None:
        """Return the SHA-256 hash for a skill's canonical entrypoint."""
        content = self.read_skill(skill_name)
        if content is None:
            return None
        return self.content_hash(content)

    def skill_hashes(self) -> dict[str, str]:
        """Return deterministic hashes for all canonical runtime skills."""
        hashes: dict[str, str] = {}
        for skill_name in self.list_skills():
            skill_hash = self.skill_hash(skill_name)
            if skill_hash is not None:
                hashes[skill_name] = skill_hash
        return hashes

    def list_skills(self) -> list[str]:
        """List skill directory names with canonical SKILL.md entrypoints."""
        return sorted(
            d.name
            for d in self._skills_dir.iterdir()
            if d.is_dir() and (d / self.ENTRYPOINT).is_file()
        )

    def validate(self) -> None:
        """Fail fast unless each skill dir has exactly one SKILL.md entrypoint."""
        invalid_entries: list[str] = []
        for skill_dir in sorted(self._skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            markdown_paths = sorted(
                path
                for path in skill_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".md"
            )
            valid = (
                len(markdown_paths) == 1 and markdown_paths[0].name == self.ENTRYPOINT
            )
            if not valid:
                found = ", ".join(path.name for path in markdown_paths) or "none"
                invalid_entries.append(f"{skill_dir} (markdown files: {found})")
        if invalid_entries:
            invalid = "; ".join(invalid_entries)
            raise ValueError(
                f"Invalid skill directory entrypoint(s): {invalid}. "
                f"Each skill directory must contain exactly one markdown "
                f"entrypoint named {self.ENTRYPOINT}."
            )
