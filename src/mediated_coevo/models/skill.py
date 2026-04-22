"""Skill update models."""

from __future__ import annotations

from pydantic import BaseModel


class SkillEdit(BaseModel):
    """Shared base for any skill edit — draft or committed."""

    old_content: str
    new_content: str
    reasoning: str = ""


class SkillUpdate(SkillEdit):
    """A committed skill edit written to disk after advisor approval."""

    skill_id: str
    task_id: str = ""
    iteration: int = 0


class SkillProposal(SkillEdit):
    """Buffered, unreviewed proposal — never written to HistoryStore."""

    iteration: int
    task_id: str
    reward: float | None = None
