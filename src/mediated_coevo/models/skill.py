"""Skill update model."""

from __future__ import annotations

from pydantic import BaseModel


class SkillUpdate(BaseModel):
    """A proposed skill edit from the Planner."""

    skill_id: str
    old_content: str
    new_content: str
    reasoning: str = ""
    iteration: int = 0
