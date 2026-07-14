"""Append-only artifact-bank transition models."""

from __future__ import annotations

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from mediated_coevo.diffusion.models import DiffusionArtifact
from mediated_coevo.execution.models import redact_sensitive_data


class ArtifactBankUpdate(BaseModel):
    """One validated append to a run-local transfer artifact bank."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    task_id: str = Field(min_length=1)
    before_artifact_ids: tuple[str, ...]
    added_artifacts: tuple[DiffusionArtifact, ...]
    after_artifact_ids: tuple[str, ...]

    @field_validator("added_artifacts")
    @classmethod
    def normalize_added_artifacts(
        cls,
        value: tuple[DiffusionArtifact, ...],
    ) -> tuple[DiffusionArtifact, ...]:
        return tuple(
            DiffusionArtifact.model_validate(
                redact_sensitive_data(artifact.model_dump(mode="python"))
            )
            for artifact in value
        )

    @property
    def added_artifact_ids(self) -> tuple[str, ...]:
        """Return IDs in their durable append order."""
        return tuple(artifact.artifact_id for artifact in self.added_artifacts)

    @model_validator(mode="after")
    def validate_append_only_transition(self) -> Self:
        """Enforce identity, uniqueness, provenance, and append-only ordering."""
        if self.run_id != self.run_id.strip():
            raise ValueError("run_id must not have surrounding whitespace")
        if self.task_id != self.task_id.strip():
            raise ValueError("task_id must not have surrounding whitespace")
        before = self.before_artifact_ids
        added = self.added_artifact_ids
        after = self.after_artifact_ids
        for label, values in (("before", before), ("added", added), ("after", after)):
            if any(not value or value != value.strip() for value in values):
                raise ValueError(f"{label} artifact IDs must be non-empty and stripped")
            if len(values) != len(set(values)):
                raise ValueError(f"{label} artifact IDs must be unique")
        if set(before).intersection(added):
            raise ValueError("artifact IDs cannot be added twice")
        if after != (*before, *added):
            raise ValueError("artifact bank update must append added artifacts in order")
        for artifact in self.added_artifacts:
            if artifact.source_task_id != self.task_id:
                raise ValueError("added artifact source_task_id must equal update task_id")
            if artifact.source_iteration != self.position:
                raise ValueError(
                    "added artifact source_iteration must equal update position"
                )
            if artifact.source_run_id != self.run_id:
                raise ValueError("added artifact source_run_id must equal update run_id")
        return self
