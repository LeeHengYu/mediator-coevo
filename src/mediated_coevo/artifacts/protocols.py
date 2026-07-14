"""Protocols for projection and transactional artifact-bank persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from mediated_coevo.artifacts.models import ArtifactBankUpdate
from mediated_coevo.diffusion.models import DiffusionArtifact
from mediated_coevo.execution.models import TaskExecutionResult, TaskProfile


class ArtifactProjector(Protocol):
    """Project compact transfer artifacts from one completed task run."""

    async def project(
        self,
        *,
        task: TaskProfile,
        execution: TaskExecutionResult,
    ) -> tuple[DiffusionArtifact, ...]: ...


class ArtifactBankUpdater(Protocol):
    """Validate and durably persist one append-only bank transition."""

    def prepare(
        self,
        *,
        run_id: str,
        position: int,
        task: TaskProfile,
        execution: TaskExecutionResult,
        current_bank: tuple[DiffusionArtifact, ...],
        projected_artifacts: tuple[DiffusionArtifact, ...],
    ) -> ArtifactBankUpdate: ...

    def persist(self, update: ArtifactBankUpdate) -> tuple[Path, ...]: ...

    def rollback(self, paths: tuple[Path, ...]) -> None: ...
