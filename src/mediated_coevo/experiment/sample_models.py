"""Immutable contracts for one causal warm-up-plus-suffix sample."""

from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime
from enum import Enum
from pathlib import PurePosixPath
from typing import Any, Literal, Self
from urllib.parse import unquote, urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from mediated_coevo.artifacts.models import ArtifactBankUpdate
from mediated_coevo.diffusion.models import DiffusionArtifact, TaskGraphSnapshot
from mediated_coevo.execution.models import (
    ContextPack,
    FrozenJsonDict,
    TaskExecutionResult,
    TaskProfile,
    empty_context_pack,
    normalized_json_object,
    redact_sensitive_text,
)
from mediated_coevo.orchestration.arms import OrchestrationArm

SchemaVersion = Literal[1]
Sha256 = str


class _FrozenV1Model(BaseModel):
    """Shared serialization boundary for new sample-runtime records."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    schema_version: SchemaVersion = 1


def _validate_identity(value: str, *, label: str) -> str:
    if not value or value != value.strip():
        raise ValueError(f"{label} must be a non-empty stripped string")
    return value


def _validate_path_component(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value != value.strip()
        or value in {".", ".."}
        or len(path.parts) != 1
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be one portable path component")
    return value


def _validate_sha256(value: str, *, label: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _validate_relative_path(value: str, *, label: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or value != value.strip()
        or path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or path.as_posix() != value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be a non-empty relative POSIX path")
    return value


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _mapping_contains_key(value: Any, target: str) -> bool:
    if isinstance(value, dict):
        target = target.casefold()
        return any(str(key).casefold() == target for key in value) or any(
            _mapping_contains_key(item, target) for item in value.values()
        )
    if isinstance(value, (list, tuple)):
        return any(_mapping_contains_key(item, target) for item in value)
    return False


class SequenceSpec(_FrozenV1Model):
    """Frozen task order and predeclared suffix-reward policy.

    This identity is arm-neutral. Duplicate task IDs are legal because a task
    occurrence is identified by its sequence position, not by task ID alone.
    """

    sequence_id: str = Field(min_length=1)
    tasks: tuple[TaskProfile, ...] = Field(min_length=1)
    warmup_count: int = Field(default=3, ge=0)
    policy_seed: int
    reward_weights: tuple[float, ...] | None = None
    task_set_id: str | None = None

    @field_validator("tasks", mode="before")
    @classmethod
    def revalidate_task_profiles(cls, value: Any) -> Any:
        """Rebuild existing model instances so ``model_copy`` cannot bypass checks."""
        if not isinstance(value, (list, tuple)):
            return value
        normalized: list[TaskProfile] = []
        for task in value:
            payload = (
                task.model_dump(mode="python")
                if isinstance(task, TaskProfile)
                else task
            )
            normalized.append(TaskProfile.model_validate(payload))
        return tuple(normalized)

    @field_validator("sequence_id")
    @classmethod
    def validate_sequence_id(cls, value: str) -> str:
        return _validate_identity(value, label="sequence_id")

    @field_validator("task_set_id")
    @classmethod
    def validate_task_set_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_identity(value, label="task_set_id")

    @model_validator(mode="after")
    def validate_suffix_contract(self) -> Self:
        if self.warmup_count >= len(self.tasks):
            raise ValueError("warmup_count must leave at least one suffix task")
        if self.reward_weights is not None:
            if len(self.reward_weights) != self.suffix_count:
                raise ValueError(
                    "reward_weights must have one entry per post-warm-up task"
                )
            if any(
                not math.isfinite(weight) or weight < 0
                for weight in self.reward_weights
            ):
                raise ValueError("reward_weights must be finite and nonnegative")
            if math.fsum(self.reward_weights) <= 0:
                raise ValueError("reward_weights must have a positive sum")
        return self

    @property
    def task_ids(self) -> tuple[str, ...]:
        """Return task occurrence identities without deduplicating them."""
        return tuple(task.task_id for task in self.tasks)

    @property
    def suffix_count(self) -> int:
        return len(self.tasks) - self.warmup_count

    @property
    def suffix_positions(self) -> tuple[int, ...]:
        return tuple(range(self.warmup_count, len(self.tasks)))


class SampleSpec(_FrozenV1Model):
    """One arm-specific execution of a frozen sequence."""

    sample_id: str = Field(min_length=1)
    sequence: SequenceSpec
    arm: OrchestrationArm
    warmup_bundle_id: Sha256 | None = None

    @field_validator("sample_id")
    @classmethod
    def validate_sample_id(cls, value: str) -> str:
        return _validate_path_component(value, label="sample_id")

    @field_validator("warmup_bundle_id")
    @classmethod
    def validate_bundle_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256(value, label="warmup_bundle_id")

    @model_validator(mode="after")
    def validate_identity_boundary(self) -> Self:
        if self.sample_id == self.sequence.sequence_id:
            raise ValueError("sample_id must differ from sequence_id")
        if self.sequence.warmup_count > 0 and self.warmup_bundle_id is None:
            raise ValueError("warmup_bundle_id is required when warmup_count > 0")
        if self.sequence.warmup_count == 0 and self.warmup_bundle_id is not None:
            raise ValueError(
                "warmup_bundle_id must be omitted when warmup_count is zero"
            )
        return self

    @property
    def run_id(self) -> str:
        """The graph and execution run ID for this arm-specific sample."""
        return self.sample_id


class AgentCallRecord(_FrozenV1Model):
    """Serializable raw input/output pair for one orchestration-agent call."""

    input_payload: dict[str, Any]
    output_payload: dict[str, Any]

    @field_validator("input_payload", "output_payload")
    @classmethod
    def normalize_payload(cls, value: Any, info: Any) -> FrozenJsonDict:
        return normalized_json_object(
            value,
            label=str(info.field_name),
            redact=True,
        )


class WarmupTaskRecord(_FrozenV1Model):
    """Arm-neutral state transition for one warm-up occurrence."""

    phase: Literal["warmup"] = "warmup"
    run_id: str = Field(min_length=1)
    sequence_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    task: TaskProfile
    artifact_ids_before: tuple[str, ...]
    context: ContextPack
    execution: TaskExecutionResult | None = None
    artifact_store_id: str | None = None
    bank_update: ArtifactBankUpdate

    @model_validator(mode="after")
    def validate_transition(self) -> Self:
        _validate_record_transition(
            run_id=self.run_id,
            position=self.position,
            task=self.task,
            artifact_ids_before=self.artifact_ids_before,
            execution=self.execution,
            bank_update=self.bank_update,
        )
        _validate_identity(self.sequence_id, label="sequence_id")
        if self.context != empty_context_pack():
            raise ValueError("warm-up records require the canonical empty context")
        if self.execution is None:
            if self.artifact_store_id != self.task.task_id:
                raise ValueError(
                    "stored warm-up record must identify its task artifact store"
                )
            return self
        if self.artifact_store_id is not None:
            raise ValueError("executed warm-up record cannot name an artifact store")
        if self.execution.metadata.get("phase") != "warmup":
            raise ValueError("warm-up execution metadata phase must equal warmup")
        treatment_keys = ("arm", "treatment_arm", "baseline_preset")
        if any(
            _mapping_contains_key(self.execution.metadata, key)
            for key in treatment_keys
        ):
            raise ValueError("warm-up execution metadata must be arm-neutral")
        if any(
            any(
                _mapping_contains_key(artifact.metadata, key)
                for key in treatment_keys
            )
            for artifact in self.bank_update.added_artifacts
        ):
            raise ValueError("warm-up artifact metadata must be arm-neutral")
        return self

    @property
    def task_id(self) -> str:
        return self.task.task_id


class SampleTaskRecord(_FrozenV1Model):
    """Arm-specific transition for one orchestrated suffix occurrence."""

    phase: Literal["orchestrated"] = "orchestrated"
    sample_id: str = Field(min_length=1)
    sequence_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    task: TaskProfile
    arm: OrchestrationArm
    artifact_ids_before: tuple[str, ...]
    graph_call: AgentCallRecord | None = None
    policy_call: AgentCallRecord | None = None
    context: ContextPack
    execution: TaskExecutionResult
    bank_update: ArtifactBankUpdate
    graph_snapshot_id_after: str | None = None

    @model_validator(mode="after")
    def validate_transition(self) -> Self:
        _validate_record_transition(
            run_id=self.sample_id,
            position=self.position,
            task=self.task,
            artifact_ids_before=self.artifact_ids_before,
            execution=self.execution,
            bank_update=self.bank_update,
        )
        _validate_identity(self.sequence_id, label="sequence_id")
        if self.execution.metadata.get("phase") != "orchestrated":
            raise ValueError(
                "sample execution metadata phase must equal orchestrated"
            )
        if self.execution.metadata.get("arm") != self.arm.value:
            raise ValueError("sample execution metadata arm must match its record")
        if self.arm is OrchestrationArm.EXECUTION_ONLY:
            if self.graph_call is not None or self.policy_call is not None:
                raise ValueError("execution_only cannot contain orchestration calls")
            if self.context != empty_context_pack():
                raise ValueError("execution_only requires the canonical empty context")
            if self.graph_snapshot_id_after is not None:
                raise ValueError("execution_only cannot carry graph identity")
        elif self.arm is OrchestrationArm.RANDOM_POLICY:
            if self.graph_call is not None or self.policy_call is None:
                raise ValueError("random_policy requires policy-only orchestration")
            if self.graph_snapshot_id_after is not None or self.context.snapshot_id:
                raise ValueError("random_policy cannot carry graph identity")
        elif self.arm is OrchestrationArm.NO_GRAPH:
            if self.graph_call is not None or self.policy_call is None:
                raise ValueError("no_graph requires a policy call without a graph call")
            if self.graph_snapshot_id_after is not None or self.context.snapshot_id:
                raise ValueError("no_graph cannot carry graph identity")
        else:
            if self.graph_call is None or self.policy_call is None:
                raise ValueError("full_orchestration requires graph and policy calls")
            if self.graph_snapshot_id_after is None:
                raise ValueError("full_orchestration requires a graph snapshot ID")
            if self.context.snapshot_id != self.graph_snapshot_id_after:
                raise ValueError("context must use the current graph snapshot")
        return self

    @property
    def run_id(self) -> str:
        return self.sample_id

    @property
    def task_id(self) -> str:
        return self.task.task_id


TaskRecord = WarmupTaskRecord | SampleTaskRecord


def _validate_record_transition(
    *,
    run_id: str,
    position: int,
    task: TaskProfile,
    artifact_ids_before: tuple[str, ...],
    execution: TaskExecutionResult | None,
    bank_update: ArtifactBankUpdate,
) -> None:
    _validate_identity(run_id, label="run_id")
    if len(artifact_ids_before) != len(set(artifact_ids_before)):
        raise ValueError("artifact_ids_before must be unique")
    if execution is not None:
        if execution.run_id != run_id:
            raise ValueError("execution run_id must match the task record")
        if execution.position != position:
            raise ValueError("execution position must match the task record")
        if execution.task_id != task.task_id:
            raise ValueError("execution task_id must match the frozen task profile")
    if bank_update.run_id != run_id:
        raise ValueError("bank update run_id must match the task record")
    if bank_update.position != position:
        raise ValueError("bank update position must match the task record")
    if bank_update.task_id != task.task_id:
        raise ValueError("bank update task_id must match the frozen task profile")
    if bank_update.before_artifact_ids != artifact_ids_before:
        raise ValueError("bank update must start from artifact_ids_before")


class SequenceRewards(_FrozenV1Model):
    """Suffix-only sequence reward with every zero and missing slot retained."""

    positions: tuple[int, ...]
    task_ids: tuple[str, ...]
    task_rewards: tuple[float | None, ...]
    weights: tuple[float, ...]
    completed_positions: tuple[int, ...]
    expected_count: int = Field(ge=0)
    scored_count: int = Field(ge=0)
    missing_count: int = Field(ge=0)
    all_tasks_completed: bool
    rewards_complete: bool
    valid_for_reporting: bool
    unweighted_sum: float | None
    unweighted_mean: float | None
    weighted_sum: float | None
    weighted_mean: float | None

    @model_validator(mode="after")
    def validate_reward_contract(self) -> Self:
        expected = self.expected_count
        aligned = (
            len(self.positions),
            len(self.task_ids),
            len(self.task_rewards),
            len(self.weights),
        )
        if any(length != expected for length in aligned):
            raise ValueError("reward positions, tasks, values, and weights must align")
        if tuple(sorted(self.positions)) != self.positions or len(
            set(self.positions)
        ) != len(self.positions):
            raise ValueError("reward positions must be ordered and unique")
        if any(position < 0 for position in self.positions):
            raise ValueError("reward positions must be nonnegative")
        if any(not task_id or task_id != task_id.strip() for task_id in self.task_ids):
            raise ValueError("reward task IDs must be non-empty stripped strings")
        if tuple(sorted(self.completed_positions)) != self.completed_positions or len(
            set(self.completed_positions)
        ) != len(self.completed_positions):
            raise ValueError("completed reward positions must be ordered and unique")
        if not set(self.completed_positions).issubset(self.positions):
            raise ValueError("completed reward positions must belong to the suffix")
        completed = set(self.completed_positions)
        if any(
            position not in completed and reward is not None
            for position, reward in zip(
                self.positions,
                self.task_rewards,
                strict=True,
            )
        ):
            raise ValueError("an uncompleted position cannot have a reward")
        if any(
            reward is not None and not math.isfinite(reward)
            for reward in self.task_rewards
        ):
            raise ValueError("task rewards must be finite when present")
        if any(not math.isfinite(weight) or weight < 0 for weight in self.weights):
            raise ValueError("reward weights must be finite and nonnegative")
        if expected == 0 or math.fsum(self.weights) <= 0:
            raise ValueError("reward weights must have a positive sum")

        scored = sum(reward is not None for reward in self.task_rewards)
        rewards_complete = scored == expected
        all_tasks_completed = self.completed_positions == self.positions
        valid = rewards_complete and all_tasks_completed
        if self.scored_count != scored or self.missing_count != expected - scored:
            raise ValueError("reward coverage counts are inconsistent")
        if self.rewards_complete is not rewards_complete:
            raise ValueError("rewards_complete is inconsistent with reward slots")
        if self.all_tasks_completed is not all_tasks_completed:
            raise ValueError("all_tasks_completed is inconsistent with positions")
        if self.valid_for_reporting is not valid:
            raise ValueError("valid_for_reporting is inconsistent with completion")

        aggregates = (
            self.unweighted_sum,
            self.unweighted_mean,
            self.weighted_sum,
            self.weighted_mean,
        )
        if not valid:
            if any(value is not None for value in aggregates):
                raise ValueError("invalid reward coverage cannot expose aggregates")
            return self

        numeric = tuple(reward for reward in self.task_rewards if reward is not None)
        unweighted_sum = math.fsum(numeric)
        weighted_sum = math.fsum(
            reward * weight
            for reward, weight in zip(numeric, self.weights, strict=True)
        )
        expected_aggregates = (
            unweighted_sum,
            unweighted_sum / expected,
            weighted_sum,
            weighted_sum / math.fsum(self.weights),
        )
        if any(
            actual is None or not math.isclose(actual, expected_value)
            for actual, expected_value in zip(
                aggregates,
                expected_aggregates,
                strict=True,
            )
        ):
            raise ValueError("sequence reward aggregates are inconsistent")
        return self

    @property
    def complete(self) -> bool:
        """Compatibility shorthand for fully reportable reward coverage."""
        return self.valid_for_reporting

    @property
    def unscored_count(self) -> int:
        return self.missing_count


def calculate_sequence_rewards(
    *,
    sequence: SequenceSpec,
    task_rewards: tuple[float | None, ...],
    completed_positions: tuple[int, ...],
) -> SequenceRewards:
    """Calculate only the frozen post-warm-up reward contract."""
    positions = sequence.suffix_positions
    if len(task_rewards) != len(positions):
        raise ValueError("task_rewards must preserve every suffix position")
    if len(completed_positions) != len(set(completed_positions)):
        raise ValueError("completed_positions must be unique")
    if tuple(sorted(completed_positions)) != completed_positions:
        raise ValueError("completed_positions must be ordered")
    if not set(completed_positions).issubset(positions):
        raise ValueError("completed_positions must belong to the suffix")
    completed = set(completed_positions)
    for position, reward in zip(positions, task_rewards, strict=True):
        if position not in completed and reward is not None:
            raise ValueError("an uncompleted position cannot have a reward")
        if reward is not None and not math.isfinite(reward):
            raise ValueError("task rewards must be finite when present")

    weights = sequence.reward_weights or tuple(1.0 for _ in positions)
    all_tasks_completed = completed_positions == positions
    rewards_complete = all(reward is not None for reward in task_rewards)
    valid_for_reporting = all_tasks_completed and rewards_complete
    unweighted_sum: float | None = None
    unweighted_mean: float | None = None
    weighted_sum: float | None = None
    weighted_mean: float | None = None
    if valid_for_reporting:
        numeric_rewards = tuple(reward for reward in task_rewards if reward is not None)
        unweighted_sum = math.fsum(numeric_rewards)
        unweighted_mean = unweighted_sum / len(numeric_rewards)
        weighted_sum = math.fsum(
            reward * weight
            for reward, weight in zip(numeric_rewards, weights, strict=True)
        )
        weighted_mean = weighted_sum / math.fsum(weights)

    scored_count = sum(reward is not None for reward in task_rewards)
    return SequenceRewards(
        positions=positions,
        task_ids=sequence.task_ids[sequence.warmup_count :],
        task_rewards=task_rewards,
        weights=weights,
        completed_positions=completed_positions,
        expected_count=len(positions),
        scored_count=scored_count,
        missing_count=len(positions) - scored_count,
        all_tasks_completed=all_tasks_completed,
        rewards_complete=rewards_complete,
        valid_for_reporting=valid_for_reporting,
        unweighted_sum=unweighted_sum,
        unweighted_mean=unweighted_mean,
        weighted_sum=weighted_sum,
        weighted_mean=weighted_mean,
    )


class ArchiveEntry(_FrozenV1Model):
    """Content-addressed file reference relative to the sequence workspace."""

    relative_path: str
    kind: str = Field(min_length=1)
    sha256: Sha256
    byte_size: int = Field(ge=0)

    @field_validator("relative_path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        return _validate_relative_path(value, label="archive relative_path")

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        return _validate_identity(value, label="archive kind")

    @field_validator("sha256")
    @classmethod
    def validate_digest(cls, value: str) -> str:
        return _validate_sha256(value, label="archive sha256")


class ExternalArchiveRef(_FrozenV1Model):
    """Explicit non-portable provenance that cannot be localized."""

    kind: str = Field(min_length=1)
    uri: str = Field(min_length=1)
    provenance: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provenance")
    @classmethod
    def normalize_provenance(cls, value: Any) -> FrozenJsonDict:
        return normalized_json_object(
            value,
            label="external archive provenance",
            redact=True,
        )

    @field_validator("kind")
    @classmethod
    def validate_kind(cls, value: str) -> str:
        return _validate_identity(value, label="kind")

    @field_validator("uri")
    @classmethod
    def validate_external_uri(cls, value: str) -> str:
        value = _validate_identity(value, label="uri")
        if redact_sensitive_text(value) != value:
            raise ValueError("external archive uri must not contain credentials")
        if "\\" in value or "\x00" in value:
            raise ValueError("external archive uri must use portable path syntax")
        parsed = urlsplit(value)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("external archive uri must not contain credentials or query")
        if not parsed.scheme:
            if not PurePosixPath(value).is_absolute():
                raise ValueError(
                    "external archive uri must be absolute or have a scheme"
                )
            return value
        if parsed.scheme == "file":
            if parsed.netloc or not PurePosixPath(unquote(parsed.path)).is_absolute():
                raise ValueError("file external archive uri must contain an absolute path")
            return value
        if not parsed.netloc or not parsed.hostname:
            raise ValueError("remote external archive uri must include an authority")
        try:
            parsed.port
        except ValueError as exc:
            raise ValueError("external archive uri contains an invalid port") from exc
        return value


class ArchiveManifest(_FrozenV1Model):
    """Portable archive inventory plus explicitly external references."""

    entries: tuple[ArchiveEntry, ...] = ()
    external_refs: tuple[ExternalArchiveRef, ...] = ()
    terminal_payload_sha256: Sha256 | None = None

    @field_validator("terminal_payload_sha256")
    @classmethod
    def validate_terminal_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256(value, label="terminal_payload_sha256")

    @model_validator(mode="after")
    def validate_unique_entries(self) -> Self:
        paths = tuple(entry.relative_path for entry in self.entries)
        if len(paths) != len(set(paths)):
            raise ValueError("archive entry relative paths must be unique")
        external = tuple((entry.kind, entry.uri) for entry in self.external_refs)
        if len(external) != len(set(external)):
            raise ValueError("external archive references must be unique")
        return self


class RuntimeProvenance(_FrozenV1Model):
    """Sanitized implementation and environment identity for one run."""

    implementation_revision: str = Field(min_length=1)
    implementation_dirty: bool
    config_hash: Sha256
    graph_implementation_hash: Sha256 | None = None
    policy_implementation_hash: Sha256 | None = None
    harness_hash: Sha256 | None = None
    model_mapping: dict[str, str] = Field(default_factory=dict)
    executor_backend: str = Field(min_length=1)
    executor_agent: str = Field(min_length=1)
    python_version: str = Field(min_length=1)
    package_version: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime

    @field_validator("model_mapping")
    @classmethod
    def normalize_model_mapping(cls, value: Any) -> FrozenJsonDict:
        return normalized_json_object(
            value,
            label="model_mapping",
            redact=True,
        )

    @field_validator(
        "implementation_revision",
        "executor_backend",
        "executor_agent",
        "python_version",
        "package_version",
    )
    @classmethod
    def validate_strings(cls, value: str, info: Any) -> str:
        return _validate_identity(value, label=str(info.field_name))

    @field_validator(
        "config_hash",
        "graph_implementation_hash",
        "policy_implementation_hash",
        "harness_hash",
    )
    @classmethod
    def validate_hashes(cls, value: str | None, info: Any) -> str | None:
        if value is None:
            return None
        return _validate_sha256(value, label=str(info.field_name))

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        if self.started_at.tzinfo is None or self.finished_at.tzinfo is None:
            raise ValueError("provenance timestamps must be timezone-aware")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at cannot precede started_at")
        for role, model in self.model_mapping.items():
            _validate_identity(role, label="model role")
            _validate_identity(model, label="model name")
            lowered = role.lower()
            if any(
                marker in lowered for marker in ("key", "secret", "token", "password")
            ):
                raise ValueError("model_mapping cannot contain credential-like keys")
        return self


class WarmupReference(_FrozenV1Model):
    """Compact pointer from an arm workspace to a shared warm-up bundle."""

    bundle_id: Sha256
    warmup_run_id: str = Field(min_length=1)
    relative_path: str
    manifest_path: str
    sequence_id: str = Field(min_length=1)
    warmup_count: int = Field(ge=1)

    @field_validator("bundle_id")
    @classmethod
    def validate_bundle_id(cls, value: str) -> str:
        return _validate_sha256(value, label="bundle_id")

    @field_validator("warmup_run_id", "sequence_id")
    @classmethod
    def validate_ids(cls, value: str, info: Any) -> str:
        return _validate_identity(value, label=str(info.field_name))

    @field_validator("relative_path", "manifest_path")
    @classmethod
    def validate_paths(cls, value: str, info: Any) -> str:
        return _validate_relative_path(value, label=str(info.field_name))


class WarmupBundle(_FrozenV1Model):
    """One shared, arm-neutral execution of the sequence prefix."""

    bundle_id: Sha256
    sequence_id: str = Field(min_length=1)
    warmup_run_id: str = Field(min_length=1)
    warmup_count: int = Field(ge=0)
    task_records: tuple[WarmupTaskRecord, ...]
    final_artifact_bank: tuple[DiffusionArtifact, ...]
    archive_manifest: ArchiveManifest
    provenance: RuntimeProvenance

    @field_validator("bundle_id")
    @classmethod
    def validate_bundle_id(cls, value: str) -> str:
        return _validate_sha256(value, label="bundle_id")

    @model_validator(mode="after")
    def validate_bundle(self) -> Self:
        _validate_identity(self.sequence_id, label="sequence_id")
        _validate_identity(self.warmup_run_id, label="warmup_run_id")
        if len(self.task_records) != self.warmup_count:
            raise ValueError("warm-up record count must equal warmup_count")
        expected_before: tuple[str, ...] = ()
        for position, record in enumerate(self.task_records):
            if record.position != position:
                raise ValueError("warm-up records must cover the prefix in order")
            if record.run_id != self.warmup_run_id:
                raise ValueError("warm-up record belongs to another warmup_run_id")
            if record.sequence_id != self.sequence_id:
                raise ValueError("warm-up record belongs to another sequence_id")
            if record.artifact_ids_before != expected_before:
                raise ValueError("warm-up records do not form one causal bank chain")
            expected_before = record.bank_update.after_artifact_ids
        final_ids = tuple(artifact.artifact_id for artifact in self.final_artifact_bank)
        if final_ids != expected_before:
            raise ValueError("final warm-up bank does not match its task records")
        if len(final_ids) != len(set(final_ids)):
            raise ValueError("final warm-up bank artifact IDs must be unique")
        _validate_final_artifact_objects(
            task_records=self.task_records,
            final_artifact_bank=self.final_artifact_bank,
            label="warm-up bundle",
        )
        if self.bundle_id != self.semantic_hash():
            raise ValueError("bundle_id does not match canonical semantic content")
        return self

    def semantic_hash(self) -> str:
        """Hash semantic warm-up state, excluding paths and provenance."""
        payload = {
            "schema_version": self.schema_version,
            "sequence_id": self.sequence_id,
            "warmup_count": self.warmup_count,
            "task_records": [
                record.model_dump(mode="json") for record in self.task_records
            ],
            "final_artifact_bank": [
                artifact.model_dump(mode="json")
                for artifact in self.final_artifact_bank
            ],
        }
        return _canonical_sha256(payload)

    @classmethod
    def create(
        cls,
        *,
        sequence_id: str,
        warmup_run_id: str,
        warmup_count: int,
        task_records: tuple[WarmupTaskRecord, ...],
        final_artifact_bank: tuple[DiffusionArtifact, ...],
        archive_manifest: ArchiveManifest,
        provenance: RuntimeProvenance,
    ) -> WarmupBundle:
        semantic_payload = {
            "schema_version": 1,
            "sequence_id": sequence_id,
            "warmup_count": warmup_count,
            "task_records": [record.model_dump(mode="json") for record in task_records],
            "final_artifact_bank": [
                artifact.model_dump(mode="json") for artifact in final_artifact_bank
            ],
        }
        return cls(
            bundle_id=_canonical_sha256(semantic_payload),
            sequence_id=sequence_id,
            warmup_run_id=warmup_run_id,
            warmup_count=warmup_count,
            task_records=task_records,
            final_artifact_bank=final_artifact_bank,
            archive_manifest=archive_manifest,
            provenance=provenance,
        )

    def reference(
        self,
        *,
        relative_path: str,
        manifest_path: str,
    ) -> WarmupReference:
        if self.warmup_count == 0:
            raise ValueError(
                "zero-length warm-up bundles are not referenced by samples"
            )
        return WarmupReference(
            bundle_id=self.bundle_id,
            warmup_run_id=self.warmup_run_id,
            relative_path=relative_path,
            manifest_path=manifest_path,
            sequence_id=self.sequence_id,
            warmup_count=self.warmup_count,
        )


class WarmupExecution(_FrozenV1Model):
    """Unarchived causal prefix returned by :class:`SampleRunner`."""

    sequence_id: str
    warmup_run_id: str
    task_records: tuple[WarmupTaskRecord, ...]
    final_artifact_bank: tuple[DiffusionArtifact, ...]
    completed_journal_paths: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        _validate_identity(self.sequence_id, label="sequence_id")
        _validate_identity(self.warmup_run_id, label="warmup_run_id")
        expected_ids: tuple[str, ...] = ()
        for position, record in enumerate(self.task_records):
            if record.position != position:
                raise ValueError("warm-up execution records must be an ordered prefix")
            if record.sequence_id != self.sequence_id:
                raise ValueError("warm-up execution record belongs to another sequence")
            if record.run_id != self.warmup_run_id:
                raise ValueError("warm-up execution record belongs to another run")
            if record.artifact_ids_before != expected_ids:
                raise ValueError("warm-up execution does not form one causal bank")
            expected_ids = record.bank_update.after_artifact_ids
        if (
            tuple(artifact.artifact_id for artifact in self.final_artifact_bank)
            != expected_ids
        ):
            raise ValueError("warm-up execution final bank is inconsistent")
        _validate_final_artifact_objects(
            task_records=self.task_records,
            final_artifact_bank=self.final_artifact_bank,
            label="warm-up execution",
        )
        _validate_journal_paths(
            self.completed_journal_paths,
            max_count=len(self.task_records),
        )
        return self


def _validate_journal_paths(paths: tuple[str, ...], *, max_count: int) -> None:
    if len(paths) != len(set(paths)) or len(paths) > max_count:
        raise ValueError("completed journal paths must be unique committed positions")
    for path in paths:
        _validate_relative_path(path, label="completed journal path")


def _validate_final_artifact_objects(
    *,
    task_records: tuple[WarmupTaskRecord, ...] | tuple[SampleTaskRecord, ...],
    final_artifact_bank: tuple[DiffusionArtifact, ...],
    label: str,
) -> None:
    final_by_id = {
        artifact.artifact_id: artifact for artifact in final_artifact_bank
    }
    for record in task_records:
        for artifact in record.bank_update.added_artifacts:
            if final_by_id.get(artifact.artifact_id) != artifact:
                raise ValueError(
                    f"final {label} bank artifact objects must match every transition"
                )


def _validate_sample_output(
    *,
    spec: SampleSpec,
    warmup_reference: WarmupReference | None,
    task_records: tuple[SampleTaskRecord, ...],
    rewards: SequenceRewards,
    final_artifact_bank: tuple[DiffusionArtifact, ...],
    final_graph: TaskGraphSnapshot | None,
) -> None:
    sequence = spec.sequence
    if sequence.warmup_count:
        if warmup_reference is None:
            raise ValueError("sample output requires its shared warm-up reference")
        if warmup_reference.bundle_id != spec.warmup_bundle_id:
            raise ValueError("warm-up reference does not match SampleSpec")
        if warmup_reference.sequence_id != sequence.sequence_id:
            raise ValueError("warm-up reference belongs to another sequence")
        if warmup_reference.warmup_count != sequence.warmup_count:
            raise ValueError("warm-up reference has the wrong prefix length")
    elif warmup_reference is not None:
        raise ValueError("zero-warm-up sample cannot carry a warm-up reference")

    if tuple(record.position for record in task_records) != sequence.suffix_positions:
        raise ValueError("sample task records must contain only the complete suffix")
    for position, record in zip(
        sequence.suffix_positions,
        task_records,
        strict=True,
    ):
        if record.sample_id != spec.sample_id:
            raise ValueError("sample task record belongs to another sample")
        if record.sequence_id != sequence.sequence_id:
            raise ValueError("sample task record belongs to another sequence")
        if record.task != sequence.tasks[position]:
            raise ValueError("sample task record differs from the frozen occurrence")
        if record.arm is not spec.arm:
            raise ValueError("sample task record belongs to another arm")

    expected_rewards = calculate_sequence_rewards(
        sequence=sequence,
        task_rewards=tuple(record.execution.reward for record in task_records),
        completed_positions=tuple(record.position for record in task_records),
    )
    if rewards != expected_rewards:
        raise ValueError("sequence rewards differ from the frozen suffix records")
    if not rewards.all_tasks_completed:
        raise ValueError("a successful sample output must complete every suffix task")

    final_ids = tuple(artifact.artifact_id for artifact in final_artifact_bank)
    if len(final_ids) != len(set(final_ids)):
        raise ValueError("final sample bank artifact IDs must be unique")
    expected_before = (
        () if sequence.warmup_count == 0 else task_records[0].artifact_ids_before
    )
    for record in task_records:
        if record.artifact_ids_before != expected_before:
            raise ValueError("sample task records do not form one causal bank chain")
        expected_before = record.bank_update.after_artifact_ids
    if final_ids != expected_before:
        raise ValueError("final sample bank does not match its causal bank chain")

    _validate_final_artifact_objects(
        task_records=task_records,
        final_artifact_bank=final_artifact_bank,
        label="sample",
    )
    if spec.arm is OrchestrationArm.FULL_ORCHESTRATION:
        if final_graph is None:
            raise ValueError("full_orchestration requires a final graph")
        if final_graph.run_id != spec.sample_id:
            raise ValueError("final graph belongs to another sample")
        final_record = task_records[-1]
        if final_graph.iteration != final_record.position:
            raise ValueError("final graph must represent the last suffix position")
        if final_graph.snapshot_id != final_record.graph_snapshot_id_after:
            raise ValueError("final graph must match the last task record")
    elif final_graph is not None:
        raise ValueError("only full_orchestration may produce a final graph")


class SampleExecution(_FrozenV1Model):
    """Unarchived arm-specific suffix returned by :class:`SampleRunner`."""

    spec: SampleSpec
    warmup_reference: WarmupReference | None
    task_records: tuple[SampleTaskRecord, ...]
    rewards: SequenceRewards
    final_artifact_bank: tuple[DiffusionArtifact, ...]
    final_graph: TaskGraphSnapshot | None
    completed_journal_paths: tuple[str, ...] = ()

    @model_validator(mode="after")
    def validate_execution(self) -> Self:
        _validate_sample_output(
            spec=self.spec,
            warmup_reference=self.warmup_reference,
            task_records=self.task_records,
            rewards=self.rewards,
            final_artifact_bank=self.final_artifact_bank,
            final_graph=self.final_graph,
        )
        _validate_journal_paths(
            self.completed_journal_paths,
            max_count=len(self.task_records),
        )
        return self


class SampleResult(_FrozenV1Model):
    """Durable suffix result; combine with its warm-up reference for a sample."""

    spec: SampleSpec
    warmup_reference: WarmupReference | None
    task_records: tuple[SampleTaskRecord, ...]
    rewards: SequenceRewards
    final_artifact_bank: tuple[DiffusionArtifact, ...]
    final_graph: TaskGraphSnapshot | None
    archive_manifest: ArchiveManifest
    provenance: RuntimeProvenance

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        _validate_sample_output(
            spec=self.spec,
            warmup_reference=self.warmup_reference,
            task_records=self.task_records,
            rewards=self.rewards,
            final_artifact_bank=self.final_artifact_bank,
            final_graph=self.final_graph,
        )
        return self


class FailureStage(str, Enum):
    RESOLVE = "resolve"
    GRAPH = "graph"
    POLICY = "policy"
    PACK = "pack"
    EXECUTE = "execute"
    PROJECT = "project"
    PERSIST = "persist"
    FINALIZE = "finalize"


class PositionJournal(_FrozenV1Model):
    """Immutable post-commit record for exactly one sequence position."""

    run_id: str
    sequence_id: str
    sample_id: str | None = None
    position: int = Field(ge=0)
    task_record: TaskRecord
    bank_artifact_ids: tuple[str, ...]
    graph_snapshot_id: str | None = None

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        record_run_id = self.task_record.run_id
        if self.run_id != record_run_id:
            raise ValueError("journal run_id must match its task record")
        if self.sequence_id != self.task_record.sequence_id:
            raise ValueError("journal sequence_id must match its task record")
        if self.position != self.task_record.position:
            raise ValueError("journal position must match its task record")
        if self.bank_artifact_ids != self.task_record.bank_update.after_artifact_ids:
            raise ValueError("journal bank IDs must match the committed transition")
        if isinstance(self.task_record, WarmupTaskRecord):
            if self.sample_id is not None or self.graph_snapshot_id is not None:
                raise ValueError("warm-up journals are arm- and graph-neutral")
        else:
            if self.sample_id != self.task_record.sample_id:
                raise ValueError("suffix journal sample_id must match its task record")
            if self.graph_snapshot_id != self.task_record.graph_snapshot_id_after:
                raise ValueError("journal graph snapshot must match its task record")
        return self


class RunProgress(_FrozenV1Model):
    """Last fully committed state attached to a staged run failure."""

    run_id: str
    sequence_id: str
    sample_id: str | None = None
    completed_positions: tuple[int, ...] = ()
    completed_journal_paths: tuple[str, ...] = ()
    bank_artifact_ids: tuple[str, ...] = ()
    graph_snapshot_id: str | None = None


def sanitize_failure_message(message: str, *, limit: int = 1_000) -> str:
    """Remove common credential shapes and collapse exception context."""
    collapsed = redact_sensitive_text(" ".join(message.split()))
    return collapsed[:limit] or "unspecified failure"


class SampleRunError(RuntimeError):
    """Failure at one named lifecycle stage with last committed progress."""

    def __init__(
        self,
        *,
        stage: FailureStage,
        position: int,
        task_id: str,
        progress: RunProgress,
        cause: BaseException,
    ) -> None:
        self.stage = stage
        self.position = position
        self.task_id = task_id
        self.progress = progress
        self.cause = cause
        detail = sanitize_failure_message(str(cause))
        super().__init__(
            f"sample run failed at stage={stage.value} position={position} "
            f"task_id={task_id}: {detail}"
        )


class FailureRecord(_FrozenV1Model):
    """Sanitized terminal failure persisted instead of a success result."""

    run_id: str
    sequence_id: str
    sample_id: str | None = None
    stage: FailureStage
    position: int = Field(ge=0)
    task_id: str
    completed_journal_paths: tuple[str, ...] = ()
    bank_artifact_ids: tuple[str, ...] = ()
    graph_snapshot_id: str | None = None
    exception_type: str
    message: str = Field(max_length=1_000)
    provenance: RuntimeProvenance

    @classmethod
    def from_error(
        cls,
        error: SampleRunError,
        *,
        provenance: RuntimeProvenance,
    ) -> FailureRecord:
        progress = error.progress
        return cls(
            run_id=progress.run_id,
            sequence_id=progress.sequence_id,
            sample_id=progress.sample_id,
            stage=error.stage,
            position=error.position,
            task_id=error.task_id,
            completed_journal_paths=progress.completed_journal_paths,
            bank_artifact_ids=progress.bank_artifact_ids,
            graph_snapshot_id=progress.graph_snapshot_id,
            exception_type=type(error.cause).__name__,
            message=sanitize_failure_message(str(error.cause)),
            provenance=provenance,
        )
