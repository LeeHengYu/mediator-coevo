"""Typed, independently injectable orchestration-agent contracts."""

from __future__ import annotations

from typing import Any, Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from mediated_coevo.diffusion.models import DiffusionArtifact, TaskGraphSnapshot
from mediated_coevo.diffusion.policy import DiffusionSubscription
from mediated_coevo.execution.models import (
    ContextPack,
    FrozenJsonDict,
    TaskProfile,
    normalized_json_object,
    redact_sensitive_data,
)


def _json_mapping(value: Any, *, label: str) -> FrozenJsonDict:
    return normalized_json_object(value, label=label, redact=True)


def _validate_causal_artifacts(
    *, artifacts: tuple[DiffusionArtifact, ...], position: int
) -> None:
    artifact_ids = tuple(artifact.artifact_id for artifact in artifacts)
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("causal artifact IDs must be unique")
    if any(artifact.source_iteration >= position for artifact in artifacts):
        raise ValueError("orchestration input contains a current or future artifact")


def _copy_artifact(value: DiffusionArtifact | dict[str, Any]) -> DiffusionArtifact:
    """Detach a mutable legacy artifact at a component boundary."""
    payload = (
        value.model_dump(mode="python")
        if isinstance(value, DiffusionArtifact)
        else value
    )
    return DiffusionArtifact.model_validate(redact_sensitive_data(payload))


def _copy_snapshot(
    value: TaskGraphSnapshot | dict[str, Any] | None,
) -> TaskGraphSnapshot | None:
    """Detach a mutable legacy graph snapshot at a component boundary."""
    if value is None:
        return None
    payload = (
        value.model_dump(mode="python")
        if isinstance(value, TaskGraphSnapshot)
        else value
    )
    return TaskGraphSnapshot.model_validate(redact_sensitive_data(payload))


def _copy_subscription(value: Any) -> DiffusionSubscription:
    """Detach a legacy subscription, including its artifact and metadata."""
    if isinstance(value, DiffusionSubscription):
        artifact = value.artifact
        policy_name = value.policy_name
        relation = value.relation
        reason = value.reason
        metadata = value.metadata
        context_channel = value.context_channel
    elif isinstance(value, dict):
        artifact = value["artifact"]
        policy_name = value["policy_name"]
        relation = value["relation"]
        reason = value["reason"]
        metadata = value.get("metadata", {})
        context_channel = value.get("context_channel", "reuse_success")
    else:
        return value
    return DiffusionSubscription(
        artifact=_copy_artifact(artifact),
        policy_name=str(policy_name),
        relation=str(relation),
        reason=str(redact_sensitive_data(str(reason))),
        metadata=_json_mapping(metadata, label="subscription metadata"),
        context_channel=str(context_channel),
    )


class GraphAgentRequest(BaseModel):
    """Complete causal input to one graph update."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    task: TaskProfile
    previous_graph: TaskGraphSnapshot | None = None
    artifacts: tuple[DiffusionArtifact, ...] = ()

    @field_validator("previous_graph", mode="before")
    @classmethod
    def detach_previous_graph(cls, value: Any) -> TaskGraphSnapshot | None:
        return _copy_snapshot(value)

    @field_validator("artifacts", mode="before")
    @classmethod
    def detach_artifacts(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return value
        return tuple(_copy_artifact(item) for item in value)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        """Reject cross-run, non-causal, or non-monotone graph input."""
        if self.run_id != self.run_id.strip():
            raise ValueError("run_id must not have surrounding whitespace")
        _validate_causal_artifacts(artifacts=self.artifacts, position=self.position)
        if self.previous_graph is not None:
            if self.previous_graph.run_id != self.run_id:
                raise ValueError("previous graph belongs to a different run")
            if self.previous_graph.iteration >= self.position:
                raise ValueError("previous graph must precede the current position")
        return self


class GraphAgentResponse(BaseModel):
    """Validated graph state plus raw agent output for auditing."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    schema_version: Literal[1] = 1
    snapshot: TaskGraphSnapshot
    raw_decision: dict[str, Any]

    @field_validator("snapshot", mode="before")
    @classmethod
    def detach_snapshot(cls, value: Any) -> TaskGraphSnapshot | None:
        return _copy_snapshot(value)

    @field_validator("raw_decision")
    @classmethod
    def normalize_decision(cls, value: Any) -> FrozenJsonDict:
        return _json_mapping(value, label="graph raw_decision")


class PolicyAgentRequest(BaseModel):
    """Complete causal input to one graph-optional policy decision."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    schema_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    policy_seed: int
    task: TaskProfile
    graph: TaskGraphSnapshot | None = None
    artifacts: tuple[DiffusionArtifact, ...] = ()

    @field_validator("graph", mode="before")
    @classmethod
    def detach_graph(cls, value: Any) -> TaskGraphSnapshot | None:
        return _copy_snapshot(value)

    @field_validator("artifacts", mode="before")
    @classmethod
    def detach_artifacts(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return value
        return tuple(_copy_artifact(item) for item in value)

    @model_validator(mode="after")
    def validate_request(self) -> Self:
        """Reject cross-run graphs and non-causal candidate artifacts."""
        if self.run_id != self.run_id.strip():
            raise ValueError("run_id must not have surrounding whitespace")
        _validate_causal_artifacts(artifacts=self.artifacts, position=self.position)
        if self.graph is not None:
            if self.graph.run_id != self.run_id:
                raise ValueError("policy graph belongs to a different run")
            if self.graph.iteration != self.position:
                raise ValueError("policy graph must be the current-position snapshot")
        return self


class PolicyAgentResponse(BaseModel):
    """Selected routes plus raw policy output and explicit treatment metadata."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    schema_version: Literal[1] = 1
    policy_name: str = Field(min_length=1)
    subscriptions: tuple[DiffusionSubscription, ...] = ()
    raw_decision: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("subscriptions", mode="before")
    @classmethod
    def detach_subscriptions(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return value
        return tuple(_copy_subscription(item) for item in value)

    @field_validator("raw_decision")
    @classmethod
    def normalize_decision(cls, value: Any) -> FrozenJsonDict:
        return _json_mapping(value, label="policy raw_decision")

    @field_validator("metadata")
    @classmethod
    def normalize_metadata(cls, value: Any) -> FrozenJsonDict:
        return _json_mapping(value, label="policy metadata")

    @model_validator(mode="after")
    def validate_selection(self) -> Self:
        """Keep one route per artifact and one explicit policy identity."""
        if self.policy_name != self.policy_name.strip():
            raise ValueError("policy_name must not have surrounding whitespace")
        artifact_ids = self.selected_artifact_ids
        if len(artifact_ids) != len(set(artifact_ids)):
            raise ValueError("policy cannot select an artifact more than once")
        if any(
            subscription.policy_name != self.policy_name
            for subscription in self.subscriptions
        ):
            raise ValueError("subscription policy_name must match its response")
        return self

    @property
    def selected_artifact_ids(self) -> tuple[str, ...]:
        """Return selected IDs in policy order."""
        return tuple(
            subscription.artifact.artifact_id
            for subscription in self.subscriptions
        )


class TaskGraphAgent(Protocol):
    """Update graph state before policy selection."""

    async def update(self, request: GraphAgentRequest) -> GraphAgentResponse: ...


class DiffusionPolicyAgent(Protocol):
    """Select artifacts, optionally using graph state."""

    async def select(self, request: PolicyAgentRequest) -> PolicyAgentResponse: ...


class ContextPacker(Protocol):
    """Pack selected routes into the executor's complete explicit context."""

    async def pack(
        self,
        *,
        run_id: str,
        position: int,
        task: TaskProfile,
        graph: TaskGraphSnapshot | None,
        policy: PolicyAgentResponse,
        eligible_artifacts: tuple[DiffusionArtifact, ...],
    ) -> ContextPack: ...
