"""Typed requests and results at the task-execution boundary."""

from __future__ import annotations

import json
import os
import re
from pathlib import PurePosixPath
from typing import Any, Literal, Self
from urllib.parse import parse_qsl, unquote, urlsplit, urlunsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from mediated_coevo.models.iteration import IterationRecord

SchemaVersion = Literal[1]
SamplePhaseName = Literal["warmup", "orchestrated"]
ExecutionArmName = Literal[
    "execution_only",
    "random_policy",
    "no_graph",
    "full_orchestration",
]

_QUOTED_ASSIGNMENT_PATTERN = re.compile(
    r"(?P<key_quote>[\"'])(?P<key>[A-Za-z][A-Za-z0-9_-]*)"
    r"(?P=key_quote)(?P<separator>\s*:\s*)"
    r"(?P<value_quote>[\"'])(?P<value>.*?)(?P=value_quote)",
)
_PLAIN_ASSIGNMENT_PATTERN = re.compile(
    r"(?P<prefix>\b(?P<key>[A-Za-z][A-Za-z0-9_-]*)\s*[:=]\s*)"
    r"(?:(?P<bearer>Bearer)\s+)?"
    r"(?P<value>\"[^\"]*\"|'[^']*'|\[redacted\]|[^\s,;\]\}]+)",
    re.IGNORECASE,
)
_BEARER_PATTERN = re.compile(
    r"\bBearer\s+(?:\[redacted\]|[^\s,;\]\}]+)",
    re.IGNORECASE,
)
_URI_PATTERN = re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s]+", re.IGNORECASE)


def is_sensitive_key(value: object) -> bool:
    """Return whether a mapping key conventionally carries a credential."""
    if not isinstance(value, str):
        return False
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return (
        "api_key" in normalized
        or "apikey" in normalized
        or "password" in normalized
        or "passwd" in normalized
        or "secret" in normalized
        or "credential" in normalized
        or "authorization" in normalized
        or "access_key" in normalized
        or "private_key" in normalized
        or normalized == "token"
        or normalized.endswith("_token")
    )


def redact_sensitive_text(value: str) -> str:
    """Remove common credential assignments from persisted free-form text."""
    def redact_quoted(match: re.Match[str]) -> str:
        if not is_sensitive_key(match.group("key")):
            return match.group(0)
        quote = match.group("key_quote")
        value_quote = match.group("value_quote")
        return (
            f"{quote}{match.group('key')}{quote}"
            f"{match.group('separator')}{value_quote}[redacted]{value_quote}"
        )

    def redact_plain(match: re.Match[str]) -> str:
        if not is_sensitive_key(match.group("key")):
            return match.group(0)
        return f"{match.group('prefix')}[redacted]"

    def sanitize_uri(match: re.Match[str]) -> str:
        raw = match.group(0)
        trailing = ""
        while raw and raw[-1] in ".,;)]}":
            trailing = raw[-1] + trailing
            raw = raw[:-1]
        try:
            parsed = urlsplit(raw)
            hostname = parsed.hostname or ""
            netloc = hostname
            if parsed.port is not None:
                netloc = f"{netloc}:{parsed.port}"
            sanitized = urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
        except ValueError:
            sanitized = "[redacted-uri]"
        return sanitized + trailing

    value = _QUOTED_ASSIGNMENT_PATTERN.sub(redact_quoted, value)
    value = _PLAIN_ASSIGNMENT_PATTERN.sub(redact_plain, value)
    value = _BEARER_PATTERN.sub("Bearer [redacted]", value)
    value = _URI_PATTERN.sub(sanitize_uri, value)
    for secret in _environment_secret_values():
        value = value.replace(secret, "[redacted]")
    return value


def redact_sensitive_data(value: Any) -> Any:
    """Recursively redact credential fields while preserving data shape."""
    if isinstance(value, dict):
        return {
            key: (
                "[redacted]"
                if is_sensitive_key(key)
                else redact_sensitive_data(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item) for item in value)
    if isinstance(value, str):
        return redact_sensitive_text(value)
    return value


def contains_sensitive_text(value: str) -> bool:
    """Detect credential shapes without rejecting benign URL parameters."""
    if any(
        is_sensitive_key(match.group("key"))
        for match in _QUOTED_ASSIGNMENT_PATTERN.finditer(value)
    ):
        return True
    if any(
        is_sensitive_key(match.group("key"))
        for match in _PLAIN_ASSIGNMENT_PATTERN.finditer(value)
    ):
        return True
    if _BEARER_PATTERN.search(value):
        return True
    for match in _URI_PATTERN.finditer(value):
        try:
            parsed = urlsplit(match.group(0).rstrip(".,;)]}"))
        except ValueError:
            continue
        if parsed.username or parsed.password:
            return True
        if any(is_sensitive_key(key) for key, _ in parse_qsl(parsed.query)):
            return True
    return any(secret in value for secret in _environment_secret_values())


def _environment_secret_values() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                value
                for key, value in os.environ.items()
                if is_sensitive_key(key) and len(value) >= 4
            },
            key=len,
            reverse=True,
        )
    )


def _find_sensitive_key(value: Any, *, prefix: str = "task_config") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key)
            current = f"{prefix}.{key_text}"
            if is_sensitive_key(key_text):
                return current
            nested = _find_sensitive_key(item, prefix=current)
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            nested = _find_sensitive_key(item, prefix=f"{prefix}[{index}]")
            if nested is not None:
                return nested
    elif isinstance(value, str) and contains_sensitive_text(value):
        return prefix
    return None


def _find_sensitive_value(value: Any, *, prefix: str = "task_config") -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            nested = _find_sensitive_value(item, prefix=f"{prefix}.{key}")
            if nested is not None:
                return nested
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            nested = _find_sensitive_value(item, prefix=f"{prefix}[{index}]")
            if nested is not None:
                return nested
    elif isinstance(value, str) and contains_sensitive_text(value):
        return prefix
    return None


def _normalize_external_archive_refs(value: Any) -> tuple[dict[str, Any], ...]:
    """Validate external provenance before it can enter a durable journal."""
    if value in (None, (), []):
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError("external_archive_refs must be a list or tuple")
    normalized: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("external_archive_refs entries must be objects")
        if set(item).difference({"kind", "uri", "provenance"}):
            raise ValueError("external_archive_refs entries contain unknown fields")
        kind = item.get("kind")
        uri = item.get("uri")
        if not isinstance(kind, str) or not kind or kind != kind.strip():
            raise ValueError("external archive reference kind must be non-empty")
        if not isinstance(uri, str) or not uri or uri != uri.strip():
            raise ValueError("external archive reference uri must be non-empty")
        if len(uri) > 2048 or "\\" in uri or "\x00" in uri:
            raise ValueError("external archive reference uri is not portable")
        if redact_sensitive_text(uri) != uri:
            raise ValueError("external archive reference uri contains credentials")
        parsed = urlsplit(uri)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError("external archive reference uri contains credentials")
        if not parsed.scheme:
            if not PurePosixPath(uri).is_absolute():
                raise ValueError(
                    "external archive reference uri must be absolute or have a scheme"
                )
        elif parsed.scheme == "file":
            if parsed.netloc or not PurePosixPath(unquote(parsed.path)).is_absolute():
                raise ValueError(
                    "file external archive reference uri must contain an absolute path"
                )
        else:
            if not parsed.netloc or not parsed.hostname:
                raise ValueError(
                    "remote external archive reference uri must include an authority"
                )
            try:
                parsed.port
            except ValueError as exc:
                raise ValueError(
                    "external archive reference uri contains an invalid port"
                ) from exc
        provenance = item.get("provenance", {})
        if not isinstance(provenance, dict):
            raise ValueError("external archive reference provenance must be an object")
        normalized.append(
            {
                "kind": kind,
                "uri": uri,
                "provenance": redact_sensitive_data(provenance),
            }
        )
    return tuple(normalized)


class FrozenJsonDict(dict[str, Any]):
    """JSON-object mapping that rejects mutation after normalization."""

    @staticmethod
    def _immutable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("normalized JSON objects are immutable")

    __setitem__ = _immutable  # type: ignore[assignment]
    __delitem__ = _immutable  # type: ignore[assignment]
    clear = _immutable  # type: ignore[assignment]
    pop = _immutable  # type: ignore[assignment]
    popitem = _immutable  # type: ignore[assignment]
    setdefault = _immutable  # type: ignore[assignment]
    update = _immutable  # type: ignore[assignment]
    __ior__ = _immutable  # type: ignore[assignment]


def _freeze_json(value: Any) -> Any:
    if isinstance(value, dict):
        return FrozenJsonDict({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _normalized_json_object(value: Any, *, label: str) -> FrozenJsonDict:
    """Return a detached, deterministically ordered JSON object."""
    if hasattr(value, "model_dump"):
        value = value.model_dump(mode="json")
    try:
        encoded = json.dumps(
            value or {},
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        normalized = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain only JSON-compatible values") from exc
    if not isinstance(normalized, dict):
        raise ValueError(f"{label} must be a JSON object")
    return _freeze_json(normalized)


def normalized_json_object(
    value: Any,
    *,
    label: str,
    redact: bool = False,
) -> FrozenJsonDict:
    """Expose canonical immutable JSON normalization to PR2 contracts."""
    if redact:
        value = redact_sensitive_data(value)
    return _normalized_json_object(value, label=label)


class TaskProfile(BaseModel):
    """Frozen, normalized description of one benchmark task occurrence."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        revalidate_instances="always",
    )

    schema_version: SchemaVersion = 1
    task_id: str = Field(min_length=1)
    instruction: str
    task_config: dict[str, Any] = Field(default_factory=FrozenJsonDict)

    @field_validator("task_id")
    @classmethod
    def validate_task_id(cls, value: str) -> str:
        """Allow family/task IDs while rejecting traversal and ambiguity."""
        path = PurePosixPath(value)
        if (
            not value
            or value != value.strip()
            or path.is_absolute()
            or path.as_posix() != value
            or any(part in {"", ".", ".."} for part in path.parts)
            or "\\" in value
            or "\x00" in value
        ):
            raise ValueError("task_id must be a safe normalized relative task ID")
        return value

    @field_validator("instruction")
    @classmethod
    def reject_instruction_credentials(cls, value: str) -> str:
        if contains_sensitive_text(value) or any(
            secret in value for secret in _environment_secret_values()
        ):
            raise ValueError("instruction must not contain credentials")
        return value

    @field_validator("task_config")
    @classmethod
    def normalize_task_config(cls, value: Any) -> FrozenJsonDict:
        """Detach repository-owned configuration into canonical JSON data."""
        normalized = _normalized_json_object(value, label="task_config")
        sensitive_path = _find_sensitive_key(normalized)
        if sensitive_path is None:
            sensitive_path = _find_sensitive_value(normalized)
        if sensitive_path is not None:
            raise ValueError(
                f"task_config must not contain credentials ({sensitive_path})"
            )
        return normalized

    def normalized_metadata(self) -> FrozenJsonDict:
        """Return flat metadata understood by transfer-artifact projection."""
        raw_metadata = self.task_config.get("metadata")
        metadata = raw_metadata if isinstance(raw_metadata, dict) else {}
        raw_verifier = self.task_config.get("verifier")
        verifier = raw_verifier if isinstance(raw_verifier, dict) else {}
        normalized: dict[str, Any] = {}
        for source, target in (
            ("category", "task_category"),
            ("difficulty", "task_difficulty"),
            ("expected_reward_range", "expected_reward_range"),
        ):
            if metadata.get(source) is not None:
                normalized[target] = metadata[source]
        if verifier.get("type") is not None:
            normalized["verifier_type"] = verifier["type"]
        return _normalized_json_object(normalized, label="normalized task metadata")


class ContextPack(BaseModel):
    """Complete, auditable orchestration input supplied to execution."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        revalidate_instances="always",
    )

    schema_version: SchemaVersion = 1
    text: str | None = None
    eligible_artifact_ids: tuple[str, ...] = ()
    selected_artifact_ids: tuple[str, ...] = ()
    rendered_artifact_ids: tuple[str, ...] = ()
    source_task_ids: tuple[str, ...] = ()
    snapshot_id: str | None = None
    policy_name: str = "none"
    token_count: int = Field(default=0, ge=0)
    max_context_tokens: int | None = Field(default=None, ge=0)
    compacted_artifact_ids: tuple[str, ...] = ()
    dropped_for_budget_artifact_ids: tuple[str, ...] = ()
    budget_violation: bool = False
    metadata: dict[str, Any] = Field(default_factory=FrozenJsonDict)

    @field_validator("text")
    @classmethod
    def redact_context_text(cls, value: str | None) -> str | None:
        return redact_sensitive_text(value) if value is not None else None

    @field_validator("policy_name")
    @classmethod
    def validate_policy_name(cls, value: str) -> str:
        """Keep policy identity explicit and stable."""
        if not value or value != value.strip():
            raise ValueError("policy_name must be a non-empty stripped string")
        return value

    @field_validator("metadata")
    @classmethod
    def normalize_metadata(cls, value: Any) -> FrozenJsonDict:
        """Keep context metadata serializable and detached from agent output."""
        return normalized_json_object(
            value,
            label="context metadata",
            redact=True,
        )

    @model_validator(mode="after")
    def validate_consistency(self) -> Self:
        """Enforce ID, rendering, graph, policy, and budget consistency."""
        id_groups = (
            ("eligible", self.eligible_artifact_ids),
            ("selected", self.selected_artifact_ids),
            ("rendered", self.rendered_artifact_ids),
            ("compacted", self.compacted_artifact_ids),
            ("dropped", self.dropped_for_budget_artifact_ids),
            ("source task", self.source_task_ids),
        )
        for label, values in id_groups:
            if any(not value or value != value.strip() for value in values):
                raise ValueError(f"{label} IDs must be non-empty stripped strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{label} IDs must be unique")

        eligible = set(self.eligible_artifact_ids)
        selected = set(self.selected_artifact_ids)
        rendered = set(self.rendered_artifact_ids)
        compacted = set(self.compacted_artifact_ids)
        dropped = set(self.dropped_for_budget_artifact_ids)
        if not selected.issubset(eligible):
            raise ValueError("selected artifacts must be eligible")
        if not rendered.issubset(selected):
            raise ValueError("rendered artifacts must be selected")
        if not compacted.issubset(rendered):
            raise ValueError("compacted artifacts must be rendered")
        if not dropped.issubset(selected):
            raise ValueError("budget-dropped artifacts must be selected")
        if rendered.intersection(dropped):
            raise ValueError("an artifact cannot be both rendered and budget-dropped")
        if selected != rendered.union(dropped):
            raise ValueError(
                "every selected artifact must be rendered or explicitly budget-dropped"
            )

        if self.text is not None and not self.text.strip():
            raise ValueError("context text must be non-blank when present")
        if self.text is None:
            if rendered or self.source_task_ids or self.token_count:
                raise ValueError(
                    "absent context text requires no rendered IDs, sources, or tokens"
                )
        elif not rendered or not self.source_task_ids or self.token_count == 0:
            raise ValueError(
                "context text requires rendered artifacts, sources, and token_count"
            )

        if self.snapshot_id is not None and (
            not self.snapshot_id or self.snapshot_id != self.snapshot_id.strip()
        ):
            raise ValueError("snapshot_id must be a non-empty stripped string")

        if dropped and not self.budget_violation:
            raise ValueError("budget-dropped artifacts require budget_violation=true")
        exceeds_budget = (
            self.max_context_tokens is not None
            and self.token_count > self.max_context_tokens
        )
        if exceeds_budget and not self.budget_violation:
            raise ValueError("context exceeding its token budget must flag a violation")
        if self.budget_violation and not (dropped or exceeds_budget):
            raise ValueError("budget_violation requires a dropped or over-budget context")

        for key, expected in (
            ("eligible_count", len(self.eligible_artifact_ids)),
            ("selected_count", len(self.selected_artifact_ids)),
            ("rendered_count", len(self.rendered_artifact_ids)),
        ):
            actual = self.metadata.get(key)
            if actual is not None and actual != expected:
                raise ValueError(f"context metadata {key} does not match artifact IDs")

        if self.policy_name == "none" and any(
            (
                self.text is not None,
                bool(self.eligible_artifact_ids),
                bool(self.selected_artifact_ids),
                bool(self.rendered_artifact_ids),
                bool(self.source_task_ids),
                self.snapshot_id is not None,
                self.token_count != 0,
                self.max_context_tokens is not None,
                bool(self.compacted_artifact_ids),
                bool(self.dropped_for_budget_artifact_ids),
                self.budget_violation,
                bool(self.metadata),
            )
        ):
            raise ValueError("policy 'none' is reserved for the canonical empty context")
        if self.policy_name == "random_uniform" and self.snapshot_id is not None:
            raise ValueError("random policy context cannot claim a graph snapshot")
        return self


def empty_context_pack() -> ContextPack:
    """Return the canonical no-orchestration executor input."""
    return ContextPack()


class TaskExecutionRequest(BaseModel):
    """The only cross-task input visible to a task-execution agent."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        revalidate_instances="always",
    )

    schema_version: SchemaVersion = 1
    run_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    phase: SamplePhaseName
    arm: ExecutionArmName | None = None
    task: TaskProfile
    context: ContextPack = Field(default_factory=empty_context_pack)

    @model_validator(mode="after")
    def validate_phase_boundary(self) -> Self:
        """Keep warm-up arm-neutral and orchestration inputs explicit."""
        if self.run_id != self.run_id.strip():
            raise ValueError("run_id must not have surrounding whitespace")
        if self.phase == "warmup":
            if self.arm is not None:
                raise ValueError("warm-up execution must not carry a treatment arm")
            if self.context != empty_context_pack():
                raise ValueError("warm-up execution requires the canonical empty context")
        elif self.arm is None:
            raise ValueError("orchestrated execution requires a treatment arm")
        if self.arm == "execution_only" and self.context != empty_context_pack():
            raise ValueError("execution-only execution requires an empty context")
        return self


class TaskExecutionResult(BaseModel):
    """Complete task record plus portable references to its run archive."""

    model_config = ConfigDict(
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
        revalidate_instances="always",
    )

    schema_version: SchemaVersion = 1
    run_id: str = Field(min_length=1)
    position: int = Field(ge=0)
    task_id: str = Field(min_length=1)
    record: IterationRecord
    archive_paths: tuple[str, ...] = ()
    metadata: dict[str, Any] = Field(default_factory=FrozenJsonDict)

    @field_validator("record", mode="before")
    @classmethod
    def detach_and_redact_record(cls, value: Any) -> IterationRecord:
        """Sanitize arbitrary executor output before it can enter a journal."""
        payload = (
            value.model_dump(mode="python")
            if isinstance(value, IterationRecord)
            else value
        )
        return IterationRecord.model_validate(redact_sensitive_data(payload))

    @field_validator("archive_paths")
    @classmethod
    def validate_archive_paths(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        """Require stable workspace-relative references in the portable result."""
        if len(values) != len(set(values)):
            raise ValueError("archive_paths must be unique")
        for value in values:
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
                raise ValueError(
                    "archive_paths must be normalized non-empty relative paths"
                )
        return values

    @field_validator("metadata")
    @classmethod
    def normalize_metadata(cls, value: Any) -> FrozenJsonDict:
        """Detach executor provenance into safe JSON data."""
        if not isinstance(value, dict):
            raise ValueError("execution metadata must be a JSON object")
        value = redact_sensitive_data(value)
        if "external_archive_refs" in value:
            value["external_archive_refs"] = _normalize_external_archive_refs(
                value["external_archive_refs"]
            )
        return normalized_json_object(
            value,
            label="execution metadata",
            redact=True,
        )

    @model_validator(mode="after")
    def validate_record_identity(self) -> Self:
        """Prevent an executor from returning a different task occurrence."""
        if self.run_id != self.run_id.strip():
            raise ValueError("run_id must not have surrounding whitespace")
        if self.task_id != self.task_id.strip():
            raise ValueError("task_id must not have surrounding whitespace")
        if self.record.iteration != self.position:
            raise ValueError("execution record iteration must equal result position")
        if self.record.task_id != self.task_id:
            raise ValueError("execution record task_id must equal result task_id")
        trace = self.record.execution_trace
        if trace is not None:
            for value in trace.harbor_paths.values():
                path = PurePosixPath(value)
                if (
                    not value
                    or value != value.strip()
                    or path.is_absolute()
                    or path == PurePosixPath(".")
                    or ".." in path.parts
                    or path.as_posix() != value
                ):
                    raise ValueError(
                        "execution trace Harbor paths must be normalized and relative"
                    )
            if not set(trace.harbor_paths.values()).issubset(self.archive_paths):
                raise ValueError(
                    "execution trace Harbor paths must be declared in archive_paths"
                )
        return self

    @property
    def reward(self) -> float | None:
        """Expose a legitimate zero reward without coercing it to missing."""
        return self.record.reward

    @property
    def is_infrastructure_failure(self) -> bool:
        """Return whether the underlying run failed before task evaluation."""
        trace = self.record.execution_trace
        return trace is not None and trace.status in {
            "env_failure",
            "parse_error",
            "harbor_failed",
        }
