"""Render diffusion subscriptions into planner context and audit records."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mediated_coevo.diffusion.models import (
    DiffusedRecord,
    DiffusionArtifact,
    TaskGraphSnapshot,
)
from mediated_coevo.diffusion.store import DiffusionStore
from mediated_coevo.prompt_text import PromptText
from mediated_coevo.runtime.token_budget import count_text_tokens

if TYPE_CHECKING:
    from mediated_coevo.diffusion.policy import DiffusionSubscription

DIFFUSED_SECTION_NAME = "Diffused Cross-Task Context"
DiffusionArtifactCompactor = Callable[[DiffusionArtifact, int], Awaitable[str]]


@dataclass(frozen=True)
class DiffusionContextBundle:
    """Rendered diffusion context plus observability fields."""

    text: str | None
    snapshot_id: str | None
    graph_policy: str
    eligible_count: int
    selected_count: int
    rendered_count: int
    context_tokens: int
    source_task_ids: list[str]
    max_context_tokens: int | None = None
    rendered_artifact_ids: list[str] | None = None
    compacted_artifact_ids: list[str] | None = None
    dropped_for_budget_artifact_ids: list[str] | None = None
    budget_violation: bool = False


async def render_diffusion_subscriptions(
    *,
    store: DiffusionStore,
    snapshot: TaskGraphSnapshot,
    model: str,
    target_task_id: str,
    target_iteration: int,
    target_run_id: str | None,
    subscriptions: list[DiffusionSubscription],
    eligible_count: int | None = None,
    max_context_tokens: int | None = None,
    compact_artifact_content: DiffusionArtifactCompactor | None = None,
) -> DiffusionContextBundle:
    """Render and audit the target's consumed diffusion subscriptions."""
    rendered_sections: list[str] = []
    rendered_subscriptions: list[DiffusionSubscription] = []
    rendered_artifact_ids: list[str] = []
    compacted_artifact_ids: list[str] = []
    dropped_artifact_ids: list[str] = []

    for subscription in subscriptions:
        rendered_section = _render_artifact_block(
            subscription.artifact,
            policy_name=subscription.policy_name,
            relation=subscription.relation,
        )
        if _sections_fit(
            model,
            [*rendered_sections, rendered_section],
            max_context_tokens,
        ):
            rendered_sections.append(rendered_section)
            rendered_subscriptions.append(subscription)
            rendered_artifact_ids.append(subscription.artifact.artifact_id)
            _append_rendered_record(
                store=store,
                snapshot=snapshot,
                model=model,
                target_task_id=target_task_id,
                target_iteration=target_iteration,
                target_run_id=target_run_id,
                subscription=subscription,
                rendered_section=rendered_section,
                metadata=None,
            )
            continue

        compacted_section = await _compacted_artifact_section(
            compact_artifact_content=compact_artifact_content,
            max_context_tokens=max_context_tokens,
            model=model,
            rendered_sections=rendered_sections,
            subscription=subscription,
        )
        if compacted_section is not None and _sections_fit(
            model,
            [*rendered_sections, compacted_section],
            max_context_tokens,
        ):
            rendered_sections.append(compacted_section)
            rendered_subscriptions.append(subscription)
            rendered_artifact_ids.append(subscription.artifact.artifact_id)
            compacted_artifact_ids.append(subscription.artifact.artifact_id)
            _append_rendered_record(
                store=store,
                snapshot=snapshot,
                model=model,
                target_task_id=target_task_id,
                target_iteration=target_iteration,
                target_run_id=target_run_id,
                subscription=subscription,
                rendered_section=compacted_section,
                metadata={"compacted_for_budget": True},
            )
            continue

        store.append_diffused_record(
            DiffusedRecord(
                artifact_id=subscription.artifact.artifact_id,
                source_task_id=subscription.artifact.source_task_id,
                source_iteration=subscription.artifact.source_iteration,
                source_run_id=subscription.artifact.source_run_id,
                target_task_id=target_task_id,
                target_iteration=target_iteration,
                target_run_id=target_run_id,
                snapshot_id=snapshot.snapshot_id,
                policy_name=subscription.policy_name,
                relation=subscription.relation,
                reason="dropped_for_diffusion_budget",
                eligible=True,
                selected=True,
                rendered=False,
                token_count=0,
                metadata=_record_metadata(
                    subscription.artifact,
                    {
                        **subscription.metadata,
                        "max_context_tokens": max_context_tokens,
                        "compaction_attempted": compact_artifact_content is not None,
                    },
                ),
            )
        )
        dropped_artifact_ids.append(subscription.artifact.artifact_id)

    text = None
    context_tokens = 0
    if rendered_sections:
        text = _context_text(rendered_sections)
        context_tokens = count_text_tokens(model, text)
    budget_violation = bool(dropped_artifact_ids) or (
        max_context_tokens is not None and context_tokens > max_context_tokens
    )

    return DiffusionContextBundle(
        text=text,
        snapshot_id=snapshot.snapshot_id,
        graph_policy=snapshot.graph_policy,
        eligible_count=(
            eligible_count if eligible_count is not None else len(subscriptions)
        ),
        selected_count=len(subscriptions),
        rendered_count=len(rendered_subscriptions),
        context_tokens=context_tokens,
        source_task_ids=list(
            dict.fromkeys(
                subscription.artifact.source_task_id
                for subscription in rendered_subscriptions
            )
        ),
        max_context_tokens=max_context_tokens,
        rendered_artifact_ids=rendered_artifact_ids,
        compacted_artifact_ids=compacted_artifact_ids,
        dropped_for_budget_artifact_ids=dropped_artifact_ids,
        budget_violation=budget_violation,
    )


def _append_rendered_record(
    *,
    store: DiffusionStore,
    snapshot: TaskGraphSnapshot,
    model: str,
    target_task_id: str,
    target_iteration: int,
    target_run_id: str | None,
    subscription: DiffusionSubscription,
    rendered_section: str,
    metadata: dict[str, Any] | None,
) -> None:
    token_count = count_text_tokens(model, rendered_section)
    store.append_diffused_record(
        DiffusedRecord(
            artifact_id=subscription.artifact.artifact_id,
            source_task_id=subscription.artifact.source_task_id,
            source_iteration=subscription.artifact.source_iteration,
            source_run_id=subscription.artifact.source_run_id,
            target_task_id=target_task_id,
            target_iteration=target_iteration,
            target_run_id=target_run_id,
            snapshot_id=snapshot.snapshot_id,
            policy_name=subscription.policy_name,
            relation=subscription.relation,
            reason=subscription.reason,
            eligible=True,
            selected=True,
            rendered=True,
            rendered_section=DIFFUSED_SECTION_NAME,
            token_count=token_count,
            citation_text=rendered_section,
            metadata=_record_metadata(
                subscription.artifact,
                {**subscription.metadata, **(metadata or {})},
            ),
        )
    )


async def _compacted_artifact_section(
    *,
    compact_artifact_content: DiffusionArtifactCompactor | None,
    max_context_tokens: int | None,
    model: str,
    rendered_sections: list[str],
    subscription: DiffusionSubscription,
) -> str | None:
    if compact_artifact_content is None or max_context_tokens is None:
        return None
    content_budget = _remaining_content_budget(
        model=model,
        rendered_sections=rendered_sections,
        subscription=subscription,
        max_context_tokens=max_context_tokens,
    )
    if content_budget <= 0:
        return None
    compacted_content = (
        await compact_artifact_content(subscription.artifact, content_budget)
    ).strip()
    if not compacted_content:
        return None
    return _render_artifact_block(
        subscription.artifact,
        policy_name=subscription.policy_name,
        relation=subscription.relation,
        content=compacted_content,
    )


def _remaining_content_budget(
    *,
    model: str,
    rendered_sections: list[str],
    subscription: DiffusionSubscription,
    max_context_tokens: int,
) -> int:
    current_tokens = count_text_tokens(model, _context_text(rendered_sections))
    empty_block_tokens = count_text_tokens(
        model,
        _render_artifact_block(
            subscription.artifact,
            policy_name=subscription.policy_name,
            relation=subscription.relation,
            content="",
        ),
    )
    return max(0, max_context_tokens - current_tokens - empty_block_tokens)


def _sections_fit(
    model: str,
    sections: list[str],
    max_context_tokens: int | None,
) -> bool:
    if max_context_tokens is None:
        return True
    if not sections:
        return True
    return count_text_tokens(model, _context_text(sections)) <= max_context_tokens


def _context_text(rendered_sections: list[str]) -> str:
    return PromptText.diffusion_context(DIFFUSED_SECTION_NAME, rendered_sections)


def _record_metadata(
    artifact: DiffusionArtifact,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record_metadata: dict[str, Any] = {
        "artifact_type": artifact.artifact_type.value,
        "risk_level": artifact.risk_level.value,
    }
    if metadata is not None:
        record_metadata.update(metadata)
    return record_metadata


def _render_artifact_block(
    artifact: DiffusionArtifact,
    *,
    policy_name: str,
    relation: str,
    content: str | None = None,
) -> str:
    return PromptText.diffusion_artifact_block(
        artifact_id=artifact.artifact_id,
        source_task_id=artifact.source_task_id,
        source_iteration=artifact.source_iteration,
        policy_name=policy_name,
        relation=relation,
        risk_level=artifact.risk_level.value,
        content=artifact.content if content is None else content,
    )
