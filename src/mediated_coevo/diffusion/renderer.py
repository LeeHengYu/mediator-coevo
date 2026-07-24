"""Render diffusion subscriptions into planner context and audit records."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from mediated_coevo.diffusion.models import (
    DiffusedRecord,
    DiffusionArtifact,
    TaskGraphSnapshot,
)
from mediated_coevo.diffusion.policy import (
    AVOID_RECHECK_CHANNEL,
    REUSE_SUCCESS_CHANNEL,
)
from mediated_coevo.diffusion.store import DiffusionStore
from mediated_coevo.prompt_text import PromptText
from mediated_coevo.runtime.token_budget import count_text_tokens

if TYPE_CHECKING:
    from mediated_coevo.diffusion.policy import DiffusionSubscription

DIFFUSED_SECTION_NAME = "Diffused Cross-Task Context"
REUSE_SECTION_NAME = "Reusable Success Artifacts"
AVOID_RECHECK_SECTION_NAME = "Avoid/Recheck Artifacts"
AVOID_RECHECK_WARNING = (
    "These artifacts came from failed source runs. Use them only to avoid or "
    "re-check failure modes; do not copy failed choices."
)
DiffusionArtifactCompactor = Callable[[DiffusionArtifact, int], Awaitable[str]]
logger = logging.getLogger(__name__)


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
    snapshot: TaskGraphSnapshot | None,
    model: str,
    target_task_id: str,
    target_iteration: int,
    target_run_id: str | None,
    subscriptions: list[DiffusionSubscription],
    graph_policy: str | None = None,
    eligible_count: int | None = None,
    max_context_tokens: int | None = None,
    compact_artifact_content: DiffusionArtifactCompactor | None = None,
) -> DiffusionContextBundle:
    """Render and audit the target's consumed diffusion subscriptions."""
    rendered_sections: list[tuple[str, str]] = []
    rendered_subscriptions: list[DiffusionSubscription] = []
    rendered_artifact_ids: list[str] = []
    compacted_artifact_ids: list[str] = []
    dropped_artifact_ids: list[str] = []

    compacted_selected_sections = await _compacted_selected_artifact_sections(
        compact_artifact_content=compact_artifact_content,
        max_context_tokens=max_context_tokens,
        model=model,
        subscriptions=subscriptions,
    )
    if compacted_selected_sections is not None:
        for subscription, rendered_section in zip(
            subscriptions,
            compacted_selected_sections,
            strict=True,
        ):
            rendered_sections.append((subscription.context_channel, rendered_section))
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
                rendered_section=rendered_section,
                metadata={"compacted_for_budget": True},
            )

    for subscription in subscriptions[len(rendered_subscriptions) :]:
        rendered_section = _render_artifact_block(
            subscription.artifact,
            policy_name=subscription.policy_name,
            relation=subscription.relation,
        )
        if _sections_fit(
            model,
            [*rendered_sections, (subscription.context_channel, rendered_section)],
            max_context_tokens,
        ):
            rendered_sections.append((subscription.context_channel, rendered_section))
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
            [*rendered_sections, (subscription.context_channel, compacted_section)],
            max_context_tokens,
        ):
            rendered_sections.append((subscription.context_channel, compacted_section))
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
                snapshot_id=snapshot.snapshot_id if snapshot is not None else None,
                policy_name=subscription.policy_name,
                relation=subscription.relation,
                reason="dropped_for_diffusion_budget",
                eligible=True,
                selected=True,
                rendered=False,
                token_count=0,
                verifier_reward=subscription.artifact.verifier_reward,
                judge_reward=subscription.artifact.judge_reward,
                success=_artifact_success(subscription.artifact),
                regression=_artifact_regression(subscription.artifact),
                metadata={
                    "artifact_type": subscription.artifact.artifact_type.value,
                    "risk_level": subscription.artifact.risk_level.value,
                    **subscription.metadata,
                    "diffusion_channel": subscription.context_channel,
                    "max_context_tokens": max_context_tokens,
                    "compaction_attempted": compact_artifact_content is not None,
                },
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
        snapshot_id=snapshot.snapshot_id if snapshot is not None else None,
        graph_policy=(
            snapshot.graph_policy
            if snapshot is not None
            else (graph_policy or "none")
        ),
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
    snapshot: TaskGraphSnapshot | None,
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
            snapshot_id=snapshot.snapshot_id if snapshot is not None else None,
            policy_name=subscription.policy_name,
            relation=subscription.relation,
            reason=subscription.reason,
            eligible=True,
            selected=True,
            rendered=True,
            rendered_section=DIFFUSED_SECTION_NAME,
            token_count=token_count,
            citation_text=rendered_section,
            verifier_reward=subscription.artifact.verifier_reward,
            judge_reward=subscription.artifact.judge_reward,
            success=_artifact_success(subscription.artifact),
            regression=_artifact_regression(subscription.artifact),
            metadata={
                "artifact_type": subscription.artifact.artifact_type.value,
                "risk_level": subscription.artifact.risk_level.value,
                **subscription.metadata,
                "diffusion_channel": subscription.context_channel,
                **(metadata or {}),
            },
        )
    )


async def _compacted_selected_artifact_sections(
    *,
    compact_artifact_content: DiffusionArtifactCompactor | None,
    max_context_tokens: int | None,
    model: str,
    subscriptions: list[DiffusionSubscription],
) -> list[str] | None:
    if (
        compact_artifact_content is None
        or max_context_tokens is None
        or len(subscriptions) < 2
    ):
        return None

    full_sections = [
        (
            subscription.context_channel,
            _render_artifact_block(
                subscription.artifact,
                policy_name=subscription.policy_name,
                relation=subscription.relation,
            ),
        )
        for subscription in subscriptions
    ]
    if _sections_fit(model, full_sections, max_context_tokens):
        return None

    empty_sections = [
        (
            subscription.context_channel,
            _render_artifact_block(
                subscription.artifact,
                policy_name=subscription.policy_name,
                relation=subscription.relation,
                content="",
            ),
        )
        for subscription in subscriptions
    ]
    empty_context_tokens = count_text_tokens(model, _context_text(empty_sections))
    content_budget = max_context_tokens - empty_context_tokens
    if content_budget <= 0:
        return None

    per_artifact_budget = max(1, content_budget // len(subscriptions))
    compacted_sections: list[str] = []
    for subscription in subscriptions:
        try:
            compacted_content = (
                await compact_artifact_content(
                    subscription.artifact,
                    per_artifact_budget,
                )
            ).strip()
        except Exception as e:
            logger.warning(
                "Diffusion artifact compaction failed for %s: %s",
                subscription.artifact.artifact_id,
                e,
            )
            return None
        if not compacted_content:
            return None
        compacted_sections.append(
            _render_artifact_block(
                subscription.artifact,
                policy_name=subscription.policy_name,
                relation=subscription.relation,
                content=compacted_content,
            )
        )

    channel_sections = [
        (subscription.context_channel, rendered_section)
        for subscription, rendered_section in zip(
            subscriptions,
            compacted_sections,
            strict=True,
        )
    ]
    if _sections_fit(model, channel_sections, max_context_tokens):
        return compacted_sections
    return None


async def _compacted_artifact_section(
    *,
    compact_artifact_content: DiffusionArtifactCompactor | None,
    max_context_tokens: int | None,
    model: str,
    rendered_sections: list[tuple[str, str]],
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
    try:
        compacted_content = (
            await compact_artifact_content(subscription.artifact, content_budget)
        ).strip()
    except Exception as e:
        logger.warning(
            "Diffusion artifact compaction failed for %s: %s",
            subscription.artifact.artifact_id,
            e,
        )
        return None
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
    rendered_sections: list[tuple[str, str]],
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
    sections: list[tuple[str, str]],
    max_context_tokens: int | None,
) -> bool:
    if max_context_tokens is None:
        return True
    if not sections:
        return True
    return count_text_tokens(model, _context_text(sections)) <= max_context_tokens


def _context_text(rendered_sections: list[tuple[str, str]]) -> str:
    return PromptText.diffusion_context(
        DIFFUSED_SECTION_NAME,
        _channel_sections(rendered_sections),
    )


def _channel_sections(rendered_sections: list[tuple[str, str]]) -> list[str]:
    reuse_sections = [
        section
        for channel, section in rendered_sections
        if channel == REUSE_SUCCESS_CHANNEL
    ]
    avoid_sections = [
        section
        for channel, section in rendered_sections
        if channel == AVOID_RECHECK_CHANNEL
    ]
    sections: list[str] = []
    if reuse_sections:
        sections.append(
            "\n\n".join(
                [
                    f"### {REUSE_SECTION_NAME}",
                    *reuse_sections,
                ]
            )
        )
    if avoid_sections:
        sections.append(
            "\n\n".join(
                [
                    f"### {AVOID_RECHECK_SECTION_NAME}",
                    AVOID_RECHECK_WARNING,
                    *avoid_sections,
                ]
            )
        )
    return sections


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


def _artifact_success(artifact: DiffusionArtifact) -> bool | None:
    if artifact.verifier_reward is None:
        return None
    return artifact.verifier_reward == 1.0


def _artifact_regression(artifact: DiffusionArtifact) -> bool | None:
    if artifact.metadata.get("regression") is True:
        return True
    return None
