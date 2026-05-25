"""Empirical gate for executor skill candidates."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.analysis.judge_rewards import (
    judge_reward_for_trace,
    judge_reward_metadata,
)
from mediated_coevo.benchmarks import SkillsBenchRepository
from mediated_coevo.core.config import Config
from mediated_coevo.evolution.candidates import (
    build_candidate_batch,
    candidate_batch_artifact_path,
    candidate_refs,
    proposal_to_candidate,
    selected_candidate,
    stable_selection_seed,
)
from mediated_coevo.evolution.skill_advisor import SkillAdvisor
from mediated_coevo.experiment.records import rollback_snapshot, task_metadata_fields
from mediated_coevo.models.skill import (
    AdvisorBatchProvenance,
    ProposalRef,
    RejectedProposalBatch,
    RejectedSkillProposal,
    SkillProposal,
    SkillUpdate,
    SkillValidationDecision,
    SkillValidationResult,
    SkillValidationTaskResult,
)
from mediated_coevo.models.judge import JudgeRewardRecord
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore

logger = logging.getLogger(__name__)

_SWE_LIKE_TAGS = {
    "bugfix",
    "build",
    "ci",
    "compilation",
    "debugging",
    "github actions",
    "java",
    "maven",
    "python",
    "pytest",
    "software",
    "swebench",
}


@dataclass(frozen=True)
class _TaskProfile:
    task_id: str
    benchmark_kind: str
    category: str
    tags: frozenset[str]


@dataclass
class ExecutorSkillGate:
    """Review, validate, and commit executor skill candidates."""

    config: Config
    skill_store: SkillStore
    history_store: HistoryStore
    planner: PlannerAgent
    skill_advisor: SkillAdvisor
    executor: ExecutorAgent
    benchmark_repo: SkillsBenchRepository
    artifact_store: ArtifactStore
    judge_llm_client: Any | None = None
    last_advisor_decision: str | None = None
    last_advisor_reason: str | None = None
    last_proposal_ids: list[str] = field(default_factory=list)
    last_rejection_id: str | None = None
    validation_task_pool: list[str] = field(default_factory=list)

    async def review_and_patch(
        self,
        *,
        iteration: int,
        proposal_buffer: list[SkillProposal],
    ) -> SkillUpdate | None:
        """Review buffered proposals and write only empirically accepted patches."""
        self.last_advisor_decision = None
        self.last_advisor_reason = None
        self.last_proposal_ids = []
        self.last_rejection_id = None

        if not self.config.experiment.skill_updates.executor:
            if proposal_buffer:
                logger.info(
                    "Executor skill updates disabled; clearing %d buffered proposal(s).",
                    len(proposal_buffer),
                )
                proposal_buffer.clear()
            return None

        if len(proposal_buffer) < self.config.experiment.advisor_buffer_max:
            return None

        logger.info("Advisor reviewing %d proposals...", len(proposal_buffer))
        current_skill = self.skill_store.read_skill("executor") or ""
        buffered_proposals = list(proposal_buffer)
        self.last_proposal_ids = [
            proposal.proposal_id for proposal in buffered_proposals
        ]
        batch_id = (
            f"coevo-iter-{iteration:04d}-batch-"
            f"{_proposal_batch_fingerprint(buffered_proposals)}"
        )
        advisor_feedback = await self.skill_advisor.review(
            current_skill=current_skill,
            proposals=buffered_proposals,
        )
        proposal_buffer.clear()

        if not advisor_feedback:
            logger.info("Advisor rejected — no skill update.")
            rejection_reason = (
                self.skill_advisor.last_rejection_reason or "advisor_rejected"
            )
            self.last_advisor_decision = "rejected"
            self.last_advisor_reason = rejection_reason
            self.last_rejection_id = self._record_rejected_proposals(
                iteration=iteration,
                batch_id=batch_id,
                current_skill=current_skill,
                proposals=buffered_proposals,
                reason=rejection_reason,
            )
            return None

        logger.info("Advisor approved — Planner patching skill...")
        self.last_advisor_decision = "approved"
        self.last_advisor_reason = advisor_feedback
        contributing_tasks = sorted(
            {p.task_id for p in buffered_proposals if p.task_id}
        )
        validation_task_ids = self._select_validation_task_ids(
            contributing_tasks=contributing_tasks,
            iteration=iteration,
            batch_id=batch_id,
        )
        contributing_task_ids = ",".join(contributing_tasks)
        edit_history = self.history_store.query(
            agent_role="planner",
            tagged_only=True,
        )
        candidates = []
        batch_method = getattr(self.planner, "suggest_skill_revision_batch", None)
        if callable(batch_method):
            candidates = await batch_method(
                current_skill_content=current_skill,
                feedback=advisor_feedback,
                edit_history=edit_history,
                skill_id="executor",
                task_ids=contributing_tasks,
                iteration=iteration,
            )
        else:
            draft_update = await self.planner.suggest_skill_revision(
                current_skill_content=current_skill,
                feedback=advisor_feedback,
                edit_history=edit_history,
                task_id=contributing_task_ids,
                iteration=iteration,
            )
            if draft_update:
                candidates = [proposal_to_candidate(draft_update)]
        if not candidates:
            return None

        old_skill_hash = SkillStore.content_hash(current_skill)
        candidate_batch_id = f"{batch_id}-executor-candidates"
        selection_seed = stable_selection_seed(
            base_seed=self.config.experiment.seed,
            iteration=iteration,
            skill_id="executor",
            batch_id=candidate_batch_id,
        )
        candidate_batch = build_candidate_batch(
            batch_id=candidate_batch_id,
            iteration=iteration,
            skill_id="executor",
            agent_role="planner",
            task_ids=contributing_tasks,
            current_skill=current_skill,
            candidates=candidates,
            selection_seed=selection_seed,
        )
        self.artifact_store.store_candidate_batch(candidate_batch)
        candidate_batch_path = candidate_batch_artifact_path(candidate_batch.batch_id)
        selected = selected_candidate(candidate_batch)
        if selected is None:
            self.last_rejection_id = self._record_rejected_proposals(
                iteration=iteration,
                batch_id=batch_id,
                current_skill=current_skill,
                proposals=buffered_proposals,
                reason="candidate_selection: no_valid_candidates",
                advisor_feedback=advisor_feedback,
                candidate_batch_id=candidate_batch.batch_id,
                candidate_batch_path=candidate_batch_path,
            )
            return None

        new_skill_hash = SkillStore.content_hash(selected.new_content)
        validation = await self._validate_candidate(
            validation_id=batch_id,
            iteration=iteration,
            task_ids=validation_task_ids,
            current_skill=current_skill,
            candidate_skill=selected.new_content,
        )
        if validation.decision != "accepted":
            logger.info(
                "Empirical validation rejected executor skill candidate: %s",
                validation.reason,
            )
            self.last_rejection_id = self._record_rejected_proposals(
                iteration=iteration,
                batch_id=batch_id,
                current_skill=current_skill,
                proposals=buffered_proposals,
                reason=f"validation: {validation.reason}",
                advisor_feedback=advisor_feedback,
                validation=validation,
                candidate_batch_id=candidate_batch.batch_id,
                candidate_batch_path=candidate_batch_path,
            )
            return None

        provenance = AdvisorBatchProvenance(
            batch_id=batch_id,
            iteration=iteration,
            skill_id="executor",
            task_ids=contributing_tasks,
            base_skill_hash=old_skill_hash,
            decision="approved",
            reason=advisor_feedback,
            rollback_snapshot=rollback_snapshot(iteration),
            validation=validation,
            candidate_batch_id=candidate_batch.batch_id,
            candidate_batch_path=candidate_batch_path,
            selected_candidate_id=selected.candidate_id,
            selected_update_kind=selected.update_kind,
            candidate_refs=candidate_refs(candidate_batch),
            proposal_refs=[
                ProposalRef(
                    proposal_id=proposal.proposal_id,
                    task_id=proposal.task_id,
                    iteration=proposal.iteration,
                    reward=proposal.reward,
                    reward_source=proposal.reward_source,
                    verifier_reward=proposal.verifier_reward,
                    judge_reward=proposal.judge_reward,
                )
                for proposal in buffered_proposals
            ],
        )
        skill_update = SkillUpdate(
            skill_id="executor",
            task_id=contributing_task_ids,
            old_content=current_skill,
            new_content=selected.new_content,
            reasoning=advisor_feedback,
            iteration=iteration,
            old_skill_hash=old_skill_hash,
            new_skill_hash=new_skill_hash,
            provenance=provenance,
        )
        self.skill_store.write_skill("executor", skill_update.new_content)
        logger.info("Skill patched and written.")
        return skill_update

    def _record_rejected_proposals(
        self,
        *,
        iteration: int,
        batch_id: str,
        current_skill: str,
        proposals: list[SkillProposal],
        reason: str,
        advisor_feedback: str | None = None,
        validation: SkillValidationResult | None = None,
        candidate_batch_id: str | None = None,
        candidate_batch_path: str | None = None,
    ) -> str:
        """Persist rejected proposal evidence without writing a skill update."""
        old_skill_hash = SkillStore.content_hash(current_skill)
        rejected_batch = RejectedProposalBatch(
            batch_id=batch_id,
            iteration=iteration,
            skill_id="executor",
            task_ids=sorted(
                proposal.task_id for proposal in proposals if proposal.task_id
            ),
            base_skill_hash=old_skill_hash,
            reason=reason,
            advisor_feedback=advisor_feedback,
            validation=validation,
            candidate_batch_id=candidate_batch_id,
            candidate_batch_path=candidate_batch_path,
            proposals=[
                RejectedSkillProposal(
                    proposal_id=proposal.proposal_id,
                    iteration=proposal.iteration,
                    task_id=proposal.task_id,
                    reward=proposal.reward,
                    reward_source=proposal.reward_source,
                    verifier_reward=proposal.verifier_reward,
                    judge_reward=proposal.judge_reward,
                    old_content=proposal.old_content,
                    new_content=proposal.new_content,
                    reasoning=proposal.reasoning,
                    old_skill_hash=SkillStore.content_hash(proposal.old_content),
                    new_skill_hash=SkillStore.content_hash(proposal.new_content),
                )
                for proposal in proposals
            ],
        )
        return self.history_store.record_rejected_proposals(rejected_batch)

    def _select_validation_task_ids(
        self,
        *,
        contributing_tasks: list[str],
        iteration: int,
        batch_id: str,
    ) -> list[str]:
        """Select holdout validation tasks compatible with proposal provenance."""
        validation_config = self.config.experiment.skill_validation
        configured_pool = [
            *validation_config.skillsbench_tasks,
            *validation_config.swebench_instances,
        ]
        pool = _dedupe_task_ids([*configured_pool, *self.validation_task_pool])
        excluded = set(contributing_tasks)
        contributor_profiles = [
            profile
            for task_id in contributing_tasks
            if (profile := self._resolve_task_profile(task_id)) is not None
        ]
        eligible: list[str] = []
        for task_id in pool:
            if task_id in excluded:
                continue
            candidate_profile = self._resolve_task_profile(task_id)
            if candidate_profile is None:
                continue
            if not contributor_profiles or any(
                _compatible_validation_profile(
                    contributor,
                    candidate_profile,
                    min_skillsbench_tag_overlap=(
                        validation_config.min_skillsbench_tag_overlap
                    ),
                    allow_swebench_replacement=(
                        validation_config.allow_swebench_replacement_for_skillsbench
                    ),
                )
                for contributor in contributor_profiles
            ):
                eligible.append(task_id)

        if eligible:
            selection_seed = stable_selection_seed(
                base_seed=self.config.experiment.seed,
                iteration=iteration,
                skill_id="executor",
                batch_id=f"{batch_id}-validation-tasks",
            )
            rng = random.Random(selection_seed)
            shuffled = sorted(_dedupe_task_ids(eligible))
            rng.shuffle(shuffled)
            sample_size = max(1, validation_config.sample_size)
            return sorted(shuffled[:sample_size])

        if validation_config.allow_contributing_fallback:
            return sorted(contributing_tasks)
        return []

    def _resolve_task_profile(self, task_id: str) -> _TaskProfile | None:
        try:
            task = self.benchmark_repo.resolve(task_id)
        except FileNotFoundError:
            return None
        return _task_profile_from_task(task_id, task)

    async def _validate_candidate(
        self,
        *,
        validation_id: str,
        iteration: int,
        task_ids: list[str],
        current_skill: str,
        candidate_skill: str,
    ) -> SkillValidationResult:
        """Run current and candidate executor skills before adopting a candidate."""
        task_results = [
            await self._validate_task(
                validation_id=validation_id,
                iteration=iteration,
                task_id=task_id,
                current_skill=current_skill,
                candidate_skill=candidate_skill,
            )
            for task_id in task_ids
        ]
        result = self._validation_decision(
            validation_id=validation_id,
            task_ids=task_ids,
            task_results=task_results,
        )
        self.artifact_store.store_validation_result(
            validation_id,
            result,
        )
        return result

    async def _validate_task(
        self,
        *,
        validation_id: str,
        iteration: int,
        task_id: str,
        current_skill: str,
        candidate_skill: str,
    ) -> SkillValidationTaskResult:
        validation_config = self.config.experiment.skill_validation
        task_metadata = task_metadata_fields(task_id=task_id, task_config=None)
        try:
            benchmark_task = self.benchmark_repo.resolve(task_id)
            task_metadata = task_metadata_fields(
                task_id=task_id,
                task_config=benchmark_task.task_config,
            )
            task_spec = TaskSpec(
                task_id=task_id,
                instruction=benchmark_task.instruction,
                iteration=iteration,
            )
            current_trace = await self.executor.execute_task(
                task_spec,
                [current_skill] if current_skill else [],
            )
            current_path = self.artifact_store.store_validation_trace(
                validation_id,
                "current",
                current_trace,
            )
            current_judge_record = None
            if current_trace.is_usable_feedback_signal:
                current_judge_record = await judge_reward_for_trace(
                    trace=current_trace,
                    config=self.config,
                    llm_client=self.judge_llm_client,
                    trace_path=current_path,
                    task_category=task_metadata.get("task_category"),
                    task_difficulty=task_metadata.get("task_difficulty"),
                    expected_reward_range=task_metadata.get("expected_reward_range"),
                    verifier_type=task_metadata.get("verifier_type"),
                    verifier_status=current_trace.status,
                )
                if current_judge_record is not None:
                    _log_validation_judge_reward(
                        variant="current",
                        trace=current_trace,
                        judge_record=current_judge_record,
                    )

            candidate_trace = await self.executor.execute_task(
                task_spec,
                [candidate_skill] if candidate_skill else [],
            )
            candidate_path = self.artifact_store.store_validation_trace(
                validation_id,
                "candidate",
                candidate_trace,
            )
            candidate_judge_record = None
            if candidate_trace.is_usable_feedback_signal:
                candidate_judge_record = await judge_reward_for_trace(
                    trace=candidate_trace,
                    config=self.config,
                    llm_client=self.judge_llm_client,
                    trace_path=candidate_path,
                    task_category=task_metadata.get("task_category"),
                    task_difficulty=task_metadata.get("task_difficulty"),
                    expected_reward_range=task_metadata.get("expected_reward_range"),
                    verifier_type=task_metadata.get("verifier_type"),
                    verifier_status=candidate_trace.status,
                )
                if candidate_judge_record is not None:
                    _log_validation_judge_reward(
                        variant="candidate",
                        trace=candidate_trace,
                        judge_record=candidate_judge_record,
                    )
        except FileNotFoundError as e:
            current_trace = _validation_env_failure(
                task_id=task_id,
                iteration=iteration,
                error_kind="task_not_found",
                exc=e,
            )
            candidate_trace = _validation_env_failure(
                task_id=task_id,
                iteration=iteration,
                error_kind="task_not_found",
                exc=e,
            )
            current_path = self.artifact_store.store_validation_trace(
                validation_id,
                "current",
                current_trace,
            )
            candidate_path = self.artifact_store.store_validation_trace(
                validation_id,
                "candidate",
                candidate_trace,
            )
            current_judge_record = None
            candidate_judge_record = None

        usable = (
            current_trace.is_usable_feedback_signal
            and candidate_trace.is_usable_feedback_signal
        )
        current_reward, current_source = _evolution_reward(
            current_trace,
            current_judge_record,
        )
        candidate_reward, candidate_source = _evolution_reward(
            candidate_trace,
            candidate_judge_record,
        )
        regressed = (
            usable
            and candidate_reward is not None
            and current_reward is not None
            and candidate_reward < current_reward - validation_config.reward_tolerance
        )
        return SkillValidationTaskResult(
            task_id=task_id,
            current_reward=current_reward,
            candidate_reward=candidate_reward,
            current_reward_source=current_source,
            candidate_reward_source=candidate_source,
            current_verifier_reward=current_trace.reward,
            candidate_verifier_reward=candidate_trace.reward,
            current_judge_reward=(
                current_judge_record.judge_reward
                if current_judge_record is not None
                else None
            ),
            candidate_judge_reward=(
                candidate_judge_record.judge_reward
                if candidate_judge_record is not None
                else None
            ),
            current_status=current_trace.status,
            candidate_status=candidate_trace.status,
            current_trace_path=str(current_path),
            candidate_trace_path=str(candidate_path),
            usable=usable,
            regressed=regressed,
        )

    def _validation_decision(
        self,
        *,
        validation_id: str,
        task_ids: list[str],
        task_results: list[SkillValidationTaskResult],
    ) -> SkillValidationResult:
        validation_config = self.config.experiment.skill_validation
        usable_results = [result for result in task_results if result.usable]
        current_mean = _mean_reward(result.current_reward for result in usable_results)
        candidate_mean = _mean_reward(
            result.candidate_reward for result in usable_results
        )
        mean_delta = (
            candidate_mean - current_mean
            if current_mean is not None and candidate_mean is not None
            else None
        )
        decision: SkillValidationDecision = "accepted"
        reason = "accepted"
        if not task_results:
            decision = "rejected"
            reason = "no_validation_tasks"
        elif validation_config.require_all_tasks_usable and (
            len(usable_results) != len(task_results)
        ):
            decision = "rejected"
            reason = "unusable_validation_trace"
        elif not usable_results:
            decision = "rejected"
            reason = "no_usable_validation_tasks"
        elif any(result.regressed for result in usable_results):
            decision = "rejected"
            reason = "task_regression"
        elif (
            mean_delta is None
            or mean_delta
            < validation_config.min_mean_delta - validation_config.reward_tolerance
        ):
            decision = "rejected"
            reason = "mean_not_improved"

        return SkillValidationResult(
            validation_id=validation_id,
            task_ids=list(task_ids),
            decision=decision,
            reason=reason,
            current_mean_reward=current_mean,
            candidate_mean_reward=candidate_mean,
            mean_delta=mean_delta,
            min_mean_delta=validation_config.min_mean_delta,
            reward_tolerance=validation_config.reward_tolerance,
            task_results=task_results,
        )


def _proposal_batch_fingerprint(proposals: list[SkillProposal]) -> str:
    """Return a stable short id for a buffered proposal batch."""
    parts = [
        "|".join(
            [
                proposal.proposal_id,
                str(proposal.iteration),
                proposal.task_id,
                SkillStore.content_hash(proposal.old_content)[:12],
                SkillStore.content_hash(proposal.new_content)[:12],
            ]
        )
        for proposal in sorted(
            proposals,
            key=lambda item: (item.iteration, item.task_id, item.proposal_id),
        )
    ]
    return SkillStore.content_hash("\n".join(parts))[:10]


def _dedupe_task_ids(task_ids: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for task_id in task_ids:
        if not task_id or task_id in seen:
            continue
        seen.add(task_id)
        deduped.append(task_id)
    return deduped


def _task_profile_from_task(task_id: str, task: Any) -> _TaskProfile:
    task_config = getattr(task, "task_config", None) or {}
    metadata = task_config.get("metadata") or {}
    benchmark_kind = _benchmark_kind(task_id, task)
    category = _normalize_token(metadata.get("category"))
    tags = set(_normalized_tags(metadata.get("tags")))
    if benchmark_kind == "swebench":
        category = category or "software"
        tags.update({"software", "swebench"})
    return _TaskProfile(
        task_id=task_id,
        benchmark_kind=benchmark_kind,
        category=category,
        tags=frozenset(tags),
    )


def _benchmark_kind(task_id: str, task: Any) -> str:
    explicit_kind = getattr(task, "benchmark_kind", None)
    if explicit_kind:
        return str(explicit_kind)
    benchmark_by_task_id = getattr(task, "benchmark_by_task_id", None)
    if isinstance(benchmark_by_task_id, dict):
        kind = benchmark_by_task_id.get(task_id)
        if kind:
            return str(kind)
    if hasattr(task, "task_dir"):
        return "skillsbench"
    if hasattr(task, "repo") and hasattr(task, "base_commit"):
        return "swebench"
    return "unknown"


def _compatible_validation_profile(
    contributor: _TaskProfile,
    candidate: _TaskProfile,
    *,
    min_skillsbench_tag_overlap: int,
    allow_swebench_replacement: bool,
) -> bool:
    if candidate.task_id == contributor.task_id:
        return False

    if contributor.benchmark_kind != candidate.benchmark_kind:
        return (
            allow_swebench_replacement
            and contributor.benchmark_kind == "skillsbench"
            and candidate.benchmark_kind == "swebench"
            and _is_swe_like_profile(contributor)
        )

    if contributor.benchmark_kind == "skillsbench":
        return _similar_skillsbench_profile(
            contributor,
            candidate,
            min_skillsbench_tag_overlap=min_skillsbench_tag_overlap,
        )
    if contributor.benchmark_kind == "swebench":
        return True
    if contributor.category and contributor.category == candidate.category:
        return True
    return len(contributor.tags & candidate.tags) >= min_skillsbench_tag_overlap


def _similar_skillsbench_profile(
    contributor: _TaskProfile,
    candidate: _TaskProfile,
    *,
    min_skillsbench_tag_overlap: int,
) -> bool:
    if contributor.category and contributor.category == candidate.category:
        return True
    return len(contributor.tags & candidate.tags) >= min_skillsbench_tag_overlap


def _is_swe_like_profile(profile: _TaskProfile) -> bool:
    if profile.benchmark_kind == "swebench":
        return True
    category = profile.category
    category_is_swe = any(token in category for token in ("build", "software"))
    return category_is_swe or bool(profile.tags & _SWE_LIKE_TAGS)


def _normalized_tags(raw_tags: Any) -> set[str]:
    if not isinstance(raw_tags, list):
        return set()
    return {
        tag
        for tag in (_normalize_token(raw_tag) for raw_tag in raw_tags)
        if tag
    }


def _normalize_token(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def _validation_env_failure(
    *,
    task_id: str,
    iteration: int,
    error_kind: str,
    exc: BaseException,
) -> ExecutionTrace:
    return ExecutionTrace(
        task_id=task_id,
        iteration=iteration,
        status="env_failure",
        error_kind=error_kind,
        error_detail=str(exc),
    )


def _evolution_reward(
    trace: ExecutionTrace,
    judge_record: JudgeRewardRecord | None,
) -> tuple[float | None, str | None]:
    if judge_record is None:
        return trace.reward, "verifier" if trace.reward is not None else None
    metadata = judge_reward_metadata(judge_record)
    return judge_record.judge_reward, metadata["reward_source"]


def _mean_reward(rewards: Iterable[float | None]) -> float | None:
    """Return the arithmetic mean for non-None reward values."""
    values = [reward for reward in rewards if reward is not None]
    if not values:
        return None
    return sum(values) / len(values)


def _log_validation_judge_reward(
    *,
    variant: str,
    trace: ExecutionTrace,
    judge_record: JudgeRewardRecord,
) -> None:
    metadata = judge_reward_metadata(judge_record)
    logger.info(
        "Validation judge reward after task run: variant=%s task=%s "
        "iteration=%d verifier_reward=%s judge_reward=%s reward_source=%s",
        variant,
        trace.task_id,
        trace.iteration + 1,
        _format_reward(trace.reward),
        _format_reward(judge_record.judge_reward),
        metadata["reward_source"],
    )


def _format_reward(reward: float | None) -> str:
    return f"{reward:.2f}" if reward is not None else "n/a"
