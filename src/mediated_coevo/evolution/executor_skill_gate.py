"""Empirical gate for executor skill candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.benchmarks import SkillsBenchRepository
from mediated_coevo.core.config import Config
from mediated_coevo.evolution.skill_advisor import SkillAdvisor
from mediated_coevo.experiment.records import rollback_snapshot
from mediated_coevo.models.skill import (
    AdvisorBatchProvenance,
    ProposalRef,
    SkillProposal,
    SkillUpdate,
    SkillValidationDecision,
    SkillValidationResult,
    SkillValidationTaskResult,
)
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore

logger = logging.getLogger(__name__)


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

    async def review_and_patch(
        self,
        *,
        iteration: int,
        proposal_buffer: list[SkillProposal],
    ) -> SkillUpdate | None:
        """Review buffered proposals and write only empirically accepted patches."""
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
        advisor_feedback = await self.skill_advisor.review(
            current_skill=current_skill,
            proposals=buffered_proposals,
        )
        proposal_buffer.clear()

        if not advisor_feedback:
            logger.info("Advisor rejected — no skill update.")
            return None

        logger.info("Advisor approved — Planner patching skill...")
        batch_id = f"coevo-iter-{iteration:04d}"
        contributing_tasks = sorted({p.task_id for p in buffered_proposals if p.task_id})
        contributing_task_ids = ",".join(contributing_tasks)
        edit_history = self.history_store.query(
            agent_role="planner",
            tagged_only=True,
        )
        draft_update = await self.planner.suggest_skill_revision(
            current_skill_content=current_skill,
            feedback=advisor_feedback,
            edit_history=edit_history,
            task_id=contributing_task_ids,
            iteration=iteration,
        )
        if not draft_update:
            return None

        old_skill_hash = SkillStore.content_hash(current_skill)
        new_skill_hash = SkillStore.content_hash(draft_update.new_content)
        validation = await self._validate_candidate(
            validation_id=batch_id,
            iteration=iteration,
            task_ids=contributing_tasks,
            current_skill=current_skill,
            candidate_skill=draft_update.new_content,
        )
        if validation.decision != "accepted":
            logger.info(
                "Empirical validation rejected executor skill candidate: %s",
                validation.reason,
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
            proposal_refs=[
                ProposalRef(
                    proposal_id=proposal.proposal_id,
                    task_id=proposal.task_id,
                    iteration=proposal.iteration,
                    reward=proposal.reward,
                )
                for proposal in buffered_proposals
            ],
        )
        skill_update = SkillUpdate(
            skill_id="executor",
            task_id=contributing_task_ids,
            old_content=current_skill,
            new_content=draft_update.new_content,
            reasoning=advisor_feedback,
            iteration=iteration,
            old_skill_hash=old_skill_hash,
            new_skill_hash=new_skill_hash,
            provenance=provenance,
        )
        self.skill_store.write_skill("executor", skill_update.new_content)
        logger.info("Skill patched and written.")
        return skill_update

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
        validation_config = self.config.experiment.skill_validation
        if not validation_config.enabled:
            return SkillValidationResult(
                validation_id=validation_id,
                task_ids=list(task_ids),
                decision="accepted",
                reason="validation_disabled",
                min_mean_delta=validation_config.min_mean_delta,
                reward_tolerance=validation_config.reward_tolerance,
            )

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
            overwrite=True,
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
        try:
            benchmark_task = self.benchmark_repo.resolve(task_id)
            task_spec = TaskSpec(
                task_id=task_id,
                instruction=benchmark_task.instruction,
                iteration=iteration,
            )
            current_trace = await self.executor.execute_task(
                task_spec,
                [current_skill] if current_skill else [],
            )
            candidate_trace = await self.executor.execute_task(
                task_spec,
                [candidate_skill] if candidate_skill else [],
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
            overwrite=True,
        )
        candidate_path = self.artifact_store.store_validation_trace(
            validation_id,
            "candidate",
            candidate_trace,
            overwrite=True,
        )
        usable = (
            current_trace.is_usable_feedback_signal
            and candidate_trace.is_usable_feedback_signal
        )
        current_reward = current_trace.reward
        candidate_reward = candidate_trace.reward
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
        current_mean = _mean_reward(
            result.current_reward for result in usable_results
        )
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


def _mean_reward(rewards: Iterable[float | None]) -> float | None:
    """Return the arithmetic mean for non-None reward values."""
    values = [reward for reward in rewards if reward is not None]
    if not values:
        return None
    return sum(values) / len(values)
