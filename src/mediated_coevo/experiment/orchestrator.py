"""Orchestrator — main iteration loop.

Wires agents together and drives the plan → execute → mediate → update loop.
Triggers co-evolution reflections every N iterations.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.analysis.judge_rewards import (
    append_judge_reward_record,
    judge_reward_for_trace,
    judge_reward_metadata,
)
from mediated_coevo.analysis.metrics import metric_row
from mediated_coevo.benchmarks import SkillsBenchRepository
from mediated_coevo.core.config import Config
from mediated_coevo.evolution.compactor import build_planner_signal
from mediated_coevo.evolution.executor_skill_gate import ExecutorSkillGate
from mediated_coevo.evolution.reflector import ReflectionDraft, Reflector
from mediated_coevo.evolution.skill_advisor import SkillAdvisor
from mediated_coevo.experiment.conditions import (
    ConditionName,
    get_cross_task_prior_context,
    get_executor_proposal_feedback,
    get_prior_context,
)
from mediated_coevo.experiment.records import (
    attach_skill_identity,
    build_coevolution_record,
    build_iteration_record,
    build_missing_task_record,
    TaskMetadataFields,
    rollback_snapshot,
    skill_update_from_reflection,
    skill_version,
    task_metadata_fields,
)
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.judge import JudgeRewardRecord
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.skill import (
    SkillProposal,
    SkillUpdate,
    SkillUpdateCandidate,
    SkillValidationResult,
    SkillValidationTaskResult,
)
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.runtime.token_budget import TokenBudgetEvent
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PlannerValidationRun:
    reward: float | None
    reward_source: str | None
    verifier_reward: float | None
    judge_reward: float | None
    status: str
    trace_path: str | None
    usable: bool


class Orchestrator:
    """Runs the plan → execute → mediate → update loop."""

    def __init__(
        self,
        planner: PlannerAgent,
        executor: ExecutorAgent,
        mediator: MediatorAgent,
        skill_store: SkillStore,
        artifact_store: ArtifactStore,
        history_store: HistoryStore,
        benchmark_repo: SkillsBenchRepository,
        config: Config,
        experiment_dir: Path,
        skill_advisor: SkillAdvisor,
        judge_llm_client: Any | None = None,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.mediator = mediator
        self.skill_store = skill_store
        self.artifact_store = artifact_store
        self.history_store = history_store
        self.benchmark_repo = benchmark_repo
        self.config = config
        self.experiment_dir = experiment_dir
        self.skill_advisor = skill_advisor
        self.judge_llm_client = judge_llm_client
        self._proposal_buffer: list[SkillProposal] = []
        self.executor_skill_gate = ExecutorSkillGate(
            config=config,
            skill_store=skill_store,
            history_store=history_store,
            planner=planner,
            skill_advisor=skill_advisor,
            executor=executor,
            benchmark_repo=benchmark_repo,
            artifact_store=artifact_store,
            judge_llm_client=judge_llm_client,
        )

        self._snapshots_dir = experiment_dir / "skills_snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_path = experiment_dir / "metrics.jsonl"
        self._previous_report_by_task: dict[str, MediatorReport] = {}
        self._previous_reward_by_task: dict[str, float] = {}

    async def run_experiment(
        self,
        task_ids: list[str],
        num_iterations: int | None = None,
    ) -> list[IterationRecord]:
        """Run the full experiment loop."""
        if num_iterations is None:
            num_iterations = self.config.experiment.num_iterations
        records: list[IterationRecord] = []
        selection = self.config.experiment.benchmark_selection
        self.executor_skill_gate.validation_task_pool = [
            *selection.skillsbench_tasks,
            *selection.swebench_instances,
            *task_ids,
        ]

        for iteration in range(num_iterations):
            iteration_records: list[IterationRecord] = []
            for task_id in task_ids:
                logger.info(
                    "=== Iteration %d/%d | Task: %s ===",
                    iteration + 1,
                    num_iterations,
                    task_id,
                )
                record = await self._run_iteration(task_id, iteration)
                records.append(record)
                iteration_records.append(record)

            # Co-evolution checkpoint
            coevolution_record: IterationRecord | None = None
            if (iteration + 1) % self.config.experiment.coevo_interval == 0:
                coevolution_record = await self._coevolve(
                    iteration,
                    self.config.experiment.condition_name,
                )

            self._snapshot_and_write_metrics(
                iteration,
                iteration_records,
                coevolution_record=coevolution_record,
            )

        logger.info(
            "Experiment complete: %d iterations, %d records",
            num_iterations,
            len(records),
        )
        return records

    async def _run_iteration(
        self,
        task_id: str,
        iteration: int,
    ) -> IterationRecord:
        start = time.time()
        condition = self.config.experiment.condition_name
        skill_hashes = self._current_skill_hashes()
        executor_skill_text = self.skill_store.read_skill("executor") or ""
        planner_skill_text = self.skill_store.read_skill("planner") or None

        try:
            benchmark_task = self.benchmark_repo.resolve(task_id)
        except FileNotFoundError as e:
            return self._record_missing_task(
                task_id=task_id,
                iteration=iteration,
                condition=condition,
                start=start,
                exc=e,
                skill_hashes=skill_hashes,
            )

        task_metadata = task_metadata_fields(
            task_id=task_id,
            task_config=benchmark_task.task_config,
        )

        self.planner.set_skill_context(
            executor_skills=executor_skill_text,
            skill_refiner=planner_skill_text,
        )

        skill_texts = [executor_skill_text] if executor_skill_text else []
        prior_context = await self._build_prior_context(
            condition,
            task_id,
            current_iteration=iteration,
        )
        logger.info("Step 1: Planner planning task (condition=%s)...", condition)
        task_spec = await self.planner.plan_task(
            task_id=task_id,
            base_instruction=benchmark_task.instruction,
            prior_context=prior_context,
            current_skills=skill_texts,
            iteration=iteration,
        )

        logger.info("Step 2: Executor running task...")
        trace = await self.executor.execute_task(task_spec, skill_texts)

        report = None
        trace_path = None
        try:
            report = await self.mediator.mediate_trace(condition, trace, task_spec)
            if report is not None:
                self.artifact_store.store_report(report)
        finally:
            trace_path = self.artifact_store.store_trace(trace)

        outcome_reward = None
        outcome_metadata = None
        if trace.iteration > 0 and trace.is_usable_feedback_signal:
            judge_record = await self._judge_evolution_reward(
                trace=trace,
                task_metadata=task_metadata,
                trace_path=trace_path,
            )
            if judge_record is not None:
                if not judge_record.metadata.get("judge_reward_fallback"):
                    append_judge_reward_record(self.experiment_dir, judge_record)
                outcome_reward = judge_record.judge_reward
                outcome_metadata = judge_reward_metadata(judge_record)

        proposal_feedback = await get_executor_proposal_feedback(
            condition=condition,
            task_id=task_id,
            artifact_store=self.artifact_store,
            mediator_report=report,
            llm_client=self.mediator.llm_client if condition == "full_traces" else None,
            model=self.planner.llm_client.model,
            budgets=self.config.budgets,
            condition_name=condition,
        )

        await self._ask_planner_for_skill_proposal(
            task_id=task_id,
            iteration=iteration,
            executor_skill=executor_skill_text,
            feedback=proposal_feedback,
        )

        self.history_store.tag_pending_outcome(
            task_id,
            trace,
            proposals=self._proposal_buffer,
            outcome_reward=outcome_reward,
            outcome_metadata=outcome_metadata,
        )

        skill_update = await self.executor_skill_gate.review_and_patch(
            iteration=iteration,
            proposal_buffer=self._proposal_buffer,
        )
        mediator_entry_id, planner_entry_id = (
            await self._record_history_and_remember_outcome(
                task_id=task_id,
                iteration=iteration,
                condition=condition,
                report=report,
                skill_update=skill_update,
            )
        )

        duration = time.time() - start
        llm_token_events = self._drain_llm_token_events()
        record = build_iteration_record(
            task_id=task_id,
            iteration=iteration,
            condition=condition,
            duration_sec=duration,
            task_spec=task_spec,
            trace=trace,
            report=report,
            skill_update=skill_update,
            mediator_entry_id=mediator_entry_id,
            planner_entry_id=planner_entry_id,
            skill_hashes=skill_hashes,
            task_metadata=task_metadata,
            llm_token_events=llm_token_events,
            config=self.config,
            previous_reward_by_task=self._previous_reward_by_task,
        )
        if self.executor_skill_gate.last_advisor_decision:
            record.advisor_decision = self.executor_skill_gate.last_advisor_decision
            record.advisor_reason = self.executor_skill_gate.last_advisor_reason
            record.advisor_rejection_id = self.executor_skill_gate.last_rejection_id
            record.proposal_ids = list(self.executor_skill_gate.last_proposal_ids)
        reward_str = f"{trace.reward:.2f}" if trace.reward is not None else "n/a"
        logger.info(
            "Iteration %d complete: condition=%s status=%s reward=%s tokens=%d duration=%.1fs",
            iteration,
            condition,
            trace.status,
            reward_str,
            record.total_tokens,
            duration,
        )
        return record

    async def _judge_evolution_reward(
        self,
        *,
        trace: ExecutionTrace,
        task_metadata: TaskMetadataFields,
        trace_path: Path | None,
    ) -> JudgeRewardRecord | None:
        """Score a usable trace for evolution, preserving verifier reward in metadata."""
        return await judge_reward_for_trace(
            trace=trace,
            config=self.config,
            llm_client=getattr(self, "judge_llm_client", None),
            trace_path=trace_path,
            task_category=task_metadata.get("task_category"),
            task_difficulty=task_metadata.get("task_difficulty"),
            expected_reward_range=task_metadata.get("expected_reward_range"),
            verifier_type=task_metadata.get("verifier_type"),
            verifier_status=trace.status,
        )

    def _record_missing_task(
        self,
        *,
        task_id: str,
        iteration: int,
        condition: ConditionName,
        start: float,
        exc: FileNotFoundError,
        skill_hashes: dict[str, str],
    ) -> IterationRecord:
        duration = time.time() - start
        trace = ExecutionTrace(
            task_id=task_id,
            iteration=iteration,
            duration_sec=duration,
            exit_code=-1,
            status="env_failure",
            error_kind="task_not_found",
            error_detail=str(exc),
        )
        self.artifact_store.store_trace(trace)
        self.history_store.tag_pending_outcome(
            task_id,
            trace,
            proposals=self._proposal_buffer,
        )
        llm_token_events = self._drain_llm_token_events()
        logger.warning(
            "Iteration %d skipped before planning: task=%s status=%s error_kind=%s",
            iteration,
            task_id,
            trace.status,
            trace.error_kind,
        )
        return build_missing_task_record(
            iteration=iteration,
            task_id=task_id,
            condition=condition,
            duration_sec=duration,
            trace=trace,
            llm_token_events=llm_token_events,
            config=self.config,
            skill_hashes=dict(skill_hashes),
        )

    async def _ask_planner_for_skill_proposal(
        self,
        *,
        task_id: str,
        iteration: int,
        executor_skill: str,
        feedback: str | None,
    ) -> None:
        if not self.config.experiment.skill_updates.executor:
            logger.info("Step 4: Skipped (executor skill updates disabled).")
            return
        if not feedback or not executor_skill:
            logger.info("Step 4: Skipped (no proposal feedback).")
            return

        logger.info("Step 4: Planner proposing skill update...")
        edit_history = self.history_store.query(
            agent_role="planner",
            tagged_only=True,
        )
        proposal = await self.planner.suggest_skill_revision(
            current_skill_content=executor_skill,
            feedback=feedback,
            edit_history=edit_history,
            task_id=task_id,
            iteration=iteration,
        )
        if proposal:
            self._proposal_buffer.append(proposal)
            logger.info(
                "Proposal buffered (buffer size=%d)", len(self._proposal_buffer)
            )
        else:
            logger.info("Planner decided: no proposal needed.")

    async def _record_history_and_remember_outcome(
        self,
        *,
        task_id: str,
        iteration: int,
        condition: ConditionName,
        report: MediatorReport | None,
        skill_update: SkillUpdate | None,
    ) -> tuple[str | None, str | None]:
        mediator_entry_id = None
        if report is not None:
            mediator_signal = await self.mediator.compact_feedback(report)
            mediator_entry_id = self.history_store.record_signal(
                iteration=iteration,
                agent_role="mediator",
                task_id=task_id,
                condition=condition,
                payload=mediator_signal,
            )

        planner_entry_id = None
        if skill_update is not None:
            planner_entry_id = self.history_store.record_signal(
                iteration=iteration,
                agent_role="planner",
                task_id=task_id,
                condition=condition,
                payload=build_planner_signal(skill_update),
            )

        self.history_store.remember_pending_outcome(
            task_id,
            mediator_entry_id=mediator_entry_id,
            planner_entry_id=planner_entry_id,
        )
        if report is not None and report.is_exposed:
            self._previous_report_by_task[task_id] = report
        return mediator_entry_id, planner_entry_id

    async def _build_prior_context(
        self,
        condition: ConditionName,
        task_id: str,
        *,
        current_iteration: int | None = None,
    ) -> str | None:
        """Build same-task prior context, with explicit opt-in cross-task context."""
        llm_client = self.mediator.llm_client if condition == "full_traces" else None
        prior_context = await get_prior_context(
            condition=condition,
            task_id=task_id,
            artifact_store=self.artifact_store,
            previous_report=self._previous_report_by_task.get(task_id),
            shared_notes=self.config.experiment.shared_notes,
            llm_client=llm_client,
            model=self.planner.llm_client.model,
            budgets=self.config.budgets,
            condition_name=condition,
            current_iteration=current_iteration,
        )
        if not self.config.experiment.allow_cross_task_feedback:
            return prior_context

        cross_context = await get_cross_task_prior_context(
            condition=condition,
            task_id=task_id,
            artifact_store=self.artifact_store,
            previous_reports_by_task=self._previous_report_by_task,
            llm_client=llm_client,
            model=self.planner.llm_client.model,
            budgets=self.config.budgets,
            condition_name=condition,
            current_iteration=current_iteration,
        )
        if not cross_context:
            return prior_context

        header = (
            "# Explicit Cross-Task Feedback\n\n"
            f"condition={condition} target_task={task_id} "
            "allow_cross_task_feedback=true\n\n"
            "The following context came from other tasks by explicit "
            "experiment configuration."
        )
        logger.info(
            "Cross-task feedback injected: condition=%s target_task=%s",
            condition,
            task_id,
        )
        if prior_context:
            return f"{prior_context}\n\n{header}\n\n{cross_context}"
        return f"{header}\n\n{cross_context}"

    async def _coevolve(
        self,
        iteration: int,
        condition: ConditionName,
    ) -> IterationRecord | None:
        """Co-evolution checkpoint: Mediator + Planner reflect on history."""
        start = time.time()
        logger.info(
            "=== Co-evolution checkpoint at iteration %d (condition=%s) ===",
            iteration,
            condition,
        )

        reflector = Reflector(
            self.history_store,
            self.skill_store,
            budgets=self.config.budgets,
            condition_name=condition,
            artifact_store=getattr(self, "artifact_store", None),
            base_seed=self.config.experiment.seed,
        )
        skill_updates: list[SkillUpdate] = []
        reflection_seed = random.randrange(1 << 32)

        if self.config.experiment.skill_updates.mediator:
            mediator_result = await reflector.reflect(
                "mediator",
                self.mediator.llm_client,
                iteration=iteration,
                selection_seed=reflection_seed,
            )
            if mediator_result:
                mediator_result.provenance.rollback_snapshot = rollback_snapshot(
                    iteration
                )
                self.mediator.load_protocol(mediator_result.new_content)
                skill_updates.append(skill_update_from_reflection(mediator_result))
        else:
            logger.info("Mediator skill evolution skipped (skill updates disabled).")

        if self.config.experiment.skill_updates.planner:
            planner_update = await self._reflect_and_validate_planner(
                reflector,
                iteration=iteration,
                selection_seed=reflection_seed + 1,
            )
            if planner_update is not None:
                skill_updates.append(planner_update)
        else:
            logger.info("Planner skill evolution skipped (skill updates disabled).")
        llm_token_events = self._drain_llm_token_events()
        if not llm_token_events and not skill_updates:
            return None
        return build_coevolution_record(
            iteration=iteration,
            condition=condition,
            duration_sec=time.time() - start,
            llm_token_events=llm_token_events,
            skill_updates=skill_updates,
            config=self.config,
            skill_hashes=self._current_skill_hashes(),
        )

    async def _reflect_and_validate_planner(
        self,
        reflector: Reflector,
        *,
        iteration: int,
        selection_seed: int,
    ) -> SkillUpdate | None:
        """Reflect Planner skill candidates and commit the best validated one."""
        draft = await reflector.draft_reflection(
            agent_role="planner",
            llm_client=self.planner.llm_client,
            iteration=iteration,
            selection_seed=selection_seed,
            candidate_count=2,
            select_candidate=False,
        )
        if draft is None:
            return None

        selected = await self._validate_planner_reflection(
            draft,
            iteration=iteration,
        )
        if selected is None:
            return None

        selected_candidate_id, validation = selected
        planner_result = reflector.commit_draft(
            draft,
            selected_candidate_id=selected_candidate_id,
            validation=validation,
        )
        if planner_result is None:
            return None
        planner_result.provenance.rollback_snapshot = rollback_snapshot(iteration)
        return skill_update_from_reflection(planner_result)

    async def _validate_planner_reflection(
        self,
        draft: ReflectionDraft,
        *,
        iteration: int,
    ) -> tuple[str, SkillValidationResult] | None:
        """Empirically select the best Planner reflection candidate."""
        candidates = _valid_planner_reflection_candidates(
            draft.candidate_batch.candidates,
        )
        if not candidates:
            return None

        validation_config = self.config.experiment.skill_validation
        validation_id = f"{draft.candidate_batch.batch_id}-planner-validation"
        if not validation_config.enabled:
            best = max(
                candidates,
                key=lambda candidate: (candidate.audit_score, candidate.candidate_id),
            )
            return best.candidate_id, SkillValidationResult(
                validation_id=validation_id,
                task_ids=[],
                decision="accepted",
                reason="validation_disabled",
                min_mean_delta=validation_config.min_mean_delta,
                reward_tolerance=validation_config.reward_tolerance,
            )

        task_ids = self.executor_skill_gate._select_validation_task_ids(
            contributing_tasks=draft.candidate_batch.task_ids,
            iteration=iteration,
            batch_id=validation_id,
        )
        executor_skill = self.skill_store.read_skill("executor") or ""
        current_runs: dict[str, _PlannerValidationRun] = {}
        try:
            for task_id in task_ids:
                current_runs[task_id] = await self._run_planner_validation_task(
                    validation_id=validation_id,
                    variant="current",
                    task_id=task_id,
                    iteration=iteration,
                    planner_skill=draft.old_content,
                    executor_skill=executor_skill,
                )

            candidate_results = [
                await self._validate_planner_candidate(
                    validation_id=validation_id,
                    iteration=iteration,
                    candidate=candidate,
                    task_ids=task_ids,
                    current_runs=current_runs,
                    executor_skill=executor_skill,
                )
                for candidate in candidates
            ]
        finally:
            self.planner.set_skill_context(
                executor_skills=executor_skill,
                skill_refiner=draft.old_content,
            )

        accepted = [
            (candidate, result)
            for candidate, result in zip(candidates, candidate_results, strict=True)
            if result.decision == "accepted" and result.candidate_mean_reward is not None
        ]
        if not accepted:
            logger.info("Planner reflection validation rejected all candidates.")
            if candidate_results:
                best_rejected = max(
                    candidate_results,
                    key=lambda result: result.candidate_mean_reward
                    if result.candidate_mean_reward is not None
                    else float("-inf"),
                )
                self.artifact_store.store_validation_result(
                    validation_id,
                    best_rejected,
                    overwrite=True,
                )
            return None

        best_candidate, best_result = max(
            accepted,
            key=lambda item: (
                item[1].candidate_mean_reward
                if item[1].candidate_mean_reward is not None
                else float("-inf"),
                item[1].mean_delta if item[1].mean_delta is not None else float("-inf"),
                item[0].audit_score,
                item[0].candidate_id,
            ),
        )
        self.artifact_store.store_validation_result(
            validation_id,
            best_result,
            overwrite=True,
        )
        return best_candidate.candidate_id, best_result

    async def _validate_planner_candidate(
        self,
        *,
        validation_id: str,
        iteration: int,
        candidate: SkillUpdateCandidate,
        task_ids: list[str],
        current_runs: dict[str, _PlannerValidationRun],
        executor_skill: str,
    ) -> SkillValidationResult:
        task_results = []
        variant = f"candidate-{candidate.candidate_id}"
        for task_id in task_ids:
            current = current_runs[task_id]
            candidate_run = await self._run_planner_validation_task(
                validation_id=validation_id,
                variant=variant,
                task_id=task_id,
                iteration=iteration,
                planner_skill=candidate.new_content,
                executor_skill=executor_skill,
            )
            usable = current.usable and candidate_run.usable
            regressed = (
                usable
                and current.reward is not None
                and candidate_run.reward is not None
                and candidate_run.reward
                < current.reward - self.config.experiment.skill_validation.reward_tolerance
            )
            task_results.append(
                SkillValidationTaskResult(
                    task_id=task_id,
                    current_reward=current.reward,
                    candidate_reward=candidate_run.reward,
                    current_reward_source=current.reward_source,
                    candidate_reward_source=candidate_run.reward_source,
                    current_verifier_reward=current.verifier_reward,
                    candidate_verifier_reward=candidate_run.verifier_reward,
                    current_judge_reward=current.judge_reward,
                    candidate_judge_reward=candidate_run.judge_reward,
                    current_status=current.status,
                    candidate_status=candidate_run.status,
                    current_trace_path=current.trace_path,
                    candidate_trace_path=candidate_run.trace_path,
                    usable=usable,
                    regressed=regressed,
                )
            )

        result = self.executor_skill_gate._validation_decision(
            validation_id=f"{validation_id}-{candidate.candidate_id}",
            task_ids=task_ids,
            task_results=task_results,
        )
        self.artifact_store.store_validation_variant_result(
            validation_id,
            variant,
            result,
            overwrite=True,
        )
        return result

    async def _run_planner_validation_task(
        self,
        *,
        validation_id: str,
        variant: str,
        task_id: str,
        iteration: int,
        planner_skill: str,
        executor_skill: str,
    ) -> _PlannerValidationRun:
        task_metadata = task_metadata_fields(task_id=task_id, task_config=None)
        try:
            benchmark_task = self.benchmark_repo.resolve(task_id)
            task_metadata = task_metadata_fields(
                task_id=task_id,
                task_config=benchmark_task.task_config,
            )
            self.planner.set_skill_context(
                executor_skills=executor_skill,
                skill_refiner=planner_skill,
            )
            skill_texts = [executor_skill] if executor_skill else []
            task_spec = await self.planner.plan_task(
                task_id=task_id,
                base_instruction=benchmark_task.instruction,
                prior_context=None,
                current_skills=skill_texts,
                iteration=iteration,
            )
            trace = await self.executor.execute_task(task_spec, skill_texts)
        except FileNotFoundError as e:
            trace = ExecutionTrace(
                task_id=task_id,
                iteration=iteration,
                status="env_failure",
                error_kind="task_not_found",
                error_detail=str(e),
            )

        trace_path = self.artifact_store.store_validation_trace(
            validation_id,
            variant,
            trace,
            overwrite=True,
        )
        judge_record = None
        if trace.is_usable_feedback_signal:
            judge_record = await self._judge_evolution_reward(
                trace=trace,
                task_metadata=task_metadata,
                trace_path=trace_path,
            )
        reward, reward_source, judge_reward = _validation_reward(
            trace,
            judge_record,
        )
        return _PlannerValidationRun(
            reward=reward,
            reward_source=reward_source,
            verifier_reward=trace.reward,
            judge_reward=judge_reward,
            status=trace.status,
            trace_path=str(trace_path),
            usable=trace.is_usable_feedback_signal,
        )

    def _write_metric(self, record: IterationRecord) -> None:
        """Append an iteration record to metrics.jsonl."""
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(metric_row(record), sort_keys=True) + "\n")

    def _snapshot_and_write_metrics(
        self,
        iteration: int,
        records: list[IterationRecord],
        *,
        coevolution_record: IterationRecord | None = None,
    ) -> None:
        """Snapshot skills and write metric rows against that exact version."""
        current_skill_version = skill_version(iteration)
        self.skill_store.snapshot(iteration, self._snapshots_dir)
        skill_hashes = self._current_skill_hashes()
        records_to_write = list(records)
        if coevolution_record is not None:
            records_to_write.append(coevolution_record)
        for record in records_to_write:
            attach_skill_identity(record, skill_hashes, current_skill_version)
            self._write_metric(record)

    def _current_skill_hashes(self) -> dict[str, str]:
        """Return hashes for current SkillStore contents."""
        return dict(self.skill_store.skill_hashes())

    def _drain_llm_token_events(self) -> list[TokenBudgetEvent]:
        """Collect token telemetry from configured LLM clients."""
        events: list[TokenBudgetEvent] = []
        for llm_client in (
            self.planner.llm_client,
            self.mediator.llm_client,
            self.skill_advisor.llm_client,
            getattr(self, "judge_llm_client", None),
        ):
            if llm_client is not None and hasattr(llm_client, "drain_token_events"):
                events.extend(llm_client.drain_token_events())
        return events


def _valid_planner_reflection_candidates(
    candidates: list[SkillUpdateCandidate],
) -> list[SkillUpdateCandidate]:
    return [
        candidate
        for candidate in candidates
        if not candidate.rejection_reason and candidate.update_kind != "no_update"
    ]


def _validation_reward(
    trace: ExecutionTrace,
    judge_record: JudgeRewardRecord | None,
) -> tuple[float | None, str | None, float | None]:
    if judge_record is None:
        reward_source = "verifier" if trace.reward is not None else None
        return trace.reward, reward_source, None
    metadata = judge_reward_metadata(judge_record)
    return (
        judge_record.judge_reward,
        str(metadata["reward_source"]),
        judge_record.judge_reward,
    )
