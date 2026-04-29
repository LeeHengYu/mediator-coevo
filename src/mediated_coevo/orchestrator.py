"""Orchestrator — main iteration loop.

Wires agents together and drives the plan → execute → mediate → update loop.
Triggers co-evolution reflections every N iterations.
"""

from __future__ import annotations

import logging
import random
import time
from pathlib import Path
from typing import TYPE_CHECKING

from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.benchmarks import SkillsBenchRepository
from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.conditions import (
    ConditionName,
    MEDIATOR_EVOLVE_CONDITIONS,
    get_cross_task_prior_context,
    get_prior_context,
)
from mediated_coevo.config import Config
from mediated_coevo.evolution.compactor import build_planner_signal
from mediated_coevo.evolution.skill_advisor import SkillAdvisor
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.skill import (
    AdvisorBatchProvenance,
    ProposalRef,
    SkillProposal,
    SkillUpdate,
)
from mediated_coevo.models.task import TaskSpec
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.token_budget import TokenBudgetEvent
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from mediated_coevo.evolution.reflector import ReflectionResult


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
        self._proposal_buffer: list[SkillProposal] = []

        self._snapshots_dir = experiment_dir / "skills_snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_path = experiment_dir / "metrics.jsonl"
        self._previous_report_by_task: dict[str, MediatorReport] = {}

    async def run_experiment(
        self,
        task_ids: list[str],
        num_iterations: int | None = None,
    ) -> list[IterationRecord]:
        """Run the full experiment loop."""
        if num_iterations is None:
            num_iterations = self.config.experiment.num_iterations
        records: list[IterationRecord] = []

        for iteration in range(num_iterations):
            iteration_records: list[IterationRecord] = []
            for task_id in task_ids:
                logger.info(
                    "=== Iteration %d/%d | Task: %s ===",
                    iteration + 1, num_iterations, task_id,
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

        logger.info("Experiment complete: %d iterations, %d records", num_iterations, len(records))
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

        self.planner.set_skill_context(
            executor_skills=executor_skill_text,
            skill_refiner=planner_skill_text,
        )

        skill_texts = [executor_skill_text] if executor_skill_text else []
        prior_context = await self._build_prior_context(condition, task_id)
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
        try:
            report = await self.mediator.mediate_trace(condition, trace, task_spec)
            if report:
                self.artifact_store.store_report(report)
        finally:
            self.artifact_store.store_trace(trace)

        await self._ask_planner_for_skill_proposal(
            task_id=task_id,
            iteration=iteration,
            executor_skill=executor_skill_text,
            feedback=report.exposed_content if report else None,
        )

        # 5. TAG previous entries with this iteration's reward + backfill buffer.
        self.history_store.tag_pending_outcome(
            task_id,
            trace,
            proposals=self._proposal_buffer,
        )

        skill_update = await self._review_proposals_and_patch_skill(iteration=iteration)
        mediator_entry_id, planner_entry_id = await self._record_history_and_remember_outcome(
            task_id=task_id,
            iteration=iteration,
            condition=condition,
            report=report,
            skill_update=skill_update,
        )

        return self._build_iteration_record(
            task_id=task_id,
            iteration=iteration,
            condition=condition,
            start=start,
            task_spec=task_spec,
            trace=trace,
            report=report,
            skill_update=skill_update,
            mediator_entry_id=mediator_entry_id,
            planner_entry_id=planner_entry_id,
            skill_hashes=skill_hashes,
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
        return IterationRecord(
            iteration=iteration,
            task_id=task_id,
            execution_trace=trace,
            duration_sec=duration,
            llm_token_events=llm_token_events,
            total_tokens=sum(e.total_tokens for e in llm_token_events),
            condition_name=condition,
            cross_task_feedback_enabled=(
                self.config.experiment.allow_cross_task_feedback
            ),
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
        if not feedback or not executor_skill:
            logger.info("Step 4: Skipped (no mediator feedback).")
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
            logger.info("Proposal buffered (buffer size=%d)", len(self._proposal_buffer))
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
        if report:
            mediator_signal = await self.mediator.compact_feedback(report)
            mediator_entry_id = self.history_store.record_signal(
                iteration=iteration,
                agent_role="mediator",
                task_id=task_id,
                condition=condition,
                payload=mediator_signal,
            )

        planner_entry_id = None
        if skill_update:
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
        if report and report.is_exposed:
            self._previous_report_by_task[task_id] = report
        return mediator_entry_id, planner_entry_id

    def _build_iteration_record(
        self,
        *,
        task_id: str,
        iteration: int,
        condition: ConditionName,
        start: float,
        task_spec: TaskSpec,
        trace: ExecutionTrace,
        report: MediatorReport | None,
        skill_update: SkillUpdate | None,
        mediator_entry_id: str | None,
        planner_entry_id: str | None,
        skill_hashes: dict[str, str],
    ) -> IterationRecord:
        duration = time.time() - start
        llm_token_events = self._drain_llm_token_events()
        executor_tokens = trace.token_usage.input_tokens + trace.token_usage.output_tokens
        total_tokens = executor_tokens + sum(e.total_tokens for e in llm_token_events)

        record = IterationRecord(
            iteration=iteration,
            task_id=task_id,
            task_spec=task_spec,
            execution_trace=trace,
            mediator_report=report if report and report.is_exposed else None,
            skill_update=skill_update,
            reward=trace.reward,
            total_tokens=total_tokens,
            llm_token_events=llm_token_events,
            duration_sec=duration,
            mediator_history_entry_id=mediator_entry_id,
            planner_history_entry_id=planner_entry_id,
            condition_name=condition,
            cross_task_feedback_enabled=(
                self.config.experiment.allow_cross_task_feedback
            ),
            skill_hashes=dict(skill_hashes),
        )
        reward_str = f"{trace.reward:.2f}" if trace.reward is not None else "n/a"
        logger.info(
            "Iteration %d complete: condition=%s status=%s reward=%s tokens=%d duration=%.1fs",
            iteration, condition, trace.status, reward_str, total_tokens, duration,
        )
        return record

    async def _build_prior_context(self, condition: ConditionName, task_id: str) -> str | None:
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

    async def _review_proposals_and_patch_skill(
        self,
        *,
        iteration: int,
    ) -> SkillUpdate | None:
        """If buffer is full, run the advisor and optionally patch the skill.

        Clears the buffer regardless of outcome.
        Returns the committed SkillUpdate if a patch was applied, else None.
        """
        if len(self._proposal_buffer) < self.config.experiment.advisor_buffer_max:
            return None

        logger.info("Advisor reviewing %d proposals...", len(self._proposal_buffer))
        current_skill = self.skill_store.read_skill("executor") or ""
        buffered_proposals = list(self._proposal_buffer)
        advisor_feedback = await self.skill_advisor.review(
            current_skill=current_skill,
            proposals=buffered_proposals,
        )
        self._proposal_buffer.clear()

        if not advisor_feedback:
            logger.info("Advisor rejected — no skill update.")
            return None

        logger.info("Advisor approved — Planner patching skill...")
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
        if draft_update:
            old_skill_hash = SkillStore.content_hash(current_skill)
            new_skill_hash = SkillStore.content_hash(draft_update.new_content)
            provenance = AdvisorBatchProvenance(
                batch_id=f"coevo-iter-{iteration:04d}",
                iteration=iteration,
                skill_id="executor",
                task_ids=contributing_tasks,
                base_skill_hash=old_skill_hash,
                decision="approved",
                reason=advisor_feedback,
                rollback_snapshot=self._rollback_snapshot(iteration),
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
        return None

    async def _coevolve(
        self,
        iteration: int,
        condition: ConditionName,
    ) -> IterationRecord | None:
        """Co-evolution checkpoint: Mediator + Planner reflect on history."""
        start = time.time()
        logger.info(
            "=== Co-evolution checkpoint at iteration %d (condition=%s) ===",
            iteration, condition,
        )

        from mediated_coevo.evolution.reflector import Reflector
        reflector = Reflector(
            self.history_store,
            self.skill_store,
            budgets=self.config.budgets,
            condition_name=condition,
        )
        skill_updates: list[SkillUpdate] = []
        reflection_seed = random.randrange(1 << 32)

        # Mediator skill evolution only for learned_mediator
        if condition in MEDIATOR_EVOLVE_CONDITIONS:
            mediator_result = await reflector.reflect(
                "mediator",
                self.mediator.llm_client,
                iteration=iteration,
                selection_seed=reflection_seed,
            )
            if mediator_result:
                mediator_result.provenance.rollback_snapshot = self._rollback_snapshot(iteration)
                self.mediator.load_protocol(mediator_result.new_content)
                skill_updates.append(self._skill_update_from_reflection(mediator_result))
        else:
            logger.info("Mediator skill evolution skipped (condition=%s).", condition)

        # Planner reflects on skill-edit history
        planner_result = await reflector.reflect(
            "planner",
            self.planner.llm_client,
            iteration=iteration,
            selection_seed=reflection_seed + 1,
        )
        if planner_result:
            planner_result.provenance.rollback_snapshot = self._rollback_snapshot(iteration)
            skill_updates.append(self._skill_update_from_reflection(planner_result))
        llm_token_events = self._drain_llm_token_events()
        if not llm_token_events and not skill_updates:
            return None
        return self._build_coevolution_record(
            iteration=iteration,
            condition=condition,
            start=start,
            llm_token_events=llm_token_events,
            skill_updates=skill_updates,
        )

    def _write_metric(self, record: IterationRecord) -> None:
        """Append an iteration record to metrics.jsonl."""
        with open(self._metrics_path, "a") as f:
            f.write(record.model_dump_json() + "\n")

    def _snapshot_and_write_metrics(
        self,
        iteration: int,
        records: list[IterationRecord],
        *,
        coevolution_record: IterationRecord | None = None,
    ) -> None:
        """Snapshot skills and write metric rows against that exact version."""
        skill_version = self._skill_version(iteration)
        self.skill_store.snapshot(iteration, self._snapshots_dir)
        skill_hashes = self._current_skill_hashes()
        records_to_write = list(records)
        if coevolution_record:
            records_to_write.append(coevolution_record)
        for record in records_to_write:
            self._attach_skill_identity(record, skill_hashes, skill_version)
            self._write_metric(record)

    @staticmethod
    def _skill_version(iteration: int) -> str:
        """Return the run-local skill snapshot label for an iteration."""
        return f"iter_{iteration:04d}"

    @staticmethod
    def _rollback_snapshot(iteration: int) -> str | None:   
        """Return the prior snapshot label that can restore pre-update state."""
        if iteration <= 0:
            return None
        return Orchestrator._skill_version(iteration - 1)

    @staticmethod
    def _skill_update_from_reflection(result: "ReflectionResult") -> SkillUpdate:
        """Convert a committed reflection result into a metrics skill update."""
        new_skill_hash = SkillStore.content_hash(result.new_content)
        return SkillUpdate(
            skill_id=result.skill_id,
            task_id=",".join(result.provenance.task_ids),
            old_content=result.old_content,
            new_content=result.new_content,
            reasoning=result.provenance.reason,
            iteration=result.provenance.iteration,
            old_skill_hash=result.provenance.base_skill_hash,
            new_skill_hash=new_skill_hash,
            provenance=result.provenance,
        )

    def _current_skill_hashes(self) -> dict[str, str]:
        """Return skill hashes when the configured store supports them."""
        skill_hashes = getattr(self.skill_store, "skill_hashes", None)
        if not callable(skill_hashes):
            return {}
        return dict(skill_hashes())

    @staticmethod
    def _attach_skill_identity(
        record: IterationRecord,
        skill_hashes: dict[str, str],
        skill_version: str,
    ) -> None:
        """Attach snapshot identity to a metric record before serialization."""
        if not record.skill_hashes:
            record.skill_hashes = dict(skill_hashes)
        record.skill_version = skill_version
        skill_updates = list(record.skill_updates)
        if record.skill_update:
            skill_updates.append(record.skill_update)
        for skill_update in skill_updates:
            skill_update.skill_version = skill_version

    def _drain_llm_token_events(self) -> list[TokenBudgetEvent]:
        """Collect token telemetry from configured LLM clients."""
        events: list[TokenBudgetEvent] = []
        for llm_client in (
            getattr(self.planner, "llm_client", None),
            getattr(self.mediator, "llm_client", None),
            getattr(self.skill_advisor, "llm_client", None),
        ):
            drain_token_events = getattr(llm_client, "drain_token_events", None)
            if callable(drain_token_events):
                events.extend(drain_token_events())
        return events

    def _build_coevolution_record(
        self,
        *,
        iteration: int,
        condition: ConditionName,
        start: float,
        llm_token_events: list[TokenBudgetEvent],
        skill_updates: list[SkillUpdate] | None = None,
    ) -> IterationRecord:
        """Build a metrics-only row for co-evolution LLM telemetry."""
        return IterationRecord(
            iteration=iteration,
            task_id="__coevolution__",
            skill_updates=list(skill_updates or []),
            total_tokens=sum(e.total_tokens for e in llm_token_events),
            llm_token_events=llm_token_events,
            duration_sec=time.time() - start,
            condition_name=condition,
            cross_task_feedback_enabled=(
                self.config.experiment.allow_cross_task_feedback
            ),
            skill_hashes=self._current_skill_hashes(),
        )
