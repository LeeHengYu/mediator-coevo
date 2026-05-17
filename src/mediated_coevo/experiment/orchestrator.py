"""Orchestrator — main iteration loop.

Wires agents together and drives the plan → execute → mediate → update loop.
Triggers co-evolution reflections every N iterations.
"""

from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path

from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.analysis.metrics import metric_row
from mediated_coevo.benchmarks import SkillsBenchRepository
from mediated_coevo.core.config import Config
from mediated_coevo.evolution.compactor import build_planner_signal
from mediated_coevo.evolution.executor_skill_gate import ExecutorSkillGate
from mediated_coevo.evolution.reflector import Reflector
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
    rollback_snapshot,
    skill_update_from_reflection,
    skill_version,
    task_metadata_fields,
)
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.skill import (
    SkillProposal,
    SkillUpdate,
)
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.runtime.token_budget import TokenBudgetEvent
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryStore
from mediated_coevo.stores.skill_store import SkillStore

logger = logging.getLogger(__name__)


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
        self.executor_skill_gate = ExecutorSkillGate(
            config=config,
            skill_store=skill_store,
            history_store=history_store,
            planner=planner,
            skill_advisor=skill_advisor,
            executor=executor,
            benchmark_repo=benchmark_repo,
            artifact_store=artifact_store,
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
        try:
            report = await self.mediator.mediate_trace(condition, trace, task_spec)
            if report is not None:
                self.artifact_store.store_report(report)
        finally:
            self.artifact_store.store_trace(trace)

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
            planner_result = await reflector.reflect(
                "planner",
                self.planner.llm_client,
                iteration=iteration,
                selection_seed=reflection_seed + 1,
            )
            if planner_result:
                planner_result.provenance.rollback_snapshot = rollback_snapshot(
                    iteration
                )
                skill_updates.append(skill_update_from_reflection(planner_result))
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
        ):
            events.extend(llm_client.drain_token_events())
        return events
