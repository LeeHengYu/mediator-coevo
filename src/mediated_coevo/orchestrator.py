"""Orchestrator — main iteration loop.

Wires agents together and drives the plan → execute → mediate → update loop.
Triggers co-evolution reflections every N iterations.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.benchmarks import SkillsBenchRepository
from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.conditions import (
    ConditionName,
    MEDIATOR_CONDITIONS,
    MEDIATOR_EVOLVE_CONDITIONS,
    get_cross_task_prior_context,
    get_prior_context,
)
from mediated_coevo.config import Config
from mediated_coevo.evolution.compactor import build_planner_signal
from mediated_coevo.evolution.skill_advisor import SkillAdvisor
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.skill import SkillProposal, SkillUpdate
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import HistoryEntry, HistoryStore
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

        self._snapshots_dir = experiment_dir / "skills_snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_path = experiment_dir / "metrics.jsonl"
        self._previous_report_by_task: dict[str, MediatorReport] = {}
        self._prev_mediator_entry_id_by_task: dict[str, str] = {}
        self._prev_planner_entry_id_by_task: dict[str, str] = {}

    async def run_experiment(
        self,
        task_ids: list[str],
        num_iterations: int | None = None,
    ) -> list[IterationRecord]:
        """Run the full experiment loop."""
        num_iterations = num_iterations or self.config.experiment.num_iterations
        records: list[IterationRecord] = []

        for iteration in range(num_iterations):
            for task_id in task_ids:
                logger.info(
                    "=== Iteration %d/%d | Task: %s ===",
                    iteration + 1, num_iterations, task_id,
                )
                record = await self._run_iteration(task_id, iteration)
                records.append(record)
                self._write_metric(record)

            # Co-evolution checkpoint
            if (iteration + 1) % self.config.experiment.coevo_interval == 0:
                await self._coevolve(iteration, self.config.experiment.condition_name)

            # Snapshot skills
            self.skill_store.snapshot(iteration, self._snapshots_dir)

        logger.info("Experiment complete: %d iterations, %d records", num_iterations, len(records))
        return records

    async def _run_iteration(
        self,
        task_id: str,
        iteration: int,
    ) -> IterationRecord:
        start = time.time()
        condition = self.config.experiment.condition_name

        # Load executor skills for the planner's context
        executor_skill_text = self.skill_store.read_skill("executor") or ""
        planner_skill_text = self.skill_store.read_skill("planner") or None
        try:
            benchmark_task = self.benchmark_repo.resolve(task_id)
        except FileNotFoundError as e:
            duration = time.time() - start
            trace = ExecutionTrace(
                task_id=task_id,
                iteration=iteration,
                duration_sec=duration,
                exit_code=-1,
                status="env_failure",
                error_kind="task_not_found",
                error_detail=str(e),
            )
            self.artifact_store.store_trace(trace)
            self._tag_previous_iteration(iteration, task_id, trace)
            logger.warning(
                "Iteration %d skipped before planning: task=%s status=%s error_kind=%s",
                iteration, task_id, trace.status, trace.error_kind,
            )
            return IterationRecord(
                iteration=iteration,
                task_id=task_id,
                execution_trace=trace,
                duration_sec=duration,
                condition_name=condition,
                cross_task_feedback_enabled=(
                    self.config.experiment.allow_cross_task_feedback
                ),
            )

        self.planner.set_skill_context(
            executor_skills=executor_skill_text,
            skill_refiner=planner_skill_text,
        )

        skill_texts = [executor_skill_text] if executor_skill_text else []

        # Determine prior context for planner based on condition.
        prior_context = await self._build_prior_context(condition, task_id)

        # 1. PLAN
        logger.info("Step 1: Planner planning task (condition=%s)...", condition)
        task_spec = await self.planner.plan_task(
            task_id=task_id,
            base_instruction=benchmark_task.instruction,
            prior_context=prior_context,
            current_skills=skill_texts,
        )

        # 2. EXECUTE
        logger.info("Step 2: Executor running task...")
        trace = await self.executor.execute_task(task_spec, skill_texts)
        self.artifact_store.store_trace(trace)

        # 3. MEDIATE — only for mediator conditions, and only if the trace is
        # usable. Env-failure traces have unreliable reward/stderr signal and
        # would poison the mediator's compaction LLM call + history payload.
        report: MediatorReport | None = None
        current_report: MediatorReport | None = None
        feedback: str | None = None

        trace_usable = trace.status == "ok" and trace.reward is not None
        if condition not in MEDIATOR_CONDITIONS:
            logger.info("Step 3: Skipped (condition=%s does not use mediator).", condition)
        elif not trace_usable:
            logger.warning(
                "Step 3: Skipped — trace unusable (status=%s error_kind=%s reward=%s)",
                trace.status, trace.error_kind, trace.reward,
            )
        else:
            logger.info("Step 3: Mediator processing trace...")
            report = await self.mediator.process_trace(trace, task_spec)
            self.artifact_store.store_report(report)
            current_report = report if not report.withheld else None
            feedback = None if report.withheld else report.content

        # 4. PROPOSE — buffer a skill update proposal (no write); only when mediator feedback exists
        if feedback and executor_skill_text:
            logger.info("Step 4: Planner proposing skill update...")
            edit_history = self.history_store.query(
                agent_role="planner",
                tagged_only=True,
            )
            proposal = await self.planner.propose_skill_update(
                current_skill_content=executor_skill_text,
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
        else:
            logger.info("Step 4: Skipped (no mediator feedback).")

        # 5. TAG previous entries with this iteration's reward + backfill buffer.
        self._tag_previous_iteration(iteration, task_id, trace)

        skill_update = await self._advise_and_patch()

        # Record history entries (mediator conditions only for mediator signal)
        mediator_entry_id: str | None = None
        planner_entry_id: str | None = None
        if feedback and current_report:
            mediator_signal = await self.mediator.compact_feedback(feedback, current_report)
            mediator_entry_id = self.history_store.add(HistoryEntry(
                iteration=iteration,
                agent_role="mediator",
                payload=mediator_signal,
                metadata={"task_id": task_id, "condition": condition},
            ))
        if skill_update:
            planner_entry_id = self.history_store.add(HistoryEntry(
                iteration=iteration,
                agent_role="planner",
                payload=build_planner_signal(skill_update),
                metadata={"task_id": task_id, "condition": condition},
            ))

        # Carry forward entry IDs and report for the next iteration's context.
        if mediator_entry_id:
            self._prev_mediator_entry_id_by_task[task_id] = mediator_entry_id
        if planner_entry_id:
            self._prev_planner_entry_id_by_task[task_id] = planner_entry_id
        if current_report:
            self._previous_report_by_task[task_id] = current_report

        duration = time.time() - start
        total_tokens = trace.token_usage.input_tokens + trace.token_usage.output_tokens

        record = IterationRecord(
            iteration=iteration,
            task_id=task_id,
            task_spec=task_spec,
            execution_trace=trace,
            mediator_report=current_report,
            skill_update=skill_update,
            reward=trace.reward,
            total_tokens=total_tokens,
            duration_sec=duration,
            mediator_history_entry_id=mediator_entry_id,
            planner_history_entry_id=planner_entry_id,
            condition_name=condition,
            cross_task_feedback_enabled=(
                self.config.experiment.allow_cross_task_feedback
            ),
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
        )
        if not self.config.experiment.allow_cross_task_feedback:
            return prior_context

        cross_context = await get_cross_task_prior_context(
            condition=condition,
            task_id=task_id,
            artifact_store=self.artifact_store,
            previous_reports_by_task=self._previous_report_by_task,
            llm_client=llm_client,
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

    def _tag_previous_iteration(
        self,
        iteration: int,
        task_id: str,
        trace: ExecutionTrace,
    ) -> None:
        """Pop the carry-forward entry IDs and tag them with this iteration's reward.

        The pop must happen on every iteration (including env failures), so a
        stale entry from iteration N can never be tagged with iteration N+2's
        reward across an intervening env_failure. We only WRITE the reward
        when the trace is usable; otherwise the entry is dropped untagged.
        """
        if iteration <= 0:
            return

        prev_mid = self._prev_mediator_entry_id_by_task.pop(task_id, None)
        prev_pid = self._prev_planner_entry_id_by_task.pop(task_id, None)

        if trace.reward is None or trace.status != "ok":
            if prev_mid or prev_pid:
                logger.info(
                    "Dropping carry-forward entry IDs untagged for task=%s "
                    "(trace status=%s reward=%s)",
                    task_id, trace.status, trace.reward,
                )
            return

        if prev_mid:
            self.history_store.tag_outcome_by_id(prev_mid, reward=trace.reward)
        if prev_pid:
            self.history_store.tag_outcome_by_id(prev_pid, reward=trace.reward)
        for p in self._proposal_buffer:
            if p.iteration == iteration - 1 and p.task_id == task_id:
                p.reward = trace.reward

    async def _advise_and_patch(self) -> SkillUpdate | None:
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
        contributing_task_ids = ",".join(
            sorted({p.task_id for p in buffered_proposals if p.task_id})
        )
        edit_history = self.history_store.query(
            agent_role="planner",
            tagged_only=True,
        )
        proposal = await self.planner.propose_skill_update(
            current_skill_content=current_skill,
            feedback=advisor_feedback,
            edit_history=edit_history,
            task_id=contributing_task_ids,
            iteration=self.planner.step,
        )
        if proposal:
            skill_update = SkillUpdate(
                skill_id="executor",
                task_id=contributing_task_ids,
                old_content=proposal.old_content,
                new_content=proposal.new_content,
                reasoning=proposal.reasoning,
                iteration=proposal.iteration,
            )
            self.skill_store.write_skill("executor", skill_update.new_content)
            logger.info("Skill patched and written.")
            return skill_update
        return None

    async def _coevolve(self, iteration: int, condition: str) -> None:
        """Co-evolution checkpoint: Mediator + Planner reflect on history."""
        logger.info(
            "=== Co-evolution checkpoint at iteration %d (condition=%s) ===",
            iteration, condition,
        )

        from mediated_coevo.evolution.reflector import Reflector
        reflector = Reflector(self.history_store, self.skill_store)

        # Mediator skill evolution only for learned_mediator
        if condition in MEDIATOR_EVOLVE_CONDITIONS:
            new_protocol = await reflector.reflect("mediator", self.mediator.llm_client)
            if new_protocol:
                self.mediator.load_protocol(new_protocol)
        else:
            logger.info("Mediator skill evolution skipped (condition=%s).", condition)

        # Planner reflects on skill-edit history
        await reflector.reflect("planner", self.planner.llm_client)

    def _write_metric(self, record: IterationRecord) -> None:
        """Append an iteration record to metrics.jsonl."""
        with open(self._metrics_path, "a") as f:
            f.write(record.model_dump_json() + "\n")
