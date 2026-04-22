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
from mediated_coevo.config import Config
from mediated_coevo.evolution.compactor import build_planner_signal
from mediated_coevo.evolution.skill_advisor import SkillAdvisor
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.skill import SkillProposal, SkillUpdate
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
                await self._coevolve(iteration)

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

        # Load executor skills for the planner's context
        executor_skill_text = self.skill_store.read_skill("executor") or ""
        planner_skill_text = self.skill_store.read_skill("planner") or None
        benchmark_task = self.benchmark_repo.resolve(task_id)

        self.planner.set_skill_context(
            executor_skills=executor_skill_text,
            skill_refiner=planner_skill_text,
        )

        skill_texts = [executor_skill_text] if executor_skill_text else []

        # 1. PLAN
        logger.info("Step 1: Planner planning task...")
        task_spec = await self.planner.plan_task(
            task_id=task_id,
            base_instruction=benchmark_task.instruction,
            mediator_report=self._previous_report_by_task.get(task_id),
            current_skills=skill_texts,
        )

        # 2. EXECUTE
        logger.info("Step 2: Executor running task...")
        trace = await self.executor.execute_task(task_spec, skill_texts)
        self.artifact_store.store_trace(trace)

        # 3. MEDIATE
        logger.info("Step 3: Mediator processing trace...")
        report = await self.mediator.process_trace(trace, task_spec)
        self.artifact_store.store_report(report)
        feedback: str | None = None if report.withheld else report.content

        # 4. PROPOSE — buffer a skill update proposal (no write)
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
            logger.info("Step 4: Skipped (no feedback).")

        # 5. TAG previous entries with this iteration's reward + backfill buffer
        if iteration > 0:
            if prev_mid := self._prev_mediator_entry_id_by_task.pop(task_id, None):
                self.history_store.tag_outcome_by_id(prev_mid, reward=trace.reward)
            if prev_pid := self._prev_planner_entry_id_by_task.pop(task_id, None):
                self.history_store.tag_outcome_by_id(prev_pid, reward=trace.reward)
            for p in self._proposal_buffer:
                if p.iteration == iteration - 1 and p.task_id == task_id:
                    p.reward = trace.reward

        skill_update = await self._advise_and_patch()

        # Record history entries
        current_report: MediatorReport | None = report if not report.withheld else None
        mediator_entry_id: str | None = None
        planner_entry_id: str | None = None
        if feedback:
            mediator_signal = await self.mediator.compact_feedback(feedback, current_report)
            mediator_entry_id = self.history_store.add(HistoryEntry(
                iteration=iteration,
                agent_role="mediator",
                payload=mediator_signal,
                metadata={"task_id": task_id},
            ))
        if skill_update:
            planner_entry_id = self.history_store.add(HistoryEntry(
                iteration=iteration,
                agent_role="planner",
                payload=build_planner_signal(skill_update),
                metadata={"task_id": task_id},
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
        )
        logger.info(
            "Iteration %d complete: reward=%.2f tokens=%d duration=%.1fs",
            iteration, trace.reward, total_tokens, duration,
        )
        return record

    async def _advise_and_patch(self) -> SkillUpdate | None:
        """If buffer is full, run the advisor and optionally patch the skill.

        Clears the buffer regardless of outcome.
        Returns the committed SkillUpdate if a patch was applied, else None.
        """
        if len(self._proposal_buffer) < self.config.experiment.advisor_buffer_max:
            return None

        logger.info("Advisor reviewing %d proposals...", len(self._proposal_buffer))
        current_skill = self.skill_store.read_skill("executor") or ""
        advisor_feedback = await self.skill_advisor.review(
            current_skill=current_skill,
            proposals=list(self._proposal_buffer),
        )
        self._proposal_buffer.clear()

        if not advisor_feedback:
            logger.info("Advisor rejected — no skill update.")
            return None

        logger.info("Advisor approved — Planner patching skill...")
        contributing_task_ids = ",".join(
            sorted({p.task_id for p in self._proposal_buffer if p.task_id})
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

    async def _coevolve(self, iteration: int) -> None:
        """Co-evolution checkpoint: Mediator + Planner reflect on history."""
        logger.info("=== Co-evolution checkpoint at iteration %d ===", iteration)

        from mediated_coevo.evolution.reflector import Reflector
        reflector = Reflector(self.history_store, self.skill_store)

        # Mediator reflects on reporting history
        new_protocol = await reflector.reflect("mediator", self.mediator.llm_client)
        if new_protocol:
            self.mediator.load_protocol(new_protocol)

        # Planner reflects on skill-edit history
        await reflector.reflect("planner", self.planner.llm_client)

    def _write_metric(self, record: IterationRecord) -> None:
        """Append an iteration record to metrics.jsonl."""
        with open(self._metrics_path, "a") as f:
            f.write(record.model_dump_json() + "\n")
