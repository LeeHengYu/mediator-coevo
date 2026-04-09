"""Orchestrator — main iteration loop.

Wires agents together according to the selected FeedbackCondition.
Drives iterations, triggers co-evolution reflections every N iterations,
and delegates to stores for persistence.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.conditions import FeedbackCondition
from mediated_coevo.config import Config
from mediated_coevo.evolution.compactor import (
    build_planner_signal,
    deterministic_mediator_signal,
)
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.skill import SkillUpdate
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.history_store import AgentRole, HistoryEntry, HistoryStore
from mediated_coevo.stores.skill_store import SkillStore

logger = logging.getLogger(__name__)


class Orchestrator:
    """Runs the plan → execute → feedback → update loop."""

    def __init__(
        self,
        planner: PlannerAgent,
        executor: ExecutorAgent,
        mediator: MediatorAgent | None,
        condition: FeedbackCondition,
        skill_store: SkillStore,
        artifact_store: ArtifactStore,
        history_store: HistoryStore,
        config: Config,
        experiment_dir: Path,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.mediator = mediator
        self.condition = condition
        self.skill_store = skill_store
        self.artifact_store = artifact_store
        self.history_store = history_store
        self.config = config
        self.experiment_dir = experiment_dir

        self._snapshots_dir = experiment_dir / "skills_snapshots"
        self._snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._metrics_path = experiment_dir / "metrics.jsonl"
        self._previous_report: MediatorReport | None = None

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
                    "=== Iteration %d/%d | Task: %s | Condition: %s ===",
                    iteration + 1, num_iterations, task_id, self.condition.name,
                )
                record = await self._run_iteration(task_id, iteration)
                records.append(record)
                self._write_metric(record)

            # Co-evolution checkpoint
            if (iteration + 1) % self.config.experiment.coevo_interval == 0:
                if self.condition.supports_coevolution():
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

        self.planner.set_skill_context(
            executor_skills=executor_skill_text,
            skill_refiner=planner_skill_text,
        )

        # Executor sees only its own skill
        skill_texts = [executor_skill_text] if executor_skill_text else []

        # 1. PLAN
        logger.info("Step 1: Planner planning task...")
        task_spec = await self.planner.plan_task(
            task_id=task_id,
            mediator_report=self._previous_report,
            current_skills=skill_texts,
        )

        # 2. EXECUTE
        logger.info("Step 2: Executor running task...")
        trace = await self.executor.execute_task(task_spec, skill_texts)
        self.artifact_store.store_trace(trace)

        # 3. FEEDBACK (condition-dependent)
        logger.info("Step 3: Producing feedback via %s...", self.condition.name)
        feedback = await self.condition.produce_feedback(
            trace=trace,
            task_context=task_spec,
            artifact_store=self.artifact_store,
            mediator=self.mediator,
        )

        # 4. UPDATE SKILL — only when there is feedback to act on
        skill_update: SkillUpdate | None = None
        if feedback and executor_skill_text:
            logger.info("Step 4: Planner deciding skill update...")
            edit_history = self.history_store.query(
                agent_role=AgentRole.PLANNER,
                tagged_only=True,
            )
            skill_update = await self.planner.update_skill(
                current_skill_content=executor_skill_text,
                feedback=feedback,
                edit_history=edit_history,
            )
            if skill_update:
                skill_update.exploration = _should_explore(self.config.experiment.epsilon)
                self.skill_store.write_skill("executor", skill_update.new_content)
                logger.info("Skill updated (exploration=%s)", skill_update.exploration)
            else:
                logger.info("Planner decided: no skill update needed.")
        else:
            logger.info("Step 4: Skipped (no feedback).")

        # 5. TAG previous entries with this iteration's reward
        if iteration > 0:
            self.history_store.tag_outcome(
                iteration=iteration - 1,
                agent_role=AgentRole.MEDIATOR,
                reward=trace.reward,
            )
            self.history_store.tag_outcome(
                iteration=iteration - 1,
                agent_role=AgentRole.PLANNER,
                reward=trace.reward,
            )

        # Fetch the underlying MediatorReport (learned_mediator only) once, so
        # both the history payload and `_previous_report` share the same object.
        current_report: MediatorReport | None = None
        if feedback and self.condition.uses_mediator_reports:
            reports = self.artifact_store.query_reports(task_id=task_id, recent=1)
            current_report = reports[0] if reports else None

        # Record history entries with structured payloads. When a MediatorAgent
        # is present, it owns its own compaction (and may make one extra LLM
        # call on its own client for long reports). For non-mediator conditions
        # we use the deterministic path — no LLM call.
        if feedback:
            if self.mediator:
                mediator_signal = await self.mediator.compact_feedback(
                    feedback, current_report
                )
            else:
                mediator_signal = deterministic_mediator_signal(
                    feedback, current_report
                )
            self.history_store.add(HistoryEntry(
                iteration=iteration,
                agent_role=AgentRole.MEDIATOR,
                payload=mediator_signal,
                metadata={"task_id": task_id},
            ))
        if skill_update:
            self.history_store.add(HistoryEntry(
                iteration=iteration,
                agent_role=AgentRole.PLANNER,
                payload=build_planner_signal(skill_update),
                metadata={"task_id": task_id},
            ))

        # Carry the current report into the next iteration's Planner context.
        self._previous_report = current_report

        duration = time.time() - start
        total_tokens = trace.token_usage.input_tokens + trace.token_usage.output_tokens

        record = IterationRecord(
            iteration=iteration,
            task_id=task_id,
            condition=self.condition.name,
            task_spec=task_spec,
            execution_trace=trace,
            mediator_report=self._previous_report,
            skill_update=skill_update,
            reward=trace.reward,
            total_tokens=total_tokens,
            duration_sec=duration,
        )
        logger.info(
            "Iteration %d complete: reward=%.2f tokens=%d duration=%.1fs",
            iteration, trace.reward, total_tokens, duration,
        )
        return record

    async def _coevolve(self, iteration: int) -> None:
        """Co-evolution checkpoint: Mediator + Planner reflect on history."""
        logger.info("=== Co-evolution checkpoint at iteration %d ===", iteration)

        from mediated_coevo.evolution.reflector import Reflector
        reflector = Reflector(self.history_store, self.skill_store)

        # Mediator reflects on reporting history
        if self.mediator:
            new_protocol = await reflector.reflect(AgentRole.MEDIATOR, self.mediator.llm_client)
            if new_protocol:
                self.mediator.load_protocol(new_protocol)

        # Planner reflects on skill-edit history
        await reflector.reflect(AgentRole.PLANNER, self.planner.llm_client)

    def _write_metric(self, record: IterationRecord) -> None:
        """Append an iteration record to metrics.jsonl."""
        with open(self._metrics_path, "a") as f:
            f.write(record.model_dump_json() + "\n")


def _should_explore(epsilon: float) -> bool:
    """Epsilon-greedy: return True with probability epsilon."""
    import random
    return random.random() < epsilon