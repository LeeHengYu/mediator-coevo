"""Orchestrator for the fixed-skill plan → execute → mediate loop."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from mediated_coevo.agents.executor import ExecutorAgent
from mediated_coevo.agents.mediator import MediatorAgent
from mediated_coevo.agents.planner import PlannerAgent
from mediated_coevo.agents.prompt_context import PlannerPriorContextBundle
from mediated_coevo.analysis.judge_rewards import (
    append_judge_reward_record,
    judge_reward_for_trace,
    judge_reward_metadata,
)
from mediated_coevo.analysis.metrics import metric_row
from mediated_coevo.benchmarks import TaskPackageRepository
from mediated_coevo.core.config import Config
from mediated_coevo.core.utils import format_optional_reward
from mediated_coevo.diffusion import (
    DiffusedRecord,
    DiffusionArtifact,
    DiffusionStore,
    DiffusionSubscription,
    LangChainGraphPolicy,
    TaskGraphSnapshot,
    diffusion_channel_for_artifact,
    emit_diffusion_artifacts,
    render_diffusion_subscriptions,
    select_capped_broadcast_subscriptions,
    select_random_k_subscriptions,
    select_top_k_similarity_subscriptions,
)
from mediated_coevo.execution.adapters import portable_execution_trace
from mediated_coevo.execution.models import (
    ContextPack,
    TaskProfile,
    redact_sensitive_data,
)
from mediated_coevo.experiment.conditions import (
    ConditionName,
    get_cross_task_prior_context,
    get_prior_context,
)
from mediated_coevo.experiment.records import (
    TaskMetadataFields,
    build_iteration_record,
    build_missing_task_record,
    task_metadata_fields,
)
from mediated_coevo.models.iteration import IterationRecord
from mediated_coevo.models.judge import JudgeRewardRecord
from mediated_coevo.models.report import MediatorReport
from mediated_coevo.models.trace import ExecutionTrace
from mediated_coevo.runtime.token_budget import TokenBudgetEvent
from mediated_coevo.stores.artifact_store import ArtifactStore
from mediated_coevo.stores.skill_store import SkillStore

logger = logging.getLogger(__name__)

_EXPLICIT_CONTEXT_UNSET = object()


def _validate_portable_filename_component(value: str, *, label: str) -> None:
    if (
        not value
        or value != value.strip()
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"{label} must be a portable filename component")


class Orchestrator:
    """Runs the fixed-skill plan → execute → mediate loop."""

    def __init__(
        self,
        planner: PlannerAgent,
        executor: ExecutorAgent,
        mediator: MediatorAgent,
        skill_store: SkillStore,
        artifact_store: ArtifactStore,
        benchmark_repo: TaskPackageRepository,
        config: Config,
        experiment_dir: Path,
        judge_llm_client: Any | None = None,
    ) -> None:
        self.planner = planner
        self.executor = executor
        self.mediator = mediator
        self.skill_store = skill_store
        self.artifact_store = artifact_store
        self._diffusion_store = DiffusionStore(experiment_dir / "diffusion")
        self.benchmark_repo = benchmark_repo
        self.config = config
        self.experiment_dir = experiment_dir
        self.judge_llm_client = judge_llm_client

        self._metrics_path = experiment_dir / "metrics.jsonl"
        self._previous_report_by_task: dict[str, MediatorReport] = {}
        self._released_cross_task_reports_by_task: dict[str, MediatorReport] = {}
        self._staged_cross_task_reports_by_task: dict[str, MediatorReport] = {}
        self._previous_reward_by_task: dict[str, float] = {}
        self._prior_context_by_target: dict[tuple[str, int], dict[str, Any]] = {}
        self._diffusion_context_by_target: dict[tuple[str, int], dict[str, Any]] = {}
        self._explicit_execution_provenance_by_key: dict[
            tuple[str, int],
            dict[str, Any],
        ] = {}
        self._diffusion_sub_board: dict[
            tuple[int, str],
            list[DiffusionSubscription],
        ] = {}
        self.freeze_diffusion_artifact_store = False
        self.preloaded_diffusion_artifact_store_path: str | None = None
        self.preloaded_diffusion_artifact_store_count = 0
        self._diffusion_prepared_iterations: set[int] = set()
        self._langchain_graph_prepared_targets: set[tuple[int, str]] = set()
        self._langchain_graph_policy: Any | None = None
        self._diffusion_snapshot_by_iteration: dict[int, TaskGraphSnapshot] = {}
        self._diffusion_target_task_ids: list[str] = []

    @property
    def diffusion_store(self) -> DiffusionStore:
        """Return the durable diffusion store used by this runtime."""
        self._ensure_diffusion_runtime_state()
        return self._diffusion_store

    async def run_experiment(
        self,
        task_ids: list[str],
        num_iterations: int | None = None,
    ) -> list[IterationRecord]:
        """Run the full experiment loop."""
        if num_iterations is None:
            num_iterations = self.config.experiment.num_iterations
        if self.config.experiment.benchmark_selection.family is not None:
            return await self._run_task_stream(task_ids)
        records: list[IterationRecord] = []
        self._diffusion_target_task_ids = list(task_ids)
        self._diffusion_sub_board.clear()
        self._diffusion_prepared_iterations.clear()
        self._langchain_graph_prepared_targets.clear()
        self._diffusion_snapshot_by_iteration.clear()

        for iteration in range(num_iterations):
            self._release_staged_cross_task_reports()
            for task_id in task_ids:
                logger.info(
                    "=== Iteration %d/%d | Task: %s ===",
                    iteration + 1,
                    num_iterations,
                    task_id,
                )
                record = await self._run_iteration(task_id, iteration)
                records.append(record)
                self._attach_diffusion_artifact_store_metrics(record)
                self._write_metric(record)

        logger.info(
            "Experiment complete: %d iterations, %d records",
            num_iterations,
            len(records),
        )
        return records

    async def _run_task_stream(
        self,
        task_ids: list[str],
    ) -> list[IterationRecord]:
        """Run one externally ordered task stream."""
        records: list[IterationRecord] = []
        self._diffusion_target_task_ids = list(task_ids)
        self._diffusion_sub_board.clear()
        self._diffusion_prepared_iterations.clear()
        self._langchain_graph_prepared_targets.clear()
        self._diffusion_snapshot_by_iteration.clear()
        for iteration, task_id in enumerate(task_ids):
            self._release_staged_cross_task_reports()
            logger.info(
                "=== Stream task %d/%d | Task: %s ===",
                iteration + 1,
                len(task_ids),
                task_id,
            )
            record = await self._run_iteration(task_id, iteration)
            records.append(record)
            self._attach_diffusion_artifact_store_metrics(record)
            self._write_metric(record)

        logger.info(
            "Task stream complete: %d records",
            len(records),
        )
        return records

    def _release_staged_cross_task_reports(self) -> None:
        """Promote cross-task reports only at iteration boundaries."""
        if not self._staged_cross_task_reports_by_task:
            return
        self._released_cross_task_reports_by_task.update(
            self._staged_cross_task_reports_by_task
        )
        self._staged_cross_task_reports_by_task.clear()

    async def execute_task_with_context(
        self,
        *,
        task_id: str,
        position: int,
        context: ContextPack,
        task: TaskProfile,
    ) -> IterationRecord:
        """Execute one task using the caller's complete context pack.

        This is the sample-runtime seam. It bypasses all legacy prior-context
        discovery and prevents the legacy loop from emitting transfer artifacts;
        the causal sample state machine performs its own validated bank update.
        """
        return await self._run_iteration(
            task_id,
            position,
            _explicit_context=context,
            _explicit_task=task,
        )

    def take_explicit_execution_provenance(
        self,
        *,
        task_id: str,
        position: int,
    ) -> dict[str, Any]:
        """Consume judge/verifier provenance produced by one explicit execution."""
        store = getattr(self, "_explicit_execution_provenance_by_key", None)
        if store is None:
            return {}
        return store.pop(
            (task_id, position),
            {},
        )

    async def _run_iteration(
        self,
        task_id: str,
        iteration: int,
        *,
        _explicit_context: ContextPack | object = _EXPLICIT_CONTEXT_UNSET,
        _explicit_task: TaskProfile | None = None,
    ) -> IterationRecord:
        start = time.time()
        condition = self.config.experiment.condition_name
        skill_hashes = self._current_skill_hashes()
        executor_skill_text = self.skill_store.read_skill("executor") or ""
        planner_skill_text = self.skill_store.read_skill("planner") or None

        if _explicit_task is not None and _explicit_task.task_id != task_id:
            raise ValueError("explicit task profile does not match task_id")
        if _explicit_task is not None:
            task_instruction = _explicit_task.instruction
            task_config = _explicit_task.task_config
        else:
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
            task_instruction = benchmark_task.instruction
            task_config = benchmark_task.task_config

        task_metadata = task_metadata_fields(
            task_id=task_id,
            task_config=task_config,
        )

        self.planner.set_skill_context(
            executor_skills=executor_skill_text,
            planner_skill=planner_skill_text,
        )

        skill_texts = [executor_skill_text] if executor_skill_text else []
        if _explicit_context is _EXPLICIT_CONTEXT_UNSET:
            prior_context = await self._build_prior_context(
                condition,
                task_id,
                current_iteration=iteration,
            )
        else:
            assert isinstance(_explicit_context, ContextPack)
            prior_context = _explicit_context.text
        logger.info("Step 1: Planner planning task (condition=%s)...", condition)
        task_spec = await self.planner.plan_task(
            task_id=task_id,
            base_instruction=task_instruction,
            prior_context=prior_context,
            current_skills=skill_texts,
            iteration=iteration,
        )
        if _explicit_context is not _EXPLICIT_CONTEXT_UNSET and (
            task_spec.task_id != task_id or task_spec.iteration != iteration
        ):
            raise ValueError(
                "explicit-context planner returned a different task occurrence"
            )

        logger.info("Step 2: Executor running task...")
        trace = await self.executor.execute_task(task_spec, skill_texts)
        explicit_external_refs: tuple[dict[str, Any], ...] = ()
        if _explicit_context is not _EXPLICIT_CONTEXT_UNSET:
            if trace.task_id != task_id or trace.iteration != iteration:
                raise ValueError(
                    "explicit-context executor returned a different task occurrence"
                )
            trace = ExecutionTrace.model_validate(
                redact_sensitive_data(trace.model_dump(mode="python"))
            )
            trace, _, explicit_external_refs = portable_execution_trace(
                trace,
                workspace=self.experiment_dir,
            )

        report = None
        trace_path = self.artifact_store.store_trace(trace)

        outcome_reward = None
        outcome_metadata = None
        if trace.is_usable_feedback_signal:
            judge_record = await self._judge_task_reward(
                trace=trace,
                task_metadata=task_metadata,
                trace_path=trace_path,
            )
            if judge_record is not None:
                if not judge_record.metadata.get("judge_reward_fallback"):
                    append_judge_reward_record(self.experiment_dir, judge_record)
                outcome_reward = judge_record.judge_reward
                outcome_metadata = judge_reward_metadata(judge_record)
                logger.info(
                    "Judge reward after task run: task=%s iteration=%d "
                    "verifier_reward=%s judge_reward=%s reward_source=%s",
                    task_id,
                    _display_iteration(trace.iteration),
                    format_optional_reward(trace.reward),
                    format_optional_reward(judge_record.judge_reward),
                    outcome_metadata["reward_source"],
                )
        else:
            logger.info(
                "Judge reward skipped after task run: task=%s iteration=%d "
                "status=%s reward=%s",
                task_id,
                _display_iteration(trace.iteration),
                trace.status,
                format_optional_reward(trace.reward),
            )

        if _explicit_context is _EXPLICIT_CONTEXT_UNSET or not hasattr(
            self.mediator,
            "_artifact_store",
        ):
            report = await self.mediator.mediate_trace(condition, trace, task_spec)
        else:
            # The explicit sample seam receives all transfer context from its
            # ContextPack; same-task summaries are legacy hidden state.
            mediator_artifact_store = self.mediator._artifact_store
            self.mediator._artifact_store = None
            try:
                report = await self.mediator.mediate_trace(condition, trace, task_spec)
            finally:
                self.mediator._artifact_store = mediator_artifact_store
        if report is not None and _explicit_context is not _EXPLICIT_CONTEXT_UNSET:
            if report.task_id != task_id or report.iteration != iteration:
                raise ValueError(
                    "explicit-context mediator returned a different task occurrence"
                )
            _validate_portable_filename_component(
                report.report_id,
                label="mediator report_id",
            )
            report = MediatorReport.model_validate(
                redact_sensitive_data(report.model_dump(mode="python"))
            )
        if report is not None:
            self.artifact_store.store_report(report)
            if _explicit_context is _EXPLICIT_CONTEXT_UNSET and report.is_exposed:
                self._previous_report_by_task[task_id] = report
                self._staged_cross_task_reports_by_task[task_id] = report

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
            skill_hashes=skill_hashes,
            task_metadata=task_metadata,
            llm_token_events=llm_token_events,
            config=self.config,
            previous_reward_by_task=(
                self._previous_reward_by_task
                if _explicit_context is _EXPLICIT_CONTEXT_UNSET
                else {}
            ),
        )
        if _explicit_context is _EXPLICIT_CONTEXT_UNSET:
            self._attach_diffusion_context_metrics(record)
            await self._emit_diffusion_artifacts(
                trace=trace,
                report=report,
                record=record,
                task_metadata=task_metadata,
                judge_reward=outcome_reward,
            )
        else:
            assert isinstance(_explicit_context, ContextPack)
            self._attach_explicit_context_metrics(record, _explicit_context)
            provenance = dict(outcome_metadata or {})
            if outcome_reward is not None:
                provenance["judge_reward"] = outcome_reward
            if explicit_external_refs:
                provenance["external_archive_refs"] = explicit_external_refs
            provenance_store = getattr(
                self,
                "_explicit_execution_provenance_by_key",
                None,
            )
            if provenance_store is None:
                provenance_store = {}
                self._explicit_execution_provenance_by_key = provenance_store
            provenance_store[(task_id, iteration)] = provenance
        return record

    async def _judge_task_reward(
        self,
        *,
        trace: ExecutionTrace,
        task_metadata: TaskMetadataFields,
        trace_path: Path | None,
    ) -> JudgeRewardRecord | None:
        """Score a usable trace while preserving verifier reward metadata."""
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
        llm_token_events = self._drain_llm_token_events()
        logger.warning(
            "Iteration %d skipped before planning: task=%s status=%s error_kind=%s",
            _display_iteration(iteration),
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

    async def _build_prior_context(
        self,
        condition: ConditionName,
        task_id: str,
        *,
        current_iteration: int | None = None,
    ) -> str | None:
        """Build same-task, diffusion, and eligible cross-task prior context."""
        bundle = await self._build_prior_context_bundle(
            condition,
            task_id,
            current_iteration=current_iteration,
        )
        fitted_bundle = await self._fit_prior_context_bundle(bundle)
        flattened = fitted_bundle.flatten()
        if current_iteration is not None:
            self._record_prior_context_metrics(
                task_id=task_id,
                current_iteration=current_iteration,
                bundle=fitted_bundle,
                flattened=flattened,
            )
        return flattened

    async def _build_prior_context_bundle(
        self,
        condition: ConditionName,
        task_id: str,
        *,
        current_iteration: int | None = None,
    ) -> PlannerPriorContextBundle:
        """Build structured prior-context sections before planner flattening."""
        self._ensure_diffusion_runtime_state()
        llm_client = self.mediator.llm_client if condition == "full_traces" else None
        same_task_prior = await get_prior_context(
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

        diffusion_context = await self._build_diffusion_context(
            task_id=task_id,
            current_iteration=current_iteration,
        )
        if diffusion_context is not None:
            logger.info(
                "Diffusion context injected: policy=%s target_task=%s",
                self.config.diffusion.policy,
                task_id,
            )
            return PlannerPriorContextBundle(
                same_task_prior=same_task_prior,
                diffusion_context=diffusion_context,
            )

        cross_context = await get_cross_task_prior_context(
            condition=condition,
            task_id=task_id,
            artifact_store=self.artifact_store,
            released_reports_by_task=self._released_cross_task_reports_by_task,
            llm_client=llm_client,
            model=self.planner.llm_client.model,
            budgets=self.config.budgets,
            condition_name=condition,
            current_iteration=current_iteration,
        )
        if not cross_context:
            return PlannerPriorContextBundle(same_task_prior=same_task_prior)

        header = (
            "# Explicit Cross-Task Feedback\n\n"
            f"condition={condition} target_task={task_id}\n\n"
            "The following context came from other tasks with causal "
            "iteration filters applied."
        )
        logger.info(
            "Cross-task feedback injected: condition=%s target_task=%s",
            condition,
            task_id,
        )
        return PlannerPriorContextBundle(
            same_task_prior=same_task_prior,
            cross_task_prior=f"{header}\n\n{cross_context}",
        )

    async def _fit_prior_context_bundle(
        self,
        bundle: PlannerPriorContextBundle,
    ) -> PlannerPriorContextBundle:
        """Fit structured prior-context sections before planner flattening."""
        from mediated_coevo.runtime.context_compactor import compact_text_for_context
        from mediated_coevo.runtime.token_budget import count_text_tokens

        model = self.planner.llm_client.model
        same_cap = self.config.budgets.max_same_task_prior_tokens
        transfer_cap = self.config.budgets.max_transfer_context_tokens
        original_same_tokens = count_text_tokens(model, bundle.same_task_prior or "")
        same_task_prior = bundle.same_task_prior
        if bundle.same_task_prior and original_same_tokens > same_cap:
            same_task_prior = await compact_text_for_context(
                bundle.same_task_prior,
                llm_client=self.mediator.llm_client,
                label="same-task prior context",
                model=model,
                budget_tokens=same_cap,
                completion_tokens=self.config.budgets.mediator_completion_tokens,
                condition_name=self.config.experiment.condition_name,
            )

        diffusion_context = bundle.diffusion_context
        cross_task_prior = bundle.cross_task_prior
        original_transfer_tokens = 0
        if bundle.diffusion_context:
            original_transfer_tokens = count_text_tokens(
                model, bundle.diffusion_context
            )
            cross_task_prior = None
            if original_transfer_tokens > transfer_cap:
                diffusion_context = await compact_text_for_context(
                    bundle.diffusion_context,
                    llm_client=self.mediator.llm_client,
                    label="diffusion transfer context",
                    model=model,
                    budget_tokens=transfer_cap,
                    completion_tokens=self.config.budgets.mediator_completion_tokens,
                    condition_name=self.config.experiment.condition_name,
                )
        elif bundle.cross_task_prior:
            original_transfer_tokens = count_text_tokens(model, bundle.cross_task_prior)
            if original_transfer_tokens > transfer_cap:
                cross_task_prior = await compact_text_for_context(
                    bundle.cross_task_prior,
                    llm_client=self.mediator.llm_client,
                    label="cross-task transfer context",
                    model=model,
                    budget_tokens=transfer_cap,
                    completion_tokens=self.config.budgets.mediator_completion_tokens,
                    condition_name=self.config.experiment.condition_name,
                )

        fitted_same_tokens = count_text_tokens(model, same_task_prior or "")
        fitted_transfer_tokens = count_text_tokens(
            model,
            diffusion_context or cross_task_prior or "",
        )
        budget_violation = (
            bundle.context_budget_violation
            or original_same_tokens > fitted_same_tokens
            or original_transfer_tokens > fitted_transfer_tokens
            or fitted_same_tokens > same_cap
            or fitted_transfer_tokens > transfer_cap
        )
        return PlannerPriorContextBundle(
            same_task_prior=same_task_prior,
            cross_task_prior=cross_task_prior,
            diffusion_context=diffusion_context,
            context_budget_violation=budget_violation,
        )

    def _record_prior_context_metrics(
        self,
        *,
        task_id: str,
        current_iteration: int,
        bundle: PlannerPriorContextBundle,
        flattened: str | None,
    ) -> None:
        from mediated_coevo.runtime.token_budget import count_text_tokens

        model = self.planner.llm_client.model
        flattened_tokens = count_text_tokens(model, flattened or "")
        same_task_tokens = count_text_tokens(model, bundle.same_task_prior or "")
        transfer_context_kind = "none"
        transfer_context = None
        if bundle.diffusion_context:
            transfer_context_kind = "diffusion"
            transfer_context = bundle.diffusion_context
        elif bundle.cross_task_prior:
            transfer_context_kind = "cross_task_prior"
            transfer_context = bundle.cross_task_prior
        transfer_context_tokens = count_text_tokens(model, transfer_context or "")
        same_cap = self.config.budgets.max_same_task_prior_tokens
        transfer_cap = self.config.budgets.max_transfer_context_tokens
        total_cap = self.config.budgets.max_total_prior_context_tokens
        self._prior_context_by_target[(task_id, current_iteration)] = {
            "same_task_prior_tokens": same_task_tokens,
            "transfer_context_kind": transfer_context_kind,
            "transfer_context_tokens": transfer_context_tokens,
            "total_planner_prior_context_tokens": flattened_tokens,
            "max_same_task_prior_tokens": same_cap,
            "max_transfer_context_tokens": transfer_cap,
            "max_total_prior_context_tokens": total_cap,
            "context_budget_violation": (
                bundle.context_budget_violation
                or same_task_tokens > same_cap
                or transfer_context_tokens > transfer_cap
            ),
        }

    async def _emit_diffusion_artifacts(
        self,
        *,
        trace: ExecutionTrace,
        report: MediatorReport | None,
        record: IterationRecord,
        task_metadata: TaskMetadataFields,
        judge_reward: float | None,
    ) -> None:
        self._ensure_diffusion_runtime_state()
        if not self.config.diffusion.enabled or self.freeze_diffusion_artifact_store:
            return

        artifacts = await emit_diffusion_artifacts(
            trace=trace,
            report=report,
            record=record,
            model=self.planner.llm_client.model,
            llm_client=self.mediator.llm_client,
            budgets=self.config.budgets,
            condition_name=self.config.experiment.condition_name,
            task_metadata=task_metadata,
            judge_reward=judge_reward,
        )
        for artifact in artifacts:
            self._diffusion_store.store_artifact(artifact, overwrite=True)

    async def _build_diffusion_context(
        self,
        *,
        task_id: str,
        current_iteration: int | None,
    ) -> str | None:
        self._ensure_diffusion_runtime_state()
        policy_name = self.config.diffusion.policy
        if (
            not self.config.diffusion.enabled
            or policy_name == "none"
            or current_iteration is None
        ):
            return None
        await self._prepare_diffusion_subscriptions(
            target_task_id=task_id,
            current_iteration=current_iteration,
        )
        subscriptions = self._diffusion_sub_board.pop(
            (current_iteration, task_id),
            [],
        )
        snapshot = self._diffusion_snapshot_by_iteration.get(current_iteration)
        if snapshot is None:
            self._record_empty_diffusion_context(task_id, current_iteration)
            return None

        if not subscriptions:
            self._record_empty_diffusion_context(
                task_id,
                current_iteration,
                snapshot=snapshot,
            )
            return None

        bundle = await render_diffusion_subscriptions(
            store=self._diffusion_store,
            snapshot=snapshot,
            model=self.planner.llm_client.model,
            target_task_id=task_id,
            target_iteration=current_iteration,
            target_run_id=None,
            subscriptions=subscriptions,
            eligible_count=self._diffusion_context_by_target.get(
                (task_id, current_iteration),
                {},
            ).get("eligible_count"),
            max_context_tokens=self.config.budgets.max_transfer_context_tokens,
            compact_artifact_content=self._compact_diffusion_artifact_content,
        )
        self._diffusion_context_by_target[(task_id, current_iteration)] = {
            "graph_snapshot_id": bundle.snapshot_id,
            "graph_policy": bundle.graph_policy,
            "eligible_count": bundle.eligible_count,
            "selected_count": bundle.selected_count,
            "rendered_count": bundle.rendered_count,
            "context_tokens": bundle.context_tokens,
            "source_task_ids": bundle.source_task_ids,
            "compacted_artifact_ids": list(bundle.compacted_artifact_ids or []),
            "dropped_artifact_ids": list(bundle.dropped_for_budget_artifact_ids or []),
            "budget_violation": bundle.budget_violation,
        }
        return bundle.text

    async def _compact_diffusion_artifact_content(
        self,
        artifact: DiffusionArtifact,
        budget_tokens: int,
    ) -> str:
        from mediated_coevo.runtime.context_compactor import compact_text_for_context

        return await compact_text_for_context(
            artifact.content,
            llm_client=self.mediator.llm_client,
            label=f"diffusion artifact {artifact.artifact_id}",
            model=self.planner.llm_client.model,
            budget_tokens=budget_tokens,
            completion_tokens=self.config.budgets.mediator_completion_tokens,
            condition_name=self.config.experiment.condition_name,
        )

    async def _prepare_diffusion_subscriptions(
        self,
        *,
        target_task_id: str,
        current_iteration: int,
    ) -> None:
        policy_name = self.config.diffusion.policy
        if policy_name == "langchain_graph":
            await self._prepare_langchain_graph_subscriptions(
                target_task_id=target_task_id,
                current_iteration=current_iteration,
            )
            return

        if current_iteration in self._diffusion_prepared_iterations:
            return

        artifacts = self._diffusion_store.query_artifacts(
            recent=None,
            before_source_iteration=current_iteration,
        )
        target_task_ids = self._target_task_ids_for_diffusion(
            fallback_task_id=target_task_id,
            artifacts=artifacts,
        )
        snapshot_task_ids = self._snapshot_task_ids_for_diffusion(
            target_task_ids=target_task_ids,
            artifacts=artifacts,
        )
        if len(snapshot_task_ids) < 2:
            self._diffusion_prepared_iterations.add(current_iteration)
            return

        snapshot = self._diffusion_snapshot(
            graph_dir=self.experiment_dir / "task-graph",
            task_ids=snapshot_task_ids,
            iteration=current_iteration,
        )
        self._diffusion_store.store_graph_snapshot(snapshot, overwrite=True)
        self._diffusion_snapshot_by_iteration[current_iteration] = snapshot

        for target_id in target_task_ids:
            eligible_artifacts = [
                artifact
                for artifact in artifacts
                if artifact.source_task_id != target_id
            ]
            subscriptions = self._select_diffusion_subscriptions(
                target_task_id=target_id,
                current_iteration=current_iteration,
                eligible_artifacts=eligible_artifacts,
                snapshot=snapshot,
            )
            self._record_prepared_diffusion_context(
                target_id,
                current_iteration,
                snapshot=snapshot,
                eligible_count=len(eligible_artifacts),
                subscriptions=subscriptions,
            )
            self._record_unselected_diffusion_candidates(
                target_task_id=target_id,
                current_iteration=current_iteration,
                eligible_artifacts=eligible_artifacts,
                subscriptions=subscriptions,
                snapshot=snapshot,
            )
            if subscriptions:
                self._diffusion_sub_board[(current_iteration, target_id)] = (
                    subscriptions
                )

        self._diffusion_prepared_iterations.add(current_iteration)

    async def _prepare_langchain_graph_subscriptions(
        self,
        *,
        target_task_id: str,
        current_iteration: int,
    ) -> None:
        key = (current_iteration, target_task_id)
        if key in self._langchain_graph_prepared_targets:
            return

        artifacts = self._diffusion_store.query_artifacts(
            recent=None,
            before_source_iteration=current_iteration,
        )
        previous_snapshot = self._latest_langchain_graph_snapshot(
            current_iteration=current_iteration,
        )
        task_profile = self._diffusion_task_profile(target_task_id)
        if self._uses_fixed_langchain_graph():
            if previous_snapshot is None:
                self._langchain_graph_prepared_targets.add(key)
                return
            result = await self._get_langchain_graph_policy().select_with_fixed_graph(
                task_profile=task_profile,
                current_iteration=current_iteration,
                snapshot=previous_snapshot,
                artifacts=artifacts,
            )
        else:
            result = await self._get_langchain_graph_policy().prepare(
                task_profile=task_profile,
                current_iteration=current_iteration,
                previous_snapshot=previous_snapshot,
                artifacts=artifacts,
            )
            self._diffusion_store.store_graph_snapshot(result.snapshot, overwrite=True)
        self._diffusion_snapshot_by_iteration[current_iteration] = result.snapshot
        self._record_prepared_diffusion_context(
            target_task_id,
            current_iteration,
            snapshot=result.snapshot,
            eligible_count=len(artifacts),
            subscriptions=result.subscriptions,
        )
        self._record_unselected_diffusion_candidates(
            target_task_id=target_task_id,
            current_iteration=current_iteration,
            eligible_artifacts=artifacts,
            subscriptions=result.subscriptions,
            snapshot=result.snapshot,
        )
        if result.subscriptions:
            self._diffusion_sub_board[(current_iteration, target_task_id)] = (
                result.subscriptions
            )
        self._langchain_graph_prepared_targets.add(key)

    def _uses_fixed_langchain_graph(self) -> bool:
        return self.config.experiment.benchmark_selection.split in {
            "validation",
            "test",
        }

    def _latest_langchain_graph_snapshot(
        self,
        *,
        current_iteration: int,
    ) -> TaskGraphSnapshot | None:
        snapshots = self._diffusion_store.query_graph_snapshots(
            recent=None,
            before_iteration=current_iteration,
        )
        snapshot = _latest_langchain_snapshot(snapshots)
        if snapshot is not None:
            return snapshot
        return _latest_langchain_snapshot(
            self._diffusion_store.query_graph_snapshots(recent=None)
        )

    def _get_langchain_graph_policy(self) -> LangChainGraphPolicy:
        if self._langchain_graph_policy is None:
            self._langchain_graph_policy = LangChainGraphPolicy(
                model=self.config.models.mediator,
                run_id=self.experiment_dir.name,
                max_artifacts=self.config.diffusion.max_artifacts,
            )
        return self._langchain_graph_policy

    def _diffusion_task_profile(self, task_id: str) -> dict[str, Any]:
        task = self.benchmark_repo.resolve(task_id)
        task_config = getattr(task, "task_config", {})
        if hasattr(task_config, "model_dump"):
            task_config = task_config.model_dump(mode="json")
        return {
            "task_id": task_id,
            "instruction": getattr(task, "instruction", ""),
            "task_config": task_config,
        }

    def _select_diffusion_subscriptions(
        self,
        *,
        target_task_id: str,
        current_iteration: int,
        eligible_artifacts: list[DiffusionArtifact],
        snapshot: TaskGraphSnapshot,
    ) -> list[DiffusionSubscription]:
        policy_name = self.config.diffusion.policy
        if policy_name == "capped_broadcast":
            return select_capped_broadcast_subscriptions(
                eligible_artifacts=eligible_artifacts,
                max_artifacts=self.config.diffusion.max_artifacts,
                avoid_recheck_max_artifacts=(
                    self.config.diffusion.avoid_recheck_max_artifacts
                ),
            )
        if policy_name == "random_k":
            return select_random_k_subscriptions(
                eligible_artifacts=eligible_artifacts,
                target_task_id=target_task_id,
                target_iteration=current_iteration,
                max_artifacts=self.config.diffusion.max_artifacts,
                seed=self.config.experiment.seed,
                avoid_recheck_max_artifacts=(
                    self.config.diffusion.avoid_recheck_max_artifacts
                ),
            )
        if policy_name == "top_k_similarity":
            return select_top_k_similarity_subscriptions(
                eligible_artifacts=eligible_artifacts,
                snapshot=snapshot,
                target_task_id=target_task_id,
                max_artifacts=self.config.diffusion.max_artifacts,
                top_k_neighbors=self.config.diffusion.top_k_neighbors,
                avoid_recheck_max_artifacts=(
                    self.config.diffusion.avoid_recheck_max_artifacts
                ),
            )
        return []

    def _target_task_ids_for_diffusion(
        self,
        *,
        fallback_task_id: str,
        artifacts: list[DiffusionArtifact],
    ) -> list[str]:
        if self._diffusion_target_task_ids:
            return list(dict.fromkeys(self._diffusion_target_task_ids))
        task_ids = {fallback_task_id}
        task_ids.update(artifact.source_task_id for artifact in artifacts)
        return sorted(task_ids)

    @staticmethod
    def _snapshot_task_ids_for_diffusion(
        *,
        target_task_ids: list[str],
        artifacts: list[DiffusionArtifact],
    ) -> list[str]:
        task_ids = set(target_task_ids)
        task_ids.update(artifact.source_task_id for artifact in artifacts)
        return sorted(task_ids)

    def _record_empty_diffusion_context(
        self,
        task_id: str,
        current_iteration: int,
        *,
        snapshot: TaskGraphSnapshot | None = None,
        eligible_count: int = 0,
    ) -> None:
        current_context = self._diffusion_context_by_target.get(
            (task_id, current_iteration),
            {},
        )
        self._diffusion_context_by_target[(task_id, current_iteration)] = {
            "graph_snapshot_id": snapshot.snapshot_id if snapshot is not None else None,
            "graph_policy": snapshot.graph_policy if snapshot is not None else None,
            "eligible_count": current_context.get("eligible_count", eligible_count),
            "selected_count": 0,
            "rendered_count": 0,
            "context_tokens": 0,
            "source_task_ids": [],
            "compacted_artifact_ids": [],
            "dropped_artifact_ids": [],
            "budget_violation": False,
        }

    def _record_prepared_diffusion_context(
        self,
        task_id: str,
        current_iteration: int,
        *,
        snapshot: TaskGraphSnapshot,
        eligible_count: int,
        subscriptions: list[DiffusionSubscription],
    ) -> None:
        self._diffusion_context_by_target[(task_id, current_iteration)] = {
            "graph_snapshot_id": snapshot.snapshot_id,
            "graph_policy": snapshot.graph_policy,
            "eligible_count": eligible_count,
            "selected_count": len(subscriptions),
            "rendered_count": 0,
            "context_tokens": 0,
            "source_task_ids": list(
                dict.fromkeys(
                    subscription.artifact.source_task_id
                    for subscription in subscriptions
                )
            ),
            "compacted_artifact_ids": [],
            "dropped_artifact_ids": [],
            "budget_violation": False,
        }

    def _record_unselected_diffusion_candidates(
        self,
        *,
        target_task_id: str,
        current_iteration: int,
        eligible_artifacts: list[DiffusionArtifact],
        subscriptions: list[DiffusionSubscription],
        snapshot: TaskGraphSnapshot,
    ) -> None:
        selected_ids = {
            subscription.artifact.artifact_id for subscription in subscriptions
        }
        for artifact in eligible_artifacts:
            if artifact.artifact_id in selected_ids:
                continue
            diffusion_channel = diffusion_channel_for_artifact(artifact)
            self._diffusion_store.append_diffused_record(
                DiffusedRecord(
                    artifact_id=artifact.artifact_id,
                    source_task_id=artifact.source_task_id,
                    source_iteration=artifact.source_iteration,
                    source_run_id=artifact.source_run_id,
                    target_task_id=target_task_id,
                    target_iteration=current_iteration,
                    snapshot_id=snapshot.snapshot_id,
                    policy_name=self.config.diffusion.policy,
                    relation="candidate",
                    reason="eligible_not_selected",
                    eligible=True,
                    selected=False,
                    rendered=False,
                    verifier_reward=artifact.verifier_reward,
                    judge_reward=artifact.judge_reward,
                    success=(
                        None
                        if artifact.verifier_reward is None
                        else artifact.verifier_reward == 1.0
                    ),
                    regression=True
                    if artifact.metadata.get("regression") is True
                    else None,
                    metadata={
                        "artifact_type": artifact.artifact_type.value,
                        "risk_level": artifact.risk_level.value,
                        **(
                            {"diffusion_channel": diffusion_channel}
                            if diffusion_channel is not None
                            else {}
                        ),
                    },
                )
            )

    def _diffusion_snapshot(
        self,
        *,
        graph_dir: Path,
        task_ids: list[str],
        iteration: int,
    ) -> TaskGraphSnapshot:
        graph_name = self.config.diffusion.graph
        if (
            graph_name in {"task_similarity", "precomputed_similarity"}
            and graph_dir.exists()
        ):
            from mediated_coevo.diffusion import DiffusionNetwork, GraphBuildSpec

            network = DiffusionNetwork.from_graph_dir(
                GraphBuildSpec(
                    graph_dir=graph_dir,
                    task_ids=task_ids,
                    run_id=self.experiment_dir.name,
                    iteration=iteration,
                    graph_policy="precomputed_similarity",
                )
            )
            return network.to_snapshot()
        return TaskGraphSnapshot(
            run_id=self.experiment_dir.name,
            iteration=iteration,
            task_ids=task_ids,
            graph_policy=graph_name if graph_name != "none" else "broadcast",
        )

    def _attach_diffusion_context_metrics(self, record: IterationRecord) -> None:
        self._ensure_diffusion_runtime_state()
        self._attach_diffusion_artifact_store_metrics(record)
        prior_context = self._prior_context_by_target.get(
            (record.task_id, record.iteration)
        )
        if prior_context is not None:
            record.same_task_prior_tokens = prior_context["same_task_prior_tokens"]
            record.transfer_context_kind = prior_context["transfer_context_kind"]
            record.transfer_context_tokens = prior_context["transfer_context_tokens"]
            record.total_planner_prior_context_tokens = prior_context[
                "total_planner_prior_context_tokens"
            ]
            record.max_same_task_prior_tokens = prior_context[
                "max_same_task_prior_tokens"
            ]
            record.max_transfer_context_tokens = prior_context[
                "max_transfer_context_tokens"
            ]
            record.max_total_prior_context_tokens = prior_context[
                "max_total_prior_context_tokens"
            ]
            record.context_budget_violation = prior_context["context_budget_violation"]
        context = self._diffusion_context_by_target.get(
            (record.task_id, record.iteration)
        )
        if context is None:
            return
        record.graph_snapshot_id = context["graph_snapshot_id"]
        record.diffusion_graph = context["graph_policy"]
        record.diffusion_artifacts_eligible = context["eligible_count"]
        record.diffusion_artifacts_selected = context["selected_count"]
        record.diffusion_artifacts_rendered = context["rendered_count"]
        record.source_task_ids = list(context["source_task_ids"])
        record.compacted_diffusion_artifact_ids = list(
            context["compacted_artifact_ids"]
        )
        record.dropped_for_budget_artifact_ids = list(context["dropped_artifact_ids"])
        record.context_budget_violation = (
            record.context_budget_violation or context["budget_violation"]
        )
        if record.diffusion_artifacts_rendered > 0:
            record.reward_after_diffusion_context = record.reward
            record.regression_after_diffusion_context = (
                record.delta_reward is not None and record.delta_reward < 0
            )

    def _attach_explicit_context_metrics(
        self,
        record: IterationRecord,
        context: ContextPack,
    ) -> None:
        """Copy the validated sample context contract onto the iteration row."""
        record.diffusion_policy = context.policy_name
        record.diffusion_enabled = context.policy_name != "none"
        record.graph_snapshot_id = context.snapshot_id
        record.diffusion_graph = (
            str(context.metadata.get("graph_policy") or "") or None
            if context.snapshot_id is not None
            else None
        )
        record.diffusion_artifacts_eligible = len(context.eligible_artifact_ids)
        record.diffusion_artifacts_selected = len(context.selected_artifact_ids)
        record.diffusion_artifacts_rendered = len(context.rendered_artifact_ids)
        record.diffusion_artifact_store_count = len(context.eligible_artifact_ids)
        record.transfer_context_kind = (
            "diffusion" if context.text is not None else "none"
        )
        record.transfer_context_tokens = context.token_count
        record.same_task_prior_tokens = 0
        record.total_planner_prior_context_tokens = context.token_count
        record.max_same_task_prior_tokens = 0
        record.max_transfer_context_tokens = context.max_context_tokens or 0
        record.max_total_prior_context_tokens = context.max_context_tokens or 0
        record.context_budget_violation = context.budget_violation
        record.compacted_diffusion_artifact_ids = list(context.compacted_artifact_ids)
        record.dropped_for_budget_artifact_ids = list(
            context.dropped_for_budget_artifact_ids
        )
        record.source_task_ids = list(context.source_task_ids)
        if record.diffusion_artifacts_rendered > 0:
            record.reward_after_diffusion_context = record.reward
            record.regression_after_diffusion_context = (
                record.delta_reward is not None and record.delta_reward < 0
            )

    def _attach_diffusion_artifact_store_metrics(
        self,
        record: IterationRecord,
    ) -> None:
        self._ensure_diffusion_runtime_state()
        record.diffusion_artifact_store_path = (
            self.preloaded_diffusion_artifact_store_path
        )
        record.diffusion_artifact_store_count = (
            self.preloaded_diffusion_artifact_store_count
        )
        record.diffusion_artifact_store_frozen = self.freeze_diffusion_artifact_store

    def _ensure_diffusion_runtime_state(self) -> None:
        if not hasattr(self, "_diffusion_store"):
            self._diffusion_store = DiffusionStore(self.experiment_dir / "diffusion")
        if not hasattr(self, "_diffusion_context_by_target"):
            self._diffusion_context_by_target = {}
        if not hasattr(self, "_prior_context_by_target"):
            self._prior_context_by_target = {}
        if not hasattr(self, "_diffusion_sub_board"):
            self._diffusion_sub_board = {}
        if not hasattr(self, "freeze_diffusion_artifact_store"):
            self.freeze_diffusion_artifact_store = False
        if not hasattr(self, "preloaded_diffusion_artifact_store_path"):
            self.preloaded_diffusion_artifact_store_path = None
        if not hasattr(self, "preloaded_diffusion_artifact_store_count"):
            self.preloaded_diffusion_artifact_store_count = 0
        if not hasattr(self, "_diffusion_prepared_iterations"):
            self._diffusion_prepared_iterations = set()
        if not hasattr(self, "_langchain_graph_prepared_targets"):
            self._langchain_graph_prepared_targets = set()
        if not hasattr(self, "_langchain_graph_policy"):
            self._langchain_graph_policy = None
        if not hasattr(self, "_diffusion_snapshot_by_iteration"):
            self._diffusion_snapshot_by_iteration = {}
        if not hasattr(self, "_diffusion_target_task_ids"):
            self._diffusion_target_task_ids = []

    def _write_metric(self, record: IterationRecord) -> None:
        """Append an iteration record to metrics.jsonl."""
        with open(self._metrics_path, "a") as f:
            f.write(json.dumps(metric_row(record), sort_keys=True) + "\n")

    def _current_skill_hashes(self) -> dict[str, str]:
        """Return hashes for current SkillStore contents."""
        return dict(self.skill_store.skill_hashes())

    def _drain_llm_token_events(self) -> list[TokenBudgetEvent]:
        """Collect token telemetry from configured LLM clients."""
        events: list[TokenBudgetEvent] = []
        for llm_client in (
            self.planner.llm_client,
            self.mediator.llm_client,
            getattr(self, "judge_llm_client", None),
        ):
            if llm_client is not None and hasattr(llm_client, "drain_token_events"):
                events.extend(llm_client.drain_token_events())
        return events


def _latest_langchain_snapshot(
    snapshots: list[TaskGraphSnapshot],
) -> TaskGraphSnapshot | None:
    return next(
        (
            snapshot
            for snapshot in snapshots
            if snapshot.graph_policy == "langchain_graph"
        ),
        None,
    )


def _display_iteration(iteration: int) -> int:
    """Return the human-facing iteration number for terminal logs."""
    return iteration + 1
