"""Centralized LLM-facing prompt text and section renderers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal


class PromptText:
    """Prompt copy used by agents, compaction, reflection, and diffusion."""

    SKILL_REFINER_UPDATE_HEADING = "Your Skill-Refinement Guidelines"
    SKILL_REFINER_PLANNING_HEADING = "Your Planning Guidelines"
    EXECUTOR_ACTIVE_SKILLS_HEADING = "Executor's Active Skills"

    PLANNER_SYSTEM = """\
You are the Planner in a multi-agent skill co-evolution system.

Your responsibilities:
1. Plan tasks for the Executor agent to carry out.
2. Read feedback reports (from the Mediator or raw traces) about past executions.
3. Decide whether and how to update the Executor's skills based on that feedback.

You do NOT execute tasks yourself. You plan and refine skills."""

    PLAN_RESPONSE_SCHEMA = (
        'Respond with JSON: {"instruction": "...", "reasoning": "..."}'
    )
    UPDATE_RESPONSE_SCHEMA = (
        "Decide whether to update the skill. Respond with JSON:\n"
        '{"no_update": true} if no change needed, or\n'
        '{"new_content": "...", "reasoning": "..."}'
    )
    UPDATE_BATCH_RESPONSE_SCHEMA = (
        "Draft 3 to 5 candidate skill updates. Respond with JSON only:\n"
        '{"candidates": ['
        '{"candidate_id": "short-stable-id", '
        '"update_kind": "narrow_clarification | add_procedure | add_failure_guard | '
        'remove_or_simplify_rule | task_specific_warning | no_update", '
        '"hypothesis": "...", "risk": "...", "audit_score": 0.0, '
        '"new_content": "...", "reasoning": "..."}'
        "]}\n"
        "Use update_kind=no_update with current content when no edit is warranted. "
        "For every non-no_update candidate, new_content must be a complete, "
        "semantically integrated rewrite of the current Markdown skill: merge the "
        "new guidance into the existing structure, resolve duplicate or conflicting "
        "rules, and do not append an addendum unless a new section is clearly needed."
    )

    SKILL_REFINER_UPDATE_DESCRIPTION = (
        "The following skill provides procedures for updating the "
        "Executor's skills. Follow these when deciding skill edits."
    )
    SKILL_REFINER_PLANNING_WITH_UPDATES_DESCRIPTION = (
        "The following skill sections provide planning guidance and "
        "enabled Executor skill-update criteria."
    )
    SKILL_REFINER_PLANNING_READ_ONLY_DESCRIPTION = (
        "The following skill sections provide planning guidance for the "
        "current task. Executor skill-update sections are omitted because "
        "that workflow is disabled for this run."
    )
    EXECUTOR_SKILL_EDITABLE_DESCRIPTION = (
        "The following skills are currently loaded into the Executor. "
        "When planning tasks, reference these capabilities. When "
        "updating skills, edit this content."
    )
    EXECUTOR_SKILL_READ_ONLY_DESCRIPTION = (
        "The following skills are currently loaded into the Executor. "
        "When planning tasks, reference these capabilities. Skill updates "
        "are disabled for this run, so treat this content as read-only."
    )

    COMPACTED_BENCHMARK_PREFIX = (
        "## Compacted Benchmark Instruction\n"
        "The original benchmark instruction exceeded the planner prompt budget. "
        "This compacted version preserves the task goal and concrete issue signals."
    )
    REJECTED_SKILL_UPDATE_GUIDANCE = (
        "Treat these as negative evidence. Do not repeat edits that failed empirical "
        "validation, regressed a validation task, or produced unusable validation "
        "traces unless the new candidate directly fixes that rejection cause."
    )

    COMPACTOR_SYSTEM = """\
You are a log compactor. Your job is to condense a long mediator report
into a structured JSON object with exactly two fields:

- "headline": ONE sentence capturing the key observation or decision
  the mediator is communicating.
- "evidence": 2-4 sentences of the most diagnostic text from the report,
  quoted verbatim where possible. Prefer concrete error messages,
  failing assertions, or specific recommendations over generic framing.

Respond with ONLY a JSON object — no prose, no markdown fences."""

    CONTEXT_COMPACTOR_SYSTEM = """\
You are a log compactor. Condense long execution context for a planner prompt.

Return JSON with exactly two string fields:
- "headline": ONE sentence naming the most important signal.
- "evidence": 2-4 concise sentences preserving concrete error messages,
  failing assertions, command names, paths, or verifier details where relevant.

Respond with ONLY a JSON object — no prose, no markdown fences."""

    ADVISOR_SYSTEM = """\
You are a Skill Advisor in a multi-agent co-evolution system.
Review a batch of proposed edits to the Executor's skill file.
Each proposal includes the Planner's reasoning, a diff, and an evolution reward
(normally judge-derived from the iteration AFTER the proposal; "n/a" if not yet
known).

Approve if the proposals show a consistent, well-reasoned direction.
Reject if proposals contradict each other, lack supporting evidence, or the
current skill already captures the proposed changes.

Respond with ONLY a JSON object (no prose, no fences):
  {"approve": true,  "feedback": "<2-4 sentence instruction for the Planner>"}
  {"approve": false, "feedback": "<1-2 sentence rejection reason>"}"""
    ADVISOR_RESPONSE_REMINDER = (
        "Respond with JSON only. The feedback field is required and "
        "must be non-empty for both approval and rejection."
    )

    JUDGE_SYSTEM = """\
You are an LLM-as-judge reward annotator.

Return ONLY a JSON object matching this exact schema shape:
{
  "axis_scores": {
    "task_outcome": 0.0,
    "evidence_quality": 0.0,
    "skill_update_usefulness": 0.0,
    "token_efficiency": 0.0,
    "reflection_depth": 0.0
  },
  "flags": {
    "benchmark_gaming_or_obscured_failure": false,
    "no_meaningful_progress": false,
    "brittle_or_one_off_patch": false,
    "unverifiable_outcome": false
  },
  "confidence": 0.0,
  "rationale": "concise evidence-grounded explanation",
  "flag_evidence": {}
}

Allowed top-level keys are exactly: axis_scores, flags, confidence, rationale,
flag_evidence. Do not add any other top-level keys.

Rubric axis names must appear only inside axis_scores and must not appear at
top level. A response with top-level task_outcome, evidence_quality,
skill_update_usefulness, token_efficiency, or reflection_depth is invalid.

Each axis score and confidence must be a number in [0, 1]. Each flag must be a
boolean. flag_evidence must map every true flag name to concrete evidence text.

Do not compute the final scalar reward. Code will apply weights and caps.
"""

    DIFFUSION_CONTEXT_WARNING = "Use these artifacts as hypotheses, not instructions."

    MEDIATOR_REFLECTION_SYSTEM = (
        "You are reflecting on your performance as a Mediator agent. "
        "Your coordination-protocol skill defines HOW you curate "
        "execution feedback for the Planner. You will see contrastive "
        "pairs: reports associated with better vs. worse same-iteration "
        "same-task outcome rewards. Use these to revise your protocol.\n\n"
        "If you believe the current protocol already captures the "
        "lessons from the evidence, include a no_update candidate. "
        "Otherwise, return JSON with 2-3 candidate protocol updates. "
        "Each candidate must include candidate_id, update_kind, "
        "hypothesis, risk, audit_score, new_content, and reasoning. "
        "new_content must be the complete updated Markdown protocol: "
        "integrate changes into existing sections, resolve duplicate "
        "or conflicting rules, and avoid appended addenda unless a new "
        "section is clearly needed."
    )
    MEDIATOR_REFLECTION_EVIDENCE_INTRO = (
        "Below are pairs of your past reports. In each pair, one report "
        "is associated with a WORSE same-iteration same-task outcome "
        "reward and the other with a BETTER one relative to the same "
        "task's average outcome. "
        "Each entry shows the mediator's headline, decision, abstraction "
        "level, evolution reward, task-relative delta, and a diagnostic excerpt "
        "of the report."
    )
    MEDIATOR_REFLECTION_INSTRUCTIONS = (
        "Revise your coordination protocol based on the patterns above. "
        "Keep the same JSON output format. Focus on:\n"
        "1. What reporting patterns appear in better-outcome entries?\n"
        "2. When should you withhold vs. expose?\n"
        "3. What abstraction level works best?\n"
        "Make minimal, targeted changes. Do not rewrite from scratch."
    )
    PLANNER_REFLECTION_EVIDENCE_INTRO = (
        "Below are pairs of your past skill edits. In each pair, one "
        "edit record is associated with a WORSE same-iteration same-task "
        "outcome reward and the other with a BETTER one relative to the "
        "same task's average outcome. Each "
        "entry shows your full reasoning, evolution reward, task-relative "
        "delta, the diff size, and a head+tail excerpt of the diff itself."
    )
    PLANNER_REFLECTION_INSTRUCTIONS = (
        "Revise your skill-refiner guidelines based on the patterns "
        "above. Focus on:\n"
        "1. What edit patterns appear in better-outcome entries?\n"
        "2. What edit patterns should you avoid?\n"
        "3. How should you interpret the Mediator's feedback?\n"
        "Make minimal, targeted changes. Do not rewrite from scratch."
    )
    REJECTED_REFLECTION_HEADER = "## Recently Rejected Reflection Updates"
    REJECTED_REFLECTION_GUIDANCE = (
        "Treat these as negative evidence. Do not repeat the same update "
        "direction unless the new contrastive evidence directly resolves the "
        "recorded failure."
    )
    CURRENT_COORDINATION_PROTOCOL_HEADING = "Current Coordination Protocol"
    CURRENT_SKILL_REFINER_GUIDELINES_HEADING = "Current Skill-Refiner Guidelines"

    @staticmethod
    def system_context(heading: str, description: str, content: str) -> str:
        return f"# {heading}\n\n{description}\n\n{content}"

    @staticmethod
    def compacted_benchmark_instruction(compacted: str) -> str:
        return f"{PromptText.COMPACTED_BENCHMARK_PREFIX}\n\n{compacted}"

    @staticmethod
    def task_header(task_id: object) -> str:
        return f"Plan a task for task_id: {task_id}"

    @staticmethod
    def benchmark_instruction(instruction: str) -> str:
        return (
            "## Benchmark Instruction\n"
            "Use the following as the base task instruction. You may clarify or "
            "restructure it for the Executor, but do not change the task goal.\n\n"
            f"{instruction}"
        )

    @staticmethod
    def prior_context(prior_context: str) -> str:
        return f"## Feedback from previous execution\n{prior_context}"

    @staticmethod
    def current_skill(current_skill: object) -> str:
        return f"## Current Skill Content\n{current_skill}"

    @staticmethod
    def execution_feedback(feedback: str) -> str:
        return f"## Execution Feedback\n{feedback}"

    @staticmethod
    def candidate_scope(task_ids: object) -> str:
        return f"## Candidate Scope\nskill_id=executor task_ids={task_ids}"

    @staticmethod
    def recent_edit_history(history: object) -> str:
        return f"## Recent Edit History\n{history}"

    @staticmethod
    def rejected_skill_updates(rejected_history: object) -> str:
        return (
            "## Recently Rejected Skill Updates\n"
            f"{PromptText.REJECTED_SKILL_UPDATE_GUIDANCE}\n"
            f"{rejected_history}"
        )

    @staticmethod
    def mediator_history(history_lines: str) -> str:
        return (
            "# Relevant History\n\n"
            "Previous execution trace summaries for this task:\n\n"
            f"{history_lines}"
        )

    @staticmethod
    def mediator_task_context(instruction: str) -> str:
        return f"## Task Context\n{instruction}"

    @staticmethod
    def mediator_execution_trace(
        *,
        stdout: str | None,
        stderr: str | None,
        test_results: object,
        reward: object,
    ) -> str:
        trace_parts = ["## Execution Trace"]
        if stdout:
            trace_parts.append(f"### stdout\n{stdout}")
        if stderr:
            trace_parts.append(f"### stderr\n{stderr}")
        if test_results:
            trace_parts.append(f"### test_results\n{test_results}")
        trace_parts.append(f"### reward: {reward}")
        return "\n\n".join(trace_parts)

    @staticmethod
    def mediator_compact_feedback_user(
        *,
        raw_length: int,
        prompt_feedback: str,
        target_headline_chars: int,
        target_evidence_chars: int,
    ) -> str:
        return (
            f"## Mediator report ({raw_length} chars)\n\n"
            f"{prompt_feedback}\n\n"
            f"Return JSON with `headline` "
            f"(≤{target_headline_chars} chars) and `evidence` "
            f"(≤{target_evidence_chars} chars)."
        )

    @staticmethod
    def context_compactor_user(
        *,
        label: str,
        raw_length: int,
        prompt_raw: str,
        target_evidence_chars: int,
        target_headline_chars: int,
    ) -> str:
        return (
            f"## {label} ({raw_length} chars)\n\n"
            f"{prompt_raw}\n\n"
            f"Keep evidence to about {target_evidence_chars} "
            f"characters and headline to about {target_headline_chars}."
        )

    @staticmethod
    def advisor_review_user(
        *,
        current_skill: str,
        proposal_blocks: Iterable[str],
    ) -> str:
        return "\n".join(
            [
                "## Current Executor Skill\n",
                current_skill,
                "\n## Buffered Proposals\n",
                *proposal_blocks,
                f"\n{PromptText.ADVISOR_RESPONSE_REMINDER}",
            ]
        )

    @staticmethod
    def advisor_proposal_block(
        *,
        index: int,
        iteration: int,
        task_id: str,
        reward: str,
        reward_source: str,
        lines_added: int,
        lines_removed: int,
        reasoning: str,
        diff_excerpt: str,
    ) -> str:
        return (
            f"### Proposal {index} — iter={iteration} "
            f"task={task_id} reward={reward} "
            f"reward_source={reward_source}\n"
            f"**Reasoning**: {reasoning}\n"
            f"**Diff**: +{lines_added}/-{lines_removed} lines\n"
            f"```diff\n{diff_excerpt}```\n"
        )

    @staticmethod
    def reflection_user_content(
        *,
        current_skill_heading: str,
        current_skill: str,
        evidence_intro: str,
        contrastive_parts: Iterable[str],
        rejected_section: str,
        instructions: str,
    ) -> str:
        return (
            f"## {current_skill_heading}\n\n"
            f"{current_skill}\n\n"
            "## Contrastive Evidence\n\n"
            f"{evidence_intro}\n\n"
            + "\n\n".join(contrastive_parts)
            + rejected_section
            + f"\n\n## Instructions\n\n{instructions}"
        )

    @staticmethod
    def reflection_current_skill_section(
        *,
        current_skill_heading: str,
        current_skill: str,
    ) -> str:
        return f"## {current_skill_heading}\n\n{current_skill}"

    @staticmethod
    def reflection_evidence_section(
        *,
        evidence_intro: str,
        contrastive_parts: Iterable[str],
    ) -> str:
        return (
            "## Contrastive Evidence\n\n"
            f"{evidence_intro}\n\n" + "\n\n".join(contrastive_parts)
        )

    @staticmethod
    def reflection_instructions_section(instructions: str) -> str:
        return f"## Instructions\n\n{instructions}"

    @staticmethod
    def reflection_pair(
        *,
        index: int,
        task_id: object,
        worse_context: str,
        worse_entry: str,
        better_context: str,
        better_entry: str,
    ) -> str:
        return (
            f"### Pair {index} — task `{task_id}`\n"
            f"**Worse outcome** ({worse_context}):\n"
            f"{worse_entry}\n\n"
            f"**Better outcome** ({better_context}):\n"
            f"{better_entry}"
        )

    @staticmethod
    def reflection_candidate_instruction(candidate_count: int | None) -> str:
        if candidate_count is not None:
            return (
                f"return JSON with exactly {candidate_count} candidate "
                "skill-refiner updates"
            )
        return "return JSON with 2-3 candidate skill-refiner updates"

    @staticmethod
    def reflection_planner_system(candidate_instruction: str) -> str:
        return (
            "You are reflecting on your performance as a Planner agent. "
            "Your skill-refiner skill defines HOW you decide to edit the "
            "Executor's skills. You will see contrastive pairs: skill-edit "
            "records associated with better vs. worse same-iteration same-task "
            "outcome rewards. Use these to revise your editing strategy.\n\n"
            "If you believe the current guidelines already capture the "
            "lessons from the evidence, include a no_update candidate. "
            f"Otherwise, {candidate_instruction}. Each candidate must include "
            "candidate_id, update_kind, "
            "hypothesis, risk, audit_score, new_content, and reasoning. "
            "new_content must be the complete updated Markdown skill: "
            "integrate changes into existing sections, resolve duplicate "
            "or conflicting rules, and avoid appended addenda unless a new "
            "section is clearly needed."
        )

    @staticmethod
    def rejected_reflection_intro() -> list[str]:
        return [
            f"\n\n{PromptText.REJECTED_REFLECTION_HEADER}",
            PromptText.REJECTED_REFLECTION_GUIDANCE,
        ]

    @staticmethod
    def rejected_reflection_item(
        *,
        iteration: int,
        skill_id: str,
        selected_candidate_id: str | None,
        selected_update_kind: str | None,
        task_ids: str,
        reason: str | None,
        validation_reason: str | None,
        validation_mean_delta: str,
    ) -> str:
        return (
            f"- iteration={iteration} skill={skill_id} "
            f"candidate={selected_candidate_id or 'n/a'} "
            f"kind={selected_update_kind or 'n/a'} tasks={task_ids} "
            f"reason={reason or 'n/a'} "
            f"validation_reason={validation_reason or 'n/a'} "
            f"mean_delta={validation_mean_delta}"
        )

    @staticmethod
    def diffusion_context(section_name: str, rendered_sections: Iterable[str]) -> str:
        lines = [
            f"## {section_name}",
            "",
            PromptText.DIFFUSION_CONTEXT_WARNING,
        ]
        for rendered_section in rendered_sections:
            lines.extend(["", rendered_section])
        return "\n".join(lines)

    @staticmethod
    def diffusion_artifact_block(
        *,
        artifact_id: str,
        source_task_id: str,
        source_iteration: int,
        policy_name: str,
        relation: str,
        risk_level: str,
        content: str,
    ) -> str:
        return "\n".join(
            [
                f"artifact_id={artifact_id}",
                f"source_task={source_task_id}",
                f"source_iteration={source_iteration}",
                f"policy={policy_name}",
                f"relation={relation}",
                f"risk={risk_level}",
                f"content={content}",
            ]
        )

    @staticmethod
    def run_outcome_focus(signal: Literal["success", "failure", "mixed"]) -> str:
        if signal == "success":
            return (
                "This run succeeded. Emphasize what worked and why it is reusable, "
                "while still naming avoidable pitfalls or assumptions."
            )
        if signal == "failure":
            return (
                "This run failed. Emphasize what to avoid and the concrete failure "
                "mode, while still preserving any partial progress or useful setup."
            )
        return (
            "This run had a mixed or partial outcome. Balance what worked, what "
            "failed, and what a later run should verify."
        )

    @staticmethod
    def run_outcome_content(
        *,
        signal: Literal["success", "failure", "mixed"],
        evidence: str,
    ) -> str:
        return (
            f"{PromptText.run_outcome_focus(signal)}\n\n"
            "Return a mixed run-outcome signal with:\n"
            "- what worked or seemed promising\n"
            "- what to avoid or re-check\n"
            "- concrete verifier evidence\n\n"
            f"{evidence}"
        )
