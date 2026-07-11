from __future__ import annotations

import json

import pytest

from mediated_coevo.analysis.evolution_audit import (
    audit_metrics_file,
    build_evolution_audit,
    main,
    write_evolution_audit,
)


def test_evolution_audit_summarizes_iteration_rewards_and_unvalidated_updates():
    rows = [
        {"iteration": 0, "task_id": "task-a", "reward": 1.0, "verifier_status": "ok"},
        {"iteration": 0, "task_id": "task-b", "reward": 0.0, "verifier_status": "ok"},
        {
            "iteration": 0,
            "task_id": "__coevolution__",
            "reward": 1.0,
            "verifier_status": "ok",
        },
        {
            "iteration": 1,
            "task_id": "task-a",
            "reward": 0.0,
            "verifier_status": "ok",
            "skill_updates": [
                {
                    "iteration": 1,
                    "skill_id": "mediator",
                    "task_id": "task-a",
                    "provenance": {
                        "kind": "contrastive_reflection",
                        "decision": "committed",
                        "skill_id": "mediator",
                    },
                }
            ],
        },
        {
            "iteration": 1,
            "task_id": "task-b",
            "reward": 1.0,
            "verifier_status": "harbor_failed",
        },
    ]

    audit = build_evolution_audit(rows, metrics_path="metrics.jsonl")

    assert [summary.iteration for summary in audit.iterations] == [0, 1]
    assert audit.iterations[0].total_runs == 2
    assert audit.iterations[0].scored_count == 2
    assert audit.iterations[0].mean_reward == pytest.approx(0.5)
    assert audit.iterations[0].task_mean_rewards == {"task-a": 1.0, "task-b": 0.0}
    assert audit.iterations[1].total_runs == 2
    assert audit.iterations[1].scored_count == 1
    assert audit.iterations[1].mean_reward == pytest.approx(0.0)

    assert len(audit.skill_updates) == 1
    update = audit.skill_updates[0]
    assert update.skill_id == "mediator"
    assert update.provenance_kind == "contrastive_reflection"
    assert update.has_validation is False

    warnings_by_kind = {warning.kind: warning for warning in audit.warnings}
    assert warnings_by_kind["missing_validation"].skill_id == "mediator"
    regression = warnings_by_kind["update_on_regressed_iteration"]
    assert regression.comparison_iteration == 0
    assert regression.current_mean_reward == pytest.approx(0.0)
    assert regression.comparison_mean_reward == pytest.approx(0.5)
    assert regression.delta == pytest.approx(-0.5)


def test_evolution_audit_extracts_validation_and_candidate_provenance():
    rows = [
        {"iteration": 1, "task_id": "task-a", "reward": 0.5, "verifier_status": "ok"},
        {
            "iteration": 2,
            "task_id": "task-a",
            "reward": 0.75,
            "verifier_status": "ok",
            "skill_updates": [
                {
                    "iteration": 2,
                    "skill_id": "executor",
                    "task_id": "task-a,task-b",
                    "provenance": {
                        "kind": "advisor_batch",
                        "decision": "approved",
                        "skill_id": "executor",
                        "candidate_batch_id": "batch-1-candidates",
                        "selected_candidate_id": "guardrail",
                        "selected_update_kind": "add_failure_guard",
                        "candidate_refs": [
                            {
                                "candidate_id": "no-update-baseline",
                                "update_kind": "no_update",
                                "audit_score": 0.7,
                                "selected": False,
                                "rejection_reason": "no_update",
                            },
                            {
                                "candidate_id": "guardrail",
                                "update_kind": "add_failure_guard",
                                "audit_score": 0.9,
                                "selected": True,
                                "rejection_reason": None,
                            },
                        ],
                        "validation": {
                            "validation_id": "batch-1",
                            "decision": "accepted",
                            "reason": "improved_mean_reward",
                            "current_mean_reward": 0.25,
                            "candidate_mean_reward": 0.5,
                            "mean_delta": 0.25,
                            "task_ids": ["task-c", "task-d"],
                            "task_results": [
                                {
                                    "task_id": "task-c",
                                    "current_status": "ok",
                                    "candidate_status": "ok",
                                }
                            ],
                        },
                    },
                }
            ],
        },
    ]

    audit = build_evolution_audit(rows)

    assert audit.warnings == []
    update = audit.skill_updates[0]
    assert update.skill_id == "executor"
    assert update.decision == "approved"
    assert update.selected_candidate_id == "guardrail"
    assert update.selected_update_kind == "add_failure_guard"
    assert update.candidate_batch_id == "batch-1-candidates"
    assert update.has_validation is True
    assert update.validation_decision == "accepted"
    assert update.validation_reason == "improved_mean_reward"
    assert update.current_mean_reward == pytest.approx(0.25)
    assert update.candidate_mean_reward == pytest.approx(0.5)
    assert update.mean_delta == pytest.approx(0.25)
    assert update.validation_task_ids == ["task-c", "task-d"]
    assert update.validation_task_count == 1
    assert [candidate.candidate_id for candidate in update.candidate_refs] == [
        "no-update-baseline",
        "guardrail",
    ]
    assert update.candidate_refs[1].selected is True


def test_evolution_audit_reports_adjacent_update_effects():
    rows = [
        {"iteration": 0, "task_id": "task-a", "reward": 0.2, "verifier_status": "ok"},
        {"iteration": 0, "task_id": "task-b", "reward": 0.4, "verifier_status": "ok"},
        {
            "iteration": 1,
            "task_id": "task-a",
            "reward": 0.3,
            "verifier_status": "ok",
            "skill_updates": [
                {
                    "iteration": 1,
                    "skill_id": "mediator",
                    "task_id": "task-a,task-b",
                    "provenance": {
                        "kind": "contrastive_reflection",
                        "decision": "committed",
                        "skill_id": "mediator",
                        "selected_candidate_id": "candidate-1",
                        "selected_update_kind": "tighten_feedback",
                    },
                }
            ],
        },
        {"iteration": 1, "task_id": "task-b", "reward": 0.4, "verifier_status": "ok"},
        {"iteration": 2, "task_id": "task-a", "reward": 0.8, "verifier_status": "ok"},
        {"iteration": 2, "task_id": "task-b", "reward": 0.2, "verifier_status": "ok"},
    ]

    audit = build_evolution_audit(rows)

    effect = audit.skill_update_effects[0]
    assert effect.skill_id == "mediator"
    assert effect.selected_candidate_id == "candidate-1"
    assert effect.pre_iteration == 0
    assert effect.post_iteration == 2
    assert effect.pre_mean_reward == pytest.approx(0.3)
    assert effect.update_mean_reward == pytest.approx(0.35)
    assert effect.post_mean_reward == pytest.approx(0.5)
    assert effect.update_delta_from_pre == pytest.approx(0.05)
    assert effect.post_delta_from_update == pytest.approx(0.15)
    assert effect.post_delta_from_pre == pytest.approx(0.2)

    task_effects = {task.task_id: task for task in effect.task_effects}
    assert task_effects["task-a"].post_delta_from_pre == pytest.approx(0.6)
    assert task_effects["task-b"].post_delta_from_pre == pytest.approx(-0.2)


def test_evolution_audit_reports_partial_update_effect_without_post_iteration():
    audit = build_evolution_audit(
        [
            {"iteration": 0, "task_id": "task-a", "reward": 0.2},
            {
                "iteration": 1,
                "task_id": "task-a",
                "reward": 0.3,
                "skill_updates": [{"skill_id": "executor"}],
            },
        ]
    )

    effect = audit.skill_update_effects[0]
    assert effect.skill_id == "executor"
    assert effect.pre_iteration == 0
    assert effect.post_iteration is None
    assert effect.update_delta_from_pre == pytest.approx(0.1)
    assert effect.post_delta_from_update is None
    assert effect.post_delta_from_pre is None


def test_evolution_audit_reports_mediator_report_effects():
    rows = [
        {"iteration": 0, "task_id": "task-a", "reward": 0.2, "verifier_status": "ok"},
        {"iteration": 1, "task_id": "task-a", "reward": 0.7, "verifier_status": "ok"},
    ]
    history_rows = [
        {
            "entry_id": "entry-1",
            "iteration": 0,
            "agent_role": "mediator",
            "payload": {"kind": "mediator", "headline": "focused guidance"},
            "reward": 0.7,
            "metadata": {
                "task_id": "task-a",
                "outcome_task_id": "task-a",
                "outcome_iteration": 1,
                "reward_source": "verifier",
                "verifier_reward": 0.7,
            },
        },
        {
            "entry_id": "entry-2",
            "iteration": 1,
            "agent_role": "mediator",
            "payload": {"kind": "mediator", "headline": "pending"},
            "reward": None,
            "metadata": {"task_id": "task-a"},
        },
    ]

    audit = build_evolution_audit(rows, history_rows=history_rows)

    effect = audit.mediator_report_effects[0]
    assert effect.entry_id == "entry-1"
    assert effect.report_iteration == 0
    assert effect.task_id == "task-a"
    assert effect.outcome_iteration == 1
    assert effect.outcome_reward == pytest.approx(0.7)
    assert effect.previous_iteration == 0
    assert effect.previous_reward == pytest.approx(0.2)
    assert effect.previous_reward_source == "metrics"
    assert effect.delta_from_previous == pytest.approx(0.5)
    assert effect.reward_source == "verifier"
    assert effect.headline == "focused guidance"


def test_mediator_report_effects_compare_judge_rewards_when_available():
    rows = [
        {"iteration": 0, "task_id": "task-a", "reward": 0.8, "verifier_status": "ok"},
        {"iteration": 1, "task_id": "task-a", "reward": 0.3, "verifier_status": "ok"},
    ]
    history_rows = [
        {
            "entry_id": "entry-1",
            "iteration": 0,
            "agent_role": "mediator",
            "payload": {"kind": "mediator", "headline": "bad direction"},
            "reward": 0.0,
            "metadata": {
                "task_id": "task-a",
                "outcome_task_id": "task-a",
                "outcome_iteration": 1,
                "reward_source": "judge",
                "verifier_reward": 0.3,
                "judge_reward": 0.0,
            },
        }
    ]
    judge_rows = [
        {"task_id": "task-a", "iteration": 0, "judge_reward": 0.82},
        {"task_id": "task-a", "iteration": 1, "judge_reward": 0.0},
    ]

    audit = build_evolution_audit(
        rows,
        history_rows=history_rows,
        judge_reward_rows=judge_rows,
    )

    effect = audit.mediator_report_effects[0]
    assert effect.previous_reward == pytest.approx(0.82)
    assert effect.previous_reward_source == "judge"
    assert effect.delta_from_previous == pytest.approx(-0.82)


def test_evolution_audit_includes_skill_update_ledger_rows():
    audit = build_evolution_audit(
        [{"iteration": 2, "task_id": "task-a", "reward": 0.75}],
        skill_update_ledger_rows=[
            {
                "update_id": "iter0002-task-a-executor-candidate-1",
                "iteration": 2,
                "skill_id": "executor",
                "skill_version": "iter_0002",
                "record_task_id": "task-a",
                "update_task_id": "task-a,task-b",
                "selected_candidate_id": "candidate-1",
                "selected_update_kind": "add_guardrail",
                "validation_decision": "accepted",
                "validation_reason": "mean_delta_met",
                "validation_mean_delta": 0.3,
                "reward": 0.75,
                "delta_reward": 0.25,
                "artifact_path": "artifacts/skill_updates/update.json",
                "diff_path": "artifacts/skill_updates/changes.diff",
            }
        ],
    )

    ledger_entry = audit.skill_update_ledger[0]
    assert ledger_entry.update_id == "iter0002-task-a-executor-candidate-1"
    assert ledger_entry.skill_id == "executor"
    assert ledger_entry.skill_version == "iter_0002"
    assert ledger_entry.selected_candidate_id == "candidate-1"
    assert ledger_entry.validation_decision == "accepted"
    assert ledger_entry.validation_mean_delta == pytest.approx(0.3)
    assert ledger_entry.reward == pytest.approx(0.75)
    assert ledger_entry.diff_path == "artifacts/skill_updates/changes.diff"


def test_evolution_audit_includes_rejected_reflection_rows():
    audit = build_evolution_audit(
        [{"iteration": 2, "task_id": "task-a", "reward": 0.75}],
        rejected_reflection_rows=[
            {
                "rejection_id": "reject-1",
                "batch_id": "reflect-planner-iter-0002",
                "iteration": 2,
                "skill_id": "planner",
                "agent_role": "planner",
                "task_ids": ["task-a", "task-b"],
                "reason": "validation_rejected_all_candidates",
                "selected_candidate_id": "candidate-1",
                "selected_update_kind": "broad_rewrite",
                "candidate_batch_id": "candidate-batch-1",
                "candidate_batch_path": "artifacts/validation/candidates.json",
                "candidate_refs": [
                    {
                        "candidate_id": "candidate-1",
                        "update_kind": "broad_rewrite",
                        "audit_score": 0.8,
                        "selected": True,
                        "rejection_reason": None,
                    }
                ],
                "validation": {
                    "decision": "rejected",
                    "reason": "mean_delta_below_threshold",
                    "mean_delta": -0.2,
                },
                "timestamp": "2026-05-25T12:00:00",
            }
        ],
    )

    rejected = audit.rejected_reflections[0]
    assert rejected.rejection_id == "reject-1"
    assert rejected.skill_id == "planner"
    assert rejected.agent_role == "planner"
    assert rejected.task_ids == ["task-a", "task-b"]
    assert rejected.selected_candidate_id == "candidate-1"
    assert rejected.validation_decision == "rejected"
    assert rejected.validation_reason == "mean_delta_below_threshold"
    assert rejected.validation_mean_delta == pytest.approx(-0.2)
    assert rejected.candidate_refs[0].candidate_id == "candidate-1"
    assert rejected.candidate_refs[0].selected is True


def test_evolution_audit_flags_disabled_and_rejected_validation():
    rows = [
        {
            "iteration": 0,
            "task_id": "task-a",
            "reward": 1.0,
            "verifier_status": "ok",
            "skill_updates": [
                {
                    "skill_id": "executor",
                    "provenance": {
                        "kind": "advisor_batch",
                        "skill_id": "executor",
                        "validation": {
                            "decision": "accepted",
                            "reason": "validation_disabled",
                        },
                    },
                },
                {
                    "skill_id": "planner",
                    "provenance": {
                        "kind": "contrastive_reflection",
                        "skill_id": "planner",
                        "validation": {
                            "decision": "rejected",
                            "reason": "mean_delta_below_threshold",
                        },
                    },
                },
            ],
        }
    ]

    audit = build_evolution_audit(rows)

    warnings_by_skill = {warning.skill_id: warning for warning in audit.warnings}
    assert warnings_by_skill["executor"].kind == "validation_disabled"
    assert warnings_by_skill["planner"].kind == "committed_rejected_validation"


def test_evolution_audit_reads_and_writes_stable_json(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"iteration": 0, "task_id": "task-a", "reward": 1.0}) + "\n",
        encoding="utf-8",
    )
    ledger_path = tmp_path / "artifacts" / "skill_updates" / "skill_update_history.jsonl"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_text(
        json.dumps(
            {
                "update_id": "iter0000-task-a-executor-candidate-1",
                "iteration": 0,
                "skill_id": "executor",
                "record_task_id": "task-a",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    judge_path = tmp_path / "artifacts" / "judge_rewards.jsonl"
    judge_path.write_text(
        json.dumps(
            {
                "task_id": "task-a",
                "iteration": 0,
                "judge_reward": 1.0,
            }
        )
        + "\n"
        + json.dumps(
            {
                "task_id": "task-a",
                "iteration": 1,
                "judge_reward": 0.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    history_path = tmp_path / "history" / "history.jsonl"
    history_path.parent.mkdir(parents=True)
    history_path.write_text(
        json.dumps(
            {
                "entry_id": "entry-1",
                "iteration": 0,
                "agent_role": "mediator",
                "payload": {"kind": "mediator", "headline": "mixed result"},
                "reward": 0.5,
                "metadata": {
                    "task_id": "task-a",
                    "outcome_task_id": "task-a",
                    "outcome_iteration": 1,
                    "reward_source": "judge",
                    "judge_reward": 0.5,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    rejected_path = tmp_path / "history" / "rejected_reflections.jsonl"
    rejected_path.write_text(
        json.dumps(
            {
                "rejection_id": "reject-1",
                "batch_id": "reflect-mediator-iter-0000",
                "iteration": 0,
                "skill_id": "mediator",
                "agent_role": "mediator",
                "base_skill_hash": "old-hash",
                "reason": "no_valid_candidates",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    audit = audit_metrics_file(metrics_path)
    audit_path = tmp_path / "evolution_audit.json"
    write_evolution_audit(audit, audit_path)

    loaded = json.loads(audit_path.read_text(encoding="utf-8"))

    assert loaded["metrics_path"] == str(metrics_path)
    assert loaded["experiment_dir"] == str(tmp_path)
    assert loaded["iterations"][0]["mean_reward"] == pytest.approx(1.0)
    assert loaded["skill_update_ledger"][0]["update_id"] == (
        "iter0000-task-a-executor-candidate-1"
    )
    assert loaded["skill_update_effects"][0]["skill_id"] == "executor"
    assert loaded["mediator_report_effects"][0]["entry_id"] == "entry-1"
    assert loaded["mediator_report_effects"][0]["delta_from_previous"] == pytest.approx(
        -0.5
    )
    assert loaded["rejected_reflections"][0]["rejection_id"] == "reject-1"
    assert loaded["rejected_reflections"][0]["skill_id"] == "mediator"


def test_evolution_audit_module_cli_prints_json(tmp_path, capsys):
    metrics_path = tmp_path / "metrics.jsonl"
    metrics_path.write_text(
        json.dumps({"iteration": 0, "task_id": "task-a", "reward": 1.0}) + "\n",
        encoding="utf-8",
    )

    exit_code = main([str(metrics_path)])

    captured = capsys.readouterr()
    loaded = json.loads(captured.out)

    assert exit_code == 0
    assert loaded["metrics_path"] == str(metrics_path)
    assert loaded["iterations"][0]["mean_reward"] == pytest.approx(1.0)
