#!/usr/bin/env python3
"""External logical-batch HWPX diffusion controller.

This script deliberately lives under /tmp and treats the repo CLI as the
execution engine. It implements the job.txt logical iteration semantics:
artifacts produced in logical iteration k are compacted/selected at a checkpoint
and are the only artifacts visible to logical iteration k+1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import secrets
import shutil
import subprocess
import sys
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("/Users/hylee_mac/Documents/Project/mediator-coevo")
SANDBOX = Path("/tmp/agent_diffusion_logical_hwpx_gpt5")
CONFIG_DIR = SANDBOX / "config"
STATE_PATH = SANDBOX / "state.json"
ANALYSIS_DIR = SANDBOX / "analysis"
LOG_DIR = SANDBOX / "logs"
SELECTED_DIR = SANDBOX / "selected"
STORES_DIR = SANDBOX / "stores"
EXPERIMENTS_DIR = SANDBOX / "medcoevo-data" / "experiments"
TASK_ROOT = ROOT / "benchmarks" / "skillflow" / "tasks" / "HWPX-Document-Automation"
FAMILY = "HWPX-Document-Automation"

TASKS = [
    "hwpx-supplier-contact-sheet",
    "hwpx-event-announcement",
    "hwpx-clinic-intake-summary",
    "hwpx-project-proposal",
    "hwpx-training-feedback",
    "hwpx-safety-audit-brief",
    "hwpx-renewal-playbook-update",
    "hwpx-inventory-report",
]

TARGET_GUIDANCE = {
    "hwpx-clinic-intake-summary": {
        "metric": "clinic intake summary completion",
        "formula": "replace all placeholders, add Korean full-year age note, normalize callback phone, preserve labels",
        "weight": "valid HWPX package saved to /root/clinic_intake_ready.hwpx",
        "coalition": "HWPX XML paragraphs and layout-cache cleanup",
    },
    "hwpx-event-announcement": {
        "metric": "event announcement completion",
        "formula": "replace every JSON-backed placeholder while preserving Korean labels and static note",
        "weight": "valid HWPX package saved to /root/event_announcement_ready.hwpx",
        "coalition": "HWPX XML paragraphs and layout-cache cleanup",
    },
    "hwpx-inventory-report": {
        "metric": "inventory report completion",
        "formula": "replace placeholders, keep Korean labels and static note, preserve empty paragraph spacing",
        "weight": "valid HWPX package saved to /root/inventory_report_ready.hwpx",
        "coalition": "HWPX XML paragraphs and layout-cache cleanup",
    },
    "hwpx-project-proposal": {
        "metric": "project proposal completion",
        "formula": "replace placeholders, append phase month spans, normalize budget commas while keeping currency symbol",
        "weight": "valid HWPX package saved to /root/project_proposal_ready.hwpx",
        "coalition": "HWPX XML paragraphs and layout-cache cleanup",
    },
    "hwpx-renewal-playbook-update": {
        "metric": "renewal playbook revision",
        "formula": "update editable customer fields and replace follow-up lines in CSV sequence order without duplicating old values",
        "weight": "valid HWPX package saved to /root/renewal_playbook_updated.hwpx",
        "coalition": "HWPX XML paragraphs and layout-cache cleanup",
    },
    "hwpx-safety-audit-brief": {
        "metric": "safety audit brief completion",
        "formula": "fill overview/table/action fields, rewrite dates, update risk tier, add mapped Korean severity note",
        "weight": "valid HWPX package saved to /root/safety_audit_brief_final.hwpx",
        "coalition": "HWPX XML paragraphs and layout-cache cleanup",
    },
    "hwpx-supplier-contact-sheet": {
        "metric": "supplier contact sheet completion",
        "formula": "replace JSON-backed placeholders while preserving Korean field labels and static note",
        "weight": "valid HWPX package saved to /root/supplier_contact_ready.hwpx",
        "coalition": "HWPX XML paragraphs and layout-cache cleanup",
    },
    "hwpx-training-feedback": {
        "metric": "training feedback sheet completion",
        "formula": "replace placeholders, normalize attendance digits, format satisfaction score, append follow-up sentence",
        "weight": "valid HWPX package saved to /root/training_feedback_ready.hwpx",
        "coalition": "HWPX XML paragraphs and layout-cache cleanup",
    },
}

# Proxy dollar weights copied from docs/WRA-matrix.md.
MODEL_COST_PER_MILLION = {
    "planner": {"prompt": 5.0, "completion": 5.0},
    "mediator": {"prompt": 0.5, "completion": 0.5},
    "judge": {"prompt": 0.0, "completion": 0.0},
    "executor": {"prompt": 5.0, "completion": 25.0, "cache_read": 0.5},
}

MAX_RAW_AFFINITY = 3.25


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def read_first_metric(exp_dir: Path) -> dict[str, Any]:
    metrics = exp_dir / "metrics.jsonl"
    if not metrics.exists():
        return {}
    for line in metrics.read_text(encoding="utf-8").splitlines():
        if line.strip():
            return json.loads(line)
    return {}


def read_summary(exp_dir: Path) -> dict[str, Any]:
    summary = exp_dir / "summary.json"
    if not summary.exists():
        return {}
    return json.loads(summary.read_text(encoding="utf-8"))


def executor_billing(metric: dict[str, Any]) -> dict[str, Any]:
    trial_path = metric.get("harbor_trial_path")
    if not trial_path:
        return {}
    trial_dir = Path(str(trial_path))
    result_path = trial_dir / "result.json"
    billing: dict[str, Any] = {}
    if result_path.exists():
        try:
            trial_result = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            trial_result = {}
        agent_result = trial_result.get("agent_result") or {}
        if isinstance(agent_result, dict):
            cost = agent_result.get("cost_usd")
            if isinstance(cost, (int, float)):
                billing["reported_cost_usd"] = float(cost)
                billing["reported_cost_source"] = "harbor_trial_agent_result"
            for key in ("n_input_tokens", "n_cache_tokens", "n_output_tokens"):
                value = agent_result.get(key)
                if isinstance(value, (int, float)):
                    billing[key] = int(value)

    claude_log = trial_dir / "agent" / "claude-code.txt"
    result_event: dict[str, Any] = {}
    if claude_log.exists():
        for line in claude_log.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("type") == "result":
                result_event = event
        if result_event:
            if "reported_cost_usd" not in billing and isinstance(result_event.get("total_cost_usd"), (int, float)):
                billing["reported_cost_usd"] = float(result_event["total_cost_usd"])
                billing["reported_cost_source"] = "claude_result_event"
            usage = result_event.get("usage") or {}
            if isinstance(usage, dict):
                billing["fresh_input_tokens"] = int(usage.get("input_tokens") or 0)
                billing["cache_read_input_tokens"] = int(usage.get("cache_read_input_tokens") or 0)
                billing["output_tokens"] = int(usage.get("output_tokens") or 0)
            if isinstance(result_event.get("num_turns"), int):
                billing["num_turns"] = result_event["num_turns"]
            if isinstance(result_event.get("duration_api_ms"), (int, float)):
                billing["duration_api_ms"] = result_event["duration_api_ms"]
    return billing


def ensure_config() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    source = ROOT / "config" / "default.toml"
    text = source.read_text(encoding="utf-8")
    text = re.sub(r"data_dir = .*", f'data_dir = "{SANDBOX / "medcoevo-data"}"', text)
    text = re.sub(r"num_iterations = .*", "num_iterations = 1", text)
    text = re.sub(r"coevo_interval = .*", "coevo_interval = 99", text)
    text = re.sub(r"advisor_buffer_max = .*", "advisor_buffer_max = 99", text)
    text = re.sub(r"max_transfer_context_tokens = .*", "max_transfer_context_tokens = 500", text)
    text = re.sub(r"max_same_task_prior_tokens = .*", "max_same_task_prior_tokens = 300", text)
    text = re.sub(r"max_total_prior_context_tokens = .*", "max_total_prior_context_tokens = 1200", text)
    text = re.sub(r"executor = true", "executor = false", text)
    text = re.sub(r"planner = true", "planner = false", text)
    text = re.sub(r"mediator = true", "mediator = false", text)
    text = re.sub(r"enabled = false", "enabled = true", text, count=1)
    text = re.sub(r'policy = "none"', 'policy = "capped_broadcast"', text, count=1)
    text = re.sub(r"max_artifacts = .*", "max_artifacts = 3", text)
    text = re.sub(r"top_k_neighbors = .*", "top_k_neighbors = 3", text)
    (CONFIG_DIR / "default.toml").write_text(text, encoding="utf-8")


def reconstruct_metadata() -> dict[str, Any]:
    ranking = json.loads((TASK_ROOT / "ALL_TASK_DIFFICULTY_RANKING.json").read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for task in TASKS:
        task_dir = TASK_ROOT / task
        instruction = (task_dir / "instruction.md").read_text(encoding="utf-8")
        meta = tomllib.loads((task_dir / "task.toml").read_text(encoding="utf-8")).get("metadata", {})
        guidance = TARGET_GUIDANCE[task]
        tokens = {
            token.lower()
            for token in re.findall(
                r"[A-Za-z]+",
                " ".join(
                    [
                        guidance["metric"],
                        guidance["formula"],
                        guidance["weight"],
                        guidance["coalition"],
                        " ".join(meta.get("tags", [])),
                    ]
                ),
            )
            if len(token) > 2
        }
        out[task] = {
            "task": task,
            "difficulty_rank": ranking.index(task) + 1,
            "declared_difficulty": meta.get("difficulty", "unknown"),
            "category": meta.get("category", "unknown"),
            "tags": meta.get("tags", []),
            "target_guidance": guidance,
            "tokens": sorted(tokens),
            "instruction_digest": " ".join(instruction.split())[:900],
        }
    write_json(ANALYSIS_DIR / "agent_reconstructed_task_metadata.json", out)
    return out


def command(args: list[str], log_name: str) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(args) + "\n")
        log.flush()
        proc = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        log.write(f"\nexit_code={proc.returncode}\n")
    return proc.returncode


def latest_experiment_dir(before: set[Path]) -> Path:
    dirs = {path for path in EXPERIMENTS_DIR.iterdir() if path.is_dir()}
    new_dirs = sorted(dirs - before, key=lambda p: p.stat().st_mtime)
    if not new_dirs:
        raise RuntimeError("no new experiment directory found")
    return new_dirs[-1]


def run_task(task: str, logical_iter: int, artifact_store: Path | None) -> dict[str, Any]:
    before = {path for path in EXPERIMENTS_DIR.iterdir()} if EXPERIMENTS_DIR.exists() else set()
    run_id = f"logicalhwpx-gpt5-L{logical_iter:02d}-{task}"
    args = [
        "uv",
        "run",
        "medcoevo",
        "matrix",
        "--index",
        "1",
        "--task",
        f"{FAMILY}/{task}",
        "--iterations",
        "1",
        "--seed",
        "42",
        "--coevo-interval",
        "99",
        "--advisor-buffer-max",
        "99",
        "--diffusion-max-artifacts",
        "3",
        "--run-id",
        run_id,
        "--config-dir",
        str(CONFIG_DIR),
    ]
    if artifact_store is not None:
        args.extend(["--artifact", str(artifact_store)])
    rc = command(args, f"run-L{logical_iter:02d}-{task}.log")
    exp_dir = latest_experiment_dir(before)
    extract_rc = command(
        [
            "uv",
            "run",
            "medcoevo",
            "extract",
            "--path",
            str(exp_dir),
            "--output-dir",
            str(STORES_DIR),
        ],
        f"extract-L{logical_iter:02d}-{task}.log",
    )
    store = STORES_DIR / exp_dir.name
    metric = read_first_metric(exp_dir)
    summary = read_summary(exp_dir)
    run = {
        "logical_iter": logical_iter,
        "task": task,
        "artifact_store": str(artifact_store) if artifact_store else None,
        "experiment_dir": str(exp_dir),
        "exported_store": str(store) if store.exists() else None,
        "return_code": rc,
        "extract_return_code": extract_rc,
        "verifier_reward": summary.get("mean_reward"),
        "judge_reward": (summary.get("judge_reward_summary") or {}).get("mean_reward"),
        "env_failure_count": summary.get("env_failure_count"),
        "metrics": metric,
        "cost": estimate_cost(metric),
        "timestamp": now(),
    }
    append_jsonl(ANALYSIS_DIR / "runs.jsonl", run)
    append_jsonl(ANALYSIS_DIR / "costs.jsonl", {"task": task, **run["cost"], "logical_iter": logical_iter})
    return run


def estimate_cost(metric: dict[str, Any]) -> dict[str, Any]:
    prompt = metric.get("prompt_tokens_by_agent") or {}
    completion = metric.get("completion_tokens_by_agent") or {}
    total = metric.get("total_tokens") or 0
    cost = 0.0
    parts: dict[str, float] = {}
    sources: dict[str, str] = {}
    billing = executor_billing(metric)
    for role in ("planner", "mediator", "judge"):
        rates = MODEL_COST_PER_MILLION[role]
        role_cost = (
            float(prompt.get(role) or 0) * rates["prompt"]
            + float(completion.get(role) or 0) * rates["completion"]
        ) / 1_000_000
        parts[role] = role_cost
        sources[role] = "proxy"
        cost += role_cost

    executor_rates = MODEL_COST_PER_MILLION["executor"]
    executor_proxy_cost = (
        float(prompt.get("executor") or 0) * executor_rates["prompt"]
        + float(completion.get("executor") or 0) * executor_rates["completion"]
        + float(metric.get("executor_cache_read_tokens") or 0) * executor_rates["cache_read"]
    ) / 1_000_000
    reported_executor_cost = billing.get("reported_cost_usd")
    if isinstance(reported_executor_cost, (int, float)):
        executor_cost = float(reported_executor_cost)
        sources["executor"] = str(billing.get("reported_cost_source") or "reported")
    else:
        executor_cost = executor_proxy_cost
        sources["executor"] = "proxy_fallback"
    parts["executor"] = executor_cost
    cost += executor_cost
    return {
        "proxy_cost_usd": cost,
        "cost_parts": parts,
        "cost_sources": sources,
        "executor_proxy_cost_usd": executor_proxy_cost,
        "executor_reported_cost_usd": reported_executor_cost,
        "executor_billing": billing,
        "total_tokens": total,
        "token_split": metric.get("total_tokens_by_agent") or {},
        "prompt_tokens_by_agent": prompt,
        "completion_tokens_by_agent": completion,
    }


def load_store_artifacts(store: Path) -> list[dict[str, Any]]:
    artifacts_dir = store / "artifacts"
    if not artifacts_dir.exists():
        return []
    artifacts: list[dict[str, Any]] = []
    for path in sorted(artifacts_dir.glob("*.json")):
        try:
            artifacts.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return artifacts


def source_signal(run: dict[str, Any]) -> dict[str, Any]:
    store = Path(run["exported_store"] or "")
    artifacts = load_store_artifacts(store)
    run_outcome = next((a for a in artifacts if str(a.get("artifact_type")).endswith("run_outcome")), None)
    mediator = next((a for a in artifacts if a.get("artifact_type") == "mediator_report_summary"), None)
    debug = next((a for a in artifacts if a.get("artifact_type") == "debug_hint"), None)
    content = "\n".join(str(a.get("content") or "") for a in (run_outcome, mediator, debug) if a)
    return {
        "task": run["task"],
        "verifier_reward": run.get("verifier_reward"),
        "judge_reward": run.get("judge_reward"),
        "store": run.get("exported_store"),
        "run_outcome": run_outcome,
        "mediator": mediator,
        "debug": debug,
        "content": content,
        "has_name_error": "#NAME?" in content,
        "has_none_error": "None" in content or "empty" in content.lower(),
        "mentions_cached_xml": any(x in content.lower() for x in ("cached", "<v>", "data_only", "xml")),
        "has_placeholder_error": "{{" in content or "placeholder" in content.lower(),
        "mentions_layout_cache": any(x in content.lower() for x in ("layout-cache", "overlap", "overlapping", "stale layout")),
        "mentions_hwpx_validity": any(x in content.lower() for x in ("hwpx", "zip", "package", "valid")),
    }


def edge_score(metadata: dict[str, Any], completed: set[str], source: dict[str, Any], target: str) -> tuple[float, str]:
    src = metadata[source["task"]]
    dst = metadata[target]
    src_tokens = set(src["tokens"])
    dst_tokens = set(dst["tokens"])
    lexical = len(src_tokens & dst_tokens) / max(1, len(src_tokens | dst_tokens))
    rank_gap = abs(int(src["difficulty_rank"]) - int(dst["difficulty_rank"]))
    rank_score = 1.0 / (1.0 + rank_gap)
    reward = source.get("verifier_reward")
    reward_value = reward if isinstance(reward, (int, float)) else 0.0
    success_bonus = 0.35 if reward_value >= 1.0 else 0.0
    failure_repair_bonus = 0.45 if source["has_name_error"] or source["has_none_error"] or source["has_placeholder_error"] else 0.0
    cached_bonus = 0.25 if source["mentions_cached_xml"] or source["mentions_layout_cache"] or source["mentions_hwpx_validity"] else 0.0
    novelty_bonus = 0.25 if target not in completed else -0.15
    same_difficulty = 0.1 if src["declared_difficulty"] == dst["declared_difficulty"] else 0.0
    raw_affinity = 1.15 * lexical + 0.70 * rank_score + success_bonus + failure_repair_bonus + cached_bonus + novelty_bonus + same_difficulty
    score = max(-1.0, min(1.0, (2.0 * raw_affinity / MAX_RAW_AFFINITY) - 1.0))
    rationale = (
        f"bounded_similarity={score:.3f}; raw_affinity={raw_affinity:.3f}; lexical={lexical:.2f}; rank_gap={rank_gap}; success_bonus={success_bonus:.2f}; "
        f"failure_repair_bonus={failure_repair_bonus:.2f}; cached_bonus={cached_bonus:.2f}; novelty_bonus={novelty_bonus:.2f}"
    )
    return score, rationale


def softmax(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_score = max(item["similarity_index"] for item in items)
    weights = [math.exp(item["similarity_index"] - max_score) for item in items]
    total = sum(weights)
    return [{**item, "probability": weight / total} for item, weight in zip(items, weights, strict=True)]


def checkpoint(
    state: dict[str, Any],
    logical_iter: int,
    source_runs: list[dict[str, Any]],
    rng: random.Random,
    *,
    overwrite_decisions: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    metadata = state["metadata"]
    completed = {run["task"] for run in state["runs"] if run.get("verifier_reward") == 1.0}
    active_next: dict[str, list[dict[str, Any]]] = {}
    for run in source_runs:
        source = source_signal(run)
        if not source.get("store"):
            continue
        candidates = []
        for target in TASKS:
            if target == source["task"]:
                continue
            score, rationale = edge_score(metadata, completed, source, target)
            candidates.append(
                {
                    "source": source["task"],
                    "target": target,
                    "similarity_index": score,
                    "rationale": rationale,
                    "source_store": source["store"],
                    "source_verifier_reward": source["verifier_reward"],
                    "source_judge_reward": source["judge_reward"],
                }
            )
        candidates.sort(key=lambda item: (-item["similarity_index"], item["target"]))
        top = candidates[:3]
        distribution = softmax(top)
        marker = rng.random()
        cumulative = 0.0
        selected = distribution[-1]
        for item in distribution:
            cumulative += item["probability"]
            if marker <= cumulative:
                selected = item
                break
        decision = {
            "logical_iter": logical_iter,
            "source": source["task"],
            "candidate_distribution": distribution,
            "random_marker": marker,
            "selected": selected["target"],
            "selected_probability": selected["probability"],
            "selected_similarity_index": selected["similarity_index"],
            "timestamp": now(),
        }
        if overwrite_decisions and not (ANALYSIS_DIR / "softmax_decisions.raw_unbounded_before_fix.jsonl").exists():
            old_path = ANALYSIS_DIR / "softmax_decisions.jsonl"
            if old_path.exists():
                (ANALYSIS_DIR / "softmax_decisions.raw_unbounded_before_fix.jsonl").write_text(
                    old_path.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
                old_path.write_text("", encoding="utf-8")
        append_jsonl(ANALYSIS_DIR / "softmax_decisions.jsonl", decision)
        active_next.setdefault(selected["target"], []).append({**selected, "source_signal": source})
    write_json(ANALYSIS_DIR / f"checkpoint_L{logical_iter:02d}.json", active_next)
    return active_next


def artifact_content(source_signal_: dict[str, Any], target: str) -> str:
    guidance = TARGET_GUIDANCE[target]
    source_task = source_signal_["task"]
    outcome = (
        f"source verifier={source_signal_.get('verifier_reward')} judge={source_signal_.get('judge_reward')}; "
        f"placeholder_error={source_signal_.get('has_placeholder_error')} none_error={source_signal_.get('has_none_error')} "
        f"layout_cache={source_signal_.get('mentions_layout_cache')} valid_hwpx={source_signal_.get('mentions_hwpx_validity')}"
    )
    return "\n".join(
        [
            f"Logical-batch selected transfer from {source_task} to {target}.",
            f"Source signal: {outcome}.",
            "This artifact was selected only from the previous logical iteration checkpoint; do not use same-iteration artifacts.",
            "Use target-specific document names, output path, JSON/CSV files, and Korean labels only; do not copy source field names or values.",
            f"Target objective: {guidance['metric']}.",
            f"Target-specific transformation: {guidance['formula']}.",
            f"Verifier output contract: {guidance['weight']}.",
            f"Shared HWPX method: {guidance['coalition']}.",
            "Required HWPX mechanics: inspect the .hwpx package as a zip, edit the relevant XML text runs/paragraphs, and preserve the package structure.",
            "Replace every required placeholder or editable value, preserve static Korean labels/notes, and keep any task-specified formatting conversions.",
            "Verifier-sensitive repair: any paragraph whose text is modified must not retain stale layout-cache elements, because stale caches can make opened documents show overlapping characters.",
            "Validate by checking the output file exists, remains a valid .hwpx zip/package, contains no leftover placeholders when applicable, and preserves required static text.",
            "Mediator output from the source run is part of the source signal and should be used to avoid repeated failure modes.",
        ]
    )


def make_target_store(logical_iter: int, target: str, incoming: list[dict[str, Any]]) -> Path:
    store = SELECTED_DIR / f"L{logical_iter:02d}-{target}"
    if store.exists():
        shutil.rmtree(store)
    artifacts_dir = store / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Keep up to 3 sources, matching job.txt and CLI max_artifacts.
    incoming_sorted = sorted(incoming, key=lambda item: -float(edge_similarity(item)))[:3]
    for idx, edge in enumerate(incoming_sorted, start=1):
        source = edge["source_signal"]
        similarity_index = edge_similarity(edge)
        artifact = {
            "artifact_id": f"logical-L{logical_iter:02d}-{idx}-{source['task']}-to-{target}",
            "source_task_id": f"{FAMILY}/{source['task']}",
            "source_iteration": logical_iter - 1,
            "source_run_id": Path(str(source.get("store") or "")).name,
            "artifact_type": "run_outcome",
            "risk_level": "low" if source.get("verifier_reward") == 1.0 else "medium",
            "content": artifact_content(source, target),
            "evidence_trace_ids": [f"{FAMILY}/{source['task']}:logical_iter_{logical_iter - 1}"],
            "evidence_report_ids": [],
            "verifier_reward": source.get("verifier_reward"),
            "judge_reward": source.get("judge_reward"),
            "token_cost": len(artifact_content(source, target).split()),
            "ttl_iterations": 1,
            "created_at": now(),
            "metadata": {
                "logical_iteration_source": logical_iter - 1,
                "logical_iteration_target": logical_iter,
                "diffusion_channel": "reuse_success" if source.get("verifier_reward") == 1.0 else "avoid_recheck",
                "agent_selected_similarity_index": similarity_index,
                "agent_selected_probability": edge.get("selected_probability", edge.get("probability")),
                "agent_selected_rationale": edge["rationale"],
                "intended_target_task_id": f"{FAMILY}/{target}",
                "source_store": source.get("store"),
                "compacted_by_external_controller": True,
            },
        }
        write_json(artifacts_dir / f"{artifact['artifact_id']}.json", artifact)
    write_json(
        store / "manifest.json",
        {
            "id": store.name,
            "artifact_count": len(incoming_sorted),
            "source": "logical_batch_external_controller",
            "intended_target_task_id": f"{FAMILY}/{target}",
            "sources": [edge["source"] for edge in incoming_sorted],
            "logical_iteration": logical_iter,
        },
    )
    return store


def edge_similarity(edge: dict[str, Any]) -> float:
    return float(edge.get("selected_similarity_index", edge.get("similarity_index", 0.0)))


def summarize(state: dict[str, Any]) -> None:
    runs = state["runs"]
    total_cost = sum(float(run.get("cost", {}).get("proxy_cost_usd") or 0.0) for run in runs)
    total_tokens = sum(int(run.get("cost", {}).get("total_tokens") or 0) for run in runs)
    success_count = sum(1 for run in runs if run.get("verifier_reward") == 1.0)
    lines = [
        "# Logical Batch HWPX Diffusion Report",
        "",
        f"Updated: {now()}",
        "",
        "## Seed Batch",
        "",
        f"- RNG seed: `{state['rng_seed']}`",
        f"- Seed tasks: {', '.join(state['seed_tasks'])}",
        "",
        "## Aggregate",
        "",
        f"- Runs: {len(runs)}",
        f"- Verifier successes: {success_count}/{len(runs)}",
        f"- Total tokens: {total_tokens}",
        f"- Proxy dollar cost: ${total_cost:.4f}",
        "- Cost model: hybrid. Executor uses Claude/Harbor reported cost when available; planner uses $5/M, mediator+compactor $0.5/M, judge $0/M. Executor proxy fallback is input $5/M, output $25/M, cache read $0.5/M.",
        "",
        "## Runs",
        "",
        "| Logical iter | Task | Sources | Tokens | Proxy $ | Verifier | Judge | Transfer tokens | Budget violation | Compacted artifacts | Dropped artifacts |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for run in runs:
        metric = run.get("metrics") or {}
        sources = ",".join(metric.get("source_task_ids") or [])
        compacted = ",".join(metric.get("compacted_diffusion_artifact_ids") or [])
        dropped = ",".join(metric.get("dropped_for_budget_artifact_ids") or [])
        lines.append(
            "| {logical_iter} | `{task}` | {sources} | {tokens} | ${cost:.4f} | {verifier} | {judge} | {transfer} | {violation} | {compacted} | {dropped} |".format(
                logical_iter=run["logical_iter"],
                task=run["task"],
                sources=sources or "seed",
                tokens=run.get("cost", {}).get("total_tokens"),
                cost=float(run.get("cost", {}).get("proxy_cost_usd") or 0.0),
                verifier=run.get("verifier_reward"),
                judge=run.get("judge_reward"),
                transfer=metric.get("transfer_context_tokens"),
                violation=metric.get("context_budget_violation"),
                compacted=compacted or "-",
                dropped=dropped or "-",
            )
        )
    lines.extend(
        [
            "",
            "## Softmax Gate Records",
            "",
            "Each source selected one target from up to three candidates. Candidate distributions are recorded in `analysis/softmax_decisions.jsonl`.",
            "",
            "## Notes",
            "",
            "- Logical iteration k+1 used only selected stores built from logical iteration k artifacts.",
            "- Physical runs were sequential, but same-logical-iteration tasks did not consume one another's artifacts.",
            "- Mediator was enabled through the `learned_mediator` condition and emitted mediator summaries in exported stores.",
            "- Existing infra compaction was available through `_fit_prior_context_bundle`; budget and compaction telemetry are reported from `metrics.jsonl`.",
        ]
    )
    (ANALYSIS_DIR / "logical_batch_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def init_state(seed_count: int) -> dict[str, Any]:
    ensure_config()
    metadata = reconstruct_metadata()
    rng_seed = secrets.randbits(63)
    rng = random.Random(rng_seed)
    seed_tasks = rng.sample(TASKS, seed_count)
    state = {
        "created_at": now(),
        "family": FAMILY,
        "rng_seed": rng_seed,
        "seed_tasks": seed_tasks,
        "metadata": metadata,
        "runs": [],
        "logical_iterations": [],
    }
    write_json(STATE_PATH, state)
    write_json(ANALYSIS_DIR / "seed_selection.json", {"rng_seed": rng_seed, "seed_tasks": seed_tasks, "created_at": now()})
    return state


def run_workflow(max_logical_iters: int, seed_count: int, min_rows: int) -> None:
    state = init_state(seed_count)
    rng = random.Random(state["rng_seed"])
    active = {task: [] for task in state["seed_tasks"]}
    logical_iter = 0
    while active and logical_iter < max_logical_iters:
        iter_record = {"logical_iter": logical_iter, "active_tasks": sorted(active), "started_at": now(), "runs": []}
        source_runs: list[dict[str, Any]] = []
        for task in sorted(active):
            incoming = active[task]
            store = make_target_store(logical_iter, task, incoming) if incoming else None
            run = run_task(task, logical_iter, store)
            state["runs"].append(run)
            iter_record["runs"].append(run)
            source_runs.append(run)
            write_json(STATE_PATH, state)
            summarize(state)
        iter_record["finished_at"] = now()
        state["logical_iterations"].append(iter_record)
        write_json(STATE_PATH, state)
        covered = {run["task"] for run in state["runs"]}
        if len(state["runs"]) >= min_rows:
            break
        active = checkpoint(state, logical_iter, source_runs, rng)
        # If all selected targets have already been run, force remaining tasks into
        # the next logical batch using the best previous source. This keeps the run
        # finite while preserving previous-iteration artifact semantics.
        remaining = [task for task in TASKS if task not in covered]
        if remaining:
            selected_targets = set(active)
            missing = [task for task in remaining if task not in selected_targets]
            if len(selected_targets) < min(3, len(remaining)) and missing:
                best_source_run = max(source_runs, key=lambda r: (r.get("verifier_reward") == 1.0, r.get("judge_reward") or 0))
                best_signal = source_signal(best_source_run)
                for target in missing[: max(0, min(3, len(remaining)) - len(selected_targets))]:
                    score, rationale = edge_score(state["metadata"], covered, best_signal, target)
                    active.setdefault(target, []).append(
                        {
                            "source": best_signal["task"],
                            "target": target,
                            "similarity_index": score,
                            "selected_similarity_index": score,
                            "selected_probability": 1.0,
                            "rationale": "coverage backfill after softmax; " + rationale,
                            "source_store": best_signal["store"],
                            "source_signal": best_signal,
                        }
                    )
        logical_iter += 1
    state["finished_at"] = now()
    write_json(STATE_PATH, state)
    summarize(state)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def advance_rng_for_existing_decisions(rng: random.Random) -> None:
    for _ in load_jsonl(ANALYSIS_DIR / "softmax_decisions.jsonl"):
        rng.random()


def resume_from_checkpoint(start_logical_iter: int, max_logical_iters: int, min_rows: int) -> None:
    state = load_json(STATE_PATH, {})
    if not state:
        raise RuntimeError(f"missing state file: {STATE_PATH}")
    state["runs"] = load_jsonl(ANALYSIS_DIR / "runs.jsonl")
    rng = random.Random(state["rng_seed"])
    advance_rng_for_existing_decisions(rng)
    active = load_json(ANALYSIS_DIR / f"checkpoint_L{start_logical_iter - 1:02d}.json", {})
    logical_iter = start_logical_iter
    while active and logical_iter < max_logical_iters:
        already_run = {run["task"] for run in state["runs"]}
        if len(state["runs"]) >= min_rows:
            break
        if len(state["runs"]) >= len(TASKS):
            active = dict(active)
        else:
            active = {
                task: incoming
                for task, incoming in active.items()
                if task not in already_run or should_rerun_activated_task(state, task, incoming)
            }
        if not active:
            break
        iter_record = {"logical_iter": logical_iter, "active_tasks": sorted(active), "started_at": now(), "runs": []}
        source_runs: list[dict[str, Any]] = []
        for task in sorted(active):
            incoming = active[task]
            store = make_target_store(logical_iter, task, incoming) if incoming else None
            run = run_task(task, logical_iter, store)
            state["runs"].append(run)
            iter_record["runs"].append(run)
            source_runs.append(run)
            write_json(STATE_PATH, state)
            summarize(state)
        iter_record["finished_at"] = now()
        state.setdefault("logical_iterations", []).append(iter_record)
        write_json(STATE_PATH, state)
        covered = {run["task"] for run in state["runs"]}
        if len(state["runs"]) >= min_rows:
            break
        active = checkpoint(state, logical_iter, source_runs, rng)
        remaining = [task for task in TASKS if task not in covered]
        if remaining:
            selected_targets = set(active)
            missing = [task for task in remaining if task not in selected_targets]
            if len(selected_targets) < min(3, len(remaining)) and missing:
                best_source_run = max(source_runs, key=lambda r: (r.get("verifier_reward") == 1.0, r.get("judge_reward") or 0))
                best_signal = source_signal(best_source_run)
                for target in missing[: max(0, min(3, len(remaining)) - len(selected_targets))]:
                    score, rationale = edge_score(state["metadata"], covered, best_signal, target)
                    active.setdefault(target, []).append(
                        {
                            "source": best_signal["task"],
                            "target": target,
                            "similarity_index": score,
                            "selected_similarity_index": score,
                            "selected_probability": 1.0,
                            "rationale": "coverage backfill after softmax; " + rationale,
                            "source_store": best_signal["store"],
                            "source_signal": best_signal,
                        }
                    )
        logical_iter += 1
    state["finished_at"] = now()
    write_json(STATE_PATH, state)
    summarize(state)


def should_rerun_activated_task(state: dict[str, Any], task: str, incoming: list[dict[str, Any]]) -> bool:
    if len(incoming) > 1:
        return True
    prior_rewards = [
        run.get("verifier_reward")
        for run in state.get("runs", [])
        if run.get("task") == task and isinstance(run.get("verifier_reward"), (int, float))
    ]
    return bool(incoming) and bool(prior_rewards) and max(prior_rewards) < 1.0


def recompute_checkpoint(logical_iter: int) -> None:
    state = load_json(STATE_PATH, {})
    if not state:
        raise RuntimeError(f"missing state file: {STATE_PATH}")
    state["runs"] = load_jsonl(ANALYSIS_DIR / "runs.jsonl")
    source_runs = [run for run in state["runs"] if int(run.get("logical_iter", -1)) == logical_iter]
    if not source_runs:
        raise RuntimeError(f"no source runs found for logical iteration {logical_iter}")
    rng = random.Random(state["rng_seed"])
    for prior_iter in range(logical_iter):
        prior_count = sum(1 for run in state["runs"] if int(run.get("logical_iter", -1)) == prior_iter)
        for _ in range(prior_count):
            rng.random()
    checkpoint(state, logical_iter, source_runs, rng, overwrite_decisions=(logical_iter == 0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-logical-iters", type=int, default=4)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--min-rows", type=int, default=10)
    parser.add_argument("--resume-from-logical-iter", type=int)
    parser.add_argument("--recompute-checkpoint", type=int)
    args = parser.parse_args()
    if args.recompute_checkpoint is not None:
        recompute_checkpoint(args.recompute_checkpoint)
    elif args.resume_from_logical_iter is not None:
        resume_from_checkpoint(args.resume_from_logical_iter, args.max_logical_iters, args.min_rows)
    else:
        run_workflow(args.max_logical_iters, args.seed_count, args.min_rows)


if __name__ == "__main__":
    main()
