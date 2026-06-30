#!/usr/bin/env python3
"""External logical-batch HWPX diffusion controller.

This script deliberately lives under the repo-copied tmp directory and treats the repo CLI as the
execution engine. It implements the job.txt logical iteration semantics:
artifacts produced in logical iteration k are compacted/selected at a checkpoint
and are the only artifacts visible to logical iteration k+1.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import secrets
import shutil
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mediated_coevo.llm.client import LLMClient


ROOT = Path("/Users/hylee_mac/Documents/Project/mediator-coevo")
SANDBOX = ROOT / "tmp" / "agent_diffusion_logical_hwpx_llmrouter_temp_gpt52_20260630"
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
        "document": "clinic intake summary",
        "sources": "patient_intake.json",
        "output": "/root/clinic_intake_ready.hwpx",
        "operations": [
            "replace all placeholders including repeated patient-name occurrences",
            "append Korean full-year age note after birth date",
            "normalize callback phone number",
            "preserve Korean labels and handwritten-signature note",
        ],
    },
    "hwpx-event-announcement": {
        "document": "event announcement",
        "sources": "event_data.json",
        "output": "/root/event_announcement_ready.hwpx",
        "operations": [
            "replace all placeholders from JSON values",
            "preserve Korean labels and static note line",
        ],
    },
    "hwpx-inventory-report": {
        "document": "inventory status report",
        "sources": "inventory_data.json",
        "output": "/root/inventory_report_ready.hwpx",
        "operations": [
            "replace all placeholders from JSON values",
            "preserve Korean labels, static note line, and empty paragraphs",
        ],
    },
    "hwpx-project-proposal": {
        "document": "project proposal",
        "sources": "project_proposal.json",
        "output": "/root/project_proposal_ready.hwpx",
        "operations": [
            "replace placeholders across both sections",
            "append phase month spans",
            "normalize budget by removing commas while keeping currency symbol",
            "preserve Korean labels and static note line",
        ],
    },
    "hwpx-renewal-playbook-update": {
        "document": "renewal playbook update",
        "sources": "renewal_update.json and followups.csv",
        "output": "/root/renewal_playbook_updated.hwpx",
        "operations": [
            "update customer, owner, renewal window, pricing band, escalation contact, and pricing note",
            "replace follow-up lines in CSV sequence order",
            "remove old values without duplicate lines",
            "preserve the appendix sentence unchanged",
        ],
    },
    "hwpx-safety-audit-brief": {
        "document": "warehouse safety audit brief",
        "sources": "audit_overview.json and corrective_actions.json",
        "output": "/root/safety_audit_brief_final.hwpx",
        "operations": [
            "fill overview fields, audit table values, and corrective-action lines",
            "update every risk-tier occurrence",
            "rewrite inspection date as YYYY.MM.DD",
            "append severity note from risk tier",
            "preserve section titles and row labels",
        ],
    },
    "hwpx-supplier-contact-sheet": {
        "document": "supplier contact sheet",
        "sources": "supplier_contact.json",
        "output": "/root/supplier_contact_ready.hwpx",
        "operations": [
            "replace all placeholders from JSON values",
            "preserve Korean field labels and static note line",
        ],
    },
    "hwpx-training-feedback": {
        "document": "training feedback sheet",
        "sources": "training_feedback.json",
        "output": "/root/training_feedback_ready.hwpx",
        "operations": [
            "replace placeholders across both sections",
            "convert attendee count to digits only",
            "rewrite satisfaction score as 점 format",
            "append the required follow-up sentence to overall opinion",
            "preserve Korean labels and static note line",
        ],
    },
}

# Proxy dollar weights copied from the WRA logical controller.
MODEL_COST_PER_MILLION = {
    "planner": {"prompt": 5.0, "completion": 5.0},
    "mediator": {"prompt": 0.5, "completion": 0.5},
    "judge": {"prompt": 0.0, "completion": 0.0},
    "executor": {"prompt": 5.0, "completion": 25.0, "cache_read": 0.5},
}

RUN_LABEL = "logicalhwpx-llmrouter-temp-gpt52-20260630"
MAX_RAW_AFFINITY = 3.35
SOFTMAX_TEMPERATURE = 0.35
ROUTER_MODEL = "openrouter/openai/gpt-5.2"
ROUTER_LLM_WEIGHT = 0.30
MIN_ACTIVE_TASKS = 2
CONSECUTIVE_ITERATION_LIMIT = 2
ROUTING_MEMORY_LAST_K = 2


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
    text = re.sub(r'executor = ".*"', 'executor = "openrouter/openai/gpt-5.2"', text, count=1)
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
                        guidance["document"],
                        guidance["sources"],
                        guidance["output"],
                        " ".join(guidance["operations"]),
                        " ".join(meta.get("tags", [])),
                    ]
                ),
            )
            if len(token) > 2
        }
        instruction_lower = instruction.lower()
        contract_areas = ["hwpx_package_integrity", "layout_cache_cleanup", "placeholder_replacement"]
        known_risks = ["output_format_error", "stale_layout_cache", "unreplaced_placeholder"]
        if "json" in instruction_lower or "csv" in instruction_lower:
            contract_areas.append("source_data_mapping")
            known_risks.append("field_mapping_error")
        if "korean" in instruction_lower or "static note" in instruction_lower or "labels" in instruction_lower:
            contract_areas.append("korean_label_preservation")
            known_risks.append("static_text_regression")
        if "valid `.hwpx` package" in instruction_lower or "valid .hwpx package" in instruction_lower:
            contract_areas.append("zip_xml_structure")
            known_risks.append("invalid_hwpx_package")
        out[task] = {
            "task": task,
            "difficulty_rank": ranking.index(task) + 1,
            "declared_difficulty": meta.get("difficulty", "unknown"),
            "category": meta.get("category", "unknown"),
            "tags": meta.get("tags", []),
            "target_guidance": guidance,
            "tokens": sorted(tokens),
            "instruction_digest": " ".join(instruction.split())[:900],
            "target_profile": {
                "domain": "hwpx_document",
                "contract_areas": sorted(set(contract_areas)),
                "known_risks": sorted(set(known_risks)),
            },
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
    run_id = f"{RUN_LABEL}-L{logical_iter:02d}-{task}"
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


def evidence_snippet(content: str, needle: str) -> str | None:
    lower = content.lower()
    index = lower.find(needle.lower())
    if index < 0:
        return None
    start = max(0, index - 70)
    end = min(len(content), index + len(needle) + 100)
    return " ".join(content[start:end].split())


def extract_artifact_signal(
    *,
    task: str,
    verifier_reward: Any,
    judge_reward: Any,
    artifacts: list[dict[str, Any]],
    content: str,
) -> dict[str, Any]:
    """Return an LLM-compatible structured transfer signal.

    This deterministic extractor is the tmp implementation of the generalized
    signal interface. A later LLM extractor can emit the same schema with
    evidence spans, while the routing code remains deterministic.
    """

    lower = content.lower()
    evidence: list[str] = []
    failure_classes: set[str] = set()
    contract_areas: set[str] = set()
    repair_patterns: set[str] = set()

    domain = "hwpx_document" if any(
        marker in lower
        for marker in ("hwpx", "placeholder", "layout-cache", "layout cache", "korean", "paragraph", "xml")
    ) else "generic"

    if "placeholder" in lower or "{{" in lower:
        failure_classes.add("unreplaced_placeholder")
        contract_areas.add("placeholder_replacement")
        repair_patterns.add("replace_all_placeholders")
        snippet = evidence_snippet(content, "placeholder") or evidence_snippet(content, "{{")
        if snippet:
            evidence.append(snippet)
    if "layout-cache" in lower or "layout cache" in lower or "overlapping" in lower:
        failure_classes.add("stale_layout_cache")
        contract_areas.add("layout_cache_cleanup")
        repair_patterns.add("remove_stale_layout_cache")
        snippet = evidence_snippet(content, "layout-cache") or evidence_snippet(content, "overlapping")
        if snippet:
            evidence.append(snippet)
    if "hwpx" in lower or "valid package" in lower or "package" in lower:
        contract_areas.add("hwpx_package_integrity")
        repair_patterns.add("preserve_hwpx_zip_structure")
        snippet = evidence_snippet(content, "hwpx") or evidence_snippet(content, "package")
        if snippet:
            evidence.append(snippet)
    if "json" in lower or "csv" in lower:
        contract_areas.add("source_data_mapping")
        repair_patterns.add("map_source_fields_exactly")
    if "korean" in lower or "static note" in lower or "labels" in lower:
        contract_areas.add("korean_label_preservation")
        repair_patterns.add("preserve_static_korean_text")

    if "#name?" in lower:
        failure_classes.add("spreadsheet_formula_error")
        contract_areas.add("computed_cell_values")
        repair_patterns.add("use_compatible_formula_names")
        snippet = evidence_snippet(content, "#NAME?")
        if snippet:
            evidence.append(snippet)
    if "none" in lower or "empty" in lower or "missing" in lower:
        failure_classes.add("missing_or_empty_output")
        contract_areas.add("required_outputs")
        repair_patterns.add("fill_required_outputs")
        snippet = evidence_snippet(content, "empty") or evidence_snippet(content, "None")
        if snippet:
            evidence.append(snippet)
    if any(marker in lower for marker in ("cached", "<v>", "data_only")):
        failure_classes.add("cached_value_issue")
        contract_areas.add("cached_values")
        repair_patterns.add("populate_cached_values")
        snippet = (
            evidence_snippet(content, "cached")
            or evidence_snippet(content, "data_only")
            or evidence_snippet(content, "<v>")
        )
        if snippet:
            evidence.append(snippet)
    if any(marker in lower for marker in ("lookup", "index", "match", "absolute reference", "relative")):
        failure_classes.add("reference_alignment_error")
        contract_areas.add("lookup_blocks")
        repair_patterns.add("preserve_relative_references")
        snippet = evidence_snippet(content, "lookup") or evidence_snippet(content, "relative")
        if snippet:
            evidence.append(snippet)
    if any(marker in lower for marker in ("percentile", "median", "average", "statistics")):
        failure_classes.add("statistics_formula_error")
        contract_areas.add("statistics_block")
        repair_patterns.add("use_supported_statistics_formulas")
        snippet = evidence_snippet(content, "percentile") or evidence_snippet(content, "statistics")
        if snippet:
            evidence.append(snippet)
    if "assertionerror" in lower or "expected" in lower:
        failure_classes.add("verifier_assertion_mismatch")
        repair_patterns.add("match_verifier_contract")
    if "timeout" in lower:
        failure_classes.add("timeout")
        contract_areas.add("runtime")
        repair_patterns.add("reduce_runtime")

    reward = verifier_reward if isinstance(verifier_reward, (int, float)) else 0.0
    if reward >= 1.0:
        outcome_type = "success"
        contract_areas.add("validated_solution_pattern")
        repair_patterns.add("reuse_successful_structure")
    else:
        outcome_type = "failure"
        if not failure_classes:
            failure_classes.add("unspecified_failure")
            repair_patterns.add("inspect_verifier_feedback")

    confidence = min(0.95, 0.35 + 0.1 * len(evidence) + 0.1 * len(contract_areas))
    signal = {
        "task": task,
        "domain": domain,
        "outcome_type": outcome_type,
        "failure_classes": sorted(failure_classes),
        "contract_areas": sorted(contract_areas),
        "repair_patterns": sorted(repair_patterns),
        "evidence": evidence[:5],
        "confidence": round(confidence, 3),
        "verifier_reward": verifier_reward,
        "judge_reward": judge_reward,
        "artifact_ids": [
            artifact.get("artifact_id")
            for artifact in artifacts
            if artifact.get("artifact_id")
        ],
        "extraction_method": "deterministic_schema_v1_llm_compatible",
    }
    return signal


def set_overlap(left: list[str], right: list[str]) -> float:
    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def signal_target_overlap(signal: dict[str, Any], target_profile: dict[str, Any]) -> dict[str, float]:
    target_risks = list(target_profile.get("known_risks") or [])
    target_contracts = list(target_profile.get("contract_areas") or [])
    failure_overlap = set_overlap(list(signal.get("failure_classes") or []), target_risks)
    contract_overlap = set_overlap(list(signal.get("contract_areas") or []), target_contracts)
    repair_overlap = set_overlap(list(signal.get("repair_patterns") or []), target_risks + target_contracts)
    domain_match = 1.0 if signal.get("domain") == target_profile.get("domain") else 0.0
    return {
        "domain_match": domain_match,
        "failure_overlap": failure_overlap,
        "contract_overlap": contract_overlap,
        "repair_overlap": repair_overlap,
        "confidence": float(signal.get("confidence") or 0.0),
    }


def source_signal(run: dict[str, Any]) -> dict[str, Any]:
    store = Path(run["exported_store"] or "")
    artifacts = load_store_artifacts(store)
    run_outcome = next((a for a in artifacts if str(a.get("artifact_type")).endswith("run_outcome")), None)
    mediator = next((a for a in artifacts if a.get("artifact_type") == "mediator_report_summary"), None)
    debug = next((a for a in artifacts if a.get("artifact_type") == "debug_hint"), None)
    content = "\n".join(str(a.get("content") or "") for a in (run_outcome, mediator, debug) if a)
    signal = extract_artifact_signal(
        task=run["task"],
        verifier_reward=run.get("verifier_reward"),
        judge_reward=run.get("judge_reward"),
        artifacts=artifacts,
        content=content,
    )
    return {
        "task": run["task"],
        "verifier_reward": run.get("verifier_reward"),
        "judge_reward": run.get("judge_reward"),
        "store": run.get("exported_store"),
        "run_outcome": run_outcome,
        "mediator": mediator,
        "debug": debug,
        "content": content,
        "signal": signal,
    }


def edge_score(
    metadata: dict[str, Any],
    completed: set[str],
    source: dict[str, Any],
    target: str,
) -> tuple[float, str, dict[str, Any]]:
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
    signal = source.get("signal") or {}
    overlap = signal_target_overlap(signal, dst.get("target_profile") or {})
    signal_match = max(
        overlap["failure_overlap"],
        overlap["contract_overlap"],
        0.5 * overlap["domain_match"],
    )
    failure_repair_bonus = 0.45 * signal_match if reward_value < 1.0 else 0.0
    repair_bonus = 0.25 * max(overlap["repair_overlap"], overlap["contract_overlap"])
    novelty_bonus = 0.25 if target not in completed else -0.15
    same_difficulty = 0.1 if src["declared_difficulty"] == dst["declared_difficulty"] else 0.0
    confidence_adjustment = 0.10 * overlap["confidence"] if signal_match > 0 else -0.05
    raw_affinity = (
        1.15 * lexical
        + 0.70 * rank_score
        + success_bonus
        + failure_repair_bonus
        + repair_bonus
        + novelty_bonus
        + same_difficulty
        + confidence_adjustment
    )
    score = max(-1.0, min(1.0, (2.0 * raw_affinity / MAX_RAW_AFFINITY) - 1.0))
    components = {
        "raw_affinity": round(raw_affinity, 6),
        "lexical": round(lexical, 6),
        "rank_gap": rank_gap,
        "rank_score": round(rank_score, 6),
        "success_bonus": round(success_bonus, 6),
        "failure_repair_bonus": round(failure_repair_bonus, 6),
        "repair_bonus": round(repair_bonus, 6),
        "novelty_bonus": round(novelty_bonus, 6),
        "same_difficulty": round(same_difficulty, 6),
        "confidence_adjustment": round(confidence_adjustment, 6),
        "signal_overlap": overlap,
        "source_signal": signal,
    }
    rationale = (
        f"signed_affinity={score:.3f}; raw_affinity={raw_affinity:.3f}; lexical={lexical:.2f}; "
        f"rank_gap={rank_gap}; success_bonus={success_bonus:.2f}; signal_match={signal_match:.2f}; "
        f"failure_repair_bonus={failure_repair_bonus:.2f}; repair_bonus={repair_bonus:.2f}; "
        f"novelty_bonus={novelty_bonus:.2f}; confidence={overlap['confidence']:.2f}"
    )
    return score, rationale, components


def truncate_text(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 20)] + " ...[truncated]"


def compact_router_packet(
    metadata: dict[str, Any],
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    signal = source.get("signal") or {}
    return {
        "source": {
            "task": source["task"],
            "verifier_reward": source.get("verifier_reward"),
            "judge_reward": source.get("judge_reward"),
            "signal": signal,
            "recent_log_excerpt": truncate_text(source.get("content") or "", 1400),
        },
        "candidates": [
            {
                "target": item["target"],
                "deterministic_score": item["similarity_index"],
                "deterministic_rationale": truncate_text(item["rationale"], 400),
                "declared_difficulty": metadata[item["target"]].get("declared_difficulty"),
                "tags": metadata[item["target"]].get("tags") or [],
                "target_guidance": metadata[item["target"]].get("target_guidance") or {},
                "target_profile": metadata[item["target"]].get("target_profile") or {},
                "instruction_digest": truncate_text(metadata[item["target"]].get("instruction_digest") or "", 700),
            }
            for item in candidates
        ],
        "scoring_instruction": (
            "Score whether the source artifact/log signal should be diffused to each target. "
            "Prefer concrete document-editing/contract reuse, known failure prevention, and target-specific fit. "
            "Penalize noun-copy risk, circular reuse, unrelated domains, and likely negative transfer."
        ),
    }


def parse_router_scores(content: str) -> dict[str, dict[str, Any]]:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    rows = payload.get("scores") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    parsed: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        target = row.get("target")
        score = row.get("score")
        confidence = row.get("confidence", 0.0)
        if not isinstance(target, str) or not isinstance(score, (int, float)):
            continue
        parsed[target] = {
            "score": max(0.0, min(1.0, float(score))),
            "confidence": max(0.0, min(1.0, float(confidence) if isinstance(confidence, (int, float)) else 0.0)),
            "rationale": truncate_text(str(row.get("rationale") or ""), 420),
        }
    return parsed


async def complete_router_call(packet: dict[str, Any]) -> dict[str, Any]:
    client = LLMClient(model=ROUTER_MODEL, max_retries=2, timeout=120)
    result = await client.complete(
        [
            {
                "role": "system",
                "content": (
                    "You are a strict transfer-routing judge. Return only JSON with shape "
                    '{"scores":[{"target":"...","score":0.0,"confidence":0.0,"rationale":"..."}]}. '
                    "Scores and confidence must be in [0,1]."
                ),
            },
            {"role": "user", "content": json.dumps(packet, sort_keys=True)},
        ],
        max_tokens=900,
        temperature=0.0,
    )
    return dict(result)


def llm_router_scores(
    metadata: dict[str, Any],
    logical_iter: int,
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    packet = compact_router_packet(metadata, source, candidates)
    try:
        result = asyncio.run(complete_router_call(packet))
    except Exception as exc:  # pragma: no cover - tmp experiment audit path
        append_jsonl(
            ANALYSIS_DIR / "llm_router_decisions.jsonl",
            {
                "logical_iter": logical_iter,
                "source": source["task"],
                "model": ROUTER_MODEL,
                "status": "error",
                "error": truncate_text(str(exc), 500),
                "packet": packet,
                "timestamp": now(),
            },
        )
        return {}
    parsed = parse_router_scores(str(result.get("content") or ""))
    append_jsonl(
        ANALYSIS_DIR / "llm_router_decisions.jsonl",
        {
            "logical_iter": logical_iter,
            "source": source["task"],
            "model": ROUTER_MODEL,
            "status": "ok" if parsed else "parse_empty",
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "content": result.get("content"),
            "parsed_scores": parsed,
            "packet": packet,
            "timestamp": now(),
        },
    )
    return parsed


def apply_llm_router(
    metadata: dict[str, Any],
    logical_iter: int,
    source: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    scores = llm_router_scores(metadata, logical_iter, source, candidates)
    routed: list[dict[str, Any]] = []
    for item in candidates:
        router_score = scores.get(item["target"])
        if not router_score:
            routed.append(item)
            continue
        confidence = float(router_score["confidence"])
        llm_weight = ROUTER_LLM_WEIGHT * confidence
        deterministic = float(item["similarity_index"])
        llm_signed = (2.0 * float(router_score["score"])) - 1.0
        combined = ((1.0 - llm_weight) * deterministic) + (llm_weight * llm_signed)
        components = dict(item.get("score_components") or {})
        components.update(
            {
                "deterministic_similarity_index": round(deterministic, 6),
                "combined_similarity_index": round(combined, 6),
                "llm_router_score": round(float(router_score["score"]), 6),
                "llm_router_confidence": round(confidence, 6),
                "llm_router_weight": round(llm_weight, 6),
                "llm_reason": router_score.get("rationale") or "",
                "router_model": ROUTER_MODEL,
            }
        )
        routed.append(
            {
                **item,
                "similarity_index": max(-1.0, min(1.0, combined)),
                "rationale": item["rationale"] + f"; llm_router={router_score['score']:.3f}; "
                f"llm_confidence={confidence:.2f}; llm_weight={llm_weight:.2f}; "
                f"llm_reason={router_score.get('rationale') or ''}",
                "score_components": components,
            }
        )
    return routed


def softmax(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_score = max(item["similarity_index"] for item in items)
    weights = [math.exp((item["similarity_index"] - max_score) / SOFTMAX_TEMPERATURE) for item in items]
    total = sum(weights)
    return [
        {**item, "probability": weight / total, "softmax_temperature": SOFTMAX_TEMPERATURE}
        for item, weight in zip(items, weights, strict=True)
    ]



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
            score, rationale, components = edge_score(metadata, completed, source, target)
            candidates.append(
                {
                    "source": source["task"],
                    "target": target,
                    "similarity_index": score,
                    "rationale": rationale,
                    "score_components": components,
                    "source_store": source["store"],
                    "source_verifier_reward": source["verifier_reward"],
                    "source_judge_reward": source["judge_reward"],
                }
            )
        candidates.sort(key=lambda item: (-item["similarity_index"], item["target"]))
        top = candidates[:3]
        top = apply_llm_router(metadata, logical_iter, source, top)
        top.sort(key=lambda item: (-item["similarity_index"], item["target"]))
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
            "softmax_temperature": SOFTMAX_TEMPERATURE,
            "router_model": ROUTER_MODEL,
            "router_llm_weight": ROUTER_LLM_WEIGHT,
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
    active_next = apply_activation_safeguards(
        state=state,
        logical_iter=logical_iter,
        source_runs=source_runs,
        active_next=active_next,
    )
    write_json(ANALYSIS_DIR / f"checkpoint_L{logical_iter:02d}.json", active_next)
    return active_next


def task_ran_in_iteration(state: dict[str, Any], task: str, logical_iter: int) -> bool:
    return any(
        run.get("task") == task and int(run.get("logical_iter", -999)) == logical_iter
        for run in state.get("runs", [])
    )


def would_make_three_consecutive(state: dict[str, Any], task: str, next_iter: int) -> bool:
    return all(
        task_ran_in_iteration(state, task, iter_index)
        for iter_index in range(next_iter - CONSECUTIVE_ITERATION_LIMIT, next_iter)
    )


def best_source_for_target(source_runs: list[dict[str, Any]], target: str) -> dict[str, Any] | None:
    candidates = [
        run
        for run in source_runs
        if run.get("task") != target and run.get("exported_store")
    ]
    if not candidates:
        candidates = [run for run in source_runs if run.get("exported_store")]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda run: (
            run.get("verifier_reward") == 1.0,
            float(run.get("judge_reward") or 0.0),
            -int(run.get("cost", {}).get("total_tokens") or 0),
            str(run.get("task")),
        ),
    )


def task_run_count(state: dict[str, Any], task: str) -> int:
    return sum(1 for run in state.get("runs", []) if run.get("task") == task)


def task_last_seen(state: dict[str, Any], task: str) -> int:
    return max(
        [
            int(run.get("logical_iter", -1))
            for run in state.get("runs", [])
            if run.get("task") == task
        ],
        default=-999,
    )


def active_task_order(state: dict[str, Any], active: dict[str, list[dict[str, Any]]]) -> list[str]:
    return sorted(
        active,
        key=lambda task: (
            task_run_count(state, task),
            task_last_seen(state, task),
            task,
        ),
    )


def candidate_wildcard_targets(
    state: dict[str, Any],
    *,
    active_targets: set[str],
    next_iter: int,
) -> list[str]:
    return sorted(
        [
            task
            for task in TASKS
            if task not in active_targets
            and not would_make_three_consecutive(state, task, next_iter)
        ],
        key=lambda task: (
            task_run_count(state, task),
            task_last_seen(state, task),
            task,
        ),
    )


def add_source_backed_rescue(
    *,
    state: dict[str, Any],
    logical_iter: int,
    source_runs: list[dict[str, Any]],
    guarded: dict[str, list[dict[str, Any]]],
    target: str,
    event: str,
    reason: str,
) -> None:
    next_iter = logical_iter + 1
    best_source = best_source_for_target(source_runs, target)
    if best_source is None:
        guarded.setdefault(target, [])
        append_jsonl(
            ANALYSIS_DIR / "activation_safeguards.jsonl",
            {
                "logical_iter": logical_iter,
                "target_logical_iter": next_iter,
                "event": f"{event}_without_transfer",
                "target": target,
                "source": None,
                "selected_similarity_index": None,
                "reason": f"{reason}; no source store is available",
                "timestamp": now(),
            },
        )
        return
    source = source_signal(best_source)
    completed = {run["task"] for run in state.get("runs", []) if run.get("verifier_reward") == 1.0}
    score, rationale, components = edge_score(state["metadata"], completed, source, target)
    guarded.setdefault(target, []).append(
        {
            "source": source["task"],
            "target": target,
            "similarity_index": score,
            "selected_similarity_index": score,
            "selected_probability": 1.0,
            "rationale": f"{reason}; " + rationale,
            "score_components": components,
            "source_store": source["store"],
            "source_signal": source,
        }
    )
    append_jsonl(
        ANALYSIS_DIR / "activation_safeguards.jsonl",
        {
            "logical_iter": logical_iter,
            "target_logical_iter": next_iter,
            "event": event,
            "target": target,
            "source": source["task"],
            "selected_similarity_index": score,
            "reason": reason,
            "timestamp": now(),
        },
    )


def apply_activation_safeguards(
    *,
    state: dict[str, Any],
    logical_iter: int,
    source_runs: list[dict[str, Any]],
    active_next: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    next_iter = logical_iter + 1
    guarded: dict[str, list[dict[str, Any]]] = {}
    for target, incoming in sorted(active_next.items()):
        if would_make_three_consecutive(state, target, next_iter):
            append_jsonl(
                ANALYSIS_DIR / "activation_safeguards.jsonl",
                {
                    "logical_iter": logical_iter,
                    "target_logical_iter": next_iter,
                    "event": "drop_third_consecutive_activation",
                    "target": target,
                    "incoming_sources": [edge.get("source") for edge in incoming],
                    "reason": "task already ran in the two previous logical iterations",
                    "timestamp": now(),
                },
            )
            continue
        guarded[target] = incoming

    while len(guarded) < MIN_ACTIVE_TASKS:
        targets = candidate_wildcard_targets(
            state,
            active_targets=set(guarded),
            next_iter=next_iter,
        )
        if not targets:
            break
        target = targets[0]
        add_source_backed_rescue(
            state=state,
            logical_iter=logical_iter,
            source_runs=source_runs,
            guarded=guarded,
            target=target,
            event="wildcard_min_active_rescue",
            reason=f"next iteration needs at least {MIN_ACTIVE_TASKS} active tasks",
        )

    max_active_tasks = min(3, len(TASKS))
    while len(guarded) < max_active_tasks:
        targets = [
            target
            for target in candidate_wildcard_targets(
                state,
                active_targets=set(guarded),
                next_iter=next_iter,
            )
            if task_run_count(state, target) == 0
        ]
        if not targets:
            break
        add_source_backed_rescue(
            state=state,
            logical_iter=logical_iter,
            source_runs=source_runs,
            guarded=guarded,
            target=targets[0],
            event="wildcard_unseen_task_rescue",
            reason="next iteration keeps a never-run task active to prevent coverage loss",
        )
    return guarded


def artifact_content(source_signal_: dict[str, Any], target: str) -> str:
    guidance = TARGET_GUIDANCE[target]
    source_task = source_signal_["task"]
    signal = source_signal_.get("signal") or {}
    outcome = (
        f"source verifier={source_signal_.get('verifier_reward')} judge={source_signal_.get('judge_reward')}; "
        f"domain={signal.get('domain')} outcome={signal.get('outcome_type')} "
        f"failures={','.join(signal.get('failure_classes') or []) or '-'} "
        f"contracts={','.join(signal.get('contract_areas') or []) or '-'}"
    )
    return "\n".join(
        [
            f"Logical-batch selected transfer from {source_task} to {target}.",
            f"Source signal: {outcome}.",
            "This artifact was selected only from the previous logical iteration checkpoint; do not use same-iteration artifacts.",
            "Use target-specific labels only; do not copy source nouns or values.",
            f"Target document: {guidance['document']}.",
            f"Source data: {guidance['sources']}.",
            f"Required output: {guidance['output']}.",
            "Required target operations: " + "; ".join(guidance["operations"]) + ".",
            "Required HWPX mechanics: preserve the .hwpx ZIP/XML package structure, edit only target document content, and keep all target-specific Korean labels/static text unless the instruction says to update it.",
            "Verifier-sensitive repair: replace every required placeholder/value occurrence, preserve empty paragraphs where required, and remove stale layout-cache elements from any paragraph whose text is modified so rendered text does not overlap.",
            "Mediator output from the source run is part of the source signal and should be used only to avoid repeated failure modes.",
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
                "agent_score_components": edge.get("score_components") or {},
                "source_transfer_signal": source.get("signal") or {},
                "intended_target_task_id": f"{FAMILY}/{target}",
                "source_store": source.get("store"),
                "compacted_by_external_controller": True,
            },
        }
        write_json(artifacts_dir / f"{artifact['artifact_id']}.json", artifact)
        update_routing_memory(artifact)
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


def update_routing_memory(artifact: dict[str, Any]) -> None:
    memory_path = ANALYSIS_DIR / "routing_memory.json"
    memory = load_json(
        memory_path,
        {
            "policy": {
                "last_k": ROUTING_MEMORY_LAST_K,
                "older_summary_kind": "task_edge_signal_summary",
            },
            "edges": {},
        },
    )
    metadata = artifact.get("metadata") or {}
    source_task = artifact.get("source_task_id")
    target_task = metadata.get("intended_target_task_id")
    edge_key = f"{source_task}->{target_task}"
    edge = memory.setdefault("edges", {}).setdefault(
        edge_key,
        {
            "active_artifacts": [],
            "older_summary": {
                "artifact_count": 0,
                "success_patterns": [],
                "failure_modes": [],
                "targets_helped_before": [],
                "targets_harmed_before": [],
                "confidence_values": [],
                "last_updated": None,
            },
        },
    )
    record = {
        "artifact_id": artifact.get("artifact_id"),
        "source_task_id": source_task,
        "target_task_id": target_task,
        "source_iteration": artifact.get("source_iteration"),
        "created_at": artifact.get("created_at"),
        "verifier_reward": artifact.get("verifier_reward"),
        "judge_reward": artifact.get("judge_reward"),
        "similarity_index": metadata.get("agent_selected_similarity_index"),
        "source_transfer_signal": metadata.get("source_transfer_signal") or {},
    }
    edge["active_artifacts"].append(record)
    if len(edge["active_artifacts"]) > ROUTING_MEMORY_LAST_K:
        older = edge["active_artifacts"][:-ROUTING_MEMORY_LAST_K]
        edge["active_artifacts"] = edge["active_artifacts"][-ROUTING_MEMORY_LAST_K:]
        summary = edge["older_summary"]
        summary["artifact_count"] += len(older)
        for item in older:
            signal = item.get("source_transfer_signal") or {}
            if item.get("verifier_reward") == 1.0:
                summary["success_patterns"].extend(signal.get("repair_patterns") or [])
                summary["targets_helped_before"].append(target_task)
            else:
                summary["failure_modes"].extend(signal.get("failure_classes") or [])
                summary["targets_harmed_before"].append(target_task)
            confidence = signal.get("confidence")
            if isinstance(confidence, (int, float)):
                summary["confidence_values"].append(confidence)
        for key in ("success_patterns", "failure_modes", "targets_helped_before", "targets_harmed_before"):
            summary[key] = sorted(set(summary[key]))
        summary["last_updated"] = now()
    memory["updated_at"] = now()
    write_json(memory_path, memory)


def selected_manifest_for_run(run: dict[str, Any]) -> dict[str, Any]:
    artifact_store = run.get("artifact_store")
    if not artifact_store:
        return {}
    manifest_path = Path(str(artifact_store)) / "manifest.json"
    return load_json(manifest_path, {}) if manifest_path.exists() else {}


def selected_artifacts_for_run(run: dict[str, Any]) -> list[dict[str, Any]]:
    artifact_store = run.get("artifact_store")
    if not artifact_store:
        return []
    return load_store_artifacts(Path(str(artifact_store)))


def run_count_board(runs: list[dict[str, Any]]) -> dict[str, Any]:
    board: dict[str, Any] = {}
    for task in TASKS:
        task_runs = [run for run in runs if run.get("task") == task]
        board[task] = {
            "run_count": len(task_runs),
            "logical_iters": [run.get("logical_iter") for run in task_runs],
            "successes": sum(1 for run in task_runs if run.get("verifier_reward") == 1.0),
            "last_verifier_reward": task_runs[-1].get("verifier_reward") if task_runs else None,
            "last_judge_reward": task_runs[-1].get("judge_reward") if task_runs else None,
        }
    return board


def selection_concentration(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {task: 0 for task in TASKS}
    for decision in decisions:
        if decision.get("transfer_rejected"):
            continue
        selected = decision.get("selected")
        if selected in counts:
            counts[selected] += 1
    total = sum(counts.values())
    max_share = max((count / total for count in counts.values()), default=0.0) if total else 0.0
    repeated = {task: count for task, count in counts.items() if count > 1}
    return {
        "selected_target_counts": counts,
        "repeated_selected_targets": repeated,
        "max_selected_target_share": round(max_share, 6),
        "over_exploitation_flag": max_share > 0.4 or any(count >= 3 for count in counts.values()),
    }


def router_usage_summary() -> dict[str, Any]:
    decisions = load_jsonl(ANALYSIS_DIR / "llm_router_decisions.jsonl")
    return {
        "calls": len(decisions),
        "ok_calls": sum(1 for row in decisions if row.get("status") == "ok"),
        "fallbacks": sum(1 for row in decisions if row.get("status") != "ok"),
        "prompt_tokens": sum(int(row.get("input_tokens") or 0) for row in decisions),
        "completion_tokens": sum(int(row.get("output_tokens") or 0) for row in decisions),
        "model": ROUTER_MODEL,
    }


def activation_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in runs:
        artifacts = selected_artifacts_for_run(run)
        manifest = selected_manifest_for_run(run)
        similarities = [
            artifact.get("metadata", {}).get("agent_selected_similarity_index")
            for artifact in artifacts
            if artifact.get("metadata", {}).get("agent_selected_similarity_index") is not None
        ]
        probabilities = [
            artifact.get("metadata", {}).get("agent_selected_probability")
            for artifact in artifacts
            if artifact.get("metadata", {}).get("agent_selected_probability") is not None
        ]
        source_count = len(manifest.get("sources") or [])
        verifier = run.get("verifier_reward")
        judge = run.get("judge_reward")
        min_similarity = min([float(value) for value in similarities], default=None)
        weak_similarity = min_similarity is not None and min_similarity < 0.0
        failed = isinstance(verifier, (int, float)) and verifier < 1.0
        rows.append(
            {
                "logical_iter": run.get("logical_iter"),
                "target": run.get("task"),
                "sources": manifest.get("sources") or [],
                "source_count": source_count,
                "source_artifact_count": len(artifacts),
                "similarity_scores": similarities,
                "softmax_probabilities": probabilities,
                "verifier_reward": verifier,
                "judge_reward": judge,
                "cost_usd": run.get("cost", {}).get("proxy_cost_usd"),
                "tokens": run.get("cost", {}).get("total_tokens"),
                "quorum_candidate": source_count > 1,
                "weak_similarity_activation": weak_similarity,
                "negative_transfer_flag": bool(failed and (weak_similarity or source_count > 1)),
                "dropped_artifact_ids": (run.get("metrics") or {}).get("dropped_for_budget_artifact_ids") or [],
            }
        )
    return rows


def checkpoint_efficiency(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    total_runs = 0
    total_cost = 0.0
    total_judge = 0.0
    total_successes = 0
    for logical_iter in sorted({int(run.get("logical_iter", -1)) for run in runs}):
        iter_runs = [run for run in runs if int(run.get("logical_iter", -1)) == logical_iter]
        total_runs += len(iter_runs)
        total_cost += sum(float(run.get("cost", {}).get("proxy_cost_usd") or 0.0) for run in iter_runs)
        total_judge += sum(float(run.get("judge_reward") or 0.0) for run in iter_runs)
        total_successes += sum(1 for run in iter_runs if run.get("verifier_reward") == 1.0)
        rows.append(
            {
                "logical_iter": logical_iter,
                "cumulative_runs": total_runs,
                "iteration_runs": len(iter_runs),
                "cumulative_cost_usd": round(total_cost, 6),
                "cumulative_successes": total_successes,
                "successes_per_dollar": round(total_successes / total_cost, 6) if total_cost else None,
                "judge_points_per_dollar": round(total_judge / total_cost, 6) if total_cost else None,
                "mean_judge_reward": round(total_judge / total_runs, 6) if total_runs else None,
            }
        )
    return rows


def seed_exploration_cost(state: dict[str, Any], runs: list[dict[str, Any]]) -> dict[str, Any]:
    seed_runs = [run for run in runs if int(run.get("logical_iter", -1)) == 0]
    total_cost = sum(float(run.get("cost", {}).get("proxy_cost_usd") or 0.0) for run in seed_runs)
    total_tokens = sum(int(run.get("cost", {}).get("total_tokens") or 0) for run in seed_runs)
    return {
        "seed_tasks": state.get("seed_tasks") or [],
        "seed_rows": len(seed_runs),
        "seed_tokens": total_tokens,
        "seed_cost_usd": round(total_cost, 6),
        "seed_successes": sum(1 for run in seed_runs if run.get("verifier_reward") == 1.0),
        "seed_judge_mean": round(
            sum(float(run.get("judge_reward") or 0.0) for run in seed_runs) / len(seed_runs),
            6,
        )
        if seed_runs
        else None,
    }


def build_reporting_board(state: dict[str, Any]) -> dict[str, Any]:
    runs = list(state.get("runs") or [])
    decisions = load_jsonl(ANALYSIS_DIR / "softmax_decisions.jsonl")
    activations = activation_rows(runs)
    skipped = [
        task
        for task, row in run_count_board(runs).items()
        if row["run_count"] == 0
    ]
    board = {
        "source": "job_discussion_fixes",
        "generated_at": now(),
        "seed_exploration_cost": seed_exploration_cost(state, runs),
        "run_count_board": run_count_board(runs),
        "skipped_tasks": skipped,
        "selection_concentration": selection_concentration(decisions),
        "router_usage": router_usage_summary(),
        "activation_rows": activations,
        "quorum_targets": [row for row in activations if row["quorum_candidate"]],
        "negative_transfer_flags": [row for row in activations if row["negative_transfer_flag"]],
        "checkpoint_efficiency": checkpoint_efficiency(runs),
        "safeguard_events": load_jsonl(ANALYSIS_DIR / "activation_safeguards.jsonl"),
        "notes": [
            "Many-source targets are quorum candidates only; target outcomes are required before claiming improvement.",
            "Signed affinity uses [-1, 1]; weak transfers are tracked after target outcomes instead of rejected before routing.",
            "Pre/post compaction token lifecycle is intentionally not reported here because it was listed as partially fixable and excluded from this implementation.",
        ],
    }
    write_json(ANALYSIS_DIR / "jobfix_reporting_board.json", board)
    return board


def summarize(state: dict[str, Any]) -> None:
    runs = state["runs"]
    total_cost = sum(float(run.get("cost", {}).get("proxy_cost_usd") or 0.0) for run in runs)
    total_tokens = sum(int(run.get("cost", {}).get("total_tokens") or 0) for run in runs)
    success_count = sum(1 for run in runs if run.get("verifier_reward") == 1.0)
    board = build_reporting_board(state)
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
            f"Each source selects one target from up to three candidates using softmax temperature `{SOFTMAX_TEMPERATURE}`. Candidate distributions are recorded in `analysis/softmax_decisions.jsonl`.",
            f"Compact LLM router packets use `{ROUTER_MODEL}` with weight cap `{ROUTER_LLM_WEIGHT}` and are recorded in `analysis/llm_router_decisions.jsonl`.",
            "",
            "## Implemented Job Fixes",
            "",
            f"- Seed exploration cost: {board['seed_exploration_cost']['seed_rows']} rows, {board['seed_exploration_cost']['seed_tokens']} tokens, ${board['seed_exploration_cost']['seed_cost_usd']:.4f}, successes {board['seed_exploration_cost']['seed_successes']}/{board['seed_exploration_cost']['seed_rows']}.",
            f"- LLM router usage: {board['router_usage']['ok_calls']}/{board['router_usage']['calls']} parsed calls, {board['router_usage']['fallbacks']} fallbacks, {board['router_usage']['prompt_tokens']} prompt tokens, {board['router_usage']['completion_tokens']} completion tokens.",
            f"- Selection concentration max share: {board['selection_concentration']['max_selected_target_share']:.3f}; over-exploitation flag: {board['selection_concentration']['over_exploitation_flag']}.",
            f"- Quorum candidate activations: {len(board['quorum_targets'])}.",
            f"- Negative-transfer flags: {len(board['negative_transfer_flags'])}.",
            f"- Safeguard events: {len(board['safeguard_events'])}.",
            f"- Skipped tasks: {', '.join(board['skipped_tasks']) if board['skipped_tasks'] else 'none'}.",
            "",
            "### Run Count Board",
            "",
            "| Task | Runs | Successes | Logical iters | Last verifier | Last judge |",
            "| --- | ---: | ---: | --- | ---: | ---: |",
        ]
    )
    for task, row in board["run_count_board"].items():
        lines.append(
            f"| `{task}` | {row['run_count']} | {row['successes']} | {row['logical_iters']} | {row['last_verifier_reward']} | {row['last_judge_reward']} |"
        )
    lines.extend(
        [
            "",
            "### Checkpoint Efficiency",
            "",
            "| Logical iter | Cumulative rows | Iter rows | Cost | Successes | Successes / $ | Judge points / $ | Mean judge |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in board["checkpoint_efficiency"]:
        lines.append(
            "| {logical_iter} | {cumulative_runs} | {iteration_runs} | ${cumulative_cost_usd:.4f} | {cumulative_successes} | {successes_per_dollar} | {judge_points_per_dollar} | {mean_judge_reward} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "### Activation Audit",
            "",
            "| Iter | Target | Sources | Similarities | Probabilities | Quorum | Weak sim | Negative-transfer flag | Dropped |",
            "| ---: | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in board["activation_rows"]:
        lines.append(
            "| {logical_iter} | `{target}` | {sources} | {similarity_scores} | {softmax_probabilities} | {quorum_candidate} | {weak_similarity_activation} | {negative_transfer_flag} | {dropped_artifact_ids} |".format(
                **row
            )
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Logical iteration k+1 used only selected stores built from logical iteration k artifacts.",
            "- Physical runs were sequential, but same-logical-iteration tasks did not consume one another's artifacts.",
            "- Mediator was enabled through the `learned_mediator` condition and emitted mediator summaries in exported stores.",
            "- Existing infra compaction was available through `_fit_prior_context_bundle`; budget and compaction telemetry are reported from `metrics.jsonl`.",
            "- The final selector is LLM-assisted: source artifacts are compacted into routing packets, GPT-5.2 scores the deterministic top 3, Python blends those scores with signed affinity, then softmax activation and source-backed wildcard rescues run.",
            "- Partially fixable pre/post compaction-token lifecycle fields are intentionally out of scope for this run.",
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
        if len(state["runs"]) >= min_rows:
            break
        ordered_tasks = active_task_order(state, active)
        iter_record = {"logical_iter": logical_iter, "active_tasks": ordered_tasks, "started_at": now(), "runs": []}
        source_runs: list[dict[str, Any]] = []
        for task in ordered_tasks:
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
        if len(state["runs"]) >= min_rows:
            break
        active = checkpoint(state, logical_iter, source_runs, rng)
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
        if len(state["runs"]) >= min_rows:
            break
        if not active:
            break
        ordered_tasks = active_task_order(state, active)
        iter_record = {"logical_iter": logical_iter, "active_tasks": ordered_tasks, "started_at": now(), "runs": []}
        source_runs: list[dict[str, Any]] = []
        for task in ordered_tasks:
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
        if len(state["runs"]) >= min_rows:
            break
        active = checkpoint(state, logical_iter, source_runs, rng)
        logical_iter += 1
    state["finished_at"] = now()
    write_json(STATE_PATH, state)
    summarize(state)


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
    parser.add_argument("--max-logical-iters", type=int, default=8)
    parser.add_argument("--seed-count", type=int, default=3)
    parser.add_argument("--min-rows", type=int, default=12)
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
