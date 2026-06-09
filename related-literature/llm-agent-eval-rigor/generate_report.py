#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthesize results/*.json into a single markdown report.

Layout: TOC grouped by the 12 categories (chips: tier · year · type · uncertain#),
then per-item field coverage ordered by the layered fields.yaml schema (core groups
+ the item category's conditional groups), then a synthesized mediator-coevo
evaluation-rigor playbook distilled from every actionable_checklist field.

Skips values that are empty, contain [不确定], or whose field is in the item's
`uncertain` array.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
OUTLINE = BASE / "outline.yaml"
FIELDS = BASE / "fields.yaml"
REPORT = BASE / "report.md"

# Human-readable labels for the 12 categories (order = report section order).
CATEGORY_LABELS = [
    ("benchmarks_coding_swe", "A. Agent Benchmarks — Coding / SWE / ML-Engineering"),
    ("benchmarks_web_computeruse", "B. Agent Benchmarks — Web & Computer-Use"),
    ("benchmarks_tooluse_general", "C. Agent Benchmarks — Tool-Use, Conversational & Generalist"),
    ("benchmarks_safety_adversarial", "D. Agent Benchmarks — Safety & Adversarial Robustness"),
    ("contamination_detection", "E. Contamination Detection Methods"),
    ("contamination_controls", "F. Contamination Controls — Prevent / Mitigate / Govern"),
    ("leakage_reward_hacking", "G. Documented Leakage & Reward-Hacking Cases"),
    ("reproducibility_nondeterminism", "H. Reproducibility & Nondeterminism"),
    ("statistics_metrics", "I. Statistical Evaluation Methodology & Metrics"),
    ("ablation_methodology", "J. Ablation Methodology & Component Findings"),
    ("frameworks_tooling", "K. Evaluation Frameworks / Harnesses / Tooling"),
    ("position_meta_surveys", "L. Position / Meta Papers & Surveys"),
]

# Readable headers for field-group keys (core_<group> and conditional group names).
GROUP_HEADERS = {
    "core_basic": "Basic",
    "core_core": "Core",
    "core_rigor": "Rigor",
    "core_critical": "Critical",
    "core_meta": "Meta",
    "core_actionable": "Actionable",
    "benchmark_properties": "Benchmark Properties",
    "detection_method_properties": "Detection-Method Properties",
    "control_properties": "Control Properties",
    "leakage_case_properties": "Leakage-Case Properties",
    "reproducibility_properties_ext": "Reproducibility Properties",
    "statistics_properties": "Statistics Properties",
    "ablation_properties": "Ablation Properties",
    "framework_properties": "Framework Properties",
    "position_paper_properties": "Position-Paper Properties",
}

SKIP_KEYS = {"uncertain", "_source_file"}


def is_uncertain(value, field_name: str, uncertain: set) -> bool:
    if field_name in uncertain:
        return True
    if value is None:
        return True
    s = str(value).strip()
    if not s:
        return True
    if "[不确定]" in s:
        return True
    return False


def fmt_value(value) -> str:
    """Markdown-format a field value (str / list / dict / list-of-dicts)."""
    if isinstance(value, str):
        s = value.strip()
        if len(s) > 160:
            return "<br>" + s
        return s
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        if not value:
            return ""
        if all(isinstance(x, dict) for x in value):
            lines = []
            for d in value:
                kv = " | ".join(f"{k}: {v}" for k, v in d.items())
                lines.append(f"  - {kv}")
            return "<br>" + "<br>".join(lines)
        parts = [str(x).strip() for x in value if str(x).strip()]
        joined = "；".join(parts)
        if len(joined) > 160:
            return "<br>" + "<br>".join(f"  - {p}" for p in parts)
        return joined
    if isinstance(value, dict):
        parts = [f"{k}: {v}" for k, v in value.items()]
        return "<br>" + "<br>".join(f"  - {p}" for p in parts)
    return str(value)


def short_type(value, uncertain: set) -> str:
    if is_uncertain(value, "type", uncertain):
        return "—"
    s = str(value).strip()
    # cut at first delimiter to keep a short chip
    for delim in ("（", "(", "—", "·", "：", ":", ",", "，"):
        idx = s.find(delim)
        if idx > 0:
            s = s[:idx]
            break
    return s.strip()[:24]


def load_schema():
    fields = yaml.safe_load(FIELDS.read_text(encoding="utf-8"))
    core_groups = fields["core_fields"]  # dict group -> [ {name, description, ...} ]
    conditional = fields["conditional_fields"]  # [ {group, applies_to, fields:[...] } ]
    # ordered core group keys as core_<group>
    core_order = [(f"core_{g}", [f["name"] for f in lst]) for g, lst in core_groups.items()]
    cond_by_cat = defaultdict(list)  # category -> [ (group_key, [field names]) ]
    for grp in conditional:
        gk = grp["group"]
        names = [f["name"] for f in grp["fields"]]
        for cat in grp["applies_to"]:
            cond_by_cat[cat].append((gk, names))
    return core_order, cond_by_cat


def anchor(item_id: str) -> str:
    return "item-" + re.sub(r"[^a-z0-9]+", "-", item_id.lower()).strip("-")


def main() -> None:
    outline = yaml.safe_load(OUTLINE.read_text(encoding="utf-8"))
    topic = outline["topic"]
    items = outline["items"]
    by_id = {it["id"]: it for it in items}
    core_order, cond_by_cat = load_schema()

    # load all results
    data = {}
    for it in items:
        p = RESULTS / f"{it['id']}.json"
        if p.exists():
            try:
                data[it["id"]] = json.load(open(p, encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                data[it["id"]] = {"_error": str(e)}

    # group item ids by category (preserve outline order)
    cat_items = defaultdict(list)
    for it in items:
        cat_items[it["category"]].append(it["id"])

    out = []
    a = out.append

    # ---- Header ----
    a(f"# {topic} — Research Report\n")
    a("> Rigorous evaluation methodology for LLM agents: reproducibility, "
      "contamination/leakage controls, statistics, and ablations.\n")
    a(f"**{len(data)} items** across **{len(CATEGORY_LABELS)} categories**. "
      "Values flagged `[不确定]` or listed in an item's `uncertain` array are skipped. "
      "All field values are in Chinese (research-phase convention).\n")
    a("\n---\n")

    # ---- Table of contents ----
    a("## Table of Contents\n")
    n = 0
    for cat, label in CATEGORY_LABELS:
        ids = cat_items.get(cat, [])
        if not ids:
            continue
        a(f"\n### {label}  ({len(ids)})\n")
        for iid in ids:
            n += 1
            it = by_id[iid]
            d = data.get(iid, {})
            unc = set(d.get("uncertain", []))
            year = d.get("year")
            year_chip = "" if is_uncertain(year, "year", unc) else f" · {str(year)[:12]}"
            tchip = short_type(d.get("type"), unc)
            ucount = len(unc)
            chips = f"`{it['tier']}`{year_chip} · _{tchip}_ · {ucount} uncertain"
            a(f"{n}. [{it['name']}](#{anchor(iid)}) — {chips}")
    a("\n---\n")

    # ---- Body, grouped by category ----
    for cat, label in CATEGORY_LABELS:
        ids = cat_items.get(cat, [])
        if not ids:
            continue
        a(f"\n## {label}\n")
        # ordered field groups for this category
        group_seq = list(core_order) + cond_by_cat.get(cat, [])
        for iid in ids:
            it = by_id[iid]
            d = data.get(iid, {})
            unc = set(d.get("uncertain", []))
            a(f'\n<a id="{anchor(iid)}"></a>')
            a(f"### {it['name']}\n")
            a(f"`{cat}` · tier `{it['tier']}`"
              + ("" if not it.get("orig") else " · _original-26_") + "\n")
            shown = set()
            for gk, names in group_seq:
                rows = []
                for fname in names:
                    if fname not in d:
                        continue
                    shown.add(fname)
                    val = d[fname]
                    if is_uncertain(val, fname, unc):
                        continue
                    rows.append(f"- **{fname}**: {fmt_value(val)}")
                if rows:
                    a(f"**{GROUP_HEADERS.get(gk, gk)}**\n")
                    a("\n".join(rows) + "\n")
            # extra fields present in JSON but not in schema
            extra = []
            for k, v in d.items():
                if k in SKIP_KEYS or k in shown or k.startswith("_"):
                    continue
                if is_uncertain(v, k, unc):
                    continue
                extra.append(f"- **{k}**: {fmt_value(v)}")
            if extra:
                a("**其他信息 / Other**\n")
                a("\n".join(extra) + "\n")
            # uncertain list (one per line)
            if unc:
                a("**Uncertain (skipped) fields**\n")
                for f in sorted(unc):
                    a(f"- {f}")
                a("")

    # ---- Playbook ----
    a("\n---\n")
    a("## Appendix — mediator-coevo Evaluation-Rigor Playbook\n")
    a("Distilled from the `actionable_checklist` field of all 114 items. "
      "Cross-cutting principles first, then per-category specifics.\n")
    a("\n### Cross-cutting principles (the recurring lessons)\n")
    for line in CROSS_CUTTING:
        a(f"- {line}")
    a("\n### Per-item actionable checklists\n")
    for cat, label in CATEGORY_LABELS:
        ids = cat_items.get(cat, [])
        if not ids:
            continue
        rows = []
        for iid in ids:
            d = data.get(iid, {})
            unc = set(d.get("uncertain", []))
            val = d.get("actionable_checklist")
            if is_uncertain(val, "actionable_checklist", unc):
                continue
            rows.append(f"- **{by_id[iid]['name']}**: {fmt_value(val)}")
        if rows:
            a(f"\n**{label}**\n")
            a("\n".join(rows) + "\n")

    REPORT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {REPORT} ({REPORT.stat().st_size} bytes, {len(data)} items)")


# Hand-curated cross-cutting principles synthesized from the corpus.
CROSS_CUTTING = [
    "**Isolate the agent from the grader.** The #1 BenchJack \"deadly pattern\": never let the "
    "system-under-test reach gold answers/tests/config (file://, shared container, git history, "
    "task-config URLs). Run scoring in a separate, network-restricted process.",
    "**Scrub the environment of future knowledge.** Strip git history/tags/reflog past the task "
    "commit; don't ship gold files or answer URLs in the sandbox (SWE-bench #465 / SWE-bench Pro / Multilingual).",
    "**Report pass@1 AND pass^k, never single-run pass@1.** Agent pass@1 varies 2–6pp run-to-run even at "
    "temperature 0; ~9 runs needed to detect a 2pp gap at 80% power (SWE-bench 60k study). Add error bars.",
    "**temperature=0 is NOT deterministic.** Nondeterminism comes from missing batch-invariance "
    "(server-load-driven batch size) and GPU/TP config; use batch-invariant kernels or report aggregate-in-CI only.",
    "**Pin dated model snapshots, not aliases** — and re-baseline on every provider update; even pinned "
    "snapshots drift via infra/safety layers (ChatGPT 84%→51% prime test in 3 months).",
    "**Stamp the harness + publish a content-addressed eval-image digest.** \"Used Docker\" is not "
    "reproducible; agent papers average 0.38/1.0 disclosure and none publish grader image digest or cost.",
    "**Report cost.** Use accuracy×cost Pareto frontiers — complex scaffolds often lose to simple baselines "
    "at equal cost (AI Agents That Matter / HAL).",
    "**Assume contamination on any public benchmark; prefer private held-out / date-gated / dynamic sets.** "
    "Treat MIA-style detectors as near-random and easily evaded — controls beat detection.",
    "**Ablations must hold cost, base model, rollouts and oracle signals constant.** Uncontrolled confounds "
    "invert conclusions (intrinsic self-correction *degrades*; filesystem memory matches specialized stores).",
    "**LLM-judges need chance-corrected agreement (Krippendorff α / Scott π), input sanitization, and a "
    "reproducible pinned judge** — raw accuracy overstates reliability and judges are prompt-injectable.",
    "**Log full trajectories and report failed/errored/skipped runs (Rollout Cards).** Reporting-rule "
    "changes alone can flip frontier rankings by up to 20.9pp.",
    "**Adversarially test your own benchmark** (null/random/injection/state-tampering agents; oracle solver) "
    "before trusting a score — and watch for eval-awareness (capable models reverse-engineer answer keys).",
]

if __name__ == "__main__":
    main()
