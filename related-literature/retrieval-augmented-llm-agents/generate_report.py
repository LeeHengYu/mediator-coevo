#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate report.md for the survey corpus.

Reads outline.yaml (grouping + ordering), fields.yaml (category structure),
and results/*.json (per-item researched content). Emits a sub-topic-grouped
table of contents (with compacted summary tags) followed by per-item detail
sections organized by field category. Uncertain / empty values are skipped.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
OUTLINE = ROOT / "outline.yaml"
FIELDS = ROOT / "fields.yaml"
RESULTS_DIR = ROOT / "results"
REPORT = ROOT / "report.md"

# Summary fields shown inline in the TOC (compacted to a short tag).
TOC_FIELDS = ["year", "what_is_retrieved", "retriever_type", "when_to_retrieve"]

# Human-facing labels for each fields.yaml category (in display order).
CATEGORY_LABELS = {
    "basic_info": "Basic Info",
    "problem_definition": "Problem Definition",
    "retrieval_mechanism": "Retrieval Mechanism",
    "utilization_mechanism": "Utilization Mechanism",
    "memory_learning": "Memory & Learning",
    "evaluation": "Evaluation",
    "relations": "Relations",
}

# Human-facing labels for the 7 sub-topic groups (in display order).
GROUP_LABELS = {
    "A_icl_demonstration_retrieval": "A — Retrieval-Augmented In-Context Learning / Demonstration Selection",
    "B_rag_task_solving": "B — Retrieval-Augmented Reasoning & Generation for Task-Solving",
    "C_rag_planning": "C — Retrieval-Augmented Planning (LM Planners)",
    "D_case_based_planning": "D — Case-Based Retrieval for LM Planning (CBR + LLM)",
    "E_tool_use": "E — Retrieval-Augmented Tool-Use Agents",
    "F_decision_memory_rl": "F — Context Retrieval for Decision-Making / Agent Memory / RL",
    "surveys": "Surveys, Taxonomies & Benchmarks",
}

# Internal / structural keys never rendered as content fields.
SKIP_KEYS = {"_source_file", "uncertain", "name"}
# Nested-category top-level keys (in case any JSON used nested structure).
CATEGORY_NESTED_KEYS = set(CATEGORY_LABELS) | {
    "basic_info", "technical_features", "performance_metrics",
}

UNCERTAIN_MARKERS = ("[uncertain]", "[不确定]", "[不確定]")


def load_yaml(path: Path) -> dict:
    import yaml
    with path.open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s or "x"


def is_uncertain_value(v: Any) -> bool:
    if v is None:
        return True
    s = str(v).strip()
    if not s or s.upper() == "N/A":
        return True
    return any(m in s for m in UNCERTAIN_MARKERS)


def compact_tag(v: Any, maxlen: int = 26) -> str:
    """Reduce a verbose value to a short TOC tag."""
    if is_uncertain_value(v):
        return "?"
    s = str(v).strip()
    # cut at the first structural delimiter
    s = re.split(r"[\(:;]", s, maxsplit=1)[0].strip()
    # if it lists alternatives, keep the first
    s = s.split(" / ")[0].split("/")[0].strip() if "/" in s else s
    s = s.rstrip(".").strip()
    if len(s) > maxlen:
        s = s[:maxlen].rstrip() + "…"
    return s or "?"


def fmt_value(v: Any, indent: int = 0) -> str:
    """Format a field value as markdown."""
    pad = "  " * indent
    if isinstance(v, list):
        if not v:
            return ""
        # list of dicts -> one line each, ` | ` between kv pairs
        if all(isinstance(x, dict) for x in v):
            lines = []
            for x in v:
                kv = " | ".join(f"{k}: {val}" for k, val in x.items() if not is_uncertain_value(val))
                if kv:
                    lines.append(f"{pad}- {kv}")
            return "\n".join(lines)
        # simple list
        flat = [str(x).strip() for x in v if not is_uncertain_value(x)]
        if not flat:
            return ""
        if sum(len(x) for x in flat) <= 100 and len(flat) <= 6:
            return ", ".join(flat)
        return "\n".join(f"{pad}- {x}" for x in flat)
    if isinstance(v, dict):
        lines = []
        for k, val in v.items():
            if is_uncertain_value(val):
                continue
            sub = fmt_value(val, indent + 1)
            if not sub:
                continue
            if "\n" in sub:
                lines.append(f"{pad}- **{k}**:\n{sub}")
            else:
                lines.append(f"{pad}- **{k}**: {sub}")
        return "\n".join(lines)
    s = str(v).strip()
    if len(s) > 280:
        # long prose -> blockquote for readability
        return "\n" + "\n".join(f"{pad}> {ln}" for ln in s.split("\n"))
    return s


def get_field(data: dict, name: str) -> Any:
    """Lookup a field: top-level first, then any nested category dict."""
    if name in data:
        return data[name]
    for v in data.values():
        if isinstance(v, dict) and name in v:
            return v[name]
    return None


def collect_present_keys(data: dict) -> set[str]:
    keys = set()
    for k, v in data.items():
        if k in SKIP_KEYS:
            continue
        if isinstance(v, dict) and k in CATEGORY_NESTED_KEYS:
            keys |= {kk for kk in v}
        else:
            keys.add(k)
    return keys


def main() -> None:
    outline = load_yaml(OUTLINE)
    fields = load_yaml(FIELDS)
    topic = outline["topic"]
    description = outline.get("description", "")

    # category -> [field names] (display order from fields.yaml)
    cat_fields: dict[str, list[str]] = {}
    all_defined: set[str] = set()
    for cat, body in fields["field_categories"].items():
        names = [f["name"] for f in body["fields"]]
        cat_fields[cat] = names
        all_defined |= set(names)

    # load items in outline order, grouped
    items = outline["items"]
    by_group: dict[str, list[dict]] = {g: [] for g in GROUP_LABELS}
    loaded: dict[str, dict] = {}
    for it in items:
        p = RESULTS_DIR / f"{it['id']}.json"
        if not p.exists():
            continue
        loaded[it["id"]] = json.loads(p.read_text(encoding="utf-8"))
        by_group.setdefault(it.get("group", "surveys"), []).append(it)

    out: list[str] = []
    out.append(f"# {topic}\n")
    if description:
        out.append(f"_{description}_\n")
    stats = outline.get("stats", {})
    out.append(
        f"**Corpus:** {len(loaded)} researched items across {len(GROUP_LABELS)} sub-topics "
        f"({stats.get('seed_items','?')} seed · {stats.get('supplemental_added','?')} supplemental · "
        f"{stats.get('gap_fill_added','?')} gap-fill). Generated from validated per-item research JSON. "
        f"Fields marked uncertain during research are omitted.\n"
    )
    out.append("_TOC tags: year · retrieved-unit · retriever · when-to-retrieve._\n")

    # ---- Table of Contents (grouped) ----
    out.append("## Table of Contents\n")
    idx = 0
    for g, label in GROUP_LABELS.items():
        grp_items = by_group.get(g, [])
        if not grp_items:
            continue
        out.append(f"\n**{label}** ({len(grp_items)})\n")
        for it in grp_items:
            idx += 1
            d = loaded[it["id"]]
            tags = []
            for tf in TOC_FIELDS:
                tags.append(compact_tag(get_field(d, tf)))
            tagstr = " · ".join(t for t in tags)
            seed = " ⟐" if it.get("is_seed") else ""
            out.append(f"{idx}. [{it['name']}](#item-{it['id']}) — `{tagstr}`{seed}")
    out.append("\n⟐ = seed item.\n")

    # ---- Detail sections (grouped) ----
    n = 0
    for g, label in GROUP_LABELS.items():
        grp_items = by_group.get(g, [])
        if not grp_items:
            continue
        out.append(f"\n---\n\n## {label}\n")
        for it in grp_items:
            n += 1
            d = loaded[it["id"]]
            name = get_field(d, "name") or it["name"]
            out.append(f'<a id="item-{it["id"]}"></a>')
            out.append(f"### {n}. {name}\n")
            meta = []
            if it.get("year"):
                meta.append(str(it["year"]))
            if it.get("venue"):
                meta.append(str(it["venue"]))
            if it.get("paper_url"):
                meta.append(f"[paper]({it['paper_url']})")
            if meta:
                out.append("  ·  ".join(meta) + "\n")

            uncertain = set(d.get("uncertain", []) if isinstance(d.get("uncertain"), list) else [])

            def emit_field(fname: str) -> bool:
                if fname in uncertain:
                    return False
                v = get_field(d, fname)
                if is_uncertain_value(v):
                    return False
                body = fmt_value(v)
                if not body.strip():
                    return False
                label_txt = fname.replace("_", " ")
                if body.startswith("\n") or "\n" in body:
                    out.append(f"- **{label_txt}**:{body if body.startswith(chr(10)) else chr(10)+body}")
                else:
                    out.append(f"- **{label_txt}**: {body}")
                return True

            for cat, names in cat_fields.items():
                emitted_buf_start = len(out)
                header = f"\n**{CATEGORY_LABELS.get(cat, cat)}**\n"
                out.append(header)
                any_emitted = False
                for fname in names:
                    if fname == "name":
                        continue
                    if emit_field(fname):
                        any_emitted = True
                if not any_emitted:
                    del out[emitted_buf_start:]  # drop empty category header

            # extra fields not defined in fields.yaml
            present = collect_present_keys(d)
            extra = sorted(present - all_defined - SKIP_KEYS)
            extra_lines_start = len(out)
            out.append("\n**Other Info**\n")
            any_extra = False
            for fname in extra:
                if emit_field(fname):
                    any_extra = True
            if not any_extra:
                del out[extra_lines_start:]

            # uncertain list (each on its own line)
            if uncertain:
                out.append("\n**Uncertain fields** (not researched / unverifiable):\n")
                for u in sorted(uncertain):
                    out.append(f"- {u}")
            out.append("")

    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT} ({REPORT.stat().st_size} bytes, {len(loaded)} items, {n} sections)")


if __name__ == "__main__":
    main()
