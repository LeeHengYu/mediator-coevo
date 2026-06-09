#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compile the 78 per-paper deep-research JSONs into a single markdown survey report.

- Reads outline.yaml (authoritative short metadata: category, venue, year, arxiv, learning_locus)
- Reads fields.yaml (field group structure + human-readable descriptions)
- Reads results/*.json (deep per-paper findings; flat structure keyed by field name)
- Joins JSON <-> outline by the `id` field (e.g. R005)
- Skips values that are [uncertain]/[不确定]/None/empty or listed in each file's `uncertain` array
- TOC grouped by category A-J, each item annotated: Category | Venue | Year | learning_locus | arXiv
- Detail sections grouped by category, each field rendered under its field-group heading
"""

from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
OUTLINE_PATH = BASE / "outline.yaml"
FIELDS_PATH = BASE / "fields.yaml"
RESULTS_DIR = BASE / "results"
REPORT_PATH = BASE / "report.md"

# Field-group key -> JSON keys that may carry it (multilingual / legacy tolerance).
# Our JSONs are FLAT (field name at top level), so these are mainly for nested-structure
# compatibility per the skill spec.
CATEGORY_MAPPING = {
    "basic_info": ["basic_info", "基本信息"],
    "core_mechanism": ["core_mechanism", "技术特性", "technical_features"],
    "learning_paradigm": ["learning_paradigm", "学习范式"],
    "memory_mechanism": ["memory_mechanism", "记忆机制"],
    "reflection_granularity": ["reflection_granularity"],
    "application_domain": ["application_domain", "应用域"],
    "evaluation_results": ["evaluation_results", "性能指标", "performance"],
    "limitations_and_critique": ["limitations_and_critique", "局限批判"],
    "failure_analysis": ["failure_analysis"],
    "relations": ["relations", "关系"],
}

# Human-readable group titles for the report
GROUP_TITLES = {
    "basic_info": "Basic info",
    "core_mechanism": "Core mechanism",
    "learning_paradigm": "Learning paradigm",
    "memory_mechanism": "Memory mechanism",
    "reflection_granularity": "Reflection granularity & timing",
    "application_domain": "Application domain",
    "evaluation_results": "Evaluation results",
    "limitations_and_critique": "Limitations & critique",
    "failure_analysis": "Failure analysis",
    "relations": "Relations",
}

SKIP_KEYS = {"_source_file", "uncertain", "id", "name"}
UNCERTAIN_MARKERS = ("[uncertain]", "[不确定]")


def is_uncertain(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return True
        if any(m in s for m in UNCERTAIN_MARKERS):
            return True
    return False


def load_outline():
    data = yaml.safe_load(OUTLINE_PATH.read_text(encoding="utf-8"))
    cats = data["categories"]  # {"A": "title", ...}
    by_id = {it["id"]: it for it in data["items"]}
    return data, cats, by_id


def load_field_groups():
    """Return ordered list of (group_key, [field_name,...]) and a flat field->desc map."""
    data = yaml.safe_load(FIELDS_PATH.read_text(encoding="utf-8"))
    fc = data.get("field_categories", {})
    groups = []
    desc = {}
    for gkey, body in fc.items():
        if not isinstance(body, dict):
            continue
        names = []
        for field in body.get("fields", []):
            names.append(field["name"])
            desc[field["name"]] = field.get("description", "")
        groups.append((gkey, names))
    reserved = data.get("uncertain", []) or []  # reserved field names
    return groups, desc, reserved


def slugify_anchor(text: str) -> str:
    """GitHub-style anchor: lowercase, drop punctuation, spaces->hyphens."""
    s = text.lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "-", s.strip())
    return s


def fmt_value(value, indent: str = "") -> str:
    """Format a field value as markdown. Handles dicts, lists, list-of-dicts, long text."""
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if is_uncertain(v):
                continue
            parts.append(f"{indent}- **{k}**: {fmt_value(v, indent + '  ')}")
        return "\n" + "\n".join(parts) if parts else ""
    if isinstance(value, list):
        if not value:
            return ""
        # list of dicts -> one line per dict
        if all(isinstance(x, dict) for x in value):
            lines = []
            for d in value:
                kv = " | ".join(
                    f"{k}: {v}" for k, v in d.items() if not is_uncertain(v)
                )
                if kv:
                    lines.append(f"{indent}- {kv}")
            return "\n" + "\n".join(lines) if lines else ""
        # simple list
        items = [str(x) for x in value if not is_uncertain(x)]
        if not items:
            return ""
        joined = ", ".join(items)
        if len(joined) <= 100:
            return joined
        return "\n" + "\n".join(f"{indent}- {x}" for x in items)
    s = str(value)
    # long text -> blockquote for readability
    if len(s) > 220:
        return "\n\n" + "\n".join(f"{indent}> {line}" for line in s.split("\n"))
    return s


def render_field(name: str, value) -> str:
    rendered = fmt_value(value)
    if rendered == "" or rendered is None:
        return ""
    label = name.replace("_", " ")
    return f"- **{label}**: {rendered}\n"


def main() -> None:
    outline, cats, _by_id = load_outline()
    groups, _desc, _reserved = load_field_groups()
    all_field_names = {n for _, names in groups for n in names}

    files = sorted(glob.glob(str(RESULTS_DIR / "*.json")))
    # Map id -> (json_data, filepath)
    data_by_id = {}
    for f in files:
        d = json.load(open(f, encoding="utf-8"))
        rid = d.get("id")
        if rid:
            data_by_id[rid] = (d, f)

    # Order items: by category (A..J) then by id, using outline order
    cat_order = list(cats.keys())
    items = sorted(
        outline["items"],
        key=lambda it: (cat_order.index(it["category"]), it["id"]),
    )

    # Build anchor per item (id + name keeps anchors unique)
    def anchor_for(it):
        return slugify_anchor(f"{it['id']}-{it['name']}")

    topic = outline["topic"]
    lines = []
    lines.append(f"# {topic}")
    lines.append("")
    lines.append(
        f"Deep-research survey of **{len(items)} works** across "
        f"{len(cats)} categories. Generated from per-paper deep-research JSONs "
        "(all validated to 100% field coverage)."
    )
    lines.append("")
    lines.append(
        "_Annotations:_ **Category** · **Venue** · **Year** · "
        "**learning_locus** (in_context_verbal / gradient_weight_update / hybrid) · **arXiv**."
    )
    lines.append("")

    # ---- Legend of categories ----
    lines.append("## Categories")
    lines.append("")
    counts = {}
    for it in items:
        counts[it["category"]] = counts.get(it["category"], 0) + 1
    for c in cat_order:
        lines.append(f"- **{c}** — {cats[c]} ({counts.get(c, 0)})")
    lines.append("")

    # ---- Table of contents grouped by category ----
    lines.append("## Table of contents")
    lines.append("")
    n = 0
    for c in cat_order:
        cat_items = [it for it in items if it["category"] == c]
        if not cat_items:
            continue
        lines.append(f"### {c}. {cats[c]}")
        lines.append("")
        for it in cat_items:
            n += 1
            anchor = anchor_for(it)
            venue = it.get("venue", "") or ""
            # year: prefer the explicit year token from the JSON
            jd = data_by_id.get(it["id"], ({}, None))[0]
            year_val = jd.get("year") if not is_uncertain(jd.get("year")) else None
            arxiv = it.get("arxiv_or_year", "")
            locus = it.get("learning_locus", "")
            annot = " · ".join(
                x for x in [
                    it["category"],
                    venue,
                    str(year_val) if year_val else "",
                    locus,
                    f"arXiv:{arxiv}" if arxiv and arxiv.replace('.', '').isdigit() else "",
                ] if x
            )
            lines.append(f"{n}. [{it['name']}](#{anchor}) — {annot}")
        lines.append("")

    # ---- Detail sections grouped by category ----
    lines.append("---")
    lines.append("")
    lines.append("## Detailed findings")
    lines.append("")

    for c in cat_order:
        cat_items = [it for it in items if it["category"] == c]
        if not cat_items:
            continue
        lines.append(f"# {c}. {cats[c]}")
        lines.append("")
        for it in cat_items:
            rid = it["id"]
            anchor = anchor_for(it)
            lines.append(f'<a id="{anchor}"></a>')
            lines.append(f"## {it['name']}")
            lines.append("")
            # metadata line from outline
            meta_bits = [
                f"**ID:** {rid}",
                f"**Category:** {it['category']} ({cats[it['category']]})",
            ]
            if it.get("authors"):
                meta_bits.append(f"**Authors:** {it['authors']}")
            if it.get("venue"):
                meta_bits.append(f"**Venue:** {it['venue']}")
            if it.get("learning_locus"):
                meta_bits.append(f"**Learning locus:** {it['learning_locus']}")
            if it.get("url"):
                meta_bits.append(f"**URL:** {it['url']}")
            lines.append("  ".join(meta_bits))
            lines.append("")

            entry = data_by_id.get(rid)
            if not entry:
                lines.append("_No deep-research JSON found for this item._")
                lines.append("")
                continue
            d, _ = entry
            uncertain_set = set(d.get("uncertain", []) or [])

            # Render fields grouped by field-group
            for gkey, names in groups:
                block = []
                for name in names:
                    if name in uncertain_set:
                        continue
                    value = d.get(name)
                    if value is None or is_uncertain(value):
                        continue
                    rendered = render_field(name, value)
                    if rendered:
                        block.append(rendered)
                if block:
                    lines.append(f"### {GROUP_TITLES.get(gkey, gkey)}")
                    lines.append("")
                    lines.extend(block)
                    lines.append("")

            # Reserved/extra fields actually filled (e.g. on_policy_distillation_link)
            extra = []
            handled = all_field_names | SKIP_KEYS
            for k, v in d.items():
                if k in handled:
                    continue
                if k in uncertain_set or is_uncertain(v):
                    continue
                rendered = render_field(k, v)
                if rendered:
                    extra.append(rendered)
            if extra:
                lines.append("### Other / reserved fields")
                lines.append("")
                lines.extend(extra)
                lines.append("")

            # Uncertain fields note (each on its own line)
            if uncertain_set:
                lines.append("### Uncertain (not determined)")
                lines.append("")
                for name in sorted(uncertain_set):
                    lines.append(f"- {name}")
                lines.append("")

            lines.append("---")
            lines.append("")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {REPORT_PATH}")
    print(f"  items: {len(items)}, with-JSON: {len(data_by_id)}")
    print(f"  size: {REPORT_PATH.stat().st_size/1024:.1f} KB, lines: {len(lines)}")


if __name__ == "__main__":
    main()
