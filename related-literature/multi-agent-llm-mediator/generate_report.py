#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate all per-item research JSONs into a single markdown report.

Reads results/*.json + fields.yaml, overwrites every field, skips uncertain /
[不确定] / empty values, and emits report.md with a TOC (index + name anchor +
cluster·tier · year · venue · paradigm) followed by per-item detail grouped by
the fields.yaml categories. Body is grouped by cluster (A→H) then tier
(core→strong→contextual).
"""
import json
import os
import re
from pathlib import Path

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results"
FIELDS_YAML = BASE / "fields.yaml"
MANIFEST = BASE / "_agent_findings" / "deep_manifest.json"
OUT = BASE / "report.md"

UNCERTAIN_MARK = "[不确定]"
SKIP_KEYS = {"_source_file", "_order", "uncertain"}
# manifest/metadata keys shown in TOC/header — don't repeat them in the body
META_KEYS = {"id", "cluster", "tier"}

# Category multilingual mapping (flat JSON here, but kept for compatibility with
# nested category-keyed JSON per the skill spec).
CATEGORY_MAPPING = {
    "basic": ["basic", "基本信息", "basic_info"],
    "classification": ["classification", "分类"],
    "core_mechanism": ["core_mechanism", "核心机制"],
    "mediation": ["mediation", "中介机制"],
    "aggregation_consensus": ["aggregation_consensus", "聚合共识"],
    "communication_efficiency": ["communication_efficiency", "通信效率"],
    "game_theoretic": ["game_theoretic", "博弈论"],
    "coordination_medium_detail": ["coordination_medium_detail", "协调介质"],
    "protocol_interop": ["protocol_interop", "协议互操作"],
    "coevolution": ["coevolution", "协同进化"],
    "outcome": ["outcome", "结果"],
    "relevance": ["relevance", "相关性"],
    "uncertain": ["uncertain", "不确定"],
}

CLUSTER_TITLES = {
    "A": "A — Multi-Agent LLM Debate & Information Filtering",
    "B": "B — Agent Frameworks / Orchestration",
    "C": "C — Agent Communication Protocols",
    "D": "D — Mediators & Mediated Communication (game-theoretic / MARL / LLM)",
    "E": "E — Blackboard Architecture",
    "F": "F — Stigmergy (environment-mediated coordination)",
    "G": "G — Learned MARL Communication",
    "H": "H — Cutting-Edge 2025–26 + Surveys",
}
TIER_ORDER = {"core": 0, "strong": 1, "contextual": 2}
TIER_LABEL = {"core": "core", "strong": "strong", "contextual": "contextual"}


# ---------------------------------------------------------------- field schema
def load_field_schema(path):
    """Return ordered [(category_name, category_desc, [(field, desc)])] and a
    flat ordered list of all field names, from fields.yaml.

    Minimal hand parse (avoids a hard PyYAML dependency at report time, but uses
    it if available)."""
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        cats = []
        for cat_name, body in (data.get("field_categories") or {}).items():
            if not isinstance(body, dict):
                continue
            fields = [(f["name"], f.get("description", "")) for f in body.get("fields", [])]
            cats.append((cat_name, body.get("description", ""), fields))
        uncertain_fields = [
            (f["name"], f.get("description", ""))
            for f in (data.get("uncertain", {}) or {}).get("fields", [])
        ]
        if uncertain_fields:
            cats.append(("uncertain_reserved", "Reserved / impact-proxy fields", uncertain_fields))
        return cats
    except Exception as e:  # pragma: no cover
        raise SystemExit(f"Failed to parse fields.yaml: {e}")


# ---------------------------------------------------------------- value helpers
def is_skippable(value):
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        if not s or UNCERTAIN_MARK in s:
            return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def format_value(value, indent=0):
    """Format a field value as markdown. Handles str / list / list-of-dict / dict."""
    if isinstance(value, str):
        s = value.strip()
        # long text → blockquote for readability
        if len(s) > 160:
            return "\n" + "\n".join("> " + line for line in _wrap_lines(s))
        return s
    if isinstance(value, list):
        # list of dicts → one line each, ` | ` separated kv
        if value and all(isinstance(x, dict) for x in value):
            lines = []
            for x in value:
                kv = " | ".join(f"{k}: {v}" for k, v in x.items() if not is_skippable(v))
                if kv:
                    lines.append(f"  - {kv}")
            return "\n" + "\n".join(lines)
        flat = [str(x).strip() for x in value if not is_skippable(x)]
        if not flat:
            return ""
        # short list → comma; long → bullets
        if sum(len(x) for x in flat) <= 120 and len(flat) <= 6:
            return "、".join(flat)
        return "\n" + "\n".join(f"  - {x}" for x in flat)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if is_skippable(v):
                continue
            parts.append(f"  - **{k}**: {format_value(v, indent + 1)}")
        return "\n" + "\n".join(parts)
    return str(value)


def _wrap_lines(s):
    # split on existing newlines; keep paragraphs intact
    return [ln.strip() for ln in s.splitlines() if ln.strip()] or [s]


# ---------------------------------------------------------------- TOC summary cleaners
def clean_year(v):
    if not isinstance(v, str):
        return ""
    m = re.search(r"(19|20)\d{2}", v)
    return m.group(0) if m else ""


def clean_venue(v):
    if not isinstance(v, str):
        return ""
    # take text before first bracket / paren / em-dash, trim
    s = re.split(r"[（(\—\-]", v.strip(), maxsplit=1)[0].strip()
    return s[:32]


def clean_paradigm(v):
    if not isinstance(v, str):
        return ""
    # first token before separator
    s = re.split(r"[（(：:，,；;/\s]", v.strip(), maxsplit=1)[0].strip()
    return s[:36]


def uncertain_field_names(d):
    """Normalize the `uncertain` entry to a set of field names.

    Agents emitted three shapes: a flat list of names, a dict {name: reason},
    or (rarely) a list of single-key dicts. Handle all of them so flagged
    fields are reliably skipped and listed."""
    u = d.get("uncertain")
    names = set()
    if isinstance(u, dict):
        names.update(u.keys())
    elif isinstance(u, list):
        for x in u:
            if isinstance(x, str):
                names.add(x.strip())
            elif isinstance(x, dict):
                names.update(x.keys())
    elif isinstance(u, str) and u.strip():
        names.update(p.strip() for p in re.split(r"[、,，\s]+", u) if p.strip())
    # ignore the literal marker if it ever appears as a "name"
    return {n for n in names if n and n != UNCERTAIN_MARK}


def field_lookup(d, field):
    """Flat-first lookup, then nested category dicts (skill-spec compatibility)."""
    if field in d:
        return d[field]
    for v in d.values():
        if isinstance(v, dict) and field in v:
            return v[field]
    return None


# ---------------------------------------------------------------- main
def main():
    cats = load_field_schema(FIELDS_YAML)
    schema_field_names = {f for _, _, fields in cats for f, _ in fields}

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else []
    # authoritative metadata keyed by output filename (many JSONs use a nested
    # category structure and omit top-level id/cluster/tier/name).
    meta_by_file = {os.path.basename(m["output_path"]): m for m in manifest}
    order_by_file = {os.path.basename(m["output_path"]): i for i, m in enumerate(manifest)}

    items = []
    for fp in sorted(RESULTS_DIR.glob("*.json")):
        try:
            d = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"WARN: skip unparseable {fp.name}: {e}")
            continue
        d["_source_file"] = fp.name
        # backfill metadata from the manifest when the JSON omits it (nested items)
        man = meta_by_file.get(fp.name, {})
        for key in ("id", "cluster", "tier"):
            if not d.get(key) and man.get(key):
                d[key] = man[key]
        if not d.get("name"):
            d["name"] = field_lookup(d, "name") or man.get("name") or fp.name
        d["_order"] = order_by_file.get(fp.name, 9999)
        items.append(d)

    # sort: cluster (A..H), tier (core→strong→contextual), then manifest order
    def sort_key(d):
        c = d.get("cluster", "Z")
        t = TIER_ORDER.get(d.get("tier", ""), 9)
        return (c, t, d.get("_order", 9999))

    items.sort(key=sort_key)

    topic = "Multi-Agent LLM Mediator & Mediated Communication"
    lines = []
    lines.append(f"# 调研报告：{topic}")
    lines.append("")
    lines.append(f"> 共 **{len(items)}** 个调研对象，按 cluster（A→H）→ tier（core→strong→contextual）分组。")
    lines.append("> 每个对象覆盖 fields.yaml 全部字段；`[不确定]` 与 `uncertain` 数组中的字段已跳过。")
    lines.append("")

    # anchors per item
    def anchor(d):
        return "item-" + re.sub(r"[^A-Za-z0-9]+", "-", d.get("id", "") or d["_source_file"]).strip("-")

    # ---- TOC grouped by cluster ----
    lines.append("## 目录")
    lines.append("")
    idx = 0
    cur_cluster = None
    for d in items:
        c = d.get("cluster", "?")
        if c != cur_cluster:
            cur_cluster = c
            lines.append("")
            lines.append(f"### {CLUSTER_TITLES.get(c, c)}")
            lines.append("")
        idx += 1
        name = (d.get("name") or d["_source_file"]).strip()
        # name can be very long (bilingual) → keep first segment for TOC
        toc_name = re.split(r"[—\n]", name, maxsplit=1)[0].strip()
        if len(toc_name) > 70:
            toc_name = toc_name[:70] + "…"
        tier = d.get("tier", "")
        yr = clean_year(field_lookup(d, "year"))
        ven = clean_venue(field_lookup(d, "venue"))
        par = clean_paradigm(field_lookup(d, "paradigm"))
        tags = [f"{c}·{TIER_LABEL.get(tier, tier)}"]
        if yr:
            tags.append(yr)
        if ven:
            tags.append(ven)
        if par:
            tags.append(par)
        meta = " · ".join(tags)
        lines.append(f"{idx}. [{toc_name}](#{anchor(d)}) — `{meta}`")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ---- BODY ----
    idx = 0
    cur_cluster = None
    for d in items:
        c = d.get("cluster", "?")
        if c != cur_cluster:
            cur_cluster = c
            lines.append("")
            lines.append(f"## {CLUSTER_TITLES.get(c, c)}")
            lines.append("")
        idx += 1
        name = (d.get("name") or d["_source_file"]).strip()
        lines.append(f'<a id="{anchor(d)}"></a>')
        lines.append("")
        lines.append(f"### {idx}. {name}")
        lines.append("")
        # header meta line
        hb = []
        if d.get("id"):
            hb.append(f"`{d['id']}`")
        if c:
            hb.append(f"cluster **{c}**")
        if d.get("tier"):
            hb.append(f"tier **{d['tier']}**")
        if hb:
            lines.append(" · ".join(hb))
            lines.append("")

        uncertain_set = uncertain_field_names(d)

        # per-category fields
        for cat_name, _cat_desc, fields in cats:
            if cat_name == "uncertain_reserved":
                continue  # handled below as a compact note
            rendered = []
            for fname, _fdesc in fields:
                if fname in uncertain_set:
                    continue
                val = field_lookup(d, fname)
                if is_skippable(val):
                    continue
                rendered.append((fname, val))
            if not rendered:
                continue
            lines.append(f"**【{cat_name}】**")
            lines.append("")
            for fname, val in rendered:
                lines.append(f"- **{fname}**: {format_value(val)}")
            lines.append("")

        # extra fields present in JSON but not in schema (skill-spec "其他信息").
        # Flatten one level into nested category dicts so a nested-structured item
        # (e.g. SocraSynth) contributes leaf keys, not whole category blobs. Skip
        # the literal uncertain marker, schema fields, meta keys, and uncertain set.
        category_container_keys = {cn for cn, _, _ in cats} | set(CATEGORY_MAPPING.keys())
        extra = []
        seen_extra = set()

        def collect_extra(key, value):
            if key in SKIP_KEYS or key in META_KEYS or key == "_source_file":
                return
            if key == UNCERTAIN_MARK or "不确定" in key:
                return
            if key in schema_field_names or key in uncertain_set:
                return
            # descend into nested category container dicts rather than dumping them
            if isinstance(value, dict) and (key in category_container_keys
                                            or key.startswith("uncertain")):
                for k2, v2 in value.items():
                    collect_extra(k2, v2)
                return
            if is_skippable(value) or key in seen_extra:
                return
            seen_extra.add(key)
            extra.append((key, value))

        for k, v in d.items():
            collect_extra(k, v)
        if extra:
            lines.append("**【其他信息】**")
            lines.append("")
            for k, v in extra:
                lines.append(f"- **{k}**: {format_value(v)}")
            lines.append("")

        # uncertain array — list each flagged field name (do not compress)
        if uncertain_set:
            lines.append("**【不确定字段 / uncertain】**")
            lines.append("")
            for fn in sorted(uncertain_set):
                lines.append(f"- {fn}")
            lines.append("")

        lines.append("---")
        lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT} ({len(items)} items, {OUT.stat().st_size//1024} KB)")


if __name__ == "__main__":
    main()
