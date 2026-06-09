#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the 50 deep-research JSONs into a single markdown report.

- Items ordered by outline.yaml (survey anchor -> clusters A..F).
- TOC shows: index, name (anchor link), + 4 summary fields (condensed to leading token).
- Detail sections grouped by the 4 research dimensions (from fields.yaml field_groups).
- Skips values containing [不确定], fields listed in each item's uncertain[] array,
  and None/empty values.
"""
import json
import re
import glob
import os
import unicodedata

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE, "results")
OUTLINE = os.path.join(BASE, "outline.yaml")
FIELDS = os.path.join(BASE, "fields.yaml")
OUT = os.path.join(BASE, "report.md")

UNCERTAIN_MARKER = "[不确定]"
# Summary fields to show inline in the TOC (user-selected).
TOC_FIELDS = ["parameter_updates_required", "year", "venue", "theory_school", "adaptation_type"]

try:
    import yaml
except ImportError:
    raise SystemExit("PyYAML required: pip install pyyaml")


# --------------------------------------------------------------------------
# Load field structure (field_groups) and outline ordering
# --------------------------------------------------------------------------
def load_field_structure(path):
    """Return (groups, field_label_map). groups = [(group_title, [field_names])]."""
    data = yaml.safe_load(open(path, encoding="utf-8"))
    groups = []
    for g in data.get("field_groups", []):
        names = [f["name"] for f in g["fields"]]
        groups.append((g["group"], names))
    # reserved uncertain fields rendered separately if filled
    reserved = [f["name"] for f in data.get("uncertain_fields", [])]
    return groups, reserved


def load_outline_order(path):
    """Return ordered list of (id, name, slug, cluster, url) and a slug->meta map."""
    data = yaml.safe_load(open(path, encoding="utf-8"))
    order = []
    for it in data["items"]:
        order.append({
            "id": it["id"],
            "name": it["name"],
            "slug": slugify_filename(it["name"]),
            "cluster": it.get("cluster", ""),
            "url": it.get("url", ""),
            "authors": it.get("authors", ""),
            "year": it.get("year", ""),
            "venue": it.get("venue", ""),
        })
    return data.get("topic", ""), order


def slugify_filename(name):
    """Mirror the slug rule used at research-deep launch time."""
    s = name.replace(" ", "_")
    s = re.sub(r"[^A-Za-z0-9_-]", "", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s[:80]


# --------------------------------------------------------------------------
# Anchors + value formatting
# --------------------------------------------------------------------------
def anchor_id(name):
    """GitHub-style anchor: lowercase, strip punctuation, spaces->hyphens.
    Keeps CJK characters (GitHub preserves them)."""
    s = name.strip().lower()
    out = []
    for ch in s:
        if ch.isspace():
            out.append("-")
        elif ch == "-" or ch.isalnum():
            out.append(ch)
        elif unicodedata.category(ch).startswith("L"):  # CJK / other letters
            out.append(ch)
        # else drop punctuation
    return "".join(out)


def is_skippable(value):
    if value is None:
        return True
    if isinstance(value, str):
        v = value.strip()
        if v == "" or UNCERTAIN_MARKER in v:
            return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def format_value(value):
    """Render a field value as markdown. Handles str / list / list-of-dict / dict."""
    if isinstance(value, str):
        v = value.strip()
        if len(v) > 160:
            # long text -> blockquote-friendly, soft-break long runs
            return "<br>" + v
        return v
    if isinstance(value, list):
        if all(isinstance(x, dict) for x in value):
            lines = []
            for d in value:
                kv = " | ".join(f"{k}: {vv}" for k, vv in d.items()
                                if not is_skippable(vv))
                if kv:
                    lines.append(f"  - {kv}")
            return "<br>" + "<br>".join(lines) if lines else ""
        # simple list
        parts = [str(x).strip() for x in value if not is_skippable(x)]
        if not parts:
            return ""
        joined = "; ".join(parts)
        if len(joined) > 160:
            return "<br>" + "<br>".join(f"  - {p}" for p in parts)
        return joined
    if isinstance(value, dict):
        lines = []
        for k, vv in value.items():
            if is_skippable(vv):
                continue
            lines.append(f"  - **{k}**: {format_value(vv)}")
        return "<br>" + "<br>".join(lines) if lines else ""
    return str(value)


def condense_toc(value):
    """For TOC: take the leading enum token before a Chinese/ASCII paren or punctuation."""
    if is_skippable(value):
        return ""
    s = str(value).strip()
    # cut at the EARLIEST separator position (paren / comma / sentence end / dash)
    cut = len(s)
    for sep in ["（", "(", "，", ",", "；", ";", "。", "：", ":", " — ", "—"]:
        idx = s.find(sep)
        if 0 < idx < cut:
            cut = idx
    return s[:cut].strip()[:48]


def extract_year(value):
    if is_skippable(value):
        return ""
    m = re.search(r"(19|20)\d{2}", str(value))
    return m.group(0) if m else condense_toc(value)


# --------------------------------------------------------------------------
# Build report
# --------------------------------------------------------------------------
def main():
    groups, reserved = load_field_structure(FIELDS)
    topic, order = load_outline_order(OUTLINE)

    # load all JSONs keyed by slug
    data_by_slug = {}
    for fp in glob.glob(os.path.join(RESULTS_DIR, "*.json")):
        slug = os.path.basename(fp)[:-5]
        try:
            data_by_slug[slug] = json.load(open(fp, encoding="utf-8"))
        except Exception as e:
            print(f"WARN parse {fp}: {e}")

    # all defined field names (for "other fields" detection)
    defined = set()
    for _, names in groups:
        defined.update(names)
    defined.update(reserved)
    INTERNAL = {"uncertain", "_source_file"}

    md = []
    md.append(f"# 调研报告：{topic}\n")
    md.append(f"> 共 {len(order)} 个研究对象 · 按 综述锚点 → 6 大簇（A 现象 / B 机制 / C 推理 / D 扩展 / E 测试时适应 / F 智能体）组织。\n")
    md.append("> 字段维度：基本信息 + 机制与理论 + 实证与任务迁移 + 推理与智能体效果 + 局限与开放问题。值为「[不确定]」或列于 uncertain 的字段已跳过。\n")
    md.append("\n> 生成日期：2026-06-08\n")

    # ---------------- TOC ----------------
    md.append("\n## 目录\n")
    current_cluster = None
    idx = 0
    for meta in order:
        slug = meta["slug"]
        d = data_by_slug.get(slug)
        if d is None:
            continue
        if meta["cluster"] != current_cluster:
            current_cluster = meta["cluster"]
            md.append(f"\n**{current_cluster}**\n")
        idx += 1
        # TOC uses the OUTLINE english name for a clean anchor + readable label
        label = meta["name"]
        # Anchor MUST be built from the exact detail-header text "### {id} — {label}"
        # so GitHub's computed slug matches (em-dash dropped, both spaces -> hyphens).
        anc = anchor_id(f"{meta['id']} — {label}")
        # summary tags
        pu = condense_toc(d.get("parameter_updates_required", ""))
        yr = extract_year(d.get("year", "")) or str(meta.get("year", ""))
        ven = condense_toc(d.get("venue", "")) or condense_toc(str(meta.get("venue", "")))
        ts = condense_toc(d.get("theory_school", ""))
        ad = condense_toc(d.get("adaptation_type", ""))
        tags = []
        if pu:
            tags.append(f"参数更新: {pu}")
        if yr:
            tags.append(f"{yr}")
        if ven:
            tags.append(f"{ven}")
        if ts:
            tags.append(f"学派: {ts}")
        if ad:
            tags.append(f"适应: {ad}")
        tagstr = " · ".join(tags)
        md.append(f"{idx}. [{meta['id']} — {label}](#{anc}) — {tagstr}")

    # ---------------- Details ----------------
    md.append("\n\n---\n\n## 详细内容\n")
    current_cluster = None
    for meta in order:
        slug = meta["slug"]
        d = data_by_slug.get(slug)
        if d is None:
            continue
        if meta["cluster"] != current_cluster:
            current_cluster = meta["cluster"]
            md.append(f"\n## {current_cluster}\n")

        label = meta["name"]
        md.append(f"\n### {meta['id']} — {label}\n")
        if meta.get("url"):
            md.append(f"🔗 {meta['url']}\n")

        uncertain_list = set(d.get("uncertain", []) or [])

        # by dimension group
        for gtitle, names in groups:
            rendered = []
            for fn in names:
                if fn in uncertain_list:
                    continue
                if fn not in d:
                    continue
                val = d[fn]
                if is_skippable(val):
                    continue
                rendered.append((fn, val))
            if not rendered:
                continue
            md.append(f"\n**{gtitle}**\n")
            for fn, val in rendered:
                md.append(f"- **{fn}**: {format_value(val)}")

        # reserved uncertain fields that DID get filled
        res_rendered = []
        for fn in reserved:
            if fn in uncertain_list or fn not in d:
                continue
            val = d[fn]
            if is_skippable(val):
                continue
            res_rendered.append((fn, val))
        if res_rendered:
            md.append("\n**扩展（保留字段）**\n")
            for fn, val in res_rendered:
                md.append(f"- **{fn}**: {format_value(val)}")

        # other fields present in JSON but not defined
        others = []
        for k, v in d.items():
            if k in INTERNAL or k in defined:
                continue
            if is_skippable(v):
                continue
            others.append((k, v))
        if others:
            md.append("\n**其他信息**\n")
            for k, v in others:
                md.append(f"- **{k}**: {format_value(v)}")

        # uncertain field list (each on its own line)
        if uncertain_list:
            md.append("\n**不确定字段**\n")
            for fn in sorted(uncertain_list):
                md.append(f"- {fn}")

    open(OUT, "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"report written: {OUT}")
    print(f"items rendered: {idx}/{len(order)}")


if __name__ == "__main__":
    main()
