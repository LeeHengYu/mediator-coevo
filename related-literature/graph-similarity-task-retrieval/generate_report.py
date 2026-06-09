#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate report.md from the 144 deep-research JSON dossiers.

- Reads outline.yaml (topic, item order, subarea grouping, needs_verification)
- Reads fields.yaml (field -> category structure + descriptions)
- Reads results/*.json (flat OR nested structure both supported)
- ToC grouped by sub-area A-F; summary line = year·venue | use | learned/fixed | code
- Skips values that are [不确定] / [不适用] / empty / listed in the file's `uncertain` array
- Collects fields present in JSON but absent from fields.yaml into "其他信息"
"""
import json
import re
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
OUTLINE = BASE / "outline.yaml"
FIELDS = BASE / "fields.yaml"
REPORT = BASE / "report.md"

SKIP_KEYS = {"_source_file", "uncertain"}
UNCERTAIN_MARKERS = ("[不确定]", "[不適用]", "[不适用]", "[N/A]", "[未知]")

# fields.yaml uses English category keys; map to readable headings + JSON-key aliases.
CATEGORY_LABEL = {
    "basic": "基本信息",
    "positioning": "定位",
    "method_mechanics": "方法机制（相似度 / 检索 / 表示）",
    "application": "应用",
    "evaluation": "评估",
    "project_relevance": "项目相关性 · OPD / mediated-coevo",
    "connections": "关联与脉络",
    "cross_cutting_dimensions": "横切维度",
}
# JSON nested-bucket keys that should be treated as category containers (not leaf fields)
CATEGORY_ALIAS_KEYS = set(CATEGORY_LABEL) | {
    "basic_info", "技术特性", "technical_features", "性能指标", "performance_metrics",
    "项目相关性", "方法机制", "评估", "应用", "定位", "关联", "横切维度", "基本信息",
}

SUBAREA_ORDER = ["A", "B", "C", "D", "E", "F"]


def load_fields():
    data = yaml.safe_load(FIELDS.open(encoding="utf-8"))
    fc = data["field_categories"]
    cat_fields = {}   # cat_key -> [(field, description), ...]
    field_to_cat = {}
    for cat_key, body in fc.items():
        flds = []
        for f in body.get("fields", []):
            flds.append((f["name"], f.get("description", "")))
            field_to_cat[f["name"]] = cat_key
        cat_fields[cat_key] = flds
    uncertain_reserved = data.get("uncertain", [])
    return cat_fields, field_to_cat, uncertain_reserved


def flatten(obj):
    """Flatten a result JSON to {field: value}, descending into category buckets."""
    out = {}
    for k, v in obj.items():
        if k in SKIP_KEYS:
            continue
        if isinstance(v, dict) and (k in CATEGORY_ALIAS_KEYS):
            for k2, v2 in v.items():
                if k2 not in SKIP_KEYS:
                    out.setdefault(k2, v2)
        else:
            out.setdefault(k, v)
    return out


def is_empty_or_uncertain(val, uncertain_list, field_name):
    if field_name in uncertain_list:
        return True
    if val is None:
        return True
    s = str(val).strip()
    if not s:
        return True
    for m in UNCERTAIN_MARKERS:
        if m in s:
            return True
    return False


def fmt_value(val, indent=0):
    """Format a field value for markdown."""
    if isinstance(val, list):
        if all(isinstance(x, dict) for x in val) and val:
            lines = []
            for d in val:
                parts = [f"**{k}**: {v}" for k, v in d.items() if v not in (None, "")]
                lines.append("  - " + " | ".join(parts))
            return "\n" + "\n".join(lines)
        items = [str(x).strip() for x in val if str(x).strip()]
        if not items:
            return ""
        joined = ", ".join(items)
        if len(items) <= 4 and len(joined) <= 90:
            return joined
        return "\n" + "\n".join(f"  - {x}" for x in items)
    if isinstance(val, dict):
        parts = []
        for k, v in val.items():
            if v in (None, ""):
                continue
            parts.append(f"  - **{k}**: {fmt_value(v)}")
        return "\n" + "\n".join(parts) if parts else ""
    s = str(val).strip()
    if len(s) > 160:
        # long text -> blockquote for readability
        return "\n  > " + s.replace("\n", "\n  > ")
    return s


def short(val, n=26):
    s = re.sub(r"\s+", " ", str(val)).strip()
    # cut at first sentence-ish boundary if long
    s = re.split(r"[；;。\n]", s)[0]
    return (s[: n - 1] + "…") if len(s) > n else s


def anchor_id(item_id):
    return f"item-{item_id.lower()}"


def main():
    outline = yaml.safe_load(OUTLINE.open(encoding="utf-8"))
    topic = outline["topic"]
    subareas = outline["subareas"]
    items = outline["items"]
    cat_fields, field_to_cat, _ = load_fields()

    def slug(name):
        s = name.replace("^", "").replace("/", "_").replace("\\", "_")
        s = re.sub(r"[()]", "", s)
        s = re.sub(r"[^A-Za-z0-9 _.+\-]", "", s).strip()
        s = re.sub(r"\s+", "_", s)
        s = re.sub(r"_+", "_", s)
        return s

    # load all results, attach to outline items
    loaded = []
    for it in items:
        fp = RESULTS / f"{it['id']}_{slug(it['name'])}.json"
        if not fp.exists():
            loaded.append((it, None, {}, []))
            continue
        d = json.load(fp.open(encoding="utf-8"))
        flat = flatten(d)
        unc = d.get("uncertain", []) or []
        if not isinstance(unc, list):
            unc = []
        loaded.append((it, d, flat, unc))

    n_total = len(loaded)
    n_have = sum(1 for _, d, _, _ in loaded if d is not None)

    out = []
    out.append(f"# 调研报告 · {topic}\n")
    out.append(
        "> 研究问题：如何通过**相似度**（尤其是**图结构**与**可学习的相似度度量**）"
        "检索相关的先验任务 / 技能 / 示例 / 案例，以驱动 (LLM) agent 的迁移、提示、课程与记忆。"
        "锚定 mediator-coevo / OPD 项目：一个中介(mediator)检索相似先验任务/技能/案例来引导 agent 协同进化。\n"
    )
    out.append(f"\n**条目总数**：{n_total}（已生成 dossier：{n_have}） · **子领域**：A–F · 数据来源：`results/*.json`（firecrawl + exa + academic-search 核实）\n")

    # group items by subarea
    by_sub = {k: [] for k in SUBAREA_ORDER}
    for rec in loaded:
        by_sub.setdefault(rec[0]["subarea"], []).append(rec)

    # ---------- Table of contents ----------
    out.append("\n## 目录\n")
    idx = 0
    for sa in SUBAREA_ORDER:
        recs = by_sub.get(sa, [])
        if not recs:
            continue
        out.append(f"\n### {sa}. {subareas.get(sa, '')}  ({len(recs)})\n")
        for it, d, flat, unc in recs:
            idx += 1
            name = flat.get("name") or it["name"]
            bits = []
            yr = flat.get("year") if not is_empty_or_uncertain(flat.get("year"), unc, "year") else it.get("year")
            ven = flat.get("venue") if not is_empty_or_uncertain(flat.get("venue"), unc, "venue") else it.get("venue")
            if yr or ven:
                bits.append(f"{yr or '?'} · {short(ven or '?', 22)}")
            tu = flat.get("target_use")
            if not is_empty_or_uncertain(tu, unc, "target_use"):
                bits.append(f"用途: {short(tu, 18)}")
            lf = flat.get("learned_vs_fixed")
            if not is_empty_or_uncertain(lf, unc, "learned_vs_fixed"):
                bits.append(short(lf, 16))
            code = flat.get("code_url")
            code_md = ""
            if not is_empty_or_uncertain(code, unc, "code_url") and "http" in str(code):
                # stop at whitespace, CJK chars, parens/brackets, quotes, and trailing punct
                m = re.search(r"https?://[^\s　-鿿（）()\[\]【】\"'，、；]+", str(code))
                if m:
                    url = m.group(0).rstrip(').,;:')
                    code_md = f" · [✓code]({url})"
            flag = " 🟡" if it.get("needs_verification") else ""
            summary = " | ".join(bits)
            out.append(f"{idx}. [{it['id']} · {name}](#{anchor_id(it['id'])}){flag} — {summary}{code_md}")
    out.append("\n🟡 = 初始 needs_verification 条目（晚近预印本，best-effort 填充，注意 dossier 内 [不确定] 标记）\n")

    # ---------- Details ----------
    out.append("\n---\n\n## 详细条目\n")
    for sa in SUBAREA_ORDER:
        recs = by_sub.get(sa, [])
        if not recs:
            continue
        out.append(f"\n# {sa}. {subareas.get(sa, '')}\n")
        for it, d, flat, unc in recs:
            name = flat.get("name") or it["name"]
            out.append(f'\n<a id="{anchor_id(it["id"])}"></a>')
            out.append(f"\n## {it['id']} · {name}\n")
            if d is None:
                out.append("_（缺少结果文件）_\n")
                continue

            shown = set()
            for cat_key, flds in cat_fields.items():
                rows = []
                for fname, fdesc in flds:
                    if fname == "name":
                        shown.add(fname)
                        continue
                    val = flat.get(fname)
                    if is_empty_or_uncertain(val, unc, fname):
                        continue
                    shown.add(fname)
                    rows.append((fname, fdesc, val))
                if not rows:
                    continue
                out.append(f"\n**{CATEGORY_LABEL.get(cat_key, cat_key)}**\n")
                for fname, fdesc, val in rows:
                    out.append(f"- **{fname}**: {fmt_value(val)}")

            # extra fields not defined in fields.yaml
            extras = []
            for k, v in flat.items():
                if k in shown or k in field_to_cat or k in SKIP_KEYS or k == "name":
                    continue
                if is_empty_or_uncertain(v, unc, k):
                    continue
                extras.append((k, v))
            if extras:
                out.append("\n**其他信息**\n")
                for k, v in extras:
                    out.append(f"- **{k}**: {fmt_value(v)}")

            # uncertain fields, listed one per line
            if unc:
                out.append("\n**不确定字段（[不确定]）**\n")
                for fn in unc:
                    out.append(f"- {fn}")
            out.append("")

    REPORT.write_text("\n".join(out), encoding="utf-8")
    print(f"wrote {REPORT}")
    print(f"items: {n_total} (with dossier: {n_have})")
    print(f"size: {REPORT.stat().st_size} bytes")


if __name__ == "__main__":
    main()
