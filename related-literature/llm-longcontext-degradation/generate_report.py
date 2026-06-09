#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Synthesize results/*.json into a single markdown report.

Reads fields.yaml for the field structure and per-field display order, walks
every item dossier, skips uncertain / not-applicable / empty values, and emits:
  - a "Relevance to mediator-coevo" focused section,
  - a table of contents (Category / Year / Type / Source columns, normalized),
  - per-category (A->B->C->D) item details grouped by field category.
"""
from __future__ import annotations

import glob
import json
import re
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
RESULTS_DIR = BASE / "results"
FIELDS_PATH = BASE / "fields.yaml"
OUTLINE_PATH = BASE / "outline.yaml"
REPORT_PATH = BASE / "report.md"

# ----------------------------------------------------------------------------
# Skip / internal handling
# ----------------------------------------------------------------------------
SKIP_KEYS = {"_source_file", "uncertain", "id", "category", "topic", "slug", "name"}
NA_MARKERS = ("[不适用]", "[不确定]", "[NA]", "[N/A]")

# field_category key -> human display heading (Chinese, matching value language)
GROUP_HEADINGS = {
    "basic": "基本信息",
    "core_findings": "核心发现",
    "methodology": "方法",
    "analysis": "分析",
    "mitigation": "缓解方法",
    "benchmark_specific": "基准专属维度",
    "meta": "元信息与定位",
}

CATEGORY_NAMES = {
    "A": "A · 基础现象论文 (Foundational phenomena)",
    "B": "B · 诊断基准 (Diagnostic benchmarks)",
    "C": "C · 机制与缓解 (Mechanisms & mitigations)",
    "D": "D · 2025–26 前沿 (Frontier)",
}

# field name -> human label (Chinese) for body rendering
FIELD_LABELS = {
    "name": "名称", "authors": "作者", "year": "年份", "venue": "发表",
    "url": "来源链接", "source_type": "来源类型", "type": "类型",
    "influence_citations": "影响力/引用",
    "phenomenon_described": "所述现象", "key_finding": "关键结论",
    "hypothesized_mechanism": "假设机制", "failure_mode_taxonomy": "失效模式归类",
    "models_evaluated": "评测模型", "model_generation": "模型世代",
    "reasoning_vs_base": "推理 vs 基础模型", "task_or_benchmark": "任务/基准",
    "context_lengths_tested": "测试上下文长度", "degradation_metric": "退化度量",
    "context_distribution": "上下文分布", "agentic_vs_single_turn": "智能体 vs 单轮",
    "degradation_pattern": "退化形态", "degradation_curve_shape": "退化曲线形状",
    "aggravating_factors": "加剧因素", "degradation_onset_length": "退化起始长度",
    "effective_vs_advertised_gap": "有效 vs 标称差距",
    "distractor_type_noise_taxonomy": "干扰/噪声类型",
    "needle_type_retrieval_difficulty": "针类型/检索难度",
    "needle_question_similarity": "针-问相似度",
    "retrieval_vs_reasoning_isolation": "检索 vs 推理解耦",
    "mitigation_proposed": "提出的缓解", "mitigation_type": "缓解类型",
    "where_in_pipeline": "干预环节", "mechanism_targeted": "针对机制",
    "compute_overhead": "计算开销", "requires_model_internals": "需模型内部访问",
    "task_taxonomy": "任务分类", "synthetic_vs_realistic": "合成 vs 真实",
    "max_context_length": "最大上下文长度", "length_controllability": "长度可控性",
    "shortcut_resistance": "抗捷径", "leakage_mitigation": "防泄漏",
    "output_type": "输出类型", "evaluation_metric_type": "评测指标类型",
    "needle_count": "针数量", "position_sensitivity_reported": "报告位置敏感性",
    "public_leaderboard_live": "公开排行榜", "modality": "模态",
    "model_class_coverage": "模型类别覆盖", "mechanism_evidence_level": "机制证据强度",
    "limitations": "局限", "relation_to_other_works": "与其他工作关系",
    "relevance_to_mediator_coevo": "与 mediator-coevo 的相关性",
}

LONG_TEXT_THRESHOLD = 100

# group-dict keys that may wrap fields in nested-structure JSONs
GROUP_DICT_KEYS = {
    "basic", "core_findings", "methodology", "analysis", "mitigation",
    "benchmark_specific", "meta",
    "基本信息", "核心发现", "方法", "分析", "缓解方法", "基准专属维度",
    "元信息", "元信息与定位",
}


def derive_id(d: dict, filename: str) -> str:
    """Top-level id, else parse the filename prefix (e.g. 'A12_Foo.json' -> 'A12')."""
    if d.get("id"):
        return str(d["id"])
    m = re.match(r"([A-D]\d+)_", Path(filename).name)
    return m.group(1) if m else "Z0"


def flatten_item(d: dict) -> dict:
    """Return a single-level {field_name: value} view.

    Handles both flat JSONs (fields at top level) and nested JSONs (fields
    inside group dicts like 'basic'/'analysis'/...). Top-level scalars win;
    nested fields fill in. Internal/structural keys are dropped.
    """
    flat = {}
    # nested groups first (so flat top-level can override if duplicated)
    for k, v in d.items():
        if k in GROUP_DICT_KEYS and isinstance(v, dict):
            for fk, fv in v.items():
                flat[fk] = fv
    for k, v in d.items():
        if k in GROUP_DICT_KEYS and isinstance(v, dict):
            continue
        if k in ("id", "category", "topic", "_source_file", "uncertain", "slug"):
            continue
        flat[k] = v
    return flat


def load_fields():
    """Return (ordered_groups, field_to_group). ordered_groups = [(key, [field_names])]."""
    data = yaml.safe_load(FIELDS_PATH.read_text(encoding="utf-8"))
    fc = data.get("field_categories", {})
    ordered = []
    field_to_group = {}
    if isinstance(fc, dict):
        for gkey, gbody in fc.items():
            if not isinstance(gbody, dict):
                continue
            names = [f["name"] for f in gbody.get("fields", [])]
            ordered.append((gkey, names))
            for n in names:
                field_to_group[n] = gkey
    return ordered, field_to_group


def is_skippable(value) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return True
        if any(m in s for m in NA_MARKERS):
            return True
    if isinstance(value, (list, dict)) and len(value) == 0:
        return True
    return False


def fmt_value(value) -> str:
    """Format a JSON value into markdown-friendly text."""
    if isinstance(value, list):
        parts = []
        for el in value:
            if isinstance(el, dict):
                parts.append(" | ".join(f"{k}: {v}" for k, v in el.items()))
            else:
                parts.append(str(el))
        if all(len(p) < 40 for p in parts) and len(parts) <= 6:
            return ", ".join(parts)
        return "<br>".join(f"- {p}" for p in parts)
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    s = str(value).strip()
    return s


def normalize_year(item) -> str:
    raw = str(item.get("year", ""))
    yrs = re.findall(r"(19|20)\d{2}", raw)
    if yrs:
        # findall with group returns prefixes; redo with full match
        m = re.findall(r"((?:19|20)\d{2})", raw)
        if m:
            uniq = sorted(set(m))
            return uniq[-1] if len(uniq) == 1 else f"{uniq[0]}"
    return raw[:8] or "—"


def lead_token(text: str) -> str:
    """Leading keyword before any bracket / punctuation."""
    if not text:
        return "—"
    t = re.split(r"[（(\[/，,;；:：—-]", text.strip())[0].strip()
    return t or text.strip()[:16]


def normalize_type(item) -> str:
    return lead_token(str(item.get("type", "")))[:18] or "—"


def normalize_source(item) -> str:
    s = str(item.get("source_type", "")).lower()
    table = [
        ("leaderboard", "leaderboard"), ("排行榜", "leaderboard"),
        ("dataset", "dataset"), ("数据集", "dataset"),
        ("preprint", "preprint"), ("预印本", "preprint"), ("arxiv", "preprint"),
        ("blog", "blog"), ("博客", "blog"),
        ("report", "report"), ("报告", "report"), ("industry", "report"),
        ("paper", "paper"), ("论文", "paper"), ("同行评审", "paper"),
    ]
    for needle, label in table:
        if needle in s:
            return label
    return lead_token(str(item.get("source_type", "")))[:12] or "—"


def anchor_for(item_id: str) -> str:
    return f"item-{item_id.lower()}"


def md_escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def main():
    ordered_groups, _ = load_fields()
    files = sorted(glob.glob(str(RESULTS_DIR / "*.json")))
    items = []
    for fn in files:
        try:
            raw = json.load(open(fn, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] skip unreadable {fn}: {e}")
            continue
        iid = derive_id(raw, fn)
        flat = flatten_item(raw)
        flat["id"] = iid
        flat["uncertain"] = raw.get("uncertain", [])
        flat["_source_file"] = Path(fn).name
        items.append(flat)

    def sort_key(d):
        cid = str(d.get("id", "Z999"))
        cat = cid[0]
        num = int(re.sub(r"\D", "", cid) or 0)
        return (cat, num)

    items.sort(key=sort_key)
    topic = "LLM 长上下文性能退化 (LLM Long-Context Performance Degradation)"

    out = []
    out.append(f"# {topic} — 调研报告\n")
    out.append(f"> 共 **{len(items)}** 个调研对象，按类别 A→B→C→D 组织。"
               f"字段值跳过 `[不确定]`/`[不适用]`/空值与 `uncertain` 列表项。\n")

    # ---- counts by category ----
    cats = {}
    for it in items:
        cats.setdefault(str(it.get("id", "?"))[0], []).append(it)
    def cat_short(c: str) -> str:
        name = CATEGORY_NAMES.get(c) or c
        return name.split(" · ")[0]

    out.append("**类别分布**：" + " · ".join(
        f"{cat_short(c)}={len(v)}" for c, v in sorted(cats.items())) + "\n")

    # ---- Relevance to mediator-coevo focused section ----
    out.append('\n<a id="mediator-coevo-relevance"></a>\n')
    out.append("## ⭐ 与 mediator-coevo 的相关性聚焦\n")
    out.append("> 抽取各条目的 `relevance_to_mediator_coevo` 字段（跳过空/不适用）。"
               "高相关条目（多轮对话、调解者架构、智能体上下文退化）置顶。\n")
    HIGH = {"D6", "D7", "A10", "D16", "A3", "A4", "D8", "D17", "D18", "D19"}
    rel_items = [it for it in items if not is_skippable(it.get("relevance_to_mediator_coevo"))]
    high = [it for it in rel_items if str(it.get("id")) in HIGH]
    rest = [it for it in rel_items if str(it.get("id")) not in HIGH]
    if high:
        out.append("### 高相关\n")
        for it in high:
            iid = it.get("id", "?")
            out.append(f"- **[{iid}](#{anchor_for(iid)}) {lead_token(str(it.get('name','')))}** — "
                       f"{fmt_value(it['relevance_to_mediator_coevo'])}")
        out.append("")
    if rest:
        out.append("<details><summary>其余条目的相关性（点击展开）</summary>\n")
        for it in rest:
            iid = it.get("id", "?")
            out.append(f"- **[{iid}](#{anchor_for(iid)})** — {fmt_value(it['relevance_to_mediator_coevo'])}")
        out.append("\n</details>\n")

    # ---- Table of contents ----
    out.append('\n<a id="toc"></a>\n')
    out.append("## 目录\n")
    out.append("| # | 条目 | 类别 | 年份 | 类型 | 来源 |")
    out.append("|---|------|------|------|------|------|")
    for i, it in enumerate(items, 1):
        iid = it.get("id", "?")
        name = md_escape_cell(str(it.get("name", "")).split("（")[0].split("(")[0].strip()[:64])
        out.append(
            f"| {i} | [{iid} {name}](#{anchor_for(iid)}) "
            f"| {iid[0]} | {normalize_year(it)} | {md_escape_cell(normalize_type(it))} "
            f"| {normalize_source(it)} |"
        )
    out.append("")

    # ---- Body grouped by category ----
    for cat in ["A", "B", "C", "D"]:
        cat_items = cats.get(cat, [])
        if not cat_items:
            continue
        out.append(f"\n---\n\n## {CATEGORY_NAMES.get(cat, cat)}\n")
        for it in cat_items:
            iid = it.get("id", "?")
            name = str(it.get("name", ""))
            out.append(f'<a id="{anchor_for(iid)}"></a>')
            out.append(f"### {iid}. {name}\n")
            url = it.get("url")
            if not is_skippable(url):
                out.append(f"🔗 {fmt_value(url)}\n")
            out.append("[↑ 返回目录](#toc)\n")

            # fields grouped per fields.yaml order
            rendered_fields = set()
            for gkey, names in ordered_groups:
                rows = []
                for fname in names:
                    if fname in SKIP_KEYS or fname in ("url",):
                        rendered_fields.add(fname)
                        continue
                    val = it.get(fname)
                    rendered_fields.add(fname)
                    if is_skippable(val):
                        continue
                    label = FIELD_LABELS.get(fname, fname)
                    text = fmt_value(val)
                    if len(text) > LONG_TEXT_THRESHOLD:
                        rows.append(f"**{label}**：\n> {text}\n")
                    else:
                        rows.append(f"- **{label}**：{text}")
                if rows:
                    out.append(f"**{GROUP_HEADINGS.get(gkey, gkey)}**\n")
                    out.extend(rows)
                    out.append("")

            # extra fields not in fields.yaml
            extras = []
            for k, v in it.items():
                if k in SKIP_KEYS or k in rendered_fields:
                    continue
                if is_skippable(v):
                    continue
                extras.append(f"- **{k}**：{fmt_value(v)}")
            if extras:
                out.append("**其他信息**\n")
                out.extend(extras)
                out.append("")

            # uncertain fields list
            unc = it.get("uncertain") or []
            if isinstance(unc, list) and unc:
                out.append("**不确定字段**（未经主来源充分核实）\n")
                for u in unc:
                    out.append(f"- {u}")
                out.append("")

    REPORT_PATH.write_text("\n".join(out), encoding="utf-8")
    print(f"[ok] wrote {REPORT_PATH} ({len(items)} items, {REPORT_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
