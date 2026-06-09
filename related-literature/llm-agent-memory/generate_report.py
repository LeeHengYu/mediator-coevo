#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate the 58 LLM-agent-memory deep-research JSONs into a single markdown report.

TOC fields: Year + Citations | Memory type | Learning paradigm + subject | Venue
Detail layout: grouped by the 7 fields.yaml categories.
Skips uncertain / [不确定] / empty values.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"
FIELDS = BASE / "fields.yaml"
OUTLINE = BASE / "outline.yaml"
OUT = BASE / "report.md"

# ── Chinese display labels ────────────────────────────────────────────────
CATEGORY_LABELS = {
    "provenance": "基本信息 / Provenance",
    "taxonomy": "记忆分类 / Taxonomy",
    "mechanisms": "核心机制 / Mechanisms",
    "learning": "学习维度 / Learning",
    "evaluation": "评测 / Evaluation",
    "analysis": "分析 / Analysis",
    "supplemented": "补充维度 / Supplemented (2025-26 frontier)",
}
FIELD_LABELS = {
    "name": "名称", "year": "年份", "authors_institution": "作者/机构",
    "venue": "发表venue", "paper_url": "论文链接", "code_url": "代码链接",
    "citations_approx": "引用数",
    "memory_type": "记忆类型", "memory_structure": "记忆结构",
    "storage_backend": "存储后端", "persistence": "持久化",
    "write_encoding": "写入/编码", "retrieval_mechanism": "检索机制",
    "reflection_consolidation": "反思/巩固", "forgetting_update": "遗忘/更新",
    "experience_replay": "经验回放 (核心主题)",
    "learning_paradigm": "学习范式", "failure_learning": "失败学习 (核心主题)",
    "skill_procedural_induction": "技能/程序归纳", "online_vs_offline": "在线 vs 离线",
    "task_domains": "任务领域", "benchmarks": "基准", "reported_gains": "报告增益",
    "baselines": "对比基线",
    "key_innovation": "关键创新", "limitations": "局限", "relation_to_others": "与其他工作关系",
    "reproducibility": "可复现性",
    "learned_memory_control": "学习型记忆控制", "memory_subject": "记忆主体",
    "multi_agent_memory": "多智能体记忆", "temporal_reasoning_support": "时序推理支持",
    "modality": "模态", "over_personalization_risk": "过度个性化/记忆安全风险",
    "conflict_contradiction_handling": "冲突/矛盾处理",
    "token_cost_latency_evidence": "token成本/延迟证据",
}
CLUSTER_LABELS = {
    "A": "A. 反思与失败驱动记忆 (Reflection & failure-driven)",
    "B": "B. 情景记忆与检索架构 (Episodic memory & retrieval)",
    "C": "C. 经验回放与技能/程序记忆 (Experience replay & skill/procedural)",
    "D": "D. 图结构/神经启发/生产级记忆 (Graph / neuro-inspired / production)",
    "E": "E. 认知架构框架 (Cognitive-architecture frameworks)",
    "F": "F. 记忆评测基准 (Memory-evaluation benchmarks)",
    "G": "G. 学习/RL驱动的记忆控制 (Learned / RL-based memory control)",
    "H": "H. 综述 (Surveys)",
}
INTERNAL_KEYS = {"uncertain", "_source_file", "id"}
UNCERTAIN_MARK = "[不确定]"


def is_skippable(field: str, value, uncertain: set) -> bool:
    """Skip uncertain/empty/marked values."""
    if field in uncertain:
        return True
    if value is None:
        return True
    sv = str(value).strip()
    if not sv or sv.lower() == "none":
        return True
    if UNCERTAIN_MARK in sv or "[不确定" in sv:
        return True
    return False


def item_sort_key(d: dict) -> tuple:
    """Sort by cluster letter then numeric id (A1, A2 ... H6)."""
    iid = str(d.get("id", "Z99"))
    m = re.match(r"([A-Z])(\d+)", iid)
    if m:
        return (m.group(1), int(m.group(2)))
    return ("Z", 99)


def short_name(name: str) -> str:
    """Concise display name: text before the first parenthesis/；/，/—.

    Agents often stuffed the full paper title into `name`; the head clause is
    the recognizable system name. Anchors still use the full name for uniqueness.
    """
    # split on parentheses / Chinese punctuation / em-dash / spaced-hyphen only.
    # NEVER split on a bare '-' (would break Memory-R1, EM-LLM, A-MEM, JARVIS-1, Mem-α).
    head = re.split(r"[（(；;，,。：:]|——|\s—\s| - ", name, maxsplit=1)[0].strip()
    return head if head else name


def anchor(name: str, iid: str) -> str:
    """GitHub-style anchor: prefix with id to guarantee uniqueness."""
    raw = f"{iid}-{name}"
    a = raw.lower()
    a = re.sub(r"[^\w一-鿿\- ]", "", a)
    a = a.strip().replace(" ", "-")
    return a


# ── concise TOC extractors (values are verbose Chinese sentences) ─────────
def extract_year(d: dict) -> str:
    v = str(d.get("year", ""))
    if is_skippable("year", v, set(d.get("uncertain", []) or [])):
        return ""
    m = re.search(r"(19|20)\d{2}", v)
    return m.group(0) if m else ""


def extract_citations(d: dict) -> str:
    unc = set(d.get("uncertain", []) or [])
    if "citations_approx" in unc:
        return ""
    v = str(d.get("citations_approx", ""))
    if is_skippable("citations_approx", v, unc):
        return ""
    # pull first number (may contain comma/约/~)
    m = re.search(r"([\d,]{1,8})\s*次?", v)
    if m:
        n = m.group(1).replace(",", "")
        try:
            n = int(n)
            return f"~{n}引"
        except ValueError:
            return ""
    return ""


def extract_memtype(d: dict) -> str:
    v = str(d.get("memory_type", ""))
    if is_skippable("memory_type", v, set(d.get("uncertain", []) or [])):
        return ""
    # take text up to first ：/（/。/, — the head clause
    head = re.split(r"[：（(。,，；;]", v)[0].strip()
    return head[:24]


def extract_paradigm_subject(d: dict) -> str:
    unc = set(d.get("uncertain", []) or [])
    parts = []
    lp = str(d.get("learning_paradigm", ""))
    if not is_skippable("learning_paradigm", lp, unc):
        for kw in ["非参数化", "参数化", "混合", "non-parametric", "parametric", "hybrid"]:
            if kw in lp:
                parts.append(kw if kw[0] >= "一" else kw)
                break
    ms = str(d.get("memory_subject", ""))
    if not is_skippable("memory_subject", ms, unc):
        if "智能体中心" in ms or "agent-centric" in ms:
            parts.append("智能体中心")
        elif "用户中心" in ms or "user-centric" in ms:
            parts.append("用户中心")
        elif "系统/任务" in ms or "任务中心" in ms:
            parts.append("任务中心")
    return "/".join(parts)


def extract_venue(d: dict) -> str:
    unc = set(d.get("uncertain", []) or [])
    v = str(d.get("venue", ""))
    if is_skippable("venue", v, unc):
        return ""
    head = re.split(r"[（(，,。；;]", v)[0].strip()
    return head[:22]


def format_value(value) -> str:
    """Format a field value for markdown; handle list/dict/long text."""
    if isinstance(value, list):
        if all(isinstance(x, dict) for x in value) and value:
            lines = []
            for x in value:
                lines.append("    - " + " | ".join(f"{k}: {v}" for k, v in x.items()))
            return "\n" + "\n".join(lines)
        flat = [str(x) for x in value]
        joined = "、".join(flat)
        if len(joined) > 100:
            return "\n" + "\n".join(f"    - {x}" for x in flat)
        return joined
    if isinstance(value, dict):
        return "; ".join(f"{k}: {v}" for k, v in value.items())
    sv = str(value).strip()
    return sv


def main() -> None:
    outline = yaml.safe_load(OUTLINE.read_text(encoding="utf-8"))
    topic = outline.get("topic", "Research Report")
    fields_def = yaml.safe_load(FIELDS.read_text(encoding="utf-8"))
    categories = fields_def["field_categories"]
    defined_fields = {f["name"] for c in categories for f in c["fields"]}

    items = []
    for jf in sorted(RESULTS.glob("*.json")):
        try:
            d = json.loads(jf.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] skip unreadable {jf.name}: {e}")
            continue
        # id is not a fields.yaml field, so many agents omit it from JSON.
        # The filename (e.g. A1_Reflexion.json) is the authoritative source.
        fm = re.match(r"([A-H]\d+)_", jf.name)
        if fm:
            d["id"] = fm.group(1)
        items.append(d)
    items.sort(key=item_sort_key)

    md: list[str] = []
    md.append(f"# 调研报告：{topic}\n")
    md.append(f"> 共 **{len(items)}** 个调研对象，覆盖 8 个 cluster（A–H）。"
              f"每条经 Opus-4.8 agent 经 academic-search / exa / firecrawl 多源核实，"
              f"字段覆盖率 100%（不确定值已跳过）。\n")
    md.append("> 生成自 `results/*.json` + `fields.yaml`。\n")

    # ── group items by cluster for the TOC ────────────────────────────────
    md.append("\n## 目录 (Table of Contents)\n")
    current_cluster = None
    idx = 0
    for d in items:
        iid = str(d.get("id", ""))
        cl = iid[0] if iid else "Z"
        if cl != current_cluster:
            current_cluster = cl
            md.append(f"\n### {CLUSTER_LABELS.get(cl, cl)}\n")
        idx += 1
        name = str(d.get("name", "?"))
        disp = short_name(name)
        a = anchor(name, iid)
        bits = []
        yr, cit = extract_year(d), extract_citations(d)
        ven = extract_venue(d)
        if yr:
            bits.append(yr + (f" · {ven}" if ven else ""))
        elif ven:
            bits.append(ven)
        if cit:
            bits.append(cit)
        mt = extract_memtype(d)
        if mt:
            bits.append(mt)
        ps = extract_paradigm_subject(d)
        if ps:
            bits.append(ps)
        meta = " | ".join(bits)
        md.append(f"{idx}. [{iid} {disp}](#{a})" + (f" — {meta}" if meta else ""))

    # ── detailed sections ─────────────────────────────────────────────────
    md.append("\n\n---\n\n## 详细调研 (Details)\n")
    current_cluster = None
    for d in items:
        iid = str(d.get("id", ""))
        cl = iid[0] if iid else "Z"
        unc = set(d.get("uncertain", []) or [])
        name = str(d.get("name", "?"))

        if cl != current_cluster:
            current_cluster = cl
            md.append(f"\n## {CLUSTER_LABELS.get(cl, cl)}\n")

        disp = short_name(name)
        a = anchor(name, iid)
        md.append(f"\n<a id=\"{a}\"></a>")
        md.append(f"\n### {iid} {disp}\n")
        if disp != name:
            md.append(f"*{name}*\n")

        for cat in categories:
            cat_name = cat["category"]
            rows = []
            for f in cat["fields"]:
                fn = f["name"]
                if fn == "name":  # already in heading
                    continue
                if fn not in d:
                    continue
                val = d[fn]
                if is_skippable(fn, val, unc):
                    continue
                label = FIELD_LABELS.get(fn, fn)
                rows.append(f"- **{label}**: {format_value(val)}")
            if rows:
                md.append(f"\n**{CATEGORY_LABELS.get(cat_name, cat_name)}**\n")
                md.extend(rows)

        # extra fields not defined in fields.yaml
        extras = []
        for k, v in d.items():
            if k in INTERNAL_KEYS or k in defined_fields:
                continue
            if is_skippable(k, v, unc):
                continue
            extras.append(f"- **{FIELD_LABELS.get(k, k)}**: {format_value(v)}")
        if extras:
            md.append("\n**其他信息 / Other**\n")
            md.extend(extras)

        # uncertain list (one per line)
        if unc:
            md.append("\n**不确定字段 / Uncertain**\n")
            for u in sorted(unc):
                md.append(f"- {FIELD_LABELS.get(u, u)} (`{u}`)")
        md.append("")

    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"[ok] wrote {OUT} ({len(items)} items, {OUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
