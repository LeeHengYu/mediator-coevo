#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the consolidated markdown report from per-item result JSONs.

- Item order, cluster, id, and name->slug mapping come from outline.yaml (authoritative).
- Field structure / categories / descriptions come from fields.yaml.
- Per item, every defined field is rendered under its category, skipping values that
  are uncertain (listed in the item's `uncertain` array), contain "[不确定]", or are empty.
- TOC is grouped by cluster with compact leading-tag summary columns.
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

# JSON category keys may appear in any language; map fields.yaml category -> JSON keys.
CATEGORY_MAPPING = {
    "基本信息": ["基本信息", "basic_info"],
    "问题定位": ["问题定位", "problem_framing", "problem"],
    "技术方法": ["技术方法", "technical_method", "method"],
    "评测": ["评测", "evaluation", "eval"],
    "分析": ["分析", "analysis"],
    "关系": ["关系", "relations"],
    "survey_specific": ["survey_specific", "综述专属"],
}
NEST_KEYS = {k for keys in CATEGORY_MAPPING.values() for k in keys}
SKIP_KEYS = {"_source_file", "uncertain"}

# Cluster display order + human labels (grouped-by-cluster layout).
CLUSTER_ORDER = ["A", "B", "C", "D", "E", "F", "G", "BENCH", "SURVEY"]
CLUSTER_LABEL = {
    "A": "Cluster A — Training-time cross-task generalization (instruction / multi-task tuning)",
    "B": "Cluster B — Meta-learning for language models",
    "C": "Cluster C — Prompt / soft-prompt transfer",
    "D": "Cluster D — ICL mechanism & cross-task ICL (incl. D↔G vector-carrier bridge)",
    "E": "Cluster E — LLM agents: experience reuse & skill libraries (core bridge cluster)",
    "F": "Cluster F — Case-based reasoning & memory-augmented transfer",
    "G": "Cluster G — Task representation & transferability prediction",
    "BENCH": "Benchmarks — cross-task generalization & agent transfer",
    "SURVEY": "Surveys — cross-task transfer, agent memory, lifelong / self-evolving agents",
}

# TOC summary columns (compact leading-tag form). (json_field, header).
TOC_FIELDS = [
    ("year", "Year"),
    ("venue", "Venue"),
    ("transfer_paradigm", "Paradigm"),
    ("knowledge_carrier", "Carrier"),
    ("parameter_update", "Params"),
    ("learns_from_failure", "Fail?"),
    ("transfer_axis_tested", "Transfer axis"),
]

UNCERTAIN_MARK = "[不确定]"


def slugify(name: str) -> str:
    s = name.strip().replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^\w\s一-鿿-]", "", s)
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def anchor(name: str) -> str:
    """GitHub-style anchor: lowercase, spaces->-, drop most punctuation, keep CJK."""
    a = name.strip().lower()
    a = a.replace(" ", "-")
    a = re.sub(r"[^\w一-鿿-]", "", a)
    return a


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def find_field(data: dict, field: str):
    """Look up a field: top-level -> mapped category dicts -> any nested dict."""
    if field in data and field not in NEST_KEYS:
        return data[field]
    for key in NEST_KEYS:
        sub = data.get(key)
        if isinstance(sub, dict) and field in sub:
            return sub[field]
    for k, v in data.items():
        if k in SKIP_KEYS:
            continue
        if isinstance(v, dict) and field in v:
            return v[field]
    return None


def is_skippable(value, field_name: str, uncertain: set) -> bool:
    if field_name in uncertain:
        return True
    if value is None:
        return True
    if isinstance(value, str):
        if not value.strip():
            return True
        if UNCERTAIN_MARK in value:
            return True
    return False


def format_value(value) -> str:
    """Render a field value as markdown."""
    if isinstance(value, list):
        if all(isinstance(x, dict) for x in value) and value:
            lines = []
            for d in value:
                parts = [f"**{k}**: {v}" for k, v in d.items()]
                lines.append("  - " + " | ".join(parts))
            return "\n" + "\n".join(lines)
        flat = [str(x) for x in value]
        joined = ", ".join(flat)
        if len(joined) <= 80:
            return joined
        return "\n" + "\n".join(f"  - {x}" for x in flat)
    if isinstance(value, dict):
        return "; ".join(f"**{k}**: {v}" for k, v in value.items())
    text = str(value).strip()
    # Long prose -> blockquote for readability.
    if len(text) > 160:
        return "\n  > " + text.replace("\n", "\n  > ")
    return text


def leading_tag(value) -> str:
    """Compact a verbose Chinese value to its leading tag for the TOC table."""
    if value is None:
        return "—"
    text = str(value).strip()
    if not text or UNCERTAIN_MARK in text:
        return "—"
    # Cut at the first opener / separator to keep just the leading tag.
    for sep in ["（", "(", "；", ";", "，", ",", "：", ":", "——", " — ", "/", "\n"]:
        idx = text.find(sep)
        if idx > 0:
            text = text[:idx]
            break
    text = text.strip().rstrip("（(")
    # Hard length cap so the TOC table stays readable.
    if len(text) > 28:
        text = text[:27] + "…"
    # Escape table-breaking pipes.
    text = text.replace("|", "/")
    return text if text else "—"


def main() -> None:
    outline = yaml.safe_load(OUTLINE.open(encoding="utf-8"))
    fields_doc = yaml.safe_load(FIELDS.open(encoding="utf-8"))
    topic = outline.get("topic", "Research Report")
    items = outline["items"]

    # Ordered category -> [(field_name, description)] from fields.yaml.
    categories = []
    for cat_name, body in fields_doc["field_categories"].items():
        flds = [(f["name"], f.get("description", "")) for f in body.get("fields", [])]
        categories.append((cat_name, body.get("description", ""), flds))
    uncertain_section = [(f["name"], f.get("description", "")) for f in fields_doc.get("uncertain", [])]

    # Load every item's JSON (None if missing).
    loaded = []
    for it in items:
        path = RESULTS / f"{slugify(it['name'])}.json"
        data = load_json(path) if path.exists() else None
        loaded.append((it, data, path))

    n_total = len(items)
    n_found = sum(1 for _, d, _ in loaded if d is not None)

    out = []
    out.append(f"# {topic}")
    out.append("")
    out.append("> 调研框架：**bridge-to-agent-skill-learning** — 将经典 NLP 跨任务迁移谱系"
               "（A/B/C/D/G）作为现代 LLM 智能体经验/技能复用（E/F）的基础，服务于 "
               "mediator-coevo / OPD 的 agent-skill-learning 方法设计。")
    out.append("")
    out.append(f"**条目数**：{n_found}/{n_total} ｜ **字段框架**："
               f"{sum(len(c[2]) for c in categories)} 个定义字段 + {len(uncertain_section)} 个保留字段 ｜ "
               f"**生成自** `results/*.json`（每条目独立深度调研，Opus 4.8 + firecrawl/exa/文献检索 MCP，"
               f"已交叉核验）。不确定值已跳过。")
    out.append("")

    # ---- Table of contents, grouped by cluster ----
    out.append("## 目录")
    out.append("")
    by_cluster = {c: [] for c in CLUSTER_ORDER}
    for it, data, _ in loaded:
        by_cluster.setdefault(it["cluster"], []).append((it, data))

    idx = 0
    for cl in CLUSTER_ORDER:
        group = by_cluster.get(cl, [])
        if not group:
            continue
        out.append(f"### {CLUSTER_LABEL.get(cl, cl)}")
        out.append("")
        header = "| # | 条目 | " + " | ".join(h for _, h in TOC_FIELDS) + " |"
        sep = "|---|---|" + "|".join(["---"] * len(TOC_FIELDS)) + "|"
        out.append(header)
        out.append(sep)
        for it, data in group:
            idx += 1
            name = it["name"]
            link = f"[{name}](#{anchor(name)})"
            cells = []
            for fld, _ in TOC_FIELDS:
                if data is None:
                    cells.append("—")
                    continue
                val = find_field(data, fld)
                cells.append(leading_tag(val))
            out.append(f"| {idx} | {it['id']} · {link} | " + " | ".join(cells) + " |")
        out.append("")

    # ---- Detailed entries, grouped by cluster ----
    out.append("---")
    out.append("")
    out.append("## 详细条目")
    out.append("")

    for cl in CLUSTER_ORDER:
        group = by_cluster.get(cl, [])
        if not group:
            continue
        out.append(f"# {CLUSTER_LABEL.get(cl, cl)}")
        out.append("")
        for it, data in group:
            name = it["name"]
            out.append(f"## {name}")
            out.append("")
            out.append(f"`{it['id']}` ｜ cluster **{it['cluster']}** ｜ "
                       f"[source]({it.get('url','')})")
            out.append("")
            if data is None:
                out.append("*（结果文件缺失）*")
                out.append("")
                continue
            uncertain = set(data.get("uncertain", []) or [])

            # Defined-field categories.
            for cat_name, _cat_desc, flds in categories:
                rendered = []
                for fname, _fdesc in flds:
                    val = find_field(data, fname)
                    if is_skippable(val, fname, uncertain):
                        continue
                    rendered.append(f"- **{fname}**: {format_value(val)}")
                if rendered:
                    out.append(f"### {cat_name}")
                    out.append("")
                    out.extend(rendered)
                    out.append("")

            # Uncertain-section fields that DID get filled (not in the uncertain array).
            extra = []
            for fname, _fdesc in uncertain_section:
                val = find_field(data, fname)
                if is_skippable(val, fname, uncertain):
                    continue
                extra.append(f"- **{fname}**: {format_value(val)}")
            if extra:
                out.append("### 补充信息 (reproducibility / open-source / cost / citations 等)")
                out.append("")
                out.extend(extra)
                out.append("")

            # Any extra fields present in JSON but not defined in fields.yaml.
            defined = {f for _, _, flds in categories for f, _ in flds}
            defined |= {f for f, _ in uncertain_section}
            others = []
            for k, v in data.items():
                if k in SKIP_KEYS or k in NEST_KEYS:
                    continue
                if k in defined:
                    continue
                if is_skippable(v, k, uncertain):
                    continue
                others.append(f"- **{k}**: {format_value(v)}")
            if others:
                out.append("### 其他信息")
                out.append("")
                out.extend(others)
                out.append("")

            # Uncertain / skipped fields, one per line.
            if uncertain:
                out.append("### 不确定字段（已跳过）")
                out.append("")
                for fld in sorted(uncertain):
                    out.append(f"- {fld}")
                out.append("")

            out.append("---")
            out.append("")

    REPORT.write_text("\n".join(out), encoding="utf-8")
    print(f"Report written to {REPORT}")
    print(f"  items: {n_found}/{n_total}")
    print(f"  lines: {len(out)}")


if __name__ == "__main__":
    main()
