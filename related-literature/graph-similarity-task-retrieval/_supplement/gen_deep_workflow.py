#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the /research-deep Workflow script from outline.yaml.

Produces a self-contained JS workflow that runs one Opus-4.8 agent per item
(worker-pool of 5 → max 5 concurrent), each agent doing web research and writing
a validated JSON dossier. Avoids hand-transcribing 144 items.
"""
import json
import re
from pathlib import Path

BASE = Path("/Users/htizhang/Documents/GitHub/mediator-coevo/graph-similarity-task-retrieval")
RESULTS = BASE / "results"
FIELDS = BASE / "fields.yaml"
OUT_JS = BASE / "_supplement" / "deep_workflow.js"

SUBAREA_LABEL = {
    "A": "Task similarity & transferability estimation",
    "B": "Retrieval-based in-context example selection",
    "C": "Graph-based retrieval + LLM (GraphRAG)",
    "D": "Curriculum / skill libraries / task-graphs for agents",
    "E": "Case-based reasoning + agent memory",
    "F": "Metric / representation retrieval backbones",
}


def slug(name: str) -> str:
    s = name.replace("^", "")
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[()]", "", s)
    s = re.sub(r"[^A-Za-z0-9 _.+\-]", "", s)
    s = s.strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"_+", "_", s)
    return s


def build_info(it: dict) -> str:
    lines = [f"name: {it['name']}"]
    if it.get("aliases"):
        lines.append(f"aliases: {it['aliases']}")
    sub = it["subarea"]
    lines.append(f"subarea: {sub} — {SUBAREA_LABEL[sub]}")
    lines.append(f"year: {it.get('year', '')}")
    lines.append(f"venue: {it.get('venue', '')}")
    lines.append(f"key_idea: {it.get('key_idea', '')}")
    lines.append(f"paper_url: {it.get('paper_url', '')}")
    if it.get("needs_verification"):
        lines.append("needs_verification: true")
    return "\n".join(lines)


def main():
    import yaml
    outline = yaml.safe_load((BASE / "outline.yaml").open(encoding="utf-8"))
    topic = outline["topic"]
    raw_items = outline["items"]

    items = []
    seen = set()
    for it in raw_items:
        fn = f"{it['id']}_{slug(it['name'])}.json"
        assert fn not in seen, f"slug collision: {fn}"
        seen.add(fn)
        items.append({
            "id": it["id"],
            "name": it["name"],
            "subarea": it["subarea"],
            "output_path": str(RESULTS / fn),
            "info": build_info(it),
            "needs_verification": bool(it.get("needs_verification", False)),
        })

    items_json = json.dumps(items, ensure_ascii=False, indent=0)
    fields_path = str(FIELDS)
    results_dir = str(RESULTS)

    js = r'''export const meta = {
  name: 'research-deep-graph-similarity',
  description: 'Deep-research 144 items (1 Opus-4.8 agent each, 5 concurrent) into validated JSON dossiers',
  phases: [
    { title: 'Cluster A', detail: 'task similarity & transferability', model: 'opus' },
    { title: 'Cluster B', detail: 'retrieval-based ICL selection', model: 'opus' },
    { title: 'Cluster C', detail: 'GraphRAG / GNN+LLM', model: 'opus' },
    { title: 'Cluster D', detail: 'curriculum / skill libraries', model: 'opus' },
    { title: 'Cluster E', detail: 'CBR + agent memory', model: 'opus' },
    { title: 'Cluster F', detail: 'retrieval backbones', model: 'opus' },
  ],
}

const TOPIC = __TOPIC__
const FIELDS = __FIELDS__
const RESULTS_DIR = __RESULTS_DIR__
const ITEMS = __ITEMS__
const WORKERS = 5

function buildPrompt(it) {
  return `## 任务
调研 ${it.info}，输出结构化JSON到 ${it.output_path}

## 字段定义
读取 ${FIELDS} 获取所有字段定义

## 输出要求
1. 按fields.yaml定义的字段输出JSON
2. 不确定的字段值标注[不确定]
3. JSON末尾添加uncertain数组，列出所有不确定的字段名
4. 所有字段值必须使用中文输出（调研过程可用英文，但最终JSON值为中文）

## 输出路径
${it.output_path}

## 验证
完成JSON输出后，运行验证脚本确保字段完整覆盖：
python ~/.claude/skills/research/validate_json.py -f ${FIELDS} -j ${it.output_path}
验证通过后才算完成任务。

## 附加要求（orchestrator 追加，不替代/不改写上面的模板）
- 调研话题（topic）: ${TOPIC}
- 模型/推理：Opus 4.8，最大推理强度，先想清楚再写。
- 必须广泛使用外部检索工具。这些工具是 deferred，先用 ToolSearch 加载 schema 再调用：
  - ToolSearch "select:mcp__firecrawl__firecrawl_search,mcp__firecrawl__firecrawl_scrape"
  - ToolSearch "select:mcp__exa__web_search_exa,mcp__exa__web_fetch_exa"
  - ToolSearch "select:mcp__academic-search__search_papers,mcp__academic-search__explore_citations,mcp__academic-search__search_by_author"
  至少进行 6–10 次跨 exa/firecrawl/academic-search 的检索；用一手来源（arXiv / ACL Anthology / OpenReview / NeurIPS-ICML-ICLR proceedings / VLDB）核实 authors、year、venue，并尽量补 code_url、datasets_benchmarks、key_results。
- project_relevance 三个字段（relevance_to_task_retrieval / adaptable_components / limitations）必须结合 mediator-coevo / OPD 项目来写：一个“中介(mediator)”检索相似的先验任务/技能/案例来引导多 agent 的协同进化(coevolution)。把本方法能否、如何被移植到“按相似度检索任务/技能/案例”里讲清楚。
- cross_cutting_dimensions 里只填与本 item 子领域(${it.subarea})相关的字段；不相关的填 "[不适用]" 或留空并不要列入 uncertain。
- 若本 item 标注 needs_verification 且你无法确认论文是否真实存在 / 作者 / venue：采用 best-effort 策略，用现有部分证据尽量填写，无法确认的字段值填 "[不确定]" 并列入 uncertain 数组；绝对不要编造引用或作者。
- uncertain 数组：列出所有标了 [不确定] 的字段名（JSON 顶层键，name 即可）。
- 写文件用 Write 工具，写到精确路径：${it.output_path}
- 写完务必运行验证脚本（若 python 不可用就用 python3 运行同一脚本），确保退出码为 0、无缺失必填字段。
- 最终只回复一行文本："done ${it.id}" 或 "error ${it.id} <一句话原因>"。这行就是你的返回值。`
}

let cursor = 0
const results = new Array(ITEMS.length)

async function runOne(i) {
  const it = ITEMS[i]
  try {
    const out = await agent(buildPrompt(it), {
      label: `${it.id} ${it.name}`,
      phase: `Cluster ${it.subarea}`,
      model: 'opus',
    })
    if (out === null) return { id: it.id, name: it.name, subarea: it.subarea, status: 'failed' }
    return { id: it.id, name: it.name, subarea: it.subarea, status: 'done', reply: String(out).slice(0, 200) }
  } catch (e) {
    return { id: it.id, name: it.name, subarea: it.subarea, status: 'error', error: String(e).slice(0, 200) }
  }
}

async function worker() {
  while (true) {
    const i = cursor++
    if (i >= ITEMS.length) return
    let r = await runOne(i)
    if (r.status !== 'done') {
      const retry = await runOne(i)
      if (retry.status === 'done') r = retry
      else r = { ...r, retried: true }
    }
    results[i] = r
    const done = results.filter(Boolean).length
    log(`[${done}/${ITEMS.length}] ${r.id} ${r.name} -> ${r.status}`)
  }
}

log(`Deep research: ${ITEMS.length} items, ${WORKERS} concurrent, model=opus, results -> ${RESULTS_DIR}`)
await Promise.all(Array.from({ length: WORKERS }, () => worker()))

const all = results.filter(Boolean)
const byStatus = {}
for (const r of all) byStatus[r.status] = (byStatus[r.status] || 0) + 1
const problems = all.filter(r => r.status !== 'done').map(r => ({ id: r.id, name: r.name, status: r.status, error: r.error || null }))
return {
  total: ITEMS.length,
  completed: all.length,
  byStatus,
  problems,
}
'''

    js = js.replace("__TOPIC__", json.dumps(topic, ensure_ascii=False))
    js = js.replace("__FIELDS__", json.dumps(fields_path))
    js = js.replace("__RESULTS_DIR__", json.dumps(results_dir))
    js = js.replace("__ITEMS__", items_json)

    OUT_JS.write_text(js, encoding="utf-8")
    print(f"wrote {OUT_JS}")
    print(f"items: {len(items)}")
    print(f"sample output_path: {items[0]['output_path']}")
    print(f"sample slug check: {items[2]['id']} -> {Path(items[2]['output_path']).name}")
    print(f"js size: {len(js)} bytes")


if __name__ == "__main__":
    main()
