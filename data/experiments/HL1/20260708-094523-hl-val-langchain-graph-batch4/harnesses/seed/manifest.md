# LangChain Graph Harness Update 0004

Source training run:
`data/experiments/20260708-063658-hl-train-langchain-graph-batch4`

Base harness:
`data/experiments/20260708-003929-hl-train-langchain-graph-batch3/harnesses/update_0003`

Targeted failure mode:
- Batch 4 showed that the graph/diffusion selector often produced high-signal reasons for selected artifacts, including exact avoid/recheck warnings for HWPX severity formatting and WRA percentile formulas.
- Those reasons and per-artifact channels were persisted in audit records but were not rendered into the actual diffusion context consumed by the task agent.
- The executor saw `artifact_id`, source, relation, risk, and content, but not the selector's explanation of why the artifact should be reused or avoided.

Harness change:
- Render each selected artifact's `context_channel` and `selection_reason` in the artifact block.
- Keep the change inside `src/mediated_coevo/diffusion/renderer.py`; no task-execution behavior changes.
- Preserve the existing channel grouping sections and audit metadata.

Expected validation effect:
- Reuse and avoid/recheck context should be less ambiguous to the downstream planner/executor.
- Exact failure artifacts should carry their actionable reason into the prompt, not only into `diffused_records.jsonl`.
- This should improve later recovery behavior without changing which artifacts are selected.

Checks run:
- `uv run ruff check .`
- `uv run mypy src tests`

Next validation command:

```bash
uv run medcoevo run \
  --harness-dir data/experiments/20260708-063658-hl-train-langchain-graph-batch4/harnesses/update_0004 \
  --family Weighted-Risk-Assessment \
  --family HWPX-Document-Automation \
  --seed 42 \
  --split validation \
  --iterations 8 \
  --condition learned_mediator \
  --skill-updates none \
  --diffusion-enabled \
  --diffusion-policy langchain_graph \
  --run-id hl-val-langchain-graph-batch4
```
