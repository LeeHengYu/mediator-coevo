# LangChain Graph Harness Update 0003

Source training run:
`data/experiments/20260708-003929-hl-train-langchain-graph-batch3`

Base harness:
`data/experiments/20260707-225000-hl-train-langchain-graph-batch2/harnesses/update_0002`

Targeted failure mode:
- Batch 3 recovered repeated failures once task-specific avoid/recheck artifacts existed, but first-repeat WRA campus and first-transfer HWPX safety still failed on verifier-critical details.
- In exact-repeat contexts, `update_0002` promoted exact-task summaries and debug hints but still let a non-exact related-node success summary outrank the exact-task run outcome.
- This meant repeated tasks could consume one artifact slot on sibling or cross-family context before exhausting the current task node's own evidence.

Harness change:
- Keep `update_0001` outcome-based channel calibration and `update_0002` exact-task summary/debug promotion.
- Adjust exact-repeat ranking so all same-task artifacts outrank non-exact related-node success artifacts.
- Preserve summary/debug priority within exact-task artifacts, then use exact-task run outcomes before sibling summaries.

Expected validation effect:
- Repeated tasks should receive a more internally consistent exact-task context bundle.
- Related-node artifacts can still transfer when there is remaining budget, but should not displace current-node evidence.
- This should reduce noisy sibling or cross-family context without hard-filtering the graph.

Checks run:
- `uv run ruff check .`
- `uv run mypy src tests`

Next validation command:

```bash
uv run medcoevo run \
  --harness-dir data/experiments/20260708-003929-hl-train-langchain-graph-batch3/harnesses/update_0003 \
  --family Weighted-Risk-Assessment \
  --family HWPX-Document-Automation \
  --seed 42 \
  --split validation \
  --iterations 8 \
  --condition learned_mediator \
  --skill-updates none \
  --diffusion-enabled \
  --diffusion-policy langchain_graph \
  --run-id hl-val-langchain-graph-batch3
```
