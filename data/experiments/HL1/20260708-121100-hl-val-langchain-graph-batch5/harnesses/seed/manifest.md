# LangChain Graph Harness Update 0005

Source training run:
`data/experiments/20260708-112223-hl-train-langchain-graph-batch5`

Base harness:
`data/experiments/20260708-003929-hl-train-langchain-graph-batch3/harnesses/update_0003`

Overlay source:
Copied unchanged from the last validated harness:
`data/experiments/20260708-003929-hl-train-langchain-graph-batch3/harnesses/update_0003`

Carried-forward failure mode:
- Batch 3 recovered repeated failures once task-specific avoid/recheck artifacts existed, but first-repeat WRA campus and first-transfer HWPX safety still failed on verifier-critical details.
- In exact-repeat contexts, `update_0002` promoted exact-task summaries and debug hints but still let a non-exact related-node success summary outrank the exact-task run outcome.
- This meant repeated tasks could consume one artifact slot on sibling or cross-family context before exhausting the current task node's own evidence.

Harness state:
- Keep `update_0001` outcome-based channel calibration and `update_0002` exact-task summary/debug promotion.
- Adjust exact-repeat ranking so all same-task artifacts outrank non-exact related-node success artifacts.
- Preserve summary/debug priority within exact-task artifacts, then use exact-task run outcomes before sibling summaries.
- No repo-root source files were manually edited for this current-batch overlay.

Expected continuation effect:
- Repeated tasks should receive a more internally consistent exact-task context bundle.
- Related-node artifacts can still transfer when there is remaining budget, but should not displace current-node evidence.
- This should reduce noisy sibling or cross-family context without hard-filtering the graph.

Batch 5 evidence and decision:
- Training batch 5 reward: mean `0.750`, macro `0.800`, env failures `0`.
- HWPX safety audit failed first, then recovered on the exact repeat after same-task avoid/recheck artifacts were selected.
- WRA campus energy succeeded first, then failed on the exact repeat even though the selected artifacts were the exact same-task success summary, debug hint, and run outcome.
- The WRA repeat failure mode was missing concrete lookup/range/type-guard detail in the available success artifact content, not bad node reuse or artifact ranking.
- No new code change is made in this overlay; it preserves the latest validated harness rather than adding a risky final selector heuristic.

Checks inherited:
- `uv run ruff check .`
- `uv run mypy src tests`

Next validation command if this copied overlay is used:

```bash
uv run medcoevo run \
  --harness-dir data/experiments/20260708-112223-hl-train-langchain-graph-batch5/harnesses/update_0005 \
  --family Weighted-Risk-Assessment \
  --family HWPX-Document-Automation \
  --seed 42 \
  --split validation \
  --iterations 8 \
  --condition learned_mediator \
  --skill-updates none \
  --diffusion-enabled \
  --diffusion-policy langchain_graph \
  --run-id hl-val-langchain-graph-batch5
```
