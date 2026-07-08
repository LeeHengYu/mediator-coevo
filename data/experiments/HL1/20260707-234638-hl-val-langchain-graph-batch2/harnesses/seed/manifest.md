# LangChain Graph Harness Update 0002

Source training run:
`data/experiments/20260707-225000-hl-train-langchain-graph-batch2`

Base harness:
`data/experiments/20260707-203548-hl-train-langchain-graph-batch1/harnesses/update_0001`

Targeted failure mode:
- Batch 2 improved several transfer cases, including HWPX safety audit and WRA API SLA, but the exact-repeat HWPX safety audit at iteration 3 regressed from verifier reward 1.0 to 0.0.
- The failing repeat selected a same-task mediator summary and run outcome plus a sibling HWPX training summary, while the same-task debug hint was left as an unselected eligible candidate.
- The failure trace showed a verifier-critical literal mismatch for `High (즉시조치)`. This points to too much generic repeat context and not enough exact-task actionable context when the graph node is reused.

Harness change:
- Preserve update 0001 verifier-outcome channel overrides.
- When exact-task mediator summaries or debug hints exist for the current target task, inject them as candidates and re-rank the selected set.
- Prioritize exact-task mediator summaries and debug hints ahead of generic run outcomes and sibling-node summaries.
- Prioritize exact-task failure summaries/debug hints as `avoid_recheck` before exact-task success artifacts so recent regressions can warn the next repeat.

Expected validation effect:
- Exact-repeat validation tasks should reuse the most task-specific successful or failed evidence first.
- Generic run-outcome artifacts should no longer displace concise task-specific summaries/debug hints under a full artifact budget.
- Sibling graph edges still provide transfer priors, but should not dominate repeated-node artifact choice.

Checks run:
- `uv run ruff check .`
- `uv run mypy src tests`

Next validation command:

```bash
uv run medcoevo run \
  --harness-dir data/experiments/20260707-225000-hl-train-langchain-graph-batch2/harnesses/update_0002 \
  --family Weighted-Risk-Assessment \
  --family HWPX-Document-Automation \
  --seed 42 \
  --split validation \
  --iterations 8 \
  --condition learned_mediator \
  --skill-updates none \
  --diffusion-enabled \
  --diffusion-policy langchain_graph \
  --run-id hl-val-langchain-graph-batch2
```
