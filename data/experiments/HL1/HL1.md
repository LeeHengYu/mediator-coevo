# HL1: LangChain Graph Heuristic-Learning Run

Date: 2026-07-08

## Scope

This report summarizes the full offline heuristic-learning sequence for
`diffusion.policy="langchain_graph"` over:

- `Weighted-Risk-Assessment`
- `HWPX-Document-Automation`

The frozen runtime harness was updated only between completed batches. The final
test used the accepted current-batch overlay:

`data/experiments/20260708-112223-hl-train-langchain-graph-batch5/harnesses/update_0005`

That overlay is code-identical to the previously accepted `update_0003` harness
and contains:

- `src/mediated_coevo/diffusion/langchain_graph.py`
- `tests/test_langchain_graph_policy.py`

## Split Pools

The CLI split logic shuffles the 16 selected family tasks with seed `42`, then
uses 10 train tasks, 3 validation tasks, and 3 test tasks. Each 8-task run stream
is sampled with replacement from the selected pool.

### Train Pool

- `HWPX-Document-Automation/hwpx-training-feedback`
- `Weighted-Risk-Assessment/campus-budget-at-risk-calc`
- `HWPX-Document-Automation/hwpx-safety-audit-brief`
- `HWPX-Document-Automation/hwpx-supplier-contact-sheet`
- `Weighted-Risk-Assessment/weighted-hospital-bedflow-calc`
- `Weighted-Risk-Assessment/factory-output-at-risk-calc`
- `Weighted-Risk-Assessment/weighted-campus-energy-balance-calc`
- `Weighted-Risk-Assessment/api-sla-at-risk-calc`
- `HWPX-Document-Automation/hwpx-event-announcement`
- `HWPX-Document-Automation/hwpx-inventory-report`

### Validation Pool

- `Weighted-Risk-Assessment/weighted-cloud-reliability-calc`
- `Weighted-Risk-Assessment/weighted-port-throughput-calc`
- `HWPX-Document-Automation/hwpx-renewal-playbook-update`

### Test Pool

- `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc`
- `HWPX-Document-Automation/hwpx-clinic-intake-summary`
- `HWPX-Document-Automation/hwpx-project-proposal`

The final test stream was:

1. `HWPX-Document-Automation/hwpx-clinic-intake-summary`
2. `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc`
3. `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc`
4. `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc`
5. `HWPX-Document-Automation/hwpx-project-proposal`
6. `HWPX-Document-Automation/hwpx-project-proposal`
7. `HWPX-Document-Automation/hwpx-project-proposal`
8. `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc`

## Harness Updates

### Update 0001

Training source:
`data/experiments/20260707-203548-hl-train-langchain-graph-batch1`

Targeted failure mode:

- The graph/diffusion agent could select artifacts with the wrong semantic
  channel, and sometimes left useful artifact budget unused.

Change:

- Calibrated artifact channel from verifier outcome:
  successful verifier artifacts use `reuse_success`; failed verifier artifacts
  use `avoid_recheck`.
- Added fallback actionable selection to fill unused budget with eligible
  mediator summaries, debug hints, and run outcomes.

Validation:

- `data/experiments/20260707-215830-hl-val-langchain-graph-batch1`
- Mean reward `0.875`, macro `0.667`, env failures `0`.
- Accepted because validation improved over train batch 1 behavior and removed
  the env failure seen in the first training run.

### Update 0002

Training source:
`data/experiments/20260707-225000-hl-train-langchain-graph-batch2`

Targeted failure mode:

- Exact repeated tasks recovered when they saw task-specific failure evidence,
  but the harness did not reliably prioritize exact same-task mediator reports
  and debug hints.

Change:

- Promoted exact-task mediator summaries and debug hints.
- Promoted same-task failed summaries/debug hints as `avoid_recheck` evidence.
- Preserved `update_0001` channel calibration and fallback behavior.

Validation:

- `data/experiments/20260707-234638-hl-val-langchain-graph-batch2`
- Mean reward `0.875`, macro `0.917`, env failures `0`.
- Accepted because macro reward improved substantially.

### Update 0003

Training source:
`data/experiments/20260708-003929-hl-train-langchain-graph-batch3`

Targeted failure mode:

- Exact-repeat contexts could still spend one of three artifact slots on a
  non-exact related success before exhausting the current task node's own
  evidence.

Change:

- Re-ranked exact-task bundles so all same-task artifacts outrank non-exact
  related-node successes.
- Preserved summary/debug priority inside exact-task artifacts, then retained
  run outcomes before sibling summaries.

Validation:

- `data/experiments/20260708-054102-hl-val-langchain-graph-batch3`
- Mean reward `1.000`, macro `1.000`, env failures `0`.
- Accepted as the first perfect validation.

### Update 0004

Training source:
`data/experiments/20260708-063658-hl-train-langchain-graph-batch4`

Targeted failure mode:

- Hypothesis: executor-visible context might improve if each selected artifact
  exposed its selector channel/reason in the rendered prompt.

Change:

- Added rendered `context_channel` and `selection_reason` metadata to selected
  artifact blocks.

Validation:

- `data/experiments/20260708-094523-hl-val-langchain-graph-batch4`
- Mean reward `0.750`, macro `0.583`, env failures `0`.
- Rejected. WRA validation regressed immediately:
  `weighted-port-throughput-calc` failed and the first
  `weighted-cloud-reliability-calc` failed. The selected-artifact ledger still
  looked reasonable, so the regression was attributed to noisier rendered
  executor context rather than graph/node selection.

### Update 0005

Training source used for analysis:
`data/experiments/20260708-112237-hl-train-langchain-graph-batch5-rerun`

Overlay storage:
`data/experiments/20260708-112223-hl-train-langchain-graph-batch5/harnesses/update_0005`

Targeted failure mode:

- The original batch 5 run had one WRA exact-repeat failure even though the
  exact same-task success summary, debug hint, and run outcome were selected.
- The rerun used the same task stream, config, and active `update_0003` harness,
  but the WRA exact repeat succeeded. This indicates the original failure was
  executor stochasticity/adherence rather than a reliable harness-selection
  failure.
- No new stable graph/diffusion failure mode was identified after the rerun.

Change:

- No new harness code change.
- Copied the latest accepted `update_0003` harness into the current batch as
  `update_0005`, preserving the accepted selector behavior.

Validation:

- `data/experiments/20260708-121100-hl-val-langchain-graph-batch5`
- Mean reward `1.000`, macro `1.000`, env failures `0`.
- Accepted as the final carried-forward harness.

## Reward Trajectory

| Stage | Run | Mean | Macro | Env failures | Tokens | Decision |
|---|---|---:|---:|---:|---:|---|
| Train 1 | `20260707-203548-hl-train-langchain-graph-batch1` | 0.714 | 0.600 | 1 | 2,165,036 | learned `update_0001` |
| Validation 1 | `20260707-215830-hl-val-langchain-graph-batch1` | 0.875 | 0.667 | 0 | 2,495,681 | accept `update_0001` |
| Train 2 | `20260707-225000-hl-train-langchain-graph-batch2` | 0.750 | 0.800 | 0 | 3,142,624 | learned `update_0002` |
| Validation 2 | `20260707-234638-hl-val-langchain-graph-batch2` | 0.875 | 0.917 | 0 | 3,296,428 | accept `update_0002` |
| Train 3 | `20260708-003929-hl-train-langchain-graph-batch3` | 0.750 | 0.800 | 0 | 3,090,476 | learned `update_0003` |
| Validation 3 | `20260708-054102-hl-val-langchain-graph-batch3` | 1.000 | 1.000 | 0 | 2,981,388 | accept `update_0003` |
| Train 4 | `20260708-063658-hl-train-langchain-graph-batch4` | 0.750 | 0.800 | 0 | 4,257,074 | learned `update_0004` |
| Validation 4 | `20260708-094523-hl-val-langchain-graph-batch4` | 0.750 | 0.583 | 0 | 3,124,046 | reject `update_0004` |
| Train 5 | `20260708-112237-hl-train-langchain-graph-batch5-rerun` | 0.875 | 0.900 | 0 | 2,108,849 | copied `update_0003` as `update_0005` |
| Validation 5 | `20260708-121100-hl-val-langchain-graph-batch5` | 1.000 | 1.000 | 0 | 1,951,692 | accept `update_0005` |
| Test | `20260708-144319-hl-test-langchain-graph` | 0.875 | 0.889 | 0 | 2,198,314 | final held-out result |

The original batch 5 run remains available at
`data/experiments/20260708-112223-hl-train-langchain-graph-batch5`. It scored
mean `0.750`, macro `0.800`, env failures `0`; the only reward difference from
the rerun was the WRA campus exact repeat at iteration 5, which failed in the
original and passed in the rerun under the same active harness and selected
artifact pattern.

## Final Test Result

Test run:
`data/experiments/20260708-144319-hl-test-langchain-graph`

Harness:
`data/experiments/20260708-112223-hl-train-langchain-graph-batch5/harnesses/update_0005`

Aggregate:

- Runs: `8`
- Mean reward: `0.875`
- Macro mean reward: `0.889`
- Env failures: `0`
- Judge mean: `0.709`
- Total tokens: `2,198,314`

Per task:

| Task | Runs | Mean reward | Notes |
|---|---:|---:|---|
| `HWPX-Document-Automation/hwpx-clinic-intake-summary` | 1 | 1.000 | First HWPX test task succeeded. |
| `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc` | 4 | 1.000 | All repeated spreadsheet test tasks succeeded. |
| `HWPX-Document-Automation/hwpx-project-proposal` | 3 | 0.667 | First two attempts succeeded; third failed. |

Iteration-level rewards:

| Iteration | Task | Reward | Judge |
|---:|---|---:|---:|
| 0 | `HWPX-Document-Automation/hwpx-clinic-intake-summary` | 1.0 | 0.820 |
| 1 | `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc` | 1.0 | 0.730 |
| 2 | `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc` | 1.0 | 0.675 |
| 3 | `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc` | 1.0 | 0.780 |
| 4 | `HWPX-Document-Automation/hwpx-project-proposal` | 1.0 | 0.865 |
| 5 | `HWPX-Document-Automation/hwpx-project-proposal` | 1.0 | 0.720 |
| 6 | `HWPX-Document-Automation/hwpx-project-proposal` | 0.0 | 0.175 |
| 7 | `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc` | 1.0 | 0.910 |

## Test Failure Analysis

The only held-out test failure was:

`HWPX-Document-Automation/hwpx-project-proposal`, iteration `6`.

Verifier failure:

- `test_modified_paragraphs_do_not_keep_layout_cache` failed.
- The output document retained an `<hp:linesegarray>` element inside at least
  one modified paragraph.
- The task had already selected successful same-task HWPX project-proposal
  artifacts that explicitly mentioned removing `<hp:lineSegArray>`, including
  exact same-task mediator summaries and debug hints.

Interpretation:

- This was not a graph/node assignment failure.
- It was not an obvious artifact-selection failure; exact same-task successful
  HWPX proposal artifacts were selected.
- The remaining issue is executor adherence under repeated HWPX modifications:
  even with the right context, one run failed to remove stale layout cache from
  every modified paragraph.

## Diffusion Behavior Observed In Test

Positive transfer:

- WRA hospital-capacity repeats selected exact same-task spreadsheet artifacts
  and achieved `4/4` verifier success.
- HWPX project proposal selected clinic-intake HWPX artifacts for the first
  proposal task, then exact project-proposal artifacts for repeats.
- The graph shape biased artifact choice without hard-filtering: HWPX-to-HWPX
  and WRA-to-WRA evidence dominated once available.

Residual risk:

- Exact HWPX summaries/debug hints can state the correct layout-cache rule, but
  the executor may still miss one modified paragraph.
- Further harness learning should focus on making failure/success artifacts more
  operationally checkable, not on broadening the selector. For example, future
  artifact summaries could explicitly encode "for every modified `hp:p`, remove
  all child `hp:lineSegArray`/`hp:linesegarray` nodes" as a checklist item when
  HWPX traces mention layout-cache tests.

## Final Assessment

The heuristic-learning loop improved validation from an unstable base to a
stable accepted harness:

- Validation progressed from `0.875 / 0.667` after `update_0001` to perfect
  `1.000 / 1.000` after `update_0003`.
- A speculative rendering change in `update_0004` regressed validation and was
  correctly rejected.
- The final held-out test scored `0.875` mean and `0.889` macro with zero env
  failures.

The final harness is useful for same-task reuse and repeated-task recovery, but
held-out HWPX failures show that selected artifact content still needs stronger
operation-level specificity to prevent executor misses on document layout-cache
cleanup.
