# HL2: LangChain Graph Heuristic-Learning Run

Date: 2026-07-09

## Scope

This report summarizes the second offline heuristic-learning sequence for
`diffusion.policy="langchain_graph"` over:

- `Weighted-Risk-Assessment`
- `HWPX-Document-Automation`

The frozen runtime harness was updated only between completed batches. Validation
was used only as a promotion gate; rejected candidates were not used as the
starting harness for later training or the final test.

The final held-out test used the last promoted harness:

`data/experiments/HL2/train_epoch_1/harnesses/update_0001`

That overlay contains:

- `src/mediated_coevo/diffusion/langchain_graph.py`
- `tests/test_langchain_graph_policy.py`

The later `update_0002`, `update_0003`, `update_0004`, and `update_0005`
candidates were rejected by validation evidence or invalid validation attempts,
so none of them were promoted for the final test.

## Effective Artifact Selection

Only the last effective run for each training, validation, and test round is used
in the analysis. Interrupted runs, environment-failure runs, and superseded
duplicates are excluded from the reward trajectory.

Effective artifacts:

- Training 1: `data/experiments/HL2/train_epoch_1`
- Validation 1: `data/experiments/HL2/validation_update_0001`
- Training 2: `data/experiments/HL2/train_epoch_2`
- Validation 2: `data/experiments/HL2/validation_update_0002`
- Validation 3: `data/experiments/HL2/validation_update_0003`
- Training 3: `data/experiments/HL2/train_epoch_3`
- Validation 4: `data/experiments/HL2/validation_update_0004`
- Training 4: `data/experiments/HL2/train_epoch_4`
- Training 5: `data/experiments/HL2/train_epoch_5`
- Test: `data/experiments/HL2/test_final`

The late training artifacts have one ordering wrinkle: `batch4` and `batch5`
were both completed from the same last-promoted `update_0001` harness, and their
wall-clock execution overlapped. Since neither path used a newly promoted
candidate, the overlap does not change the final harness lineage. The analysis
keeps the later completed `batch4` artifact because it is the one that stores
`update_0005`, and keeps the completed `batch5` artifact as the fifth training
epoch. Earlier or partial duplicates are listed in the cleanup section.

## Split Pools

The CLI split logic shuffles the selected family tasks with seed `42`, then uses
train, validation, and test pools. Each 8-task run stream is sampled with
replacement from the selected pool.

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

## Final Training Graph

The final training graph is the latest runtime snapshot from
`data/experiments/HL2/train_epoch_5`:

`data/experiments/HL2/train_epoch_5/diffusion/graph_snapshots/66685250fc994abcbb0b5af2a987a8cb.json`

This is the iteration 7 snapshot, created at `2026-07-09T12:08:43`. It contains
5 task nodes and 17 directed transfer-prior edge records. Because
`train_epoch_5` used the promoted `update_0001` harness as its active harness,
the graph includes the carried graph prior from that promoted state plus the
updates produced during the fifth training epoch.

The epoch-5 task order was:

```text
weighted-campus-energy-balance-calc
-> hwpx-training-feedback
-> hwpx-safety-audit-brief
-> hwpx-safety-audit-brief
-> api-sla-at-risk-calc
-> weighted-campus-energy-balance-calc
-> hwpx-event-announcement
-> hwpx-training-feedback
```

The final graph separates into two disconnected clusters.

WRA spreadsheet cluster:

- `weighted-campus-energy-balance-calc`
- `api-sla-at-risk-calc`

This cluster contains direct self-transfer edges with weight `1.0`, a
`weighted-campus-energy-balance-calc -> api-sla-at-risk-calc` edge with weight
`0.9`, and three `api-sla-at-risk-calc -> weighted-campus-energy-balance-calc`
edges with weight `0.95`. The edge reasons focus on shared spreadsheet formula
structure, lookup ranges, weighted means, percentile formulas, and the
`PERCENTILE.INC` / `_xlfn` formula-compatibility failure mode.

HWPX document cluster:

- `hwpx-training-feedback`
- `hwpx-safety-audit-brief`
- `hwpx-event-announcement`

This cluster contains self-transfer edges for repeated HWPX tasks, bidirectional
transfer between `hwpx-training-feedback` and `hwpx-safety-audit-brief`, and
incoming transfer into `hwpx-event-announcement` from both prior HWPX nodes.
The edge reasons focus on HWPX XML package manipulation, Korean text handling,
placeholder replacement, zip reconstruction, and clearing stale layout-cache
attributes.

There are no cross-family WRA-to-HWPX edges in the final training graph. The
learned graph expresses strong within-family transfer, especially HWPX
layout/encoding reuse and WRA formula-failure reuse, but it does not connect
spreadsheet and document-editing tasks.

## Harness Updates

### Update 0001

Training source:
`data/experiments/HL2/train_epoch_1`

Targeted failure mode:

- The diffusion audit showed useful prior artifacts in the graph, but the
  diffusion agent could still return an empty artifact selection.

Change:

- Preserve explicit LangChain artifact selections.
- When no valid selection is returned, select a deterministic fallback from
  same-task history, same-node history, or incoming graph-prior artifacts.
- Prefer higher-reward artifacts and mediator summaries/debug hints.
- Route successful prior artifacts to `reuse_success`; route low-reward prior
  artifacts to `avoid_recheck`.

Validation:

- `data/experiments/HL2/validation_update_0001`
- Mean reward `0.875`, macro `0.917`, env failures `0`.
- Accepted. This became the only promoted harness for the remaining HL2 run.

### Update 0002

Training source:
`data/experiments/HL2/train_epoch_2`

Targeted failure mode:

- The empty-selection fallback was not triggered in batch 2; the remaining issue
  was selection quality when the artifact budget was already full.
- Some full-budget selections spent slots on cross-family artifacts while
  same-task or same-family evidence was available.

Change:

- Keep explicit selections when they leave budget unused.
- When the selection budget is full, replace the weakest cross-family selected
  artifact with the strongest unselected same-task, same-node, or same-family
  artifact.

Validation:

- `data/experiments/HL2/validation_update_0002`
- Mean reward `0.875`, macro `0.667`, env failures `0`.
- Rejected. Mean reward tied the accepted baseline, but macro reward regressed
  from `0.917` to `0.667`, and the targeted replacement path did not justify the
  regression.

### Update 0003

Source evidence:

- Last promoted harness:
  `data/experiments/HL2/train_epoch_1/harnesses/update_0001`
- Prior completed training:
  `data/experiments/HL2/train_epoch_2`

Targeted failure mode:

- Provider errors and hangs in graph/diffusion agent calls could stop a batch
  before the runtime reached completed training evidence.

Change:

- Keep the `update_0001` empty-selection fallback.
- If the graph agent fails, create a deterministic graph decision for the
  current task node and continue.
- If the diffusion agent fails, return an empty diffusion decision so the
  deterministic fallback selection can run.
- Bound graph and diffusion calls with a timeout.

Validation:

- `data/experiments/HL2/validation_update_0003`
- Mean reward `0.625`, macro `0.472`, env failures `0`.
- Rejected. The completed validation was materially below the accepted
  `update_0001` baseline.

### Update 0004

Training source:
`data/experiments/HL2/train_epoch_3`

Targeted failure mode:

- Epoch 3 showed later same-task and sibling-task recovery after corrective
  artifacts entered the graph. The candidate tested whether carrying only the
  allowed graph/audit state from epoch 3 improved validation.

Change:

- No code behavior change from `update_0001`.
- Carry epoch 3 graph snapshots and `diffused_records.jsonl`.
- Do not copy diffusion artifacts, jobs, benchmarks, or skills.

Validation:

- `data/experiments/HL2/validation_update_0004`
- Mean reward `0.875`, macro `0.667`, env failures `0`.
- Rejected. Mean tied the accepted baseline, but macro reward regressed and
  `Weighted-Risk-Assessment/weighted-port-throughput-calc` regressed from `1.0`
  in the accepted baseline to `0.0`.

### Update 0005

Training source:
`data/experiments/HL2/train_epoch_4`

Targeted failure mode:

- Later training/provider-run evidence exposed timeout and Harbor invalid-run
  behavior. The candidate isolated timeout/error fallback behavior while keeping
  the accepted `update_0001` graph state. Validation was used only to decide
  whether this candidate could be promoted.

Change:

- Bound graph-agent and diffusion-agent calls with `_AGENT_TIMEOUT_SEC`.
- If the graph agent times out or raises, reuse a deterministic task node.
- If the diffusion agent times out or raises, return an empty diffusion decision
  so the existing deterministic fallback selection can run.
- Preserve the fixed-graph validation/test path by applying the same diffusion
  timeout fallback in `select_with_fixed_graph`.

Validation:

- Validation attempts did not produce a valid promotion artifact.
- `20260709-181903-hl-val-langchain-graph-batch5-timeout-update5` stopped after
  a task recorded `verifier_status=env_failure`.
- `20260709-191344-hl-val-langchain-graph-batch5-timeout-update5-retry1` stopped
  after another `env_failure`.
- `20260709-212200-hl-val-langchain-graph-batch5-timeout-update5-retry2` exited
  without `summary.json` and only 2/8 finalized metric rows.
- Rejected. Validation did not complete the 8-task gate, so `update_0005` was
  not promoted.

## Reward Trajectory

| Stage | Run | Mean | Macro | Env failures | Tokens | Decision |
|---|---|---:|---:|---:|---:|---|
| Train 1 | `data/experiments/HL2/train_epoch_1` | 0.625 | 0.600 | 0 | 3,188,301 | learned `update_0001` |
| Validation 1 | `data/experiments/HL2/validation_update_0001` | 0.875 | 0.917 | 0 | 1,911,158 | accept `update_0001`; accepted baseline |
| Train 2 | `data/experiments/HL2/train_epoch_2` | 0.625 | 0.600 | 0 | 2,705,906 | learned `update_0002`; later `update_0003` source evidence |
| Validation 2 | `data/experiments/HL2/validation_update_0002` | 0.875 | 0.667 | 0 | 2,598,796 | reject `update_0002` |
| Validation 3 | `data/experiments/HL2/validation_update_0003` | 0.625 | 0.472 | 0 | 2,428,669 | reject `update_0003` |
| Train 3 | `data/experiments/HL2/train_epoch_3` | 0.750 | 0.800 | 0 | 2,869,574 | learned state-only `update_0004` |
| Validation 4 | `data/experiments/HL2/validation_update_0004` | 0.875 | 0.667 | 0 | 2,886,572 | reject `update_0004` |
| Train 4 | `data/experiments/HL2/train_epoch_4` | 0.875 | 0.900 | 0 | 2,397,458 | learned `update_0005` |
| Train 5 | `data/experiments/HL2/train_epoch_5` | 0.875 | 0.900 | 0 | 2,596,250 | fifth completed training epoch; no promotion |
| Validation 5 | `update_0005` validation attempts | n/a | n/a | invalid | n/a | reject `update_0005`; no completed 8-task validation |
| Test | `data/experiments/HL2/test_final` | 0.750 | 0.833 | 0 | 2,505,595 | final held-out result using `update_0001` |

## Final Test Result

Test run:
`data/experiments/HL2/test_final`

Harness:
`data/experiments/HL2/train_epoch_1/harnesses/update_0001`

Aggregate:

- Runs: `8`
- Scored: `8`
- Mean reward: `0.750`
- Macro mean reward: `0.833`
- Env failures: `0`
- Judge mean: `0.534`
- Judge macro mean: `0.592`
- Total tokens: `2,505,595`

Per task:

| Task | Runs | Mean reward | Notes |
|---|---:|---:|---|
| `HWPX-Document-Automation/hwpx-clinic-intake-summary` | 1 | 1.000 | First HWPX test task succeeded without prior diffusion context. |
| `HWPX-Document-Automation/hwpx-project-proposal` | 3 | 1.000 | All repeated HWPX proposal tasks succeeded. |
| `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc` | 4 | 0.500 | First two attempts failed; later two attempts succeeded after more same-task evidence accumulated. |

Iteration-level rewards:

| Iteration | Task | Reward | Judge |
|---:|---|---:|---:|
| 0 | `HWPX-Document-Automation/hwpx-clinic-intake-summary` | 1.0 | 0.760 |
| 1 | `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc` | 0.0 | 0.025 |
| 2 | `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc` | 0.0 | 0.165 |
| 3 | `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc` | 1.0 | 0.845 |
| 4 | `HWPX-Document-Automation/hwpx-project-proposal` | 1.0 | 0.000 |
| 5 | `HWPX-Document-Automation/hwpx-project-proposal` | 1.0 | 0.835 |
| 6 | `HWPX-Document-Automation/hwpx-project-proposal` | 1.0 | 0.810 |
| 7 | `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc` | 1.0 | 0.835 |

The judge score at iteration 4 disagreed with the verifier, but the main reward
contract is verifier reward. The task is counted as successful by the experiment
summary.

## Test Failure Analysis

The held-out test failures were:

- `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc`, iteration `1`
- `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc`, iteration `2`

Iteration 1 had cross-task prior available but selected and rendered no diffusion
artifacts. Iteration 2 selected and rendered 3 artifacts, including 200
same-task prior tokens and 498 transfer tokens, but still failed. Iterations 3
and 7 then succeeded with diffusion context rendered, indicating that the
accepted harness can support repeated-task recovery but does not guarantee early
success on the hard WRA hospital-capacity task.

The HWPX held-out tasks were stable in this run:

- `hwpx-clinic-intake-summary` passed on the seed attempt.
- `hwpx-project-proposal` passed all three repeated attempts, including the two
  exact-repeat attempts with same-task prior tokens.

## Diffusion Behavior Observed In Test

Positive transfer:

- HWPX-to-HWPX transfer worked for `hwpx-project-proposal`; the first proposal
  task had no same-task prior but still selected/rendered HWPX-context artifacts
  and passed.
- Later WRA hospital-capacity repeats recovered after the test stream contained
  same-task failure and success evidence.
- The accepted `update_0001` deterministic fallback remains useful for avoiding
  empty context when the diffusion agent returns no usable selection.

Residual risk:

- The hard WRA hospital-capacity task still failed twice before same-task
  evidence stabilized the repeated attempts.
- `update_0002` and `update_0004` showed that selection rebalancing and graph
  state carry can preserve aggregate mean while regressing macro/per-task
  behavior. The validation gate correctly rejected those candidates.
- Timeout/provider fallback candidates need a clean 8-task validation before
  they can be trusted. `update_0005` did not meet that gate.

## Final Assessment

HL2 completed five training epochs, validation gates for the candidate harnesses,
and one final held-out test batch.

The only promoted harness was `update_0001`. It improved the empty-selection
case and remained the safest validated state. Every later candidate either
regressed validation macro/per-task behavior or failed to produce valid
validation evidence.

The final held-out result was lower than HL1: mean `0.750` and macro `0.833`
with zero environment failures. The main remaining failure mode is hard WRA
spreadsheet recovery before enough exact same-task evidence exists. HWPX
document-editing transfer was stable in the held-out run.

## Dropped Directories

The following directories were excluded from the effective analysis and are safe
to delete because they were interrupted, invalid, or superseded by a later
effective batch:

- `data/experiments/20260708-225729-hl-train-langchain-graph-batch2`
- `data/experiments/20260708-235422-hl-val-langchain-graph-batch2`
- `data/experiments/20260709-044710-hl-train-langchain-graph-batch3`
- `data/experiments/20260709-044755-hl-train-langchain-graph-batch3-retry1`
- `data/experiments/20260709-104206-hl-train-langchain-graph-batch4-from-update1-restart1`
- `data/experiments/20260709-121548-hl-train-langchain-graph-batch5-from-update1-restart1`
- `data/experiments/20260709-160451-hl-train-langchain-graph-batch5-from-update1-retry1`
- `data/experiments/20260709-181903-hl-val-langchain-graph-batch5-timeout-update5`
- `data/experiments/20260709-183918-hl-train-langchain-graph-batch5-from-update1-retry2`
- `data/experiments/20260709-191344-hl-val-langchain-graph-batch5-timeout-update5-retry1`
- `data/experiments/20260709-203817-hl-train-langchain-graph-batch5-from-update1-retry3`
- `data/experiments/20260709-212200-hl-val-langchain-graph-batch5-timeout-update5-retry2`
