# update_0004: carry epoch 3 graph state without code changes

## Source evidence

- Last promoted harness: `data/experiments/HL2/train_epoch_1/harnesses/update_0001`
- Rejected candidate: `data/experiments/HL2/train_epoch_2/harnesses/update_0003`
- Rejected validation: `data/experiments/HL2/validation_update_0003`
  - Completed 8 tasks with 0 environment failures.
  - Mean reward was 0.625, below the accepted validation baseline 0.875.
- Completed training epoch 3: `data/experiments/HL2/train_epoch_3`
  - Completed 8 tasks with 0 environment failures.
  - Mean reward was 0.750.

## Candidate change

This is a state-only candidate. It keeps the accepted `update_0001` overlay code
and tests unchanged, then carries the allowed graph/audit state from epoch 3.

Overlay source:

- `overlay/src/mediated_coevo/diffusion/langchain_graph.py`
- `overlay/tests/test_langchain_graph_policy.py`

Behavior:

- No code behavior change from the latest promoted harness.
- Carry epoch 3 graph snapshots so validation can test whether the updated
  graph prior improves artifact selection.
- Carry `diffused_records.jsonl` for audit continuity only.

## Carried state

- Copied `diffusion/diffused_records.jsonl` for audit continuity.
- Copied graph snapshots under `state/diffusion/graph_snapshots/`.
- Did not copy `diffusion/artifacts/`.
- Did not copy skills, jobs, benchmark copies, or run artifacts.

## Training evidence

Epoch 3 scored failures:

- `Weighted-Risk-Assessment/weighted-campus-energy-balance-calc` iteration 0
  failed with `#NAME?` in percentile formulas.
- `HWPX-Document-Automation/hwpx-safety-audit-brief` iteration 2 failed because
  the risk-tier note needed parenthetical formatting, such as
  `High (즉시조치)`.

Later in the same epoch, repeated/sibling tasks succeeded after the graph and
artifact stream contained those corrective artifacts:

- `Weighted-Risk-Assessment/api-sla-at-risk-calc` iteration 4 succeeded.
- `Weighted-Risk-Assessment/weighted-campus-energy-balance-calc` iteration 5
  succeeded.
- `HWPX-Document-Automation/hwpx-safety-audit-brief` iteration 3 succeeded.
- Later HWPX event/training-feedback tasks also succeeded.

## Validation command

`uv run medcoevo run --harness-dir data/experiments/HL2/train_epoch_3/harnesses/update_0004 --family Weighted-Risk-Assessment --family HWPX-Document-Automation --seed 42 --split validation --condition learned_mediator --skill-updates none --diffusion-enabled --diffusion-policy langchain_graph --run-id hl-val-langchain-graph-batch3-state-update4`

## Validation status

- Decision: rejected; do not promote.
- Validation run: `data/experiments/HL2/validation_update_0004`
- Result: 8 scored tasks, 0 environment failures, mean reward 0.875,
  macro mean reward 0.667.
- Accepted baseline remains:
  `data/experiments/HL2/validation_update_0001`
  with mean reward 0.875 and macro mean reward 0.917.
- Rationale: the main mean tied the accepted baseline, but the state carry
  regressed `Weighted-Risk-Assessment/weighted-port-throughput-calc` from 1.0
  in the accepted baseline to 0.0. That per-task regression and lower macro
  mean make the result ambiguous/negative-transfer, so continuation must use
  the previous promoted harness.

## Next training command

`uv run medcoevo run --harness-dir data/experiments/HL2/train_epoch_1/harnesses/update_0001 --family Weighted-Risk-Assessment --family HWPX-Document-Automation --seed 42 --split train --condition learned_mediator --skill-updates none --diffusion-enabled --diffusion-policy langchain_graph --run-id hl-train-langchain-graph-batch4-from-update1-restart1`
