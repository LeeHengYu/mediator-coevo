# HL4 validation decision: update_0003 rejected

- decision: rejected
- candidate: `data/experiments/HL4/train_epoch_2/harnesses/update_0003`
- validation run: `data/experiments/HL4/validation_update_0003`
- accepted validation baseline: `data/experiments/HL4/validation_update_0002`

## Gate evidence

- baseline summary:
  - `mean_reward = 1.0`
  - `macro_mean_reward = 1.0`
  - `env_failure_count = 0`
  - `total_runs = 8`
- validation evidence before manual stop:
  - iteration 0: `Weighted-Risk-Assessment/weighted-cloud-reliability-calc`
    - `reward = 1.0`
    - `verifier_status = ok`
  - iteration 1: `Production-Capacity-Planning/harbor_gdpval_36_task1`
    - `reward = 0.0`
    - `verifier_status = task_failed`
    - `transfer_context_kind = diffusion`
    - `source_task_ids = ["Weighted-Risk-Assessment/weighted-cloud-reliability-calc"]`

## Reason for rejection

The accepted validation baseline is perfect (`mean_reward = 1.0`). After
iteration 1 failed with `reward = 0.0`, this validation run could no longer
finish non-regressing on the main reward summary. That is a hard regression
under `instructions.txt`, so `update_0003` is rejected for continuation.

The run was manually interrupted after the rejection was already decided to stop
further validation spend. The cancelled third Harbor task is not part of the
gate evidence and should not be used as harness-edit evidence.

## Next command

Continue from the promoted harness and latest graph channel:

```bash
uv run medcoevo run \
  --harness-ref promoted:HL4 \
  --state-ref latest-graph:HL4 \
  --publish-state-ref latest-graph:HL4 \
  --family HWPX-Document-Automation \
  --family Production-Capacity-Planning \
  --family Weighted-Risk-Assessment \
  --family 'Inventory-&-Finance-Integration' \
  --seed 42 \
  --split train \
  --condition learned_mediator \
  --skill-updates none \
  --diffusion-enabled \
  --diffusion-policy langchain_graph \
  --run-id hl4-train-langchain-graph-batch4
```
