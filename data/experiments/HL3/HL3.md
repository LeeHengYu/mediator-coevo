# HL3 Loop Analysis

## Scope and conclusion

This review covers the completed campaign in `data/experiments/HL3`:

- five 8-task training batches;
- validation after the first two candidate harness edits; and
- one 8-task final test.

No experiment or source files were changed for this review.

**Conclusion:** HL3 completed operationally and the final test scored **7/8
verifier reward (`0.875`)**, with no environment failures. It produced two
small, coherent graph/diffusion harness changes, and `update_0002` is the
correct final promoted code snapshot. However, it does **not** demonstrate a
five-epoch cumulative graph-learning loop: batches 3-5 and the final test
carried only the batch-1 graph snapshots, not the latest valid training graph.
The promotion improvement is also not a paired comparison because every run
uses a newly sampled 8-task stream.

The defensible result is therefore: **a promising final evaluation of the
promoted code with a batch-1 graph prior and a fresh per-run artifact bank**.
It is not yet evidence that the graph accumulated useful state across all five
training batches.

## Run ledger

Every run completed all eight tasks with `env_failure_count = 0`.

| Phase | Verifier mean / macro | Judge mean | Unique tasks | Graph snapshots on disk | Decision |
| --- | ---: | ---: | ---: | --- | --- |
| [Train 1](data/experiments/HL3/20260710-202409-hl3-train-langchain-graph-batch1/summary.json) | 0.375 / 0.375 | 0.396 | 8 | 8 from train 1 | Created `update_0001` |
| [Validation 1](data/experiments/HL3/20260710-213325-hl3-val-langchain-graph-batch1/summary.json) | 0.875 / 0.800 | 0.664 | 5 | 8 from train 1 | Promoted `update_0001` |
| [Train 2](data/experiments/HL3/20260710-222312-hl3-train-langchain-graph-batch2/summary.json) | 0.625 / 0.625 | 0.518 | 8 | 8 from train 1 + 8 from train 2 | Created `update_0002` |
| [Validation 2](data/experiments/HL3/20260710-233938-hl3-val-langchain-graph-batch2/summary.json) | 1.000 / 1.000 | 0.751 | 5 | 8 from train 1 only | Promoted `update_0002` |
| [Train 3](data/experiments/HL3/20260711-002836-hl3-train-langchain-graph-batch3/summary.json) | 0.625 / 0.571 | 0.364 | 7 | 8 from train 1 + 8 from train 3 | No change |
| [Train 4](data/experiments/HL3/20260711-012231-hl3-train-langchain-graph-batch4/summary.json) | 0.625 / 0.643 | 0.561 | 7 | 8 from train 1 + 8 from train 4 | No change |
| [Train 5](data/experiments/HL3/20260711-021503-hl3-train-langchain-graph-batch5/summary.json) | 0.625 / 0.571 | 0.509 | 7 | 8 from train 1 + 8 from train 5 | Stop after five valid epochs |
| [Final test](data/experiments/HL3/20260711-091753-hl3-test-langchain-graph-final/summary.json) | 0.875 / 0.950 | 0.659 | 5 | 8 from train 1 only | Reported final result |

The verifier trajectory is `0.375 -> 0.625 -> 0.625 -> 0.625 -> 0.625`
across training. Judge reward is more variable (`0.396`, `0.518`, `0.364`,
`0.561`, `0.509`), so the later training batches do not show a monotonic
quality improvement.

## What the harness learned

The immutable base harness is preserved under
[`data/experiments/HL3/immutable_baseline`](data/experiments/HL3/immutable_baseline).
The final promoted harness is
[`update_0002`](data/experiments/HL3/20260710-222312-hl3-train-langchain-graph-batch2/harnesses/update_0002),
as recorded in
[`latest_promoted_harness.txt`](data/experiments/HL3/latest_promoted_harness.txt).

### `update_0001`: restrict low-reward cross-family failure reuse

`update_0001` changed only the graph policy and its focused test. It:

- accepts successful artifacts;
- accepts failed artifacts only from the same graph node or task family;
- routes an accepted failed artifact through `avoid_recheck`; and
- applies the same restriction to the fallback graph-prior selection path.

This targets the batch-1 failure mode documented in its
[`manifest`](data/experiments/HL3/20260710-202409-hl3-train-langchain-graph-batch1/harnesses/update_0001/manifest.md):
broad cross-family diffusion of low-reward failure artifacts. The candidate
test explicitly covers filtering a cross-family failure while retaining a
same-family failure in the correct channel.

### `update_0002`: constrain off-prior cross-family success reuse

`update_0002` retained `update_0001` and added one narrow guard: a successful
cross-family artifact is rejected unless it is from the target's graph node or
an incoming graph prior. Its new test verifies that a generic cross-family
success is filtered when the graph does not support it. This is a coherent
response to the batch-2 failure mode in its
[`manifest`](data/experiments/HL3/20260710-222312-hl3-train-langchain-graph-batch2/harnesses/update_0002/manifest.md).

The later no-change decisions are reasonable on their stated evidence. The
remaining batch-3 through batch-5 misses were concrete executor output errors
(workbook values/formulas, headers, or HWPX formatting), not a demonstrated
graph placement or artifact-selection defect. Creating further harness changes
would have violated the training-evidence rule.

## Carry-forward audit

### Correctly carried

- **Code overlay:** validation 1 and train 2 used `update_0001`; validation 2,
  training batches 3-5, and the final test used `update_0002`. Each later
  run's `harnesses/active_harness.json` records the two applied files.
- **Per-run artifact bank:** it was fresh. Each completed 8-task train or
  validation run emitted 24 current-run artifacts; the final test emitted 22.
  Their creation times and source-run IDs belong to the individual run, rather
  than a preceding batch.
- **Audit ledger:** later `diffused_records.jsonl` files include the 84
  batch-1 audit records plus current-run records. This is audit continuity, not
  evidence that artifacts themselves leaked across batches.

### Not cumulatively carried: the graph state

This is the principal defect in the loop.

1. Train 1 generated eight graph snapshots and stored them in
   `update_0001/state/diffusion/graph_snapshots/`.
2. Train 2 contains those eight snapshots plus eight new train-2 snapshots,
   proving that train 2 had a batch-1 graph prior.
3. `update_0002/state/` nevertheless contains only the original eight
   train-1 snapshot IDs. It does not include train-2's new graph snapshots.
4. Each later training run contains exactly the eight train-1 snapshots plus
   its own eight snapshots. None contains train-2 plus train-3 plus later
   graph history.
5. The final test contains only the original eight train-1 snapshots. It did
   not start from the graph learned through training batch 5.

There is also no `data/experiments/HL3/channels/graph_state.json`, no state
bundle, and no `state/active_state.json` in the runs. The recorded next-run
commands in the review files use `--harness-dir` only; they omit both
`--state-ref latest-graph:HL3` and
`--publish-state-ref latest-graph:HL3`.

That conflicts with the current loop contract in
[`instructions.txt`](instructions.txt): valid train batches must publish the
graph channel; later train, validation, and test runs must select it explicitly.
The current CLI makes the same distinction: `--harness-dir` applies code, while
runtime state requires `--state-dir` or `--state-ref`
([`run.py`](src/mediated_coevo/cli/run.py)).

The historical run folders show that batch-1 state was implicitly copied into
the run directories. Regardless of that legacy behavior, the evidence above
proves that it was never advanced past batch 1. Treat the later graph as a
stale prior, not a cumulative one.

## Evaluation quality

### Promotions are plausible, but not causal estimates

Validation 1 (`0.875`) established the first acceptable baseline. Validation
2 (`1.000`) then promoted `update_0002` over that baseline. Both runs were
complete and clean, so the administrative promotion decisions followed the
recorded rule.

They are not a controlled estimate of code effect, however. The runner draws a
new task stream with replacement for every invocation using a fresh random
stream seed. The fixed campaign seed holds the split pools stable, not the
eight-task sequence. The two validation streams therefore had different task
mixtures:

- validation 1 had 5 unique tasks, including three draws of
  `harbor_gdpval_36_task5`;
- validation 2 had 5 unique tasks, with 6 of 8 draws from HWPX and no
  Inventory-and-Finance task.

The change from `0.875` to `1.000` is one task on two different 8-draw
episodes. It supports keeping the safer policy guard, but it cannot isolate
the effect of `update_0002`.

### The final test is encouraging but narrow

The final test was 7/8 verifier successes. Its five unique task IDs were:

- `Weighted-Risk-Assessment/hospital-capacity-at-risk-calc`: 4 draws, 3/4
  successful;
- four other tasks: one draw each, all successful.

Thus one task accounts for half of the final test stream. The reported macro
reward of `0.950` averages the five unique task means; it is not a full,
balanced test-pool estimate. Report the result with the exact episode size and
composition: **7/8 on a fresh test stream with five unique tasks, under the
final code overlay and a batch-1 graph prior**.

## Validation and reproducibility evidence

Both candidate validation trees exist under
[`data/experiments/HL3/validation_trees`](data/experiments/HL3/validation_trees),
including the overlaid source and focused test. Cached Ruff, mypy, and pytest
outputs indicate that validation tooling was run. The campaign does not retain
human-readable command output or a `$ponytail` result, so those checks cannot
be independently re-verified from the stored evidence alone.

No `medcoevo` or Harbor process was active at inspection time.

## Recommended interpretation and next run

1. Keep `update_0002` as the promoted code harness; its behavioral scope is
   small, test-covered, and consistent with the batch evidence.
2. Do not cite HL3 as a completed cumulative graph-learning result. Cite it as
   a code-harness experiment whose graph state was stale after batch 1.
3. Before another HL campaign, make state provenance mandatory: train batch 1
   publishes `latest-graph:<campaign>`; every later train batch uses and
   republishes it; validation and final test use it but do not publish it.
   Persist `state/active_state.json` and the channel manifest for every run.
4. Add a paired candidate-versus-baseline evaluation on the same resolved task
   IDs and order. The current fresh-stream design is appropriate for training
   diversity but cannot identify a small code effect in an 8-task validation.
5. Run multiple balanced final-test episodes, or report per-family coverage,
   before making a generalization claim. Avoid a final episode where one task
   contributes half of all draws.

