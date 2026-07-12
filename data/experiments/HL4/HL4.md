# HL4 Final Handoff

Status: complete as of 2026-07-12 02:18 UTC+8.

`update_0005` passed a fresh, complete validation gate, is promoted on the
HL4 `promoted_harness` channel, and completed the approved final test. The
canonical batch-5 layout is now entirely under this directory.

## Retained Artifacts

- Training: `data/experiments/HL4/train_epoch_5`
- Promoted candidate: `data/experiments/HL4/train_epoch_5/harnesses/update_0005`
- Final validation rerun: `data/experiments/HL4/validation_update_0005`
- Final test: `data/experiments/HL4/test_final`
- Frozen final-test stream: `data/experiments/HL4/test_final/task_manifest.json`
- Promoted harness channel: `data/experiments/HL4/channels/promoted_harness.json`
- Graph channel: `data/experiments/HL4/channels/graph_state.json`
- Final promotion record: `data/experiments/HL4/promotions/20260712-014217-556278-promoted-harness.json`

The three incomplete batch-5 validation attempts, the invalidated/stalled
training placeholders, the superseded promotion record, and generated tool
caches were removed. The `reviews/`, `bundles/`, and previous completed
epoch/validation artifacts remain as provenance.

## Results

| Stage | Run | Mean reward | Macro mean | Scored | Environment failures |
| --- | --- | ---: | ---: | ---: | ---: |
| Training batch 5 | `train_epoch_5` | 0.375 | 0.375 | 8/8 | 0 |
| Fresh validation | `validation_update_0005` | 1.0 | 1.0 | 8/8 | 0 |
| Final test | `test_final` | 1.0 | 1.0 | 8/8 | 0 |

The validation and test metrics above are the top-level reward metrics in each
`summary.json`. They are independent streams and are promotion/test evidence;
the lower training mean is not a final-test regression.

## Promoted Change

The promoted overlay changes:

- `overlay/src/mediated_coevo/diffusion/langchain_graph.py`
- `overlay/tests/test_langchain_graph_policy.py`

It retains the batch-5 selection fix: after an empty agent selection, no more
than one same-family failure artifact can be selected from a failed source
task. It also hardens the live provider path by treating an agent timeout,
worker exception, or malformed structured output as a soft failure instead of
aborting the run. The focused overlay coverage includes timeout success/expiry,
worker exception handling, Python-dict-style output parsing, and rejection of
non-object text.

## Gate History

The first three batch-5 validation attempts did not produce complete
8-task summaries, so they were not promotion evidence. The retained rerun
(`validation_update_0005`) is the first complete fresh validation after the
provider timeout/parsing hardening. Its successful result promoted
`update_0005`; `test_final` then completed with the same top-level score and
zero environment failures.

## Operational State

- `promoted:HL4` resolves to `train_epoch_5/harnesses/update_0005`.
- `latest-graph:HL4` resolves to bundle
  `0aa32962e37bae99ce557a0cb805b0ce51419d6d2821835d81ba1be36a416471`.
- Graph snapshot `run_id` values retain the original timestamped identifier;
  this is immutable historical identity, not a live filesystem reference.

Further HL4 work should start from the promoted harness and graph channels,
not from any removed partial rerun.
