# HL4 End-to-End A/B/C Test Protocol

## Goal

Evaluate the completed HL4 system as an end-to-end learned method, not merely
as a revision to `langchain_graph.py`.

The primary question is whether the final learned harness and cumulative graph
outperform no experience transfer and unguided experience transfer on held-out
task sequences.

## Conditions

| Arm | Diffusion behavior | Harness and graph state | Purpose |
| --- | --- | --- | --- |
| A: No diffusion | `diffusion.enabled=false`, `diffusion.policy=none` | No graph state is used. | Establish task-solving performance with no cross-task experience. |
| B: Random-K | `diffusion.policy=random_k` with the same `max_artifacts`, transfer-token budget, and failed-artifact cap as C. | Starts with an empty artifact bank and does not load HL4 graph state. | Test whether random artifact transfer helps. |
| C: Final HL4 | `diffusion.policy=langchain_graph`. | Use the final promoted HL4 harness and final cumulative HL4 graph bundle. | Test the complete learned system: offline harness learning plus graph-guided artifact selection. |

C is the ultimate HL agent only after the loop closes. Freeze the exact
promoted-harness overlay and graph-state bundle before any test result is read.
Record their paths and digests rather than evaluating against moving
`promoted:HL4` or `latest-graph:HL4` references.

## Valid Test Semantics

All three arms must use the same held-out task sequence in the same order.
Every sequence starts with an empty `diffusion/artifacts/` directory. The bank
then grows only from tasks that arm has already executed in that sequence.

This is intentional:

- C receives its frozen graph prior but builds a fresh causal artifact bank.
- B receives no graph prior and builds a fresh causal artifact bank.
- A receives no diffusion context.
- No artifact, trace, report, or job output may cross from one arm to another.

Validation and test graph state is read-only. Do not publish test graph state
or let test results update the HL4 campaign channel.

## Pairing Requirement

`medcoevo run --task-manifest <path>` replays the ordered task IDs from a
frozen JSON manifest exactly, including duplicate task occurrences. It does not
resample from the split pool. A manifest and `--family` are mutually exclusive.

The recovered final-HL4 stream is frozen at:

`data/experiments/HL4/test_final/task_manifest.json`

The controls below use the test-final model and budget configuration frozen at:

`data/experiments/HL4/ab_control/default.toml`

Harness overlays are process-scoped: the CLI restores the checkout after the
run exits. Run benchmark arms serially; concurrent overlay runs are not safe.

### Recovered Stream: B and C Commands

Option B uses the stored pre-HL4 baseline overlay, an empty per-run artifact
bank, and no graph state:

```zsh
PYTHONDONTWRITEBYTECODE=1 \
UV_CACHE_DIR=data/experiments/HL4/.uv-cache-ab-b-stream-01 \
uv run medcoevo run \
  --config-dir data/experiments/HL4/ab_control \
  --task-manifest data/experiments/HL4/test_final/task_manifest.json \
  --harness-dir data/experiments/HL4/baseline \
  --diffusion-enabled \
  --diffusion-policy random_k \
  --diffusion-max-artifacts 3 \
  --run-id hl4-ab-b-random-k-stream-01
```

Option C uses the final promoted harness and frozen final graph bundle:

```zsh
PYTHONDONTWRITEBYTECODE=1 \
UV_CACHE_DIR=data/experiments/HL4/.uv-cache-ab-c-stream-01 \
uv run medcoevo run \
  --config-dir data/experiments/HL4/ab_control \
  --task-manifest data/experiments/HL4/test_final/task_manifest.json \
  --harness-dir data/experiments/HL4/train_epoch_5/harnesses/update_0005 \
  --state-dir data/experiments/HL4/bundles/0aa32962e37bae99ce557a0cb805b0ce51419d6d2821835d81ba1be36a416471/state \
  --diffusion-enabled \
  --diffusion-policy langchain_graph \
  --diffusion-max-artifacts 3 \
  --run-id hl4-ab-c-final-stream-01
```

Do not add `--state-dir`, `--state-ref`, or `--publish-state-ref` to B. Do not
use a moving channel reference for C. The original final-C result already uses
this manifest, so a B run is paired to that stored C stream; rerun C only when
measuring runtime/provider variance.

Recommended minimum: five independently sampled, predeclared held-out streams,
with each stream executed under A, B, and C. Randomize arm execution order per
stream. Keep their task manifests, frozen harness paths, and graph-bundle
digests alongside the results.

## Controls

Hold these constant across A, B, and C:

- task IDs and task order;
- executor, planner, mediator, and judge model IDs;
- task resources and verifier contracts;
- role-skill versions and `skill_updates=none`;
- `max_artifacts`, transfer-context token cap, and total prior-context cap;
- artifact emitter, compactor, and renderer behavior;
- runtime environment and Harbor image;
- test split only, with no harness selection or tuning after test starts.

The final harness and graph bundle for C are deliberately different from B.
That difference is the treatment: C measures the end-to-end value of HL4,
including its offline learning process.

## Outcomes

Primary outcome:

- verifier macro mean reward, aggregated per task ID within each stream.

Primary paired deltas:

- `B - A`: value of unguided artifact transfer;
- `C - B`: value of the learned graph and harness over random transfer;
- `C - A`: total value of the completed HL4 system.

Report each stream-level delta and a paired confidence interval. Do not treat
repeated occurrences of the same task within a short stream as independent
samples.

Secondary outcomes:

- verifier reward by task family and task position;
- environment failures, reported separately and excluded from efficacy claims;
- `regression_after_diffusion_context` rate for negative transfer;
- eligible, selected, rendered, compacted, and budget-dropped artifact counts;
- transfer-context tokens, total tokens, wall-clock duration, and cost;
- final graph-bundle digest and harness-overlay digest for provenance.

## Optional Stronger Baseline

Add a static `top_k_similarity` arm when budget permits. It distinguishes
learned LangChain graph routing from a non-learned graph heuristic:

```text
no diffusion -> random_k -> top_k_similarity -> final HL4
```

This additional arm is useful, but it does not replace the primary
random-K comparison.

## Selector-Only Diagnostic (Not a Benchmark Arm)

A frozen cross-batch artifact bank is not a valid benchmark condition because
the HL contract requires a fresh artifact bank at the start of every sequence.

For a diagnostic only, take C's causal artifact pool at each iteration and
simulate repeated `random_k` selections without executing them. Compare C's
chosen artifacts with the random-selection distribution on source reward,
relation, risk channel, and token cost. This can reveal whether C selects
unusually strong available artifacts, but it cannot establish counterfactual
task reward and must not be reported as the main A/B/C result.

## Interpretation

Claim an end-to-end HL4 benefit only when C improves on both A and B under
the paired held-out manifests, without a disproportionate rise in environment
failures, negative transfer, or cost. Test outcomes must not be used to alter
the harness, graph state, or selected final condition.
