# DCA Matrix: Diffusion Policy Comparison

## Scope

This note summarizes the four recent `medcoevo matrix` rows for the
`Distribution-Center-Auditing` task family. All four rows use
`condition_name = "learned_mediator"` and fixed skills, so the main changing
factor is the diffusion policy. Skill updates were disabled for executor,
planner, and mediator in all rows.

| Row | Experiment directory | Diffusion policy | Metrics graph label |
| --- | --- | --- | --- |
| No diffusion | `20260610-091144-dca-matrix-skill-none-diffusion-none` | `none` | `null` |
| Broadcast | `20260611-004017-dca-matrix-skill-none-diffusion-capped-broadcast` | `capped_broadcast` | `broadcast` |
| Random-k | `20260611-105315-dca-matrix-skill-none-diffusion-random-k` | `random_k` | `broadcast` |
| Top-k similarity | `20260611-134209-dca-matrix-skill-none-diffusion-top-k-similarity` | `top_k_similarity` | `precomputed_similarity` |

The result is positive only for random-k. Random-k has the best raw verifier
reward and best judge reward. Top-k similarity is the weakest diffusion row by
raw verifier reward and underperforms both no diffusion and broadcast.

## Aggregate Reward Summary

Raw reward is verifier reward. `raw mean, null=0` treats environment failures
as failed tasks. Judge reward is the post-hoc rubric reward from
`artifacts/judge_rewards.jsonl`.

| Row | Raw pass count | Raw mean, null=0 | Raw scored mean | Judge mean | Iter 0 raw | Iter 1 raw | Iter 2 raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 18 / 24 | 0.750 | 0.750 | 0.566 | 0.625 | 0.750 | 0.875 |
| Broadcast | 17 / 24 | 0.708 | 0.708 | 0.611 | 0.750 | 0.625 | 0.750 |
| Random-k | 20 / 24 | 0.833 | 0.870 | 0.703 | 0.750 | 0.875 | 0.875 |
| Top-k similarity | 15 / 24 | 0.625 | 0.652 | 0.620 | 0.625 | 0.500 | 0.750 |

No diffusion improves steadily from 5/8 to 7/8 raw passes. Broadcast is flat
overall and never exceeds the control. Random-k improves to 7/8 in iteration 1
and holds that level in iteration 2, giving the best aggregate result. Top-k
similarity drops to 4/8 in iteration 1 and only recovers to 6/8 in iteration 2.

## Raw Reward by Task

Each cell is the raw verifier reward trajectory `iter0/iter1/iter2`. `E` means
environment failure or missing raw reward.

| Task | No diffusion | Broadcast | Random-k | Top-k similarity |
| --- | --- | --- | --- | --- |
| `harbor_cycle_count_variance_audit` | 0/0/1 | 0/1/0 | 0/1/1 | E/0/1 |
| `harbor_outbound_manifest_audit` | 1/1/1 | 1/1/1 | 1/1/1 | 1/1/1 |
| `harbor_promo_register_audit` | 0/1/1 | 1/0/1 | 1/1/1 | 1/0/1 |
| `harbor_receiving_exception_audit` | 1/1/1 | 1/1/1 | 1/1/1 | 1/1/1 |
| `harbor_returns_disposition_audit` | 0/0/0 | 0/0/0 | E/0/0 | 0/0/0 |
| `harbor_service_queue_sla_audit` | 1/1/1 | 1/1/1 | 1/1/1 | 1/1/1 |
| `harbor_timesheet_policy_audit` | 1/1/1 | 1/1/1 | 1/1/1 | 0/0/1 |
| `harbor_trailer_detention_audit` | 1/1/1 | 1/0/1 | 1/1/1 | 1/1/0 |

The stable easy tasks are `harbor_outbound_manifest_audit`,
`harbor_receiving_exception_audit`, and `harbor_service_queue_sla_audit`, which
pass in every row except for no observed failures. The persistent hard case is
`harbor_returns_disposition_audit`, which fails in every scored run across all
rows.

Random-k's advantage comes from preserving the stable tasks while improving
`harbor_cycle_count_variance_audit` and avoiding the top-k regressions on
`harbor_timesheet_policy_audit` and `harbor_trailer_detention_audit`.

## Diffusion Artifact Use

| Row | Selected artifacts | Artifact types | Diffusion context tokens | Dominant selected sources |
| --- | ---: | --- | ---: | --- |
| Broadcast | 48 | 16 debug hints, 15 mediator summaries, 10 run outcomes, 7 regression warnings | 7,431 | `harbor_trailer_detention_audit` 42/48 |
| Random-k | 48 | 19 debug hints, 18 run outcomes, 11 mediator summaries | 6,746 | balanced: `receiving_exception` 11, `service_queue_sla` 10, `cycle_count` 6, `timesheet_policy` 6 |
| Top-k similarity | 40 | 13 debug hints, 13 mediator summaries, 14 run outcomes | 5,617 | `harbor_trailer_detention_audit` 30/40 |

The `diffused_records.jsonl` source verifier and source judge fields are mostly
null for these rows, so selected-source artifact quality cannot be reliably
computed directly from those records. The useful signal is instead the target
reward trajectory after diffusion context is rendered.

Broadcast and top-k similarity both over-concentrate source material from
`harbor_trailer_detention_audit`. Broadcast remains serviceable but does not
beat the no-diffusion control. Top-k similarity is more damaging: it routes a
narrow neighborhood of artifacts and coincides with regressions in
`timesheet_policy`, `promo_register`, and final `trailer_detention`.

Random-k is the only row where diffusion aligns with reward improvement. It
uses a similar number of selected artifacts to broadcast but distributes source
tasks more broadly and reaches the best raw and judge aggregate rewards.

## Token Accounting

Task-row token totals from `summary.json` and `metrics.jsonl`:

| Row | Total tokens | Prompt tokens | Completion tokens |
| --- | ---: | ---: | ---: |
| No diffusion | 280,756 | 202,570 | 78,186 |
| Broadcast | 312,966 | 236,425 | 76,541 |
| Random-k | 287,017 | 212,029 | 74,988 |
| Top-k similarity | 299,374 | 222,949 | 76,425 |

By agent:

| Row | Planner | Mediator | Judge | Compactor | Executor |
| --- | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 127,239 | 87,086 | 66,431 | 0 | 0 |
| Broadcast | 117,906 | 84,551 | 69,350 | 41,159 | 0 |
| Random-k | 118,482 | 75,095 | 61,458 | 31,982 | 0 |
| Top-k similarity | 119,430 | 80,460 | 64,680 | 34,804 | 0 |

By iteration:

| Row | Iter 0 | Iter 1 | Iter 2 |
| --- | ---: | ---: | ---: |
| No diffusion | 84,651 | 98,947 | 97,158 |
| Broadcast | 93,346 | 110,508 | 109,112 |
| Random-k | 80,100 | 105,024 | 101,893 |
| Top-k similarity | 89,444 | 109,941 | 99,989 |

Prior context:

| Row | Same-task prior | Cross-task prior | Diffusion context | Total planner prior |
| --- | ---: | ---: | ---: | ---: |
| No diffusion | 2,315 | 8,642 | 0 | 10,957 |
| Broadcast | 2,338 | 0 | 7,431 | 9,769 |
| Random-k | 2,079 | 0 | 6,746 | 8,825 |
| Top-k similarity | 2,225 | 985 | 5,617 | 8,828 |

Executor token usage is recorded as zero because Harbor/Hermes executor tokens
are not captured in these metrics. Random-k and top-k similarity each include
one `__coevolution__` bookkeeping row in `metrics.jsonl`; the tables above use
task-row totals only.

## Diffusion Usefulness

The main conclusion is that diffusion is policy-sensitive for DCA:

1. Random-k helps. It improves raw reward from 18/24 in the no-diffusion control
   to 20/24 and raises judge mean from 0.566 to 0.703.
2. Broadcast is not worth its extra token cost. It uses the most tokens and
   selects 48 artifacts, but raw reward drops to 17/24.
3. Top-k similarity is actively worse on verifier reward. It uses fewer
   diffusion tokens than broadcast and random-k, but selected artifacts are
   concentrated around `harbor_trailer_detention_audit` and final reward drops
   to 15/24.
4. Similarity is not enough. The top-k graph routes related task artifacts, but
   related artifacts are not necessarily transferable or correct.
5. Skill updates do not explain the results. Skill update policies are disabled
   for all rows, so the observed differences come from planner context,
   diffusion routing, stochastic executor behavior, and environment failures.

Future DCA diffusion should keep the source-diversity benefit of random-k but
add quality gates before selection. Useful gates would include pass/fail-aware
selection, judge-reward-aware filtering once source quality is available in
diffused records, and source-task caps to prevent trailer-detention dominance.

## Anomalies and Caveats

| Row | Anomaly | Impact |
| --- | --- | --- |
| Random-k | `harbor_returns_disposition_audit` has an iteration-0 environment failure and no raw reward. | Raw mean should report both null=0 and scored-only values. |
| Top-k similarity | `harbor_cycle_count_variance_audit` has an iteration-0 environment failure and no raw reward. | Top-k remains weak even when scored-only raw mean is used. |
| Top-k similarity | `harbor_service_queue_sla_audit` iteration 2 had a planner timeout on attempt 1/3, then recovered. | Adds latency but did not prevent a passing final result. |
| Top-k similarity | Final `harbor_trailer_detention_audit` live judge response was invalid and fell back to verifier reward before post-run annotation completed. | The final artifacts contain 23 judge records; raw reward remains the primary complete signal. |
| Diffusion rows | `diffused_records.jsonl` selected artifact source reward fields are mostly null. | Artifact quality cannot be scored directly from selected-source reward fields. |

These caveats do not change the ranking. Random-k is the best DCA row in the
current matrix; no diffusion is second by raw reward; broadcast is weaker than
the control; top-k similarity is the worst raw-reward row.
