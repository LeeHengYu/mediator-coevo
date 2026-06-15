# HCBA Seed42 Diffusion Comparison

## Scope

This note compares the latest four `Healthcare-Cost-Benefit-Analysis` rows in
`data/experiments`. All rows use fixed executor, planner, and mediator skills,
so the main changing factor is the diffusion policy.

| Row | Experiment directory | Diffusion policy |
| --- | --- | --- |
| No diffusion | `20260614-161720-hcba-seed42-row0-skill-none-diffusion-none` | `none` |
| Broadcast | `20260614-193534-hcba-seed42-row1-skill-none-broadcast` | `capped_broadcast` |
| Random-k | `20260614-232453-hcba-seed42-row2-skill-none-random-k` | `random_k` |
| Top-k similarity | `20260614-232453-hcba-seed42-row3-skill-none-top-k` | `top_k_similarity` |

## Aggregate Outcome

Raw reward is verifier reward. `Raw mean` treats missing raw rewards as zero
over the 27 task-run denominator. Judge reward is the post-hoc rubric reward
from `artifacts/judge_rewards.jsonl`.

| Row | Raw pass count | Raw mean | Judge mean | Iter 0 raw | Iter 1 raw | Iter 2 raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 6 / 27 | 0.222 | 0.252 | 0.111 | 0.333 | 0.222 |
| Broadcast | 8 / 27 | 0.296 | 0.320 | 0.111 | 0.556 | 0.222 |
| Random-k | 7 / 27 | 0.259 | 0.312 | 0.222 | 0.333 | 0.222 |
| Top-k similarity | 6 / 27 | 0.222 | 0.220 | 0.111 | 0.222 | 0.333 |

Broadcast is the strongest HCBA row. It has the best all-iteration raw reward,
judge reward, adjusted-token efficiency, and weighted dollar-cost efficiency.
The gain is front-loaded: broadcast passes 5/9 tasks in iteration 1, then falls
back to 2/9 in iteration 2.

Random-k improves judge reward over no diffusion, but it does not improve raw
reward post-warmup and is much more expensive. Top-k similarity has the weakest
judge mean and does not beat the no-diffusion control on raw reward.

## Token Accounting

Executor tokens are taken from the persisted Hermes executor accounting in
`metrics.jsonl`. `Adjusted total = orchestration token + executor token`.
Cache-read tokens are reported for audit. They are excluded from adjusted-token
efficiency and included only in weighted dollar-cost efficiency.

| Row | Orchestration token | Executor token | Adjusted total | Cache read audit | Efficiency | Judge efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 370,734 | 772,162 | 1,142,896 | 6,992,896 | 0.019 | 0.022 |
| Broadcast | 403,734 | 772,685 | 1,176,419 | 7,524,352 | 0.025 | 0.027 |
| Random-k | 423,467 | 916,024 | 1,339,491 | 8,631,808 | 0.019 | 0.023 |
| Top-k similarity | 421,706 | 761,687 | 1,183,393 | 7,660,544 | 0.019 | 0.019 |

Broadcast spends only slightly more adjusted tokens than no diffusion, so its
reward gain survives simple token-efficiency accounting. Random-k has the
largest adjusted total and largest cache-read volume.

## Post-Warmup Efficiency

Iteration 0 is a cold-start pass. The post-warmup view filters to
`iteration >= 1`, then reports raw and judge reward per 100k adjusted tokens.

| Row | Runs | Raw pass count | Raw mean | Judge mean | Orchestration token | Executor token | Adjusted total | Cache read audit | Post-warmup efficiency | Post-warmup judge efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 18 | 5 / 18 | 0.278 | 0.295 | 248,559 | 487,288 | 735,847 | 4,742,144 | 0.038 | 0.040 |
| Broadcast | 18 | 7 / 18 | 0.389 | 0.374 | 268,237 | 550,704 | 818,941 | 5,720,576 | 0.047 | 0.046 |
| Random-k | 18 | 5 / 18 | 0.278 | 0.328 | 277,247 | 675,337 | 952,584 | 6,596,096 | 0.029 | 0.034 |
| Top-k similarity | 18 | 5 / 18 | 0.278 | 0.235 | 282,408 | 505,070 | 787,478 | 5,188,096 | 0.035 | 0.030 |

Broadcast is also the best post-warmup row. Random-k is the main warning case:
it has a higher post-warmup judge mean than no diffusion, but the extra executor
load pushes both raw and judge efficiency below the control.

## Post-Warmup Role Token Split

The rows below use the same post-warmup filter. Models are:

| Role | Model |
| --- | --- |
| Planner | `openrouter/anthropic/claude-opus-4.6` |
| Mediator/compactor | `openrouter/google/gemini-3-flash-preview` |
| Judge | `openrouter/openai/gpt-oss-120b` |
| Executor | `openai/gpt-5.5` |

| Row | Planner | Mediator + compactor | Judge | Executor | Total | Executor share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 118,417 | 75,565 | 54,577 | 487,288 | 735,847 | 66.2% |
| Broadcast | 108,738 | 105,612 | 53,887 | 550,704 | 818,941 | 67.2% |
| Random-k | 110,917 | 112,450 | 53,880 | 675,337 | 952,584 | 70.9% |
| Top-k similarity | 109,885 | 115,913 | 56,610 | 505,070 | 787,478 | 64.1% |

Executor tokens dominate the post-warmup denominator. Diffusion mainly increases
mediator/compactor context handling, but random-k also drives a large executor
increase.

## Weighted Dollar-Cost Efficiency

This section applies the same proxy dollar-cost model as `docs/WRA-matrix.md`,
per 1M tokens:

| Token class | Weight |
| --- | ---: |
| Planner | 5.0 |
| Mediator + compactor | 0.5 |
| Judge | 0.0 |
| Executor input | 5.0 |
| Executor output | 25.0 |
| Executor cache read | 0.5 |

`Weighted cost = planner*5 + mediator_compactor*0.5 + judge*0 +
executor_input*5 + executor_output*25 + cache_read*0.5`, divided by 1,000,000.
`Raw/$` and `Judge/$` are post-warmup reward mean divided by weighted cost.

| Row | Raw | Judge | Cost no cache | Cache $ | Cost w/cache | Raw/$ | Judge/$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 0.278 | 0.295 | 5.027 | 2.371 | 7.398 | 0.038 | 0.040 |
| Broadcast | 0.389 | 0.374 | 5.293 | 2.860 | 8.153 | 0.048 | 0.046 |
| Random-k | 0.278 | 0.328 | 5.965 | 3.298 | 9.263 | 0.030 | 0.035 |
| Top-k similarity | 0.278 | 0.235 | 4.866 | 2.594 | 7.460 | 0.037 | 0.032 |

The weighted cost view keeps broadcast in first place. Broadcast costs more than
no diffusion, but the reward lift is large enough to improve both Raw/$ and
Judge/$. Random-k is penalized most heavily because it has the highest executor
and cache-read cost. Top-k similarity is relatively cheap before cache, but weak
judge reward keeps it below the no-diffusion control.

## Diffusion Artifact Use

| Row | Selected/rendered artifacts | Eligible artifacts | Transfer context tokens | Regressions after context | Budget violations | Selected artifact types |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| No diffusion | 0 / 0 | 0 | 12,532 | 0 | 1 | none |
| Broadcast | 53 / 53 | 648 | 10,486 | 4 | 0 | `run_outcome` |
| Random-k | 61 / 61 | 648 | 11,844 | 4 | 2 | `run_outcome` |
| Top-k similarity | 30 / 30 | 648 | 7,172 | 1 | 4 | `run_outcome` |

The artifact pipeline is operational in all diffusion rows: selected artifacts
are rendered, and all selected artifacts are `run_outcome` artifacts. Broadcast
uses fewer artifacts than random-k, more than top-k, and gets the best reward.
That makes the result look more like useful medium-volume context than a simple
"more context is better" pattern.

## Context Budget Binding

This table reports post-warmup average utilization of the available transfer,
same-task, and total-prior budgets. A hard hit is a run using at least 98% of
the relevant cap.

| Row | Transfer fill | Transfer hard hits | Same-task fill | Same-task hard hits | Total-prior fill | Total hard hits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 77.4% | 6 / 18 | 56.1% | 0 / 18 | 72.0% | 0 / 18 |
| Broadcast | 64.7% | 0 / 18 | 59.9% | 0 / 18 | 63.5% | 0 / 18 |
| Random-k | 73.1% | 0 / 18 | 60.2% | 0 / 18 | 69.9% | 0 / 18 |
| Top-k similarity | 44.3% | 0 / 18 | 56.7% | 1 / 18 | 47.4% | 0 / 18 |

HCBA uses far more of its context budgets than the WRA seed42 rows did. The
budget-fill signal does not explain reward by itself: no diffusion has the
highest transfer hard-hit count, while broadcast wins with lower transfer fill.

## Raw Reward By Task

Each cell is the raw verifier reward trajectory `iter0/iter1/iter2`.

| Task | No diffusion | Broadcast | Random-k | Top-k similarity |
| --- | --- | --- | --- | --- |
| `cyclemargin_30v90` | 1/1/0 | 0/1/1 | 1/0/1 | 0/0/1 |
| `diagpanel_14v28` | 0/0/0 | 0/0/0 | 0/0/0 | 0/0/0 |
| `gdpval_42` | 0/0/0 | 0/1/0 | 0/0/0 | 0/0/1 |
| `infusionbatch_7v14` | 0/0/0 | 0/0/0 | 0/1/0 | 0/1/0 |
| `mailerfill_45v90` | 0/1/1 | 0/1/0 | 1/1/0 | 1/1/1 |
| `oncocooler_10v20` | 0/1/0 | 0/1/0 | 0/1/0 | 0/0/0 |
| `reagentkit_bulk` | 0/0/0 | 1/1/0 | 0/0/0 | 0/0/0 |
| `syncpack_28v56` | 0/0/1 | 0/0/1 | 0/0/0 | 0/0/0 |
| `vaxcrate_6v12` | 0/0/0 | 0/0/0 | 0/0/1 | 0/0/0 |

`diagpanel_14v28` is a hard zero across all rows. `mailerfill_45v90` is the
most stable task, especially under top-k similarity. Broadcast's aggregate gain
comes from a broad iteration-1 lift: `cyclemargin_30v90`, `gdpval_42`,
`mailerfill_45v90`, `oncocooler_10v20`, and `reagentkit_bulk` all pass in
iteration 1.

## Interpretation and Conclusion

The HCBA seed42 batch is positive for broadcast diffusion, but the effect is not
durable across iterations.

1. Broadcast is the best overall row. It has the best raw mean, judge mean,
   post-warmup adjusted-token efficiency, and weighted dollar-cost efficiency.
2. The broadcast gain is mostly iteration 1. By iteration 2, broadcast returns
   to 2/9 passing tasks, matching no diffusion and random-k.
3. Random-k is not cost-effective. It selects the most artifacts and achieves a
   higher judge mean than no diffusion, but its executor and cache costs are too
   high.
4. Top-k similarity is selective but weak. It selects only 30 artifacts and has
   the lowest transfer-token load, but it has the weakest judge reward.
5. Diffusion usefulness is therefore policy-specific. Medium-volume broadcast
   context helped this HCBA batch; selective top-k did not, and random-k spent
   too much for too little durable reward.

## Anomalies and Caveats

| Issue | Impact |
| --- | --- |
| Diffusion rows include one synthetic `__coevolution__` metrics row. | Synthetic rows are excluded from task denominators and token tables. |
| No environment failures were reported in the persisted summaries. | Raw means use the full 27 task-run denominator without environment-failure adjustment. |
| The no-diffusion row still records cross-task prior tokens. | Interpret "no diffusion" as no artifact diffusion, not as no prior context of any kind. |
| Several context-budget violation flags appear even when visible transfer, same-task, and total-prior counts are below caps and no artifacts were dropped. | Budget flags should be audited before treating them as strict cap violations. |
| Broadcast and random-k each record four regressions after diffusion context. | Broadcast still wins in aggregate, but the iteration-2 falloff is a real durability caveat. |
| This is one seed batch. | The result is descriptive evidence for HCBA seed42, not a broad causal claim about all HCBA seeds. |

## Accounting Guidance

Use both adjusted-token and weighted-cost accounting, but keep their claims
separate.

| Question | Accounting view |
| --- | --- |
| Which policy is most efficient end-to-end by token volume? | Use adjusted total tokens, raw efficiency, and judge efficiency. |
| Which policy is most efficient under model-price asymmetry? | Use weighted dollar-cost efficiency with executor input/output and cache-read split. |
| Did prior context bind the run? | Use transfer, same-task, and total-prior binding rates. |
| Did diffusion help beyond ordinary cross-task prior? | Use post-warmup reward and cost deltas, not all-iteration reward alone. |
| Did the graph or routing matter? | Compare broadcast, random-k, and top-k at their observed transfer-token scale. |

For future HCBA comparisons, keep reporting post-warmup adjusted-token
efficiency and cache-aware weighted dollar-cost efficiency. The current robust
claim is that broadcast diffusion helped this latest HCBA seed42 batch, but the
gain was front-loaded and should be checked against additional seeds.
