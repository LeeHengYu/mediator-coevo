# HCBA Diffusion Family Analysis

## Scope

This note compares the `Healthcare-Cost-Benefit-Analysis` rows currently
available in `data/experiments`. Seed42 is the complete four-policy batch.
Seed1023 currently has three diffusion rows and no matching no-diffusion
control, so seed1023 supports within-diffusion comparison and partial
replication, not a matched causal comparison against no diffusion.

All rows use fixed executor, planner, and mediator skills, so the main changing
factor is the diffusion policy.

| Batch | Row | Experiment directory | Diffusion policy |
| --- | --- | --- | --- |
| Seed42 | No diffusion | `20260614-161720-hcba-seed42-row0-skill-none-diffusion-none` | `none` |
| Seed42 | Broadcast | `20260614-193534-hcba-seed42-row1-skill-none-broadcast` | `capped_broadcast` |
| Seed42 | Random-k | `20260614-232453-hcba-seed42-row2-skill-none-random-k` | `random_k` |
| Seed42 | Top-k similarity | `20260614-232453-hcba-seed42-row3-skill-none-top-k` | `top_k_similarity` |
| Seed1023 | Broadcast | `20260619-220108-HCBA-seed1023-row1` | `capped_broadcast` |
| Seed1023 | Random-k | `20260620-092733-HCBA-seed1023-row2` | `random_k` |
| Seed1023 | Top-k similarity | `20260620-184223-HCBA-seed1023-row3` | `top_k_similarity` |

## Aggregate Outcome

Raw reward is verifier reward. `Raw mean` treats missing raw rewards and
environment failures as zero over the fixed 27 task-run denominator. Judge
reward is the post-hoc rubric reward from `artifacts/judge_rewards.jsonl`; when
a run is not judged, judge mean follows the judged-row convention.

| Batch | Row | Raw pass count | Raw mean | Judge mean | Iter 0 raw | Iter 1 raw | Iter 2 raw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed42 | No diffusion | 6 / 27 | 0.222 | 0.252 | 0.111 | 0.333 | 0.222 |
| Seed42 | Broadcast | 8 / 27 | 0.296 | 0.320 | 0.111 | 0.556 | 0.222 |
| Seed42 | Random-k | 7 / 27 | 0.259 | 0.312 | 0.222 | 0.333 | 0.222 |
| Seed42 | Top-k similarity | 6 / 27 | 0.222 | 0.220 | 0.111 | 0.222 | 0.333 |
| Seed1023 | Broadcast | 6 / 27 | 0.222 | 0.235 | 0.000 | 0.222 | 0.444 |
| Seed1023 | Random-k | 7 / 27 | 0.259 | 0.271 | 0.222 | 0.222 | 0.333 |
| Seed1023 | Top-k similarity | 3 / 27 | 0.111 | 0.170 | 0.000 | 0.222 | 0.111 |

Seed42 favors capped broadcast. Broadcast has the best all-iteration raw reward
and judge reward, but the lift is front-loaded: it passes 5/9 tasks in
iteration 1 and returns to 2/9 in iteration 2.

Seed1023 changes the within-diffusion picture. Random-k has the best
all-iteration raw and judge reward among the three available diffusion rows.
Broadcast is weaker all-iteration, but it improves across the run from 0/9 to
4/9 passing tasks and becomes the strongest post-warmup row. Top-k similarity is
weak in seed1023 and does not look like a reliable improvement over simpler
routing.

## Token Accounting

Executor tokens are taken from the persisted Hermes executor accounting in
`metrics.jsonl`. `Adjusted total = orchestration token + executor token`.
Cache-read tokens are reported for audit. They are excluded from adjusted-token
efficiency and included only in weighted dollar-cost efficiency.

| Batch | Row | Orchestration token | Executor token | Adjusted total | Cache read audit | Efficiency | Judge efficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed42 | No diffusion | 370,734 | 772,162 | 1,142,896 | 6,992,896 | 0.019 | 0.022 |
| Seed42 | Broadcast | 403,734 | 772,685 | 1,176,419 | 7,524,352 | 0.025 | 0.027 |
| Seed42 | Random-k | 423,467 | 916,024 | 1,339,491 | 8,631,808 | 0.019 | 0.023 |
| Seed42 | Top-k similarity | 421,706 | 761,687 | 1,183,393 | 7,660,544 | 0.019 | 0.019 |
| Seed1023 | Broadcast | 412,485 | 885,809 | 1,298,294 | 5,492,736 | 0.017 | 0.018 |
| Seed1023 | Random-k | 400,451 | 695,431 | 1,095,882 | 6,290,944 | 0.024 | 0.025 |
| Seed1023 | Top-k similarity | 422,941 | 678,974 | 1,101,915 | 5,745,152 | 0.010 | 0.015 |

Seed42 broadcast spends only slightly more adjusted tokens than no diffusion, so
its all-iteration reward gain survives simple token-efficiency accounting.
Seed1023 random-k has the best all-iteration adjusted-token efficiency among
available seed1023 rows because it combines the highest pass count with a lower
executor total than broadcast.

## Post-Warmup Efficiency

Iteration 0 is a cold-start pass. The post-warmup view filters to
`iteration >= 1`, then reports raw and judge reward per 100k adjusted tokens.

| Batch | Row | Runs | Raw pass count | Raw mean | Judge mean | Orchestration token | Executor token | Adjusted total | Cache read audit | Post-warmup efficiency | Post-warmup judge efficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed42 | No diffusion | 18 | 5 / 18 | 0.278 | 0.295 | 248,559 | 487,288 | 735,847 | 4,742,144 | 0.038 | 0.040 |
| Seed42 | Broadcast | 18 | 7 / 18 | 0.389 | 0.374 | 268,237 | 550,704 | 818,941 | 5,720,576 | 0.047 | 0.046 |
| Seed42 | Random-k | 18 | 5 / 18 | 0.278 | 0.328 | 277,247 | 675,337 | 952,584 | 6,596,096 | 0.029 | 0.034 |
| Seed42 | Top-k similarity | 18 | 5 / 18 | 0.278 | 0.235 | 282,408 | 505,070 | 787,478 | 5,188,096 | 0.035 | 0.030 |
| Seed1023 | Broadcast | 18 | 6 / 18 | 0.333 | 0.313 | 276,841 | 557,824 | 834,665 | 3,652,608 | 0.040 | 0.037 |
| Seed1023 | Random-k | 18 | 5 / 18 | 0.278 | 0.270 | 266,286 | 418,582 | 684,868 | 4,116,480 | 0.041 | 0.039 |
| Seed1023 | Top-k similarity | 18 | 3 / 18 | 0.167 | 0.200 | 280,267 | 450,954 | 731,221 | 4,067,840 | 0.023 | 0.027 |

Post-warmup is the most relevant view for diffusion because it removes the
cold-start iteration. In seed42, broadcast is best on raw reward, judge reward,
and adjusted-token efficiency. In seed1023, broadcast has the best post-warmup
raw and judge reward, while random-k narrowly wins adjusted-token efficiency
because it is cheaper. Top-k similarity is consistently weak.

## Post-Warmup Role Token Split

The rows below use the same post-warmup filter. Models are:

| Role | Model |
| --- | --- |
| Planner | `openrouter/anthropic/claude-opus-4.6` |
| Mediator/compactor | `openrouter/google/gemini-3-flash-preview` |
| Judge | `openrouter/openai/gpt-oss-120b` |
| Executor | `openai/gpt-5.5` |

| Batch | Row | Planner | Mediator + compactor | Judge | Executor | Total | Executor share |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed42 | No diffusion | 118,417 | 75,565 | 54,577 | 487,288 | 735,847 | 66.2% |
| Seed42 | Broadcast | 108,738 | 105,612 | 53,887 | 550,704 | 818,941 | 67.2% |
| Seed42 | Random-k | 110,917 | 112,450 | 53,880 | 675,337 | 952,584 | 70.9% |
| Seed42 | Top-k similarity | 109,885 | 115,913 | 56,610 | 505,070 | 787,478 | 64.1% |
| Seed1023 | Broadcast | 112,564 | 111,058 | 53,219 | 557,824 | 834,665 | 66.8% |
| Seed1023 | Random-k | 106,092 | 107,174 | 53,020 | 418,582 | 684,868 | 61.1% |
| Seed1023 | Top-k similarity | 107,440 | 116,532 | 56,295 | 450,954 | 731,221 | 61.7% |

Executor tokens dominate the adjusted-token denominator, though the share varies
from 61-71% post-warmup. Diffusion mainly moves mediator/compactor context
handling, but executor behavior is still the larger cost driver.

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

| Batch | Row | Raw | Judge | Cost no cache | Cache $ | Cost w/cache | Raw/$ | Judge/$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed42 | No diffusion | 0.278 | 0.295 | 5.027 | 2.371 | 7.398 | 0.038 | 0.040 |
| Seed42 | Broadcast | 0.389 | 0.374 | 5.293 | 2.860 | 8.153 | 0.048 | 0.046 |
| Seed42 | Random-k | 0.278 | 0.328 | 5.965 | 3.298 | 9.263 | 0.030 | 0.035 |
| Seed42 | Top-k similarity | 0.278 | 0.235 | 4.866 | 2.594 | 7.460 | 0.037 | 0.032 |
| Seed1023 | Broadcast | 0.333 | 0.313 | 5.085 | 1.826 | 6.911 | 0.048 | 0.045 |
| Seed1023 | Random-k | 0.278 | 0.270 | 4.385 | 2.058 | 6.443 | 0.043 | 0.042 |
| Seed1023 | Top-k similarity | 0.167 | 0.200 | 4.624 | 2.034 | 6.658 | 0.025 | 0.030 |

The weighted cost view is the strongest current HCBA argument for capped
broadcast. Broadcast has the best post-warmup Raw/$ in both seed42 and seed1023.
Random-k is cheaper in seed1023 and close on adjusted-token efficiency, but it
does not match broadcast's raw or judge reward after warmup. Top-k similarity is
cheaper than seed42 random-k but loses too much reward to be attractive.

## Family-Level Policy Average

This table averages the observed HCBA rows by policy. It should be read
descriptively because no-diffusion has only seed42 coverage and seed1023 lacks a
matched no-diffusion row.

| Policy | Seeds | Avg raw | Avg judge | Avg post raw | Avg post judge | Avg post-warmup efficiency | Avg Raw/$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 1 | 0.222 | 0.252 | 0.278 | 0.295 | 0.038 | 0.038 |
| Broadcast | 2 | 0.259 | 0.278 | 0.361 | 0.343 | 0.044 | 0.048 |
| Random-k | 2 | 0.259 | 0.291 | 0.278 | 0.299 | 0.035 | 0.037 |
| Top-k similarity | 2 | 0.167 | 0.195 | 0.222 | 0.217 | 0.029 | 0.031 |

The family-level signal is therefore not "more diffusion always helps." It is
more specific: capped broadcast is the best observed HCBA policy after warmup
and under cache-aware weighted cost. Random-k has a better all-iteration judge
average than broadcast because of seed1023, but its post-warmup and cost views
are weaker. Top-k similarity underperforms across both seeds.

## Diffusion Artifact Use

| Batch | Row | Selected/rendered artifacts | Eligible artifacts | Transfer context tokens | Regressions after context | Budget violations | Selected artifact types |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Seed42 | No diffusion | 0 / 0 | 0 | 12,532 | 0 | 1 | none |
| Seed42 | Broadcast | 53 / 53 | 648 | 10,486 | 4 | 0 | `run_outcome` |
| Seed42 | Random-k | 61 / 61 | 648 | 11,844 | 4 | 2 | `run_outcome` |
| Seed42 | Top-k similarity | 30 / 30 | 648 | 7,172 | 1 | 4 | `run_outcome` |
| Seed1023 | Broadcast | 34 / 34 | 648 | 7,105 | 1 | 1 | `run_outcome` |
| Seed1023 | Random-k | 58 / 58 | 648 | 11,360 | 3 | 2 | `run_outcome` |
| Seed1023 | Top-k similarity | 27 / 27 | 648 | 7,075 | 1 | 3 | `run_outcome` |

The artifact pipeline is operational in all diffusion rows: selected artifacts
are rendered, and all selected artifacts are `run_outcome` artifacts. Broadcast
does not win by selecting the most artifacts. In seed1023 it selects far fewer
artifacts than random-k but has better post-warmup raw reward and Raw/$.

## Context Budget Binding

This table reports post-warmup average utilization of the available transfer,
same-task, and total-prior budgets. A hard hit is a run using at least 98% of
the relevant cap.

| Batch | Row | Transfer fill | Transfer hard hits | Same-task fill | Same-task hard hits | Total-prior fill | Total hard hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed42 | No diffusion | 77.4% | 6 / 18 | 56.1% | 0 / 18 | 72.0% | 0 / 18 |
| Seed42 | Broadcast | 64.7% | 0 / 18 | 59.9% | 0 / 18 | 63.5% | 0 / 18 |
| Seed42 | Random-k | 73.1% | 0 / 18 | 60.2% | 0 / 18 | 69.9% | 0 / 18 |
| Seed42 | Top-k similarity | 44.3% | 0 / 18 | 56.7% | 1 / 18 | 47.4% | 0 / 18 |
| Seed1023 | Broadcast | 43.9% | 0 / 18 | 62.6% | 0 / 18 | 48.5% | 0 / 18 |
| Seed1023 | Random-k | 70.1% | 0 / 18 | 57.1% | 0 / 18 | 66.9% | 0 / 18 |
| Seed1023 | Top-k similarity | 43.7% | 0 / 18 | 62.3% | 0 / 18 | 48.3% | 0 / 18 |

Budget fill does not explain reward by itself. Broadcast wins the post-warmup
family view with lower transfer fill than random-k. Top-k also has low transfer
fill, but low context volume did not translate into reliable reward.

## Raw Reward By Task

Each cell is the raw verifier reward trajectory `iter0/iter1/iter2`.

### Seed42

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

### Seed1023

| Task | Broadcast | Random-k | Top-k similarity |
| --- | --- | --- | --- |
| `cyclemargin_30v90` | 0/0/1 | 0/1/0 | 0/1/0 |
| `diagpanel_14v28` | 0/0/0 | 0/0/0 | 0/0/0 |
| `gdpval_42` | 0/1/1 | 0/0/1 | 0/0/0 |
| `infusionbatch_7v14` | 0/1/0 | 0/0/1 | 0/0/0 |
| `mailerfill_45v90` | 0/0/1 | 1/1/0 | 0/1/0 |
| `oncocooler_10v20` | 0/0/0 | 0/0/env | 0/0/1 |
| `reagentkit_bulk` | 0/0/0 | 1/0/0 | 0/0/0 |
| `syncpack_28v56` | 0/0/1 | 0/0/1 | 0/0/0 |
| `vaxcrate_6v12` | 0/0/0 | 0/0/0 | 0/0/0 |

`diagpanel_14v28` is a hard zero across all observed HCBA rows. `mailerfill_45v90`
is the most stable task in seed42 and remains one of the easier tasks in
seed1023. Broadcast's seed42 gain comes from a broad iteration-1 lift; in
seed1023, broadcast's strength is later, with four passing tasks in iteration 2.

## Interpretation and Conclusion

The HCBA family is positive for capped broadcast after warmup, but the evidence
should be phrased as policy-family matching rather than a universal diffusion
effect.

1. Seed42 is the clean matched comparison and favors broadcast. Broadcast has
   the best raw mean, judge mean, post-warmup adjusted-token efficiency, and
   weighted dollar-cost efficiency.
2. Seed1023 is partial replication, not a full matched batch. It lacks a
   no-diffusion row, but among diffusion policies random-k has the best
   all-iteration raw and judge reward while broadcast has the best post-warmup
   raw reward and Raw/$.
3. Across observed HCBA rows, broadcast is the strongest post-warmup and
   cache-aware cost candidate. It averages 0.361 post-warmup raw reward and
   0.048 Raw/$ across two seeds.
4. Random-k is mixed. It can improve all-iteration judge reward, but its
   post-warmup reward and cost efficiency do not beat broadcast in HCBA.
5. Top-k similarity is the main negative HCBA result. It is selective, but it
   underperforms both broadcast and random-k across the two observed seeds.
6. The useful signal is medium-volume, family-level broadcast context, not
   simply more artifacts or graph-aware selection.

## Anomalies and Caveats

| Issue | Impact |
| --- | --- |
| Seed1023 has no matching no-diffusion row. | Seed1023 cannot prove diffusion helped over no diffusion; it only compares diffusion policies and checks whether seed42 patterns repeat. |
| Diffusion rows include one synthetic `__coevolution__` metrics row. | Synthetic rows are excluded from task denominators and token tables. |
| Seed1023 random-k has one environment failure: `oncocooler_10v20` at iteration 2. | Raw means treat the failure as zero in the 27-run denominator; judge means are over judged rows. |
| The seed42 no-diffusion row still records cross-task prior tokens. | Interpret "no diffusion" as no artifact diffusion, not as no prior context of any kind. |
| Several context-budget violation flags appear even when visible transfer, same-task, and total-prior counts are below caps and no artifacts were dropped. | Budget flags should be audited before treating them as strict cap violations. |
| Broadcast and random-k can record regressions after diffusion context. | Broadcast still wins the post-warmup family view, but per-task regressions are a durability caveat. |
| HCBA currently has one complete seed and one partial seed. | The result is descriptive evidence for a paper figure, not yet a broad causal claim across seeds. |

## Accounting Guidance

Use both adjusted-token and weighted-cost accounting, but keep their claims
separate.

| Question | Accounting view |
| --- | --- |
| Which policy is most efficient end-to-end by token volume? | Use adjusted total tokens, raw efficiency, and judge efficiency. |
| Which policy is most efficient under model-price asymmetry? | Use weighted dollar-cost efficiency with executor input/output and cache-read split. |
| Did prior context bind the run? | Use transfer, same-task, and total-prior binding rates. |
| Did diffusion help beyond ordinary cross-task prior? | Use matched seed42 post-warmup reward and cost deltas; treat seed1023 as partial replication until row0 exists. |
| Did the graph or routing matter? | Compare broadcast, random-k, and top-k at their observed transfer-token scale. |

For future HCBA comparisons, keep reporting post-warmup adjusted-token
efficiency and cache-aware weighted dollar-cost efficiency. The current robust
claim is that capped broadcast is the strongest HCBA diffusion candidate after
warmup, while top-k similarity is not yet supported and the missing seed1023
control should be filled before making a stronger causal statement.
