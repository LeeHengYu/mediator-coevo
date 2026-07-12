# WRA Seed Diffusion Comparison

## Scope

This note compares three four-row seed batches for the
`Weighted-Risk-Assessment` family: `seed42`, `seed1023`, and `seed1117`.
All rows use fixed skills, so the main changing factor is the diffusion policy.

| Batch | Row | Experiment directory | Diffusion policy |
| --- | --- | --- | --- |
| Seed42 | No diffusion | `20260612-005909-wra-matrix-skill-none-diffusion-none-infra1` | `none` |
| Seed42 | Broadcast | `20260612-041549-wra-matrix-skill-none-diffusion-capped-broadcast-infra1` | `capped_broadcast` |
| Seed42 | Top-k similarity | `20260612-132411-wra-matrix-skill-none-diffusion-top-k-similarity-infra1` | `top_k_similarity` |
| Seed42 | Random-k | `20260612-095835-wra-matrix-skill-none-diffusion-random-k-infra1` | `random_k` |
| Seed1023 | No diffusion | `20260612-221103-wra-seed1023-row0-skill-none-diffusion-none` | `none` |
| Seed1023 | Broadcast | `20260613-012706-wra-seed1023-row1-skill-none-capped-broadcast` | `capped_broadcast` |
| Seed1023 | Top-k similarity | `20260613-123456-wra-seed1023-row3-skill-none-top-k-similarity` | `top_k_similarity` |
| Seed1023 | Random-k | `20260613-033012-wra-seed1023-row2-skill-none-random-k` | `random_k` |
| Seed1117 | No diffusion | `20260613-192006-wra-seed1117-row0-skill-none-diffusion-none` | `none` |
| Seed1117 | Broadcast | `20260613-215935-wra-seed1117-row1-skill-none-broadcast` | `capped_broadcast` |
| Seed1117 | Top-k similarity | `20260614-101928-wra-seed1117-row3-skill-none-top-k` | `top_k_similarity` |
| Seed1117 | Random-k | `20260614-014929-wra-seed1117-row2-skill-none-random-k` | `random_k` |

## Aggregate Outcome

Raw reward is verifier reward. `Raw mean` treats environment failures or missing
raw rewards as zero over the 24 task-run denominator. Judge reward is the
post-hoc rubric reward from `artifacts/judge_rewards.jsonl`; when a run is not
judged, the judge mean follows the persisted summary convention.

| Batch | Row | Raw pass count | Raw mean | Judge mean | Iter 0 raw | Iter 1 raw | Iter 2 raw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed42 | No diffusion | 17 / 24 | 0.708 | 0.653 | 0.500 | 0.875 | 0.750 |
| Seed42 | Broadcast | 15 / 24 | 0.625 | 0.584 | 0.375 | 0.875 | 0.625 |
| Seed42 | Top-k similarity | 20 / 24 | 0.833 | 0.681 | 0.750 | 0.875 | 0.875 |
| Seed42 | Random-k | 21 / 24 | 0.875 | 0.698 | 0.875 | 0.875 | 0.875 |
| Seed1023 | No diffusion | 19 / 24 | 0.792 | 0.702 | 0.750 | 0.875 | 0.750 |
| Seed1023 | Broadcast | 17 / 24 | 0.708 | 0.571 | 0.625 | 0.625 | 0.875 |
| Seed1023 | Top-k similarity | 16 / 24 | 0.667 | 0.600 | 0.500 | 0.750 | 0.750 |
| Seed1023 | Random-k | 15 / 24 | 0.625 | 0.556 | 0.375 | 0.625 | 0.875 |
| Seed1117 | No diffusion | 16 / 24 | 0.667 | 0.591 | 0.375 | 0.750 | 0.875 |
| Seed1117 | Broadcast | 20 / 24 | 0.833 | 0.667 | 0.875 | 1.000 | 0.625 |
| Seed1117 | Top-k similarity | 17 / 24 | 0.708 | 0.588 | 0.625 | 0.750 | 0.750 |
| Seed1117 | Random-k | 18 / 24 | 0.750 | 0.632 | 0.625 | 0.750 | 0.875 |

Seed42 is positive for selective diffusion. Random-k and top-k both outperform
the no-diffusion control on raw reward, final-iteration reward, and post-warmup
efficiency. Broadcast remains weak.

Seed1023 is negative for diffusion overall. No diffusion has the best raw mean,
judge mean, and most efficiency views. Top-k becomes competitive only when
costed by the weighted dollar model, because it is cheaper.

Seed1117 is mixed. Broadcast has the best all-iteration raw and judge reward,
but its benefit is front-loaded: it passes 8/8 tasks in iteration 1 and falls to
5/8 in iteration 2. No diffusion matches broadcast post-warmup raw reward with
substantially lower token and dollar cost.

## Token Accounting

The old Harbor `result.json` `agent_result` token fields are zero in persisted
runs. Executor tokens are recovered from each trial's
`agent/hermes-session.jsonl`. Seed1117 metrics usually already include Hermes
executor tokens, but the same recovery path is still used for consistency and
for the seed1117 broadcast environment-failure row.

`Executor token = input_tokens + output_tokens` from Hermes session logs.

`Adjusted total = orchestration token + executor token`.

Cache-read tokens are reported in the token tables for audit. They are excluded
from adjusted-token efficiency and included only in the weighted dollar-cost
section.

| Batch | Row | Orchestration token | Executor token | Adjusted total | Cache read audit | Efficiency | Judge efficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed42 | No diffusion | 235,433 | 694,872 | 930,305 | 5,211,648 | 0.076 | 0.070 |
| Seed42 | Broadcast | 288,463 | 670,417 | 958,880 | 6,046,720 | 0.065 | 0.061 |
| Seed42 | Top-k similarity | 269,017 | 532,978 | 801,995 | 5,204,480 | 0.104 | 0.085 |
| Seed42 | Random-k | 273,744 | 541,863 | 815,607 | 5,409,280 | 0.107 | 0.086 |
| Seed1023 | No diffusion | 238,478 | 626,173 | 864,651 | 4,826,624 | 0.092 | 0.081 |
| Seed1023 | Broadcast | 293,450 | 644,635 | 938,085 | 5,507,072 | 0.076 | 0.061 |
| Seed1023 | Top-k similarity | 296,935 | 537,654 | 834,589 | 4,979,200 | 0.080 | 0.072 |
| Seed1023 | Random-k | 291,219 | 599,640 | 890,859 | 5,071,360 | 0.070 | 0.062 |
| Seed1117 | No diffusion | 275,576 | 602,381 | 877,957 | 4,918,784 | 0.076 | 0.067 |
| Seed1117 | Broadcast | 260,602 | 658,593 | 919,195 | 5,658,624 | 0.091 | 0.073 |
| Seed1117 | Top-k similarity | 299,644 | 720,353 | 1,019,997 | 5,680,640 | 0.069 | 0.058 |
| Seed1117 | Random-k | 294,084 | 691,358 | 985,442 | 5,789,696 | 0.076 | 0.064 |

## Post-Warmup Efficiency

Iteration 0 is a cold-start pass with no same-task prior and little or no
transfer context. The post-warmup view filters to `iteration >= 1`, then
reports raw and judge reward per 100k adjusted tokens.

| Batch | Row | Runs | Raw pass count | Raw mean | Orchestration token | Executor token | Adjusted total | Cache read audit | Post-warmup efficiency | Post-warmup judge efficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed42 | No diffusion | 16 | 13 / 16 | 0.812 | 151,951 | 438,259 | 590,210 | 3,276,800 | 0.138 | 0.126 |
| Seed42 | Broadcast | 16 | 12 / 16 | 0.750 | 183,219 | 438,032 | 621,251 | 4,180,480 | 0.121 | 0.107 |
| Seed42 | Top-k similarity | 16 | 14 / 16 | 0.875 | 178,682 | 351,433 | 530,115 | 3,611,648 | 0.165 | 0.126 |
| Seed42 | Random-k | 16 | 14 / 16 | 0.875 | 188,248 | 321,814 | 510,062 | 3,538,944 | 0.172 | 0.150 |
| Seed1023 | No diffusion | 16 | 13 / 16 | 0.812 | 159,564 | 396,926 | 556,490 | 3,345,920 | 0.146 | 0.133 |
| Seed1023 | Broadcast | 16 | 12 / 16 | 0.750 | 197,308 | 418,353 | 615,661 | 3,676,672 | 0.122 | 0.105 |
| Seed1023 | Top-k similarity | 16 | 12 / 16 | 0.750 | 197,623 | 331,290 | 528,913 | 3,252,224 | 0.142 | 0.123 |
| Seed1023 | Random-k | 16 | 12 / 16 | 0.750 | 189,627 | 359,153 | 548,780 | 3,397,632 | 0.137 | 0.114 |
| Seed1117 | No diffusion | 16 | 13 / 16 | 0.812 | 189,114 | 351,445 | 540,559 | 3,101,184 | 0.150 | 0.124 |
| Seed1117 | Broadcast | 16 | 13 / 16 | 0.812 | 174,640 | 421,982 | 596,622 | 3,827,712 | 0.136 | 0.114 |
| Seed1117 | Top-k similarity | 16 | 12 / 16 | 0.750 | 202,893 | 471,964 | 674,857 | 3,518,464 | 0.111 | 0.093 |
| Seed1117 | Random-k | 16 | 13 / 16 | 0.812 | 194,686 | 465,553 | 660,239 | 3,675,648 | 0.123 | 0.102 |

Post-warmup no diffusion is the most stable policy: raw efficiency is 0.138,
0.146, and 0.150 across seeds. Diffusion clearly wins post-warmup efficiency
only in seed42. In seed1023 and seed1117, the control is the strongest or tied
on raw reward with lower cost.

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
| Seed42 | No diffusion | 65,805 | 44,581 | 41,565 | 438,259 | 590,210 | 74.3% |
| Seed42 | Broadcast | 72,754 | 68,507 | 41,958 | 438,032 | 621,251 | 70.5% |
| Seed42 | Top-k similarity | 67,830 | 68,070 | 42,782 | 351,433 | 530,115 | 66.3% |
| Seed42 | Random-k | 72,623 | 72,021 | 43,604 | 321,814 | 510,062 | 63.1% |
| Seed1023 | No diffusion | 64,697 | 49,037 | 45,830 | 396,926 | 556,490 | 71.3% |
| Seed1023 | Broadcast | 72,598 | 79,147 | 45,563 | 418,353 | 615,661 | 68.0% |
| Seed1023 | Top-k similarity | 69,616 | 81,681 | 46,326 | 331,290 | 528,913 | 62.6% |
| Seed1023 | Random-k | 72,768 | 75,021 | 41,838 | 359,153 | 548,780 | 65.4% |
| Seed1117 | No diffusion | 80,608 | 62,533 | 45,973 | 351,445 | 540,559 | 65.0% |
| Seed1117 | Broadcast | 70,595 | 62,762 | 41,283 | 421,982 | 596,622 | 70.7% |
| Seed1117 | Top-k similarity | 72,845 | 83,203 | 46,845 | 471,964 | 674,857 | 69.9% |
| Seed1117 | Random-k | 72,359 | 77,003 | 45,324 | 465,553 | 660,239 | 70.5% |

Executor tokens dominate the adjusted-token denominator, usually 63-74% of
post-warmup total. Planner and judge usage are comparatively stable. Diffusion
mainly moves mediator/compactor tokens and, depending on seed, executor tokens.

## Weighted Dollar-Cost Efficiency

This section applies a proxy dollar-cost model per 1M tokens:

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
`Raw/$` and `Judge/$` are reward mean divided by this weighted cost.

| Batch | Row | Raw | Judge | Cost no cache | Cache $ | Cost w/cache | Raw/$ | Judge/$ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed42 | No diffusion | 0.812 | 0.743 | 4.032 | 1.638 | 5.670 | 0.143 | 0.131 |
| Seed42 | Broadcast | 0.750 | 0.662 | 4.147 | 2.090 | 6.237 | 0.120 | 0.106 |
| Seed42 | Top-k similarity | 0.875 | 0.670 | 3.539 | 1.806 | 5.345 | 0.164 | 0.125 |
| Seed42 | Random-k | 0.875 | 0.765 | 3.425 | 1.769 | 5.194 | 0.168 | 0.147 |
| Seed1023 | No diffusion | 0.812 | 0.741 | 3.814 | 1.673 | 5.487 | 0.148 | 0.135 |
| Seed1023 | Broadcast | 0.750 | 0.645 | 3.960 | 1.838 | 5.798 | 0.129 | 0.111 |
| Seed1023 | Top-k similarity | 0.750 | 0.651 | 3.425 | 1.626 | 5.051 | 0.148 | 0.129 |
| Seed1023 | Random-k | 0.750 | 0.626 | 3.632 | 1.699 | 5.331 | 0.141 | 0.117 |
| Seed1117 | No diffusion | 0.812 | 0.673 | 3.527 | 1.551 | 5.078 | 0.160 | 0.132 |
| Seed1117 | Broadcast | 0.812 | 0.680 | 4.106 | 1.914 | 6.020 | 0.135 | 0.113 |
| Seed1117 | Top-k similarity | 0.750 | 0.628 | 4.267 | 1.759 | 6.026 | 0.124 | 0.104 |
| Seed1117 | Random-k | 0.812 | 0.671 | 4.367 | 1.838 | 6.205 | 0.131 | 0.108 |

| Row | Avg raw | Avg judge | Avg cost | Avg cache $ | Avg Raw/$ | Avg Judge/$ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 0.812 | 0.719 | 5.412 | 1.621 | 0.150 | 0.133 |
| Broadcast | 0.771 | 0.663 | 6.018 | 1.947 | 0.128 | 0.110 |
| Top-k similarity | 0.792 | 0.650 | 5.474 | 1.730 | 0.146 | 0.119 |
| Random-k | 0.812 | 0.687 | 5.577 | 1.769 | 0.147 | 0.124 |

The weighted cost view sharpens the penalty for executor-heavy rows. Seed42
random-k remains the strongest result. Seed42 top-k remains second. Seed1117
clearly favors no diffusion, because top-k and random-k spend much more
executor input/output without enough reward gain. Seed1023 top-k is cost
competitive with no diffusion because it is cheaper, not because it wins on raw
reward.

Adding cache-read cost lowers all reward-per-dollar values by roughly 29-34%.
It does not change the top-line ranking, but it makes broadcast worse: broadcast
has the highest average weighted cost and the lowest average Raw/$.

## Diffusion Artifact Use

| Batch | Row | Selected/rendered artifacts | Eligible artifacts | Transfer context tokens | Regressions after context | Budget violations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Seed42 | Broadcast | 61 / 61 | 504 | 11,051 | 1 | 0 |
| Seed42 | Top-k similarity | 42 / 42 | 490 | 8,044 | 0 | 0 |
| Seed42 | Random-k | 62 / 62 | 504 | 11,111 | 0 | 0 |
| Seed1023 | Broadcast | 64 / 64 | 504 | 11,338 | 2 | 0 |
| Seed1023 | Top-k similarity | 39 / 39 | 490 | 7,395 | 1 | 0 |
| Seed1023 | Random-k | 61 / 61 | 483 | 10,924 | 0 | 0 |
| Seed1117 | Broadcast | 62 / 62 | 504 | 11,290 | 2 | 0 |
| Seed1117 | Top-k similarity | 42 / 42 | 490 | 8,907 | 2 | 0 |
| Seed1117 | Random-k | 64 / 64 | 476 | 11,406 | 0 | 0 |

The artifact pipeline is operational in all seed batches: artifacts are
selected, rendered, and kept within budget. The main difference is whether the
selected context turns into durable reward. Seed42 random-k and top-k are the
clearest positive cases. Seed1023 and seed1117 show that more context does not
reliably improve reward or cost efficiency.

## Context Budget Binding

This table reports post-warmup average utilization of the available transfer,
same-task, and total-prior budgets. A hard hit is a run using at least 98% of
the relevant cap.

| Batch | Row | Transfer fill | Transfer hard hits | Same-task fill | Same-task hard hits | Total-prior fill | Total hard hits |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Seed42 | No diffusion | 11.2% | 0 / 16 | 3.2% | 0 / 16 | 14.4% | 0 / 16 |
| Seed42 | Broadcast | 17.3% | 0 / 16 | 3.4% | 0 / 16 | 20.6% | 0 / 16 |
| Seed42 | Top-k similarity | 14.1% | 0 / 16 | 2.8% | 0 / 16 | 16.9% | 0 / 16 |
| Seed42 | Random-k | 17.4% | 0 / 16 | 2.8% | 0 / 16 | 20.2% | 0 / 16 |
| Seed1023 | No diffusion | 11.4% | 0 / 16 | 3.0% | 0 / 16 | 14.4% | 0 / 16 |
| Seed1023 | Broadcast | 17.7% | 0 / 16 | 3.3% | 0 / 16 | 21.0% | 0 / 16 |
| Seed1023 | Top-k similarity | 13.4% | 0 / 16 | 3.7% | 0 / 16 | 17.1% | 0 / 16 |
| Seed1023 | Random-k | 17.1% | 0 / 16 | 3.6% | 0 / 16 | 20.6% | 0 / 16 |
| Seed1117 | No diffusion | 51.8% | 0 / 16 | 39.0% | 0 / 16 | 48.6% | 0 / 16 |
| Seed1117 | Broadcast | 78.4% | 0 / 16 | 36.0% | 0 / 16 | 67.8% | 0 / 16 |
| Seed1117 | Top-k similarity | 61.9% | 0 / 16 | 43.0% | 0 / 16 | 57.1% | 0 / 16 |
| Seed1117 | Random-k | 79.2% | 0 / 16 | 46.8% | 0 / 16 | 71.1% | 0 / 16 |

No row in any seed is hard budget-bound post-warmup. Seed1117 uses much more of
its available transfer and same-task budgets because its caps are smaller. That
extra utilization does not translate into better dollar-cost efficiency for
diffusion rows.

## Raw Reward By Task

Seed42 is the selective-diffusion success case. Top-k and random-k preserve
high iteration-1 and iteration-2 performance, while no diffusion slips from 7/8
passing tasks in iteration 1 to 6/8 in iteration 2. Random-k is the most stable
seed42 row, passing 7/8 tasks in every iteration.

Seed1023 favors no diffusion. The control passes
`campus-budget-at-risk-calc`, `factory-output-at-risk-calc`,
`hospital-capacity-at-risk-calc`, `weighted-cloud-reliability-calc`, and
`weighted-port-throughput-calc` in every iteration. Diffusion rows recover in
iteration 2 but introduce earlier misses on otherwise tractable tasks.
`weighted-hospital-bedflow-calc` remains a hard zero across seed1023 rows.

Seed1117 favors broadcast on all-iteration raw reward but not on durable
post-warmup efficiency. Broadcast passes every task in iteration 1, including
one `weighted-hospital-bedflow-calc` success, then falls to 5/8 in iteration 2.
No diffusion and random-k both finish at 7/8 in iteration 2, but no diffusion
does so with lower adjusted-token and weighted-dollar cost.

## Interpretation and Conclusion

The current evidence supports a conditional claim, not a universal claim:
diffusion is seed-sensitive.

1. Seed42 random-k is the best overall row: best post-warmup raw efficiency,
   best weighted Raw/$, best weighted Judge/$, and strong raw reward.
2. Seed42 top-k is also positive and remains second by weighted Raw/$.
3. Seed1023 does not replicate the seed42 reward gain. Top-k is cost
   competitive only because it is cheaper.
4. Seed1117 shows a front-loaded broadcast reward gain, but no diffusion is the
   better post-warmup and weighted-cost policy.
5. Broadcast is consistently weak under cost accounting. It adds context and
   cache-read volume but does not return enough reward.

Across the three seed batches, no diffusion is the most stable baseline.
Selective diffusion can help, but only when it reduces executor burden or
selects context that transfers cleanly. Seed1117 is the strongest warning that
high transfer-context utilization alone is not evidence of useful diffusion.

## Anomalies and Caveats

| Issue | Impact |
| --- | --- |
| Old Harbor `result.json` `agent_result` token fields are zero in persisted runs. | Executor tokens must be recovered from `agent/hermes-session.jsonl` for seed42 and seed1023, and for the seed1117 broadcast env-failure row. |
| Seed1117 broadcast has one environment failure in `hospital-capacity-at-risk-calc`, iteration 2. | Raw reward is treated as zero over the 24-task denominator; judge mean excludes the missing judge row according to the persisted summary convention. |
| Diffusion rows may include a synthetic `__coevolution__` metrics row. | Synthetic rows are excluded from task denominators and token tables. |
| Cache-read tokens are large. | They are excluded from adjusted-token efficiency and included only in the weighted dollar-cost section. |
| Three seed batches are descriptive evidence. | Seed42 is promising, but seed1023 and seed1117 show the effect is not robust enough for a broad causal claim. |
| Broadcast/random-k graph labels can differ between config and runtime metrics. | Interpret row identity by diffusion policy first and graph label second. |

## Accounting Guidance

Use both adjusted-token and weighted-cost accounting, but keep their claims
separate.

| Question | Accounting view |
| --- | --- |
| Which policy is most efficient end-to-end by token volume? | Use adjusted total tokens, raw efficiency, and judge efficiency. |
| Which policy is most efficient under model-price asymmetry? | Use weighted dollar-cost efficiency with executor input/output and cache-read split. |
| Did prior context bind the run? | Use transfer, same-task, and total-prior binding rates. |
| Did diffusion help beyond ordinary cross-task prior? | Use post-warmup reward and cost deltas, not all-iteration reward alone. |
| Did the graph or routing matter? | Compare top-k, random-k, and broadcast at similar transfer-token scale. |

For future WRA comparisons, keep reporting both post-warmup adjusted-token
efficiency and cache-aware weighted dollar-cost efficiency. The robust claim is
seed-sensitive diffusion benefit, with seed42 as the positive case and seed1117
as the cautionary cost case.
