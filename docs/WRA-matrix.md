# WRA Matrix: Two-Batch Diffusion Comparison

## Scope

This note compares two four-row batches for the `Weighted-Risk-Assessment`
family. The first batch is the `manual-matrix` set, and the second batch is the
`infra1` set. All rows use fixed skills, so the main changing factor is the
diffusion policy.

Only experiment directories at or after
`20260608-122956-manual-matrix-skill-none-diffusion-none` are included.

| Batch | Row | Experiment directory | Diffusion policy |
| --- | --- | --- | --- |
| Manual matrix | No diffusion | `20260608-122956-manual-matrix-skill-none-diffusion-none` | `none` |
| Manual matrix | Broadcast | `20260608-173727-manual-matrix-skill-none-diffusion-broadcast` | `capped_broadcast` |
| Manual matrix | Top-k similarity | `20260608-233458-manual-matrix-skill-none-diffusion-top-k-similarity` | `top_k_similarity` |
| Manual matrix | Random-k | `20260609-182541-manual-matrix-skill-none-diffusion-random-k` | `random_k` |
| Infra1 | No diffusion | `20260612-005909-wra-matrix-skill-none-diffusion-none-infra1` | `none` |
| Infra1 | Broadcast | `20260612-041549-wra-matrix-skill-none-diffusion-capped-broadcast-infra1` | `capped_broadcast` |
| Infra1 | Top-k similarity | `20260612-132411-wra-matrix-skill-none-diffusion-top-k-similarity-infra1` | `top_k_similarity` |
| Infra1 | Random-k | `20260612-095835-wra-matrix-skill-none-diffusion-random-k-infra1` | `random_k` |

## Aggregate Outcome

Raw reward is verifier reward. `Raw mean` treats environment failures or missing
raw rewards as zero. Judge reward is the post-hoc rubric reward from
`artifacts/judge_rewards.jsonl`.

| Batch | Row | Raw pass count | Raw mean | Judge mean | Iter 0 raw | Iter 1 raw | Iter 2 raw |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Manual | No diffusion | 18 / 24 | 0.750 | 0.669 | 0.625 | 0.750 | 0.875 |
| Manual | Broadcast | 15 / 24 | 0.625 | 0.562 | 0.500 | 0.625 | 0.750 |
| Manual | Top-k similarity | 12 / 24 | 0.500 | 0.506 | 0.625 | 0.750 | 0.125 |
| Manual | Random-k | 15 / 24 | 0.625 | 0.575 | 0.375 | 0.750 | 0.750 |
| Infra1 | No diffusion | 17 / 24 | 0.708 | 0.653 | 0.500 | 0.875 | 0.750 |
| Infra1 | Broadcast | 15 / 24 | 0.625 | 0.584 | 0.375 | 0.875 | 0.625 |
| Infra1 | Top-k similarity | 20 / 24 | 0.833 | 0.681 | 0.750 | 0.875 | 0.875 |
| Infra1 | Random-k | 21 / 24 | 0.875 | 0.698 | 0.875 | 0.875 | 0.875 |

The two batches disagree in the direction of the diffusion effect.

Manual matrix is negative for diffusion. No diffusion has the best raw mean,
best efficiency, and strongest final-iteration result. Broadcast and random-k
recover to 6/8 passing tasks in iteration 2 but remain below the control.
Top-k similarity collapses in iteration 2.

Infra1 is positive for selective diffusion. Random-k and top-k both outperform
the no-diffusion control on raw reward, judge reward, final-iteration reward,
and recovered-token efficiency. Broadcast remains weak in both batches.

## Token Accounting

The original `metrics.jsonl` rows counted orchestration-side LLM calls but not
the executor-side Hermes tokens, because Harbor `result.json` had
`agent_result.n_input_tokens = 0` and `agent_result.n_output_tokens = 0`.
The executor tokens are recoverable from each trial's
`agent/hermes-session.jsonl`.

For token-adjusted comparisons:

`executor_token = sum(input_tokens + output_tokens)` from Hermes session logs.

`adjusted_total = current total_tokens + executor_token`.

`cache_read` is reported only for audit. It is not included in
`executor_token`, `adjusted_total`, or reward adjustment.

`Efficiency` below is raw mean per 100k adjusted tokens.

| Batch | Row | Current total | Executor token | Adjusted total | Cache read audit | Efficiency |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Manual | No diffusion | 237,941 | 689,404 | 927,345 | 4,572,160 | 0.081 |
| Manual | Broadcast | 295,191 | 558,930 | 854,121 | 5,814,272 | 0.073 |
| Manual | Top-k similarity | 297,144 | 657,639 | 954,783 | 4,846,592 | 0.052 |
| Manual | Random-k | 301,166 | 578,458 | 879,624 | 4,360,704 | 0.071 |
| Infra1 | No diffusion | 235,433 | 694,872 | 930,305 | 5,211,648 | 0.076 |
| Infra1 | Broadcast | 288,463 | 670,417 | 958,880 | 6,046,720 | 0.065 |
| Infra1 | Top-k similarity | 269,017 | 532,978 | 801,995 | 5,204,480 | 0.104 |
| Infra1 | Random-k | 273,744 | 541,863 | 815,607 | 5,409,280 | 0.107 |

This corrected token picture strengthens the infra1 result. Top-k and random-k
are not merely higher reward; they are also cheaper than no diffusion after
executor tokens are recovered. Random-k is the best row by raw mean and adjusted
efficiency in infra1, with top-k close behind. Broadcast is worse than
no-diffusion in both reward and adjusted efficiency.

In the manual batch, the corrected token picture does not rescue diffusion.
Broadcast and random-k use fewer adjusted tokens than no diffusion, but their
reward loss is large enough that their efficiency remains lower. Top-k is both
lower reward and slightly more expensive than no diffusion.

## Diffusion Artifact Use

| Batch | Row | Selected/rendered artifacts | Eligible artifacts | Diffusion context tokens | Regressions after context | Budget violations |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Manual | Broadcast | 48 / 48 | 504 | 6,953 | 3 | 0 |
| Manual | Top-k similarity | 42 / 42 | 511 | 6,350 | 5 | 0 |
| Manual | Random-k | 48 / 48 | 483 | 7,570 | 2 | 0 |
| Infra1 | Broadcast | 61 / 61 | 504 | 11,051 | 1 | 0 |
| Infra1 | Top-k similarity | 42 / 42 | 490 | 8,044 | 0 | 0 |
| Infra1 | Random-k | 62 / 62 | 504 | 11,111 | 0 | 0 |

The artifact pipeline is operational in both batches: artifacts are selected,
rendered, and kept within budget. The difference is quality and interaction
with the run. Manual top-k selected a brittle neighborhood and had the largest
number of regressions after context. Infra1 top-k selected fewer artifacts than
broadcast or random-k but had zero regressions, suggesting the routed context
was cleaner or better aligned. Random-k also had zero regressions in infra1 and
the best reward outcome.

## Raw Reward By Task

The manual batch favors the control. No diffusion passes consistently on
`campus-budget-at-risk-calc`, `factory-output-at-risk-calc`,
`weighted-cloud-reliability-calc`, and `weighted-port-throughput-calc`, while
diffusion rows introduce more task-level instability. The persistent hard case
is `weighted-hospital-bedflow-calc`, which rarely benefits from any policy.

The infra1 batch changes the task-level pattern. Top-k and random-k preserve
high iteration-1 and iteration-2 performance, while no diffusion slips from
7/8 passing tasks in iteration 1 to 6/8 in iteration 2. Random-k is the most
stable infra1 row, passing 7/8 tasks in every iteration. Top-k is nearly as
strong, reaching 7/8 in iterations 1 and 2.

## Interpretation

The current evidence supports a conditional claim, not a universal claim:
diffusion is not automatically helpful, but the infra1 implementation makes
top-k similarity and random-k look beneficial for WRA.

Policy ranking after recovered executor tokens:

1. Infra1 random-k: best raw mean, best adjusted efficiency, zero recorded
   regressions after diffusion context.
2. Infra1 top-k similarity: strong raw mean, lowest adjusted total, zero
   recorded regressions after diffusion context.
3. No diffusion controls: best in the manual batch, but beaten by top-k and
   random-k in infra1.
4. Broadcast: consistently weak. It adds context but does not improve reward or
   efficiency in either batch.
5. Manual top-k similarity: worst row because of the iteration-2 collapse.

The batch comparison is also important. The no-diffusion row is similar across
batches, moving from 0.750 raw mean in manual to 0.708 in infra1. The large
change is in selective diffusion: top-k improves from 0.500 to 0.833, and
random-k improves from 0.625 to 0.875. That makes infra1 the first WRA batch
where diffusion has a credible positive reward and efficiency signal.

## Anomalies and Caveats

| Issue | Impact |
| --- | --- |
| The old Harbor `result.json` `agent_result` token fields are zero in persisted runs. | Past `metrics.jsonl` rows undercount executor tokens unless recovered from `agent/hermes-session.jsonl`. |
| One manual top-k trial is missing a persisted Hermes session. | Manual top-k `executor_token` and `adjusted_total` are lower bounds; this does not change its negative conclusion. |
| Cache-read tokens are large. | They should be audited separately from `executor_token`; including them would answer a different cost question. |
| Two batches are still descriptive evidence. | The infra1 improvement is promising but not enough by itself for a causal conclusion. More seeds/families or a paired token-controlled design are still needed. |
| Broadcast/random-k graph labels can differ between config and runtime metrics. | Interpret row identity by diffusion policy first and graph label second. |

## Next Step

Use the infra1 token accounting path for future runs: parse Hermes session
`input_tokens` and `output_tokens` into executor token usage, surface cache-read
as audit metadata, and compare policies on adjusted total tokens. The next
experiment should run paired seeds across at least three task families so that
the positive top-k/random-k signal can be separated from batch noise.
