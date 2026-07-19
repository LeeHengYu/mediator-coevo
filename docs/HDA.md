# HDA Seed42 Diffusion Batch

## Scope

This note compares the four seed42 rows for the
`HWPX-Document-Automation` family. All rows use fixed skills, so the main
changing factor is diffusion policy. Row0 is the recovered no-diffusion
baseline.

| Row | Experiment directory | Baseline preset | Diffusion policy |
| --- | --- | --- | --- |
| No diffusion | `20260619-100320-HDA-seed42-row0` | `skill_none_diffusion_none` | `none` |
| Broadcast | `20260617-152938-HDA-seed42-row1` | `skill_none_capped_broadcast` | `capped_broadcast` |
| Random-k | `20260617-203909-HDA-seed42-row2` | `skill_none_random_k` | `random_k` |
| Top-k similarity | `20260618-222638-HDA-seed42-row3` | `skill_none_top_k_similarity` | `top_k_similarity` |

## Aggregate Outcome

Raw reward is verifier reward. Raw mean treats environment failures as zero
over the 24 task-run denominator. Judge mean follows the persisted summary
convention: it averages judged runs, so env-failure rows without judge records
are not included in that mean.

| Row | Policy | Raw pass | Raw mean | Judge mean | Iter 0 raw | Iter 1 raw | Iter 2 raw | Env/missing |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | `none` | 19 / 24 | 0.792 | 0.638 | 0.875 | 0.750 | 0.750 | 0 / 0 |
| Broadcast | `capped_broadcast` | 21 / 24 | 0.875 | 0.708 | 0.750 | 1.000 | 0.875 | 1 / 0 |
| Random-k | `random_k` | 22 / 24 | 0.917 | 0.667 | 0.875 | 1.000 | 0.875 | 0 / 0 |
| Top-k similarity | `top_k_similarity` | 18 / 24 | 0.750 | 0.672 | 0.625 | 0.750 | 0.875 | 2 / 0 |

Random-k is the strongest row on raw reward and adjusted-token efficiency.
Broadcast has the highest judge mean and ties random-k after warmup on raw pass
count, but it has one timeout and higher token cost. Top-k similarity improves
by iteration 2, but its aggregate result is dragged down by two environment
failures and very high executor token use.

## Post-Warmup Efficiency

Iteration 0 is the cold-start pass. The post-warmup view filters to
`iteration >= 1` and reports reward mean per 100k total tokens.

| Row | Runs | Raw pass | Raw mean | Judge mean | Tokens | Raw/100k | Judge/100k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 16 | 12 / 16 | 0.750 | 0.624 | 807,995 | 0.093 | 0.077 |
| Broadcast | 16 | 15 / 16 | 0.938 | 0.716 | 971,706 | 0.096 | 0.074 |
| Random-k | 16 | 15 / 16 | 0.938 | 0.639 | 726,786 | 0.129 | 0.088 |
| Top-k similarity | 16 | 13 / 16 | 0.812 | 0.660 | 1,430,650 | 0.057 | 0.046 |

Random-k is the only clear efficiency win: same post-warmup raw mean as
broadcast, fewer tokens than no diffusion, and the best raw and judge reward
per 100k tokens.

## Raw Reward By Task

Cells are `iter0/iter1/iter2`; `E` is an environment failure and counts as zero
in the aggregate raw mean.

| Task | No diffusion | Broadcast | Random-k | Top-k similarity |
| --- | ---: | ---: | ---: | ---: |
| `hwpx-clinic-intake-summary` | 1/1/1 | 1/1/1 | 1/1/1 | 0/0/1 |
| `hwpx-event-announcement` | 1/1/1 | 1/1/1 | 1/1/1 | 1/1/1 |
| `hwpx-inventory-report` | 1/1/1 | 1/1/1 | 1/1/1 | 1/1/1 |
| `hwpx-project-proposal` | 1/0/0 | 0/1/1 | 1/1/0 | 1/0/1 |
| `hwpx-renewal-playbook-update` | 1/1/1 | 1/1/E | 1/1/1 | E/1/1 |
| `hwpx-safety-audit-brief` | 0/0/0 | 0/1/1 | 0/1/1 | 0/1/E |
| `hwpx-supplier-contact-sheet` | 1/1/1 | 1/1/1 | 1/1/1 | 1/1/1 |
| `hwpx-training-feedback` | 1/1/1 | 1/1/1 | 1/1/1 | 1/1/1 |

Diffusion mainly helps `hwpx-safety-audit-brief` and
`hwpx-project-proposal`, the two tasks that the no-diffusion row cannot hold
after cold start. Random-k gets those gains without introducing environment
failures.

## Diffusion Artifact Use

| Row | Selected/rendered artifacts | Eligible artifacts | Transfer context tokens | Regressions after context | Budget violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 0 / 0 | 0 | 9,625 | 0 | 0 |
| Broadcast | 64 / 64 | 490 | 11,552 | 0 | 0 |
| Random-k | 62 / 62 | 504 | 11,196 | 1 | 0 |
| Top-k similarity | 43 / 43 | 462 | 9,054 | 1 | 0 |

No-diffusion still has transfer tokens because learned-mediator cross-task
prior is active; it just has no diffusion artifacts. Broadcast and random-k
render similar artifact volume. Top-k renders less context, but that does not
translate into lower total cost because executor tokens dominate its row.

Planner, Executor, and Mediator use the same fixed prompt-injected skills in all
four rows, so row differences come from routed context and execution effects.

## Token Accounting

| Row | Orchestration tokens | Executor tokens | Total tokens | Cache-read audit | Raw/100k | Judge/100k |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 245,708 | 902,758 | 1,148,466 | 8,267,104 | 0.069 | 0.056 |
| Broadcast | 242,456 | 1,083,549 | 1,326,005 | 7,930,176 | 0.066 | 0.053 |
| Random-k | 253,010 | 798,044 | 1,051,054 | 9,378,464 | 0.087 | 0.063 |
| Top-k similarity | 236,146 | 1,510,934 | 1,747,080 | 7,548,288 | 0.043 | 0.038 |

Post-warmup role split:

| Row | Planner | Mediator | Judge | Executor | Total | Executor share |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 83,101 | 53,804 | 38,490 | 632,600 | 807,995 | 78.3% |
| Broadcast | 73,689 | 37,807 | 35,544 | 811,362 | 971,706 | 83.5% |
| Random-k | 76,167 | 46,446 | 39,440 | 547,090 | 726,786 | 75.3% |
| Top-k similarity | 72,176 | 41,275 | 35,922 | 1,265,669 | 1,430,650 | 88.5% |

Executor cost dominates every row. Top-k similarity is especially expensive:
its post-warmup executor share is 88.5%, and its total token use is 66% higher
than random-k while producing lower raw reward.

## Context Budget Binding

Post-warmup average utilization of available transfer, same-task, and
total-prior budgets:

| Row | Transfer fill | Transfer hard hits | Same-task fill | Same hard hits | Total fill | Total hard hits | Violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 66.8% | 0 / 16 | 47.9% | 0 / 16 | 62.1% | 0 / 16 | 0 |
| Broadcast | 80.2% | 0 / 16 | 42.1% | 0 / 16 | 70.7% | 0 / 16 | 0 |
| Random-k | 77.8% | 0 / 16 | 47.8% | 0 / 16 | 70.2% | 0 / 16 | 0 |
| Top-k similarity | 62.9% | 0 / 16 | 46.5% | 0 / 16 | 58.8% | 0 / 16 | 0 |

No row is hard budget-bound. The failures are not explained by prior-context
truncation. Broadcast and random-k use more transfer context than no diffusion,
but random-k converts it into better efficiency.

## Anomalies And Caveats

- Row0 was recovered after interruption. Its final state has 24 scored rows and
  0 environment failures.
- Broadcast has one environment failure:
  `hwpx-renewal-playbook-update` iteration 2 timed out during agent execution.
- Top-k similarity has two environment failures:
  `hwpx-renewal-playbook-update` iteration 0 timed out, and
  `hwpx-safety-audit-brief` iteration 2 failed during Hermes setup with exit
  code 128.
- Random-k has no environment failures.
- Judge means for broadcast and top-k exclude their unjudged env-failure rows;
  raw means count those rows as zero.

## Discussion

Seed42 HDA is positive for diffusion, but only for the lazy kind of diffusion:
random-k. Random-k has the best raw mean, best post-warmup efficiency, no
environment failures, and fixes most of the no-diffusion failures on
`hwpx-safety-audit-brief` and `hwpx-project-proposal`.

Broadcast also improves raw reward, but it spends more tokens and has an agent
timeout. It is useful, but less efficient than random-k. Top-k similarity is
not useful in this batch: it renders fewer artifacts, but it is the most
expensive row and has the worst raw efficiency.

The practical read is narrow: diffusion context can help HWPX task transfer,
but the best policy in this seed is random-k, not the more selective top-k
similarity policy.

Bottom line: for HDA seed42, random-k is the row to keep. Broadcast is a
reasonable second-best on raw score, and top-k similarity is too costly for its
reward.
