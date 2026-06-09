# WRA Manual Matrix: Diffusion Policy Comparison

## Scope

This note summarizes the four recent `manual-matrix` runs for the
`Weighted-Risk-Assessment` task family. All four rows use
`condition_name = "learned_mediator"` and fixed skills, so the main changing
factor is the diffusion policy.

| Row | Experiment directory | Diffusion policy | Metrics graph label |
| --- | --- | --- | --- |
| No diffusion | `20260608-122956-manual-matrix-skill-none-diffusion-none` | `none` | `null` |
| Broadcast | `20260608-173727-manual-matrix-skill-none-diffusion-broadcast` | `capped_broadcast` | `broadcast` |
| Top-k similarity | `20260608-233458-manual-matrix-skill-none-diffusion-top-k-similarity` | `top_k_similarity` | `precomputed_similarity` |
| Random-k | `20260609-182541-manual-matrix-skill-none-diffusion-random-k` | `random_k` | `broadcast` |

The result is negative for diffusion: none of the diffusion policies improves
over the no-diffusion control. The no-diffusion run has the best aggregate raw
reward and best aggregate judge reward.

## Aggregate Reward Summary

Raw reward is verifier reward. `raw mean, null=0` treats environment failures
as failed tasks. Judge reward is the post-hoc rubric reward from
`artifacts/judge_rewards.jsonl`.

| Row | Raw pass count | Raw mean, null=0 | Judge mean | Iter 0 raw | Iter 1 raw | Iter 2 raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| No diffusion | 18 / 24 | 0.750 | 0.669 | 0.625 | 0.750 | 0.875 |
| Broadcast | 15 / 24 | 0.625 | 0.562 | 0.500 | 0.625 | 0.750 |
| Top-k similarity | 12 / 24 | 0.500 | 0.506 | 0.625 | 0.750 | 0.125 |
| Random-k | 15 / 24 | 0.625 | 0.575 | 0.375 | 0.750 | 0.750 |

No diffusion improves steadily across iterations and finishes at 7/8 passing
tasks in iteration 2. Broadcast and random-k recover to 6/8 but never surpass
the control. Top-k similarity reaches 6/8 in iteration 1, then collapses to
1/8 in iteration 2.

## Raw Reward by Task

Each cell is the raw verifier reward trajectory `iter0/iter1/iter2`. `E` means
environment failure or missing raw reward.

| Task | No diffusion | Broadcast | Top-k similarity | Random-k |
| --- | --- | --- | --- | --- |
| `api-sla-at-risk-calc` | E/1/1 | 1/1/1 | 1/1/0 | 0/1/1 |
| `campus-budget-at-risk-calc` | 1/1/1 | 1/0/1 | 0/1/0 | 0/1/1 |
| `factory-output-at-risk-calc` | 1/1/1 | 0/1/1 | 1/1/0 | 1/1/1 |
| `hospital-capacity-at-risk-calc` | 0/1/1 | 1/1/1 | 1/1/0 | 0/1/1 |
| `weighted-campus-energy-balance-calc` | 1/0/1 | 1/0/1 | 1/1/1 | 0/1/1 |
| `weighted-cloud-reliability-calc` | 1/1/1 | 0/1/1 | 0/1/0 | 1/1/0 |
| `weighted-hospital-bedflow-calc` | 0/0/0 | 0/1/0 | 0/0/E | 0/0/0 |
| `weighted-port-throughput-calc` | 1/1/1 | 0/0/0 | 1/0/0 | 1/0/1 |

The strongest control tasks are `campus-budget`, `factory-output`,
`weighted-cloud-reliability`, and `weighted-port-throughput`, which all pass
consistently or recover under no diffusion. The persistent hard case is
`weighted-hospital-bedflow-calc`, which fails in every no-diffusion and
random-k iteration and only briefly passes in broadcast iteration 1.

## Diffusion Artifact Use

| Row | Selected artifacts | Artifact types | Source raw profile | Dominant source tasks | Source judge mean |
| --- | ---: | --- | --- | --- | ---: |
| Broadcast | 48 | 16 debug hints, 16 mediator summaries, 16 run outcomes | 45 failed, 3 passed | `weighted-port-throughput` 42, `weighted-hospital-bedflow` 6 | 0.232 |
| Top-k similarity | 42 | 14 debug hints, 14 mediator summaries, 10 run outcomes, 4 regression warnings | 27 failed, 15 passed | `weighted-port-throughput` 24, `weighted-hospital-bedflow` 12, `weighted-cloud-reliability` 6 | 0.361 |
| Random-k | 48 | 19 run outcomes, 18 mediator summaries, 11 debug hints | 25 failed, 23 passed | `weighted-campus-energy-balance` 11, `campus-budget` 8, `hospital-capacity` 8, `weighted-cloud-reliability` 7 | 0.466 |

Broadcast is heavily skewed toward recent artifacts from
`weighted-port-throughput-calc`, and almost all selected source artifacts come
from failed source runs. This explains why broadcast adds context without
improving reward: it mostly rebroadcasts low-quality failure evidence.

Top-k similarity is less extreme than broadcast but still concentrates on the
same brittle neighborhood: `weighted-port-throughput`, `weighted-hospital-bedflow`,
and `weighted-cloud-reliability`. The policy does not route broadly useful
lessons; it routes local artifacts from tasks that are themselves unstable.
The iteration-2 collapse is consistent with over-amplifying misleading or
non-transferable local context.

Random-k is the most source-balanced policy and has the highest selected-source
judge mean among diffusion rows, but it still does not beat no diffusion.
Balanced random sampling reduces the recency/similarity concentration problem,
yet it also injects mixed-quality context: 25 selected artifacts come from
failed source runs and 23 from passing source runs. That is not reliable enough
to improve the planner over the fixed no-diffusion control.

## Diffusion Usefulness

The main implication is that cross-task diffusion is operationally working but
not useful in this WRA manual matrix. The renderer inserts artifacts, token
budgets are used, and target tasks receive prior context after iteration 0.
However, the added context does not improve final task performance.

The likely reasons are:

1. Artifact selection quality is too low. Broadcast and top-k select many
   artifacts from failed source tasks.
2. Source concentration is severe. Broadcast and top-k repeatedly route a small
   set of tasks into many targets, especially `weighted-port-throughput-calc`.
3. Similarity is not enough. The top-k graph finds related tasks, but related
   failure modes can be harmful when the artifacts are not filtered for
   correctness.
4. Random selection is safer but still noisy. Random-k avoids source collapse,
   but it mixes passing and failing source evidence without enough quality
   control.
5. No diffusion already has strong within-run recovery. The control improves
   from 5/8 or 6/8 effective pass rate to 7/8 by iteration 2, leaving little
   room for low-quality diffusion to help.

The practical conclusion is to treat these diffusion policies as negative
baselines. Future diffusion should probably be quality-gated before selection:
prefer passing source runs, downweight low judge-reward artifacts, avoid repeated
source-task dominance, and prevent failed debug hints from being broadcast
unless the target specifically needs a known failure signature.

## Anomalies and Caveats

| Row | Anomaly | Impact |
| --- | --- | --- |
| No diffusion | `api-sla-at-risk-calc` has an iteration-0 environment failure and no raw reward. | Aggregate raw mean should either count it as failure or report non-null mean separately. |
| Top-k similarity | `weighted-hospital-bedflow-calc` has an iteration-2 environment failure and no raw reward. | The iteration-2 collapse remains severe even before counting the env failure as zero. |
| Broadcast | Config says `graph = "none"`, while metrics record `diffusion_graph = "broadcast"`. | Interpret as runtime graph label, not a config mismatch in reward behavior. |
| Top-k similarity | Config says `graph = "task_similarity"`, while metrics record `precomputed_similarity`. | Likely naming difference between config and runtime graph snapshot. |
| Random-k | Config says `graph = "none"`, while metrics record `broadcast`. | Runtime appears to use a broadcast-style candidate graph with random-k selection. |

These anomalies do not change the main conclusion. The corrected random-k row
uses `condition_name = "learned_mediator"`, so the four manual rows are now a
reasonable descriptive matrix. The evidence still favors no diffusion.
