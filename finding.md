Source: data/experiments/20260603-095843-fix_env_test

Last run before removing per task veto 

Mean of validation task runs saw clear increases, but 1 task regresses so skill update was rejected. 

Policy: random_k

---

Weighted-Risk-Assessment 5-task run findings

Run:
data/experiments/20260603-214227-wra-5-random-k-skill-updates-prefixed

Configuration:
- Tasks: 5 Weighted-Risk-Assessment tasks
- Iterations: 3
- Skill updates: all
- Diffusion: enabled
- Diffusion policy: random_k
- Diffusion graph: none
- Mode: non-verbose

Overall scoring:
- Scored tasks: 15/15
- Environment failures: 0
- Verifier mean: 0.4666666667
- Verifier median: 0.0
- Judge mean: 0.4645333333
- Judge median: 0.2
- Total tokens: 411371
- Judge tokens: 54801

V/J reward table

V = verifier reward, J = judge reward.

| Task | Iter 1 V/J | Iter 2 V/J | Iter 3 V/J | Verifier mean | Judge mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| weighted-cloud-reliability-calc | 1.00 / 0.808 | 1.00 / 0.775 | 1.00 / 0.855 | 1.000 | 0.813 |
| weighted-campus-energy-balance-calc | 1.00 / 0.845 | 0.00 / 0.185 | 1.00 / 0.785 | 0.667 | 0.605 |
| api-sla-at-risk-calc | 0.00 / 0.175 | 0.00 / 0.175 | 1.00 / 0.750 | 0.333 | 0.367 |
| weighted-port-throughput-calc | 0.00 / 0.160 | 0.00 / 0.155 | 1.00 / 0.760 | 0.333 | 0.358 |
| weighted-hospital-bedflow-calc | 0.00 / 0.200 | 0.00 / 0.155 | 0.00 / 0.185 | 0.000 | 0.180 |

Verifier pass rate by iteration:

| Iteration | Passing tasks | Pass rate | Notes |
| ---: | ---: | ---: | --- |
| 1 | 2/5 | 40% | Cloud and campus passed. |
| 2 | 1/5 | 20% | Cloud passed; campus regressed under random_k context. |
| 3 | 4/5 | 80% | Cloud, campus, port, and API passed; hospital remained failing. |

Validation rewards for rejected skill-update batches

All four advisor-approved executor skill-update batches were rejected by validation with reason
`validation: mean_not_improved`. No skill updates were committed.

| Batch | Iteration | Decision | Current mean | Candidate mean | Delta | Reason |
| --- | ---: | --- | ---: | ---: | ---: | --- |
| coevo-iter-0000-batch-70363f5cbb | 0 | rejected | 0.1222 | 0.1033 | -0.0188 | validation: mean_not_improved |
| coevo-iter-0001-batch-c6a701fd73 | 1 | rejected | 0.1800 | 0.1900 | +0.0100 | validation: mean_not_improved |
| coevo-iter-0001-batch-aeb3330987 | 1 | rejected | 0.1770 | 0.1658 | -0.0112 | validation: mean_not_improved |
| coevo-iter-0002-batch-3cd3c8c5c1 | 2 | rejected | 0.1848 | 0.1560 | -0.0288 | validation: mean_not_improved |

Per-task validation reward changes:

| Batch | Task | Current | Candidate | Delta | Direction |
| --- | --- | ---: | ---: | ---: | --- |
| 70363f5cbb | api-sla-at-risk-calc | 0.1850 | 0.1400 | -0.0450 | regress |
| 70363f5cbb | weighted-campus-energy-balance-calc | 0.0000 | 0.0000 | +0.0000 | unchanged |
| 70363f5cbb | weighted-cloud-reliability-calc | 0.1815 | 0.1700 | -0.0115 | regress |
| c6a701fd73 | weighted-campus-energy-balance-calc | 0.1550 | 0.1700 | +0.0150 | improve |
| c6a701fd73 | weighted-cloud-reliability-calc | 0.1850 | 0.2000 | +0.0150 | improve |
| c6a701fd73 | weighted-port-throughput-calc | 0.2000 | 0.2000 | +0.0000 | unchanged |
| aeb3330987 | api-sla-at-risk-calc | 0.1530 | 0.1700 | +0.0170 | improve |
| aeb3330987 | weighted-cloud-reliability-calc | 0.1780 | 0.1775 | -0.0005 | regress |
| aeb3330987 | weighted-hospital-bedflow-calc | 0.2000 | 0.1500 | -0.0500 | regress |
| 3cd3c8c5c1 | weighted-campus-energy-balance-calc | 0.1845 | 0.1550 | -0.0295 | regress |
| 3cd3c8c5c1 | weighted-cloud-reliability-calc | 0.1700 | 0.1750 | +0.0050 | improve |
| 3cd3c8c5c1 | weighted-port-throughput-calc | 0.2000 | 0.1380 | -0.0620 | regress |

Skill-update finding:
- Skill-update generation happened, but validation blocked every proposed update.
- The only positive mean delta was batch c6a701fd73 at +0.0100, but it still did not satisfy the improvement threshold.
- The validation gate prevented regressions from being committed, especially batches with task-level drops on API, hospital, campus, and port tasks.
- The final task-score improvement therefore appears to come from run dynamics and diffusion context, not from committed skill file updates.

---

Weighted-Risk-Assessment rerun findings: contrastive diffusion focus

Run:
data/experiments/20260604-110853-wra-5-random-k-skill-updates-contrastive-20260604

Configuration:
- Tasks: 5 Weighted-Risk-Assessment tasks
- Iterations: 3
- Skill updates: all
- Diffusion: enabled
- Diffusion policy: random_k
- Diffusion graph: none
- Mode: non-verbose

Overall scoring:
- Scored tasks: 15/15
- Environment failures: 0
- Verifier mean: 0.4000
- Verifier median: 0.0000
- Judge mean: 0.4120
- Judge median: 0.1850
- Total tokens: 420502

V/J reward table

V = verifier reward, J = judge reward.

| Task | Iter 1 V/J | Iter 2 V/J | Iter 3 V/J | Verifier mean | Judge mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| weighted-cloud-reliability-calc | 0.00 / 0.200 | 1.00 / 0.820 | 1.00 / 0.750 | 0.667 | 0.590 |
| weighted-hospital-bedflow-calc | 0.00 / 0.185 | 0.00 / 0.175 | 0.00 / 0.000 | 0.000 | 0.120 |
| weighted-campus-energy-balance-calc | 0.00 / 0.185 | 1.00 / 0.785 | 0.00 / 0.155 | 0.333 | 0.375 |
| weighted-port-throughput-calc | 0.00 / 0.170 | 1.00 / 0.785 | 1.00 / 0.785 | 0.667 | 0.580 |
| api-sla-at-risk-calc | 0.00 / 0.185 | 0.00 / 0.185 | 1.00 / 0.815 | 0.333 | 0.395 |

Verifier pass rate by iteration:

| Iteration | Passing tasks | Verifier mean | Judge mean | Notes |
| ---: | ---: | ---: | ---: | --- |
| 1 | 0/5 | 0.000 | 0.185 | All tasks failed; only negative artifacts were available for later diffusion. |
| 2 | 3/5 | 0.600 | 0.550 | Cloud, campus, and port recovered under prior-iteration cross-task diffusion. |
| 3 | 3/5 | 0.600 | 0.501 | Cloud and port stayed passing; API recovered; campus regressed; hospital remained failing. |

Validation rewards for rejected skill-update batches

All advisor-approved executor skill-update batches were rejected by validation. No skill updates were committed.

| Batch | Iteration | Decision | Current mean | Candidate mean | Delta | Threshold | Reason |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| coevo-iter-0000-batch-bd6ad3b81d | 0 | rejected | 0.1633 | 0.1667 | +0.0033 | +0.0100 | mean_not_improved |
| coevo-iter-0001-batch-e60cd6258e | 1 | rejected | 0.1717 | 0.1783 | +0.0067 | +0.0100 | mean_not_improved |
| coevo-iter-0002-batch-8035ea8940 | 2 | rejected | 0.1700 | 0.1650 | -0.0050 | +0.0100 | mean_not_improved |

Per-task validation reward changes:

| Batch | Task | Current | Candidate | Delta | Direction |
| --- | --- | ---: | ---: | ---: | --- |
| bd6ad3b81d | api-sla-at-risk-calc | 0.155 | 0.150 | -0.005 | regress |
| bd6ad3b81d | weighted-cloud-reliability-calc | 0.150 | 0.150 | +0.000 | unchanged |
| bd6ad3b81d | weighted-hospital-bedflow-calc | 0.185 | 0.200 | +0.015 | improve |
| e60cd6258e | weighted-campus-energy-balance-calc | 0.180 | 0.200 | +0.020 | improve |
| e60cd6258e | weighted-cloud-reliability-calc | 0.150 | 0.200 | +0.050 | improve |
| e60cd6258e | weighted-port-throughput-calc | 0.185 | 0.135 | -0.050 | regress |
| 8035ea8940 | weighted-campus-energy-balance-calc | 0.135 | 0.170 | +0.035 | improve |
| 8035ea8940 | weighted-cloud-reliability-calc | 0.200 | 0.140 | -0.060 | regress |
| 8035ea8940 | weighted-port-throughput-calc | 0.175 | 0.185 | +0.010 | improve |

Diffusion cross-task injection

Same-iteration injection was prohibited and avoided. The diffusion log had 30 selected/rendered records and 0 records with source_iteration >= target_iteration. Every injected artifact came from a prior iteration.

Representative injected artifacts:

| Target prompt | Injected source artifact | Why it mattered |
| --- | --- | --- |
| Iter 2 weighted-port-throughput-calc | Iter 1 api-sla-at-risk-calc failure_signature, debug_hint, mediator_report_summary | API had #NAME? failures in Task!H46:L47 statistics formulas. Port then recovered from 0.00 / 0.170 in iter 1 to 1.00 / 0.785 in iter 2 and stayed passing in iter 3. |
| Iter 3 weighted-cloud-reliability-calc | Iter 2 weighted-campus-energy-balance-calc mediator_report_summary plus iter 1 failure artifacts | The campus success artifact said using PERCENTILE rather than PERCENTILE.INC avoided #NAME? errors. Cloud stayed passing in iter 3 at 1.00 / 0.750. |
| Iter 3 api-sla-at-risk-calc | Iter 2 weighted-hospital-bedflow-calc failure_signature and mediator_report_summary plus iter 1 port failure_signature | API received prior-iteration warnings about missing workbook values and #NAME? statistics formula failures, then recovered in iter 3 at 1.00 / 0.815. |

Skill-evolution finding:
- The reward improvement came without any accepted skill update; artifacts/skill_updates is empty.
- Skill proposals repeatedly targeted plausible spreadsheet lessons such as formula compatibility and readback verification, but validation blocked them because each candidate either failed the +0.010 mean threshold or regressed at least one validation task.
- Cross-task diffusion had the stronger observed effect: iteration 1 had 0/5 passes; iteration 2 rose to 3/5 passes after prior-iteration artifacts were injected.
- The diffusion signal was not uniformly sufficient. Hospital stayed at 0/3, and campus regressed in iteration 3. The selected context for those failures was mostly negative prior-failure context rather than a positive same-task or same-iteration success pattern.

Infra finding:
- No environment failures occurred.
- Verifier and trace statuses were ok for all 15 main records.
- Transient LLM judge/planner timeouts occurred during long validation/scoring calls but recovered via retry.

---

Weighted-Risk-Assessment whole-family findings: top-k similarity, context-only

Run:
data/experiments/20260605-002542-all-wra-top-k-similarity-hermes

Configuration:
- Tasks: 8 Weighted-Risk-Assessment tasks
- Iterations: 3
- Skill updates: none
- Diffusion: enabled
- Diffusion policy: top_k_similarity
- Diffusion graph: task_similarity from CLI, stored as precomputed_similarity in run metrics
- Mode: non-verbose

Overall scoring:
- Total records: 24
- Scored tasks: 20/24
- Environment failures: 4
- Verifier mean: 0.5000
- Verifier median: 0.5000
- Verifier macro mean: 0.4792
- Verifier bootstrap CI: 95% [0.300, 0.700]
- Judge mean: 0.4047
- Judge median: 0.1775
- Judge macro mean: 0.3996
- Total tokens: 212178

V/J reward table

V = verifier reward, J = judge reward. ENV means the Harbor task ended as an
environment failure and was not scored.

| Task | Iter 1 V/J | Iter 2 V/J | Iter 3 V/J | Verifier mean | Judge mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| api-sla-at-risk-calc | 0.00 / 0.105 | 0.00 / 0.150 | 1.00 / 0.720 | 0.333 | 0.325 |
| campus-budget-at-risk-calc | 0.00 / 0.155 | 1.00 / 0.750 | 1.00 / 0.725 | 0.667 | 0.543 |
| factory-output-at-risk-calc | 1.00 / 0.835 | 1.00 / 0.655 | ENV / ENV | 1.000 | 0.745 |
| hospital-capacity-at-risk-calc | 0.00 / 0.125 | 1.00 / 0.820 | 1.00 / 0.800 | 0.667 | 0.582 |
| weighted-campus-energy-balance-calc | 0.00 / 0.135 | ENV / ENV | ENV / ENV | 0.000 | 0.135 |
| weighted-cloud-reliability-calc | 0.00 / 0.150 | 1.00 / 0.000 | 1.00 / 0.700 | 0.667 | 0.283 |
| weighted-hospital-bedflow-calc | 0.00 / 0.000 | 0.00 / 0.155 | 0.00 / 0.155 | 0.000 | 0.103 |
| weighted-port-throughput-calc | 0.00 / 0.200 | 1.00 / 0.760 | ENV / ENV | 0.500 | 0.480 |

Verifier pass rate by iteration:

| Iteration | Passing tasks | Scored tasks | Env failures | Verifier mean | Judge mean | Diffusion context rows | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 1/8 | 8/8 | 0 | 0.125 | 0.213 | 0 | No prior records, so diffusion had no context to inject. |
| 2 | 5/7 | 7/8 | 1 | 0.714 | 0.470 | 7 | Campus budget, factory, hospital capacity, cloud, and port passed after prior-iteration failure signatures became available. |
| 3 | 4/5 | 5/8 | 3 | 0.800 | 0.620 | 7 | API recovered; campus budget, hospital capacity, and cloud stayed passing; hospital bedflow still failed. |

Diffusion cross-task injection

Same-iteration injection was prohibited and avoided. The diffusion log had 112
eligible records, 36 selected/rendered records, and 0 rendered records with
source_iteration >= target_iteration.

Rendered artifacts were all `failure_signature` records. Rendered sources:

| Source task | Rendered count |
| --- | ---: |
| weighted-hospital-bedflow-calc | 9 |
| api-sla-at-risk-calc | 7 |
| weighted-port-throughput-calc | 6 |
| weighted-cloud-reliability-calc | 6 |
| weighted-campus-energy-balance-calc | 4 |
| campus-budget-at-risk-calc | 4 |

Diffusion finding:
- The verifier reward improved sharply after context became available: iteration 1 had 1/8 passes, while iteration 2 had 5/7 scored passes and iteration 3 had 4/5 scored passes.
- Among scored rows with rendered diffusion context, reward_after_diffusion_context averaged 0.700 across 10 rows.
- The improvement was not uniform. Weighted hospital bedflow failed all three scored attempts, and weighted campus energy balance became unscored in iterations 2 and 3 because of environment failures.
- The run only injected failure signatures, so the positive movement appears to come from cross-task warnings about failure modes rather than positive worked examples.
- The evidence supports top-k similarity diffusion as a useful context-only signal for this WRA family, but the four environment failures make the final iteration less clean than the verifier mean alone suggests.

Skill-update finding:
- Skill updates were disabled in the run config for executor, planner, and mediator.
- artifacts/skill_updates is empty.
- The reward movement therefore came without committed skill file changes. Under `condition=no_feedback`, the normal condition prior channel and mediator path were disabled; the active context mechanism was diffusion.

Infra finding:
- Four environment failures occurred: weighted-campus-energy-balance-calc in iterations 2 and 3, factory-output-at-risk-calc in iteration 3, and weighted-port-throughput-calc in iteration 3.
- Several long Harbor/Hermes executions took tens of minutes, including cloud reliability in iteration 2 and hospital bedflow in iteration 1.
- Planner and judge LLM calls had transient timeouts but recovered via retry. The run completed and wrote summary.json, metrics.jsonl, judge_rewards.jsonl, and diffusion/diffused_records.jsonl.
