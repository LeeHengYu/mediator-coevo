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

---

Distribution-Center-Auditing whole-family findings: top-k similarity, context-only

Run:
data/experiments/20260607-094758-0607-DCA-3

Configuration:
- Tasks: 8 Distribution-Center-Auditing tasks
- Iterations: 3
- Skill updates: none
- Diffusion: enabled
- Diffusion policy: top_k_similarity
- Diffusion graph: task_similarity from config, stored as precomputed_similarity in run metrics
- Mode: non-verbose

Overall scoring:
- Total records: 24
- Scored tasks: 24/24
- Environment failures: 0
- Verifier mean: 0.7500
- Verifier median: 1.0000
- Verifier macro mean: 0.7500
- Verifier bootstrap CI: 95% [0.583, 0.917]
- Judge mean: 0.6417
- Judge median: 0.7425
- Judge macro mean: 0.6417
- Judge bootstrap CI: 95% [0.528, 0.743]
- Total tokens: 365239

V/J reward table

V = verifier reward, J = judge reward.

| Task | Iter 1 V/J | Iter 2 V/J | Iter 3 V/J | Verifier mean | Judge mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| cycle_count_variance_audit | 0.00 / 0.135 | 1.00 / 0.875 | 1.00 / 0.750 | 0.667 | 0.587 |
| outbound_manifest_audit | 1.00 / 0.825 | 1.00 / 0.730 | 1.00 / 0.890 | 1.000 | 0.815 |
| promo_register_audit | 1.00 / 0.815 | 0.00 / 0.200 | 1.00 / 0.825 | 0.667 | 0.613 |
| receiving_exception_audit | 1.00 / 0.810 | 1.00 / 0.735 | 1.00 / 0.715 | 1.000 | 0.753 |
| returns_disposition_audit | 0.00 / 0.200 | 0.00 / 0.155 | 0.00 / 0.200 | 0.000 | 0.185 |
| service_queue_sla_audit | 1.00 / 0.685 | 1.00 / 0.700 | 1.00 / 0.970 | 1.000 | 0.785 |
| timesheet_policy_audit | 1.00 / 0.825 | 1.00 / 0.815 | 1.00 / 0.735 | 1.000 | 0.792 |
| trailer_detention_audit | 1.00 / 0.780 | 0.00 / 0.200 | 1.00 / 0.830 | 0.667 | 0.603 |

Verifier pass rate by iteration:

| Iteration | Passing tasks | Scored tasks | Env failures | Verifier mean | Judge mean | Rendered artifacts | Notes |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 6/8 | 8/8 | 0 | 0.750 | 0.634 | 0 | No prior records, so diffusion had no context to inject. Cycle count and returns disposition failed. |
| 2 | 5/8 | 8/8 | 0 | 0.625 | 0.551 | 18 | Cycle count recovered, but promo register and trailer detention regressed under prior-context/diffusion exposure. |
| 3 | 7/8 | 8/8 | 0 | 0.875 | 0.739 | 20 | Promo register and trailer detention recovered. Returns disposition remained the only verifier failure. |

Diffusion cross-task injection

Same-iteration injection was prohibited and avoided. The diffusion log had 294
eligible records, 38 selected/rendered records, and 0 rendered records with
source_iteration >= target_iteration. No diffusion context exceeded the 4000
token budget, and no selected artifacts were dropped for budget.

Selected/rendered artifact types:

Counts below are selected diffusion records. All selected records rendered in
this run, so selected and rendered counts match.

| Artifact type | Rendered count |
| --- | ---: |
| debug_hint | 7 |
| mediator_report_summary | 5 |
| regression_warning | 5 |
| run_outcome | 21 |

Selected/rendered artifacts by type:

| Artifact type | Artifact | Source iteration | Selected/rendered count |
| --- | --- | ---: | ---: |
| debug_hint | trailer-detention-audit-iter0001-debug-hint | 1 | 5 |
| debug_hint | returns-disposition-audit-iter0000-debug-hint | 0 | 1 |
| debug_hint | returns-disposition-audit-iter0001-debug-hint | 1 | 1 |
| mediator_report_summary | trailer-detention-audit-iter0001-mediator-report-summary | 1 | 5 |
| regression_warning | trailer-detention-audit-iter0001-regression-warning | 1 | 5 |
| run_outcome | trailer-detention-audit-iter0000-run-outcome | 0 | 5 |
| run_outcome | receiving-exception-audit-iter0000-run-outcome | 0 | 5 |
| run_outcome | service-queue-sla-audit-iter0000-run-outcome | 0 | 4 |
| run_outcome | timesheet-policy-audit-iter0000-run-outcome | 0 | 3 |
| run_outcome | promo-register-audit-iter0000-run-outcome | 0 | 1 |
| run_outcome | timesheet-policy-audit-iter0001-run-outcome | 1 | 1 |
| run_outcome | service-queue-sla-audit-iter0001-run-outcome | 1 | 1 |
| run_outcome | receiving-exception-audit-iter0001-run-outcome | 1 | 1 |

Rendered sources:

| Source task | Rendered count |
| --- | ---: |
| trailer_detention_audit | 20 |
| receiving_exception_audit | 6 |
| service_queue_sla_audit | 5 |
| timesheet_policy_audit | 4 |
| returns_disposition_audit | 2 |
| promo_register_audit | 1 |

Diffusion finding:
- The verifier reward dipped after diffusion first became available, then recovered strongly: iteration 1 had 6/8 passes, iteration 2 had 5/8 passes, and iteration 3 had 7/8 passes.
- Diffusion was operationally clean: all selected artifacts rendered, no same-or-future-iteration artifacts were rendered, and no context-budget violations occurred.
- The strongest positive movement was recovery in cycle_count_variance_audit, promo_register_audit, and trailer_detention_audit. The evidence is correlational because there is no no-diffusion ablation for the same run.
- Final-iteration diffusion was heavily concentrated on trailer_detention_audit. It contributed 15 of 20 selected artifacts in iteration 3, often as regression_warning, debug_hint, and mediator_report_summary variants with the same source-task similarity score.
- This is useful as artifact-type diversity but weak as source diversity. The run demonstrates why top-k similarity should distinguish "top-k source eligibility" from "max selected artifacts" and why duplicate source/type handling matters.
- The most persistent failure was returns_disposition_audit, which stayed at 0/3 despite receiving diffusion in iterations 2 and 3. Diffusion was not uniformly sufficient for hard spreadsheet-audit failures.

Skill-update finding:
- Skill updates were disabled in the run config for executor, planner, and mediator.
- artifacts/skill_updates is empty.
- The reward movement therefore came without committed skill file changes. The active adaptation channels were same-task prior context, cross-task prior context, mediator feedback, and top-k similarity diffusion.

Infra finding:
- No environment failures occurred.
- Verifier and trace statuses were ok for all 24 scored task records.
- The run completed and wrote summary.json, metrics.jsonl, judge_rewards.jsonl, diffusion/diffused_records.jsonl, graph snapshots, and three skill snapshots.
