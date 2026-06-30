# Logical Batch HWPX Diffusion Report

Updated: 2026-06-30T03:06:37.730307+00:00

## Seed Batch

- RNG seed: `7880645489880290376`
- Seed tasks: hwpx-project-proposal, hwpx-supplier-contact-sheet, hwpx-safety-audit-brief

## Aggregate

- Runs: 12
- Verifier successes: 10/12
- Total tokens: 4190574
- Proxy dollar cost: $5.7811
- Cost model: hybrid. Executor uses Claude/Harbor reported cost when available; planner uses $5/M, mediator+compactor $0.5/M, judge $0/M. Executor proxy fallback is input $5/M, output $25/M, cache read $0.5/M.

## Runs

| Logical iter | Task | Sources | Tokens | Proxy $ | Verifier | Judge | Transfer tokens | Budget violation | Compacted artifacts | Dropped artifacts |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | `hwpx-project-proposal` | seed | 173773 | $0.4069 | 1.0 | 0.7100000000000001 | 0 | False | - | - |
| 0 | `hwpx-safety-audit-brief` | seed | 386019 | $0.5400 | 0.0 | 0.15500000000000003 | 0 | False | - | - |
| 0 | `hwpx-supplier-contact-sheet` | seed | 375239 | $0.4538 | 1.0 | 0.77 | 0 | False | - | - |
| 1 | `hwpx-clinic-intake-summary` | HWPX-Document-Automation/hwpx-supplier-contact-sheet | 158885 | $0.2977 | 1.0 | 0.7100000000000001 | 361 | False | - | - |
| 1 | `hwpx-renewal-playbook-update` | HWPX-Document-Automation/hwpx-safety-audit-brief | 1027089 | $1.0857 | 1.0 | 0.8350000000000001 | 418 | False | - | - |
| 1 | `hwpx-training-feedback` | HWPX-Document-Automation/hwpx-project-proposal | 139439 | $0.3127 | 1.0 | 0.795 | 361 | False | - | - |
| 2 | `hwpx-event-announcement` | HWPX-Document-Automation/hwpx-renewal-playbook-update | 404246 | $0.4759 | 1.0 | 0.7000000000000001 | 342 | False | - | - |
| 2 | `hwpx-inventory-report` | HWPX-Document-Automation/hwpx-training-feedback | 301296 | $0.4289 | 1.0 | 0.0 | 339 | False | - | - |
| 2 | `hwpx-safety-audit-brief` | HWPX-Document-Automation/hwpx-clinic-intake-summary | 366320 | $0.4104 | 0.0 | 0.11499999999999999 | 386 | False | - | - |
| 3 | `hwpx-training-feedback` | HWPX-Document-Automation/hwpx-safety-audit-brief | 235729 | $0.4091 | 1.0 | 0.8025 | 391 | False | - | - |
| 3 | `hwpx-event-announcement` | HWPX-Document-Automation/hwpx-inventory-report | 396604 | $0.5849 | 1.0 | 0.8450000000000001 | 333 | False | - | - |
| 3 | `hwpx-inventory-report` | HWPX-Document-Automation/hwpx-event-announcement | 225935 | $0.3752 | 1.0 | 0.795 | 343 | False | - | - |

## Softmax Gate Records

Each source selects one target from up to three candidates using softmax temperature `0.35`. Candidate distributions are recorded in `analysis/softmax_decisions.jsonl`.
Compact LLM router packets use `openrouter/openai/gpt-5.2` with weight cap `0.3` and are recorded in `analysis/llm_router_decisions.jsonl`.

## Implemented Job Fixes

- Seed exploration cost: 3 rows, 935031 tokens, $1.4007, successes 2/3.
- LLM router usage: 9/9 parsed calls, 0 fallbacks, 18983 prompt tokens, 3995 completion tokens.
- Selection concentration max share: 0.222; over-exploitation flag: False.
- Quorum candidate activations: 0.
- Negative-transfer flags: 1.
- Safeguard events: 0.
- Skipped tasks: none.

### Run Count Board

| Task | Runs | Successes | Logical iters | Last verifier | Last judge |
| --- | ---: | ---: | --- | ---: | ---: |
| `hwpx-supplier-contact-sheet` | 1 | 1 | [0] | 1.0 | 0.77 |
| `hwpx-event-announcement` | 2 | 2 | [2, 3] | 1.0 | 0.8450000000000001 |
| `hwpx-clinic-intake-summary` | 1 | 1 | [1] | 1.0 | 0.7100000000000001 |
| `hwpx-project-proposal` | 1 | 1 | [0] | 1.0 | 0.7100000000000001 |
| `hwpx-training-feedback` | 2 | 2 | [1, 3] | 1.0 | 0.8025 |
| `hwpx-safety-audit-brief` | 2 | 0 | [0, 2] | 0.0 | 0.11499999999999999 |
| `hwpx-renewal-playbook-update` | 1 | 1 | [1] | 1.0 | 0.8350000000000001 |
| `hwpx-inventory-report` | 2 | 2 | [2, 3] | 1.0 | 0.795 |

### Checkpoint Efficiency

| Logical iter | Cumulative rows | Iter rows | Cost | Successes | Successes / $ | Judge points / $ | Mean judge |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 3 | $1.4007 | 2 | 1.427819 | 1.167242 | 0.545 |
| 1 | 6 | 3 | $3.0968 | 5 | 1.614567 | 1.283581 | 0.6625 |
| 2 | 9 | 3 | $4.4119 | 7 | 1.586607 | 1.085692 | 0.532222 |
| 3 | 12 | 3 | $5.7811 | 10 | 1.729768 | 1.251055 | 0.602708 |

### Activation Audit

| Iter | Target | Sources | Similarities | Probabilities | Quorum | Weak sim | Negative-transfer flag | Dropped |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `hwpx-project-proposal` | [] | [] | [] | False | False | False | [] |
| 0 | `hwpx-safety-audit-brief` | [] | [] | [] | False | False | False | [] |
| 0 | `hwpx-supplier-contact-sheet` | [] | [] | [] | False | False | False | [] |
| 1 | `hwpx-clinic-intake-summary` | ['hwpx-supplier-contact-sheet'] | [0.012375323383084613] | [0.16657938606584502] | False | False | False | [] |
| 1 | `hwpx-renewal-playbook-update` | ['hwpx-safety-audit-brief'] | [-0.04695824459790601] | [0.33408589217581935] | False | True | False | [] |
| 1 | `hwpx-training-feedback` | ['hwpx-project-proposal'] | [0.22881287071137824] | [0.36641653995359474] | False | False | False | [] |
| 2 | `hwpx-event-announcement` | ['hwpx-renewal-playbook-update'] | [0.03198112797357747] | [0.26078230928019414] | False | False | False | [] |
| 2 | `hwpx-inventory-report` | ['hwpx-training-feedback'] | [0.15453578410468666] | [0.33859431129914364] | False | False | False | [] |
| 2 | `hwpx-safety-audit-brief` | ['hwpx-clinic-intake-summary'] | [-0.028276467555612292] | [0.21577076638612114] | False | True | True | [] |
| 3 | `hwpx-training-feedback` | ['hwpx-safety-audit-brief'] | [-0.22375243781094536] | [0.41998192799647543] | False | True | False | [] |
| 3 | `hwpx-event-announcement` | ['hwpx-inventory-report'] | [0.15264762363298726] | [0.3941679906560765] | False | False | False | [] |
| 3 | `hwpx-inventory-report` | ['hwpx-event-announcement'] | [0.11782684091065418] | [0.3119058280067466] | False | False | False | [] |

## Notes

- Logical iteration k+1 used only selected stores built from logical iteration k artifacts.
- Physical runs were sequential, but same-logical-iteration tasks did not consume one another's artifacts.
- Mediator was enabled through the `learned_mediator` condition and emitted mediator summaries in exported stores.
- Existing infra compaction was available through `_fit_prior_context_bundle`; budget and compaction telemetry are reported from `metrics.jsonl`.
- The final selector is LLM-assisted: source artifacts are compacted into routing packets, GPT-5.2 scores the deterministic top 3, Python blends those scores with signed affinity, then softmax activation and source-backed wildcard rescues run.
- Partially fixable pre/post compaction-token lifecycle fields are intentionally out of scope for this run.
