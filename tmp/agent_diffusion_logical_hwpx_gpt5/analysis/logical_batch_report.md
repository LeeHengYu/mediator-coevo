# Logical Batch HWPX Diffusion Report

Updated: 2026-06-25T16:15:04.577068+00:00

## Seed Batch

- RNG seed: `2053740363624367241`
- Seed tasks: hwpx-event-announcement, hwpx-project-proposal, hwpx-inventory-report

## Aggregate

- Runs: 10
- Verifier successes: 7/10
- Total tokens: 2073579
- Proxy dollar cost: $5.1448
- Cost model: hybrid. Executor uses Claude/Harbor reported cost when available; planner uses $5/M, mediator+compactor $0.5/M, judge $0/M. Executor proxy fallback is input $5/M, output $25/M, cache read $0.5/M.

## Runs

| Logical iter | Task | Sources | Tokens | Proxy $ | Verifier | Judge | Transfer tokens | Budget violation | Compacted artifacts | Dropped artifacts |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | `hwpx-event-announcement` | seed | 188491 | $0.7174 | 1.0 | 0.8150000000000001 | 0 | False | - | - |
| 0 | `hwpx-inventory-report` | seed | 230497 | $0.6934 | 1.0 | 0.76 | 0 | False | - | - |
| 0 | `hwpx-project-proposal` | seed | 4230 | $0.0211 | None | None | 0 | False | - | - |
| 1 | `hwpx-clinic-intake-summary` | HWPX-Document-Automation/hwpx-event-announcement | 4149 | $0.0207 | None | None | 392 | False | - | - |
| 1 | `hwpx-renewal-playbook-update` | HWPX-Document-Automation/hwpx-inventory-report | 188369 | $0.7174 | 1.0 | 0.785 | 390 | False | - | - |
| 1 | `hwpx-supplier-contact-sheet` | HWPX-Document-Automation/hwpx-event-announcement | 3481 | $0.0174 | None | None | 384 | False | - | - |
| 2 | `hwpx-project-proposal` | seed | 283313 | $0.6440 | 1.0 | 0.761 | 0 | False | - | - |
| 2 | `hwpx-training-feedback` | HWPX-Document-Automation/hwpx-renewal-playbook-update | 511632 | $0.8763 | 1.0 | 0.7050000000000001 | 387 | False | - | - |
| 3 | `hwpx-clinic-intake-summary` | HWPX-Document-Automation/hwpx-project-proposal | 263704 | $0.7325 | 1.0 | 0.8500000000000001 | 387 | False | - | - |
| 3 | `hwpx-supplier-contact-sheet` | HWPX-Document-Automation/hwpx-training-feedback | 395713 | $0.7045 | 1.0 | 0.7150000000000001 | 381 | False | - | - |

## Softmax Gate Records

Each source selected one target from up to three candidates. Candidate distributions are recorded in `analysis/softmax_decisions.jsonl`.

## Notes

- Logical iteration k+1 used only selected stores built from logical iteration k artifacts.
- Physical runs were sequential, but same-logical-iteration tasks did not consume one another's artifacts.
- Mediator was enabled through the `learned_mediator` condition and emitted mediator summaries in exported stores.
- Existing infra compaction was available through `_fit_prior_context_bundle`; budget and compaction telemetry are reported from `metrics.jsonl`.
