# Logical Batch WRA Diffusion Report

Updated: 2026-06-26T16:24:39.697649+00:00

## Seed Batch

- RNG seed: `3441696556429212968`
- Seed tasks: campus-budget-at-risk-calc, factory-output-at-risk-calc, weighted-cloud-reliability-calc

## Aggregate

- Runs: 10
- Verifier successes: 9/10
- Total tokens: 2380706
- Proxy dollar cost: $5.3672
- Cost model: hybrid. Executor uses Claude/Harbor reported cost when available; planner uses $5/M, mediator+compactor $0.5/M, judge $0/M. Executor proxy fallback is input $5/M, output $25/M, cache read $0.5/M.

## Runs

| Logical iter | Task | Sources | Tokens | Proxy $ | Verifier | Judge | Transfer tokens | Budget violation | Compacted artifacts | Dropped artifacts |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | `campus-budget-at-risk-calc` | seed | 200739 | $0.3902 | 1.0 | 0.8875000000000001 | 0 | False | - | - |
| 0 | `factory-output-at-risk-calc` | seed | 126166 | $0.3614 | 0.0 | 0.13 | 0 | False | - | - |
| 0 | `weighted-cloud-reliability-calc` | seed | 142586 | $0.3381 | 1.0 | 0.8625 | 0 | False | - | - |
| 1 | `api-sla-at-risk-calc` | Weighted-Risk-Assessment/campus-budget-at-risk-calc | 329930 | $0.6875 | 1.0 | 0.78 | 399 | False | - | - |
| 1 | `hospital-capacity-at-risk-calc` | Weighted-Risk-Assessment/factory-output-at-risk-calc | 308783 | $0.6015 | 1.0 | 0.8275 | 422 | False | - | - |
| 1 | `weighted-campus-energy-balance-calc` | Weighted-Risk-Assessment/weighted-cloud-reliability-calc | 269900 | $0.6169 | 1.0 | 0.8400000000000001 | 403 | False | - | - |
| 2 | `weighted-hospital-bedflow-calc` | Weighted-Risk-Assessment/hospital-capacity-at-risk-calc | 183848 | $0.5729 | 1.0 | 0.7050000000000001 | 398 | True | - | logical-L02-2-api-sla-at-risk-calc-to-weighted-hospital-bedflow-calc |
| 2 | `weighted-port-throughput-calc` | Weighted-Risk-Assessment/weighted-campus-energy-balance-calc | 188528 | $0.4726 | 1.0 | 0.8500000000000001 | 400 | False | - | - |
| 3 | `api-sla-at-risk-calc` | Weighted-Risk-Assessment/weighted-port-throughput-calc | 304763 | $0.5486 | 1.0 | 0.8150000000000001 | 400 | True | - | logical-L03-2-weighted-hospital-bedflow-calc-to-api-sla-at-risk-calc |
| 3 | `factory-output-at-risk-calc` | Weighted-Risk-Assessment/weighted-port-throughput-calc | 325463 | $0.7775 | 1.0 | 0.8150000000000001 | 393 | True | - | logical-L03-2-weighted-hospital-bedflow-calc-to-factory-output-at-risk-calc |

## Softmax Gate Records

Each source selected one target from up to three candidates. Candidate distributions are recorded in `analysis/softmax_decisions.jsonl`.

## Notes

- Logical iteration k+1 used only selected stores built from logical iteration k artifacts.
- Physical runs were sequential, but same-logical-iteration tasks did not consume one another's artifacts.
- Mediator was enabled through the `learned_mediator` condition and emitted mediator summaries in exported stores.
- Existing infra compaction was available through `_fit_prior_context_bundle`; budget and compaction telemetry are reported from `metrics.jsonl`.
