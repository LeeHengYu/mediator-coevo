# Logical Batch WRA Diffusion Report

Updated: 2026-06-25T13:25:41.282740+00:00

## Seed Batch

- RNG seed: `665384088737334353`
- Seed tasks: weighted-port-throughput-calc, factory-output-at-risk-calc, hospital-capacity-at-risk-calc

## Aggregate

- Runs: 10
- Verifier successes: 8/10
- Total tokens: 3109273
- Proxy dollar cost: $6.6127
- Cost model: hybrid. Executor uses Claude/Harbor reported cost when available; planner uses $5/M, mediator+compactor $0.5/M, judge $0/M. Executor proxy fallback is input $5/M, output $25/M, cache read $0.5/M.

## Runs

| Logical iter | Task | Sources | Tokens | Proxy $ | Verifier | Judge | Transfer tokens | Budget violation | Compacted artifacts | Dropped artifacts |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | `factory-output-at-risk-calc` | seed | 262064 | $0.6800 | 0.0 | 0.17500000000000004 | 0 | False | - | - |
| 0 | `hospital-capacity-at-risk-calc` | seed | 181504 | $0.4515 | 1.0 | 0.8425 | 0 | False | - | - |
| 0 | `weighted-port-throughput-calc` | seed | 261287 | $0.4335 | 0.0 | 0.1795 | 0 | False | - | - |
| 1 | `campus-budget-at-risk-calc` | Weighted-Risk-Assessment/hospital-capacity-at-risk-calc | 366072 | $0.7322 | 1.0 | 0.6950000000000001 | 390 | True | - | logical-L01-1-factory-output-at-risk-calc-to-campus-budget-at-risk-calc |
| 1 | `weighted-campus-energy-balance-calc` | Weighted-Risk-Assessment/weighted-port-throughput-calc | 261424 | $0.7050 | 1.0 | 0.8225 | 428 | False | - | - |
| 1 | `weighted-cloud-reliability-calc` | Weighted-Risk-Assessment/hospital-capacity-at-risk-calc | 295000 | $0.6323 | 1.0 | 0.795 | 394 | False | - | - |
| 2 | `api-sla-at-risk-calc` | Weighted-Risk-Assessment/weighted-cloud-reliability-calc | 303777 | $0.7419 | 1.0 | 0.7050000000000001 | 398 | True | - | logical-L02-2-campus-budget-at-risk-calc-to-api-sla-at-risk-calc |
| 2 | `weighted-port-throughput-calc` | Weighted-Risk-Assessment/weighted-campus-energy-balance-calc | 274565 | $0.4995 | 1.0 | 0.775 | 396 | False | - | - |
| 3 | `factory-output-at-risk-calc` | Weighted-Risk-Assessment/api-sla-at-risk-calc | 474841 | $0.7459 | 1.0 | 0.78 | 394 | False | - | - |
| 3 | `weighted-hospital-bedflow-calc` | Weighted-Risk-Assessment/weighted-port-throughput-calc | 428739 | $0.9910 | 1.0 | 0.73 | 394 | False | - | - |

## Softmax Gate Records

Each source selected one target from up to three candidates. Candidate distributions are recorded in `analysis/softmax_decisions.jsonl`.

## Notes

- Logical iteration k+1 used only selected stores built from logical iteration k artifacts.
- Physical runs were sequential, but same-logical-iteration tasks did not consume one another's artifacts.
- Mediator was enabled through the `learned_mediator` condition and emitted mediator summaries in exported stores.
- Existing infra compaction was available through `_fit_prior_context_bundle`; budget and compaction telemetry are reported from `metrics.jsonl`.
