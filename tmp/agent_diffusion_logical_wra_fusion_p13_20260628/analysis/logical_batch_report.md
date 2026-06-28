# Logical Batch WRA Diffusion Report

Updated: 2026-06-28T16:29:34.489269+00:00

## Seed Batch

- RNG seed: `6433290826923716130`
- Seed tasks: campus-budget-at-risk-calc, weighted-cloud-reliability-calc, factory-output-at-risk-calc

## Aggregate

- Runs: 12
- Verifier successes: 11/12
- Total tokens: 2966155
- Proxy dollar cost: $6.0044
- Cost model: hybrid. Executor uses Claude/Harbor reported cost when available; planner uses $5/M, mediator+compactor $0.5/M, judge $0/M. Executor proxy fallback is input $5/M, output $25/M, cache read $0.5/M.

## Runs

| Logical iter | Task | Sources | Tokens | Proxy $ | Verifier | Judge | Transfer tokens | Budget violation | Compacted artifacts | Dropped artifacts |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | `campus-budget-at-risk-calc` | seed | 225301 | $0.5164 | 1.0 | 0.8450000000000001 | 0 | False | - | - |
| 0 | `factory-output-at-risk-calc` | seed | 204984 | $0.3959 | 0.0 | 0.13 | 0 | False | - | - |
| 0 | `weighted-cloud-reliability-calc` | seed | 175054 | $0.3140 | 1.0 | 0.8 | 0 | False | - | - |
| 1 | `api-sla-at-risk-calc` | Weighted-Risk-Assessment/campus-budget-at-risk-calc | 220288 | $0.5293 | 1.0 | 0.7050000000000001 | 409 | False | - | - |
| 1 | `hospital-capacity-at-risk-calc` | Weighted-Risk-Assessment/campus-budget-at-risk-calc,Weighted-Risk-Assessment/factory-output-at-risk-calc | 255091 | $0.5318 | 1.0 | 0.8350000000000001 | 405 | False | logical-L01-2-campus-budget-at-risk-calc-to-hospital-capacity-at-risk-calc,logical-L01-1-factory-output-at-risk-calc-to-hospital-capacity-at-risk-calc | - |
| 1 | `weighted-hospital-bedflow-calc` | Weighted-Risk-Assessment/weighted-cloud-reliability-calc | 355004 | $0.6294 | 1.0 | 0.685 | 408 | False | - | - |
| 2 | `weighted-campus-energy-balance-calc` | Weighted-Risk-Assessment/api-sla-at-risk-calc,Weighted-Risk-Assessment/weighted-hospital-bedflow-calc | 297787 | $0.6204 | 1.0 | 0.68 | 373 | False | logical-L02-2-api-sla-at-risk-calc-to-weighted-campus-energy-balance-calc,logical-L02-1-weighted-hospital-bedflow-calc-to-weighted-campus-energy-balance-calc | - |
| 2 | `weighted-port-throughput-calc` | Weighted-Risk-Assessment/hospital-capacity-at-risk-calc | 242367 | $0.4470 | 1.0 | 0.705 | 425 | False | - | - |
| 2 | `factory-output-at-risk-calc` | Weighted-Risk-Assessment/hospital-capacity-at-risk-calc | 225384 | $0.5567 | 1.0 | 0.6749999999999999 | 421 | False | - | - |
| 3 | `hospital-capacity-at-risk-calc` | Weighted-Risk-Assessment/factory-output-at-risk-calc | 271047 | $0.5147 | 1.0 | 0.74 | 424 | False | - | - |
| 3 | `weighted-campus-energy-balance-calc` | Weighted-Risk-Assessment/weighted-port-throughput-calc | 245480 | $0.4992 | 1.0 | 0.7075 | 424 | False | - | - |
| 3 | `weighted-port-throughput-calc` | Weighted-Risk-Assessment/weighted-campus-energy-balance-calc | 248368 | $0.4496 | 1.0 | 0.64 | 420 | False | - | - |

## Softmax Gate Records

Each source selects one target from up to three candidates. Candidate distributions are recorded in `analysis/softmax_decisions.jsonl`.

## Implemented Job Fixes

- Seed exploration cost: 3 rows, 605339 tokens, $1.2263, successes 2/3.
- Selection concentration max share: 0.333; over-exploitation flag: True.
- Quorum candidate activations: 2.
- Negative-transfer flags: 0.
- Safeguard events: 2.
- Skipped tasks: none.

### Run Count Board

| Task | Runs | Successes | Logical iters | Last verifier | Last judge |
| --- | ---: | ---: | --- | ---: | ---: |
| `weighted-cloud-reliability-calc` | 1 | 1 | [0] | 1.0 | 0.8 |
| `weighted-hospital-bedflow-calc` | 1 | 1 | [1] | 1.0 | 0.685 |
| `weighted-campus-energy-balance-calc` | 2 | 2 | [2, 3] | 1.0 | 0.7075 |
| `weighted-port-throughput-calc` | 2 | 2 | [2, 3] | 1.0 | 0.64 |
| `api-sla-at-risk-calc` | 1 | 1 | [1] | 1.0 | 0.7050000000000001 |
| `campus-budget-at-risk-calc` | 1 | 1 | [0] | 1.0 | 0.8450000000000001 |
| `factory-output-at-risk-calc` | 2 | 1 | [0, 2] | 1.0 | 0.6749999999999999 |
| `hospital-capacity-at-risk-calc` | 2 | 2 | [1, 3] | 1.0 | 0.74 |

### Checkpoint Efficiency

| Logical iter | Cumulative rows | Iter rows | Cost | Successes | Successes / $ | Judge points / $ | Mean judge |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 3 | $1.2263 | 2 | 1.630875 | 1.447402 | 0.591667 |
| 1 | 6 | 3 | $2.9168 | 5 | 1.714199 | 1.371359 | 0.666667 |
| 2 | 9 | 3 | $4.5409 | 8 | 1.761763 | 1.334536 | 0.673333 |
| 3 | 12 | 3 | $6.0044 | 11 | 1.831997 | 1.356927 | 0.678958 |

### Activation Audit

| Iter | Target | Sources | Similarities | Probabilities | Quorum | Weak sim | Negative-transfer flag | Dropped |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `campus-budget-at-risk-calc` | [] | [] | [] | False | False | False | [] |
| 0 | `factory-output-at-risk-calc` | [] | [] | [] | False | False | False | [] |
| 0 | `weighted-cloud-reliability-calc` | [] | [] | [] | False | False | False | [] |
| 1 | `api-sla-at-risk-calc` | ['campus-budget-at-risk-calc'] | [-0.13746651358591655] | [1.0] | False | True | False | [] |
| 1 | `hospital-capacity-at-risk-calc` | ['factory-output-at-risk-calc', 'campus-budget-at-risk-calc'] | [-0.10562571756601613, -0.2070757324488668] | [0.37551878096199537, 0.31527479765828226] | True | True | False | [] |
| 1 | `weighted-hospital-bedflow-calc` | ['weighted-cloud-reliability-calc'] | [-0.08059701492537319] | [0.3706576450365269] | False | True | False | [] |
| 2 | `weighted-campus-energy-balance-calc` | ['weighted-hospital-bedflow-calc', 'api-sla-at-risk-calc'] | [-0.1732764747690122, -0.18322672352523106] | [0.3436830531021504, 0.32299337601859607] | True | True | False | [] |
| 2 | `weighted-port-throughput-calc` | ['hospital-capacity-at-risk-calc'] | [-0.24759717894046251] | [1.0] | False | True | False | [] |
| 2 | `factory-output-at-risk-calc` | ['hospital-capacity-at-risk-calc'] | [-0.03611612268328679] | [0.385264609441493] | False | True | False | [] |
| 3 | `hospital-capacity-at-risk-calc` | ['factory-output-at-risk-calc'] | [-0.27492209283254043] | [0.3559608780329828] | False | True | False | [] |
| 3 | `weighted-campus-energy-balance-calc` | ['weighted-port-throughput-calc'] | [-0.34685684876112055] | [0.3352753733284637] | False | True | False | [] |
| 3 | `weighted-port-throughput-calc` | ['weighted-campus-energy-balance-calc'] | [-0.34685684876112055] | [0.34910880985503673] | False | True | False | [] |

## Notes

- Logical iteration k+1 used only selected stores built from logical iteration k artifacts.
- Physical runs were sequential, but same-logical-iteration tasks did not consume one another's artifacts.
- Mediator was enabled through the `learned_mediator` condition and emitted mediator summaries in exported stores.
- Existing infra compaction was available through `_fit_prior_context_bundle`; budget and compaction telemetry are reported from `metrics.jsonl`.
- The final selector is deterministic: source artifacts are converted into structured transfer signals, then Python computes signed affinity, softmax activation, safeguard drops, and source-backed wildcard rescues.
- Partially fixable pre/post compaction-token lifecycle fields are intentionally out of scope for this run.
