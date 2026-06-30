# Logical Batch WRA Diffusion Report

Updated: 2026-06-29T17:42:21.458076+00:00

## Seed Batch

- RNG seed: `6035282358295677311`
- Seed tasks: weighted-campus-energy-balance-calc, weighted-port-throughput-calc, api-sla-at-risk-calc

## Aggregate

- Runs: 10
- Verifier successes: 8/10
- Total tokens: 2167286
- Proxy dollar cost: $4.8570
- Cost model: hybrid. Executor uses Claude/Harbor reported cost when available; planner uses $5/M, mediator+compactor $0.5/M, judge $0/M. Executor proxy fallback is input $5/M, output $25/M, cache read $0.5/M.

## Runs

| Logical iter | Task | Sources | Tokens | Proxy $ | Verifier | Judge | Transfer tokens | Budget violation | Compacted artifacts | Dropped artifacts |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| 0 | `api-sla-at-risk-calc` | seed | 238494 | $0.5115 | 0.0 | 0.14500000000000002 | 0 | False | - | - |
| 0 | `weighted-campus-energy-balance-calc` | seed | 213775 | $0.3993 | 1.0 | 0.8450000000000001 | 0 | False | - | - |
| 0 | `weighted-port-throughput-calc` | seed | 176670 | $0.3813 | 1.0 | 0.8350000000000001 | 0 | False | - | - |
| 1 | `factory-output-at-risk-calc` | Weighted-Risk-Assessment/api-sla-at-risk-calc | 332087 | $0.6001 | 1.0 | 0.8350000000000001 | 452 | False | - | - |
| 1 | `weighted-cloud-reliability-calc` | Weighted-Risk-Assessment/weighted-port-throughput-calc | 4226 | $0.0211 | None | None | 406 | False | - | - |
| 1 | `weighted-hospital-bedflow-calc` | Weighted-Risk-Assessment/weighted-campus-energy-balance-calc | 204574 | $0.6138 | 1.0 | 0.855 | 413 | False | - | - |
| 2 | `campus-budget-at-risk-calc` | Weighted-Risk-Assessment/factory-output-at-risk-calc | 352398 | $0.6120 | 1.0 | 0.755 | 416 | False | - | - |
| 2 | `hospital-capacity-at-risk-calc` | seed | 165496 | $0.3107 | 1.0 | 0.775 | 0 | False | - | - |
| 2 | `api-sla-at-risk-calc` | Weighted-Risk-Assessment/weighted-hospital-bedflow-calc | 204376 | $0.8256 | 1.0 | 0.655 | 430 | False | - | - |
| 3 | `weighted-cloud-reliability-calc` | Weighted-Risk-Assessment/campus-budget-at-risk-calc,Weighted-Risk-Assessment/api-sla-at-risk-calc | 275190 | $0.5815 | 1.0 | 0.755 | 390 | False | logical-L03-2-campus-budget-at-risk-calc-to-weighted-cloud-reliability-calc,logical-L03-1-api-sla-at-risk-calc-to-weighted-cloud-reliability-calc | - |

## Softmax Gate Records

Each source selects one target from up to three candidates using softmax temperature `0.35`. Candidate distributions are recorded in `analysis/softmax_decisions.jsonl`.
Compact LLM router packets use `openrouter/openai/gpt-5.2` with weight cap `0.3` and are recorded in `analysis/llm_router_decisions.jsonl`.

## Implemented Job Fixes

- Seed exploration cost: 3 rows, 628939 tokens, $1.2921, successes 2/3.
- LLM router usage: 9/9 parsed calls, 0 fallbacks, 19138 prompt tokens, 5179 completion tokens.
- Selection concentration max share: 0.333; over-exploitation flag: True.
- Quorum candidate activations: 1.
- Negative-transfer flags: 0.
- Safeguard events: 0.
- Skipped tasks: none.

### Run Count Board

| Task | Runs | Successes | Logical iters | Last verifier | Last judge |
| --- | ---: | ---: | --- | ---: | ---: |
| `weighted-cloud-reliability-calc` | 2 | 1 | [1, 3] | 1.0 | 0.755 |
| `weighted-hospital-bedflow-calc` | 1 | 1 | [1] | 1.0 | 0.855 |
| `weighted-campus-energy-balance-calc` | 1 | 1 | [0] | 1.0 | 0.8450000000000001 |
| `weighted-port-throughput-calc` | 1 | 1 | [0] | 1.0 | 0.8350000000000001 |
| `api-sla-at-risk-calc` | 2 | 1 | [0, 2] | 1.0 | 0.655 |
| `campus-budget-at-risk-calc` | 1 | 1 | [2] | 1.0 | 0.755 |
| `factory-output-at-risk-calc` | 1 | 1 | [1] | 1.0 | 0.8350000000000001 |
| `hospital-capacity-at-risk-calc` | 1 | 1 | [2] | 1.0 | 0.775 |

### Checkpoint Efficiency

| Logical iter | Cumulative rows | Iter rows | Cost | Successes | Successes / $ | Judge points / $ | Mean judge |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 3 | 3 | $1.2921 | 2 | 1.547868 | 1.412429 | 0.608333 |
| 1 | 6 | 3 | $2.5272 | 4 | 1.582799 | 1.390885 | 0.585833 |
| 2 | 9 | 3 | $4.2754 | 7 | 1.637256 | 1.333194 | 0.633333 |
| 3 | 10 | 1 | $4.8570 | 8 | 1.647117 | 1.329017 | 0.6455 |

### Activation Audit

| Iter | Target | Sources | Similarities | Probabilities | Quorum | Weak sim | Negative-transfer flag | Dropped |
| ---: | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | `api-sla-at-risk-calc` | [] | [] | [] | False | False | False | [] |
| 0 | `weighted-campus-energy-balance-calc` | [] | [] | [] | False | False | False | [] |
| 0 | `weighted-port-throughput-calc` | [] | [] | [] | False | False | False | [] |
| 1 | `factory-output-at-risk-calc` | ['api-sla-at-risk-calc'] | [-0.048454811656005764] | [0.2832954494579406] | False | True | False | [] |
| 1 | `weighted-cloud-reliability-calc` | ['weighted-port-throughput-calc'] | [-0.045863482587064824] | [0.31042953596027373] | False | True | False | [] |
| 1 | `weighted-hospital-bedflow-calc` | ['weighted-campus-energy-balance-calc'] | [-0.002727164179104513] | [0.34416692328660614] | False | True | False | [] |
| 2 | `campus-budget-at-risk-calc` | ['factory-output-at-risk-calc'] | [0.1311917239200821] | [0.35333164442122167] | False | False | False | [] |
| 2 | `hospital-capacity-at-risk-calc` | ['weighted-cloud-reliability-calc'] | [-0.008469651741293549] | [0.33274285733790043] | False | True | False | [] |
| 2 | `api-sla-at-risk-calc` | ['weighted-hospital-bedflow-calc'] | [-0.01096119402985088] | [0.2748030908106407] | False | True | False | [] |
| 3 | `weighted-cloud-reliability-calc` | ['api-sla-at-risk-calc', 'campus-budget-at-risk-calc'] | [0.035745273631840796, -0.028367775408670998] | [0.397976721800909, 0.3496371530229769] | True | True | False | [] |

## Notes

- Logical iteration k+1 used only selected stores built from logical iteration k artifacts.
- Physical runs were sequential, but same-logical-iteration tasks did not consume one another's artifacts.
- Mediator was enabled through the `learned_mediator` condition and emitted mediator summaries in exported stores.
- Existing infra compaction was available through `_fit_prior_context_bundle`; budget and compaction telemetry are reported from `metrics.jsonl`.
- The final selector is LLM-assisted: source artifacts are compacted into routing packets, GPT-5.2 scores the deterministic top 3, Python blends those scores with signed affinity, then softmax activation and source-backed wildcard rescues run.
- Partially fixable pre/post compaction-token lifecycle fields are intentionally out of scope for this run.
