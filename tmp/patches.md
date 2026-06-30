# Baseline vs Adopted Patch 1+3 Comparison

## Change And Patches

Comparison scope:

- Baseline: `tmp/agent_diffusion_logical_wra_realcost_fresh_20260626`
- Fusion: `tmp/agent_diffusion_logical_wra_fusion_p13_20260628`

Adopted patch id: `patch 1+3`

Changes relative to the baseline:

- Patch 1: selected transfer artifacts are compacted as a set before any budget-driven drop.
- Patch 3: source-backed coverage rescue and run ordering keep never-run or under-observed tasks active.
- Runner behavior changed: the adopted patch run uses `12` rows as a lower bound while finishing the active logical iteration.

Caveat: this is not a strict same-seed A/B. Baseline used seed `3441696556429212968`; fusion used seed `6433290826923716130`. Baseline ran 10 rows; fusion ran 12 rows.

## Aggregate Result

| Metric | Baseline | Fusion | Direction |
| --- | ---: | ---: | --- |
| Rows | 10 | 12 | different budget |
| Verifier successes | 9/10 | 11/12 | slight rate improvement |
| Verifier success rate | 90.0% | 91.7% | improved |
| Judge sum | 7.5125 | 8.1475 | improved total |
| Mean judge | 0.75125 | 0.678958 | regressed |
| Tokens | 2,380,706 | 2,966,155 | higher total |
| Tokens / row | 238,071 | 247,180 | regressed |
| Proxy cost | $5.3672 | $6.0044 | higher total |
| Cost / row | $0.5367 | $0.5004 | improved |
| Verifier successes / $ | 1.6769 | 1.8320 | improved |
| Judge points / $ | 1.3997 | 1.3569 | slight regression |
| Budget violations | 3 | 0 | improved |
| Dropped artifacts | 3 | 0 | improved |
| Skipped tasks | none | none | flat |

For the first 10 fusion rows only:

| Metric | Baseline 10 rows | Fusion first 10 rows | Direction |
| --- | ---: | ---: | --- |
| Verifier successes | 9/10 | 9/10 | flat |
| Mean judge | 0.75125 | 0.68000 | regressed |
| Tokens | 2,380,706 | 2,472,307 | higher |
| Proxy cost | $5.3672 | $5.0556 | improved |
| Cost / row | $0.5367 | $0.5056 | improved |
| Verifier successes / $ | 1.6769 | 1.7802 | improved |
| Judge points / $ | 1.3997 | 1.3450 | regressed |

## Post-Warmup Comparison

Post-warmup excludes logical iteration 0, because seed rows have no diffusion artifacts. This isolates the activated rows where transfer, compaction, routing, and coverage rescue matter.

| Metric | Baseline post-warmup | Fusion post-warmup | Direction |
| --- | ---: | ---: | --- |
| Rows | 7 | 9 | different budget |
| Verifier successes | 7/7 | 9/9 | flat at 100% |
| Verifier success rate | 100.0% | 100.0% | flat |
| Judge sum | 5.6325 | 6.3725 | improved total |
| Mean judge | 0.804643 | 0.708056 | regressed |
| Tokens | 1,911,215 | 2,360,816 | higher total |
| Tokens / row | 273,031 | 262,313 | improved |
| Proxy cost | $4.2774 | $4.7780 | higher total |
| Cost / row | $0.6111 | $0.5309 | improved |
| Verifier successes / $ | 1.6365 | 1.8836 | improved |
| Judge points / $ | 1.3168 | 1.3337 | slight improvement |
| Budget violations | 3 | 0 | improved |
| Dropped artifacts | 3 | 0 | improved |
| Compacted artifacts | 0 | 4 | improved retention mechanism |

Post-warmup conclusion: adopted `patch 1+3` is cleaner than the aggregate view suggests. Once seed rows are removed, both runs have perfect verifier success, but fusion uses fewer tokens per activated row, lower cost per activated row, better verifier success per dollar, slightly better judge points per dollar, and no context drops. The remaining weakness is judge quality per row: fusion's mean judge is lower.

## What Improved

The clearest improvement is transfer-context reliability. Baseline had 3 budget violations and dropped 3 selected artifacts:

- `logical-L02-2-api-sla-at-risk-calc-to-weighted-hospital-bedflow-calc`
- `logical-L03-2-weighted-hospital-bedflow-calc-to-api-sla-at-risk-calc`
- `logical-L03-2-weighted-hospital-bedflow-calc-to-factory-output-at-risk-calc`

Fusion had 0 budget violations and 0 dropped artifacts. Multi-source rows were retained by compaction instead of dropping context:

- `hospital-capacity-at-risk-calc` at L1 compacted 2 artifacts and succeeded.
- `weighted-campus-energy-balance-calc` at L2 compacted 2 artifacts and succeeded.

Verifier efficiency improved. Fusion achieved 1.832 verifier successes per proxy dollar, versus 1.677 in baseline. Even the first 10 fusion rows had 1.780 successes per dollar.

Cost per row improved. Fusion was $0.5004 per row versus baseline $0.5367 per row.

Coverage remained complete. Both runs touched all 8 tasks, but fusion did so while adding source-backed rescue events and without dropping selected transfer artifacts.

The failed seed task recovered again. In both runs, `factory-output-at-risk-calc` failed at seed. In fusion, it was retried at L2 using transfer from `hospital-capacity-at-risk-calc` and succeeded with verifier `1.0`, judge `0.675`.

## What Regressed Or Stayed Weak

Judge quality regressed. Baseline mean judge was 0.75125; fusion mean judge was 0.678958. The first 10 fusion rows also trailed baseline at 0.68000.

Aggregate judge efficiency slightly regressed. Baseline produced 1.3997 judge points per dollar; fusion produced 1.3569. Post-warmup judge efficiency slightly improved, from 1.3168 to 1.3337 judge points per dollar.

Tokens per row increased. Baseline used about 238k tokens per row; fusion used about 247k. This is expected from retained transfer context, but it is still a cost-side regression.

Selection concentration needs watching. Fusion report marked over-exploitation true with max selected target share 0.333. It did not break coverage in this run, but it means repeated activation still needs audit.

## Task-Level Notes

| Task | Baseline | Fusion |
| --- | --- | --- |
| `campus-budget-at-risk-calc` | 1 run, success | 1 run, success |
| `factory-output-at-risk-calc` | seed fail, later success | seed fail, later success |
| `weighted-cloud-reliability-calc` | 1 run, success | 1 run, success |
| `api-sla-at-risk-calc` | 2 runs, both success | 1 run, success |
| `hospital-capacity-at-risk-calc` | 1 run, success | 2 runs, both success |
| `weighted-campus-energy-balance-calc` | 1 run, success | 2 runs, both success |
| `weighted-hospital-bedflow-calc` | 1 run, success, but had dropped source artifact | 1 run, success, no drop |
| `weighted-port-throughput-calc` | 1 run, success | 2 runs, both success |

## Conclusion

Fusion improves the experiment mechanics and verifier-dollar profile:

- context is no longer dropped,
- verifier success rate is slightly higher,
- success per dollar is higher,
- cost per row is lower,
- all tasks remain covered.

Fusion does not improve judge quality per row. Mean judge is lower than baseline in aggregate, first-10, and post-warmup views. Judge points per dollar are lower in aggregate but slightly higher post-warmup. The practical conclusion is that adopted `patch 1+3` is a better routing/context-retention setup, but it still needs a judge-quality improvement before it can be called a clear reward-quality improvement.

# Refined WRA And HWPX Family Runs Versus Baselines

Compared runs:

- WRA baseline: `tmp/agent_diffusion_logical_wra_realcost_fresh_20260626`
- WRA refined: `tmp/agent_diffusion_logical_wra_llmrouter_temp_gpt52_20260630`
- HWPX baseline: `tmp/agent_diffusion_logical_hwpx_gpt5`
- HWPX refined: `tmp/agent_diffusion_logical_hwpx_llmrouter_temp_gpt52_20260630`

Refined patch under comparison:

- Add compact GPT-5.2 LLM router packets for scoring diffusion targets.
- Blend LLM router scores with deterministic signed affinity.
- Use softmax target activation with temperature `0.35`.
- Keep the existing logical-iteration rule: iteration `k+1` only consumes artifacts from iteration `k`.
- Use executor billing cost when Harbor reports it; planner and mediator remain proxy-priced.

The common baseline policy is the earlier logical diffusion batch setup without the GPT-5.2 LLM-router/temp refinement. The family baselines differ by task family, so the cleanest reading is per-family first, then combined.

## WRA Result

| Metric | WRA baseline | WRA refined | Direction |
| --- | ---: | ---: | --- |
| Rows | 10 | 10 | equal |
| Verifier successes | 9/10 | 8/10 | regressed |
| Scored verifier success rate | 9/10 | 8/9 | slight regression |
| Env failures | 0 | 1 | regressed |
| Success rate | 90.0% | 80.0% | regressed |
| Mean judge | 0.751 | 0.646 | regressed |
| Tokens | 2.38M | 2.17M | improved |
| Cost | $5.367 | $4.857 | improved |
| Cost / success | $0.596 | $0.607 | regressed slightly |
| Success / $ | 1.677 | 1.647 | regressed slightly |
| Success / 100k tokens | 0.378 | 0.369 | regressed slightly |
| Budget violations | 3 | 0 | improved |
| Dropped artifacts | 3 | 0 | improved |

WRA refined had one environment failure: `weighted-cloud-reliability-calc` at logical iter 1 returned `verifier_reward=None` with `env_failure_count=1`, using only 4,226 tokens and $0.02113. So the aggregate should be read as 8/10 total-row successes, but 8/9 scored-row successes.

WRA conclusion: refined WRA is cleaner operationally and cheaper in total dollars/tokens, but not better on reward. It removed budget violations and dropped artifacts, but lost one total-row verifier success and judge quality fell. Post-warmup is more favorable on efficiency: refined WRA used 19.5% fewer post-warmup tokens, cost 16.7% less, and success per dollar improved by 2.8%. Post-warmup total-row success dropped from 7/7 to 6/7, but excluding the env failure it was 6/6 on scored rows; mean judge still fell from 0.805 to 0.661.

## HWPX Result

HWPX refined ran 12 rows while baseline ran 10. Both views matter:

| Metric | HWPX baseline 10 | HWPX refined first 10 | HWPX refined full 12 |
| --- | ---: | ---: | ---: |
| Verifier successes | 7/10 | 8/10 | 10/12 |
| Success rate | 70.0% | 80.0% | 83.3% |
| Mean judge | 0.539 | 0.559 | 0.603 |
| Tokens | 2.07M | 3.57M | 4.19M |
| Cost | $5.145 | $4.821 | $5.781 |
| Cost / success | $0.735 | $0.603 | $0.578 |
| Success / $ | 1.361 | 1.659 | 1.730 |
| Success / 100k tokens | 0.338 | 0.224 | 0.239 |

HWPX conclusion: refined HWPX is a clear dollar-efficiency and success-rate win. In the first 10 rows, it gets one more success while costing 6.3% less, so cost per success improves by 18.0% and success per dollar improves by 22.0%. In the full 12-row run, cost per success improves by 21.3% versus baseline. The weakness is token efficiency: first-10 refined HWPX uses 72.1% more tokens than baseline, and full refined HWPX uses 102.1% more tokens. This is mostly executor-side prompt/cache churn.

## Token And Cost Shape

| View | Executor token share | Executor cost share | Prompt share | Completion share |
| --- | ---: | ---: | ---: | ---: |
| WRA baseline | 95.1% | 95.3% | 94.1% | 5.9% |
| WRA refined | 95.6% | 95.5% | 94.1% | 5.9% |
| HWPX baseline | 96.4% | 96.1% | 93.8% | 6.2% |
| HWPX refined | 97.3% | 95.6% | 97.0% | 3.0% |

The refined patch is not expensive because of the router itself. Router usage was small:

- WRA refined router: 9/9 parsed calls, 19,138 input tokens, 5,179 output tokens.
- HWPX refined router: 9/9 parsed calls, 18,983 input tokens, 3,995 output tokens.

The main token increase is executor-side. HWPX refined in particular has much more prompt/cache traffic, which raises token count but not dollar cost proportionally because the cost model uses Harbor executor billing where available.

## Patch Assessment

The refined patch improves routing observability and dollar efficiency, especially for HWPX. It also eliminates WRA baseline budget/drop failures. The patch does not uniformly improve reward quality:

- WRA: operationally cleaner and cheaper total run, but reward regressed; one refined WRA row was an env failure rather than a scored verifier failure.
- HWPX: better success rate, better judge average, and better dollar efficiency, but worse token efficiency.
- Combined: same first-20 success count at lower dollar cost, but lower judge average and materially higher token usage.

Practical conclusion: keep the refined patch as a cost-efficient routing/context mechanism, but do not call it a token-efficiency improvement. The next patch should target executor token bloat and task-specific failure repair, especially `hwpx-safety-audit-brief` exact risk-tier formatting and WRA reward-quality regression.

# Medical-Data-Standardization LLM-Router Temp Run Versus Same-Code Family Runs

Patch/code under comparison:

- Same adopted `llmrouter_temp_gpt52` mechanism as the refined WRA and HWPX runs: compact GPT-5.2 router packets, deterministic signed-affinity blend, softmax target activation at temperature `0.35`, and logical-iteration isolation.
- Only the task family configuration/guidance changed for `Medical-Data-Standardization`; no shared infra change was needed.
- Executor dollar cost uses Harbor billing when available; planner, mediator, judge, and compactor remain proxy-priced.

Compared runs:

- WRA: `tmp/agent_diffusion_logical_wra_llmrouter_temp_gpt52_20260630`
- HWPX: `tmp/agent_diffusion_logical_hwpx_llmrouter_temp_gpt52_20260630`
- Medical: `tmp/agent_diffusion_logical_medical_llmrouter_temp_gpt52_20260630`

## Aggregate Result

| Metric | WRA llmrouter_temp | HWPX llmrouter_temp | Medical llmrouter_temp |
| --- | ---: | ---: | ---: |
| Rows | 10 | 12 | 12 |
| Verifier successes | 8/10 | 10/12 | 1/12 |
| Success rate | 80.0% | 83.3% | 8.3% |
| Mean verifier reward | 0.800 | 0.833 | 0.083 |
| Mean judge | 0.646 | 0.603 | 0.215 |
| Env failures | 1 | 0 | 0 |
| Total cost | $4.857 | $5.781 | $8.691 |
| Executor billing cost | $4.639 | $5.525 | $8.369 |
| Success / $ | 1.647 | 1.730 | 0.115 |
| Judge points / $ | 1.329 | 1.251 | 0.297 |
| Total tokens | 2.17M | 4.19M | 4.88M |
| Tokens / row | 216.7k | 349.2k | 406.3k |
| Executor token share | 95.6% | 97.3% | 96.1% |

Medical is a clear regression versus both same-code refined family runs. It costs more than WRA and HWPX, uses more total tokens, and gets only one verifier success. The only Medical success was the seed `transplant-panel-alignment-harmonization` row; all diffusion-era Medical rows failed verifier.

## Post-Warmup Result

| Metric | WRA post-warmup | HWPX post-warmup | Medical post-warmup |
| --- | ---: | ---: | ---: |
| Rows | 7 | 9 | 9 |
| Verifier successes | 6/7 | 8/9 | 0/9 |
| Mean verifier reward | 0.857 | 0.889 | 0.000 |
| Mean judge | 0.661 | 0.622 | 0.164 |
| Env failures | 1 | 0 | 0 |
| Total cost | $3.565 | $4.380 | $6.343 |
| Executor billing cost | $3.406 | $4.181 | $6.093 |
| Success / $ | 1.683 | 1.826 | 0.000 |
| Judge points / $ | 1.299 | 1.278 | 0.233 |
| Total tokens | 1.54M | 3.26M | 3.78M |
| Tokens / row | 219.8k | 361.7k | 420.5k |

Post-warmup is the main signal: the Medical router/diffusion loop did not convert any selected artifact into a verifier success. This is worse than flat; it is a practical family mismatch for the adopted patch.

## Medical Row Notes

| Logical iter | Task | Verifier | Judge | Cost | Tokens |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `icu-metabolic-harmonization` | 0.0 | 0.200 | $1.238 | 369.9k |
| 0 | `respiratory-panel-json-harmonization` | 0.0 | 0.145 | $0.607 | 434.5k |
| 0 | `transplant-panel-alignment-harmonization` | 1.0 | 0.760 | $0.503 | 286.1k |
| 1 | `cardio-panel-template-harmonization` | 0.0 | 0.140 | $0.471 | 153.0k |
| 1 | `electrolyte-rounding-harmonization` | 0.0 | 0.145 | $0.348 | 156.7k |
| 1 | `respiratory-panel-json-harmonization` | 0.0 | 0.185 | $0.595 | 213.4k |
| 2 | `hepatic-panel-harmonization` | 0.0 | 0.145 | $0.311 | 292.3k |
| 2 | `neonatal-sepsis-harmonization` | 0.0 | 0.180 | $0.805 | 290.5k |
| 2 | `thyroid-monitoring-harmonization` | 0.0 | 0.200 | $2.367 | 1.87M |
| 3 | `cardio-panel-template-harmonization` | 0.0 | 0.200 | $0.635 | 376.9k |
| 3 | `electrolyte-rounding-harmonization` | 0.0 | 0.145 | $0.364 | 195.4k |
| 3 | `respiratory-panel-json-harmonization` | 0.0 | 0.140 | $0.447 | 236.1k |

Anomalies and mechanics:

- No Medical env failures: all 12 rows completed as valid runs.
- The Medical run skipped `oncology-followup-dedup-harmonization`.
- `thyroid-monitoring-harmonization` was the major cost outlier: 1.87M total tokens, $2.367 total cost, $2.340 executor billing, and verifier `0.0`.
- Router overhead was not the main cost driver. Executor tokens were 96.1% of all Medical tokens; the extra cost came from executor-side task solving.
- The run recorded 9/9 parsed LLM-router calls, 0 fallbacks, 2 negative-transfer flags, 1 quorum candidate activation, and over-exploitation flagged at max target share 0.333.

## Conclusion

The adopted `llmrouter_temp_gpt52` patch generalizes well to WRA/HWPX relative to dollar efficiency, but it does not generalize to `Medical-Data-Standardization` in this 12-row iteration. Medical has much lower reward, worse judge quality, higher total cost, and worse token use than both same-code refined family runs.

The failure mode is not infrastructure. It is task-family fit: the selected artifacts did not teach the executor enough about exact medical harmonization contracts such as numeric normalization, per-value unit conversion, required CSV shape, missing-row filtering, and header/order constraints. The next Medical-specific patch should target executor-facing contract repair and token containment, not another router-only change.
