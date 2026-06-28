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
