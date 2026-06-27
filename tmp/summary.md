# WRA Run Comparison

Generated: 2026-06-27

## Scope

This comparison uses only repo-copied artifacts under `/Users/hylee_mac/Documents/Project/mediator-coevo/tmp`.

| Label | Source | Unit |
| --- | --- | --- |
| Previous WRA real-cost logical | `tmp/agent_diffusion_logical_wra_realcost/analysis/runs.jsonl` | 10 logical rows |
| Fresh WRA real-cost logical | `tmp/agent_diffusion_logical_wra_realcost_fresh_20260626/analysis/runs.jsonl` | 10 logical rows |
| Hermes WRA matrix Random-k | `tmp/wra_matrix_randomk_gpt52_20260627/data/experiments/20260627-010707-wra-random-k-gpt52-20260627` | 24 fixed matrix task-runs |
| Claude Code WRA matrix Random-k | `tmp/wra_matrix_randomk_gpt52_claudecode_20260627/data/experiments/20260627-112631-wra-random-k-gpt52-claudecode-20260627` | 24 fixed matrix task-runs |

## Cost Provenance

Logical real-cost runs use executor-reported billing plus orchestration proxy:

- `tmp/agent_diffusion_logical_wra_realcost/analysis/runs.jsonl`
- `tmp/agent_diffusion_logical_wra_realcost/analysis/costs.jsonl`
- `tmp/agent_diffusion_logical_wra_realcost/medcoevo-data/experiments/*/jobs/*/run-*/result.json`
- `tmp/agent_diffusion_logical_wra_realcost_fresh_20260626/analysis/runs.jsonl`
- `tmp/agent_diffusion_logical_wra_realcost_fresh_20260626/analysis/costs.jsonl`
- `tmp/agent_diffusion_logical_wra_realcost_fresh_20260626/medcoevo-data/experiments/*/jobs/*/run-*/result.json`

Matrix rows use the repo-copied matrix artifacts:

- Hermes matrix metrics: `tmp/wra_matrix_randomk_gpt52_20260627/data/experiments/20260627-010707-wra-random-k-gpt52-20260627/metrics.jsonl`
- Hermes matrix raw trials: `tmp/wra_matrix_randomk_gpt52_20260627/data/experiments/20260627-010707-wra-random-k-gpt52-20260627/jobs/*/run-*/result.json`
- Claude Code matrix metrics: `tmp/wra_matrix_randomk_gpt52_claudecode_20260627/data/experiments/20260627-112631-wra-random-k-gpt52-claudecode-20260627/metrics.jsonl`
- Claude Code matrix raw trials: `tmp/wra_matrix_randomk_gpt52_claudecode_20260627/data/experiments/20260627-112631-wra-random-k-gpt52-claudecode-20260627/jobs/*/run-*/result.json`

Dollar scheme:

- Logical executor: `cost.executor_reported_cost_usd` from `costs.jsonl`, then plus planner/mediator/judge proxy already stored as `cost.proxy_cost_usd`.
- Claude Code matrix executor: raw billing from `agent_result.cost_usd` in each run-level `result.json`.
- Hermes matrix executor: no billing is available (`agent_result.cost_usd = null`), so executor proxy is used.
- Hermes matrix executor proxy: fresh input `$1.75/M`, cache-read input `$0.175/M`, output `$14.00/M`.
- Matrix orchestration proxy: planner `$5.00/M`, mediator plus compactor `$0.50/M`, judge `$0`.
- Cache-adjusted tokens count cache-read input as 1/10 of ordinary fresh input.

## Model Note

| Run | Executor agent | Executor model | Planner | Mediator | Judge |
| --- | --- | --- | --- | --- | --- |
| Previous WRA real-cost logical | `claude-code` | `openai/gpt-5.2` | `openrouter/anthropic/claude-opus-4.6` | `openrouter/google/gemini-3-flash-preview` | `openrouter/openai/gpt-oss-120b` |
| Fresh WRA real-cost logical | `claude-code` | `openai/gpt-5.2` | same | same | same |
| Hermes WRA matrix Random-k | `hermes` | `openai/gpt-5.2` | same | same | same |
| Claude Code WRA matrix Random-k | `claude-code` | `openai/gpt-5.2` | same | same | same |

The two matrix rows align on executor model and orchestration models. Their main executor difference is the runtime surface: Hermes vs Claude Code.

## Aggregate Outcome

Full-run comparison:

| Metric | Previous logical | Fresh logical | Hermes matrix | Claude Code matrix |
| --- | ---: | ---: | ---: | ---: |
| Runs | 10 | 10 | 24 | 24 |
| Verifier successes | 8 | 9 | 18 | 15 |
| Raw mean / success rate | 0.800 | 0.900 | 0.750 | 0.625 |
| Judge mean | 0.650 | 0.751 | 0.628 | 0.481 |
| Cost basis | Billing + orchestration proxy | Billing + orchestration proxy | Executor proxy + orchestration proxy | Executor billing + orchestration proxy |
| Cost | $6.613 | $5.367 | $5.818 | $10.144 |
| Executor billing/proxy component | included | included | $5.244 proxy | $9.563 billing |
| Raw tokens | 3,109,273 | 2,380,706 | 6,740,898 | 6,500,304 |
| Cache-adjusted tokens | 855,385 | 717,794 | 1,541,116 | 1,557,302 |
| Raw mean / 100k adjusted tokens | 0.094 | 0.125 | 0.049 | 0.040 |
| Judge mean / 100k adjusted tokens | 0.076 | 0.105 | 0.041 | 0.031 |

Main read:

- Fresh logical remains the strongest run on raw mean, judge mean, and adjusted-token reward density.
- Hermes matrix outperforms Claude Code matrix on reward quality: `18/24` vs `15/24` verifier successes and `0.628` vs `0.481` judge mean.
- With the old-Hermes proxy, Hermes matrix is also cheaper than Claude Code matrix in dollars.
- Claude Code matrix uses fewer raw tokens than Hermes matrix, but slightly more cache-adjusted tokens because it has more fresh executor input and similar orchestration load.

## Post-Warmup Dollar Efficiency

This table compares post-warmup to post-warmup. For logical runs, post-warmup means `logical_iter >= 1`, excluding the L0 seed batch. For matrix rows, post-warmup means iterations `1` and `2`, excluding matrix iteration `0`.

| Metric | Previous logical | Fresh logical | Hermes matrix | Claude Code matrix |
| --- | ---: | ---: | ---: | ---: |
| Cost basis | Billing + orchestration proxy | Billing + orchestration proxy | Executor proxy + orchestration proxy | Executor billing + orchestration proxy |
| Rows in scope | 7 | 7 | 16 | 16 |
| Successful task-runs | 7 | 7 | 14 | 13 |
| Raw mean | 1.000 | 1.000 | 0.875 | 0.813 |
| Judge mean | 0.758 | 0.805 | 0.746 | 0.568 |
| Cost | $5.048 | $4.277 | $3.418 | $6.637 |
| Raw mean / $ | 0.198 | 0.234 | 0.256 | 0.122 |
| Judge mean / $ | 0.150 | 0.188 | 0.218 | 0.086 |
| Successful task-runs / $ | 1.387 | 1.636 | 4.096 | 1.959 |
| Total judge-reward points / $ | 1.050 | 1.317 | 3.494 | 1.368 |
| Cost / row | $0.721 | $0.611 | $0.214 | $0.415 |

Main read:

- Fresh logical is best on absolute post-warmup mean quality, but Hermes matrix is best on dollar-normalized quality under the old-Hermes proxy.
- Hermes matrix is best on total post-warmup completed work per dollar and total judge-reward points per dollar.
- Claude Code matrix is faster, but it loses to Hermes on post-warmup dollar efficiency once old-Hermes pricing is used.

## Token Split

Full-run token split:

| Metric | Previous logical | Fresh logical | Hermes matrix | Claude Code matrix |
| --- | ---: | ---: | ---: | ---: |
| Executor fresh input | 361,637 | 312,378 | 416,848 | 532,037 |
| Executor cache-read input | 2,504,320 | 1,847,680 | 5,777,536 | 5,492,224 |
| Executor output | 132,948 | 105,183 | 250,258 | 166,280 |
| Non-executor tokens | 110,368 | 115,465 | 296,256 | 309,763 |
| Raw token total | 3,109,273 | 2,380,706 | 6,740,898 | 6,500,304 |
| Cache-adjusted effective tokens | 855,385 | 717,794 | 1,541,116 | 1,557,302 |

Post-warmup token split:

| Metric | Previous logical | Fresh logical | Hermes matrix | Claude Code matrix |
| --- | ---: | ---: | ---: | ---: |
| Rows | 7 | 7 | 16 | 16 |
| Executor fresh input | 264,077 | 232,914 | 222,329 | 297,242 |
| Executor cache-read input | 1,961,600 | 1,505,536 | 3,593,216 | 4,387,456 |
| Executor output | 103,057 | 86,665 | 142,151 | 101,947 |
| Non-executor tokens | 75,684 | 86,100 | 194,961 | 199,052 |
| Raw token total | 2,404,418 | 1,911,215 | 4,152,657 | 4,985,697 |
| Cache-adjusted effective tokens | 638,978 | 556,233 | 918,763 | 1,036,987 |
| Cache-adjusted tokens / row | 91,283 | 79,462 | 57,423 | 64,812 |
| Cost | $5.048 | $4.277 | $3.418 | $6.637 |

Main read:

- The requested post-warmup split is complete here: fresh input, cache input, output, non-executor tokens, raw total, adjusted total, adjusted tokens per row, and cost are all separated.
- Claude Code matrix post-warmup is faster than Hermes matrix, but under old-Hermes pricing it is not cheaper and it is more token-heavy after cache adjustment: `1.037M` vs `0.919M`.
- Hermes matrix has much more executor output full-run (`250,258` vs `166,280`), while Claude Code matrix has more fresh input and post-warmup cache input.
- Fresh logical remains the best post-warmup adjusted-token run among the logical rows.

## First Matrix Task Row

Both matrix rows start with `Weighted-Risk-Assessment/api-sla-at-risk-calc` at iteration `0`.

| Metric | Hermes matrix first row | Claude Code matrix first row |
| --- | ---: | ---: |
| Verifier reward | 0.0 | 1.0 |
| Executor fresh input | 31,832 | 24,633 |
| Executor cache-read input | 274,304 | 103,552 |
| Total executor input | 306,136 | 128,185 |
| Executor output | 10,209 | 6,523 |
| Executor billing/proxy | $0.246635 proxy | $0.338016 billing |

## Wall Clock

Durations are summed from row-level `duration_sec`.

| Metric | Previous logical | Fresh logical | Hermes matrix | Claude Code matrix |
| --- | ---: | ---: | ---: | ---: |
| Full duration | 1.874 h | 0.904 h | 4.650 h | 1.932 h |
| Post-warmup duration | 1.382 h | 0.693 h | 1.854 h | 1.359 h |
| Post-warmup successful task-runs / hour | 5.066 | 10.094 | 7.551 | 9.568 |

Main read:

- Fresh logical is fastest on post-warmup throughput.
- Claude Code matrix is much faster than Hermes matrix: full run `1.93 h` vs `4.65 h`, and post-warmup `1.36 h` vs `1.85 h`.
- Hermes still beats Claude Code on reward quality despite taking longer.

## Learning Shape

Logical levels vs matrix iterations:

| Stage | Previous logical | Fresh logical | Hermes matrix | Claude Code matrix |
| --- | --- | --- | --- | --- |
| Seed | L0: 1/3 success, judge 0.399 | L0: 2/3 success, judge 0.627 | Iter 0: 4/8 success, judge 0.393 | Iter 0: 2/8 success, judge 0.309 |
| First diffusion | L1: 3/3 success, judge 0.771 | L1: 3/3 success, judge 0.816 | Iter 1: 7/8 success, judge 0.749 | Iter 1: 6/8 success, judge 0.503 |
| Follow-up | L2: 2/2 success, judge 0.740 | L2: 2/2 success, judge 0.778 | Iter 2: 7/8 success, judge 0.744 | Iter 2: 7/8 success, judge 0.632 |
| Continuation | L3: 2/2 success, judge 0.755 | L3: 2/2 success, judge 0.815 | n/a | n/a |

Main read:

- Both logical runs learn immediately after the seed batch: post-warmup logical rows are `7/7` successes in both runs.
- Hermes matrix improves after the first sweep, from `4/8` to `7/8`, then holds at `7/8`.
- Claude Code matrix also improves, from `2/8` to `6/8` to `7/8`, but its judge score remains below Hermes at every matrix iteration.
- The matrix approach requires a full 8-task seed pass before the strong post-warmup phase appears; logical diffusion reaches its high-success phase after a 3-task seed batch.

## Claude Code vs Hermes Matrix

The two matrix rows are the closest apples-to-apples comparison because they share the same WRA task family, seed, `skill_none_random_k` baseline, `openai/gpt-5.2` executor model, planner, mediator, judge, and 24-row structure.

| Metric | Hermes matrix | Claude Code matrix | Read |
| --- | ---: | ---: | --- |
| Executor runtime | `hermes` | `claude-code` | Main experimental difference |
| Executor cost source | proxy | raw billing | Hermes lacks executor billing |
| Full verifier successes | 18/24 | 15/24 | Hermes better |
| Full judge mean | 0.628 | 0.481 | Hermes better |
| Full cost | $5.818 | $10.144 | Hermes lower |
| Full duration | 4.650 h | 1.932 h | Claude Code faster |
| Post-warmup successes | 14/16 | 13/16 | Hermes slightly better |
| Post-warmup judge mean | 0.746 | 0.568 | Hermes better |
| Post-warmup cost | $3.418 | $6.637 | Hermes lower |
| Post-warmup adjusted tokens | 918,763 | 1,036,987 | Hermes lighter |
| Post-warmup successes/hour | 7.551 | 9.568 | Claude Code faster |
| Post-warmup judge-points/$ | 3.494 | 1.368 | Hermes better |

Main read:

- Claude Code is the faster matrix executor.
- Under old-Hermes proxy pricing, Hermes is also cheaper than Claude Code while remaining stronger on reward, judge quality, adjusted-token efficiency, and judge-points per dollar.
- The difference is not explained by model selection; both use `openai/gpt-5.2` as executor model. It is more likely from executor runtime behavior, prompt/turn handling, and artifact use.

## Diffusion Mechanism

The logical and matrix runs should not be described as using the same selector.

- Logical runs use a controller-selected logical batch sequence. The fresh run also has a continuation caveat in `tmp/agent_diffusion_logical_wra_realcost_fresh_20260626/analysis/checkpoint_L02.original_softmax_8row.json` and `tmp/agent_diffusion_logical_wra_realcost_fresh_20260626/analysis/checkpoint_L02.manual_10row_continuation_note.json`.
- Matrix rows use Random-k artifact diffusion. Evidence is in `tmp/wra_matrix_randomk_gpt52_20260627/data/experiments/20260627-010707-wra-random-k-gpt52-20260627/diffusion/diffused_records.jsonl` and `tmp/wra_matrix_randomk_gpt52_claudecode_20260627/data/experiments/20260627-112631-wra-random-k-gpt52-claudecode-20260627/diffusion/diffused_records.jsonl`.
- Hermes matrix selected/rendered 0 artifacts in iteration 0 and 4 artifacts for every task in iterations 1 and 2.
- Claude Code matrix selected/rendered 0 artifacts in iteration 0, 2 or 3 artifacts in iteration 1, and 4 artifacts for every task in iteration 2.
- Matrix source tasks were sampled from eligible prior artifacts. They were not chosen by a softmax gate or task-similarity targeting, so the matrix rows demonstrate broad Random-k diffusion rather than targeted logical routing.

## Bottom Line

- Fresh logical is best for absolute post-warmup mean quality, adjusted-token efficiency, and post-warmup wall-clock throughput.
- Hermes matrix is best for dollar efficiency under old-Hermes proxy pricing: mean reward per dollar, total post-warmup completed work per dollar, and total judge-reward points per dollar.
- Claude Code matrix is best among matrix rows for wall-clock speed, but not for dollar efficiency after applying the old-Hermes executor proxy.
- The clean phrasing is: fresh logical learns fastest and is most token-efficient per row; Hermes matrix amortizes a larger campaign and converts diffusion into the best post-warmup reward per dollar under old-Hermes pricing; Claude Code matrix runs fast but spends more adjusted post-warmup input for lower judged quality.
