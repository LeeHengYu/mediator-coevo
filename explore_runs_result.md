# Diffusion Findings

Pricing source: OpenRouter Models API/docs. Proxy dollars use current per-token `prompt`, `completion`, and `input_cache_read` prices from `https://openrouter.ai/api/v1/models`. If a model has no `input_cache_read` field, recorded cache-read tokens are not charged in the proxy dollar calculation. For `moonshotai/kimi-k2.5`, the current OpenRouter fields are `prompt = 0.000000375`, `completion = 0.000002025`, and no `input_cache_read`.

Definitions: `success` means verifier reward is present and greater than `0`. `Successes / $` and `$ / success` are the main dollar-efficiency metrics. Lower `tokens / success` means the method does more successful work with less token spend.

Important comparability caveat: executor model differs across many cells. WRA has a same-model GPT-5.2 comparison between deterministic `random_k` and softmax. HWPX softmax comparisons are cross-model against older deterministic baselines, so their dollar-efficiency and token-efficiency conclusions are directionally useful but not fully controlled.

## Executor Model Inventory

| Family | Mode / policy                   | Executor model(s)                     |
| ------ | ------------------------------- | ------------------------------------- |
| WRA    | no diffusion                    | `openai/gpt-5.5`                      |
| WRA    | deterministic: capped_broadcast | `openai/gpt-5.5`                      |
| WRA    | deterministic: random_k         | `openai/gpt-5.5` and `openai/gpt-5.2` |
| WRA    | deterministic: top_k_similarity | `openai/gpt-5.5`                      |
| WRA    | softmax gates                   | `openai/gpt-5.2`                      |
| HWPX   | deterministic policies          | `moonshotai/kimi-k2.5`                |
| HWPX   | softmax gates                   | `openai/gpt-5` and `openai/gpt-5.2`   |

## Efficiency Summary

| Family | Mode / policy                   | Runs | Successes | Verifier mean | Judge mean | Proxy $ | Runs / $ | Successes / $ | $ / success | Avg tokens / run | Tokens / success |
| ------ | ------------------------------- | ---: | --------: | ------------: | ---------: | ------: | -------: | ------------: | ----------: | ---------------: | ---------------: |
| WRA    | deterministic: capped_broadcast |   72 |        52 |         0.743 |      0.607 |   4.183 |    17.21 |         12.43 |       0.080 |           20,812 |           28,816 |
| WRA    | deterministic: random_k         |  120 |        87 |         0.731 |      0.630 |  11.374 |    10.55 |          7.65 |       0.131 |           75,117 |          103,610 |
| WRA    | deterministic: top_k_similarity |   72 |        53 |         0.736 |      0.623 |   4.227 |    17.04 |         12.54 |       0.080 |           22,027 |           29,924 |
| WRA    | softmax gates                   |   42 |        36 |         0.878 |      0.698 |  12.862 |     3.27 |          2.80 |       0.357 |          252,939 |          295,095 |
| HWPX   | deterministic                   |   72 |        61 |         0.884 |      0.682 |   2.049 |    35.14 |         29.77 |       0.034 |           57,280 |           67,609 |
| HWPX   | softmax gates                   |   37 |        28 |         0.848 |      0.643 |  10.179 |     3.64 |          2.75 |       0.364 |          326,420 |          431,341 |

## Same-Model Comparable View

The cleanest same-executor comparison is WRA with `openai/gpt-5.2`. HWPX lacks same-model deterministic-vs-softmax controls in the current artifact set.

| Family | Comparable group                             | Executor model   | Runs | Successes | Verifier mean | Judge mean | Proxy $ | Successes / $ | $ / success | Avg tokens / run | Tokens / success |
| ------ | -------------------------------------------- | ---------------- | ---: | --------: | ------------: | ---------: | ------: | ------------: | ----------: | ---------------: | ---------------: |
| WRA    | deterministic random_k                       | `openai/gpt-5.2` |   48 |        33 |         0.688 |         NA |   7.149 |          4.62 |       0.217 |          155,493 |          226,172 |
| WRA    | deterministic random_k warmup, iter 0        | `openai/gpt-5.2` |   16 |         6 |         0.375 |         NA |   2.411 |          2.49 |       0.402 |          119,908 |          319,755 |
| WRA    | deterministic random_k post-warmup, iter > 0 | `openai/gpt-5.2` |   32 |        27 |         0.844 |         NA |   4.738 |          5.70 |       0.175 |          173,286 |          205,375 |
| WRA    | softmax all                                  | `openai/gpt-5.2` |   42 |        36 |         0.878 |      0.698 |  12.862 |          2.80 |       0.357 |          252,939 |          295,095 |
| WRA    | softmax warmup                               | `openai/gpt-5.2` |   13 |         8 |         0.615 |      0.573 |   3.056 |          2.62 |       0.382 |          198,009 |          321,765 |
| WRA    | softmax post-warmup                          | `openai/gpt-5.2` |   29 |        28 |         1.000 |      0.756 |   9.806 |          2.86 |       0.350 |          277,562 |          287,475 |

Within GPT-5.2 WRA, the fair post-warmup comparison is deterministic `random_k` iter `> 0` versus softmax post-warmup. Softmax improves verifier performance from `0.844` to `1.000`, but still loses on efficiency: deterministic post-warmup gets `5.70` successes per dollar and `205k` tokens per success, while softmax post-warmup gets `2.86` successes per dollar and `287k` tokens per success.

## Cross-Model Baseline View

These rows are useful for current campaign accounting, but not fully controlled because executor model differs.

| Family | Group                               | Executor model         | Runs | Successes | Verifier mean | Judge mean | Proxy $ | Successes / $ | $ / success | Avg tokens / run | Tokens / success |
| ------ | ----------------------------------- | ---------------------- | ---: | --------: | ------------: | ---------: | ------: | ------------: | ----------: | ---------------: | ---------------: |
| WRA    | best deterministic GPT-5.5 random_k | `openai/gpt-5.5`       |   72 |        54 |         0.761 |      0.630 |   4.225 |         12.78 |       0.078 |           21,533 |           28,711 |
| HWPX   | deterministic all                   | `moonshotai/kimi-k2.5` |   72 |        61 |         0.884 |      0.682 |   2.049 |         29.77 |       0.034 |           57,280 |           67,609 |
| HWPX   | softmax GPT-5 post-warmup           | `openai/gpt-5`         |    6 |         4 |         1.000 |      0.764 |   1.300 |          3.08 |       0.325 |          227,841 |          341,762 |
| HWPX   | softmax GPT-5.2 post-warmup         | `openai/gpt-5.2`       |   21 |        18 |         0.900 |      0.675 |   5.884 |          3.06 |       0.327 |          379,444 |          442,684 |

## HWPX Deterministic Policy Breakdown

The HWPX deterministic rows use the same executor model, `moonshotai/kimi-k2.5`. The diffusion-only aggregate is `capped_broadcast + random_k + top_k_similarity`; the no-diffusion row is shown as a baseline, but is not included in the `72`-run deterministic diffusion aggregate above.

| Policy             | Runs | Successes | Verifier mean | Judge mean | Proxy $ | Runs / $ | Successes / $ | $ / success | Avg tokens / run | Tokens / success | Env failures |
| ------------------ | ---: | --------: | ------------: | ---------: | ------: | -------: | ------------: | ----------: | ---------------: | ---------------: | -----------: |
| `none`             |   24 |        19 |         0.792 |      0.638 |   0.641 |    37.42 |         29.63 |       0.034 |           47,853 |           60,446 |            0 |
| `capped_broadcast` |   24 |        21 |         0.913 |      0.708 |   0.666 |    36.03 |         31.53 |       0.032 |           55,250 |           63,143 |            1 |
| `random_k`         |   24 |        22 |         0.917 |      0.667 |   0.578 |    41.54 |         38.08 |       0.026 |           43,794 |           47,775 |            0 |
| `top_k_similarity` |   24 |        18 |         0.818 |      0.672 |   0.805 |    29.80 |         22.35 |       0.045 |           72,795 |           97,060 |            2 |

Cost reproduction with current OpenRouter Kimi K2.5 fields:

| Policy                       | Executor prompt tokens | Executor completion tokens | Formula                                        | Proxy $ |
| ---------------------------- | ---------------------: | -------------------------: | ---------------------------------------------- | ------: |
| `none`                       |                719,244 |                    183,514 | `719244 * 0.000000375 + 183514 * 0.000002025`  |   0.641 |
| `capped_broadcast`           |                926,134 |                    157,415 | `926134 * 0.000000375 + 157415 * 0.000002025`  |   0.666 |
| `random_k`                   |                629,282 |                    168,762 | `629282 * 0.000000375 + 168762 * 0.000002025`  |   0.578 |
| `top_k_similarity`           |              1,366,188 |                    144,746 | `1366188 * 0.000000375 + 144746 * 0.000002025` |   0.805 |
| deterministic diffusion-only |              2,921,604 |                    470,923 | `2921604 * 0.000000375 + 470923 * 0.000002025` |   2.049 |

Within HWPX deterministic policies, `random_k` is the best current policy by both dollar efficiency and token efficiency: `38.08` successes per dollar, `$0.026` per success, and `47,775` tokens per success.

## Token Split

| Family | Mode / policy                   | Fresh input | Cache input | Output | Cache share |
| ------ | ------------------------------- | ----------: | ----------: | -----: | ----------: |
| WRA    | deterministic: capped_broadcast |       0.65M |           0 |  0.19M |        0.0% |
| WRA    | deterministic: random_k         |       1.67M |       5.49M |  0.49M |       76.6% |
| WRA    | deterministic: top_k_similarity |       0.68M |           0 |  0.19M |        0.0% |
| WRA    | softmax gates                   |       1.48M |       8.55M |  0.60M |       85.2% |
| HWPX   | deterministic                   |       0.53M |           0 |  0.20M |        0.0% |
| HWPX   | softmax gates                   |       1.10M |      10.50M |  0.48M |       90.5% |

## Softmax Post-Warmup Efficiency

Warmup is the seed/no-transfer phase: logical iteration `0` or rows with `transfer_context_tokens == 0`. Post-warmup rows are the activated logical-executor rows that actually consume transfer context.

| Family | Phase       | Runs | Successes | Verifier mean | Judge mean | Proxy $ | Successes / $ | $ / success | Avg tokens / run | Tokens / success |
| ------ | ----------- | ---: | --------: | ------------: | ---------: | ------: | ------------: | ----------: | ---------------: | ---------------: |
| WRA    | all softmax |   42 |        36 |         0.878 |      0.698 |  12.862 |          2.80 |       0.357 |          252,939 |          295,095 |
| WRA    | warmup only |   13 |         8 |         0.615 |      0.573 |   3.056 |          2.62 |       0.382 |          198,009 |          321,765 |
| WRA    | post-warmup |   29 |        28 |         1.000 |      0.756 |   9.806 |          2.86 |       0.350 |          277,562 |          287,475 |
| HWPX   | all softmax |   37 |        28 |         0.848 |      0.643 |  10.179 |          2.75 |       0.364 |          326,420 |          431,341 |
| HWPX   | warmup only |   10 |         6 |         0.667 |      0.519 |   2.994 |          2.00 |       0.499 |          274,218 |          457,030 |
| HWPX   | post-warmup |   27 |        22 |         0.917 |      0.689 |   7.185 |          3.06 |       0.327 |          345,754 |          424,335 |

## Startup Cost

The logical executor starts cheaply because it runs only a seed subset before activating transfer. The intended startup scale is around `1/e = 0.368` of the total target set; observed warmup shares are close to that or lower in most current softmax folders.

| Root         | Family | Runs | Warmup | Post-warmup | Warmup share |   1/e | All verifier | Post-warmup verifier | Total $ | Warmup $ | Post-warmup $ |
| ------------ | ------ | ---: | -----: | ----------: | -----------: | ----: | -----------: | -------------------: | ------: | -------: | ------------: |
| HWPX current | HWPX   |   15 |      3 |          12 |        0.200 | 0.368 |        0.786 |                0.909 |   4.812 |    1.333 |         3.479 |
| HWPX GPT-5   | HWPX   |   10 |      4 |           6 |        0.400 | 0.368 |        1.000 |                1.000 |   2.214 |    0.913 |         1.300 |
| HWPX router  | HWPX   |   12 |      3 |           9 |        0.250 | 0.368 |        0.833 |                0.889 |   3.153 |    0.748 |         2.405 |
| WRA fusion   | WRA    |   12 |      3 |           9 |        0.250 | 0.368 |        0.917 |                1.000 |   3.427 |    0.700 |         2.727 |
| WRA router   | WRA    |   10 |      4 |           6 |        0.400 | 0.368 |        0.889 |                1.000 |   2.758 |    0.894 |         1.865 |
| WRA realcost | WRA    |   10 |      3 |           7 |        0.300 | 0.368 |        0.800 |                1.000 |   3.575 |    0.842 |         2.734 |
| WRA fresh    | WRA    |   10 |      3 |           7 |        0.300 | 0.368 |        0.900 |                1.000 |   3.101 |    0.621 |         2.480 |

## Five Findings

1. **Softmax's main controlled advantage is WRA post-warmup performance.** In the same `openai/gpt-5.2` WRA comparison, deterministic `random_k` post-warmup has verifier mean `0.844`, while softmax post-warmup reaches `1.000`. The performance gain is real in the controlled WRA subset, even though judge reward for GPT-5.2 deterministic random-k is unavailable.

2. **Softmax still loses dollar efficiency in the same-model post-warmup WRA comparison.** GPT-5.2 deterministic `random_k` post-warmup gets `5.70` successes per dollar and `0.175` dollars per success. GPT-5.2 softmax post-warmup gets `2.86` successes per dollar and `0.350` dollars per success.

3. **The startup cost is cheap by design.** Softmax/logical-executor runs begin with a seed subset near `1/e` of the full target set. Observed warmup shares are `0.25-0.40` for WRA and HWPX softmax roots, and the current HWPX run is even lower at `0.20`. This means the method can test transfer viability before paying for a full-family campaign.

4. **Current softmax does not yet lower token usage per success.** In same-model WRA post-warmup, GPT-5.2 softmax uses `287k` tokens per success, while GPT-5.2 deterministic `random_k` uses `205k`. Cross-model HWPX is worse: softmax GPT-5.2 post-warmup uses `443k` tokens per success, while deterministic Kimi uses `68k`.

5. **HWPX remains the needed easy-task control.** Current HWPX evidence is promising post-warmup, but it is cross-model: deterministic rows use `moonshotai/kimi-k2.5`, while softmax rows use GPT-5/GPT-5.2. The next HWPX runs should keep the executor fixed at Kimi/K2.5 for both deterministic and softmax.

## Current Interpretation

The current evidence supports this narrower claim: in the same-model WRA post-warmup subset, softmax diffusion improves performance quality, and it has a cheap startup phase because it seeds only about `1/e` of the target set before activation. It is not yet a cost-saving method in the completed runs: even in same-model post-warmup WRA, it spends more tokens and more dollars per successful task than deterministic `random_k`. HWPX should be treated as cross-model evidence until matching deterministic and softmax runs exist for the same executor model.

## Remaining Runs Required

The remaining plan should complete two comparable families:

1. WRA with the default executor, `openrouter/openai/gpt-5.2`.
2. HWPX as the easier family with `moonshotai/kimi-k2.5`.

For both families, use the same seed, same iteration budget, same skill-update setting, and compare deterministic post-warmup (`iter > 0`) against softmax post-warmup.

### Required Run Grid

| Priority | Family | Executor                    | Runs needed                                          | Status                                   | Purpose / remaining work                                                                                                                                                                                                                                       |
| -------- | ------ | --------------------------- | ---------------------------------------------------- | ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| P0       | WRA    | `openrouter/openai/gpt-5.2` | deterministic rows `0,1,2,3`                         | Partial Claude Code evidence             | Count only `executor_agent = claude-code`. Row `2` `random_k` has Claude Code evidence in `tmp/wra_matrix_randomk_gpt52_claudecode_20260627`: 24 runs / 15 successes. The Hermes row-2 run does not count. Rows `0`, `1`, and `3` still need Claude Code runs. |
| P0       | WRA    | `openrouter/openai/gpt-5.2` | `llm_router_softmax`                                 | Completed with Claude Code temp evidence | `tmp/agent_diffusion_logical_wra_llmrouter_temp_gpt52_20260630` has 10 runs / 8 successes with seed `42`, executor `openai/gpt-5.2`, and `executor_agent = claude-code`. Optional rerun under `final-wra-gpt52-softmax` for clean naming/config parity.        |
| P0       | HWPX   | `moonshotai/kimi-k2.5`      | deterministic rows `0,1,2,3`                         | Missing Claude Code evidence             | `data/experiments/HDA-Hermes` has rows `0`, `1`, `2`, and `3`, but they use `executor_agent = hermes`, so they do not count for the final grid. Run all four deterministic rows with Claude Code.                                                              |
| P0       | HWPX   | `moonshotai/kimi-k2.5`      | `llm_router_softmax`                                 | Missing                                  | Existing HWPX softmax runs use GPT-5/GPT-5.2, not Kimi. Run `final-hwpx-kimi25-softmax`.                                                                                                                                                                       |
| P1       | WRA    | `openrouter/openai/gpt-5.2` | repeat best deterministic + softmax with seed `1023` | Pending                                  | Robustness check after the seed-42 story is complete.                                                                                                                                                                                                          |
| P1       | HWPX   | `moonshotai/kimi-k2.5`      | repeat best deterministic + softmax with seed `1023` | Pending                                  | Easy-task robustness check after P0 is complete.                                                                                                                                                                                                               |

Counting only Claude Code, substantively remaining P0 work: WRA GPT-5.2 deterministic rows `0`, `1`, and `3`; HWPX Kimi deterministic rows `0`, `1`, `2`, and `3`; and HWPX Kimi `llm_router_softmax`. Strictly completed under the final `final-*` run IDs: none found.

Matrix row indexes:

```text
0: no diffusion
1: capped_broadcast
2: random_k
3: top_k_similarity
```

### Config Setup

Create two config dirs copied from `config/default.toml`.

`config/final-gpt52/default.toml`:

```toml
[models]
executor = "openrouter/openai/gpt-5.2"

[experiment]
num_iterations = 4
seed = 42
condition_name = "learned_mediator"
coevo_interval = 99
advisor_buffer_max = 99

[experiment.skill_updates]
executor = false
planner = false
mediator = false

[diffusion]
llm_router_model = "openrouter/openai/gpt-5.2"
logical_seed_count = 3
softmax_temperature = 0.35
llm_router_weight = 0.30
consecutive_iteration_limit = 2
```

`config/final-kimi25/default.toml`:

```toml
[models]
executor = "moonshotai/kimi-k2.5"

[experiment]
num_iterations = 4
seed = 42
condition_name = "learned_mediator"
coevo_interval = 99
advisor_buffer_max = 99

[experiment.skill_updates]
executor = false
planner = false
mediator = false

[diffusion]
llm_router_model = "openrouter/openai/gpt-5.2"
logical_seed_count = 3
softmax_temperature = 0.35
llm_router_weight = 0.30
consecutive_iteration_limit = 2
```

### Existing Log Rename Map

Only Claude Code logs count. Hermes logs should stay as prior evidence and should not be renamed into `final-*`.

| Use                                   | Current folder                                                                                                             | Rename / copy target                                           | Notes                                                                                                                                                      |
| ------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| WRA deterministic row `2`, `random_k` | `tmp/wra_matrix_randomk_gpt52_claudecode_20260627/data/experiments/20260627-112631-wra-random-k-gpt52-claudecode-20260627` | `data/experiments/final-wra-gpt52-deterministic-row2-random-k` | Usable Claude Code evidence: 24 runs / 15 successes, seed `42`, executor `openai/gpt-5.2`. Not full row `0,1,2,3`; rows `0`, `1`, and `3` still need runs. |
| WRA softmax                           | `tmp/agent_diffusion_logical_wra_llmrouter_temp_gpt52_20260630`                                                            | `data/experiments/final-wra-gpt52-softmax`                     | Usable Claude Code temp evidence: 10 runs / 8 successes, seed `42`, executor `openai/gpt-5.2`.                                                             |
| HWPX deterministic Kimi rows          | `data/experiments/HDA-Hermes/*`                                                                                            | Do not rename                                                  | Hermes only; does not count for final Claude Code grid.                                                                                                    |
| HWPX softmax                          | `data/experiments/20260702-005247-HWPX-logical-llmrouter-12iter-current`                                                   | Do not rename                                                  | Claude Code, but executor is `openai/gpt-5.2`, not Kimi; does not satisfy HWPX Kimi P0.                                                                    |

### OpenRouter cost table (/M tok)

| Model                         | Fresh input | Cache input | Output |
| ----------------------------- | ----------- | ----------- | ------ |
| openai/gpt-5.2                | 1.75        | 0.175       | 14     |
| moonshotai/kimi-k2.5          | 0.375       | 0.15        | 2.025  |
| google/gemini-3-flash-preview | 0.5         | 0.05        | 3      |
| anthropic/claude-opus-4.6     | 5           | 0.5         | 20     |
| openai/gpt-oss-120b           | 0           | 0           | 0      |
