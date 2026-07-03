# Diffusion Results for Paper Framing

This note is intentionally paper-facing. It keeps the HWPX result, the cost caveat, and the remaining run status while omitting phase-split comparisons and verbose cross-run accounting.

Pricing: Kimi proxy dollars use fresh input `$0.375/M`, cache input `$0.0375/M`, and output `$2.025/M`. Total proxy cost includes executor proxy plus planner, mediator/compactor, and router proxy. Harbor executor billing is tracked for audit but is not used for Kimi dollar-efficiency claims.

## HWPX Main Comparison

All rows below use Claude Code with `moonshotai/kimi-k2.5` as executor. Deterministic policies use seed `42`, 3 matrix iterations, 8 HWPX tasks per iteration. `llm_router_softmax` uses the current main infra Kimi run with 21 logical task rows.

| Policy | Runs | Successes | Verifier mean | Judge mean | Total proxy $ | $ / success | Paper read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `none` | 24 | 18 | 0.783 | 0.608 | 2.286 | 0.127 | Baseline is already nontrivial. |
| `capped_broadcast` | 24 | 20 | 0.833 | 0.631 | 2.220 | 0.111 | Best simple HWPX baseline. |
| `random_k` | 24 | 18 | 0.750 | 0.604 | 2.554 | 0.142 | Cheap exploration, but not cost-best here. |
| `llm_router_softmax` | 21 | 16 | 0.762 | 0.585 | 2.548 | 0.159 | Adaptive routing, but not yet total-cost winner. |

`top_k_similarity` was also run as a diagnostic deterministic policy: 24 runs, 19 successes, verifier mean `0.826`, judge mean `0.664`, total proxy `$2.544`, `$0.134/success`. It is useful appendix evidence, but the main paper table can omit it to keep the story focused.

## Paper Interpretation

The clean HWPX result is not "softmax beats all baselines." The better paper claim is:

> On HWPX, direct softmax diffusion gives an adaptive exploration mechanism, but simple deterministic diffusion remains a strong cost baseline. The router's current value is selectivity and experiment control, not raw dollar efficiency yet.

This is still useful:

- It shows the baselines are meaningful and not strawmen.
- `capped_broadcast` gives a strong lower bound for any learned router.
- `random_k` tests whether learned routing beats cheap stochastic reuse.
- `llm_router_softmax` exposes where the new architecture spends extra cost: planner, mediator, and router overhead.

The important nuance is executor-only versus total proxy cost. Executor-only, Kimi softmax is close to deterministic diffusion: `$0.816 / 16 = $0.051/success`, while deterministic diffusion-only is `$3.102 / 57 = $0.054/success`. After adding planner, mediator, and router cost, softmax loses: `$0.159/success` versus `$0.128/success` for deterministic diffusion-only.

## Explanation of Outcome

The current HWPX softmax run appears to pay for richer routing before it converts that routing into enough additional successes. Deterministic diffusion is blunt but cheap; it spreads reusable artifacts broadly without router calls. On an easier family like HWPX, that simple reuse is already strong, so the learned router has less room to show a cost advantage.

For the paper, frame HWPX as a diagnostic result:

- HWPX validates that direct diffusion can be evaluated under controlled same-model, same-agent conditions.
- HWPX shows simple diffusion can be a very competitive baseline.
- HWPX motivates the next router changes: budget-aware routing, less over-exploitation, and stricter selection when deterministic broadcast already performs well.

Avoid overclaiming. The strongest statement is that softmax is a promising adaptive mechanism, not that it is currently the cheapest HWPX policy.

## Run Status

HWPX Kimi seed-42 is complete:

| Slot | Path | Status |
| --- | --- | --- |
| `none` | `data/experiments/20260703-095745-final-hwpx-kimi25-row0-none-3iter-runtimefix2` | Complete |
| `capped_broadcast` | `data/experiments/20260703-134237-final-hwpx-kimi25-row1-capped-broadcast-3iter-runtimefix2` | Complete |
| `random_k` | `data/experiments/20260703-170912-final-hwpx-kimi25-row2-random-k-3iter-runtimefix2` | Complete |
| `top_k_similarity` | `data/experiments/20260703-195346-final-hwpx-kimi25-row3-top-k-similarity-3iter-runtimefix2` | Complete |
| `llm_router_softmax` | `data/experiments/20260702-194153-final-hwpx-kimi25-softmax-10iter-maininfra` | Complete |

Remaining P0 work is WRA GPT-5.2 deterministic rows `0`, `1`, and `3`. HWPX no longer blocks the seed-42 comparison.
