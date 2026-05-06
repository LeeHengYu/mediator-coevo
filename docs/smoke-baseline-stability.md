# Baseline Stability Smoke Result

Status: **EXECUTED — Docker/Compose available; all six baseline rows produced scored traces with zero environment failures**.

This document is smoke validation tracking only, not scientific evidence of superiority. The required smoke command completed with exit code 0 after installing Harbor, Docker CLI, Docker Compose, and Colima. The run produced one row summary for each of the six approved baseline rows.

## Required command

```bash
uv run medcoevo matrix --tasks fix-build-google-auto --iterations 1 --seed 42
```

Run artifact root:

```text
data/experiments/20260505-214850-42-baseline-matrix
```

Runtime evidence:

```text
Docker CLI: 29.4.2
Docker server: 29.2.1 via Colima
Docker Compose: 5.1.3
Harbor: 0.6.4
```

## Smoke table

| row | condition | skill updates | tasks | seed | iterations | scored count | env failures | trace status | reward | mean reward | macro mean | total tokens | duration | notes | date | repo commit | benchmark provenance | Harbor agent/model | remote-fetch status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| no_feedback | no_feedback | none | fix-build-google-auto | 42 | 1 | 1 | 0 | ok | 0.000 | 0.000 | 0.000 | 5975460 | 469.5s | Executed; trace status=ok; error_kind=none. | 2026-05-06 | efdff89 | Local task present at benchmarks/skillsbench/tasks/fix-build-google-auto. | opencode / openrouter/google/gemini-3-flash-preview | Not needed; local task resolved from repository cache. |
| full_trace_same_task | full_traces | none | fix-build-google-auto | 42 | 1 | 1 | 0 | ok | 0.000 | 0.000 | 0.000 | 6078078 | 336.5s | Executed; trace status=ok; error_kind=none. | 2026-05-06 | efdff89 | Local task present at benchmarks/skillsbench/tasks/fix-build-google-auto. | opencode / openrouter/google/gemini-3-flash-preview | Not needed; local task resolved from repository cache. |
| static_mediator_same_task | static_mediator | none | fix-build-google-auto | 42 | 1 | 1 | 0 | ok | 0.000 | 0.000 | 0.000 | 8356970 | 1775.6s | Executed; trace status=ok; error_kind=none. | 2026-05-06 | efdff89 | Local task present at benchmarks/skillsbench/tasks/fix-build-google-auto. | opencode / openrouter/google/gemini-3-flash-preview | Not needed; local task resolved from repository cache. |
| planner_only_skill_evolution | learned_mediator | planner | fix-build-google-auto | 42 | 1 | 1 | 0 | ok | 0.000 | 0.000 | 0.000 | 10631339 | 448.4s | Executed; trace status=ok; error_kind=none. | 2026-05-06 | efdff89 | Local task present at benchmarks/skillsbench/tasks/fix-build-google-auto. | opencode / openrouter/google/gemini-3-flash-preview | Not needed; local task resolved from repository cache. |
| mediator_only_protocol_evolution | learned_mediator | mediator | fix-build-google-auto | 42 | 1 | 1 | 0 | ok | 0.000 | 0.000 | 0.000 | 5315116 | 282.1s | Executed; trace status=ok; error_kind=none. | 2026-05-06 | efdff89 | Local task present at benchmarks/skillsbench/tasks/fix-build-google-auto. | opencode / openrouter/google/gemini-3-flash-preview | Not needed; local task resolved from repository cache. |
| full_coevolution | learned_mediator | executor, planner, mediator | fix-build-google-auto | 42 | 1 | 1 | 0 | ok | 0.000 | 0.000 | 0.000 | 4817555 | 309.0s | Executed; trace status=ok; error_kind=none. | 2026-05-06 | efdff89 | Local task present at benchmarks/skillsbench/tasks/fix-build-google-auto. | opencode / openrouter/google/gemini-3-flash-preview | Not needed; local task resolved from repository cache. |

## Diffusion gate

This is real scored matrix smoke evidence for six-row wiring, design validation, Harbor execution, Docker/Compose execution, trace parsing, row-level aggregation, and failure accounting. It is still not evidence of benchmark superiority or task-solving quality: it covers one task, one iteration per row, and every row received reward `0.000`. Use this smoke as the baseline-stability plumbing gate, not as a statistical result.
