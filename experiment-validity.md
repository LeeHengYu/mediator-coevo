# Prompt-Budget Confounding

Status: Partial pass.

The infrastructure now records same-task prior tokens, cross-task prior tokens,
diffusion tokens, total planner prior-context tokens, effective caps, compaction
events, and budget drops. It also caps diffusion context with
`budgets.max_diffusion_context_tokens` and provides a read-only command for
comparing two completed runs.

The remaining deficit is causal matching of usefulness: the system can audit
and constrain realized context length after the run, but it still does not
guarantee that diffusion and non-diffusion conditions receive matched useful
context or matched semantic information. Programmatic controls can equalize
token counts, provenance, timing, and future-leakage constraints, but they
cannot establish that two equally long context snippets are equally actionable,
specific, relevant, or transferable.

This leaves a semantic-usefulness loophole. A diffusion condition may receive a
short concrete fix while a non-diffusion condition receives equally long but
vague feedback. A budget-only comparison can reduce prompt-budget confounding,
but it is not proof that observed effects came from diffusion rather than from
differences in retained context content. Closing this would require an
additional usefulness evaluation layer, such as human annotation, blinded LLM
judging before outcomes are known, placebo/matched irrelevant context, explicit
artifact citation/use analysis, or ablation runs with the same context removed.

# Multiple Comparisons Grid (Skill update disabled)

## Purpose

This grid is a minimal procedural answer to the multiple-comparisons concern in
`validity.md`: avoid reporting only the best positive result after trying many
task, seed, policy, or graph combinations.

No infrastructure change is required for the minimal solution. The requirement is
to define the grid before interpreting results, run the full grid, and report all
attempted cells.

## Task Run Definition

One TASK RUN means:

```text
1 task executed once under 1 policy, 1 seed, and 1 iteration
```

Grid size:

```text
TASK RUNS = tasks x iterations x seeds x policies
```

## Primary Context-Only Diffusion Grid

This is the recommended first meaningful grid.

| Dimension         | Value                                                                 |
| ----------------- | --------------------------------------------------------------------- |
| Task set          | self-defined.                                                         |
| Tasks             | 10                                                                    |
| Iterations        | 3 minimum, 5 preferred                                                |
| Seeds             | 3 minimum, 5 stronger                                                 |
| Skill updates     | none                                                                  |
| Primary policies  | off, capped broadcast, random_k, top_k_similarity                     |
| Primary metric    | verifier reward macro mean                                            |
| Secondary metrics | verifier reward mean, success rate, env failure rate, regression rate |
| Budget rule       | compare policies under the same effective context/token budget        |
| Reporting rule    | report every attempted policy/seed/task/iteration cell                |

## Recommended Grid Sizes

| Tier                          |                                         Calculation | TASK RUNS | What it can support                                                                                                 |
| ----------------------------- | --------------------------------------------------: | --------: | ------------------------------------------------------------------------------------------------------------------- |
| Plumbing only                 | `3-5 tasks x 2 iterations x 1-2 seeds x 3 policies` |     18-60 | Checks logging, graph snapshots, prompt rendering, and leakage controls. Not enough for diffusion claims.           |
| Minimal meaningful            |                                    `10 x 3 x 3 x 4` |       360 | First credible descriptive comparison across policies. Can catch large effects and obvious regressions.             |
| Preferred                     |                                    `10 x 5 x 3 x 4` |       600 | Better view of iteration trends, transfer behavior, and regression rate.                                            |
| Stronger                      |                                    `10 x 5 x 5 x 4` |      1000 | More stable policy ranking across seed variance. Better for a serious result report.                                |
| Full six-row diffusion matrix |                                    `10 x 5 x 5 x 6` |      1500 | Can include IC and hybrid rows, but those should be treated as secondary/exploratory unless predeclared as primary. |

## Recommended Minimum

Use **360 TASK RUNS** as the minimum meaningful grid:

```text
10 tasks x 3 iterations x 3 seeds x 4 policies = 360 TASK RUNS
```

This gives:

```text
10 tasks x 3 seeds x 4 policies = 120 task-seed-policy cells
```

The iteration-level runs are useful, but they are not fully independent because
later iterations can depend on earlier context and artifacts. Treat
task-seed-policy cells as the more conservative comparison unit.

## What 360 TASK RUNS Can Do

A 360-run grid can support a narrow descriptive claim such as:

```text
In a predeclared 10-task family context-only diffusion grid, top_k_similarity
outperformed random_k and capped_broadcast on verifier reward macro mean under
the same skill-update and budget constraints.
```

It can also show:

- whether selective context has a consistent direction of effect
- whether random diffusion is a weak or harmful control
- whether capped broadcast gains are mostly budget exposure rather than selectivity
- whether environment failures differ by policy
- whether negative transfer is visible and frequent enough to matter

It should not be used to claim:

- diffusion generally helps across benchmarks
- graph-aware diffusion is statistically proven
- IC or hybrid diffusion is better, unless those rows were included and predeclared
- policy differences are significant after broad search, unless correction or full disclosure is used

## What 600-1000 TASK RUNS Add

A 600-run grid adds a better view of learning and transfer over time:

```text
10 tasks x 5 iterations x 3 seeds x 4 policies = 600 TASK RUNS
```

A 1000-run grid adds better seed stability:

```text
10 tasks x 5 iterations x 5 seeds x 4 policies = 1000 TASK RUNS
```

These grids are better suited for a result report because they reduce the chance
that the policy ranking is driven by one seed, one task, or one early iteration.

## Multiple-Comparisons Handling Without Infra Changes

Before reading outcomes, write down:

| Field              | Required value                                                |
| ------------------ | ------------------------------------------------------------- |
| Task set           | exact task set name                                           |
| Seeds              | exact seed list                                               |
| Iterations         | exact count                                                   |
| Policies           | exact policy list                                             |
| Primary comparison | one declared comparison                                       |
| Primary metric     | one declared metric                                           |
| Secondary metrics  | listed separately                                             |
| Exclusions         | allowed exclusion rules, such as environment failure handling |
| Report path        | where all results will be summarized                          |

Recommended primary comparison:

```text
top_k_similarity vs random_k
```

Recommended secondary comparisons:

```text
top_k_similarity vs capped_broadcast
top_k_similarity vs off
capped_broadcast vs off
random_k vs off
```

If additional task sets, graph settings, top-k values, thresholds, or policies are
tried after seeing results, label them exploratory and report them separately.

## Claim Levels

| Evidence        | Allowed claim                                                                          |
| --------------- | -------------------------------------------------------------------------------------- |
| 18-60 TASK RUNS | The plumbing works or fails.                                                           |
| 240 TASK RUNS   | Exploratory signal only.                                                               |
| 360 TASK RUNS   | Minimal descriptive evidence for a narrow predeclared comparison.                      |
| 600 TASK RUNS   | Meaningful context-only diffusion result report.                                       |
| 1000 TASK RUNS  | Stronger descriptive evidence with better seed robustness.                             |
| 1500+ TASK RUNS | Broader policy comparison, but multiple-comparisons discipline becomes more important. |
