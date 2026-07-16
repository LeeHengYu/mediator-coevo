# HL5 six-episode reward and harness-update analysis

## Scope and reconstruction rule

This report reads the six completed HL5 learning episodes that produced `update_0001` through `update_0006`. An episode is three valid `sample_result.json` records under one frozen harness. Each record scores seven tasks, so each episode contains 21 scored tasks.

Episodes 3 and 4 were interrupted and resumed in new sequence directories. Their three valid results are therefore reconstructed chronologically across two directories. Specs without `sample_result.json` are excluded. This produces exactly 18 valid iteration means.

## Basic statistics

| Episode | Harness used | Three iteration rewards | Successes | Episode mean | Change from prior episode | Update checkpoint |
|---|---|---:|---:|---:|---:|---|
| 1 | baseline | `.571429, .428571, .714286` | 12/21 | **.571429** | — | `update_0001` |
| 2 | `update_0001` | `.285714, .285714, .857143` | 10/21 | **.476190** | −.095238 | `update_0002` |
| 3 | `update_0002` | `.571429, .714286, .571429` | 13/21 | **.619048** | +.142857 | `update_0003` |
| 4 | `update_0003` | `.428571, .571429, .857143` | 13/21 | **.619048** | .000000 | `update_0004` |
| 5 | `update_0004` | `.714286, .857143, .571429` | 15/21 | **.714286** | +.095238 | `update_0005` |
| 6 | `update_0005` | `.571429, .142857, .428571` | 8/21 | **.380952** | −.333333 | `update_0006` |

Across all six episodes, reward is **71/126 = .563492**. Episode 5 is best at `.714286`; episode 6 is worst at `.380952`. The highest individual iteration reward is `.857143` at points 6, 12, and 14. The lowest is `.142857` at point 17.

These are descriptive comparisons across intentionally fresh random task streams. The episode-to-episode deltas are learning signals, not same-task causal estimates.

## Sequence sources for the 18 means

| Points | Episode | Source sequence and iteration |
|---:|---:|---|
| 1–3 | 1 | `sequence-20260714-221157-42`, `iter-1` through `iter-3` |
| 4–6 | 2 | `sequence-20260715-191411-2118514157`, `iter-1` through `iter-3` |
| 7 | 3 | `sequence-20260715-213005-3732207999/iter-1` |
| 8–9 | 3 | `sequence-20260715-225048-3732208000/iter-1` and `iter-2` |
| 10–11 | 4 | `sequence-20260716-104906-2014387070/iter-1` and `iter-2` |
| 12 | 4 | `sequence-20260716-131438-2014387072/iter-1` |
| 13–15 | 5 | `sequence-20260716-142426-3147520901`, `iter-1` through `iter-3` |
| 16–18 | 6 | `sequence-20260716-164851-2865147724`, `iter-1` through `iter-3` |

The promotion registry records only one `source_sequence` per update. Consequently, `update_0003` points to `sequence-20260715-225048-3732208000` and `update_0004` points to `sequence-20260716-131438-2014387072`; those fields do not by themselves preserve the earlier resumed results that complete episodes 3 and 4.

## Reward plot

![HL5 18-point reward evolution with update checkpoints](six_episode_reward_evolution.svg)

The machine-readable point-level data is in [`six_episode_reward_stats.csv`](six_episode_reward_stats.csv).

## Harness update reasoning

All six promoted updates are cumulative sparse overlays whose only published source file is `src/mediated_coevo/diffusion/policy_agent.py`.

### `update_0001` after episode 1

- Observed signal: baseline reward was 12/21, with repeated failures on `harbor_oncocooler_10v20` and `harbor_vaxcrate_6v12` plus failures across HWPX, spreadsheet, and JSON tasks.
- Change: low-reward artifacts are forcibly materialized on `AVOID_RECHECK_CHANNEL` when `artifact_reward(artifact) < 0.5`.
- Reasoning: the first intervention prevents failed experience from being presented as reusable success. It changes the semantic delivery channel without altering graph construction or task execution.
- Outcome in the next episode: reward fell to 10/21. The safeguard was directionally sensible but insufficient; merely relabeling bad artifacts did not make selection graph-aware or contract-aware.

### `update_0002` after episode 2

- Observed signal: episode 2 was the first regression, with repeated failures on datacenter capacity and `harbor_syncpack_28v56` and broad cross-family failures.
- Change: prefer successful artifacts from the strongest incoming graph priors, especially same-family or same-output-format summaries/outcomes; favor exact contract shape over broad calculation analogies; use failed artifacts only as warnings.
- Reasoning: the update targets overly broad transfer. It makes source-node strength, success, output format, and contract shape explicit selection priorities.
- Outcome in the next episode: reward recovered to 13/21, consistent with improved routing, though the random task stream prevents assigning the gain causally.

### `update_0003` after episode 3

- Observed signal: reward improved, but failures remained spread across exact-schema JSON, spreadsheets, and recurring datacenter/healthcare tasks.
- Change: treat prior literals—entity names, sheet titles, JSON keys, filenames, and summary formatting—as contamination risks; make the current task contract authoritative; select fewer artifacts or none when prior layouts conflict.
- Reasoning: same-family experience can still transfer the wrong literal schema. This update separates portable method knowledge from task-specific values and structure.
- Outcome in the next episode: reward stayed at 13/21. Literal-contamination protection avoided a regression but did not improve the aggregate.

### `update_0004` after episode 4

- Observed signal: reward plateaued, with two more `harbor_syncpack_28v56` failures and continuing JSON-analysis failures.
- Change: for JSON-plus-Markdown tasks, turn failure artifacts into checklists for exact key spelling, nesting, suffixes, ordering, rounding, and summary-line requirements.
- Reasoning: generic “respect the current contract” guidance was too abstract. The update makes the verifier-facing schema dimensions concrete.
- Outcome in the next episode: reward rose to the six-episode high of 15/21. The remaining six failures nevertheless spanned spreadsheets and several healthcare JSON tasks.

### `update_0005` after episode 5

- Observed signal: episode 5 was strongest, but failures exposed exact verifier mismatches beyond JSON keys—spreadsheet formulas/cells and formatting-sensitive outputs remained vulnerable.
- Change: prioritize same-family or same-format failure warnings that name concrete mismatches such as missing/extra keys, nesting, suffixes, cell coordinates, header locations, and unsupported formulas; include those exact terms in the selection reason.
- Reasoning: the executor is more likely to act on a precise checklist than on a generic failure warning. This broadens the verifier-surface treatment across structured deliverables.
- Outcome in the next episode: reward collapsed to 8/21. Failures covered HWPX, spreadsheets, and healthcare JSON tasks, suggesting that precise warnings were still being paired with misleading cross-family success artifacts or were not sufficient to guide execution.

### `update_0006` after episode 6

- Observed signal: episode 6 had 13 failures, including repeated failures on HWPX safety audit, vaccination JSON analysis, and campus-budget spreadsheets.
- Change: make output container and verifier surface first-class; prefer the same container type and task family; reject cross-family financial/calculation analogies that do not preserve file format, literals, cell ranges, formula dialect, required keys, or formatting. Cross-family successes become secondary portable-method evidence only.
- Reasoning: `update_0005` emphasized precise mismatch warnings but did not make container compatibility a hard enough selection criterion. `update_0006` directly addresses negative transfer across HWPX/XML, workbooks, and JSON/Markdown.
- Validation status: no post-update episode is present yet, so `update_0006` has no reward outcome in this six-episode window.

## Why only one source file changed

The single-file pattern is a deliberate hypothesis, not a registry limitation:

1. `policy_agent.py` owns the diffusion system prompt, artifact selection/materialization, relation/reason fields, fallback behavior, and context-channel mapping. Every learned intervention above targets those responsibilities.
2. The experiment treated the dominant problem as negative transfer and artifact-selection quality. Keeping graph construction, rendering, observation tools, task execution, and evaluation fixed isolates the policy intervention.
3. HL5 overlays are cumulative against the repository baseline. Each update stores the full modified `policy_agent.py`, not merely its delta from the previous update. Thus “one file” means one harness-owned source surface differs from baseline across all six versions.
4. The harness boundary permits changes to graph heuristics, observation tools, rendering, configuration, tests, and manifests. Nothing in the overlay format requires a one-file update. Those surfaces were simply not selected by the HL5 update hypothesis.
5. This minimality has a limit: selection-policy prompts cannot repair task-executor logic. Persistent task-specific failures and the episode-6 collapse are evidence that further prompt growth in this file alone may have diminishing returns.

The promotion registry records `validation_run: null` for all six versions. Therefore the updates were promoted from training-sequence evidence without a separately recorded validation gate, even though the design notes call for validation before freezing a harness.

## Current provenance warning

At publication time, `update_0006` recorded only `policy_agent.py` in `applied_files`. The overlay now also contains `src/mediated_coevo/diffusion/__pycache__/policy_agent.cpython-314.pyc`, created after the promotion record. The registry enumerates all overlay files when recomputing the digest, so `promoted:HL5` currently fails resolution with:

```text
BadParameter: promoted harness contents changed after publication: .../data/experiments/HL5/update_0006
```

The cache file is not a learned source change, but it has mutated the supposedly immutable published overlay and should be cleaned or excluded by registry policy before the next run.
