# update_0002: same-task/family replacement for full cross-family selections

## Source run

- Promoted seed harness: `data/experiments/HL2/train_epoch_1/harnesses/update_0001`
- Discarded retry: `data/experiments/20260708-225729-hl-train-langchain-graph-batch2` stopped after a Harbor setup `env_failure` before executor tokens; it is not counted as a training epoch.
- Training run: `data/experiments/HL2/train_epoch_2`
- Train command: `uv run medcoevo run --harness-dir data/experiments/HL2/train_epoch_1/harnesses/update_0001 --family Weighted-Risk-Assessment --family HWPX-Document-Automation --seed 42 --split train --condition learned_mediator --skill-updates none --diffusion-enabled --diffusion-policy langchain_graph --run-id hl-train-langchain-graph-batch2-retry1`
- Train result: 8 scored runs, 0 environment failures, mean verifier reward 0.625, median verifier reward 1.000.

## Observed failure mode

The `update_0001` empty-selection fallback was not triggered in batch 2; the diffusion agent selected artifacts for every post-seed target. The remaining issue is selection quality when the artifact budget is already full:

- Validation batch 1 final WRA cloud reliability failure selected one same-task WRA debug hint plus one HWPX mediator summary, while same-task WRA success summaries were available.
- Training batch 2 API SLA failure selected one WRA failed artifact and two HWPX artifacts, including a failed HWPX artifact, while same-family WRA evidence existed.

This suggests a narrow postprocessing rule: keep explicit selections, but if a full selection spends budget on cross-family artifacts while same-task or same-family candidates are available, replace the weakest cross-family artifact.

## Candidate change

Overlay source:

- `overlay/src/mediated_coevo/diffusion/langchain_graph.py`

Behavior:

- Preserve explicit LangChain selections when they leave budget unused.
- Preserve explicit selections when they already include same-task or same-family evidence without crowding it out.
- When the selection budget is full, replace the weakest cross-family selected artifact with the strongest unselected same-task, same-node, or same-family artifact.
- Keep `update_0001` empty-selection fallback unchanged.
- Mark deterministic replacements with `metadata.fallback = "selection_rebalance"`.

Overlay tests:

- `overlay/tests/test_langchain_graph_policy.py`
- Added coverage for replacing a full-budget cross-family selection with same-task evidence.

## Carried state

- Copied `diffusion/diffused_records.jsonl` for audit.
- Copied graph snapshots under `state/diffusion/graph_snapshots/`.
- Did not copy `diffusion/artifacts/`.
- Did not edit Planner, Executor, Mediator skills, benchmarks, jobs, or base tasks.

## Validation before CLI validation

- `PYTHONPATH=/private/tmp/hl-overlay-validate-20260708-batch2-u2/src /Users/hylee_mac/Documents/Project/mediator-coevo/.venv/bin/ruff check /private/tmp/hl-overlay-validate-20260708-batch2-u2/src /private/tmp/hl-overlay-validate-20260708-batch2-u2/tests`
- `PYTHONPATH=/private/tmp/hl-overlay-validate-20260708-batch2-u2/src /Users/hylee_mac/Documents/Project/mediator-coevo/.venv/bin/mypy /private/tmp/hl-overlay-validate-20260708-batch2-u2/src /private/tmp/hl-overlay-validate-20260708-batch2-u2/tests`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/hl-overlay-validate-20260708-batch2-u2/src /Users/hylee_mac/Documents/Project/mediator-coevo/.venv/bin/pytest -q /private/tmp/hl-overlay-validate-20260708-batch2-u2/tests`
- `uv run ruff check .`
- `uv run mypy src tests`
- `uv run pytest -q`

All commands passed before validation.

## Intended validation command

`uv run medcoevo run --harness-dir data/experiments/HL2/train_epoch_2/harnesses/update_0002 --family Weighted-Risk-Assessment --family HWPX-Document-Automation --seed 42 --split validation --condition learned_mediator --skill-updates none --diffusion-enabled --diffusion-policy langchain_graph --run-id hl-val-langchain-graph-batch2`
