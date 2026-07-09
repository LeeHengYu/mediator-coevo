# update_0001: empty-selection fallback for langchain_graph diffusion

## Source run

- Training run: `data/experiments/HL2/train_epoch_1`
- Train command: `uv run medcoevo run --family Weighted-Risk-Assessment --family HWPX-Document-Automation --seed 42 --split train --condition learned_mediator --skill-updates none --diffusion-enabled --diffusion-policy langchain_graph --run-id hl-train-langchain-graph-batch1`
- Train result: 8 scored runs, 0 environment failures, mean verifier reward 0.625, median verifier reward 1.000.

## Observed failure mode

The diffusion audit showed useful prior artifacts in the graph, but the diffusion agent could still return an empty artifact selection. In iteration 7, the HWPX training-feedback task had same-task and incoming graph-prior history available, yet `selected_artifacts` was empty and no context rendered. The task succeeded anyway, but the policy did not use available experience when the agent omitted selections.

## Candidate change

Overlay source:

- `overlay/src/mediated_coevo/diffusion/langchain_graph.py`

Behavior:

- Preserve explicit LangChain artifact selections exactly as before.
- If the diffusion agent returns no valid selections, select a deterministic fallback from same-task history, same-node history, or incoming graph-prior artifacts.
- Prefer higher-reward artifacts and mediator summaries/debug hints.
- Route successful prior artifacts to `reuse_success`; route low-reward prior artifacts to `avoid_recheck`.
- Mark fallback subscriptions with `metadata.fallback = "empty_agent_selection"`.

Overlay tests:

- `overlay/tests/test_langchain_graph_policy.py`
- Added coverage for empty agent selection falling back to a same-task artifact.

## Carried state

- Copied `diffusion/diffused_records.jsonl` for audit.
- Copied graph snapshots under `state/diffusion/graph_snapshots/`.
- Did not copy `diffusion/artifacts/`.
- Did not edit Planner, Executor, Mediator skills, benchmarks, jobs, or base tasks.

## Validation before CLI validation

- `PYTHONPATH=/private/tmp/hl-overlay-validate-20260708-batch1/src .venv/bin/ruff check /private/tmp/hl-overlay-validate-20260708-batch1/src /private/tmp/hl-overlay-validate-20260708-batch1/tests`
- `PYTHONPATH=/private/tmp/hl-overlay-validate-20260708-batch1/src .venv/bin/mypy /private/tmp/hl-overlay-validate-20260708-batch1/src /private/tmp/hl-overlay-validate-20260708-batch1/tests`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/hl-overlay-validate-20260708-batch1/src /Users/hylee_mac/Documents/Project/mediator-coevo/.venv/bin/pytest -q /private/tmp/hl-overlay-validate-20260708-batch1/tests`
- `uv run ruff check .`
- `uv run mypy src tests`
- `uv run pytest -q`

All commands passed before validation.

## Intended validation command

`uv run medcoevo run --harness-dir data/experiments/HL2/train_epoch_1/harnesses/update_0001 --family Weighted-Risk-Assessment --family HWPX-Document-Automation --seed 42 --split validation --condition learned_mediator --skill-updates none --diffusion-enabled --diffusion-policy langchain_graph --run-id hl-val-langchain-graph-batch1`
