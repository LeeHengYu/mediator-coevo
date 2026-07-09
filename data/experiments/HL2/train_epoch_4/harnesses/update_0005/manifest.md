# update_0005: timeout-only agent fallback with accepted graph state

## Source evidence

- Last promoted harness: `data/experiments/HL2/train_epoch_1/harnesses/update_0001`
- Rejected state-only candidate: `data/experiments/HL2/train_epoch_3/harnesses/update_0004`
- Completed training epoch 4: `data/experiments/HL2/train_epoch_4`
  - Completed 8 tasks with 0 environment failures.
  - Mean reward was 0.875.
- Invalid epoch 5 attempts from `update_0001`:
  - `data/experiments/20260709-121548-hl-train-langchain-graph-batch5-from-update1-restart1`
    stopped after 6/8 metrics when the run stalled before task 7 with no Harbor job.
  - `data/experiments/20260709-160451-hl-train-langchain-graph-batch5-from-update1-retry1`
    stopped after 1/8 metrics when the run stalled before task 2 with no Harbor job.

## Candidate change

This candidate isolates the timeout/error fallback behavior from rejected
`update_0003` while keeping the accepted `update_0001` graph state.

Overlay source:

- `overlay/src/mediated_coevo/diffusion/langchain_graph.py`
- `overlay/tests/test_langchain_graph_policy.py`

Behavior:

- Bound graph-agent and diffusion-agent calls with `_AGENT_TIMEOUT_SEC`.
- If the graph agent times out or raises, reuse a deterministic task node and
  continue instead of hanging the sequence.
- If the diffusion agent times out or raises, return an empty diffusion
  decision so the existing deterministic fallback selection can run.
- Preserve fixed-graph validation/test behavior by applying the same diffusion
  timeout fallback in `select_with_fixed_graph`.

## Carried state

- Copied `update_0001` state exactly.
- Did not carry state from rejected `update_0003` or rejected `update_0004`.
- Did not copy `diffusion/artifacts/`.

## Validation command

`uv run medcoevo run --harness-dir data/experiments/HL2/train_epoch_4/harnesses/update_0005 --family Weighted-Risk-Assessment --family HWPX-Document-Automation --seed 42 --split validation --condition learned_mediator --skill-updates none --diffusion-enabled --diffusion-policy langchain_graph --run-id hl-val-langchain-graph-batch5-timeout-update5`

## Validation before CLI validation

- Merged-tree focused checks:
  - `.venv/bin/ruff check /private/tmp/hl-update5-validate-20260709-001/src /private/tmp/hl-update5-validate-20260709-001/tests/test_langchain_graph_policy.py`
  - `PYTHONPATH=/private/tmp/hl-update5-validate-20260709-001/src .venv/bin/mypy /private/tmp/hl-update5-validate-20260709-001/src /private/tmp/hl-update5-validate-20260709-001/tests/test_langchain_graph_policy.py`
  - `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/hl-update5-validate-20260709-001/src .venv/bin/pytest -q /private/tmp/hl-update5-validate-20260709-001/tests/test_langchain_graph_policy.py`
- Standard repo checks after applying the candidate overlay:
  - `.venv/bin/ruff check .`
  - `.venv/bin/mypy src tests`
  - `PYTHONDONTWRITEBYTECODE=1 .venv/bin/pytest -q`

All focused and standard repo checks passed.

## Validation status

- Decision: rejected; do not promote.
- Validation run: `data/experiments/20260709-181903-hl-val-langchain-graph-batch5-timeout-update5`
- Status: stopped after task 1 recorded `verifier_status=env_failure`,
  `error_kind=harbor_trial_exception`.
- Validation retry: `data/experiments/20260709-191344-hl-val-langchain-graph-batch5-timeout-update5-retry1`
  - Status: stopped after task 2 recorded `verifier_status=env_failure`,
    `error_kind=harbor_trial_exception`.
- Validation retry 2: `data/experiments/20260709-212200-hl-val-langchain-graph-batch5-timeout-update5-retry2`
  - Status: incomplete; the process exited without `summary.json`.
  - Evidence: `metrics.jsonl` contains only 2/8 finalized records, both with
    `verifier_status=ok`. A third task has a raw Harbor trace and job result
    with reward 0.0, but it was not finalized into metrics or a summary.
  - Decision impact: invalid as validation evidence.
- Rationale: validation had an environment failure, so it cannot be counted as
  validation evidence and the candidate cannot be promoted. The later partial
  retry also cannot be counted because it did not complete all 8 tasks.
- Continuation: use the previous promoted harness
  `data/experiments/HL2/train_epoch_1/harnesses/update_0001`.
