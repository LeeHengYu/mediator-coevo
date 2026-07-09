# update_0003: deterministic fallback on graph/diffusion agent provider errors

## Source evidence

- Last promoted harness: `data/experiments/HL2/train_epoch_1/harnesses/update_0001`
- Completed training run: `data/experiments/HL2/train_epoch_2`
- Rejected candidate: `data/experiments/HL2/train_epoch_2/harnesses/update_0002`
- Failed continuation attempts:
  - `data/experiments/20260709-044710-hl-train-langchain-graph-batch3`
  - `data/experiments/20260709-044755-hl-train-langchain-graph-batch3-retry1`

Both batch 3 attempts stopped before any scored task because the LangChain diffusion agent raised an OpenRouter provider error. The second attempt exposed an OpenRouter response-validation failure around an error body with code 400. A first validation attempt for this candidate, `data/experiments/20260709-045232-hl-val-langchain-graph-batch3-unblock`, was stopped before planning output because the graph/diffusion preparation call hung. These are not completed training or validation epochs.

## Candidate change

Overlay source:

- `overlay/src/mediated_coevo/diffusion/langchain_graph.py`

Behavior:

- Keep `update_0001` empty-selection fallback unchanged.
- If the graph agent call fails, create a deterministic graph decision for the current task node and continue.
- If the diffusion agent call fails, use an empty diffusion decision so existing deterministic fallback selection can run.
- Bound graph and diffusion agent calls with a timeout so provider hangs also degrade to deterministic fallback.
- Preserve the fixed validation graph path; the same diffusion-agent fallback applies when `select_with_fixed_graph` is used.

Overlay tests:

- `overlay/tests/test_langchain_graph_policy.py`
- Added coverage for diffusion-agent provider failure falling back to same-task artifacts instead of raising.
- Added coverage for diffusion-agent timeout falling back to same-task artifacts.

## Carried state

- Copied `diffusion/diffused_records.jsonl` for audit.
- Copied graph snapshots under `state/diffusion/graph_snapshots/`.
- Did not copy `diffusion/artifacts/`.
- Did not edit Planner, Executor, Mediator skills, benchmarks, jobs, or base tasks.

## Validation before CLI validation

- `PYTHONPATH=/private/tmp/hl-overlay-validate-20260709-batch2-u3/src /Users/hylee_mac/Documents/Project/mediator-coevo/.venv/bin/ruff check /private/tmp/hl-overlay-validate-20260709-batch2-u3/src /private/tmp/hl-overlay-validate-20260709-batch2-u3/tests`
- `PYTHONPATH=/private/tmp/hl-overlay-validate-20260709-batch2-u3/src /Users/hylee_mac/Documents/Project/mediator-coevo/.venv/bin/mypy /private/tmp/hl-overlay-validate-20260709-batch2-u3/src /private/tmp/hl-overlay-validate-20260709-batch2-u3/tests`
- `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/private/tmp/hl-overlay-validate-20260709-batch2-u3/src /Users/hylee_mac/Documents/Project/mediator-coevo/.venv/bin/pytest -q /private/tmp/hl-overlay-validate-20260709-batch2-u3/tests`
- `uv run ruff check .`
- `uv run mypy src tests`
- `uv run pytest -q`

All commands passed before validation.

## Intended validation command

`uv run medcoevo run --harness-dir data/experiments/HL2/train_epoch_2/harnesses/update_0003 --family Weighted-Risk-Assessment --family HWPX-Document-Automation --seed 42 --split validation --condition learned_mediator --skill-updates none --diffusion-enabled --diffusion-policy langchain_graph --run-id hl-val-langchain-graph-batch3-unblock`

## Validation status

- Decision: not promoted; validation is blocked by external run failures.
- Stopped validation run: `data/experiments/20260709-045232-hl-val-langchain-graph-batch3-unblock`
  - Status: interrupted before planner output after graph/diffusion preparation hung.
- Stopped validation retry: `data/experiments/20260709-053049-hl-val-langchain-graph-batch3-unblock-retry1`
  - Status: stopped after task 1 recorded `verifier_status=env_failure`, `error_kind=harbor_trial_exception`.
  - Harbor evidence: Claude Code exited nonzero after a long agent run for WRA port-throughput. The verifier artifact recorded reward 0.0, but the orchestrator classified the trial as an environment failure, so the run cannot be counted by the experiment rules.

No completed validation evidence exists for `update_0003`; continue only after the Harbor/agent failure condition is cleared or after the environment-failure classification path is intentionally changed and validated separately.
