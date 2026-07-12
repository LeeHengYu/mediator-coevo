decision: stop_and_ask
campaign: HL4
subject: batch4_rerun_policy
status: blocked_pending_user_direction

Summary:
- Batch 4 remains invalidated and must not be used as training evidence.
- An unchanged automatic rerun is not justified under instructions.txt because the failure is not clearly transient.

Why this is a policy blocker:
- The invalidating task `Inventory-&-Finance-Integration/new_task_10_maintenance_calcfields_restock` explicitly requires `openpyxl`.
- The benchmark image for that task installs Node `xlsx` and Python test packages, but does not preinstall `openpyxl`.
- In the invalid batch 4 run, the executor hit `ModuleNotFoundError: openpyxl`, failed a system pip install, then created `/root/.venv` and installed `openpyxl` inside the task container.
- The same task was previously completed only after the same kind of task-local venv setup, so the behavior is recurrent rather than a one-off transient.

Contract consequence:
- instructions.txt allows one unchanged rerun only when the failure is clearly transient and needs no source/config edit, dependency setup, or broader permission.
- That condition is not met here.
- Therefore the next step requires explicit user direction rather than an autonomous rerun.

Evidence:
- Invalidated run review:
  - data/experiments/HL4/reviews/20260711-1955-batch4-invalidated-env-fix.md
- Invalid task trace:
  - data/experiments/HL4/train_epoch_4_invalidated/jobs/2026-07-11__19-51-10/run-5fd4ba98__exGAQkP/agent/claude-code.txt
- Invalid task job result:
  - data/experiments/HL4/train_epoch_4_invalidated/jobs/2026-07-11__19-51-10/result.json
- Task image definition lacking `openpyxl`:
  - data/experiments/HL4/train_epoch_4_invalidated/benchmarks/Inventory-&-Finance-Integration_new_task_10_maintenance_calcfields_restock/run-5fd4ba98/environment/Dockerfile
- Prior successful occurrence of the same task with the same repair pattern:
  - data/experiments/HL4/train_epoch_2/jobs/2026-07-11__15-49-15/run-243916b1__8WNGb4F/agent/claude-code.txt

User decision needed:
- Approve an unchanged rerun anyway, accepting that it may invalidate again and will be stopped again if it performs environment repair.
- Or explicitly relax the HL contract for this task family to allow task-local package setup inside Harbor.
- Or defer batch 4 rerun and reconsider the workflow boundary.

Exact unchanged rerun command if explicitly approved:
`uv run medcoevo run --harness-ref promoted:HL4 --state-ref latest-graph:HL4 --publish-state-ref latest-graph:HL4 --family HWPX-Document-Automation --family Production-Capacity-Planning --family Weighted-Risk-Assessment --family 'Inventory-&-Finance-Integration' --seed 42 --split train --condition learned_mediator --skill-updates none --diffusion-enabled --diffusion-policy langchain_graph --run-id hl4-train-langchain-graph-batch4-rerun1`
