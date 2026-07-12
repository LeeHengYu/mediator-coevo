decision: policy_override_approved
campaign: HL4
subject: batch4_rerun_policy
status: authorized_to_resume

User authorization:
- The user explicitly approved task-local package setup for the failing task family.
- Scope interpreted for HL4 continuation: Inventory-&-Finance-Integration tasks may perform task-local package setup inside Harbor when needed by the benchmark task instructions.

Effect on the previously blocked rerun:
- This authorization removes the policy blocker recorded in:
  - data/experiments/HL4/reviews/20260711-200246-batch4-rerun-policy-blocked.md
- Batch 4 may now be rerun with the same promoted harness and latest graph state.
- If the rerun again performs task-local setup for `openpyxl` on `Inventory-&-Finance-Integration/new_task_10_maintenance_calcfields_restock`, that behavior is now accepted under this explicit override and is not by itself an invalidation reason for this task family.

Next command:
`uv run medcoevo run --harness-ref promoted:HL4 --state-ref latest-graph:HL4 --publish-state-ref latest-graph:HL4 --family HWPX-Document-Automation --family Production-Capacity-Planning --family Weighted-Risk-Assessment --family 'Inventory-&-Finance-Integration' --seed 42 --split train --condition learned_mediator --skill-updates none --diffusion-enabled --diffusion-policy langchain_graph --run-id hl4-train-langchain-graph-batch4-rerun1`
