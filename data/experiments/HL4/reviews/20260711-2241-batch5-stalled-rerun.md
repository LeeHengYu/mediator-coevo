decision: invalidated
campaign: HL4
run_path: data/experiments/20260711-221502-hl4-train-langchain-graph-batch5
status: stalled_before_iteration_3

Reason:
- This training run is not valid harness evidence under instructions.txt because it stopped making observable progress after committing iteration 2.
- The run stayed alive for more than 25 minutes with no new Harbor job materialized for the next task, no new metrics/history lines, and no summary.json.
- The live Python worker was near-idle and held one long-lived outbound TLS connection, which is consistent with an external-provider wait or hang rather than normal batch execution.

Evidence:
- Last committed batch artifacts:
  - metrics: data/experiments/20260711-221502-hl4-train-langchain-graph-batch5/metrics.jsonl
  - history: data/experiments/20260711-221502-hl4-train-langchain-graph-batch5/history/history.jsonl
  - committed iterations: 3
  - latest committed iteration: 2
  - latest task: Weighted-Risk-Assessment/campus-budget-at-risk-calc
  - reward: 1.0
  - verifier_status: ok
- Missing completion evidence:
  - summary: data/experiments/20260711-221502-hl4-train-langchain-graph-batch5/summary.json (absent)
  - Harbor jobs present: only
    - data/experiments/20260711-221502-hl4-train-langchain-graph-batch5/jobs/2026-07-11__22-16-14
    - data/experiments/20260711-221502-hl4-train-langchain-graph-batch5/jobs/2026-07-11__22-21-24
    - data/experiments/20260711-221502-hl4-train-langchain-graph-batch5/jobs/2026-07-11__22-27-25
- Live process state at invalidation:
  - wrapper PID: 96983
  - Python PID: 96984
  - process snapshot: `STAT=Ss/S`, `%CPU=0.0/0.8`, elapsed about 25 minutes
  - socket snapshot: one established TLS connection from PID 96984 to `104.18.3.115:443`

Do not use for:
- candidate design
- promotion/rejection of a harness candidate
- counting toward the 5 completed HL4 training epochs

Next action:
- Stop the stalled batch5 process pair.
- Perform one unchanged rerun from the promoted harness and latest graph channel.

Proposed next command:
`UV_CACHE_DIR=data/experiments/HL4/.cache/uv uv run medcoevo run --harness-ref promoted:HL4 --state-ref latest-graph:HL4 --publish-state-ref latest-graph:HL4 --family HWPX-Document-Automation --family Production-Capacity-Planning --family Weighted-Risk-Assessment --family 'Inventory-&-Finance-Integration' --seed 42 --split train --condition learned_mediator --skill-updates none --diffusion-enabled --diffusion-policy langchain_graph --run-id hl4-train-langchain-graph-batch5-rerun1`
