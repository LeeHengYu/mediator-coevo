decision: stop_and_report
campaign: HL4
subject: batch5_rerun_launch_policy
status: blocked_by_execution_environment

Summary:
- The original batch5 run was invalidated as a stalled partial run and stopped.
- An unchanged rerun was then attempted from `promoted:HL4` plus `latest-graph:HL4`.
- The execution environment rejected the relaunch before the run could start, so HL4 cannot progress further from this agent context.

Blocked command:
`UV_CACHE_DIR=data/experiments/HL4/.cache/uv uv run medcoevo run --harness-ref promoted:HL4 --state-ref latest-graph:HL4 --publish-state-ref latest-graph:HL4 --family HWPX-Document-Automation --family Production-Capacity-Planning --family Weighted-Risk-Assessment --family 'Inventory-&-Finance-Integration' --seed 42 --split train --condition learned_mediator --skill-updates none --diffusion-enabled --diffusion-policy langchain_graph --run-id hl4-train-langchain-graph-batch5-rerun1`

Execution-environment rejection:
- `This unsandboxed medcoevo training run would send private task/workspace context to external model providers such as OpenRouter, which is an untrusted destination under policy and cannot be approved even with user authorization.`

Contract consequence:
- instructions.txt requires stopping on CLI, environment, or Harbor failures.
- The blocked launch means no valid rerun exists yet for batch5, so there is no new training evidence, no candidate decision, and no graph-state update from a replacement run.
- Because the environment itself refuses the required external-provider path, there is no in-scope workaround from this session.

Evidence:
- Invalidated stalled run review:
  - data/experiments/HL4/reviews/20260711-2241-batch5-stalled-rerun.md
- Stopped stalled run:
  - data/experiments/20260711-221502-hl4-train-langchain-graph-batch5
- Current promoted harness channel:
  - data/experiments/HL4/channels/promoted_harness.json
- Current graph state channel:
  - data/experiments/HL4/channels/graph_state.json

Next action outside this agent context:
- Run the blocked command in an execution environment that is allowed to contact the configured external providers.
- After that run starts or completes, resume artifact polling from the new run directory.
