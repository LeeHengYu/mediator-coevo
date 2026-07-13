# Graph and diffusion-policy orchestration

This package composes three independent responsibilities: graph update,
artifact selection, and context packing. None of them executes a task or
mutates the artifact bank.

## Public contracts

`GraphAgentRequest`/`GraphAgentResponse`,
`PolicyAgentRequest`/`PolicyAgentResponse`, and the `TaskGraphAgent`,
`DiffusionPolicyAgent`, and `ContextPacker` protocols make each orchestration
step independently injectable and auditable. Requests carry the frozen task
occurrence, run and position identity, and the exact causal artifact tuple.
Responses retain the raw agent decision as JSON alongside validated graph or
subscription state.

`OrchestrationArm`, `ArmPlan`, and `plan_for_arm()` are the only component
composition switch used by the sample runner. The concrete adapters are
`LangChainTaskGraphAdapter`, `LangChainDiffusionPolicyAdapter`,
`RandomPolicyAgent`, and `DiffusionContextPacker`.

## Four fixed arms

`arms.py` is the single source of truth for suffix composition:

| Arm | Graph agent | Policy | Context packer |
|---|---:|---|---:|
| `execution_only` | no | none | no |
| `random_policy` | no | deterministic uniform | yes |
| `no_graph` | no | diffusion policy with `graph=None` | yes |
| `full_orchestration` | yes | diffusion policy | yes |

Warm-up bypasses all four plans and always executes with an empty context. All
arms nevertheless append task outputs to their own causal banks. They share
the same initial warm-up bank, but suffix banks evolve endogenously with each
arm's executions. Results therefore compare complete systems, not selectors on
a permanently frozen bank.

For every suffix position, all policy arms receive the same causal candidate
definition: the bank produced only by positions strictly before the current
one. Selection and packing happen before execution; artifacts from the current
task are projected and appended only after execution succeeds.

`RandomPolicyAgent` samples uniformly without replacement from the exact causal
candidate tuple and uses the learned policy's single total artifact cap. Its
seed is derived only from the declared policy seed, position, sorted candidate
IDs, cap, and a fixed namespace. It does not reuse the legacy Random-K
success/failure quotas.

## Direct split-agent adapters

`LangChainTaskGraphAdapter` and `LangChainDiffusionPolicyAdapter` use the split
agents directly, never the legacy combined facade. The graph adapter
materializes warm-up source nodes, validates the snapshot `run_id`, position,
and current occurrence assignment, and only then persists it. The policy
adapter supports `graph=None`, enforces `fallback_strategy="none"`, preserves
the raw decision, and rejects anything outside the exact causal artifact tuple.

`DiffusionContextPacker` uses the audited renderer for graph and graph-free
arms. Its result is a strict `ContextPack` containing the entire candidate set,
selection and rendering outcomes, real snapshot identity or `None`, and token
budget decisions.

The historical `LangChainGraphPolicy`, renderer call path, harness overlay, and
legacy `run`/`matrix` commands remain compatible. They are not invoked by the
new sample contracts.

## Scope

The [July 4 design](../../../docs/july_4_note.md) is the final target. This
sample runtime adds no train/validation/test split, sample CLI, batch campaign,
ten-sequence aggregator, or automated heuristic-learning implementation. The
existing legacy split CLI remains available outside the sample API.
