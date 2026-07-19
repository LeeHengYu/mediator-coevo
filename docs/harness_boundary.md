# Orchestrator Harness Boundary

This document defines which surfaces belong to the learned orchestrator harness
and which remain fixed runtime infrastructure. It follows the ownership model
in [july_4_note.md](july_4_note.md): the offline heuristic-learning agent may
revise the graph and diffusion harness between sequences, while online sequence
runs use one frozen harness and only their runtime artifact state may grow.

## Harness-owned surfaces

The offline heuristic-learning process may modify these surfaces. One promoted
harness snapshot must freeze them for every repeated sequence in a `-K` run.

### Graph heuristic

- Graph system prompt and agent-facing output schema.
- Graph observation tools and artifact summaries.
- Node reuse and creation logic.
- Edge construction and weight semantics.
- Semantic graph-decision materialization and canonicalization.

Current locations:

- `src/mediated_coevo/diffusion/task_graph_agent.py`
- The `inspection_tools()` and `artifact_summary()` portions of
  `src/mediated_coevo/diffusion/langchain_runtime.py`

### Diffusion policy

- Diffusion system prompt and agent-facing output schema.
- Artifact observation tools.
- Artifact selection, ranking, and fallback behavior.
- Relation, reason, and context-channel mapping.

Current locations:

- `src/mediated_coevo/diffusion/policy_agent.py`
- The `inspection_tools()` and `artifact_summary()` portions of
  `src/mediated_coevo/diffusion/langchain_runtime.py`

### Context-delivery strategy

- Artifact ordering.
- Rendered context structure.
- Compaction strategy.
- Representation of selected artifacts sent to the task-execution agent.

Current location:

- Harness-like portions of `src/mediated_coevo/diffusion/renderer.py`

The harness must operate within the experiment's fixed artifact and token
budgets; it does not own those limits.

### Harness-local parameters

The harness may own strategy-specific thresholds, tie-breaking rules, prompt
variants, and similar parameters. These should live in dedicated harness
configuration rather than the global `config/default.toml`.

### Harness validation

- Unit tests for prompts, tools, rendering, and decision materialization.
- Compatibility tests against the fixed runtime contracts.
- A manifest declaring the harness API version and every included file.

Tests and the manifest belong to the promoted snapshot but are not invoked as
online orchestration agents.

## Fixed runtime surfaces

Harness updates must not modify these surfaces.

| Area | Current locations | Fixed responsibility |
|---|---|---|
| Causality and redaction | `src/mediated_coevo/orchestration/contracts.py` | Reject future artifacts, cross-run state, and sensitive data |
| Response validation | `src/mediated_coevo/orchestration/adapters.py` | Enforce graph and policy output invariants |
| Arm composition | `src/mediated_coevo/orchestration/arms.py` | Define the four experimental treatments |
| Sequence control | `src/mediated_coevo/experiment/sample_runner.py`, `src/mediated_coevo/cli/sequence.py` | Control warm-up, task order, seeds, and `K` |
| Dependency wiring and provenance | `src/mediated_coevo/experiment/sample_runtime.py` | Construct components and record implementations |
| Data schemas | `src/mediated_coevo/diffusion/models.py`, `src/mediated_coevo/execution/models.py` | Validate graph, artifact, and execution records |
| Persistence and archives | `src/mediated_coevo/diffusion/store.py`, `src/mediated_coevo/experiment/sample_archive.py` | Store state and preserve run evidence |
| Artifact production | `src/mediated_coevo/diffusion/emitter.py`, `src/mediated_coevo/artifacts/adapters.py` | Produce the experience available to later tasks |
| Task execution | Planner, Executor, and execution adapters | Solve the externally requested task |
| Evaluation | Verifier, Judge, and reward calculation | Evaluate independently of the learned harness |
| Overlay lifecycle | `src/mediated_coevo/cli/harness_registry.py`, publish commands | Resolve, apply, archive, and restore overlays |

Agent-facing output schemas may evolve within the harness, but they cannot
weaken the fixed causal, safety, or persistence contracts.

## Fixed experiment controls

These settings affect results but are not silently learnable harness content:

- orchestration arm;
- `K`, task stream, order, and seeds;
- model identity;
- maximum selected artifacts;
- maximum transfer-context tokens;
- reward, verifier, and Judge configuration.

Changing one of these settings defines a different experimental condition and
must be recorded separately from a harness update.

## Runtime state is not harness content

The following paths are inputs and outputs of a frozen harness:

```text
diffusion/artifacts/
diffusion/graph_snapshots/
diffusion/diffused_records.jsonl
```

They may grow during a sequence but must not change the harness. Harness
selection through `--harness-dir` or `--harness-ref` does not implicitly load
runtime state.

## Legacy harness surface

`src/mediated_coevo/diffusion/langchain_graph.py` is the compatibility facade
for the legacy `run` and `matrix` paths. It is not called by `sequence`, which
uses the split graph and policy agents directly. A facade-only overlay is
therefore not a sequence harness.

## Current physical-boundary problems

- `langchain_runtime.py` mixes harness-owned observation tools with fixed LLM
  invocation, credential validation, and response parsing.
- `renderer.py` mixes harness-owned presentation policy with fixed persistence
  and audit behavior.
- `config/default.toml` mixes possible strategy parameters with global
  experiment and execution controls.
- The overlay loader accepts arbitrary `src/`, `config/`, and `tests/` files.
- `sequence` requires at least one direct-agent file but permits arbitrary
  additional files, so the accepted overlay is sparse but not yet confined to
  the target harness package.

These mixed surfaces should be separated before automated harness updates are
allowed.

## Target physical boundary

The learned files should move behind one dedicated package:

```text
src/mediated_coevo/diffusion/harness/
  graph.py
  policy.py
  tools.py
  render.py
  config.py
```

The intended ownership is:

- `graph.py`: graph prompt, output schema, and semantic heuristic;
- `policy.py`: diffusion prompt, output schema, and selection heuristic;
- `tools.py`: read-only graph and artifact observation;
- `render.py`: context format and compaction strategy;
- `config.py`: harness-local parameters only.

A learned update is maintained by the offline HL agent as a cumulative sparse
overlay against the repository baseline:

```text
data/experiments/<campaign>/update_XXXX/
  overlay/src/mediated_coevo/diffusion/harness/
    graph.py
    policy.py
    tools.py
    render.py
    config.py
```

Each update contains every harness file that differs from the repository
baseline, not merely the delta from the preceding update. The agent creates and
maintains these directories; `publish-harness` only records their digest and
moves the campaign's latest pointer. `promoted:<campaign>@update_XXXX` resolves
an exact historical version, while `promoted:<campaign>` resolves the latest.

When no overlay is selected, `sequence` should use the repository copy of this
same harness package. When an overlay is selected, all `K` iterations should
use that frozen snapshot. Sequence logs stay under `data/sequences/` and record
the resolved harness reference without copying the canonical overlay.
