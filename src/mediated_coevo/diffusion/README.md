# Diffusion architecture

This package owns task-graph construction and artifact-selection primitives. It
does not execute benchmark tasks or own an experiment sequence.

## Standalone agents

`task_graph_agent.py` defines `LangChainTaskGraphAgent`. Given the current task,
the previous graph snapshot, and causal artifacts, it asks the graph agent for a
decision and materializes a validated `TaskGraphSnapshot`. The standalone graph
path can register artifact-source tasks as graph nodes before validating new
edges, so prior evidence is representable even when no graph snapshot exists
yet.

`policy_agent.py` defines `LangChainDiffusionPolicyAgent`. Given a task, an
optional graph snapshot, and causal artifacts, it asks the policy agent which
artifacts to select and converts that decision into validated
`DiffusionSubscription` values. It selects context candidates only; rendering,
task execution, persistence, and experiment control remain outside the agent.

New code should import these classes directly from their submodules:

```python
from mediated_coevo.diffusion.policy_agent import LangChainDiffusionPolicyAgent
from mediated_coevo.diffusion.task_graph_agent import LangChainTaskGraphAgent
```

They are intentionally not re-exported through `diffusion.__init__`. Direct
submodule imports keep the existing package-level API unchanged and avoid
making new standalone names part of the historical facade contract.

## Shared LangChain runtime

`langchain_runtime.py` contains the mechanics shared by both standalone agents:
model-name normalization, credential validation, LangChain invocation, JSON
object parsing, message extraction, and read-only graph/artifact inspection
tools. It is an implementation helper rather than an orchestration policy.

## Legacy facade and overlays

`langchain_graph.py` remains the compatibility facade used by the existing
Orchestrator and CLI. `LangChainGraphPolicy` preserves the historical
constructor, `prepare()`, `select_with_fixed_graph()`, and the protected graph
and policy hooks. It also retains the top-level `_run_agent`, graph
materializer, and subscription materializer patch seams used by compatibility
tests; the remaining legacy helper names stay importable from the facade.

Historical heuristic-learning harnesses replace this exact file. The legacy
CLI re-executes after applying such an overlay, so the facade path and its
interfaces must remain stable even when the standalone implementation is
refined. The `sequence` CLI instead requires overlays to replace both
`task_graph_agent.py` and `policy_agent.py`; `langchain_runtime.py` remains
optional shared infrastructure.

## Legacy and standalone defaults

The legacy facade preserves the historical combined behavior: graph update
followed by artifact selection, including its deterministic fallback when an
agent returns an empty selection. It also preserves the historical graph
materialization rules.

The standalone policy defaults to no fallback, so an empty agent selection
remains empty unless a caller chooses a named fallback strategy explicitly. The
standalone graph agent may materialize artifact-source nodes as described above.
These differences are deliberate: the facade reproduces existing treatments,
while standalone agents expose explicit behavior for new integrations.

## Scope boundary

This package supplies independently callable graph and policy components; it
still does not own task execution, artifact-bank mutation, or experiment
sequencing. The sample integration composes these agents through the
[orchestration contracts](../orchestration/README.md) and the
[single-sample runtime](../experiment/README.md), without routing new code
through `LangChainGraphPolicy`.

The [July 4 design](../../../docs/july_4_note.md) remains the final target. The
sample runtime has no train/validation/test split or automated
heuristic-learning loop. The `sequence` CLI adds repeated deployment episodes
and a process-scoped direct-agent overlay without routing through the legacy
facade.
