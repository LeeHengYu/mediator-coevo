# Explicit task-execution boundary

This package defines exactly what the task-execution agent receives and
returns. Execution may use the frozen task occurrence and the caller-supplied
`ContextPack`; it may not discover graph state, inspect the transfer bank,
select cross-task evidence, or append artifacts itself.

## Models

`TaskProfile` is a schema-versioned, frozen snapshot of a benchmark task. Its
configuration is normalized to deterministic JSON, detached from repository
objects, and recursively immutable. Duplicate task IDs in a sequence remain
valid because each execution also has a position and `run_id`.

`ContextPack` is the complete orchestration input. It records eligible,
selected, rendered, compacted, and budget-dropped artifact IDs; rendered source
tasks; graph snapshot and policy identities; text and token budget state; and
small JSON metadata. Validation enforces the exact subset relations and rejects
contradictory text, source, graph, policy, or budget claims. The canonical
`empty_context_pack()` is used for warm-up and `execution_only` calls.

`TaskExecutionRequest` is arm-neutral during warm-up (`arm=None`) and carries a
treatment arm only for the suffix. Its `run_id` is the `warmup_run_id` during
the prefix and the `sample_id` during an arm-specific suffix.
`TaskExecutionResult` keeps the full `IterationRecord` and portable archive
references under the same task and position identity. Infrastructure trace
statuses (`env_failure`, `parse_error`, and `harbor_failed`) are exposed as
failures without misclassifying verifier score `0.0`.

The public protocol surface is deliberately small:

- `TaskProfileProvider.resolve()` freezes a benchmark task before a sequence
  starts.
- `TaskExecutionAgent.execute()` consumes one complete
  `TaskExecutionRequest` and returns a matching `TaskExecutionResult`.
- `BenchmarkTaskProfileProvider` and
  `ExplicitContextOrchestratorExecutionAgent` adapt the existing benchmark
  repository and Orchestrator to those contracts.

## Existing Orchestrator adapter

`ExplicitContextOrchestratorExecutionAgent` calls:

```python
await orchestrator.execute_task_with_context(
    task_id=request.task.task_id,
    position=request.position,
    context=request.context,
    task=request.task,
)
```

The complete pack is passed—not just its rendered text. That seam bypasses
internal prior-context discovery and diffusion emission. It executes the
frozen task occurrence with fixed prompt-injected skills, then stamps the
policy, graph snapshot, counts, sources, token budget, compaction, and dropping
fields into the returned `IterationRecord`.
The adapter rejects a record whose identity or observability fields do not
match the supplied request.

## Scope

The [July 4 design](../../../docs/july_4_note.md) remains the final target. The
current runtime is a library-only, single-sample foundation with no
train/validation/test split, sample CLI, batch runner, ten-sequence aggregator,
or automatic heuristic learning. The existing legacy split CLI is preserved
but is outside this API.
