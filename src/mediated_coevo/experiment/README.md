# Causal single-sample runtime

This package provides a programmable foundation for executing and archiving
one frozen task sequence. A sample consists of one arm-neutral warm-up prefix
plus one arm-specific orchestrated suffix:

```text
WarmupBundle [0, W) + SampleResult [W, N) = one complete sample
```

The [July 4 design](../../../docs/july_4_note.md) is the final target. This
runtime implements the current causal data-collection layer; it is not the
heuristic-learning loop itself.

## Identities and immutable specifications

`SequenceSpec` freezes the ordered, normalized `TaskProfile` occurrences,
`warmup_count`, policy seed, optional suffix reward weights, and optional task
set identity. It is arm-neutral and contains no train/validation/test split.
The same task ID may occur at multiple positions.

`SampleSpec` embeds that complete sequence and adds one `OrchestrationArm`, a
unique `sample_id`, and the shared warm-up bundle identity when the prefix is
non-empty. The identity boundary is intentional:

- `sequence_id` identifies only the frozen task sequence.
- `warmup_run_id` identifies the one arm-neutral prefix execution.
- `sample_id` identifies one arm-specific suffix execution and is also its
  graph `run_id`.

All new sample models are frozen, reject unknown fields, and use
`schema_version=1`.

## Causal lifecycle

Every warm-up position bypasses graph construction, policy selection, and
context packing:

```text
resolve frozen task
-> execute with the canonical empty ContextPack
-> project compact artifacts
-> validate and append the bank
-> write the immutable position journal
```

Every suffix position follows the component plan for its arm:

```text
resolve frozen task
-> update graph when enabled
-> select artifacts
-> pack explicit ContextPack
-> execute the task with exactly that context
-> project compact artifacts
-> validate and append the bank
-> write the immutable position journal
```

At position `i`, the graph and policy can see only artifacts whose source
position is less than `i`. The current task's artifacts enter the bank only
after its execution, projection, persistence, and journal commit succeed.

The four suffix arms are:

| Arm | Components invoked |
|---|---|
| `execution_only` | task execution only; the bank updates but is never routed |
| `random_policy` | deterministic uniform selection and context packing; no graph |
| `no_graph` | diffusion-policy agent with `graph=None`, then context packing |
| `full_orchestration` | task-graph agent, diffusion-policy agent, and context packing |

All arms start from the same shared warm-up bank. Their suffix banks then
evolve endogenously because selected context can change execution and future
artifacts. Arm results are therefore system-level comparisons, not a claim of
selector-only comparison against a permanently frozen bank.

## Runtime API

`SampleRuntime` is intentionally one-shot. Construct a fresh Orchestrator and
workspace for exactly one call to either `prepare_warmup()` or `run()`:

```python
from mediated_coevo.experiment.sample_archive import (
    load_sample_result,
    load_warmup_bundle,
)
from mediated_coevo.experiment.sample_runtime import build_sample_runtime

warmup_runtime = build_sample_runtime(
    orchestrator=fresh_warmup_orchestrator,
    run_id=warmup_run_id,
    sequence_dir=sequence_dir,
    implementation_revision=git_revision,
    implementation_dirty=git_dirty,
)
warmup = await warmup_runtime.prepare_warmup(sequence)

sample_runtime = build_sample_runtime(
    orchestrator=fresh_arm_orchestrator,
    run_id=sample_spec.sample_id,
    sequence_dir=sequence_dir,
    implementation_revision=git_revision,
    implementation_dirty=git_dirty,
)
result = await sample_runtime.run(sample_spec, warmup=warmup)

verified_warmup = load_warmup_bundle(warmup_workspace)
verified_result = load_sample_result(sample_workspace)
```

The builder wires the split `LangChainTaskGraphAgent` and
`LangChainDiffusionPolicyAgent` through their direct adapters; it does not use
the legacy `LangChainGraphPolicy` facade. Online executor, planner, and
mediator skill updates must be disabled for this explicit sample path.

Both runtime operations reject a reused runtime or a workspace with existing
journals, terminal records, jobs, metrics, history, reports, traces, or
diffusion state before calling an agent. There is no reset or resume API.

## Shared warm-up and durable archives

The warm-up executes once per sequence and remains arm-neutral. Its
`WarmupBundle` records the prefix transitions, final initial artifact bank,
archive references, and runtime provenance. `bundle_id` is the SHA-256 of
canonical semantic content, excluding archive paths and runtime provenance.
Warm-up records are never rewritten with a treatment arm.

Each arm receives a `WarmupReference` and materializes only the compact
transfer artifacts into its fresh diffusion store. Full Harbor jobs, traces,
reports, and judge evidence remain in the shared warm-up archive instead of
being copied into every arm workspace.

```text
<sequence_dir>/
  sequence_spec.json
  warmup/<warmup_run_id>/
    warmup_bundle.json
    journal/position-0000.json
    artifacts/
    diffusion/
    archive_manifest.json
    warmup_failure.json          # failure only
  samples/<sample_id>/
    sample_spec.json
    warmup_ref.json
    journal/position-000W.json
    artifacts/
    diffusion/
    archive_manifest.json
    sample_result.json           # success only
    sample_failure.json          # failure only
```

`ArchiveManifest` records portable workspace-relative paths, kind, SHA-256,
and byte size. Evidence that cannot be localized is retained as explicitly
external provenance rather than treated as a portable local path.
`RuntimeProvenance` records the implementation revision and dirty flag, config
and implementation hashes, model mapping, executor identity, package and
Python versions, and start/finish timestamps without persisting credentials.

## Rewards, journals, and failures

`SequenceRewards` covers only `[W, N)` and preserves every position. A valid
score of `0.0` remains a score; a missing score remains `None`. If any suffix
reward is missing, the primary sum, mean, weighted sum, and weighted mean are
all `None`. `valid_for_reporting` is true only when every suffix task completed
and every reward is present.

A position journal is published only after execution, artifact projection,
whole-transition validation, persistence, and in-memory state advance are all
successful. Journals are immutable and cannot be overwritten. Successful
terminal results and failure records are mutually exclusive.

Failures are classified as `resolve`, `graph`, `policy`, `pack`, `execute`,
`project`, `persist`, or `finalize`. A `SampleRunError` carries that stage and
the last committed progress; the runtime writes `warmup_failure.json` or
`sample_failure.json`, stops the sequence, and re-raises. Infrastructure
statuses such as `env_failure`, `parse_error`, and `harbor_failed` are failures,
while verifier score `0.0` is not.

## Scope and compatibility

This is a library-only, single-sample runtime. It deliberately adds no sample
CLI, batch campaign runner, ten-sequence aggregator, train/validation/test
protocol, or automated heuristic-learning implementation. Those are later
layers built from the durable sample corpus and remain subject to human-guided
revision under the July 4 target design.

The existing `run`, `matrix`, renderer, harness-overlay, split CLI, and legacy
artifact import/export behavior remain compatibility paths. They are not part
of the new sample API.
