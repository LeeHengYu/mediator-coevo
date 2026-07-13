# Run archives and transfer artifact banks

This package separates loss-preserving task evidence from the compact artifact
bank that later tasks may receive. A complete run archive can include Harbor
jobs, traces, verifier and judge records, reports, and logs. A
`DiffusionArtifact` is only a portable projection of that evidence; it is not a
replacement for the archive.

## Public contracts

- `ArtifactProjector.project()` converts a completed `TaskExecutionResult` and
  its frozen `TaskProfile` into zero or more compact transfer artifacts.
- `ArtifactBankUpdate` is the immutable, append-only transition
  `before + added == after` for one run and position.
- `ArtifactBankUpdater.prepare()`, `persist()`, and `rollback()` separate full
  transition validation from durable writes and cleanup.
- `DiffusionEmitterProjector` and `DiffusionArtifactBankUpdater` adapt the
  existing diffusion emitter and store to those contracts.

## Transaction boundary

Projection and persistence are deliberately separate:

```text
TaskExecutionResult
    -> ArtifactProjector.project()          # may use an emitter or compactor
    -> ArtifactBankUpdater.prepare()        # validates the whole transition
    -> ArtifactBankUpdater.persist()        # preflights every target, then writes
    -> immutable position journal
    -> commit the runner's next state
```

`prepare()` checks occurrence identity, causal ordering, source provenance,
duplicate IDs, and the exact append relation
`after == before + added` without writing. `persist()` rejects any existing
target before the first write. If a batch write fails, it cleans up that batch;
the runner also calls `rollback()` if the later journal write fails. The
runner commits its next in-memory state only after the immutable journal is
published, and archive validation rejects orphaned or mismatched durable state.

The generic identity is `run_id`: it is the arm-neutral `warmup_run_id` for a
prefix execution and the arm-specific `sample_id` for a suffix execution.
Projected artifacts retain that identity as `source_run_id`, along with the
normalized task occurrence, verifier score (including a legitimate `0.0`), and
available judge provenance.

## Causality and shared warm-up

At position `i`, the candidate bank may contain only artifacts with
`source_iteration < i`. The current task is projected and appended only after
its execution completes. Repeated task IDs are valid because position and run
identity distinguish occurrences.

A sequence's full warm-up archive is stored once. Each arm materializes only
the compact warm-up transfer artifacts into its own fresh diffusion store;
Harbor jobs and other full evidence remain referenced by the shared bundle.
Suffix artifacts then evolve independently within each arm.

## Scope

The [July 4 design](../../../docs/july_4_note.md) is the final target. The
current sample runtime has no train/validation/test split, sample CLI, batch
campaign, ten-sequence aggregator, or automated heuristic-learning loop. The legacy
split-oriented CLI and artifact import/export paths still exist, but they are
not part of this sample API.
