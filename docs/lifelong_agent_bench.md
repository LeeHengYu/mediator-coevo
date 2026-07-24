# LifelongAgentBench as a secondary HL dataset

## Decision

Adopt `os_interaction` first through the existing SkillFlow-compatible Harbor
path. Keep `db_bench` and `knowledge_graph` disabled until their released
execution environments can be reproduced without changing their scoring
semantics.

This is a secondary dataset: each sequence selects exactly one family, and the
normal sequence configuration creates a deterministic 10-task batch by default.
LifelongAgentBench tasks do not mix with SkillFlow tasks in the same sequence.

## Inspected source

- Code: `caixd-220529/LifelongAgentBench` at
  `d6f19b42eb358d9150379f0c68c2985c5a867520`.
- Data: `csyq/LifelongAgentBench` at
  `75054b60177d4dcddb93b984413ff799b0a1fdbc`.
- Local manifest and hashes:
  `benchmarks/lifelong_agent_bench/SOURCE.json`.
- Released rows: 500 DB tasks, 500 OS tasks, and 396 knowledge-graph tasks.

No code or dataset license was visible in the inspected releases on 2026-07-24.
Do not redistribute the vendored source or publish benchmark-derived task
packages until the license is clarified.

## Execution fit

| Family | Upstream execution and scoring | Current-infrastructure fit | Decision |
| --- | --- | --- | --- |
| `os_interaction` | Fresh Ubuntu container, initialization command, interactive shell actions, then an evaluation command whose exit code is the binary score | Maps directly to a Harbor environment plus hidden verifier | Implemented |
| `db_bench` | Persistent MySQL container, per-task database/table initialization, iterative SQL operations, and direct-answer or post-mutation MD5 scoring | Harbor can host it, but replacing MySQL with SQLite or a one-shot SQL verifier would change semantics | Fidelity-gated |
| `knowledge_graph` | Stateful API over a SPARQL endpoint and ontology files, with final answer-set comparison | The released archive references a local SPARQL endpoint and ontology directory that are not included | Resource-gated |

The OS adapter supports the upstream command types (`bash`, `python`, `c`, and
`cpp`), copies only initialization state into the agent environment, and keeps
the evaluation command under `tests/`. The upstream ground-truth command and
`skill_list` are deliberately omitted from agent-visible files and task
metadata.

The adapter layers one shared `lifelong-agent-bench/os-base:ubuntu24.04` image
over the existing Harbor CLI base and installs the upstream OS tools there.
That common image is built explicitly before a run. Each much smaller task
image adds only its initialization command and is built lazily if missing.
Generated tasks allow internet access when the agent or task needs it; the
integration does not impose an additional offline restriction. Future DB and
knowledge-graph adapters must retain the same policy—their current blockers are
missing faithful services/resources, not network permission.
This preserves the current executor, credential, archive, and verifier
boundaries rather than introducing a second runtime.

## HL artifact reuse

Upstream lifelong reuse keeps a bounded list of successful prior sessions and
injects their chat histories into later prompts. The local integration uses the
existing HL artifact pipeline instead:

1. Harbor remains the source of truth for the full run archive.
2. `ExecutionTrace` records reward, verifier results, stdout/stderr, token
   usage, and the same Harbor provenance paths used by SkillFlow.
3. The sample archive transfers those Harbor paths with the execution trace, so
   later agents can read the original run artifacts through the existing
   SkillFlow-compatible contract.
4. LifelongAgentBench does not parse, redact, truncate, or reformat agent logs.
5. Environment failures and missing rewards do not become reusable HL
   artifacts. Legitimate zero-reward task attempts remain usable negative
   evidence.

Log compatibility is a soft requirement: usable upstream artifacts should
follow the normal SkillFlow archive shape, while reward and verifier parsing
remain part of the shared Harbor execution contract.

## Commands

The executable family uses a repo-local JSONL mirror of the pinned Parquet
source, so normal invocation needs no optional dependency or cache override:
Like SkillFlow, its cached task index lives under `docs/`, at
`docs/lifelong_agent_bench_tasks.txt`.

```bash
uv run medcoevo sync --family os_interaction
uv run medcoevo list --family os_interaction
uv run medcoevo build-base-image
uv run medcoevo base-artifacts --family os_interaction
```

The base-image command builds both the shared SkillFlow and OS images and
performs environment preparation only. The existing
`base-artifacts` command is a separate pre-run task-execution phase that creates
portable per-task HL stores used by sequence warm-up; it does not start a
sequence. Because the OS family is large, `sequence` creates and persists only
the selected warm-up stores that are missing, then loads them normally. Existing
stores are reused. Other benchmark families still require prebuilt stores.
Run the normal sequence path separately; `config/default.toml` supplies length
10:

```bash
uv run medcoevo sequence --family os_interaction
```

Use `--length` to override the batch size. Supplying multiple `--family`
options is rejected before runtime preflight. The local benchmark root is
inferred from the selected family. When a selected task-specific image is
missing, the existing runner builds it immediately before that task executes;
it never rebuilds the shared family base implicitly.

## Acceptance gates

Before using the family for reported benchmark results:

1. Materialize actual repo-local Parquet rows and confirm there are 500 unique
   OS task IDs.
2. Build and run a small real-row Harbor batch, verifying environment
   initialization, hidden evaluation, binary rewards, and complete archives.
3. Inspect generated tasks for oracle/skill-label leakage.
4. Compare several tasks against the upstream runner to confirm equal rewards.
5. Resolve the upstream licensing ambiguity before redistribution.

DB adoption additionally requires MySQL-compatible mutation and answer
validation tests. Knowledge-graph adoption additionally requires the exact
ontology and SPARQL data release.
