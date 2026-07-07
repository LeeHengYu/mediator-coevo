# Mediated Co-Evolution

Mediated Co-Evolution is an experiment runner for studying how diffusion network policies can help later task runs to perform better. Skill update feature is optional and auxiliary and is compatible with context diffusion.

The executor backend now targets SkillFlow tasks through Harbor, while the
planner, mediator, judge, reward tagging, reflection, validation, and diffusion
concepts remain the same experimental frame.

## How It Works

Mediated Co-Evolution studies skill files as runtime policies. It uses a
GRPO-like loop at the level of reward-relative skill editing: completed task
traces produce rewards, same-task history forms group-relative evidence, and LLM
reflection rewrites Markdown skills.

Note that skill rewrites only occur when any of the role skill update is enabled. Even when there is no skill update, the mediator still process the logs of prior runs and serve as prior context under `learned_mediator` condition, just that the artifacts are used in different ways under different protocols.

```text
SkillFlow task
    |
    v
Planner -------------- plan/instructions -------------> Executor
   ^                                                     |
   |                                                     | Harbor trace, logs, verifier reward
   |                                                     v
   |<---------------- curated feedback ------------- Mediator
                                                         |
                                                         | trace/report evidence
                                                         v
                                                       Judge
                                                         |
                                                         v
                                             judge reward annotations
```

### Runtime Roles And Policies

The experiment loop has four roles:

- Planner: reads the task instruction plus condition-selected prior context,
  then produces an execution plan.
- Executor: runs the task through the SkillFlow/Harbor backend and returns a
  normalized execution trace and verifier reward.
- Mediator: when the condition uses mediation, compresses and filters execution
  feedback before it is exposed to later Planner iterations.
- Judge: annotates completed traces and reports with rubric-based rewards. Ideally this model can be small.
- Compactor: when context exceeds configurable token budget, another LLM call is invoked to compact the context into more concise summaries.

The mutable runtime policies are:

- `skills/executor/SKILL.md`: portable Executor policy for workflow,
  verification, resource use, and failure handling.
- `skills/planner/SKILL.md`: planning and executor-skill-refinement behavior.
- `skills/mediator/SKILL.md`: feedback filtering and reporting behavior.

Runtime skill files are copied into each experiment directory before the run
starts. Normal experiment runs do not edit the repo-level `skills/` directory.

Executor policy is exposed to SkillFlow through a shared runtime envelope:

```text
Task Instruction
Executor Policy
Task Resources
Verifier Contract
```

For SkillFlow, the runner copies the original task directory, preserves any
task-local `environment/skills/` resources, and rewrites the copied
`instruction.md` to include the envelope. The evolved executor policy remains a
portable policy channel; task-local resources remain task resources.

### Reward Sources And Tagging

The main online evolution reward is the Judge reward. The verifier reward is
kept as provenance and used as fallback when Judge scoring is unavailable and audit.

After a usable task run finishes, the Judge-scored evolution reward is assigned
directly to same-task history entries from that run once those entries exist.
This keeps the reward label aligned with the trace/report/proposal record from
the same iteration, while preserving verifier reward provenance for audit and
fallback scoring.

History entries keep:

- `reward`: the evolution reward used for ranking and reflection.
- `metadata.verifier_reward`: the raw verifier score from Harbor.
- `metadata.reward_source`: `judge`, `verifier_fallback`, or `verifier`.

### Diffusion

Diffusion is an optional graph-aware context route layered on top of the core
planner, executor, mediator, and judge loop. It emits task artifacts, builds a
per-iteration subscription board, and renders selected cross-task artifacts into
planner context according to the configured policy.

Diffusion artifact emission is controlled by `diffusion.enabled`. Rendering
those artifacts into another task's planner context also requires a non-`none`
policy and an iteration number. Eligible artifacts must come from a different
task and a prior source iteration.

Run-outcome artifacts summarize each usable verifier run as mixed signal: what
worked or looked promising, what to avoid or re-check, and concrete verifier
evidence. Successful runs emphasize reusable choices; failed runs emphasize
failure modes while preserving useful partial progress.

Rendered diffusion context shares the transfer-context slot capped by
`budgets.max_transfer_context_tokens`, the same slot used by explicit
cross-task prior context in non-diffusion rows. When a selected artifact
overflows that cap, the renderer first compacts the artifact with the same
compactor path used for other planner-facing context. If the compacted artifact
still cannot fit, it is dropped from the prompt and recorded in the diffusion
audit ledger.

Current diffusion policy values:

- `none`: do not render diffusion context.
- `capped_broadcast`: render the most recent eligible cross-task artifacts up
  to `diffusion.max_artifacts`.
- `random_k`: render a deterministic seeded random sample of eligible
  cross-task artifacts up to `diffusion.max_artifacts`.
- `top_k_similarity`: render eligible artifacts from the strongest incoming
  graph neighbors for the target task, capped by `diffusion.max_artifacts` and
  `diffusion.top_k_neighbors`.
- `langchain_graph`: use LangChain graph and diffusion agents to assign the
  current task node and select causal artifacts from the full artifact store.

The graph precompute command scores directed SkillFlow edge candidates using
family rankings, metadata, task resources, output shape, and instruction text.
Same-family edges flow only from earlier to later ranked tasks; cross-family
edges use lower-weight semantic similarity. It writes profiles, edge weights,
thresholds, kept/cut edges, and connected components for later inspection.

When `medcoevo run` starts with `diffusion.enabled = true` and
`diffusion.graph` set to `task_similarity` or `precomputed_similarity`, it
materializes run-local graph artifacts under `task-graph/` using the configured
local SkillFlow task cache and the default edge threshold `0.05`.

### Skill Update Paths

Two skill update paths can be enabled independently.

Executor skill evolution is advisor-gated:

```text
Planner proposals -> proposal buffer -> SkillAdvisor batch review
                  -> Planner candidate rewrites -> candidate audit
                  -> empirical validation -> Executor skill commit
```

The SkillAdvisor evaluates the proposal batch against the current Executor
skill, including proposal reasoning, diffs, evolution rewards, and reward
sources. If the advisor approves, the Planner drafts candidate Executor skill
rewrites, and the selected candidate is validated before commit.

Planner and Mediator meta-skill evolution runs at the co-evolution interval:

```text
same-task history -> group-relative pairs -> LLM reflection prompt
                  -> candidate skill rewrites -> candidate audit
                  -> empirical validation -> meta-skill commit
```

The reflection prompt shows each worse/better pair with its evolution reward and
task-relative delta. Planner and Mediator reflection each ask for candidate skill
rewrites, validate them empirically, and commit only accepted candidates.
Mediator validation replays a shared source trace through current and candidate
mediator protocols, then scores the executor skill candidate induced by that
feedback through the executor validation gate.

Executor validation compares old and candidate Executor policies on selected
SkillFlow tasks under the same task instruction, task resources, and verifier
contract. It accepts only when the candidate's summed validation reward exceeds
the current skill's summed validation reward by more than
`min_mean_delta * validation_task_count`; per-task regressions are recorded as
validation evidence but do not veto an aggregate improvement.

### Group-Relative Evidence

Co-evolution reflection builds same-role, same-task contrastive pairs from
tagged history. For each task and role, the history store computes:

```text
relative_reward = entry.reward - task_mean_reward
```

It then takes bottom and top reward buckets, forms worse/better pairs, and sorts
them by relative reward gap. This is the GRPO-like step: behavior is judged
relative to other attempts in the same task group rather than by a global scalar
alone.

## Requirements

- Python `>=3.13`
- `uv`
- Harbor CLI on `PATH`
- Docker for local Harbor execution
- `OPENROUTER_API_KEY` for planner, mediator, judge, and any executor agent
  configured to route through OpenRouter

MedCoevo runs SkillFlow tasks through the Harbor agent configured in
`executor_runtime.agent_name`; the repo default is currently `claude-code`.
`executor_runtime.agent_env` passes agent-specific environment variables.
The default config routes Claude-compatible executor traffic through
OpenRouter by setting `ANTHROPIC_AUTH_TOKEN=${OPENROUTER_API_KEY}` and
`ANTHROPIC_BASE_URL=https://openrouter.ai/api`.

## Quick Start

Use `--help` for flag details. README only provides some quick commands.

```bash
export OPENROUTER_API_KEY=...
uv sync --dev
uv run medcoevo --help
```

Build the local Harbor base image once:

```bash
uv run medcoevo build-base-image
```

Run the local smoke task:

```bash
uv run medcoevo run \
  --family smoke \
  --iterations 1 \
  --condition no_feedback \
  --skill-updates none \
  --run-id smoke
```

Run one family with graph diffusion:

```bash
uv run medcoevo run \
  --family Weighted-Risk-Assessment \
  --iterations 3 \
  --condition no_feedback \
  --skill-updates none \
  --coevo-interval 99 \
  --advisor-buffer-max 99 \
  --diffusion-enabled \
  --diffusion-policy top_k_similarity \
  --diffusion-graph task_similarity \
  --run-id all-wra-top-k-similarity
```

Or set those parameters in `config/default.toml`

Experiment outputs are written under:

```text
data/experiments/<timestamp>-<run-id>/
```

Saved warmup stores are written under:

```text
data/artifact-stores/<experiment-folder>/
```

Use the CLI for the full flag list:

```bash
uv run medcoevo run --help
uv run medcoevo matrix --help
uv run medcoevo extract --help
uv run medcoevo inspect --help
```

## Local Smoke Task

A minimal task is included at:

```text
benchmarks/skillflow/tasks/smoke-skillflow/
```

It lets us confirm the Harbor output shape and parser contract without pulling a
remote dataset.

## CLI Overview

Pick the experiment family:

- `--family <name>`: bootstrap an 8-task stream from the cached family, with replacement.
- Repeat `--family` to sample from multiple cached families.
- `--split train|validation|test`: optionally sample from a deterministic 60/20/20
  split of the selected family pool.

Remote Harbor run on the configured GCP VM:

```bash
uv run medcoevo run \
  --family smoke \
  --iterations 1 \
  --seed 1 \
  --cloud \
  --cloud-env-file .env \
  --run-id remote-smoke
```

## `matrix`

`matrix` runs fixed `skill_updates x diffusion_policy` rows. Use `--list` to
see row indexes.

Run one row:

```bash
uv run medcoevo matrix \
  --family Compensation-Scenario-Modeling \
  --iterations 3 \
  --seed 42 \
  --index 0 \
  --run-id csm-matrix-skill-none-diffusion-none
```

Cheap matrix smoke:

```bash
uv run medcoevo matrix \
  --family smoke \
  --iterations 1 \
  --seed 1
```

## Warmup Artifact Stores

Save a first-batch diffusion store:

```bash
uv run medcoevo matrix \
  --family Weighted-Risk-Assessment \
  --iterations 1 \
  --index 1 \
  --save \
  --run-id wra-warmup
```

This writes `data/artifact-stores/<experiment-folder>/`.

Start from that saved store:

```bash
uv run medcoevo matrix \
  --family Weighted-Risk-Assessment \
  --iterations 3 \
  --index 1 \
  --artifact data/artifact-stores/<experiment-folder> \
  --run-id wra-preloaded
```

Freeze the store so only the preloaded artifacts can diffuse:

```bash
uv run medcoevo matrix \
  --family Weighted-Risk-Assessment \
  --iterations 3 \
  --index 1 \
  --artifact data/artifact-stores/<experiment-folder> \
  --freeze \
  --run-id wra-frozen
```

Rebuild a store from an old experiment that already has `diffusion/artifacts/`:

```bash
uv run medcoevo extract -p data/experiments/<old-run>
```

## `inspect`

Inspect the experiment run basic statistics:

```bash
uv run medcoevo inspect
```

Inspect a specific experiment:

```bash
uv run medcoevo inspect data/experiments/<run-dir>
uv run medcoevo inspect data/experiments/<batch-dir>/<row-dir>
```

Emit machine-readable JSON:

```bash
uv run medcoevo inspect --json
```

`inspect` understands both single-run directories and baseline matrix
directories. For runs with diffusion output it reports artifact counts,
rendered subscription counts, graph snapshot counts, and planner prior-context
token summaries. `--config-dir` is used only when `inspect` needs to locate the
newest experiment from the configured `paths.data_dir`.

## `compare-context-budgets`

`compare-context-budgets` is read-only. It does not call Harbor or an LLM.

```bash
uv run medcoevo compare-context-budgets \
  data/experiments/<run-a> \
  data/experiments/<run-b>
```

Emit JSON for notebooks or scripts:

```bash
uv run medcoevo compare-context-budgets \
  data/experiments/<run-a> \
  data/experiments/<run-b> \
  --json
```

The command treats same-task prior tokens, transfer-context tokens, and total
planner prior-context tokens as observed metrics, not config knobs. Changes to
`budgets.max_same_task_prior_tokens` or `budgets.max_transfer_context_tokens`
are experiment setup differences; changes in the token fields are outcomes of
that setup.

Audit skill-update provenance from an experiment metrics file:

```bash
uv run python -m mediated_coevo.analysis.evolution_audit data/experiments/<run-dir>
```

## `create-graph`

Create a directed SkillFlow graph from local SkillFlow tasks:

```bash
uv run medcoevo create-graph \
  --tasks-root benchmarks/skillflow/tasks \
  --output-dir data/task_graphs/skillflow-local \
  --threshold 0.05
```

The graph artifacts include task profiles, directed edge components, score
weights, active thresholds, kept/cut edges, and connected components.

## `build-base-image`

Build the required SkillFlow Harbor CLI base image:

```bash
uv run medcoevo build-base-image
```

Preview the build command:

```bash
uv run medcoevo build-base-image --dry-run
```

## `sync`

List remote SkillFlow task IDs before downloading:

```bash
uv run medcoevo list
uv run medcoevo list --family Distribution-Center-Auditing
```

Use `--local` to list already cached tasks under the configured local task
directory.

Download all remote SkillFlow test tasks into the configured local cache:

```bash
uv run medcoevo sync --tasks all
```

Download only selected tasks by repeating `--tasks` or passing comma-separated
IDs:

```bash
uv run medcoevo sync \
  --tasks Distribution-Center-Auditing/harbor_returns_disposition_audit
```

By default the configured dataset is `zhang-ziao/SkillFlow-Task` and the target
directory is `benchmarks/skillflow/tasks/`. Hugging Face files are downloaded
from `test_tasks/` and flattened into the local `tasks/` cache, so runtime task
IDs are available directly under `tasks/<Family>/<Task>/`:

```bash
uv run medcoevo run --family Distribution-Center-Auditing
```

| `--config-dir` | `config/` | Directory containing `default.toml`. |
| `--verbose`, `-v` | false | Enable debug logging. |

`sync` options:

| Option                    | Default                     | Meaning                                                      |
| ------------------------- | --------------------------- | ------------------------------------------------------------ |
| `--tasks`, `--task`, `-t` | all remote test tasks       | Task IDs to download; repeat, comma-separate, or pass `all`. |
| `--output-dir`            | configured local cache      | Destination tasks directory.                                 |
| `--dataset`               | `zhang-ziao/SkillFlow-Task` | Hugging Face dataset ID.                                     |
| `--config-dir`            | `config/`                   | Directory containing `default.toml`.                         |
| `--verbose`, `-v`         | false                       | Enable debug logging.                                        |

## Cloud VM Harbor Setup

`medcoevo run --cloud` keeps the co-evolution control plane on the local
machine, but sends each prepared SkillFlow task workspace to an existing GCP VM
for `harbor run`. The VM is only a remote Docker/Harbor host: the full repo is
not copied to the VM, `medcoevo run` is not run on the VM, and experiment
outputs stay under local `data/experiments/`.

```bash
uv run medcoevo run \
  --family smoke \
  --iterations 1 \
  --condition no_feedback \
  --skill-updates none \
  --cloud
```

Use `--cloud-env-file` when the GCP settings are not in `.env`.

The local machine needs:

- `gcloud` installed and authenticated for the configured VM.
- `OPENROUTER_API_KEY` exported locally for planner, mediator, and judge model
  calls. The remote Hermes executor reads its OpenRouter key from the configured
  Secret Manager secret.

The VM must have Docker, `uv`, Harbor, and `gcloud` on `PATH`, plus access to
the configured OpenRouter secret when remote execution reads credentials from
Secret Manager.

## Outputs

Typical single-run output:

```text
data/experiments/<timestamp>-<run-id>/
|-- config.toml
|-- metrics.jsonl
|-- summary.json
|-- artifacts/
|   |-- judge_rewards.jsonl
|   |-- reports/
|   |-- traces/
|   |-- validation/
|   `-- skill_updates/
|-- history/
|-- jobs/
|-- harnesses/
|-- skills/
|-- skills_snapshots/
|-- task-graph/
`-- diffusion/
```

Important files:

- `config.toml`: resolved config after CLI overrides.
- `metrics.jsonl`: per-iteration records.
- `summary.json`: aggregate verifier rewards, Judge rewards, confidence
  intervals, token totals, per-task summaries, and environment failure count.
- `artifacts/traces/`: normalized SkillFlow/Harbor execution traces.
- `artifacts/reports/`: Mediator reports.
- `artifacts/validation/`: Executor, Planner, and Mediator validation evidence.
- `artifacts/skill_updates/`: committed skill update ledger, full update JSON,
  and readable diffs for post-run regression analysis.
- `history/`: feedback history entries, rejected proposal batches, and rejected
  reflection candidates.
- `harnesses/`: learned repo-root harness overlays. When `--harness-dir` is
  used, the source harness is copied to `harnesses/seed/` and recorded in
  `harnesses/active_harness.json`.
- `skills/`: run-local skill copy.
- `skills_snapshots/`: committed skill snapshots.
- `task-graph/`: run-local precomputed graph artifacts when graph-aware
  diffusion is enabled for `run`.
- `diffusion/`: graph diffusion artifacts and rendered subscriptions when
  enabled.

Diffusion output includes:

- `diffusion/artifacts/*.json`: emitted low-risk source artifacts such as
  `run_outcome` summaries, mediator report summaries, debug hints, and
  regression warnings.
- `diffusion/graph_snapshots/*.json`: per-iteration graph snapshots used for
  artifact selection.
- `diffusion/diffused_records.jsonl`: audit ledger for eligible, selected, and
  rendered artifact routes. Softmax policies also write not-selected candidate
  rows with candidate probabilities and selected-target metadata.

To start a new batch from a validated learned harness snapshot, pass the
snapshot as a repo-root overlay:

```bash
uv run medcoevo run \
  --harness-dir data/experiments/<run>/harnesses/<update-id> \
  --family Weighted-Risk-Assessment \
  --family HWPX-Document-Automation \
  --split validation \
  --condition learned_mediator \
  --skill-updates none \
  --diffusion-enabled \
  --diffusion-policy langchain_graph \
  --diffusion-graph none
```

The overlay may either contain `src/`, `config/`, or `tests/` directly, or put
those paths under an `overlay/` subdirectory. Root-level `manifest.*` files are
kept as metadata and are not copied into the repo.

Executor policy observability fields in `metrics.jsonl`:

- `executor_policy_hash`: hash of the policy text injected for the run.
- `executor_policy_injected`: whether a non-empty policy was included.
- `executor_policy_injection`: where the policy was exposed.
- `task_resource_count` and `task_resource_names`: task-local resources exposed
  alongside the policy.
- `verifier_contract_kind`: the SkillFlow verifier contract shown to the
  Executor.

Diffusion observability fields in `metrics.jsonl`:

- `diffusion_enabled`, `diffusion_policy`, and `diffusion_graph`.
- `graph_snapshot_id`.
- `diffusion_artifacts_eligible`, `diffusion_artifacts_selected`, and
  `diffusion_artifacts_rendered`.
- `same_task_prior_tokens`, `transfer_context_kind`,
  `transfer_context_tokens`, and `total_planner_prior_context_tokens`.
- `max_same_task_prior_tokens`, `max_transfer_context_tokens`, and
  `max_total_prior_context_tokens`, recorded as the effective caps used for
  that row.
- `context_budget_violation`, `compacted_diffusion_artifact_ids`, and
  `dropped_for_budget_artifact_ids`.
- `source_task_ids`.
- `reward_after_diffusion_context` and `regression_after_diffusion_context`
  when rendered diffusion context was present.

Executor cost provenance fields in `metrics.jsonl`:

- `executor_reported_cost_usd` and `executor_reported_cost_source` when Harbor
  exposes an executor billing cost in `agent_result.cost_usd`.
- Token totals remain separately reported in `prompt_tokens_by_agent`,
  `completion_tokens_by_agent`, `total_tokens_by_agent`, and
  `executor_cache_read_tokens`.

## Configuration

Default configuration lives in:

```text
config/default.toml
```

The current config controls:

- model IDs for Planner, Executor, Mediator, and Judge;
- prompt, completion, same-task-prior, and transfer-context token budgets;
- default iteration cadence;
- skill update and validation defaults;
- diffusion emission, selection, and graph-routing settings;
- local output and benchmark paths;
- Harbor executor runtime settings;
- SkillFlow dataset synchronization settings.

CLI options override the loaded config for a single run. The resolved config is
persisted in the experiment directory as `config.toml`. Required experiment
settings must be present in `default.toml` unless the CLI provides an override;
otherwise the command fails before runtime setup and names the missing setting.
Budget fields are required config fields as well; `core/config.py` intentionally
does not provide runtime defaults for `[budgets]`.

Current task/runtime defaults include:

```toml
[budgets]
max_skill_tokens = 4000
max_same_task_prior_tokens = 300
max_transfer_context_tokens = 900
trace_excerpt_tokens = 6000
historical_summary_tokens = 3000
mediator_report_tokens = 1200
planner_context_tokens = 24000

[diffusion]
enabled = false
policy = "none"
graph = "none"
max_artifacts = 3
top_k_neighbors = 5
avoid_recheck_max_artifacts = 1

[paths]
benchmarks_dir = "benchmarks/skillflow"

[executor_runtime]
task_dirs = ["tasks"]
agent_name = "claude-code"
sync_enabled = false
dataset = "zhang-ziao/SkillFlow-Task"
dataset_repo_type = "dataset"
harbor_agent_setup_timeout_multiplier = 2.0

[executor_runtime.agent_env]
ANTHROPIC_API_KEY = ""
ANTHROPIC_AUTH_TOKEN = "${OPENROUTER_API_KEY}"
ANTHROPIC_BASE_URL = "https://openrouter.ai/api"

[experiment.skill_validation]
sample_size = 3
min_tag_overlap = 1
```

## Testing

Run the default unit suite:

```bash
uv run pytest
```

Run static checks:

```bash
uv run ruff check .
uv run mypy src tests
```

## Troubleshooting

`OPENROUTER_API_KEY is required`

Export `OPENROUTER_API_KEY` before running experiments.

`harbor CLI not found on PATH`

Install Harbor, then confirm `harbor --version`. For orchestrator-only checks
that should not call Harbor, set `executor_runtime.harbor_required = false` in
`config/default.toml`.

Docker failures in local runs

Start Docker Desktop or another local Docker daemon, then confirm:

```bash
docker info
```

Missing SkillFlow task

Use `uv run medcoevo list`, download with `uv run medcoevo sync --tasks ...`,
select an existing local `--family`, or add a task under
`benchmarks/skillflow/tasks/<Family>/`.

Missing SkillFlow prebuilt image

MedCoevo's required setup only builds the SkillFlow base image:

```bash
uv run medcoevo build-base-image
```

If a task declares `[environment].docker_image`, that task is opting into
SkillFlow's optional task-image prebuild path. Either remove the stale
`docker_image` field so Harbor builds from the task Dockerfile, or run the
optional upstream task prebuilder yourself.

Invalid experiment design

The CLI validates contradictory condition/update combinations before starting
runtime side effects. Examples:

- `no_feedback` cannot enable any skill updates.
- Mediator skill updates require `learned_mediator`.
- `shared_notes` cannot enable Executor updates.
- `static_mediator` cannot enable Mediator updates.

## Related Work

- Claude API Advisor: https://platform.anthropic.com/docs/en/agents-and-tools/tool-use/advisor-tool
- Spark - Shared Agentic Memory: https://arxiv.org/abs/2511.08301
- Multi-Agent Evolve: https://arxiv.org/abs/2510.23595
- OpenSpace: https://github.com/HKUDS/OpenSpace
- Group-Evolving Agents: https://arxiv.org/abs/2602.04837
- Self-Evolving Coordination Protocol: https://arxiv.org/abs/2602.02170
- Rubric as Reward: https://arxiv.org/pdf/2507.17746
- Skill Collective Evolution: https://github.com/AMAP-ML/SkillClaw
- LLM-as-Judge guide: https://arxiv.org/pdf/2306.05685
- Textual Parameter Graph Optimization: https://arxiv.org/pdf/2604.20714
