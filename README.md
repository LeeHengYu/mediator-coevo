# Mediated Co-Evolution

Mediated Co-Evolution is an experiment runner for studying how agent skills
change when execution feedback is routed through different context policies.
The executor backend now targets SkillFlow tasks through Harbor, while the
planner, mediator, judge, reward tagging, reflection, validation, and diffusion
concepts remain the same experimental frame.

## How It Works

Mediated Co-Evolution studies skill files as runtime policies. It uses a
GRPO-like loop at the level of reward-relative skill editing: completed task
traces produce rewards, same-task history forms group-relative evidence, and LLM
reflection rewrites Markdown skills. It does not train model weights.

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
- Judge: annotates completed traces and reports with rubric-based rewards.

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
kept as provenance and used as fallback when Judge scoring is unavailable.

After a usable task run finishes, the Judge-scored evolution reward is assigned
directly to same-task history entries from that run once those entries exist.
This keeps the reward label aligned with the trace/report/proposal record from
the same iteration, while preserving verifier reward provenance for audit and
fallback scoring.

History entries keep:

- `reward`: the evolution reward used for ranking and reflection.
- `metadata.verifier_reward`: the raw verifier score from Harbor.
- `metadata.reward_source`: `judge`, `verifier_fallback`, or `verifier`.

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

Rendered diffusion context is capped by
`budgets.max_diffusion_context_tokens`. When a selected artifact overflows that
cap, the renderer first compacts the artifact with the same compactor path used
for other planner-facing context. If the compacted artifact still cannot fit, it
is dropped from the prompt and recorded in the diffusion audit ledger.

Current diffusion policy values:

- `none`: do not render diffusion context.
- `capped_broadcast`: render the most recent eligible cross-task artifacts up
  to `diffusion.max_artifacts`.
- `random_k`: render a deterministic seeded random sample of eligible
  cross-task artifacts up to `diffusion.max_artifacts`.
- `top_k_similarity`: render eligible artifacts from the strongest incoming
  graph neighbors for the target task, capped by `diffusion.max_artifacts` and
  `diffusion.top_k_neighbors`.

The graph precompute command scores directed SkillFlow edge candidates using
family rankings, metadata, task resources, output shape, and instruction text.
Same-family edges flow only from earlier to later ranked tasks; cross-family
edges use lower-weight semantic similarity. It writes profiles, edge weights,
thresholds, kept/cut edges, and connected components for later inspection.

When `medcoevo run` starts with `diffusion.enabled = true` and
`diffusion.graph` set to `task_similarity` or `precomputed_similarity`, it
materializes run-local graph artifacts under `task-graph/` using the configured
local SkillFlow task cache and the default edge threshold `0.05`.

## Requirements

- Python `>=3.13`
- `uv`
- Harbor CLI on `PATH`
- Docker for local Harbor execution
- `OPENROUTER_API_KEY` for planner, mediator, judge, and the Hermes executor

MedCoevo always runs SkillFlow tasks through Harbor's `hermes` agent. The
executor agent is not configurable in `config/default.toml`; local Harbor
subprocesses deliberately drop `OPENAI_API_KEY` so Hermes routes through
`OPENROUTER_API_KEY`.

## Quick Start

Inspect the CLI:

```bash
uv run medcoevo --help
```

Build the required SkillFlow base image. This wraps the upstream
`docker/harbor-cli-base/build.sh` quick-start step:

```bash
uv run medcoevo build-base-image
```

SkillFlow's task-image prebuild script is optional upstream and is not part of
the default MedCoevo setup path.

Run a short local SkillFlow smoke experiment:

```bash
uv run medcoevo run \
  --task smoke-skillflow \
  --iterations 1 \
  --condition no_feedback \
  --skill-updates none \
  --run-id smoke
```

Run the current Hermes-backed Weighted-Risk-Assessment diffusion experiment:

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

Experiment outputs are written under:

```text
data/experiments/<timestamp>-<run-id>/
```

## Local Smoke Task

A minimal task is included at:

```text
benchmarks/skillflow/tasks/smoke-skillflow/
```

It lets us confirm the Harbor output shape and parser contract without pulling a
remote dataset.

## CLI Overview

Top-level commands:

```bash
uv run medcoevo run
uv run medcoevo matrix
uv run medcoevo inspect <run-path>
uv run medcoevo compare-context-budgets <run-a-path> <run-b-path>
uv run medcoevo create-graph
uv run medcoevo build-base-image
uv run medcoevo list
uv run medcoevo sync
```

Shell completion helpers:

```bash
uv run medcoevo --install-completion
uv run medcoevo --show-completion
```

## `run`

`run` executes one SkillFlow co-evolution experiment. It requires at least one
task selector.

Selectors:

- `--task <id>`: repeatable; comma-separated IDs are also accepted.
- `--family <name>`: run all local tasks with matching SkillFlow family
  metadata.
- `--task-set <name>`: read `benchmarks/skillflow/task_sets/<name>.txt`.

Core run options:

| Option                                          | Default      | Meaning                                                             |
| ----------------------------------------------- | ------------ | ------------------------------------------------------------------- |
| `--iterations`                                  | config value | Number of experiment iterations.                                    |
| `--seed`                                        | config value | Random seed.                                                        |
| `--condition`                                   | config value | Feedback routing condition.                                         |
| `--skill-updates`                               | config value | Which skill families may be committed.                              |
| `--advisor-buffer-max`                          | config value | Executor proposal batch size override.                              |
| `--coevo-interval`                              | config value | Planner/Mediator reflection interval override.                      |
| `--diffusion-enabled`, `--no-diffusion-enabled` | config value | Override diffusion emission and routing.                            |
| `--diffusion-policy`                            | config value | Override artifact selection policy.                                 |
| `--diffusion-graph`                             | config value | Override graph policy/name.                                         |
| `--diffusion-max-artifacts`                     | config value | Maximum artifacts rendered for a target context.                    |
| `--diffusion-top-k-neighbors`                   | config value | Neighbor cap for `top_k_similarity`.                                |
| `--harbor-agent-setup-timeout-multiplier`       | config value | Forward a Harbor setup timeout multiplier for slow agent setup.     |
| `--run-id`                                      | auto suffix  | Optional run id suffix for the timestamp-prefixed output directory. |
| `--config-dir`                                  | `config/`    | Directory containing `default.toml`.                                |
| `--cloud`                                       | false        | Run Harbor jobs on the configured GCP VM.                           |
| `--cloud-env-file`                              | `.env`       | Dotenv file containing GCP VM Harbor settings.                      |
| `--verbose`, `-v`                               | false        | Enable debug logging.                                               |

Diffusion override values:

- `--diffusion-policy` accepts `none`, `capped_broadcast`, `random_k`, or
  `top_k_similarity`.
- `--diffusion-graph none` records broadcast-style snapshots for non-graph
  policies.
- `--diffusion-graph task_similarity` and
  `--diffusion-graph precomputed_similarity` use run-local precomputed
  similarity artifacts when graph-aware diffusion is enabled.
- `--diffusion-max-artifacts` and `--diffusion-top-k-neighbors` must be at
  least `1`.

Feedback conditions:

- `no_feedback`: no prior feedback; cannot enable skill updates.
- `full_traces`: Planner receives compact trace summaries from prior runs.
- `shared_notes`: Planner receives shared notes configured in the config file.
- `static_mediator`: Mediator reports are used, but Mediator skill updates are
  invalid.
- `learned_mediator`: Mediator reports are used, and Mediator/Planner
  co-evolution can be enabled.

Skill update values:

- `none`
- `executor`
- `planner`
- `mediator`
- `all`
- comma-separated role combinations such as `executor,planner`

`none` and `all` cannot be combined with other values.

Current full-family experiment command:

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
  --diffusion-max-artifacts 3 \
  --diffusion-top-k-neighbors 3 \
  --run-id all-wra-top-k-similarity-hermes
```

## `matrix`

`matrix` runs the six baseline rows against the same SkillFlow task selection,
seed, model config, and budget config.

```bash
uv run medcoevo matrix \
  --task smoke-skillflow \
  --iterations 1 \
  --seed 42
```

Baseline rows:

| Preset                             | Condition          | Skill updates               |
| ---------------------------------- | ------------------ | --------------------------- |
| `no_feedback`                      | `no_feedback`      | `none`                      |
| `full_trace_same_task`             | `full_traces`      | `none`                      |
| `static_mediator_same_task`        | `static_mediator`  | `none`                      |
| `planner_only_skill_evolution`     | `learned_mediator` | `planner`                   |
| `mediator_only_protocol_evolution` | `learned_mediator` | `mediator`                  |
| `full_coevolution`                 | `learned_mediator` | `executor,planner,mediator` |

Each row gets an isolated copy of the skill tree under its experiment
directory.

`matrix` accepts the same task selectors as `run`, plus `--iterations`,
`--seed`, `--coevo-interval`, `--advisor-buffer-max`, the diffusion override
options, `--config-dir`, and `--verbose`. Condition and skill-update settings
come from the fixed baseline presets and are not CLI options on `matrix`.

## `inspect`

Inspect the newest experiment:

```bash
uv run medcoevo inspect
```

Inspect a specific experiment:

```bash
uv run medcoevo inspect data/experiments/<run-dir>
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

`compare-context-budgets` is a read-only evaluation command for two completed
experiment directories. It does not call Harbor or any LLM. It reads
`config.toml`, `metrics.jsonl`, and diffusion audit artifacts, then reports
whether the runs are comparable, which budget fields differ, how realized
prior-context tokens changed, and whether rendered diffusion records pass
basic provenance checks.

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

The command treats same-task prior tokens, cross-task prior tokens, diffusion
tokens, and total planner prior-context tokens as observed metrics, not config
knobs. A change to `budgets.max_diffusion_context_tokens` is an experiment setup
difference; changes in the token fields are outcomes of that setup.

Audit skill-update provenance, adjacent reward effects, Mediator report
effects, committed-update ledger entries, rejected reflection evidence, diff
artifact paths, and adjacent reward regressions from an experiment metrics file:

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

`create-graph` options:

| Option         | Default                            | Meaning                                            |
| -------------- | ---------------------------------- | -------------------------------------------------- |
| `--threshold`  | `0.05`                             | Minimum similarity score required to keep an edge. |
| `--tasks-root` | `benchmarks/skillflow/tasks`       | Local SkillFlow task directory to analyze.         |
| `--output-dir` | `data/task_graphs/skillflow-local` | Directory where JSON graph artifacts are written.  |

The command writes `task_profiles.json`, `pairwise_similarity.json`, and
`graph_summary.json`.

## `build-base-image`

Build the required SkillFlow Harbor CLI base image:

```bash
uv run medcoevo build-base-image
```

Options:

| Option             | Default                                 | Meaning                                     |
| ------------------ | --------------------------------------- | ------------------------------------------- |
| `--base-image-tag` | `skillflow/harbor-cli-base:ubuntu24.04` | Docker tag for the base image.              |
| `--dry-run`        | false                                   | Print the build command without running it. |
| `--verbose`, `-v`  | false                                   | Enable debug logging.                       |

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
uv run medcoevo run --task Distribution-Center-Auditing/harbor_returns_disposition_audit
```

`list` options:

| Option            | Default                     | Meaning                                                       |
| ----------------- | --------------------------- | ------------------------------------------------------------- |
| `--family`        | none                        | Filter task IDs by SkillFlow family.                          |
| `--local`         | false                       | List cached local tasks instead of remote Hugging Face tasks. |
| `--dataset`       | `zhang-ziao/SkillFlow-Task` | Hugging Face dataset ID for remote listing.                   |
| `--config-dir`    | `config/`                   | Directory containing `default.toml`.                          |
| `--verbose`, `-v` | false                       | Enable debug logging.                                         |

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
  --task smoke-skillflow \
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
  rendered artifact routes.

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
- `same_task_prior_tokens`, `cross_task_prior_tokens`,
  `diffusion_context_tokens`, and `total_planner_prior_context_tokens`.
- `max_same_task_prior_tokens`, `max_cross_task_prior_tokens`,
  `max_diffusion_context_tokens`, and `max_total_prior_context_tokens`, recorded
  as the effective caps used for that row.
- `context_budget_violation`, `compacted_diffusion_artifact_ids`, and
  `dropped_for_budget_artifact_ids`.
- `source_task_ids`.
- `reward_after_diffusion_context` and `regression_after_diffusion_context`
  when rendered diffusion context was present.

## Configuration

Default configuration lives in:

```text
config/default.toml
```

The current config controls:

- model IDs for Planner, Executor, Mediator, and Judge;
- prompt, completion, and diffusion-context token budgets;
- default iteration cadence;
- skill update and validation defaults;
- diffusion emission, selection, and graph-routing settings;
- local output and benchmark paths;
- Hermes/Harbor runtime settings;
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
max_diffusion_context_tokens = 4000
trace_excerpt_tokens = 6000
historical_summary_tokens = 3000
mediator_report_tokens = 4000
planner_context_tokens = 24000

[diffusion]
enabled = false
policy = "none"
graph = "none"
max_artifacts = 3
top_k_neighbors = 3

[paths]
benchmarks_dir = "benchmarks/skillflow"

[executor_runtime]
task_dirs = ["tasks"]
sync_enabled = false
dataset = "zhang-ziao/SkillFlow-Task"
dataset_repo_type = "dataset"
harbor_agent_setup_timeout_multiplier = 3.0

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
uv run ruff check src tests
uv run mypy src
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
select an existing local `--task`, or add a task under
`benchmarks/skillflow/tasks/`.

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
