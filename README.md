# Mediated Co-Evolution

Mediated Co-Evolution is an experiment runner for studying how agent skills
change when execution feedback is routed through different context policies.
It supports SkillsBench tasks through Harbor, SWE-bench tasks through the
official SWE-bench/Modal harness, and mixed runs that include both task types.

The main CLI entrypoint is:

```bash
uv run medcoevo --help
```

## Normal Experiment

For a normal short SkillsBench experiment, run:

```bash
uv run medcoevo run \
  --skillsbench-task fix-build-agentops \
  --swebench-instance sympy__sympy-13915 \
  --iterations 2 \
  --skill-updates none \
  --advisor-buffer-max 2 \
  --coevo-interval 2 \
  --run-id <suffix>
  --no-skill-validation
```

This selects one SkillsBench task, runs two iterations, disables committed skill
updates, sets the advisor and reflection cadence to two iterations, and skips
executor skill candidate validation. The default condition is
`learned_mediator`, so the Mediator still produces feedback reports unless you
override `--condition`.

Experiment outputs are written under:

```text
data/experiments/<timestamp>-<suffix>/
```

Use `--run-id <suffix>` to choose the suffix while keeping the timestamp prefix:

```bash
uv run medcoevo run \
  --skillsbench-task fix-build-agentops \
  --iterations 2 \
  --skill-updates none \
  --run-id agentops-smoke
```

## Setup

Install project dependencies with `uv`:

```bash
uv sync --dev
```

Export the model credential used by all LLM calls:

```bash
export OPENROUTER_API_KEY=...
```

SkillsBench runs require Harbor and a local container runtime:

```bash
uv tool install harbor
harbor --version
docker --version
docker compose version
```

SWE-bench runs use Modal instead of local Docker. Configure Modal before running
SWE-bench commands:

```bash
modal token new
```

## How It Works

```text
Benchmark task
    |
    v
Planner -------------- plan/instructions -------------> Executor
   ^                                                     |
   |                                                     | trace, logs, verifier reward
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

The experiment loop has three agent roles:

- Planner: reads the benchmark instruction plus condition-selected prior
  context, then produces an execution plan.
- Executor: runs the task in the selected benchmark backend and returns a
  normalized execution trace and reward.
- Mediator: when the condition uses mediation, compresses and filters execution
  feedback before it is exposed to later Planner iterations.
- Judge: annotates completed traces and reports with rubric-based rewards for
  analysis.

Two skill update paths can be enabled independently:

- Executor skill updates edit `skills/executor/SKILL.md` through proposal,
  advisor review, and optional validation.
- Planner and Mediator meta-skill updates edit `skills/planner/SKILL.md` and
  `skills/mediator/SKILL.md` on the co-evolution interval using contrastive
  history pairs.

Runtime skill files are copied into each experiment directory before the run
starts. Normal experiment runs do not edit the repo-level `skills/` directory.

## CLI Overview

Top-level commands:

```bash
uv run medcoevo run
uv run medcoevo matrix
uv run medcoevo inspect
uv run medcoevo skillsbench sync
uv run medcoevo swebench list-instances
uv run medcoevo swebench smoke
```

Shell completion helpers:

```bash
uv run medcoevo --install-completion
uv run medcoevo --show-completion
```

## `run`

`run` executes one SkillsBench, SWE-bench, or mixed co-evolution experiment. It
requires at least one task selector.

SkillsBench selectors:

- `--skillsbench-task <id>`: repeatable; comma-separated IDs are also accepted.
- `--skillsbench-task-set skillsbench-10`: curated 10-task set.
- `--skillsbench-task-set skillsbench-all`: discover all local and remote
  SkillsBench tasks, then fetch missing tasks lazily.
- `--tasks` and `--task-set`: legacy aliases for SkillsBench selection.

SWE-bench selectors:

- `--swebench-instance <id>`: repeatable; comma-separated IDs are also accepted.
- `--swebench-limit <n>`: first `n` instances from the configured split.
- `--swebench-eval-instance <id>`: optional frozen eval after evolution.
- `--swebench-eval-limit <n>`: first `n` instances for frozen eval.

Core run options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--iterations` | `30` | Number of experiment iterations. |
| `--seed` | `42` | Random seed. |
| `--condition` | `learned_mediator` | Feedback routing condition. |
| `--skill-updates` | `all` | Which skill families may be committed. |
| `--advisor-buffer-max` | config value | Executor proposal batch size override. |
| `--coevo-interval` | config value | Planner/Mediator reflection interval override. |
| `--skill-validation` / `--no-skill-validation` | config value | Enable or disable executor candidate validation. |
| `--run-id` | generated | Timestamp-prefixed output directory suffix. |
| `--config-dir` | `config/` | Directory containing `default.toml`. |
| `--verbose`, `-v` | false | Enable debug logging. |

Feedback conditions:

- `no_feedback`: no prior feedback; cannot enable skill updates.
- `full_traces`: Planner receives compact same-task trace summaries.
- `shared_notes`: Planner receives configured shared notes configured in config file.
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

Examples:

```bash
uv run medcoevo run \
  --skillsbench-task fix-build-google-auto \
  --iterations 1 \
  --seed 42
```

```bash
uv run medcoevo run \
  --skillsbench-task-set skillsbench-10 \
  --condition learned_mediator \
  --skill-updates executor,planner,mediator \
  --iterations 4
```

```bash
uv run medcoevo run \
  --skillsbench-task fix-build-agentops \
  --swebench-instance sympy__sympy-13915 \
  --iterations 4 \
  --advisor-buffer-max 2 \
  --coevo-interval 1 \
  --skill-validation
```

## `matrix`

`matrix` runs the six baseline rows against the same SkillsBench task selection,
seed, model config, and budget config. Matrix runs are SkillsBench-only.

```bash
uv run medcoevo matrix \
  --task-set skillsbench-10 \
  --iterations 1 \
  --seed 42
```

Supported options include `--tasks`, `--task-set`, `--iterations`, `--seed`,
`--coevo-interval`, `--advisor-buffer-max`,
`--skill-validation` / `--no-skill-validation`, `--config-dir`, and
`--verbose`.

Baseline rows:

| Preset | Condition | Skill updates |
| --- | --- | --- |
| `no_feedback` | `no_feedback` | `none` |
| `full_trace_same_task` | `full_traces` | `none` |
| `static_mediator_same_task` | `static_mediator` | `none` |
| `planner_only_skill_evolution` | `learned_mediator` | `planner` |
| `mediator_only_protocol_evolution` | `learned_mediator` | `mediator` |
| `full_coevolution` | `learned_mediator` | `executor,planner,mediator` |

Each row gets an isolated copy of the skill tree under its experiment
directory.

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
directories.

## `skillsbench sync`

SkillsBench tasks are cached under `benchmarks/skillsbench/tasks/`. Missing
tasks are fetched on demand when `executor_runtime.remote_fetch = true`.

Pre-cache selected tasks:

```bash
uv run medcoevo skillsbench sync \
  --tasks fix-build-agentops,dialogue-parser
```

Pre-cache the curated set:

```bash
uv run medcoevo skillsbench sync \
  --task-set skillsbench-10
```

`skillsbench sync` intentionally does not support `skillsbench-all`, because
syncing every remote task can be expensive.

The task archive is configured in `config/default.toml`:

```toml
[executor_runtime]
remote_fetch = true
archive_url = "https://github.com/benchflow-ai/skillsbench/archive/refs/heads/main.zip"
# archive_sha256 = "<64 hex chars>"
```

For reproducible experiments, pin `archive_url` to a commit or tag archive and
set `archive_sha256`.

## `swebench`

List valid SWE-bench Lite instance IDs:

```bash
uv run medcoevo swebench list-instances --limit 20
```

Filter by repository substring:

```bash
uv run medcoevo swebench list-instances \
  --repo-filter django \
  --limit 20
```

Run the standalone SWE-bench smoke command:

```bash
uv run medcoevo swebench smoke
```

The smoke command defaults to the SWE-bench Lite `test` split and the
`sympy__sympy-20590` instance when no `--instance-id` is provided.

For SWE-bench co-evolution, use the unified `run` command:

```bash
uv run medcoevo run \
  --swebench-instance django__django-11910 \
  --iterations 4 \
  --run-id swebench-django
```

Add a frozen eval phase:

```bash
uv run medcoevo run \
  --swebench-instance django__django-11910 \
  --swebench-eval-instance django__django-11099 \
  --run-id swebench-django
```

SWE-bench options for `run`:

| Option | Default | Meaning |
| --- | --- | --- |
| `--swebench-dataset-name` | `SWE-bench/SWE-bench_Lite` | Dataset name or local dataset path. |
| `--swebench-split` | `test` | Dataset split. |
| `--timeout` | `1800` | Per-instance test timeout in seconds. |
| `--max-workers` | `1` | Modal harness worker count. |

Standalone SWE-bench smoke outputs are written under:

```text
data/swebench-evals/<timestamp>-<run-id>/
```

## Outputs

Typical single-run output:

```text
data/experiments/<timestamp>-<suffix>/
|-- config.toml
|-- metrics.jsonl
|-- summary.json
|-- artifacts/
|   |-- reports/
|   |-- traces/
|   `-- validation/
|-- history/
|   |-- history.jsonl
|   `-- rejected_proposals.jsonl
|-- jobs/
|-- skills/
`-- skills_snapshots/
```

Important files:

- `config.toml`: resolved config after CLI overrides.
- `metrics.jsonl`: per-iteration records.
- `summary.json`: aggregate rewards, bootstrap confidence interval, token
  totals, per-task summaries, and environment failure count.
- `artifacts/traces/`: normalized task execution traces.
- `artifacts/reports/`: Mediator reports.
- `artifacts/validation/`: executor skill validation evidence when enabled.
- `history/history.jsonl`: feedback history entries used for later context and
  contrastive reflection.
- `history/rejected_proposals.jsonl`: rejected advisor batches or validation
  failures.
- `skills/`: run-local skill copy.
- `skills_snapshots/`: committed skill snapshots.

Matrix outputs use one subdirectory per preset:

```text
data/experiments/<timestamp>-42-baseline-matrix/
|-- no_feedback/
|-- full_trace_same_task/
|-- static_mediator_same_task/
|-- planner_only_skill_evolution/
|-- mediator_only_protocol_evolution/
`-- full_coevolution/
```

## Configuration

Default configuration lives in:

```text
config/default.toml
```

The current config controls:

- model IDs for Planner, Executor, Mediator, and Judge;
- prompt and completion token budgets;
- default iteration cadence;
- skill update and validation defaults;
- local output and benchmark paths;
- Harbor runtime settings;
- SkillsBench remote archive settings.

CLI options override the loaded config for a single run. The resolved config is
persisted in the experiment directory as `config.toml`.

Current config defaults include:

- `experiment.num_iterations = 30`
- `experiment.coevo_interval = 3`
- `experiment.advisor_buffer_max = 3`
- `experiment.seed = 42`
- `experiment.allow_cross_task_feedback = true`
- `experiment.skill_validation.enabled = true`
- `executor_runtime.agent_name = "hermes"`
- `executor_runtime.harbor_timeout_sec = 1800`

To use the agent configured (for Skillsbench tasks), pre-installation of respective tools, such as CLI, is required.

## Testing

Run the default unit suite:

```bash
uv run pytest
```

Run one test file:

```bash
uv run pytest tests/test_skillsbench.py
```

Run the opt-in Harbor integration test:

```bash
uv run pytest tests/test_skillsbench_integration.py -m integration -v -s
```

The project config excludes integration tests from the default pytest run.

## Troubleshooting

`OPENROUTER_API_KEY is required`

Export `OPENROUTER_API_KEY` before running experiments.

`harbor CLI not found on PATH`

Install Harbor with `uv tool install harbor`, then confirm `harbor --version`.
For orchestrator-only checks that should not call Harbor, set
`executor_runtime.harbor_required = false` in `config/default.toml`.

Docker or Compose failures in SkillsBench runs

Start Docker Desktop or Colima, then confirm:

```bash
docker --version
docker compose version
```

Missing SkillsBench task

Sync selected tasks with `skillsbench sync`, or keep
`executor_runtime.remote_fetch = true` so the runner can fetch missing tasks on
demand.

SWE-bench Modal credential failure

Run `modal token new` before SWE-bench commands.

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
- Multi-Agent Evolve (MAE): https://arxiv.org/abs/2510.23595
- OpenSpace: https://github.com/HKUDS/OpenSpace
- Group-Evolving Agents (GEA): https://arxiv.org/abs/2602.04837
- Self-Evolving Coordination Protocol (SECP): https://arxiv.org/abs/2602.02170
- Rubric as Reward: https://arxiv.org/pdf/2507.17746
- Skill Collective Evolution: https://github.com/AMAP-ML/SkillClaw
