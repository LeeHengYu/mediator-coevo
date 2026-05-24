# Mediated Co-Evolution

Mediated Co-Evolution is an experiment runner for studying how agent skills
change when execution feedback is routed through different context policies.
It supports SkillsBench tasks through Harbor, SWE-bench tasks through the
official SWE-bench/Modal harness, and mixed runs that include both task types.

## How It Works: GRPO-Like Skill Evolution

Mediated Co-Evolution studies skill files as runtime policies. It uses a
GRPO-like loop at the level of reward-relative skill editing: completed task
traces produce rewards, same-task history forms group-relative evidence, and LLM
reflection rewrites Markdown skills. It does not train model weights.

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

### Runtime Roles And Policies

The experiment loop has four roles:

- Planner: reads the benchmark instruction plus condition-selected prior
  context, then produces an execution plan.
- Executor: runs the task in the selected benchmark backend and returns a
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

Executor policy is exposed to benchmark backends through a shared runtime
envelope rather than as benchmark-specific domain knowledge:

```text
Task Instruction
Executor Policy
Task Resources
Verifier Contract
```

SkillsBench and SWE-bench use the same logical envelope. For SkillsBench, the
runner copies the original task directory, preserves any curated
`environment/skills/` entries, and rewrites the copied task `instruction.md` to
include the envelope. The curated SkillsBench skills remain task resources; the
evolved `executor` policy is not written as a competing
`environment/skills/executor/SKILL.md` skill. For SWE-bench, the same envelope is
included in the patch-generation prompt.

### Reward Sources And Tagging

The main online evolution reward is the Judge reward. The verifier reward is
kept as provenance and used as fallback when Judge scoring is unavailable.

After a usable task run finishes, the Judge-scored evolution reward is assigned
to pending same-task history entries from the prior iteration. This delayed
tagging matches the loop timing: a Mediator report, Planner edit, or buffered
Executor proposal produced at iteration `N` can only affect the downstream task
run at iteration `N + 1`.

History entries keep:

- `reward`: the evolution reward used for ranking and reflection.
- `metadata.verifier_reward`: the raw verifier score from the benchmark.
- `metadata.reward_source`: `judge`, `verifier_fallback`, or `verifier`.

### Group-Relative Contrastive Evidence

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
                  -> candidate skill rewrites -> skill commit gate
```

The reflection prompt shows each worse/better pair with its evolution reward and
task-relative delta. Mediator reflection currently commits after candidate audit
and similarity checking. Planner reflection asks for two candidate skill
rewrites, validates both empirically, and commits the accepted candidate with
the higher validation reward.

The validation task pool comes from `experiment.skill_validation`. Executor
validation compares the old and candidate Executor policies on the same selected
tasks under the same task instruction, task resources, and verifier contract.
It accepts only when the candidate improves by at least `min_mean_delta` without
violating configured regression or usability rules.

### What Is Not GRPO

The framework is GRPO-like only at the skill-editing layer. It does not:

- update LLM weights;
- compute token-level policy gradients;
- use a learned value model;
- treat one reward tag as causal proof.

The reward tags are noisy downstream labels. The intended signal comes from
repeated same-task, group-relative comparisons over many iterations.

## Quick Start

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
```

This selects one SkillsBench task, runs two iterations, disables committed skill
updates, and sets the advisor and reflection cadence to two iterations. Skill
validation gates are always required for skill candidates. The default condition is
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

Local SkillsBench runs require Harbor and a local container runtime:

```bash
uv tool install harbor
harbor --version
docker --version
docker compose version
```

To run Docker-heavy SkillsBench tasks on the configured GCP VM instead of the
local machine, use the `--cloud` flag. See
[Cloud VM Harbor Setup](#cloud-vm-harbor-setup).

SWE-bench runs use Modal instead of local Docker. Configure Modal before running
SWE-bench commands:

```bash
modal token new
```

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

## Cloud VM Harbor Setup

`medcoevo run --cloud` keeps the co-evolution control plane on the local
machine, but sends each prepared SkillsBench task workspace to an existing GCP
VM for `harbor run`. The VM is only a remote Docker/Harbor host: the full repo
is not copied to the VM, `medcoevo run` is not run on the VM, and experiment
outputs stay under local `data/experiments/`.

Current CLI shape:

```bash
uv run medcoevo run \
  --skillsbench-task dialogue-parser \
  --iterations 1 \
  --condition no_feedback \
  --skill-updates none \
  --cloud
```

Use `--cloud-env-file` when the GCP settings are not in `.env`:

```bash
uv run medcoevo run \
  --skillsbench-task dialogue-parser \
  --iterations 1 \
  --cloud \
  --cloud-env-file .env-another
```

### Local Dotenv Keys

The cloud path reads VM connection settings and the OpenRouter Secret Manager
resource from the dotenv file. The API key value itself should stay in Secret
Manager and is read by the VM service account at runtime.

Copy [.env.example](.env.example) to `.env` and fill in local-only secrets:

```bash
cp .env.example .env
```

`GCP_REGION` is optional when `GCP_ZONE` is set. `GCP_REMOTE_DIR` is optional
and defaults to `/tmp/mediator-coevo`; this project’s VM smoke run used
`~/mediator-coevo`. `GCP_SERVICE_ACCOUNT` is informational for this path; the
VM’s attached service account is what actually accesses Secret Manager.

The example file also lists GCS keys that are not used by direct VM Harbor mode
so they are not confused with the active `--cloud` path.

### Local Requirements

The local machine needs:

- `gcloud` installed and authenticated.
- permission to `gcloud compute ssh` and `gcloud compute scp` to the VM.
- `OPENROUTER_API_KEY` exported locally for planner, mediator, and judge model
  calls.

Local Harbor and local Docker are not required when `--cloud` is used for a
SkillsBench-only run. The CLI checks `gcloud` locally and skips the local Harbor
preflight.

### VM Requirements

The VM must have:

- Docker daemon running.
- Docker Compose v2 available as `docker compose`.
- `uv` on `PATH`.
- Harbor on `PATH`.
- `gcloud` on `PATH`.
- VM service account access to the OpenRouter secret stored in the secret manager within the same GCP project.
- an OAuth scope that permits Secret Manager access, such as `cloud-platform`.

The Debian 12 VM setup used for the smoke run was:

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose python3-venv pipx
sudo systemctl enable --now docker

mkdir -p "$HOME/.local/bin"
curl -LsSf https://astral.sh/uv/install.sh | sh
"$HOME/.local/bin/uv" tool install harbor

sudo ln -sf "$HOME/.local/bin/uv" /usr/local/bin/uv
sudo ln -sf "$HOME/.local/bin/harbor" /usr/local/bin/harbor

tmp="$(mktemp)"
curl -fL https://github.com/docker/compose/releases/latest/download/docker-compose-linux-x86_64 -o "$tmp"
sudo mkdir -p /usr/local/lib/docker/cli-plugins
sudo install -m 0755 "$tmp" /usr/local/lib/docker/cli-plugins/docker-compose
rm -f "$tmp"
```

Verify the VM runtime:

```bash
docker ps >/dev/null
docker --version
docker compose version
uv --version
harbor --version
```

Grant Secret Manager access to the VM service account:

```bash
gcloud secrets add-iam-policy-binding OPENROUTER_API_KEY \
  --project agent-coevolution \
  --member serviceAccount:vm-service-account@developer.gserviceaccount.com \
  --role roles/secretmanager.secretAccessor
```

If the VM was created with narrow OAuth scopes, update it to use
`cloud-platform` scope. This requires stopping the VM:

```bash
gcloud compute instances stop vm-instance-name \
  --project agent-coevolution \
  --zone us-central1-a

gcloud compute instances set-service-account vm-instance-name \
  --project agent-coevolution \
  --zone us-central1-a \
  --service-account vm-service-account@developer.gserviceaccount.com \
  --scopes cloud-platform

gcloud compute instances start vm-instance-name \
  --project agent-coevolution \
  --zone us-central1-a
```

After changing scopes, clear stale VM-side gcloud token cache and verify the VM
can read the secret without printing the secret value:

```bash
rm -f ~/.config/gcloud/access_tokens.db ~/.config/gcloud/credentials.db
gcloud secrets versions access latest \
  --secret OPENROUTER_API_KEY \
  --project agent-coevolution >/dev/null
echo secret-ok
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

| Option                                         | Default            | Meaning                                                                 |
| ---------------------------------------------- | ------------------ | ----------------------------------------------------------------------- |
| `--iterations`                                 | `30`               | Number of experiment iterations.                                        |
| `--seed`                                       | `42`               | Random seed.                                                            |
| `--condition`                                  | `learned_mediator` | Feedback routing condition.                                             |
| `--skill-updates`                              | `all`              | Which skill families may be committed.                                  |
| `--advisor-buffer-max`                         | config value       | Executor proposal batch size override.                                  |
| `--coevo-interval`                             | config value       | Planner/Mediator reflection interval override.                          |
| `--run-id`                                     | generated          | Timestamp-prefixed output directory suffix.                             |
| `--config-dir`                                 | `config/`          | Directory containing `default.toml`.                                    |
| `--cloud`                                      | false              | Run SkillsBench Harbor jobs on the configured GCP VM.                   |
| `--cloud-env-file`                             | `.env`             | Dotenv file containing GCP VM Harbor settings.                          |
| `--verbose`, `-v`                              | false              | Enable debug logging.                                                   |

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
`--config-dir`, and `--verbose`.

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

| Option                    | Default                    | Meaning                               |
| ------------------------- | -------------------------- | ------------------------------------- |
| `--swebench-dataset-name` | `SWE-bench/SWE-bench_Lite` | Dataset name or local dataset path.   |
| `--swebench-split`        | `test`                     | Dataset split.                        |
| `--timeout`               | `1800`                     | Per-instance test timeout in seconds. |
| `--max-workers`           | `1`                        | Modal harness worker count.           |

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
- `artifacts/validation/`: Executor skill and Planner reflection validation
  evidence when enabled.
- `history/history.jsonl`: feedback history entries used for later context and
  contrastive reflection.
- `history/rejected_proposals.jsonl`: rejected advisor batches or validation
  failures.
- `skills/`: run-local skill copy.
- `skills_snapshots/`: committed skill snapshots.

Executor policy observability fields in `metrics.jsonl`:

- `executor_policy_hash`: hash of the policy text injected for the run.
- `executor_policy_injected`: whether a non-empty policy was included.
- `executor_policy_injection`: where the policy was exposed, such as
  `instruction_envelope` for SkillsBench or `prompt_envelope` for SWE-bench.
- `task_resource_count` and `task_resource_names`: task-local resources exposed
  alongside the policy, such as curated SkillsBench skills.
- `verifier_contract_kind`: the benchmark verifier contract shown to the
  Executor.

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
- `executor_runtime.agent_name = "hermes"`
- `executor_runtime.harbor_timeout_sec = 7200`
- `executor_runtime.injected_skill_name = "executor"` names the Executor policy
  channel. The policy is rendered through the shared envelope; it is not
  automatically copied as a task-local domain skill for every benchmark.

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

## Progress

1. SkillsBench and SWE-bench datasets are integrated; SWE-bench evaluation runs
   on Modal.
2. The experimental LLM Judge layer runs after each task and scores traces,
   logs, verifier reward, and quality flags.

### Further Experiment 

1. Larger scale experiment (5+ tasks across two bench, 2hr+ time)
  - Lack of device resource (memory, storage, cloud computation for SWE-bench verifier)
2. GRPO framework application
3. LLM-as-judge design
  - Pass 2-3 traces, so the reward is relative
  - Few shot examples

## Related Work

- Claude API Advisor: https://platform.anthropic.com/docs/en/agents-and-tools/tool-use/advisor-tool
- Spark - Shared Agentic Memory: https://arxiv.org/abs/2511.08301
- Multi-Agent Evolve (MAE): https://arxiv.org/abs/2510.23595
- OpenSpace: https://github.com/HKUDS/OpenSpace
- Group-Evolving Agents (GEA): https://arxiv.org/abs/2602.04837
- Self-Evolving Coordination Protocol (SECP): https://arxiv.org/abs/2602.02170
- Rubric as Reward: https://arxiv.org/pdf/2507.17746
- Skill Collective Evolution: https://github.com/AMAP-ML/SkillClaw
- LLM-as-Judge guide (23 Jun): https://arxiv.org/pdf/2306.05685
- Textual Parameter Graph Optimization: https://arxiv.org/pdf/2604.20714
