# Mediated Experience Diffusion

`medcoevo` is a fixed-skill experiment runner for studying whether artifacts
from earlier SkillFlow tasks improve later task execution through causal context
diffusion.

Planner, Executor, and Mediator skills are immutable runtime inputs. Their
complete `SKILL.md` files are injected directly into the relevant prompts and
are never rewritten by an experiment. Task execution is local through Harbor;
the CLI has no remote-VM execution mode.

## Architecture

```text
frozen task sequence
        |
        v
Planner -> Executor -> execution trace -> Mediator -> compact artifacts
                     \-> Judge reward          |
                                                 v
                                  causal artifact bank
                                                 |
                               graph/policy selection
                                                 |
                                                 v
                                later Planner context
```

The runtime separates four responsibilities:

- The task-graph agent describes causal task relationships.
- The diffusion-policy agent selects prior artifacts for a target task.
- The context packer renders selected artifacts within configured budgets.
- The execution agent receives one frozen task and one explicit context pack.

Package guides document the public boundaries:

- [diffusion](src/mediated_coevo/diffusion/README.md)
- [orchestration](src/mediated_coevo/orchestration/README.md)
- [execution](src/mediated_coevo/execution/README.md)
- [artifacts](src/mediated_coevo/artifacts/README.md)
- [experiment](src/mediated_coevo/experiment/README.md)

## Fixed Skills

The three role policies live at:

- `skills/planner/SKILL.md`
- `skills/executor/SKILL.md`
- `skills/mediator/SKILL.md`

The Planner receives the full Planner skill as its system policy and the full
Executor skill as read-only capability context. The Executor receives the full
Executor skill in its task envelope. The Mediator receives the full Mediator
skill as its system policy. Runs record deterministic skill hashes for
provenance, but the runtime exposes no skill-writing API.

## Diffusion Policies

The general experiment runtime supports:

- `none`: no artifact diffusion.
- `capped_broadcast`: most recent eligible cross-task artifacts, capped by
  `diffusion.max_artifacts`.
- `random_k`: a deterministic seeded sample of eligible artifacts, capped by
  `diffusion.max_artifacts`.
- `top_k_similarity`: artifacts from the strongest incoming task-graph
  neighbors.
- `langchain_graph`: learned task-graph and diffusion agents inspect the causal
  artifact store and choose artifacts.

Eligible artifacts must come from an earlier sequence position and a different
task. Rendered context shares `budgets.max_transfer_context_tokens`; oversized
artifacts are compacted and, if they still do not fit, dropped and recorded in
the audit ledger.

## Requirements

- Python `>=3.13`
- `uv`
- Harbor CLI on `PATH`
- Docker for local Harbor execution
- `OPENROUTER_API_KEY` for configured OpenRouter-backed agents

Install dependencies and inspect the CLI:

```bash
uv sync --dev
uv run medcoevo --help
```

Build the local Harbor base image once:

```bash
uv run medcoevo build-base-image
```

## Run An Experiment

Run the local smoke task:

```bash
uv run medcoevo run \
  --family smoke \
  --iterations 1 \
  --condition no_feedback \
  --run-id smoke
```

Run graph-aware diffusion:

```bash
uv run medcoevo run \
  --family Weighted-Risk-Assessment \
  --iterations 3 \
  --condition learned_mediator \
  --diffusion-enabled \
  --diffusion-policy top_k_similarity \
  --diffusion-graph task_similarity \
  --run-id wra-top-k
```

Experiment output is written under:

```text
data/experiments/<timestamp>-<run-id>/
```

The resolved config and fixed-skill hashes are persisted with each run.

## Four-Row Matrix

`matrix` compares the four fixed-skill diffusion policies using one shared task
stream:

```bash
uv run medcoevo matrix --list
uv run medcoevo matrix \
  --family Compensation-Scenario-Modeling \
  --iterations 3 \
  --seed 42 \
  --index 0 \
  --run-id csm-matrix
```

The rows are `diffusion_none`, `capped_broadcast`, `random_k`, and
`top_k_similarity`.

## Repeated Sequences

Sequence defaults live in `config/default.toml`:

```toml
[sequence]
length = 10
warmup = 3
```

Generate any missing per-task base artifact stores, then run repeated seeded
sequences:

```bash
uv run medcoevo base-artifacts --family <family>
uv run medcoevo sequence \
  --family <family> \
  --seed 0 -K 10 -n 10 --warmup 3
```

- Every sequence stays within one task family. Task IDs are repeated as needed
  with balanced multiplicities; the preloaded warmup prefix remains distinct.
- `-K` is the repeat count.
- `-n/--length` is the total tasks per sequence.
- `--warmup` is the number of warmup tasks.
- CLI task-count flags override `[sequence]`.
- Loop `i` uses `seed + i` for both the task stream and policy seed.

The graph and diffusion flags are independent booleans, both defaulting to
false:

```bash
# execution only
uv run medcoevo sequence ... --seed 0 -K 10 -n 10 --warmup 3

# graph only
uv run medcoevo sequence ... --seed 0 -K 10 -n 10 --warmup 3 --graph-agent

# learned diffusion without graph
uv run medcoevo sequence ... --seed 0 -K 10 -n 10 --warmup 3 --diffusion-agent

# graph plus learned diffusion
uv run medcoevo sequence ... --seed 0 -K 10 -n 10 --warmup 3 \
  --graph-agent --diffusion-agent
```

Graph-only mode follows the graph and uses deterministic random selection,
capped by `diffusion.random_policy_max_artifacts` (default `2`). Learned
diffusion selects artifacts itself, capped by `diffusion.max_artifacts`
(default `3`); when a graph is present it is supplied as selection evidence.

All four settings start from the same arm-neutral warmup bank. Each suffix bank
then evolves independently because routed context can change execution and
future artifacts.

## Harness Overlays

A sequence harness is a cumulative sparse code overlay. It must replace at
least one direct-agent file:

- `src/mediated_coevo/diffusion/task_graph_agent.py`
- `src/mediated_coevo/diffusion/policy_agent.py`

Publish an existing campaign update and use it later:

```bash
uv run medcoevo publish-harness \
  --campaign <campaign> \
  --harness-dir data/experiments/<campaign>/update_XXXX \
  --source-sequence data/sequences/<sequence-run>

uv run medcoevo sequence ... --harness-ref promoted:<campaign>
```

`promoted:<campaign>` resolves the latest registered update;
`promoted:<campaign>@update_XXXX` pins one version. The overlay is applied only
for the command process and restored afterward.

## Artifact Stores

Save and reuse a diffusion store:

```bash
uv run medcoevo matrix --family Weighted-Risk-Assessment \
  --iterations 1 --index 1 --save --run-id wra-warmup

uv run medcoevo matrix --family Weighted-Risk-Assessment \
  --iterations 3 --index 1 \
  --artifact data/artifact-stores/<experiment-folder> \
  --freeze --run-id wra-frozen
```

## Configuration

Default configuration lives in `config/default.toml`. It controls:

- Planner, Executor, Mediator, and Judge model IDs;
- prompt, completion, and context token budgets;
- experiment condition, iterations, and seed;
- sequence length and warmup count;
- diffusion emission, selection, and graph routing;
- local output, benchmark, Harbor, and SkillFlow settings.

CLI options override config for one invocation. The resolved configuration is
persisted as `config.toml` in the run directory.

## Testing

```bash
PYTHONDONTWRITEBYTECODE=1 UV_NO_CACHE=1 uv run --no-cache pytest -p no:cacheprovider
uv run ruff check .
uv run mypy src tests
```

## Troubleshooting

`OPENROUTER_API_KEY is required`

Export `OPENROUTER_API_KEY` before running an experiment.

`harbor CLI not found on PATH`

Install Harbor and confirm `harbor --version`. For tests that must not invoke
Harbor, set `executor_runtime.harbor_required = false` in the test config.

Docker failures

Start Docker Desktop or another local Docker daemon and confirm `docker info`.

Missing SkillFlow task

Use `uv run medcoevo list`, fetch tasks with
`uv run medcoevo sync --tasks ...`, select an existing family, or add a task
under `benchmarks/skillflow/tasks/<Family>/`.

Missing SkillFlow base image

Run `uv run medcoevo build-base-image`. If a task declares an optional
`[environment].docker_image`, either provide that image or remove the stale
field so Harbor builds from the task Dockerfile.
