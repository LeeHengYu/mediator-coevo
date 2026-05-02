## Context-Aware Mediation for Multi-Agent Skill Co-Evolution

### Problem

Current multi-agent systems either:

1. Evolve in isolation — single-agent self-improvement ignores cross-agent knowledge (OpenSpace, EvolveR, Self-Consolidation)
2. Share passively — dump everything into a shared repo or context, causing context bloat and attention degradation (Group Evolving Agents' `S = ∪ Tⱼ`, Spark: shared agentic memory, which is similar to naive note-taking for mutual use)
3. Require weight-level updates — expensive and only works with homogeneous agents sharing parameters (MAE's REINFORCE++ on shared θ)

### Core Idea

Introduce a **Mediator Agent** — a context-aware orchestrator whose primary function is not task routing but **knowledge routing** between heterogeneous agents. The Mediator decides _what_ knowledge flows between agents, _when_, _in what form_, and _at what context budget_. It co-evolves with the planner agent through periodical reflection.

### Advantages

- Concise reflection context via continuous Mediator LLM call to compact feedback to store refined history
- LLM choice can be cheap in skill training phase
  - Only the planner needs smart model (idea borrowed from Claude Code Advisor pattern of Sonnet running + Opus Advising); Mediator only handles text reports/log/compact reports, so can use less powerful models, such as `gemini-3-flash`.
  - Executor can be mediocre so it doesn't perform too well on the tasks, there can be more effective contrasive pairs.

### Example Architecture

```
User → Claude (plans) ──── task goal (unmodified) ───► Gemini (executes)
              ▲                                      │
              │                                      │  traces, errors, task score
              │  filtered reports                    │
              │                                      ▼
         ┌────────────────────────────────────────────────────────────┐
         │                       Mediator Agent                       │
         │                                                            │
         │  Observes Gemini's execution outputs.                      │
         │  Filters, compresses, and selects what to                  │
         │  expose to Claude.                                         │
         │  Does NOT modify tasks sent to Gemini.                     │
         │  Does NOT update skills directly.                          │
         │  Co-evolves its own mediation, and planning skill.         │
         └────────────────────────────────────────────────────────────┘
```

1. Planner: constructs each run's task plan from the benchmark instruction, active Executor skills, and condition-selected prior context. Prior context is selected by `--condition`:
   - `no_feedback`: no previous-run context.
   - `full_traces`: compact summaries of recent same-task traces (LLM-compacted when stderr is long).
   - `shared_notes`: configured shared notes from experiment config.
   - `static_mediator` / `learned_mediator`: the previous non-withheld Mediator report for the same task.
     Prior reports are task-keyed; cross-task context is opt-in via `experiment.allow_cross_task_feedback`.
     For meta-skill reflection, the Planner consumes `MediatorSignal` / `PlannerSignal` payloads from `HistoryEntry`.
2. Executor: containerized environment to run benchmark or well-defined tasks, output reward or score.
3. Mediator: only runs for `static_mediator` and `learned_mediator`; processes usable execution traces into curated reports for the Planner.
4. Planner + Mediator: Coevolve by querying contrastive pairs — pairs are formed by linking history entries (via stable entry IDs) to their delayed rewards from the next iteration of the same task.

## Current Implementation

- Planner grounds each run in a real local benchmark instruction instead of planning from a bare `task_id`.
- Executor runs a vendored local SkillsBench-style Harbor task and parses the resulting `reward`, verifier output, and agent logs into `ExecutionTrace`.
- The local benchmark task tree lives under `benchmarks/skillsbench/`. The initial migrated task is `benchmarks/skillsbench/tasks/fix-build-google-auto/`.
- Experiment conditions selectable via `--condition` (`no_feedback` | `full_traces` | `shared_notes` | `static_mediator` | `learned_mediator`).
- Skill update permissions are independently selectable via `--skill-updates` (`none` | `executor` | `planner` | `mediator` | `all`, comma-separated except for `none` and `all`).
- Previous-report state is task-keyed; cross-task feedback is opt-in via `experiment.allow_cross_task_feedback` (default `false`).
- Skill directories require a canonical `SKILL.md` entrypoint, validated at startup by `SkillStore.validate()`.
- History outcome tagging uses stable entry IDs so multi-task runs cannot attribute a reward to the wrong task's planner/mediator entry.

### Baseline Matrix

The baseline matrix separates two axes:

1. Feedback routing, controlled by `--condition` and responsible for Planner prior context plus Mediator calls.
2. Skill-update permission, controlled by `--skill-updates` or by a matrix preset and responsible only for whether committed skill edits are allowed.

Run one custom row:

```
uv run medcoevo run --condition learned_mediator --skill-updates executor,planner,mediator
```

Run all seven baseline rows with identical tasks, seed, model config, budgets, and isolated per-row skill copies:

```
uv run medcoevo matrix --tasks fix-build-google-auto --iterations 30 --seed 42
```

Matrix rows:

| Preset | Feedback condition | Skill updates |
| --- | --- | --- |
| `no_feedback` | `no_feedback` | `none` |
| `full_trace_same_task` | `full_traces` | `none` |
| `static_mediator_same_task` | `static_mediator` | `none` |
| `learned_mediator_same_task` | `learned_mediator` | `mediator` |
| `planner_only_skill_evolution` | `learned_mediator` | `planner` |
| `mediator_only_protocol_evolution` | `learned_mediator` | `mediator` |
| `full_coevolution` | `learned_mediator` | `executor,planner,mediator` |

The `matrix` command copies the initial configured `skills/` tree into each row's experiment directory before the row starts. Matrix rows therefore cannot write to repo-level skills or contaminate one another.

Assumptions encoded in the presets:

- `learned_mediator_same_task` uses learned Mediator reports and allows the Mediator skill to evolve; Planner and Executor updates stay disabled.
- `planner_only_skill_evolution` uses learned Mediator reports, but only the Planner meta-skill evolves.
- `mediator_only_protocol_evolution` disables Executor skill updates and allows only Mediator protocol evolution.
- `full_coevolution` is the only baseline row that permits Executor, Planner, and Mediator skill commits together.

Metrics persist `baseline_preset` when present and `skill_update_policy` for every row. The existing `skill_updates` metrics field remains the list of committed co-evolution skill updates.

## Testing

This project uses `uv` as the single test entrypoint. `uv` manages and reuses the local `.venv`, so tests should be run through `uv` instead of mixing direct `.venv/bin/python` and `uv run` commands.

Install/sync dependencies:

```
uv sync --dev
```

Run the default unit suite:

```
uv run pytest
```

Run the opt-in Harbor integration test:

```
uv run pytest tests/test_skillsbench_integration.py -m integration -v -s
```

The integration test uses Harbor's `opencode` agent with
`openrouter/google/<model>` by default. Source the shell
environment that exports `OPENROUTER_API_KEY` before running it, or override
with `MEDIATED_COEVO_INTEGRATION_AGENT` and `MEDIATED_COEVO_INTEGRATION_MODEL`.

### Two Distinct Skill Update Flows

**Flow 1 — Executor skill gating (count-triggered)**

Updates `skills/executor/SKILL.md` — what the Executor knows how to do.
Enabled only when `skill_updates.executor = true`.

```
Each iteration:
  Planner proposes a SkillProposal (based on Mediator feedback) → buffered

When buffer hits advisor_buffer_max (default 10):
  SkillAdvisor reviews the full batch → approve / reject
  Buffer is cleared regardless of outcome
  If approved: Planner drafts a new SkillUpdate (based on Advisor's aggregated feedback)
             → written to skills/executor/SKILL.md with AdvisorBatchProvenance
```

**Flow 2 — Agent meta-skill co-evolution (iteration-triggered)**

Updates `skills/mediator/SKILL.md` and `skills/planner/SKILL.md` — _how_ each agent behaves, not what the Executor executes.
Mediator reflection requires `skill_updates.mediator = true`; Planner reflection requires `skill_updates.planner = true`.

```
Every coevo_interval iterations (default 5):
  Reflector queries HistoryStore for contrastive pairs
  (pairs are formed from entries tagged with delayed rewards from the next iteration)

  Mediator reflection → rewrites skills/mediator/SKILL.md
    (coordination-protocol: how to curate and present feedback)
    → loaded into MediatorAgent immediately
    → recorded with ContrastiveReflectionProvenance

  Planner reflection → rewrites skills/planner/SKILL.md
    (skill-refiner: how to decide when and how to edit executor skills)
    → injected into Planner context at the next iteration start
    → recorded with ContrastiveReflectionProvenance
```

`SkillProposal` and `SkillUpdate` share a `SkillEdit` base (`old_content`, `new_content`, `reasoning`). Proposals are never written to `HistoryStore`; committed updates are serialized in `metrics.jsonl`. Executor updates use `IterationRecord.skill_update`; co-evolution checkpoints can record mediator/planner updates in `IterationRecord.skill_updates`. Provenance is intentionally concise and points back to proposal IDs, `HistoryStore` entry IDs, rewards, hashes, and skill snapshots instead of duplicating full evidence.

## Progress

P0, Week 1 scope + a bit of Week 2

## Further Direction

1. Overall flow of the information, who sees what?
2. Conditions that trigger co-evolution
3. Refine the task definition and output score, as some Skillsbench task outputs binary score so little variation can be used. The performace (hence the score) can be evaluate based on the output of the model, including clarity or formatting.

## Related Work & Papers

1. Anthropic: Claude API Advisor (beta): https://platform.claude.com/docs/en/agents-and-tools/tool-use/advisor-tool (Proof that a reference model is useful in guidance of a weaker model)
2. Spark — Shared Agentic Memory: https://arxiv.org/abs/2511.08301
3. Multi-Agent Evolve (MAE): https://arxiv.org/abs/2510.23595
4. OpenSpace: https://github.com/HKUDS/OpenSpace
5. Group-Evolving Agents (GEA): https://arxiv.org/abs/2602.04837
6. Self-Evolving Coordination Protocol (SECP): https://arxiv.org/abs/2602.02170
