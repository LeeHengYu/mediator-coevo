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

1. Planner: construct run task plan based on the curated logs released from mediator, building the prompt and send to executor. Planner has access to:
   1. the complete log of the previous run (ONLY the previous run) to construct the plan
   2. curated summaries (`MediatorSignal` / `PlannerSignal` payloads in `HistoryEntry`) of the logs to reflect on its own planning skill
2. Executor: containerized environment to run benchmark or well-defined tasks, output reward or score.
3. Mediator: Process the execution trace and produce a curated report for the Planner; may reference context from recent past runs.
4. Planner + Mediator: Coevolve by querying the `contrasive pairs`, a contrasive pair is by pairing good and bad performance and the improvement with harness diff can be used to reflect on skill improvement.

## Current Implementation

- Planner grounds each run in a real local benchmark instruction instead of planning from a bare `task_id`.
- Executor runs a vendored local SkillsBench-style Harbor task and parses the resulting `reward`, verifier output, and agent logs into `ExecutionTrace`.
- The local benchmark task tree lives under `benchmarks/skillsbench/`. The initial migrated task is `benchmarks/skillsbench/tasks/fix-build-google-auto/`.

### Two Distinct Skill Update Flows

**Flow 1 — Executor skill gating (count-triggered)**

Updates `skills/executor/SKILL.md` — what the Executor knows how to do.

```
Each iteration:
  Planner proposes a SkillProposal (based on Mediator feedback) → buffered

When buffer hits advisor_buffer_max (default 10):
  SkillAdvisor reviews the full batch → approve / reject
  Buffer is cleared regardless of outcome
  If approved: Planner drafts a new SkillUpdate (based on Advisor's aggregated feedback)
             → written to skills/executor/SKILL.md
```

**Flow 2 — Agent meta-skill co-evolution (iteration-triggered)**

Updates `skills/mediator/SKILL.md` and `skills/planner/SKILL.md` — _how_ each agent behaves, not what the Executor executes.

```
Every coevo_interval iterations (default 5):
  Reflector queries HistoryStore for contrastive pairs
  (pairs are formed from entries tagged with delayed rewards from the next iteration)

  Mediator reflection → rewrites skills/mediator/SKILL.md
    (coordination-protocol: how to curate and present feedback)
    → loaded into MediatorAgent immediately

  Planner reflection → rewrites skills/planner/SKILL.md
    (skill-refiner: how to decide when and how to edit executor skills)
    → injected into Planner context at the next iteration start
```

`SkillProposal` and `SkillUpdate` share a `SkillEdit` base (`old_content`, `new_content`, `reasoning`). Proposals are never written to `HistoryStore`; only committed `SkillUpdate`s appear in `IterationRecord` and `metrics.jsonl`.

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
