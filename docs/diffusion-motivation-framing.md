# Diffusion-First Motivation for the Mediator-Coevo Report

## Purpose of this note

This note reframes the report around **experience diffusion** as the primary scientific object, while treating **skill updates** as an auxiliary downstream mechanism. This framing matches the current context-only experiments where executor, planner, and mediator skill updates are disabled, so observed performance changes should not be attributed to durable skill learning.

## Core motivation

LLM agents often execute tasks episodically: a useful failure mode, debug hint, or partial success from one task does not automatically inform later attempts on related tasks. The central motivation of this project is to test whether prior task experience can be preserved, selected, and routed into later planning contexts in a controlled way.

In this framing, diffusion is a mechanism for **cross-task contextual transfer**:

1. A source task runs and produces execution evidence.
2. The system emits compact artifacts from that evidence, such as run outcomes, debug hints, mediator summaries, or regression warnings.
3. A diffusion policy selects which prior artifacts are eligible for a later target task.
4. The selected artifacts are rendered into the Planner's context under provenance and token-budget constraints.
5. The target task is attempted with the same fixed skills, but with potentially more useful prior context.

The main question is therefore:

> Holding agent skills fixed, can graph-aware diffusion of prior task experience improve later agent performance compared with no diffusion, broadcast diffusion, or random diffusion?

## Why skill updates should be auxiliary in the current report

Skill updates remain important to the broader mediated co-evolution agenda, but they should not be the primary claim for the current report if the experiments disable skill updates.

When skill updates are disabled:

- the executor, planner, and mediator `SKILL.md` policies remain fixed during the run;
- any performance movement comes from task context, prior feedback, diffusion artifacts, stochasticity, or execution effects;
- the report can isolate whether routed cross-task context is useful before introducing durable policy rewriting.

This is a strength, not a weakness. It makes the current study a cleaner baseline for diffusion. Skill updates can then be positioned as a later **consolidation mechanism**:

> If diffusion repeatedly surfaces transferable lessons, future experiments can test whether those lessons should be distilled into persistent executor, planner, or mediator skills.

Thus, the report should treat skill updating as an auxiliary extension rather than the basis of the current empirical claim.

## Recommended thesis statement

A conservative report thesis could be:

> This report studies graph-aware experience diffusion for fixed-policy LLM agents. Rather than immediately rewriting agent skills, we first ask whether selected artifacts from prior task executions can improve the context available to later planning under causal, provenance, and token-budget constraints. Skill updates are treated as a downstream consolidation mechanism to be evaluated after the context-routing effect is established.

## Suggested introduction paragraph

Modern LLM agents can execute complex task benchmarks, but their experience is often episodic: a failure in one task does not automatically inform later attempts on related tasks unless the system explicitly preserves and routes that experience. This report studies **experience diffusion** as a mechanism for controlled cross-task transfer. Instead of immediately modifying agent skills, we first ask a narrower question: when agent policies are held fixed, can selected artifacts from prior task executions improve the context available to future planning? This framing isolates contextual transfer from durable skill learning and allows diffusion policies such as no diffusion, capped broadcast, random selection, and graph-aware top-k similarity to be compared under shared budget and provenance constraints.

## Suggested skill-update positioning paragraph

Skill updates remain an important long-term objective, but they are auxiliary to the present diffusion question. If diffusion surfaces reusable lessons across related tasks, those lessons may later be distilled into persistent executor, planner, or mediator policies. In the current context-only setting, disabling skill updates is a deliberate experimental control: it prevents performance changes from being attributed to policy rewriting and lets the report focus on whether cross-task context alone provides useful signal.

## Contribution framing

The report can frame its contributions as follows:

1. **Diffusion-first mediated co-evolution.**
   The report reframes mediated co-evolution around controlled movement of execution experience across tasks before durable skill rewriting.

2. **Fixed-skill context-transfer evaluation.**
   By disabling skill updates, the experiments isolate whether contextual diffusion affects task performance under fixed agent policies.

3. **Graph-aware artifact routing.**
   The system emits compact artifacts from prior task runs and routes them through policies such as no diffusion, capped broadcast, random-k, and top-k similarity.

4. **Validity-aware observability.**
   The infrastructure tracks source task, source iteration, target task, selected/rendered artifacts, token budgets, and leakage constraints.

5. **Path toward skill consolidation.**
   Skill updates are positioned as a later mechanism for converting repeatedly useful diffused lessons into durable policies.

## Safe claim ladder

### Safe current claims

The report can safely claim that:

- the system implements a diffusion mechanism for routing prior-task artifacts into later-task Planner context;
- skill updates can be disabled to isolate context-routing effects;
- diffusion can be evaluated as temporary contextual transfer rather than durable agent learning;
- the infrastructure records provenance, token budgets, source iterations, selected/rendered artifacts, and leakage checks;
- ongoing experiments should be interpreted descriptively until matched controls and sufficient seeds are complete.

### Tentative claims for ongoing experiments

The report may cautiously say that:

- top-k similarity diffusion may provide useful contextual signals;
- prior failure signatures or mediator summaries may help later tasks avoid related failure modes;
- graph-aware selection may be a better use of limited context budget than broadcast or random routing;
- current results are suggestive but not yet sufficient for a causal or general performance claim.

### Claims to avoid until stronger evidence is complete

The report should not yet claim that:

- diffusion generally improves LLM agents;
- graph-aware diffusion is statistically proven;
- agents have learned better durable skills in skill-update-disabled runs;
- observed gains are causal without matched off/random/broadcast/top-k controls;
- skill co-evolution has been empirically demonstrated by context-only runs.

## Recommended title options

- **Graph-Aware Experience Diffusion for Fixed-Policy LLM Agents**
- **Experience Diffusion Before Skill Evolution**
- **Contextual Transfer for Agentic Task Execution Under Fixed Skills**
- **From Episodic Failures to Cross-Task Context: A Diffusion-First Study of LLM Agents**
- **Controlled Cross-Task Experience Routing for LLM Agent Planning**

Recommended title:

> **Graph-Aware Experience Diffusion for Fixed-Policy LLM Agents**

## Suggested abstract framing

This report studies whether prior task experience can be routed as useful context for later LLM-agent planning. The system emits compact artifacts from completed task executions and selects prior cross-task artifacts through diffusion policies, including no diffusion, capped broadcast, random-k, and graph-aware top-k similarity. To isolate the effect of context routing, the current experiments hold executor, planner, and mediator skills fixed: no durable `SKILL.md` updates are committed during the run. This design separates temporary contextual transfer from persistent skill learning. We therefore interpret the current experiments as a diffusion-first, context-only evaluation. Skill updates remain part of the broader mediated co-evolution architecture, but in this report they are treated as a downstream consolidation mechanism to be evaluated after diffusion effects are established under matched controls, shared context budgets, and predeclared comparison rules.

## Recommended results-language template

When reporting ongoing results, use conservative language such as:

> Preliminary context-only runs suggest that diffusion can change the information available to later task attempts, and in some cases performance improves after prior artifacts become available. However, these results are descriptive rather than causal: matched no-diffusion, random-k, capped-broadcast, and top-k runs across multiple seeds are still needed before claiming that graph-aware diffusion improves agent performance.

Avoid wording such as:

> Diffusion improves agents.

Prefer:

> Diffusion may improve fixed-policy agent performance by routing relevant prior task artifacts into later planning contexts.

## Practical report structure

A diffusion-first report could use this structure:

1. **Problem:** LLM-agent experience is episodic and not automatically transferred across related tasks.
2. **Hypothesis:** Selective prior-task artifact diffusion can improve later planning under fixed skills.
3. **System:** Planner, Executor, Mediator, Judge, artifact emission, diffusion policy, renderer, audit log.
4. **Experimental control:** Skill updates disabled to isolate context transfer.
5. **Diffusion policies:** off, capped broadcast, random-k, top-k similarity.
6. **Validity controls:** source iteration ordering, same-task exclusion, token-budget tracking, provenance/citation logging.
7. **Preliminary results:** report only descriptive trends and operational checks while experiments continue.
8. **Future consolidation:** enable skill updates after diffusion effects are better established, testing whether transient lessons can become durable policies.

## One-sentence bottom line

The strongest current motivation is not that the agents already learn better skills, but that **controlled diffusion may let fixed-policy agents reuse prior task experience as context; skill updates are the later mechanism for consolidating those reusable lessons into durable behavior**.
