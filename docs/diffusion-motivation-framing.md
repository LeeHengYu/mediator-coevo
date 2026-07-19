# Experience-Diffusion Motivation

## Purpose

This note frames experience diffusion as the primary scientific object. The
Planner, Executor, and Mediator use fixed policies throughout every run, so the
experiment isolates temporary contextual transfer from policy differences.

## Core Motivation

LLM agents often execute tasks episodically: a useful failure signature, debug
hint, or partial success from one task does not automatically inform later
attempts on related tasks. This project tests whether prior task experience can
be preserved, selected, and routed into later planning contexts in a controlled
way.

Diffusion provides cross-task contextual transfer:

1. A source task produces execution evidence.
2. The runtime emits compact artifacts from that evidence.
3. A policy selects causally eligible artifacts for a later target task.
4. Selected artifacts enter the Planner context under provenance and token
   budgets.
5. The target task runs with the same fixed skills and potentially more useful
   context.

The central question is:

> Holding agent policies fixed, can graph-aware diffusion of prior task
> experience improve later performance compared with execution only,
> graph-constrained random selection, or learned selection without a graph?

## Recommended Thesis

> This report studies graph-aware experience diffusion for fixed-policy LLM
> agents. It asks whether selected artifacts from prior task executions improve
> later planning under causal, provenance, and token-budget constraints.

## Contribution Framing

1. **Fixed-policy context-transfer evaluation.** All arms use identical role
   skills, isolating routed context and execution effects.
2. **Graph-aware artifact routing.** The system can build a task graph and route
   prior artifacts through deterministic or learned policies.
3. **Causal sample construction.** A target position sees only artifacts from
   earlier positions; its own artifacts enter the bank after execution.
4. **Validity-aware observability.** Archives record source tasks, source
   positions, eligible/selected/rendered artifacts, budgets, and dropped input.
5. **Matched four-arm comparison.** Independent graph-agent and
   diffusion-agent flags create execution-only, graph-only, diffusion-only, and
   full-orchestration settings.

## Safe Claim Ladder

Safe current claims:

- the system routes prior-task artifacts into later Planner context;
- graph and learned-diffusion components can be evaluated independently;
- fixed skills make observed differences attributable to context, execution,
  and stochastic effects rather than policy changes;
- the archive records causal provenance and context-budget behavior;
- results should remain descriptive until matched arms and sufficient seeds are
  complete.

Tentative claims for ongoing experiments:

- graph-aware selection may provide useful contextual signals;
- prior failure signatures may help later tasks avoid related failure modes;
- selective routing may use a limited context budget better than broad routing.

Claims to avoid without stronger evidence:

- diffusion generally improves LLM agents;
- graph-aware diffusion is statistically established;
- observed gains are causal without matched four-arm controls;
- improvements from one family or seed generalize to other settings.

## Recommended Title

> **Graph-Aware Experience Diffusion for Fixed-Policy LLM Agents**

## Abstract Framing

This report studies whether prior task experience can be routed as useful
context for later LLM-agent planning. The system emits compact artifacts from
completed task executions and selects causal cross-task artifacts through four
settings: execution only, graph-constrained random selection, learned diffusion
without a graph, and graph-informed learned diffusion. Planner, Executor, and
Mediator skills remain fixed across every setting. The design therefore tests
context routing under matched policies, shared context budgets, and explicit
provenance constraints.

## Results-Language Template

Prefer conservative language:

> Preliminary fixed-policy runs show that diffusion changes the information
> available to later task attempts. Matched arms across multiple seeds are
> required before claiming a general performance benefit from graph-aware
> routing.

Avoid “diffusion improves agents.” Prefer:

> Diffusion may improve fixed-policy agent performance by routing relevant
> prior-task artifacts into later planning contexts.

## Suggested Report Structure

1. Problem: useful task experience is episodic.
2. Hypothesis: selected prior artifacts can improve later planning.
3. System: fixed agents, artifact emission, graph, policy, renderer, and audit.
4. Experimental arms: execution only, graph only, diffusion only, and both.
5. Validity controls: causal ordering, same-task exclusion, budgets, and
   provenance.
6. Results: matched reward, reliability, and cost comparisons across seeds.
7. Limitations: family specificity, model stochasticity, and executor variance.

## Bottom Line

Controlled experience diffusion may let fixed-policy agents reuse prior task
evidence as context; the experiment measures when that routing helps, harms, or
adds cost.
