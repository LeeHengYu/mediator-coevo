# Diffusion-First Literature Positioning for `mediator-coevo`

Generated: 2026-06-08

Supporting literature reports: [`related-literature/`](related-literature/).
Conservative scratch note: [`docs/diffusion-motivation-framing.md`](docs/diffusion-motivation-framing.md) preserves the older, more conservative framing memo; treat it as historical working context rather than the current main positioning.

## Executive thesis

The strongest high-standard motivation for this project is:

> **`mediator-coevo` studies graph-aware experience diffusion for LLM agent co-evolution.** A mediator preserves, selects, routes, and renders prior execution evidence into later planning contexts under provenance, ordering, and token-budget constraints. The first-stage experimental control keeps the agent policy/skills fixed so the causal effect of diffusion can be isolated against matched no-diffusion, random, and broadcast controls. Skill updates are a downstream consolidation layer: once diffusion repeatedly identifies transferable evidence, later experiments can distill it into persistent prompts, playbooks, workflows, or executable skills.

This is an ambitious framing, not a conservative retreat. It makes diffusion the fast adaptation mechanism and skill updates the slow consolidation mechanism.

## Why this is a strong motivation

Modern LLM agents remain largely episodic. A failure signature, debug hint, workflow, or partial success from one task usually disappears unless a system explicitly stores and routes it. At the same time, dumping all prior context is not a principled solution: long-context and retrieval-noise studies show that extra context can become distraction, latency, cost, leakage, or positional degradation.

The project therefore targets a central open problem:

> **How should useful experience move between agents and tasks when the base LLM policy and durable skills are held fixed?**

That question is important because it isolates a mechanism that many literatures touch but rarely combine:

1. **No-weight-update agent adaptation:** agents can improve from feedback, demonstrations, workflows, or memory injected into context.
2. **Retrieval/ICL selection:** which example or experience enters context matters; random or broad inclusion is not equivalent.
3. **Graph/task similarity:** experience transfer is relational, directional, and failure-prone; graph structure can encode useful transfer paths and negative-transfer boundaries.
4. **Multi-agent mediation:** communication topology is an information bottleneck, not merely “more agents talking.”
5. **Long-context failure:** naive broadcast is a weak baseline because relevance, order, redundancy, and noise affect model behavior.
6. **Evaluation rigor:** agent benchmarks can be scaffold-, leakage-, reward-, and variance-sensitive; diffusion claims need predeclared controls.
7. **Skill consolidation:** durable skill updates remain valuable, but after diffusion is understood as a context-routing effect.

## Core literature map

| Evidence cluster | Representative sources verified or prioritized | What it supports | Boundary for this project |
|---|---|---|---|
| No-weight-update agent memory | Reflexion (arXiv:2303.11366), ExpeL (arXiv:2308.10144), Agent Workflow Memory (arXiv:2409.07429) | LLM agents can reuse feedback, memories, experiences, and workflows without updating model weights. | These are not all graph-aware and not all multi-agent; use them as adaptation mechanism anchors. |
| In-context retrieval and selection | KATE (arXiv:2101.06804), EPR (arXiv:2112.08633), ICL surveys | Selected examples outperform random examples in many settings. | Retrieval quality matters; bad similarity can hurt. |
| Task similarity and transfer graphs | Task2Vec (arXiv:1902.03545), Taskonomy (arXiv:1804.08328), TaskEmb (arXiv:2005.00770), TaskWeb/TaskShop (arXiv:2305.13256) | Transfer is structured and directional; prior-task usefulness can be modeled rather than broadcast. | Task similarity is a proxy, not proof of usefulness; validate with outcome feedback. |
| Agent trajectory/case reuse | Synapse (arXiv:2306.07863), CoPS (arXiv:2410.16670), case-based reasoning literature | Prior trajectories/cases can be reused for new tasks, and cross-task sharing is a legitimate object. | Must prevent task-answer leakage and distinguish reuse from memorization. |
| Graph memory/retrieval | HippoRAG (arXiv:2405.14831), GraphRAG, PathRAG, graph-based agent memory surveys, A-Mem/Mem0/MemGPT | Graph traversal/PPR/structured memory can support multi-hop, relational, or long-term retrieval better than flat buffers. | Some graph-memory work is recent/preprint-heavy; keep peer-reviewed/stable anchors central. |
| Multi-agent mediation/topology | Sparse MAD (EMNLP Findings 2024), AgentPrune, G-Designer, DyLAN, ReConcile, MAD critiques | Multi-agent communication should be sparse, selective, and topology-aware; dense communication can propagate error. | More agents or more messages are not automatically better. |
| Long-context and retrieval noise | Lost in the Middle (TACL 2024), Power of Noise, context-length degradation, irrelevant-input studies | Naive broadcast can degrade reasoning through position, redundancy, hard negatives, and context length. | Long-context effects are model/task-specific; use as motivation for controls, not universal law. |
| Evaluation rigor | AgentBench, SWE-bench/WebArena/OSWorld, Agentic Benchmark Checklist, rollout cards, error-bar/variance papers | Agent claims require scaffolding disclosure, leakage controls, multiple seeds, ablations, and cost/token accounting. | Current runs are ongoing; avoid final empirical claims until controls finish. |
| Skill/prompt consolidation | Voyager, PromptBreeder, TextGrad, DSPy/MIPRO, ACE/Dynamic Cheatsheet, Gödel/Darwin-Gödel lines | Repeated useful evidence can be compressed into durable skills, prompts, workflows, or code. | This is auxiliary in skill-update-disabled runs; do not attribute fixed-policy gains to durable skill learning. |

## Strong positioning statement

A strong paper/report claim should be framed as a **research target**:

> We propose a diffusion-first view of mediated co-evolution: agent experience should first move as bounded, provenance-tracked context across task-graph neighborhoods before being consolidated into durable skill updates. This lets us evaluate whether selective routed experience changes fixed-policy behavior, while avoiding the confound that performance gains came from rewritten skills.

This target is strong because it does not merely say “memory helps agents.” It asks a sharper systems question:

- What is the unit of experience: trace, failure, reflection, workflow, patch, test, or abstract lesson?
- What relation controls movement: semantic similarity, graph edge, task transfer estimate, temporal order, utility, or causal/failure relation?
- What is the diffusion policy: off, random, top-k, graph traversal, capped broadcast, or learned mediator?
- What is the render policy: raw trace, summary, workflow, warning, contrastive example, or anti-example?
- What is the estimand: selector quality over a frozen bank, or full endogenous system-policy improvement?
- When should transient diffusion become persistent skill consolidation?

## Recommended title / abstract / contributions

### Recommended title

**Graph-Aware Experience Diffusion for LLM Agent Co-Evolution**

### Abstract-style motivation paragraph

LLM agents can solve increasingly complex tasks, yet their experience is often episodic: a useful failure signature or workflow from one task rarely informs later attempts unless the system explicitly stores and routes it. This report studies graph-aware experience diffusion as a context-level mechanism for cross-task transfer. Completed task runs emit compact artifacts, and a mediator selects which prior artifacts to render into future planning contexts under provenance, source-ordering, and token-budget constraints. By holding executor, planner, and mediator skills fixed in the primary setting, the evaluation isolates temporary contextual transfer from durable policy rewriting. Skill updates remain part of the broader co-evolution agenda, but here they are treated as a downstream consolidation layer to test after diffusion effects are characterized against no-diffusion, random, and capped-broadcast controls.

### Contributions to claim now

1. **Diffusion-first mediated co-evolution.** Reframe co-evolution around controlled movement of execution evidence across a task graph before durable skill rewriting.
2. **Fixed-policy context-transfer design.** Disable skill updates to isolate whether routed prior artifacts change agent behavior through context alone.
3. **Graph-aware routing policies.** Compare `off`, `random_k`, `top_k_similarity`, and `capped_broadcast` under matched provenance and token-budget constraints.
4. **Validity-aware observability.** Log source task, source iteration, target task, selected artifacts, rendered artifacts, token cost, same-task/future leakage, and outcome deltas.
5. **Consolidation roadmap.** Treat skill updates as a second-stage mechanism for compressing repeatedly useful diffused evidence into persistent prompts, workflows, or skills.

## Claim ladder

### Safe now

- The project can be positioned as studying graph-aware experience diffusion for LLM agent co-evolution.
- Skill-update-disabled runs isolate context-routing effects better than runs that rewrite durable skills.
- The literature strongly motivates selective routing over naive broadcast.
- The local report corpus supports a cross-literature synthesis around memory, retrieval, task similarity, multi-agent topology, long-context failure, and benchmark rigor.

### Ambitious target claim

- Selective graph-aware diffusion should first be evaluated under a fixed-policy control by moving only relevant prior execution evidence into future contexts; broader co-evolution claims should then add skill-consolidation experiments.
- The high-standard target is to show that `top_k_similarity` or graph-aware routing beats matched `random_k` and `off` controls, and remains competitive with `capped_broadcast` at lower token cost and lower leakage/noise risk.

### Empirical claims to defer until runs finish

- “Graph-aware diffusion improves agents.”
- “Diffusion is statistically proven.”
- “Skill co-evolution has been demonstrated.”
- “The method generalizes beyond the benchmark/task family.”

## Evaluation contract

For a strong paper-quality claim, use the following contract.

1. **Predeclare the primary contrast.** Use `top_k_similarity` vs `random_k` as the clean selector-quality test.
2. **Include secondary controls.** `off` tests whether any diffusion helps; `capped_broadcast` tests whether selection beats broad sharing under a budget.
3. **Freeze artifact bank when testing selector quality.** If artifact histories are endogenous, label results as system-policy comparisons.
4. **Match token budgets and placement.** Token count, summary length, order, and high-attention placement can confound results.
5. **Separate source and target tasks.** Exclude future artifacts and same-instance leakage; record source iteration ordering.
6. **Run multiple seeds / repeats.** Report variance, confidence intervals, failures, timeouts, and cost.
7. **Run artifact-removal ablations.** Remove top artifact, warnings only, successes only, failures only, workflows only.
8. **Track negative transfer.** Some similar tasks hurt; graph edges should be updated by observed utility, not similarity alone.
9. **Disclose scaffold.** Keep base model, prompts, tools, retry policy, judge, and stopping rules fixed across policies.
10. **Add null and adversarial checks.** Use no-op/random agents, shuffled artifacts, future-leak canaries, and reward-hacking probes.

## How to position skill updates

Skill updates should be framed as **auxiliary consolidation**, not as a demotion.

- **Diffusion = fast path:** trial-level or task-level context transfer; reversible, auditable, and budgeted.
- **Consolidation = slow path:** repeated high-utility diffused evidence becomes a persistent skill, prompt, workflow, or code artifact.
- **Experimental order:** first prove diffusion under fixed skills; then enable skill updates to test whether consolidation improves persistence and cost efficiency.

Recommended language:

> Skill updates are the consolidation layer of mediated co-evolution. The present diffusion-first evaluation asks whether the system can identify and route useful experience while policies remain fixed. If a pattern repeatedly transfers across tasks, later consolidation experiments can compress it into durable skills.

Avoid:

> The agent learned new skills in skill-update-disabled runs.

## Literature-search expansion topics

Use these query families for continued research:

1. `LLM agent experience replay trajectory memory fixed policy no weight updates`
2. `retrieval augmented LLM agents prior trajectories cross task transfer`
3. `graph based agent memory task similarity experience retrieval`
4. `case based reasoning LLM agents trajectory retrieval`
5. `multi agent LLM communication topology information bottleneck sparse communication`
6. `long context LLM degradation retrieval noise lost in the middle hard negatives`
7. `agent benchmark rigor leakage scaffold variance ablation reproducibility`
8. `skill library LLM agents prompt optimization textual policy consolidation`
9. `workflow memory web agents cross domain task transfer`
10. `negative transfer task similarity LLM in context examples`

## Source verification status

The synthesis is grounded in:

- Local corpus audit: 10 reports, ~138,932 lines, 1,222 parsed source blocks.
- Six native subagents: memory/RAG, ICL/cross-task, reflection/self-improvement, graph/long-context, mediator/eval, global manifest verification.
- Firecrawl search and direct scrapes for core sources: Reflexion, ExpeL, AWM, HippoRAG, Lost in the Middle, KATE, Sparse MAD, Agentic Benchmark Checklist.
- Exa semantic searches/fetches across graph memory, multi-agent topology, evaluation rigor, self-improving agents, ICL/cross-task transfer, and long-context degradation.
- Firecrawl autonomous deep research, used mainly to identify frontier candidates; very new 2026 sources should be independently verified before becoming core citations.
- OpenRouter expert verdict saved separately in `.omx/artifacts/diffusion-positioning/expert-verdict.md` once complete.

## Bottom line

The project should not be sold as “skill updates make agents improve” or as an already-proven general claim that “diffusion improves agents.” The stronger and cleaner motivation is:

> **LLM agents need a principled way to move useful execution experience across related tasks. `mediator-coevo` treats that movement as graph-aware diffusion through context under strict controls, then treats skill updates as the later consolidation of repeatedly useful diffused evidence.**


## External expert verdict integration

OpenRouter GPT Pro (`openai/gpt-5.5-pro-20260423`, reasoning effort `high`) reviewed the compact evidence packet and returned the following bottom-line judgment:

- **Verdict:** approve the diffusion-first positioning as a strong, defensible research framing, provided it is presented as an empirical target rather than an already-established result.
- **Best framing:** graph-aware, provenance-tracked, token-bounded experience diffusion is the fast adaptation mechanism; skill updates are later consolidation.
- **Novelty scope:** the novelty is not “first memory” or “first multi-agent retrieval,” but the combination of graph-aware experience diffusion, mediator-controlled communication, broadcast/noise controls, a fixed-policy isolation phase, and delayed skill consolidation.
- **Key evaluation instruction:** distinguish selector quality over a frozen experience bank from endogenous system improvement as new artifacts are created.
- **Attribution rule:** in skill-update-disabled runs, gains must be attributed to contextual diffusion, not durable skill learning.

The full expert artifact is saved at `.omx/artifacts/diffusion-positioning/expert-verdict.md`.
