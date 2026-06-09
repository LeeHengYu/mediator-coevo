I now have thoroughly verified, comprehensive coverage of Cluster 2. I've conducted 18+ distinct searches/fetches across academic-search and Exa, cross-verifying every item against primary sources (arXiv IDs, citation counts, venues). Let me compile the final structured output.

## Cluster 2 — Prompt Evolution: Validation & Supplement

### Validation of existing 4 items (all confirmed correct, with corrected metadata)
- **EvoPrompt** (Guo et al., 2023) — arXiv 2309.08532, ICLR 2024, ~202 cites. Confirmed: GA + DE operators, 31 datasets, +25% on BBH. Note venue is ICLR 2024.
- **PromptBreeder** (Fernando et al., DeepMind, 2023) — arXiv 2309.16797, ICML 2024, ~480 cites. Confirmed self-referential (mutation-prompts evolve too). This is the canonical "self-referential prompt evolution" anchor.
- **GEPA** (Agrawal et al., 2025) — arXiv 2507.19457, ~201 cites. Confirmed Genetic-Pareto; beats GRPO by avg 6% (up to 20%) with up to 35x fewer rollouts; beats MIPROv2 by >10%.
- **EvoAgent** (Yuan et al., 2024) — arXiv 2406.14228, ~94 cites. Confirmed evolutionary extension of single→multi-agent.

Note: "Phaedrus" from the brief is NOT a prompt-evolution paper — arXiv 2412.06994 "Phaedrus" is a compiler/profiling deep-learning framework (program behavior prediction). Recommend dropping it from the watchlist. I could not find a distinct prompt-evolution "ECLIPSE"; the closest live items are below.

---

### 补充 Items (supplement — ranked by importance/relevance)

**Tier 1 — foundational / must-add (high citation, define the subfield)**

- **FunSearch** — "Mathematical discoveries from program search with LLMs" (Romera-Paredes et al., DeepMind, 2023). Nature 625, arXiv via Nature, **~953 cites**. https://www.nature.com/articles/s41586-023-06924-6 — The progenitor of LLM-as-mutation-operator + evaluator evolutionary search (cap-set, online bin-packing). Missing foundational anchor; nearly everything in this cluster (AlphaEvolve, EoH, ShinkaEvolve) descends from it.

- **AlphaEvolve** — "A coding agent for scientific and algorithmic discovery" (Novikov et al., Google DeepMind, 2025). arXiv 2506.13131, **~529 cites**. https://arxiv.org/abs/2506.13131 — The flagship "AlphaEvolve-style code/prompt evolution" item explicitly requested. Evolves whole code files via LLM ensemble + evaluators; found 48-multiplication 4×4 matrix mult (first improvement on Strassen in 56 yrs), optimized Google datacenter scheduling. This is the headline 2025 item and is currently absent.

- **EoH (Evolution of Heuristics)** — "Towards Efficient Automatic Algorithm Design Using LLM" (Liu et al., 2024). arXiv 2401.02051, ICML 2024, **~272 cites**. https://arxiv.org/abs/2401.02051 — Co-evolves natural-language "thoughts" + executable code; outperforms FunSearch at far lower query budget. Key bridge between prompt-level and code-level evolution.

- **ADAS / Meta Agent Search** — "Automated Design of Agentic Systems" (Hu, Lu, Clune, 2024). arXiv 2408.08435, **~207 cites**. https://arxiv.org/abs/2408.08435 — Directly fulfills "evolutionary multi-agent system design." Meta-agent programs ever-better agents (prompts+tools+workflows) in code against a growing archive; strong cross-domain/model transfer.

- **AFlow** — "Automating Agentic Workflow Generation" (Zhang et al., 2024). arXiv 2410.10762, ICLR 2025, **~250 cites**. https://arxiv.org/abs/2410.10762 — MCTS over code-represented workflows; complements ADAS as the search-based (vs evolutionary-archive) workflow optimizer; lets small models beat GPT-4o at 4.55% cost.

**Tier 2 — strongly recommended (recent, high-impact, fill specific sub-areas requested)**

- **Darwin Gödel Machine (DGM)** — "Open-Ended Evolution of Self-Improving Agents" (Zhang, Hu, Lu, Lange, Clune, 2025). arXiv 2505.22954, **~110 cites**. https://arxiv.org/abs/2505.22954 — Self-referential agent that rewrites its OWN code, validated empirically (resolves the unprovability of the original Gödel machine). SWE-bench 20→50%. The strongest "self-referential evolution" exemplar beyond PromptBreeder.

- **ShinkaEvolve** (Lange, Imajuku, Cetin; Sakana AI, 2025). arXiv 2509.19349, **~88 cites**. https://arxiv.org/abs/2509.19349 — Sample-efficient open-source AlphaEvolve successor (new circle-packing SOTA in 150 samples). Innovations: parent-sampling, novelty rejection-sampling, bandit LLM-ensemble selection. Anchors the sample-efficiency axis.

- **Rainbow Teaming** (Samvelyan et al., Meta, 2024). arXiv 2402.16822, NeurIPS 2024, **~196 cites**. https://arxiv.org/abs/2402.16822 — Casts adversarial prompt generation as a **quality-diversity (MAP-Elites)** problem. The canonical QD-prompt item explicitly requested; spawned RainbowPlus.

- **DEEVO / Tournament of Prompts** (Nair et al., 2025). arXiv 2506.00178, ~5 cites. https://arxiv.org/abs/2506.00178 — Debate-driven evolution with Elo-based fitness; notable for optimizing **without ground-truth/numeric fitness** (subjective/open-ended tasks). Distinct feedback-signal paradigm.

- **PromptWizard** (Agarwal et al., Microsoft Research, 2024). arXiv 2405.18369, MSR-TR. https://arxiv.org/abs/2405.18369 — Self-evolving feedback-driven critique+synthesis; jointly optimizes instructions AND in-context examples; large cost reduction across 45 tasks. Widely adopted open-source (microsoft/PromptWizard). Borderline C1/C2 but its self-evolving evolutionary framing fits C2.

- **PhaseEvo** (Cui et al., 2024). arXiv 2402.11347, ICML 2024 LCFM workshop. https://arxiv.org/abs/2402.11347 — Unified joint optimization of instruction + in-context examples via multi-phase global(evolutionary)/local schedule with LLM-based mutation operators; 35 tasks.

- **ReflectivePrompt** (Zhuravlev et al., 2025). arXiv 2508.18870, ~1 cite. https://arxiv.org/abs/2508.18870 — Adds short/long-term reflection ops before crossover/elitist mutation; +28% on BBH over EvoPrompt. Bridges C2 (evolution) and C4 (reflection).

**Tier 3 — emerging 2025–2026 (frontier, the requested "→2025" newest)**

- **SCOPE** — "Self-evolving Context Optimization via Prompt Evolution" (Pei et al., 2025). arXiv 2512.15374, ~11 cites. https://arxiv.org/abs/2512.15374 — Online prompt evolution from execution traces; dual-stream tactical/strategic memory; HLE 14.23→38.64%. Strong agent-deployment-time evolution exemplar.

- **E-SPL** — "Evolutionary System Prompt Learning for RL in LLMs" (Zhang, Chen, Stadie, 2026). arXiv 2602.14697. https://arxiv.org/abs/2602.14697 — **Jointly** evolves system prompts (mutation/crossover via self-reflection) AND updates weights via RL; declarative-in-prompt vs procedural-in-weights division. Bridges C2 and C6 (RL policy) — high conceptual value.

- **SePO** — "Self-Evolving Prompt Agent for System Prompt Optimization" (2026). arXiv 2606.04465. https://arxiv.org/abs/2606.04465 — Self-referential: the prompt agent optimizes its OWN system prompt alongside task prompts via open-ended archive search; generalizes across tasks. A direct modern PromptBreeder descendant.

- **EoH-S** — "Evolution of Heuristic Set" (Liu et al., 2025). arXiv 2508.03082, AAAI 2026, ~16 cites. https://arxiv.org/abs/2508.03082 — Evolves a complementary SET of heuristics (vs single) for generalization; quality-diversity flavor.

- **Diverse Prompts (MAP-Elites)** (Santos et al., 2025). arXiv 2504.14367, IEEE CEC 2025. https://arxiv.org/abs/2504.14367 — Explicit CFG-grammar + MAP-Elites illumination of prompt space (shots × reasoning-depth phenotypes). The clearest pure QD-prompt-illumination paper.

- **Swarm-Prompt / PSO** (Yu, 2025). IEEE CCAI 2025, DOI 10.1109/CCAI65422.2025.11189827. — First PSO (swarm-intelligence) discrete prompt optimizer; extends EvoPrompt's GA/DE setup. Useful to cover the PSO branch explicitly requested.

- **GAAPO** — "Genetic Algorithm Applied to Prompt Optimization" (Sécheresse et al., 2025). arXiv 2504.07157, Frontiers in AI, ~15 cites. — Hybrid GA integrating multiple specialized generation strategies (not just mutation/crossover); ETHOS/MMLU-Pro/GPQA.

- **CodeEvolve** (Assumpção et al., 2025). arXiv 2510.14150, ~14 cites — open-source AlphaEvolve w/ CVT-MAP-Elites + island model; matches AlphaEvolve on 5/9. Good open-source reproducibility data point alongside ShinkaEvolve/OpenEvolve.

**Survey to add (lens/taxonomy)**
- **"Evolutionary Computation and Large Language Models: A Survey"** (Chauhan et al., 2025). arXiv 2505.15741. https://arxiv.org/abs/2505.15741 — Bidirectional EC↔LLM taxonomy (EC-for-prompt/NAS/HPO vs LLM-for-EC operators); the natural Cluster-2 survey anchor.

---

### 推荐补充字段 (new fields specific to Prompt Evolution)

- **evolutionary_operators**: Which EA operators are used and how the LLM implements them — mutation / crossover / selection / elitism, plus variant (GA / Differential Evolution / PSO / genetic programming). Core taxonomic axis distinguishing EvoPrompt (GA+DE) vs Swarm-Prompt (PSO) vs grammar-guided GP.

- **llm_operator_role**: How the LLM is wired into the evolutionary loop — as mutation/crossover operator, as fitness evaluator/judge, as selection/pairing chooser (PAIR), or as meta-controller of EA hyperparameters (LAPC). Distinguishes "LLM-as-operator" from "LLM-as-optimizer."

- **search_unit / genotype**: What evolves — task instruction prompt / few-shot exemplars (jointly?) / system prompt / mutation-prompt (self-referential) / whole code file / agentic workflow graph / multi-agent topology. Cleanly separates prompt-only from code/agent evolution.

- **self_referential_depth**: Does it evolve only the artifact, or also the operators/its-own-improver (PromptBreeder mutation-prompts, DGM self-code-rewrite, SePO self-prompt)? Key conceptual divide for "self-improving."

- **population_diversity_mechanism**: How diversity is maintained — quality-diversity/MAP-Elites archive, island model, Pareto frontier (GEPA), Elo/tournament, novelty rejection-sampling. Critical to avoid premature convergence; differentiates QD methods.

- **fitness_signal_type**: Scalar metric on dev set / ground-truth-free (Elo, debate, LLM-judge) / multi-objective (Pareto, NSGA-II) / execution-validated (code correctness). Maps to framework field C.feedback_signal but with evolution-specific granularity.

- **rollout_sample_efficiency**: Quantified search cost — #LLM calls / #candidates / #generations to reach result (e.g., ShinkaEvolve 150 samples; GEPA 35x fewer rollouts than GRPO; PhaseEvo ~4000 API calls). The central practical axis for this cluster.

- **online_vs_offline_evolution**: Evolved once offline against a train set (EvoPrompt, PromptWizard) vs continuously at deployment from live execution traces (SCOPE). Determines lifelong-learning applicability.

- **reward_hacking / evaluator_robustness**: Whether the work addresses prompts/programs gaming the evaluator (noted by AlphaEvolve harness work, ArchAgent "simulator escapes"). Increasingly important safety dimension as model capability scales.

---

### 信息来源
- [FunSearch (Nature 2023)](https://www.nature.com/articles/s41586-023-06924-6)
- [AlphaEvolve (arXiv 2506.13131)](https://arxiv.org/abs/2506.13131)
- [EvoPrompt (arXiv 2309.08532)](https://arxiv.org/abs/2309.08532)
- [PromptBreeder (arXiv 2309.16797)](https://arxiv.org/abs/2309.16797)
- [GEPA (arXiv 2507.19457)](https://arxiv.org/abs/2507.19457)
- [EvoAgent (arXiv 2406.14228)](https://arxiv.org/abs/2406.14228)
- [EoH (arXiv 2401.02051)](https://arxiv.org/abs/2401.02051) / [EoH-S (arXiv 2508.03082)](https://arxiv.org/abs/2508.03082)
- [ADAS / Meta Agent Search (arXiv 2408.08435)](https://arxiv.org/abs/2408.08435)
- [AFlow (arXiv 2410.10762)](https://arxiv.org/abs/2410.10762)
- [Darwin Gödel Machine (arXiv 2505.22954)](https://arxiv.org/abs/2505.22954)
- [ShinkaEvolve (arXiv 2509.19349)](https://arxiv.org/abs/2509.19349)
- [Rainbow Teaming (arXiv 2402.16822)](https://arxiv.org/abs/2402.16822) / [RainbowPlus (arXiv 2504.15047)](https://arxiv.org/abs/2504.15047)
- [DEEVO/Tournament of Prompts (arXiv 2506.00178)](https://arxiv.org/abs/2506.00178)
- [PromptWizard (arXiv 2405.18369)](https://arxiv.org/abs/2405.18369) / [MSR page](https://www.microsoft.com/en-us/research/publication/promptwizard-task-aware-agent-driven-prompt-optimization-framework/)
- [PhaseEvo (arXiv 2402.11347)](https://arxiv.org/abs/2402.11347)
- [ReflectivePrompt (arXiv 2508.18870)](https://arxiv.org/abs/2508.18870)
- [SCOPE (arXiv 2512.15374)](https://arxiv.org/abs/2512.15374)
- [E-SPL (arXiv 2602.14697)](https://arxiv.org/abs/2602.14697)
- [SePO (arXiv 2606.04465)](https://arxiv.org/html/2606.04465v1)
- [Diverse Prompts MAP-Elites (arXiv 2504.14367)](https://arxiv.org/abs/2504.14367)
- [Swarm-Prompt PSO (IEEE CCAI 2025, DOI 10.1109/CCAI65422.2025.11189827)](https://doi.org/10.1109/CCAI65422.2025.11189827)
- [GAAPO (arXiv 2504.07157)](https://arxiv.org/abs/2504.07157)
- [CodeEvolve (arXiv 2510.14150)](https://arxiv.org/abs/2510.14150)
- [EC + LLM Survey (arXiv 2505.15741)](https://arxiv.org/abs/2505.15741)

---

### Summary report

**Validation:** All 4 existing Cluster-2 items (EvoPrompt, PromptBreeder, GEPA, EvoAgent) verified correct against primary sources; only metadata refinements suggested (EvoPrompt → ICLR 2024; cite counts attached). The brief's "Phaedrus" hint is a false lead (it's a compiler paper, not prompt evolution) — recommend removing. No distinct prompt-evolution "ECLIPSE" exists in the live literature; closest analogues are listed.

**Key gaps found:** The cluster is missing its three most important anchors — **FunSearch** (the progenitor, ~953 cites), **AlphaEvolve** (~529 cites, the explicitly-requested flagship), and **EoH** (~272 cites) — plus the **ADAS/AFlow** evolutionary-agent-design line (~207/250 cites) and **Rainbow Teaming** (the canonical QD/MAP-Elites prompt item, ~196 cites). These five are high-confidence additions.

**Supplemented:** 20 items across 3 tiers spanning the requested sub-areas: GA/DE/PSO search (EvoPrompt, Swarm-Prompt, GAAPO), self-referential evolution (PromptBreeder, DGM, SePO), quality-diversity/MAP-Elites (Rainbow Teaming, RainbowPlus, Diverse Prompts, CodeEvolve), LLM-as-mutation-operator (FunSearch, AlphaEvolve, ShinkaEvolve, EoH), evolutionary multi-agent design (EvoAgent, ADAS, AFlow), AlphaEvolve-style code evolution (AlphaEvolve, ShinkaEvolve, CodeEvolve), reflective/Pareto evolution (GEPA, ReflectivePrompt, DEEVO), and 2025-26 emerging online/RL-hybrid evolution (SCOPE, E-SPL, SePO). Added 1 survey anchor.

**New fields:** 9 prompt-evolution-specific dimensions proposed (evolutionary_operators, llm_operator_role, search_unit/genotype, self_referential_depth, population_diversity_mechanism, fitness_signal_type, rollout_sample_efficiency, online_vs_offline_evolution, reward_hacking/evaluator_robustness) — chosen to discriminate the items the existing A–G framework collapses together (e.g., it cannot currently separate GA vs PSO vs QD, or prompt-only vs whole-code/agent evolution, or sample-efficiency which is the cluster's headline competitive axis).

All claims were cross-verified against ≥1 primary source (arXiv/Nature/venue) with live citation counts via Semantic Scholar; 18+ distinct tool calls were used across academic-search and Exa.

<!-- ===== SECONDARY BLOCK ===== -->

Research complete. Summary of what was delivered for Cluster 2 — Prompt Evolution:

**Validation:** Confirmed all 4 existing items (EvoPrompt, PromptBreeder, GEPA, EvoAgent) against primary sources. Flagged that "Phaedrus" from the brief is a false lead (it is a compiler/profiling paper, not prompt evolution), and that no distinct prompt-evolution "ECLIPSE" exists in the live literature.

**Key gaps identified:** The three most-cited foundational anchors were missing — FunSearch (~953 cites), AlphaEvolve (~529 cites, the requested flagship), and EoH (~272 cites) — plus the ADAS/AFlow evolutionary-agent-design line and Rainbow Teaming (canonical QD/MAP-Elites prompt item).

**Supplemented:** 20 items across 3 tiers, covering every requested sub-area (GA/DE/PSO search, self-referential evolution, quality-diversity/MAP-Elites, LLM-as-mutation-operator, evolutionary multi-agent design, AlphaEvolve-style code evolution, reflective/Pareto evolution, and 2025-26 emerging online/RL-hybrid methods), plus 1 survey anchor.

**New fields:** 9 prompt-evolution-specific dimensions proposed to discriminate items the current A–G framework collapses together (notably evolutionary_operators, llm_operator_role, search_unit/genotype, self_referential_depth, population_diversity_mechanism, rollout_sample_efficiency, and reward_hacking/evaluator_robustness).

All claims cross-verified against arXiv/Nature/venue primary sources with live Semantic Scholar citation counts. The full structured output (补充Items / 推荐补充字段 / 信息来源 with URLs) is in my previous message.