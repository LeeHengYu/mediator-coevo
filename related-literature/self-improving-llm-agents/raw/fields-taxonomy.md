I now have thorough, cross-validated coverage from 8+ surveys. I have enough to synthesize a rigorous, survey-grounded field schema. Let me compile the final deliverable.

---

# Recommended Field Schema for ~45 Self-Improving-LLM-Agent Methods

I read 8 survey/taxonomy papers in depth. The consensus across the literature is that this field is best organized by a **closed-loop framework** (Agent System / Environment / Optimiser / Feedback) crossed with the **What–When–How–Where** axes (Gao et al.) and the **5-stage optimization-pipeline** axes (Ramnath et al., Cui et al., GFCR). My schema below restructures the original A-G categories to align tightly with how surveys actually slice this space, while keeping your provenance/evaluation pragmatics.

Detail tags: 极简 = one enum/short token; 简要 = short phrase / small set; 详细 = sentence(s)/structured.

---

### Category A. Identity & Provenance (极简, mostly)
| Field | Description | Detail |
|---|---|---|
| name / aliases | Canonical method name + aliases | 极简 |
| year, venue | Pub year; venue (arXiv/conf/journal) | 极简 |
| authors, institution | First author + lead lab/org | 极简 |
| paper_url, code_url | Links; code presence is an adoption signal | 极简 |
| cluster | Your grouping bucket (assigned) | 极简 |

### Category B. Problem & Scope — *"What is the agent / what gets improved"* (Gao §2-3; Fang 4-component)
| Field | Description | Detail |
|---|---|---|
| **optimization_target** | Which component is evolved — the single most-used axis ("What to evolve") | 极简 (enum) |
| **target_granularity** | For prompt targets: instruction-only / few-shot demos / full prompt (Cui §4; Ramnath §3) | 极简 |
| **system_scope** | single-agent node / single-agent holistic / multi-agent topology (Gao §3.4) | 极简 |
| **task_setting** | single-turn QA / multi-step agent / tool-use / web / GUI-OS / embodied / code-SWE / scientific | 简要 |
| **access_model** | white-box (weights) / black-box (API-only) / gray-box (Ramnath emphasizes black-box) | 极简 |
| **agent_formalism** | single-step MDP vs temporally-extended POMDP (Agentic-RL survey distinguishes these) | 极简 |

### Category C. Mechanism — *"How it evolves"* (Gao §5; Ramnath/Cui pipeline; GFCR)
| Field | Description | Detail |
|---|---|---|
| **paradigm** | Top-level method family (reward-based RL / imitation-demonstration / population-evolutionary / textual-gradient / reflection-verbal / search) — Gao's 3 families + APO operators | 极简 (enum) |
| **feedback_type** | scalar reward / natural-language / execution-result / self-critique / comparative-preference / fitness (Gao Table 3 "Feedback Type") | 极简 |
| **feedback_source** | internal (self) / external (env, rules, other agent) / human (Gao Table 3) | 极简 |
| **reward_granularity** | outcome / process(step) / hybrid (Gao §5.4.3 cross-cutting) | 极简 |
| **update_operator** | How candidates are generated: meta-prompt edit / textual-gradient / genetic / MCTS-beam-bandit / RL-gradient / code-rewrite (Cui §6 zero/single/multi-parent; Ramnath §5) | 简要 |
| **selection_strategy** | How candidates are kept/filtered: TopK-greedy / UCB-bandit / tournament-ELO / Pareto / archive-threshold (Ramnath §6; GFCR "Filter/Control") | 简要 |
| **updated_components** | full-params / partial-params / context-prompt / memory / tools / codebase / topology (Gao Table 3 "Updated Components") | 极简 |
| **optimizer_role** | Who/what drives optimization: same LLM self-referential / separate optimizer-LLM / meta-agent / evolutionary controller | 简要 |
| **credit_assignment** | How blame/credit is assigned across multi-step trajectory (Nie et al. "credit horizon"; Gao single-agent node problem) | 简要 |

### Category D. Memory & Skills (Du et al. memory survey; SoK Agentic Skills) — *split out because two whole surveys taxonomize only this*
| Field | Description | Detail |
|---|---|---|
| **memory_form** | none / parametric / contextual-structured / contextual-unstructured (Du et al. taxonomy) | 极简 |
| **memory_operations** | which of {consolidation, indexing, updating, forgetting, retrieval, compression} are used (Du et al. 6 ops) | 简要 |
| **memory_temporal_scope** | short-term/session vs long-term/persistent (Du 2026 3D taxonomy) | 极简 |
| **skill_representation** | NL / code / policy / hybrid (SoK Agentic Skills "representation × scope") | 极简 |
| **skill_lifecycle** | which of {discovery, distillation, storage, composition, reuse, update} (SoK skill lifecycle) | 简要 |

### Category E. Temporal & Learning Properties — *"When it evolves" + lifelong properties* (Gao §4, §5.4)
| Field | Description | Detail |
|---|---|---|
| **evolve_timing** | intra-test-time / inter-test-time / pre-test (training) (Gao §4; Table 3 "Update Timing") | 极简 |
| **learning_paradigm** | online / offline / both (Gao §5.4.1) | 极简 |
| **policy_consistency** | on-policy / off-policy / both (Gao §5.4.2) | 极简 |
| **lifelong_continual** | Is it explicitly designed for sequential-task continual learning? (Gao §2.2 vs lifelong learning) | 极简 |
| **self_referential** | Does it modify its own optimization process / code (RSI, Gödel/DGM)? | 极简 |
| **needs_labels_or_reward** | requires labels / requires reward / label-free-self-supervised (e.g., MM-UPT, EvolveSearch) | 极简 |
| **autonomy_level** | proto-evolution (feedback prompting) → strong-evolution (autonomous diagnosis+reconfig) (Gao operational def.; MSE 6-level frameworks) | 简要 |

### Category F. Evaluation (Gao §7 five-goal framework — the cleanest eval taxonomy in the field)
| Field | Description | Detail |
|---|---|---|
| **benchmarks** | BBH/GSM8K/MMLU/HotpotQA/ALFWorld/WebShop/WebArena/GAIA/SWE-bench/AgentBench/Minecraft/HumanEval etc. | 简要 |
| **eval_goals_covered** | which of {Adaptivity, Retention, Generalization, Efficiency, Safety, Self-directedness} (Gao §7.1) | 简要 |
| **eval_paradigm** | static / short-horizon-adaptive / long-horizon-lifelong (Gao §7.2) | 极简 |
| **headline_result** | Best reported gain vs baseline | 简要 |
| **baselines, ablations** | Key comparisons + which components ablated | 简要 |
| **efficiency_cost** | rollouts / LLM-calls / wall-time / $ (Gao §7.1.4; Nie et al. cost emphasis) | 简要 |

### Category G. Significance & Relations (详细 where needed)
| Field | Description | Detail |
|---|---|---|
| **key_innovation** | One-sentence novelty | 详细 |
| **builds_on / influenced_by** | Lineage (e.g., builds on Reflexion/STaR/Voyager/TextGrad/DGM) | 简要 |
| **limitations** | Stated failure modes | 简要 |
| **adoption** | open-source maturity, downstream use | 简要 |

### Category H. Uncertain / Deep-phase (skip if unknown)
| Field | Description | Detail |
|---|---|---|
| reproducibility | Reproduced? code runs? | 简要 |
| true_compute_cost | Real GPU/API cost if disclosed | 简要 |
| follow_ups | 2025-2026 successors | 简要 |
| safety_mechanisms | rollback / sandbox / dual-audit / human gate (Gao §8.3; autonomy-risk survey) | 简要 |

---

# New Fields to ADD (each tied to a survey's axis)

- **target_granularity** [B] — instruction-only vs instruction+few-shot vs full-prompt — *Cui et al. §4 "What is Optimized" and Ramnath §3 make this a first-class axis; your original folds it into optimization_target, but surveys separate them.*
- **selection_strategy** [C] — TopK/UCB/tournament/Pareto/archive-threshold — *Ramnath §6 "Filter and retain promising candidates" and GFCR "Filter/Control" treat candidate selection as a distinct pipeline stage separate from generation; your `update_operator` conflates them.*
- **reward_granularity** [C] — outcome/process/hybrid — *Gao §5.4.3 explicit cross-cutting axis; also central to GFCR and rollout surveys (process vs outcome supervision).*
- **feedback_source** [C] — internal/external/human — *Gao Table 3 separates "Feedback Type" from "Feedback Source"; your schema only has feedback_signal (=type).*
- **memory_operations** [D] — the 6 ops — *Du et al. 2025 entire taxonomy is built on Consolidation/Updating/Indexing/Forgetting/Retrieval/Compression; richer than a single `memory_structure` field.*
- **memory_form** [D] — parametric / contextual-structured / contextual-unstructured — *Du et al. representation axis.*
- **skill_lifecycle** [D] — discovery→distillation→storage→composition→reuse→update — *SoK Agentic Skills (2026) lifecycle taxonomy; complements your `skill_representation`.*
- **policy_consistency** [E] — on/off-policy — *Gao §5.4.2 explicit cross-cutting axis, absent from your D.*
- **learning_paradigm** [E] — online/offline — *Gao §5.4.1; distinct from evolve_timing.*
- **agent_formalism** [B] — single-step MDP vs POMDP — *Agentic-RL survey's central formal distinction; cheap to record and separates "reasoning-trick" methods from true sequential agents.*
- **autonomy_level** [E] — proto→strong evolution — *Gao operational definition gradient + the 6-level autonomy frameworks (agentic MSE, Cihon et al.); lets you rank "how self-evolving" a method really is.*
- **eval_goals_covered** [F] — Adaptivity/Retention/Generalization/Efficiency/Safety/Self-directedness — *Gao §7.1 five(+1)-goal framework; far more diagnostic than a flat benchmark list and exposes coverage gaps.*
- **eval_paradigm** [F] — static / short-horizon / long-horizon-lifelong — *Gao §7.2.*
- **credit_assignment** [C] — *Nie et al. 2026 show "credit horizon" is a hidden design choice that determines success; Gao flags it as the core single-agent-node difficulty.*
- **safety_mechanisms** [H] — rollback/sandbox/dual-audit/human-gate — *Gao §8.3 + autonomy-induced-security-risk survey; load-bearing for self-modifying methods (DGM, Gödel Agent).*

**Structural changes I recommend:** (1) Promote **Memory & Skills to its own category D** — two dedicated surveys taxonomize only this, so a single nested field undersells it. (2) Split your `paradigm` into `paradigm` (family) + `update_operator` + `selection_strategy` to match the generation/filtering pipeline split that 3 surveys use. (3) Merge your old "C credit-assignment / optimizer-LLM role / memory / skill" cluster: keep optimizer_role & credit_assignment in Mechanism, move memory/skill to D. (4) Replace flat `feedback_signal` with the **type × source × granularity** triple (Gao Table 3 + §5.4.3).

---

# Controlled Vocabularies (enum suggestions)

- **optimization_target** ∈ {weights, prompt-instruction, few-shot-demos, full-prompt, memory, skills/tools, single-agent-architecture, multi-agent-topology, codebase-self}
- **paradigm** ∈ {reward-RL, imitation-demonstration, population-evolutionary, textual-gradient, reflection-verbal, search(beam/MCTS/bandit/Bayesian), self-referential-code}
- **feedback_type** ∈ {scalar-reward, natural-language, execution-result, self-critique/confidence, comparative-preference, fitness-score}
- **feedback_source** ∈ {internal, external-env, external-rules, other-agent, human}
- **reward_granularity** ∈ {outcome, process, hybrid}
- **update_operator** ∈ {meta-prompt-edit, textual-gradient, genetic/crossover-mutation, MC-sampling, beam/bandit/MCTS, RL-gradient(PPO/GRPO/DPO), SFT, code-rewrite}
- **selection_strategy** ∈ {topK-greedy, UCB-bandit, region/joint-search, tournament/ELO, Pareto-multiobjective, archive-threshold, none}
- **updated_components** ∈ {full-params, partial-params, context-prompt, memory, tools, codebase, topology}
- **evolve_timing** ∈ {pre-test/training, intra-test-time, inter-test-time}
- **learning_paradigm** ∈ {online, offline, both}; **policy_consistency** ∈ {on-policy, off-policy, both}
- **system_scope** ∈ {single-agent-node, single-agent-holistic, multi-agent}
- **access_model** ∈ {white-box, black-box, gray-box}
- **agent_formalism** ∈ {single-step-MDP, POMDP-sequential}
- **memory_form** ∈ {none, parametric, contextual-structured, contextual-unstructured}
- **memory_operations** ⊆ {consolidation, indexing, updating, forgetting, retrieval, compression}
- **skill_representation** ∈ {none, NL, code, policy, hybrid}
- **skill_lifecycle** ⊆ {discovery, practice, distillation, storage, composition, evaluation, update}
- **eval_goals_covered** ⊆ {adaptivity, retention, generalization, efficiency, safety, self-directedness}
- **eval_paradigm** ∈ {static, short-horizon-adaptive, long-horizon-lifelong}
- **autonomy_level** ∈ {proto-evolution, partial, strong-evolution}
- **task_setting** ∈ {single-turn, multi-step-agent, tool-use, web, GUI-OS, embodied, code-SWE, scientific, dialogue}

---

# Survey Papers Used (with axes each contributes)

- [Gao et al. 2025, *A Survey of Self-Evolving Agents: What/When/How/Where*](https://arxiv.org/abs/2507.21046): **the spine of the schema** — axes = What-to-evolve {models, context(memory/prompt), tools, architecture(single/multi)}; When {intra- vs inter-test-time}; How {reward / imitation / population} + cross-cutting {online-offline, on/off-policy, reward granularity, feedback type/source, updated components, update timing, sample efficiency, stability, scalability}; Where {general vs specialized domain}; Eval {Adaptivity, Retention(FGT/BWT), Generalization, Efficiency, Safety, Self-directedness; static/short/long-horizon}.
- [Fang et al. 2025, *A Comprehensive Survey of Self-Evolving AI Agents*](https://arxiv.org/abs/2508.07407): axes = 4-component closed loop {System Inputs, Agent System, Environment, Optimisers}; single-/multi-/domain-specific optimization; "Three Laws" Endure/Excel/Evolve → safety, retention, autonomy framing.
- [Ramnath et al. 2025, *A Systematic Survey of Automatic Prompt Optimization*](https://arxiv.org/abs/2502.16923): axes = 5-stage pipeline {Seed prompts → Inference eval & feedback (numeric[accuracy/reward/entropy/NLL] / LLM / human) → Candidate generation (heuristic-edit / aux-NN[RL/finetune/GAN] / metaprompt / coverage / program-synthesis) → Filter&retain (TopK/UCB/region/ensemble) → Iteration depth (fixed/variable)}; black-box framing.
- [Cui et al. 2025, *APO via Heuristic Search: A Survey*](https://arxiv.org/abs/2502.18746): axes = Where (soft vs discrete prompt space); What is optimized (instruction-only / +example / +optional example); What criteria; Which operators (zero/single/multi-parent); Which iterative algorithm (bandit / beam / heuristic-sampling / MCTS / metaheuristic / iterative-refinement).
- [Du et al. 2025, *Rethinking Memory in LLM-based Agents*](https://arxiv.org/abs/2505.00675): axes = representation {parametric, contextual-structured, contextual-unstructured} × 6 operations {consolidation, updating, indexing, forgetting, retrieval, compression} (management vs utilization) × topics {long-term, long-context, parametric-modification, multi-source}.
- [Zhang et al. 2025, *The Landscape of Agentic RL for LLMs*](https://arxiv.org/abs/2509.02547): axes = MDP-vs-POMDP formalism; core capabilities {planning, tool-use, memory, reasoning, self-improvement, perception} × task domains; RL as the conversion mechanism.
- [Surana et al. 2026, *GFCR: Rollout Strategies for LLM RL*](https://arxiv.org/abs/2605.02913): axes = Generate-Filter-Control-Replay lifecycle + criteria {reliability, coverage, cost-sensitivity}; reinforces selection_strategy / credit_assignment / efficiency_cost fields and replay-for-self-improvement.
- [Nie et al. 2026, *Understanding Challenges in Iterative Generative Optimization*](https://arxiv.org/abs/2603.23994): axes = hidden design choices {starting artifact, credit horizon, batching of trials into learning evidence} — motivates credit_assignment + reproducibility fields; notes only ~9% of agents use any automated optimization.
- Supporting (skills): [SoK Agentic Skills 2026](https://arxiv.org/abs/2602.20867) + [Agent Skills survey 2026](https://arxiv.org/abs/2602.12430): skill lifecycle {discovery/distillation/storage/composition/reuse/update} and representation × scope {NL/code/policy/hybrid × web/OS/SWE/robotics}.

---

# 信息来源
- [Gao et al. — Survey of Self-Evolving Agents](https://arxiv.org/abs/2507.21046) (full text scraped, arXiv html v4)
- [Fang et al. — Comprehensive Survey of Self-Evolving AI Agents](https://arxiv.org/abs/2508.07407) ([HF page](https://huggingface.co/papers/2508.07407))
- [Ramnath et al. — Systematic Survey of APO](https://arxiv.org/abs/2502.16923) (full text fetched)
- [Cui et al. — APO via Heuristic Search](https://arxiv.org/abs/2502.18746) (full text scraped)
- [Du et al. — Rethinking Memory in LLM-based Agents](https://arxiv.org/abs/2505.00675) (full text scraped)
- [Zhang et al. — Landscape of Agentic RL for LLMs](https://arxiv.org/abs/2509.02547)
- [Surana et al. — GFCR rollout survey](https://arxiv.org/abs/2605.02913)
- [Nie et al. — Iterative Generative Optimization challenges](https://arxiv.org/abs/2603.23994)
- [Tao et al. — Survey on Self-Evolution of LLMs](https://arxiv.org/abs/2404.14387) (4-phase loop: acquisition/refinement/updating/evaluation)
- [Du 2026 — Memory for Autonomous LLM Agents](https://arxiv.org/abs/2603.07670) (write-manage-read loop; 3D taxonomy: temporal scope × substrate × control policy)
- [Emergent Mind — Self-Evolving AI Agents topic page](https://www.emergentmind.com/topics/self-evolving-ai-agent) (cross-survey synthesis + benchmark/result table)
- [SoK: Agentic Skills](https://arxiv.org/abs/2602.20867); [Agent Skills survey](https://arxiv.org/abs/2602.12430)

**Bottom line:** Your original A-G structure is sound. The highest-value changes are (1) carving Memory & Skills into a standalone category with the 6-operation + representation enums, (2) splitting feedback into type×source×granularity and operator into generate vs select stages, and (3) adopting Gao's six eval-goals as a coverage checklist rather than a flat benchmark list. These three changes alone make ~45 methods directly comparable along the exact axes the field's own surveys use.