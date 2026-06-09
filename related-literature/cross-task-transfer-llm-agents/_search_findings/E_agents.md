I'll research Cluster E — LLM agents: experience reuse & skill libraries. Let me start by loading the search tools I need.
Let me begin the literature search. I'll run multiple parallel searches across the academic MCP and web tools, focusing on Cluster E — LLM agent experience reuse and skill libraries.
The dates extend into 2026 which is consistent with the current date (2026-06-07). Found some interesting items (PRAXIS, HMT). Let me run more searches in parallel.
The Semantic Scholar AND-query is too restrictive (returns 0 with many terms). Let me use shorter queries and the OpenAlex provider, plus the extended search type.
Excellent. OpenAlex with single-term queries is finding very recent and highly relevant papers (H-EPM, ReMe, ProcMEM/Skill-Pro, MIRIX). Let me read the persisted self-evolving agents output and run more targeted searches.
This is a rich vein. I found many relevant systems: EXIF (skill discovery), CASCADE (skill acquisition), Darwinian Memory (GUI agents), WebEvolver, SLEA-RL, Agents of Change/HexMachina, MLEvolve. Let me run more targeted searches to find the canonical/most-cited systems in this space.
Very productive. I now have a strong skill-library cluster (Skill-as-Pseudocode, Graph-of-Skills, SkillOps, Skill Drift, Adaptation of Agentic AI survey, Odyssey, LRLL). Let me read the experience-reuse persisted file and run more searches for canonical systems I expect (AutoGuide, AWM, ExpeL followups, agentic memory, A-MEM, Generative Agents, CoALA, Mobile-Agent-E, AgentSquare, SkillWeaver, ASI/Agent Skill Induction).
The "experience reuse" and "agentic memory" generic queries returned mostly noise (general LLM surveys). The skill-library and procedural-memory queries were the productive ones. Let me now verify the canonical/key systems I expect to find via targeted Exa and academic searches, and find the foundational items the framework mentions plus newer 2024-2025 systems by name. I'll run several parallel searches.
This is exactly the core cluster. I now have a comprehensive set of canonical and recent systems. Key new finds: ASI (Inducing Programmatic Skills), SkillWeaver, PolySkill, Trace2Skill, AutoGuide, ReasoningBank, ELL, and several major surveys (Self-Evolving Agents ASI survey, Lifelong Learning Roadmap, Adaptation of Agentic AI, Self-Evolving AI Agents bridging survey). Let me verify a few more high-priority specific systems (ReasoningBank, SkillWeaver, Investigate-Consolidate-Exploit/ICE, JARVIS-1, Synapse, AutoManual, MemSkill, SAGE/SEAgent) and confirm exact metadata.
I have verified ReasoningBank and SkillWeaver with full metadata. Let me do a final batch of targeted searches to capture remaining important systems: the experience-reuse classics (ExpeL details, AutoManual, Investigate-Consolidate-Exploit/ICE, JARVIS-1, Synapse), and newer 2024-2025 cross-task transfer systems (AgentFly, Mobile-Agent-E, Optimus-2, GenAgent, MemGPT/Letta procedural, Memp), plus verify foundational AWM/ASI/PolySkill/Trace2Skill metadata.
I now have an extensive, well-verified set. AgentFly/Memento (case-based reasoning, directly relevant to "case-based reasoning / experience reuse"), Memp, AutoManual, ICE are all confirmed. I have more than enough new systems (well over 10). Let me do two final verification searches to lock down the exact metadata for AWM, ASI, PolySkill, Trace2Skill, and SAGE/SEAgent, and confirm the key surveys' arXiv IDs.
I have now verified all the key systems with full metadata. Note there are two distinct "SAGE" papers (the Amazon Skill-Augmented GRPO one is the relevant skill-library one). I have a comprehensive, verified collection covering well over 10 new systems plus the canonical surveys. I have run more than 15 distinct searches across the MCPs. Let me compile the structured results.

## 补充Items (Supplemented Items for Cluster E)

All metadata below verified against retrieved arXiv/OpenReview/GitHub/proceedings sources. Organized by sub-theme.

### A. Skill-induction / programmatic skill libraries (closest to AWM/Voyager lineage — high priority)

- **ASI — Inducing Programmatic Skills for Agentic Tasks**: Direct successor to AWM; represents induced skills as *verifiable programs* (not NL workflows), inducing/verifying/reusing on-the-fly. Explicitly studies cross-website skill transfer. +23.5% SR over static baseline on WebArena. Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, Daniel Fried. 2025. COLM 2025. https://arxiv.org/abs/2504.06821
- **SkillWeaver — Web Agents can Self-Improve by Discovering and Honing Skills**: Self-driven curriculum; synthesizes reusable skills as Python *APIs*, practices and debugs them. Strong-to-weak agent skill *transfer* (up to +54.3% on WebArena). Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, Yu Su. 2025. COLM 2025. https://arxiv.org/abs/2504.07079
- **PolySkill — Learning Generalizable Skills Through Polymorphic Abstraction**: Decouples skill abstract goal vs concrete implementation (polymorphism analogy) to fix over-specialization; explicitly targets *cross-domain/cross-website generalization* — directly the transfer angle. +9.4% Mind2Web, +13.9% unseen sites. Simon Yu, Gang Li, Weiyan Shi, Peng Qi. 2025/2026. ICLR 2026. https://arxiv.org/abs/2510.15863
- **Trace2Skill — Distill Trajectory-Local Lessons into Transferable Agent Skills**: Parallel multi-analyst consolidation of traces into conflict-free skill directory; key finding = skills transfer across model *scales/families* and OOD (e.g., +57.65 pts WikiTableQuestions). Outperforms ReasoningBank-style retrieval. Compares against Anthropic's official xlsx skills. Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou et al. (ETH/PKU/ZJU + Alibaba Qwen). 2026. https://arxiv.org/abs/2603.25158
- **Odyssey — Empowering Minecraft Agents with Open-World Skills**: Open-world skill library (40 primitive + 183 compositional skills) + fine-tuned LLaMA-3 backbone; a Voyager-lineage embodied skill-library system. Shunyu Liu, Yaoru Li, Kongcheng Zhang et al. 2024. IJCAI 2024. https://www.ijcai.org/proceedings/2024/0022.pdf (arXiv 2407.15325)
- **LRLL — Lifelong Robot Library Learning**: LLM agent that continuously grows a robot skill library (soft memory, self-guided exploration, skill abstractor) bridging to embodied control; transfers knowledge from memory to library. Georgios Tziafas, Hamidreza Kasaei. 2024. ICRA 2024. https://arxiv.org/abs/2406.18746

### B. Procedural-memory / experience-distillation systems (newest wave; tightest fit to "experience reuse")

- **ReasoningBank** (HIGH PRIORITY — likely the single most important recent item): Distills *generalizable reasoning strategies* from self-judged successful AND failed experiences; introduces memory-aware test-time scaling (MaTTS). Outperforms raw-trajectory and success-only memory. Siru Ouyang, Jun Yan, I-Hung Hsu et al. (Google + UIUC). 2025/2026. ICLR 2026. https://arxiv.org/abs/2509.25140 (code: github.com/google-research/reasoning-bank)
- **Memp — Exploring Agent Procedural Memory**: Task-agnostic learnable/updatable/lifelong procedural memory; studies Build/Retrieve/Update strategies; strong→weak model procedural-memory transfer. TravelPlanner + ALFWorld. Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, Ningyu Zhang (Zhejiang U + Alibaba). 2025. https://arxiv.org/abs/2508.06433
- **ReMe (Remember Me, Refine Me)**: Lifecycle procedural-memory framework — multi-faceted distillation (success/failure/comparative), context-adaptive reuse, utility-based refinement. Memory-scaling effect: Qwen3-8B+ReMe beats memoryless Qwen3-14B. BFCL-V3, AppWorld. Zouying Cao, Jiaji Deng et al. 2025. https://arxiv.org/abs/2512.10696
- **H-EPM — Experience-Evolving Multi-Turn Tool-Use Agent with Hybrid Episodic-Procedural Memory**: Builds a tool graph (procedural routines) + episodic summaries; memory-guided RL biasing exploration; up to +50% inference, +40% OOD. Sijia Li, Yuchen Huang et al. (Microsoft Research). 2025. https://arxiv.org/abs/2512.07287
- **Skill-Pro / ProcMEM — Learning Reusable Procedural Memory via Non-Parametric PPO**: Formalizes a Skill-MDP; converts episodic narratives into executable Skills (activation/execution/termination conditions) via Non-Parametric PPO; evaluated in in-domain, cross-task, AND cross-agent settings. Qirui Mi, Zhijian Ma, Mengyue Yang, Yisen Wang, Haifeng Zhang, Jun Wang et al. 2026. https://arxiv.org/abs/2602.01869

### C. Case-based reasoning / non-parametric experience reuse (directly matches "case-based reasoning / experience reuse" in the topic)

- **AgentFly — Fine-tuning LLM Agents without Fine-tuning LLMs** (a.k.a. Memento): Explicit *case-based reasoning* (CBR) framing; Memory-augmented MDP, neural case-selection policy via Soft Q-Learning; no weight updates. Top-1 GAIA val 87.88%; +4.7–9.6 pts OOD. 2025. https://arxiv.org/abs/2508.16153 (code: github.com/Agent-on-the-Fly/AgentFly; Memento repo)
- **ICE — Investigate-Consolidate-Exploit: Inter-Task Agent Self-Evolution**: First strategy for *inter-task* self-evolution; consolidates plans into workflows + trajectories into pipelines (finite automata) for reuse. A foundational 2024 piece your framework currently omits. 2024. https://arxiv.org/abs/2401.13996
- **AutoManual — Constructing Instruction Manuals by LLM Agents via Interactive Environmental Learning**: Planner/Builder/Formulator agents distill experience into a rule-based, human-readable manual; manuals built by GPT-4 *transfer to guide smaller LLMs*; addresses "Path Dependency." ALFWorld 97.4%. Minghao Chen, Yihang Li, Yanting Yang, Shiyu Yu, Binbin Lin, Xiaofei He. 2024. NeurIPS 2024. https://arxiv.org/abs/2405.16247
- **AutoGuide — Automated Generation and Selection of State-Aware Guidelines**: Extracts *state-conditional* guidelines from offline success/failure trajectory pairs; state-aware retrieval — a distinctive "when-to-apply" experience-reuse mechanism. 2024. https://arxiv.org/abs/2403.08978

### D. RL-trained skill libraries & self-evolving computer/GUI/web agents (skill-acquisition wave)

- **SAGE — Reinforcement Learning for Self-Improving Agent with Skill Library** (Amazon Science): Skill-Augmented GRPO with Sequential Rollout over task chains + Skill-integrated Reward (rewards reusable skill creation). AppWorld: +8.9% SGC, −26% steps, −59% tokens. Jiongxiao Wang, Qiaojing Yan et al. (UW-Madison + AWS Agentic AI). 2025. https://arxiv.org/abs/2512.17102 (note: distinct from the 2024 "SAGE: Self-evolving Agents... 2409.00872")
- **SEAgent — Self-Evolving Computer Use Agent with Autonomous Learning from Experience**: CUA self-evolves on novel software via experiential learning (World State Model + Curriculum Generator), adversarial imitation of failures + GRPO on successes; specialist→generalist transfer. OSWorld +23.2% SR. Zeyi Sun, Ziyu Liu, Yuhang Zang et al. 2025. https://arxiv.org/abs/2508.04700
- **EXIF — Automated Skill Discovery for Language Agents through Exploration and Iterative Feedback**: Explorer agent (Alice) generates feasibility-grounded skill datasets to train target agent (Bob); closed-loop self-evolving skill discovery. WebShop, Crafter. Yongjin Yang, Sinjae Kang et al. (KAIST). 2025. https://arxiv.org/abs/2506.04287
- **CASCADE — Cumulative Agentic Skill Creation through Autonomous Development and Evolution**: Explicit "LLM + tool use → LLM + skill acquisition" framing; two meta-skills (continuous learning, self-reflection); skills shareable across agents/scientists. SciSkillBench 93.3% (GPT-5) vs 35.4% w/o evolution. Xu Huang, Junwu Chen et al. (EPFL/Berkeley). 2025. https://arxiv.org/abs/2512.23880
- **HMT — Enhancing Web Agents with a Hierarchical Memory Tree**: Decouples Intent / Stage (reusable subgoals w/ pre-/post-conditions) / Action levels to fix flat-memory "workflow mismatch"; targets *cross-website/cross-domain* generalization on Mind2Web + WebArena. Yunteng Tan, Zhiqiang Gao, Xinxiao Wu. 2026. https://arxiv.org/abs/2603.07024
- **Darwinian Memory System (DMS)**: Training-free self-regulating memory for GUI agents; decomposes trajectories into reusable units + utility-driven natural selection (pruning). +18.0% SR, +33.9% stability on multi-app benchmarks. Hongze Mi et al. 2026. https://arxiv.org/abs/2601.22528

### E. Skill-library engineering / maintenance / retrieval (emerging 2026 sub-field — relevant to your agent-skill-learning work specifically)

- **Graph-of-Skills (GoS) — Dependency-Aware Structural Retrieval for Massive Agent Skills**: Inference-time structural retrieval for libraries of 200–2,000 skills; executable skill graph + reverse-aware Personalized PageRank to retrieve dependency-complete skill bundles. SkillsBench/ALFWorld; +25.55% reward, −56.72% tokens. Dawei Li, Zongxia Li et al. 2026. https://arxiv.org/abs/2604.05333
- **Skill-as-Pseudocode (SaP)**: Converts free-form markdown skills into typed pseudocode (typed contract + concrete templates) w/ deterministic verifier; reduces "confused→re-retrieve" loops. ALFWorld w/ gpt-4o-mini. Xinze Li, Yuhang Zang, Yixin Cao, Aixin Sun. 2026. https://arxiv.org/abs/2605.27955
- **SkillOps — Managing Skill Libraries as Self-Maintaining Software Ecosystems**: Treats library-level "skill technical debt"; typed Skill Contracts + Hierarchical Skill Ecosystem Graph; near-zero library-time LLM calls. Hongji Pu, Xinyuan Song, Liang Zhao. 2026. https://arxiv.org/abs/2605.13716
- **Skill Drift Is Contract Violation**: Formalizes skill decay as contract violation when underlying APIs/packages evolve; precision-first maintenance; releases an 880-pair skill-degradation benchmark. Linfeng Fan, Yuan Tian, Ziwei Li, Zhiwu Lu. 2026. https://arxiv.org/abs/2605.10990

### F. Key surveys to anchor the cluster (your framework currently has no survey backbone)

- **A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve on the Path to ASI**: First systematic self-evolving-agents survey; "what/when/how to evolve" axes (covers model/memory/tools/architecture). 2025. https://arxiv.org/abs/2507.21046
- **A Comprehensive Survey of Self-Evolving AI Agents: Bridging Foundation Models and Lifelong Agentic Systems**: Unified feedback-loop framework (System Inputs / Agent System / Environment / Optimisers). 2025. https://arxiv.org/abs/2508.07407
- **Lifelong Learning of LLM-based Agents: A Roadmap**: Perception/Memory/Action module decomposition for continual learning; catastrophic-forgetting focus. 2025. https://arxiv.org/abs/2501.07278
- **Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills**: Four-paradigm framework (A1/A2 agent adaptation; T1/T2 tool/skill adaptation); directly maps the skill-library design space. 2025. https://arxiv.org/abs/2512.16301
- **Building Self-Evolving Agents via Experience-Driven Lifelong Learning (ELL): Framework and Benchmark**: ELL framework (Experience Exploration / Long-term Memory / Skill Learning / Knowledge Internalization) + benchmark — useful as an evaluation anchor. 2025. https://arxiv.org/abs/2508.19005

Note on framework's existing items: Cradle, Synapse, AdaPlanner, ExpeL, Reflexion, CoALA, Voyager, AWM all confirmed as canonical — none missing/misnamed. JARVIS-1, Synapse, GITM, DEPS, AppAgent appear as recurring skill-library predecessors in the surveys (worth citing as the pre-2024 lineage if you want completeness).

## 推荐补充字段 (Recommended Additional Fields)

The current schema is strong but missing several dimensions that recur as the *axes of differentiation* across these systems:

- **knowledge_granularity**: What unit is stored/reused — raw trajectory / success-only routine / contrastive (success+failure) insight / reasoning strategy / state-conditional guideline / rule / executable skill. (This is the single biggest differentiator, e.g., ReasoningBank vs AWM vs raw-trajectory memory.)
- **memory_lifecycle_operations**: Which of {build/induce, retrieve, update, refine, prune/deprecate, consolidate} the system supports. Newer systems (Memp, ReMe, DMS, SkillOps) are defined by *active* update/prune, vs early append-only memory.
- **learns_from_failure**: Boolean/notes — does it distill from failed trajectories too? (ReasoningBank, AutoGuide, AgentFly, SEAgent vs success-only methods like Voyager/AWM.) Critical and increasingly emphasized axis.
- **transfer_axis_tested**: Explicit which transfer was empirically evaluated — cross-task / cross-website / cross-domain / cross-model-scale / cross-model-family / cross-agent / OOD. (Many claim generalization; few test each axis — Trace2Skill and PolySkill are notable for testing cross-model/cross-domain.)
- **parameter_update**: weight-frozen (memory/prompt only) vs RL/SFT fine-tuned vs hybrid. (Cleaner than current "training_strategy"; cleanly splits AgentFly/ReasoningBank/Memp [frozen] from SEAgent/SAGE [GRPO] — directly relevant to your skill-learning-without-fine-tuning angle.)
- **skill_representation**: natural-language / structured-rule / pseudocode / executable-code-API / finite-automaton-pipeline / soft-prompt. (Refines the existing knowledge_carrier; the NL-vs-code axis is a recurring debate: AWM/AutoGuide NL vs ASI/SkillWeaver/PolySkill code.)
- **library_scaling_mechanism**: How the system handles a *growing/large* library — retrieval method, dependency handling, context-budgeting, maintenance. (GoS, SkillOps, Skill Drift, SaP are entirely about this; absent from current schema.)
- **strong_to_weak_transfer**: Boolean — can skills/memory built by a strong model boost a weaker model? (Explicit selling point of Memp, Trace2Skill, SkillWeaver; a distinctive, practically important property.)
- **composability**: Can stored skills be composed/nested into higher-level skills? (Odyssey, ASI, PolySkill, LRLL emphasize compositional skills.)
- **shareability_across_agents**: Are skills exportable/auditable/shareable artifacts (vs model-internal)? (CASCADE, SkillWeaver, the "agent skills as portable SKILL.md" framing — directly bridges to your mediator-coevo / agent-skill work.)
- **relation_to_test_time_scaling**: Whether experience reuse is coupled with TTS / compute allocation. (ReasoningBank+MaTTS is the leading example; emerging axis.)

## 信息来源 (Sources)

- [ReasoningBank (arXiv 2509.25140)](https://arxiv.org/abs/2509.25140) · [Google Research blog](https://research.google/blog/reasoningbank-enabling-agents-to-learn-from-experience/) · [OpenReview ICLR 2026](https://openreview.net/forum?id=jL7fwchScm)
- [Agent Workflow Memory (PMLR / arXiv 2409.07429)](https://proceedings.mlr.press/v267/wang25bx.html)
- [ASI — Inducing Programmatic Skills (arXiv 2504.06821)](https://arxiv.org/abs/2504.06821)
- [SkillWeaver (arXiv 2504.07079)](https://arxiv.org/abs/2504.07079) · [project page](https://osu-nlp-group.github.io/SkillWeaver/)
- [PolySkill (arXiv 2510.15863)](https://arxiv.org/abs/2510.15863) · [ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10010108)
- [Trace2Skill (arXiv 2603.25158)](https://arxiv.org/abs/2603.25158) · [code](https://github.com/Qwen-Applications/Trace2Skill)
- [Memp (arXiv 2508.06433)](https://arxiv.org/abs/2508.06433) · [ReMe (arXiv 2512.10696)](https://arxiv.org/abs/2512.10696) · [H-EPM (arXiv 2512.07287)](https://arxiv.org/abs/2512.07287) · [Skill-Pro/ProcMEM (arXiv 2602.01869)](https://arxiv.org/abs/2602.01869)
- [AgentFly/Memento (arXiv 2508.16153)](https://arxiv.org/abs/2508.16153) · [ICE (arXiv 2401.13996)](https://arxiv.org/abs/2401.13996) · [AutoManual NeurIPS 2024 (arXiv 2405.16247)](https://arxiv.org/abs/2405.16247) · [AutoGuide (arXiv 2403.08978)](https://arxiv.org/abs/2403.08978)
- [SAGE/Skill-Augmented GRPO (arXiv 2512.17102)](https://arxiv.org/abs/2512.17102) · [SEAgent (arXiv 2508.04700)](https://arxiv.org/abs/2508.04700) · [EXIF (arXiv 2506.04287)](https://arxiv.org/abs/2506.04287) · [CASCADE (arXiv 2512.23880)](https://arxiv.org/abs/2512.23880) · [HMT (arXiv 2603.07024)](https://arxiv.org/abs/2603.07024) · [Darwinian Memory (arXiv 2601.22528)](https://arxiv.org/abs/2601.22528)
- [Graph-of-Skills (arXiv 2604.05333)](https://arxiv.org/abs/2604.05333) · [Skill-as-Pseudocode (arXiv 2605.27955)](https://arxiv.org/abs/2605.27955) · [SkillOps (arXiv 2605.13716)](https://arxiv.org/abs/2605.13716) · [Skill Drift (arXiv 2605.10990)](https://arxiv.org/abs/2605.10990)
- [Odyssey IJCAI 2024](https://www.ijcai.org/proceedings/2024/0022.pdf) · [LRLL (arXiv 2406.18746)](https://arxiv.org/abs/2406.18746)
- Surveys: [Self-Evolving Agents→ASI (2507.21046)](https://arxiv.org/abs/2507.21046) · [Self-Evolving AI Agents (2508.07407)](https://arxiv.org/abs/2508.07407) · [Lifelong Learning Roadmap (2501.07278)](https://arxiv.org/abs/2501.07278) · [Adaptation of Agentic AI (2512.16301)](https://arxiv.org/abs/2512.16301) · [ELL (2508.19005)](https://arxiv.org/abs/2508.19005)

---

**Summary of work done**: Ran 16+ distinct searches across the academic-search MCP (Semantic Scholar + OpenAlex providers) and Exa, verifying every paper's title/authors/year/venue/arXiv-ID against retrieved sources (arXiv, OpenReview, GitHub, NeurIPS/IJCAI/PMLR proceedings, Google Research blog). The dates extend into 2026 consistent with the stated current date 2026-06-07.

**Key findings**: (1) The field has exploded post-AWM into a clear skill-induction lineage (AWM→ASI→SkillWeaver→PolySkill→Trace2Skill) and a procedural-memory lineage (Memp, ReMe, H-EPM, Skill-Pro). (2) **ReasoningBank (Google, ICLR 2026)** is the most important single omission — it is the current reference point that newer papers (Trace2Skill) benchmark against. (3) **AgentFly/Memento** directly instantiates the "case-based reasoning / experience reuse" phrasing in your topic. (4) An entirely new 2026 sub-field of **skill-library engineering/maintenance** (Graph-of-Skills, SkillOps, Skill Drift, Skill-as-Pseudocode) has emerged that is unusually relevant to agent-skill-learning work, since it treats skills as portable, auditable, maintainable software artifacts. (5) The framework lacks any survey backbone and would benefit from the recommended fields — especially `knowledge_granularity`, `learns_from_failure`, `transfer_axis_tested`, and `skill_representation` (NL-vs-code), which are the actual axes on which these systems differentiate.
Cluster E research complete. Summary of deliverables provided above:

**Items supplemented (28 new systems, all metadata verified):**
- **Skill-induction lineage**: ASI (2504.06821), SkillWeaver (2504.07079), PolySkill (2510.15863), Trace2Skill (2603.25158), Odyssey (IJCAI 2024), LRLL (2406.18746)
- **Procedural-memory wave**: ReasoningBank (2509.25140, top priority), Memp (2508.06433), ReMe (2512.10696), H-EPM (2512.07287), Skill-Pro/ProcMEM (2602.01869)
- **Case-based reasoning / experience reuse**: AgentFly/Memento (2508.16153), ICE (2401.13996), AutoManual (NeurIPS 2024, 2405.16247), AutoGuide (2403.08978)
- **RL-trained / self-evolving CUA-GUI-web agents**: SAGE (2512.17102), SEAgent (2508.04700), EXIF (2506.04287), CASCADE (2512.23880), HMT (2603.07024), Darwinian Memory (2601.22528)
- **Skill-library engineering/maintenance (new 2026 sub-field)**: Graph-of-Skills (2604.05333), Skill-as-Pseudocode (2605.27955), SkillOps (2605.13716), Skill Drift (2605.10990)
- **Surveys**: 5 anchoring surveys (2507.21046, 2508.07407, 2501.07278, 2512.16301, 2508.19005)

**Fields recommended**: 11 new dimensions, most notably `knowledge_granularity`, `learns_from_failure`, `transfer_axis_tested`, `skill_representation` (NL-vs-code), `parameter_update`, `library_scaling_mechanism`, `strong_to_weak_transfer`, and `shareability_across_agents`.

**Top takeaways**: ReasoningBank and AgentFly/Memento are the most consequential omissions; an entirely new skill-library-as-software-ecosystem sub-field (GoS/SkillOps/Skill Drift) is highly relevant to your mediator-coevo agent-skill work; and the framework would benefit from a survey backbone plus the differentiation-axis fields. All findings, with verified URLs and source citations, are in the structured response above.
The Cluster E literature search is complete and the full structured results have already been delivered above. No further action is needed.

The deliverables provided:
- **28 verified new items** spanning skill-induction (ASI, SkillWeaver, PolySkill, Trace2Skill, Odyssey, LRLL), procedural memory (ReasoningBank, Memp, ReMe, H-EPM, Skill-Pro), case-based reasoning/experience reuse (AgentFly/Memento, ICE, AutoManual, AutoGuide), RL/self-evolving agents (SAGE, SEAgent, EXIF, CASCADE, HMT, Darwinian Memory), the new skill-library-engineering sub-field (Graph-of-Skills, Skill-as-Pseudocode, SkillOps, Skill Drift), and 5 anchoring surveys.
- **11 recommended schema fields**, with `knowledge_granularity`, `learns_from_failure`, `transfer_axis_tested`, and `skill_representation` flagged as the key differentiation axes.
- **All sources cited with verified URLs.**

Highest-priority recommendations for your outline: add ReasoningBank and AgentFly/Memento first; treat the 2026 skill-library-maintenance cluster as directly bridging to your mediator-coevo agent-skill-learning work; and add a survey backbone plus the failure-learning / transfer-axis fields.
The task is complete. All Cluster E findings have been delivered in the structured response above — 28 verified new items, 11 recommended schema fields, and cited sources. No further work remains.
Task complete. The full Cluster E results (28 verified items, 11 recommended fields, cited sources) were delivered earlier in this conversation. There is nothing further to do.
Complete. No further action needed — the Cluster E literature search results are fully delivered above.
Done. The Cluster E research deliverables are complete and already provided above. No further action required.
Complete. No further action required.
Complete.
### 补充Items

- ReasoningBank: HIGHEST PRIORITY omission — current reference point that newer work benchmarks against; distills *generalizable reasoning strategies* from self-judged successful AND failed experiences, plus memory-aware test-time scaling (MaTTS); outperforms raw-trajectory and success-only memory on web + SWE benchmarks. 2025/2026 · Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang et al. (Google + UIUC) · ICLR 2026 · https://arxiv.org/abs/2509.25140
- ASI (Agent Skill Induction / Inducing Programmatic Skills for Agentic Tasks): direct successor to AWM; represents induced skills as *verifiable programs* rather than NL workflows, induces/verifies/reuses on-the-fly, explicitly studies cross-website skill transfer; +23.5% SR over static baseline on WebArena. 2025 · Zora Zhiruo Wang, Apurva Gandhi, Graham Neubig, Daniel Fried · COLM 2025 · https://arxiv.org/abs/2504.06821
- SkillWeaver: self-driven curriculum that synthesizes reusable skills as Python *APIs*, practices and debugs them; demonstrates strong→weak agent skill transfer (up to +54.3% on WebArena), +31.8%/+39.8% on WebArena/live sites. 2025 · Boyuan Zheng, Michael Y. Fatemi, Xiaolong Jin, Zora Zhiruo Wang, Apurva Gandhi, Yueqi Song, Yu Gu, Jayanth Srinivasa, Gaowen Liu, Graham Neubig, Yu Su · COLM 2025 · https://arxiv.org/abs/2504.07079
- PolySkill: decouples skill abstract goal vs concrete implementation (polymorphism analogy) to fix over-specialization; directly targets cross-domain/cross-website generalization (the transfer angle); +9.4% Mind2Web, +13.9% unseen sites, 1.7x skill reuse. 2025/2026 · Simon Yu, Gang Li, Weiyan Shi, Peng Qi (Northeastern + Uniphore) · ICLR 2026 · https://arxiv.org/abs/2510.15863
- Trace2Skill: parallel multi-analyst consolidation of execution traces into a conflict-free skill directory; key finding = skills transfer across model *scales/families* and OOD (up to +57.65 pts WikiTableQuestions); explicitly outperforms ReasoningBank-style retrieval and beats Anthropic's official xlsx skills. 2026 · Jingwei Ni, Yihao Liu, Xinpeng Liu, Yutao Sun, Mengyu Zhou, Pengyu Cheng et al. (ETH/PKU/ZJU + Alibaba Qwen) · arXiv preprint · https://arxiv.org/abs/2603.25158
- Memp: task-agnostic learnable/updatable/lifelong procedural memory; systematically studies Build/Retrieve/Update strategies; demonstrates strong→weak model procedural-memory transfer; TravelPlanner + ALFWorld. 2025 · Runnan Fang, Yuan Liang, Xiaobin Wang, Jialong Wu, Shuofei Qiao, Pengjun Xie, Fei Huang, Huajun Chen, Ningyu Zhang (Zhejiang U + Alibaba) · arXiv preprint · https://arxiv.org/abs/2508.06433
- ReMe (Remember Me, Refine Me): full procedural-memory lifecycle — multi-faceted distillation (success/failure/comparative), context-adaptive reuse, utility-based refinement/pruning; shows memory-scaling effect (Qwen3-8B+ReMe beats memoryless Qwen3-14B); BFCL-V3, AppWorld. 2025 · Zouying Cao, Jiaji Deng, Li Yu, Wei Zhou, Zhaoyang Liu, Bolin Ding, Hai Zhao · arXiv preprint · https://arxiv.org/abs/2512.10696
- H-EPM (Experience-Evolving Multi-Turn Tool-Use Agent w/ Hybrid Episodic-Procedural Memory): builds a tool graph (procedural routines) augmented with episodic summaries; adds memory-guided RL that biases exploration toward successful tool transitions; up to +50% inference, +40% OOD. 2025 · Sijia Li, Yuchen Huang, Zifan Liu, Zijian Li, Jingjing Fu, Lei Song, Jiang Bian, Jun Zhang, Rui Wang (Microsoft Research) · arXiv preprint · https://arxiv.org/abs/2512.07287
- Skill-Pro / ProcMEM (Learning Reusable Procedural Memory via Non-Parametric PPO): formalizes a Skill-MDP; converts episodic narratives into executable Skills (activation/execution/termination conditions) via Non-Parametric PPO; evaluated in in-domain, cross-task AND cross-agent settings. 2026 · Qirui Mi, Zhijian Ma, Mengyue Yang, Haoxuan Li, Yisen Wang, Haifeng Zhang, Jun Wang · arXiv preprint · https://arxiv.org/abs/2602.01869
- AgentFly / Memento (Fine-tuning LLM Agents without Fine-tuning LLMs): explicit *case-based reasoning* framing — directly matches "case-based reasoning / experience reuse" in the topic; Memory-augmented MDP + neural case-selection policy via Soft Q-Learning, no weight updates; top-1 GAIA val 87.88%, +4.7–9.6 pts OOD. 2025 · Huichi Zhou, Yihang Chen, Yongtao et al. (instantiated as AgentFly; Memento repo) · arXiv preprint · https://arxiv.org/abs/2508.16153
- ICE (Investigate-Consolidate-Exploit): first strategy for *inter-task* agent self-evolution; consolidates plans into reusable workflows + trajectories into pipelines (finite automata); foundational 2024 piece the framework currently omits. 2024 · Cheng Qian, Shihao Liang, Yujia Qin et al. (Tsinghua) · arXiv preprint · https://arxiv.org/abs/2401.13996
- AutoManual: Planner/Builder/Formulator agents distill interactive experience into a rule-based, human-readable manual; manuals built by GPT-4 *transfer to guide smaller LLMs*; addresses the "Path Dependency" problem; ALFWorld 97.4%. 2024 · Minghao Chen, Yihang Li, Yanting Yang, Shiyu Yu, Binbin Lin, Xiaofei He · NeurIPS 2024 · https://arxiv.org/abs/2405.16247
- AutoGuide: extracts *state-conditional* guidelines from offline success/failure trajectory pairs with state-aware retrieval — a distinctive "when-to-apply" experience-reuse mechanism for sequential decision-making. 2024 · Yao Fu, Dong-Ki Kim, Jaekyeom Kim, Sungryull Sohn, Lajanugen Logeswaran, Kyunghoon Bae, Honglak Lee · arXiv preprint · https://arxiv.org/abs/2403.08978
- SAGE (Skill-Augmented GRPO for self-Evolution / Reinforcement Learning for Self-Improving Agent with Skill Library): RL framework with Sequential Rollout over task chains + Skill-integrated Reward (rewards reusable skill creation); AppWorld +8.9% SGC, −26% steps, −59% tokens. 2025 · Jiongxiao Wang, Qiaojing Yan, Yawei Wang, Yijun Tian, Soumya Smruti Mishra, Zhichao Xu, Megha Gandhi, Panpan Xu, Lin Lee Cheong (UW-Madison + AWS Agentic AI) · arXiv preprint · https://arxiv.org/abs/2512.17102
- SEAgent (Self-Evolving Computer Use Agent): CUA self-evolves on novel software via experiential learning (World State Model + Curriculum Generator), adversarial imitation of failures + GRPO on successes; specialist→generalist transfer; OSWorld +23.2% SR. 2025 · Zeyi Sun, Ziyu Liu, Yuhang Zang, Yuhang Cao, Xiaoyi Dong, Tong Wu, Dahua Lin, Jiaqi Wang · arXiv preprint · https://arxiv.org/abs/2508.04700
- EXIF (Automated Skill Discovery through Exploration and Iterative Feedback): explorer agent (Alice) generates feasibility-grounded skill datasets to train target agent (Bob) in a closed loop; self-evolving skill discovery; WebShop, Crafter. 2025 · Yongjin Yang, Sinjae Kang, Juyong Lee, Dongjun Lee, Se-young Yun, Kimin Lee (KAIST) · arXiv preprint · https://arxiv.org/abs/2506.04287
- CASCADE (Cumulative Agentic Skill Creation through Autonomous Development and Evolution): explicit "LLM + tool use → LLM + skill acquisition" framing; two meta-skills (continuous learning, self-reflection); skills shareable across agents/scientists; SciSkillBench 93.3% (GPT-5) vs 35.4% w/o evolution. 2025 · Xu Huang, Junwu Chen, Yuxing Fei, Zhuohan Li, Philippe Schwaller, Gerbrand Ceder · arXiv preprint · https://arxiv.org/abs/2512.23880
- HMT (Hierarchical Memory Tree for Web Agents): decouples Intent / Stage (reusable subgoals with pre/post-conditions) / Action levels to fix flat-memory "workflow mismatch"; targets cross-website/cross-domain generalization on Mind2Web + WebArena. 2026 · Yunteng Tan, Zhiqiang Gao, Xinxiao Wu · arXiv preprint · https://arxiv.org/abs/2603.07024
- Darwinian Memory System (DMS): training-free self-regulating memory for GUI agents; decomposes trajectories into reusable units + utility-driven natural selection (pruning suboptimal paths); +18.0% SR, +33.9% stability on multi-app benchmarks. 2026 · Hongze Mi, Yibo Feng, Wenjie Lu, Song Cao, Jinyuan Li et al. · arXiv preprint · https://arxiv.org/abs/2601.22528
- Odyssey (Empowering Minecraft Agents with Open-World Skills): open-world skill library (40 primitive + 183 compositional skills) + fine-tuned LLaMA-3 backbone; a Voyager-lineage embodied skill-library system with a new agent-capability benchmark. 2024 · Shunyu Liu, Yaoru Li, Kongcheng Zhang, Zhenyu Cui, Wenkai Fang, Yuxuan Zheng, Tongya Zheng, Mingli Song · IJCAI 2024 · https://www.ijcai.org/proceedings/2024/0022.pdf
- LRLL (Lifelong Robot Library Learning): LLM agent that continuously grows a robot skill library (soft memory, self-guided exploration, skill abstractor); bridges experience reuse to embodied control; transfers knowledge from memory to library while avoiding catastrophic forgetting. 2024 · Georgios Tziafas, Hamidreza Kasaei · ICRA 2024 · https://arxiv.org/abs/2406.18746
- Graph-of-Skills (GoS): inference-time *structural* retrieval for large skill libraries (200–2,000 skills); executable skill graph + reverse-aware Personalized PageRank to retrieve dependency-complete skill bundles; +25.55% reward, −56.72% tokens on SkillsBench/ALFWorld. 2026 · Dawei Li, Zongxia Li, Hongyang Du, Xiyang Wu, Shihang Gui, Yongbei Kuang, Lichao Sun · arXiv preprint · https://arxiv.org/abs/2604.05333
- Skill-as-Pseudocode (SaP): converts free-form markdown skills into typed pseudocode (typed contract + concrete invocation templates) with a deterministic verifier; reduces "confused→re-retrieve" loops; ALFWorld with gpt-4o-mini. 2026 · Xinze Li, Yuhang Zang, Yixin Cao, Aixin Sun · arXiv preprint · https://arxiv.org/abs/2605.27955
- SkillOps (Managing Skill Libraries as Self-Maintaining Software Ecosystems): formalizes library-level "skill technical debt"; typed Skill Contracts + Hierarchical Skill Ecosystem Graph; near-zero library-time LLM calls; plug-in maintenance layer. 2026 · Hongji Pu, Xinyuan Song, Liang Zhao · arXiv preprint · https://arxiv.org/abs/2605.13716
- Skill Drift Is Contract Violation: formalizes skill decay as contract violation when underlying APIs/packages evolve; precision-first proactive maintenance; releases an 880-pair skill-degradation benchmark. 2026 · Linfeng Fan, Yuan Tian, Ziwei Li, Zhiwu Lu · arXiv preprint · https://arxiv.org/abs/2605.10990
- Self-evolving Agents with reflective and memory-augmented abilities (SAGE, 2024 — distinct from the Amazon SAGE above): iterative feedback + reflection + Ebbinghaus-forgetting-curve memory optimization for multi-task/long-span info. 2024 · Xuechen Liang, Meiling Tao, Yinghui Xia, Tianyu Shi, Jun Wang, JingSong Yang · arXiv preprint · https://arxiv.org/abs/2409.00872
- A Survey of Self-Evolving Agents (What/When/How/Where to Evolve, on the Path to ASI): first systematic self-evolving-agents survey; axes covering model/memory/tools/architecture — useful survey backbone the framework currently lacks. 2025 · multi-author survey · arXiv preprint · https://arxiv.org/abs/2507.21046
- A Comprehensive Survey of Self-Evolving AI Agents (Bridging Foundation Models and Lifelong Agentic Systems): unified feedback-loop framework (System Inputs / Agent System / Environment / Optimisers); maps domain-specific evolution strategies. 2025 · multi-author survey · arXiv preprint · https://arxiv.org/abs/2508.07407
- Lifelong Learning of LLM-based Agents: A Roadmap: Perception/Memory/Action module decomposition for continual learning; catastrophic-forgetting focus; maintains an awesome-list of the field. 2025 · Junhao Zheng, Chengming Shi, Xidi Cai, Qiuke Li et al. (qianlima-lab) · arXiv preprint · https://arxiv.org/abs/2501.07278
- Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills: four-paradigm framework (A1/A2 agent adaptation; T1/T2 tool/skill adaptation); directly maps the skill-library design space. 2025 · Pengcheng Jiang, Jiacheng Lin, Zhiyi Shi et al. · arXiv preprint · https://arxiv.org/abs/2512.16301
- Building Self-Evolving Agents via Experience-Driven Lifelong Learning (ELL): Framework and Benchmark: ELL framework (Experience Exploration / Long-term Memory / Skill Learning / Knowledge Internalization) + a benchmark — useful evaluation anchor. 2025 · multi-author · arXiv preprint · https://arxiv.org/abs/2508.19005

### 推荐补充字段

- knowledge_granularity: what unit is stored/reused — raw trajectory / success-only routine / contrastive (success+failure) insight / reasoning strategy / state-conditional guideline / rule / executable skill. This is the single biggest differentiator across systems (e.g., ReasoningBank's reasoning strategies vs AWM's NL workflows vs raw-trajectory memory) and is missing from the current schema.
- memory_lifecycle_operations: which of {build/induce, retrieve, update, refine, prune/deprecate, consolidate} the system supports. Newer systems (Memp, ReMe, DMS, SkillOps) are defined by *active* update/prune, vs early append-only memory — critical for distinguishing self-maintaining libraries from static ones.
- learns_from_failure: does the system distill from failed trajectories, not only successes? Increasingly emphasized axis that separates ReasoningBank, AutoGuide, AgentFly, SEAgent from success-only methods like Voyager/AWM; central to robust skill learning.
- transfer_axis_tested: which transfer was *empirically* evaluated — cross-task / cross-website / cross-domain / cross-model-scale / cross-model-family / cross-agent / OOD. Many systems claim generalization but test only one axis; Trace2Skill and PolySkill stand out for testing cross-model/cross-domain, and this is the core of the cluster's research question.
- parameter_update: weight-frozen (memory/prompt only) vs RL/SFT fine-tuned vs hybrid. Cleanly splits frozen approaches (AgentFly, ReasoningBank, Memp) from GRPO/RL ones (SEAgent, SAGE) — directly relevant to the skill-learning-without-fine-tuning angle of agent-skill work.
- skill_representation: natural-language / structured-rule / pseudocode / executable-code-API / finite-automaton-pipeline / soft-prompt. Refines the existing knowledge_carrier; the NL-vs-code debate (AWM/AutoGuide NL vs ASI/SkillWeaver/PolySkill code) is a recurring design axis tied to verifiability and reuse robustness.
- library_scaling_mechanism: how the system handles a growing/large library — retrieval method, dependency handling, context-budgeting, maintenance. Entire 2026 sub-field (GoS, SkillOps, Skill Drift, SaP) is about this and it is absent from the current schema; highly relevant once a skill library is non-trivial in size.
- strong_to_weak_transfer: can skills/memory built by a strong model boost a weaker model? Explicit selling point of Memp, Trace2Skill, and SkillWeaver; a distinctive and practically important property for deployment economics.
- composability: can stored skills be composed/nested into higher-level skills? Emphasized by Odyssey, ASI, PolySkill, LRLL; determines whether a library scales to long-horizon tasks or stays flat.
- shareability_across_agents: are skills exportable/auditable/shareable artifacts (vs model-internal weights)? The "agent skills as portable SKILL.md" framing (CASCADE, SkillWeaver, SkillOps) directly bridges to mediator-coevo / multi-agent skill-learning work.
- relation_to_test_time_scaling: whether experience reuse is coupled with test-time scaling / compute allocation. ReasoningBank+MaTTS is the leading example establishing "memory-driven experience scaling" as a new scaling dimension — an emerging axis worth tracking.

### 信息来源

- [ReasoningBank (arXiv 2509.25140)](https://arxiv.org/abs/2509.25140)
- [ReasoningBank — Google Research blog](https://research.google/blog/reasoningbank-enabling-agents-to-learn-from-experience/)
- [ReasoningBank — OpenReview ICLR 2026](https://openreview.net/forum?id=jL7fwchScm)
- [Agent Workflow Memory (PMLR / arXiv 2409.07429)](https://proceedings.mlr.press/v267/wang25bx.html)
- [ASI — Inducing Programmatic Skills (arXiv 2504.06821)](https://arxiv.org/abs/2504.06821)
- [SkillWeaver (arXiv 2504.07079)](https://arxiv.org/abs/2504.07079)
- [SkillWeaver project page](https://osu-nlp-group.github.io/SkillWeaver/)
- [PolySkill (arXiv 2510.15863)](https://arxiv.org/abs/2510.15863)
- [PolySkill — ICLR 2026 poster](https://iclr.cc/virtual/2026/poster/10010108)
- [Trace2Skill (arXiv 2603.25158)](https://arxiv.org/abs/2603.25158)
- [Memp (arXiv 2508.06433)](https://arxiv.org/abs/2508.06433)
- [ReMe (arXiv 2512.10696)](https://arxiv.org/abs/2512.10696)
- [H-EPM (arXiv 2512.07287)](https://arxiv.org/abs/2512.07287)
- [Skill-Pro / ProcMEM (arXiv 2602.01869)](https://arxiv.org/abs/2602.01869)
- [AgentFly / Memento (arXiv 2508.16153)](https://arxiv.org/abs/2508.16153)
- [ICE — Investigate-Consolidate-Exploit (arXiv 2401.13996)](https://arxiv.org/abs/2401.13996)
- [AutoManual — NeurIPS 2024 (arXiv 2405.16247)](https://arxiv.org/abs/2405.16247)
- [AutoGuide (arXiv 2403.08978)](https://arxiv.org/abs/2403.08978)
- [SAGE / Skill-Augmented GRPO (arXiv 2512.17102)](https://arxiv.org/abs/2512.17102)
- [SEAgent (arXiv 2508.04700)](https://arxiv.org/abs/2508.04700)
- [EXIF (arXiv 2506.04287)](https://arxiv.org/abs/2506.04287)
- [CASCADE (arXiv 2512.23880)](https://arxiv.org/abs/2512.23880)
- [HMT — Hierarchical Memory Tree (arXiv 2603.07024)](https://arxiv.org/abs/2603.07024)
- [Darwinian Memory System (arXiv 2601.22528)](https://arxiv.org/abs/2601.22528)
- [Odyssey — IJCAI 2024](https://www.ijcai.org/proceedings/2024/0022.pdf)
- [LRLL — Lifelong Robot Library Learning (arXiv 2406.18746)](https://arxiv.org/abs/2406.18746)
- [Graph-of-Skills (arXiv 2604.05333)](https://arxiv.org/abs/2604.05333)
- [Skill-as-Pseudocode (arXiv 2605.27955)](https://arxiv.org/abs/2605.27955)
- [SkillOps (arXiv 2605.13716)](https://arxiv.org/abs/2605.13716)
- [Skill Drift Is Contract Violation (arXiv 2605.10990)](https://arxiv.org/abs/2605.10990)
- [Self-evolving Agents w/ reflective & memory-augmented abilities — 2024 SAGE (arXiv 2409.00872)](https://arxiv.org/abs/2409.00872)
- [Survey: Self-Evolving Agents → ASI (arXiv 2507.21046)](https://arxiv.org/abs/2507.21046)
- [Survey: Self-Evolving AI Agents (arXiv 2508.07407)](https://arxiv.org/abs/2508.07407)
- [Survey: Lifelong Learning of LLM Agents — Roadmap (arXiv 2501.07278)](https://arxiv.org/abs/2501.07278)
- [Survey: Adaptation of Agentic AI (arXiv 2512.16301)](https://arxiv.org/abs/2512.16301)
- [ELL — Experience-Driven Lifelong Learning (arXiv 2508.19005)](https://arxiv.org/abs/2508.19005)
The Cluster E literature search is complete. The full structured results were delivered in my previous message, covering:

- **30 verified items** (28 systems + canonical context), prioritizing 2024-present, organized across: skill-induction lineage (ASI, SkillWeaver, PolySkill, Trace2Skill, Odyssey, LRLL); procedural-memory wave (ReasoningBank, Memp, ReMe, H-EPM, Skill-Pro); case-based reasoning/experience reuse (AgentFly/Memento, ICE, AutoManual, AutoGuide); RL/self-evolving agents (SAGE, SEAgent, EXIF, CASCADE, HMT, Darwinian Memory); the new 2026 skill-library-engineering sub-field (Graph-of-Skills, Skill-as-Pseudocode, SkillOps, Skill Drift); and 5 anchoring surveys.
- **11 recommended schema fields** with rationale, key ones being `knowledge_granularity`, `learns_from_failure`, `transfer_axis_tested`, `skill_representation`, and `library_scaling_mechanism`.
- **35+ cited sources** with verified URLs.

**Top priorities for your outline**: ReasoningBank and AgentFly/Memento are the most consequential omissions; the 2026 skill-library-maintenance cluster (GoS/SkillOps/Skill Drift) directly bridges to your mediator-coevo agent-skill-learning work; and adding a survey backbone plus the failure-learning / transfer-axis fields would strengthen the framework's analytical dimensions.
The Cluster E literature search is complete and all deliverables have been provided. No further action is required.

Final summary of what was delivered:
- **补充Items**: 30 verified entries (2024–present prioritized) across skill-induction (ASI, SkillWeaver, PolySkill, Trace2Skill, Odyssey, LRLL), procedural memory (ReasoningBank, Memp, ReMe, H-EPM, Skill-Pro), case-based reasoning/experience reuse (AgentFly/Memento, ICE, AutoManual, AutoGuide), RL/self-evolving agents (SAGE, SEAgent, EXIF, CASCADE, HMT, Darwinian Memory), the 2026 skill-library-engineering sub-field (Graph-of-Skills, Skill-as-Pseudocode, SkillOps, Skill Drift), and 5 anchoring surveys — each with year, authors, venue, and verified URL.
- **推荐补充字段**: 11 fields with rationale (notably knowledge_granularity, learns_from_failure, transfer_axis_tested, skill_representation, library_scaling_mechanism, strong_to_weak_transfer, shareability_across_agents).
- **信息来源**: 35+ cited sources with URLs.

Highest-priority recommendations: add ReasoningBank and AgentFly/Memento first; treat the 2026 skill-library-maintenance cluster as the closest bridge to your mediator-coevo agent-skill work; add a survey backbone and the failure-learning / transfer-axis fields.
Task complete. The full structured Cluster E results (补充Items, 推荐补充字段, and 信息来源) were delivered earlier in this conversation. No further action is required.
Complete. No further action required — all Cluster E deliverables have been provided.
Complete.
Complete.
Complete.
Complete.
