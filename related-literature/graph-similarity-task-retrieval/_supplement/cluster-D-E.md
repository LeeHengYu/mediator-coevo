# Research Supplement — Cluster D & E

Topic: similarity-driven curriculum / task-graph selection and case-based / experience-memory retrieval for LLM agents (skill libraries, auto-curriculum, CBR, episodic memory reuse)
Date: 2026-06-07

### 补充Items

**Sub-area D — Skill libraries / skill induction / automatic & open-ended curriculum / task-graphs / similarity-driven task selection**

- **GITM (Ghost in the Minecraft)**: LLM agent with structured *text-based knowledge + memory* and goal-decomposition tree for open-world Minecraft; foundational skill/knowledge-memory agent that complements Voyager/JARVIS-1 and is missing from D. — Zhu et al., 2023, arXiv:2305.17144 (332 cites).
- **JARVIS-1**: Multimodal memory-augmented open-world agent whose *multimodal memory retrieval* of past survival experiences drives lifelong self-improvement; canonical D item the framework omits. — Z. Wang, Cai, Liu, Ma, Liang et al., 2023 (IEEE TPAMI 2025), arXiv:2311.05997 (189 cites).
- **Agent Workflow Memory (AWM)**: *Induces reusable text "workflows" (routines) from past trajectories and selectively retrieves them*; offline + online modes; strong cross-task/website/domain generalization on Mind2Web/WebArena — arguably the single most project-critical item (similarity-driven retrieval of induced routines). — Z. Wang, Mao, Fried, Neubig, 2024 (ICML 2025), arXiv:2409.07429 (174 cites).
- **Agent Skill Induction (ASI / "Inducing Programmatic Skills for Agentic Tasks")**: Represents skills as *verified executable programs* injected directly into the action space; explicit successor to AWM, adds programmatic verification before reuse. — Z. Wang, Gandhi, Neubig, Fried, 2025, arXiv:2504.06821 (59 cites).
- **SkillWeaver**: Web agents *autonomously synthesize reusable skills as APIs*, practice them, distill into robust APIs, retrieve by relevance; strong→weak agent skill transfer (+54.3% on WebArena). Directly models skill-library steering of agent coevolution. — Zheng, Fatemi, Jin, Wang, Gandhi, Song, Gu, Srinivasa, Liu, Neubig, Su, 2025, arXiv:2504.07079 (78 cites).
- **PAE (Proposer-Agent-Evaluator)**: *Context-aware task proposer + autonomous VLM success-evaluator* lets foundation-model internet agents discover/practice skills (auto-curriculum via task proposal + RL); ~50% relative zero-shot generalization gain. — Zhou, Yang, Lin, Bai, Zhou, Wang, Levine, L. Li, 2024 (ICML 2025), arXiv:2412.13194 (39 cites).
- **Mobile-Agent-E**: Self-evolving hierarchical mobile assistant with long-term memory of *Tips (general lessons) + Shortcuts (reusable executable subroutines)* — concrete skill-library + experience-reuse instance for GUI agents. — Z. Wang, Xu, J. Wang, Zhang, Yan, Zhang, Huang, Ji, 2025, arXiv:2501.11733 (113 cites).
- **AutoManual**: Builds diverse *rules/manuals from interactive environmental experience* (Planner/Builder/Formulator) and uses a *case-conditioned prompting* strategy; self-generated manual guides smaller LLMs. Bridges D and E (cross-cutting). — M. Chen, Y. Li, Yang, Yu, Lin, He, 2024 (NeurIPS 2024), arXiv:2405.16247 (38 cites).
- **Eurekaverse**: LLM generates an *unsupervised curriculum of progressively harder environments represented as code* (automatic environment/curriculum design), validated on quadruped parkour. Strong D auto/open-ended-curriculum item beyond POET/PAIRED. — Liang, S. Wang, H. Wang, Bastani, Jayaraman, Ma, 2024 (CoRL), arXiv:2411.01775 (6 cites).
- **R-Zero**: Independent *Challenger + Solver co-evolve a self-generated curriculum* — Challenger rewarded for tasks at the edge of Solver capability (ZPD/learning-progress); maps almost exactly onto a "mediated coevolution" framing. — C. Huang, Yu, X. Wang, H. Zhang, Z. Li, R. Li, J. Huang, Mi, D. Yu, 2025, arXiv:2508.05004 (136 cites).
- **Absolute Zero / Absolute Zero Reasoner (AZR)**: A single model *proposes tasks that maximize its own learning progress* and solves them, with a code executor as verifiable reward; foundational self-curriculum/coevolution-from-zero-data item. — Zhao, Wu, Yue, T. Wu, Q. Xu, Lin, S. Wang, Q. Wu, Zheng, G. Huang, 2025, arXiv:2505.03335 (250 cites).
- **Self-Evolving Curriculum (SEC)**: Frames *curriculum/task selection as a non-stationary multi-armed bandit*, using absolute advantage as a learning-gain proxy across problem categories; directly relevant to "task selection by learning value." — 2025, arXiv:2505.14970 (v4 revised Oct 2025).
- **SkillGraph**: A *directed weighted execution-transition graph mined from ~50k successful agent trajectories* used as a graph foundation prior for tool-sequence retrieval + ordering; shows semantic similarity alone fails and graph structure is needed (Kendall-τ −0.433→+0.613). Bridges C (graph retrieval) and D (task/skill-graph) — exactly the project's graph-structured-similarity thesis (cross-cutting). — H. Liu, D. Li, 2026, arXiv:2604.19793.
- **Survey of Self-Evolving Agents ("What, When, How, Where to Evolve")**: Organizes evolution across models/memory/tools/architecture and treats co-evolutionary dynamics; best single survey to scope D+E. — Gao, Geng, Hua, Hu, Juan, H. Liu, S. Liu, Qiu, Qi, Y. Wu, H. Wang, Xiao, Y. Zhou, S. Zhang, J. Zhang, Xiang et al., 2025, arXiv:2507.21046 (73 cites).

*Emerging 2025–2026 skill-library frontier (surfaced via primary arXiv sources; several carry late-2025/2026 dates — re-verify final authors/venue before formal inclusion):*

- **SkillRL**: Recursive skill-augmented RL building a hierarchical SkillBank with adaptive retrieval; skill library *co-evolves with the agent's policy* during RL (ALFWorld/WebShop/search tasks). — Xia, J. Chen, H. Wang, J. Liu, Zeng, Y. Wang, S. Han, Zhou, X. Zhao, H. Chen, Zheng, Xie, Yao, 2026, arXiv:2602.08234.
- **Skill-Pro**: Learns reusable procedural Skills (Activation/Execution/Termination) via *non-parametric PPO* over a Skill-MDP, with semantic-gradient skill proposal and a PPO-gate for in-domain/cross-task/cross-agent reuse. — 2026, arXiv:2602.01869.
- **PolySkill**: Learns a *domain-driven skill hierarchy via polymorphic abstraction* (abstract site classes → concrete implementations) for cross-website skill transfer/generalization; introduces transfer/reuse metrics. — 2025, arXiv:2510.15863.
- **SkillX**: Fully automated pipeline constructing a *multi-level (Planning/Functional/Atomic) skill knowledge base* from agent trajectories with consolidation/validation and experience-guided exploration. — 2026, arXiv:2604.04804.
- **SkillPyramid**: *Hierarchical skill consolidation* with a task-driven Skill Creator that retrieves and composes existing skills into new ones for self-evolving agents. — 2026, arXiv:2606.03692.
- **Trace2Skill**: Distills *trajectory-local lessons into transferable skills* via parallel multi-agent patch proposal and conflict-free consolidation into a skill directory. — 2026, arXiv:2603.25158.
- **SkillFlow**: Scalable *agent skill retrieval system* over large heterogeneous community skill libraries (SKILL.md bundles); addresses similarity retrieval over large skill corpora (shared vs self-generated libraries). — 2025, arXiv:2504.06188.

**Sub-area E — Case-based reasoning with LLMs / experience & episodic memory / case retrieval / reflection-reuse**

- **DS-Agent**: The seminal *CBR-cycle (retrieve-reuse-revise-retain) LLM agent* for automated data science; retrieves Kaggle "cases," revises plans via execution feedback, retains successes, simplified CBR at deployment. Top E item the framework should add. — Guo, Deng, Wen, H. Chen, Y. Chang, J. Wang, 2024 (ICML 2024), arXiv:2402.17453 (115 cites).
- **Review of CBR for LLM Agents**: Systematic review + *mathematical model of case retrieval/adaptation/learning* for LLM agents; ties CBR to self-reflection/introspection/curiosity and goal-driven autonomy; compares CBR vs CoT vs standard RAG. Best E survey anchor. — Hatalis, Christou, Kondapalli, 2025, arXiv:2504.06943 (15 cites).
- **MCBR-RAG (Multimodal CBR-RAG)**: Generalizes CBR-RAG to *multimodal cases* by converting non-text components to text and learning application-specific indexable latent representations for Retrieve/Reuse. Natural extension of the framework's CBR-RAG. — 2025, arXiv:2501.05030.
- **CBR-DDI**: CBR + *LLM-GNN collaborative case retrieval* with dual-layer knowledge-enhanced reuse and representative-sampling case refinement (drug-drug interaction); demonstrates graph-aware case retrieval (cross-cutting C/E). — 2025, arXiv:2505.23034.
- **A-MEM (Agentic Memory)**: *Zettelkasten-style dynamic memory* that auto-generates structured notes (context/keywords/tags), links memories by similarity, and evolves links/attributes when new memories arrive — relevance-driven, self-organizing episodic memory. Major E item, not in framework. — W. Xu, Liang, Mei, H. Gao, Tan, Y. Zhang, 2025, arXiv:2502.12110 (603 cites).
- **Synapse**: *Trajectory-as-exemplar prompting with an exemplar memory retrieved by similarity* (plus state abstraction); explicitly exploits task similarity for generalization to novel tasks on MiniWoB++/Mind2Web. Foundational E retrieval item. — L. Zheng, R. Wang, B. An, 2023 (ICLR 2024), arXiv:2306.07863 (147 cites).
- **RAP (Retrieval-Augmented Planning)**: Dynamically *retrieves past experiences matching the current situation/context* to guide planning; works in text-only and multimodal embodied settings. Core E item. — Kagaya, Yuan, Lou, Karlekar, Pranata, Kinose, Oguri, Wick, You, 2024, arXiv:2402.03610 (83 cites).
- **Memp (Mem^p)**: *Learnable, updatable, lifelong procedural memory* that distills trajectories into both step-level instructions and script-like abstractions, with explicit Build/Retrieve/Update strategies and stronger→weaker model transfer (TravelPlanner/ALFWorld). Direct E item on experience-reuse mechanics. — zjunlp (Zhejiang Univ.), 2025, arXiv:2508.06433.
- **MemGen**: *Generative latent memory* woven into reasoning via a memory trigger + memory weaver; benchmarks against ExpeL and AWM (up to +38%). Useful E counterpoint (latent/generative vs retrieval-based memory). — G. Zhang, Fu, Yan, 2025, arXiv:2509.24704 (59 cites).
- **CTIM-Rover**: *Negative/limitation result* — a cross-task-instance episodic memory (built on ExpeL) does NOT beat AutoCodeRover on real-world SWE; noise from distracting retrieved CTIM items/exemplar trajectories degrades performance. Important cautionary item on retrieval quality. — Lindenbauer, Groh, Schütze, 2025, arXiv:2505.23422 (2 cites).
- **HippoRAG**: Hippocampal-indexing-inspired retrieval (LLM + knowledge graph + Personalized PageRank) for integrating new experiences into long-term memory; bridges C (GraphRAG) and E (episodic memory). Complements the framework's HippoRAG mention by emphasizing its episodic-memory role (cross-cutting). — Gutierrez, Shu, Gu, Yasunaga, Su, 2024, arXiv:2405.14831 (223 cites).

### 推荐补充字段

- **case_representation / skill_representation**: How a stored unit is encoded — free-form text, structured note, executable program/API, parameterized procedure, or multimodal. Sharply differentiates AWM (text) vs ASI/SkillWeaver (code) vs Memp (script abstractions); central to "what gets retrieved" for the mediator.
- **memory_lifecycle_operations**: Which of Build/Add, Retrieve, Reuse/Adapt, Revise, Update, Deprecate/Forget, Retain are supported. Maps directly onto the CBR 4R cycle and distinguishes static stores from evolving ones (A-MEM, Memp).
- **retrieval_granularity**: Step / sub-trajectory / full-trajectory / task-level / skill-API / rule. Synapse uses full trajectories; Memp uses both step and script levels — matters for the mediator's retrieval design.
- **abstraction_level**: Raw instance vs induced/abstracted routine vs hierarchical skill tree. Voyager/AWM/SkillPyramid induce abstractions; key determinant of transfer.
- **transfer_scope_evaluated**: in-domain / cross-task / cross-website / cross-domain / cross-agent (strong→weak model). SkillWeaver and AWM explicitly test these; the project cares about transfer-by-similarity.
- **task_proposal_mechanism**: none / fixed-curriculum / LLM-proposed / adversarial-coevolution (Challenger–Solver) / learning-progress-or-ZPD-gated. Captures the auto-curriculum + coevolution axis (PAE, R-Zero, Absolute Zero, SEC).
- **difficulty_or_learning_gain_signal**: How a curriculum task is selected — uncertainty/self-consistency, edge-of-solvability (~50% pass), absolute advantage, novelty/diversity coverage. Directly informs a similarity/learning-value-driven mediator.
- **co_evolution_dynamics**: Whether two+ components (task-generator vs solver, skill-library vs policy) update against each other, and how instability (diversity/curriculum collapse) is mitigated. The project is literally about mediated coevolution; flagged by R-Few/Prism collapse-prevention work.
- **similarity_pitfalls / retrieval_failure_modes**: Documented cases where similarity retrieval misleads — semantic-only fails for ordering (SkillGraph), noisy episodic items hurt (CTIM-Rover). Forces honest assessment of when similarity-driven steering backfires.
- **verification_of_retrieved_item**: Whether reused skills/cases are validated before use — programmatic execution check, self-verification, judge/rubric gating. ASI's programmatic verification and self-play judges are why retrieval helps rather than harms.
- **parametric_vs_nonparametric_reuse**: Retrieval/in-context (no weight update) vs RL/SFT distillation vs latent-memory. Separates ExpeL/AWM/Synapse from SkillRL/MemGen; affects deployment cost and continual-learning behavior.
- **shared_vs_private_library**: Self-generated per-agent library vs shared/community corpus across agents. SkillWeaver transfer and SkillFlow community-skill retrieval raise this; relevant to a mediator serving many agents.

### 信息来源

- [Voyager (Wang et al., 2023, TMLR)](https://arxiv.org/abs/2305.16291)
- [GITM — Ghost in the Minecraft (Zhu et al., 2023)](https://arxiv.org/abs/2305.17144)
- [JARVIS-1 (Wang et al., 2023; IEEE TPAMI 2025)](https://arxiv.org/abs/2311.05997)
- [Agent Workflow Memory / AWM (Wang, Mao, Fried, Neubig, 2024; ICML 2025)](https://arxiv.org/abs/2409.07429)
- [Inducing Programmatic Skills / ASI (Wang, Gandhi, Neubig, Fried, 2025)](https://arxiv.org/abs/2504.06821)
- [SkillWeaver (Zheng et al., 2025)](https://arxiv.org/abs/2504.07079)
- [PAE — Proposer-Agent-Evaluator (Zhou et al., 2024; ICML 2025)](https://arxiv.org/abs/2412.13194)
- [Mobile-Agent-E (Wang et al., 2025)](https://arxiv.org/abs/2501.11733)
- [AutoManual (Chen et al., NeurIPS 2024)](https://arxiv.org/abs/2405.16247)
- [Eurekaverse (Liang et al., 2024, CoRL)](https://arxiv.org/abs/2411.01775)
- [R-Zero (Huang et al., 2025)](https://arxiv.org/abs/2508.05004)
- [Absolute Zero / AZR (Zhao et al., 2025)](https://arxiv.org/abs/2505.03335)
- [Self-Evolving Curriculum / SEC (2025)](https://arxiv.org/abs/2505.14970)
- [SkillGraph (Liu & Li, 2026)](https://arxiv.org/abs/2604.19793)
- [Survey of Self-Evolving Agents (Gao et al., 2025)](https://arxiv.org/abs/2507.21046)
- [SkillRL (Xia et al., 2026)](https://arxiv.org/abs/2602.08234)
- [Skill-Pro (2026)](https://arxiv.org/abs/2602.01869)
- [PolySkill (2025)](https://arxiv.org/abs/2510.15863)
- [SkillX (2026)](https://arxiv.org/abs/2604.04804)
- [SkillPyramid (2026)](https://arxiv.org/abs/2606.03692)
- [Trace2Skill (2026)](https://arxiv.org/abs/2603.25158)
- [SkillFlow (2025)](https://arxiv.org/abs/2504.06188)
- [DS-Agent (Guo et al., ICML 2024)](https://arxiv.org/abs/2402.17453)
- [Review of CBR for LLM Agents (Hatalis, Christou, Kondapalli, 2025)](https://arxiv.org/abs/2504.06943)
- [CBR-RAG (Wiratunga et al., ICCBR 2024)](https://arxiv.org/abs/2404.04302)
- [MCBR-RAG — Multimodal CBR-RAG (2025)](https://arxiv.org/abs/2501.05030)
- [CBR-DDI (2025)](https://arxiv.org/abs/2505.23034)
- [A-MEM — Agentic Memory (Xu et al., 2025)](https://arxiv.org/abs/2502.12110)
- [Synapse (Zheng, Wang, An, ICLR 2024)](https://arxiv.org/abs/2306.07863)
- [RAP — Retrieval-Augmented Planning (Kagaya et al., 2024)](https://arxiv.org/abs/2402.03610)
- [Memp — Exploring Agent Procedural Memory (2025)](https://arxiv.org/abs/2508.06433)
- [MemGen (Zhang, Fu, Yan, 2025)](https://arxiv.org/abs/2509.24704)
- [CTIM-Rover (Lindenbauer, Groh, Schütze, 2025)](https://arxiv.org/abs/2505.23422)
- [HippoRAG (Gutierrez et al., 2024)](https://arxiv.org/abs/2405.14831)
- [ExpeL — LLM Agents Are Experiential Learners (Zhao et al., AAAI 2024)](https://arxiv.org/abs/2308.10144)
