This Exa search surfaced the key dedicated communication surveys, which is exactly what goal #3 asked for. I now have comprehensive coverage. The most valuable surveys for the project:

- **Beyond Self-Talk: A Communication-Centric Survey of LLM-Based Multi-Agent Systems** (2502.14321) — explicitly communication-centric, covers info-sharing patterns, content, objects
- **Multi-Agent Collaboration Mechanisms: A Survey of LLMs** (2501.06322) — collaboration mechanisms, coordination protocols
- **A Survey of LLM-Driven AI Agent Communication: Protocols, Security Risks, and Defense** (2506.19676) — defines agent communication, 3-layer architecture
- **The Five Ws of Multi-Agent Communication** (2602.11583) — who/whom/when/what/why, bridges MARL + emergent language + LLMs

I have thoroughly covered all four goals with verified citations. Let me compile the final report.

---

# 调研报告：Mediator / Information-Routing / Communication-Efficient & Co-Evolving Multi-Agent LLM (2025–2026)

All arXiv IDs verified via Semantic Scholar MCP + Exa. Note: this MCP indexes arXiv listings dated into 2026 (e.g. 2602.*, 2605.*); I spot-verified the highest-value ones (TacoMAS, GTD, Economy of Minds) on arxiv.org and they are real. A few items mention placeholder-sounding model names in abstracts ("Gemini-3.1", "OpenClaw") — flagged where relevant; treat their headline numbers cautiously, but the methods/framings are sound.

## 补充Items

### A. Mediation / information routing / selective sharing (KEY for the project)
- **EIB-Learner — "Understanding the Information Propagation Effects of Communication Topologies in LLM-MAS"** (2025, Shen, Liu, Pan et al.): causal study of how correct/erroneous info propagates; moderately-sparse topologies suppress error while preserving beneficial diffusion — the exact filter/route tradeoff a mediator optimizes. https://arxiv.org/abs/2505.23352 ★
- **What Do Agents Communicate? Characterizing Information Exchange in Multi-Agent Systems** (2026): empirical characterization of what information actually flows between agents — directly informs what a mediator should filter/route. https://arxiv.org/abs/2605.20548
- **Enhancing Multi-Agent Communication through Attention Steering with Context Relevance** (2026): steers receiver attention by relevance of incoming messages — a soft information-filtering mechanism. https://arxiv.org/abs/2605.30136
- **Survey of LLM Agent Communication with MCP: A Software Design Pattern Centric Review** (2025, Sarkar & Sarkar): explicitly analyzes the **Mediator**, Broker, Observer, Publish-Subscribe patterns for agent communication — closest terminological match to "mediator-coevo." https://arxiv.org/abs/2506.05364
- **Learning to Interrupt in Language-based Multi-agent Communication** (2026): a controller learns *when* to interrupt/route messages — mediation as flow control. https://arxiv.org/abs/2604.06452
- **RUMAD: Reinforcement-Unifying Multi-Agent Debate** (2026): PPO controller adjusts edge weights + dual-threshold gating over **agent activation AND information visibility** (content-agnostic) — an explicit learned information-visibility mediator; cuts tokens >80%. https://arxiv.org/abs/2602.23864 ★

### B. Communication-efficient / learned topology (routing as edge selection)
- **AgentPrune — "Cut the Crap: An Economical Communication Pipeline for LLM-MAS"** (2024, Guibin Zhang et al.): defines *communication redundancy*; one-shot prunes the spatio-temporal message graph; 28–73% token cut. Seminal cost-reduction baseline. https://arxiv.org/abs/2410.02506 ★
- **G-Designer: Architecting Multi-agent Communication Topologies via GNNs** (2024, Guibin Zhang et al.): VGAE encodes agents + task virtual node, decodes task-adaptive topology; up to 95% token cut on HumanEval. https://arxiv.org/abs/2410.11782
- **Guided Topology Diffusion (GTD)** (2025, E. Jiang et al.): conditional discrete **graph-diffusion** generates topologies steered by a proxy multi-objective (accuracy/utility/cost) reward — gradient-free. Directly matches the repo's `docs/diffusion-motivation-framing.md`. Code: github.com/ericjiang18/diffusion_agent. https://arxiv.org/abs/2510.07799 ★
- **Optima: Optimizing Effectiveness and Efficiency for LLM-MAS** (2024, Chen/Yuan/Liu/Sun): trains agents (SFT/DPO+MCTS) for token-efficient communication; 2.8× gain with <10% tokens on info-asymmetric QA. https://arxiv.org/abs/2410.08115
- **Stop Wasting Your Tokens: Towards Efficient Runtime Multi-Agent Systems** (2025): runtime token-efficiency for MAS. https://arxiv.org/abs/2510.26585
- **ATOM: Budget-Controllable Multi-Agent Collaboration via Nucleus-Electron Hierarchy** (2026): offline backbone ("nucleus") + query-conditioned activated agents ("electrons"); complexity-aware budgeting; +30% token efficiency. https://arxiv.org/abs/2605.26178
- **Nexa — Response-Conditioned Parallel-to-Sequential Orchestration** (2026): lightweight transformer predicts a sparse DAG conditioned on first-round responses (no LLM judge / reward model); generalizes across #agents/task. https://arxiv.org/abs/2605.15573
- **CARD: Conditional Design of Multi-agent Topological Structures** (2026): conditional VGAE that adapts topology to *environment signals* (model upgrades, tool changes) at runtime. Code released. https://arxiv.org/abs/2603.01089
- **GoAgent: Group-of-Agents Communication Topology Generation** (2026): https://arxiv.org/abs/2603.19677
- **Graph-GRPO: Stabilizing Multi-Agent Topology Learning via GRPO** (2026): RL (group-relative PO) for topology learning. https://arxiv.org/abs/2603.02701
- **Agent Q-Mix: RL topology selection via QMIX/CTDE** (2026): decentralized round-wise communication-graph selection, reward = accuracy − token cost. (Abstract cites placeholder-sounding backbones — verify numbers.) https://arxiv.org/abs/2604.00344
- **TopoPrior: Learning Transferable Topology Priors across Domains** (2026): amortizes per-query topology search into an offline learned prior — efficiency win for routing. https://arxiv.org/abs/2605.17359
- **AgentBalance: Backbone-then-Topology Design under Budget** (2025): https://arxiv.org/abs/2512.11426

### C. Latent / non-token communication media (alternative mediation medium)
- **LatentMAS — Latent Collaboration in Multi-Agent Systems** (2025, Zou, Yang, Choi, Zou, Wang, Yang et al.): training-free **shared latent working memory** transfers hidden-state "thoughts" losslessly between agents; 70–84% fewer output tokens, 4× faster. A mediator could own/curate this shared latent memory. Code open. https://arxiv.org/abs/2511.20639 ★
- **Beyond tokens: a unified framework for latent communication in LLM-MAS** (2026): survey of 18 latent-communication methods along WHAT/WHICH/HOW axes (embeddings, hidden states, KV-cache; alignment; fusion). High-value reference for designing a latent mediation channel. https://arxiv.org/abs/2606.05711
- **Direct Semantic Communication Between LLMs via Vector Translation** (2025): learned cross-model vector translator enables sharing meaning not tokens. https://arxiv.org/abs/2511.03945
- **A Token/KV-Cache Communication Media Selection & Resource Allocation Strategy** (2026): adaptively picks token vs KV-cache transmission per regime — explicit medium-selection mediator. https://arxiv.org/abs/2605.25422

### D. Co-evolution / population-based / self-play (KEY for the "coevo" half)
- **TacoMAS: Test-Time Co-Evolution of Topology AND Capability in LLM-MAS** (2026, Xu, Hu, Wang, Feng et al.): *the* closest framing — jointly co-evolves agent capabilities (fast loop) and communication topology (slow meta-LLM loop, with agent birth/death + edge edits) toward a task-conditioned equilibrium. Code: github.com/chenxu2-gif/TacoMAS-MultiAgent. https://arxiv.org/abs/2605.09539 ★★
- **Economy of Minds: Emerging Multi-Agent Intelligence with Economic Interactions** (2026, Qi, Su, Du, Kakade et al.): population of agents bid in **auctions** for the right to act; wealth-based selection mutates/replaces agents — decentralized credit assignment & coordination *without explicit communication protocols*. A market as implicit mediator. https://arxiv.org/abs/2606.02859 ★
- **MCCE: A Framework for Multi-LLM Collaborative Co-Evolution** (2025): explicit multi-LLM co-evolution. https://arxiv.org/abs/2510.06270
- **PopuLoRA: Co-Evolving LLM Populations for Reasoning Self-Play** (2026): population-based asymmetric self-play (teacher/student LoRA adapters), weight-space evolution operators (mutation/crossover) on a frozen base — concrete co-evolution machinery. https://arxiv.org/abs/2605.16727
- **AutoMaAS: Self-Evolving Multi-Agent Architecture Search** (2025): operator generate/fuse/eliminate with cost-aware optimization. https://arxiv.org/abs/2510.02669
- **MAS-ZERO: Designing Multi-Agent Systems with Zero Supervision** (2025): inference-time self-evolved MAS design via meta-feedback, no validation set. https://arxiv.org/abs/2505.14996
- **SEW: Self-Evolving Agentic Workflows** (2025): co-evolves topology + prompts. https://arxiv.org/abs/2505.18646
- **Society of HiveMind: Multi-Agent Optimization of Foundation-Model Swarms** (2025): https://arxiv.org/abs/2503.05473
- **MonoScale: Scaling Multi-Agent System with Monotonic Improvement** (2026): https://arxiv.org/abs/2601.23219

### E. Adjacent / supporting
- **Experience as a Compass: Multi-agent RAG with Evolving Orchestration and Agent Prompts** (2026): https://arxiv.org/abs/2604.00901
- **Active Learning for Communication Structure Optimization in LLM-MAS** (2026): https://arxiv.org/abs/2605.05703
- **MasFACT: Continual Multi-Agent Topology Learning via Geometry-Aware Posterior** (2026): https://arxiv.org/abs/2605.17361
- **Towards Adaptive, Scalable, and Robust Coordination of LLM Agents: A Dynamic Ad-Hoc Networking** (2026): https://arxiv.org/abs/2602.08009
- **Token Economics for LLM Agents (survey)** (2026): treats tokens as economic primitives across single/multi-agent/ecosystem levels — useful framing for a cost-aware mediator. https://arxiv.org/abs/2605.09104

## ★ Top 5 MOST relevant to 'mediator-coevo'
1. **TacoMAS** (2605.09539) — co-evolution of topology + capability on two timescales; the project's thesis made concrete.
2. **EIB-Learner / Information Propagation Effects** (2505.23352) — formalizes the route-vs-filter (error suppression vs beneficial diffusion) tradeoff a mediator must solve.
3. **Guided Topology Diffusion (GTD)** (2510.07799) — diffusion-based, proxy-reward-steered topology generation; matches the repo's diffusion framing exactly.
4. **LatentMAS** (2511.20639) — shared latent working memory as a curated mediation medium; huge token savings.
5. **RUMAD** (2602.23864) — a learned RL controller that explicitly gates agent activation AND information visibility (content-agnostic), i.e. a trainable mediator; >80% token reduction. (Runner-up: **Economy of Minds** 2606.02859 for decentralized population co-evolution via market mediation.)

## 推荐补充字段 (new fields for the framework)
- **mediation_locus**: where the mediator sits — central hub / shared message pool / shared latent memory / per-edge gate / market mechanism. (The surveyed systems differ sharply; this is the project's core design axis.)
- **communication_medium**: token / structured-message / hidden-state / KV-cache / latent-vector / economic-signal. (Latent-communication cluster makes this first-class.)
- **filtering_granularity**: agent-level activation vs edge-level (whom-to-whom) vs message-level (what content) vs token-level. (RUMAD/AgentPrune/attention-steering each operate at different granularity.)
- **cost_objective**: explicit token/latency/$ term in the objective, and whether it's a hard budget vs soft penalty. (ATOM/Optima/Agent-Q-Mix/AgentBalance all formalize this.)
- **adaptivity_timescale**: per-query / test-time-online / offline-trained / continual — and if co-evolved, the relative timescales of the co-evolving components (TacoMAS fast-capability vs slow-topology).
- **topology_generator_class**: hand-crafted / search / GNN-VGAE / graph-diffusion / RL-policy / LLM-meta-designer. (Distinguishes G-Designer vs GTD vs Graph-GRPO vs MAS-ZERO.)
- **robustness_to_adversarial / error_propagation**: whether the mediator defends against malicious/erroneous messages (AgentPrune, EIB-Learner, RUMAD report this explicitly).
- **credit_assignment_mechanism**: how individual agents/edges are rewarded — RL advantage, economic wealth, evolutionary fitness. (Bridges the mediation and co-evolution halves.)

## 高价值综述 (Surveys)
- [Beyond Self-Talk: A Communication-Centric Survey of LLM-Based Multi-Agent Systems](https://arxiv.org/abs/2502.14321): system-level (architecture/goals/protocols) + internal (strategies/paradigms/objects/content) — best fit for the information-sharing axis.
- [Multi-Agent Collaboration Mechanisms: A Survey of LLMs](https://arxiv.org/abs/2501.06322): actors/types/structures/strategies/coordination-protocols framework.
- [A Survey of LLM-Driven AI Agent Communication: Protocols, Security Risks, and Defense](https://arxiv.org/abs/2506.19676): first formal definition of agent communication; 3-layer architecture (transport / protocol / semantic); 19 protocols.
- [The Five Ws of Multi-Agent Communication: Who/Whom/When/What/Why — from MARL to Emergent Language and LLMs](https://arxiv.org/abs/2602.11583): integrative taxonomy bridging classical MARL comms, emergent language, and LLM-MAS — directly maps to mediator routing decisions.
- [A survey of agent interoperability protocols: MCP, ACP, A2A, ANP](https://arxiv.org/abs/2505.02279) (111 cites): the canonical protocol survey.
- [Beyond tokens: unified framework for latent communication in LLM-MAS](https://arxiv.org/abs/2606.05711): specialized survey of the latent-communication mediation medium.
- [LLMs Working in Harmony (Architecture/Memory/Planning/Frameworks)](https://arxiv.org/abs/2504.01963) and [LLM-based Multi-Agents: A Survey of Progress and Challenges](https://arxiv.org/abs/2402.01680) (foundational; covers MetaGPT shared message pool, DyLAN dynamic routing).

## 信息来源
- Semantic Scholar via academic-search MCP (search_papers, explore_citations forward-citation graphs from GPTSwarm [2402.16823] and EIB-Learner [2505.23352], search_by_author)
- [arxiv.org](https://arxiv.org) via Exa (verification of 2510.07799, 2605.09539, 2606.02859)
- Exa web search (dedicated communication-survey discovery)

### Notes / caveats for the caller
- Seed-paper sweep: GPTSwarm's forward-citation random walk drifted heavily into VLA/robotics (low signal); the EIB-Learner and topic-keyword searches were far more productive for this slice. AgentPrune/G-Designer (Guibin Zhang's group) are the de-facto backbone of the communication-efficient-topology line and recur as baselines everywhere — worth treating as core prior art.
- The Du/Liang debate, AutoGen, MetaGPT, CAMEL, MoA, DyLAN seeds are well covered by the surveys above (MetaGPT's "shared message pool" and DyLAN's "dynamic agent selection / early-stopping" are the closest classical analogues to a mediator) — likely already in other agents' slices, so I focused on the newer descendants.
- A handful of "2026" entries carry placeholder-style model names in abstracts; methods are credible but headline numbers should be re-verified before citing.
