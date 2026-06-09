I now have AlphaEvolve (2506.13131) confirmed and comprehensive coverage. I've conducted 20+ distinct searches/fetches across academic-search and Exa, cross-verified key papers, and identified the major surveys and emerging methods. Let me compile the structured findings.

---

# Cluster 7 — Emerging / Hybrid (2025–2026) Self-Improving Agent Methods + Survey Overview

I ran 20+ searches/fetches (Semantic Scholar + Exa), prioritizing 2025–2026 arXiv, NeurIPS/ICLR 2025–2026 proceedings, and lab pages. Cross-verified each item below.

## Validation of already-listed items
All five already-listed items are confirmed real and correctly placed in Cluster 7. Verified IDs/venues for citation hygiene:
- **AgentSquare** — modular agent search (already listed; confirmed as a distinct line from ADAS/AFlow).
- **Optima** — Chen et al., arXiv 2410.08115, **ACL 2025**. Note: this is a 2024 paper; it's a multi-agent training (effectiveness+efficiency) method, overlaps your C6/multi-agent.
- **SiriuS** — Zhao, Yuksekgonul, Wu, Zou, arXiv **2502.04780** (2025). Self-improving MAS via bootstrapped reasoning + experience library. Confirmed.
- **Reflect-Retry-Reward** — confirmed as listed.
- **AlphaEvolve** — Novikov et al., arXiv **2506.13131** (DeepMind, 2025). Evolutionary coding agent; matrix-multiply (48 mults for 4×4) result. Confirmed.

**Gap finding:** The single most important omission is the **anchor surveys** and the **memory-as-policy / skill-as-policy** sub-cluster (2025–2026), plus the **co-evolution self-play** line (Absolute Zero, R-Zero) and **Darwin-Gödel Machine**. Details below.

---

## 补充Items (Supplementary Items)

### A. Self-improving/self-modifying agent frameworks (core Cluster 7)
- **Darwin Gödel Machine (DGM)**: Self-improving system that iteratively rewrites its own coding-agent code, validated empirically on SWE-bench (20%→50%) and Polyglot (14.2%→30.7%); maintains an evolutionary archive of agents. Zhang, Hu, Lu, Lange, Clune (Sakana AI / UBC), arXiv **2505.22954** (2025-05), 110+ citations. https://arxiv.org/abs/2505.22954
- **Gödel Agent**: Self-referential framework where the agent recursively modifies its own logic/code at runtime (not just hyperparameters). Yin, Wang, Pan, Wan, Wang, arXiv **2410.04444**; **ACL 2025 Long** (2025.acl-long.1354). The conceptual precursor to DGM. https://arxiv.org/abs/2410.04444
- **ADAS (Automated Design of Agentic Systems) / Meta Agent Search**: Meta-agent programs ever-better agents in code; the foundational "agents-designing-agents" paper. Hu, Lu, Clune, arXiv **2408.08435** (2024), 200+ citations. The direct lineage parent of DGM. https://arxiv.org/abs/2408.08435

### B. Automated prompt + workflow co-optimization (textual policy, hybrid)
- **GEPA (Genetic-Pareto)**: Reflective prompt evolution that reads full execution traces, mutates via LLM reflection, and uses Pareto-front selection; beats GRPO by up to 19pp with 35x fewer rollouts and beats MIPROv2. Agrawal, Tan, Soylu, … Zaharia, Khattab, arXiv **2507.19457**; **ICLR 2026 Oral**. The flagship "textual optimization > RL" item; co-evolves multiple textual components. https://arxiv.org/abs/2507.19457
- **MaAS (Multi-agent Architecture Search via Agentic Supernet)**: Optimizes a probabilistic *distribution* of agentic architectures (supernet), samples query-dependent systems; 6–45% of baseline cost. G. Zhang et al., arXiv **2502.04180**, **ICML 2025**, 119 citations. https://arxiv.org/abs/2502.04180
- **MASS (Multi-Agent System Search)**: Interleaved 3-stage optimization (block-level prompt → topology → workflow-level prompt). Han Zhou et al. (Google/Cambridge), arXiv **2502.02533** (2025), 89 citations. https://arxiv.org/abs/2502.02533
- **EvoFlow**: Niching evolutionary algorithm searching a *population* of heterogeneous, complexity-adaptive workflows (crossover/mutation/niching selection). G. Zhang et al., arXiv **2502.07373** (2025). https://arxiv.org/abs/2502.07373
- **MAPRO (Multi-Agent Prompt Optimization)**: Recasts MAS prompt optimization as MAP inference solved via language-guided max-product belief propagation with topology-aware credit assignment. Z. Zhang et al., arXiv **2510.07475** (2025). https://arxiv.org/abs/2510.07475
- **TEP (Textual Equilibrium Propagation)**: TextGrad successor; local-learning principle (free/nudged phases) that fixes exploding/vanishing textual gradients in deep compound AI systems. Chen, Deng, Zou, Yu, Li, arXiv **2601.21064** (2026). https://arxiv.org/abs/2601.21064

### C. Test-time / self-play co-evolution (RL ↔ self-improvement bridge)
- **Absolute Zero Reasoner (AZR)**: Single model proposes tasks that maximize its own learning progress and solves them, using a code executor as verifiable reward — zero external data. Zhao, Wu, Yue et al., arXiv **2505.03335** (2025), 250 citations. https://arxiv.org/abs/2505.03335
- **R-Zero**: Challenger–Solver co-evolution from zero data; Challenger rewarded for edge-of-ability tasks, Solver for solving them (GRPO + self-consistency pseudo-labels). C. Huang et al. (Tencent AI), arXiv **2508.05004**; **ICLR 2026**. https://arxiv.org/abs/2508.05004
- **TTRL (Test-Time Reinforcement Learning)**: RL on unlabeled test data using majority-vote rewards; enables self-evolution at test time (+211% pass@1 on AIME24 for Qwen-2.5-Math-7B). Zuo, K. Zhang et al. (Tsinghua), **NeurIPS 2025**, arXiv 2504.16084. https://openreview.net/forum?id=VuVhgEiu20
- **SPC (Self-Play Critic)**: Adversarial sneaky-generator vs. critic self-play to evolve step-level reasoning evaluation without annotation. Chen et al., arXiv **2504.19162** (2025). https://arxiv.org/abs/2504.19162

### D. Memory-as-policy / skill-as-policy (the fastest-growing 2025–2026 sub-cluster — currently underrepresented)
- **AgentFly / Memento**: Memory-based online RL over a Memory-augmented MDP (M-MDP) with a neural case-selection policy; continual learning **without** any LLM weight updates; top-1 on GAIA. Zhou et al., arXiv **2508.16153** (2025). The canonical "learn from experience, not gradients" item. https://arxiv.org/abs/2508.16153
- **ACE (Agentic Context Engineering)**: Treats context as an evolving "playbook" via generate/reflect/curate; prevents context collapse; works offline (system prompts) and online (agent memory) with execution feedback only; +10.6% agents, +8.6% finance. Q. Zhang et al. (Stanford/SambaNova), arXiv **2510.04618** (2025), 169 citations. https://arxiv.org/abs/2510.04618
- **Memory-as-Action**: Working-memory editing as part of a unified RL policy; introduces Dynamic Context Policy Optimization (DCPO) to handle "trajectory fractures." arXiv **2510.12635** (2025). https://arxiv.org/abs/2510.12635
- **Skill1**: Single policy that co-evolves skill *selection + utilization + distillation* from one task-outcome RL signal. arXiv **2605.06130** (2026). https://arxiv.org/abs/2605.06130
- **MemSkill**: Closed-loop RL controller for skill selection + LLM-guided skill-bank evolution from hard cases. arXiv **2602.02474** (2026). https://arxiv.org/abs/2602.02474
- **Skill-Pro**: Learns reusable procedural skills via Non-Parametric PPO (semantic gradients + trust-region "PPO gate"), no parameter updates. arXiv **2602.01869** (2026). https://arxiv.org/abs/2602.01869
- **Mem-π**: Models memory as a parametric policy that learns *when* and *what* to generate (decision-content decoupled GRPO). arXiv **2605.21463** (2026). https://arxiv.org/abs/2605.21463
- (Adjacent, optional) **SAGE** (Skill-Augmented GRPO, arXiv 2512.17102), **AgeMem** (2601.01885), **Meta-Policy Reflexion** (2509.03990) — variants in the same lane; cite as a cluster if space-limited.

### E. Experience-driven / early-experience learning
- **Era of Experience** (Silver & Sutton, DeepMind, 2025): Position paper arguing experiential data will dominate human data; foundational *position* anchor for the whole self-improving-agents motivation. (DeepMind PDF; widely cited.)
- **Agent Learning via Early Experience**: Reward-free paradigm (implicit world modeling + self-reflection) between imitation and RL. **ICLR 2026** submission (OpenReview pEGnJbmSUy).
- **ELL — Experience-driven Lifelong Learning framework + benchmark**: Four principles (exploration, long-term memory, skill learning, knowledge internalization). arXiv **2508.19005** (2025). https://arxiv.org/abs/2508.19005
- **Evolving-RL**: End-to-end RL that jointly optimizes experience *extraction* and *utilization* (co-evolution of extractor + solver). arXiv **2605.10663** (2026). https://arxiv.org/abs/2605.10663

### F. Agent distillation (skill acquisition via transfer — relevant new lane)
- **A Survey of On-Policy Distillation for LLMs** (review): formalizes OPD as f-divergence minimization; explicitly flags agentic distillation + self-play. Song & Zheng, arXiv **2604.00626** (2026), 40 citations. https://arxiv.org/abs/2604.00626
- **π-Distill / OPSD** (privileged-information distillation for multi-turn agents, arXiv 2602.04942) and **Structured Distillation of Web Agent Capabilities** (Agent-as-Annotators, arXiv 2604.07776) — strongest concrete agent-distillation items if you want to seed an "agent distillation" subcategory.

---

## 推荐补充字段 (Recommended Supplementary Fields)

- **H. Self-Modification Locus / Substrate**: What the method actually mutates — frozen-model + code/architecture (DGM, ADAS, Gödel Agent), prompts/textual params (GEPA, MASS), context/memory (ACE, AgentFly), or model weights (AZR, TTRL). Cluster 7's defining axis; cleanly separates "weight-free" vs "weight-update" self-improvement. (Aligns with the Fang survey's "which component evolves" and Gao survey's "what to evolve.")
- **I. Update Cadence / When-to-Evolve**: intra-test-time (TTRL, Memory-as-Action), inter-task/online (AgentFly, ACE), offline/batch (MASS, GEPA training). Directly mirrors the TMLR survey's "when to evolve" dimension and is currently not captured by A–G.
- **J. Optimization Signal Type**: scalar/verifiable reward (AZR, R-Zero, TTRL), textual/natural-language feedback ("verbal gradient" — GEPA, ACE, TEP), or hybrid (SiriuS, Skill1). The pivotal "textual policy optimization" distinction your topic centers on.
- **K. Human-Supervision / Data Dependency**: zero-data self-play (AZR, R-Zero), unlabeled-only (TTRL, MM-UPT), execution-feedback-only (ACE, GEPA), or labeled. Captures the "scalable beyond human data" claim that unifies the cluster.
- **L. Recursivity / Self-Acceleration**: whether improvements to the system improve its *ability to further improve* (DGM, Gödel Agent, AlphaEvolve-trains-itself) vs. one-shot downstream gains. The RSI / open-endedness lens — a key F (Significance) sub-question for Cluster 7.
- **M. Co-Evolution Structure**: single-agent, challenger–solver (R-Zero, AZR), generator–critic (SPC), multi-agent population/supernet (MaAS, EvoFlow), or agent–environment co-evolution. (Highlighted as the key emerging direction by the XMU "Systematic Survey" 2026.)
- **N. Safety / Containment Provisions**: sandboxing, human oversight, reward-hacking mitigation (DGM Appendix F, AZR). Increasingly mandatory for self-modifying systems; surveys devote dedicated sections to it.

---

## 推荐Survey锚点 (Recommended Survey Anchors — 3–5)

1. **A Comprehensive Survey of Self-Evolving AI Agents** (Fang, Peng, Zhang et al., arXiv **2508.07407**, 2025). Unified feedback-loop framework: *System Inputs, Agent System, Environment, Optimisers*; "Three Laws (Endure/Excel/Evolve)"; single-/multi-agent/domain-specific taxonomy. **Best primary taxonomy anchor.** https://arxiv.org/abs/2508.07407
2. **A Survey of Self-Evolving Agents: What, When, How, and Where to Evolve** (Gao, Geng, Hua et al., Princeton, arXiv **2507.21046**, **TMLR 2026**). Three-axis taxonomy (what/when/how) + benchmarks + safety; toward ASI. **Best for the field-axis fields (H/I/J above).** https://arxiv.org/abs/2507.21046
3. **Adaptation of Agentic AI: A Survey of Post-Training, Memory, and Skills** (arXiv **2512.16301**, late 2025/2026). Four-paradigm A1/A2/T1/T2 framework spanning agent vs. tool adaptation; explicitly unifies post-training + memory + skill libraries. **Best anchor for the memory-as-policy + distillation lanes.** https://arxiv.org/pdf/2512.16301
4. **Beyond Individual Intelligence: Collaboration, Failure Attribution, and Self-Evolution in LLM-MAS** (Qi, Ma et al., arXiv **2605.14892**, 2026). LIFE progression (Lay→Integrate→Find→Evolve). **Best multi-agent self-improvement anchor** (covers MaAS/MASS/SiriuS/Optima context). https://arxiv.org/abs/2605.14892
5. **A Survey of On-Policy Distillation for LLMs** (Song & Zheng, arXiv **2604.00626**, 2026) — if you add the agent-distillation sub-cluster; or **A Systematic Survey of Self-Evolving Agents: Model-Centric → Environment-Driven Co-Evolution** (XMU, ResearchGate/2026, repo: github.com/XMUDeepLIT/Awesome-Self-Evolving-Agents) as an alternative co-evolution-focused anchor.

Secondary/optional surveys worth noting: *SoK: Agentic Skills* (2602.20867) and *Agent Skills for LLMs* (2602.12430) for the skill-acquisition lane; *Lifelong Learning of LLM Agents: A Roadmap* (2501.07278) for continual-learning framing.

---

## 信息来源 (Sources)
- [A Comprehensive Survey of Self-Evolving AI Agents (2508.07407)](https://arxiv.org/abs/2508.07407)
- [A Survey of Self-Evolving Agents: What/When/How/Where (2507.21046, TMLR 2026)](https://arxiv.org/abs/2507.21046)
- [Adaptation of Agentic AI: Post-Training, Memory, Skills (2512.16301)](https://arxiv.org/pdf/2512.16301)
- [Beyond Individual Intelligence: MAS Self-Evolution survey (2605.14892)](https://arxiv.org/abs/2605.14892)
- [Darwin Gödel Machine (2505.22954)](https://arxiv.org/abs/2505.22954)
- [Gödel Agent (2410.04444, ACL 2025)](https://arxiv.org/abs/2410.04444)
- [ADAS / Meta Agent Search (2408.08435)](https://arxiv.org/abs/2408.08435)
- [GEPA (2507.19457, ICLR 2026 Oral)](https://openreview.net/forum?id=RQm2KQTM5r)
- [MaAS — Agentic Supernet (2502.04180, ICML 2025)](https://arxiv.org/abs/2502.04180)
- [MASS — Multi-Agent System Search (2502.02533)](https://arxiv.org/abs/2502.02533)
- [EvoFlow (2502.07373)](https://arxiv.org/abs/2502.07373)
- [MAPRO (2510.07475)](https://arxiv.org/abs/2510.07475)
- [TEP — Textual Equilibrium Propagation (2601.21064)](https://arxiv.org/abs/2601.21064)
- [Absolute Zero Reasoner (2505.03335)](https://arxiv.org/abs/2505.03335)
- [R-Zero (2508.05004, ICLR 2026)](https://openreview.net/forum?id=96apU6YzSO)
- [TTRL (NeurIPS 2025)](https://openreview.net/forum?id=VuVhgEiu20)
- [SPC — Self-Play Critic (2504.19162)](https://arxiv.org/abs/2504.19162)
- [AgentFly / Memento (2508.16153)](https://arxiv.org/abs/2508.16153)
- [ACE — Agentic Context Engineering (2510.04618)](https://arxiv.org/abs/2510.04618)
- [SiriuS (2502.04780)](https://arxiv.org/abs/2502.04780)
- [Optima (2410.08115, ACL 2025)](https://arxiv.org/abs/2410.08115)
- [AlphaEvolve (2506.13131)](https://arxiv.org/abs/2506.13131)
- [Era of Experience (Silver & Sutton, DeepMind 2025)](https://storage.googleapis.com/deepmind-media/Era-of-Experience%20/The%20Era%20of%20Experience%20Paper.pdf)
- [Agent Learning via Early Experience (ICLR 2026)](https://openreview.net/pdf?id=pEGnJbmSUy)
- [ELL framework + benchmark (2508.19005)](https://arxiv.org/pdf/2508.19005)
- [A Survey of On-Policy Distillation for LLMs (2604.00626)](https://arxiv.org/abs/2604.00626)
- [Awesome-Self-Evolving-Agents repo (XMU, taxonomy reference)](https://github.com/XMUDeepLIT/Awesome-Self-Evolving-Agents)

### Brief summary for the caller
The existing 5 Cluster-7 items are all valid but the framework was missing: (1) the **two anchor surveys** (Fang 2508.07407 and Gao/TMLR 2507.21046) plus 2–3 newer surveys; (2) the **self-modifying-code line** (DGM 2505.22954, Gödel Agent, ADAS); (3) the **memory-as-policy / skill-as-policy sub-cluster**, which is the most explosive 2025–2026 area (AgentFly/Memento, ACE, Memory-as-Action, Skill1, MemSkill, Skill-Pro, Mem-π); (4) the **zero-data self-play co-evolution** line (Absolute Zero, R-Zero, TTRL, SPC); (5) the **prompt+workflow co-optimization** family (GEPA — the standout, MaAS, MASS, EvoFlow, MAPRO, TEP); and (6) an **agent-distillation** lane. I also recommend 7 new fields (H–N), the most important being **Self-Modification Locus**, **Optimization Signal Type** (scalar vs textual), and **Recursivity/Self-Acceleration**, which are the dimensions that distinguish Cluster 7 from C1–C6. All arXiv IDs and venues above are verified.