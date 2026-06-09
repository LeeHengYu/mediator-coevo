I have comprehensive, cross-verified coverage. All major gap areas are confirmed against primary sources (arXiv/Nature/venue). Here is my gap-hunting report.

---

# GAP-HUNTER REPORT: What the 7-Cluster Framework (C1–C7) Missed

I ran 24+ searches/fetches across Semantic Scholar, Exa, and full-text survey scrape. The framework is strong on discrete prompt optimization, textual-gradient, reflection, and skill learning, but has **systematic blind spots** in five areas: (1) the *parameter-side* prompt-tuning lineage that C1 contrasts against; (2) **black-box/Bayesian/bandit** instruction optimization; (3) **self-training / self-play / zero-data co-evolution** (the fastest-growing 2024–2026 paradigm); (4) **multi-agent system optimization**; and (5) the **dedicated survey layer**. Most critically, the framework has **no survey anchors** despite 4–5 authoritative 2024–2026 surveys existing that define the exact taxonomy this framework is reinventing.

## Missing Items (grouped by target cluster)

### → C1 (APO) — the soft-prompt / continuous-prompt contrast lineage (entirely absent)
- **Prefix-Tuning** [C1/new sub-area]: Li & Liang, 2021. The foundational continuous-prompt method; the "white-box gradient" contrast to discrete APO. Any APO framing is incomplete without it.
- **P-Tuning / "GPT Understands, Too"** [C1]: Liu et al., 2021, AI Open (arXiv 2103.10385, ~1436 cites). Trainable continuous prompt embeddings; shows discrete prompts are unstable — the direct motivation for APO.
- **MIPRO / MIPROv2** [C1, **high priority**]: Opsahl-Ong et al., EMNLP 2024 (arXiv 2406.11695, ~212 cites). The DSPy optimizer that jointly optimizes instructions AND demonstrations for multi-stage LM programs. C3 lists "DSPy" but omits its actual optimization algorithm — a significant omission.
- **Demonstration selection & ordering optimization** [C1/new]: e.g., Long et al. 2024 (arXiv 2408.07505) self-selects/ranks/orders in-context examples via RL. An entire optimization axis (which examples, what order) is missing — the framework only covers instruction text.

### → C2 (Evolution) — extend
- **FunSearch** [C2, **high priority**]: Romera-Paredes et al., **Nature 2023** (~953 cites). The seminal LLM+evolutionary program search (cap-set, bin-packing). AlphaEvolve's direct predecessor and the canonical evolution anchor — its absence is the biggest single gap in C2.
- **GEPA** [C2, verify placement]: Agrawal et al., 2025 (arXiv 2507.19457, ~201 cites). Already listed in C2, but note its real identity: "Genetic-Pareto" reflective prompt evolution beating GRPO with 35x fewer rollouts — it actually bridges C2+C3+C4 and should be cross-referenced.
- **KTCE (Knowledge-grounded Tool Creation with Evolution)** [C2/C5]: Ma et al., AAAI 2025. Evolutionary search over toolsets (mutation/crossover) — evolution applied to tools, not prompts.

### → C5 (Skill learning) — major tool-creation sub-area missing
- **Toolformer** [C5/new]: Schick et al., 2023 (arXiv 2302.04761, ~4145 cites). Self-supervised tool-use learning — seminal, very high-cited, absent.
- **LATM (LLMs As Tool Makers)** [C5]: Cai et al., 2023 (arXiv 2305.17126, ~300 cites). Closed-loop tool *creation* and caching. Tool/skill *creation* (vs. skill *reuse* in Voyager) is an underrepresented axis.
- **CREATOR / CRAFT / Trove** [C5]: the tool-creation family flagged in the brief — confirmed as a distinct, uncovered sub-area.

### → C6 (RL policy) — self-training & reward-modeling lineage missing
- **STaR** [C6/new, **high priority**]: Zelikman et al., 2022 (arXiv 2203.14465, ~909 cites). The seminal bootstrapped self-improvement loop — foundational ancestor of nearly all "self-improvement" work; conspicuously absent.
- **ReST / ReST-EM** [C6]: Gulcehre et al. 2023 / Singh et al. 2023. Reinforced self-training via EM — the offline self-improvement counterpart to online RL.
- **Self-Rewarding Language Models** [C6, **high priority**]: Yuan et al., 2024 (arXiv 2401.10020, ~611 cites). LLM-as-judge providing its own reward in iterative DPO. C6 lists "Self-Rewarding LMs" by name but this is the canonical citation to anchor; also add **CREAM** (consistency-regularized variant, 2024).
- **Process Reward Models (PRMs)** [C6/new sub-area]: e.g., "Let's Verify Step by Step" (Lightman et al. 2023) + agentic PRMs (SWE-Shepherd 2026, DataPRM 2026). Step-level reward modeling for agents is a full sub-area with zero coverage.
- **Constitutional AI / RLAIF** [C6/new]: Bai et al., 2022 (arXiv 2212.08073, ~3109 cites). Self-critique-as-training — the seminal self-improvement-via-AI-feedback paper; very high-cited, absent.

### → C7 (Emerging 2025) — incomplete; missing self-play & multi-agent optimization
- **Self-Discover** [C3/C7]: Zhou et al., 2024 (arXiv 2402.03620, ~114 cites). Self-composed reasoning structures — CoT-structure optimization.
- **Buffer of Thoughts** [C5/C7]: Yang et al., NeurIPS 2024 (arXiv 2406.04271, ~105 cites). Meta-buffer of distilled thought-templates with a buffer-manager — a reasoning-template memory/optimization hybrid.
- **GPTSwarm** [new C8, **high priority**]: Zhuge et al., ICML 2024 Oral (arXiv 2402.16823). Language agents as optimizable graphs — node (prompt) + edge (topology) optimization. Foundational multi-agent optimization.
- **MaAS (Multi-agent Architecture Search via Agentic Supernet)** [new C8]: Zhang et al., ICML 2025 (arXiv 2502.04180, ~119 cites). Query-dependent agentic-architecture sampling.
- **MASS (Multi-Agent System Search)** [new C8]: Google, 2025 (arXiv 2502.02533). Staged prompt+topology optimization for MAS.

## Proposed NEW clusters/sub-areas

**C8 — Multi-agent system / topology optimization** (most clear-cut new cluster)
Optimizing *the system* (agent connectivity, role assignment, workflow) rather than a single prompt/policy. The framework's ADAS/AFlow (in C5) and AgentSquare (C7) actually belong to this emerging cluster. Representatives: **GPTSwarm, MaAS, MASS, DyLAN, Archon** (plus ADAS/AFlow recharacterized).

**C9 — Self-play & zero-data co-evolution / automatic curriculum** (fastest-growing 2025–2026 paradigm; ~zero coverage)
A challenger/solver (or proposer/solver) co-evolves, generating its own training curriculum without human data. Representatives: **Absolute Zero / AZR** (Zhao et al. 2025, arXiv 2505.03335), **R-Zero** (Huang et al. 2025, arXiv 2508.05004), **Self-Challenging Agents (SCA)** (Meta 2025, arXiv 2506.01716), **Agent0** (2025, arXiv 2511.16043), **AgentEvolver** (2025, arXiv 2511.10395), **SPIN (Self-Play Fine-Tuning)** (Chen et al. 2024, arXiv 2401.01335), **Eurekaverse** (LLM environment-curriculum generation). This is arguably the single most important missing cluster.

**C10 — Black-box / Bayesian / bandit instruction optimization** (could fold into C1 but is methodologically distinct — gradient-free, query-efficient)
Representatives: **Black-Box Tuning (BBT/BBTv2)** (Sun et al. 2022, arXiv 2201.03514, ~349 cites), **InstructZero** (Chen et al. 2023, arXiv 2306.03082), **INSTINCT** (Lin et al. ICML 2024, arXiv 2310.02905), **BOInG** (2024), Black-Box Prompt Learning for VLMs.

**C11 — Self-referential / recursively self-modifying agents** (distinct from skill learning)
The agent rewrites its *own code/scaffold*. Representatives: **STOP (Self-Taught Optimizer)** (Zelikman et al. 2023, arXiv 2310.02304, ~96 cites), **Darwin-Gödel Machine** (Zhang/Hu/Lu/Clune 2025, arXiv 2505.22954, ~110 cites — open-ended self-improving coding agents, SWE-bench 20%→50%), **Gödel Agent**. The brief flagged this; it is confirmed as a coherent, distinct cluster.

## Missing SURVEY layer (the framework has none — high-priority structural gap)
- **"A Survey on Self-Evolution of LLMs"** — Tao et al. (Alibaba), 2024 (arXiv 2404.14387). 4-phase loop (acquisition/refinement/updating/evaluation).
- **"A Survey of Self-Evolving Agents: What, When, How, Where"** — Gao et al. (Princeton/Tsinghua), TMLR 2026 (arXiv 2507.21046). Its taxonomy (What: models/context/tools/architecture; How: reward-based / imitation / population-evolutionary; When: intra/inter-test-time) is essentially the supergraph of this entire framework — the single best organizing reference.
- **"A Comprehensive Survey of Self-Evolving AI Agents"** — Fang et al., 2025. Four-component loop (Inputs/Agent/Environment/Optimisers).
- **"Lifelong Learning of LLM-based Agents: A Roadmap"** — Zheng et al., 2025 (arXiv 2501.07278).
- **"A Survey of Automatic Prompt Optimization (heuristic search)"** — Cui et al., 2025 (arXiv 2502.18746). Directly covers C1's taxonomy (where/what/criteria/operators/search).

## 推荐补充字段 (recommended new schema fields)
- **optimization_target** [what is optimized: discrete-prompt / soft-prompt / demonstrations / topology / weights / tools / curriculum]: the framework conflates these; the surveys' #1 organizing axis.
- **gradient_access** [white-box / black-box-API / textual-gradient / gradient-free-BO]: cleanly separates C1/C3/C10 lineages that currently blur together.
- **update_signal** [scalar-reward / verbal-feedback / AI-feedback(RLAIF) / verifiable-reward / self-generated]: distinguishes RL (C6) from reflective (C4) from self-play (C9).
- **when_to_evolve** [intra-test-time / inter-test-time / offline-training]: the survey's "When" axis; the framework has no temporal dimension.
- **data_dependence** [human-labeled / self-generated / zero-data]: surfaces the C9 self-play/zero-data frontier.
- **single_vs_multi_agent** [single / multi-agent]: needed to justify the proposed C8.

## 信息来源
- [A Survey of Self-Evolving Agents (Gao et al., TMLR 2026)](https://arxiv.org/html/2507.21046v4)
- [A Survey on Self-Evolution of LLMs (Tao et al., 2024)](https://arxiv.org/html/2404.14387)
- [A Comprehensive Survey of Self-Evolving AI Agents (Fang et al., 2025)](https://x-izhang.github.io/publication/fang-2025-comprehensivesurveyselfevolvingai/)
- [Lifelong Learning of LLM-based Agents: Roadmap (2025)](https://arxiv.org/html/2501.07278)
- [Survey of Automatic Prompt Optimization (Cui et al., 2025)](https://arxiv.org/abs/2502.18746)
- [FunSearch — Nature 2023](https://doi.org/10.1038/s41586-023-06924-6) · [Darwin-Gödel Machine](https://arxiv.org/abs/2505.22954) · [STOP](https://arxiv.org/abs/2310.02304) · [STaR](https://arxiv.org/abs/2203.14465) · [Constitutional AI](https://arxiv.org/abs/2212.08073) · [Self-Rewarding LMs](https://arxiv.org/abs/2401.10020)
- [Absolute Zero](https://arxiv.org/abs/2505.03335) · [R-Zero](https://arxiv.org/html/2508.05004) · [Self-Challenging Agents](https://arxiv.org/html/2506.01716) · [SPIN](https://arxiv.org/abs/2401.01335)
- [GPTSwarm — ICML 2024](https://arxiv.org/html/2402.16823) · [MaAS](https://arxiv.org/abs/2502.04180) · [MASS](https://arxiv.org/pdf/2502.02533)
- [BBT — LMaaS](https://arxiv.org/abs/2201.03514) · [InstructZero](https://arxiv.org/abs/2306.03082) · [INSTINCT](https://arxiv.org/abs/2310.02905)
- [Toolformer](https://arxiv.org/abs/2302.04761) · [LATM](https://arxiv.org/abs/2305.17126) · [MIPRO](https://arxiv.org/abs/2406.11695) · [Prefix-Tuning context / P-Tuning](https://arxiv.org/abs/2103.10385) · [Self-Discover](https://arxiv.org/abs/2402.03620) · [Buffer of Thoughts](https://arxiv.org/abs/2406.04271)

**Top 5 must-add (by impact):** (1) the **survey layer** (Gao 2026 + Tao 2024) as the framework's missing organizing backbone; (2) **C9 self-play/zero-data** cluster (Absolute Zero, R-Zero, Self-Challenging); (3) **STaR + Constitutional AI** as the seminal self-improvement ancestors in C6; (4) **FunSearch** in C2; (5) **C8 multi-agent optimization** (GPTSwarm/MaAS/MASS), which also recharacterizes the misplaced ADAS/AFlow/AgentSquare items.

Note: The local `related-literature/llm-agent-memory/outline.yaml` I found is a *memory*-themed outline (different project), not the C1–C7 prompt/agent self-improvement framework in the brief — so this report is built entirely against the framework as stated in the task, not that file.