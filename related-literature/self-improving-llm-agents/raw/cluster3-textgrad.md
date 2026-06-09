I have all the confirmations I need. TSGD-M (Ding et al. 2025), mmGRPO (Ziems/Soylu/Khattab 2025, ACM AI&Agentic Systems), and SAMMO (Schnabel & Neville 2024, Microsoft, structure-aware symbolic prompt program search — a direct DSPy-adjacent declarative compiler) are all confirmed. I've done 15+ distinct searches/fetches cross-verified against arXiv, ACL, OpenReview, GitHub, and the compound-AI-optimization survey. Let me compile the final structured results.

---

# Cluster 3 — Textual Gradient / Feedback "Backprop" / Declarative Prompt-Program Compilers

## Validation of existing items
The 4 already-listed anchors are correctly the foundational pillars and all verified against primary sources:
- **TextGrad** (Yuksekgonul 2024, arXiv:2406.07496; Nature 639:609–616, 2025) — confirmed.
- **Trace/OptoPrime** (Cheng, Nie, Swaminathan, NeurIPS 2024, arXiv:2406.16218) — confirmed; OPTO formalism + execution-trace-as-gradient.
- **DSPy** (Khattab 2023, arXiv:2310.03714) + **MIPROv2** (the underlying paper is **MIPRO**, Opsahl-Ong 2024, EMNLP, arXiv:2406.11695, 212 cites), COPRO, BootstrapFewShot — confirmed.
- **AdalFlow** — its core research artifact is **LLM-AutoDiff** (see below); worth listing the paper explicitly.

**Key gap found:** the framework is missing the entire 2024–2026 "successor wave" of textual-gradient credit-assignment refinements (GASO, REVOLVE, LLM-AutoDiff, AIME, TSGD-M, TEP, TextResNet), the DSPy team's own major follow-ups (BetterTogether, LeReT, GEPA, mmGRPO), the meta-optimizer layer (metaTextGrad), the symbolic-network agent compilers (Agent Symbolic Learning, Agentic Neural Networks, HiVA), and the compile-time symbolic-search compiler **SAMMO**. GEPA in particular is a glaring omission (201 citations, bridges this cluster with C2-evolution and C6-RL).

---

## 补充 Items (supplementary items)

**DSPy-team / declarative-compiler lineage**
- **MIPRO / MIPROv2**: Opsahl-Ong, Ryan, Purtell, Broman, Potts, Zaharia, Khattab — *Optimizing Instructions and Demonstrations for Multi-Stage LM Programs*, EMNLP 2024 (arXiv:2406.11695). Surrogate-model + Bayesian meta-optimization for joint instruction+demo optimization with cross-module credit assignment. The actual paper behind the listed "MIPROv2." 212 cites.
- **GEPA (Genetic-Pareto)**: Agrawal, Tan, Soylu, Ziems, …, Klein, Zaharia, Khattab, 2025 (arXiv:2507.19457). Reflective prompt evolution using NL reflection over sampled trajectories + Pareto-frontier of attempts; beats GRPO by ~6–20% with up to 35x fewer rollouts, beats MIPROv2 by >10%. **High-priority addition** — directly bridges C3↔C2(evolution)↔C6(RL). 201 cites. Code: gepa-ai/gepa.
- **BetterTogether**: Soylu, Potts, Khattab — *Fine-Tuning and Prompt Optimization: Two Great Steps that Work Better Together*, EMNLP 2024 (arXiv:2407.10930). Alternates prompt-opt and weight-opt (LM teaches itself); +60% over weights-only, +6% over prompts-only. Released as a DSPy meta-optimizer.
- **mmGRPO (multi-module GRPO)**: Ziems, Soylu, Agrawal, …, Potts, Khattab — *Composing Policy Gradients and Prompt Optimization for LM Programs*, 2025, ACM Conf. on AI & Agentic Systems (arXiv:2508.04660). Module-level GRPO that composes with MIPROv2 à la BetterTogether (+11% avg). Bridges C3↔C6.
- **LeReT (Learning to Retrieve by Trying)**: Hsu, Khattab, Finn, Sharma — *Grounding by Trying*, ICLR 2025 (arXiv:2410.23214). DSPy-built RL framework (IPO/preference-based) that diversifies few-shot prompts to learn better retrieval queries; +29% retrieval, +17% downstream. C3↔C6 boundary item.

**TextGrad successor wave (textual-gradient credit-assignment refinements)**
- **LLM-AutoDiff**: Li Yin & Zhangyang Wang, 2025 (arXiv:2501.16673). The **AdalFlow** research paper. Extends textual gradients to multi-component, *cyclic* graphs; introduces time-sequential gradients (for repeated/looped nodes), functional-node support, skip-connections, selective gradient computation. 17 cites.
- **GASO / Semantic Backpropagation**: Wang, Alyahya, Ashley, Serikov, Khizbullin, Faccio, Schmidhuber — *How to Correctly Do Semantic Backpropagation on Language-based Agentic Systems*, 2024 (arXiv:2412.03624). Formalizes "semantic gradients" generalizing reverse-mode autodiff + TextGrad; fixes neglected sibling-input interactions via "semantic gradient descent"; beats TextGrad/OptoPrime/COPRO on BBH & GSM8K. Repo: HishamAlyahya/semantic_backprop.
- **REVOLVE**: P. Zhang, Jin, Hu, Li, Kang, Luo, Song, H. Wang, 2024 (arXiv:2412.03092), ICML 2025. "Second-order"/curvature-aware analog: tracks how *Responses EVOLVE* across iterations (concise history) to escape stagnation/oscillation in first-order TextGrad. +7.8% prompt / +20.7% solution / +29.2% code, fewer iterations.
- **AIME**: Patel et al., 2024 (arXiv:2410.03131). *AI System Optimization via Multiple LLM Evaluators* — concatenating multiple evaluator-LLM outputs in the TextGrad backward pass catches code errors a single evaluator misses. Targets the evaluator/loss module of the textual-gradient loop.
- **TSGD-M**: Ding, Hong, J.T. Wang, Z. Lin, Z. Wang, Y. Chen — *Scaling Textual Gradients via Sampling-Based Momentum*, 2025 (arXiv:2506.00400). Textual SGD with momentum: reweights updates via bootstrapped minibatch-validation importance weights over historical prompts; Gumbel-Top-k sampling. Framework-agnostic (integrates into TextGrad, DSPy-COPRO, AdalFlow). Addresses the "context wall" when scaling training data.
- **metaTextGrad**: zou-group, NeurIPS 2025 (OpenReview 10s01YrlKp; repo zou-group/metatextgrad). **Meta-optimizer that optimizes the LLM optimizers themselves** (meta prompt optimizer + meta structure optimizer); tunes TextGrad/ADAS/DSPy optimizer prompts per-task, +up to 6%. Novel "optimize-the-optimizer" layer not represented in the framework.
- **TEP (Textual Equilibrium Propagation)**: M. Chen, Deng, Zou, Yu, Li, 2026 (arXiv:2601.21064). Diagnoses *exploding/vanishing textual gradients* in deep compound systems; replaces global backprop with local equilibrium-propagation-style (free + nudged phase) bounded updates. Repo: MinghuiChen43/TEP.
- **TextResNet**: 2026 (arXiv:2602.08306). Residual-tuning analog for textual gradients: additive semantic deltas (identity highway) + semantic-gradient decomposition via a "Semantic Projector" + causal routing to fix "Semantic Entanglement/Attribution Ambiguity" in deep chains.
- **ADPO (Adaptive Dependency-aware Prompt Optimization)**: 2025 (arXiv:2512.24933). Critiques MIPRO/TextGrad/Trace's fixed acyclic decomposition; dynamically reassigns module responsibilities and handles loops.

**Symbolic-network / agent-graph compilers (textual backprop over whole agent systems)**
- **Agent Symbolic Learning**: Zhou et al. (AIWaves), 2024 (arXiv:2406.18532), "Symbolic Learning Enables Self-Evolving Agents." Treats agent as a symbolic network; NL "language loss → language gradients → symbolic optimizers" (PromptOptimizer/ToolOptimizer/PipelineOptimizer) — jointly optimizes prompts, tools, *and pipeline topology*. Direct precursor to backprop-over-agent-systems.
- **Agentic Neural Networks (ANN)**: Ma, Lin, Y. Zhang, Tresp, Y. Ma, 2025 (arXiv:2506.09046). Layered multi-agent "neural network"; forward = task decomposition into agent teams, backward = textual backprop refining roles/prompts/coordination. C3↔C2 multi-agent boundary.
- **HiVA (Hierarchical Variable Agent)**: Tang, Zhang, Lv, …, K. Wang, 2025 (arXiv:2509.00189). Semantic-Topological Evolution (STEV): uses textual gradients as discrete-domain surrogates for backprop to co-evolve *both* node semantics and graph topology; multi-armed-bandit forward routing.
- **Symbolic-MoE / Skill-MoE**: J. Chen, Yun, Stengel-Eskin, T. Chen, Bansal, 2025 (arXiv:2503.05641). *Skill-Based Mixture-of-Experts* — symbolic, gradient-free, skill-inferred instance-level expert routing + aggregator selection (the "Symbolic-MoE" the brief references). +8.15% avg; boundary with C5 skill libraries.
- **SAMMO**: Schnabel & Neville (Microsoft), 2024 (arXiv:2404.02319). *Symbolic Prompt Program Search* — compile-time optimization representing prompts as symbolic prompt programs (SPP, abstract program graphs) searched via mutation operators; explicitly generalizes DSPy and subsumes specialized prompt tuners. The main non-DSPy declarative-compiler competitor; covers instruction tuning, RAG-pipeline tuning, prompt compression.

**Survey to cite as landscape anchor**
- **Compound AI Systems Optimization: A Survey** (Lee, Yi, Liu, Lu, Yang, Y.-N. Chen, NTU), 2025 (arXiv:2506.08234; repo MiuLab/AISysOpt-Survey). 26 representative works in a 2×2 taxonomy; the source for the field dimensions below.

---

## 推荐补充字段 (recommended additional fields)

These extend the existing C-Method block, motivated by distinctions that *actually separate* the items above (verified against the NTU survey's four principled dimensions + the recurring axes that differentiate successor papers):

- **structural_flexibility** (Fixed vs. Flexible structure): Does the method optimize only node parameters on a *fixed* graph (TextGrad, MIPRO, REVOLVE, AIME, TSGD-M) or can it also mutate *topology/pipeline structure* (Agent Symbolic Learning, HiVA, SAMMO, ANN, ADPO)? This is the single biggest discriminator in the field and the survey's primary taxonomy axis. Currently the framework has no field capturing "can it change the graph, not just the prompts."

- **gradient_order / history_usage** (first-order vs. curvature/momentum-aware): Whether the textual update uses only immediate feedback (vanilla TextGrad/OptoPrime) or incorporates optimization *history* — second-order/curvature (REVOLVE), momentum (TSGD-M), or trajectory memory. Distinguishes a whole sub-lineage that the current "update_operator" field flattens.

- **credit_assignment_mechanism** (how blame is routed across modules): single-node / chain backprop (TextGrad) vs. sibling-interaction-aware semantic gradients (GASO) vs. time-sequential accumulation for looped nodes (LLM-AutoDiff) vs. local-equilibrium/bounded (TEP) vs. causal routing/decomposition (TextResNet) vs. surrogate-model meta-opt (MIPRO). The brief lists "credit-assignment" as a sub-field of C; recommend promoting it to a first-class field with these enumerated values since it is the core technical contribution of most successor papers.

- **handles_cyclic_loops** (acyclic-only vs. loops/repeated-node support): Most early methods (TextGrad, Trace, MIPRO) assume acyclic/single-pass graphs; LLM-AutoDiff, ADPO, agentic systems explicitly support cycles/multi-hop loops. Important boolean for agent applicability.

- **optimizer_is_meta_optimized** (is the optimizer itself optimized?): Captures the new meta-layer — metaTextGrad optimizes the optimizer's own prompts/structure; GEPA reflects on its own attempts. Distinguishes object-level vs. meta-level self-improvement.

- **depth_scalability / failure_modes** (behavior as system depth grows): Whether the paper characterizes or mitigates exploding/vanishing textual gradients, context-wall, semantic entanglement (TEP, TextResNet, TSGD-M explicitly study this; foundational methods do not). Useful "G. Uncertain"/limitations-adjacent field.

- **composes_with_weight_optimization** (PO-only vs. PO+RL/SFT hybrid): Whether the method is prompt-only or designed to interleave with weight tuning / RL (BetterTogether, mmGRPO, GEPA-vs-GRPO framing). Marks the C3↔C6 boundary explicitly so reviewers can see hybridization.

- **shared_optimizer_LLM_vs_separate_roles** (optimizer-LLM role detail): Whether evaluator, gradient-estimator, and optimizer are the same frozen LLM or distinct (TextGrad uses separate evaluator+gradient+optimizer roles; AIME multiplies the *evaluator*; TEP defaults critic = node model). Refines the existing "optimizer-LLM role" field with the standard 3-role decomposition.

---

## 信息来源 (sources)
- TextGrad — [arXiv:2406.07496](https://arxiv.org/abs/2406.07496) / [Nature s41586-025-08661-4](https://www.nature.com/articles/s41586-025-08661-4) / [zou-group/textgrad](https://github.com/zou-group/textgrad)
- Trace/OptoPrime — [arXiv:2406.16218](https://arxiv.org/abs/2406.16218) / [microsoft.github.io/Trace](https://microsoft.github.io/Trace/)
- DSPy — [arXiv:2310.03714](https://arxiv.org/abs/2310.03714); MIPRO — [arXiv:2406.11695](https://arxiv.org/abs/2406.11695) (EMNLP 2024)
- GEPA — [arXiv:2507.19457](https://arxiv.org/abs/2507.19457) / [gepa-ai/gepa](https://github.com/gepa-ai/gepa)
- BetterTogether — [arXiv:2407.10930](https://arxiv.org/abs/2407.10930) / [ACL 2024.emnlp-main.597](https://aclanthology.org/2024.emnlp-main.597/) / [DSPy docs](https://dspy.ai/api/optimizers/BetterTogether/)
- mmGRPO — [arXiv:2508.04660](https://arxiv.org/abs/2508.04660)
- LeReT — [arXiv:2410.23214](https://arxiv.org/abs/2410.23214) / [sherylhsu.com/LeReT](https://sherylhsu.com/LeReT/) / [sher222/LeReT](https://github.com/sher222/LeReT)
- LLM-AutoDiff (AdalFlow) — [arXiv:2501.16673](https://arxiv.org/abs/2501.16673)
- Semantic Backpropagation / GASO — [arXiv:2412.03624](https://arxiv.org/abs/2412.03624) / [HishamAlyahya/semantic_backprop](https://github.com/HishamAlyahya/semantic_backprop)
- REVOLVE — [arXiv:2412.03092](https://arxiv.org/abs/2412.03092) / [PMLR v267](https://proceedings.mlr.press/v267/zhang25aj.html) / [llm-revolve.netlify.app](https://llm-revolve.app)
- AIME — [arXiv:2410.03131](https://arxiv.org/html/2410.03131v3)
- TSGD-M — [arXiv:2506.00400](https://arxiv.org/abs/2506.00400)
- metaTextGrad — [OpenReview 10s01YrlKp](https://openreview.net/pdf?id=10s01YrlKp) (NeurIPS 2025) / [zou-group/metatextgrad](https://github.com/zou-group/metatextgrad)
- TEP — [arXiv:2601.21064](https://arxiv.org/abs/2601.21064) / [MinghuiChen43/TEP](https://github.com/MinghuiChen43/TEP)
- TextResNet — [arXiv:2602.08306](https://arxiv.org/html/2602.08306v1)
- ADPO — [arXiv:2512.24933](https://arxiv.org/pdf/2512.24933)
- Agent Symbolic Learning — [arXiv:2406.18532](https://arxiv.org/abs/2406.18532) / [aiwaves-cn.github.io/agents](https://aiwaves-cn.github.io/agents/)
- Agentic Neural Networks — [arXiv:2506.09046](https://arxiv.org/abs/2506.09046)
- HiVA — [arXiv:2509.00189](https://arxiv.org/abs/2509.00189)
- Symbolic-MoE / Skill-MoE — [arXiv:2503.05641](https://arxiv.org/abs/2503.05641)
- SAMMO — [arXiv:2404.02319](https://arxiv.org/abs/2404.02319) / [Microsoft Research](https://www.microsoft.com/en-us/research/publication/prompts-as-programs-a-structure-aware-approach-to-efficient-compile-time-prompt-optimization/)
- Compound AI Systems Optimization Survey — [arXiv:2506.08234](https://arxiv.org/html/2506.08234v1) / [MiuLab/AISysOpt-Survey](https://github.com/MiuLab/AISysOpt-Survey)

---

**Note on overlaps with other clusters (to coordinate, not duplicate):** GEPA, BetterTogether, mmGRPO, LeReT sit on the C3↔C6(RL)/C2(evolution) boundary — recommend they live in C3 (their mechanism is prompt/textual optimization of programs) with cross-references. Symbolic-MoE borders C5(skill libraries). The 2026 items (TEP, TextResNet, ADPO, "Hierarchical multi-robot," "Compiled AI") appear in the index but are very recent/low-citation — flag as "G. Uncertain" maturity. metaTextGrad and TSGD-M are the highest-confidence net-new additions alongside GEPA, GASO, REVOLVE, LLM-AutoDiff, and SAMMO.