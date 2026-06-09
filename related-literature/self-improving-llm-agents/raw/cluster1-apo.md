I now have comprehensive, primary-source-verified coverage. I've done 14+ distinct searches/fetches across academic-search, exa, and firecrawl. MIPROv2 belongs in Cluster 3 (DSPy family). Note: GPO here is Li 2023 "Generative Prompt Optimization" (instruction induction via clustering), distinct from the GRPO/policy methods. I have enough to synthesize the final answer with confidence.

---

# Cluster 1 — Automatic Prompt Optimization: Validation & Supplement

## Validation of Existing Items
The 10 existing items are all correctly placed in Cluster 1 and well-chosen as canonical anchors. However, I found that the cluster has **significant gaps**, especially in three sub-lineages: (a) the **foundational black-box/RL discrete prompt tuning** line (2022), (b) the **search/planning-based optimizers** (MCTS, bandits, beam search), and (c) the **reflection-then-rewrite** optimizers that are now treated as core APO baselines in every 2025 survey. The authoritative reference for this validation is the **AWS "Systematic Survey of Automatic Prompt Optimization" (Ramnath et al. 2025, EMNLP)**, whose 5-part taxonomy I used as the cross-check.

## 补充 Items (recommended additions to Cluster 1)

**High priority — canonical methods missing, cited as core APO baselines everywhere:**

- **PromptAgent (Wang et al. 2023)**: MCTS-based "strategic planning" view of prompt optimization; induces expert-level prompts via error reflection + simulated rewards. ~238 citations, a top-3 most-cited APO baseline alongside APE/OPRO. Should arguably be a headline item. arXiv:2310.16427 (ICLR 2024). https://arxiv.org/abs/2310.16427
- **PE2 — "Prompt Engineering a Prompt Engineer" (Ye, Pryzant et al. 2023)**: Improves meta-prompt design with detailed descriptions, context spec, and step-by-step reasoning templates; direct successor to APE/ProTeGi. ~125 citations, ACL 2024. arXiv:2311.05661. https://arxiv.org/abs/2311.05661
- **BDPL — Black-box Discrete Prompt Learning (Diao et al. 2022)**: Foundational gradient-free discrete prompt optimization using a variance-reduced policy gradient (VR-PGE) over categorical distributions; the canonical "black-box, API-only, RL" anchor that predates/parallels RLPrompt. TMLR. arXiv:2201.08531. https://arxiv.org/abs/2201.08531
- **BBT — Black-Box Tuning for LMaaS (Sun et al. 2022, ICML)**: Defines the "Language-Model-as-a-Service" black-box prompt setting; derivative-free (CMA-ES) optimization in a low-dim subspace. ~349 citations — the most-cited black-box prompt paper and the origin of the "LMaaS" framing your task statement uses. arXiv:2201.03514. https://arxiv.org/abs/2201.03514

**Medium priority — distinct mechanisms / strengthen coverage:**

- **GReaTer (Das et al. 2024, Salesforce)**: "Gradients over Reasoning" — uses task-loss gradients over reasoning tokens so that *small open-source* models can self-optimize prompts without a large closed-source optimizer LLM. Fills the "white-box gradient for discrete prompts" cell. arXiv:2412.09722. https://arxiv.org/abs/2412.09722
- **StablePrompt (Kwon et al. 2024, EMNLP)**: RL-based prompt tuning with Adaptive Proximal Policy Optimization (APPO) + anchor model for training stability; SOTA across classification/QA/generation; includes input-dependent variant TTE-StablePrompt. arXiv:2410.07652. https://arxiv.org/abs/2410.07652
- **DLN — Deep Language Networks (Sordoni et al. 2023, NeurIPS)**: Joint prompt optimization of *stacked* LLM layers via variational inference (treating intermediate output as a latent variable). Bridges APO and multi-module/compound-system optimization. arXiv:2306.12509. https://arxiv.org/abs/2306.12509
- **Plum (Pan et al. 2023)**: Metaheuristics for prompt learning (hill-climbing, simulated annealing, GA, tabu/harmony search) — establishes the "general, automatic, discrete, black-box, gradient-free, interpretable" criteria that define the cluster. arXiv:2311.08364. https://github.com/research4pan/Plum
- **CAPO — Cost-Aware Prompt Optimization (Zehle et al. 2025)**: GA-based optimizer (builds on EvoPromptGA) adding racing/early-stopping + length penalty for compute efficiency; jointly optimizes instructions + few-shot like PromptWizard. Cited as current SOTA discrete optimizer in the promptolution framework. arXiv:2504.16005. https://arxiv.org/abs/2504.16005

**Lower priority — niche but taxonomically useful (each fills a distinct cell in the APO survey):**

- **PACE (Dong et al. 2024)**: Actor-critic RL framing of prompt *editing* itself. arXiv exists; AAAI.
- **StraGo (Wu et al. 2024)**: Strategic guidance from both correct AND incorrect predictions as feedback (vs. ProTeGi's error-only).
- **PREFER (Zhang et al. 2024, AAAI)**: Feedback-reflect-refine with prompt **ensembling** / boosting.
- **SPRIG (Zhang et al. 2024)**: Optimizes *system* prompts from a 300-component corpus (roles, styles, CoT) via token-level genetic edits.
- **SCULPT (Kumar et al. 2024)**: Hierarchical tree structure for tuning *long, unstructured* prompts.
- **GPO / "Generative Prompt Optimization" (Li et al. 2023)**: Cluster-specific instruction induction + majority-vote ensemble. (Note: disambiguate from GRPO in Cluster 6 — name collision risk.)
- **EvoPrompt (Guo et al. 2023)**: Currently in Cluster 2, but be aware every APO survey classifies it as a core Cluster-1 discrete optimizer — flag the Cluster 1/2 boundary.

## 推荐补充字段 (recommended new fields)

The AWS APO survey's 5-part framework reveals several dimensions your current A–G schema does not capture:

- **seed_prompt_source**: How the initial/seed prompt is obtained — manually-created vs. instruction-induction-from-examples vs. from task-README/description vs. empty/from-scratch. (Core axis #1 of the APO taxonomy; sharply separates APE/GPO from ProTeGi/OPRO.)
- **candidate_generation_operator**: The specific operator producing new candidates — LLM-rewriter, metaprompt-resampling, GA mutation+crossover (token vs. sentence level), RL-trained generator, metaheuristic (hill-climb/annealing/PSO). (Axis #4; finer-grained than your current `update_operator`.)
- **search_strategy / filter_prune_step**: The iterative search controller over candidates — greedy/TopK, beam search, MCTS, UCB/bandit, Bayesian/TPE, evolutionary population. (Axis #5; currently conflated under "paradigm" — separating it cleanly distinguishes PromptAgent (MCTS) from ProTeGi (beam) from EvoPrompt (GA).)
- **optimization_criteria**: What is optimized beyond task accuracy — explicitly capturing multi-objective signals: prompt length/cost, perplexity/fluency, robustness/transferability, entropy-based scores. (The "what criteria" axis; needed to place CAPO, PIN, CLAPS.)
- **api_call_budget / query_efficiency**: Quantified inference-call or rollout budget (e.g., GEPA "35x fewer rollouts," CAPO racing, BBT API-bounded). Strengthens your field D "sample/compute efficiency" with a concrete, comparable metric — this is the single most-emphasized differentiator in 2025 papers.
- **optimizer_LLM vs target_LLM (decoupled)**: Whether the meta/optimizer LLM differs from the task LLM, and whether a *small* model can drive optimization (GReaTer's whole thesis; PromptWizard degrades on Llama3-8B). Your field C has "optimizer-LLM role" but does not capture the size-asymmetry / self-optimization distinction.
- **transferability_evidence**: Cross-model / cross-task prompt transfer claims (BDPL, GReaTer, FedDTPT, GRACE all report this explicitly). Refines field D "transfer" into an evidence-backed field.
- **system_vs_user_prompt_target**: Whether the method optimizes the system prompt, the user/task instruction, few-shot exemplars, or jointly (SPRIG=system; MIPROv2/PromptWizard=joint instruction+demos). Refines field B "optimization_target."

## 信息来源

- [A Systematic Survey of Automatic Prompt Optimization Techniques — Ramnath et al. (AWS), EMNLP 2025](https://arxiv.org/abs/2502.16923) (primary taxonomy source)
- [A Survey of Automatic Prompt Engineering: An Optimization Perspective (2025)](https://arxiv.org/pdf/2502.11560)
- [PromptAgent (Wang et al. 2023)](https://arxiv.org/abs/2310.16427)
- [PE2 — Prompt Engineering a Prompt Engineer (Ye et al. 2023)](https://arxiv.org/abs/2311.05661)
- [BDPL — Black-box Prompt Learning (Diao et al. 2022, TMLR)](https://arxiv.org/abs/2201.08531) · [OpenReview](https://openreview.net/forum?id=IvsGP7xRvm)
- [BBT — Black-Box Tuning for LMaaS (Sun et al. 2022, ICML)](https://arxiv.org/abs/2201.03514)
- [GReaTer (Das et al. 2024)](https://arxiv.org/abs/2412.09722) · [code](https://github.com/psunlpgroup/GreaTer)
- [StablePrompt (Kwon et al. 2024, EMNLP)](https://arxiv.org/abs/2410.07652)
- [DLN — Stacked LLMs via Variational Inference (Sordoni et al. 2023, NeurIPS)](https://arxiv.org/abs/2306.12509)
- [Plum: Prompt Learning using Metaheuristic (Pan et al. 2023)](https://arxiv.org/abs/2311.08364)
- [CAPO: Cost-Aware Prompt Optimization (Zehle et al. 2025)](https://arxiv.org/pdf/2504.16005)
- [GEPA: Reflective Prompt Evolution (Agrawal et al. 2025)](https://arxiv.org/abs/2507.19457) (context for efficiency/rollout-budget field)
- [Swarm-Prompt: PSO meets LLMs (Yu 2025)](https://doi.org/10.1109/CCAI65422.2025.11189827)
- [Genetic Prompt Search via LM Probabilities (Zhao et al. 2023, IJCAI)](https://www.ijcai.org/proceedings/2023/0588.pdf)
- [EvoPrompt (Guo et al. 2023)](https://arxiv.org/html/2309.08532v3) (Cluster 1/2 boundary note)

### Key findings summary
1. **4 high-priority additions** that any reviewer would expect in this cluster: PromptAgent, PE2, BDPL, BBT (the last two are the missing 2022 foundational black-box/RL anchors).
2. The cluster's strongest organizing principle is the **AWS APO 5-part taxonomy** (seed prompt → candidate generation operator → optimization criteria → filter/prune search strategy → inference evaluation) — I recommend adopting its axes as new fields, since they expose the dimensions (search strategy, query budget, optimizer/target size-asymmetry) that the current schema collapses.
3. A **naming-collision caution**: "GPO" (Generative Prompt Optimization, Li 2023, Cluster 1) vs. "GRPO" (Cluster 6) — disambiguate in the items list.

<!-- ===== SECONDARY BLOCK ===== -->

### 补充Items (new/missing Cluster 1 items only)
- **BBT — Black-Box Tuning for LMaaS** (2022, Sun/Shao/Qian/Huang/Qiu, ICML): Origin of the "Language-Model-as-a-Service" black-box framing; derivative-free CMA-ES in a low-dim subspace. ~349 cites, the most-cited black-box prompt paper. https://arxiv.org/abs/2201.03514
- **BDPL — Black-box Discrete Prompt Learning** (2022, Diao/Huang/Xu/Li/Lin/Zhou/Zhang, TMLR): Foundational gradient-free discrete prompt optimization via variance-reduced policy gradient (VR-PGE) over categorical token distributions; API-only RL anchor predating/parallel to RLPrompt. https://arxiv.org/abs/2201.08531
- **PromptAgent** (2023, Wang/Li/Wang/Bai/Luo/Zhang/Jojic/Xing/Hu, ICLR 2024): Frames APO as strategic planning via Monte Carlo Tree Search; induces expert-level prompts from error reflection + simulated rewards. ~238 cites — a top-3 APO baseline. https://arxiv.org/abs/2310.16427
- **PE2 — Prompt Engineering a Prompt Engineer** (2023, Ye/Axmed/Pryzant/Khani, ACL 2024): Enriches the meta-prompt with detailed descriptions, context spec, and step-by-step reasoning template; direct successor to APE/ProTeGi. ~125 cites. https://arxiv.org/abs/2311.05661
- **DLN — Deep Language Networks (Joint Prompt Opt. of Stacked LLMs via Variational Inference)** (2023, Sordoni/Yuan/Côté/Pereira/Trischler/Xiao/Hosseini/Niedtner/Le Roux, NeurIPS): Optimizes prompts of stacked LLM layers, treating intermediate output as a latent variable; bridges APO and compound-system optimization. https://arxiv.org/abs/2306.12509
- **Plum — Prompt Learning using Metaheuristic** (2023, Pan/Xing/Diao/Liu/Shum/Zhang/Zhang): Applies 6 metaheuristics (hill-climb, simulated annealing, GA, tabu, harmony search); codifies the "general, automatic, discrete, black-box, gradient-free, interpretable" criteria defining the cluster. https://arxiv.org/abs/2311.08364 · https://github.com/research4pan/Plum
- **GReaTer — Gradients over Reasoning** (2024, Das/Kamoi/Pang/Zhang/Xiong/Zhang, Salesforce/PSU): Uses task-loss gradients over reasoning tokens so small open-source models can self-optimize prompts without a large closed-source optimizer LLM. https://arxiv.org/abs/2412.09722 · https://github.com/psunlpgroup/GreaTer
- **StablePrompt** (2024, Kwon/Kim/Kim/Lee/Kim, EMNLP): RL prompt tuning with Adaptive Proximal Policy Optimization (APPO) + anchor model for training stability; SOTA across classification/QA/generation; input-dependent TTE variant. https://arxiv.org/abs/2410.07652
- **CAPO — Cost-Aware Prompt Optimization** (2025, Zehle et al.): GA optimizer (builds on EvoPromptGA) adding racing/early-stopping + length penalty; jointly optimizes instructions + few-shot. Cited as current SOTA discrete optimizer. https://arxiv.org/abs/2504.16005
- **PACE** (2024, Dong et al., AAAI): Actor-critic RL applied to prompt *editing* itself. https://arxiv.org/abs/2308.10088
- **StraGo** (2024, Wu et al.): Summarizes strategic guidance from both correct AND incorrect predictions as feedback (vs. ProTeGi's error-only). https://arxiv.org/abs/2410.08601
- **PREFER** (2024, Zhang et al., AAAI): Feedback-reflect-refine cycle with prompt **ensembling/boosting**. https://arxiv.org/abs/2308.12033
- **SPRIG** (2024, Zhang et al.): Optimizes *system* prompts from a 300-component corpus (roles, styles, CoT) via token-level genetic edits. https://arxiv.org/abs/2410.14826
- **SCULPT** (2024, Kumar et al.): Hierarchical tree structure + two-step feedback for tuning *long, unstructured* prompts. https://arxiv.org/abs/2410.20964

### 已有items修正
- **APE** (Zhou 2022): venue is **ICLR 2023** (submitted Nov 2022, arXiv:2211.01910); listing year 2022 is fine for arXiv but venue should read ICLR 2023.
- **OPRO** (Yang 2023): institution is **Google DeepMind**; venue **ICLR 2024** (arXiv:2309.03409, Sep 2023). "DeepMind 2023" is acceptable but venue=ICLR 2024.
- **ProTeGi/APO** (Pryzant 2023): venue **EMNLP 2023** (arXiv:2305.03495); authors Pryzant/Iter/Li/Lee/Zhu/Zeng (Microsoft). Confirm "ProTeGi" = "Prompt Optimization with Textual Gradients" — the paper title is "Automatic Prompt Optimization with 'Gradient Descent' and Beam Search."
- **PromptWizard** (Microsoft 2024): authors **Agarwal/Singh/Dani/Magazine/Ganu/Nambi**, arXiv:2405.18369; jointly optimizes instructions + few-shot exemplars (note: degrades on Llama3-8B per follow-ups).
- **SAMMO** (Microsoft 2024): full name "Structure-Aware Multi-objective Metaprompt Optimization"; optimizes metaprompt *programs* as structured objects (arXiv:2404.02319) — slightly broader than pure instruction search; flag the compound-prompt scope.
- **SPO — Self-Supervised Prompt Optimization** (2025): venue **EMNLP 2025 Findings**; reference-free, uses pairwise LLM output comparisons (~$0.15/dataset); confirm authorship (MetaGPT/FoundationAgents group, Xiang et al.).
- AutoPrompt (Shin 2020, EMNLP 2020), GrIPS (Prasad 2022, EACL 2023), RLPrompt (Deng 2022, EMNLP 2022), TEMPERA (Zhang 2022, ICLR 2023): no corrections — all verified correct.

### 推荐补充字段
- **seed_prompt_source**: how the initial prompt is obtained — manual vs. instruction-induction-from-examples vs. task-README/description vs. empty/from-scratch (APO taxonomy axis #1; separates APE/GPO from ProTeGi/OPRO).
- **candidate_generation_operator**: operator producing new candidates — LLM-rewriter, metaprompt-resampling, GA mutation+crossover (token vs. sentence level), RL-trained generator, metaheuristic (annealing/PSO); finer than current `update_operator`.
- **search_strategy / filter_prune_step**: iterative search controller — greedy/TopK, beam search, MCTS, UCB/bandit, Bayesian/TPE, evolutionary population; distinguishes PromptAgent (MCTS) from ProTeGi (beam) from EvoPrompt (GA).
- **optimization_criteria**: objectives beyond accuracy — prompt length/cost, perplexity/fluency, robustness, entropy-based scores (places CAPO, PIN, CLAPS).
- **api_call_budget / query_efficiency**: quantified inference-call or rollout budget (e.g., GEPA "35x fewer rollouts," BBT API-bounded, CAPO racing) — the single most-emphasized 2025 differentiator.
- **optimizer_vs_target_size_asymmetry**: whether a *small* model can drive optimization (GReaTer's thesis; PromptWizard degrades on small models) — captures self-optimization vs. dependence on a large closed-source optimizer LLM.
- **transferability_evidence**: cross-model/cross-task prompt transfer claims (BDPL, GReaTer, FedDTPT) — evidence-backed refinement of field D "transfer."
- **system_vs_user_prompt_target**: optimizes system prompt vs. user/task instruction vs. few-shot exemplars vs. jointly (SPRIG=system; MIPROv2/PromptWizard=joint) — refines field B "optimization_target."

### 信息来源
- [A Systematic Survey of Automatic Prompt Optimization Techniques — Ramnath et al. (AWS), EMNLP 2025](https://arxiv.org/abs/2502.16923) (anchoring 5-part taxonomy)
- [A Survey of Automatic Prompt Engineering: An Optimization Perspective (2025)](https://arxiv.org/pdf/2502.11560)
- [PromptAgent (Wang et al. 2023)](https://arxiv.org/abs/2310.16427)
- [PE2 (Ye et al. 2023)](https://arxiv.org/abs/2311.05661)
- [BDPL (Diao et al. 2022, TMLR)](https://arxiv.org/abs/2201.08531) · [OpenReview](https://openreview.net/forum?id=IvsGP7xRvm)
- [BBT (Sun et al. 2022, ICML)](https://arxiv.org/abs/2201.03514)
- [GReaTer (Das et al. 2024)](https://arxiv.org/abs/2412.09722) · [code](https://github.com/psunlpgroup/GreaTer)
- [StablePrompt (Kwon et al. 2024, EMNLP)](https://arxiv.org/abs/2410.07652)
- [DLN (Sordoni et al. 2023, NeurIPS)](https://arxiv.org/abs/2306.12509)
- [Plum (Pan et al. 2023)](https://arxiv.org/abs/2311.08364)
- [CAPO (Zehle et al. 2025)](https://arxiv.org/abs/2504.16005)
- [GEPA (Agrawal et al. 2025)](https://arxiv.org/abs/2507.19457) (rollout-budget field context)
- [EvoPrompt (Guo et al. 2023)](https://arxiv.org/html/2309.08532v3) (Cluster 1/2 boundary note)
- [Genetic Prompt Search via LM Probabilities (Zhao et al. 2023, IJCAI)](https://www.ijcai.org/proceedings/2023/0588.pdf)