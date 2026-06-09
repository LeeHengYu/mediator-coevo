Found LOOP (2502.01600) and a useful practitioner's guide (2510.01132). I now have thorough, cross-verified coverage spanning foundations, agentic multi-turn RL, search/web/tool RL, reward models, and RL+memory. I have run 18+ distinct searches/fetches. Let me compile the final structured output.

---

## Cluster 6 — RL-based Policy Improvement for LM Agents — Validation & Supplement

### Validation of already-listed items
All six anchors are correctly placed and remain canonical. Two notes:
- **GRPO** — the precise citation is DeepSeekMath (Shao et al., arXiv:2402.03300, Feb 2024), not "DeepSeek 2024" generically. GRPO is the critic-free, group-normalized advantage algorithm now used by most agent-RL papers below. Worth recording arXiv:2402.03300 explicitly.
- **ReAct** is a prompting baseline (no weight update) — correctly a contrast point. **Self-Rewarding LMs** (Yuan 2024) updates weights via DPO on self-generated preference labels — fits C6.
No important object in the listed set is misclassified. Gaps are in (a) the foundational self-training lineage, (b) the 2025 agentic multi-turn-RL algorithm wave, (c) search/web/tool RL, (d) process/outcome reward models, (e) RL+memory hybrids. Supplemented below.

---

### 补充 Items (all WEIGHT-updating unless flagged)

**A. Foundational RL / self-training lineage (anchors the cluster's "before agents" history)**
- **DeepSeekMath / GRPO**: Origin of Group Relative Policy Optimization (critic-free, group-normalized advantage; the workhorse algorithm for nearly all 2025 agent-RL). Shao et al., 2024, arXiv:2402.03300. WEIGHTS.
- **DeepSeek-R1 / R1-Zero**: Pure-RL (GRPO, verifiable rewards) elicits long-CoT reasoning with no SFT cold-start (R1-Zero). DeepSeek-AI, Jan 2025, arXiv:2501.12948 (published *Nature* 645:633, 2025). The reference point for "reasoning-RL." WEIGHTS.
- **ReST (Reinforced Self-Training)**: Growing-batch offline RL — generate, filter by reward, fine-tune; framed for LM alignment/MT. Gulcehre et al. (DeepMind), 2023, arXiv:2308.08998. WEIGHTS.
- **ReST-EM ("Beyond Human Data")**: EM-style self-training with binary/scalar feedback; precursor to STaR-style loops at scale. Singh et al. (DeepMind), TMLR 2024, arXiv:2312.06585. WEIGHTS.
- **STaR (Self-Taught Reasoner)**: Bootstrap rationales, fine-tune on those yielding correct answers, with rationalization. Zelikman et al., NeurIPS 2022, arXiv:2203.14465. WEIGHTS. (Direct ancestor of reasoning-RL; good contrast — uses SFT on filtered samples, not policy-gradient.)
- **V-STaR**: Trains a *verifier* via DPO on correct+incorrect self-generated solutions; verifier ranks at inference. Hosseini et al., COLM 2024, arXiv:2402.06457. WEIGHTS (policy + verifier).
- **RLAIF / d-RLAIF**: RL from AI-generated preferences; direct-RLAIF skips the reward model and scores online. Lee et al. (Google), ICML 2024, arXiv:2309.00267. WEIGHTS. (Key axis: feedback signal = AI judge vs human.)
- **DAPO**: Open-source large-scale RL recipe — Decoupled Clip + Dynamic Sampling, token-level loss, overlong-reward shaping; fixes GRPO instabilities. Yu et al. (ByteDance), 2025, arXiv:2503.14476. WEIGHTS. (Important as the de-facto algorithmic upgrade over vanilla GRPO in agent settings.)

**B. Reward models (the credit/feedback machinery — own sub-axis)**
- **Let's Verify Step by Step / PRM800K**: Process supervision > outcome supervision; releases PRM800K step-level human labels. Lightman et al. (OpenAI), ICLR 2024, arXiv:2305.20050. (Process Reward Model = dense feedback signal; cross-cuts every agent-RL item below.)
- **Math-Shepherd**: Automatic PRM via MC rollout estimation (no human step labels) + step-by-step RL. Wang et al., ACL 2024, arXiv:2312.08935. WEIGHTS (PRM-guided PPO). (Note: confirm exact ID — surfaced via citing works; recommend a direct fetch before final inclusion.)

**C. Agentic multi-turn RL algorithms (core of cluster, the 2025 wave) — all WEIGHTS**
- **ArCHer** (already partially listed as "ARCHer"): Hierarchical multi-turn RL — high-level value over utterances + low-level token PPO; ~100x sample efficiency. Zhou et al., ICML 2024, arXiv:2402.19446.
- **RAGEN / StarPO**: State-Thinking-Actions-Reward trajectory-level RL; StarPO-S stabilizes against "echo trap" collapse in stochastic multi-turn envs. Wang et al., 2025, arXiv:2504.20073.
- **GiGPO (Group-in-Group Policy Optimization)**: Two-level grouping (episode + repeated-state "step groups") for fine-grained turn-level credit, critic-free. Feng et al., NeurIPS 2025, arXiv:2505.10978.
- **SWEET-RL**: Step-wise evaluation using *training-time privileged information* (asymmetric critic) for collaborative multi-turn tasks; introduces ColBench. Zhou et al. (Meta), 2025, arXiv:2503.15478.
- **LOOP (Leave-One-Out PPO)**: Value-network-free PPO variant for long-horizon interactive digital agents; single in-memory LLM; strong on AppWorld (beats o1 agent). Chen et al. (Apple), 2025, arXiv:2502.01600.
- **Turn-Level Reward Design (MT-GRPO / MT-PPO)**: First systematic study of turn-level (verifiable + LLM-judge) rewards for multi-turn agents. Wei et al., 2025, arXiv:2505.11821.
- **IGPO (Information-Gain Policy Optimization)**: Intrinsic dense turn-level reward = marginal increase in policy's probability of the correct answer; no external PRM/MC. Wang et al., 2025, arXiv:2510.14967.
- **AgentFlow / Flow-GRPO**: In-the-flow optimization of a planner module inside a multi-module loop; broadcasts trajectory outcome to each turn. Li et al., 2025, arXiv:2510.05592.
- **VAGEN (Bi-Level GAE)**: Multi-turn RL for VLM agents; world-modeling reward + turn-aware bi-level advantage estimation. Wang et al., 2025, arXiv:2510.16907. (Extends cluster to vision-language agents.)

**D. Search / Web / Tool-use RL — all WEIGHTS**
- **Search-R1**: RL teaches LLM to interleave reasoning with autonomous search-engine calls; retrieved-token masking + outcome reward. Jin et al., 2025, arXiv:2503.09516 (1000+ citations — high-impact).
- **R1-Searcher**: Two-stage outcome-supervised RL for autonomous search invocation (R1-Searcher++ adds internal/external knowledge balance). Song et al. (RUC), 2025, arXiv:2503.05592.
- **WebRL**: Self-evolving online curriculum RL + outcome-supervised reward model for open-LLM web agents; Llama-3.1-8B 4.8%→42.4% on WebArena-Lite. Qi et al., ICLR 2025, arXiv:2411.02337.
- **WebAgent-R1**: End-to-end multi-turn RL directly from online browser interactions. Wei et al., EMNLP 2025, arXiv:2505.16421.
- **ToolRL**: First comprehensive study of *reward design* for tool selection/application under RL. Qian et al., NeurIPS 2025, arXiv:2504.13958.
- **ToRL (Tool-Integrated RL)**: RL lets models autonomously discover when/how to call computational tools, beyond predefined constraints. Li et al., 2025, arXiv:2503.23383.
- **ReTool**: RL for strategic code-interpreter tool use with cold-start synthetic traces. Feng et al., 2025, arXiv:2504.11536.
- **RLEF (already-listed-adjacent)**: End-to-end RL to ground code LLMs in *execution feedback* across multi-turn code synthesis. Gehring et al. (Meta), ICML 2025, arXiv:2410.02089.
- **MURPHY**: Feedback-conditioned multi-turn GRPO with retrospective credit propagation over rollout trees for self-correcting code-gen. Ekbote et al. (Amazon), 2025, arXiv:2511.07833.

**E. RL + memory / self-play hybrids (key contrast: weights AND external store)**
- **Memory-R1**: RL (PPO/GRPO) trains a Memory Manager (add/update/delete/noop) + Answer Agent to manage external memory — updates WEIGHTS to govern a non-parametric memory store. Yan et al., 2025, arXiv:2508.19828. (Bridges C6↔C5 skill libraries / memory clusters.)
- **Absolute Zero Reasoner (AZR)**: RLVR self-play — one model proposes tasks maximizing learnability and solves them, zero external data; self-evolving curriculum. Zhao et al., NeurIPS 2025, arXiv:2505.03335. WEIGHTS.

**F. Surveys / synthesis (use for taxonomy + completeness, not as items)**
- **"The Landscape of Agentic RL for LLMs: A Survey"**, Zhang et al., 2025, arXiv:2509.02547 — formalizes the LLM-RL (degenerate single-step MDP) → Agentic-RL (temporally-extended POMDP) shift; taxonomy over planning/tool-use/memory/reasoning/self-improvement/perception. This is the best framing-anchor for the whole cluster.
- **"From Reasoning to Agentic: Credit Assignment in RL for LLMs"**, Zhang, 2026, arXiv:2604.09459 — surveys 47 credit-assignment methods; 2D taxonomy (granularity: token/segment/step/turn/multi-agent × methodology: MC/TD/model-based/game-theoretic/info-theoretic). Excellent source for the credit-assignment field below.
- **"A Practitioner's Guide to Multi-turn Agentic RL"**, 2025, arXiv:2510.01132 — analyzes environment × policy × reward pillars; useful for evaluation-field design.

---

### 推荐补充字段 (new fields for the framework, specific to weight-updating RL)

- **policy_gradient_algorithm**: Which base optimizer (PPO / GRPO / RLOO / DPO / REINFORCE / DAPO / Flow-GRPO / hierarchical actor-critic). Needed because the *algorithm family* is the primary differentiator within C6 and the sharpest contrast vs textual-optimization clusters (which have no policy gradient at all).
- **critic_design**: none (critic-free/group-baseline) vs learned value network vs **asymmetric/privileged critic** (training-time info, e.g. SWEET-RL) vs PRM-as-critic. This axis is where the 2025 innovations concentrate and predicts memory/compute cost.
- **credit_assignment_granularity**: token / segment / step / turn / trajectory(outcome-only) / multi-agent. The single most discriminating axis in agentic RL (per arXiv:2604.09459 survey); the listed framing's "credit-assignment" should be made an explicit enumerated field.
- **reward_signal_source**: verifiable/programmatic (rule, unit-test, exact-match) vs outcome-RM (ORM) vs process-RM (PRM) vs LLM-as-judge vs intrinsic (information-gain / entropy / self-reward). Refines the existing "feedback_signal (scalar reward)" field — the *origin* of the scalar matters as much as its scalar nature.
- **reward_density**: sparse outcome-only vs dense turn-level vs dense step/token-level. Directly governs sample efficiency and is the explicit motivation of most C6 papers.
- **interaction_horizon / MDP_formulation**: single-step (degenerate MDP) vs multi-turn POMDP; typical turn count / token horizon (100K–1M tokens for long-horizon agents). Captures the LLM-RL→Agentic-RL distinction (arXiv:2509.02547) and is absent from the current scope fields.
- **on_policy_vs_off_policy / rollout_reuse**: on-policy, off-policy sample reuse (LOOP), or offline growing-batch (ReST/ReST-EM). Belongs under "Learning Properties — rollouts"; determines compute profile.
- **environment_interaction_type**: static dataset vs live tool/search/browser/code-exec environment vs self-play task generation. Distinguishes reasoning-RL from genuinely agentic RL and flags reproducibility cost.
- **cold_start_dependency**: pure-RL-from-base (R1-Zero, AZR) vs SFT/cold-start-then-RL (most). A salient design choice in the reasoning-RL line worth its own field.
- **updates_what (weights vs prompt/memory)** [CROSS-CLUSTER KEY AXIS]: For C6, record explicitly "model weights"; for hybrids (Memory-R1) record "weights → govern external memory." This is the load-bearing axis separating C6 from C1–C5/C7 and should be a top-level field across the whole survey.

---

### 信息来源
- [Search-R1, arXiv:2503.09516](https://arxiv.org/abs/2503.09516)
- [RAGEN/StarPO, arXiv:2504.20073](https://arxiv.org/abs/2504.20073)
- [GiGPO, arXiv:2505.10978](https://arxiv.org/abs/2505.10978)
- [SWEET-RL, arXiv:2503.15478](https://arxiv.org/abs/2503.15478)
- [ToolRL, arXiv:2504.13958](https://arxiv.org/abs/2504.13958)
- [RLEF, arXiv:2410.02089](https://arxiv.org/abs/2410.02089)
- [R1-Searcher, arXiv:2503.05592](https://arxiv.org/abs/2503.05592)
- [Reinforcement Pre-Training, arXiv:2506.08007](https://arxiv.org/abs/2506.08007)
- [DeepSeekMath/GRPO, arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1, arXiv:2501.12948](https://arxiv.org/abs/2501.12948)
- [ReST-EM "Beyond Human Data", arXiv:2312.06585](https://arxiv.org/abs/2312.06585)
- [ReST, arXiv:2308.08998](https://arxiv.org/abs/2308.08998)
- [STaR, arXiv:2203.14465](https://arxiv.org/abs/2203.14465)
- [V-STaR, arXiv:2402.06457](https://arxiv.org/abs/2402.06457)
- [RLAIF, arXiv:2309.00267](https://arxiv.org/abs/2309.00267)
- [DAPO, arXiv:2503.14476](https://arxiv.org/abs/2503.14476)
- [Let's Verify Step by Step / PRM800K, arXiv:2305.20050](https://arxiv.org/abs/2305.20050)
- [Agent Q, arXiv:2408.07199](https://arxiv.org/abs/2408.07199)
- [WebRL, arXiv:2411.02337](https://arxiv.org/abs/2411.02337)
- [WebAgent-R1, arXiv:2505.16421](https://arxiv.org/abs/2505.16421)
- [ToRL, arXiv:2503.23383](https://arxiv.org/abs/2503.23383) / [ReTool, arXiv:2504.11536](https://arxiv.org/abs/2504.11536)
- [LOOP, arXiv:2502.01600](https://arxiv.org/abs/2502.01600)
- [Turn-Level Reward Design, arXiv:2505.11821](https://arxiv.org/abs/2505.11821)
- [IGPO, arXiv:2510.14967](https://arxiv.org/abs/2510.14967)
- [AgentFlow/Flow-GRPO, arXiv:2510.05592](https://arxiv.org/abs/2510.05592)
- [VAGEN, arXiv:2510.16907](https://arxiv.org/abs/2510.16907)
- [MURPHY, arXiv:2511.07833](https://arxiv.org/abs/2511.07833)
- [Memory-R1, arXiv:2508.19828](https://arxiv.org/abs/2508.19828)
- [Absolute Zero, arXiv:2505.03335](https://arxiv.org/abs/2505.03335)
- [Survey: Landscape of Agentic RL, arXiv:2509.02547](https://arxiv.org/abs/2509.02547)
- [Survey: Credit Assignment in RL for LLMs, arXiv:2604.09459](https://arxiv.org/abs/2604.09459)
- [Practitioner's Guide to Multi-turn Agentic RL, arXiv:2510.01132](https://arxiv.org/abs/2510.01132)

---

## Summary

Ran 20+ distinct searches/fetches across Semantic Scholar, Firecrawl, and Exa, cross-verified against arXiv/venue primary sources. Found that the 6 listed Cluster-6 anchors are correctly classified but the cluster is **substantially under-populated** — I supplemented **~28 items** organized into 6 sub-groups: (A) foundational RL/self-training lineage (GRPO origin, DeepSeek-R1, ReST, ReST-EM, STaR, V-STaR, RLAIF, DAPO), (B) reward models (PRM800K, Math-Shepherd — the latter needs a direct ID confirmation), (C) the 2025 agentic multi-turn-RL algorithm wave (ArCHer, RAGEN/StarPO, GiGPO, SWEET-RL, LOOP, IGPO, AgentFlow, VAGEN, Turn-Level Reward Design), (D) search/web/tool RL (Search-R1, R1-Searcher, WebRL, WebAgent-R1, ToolRL, ToRL, ReTool, RLEF, MURPHY), and (E) RL+memory/self-play hybrids (Memory-R1, Absolute Zero). Two 2025-2026 surveys (arXiv:2509.02547, 2604.09459) anchor the taxonomy.

Recommended **10 new fields** tailored to weight-updating RL, the most important being **`updates_what (weights vs prompt/memory)`** — the load-bearing cross-cluster axis separating C6 from the textual-optimization clusters — plus **`credit_assignment_granularity`**, **`critic_design`** (notably the asymmetric/privileged-critic innovation), **`reward_signal_source`**, **`reward_density`**, **`policy_gradient_algorithm`**, and **`MDP_formulation` (single-step vs multi-turn POMDP)**.

Key cross-cluster insight: the boundary with other clusters is whether parameters change. C6 methods update model weights via policy gradient; the one genuine hybrid is **Memory-R1**, which updates weights *in order to* govern an external (non-parametric) memory store — making it the natural bridge to the memory/skill-library clusters.

One caveat: Math-Shepherd's exact arXiv ID (2312.08935) was inferred from citing works rather than a direct hit (Semantic Scholar strips the hyphen in "Math-Shepherd"); recommend a direct fetch before final inclusion.