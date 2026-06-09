I have done 15+ distinct searches and cross-verified all items against primary sources (arXiv, ACL Anthology, NeurIPS/ICML/ICLR proceedings, OpenReview). I now have a comprehensive picture of Cluster 4. Let me compile the final structured results.

## Summary of Research — Cluster 4 (Reflective / Verbal RL)

I ran 16+ distinct searches/fetches across Semantic Scholar, Exa, and primary-source pages (arXiv, ACL Anthology, NeurIPS/ICML/ICLR proceedings, OpenReview, Microsoft Research). The 6 already-listed items (Self-Refine, Reflexion, CRITIC, Retroformer, ExpeL, Agent-Pro) are all validated as canonical Cluster-4 anchors with confirmed identities. Below are the gaps to fill plus recommended new fields.

---

### 补充 Items (Supplemental Items)

**Critique / debate sub-cluster (essential — the "negative result" anchors)**
- **LLMs Cannot Self-Correct Reasoning Yet (Huang et al.)**: The pivotal negative-result paper showing intrinsic self-correction (no external feedback) degrades reasoning. ICLR 2024, Google DeepMind. arXiv:2310.01798. 931 citations. *Must include — it is the central counter-thesis the whole cluster responds to.*
- **When Can LLMs Actually Correct Their Own Mistakes? A Critical Survey (Kamoi, Zhang et al.)**: TACL 2024 critical survey; categorizes self-correction research questions and shows prompted-LLM self-feedback rarely works; reliable external feedback or large-scale fine-tuning is required. arXiv:2406.01297. The definitive framing reference for evaluation rigor.
- **Self-Contrast: Better Reflection Through Inconsistent Solving Perspectives (Zhang et al.)**: Diagnoses that the bottleneck is poor self-evaluated feedback (overconfidence/inconsistency); generates diverse perspectives, contrasts them into a checklist. ACL 2024. arXiv:2401.02009. (Explicitly named in your hint list.)

**Memory-of-feedback / experiential verbal RL (core)**
- **REMEMBERER / RLEM (Zhang, Chen et al.)**: "LLMs Are Semi-Parametric RL Agents." Long-term experience memory updated via Reinforcement Learning with Experience Memory (RLEM) across episodes; learns from success+failure without parameter updates. NeurIPS 2023. arXiv:2306.07929. (In your hint list.)
- **AutoGuide (Fu, Kim, Sohn, Logeswaran, Lee et al.)**: Auto-generates *context/state-conditional* natural-language guidelines from contrastive offline trajectories (different returns), retrieves relevant ones at test time. NeurIPS 2024. arXiv:2403.08978. (In your hint list — confirmed.)
- **MetaReflection (Gupta, Kirtania et al., Microsoft)**: Offline-RL technique that generalizes failed-trial self-reflections into reusable "meta-reflection" instructions stored in semantic memory. EMNLP 2024. arXiv:2405.13009. Bridges Reflexion → prompt optimization.
- **MoT: Memory-of-Thought (Li & Qiu)**: Pre-thinks over unlabeled data, stores high-confidence thoughts as external memory, recalls them at test time — self-improvement without annotations or parameter updates. EMNLP 2023. arXiv:2305.05181. (Cited as foundational in several memory surveys.)
- **RAP: Retrieval-Augmented Planning with Contextual Memory (Kagaya et al.)**: Retrieves situation-relevant past experiences to guide planning; works text-only and multimodal. arXiv:2402.03610 (2024).

**Reflection-as-policy / trajectory-contrastive (core-to-boundary)**
- **Agent Workflow Memory (AWM) (Wang, Mao, Fried, Neubig)**: Induces reusable natural-language *workflows* (abstracted sub-routines) from past trajectories, offline and online; strong on Mind2Web/WebArena. ICML 2025. arXiv:2409.07429. Strong complement to AutoGuide.
- **Trial and Error: Exploration-based Trajectory Optimization (ETO) (Song et al.)**: Learns from *failure* trajectories via contrastive success/failure pairs + DPO. ACL 2024. arXiv:2403.02502. (Boundary: textual trajectories but gradient-based DPO update — flag as C4/C6 overlap.)
- **LATS: Language Agent Tree Search (Zhou et al.)**: Unifies reasoning/acting/planning via MCTS with LLM value functions + self-reflection. ICML 2024. arXiv:2310.04406. (Your hint's "Tree-of-Thought-as-policy" candidate; boundary — search-augmented reflection.)

**Self-correction trained into parameters (boundary — overlaps C6 but central to the cluster's debate)**
- **SCoRe: Training Language Models to Self-Correct via RL (Kumar et al., DeepMind)**: Multi-turn online RL on self-generated correction traces; directly answers Huang 2023's negative result by *training* self-correction. ICLR 2025. arXiv:2409.12917. (Flag as C4↔C6 boundary.)
- **Reflect, Retry, Reward (Xu/Olteanu et al.)**: Rewards self-reflection tokens via GRPO when a retry succeeds — verbal reflection optimized into the policy. arXiv:2505.24726 (2025). (Boundary; nice 2025 bridge of verbal-RL → gradient-RL.)

**Empirical / 2025 items**
- **Self-Reflection in LLM Agents: Effects on Problem-Solving (Renze & Guven)**: Controlled study across 9 LLMs and 8 self-reflection types showing significant gains (p<0.001); useful as an evaluation/ablation reference. arXiv:2405.06682 (2024).
- *(Optional, niche-2025)* **Meta-Policy Reflexion (MPR)** arXiv:2509.03990 and **H²R: Hierarchical Hindsight Reflection** arXiv:2509.12810 — both distill textual reflections into structured/predicate memory; include only if depth on 2025 reflective-memory variants is desired.

---

### 推荐补充字段 (Recommended New Fields)

- **memory_scope**: none / episodic (within-task transient) / cross-task persistent / cross-domain. Cluster 4 splits sharply here (Self-Refine = none; Reflexion = episodic; ExpeL/REMEMBERER/AutoGuide/AWM = cross-task), and current C/D fields don't capture it cleanly.
- **feedback_source**: intrinsic-self / self-vs-self-contrast / external-tool / environment-reward / oracle-label. Distinguishes the items that "work" (external/environment per Kamoi survey) from those that don't (pure intrinsic). Critical given the cluster's central debate.
- **artifact_form** (what the textual "policy update" produces): per-instance critique / reflection note / conditional guideline (if-context-then-action) / reusable workflow / predicate rule. Captures the granularity progression Reflexion → AutoGuide → AWM.
- **parameter_update**: none (in-context only) / fine-tunes-retrospective-model / fine-tunes-policy (DPO/PPO/GRPO). Cleanly separates pure verbal-RL (C4) from boundary items (Retroformer, ETO, SCoRe, Reflect-Retry-Reward) that cross into C6.
- **retrieval_mechanism**: n/a / embedding similarity / LLM-as-retriever / state/context-conditional match. Differentiates how stored feedback is reused (MoT's LLM-retrieval vs AutoGuide's context-matching vs FAISS vector memory).
- **negative_result_evidence / robustness_caveat**: whether the method's gains survive the Huang-2023 / Kamoi-2024 critiques (e.g., relies on oracle stopping, unfair baselines, or only intrinsic feedback). Important for honest "Significance / Uncertain" reporting in a cluster full of contested claims.

---

### 信息来源 (Sources)
- [LLMs Cannot Self-Correct Reasoning Yet (arXiv:2310.01798)](https://arxiv.org/abs/2310.01798)
- [When Can LLMs Actually Correct Their Own Mistakes? — Kamoi et al., TACL (arXiv:2406.01297)](https://arxiv.org/abs/2406.01297)
- [Self-Contrast (arXiv:2401.02009)](https://arxiv.org/abs/2401.02009)
- [REMEMBERER / RLEM — NeurIPS 2023 (arXiv:2306.07929)](https://arxiv.org/abs/2306.07929)
- [AutoGuide — NeurIPS 2024 (arXiv:2403.08978)](https://arxiv.org/abs/2403.08978)
- [MetaReflection — EMNLP 2024 (aclanthology.org/2024.emnlp-main.477)](https://aclanthology.org/2024.emnlp-main.477/)
- [MoT: Memory-of-Thought — EMNLP 2023 (arXiv:2305.05181)](https://arxiv.org/abs/2305.05181)
- [RAP — arXiv:2402.03610](https://arxiv.org/abs/2402.03610)
- [Agent Workflow Memory — ICML 2025 (arXiv:2409.07429)](https://arxiv.org/abs/2409.07429)
- [ETO: Trial and Error — ACL 2024 (aclanthology.org/2024.acl-long.409)](https://aclanthology.org/2024.acl-long.409/)
- [LATS — ICML 2024 (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406)
- [SCoRe — ICLR 2025 (arXiv:2409.12917)](https://arxiv.org/abs/2409.12917)
- [Reflect, Retry, Reward (arXiv:2505.24726)](https://arxiv.org/abs/2505.24726)
- [Self-Reflection in LLM Agents — Renze & Guven (arXiv:2405.06682)](https://arxiv.org/abs/2405.06682)
- [CRITIC — ICLR 2024 (OpenReview WSrRF5Wy6v)](https://openreview.net/forum?id=WSrRF5Wy6v)
- [Self-Refine — NeurIPS 2023 (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)
- [Retroformer — ICLR 2024 (arXiv:2308.02151)](https://arxiv.org/abs/2308.02151)
- [ExpeL — AAAI 2024 (ojs.aaai.org/AAAI/article/view/29936)](https://ojs.aaai.org/index.php/AAAI/article/view/29936)
- [Agent-Pro — ACL 2024 (arXiv:2402.17574)](https://arxiv.org/abs/2402.17574)

**Note on boundaries:** ETO, SCoRe, and Reflect-Retry-Reward are C4↔C6 boundary items (textual reflection but gradient-based parameter updates); the new `parameter_update` field is the cleanest way to tag them. The Algorithm-Distillation / DPT / ICRL transformer line (Laskin 2022, etc.) surfaced but is parametric in-context RL over trajectories, not *verbal/textual* policy — I'd keep it out of C4 unless the survey wants an explicit "in-context RL (parametric)" contrast note.