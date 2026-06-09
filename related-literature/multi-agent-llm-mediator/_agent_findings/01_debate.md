I now have a thorough, verified corpus. I have everything needed to produce the final report.

---

# Research Report: Multi-Agent LLM Debate & Information Filtering/Aggregation Between Debating Agents

## Seed-item validation (all 7 confirmed accurate)

All 7 seed items verified with correct names/years/authors:
1. **Du et al. 2023** — "Improving Factuality and Reasoning in Language Models through Multiagent Debate" (arXiv:2305.14325; Du, Li, Torralba, Tenenbaum, Mordatch). ✓ Note: aggregation = concatenate-all-peer-answers + final round answer; ICML 2024.
2. **Liang et al. 2023** — "Encouraging Divergent Thinking in LLMs through Multi-Agent Debate" (arXiv:2305.19118; Liang, He, Jiao, Wang, Wang, Wang, Yang, Tu, Shi). ✓ Has explicit **judge** that extracts the final answer; "tit-for-tat" + adaptive break; EMNLP 2024.
3. **Chen, Saha, Bansal 2023** — "ReConcile" (arXiv:2309.13007). ✓ KEY for your slice: **confidence-weighted voting** + discussion prompt that bundles grouped answers + confidence + answer-rectifying demonstrations.
4. **Chan et al. 2023** — "ChatEval" (arXiv:2308.07201). ✓ ICLR 2024.
5. **Yin et al. 2023** — "Exchange-of-Thought (EoT)" (arXiv:2312.01823, EMNLP 2023). ✓ KEY: 4 communication paradigms (Memory/Report/Relay/Debate) + confidence-based filtering of incorrect chains.
6. **Smit et al. 2024** — "Should we be going MAD?" (arXiv:2311.17371, ICML 2024; Smit, Grinsztajn, Duckworth, Barrett, Pretorius). ✓ KEY: agreement-modulation, MAD often loses to self-consistency.
7. **Liu et al. 2023** — "DyLAN" (arXiv:2310.02170, ICLR 2024). ✓ KEY: Agent Importance Score, inference-time agent selection, early-stopping.

**Gaps in the seed set:** It is missing (a) the AI-safety/scalable-oversight debate lineage (Irving 2018, Khan 2024, Kenton 2024), which is the conceptual root of "weak judge filters strong debaters"; (b) the entire sparse/efficient-communication cluster (most directly relevant to your mediator-coevo "information filtering" axis); (c) explicit mediator/moderator/summarizer architectures (MoDS, SocraSynth); (d) the critical/negative-results literature (sycophancy, error propagation, "Stop Overvaluing MAD"); (e) aggregation-mechanism papers beyond majority vote (confidence-weighted, score-trajectory, conformal, Mixture-of-Agents).

---

## 补充 Items

### A. Foundational scalable-oversight / debate-as-alignment lineage (the conceptual root of judge-based information filtering)
- **AI Safety via Debate** (2018, Irving, Christiano, Amodei): seminal — two agents argue, a (weaker) judge decides; PSPACE-vs-NP framing; the origin of "debate as an information-aggregation/oversight protocol." https://arxiv.org/abs/1805.00899
- **Debating with More Persuasive LLMs Leads to More Truthful Answers** (2024, Khan, Hughes, Valentine, Ruis, Sachan, Radhakrishnan, Grefenstette, Bowman, Rocktäschel, Perez; ICML 2024): KEY — non-expert judge selects answer; optimizing debaters for *persuasiveness* raises judge truth-detection; debate > consultancy. The core "information asymmetry + judge filtering" result. https://arxiv.org/abs/2402.06782
- **On Scalable Oversight with Weak LLMs Judging Strong LLMs** (2024, Kenton, Siegel, Kramár, Brown-Cohen, Albanie, Bulian, Agarwal, Lindner, Tang, Goodman, Shah; DeepMind, NeurIPS 2024): debate vs consultancy vs direct-QA across QA/math/code/logic/multimodal asymmetries; when debate's judge-filtering actually helps. https://arxiv.org/abs/2407.04622

### B. Sparse / efficient communication topologies (CORE to your "information filtering/routing/gating" axis)
- **Improving Multi-Agent Debate with Sparse Communication Topology** (2024, Li et al., Google; EMNLP 2024 Findings): KEY — systematically shows neighbor-connected (sparse) MAD matches/beats fully-connected at ~40% lower token cost; assign stronger LLMs to higher-degree nodes. https://aclanthology.org/2024.findings-emnlp.427.pdf
- **GroupDebate** (2024, arXiv:2409.14051): divides agents into groups; each group debates internally, **summarizes** results into a **shared pool**, groups retrieve summaries as next-round input — explicit summarization-based information aggregation; up to ~50% token reduction. https://arxiv.org/abs/2409.14051
- **S²-MAD: Breaking the Token Barrier** (2025, NAACL 2025; arXiv:2502.04790): KEY — **Decision-Making Mechanism** lets each agent *selectively* incorporate only non-redundant peer viewpoints (gating); up to 94.5% token reduction vs MAD with <2% accuracy loss. https://arxiv.org/abs/2502.04790
- **CortexDebate** (2025, ACL 2025 Findings): KEY — sparse debating graph + **McKinsey Trust Formula**-based "white matter" Debate-Mediator Module (DMM) that credibly weights edges to make debate "sparse and equal," fixing over-confidence-driven unequal debates; cuts per-agent context up to ~70%. https://aclanthology.org/2025.findings-acl.495.pdf
- **AgentPrune ("Cut the Crap")** (2025, Zhang et al., ICLR 2025; arXiv:2410.02506): defines **Communication Redundancy**; trains a low-rank graph mask to one-shot-prune the spatial-temporal message graph → token-economic, also defends adversarial messages. https://arxiv.org/abs/2410.02506
- **GPTSwarm** (2024, Zhuge et al., ICML 2024 Oral; arXiv:2402.16823): agents as optimizable computational graphs; **edge optimization** learns which inter-agent information-flow edges to keep — a learned-topology view of information routing. https://arxiv.org/abs/2402.16823

### C. Explicit mediator / moderator / summarizer architectures (most relevant to "mediator-coevo")
- **MoDS: Moderating a Mixture of Document Speakers** (2025, Balepur et al., NAACL 2025): KEY for mediation — a **Moderator LLM picks which Speaker agents respond** to tailored sub-queries, tracks perspectives in an outline → content plan; explicit panel-discussion mediation + balanced aggregation. https://aclanthology.org/2025.naacl-long.20/
- **SocraSynth** (2024, Edward Y. Chang; arXiv:2402.06634): KEY — a **human/LLM moderator** sets debate topic and tunes a **"contentiousness" level** (confrontational→collaborative), gathering conciliatory final remarks; explicit moderator-controlled agreement modulation. https://arxiv.org/abs/2402.06634
- **Enhancing Multi-Agent Consensus Through Third-Party LLM Integration** (2024, Duan & Wang; arXiv:2411.16189): a **third-party LLM** adjusts agents' attention weights via uncertainty/confidence estimation — an explicit external aggregation/gating mediator. https://arxiv.org/abs/2411.16189

### D. Aggregation / consensus mechanisms beyond majority vote
- **Mixture-of-Agents (MoA)** (2024, Wang, Wang, Athiwaratkun, Zhang, Zou; ICLR 2025 Spotlight; arXiv:2406.04692): KEY — layered architecture where each layer's **aggregator** synthesizes all prior-layer agent outputs; canonical learned/structured aggregation topology. https://arxiv.org/abs/2406.04692
- **More Agents Is All You Need (Agent Forest)** (2024, Li, Zhang, Yu, Fu, Ye; TMLR; arXiv:2402.05120): KEY baseline — pure sampling-and-**majority-voting** scales with #agents; the simplest aggregation; orthogonal to debate. https://arxiv.org/abs/2402.05120
- **Free-MAD: Consensus-Free Multi-Agent Debate** (2025, Cui et al.; arXiv:2509.11035): KEY — replaces last-round majority vote with a **score-based mechanism over the whole debate trajectory** + anti-conformity; single-round. https://arxiv.org/abs/2509.11035
- **Roundtable Policy** (2025, Yao, Dong, Yang, Li, Du; arXiv:2509.16839): **confidence-weighted-consensus** aggregation inspired by democratic committees / Society of Mind; black-box, interpretable. https://arxiv.org/abs/2509.16839
- **From Debate to Decision: Conformal Social Choice** (2026, Wang et al.; arXiv:2604.07667): post-hoc layer aggregates verbalized agent probability distributions via **linear opinion pool + split conformal prediction** → calibrated act-vs-escalate decisions; intercepts wrong-consensus. https://arxiv.org/abs/2604.07667
- **Beyond Majority Voting: Radial Consensus Score** (2026, Nguyen, Gupta, Le; arXiv:2604.12196): geometric aggregation — weighted Fréchet mean of answer embeddings; drop-in replacement for majority voting in MAD. https://arxiv.org/abs/2604.12196
- **Multi-Agent Debate for LLM Judges with Adaptive Stability Detection** (2025, Hu, Tan, Wang, Qu, Chen; NeurIPS 2025): debate-judge with **Beta-Binomial consensus dynamics + KS-test adaptive stopping**; proves debate amplifies correctness over static ensembles. https://openreview.net/forum?id=Vusd1Hw2D9
- **Sequential Consensus (Wald-SPRT compute governor)** (2026, Morandi; arXiv:2605.19193): LLM-judge emits a [0,1] consensus score each round; **SPRT** stops debate adaptively — a compute-control/failure-detection layer over aggregation. https://arxiv.org/abs/2605.19193

### E. Knowledge-grounding / shared-pool filtering during debate
- **MADKE — "Learning to Break: Knowledge-Enhanced Reasoning in Multi-Agent Debate"** (2023/2025, Wang et al.; Neurocomputing 2025; arXiv:2312.04854): KEY — **shared retrieval knowledge pool** + **adaptive knowledge selection** (each agent chooses per-round whether to use external knowledge) to break "cognitive islands." https://arxiv.org/abs/2312.04854

### F. Critical / negative-results & failure-mode literature (essential for honest framing)
- **Rethinking the Bounds of LLM Reasoning: Are Multi-Agent Discussions the Key?** (2024, Wang, Wang, Su, Tong, Song; ACL 2024; arXiv:2402.18272): a strong single agent with good prompt ≈ best discussion framework; discussion only wins without in-context demos. https://arxiv.org/abs/2402.18272
- **Stop Overvaluing Multi-Agent Debate — Rethink Evaluation and Embrace Model Heterogeneity** (2025, Zhang et al.; NeurIPS 2025 Position; arXiv:2502.08788): 5 MAD methods × 9 benchmarks × 4 models; MAD often loses to CoT/Self-Consistency; **model heterogeneity** is the universal fix. https://arxiv.org/abs/2502.08788
- **CONSENSAGENT** (2025, Pitre, Ramakrishnan, Wang; ACL 2025 Findings): formalizes **sycophancy** in MAD (agents reinforce each other), mitigates via dynamic prompt refinement. https://aclanthology.org/2025.findings-acl.1141.pdf
- **Talk Isn't Always Cheap: Understanding Failure Modes in Multi-Agent Debate** (2025, Wynn et al.; arXiv:2509.05396): debate can *decrease* accuracy over rounds even when strong models outnumber weak; correct→incorrect shifts from peer pressure. https://arxiv.org/abs/2509.05396
- **Peacemaker or Troublemaker: How Sycophancy Shapes Multi-Agent Debate** (2025, Yao et al.; arXiv:2509.23055): first operational sycophancy framework for MADS; metrics for sycophancy's effect on information exchange; debater- vs judge-driven failure modes. https://arxiv.org/abs/2509.23055
- **Beware of the Woozle Effect: Hallucination Propagation in MAD → DIGRA** (2026, Zhang et al.; IEEE TASLP): identifies hallucination propagation under static fully-connected topology; proposes **Information Gain Ratio-driven dynamic topology** (DIGRA). (DOI 10.1109/TASLPRO.2026.3675803)
- **MAD-Spear** (2025, Cui & Du; arXiv:2507.13038): conformity-driven prompt-injection attack on MAD; formal MAD fault-tolerance — security of the aggregation channel.

### G. Additional notable mechanism variants (optional / second-tier)
- **Multi-Persona / "tit-for-tat" + judge** is Liang 2023 (already #2). Related single-LLM analog: **Solo Performance Prompting (SPP)** (Wang et al. 2023, NAACL 2024; arXiv:2307.05300) — one LLM splits into personas (useful as a "no-real-multi-agent" baseline).
- **CONCAT** (2026, Ma et al.; arXiv:2605.29612): training-free **Consensus- and Confidence-driven Ad hoc Teaming** — clusters agents by initial answer, picks leaders by confidence, Theory-of-Mind heuristic prunes communication.
- **Small-World Networks for MAS** (2025, Wang et al.; arXiv:2512.18094): uncertainty-guided rewiring (semantic-entropy) between epistemically divergent agents — topology as information-routing prior.
- **FinCom / Disagree-or-Commit** (2026, Yang et al.; arXiv:2606.00939): embeds **structured dissent** to fight premature sycophantic consensus.
- Surveys to anchor the slice: **"Beyond Self-Talk: A Communication-Centric Survey of LLM-MAS"** (2025, arXiv:2502.14321); **"Multi-Agent Collaboration Mechanisms: A Survey of LLMs"** (2025, arXiv:2501.06322); **"Topological Structure Learning Should Be a Research Priority for LLM-MAS"** (2025, arXiv:2505.22467).

---

## 推荐补充字段 (new fields specific to this slice)

- **aggregation_rule**: the exact mechanism converting per-agent positions into a final answer — `majority_vote` / `confidence_weighted` / `judge_LLM` / `trajectory_score` / `opinion_pool+conformal` / `embedding_geometric` / `layered_aggregator`. (Your single most discriminating dimension; the seed framework folds this into "mediation_and_information_filtering" but it deserves its own column.)
- **judge_or_mediator_role**: none / passive-extractor / active-moderator (selects speakers, sets topic) / weaker-than-debaters (scalable-oversight) / third-party-reweighter. Distinguishes ChatEval/Liang (judge) vs MoDS/SocraSynth (active mediator) vs Khan/Kenton (weak judge).
- **information_filtering_granularity**: what gets gated/routed — `all-messages` (brute force) / `neighbor-subset` (sparse topology) / `summary-only` (GroupDebate) / `non-redundant-viewpoints` (S²-MAD) / `trust-weighted-edges` (CortexDebate) / `learned-graph-mask` (AgentPrune).
- **agreement_modulation**: whether the method explicitly tunes consensus pressure — none / contentiousness-knob (SocraSynth) / agreement-level prompt (Smit) / anti-conformity (Free-MAD) / disagree-or-commit (FinCom). Directly relevant to mediator-coevo "modulating agreement."
- **confidence_signal_used**: none / verbalized-confidence / token-logprob / semantic-entropy / uncertainty-estimate — what scalar quality signal drives routing or voting (ReConcile, EoT, DIGRA, conformal, small-world).
- **termination_criterion**: fixed-rounds / consensus / early-stop-by-importance (DyLAN) / SPRT (Wald) / KS-stability (Hu 2025) / adaptive-break (Liang). Captures compute–accuracy governance.
- **failure_mode_addressed**: sycophancy / conformity / hallucination-propagation (Woozle) / degeneration-of-thought / cognitive-islands / over-confidence-unequal-debate / communication-redundancy / prompt-injection. Lets you map each method to the pathology it targets — useful contrast axis for a coevolution mediator.
- **judge_capability_relative_to_debaters** (weak / equal / strong): central to the scalable-oversight subcluster (weak-judge-supervises-strong is a distinct paradigm from peer debate).
- **token_cost_vs_accuracy_tradeoff**: quantified efficiency claim (e.g., S²-MAD −94.5% tokens / <2% acc; sparse-MAD −40% tokens). The field's dominant evaluation axis post-2024.

---

## 信息来源 (primary sources, verified)
- [Du et al. 2023 — Multiagent Debate](https://arxiv.org/abs/2305.14325)
- [Liang et al. 2023 — Encouraging Divergent Thinking (MAD)](https://arxiv.org/abs/2305.19118)
- [Chen, Saha, Bansal 2023 — ReConcile](https://arxiv.org/abs/2309.13007)
- [Chan et al. 2023 — ChatEval](https://arxiv.org/abs/2308.07201)
- [Yin et al. 2023 — Exchange-of-Thought](https://arxiv.org/abs/2312.01823)
- [Smit et al. 2024 — Should we be going MAD?](https://arxiv.org/abs/2311.17371)
- [Liu et al. 2023 — DyLAN](https://arxiv.org/abs/2310.02170)
- [Irving, Christiano, Amodei 2018 — AI Safety via Debate](https://arxiv.org/abs/1805.00899)
- [Khan et al. 2024 — More Persuasive LLMs → Truthful Answers](https://arxiv.org/abs/2402.06782)
- [Kenton et al. 2024 — Weak LLMs Judging Strong LLMs](https://arxiv.org/abs/2407.04622)
- [Li et al. 2024 — Sparse Communication Topology (EMNLP Findings)](https://aclanthology.org/2024.findings-emnlp.427.pdf)
- [GroupDebate 2024](https://arxiv.org/abs/2409.14051)
- [S²-MAD 2025](https://arxiv.org/abs/2502.04790)
- [CortexDebate 2025 (ACL Findings)](https://aclanthology.org/2025.findings-acl.495.pdf)
- [AgentPrune / Cut the Crap 2025 (ICLR)](https://arxiv.org/abs/2410.02506)
- [GPTSwarm 2024 (ICML)](https://arxiv.org/abs/2402.16823)
- [MoDS 2025 (NAACL)](https://aclanthology.org/2025.naacl-long.20/)
- [SocraSynth 2024](https://arxiv.org/abs/2402.06634)
- [Mixture-of-Agents 2024 (ICLR 2025)](https://arxiv.org/abs/2406.04692)
- [More Agents Is All You Need 2024 (TMLR)](https://arxiv.org/abs/2402.05120)
- [Free-MAD 2025](https://arxiv.org/abs/2509.11035)
- [Roundtable Policy 2025](https://arxiv.org/abs/2509.16839)
- [MADKE 2023/2025](https://arxiv.org/abs/2312.04854)
- [Rethinking the Bounds of LLM Reasoning 2024 (ACL)](https://arxiv.org/abs/2402.18272)
- [Stop Overvaluing MAD 2025](https://arxiv.org/abs/2502.08788)
- [CONSENSAGENT 2025 (ACL Findings)](https://aclanthology.org/2025.findings-acl.1141.pdf)
- [Talk Isn't Always Cheap 2025](https://arxiv.org/abs/2509.05396)
- [Peacemaker or Troublemaker (sycophancy) 2025](https://arxiv.org/abs/2509.23055)
- [Conformal Social Choice 2026](https://arxiv.org/abs/2604.07667)
- [MA Debate for LLM Judges + Adaptive Stopping 2025 (NeurIPS)](https://openreview.net/forum?id=Vusd1Hw2D9)
- [Beyond Self-Talk: Communication-Centric Survey 2025](https://arxiv.org/abs/2502.14321)

**Bottom line:** The 7 seed items are all accurate but cover only the "early canonical MAD" core. The slice's most relevant frontier for *mediator-coevo* — explicit mediators/moderators (MoDS, SocraSynth, third-party reweighter), sparse/learned information-filtering topologies (Sparse-MAD, S²-MAD, CortexDebate, AgentPrune, GPTSwarm), non-majority aggregation rules (MoA, Free-MAD, Roundtable, conformal, RCS), the scalable-oversight weak-judge lineage (Irving→Khan→Kenton), and the negative-results/sycophancy literature — is entirely missing and should be added. I also recommend the 9 new fields above, especially `aggregation_rule`, `judge_or_mediator_role`, `information_filtering_granularity`, and `agreement_modulation`, which are the dimensions that actually separate methods in this slice.
