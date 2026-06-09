# Cluster B Supplement — Nearest-Neighbor / Learned-Retriever Based In-Context Demonstration Selection

Research date: 2026-06-07
Topic: nearest-neighbor & learned-retriever based in-context example / demonstration selection for LLM prompting

### 补充Items

**Cluster B — nearest-neighbor / learned-retriever based in-context demonstration selection (NEW items, 2023–present prioritized):**

- **LLM-R (Learning to Retrieve In-Context Examples for LLMs)**: Iteratively trains a bi-encoder dense retriever using a reward model built on frozen-LLM feedback plus knowledge distillation; canonical "retriever trained on LLM feedback," evaluated on 30 tasks with transfer to unseen tasks and varying LLM sizes — Liwei Wang, Nan Yang, Furu Wei, EACL 2024 (arXiv 2307.07164).
- **RetICL (Sequential Retrieval of In-Context Examples with Reinforcement Learning)**: Frames sequential exemplar selection as a Markov decision process and trains the retriever with RL, modeling inter-example dependency and ordering rather than scoring examples independently — Alexander Scarlatos, Andrew Lan, 2023 (arXiv 2305.14502).
- **Learning to Retrieve Iteratively for In-Context Learning**: Converts an off-the-shelf dense retriever into a stateful iterative retriever (+4M params for state encoding) trained via policy-gradient RL with LLM feedback, treating exemplar-set construction as NP-hard combinatorial optimization; generalizes across inference LLMs — Yunmo Chen, Tongfei Chen, Harsh Jhamtani, Patrick Xia, Richard Shin, Jason Eisner, Benjamin Van Durme, EMNLP 2024 (arXiv 2406.14739).
- **Se2 (Sequential Example Selection for In-Context Learning)**: Sequential-aware selection using LLM feedback on varying context plus beam search to construct ordered example *sequences*, addressing the train/inference inconsistency of "select-then-organize"; 42% relative gain over random across 23 tasks — Haoyu Liu, Jianfeng Liu, Shaohan Huang, Yuefeng Zhan, Hao Sun, Weiwei Deng, Furu Wei, Qi Zhang (Microsoft), 2024 (arXiv 2402.13874).
- **Skill-KNN (Skill-Based Few-Shot Selection for In-Context Learning)**: Training-free kNN selection that first generates skill-based descriptions via few-shot prompting to strip surface features before embedding retrieval; strong on cross-domain semantic parsing — Shengnan An, Bo Zhou, Zeqi Lin, Qiang Fu, Bei Chen, Nanning Zheng, Weizhu Chen, Jian-Guang Lou, EMNLP 2023 (arXiv 2305.14210).
- **DemoRank (Selecting Effective Demonstrations for LLMs in Ranking Task)**: Retrieve-then-rerank framework with a dependency-aware demonstration reranker trained via a list-pairwise objective on LLM feedback, explicitly modeling demonstration dependency and diversity — Wenhan Liu, Yutao Zhu, Zhicheng Dou, ACM TOIS 2024 (arXiv 2406.16332).
- **MoD (Mixture of Demonstrations for In-Context Learning)**: Partitions the demonstration pool into expert-governed groups to shrink the search space and uses expert-wise training to suppress unhelpful demos; experts collaboratively retrieve at inference — Song Wang, Zihan Chen, Chengshuai Shi, Cong Shen, Jundong Li, NeurIPS 2024.
- **IDEAL (Influence-Driven Selective Annotations Empower In-Context Learners)**: Builds a directed graph over unlabeled data, quantifies subset influence via a diffusion process, and greedily selects which examples to annotate for the demo pool by maximum marginal gain; cross-cutting graph + influence — Shaokun Zhang, Xiaobo Xia, Zhaoqing Wang, Ling-Hao Chen, Jiale Liu, Qingyun Wu, Tongliang Liu, ICLR 2024 (arXiv 2310.10873).
- **GistScore / Example Gisting**: Trains example encoders with an attention "gist bottleneck" between inputs and outputs to produce a scoring metric for selection; multi-task variant enables training-free ICL on new tasks, ~20% absolute gain over off-the-shelf retrievers and ~1000x faster than the strongest training-free baseline — Shivanshu Gupta et al., ICML 2024 (PMLR v235, gupta24c).
- **Learn-by-interact**: Data-centric agent framework with agentic retrieval (model-based query generation + observation-based matching) over self-synthesized trajectories used as ICL demonstrations; evaluated on SWE-bench/WebArena/OSWorld/Spider2-V — Hongjin Su, Ruoxi Sun, Jinsung Yoon, Pengcheng Yin, Tao Yu, Sercan Ö. Arık (Google / HKU), 2025 (arXiv 2501.10893). Cross-cutting B↔D/E.
- **Self-Generated In-Context Examples for LLM Agents (trajectory bootstrapping)**: Agent builds and curates a database of its own successful trajectories as in-context examples, with database-level (population-based) and exemplar-level utility-based curation, and two-level (trajectory + state) retrieval; ALFWorld 73%→93% — OpenReview 2025 (id YurjMGGTTj). Cross-cutting B↔E (agent memory); directly analogous to mediated-coevolution task/skill retrieval.
- **MART (Multimodal Agent trajectory Retriever)**: Fine-tunes a general-purpose MLLM as a trajectory retriever via interactive-feedback preference pairs (selecting trajectories that most improve task success) plus trajectory abstraction for token efficiency; embodied agents — 2024 (arXiv 2410.03450). Cross-cutting B↔D.

**Foundational / sensitivity items relevant to B (add as context / baselines):**

- **Self-Adaptive ICL (select-then-rank, information-compression perspective)**: Per-sample example selection plus permutation via a minimum-description-length / information-compression criterion in a general select-then-rank framework; ~40% relative improvement over random — Zhiyong Wu, Yaoxiang Wang, Jiacheng Ye, Lingpeng Kong, ACL 2023 (arXiv 2212.10375).
- **Complexity-Based Prompting for Multi-Step Reasoning**: Selects exemplars by reasoning-step complexity (a cheap heuristic alternative to learned retrieval) for multi-step reasoning; annotation-efficient, robust to format perturbation — Yao Fu, Hao Peng, Ashish Sabharwal, Peter Clark, Tushar Khot, ICLR 2023 (arXiv 2210.00720).
- **Fantastically Ordered Prompts and Where to Find Them**: Establishes prompt-*order* sensitivity (permutations range from near-SOTA to random) and proposes an entropy-based ordering selection using a generated artificial dev set; the canonical ordering/sensitivity reference — Yao Lu, Max Bartolo, Alastair Moore, Sebastian Riedel, Pontus Stenetorp, ACL 2022 (arXiv 2104.08786).

**Niche / applied learned-retriever variants for B (domain-specific evidence, optional):**

- **MDR (Model-Specific Demonstration Retrieval at Inference Time)**: Retriever that accounts for per-LLM demonstration bias ("a good demo for one LLM may be bad for another"); tested across 23 datasets / 11 domains with up to 41.2% improvement over model-agnostic methods — Huazheng Wang, Jinming Wu et al., NAACL 2024 (aclanthology 2024.naacl-long.235).
- **RUIE (Retrieval-based Unified Information Extraction)**: First trainable retrieval framework for unified IE — combines LLM preferences with a keyword-enhanced reward model and a bi-encoder retriever trained via contrastive learning + knowledge distillation; universal plugin across LLMs — Xincheng Liao, Junwen Duan, Yixin Huang, Jianxin Wang, COLING 2025 (arXiv 2409.11673).
- **Delta-KNN**: kNN retriever using a delta score (relative gain of each training example) to dynamically select "representatives" for hard tasks (Alzheimer's-disease detection) where similarity-based selection fails — Chuyuan Li, Raymond Li, Thalia Field, Giuseppe Carenini, ACL 2025 (arXiv 2506.03476).
- **Refract ICL**: Studies example selection in the long-context/many-shot (thousands of demos) regime; shows smart selection still matters and proposes repeating challenging examples plus zero-shot error signals — Arjun Akula, Kazuma Hashimoto, Krishna Srinivasan, Aditi Chaudhary, Karthik Raman, Michael Bendersky, 2025 (arXiv 2506.12346).
- **DeTriever (Decoder-representation-based Retriever)**: Learns a weighted combination of LLM hidden states as the example representation, trained with a proxy score estimating relative example benefit; NL2SQL one-shot — Yuxi Feng, Raymond Li, Zhenan Fan, Giuseppe Carenini et al., COLING 2025 (arXiv 2406.07913).
- **Learning to Rank for In-Context Example Retrieval**: Trains the retriever with a *ranking* (preference) objective — rankings derived from LLM likelihood of the correct answer per exemplar — instead of a classification objective, aligning training with score-ranked inference; top-1 SOTA across 9 NLP tasks — Yuwen Ji, Luodan Zhang, Ambyer Han, Haoran Que, Lei Shi, Wang Chao, Yue Zhang, NeurIPS 2025.
- **PromptRefine**: Alternating-minimization example selection leveraging auxiliary example banks from related high-resource languages with multi-task-aligned retrievers plus diversity, for low-resource Indic-language ICL — Soumya Suvra Ghosal, Soumyabrata Pal, Koyel Mukherjee, Dinesh Manocha, NAACL 2025 (arXiv 2412.05710).
- **Dual-Div (Diversity-Enhanced Data-Efficient biomedical ICL)**: Two-stage retrieve-then-rank optimizing representativeness + diversity, finding diversity in initial retrieval more critical than ranking-stage optimization — Jun Wang, Zaifu Zhan, Qixin Zhang, Mingquan Lin, Meijia Song, Rui Zhang, 2025 (arXiv 2508.08140).

**Already-in-framework items confirmed / corrections:**

- **UDR (Unified Demonstration Retriever for In-Context Learning)**: Single multi-task model for demonstration retrieval across many tasks via improved contrastive learning — Xiaonan Li et al., ACL 2023 (aclanthology 2023.acl-long.256). (Present in B as "UDR (Li 2023)" — confirmed correct.)
- **Survey anchor — In-context Learning with Retrieved Demonstrations: A Survey**: Comprehensive review of retrieval models, retrieval training procedures, and inference algorithms for demonstration retrieval — Man Luo, Xin Xu, Yue Liu, Panupong Pasupat, Mehran Kazemi, TMLR 2024 (arXiv 2401.11624). Recommended as the cluster-B survey anchor.
- **Correction**: pin the existing "LLM-R (Wang 2024)" entry to EACL 2024 (Wang/Yang/Wei) to remove year/venue ambiguity. CEIL/DPP (Ye 2023) and Cover-LS (Levy 2023) are correctly present and need no duplication.

### 推荐补充字段

- **dependency_modeling**: Whether the method scores examples independently or models inter-example dependency / set-level interactions (e.g., TopK/KATE/EPR independent vs. CEIL-DPP, RetICL, Se2, DemoRank, MoD set-aware). Captures a first-order axis the current B list does not represent.
- **ordering_permutation_handling**: Whether and how the method addresses demonstration order/permutation — none / heuristic / learned-sequential / search-based (beam). Order sensitivity (Lu 2022) is orthogonal to selection and frequently decisive.
- **training_signal_source**: Granular supervision source — frozen-LLM log-likelihood, RL reward from LLM/task success, contrastive positives/negatives from LLM labeling, knowledge distillation from a reward model, proxy scores, or none (training-free). Refines the existing learned_vs_fixed / training_signal fields specifically for B.
- **set_selection_objective**: The combinatorial objective optimized — pure top-k similarity, MMR/diversity, submodular coverage, DPP, influence/marginal-gain, or RL portfolio. Distinguishes diversity/coverage methods (CEIL, IDEAL, Cover-LS, Dual-Div) from similarity-only.
- **inference_cost_efficiency**: Retrieval-time cost and added parameters (e.g., RetICL / Iterative-Retriever +4M params; GistScore ~1000x faster than the strongest training-free baseline; many-shot token cost). Important for a deployed mediated-coevolution retrieval loop.
- **generalization_scope**: Demonstrated transfer across unseen tasks, unseen inference LLMs (cross-model), and cross-lingual/cross-domain. LLM-R, Iterative-Retriever, UDR, MDR, PromptRefine differ sharply; key for reusability.
- **dynamic_pool_online_update**: Whether the example/memory bank is static vs. dynamic and self-evolving (agentic memory: Learn-by-interact, Self-Generated Trajectories, MART), including phenomena like experience-following and misaligned-replay. Directly maps to coevolving task/skill memory.
- **query_transformation**: Whether the query (or candidate) is rewritten before retrieval — skill descriptions (Skill-KNN), zero-shot-CoT reasoning paths, hypothetical descriptions (HyDE/ToolDreamer-style), or raw text. A distinct mechanistic lever for similarity.
- **example_granularity**: Unit retrieved — full input-output pair, full trajectory, sub-trajectory/snippet, state-level, or abstracted insight (relevant to agentic items: snippet vs. full-trajectory demonstrations).
- **graph_or_structure_used**: Make explicit for B items whether/which graph structure is involved (IDEAL directed influence graph; DPP kernel; tool-dependency graphs), enabling a clean cross-link to cluster C (GraphRAG) and the project's graph-based task retrieval.

### 信息来源

- [LLM-R — Learning to Retrieve In-Context Examples (EACL 2024)](https://aclanthology.org/2024.eacl-long.105.pdf)
- [LLM-R — arXiv 2307.07164](https://arxiv.org/html/2307.07164)
- [RetICL — Sequential Retrieval with RL (arXiv 2305.14502)](https://arxiv.org/abs/2305.14502)
- [Learning to Retrieve Iteratively (EMNLP 2024, arXiv 2406.14739)](https://aclanthology.org/2024.emnlp-main.406.pdf)
- [Se2 — Sequential Example Selection (arXiv 2402.13874)](https://arxiv.org/abs/2402.13874)
- [Skill-KNN — Skill-Based Few-Shot Selection (EMNLP 2023, arXiv 2305.14210)](https://arxiv.org/abs/2305.14210)
- [DemoRank (ACM TOIS 2024, arXiv 2406.16332)](https://arxiv.org/html/2406.16332v1)
- [Mixture of Demonstrations / MoD (NeurIPS 2024)](https://neurips.cc/virtual/2024/poster/93243)
- [Mixture of Demonstrations — OpenReview](https://openreview.net/forum?id=uqxSLoCw3K)
- [IDEAL — Influence-Driven Selective Annotation (ICLR 2024, arXiv 2310.10873)](https://arxiv.org/abs/2310.10873)
- [GistScore (ICML 2024, PMLR v235)](https://proceedings.mlr.press/v235/gupta24c.html)
- [Learn-by-interact (arXiv 2501.10893)](https://arxiv.org/html/2501.10893)
- [Self-Generated In-Context Examples for LLM Agents (OpenReview 2025)](https://openreview.net/pdf?id=YurjMGGTTj)
- [MART — Multimodal trajectory retriever (arXiv 2410.03450)](https://arxiv.org/pdf/2410.03450)
- [Self-Adaptive ICL (ACL 2023, arXiv 2212.10375)](https://arxiv.org/abs/2212.10375)
- [Complexity-Based Prompting (ICLR 2023, arXiv 2210.00720)](https://arxiv.org/abs/2210.00720)
- [Fantastically Ordered Prompts (ACL 2022, arXiv 2104.08786)](https://aclanthology.org/2022.acl-long.556.pdf)
- [MDR — Model-Specific Demonstration Retrieval (NAACL 2024)](https://aclanthology.org/2024.naacl-long.235.pdf)
- [RUIE — Retrieval-based Unified Information Extraction (COLING 2025, arXiv 2409.11673)](https://arxiv.org/abs/2409.11673)
- [Delta-KNN (ACL 2025, arXiv 2506.03476)](https://arxiv.org/abs/2506.03476)
- [Refract ICL (arXiv 2506.12346)](https://arxiv.org/abs/2506.12346)
- [DeTriever (COLING 2025, arXiv 2406.07913)](https://arxiv.org/abs/2406.07913)
- [Learning to Rank for In-Context Example Retrieval (NeurIPS 2025)](https://neurips.cc/virtual/2025/poster/117557)
- [PromptRefine (NAACL 2025, arXiv 2412.05710)](https://arxiv.org/abs/2412.05710)
- [Dual-Div — Diversity-Enhanced biomedical ICL (arXiv 2508.08140)](https://arxiv.org/abs/2508.08140)
- [UDR — Unified Demonstration Retriever (ACL 2023)](https://aclanthology.org/2023.acl-long.256.pdf)
- [In-context Learning with Retrieved Demonstrations: A Survey (TMLR 2024, arXiv 2401.11624)](https://arxiv.org/html/2401.11624v1)
