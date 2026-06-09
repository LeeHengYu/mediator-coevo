# Supplement — Cluster A (task-similarity & transferability estimation) + Cluster F (metric/representation-learning retrieval backbones)

Research question context: How to retrieve relevant prior tasks/skills/examples/cases by similarity — especially via graph structure and learned similarity metrics — to drive transfer, prompting, curriculum, and memory for (LLM) agents. Focus clusters: A (task-similarity & transferability estimation) + F (metric/representation retrieval backbones). 2023–present prioritized; all authors/year/venue verified against primary sources.

### 补充Items

**Cluster A — Task similarity & transferability estimation**

- **SFDA (Self-challenging Fisher Space)**: 投影预训练特征到 self-challenging Fisher 判别空间并通过 Bayes 分类近似 fine-tuning 动态；是 LogME 之后被引用最多的现代可迁移性估计器，几乎所有后续工作的通用 baseline，可直接用作"哪个先验任务最可迁移"的打分器。Shao, Yang, et al., "Not All Models Are Equal: Predicting Model Transferability in a Self-challenging Fisher Space", ECCV 2022 (arXiv 2207.03036).
- **GBC (Gaussian Bhattacharyya Coefficient)**: 用类内特征均值/协方差的 Bhattacharyya 重叠度量做 training-free 可迁移性代理；干净的 embedding-only 相似度度量，可直接计算候选先验任务与目标任务特征分布之间的 GBC。Pándy, Agostinelli, Uijlings, Ferrari, Mensink, "Transferability Estimation using Bhattacharyya Class Separability", CVPR 2022 (arXiv 2111.12780).
- **ETran (Energy-based Transferability)**: 首个能量驱动的可迁移性度量，结合 energy score（目标对候选模型的 in-/out-of-distribution 检验）+ classification score + regression score；不同于 GBC/SFDA/LEEP 还能用于回归任务。Gholami, Akbari, et al., "ETran: Energy-Based Transferability Estimation", ICCV 2023 (arXiv 2308.02027).
- **PED (Potential Energy Decline)**: 引入物理启发的"势能下降"模型来模拟 fine-tuning 中特征的演化，超越静态特征度量；代表可迁移性估计的"动态"分支，直接关系到预测迁移收益。Li et al., 2023（见 2024 transferability 综述 arXiv 2402.15231 引用）。
- **LEAD (Logit Space Evolution)**: 通过 Neural Tangent Kernel 推导 ODE 建模 logits 朝 fine-tuning 后状态的非线性演化，含 class-aware decomposition；在 24 个监督+自监督模型 × 10 数据集上为 SOTA，是最强的 2024 fine-tuning-aware 打分器。Hu, Li, Tang, Liu, Hu, Duan, "LEAD: Exploring Logit Space Evolution for Model Selection", CVPR 2024。**注意：与同年 SF-UniDA 的 "Learning Decomposition (LEAD)" by Qu et al. 不同，需避免混淆。**
- **EMMS (Efficient Multi-task Model Selector)**: 首个多模态多任务场景的可迁移性/模型选择度量；用 foundation model 把异构标签格式（类别、文本、bounding box）映射到统一的 noisy label embedding，再用加权线性回归；展示了如何让可迁移性估计模态无关，对异构 OPD 任务描述符尤为相关。Meng, Shao, Peng, Zhang, Qiao, Luo, "Foundation Model is Efficient Multimodal Multitask Model Selector", NeurIPS 2023 (arXiv 2308.06262).
- **PACTran (PAC-Bayesian Transferability)**: 理论扎实（PAC-Bayesian bound）的可迁移性度量家族；现有框架列了 LogME/LEEP/H-score 却遗漏了这个有原则的后继者，可作为"学习理论"锚点。Ding, Chen, Levinboim, Changpinyo, Soricut, "PACTran: PAC-Bayesian Metrics for Estimating the Transferability of Pretrained Models to Classification Tasks", ECCV 2022 (arXiv 2203.05126).
- **Task-Relatedness**: 给出可迁移性上界，分解为 re-weighted reference-task loss + label mismatch + distribution mismatch，在 penultimate layer 对一个 reference 任务计算（无需目标标签）；概念上最接近通过 reference/anchor 任务的"中介式"迁移，与 mediated-coevolution 任务检索强对齐。Agostinelli et al., "Understanding the Transferability of Representations via Task-Relatedness", NeurIPS 2024.
- **DATE (Discriminability And Transferability Estimation)**: 从 Bayesian 视角连接 source importance 的后验概率与 discriminability 和 transferability，用于多源-free 域适应（MSFDA）的源模型贡献估计；对"从多个先验任务中选择并加权"的检索场景直接相关。Han, Zhang, Wang, He, Su, Xi, Yin, "Discriminability and Transferability Estimation: A Bayesian Source Importance Estimation Approach for Multi-Source-Free Domain Adaptation", AAAI 2023.
- **s-OTDD (Sliced Optimal Transport Dataset Distance)**: 近线性时间、模型无关、embedding 无关的数据集距离（OTDD 的 sliced-OT 推广）；处理 disjoint label set 并与迁移学习性能差距相关；OTDD 在大规模任务距离检索中的可扩展后继者。K. Nguyen, H. Nguyen, Pham, Ho, "Lightspeed Geometric Dataset Distance via Sliced Optimal Transport", ICML 2025 (arXiv 2501.18901).
- **Wasserstein Task Embedding**: 模型无关、training-free 的任务嵌入，用 2-Wasserstein 距离 + MDS 标签嵌入；比 OTDD 快得多，把任务嵌入到 Euclidean 空间使距离近似任务相似度并与 forward/backward transfer 相关；A 簇直接的"把每个任务向量化做 kNN 检索"骨干。Liu, Bai, Lu, Soltoggio, Kolouri, "Wasserstein Task Embedding for Measuring Task Similarities", 2022, Neural Networks 2024 (arXiv 2208.11726).
- **MetaRank**: meta-learning + learning-to-rank 框架，按目标任务选择*使用哪个*可迁移性度量，把数据集和度量的文本描述编码到共享语义空间；重要的横切洞见——没有单一可迁移性度量普适最优，对需要度量集成的稳健任务检索打分器相关。Liu, Zhao, Guo, "MetaRank: Task-Aware Metric Selection for Model Transferability Estimation", arXiv 2511.21007 (2025-11).
- **EMNLP 2023 NLP 可迁移性实证综述**: 系统对比无需暴力 fine-tuning 选择最强 PLM 的度量（loss-approximation 类 vs fine-tuning-dynamics 类）；为 LLM/transformer 任务检索的可迁移性估计提供 NLP 专门视角。"How to Determine the Most Powerful Pre-trained Language Model without Brute Force Fine-tuning? An Empirical Survey", EMNLP Findings 2023.
- **"Choose Your Transformer" (ACL Findings 2024)**: 展示 layer-mean 表示聚合可提升 H-score（+0.13）和 LogME（+0.28）的排序相关性——对 embedding-based 可迁移性任务检索是一个具体、廉价的改进。"Choose Your Transformer: Improved Transferability Estimation of Transformer Models on Classification Tasks", ACL Findings 2024.
- **"Which Model to Transfer? A Survey on Transferability Estimation" (2024)**: 把度量分类为 source-free（SF-MTE）vs source-dependent 的参考综述；应作为 A 簇的组织性脚手架引用。Bao et al., arXiv 2402.15231 (2024).

**Cluster F — Metric/representation-learning retrieval backbones**

- **Contriever**: 经典无监督对比稠密检索器；默认 zero-shot 骨干（如 HyDE 内部使用）；鉴于 DPR/SBERT/ColBERT 已列出而它缺失，是明显遗漏。Izacard, Caron, Hosseini, Riedel, Bojanowski, Joulin, Grave, "Unsupervised Dense Information Retrieval with Contrastive Learning", TMLR 2022 (arXiv 2112.09118).
- **E5**: 首个在 BEIR zero-shot 上无标签击败 BM25 的模型，通过 CCPairs 弱监督对比训练；现代通用嵌入骨干的奠基者。Wang, Yang, Huang, Jiao, Yang, Jiang, Majumder, Wei (Microsoft), "Text Embeddings by Weakly-Supervised Contrastive Pre-training", arXiv 2212.03533 (2022).
- **GTE**: 多阶段（无监督 → 监督）对比训练配方；强 code-search 迁移，被广泛使用的骨干。Li, Zhang, Zhang, Long, Xie, Zhang (Alibaba), "Towards General Text Embeddings with Multi-stage Contrastive Learning", arXiv 2308.03281 (2023).
- **BGE / C-Pack**: BAAI General Embeddings；三阶段配方（plain-text 预训练 → 无标签对比 → 有标签多任务）；最被部署的开源检索骨干之一。Xiao, Liu, Zhang, Muennighoff, et al. (BAAI), "C-Pack: Packaged Resources To Advance General Chinese Embedding", arXiv 2309.07597, SIGIR 2024.
- **E5-mistral**: 主要在 LLM 合成的多语言数据上训练的 decoder-LLM 嵌入器；向 LLM-based embedder 的关键转变（后续由 Linq-Embed-Mistral, arXiv 2412.03223 延续）。Wang, Yang, Huang, Yang, Majumder, Wei, "Improving Text Embeddings with Large Language Models", ICLR 2024 (arXiv 2401.00368).
- **NV-Embed**: decoder-LLM 通用嵌入器，带 latent-attention 池化层、移除 causal mask、两阶段对比指令微调；2024 年 MTEB #1；指令条件检索的 SOTA 表示骨干。Lee, Roy, Xu, Raiman, Shoeybi, Catanzaro, Ping (NVIDIA), "NV-Embed: Improved Techniques for Training LLMs as Generalist Embedding Models", ICLR 2025 (arXiv 2405.17428).
- **INSTRUCTOR**: 单一嵌入器，以任务指令为条件生成任务定制嵌入（330 任务，对比）；直接相关——让一个骨干产生*任务感知*表示用于检索，桥接 A 簇和 F 簇。Su, Shi, Kasai, Wang, Hu, Ostendorf, Yih, Smith, Zettlemoyer, Yu, "One Embedder, Any Task: Instruction-Finetuned Text Embeddings", ACL Findings 2023 (arXiv 2212.09741).
- **ColBERTv2**: 列出的 ColBERT 的去噪监督 + 残差压缩后继者；标准的现代 late-interaction 骨干。Santhanam, Khattab, Saad-Falcon, Potts, Zaharia, "ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction", NAACL 2022 (arXiv 2112.01488).
- **SPLADE / SPLADEv2**: 学习型*稀疏*词汇-扩展检索——与稠密 bi-encoder 正交的 learned-similarity 骨干家族，具有强 out-of-domain 鲁棒性；填补空白（现有框架只有稠密 + late-interaction 骨干）。Formal, Lassance, Piwowarski, Clinchant, "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval", 2021 (arXiv 2109.10086).
- **Matryoshka Representation Learning (MRL)**: 嵌套嵌入，支持推理时弹性维度——可用于成本/延迟可调的任务检索索引（2023–25 后继：2D-MSE, SMEC arXiv 2510.12474）。Kusupati et al., "Matryoshka Representation Learning", NeurIPS 2022 (arXiv 2205.13147).

**Cross-cutting items (flagged)**

- **Agent Workflow Memory (AWM)**: 从过去 agent 轨迹诱导可复用 workflow 并选择性检索以指导新任务；随 train-test 任务差距扩大而泛化；横切 D/E，是最强的近期"按任务相关性为 agent 检索先验技能"项，直接契合 OPD/mediated-coevolution 任务迁移主题。Wang, Mao, Fried, Neubig, "Agent Workflow Memory", ICML 2025 (arXiv 2409.07429).
- **repLLaMA / rankLLaMA**: LLaMA-2 微调为稠密检索器（repLLaMA）+ pointwise reranker（rankLLaMA）；BEIR 上强泛化；横切 F（LLM 作为检索骨干）和 B（基于检索的提示）。Ma, Wang, Yang, Wei, Lin, "Fine-Tuning LLaMA for Multi-Stage Text Retrieval", SIGIR 2024 (arXiv 2310.08319).
- **HyDE**: 用 LLM 生成假设性答案/文档，再用 Contriever 嵌入以检索真实邻居；横切 B/F；当任务描述稀疏需 LLM 扩展后再做相似度检索时相关。Gao, Ma, Lin, Callan, "Precise Zero-Shot Dense Retrieval without Relevance Labels", ACL 2023 (arXiv 2212.10496).

### 推荐补充字段

- **transferability_signal_type**: 区分 source-label-based（LEEP/NCE）vs embedding/feature-based（H-score, GBC, SFDA, LogME, ETran）vs fine-tuning-dynamics/evolution-based（SFDA, PED, LEAD）。这是组织 A 簇的主轴，目前是隐含的；综述文献（Bao 2024）表明它能预测度量行为。
- **requires_target_labels**（none / few-shot / full）: 关键实用维度。许多估计器（Task-Relatedness, LogME-variants）无标签工作，这对在任何目标监督存在之前检索先验任务很重要——直接关系到冷启动 OPD 任务检索。
- **supported_task_outputs / modality_of_target**: classification-only vs regression vs detection vs multimodal。ETran（回归）、EMMS（多模态）、LogME（通用）差异显著；需要知道某度量是否适用于给定 OPD 任务类型。
- **computational_cost / wall_clock_speedup**: 几乎所有可迁移性和检索论文都以速度为核心卖点（EMMS 报告 3.6–6.3× 加速；LogME 是速度参考；s-OTDD 近线性）。成本/延迟字段对判断检索规模下的可行性至关重要。
- **rank_correlation_metric_used**: weighted Kendall's τw / Pearson / Spearman / top-k recall。A 簇论文不记录用于"与 fine-tuning 准确率相关"的度量就无法直接可比（τw 在 LEAD/PED 之后已成标准）。
- **reference_or_anchor_task_used**（布尔 + 描述）: 相似度是直接对目标计算，还是*中介*通过一个 reference/anchor 任务（Task-Relatedness，中介式迁移）。该字段映射到项目的 mediated-coevolution 框架，目前缺失。
- **embedding_pooling_strategy**: mean / [CLS] / last-token / latent-attention（NV-Embed）/ layer-mean（Choose-Your-Transformer）。已被证明会实质性改变检索质量和可迁移性估计准确率。
- **instruction_conditioned**（布尔）: 表示是否以任务指令为条件（INSTRUCTOR, E5-mistral, NV-Embed）vs 固定。决定该骨干能否免费产生*任务感知*相似度。
- **label_set_handling**: 需要共享标签空间 / 处理 disjoint 或 partial 标签集（OTDD, s-OTDD, Wasserstein Task Embedding）。对先验任务与目标任务有不同标签空间的跨任务检索至关重要。
- **adaptable_index_structure**（横切 C/F）: single-vector（ANN）/ multi-vector late-interaction / sparse inverted-index / nested（Matryoshka）。决定集成进任务检索系统的成本。
- **negative_transfer_awareness**: 方法是否显式建模/缓解负迁移（causal task embeddings, Task-Relatedness, multi-source selectors）。重要，因为朴素的相似度检索可能浮现有害的先验。

### 信息来源
- [SFDA – Not All Models Are Equal (Self-challenging Fisher Space), arXiv 2207.03036](https://ar5iv.labs.arxiv.org/html/2207.03036)
- [GBC – Transferability via Bhattacharyya Class Separability (CVPR 2022), arXiv 2111.12780](http://arxiv.org/pdf/2111.12780v1)
- [ETran – Energy-Based Transferability Estimation (ICCV 2023), arXiv 2308.02027](https://export.arxiv.org/pdf/2308.02027v1.pdf)
- [LEAD – Exploring Logit Space Evolution for Model Selection (CVPR 2024)](http://openaccess.thecvf.com/content/CVPR2024/html/Hu_LEAD_Exploring_Logit_Space_Evolution_for_Model_Selection_CVPR_2024_paper.html)
- [EMMS – Foundation Model is Efficient Multimodal Multitask Model Selector (NeurIPS 2023), arXiv 2308.06262](https://arxiv.org/abs/2308.06262)
- [PACTran – PAC-Bayesian Transferability Metrics (ECCV 2022), arXiv 2203.05126](https://arxiv.org/abs/2203.05126)
- [Understanding Transferability via Task-Relatedness (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/hash/d3602fc92fb8b9e0d55356c9e8815e2b-Abstract-Conference.html)
- [DATE – Discriminability and Transferability Estimation (AAAI 2023)](https://ojs.aaai.org/index.php/AAAI/article/download/25946/25718)
- [Which Model to Transfer? A Survey on Transferability Estimation, arXiv 2402.15231](https://arxiv.org/html/2402.15231v1)
- [How to Determine the Most Powerful PLM without Brute Force Fine-tuning (EMNLP Findings 2023)](https://aclanthology.org/2023.findings-emnlp.357.pdf)
- [Choose Your Transformer: Improved Transferability Estimation (ACL Findings 2024)](https://aclanthology.org/2024.findings-acl.757.pdf)
- [Wasserstein Task Embedding for Measuring Task Similarities, arXiv 2208.11726](https://arxiv.org/abs/2208.11726)
- [s-OTDD – Lightspeed Geometric Dataset Distance via Sliced OT (ICML 2025), arXiv 2501.18901](https://arxiv.org/abs/2501.18901)
- [MetaRank – Task-Aware Metric Selection for MTE, arXiv 2511.21007](https://arxiv.org/abs/2511.21007)
- [E5 – Text Embeddings by Weakly-Supervised Contrastive Pre-training, arXiv 2212.03533](https://arxiv.org/abs/2212.03533)
- [GTE – Towards General Text Embeddings with Multi-stage Contrastive Learning, arXiv 2308.03281](https://arxiv.org/pdf/2308.03281)
- [BGE / C-Pack – Packed Resources for General Chinese Embeddings, arXiv 2309.07597](https://arxiv.org/html/2309.07597v4)
- [NV-Embed (ICLR 2025), arXiv 2405.17428](https://arxiv.org/abs/2405.17428)
- [INSTRUCTOR – One Embedder, Any Task (ACL Findings 2023), arXiv 2212.09741](https://arxiv.org/abs/2212.09741)
- [Contriever / HyDE – Precise Zero-Shot Dense Retrieval (ACL 2023), arXiv 2212.10496](https://aclanthology.org/2023.acl-long.99.pdf)
- [Unsupervised Dense Retrieval with Relevance-Aware Contrastive Pre-training (ACL 2023), arXiv 2306.03166](http://arxiv.org/pdf/2306.03166)
- [repLLaMA/rankLLaMA – Fine-Tuning LLaMA for Multi-Stage Text Retrieval (SIGIR 2024), arXiv 2310.08319](https://arxiv.org/pdf/2310.08319)
- [Matryoshka / SMEC – Rethinking MRL for Retrieval Embedding Compression (EMNLP 2025), arXiv 2510.12474](https://arxiv.org/abs/2510.12474)
- [Agent Workflow Memory (ICML 2025), arXiv 2409.07429](https://arxiv.org/abs/2409.07429)
- [A Unified Framework for Learned Sparse Retrieval (ECIR 2023), arXiv 2303.13416](https://arxiv.org/abs/2303.13416)
