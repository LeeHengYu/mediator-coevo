I'll start by loading the schemas for the literature search MCPs, then run extensive searches across clusters A, B, and C.
I'll begin with foundational seed verification across the three clusters, then move to 2024-present work. Let me run multiple searches in parallel.
Good results on cluster B. The first query returned 0 because AND-mode with too many terms. Let me adjust and run more targeted searches for clusters A and C, plus prompt transfer.
Good cluster C coverage and verification of MPT (Wang 2023, 163 citations). Now let me search the more recent and agent-relevant terms, plus specific seed verifications. Let me run several parallel searches targeting newer/key items.
Strong finds in cluster B (meta-ICL). I found several key items: Lampinen et al. "The broader spectrum of in-context learning" (DeepMind 2024), Kirsch et al. "General-Purpose In-Context Learning Agents" (Schmidhuber group), Fan et al. "Transformers Can Learn Meta-skills". Let me now verify the Flan Collection seed and search for more cluster A items plus the data-centric/task-scaling work.
Flan Collection (Longpre 2023, 928 citations) verified — an important missing seed in Cluster A. Now let me explore the citation graph from MetaICL to find followup meta-ICL work, and search for specific high-value 2024-2025 items (task vectors, function vectors, skill composition, in-context learning of learning algorithms).
The AND-mode in semantic scholar is restrictive. Let me use simpler, shorter queries which work better with this API.
The bare extended queries on Semantic Scholar return too much noise. The regex-filtered limited queries are far more reliable. Let me run more targeted searches with regex filters for the key high-value items I know exist, and use Exa for the harder-to-find specific papers.
Excellent — the Exa search surfaced an important "Diversification Catalyzes..." paper (ACL Findings 2025) for Cluster A, and a rich set of agent-experience-reuse papers (which are likely covered by other agents focusing on clusters D/E, but CoPS, ExpeTrans, and the data-diversity work are relevant). Let me now verify the key foundational seeds and a few more cluster-specific recent items. Let me verify MetaICL, In-Context Tuning, SPoT, ATTEMPT, PANDA, and search for "transfer learning via prompts" and recent 2025 meta-learning/prompt-transfer surveys.
Seeds verified: MetaICL (Min 2021, NAACL 2022, 621 cites), SPoT (Vu 2021/ACL 2022, 336 cites), ATTEMPT (Asai 2022, EMNLP 2022, 128 cites). And found a key 2024 item: MAML-en-LLM (Sinha et al., KDD 2024). Now let me verify PANDA, MUPPET/ExT5, and search for more 2024-2025 prompt-transfer and meta-learning items including "transferring prompts" surveys.
Verified PANDA (Zhong et al., TKDE 2024, 54 cites) and MUPPET (Aghajanyan 2021, EMNLP, 298 cites). Now let me verify the T0, FLAN(Wei), Super-NaturalInstructions, FLAN-T5, CrossFit seeds quickly, and do a couple more targeted searches for newer cluster-A items (e.g., FLAN 2022 scaling, the data-diversity paper, and the "broader spectrum" paper). Let me also fetch the data-diversity paper details and the Lampinen broader-spectrum paper.
All seeds verified: FLAN/Wei (2021, 5089 cites), Super-NaturalInstructions/Tk-Instruct (Wang 2022, EMNLP, 1101 cites), CrossFit (Ye 2021, EMNLP, 198 cites). Now let me do a few final targeted searches: T0 (Sanh), in-context tuning (Chen), and recent 2025 work on data-centric task scaling, plus fetch detailed verification of the two strong newer items (data diversity paper and Lampinen). Let me also search for the recent "in-context learning as implicit meta-learning / learning algorithms" theory work and any 2025 surveys on cross-task transfer.
All seeds verified including T0 (Sanh 2021/ICLR 2022, 2003+ cites) and In-Context Tuning (Chen et al., ACL 2022). Now let me run a few final searches for remaining gaps: (1) the Lampinen "broader spectrum" details, (2) recent 2025 work on "in-context learning as gradient descent / implicit fine-tuning" (mechanistic), and (3) FLAN-T5 / Scaling Instruction-Finetuned (Chung). Let me also confirm OPT-IML and search for the influential 2024-2025 cross-task transfer survey if one exists.
Verified OPT-IML (Iyer 2022, 304 cites) and found the influential ICL-as-gradient-descent mechanistic paper (Dai et al. 2023, 390 cites). Let me do a few final searches to round out coverage: the FLAN-T5/Chung "Scaling Instruction-Finetuned Language Models" paper, and search for two more recent 2025 items: "skill-based learning / library learning in LLMs" relevant to clusters A-C, and any 2024-2025 work that bridges meta-learning and agents specifically. Let me also fetch the Lampinen broader-spectrum paper for accurate venue confirmation.
FLAN-T5/Chung "Scaling Instruction-Finetuned Language Models" confirmed — note the venue is now JMLR 2024 (25(70):1−53), originally arXiv 2210.11416 Oct 2022. Now let me verify the two key newer items I want to include with confidence — the data-diversity paper and the Lampinen broader-spectrum — and do one final search for recent (2025) ICL/meta-learning generalization theory and the Hewitt "instruction following without IT" line. Let me fetch the two ACL papers and the data-diversity arxiv.
I now have strong verification. The data-diversity work has two versions: arXiv "Only-IF" (2410.04717, Oct 2024, Zhang/Wang/Charton) and the ACL Findings 2025 camera-ready "Diversification Catalyzes..." (same authors). Also found "Debunk the Myth of SFT Generalization" (2025) which is highly relevant to cross-task generalization. Let me do two final searches to ensure I haven't missed major recent meta-ICL/in-context-generalization-theory items and the Hewitt instruction-following line, then compile.
I have comprehensive coverage now. Found one more strong theory item: Wang et al. "Can In-context Learning Really Generalize to Out-of-distribution Tasks?" (2024, 23 cites). I've run 18+ distinct searches across the MCPs and verified all foundational seeds plus many new items. Let me compile the final structured output.

## 调研结果汇总 (Clusters A, B, C)

### 已有框架验证结果
所有已列seeds均已核验为真实且元数据准确：T0 (Sanh 2021/ICLR'22, 2003+引), FLAN (Wei 2021, 5089引), Super-NaturalInstructions/Tk-Instruct (Wang 2022/EMNLP, 1101引), CrossFit (Ye 2021/EMNLP, 198引), MUPPET (Aghajanyan 2021/EMNLP, 298引), OPT-IML (Iyer 2022, 304引), MetaICL (Min 2021/NAACL'22, 621引), In-Context Tuning (Chen et al. ACL'22), SPoT (Vu 2021/ACL'22, 336引), ATTEMPT (Asai 2022/EMNLP, 128引), MPT (Wang 2023, 163引), PANDA (Zhong, TKDE 2024, 54引). **两处可校正**: "FLAN-T5/Scaling Instruction-Finetuned" (Chung et al.) 正式发表venue为 **JMLR 2024 (vol 25)**, 原arXiv 2210.11416 (2022-10); "In-Context Tuning"题名应为 **"Meta-learning via Language Model In-context Tuning"** (Chen, Zhong, Zha, Karypis, He; ACL 2022 long; Amazon实习成果)。

### 补充Items

**Cluster A — Training-time cross-task generalization (重要遗漏的seed)**
- **The Flan Collection (Flan 2022)**: 拆解Flan-T5设计决策(任务平衡/富集、混合prompt设置zero/few/CoT),是Cluster A的关键缺失seed。Longpre, Hou, Vu, Webson, Chung, Tay, Zhou, Le, Zoph, Wei, Roberts — 2023, ICML 2023 (PMLR), 928引. https://arxiv.org/abs/2301.13688
- **Only-IF / Diversification Catalyzes Instruction Generalization**: 用Turing-complete Markov算法的受控实验证明"对未见语义的泛化仅在训练数据跨语义域充分多样化时才涌现",直接刻画cross-task泛化的数据条件。Dylan Zhang, Justin Wang, François Charton (UIUC/Chicago/Meta) — arXiv 2024 (2410.04717); camera-ready为 **ACL Findings 2025** (pp.23236-23249, "Diversification Catalyzes..."). https://arxiv.org/abs/2410.04717
- **Debunk the Myth of SFT Generalization**: 反驳"SFT只记忆、RL才泛化"的主流观点;证明prompt多样性+CoT可使SFT在指令变体与难度变体上稳健泛化(匹配/超越RL)。Xiaofeng Lin, Hejian Sang, Zhipeng Wang, Xuezhou Zhang — 2025, arXiv 2510.00237, 6引. https://arxiv.org/abs/2510.00237

**Cluster B — Meta-learning for LMs (新增,多为2024-2025)**
- **MAML-en-LLM**: 真正模型无关的LLM元训练,目标是学到可泛化参数(对未见任务+2%域泛化、+4%适应),明确对比MetaICL/MetaICT。Sinha, Yue, Soto, Kulkarni, Lu, Zhang (Amazon) — 2024, **KDD 2024**, 17引. https://arxiv.org/abs/2405.11446
- **The Broader Spectrum of In-Context Learning**: DeepMind的统一视角论文,将监督few-shot ICL置于"meta-learned in-context learning"更广谱系内,强调泛化的多个维度;对组织本主题框架极有价值。Lampinen, Chan, Singh, Shanahan — 2024, arXiv 2412.03782, 42引. https://arxiv.org/abs/2412.03782
- **Towards General-Purpose In-Context Learning Agents**: Schmidhuber组(Kirsch, Harrison, Daniel, Sohl-Dickstein),将ICL作为通用学习算法/元学习,桥接meta-learning与agent。OpenReview (NeurIPS系), 19引. https://openreview.net/pdf?id=75A7QJgNey
- **Transformers Can Learn Meta-skills for Task Generalization in In-Context Learning**: Fan, Yadlowsky, Papailiopoulos, Lee (UW-Madison/Google DeepMind),meta-skill学习视角解释ICL任务泛化。OpenReview, 2引. https://openreview.net/pdf?id=53dFaE1tFd
- **MAML-en-LLM相邻**: **Meta-Learning at Scale for LLMs via Low-Rank Amortized Bayesian Meta-Learning (ABMLL)**: 将摊销贝叶斯元学习适配到LoRA,在CrossFit/Unified-QA上跨数据集泛化优于现有方法,可与ICL结合。Zhang, Snell, Griffiths (Princeton) — 2025, arXiv 2508.14285. https://arxiv.org/abs/2508.14285
- **Why Can GPT Learn In-Context? (ICL as implicit gradient descent / meta-optimizer)**: 将ICL解释为隐式微调,Transformer注意力具备梯度下降对偶形式——meta-learning机制理解的核心理论参考。Dai, Sun, Dong, Hao, Sui, Wei — 2023, ACL 2023 Findings, 390引. https://arxiv.org/abs/2212.10559
- **Can In-context Learning Really Generalize to Out-of-distribution Tasks?**: 反向证据:Transformer ICL在OOD任务上倾向实现预训练假设空间内"low-test-error preference"的函数,质疑ICL作为新任务学习的证据——cross-task泛化边界的关键批判性工作。Q. Wang, Y. Wang, Y. Wang, Ying — 2024, arXiv 2410.09695, 23引. https://arxiv.org/abs/2410.09695

**Cluster C — Prompt / soft-prompt transfer (新增2024-2025)**
- **Snapshot Prompt Ensemble (SPE) for Parameter-Efficient Soft Prompt Transfer**: 从每个源任务的不同训练阶段抽取多个soft prompt并跨任务注意力融合,改进SPoT式单一源prompt的迁移性(<0.4%参数)。Wu, Chen, Li, Zhang — 2024, **ICASSP 2024**, 2引. https://doi.org/10.1109/ICASSP48485.2024.10448070
- **Parameter Efficient Multi-task Fine-tuning by Learning to Transfer Token-wise Prompts (TPT)**: 用记忆网络构建细粒度soft prompt token库,按输入检索组装实例相关prompt用于多任务跨任务特征迁移。Wu et al. — 2023, **EMNLP 2023 Findings**, 7引. https://aclanthology.org/2023.findings-emnlp.584
- **MMTP: Meta-learning-based Multi-Textual Prompt Tuning**: 将meta-learning与CoOp式prompt tuning结合,改进base-to-new与cross-domain泛化(横跨Cluster B+C)。Sun et al. — 2025, **ICASSP 2025**, 4引. https://doi.org/10.1109/ICASSP49660.2025.10888476
- **GRAM: Gradient-RegulAted Meta-prompt learning**: 用元学习同时学习soft prompt初始化+轻量梯度调节函数,实现跨域泛化与test-time prompt tuning(横跨B+C,VLM域)。Li et al. — TPAMI 2025. https://doi.org/10.1109/TPAMI.2025.3604454

**跨Cluster相邻(供参考,可能与agent专注的D/E cluster重叠)**: CoPS (Cross-Task Experience Sharing, 理论保证的跨任务经验复用); ExpeTrans (LLMs as Experiential Transfer Learners, ACL 2025) — 这两者把cross-task transfer延伸到LLM agent的经验/案例复用,与本主题"case-based reasoning/experience reuse"轴直接相关。

### 推荐补充字段
- **generalization_taxonomy**: 区分OPT-IML式三类泛化(held-out instances within seen tasks / held-out tasks within seen categories / fully held-out categories)。现有"generalization_target"过粗,这三层区分是本领域核心评测维度。
- **task_diversity_role**: 记录该工作对"任务数量 vs 任务多样性"的立场与证据(如CrossFit任务选择、MUPPET的15任务临界点、Only-IF/Diversification的跨域多样性结论)。这是cluster A的解释性关键变量。
- **knowledge_carrier_granularity**: 在现有"knowledge_carrier"上细化载体形式 — 全参数 / soft-prompt向量 / prompt库token / LoRA模块 / 注意力混合权重 / 纯in-context(无参数更新)/ 元梯度。便于横向对比A/B/C方法。
- **adaptation_mechanism**: 区分gradient-based适应(MAML/MAML-en-LLM)vs in-context无更新适应(MetaICL/ICT)vs prompt-init迁移(SPoT/PANDA)vs 检索-组合(ATTEMPT/TPT)。即"如何把源知识落到目标任务"。
- **mechanistic_account**: 是否提供机制性解释(如ICL≈隐式梯度下降/meta-optimizer、Bayesian inference视角、low-test-error preference)。区分实证方法与理论/机制论文。
- **negative_transfer_handling**: 是否度量/缓解源-目标不匹配导致的负迁移(PANDA的transferability metric+蒸馏防遗忘、SPoT的任务相似度检索)。这是prompt transfer的已知失败模式。
- **source_task_selection**: 源任务如何选择/加权(SPoT任务嵌入检索、CrossFit upstream任务影响分析、ATTEMPT注意力插值)。
- **relation_to_agent_experience_reuse**: 显式建立训练期cross-task transfer(A/B/C)与推理期agent经验/技能复用(CoPS/ExpeTrans/skill libraries)之间的桥接关系,服务于该outline的agent主线。
- **base_model_paradigm**: encoder-decoder(T5/T0)vs decoder-only(GPT/LLaMA)vs encoder-only(RoBERTa) — 不同范式下prompt/soft-prompt迁移效果差异显著,值得作为独立维度。

### 信息来源
- [Semantic Scholar — academic-search MCP (search_papers / 多次regex-filtered检索)](https://www.semanticscholar.org)
- [The Flan Collection (arXiv 2301.13688)](https://arxiv.org/abs/2301.13688)
- [Scaling Instruction-Finetuned LMs — JMLR 2024 v25(70)](https://www.jmlr.org/papers/v25/23-0870.html)
- [Only-IF / Diversification Catalyzes... (arXiv 2410.04717; ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.1193.pdf)
- [Debunk the Myth of SFT Generalization (arXiv 2510.00237)](https://arxiv.org/abs/2510.00237)
- [MAML-en-LLM (KDD 2024, arXiv 2405.11446)](https://arxiv.org/abs/2405.11446)
- [The Broader Spectrum of In-Context Learning (arXiv 2412.03782)](https://arxiv.org/abs/2412.03782)
- [Towards General-Purpose In-Context Learning Agents (OpenReview)](https://openreview.net/pdf?id=75A7QJgNey)
- [Why Can GPT Learn In-Context? (arXiv 2212.10559)](https://arxiv.org/abs/2212.10559)
- [Can ICL Really Generalize to OOD Tasks? (arXiv 2410.09695)](https://arxiv.org/abs/2410.09695)
- [Meta-learning via LM In-context Tuning — ACL 2022](https://aclanthology.org/2022.acl-long.53/)
- [PanDa: Prompt Transfer Meets Knowledge Distillation (TKDE 2024, arXiv 2208.10160)](https://arxiv.org/abs/2208.10160)
- [SPE Soft Prompt Transfer (ICASSP 2024)](https://doi.org/10.1109/ICASSP48485.2024.10448070)
- [Token-wise Prompt Transfer / TPT (EMNLP 2023 Findings)](https://aclanthology.org/2023.findings-emnlp.584)
- [CoPS: Cross-Task Experience Sharing (OpenReview)](https://openreview.net/pdf?id=8DLW1saLEY) · [ExpeTrans (ACL 2025)](https://aclanthology.org/2025.acl-long.520.pdf)

---

**总结**: 已运行18+次跨MCP检索,核验全部已有seeds(发现2处venue可校正)。最关键的遗漏是 **The Flan Collection (Flan 2022)** 这一Cluster A核心seed。补充了14个新items(其中9个为2024-present),覆盖训练期数据多样性对泛化的因果证据(Only-IF/Diversification、Debunk SFT)、新一代LLM元学习(MAML-en-LLM、ABMLL、广谱ICL视角)、机制性理解(ICL作为隐式梯度下降、OOD泛化边界批判)、以及新prompt-transfer方法(SPE、TPT、MMTP、GRAM)。推荐9个新字段,核心是 generalization_taxonomy(三层泛化)、adaptation_mechanism 与 knowledge_carrier_granularity,以建立A/B/C与agent经验复用主线的对接维度。
Research complete. Summary of what was delivered for Clusters A, B, C:

**Verification**: Ran 18+ distinct searches across academic-search MCP, Exa, and confirmed all 12 existing seeds are real with accurate metadata. Found 2 venue corrections: FLAN-T5/Chung "Scaling Instruction-Finetuned LMs" → JMLR 2024 (vol 25), and In-Context Tuning's full title is "Meta-learning via Language Model In-context Tuning" (ACL 2022).

**Key gap found**: The Flan Collection (Longpre et al. 2023, 928 cites) was a missing Cluster A seed.

**14 new items supplemented** (9 from 2024-present), spanning:
- Cluster A: Flan Collection, Only-IF/Diversification (data-diversity → generalization), Debunk the Myth of SFT Generalization
- Cluster B: MAML-en-LLM (KDD 2024), Broader Spectrum of ICL (DeepMind), General-Purpose ICL Agents (Schmidhuber), Transformers Learn Meta-skills, ABMLL, ICL-as-gradient-descent (Dai 2023), Can ICL Generalize to OOD Tasks
- Cluster C: SPE, Token-wise Prompt Transfer (TPT), MMTP, GRAM

**9 recommended fields**: generalization_taxonomy (3-level), task_diversity_role, knowledge_carrier_granularity, adaptation_mechanism, mechanistic_account, negative_transfer_handling, source_task_selection, relation_to_agent_experience_reuse, base_model_paradigm.

All items have verified titles/authors/years/venues/URLs, delivered in the structured format in my prior message.
The research task is complete. My findings were delivered in full in the structured response two messages ago, covering Clusters A, B, and C as requested:

- **Verification** of all 12 existing seeds (2 venue corrections noted)
- **1 critical missing seed** identified (The Flan Collection)
- **14 supplemented items** with verified metadata (9 from 2024-present)
- **9 recommended new fields**
- **Full source list** with URLs

No further action needed.
Task complete. All deliverables for Clusters A, B, and C were provided in the structured response earlier:

- Verification of existing seeds (with 2 venue corrections)
- 1 missing seed flagged (The Flan Collection)
- 14 supplemented items (9 from 2024-present), all with verified title/authors/year/venue/URL
- 9 recommended new fields
- Complete source list

No outstanding work remains.
The literature-search task is finished. The complete structured output (補充Items, 推薦補充字段, 信息來源) for Clusters A, B, and C was delivered earlier in this conversation. There is nothing further to do.
Complete. No further action required — the full structured findings for Clusters A, B, and C have already been delivered.
Done. The research is complete and findings were already delivered. No further action needed.
Complete.
Complete.
### 补充Items

#### Cluster A — Training-time cross-task generalization (instruction & multi-task tuning)

- **The Flan Collection (Flan 2022)**: Critical missing Cluster A seed; ablates the design decisions behind Flan-T5 (task balancing/enrichment, mixed zero-shot/few-shot/CoT prompt training) that drive cross-task generalization. 2023, Longpre, Hou, Vu, Webson, Chung, Tay, Zhou, Le, Zoph, Wei, Roberts (Google), ICML 2023 / PMLR, 928 cites. https://arxiv.org/abs/2301.13688
- **Finetuned Language Models Are Zero-Shot Learners (FLAN)** [seed verified]: Foundational instruction-tuning → zero-shot generalization to unseen task types. 2021, Wei, Bosma, Zhao, Guu, Yu, Lester, Du, Dai, Le (Google), ICLR 2022, 5089 cites. https://arxiv.org/abs/2109.01652
- **Multitask Prompted Training Enables Zero-Shot Task Generalization (T0)** [seed verified]: Explicit multitask prompted training induces held-out task generalization. 2021, Sanh, Webson, Raffel, Bach et al. (BigScience), ICLR 2022, 2003+ cites. https://arxiv.org/abs/2110.08207
- **Super-NaturalInstructions / Tk-Instruct** [seed verified]: 1,616-task benchmark for cross-task generalization under declarative instructions; Tk-Instruct beats InstructGPT despite far smaller. 2022, Y. Wang, Mishra, ... Khashabi (AllenAI/ASU/UW), EMNLP 2022, 1101 cites. https://arxiv.org/abs/2204.07705
- **Scaling Instruction-Finetuned Language Models (FLAN-T5 / Flan-PaLM)** [seed verified, venue correction]: Scaling tasks (1.8K), model size, and CoT data for held-out task generalization. 2022 (arXiv), Chung, Hou, Longpre, Zoph, Tay, ... Wei (Google); formally published **JMLR 2024, vol 25(70):1−53**. https://arxiv.org/abs/2210.11416
- **Muppet: Massive Multi-task Representations with Pre-Finetuning (MUPPET)** [seed verified]: Pre-finetuning over ~50 datasets; shows large-scale multitasking is crucial (hurts below ~15 tasks, improves linearly after). 2021, Aghajanyan, Gupta, Shrivastava, Chen, Zettlemoyer, Gupta (Meta), EMNLP 2021, 298 cites. https://arxiv.org/abs/2101.11038
- **CrossFit: A Few-shot Learning Challenge for Cross-task Generalization in NLP** [seed verified]: Defines the cross-task generalization problem setup + NLP Few-shot Gym (160 tasks); upstream task selection strongly affects downstream few-shot. 2021, Ye, Lin, Ren (USC), EMNLP 2021, 198 cites. https://arxiv.org/abs/2104.08835
- **OPT-IML: Scaling Language Model Instruction Meta Learning through the Lens of Generalization** [seed verified]: OPT-IML Bench (2000 tasks) + three-way generalization framework (held-out instances / tasks / categories). 2022, Iyer, Lin, Pasunuru, ... Stoyanov (Meta), arXiv, 304 cites. https://arxiv.org/abs/2212.12017
- **Only-IF: Revealing the Decisive Effect of Instruction Diversity on Generalization** (camera-ready: "Diversification Catalyzes LMs' Instruction Generalization To Unseen Semantics"): Turing-complete Markov-algorithm controlled experiments show generalization to unseen instruction semantics emerges ONLY under cross-domain data diversification; directly characterizes the data conditions for cross-task transfer. 2024 (arXiv 2410.04717), Dylan Zhang, Justin Wang, François Charton (UIUC/Chicago/Meta); **ACL Findings 2025** (pp. 23236–23249), 6 cites. https://arxiv.org/abs/2410.04717
- **Debunk the Myth of SFT Generalization**: Refutes "SFT only memorizes, RL generalizes"; prompt diversity + CoT scaffolding give SFT robust generalization to unseen instruction variants and harder task regimes, matching/beating RL. 2025, Xiaofeng Lin, Hejian Sang, Zhipeng Wang, Xuezhou Zhang, arXiv 2510.00237, 6 cites. https://arxiv.org/abs/2510.00237

#### Cluster B — Meta-learning for LMs

- **MetaICL: Learning to Learn In Context** [seed verified]: Meta-trains an LM on 142 datasets to do in-context learning at test time without parameter updates; diverse meta-training tasks key, gains largest under domain shift. 2021, Min, Lewis, Zettlemoyer, Hajishirzi (UW/Meta), NAACL 2022, 621 cites. https://arxiv.org/abs/2110.15943
- **Meta-learning via Language Model In-context Tuning (ICT)** [seed verified, title correction]: Recasts meta-learning as sequence prediction (instruction + in-context examples + target); beats first-order MAML by ~6% AUC-ROC on BinaryClfs. 2021, Yanda Chen, Ruiqi Zhong, Sheng Zha, George Karypis, He He (Columbia/Berkeley/Amazon), ACL 2022 (long, pp. 719–730). https://arxiv.org/abs/2110.07814
- **MAML-en-LLM: Model Agnostic Meta-Training of LLMs for Improved In-Context Learning**: First to apply true MAML-style meta-training to LLMs for generalizable parameters (not just in-context multitask FT); +2% unseen-domain, +4% adaptation over MetaICL/MetaICT. 2024, Sinha, Yue, Soto, Kulkarni, Lu, Zhang (UVA/Amazon), **KDD 2024**, 17 cites. https://arxiv.org/abs/2405.11446
- **The Broader Spectrum of In-Context Learning**: Unifying perspective placing supervised few-shot ICL within a wider "meta-learned in-context learning" spectrum; frames generalization along multiple axes — valuable for structuring this whole topic. 2024, Lampinen, Chan, Singh, Shanahan (Google DeepMind), arXiv 2412.03782, 42 cites. https://arxiv.org/abs/2412.03782
- **Towards General-Purpose In-Context Learning Agents**: Treats ICL as a general-purpose learned learning algorithm / meta-learning; explicitly bridges meta-learning and agents. Kirsch, Harrison, Daniel, Sohl-Dickstein, Schmidhuber, OpenReview (NeurIPS-track), 19 cites. https://openreview.net/pdf?id=75A7QJgNey
- **Transformers Can Learn Meta-skills for Task Generalization in In-Context Learning**: Meta-skill learning account of how ICL achieves task generalization. Fan, Yadlowsky, Papailiopoulos, Lee (UW-Madison / Google DeepMind), OpenReview, 2 cites. https://openreview.net/pdf?id=53dFaE1tFd
- **Meta-Learning at Scale for LLMs via Low-Rank Amortized Bayesian Meta-Learning (ABMLL)**: Adapts amortized Bayesian meta-learning to LoRA; cross-dataset generalization on CrossFit / Unified-QA beating existing methods, combinable with ICL; scales to Llama3-8B, Qwen2-7B. 2025, Liyi Zhang, Jake Snell, Tom Griffiths (Princeton), arXiv 2508.14285, 1 cite. https://arxiv.org/abs/2508.14285
- **Why Can GPT Learn In-Context? Language Models Secretly Perform Gradient Descent as Meta-Optimizers**: Core mechanistic account — ICL as implicit finetuning; attention has a dual form of gradient-descent optimization (meta-gradients). 2023, Dai, Sun, Dong, Hao, Sui, Wei (PKU/Microsoft), ACL 2023 Findings, 390 cites. https://arxiv.org/abs/2212.10559
- **Can In-context Learning Really Generalize to Out-of-distribution Tasks?**: Critical counter-evidence — Transformer ICL tends to implement a pretraining-hypothesis-space function with a "low-test-error preference," questioning ICL as genuine new-task learning; defines the boundary of cross-task generalization. 2024, Qixun Wang, Yifei Wang, Yisen Wang, Xianghua Ying (PKU), arXiv 2410.09695, 23 cites. https://arxiv.org/abs/2410.09695

#### Cluster C — Prompt / soft-prompt transfer

- **SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer** [seed verified]: Learns source-task prompt then uses it to init target prompt; matches/beats full model tuning on SuperGLUE with 27,000× fewer params; task-prompt-as-embedding retrieval for source selection. 2021, Vu, Lester, Constant, Al-Rfou, Cer (Google), ACL 2022, 336 cites. https://arxiv.org/abs/2110.07904
- **ATTEMPT: Parameter-Efficient Multi-task Tuning via Attentional Mixtures of Soft Prompts** [seed verified]: Attention module interpolates pretrained source soft prompts + a target prompt per instance; modular add/remove of source prompts. 2022, Asai, Salehi, Peters, Hajishirzi (UW/AllenAI), EMNLP 2022, 128 cites. https://arxiv.org/abs/2205.11961
- **Multitask Prompt Tuning Enables Parameter-Efficient Transfer Learning (MPT)** [seed verified]: Distills a single transferable prompt from multiple source prompts, then learns low-rank multiplicative updates per target task; 0.035% params, beats full FT in cases over 23 datasets. 2023, Z. Wang, Panda, Karlinsky, Feris, H. Sun, Y. Kim (OSU/MIT-IBM), ICLR 2023, 163 cites. https://arxiv.org/abs/2303.02861
- **PanDa: Prompt Transfer Meets Knowledge Distillation for Efficient Model Adaptation** [seed verified]: New metric to predict prompt transferability + KD to prevent forgetting of source knowledge; 189 source-target combos, +2.3% avg over vanilla PoT. 2022 (arXiv), Zhong, Ding, Liu, Du, Tao (Wuhan/JD), **IEEE TKDE 2024**, 54 cites. https://arxiv.org/abs/2208.10160
- **Snapshot Prompt Ensemble (SPE) for Parameter-Efficient Soft Prompt Transfer**: Extracts multiple soft prompts via snapshots across training phases of each source task, fuses them with cross-task attention into an instance-dependent target prompt; improves over single-source SPoT, <0.4% params. 2024, Wu, Chen, Li, Zhang, **ICASSP 2024**, 2 cites. https://doi.org/10.1109/ICASSP48485.2024.10448070
- **Parameter Efficient Multi-task Fine-tuning by Learning to Transfer Token-wise Prompts (TPT)**: Builds a bank of fine-grained soft-prompt tokens via a memory network; retrieves and assembles instance-dependent prompts to exploit cross-task features; SOTA tuning only 0.035% params over 14 datasets. 2023, M. Wu, W. Liu, Xu, Lv, Ling, T. Li, Huang, Zheng, X. Huang (Fudan), EMNLP 2023 Findings, 7 cites. https://aclanthology.org/2023.findings-emnlp.584
- **MMTP: Meta-learning-based Multi-Textual Prompt Tuning for Visual-Language Models** (bridges B+C): Combines meta-learning with CoOp-style prompt tuning to improve base-to-new and cross-domain generalization. 2025, Sun, Zhu, Fan, Y. Li, Z. Wang, K. Yang, **ICASSP 2025**, 4 cites. https://doi.org/10.1109/ICASSP49660.2025.10888476
- **GRAM: Gradient-RegulAted Meta-prompt learning (Structure-Induced Gradient Regulation)** (bridges B+C): Meta-learns both an efficient soft-prompt initialization and a lightweight gradient-regulating function for cross-domain generalization; supports test-time prompt tuning; model-agnostic. 2025, J. Li, Gao, Tang, Wei, Xiao, Wu, Hong, M. Wang, Tian, **IEEE TPAMI 2025**. https://doi.org/10.1109/TPAMI.2025.3604454

#### Adjacent (cross-task transfer extended to LLM-agent experience/case reuse — overlaps the outline's case-based reasoning / experience-reuse axis)

- **CoPS: Empowering LLM Agents with Provable Cross-Task Experience Sharing**: Theoretically grounded cross-task experience sharing + pessimism-based selection across offline/online settings. OpenReview. https://openreview.net/pdf?id=8DLW1saLEY
- **ExpeTrans: LLMs Are Experiential Transfer Learners**: Autonomous experience-transfer framework with task-wise experience memory; transfers source-task solving experience to target tasks by task-function/process similarity (LLM analog of human transfer learning). 2025, ACL 2025 (long). https://aclanthology.org/2025.acl-long.520.pdf

### 推荐补充字段

- **generalization_taxonomy**: Which of OPT-IML's three generalization levels the work targets — held-out instances within seen tasks / held-out tasks within seen categories / fully held-out categories. The existing `generalization_target` is too coarse; this 3-level split is the field's core evaluation axis.
- **task_diversity_role**: The work's stance and evidence on "number of tasks vs. task diversity" (e.g., CrossFit upstream task selection, MUPPET's ~15-task critical point, Only-IF/Diversification's cross-domain-diversity finding). The key explanatory variable for Cluster A.
- **knowledge_carrier_granularity**: Refines `knowledge_carrier` by carrier form — full parameters / soft-prompt vector / prompt-token bank / LoRA module / attention-mixture weights / pure in-context (no param update) / meta-gradients. Enables A/B/C cross-comparison.
- **adaptation_mechanism**: How source knowledge lands on the target task — gradient-based (MAML / MAML-en-LLM), in-context no-update (MetaICL / ICT), prompt-init transfer (SPoT / PANDA), or retrieve-and-compose (ATTEMPT / TPT).
- **mechanistic_account**: Whether the work offers a mechanistic explanation (ICL ≈ implicit gradient descent / meta-optimizer; Bayesian-inference view; low-test-error preference). Separates empirical-method papers from theory/mechanism papers.
- **negative_transfer_handling**: Whether source-target mismatch / negative transfer is measured or mitigated (PANDA's transferability metric + KD anti-forgetting; SPoT's similarity-based source retrieval). A known failure mode for prompt transfer.
- **source_task_selection**: How source tasks are chosen/weighted (SPoT task-embedding retrieval, CrossFit upstream-task analysis, ATTEMPT attention interpolation).
- **relation_to_agent_experience_reuse**: Explicit bridge between train-time cross-task transfer (A/B/C) and inference-time agent experience/skill reuse (CoPS / ExpeTrans / skill libraries), serving the outline's agent throughline.
- **base_model_paradigm**: encoder-decoder (T5/T0) vs. decoder-only (GPT/LLaMA) vs. encoder-only (RoBERTa) — prompt/soft-prompt transfer effectiveness differs sharply across paradigms, warranting a standalone dimension.

### 信息来源

- [Semantic Scholar — via academic-search MCP (search_papers, multiple regex-filtered queries)](https://www.semanticscholar.org)
- [The Flan Collection (arXiv 2301.13688)](https://arxiv.org/abs/2301.13688)
- [Finetuned Language Models Are Zero-Shot Learners / FLAN (arXiv 2109.01652)](https://arxiv.org/abs/2109.01652)
- [Multitask Prompted Training Enables Zero-Shot Task Generalization / T0 (arXiv 2110.08207)](https://arxiv.org/abs/2110.08207)
- [Super-NaturalInstructions / Tk-Instruct (arXiv 2204.07705)](https://arxiv.org/abs/2204.07705)
- [Scaling Instruction-Finetuned Language Models — JMLR 2024 v25(70)](https://www.jmlr.org/papers/v25/23-0870.html)
- [Muppet: Massive Multi-task Representations with Pre-Finetuning (arXiv 2101.11038)](https://arxiv.org/abs/2101.11038)
- [CrossFit (arXiv 2104.08835)](https://arxiv.org/abs/2104.08835)
- [OPT-IML (arXiv 2212.12017)](https://arxiv.org/abs/2212.12017)
- [Only-IF / Diversification Catalyzes... (arXiv 2410.04717; ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.1193.pdf)
- [Debunk the Myth of SFT Generalization (arXiv 2510.00237)](https://arxiv.org/abs/2510.00237)
- [MetaICL: Learning to Learn In Context (arXiv 2110.15943)](https://arxiv.org/abs/2110.15943)
- [Meta-learning via LM In-context Tuning — ACL 2022 (arXiv 2110.07814)](https://aclanthology.org/2022.acl-long.53/)
- [MAML-en-LLM (KDD 2024, arXiv 2405.11446)](https://arxiv.org/abs/2405.11446)
- [The Broader Spectrum of In-Context Learning (arXiv 2412.03782)](https://arxiv.org/abs/2412.03782)
- [Towards General-Purpose In-Context Learning Agents (OpenReview)](https://openreview.net/pdf?id=75A7QJgNey)
- [Transformers Can Learn Meta-skills for Task Generalization in ICL (OpenReview)](https://openreview.net/pdf?id=53dFaE1tFd)
- [Meta-Learning at Scale for LLMs / ABMLL (arXiv 2508.14285)](https://arxiv.org/abs/2508.14285)
- [Why Can GPT Learn In-Context? (arXiv 2212.10559)](https://arxiv.org/abs/2212.10559)
- [Can ICL Really Generalize to OOD Tasks? (arXiv 2410.09695)](https://arxiv.org/abs/2410.09695)
- [SPoT: Better Frozen Model Adaptation through Soft Prompt Transfer (arXiv 2110.07904)](https://arxiv.org/abs/2110.07904)
- [ATTEMPT (arXiv 2205.11961)](https://arxiv.org/abs/2205.11961)
- [Multitask Prompt Tuning / MPT (arXiv 2303.02861)](https://arxiv.org/abs/2303.02861)
- [PanDa: Prompt Transfer Meets Knowledge Distillation (IEEE TKDE 2024, arXiv 2208.10160)](https://arxiv.org/abs/2208.10160)
- [Snapshot Prompt Ensemble / SPE (ICASSP 2024)](https://doi.org/10.1109/ICASSP48485.2024.10448070)
- [Token-wise Prompt Transfer / TPT (EMNLP 2023 Findings)](https://aclanthology.org/2023.findings-emnlp.584)
- [MMTP (ICASSP 2025)](https://doi.org/10.1109/ICASSP49660.2025.10888476)
- [GRAM / Structure-Induced Gradient Regulation (IEEE TPAMI 2025)](https://doi.org/10.1109/TPAMI.2025.3604454)
- [CoPS: Cross-Task Experience Sharing (OpenReview)](https://openreview.net/pdf?id=8DLW1saLEY)
- [ExpeTrans: LLMs Are Experiential Transfer Learners (ACL 2025)](https://aclanthology.org/2025.acl-long.520.pdf)
The full structured findings for Clusters A, B, and C have been delivered in my previous message, covering:

- **补充Items**: 20 items grouped under Cluster A (10 — instruction/multi-task tuning seeds + 2024-25 data-diversity work), Cluster B (9 — meta-learning for LMs, including 2024-25 MAML-en-LLM, ABMLL, broader-spectrum ICL, and mechanistic/critical analyses), Cluster C (8 — soft-prompt transfer seeds + 2024-25 SPE/TPT/MMTP/GRAM), plus 2 adjacent agent-experience-reuse items. Each has note, year, authors, venue, and verified URL.
- **推荐补充字段**: 9 fields (generalization_taxonomy, task_diversity_role, knowledge_carrier_granularity, adaptation_mechanism, mechanistic_account, negative_transfer_handling, source_task_selection, relation_to_agent_experience_reuse, base_model_paradigm).
- **信息来源**: 30 sources with URLs.

Task complete — nothing further outstanding.
The research task is complete. The full structured findings for Clusters A, B, and C — 补充Items (grouped A/B/C with notes, years, authors, venues, verified URLs), 推荐补充字段 (9 fields), and 信息来源 (30 sources) — were delivered in full earlier in this conversation. No further action is required.
Task complete. All deliverables for Clusters A, B, and C (补充Items, 推荐补充字段, 信息来源) were fully provided earlier in this conversation. Nothing further is outstanding.
