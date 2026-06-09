# Cluster C — Graph-based Retrieval + LLMs (GraphRAG family)

Research-supplement deliverable. Sub-area C focus: knowledge-graph retrieval, GNN+LLM,
subgraph retrieval, reasoning/traversal over graphs, hierarchical/community graph indexes,
graph-structured memory. Cross-cutting items flagged inline.

All authors/year/venue verified via academic-search MCP (Semantic Scholar) and primary
arXiv / ACL Anthology / VLDB sources.

### 补充Items

- **From Local to Global: A GraphRAG Approach to Query-Focused Summarization** — Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson (Microsoft Research), 2024, arXiv:2404.16130. Canonical primary paper behind the framework's "Microsoft GraphRAG (Edge 2024)" entry; defines community-detection + community-summary + map-reduce global sensemaking. Should be pinned with this exact ID as the cluster-C anchor.

- **HippoRAG 2 / "From RAG to Memory: Non-Parametric Continual Learning for Large Language Models"** — Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, Yu Su, 2025, arXiv:2502.14802 (151 cites). Direct successor to HippoRAG (already in framework); adds deeper passage integration + improved online Personalized PageRank, beating standard RAG on factual, sense-making, AND associative memory simultaneously. Strongest current graph-structured-memory baseline; closes HippoRAG's regression on simple factual recall.

- **LightRAG: Simple and Fast Retrieval-Augmented Generation** — Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang (HKUDS), EMNLP 2024, arXiv:2410.05779 (309 cites). Dual-level (low/high) retrieval over an LLM-built entity-relation graph fused with vector indices, plus incremental update algorithm. De-facto efficiency baseline that nearly every 2025 GraphRAG paper compares against. Major omission from the current list.

- **PathRAG: Pruning Graph-based Retrieval Augmented Generation with Relational Paths** — Boyu Chen, Zirui Guo, Zi Yang, Yuluo Chen, Junze Chen, Zhenghao Liu, Chuan Shi, Cheng Yang, 2025, arXiv:2502.14902 (56 cites). Argues GraphRAG's problem is redundancy not insufficiency; retrieves key relational paths via flow-based pruning + path-based prompting. Directly maps to "retrieve relevant prior cases by graph structure" — path pruning = case-selection mechanism.

- **SubgraphRAG / "Simple is Effective: The Roles of Graphs and LLMs in Knowledge-Graph-Based RAG"** — Mufei Li, Siqi Miao, Pan Li, ICLR 2025, arXiv:2410.20724 (89 cites). Lightweight MLP + parallel triple-scoring subgraph retriever encoding directional structural distance; adjustable subgraph size; smaller LLMs deliver competitive explainable reasoning without fine-tuning. Canonical LLM-era update to the framework's "Subgraph Retrieval SR (Zhang 2022)" slot — a learned, tunable subgraph retriever.

- **Think-on-Graph 2.0 (ToG-2): Deep and Faithful LLM Reasoning with Knowledge-guided RAG** — Shengjie Ma, Chengjin Xu, Xuhui Jiang, Muzhi Li, Huaijun Qu, Cehao Yang, Jiaxin Mao, Jian Guo, ICLR 2025, arXiv:2407.10805 (74 cites). Hybrid tight-coupling of graph retrieval and document (context) retrieval, alternating iteratively; training-free, plug-and-play. Successor to ToG (already listed); reference point for agentic-traversal + hybrid retrieval.

- **GFM-RAG: Graph Foundation Model for Retrieval Augmented Generation** — Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Dinh Phung, Chen Gong, Shirui Pan, 2025, arXiv:2502.01113 (33 cites). 8M-param GNN graph foundation model pre-trained on 60 KGs / 14M triples / 700k docs; first GFM that generalizes to unseen datasets for retrieval without fine-tuning. Strongly relevant to "learned similarity metrics for transfer" — a transferable graph retriever (GNN-backbone analog of cluster F).

- **KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation** — Lei Liang, Mengshu Sun, Zhengke Gui, Zhongshu Zhu, Zhouyu Jiang, Ling Zhong, Peilong Zhao, Zhongpu Bo, Jin Yang, et al. (Ant Group / OpenSPG), 2024, arXiv:2409.13731. KG+vector mutual-indexing with logical-form-guided hybrid reasoning engine and knowledge alignment; addresses the vector-similarity vs. knowledge-reasoning gap central to task retrieval. Cross-cutting with E (logical-form solver). Production-grade.

- **Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning** — Haoran Luo, Haihong E, Guanting Chen, Qika Lin, Yikai Guo, Fangzhi Xu, Zemin Kuang, Meina Song, Xiaobao Wu, Yifan Zhu, Anh Tuan Luu, 2025, arXiv:2507.21892 (submitted ICLR 2026). Models graph retrieval as a multi-turn agent-environment loop ("think–retrieve–rethink–generate") over a lightweight knowledge hypergraph, trained end-to-end with RL. Leading 2025 agentic/RL graph-traversal exemplar. Cross-cutting with D (agent curriculum/RL).

- **NodeRAG: Structuring Graph-based RAG with Heterogeneous Nodes** — Tianyang Xu, Haojie Zheng, Chengze Li, Hao Chen, Yixin Liu, Ruoxi Chen, Lichao Sun, 2025, arXiv:2504.11544 (17 cites). Heterogeneous-node graph design (entities, summaries, semantic units) enabling clean integration of graph algorithms (PageRank, etc.); beats GraphRAG and LightRAG on indexing/query time and multi-hop QA. Relevant for graph-index design choices.

- **GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of LLMs** — (Shilong) Li et al., EMNLP Findings 2024, arXiv:2406.14550 (aclanthology 2024.findings-emnlp.746). Builds a graph from long text; an autonomous agent calls read-node/read-neighbor functions for coarse-to-fine traversal; a 4k-window model beats GPT-4-128k. Canonical agentic graph-traversal-over-text item. Cross-cutting with E (memory/reflection) and D (planning).

- **GRAG: Graph Retrieval-Augmented Generation** — Yuntong Hu et al., NAACL Findings 2025, aclanthology 2025.findings-naacl.232. Retrieves textual subgraphs via a linear-time divide-and-conquer strategy and feeds dual text-view + graph-view into the LLM. Subgraph retrieval over text-attributed graphs for multi-hop; methodological contrast to G-Retriever's Steiner-tree approach.

- **Zep: A Temporal Knowledge Graph Architecture for Agent Memory** — Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, Daniel Chalef, 2025, arXiv:2501.13956 (197 cites). Production memory layer built on Graphiti, a temporally-aware KG engine fusing conversational + business data with historical (time-valid) edges; beats MemGPT on DMR and LongMemEval. Premier dynamic/temporal graph-structured-memory item for agents — squarely on the "graph-structured memory" sub-focus and missing from the list.

- **A-MEM: Agentic Memory for LLM Agents** — Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang, 2025, arXiv:2502.12110 (603 cites). Zettelkasten-style dynamically self-organizing memory network — new memories generate structured notes, auto-link to related historical memories, and trigger memory evolution/updates. Directly relevant to retrieving prior cases/experiences by learned graph links. Cross-cutting with E (case-based/generative-agent memory).

- **Mem0 / Mem0^g: Building Production-Ready AI Agents with Scalable Long-Term Memory** — Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, Deshraj Yadav, 2025, arXiv:2504.19413 (390 cites). Scalable long-term memory with an optional graph-memory variant (Mem0^g) capturing relational structure among conversational elements; strong LOCOMO results with 90%+ token/latency savings. Practical graph-memory baseline. Cross-cutting with E.

- **GNN-RAG: Graph Neural Retrieval for Large Language Model Reasoning** — Costas Mavromatis, George Karypis, 2024, arXiv:2405.20139 (180 cites). Already in framework — supplying verified canonical citation: GNN reasons over a dense KG subgraph to retrieve answer candidates, shortest paths verbalized for LLM RAG; SOTA on WebQSP/CWQ with a 7B tuned LLM. Confirmed as the key GNN-retriever-then-LLM item.

- **G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and QA** — Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, Bryan Hooi, 2024, arXiv:2402.07630 (279 cites). Already in framework — supplying verified canonical citation: formulates graph RAG as a Prize-Collecting Steiner Tree problem with soft-prompted GNN+LLM; resists hallucination and scales beyond context window.

- **Reasoning on Graphs (RoG): Faithful and Interpretable LLM Reasoning** — Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, Shirui Pan, 2023, arXiv:2310.01061 (525 cites). Already in framework — supplying verified canonical citation: planning-retrieval-reasoning framework generating KG-grounded relation paths as faithful plans.

- **Graph of Thoughts: Solving Elaborate Problems with Large Language Models** — Maciej Besta, Nils Blach, Aleš Kubíček, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, Torsten Hoefler, AAAI 2024, arXiv:2308.09687 (1368 cites). Cross-cutting: models LLM thoughts as an arbitrary graph with feedback/aggregation; foundational for graph-structured reasoning (vs. retrieval). Relevant as the reasoning-graph counterpart to traversal-based GraphRAG.

- **Survey: Graph Retrieval-Augmented Generation: A Survey** — Boci Peng, Yun Zhu, Yongchao Liu, Xiaohe Bo, Haizhou Shi, Chuntao Hong, Yan Zhang, Siliang Tang, 2024, arXiv:2408.08921 (also ACM TOIS 2025, doi:10.1145/3777378). First GraphRAG survey; G-Indexing / G-Retrieval / G-Generation taxonomy. Cluster-C anchor.

- **Survey: Retrieval-Augmented Generation with Graphs (GraphRAG)** — Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A. Rossi, Subhabrata Mukherjee, Xianfeng Tang, Qi He, Zhigang Hua, Bo Long, Tong Zhao, Neil Shah, Amin Javari, Yinglong Xia, Jiliang Tang, 2024, arXiv:2501.00309 (215 cites). Query-processor/retriever/organizer/generator/data-source framework with per-domain review. Cluster-C anchor.

- **Survey: A Survey of Graph Retrieval-Augmented Generation for Customized LLMs** — Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan Dong, Hao Chen, Yi Chang, Xiao Huang (DEEP-PolyU), 2025, arXiv:2501.13958 (109 cites, 252 refs). Professional-domain focus; maintains Awesome-GraphRAG resource repo. Cluster-C anchor.

- **Benchmark: In-depth Analysis of Graph-based RAG in a Unified Framework** — Yingli Zhou, Yaodong Su, Youran Sun, Shu Wang, Taotao Wang, Runyuan He, Yongwei Zhang, Sicong Liang, Xilin Liu, Yuchi Ma, Yixiang Fang (CUHK-Shenzhen / Huawei Cloud), VLDB 2025, arXiv:2503.04338 (doi:10.14778/3773731.3773738). Open-source testbed re-implementing 12 representative methods (RAPTOR, KGP, DALK, HippoRAG, G-Retriever, ToG, MS GraphRAG, FastGraphRAG, LightRAG, …) over 11 datasets with 100+ variants. Best apples-to-apples benchmark for choosing retrieval components.

- **Benchmark/Framework: LEGO-GraphRAG: Modularizing Graph-based RAG for Design Space Exploration** — Cao et al., VLDB 2025 (doi:10.14778/3748191.3748194). Modular subgraph-extraction + path-retrieval decomposition with structure-based vs. semantic-augmented method classification and reasoning-quality/efficiency/cost trade-off analysis. Useful as a design-space map for the retrieval module.

- **Evaluation: RAG vs. GraphRAG: A Systematic Evaluation and Key Insights** — Haoyu Han, Harry Shomer, Yu Wang, Yongjia Lei, Kai Guo, Zhigang Hua, Bo Long, Hui Liu, Jiliang Tang, 2025, arXiv:2502.11371 (67 cites). Unified evaluation protocol comparing RAG vs. GraphRAG on QA and query-based summarization; analyzes when graph structure helps vs. hurts, failure modes, efficiency trade-offs. Guidance for designing task/case retrieval loops.

- **Iterative retrieval: Beyond Static Retrieval: Opportunities and Pitfalls of Iterative Retrieval in GraphRAG** — Kai Guo, Xinnan Dai, Shenglai Zeng, Harry Shomer, Haoyu Han, Yu Wang, Jiliang Tang, 2025, arXiv:2509.25530. First systematic study of iterative (multi-round) retrieval within GraphRAG; proposes Bridge-Guided Dual-Thought Retrieval (BDTR) to promote bridge evidence into leading positions. Relevant to multi-hop case-retrieval loop design.

### 推荐补充字段

- **graph_construction_method**: How the graph index is built — LLM open-IE triple extraction vs. pre-existing KG vs. passage/proximity graph vs. heterogeneous nodes vs. hypergraph. Distinguishes LightRAG/MS-GraphRAG/NodeRAG/Graph-R1; determines reusability for an OPD task graph.
- **graph_granularity / node_types**: Entity-level, passage/chunk-level, community/summary-level, tree, or mixed-heterogeneous. Critical because a task-retrieval graph could mix task-nodes, skill-nodes, and case-nodes.
- **retrieval_traversal_strategy**: One-shot vs. iterative/multi-hop vs. agentic multi-turn vs. PageRank/spreading-activation vs. GNN message-passing vs. PCST/Steiner-tree vs. BFS/beam path search. Core mechanism differentiator; maps directly to "traversal over graphs."
- **structure_vs_semantic_signal**: Whether retrieval relevance comes from graph topology (degree, distance, PageRank), from dense/text semantic similarity, or a learned fusion. Operationalizes the "learned vs. fixed similarity metric" axis at the graph level.
- **incremental_update_support**: Whether the graph index updates online without full rebuild (LightRAG incremental algo, Jigsaw-LightRAG delta updates, Zep/Graphiti temporal edges). Essential for a coevolving task/case memory that grows over time.
- **temporal_dynamics**: Whether the method models time-valid edges / memory evolution / recency (Zep, A-MEM, Mem0). New dimension required for the "memory for agents" and coevolution angle; absent from current fields.
- **efficiency_profile (indexing_cost / query_token_cost / latency)**: Token and time cost of building and querying — repeatedly the deciding factor in 2025 papers (PathRAG, NodeRAG, community-embedding GraphRAG report 40–96% token reductions). Needed for practical adoption decisions.
- **training_requirement**: Training-free / plug-and-play vs. fine-tuned retriever vs. RL-trained agent vs. pre-trained graph foundation model. Distinguishes ToG-2/SubgraphRAG vs. GFM-RAG vs. Graph-R1 at deployment-cost level.
- **cross_domain_transferability**: Whether the retriever generalizes to unseen graphs/domains without re-training (GFM-RAG, GFM-Retriever). Directly relevant to the core transfer/transferability research question; links cluster A.
- **hybrid_retrieval_fusion**: Whether and how graph retrieval is combined with vector/BM25 retrieval (KAG, ToG-2, DualGraphRAG, Hybrid GraphRAG). Most production systems are hybrid; a missing but decision-relevant axis.
- **interpretability_output**: Form of explainable evidence returned — relation paths, reasoning chains, subgraph, community report, source provenance. Relevant to auditability of retrieved cases driving agent decisions.
- **memory_role**: Whether the graph is a static knowledge index vs. a persistent, writable agent memory (HippoRAG/HippoRAG 2, Zep, A-MEM, Mem0). Separates "GraphRAG retrieval" from "graph-structured memory" — the two halves of cluster C that the current single subarea label conflates.

### 信息来源
- [From Local to Global: A GraphRAG Approach (Edge et al., 2024)](https://arxiv.org/abs/2404.16130)
- [HippoRAG (Gutiérrez et al., 2024)](https://arxiv.org/abs/2405.14831)
- [From RAG to Memory / HippoRAG 2 (Gutiérrez et al., 2025)](https://arxiv.org/abs/2502.14802)
- [LightRAG (Guo et al., EMNLP 2024)](https://arxiv.org/abs/2410.05779)
- [PathRAG (Chen et al., 2025)](https://arxiv.org/abs/2502.14902)
- [SubgraphRAG / Simple is Effective (Li et al., ICLR 2025)](https://arxiv.org/abs/2410.20724)
- [Think-on-Graph 2.0 (Ma et al., ICLR 2025)](https://arxiv.org/abs/2407.10805)
- [GFM-RAG (Luo et al., 2025)](https://arxiv.org/abs/2502.01113)
- [KAG (Liang et al., 2024)](https://arxiv.org/abs/2409.13731)
- [Graph-R1 (Luo et al., 2025)](https://arxiv.org/abs/2507.21892)
- [NodeRAG (Xu et al., 2025)](https://arxiv.org/abs/2504.11544)
- [GraphReader (Li et al., EMNLP Findings 2024)](https://arxiv.org/abs/2406.14550)
- [GRAG (Hu et al., NAACL Findings 2025)](https://aclanthology.org/2025.findings-naacl.232.pdf)
- [Zep / Graphiti (Rasmussen et al., 2025)](https://arxiv.org/abs/2501.13956)
- [A-MEM (Xu et al., 2025)](https://arxiv.org/abs/2502.12110)
- [Mem0 (Chhikara et al., 2025)](https://arxiv.org/abs/2504.19413)
- [GNN-RAG (Mavromatis & Karypis, 2024)](https://arxiv.org/abs/2405.20139)
- [G-Retriever (He et al., 2024)](https://arxiv.org/abs/2402.07630)
- [Reasoning on Graphs / RoG (Luo et al., 2023)](https://arxiv.org/abs/2310.01061)
- [Graph of Thoughts (Besta et al., AAAI 2024)](https://arxiv.org/abs/2308.09687)
- [GraphRAG Survey (Peng et al., 2024 / ACM TOIS 2025)](https://arxiv.org/abs/2408.08921)
- [RAG with Graphs Survey (Han et al., 2024)](https://arxiv.org/abs/2501.00309)
- [GraphRAG for Customized LLMs Survey (Zhang et al., 2025)](https://arxiv.org/abs/2501.13958)
- [In-depth Analysis of Graph-based RAG, Unified Framework (Zhou et al., VLDB 2025)](https://arxiv.org/abs/2503.04338)
- [LEGO-GraphRAG (Cao et al., VLDB 2025)](https://www.vldb.org/pvldb/vol18/p3269-cao.pdf)
- [RAG vs. GraphRAG Systematic Evaluation (Han et al., 2025)](https://arxiv.org/abs/2502.11371)
- [Beyond Static Retrieval: Iterative Retrieval in GraphRAG (Guo et al., 2025)](https://arxiv.org/abs/2509.25530)
