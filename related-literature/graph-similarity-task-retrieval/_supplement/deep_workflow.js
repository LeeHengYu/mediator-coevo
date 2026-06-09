export const meta = {
  name: 'research-deep-graph-similarity',
  description: 'Deep-research 144 items (1 Opus-4.8 agent each, 5 concurrent) into validated JSON dossiers',
  phases: [
    { title: 'Cluster A', detail: 'task similarity & transferability', model: 'opus' },
    { title: 'Cluster B', detail: 'retrieval-based ICL selection', model: 'opus' },
    { title: 'Cluster C', detail: 'GraphRAG / GNN+LLM', model: 'opus' },
    { title: 'Cluster D', detail: 'curriculum / skill libraries', model: 'opus' },
    { title: 'Cluster E', detail: 'CBR + agent memory', model: 'opus' },
    { title: 'Cluster F', detail: 'retrieval backbones', model: 'opus' },
  ],
}

const TOPIC = "Graph-based & learned-similarity retrieval for task / skill / case transfer in (LLM) agents"
const FIELDS = "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/fields.yaml"
const RESULTS_DIR = "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results"
const ITEMS = [
{
"id": "A01",
"name": "Task2Vec",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A01_Task2Vec.json",
"info": "name: Task2Vec\nsubarea: A — Task similarity & transferability estimation\nyear: 2019\nvenue: ICCV\nkey_idea: FIM-based task embeddings; distance ~ transferability\npaper_url: https://arxiv.org/abs/1902.03545",
"needs_verification": false
},
{
"id": "A02",
"name": "Taskonomy",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A02_Taskonomy.json",
"info": "name: Taskonomy\nsubarea: A — Task similarity & transferability estimation\nyear: 2018\nvenue: CVPR\nkey_idea: Empirical task-transfer affinity graph across 26 vision tasks\npaper_url: https://arxiv.org/abs/1804.08328",
"needs_verification": false
},
{
"id": "A03",
"name": "TaskEmb (Vu 2020)",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A03_TaskEmb_Vu_2020.json",
"info": "name: TaskEmb (Vu 2020)\naliases: Predicting Transferability across NLP Tasks\nsubarea: A — Task similarity & transferability estimation\nyear: 2020\nvenue: EMNLP\nkey_idea: Task embeddings + data/transfer prediction for intermediate-task selection\npaper_url: https://arxiv.org/abs/2005.00770",
"needs_verification": false
},
{
"id": "A04",
"name": "STILTs / Intermediate-task transfer",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A04_STILTs_Intermediate-task_transfer.json",
"info": "name: STILTs / Intermediate-task transfer\nsubarea: A — Task similarity & transferability estimation\nyear: 2020\nvenue: ACL (Pruksachatkun); 2018 (Phang)\nkey_idea: Supplementary intermediate-task training; when/which task helps\npaper_url: https://arxiv.org/abs/2005.00628",
"needs_verification": false
},
{
"id": "A05",
"name": "LEEP",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A05_LEEP.json",
"info": "name: LEEP\nsubarea: A — Task similarity & transferability estimation\nyear: 2020\nvenue: ICML\nkey_idea: Log Expected Empirical Prediction — label-based transferability score\npaper_url: https://arxiv.org/abs/2002.12462",
"needs_verification": false
},
{
"id": "A06",
"name": "LogME",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A06_LogME.json",
"info": "name: LogME\nsubarea: A — Task similarity & transferability estimation\nyear: 2021\nvenue: ICML\nkey_idea: Log marginal evidence of labels given features; fast model selection\npaper_url: https://arxiv.org/abs/2102.11005",
"needs_verification": false
},
{
"id": "A07",
"name": "H-score",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A07_H-score.json",
"info": "name: H-score\nsubarea: A — Task similarity & transferability estimation\nyear: 2019\nvenue: ICIP\nkey_idea: Information-theoretic feature-vs-label transferability\npaper_url: https://arxiv.org/abs/2212.10082",
"needs_verification": false
},
{
"id": "A08",
"name": "OTCE",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A08_OTCE.json",
"info": "name: OTCE\nsubarea: A — Task similarity & transferability estimation\nyear: 2021\nvenue: CVPR\nkey_idea: Optimal-transport conditional-entropy task similarity\npaper_url: https://arxiv.org/abs/2103.13843",
"needs_verification": false
},
{
"id": "A09",
"name": "TransRate",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A09_TransRate.json",
"info": "name: TransRate\nsubarea: A — Task similarity & transferability estimation\nyear: 2022\nvenue: ICML\nkey_idea: Mutual-information (coding-rate) transferability, label-aware\npaper_url: https://arxiv.org/abs/2106.09362",
"needs_verification": false
},
{
"id": "A10",
"name": "SFDA",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A10_SFDA.json",
"info": "name: SFDA\naliases: Self-challenging Fisher Space\nsubarea: A — Task similarity & transferability estimation\nyear: 2022\nvenue: ECCV\nkey_idea: Project features to self-challenging Fisher space + Bayes classifier\npaper_url: https://arxiv.org/abs/2207.03036",
"needs_verification": false
},
{
"id": "A11",
"name": "GBC",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A11_GBC.json",
"info": "name: GBC\naliases: Gaussian Bhattacharyya Coefficient\nsubarea: A — Task similarity & transferability estimation\nyear: 2022\nvenue: CVPR\nkey_idea: Bhattacharyya class-separability as training-free transferability proxy\npaper_url: https://arxiv.org/abs/2111.12780",
"needs_verification": false
},
{
"id": "A12",
"name": "ETran",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A12_ETran.json",
"info": "name: ETran\nsubarea: A — Task similarity & transferability estimation\nyear: 2023\nvenue: ICCV\nkey_idea: Energy-based transferability (in/out-of-dist) incl. regression\npaper_url: https://arxiv.org/abs/2308.02027",
"needs_verification": false
},
{
"id": "A13",
"name": "PED",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A13_PED.json",
"info": "name: PED\naliases: Potential Energy Decline\nsubarea: A — Task similarity & transferability estimation\nyear: 2023\nvenue: (transferability survey 2402.15231)\nkey_idea: Physics-inspired dynamic model of feature evolution in fine-tuning\npaper_url: https://arxiv.org/abs/2402.15231\nneeds_verification: true",
"needs_verification": true
},
{
"id": "A14",
"name": "LEAD",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A14_LEAD.json",
"info": "name: LEAD\naliases: Logit Space Evolution\nsubarea: A — Task similarity & transferability estimation\nyear: 2024\nvenue: CVPR\nkey_idea: NTK-ODE models nonlinear logit evolution to post-FT state; 2024 SOTA scorer\npaper_url: https://openaccess.thecvf.com/content/CVPR2024/html/Hu_LEAD_Exploring_Logit_Space_Evolution_for_Model_Selection_CVPR_2024_paper.html",
"needs_verification": false
},
{
"id": "A15",
"name": "EMMS",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A15_EMMS.json",
"info": "name: EMMS\naliases: Efficient Multi-task Model Selector\nsubarea: A — Task similarity & transferability estimation\nyear: 2023\nvenue: NeurIPS\nkey_idea: Foundation model unifies heterogeneous labels into noisy-label embedding for multimodal model selection\npaper_url: https://arxiv.org/abs/2308.06262",
"needs_verification": false
},
{
"id": "A16",
"name": "PACTran",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A16_PACTran.json",
"info": "name: PACTran\nsubarea: A — Task similarity & transferability estimation\nyear: 2022\nvenue: ECCV\nkey_idea: PAC-Bayesian transferability metric family (principled bound)\npaper_url: https://arxiv.org/abs/2203.05126",
"needs_verification": false
},
{
"id": "A17",
"name": "Task-Relatedness",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A17_Task-Relatedness.json",
"info": "name: Task-Relatedness\nsubarea: A — Task similarity & transferability estimation\nyear: 2024\nvenue: NeurIPS\nkey_idea: Transferability upper bound via a reference/anchor task — mediated transfer ★\npaper_url: https://proceedings.neurips.cc/paper_files/paper/2024/hash/d3602fc92fb8b9e0d55356c9e8815e2b-Abstract-Conference.html",
"needs_verification": false
},
{
"id": "A18",
"name": "DATE",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A18_DATE.json",
"info": "name: DATE\naliases: Discriminability & Transferability Estimation\nsubarea: A — Task similarity & transferability estimation\nyear: 2023\nvenue: AAAI\nkey_idea: Bayesian source-importance for multi-source-free DA; select+weight prior tasks\npaper_url: https://ojs.aaai.org/index.php/AAAI/article/view/25946",
"needs_verification": false
},
{
"id": "A19",
"name": "s-OTDD",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A19_s-OTDD.json",
"info": "name: s-OTDD\naliases: Sliced Optimal Transport Dataset Distance\nsubarea: A — Task similarity & transferability estimation\nyear: 2025\nvenue: ICML\nkey_idea: Near-linear, model/embedding-free dataset distance; disjoint labels\npaper_url: https://arxiv.org/abs/2501.18901",
"needs_verification": false
},
{
"id": "A20",
"name": "Wasserstein Task Embedding",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A20_Wasserstein_Task_Embedding.json",
"info": "name: Wasserstein Task Embedding\nsubarea: A — Task similarity & transferability estimation\nyear: 2024\nvenue: Neural Networks\nkey_idea: 2-Wasserstein + MDS task vectors in Euclidean space for kNN task retrieval\npaper_url: https://arxiv.org/abs/2208.11726",
"needs_verification": false
},
{
"id": "A21",
"name": "MetaRank",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A21_MetaRank.json",
"info": "name: MetaRank\nsubarea: A — Task similarity & transferability estimation\nyear: 2025\nvenue: arXiv 2511.21007\nkey_idea: Meta-learn which transferability metric to use per target task (no metric is universal)\npaper_url: https://arxiv.org/abs/2511.21007\nneeds_verification: true",
"needs_verification": true
},
{
"id": "A22",
"name": "NLP Transferability Empirical Survey",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A22_NLP_Transferability_Empirical_Survey.json",
"info": "name: NLP Transferability Empirical Survey\naliases: Most Powerful PLM without Brute Force FT\nsubarea: A — Task similarity & transferability estimation\nyear: 2023\nvenue: EMNLP Findings\nkey_idea: Empirical comparison of PLM-selection metrics (loss-approx vs FT-dynamics)\npaper_url: https://aclanthology.org/2023.findings-emnlp.357/",
"needs_verification": false
},
{
"id": "A23",
"name": "Choose Your Transformer",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A23_Choose_Your_Transformer.json",
"info": "name: Choose Your Transformer\nsubarea: A — Task similarity & transferability estimation\nyear: 2024\nvenue: ACL Findings\nkey_idea: Layer-mean aggregation improves H-score/LogME ranking correlation\npaper_url: https://aclanthology.org/2024.findings-acl.757/",
"needs_verification": false
},
{
"id": "A24",
"name": "Which Model to Transfer? (Survey)",
"subarea": "A",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/A24_Which_Model_to_Transfer_Survey.json",
"info": "name: Which Model to Transfer? (Survey)\nsubarea: A — Task similarity & transferability estimation\nyear: 2024\nvenue: arXiv 2402.15231\nkey_idea: Source-free vs source-dependent transferability-estimation taxonomy\npaper_url: https://arxiv.org/abs/2402.15231",
"needs_verification": false
},
{
"id": "B01",
"name": "KATE",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B01_KATE.json",
"info": "name: KATE\naliases: What Makes Good ICL Examples\nsubarea: B — Retrieval-based in-context example selection\nyear: 2021\nvenue: DeeLIO/ACL-WS\nkey_idea: kNN in embedding space picks good GPT-3 demonstrations\npaper_url: https://arxiv.org/abs/2101.06804",
"needs_verification": false
},
{
"id": "B02",
"name": "EPR",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B02_EPR.json",
"info": "name: EPR\naliases: Learning to Retrieve Prompts\nsubarea: B — Retrieval-based in-context example selection\nyear: 2022\nvenue: NAACL\nkey_idea: Train dense demo retriever from LM scoring signal (pos/neg)\npaper_url: https://arxiv.org/abs/2112.08633",
"needs_verification": false
},
{
"id": "B03",
"name": "kNN-LM",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B03_kNN-LM.json",
"info": "name: kNN-LM\nsubarea: B — Retrieval-based in-context example selection\nyear: 2020\nvenue: ICLR\nkey_idea: Interpolate LM with nearest-neighbor datastore over hidden states\npaper_url: https://arxiv.org/abs/1911.00172",
"needs_verification": false
},
{
"id": "B04",
"name": "kNN-Prompting",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B04_kNN-Prompting.json",
"info": "name: kNN-Prompting\nsubarea: B — Retrieval-based in-context example selection\nyear: 2023\nvenue: ICLR\nkey_idea: Calibration-free nearest-neighbor inference over ICL anchors\npaper_url: https://arxiv.org/abs/2303.13824",
"needs_verification": false
},
{
"id": "B05",
"name": "UDR",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B05_UDR.json",
"info": "name: UDR\naliases: Unified Demonstration Retriever\nsubarea: B — Retrieval-based in-context example selection\nyear: 2023\nvenue: ACL\nkey_idea: Single multi-task demo retriever via improved contrastive learning\npaper_url: https://aclanthology.org/2023.acl-long.256/",
"needs_verification": false
},
{
"id": "B06",
"name": "LLM-R",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B06_LLM-R.json",
"info": "name: LLM-R\naliases: Learning to Retrieve ICL Examples for LLMs\nsubarea: B — Retrieval-based in-context example selection\nyear: 2024\nvenue: EACL\nkey_idea: Iteratively train bi-encoder via LLM-feedback reward model + KD; 30 tasks ★\npaper_url: https://aclanthology.org/2024.eacl-long.105/",
"needs_verification": false
},
{
"id": "B07",
"name": "CEIL",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B07_CEIL.json",
"info": "name: CEIL\naliases: Compositional Exemplars / DPP\nsubarea: B — Retrieval-based in-context example selection\nyear: 2023\nvenue: ICML\nkey_idea: DPP models set-level relevance+diversity for exemplar selection\npaper_url: https://arxiv.org/abs/2302.05698",
"needs_verification": false
},
{
"id": "B08",
"name": "Cover-LS",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B08_Cover-LS.json",
"info": "name: Cover-LS\naliases: Diverse Demonstrations / coverage\nsubarea: B — Retrieval-based in-context example selection\nyear: 2023\nvenue: ACL\nkey_idea: Structural coverage retrieval for compositional generalization\npaper_url: https://arxiv.org/abs/2212.06800",
"needs_verification": false
},
{
"id": "B09",
"name": "Analogical Prompting",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B09_Analogical_Prompting.json",
"info": "name: Analogical Prompting\nsubarea: B — Retrieval-based in-context example selection\nyear: 2023\nvenue: ICLR 2024\nkey_idea: LLM self-generates relevant exemplars (no external retrieval)\npaper_url: https://arxiv.org/abs/2310.01714",
"needs_verification": false
},
{
"id": "B10",
"name": "RetICL",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B10_RetICL.json",
"info": "name: RetICL\nsubarea: B — Retrieval-based in-context example selection\nyear: 2023\nvenue: arXiv 2305.14502\nkey_idea: Sequential exemplar selection as MDP, trained with RL (ordering+dependency)\npaper_url: https://arxiv.org/abs/2305.14502",
"needs_verification": false
},
{
"id": "B11",
"name": "Learning to Retrieve Iteratively",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B11_Learning_to_Retrieve_Iteratively.json",
"info": "name: Learning to Retrieve Iteratively\nsubarea: B — Retrieval-based in-context example selection\nyear: 2024\nvenue: EMNLP\nkey_idea: Stateful iterative retriever (+4M params), policy-gradient w/ LLM feedback\npaper_url: https://aclanthology.org/2024.emnlp-main.406/",
"needs_verification": false
},
{
"id": "B12",
"name": "Se2",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B12_Se2.json",
"info": "name: Se2\naliases: Sequential Example Selection\nsubarea: B — Retrieval-based in-context example selection\nyear: 2024\nvenue: arXiv 2402.13874\nkey_idea: Sequential-aware selection + beam search to build ordered example sequences\npaper_url: https://arxiv.org/abs/2402.13874",
"needs_verification": false
},
{
"id": "B13",
"name": "Skill-KNN",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B13_Skill-KNN.json",
"info": "name: Skill-KNN\nsubarea: B — Retrieval-based in-context example selection\nyear: 2023\nvenue: EMNLP\nkey_idea: Generate skill-based descriptions before embedding → training-free kNN\npaper_url: https://arxiv.org/abs/2305.14210",
"needs_verification": false
},
{
"id": "B14",
"name": "DemoRank",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B14_DemoRank.json",
"info": "name: DemoRank\nsubarea: B — Retrieval-based in-context example selection\nyear: 2024\nvenue: ACM TOIS\nkey_idea: Retrieve-then-rerank with dependency-aware demo reranker (list-pairwise)\npaper_url: https://arxiv.org/abs/2406.16332",
"needs_verification": false
},
{
"id": "B15",
"name": "MoD",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B15_MoD.json",
"info": "name: MoD\naliases: Mixture of Demonstrations\nsubarea: B — Retrieval-based in-context example selection\nyear: 2024\nvenue: NeurIPS\nkey_idea: Partition demo pool into expert groups to shrink search space\npaper_url: https://openreview.net/forum?id=uqxSLoCw3K",
"needs_verification": false
},
{
"id": "B16",
"name": "IDEAL",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B16_IDEAL.json",
"info": "name: IDEAL\naliases: Influence-Driven Selective Annotation\nsubarea: B — Retrieval-based in-context example selection\nyear: 2024\nvenue: ICLR\nkey_idea: Directed influence graph + diffusion to pick which examples to annotate (graph ★)\npaper_url: https://arxiv.org/abs/2310.10873",
"needs_verification": false
},
{
"id": "B17",
"name": "GistScore",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B17_GistScore.json",
"info": "name: GistScore\naliases: Example Gisting\nsubarea: B — Retrieval-based in-context example selection\nyear: 2024\nvenue: ICML\nkey_idea: Attention gist-bottleneck encoder scores examples; ~1000x faster, multi-task\npaper_url: https://proceedings.mlr.press/v235/gupta24c.html",
"needs_verification": false
},
{
"id": "B18",
"name": "Learn-by-interact",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B18_Learn-by-interact.json",
"info": "name: Learn-by-interact\nsubarea: B — Retrieval-based in-context example selection\nyear: 2025\nvenue: arXiv 2501.10893\nkey_idea: Agentic retrieval over self-synthesized trajectories as ICL demos (B↔D/E ★)\npaper_url: https://arxiv.org/abs/2501.10893",
"needs_verification": false
},
{
"id": "B19",
"name": "Self-Generated ICL Examples for Agents",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B19_Self-Generated_ICL_Examples_for_Agents.json",
"info": "name: Self-Generated ICL Examples for Agents\nsubarea: B — Retrieval-based in-context example selection\nyear: 2025\nvenue: OpenReview YurjMGGTTj\nkey_idea: Agent curates DB of own successful trajectories; ALFWorld 73→93% (B↔E ★)\npaper_url: https://openreview.net/forum?id=YurjMGGTTj\nneeds_verification: true",
"needs_verification": true
},
{
"id": "B20",
"name": "MART",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B20_MART.json",
"info": "name: MART\naliases: Multimodal Agent trajectory Retriever\nsubarea: B — Retrieval-based in-context example selection\nyear: 2024\nvenue: arXiv 2410.03450\nkey_idea: Fine-tune MLLM as trajectory retriever via interactive-feedback preference pairs (B↔D)\npaper_url: https://arxiv.org/abs/2410.03450",
"needs_verification": false
},
{
"id": "B21",
"name": "Self-Adaptive ICL",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B21_Self-Adaptive_ICL.json",
"info": "name: Self-Adaptive ICL\nsubarea: B — Retrieval-based in-context example selection\nyear: 2023\nvenue: ACL\nkey_idea: Select-then-rank via MDL/information-compression criterion\npaper_url: https://arxiv.org/abs/2212.10375",
"needs_verification": false
},
{
"id": "B22",
"name": "Complexity-Based Prompting",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B22_Complexity-Based_Prompting.json",
"info": "name: Complexity-Based Prompting\nsubarea: B — Retrieval-based in-context example selection\nyear: 2023\nvenue: ICLR\nkey_idea: Select exemplars by reasoning-step complexity (cheap heuristic)\npaper_url: https://arxiv.org/abs/2210.00720",
"needs_verification": false
},
{
"id": "B23",
"name": "Fantastically Ordered Prompts",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B23_Fantastically_Ordered_Prompts.json",
"info": "name: Fantastically Ordered Prompts\nsubarea: B — Retrieval-based in-context example selection\nyear: 2022\nvenue: ACL\nkey_idea: Canonical prompt-ORDER sensitivity + entropy-based ordering selection\npaper_url: https://aclanthology.org/2022.acl-long.556/",
"needs_verification": false
},
{
"id": "B24",
"name": "MDR",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B24_MDR.json",
"info": "name: MDR\naliases: Model-Specific Demonstration Retrieval\nsubarea: B — Retrieval-based in-context example selection\nyear: 2024\nvenue: NAACL\nkey_idea: Account for per-LLM demo bias; good demo for one LLM ≠ another\npaper_url: https://aclanthology.org/2024.naacl-long.235/",
"needs_verification": false
},
{
"id": "B25",
"name": "RUIE",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B25_RUIE.json",
"info": "name: RUIE\naliases: Retrieval-based Unified IE\nsubarea: B — Retrieval-based in-context example selection\nyear: 2025\nvenue: COLING\nkey_idea: Trainable retrieval for unified IE; LLM-pref reward + contrastive + KD\npaper_url: https://arxiv.org/abs/2409.11673",
"needs_verification": false
},
{
"id": "B26",
"name": "Delta-KNN",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B26_Delta-KNN.json",
"info": "name: Delta-KNN\nsubarea: B — Retrieval-based in-context example selection\nyear: 2025\nvenue: ACL\nkey_idea: Delta-score (relative gain) selects representatives when similarity fails\npaper_url: https://arxiv.org/abs/2506.03476",
"needs_verification": false
},
{
"id": "B27",
"name": "Refract ICL",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B27_Refract_ICL.json",
"info": "name: Refract ICL\nsubarea: B — Retrieval-based in-context example selection\nyear: 2025\nvenue: arXiv 2506.12346\nkey_idea: Selection in long-context/many-shot regime; repeat hard examples + zero-shot error\npaper_url: https://arxiv.org/abs/2506.12346\nneeds_verification: true",
"needs_verification": true
},
{
"id": "B28",
"name": "DeTriever",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B28_DeTriever.json",
"info": "name: DeTriever\nsubarea: B — Retrieval-based in-context example selection\nyear: 2025\nvenue: COLING\nkey_idea: Weighted LLM hidden states as example representation; proxy-score training\npaper_url: https://arxiv.org/abs/2406.07913",
"needs_verification": false
},
{
"id": "B29",
"name": "Learning to Rank for ICE Retrieval",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B29_Learning_to_Rank_for_ICE_Retrieval.json",
"info": "name: Learning to Rank for ICE Retrieval\nsubarea: B — Retrieval-based in-context example selection\nyear: 2025\nvenue: NeurIPS\nkey_idea: Train retriever with ranking/preference objective from LLM likelihood\npaper_url: https://neurips.cc/virtual/2025/poster/117557\nneeds_verification: true",
"needs_verification": true
},
{
"id": "B30",
"name": "PromptRefine",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B30_PromptRefine.json",
"info": "name: PromptRefine\nsubarea: B — Retrieval-based in-context example selection\nyear: 2025\nvenue: NAACL\nkey_idea: Alternating-minimization selection w/ auxiliary high-resource banks (Indic ICL)\npaper_url: https://arxiv.org/abs/2412.05710",
"needs_verification": false
},
{
"id": "B31",
"name": "Dual-Div",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B31_Dual-Div.json",
"info": "name: Dual-Div\nsubarea: B — Retrieval-based in-context example selection\nyear: 2025\nvenue: arXiv 2508.08140\nkey_idea: Two-stage retrieve-then-rank; diversity in initial retrieval > ranking opt\npaper_url: https://arxiv.org/abs/2508.08140\nneeds_verification: true",
"needs_verification": true
},
{
"id": "B32",
"name": "ICL with Retrieved Demonstrations: A Survey",
"subarea": "B",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/B32_ICL_with_Retrieved_Demonstrations_A_Survey.json",
"info": "name: ICL with Retrieved Demonstrations: A Survey\nsubarea: B — Retrieval-based in-context example selection\nyear: 2024\nvenue: TMLR\nkey_idea: Survey anchor: retrieval models / training / inference algorithms\npaper_url: https://arxiv.org/abs/2401.11624",
"needs_verification": false
},
{
"id": "C01",
"name": "Microsoft GraphRAG",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C01_Microsoft_GraphRAG.json",
"info": "name: Microsoft GraphRAG\naliases: From Local to Global\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2024\nvenue: arXiv 2404.16130\nkey_idea: Community detection + community summaries + map-reduce global sensemaking ★\npaper_url: https://arxiv.org/abs/2404.16130",
"needs_verification": false
},
{
"id": "C02",
"name": "G-Retriever",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C02_G-Retriever.json",
"info": "name: G-Retriever\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2024\nvenue: NeurIPS\nkey_idea: Graph RAG as Prize-Collecting Steiner Tree; soft-prompted GNN+LLM ★\npaper_url: https://arxiv.org/abs/2402.07630",
"needs_verification": false
},
{
"id": "C03",
"name": "GNN-RAG",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C03_GNN-RAG.json",
"info": "name: GNN-RAG\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2024\nvenue: arXiv 2405.20139\nkey_idea: GNN reasons over dense KG subgraph; shortest paths verbalized for LLM ★\npaper_url: https://arxiv.org/abs/2405.20139",
"needs_verification": false
},
{
"id": "C04",
"name": "Subgraph Retrieval (SR)",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C04_Subgraph_Retrieval_SR.json",
"info": "name: Subgraph Retrieval (SR)\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2022\nvenue: ACL\nkey_idea: Trainable subgraph retriever decoupled from KBQA reasoner\npaper_url: https://arxiv.org/abs/2202.13296",
"needs_verification": false
},
{
"id": "C05",
"name": "RoG",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C05_RoG.json",
"info": "name: RoG\naliases: Reasoning on Graphs\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2023\nvenue: ICLR 2024\nkey_idea: Planning-retrieval-reasoning; KG relation paths as faithful plans\npaper_url: https://arxiv.org/abs/2310.01061",
"needs_verification": false
},
{
"id": "C06",
"name": "ToG",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C06_ToG.json",
"info": "name: ToG\naliases: Think-on-Graph\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2024\nvenue: ICLR\nkey_idea: LLM agent beam-searches over KG (iterative traversal)\npaper_url: https://arxiv.org/abs/2307.07697",
"needs_verification": false
},
{
"id": "C07",
"name": "StructGPT",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C07_StructGPT.json",
"info": "name: StructGPT\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2023\nvenue: EMNLP\nkey_idea: Iterative reading-then-reasoning over structured data (KG/table/DB)\npaper_url: https://arxiv.org/abs/2305.09645",
"needs_verification": false
},
{
"id": "C08",
"name": "RAPTOR",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C08_RAPTOR.json",
"info": "name: RAPTOR\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2024\nvenue: ICLR\nkey_idea: Recursive clustering+summarization tree for multi-level retrieval\npaper_url: https://arxiv.org/abs/2401.18059",
"needs_verification": false
},
{
"id": "C09",
"name": "HippoRAG",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C09_HippoRAG.json",
"info": "name: HippoRAG\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2024\nvenue: NeurIPS\nkey_idea: Hippocampal-index: KG + Personalized PageRank single-step multi-hop memory ★\npaper_url: https://arxiv.org/abs/2405.14831",
"needs_verification": false
},
{
"id": "C10",
"name": "HippoRAG 2",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C10_HippoRAG_2.json",
"info": "name: HippoRAG 2\naliases: From RAG to Memory\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: arXiv 2502.14802\nkey_idea: Deeper passage integration + online PPR; factual+sensemaking+associative memory\npaper_url: https://arxiv.org/abs/2502.14802",
"needs_verification": false
},
{
"id": "C11",
"name": "LightRAG",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C11_LightRAG.json",
"info": "name: LightRAG\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2024\nvenue: EMNLP\nkey_idea: Dual-level retrieval over LLM-built entity-relation graph + vector; incremental update\npaper_url: https://arxiv.org/abs/2410.05779",
"needs_verification": false
},
{
"id": "C12",
"name": "PathRAG",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C12_PathRAG.json",
"info": "name: PathRAG\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: arXiv 2502.14902\nkey_idea: Flow-based pruning of key relational paths (redundancy, not insufficiency) ★\npaper_url: https://arxiv.org/abs/2502.14902",
"needs_verification": false
},
{
"id": "C13",
"name": "SubgraphRAG",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C13_SubgraphRAG.json",
"info": "name: SubgraphRAG\naliases: Simple is Effective\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: ICLR\nkey_idea: Lightweight MLP + parallel triple-scoring; tunable subgraph, directional distance ★\npaper_url: https://arxiv.org/abs/2410.20724",
"needs_verification": false
},
{
"id": "C14",
"name": "ToG-2",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C14_ToG-2.json",
"info": "name: ToG-2\naliases: Think-on-Graph 2.0\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: ICLR\nkey_idea: Tight-coupled iterative graph + document retrieval; training-free\npaper_url: https://arxiv.org/abs/2407.10805",
"needs_verification": false
},
{
"id": "C15",
"name": "GFM-RAG",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C15_GFM-RAG.json",
"info": "name: GFM-RAG\naliases: Graph Foundation Model RAG\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: arXiv 2502.01113\nkey_idea: 8M-param GNN graph FM pretrained on 60 KGs; generalizes to unseen graphs (C↔A/F ★)\npaper_url: https://arxiv.org/abs/2502.01113",
"needs_verification": false
},
{
"id": "C16",
"name": "KAG",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C16_KAG.json",
"info": "name: KAG\naliases: Knowledge Augmented Generation\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2024\nvenue: arXiv 2409.13731\nkey_idea: KG+vector mutual index + logical-form hybrid reasoning (C↔E)\npaper_url: https://arxiv.org/abs/2409.13731",
"needs_verification": false
},
{
"id": "C17",
"name": "Graph-R1",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C17_Graph-R1.json",
"info": "name: Graph-R1\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: arXiv 2507.21892\nkey_idea: Agentic GraphRAG: think-retrieve-rethink-generate over hypergraph, end-to-end RL (C↔D ★)\npaper_url: https://arxiv.org/abs/2507.21892\nneeds_verification: true",
"needs_verification": true
},
{
"id": "C18",
"name": "NodeRAG",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C18_NodeRAG.json",
"info": "name: NodeRAG\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: arXiv 2504.11544\nkey_idea: Heterogeneous-node graph (entities/summaries/semantic units) for clean graph algos\npaper_url: https://arxiv.org/abs/2504.11544",
"needs_verification": false
},
{
"id": "C19",
"name": "GraphReader",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C19_GraphReader.json",
"info": "name: GraphReader\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2024\nvenue: EMNLP Findings\nkey_idea: Agent traverses text-built graph via read-node/read-neighbor; 4k beats GPT4-128k (C↔D/E)\npaper_url: https://arxiv.org/abs/2406.14550",
"needs_verification": false
},
{
"id": "C20",
"name": "GRAG",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C20_GRAG.json",
"info": "name: GRAG\naliases: Graph Retrieval-Augmented Generation\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: NAACL Findings\nkey_idea: Linear-time textual-subgraph retrieval; dual text-view + graph-view to LLM\npaper_url: https://aclanthology.org/2025.findings-naacl.232/",
"needs_verification": false
},
{
"id": "C21",
"name": "Zep / Graphiti",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C21_Zep_Graphiti.json",
"info": "name: Zep / Graphiti\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: arXiv 2501.13956\nkey_idea: Temporal KG agent memory w/ time-valid edges; beats MemGPT (graph memory ★)\npaper_url: https://arxiv.org/abs/2501.13956",
"needs_verification": false
},
{
"id": "C22",
"name": "A-MEM",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C22_A-MEM.json",
"info": "name: A-MEM\naliases: Agentic Memory\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: arXiv 2502.12110\nkey_idea: Zettelkasten self-organizing memory graph; auto-link + memory evolution (C↔E ★)\npaper_url: https://arxiv.org/abs/2502.12110",
"needs_verification": false
},
{
"id": "C23",
"name": "Mem0 / Mem0^g",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C23_Mem0_Mem0g.json",
"info": "name: Mem0 / Mem0^g\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: arXiv 2504.19413\nkey_idea: Scalable long-term memory + optional graph variant; 90%+ token savings (C↔E)\npaper_url: https://arxiv.org/abs/2504.19413",
"needs_verification": false
},
{
"id": "C24",
"name": "Graph of Thoughts",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C24_Graph_of_Thoughts.json",
"info": "name: Graph of Thoughts\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2024\nvenue: AAAI\nkey_idea: LLM thoughts as arbitrary graph w/ aggregation/feedback (reasoning-graph)\npaper_url: https://arxiv.org/abs/2308.09687",
"needs_verification": false
},
{
"id": "C25",
"name": "GraphRAG Survey (Peng)",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C25_GraphRAG_Survey_Peng.json",
"info": "name: GraphRAG Survey (Peng)\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2024\nvenue: arXiv 2408.08921 / ACM TOIS 2025\nkey_idea: First GraphRAG survey: G-Indexing / G-Retrieval / G-Generation taxonomy\npaper_url: https://arxiv.org/abs/2408.08921",
"needs_verification": false
},
{
"id": "C26",
"name": "RAG with Graphs Survey (Han)",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C26_RAG_with_Graphs_Survey_Han.json",
"info": "name: RAG with Graphs Survey (Han)\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: arXiv 2501.00309\nkey_idea: Query-processor/retriever/organizer/generator framework, per-domain\npaper_url: https://arxiv.org/abs/2501.00309",
"needs_verification": false
},
{
"id": "C27",
"name": "GraphRAG for Customized LLMs Survey (Zhang)",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C27_GraphRAG_for_Customized_LLMs_Survey_Zhang.json",
"info": "name: GraphRAG for Customized LLMs Survey (Zhang)\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: arXiv 2501.13958\nkey_idea: Professional-domain GraphRAG; maintains Awesome-GraphRAG repo\npaper_url: https://arxiv.org/abs/2501.13958",
"needs_verification": false
},
{
"id": "C28",
"name": "In-depth Analysis of Graph-based RAG (Benchmark)",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C28_In-depth_Analysis_of_Graph-based_RAG_Benchmark.json",
"info": "name: In-depth Analysis of Graph-based RAG (Benchmark)\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: VLDB\nkey_idea: Unified testbed: 12 methods × 11 datasets, 100+ variants (apples-to-apples)\npaper_url: https://arxiv.org/abs/2503.04338",
"needs_verification": false
},
{
"id": "C29",
"name": "LEGO-GraphRAG",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C29_LEGO-GraphRAG.json",
"info": "name: LEGO-GraphRAG\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: VLDB\nkey_idea: Modular subgraph-extraction + path-retrieval design-space exploration\npaper_url: https://www.vldb.org/pvldb/vol18/p3269-cao.pdf",
"needs_verification": false
},
{
"id": "C30",
"name": "RAG vs GraphRAG Evaluation",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C30_RAG_vs_GraphRAG_Evaluation.json",
"info": "name: RAG vs GraphRAG Evaluation\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: arXiv 2502.11371\nkey_idea: When graph structure helps vs hurts; failure modes, efficiency trade-offs\npaper_url: https://arxiv.org/abs/2502.11371",
"needs_verification": false
},
{
"id": "C31",
"name": "Beyond Static Retrieval (Iterative GraphRAG)",
"subarea": "C",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/C31_Beyond_Static_Retrieval_Iterative_GraphRAG.json",
"info": "name: Beyond Static Retrieval (Iterative GraphRAG)\nsubarea: C — Graph-based retrieval + LLM (GraphRAG)\nyear: 2025\nvenue: arXiv 2509.25530\nkey_idea: First study of iterative retrieval in GraphRAG; Bridge-Guided Dual-Thought\npaper_url: https://arxiv.org/abs/2509.25530\nneeds_verification: true",
"needs_verification": true
},
{
"id": "D01",
"name": "Curriculum Learning",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D01_Curriculum_Learning.json",
"info": "name: Curriculum Learning\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2009\nvenue: ICML\nkey_idea: Foundational easy→hard ordered training\npaper_url: https://dl.acm.org/doi/10.1145/1553374.1553380",
"needs_verification": false
},
{
"id": "D02",
"name": "Teacher-Student Curriculum Learning",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D02_Teacher-Student_Curriculum_Learning.json",
"info": "name: Teacher-Student Curriculum Learning\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2019\nvenue: IEEE TNNLS\nkey_idea: Teacher picks tasks maximizing student learning progress\npaper_url: https://arxiv.org/abs/1707.00183",
"needs_verification": false
},
{
"id": "D03",
"name": "POET",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D03_POET.json",
"info": "name: POET\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2019\nvenue: GECCO/arXiv\nkey_idea: Open-ended coevolution of environments and agents\npaper_url: https://arxiv.org/abs/1901.01753",
"needs_verification": false
},
{
"id": "D04",
"name": "PAIRED",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D04_PAIRED.json",
"info": "name: PAIRED\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2020\nvenue: NeurIPS\nkey_idea: Adversarial environment generation via regret (unsupervised env design)\npaper_url: https://arxiv.org/abs/2012.02096",
"needs_verification": false
},
{
"id": "D05",
"name": "Voyager",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D05_Voyager.json",
"info": "name: Voyager\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2023\nvenue: TMLR\nkey_idea: LLM agent w/ growing skill library + automatic curriculum (Minecraft) ★\npaper_url: https://arxiv.org/abs/2305.16291",
"needs_verification": false
},
{
"id": "D06",
"name": "ExpeL",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D06_ExpeL.json",
"info": "name: ExpeL\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2024\nvenue: AAAI\nkey_idea: Experiential learner: extract insights + retrieve past experiences\npaper_url: https://arxiv.org/abs/2308.10144",
"needs_verification": false
},
{
"id": "D07",
"name": "ALP-GMM",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D07_ALP-GMM.json",
"info": "name: ALP-GMM\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2019\nvenue: CoRL\nkey_idea: Absolute-learning-progress GMM task sampling (continuous curriculum)\npaper_url: https://arxiv.org/abs/1910.07224",
"needs_verification": false
},
{
"id": "D08",
"name": "GITM",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D08_GITM.json",
"info": "name: GITM\naliases: Ghost in the Minecraft\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2023\nvenue: arXiv 2305.17144\nkey_idea: Structured text knowledge/memory + goal-decomposition tree\npaper_url: https://arxiv.org/abs/2305.17144",
"needs_verification": false
},
{
"id": "D09",
"name": "JARVIS-1",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D09_JARVIS-1.json",
"info": "name: JARVIS-1\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2023\nvenue: IEEE TPAMI 2025\nkey_idea: Multimodal memory-augmented open-world agent; experience retrieval for lifelong learning\npaper_url: https://arxiv.org/abs/2311.05997",
"needs_verification": false
},
{
"id": "D10",
"name": "Agent Workflow Memory (AWM)",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D10_Agent_Workflow_Memory_AWM.json",
"info": "name: Agent Workflow Memory (AWM)\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2024\nvenue: ICML 2025\nkey_idea: Induce reusable workflows from trajectories + selective retrieval (D/E ★★)\npaper_url: https://arxiv.org/abs/2409.07429",
"needs_verification": false
},
{
"id": "D11",
"name": "Agent Skill Induction (ASI)",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D11_Agent_Skill_Induction_ASI.json",
"info": "name: Agent Skill Induction (ASI)\naliases: Inducing Programmatic Skills\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2025\nvenue: arXiv 2504.06821\nkey_idea: Skills as verified executable programs injected into action space (AWM successor ★)\npaper_url: https://arxiv.org/abs/2504.06821",
"needs_verification": false
},
{
"id": "D12",
"name": "SkillWeaver",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D12_SkillWeaver.json",
"info": "name: SkillWeaver\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2025\nvenue: arXiv 2504.07079\nkey_idea: Web agent synthesizes skills as APIs, practices, retrieves; strong→weak transfer +54% ★\npaper_url: https://arxiv.org/abs/2504.07079",
"needs_verification": false
},
{
"id": "D13",
"name": "PAE",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D13_PAE.json",
"info": "name: PAE\naliases: Proposer-Agent-Evaluator\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2024\nvenue: ICML 2025\nkey_idea: Task proposer + VLM evaluator → auto-curriculum skill discovery (RL) ★\npaper_url: https://arxiv.org/abs/2412.13194",
"needs_verification": false
},
{
"id": "D14",
"name": "Mobile-Agent-E",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D14_Mobile-Agent-E.json",
"info": "name: Mobile-Agent-E\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2025\nvenue: arXiv 2501.11733\nkey_idea: Self-evolving GUI agent: long-term Tips + reusable Shortcuts\npaper_url: https://arxiv.org/abs/2501.11733",
"needs_verification": false
},
{
"id": "D15",
"name": "AutoManual",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D15_AutoManual.json",
"info": "name: AutoManual\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2024\nvenue: NeurIPS\nkey_idea: Build rules/manuals from interaction + case-conditioned prompting (D↔E)\npaper_url: https://arxiv.org/abs/2405.16247",
"needs_verification": false
},
{
"id": "D16",
"name": "Eurekaverse",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D16_Eurekaverse.json",
"info": "name: Eurekaverse\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2024\nvenue: CoRL\nkey_idea: LLM generates unsupervised curriculum of envs as code\npaper_url: https://arxiv.org/abs/2411.01775",
"needs_verification": false
},
{
"id": "D17",
"name": "R-Zero",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D17_R-Zero.json",
"info": "name: R-Zero\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2025\nvenue: arXiv 2508.05004\nkey_idea: Challenger+Solver co-evolve self-curriculum at edge of capability (ZPD) ★★\npaper_url: https://arxiv.org/abs/2508.05004",
"needs_verification": false
},
{
"id": "D18",
"name": "Absolute Zero / AZR",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D18_Absolute_Zero_AZR.json",
"info": "name: Absolute Zero / AZR\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2025\nvenue: arXiv 2505.03335\nkey_idea: Single model proposes+solves tasks maximizing own learning; code-executor reward ★\npaper_url: https://arxiv.org/abs/2505.03335",
"needs_verification": false
},
{
"id": "D19",
"name": "Self-Evolving Curriculum (SEC)",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D19_Self-Evolving_Curriculum_SEC.json",
"info": "name: Self-Evolving Curriculum (SEC)\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2025\nvenue: arXiv 2505.14970\nkey_idea: Curriculum/task selection as non-stationary bandit; absolute-advantage learning gain\npaper_url: https://arxiv.org/abs/2505.14970",
"needs_verification": false
},
{
"id": "D20",
"name": "SkillGraph",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D20_SkillGraph.json",
"info": "name: SkillGraph\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2026\nvenue: arXiv 2604.19793\nkey_idea: Execution-transition graph from 50k trajectories; graph≫semantic for ordering (τ −0.43→+0.61) ★★\npaper_url: https://arxiv.org/abs/2604.19793\nneeds_verification: true",
"needs_verification": true
},
{
"id": "D21",
"name": "Survey of Self-Evolving Agents",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D21_Survey_of_Self-Evolving_Agents.json",
"info": "name: Survey of Self-Evolving Agents\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2025\nvenue: arXiv 2507.21046\nkey_idea: What/When/How/Where to evolve; co-evolutionary dynamics — D+E scoping anchor\npaper_url: https://arxiv.org/abs/2507.21046",
"needs_verification": false
},
{
"id": "D22",
"name": "SkillRL",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D22_SkillRL.json",
"info": "name: SkillRL\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2026\nvenue: arXiv 2602.08234\nkey_idea: Recursive skill-augmented RL; SkillBank co-evolves with policy\npaper_url: https://arxiv.org/abs/2602.08234\nneeds_verification: true",
"needs_verification": true
},
{
"id": "D23",
"name": "Skill-Pro",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D23_Skill-Pro.json",
"info": "name: Skill-Pro\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2026\nvenue: arXiv 2602.01869\nkey_idea: Procedural skills via non-parametric PPO over Skill-MDP; PPO-gate reuse\npaper_url: https://arxiv.org/abs/2602.01869\nneeds_verification: true",
"needs_verification": true
},
{
"id": "D24",
"name": "PolySkill",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D24_PolySkill.json",
"info": "name: PolySkill\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2025\nvenue: arXiv 2510.15863\nkey_idea: Domain-driven skill hierarchy via polymorphic abstraction; cross-site transfer\npaper_url: https://arxiv.org/abs/2510.15863\nneeds_verification: true",
"needs_verification": true
},
{
"id": "D25",
"name": "SkillX",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D25_SkillX.json",
"info": "name: SkillX\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2026\nvenue: arXiv 2604.04804\nkey_idea: Auto multi-level (Planning/Functional/Atomic) skill KB from trajectories\npaper_url: https://arxiv.org/abs/2604.04804\nneeds_verification: true",
"needs_verification": true
},
{
"id": "D26",
"name": "SkillPyramid",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D26_SkillPyramid.json",
"info": "name: SkillPyramid\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2026\nvenue: arXiv 2606.03692\nkey_idea: Hierarchical skill consolidation; Skill Creator composes existing skills\npaper_url: https://arxiv.org/abs/2606.03692\nneeds_verification: true",
"needs_verification": true
},
{
"id": "D27",
"name": "Trace2Skill",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D27_Trace2Skill.json",
"info": "name: Trace2Skill\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2026\nvenue: arXiv 2603.25158\nkey_idea: Distill trajectory-local lessons into transferable skills; conflict-free merge\npaper_url: https://arxiv.org/abs/2603.25158\nneeds_verification: true",
"needs_verification": true
},
{
"id": "D28",
"name": "SkillFlow",
"subarea": "D",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/D28_SkillFlow.json",
"info": "name: SkillFlow\nsubarea: D — Curriculum / skill libraries / task-graphs for agents\nyear: 2025\nvenue: arXiv 2504.06188\nkey_idea: Skill retrieval over large heterogeneous community skill libraries (SKILL.md)\npaper_url: https://arxiv.org/abs/2504.06188\nneeds_verification: true",
"needs_verification": true
},
{
"id": "E01",
"name": "CBR cycle (Aamodt & Plaza)",
"subarea": "E",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/E01_CBR_cycle_Aamodt_Plaza.json",
"info": "name: CBR cycle (Aamodt & Plaza)\nsubarea: E — Case-based reasoning + agent memory\nyear: 1994\nvenue: AI Communications\nkey_idea: Retrieve-Reuse-Revise-Retain — foundational CBR 4R cycle ★\npaper_url: https://content.iospress.com/articles/ai-communications/aic7-1-04",
"needs_verification": false
},
{
"id": "E02",
"name": "CBR-RAG",
"subarea": "E",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/E02_CBR-RAG.json",
"info": "name: CBR-RAG\nsubarea: E — Case-based reasoning + agent memory\nyear: 2024\nvenue: ICCBR\nkey_idea: CBR-structured retrieval augmentation for LLM generation\npaper_url: https://arxiv.org/abs/2404.04302",
"needs_verification": false
},
{
"id": "E03",
"name": "Review of CBR for LLM Agents",
"subarea": "E",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/E03_Review_of_CBR_for_LLM_Agents.json",
"info": "name: Review of CBR for LLM Agents\nsubarea: E — Case-based reasoning + agent memory\nyear: 2025\nvenue: arXiv 2504.06943\nkey_idea: Math model of case retrieval/adaptation/learning; CBR vs CoT vs RAG (survey anchor)\npaper_url: https://arxiv.org/abs/2504.06943",
"needs_verification": false
},
{
"id": "E04",
"name": "Generative Agents memory",
"subarea": "E",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/E04_Generative_Agents_memory.json",
"info": "name: Generative Agents memory\nsubarea: E — Case-based reasoning + agent memory\nyear: 2023\nvenue: UIST\nkey_idea: Memory stream scored by recency × importance × relevance ★\npaper_url: https://arxiv.org/abs/2304.03442",
"needs_verification": false
},
{
"id": "E05",
"name": "Reflexion",
"subarea": "E",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/E05_Reflexion.json",
"info": "name: Reflexion\nsubarea: E — Case-based reasoning + agent memory\nyear: 2023\nvenue: NeurIPS\nkey_idea: Verbal self-reflection stored in episodic memory for retry\npaper_url: https://arxiv.org/abs/2303.11366",
"needs_verification": false
},
{
"id": "E06",
"name": "DS-Agent",
"subarea": "E",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/E06_DS-Agent.json",
"info": "name: DS-Agent\nsubarea: E — Case-based reasoning + agent memory\nyear: 2024\nvenue: ICML\nkey_idea: Full 4R CBR LLM agent for data science; retrieve Kaggle cases, revise, retain ★\npaper_url: https://arxiv.org/abs/2402.17453",
"needs_verification": false
},
{
"id": "E07",
"name": "MCBR-RAG",
"subarea": "E",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/E07_MCBR-RAG.json",
"info": "name: MCBR-RAG\naliases: Multimodal CBR-RAG\nsubarea: E — Case-based reasoning + agent memory\nyear: 2025\nvenue: arXiv 2501.05030\nkey_idea: CBR-RAG for multimodal cases via learned indexable latent representations\npaper_url: https://arxiv.org/abs/2501.05030\nneeds_verification: true",
"needs_verification": true
},
{
"id": "E08",
"name": "CBR-DDI",
"subarea": "E",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/E08_CBR-DDI.json",
"info": "name: CBR-DDI\nsubarea: E — Case-based reasoning + agent memory\nyear: 2025\nvenue: arXiv 2505.23034\nkey_idea: CBR + LLM-GNN collaborative case retrieval (graph-aware cases; C↔E)\npaper_url: https://arxiv.org/abs/2505.23034\nneeds_verification: true",
"needs_verification": true
},
{
"id": "E09",
"name": "Synapse",
"subarea": "E",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/E09_Synapse.json",
"info": "name: Synapse\nsubarea: E — Case-based reasoning + agent memory\nyear: 2023\nvenue: ICLR 2024\nkey_idea: Trajectory-as-exemplar prompting; similarity-retrieved exemplar memory + state abstraction ★\npaper_url: https://arxiv.org/abs/2306.07863",
"needs_verification": false
},
{
"id": "E10",
"name": "RAP",
"subarea": "E",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/E10_RAP.json",
"info": "name: RAP\naliases: Retrieval-Augmented Planning\nsubarea: E — Case-based reasoning + agent memory\nyear: 2024\nvenue: arXiv 2402.03610\nkey_idea: Retrieve past experiences matching current context to guide planning (text+multimodal)\npaper_url: https://arxiv.org/abs/2402.03610",
"needs_verification": false
},
{
"id": "E11",
"name": "Memp",
"subarea": "E",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/E11_Memp.json",
"info": "name: Memp\naliases: Procedural Memory\nsubarea: E — Case-based reasoning + agent memory\nyear: 2025\nvenue: arXiv 2508.06433\nkey_idea: Learnable lifelong procedural memory: step + script abstractions; Build/Retrieve/Update\npaper_url: https://arxiv.org/abs/2508.06433",
"needs_verification": false
},
{
"id": "E12",
"name": "MemGen",
"subarea": "E",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/E12_MemGen.json",
"info": "name: MemGen\nsubarea: E — Case-based reasoning + agent memory\nyear: 2025\nvenue: arXiv 2509.24704\nkey_idea: Generative latent memory (trigger + weaver); beats ExpeL/AWM up to +38%\npaper_url: https://arxiv.org/abs/2509.24704\nneeds_verification: true",
"needs_verification": true
},
{
"id": "E13",
"name": "CTIM-Rover",
"subarea": "E",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/E13_CTIM-Rover.json",
"info": "name: CTIM-Rover\nsubarea: E — Case-based reasoning + agent memory\nyear: 2025\nvenue: arXiv 2505.23422\nkey_idea: NEGATIVE result: cross-task episodic memory hurts SWE via retrieval noise ★\npaper_url: https://arxiv.org/abs/2505.23422\nneeds_verification: true",
"needs_verification": true
},
{
"id": "F01",
"name": "DPR",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F01_DPR.json",
"info": "name: DPR\naliases: Dense Passage Retrieval\nsubarea: F — Metric / representation retrieval backbones\nyear: 2020\nvenue: EMNLP\nkey_idea: Bi-encoder dense retrieval trained on QA pairs ★\npaper_url: https://arxiv.org/abs/2004.04906",
"needs_verification": false
},
{
"id": "F02",
"name": "Sentence-BERT",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F02_Sentence-BERT.json",
"info": "name: Sentence-BERT\nsubarea: F — Metric / representation retrieval backbones\nyear: 2019\nvenue: EMNLP\nkey_idea: Siamese BERT for efficient sentence embeddings / cosine similarity ★\npaper_url: https://arxiv.org/abs/1908.10084",
"needs_verification": false
},
{
"id": "F03",
"name": "Contrastive / triplet metric learning",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F03_Contrastive_triplet_metric_learning.json",
"info": "name: Contrastive / triplet metric learning\nsubarea: F — Metric / representation retrieval backbones\nyear: 2015\nvenue: (foundational)\nkey_idea: Learn embedding space where similar items are close (triplet/InfoNCE)\npaper_url: https://arxiv.org/abs/1503.03832",
"needs_verification": false
},
{
"id": "F04",
"name": "ColBERT",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F04_ColBERT.json",
"info": "name: ColBERT\nsubarea: F — Metric / representation retrieval backbones\nyear: 2020\nvenue: SIGIR\nkey_idea: Late-interaction token-level MaxSim retrieval\npaper_url: https://arxiv.org/abs/2004.12832",
"needs_verification": false
},
{
"id": "F05",
"name": "Contriever",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F05_Contriever.json",
"info": "name: Contriever\nsubarea: F — Metric / representation retrieval backbones\nyear: 2022\nvenue: TMLR\nkey_idea: Unsupervised contrastive dense retriever; default zero-shot backbone\npaper_url: https://arxiv.org/abs/2112.09118",
"needs_verification": false
},
{
"id": "F06",
"name": "E5",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F06_E5.json",
"info": "name: E5\nsubarea: F — Metric / representation retrieval backbones\nyear: 2022\nvenue: arXiv 2212.03533\nkey_idea: Weakly-supervised contrastive (CCPairs); first to beat BM25 zero-shot unlabeled\npaper_url: https://arxiv.org/abs/2212.03533",
"needs_verification": false
},
{
"id": "F07",
"name": "GTE",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F07_GTE.json",
"info": "name: GTE\nsubarea: F — Metric / representation retrieval backbones\nyear: 2023\nvenue: arXiv 2308.03281\nkey_idea: Multi-stage contrastive general text embeddings; strong code search\npaper_url: https://arxiv.org/abs/2308.03281",
"needs_verification": false
},
{
"id": "F08",
"name": "BGE / C-Pack",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F08_BGE_C-Pack.json",
"info": "name: BGE / C-Pack\nsubarea: F — Metric / representation retrieval backbones\nyear: 2023\nvenue: SIGIR 2024\nkey_idea: Three-stage recipe; widely-deployed open retrieval backbone\npaper_url: https://arxiv.org/abs/2309.07597",
"needs_verification": false
},
{
"id": "F09",
"name": "E5-mistral",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F09_E5-mistral.json",
"info": "name: E5-mistral\naliases: Improving Text Embeddings with LLMs\nsubarea: F — Metric / representation retrieval backbones\nyear: 2024\nvenue: ICLR\nkey_idea: Decoder-LLM embedder trained on LLM-synthesized data; shift to LLM embedders\npaper_url: https://arxiv.org/abs/2401.00368",
"needs_verification": false
},
{
"id": "F10",
"name": "NV-Embed",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F10_NV-Embed.json",
"info": "name: NV-Embed\nsubarea: F — Metric / representation retrieval backbones\nyear: 2025\nvenue: ICLR\nkey_idea: LLM generalist embedder: latent-attention pooling, 2-stage instruction-tuned; MTEB #1 ★\npaper_url: https://arxiv.org/abs/2405.17428",
"needs_verification": false
},
{
"id": "F11",
"name": "INSTRUCTOR",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F11_INSTRUCTOR.json",
"info": "name: INSTRUCTOR\naliases: One Embedder, Any Task\nsubarea: F — Metric / representation retrieval backbones\nyear: 2023\nvenue: ACL Findings\nkey_idea: Instruction-conditioned embeddings → task-aware similarity (A↔F ★)\npaper_url: https://arxiv.org/abs/2212.09741",
"needs_verification": false
},
{
"id": "F12",
"name": "ColBERTv2",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F12_ColBERTv2.json",
"info": "name: ColBERTv2\nsubarea: F — Metric / representation retrieval backbones\nyear: 2022\nvenue: NAACL\nkey_idea: Denoised supervision + residual compression late interaction\npaper_url: https://arxiv.org/abs/2112.01488",
"needs_verification": false
},
{
"id": "F13",
"name": "SPLADE / SPLADEv2",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F13_SPLADE_SPLADEv2.json",
"info": "name: SPLADE / SPLADEv2\nsubarea: F — Metric / representation retrieval backbones\nyear: 2021\nvenue: SIGIR/arXiv\nkey_idea: Learned SPARSE lexical-expansion retrieval; strong OOD robustness\npaper_url: https://arxiv.org/abs/2109.10086",
"needs_verification": false
},
{
"id": "F14",
"name": "Matryoshka Representation Learning",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F14_Matryoshka_Representation_Learning.json",
"info": "name: Matryoshka Representation Learning\nsubarea: F — Metric / representation retrieval backbones\nyear: 2022\nvenue: NeurIPS\nkey_idea: Nested embeddings; elastic inference-time dimension for cost/latency-tunable index\npaper_url: https://arxiv.org/abs/2205.13147",
"needs_verification": false
},
{
"id": "F15",
"name": "repLLaMA / rankLLaMA",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F15_repLLaMA_rankLLaMA.json",
"info": "name: repLLaMA / rankLLaMA\nsubarea: F — Metric / representation retrieval backbones\nyear: 2024\nvenue: SIGIR\nkey_idea: LLaMA-2 fine-tuned as dense retriever + pointwise reranker; strong BEIR (F↔B)\npaper_url: https://arxiv.org/abs/2310.08319",
"needs_verification": false
},
{
"id": "F16",
"name": "HyDE",
"subarea": "F",
"output_path": "/Users/htizhang/Documents/GitHub/mediator-coevo/related-literature/graph-similarity-task-retrieval/results/F16_HyDE.json",
"info": "name: HyDE\naliases: Hypothetical Document Embeddings\nsubarea: F — Metric / representation retrieval backbones\nyear: 2023\nvenue: ACL\nkey_idea: LLM generates hypothetical answer, embed it to retrieve neighbors (B↔F)\npaper_url: https://arxiv.org/abs/2212.10496",
"needs_verification": false
}
]
const WORKERS = 5

function buildPrompt(it) {
  return `## 任务
调研 ${it.info}，输出结构化JSON到 ${it.output_path}

## 字段定义
读取 ${FIELDS} 获取所有字段定义

## 输出要求
1. 按fields.yaml定义的字段输出JSON
2. 不确定的字段值标注[不确定]
3. JSON末尾添加uncertain数组，列出所有不确定的字段名
4. 所有字段值必须使用中文输出（调研过程可用英文，但最终JSON值为中文）

## 输出路径
${it.output_path}

## 验证
完成JSON输出后，运行验证脚本确保字段完整覆盖：
python ~/.claude/skills/research/validate_json.py -f ${FIELDS} -j ${it.output_path}
验证通过后才算完成任务。

## 附加要求（orchestrator 追加，不替代/不改写上面的模板）
- 调研话题（topic）: ${TOPIC}
- 模型/推理：Opus 4.8，最大推理强度，先想清楚再写。
- 必须广泛使用外部检索工具。这些工具是 deferred，先用 ToolSearch 加载 schema 再调用：
  - ToolSearch "select:mcp__firecrawl__firecrawl_search,mcp__firecrawl__firecrawl_scrape"
  - ToolSearch "select:mcp__exa__web_search_exa,mcp__exa__web_fetch_exa"
  - ToolSearch "select:mcp__academic-search__search_papers,mcp__academic-search__explore_citations,mcp__academic-search__search_by_author"
  至少进行 6–10 次跨 exa/firecrawl/academic-search 的检索；用一手来源（arXiv / ACL Anthology / OpenReview / NeurIPS-ICML-ICLR proceedings / VLDB）核实 authors、year、venue，并尽量补 code_url、datasets_benchmarks、key_results。
- project_relevance 三个字段（relevance_to_task_retrieval / adaptable_components / limitations）必须结合 mediator-coevo / OPD 项目来写：一个“中介(mediator)”检索相似的先验任务/技能/案例来引导多 agent 的协同进化(coevolution)。把本方法能否、如何被移植到“按相似度检索任务/技能/案例”里讲清楚。
- cross_cutting_dimensions 里只填与本 item 子领域(${it.subarea})相关的字段；不相关的填 "[不适用]" 或留空并不要列入 uncertain。
- 若本 item 标注 needs_verification 且你无法确认论文是否真实存在 / 作者 / venue：采用 best-effort 策略，用现有部分证据尽量填写，无法确认的字段值填 "[不确定]" 并列入 uncertain 数组；绝对不要编造引用或作者。
- uncertain 数组：列出所有标了 [不确定] 的字段名（JSON 顶层键，name 即可）。
- 写文件用 Write 工具，写到精确路径：${it.output_path}
- 写完务必运行验证脚本（若 python 不可用就用 python3 运行同一脚本），确保退出码为 0、无缺失必填字段。
- 最终只回复一行文本："done ${it.id}" 或 "error ${it.id} <一句话原因>"。这行就是你的返回值。`
}

let cursor = 0
const results = new Array(ITEMS.length)

async function runOne(i) {
  const it = ITEMS[i]
  try {
    const out = await agent(buildPrompt(it), {
      label: `${it.id} ${it.name}`,
      phase: `Cluster ${it.subarea}`,
      model: 'opus',
    })
    if (out === null) return { id: it.id, name: it.name, subarea: it.subarea, status: 'failed' }
    return { id: it.id, name: it.name, subarea: it.subarea, status: 'done', reply: String(out).slice(0, 200) }
  } catch (e) {
    return { id: it.id, name: it.name, subarea: it.subarea, status: 'error', error: String(e).slice(0, 200) }
  }
}

async function worker() {
  while (true) {
    const i = cursor++
    if (i >= ITEMS.length) return
    let r = await runOne(i)
    if (r.status !== 'done') {
      const retry = await runOne(i)
      if (retry.status === 'done') r = retry
      else r = { ...r, retried: true }
    }
    results[i] = r
    const done = results.filter(Boolean).length
    log(`[${done}/${ITEMS.length}] ${r.id} ${r.name} -> ${r.status}`)
  }
}

log(`Deep research: ${ITEMS.length} items, ${WORKERS} concurrent, model=opus, results -> ${RESULTS_DIR}`)
await Promise.all(Array.from({ length: WORKERS }, () => worker()))

const all = results.filter(Boolean)
const byStatus = {}
for (const r of all) byStatus[r.status] = (byStatus[r.status] || 0) + 1
const problems = all.filter(r => r.status !== 'done').map(r => ({ id: r.id, name: r.name, status: r.status, error: r.error || null }))
return {
  total: ITEMS.length,
  completed: all.length,
  byStatus,
  problems,
}
