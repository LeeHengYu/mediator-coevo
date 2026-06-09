I now have exhaustive coverage. I found AgentDropout as a complement to AgentPrune ("who communicates" vs "what is communicated"). I have all the data needed to synthesize the report. Let me compile the findings.

---

# 调研报告: Multi-Agent LLM 通信框架 / 编排系统

## 种子项验证 (8–16)

All 7 seed items verified against primary sources (arXiv + venue + repo). Corrections/notes:
- **CAMEL** — arXiv **2303.17760** (Mar 2023), Li, Hammoud, Itani, Khizbullin, Ghanem (KAUST). NeurIPS 2023. ~1,460 cites (SS). Mechanism: role-play + **Inception Prompting** (task-specifier/assistant/user prompts), optional **critic-in-the-loop** (an explicit mediator). github.com/camel-ai/camel
- **AutoGen** — arXiv **2308.08155** (Wu et al., Microsoft), COLM 2024. Conversable agents + `GroupChatManager` (LLM/round-robin speaker selection). Mediation = group-chat manager selects next speaker. v0.4 = actor/event-driven runtime.
- **MetaGPT** — arXiv **2308.00352** (Hong, Zhuge et al.), ICLR 2024 Oral. SOP-encoded roles + **shared message pool with publish-subscribe** (agents subscribe to role-relevant messages — a structural information filter to cut irrelevant context).
- **ChatDev** — arXiv **2307.07924** (Qian et al., OpenBMB/Tsinghua), ACL 2024. **Chat-chain** (what to communicate) + **communicative dehallucination** (how); shares only *subtask solutions* not full history (filtering by design).
- **AgentVerse** — arXiv **2308.10848** (Chen et al., Tsinghua), ICLR 2024. 4-stage loop with **dynamic expert recruitment** + collaborative decision-making; studies emergent volunteer/conformity/destructive behaviors.
- **Mixture-of-Agents (MoA)** — arXiv **2406.04692** (Wang et al., Together AI), ICLR 2025. **Layered proposer→aggregator**; aggregator is an explicit synthesizer/mediator. 65.1% AlpacaEval 2.0 > GPT-4o. github.com/togethercomputer/moa
- **GPTSwarm** — arXiv **2402.16823** (Zhuge et al., KAUST/IDSIA), ICML 2024 Oral. Agents as **optimizable computational graphs**; node-optimization (prompts) + **edge-optimization** (continuous relaxation of DAG connectivity = learned information flow).

(Note: the task's seed numbering 8–16 corresponds to this slice; all confirmed.)

## 补充Items (推荐新增)

Organized by sub-cluster; the **mediation-relevant** ones are marked ⭐ (highest priority for the mediator-coevo theme).

**Cluster: Foundational orchestration frameworks (industry/OSS)**
- **LangGraph + langgraph-supervisor/langgraph-swarm** (2023–25, LangChain): graph/state-based orchestration; supervisor routes via `Command(goto=)`; ⭐`create_forward_message_tool` (forward worker output verbatim, skip supervisor paraphrase) and handoff tools pass **full vs. filtered** message history; subgraph state isolation. github.com/langchain-ai/langgraph
- **OpenAI Swarm** (2024, OpenAI): minimal stateless `Agent`+`handoff` primitives; `context_variables`. github.com/openai/swarm
- ⭐**OpenAI Agents SDK** (2025, OpenAI): production successor to Swarm; **`input_filter`/`HandoffInputData`** lets a handoff rewrite/filter the transcript passed to the next agent; **`nest_handoff_history`** collapses prior transcript into a single summary message — a built-in inter-agent mediator. openai.github.io/openai-agents-python
- **CrewAI** (2024, OSS): role/goal/backstory agents; **hierarchical process** with a `manager_llm`/`manager_agent` that delegates+validates; `allow_delegation` collaboration tools. docs.crewai.com
- **AgentScope** (arXiv **2402.14034**, Gao et al., Alibaba 2024): **message-exchange as core**; `MsgHub` for multi-agent broadcast; actor-based distribution + fault tolerance; v2 adds MCP/A2A. github.com/agentscope-ai/agentscope

**Cluster: Orchestrator/manager with explicit mediation (⭐ most relevant)**
- ⭐**Magentic-One** (arXiv **2411.04468**, Fossati/Fourney et al., Microsoft 2024): lead **Orchestrator** maintains **Task Ledger** (facts/plan) + **Progress Ledger** (self-reflection, stall detection); outer/inner loop directs FileSurfer/WebSurfer/Coder. Ledger = mediating shared state. `MagenticOneOrchestrator(BaseGroupChatManager)`.
- ⭐**Anthropic Multi-Agent Research System** (engineering blog, 2025-06): orchestrator–worker; subagents act as "**intelligent filters**", run in **isolated context windows**, return **condensed summaries** to the lead; separate CitationAgent pass; plan persisted to external memory. anthropic.com/engineering/multi-agent-research-system
- ⭐**Cognition "Don't Build Multi-Agents"** (2025-06): context-engineering principles; advocates **summarization at agent-agent boundaries via a fine-tuned compression model** — a literal mediator. cognition.ai/blog/dont-build-multi-agents
- ⭐**Anthropic "Effective context engineering"** (2025-09): formalizes **compaction** (summarize-then-reinitialize), structured note-taking, sub-agent isolation as the three context-management levers. anthropic.com/engineering/effective-context-engineering-for-ai-agents
- **OWL / Workforce** (NeurIPS 2025, CAMEL-AI/HKU): hierarchical **Planner + Coordinator + Workers** with **context isolation** (workers keep isolated histories); RL-trained domain-agnostic planner; GAIA 69.70%. github.com/camel-ai/owl
- **TalkHier** (arXiv **2502.11098**, Wang et al., Sony 2025): ⭐**structured communication protocol** (typed, context-rich messages) + **hierarchical refinement** where a supervisor produces *summarized & coordinated* feedback to mitigate order-bias and feedback-overload. github.com/sony/talkhier

**Cluster: Communication-efficiency / pruning / compression (⭐ direct mediator analogues)**
- ⭐**Optima** (arXiv **2410.08115**, Chen et al., THUNLP), ACL 2025 Findings: trains MAS for communication efficiency+effectiveness (iSFT/iDPO); up to **2.8× accuracy with <10% tokens**. github.com/thunlp/Optima
- ⭐**AgentPrune / "Cut the Crap"** (arXiv **2410.02506**, Zhang et al.), ICLR 2025: defines **Communication Redundancy**; one-shot prunes the spatial-temporal message graph; 28–73% token cut, also filters malicious messages. github.com/yanweiyue/AgentPrune
- ⭐**AgentDropout** (Wang et al., 2025): complement to AgentPrune — prunes **agent nodes** ("who communicates") via per-round degree scores, vs AgentPrune's edges ("what is communicated").
- ⭐**EIB-Learner** (arXiv **2505.23352**, 2025): causal analysis — sparse topologies suppress error propagation but also kill beneficial insights; learns **moderate-sparsity topology balancing error-suppression vs insight-propagation** (core mediation tradeoff).
- ⭐**SafeSieve** (arXiv **2508.11733**, 2025): progressive pruning, LLM semantic-compatibility → experience-based edge scoring + 0-extension clustering.
- ⭐**Q-KVComm** (arXiv **2512.17914**, 2025): transmits **compressed KV-cache + extracted-fact summaries** between agents instead of raw text — representation-level inter-agent mediation.

**Cluster: Learned/automated topology & workflow design**
- **DyLAN** (arXiv **2310.02170**, Liu, Zhang, Li, Liu, Yang, Diyi Yang 2023): dynamic agent network; **Agent Importance Score** (forward-backward message passing) selects team; inference-time agent deactivation + early-stopping.
- **G-Designer** (arXiv **2410.11782**, Zhang et al.), ICML 2025 Spotlight: **VGAE** designs task-adaptive communication topology per query; up to 95.3% token cut on HumanEval; +adversarial robustness. github.com/yanweiyue/GDesigner
- **MaAS (Agentic Supernet)** (arXiv **2502.04180**, Zhang et al.), ICML 2025 Oral: samples **query-dependent** sub-architectures from a probabilistic supernet; 6–45% of inference cost of static MAS. github.com/bingreeky/MaAS
- **AFlow** (arXiv **2410.10762**, Zhang et al., MetaGPT/FoundationAgents), ICLR 2025: **MCTS over code-represented workflows** (nodes=LLM calls, operators=Generate/Review/Revise/Ensemble).
- **AgentSquare** (arXiv **2410.06153**), ICLR 2025: modular agent search (Planning/Reasoning/ToolUse/Memory) via module evolution+recombination.
- **ScoreFlow** (arXiv **2502.04306**, 2025): gradient-based workflow optimization via **Score-DPO**.

**Cluster: Consensus / debate paradigms**
- **Multiagent Debate (Society of Minds)** (arXiv **2305.14325**, Du et al., MIT/Google), ICML 2024: agents read+critique all peers' answers over rounds → consensus; improves factuality/reasoning. (Mediation = full-broadcast concatenation, the *unfiltered* baseline.)
- **ChatEval** (arXiv **2308.07201**, Chan et al.), ICLR 2024: multi-agent **referee team** with debate communication strategies (one-by-one / simultaneous / summarizer).

**Cluster: Interoperability protocols (cross-framework messaging standards)**
- **MCP (Model Context Protocol)** (Anthropic, 2024-11): host–client–server, JSON-RPC; standardizes agent↔tool/data (and resources/prompts/sampling). Context **aggregation across clients** by the host.
- **A2A (Agent2Agent)** (Google→Linux Foundation, 2025-04): agent↔agent; **Agent Cards** (capability discovery), Task lifecycle, **Messages vs Artifacts** separation (communication vs output). Absorbed IBM ACP.
- **ANP / Agora** (2024–25): decentralized agent-web protocol (ANP); **Agora** meta-protocol balancing versatility/efficiency/portability.

**Cluster: Unifying infrastructure / surveys (for the FIELD FRAMEWORK)**
- **MASLab** (arXiv **2505.16988**, Ye et al. 2025): unified codebase, 20+ MAS methods, fair benchmarking — useful provenance/reproducibility anchor.
- **Beyond Self-Talk: Communication-Centric Survey of LLM-MAS** (arXiv **2502.14321**, v3 2026): the central taxonomy lens — system-level (architecture/goals/protocols) + internal (strategies/paradigms/objects/content).
- **Multi-Agent Collaboration Mechanisms: A Survey** (arXiv **2501.06322**, 2025): actors/types/structures/strategies/protocols; centralized/decentralized/hierarchical topologies.
- **LLM-Based MA Orchestration Survey** (Preprints 202604.2147, 2026): three-topology + adaptivity axis; compares LangGraph/CrewAI/AutoGen/OpenAI-SDK on state-mgmt, token cost, failure recovery.
- **Beyond Message Passing: Semantic View of Agent Communication Protocols** (arXiv **2604.02369**, 2026): 3-layer (communication/syntactic/semantic) analysis of 18 protocols — finds semantic mediation (clarification/context-alignment/verification) is under-supported and pushed into prompts/wrappers (directly motivates a dedicated mediator).

## 推荐补充字段 (to extend the FIELD FRAMEWORK)

Your current schema covers most dimensions well. Recommended additions/refinements for this slice:

- **mediation_locus**: *Where* filtering happens — (a) none/full-broadcast, (b) orchestrator/manager agent (Magentic-One ledger, AutoGen GroupChatManager), (c) handoff-boundary filter (OpenAI SDK `input_filter`, LangGraph forward/handoff), (d) topology pruning (AgentPrune/G-Designer), (e) dedicated compression model (Cognition fine-tuned summarizer, Q-KVComm). *This is the single most decisive axis for mediator-coevo.*
- **mediation_granularity**: message-level (compress each message) vs edge-level (who-talks-to-whom) vs node-level (who participates: AgentDropout) vs context-level (compaction/summarization). Distinguishes "what" vs "who" vs "how much."
- **context_isolation**: Do workers share one context or hold isolated windows handing back summaries? (shared-pool MetaGPT vs isolated Anthropic/OWL). Critical for the filtering hypothesis.
- **information_loss_vs_fidelity**: Lossy (extractive summary) vs lossless/ground-truth-preserving vs representation-level (KV-cache). Captures the mediator's compression tradeoff.
- **error_vs_insight_propagation**: Does the framework explicitly trade off suppressing error propagation against preserving beneficial insight (EIB-Learner)? A mediator's core failure mode.
- **adaptivity_of_topology** (refine your `adaptivity`): fixed / heuristic / learned-offline / query-adaptive / co-evolved — many 2025 frameworks are query-adaptive (G-Designer, MaAS), which is where a *co-evolving* mediator would sit.
- **coordination_medium** (you already list this): suggest enumerating shared-message-pool / blackboard / ledger / state-object / handoff-token / KV-cache to make it comparable.
- **token_cost_evidence**: quantified token/cost reduction (Optima -90%, G-Designer -95.3%, AgentPrune -28–73%, MaAS 6–45% cost) — load-bearing for "is mediation worth it."
- **security_robustness**: many pruning frameworks (AgentPrune, G-Designer, EIB-Learner) double as adversarial-message defenses — relevant if the mediator also filters malicious/erroneous content.

## 信息来源 (primary)
- [LLM-MA Survey (Guo et al.)](https://arxiv.org/html/2402.01680) · [Communication-Centric Survey](https://arxiv.org/html/2502.14321v2) · [Collaboration Mechanisms Survey](https://arxiv.org/pdf/2501.06322) · [Orchestration Survey 2026](https://www.preprints.org/manuscript/202604.2147) · [Semantic View of Protocols](https://arxiv.org/html/2604.02369v2)
- [CAMEL](https://arxiv.org/abs/2303.17760) · [AutoGen](https://arxiv.org/abs/2308.08155) · [MetaGPT](https://arxiv.org/pdf/2308.00352) · [ChatDev](https://arxiv.org/abs/2307.07924) · [AgentVerse](https://openreview.net/forum?id=EHg5GDnyq1) · [MoA](https://arxiv.org/html/2406.04692v1) · [GPTSwarm](https://arxiv.org/abs/2402.16823)
- [Magentic-One](https://arxiv.org/html/2411.04468) · [AutoGen GroupChat](https://microsoft.github.io/autogen/0.4.6/user-guide/core-user-guide/design-patterns/group-chat.html) · [Anthropic multi-agent](https://www.anthropic.com/engineering/multi-agent-research-system) · [Anthropic context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) · [Cognition Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents) · [OWL](https://papers.neurips.cc/paper_files/paper/2025/file/48dcc43a534c5b582f9d0fdb778e9b84-Paper-Conference.pdf) · [TalkHier](https://arxiv.org/abs/2502.11098)
- [LangGraph supervisor](https://github.com/langchain-ai/langgraph-supervisor-py) · [LangGraph swarm](https://github.com/langchain-ai/langgraph-swarm-py) · [OpenAI Swarm](https://github.com/openai/swarm) · [OpenAI Agents SDK handoffs](https://openai.github.io/openai-agents-python/handoffs/) · [CrewAI processes](https://docs.crewai.com/en/concepts/processes) · [AgentScope](https://arxiv.org/html/2402.14034)
- [Optima](https://arxiv.org/html/2410.08115v2) · [AgentPrune](https://arxiv.org/abs/2410.02506) · [EIB-Learner](https://arxiv.org/html/2505.23352) · [SafeSieve](https://arxiv.org/html/2508.11733v2) · [Q-KVComm](https://arxiv.org/pdf/2512.17914) · [Token Economics survey (AgentDropout)](https://arxiv.org/html/2605.09104v1)
- [DyLAN](https://arxiv.org/abs/2310.02170) · [G-Designer](https://arxiv.org/abs/2410.11782) · [MaAS](https://github.com/bingreeky/MaAS) · [AFlow](https://github.com/FoundationAgents/MetaGPT/tree/main/examples/aflow) · [AgentSquare](https://arxiv.org/html/2410.06153v3) · [Multiagent Debate](https://arxiv.org/abs/2305.14325) · [ChatEval](https://arxiv.org/abs/2308.07201) · [DSPy](https://arxiv.org/abs/2310.03714)
- [MCP](https://www.anthropic.com/news/model-context-protocol) · [A2A](https://a2a-protocol.org/latest/specification/) · [MASLab](https://arxiv.org/html/2505.16988v1)

### Key synthesis for mediator-coevo
The field cleanly splits the "mediator" function into four loci that a co-evolving mediator could occupy: **(1) manager/orchestrator agents** (AutoGen GroupChatManager, Magentic-One ledger, CrewAI manager) decide *who speaks*; **(2) handoff-boundary filters** (OpenAI SDK `input_filter`/`nest_handoff_history`, LangGraph forward/handoff, Cognition's fine-tuned summarizer) decide *what context transfers*; **(3) topology learners** (G-Designer, MaAS, AgentPrune, AgentDropout, DyLAN, EIB-Learner) decide *the communication graph itself* — and notably G-Designer/MaAS are already **query-adaptive and learned**, the closest existing analog to a co-evolved mediator; **(4) representation-level compressors** (Q-KVComm, Optima-trained protocols). The recurring empirical finding (EIB-Learner, AgentPrune, Anthropic) is that filtering is a genuine tradeoff — sparser/more-summarized communication suppresses error and saves tokens but can also drop load-bearing insights and "why-not" failure records — which is precisely the optimization tension a learned mediator should target.
