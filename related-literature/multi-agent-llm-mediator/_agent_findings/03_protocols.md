I now have comprehensive, verified coverage across all goals. I have enough primary-source evidence to synthesize the final report covering standardized protocols (MCP, A2A, ACP, ANP, AGNTCY/ADS, NANDA, AITP, Coral, LMOS), emergent/learned communication (Foerster RIAL/DIAL, Lazaridou referential games, Agora, EcoLANG, invented covert languages, latent/activation communication), the mediator/broker dimension, and proposed new fields.

---

# 调研报告: Agent Communication Protocols (标准化互操作 + 涌现/学习通信)

调研日期 2026-06-07。覆盖 13 个新增 items + 9 个推荐补充字段。所有引用经 web 检索/一手源核实。

## 补充Items

### A. 标准化互操作协议 (standardized-protocol)

- **MCP — Model Context Protocol** (2024-11, Anthropic; → Linux Foundation/AI Alliance Dec 2025): JSON-RPC 2.0 client-host-server 协议，标准化单 agent 对外部工具/数据/上下文的 *纵向* 访问 (resources/prompts/tools + sampling/roots/elicitation 客户端能力)。**Host 是显式 mediator**: 管理多个 client 实例、强制安全策略/用户同意、聚合上下文，并 *有意限制 server 对 prompt 的可见性* (sampling 时 server 看不到完整 prompt) — 这是一个明确的信息过滤/门控机制。https://modelcontextprotocol.io/specification/2025-11-25 ; https://www.anthropic.com/news/model-context-protocol

- **A2A — Agent2Agent** (2025-04, Google; → Linux Foundation Jun 2025, v1.0, 150+ orgs): *横向* peer-to-peer 任务委派。核心机制 = **Agent Card** (`/.well-known/agent-card.json` 能力描述符, 支持 discovery + capability negotiation)，Task 有状态生命周期 (SUBMITTED→WORKING→COMPLETED/FAILED/CANCELED/REJECTED/INPUT_REQUIRED/AUTH_REQUIRED)。支持 JSON-RPC 2.0/gRPC/REST、streaming (SSE)、push notifications (webhooks)。已集成进 Azure AI Foundry、Amazon Bedrock AgentCore、Google Agent Engine。https://a2a-protocol.org/latest/specification/ ; https://github.com/a2aproject/A2A

- **ACP — Agent Communication Protocol** (2024–2025, IBM Research/BeeAI; → Linux Foundation; roadmap **merged into A2A ~Sept 2025**): REST-native, MIME multipart messages, async-by-default + SSE streaming, SDK-optional (可用 curl)。**显式 broker 架构**: ACP Server = "protocol broker"，维护 Agent Registry (Agent Detail Schema 元数据目录)，执行 auth/authz/rate-limiting，做 *agent lookup + routing*，按序传回 response parts。这是本调研中 *最清晰的 broker/registry mediation* 范例。https://research.ibm.com/projects/agent-communication-protocol ; https://github.com/i-am-bee/acp

- **ANP — Agent Network Protocol** (2024, Gaowei Chang, MIT license; W3C community): 面向 *开放互联网* 的去中心化 agent mesh。基于 W3C DIDs (decentralized identifiers) + JSON-LD 语义图，`.well-known` 发现，最具雄心的 trustless identity 设计 (任意 agent 无需预建信任即可发现/通信)。https://arxiv.org/html/2505.02279

- **AGNTCY / Agent Directory Service (ADS)** (2025, Cisco Outshift + LangChain + Galileo; "Internet of Agents" 基础设施): 分布式 capability-aware 目录。**Mediator = 分布式目录服务**: agents 通过 OASF (Open Agent Schema Framework) 结构化元数据 announce 能力，consumer 通过 capability-based 查询发现。基于 Kademlia DHT 内容路由 (skill→record ID 映射)、OCI registry as storage、Sigstore 加密 provenance。原生支持 A2A/MCP/自定义协议作为可发现记录。https://docs.agntcy.org/dir/overview/ ; https://arxiv.org/pdf/2509.18787

- **NANDA Index / AgentFacts** (2025-07, MIT Media Lab, Ramesh Raskar): "Beyond DNS" — agent 的 DNS 类比层。**Mediator = 精简 index + 可验证 AgentFacts**: 解析流 `Agent Name → NANDA Index → AgentAddr → AgentFacts → Endpoint`，支持多端点路由/负载均衡、亚秒级撤销与 key rotation、schema 验证的能力断言，以及 **隐私保护的最小披露查询 (least-disclosure)** — 一个明确的信息过滤机制。"quilt" 索引可桥接 MCP/A2A/NLWeb registries。https://arxiv.org/pdf/2507.14263

- **AITP — Agent Interaction & Transaction Protocol** (2024-12/2025-03, NEAR AI): 专注 *跨信任边界* 的 user↔agent / agent↔agent 通信。组成: Chat Threads (兼容 OpenAI Assistants/Threads API) + 可扩展 Capabilities (multimodal、generative UI、payments、human-in-the-loop attestations)。**支付请求 (Quote) 沿 assistant 链 *上游流动*，每个中间 agent 可处理/修改/拒绝** — 一种链式 mediation/gating。引入 Discovery Agent 角色。https://aitp.dev/ ; https://github.com/nearai/aitp

- **Coral Protocol** (2025-05): 开放基础设施连接 Internet of Agents，内建经济交易 (区块链不可篡改支付审计 trail、HTTP 402/x402 流、escrow payment sessions、Solana 结算)。agent 完成付费任务时由 Secure Payments service 签名广播链上交易。https://arxiv.org/pdf/2505.00749 ; https://docs.coralos.ai/concepts/payments

- **Eclipse LMOS Protocol** (Eclipse Foundation): transport-agnostic 多 agent 协议，基于 W3C Web of Things (WoT) Thing Description 抽象。**关键: agents 动态协商并选择最佳 transport** (HTTP/MQTT/WebSocket/AMQP) 和 media type (JSON/CBOR)，通过协议绑定层翻译 — 接近 "可适配 mediation"。https://eclipse.dev/lmos/docs/lmos_protocol/introduction/

### B. 涌现 / 学习通信 (emergent-language / learned-comm)

- **RIAL / DIAL — Learning to Communicate** (Foerster et al., NeurIPS 2016): 奠基性工作。多 agent 通过 deep RL *端到端学习* 离散通信协议。RIAL=deep Q-learning (agent 内端到端); DIAL=可微分跨 agent (集中训练时梯度穿过通信信道, 分散执行时离散化)。CTDE (centralized training, decentralized execution) 范式起点。https://arxiv.org/abs/1605.06676

- **Lazaridou referential games** (ICLR 2017 *Multi-Agent Cooperation & Emergence of (Natural) Language*; ICLR 2018 *Referential Games w/ Symbolic & Pixel Input*): sender/receiver 在指代游戏中自发发展语言;研究 compositionality 与输入结构的关系 (结构化输入 → 结构化语言)。Agora 论文明确引用这一脉络作为 LLM 网络涌现协议的理论基础。https://arxiv.org/abs/1612.07182 ; https://ar5iv.labs.arxiv.org/html/1804.03984

- **Agora (seed #13, 已确认)** (Marro, La Malfa, Wright, Li, Shadbolt, Wooldridge, Torr; Oxford/CAMEL-AI; arXiv 2410.11905, ICLR 2025 submission — 注: OpenReview 显示 *Withdrawn*): meta-protocol 化解 "Agent Communication Trilemma" (versatile/efficient/portable)。机制: 频繁通信用 routines (结构化), 罕见用自然语言, 中间用 LLM 写的 routines。**Protocol Documents (PD)** 由 SHA1 hash 唯一标识 (借鉴 IPFS, 无中央机构)。100-agent demo *涌现去中心化协议共识*，1000 queries 成本 $36.23→$7.67 (~5× 降低)。这是连接标准化与涌现两个范式的关键桥梁 item。https://arxiv.org/html/2410.11905 ; https://agoraprotocol.org/docs/protocol/specification

- **"Beyond Natural Language: Invented Communication in VLMs"** (anonymous, ICLR 2026 under review, OpenReview 2CxqnHClyN): 用指代游戏测 VLM 是否能发明协议。4 关键发现: (1) 紧约束下自发造新词; (2) prompt 鼓励压缩时发明的语言 *效率超过自然语言*; (3) 可发明 **covert (隐蔽) 协议** — 对外部观察者/人类不可读 (透明性/可控性风险); (4) 相同架构模型无需共享协议即可自发协调。直接相关于 mediator 的 *安全/可解释性* 维度。https://openreview.net/pdf?id=2CxqnHClyN

- **EcoLANG** (arXiv 2505.06904, 2025): 为大规模社会模拟诱导高效 agent 通信语言。两阶段: (1) language evolution — 基于 Zipf 最省力原则压缩词表 (按频率/长度过滤同义词) + natural-selection 范式迭代优化规则; (2) utilization — 修改推理模型词表强制使用。framework-agnostic。https://arxiv.org/html/2505.06904

- **"Emergence of Machine Language in LLM-based Agent Communication"** (Zou, Ren, Chen, Wang, Hu; ICLR 2026 submission, withdrawn): 问 LLM agents 间能否涌现 *可能非人类可读* 的机器语言。541 objects 上仅 4 轮通信即建立共享语言，呈现 compositionality/generalizability/morphemes/polysemy。https://openreview.net/forum?id=zy06mHNoO2

补充值得记录的相邻发现 (latent/representation-based communication — 可作为新兴子类):
- **LatentMAS** (arXiv 2511.20639): agents 完全在连续 latent space 协作 (last-layer hidden representations + 跨 agent KV-cache latent working memory transfer), training-free, 无损通信保真。
- **Q-KVComm** (arXiv 2512.17914): 直接传输压缩 KV-cache 表示 (自适应 layer-wise 量化), 把 inter-agent 通信从 text-based 转为 representation-based。
- **"Communicating Activations Between LM Agents"** (arXiv 2501.14082): agents 通过 activation 注入通信, 高熵、省 compute。
这些代表 "neuralese"/latent 通信方向, 是 emergent-comm 的训练时延伸。

## 推荐补充字段 (针对 protocols 子主题，扩展现有 field framework)

- **interoperability_layer**: 协议在 agent 通信栈中的层级 (tool-access/纵向 vs agent-coordination/横向 vs internet-discovery/全局)。MCP/A2A/ANP 的核心区分维度 — 多数 2025-26 surveys 共识是 *互补分层而非竞争*。
- **mediation_architecture**: broker vs registry vs P2P/decentralized vs host-mediated。直接对应 KEY 字段;取值如 ACP=brokered server、A2A=peer-like、ANP=P2P DID、MCP=host-mediated、AGNTCY/NANDA=distributed directory。
- **discovery_mechanism**: 如何发现对端 (manual/static、Agent Card `.well-known`、registry API、DID+DHT、capability-based semantic query)。是 mediation 的具体落地。
- **information_filtering_gating**: KEY — mediator 过滤/门控什么。例: MCP host 限制 server 对 prompt 的可见性;NANDA least-disclosure 查询;AITP 支付请求链式审批;ACP server 的 capability 合规检查。
- **identity_trust_model**: 身份与信任机制 (none/API-key、OAuth2/OIDC/mTLS、W3C DIDs、verifiable credentials、Sigstore content-addressing、ZK proofs)。2025-26 survey 高频对比维度。
- **security_threat_surface**: 安全风险面 (prompt injection across chains、supply-chain/registry poisoning、auth bypass、covert channels)。多篇专门 security survey (arXiv 2506.19676, 2602.11327) 用 STRIDE/CIA 建模。
- **adaptivity (fixed/negotiated/learned/evolved)**: 协议固定 vs 运行时协商 (Agora PD negotiation、LMOS transport negotiation) vs RL 学习 (RIAL/DIAL) vs natural-selection 演化 (EcoLANG)。这正是 emergent vs standardized 的连续谱。
- **efficiency_evidence**: 量化通信成本/token 节省 (Agora ~5×/98% token 降低;EcoLANG 压缩;Q-KVComm/LatentMAS latent 压缩)。是 mediator-coevo "信息过滤降本" 论点的直接证据。
- **human_interpretability / covertness**: 协议消息对人类是否可读 — standardized 协议天然可读;emergent 语言可能 covert/不可读 (透明性风险)。对 mediator 设计的 governance 含义重大。

## 与 mediator-coevo 的相关性提示
- **最相关的 mediation 范例**: IBM ACP (broker server + registry routing)、MCP host (信息门控/限制 server 可见性)、NANDA (least-disclosure 过滤查询)、AITP (链式支付审批)。这些都展示了 "中介过滤/门控信息" 的不同形态。
- **协议 co-evolution / emergence 的直接证据**: Agora (去中心化协议共识涌现 + PD 协商演化)、EcoLANG (natural-selection 演化通信规则)、RIAL/DIAL & Lazaridou (学习/演化通信语言)、invented covert languages (效率 vs 可控性张力)。
- **关键张力**: efficiency (emergent/compressed/latent 通信省成本) vs interpretability/control (covert 语言带来的透明性丧失) — 这恰是 mediator 作为信息过滤器需权衡的核心。

## 信息来源 (精选, 全部已核实)
- [Agora — A Scalable Communication Protocol for Networks of LLMs (Marro et al.)](https://arxiv.org/html/2410.11905) · [spec](https://agoraprotocol.org/docs/protocol/specification)
- [Internet of Agents (IoA), OpenBMB, ICLR 2025](https://arxiv.org/abs/2407.07061)
- [A Survey of AI Agent Protocols (Yang et al., 2025) — context-oriented vs inter-agent 二维分类](https://arxiv.org/pdf/2504.16736)
- [A Survey of Agent Interoperability Protocols: MCP/ACP/A2A/ANP (Ehtesham, Singh, Gupta, Kumar 2025)](https://arxiv.org/html/2505.02279)
- [A Survey of LLM-Driven AI Agent Communication: Protocols, Security Risks, Defenses (2506.19676)](https://arxiv.org/html/2506.19676v4)
- [Security Threat Modeling for MCP/A2A/Agora/ANP (2602.11327)](https://arxiv.org/pdf/2602.11327v1)
- [Beyond Message Passing: A Semantic View of Agent Communication Protocols (2604.02369)](https://arxiv.org/html/2604.02369)
- [Survey of LLM Agent Communication with MCP: Design-Pattern Centric (broker/mediator pattern) (2506.05364)](https://arxiv.org/pdf/2506.05364)
- [MCP spec](https://modelcontextprotocol.io/specification/2025-11-25) · [Anthropic announcement](https://www.anthropic.com/news/model-context-protocol)
- [A2A spec (Linux Foundation)](https://a2a-protocol.org/latest/specification/) · [A2A repo](https://github.com/a2aproject/A2A)
- [IBM ACP (Research)](https://research.ibm.com/projects/agent-communication-protocol) · [ACP openapi](https://github.com/i-am-bee/acp)
- [AGNTCY Agent Directory Service spec/paper](https://arxiv.org/pdf/2509.18787) · [docs](https://docs.agntcy.org/dir/overview/)
- [NANDA Index / AgentFacts (MIT)](https://arxiv.org/pdf/2507.14263)
- [AITP (NEAR AI)](https://aitp.dev/) · [Coral Protocol](https://arxiv.org/pdf/2505.00749) · [Eclipse LMOS](https://eclipse.dev/lmos/docs/lmos_protocol/introduction/)
- [Foerster RIAL/DIAL (NeurIPS 2016)](https://arxiv.org/abs/1605.06676)
- [Lazaridou ICLR 2017](https://arxiv.org/abs/1612.07182) · [Lazaridou ICLR 2018 pixel/symbolic](https://ar5iv.labs.arxiv.org/html/1804.03984)
- [Beyond Natural Language: Invented Communication in VLMs (efficiency + covertness)](https://openreview.net/pdf?id=2CxqnHClyN)
- [EcoLANG (2505.06904)](https://arxiv.org/html/2505.06904) · [Emergent language: a survey & taxonomy (Springer 2025)](https://link.springer.com/article/10.1007/s10458-025-09691-y)
- [LatentMAS (2511.20639)](https://arxiv.org/pdf/2511.20639) · [Q-KVComm (2512.17914)](https://arxiv.org/pdf/2512.17914) · [Communicating Activations (2501.14082)](https://arxiv.org/pdf/2501.14082)
- 历史背景: [FIPA-ACL / KQML standardization](https://link.springer.com/chapter/10.1007/3-540-47745-4_4) — speech-act performatives, agent 通信语言的前 LLM 时代根源 (advertise/recruit 等 mediation primitives)。

**核心发现摘要**: 本领域 2025-26 已清晰分为两个范式: (a) standardized interoperability — 业界共识为 *互补分层* (MCP=纵向工具、A2A=横向 agent 委派、ANP/AGNTCY/NANDA=互联网级发现);ACP 已并入 A2A。Mediation 形态多样: host-mediated (MCP)、brokered registry (ACP/BeeAI)、distributed directory+DHT (AGNTCY/NANDA)、P2P DID (ANP)。(b) emergent/learned communication — 从 Foerster/Lazaridou 的 RL 学习离散语言，到 Agora 的 LLM 协议协商涌现，再到 2025-26 的 EcoLANG (演化压缩)、invented covert languages、latent/neuralese 通信。**对 mediator-coevo 最具价值的是 efficiency↔interpretability 张力，及 Agora/EcoLANG 展示的协议本身可协商/演化 (co-evolve) 的能力。** 注意 Agora 与 "Emergence of Machine Language" 两篇 ICLR 投稿在 OpenReview 显示 Withdrawn 状态 (arXiv 版本仍有效)，引用时宜以 arXiv 为准。
