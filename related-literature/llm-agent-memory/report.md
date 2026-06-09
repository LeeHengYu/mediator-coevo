# 调研报告：LLM Agents Memory — experience replay / episodic / long-term retrieval / reflection / failure memory

> 共 **58** 个调研对象，覆盖 8 个 cluster（A–H）。每条经 Opus-4.8 agent 经 academic-search / exa / firecrawl 多源核实，字段覆盖率 100%（不确定值已跳过）。

> 生成自 `results/*.json` + `fields.yaml`。


## 目录 (Table of Contents)


### A. 反思与失败驱动记忆 (Reflection & failure-driven)

1. [A1 Reflexion](#a1-reflexion言语强化学习的语言智能体论文标题reflexion-language-agents-with-verbal-reinforcement-learning亦被称作言语强化学习verbal-rl范式) — 2023 · NeurIPS 2023 | ~3776引 | 以情景记忆 | 非参数化/智能体中心
2. [A2 Retroformer](#a2-retroformer回顾式大语言智能体含可训练的回顾模型retrospective-model论文标题retroformer-retrospective-large-language-agents-with-policy-gradient-optimization) — 2023 · ICLR 2024 | ~126引 | 以情景记忆 | 非参数化/智能体中心
3. [A3 CLIN](#a3-clincontinual-learning-from-interactions持续从交互中学习的语言智能体) — 2023 | ~78引 | 以程序性/语义性记忆为主 | non-parametric/智能体中心
4. [A4 AutoGuide](#a4-autoguide自动生成与选择状态感知情境感知指南的框架论文标题-v1-用-state-aware-guidelinesneurips-2024-正式版与-arxiv-v2-改为-context-aware-guidelines二者指同一方法) — 2024 · NeurIPS 2024 | ~27引 | 以语义记忆 | 非参数化/智能体中心
5. [A5 ExpeL](#a5-expelexperiential-learning-agent经验学习智能体) — 2023 · AAAI 2024 | ~596引 | 以情景记忆 | 非参数化/智能体中心
6. [A6 ReasoningBank](#a6-reasoningbank推理记忆库配套提出-matts-记忆感知的测试时扩展) — 2025 | 以程序性记忆 | 非参数化/智能体中心
7. [A7 Memento 2](#a7-memento-2memento-ii--stateful-reflective-memory提出-stateful-reflective-decision-process-srdp-与-readwrite-reflective-learning) — 2025 · arXiv 预印本 | 情景记忆 | 非参数化/智能体中心
8. [A8 MUSE](#a8-musememory-utilizing-and-self-evolving记忆驱动的自我进化智能体框架论文题名learning-on-the-job-an-experience-driven-self-evolving-agent-for-long-horizon-tasks) — 2025 | ~15引 | 以程序性记忆 | 非参数化/智能体中心

### B. 情景记忆与检索架构 (Episodic memory & retrieval)

9. [B1 生成式智能体](#b1-生成式智能体-generative-agents论文标题generative-agents-interactive-simulacra-of-human-behavior核心组件别名记忆流-memory-streamsmallville-沙盒小镇) — 2023 · UIST 2023 | ~4322引 | 以情景记忆 | 非参数化/智能体中心
10. [B2 MemoryBank](#b2-memorybank记忆库配套提出基于其的双语-ai-陪伴聊天机器人-siliconfriend) — 2023 · AAAI 2024 | ~481引 | 以情景性记忆 | 非参数化/智能体中心
11. [B3 MemGPT](#b3-memgptmemory-gpt又名-memorygpt提出虚拟上下文管理--llm-as-os范式后产品化为开源框架-letta) — 2023 · COLM 2024 | ~767引 | 以工作记忆 | 非参数化/智能体中心
12. [B4 A-MEM](#b4-a-memagentic-memory智能体记忆系统亦写作-a-mem) — 2025 | ~603引 | 以语义性记忆 | 非参数化/智能体中心
13. [B5 Think-in-Memory](#b5-think-in-memorytim中文记忆中思考在记忆中思考完整标题think-in-memory-recalling-and-post-thinking-enable-llms-with-long-term-memory) — 2023 | ~50引 | 情景记忆与语义记忆的混合 | 非参数化/智能体中心
14. [B6 Larimar](#b6-larimar大语言模型情景记忆控制架构) — 2024 · ICML 2024 | 情景记忆 | 非参数化/智能体中心
15. [B7 MemoryOS](#b7-memoryos论文题名memory-os-of-ai-agent框架别名-memoryos--memory-operating-system由北京邮电大学-baijia-ai-团队开源仓库-bai-labmemoryos) — 2025 · EMNLP 2025 主会 | ~68引 | 以「用户中心」的长期情景/语义记忆为主 | 非参数化/智能体中心
16. [B8 EM-LLM](#b8-em-llmepisodic-memory-llm论文题为human-inspired-episodic-memory-for-infinite-context-llms注意本研究-outline-题录写作human-like与-arxiviclr-正式题目human-inspired略有出入系统简称统一为-em-llm) — 2024 · ICLR 2025 | 情景性记忆 | 非参数化/用户中心
17. [B9 MIRIX](#b9-mirixmulti-agent-memory-system-for-llm-based-agents模块化多智能体记忆系统含六类记忆--八个智能体支持多模态截图输入由-mirix-ai-团队开发附带屏幕监控个人助手应用) — 2025 · arXiv 预印本 | ~106引 | 横跨 CoALA 全部主要记忆类别且显式分型最细 | 非参数化/智能体中心
18. [B10 潜在学习 / Oracle 检索](#b10-潜在学习--oracle-检索latent-learning--oracle-retrieval情景记忆补充参数化学习实现经验的灵活复用) — 2025 · arXiv 预印本 | ~16引 | 情景记忆 | 非参数化/智能体中心

### C. 经验回放与技能/程序记忆 (Experience replay & skill/procedural)

19. [C1 Voyager](#c1-voyager首个-llm-驱动的-minecraft-终身学习具身智能体核心为不断增长的技能库-skill-library) — 2023 · arXiv 2023 预印本 | ~1766引 | 以程序性记忆 | 非参数化
20. [C2 Agent Workflow Memory](#c2-agent-workflow-memoryawm智能体工作流记忆) — 2024 · ICML 2025 | ~174引 | 程序性记忆 | 非参数化/智能体中心
21. [C3 Synapse](#c3-synapse轨迹即范例提示--范例记忆trajectory-as-exemplar-prompting-with-memory) — 2023 · ICLR 2024 | 情景性记忆 | 非参数化/智能体中心
22. [C4 JARVIS-1](#c4-jarvis-1开放世界-minecraft-多任务智能体基于记忆增强的多模态语言模型-mlm) — 2023 · IEEE TPAMI 2024 | ~189引 | 以情景记忆 | 非参数化/智能体中心
23. [C5 AutoManual](#c5-automanual由-llm-智能体通过交互式环境学习自动构建指令手册的框架) — 2024 · NeurIPS 2024 | ~38引 | 以程序性记忆 | 非参数化/智能体中心
24. [C6 ExpGraph](#c6-expgraphmodel-agnostic-experience-learning-with-graph-structured-memory) — 2026 · arXiv 预印本 | ~0引 | 以程序性记忆 | 非参数化/智能体中心
25. [C7 WebCoach](#c7-webcoachself-evolving-web-agents-with-cross-session-memory-guidance模型无关的网页智能体跨会话记忆教练框架) — 2025 · arXiv 预印本 | ~6引 | 以情景记忆 | 非参数化/智能体中心
26. [C8 UI-Mem](#c8-ui-mem全称self-evolving-experience-memory-for-online-reinforcement-learning-in-mobile-gui-agents面向移动-gui-智能体在线强化学习的自演化经验记忆论文中亦写作-uimem) — 2026 · arXiv 预印本 | ~7引 | 以程序性记忆 | 参数化/智能体中心
27. [C9 ELL / StuLife](#c9-ell--stulife经验驱动终身学习框架--stulife-基准ell--experience-driven-lifelong-learning论文题为building-self-evolving-agents-via-experience-driven-lifelong-learning-a-framework-and-benchmark注意本文主要是一个概念性框架--评测基准而非单一可运行的记忆系统) — 2025 | 情景 | 参数化/智能体中心
28. [C10 Memp](#c10-memp写作-memp即-memory-procedural--智能体程序性记忆框架) — 2025 | ~39引 | 程序性记忆 | 非参数化/智能体中心

### D. 图结构/神经启发/生产级记忆 (Graph / neuro-inspired / production)

29. [D1 HippoRAG](#d1-hipporag受海马体启发的-llm-长期记忆检索框架) — 2024 · NeurIPS 2024 | ~223引 | 语义记忆 | 非参数化/智能体中心
30. [D2 HippoRAG 2](#d2-hipporag-2神经生物学启发的大语言模型长期记忆框架论文标题from-rag-to-memory-non-parametric-continual-learning-for-large-language-models为-hipporag-的升级版别名-hipporag-v2) — 2025 · ICML 2025 | ~151引 | 以语义记忆 | 非参数化/智能体中心
31. [D3 Zep / Graphiti](#d3-zep--graphitizep-为面向-ai-智能体的记忆层服务graphiti-为其核心的时序感知动态知识图谱引擎亦作开源框架单独发布论文标题zep-a-temporal-knowledge-graph-architecture-for-agent-memory) — 2025 · 工业界 / 开源系统 | ~197引 | 情景记忆 | 非参数化/智能体中心
32. [D4 Mem0 / Mem0^g](#d4-mem0--mem0gmem0面向生产可扩展长期记忆层mem0g-为其图记忆增强变体发音-mem-zero) — 2025 · arXiv 预印本 | 以语义记忆 | 非参数化/智能体中心
33. [D5 G-Memory](#d5-g-memorygraph-based-agentic-memory-for-llm-based-multi-agent-systems面向多智能体系统的图式分层记忆受组织记忆理论启发由-insightqueryinteraction-三层图构成的即插即用记忆模块) — 2025 · arXiv 预印本 | ~62引 | 以程序性与语义记忆为主、含情景成分的跨试验 | 非参数化/智能体中心
34. [D6 Letta](#d6-letta前身为-memgptletta-是将-memgpt-研究arxiv-231008560产品化的有状态智能体运行时平台公司由原-memgpt-团队创立别名相关memgpt现指论文中具备自编辑记忆工具的-llm-os-智能体设计范式letta-框架开源智能体框架原-memgpt-仓库改名而来letta-code记忆优先的编码智能体-cliappletta-cloud托管-api-平台adeagent-development-environment-可视化调试环境) — 2024 · 工业界 / 开源系统 | ~767引 | 覆盖多种 CoALA 记忆类型 | 非参数化/智能体中心
35. [D7 MemMachine](#d7-memmachine别名memmachine-记忆层--memverge-开源记忆系统论文标题memmachine-a-ground-truth-preserving-memory-system-for-personalized-ai-agents) — 情景记忆 | 非参数化/智能体中心
36. [D8 PlugMem](#d8-plugmem任务无关的即插即用插件式记忆模块将情景记忆结构化为以知识为单元的知识中心记忆图) — 2026 · ICML 2026 | 三类记忆统一支持 | 非参数化/智能体中心

### E. 认知架构框架 (Cognitive-architecture frameworks)

37. [E1 CoALA](#e1-coala面向语言智能体的认知架构--cognitive-architectures-for-language-agents论文标题cognitive-architectures-for-language-agents非系统方法而是一个概念性蓝图框架配套资源别名coalaawesome-language-agents基于-coala-框架的语言智能体清单仓库) — 2023 · TMLR 2024 | ~391引 | 四类记忆的统一分类学 | 混合/智能体中心
38. [E2 MemoRAG](#e2-memorag全局记忆增强检索的下一代-rag-框架全称-memorag-boosting-long-context-processing-with-global-memory-enhanced-retrieval-augmentation) — 2024 · WWW 2025 | ~104引 | 语义记忆 / 工作记忆混合 | 非参数化/智能体中心

### F. 记忆评测基准 (Memory-evaluation benchmarks)

39. [F1 LongMemEval](#f1-longmemeval聊天助手长期交互记忆评测基准含-longmemevals-约-115k-tokens-与-longmemevalm-约-500-会话约-150-万-tokens-两个标准设置外加-longmemeval_oracle-理想检索设置202509-发布去干扰的-cleaned-版202605-推出后续-longmemeval-v2-面向智能体场景) — 2024 · ICLR 2025 | 本身是"评测基准"而非记忆系统 | 非参数化/智能体中心
40. [F2 LoCoMo](#f2-locomolong-conversational-memory超长期对话记忆评测基准与数据集maharana-等人-2024-提出含问答事件摘要多模态对话生成三任务) — 2024 · ACL 2024 | ~524引 | 本身为评测基准而非记忆系统 | 非参数化/智能体中心
41. [F3 MemBench](#f3-membench面向-llm-智能体记忆能力的更全面评测基准其数据集别名亦写作-membench--membench基于-memsimmemengine-生态扩展引入事实记忆--反思记忆两个记忆层级与参与--观察两种交互场景并提供-effectivenessefficiencycapacity-多维度指标) — 2025 · ACL 2025 Findings | ~55引 | 本身是评测基准而非记忆系统 | 非参数化/智能体中心
42. [F4 Evo-Memory](#f4-evo-memory自演化记忆流式基准与框架配套基线方法-exprag-与提出的-remem行动-思考-记忆精炼流水线) — 2025 | 作为基准/框架 | 非参数化/智能体中心
43. [F5 MEMTRACK](#f5-memtrack多平台动态智能体环境下的长期记忆与状态追踪评测基准全称-memtrack-evaluating-long-term-memory-and-state-tracking-in-multi-platform-dynamic-agent-environmentspatronus-ai-出品它不是记忆系统而是一个面向企业级-swe-工作流的容器化记忆评测基准环境跨-slacklineargitgitea-三平台模拟异步事件时间线考核记忆的获取选择冲突消解能力) — 2025 · NeurIPS 2025 工作坊 SEA | ~9引 | 本身是"记忆评测基准/环境"而非记忆系统 | 非参数化/智能体中心
44. [F6 LoCoMo-Plus](#f6-locomo-plus全称-locomo-plus-beyond-factual-cognitive-memory-evaluation-framework-for-llm-agents超越事实的认知记忆评测基准与框架在-locomo-原有五类问题单跳多跳时序常识对抗之上新增第六类认知记忆-cognitive任务并配套提出基于约束一致性-constraint-consistency的统一评测范式亦写作-locomo-plus) — 2026 | 本身是「评测基准+评测框架」而非记忆系统 | 非参数化/智能体中心
45. [F7 OP-Bench](#f7-op-bench过度个性化基准配套提出-self-recheck-记忆过滤方法) — 2026 · arXiv 预印本 | ~0引 | 本身不实现记忆机制 | non-parametric/智能体中心
46. [F8 Causal-LoCoMo / 因果记忆干预](#f8-causal-locomo--因果记忆干预-cmi别名causal-memory-intervention论文题为causal-intervention-based-memory-selection-for-long-horizon-llm-agentscausal-locomo-是其配套基准cmi-是其配套方法) — 2026 · arXiv 预印本 | ~0引 | 面向情景式/语义式持久记忆 | 非参数化/智能体中心
47. [F9 MemoryAgentBench](#f9-memoryagentbench论文标题evaluating-memory-in-llm-agents-via-incremental-multi-turn-interactions面向记忆智能体memory-agents的统一评测基准提出按四项核心记忆能力精确检索-accurate-retrieval--测试时学习-test-time-learning--长程理解-long-range-understanding--选择性遗忘-selective-forgetting评测并以增量多轮交互incremental-multi-turn协议把长上下文数据集改造为逐块顺序注入自建-eventqa-与-factconsolidation-两个新数据集官方-github-仓库与-huggingface-数据集名亦作-memoryagentbench注意与同处f-记忆评测基准簇的-f3-membencharxiv-250621605人民大学华为为两个不同基准本条-arxiv-250705257由-ucsd-出品二者无血缘关系) — 2025 · ICLR 2026 | ~108引 | 本身是评测基准而非记忆系统 | 非参数化/智能体中心

### G. 学习/RL驱动的记忆控制 (Learned / RL-based memory control)

48. [G1 Memory-R1](#g1-memory-r1基于强化学习的-llm-外部记忆管理框架双智能体-memory-manager--answer-agent) — 2025 · ACL 2026 | 情景性/语义性记忆 | 参数化/智能体中心
49. [G2 Mem-α](#g2-mem-α-mem-alpha论文标题mem-α-learning-memory-construction-via-reinforcement-learning模型权重发布名-memalpha-4b基于-qwen3-4b-训练) — 2025 | 组合式多类型记忆 | 非参数化/智能体中心
50. [G3 Mem-π](#g3-mem-π-mem-pi全称-adaptive-memory-through-learning-when-and-what-to-generate将记忆建模为生成式策略-π_mem-而非检索库) — 2026 · arXiv 预印本 | ~0引 | 程序性/语义性记忆为主 | 非参数化
51. [G4 SkillOS](#g4-skillos全称-skillos-learning-skill-curation-for-self-evolving-agents一种用强化学习训练技能策展skill-curation策略的自进化智能体训练配方架构为冻结的-agent-executor--可训练的-skill-curator双模块外接一个可演化的技能仓库-skillrepo) — 2026 · arXiv 预印本 | ~3引 | 程序性记忆 | 参数化/智能体中心
52. [G5 CODESKILL](#g5-codeskill全称-codeskill-learning-self-evolving-skills-for-coding-agents为编码智能体学习自演化技能) — 2026 · arXiv 预印本 | ~0引 | 程序性记忆 | 非参数化/智能体中心

### H. 综述 (Surveys)

53. [H1 《A Survey on the Memory Mechanism of Large Language Model based Agents》](#h1-a-survey-on-the-memory-mechanism-of-large-language-model-based-agents基于大语言模型智能体的记忆机制综述这是一篇综述survey而非具体系统方法别名配套资源llm_agent_memory_survey官方-github-论文清单仓库约-495-stars被公认为该领域最早最权威的系统性综述之一提出记忆来源sources记忆形式forms记忆操作operations设计三维度--直接评估direct间接评估indirect评估二分法) — 2024 · ACM Transactions on In | ~568引 | 本综述不直接采用 CoALA 的 episodi | non-parametric/智能体中心
54. [H2 《Rethinking Memory in AI](#h2-rethinking-memory-in-ai-taxonomy-operations-topics-and-future-directionsai-中的记忆再思考分类学操作主题与未来方向最新修订版huggingfaceads-收录改题为rethinking-memory-in-llm-based-agents-representations-operations-and-emerging-topics即任务锚点中的survey-rethinking-memory-in-llm-based-agents之由来配套资源别名memory-compassgithub-仓库-survey_memory_in_ai这是一篇综述立场论文非具体记忆系统) — 2025 · arXiv 预印本 | 作为综述 | 非参数化/智能体中心
55. [H3 Memory for Autonomous LLM Agents](#h3-memory-for-autonomous-llm-agents机制评测与新兴前沿综述别名survey-memory-for-autonomous-llm-agents) — 2026 | 综述覆盖全部四类 | 参数化/智能体中心
56. [H4 From Storage to Experience](#h4-from-storage-to-experience从存储到经验llm-智能体记忆机制演化综述提出-storagereflectionexperience-三阶段演化框架) — 2026 · ACL 2026 Findings | ~6引 | 作为综述 | 非参数化/智能体中心
57. [H5 Graph-based Agent Memory](#h5-graph-based-agent-memory-taxonomy-techniques-and-applications基于图的智能体记忆综述配套开源资源库-awesome-graphmemory) — 2026 | 作为综述 | 非参数化/智能体中心
58. [H6 《Anatomy of Agentic Memory](#h6-anatomy-of-agentic-memory-taxonomy-and-empirical-analysis-of-evaluation-and-system-limitations智能体记忆解剖评测与系统局限的分类学与实证分析这是一篇综述实证分析论文非具体记忆系统其核心立场是从评测有效性--系统局限的实证透镜审视智能体记忆提出-memory-augmented-generation-mag-记忆增强生成-四结构分类学并系统暴露基准饱和指标失配骨干模型敏感性与系统级开销四大痛点配套资源别名github-awesome-list-仓库-fredjiang0324anatomy-of-agentic-memory) — 2026 · arXiv 预印本 | ~10引 | 作为综述/实证分析论文 | 非参数化/智能体中心


---

## 详细调研 (Details)


## A. 反思与失败驱动记忆 (Reflection & failure-driven)


<a id="a1-reflexion言语强化学习的语言智能体论文标题reflexion-language-agents-with-verbal-reinforcement-learning亦被称作言语强化学习verbal-rl范式"></a>

### A1 Reflexion

*Reflexion（言语强化学习的语言智能体；论文标题：Reflexion: Language Agents with Verbal Reinforcement Learning。亦被称作“言语强化学习/Verbal RL”范式）*


**基本信息 / Provenance**

- **年份**: 2023年（arXiv 预印本 v1 首次公开于 2023-03-20，arXiv:2303.11366）。
- **作者/机构**: 第一作者 Noah Shinn（东北大学 Northeastern University）；合作者 Federico Cassano、Edward Berman（东北大学），Ashwin Gopinath（麻省理工 MIT），Karthik Narasimhan、Shunyu Yao（普林斯顿大学 Princeton）。注：Semantic Scholar 作者列表把 Edward Berman 记为 “Beck Labash”（GitHub 贡献者 becklabs），论文与官方仓库以 Edward Berman 为准。
- **发表venue**: NeurIPS 2023（Advances in Neural Information Processing Systems 36）；预印本为 arXiv 2023。
- **论文链接**: https://arxiv.org/abs/2303.11366
- **代码链接**: https://github.com/noahshinn/reflexion （官方代码，MIT 许可，约 3,146 颗 star、306 fork；附带新基准 LeetcodeHardGym：https://github.com/GammaTauAI/leetcode-hard-gym ）
- **引用数**: 约 3,776 次引用（Semantic Scholar 实时数据，与任务备注 ~3775 一致），是 LLM 智能体记忆/反思方向被引最高的奠基性工作之一。

**记忆分类 / Taxonomy**

- **记忆类型**: 以情景记忆（episodic）为核心：长期记忆存储智能体对每次试错（尤其是失败）的言语自我反思文本；短期记忆为当前回合的原始交互轨迹（working/short-term memory）。论文明确区分“短期记忆=轨迹历史”与“长期记忆=自我反思模型输出”，并类比人类“既记住近期细节、又调用从长期记忆蒸馏出的重要经验”。
- **记忆结构**: 原始文本缓冲区（raw text buffer）：长期记忆是一个追加式的自我反思文本列表 mem=[sr_0, sr_1, ...]，受最大容量 Ω 约束的滑动窗口（sliding window）。不使用向量库或知识图谱；论文在“局限”一节明确建议未来工作可扩展为向量嵌入数据库或 SQL 数据库等更高级结构。
- **存储后端**: 全部为上下文内文本提示（in-context prompt）+ 本地日志文件。反思文本作为附加上下文拼接进 Actor 的下一回合提示中；官方实现把各次运行的轨迹与反思记录到本地 ./root/ 等日志目录。无外部向量数据库（FAISS/Chroma）或图数据库（Neo4j）。
- **持久化**: 上下文内、单任务/单回合序列内持久（in-context、ephemeral）：长期反思记忆在同一任务的多次试错之间被保留并累积，但不写入参数、也不构成跨任务的持久外部存储；属于非参数化、任务内持久的记忆。受 LLM 上下文窗口限制，记忆容量被有意截断（Ω 通常 1–3）。

**核心机制 / Mechanisms**

- **写入/编码**: 经验以两种形式写入：(1) 原始轨迹——Actor（LLM）与环境逐步交互产生的 (动作 a_i, 观测 o_i) 序列原样缓存为短期记忆；(2) 摘要化反思（核心）——每个回合结束、由 Evaluator 给出标量/二元奖励 r_t 后，Self-Reflection 模型 M_sr 接收三元组 {轨迹 τ_t, 奖励 r_t, 当前记忆 mem}，把稀疏的标量/二元反馈“放大（amplify）”为一段细致、可执行的自然语言经验摘要 sr_t（诊断错在哪、下次该怎么改），再追加进长期记忆 mem。这段言语反馈被论文比喻为“语义梯度信号（semantic gradient signal）”，为智能体提供明确改进方向。反馈来源灵活：可来自外部环境（二元成功/失败、精确匹配 EM），也可由 LLM 内部模拟（LLM 自评分类、自写单元测试）。
- **检索机制**: 不使用相似度/向量检索，也无 recency·importance·relevance 评分公式。读取方式是“提示拼接式”全量召回：在每个新回合，Actor 的策略 π_θ 直接以“短期轨迹记忆 + 全部（受 Ω 截断的）长期反思文本”为条件生成下一步推理与动作。本质是把最近的若干条反思（AlfWorld/HotPotQA 取最近 3 条、编程取最近 1 条）整体放入上下文，而非按查询检索。
- **反思/巩固**: 这是 Reflexion 的标志性机制——把原始失败轨迹巩固为高层洞见。Self-Reflection 模型（实例化为一个 LLM，与 Actor 同款或独立实例）在每个回合结束时被触发一次，针对失败轨迹与稀疏奖励生成第一人称、具体且可执行的反思，承担信用分配（credit assignment）任务：推断“某动作 a_i 导致后续 a_{i+1}、a_{i+2} 连锁出错”，并言语化地提出应改取替代动作 a_i'。该反思比标量奖励信息量更大，被存入 mem 供后续回合作为“自我提示（self-hints）”。HotPotQA 消融显示：加入自我反思相比仅有情景记忆（EPM，仅含最近轨迹）能带来 +8% 的绝对学习提升，说明“仅精炼/重试”不如“反思引导的精炼”有效。
- **遗忘/更新**: 采用最简单的容量截断式遗忘：长期记忆是有最大容量 Ω 的滑动窗口（论文常设 Ω=1–3，AlfWorld/HotPotQA 截断为最近 3 条反思、编程为 1 条），超出则丢弃最旧反思以适配上下文窗口。无 Ebbinghaus 衰减、无 ADD/UPDATE/DELETE 语义操作、无去重或冲突消解；更新即“追加新反思 + 滑窗淘汰旧反思”。
- **经验回放 (核心主题)**: 经验复用以“言语自我提示”的形式实现：智能体把历次（尤其失败）试错蒸馏成的反思文本持久保存在情景记忆缓冲中，并在后续同一任务的重试中反复作为上下文条件被复用，从而避免重复犯同样错误、压缩搜索空间（如 AlfWorld 中“早期就识别拿错物品的错误”“系统化搜遍房间所有容器”）。这并非传统 RL 的参数化 replay buffer，而是非参数化、提示级的经验回放——把过去轨迹的教训以自然语言反复注入未来回合的决策上下文。论文将此形式化为 Algorithm 1：trial→evaluate→self-reflect→append-to-mem 的迭代循环，直到 Evaluator 判定通过或达到最大试错次数。

**学习维度 / Learning**

- **学习范式**: 非参数化（in-context、提示级）：不更新任何 LLM 权重、不做梯度下降或微调，纯粹通过把言语反馈加入上下文来“强化”智能体——故称“言语强化学习（verbal RL）”。论文把策略参数化为 θ={M_a, mem}，即“Actor 的 LLM 参数 + 记忆编码”，学习只发生在 mem（记忆）这一侧。
- **失败学习 (核心主题)**: 核心主题，整篇工作即围绕“从过往失败中学习”构建。(1) 检测——由 Evaluator M_e 产生稀疏反馈：推理任务用精确匹配 EM 二元打分，决策任务用预定义启发式（如同一动作-响应循环超过 3 次、或单环境动作数超过 30 视为低效/幻觉而触发反思）与 LLM 自评分类，编程任务用自写单元测试。(2) 利用——Self-Reflection 把失败信号放大为针对性的诊断与改进计划，存入记忆并在重试时引导 Actor 改换决策。论文用 AlfWorld 分析指出：基线 ReAct 常陷入“误以为已持有某物品”的幻觉且无法回溯，收敛于约 22% 的幻觉失败率且无长期恢复；Reflexion 几乎消除此类失败，通过把冗长失败轨迹蒸馏为相关经验作为未来“自我提示”。论文还探讨了三类反馈生成方式：简单二元环境反馈、预定义启发式、以及 LLM 自评。
- **在线 vs 离线**: 在线（online）、部署时逐回合构建：记忆在与环境交互的过程中按 episode 逐步生成与累积，无离线批量训练阶段；每个任务从空记忆开始，在有限次试错（如 AlfWorld 12 步、编程取最近 1 条经验）内在线积累反思。

**评测 / Evaluation**

- **任务领域**: 三大类：(1) 序列决策（具身/文本家居环境，AlfWorld）；(2) 知识密集型推理问答（HotPotQA，基于 Wikipedia 的多跳问答）；(3) 代码生成/编程（Python 与 Rust 函数体生成、竞赛级编程）。
- **基准**: AlfWorld（134 个文本交互环境，6 类家居任务，基于 TextWorld）；HotPotQA（11.3 万 QA 对的 Wikipedia 多跳数据集，实验取 distractor 划分中随机 100 题；CoT 用 6-shot、ReAct 用 2-shot、自我反思用 2-shot）；HumanEval（Python 与经 MultiPL-E 翻译的 Rust）；MBPP（Python 与 Rust）；LeetcodeHardGym（作者新提出的基准，含 40 道 GPT-4 预训练截止日 2022-10-08 之后发布的 Leetcode hard 题，覆盖 19 种编程语言）。
- **报告增益**: 相对强基线的绝对提升标题数字：AlfWorld 决策任务在 12 次迭代学习步内提升约 +22%（ReAct+Reflexion 用简单启发式检测幻觉/低效规划，在 134 个任务中完成 130 个，对比 ReAct 仅基线在第 6–7 试错之间停止提升、收敛于 22% 幻觉率）；HotPotQA 推理任务提升约 +20%（在 CoT-GT 情境下，CoT(GT) 仍有 39% 题答错，Reflexion 在无真值答案下帮助纠错、把准确率提升约 +14%；消融显示自我反思比仅情景记忆再 +8% 绝对值）；编程任务 HumanEval Python 达 91.0% pass@1，超过此前 SOTA 的 GPT-4（80.1%），绝对提升约 +11。表 1 完整 pass@1：HumanEval-PY 91.0（GPT-4 SOTA 80.1，前 SOTA CodeT+GPT-3.5 65.8）；HumanEval-RS 68.0（GPT-4 60.0）；MBPP-PY 77.1（GPT-4 80.1——唯一未超基线，因内部单测假阳性率 16.3% 远高于 HumanEval-PY 的 1.4%）；MBPP-RS 75.4（GPT-4 70.9）；LeetcodeHard-PY 15.0（GPT-4 7.5）。编程消融（HumanEval-Rust 最难 50 题，GPT-4）：基线 0.60、仅测试生成无反思 0.60、仅反思无测试 0.52（反而劣于基线）、完整 Reflexion 0.68，证明测试生成与自我反思协同缺一不可。论文未提供 token 成本/延迟的量化对比。
- **对比基线**: 对照对象包括：无记忆/不反思的基线智能体——ReAct（决策与问答的 SOTA 提示架构）、Chain-of-Thought / CoT(GT)（带真值上下文的推理基线）、各任务的“无反思”重试版；编程上与 GPT-4 零样本、前 SOTA（CodeT+GPT-3.5 / CodeT+Codex）对比；以及消融对照 LAST_ATTEMPT（仅给上次轨迹）与 episodic-memory（EPM，仅最近轨迹无反思）。关联工作横向对比 Self-Refine、Beam search 等（按是否支持自我精炼/隐含约束/决策/二元奖励/记忆五维区分，Reflexion 是唯一全满足者）。

**分析 / Analysis**

- **关键创新**: 提出“言语强化学习（verbal reinforcement）”新范式：不更新模型权重，而是把环境的二元/标量反馈转化为自然语言自我反思，存入情景记忆缓冲并作为下一回合的附加上下文，充当“语义梯度信号”引导智能体从失败中快速少样本学习。把策略显式参数化为“LLM 参数 + 记忆编码”，首次系统性地把可解释的情景记忆 + 自我反思打通为可跨决策/推理/编程三类任务通用的轻量自改进框架，并在 HumanEval 上以 91% pass@1 超越 GPT-4。
- **局限**: (1) 本质是用自然语言做策略优化，可能陷入非最优局部极小；(2) 长期记忆仅为有上限的滑动窗口（Ω 通常 1–3），无真正遗忘/合并/冲突消解，论文自承可扩展为向量库或 SQL 库；(3) 强依赖 LLM 自评能力或启发式的质量，无成功的形式化保证；(4) 编程任务受限于自写测试质量——测试不全会产生假阳性（如 MBPP-PY 假阳性率 16.3% 致其低于 GPT-4 基线）、写错测试会产生假阴性；测试驱动开发对非确定性/带副作用/依硬件或并发的函数难以指定准确输入输出；(5) 评测任务集偏小（每基准约 40–134 任务），且 GPT-4 访问受限、API 费用高使逐条复现困难；(6) 反思以文本注入上下文，受 LLM 上下文窗口约束。
- **与其他工作关系**: 本研究 A 簇“反思/失败驱动”的奠基项（A1）。建立在 ReAct（Yao et al. 2023，沿用其推理-行动提示与 few-shot 轨迹）与 Self-Refine（Madaan et al. 2023，单步自我精炼）之上，并区别于后者——Reflexion 增加了跨试错持久的情景记忆与失败信用分配。被本研究多个后续项直接扩展：A2 Retroformer 在其上加入“可训练的回顾模型 + 策略梯度（PPO）”把反思生成参数化优化；A5 ExpeL、A6 ReasoningBank 等把“失败/成功经验”进一步抽象为跨任务可检索的经验/规则库；下游工作（如本批 2025 年 ISAICS 的 “Robust Verbal RL”）用 FAISS 向量记忆替换其本地文本存储、加入反思质量过滤，正面回应了其“记忆结构过简”的局限。受 Brooks et al. (2022) 的上下文内策略迭代启发设计记忆组件。概念上把传统 RL 的轨迹/价值/策略概念映射到语言空间。
- **可复现性**: 高：官方开源代码（GitHub noahshinn/reflexion，MIT，~3,146 star、306 fork，5 位贡献者，2025-01 仍在维护），提供 HotPotQA、AlfWorld、编程三套实验的 notebook/脚本及论文全部运行日志（./alfworld_runs/root、./hotpotqa_runs/root、./programming_runs/root），并开源新基准 LeetcodeHardGym。所用基准均为公开数据集。社区采用度极高（~3,776 引用，多个第三方/APPL 等复现实现）。唯一复现障碍是 GPT-4 访问受限与 API 费用，论文明确提示用隔离执行环境运行自动代码实验。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式管线）：记忆的写入/检索/遗忘均由固定启发式与提示流程决定（每回合触发一次反思、滑动窗口截断、全量上下文拼接），并不用 RL/训练去学习“何时存/取/更新”的记忆管理策略本身。它是 2025–26 “学习型记忆控制”代际划分之前的非参数化代表，后续 A2 Retroformer、G 簇 Memory-R1/Mem-α 等正是针对这一点把记忆控制策略本身参数化/可训练化。
- **记忆主体**: 智能体中心（agent-centric）：记忆与学习对象是智能体自身的试错经验（失败轨迹与自我反思），目的在于自我改进任务表现，而非记住用户信息做个性化。
- **多智能体记忆**: 单智能体框架。不涉及多智能体共享或路由记忆；但在概念上把系统分解为 Actor、Evaluator、Self-Reflection 三个 LLM 角色协作（非多智能体记忆系统，而是单智能体内部的模块分工）。
- **时序推理支持**: 不支持显式时间推理：不建模事实有效期窗口、事件排序或时间日历。仅隐含“试错回合的先后顺序”（trial t → t+1），用于决定反思的累积与滑窗淘汰。
- **模态**: 纯文本（text-only）。AlfWorld（文本交互）、HotPotQA（文本问答）、编程（代码/编译器反馈）均为文本观测与文本/动作 API 交互，无视觉或多模态记忆。
- **过度个性化/记忆安全风险**: 基本不适用/未涉及：该工作属智能体自我改进而非用户个性化，不处理过度个性化、有害/过时/谄媚记忆或隐私治理。论文“更广泛影响”一节提及自改进智能体被滥用的风险，并正面指出言语反思可被监控从而比黑箱 RL 策略更可解释、可诊断、利于对齐，但未提出具体记忆安全机制。
- **冲突/矛盾处理**: 无专门的冲突/矛盾事实消解机制。反思记忆只做追加与滑窗淘汰；“纠错”通过新反思引导 Actor 改换动作来间接实现，不存在显式 UPDATE/合并矛盾事实的操作。

**不确定字段 / Uncertain**

- 技能/程序归纳 (`skill_procedural_induction`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="a2-retroformer回顾式大语言智能体含可训练的回顾模型retrospective-model论文标题retroformer-retrospective-large-language-agents-with-policy-gradient-optimization"></a>

### A2 Retroformer

*Retroformer（回顾式大语言智能体，含可训练的回顾模型/retrospective model；论文标题：Retroformer: Retrospective Large Language Agents with Policy Gradient Optimization）*


**基本信息 / Provenance**

- **年份**: 2023年（arXiv 预印本首次公开于 2023-08-04，v1 版本 arXiv:2308.02151）
- **作者/机构**: 第一作者 Weiran Yao，通讯/合作者包括 Shelby Heinecke、Juan Carlos Niebles、Zhiwei Liu、Yihao Feng、Le Xue、Rithesh Murthy、Caiming Xiong、Silvio Savarese 等，主要隶属 Salesforce Research（Salesforce AI Research）。
- **发表venue**: ICLR 2024（被接收为 Spotlight 论文；预印本为 arXiv 2023）。
- **论文链接**: https://arxiv.org/abs/2308.02151 （OpenReview: https://openreview.net/forum?id=KOZu91CzbK ）
- **代码链接**: https://github.com/weirayao/Retroformer （官方代码，Apache-2.0，约 39 颗 star；项目主页 https://Retroformer.github.io/ ）
- **引用数**: 约 126 次引用（Semantic Scholar 实时数据，与任务备注 ~126 一致）。

**记忆分类 / Taxonomy**

- **记忆类型**: 以情景记忆（episodic）为核心：长期记忆是对历次失败试错的自我反思（self-reflection）文本；同时具备工作记忆（working/short-term，当前回合轨迹）。可训练的回顾模型本身则把失败经验固化为参数化的程序性/语义性知识（procedural/semantic）。
- **记忆结构**: 分层文本记忆 + 强化学习经验回放缓冲（replay buffer）。短期记忆为当前回合的原始轨迹缓冲（state-action-reward 序列）；长期记忆为追加到 Actor 提示中的反思文本；replay buffer 为存储 (反思指令 x, 反思响应 y, 回合回报 G) 三元组的本地数据集，用于训练回顾模型。
- **存储后端**: 全部为上下文内文本提示（in-context prompt）+ 本地数据集文件（replay buffer，存储指令-响应-回报三元组）；可训练知识最终落入回顾模型（LongChat-7B-16k）的参数（LoRA 适配器权重）。未使用向量数据库或图数据库。
- **持久化**: 混合：短期/长期反思记忆为上下文内（ephemeral，跨回合追加到提示，单任务内累积）；replay buffer 为外部持久化文件；最关键的“记忆”最终以参数形式（LoRA 微调权重）烘焙进回顾模型，可跨任务/跨环境复用。

**核心机制 / Mechanisms**

- **写入/编码**: 经验以两种方式写入：(1) 原始轨迹——Actor 与环境逐步交互产生的 (状态 s, 动作 a, 奖励 r) 序列作为短期记忆原样缓存；(2) 摘要化反思——每个回合结束时，回顾模型 M_r 接收完整失败轨迹与回合回报 G 作为反思提示 x，生成一段简洁的自我反思 y，诊断失败根因并提出新的高层行动计划。该反思 y 被追加进 Actor 提示作为长期记忆，防止重复犯错。同时把 (x, y, G) 三元组写入 replay buffer 供训练用。奖励采用奖励塑形（reward shaping），尽量用软匹配（如 F1 分数）而非二元精确匹配来评估输出与答案/商品的对齐度。
- **检索机制**: 不依赖相似度/向量检索。读取是“提示拼接式”召回：当前回合把短期轨迹记忆 + 累积的长期反思文本直接拼进 Actor 的 ReAct 提示中条件化生成下一步推理与动作。训练得到的回顾模型在在线执行时用 best-of-n 采样器（n 个候选反思响应，由 RLHF 学到的奖励模型打分）挑选更优的反思响应注入提示。本质是“把全部相关失败反思放入上下文”而非按需检索。
- **反思/巩固**: 这是该工作的核心机制：把原始失败轨迹经验提炼为高层洞见（反思）。普通 Reflexion 用冻结 LLM 生成反思，常常只是复述失败动作序列、信息量低；Retroformer 的创新在于用强化学习把“反思生成”这一巩固过程本身训练出来——回顾模型被微调成能做更准确的信用分配（credit assignment）与根因分析，输出可执行的改进计划。触发时机：每个回合（episode）失败/结束时触发一次反思生成；训练时离线在训练集任务上滚动 N 次试错收集数据后用 RLHF/PPO 微调。
- **遗忘/更新**: 无显式遗忘/衰减机制（不含 Ebbinghaus 衰减、ADD/UPDATE/DELETE 操作）。长期反思记忆在单任务内单调累积；记忆“更新”体现在两层：回合间用新反思替换/补充旧反思以避免重复错误，以及通过 PPO 迭代微调回顾模型参数来整体改进反思质量。replay buffer 持续追加样本。
- **经验回放 (核心主题)**: 核心贡献之一。past 轨迹通过 replay buffer 被显式复用：把跨任务、跨环境的 (反思指令, 反思响应, 回合回报) 三元组存入本地数据集 D_RL，再从中采样以 RLHF/PPO 微调回顾模型。回报差 ΔG = G_{i+1} − G_i（相邻两次试错的回报变化）被当作反思响应的评分/奖励，正 ΔG 表示该反思帮助 Actor 改正、应给高分。这样智能体不仅利用当前任务历次失败的教训，还能从其它相关任务的成败中探索学习，把经验蒸馏成参数化的反思策略，实现“跨任务经验复用”。

**学习维度 / Learning**

- **学习范式**: 混合（hybrid）：Actor LLM（GPT-3/GPT-4）始终冻结、仅做非参数化的提示级（in-context）适应；而回顾模型（LongChat-7B 本地小模型）走参数化梯度学习（SFT + 奖励模型 + PPO，使用 LoRA r=1 或 r=4）。把 Actor 视为环境的一部分，从而能用任意奖励信号对系统中的某个组件做策略梯度优化，而无需访问 Actor 参数或对其反传梯度。
- **失败学习 (核心主题)**: 核心主题。整套框架就是围绕“从失败中学习”构建：(1) 检测——以稀疏奖励（如二元成功/失败状态，或软 F1 奖励）判定失败，回顾模型结合当前轨迹与持久记忆定位失败根因；(2) 利用——生成针对失败的诊断式反思与新计划，追加进 Actor 提示阻止重复错误；(3) 训练——用相邻试错的回报差 ΔG 作为奖励，通过 PPO 微调回顾模型，使其学会更好的信用分配，把“哪个动作 a_t 导致了后续连锁错误与最终失败”识别出来并在下次试错中改换替代动作 a_t'。论文图 1/图 5 用 Teen Titans 案例说明：冻结 LLM 的反思只是复述失败动作导致无限循环，而强化后的反思给出可执行的纠错指引。
- **在线 vs 离线**: 两者兼有：离线训练——在训练集任务上用冻结 Actor + 初始回顾模型滚动 N 次试错收集 D_RL，再用标准 RLHF/PPO 流程离线微调回顾模型；在线执行——评估时用 best-of-n 采样器（奖励模型打分）在线生成更优反思响应。

**评测 / Evaluation**

- **任务领域**: 三类：搜索式知识问答（基于 Wikipedia 检索）、具身/文本决策（embodied 机器人式文本动作）、网页购物浏览（GUI/web 导航）。
- **基准**: HotPotQA（distractor dev 划分，100 个验证任务，搜索式问答）、AlfWorld（134 个任务，具身文本决策）、WebShop（100 个任务，网页购物）。
- **报告增益**: 以成功率（success rate）为指标，相对 ReAct/Reflexion 基线提升明显。摘要级提升：HotPotQA 在 4 次重试下成功率约 +18%，AlfWorld 在 3 次重试下约 +36%，WebShop 在 4 次重试下约 +4%。表 2 具体数值（GPT-4 Actor，LoRA r=4，N=4 重试）：HotPotQA 54%（ReAct 40%、Reflexion 52%）；AlfWorld 100%（ReAct 77.61%、Reflexion 85.07%，Retroformer 3 次重试内解到 100%）；WebShop 46%（ReAct 42%、Reflexion 44%）。GPT-3 Actor 下 HotPotQA 也达 54%，优于 Jang(2023) 用更大的 text-davinci-003 作反思组件报告的 50% SOTA。在线 RL 基线 SAC（2.25M 参数）仅 27%/58.95%/30%。可训练参数极小：LoRA r=1 仅 0.53M、r=4 仅 2.25M（Actor 0 可训练参数）。强调“学习速度更快”（前几次试错提升最显著）。论文未报告 token/延迟成本指标。
- **对比基线**: (1) ReAct——冻结、完全不用环境奖励的 SOTA 语言智能体架构；(2) Reflexion——用环境的言语反馈但不用显式梯度信号的 SOTA 反思式智能体；(3) SAC（Soft Actor-Critic）——一种在线 RL 算法基线。还对照了 Jang(2023) 报告的 HotPotQA 50% 成绩。

**分析 / Analysis**

- **关键创新**: 首次（作者自称是最早之一）把策略梯度/RLHF 引入语言智能体：把冻结的 Actor LLM 视为环境的一部分，用相邻试错的回报差作奖励，通过 PPO 微调一个即插即用的小型“回顾模型”，使其学会生成更高质量、可做信用分配的失败反思——从而以梯度方式优化提示，而无需访问或反传 Actor 参数。等于把 Reflexion 的“言语反思”从启发式升级为可训练的参数化策略。
- **局限**: (1) WebShop 提升有限（仅约 +4%），说明言语反思类方法对需要大量探索/更精确查询的网页浏览环境并非最优；(2) 需要额外训练一个本地 7B 回顾模型（SFT+奖励模型+PPO 三阶段管线），工程与算力开销高于纯提示方法；(3) 长期记忆只在单任务内单调累积、无真正遗忘/冲突消解机制，记忆随重试增长可能膨胀；(4) 依赖可定义的（软）奖励函数，奖励稀疏或难以塑形的任务受限；(5) 评测任务集较小（每环境约 100–134 任务），泛化与规模化证据有限；(6) 反思仍以文本注入上下文，受 Actor 上下文窗口约束。
- **与其他工作关系**: 直接扩展并改进 A 簇的 Reflexion（Shinn et al. 2023）：Reflexion 用冻结 LLM 做言语自我反思且不用梯度，Retroformer 增加“可训练的回顾模型 + 策略梯度（PPO）”把反思生成参数化优化（对应本研究 A 簇“反思/失败驱动”定位，可表述为“在 A1 Reflexion 上加上参数化的 retro-model”）。Actor 沿用 ReAct（Yao et al. 2023）提示框架；与 RAP（用 MCTS + 世界模型）、Self-Refine（单 LLM 自我精炼）形成对比——其表 1 从是否支持梯度学习/任意奖励/迭代精炼/隐含约束/决策/记忆六维区分，Retroformer 是唯一全部满足者。RL 工具链基于 trl/trlx，方法谱系上属于把 RLHF/PPO（Schulman 2017）应用到智能体提示优化。
- **可复现性**: 较好：官方开源代码（GitHub weirayao/Retroformer，Apache-2.0，~39 star），提供 SFT/奖励/PPO 三个训练脚本（sft_run.py、reward_run.py、ppo_run.py）与 HotPotQA/AlfWorld/WebShop 三套评测环境（评测环境大量复用 Reflexion 官方仓库），并提供偏好数据样例与模型/数据集链接（HF）。所用环境与基准均为公开数据集。社区影响中等（约 126 引用，ICLR 2024 Spotlight）。需配置多个独立 Python 环境且要自部署本地 LLM，复现门槛偏中。

**补充维度 / Supplemented (2025-26 frontier)**

- **记忆主体**: 智能体中心（agent-centric）：记忆与学习对象是智能体自身的试错经验（失败轨迹与反思），目的是自我改进任务表现，而非记住用户信息做个性化。
- **多智能体记忆**: 单智能体框架。不涉及多智能体共享/路由记忆；但架构在概念上区分了 Actor 与 Retrospective 两个 LLM 角色（actor-critic 式分工），并非多智能体记忆系统。
- **时序推理支持**: 不支持显式时间推理：不建模事实有效期窗口、事件排序或时间日历。仅隐含“试错回合先后顺序”（episode i → i+1）用于计算回报差奖励。
- **模态**: 纯文本（text-only）。HotPotQA/AlfWorld/WebShop 均以文本观测与文本动作 API 交互，无视觉/多模态记忆。
- **过度个性化/记忆安全风险**: 未涉及。该工作不处理记忆安全、过度个性化、有害/过时/谄媚记忆或隐私治理等问题（属智能体自我改进而非用户个性化场景），相关维度不适用。
- **冲突/矛盾处理**: 无专门的冲突/矛盾事实消解机制。反思记忆只是累积与覆盖式更新；“纠错”通过新反思指引 Actor 改换动作、以及用回报差信号训练回顾模型来间接实现，不存在显式 UPDATE/合并矛盾事实的操作。

**不确定字段 / Uncertain**

- 学习型记忆控制 (`learned_memory_control`)
- 技能/程序归纳 (`skill_procedural_induction`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="a3-clincontinual-learning-from-interactions持续从交互中学习的语言智能体"></a>

### A3 CLIN

*CLIN（Continual Learning from INteractions，持续从交互中学习的语言智能体）*


**基本信息 / Provenance**

- **年份**: 2023（arXiv 预印本于 2023 年 10 月 16 日首次公开）
- **作者/机构**: Bodhisattwa Prasad Majumder、Bhavana Dalvi Mishra、Peter Jansen、Oyvind Tafjord、Niket Tandon、Peter Clark（艾伦人工智能研究所 AI2 / Aristo 团队）以及 Li Zhang、Chris Callison-Burch（宾夕法尼亚大学），Peter Jansen 兼属亚利桑那大学。第一作者与通讯单位均为 AI2。
- **论文链接**: https://arxiv.org/abs/2310.10134
- **代码链接**: https://github.com/allenai/clin （AI2 官方仓库，约 89 颗星，最近一次推送 2023-12）；项目主页 https://allenai.github.io/clin/
- **引用数**: 约 78 次引用（Semantic Scholar，截至 2026 年中查询）。

**记忆分类 / Taxonomy**

- **记忆类型**: 以程序性/语义性记忆为主：存储的是“因果抽象（causal abstractions）”——关于动作与其效果之间因果关系的陈述句（如“打开冰箱对于拿到苹果汁是必要的”），即一种语言化的‘动作模型/世界如何运作’的知识，兼具情景性（记录上一轮试验中失败的动作）。属于 CoALA 框架中的程序性+语义性记忆混合，而非纯情景缓冲。
- **记忆结构**: 持久化、动态演化的自然语言句子集合（textual memory S），每条句子是一个因果抽象三元组式陈述：X（动作）—关系—Y（动作/目标）。关系仅取两类语义：‘necessary（必要）’与‘does not contribute（无贡献）’；并用‘may（可能）/ should（应该）’两个语言化不确定性等级修饰置信度。非向量库、非知识图谱，而是结构化受限的 NL 列表（编号列表）。跨回合（episode）层面再蒸馏出‘元记忆 meta-memory’。
- **存储后端**: 纯文本记忆，作为 prompt 上下文注入冻结 LLM；不使用向量数据库或图数据库存储记忆本体。仅在 executor 做动作合法性匹配时使用 sentence-transformer 预训练嵌入（Reimers & Gurevych 2019）做近似检索，阈值 0.9。底层 LLM 为冻结模型（论文主实验基于 GPT 系列，无参数更新）。
- **持久化**: 外部、持久化、durable：记忆 S 在试验（trial）之间、回合（episode）之间、甚至跨任务/跨环境持续保留并更新，独立于模型权重之外。模型本身保持冻结（in-context / 非参数），记忆不写入权重。

**核心机制 / Mechanisms**

- **写入/编码**: 在每个 trial 结束时（无论成功或失败），由一个独立的‘记忆生成器（memory generator，冻结 LLM）’对当前 trial 的完整轨迹（goal-action-observation 三元组序列）+ 最终奖励 r∈[0,1] + 此前记忆进行反思，生成/更新一份编号的因果抽象学习清单。生成被严格约束为固定句式模板：‘X MAY/SHOULD BE NECESSARY to Y’ 或 ‘X MAY NOT/DOES NOT CONTRIBUTE to Y’，从而把原始经验编码为简洁的、可迁移的因果命题，而非逐字轨迹或泛泛的“有用提示”。这是与 Reflexion 关键区别：编码的是‘世界如何运作’的因果模型而非任务专属提示。
- **检索机制**: 由 controller（冻结 LLM）在每一步用‘当前任务指令 m + 至今的 trial 历史’作为查询，从记忆 S 中选取一条或多条它判断对推进下一动作有用的记忆条目（基于 LLM 自身判断的相关性选择，而非显式相似度打分公式）；若选中则把该学习追加进上下文以生成下一个子目标 g_{t+1}，否则仅依据轨迹历史生成目标。动作侧（executor）则用 sentence-transformer 嵌入做候选动作到合法动作模板的相似度匹配（阈值 0.9），不足则迭代 self-refine 重试最多 5 次。因此‘记忆检索’是 LLM 语义选择式，而非经典的 recency·importance·relevance 加权检索。
- **反思/巩固**: 核心机制。两层 raw→insight 转化：(1) 回合内适应（adaptation）：每个 trial 结束后，记忆生成器把 T_k + r_k + 旧记忆 {S} 反思蒸馏为新记忆 S_{k+1}（论文式(2)），记忆非单调增长——可丢弃旧的错误条目、新增正确条目，实现增删；(2) 跨回合泛化（generalization）：面对新任务/新环境时，选取每个先前回合的‘最佳（best）’记忆作为输入，蒸馏出更抽象的‘元记忆 meta-memory’S_new（式(3)），用更通用的因果命题（如‘移动到不同房间有助于寻找物体’）支撑零样本迁移，随后还可在新环境继续 adaptation 精炼。反思由冻结 LLM 通过专用 prompt（Adapt/Gen-Env/Gen-Task 三套模板）触发。
- **遗忘/更新**: 记忆非单调：每轮由记忆生成器重写整份清单，可主动丢弃被判定为错误或无用的旧条目、修改置信度（may↔should）、新增条目；负面学习（‘does not contribute’）也被显式保留以避免重复无效动作。无 Ebbinghaus 时间衰减，无显式去重/合并算子，更新是‘整体重生成式’而非逐条 ADD/UPDATE/DELETE 操作。论文承认部分记忆条目可能错误，期望后续迭代被丢弃或修正。
- **经验回放 (核心主题)**: 核心主题。过去轨迹不被逐字回放，而是被蒸馏成因果抽象后复用：同任务同环境下，前几轮试验学到的因果命题在后续 trial 的 controller prompt 中被检索复用，缩小动作搜索空间、减少步数（后期 trial 平均步数显著下降）；跨回合时，多个回合的‘最佳记忆’被汇总为元记忆，作为新任务/新环境的零样本起点——相当于把分散经验蒸馏为可迁移策略知识，而非保留 replay buffer。是‘蒸馏式策略复用’而非示例提示或代码技能库。

**学习维度 / Learning**

- **学习范式**: 非参数（non-parametric / in-context、prompt 层）持续学习：冻结底层 LLM，完全不做梯度更新或微调，所有‘学习’发生在外部文本记忆的迭代重写上。论文明确称之为一种‘新颖的非参数学习范式’。
- **失败学习 (核心主题)**: 核心主题。失败被显式利用：记忆生成器接收带‘EVALUATION REPORT / REWARD_FINAL’的轨迹（明确标注本轮成功/部分成功/失败程度），既从成功轨迹提炼‘必要（necessary）’因果，也从失败/无效动作提炼‘无贡献（does not contribute）’的负面学习条目（如‘移动到另一房间对冻结水银无贡献’、‘在工坊拿电池对找豌豆种子无贡献’）。这些负面命题被持久写入记忆，用于在未来 trial 中规避无效/错误动作，缩小搜索空间。借鉴 Reflexion/Generative Agents 的反思思想，但把失败抽象为可跨任务迁移的因果命题而非任务专属提示。
- **技能/程序归纳**: 在‘语言化动作模型（action model）’意义上归纳可复用程序性知识：因果抽象本质上描述‘何种动作对达成何种子目标是必要的’，可视为正式规划中 action model learning 的语言版本/现代化版本。但不显式产出可调用的代码技能或命名子例程（与 Voyager 的代码技能库不同），而是以自然语言因果命题形式被 controller 隐式调用。
- **在线 vs 离线**: 在线（online，部署期、逐 trial）构建为主：记忆在与环境真实交互的回合中、在 trial 之间（而非 trial 之内）增量更新；跨回合泛化时也是基于已积累的在线经验做 batch 式蒸馏（meta-memory），但不依赖离线的固定训练轨迹语料。无 gold 轨迹/示范数据。

**评测 / Evaluation**

- **任务领域**: 文本型具身/交互式科学实验环境（embodied text-game）。主战场为 ScienceWorld（执行如煮沸液体、种植物、孟德尔遗传判定、测摩擦力等基于科学的具身目标）；并在 ALFWorld（家庭场景 pick/place/clean 等）做迁移验证。属于具身决策与持续学习领域，而非 QA / 多轮对话 / GUI / 编码。
- **基准**: ScienceWorld（Wang et al. 2022；选取 18 个任务、每任务前 10 个测试变体，排除电学任务，区分 Short<37 步 / Long≥37 步）；ALFWorld（Shridhar et al. 2021）。对比中位任务长度约 37 步，强调长程复杂任务。
- **报告增益**: ScienceWorld 同任务同环境适应（adaptation）：CLIN 总分 ADAPT=62.2，超越 SOTA 反思智能体 Reflexion(39.4) 约 23 个绝对点（ReAct=29.6，BASE=48.6；Short 类 62.8 vs Reflexion 49.9，Long 类 61.6 vs 28.9）。迁移/泛化：迁移到新环境使零样本性能提升约 4 点、新任务提升约 13 点；再经持续记忆更新（适应）可在新环境再提升 17 点、新任务再提升 7 点（即论文摘要数字）。后续 OpenReview/COLM 版给出合并口径：ScienceWorld 总体泛化提升约 21 点、ALFWorld 较 Reflexion 高约 1.4 点且泛化提升约 11 点，并在不使用任何 gold 轨迹的情况下超越既有 RL 智能体与语言智能体达到 SOTA。效率上：后期 trial 平均步数下降，体现更快收敛（无显式 token/延迟数字报告）。
- **对比基线**: ReAct（Yao et al. 2022，无长期记忆）、Reflexion（Shinn et al. 2023，任务/环境专属‘提示’式反思，主要 SOTA 对手）、CLIN 自身的 BASE（无记忆/首轮 Trial-0）作为消融对照；并与既有强化学习智能体（如 DRRN、CALM 等 ScienceWorld 过往 SOTA）以及 SwiftSage 等语言智能体在泛化迁移上比较。

**分析 / Analysis**

- **关键创新**: 首个无需参数更新、即可在‘任务与环境同时变化’条件下持续改进的语言智能体；关键创新是把反思记忆从 Reflexion 式‘任务专属提示（hints）’升级为受限句式表达的‘因果抽象（causal abstractions / 语言化动作模型）’持久动态记忆，并通过‘适应（回合内）+ 泛化（跨回合元记忆）’两层蒸馏实现跨试验、跨任务、跨环境的快速迁移与持续学习。
- **局限**: (1) 记忆可能含错误条目，依赖后续迭代自我修正，无形式化的冲突检测/去重机制；(2) 句式被严格限制为 necessary / does-not-contribute 两类关系，表达力受限，难以刻画数值/时间/条件复杂因果；(3) 评测局限于文本模拟世界（ScienceWorld/ALFWorld），动作空间由模拟器提供合法模板，未验证开放真实环境或高维感知；(4) 无真正的时间衰减/遗忘曲线，记忆增长与质量随回合累积的可扩展性未充分压力测试；(5) 完全依赖强冻结 LLM 的反思与选择能力，记忆检索为 LLM 主观选择、无可控打分公式；(6) 无 token/成本/延迟量化分析。
- **与其他工作关系**: 属本研究 A 簇（反思与失败驱动）。直接建立在 A1 Reflexion（Shinn et al. 2023）之上并将其作为主要对比基线——把 Reflexion 的短期、任务专属‘提示式反思’替换为长期、可跨任务迁移的‘因果抽象记忆’，并新增跨回合元记忆泛化层；与 Generative Agents（Park et al. 2023，记录失败动作的反思）思想相通。与同期 Voyager（代码技能库、Minecraft）和 ExpeL（经验池）同属‘冻结模型 + 非参数持续学习’谱系，但 CLIN 强调‘因果世界模型’而非代码技能或经验检索。可视为经典规划中 action-model learning 在 LLM 语境下的语言化复兴；其‘把经验蒸馏为可迁移命题’的思路是后续 ExpeL/ReasoningBank 类经验记忆方法的早期代表。
- **可复现性**: 可复现性较好：官方开源代码 https://github.com/allenai/clin （AI2，约 89 星），有完整项目主页、论文附录给出全部 prompt 模板（controller/executor/三套 memory generator）、评测任务清单与超参（相似度阈值 0.9、重试 5 次）。所用 ScienceWorld、ALFWorld 均为公开基准；底层依赖商用冻结 LLM（如 GPT-4），存在 API 版本漂移导致的数字复现风险。社区采用度中等，常被作为持续学习/记忆智能体综述的代表性引用。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否——记忆管理策略是启发式/prompt 流水线，而非用 RL/训练学得的策略。何时写、写什么、检索什么均由冻结 LLM 按固定 prompt 模板在固定时机（每个 trial 末写、每步检索）执行，无可训练的记忆控制器。属于 2025-26 ‘学习型记忆控制’分水岭之前的启发式一代。
- **记忆主体**: 智能体中心（agent-centric）：记忆的是智能体自身在环境中的交互经验（动作—效果因果），用于自我改进与跨任务迁移，而非记忆用户信息做个性化。与 ReasoningBank/Voyager 同属 agent-centric 自改进谱系，区别于 Mem0/Zep 的 user-centric 路线。
- **多智能体记忆**: 单智能体。记忆 S 归属单个 CLIN 智能体，无多智能体共享/路由记忆机制（无 G-Memory / MIRIX 式的洞见层、查询层、跨智能体记忆路由）。内部模块分工（controller/executor/memory generator）是单智能体内的角色分解而非多智能体协作。
- **时序推理支持**: 弱。记忆按 trial/episode 序号顺序累积更新（隐含时间顺序与‘最近 trial’概念），但不显式建模事实有效期窗口、事件日历或时间区间推理（无 Zep/Graphiti 式时间双时态建模）。因果关系本身不带时间戳。
- **模态**: 纯文本（text-only）。环境（ScienceWorld/ALFWorld）动作与观测均为自然语言，无视觉/截图/视频/多模态记忆。
- **过度个性化/记忆安全风险**: 基本不适用/未涉及。CLIN 为 agent-centric 任务智能体，不存储用户隐私或个性化偏好，因此无过度个性化、谄媚或隐私治理问题；其相关‘记忆安全’风险体现为‘错误因果命题可能误导后续决策’，论文承认并依赖迭代修正，但未做 OP-Bench/Causal-LoCoMo 式系统性负面评测。
- **冲突/矛盾处理**: 隐式处理：每轮由记忆生成器整体重写记忆清单，矛盾/过时的因果命题可在重生成时被丢弃或把置信度从‘should’降为‘may’（或反之）、或转为‘does not contribute’，从而消解冲突；但无显式的冲突检测算子或 UPDATE/DELETE 操作（不同于 Memory-R1 / MEMTRACK 的显式冲突解决），且不保证矛盾被可靠识别。
- **token成本/延迟证据**: 无量化的 token/延迟/成本节省证据。论文报告的是‘后期 trial 平均步数下降（更快完成任务、更高样本效率）’作为效率体现，但未给出与全上下文或其他记忆层对比的 p95 延迟或输入 token 削减百分比等数字。

**其他信息 / Other**

- **cluster**: A. 反思与失败驱动 (Reflection & failure-driven)

**不确定字段 / Uncertain**

- 发表venue (`venue`)


<a id="a4-autoguide自动生成与选择状态感知情境感知指南的框架论文标题-v1-用-state-aware-guidelinesneurips-2024-正式版与-arxiv-v2-改为-context-aware-guidelines二者指同一方法"></a>

### A4 AutoGuide

*AutoGuide（自动生成与选择状态感知/情境感知指南的框架；论文标题 v1 用 State-Aware Guidelines，NeurIPS 2024 正式版与 arXiv v2 改为 Context-Aware Guidelines，二者指同一方法）*


**基本信息 / Provenance**

- **年份**: 2024（arXiv 预印本 2403.08978，首次公开于 2024 年 3 月 13 日，v2 修订）
- **作者/机构**: Yao Fu（傅尧，共同一作）、Dong-Ki Kim（共同一作）、Jaekyeom Kim、Sungryull Sohn、Lajanugen Logeswaran、Kyunghoon Bae、Honglak Lee（资深作者）；隶属美国密歇根大学（University of Michigan）与 LG AI Research（LG AI 研究院），两位一作分属密歇根大学与 LG AI Research。
- **发表venue**: NeurIPS 2024（第 38 届神经信息处理系统大会，Poster 海报；论文集 DOI 10.52202/079017-3811；OpenReview id mRIQz8Zd6O）。
- **论文链接**: https://arxiv.org/abs/2403.08978（NeurIPS 论文页 https://proceedings.neurips.cc/paper_files/paper/2024/hash/d8efbb5dd415974eb095c3f06bff1f48-Abstract-Conference.html）
- **引用数**: 约 27 次（Semantic Scholar 实时数据，截至 2026 年 6 月；CorpusId 268385171，属中等影响力工作）。

**记忆分类 / Taxonomy**

- **记忆类型**: 以语义记忆（semantic memory）为主：从离线经验中抽象出的‘情境→指南’（context→guideline）自然语言知识，本质是领域知识/启发式规则，对应 CoALA 的语义记忆；不保存原始轨迹作为情景记忆示例（这点区别于其基线 ExpeL）；底层规划基于 ReAct/SoM 的工作记忆（上下文窗口）。不显式建模程序性记忆。
- **记忆结构**: 非参数化的字典型知识库：以‘情境’（context，对智能体当前状态的简洁自然语言描述）为键（key）、对应的若干条‘情境感知指南’（context-aware guideline，条件式自然语言陈述）为值（value）组织成指南字典 G。每条指南为‘当处于某情境时，应/不应执行某动作’的条件结构。整体为人类可读的纯文本结构。
- **存储后端**: 外部文本知识库（指南字典 G，以情境为键的字典结构存放于上下文/文件中）。检索时以情境作为键直接匹配（key lookup），候选过多时再由 LLM 做 top-k 选择，未使用向量数据库/嵌入相似度检索（与 ExpeL 的 Faiss kNN 不同）。不写入模型参数。
- **持久化**: 外部持久化（durable external store）：指南字典在离线阶段从训练轨迹中构建并保存，推理阶段被检索注入提示；不依赖模型参数（底座 LLM 为冻结的 GPT-3.5/4-turbo、GPT-4V 闭源 API），也非纯临时上下文记忆。

**核心机制 / Mechanisms**

- **写入/编码**: 采用 summarized insight（抽象指南）编码，而非保存原始轨迹原文。写入流程依赖一对对比轨迹（contrastive trajectory pair）：给定同一训练任务 i 的成功轨迹 τ⁺ 与失败/低回报轨迹 τ⁻（满足回报 R(τ⁺)>R(τ⁻)），(1) 先定位两条轨迹开始‘分叉’（deviation）的时间步 t；(2) 情境识别模块 M_context 把分叉前的共享部分轨迹 τ_:t 抽象为一句简洁的情境描述（如‘在 Reddit 主页上’）；(3) 指南抽取模块 M_guideline 对比 τ⁺ 与 τ⁻ 在该情境下的不同动作，生成一条对应该情境的条件式指南（如‘当在 Reddit 主页、想进入某个论坛时，应点击位于 Wiki 链接正上方的 Forums 链接’）。所有指南整理进以情境为键的字典 G；为减少冗余，再用 LLM 判断新情境是否与已有情境等价，等价则复用、否则新增。整个写入过程无梯度更新，由 GPT-4-turbo 执行抽取。
- **检索机制**: 测试时分两步基于情境进行键匹配 + LLM 选择，而非嵌入相似度打分：(1) 每个时间步用情境识别模块 M_context 把当前测试轨迹 τ_:t 抽象为当前情境 context；(2) 以 context 作为键在指南字典 G 中取出候选指南集合 G[context]；(3) 若候选数超过 k，则用指南选择模块 M_select 提示 LLM 针对当前轨迹挑选 top-k 条最相关指南（公式 3：relevant_guidelines ← M_select(context, τ; G, k)），否则全取；若当前情境不匹配任何已有键则不注入指南（guidelines=∅）。被选指南连同情境一起拼入动作生成提示，再由策略 π 生成动作。无 recency/importance 复合打分公式，检索核心是‘情境作键 + 上下文条件式 top-k 选择’。消融显示 k=3 在 WebShop 上最优（SR 47%），k=0/1/2/3/5 分别为 30%/42%/46%/47%/43%（k 过大致 LLM 过度思考、k 过小致遗漏）。
- **反思/巩固**: 核心机制即一种离线、跨任务的 raw→insight 抽象/反思：通过对比成功 vs 失败轨迹（contrastive reflection），在它们行为分叉的关键时间步上归纳出‘在该情境下应采取/避免何种动作’的高层指南，把分散在多条离线轨迹中的隐性领域知识压缩为简洁可读的条件式自然语言。触发时机为离线构建阶段（非推理时在线反思），在遍历所有轨迹对时逐对触发情境识别 + 指南抽取，并通过 LLM 情境匹配做去重合并。作者强调这是‘跨任务（inter-task）’知识，与 Reflexion 的‘任务内（intra-task）’即时反思正交互补（实验证明二者可叠加）。
- **遗忘/更新**: 几乎无显式遗忘/衰减机制；更新主要体现在构建期的‘情境去重合并’：用 LLM 判断新生成情境是否与既有情境描述同义，若同义则归并到同一键下、避免冗余键，否则新增。指南字典在构建后基本固定，无 Ebbinghaus 时间衰减、无重要度计数的增删（区别于 ExpeL 的 ADD/EDIT/UPVOTE/DOWNVOTE）、无显式删除/失效算子。
- **经验回放 (核心主题)**: 核心主题之一：把离线积累的成功/失败轨迹作为可复用经验来改进未来行为，但复用形式是‘蒸馏成条件式指南后按情境检索注入’，而非原样回放轨迹。具体地，每对成功 vs 失败轨迹被对比蒸馏为一条情境化指南存入字典；测试时按当前情境检索 top-k 条相关指南拼入提示指导动作选择。相对直接把原始轨迹作为 few-shot 示例（如 ReAct n-shot、ExpeL 检索轨迹），AutoGuide 把经验压缩为更紧凑、上下文级、条件可用的知识，规避了上下文长度与提示敏感问题（ReAct 增到 6-shot 即逼近 GPT-3.5 token 上限且性能饱和：1/2/4/6-shot 奖励 66.4/66.0/70.2/71.0、SR 30%/35%/37%/38%，均低于 AutoGuide 73.4/46%）。还展示了把 WebShop 指南迁移复用到 WebArena-Shopping 的跨域经验复用。

**学习维度 / Learning**

- **学习范式**: 纯非参数化（non-parametric，prompt/in-context 层面）。完全不更新 LLM 参数，仅通过外部自然语言指南知识库 + 提示工程实现‘学习’，兼容 GPT-3.5-turbo、GPT-4-turbo、GPT-4V 等闭源 API 模型。
- **失败学习 (核心主题)**: 核心主题之一：失败被显式、对比式地利用。方法的基石是成对的成功轨迹 τ⁺ 与失败/低回报轨迹 τ⁻ —— 通过定位二者动作开始分叉的时间步、对比该处的有效与无效动作来抽取‘在某情境下应避免何种错误、应采取何种正确动作’的指南。论文图 4 案例展示：ReAct 因常见错误（取不可见的 soapbar、或因名称相近误取 soapbottle）失败，AutoGuide 从离线经验中类似错误抽取出的指南帮助避免这些失败。失败知识因此以条件式告诫/纠正规则沉淀进指南库。离线轨迹通过运行 ReAct/Reflexion 或人类示范收集，天然含成功与失败两类样本以构成对比对。
- **技能/程序归纳**: 部分支持：抽取出的‘情境→条件式指南’可视为可复用的、情境触发的启发式操作规则（如‘当在 Reddit 主页要进入论坛时点击 Forums 链接’），在推理时按情境检索注入提示被复用；但指南是软性自然语言提示而非像 Voyager 那样固化为可调用的代码/技能函数库，亦无显式的多步工作流封装。
- **在线 vs 离线**: 以离线（offline）为主：在训练任务轨迹集合上批量对比抽取指南，构成固定的指南字典；随后在不重叠的测试任务上应用（推理时仅做情境识别 + 指南选择，不再扩充指南）。可选地与在线自反思方法 Reflexion 结合，引入推理时的任务内反馈（Q2 实验），但指南本身的构建是离线的。

**评测 / Evaluation**

- **任务领域**: 交互式序列决策任务，覆盖具身/网页/多模态三类：具身家务模拟（ALFWorld）、网页购物导航（WebShop）、真实复杂网站导航（WebArena，主实验聚焦 Reddit 域）、以及真实多模态网站任务（GitHub 协作开发、Google Flights 机票搜索、Coursera 在线教育，需同时利用图像与 HTML 文本）。
- **基准**: ALFWorld（具身家务）、WebShop（在线购物）、WebArena（主实验为 Reddit 域，另含 WebArena-Shopping 做跨域泛化，泛化集为 98 个含商品意图的任务）、以及自建的 3 个真实多模态网站任务集（GitHub 30 题、Google Flights 20 题、Coursera 20 题）。
- **报告增益**: 在三大文本基准上一致大幅领先（成功率 SR / WebShop 另报奖励 Reward；ALFWorld、WebShop 底座为 GPT-3.5-turbo，WebArena 底座为 GPT-4-turbo）。表 1 关键数字：ALFWorld SR —— ReAct 54.5% / ExpeL 59.0% / AutoGuide 79.1%（较 ReAct 约 +24.6 个百分点，较 ExpeL 约 +20）；WebShop Reward —— ReAct 66.4 / ExpeL 60.9 / AutoGuide 73.4，SR —— 30% / 35% / 46%；WebArena-Reddit SR —— ReAct 8.0% / ExpeL 21.8% / AutoGuide 47.1%（较 ReAct 约 +39，较 ExpeL 约 +25 个百分点）。与 Reflexion 结合（最多 3 次试错）进一步提升：ALFWorld SR ReAct+Reflexion 67.2% / ExpeL+Reflexion 71.6% / AutoGuide+Reflexion 88.1%；WebShop Reward 77.1/71.7/81.4、SR 51%/42%/57%（WebArena 因 token 上限未做 Reflexion）。真实多模态（表 2，底座 GPT-4V + SoM）：SoM 基线 vs AutoGuide —— GitHub 2/30→19/30、Flights 5/20→9/20、Coursera 1/20→14/20。跨域泛化（表 5，WebShop 指南迁移到 WebArena-Shopping）：ReAct 10.2% → AutoGuide 20.4%（约翻倍）。组件消融（表 6，WebShop SR）：ReAct 30% → +仅情境识别 CI 36% → +仅指南抽取选择 GES 37% → 完整 AutoGuide(CI+GES) 46%，证明情境与指南二者协同。
- **对比基线**: 无离线经验/无情境的规划基线 ReAct；利用离线经验但‘非情境感知、无过滤地注入全部指南’的 ExpeL（主对照，作者仅取其指南生成部分）；任务内自反馈基线 Reflexion（converts 环境反馈为文本辅助下一次试错，可与本方法叠加）；多模态实验用 Set-of-Marks（SoM）智能体作底座基线；分析实验另对比不同 shot 数（1/2/4/6-shot）的 ReAct 在 WebShop 上的表现。

**分析 / Analysis**

- **关键创新**: 提出‘情境感知指南’（context-aware / state-aware guideline）这一条件结构知识表示：不像 ExpeL 把所有抽取的指南无差别塞进提示，而是为每条指南显式生成‘适用情境’，构建期以情境为键组织指南字典、测试期先识别当前情境再做 top-k 指南选择，从而只向智能体提供与当前决策状态切实相关的知识，避免无关指南干扰推理。这是把离线对比经验在‘正确的时间、提供正确的知识’的核心贡献。
- **局限**: (1) 指南字典在构建后基本静态，缺乏真正的遗忘/失效/重要度更新机制，难以应对环境漂移或错误指南的长期累积；(2) 高度依赖强 LLM（指南抽取用 GPT-4-turbo，动作/情境/选择共用底座 GPT 系列），成本与对闭源模型的依赖较高；(3) 需要成对的成功 vs 失败对比轨迹与可比较的回报信号，依赖离线数据质量与对比对的可得性；(4) WebArena 仅在 Reddit 单域做主实验，环境/任务规模相对有限；(5) 多模态与跨域泛化虽有验证但样本量小（如多模态各 20–30 题、跨域 98 题），统计强度有限；(6) 缺乏理论保证，性能受 LLM 情境识别/选择质量与提示敏感性影响；(7) 官方代码未公开，复现门槛较高。
- **与其他工作关系**: 属于本研究 A 类‘反思与失败驱动（Reflection & failure-driven）’簇。最直接的对照与改进对象是 A5 ExpeL（Zhao et al. 2023）：二者都从离线经验抽取自然语言知识，但 ExpeL 把全部指南无差别注入（non-contextual）且还检索成功轨迹作示例，AutoGuide 关键创新在于为指南附加‘适用情境’并在测试期做情境匹配 + top-k 选择，实验证明该情境化显著优于 ExpeL（如 WebArena 47.1% vs 21.8%）。与 A1 Reflexion（Shinn et al. 2023）正交互补：Reflexion 提供任务内（intra-task）即时反思，AutoGuide 提供跨任务（inter-task）离线指南，二者可叠加（AutoGuide+Reflexion 取得各域最佳）。底层规划复用 ReAct（Yao et al. 2023），多模态实验复用 Set-of-Marks（SoM）。其‘把经验蒸馏为可复用指南/规则’的思路与 AutoManual、Agent Workflow Memory（C 类）及 ReasoningBank（A6）同属 agent-centric 经验记忆脉络，但 AutoGuide 强调‘情境条件化检索’这一独特维度。
- **可复现性**: 可复现性中等偏弱：方法描述、算法（Algorithm 1/2）、提示模板（附录 C.1–C.4）较完整，所用 ALFWorld/WebShop/WebArena 均为公开基准；但未发现作者公开的官方代码仓库，且依赖 GPT-3.5-turbo/GPT-4-turbo/GPT-4V 等已演进的闭源模型快照，精确复现受模型版本变化与提示敏感性影响。多模态任务为自建、需自行搭建真实网站环境。社区采用信号较弱（约 27 次引用，无显著开源生态）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否。AutoGuide 采用启发式流水线管理‘记忆’（指南）：指南的抽取靠 LLM 对比轨迹生成、去重靠 LLM 情境匹配、检索靠‘情境作键 + LLM top-k 选择’，全程无 RL/训练去学习‘何时存/取/更新’的记忆管理策略。属于 2025–26 代际划分中的‘启发式（pre-learned-control）’一侧，是 Memory-R1、Mem-α 等可学习记忆控制工作的早期参照。
- **记忆主体**: 智能体中心（agent-centric）：记忆的是智能体在训练任务上对比成功/失败经验抽象出的领域知识/操作指南，目的是自我提升任务决策能力，而非记住用户信息做个性化（区别于 Mem0/Zep 等 user-centric 系统）。
- **多智能体记忆**: 单智能体（single-agent）。指南字典服务于单个 LLM 智能体的决策，无多智能体共享/路由记忆机制（不涉及 G-Memory/MIRIX 式的跨智能体记忆分层与路由）。
- **时序推理支持**: 否。不显式建模时间有效性、事件顺序或事实有效期窗口（无 Zep/Graphiti 式时间维度）；指南以情境为键组织，不带时间戳语义。其‘情境’刻画的是空间/界面状态而非时间关系。
- **模态**: 文本与多模态（text + multimodal）：主实验为纯文本（ALFWorld/WebShop/WebArena），并专门在真实多模态网站（GitHub/Flights/Coursera）上以 GPT-4V + Set-of-Marks 验证从含图像与文本观测的轨迹中生成情境感知指南的能力，是其相对纯文本前作（如 ExpeL）的扩展。
- **冲突/矛盾处理**: 弱：构建期仅通过 LLM 情境匹配把同义情境归并到同一键下以减少冗余，并未显式检测或解决相互矛盾的指南；同一情境键下可保留多条指南，测试期靠 LLM top-k 选择隐式回避不相关/相冲突项，但无版本化、无事实级冲突仲裁机制（不及 Memory-R1 的 UPDATE 或 MEMTRACK 精细）。

**不确定字段 / Uncertain**

- 代码链接 (`code_url`)
- 过度个性化/记忆安全风险 (`over_personalization_risk`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="a5-expelexperiential-learning-agent经验学习智能体"></a>

### A5 ExpeL

*ExpeL（Experiential Learning agent，经验学习智能体）*


**基本信息 / Provenance**

- **年份**: 2023（arXiv 预印本 2023 年 8 月 20 日，v3 于 2024 年 12 月）
- **作者/机构**: Andrew Zhao（赵启晨，通讯第一作者）、Daniel Huang、Quentin Xu、Matthieu Lin、Yong-Jin Liu（刘永进）、Gao Huang（黄高，通讯作者）；隶属清华大学自动化系与计算机科学系（BNRist，LeapLab 实验室）。
- **发表venue**: AAAI 2024（第 38 届人工智能大会，Oral 口头报告；论文集页码 19632–19642，DOI 10.1609/aaai.v38i17.29936）。
- **论文链接**: https://arxiv.org/abs/2308.10144（OJS 正式版 https://ojs.aaai.org/index.php/AAAI/article/view/29936）
- **代码链接**: https://github.com/LeapLabTHU/ExpeL（官方实现，Apache-2.0 许可，约 219 stars / 26 forks；项目主页 https://andrewzh112.github.io/expel）
- **引用数**: 约 596 次（Semantic Scholar 实时数据，截至 2026 年 6 月；属高影响力工作）。

**记忆分类 / Taxonomy**

- **记忆类型**: 以情景记忆（episodic memory，保存成功轨迹作为可检索示例）与语义记忆（semantic memory，从经验抽象出的自然语言 insight/规则）为主；底层规划基于 ReAct 的工作记忆（上下文窗口）。不显式建模程序性记忆。
- **记忆结构**: 双层非参数化记忆：(1) 经验池（experience pool），保存训练阶段收集的成功/失败轨迹原文，以向量存储索引；(2) 一个动态维护的自然语言 insight 列表（带重要度计数）。两者均为文本形式、人类可读。
- **存储后端**: 经验池使用 Faiss 向量存储 + kNN 检索器 + all-mpnet-base-v2 句向量编码器（基于 LangChain 实现）；insight 列表以纯文本/列表形式保存在文件/上下文中。均为外部存储，不写入模型参数。
- **持久化**: 外部持久化（durable external store）：经验池与 insight 列表在离线训练阶段构建并落盘保存，推理阶段被检索调用；不依赖模型参数（parametric weights 保持冻结，兼容 GPT-4、Claude 等闭源 API 模型），也不属于纯上下文临时记忆。

**核心机制 / Mechanisms**

- **写入/编码**: 采用 verbatim（原文轨迹）+ summarized insight（抽象洞见）两种编码并存。写入分两条路径：(1) 经验收集阶段，智能体用 ReAct 在训练任务上 trial-and-error，并以 Reflexion 对失败任务自我反思后重试，最终把成功与失败的完整轨迹原文存入经验池；(2) insight 提取阶段，用 LLMinsights（默认 gpt-4-0613）读取‘失败/成功配对’或‘L 条成功轨迹列表’（无放回采样），将经验抽象/蒸馏为自然语言洞见。整个写入过程无梯度更新、无需大量数据或人工标注。
- **检索机制**: 读取分两部分：(1) 相似经验检索（experience recall）——用 all-mpnet-base-v2 对评测任务描述编码，在 Faiss 经验池中用 kNN 按‘任务相似度’（最大内积 inner-product task similarity）检索 top-k 条成功轨迹作为 in-context few-shot 示例（k 随环境取 2–6）；论文消融显示按任务相似度检索优于按推理相似度（reasoning similarity，ALFWorld 59.0% vs 48.5%）和随机采样（42.5%）。(2) insight 全量注入——抽取出的洞见列表直接拼入任务说明（task specification）部分。无 recency/importance 复合打分公式（与 Generative Agents 不同），检索 rank 即单纯的任务相似度。
- **反思/巩固**: 这是 ExpeL 的核心机制。两阶段反思与抽象：(1) 收集阶段借助 Reflexion 对失败轨迹生成自然语言反思并重试（最多 3 次），以获得更多成功/失败样本对；(2) insight 提取阶段，LLMinsights 通过四种算子动态维护洞见列表——ADD（新增洞见，初始重要度计数=2）、EDIT（修改并 +1）、UPVOTE（赞同 +1）、DOWNVOTE（反对 −1，计数归零即删除）。它通过两种比较抽象知识：对同一任务比较失败 vs 成功轨迹（定位错误动作），以及在跨任务的多条成功轨迹中归纳‘好实践’（best practices）。该 raw→insight 的跨任务知识抽象在离线训练阶段触发，是其区别于 Reflexion（仅任务内反思、无跨任务记忆）的关键。
- **遗忘/更新**: 通过 insight 的重要度计数实现轻量更新/遗忘：DOWNVOTE 使计数递减，计数为 0 时该洞见被移除；EDIT 可合并/改写已有洞见。此设计可抑制次优成功轨迹误导出的错误洞见。经验池本身只增不删（无衰减/去重机制），无 Ebbinghaus 式时间衰减。
- **经验回放 (核心主题)**: 这是其核心主题之一。ExpeL 把训练阶段自生成的成功轨迹当作可复用经验，在推理时按任务相似度检索 top-k 条作为 in-context 示例（论文将此明确类比为强化学习中的 experience replay 与 off-policy 学习：用 Reflexion 作行为策略收集经验，用 insight 提取 + 相似任务检索作策略改进）。同时跨任务抽象出的 insight 也是经验的蒸馏复用。消融证明‘检索成功轨迹’与‘insight 抽象’二者协同、缺一不可（retrieve-only 与 insights-only 均逊于完整 ExpeL）。还展示了 source→target 任务的迁移学习（HotpotQA insight 迁移到 FEVER）。

**学习维度 / Learning**

- **学习范式**: 纯非参数化（non-parametric，prompt/in-context 层面）。完全不更新 LLM 参数，仅通过外部记忆（经验池 + insight 列表）与提示工程实现‘学习’，因而兼容 GPT-4、Claude 等仅 API 可用的闭源模型。
- **失败学习 (核心主题)**: 这是其核心主题之一。失败被显式利用：收集阶段用 Reflexion 对失败任务自我反思并重试，从而主动产生失败/成功配对；insight 提取阶段，LLM 把同一任务的失败轨迹与成功轨迹并排比较，定位‘正确与错误动作’，并被特别要求‘提取常见失败模式（prevalent failure patterns）’与最佳实践。消融显示：用 Reflexion 收集的多样化成功/失败数据优于仅用 ReAct 收集，证明失败样本的多样性对最终性能至关重要。失败知识以负面规则/告诫形式沉淀进 insight 列表（如‘若某尝试未推进任务则重新评估并考虑替代动作’）。
- **技能/程序归纳**: 部分支持：抽象出的自然语言 insight 可视为可复用的‘启发式规则/好实践’（如‘搜索物品时应考虑其性质与典型用途’），在推理时注入任务说明被复用；但不像 Voyager 那样把技能固化为可调用的代码/函数库，洞见仍是软性提示而非结构化可执行程序。
- **在线 vs 离线**: 以离线（offline）为主：在训练任务集合上批量收集经验并提取 insight，构成固定的记忆，随后在评测任务上单次尝试（类比‘学生备考后一次性考试’）。部署时默认不再依赖重试/环境反馈（这是其相对 Reflexion 的优势），但 5.5 节也展示了可选地与 Reflexion 结合做在线任务重试。

**评测 / Evaluation**

- **任务领域**: 文本型交互式决策任务三大类：知识密集型问答与推理（HotpotQA，Wikipedia Docstore 检索）、具身/家务模拟（ALFWorld 家庭环境）、网页购物导航（WebShop 在线购物）；并用事实核查（FEVER）做迁移学习目标域。均为纯文本观测（无视觉）。
- **基准**: HotpotQA（distractor dev split，100 个验证任务）、ALFWorld（134 个可解任务）、WebShop（100 个任务）、FEVER（迁移学习目标域）。均沿用 ReAct/Reflexion 所用同一任务子集，采用四折验证（four-fold validation），报告均值与标准误。
- **报告增益**: 在与基线 ReAct/Act 的对比中各域均一致提升（均用 gpt-3.5-turbo-0613 执行动作、温度 0 贪心解码）。消融表关键数字：HotpotQA 成功率 ReAct 28.0%±1.4 → ExpeL 39.0%±1.7（约 +11 个百分点）；ALFWorld 成功率 ReAct 40.0%±0.3 → ExpeL 59.0%±0.3（约 +19 个百分点）；WebShop 平均奖励 ReAct 0.665 → ExpeL 0.701（IL 0.599）。受限模式对比（仅一种学习方式）显示协同性：HotpotQA insights-only/retrieve-only 为 36%/31%，ALFWorld 为 50%/55%，WebShop 成功率 37%/38%、奖励 0.675/0.67，均低于完整 ExpeL。迁移学习：FEVER 成功率 Act 58.0% / ReAct 63.0% / ExpeL Transfer(无任务示例) 65.0% / ExpeL Transfer 70.0%。与 Reflexion 结合（ALFWorld，R0→R3）：ExpeL+Reflexion 59.0%→64.2%，优于 ReAct+Reflexion 40.3%→54.4%。
- **对比基线**: 主要对比无记忆/无跨任务学习的规划基线：ReAct、Act（无推理步骤）、模仿学习（IL，数据取自 ReAct 论文）；自身消融基线：ExpeL(insights-only)、ExpeL(retrieve-only)、人工编写 insight、随机采样示例、按推理相似度检索、gpt-3.5 提取 insight；以及自我改进基线 Reflexion / ReAct+Reflexion。

**分析 / Analysis**

- **关键创新**: 首个无需梯度更新、跨任务的‘经验学习’LLM 智能体范式：通过 ADD/EDIT/UPVOTE/DOWNVOTE 算子从成功/失败经验中自主抽象出自然语言 insight，并按任务相似度检索自生成的成功轨迹作为示例——把 Reflexion 的任务内反思扩展为跨任务（inter-task）的非参数化学习，兼容闭源 API 模型。
- **局限**: 作者明确列出：(1) 仅限文本观测，缺乏图像/多模态能力；(2) 主要基于闭源 API LLM，某些应用不可用；(3) 当 insight 列表随终身学习增长可能超出上下文窗口，需额外对 insight 也做检索；(4) 提示式方法缺乏理论保证（相对 RL），可能影响策略最优性；(5) 仅在确定性环境评测，规模较小（每域约 100–134 任务）；(6) 经验池只增不删、无真正遗忘机制。
- **与其他工作关系**: 属于 A 类‘反思与失败驱动’簇。直接构建于 A1 Reflexion（Shinn et al. 2023）之上：复用 Reflexion 的失败自反思来收集多样经验，但关键扩展为跨任务记忆与 insight 抽象（Reflexion 仅任务内、无跨任务记忆且部署时需重试）。底层规划用 ReAct（Yao et al. 2023）。检索成功轨迹的思路借鉴 RAG 与 RL 的 experience replay，但检索对象是智能体自生成经验而非黄金示例。记忆‘按相关性检索’的理念与 Generative Agents（Park et al. 2023）相关，但 ExpeL 面向任务求解而非开放式模拟，且未采用 recency/importance 复合打分。技能学习方向与 Voyager（Wang et al. 2023）相关但不固化为代码技能。它是后续‘智能体经验记忆/ReasoningBank 类’工作的早期代表（agent-centric、非参数化）。
- **可复现性**: 可复现性较好：官方代码开源（Apache-2.0，含 train/insight_extraction/eval 三阶段脚本与配置），沿用 ReAct/Reflexion 的公开任务子集与提示，超参数在附录完整列出；但需自行安装 ALFWorld、WebShop（需本地起服务器）等环境，且依赖 OpenAI gpt-3.5/gpt-4-0613 等已演进的闭源模型快照，精确复现可能受模型版本变化影响。社区采用信号中等（约 219 stars）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否。ExpeL 采用启发式流水线管理记忆：insight 的增删改靠 LLM 调用四个算子 + 重要度计数规则，检索靠固定的任务相似度 kNN——均无 RL/训练去学习‘何时存/取/更新’的记忆管理策略。属于 2025–26 代际划分中的‘启发式（pre-learned-control）’一侧，是 Memory-R1/Mem-α 等可学习记忆控制工作的前身参照。
- **记忆主体**: 智能体中心（agent-centric）：记忆的是智能体自身在训练任务上的经验（成功/失败轨迹与抽象洞见），目的是自我改进任务求解能力，而非记住用户信息做个性化（区别于 Mem0/Zep 等 user-centric 系统）。
- **多智能体记忆**: 单智能体（single-agent）。无多智能体共享/路由记忆机制；论文相关工作讨论了 Generative Agents 等多智能体记忆，但 ExpeL 本身为单体经验记忆。
- **时序推理支持**: 否。不显式建模时间有效性、事件顺序或事实有效期窗口（无 Zep/Graphiti 式时间维度）；经验池与 insight 不带时间戳语义，仅用重要度计数排序。
- **模态**: 纯文本（text-only）。论文明确将缺乏图像/多模态观测列为局限，并提出用 VLM/captioning 扩展为未来工作。
- **冲突/矛盾处理**: 通过 insight 提取算子隐式处理冲突：DOWNVOTE 降低与现有洞见相悖的条目计数（归零即删除），EDIT 可改写/合并矛盾洞见，从而在更新时缓解次优或相互矛盾的经验误导。但无显式的事实级冲突检测/版本化机制（不及 Memory-R1 的 UPDATE 或 MEMTRACK 精细）。

**不确定字段 / Uncertain**

- 过度个性化/记忆安全风险 (`over_personalization_risk`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="a6-reasoningbank推理记忆库配套提出-matts-记忆感知的测试时扩展"></a>

### A6 ReasoningBank

*ReasoningBank（推理记忆库；配套提出 MaTTS 记忆感知的测试时扩展）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本，2025-09-29 首次公开）
- **作者/机构**: Siru Ouyang、Jun Yan、I-Hung Hsu、Chen-Yu Lee、Tomas Pfister 等（共 17 位作者）；主要单位为 Google Cloud AI Research，合作单位包括伊利诺伊大学厄巴纳-香槟分校（UIUC，Jiawei Han 课题组）与耶鲁大学。第一作者 Siru Ouyang 来自 UIUC（在 Google 实习期间完成）。
- **论文链接**: https://arxiv.org/abs/2509.25140

**记忆分类 / Taxonomy**

- **记忆类型**: 以程序性记忆（procedural）为主，存储可迁移的高层推理策略/启发式（reasoning strategies & hints）；同时具备情景性记忆（episodic）属性，因为每条记忆都从具体成功或失败的交互轨迹中蒸馏而来。本质属于 CoALA 框架中的程序性+情景性记忆混合，但抽象层级高于原始轨迹。
- **记忆结构**: 结构化的「记忆条目（memory item）」集合，每条包含三个字段：title（策略标题/标识）、description（一句话摘要）、content（蒸馏出的推理步骤、决策依据或操作要点）。整体为扁平的条目池（pool），通过 embedding 索引检索，刻意保持简单（无层级、无图结构）。
- **存储后端**: 外部化的记忆条目库，配合基于嵌入向量的相似度检索（embedding-based similarity search）；检索到的条目在运行时被注入到智能体的 system instruction（上下文）中。论文未指定具体向量数据库实现，强调流程从简以凸显记忆内容本身的贡献。
- **持久化**: 外部持久化存储（durable external store）。记忆库在测试时学习（test-time learning）的任务流中持续累积、跨任务保留；不修改模型参数（非参数化），检索结果以临时上下文形式注入。

**核心机制 / Mechanisms**

- **写入/编码**: 对原始轨迹进行蒸馏式编码（distillation），而非逐字保存。任务完成后进入「记忆构建（memory construction）」阶段：先用 LLM-as-a-judge（无需 ground-truth 标签）将轨迹判为成功或失败；成功轨迹蒸馏出经过验证的有效策略，失败轨迹蒸馏出反事实信号与「避坑」教训（guardrails）。每条轨迹/经验可抽取多条记忆条目（含 title/description/content 三字段），抽象掉低层执行细节、保留可迁移的推理模式。Web 浏览任务用文本可访问性树（accessibility tree）作为观察、用 LLM 思考过程近似观察历史；SWE 任务用代码片段。
- **检索机制**: 面对新任务时，用当前查询上下文对记忆库做基于嵌入向量的相似度检索（embedding-based similarity search），取 top-k 相关经验及其对应记忆条目。检索到的条目作为附加 system instruction 注入策略模型 π_L，使其决策被既往有效洞见与失败教训所约束。论文刻意采用简单的语义检索（无重排序、无层级、无学习型路由），以隔离记忆内容质量的影响；附录 E 指出可升级为自适应/层级化检索或推理密集型控制器作为未来方向。
- **反思/巩固**: 核心机制即「原始经验→高层洞见」的反思蒸馏。闭环三步：检索（retrieval）→构建（construction，从当前成功与失败轨迹蒸馏新条目）→巩固（consolidation，以简单的「添加（addition）」操作并入记忆库）。在 MaTTS 设定下进一步强化：并行扩展时通过跨多条轨迹的自对比（self-contrast）筛除虚假解、保留一致推理模式；顺序扩展时通过自精炼（self-refinement）将中间笔记/纠错信号也纳入记忆。论文报告记忆条目会随时间演化，从「执行式/过程式策略」逐步升级到「自反思」「自适应检查」直至「组合式策略」，呈现类 RL 的涌现学习动态。
- **遗忘/更新**: 巩固阶段仅做简单的「添加（addition）」操作，刻意不做复杂的更新/合并/删除/去重，亦无显式遗忘或衰减机制（论文承认这是为隔离内容质量贡献而做的简化，可作为未来增强方向）。
- **经验回放 (核心主题)**: 这是论文的核心主题。它不重放原始轨迹，而是把过去成功与失败经验蒸馏为可复用的「推理策略单元」并在新任务中通过检索复用，从而避免重复成功路径的再发现、并规避既往错误。相较于复用原始轨迹（Synapse）或仅复用成功工作流（AWM），ReasoningBank 复用的是更抽象、更可迁移的推理提示。配套的 MaTTS（记忆感知测试时扩展）进一步把「在同一任务上多次探索」产生的丰富成功/失败轨迹作为对比信号，反哺出更高质量、更具泛化性的记忆，形成「记忆↔扩展」的正反馈飞轮——作者称之为面向智能体的新「扩展维度（memory-driven experience scaling）」。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric）/ 提示层（prompt-level）的测试时学习（test-time learning）。不进行任何梯度更新，仅通过外部记忆库的累积与上下文注入实现自我进化；适用于查询以流式（streaming）到来、无 ground-truth、无未来信息可见的部署场景。
- **失败学习 (核心主题)**: 核心创新之一：显式从失败中学习。任务完成后由 LLM-as-a-judge 自判成功/失败（无标签），失败轨迹被蒸馏为反事实信号与「避坑」教训（preventative lessons / guardrails），与成功策略一同入库。消融实验（WebArena-Shopping，Gemini-2.5-flash）显示：仅用成功轨迹时 ReasoningBank 达 46.5% SR，纳入失败后升至 49.7%；而仅依赖成功轨迹的基线（Synapse 40.6→41.7，AWM 44.4→42.2，甚至下降）无法有效利用失败。说明 ReasoningBank 能把失败转化为建设性信号而非噪声。
- **技能/程序归纳**: 是。它从经验中归纳可复用的高层推理策略/启发式（以 title/description/content 三字段表示），通过检索注入 system instruction 来调用。这些策略并非固定工作流，而是随经验演化、可组合的推理单元；论文观察到它们从过程式动作规则逐步演化为自适应检查与组合式策略。
- **在线 vs 离线**: 在线（online）。记忆在部署/测试时按流式任务逐条增量构建与巩固（test-time learning，每完成一个任务即更新），不依赖离线批量训练语料。

**评测 / Evaluation**

- **任务领域**: 网页浏览/网页导航（web browsing）与软件工程（software engineering，仓库级 issue 修复）两大类智能体任务。
- **基准**: WebArena（5 个子域：Shopping/Admin/Gitlab/Reddit/Multi，共 684 任务，排除 Map 子域）、Mind2Web（cross-task / cross-website / cross-domain 泛化）、SWE-Bench-Verified（仓库级 issue 修复）。MaTTS 的扩展因子分析主要在 WebArena-Shopping 子集上以 Gemini-2.5-flash 进行。
- **报告增益**: WebArena 总体成功率（SR）相对 No Memory 基线提升：Gemini-2.5-flash +8.3（40.5→48.8）、Gemini-2.5-pro +7.2（46.7→53.9）、Claude-3.7-sonnet +4.6（41.7→46.3），且全面优于 Synapse/AWM。效率上交互步数最多减少约 16.0%（成功用例上减少更明显，最多减少 2.1 步，约 26.9% 相对减少）。SWE-Bench-Verified 解决率：flash 34.2→38.8（+4.6），pro 54.0→57.4（+3.4），步数分别减少 2.8 与 1.3。Mind2Web 在 cross-task/website/domain 各设定均提升（如 flash cross-task SR 3.3→4.8）。MaTTS（WebArena-Shopping，flash）：并行扩展 SR 从 k=1 的 49.7 升至 k=5 的 55.1，顺序扩展升至 54.5，均显著优于无记忆扩展（39.0~42.2）与 vanilla TTS（k=5 时 52.4/51.9）。摘要称效果上「最高 34.2% 相对提升」、效率上「16.0% 更少交互步数」。
- **对比基线**: No Memory（无记忆智能体）、Synapse（基于原始轨迹的记忆）、AWM / Agent Workflow Memory（基于成功工作流的记忆）；MaTTS 部分还对比 MaTTS w/o memory（无记忆扩展）与 MaTTS w/o aggregation（即 vanilla TTS）。骨干 LLM 为 Gemini-2.5-flash/pro 与 Claude-3.7-sonnet，环境为 BrowserGym（网页）与 bash-only（SWE），采用 ReAct 风格。

**分析 / Analysis**

- **关键创新**: 提出从智能体自判的成功「和」失败经验中蒸馏可迁移的「推理策略」作为记忆（超越仅存原始轨迹或仅存成功工作流的前作），并首创「记忆感知的测试时扩展（MaTTS）」，揭示记忆与测试时计算之间的协同飞轮，将「记忆驱动的经验扩展」确立为智能体的新扩展维度。
- **局限**: （1）聚焦记忆内容本身，未系统比较情景/层级等其他记忆架构（结构层面的设计被视为正交工作）；（2）刻意采用简单的嵌入检索与「仅添加」式巩固，缺乏自适应/层级检索、更新、去重与真正的遗忘机制；（3）成功/失败信号依赖 LLM-as-a-judge 自判，任务模糊或裁判出错时可能引入噪声（虽称框架对此较鲁棒，但建议引入更强验证器、人类反馈或集成裁判）；（4）随记忆库增长可能存在的扩展性/成本与冲突处理问题论文未深入。
- **与其他工作关系**: 属于「A. 反思与失败驱动（Reflection & failure-driven）」簇，与 A1 Reflexion 共享「从失败中反思」的理念，但 Reflexion 多停留在单任务自反思文本，ReasoningBank 则把成功+失败均蒸馏为跨任务可迁移的结构化策略库。相较 Synapse（复用原始轨迹）与 AWM/Agent Workflow Memory（仅从成功轨迹归纳工作流），它存储更抽象的推理提示并显式吸收失败教训。属于智能体中心（agent-centric）自我进化记忆，与 Voyager 的技能库理念同源但表示更偏推理策略；区别于以用户个性化为目标的 Mem0/Zep/LongMemEval 类用户中心记忆。MaTTS 将记忆与测试时扩展（TTS）结合，借鉴 self-contrast 与 self-refinement（Self-Refine）思想。论文亦指出可与更复杂的检索/巩固/学习型路由（如 RL 记忆管理）正交组合。
- **可复现性**: 可复现性较好：官方开源代码（github.com/google-research/reasoning-bank，Python，Apache-2.0，约 398 stars），所用基准均为公开数据集（WebArena、Mind2Web、SWE-Bench-Verified），骨干模型为商用 API（Gemini-2.5、Claude-3.7）。流程刻意从简，便于复现；社区关注度较高（短期内约 119 次引用）。但因依赖闭源商用 LLM 与 LLM-as-a-judge，结果对模型版本与裁判稳定性敏感。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式流程）。记忆的存/取/巩固均为固定的非学习型流程：嵌入相似度检索 + 简单添加式巩固，不使用 RL/训练来学习记忆管理策略本身。论文将此作为有意的简化，并把「学习型路由/巩固策略」列为未来方向，因此处于 2025-26「学习型记忆控制」分水岭中的启发式一侧（与 Memory-R1、Mem-α 等学习型方法相对）。
- **记忆主体**: 智能体中心（agent-centric）：记忆的是智能体自身的交互经验（成功/失败的推理策略），目的是让智能体自我进化、跨任务自我提升，而非记住用户信息做个性化。与 Voyager 同类，区别于 Mem0/Zep/LongMemEval 等用户中心记忆。
- **多智能体记忆**: 单智能体（single-agent）。记忆库服务于单个智能体的测试时学习，未涉及多智能体间共享/路由记忆；不过作者在讨论中设想可演化为「跨域、跨团队」的可部署记忆服务。
- **时序推理支持**: 否。不显式建模时间有效性、事件先后或事实时效窗口；记忆是与时间无关的推理策略，巩固也无时间衰减/刷新策略（论文提及可与带衰减/刷新的长期记忆策略兼容，但本身未实现）。
- **模态**: 纯文本（text-only）。Web 任务使用文本化的可访问性树（accessibility tree）作为观察，SWE 任务使用代码片段；无视觉/截图/具身或多模态记忆。
- **过度个性化/记忆安全风险**: 未涉及。论文不处理个性化，也未讨论有害/过时/侵入性记忆、隐私治理或过度个性化风险（这类是用户中心记忆的安全维度）；相关风险在本工作范围之外。
- **冲突/矛盾处理**: 基本未处理。巩固仅做简单添加，无显式的冲突/矛盾事实检测与合并机制。对相互矛盾或虚假的策略，主要依赖 MaTTS 并行扩展中的自对比（self-contrast）来筛除虚假/不一致解、保留一致推理模式，但这属于记忆「构建」阶段的过滤，而非入库后的冲突消解；作者承认更复杂的更新/巩固为未来方向。
- **token成本/延迟证据**: 以「交互步数（interaction steps）」作为效率代理而非 token/延迟。WebArena 上相对 No Memory 最多减少约 1.4 步、相对其他记忆基线最多减少 1.6 步，总体约 16.0% 更少步数；SWE-Bench-Verified 减少 2.8/1.3 步。分解分析显示成功用例上步数减少更显著（最多 2.1 步，约 26.9% 相对减少），表明效率提升来自更有目的的决策而非草率截断失败轨迹。论文未直接报告 token 成本或墙钟延迟的百分比节省。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)
- 代码链接 (`code_url`)
- 发表venue (`venue`)


<a id="a7-memento-2memento-ii--stateful-reflective-memory提出-stateful-reflective-decision-process-srdp-与-readwrite-reflective-learning"></a>

### A7 Memento 2

*Memento 2（Memento-II / Stateful Reflective Memory；提出 Stateful Reflective Decision Process (SRDP) 与 Read–Write Reflective Learning）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本，2025-12-27 首发；v3 于 2026-01 更新）
- **作者/机构**: Jun Wang（独著），英国伦敦大学学院 UCL 人工智能中心（UCL Centre for Artificial Intelligence，jun.wang@ucl.ac.uk）。延续其团队前作 Memento（Zhou et al. 2025）与基于案例推理的 LLM 智能体（Guo et al.）。
- **发表venue**: arXiv 预印本（cs.AI / cs.CV / cs.LG），截至调研未见正式会议/期刊收录。
- **论文链接**: https://arxiv.org/abs/2512.22716
- **代码链接**: 无官方代码仓库（本文为纯理论论文，未发布代码；其经验证据来自前作 Memento，代码见 https://github.com/Agent-on-the-Fly/Memento）

**记忆分类 / Taxonomy**

- **记忆类型**: 情景记忆（episodic memory）为核心——将每次交互记为案例 m=(s,a,r,s')；通过反思读写实现持续学习，亦统一了情景控制、案例推理与上下文学习；不涉及独立的语义/程序记忆模块。
- **记忆结构**: 外置情景记忆库（case bank / replay buffer），即随时间增长的有限案例多重集合 M={(s_i,a_i,r_i,s'_i)}；案例按状态嵌入 ψ(s) 组织，用 Parzen 窗（核密度）度量相似度，是非参数化的“增长型案例库”。
- **存储后端**: 外部非参数案例库 + 状态嵌入向量（ψ: S→R^d）做相似度检索；理论层面抽象为 Parzen 核密度估计上的检索分布，未规定具体向量数据库实现。未来工作展望将记忆内化进 LLM 架构（参考 Titans/MIRAS/EM-LLM）。
- **持久化**: 外置且持久（durable external store）：记忆跨回合、跨任务持续累积并随经验不断增长，参数（LLM 权重）保持不变；适应完全由外部记忆演化驱动，不做反向传播/微调。

**核心机制 / Mechanisms**

- **写入/编码**: 写操作 Write(M_t, s_t, a_t, r_t, s_{t+1}) 把交互结果（状态、动作、奖励、后继状态）作为案例存入情景记忆。理论上写=策略评估（policy evaluation）：通过执行当前策略产生轨迹 τ~π_t，对其结果做评估 V̂(π_{t+1})←Eval(τ_t) 并写回记忆供未来读取。多状态设置下奖励 r 可替换为估计值 Q，沿记忆做 Bellman/TD 传播。写不是简单 append，可承载结构化、状态相关的记忆更新。
- **检索机制**: 读操作=策略改进（policy improvement）：基于密度感知的检索。对查询状态 s 用 Parzen 窗给每个案例 c 赋相似度权重 w_parzen(s,c)=K((ψ(s)-ψ(s(c)))/h)/Σ_{c'} K(...)（公式5，K 为高斯核，带宽 h）。该 Parzen 权重作为检索先验 μ0(c|x)。引入“空案例 c_∅”：以常数核分 K_∅ 表示 LLM 不依赖记忆、直接用自身世界知识生成动作。先验写成混合形式 μ0=λ(x)·μ_mem + (1-λ(x))·δ_{c_∅}，当与已存案例相似度高(λ≈1)走检索驱动，相似度低(λ≈0)走知识驱动探索。检索动作 c_t~μ(·|s_t,M_t)，再由 LLM 核 p_LLM(a|s,c) 生成动作，构成复合策略 π^μ(a|s,M)=Σ_c μ(c|s,M)p_LLM(a|s,c)。检索策略经 KL（熵）正则化软策略迭代学习（Parzen-KL soft policy iteration）。
- **反思/巩固**: “反思（reflection）”是全文核心机制：定义为 LLM 智能体通过与情景记忆、内部推理、环境反馈协同交互来逐步改进有效策略的迭代过程，明确表述为“先 Read 后 Write”的闭环（算子 π_{t+1}=T(π_t)=Read(Write(π_t))）。统一三种广义反思形态：上下文学习、反馈驱动反思（如 ReAct）、链式思维内部推理。本文不是把原始经验摘要成更高层文本洞见，而是从控制论角度把反思形式化为策略迭代——Read 对应策略改进、Write 对应策略评估，每回合触发，形成持续自改进而无需重训。
- **遗忘/更新**: 记忆主要随经验单调增长（覆盖状态空间）；论文未提出显式遗忘/衰减或去重机制，而是通过带宽 h 随记忆增长自适应缩小（Silverman 规则 h∝n^{-1/(d+4)}、交叉验证或自适应策略）来在局部相似性与统计鲁棒性间平衡，从而抑制旧/远案例的影响。承认记忆变大会带来计算挑战。
- **经验回放 (核心主题)**: 经验复用是核心：把过去轨迹/案例存入增长型情景记忆，并通过相似度检索在新状态下复用相关案例为 LLM 提供“局部语义邻域”的上下文 grounding（而非像经典情景控制那样直接规定动作）。与深度 RL 的经验回放（DQN/优先回放按 TD 误差采样）不同，本文用密度感知的 Parzen 核检索做更细粒度、上下文敏感的复用，把案例推理(CBR)+检索增强生成(RAG)+经验回放统一进策略迭代框架，并证明随记忆增长复合策略收敛到最优。

**学习维度 / Learning**

- **学习范式**: 非参数化 / in-context（提示与记忆层面）持续学习——不更新 LLM 权重、不做反向传播；适应来自外置情景记忆的读写演化。理论上等价于在“反思 MDP”上的（软）策略迭代。展望中提出可与参数内化（test-time memorisation）结合形成混合架构。
- **技能/程序归纳**: 本文（理论论文）不直接归纳可复用技能/工作流；情景记忆存的是 (s,a,r,s') 案例而非显式技能。但其 SRDP/Read–Write 框架被同团队后续 Memento-Skills 用作把技能（结构化 markdown）作为持久演化记忆、做技能归纳与路由的理论基础。
- **在线 vs 离线**: 在线（online / 部署期、逐回合）持续学习为主：记忆随部署中的交互即时增长与读写，模糊了训练/测试分界（“学习可在部署时发生”）。理论也覆盖记忆相对策略缓慢演化的两时间尺度（two-time-scale）情形。

**评测 / Evaluation**

- **任务领域**: 本文为纯理论论文，无自有任务域实验；其经验依据来自前作所覆盖领域：深度研究（deep research，Memento）、自动数据科学（data-science agents）、软件测试（software-testing agents）等开放式长程任务。
- **报告增益**: 本文不报告新的定量基准结果——其“收益”是理论保证而非数值：(1) 定理8 在有界奖励且 γ<1、记忆平稳假设下，Parzen-KL 软策略迭代收敛到 KL 正则化目标（带 Parzen 先验）的最优不动点 (Q*,μ*)；(2) 定理10 给出记忆相对策略缓慢演化时的两时间尺度收敛；(3) 推论15（渐近最优）证明随记忆增长覆盖状态空间，复合策略与底层 MDP 最优策略的差距上界 sup_s|V^{π*}-V^{π_M}| ≤ (2R_max/(1-γ)^2)·Δ_M → 0，其中 Δ_M ≤ ε_LLM(r_M)+δ_M（LLM 近似误差 + 记忆覆盖误差）。经验支撑援引前作 Memento：GAIA 验证集 top-1 87.88% Pass@3、测试集 79.40%；DeepResearcher 数据集 66.6% F1 / 80.4% PM；案例记忆在分布外任务带来 +4.7%~+9.6% 绝对提升（这些数字来自 Memento 论文，非本文重做）。
- **对比基线**: 本文无实证对比基线；理论上把自身框架与若干范式作对照统一：案例推理(CBR，被视为无状态特例)、检索增强生成(RAG)、经典/最大熵强化学习、情景控制(MFEC/NEC)、RNN/LSTM 记忆 RL、优先经验回放等。

**分析 / Analysis**

- **关键创新**: 首次为“记忆驱动、无需微调”的反思式 LLM 智能体提供严格数学基础：提出 Stateful Reflective Decision Process (SRDP)，把情景记忆并入状态 x=(s,M) 恢复马尔可夫性（Reflected MDP），并证明 Read=策略改进、Write=策略评估，从而用 Parzen-KL 软策略迭代获得收敛与渐近最优保证，把 CBR/RAG/RL 统一为单一控制论框架。
- **局限**: (1) 纯理论，无自有实验验证；经验证据全部援引前作。(2) 记忆持续增长，规模增大带来计算/检索开销挑战，且无显式遗忘机制。(3) 性能高度依赖状态嵌入 ψ 的质量，论文承认未充分处理嵌入质量影响。(4) 收敛依赖较强假设（有界奖励、记忆平稳/两时间尺度、LLM 局部一致性 Assumption 11、检索能力 Lemma 12）。(5) 空案例核分 K_∅ 等超参需调。
- **与其他工作关系**: 属 A 类“反思与失败驱动”集群，是该作者团队工作的理论化升级：直接为前作 Memento（arXiv 2508.16153，Zhou et al. 2025，把范式形式化为 Memory-augmented MDP / M-MDP，含神经案例选择策略，GAIA 87.88% 等）与基于案例推理的 LLM 智能体（Guo et al.）补上收敛/最优性证明，把 M-MDP 推广为 SRDP/Reflected MDP。与 Reflexion 类语言反思不同：本文把“反思”从经验设计模式抽象为控制论对象并给出 RL 收敛保证。与情景控制(MFEC/NEC)区别在于检索提供语义邻域上下文而非直接规定动作。被后续 Memento-Skills（arXiv 2603.18743）作为 SRDP 与收敛定理（定理1.3）依据，扩展到技能级记忆与“让智能体设计智能体”。展望将记忆内化进 LLM 架构，关联 Titans、MIRAS、EM-LLM（作者另一工作）。
- **可复现性**: 可复现性以理论证明为主（附录 A–E 提供完整定理证明，如 KL 正则化评估算子收缩、单调改进、两时间尺度跟踪引理等），数学层面可核验。无新代码/数据集发布；实证可复现性依赖前作 Memento 的开源实现。社区采纳信号有限但已被同团队后续论文引用沿用。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 部分是“学习型记忆控制”：检索策略 μ(c|s,M) 不止启发式，可经 KL 正则化软策略迭代被学习/优化（把检索整合进策略改进），并证明收敛到最优检索策略 μ*；但写入/遗忘策略仍较启发式（Parzen 先验 + 记忆增长）。整体定位为“为学习型记忆控制提供 RL 理论基础”，而非端到端 RL 训练一个记忆管理器。
- **记忆主体**: 以智能体为中心（agent-centric）：记忆存储智能体自身的交互经验 (s,a,r,s') 用于自我改进与持续学习，而非记住用户信息做个性化。与 ReasoningBank/Voyager 类同属自改进路线，但本文给出统一的控制论理论框架。
- **多智能体记忆**: 单智能体框架：SRDP 建模单个智能体与其情景记忆的读写交互，未涉及多智能体共享/路由记忆。（多智能体的技能记忆扩展见后续 Memento-Skills，不在本文范围）
- **模态**: 理论上模态无关，论文以文本/RL 智能体为主表述；归类含 cs.CV 且明确指出广义反思也见于多模态 LLM 与非语言 transformer，但本文不针对具体视觉/具身记忆做实例化（文本为主，模态中立）。

**其他信息 / Other**

- **theoretical_grounding**: 极强——这是本文的全部贡献：提出 SRDP 形式化定义（Def.3，元组 ⟨S,A,P,R,γ,𝔐,p_LLM⟩）与 Reflected MDP（Def.4）；证明 Read=策略改进/Write=策略评估的等价性；给出 Parzen-KL 软策略迭代的收敛（定理8）、两时间尺度收敛（定理10）、渐近最优性（推论15，价值差上界 2R_max/(1-γ)^2·Δ_M）等带证明定理（附录含 KL 正则化评估算子收缩、单调改进、Lipschitz 连续性、跟踪引理等）。是首个为记忆驱动 LLM 智能体提供收敛保证的控制论决策过程刻画。
- **biological_inspiration_detail**: 受认知科学启发：明确对标 Tulving 的情景记忆（cognitive psychology episodic memory）概念，并与人类“少样本、靠语义表征+情景记忆+反思复用经验”的学习对比；相关工作讨论海马体记忆形成/检索（如 HAMI），但本文实现是概率/信息论式（Parjen-KL）而非神经生理机制的忠实复刻。

**不确定字段 / Uncertain**

- 基准 (`benchmarks`)
- 引用数 (`citations_approx`)
- compute_cost (`compute_cost`)
- 冲突/矛盾处理 (`conflict_contradiction_handling`)
- 失败学习 (核心主题) (`failure_learning`)
- information_density (`information_density`)
- 过度个性化/记忆安全风险 (`over_personalization_risk`)
- scalability_evidence (`scalability_evidence`)
- 时序推理支持 (`temporal_reasoning_support`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="a8-musememory-utilizing-and-self-evolving记忆驱动的自我进化智能体框架论文题名learning-on-the-job-an-experience-driven-self-evolving-agent-for-long-horizon-tasks"></a>

### A8 MUSE

*MUSE（Memory-Utilizing and Self-Evolving，记忆驱动的自我进化智能体框架；论文题名《Learning on the Job: An Experience-Driven, Self-Evolving Agent for Long-Horizon Tasks》）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本，2025-10-09 首次公开；cs.CL）
- **作者/机构**: Cheng Yang（杨成，第一作者）、Xuemeng Yang、Licheng Wen（三人共同一作）、Daocheng Fu、Jianbiao Mei、Rong Wu、Pinlong Cai、Yufan Shen、Nianchen Deng、Botian Shi（石博天，通讯）、Yu Qiao、Haifeng Li（李海峰，通讯）等共 12 位作者。主要单位为中南大学（Central South University）与上海人工智能实验室（Shanghai Artificial Intelligence Laboratory），合作单位含复旦大学、上海创智学院（Shanghai Innovation Institute）、浙江大学。
- **论文链接**: https://arxiv.org/abs/2510.08002
- **引用数**: 约 15 次引用（Semantic Scholar，CorpusId 281951621，截至调研日；DBLP corr/abs-2510-08002）

**记忆分类 / Taxonomy**

- **记忆类型**: 以程序性记忆（procedural）为核心，并显式分层为三类：Procedural Memory（程序性，标准操作流程 SOP）、Strategic Memory（策略性，宏观行为范式/困境-策略对）、Tool Memory（工具性，单工具用法的「肌肉记忆」）。整体可视为 CoALA 框架下程序性记忆为主、辅以高层策略性洞见与工具语义记忆的复合体；运行时的 ReAct 历史构成工作记忆（working memory）。无独立的用户级语义记忆。
- **记忆结构**: 分层式记忆模块 M = {M_strat, M_proc, M_tool}，三层对应不同抽象层级，均以自然语言（NL）表示。Strategic Memory 为 <Dilemma, Strategy>（困境-策略）键值对集合；Procedural Memory 为层级化 SOP 知识库——先按应用/平台（application/API）一级索引、再按子任务二级 SOP 索引（记录关键分析、注意事项、核心参数与操作步骤），且 SOP 拆为 (index_p, content_p) 索引-内容分离结构；Tool Memory = {D_static 静态描述, I_dynamic 动态指令}。刻意保持自然语言、人类可读、LLM-agnostic（与具体模型无关、可跨模型迁移）。
- **存储后端**: 外部化的自然语言记忆库（结构化文本/键值条目与层级 SOP 库），非向量数据库、非知识图谱、非模型参数。Strategic Memory 与 Tool Memory 的静态描述在初始化时整体载入 system prompt；Procedural Memory 仅将轻量级 SOP 索引载入上下文，详细内容通过专用「记忆检索」工具（a_mem）按需拉取。论文未指明具体存储引擎，强调以文本形式维护以保证跨 LLM 可迁移性。
- **持久化**: 外部持久化存储（durable external store），跨任务/跨迭代长期保留并增量累积，实现「在岗学习（learning on the job）」的测试时学习。不进行任何 LLM 微调，故非参数化；记忆内容以临时上下文（system prompt 或按需检索结果）形式注入推理过程。

**核心机制 / Mechanisms**

- **写入/编码**: 采用「反思蒸馏」式编码而非逐字保存原始轨迹，由独立的 Reflect Agent 在两个时机自主完成（无需人工干预）：(1) 子任务成功后（即时）——把 PE Agent 的成功执行轨迹 h_{k:t}=(o_{k:t}, a_{k:t-1}) 蒸馏为一条新的 SOP（p_new），记录关键分析、注意事项、核心参数与操作步骤，并立即加入 Procedural Memory 供复用；子任务失败则生成失败原因诊断报告 R_fail 并指示重规划。(2) 整个任务完成后（Post-Task Distill）——对完整轨迹做全局分析，抽取 <Dilemma, Resolution Pattern> 对强化 Strategic Memory、编码有效工具用法增补 Tool Memory，随后对三类记忆做统一的去重、泛化与新旧知识整合。所有编码结果均为自然语言结构化知识，刻意把低层动作序列转写为可复用的高层经验以减少冗余探索。
- **检索机制**: 三层记忆采用差异化、轻量级、以「主动按需检索」为主的读取策略，刻意不依赖嵌入向量相似度检索：(1) Strategic Memory 全量载入 system prompt，全程指导；(2) Tool Memory 中的静态描述载入 system prompt，动态指令 I_dynamic 在每次工具调用后随观察 o_t 返回以指导下一动作 a_{t+1}（自动、无需主动检索，类「肌肉记忆」）；(3) Procedural Memory 采用索引-内容分离的「主动检索」机制——启动时仅载入全部 SOP 的轻量索引 I_{M_proc}={index_p}，PE Agent 在执行中遇到不确定时通过专用记忆检索工具 a_mem 按需拉取特定 SOP 的完整 content_p（论文称此设计「模拟人类专家查阅过往案例」，并用提示工程鼓励在每个子任务开头优先查询）。该机制在尊重上下文长度限制的前提下实现低成本经验复用。注意：retry（重试）阶段刻意不再强制使用 Procedural Memory，以鼓励探索新方法、避免被错误既有知识误导。
- **反思/巩固**: 反思巩固是本框架的核心，由独立第三方监督者 Reflect Agent 在「Plan-Execute-Reflect-Memorize」闭环中执行。子任务级评估：每当 PE Agent 完成子任务或达到动作上限 N（设为 20）即触发，按三维有序检查清单评估——(1) 真实性核验 Truthfulness（结论须基于真实环境反馈，抑制幻觉）、(2) 交付物核验 Deliverable（输出文件/报告的存在性、完整性、正确性）、(3) 数据保真 Data Fidelity（数据未丢失/截断/篡改），并用「轨迹回溯（trajectory referencing）」与「主动验证（active verification，亲自调用工具与环境交叉核对）」两种方法检查，输出 success/failure 标志 f 与检查报告。成功则蒸馏 SOP，失败则产出失败诊断并触发重规划。任务级巩固（Memory Update）：任务结束后对全轨迹做全面升级，抽取困境-解决模式强化策略记忆、编码工具用法增补工具记忆，并对三类记忆统一精炼整合——融合新旧知识、消除冗余、泛化共性模式。消融实验证实 Reflect Agent 不可或缺（去除后在 18 任务子集上 S_partial 从 55.85% 降至 43.21%）。
- **遗忘/更新**: 无生物启发的衰减/遗忘曲线，但具备显式的更新-合并-精炼机制。Strategic Memory 在每次任务后被「更新、合并、精炼（updated, merged, refined）」以始终保持简洁、防止上下文膨胀；Procedural Memory 采用两阶段更新——子任务成功即时新增 SOP，任务完成后再做全局精炼（去重 deduplication、泛化 generalization）以持续优化长期质量；Tool Memory 在每次任务后由 Reflect Agent 更新。整体通过「新旧知识整合 + 去冗余 + 泛化」实现质量演化，而非简单累加。论文未实现基于时间的自动遗忘/衰减。
- **经验回放 (核心主题)**: 经验复用是本框架的核心主题与设计目标。MUSE 不重放原始动作序列，而是把成功子任务轨迹蒸馏为结构化 SOP（程序性记忆）、把困境-解决经验蒸馏为策略性记忆、把工具用法蒸馏为工具记忆，并在后续任务中通过「索引查询 + 按需检索 + system prompt 注入」复用，使智能体「避免既往失败的探索路径、把探索算力重新分配到更有希望的区域」，从而剪枝决策空间、实现更深更成功的搜索。连续学习实验显示：在 18 任务子集上随三轮迭代不断把累积经验前向携带，S_ckpt 与 S_partial 单调递增，末轮较无记忆基线高出 10% 以上。泛化实验进一步表明蒸馏经验可零样本迁移到全新困难任务（learns transferable and generalizable memory，而非死记任务专属解）。由于记忆为自然语言、LLM-agnostic，一个模型积累的经验可被另一模型直接复用（如从 Gemini 迁移到 DeepSeek-V3）。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric）/ 提示层（prompt-level）的测试时学习（test-time learning）。明确不对 LLM 做任何微调（论文论证长程任务下微调计算不可行、RL 奖励稀疏且难设计），仅通过外部分层记忆模块的累积、精炼与上下文注入实现自我进化。属纯非参数路线。
- **失败学习 (核心主题)**: 显式的失败驱动学习是核心机制之一。每个子任务设动作上限 N=20，达到上限或被判失败时，Reflect Agent 介入：生成失败原因诊断分析报告 R_fail，反馈给 PE Agent 触发重规划/重试；retry 阶段刻意解除「必须使用 Procedural Memory」的约束，鼓励在既有知识错误或不适用时探索新方法（强调 exploration over exploitation）。若二次尝试仍失败则触发子任务重规划。Reflect Agent 的真实性核验专门用于抑制 PE Agent「误以为任务已完成」的幻觉。论文在展望中明确提出未来可「对比成功与失败轨迹学习」。失败信号本身（诊断报告）作为历史记录回灌，避免重复踩坑；但论文未把「失败模式」单独固化为负例记忆条目，主要以即时诊断+重规划+策略蒸馏的方式吸收失败。
- **技能/程序归纳**: 是，且为框架重点。它从成功子任务轨迹自主归纳可复用的标准操作流程（SOP，层级化按应用→子任务索引，含关键分析/注意事项/核心参数/操作步骤），并从困境抽象出 <困境,策略> 行为范式、从工具调用沉淀工具用法指令。技能/流程以自然语言表示，通过 system prompt 注入（策略、工具静态描述）或专用检索工具 a_mem 按需调用（SOP），调用后还可获动态工具指令引导下一步。
- **在线 vs 离线**: 在线为主（online，部署/测试时按任务流逐次增量构建与巩固记忆），但评测中也采用「先在 T_cl 子集上做三轮连续学习以积累记忆、再冻结记忆评估全集/困难集」的离线-在线混合协议；记忆累积不依赖离线训练语料，全部源于智能体自身与环境的实时交互。

**评测 / Evaluation**

- **任务领域**: 真实世界长程生产力任务（long-horizon productivity tasks）/ 企业办公自动化。在高保真模拟公司环境中跨多应用操作（聊天客户端、云存储、项目管理软件、代码编辑器、网页浏览器等），覆盖 HR、PM、SDE 等六类岗位，单任务平均 >40 步、常跨 2 个以上应用、部分超 100 步。属 GUI/Web/编码/企业混合的交互式具身-数字办公领域（文本/可访问性树为主）。
- **基准**: TheAgentCompany（TAC，xu2024）——含 175 个任务、模拟高保真公司环境、六类岗位、共 776 个评估检查点（checkpoint）。子集划分：T_cl（连续学习子集，18 任务覆盖六岗位）、T_hard（泛化困难子集，12 任务，连 Claude-4 Sonnet 都近乎 0 分）、以及全 175 任务全集。未使用 LoCoMo/LongMemEval/WebArena/ALFWorld 等传统记忆基准（论文论证这些基准复杂度与长程依赖不足）。
- **报告增益**: 评测指标为部分完成分 S_partial = 0.5·(已完成checkpoint/总checkpoint) + 0.5·S_full（S_full 为是否完整完成的 0/1 指标）、聚合检查点分 S_ckpt 与 PCR（完美完成率）。核心模型为 Gemini-2.5 Flash（NPC 用 GPT-4o）。(1) TAC 全 175 任务全集：MUSE 取得 S_partial=51.78%、S_ckpt=59.92%（465/776）、PCR=41.14%，首次突破 50% 阈值，较前 SOTA（OpenHands-Versa + Claude-4 Sonnet，S_partial=43.19%、S_ckpt=50.52%、PCR=33.14%）相对提升近 20%（绝对 +8.59）；且仅用更轻的 Gemini-2.5 Flash、记忆仅从约 10% 任务习得。(2) 连续学习（T_cl 18 任务，5 次完整运行取均值，三轮迭代）：S_ckpt 与 S_partial 随迭代单调递增，末轮较无记忆基线高出 10% 以上。(3) 泛化（T_hard 12 任务，零样本）：MUSE w/o mem 已达 S_partial=23.65%（S_ckpt=30.51%，18/59），加冻结记忆后升至 S_partial=33.41%（S_ckpt=40.68%，24/59）；对照 OpenHands+Gemini-2.5 Pro 仅 3.00%、OpenHands-Versa+Claude-4 Sonnet 仅 2.00%。(4) 跨模型迁移（DeepSeek-V3，T_cl）：w/o memory S_partial=28.01%（已超所有开源模型框架，如 Llama-3.1-405B 9.78%），w memory 升至 36.75%，证明记忆模型无关、可跨 LLM 迁移。
- **对比基线**: 无记忆基线（MUSE w/o mem，同模型 Gemini-2.5 Flash / DeepSeek-V3）、去反思变体（No Reflection Variant，消融）；外部 SOTA 框架对照包括 OpenHands（gemini-1.5/2.0/2.5 pro 等）、OpenHands-Versa（claude-3.7 / claude-4 sonnet）、OWL-RolePlay（gpt-4o + o3-mini）；开源模型对照含 Llama-3.1-405B、Llama-3.3-70B、Qwen-2.5-72B。

**分析 / Analysis**

- **关键创新**: 提出以「分层（策略/程序/工具三级）自然语言记忆模块」为核心、「Plan-Execute-Reflect-Memorize」闭环驱动的测试时自我进化框架，让仅用轻量 Gemini-2.5 Flash 的智能体在真实世界长程生产力基准 TAC 上自主把原始轨迹蒸馏为可复用结构化经验、随在岗实践持续进化并取得新 SOTA（首破 50%），且记忆为 LLM-agnostic 可跨模型迁移、可零样本泛化到全新困难任务。
- **局限**: 作者承认记忆架构非万能：在高层规划（high-level planning）与多跳搜索（multi-hop search）类任务上仍有局限；评测基准 TAC 自身存在任务描述模糊/不准确、评分脚本僵化（未覆盖全部有效解、会误判合理策略）等问题；框架当前为全自主、未纳入人类反馈（虽设计上预留增删改查接口便于人机协同，但未实现/未评估）；失败经验未单独固化为负例记忆库、未对比学习成功与失败轨迹（列为未来方向）；记忆库随规模增长的扩展性/成本、对闭源商用 LLM 与 LLM 自评（Reflect Agent）稳定性的依赖未深入分析。
- **与其他工作关系**: 属「A. 反思与失败驱动（Reflection & failure-driven）」簇。与 A1 Reflexion（shinn2023）共享「语言反思 + 自我迭代」理念，但 Reflexion 多为单任务自反思文本，MUSE 由独立 Reflect Agent 把成功轨迹蒸馏为跨任务可复用的分层 SOP/策略/工具记忆并做全局精炼。与 A5 ExpeL（zhao2024）类似把执行轨迹提炼为自然语言洞见与规则，但 MUSE 进一步分三层抽象并引入索引-内容分离的主动检索与「肌肉记忆」式工具指令；与 A6 ReasoningBank 同为智能体中心、自然语言、非参数测试时学习记忆，但 ReasoningBank 偏扁平的「推理策略条目 + 嵌入检索 + 仅添加巩固」并显式吸收失败，MUSE 偏层级 SOP/策略/工具 + 主动按需检索 + 去重泛化式精炼，且聚焦更长程的真实办公任务。论文正文引用并区别于 Mem0（显式记忆操作）、MemInsight（摘要+标签增强检索）、Agent Workflow Memory（仅从成功归纳工作流）、Memp（可更新终身程序性记忆）、Voyager（技能库/课程学习自由探索）。属智能体中心自我进化记忆，区别于 Mem0/Zep/LongMemEval 等用户中心个性化记忆。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式流程）。记忆的存/取/更新/巩固全部由固定的非学习型流程与提示工程驱动：主动按需检索（SOP 索引→a_mem 拉取）、规则化的三维评估清单、任务后启发式精炼（去重/泛化/整合），不使用 RL 或梯度训练来学习「何时/何物存取更新」的记忆管理策略本身。处于 2025-26「学习型记忆控制」分水岭的启发式一侧（与 Memory-R1、Mem-α 等学习型记忆策略相对）。
- **记忆主体**: 智能体中心（agent-centric）：记忆的是智能体自身的执行经验（成功 SOP、困境-策略、工具用法），目的是让智能体在岗自我进化、跨任务自我提升与提效，而非记住用户信息做个性化。与 Voyager、ReasoningBank 同类，区别于 Mem0/Zep/LongMemEval 等用户中心记忆。
- **多智能体记忆**: 单智能体的共享记忆（single-agent，但内部多角色协作）。框架含 Planning-Execution（PE）Agent 与 Reflect Agent 两个角色，共享同一分层记忆模块 M 与同一工具集 A_tool；但这是单一智能体内部的职责分工，而非多个独立智能体间的记忆路由/共享。未涉及 G-Memory/MIRIX 式的多智能体记忆分层与路由。
- **时序推理支持**: 否。不显式建模时间有效性、事件先后或事实时效窗口；记忆为与时间无关的程序/策略/工具知识，更新以「新旧整合+去冗余+泛化」而非时间衰减/刷新。长程任务中的时序由 PE Agent 的子任务队列 Q 与 ReAct 历史维护，而非记忆层的时间建模。
- **模态**: 以文本为主（text-only-leaning）。观察以文本化环境反馈/页面信息为主，但工具集含「视觉提取器（vision extractor）」用于从界面提取信息，故具备一定视觉辅助能力；非系统性的多模态记忆（不存储截图/视频，记忆本体为自然语言）。
- **过度个性化/记忆安全风险**: 未涉及。论文不处理用户个性化，亦未讨论有害/过时/侵入性/谄媚记忆、隐私治理或过度个性化风险——这些属用户中心记忆的安全维度，在本智能体中心工作范围之外。其相关安全机制是 Reflect Agent 的真实性核验（抑制幻觉、防止把错误结论沉淀为经验）。
- **冲突/矛盾处理**: 部分处理，但非面向「矛盾事实」的显式消解。Strategic/Procedural/Tool 三类记忆在每次任务后做精炼整合时进行去重（deduplication）、泛化（generalization）与新旧知识融合，可在一定程度合并冗余/不一致的经验条目；Strategic Memory 通过持续「更新、合并、精炼」保持简洁。但论文未提供专门的冲突事实检测与版本化消解机制（不同于 MEMTRACK / Memory-R1 的显式 UPDATE 冲突处理），主要依赖任务后全局精炼与 Reflect Agent 的真实性核验来维持一致性。
- **token成本/延迟证据**: 以「减少冗余探索 / 剪枝决策空间 / 更少探索步数」与「上下文精简」作为效率论据，但未给出量化的 token 节省或墙钟延迟百分比。核心效率设计：Procedural Memory 索引-内容分离 + 按需检索，避免把全部 SOP 内容载入上下文以尊重上下文长度限制、降低 overhead；Strategic Memory 保持简洁防上下文膨胀。定性结论是积累经验使智能体「避免既往失败路径、聚焦有效解、streamline LLM 的上下文、实现更深探索」，并以仅用轻量 Gemini-2.5 Flash 即超越使用 Claude-4 Sonnet/Gemini-2.5 Pro 的重型基线来佐证成本效率，但论文未报告精确的 token/延迟 delta。

**不确定字段 / Uncertain**

- 代码链接 (`code_url`)
- 可复现性 (`reproducibility`)
- 发表venue (`venue`)


## B. 情景记忆与检索架构 (Episodic memory & retrieval)


<a id="b1-生成式智能体-generative-agents论文标题generative-agents-interactive-simulacra-of-human-behavior核心组件别名记忆流-memory-streamsmallville-沙盒小镇"></a>

### B1 生成式智能体

*生成式智能体 (Generative Agents)；论文标题《Generative Agents: Interactive Simulacra of Human Behavior》；核心组件别名：记忆流 (Memory Stream)、Smallville 沙盒小镇*


**基本信息 / Provenance**

- **年份**: 2023（arXiv 预印本 2023-04-07；UIST 会议同年 2023）
- **作者/机构**: Joon Sung Park、Joseph C. O'Brien、Percy Liang、Michael S. Bernstein（斯坦福大学）；Carrie J. Cai、Meredith Ringel Morris（Google Research / Google DeepMind）。第一作者 Joon Sung Park，通讯/资深作者 Michael S. Bernstein，均来自斯坦福大学。
- **发表venue**: UIST 2023（第 36 届 ACM 用户界面软件与技术研讨会），DOI 10.1145/3586183.3606763
- **论文链接**: https://arxiv.org/abs/2304.03442 （venue DOI: https://doi.org/10.1145/3586183.3606763）
- **代码链接**: https://github.com/joonspk-research/generative_agents （官方开源仓库，约 21,480 stars / 3,011 forks，截至 2026-06）
- **引用数**: 约 4322 次（Semantic Scholar 实时数据，2026-06 核实；属于该领域奠基性高引用工作）

**记忆分类 / Taxonomy**

- **记忆类型**: 以情景记忆 (episodic) 为主：记忆流逐条记录智能体的观察、对话与经历（带时间戳）。通过反思 (reflection) 将情景记忆综合为更高层的语义记忆 (semantic)，即关于自身与他人的抽象推断；规划 (planning) 产物也写回记忆流，带有一定的程序性/计划性记忆色彩。无独立的参数化记忆。
- **记忆结构**: 原始自然语言记忆流 (memory stream)——一个随时间增长的、带时间戳的自然语言记录列表。反思在其上构建为一棵『反思树 (reflection tree)』：叶节点为观察，内部节点为越来越抽象的反思，形成层级化抽象结构。整体可视为线性缓冲 + 反思层级树的混合。
- **存储后端**: 外部持久化存储：记忆对象以自然语言文本形式存储，并为每条记忆用语言模型生成嵌入向量 (embedding) 以支持相关性检索；开源实现以本地文件/JSON 形式保存模拟状态，检索时在上下文窗口内拼接。底层 LLM 为 ChatGPT (gpt-3.5-turbo 系列)。
- **持久化**: 外部持久化 (durable external store)：完整记忆流在智能体整个生命周期（两天模拟）中持续累积、永久保留，不随单次推理而丢失；检索结果临时注入上下文窗口（短暂态）。不写入模型参数，因此为非参数化的外部记忆。

**核心机制 / Mechanisms**

- **写入/编码**: 每一次观察 (observation)、智能体对话和行为都作为一条记忆对象逐字 (verbatim) 以自然语言写入记忆流，并附带创建时间戳和最近访问时间戳。写入时立即由语言模型为该记忆生成两类附加信息：(1) 一个 1–10 的重要性/感伤度分数 (importance / poignancy)，通过直接提示 LLM『在 1 到 10 的尺度上，1 为纯粹平凡（如刷牙、整理床铺），10 为极其重要（如分手、被大学录取），给该记忆打分』获得（示例：『打扫房间』得 2，『约心仪对象出去』得 8）；(2) 一个文本描述的嵌入向量。除原始观察外，反思生成的高层推断与规划生成的计划同样作为记忆对象写回记忆流。
- **检索机制**: 三因子加权检索：对查询记忆，对记忆流中每条记忆计算三个分数并 min-max 归一化到 [0,1]，按公式 score = α_recency·recency + α_importance·importance + α_relevance·relevance 加权求和（实现中三个 α 全部设为 1）。其中：Recency（近因性）= 自上次检索以来的沙盒游戏小时数上的指数衰减函数，衰减因子 0.995；Importance（重要性）= 写入时 LLM 打出的 1–10 感伤度分数；Relevance（相关性）= 记忆嵌入向量与查询记忆嵌入向量的余弦相似度。最终取排名最高、且能放入 LLM 上下文窗口的若干记忆注入提示。
- **反思/巩固**: 反思是核心创新之一，将原始观察综合为更高层推断。触发条件：当智能体最近感知到的记忆的重要性分数之和超过一个阈值时触发（论文中约每天触发 2–3 次）。流程：(1) 用记忆流中『最近 100 条记录』查询 LLM，提示『仅根据以上信息，关于这些主体我们能回答的 3 个最显著的高层问题是什么？』生成候选问题（如『Klaus Mueller 对什么充满热情？』）；(2) 以这些问题为查询检索相关记忆（包括其他反思）；(3) 提示 LLM 基于检索到的证据生成带引用来源的高层洞见 (insight)，写回记忆流。反思可基于其他反思，从而构成多层反思树。
- **遗忘/更新**: 无显式的删除/遗忘/合并/去重机制。近因性的指数衰减（因子 0.995）只是在检索排序中降低旧记忆的得分（软性『淡化』而非真正删除）；记忆流只增不删，所有记忆永久保留。不做冲突消解或事实更新。
- **经验回放 (核心主题)**: 通过检索机制实现隐式经验复用：过去的观察、对话、反思与计划被持续写入记忆流，并在每次决策、对话生成、日程规划时按 relevance·recency·importance 动态检索回上下文，从而以自然语言上下文示例的形式『复用』过去经历来塑造当前行为（如记住要参加情人节派对、记住与某人的关系）。规划同样以前一日记忆为基础递归生成当天日程。但没有显式的回放缓冲区 (replay buffer)、技能蒸馏或梯度层面的轨迹再训练——复用完全发生在提示/上下文层面。

**学习维度 / Learning**

- **学习范式**: 非参数化 (non-parametric)、提示/上下文层面的学习：底层 LLM (ChatGPT) 冻结不做梯度更新，全部『学习』通过记忆流的累积、反思综合与动态检索注入上下文实现。属纯 in-context、prompt-level 学习。
- **失败学习 (核心主题)**: 无专门的失败检测或失败学习机制。智能体不显式识别失败轨迹、不维护失败模式记忆、不使用负例或错误驱动规则。论文反而把失败作为系统自身的局限来分析（最常见错误：未能检索到相关记忆、对记忆虚构夸大式的幻觉、继承了 LLM 过于正式的语气）。反思虽能产生对自身/他人的推断，但其目标是『可信度 (believability)』而非从失败中纠错改进。
- **技能/程序归纳**: 不诱导可复用技能/工作流。规划 (planning) 会把高层日程递归分解为细粒度行为（带程序性色彩），并能在被打断时作出反应 (react) 并重新规划，但这是每日即时生成的计划，而非从经验中归纳出的、可跨任务调用的命名技能库（与 Voyager 的技能库形成对比）。
- **在线 vs 离线**: 完全在线 (online)：记忆、反思与计划都在部署/模拟运行过程中逐时刻、逐天即时构建与更新；不存在离线批量训练轨迹语料的阶段。

**评测 / Evaluation**

- **任务领域**: 可交互的社会行为模拟 / 沙盒游戏环境：受《模拟人生 (The Sims)》启发的 Smallville 小镇，含 25 个生成式智能体的日常生活、对话、关系形成与社会协调（如自发组织情人节派对、信息扩散、市长竞选讨论）。非传统 QA / 网页导航 / 具身基准任务，而是面向 HCI 的可信人类行为模拟。
- **基准**: 无标准化学术基准（如 LoCoMo、WebArena、ALFWorld）。采用自建评测：(1) 受控的个体『访谈』评测，向智能体提问以考察记忆保持、计划、反应、反思、自我认知；(2) 对 25 个智能体两个完整游戏日的端到端涌现行为分析（信息扩散、关系形成、协调）。
- **报告增益**: 核心定量结果为消融研究（100 名被试在 within-subjects 设计下对 5 种条件按可信度排名，转换为 TrueSkill μ 评分，μ 越高越可信）：完整架构 μ=29.89 (σ=0.72) 为最佳；去掉反思 μ=26.88 (σ=0.69)；去掉反思+规划 μ=25.64 (σ=0.68)；人类众包工作者条件 μ=22.95 (σ=0.69)；无记忆/规划/反思（代表此前 LLM 智能体的 SOTA）μ=21.21 (σ=0.70) 最差。完整架构 vs. 无记忆条件的标准化效应量 Cohen's d=8.16（约 8 个标准差）。Kruskal-Wallis 检验 H(4)=150.29, p<0.001，Dunn 事后检验经 Holm-Bonferroni 校正确认各条件两两差异显著。每移除一个组件性能都单调下降，且完整架构甚至超过人类众包工作者基线。端到端定量结果：信息扩散——已知 Sam 参选市长的智能体比例从 4%(1/25) 升至 32%(8/25)，已知 Isabella 派对的从 4%(1/25) 升至 52%(13/25)，两天内无人虚构这些信息；关系密度（图的网络密度）从 0.167 升至 0.74；被邀请的 12 人中 5 人到场参加派对。
- **对比基线**: 对比对象：(1) 无观察/反思/规划架构——无任何记忆流访问，论文明确指出其『等同于此前 LLM 智能体的 SOTA』；(2) 无反思/规划架构（仅有观察）；(3) 无反思架构（有观察+规划）；(4) 人类众包工作者撰写的条件 (crowdworker)。即逐组件消融 + 人类基线。

**分析 / Analysis**

- **关键创新**: 首次提出『记忆流 + 反思 + 规划』的 LLM 智能体认知架构：用自然语言完整记录经历，用 relevance·recency·importance 三因子动态检索把相关记忆注入上下文，并用反思将原始观察综合为更高层洞见，从而支撑长期连贯、可信的个体与涌现性社会行为（如 25 个智能体自发协调一场派对）。开创了 LLM 智能体长期记忆与检索的范式，是后续情景记忆工作的奠基参考。
- **局限**: (1) 成本高昂：模拟 25 个智能体两天耗费数千美元 token 费用并需数日，难以实时；(2) 检索失败——常因未检索到相关记忆而出错；(3) 幻觉/夸大——会对记忆作虚构性夸大（虽很少完全捏造）；(4) 继承 LLM 过于正式/刻板的语气与行为；(5) 无真正遗忘/去重/冲突消解，记忆流只增不减，长期可扩展性存疑；(6) 评测时间尺度短（仅两天），人类基线为众包工作者而非最佳人类表现；(7) 记忆模块的安全/隐私与价值观偏差未充分探讨。
- **与其他工作关系**: 属于本研究 B 类（情景记忆与检索）的奠基性工作，几乎被后续所有 LLM 智能体记忆系统引用为参照。其『非参数、自然语言、外部检索』范式与 A 类的 Reflexion（自我反思失败经验）互补但侧重不同：Reflexion 面向任务失败纠错（agent-centric 自我改进），而 Generative Agents 的反思面向可信社会行为而非失败学习。其三因子检索评分被许多后续工作借鉴或简化；其『反思树』层级抽象思想被 A-MEM、记忆图谱类工作进一步发展；缺乏的遗忘/冲突消解/学习式记忆控制（如 Memory-R1、Mem-α）正是后续 2024–2026 前沿工作的改进方向。与 Voyager（技能库式 agent-centric 经验复用）相比，本工作不做技能归纳。
- **可复现性**: 可复现性较好：官方开源完整模拟代码 (github.com/joonspk-research/generative_agents)，约 21,480 stars，社区采用度极高，催生大量衍生项目（AI Town 等）。依赖 OpenAI ChatGPT API，复现需付费且整体模拟成本高（数千美元/两天）。环境与评测为自建、非标准基准，定量复现需重跑模拟（因架构不同会发散到不同状态）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 无：记忆的写入、重要性打分、检索权重与反思触发全部为启发式/固定规则（检索权重 α 全设为 1、衰减因子固定 0.995、重要性由 LLM 一次性打分、反思按重要性累加阈值触发）。不使用 RL 或训练来学习记忆管理策略本身——这正是 2025–26 代学习式记忆控制工作（Memory-R1、Mem-α 等）相对它的代际差异。
- **记忆主体**: 智能体中心 (agent-centric) 与社会中心：每个智能体记忆自身的经历、观察与对他人的反思，用以维持长期连贯、可信的自主行为和社会互动；同时也涉及对其他智能体的记忆（多智能体社会模拟）。不属于以记住用户偏好做个性化的『用户中心』范式。
- **多智能体记忆**: 多智能体环境（25 个智能体），但每个智能体维护各自独立的私有记忆流，不共享统一记忆库；信息只能通过智能体间的自然语言对话传播（涌现出信息扩散）。没有显式的跨智能体共享/路由记忆层（区别于 G-Memory、MIRIX 的记忆路由/分层架构）。
- **时序推理支持**: 部分：每条记忆带创建时间戳与最近访问时间戳，近因性按游戏小时数指数衰减；规划以时间日程（钟点级时间表）组织。但不显式建模事实有效期窗口、事件有效性区间或时序冲突消解（不及 Zep/Graphiti 的时间双时态建模）。
- **过度个性化/记忆安全风险**: 未专门处理：论文承认存在幻觉/记忆夸大、继承 LLM 刻板偏见的风险，并在伦理讨论中提到拟人化、过度依赖、深伪式滥用与价值观对齐等社会风险，但没有针对有害/陈旧/侵入性记忆的治理机制或基准（早于 OP-Bench 等记忆安全工作）。
- **冲突/矛盾处理**: 无冲突/矛盾消解机制：记忆流只追加不修改，新旧矛盾事实并存，依赖检索排序（近因性）隐式偏向较新记忆，但不做显式的 UPDATE/合并/失效（区别于 Memory-R1 的 UPDATE、MEMTRACK）。
- **token成本/延迟证据**: 未给出相对其他记忆系统的 token/延迟节省数据（本身就是被后续系统优化的高成本基线）。仅报告绝对成本：模拟 25 个智能体两天需数千美元 token 费用、耗时数日，无法实时交互；明确将降低成本、并行化、训练专用模型列为未来工作。

**其他信息 / Other**

- **cluster**: B. 情景记忆与检索 (Episodic memory & retrieval)

**不确定字段 / Uncertain**

- 模态 (`modality`)


<a id="b2-memorybank记忆库配套提出基于其的双语-ai-陪伴聊天机器人-siliconfriend"></a>

### B2 MemoryBank

*MemoryBank（记忆库；配套提出基于其的双语 AI 陪伴聊天机器人 SiliconFriend）*


**基本信息 / Provenance**

- **年份**: 2023（arXiv 预印本，2023-05-17 首次公开，编号 2305.10250）
- **作者/机构**: Wanjun Zhong（钟万均，第一作者）、Lianghong Guo、Qiqi Gao、He Ye、Yanlin Wang（王彦霖，通讯作者）。主要单位为中山大学（Sun Yat-Sen University，第一作者与通讯作者所在），合作单位包括哈尔滨工业大学（Qiqi Gao）与瑞典皇家理工学院（KTH，He Ye）。
- **发表venue**: AAAI 2024（第三十八届 AAAI 人工智能大会，AAAI-24）正式收录，vol. 38, no. 17, pp. 19724–19731，DOI: 10.1609/aaai.v38i17.29946；arXiv 预印本于 2023 年公开。属学术界成果并开源代码。
- **论文链接**: https://arxiv.org/abs/2305.10250 （AAAI 正式版：https://ojs.aaai.org/index.php/AAAI/article/view/29946 ，DOI 10.1609/aaai.v38i17.29946）
- **代码链接**: https://github.com/zhongwanjun/MemoryBank-SiliconFriend （官方开源，Python，MIT 许可，约 431 stars、62 forks，截至 2026-06）
- **引用数**: 约 481 次引用（Semantic Scholar，截至调研日，CorpusId 258741194；DBLP journals/corr/abs-2305-10250）。属于该领域被引较多、影响力较高的早期长期记忆代表作。

**记忆分类 / Taxonomy**

- **记忆类型**: 以情景性记忆（episodic）为主——逐字、带时间戳地存储多轮对话记录；并通过分层事件摘要与动态用户画像构建出语义性记忆（semantic）层（高层事件摘要、全局事件摘要、用户人格洞见）。属 CoALA 框架中的情景性+语义性记忆混合，无显式程序性技能记忆。
- **记忆结构**: 多层（hierarchical）外部记忆仓库（memory storage），含三类内容：(1) 按时间顺序、带时间戳的逐轮原始对话记录；(2) 分层事件摘要（每日事件摘要→全局事件摘要）；(3) 动态用户画像（每日人格洞见→全局人格总结）。每条记忆片段（memory piece）被编码为向量，用 FAISS 建立向量索引以供检索。整体是「原始缓冲 + 摘要层 + 画像层 + 向量索引」的多层结构，而非知识图谱或 Zettelkasten 笔记图。
- **存储后端**: 外部向量存储：记忆片段经编码器 E(·) 预编码为上下文向量，使用 FAISS 建立向量索引进行高效相似度检索。检索流程通过 LangChain 实现（支持开源嵌入模型与 FAISS 索引）；记忆内容以 JSON 文件（memory.json / memory_bank_*.json）形式持久化在本地。无图数据库。
- **持久化**: 外部持久化存储（durable external store）。记忆以本地 JSON 文件加 FAISS 索引形式跨会话长期保留，独立于模型参数；检索到的记忆在运行时以临时上下文（prompt）注入对话。注意：SiliconFriend 的心理对话能力是通过 LoRA 微调（参数化）获得的，但这属于人格/共情能力的注入，与 MemoryBank 的「记忆」本身解耦——记忆机制本身是非参数化、外部化的。

**核心机制 / Mechanisms**

- **写入/编码**: 采用「逐字存储 + LLM 摘要抽象」双轨编码。写入分三类：(1) 原始对话被逐轮、带时间戳地完整记录，形成按时间顺序的对话史索引；(2) 事件层抽象——用 LLM 以提示「Summarize the events and key information in the content [dialog/events]」将冗长对话蒸馏为简洁的每日事件摘要，并进一步综合为全局事件摘要，形成分层（模仿人类对关键经历的记忆）；(3) 用户画像层——以提示「Based on the following dialogue, please summarize the user's personality traits and emotions.[dialog]」从每日对话推断用户人格与情绪，再以提示「The following are the user's exhibited personality traits and emotions throughout multiple days. Please provide a highly concise and general summary of the user's personality[daily Personalities]」聚合为全局人格画像。所有记忆片段（每轮对话与事件摘要）由编码器 E(·)（英文用 MiniLM、中文用 Text2vec，可替换）预编码为向量并存入 FAISS。
- **检索机制**: 采用类似 Dense Passage Retrieval（DPR，Karpukhin 等 2020）的双塔密集检索（dual-tower dense retrieval）。整个记忆库 M 中的每条记忆片段 m 被预编码为向量 h_m，构成 M={h_m^0, h_m^1, ..., h_m^|M|} 并用 FAISS 索引；当前对话上下文 c 由同一编码器 E(·) 编码为查询向量 h_c，在 M 中检索最相关的记忆片段。检索通过 LangChain 实现，嵌入模型可灵活替换（英文 MiniLM、中文 Text2vec，亦可用多语模型）。检索后将相关记忆、全局用户画像、全局事件摘要等信息组织进对话提示中，供 LLM 生成回应。检索机制本身是纯相似度检索，不含 recency·importance·relevance 三因素打分（与 Generative Agents 不同），但记忆「保留与否」由独立的 Ebbinghaus 遗忘更新机制决定。
- **反思/巩固**: 具备明确的「原始对话→高层洞见」反思/巩固机制，体现为两条抽象链：(1) 事件摘要链——把逐轮对话蒸馏为每日事件摘要，再综合为全局事件摘要，提供对过往交互的「鸟瞰图」；(2) 用户画像链——从每日对话推断用户人格与情绪洞见，再聚合为全局人格总结。两者均由 LLM 通过专门提示触发（通常按「天」为单位对当日对话进行总结），随长期交互持续更新画像。这种分层摘要刻意模仿人类记忆只保留关键经历的特性，是 MemoryBank 区别于纯原始缓冲的核心抽象能力。但反思仅做「摘要/抽象」，不涉及从失败中提炼规则或推理策略（非智能体自我改进式反思）。
- **遗忘/更新**: 核心特色：基于 Ebbinghaus 遗忘曲线（Ebbinghaus Forgetting Curve）的记忆更新机制。采用指数衰减模型 R=e^(−t/S)，R 为记忆保持率、t 为距上次学习的时间、S 为记忆强度。简化实现：S 取离散整数，记忆首次提及时初始化为 1；每当该记忆在对话中被回忆/复述一次，S 增 1 且 t 重置为 0，从而以更低概率被遗忘（体现「间隔效应/复习」与「遗忘速率随时间先快后慢」）。该遗忘机制可选开关（论文强调在「带/不带遗忘机制」两种设定下均适用），主要服务于 AI 陪伴、虚拟 IP 等需要拟人化记忆行为的场景。论文明确承认这是「探索性、高度简化」的模型，无显式的合并/去重/冲突消解/编辑操作。
- **经验回放 (核心主题)**: 本系统的「经验复用」属于用户中心的记忆召回，而非智能体自我改进式的轨迹复用。其复用方式是：在新一轮对话中，通过密集检索召回与当前语境最相关的历史对话片段、全局事件摘要与全局用户画像，并注入提示，使 LLM 能「记起」用户既往陈述（如此前推荐过的书/算法、用户的偏好与情绪状态）并据此给出连贯、个性化的回应。换言之，被「重放」的是用户的历史交互信息与对用户的理解，目的是维持长期一致的陪伴关系与个性化，而非从过去任务轨迹中蒸馏可迁移的技能/策略来提升任务成功率。它不含 RL 回放缓冲、技能库或失败示例复用。

**学习维度 / Learning**

- **学习范式**: 记忆机制本身为非参数化（non-parametric）/ 提示层（prompt-level）：通过外部记忆库的累积、摘要与检索注入实现「持续进化」，不修改 LLM 权重。但整个 SiliconFriend 系统是混合（hybrid）的——其心理共情能力来自对开源模型（ChatGLM、BELLE）用 38k 心理对话数据做的 LoRA 参数高效微调（rank r=16、训练 3 个 epoch、单张 A100）；该微调与记忆机制解耦，且仅对开源模型进行，闭源 ChatGPT 版本不微调。
- **失败学习 (核心主题)**: 无失败学习机制。MemoryBank 面向的是开放域陪伴对话，不存在明确的任务成功/失败信号，也不进行自我反思纠错、失败模式记忆、负例示例或错误驱动规则提炼。其「学习」仅指对用户信息的累积与画像演化，以及基于遗忘曲线的记忆强度调整。这与 A 簇（反思与失败驱动，如 Reflexion、ReasoningBank）形成鲜明对比——后者显式从失败轨迹中蒸馏教训，而 MemoryBank 不涉及此维度。
- **技能/程序归纳**: 否。不从经验中归纳可复用的技能、工作流或操作程序，无程序性记忆/技能库（与 Voyager 的技能库或 Agent Workflow Memory 不同）。它产出的是事件摘要与用户画像等陈述性/语义性知识，用于个性化对话而非任务执行。
- **在线 vs 离线**: 在线（online）。记忆在部署/真实交互过程中按对话逐步增量构建：对话被实时记录，事件摘要与用户画像通常按「天」为单位定期生成与更新，遗忘强度 S 随每次回忆即时调整。不依赖离线批量轨迹训练（LoRA 微调阶段虽为离线，但那是注入共情能力，与记忆构建无关）。

**评测 / Evaluation**

- **任务领域**: 长期多轮对话 / 个人 AI 陪伴（personal AI companion）、心理咨询陪伴（psychological companionship）、个性化对话。具体落地为双语（中/英）AI 陪伴聊天机器人 SiliconFriend。属于「多会话对话 + 用户个性化记忆」域，不涉及网页导航、具身、编码、GUI 等智能体任务。
- **对比基线**: 主要为带 MemoryBank 的三种骨干 LLM 之间的横向对比（SiliconFriend ChatGPT / ChatGLM / BELLE）；定性分析中还将 SiliconFriend 与未加记忆与未做心理微调的基线 LLM（如原始 ChatGLM）就共情与记忆召回能力作对照。无 RAG、无 full-context、无其他记忆系统等量化基线（属早期工作，当时同类记忆系统尚少）。

**分析 / Analysis**

- **关键创新**: 首次将心理学的 Ebbinghaus 遗忘曲线引入 LLM 长期记忆，提出可「选择性遗忘与强化」的拟人化记忆更新机制 R=e^(−t/S)（按时间流逝与回忆频次调节记忆强度）；并将其与「逐字对话存储 + 分层事件摘要 + 动态用户画像」的多层记忆仓库及双塔密集检索整合为统一、可即插即用、兼容开/闭源 LLM 与中英双语的长期记忆框架，落地为 AI 陪伴机器人 SiliconFriend。
- **局限**: （1）遗忘/更新模型作者自认「探索性、高度简化」（S 取离散整数、仅计回忆次数），与真实人类记忆相去甚远，且无合并/去重/冲突消解机制；（2）评测为作者自建的小规模模拟数据（15 虚拟用户、10 天、194 探测题），且大量依赖人工评分与 ChatGPT 模拟用户，缺乏当时公认的标准基准，泛化性与可比性受限；（3）缺乏严格的「同骨干有/无记忆」消融与对 RAG 等基线的量化对比；（4）无 token 成本/延迟等效率量化；（5）随对话累积，记忆库规模、检索精度与成本的可扩展性未深入评估；（6）专注用户中心个性化陪伴，不处理智能体任务、失败学习或时间事实有效性等维度；（7）存储用户长期心理/人格信息带来隐私与安全治理问题，论文未讨论。
- **与其他工作关系**: 属于「B. 情景记忆与检索（Episodic memory & retrieval）」簇的早期、用户中心代表作。与同时期/稍早的 Generative Agents（斯坦福小镇）同属「带摘要/反思的外部记忆 + 检索注入」思路，但 Generative Agents 的检索用 recency·importance·relevance 三因素打分且面向智能体行为模拟，MemoryBank 则用纯密集检索、面向真人用户的长期陪伴与个性化，且独创性地以 Ebbinghaus 遗忘曲线实现拟人化遗忘。相较 A 簇的反思/失败驱动记忆（A1 Reflexion、A6 ReasoningBank 等智能体中心、从成败轨迹蒸馏可迁移策略以自我改进），MemoryBank 是用户中心、记忆用户信息以个性化，不做失败学习、不归纳技能。它是后续大量「用户中心长期对话记忆/个性化记忆」工作（Mem0、Zep、MemGPT 类、LongMemEval 评测线）的先行者之一，其自建评测数据被后续研究复用为「MemoryBank dataset」。其遗忘曲线思想也启发了后续把生物记忆衰减机制引入 LLM 记忆的研究。
- **可复现性**: 可复现性较好：官方完整开源代码、模型与数据（github.com/zhongwanjun/MemoryBank-SiliconFriend，Python，MIT 许可，约 431 stars、62 forks），含 ChatGLM/BELLE/ChatGPT 三个版本的运行脚本、LoRA 检查点（基于 38k 心理对话微调）、以及中英双语评测数据（对话史 + 探测问题）与记忆摘要脚本（summarize_memory.py）。实验环境为单张 Tesla A100 80GB + CUDA 11.7。检索依赖 LangChain + FAISS + 可替换嵌入模型（MiniLM/Text2vec）。社区采用度较高（约 481 引用、被多项后续工作当基准复用）。局限：评测含人工评分与 ChatGPT 模拟用户，ChatGPT 版本结果对 OpenAI 模型版本敏感，严格数值复现有一定难度。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式流程）。记忆的存/取/更新均为固定的非学习型规则：双塔密集检索 + 基于 Ebbinghaus 曲线的确定性强度更新 + LLM 提示驱动的摘要/画像。不使用 RL/训练来学习「何时/何物存取与更新」的记忆管理策略本身，处于 2025-26「学习型记忆控制」分水岭中的启发式一侧（与 Memory-R1、Mem-α 等学习型方法相对）。
- **记忆主体**: 用户中心（user-centric）：记忆并理解用户的历史交互、偏好、情绪与人格画像，目的是提供长期一致、个性化、富有共情的陪伴体验，而非记住智能体自身经验以自我改进。与 Mem0/Zep/LongMemEval 等用户中心记忆同类，区别于 Voyager/ReasoningBank 等智能体中心记忆。
- **多智能体记忆**: 单智能体（single-agent）。记忆库服务于单个 AI 陪伴助手与单个/各自独立的用户，不涉及多智能体间共享、路由或分层（insight/query/interaction）记忆。
- **时序推理支持**: 部分支持但较弱。记忆按时间戳、时间顺序存储，且遗忘机制显式依赖「距上次学习/回忆的时间 t」来计算保持率，事件摘要也以「天」为粒度组织，体现了对时间维度的基本建模；但它不显式建模事实有效性窗口、事件先后推理或随时间变化的事实更新（不像 Zep/Graphiti 的时序知识图谱），缺乏对矛盾事实随时间演变的结构化时序推理能力。
- **模态**: 纯文本（text-only）。仅处理中英双语文本对话，无视觉/截图/音频/具身或多模态记忆。
- **过度个性化/记忆安全风险**: 未涉及。论文未讨论有害/过时/侵入性记忆、记忆中毒、谄媚（sycophancy）或过度个性化风险，也未涉及隐私治理（尽管系统长期存储用户心理与人格画像，存在明显隐私敏感性）。属于该工作未覆盖的安全维度。
- **冲突/矛盾处理**: 基本未处理。记忆更新仅做「追加新记录 + 调整遗忘强度 S + 定期重新摘要/画像」，无显式的矛盾/冲突事实检测、合并或失效（invalidate）机制（不含 Memory-R1 式 UPDATE、MEMTRACK 式冲突追踪）。当用户陈述前后矛盾时，系统依赖检索召回相关片段与定期重生成的全局摘要/画像来「软性」反映最新理解，但无原则化的冲突消解流程。
- **token成本/延迟证据**: 无量化效率证据。论文未报告 token 成本、上下文长度节省或检索/生成延迟等数据；仅定性地以「需要先编码并 FAISS 索引以高效检索、LoRA 用于资源受限场景下高效微调」说明效率考量，但未给出相对 full-context 或其他基线的百分比节省。

**不确定字段 / Uncertain**

- 基准 (`benchmarks`)
- 报告增益 (`reported_gains`)


<a id="b3-memgptmemory-gpt又名-memorygpt提出虚拟上下文管理--llm-as-os范式后产品化为开源框架-letta"></a>

### B3 MemGPT

*MemGPT（Memory-GPT；又名 MemoryGPT，提出「虚拟上下文管理 / LLM as OS」范式；后产品化为开源框架 Letta）*


**基本信息 / Provenance**

- **年份**: 2023（arXiv 预印本 v1 于 2023-10-12 首次公开；v2 于 2024-02-12 更新）
- **作者/机构**: Charles Packer、Sarah Wooders、Kevin Lin、Vivian Fang、Shishir G. Patil、Ion Stoica、Joseph E. Gonzalez（共 7 位作者）；主要单位为加州大学伯克利分校（UC Berkeley，RISELab/Sky Computing Lab）。第一作者 Charles Packer。
- **发表venue**: COLM 2024（Conference on Language Modeling，2024-10-07 至 09 于宾夕法尼亚大学召开；OpenReview id 0Kk142lP62）。arXiv 预印本最初标注投往 ICML。属于学术成果，后由作者团队成立 Letta 公司并产品化。
- **论文链接**: https://arxiv.org/abs/2310.08560
- **代码链接**: https://github.com/letta-ai/letta（即原 MemGPT 仓库，现更名为 Letta；Python，Apache-2.0 许可，约 23,200 stars、约 2,500 forks，截至 2026-06）。PyPI 包与 Docker 镜像已分别迁移至 letta 与 letta/letta-server。
- **引用数**: 约 767 次引用（Semantic Scholar，CorpusId 263909014，截至调研日）；属于 LLM 智能体记忆方向被引最高、影响力最大的奠基性工作之一。

**记忆分类 / Taxonomy**

- **记忆类型**: 以工作记忆（working memory，即上下文窗口内的主上下文）+ 长期情景/语义记忆（external context 中的对话与文档历史）为主，并通过可自编辑的「working context」块承载半结构化的用户事实与画像（偏语义性）。在 CoALA 框架中横跨工作记忆与情景/语义记忆，强调的是「分层记忆管理」而非某一种记忆类型；不显式建模程序性记忆。
- **记忆结构**: 受操作系统启发的分层分级（hierarchical tiered）内存结构，类比虚拟内存分页。两大层级：①主上下文（main context，类比 RAM/物理内存），再细分为三段连续区域——只读的系统指令（system instructions）、固定大小可读写的工作上下文（working context，产品中称 core memory）、以及滚动消息历史 FIFO 队列（其首位存放被驱逐消息的递归摘要）；②外部上下文（external context，类比磁盘），含召回存储（recall storage，完整消息数据库）与归档存储（archival storage，任意长度文本对象的读写数据库，支持向量检索）。产品 Letta 中对应 core / recall / archival 三类记忆。
- **存储后端**: 主上下文即 LLM 的提示词上下文窗口（in-context）。外部上下文采用外部数据库：默认实现用 PostgreSQL 存储归档记忆，并经 pgvector 扩展启用向量检索，使用 HNSW 索引实现近似、亚秒级查询；召回存储为消息数据库。文档 QA 实验中用 OpenAI text-embedding-ada-002 嵌入、按余弦相似度检索。
- **持久化**: 混合：主上下文为易失的上下文内存（ephemeral，随窗口溢出被驱逐）；外部上下文（召回 + 归档存储）为外部持久化数据库（durable external store），被驱逐的消息无限期保存在召回存储中、可经函数调用随时调回。非参数化——不修改任何模型权重，所有记忆均以上下文/外部存储形式存在。产品 Letta 进一步将整个智能体状态持久化为有状态服务。

**核心机制 / Mechanisms**

- **写入/编码**: 写入完全由 LLM 自主（self-directed）通过函数调用驱动，无需用户介入。①每条到来的消息与 LLM 生成的输出由队列管理器（queue manager）原样写入召回存储（消息数据库），实现逐字（verbatim）的对话历史保存；②当主上下文逼近「警告 token 数」（如窗口 70%）时，系统插入「记忆压力（memory pressure）」警告，提示 LLM 调用函数把 FIFO 队列中的重要信息主动写入工作上下文（core memory，存关键事实/偏好/画像，属摘要式提炼）或归档存储（archival，存任意长文本）；③当超过「冲刷 token 数」（如窗口 100%）时，队列管理器驱逐约 50% 消息，并用「既有递归摘要 + 被驱逐消息」生成新的递归摘要（recursive summarization）压缩进上下文首位。文档分析场景下，外部文档（如 20M 维基百科段落的嵌入）被批量载入归档存储。因此写入兼具逐字保存（召回库）、摘要提炼（工作上下文/递归摘要）与嵌入索引（归档库）三种形态。
- **检索机制**: 读取同样由 LLM 自主通过函数调用完成。①对召回存储：可按时间/内容分页搜索历史消息，命中后由队列管理器追加到 FIFO 队列尾部、重新进入上下文窗口；②对归档存储：执行基于余弦相似度的向量检索（pgvector + HNSW），返回 top-K 结果，并支持「分页（pagination）」逐页翻阅以避免一次性检索撑爆上下文，且检索机制对 token 限制有感知。整个检索是「LLM 主动决定何时检索什么」的自主式检索（类似 FLARE 的主动检索思想，但内嵌于 OS 式控制流）。检索后内容拼入主上下文供下一轮推理。论文未使用 recency·importance·relevance 三因子打分公式（区别于 Generative Agents）。
- **反思/巩固**: 存在「巩固/压缩」式整合，但不同于反思蒸馏出新知识的范式。主要体现为：①递归摘要（recursive summarization）——队列冲刷时把被驱逐的旧消息与既有摘要重新归并为更紧凑的摘要，持续保持长程对话的可用上下文；②自编辑工作上下文——LLM 可在对话中主动改写 working context（core memory），以反映对当前目标、用户画像与自身角色的「演化中的理解」（论文图 3/图 4 示例）。这些都是由 memory-pressure 警告或 LLM 自主判断触发，而非定期对经验做高层洞见抽象。论文强调智能体能「remember, reflect, and evolve（记忆、反思、演进）」，但其「反思」更接近上下文管理与自我编辑，而非如 Reflexion/ExpeL 那样把失败经验蒸馏为可迁移规则。
- **遗忘/更新**: 无真正的遗忘/衰减机制：被驱逐出上下文的消息并未删除，而是无限期保存在召回存储、可随时调回（仅是从「在上下文内」变为「在上下文外」）。更新主要通过 LLM 自主改写工作上下文（core memory，含 ADD/replace 式自编辑）与递归摘要的归并压缩实现；归档存储为持续追加。论文未实现基于 Ebbinghaus 衰减的遗忘或显式的去重/合并/失效操作（这类能力在后续记忆系统中才被强调）。
- **经验回放 (核心主题)**: 不属于以「重放过去轨迹来自我改进」为核心的经验复用范式。MemGPT 的「复用」体现为对过往对话/文档信息的检索式调回（把外部存储中的历史信息分页调入上下文以维持跨会话一致性与个性化），而非把成功/失败的决策轨迹蒸馏为可迁移技能或策略在新任务上复用。它解决的是「无限上下文」与「长期记忆访问」问题，不显式做技能复用、范例提示或经验回放缓冲。因此在「以经验复用驱动行为改进」这一核心主题上，MemGPT 是基础设施层（提供分层记忆与自主读写），而非经验蒸馏/技能归纳层。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric）/ 提示层、上下文层。完全不进行任何梯度更新，纯靠分层记忆的自主读写与上下文管理实现「学习/演进」；依赖底层冻结 LLM（实验用 GPT-4 / GPT-4 Turbo / GPT-3.5 Turbo）的函数调用能力。属于运行时（inference-time）的有状态记忆，而非训练式学习。
- **失败学习 (核心主题)**: 几乎不涉及从失败中学习这一主题。MemGPT 不检测任务失败、不存储失败模式、不生成负面范例或纠错规则。其唯一与「错误」相关的反馈是：当函数调用出错（如向已满的主上下文写入、参数解析失败、超出 token 限制）时，运行时错误会被反馈给 LLM 处理器，使其在「同一轮交互内」调整记忆管理动作（如改用归档存储、分页检索）。这是即时的运行时自纠错，而非跨任务的失败经验学习。这是 MemGPT 与 Reflexion/Retroformer/CLIN 等失败驱动系统的根本区别。
- **技能/程序归纳**: 否。MemGPT 不从经验中归纳可复用的技能/工作流/程序。它提供一组固定的「记忆管理函数」（搜索召回库、写归档库、改写工作上下文、分页等）供 LLM 调用，但这些函数是人工预定义的工具，而非由智能体从经验中induce出来的程序性技能（区别于 Voyager 的技能库）。
- **在线 vs 离线**: 在线（online）。记忆在部署/交互过程中实时、增量构建：每条消息即时写入召回存储，上下文压力触发时即时编辑工作上下文与归档存储，跨会话/跨文档持续累积。不依赖离线批量训练语料；文档分析中虽会预先把文档嵌入载入归档库，但检索与上下文管理仍在运行时在线发生。

**评测 / Evaluation**

- **任务领域**: 两大长上下文域：①对话智能体（多会话长期对话、虚拟伴侣/个性化助手），考察一致性与个性化参与度；②文档分析（超长文档问答、跨多文档信息汇总、嵌套键值多跳检索）。后续 Letta 产品扩展至更通用的有状态智能体（自定义工具、数据源、可部署持久化服务等）。
- **基准**: ①扩展版 Multi-Session Chat（MSC，Xu et al. 2021；作者新增 session 6 并自构 DMR 问答对）上的两项新任务——深度记忆检索（Deep Memory Retrieval, DMR，测一致性）与对话开场白（Conversation Opener，测参与度）；②文档问答基于 NaturalQuestions-Open（检索-阅读设定，沿用 Liu et al. 2023a「Lost in the Middle」任务，2018 年维基百科 dump，抽样 50 题，另公开 20M 维基百科文章嵌入）；③作者新提出的嵌套键值检索（Nested Key-Value Retrieval，140 对 UUID、约 8k token、0–4 层嵌套、30 种排序配置）测多跳能力。评测用准确率、ROUGE-L 召回、CSIM 相似度（SIM-1/3/H）及 LLM-as-judge（GPT-4 裁判）。
- **报告增益**: ①深度记忆检索 DMR（一致性，相对各自无 MemGPT 的固定上下文基线，基线仅能看到对前 5 段对话的有损摘要）：GPT-3.5 Turbo 准确率 38.7%→66.9%（ROUGE-L 0.394→0.629）；GPT-4 32.1%→92.5%（ROUGE-L 0.296→0.814，提升约 +60.4 个百分点）；GPT-4 Turbo 35.3%→93.4%（ROUGE-L 0.359→0.827）。②对话开场白（参与度，相对人类开场白的相似度）：MemGPT 各底模均能匹敌甚至超越人工开场白——如 GPT-4 的 SIM-1=0.868、SIM-3=0.843（人类 0.800），GPT-3.5 Turbo SIM-H=0.817（人类基准 1.000，但优于其他底模）。③文档 QA：固定上下文基线性能受限于检索器与窗口大小、并随截断压缩而下降；MemGPT 可通过多次调用检索器分页扩展有效上下文，性能不随上下文长度增长而退化（GPT-4 与 GPT-4 Turbo 结果相当）。④嵌套键值检索：GPT-3.5 在 1 层嵌套即降为 0%，GPT-4/GPT-4 Turbo 在 3 层降为 0%；而 MemGPT+GPT-4 在各嵌套层级几乎不受影响、可稳定完成多跳查找（是唯一能稳定完成 2 层以上嵌套的方法）。
- **对比基线**: 主要对比「固定上下文」基线，即不加 MemGPT 的同款底层 LLM（GPT-4 / GPT-4 Turbo / GPT-3.5 Turbo），在对话任务中基线获得对过往会话的有损递归摘要、在文档任务中使用同一检索器但独立于推理（检索-阅读式 RAG），并通过截断（truncation）把更多文档塞入有限窗口作为扩展上下文的对照；文档/KV 任务还隐含与朴素长上下文/中段丢失（Lost in the Middle）现象对照。对话开场白任务额外对比人类撰写的开场白（Human）。

**分析 / Analysis**

- **关键创新**: 提出「虚拟上下文管理（virtual context management）」——把操作系统的分层内存与虚拟内存分页思想迁移到 LLM：用 LLM 自身作为「处理器」，通过函数调用与中断式控制流，让模型自主地在主上下文（RAM）与外部上下文（磁盘）之间分页搬运信息，从而在有限上下文窗口之上提供「无限/超长上下文」的假象。首创「LLM as Operating System」范式与可自编辑记忆（self-editing memory）的智能体设计模式，奠定了后续大量智能体记忆系统与有状态智能体框架（Letta）的基础。
- **局限**: ①依赖底层 LLM 强大的函数调用与指令遵循能力——GPT-3.5 因函数调用能力弱表现显著退化，弱模型难以驱动该机制；②无真正遗忘/去重/冲突消解机制，长期运行下外部存储无界增长、可能累积过时或矛盾信息；③检索质量受限于嵌入相似度，且观察到 MemGPT 常在穷尽检索结果前就停止翻页，导致召回不全；④「记忆/反思/演进」偏上下文管理与摘要压缩，缺乏把经验蒸馏为可迁移技能/规则的能力，不处理失败学习；⑤递归摘要存在信息有损压缩，可能丢失细节；⑥成本/延迟与多轮函数调用开销在论文中未做系统量化（自主多步函数链会增加调用次数与 token 消耗）。
- **与其他工作关系**: 属于「B. 情景记忆与检索（Episodic memory & retrieval）」簇中的基础设施型工作，与同簇侧重检索/分层记忆的系统并列。它在检索增强（RAG）与「LLM 即智能体」两条脉络之上提出分层自管理记忆：借鉴 FLARE（Jiang et al. 2023）的主动检索、WebGPT（Nakano et al. 2021）的分页思想、ReAct（Yao et al. 2022）的「出声思考/函数规划」，并与 Generative Agents（Park et al. 2023）共享「给 LLM 加记忆」的目标，但 Generative Agents 用 recency·importance·relevance 三因子打分检索、MemGPT 则用 OS 式分页 + 自编辑函数调用。与本研究中以「经验蒸馏/失败反思」为核心的 A 簇（Reflexion / Retroformer / ExpeL / CLIN / ReasoningBank）正交互补——A 簇关注从轨迹蒸馏可迁移知识，MemGPT 关注长期记忆的存取与上下文调度，二者可叠加。其工程影响最大：被产品化为 Letta，成为「有状态智能体 / agent memory」赛道（Mem0、Zep、MIRIX、MemMachine 等）的重要先驱与对比基线，core/recall/archival 记忆分层被广泛沿用。
- **可复现性**: 可复现性强且社区采用度极高：官方完整开源（github.com/letta-ai/letta，原 MemGPT 仓库，Python，Apache-2.0，约 23.2k stars），公开发布扩展版 MSC 数据集、嵌套 KV 检索数据集与 20M 维基百科文章嵌入；默认采用可自托管的 PostgreSQL+pgvector 后端，便于本地复现。论文随附完整提示词与 LLM-judge 指令（附录）。已演进为成熟的开源智能体框架并由 Letta 公司持续维护（提供 PyPI 包、Docker 镜像、商业部署）。局限在于实验底模为闭源商用 OpenAI 模型（gpt-4-1106-preview / gpt-4-0613 / gpt-3.5-turbo-1106），结果对模型版本与函数调用能力敏感。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式/规则化 + LLM 自主决策的混合，但非学习型）。记忆管理策略由「人工设定的阈值规则（警告 token 数 70%、冲刷 token 数 100%、驱逐 50%）+ 系统指令引导下 LLM 在运行时自主决定调用哪个记忆函数」共同构成，不使用 RL/训练去学习「何时存/取/更新」的策略本身。因此处于 2025-26「学习型记忆控制」分水岭的启发式/提示驱动一侧（与 Memory-R1、Mem-α 等用 RL 学习记忆管理策略的方法相对）；但其「LLM 自主调用记忆函数」的设计为后续学习型控制提供了动作空间雏形。
- **记忆主体**: 以用户中心（user-centric）为主、兼具智能体自身状态管理。对话场景下记忆的是用户的事实、偏好、画像与历史，以维持跨会话一致性与个性化（user-centric）；同时也维护智能体自身的角色/目标（working context 自编辑）。但其目标是「记住信息以保持长期对话连贯与个性化」，而非「记住自身经验以自我提升任务能力」（区别于 ReasoningBank/Voyager 的 agent-centric 自进化）。文档场景则是以外部知识为对象的长上下文管理。
- **多智能体记忆**: 单智能体（single-agent）。论文聚焦单个 MemGPT 智能体的分层记忆管理，不涉及多智能体间的共享/路由记忆。后续 Letta 框架支持多智能体与子智能体，但原论文范围内不含多智能体记忆机制（区别于 G-Memory、MIRIX 等）。
- **时序推理支持**: 弱/隐式。召回存储保留完整消息的时间顺序，FIFO 队列与递归摘要隐含时间先后，可按时间检索历史消息；但不显式建模事实有效性窗口、事件日历或时序冲突消解（区别于 Zep/Graphiti 的显式时间有效性建模）。时序仅作为消息组织维度，而非一等的可推理对象。
- **模态**: 纯文本（text-only）。处理对话文本与文本文档；原论文不涉及视觉/截图/具身/视频等多模态记忆（区别于 MIRIX 的多模态记忆）。
- **过度个性化/记忆安全风险**: 未涉及。论文不讨论有害/过时/侵入性记忆、谄媚（sycophancy）、隐私治理或过度个性化风险，也无相应的记忆安全机制或基准（如 OP-Bench/Causal-LoCoMo）。其无遗忘、无冲突消解的设计反而可能在长期运行中累积过时/矛盾的用户记忆——这一安全维度超出本工作范围。
- **冲突/矛盾处理**: 基本未处理。无显式的矛盾事实检测与合并机制。冲突信息可能并存于召回存储（旧消息不删除）与工作上下文中；唯一的「更新」途径是 LLM 在自编辑 working context 时主动覆盖/改写旧内容，但这依赖模型自发判断、无系统化的冲突消解流程（区别于 Memory-R1 的 UPDATE、MEMTRACK 的冲突追踪）。
- **token成本/延迟证据**: 原论文未系统量化 token 成本或延迟的节省百分比。其价值主张是「在固定/有限上下文窗口下实现超长有效上下文」，而非降低 token 成本——相反，自主的多步函数链（检索、分页、自编辑、递归摘要）会引入额外的 LLM 调用与 token 开销。工程上仅报告归档存储用 HNSW 索引可实现「近似、亚秒级」向量查询。整体而言效率/成本是其相对弱项，论文将注意力放在能力（一致性、多跳检索）而非效率指标上。


<a id="b4-a-memagentic-memory智能体记忆系统亦写作-a-mem"></a>

### B4 A-MEM

*A-MEM（Agentic Memory；智能体记忆系统，亦写作 A-Mem）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本，2025-02-17 首次公开，arXiv:2502.12110；多次修订）
- **作者/机构**: Wujiang Xu（徐武江，第一作者）、Zujie Liang、Kai Mei（梅凯）、Hang Gao、Juntao Tan、Yongfeng Zhang（张永锋，通讯/资深作者）。主要单位为美国罗格斯大学（Rutgers University）；另含独立研究者（Zujie Liang）与 AIOS Foundation。注意：题目中给出的「Xu, Mei, Zhang et al.」与实际作者顺序一致（Xu 为一作、Mei 为三作、Zhang 为末位资深作者）。
- **论文链接**: https://arxiv.org/abs/2502.12110
- **代码链接**: https://github.com/agiresearch/A-mem（官方记忆系统库，Python，MIT 许可，约 1k stars / 113 forks，截至 2026-06）；另有论文结果复现仓库 https://github.com/WujiangXu/AgenticMemory 与 https://github.com/WujiangXu/A-mem-sys。
- **引用数**: 约 603 次引用（Semantic Scholar，CorpusId 276421617，截至调研日），表明在 2025 年 LLM 智能体记忆方向影响力极高、被广泛作为基线/对比对象。

**记忆分类 / Taxonomy**

- **记忆类型**: 以语义性记忆（semantic）与情景性记忆（episodic）混合为主：每条交互被编码为带结构化属性（关键词、标签、上下文描述）的「原子笔记（atomic note）」，既保留具体交互内容（情景），又通过 LLM 抽取语义概念（语义）。面向长期多会话对话/问答场景，本质属于 CoALA 框架中的情景+语义记忆，但通过笔记网络（note-graph）组织，强调可演化的语义结构而非纯轨迹复用。
- **记忆结构**: 笔记图/卡片盒网络（note-graph，Zettelkasten 卡片盒式）。每条记忆笔记 m_i 形式化为七元组 {c_i, t_i, K_i, G_i, X_i, e_i, L_i}：c_i 原始交互内容、t_i 时间戳、K_i 关键词、G_i 标签、X_i 上下文描述、e_i 稠密向量嵌入、L_i 关联链接集合。笔记之间通过 LLM 判定的语义链接互联，形成可同时归属于多个「盒（box）」的动态知识网络（一条记忆可同属多个 box，区别于传统单一层级）。非固定层级、非预定义图 schema，而是有机涌现。
- **存储后端**: 向量数据库 ChromaDB 作为底层存储与检索引擎；文本嵌入采用 all-MiniLM-L6-v2（sentence-transformers）模型；LLM 后端可选 OpenAI（如 GPT-4o-mini）或本地 Ollama（Qwen、Llama 3.2 等）。检索为基于余弦相似度的稠密向量近邻搜索。
- **持久化**: 外部持久化存储（durable external store）。记忆笔记及其链接持久保存在 ChromaDB 中，跨会话/跨交互长期保留；不修改任何模型参数（非参数化），检索结果以临时上下文注入智能体提示。

**核心机制 / Mechanisms**

- **写入/编码**: 对每次交互进行「LLM 驱动的结构化笔记构建（note construction）」而非逐字保存。流程：(1) 以模板 P_s1 提示 LLM 分析交互内容 c_i 与时间戳 t_i，自动生成关键词 K_i、标签 G_i 与上下文描述 X_i（K_i,G_i,X_i ← LLM(c_i ‖ t_i ‖ P_s1)），遵循 Zettelkasten「原子性」原则——每条笔记捕获单一自包含知识单元；(2) 用文本编码器对 c_i、K_i、G_i、X_i 拼接后计算稠密向量 e_i = f_enc[concat(c_i,K_i,G_i,X_i)]，兼顾语义丰富性与高效相似度匹配。由此把原始交互编码为「内容+LLM 生成语义属性+嵌入+链接」的多面笔记，实现对隐含知识的自主抽取。
- **检索机制**: 面向当前交互查询 q 的上下文感知检索：先用同一文本编码器计算查询嵌入 e_q = f_enc(q)，再对记忆库 M 中所有笔记计算余弦相似度 s_{q,i} = (e_q·e_i)/(|e_q||e_i|)，取 rank(s_{q,i})≤k 的 top-k 笔记构成检索集 M_retrieved（论文主实验 k=10，按类别可微调，超参分析覆盖 k=10/20/30/40/50）。检索的独特之处：当某笔记被检索命中时，与其在同一「盒」内（通过链接 L_i 互联）的相关笔记会被自动一并访问（link-aware retrieval），从而把语义网络的连通结构注入检索结果。检索时间随规模增长极缓（100 万条记忆时仅约 3.70μs）。
- **反思/巩固**: 核心机制之一是「记忆演化（Memory Evolution）」——一种持续的原始经验→精炼语义结构的巩固过程，而非一次性反思摘要。当新笔记 m_n 加入并完成链接后，系统对其近邻集合 M^n_near 中的每条旧笔记 m_j 用模板 P_s3 提示 LLM 判定是否更新其上下文 X_j、关键词 K_j、标签 G_j：m_j* ← LLM(m_n ‖ M^n_near\m_j ‖ m_j ‖ P_s3)，演化后的 m_j* 替换原 m_j。如此，新经验会触发既往记忆的语义表征与属性刷新，使记忆网络「随时间持续精炼理解」、涌现更高阶模式（类人学习）。这是区别于「静态知识库」的关键：A-MEM 在存储/组织层而非仅检索层具备能动性（agency）。消融显示移除记忆演化（w/o ME）会显著掉点（如 Multi-Hop F1 从 27.02 降至 21.35）。
- **遗忘/更新**: 有「更新/演化」但无显式遗忘/衰减机制。记忆演化阶段会对近邻旧笔记的上下文、关键词、标签做 LLM 驱动的就地更新与替换（m_j*←m_j），并随新链接不断扩展网络；官方代码库另提供 update/delete API 供手动操作。但论文未实现基于时间的自动遗忘曲线（如 Ebbinghaus）、去重或冲突消解策略，记忆基本只增不删（除人工删除外）。
- **经验回放 (核心主题)**: 属于「用户/对话历史复用」而非「智能体技能复用」型。它不重放原始动作轨迹，而是把历史交互沉淀为可检索的语义笔记网络，在新交互时通过相似度检索+链接传播复用相关历史经验，为智能体推理提供「连接当前交互与过往相关经验」的上下文。复用的价值主要体现在长程多跳推理：通过笔记间的动态语义链接，把分散在不同会话的相关信息串联，使多跳问答性能在 GPT 系模型上相对基线「至少翻倍」。区别于 Voyager/ReasoningBank 那类把成功/失败轨迹蒸馏为可迁移技能/策略的「智能体自我提升」复用——A-MEM 复用的是结构化的历史事实/语义记忆，服务于长期对话问答的连贯性与多跳整合。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric）/ 提示层（prompt-level）。全程不做梯度更新，仅依靠 LLM 驱动的笔记生成、链接判定与记忆演化在外部向量库中累积与重组知识；适配可插拔的开源/商用 LLM 后端。
- **失败学习 (核心主题)**: 不涉及。A-MEM 面向长期对话/问答的记忆组织，没有显式的失败检测、失败轨迹反思或负例/避坑规则机制。它不区分「成功/失败经验」，也不从错误中提炼教训——这与 A 簇（Reflexion/Retroformer/ReasoningBank 等反思与失败驱动方法）形成鲜明对照。其改进来自更好的记忆组织与多跳连接，而非错误修正。
- **技能/程序归纳**: 否。不归纳可复用的技能/工作流/过程（procedure）。它归纳的是语义层面的关键词、标签、上下文描述与笔记间链接（知识组织），而非可被调用的操作技能或程序。无技能库、无 skill 调用机制。
- **在线 vs 离线**: 在线（online）增量构建。记忆随智能体与环境/对话的每次交互即时构建笔记、生成链接并触发对近邻旧记忆的演化更新；不依赖离线批量训练语料，记忆库随会话流持续增长与重组。

**评测 / Evaluation**

- **任务领域**: 长期多会话对话理解与问答（multi-session dialogue QA）为主，覆盖多跳推理、时序推理、开放域知识、单跳事实检索与对抗性（不可答）问题五类；测试材料含真实长对话与基于流行美剧（Friends、The Big Bang Theory、The Office）的多方对话问答。属于用户/对话中心的长程记忆评测，而非 web 导航/具身/编码类智能体任务。
- **基准**: (1) LoCoMo：超长多会话对话数据集，平均约 9K tokens、最多 35 个会话，含 7,512 个 QA 对，分五类问题（single-hop / multi-hop / temporal / open-domain / adversarial）；(2) DialSim：源自三部美剧的长期多方对话问答，约 1,300 个会话、跨度五年、约 35 万 tokens、每会话 1,000+ 问题（含粉丝测验题与基于时序知识图谱生成的复杂问题）。嵌入模型 all-MiniLM-L6-v2，主实验 k=10。
- **报告增益**: 在 LoCoMo 上跨六个基础模型（GPT-4o-mini、GPT-4o、Qwen2.5-1.5B/3B、Llama3.2-1B/3B）评测，报告 F1 与 BLEU-1。亮点：(1) 多跳推理大幅领先——GPT-4o-mini 上 Multi-Hop F1 27.02 vs MemGPT 26.65/LoCoMo 25.02，但在多数模型上对最强基线实现「至少 2 倍」提升（如 Qwen2.5-3B：A-Mem Multi-Hop F1 12.57 vs MemGPT 5.07，约 2.5×；Llama3.2-3B：17.44 vs 5.32，约 3.3×）；(2) 时序推理提升尤为显著（GPT-4o-mini Temporal F1 45.85 vs MemGPT 25.52、LoCoMo 18.41）；(3) 综合排名（Ranking）A-Mem 在六模型上多数取得第 1.0（最优）；(4) 平均答题 token 长度仅约 1,200–2,520，相较 LoCoMo/MemGPT 的约 16,900 tokens 降低 85–93%；(5) DialSim：A-Mem F1=3.45，较 LoCoMo（2.55）提升约 35%、较 MemGPT（1.18）高约 192%，且 BLEU-1/ROUGE-L/ROUGE-2/METEOR/SBERT 全面领先。成本：每次记忆操作约 1,200 tokens、<$0.0003（商用 API）；处理耗时 GPT-4o-mini 约 5.4s、本地 Llama3.2-1B 约 1.1s。消融：去除链接生成+记忆演化（w/o LG&ME）Multi-Hop F1 仅 9.65，仅去 ME（w/o ME）为 21.35，完整 A-Mem 为 27.02，验证两模块互补且关键。
- **对比基线**: LoCoMo（原数据集配套的长上下文/全历史方法）、ReadAgent、MemoryBank、MemGPT（即 Letta 前身的分页记忆方法）。属于「无结构化组织的记忆/全上下文」类对比，未与基于知识图谱的图记忆（如 GraphRAG/HippoRAG/Zep）或学习型记忆控制方法直接对比。

**分析 / Analysis**

- **关键创新**: 将 Zettelkasten「卡片盒」原子笔记+灵活链接思想引入 LLM 智能体记忆，首创在「存储与组织层」具备能动性的 agentic memory：新记忆自动生成结构化语义属性、由 LLM 自主判定并建立跨记忆链接、并触发既往记忆的「记忆演化（更新上下文/关键词/标签）」，从而让记忆网络持续自组织、自精炼——突破了此前记忆系统「固定操作、固定结构、仅在检索层有能动性」的局限。
- **局限**: (1) 无真正的遗忘/衰减、去重与冲突消解机制，记忆基本只增不删，长期运行可能累积冗余/过时信息；(2) 不处理失败学习与个性化安全（无害化/隐私治理）维度；(3) 记忆演化与链接生成依赖多次 LLM 调用，质量受 LLM 与提示模板稳定性影响，对模糊交互可能误判链接；(4) 评测局限于长对话 QA（LoCoMo/DialSim），未覆盖 web/具身/编码等智能体决策任务，亦未与图数据库类强基线（GraphRAG/Zep）正面比较；(5) 在 GPT 强模型上对 Open-Domain/Adversarial 等「简单事实检索」任务相对基线优势有限（强基线靠预训练知识占优）。
- **与其他工作关系**: 属本研究「B. 情景记忆与检索（Episodic memory & retrieval）」簇。其核心区别于「智能体中心、失败驱动」的 A 簇（A2 Retroformer、A3 CLIN、A5 ExpeL、A6 ReasoningBank）：A 簇蒸馏成功/失败轨迹为可迁移策略/技能以自我提升，A-MEM 则不区分成败、面向长期对话把交互沉淀为可演化的语义笔记网络以增强长程多跳问答。相对同样引入图结构的记忆系统：A-MEM 用 LLM 自主生成的「笔记+动态链接」涌现网络，强调记忆可演化更新（区别于 Zep/Graphiti 的时序知识图、HippoRAG 的 PPR 检索这类预定义图 schema）；相对 MemGPT（分页/操作系统式记忆）与 MemoryBank（Ebbinghaus 衰减），A-MEM 强调结构组织与演化而非分页或遗忘曲线。它常被后续工作（如 Mem-α、MEMAUDIT、MemMark、Pancake 等）当作标准记忆基线/被审计对象，可与学习型记忆控制（Memory-R1、Mem-α）正交组合——后者用 RL 学习「何时/如何存取」，A-MEM 则提供启发式但可演化的组织底座。
- **可复现性**: 可复现性良好：官方开源系统库 agiresearch/A-mem（Python，MIT 许可，约 1k stars，基于 ChromaDB + all-MiniLM-L6-v2，支持 OpenAI/Ollama 后端），另提供论文结果复现专用仓库 WujiangXu/AgenticMemory 与 A-mem-sys；所用基准 LoCoMo、DialSim 均公开。流程清晰、依赖常见组件，社区采用度高（约 603 引用、被多项后续工作作为基线）。不确定性主要来自对商用/本地 LLM 版本与提示模板的敏感性。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式流程）。笔记生成、链接判定、记忆演化、检索均为固定的 LLM 提示驱动启发式管线，不使用 RL/训练来学习「何时/存什么/如何更新/如何检索」的记忆管理策略本身。处于 2025–26「学习型记忆控制」分水岭中的启发式一侧（与 Memory-R1、Mem-α、Mem-π 等学习型方法相对，后者常把 A-MEM 列为非学习型基线）。
- **记忆主体**: 用户/对话中心（user-/conversation-centric）。记忆的是长期多会话对话中的历史信息（事实、事件、关系），目的是让智能体在后续交互中保持长程一致性、支持多跳/时序问答与个性化对话理解，而非记录智能体自身的操作经验以自我进化（区别于 Voyager/ReasoningBank 的 agent-centric）。
- **多智能体记忆**: 单智能体（single-agent）。记忆网络服务于单个智能体的长期对话记忆，未涉及多智能体间共享/路由记忆（区别于 G-Memory、MIRIX 等多智能体记忆分层/路由方案）。
- **时序推理支持**: 部分支持但非显式时序模型。每条笔记带时间戳 t_i 并参与笔记构建，LoCoMo 评测专设「时序推理（temporal reasoning）」类问题且 A-MEM 在该类提升最显著（如 GPT-4o-mini Temporal F1 45.85 远超基线）。但它不像 Zep/Graphiti 那样显式建模事实有效性窗口、事件先后或双时态（bi-temporal）边——时间信息以属性形式参与语义编码与检索，而非独立的时序推理引擎。
- **模态**: 纯文本（text-only）。交互内容、笔记属性与检索均为文本/文本嵌入，无视觉、截图、视频或具身多模态记忆。
- **过度个性化/记忆安全风险**: 未涉及。论文不讨论有害/过时/侵入性/谄媚式记忆、隐私治理或过度个性化风险，也无相应的记忆安全/治理机制；缺乏遗忘与冲突消解更使长期累积的过时信息可能持续影响输出（属潜在风险但未被本工作分析）。
- **冲突/矛盾处理**: 弱处理且非显式。记忆演化阶段会用 LLM 在新旧记忆并置下更新近邻旧笔记的上下文/关键词/标签，可在一定程度上随新信息刷新旧表征；但没有专门的矛盾/冲突事实检测与消解逻辑，也无版本化或事实失效标记（区别于 Memory-R1 的 UPDATE 操作或 MEMTRACK 类显式冲突追踪）。相互矛盾的记忆可能并存。
- **token成本/延迟证据**: 有明确量化证据。每次记忆操作约 1,200 tokens，平均答题 token 长度约 1,200–2,520，相较全上下文类基线（LoCoMo/MemGPT 约 16,900 tokens）降低 85–93%；单次记忆操作成本 <$0.0003（商用 API）。处理延迟：GPT-4o-mini 约 5.4s/次、本地 Llama3.2-1B（单 GPU）约 1.1s/次。检索时间随规模近乎恒定：1,000→1,000,000 条记忆时仅由约 0.31μs 增至约 3.70μs；空间复杂度 O(N) 与基线一致，无额外存储开销。

**不确定字段 / Uncertain**

- 发表venue (`venue`)


<a id="b5-think-in-memorytim中文记忆中思考在记忆中思考完整标题think-in-memory-recalling-and-post-thinking-enable-llms-with-long-term-memory"></a>

### B5 Think-in-Memory

*Think-in-Memory（TiM；中文「记忆中思考/在记忆中思考」，完整标题：Think-in-Memory: Recalling and Post-thinking Enable LLMs with Long-Term Memory）*


**基本信息 / Provenance**

- **年份**: 2023（arXiv 预印本，2023-11-15 首次公开，arXiv:2311.08719 v1）。论文 PDF 模板内署「2024」并套用 ACM 投稿格式占位符（doi/isbn 均为 XXXX 占位），暗示作者曾按 ACM 格式投稿，但无确认的正式收录证据。
- **作者/机构**: Lei Liu（刘磊，第一作者）、Xiaoyan Yang、Yue Shen、Binbin Hu、Zhiqiang Zhang、Jinjie Gu、Guannan Zhang，共 7 位作者。第一作者 Lei Liu 隶属香港中文大学（深圳）CUHK-Shenzhen 与蚂蚁集团（Ant Group），其余作者均为蚂蚁集团（Ant Group）。属工业界（蚂蚁集团）主导的研究。
- **论文链接**: https://arxiv.org/abs/2311.08719
- **引用数**: 约 50 次引用（Semantic Scholar，CorpusId 265212826，截至调研日 2026-06）。作为 2023 年较早的「存储思考而非原始文本」长期记忆工作，常被后续记忆综述与系统（CAIM、MemoryART、Memory-R1、Mem-α、SAGE 等）引用为相关工作/早期代表。

**记忆分类 / Taxonomy**

- **记忆类型**: 情景记忆与语义记忆的混合，偏语义化。它不保存原始对话逐字文本（情景细节），而是把每轮对话提炼为「归纳性思考（inductive thought）」——即实体-关系三元组形式的语义命题（E_head, r_i, E_tail），存入外部记忆。对应 CoALA 框架中的情景/语义记忆（半结构化语义事实），服务于长期多轮人机对话。本质是「把推理结论（thought）而非事件原文（event）作为记忆」，类比人类元认知（metacognition）中「大脑保存思考而非事件细节」。
- **记忆结构**: 持续增长的哈希表（hash table）作为记忆缓存 M：键为局部敏感哈希（LSH）索引、值为单条思考（thought）。相似思考被分配相同哈希索引、归入同一「组（group）」。每条思考以二元组 (H_idx, T) 存储，H_idx = F(T) 为哈希索引。思考内容采用关系三元组（head entity, relation, tail entity）的归纳命题文本表示，可视为轻量级知识图谱式语义记忆，但底层用 LSH 哈希分桶组织（而非显式图遍历）。非向量库默认结构、非层级分页、非笔记网络。
- **存储后端**: 外部记忆缓存（external memory cache），实现为一个不断增长的键值哈希表（LSH 索引→思考组）。嵌入向量由 LLM 智能体本身产生（用于 LSH 随机投影与组内相似度计算）。LLM 后端为可插拔的开源/闭源模型（实验用 ChatGLM-6B、Baichuan2-13B；声称兼容 ChatGPT）。资源受限场景下用 LoRA（rank=16，10 epochs）对 LLM 做参数高效微调以适配多轮对话。无特定商用向量数据库（如 FAISS/Chroma）绑定，检索靠自实现的 LSH 随机投影。
- **持久化**: 外部持久化存储（durable external store）。思考长期保存在外部哈希表记忆缓存中，跨多轮/多日对话持续累积与演化；检索结果以临时上下文注入提示，不改写 LLM 主体参数。可选的 LoRA 微调会改动少量适配参数（半参数化成分），但记忆本体仍是外部非参数化存储。

**核心机制 / Mechanisms**

- **写入/编码**: 采用「后思考（post-think）」两阶段写入而非逐字保存原始对话。流程：(1) 在生成回答后，LLM 智能体对当前 (Q, R) 问答对进行后思考（post-thinking），自生成新的「归纳性思考（inductive thought）」——定义为表达两实体间关系的文本，满足关系三元组 (E_head, r_i, E_tail)（E_head 经关系 r_i 连接 E_tail，r_i∈[0,N]）。思考由 LLM 通过少样本上下文学习（few-shot prompt）抽取（论文图 3 给出生成思考的提示模板；也提及可用 OpenIE 等预训练开放信息抽取模型作为替代方案）。(2) 每条思考 T 计算 LSH 哈希索引 H_idx=F(T)，以 (H_idx, T) 形式插入（insert）记忆哈希表，相似思考进入同一组。关键创新在于「存的是 LLM 自己推理出的思考（结论/关系），而非历史原文」，从而下次直接召回结论、避免对原文重复推理（消除「重复推理导致的不一致/有偏思考」）。
- **检索机制**: 两阶段检索（先 LSH 分桶、后组内相似度），在「回答生成前」的召回阶段执行：阶段 1（LSH-based Retrieval）——对新查询 Q 由 LLM 得到嵌入向量 x，用局部敏感哈希函数 F(x)=argmax([xR; -xR]) 计算其哈希索引，其中 R 为 (d, b/2) 的随机投影矩阵、b 为记忆分组数；该哈希索引依 LSH 性质指向最近的相似思考组。阶段 2（Similarity-based Retrieval）——只在命中的那一组内部计算查询与各思考的成对相似度，取 top-k 思考作为相关历史召回（主实验 top-5；top-k 分析覆盖 k=1…10）。核心优势：相似度只在组内而非全量记忆上计算，检索更高效（实测记忆长度 140 时单次检索 0.5305ms vs 全量成对相似度基线 0.6287ms，约省 0.1ms/约 16%）。召回准确率随 k 增大提升：top-1>0.7、top-10 达 0.973。
- **反思/巩固**: 其「后思考（post-thinking）」本身即一种反思/抽象：在每轮回答后，LLM 对问答对进行推理、把原始交互抽象为更高层的归纳性关系命题（thought），而非保存事件原文——类比人类元认知中「保存思考而非事件细节」。这是 raw→insight 的转化：把一次性推理结论沉淀为可复用记忆，避免今后对同一历史的重复推理与由此产生的不一致/有偏思考。除写入时的后思考外，记忆组织阶段还通过 forget/merge 操作对既有思考做持续巩固（去矛盾、合并同实体思考），实现思考的动态更新与演化。但它不做跨多条经验的周期性高层「反思摘要」（如 Generative Agents 的 reflection tree），抽象粒度停留在单轮问答→三元组命题。
- **遗忘/更新**: 提供三种显式记忆组织操作（在同一哈希组内进行）：Insert（插入新思考）、Forget（移除不必要/矛盾的思考，论文图 4 给出遗忘提示模板）、Merge（合并相似思考，如同一 head entity 的思考，论文图 5 给出合并提示模板）。这三种操作共同支持记忆的动态更新与演化，是 TiM 相对此前机制（SCM、RelationLM、LongMem、MemoryBank 多仅支持 insert/read，MemoryBank 另含 forget）的组织能力优势——TiM 是表 1 中唯一三种操作（insert+forget+merge）齐备者。遗忘/合并由 LLM 提示驱动（启发式），未实现基于时间的自动衰减曲线（如 Ebbinghaus）。
- **经验回放 (核心主题)**: 属「用户/对话历史复用」型，且复用单位是「思考（推理结论）」而非原始轨迹或可执行技能。机制为：把历史多轮对话中 LLM 自生成的归纳性思考沉淀为可检索记忆，新查询到来时通过 LSH+组内相似度召回相关历史思考、注入提示以生成更准确连贯的回答。其复用价值在于「一次推理、长期复用结论」：避免对同一历史在不同问题下重复推理，从而消除重复推理引起的不一致/有偏思考，提升长期对话的回答正确性与上下文连贯性。区别于 Voyager/ReasoningBank 那类把成功/失败动作轨迹蒸馏为可迁移技能/策略的智能体自我提升式复用——TiM 复用的是对话语义层面的推理结论（关系三元组），服务长期人机对话一致性，而非决策类任务的策略迁移。

**学习维度 / Learning**

- **学习范式**: 以非参数化（non-parametric）为主、含可选半参数成分的混合。记忆本体为外部哈希表中的思考累积与重组，全程非参数、提示驱动；同时提供可选的 LoRA 参数高效微调（rank=16、10 epochs）以把 LLM 适配到多轮对话风格（半参数化适配）。但记忆的存取/更新策略本身不经梯度学习，属启发式管线。
- **失败学习 (核心主题)**: 不涉及失败学习。TiM 面向长期人机对话的记忆一致性，不区分成功/失败经验，没有失败检测、失败轨迹反思、负例记忆或避坑规则机制。其 forget 操作针对的是「矛盾/不必要思考」（一致性维护），并非从任务失败中提炼教训。它要解决的核心痛点是「重复推理导致的不一致/有偏思考（inconsistent reasoning paths）」，属推理一致性问题而非错误驱动学习——这与 A 簇 Reflexion/Retroformer/CLIN/ExpeL/ReasoningBank 等失败驱动方法形成鲜明对照。
- **技能/程序归纳**: 否。不归纳可复用的技能/工作流/过程（procedure），无技能库与技能调用机制。它归纳的是语义层面的关系三元组思考（事实/关系命题），属知识性记忆而非可执行操作技能。
- **在线 vs 离线**: 在线（online）增量构建。记忆随每轮对话即时进行：回答前召回、回答后后思考并插入新思考，并按需触发 forget/merge 组织更新；不依赖离线批量训练语料，记忆随对话流持续增长与演化（可选的 LoRA 适配为一次性离线微调，与在线记忆累积相互独立）。

**评测 / Evaluation**

- **任务领域**: 长期多轮人机对话理解与回答（multi-turn / long-term dialogue），覆盖开放域闲聊与垂直领域，且支持中英双语；垂直领域含影视/音乐/旅游知识对话与真实医疗问诊对话。落地为医疗智能体应用（TiM-LLM，辅助医生给出诊断/治疗建议）。属用户/对话中心的长期记忆评测，未覆盖 web 导航/具身/编码/GUI 等智能体决策任务。
- **基准**: 三个数据集：(1) KdConv——中文多领域知识驱动对话基准（基于知识图谱），约 4.5K 对话、86K 话语，含影视/音乐/旅游三域，平均每对话 19 轮；(2) GVD（Generated Virtual Dataset）——源自 MemoryBank 的长期对话数据集，15 个 ChatGPT 虚拟用户跨 10 天对话（中英双语），测试集人工构造 194 个查询问题（中英各 97）；(3) RMD（Real-world Medical Dataset）——作者自建真实医疗问诊数据集，含 1,800 段对话，测试集 80 段用于评估诊断准确性。LLM 后端：ChatGLM（6.2B）与 Baichuan2（13B）。主实验召回 top-5 思考。
- **报告增益**: 评测采用人工评分三指标（先打乱模型来源做盲评）：Retrieval Accuracy（召回是否命中，0/1）、Response Correctness（回答正确性，0/0.5/1）、Contextual Coherence（上下文连贯性，0/0.5/1）。对比「无记忆」与 SiliconFriend（MemoryBank 的存原文记忆机制）基线。代表性结果（表 2，越高越好）：GVD-中文（ChatGLM）TiM 的 Response Correctness 0.605 vs SiliconFriend 0.418（约 +0.187），Coherence 0.665 vs 0.428（约 +0.237），Retrieval Acc 0.850 vs 0.840；GVD-英文 TiM Correctness 0.450 vs 0.438、Coherence 0.735 vs 0.680。KdConv（无记忆基线无召回指标）：Baichuan2-影视 TiM Correctness 0.743 vs 无记忆 0.413、Coherence 0.870 vs 0.413，Retrieval Acc 高达 0.913；ChatGLM-影视 TiM Correctness 0.827 vs 0.657。RMD 医疗：ChatGLM TiM Correctness 0.843 vs 无记忆 0.806、Coherence 0.943 vs 0.893、Retrieval Acc 0.900；Baichuan2 TiM Coherence 0.663 vs 0.538、Retrieval Acc 0.873。效率：记忆长度 140 时单次检索 0.5305ms vs 全量成对相似度基线 0.6287ms（约省 16%/0.1ms）。Top-k 召回准确率：top-1>0.7、top-10=0.973（KdConv-旅游）。总体：TiM 在三数据集、两 LLM、多语言/多主题上全面优于对应基线，尤以连贯性与中文场景提升显著。
- **对比基线**: (1) 无记忆机制（直接回答，无长期记忆）；(2) SiliconFriend——MemoryBank（Zhong et al. 2023）配套的经典存原文记忆机制，把原始对话文本存入记忆并支持读取操作。对比维度集中于「存原文、仅读写」类记忆 vs TiM「存思考、可 insert/forget/merge」。论文未与向量库 RAG、知识图谱图记忆（GraphRAG/HippoRAG/Zep）或学习型记忆控制方法直接对比。组织能力对比（表 1）另列 SCM、RelationLM、LongMem、MemoryBank 作为定性参照。

**分析 / Analysis**

- **关键创新**: 提出「在记忆中思考」范式：把 LLM 自生成的「思考（推理结论，以关系三元组形式的归纳命题）」而非历史原始文本作为长期记忆，通过「回答前召回思考 + 回答后后思考并更新记忆」两阶段，使 LLM 召回的是已推理好的结论、无需对同一历史重复推理——从根本上消除「重复推理导致的不一致/有偏思考（inconsistent reasoning paths）」。并首次为记忆组织配齐 insert/forget/merge 三操作以支持思考的动态更新与演化，同时引入局部敏感哈希（LSH）实现「组内相似度」的高效长程检索，且设计为 LLM-agnostic 可插拔模块。
- **局限**: (1) 评测高度依赖小样本人工主观评分（盲评，三个 0/0.5/1 级指标），无 F1/BLEU 等自动指标的大规模量化，可比性与可复现性受限；(2) 测试规模偏小（GVD 测试仅 194 问、RMD 测试仅 80 段、检索效率实验记忆长度仅 140），缺乏百万级记忆扩展性验证；(3) 无作者官方开源代码，社区仅有第三方简化复现，复现成本高；(4) 思考质量完全依赖 LLM 的关系抽取与 forget/merge 提示判定，对模糊对话可能误抽/误并/误删；无基于时间的自动遗忘曲线、无显式冲突版本化；(5) 仅限长对话 QA 领域，未覆盖 web/具身/编码等决策任务，亦未与图记忆、向量 RAG、学习型记忆控制强基线正面比较；(6) LSH 随机投影分桶可能漏召（同义但跨桶的思考）。
- **与其他工作关系**: 属本研究「B. 情景记忆与检索（Episodic memory & retrieval）」簇。与同簇对比：相对 B2 MemoryBank（其 SiliconFriend 正是 TiM 的主要基线，存原文+Ebbinghaus 遗忘曲线），TiM 改为「存思考而非原文」并新增 merge 操作、引入 LSH 分桶检索；相对 B1 Generative Agents（存观察+周期性 reflection 树+recency/importance/relevance 检索），TiM 抽象粒度更细（单轮→三元组）、无显式重要性评分、用 LSH 而非加权检索；相对 B4 A-MEM（Zettelkasten 笔记网络+记忆演化+ChromaDB 向量检索），二者都做「raw→语义结构」转化与就地更新，但 A-MEM 用 LLM 自主链接的笔记网络与向量库，TiM 用关系三元组思考+LSH 哈希表，且 A-MEM 仅在检索层有演化、TiM 显式提供 forget/merge；相对 B3 MemGPT（OS 式分页原文记忆），TiM 关注「思考内容」而非分页调度。与 A 簇（A2 Retroformer/A3 CLIN/A5 ExpeL/A6 ReasoningBank 等失败/经验驱动自我提升）正交：TiM 不区分成败、不蒸馏技能，仅维护对话推理结论的一致性。后续工作（CAIM、MemoryART、Memory-R1、Mem-α、SAGE 等）常把 TiM 列为「存思考型」早期相关工作；其启发式 insert/forget/merge 与 Memory-R1/Mem-α 的 RL 学习型记忆控制形成代际对照。
- **可复现性**: 可复现性中等偏弱。无作者官方开源实现（蚂蚁集团未公布官方代码），社区仅有第三方非官方简化复现（tractorjuice/TiMSystem 等，星标极低、为 demo）。所用基准中 KdConv、GVD（源自 MemoryBank）公开，但 RMD 医疗数据为自建且未公开；核心评测依赖人工主观盲评，难以严格复算。方法本身（LSH 随机投影 + 提示驱动思考抽取/组织）原理清晰、易于按论文重写，但缺乏官方权重/脚本/提示全集与统一自动指标，社区采用度有限（约 50 引用，多作相关工作引用而非作为标准基线代码被广泛运行）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式管线）。思考抽取、insert/forget/merge 组织、LSH 分桶与 top-k 召回均为固定的提示驱动启发式或确定性算法，不使用 RL/训练来学习「何时存/存什么/如何更新/如何检索」的记忆管理策略本身。处于 2025–26「学习型记忆控制」分水岭中的启发式一侧（与 Memory-R1、Mem-α 等用 RL 学习记忆操作策略的方法相对，后者亦常引用 TiM 作为早期非学习型记忆代表）。可选 LoRA 仅用于适配对话风格，不学习记忆控制策略。
- **记忆主体**: 用户/对话中心（user-/conversation-centric）。记忆的是长期多轮人机对话中的历史信息（被抽象为思考/关系三元组），目的在于让 LLM 在后续对话中保持长程一致、避免重复推理、提升回答正确性与连贯性（如医疗问诊中记住既往症状），而非记录智能体自身的操作经验以自我进化（区别于 Voyager/ReasoningBank 的 agent-centric）。
- **多智能体记忆**: 单智能体（single-agent）。记忆缓存服务于单个 LLM 智能体与用户的长期对话，未涉及多智能体间共享/路由记忆（区别于 G-Memory、MIRIX 等多智能体记忆分层/路由方案）。
- **时序推理支持**: 弱/不显式。TiM 沿对话流（conversation stream）按轮次顺序累积思考，隐含时间顺序；但不显式建模事实有效性窗口、事件先后或双时态（bi-temporal）边，也未设专门时序推理评测维度（区别于 Zep/Graphiti 的时序知识图）。其评测关注医疗问诊等需「跨轮记住既往信息」的长程一致性，而非时间区间推理。
- **模态**: 纯文本（text-only）。对话内容、思考（关系三元组命题）、嵌入与检索均为文本/文本嵌入，无视觉、截图、视频或具身多模态记忆。
- **过度个性化/记忆安全风险**: 未涉及。论文不讨论有害/过时/侵入性/谄媚式记忆、隐私治理或过度个性化风险。其 forget 操作可移除「矛盾/不必要思考」、merge 可去重，间接有助于抑制陈旧/重复信息累积，但并非面向记忆安全/隐私治理的机制设计，亦无相关基准评测（如 OP-Bench）。医疗落地场景下的数据隐私问题未被系统讨论。
- **冲突/矛盾处理**: 部分支持且为显式操作之一。Forget 操作明确以「移除矛盾的思考（contradictory thoughts）」为目标，Merge 操作合并同实体/相似思考以消冗——二者由 LLM 提示判定，可在一定程度上随对话演进消解矛盾、合并重复，使记忆中思考保持一致。但无版本化、无事实失效时间标记、无系统化的冲突检测追踪逻辑（区别于 Memory-R1 的显式 UPDATE 操作或 MEMTRACK 类冲突追踪），冲突消解质量取决于 LLM 与提示稳定性。
- **token成本/延迟证据**: 部分量化，主要在检索时延维度。检索效率：固定记忆长度 140、记忆内容固定下，TiM 单次检索 0.5305ms vs「全量成对相似度」基线 0.6287ms，约降低 0.1ms（约 16%），因相似度只在 LSH 命中组内而非全量记忆上计算。优势随记忆规模增大而扩大（思考 vs 原文也更短，隐含上下文 token 更省），但论文未给出系统化的 token 数/美元成本对比或大规模（百万级）时延曲线，亦未量化「存思考替代存原文」带来的上下文 token 节省比例。整体效率证据相对早期、规模有限。

**不确定字段 / Uncertain**

- 代码链接 (`code_url`)
- 发表venue (`venue`)


<a id="b6-larimar大语言模型情景记忆控制架构"></a>

### B6 Larimar

*Larimar（大语言模型情景记忆控制架构）*


**基本信息 / Provenance**

- **年份**: 2024（arXiv 预印本于 2024 年 3 月 18 日首次公开，v4 修订于 2024 年 8 月）
- **作者/机构**: Payel Das、Subhajit Chaudhury、Elliot Nelson、Igor Melnyk、Pin-Yu Chen 等共 12 位作者，全部来自 IBM Research（IBM 研究院；其中 Sihui Dai 同时隶属普林斯顿大学）。Payel Das 与 Subhajit Chaudhury 为共同第一作者。
- **发表venue**: ICML 2024（第 41 届国际机器学习大会，PMLR v235，DBLP: conf/icml/DasCNMSDLKC0DC24）
- **论文链接**: https://arxiv.org/abs/2403.11901
- **代码链接**: https://github.com/IBM/larimar （官方 IBM 仓库，Apache-2.0 许可，约 35 个 star、1 个 fork，提供参考实现、训练/评测脚本与单事实编辑演示 notebook）

**记忆分类 / Taxonomy**

- **记忆类型**: 情景记忆（episodic memory）为主——以分布式、可一次性写入的外部联想记忆显式存储“事实/事件”级编码（episode），用于对 LLM 进行测试时的知识更新与编辑；架构上属于受脑启发的“互补学习系统”，即快速学习的记忆模块与慢速学习的 LLM 解码器配合。不属于 CoALA 的语义/程序/工作记忆类别，本质是参数外置但可微训练的情景型记忆。
- **记忆结构**: 固定大小的记忆矩阵（memory matrix）M ∈ R^{K×C}，K 行（论文实现中 K=512，特征维 C=768），每行存储一个被编码后的 episode 潜向量；本质是 Kanerva Machine 风格的分布式/联想记忆（distributed associative memory），通过读/写权重矩阵 W、W0（N×K）与潜空间编码交互。多个记忆块 M_i 还可分层堆叠（用于长上下文的递归层级记忆）。
- **存储后端**: 模型内部的潜空间记忆矩阵（latent memory matrix），由 BERT-large 编码器编码、经端到端梯度训练学习得到；记忆读出经线性投影 W_M 转换为各层共享的 KV cache 注入 GPT 解码器。不依赖外部向量数据库或图数据库；另可选配一个“范围检测器”（scope detector），其外部版本 ESD 用 MiniLM 句向量做最近邻余弦相似度检索。
- **持久化**: 外部、可持久且可动态更新的潜空间记忆（区别于 in-context 临时缓存，也区别于把知识烘焙进 LLM 权重的参数化方法）。记忆作为解码器的“显式状态”：推理时把新事实一次性写入记忆矩阵，更新后的记忆条件化解码；记忆是 LLM 无关（LLM-agnostic）的外挂模块，可跨会话/编辑序列累积，并支持选择性遗忘以回收容量。

**核心机制 / Mechanisms**

- **写入/编码**: 写入（write）为一次性、无梯度的最小二乘求解：将一个 episode X={x1..xN}（如把编辑 prompt 与新目标对象拼接成的事实句）经编码器 e（BERT-large）编码为潜向量 Z=e(X)，加噪 Z_ξ=Z+ξ 后，计算寻址权重 W0=Z_ξ·M0†（M0 为训练得到的先验记忆，†表示伪逆），再求后验记忆 M=W0†·Z_ξ，即求解 min_M ||Z_ξ − W0·M||²。源自 Kanerva Machine / 广义伪逆记忆（GPM, Pham et al. 2021）的确定性重构：把贝叶斯记忆更新改写为线性系统的最小二乘解，借矩阵伪逆高效完成。整个写入无需对 LLM 做梯度更新或事实定位（fact tracing）。
- **检索机制**: 读取（read）：给定查询编码 Z=e(X_query)，计算寻址权重均值 W=Z·M†，按 W~N(W, σ_W²I) 采样后得到记忆读出潜向量 Z_read=W·M；该读出经线性投影 W_M 转为各层 KV cache 注入解码器，实现记忆条件化解码（每生成一个 token 仅需 O(1) 记忆）。可选“范围检测器”：外部 ESD 用 MiniLM(384维) 句向量做 1-近邻余弦相似度判定查询是否 in-scope（在 EasyEdit 上等错误率 2.9%、F1=0.974），in-scope 则用记忆读出条件化解码，否则走无条件解码；内部 ISD 用 Larimar 编码器训练二分类器。长上下文则在潜记忆空间做递归搜索：将文本切成 T 块各存入独立记忆 M_i，逐层读出并构造更高层后继记忆再次查询，直至读出数量贴合训练上下文窗口。
- **反思/巩固**: 无传统意义的“反思/经验抽象/摘要成更高层洞见”机制——Larimar 不把原始经验提炼为反思性见解或技能。其“整合”体现在记忆的数学性质上：序列写入公式 M_i=M_{i-1}+α_i·C_i^{-1}·W_i^T·(Z_i−W_i·M_{i-1}) 始终维持对全部已写编码 Z_{0:i} 的最小二乘最优解，相当于把新事实增量整合进固定大小记忆而保持对历史的最优压缩（实验显示能把多于 K 的事实压入 K 行记忆，K=512 时 1024 条仍达 82% 改写准确率）。这种整合是闭式代数更新，而非基于 LLM 的语义反思或摘要。
- **遗忘/更新**: 支持显式的选择性遗忘/删除：用与写入相同的序列更新公式，但取 α_i=−1，配合固定参考记忆 M(ref) 重算原写入键 W_iwrite 来定位并移除该事实，使记忆变为去掉该编码后的最小二乘解（公式 6）。实践中常把被忘事实替换写入答案为“unknown”的同一事实。更新（add）取 α_i=+1。无需负例微调，也无需重训练。
- **经验回放 (核心主题)**: 不适用于强化学习式“轨迹回放/技能复用”范式——Larimar 面向事实知识编辑而非智能体行为自我改进，因此没有经验回放缓冲、示例提示复用或蒸馏策略等机制。最接近的“经验复用”是测试时把同一事实的 1–2 个改写（rephrase）额外写入记忆以提升泛化（CounterFact 上泛化从 88.4 升至 93.6；ZsRE 上从 70.4% 升至 82.2%），以及长上下文中把历史记忆读出递归用于构造后继记忆。整体上记忆服务于“知识更新”而非“行为经验回放”。

**学习维度 / Learning**

- **学习范式**: 混合（hybrid），但与多数智能体记忆系统方向相反：记忆模块（编码器 e、联想记忆 M、解码器 d）通过端到端梯度下降在通用语料（700+ 万条 WikiText 64-token 片段）上一次性预训练（参数化/慢学习部分）；而部署后的每一次知识编辑/写入/遗忘都是无梯度、一次性的闭式最小二乘更新（非参数化/快学习部分）。即“离线参数化训练得到记忆能力 + 在线非参数化一次性写入更新”。
- **失败学习 (核心主题)**: 不涉及——Larimar 不检测或利用智能体执行失败/错误轨迹来学习（无自我反思失败、无失败模式记忆、无负例驱动规则）。它处理的是“错误/过时事实”的纠正：通过把新（正确）事实一次性写入记忆来覆盖旧知识，以及通过写入“unknown/空响应”实现选择性遗忘与信息泄露防护（对改写攻击的成功率 17.6%，优于 ROME 29% 与 MEMIT 49.3%）。这属于知识纠错而非智能体失败学习。
- **技能/程序归纳**: 不支持——不从经验中归纳可复用技能、工作流或程序。Larimar 的能力固定为“写入/读取/生成/序列写入/遗忘”五种代数记忆操作，记忆内容是事实级编码而非可调用的技能或子程序。
- **在线 vs 离线**: 两者结合：记忆-解码器架构在大规模通用语料上离线（offline）端到端预训练一次；之后所有知识编辑、序列编辑、批量编辑、遗忘均为在线（online）、逐事实/逐 episode 的一次性写入，无需访问编辑数据进行训练。强调“实时（real-time）测试时适应”。

**评测 / Evaluation**

- **任务领域**: 知识编辑/事实编辑（factual knowledge editing）：单事实编辑、序列编辑、批量编辑、选择性遗忘、信息泄露防护，以及长输入上下文的事实召回泛化。属于 QA/事实纠错领域，而非网页导航、具身、对话或游戏类智能体任务。
- **基准**: CounterFact（21,919 条反事实编辑记录，取前 2000 条评测）、ZsRE（关系抽取阅读理解 QA，序列编辑取 ZsRE 验证集 200/1000/511 条事实并配 5–20 个改写）、EasyEdit 框架（用于墙钟时间与范围检测器评测）、CNN Fast Facts 2021–2023（长上下文事实召回，base 解码器未见过的数据）。
- **报告增益**: 速度：在 EasyEdit 框架单 A100 上 10 次编辑的墙钟时间，Larimar 1.1s（GPT-2）/1.7s（GPT-J），相对 ROME（4.8s/13.9s）与 GRACE（13.9s/19.3s）快约 4–10 倍（摘要称 8–10 倍）。单事实编辑（CounterFact）：Larimar-1.3B 编辑成功率 S=100.0、Larimar-6B S=99.6，邻域特异性 80.4（GPT-J 上优于 ROME 的 78.9），写入 1–2 条改写后改写泛化从 88.4 升至 93.6。序列编辑（ZsRE，1000 次连续编辑后的编辑保持率 ERR）：Larimar-1.3B 0.97、Larimar-6B 0.92，对比 GRACE 0.93、MEND 0.27；在含重复事实的数据集上 Larimar ERR=0.98 vs GRACE 0.96 vs SERAC/DEFER 0.31，且约快 10 倍以上；约 600 次编辑后泛化超过 GRACE。批量编辑：≤512 条（=记忆容量 K）改写准确率近 100%，1024 条降至 82%（优于 MEND/ROME，逊于 MEMIT）。选择性遗忘（K=512）：被忘事实召回降至约 0.0–0.03，同时保留事实召回 0.86–0.997，远优于 Llama2-13B 6-shot 上下文学习（被忘 0.75/保留 0.77）。信息泄露防护：改写攻击成功率 17.6%（Larimar 单条）vs ROME 29.0% / MEMIT 49.3%。长上下文：128 条事实平均读取 0.36s vs Mistral-7B 1.44s，且召回随上下文增长不明显退化。困惑度：Larimar-1.3B 14.6、Larimar-6B 15.9（WikiText 1000 样本），表明加记忆几乎不损base性能。
- **对比基线**: 知识编辑基线：ROME、MEMIT、MEND/MEND-CF、KE/KE-CF、KN、FT/FT+L（微调类）、GRACE（序列编辑最强基线，外置 codebook 适配器）、SERAC/DEFER（批量编辑）、IKE 与 PROMPT（上下文编辑，需多示例）、Llama2-13B k-shot 上下文学习（遗忘对照）；长上下文对照含 Mistral-7B 等长上下文 LLM 与 Supersizing Transformer（记忆增强模型）。

**分析 / Analysis**

- **关键创新**: 首次提出并验证“在线、分布式地写入一个分层条件式生成记忆模型（Kanerva Machine / 伪逆记忆的确定性变体）”作为 LLM 测试时知识适应的方案：把记忆读/写转化为闭式最小二乘/伪逆求解，从而实现无需重训练或事实定位的一次性（one-shot）知识编辑、序列编辑、选择性遗忘与信息泄露防护，且架构简单、LLM 无关、速度快 8–10 倍。
- **局限**: 泛化（paraphrase generalization）仍有提升空间，常需额外写入 1–2 个改写来补偿；记忆容量固定为 K（实现中 512），批量编辑超过 K（如 1024）准确率下降到 82%，大批量编辑逊于 MEMIT；2K 规模写入时保留召回明显下降（如 ZsRE 降至 0.50–0.52）；只处理事实/episode 级知识，不涉及智能体行为、技能归纳或失败学习；范围检测器提升泛化会以牺牲邻域特异性为代价；长上下文依赖递归记忆搜索这一较为工程化的方案；记忆-解码器需先在大规模语料上端到端预训练（约 800 万条样本、8×A100 训练 10 epoch）。
- **与其他工作关系**: 属于本研究 B 簇（情景记忆与检索）的“参数外置但可微”的脑启发记忆代表，与 A 簇 Reflexion/Voyager 等“非参数、面向智能体自我改进”的经验型记忆方向显著不同：Larimar 面向用户/世界知识的事实编辑（user/world-knowledge-centric），而非智能体行为经验自改进（agent-centric）。它直接对标并对比知识编辑/模型编辑系列工作——ROME、MEMIT（定位-再写参数更新）、MEND（超网络）、GRACE（外置 codebook 序列编辑）、SERAC（带范围分类器的外存编辑，Larimar 的 scope detector 与之概念相似）、IKE/PROMPT（上下文编辑）；技术根基承接 Kanerva Machine（Wu et al. 2018）与广义伪逆记忆 GPM（Pham et al. 2021），并新增了序列写入与遗忘算子。相对其它记忆系统，它独特地把记忆做成可微训练且闭式更新的潜空间矩阵，而非向量库/知识图谱/原始缓冲。
- **可复现性**: 可复现性较好：官方开源 IBM/larimar（Apache-2.0），含训练脚本 train_larimar.sh、配置 config_train_larimar.yaml、评测脚本 eval.sh/eval_rephrase.sh 与单事实编辑演示 notebook，并提供训练数据 tarball 下载（IBM Box）与 larimar-1.3b 检查点说明；依赖 Python 3.10、PyTorch Lightning、DeepSpeed ZeRO-2。所用基准（CounterFact、ZsRE、EasyEdit、CNN Fast Facts）均为公开数据。仓库仅 1 名主要贡献者、约 35 star，社区采用度中等；6B 模型训练需 8×A100-80GB，重训成本较高但编辑/推理本身轻量。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否——记忆管理策略（何时写/读/遗忘）不是通过 RL 或训练学到的；写/读/遗忘是确定性的闭式代数（最小二乘/伪逆）启发式流程，α_i 取 +1/−1 由人工规则指定。仅记忆编码-解码能力是端到端训练得到，但管理“策略”本身非学习式。属于 2025–26 “学习式记忆控制”分代前的启发式管线一侧（但具备可微的记忆表征学习）。
- **记忆主体**: 以用户/世界知识为中心（knowledge/fact-centric，类似 user-centric 路线）：记忆存储外部事实知识以便对 LLM 进行更新、纠错、个性化与遗忘，而非记录智能体自身的执行经验来自我改进（非 agent-centric）。
- **多智能体记忆**: 单智能体/单模型架构——记忆为一个 LLM 解码器服务，不涉及多智能体共享或路由记忆（无 G-Memory/MIRIX 式的洞见/查询/交互分层）。
- **时序推理支持**: 不显式建模时间有效性或事件排序（无事实有效期窗口、事件日历等时间推理结构）。其“序列”概念指编辑到达的先后顺序，记忆维持对编辑序列的最小二乘最优解，可在序列中后期定位并删除较早写入的事实，但这是序列更新而非时间有效性推理。
- **模态**: 纯文本（text-only）
- **过度个性化/记忆安全风险**: 正面涉及记忆安全的删除/防泄露维度：提供选择性事实遗忘与信息泄露防护（写入空/‘unknown’响应以抵御改写攻击，攻击成功率 17.6% 显著低于 ROME/MEMIT），用于移除过时、敏感或不应生成的信息，体现“可控、可删除”的记忆治理；但未专门评测谄媚、侵入性或过度个性化等 OP-Bench 类风险。
- **冲突/矛盾处理**: 通过“覆盖式写入 + 选择性遗忘”处理事实冲突/过时：写入新（正确）事实即覆盖旧输出，需要时用 α_i=−1 删除旧事实并写入替代；序列更新公式保证记忆始终是对当前有效事实集合的最小二乘解。但没有显式的冲突检测/合并仲裁逻辑（不像 Memory-R1 的 UPDATE 或 MEMTRACK 那样判别矛盾），冲突解决依赖人工指定写入/遗忘操作。
- **token成本/延迟证据**: 有量化效率证据：编辑速度比 ROME/GRACE 快约 4–10 倍（10 次编辑墙钟时间 GPT-J 上 1.7s vs 13.9s/19.3s）；记忆条件化解码每 token 仅需 O(1) 记忆；写入改写到记忆比上下文示例更省——上下文长度不随事实数增长；长上下文场景 128 条事实平均读取 0.36s vs Mistral-7B 1.44s（约 4 倍），且因潜空间记忆处理，KV cache token 计算量低于基线。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)


<a id="b7-memoryos论文题名memory-os-of-ai-agent框架别名-memoryos--memory-operating-system由北京邮电大学-baijia-ai-团队开源仓库-bai-labmemoryos"></a>

### B7 MemoryOS

*MemoryOS（论文题名「Memory OS of AI Agent」；框架别名 MemoryOS / Memory Operating System；由北京邮电大学 BaiJia AI 团队开源，仓库 BAI-LAB/MemoryOS）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本 v1 于 2025-05-30 首次公开，编号 2506.06326，cs.AI；同年 8 月被 EMNLP 2025 主会接收，11 月正式发表）
- **作者/机构**: Jiazheng Kang（康嘉政，北京邮电大学 BUPT，第一作者）、Mingming Ji（季明明，腾讯 AI Lab）、Zhe Zhao（赵哲，腾讯 AI Lab）、Ting Bai（白婷，北京邮电大学，通讯作者，BaiJia AI 团队负责人）。主要单位为北京邮电大学与腾讯 AI Lab。
- **发表venue**: EMNLP 2025 主会（Conference on Empirical Methods in Natural Language Processing，2025-11-04 至 09 于中国苏州召开；被标注为 Oral 口头报告）。ACL Anthology id 2025.emnlp-main.1318，DOI 10.18653/v1/2025.emnlp-main.1318，页码 25961–25970，ISBN 979-8-89176-332-6。属于学术成果。
- **论文链接**: https://arxiv.org/abs/2506.06326（ACL Anthology 正式版：https://aclanthology.org/2025.emnlp-main.1318/）
- **代码链接**: https://github.com/BAI-LAB/MemoryOS（Python，Apache-2.0 许可，约 1.4k stars、141 forks，截至 2026-06）。提供 PyPI 包（pip install memoryos-pro）、MemoryOS-MCP（MCP 服务器）、ChromaDB 后端实现、Docker 镜像（ghcr.io/bai-lab/memoryos）与在线 Playground（baijia.online/memoryos）。
- **引用数**: 约 68 次引用（Semantic Scholar，CorpusId 279250574，influentialCitationCount 14，截至调研日 2026-06）；作为 2025 年 OS 启发式分层智能体记忆的代表性新作，影响力增长迅速。

**记忆分类 / Taxonomy**

- **记忆类型**: 以「用户中心」的长期情景/语义记忆为主：在 CoALA 框架下，短期记忆（STM）承载工作记忆性质的即时对话；中期记忆（MTM）以话题分段聚合属情景记忆；长期个人记忆（LPM，含用户画像/用户知识库/用户特质与智能体画像/特质）偏语义性的稳定知识与人设。整体横跨工作记忆与情景/语义记忆三层，不显式建模程序性记忆。
- **记忆结构**: 受操作系统内存管理（分段+分页，Multics/虚拟内存）启发的三级分层分级（hierarchical tiered）结构：①短期记忆 STM——固定长度的对话页队列（dialogue page，含 Q/R/时间戳与对话链 meta）；②中期记忆 MTM——分段分页（segmented paging）结构，同话题对话页聚合为 segment，每段含多页并由 LLM 生成话题摘要；③长期个人记忆 LPM——结构化画像，包括用户静态画像、用户知识库 User KB、90 维用户特质（User Traits）、智能体画像与智能体特质（均为定长 FIFO 队列）。属于多层级 + 分段分页 + 结构化画像的混合结构。
- **存储后端**: STM 为内存内的定长对话页队列。MTM/LPM 为外部持久化存储，默认实现以本地文件（JSON 数据目录 data_storage_path）落盘；嵌入向量用于语义相似度检索，默认嵌入模型为 BAAI/bge-m3（也支持 Qwen3-Embedding-0.6B、all-MiniLM-L6-v2）。官方另提供基于 ChromaDB 向量数据库的实现版本（memoryos-chromadb）。底层 LLM 通过 OpenAI 兼容接口调用（实验用 GPT-4o-mini、Qwen2.5-7B/3B）。
- **持久化**: 混合偏外部持久化：STM 为易失的内存队列；MTM 与 LPM 为外部持久化存储（文件/向量库），跨会话长期保留并随交互动态演化。非参数化——不修改底层 LLM 权重，所有记忆以外部结构化存储 + 上下文拼接形式存在。Letta 式的「智能体即有状态服务」在此体现为可落盘、可重载的用户/智能体记忆目录。

**核心机制 / Mechanisms**

- **写入/编码**: 写入分三层、以「分页 + 摘要提炼 + 结构化抽取」为核心，由 LLM 驱动生成元信息：①STM 写入——每轮交互原样（verbatim）封装为对话页 page_i={Q_i,R_i,T_i}，并由 LLM 两步生成对话链元信息 meta^chain：先判断新页与既有页的语境相关性以决定续链或重置，再对链上各页做摘要，保证短程上下文连贯；②MTM 写入——当 STM 队列满（默认长度 7）时按 FIFO 把最旧对话页迁入 MTM，依匹配分 F_score=cos(e_s,e_p)+Jaccard(K_s,K_p)（语义嵌入相似度 + 关键词 Jaccard 相似度，阈值 θ=0.6）归并入同话题 segment，段内容由 LLM 摘要；③LPM 写入——热度超阈 τ（默认 5）的 segment 触发画像更新：由 LLM 从对话页/段中抽取并增量更新 90 维用户特质（三类：基本需求与个性、AI 对齐维度、内容平台兴趣标签）、向 User KB 与 Agent Traits 抽取事实信息（定长 100 的 FIFO 队列）。因此写入兼具逐字保存（STM 页）、话题摘要（MTM 段）与结构化特质/事实抽取（LPM）三种编码形态。
- **检索机制**: 三层协同检索 F_Retrieval(STM,MTM,LPM|Q)：①STM 检索——直接取回队列中全部对话页作为最近上下文；②MTM 检索——受心理学记忆回忆机制启发的两阶段检索：先用匹配分（Eq.3，cos 语义 + Jaccard 关键词）选出 top-m 候选 segment（默认 m=5），再在段内按语义相似度选 top-k 对话页（GVD 上 k=5，LoCoMo 上 k=10）；检索后更新该段的访问计数 N_visit 与近因因子 R_recency；③LPM 检索——User KB 与 Agent Traits 各取语义相关度最高的 top-10 条作为背景知识，User Profile/Agent Profile/User Traits 全量纳入。三层结果与查询拼成最终提示交由 LLM 生成。检索同时驱动热度更新，与遗忘/晋级机制耦合。无 Generative Agents 式 recency·importance·relevance 三因子打分公式，但 MTM 的热度公式在「更新/驱逐」侧扮演类似角色。
- **反思/巩固**: 存在「整合/巩固」式的层级提炼，但非失败反思蒸馏范式。体现为：①对话链摘要——LLM 对 STM 链上各页做摘要生成 meta^chain，并判断语境连续性；②MTM 段摘要——LLM 对同话题对话页聚合生成 segment 话题摘要，实现情景信息的话题级抽象；③LPM 画像演化——LLM 从高热度段抽取并自主演化 90 维用户特质、用户知识库与智能体特质，实现「用户偏好/人设的持续提炼与进化」。这些整合由记忆迁移（STM→MTM→LPM）与热度阈值触发，是把原始对话逐级抽象为话题摘要再到稳定画像的过程，而非如 Reflexion/ExpeL 那样把失败经验蒸馏为可迁移决策规则。
- **遗忘/更新**: 具备较显式的热度驱动遗忘/晋级机制：MTM 段维护热度 Heat=α·N_visit+β·L_interaction+γ·R_recency（默认 α=β=γ=1），其中 R_recency=exp(-Δt/μ)（时间衰减，μ=1e+7 秒）；当段数超过最大容量（默认 200）时驱逐热度最低的段（类 LRU/工作集驱逐）；热度超阈 τ=5 的段晋级写入 LPM 后将其 L_interaction 重置为零以衰减热度、避免冗余。User KB / Agent Traits 为定长 100 的 FIFO 队列，满时按先进先出淘汰。更新主要靠 LLM 重写画像与摘要归并，并含画像更新前的内容质量校验（仓库后续提交修复，防止低质量响应覆盖已有画像）。
- **经验回放 (核心主题)**: 不属于以「重放过去决策轨迹来自我改进任务能力」为核心的经验复用范式。MemoryOS 的「复用」是对过往对话信息的分层检索式调回——把跨会话历史（STM/MTM）与用户画像（LPM）拼回上下文，以维持长对话连贯与个性化回复，而非把成功/失败轨迹蒸馏为可迁移技能或策略在新任务上复用。其定位是面向多会话长对话的个性化记忆基础设施层（user-centric），不做技能复用、范例提示或 RL 式回放缓冲。注：同实验室后续工作 LightSearcher（arXiv 2512.06653）才把「经验式记忆」引入深度搜索做轨迹学习，但不属于本论文范围。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric）/ 提示层、上下文层。完全不进行任何梯度更新，依靠分层记忆的写入/检索/更新与上下文拼接实现「记忆与个性化演进」；底层 LLM 冻结（实验用 GPT-4o-mini、Qwen2.5-7B、Qwen2.5-3B）。属运行时（inference-time）有状态记忆，而非训练式学习。
- **失败学习 (核心主题)**: 几乎不涉及从失败中学习。MemoryOS 面向多会话对话连贯与个性化，不检测任务失败、不存储失败模式、不生成负面范例或纠错规则。其唯一与「错误/质量」相关的处理是工程层面的画像更新质量校验（防止低质量 LLM 响应覆盖已有用户画像，见仓库提交），属于写路径的质量门控而非跨任务的失败经验学习。这是 MemoryOS 与 Reflexion/Retroformer/CLIN/ReasoningBank 等失败驱动自进化系统的根本区别。
- **技能/程序归纳**: 否。MemoryOS 不从经验中归纳可复用的技能/工作流/程序性记忆。它提供固定的记忆管理动作（对话链构建、分段分页、热度驱逐、画像抽取、两阶段检索等）和 MCP 工具（add_memory / retrieve_memory / get_user_profile），均为人工预定义流程，而非由智能体从经验中 induce 出的程序性技能（区别于 Voyager 的技能库）。
- **在线 vs 离线**: 在线（online）。记忆在部署/交互过程中实时、增量构建：每轮对话即时写入 STM 对话页，队列满时即时迁移至 MTM 并按热度驱逐/晋级，高热度段即时触发 LPM 画像演化。不依赖离线批量训练语料；评测在 GVD 与 LoCoMo 的逐轮对话流上在线进行。

**评测 / Evaluation**

- **任务领域**: 多会话长期对话域：①多轮个性化对话/虚拟助手（考察跨会话一致性、上下文连贯与用户人设保持）；②超长对话记忆问答（单跳、多跳、时序、开放域问答）。产品化后扩展至接入 Claude Desktop / Cline / Cursor 等客户端的个性化 AI 助手长期记忆。不涉及网页导航、具身、代码、GUI 等智能体任务域。
- **基准**: ①GVD 数据集（源自 MemoryBank/Zhong et al. 2024）：模拟 15 个虚拟用户与助手在 10 天内的多轮对话，每天至少两个话题；指标为记忆检索准确率 Acc.（0/1 二值）、回复正确性 Corr.、上下文连贯性 Cohe.（0/0.5/1 三档），由 DeepSeek-R1 自动评分。②LoCoMo 基准（Maharana et al. 2024）：专为长期对话记忆设计，超长对话平均约 300 轮、约 9K token；问题分单跳（Single-hop）、多跳（Multi-hop）、时序（Temporal）、开放域（Open-domain）四类，用标准 F1 与 BLEU-1 评测。
- **报告增益**: 头条结果（GPT-4o-mini，LoCoMo）：相对所有基线平均提升 F1 +49.11%、BLEU-1 +46.18%（摘要数字，部分版本写作 +48.36% F1；以 ACL/arXiv 终版 49.11%/46.18% 为准）。LoCoMo 分项 F1（GPT-4o-mini，MemoryOS=Ours）：单跳 35.27、多跳 41.15、时序 20.02、开放域 48.62，平均排名第 1（F1 与 BLEU-1 均 Avg.Rank=1.0）；相对最强基线的提升尤以时序题最显著（时序 F1 +118.80%、BLEU-1 +111.52%），单跳 F1 +32.35%、开放域 F1 +18.47%。其中 MemGPT 单跳/多跳/时序/开放域 F1 为 26.65/25.52/9.15/41.04，A-Mem*（同环境复现）为 22.61/33.23/8.04/34.13。Qwen2.5-3B 上 MemoryOS 同样夺得 Avg.Rank 1.0，单跳 F1 23.26（+125.61%）、开放域 F1 26.23（+112.56%）。GVD（GPT-4o-mini）：Acc.=93.3 / Corr.=91.2 / Cohe.=92.3，较 SOTA 基线 A-Mem（90.4/86.5/91.4）分别 +3.2% / +5.4% / +1.0%；Qwen2.5-7B 上为 91.8/82.3/90.5。效率（LoCoMo，Table 3）：MemoryOS 检索 token 3,874、平均 LLM 调用 4.9 次、平均 F1=36.23，显著优于 MemGPT（16,977 token / 4.3 calls / F1 29.13）与 A-Mem*（2,712 token / 13.0 calls / F1 26.55）——即调用次数远少于 A-Mem、token 远少于 MemGPT，同时 F1 最高。
- **对比基线**: 与代表性记忆方法对比：TiM（Think-in-Memory，存推理「思考」并用 LSH 检索 + 后思考反思）、MemoryBank（基于艾宾浩斯遗忘曲线的向量记忆 + 用户画像）、MemGPT（OS 式双层主/外部上下文 + 显式读写调用）、A-Mem（Agentic Memory，结构化笔记互联网络；含原文报告值 A-Mem 与同环境复现 A-Mem*）。底层 LLM 涵盖 GPT-4o-mini、Qwen2.5-7B（GVD）与 Qwen2.5-3B（LoCoMo）。无显式「无记忆/全上下文」对照，但消融（-MemoryOS）等价于默认 LLM 无记忆基线。

**分析 / Analysis**

- **关键创新**: 首次提出面向 AI 智能体的「记忆操作系统」（MemoryOS）：把操作系统的分段+分页内存管理与优先级（热度/LRU）驱逐思想系统性迁移到对话记忆，构建短期 STM / 中期 MTM / 长期个人记忆 LPM 三级分层存储，并统一为存储、更新、检索、生成四大模块协同；其核心新意在于「分段分页 + 热度驱动晋级/驱逐 + 持久化个人画像（90 维用户特质）」的一体化设计——相对此前各自孤立聚焦存储结构、检索或更新策略的方法，提供了统一、可插拔、兼顾效果与效率的综合记忆管理框架。
- **局限**: ①面向多会话对话个性化，任务域较窄，不覆盖网页/具身/代码等智能体决策任务，亦不做失败学习或技能归纳；②大量依赖 LLM 在线生成元信息（对话链摘要、段摘要、关键词、画像抽取），多次 LLM 调用带来开销与误差累积风险，画像质量依赖底模能力（弱模型如 Qwen2.5-3B 绝对分数明显偏低）；③众多人工超参（STM 长度 7、MTM 容量 200、热度阈 τ=5、相似度阈 θ=0.6、时间常数 μ=1e+7、α=β=γ=1）需调，泛化性与鲁棒性待验证；④遗忘为热度/容量驱动的驱逐，被驱逐的中期内容不一定保留，可能丢失长尾细节；⑤无显式矛盾消解与时间有效性建模；⑥安全/隐私治理与过度个性化风险未讨论。
- **与其他工作关系**: 属于「B. 情景记忆与检索（Episodic memory & retrieval）」簇中以分层结构 + 个性化记忆为核心的工作。它在 MemGPT（B3，OS 式分层主/外部上下文、显式读写）之上更进一步：论文将自身归为「架构驱动」类并明确以 MemGPT 为对比基线，批评其扁平 FIFO 队列随对话变长会导致「话题混杂」，故引入按话题分段的分段分页 + 热度驱逐解决话题对齐问题；同时借鉴 MemoryBank（B2，艾宾浩斯遗忘曲线 + 用户画像，作者实验室 EmotionalRAG 与之同源）的画像/衰减思想、并以其 GVD 数据集做评测；与 A-Mem（B4，Agentic Memory 笔记互联网络）正面对比并指出其多步链接生成带来延迟与误差累积，而 MemoryOS 在 LoCoMo 上以更少 LLM 调用取得更高 F1；与 TiM（存「思考」）对比指出其单阶段哈希检索难保跨话题依赖。整体上是 user-centric 个性化长对话记忆系统（与 Mem0、Zep、MIRIX 等同赛道），区别于 A 簇（Reflexion/Retroformer/ExpeL/CLIN/ReasoningBank）以「从轨迹蒸馏可迁移知识/失败反思」为核心的 agent-centric 自进化路线——二者机制与评测目标不同、可互补。
- **可复现性**: 可复现性强、社区采用度较高：官方完整开源（github.com/BAI-LAB/MemoryOS，Apache-2.0，约 1.4k stars、141 forks），提供 PyPI 包（memoryos-pro）、MCP 服务器、ChromaDB 后端、Docker 镜像与在线 Playground，并在仓库 eval/ 下公开 LoCoMo 复现脚本（main_loco_parse.py / evalution_loco.py）；支持 OpenAI/Deepseek/Qwen/vLLM/本地部署等多种 LLM 与多种开源嵌入模型，便于本地复现。GVD 评测依赖 DeepSeek-R1 自动评分、A-Mem 同环境复现（A-Mem*）与原文报告值存在差距（如多跳 F1 45.85 vs 复现 33.23），说明部分对比结果对实现/环境敏感；底模含闭源 GPT-4o-mini，结果对模型版本有一定依赖。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式 + LLM 生成的混合管线，非学习型）。记忆管理策略由人工设定的规则与阈值（STM 长度 7、FIFO 迁移、热度公式 α=β=γ=1、容量驱逐、晋级阈 τ=5、相似度阈 θ=0.6 等）加 LLM 在线生成元信息共同构成，不使用 RL/训练去学习「何时存/取/更新/遗忘」的策略本身。因此处于 2025-26「学习型记忆控制」分水岭的启发式/规则驱动一侧（与 Memory-R1、Mem-α 等用 RL 学习记忆管理策略的方法相对）。
- **记忆主体**: 明确以用户中心（user-centric）。系统目标是记住用户事实、偏好、画像与历史以维持跨会话一致性与个性化（LPM 含用户画像、User KB、90 维用户特质），并辅以智能体画像/特质维持稳定人设；评测（GVD/LoCoMo）也聚焦对话连贯与个性化回忆，而非「记住自身经验以提升任务能力」。与 ReasoningBank/Voyager 的 agent-centric 自进化方向相对。
- **多智能体记忆**: 单智能体（single-agent）。论文聚焦单个智能体的三级分层记忆管理与单一用户-助手对话，不涉及多智能体间的共享/路由记忆（区别于 G-Memory、MIRIX）。仅维护「用户人设」与「智能体人设」两套画像，属同一智能体内部的双画像，而非多智能体协作记忆。
- **时序推理支持**: 中等/部分支持。对话页含时间戳 T，热度公式显式包含近因时间衰减 R_recency=exp(-Δt/μ)，且 LoCoMo 设有专门的「时序（Temporal）」问答类别——MemoryOS 在时序题上提升最大（GPT-4o-mini 时序 F1 +118.80%），表明其分层 + 时间戳 + 近因加权对时序回忆有实质帮助。但它不显式建模事实有效性窗口或事件日历式时间区间（区别于 Zep/Graphiti 的双时间有效性建模），时间主要作为对话组织与热度衰减维度，而非一等的可推理对象。
- **模态**: 纯文本（text-only）。处理多轮对话文本，不涉及视觉/截图/具身/视频等多模态记忆（区别于 MIRIX）。
- **过度个性化/记忆安全风险**: 基本未涉及作为研究维度。论文不讨论有害/过时/侵入性记忆、谄媚（sycophancy）、隐私治理或过度个性化风险，也无相应安全基准（如 OP-Bench/Causal-LoCoMo）。工程上仅有一项相关防护：仓库后续提交增加「用户画像更新前的内容质量校验，防止低质量响应覆盖已有画像」，属轻量写路径质量门控；其热度驱逐机制可在一定程度上淘汰陈旧低热信息，但整体记忆安全/隐私治理超出本工作范围。
- **冲突/矛盾处理**: 弱/隐式。无显式的矛盾事实检测与合并机制。冲突信息的处理主要依赖：①LPM 画像由 LLM 抽取并增量更新/演化（可隐式覆盖旧值），User KB/Agent Traits 为定长 FIFO 队列（旧信息随队列滚动被淘汰）；②MTM 段摘要由 LLM 重写归并。但这些都依赖 LLM 自发判断，缺乏系统化的冲突消解流程（区别于 Memory-R1 的显式 UPDATE、MEMTRACK 的冲突追踪）。
- **token成本/延迟证据**: 有量化效率证据（LoCoMo，Table 3）：MemoryOS 检索消耗 3,874 token、平均每次响应 4.9 次 LLM 调用，相比 MemGPT（16,977 token）token 消耗降约 77%，相比 A-Mem*（13.0 次调用）调用次数降约 62%，同时平均 F1 最高（36.23）。仓库另报告工程优化使 PyPI 实现「快 5 倍」（并行化降延迟）、MCP 并行化加速。论文将效率作为正面卖点之一（在更少调用/更低 token 下取得更高效果），是相对 MemGPT/A-Mem 的明确优势。


<a id="b8-em-llmepisodic-memory-llm论文题为human-inspired-episodic-memory-for-infinite-context-llms注意本研究-outline-题录写作human-like与-arxiviclr-正式题目human-inspired略有出入系统简称统一为-em-llm"></a>

### B8 EM-LLM

*EM-LLM（Episodic Memory LLM；论文题为《Human-inspired Episodic Memory for Infinite Context LLMs》。注意：本研究 outline 题录写作「Human-like」，与 arXiv/ICLR 正式题目「Human-inspired」略有出入，系统简称统一为 EM-LLM）*


**基本信息 / Provenance**

- **年份**: 2024（arXiv 预印本，2024-07-12 首次公开，arXiv:2407.09450，CorpusId 271162171；正式发表于 ICLR 2025）
- **作者/机构**: Zafeirios Fountas（一作）、Martin A. Benfeghoul（与一作同等贡献）、Adnan Oomerjee、Fenia Christopoulou、Gerasimos Lampouras、Haitham Bou-Ammar、Jun Wang（王军）。主要单位为华为诺亚方舟实验室伦敦分部（Huawei Noah's Ark Lab, London, UK）；Bou-Ammar 与 Jun Wang 同时隶属伦敦大学学院计算机系 AI Centre（University College London, UCL）。
- **发表venue**: ICLR 2025（The Thirteenth International Conference on Learning Representations，会议论文，OpenReview id=BI2int5SAC，DBLP conf/iclr/FountasBOCLB025）。Semantic Scholar 将首发年份标为 2024（对应 arXiv 预印本），会议年份为 2025。
- **论文链接**: https://arxiv.org/abs/2407.09450（OpenReview 正式版：https://openreview.net/forum?id=BI2int5SAC）
- **代码链接**: https://github.com/em-llm/EM-LLM-model（官方代码，Python，MIT 许可，约 275 stars / 20 forks，截至 2026-06；作者 zfountas、MartinBenfeghoul 维护，基于 InfLLM 框架修改）。

**记忆分类 / Taxonomy**

- **记忆类型**: 情景性记忆（episodic）为核心。把 LLM 的过往 token（具体为各注意力头的 key-value 对）组织为「事件（events）」集合，类比人脑对个人经历的情景记忆；同时显式对应认知模型中的工作记忆（local context 作为「注意力焦点」focus of attention，类 Cowan 2001 嵌入式过程模型）。不实现语义抽象/事实库式的语义记忆，也无技能/程序记忆——属于 CoALA 框架中的情景记忆，但其「情景单元」是 KV 缓存片段而非自然语言笔记。
- **记忆结构**: 分层 KV 缓存式情景记忆。上下文被划分为三组：(1) initial tokens（固定 128 个，充当 attention sink）；(2) evicted tokens（被驱逐的历史 token，由情景记忆模型管理，组织为「事件」block）；(3) local context（最近 token，全 softmax 注意力，类工作记忆）。历史按「基于惊讶度的事件分割 + 图论边界精修」切成不重叠的事件块（block），每个事件块保留若干「代表性 token（representative tokens）」用于检索。本质是 KV-retrieval / 内存块组织（非向量笔记、非知识图谱、非参数权重）。
- **存储后端**: 底层为 LLM 自身的 KV 缓存（key-value pairs），分层存储于 GPU 显存的内存块（memory blocks），并支持溢出到 CPU 内存（vector_offload_threshold≈5万 token 时把代表性 token 卸载到 CPU）与磁盘（disk_offload_threshold≈30万 token 起启用磁盘卸载），以支撑超长（千万级 token）序列。大规模检索使用近似 k-NN（FAISS，Douze et al. 2024）。无外部向量数据库/图数据库；不修改任何模型参数。
- **持久化**: 上下文内/缓存内（in-context / KV-cache）持久化，单次推理会话级别。记忆是当前长序列处理过程中累积的 KV 事件，可分级卸载到 CPU/磁盘以容纳实质无限上下文，但不跨会话长期落盘为可复用知识库，亦不写入模型参数（无微调，no fine-tuning）。属于「扩展上下文窗口」式的临时但可达千万 token 规模的记忆，区别于 Mem0/A-MEM 那类跨会话外部持久存储。

**核心机制 / Mechanisms**

- **写入/编码**: 逐字保留 KV 对，再在线分块为「事件」。流程：(1) 处理长序列时，对每个新 token 计算其惊讶度（surprise）= 自回归负对数似然 −log P(x_t | x_1..x_{t-1}; θ)，作为「事件边界」候选信号；(2) 用自适应阈值 T = μ_{t−τ:t} + γσ_{t−τ:t}（滑动窗口的惊讶度均值 μ 与标准差 σ，γ 为缩放因子，配置默认 γ=1.0）判定潜在边界，惊讶度超阈即为候选边界，得到初始边界集 B；(3) 图论边界精修（boundary refinement）：将局部窗口内各 token 的 key 向量两两点积相似度构成邻接矩阵 A^h，在相邻初始边界 (α,β] 间搜索使「簇内相似度高、簇间相似度低」目标最优的位置，优化指标为模块度 modularity（最大化，最优结果）或电导率 conductance（最小化），算法复杂度 O(nm)（n 序列长、m 块大小）。最终每个事件块编码为：原始 KV 对 + 一组代表性 token（按 InfLLM/Xiao et al. 2024a 选最具影响力的 token，repr_topk 默认 4）。block 大小受 min_block_size=8、max_block_size=128 约束。
- **检索机制**: 两阶段、逐层、人脑启发的检索。对每个新生成 token，在每一层独立检索 k=k_s+k_c 个事件加入扩展上下文：(1) 相似度缓冲（similarity buffer）——用当前 query 与各事件「代表性 token」做点积相似度 k-NN，取 top-k_s 个最相关事件（大规模用近似 k-NN）；(2) 连续性缓冲（contiguity buffer）——一个大小 k_c 的队列，当某事件被检索命中时，把其在原始序列中相邻（±n 位）的事件也入队，从而复现人类记忆的「时间连续性效应（temporal contiguity）」与非对称前向效应，让 induction head 利用时间邻接信息；队列随新事件入队而自然淘汰旧/重复事件（recency 衰减）。最终上下文窗口 = initial tokens（128，attention sink）+ contiguity buffer + similarity buffer + local context（默认 n_local=4096）。配置示例 n_mem=2048 为检索 token 预算，contiguity buffer 默认占 n_mem 的 0.3。检索按层独立，使不同层可聚焦上下文不同部分。
- **反思/巩固**: 无传统意义上的「原始经验→高层洞见」反思/摘要巩固。EM-LLM 不调用 LLM 生成摘要或抽取规则，其「巩固」体现在记忆形成阶段的事件分割与边界精修——把连续 token 流组织成内部高内聚、彼此可区分的「事件单元」，类比人脑「事件感知/记忆形成」的认知过程（受 Zacks、Baldassano、Clewett 等事件认知与海马研究启发）。这是一种结构化组织而非语义抽象，不产生可被自然语言复述的「教训/洞见」。论文将其与 Baddeley 工作记忆、Ericsson & Kintsch 长期工作记忆、Cowan 嵌入式过程等认知模型对应，但未做摘要式 reflection。
- **遗忘/更新**: 无显式遗忘曲线/去重/冲突消解机制。历史 KV 事件一旦形成基本保留（可分级卸载到 CPU/磁盘，受 max_cached_block 等限制管理显存而非语义遗忘）。唯一近似「遗忘」是连续性缓冲队列的自然出队（FIFO 式淘汰旧/重复事件，体现 recency 效应）以及每个 token 只把 top-k 事件纳入当前上下文（未被检索的事件暂不参与计算但不被删除）。不实现 Ebbinghaus 衰减、ADD/UPDATE/DELETE 语义操作或矛盾事实合并。
- **经验回放 (核心主题)**: 属于「长上下文回忆」而非「智能体技能复用」。EM-LLM 不重放动作轨迹、不蒸馏可迁移策略/技能；它把单条超长输入流（如长文档、对话、代码、检索语料）切成情景事件，在生成时按相似度+时间连续性把相关历史事件「调回」当前上下文，实现对极长历史的精准回忆与利用。其复用价值体现为：在检索/QA 类任务上比 InfLLM 提升最高 40%（检索类）与 29.7%（QA 类），在 10.2M token 的 Passkey.Retrieval 上达 100% 准确率。这是面向「上下文内信息回忆」的复用，区别于 Voyager/ReasoningBank/ExpeL 那类把成功/失败经验沉淀为可跨任务复用的技能或策略。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric）/ 无训练（training-free）。整套机制（惊讶度分割、图论精修、两阶段检索）直接作用于预训练 LLM 的前向计算与 KV 缓存，全程无梯度更新、无微调（no fine-tuning），可即插即用于 Mistral-7B、LLaMA-2/3/3.1、Phi-3/3.5 等多种基础模型。
- **失败学习 (核心主题)**: 不涉及。EM-LLM 面向长上下文回忆与处理，不进行失败检测、失败轨迹反思或负例/避坑规则学习；它不区分「成功/失败经验」，也不针对错误生成修正机制。改进完全来自更好的上下文组织与检索精度，而非从错误中学习——与 A 簇反思/失败驱动方法（Reflexion、Retroformer、CLIN、ReasoningBank）形成鲜明对照。
- **技能/程序归纳**: 否。不归纳可复用技能/工作流/过程。EM-LLM 组织的是输入 token 流的「事件结构」，既无技能库也无 skill 调用/程序记忆机制。
- **在线 vs 离线**: 在线（online）流式构建。事件分割与边界精修在处理序列时实时进行（online fashion），随 token 流动态形成情景记忆并即时检索；不依赖对历史轨迹语料的离线批量训练。

**评测 / Evaluation**

- **任务领域**: 长上下文语言处理为主：长文档 QA（叙事/科学/多领域）、多跳问答、长文摘要、少样本学习（few-shot）、合成检索（passkey/数字/KV 检索）、代码补全/检索，以及对长篇书籍语料（PG-19）的事件分割分析。另含一项与人类认知对照的实验：用人工标注的播客转录数据评估事件边界与「人类感知事件」的相关性。属于 agent-centric 的「自身上下文回忆」而非用户个性化对话记忆，亦非 web 导航/具身决策。
- **基准**: (1) LongBench（按任务分组：单文档 QA/多文档 QA/摘要/少样本/合成检索/代码）；(2) ∞-Bench（InfiniteBench：含 C.D、M.F、MC、Retrieve.KV、Retrieve.PassKey、Retrieve.Number 等子任务），并扩展 Passkey 检索至 10.2M token；(3) PG-19（长书籍语料，做事件分割的图论指标对比，γ=1e-3）；(4) Kumar et al. 2023 的人类事件标注音频/播客转录数据集（与 Michelmann 2021、Lositsky 2016 结果对照）。基础模型涵盖 Mistral-7B-Instruct-v0.2、LLaMA-2-7B、LLaMA-3-8B、LLaMA-3.1-8B、Phi-3-mini、Phi-3.5-mini。
- **报告增益**: (1) 对比 SOTA KV 检索基线 InfLLM：在 5 个基础模型上全面改进，覆盖 LongBench 80% 的任务组与整体平均；逐项看在所有 ablation 下均超过 InfLLM，检索类任务（Passage/KV/Passkey/Number）最高提升 40%，QA 类（Narrative/Qasper/MultiField/Hotpot/2Wiki/Musique）最高提升 29.7%。典型整体均分：LLaMA-3.1 上 EM-LLM_SM 51.3 vs InfLLM 51.1（LongBench Avg.）、∞-Bench R.KV 90.2 vs 81；Mistral-v2 上 LongBench Avg. 43.7 vs 41.9、Ret 84.1 vs 64、∞-Bench R.KV 99 vs 95.6。(2) 对比 RAG（SOTA NV-Embed-v2 检索器）与全上下文（full-context）：在 LLaMA-3.1-8B 上 EM-LLM 大多数任务优于二者，较 NV-Embed-v2 在 LongBench 上高 30.5%、在 ∞-Bench 上高 11.5%，且多数任务甚至超过全上下文模型，所需资源与 RAG 相当。(3) 可扩展性：在扩展版 ∞-Bench Passkey.Retrieval 上对长达 10.2M token 序列达 100% 准确率（全上下文模型在此规模计算不可行）。(4) 消融：边界精修（refinement）在 LongBench+∞-Bench 的 60% 任务上取得最优，连续性缓冲（contiguity）在 44% 任务上最优，二者互补；PG-19 上 SM/SC（惊讶度+精修）在模块度/电导率/I-IS 等图论指标上一致最优（如 LLaMA3-8B：Mod SM=27.0±35.6 vs F=−1.6，Con SM=−30.6 vs F=11.3）。(5) 认知对照：惊讶度分割（S）所得事件边界与人类感知事件最接近，加精修（SM/SC）进一步提升与人类一致性。
- **对比基线**: InfLLM（训练免微调的 KV 检索/分块基线，被作者视为当时 LongBench/∞-Bench 上的 SOTA，作为首要对比）；RAG（含 SOTA NV-Embed-v2 检索器，以及作者自建的「基于惊讶度的 RAG」变体）；full-context（暴力全 token softmax 注意力）；多种固定分块（fixed/F、FM、FC）与随机分割作为消融对照。属于「长上下文处理/KV 检索/RAG/全上下文」类基线，未与跨会话外部记忆系统（Mem0/A-MEM/MemGPT）正面比较。

**分析 / Analysis**

- **关键创新**: 首次把人脑「事件认知（event cognition）」与情景记忆机制无微调地引入 LLM 长上下文处理：用贝叶斯惊讶度（自回归负对数似然超自适应阈值）在线分割 token 流为「事件」，并以图论边界精修（最大化模块度 / 最小化电导率，基于注意力 key 相似度图）优化事件的内聚-分离结构，再以「相似度 + 时间连续性」两阶段、逐层检索复现人类记忆的连续性效应——由此让普通预训练 LLM 实现实质无限（10M+ token）上下文，并在性能上超越 InfLLM、RAG 乃至全上下文。同时实证表明其事件分割与人类感知事件强相关，为「用 LLM 研究人类记忆」提供了计算框架。
- **局限**: (1) 记忆仅为单次推理会话内的 KV 缓存，不跨会话长期持久为可复用知识库，亦无语义抽象/反思；(2) 无真正遗忘、去重与冲突消解，长序列下事件只增（靠 CPU/磁盘卸载维持，存在 I/O 与显存管理开销）；(3) 不处理失败学习、技能归纳、个性化安全与隐私治理；(4) 边界精修依赖逐层相似度图计算（开销随窗口增大上升，配置中可选只对部分层 refine_from_layer 起作用以省算力），千万级 token 需磁盘卸载，延迟/工程复杂度提高；(5) 图聚类指标（模块度/电导率）为启发式选择，作者承认其他分割/聚类算法可能更优；(6) 评测集中于长上下文语言任务（LongBench/∞-Bench/PG-19），未覆盖智能体决策、具身、多模态或跨会话个性化场景。
- **与其他工作关系**: 属本研究「B. 情景记忆与检索（Episodic memory & retrieval）」簇，且偏「上下文回忆/长上下文架构」一端，而非外部持久记忆库。其直接继承并改造 InfLLM（Xiao et al. 2024a 的 training-free KV 分块检索与代表性 token 选择、attention sink、固定位置编码），核心增量是把「固定/均匀分块」替换为「惊讶度事件分割 + 图论精修」，并新增连续性缓冲。与同簇其他 item 的区别：B1 Generative Agents 用自然语言记忆流 + recency/importance/relevance 评分检索，B2 MemoryBank 用 Ebbinghaus 衰减、B3 MemGPT 用分页 OS 式上下文管理、B4 A-MEM 用 Zettelkasten 笔记图——它们多为「跨会话、自然语言/向量笔记、外部持久」的对话/QA 记忆；EM-LLM 则是「会话内、KV 缓存事件、无微调」的长上下文回忆机制，强调认知科学（事件分割、时间连续性效应、工作记忆）的可解释对应。它与 A 簇（失败/技能驱动的智能体自我提升）正交，与学习型记忆控制（Memory-R1、Mem-α）亦正交——EM-LLM 全程启发式、无策略学习。
- **可复现性**: 可复现性较好：官方开源 em-llm/EM-LLM-model（Python，MIT，约 275 stars，基于 InfLLM 框架，含 config YAML、下载与评测脚本，支持 Mistral/LLaMA3/LLaMA3.1/Phi3/Phi3.5 五种基础模型），所用基准 LongBench、∞-Bench、PG-19 均公开，并提供更完整结果表（benchmark/further_results.md）。论文经 ICLR 2025 同行评审（OpenReview 公开评审）。不确定性主要来自显存/磁盘卸载相关的硬件配置敏感性与千万级 token 实验的算力门槛。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式流程）。事件分割阈值（惊讶度 μ+γσ）、边界精修（模块度/电导率优化）、两阶段检索（相似度 k-NN + 连续性队列）均为固定的启发式/图论规则，不使用 RL 或训练去学习「何时/存什么/如何检索」的记忆管理策略本身。处于 2025–26「学习型记忆控制」分水岭的启发式一侧（与 Memory-R1、Mem-α 等学习型方法相对）。
- **记忆主体**: 智能体/模型自身上下文中心（agent/self-context-centric）。记忆的是模型当前正在处理的超长输入序列（文档/对话/代码/检索语料）本身，目的是让模型回忆并利用自己「读过」的远距离内容，而非积累用户画像做个性化（区别于 Mem0/Zep/A-MEM 的用户中心记忆），也非积累自身操作经验做自我进化（区别于 Voyager/ReasoningBank 的经验复用）。论文亦提出其可支撑「持续、个性化的长期交互」，但本工作评测仍以会话内长上下文回忆为主。
- **多智能体记忆**: 单智能体/单模型（single-agent）。EM-LLM 服务于单个 LLM 的长上下文处理，未涉及多智能体间共享或路由记忆（区别于 G-Memory、MIRIX）。
- **时序推理支持**: 部分支持时间结构但非显式时序事实推理引擎。通过连续性缓冲显式建模「事件的时间邻接/先后」并复现人类记忆的时间连续性效应（temporal contiguity）与非对称前向效应、recency 效应（队列淘汰）；初始/被检索事件保留其在原序列的相对顺序。但它不像 Zep/Graphiti 那样建模事实有效性窗口、双时态边或事件日历，无显式「某事实何时为真」的时序推理。
- **模态**: 纯文本（text-only）。所有事件分割、检索与生成均基于文本 token 的 KV 表示，无视觉/截图/视频/具身多模态记忆（论文仅在展望中提出可借鉴 Baddeley 多组件模型加入模态特定缓冲以支持多模态）。
- **过度个性化/记忆安全风险**: 未涉及。论文不讨论有害/过时/侵入性/谄媚式记忆、过度个性化或隐私治理；EM-LLM 是会话内上下文回忆机制，无跨用户长期画像，故个性化安全风险面较小但也未被分析；缺乏遗忘/冲突消解意味着会话内若输入含矛盾信息可能被一并回忆（属潜在但未被本工作讨论的风险）。
- **冲突/矛盾处理**: 不处理。无矛盾/冲突事实的检测、合并、版本化或失效标记机制。相互矛盾的内容若同处一条长序列，会作为不同事件共存并按相似度/连续性被检索，由底层 LLM 在推理时自行权衡，而非由记忆层显式消解（区别于 Memory-R1 的 UPDATE 操作或 MEMTRACK 类显式冲突追踪）。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="b9-mirixmulti-agent-memory-system-for-llm-based-agents模块化多智能体记忆系统含六类记忆--八个智能体支持多模态截图输入由-mirix-ai-团队开发附带屏幕监控个人助手应用"></a>

### B9 MIRIX

*MIRIX（Multi-Agent Memory System for LLM-Based Agents；模块化多智能体记忆系统，含六类记忆 + 八个智能体，支持多模态/截图输入；由 MIRIX AI 团队开发，附带屏幕监控个人助手应用）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本 2507.07957 于 2025-07-10 首次公开 v1）
- **作者/机构**: Yu Wang（王宇，加州大学圣地亚哥分校 UCSD，邮箱 yuw164@ucsd.edu）、Xi Chen（陈曦，纽约大学斯特恩商学院 NYU Stern，邮箱 xc13@stern.nyu.edu）；二人共属创业公司 MIRIX AI（mirix.io）。第一作者 Yu Wang（同时是 MemoryLLM、M+、Towards Lifespan Cognitive Systems 等记忆方向工作的主导者）。
- **发表venue**: arXiv 预印本（卷 abs/2507.07957，DOI 10.48550/arXiv.2507.07957，DBLP journals/corr/abs-2507-07957），尚无正式会议/期刊发表记录；属于带有开源代码与商业产品（MIRIX AI 公司、桌面应用、mirix-client PyPI 包）的工业/开源系统。
- **论文链接**: https://arxiv.org/abs/2507.07957
- **代码链接**: https://github.com/Mirix-AI/MIRIX（Python 为主，Apache-2.0 许可，约 3.6k stars、约 287 forks，9 位贡献者，截至 2026-06；评测代码在 public_evaluation 分支）。配套 PyPI 包 mirix-client、官网 mirix.io、文档 docs.mirix.io、Discord 社区。记忆系统以 Letta（原 MemGPT）框架为基础构建。
- **引用数**: 约 106 次引用（Semantic Scholar，CorpusId 280277519，截至调研日 2026-06）；发布仅约一年即被高频引用，是 2025 年多智能体记忆/个性化记忆方向的高关注度工作。

**记忆分类 / Taxonomy**

- **记忆类型**: 横跨 CoALA 全部主要记忆类别且显式分型最细的系统之一：情景记忆（Episodic，时间戳事件/经历）、语义记忆（Semantic，概念与命名实体/社交图谱）、程序性记忆（Procedural，分步操作流程/脚本）三类齐备；另加核心记忆（Core，类 MemGPT 的 persona/human 持久画像，偏工作记忆 + 语义）、资源记忆（Resource，文档/转写/多模态文件）与知识库（Knowledge Vault，逐字保存的敏感凭据/地址等）。共六类记忆组件，是其相对扁平记忆系统的核心区分点。
- **记忆结构**: 模块化、组件化的多类型记忆架构：六个独立记忆组件，每个组件内部再用层级化结构组织（如情景记忆含 summary/details/actor/timestamp 字段；语义记忆按 name/summary/details/source 组织并可呈树状层级——如 Social Network、Favorites→Sports/Pets/Music；程序性记忆含 entry_type/description/steps 列表）。每类记忆由专属 Memory Manager 智能体维护，再由一个 Meta Memory Manager 做路由调度。属「按功能分型的结构化记忆库 + 多智能体管理层」，区别于 Mem0/Letta 的扁平事实库与 Zep/Cognee 的单一知识图谱。
- **存储后端**: 外部持久化数据库后端：论文 ScreenshotVQA 实验用 SQLite（sqlite.db 文件，仅存抽取出的结构化信息，不存原始图像）；开源/产品版采用 PostgreSQL（README 标注 PostgreSQL-native BM25 全文检索 + 向量相似度检索）。检索层支持 embedding_match（向量嵌入匹配）、bm25_match（BM25）、string_match（字符串匹配）三类工具。ScreenshotVQA 用 gemini-2.5-flash-preview-04-17 作骨干、google/siglip-so400m-patch14-384 作对比检索器；LOCOMO 用 gpt-4.1-mini 作骨干（因其函数调用能力强于 gpt-4o-mini）。
- **持久化**: 外部持久化（durable external store）+ 非参数化。所有长期记忆存于外部数据库（SQLite/PostgreSQL），跨会话长期保留并本地存储以保障隐私；不修改任何模型权重（依赖冻结的 Gemini/GPT 闭源模型）。核心记忆（Core）作为高优先级信息常驻于智能体上下文/系统提示中（接近 in-context），但其底层状态同样持久化。产品支持混合的本地/云记忆管理（敏感的 Knowledge Vault 本地存，体量大的 Resource Memory 可卸载到云端按需取回）。

**核心机制 / Mechanisms**

- **写入/编码**: 写入由多智能体协同、LLM 自主通过函数调用完成，且强调「抽象 + 路由」而非原样堆叠。新输入（含截图/对话）到来后系统先对记忆库做一次自动检索，再连同输入一起交给 Meta Memory Manager；后者分析内容、判断涉及哪些记忆组件，并把输入路由到相应的若干 Memory Manager（每步对 Meta Manager 1 次调用、对其余记忆管理器 0–6 次调用）。各 Memory Manager 并行更新各自记忆，并在各自类型内做去冗（avoid redundant information）。编码形态因组件而异：情景记忆抽取为带时间戳的事件摘要 + 细节；语义记忆抽取为概念/实体的 name+summary+details；程序性记忆抽取为 workflow/guide/script 的步骤列表；资源记忆保留文档全文或摘录；知识库逐字（verbatim）保存敏感值并标注敏感级别。对截图输入则不存原图，而是抽取显著信息后写入（实现 99.9% 存储压缩）。屏幕监控应用每 1.5 秒截图、丢弃过相似图像、积累 20 张不重复截图（约每 60 秒）触发一次记忆更新，并用流式上传 + Gemini 云 URL 把端到端延迟从 GPT-4 直传的约 50 秒降到 5 秒内。
- **检索机制**: 核心创新为「主动检索（Active Retrieval）」两阶段机制：第一阶段，智能体在回答问题或执行下一步前先根据上下文自动生成一个「当前话题（topic）」；第二阶段，用该话题从六个记忆组件各检索 top-10 最相关条目，结果按来源打标签后注入系统提示，从而无需用户显式指令即可触发检索、避免模型回退到过时的参数化知识（论文以「Twitter CEO 是 Linda Yaccarino 还是 Elon Musk」为例说明）。对话检索工作流（Chat Agent）则分两层：先做一次跨六组件的粗检索仅返回高层摘要，Chat Agent 再分析查询决定对哪些组件做更有针对性的精检索并选择合适的检索方法（embedding_match / bm25_match / string_match），最后整合信息合成回答；若查询涉及更新记忆，Chat Agent 可直接与对应 Memory Manager 交互做精准更新。检索方式由智能体按情境自主选择，论文未使用 recency·importance·relevance 三因子打分公式。
- **反思/巩固**: 存在显著的「抽象/巩固」式整合，是其相对原始 token 级记忆系统的关键优势。①写入时由各 Memory Manager 对原始输入（尤其是大量截图）做抽取与抽象，只保留显著信息形成结构化记忆（论文强调这是相对存原图/原文的「abstraction layer」）；②对多跳与时序信息做事件级巩固——例如把分散线索合并为「Caroline 4 年前从家乡瑞典搬来」这样的整合事件，或把「Melanie 一家 2023-10-19 在公路旅行后去露营」固化为确证事件，使查询时无需在线拼接碎片信息（这也是其多跳准确率领先 24+ 分的原因）。③核心记忆在容量超过 90% 时触发受控重写（controlled rewrite）以保持紧凑而不丢失关键信息。整体是「写入即抽象/巩固」，由记忆压力或路由触发，而非如 Reflexion/ExpeL 那样定期把失败经验蒸馏为可迁移规则。
- **遗忘/更新**: 更新主要通过各 Memory Manager 在各自记忆类型内做去冗与改写实现：写入时避免重复信息；核心记忆达 90% 容量时受控重写压缩；Chat Agent 可在收到新事实/纠正时与对应 Memory Manager 交互做精准更新。语义记忆条目设计为持久存在、除非被概念性覆盖。但论文未实现基于 Ebbinghaus 遗忘曲线的衰减，也未提供系统化的显式删除/合并/失效（invalidate）算子或明确的冲突消解流程，属相对弱项。
- **经验回放 (核心主题)**: 不属于以「重放过去决策轨迹来自我改进」为核心的经验复用范式（区别于 ExpeL/ReasoningBank/Voyager）。MIRIX 的「复用」体现为对用户长期信息的结构化检索式调回：把过往事件、概念、文档、流程从六类记忆中按主题精准取回并注入提示，以维持跨会话的个性化、一致性与多跳推理能力。它面向「用户中心的长期记忆个性化」，记忆的是用户的经历/偏好/资料而非智能体自身的成功/失败轨迹，因此不做技能复用缓冲、范例提示或失败轨迹回放。程序性记忆虽存「how-to」流程，但这些流程来自对用户操作的抽取而非从任务奖励中蒸馏的可迁移技能。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric）/ 提示层 + 外部记忆层。不做任何梯度更新，纯靠多智能体对外部结构化记忆的自主读写与主动检索来实现「记忆与个性化」；完全依赖冻结的闭源骨干模型（Gemini-2.5-Flash / GPT-4.1-mini）的函数调用能力。属运行时（inference-time）有状态记忆，而非训练式学习。
- **失败学习 (核心主题)**: 几乎不涉及「从失败中学习」这一主题。MIRIX 不检测任务失败、不维护失败模式记忆、不生成负面范例或纠错规则，也不把失败轨迹蒸馏为经验。其面向的是个性化长期记忆与多模态信息抽取/检索，而非智能体自我纠错式进化。这与 Reflexion/Retroformer/CLIN/ReasoningBank 等以失败反思为核心的 A 簇工作形成根本区别——MIRIX 提供的是记忆基础设施（结构化存取层），失败学习超出其设计范围。
- **技能/程序归纳**: 部分/弱。MIRIX 设有独立的程序性记忆（Procedural Memory）组件，显式存储 workflow/guide/script 形式的分步操作知识（如「如何报销差旅」「如何用 OpenTable 订餐」），支持任务分解与指令式规划，并可在对话时被检索调用。但这些流程是从用户输入/屏幕活动中抽取并结构化保存的，而非智能体从自身任务奖励/试错中归纳（induce）出的可迁移技能（区别于 Voyager 的自生成技能库）。因此属「程序性知识的存储与调用」，而非「技能归纳学习」。
- **在线 vs 离线**: 在线（online）为主。记忆在部署/交互过程中实时、增量构建：屏幕监控应用持续截图并周期性（约每 60 秒）触发记忆更新；对话中即时把新事实路由写入相应记忆组件并支持即时更正。LOCOMO 实验中则是先把整段对话注入记忆、再据记忆回答约 200 个问题，亦属逐会话在线构建记忆。不依赖离线批量训练语料。

**评测 / Evaluation**

- **任务领域**: 两大域：①多模态屏幕活动理解（multimodal GUI / 截图问答）——基于真实 PhD 学生数月计算机使用截图，回答与其视觉活动史相关的问题，并落地为屏幕监控个人助手与可穿戴设备（AI 眼镜/AI pin）场景；②长程多会话对话记忆（multi-session dialogue / 个性化问答）——LOCOMO 长对话基准。论文还展望了 Agent Memory Marketplace、可穿戴设备记忆等应用方向。
- **基准**: ①ScreenshotVQA（作者自建多模态基准）：采集三名计算机科学/物理 PhD 学生 1 周至 1 个月的高分辨率屏幕截图（学生1：03/09/2025 单日 5,886 张；学生2：05/16–06/06/2025 共 18,178 张/20 天；学生3：05/02–06/14/2025 共 5,349 张/逾一月，单图 2K–4K 分辨率），人工构建并复核问题（分别 11/21/55 题），评测单序列近 20,000 张截图，现有记忆系统均无法处理。②LOCOMO（Maharana et al. 2024，沿用 Mem0 设定）：10 段对话、每段约 600 轮/约 26,000 token、每段约 200 题，问题分单跳/多跳/时序/开放域四类（排除 adversarial 类以与既有方法公平对比）。两者均用 GPT-4.1 作 LLM-as-a-Judge 评分。
- **报告增益**: ①LOCOMO（LLM-as-Judge 总分 J，骨干 gpt-4.1-mini，MIRIX 取三次均值）：MIRIX 总体 85.38%，达 SOTA，较最强开源竞争者 LangMem（78.05%）高约 8.0 分（论文称「超出最佳现有方法 8.0」），并逼近 Full-Context 上界 87.52%。分项：单跳 85.11（Full-Context 88.53，因事件巩固偏向确证发生而在歧义题略低）、多跳 83.70（领先所有基线 24+ 分，得益于显式存储巩固事件）、时序 88.39、开放域 65.62（差距最小，受 RAG 缺乏全局理解所限）。对比基线：gpt-4.1-mini 下 Zep 79.09、Mem0 62.47、RAG-500 51.62；gpt-4o-mini 下 Zep 75.14、Mem0g 68.44、Memobase 70.91、A-Mem 48.38。三次单独运行总分 83.98/87.34/84.82（附录表3）。②ScreenshotVQA（三人均值，Acc↑/Storage↓）：MIRIX Acc 0.5950、存储仅 15.89MB；SigLIP@50（RAG 基线）Acc 0.4410、存储 15.07GB；Gemini（长上下文基线，图像缩至 256×256）Acc 0.1166、存储 236.70MB。即相对 RAG 基线准确率提升 35%、存储减少 99.9%；相对长上下文 Gemini 基线准确率提升 410%、存储减少 93.3%。③效率：屏幕处理端到端延迟由 GPT-4 直传约 50 秒降至 Gemini 流式上传 <5 秒。
- **对比基线**: 无记忆/扁平记忆与图谱记忆系统的横向对比。LOCOMO 上对比：A-Mem、LangMem、OpenAI（ChatGPT 记忆）、Mem0 与图谱版 Mem0g、Memobase、Zep、RAG-500，以及作为上界的 Full-Context（全文直接喂入）；所有基线统一用 gpt-4.1-mini 复现，并另列 Mem0 论文中 gpt-4o-mini 的结果。ScreenshotVQA 上因现有记忆系统（Letta/Mem0 等）无法处理多模态输入，仅对比 RAG 基线（SigLIP@50 检索 + Gemini 生成）与长上下文基线（Gemini 直接读最近 3,600 张缩放截图）。

**分析 / Analysis**

- **关键创新**: 把「记忆类型分型化 + 多智能体路由管理 + 多模态」三者首次系统整合：用六类专门记忆组件（Core/Episodic/Semantic/Procedural/Resource/Knowledge Vault）替代主流系统的扁平单一记忆，再用「六个 Memory Manager + 一个 Meta Memory Manager（共八个智能体，含 Chat Agent）」做动态路由与协同更新/检索；并提出「主动检索（Active Retrieval）」——先生成话题、再据话题跨六组件取回并注入提示，规避对过时参数知识的回退。最具突破性的是首次让记忆系统处理大规模多模态（截图）输入，通过抽象只存显著信息实现 99.9% 存储压缩，使「记忆」可用于此前所有记忆系统都无法处理的高分辨率屏幕活动理解场景。
- **局限**: ①无真正遗忘/衰减、缺乏系统化显式删除/合并/失效与冲突消解机制，长期运行可能累积过时或矛盾信息；②事件巩固偏向「确证发生」可能在歧义题（如问『计划』日期 vs『实际』日期）上判错，单跳略逊 Full-Context；③开放域问题受 RAG 范式「缺乏全局理解」限制，与上界仍有差距；④高度依赖骨干模型的强函数调用能力（论文明确指出弱函数调用模型不适配，故弃用 gpt-4o-mini 改用 gpt-4.1-mini），每步需多次智能体函数调用、成本/延迟开销较高；⑤ScreenshotVQA 数据集规模小（仅 3 名用户、共 87 题）、依赖闭源 Gemini/GPT 骨干，泛化与可重复性受限；⑥论文大量篇幅描述商业愿景（Agent Memory Marketplace 等）而非严格科学验证；⑦尚未正式同行评审发表。
- **与其他工作关系**: 属「B. 情景记忆与检索（Episodic memory & retrieval）」簇。其记忆系统直接以 Letta（即 B3 MemGPT 的产品化框架）为基础构建（README 致谢 Letta），Core Memory 的 persona/human 双块设计沿用 MemGPT；并在记忆分型上扩展自强调情景记忆（Pink et al.、Liao et al.、Anokhin ARIGRAPH）、语义记忆与程序性记忆（Wheeler & Jeunen）的一系列认知科学启发工作。横向对比与定位：相对扁平事实库 Mem0（B 簇）与图谱记忆 Zep/Graphiti、Cognee——MIRIX 批评其缺乏组件化路由、多模态支持与抽象层，并在 LOCOMO 上全面超越（85.38 vs Zep 79.09 vs Mem0 62.47）；相对 A-Mem（Zettelkasten 笔记图谱）、LangMem、Memobase（画像式）也显著领先。与第一作者 Yu Wang 自己的参数化记忆线（MemoryLLM、M+）形成互补：后者改模型权重/隐状态做 latent-space 记忆，MIRIX 则是不改权重、token 级 + 结构化的外部多智能体记忆。与以「经验/失败蒸馏」为核心的 A 簇（Reflexion/Retroformer/ExpeL/CLIN/ReasoningBank）正交——A 簇做 agent-centric 自进化，MIRIX 做 user-centric 个性化记忆基础设施，二者可叠加。在多智能体记忆维度与 G-Memory 并列为代表性工作。
- **可复现性**: 可复现性与社区采用度较强：官方完整开源（github.com/Mirix-AI/MIRIX，Apache-2.0，约 3.6k stars、287 forks、9 贡献者），提供 pip 安装的 mirix-client、Docker 部署（PostgreSQL+Redis+后端）、官网/文档/Discord 与每周社区讨论会；LOCOMO 评测代码与各基线/MIRIX 预测结果及 LLM-Judge 分数公开于 public_evaluation 分支。局限：ScreenshotVQA 数据集涉及个人隐私（真实截图）未必完全公开，实验依赖闭源 Gemini/GPT 骨干、结果对模型版本与函数调用能力敏感，且尚未经同行评审。仓库持续活跃维护（2026 年仍有新提交，被多家如 Intuit 的贡献者参与）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式管线 + LLM 自主路由的混合，但非学习型）。记忆管理策略由「人工设定规则（核心记忆 90% 容量触发重写、20 张截图/约 60 秒触发更新、各组件取 top-10、相似度 0.99 去重等）+ Meta Memory Manager 在提示引导下自主决定路由到哪些记忆组件、Chat Agent 自主选择检索方法」共同构成，不使用 RL/训练去学习「何时存/取/更新」的策略本身。因此处于 2025-26「学习型记忆控制」分水岭的启发式/提示驱动一侧（与 Memory-R1、Mem-α 等用 RL 学习记忆管理策略的方法相对），但其多智能体路由为学习型控制提供了清晰的动作空间雏形。
- **记忆主体**: 用户中心（user-centric）。MIRIX 的明确目标是记住用户特定信息（事件、偏好、社交图谱、文档、敏感资料）以实现长期个性化、跨会话一致性与准确召回（论文反复强调 personalize、user-specific data、个人助手）。它不以记住智能体自身经验来自我提升任务能力为目标（区别于 ReasoningBank/Voyager 的 agent-centric 自进化）。评测（LOCOMO 个性化问答、ScreenshotVQA 个人活动史）与机制（六类用户记忆、隐私本地存储、记忆市场愿景）都围绕用户中心展开。
- **多智能体记忆**: 多智能体（multi-agent）共享/路由记忆——这是其核心标签之一。系统由八个专门智能体构成：六个 Memory Manager（各管一类记忆）+ 一个 Meta Memory Manager（任务路由/调度，分析输入并把更新路由给相关 Memory Manager，再收集回报并确认）+ 一个 Chat Agent（与用户对话、跨组件粗检索后精检索并合成回答，也可直接驱动 Memory Manager 更新）。记忆更新时各 Memory Manager 并行工作。这种「Meta 路由 + 专门记忆智能体」的分层协同是 MIRIX 区别于单智能体记忆系统（Mem0/MemGPT/Zep）的关键，与 G-Memory 同为多智能体记忆代表。
- **时序推理支持**: 中等偏强（显式但非有效性窗口建模）。情景记忆显式带 timestamp 字段并充当「结构化日志/日历」，使智能体能按时间索引记忆、追踪变化、推理用户作息与近因；LOCOMO 时序题得分 88.39（领先多数基线）。但它通过把多条线索巩固为确证事件来支持时序/多跳推理，而非像 Zep/Graphiti 那样显式建模事实有效性窗口（fact-validity window）与事件双时间区间，因此在「计划日期 vs 实际发生日期」这类需要保留时序多版本的歧义题上可能出错。
- **模态**: 多模态（multimodal / 视觉）。这是 MIRIX 相对几乎所有同期记忆系统的最大差异点：可处理文本、图像、语音转写、屏幕截图等多模态输入（README 列 text/images/voice/screen captures）；ScreenshotVQA 基准即处理单序列近 20,000 张 2K–4K 高分辨率屏幕截图，并展望可穿戴设备（AI 眼镜/pin）的音频+视觉流记忆。论文明确批评主流系统的「文本中心」记忆在非语言输入占主导时失效。
- **过度个性化/记忆安全风险**: 部分涉及隐私治理，但未系统研究过度个性化的安全风险。正向设计：所有长期记忆默认本地存储、用户可控隐私设置，知识库（Knowledge Vault）对敏感条目分级（low/medium/high）、高敏感条目经访问控制并排除于常规检索之外以防泄露，产品强调端到端加密、细粒度权限与去中心化存储愿景。但论文未讨论有害/过时/侵入性记忆、谄媚（sycophancy）、记忆投毒或「记忆越多越好」的反面风险，也无对应的安全基准（如 OP-Bench/Causal-LoCoMo）评估——加之缺乏遗忘/冲突消解，长期可能累积过时记忆。该安全维度大体超出本工作范围。
- **冲突/矛盾处理**: 较弱、未系统化。各 Memory Manager 在写入时做「类型内去冗（avoid redundant information）」，语义记忆条目除非被概念性覆盖否则持久存在，Chat Agent 可在收到更正时驱动对应 Manager 做精准更新；核心记忆超 90% 容量时受控重写。但论文未提供显式的矛盾事实检测与合并算子，也未做冲突消解评测；LOCOMO 单跳歧义题（计划 vs 实际露营日期）暴露其在多版本事实上倾向保留确证版本而可能与期望不符。整体弱于 Memory-R1 的 UPDATE 算子或 MEMTRACK 的冲突追踪。
- **token成本/延迟证据**: 主要量化的是存储与延迟而非 token 成本。①存储：ScreenshotVQA 上相对 RAG 基线（SigLIP 保留原图 15.07GB）减少 99.9%（MIRIX 仅 15.89MB），相对长上下文 Gemini 基线（236.70MB）减少 93.3%——源于只存抽取信息、不存原图。②延迟：屏幕截图处理端到端延迟通过流式上传 + Gemini 云 URL，从 GPT-4 直传的约 50 秒降至 <5 秒。③成本/token 取舍为相对弱项：每个交互步需 Meta Memory Manager 1 次 + 其余记忆管理器 0–6 次函数调用，多智能体协同会增加 LLM 调用与 token 开销，论文未给出相对全上下文/其他记忆层的 token 成本节省百分比（区别于 Mem0 的 -91% p95 延迟/-90% token 等量化口径）。


<a id="b10-潜在学习--oracle-检索latent-learning--oracle-retrieval情景记忆补充参数化学习实现经验的灵活复用"></a>

### B10 潜在学习 / Oracle 检索

*潜在学习 / Oracle 检索（Latent learning / oracle retrieval）——情景记忆补充参数化学习，实现经验的灵活复用*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本 v1 于 2025-09-19 提交；v3 于 2025-12-23 修订）
- **作者/机构**: Andrew Kyle Lampinen、Martin Engelcke、Yuxuan Li、Arslan Chaudhry（均为 Google DeepMind），James L. McClelland（Google DeepMind 兼斯坦福大学心理学系）。通讯作者 Lampinen（lampinen@google.com）。
- **发表venue**: arXiv 预印本（cs.LG / cs.CL），2025 年；截至调研时尚无正式会议/期刊发表记录。
- **论文链接**: https://arxiv.org/abs/2509.16189 （DOI: 10.48550/arXiv.2509.16189）
- **引用数**: 约 16 次（Semantic Scholar，截至 2026 年；参考文献 88 条）。

**记忆分类 / Taxonomy**

- **记忆类型**: 情景记忆（episodic memory）。论文的核心论点是：情景记忆（以非参数化检索形式实现）补充神经网络的参数化（语义/皮层式）知识；明确借用认知科学的『互补学习系统』（Complementary Learning Systems）框架，将参数化学习类比为新皮层的缓慢泛化学习，将情景检索类比为海马体的快速、逐情景记忆。
- **记忆结构**: 原始（逐字、veridical）情景缓冲：存储完整未经压缩的训练经验/轨迹（文档或 maze 轨迹片段），而非摘要或抽取的事实/三元组。检索时把整段原始经验（输入上下文 x）重新放回模型上下文窗口中。
- **存储后端**: 概念性『情景存储 + oracle 检索器』。语言类任务中检索到的情景被重新编码后前置到上下文；RL gridworld 任务中以缓存的记忆状态（cached memory states）形式检索。论文刻意回避具体检索后端（如向量库）的实现，使用 oracle（理想）检索绕开检索本身的难题。
- **持久化**: 外部、持久的情景存储（durable external store），与模型参数中的参数化知识互补；情景以原始形式持久保留，在测试/未来任务时按需重新载入上下文（ephemeral 上下文），而非烘焙进权重。

**核心机制 / Mechanisms**

- **写入/编码**: 写入采用『保真原始情景』策略：把训练经验作为完整、逐字（veridical）的情景存入记忆，不在编码时判断哪些信息将来有用，也不做摘要/抽象/事实抽取。论文强调这正是关键——存储原始情景使测试时能比纯参数化泛化更灵活地访问其中『潜在（latent）』信息。形式化框架中，一条经验记为元组 [x, t, f(x,t)]（输入序列 x、任务线索 t、输出 f）；训练同时让模型重构输入与任务线索（语言任务用因果语言建模目标，RL 任务用 IMPALA 并加视觉/文本重构辅助损失），以保证模型重构经验的全部信息而非忽略部分。
- **检索机制**: 采用 oracle（理想）检索：测试与训练时都给模型/智能体提供至少一段与当前任务相关的情景（文档/轨迹），前置到上下文，同时混入若干均匀采样的无关干扰情景；检索情景总数（含干扰项）在不同任务中为 3 到 7 段（BC gridworld 只提供相关轨迹）。语言任务中检索情景被重新编码进上下文，RL 中以缓存记忆状态检索。论文未实现相似度评分/图遍历等真实检索机制，明确把『如何做有效检索』作为留待解决的难题；其贡献在于证明若能检索到正确情景、并把它放回上下文，模型即可借助上下文学习（ICL）灵活使用其中潜在信息。
- **反思/巩固**: 无显式『反思/摘要/抽象成高层知识』机制。论文反其道而行：不在写入时做抽象，而是保留原始情景，把『灵活复用』推迟到测试时通过上下文内推理（in-context reasoning）实现。它在认知科学层面讨论了海马回放（replay）对皮层巩固与泛化的作用（在线/离线 (p)replay），并指出训练时增广（offline replay/preplay）与测试时检索（online retrieval）是同一计算技巧的互补形式，但本文实验本身不含 raw→insight 的巩固步骤。
- **遗忘/更新**: 不适用（未涉及遗忘/衰减/合并/去重/冲突失效等机制）。情景以原始形式持久保留；论文聚焦『检索能否解锁潜在泛化』，未研究记忆随时间的编辑或衰减。
- **经验回放 (核心主题)**: 经验复用是论文的核心主题，但其形式是『把过去的原始经验通过检索重新放回当前上下文』，而非蒸馏策略或技能复用。机制：参数化学习只能学到训练经验中『显式（explicit）』的信息（如正向关系『Plato taught Aristotle』、训练中用过的编码索引、训练过的导航目标），无法灵活使用其中『潜在（latent）』含义（反向关系、未用过的索引、未训练过的导航目标）；oracle 检索把含相关信息的原始情景调回上下文后，模型可用其已具备的上下文内推理能力（ICL）来完成这些潜在任务，从而把困难的『无上下文反向/潜在测试』转化为容易的『有上下文』情形。论文还把认知科学中的海马在线/离线回放（replay/preplay）解释为同一复用机制在不同时间点的实例。

**学习维度 / Learning**

- **学习范式**: 混合：基底是参数化学习（从零训练 decoder-only transformer / IMPALA RL 智能体），但关键泛化能力来自非参数化的情景检索 + 上下文内学习（in-context learning）。论文核心命题即『参数化学习与（非参数化）情景记忆互补』。
- **失败学习 (核心主题)**: 不适用于传统意义的『失败学习』（无自我反思失败轨迹、无负样本记忆、无错误驱动规则）。论文研究的是另一类『失败』——参数化学习在潜在学习（latent learning）上的系统性泛化失败：它系统性地刻画并复现了基线 transformer 在 latent 测试上几乎为零的表现（反向关系『reverse, no context』、latent 编码索引、latent 导航目标），并以此动机引入检索来弥补；但弥补手段是检索而非从失败中学习。可视为对『模型何时/为何不泛化』这一失败模式的诊断性研究。
- **技能/程序归纳**: 不诱导可复用技能/工作流。论文明确区分两类能力：(1) 参数化学习确实能习得可迁移的『程序性泛化』（procedural generalization，如把反转/编码这类过程套用到新实例）与『相似性泛化』；(2) 它无法做到的『潜在学习』则靠检索原始情景 + ICL 解决。因此程序性知识由参数学习提供，而非从经验中被显式归纳成技能模块。
- **在线 vs 离线**: 两者皆涉及，且训练与测试时都使用检索。实验中 oracle 检索在训练和测试阶段均启用（per-episode 在线提供相关情景）；论文讨论部分进一步把『测试时（在线）情景检索』与『训练时（离线）数据增广/preplay』视为互补，对应自然智能中海马在线与离线回放的双重作用。

**评测 / Evaluation**

- **任务领域**: 受控合成域 + 简单具身导航：(1) 编码/解码（codebooks）；(2) 关系反转事实推理（simple reversals）；(3) 嵌入自然文本的语义结构推理（rephrasing、reversals、syllogisms 三段论、category-inclusion 类别归纳）；(4) gridworld 迷宫导航（latent gridworld，含基于像素的 RL 与基于 ASCII 的行为克隆 BC 两种实例），灵感来自 Tolman 1948 鼠类潜在学习实验。
- **基准**: 均为论文自建/改编的受控基准，非社区标准基准：Codebooks（40 输入 token→128 输出 token 的码本映射，含 latent 码本留出对）；Simple Reversals（1000 实体、20 关系+20 反关系，训练 20000 正向 + 19800 反向，留出 200 反向作测试，含 12.5% 同文档正反向 ICL 序列；改编自 Berglund 2024 反转诅咒与 Lampinen 2025）；Semantic Structure（1100 实体、11000 文档，64 选项多选；改编自 Lampinen 2025，含强/弱相似性线索两版）；Latent Gridworld（基于 Chan 2022 Zipfian Gridworld，每迷宫 20 物体，15 个训练导航目标、5 个 latent 留出目标；RL 从像素 + BC 从 ASCII）。
- **报告增益**: 核心定性结论为：基线纯参数化 transformer 在『潜在测试』上几乎为零或接近随机水平，而加入 oracle 检索后在这些 latent 测试上达到显著高于随机的表现，且不损害基线已能完成的验证任务。具体（图示，非完整数值表）：Codebooks 与 Simple Reversals——基线在 latent 编码 / 反向（无上下文）测试上接近 0，检索模型大幅高于随机并接近其他验证条件；Semantic Structure——强相似线索时基线即可较好泛化，减弱相似线索后检索优势更明显但整体增益较温和（归因于缺少干净的 ICL 示例）；Gridworld（RL 与 BC 两版）——检索在 latent 目标导航成功率上均显著高于基线，但仍远低于验证（trained-objects）天花板。关键消融——同文档内 ICL 示例对发挥检索效益至关重要：移除 ICL 序列后，尽管训练和测试都有 oracle 检索，Simple Reversals 的 latent 测试仅约 12%、Codebooks 的 latent 测试仅约 6%，gridworld 同样在 latent 任务上失败。其他消融排除了『仅因 batch 中 token 更多 / 数据增广 / 序列更长』等替代解释（检索无关情景不带来 latent 提升；增大 batch 甚至训练额外 token 也无 latent 提升）。误差棒为 4 次运行的 95% CI（RL 为 3 次运行 bootstrap CI）。
- **对比基线**: 主要对照为『无检索的纯参数化基线 transformer / RL 智能体』（baseline，仅靠权重中参数化知识泛化）。消融中额外对照：增大 batch size（并训练额外 token）、检索无关/干扰情景的增广、增加序列长度，以及『有 oracle 检索但训练数据中缺少 ICL 支持序列』的 Retrieval(No-ICL) 变体。隐含对照对象还包括 RAG（论文把自身结果解释为对 RAG 为何有效的新视角）。

**分析 / Analysis**

- **关键创新**: 把认知科学的『潜在学习（latent learning）』概念引入机器学习，作为统一解释一系列泛化失败（语言模型的反转诅咒、跨语言知识迁移失败、多跳推理失败、智能体对新导航目标的泛化失败）的框架，并论证这些失败源于参数化学习无法灵活使用训练经验中『潜在』的信息；进而证明（用 oracle 检索）把原始情景调回上下文即可弥补该缺口，并识别出关键前提：同经验内（within-experience）的上下文内学习是模型学会跨检索经验使用信息的必要条件。它为 RAG/情景记忆为何有效提供了基于互补学习系统的新理论解释。
- **局限**: (1) 使用 oracle（理想）检索，刻意回避真实检索难题——实际系统受限于能否从众多相关经验中检索到正确情景；(2) gridworld 上即便加检索，latent 目标表现仍远低于天花板，表明用记忆驱动长序列动作比回忆原子事实更难；(3) 全部为从零训练的小规模受控合成基准，未在真实大规模预训练 LLM 或标准社区基准上验证，规模化证据不足；(4) 无公开代码（依赖 DeepMind 内部基础设施），复现性受限；(5) 形式化框架（假设 x、t、f 可分离）比实际语言建模中观测/任务/输出的纠缠情形更受限，作者自承未完全刻画 latent learning 的全部形态；(6) 不涉及遗忘、记忆更新、冲突解决、多智能体、安全/隐私等工程维度。
- **与其他工作关系**: 属于本研究 B 类（情景记忆与检索）的认知科学/理论锚点条目。它与 A 类自改进/经验复用系统（如 Reflexion、Voyager、ReasoningBank）形成对照：后者多在部署时把经验蒸馏为反思/技能/规则并存入记忆，而本文不做抽象、保留原始情景并把灵活性推迟到测试时的上下文内推理。它为 RAG（Lewis 2020）提供了『为何有效』的潜在学习新解释，并与训练时增广/数据扩充方法（Akyürek 2024、Lampinen 2025、Yang 2025）和 preplay（Carvalho 2025）互补（离线缓存解 vs 在线检索解）。理论根基直接承接互补学习系统理论（McClelland 1995；Kumaran 2016）与海马回放/preplay 文献，并呼应将 transformer 注意力类比为脑内键-值记忆的工作（Gershman 2025；Whittington 2022）。作者团队前作 Lampinen 等 2025（上下文 vs 权重泛化差异）是其直接前身。
- **可复现性**: 复现性较弱：无公开代码与数据集发布链接；基准为自建且依赖 DeepMind 内部 JAX 生态（IMPALA、Zipfian Gridworld 等），细节见附录 B；arXiv 提供论文与 HTML 版本。社区采用以理论引用为主（约 16 次引用，含 OpenReview 等后续工作引述其潜在学习/情景知识绑定概念），尚无基于其代码的复现报告。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 无。检索/记忆管理策略不通过 RL 或训练习得，而是 oracle（人为提供相关情景 + 随机干扰项）的固定启发式设置；论文明确把『学习何时/检索什么』留作未来方向。
- **记忆主体**: 智能体中心（agent-centric）——记忆的是模型/智能体自身的训练经验（文档、码本、导航轨迹），用于改善其自身在未来任务上的泛化，而非记住用户信息做个性化。但其经验复用方式（检索原始情景而非蒸馏策略）与 Voyager/ReasoningBank 这类自改进系统不同，更接近通用的『把训练经验调回上下文』。
- **多智能体记忆**: 不适用（单模型/单智能体设置，无多智能体共享或路由记忆）。
- **时序推理支持**: 不适用（不显式建模时间有效性/事件顺序/事实有效期窗口）；任务涉及的『顺序』限于关系反转与导航轨迹，而非时间推理。
- **模态**: 多模态（文本 + 像素具身）：语言类任务为纯文本（codebooks、reversals、semantic structure、ASCII gridworld），RL gridworld 为基于像素的视觉观测，体现文本与视觉/具身两种情景记忆。
- **过度个性化/记忆安全风险**: 不适用（论文不涉及个性化、有害/陈旧/侵入式记忆、谄媚或隐私治理等记忆安全维度）。
- **冲突/矛盾处理**: 不适用（无矛盾/冲突事实的解析或合并机制）；情景以原始形式并存，论文未处理更新时的冲突。
- **token成本/延迟证据**: 无量化的成本/延迟节省数据；论文是受控科学实验而非效率优化系统。它通过消融控制了 token 数量（证明检索的收益并非来自 batch 中更多 token 或更长序列），但未报告延迟、token 成本或与全上下文/其他记忆层的效率对比。论文整体动机之一是解释当前机器学习相对自然智能的数据低效性，但未给出推理期成本数字。

**不确定字段 / Uncertain**

- 代码链接 (`code_url`)


## C. 经验回放与技能/程序记忆 (Experience replay & skill/procedural)


<a id="c1-voyager首个-llm-驱动的-minecraft-终身学习具身智能体核心为不断增长的技能库-skill-library"></a>

### C1 Voyager

*Voyager（首个 LLM 驱动的 Minecraft 终身学习具身智能体；核心为不断增长的技能库 skill library）*


**基本信息 / Provenance**

- **年份**: 2023（arXiv 预印本 2305.16291，2023-05-25 首次公开）
- **作者/机构**: Guanzhi Wang、Yuqi Xie、Yunfan Jiang、Ajay Mandlekar、Chaowei Xiao、Yuke Zhu、Linxi (Jim) Fan、Anima Anandkumar；主要机构为 NVIDIA，合作机构包括 Caltech、UT Austin、Stanford、UW Madison。通讯作者为 Guanzhi Wang 与 Linxi Fan。
- **发表venue**: arXiv 2023 预印本；正式发表于 TMLR 2024（Transactions on Machine Learning Research，DBLP: journals/tmlr/WangX0MXZFA24）
- **论文链接**: https://arxiv.org/abs/2305.16291
- **代码链接**: https://github.com/MineDojo/Voyager（MIT 许可，约 6937 stars、672 forks；项目主页 https://voyager.minedojo.org/）
- **引用数**: 约 1766 次引用（Semantic Scholar 实时数据，截至核实时），属于该领域高影响力工作

**记忆分类 / Taxonomy**

- **记忆类型**: 以程序性记忆（procedural memory）为核心——存储可执行代码形式的技能；同时通过完成/失败任务列表与课程进度体现轻量的情景记忆（episodic），并借助 GPT-4 内化的世界知识充当语义记忆。无独立的工作记忆模块，工作上下文由迭代提示窗口承担。
- **记忆结构**: 外部技能库（skill library），底层为向量数据库；每条技能是一段可执行 JavaScript（Mineflayer API）程序，由其自然语言描述的嵌入向量作为索引键、程序本身作为值。技能之间可组合调用，形成层级化的可复用行为库。
- **存储后端**: 向量数据库（以技能描述的 text-embedding-ada-002 嵌入为键，程序代码为值）；技能与检查点以文件形式持久化到磁盘（ckpt_dir / skill_library_dir，支持 resume 与跨世界加载）。
- **持久化**: 外部持久化存储：技能库独立于 LLM 之外、可跨 episode 与跨 Minecraft 世界保存与重载，不写入模型参数（黑盒查询 GPT-4，不做任何微调），因此为非参数化的外部持久记忆。

**核心机制 / Mechanisms**

- **写入/编码**: 当一个由自动课程提出的任务通过自验证模块确认成功后，将该轮迭代提示生成并验证过的可执行程序（技能）写入技能库；写入前由 GPT-3.5 为该程序生成一段通用的自然语言描述，再用 text-embedding-ada-002 把描述编码为嵌入向量作为索引键，程序代码作为值存储。技能被刻意要求写得通用、可复用，以便日后被更复杂的技能组合调用。因此写入的是经过验证、抽象化、可执行的高层行为，而非原始轨迹逐帧记录。
- **检索机制**: 基于嵌入相似度的向量检索：面对自动课程提出的新任务时，先用 GPT-3.5 生成解决该任务的通用建议（self-generated task plan），将其与环境反馈拼成查询上下文，编码为嵌入向量后在技能库中检索 top-5 最相关技能，作为上下文示例注入 GPT-4 的代码生成提示中。检索为纯启发式相似度匹配，无学习型检索器、无 recency/importance 加权评分、无图遍历。
- **反思/巩固**: 通过自验证（self-verification）模块实现 raw→insight 的转化：实例化另一个 GPT-4 智能体充当 critic，输入当前状态与任务，判断程序是否完成任务；若失败则给出批评与改进建议（critique）。论文强调这比 Reflexion 的自反思更全面（既检查成功又反思错误）。固化层面，只有验证成功的程序才被抽象/泛化后写入技能库，这一步即把单次成功经验巩固为可长期复用的高层技能；技能可被更复杂技能组合，从而缓解灾难性遗忘。
- **遗忘/更新**: 本质上是只增不减（ever-growing）的技能库，没有显式的遗忘、衰减或失效机制；新技能在原有技能之上组合构建，论文称这种增量方式可缓解持续学习中的灾难性遗忘。更新主要体现在同一任务多轮迭代提示中对代码的逐轮 refine，而非对已入库技能的编辑/合并/去重。
- **经验回放 (核心主题)**: 核心主题。Voyager 通过技能库实现经验复用：过去成功完成任务所生成的可执行程序被存档，并在遇到相似情境时被检索回来作为上下文示例（exemplar）注入提示，或作为子程序被更复杂技能直接调用组合。这使能力随时间快速复利累积（compounding）。实验证明该复用能跨世界、跨任务迁移：把学到的技能库带到全新 Minecraft 世界可零样本解决未见任务（如 19±3 次迭代造出钻石镐），而无技能库版本明显更慢或失败；技能库甚至能即插即用提升 AutoGPT 的表现。这是一种以代码技能为载体的程序性经验回放，而非原始轨迹的强化学习式 replay buffer。

**学习维度 / Learning**

- **学习范式**: 非参数化（in-context / prompt-level）：完全通过对 GPT-4 的黑盒提示与上下文学习实现，不访问模型参数、不做梯度更新或微调。学习体现在外部技能库的不断增长与课程的自适应推进上。
- **失败学习 (核心主题)**: 核心主题。Voyager 通过迭代提示机制从失败中学习：每轮执行生成的程序后，收集三类反馈——(1) 环境反馈（如『缺少 7 个铁锭』揭示失败原因）；(2) 代码解释器的执行错误/语法错误（用于修 bug）；(3) 自验证模块判定的失败信号及其改进批评（critique）。这些反馈被并入下一轮 GPT-4 提示进行代码修正，最多迭代 4 轮；若仍卡住则放弃当前任务、由自动课程稍后再安排重试。失败任务还会被记入课程的『已失败任务』列表以反映能力边界。失败经验主要用于即时的代码自我修正与课程调度，而非沉淀为可检索的负面示例库或失败模式记忆。
- **技能/程序归纳**: 是，且为其招牌能力：从自驱探索经验中归纳出可复用、可解释、时序延展、可组合的技能，表示为带自然语言描述的可执行代码（如 craftStoneShovel()、combatZombieWithSword()）；通过向量检索按相似情境调用，并可被更复杂技能组合调用，从而随时间复利式扩展能力。
- **在线 vs 离线**: 在线（online）构建：技能库在部署/游玩过程中逐 episode、逐任务地实时增长，由自动课程驱动持续探索；不依赖离线轨迹语料的批量训练。学到的库可保存后在新世界继续在线复用。

**评测 / Evaluation**

- **任务领域**: 具身/开放世界游戏——Minecraft 终身学习（探索、技能掌握、地图遍历、科技树进阶、零样本迁移到新世界的未见任务）。单一域、无 web/QA/对话等其他领域评测。
- **基准**: 在 MineDojo（开源 Minecraft AI 框架）之上构建评测，使用 Mineflayer JavaScript API 做底层控制；评测协议自定义：唯一物品发现数、Minecraft 科技树里程碑解锁速度（木→石→铁→钻石）、地图遍历距离、新世界零样本任务（钻石镐、金剑、熔岩桶、指南针）。无外部标准记忆 benchmark（如 LoCoMo/WebArena）。
- **报告增益**: 相对此前 SOTA/基线（ReAct、Reflexion、AutoGPT，均改造到 MineDojo）：在 160 次提示迭代内发现 63 种唯一物品，约为基线的 3.3×；解锁科技树里程碑最快达 15.3×（木质工具 15.3×、石质 8.5×、铁质 6.4× 更快），且是唯一解锁钻石级的方法（102 次迭代、3 试中 1 成）；地图遍历距离 2.3× 更长。零样本泛化到新世界：Voyager 全部 4 个未见任务均 3/3 成功（钻石镐 19±3、金剑 18±7、熔岩桶 21±5、指南针 18±2 次迭代），而 ReAct/Reflexion/AutoGPT 在 50 次迭代内 0 通过；技能库还能即插即用提升 AutoGPT 表现。消融：去掉自动课程物品数下降约 93%，去掉自验证下降约 73%，用 GPT-3.5 代替 GPT-4 仅获 1/5.7 的物品（GPT-4 多 5.7×）。成本：GPT-4 API 比 GPT-3.5 贵约 15×。
- **对比基线**: ReAct（思维链+行动）、Reflexion（在 ReAct 上加自反思）、AutoGPT（目标分解+ReAct 式循环），均为原本面向 NLP、被改造进 MineDojo 的 LLM 智能体；同时含内部消融基线（Voyager w/o Skill Library、随机/人工课程、各反馈类型剔除、GPT-3.5 替代 GPT-4）。不直接与基于像素输入/低层控制的 RL 方法（VPT、DreamerV3 等）对比，因控制接口不同。

**分析 / Analysis**

- **关键创新**: 首个无需人工干预、不微调模型参数的 LLM 驱动具身终身学习智能体；核心创新是『不断增长的可执行代码技能库』作为程序性记忆——把已验证经验固化为可检索、可组合、可解释、可跨世界迁移的代码技能，配合自动课程与含自验证的迭代提示机制，实现能力的复利式累积并缓解灾难性遗忘。
- **局限**: 成本高（依赖昂贵的 GPT-4，比 GPT-3.5 贵约 15×，开源/GPT-3.5 模型无法替代）；存在不准确性（智能体偶尔卡住、自验证可能误判，如不认成功信号）；GPT-4 幻觉（课程提出不存在的物品如『铜剑』，代码用无效燃料或调用不存在的 API）；当时不支持视觉感知（GPT-4 API 仅文本）；技能库只增不减、无真正遗忘/失效机制；评测局限于单一 Minecraft 域。
- **与其他工作关系**: 在本研究中属于『C. 经验回放与技能/程序性』簇的代表：把记忆实现为可执行代码技能库，与 A1 Reflexion 形成对照——论文明确指出其自验证比 Reflexion 的自反思更全面（既判定成功又反思错误），并以 ReAct、Reflexion、AutoGPT 为直接基线。它代表 agent-centric（记住自身经验以自我提升）路线，与 Mem0/Zep/LongMemEval 等 user-centric（记住用户信息以个性化）记忆系统形成鲜明对比，后续 ReasoningBank 等自我改进记忆工作可视为同一谱系。其『技能库即记忆』思路启发了大量后续具身/程序性记忆工作（如 MindForge 在其基础上加入心智理论与多组件记忆系统并显著超越 Voyager 基线）。
- **可复现性**: 可复现性强：作者在 https://github.com/MineDojo/Voyager 开源完整代码库与全部提示（MIT 许可，约 6937 stars、672 forks、20 名贡献者），提供安装、断点续训（resume）、加载学到的技能库与任务分解推理等说明；依赖闭源 GPT-4 API 与 Minecraft/Mineflayer 环境，故复现需付费 API 与游戏环境搭建，存在一定门槛但社区采用度高。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式管线）。技能的写入/检索完全由固定流程与嵌入相似度决定，未用 RL 或训练去学习『何时/写什么/检索什么』的记忆管理策略；写入门控由 GPT-4 自验证启发式判定，检索为固定 top-5 相似度。属于 2025-26『学习型记忆控制』分水岭之前的启发式一代。
- **记忆主体**: Agent-centric（以智能体为中心）：记住的是智能体自身在环境中获得的可复用技能与探索经验，用于自我改进与跨任务/跨世界迁移，而非记住用户信息做个性化。
- **多智能体记忆**: 单智能体记忆。技能库服务于单个 Voyager 智能体；不过它内部用多个 GPT 角色实例（GPT-4 做课程/代码生成/自验证 critic，GPT-3.5 做描述生成与自问自答）。技能库可被其他方法（如 AutoGPT）即插即用复用，但不属于多智能体共享/路由记忆架构（如 G-Memory、MIRIX）。
- **时序推理支持**: 不显式建模时间有效性或事件排序；不维护事实有效期窗口或事件日历。仅在课程提示中维护『已完成/已失败任务』的进度列表以反映能力边界，不涉及时间推理。
- **模态**: 纯文本（text-only）。论文撰写时 GPT-4 API 仅支持文本，不支持视觉感知；可经人类多模态反馈构建 3D 建筑，但记忆与控制本身基于文本/代码。
- **过度个性化/记忆安全风险**: 未涉及该维度。作为 agent-centric 经验型记忆，论文不讨论过度个性化、有害/陈旧/谄媚记忆或隐私治理；相关风险更多体现为 GPT-4 幻觉导致入库无效技能或不可达任务，但无专门的记忆安全/治理机制。
- **冲突/矛盾处理**: 无显式的冲突/矛盾事实解决机制。技能库只增不减，不对已入库技能做冲突检测、合并或失效更新；唯一的『纠错』发生在同一任务的多轮代码迭代中（用执行错误与环境反馈修正当前程序），不针对库内技能间矛盾。
- **token成本/延迟证据**: 未提供 token/延迟节省的量化优势数据；相反，论文从成本角度指出依赖 GPT-4 代价高昂（GPT-4 比 GPT-3.5 贵约 15×），并出于预算考虑把部分标准 NLP 子任务（描述生成、自问自答）交给更便宜的 GPT-3.5。无 p95 延迟或输入 token 削减比例等效率证据。


<a id="c2-agent-workflow-memoryawm智能体工作流记忆"></a>

### C2 Agent Workflow Memory

*Agent Workflow Memory（AWM，智能体工作流记忆）*


**基本信息 / Provenance**

- **年份**: 2024（arXiv 预印本，2024-09-11 首次公开；正式发表于 ICML 2025）
- **发表venue**: ICML 2025（第 42 届国际机器学习大会，PMLR 第 267 卷，论文编号 wang25bx）；首发于 arXiv（cs.CL/cs.AI，2024-09），属学术界研究成果并开源代码。
- **论文链接**: https://arxiv.org/abs/2409.07429
- **代码链接**: https://github.com/zorazrw/agent-workflow-memory（官方开源，Python，Apache-2.0 许可，约 442 stars / 50 forks，截至 2026-06）
- **引用数**: 约 174 次引用（Semantic Scholar，截至调研日；influentialCitationCount 约 16；paperId c68cc84ec7808d7bbd5686a6bd1393752a9d8e8d）。

**记忆分类 / Taxonomy**

- **记忆类型**: 程序性记忆（procedural）为主：将过往轨迹中可复用的子例程归纳为「工作流（workflow）」并存入文本记忆。本质上是 CoALA 框架中的程序性记忆（可复用技能/例程），其抽象层级高于原始情景轨迹（episodic），但工作流由具体经验归纳而来，带有情景蒸馏色彩。
- **记忆结构**: 扁平的工作流条目集合（workflow memory）。每条工作流包含两部分：（1）一段文本描述 d（说明该工作流的高层目标，如「查找指定 ID 的客户订单」）；（2）一串经过抽象的动作步骤序列 (p1,p2,…)。工作流以文本形式存储，按双换行符切分、分条独立保存；无图结构、无层级。基础记忆中还含有内置动作（CLICK、TYPE 等）的文档。
- **存储后端**: 纯文本记忆 M（in-context 文本记忆），工作流以自然语言/伪代码文本形式直接拼接进智能体的上下文（system prompt / 记忆段）。论文未使用向量数据库或图数据库；记忆本身即语言模型可读的文本块。
- **持久化**: 外部化的文本记忆库（durable external store），跨任务持久保留并持续增长；不修改模型参数（非参数化）。检索/选用的工作流以临时上下文形式注入提示。在线模式下记忆随任务流逐步累积，呈「滚雪球」式扩展。

**核心机制 / Mechanisms**

- **写入/编码**: 核心是「LM 工作流归纳模块 I」：将一条或多条过往经验 E={e_i}（每条经验 e=(q,P^e) 含自然语言任务指令 q 与观察-动作步骤序列 P^e）输入 LLM，提示其抽取出常被复用的子例程，产出工作流集合 W={(d_j,P_j^d)}。与逐字保存轨迹不同，AWM 刻意在更细的粒度上归纳（例如从「在亚马逊买猫粮并寄到我家」中抽出反复出现的子任务「在亚马逊搜索某商品」），并通过把示例特定内容抽象掉来增强通用性（如把「dry cat food」替换为占位符「{product-name}」）。论文还实现了一个对照用的规则式归纳 I_rule：先抽取动作序列（如 CLICK→CLICK→TYPE）、按动作序列去重，但不做上下文与子例程抽象。网页以可访问性树（accessibility tree）表示（BrowserGym 框架）。
- **检索机制**: 选择性地将工作流提供给智能体以指导后续生成。在论文主设定中，归纳出的工作流被整合进智能体的文本记忆/系统提示，作为可复用例程供模型在生成动作时参考；记忆刻意保持简单（文本拼接 + 选用相关工作流），未强调复杂的向量重排或学习型检索。论文另探索了 AWM_AS 变体，将工作流作为扩展的动作空间（action space）而非仅作记忆注入。检索强调「选择性提供」与高层目标描述，使智能体能定位到与当前任务相关的子例程。
- **反思/巩固**: 工作流归纳本身即「原始轨迹→高层可复用例程」的抽象/巩固过程：LM 归纳模块从经验中提炼高层目标描述 d 与抽象动作步骤，剥离示例特定细节。在线模式下形成持续学习闭环：解任务→对成功轨迹归纳新工作流→并入记忆→指导后续更复杂任务，产生「滚雪球效应」，逐步从简单工作流组合出更复杂的工作流（如在「查找地点」基础上叠加新步骤构建更复杂工作流），从而随时间不断扩展记忆并拉开与不自适应的 vanilla 智能体的性能差距。
- **遗忘/更新**: 更新机制较简单：规则式归纳会按动作序列对经验去重（deduplicate）；在线模式以「添加/整合（integrate）」为主，将新归纳的工作流并入记忆，缺乏显式的遗忘、衰减、合并或失效（invalidation）机制。无 Ebbinghaus 式衰减，也无 UPDATE/DELETE 式细粒度编辑。
- **经验回放 (核心主题)**: 这是论文的核心主题，属「经验复用/技能复用」范式。AWM 不重放原始轨迹，而是把过往（成功）经验蒸馏为抽象、可复用的「工作流/子例程」并在未来任务中选择性复用，以指导动作生成、避免长程任务中重复摸索。相较仅检索具体示例的方法（如 Synapse 检索相关轨迹示例），AWM 复用的是抽象掉示例特定内容的子例程，对元素选择引入更少偏置、泛化性更强。它支持离线（从训练/标注示例预先归纳工作流）与在线（无辅助数据，从测试过程中自身轨迹即时归纳并复用）两种复用模式，是「从经验归纳可复用技能」这一思路在网页智能体上的代表性实现。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric）/ 提示层（prompt-level）。不进行任何梯度更新，仅通过文本工作流记忆的累积与上下文注入实现自我改进。归纳模块与动作生成共用同一 LLM 骨干。
- **失败学习 (核心主题)**: AWM 主要从成功经验中学习：在线模式下用一个 LM 评估器（neural evaluator，输出二元标签 L_eval(e_t)∈{0,1} 判断轨迹 e_t 是否成功解决任务 q_t），仅对被判为成功的轨迹归纳工作流并入记忆，失败轨迹一般被滤除而非显式提炼为「避坑」教训。这与后续 ReasoningBank（A6）形成对比——后者明确指出 AWM 仅利用成功轨迹、无法有效从失败中学习，并在消融中显示 AWM 纳入失败信号反而可能掉点（44.4→42.2）。因此 AWM 的失败学习能力有限，是其相对短板。
- **技能/程序归纳**: 是，且为核心贡献。AWM 显式地从智能体轨迹中归纳可复用的工作流/子例程（reusable routines），以「文本描述 + 抽象动作步骤序列」表示，通过整合进记忆（或扩展动作空间 AWM_AS）来调用。工作流可被组合，复杂工作流可在简单工作流基础上叠加构建。
- **在线 vs 离线**: 两者皆可（both，论文重要卖点）。离线（offline）：当有标注/训练示例（如 ground-truth 标注的规范经验）时，预先从中归纳工作流再用于测试。在线（online）：无任何辅助数据时，从测试查询的自身过往轨迹即时归纳工作流并立即复用，按任务流增量构建记忆。

**评测 / Evaluation**

- **任务领域**: 网页导航/网页自动化（web navigation / web automation），覆盖旅行、购物、社交媒体、地图、论坛、代码托管（GitLab）、内容管理（CMS）等域，合计 1000+ 任务、200+ 域。
- **基准**: 两大网页导航基准：（1）WebArena（执行结果评测，5 个网站子域 Shopping/CMS(shopping_admin)/Reddit/GitLab/Maps）；（2）Mind2Web（广覆盖，含 cross-task / cross-website / cross-domain 泛化测试划分）。环境采用 BrowserGym 框架、以可访问性树表示网页。
- **报告增益**: WebArena（GPT-4，gpt-4-0613）：总体成功率（Total SR）由 BrowserGym 基线 23.5% 提升至 35.5%（AWM LM 归纳）/ 35.6%（AWM 规则归纳），即约 +12.0 绝对点、约 51.1% 相对提升，达到当时 SOTA（论文称 35.6%），并超过用人类专家手写工作流增强的方法（SteP，Sodhi et al.）；同时减少解题平均步数（成功用例上较 BrowserGym 基线减少若干步）。Mind2Web（GPT-4）：cross-task 相对步级成功率提升约 24.6%；在 cross-website / cross-domain 泛化测试中较基线高出约 8.9–14.0 绝对点（步级成功率改进约 14.0–16.9 绝对点），且随训练-测试分布差距加大、优势更显著。摘要总结：Mind2Web +24.6%、WebArena +51.1% 相对成功率提升，并减少 WebArena 成功解题步数。
- **对比基线**: 无工作流/无记忆的基线智能体；WebArena 上对比 BrowserGym（Drouin et al. 2024，含 ax-tree 仅可访问性树版本 BrowserGym_ax-tree）与 SteP（Sodhi et al.，用人类专家手写工作流）；Mind2Web 上对比 MindAct（gpt-4，含网页元素过滤+多选格式）与 Synapse（Zheng et al.，轨迹式格式+检索相关具体示例）。AWM 用 gpt-3.5-turbo 与 gpt-4 两种骨干、temperature 0 运行以公平对比。核心对照在于「抽象可复用工作流」对「检索具体示例」的优越性。

**分析 / Analysis**

- **关键创新**: 提出从智能体轨迹中归纳「抽象、可复用的工作流（子例程）」并整合进文本记忆以指导后续任务，且同一框架可在离线（从标注示例预归纳）与在线（无辅助数据、从自身轨迹即时归纳并复用）两种场景灵活运行；通过抽象掉示例特定内容获得更强跨任务/跨网站/跨域泛化，并形成「滚雪球式」持续学习——这是「记忆即可复用技能」在网页智能体上的里程碑式实现。
- **局限**: （1）主要利用成功轨迹，失败学习能力弱——在线模式依赖 LM 评估器自判成功，失败轨迹被滤除而非提炼为教训（ReasoningBank 指出 AWM 纳入失败反而掉点）；（2）记忆管理简单：仅做添加/去重，无显式遗忘、衰减、合并、冲突消解或失效机制，随任务增多工作流库可能膨胀、含冗余或过时例程；（3）评测局限于网页导航单一大类、纯文本（可访问性树）观察，未覆盖多模态/具身/对话；（4）依赖商用 LLM（GPT-3.5/4）与 LM 评估器，结果对模型版本与自判准确性敏感；（5）智能体在「何时偏离工作流指引」上仍有困难，工作流过强可能误导。
- **与其他工作关系**: 属于「C. 经验回放 & 技能/程序性记忆」簇。它把「从经验归纳可复用技能/例程」的思路（同源于 Voyager 的技能库、CLIN/ICAL 的持续学习与轨迹转化为可操作洞见）落地到网页导航。相较 Synapse（检索复用具体轨迹示例）与 MindAct（元素过滤+多选），AWM 复用的是抽象掉上下文的工作流，泛化更好、偏置更小。它常被后续工作作为关键基线：ReasoningBank（A6）明确将 AWM（仅从成功工作流学习）作为对比对象，指出其无法从失败中学习，并主张蒸馏更抽象、含失败教训的「推理策略」；与 Reflexion（A1，从失败自反思）互补——AWM 偏成功例程归纳，A6/A1 偏失败/反思驱动。属智能体中心（agent-centric）自我改进记忆，区别于 Mem0/Zep/LongMemEval 等用户中心记忆。其在线归纳的 LM 评估器机制与后续自判（LLM-as-a-judge）类方法一脉相承。
- **可复现性**: 可复现性良好：官方开源代码（github.com/zorazrw/agent-workflow-memory，Python，Apache-2.0，约 442 stars，提供 WebArena 与 Mind2Web 两套 pipeline.py 与数据/环境配置说明）；所用基准（WebArena、Mind2Web）均为公开数据集；已正式发表于 ICML 2025、引用约 174 次、社区采用度高，常被作为标准基线复现。主要不确定性来自对闭源商用 LLM（GPT-3.5/4）与 LM 评估器的依赖，结果对模型版本敏感。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式流程）。工作流的归纳、整合、选用与去重均为固定的非学习型流程（LM 提示归纳 + 文本整合 + 动作序列去重），不使用 RL/训练来学习「何时/存什么/取什么」的记忆管理策略本身。处于 2025-26「学习型记忆控制」分水岭中的启发式一侧（与 Memory-R1、Mem-α 等学习型方法相对）。
- **记忆主体**: 智能体中心（agent-centric）：记忆的是智能体自身解题经验所归纳出的可复用工作流，目的是让网页智能体跨任务自我改进、提升长程任务成功率，而非记住用户个人信息做个性化。与 Voyager、ReasoningBank 同类，区别于 Mem0/Zep/LongMemEval 等用户中心记忆。
- **多智能体记忆**: 单智能体（single-agent）。工作流记忆服务于单个网页智能体的离线/在线学习，未涉及多智能体间共享或路由记忆。
- **时序推理支持**: 否。不显式建模时间有效性、事件先后或事实时效窗口；工作流是与时间无关的程序性例程，记忆累积也无时间衰减/刷新策略。
- **模态**: 纯文本（text-only）。网页观察以文本化的可访问性树（accessibility tree）表示，工作流为文本描述+动作步骤；无视觉截图、具身或多模态记忆。
- **过度个性化/记忆安全风险**: 未涉及。论文不处理用户个性化，也未讨论有害/过时/侵入性记忆、隐私治理或过度个性化风险（这类属用户中心记忆的安全维度，超出本工作范围）。
- **冲突/矛盾处理**: 基本未处理。记忆整合仅做添加与按动作序列去重，无显式的冲突/矛盾工作流检测与合并/消解机制；过时或相互矛盾的工作流缺乏专门的更新或失效流程（作者将更复杂的记忆管理留待未来）。

**不确定字段 / Uncertain**

- 作者/机构 (`authors_institution`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="c3-synapse轨迹即范例提示--范例记忆trajectory-as-exemplar-prompting-with-memory"></a>

### C3 Synapse

*Synapse（轨迹即范例提示 + 范例记忆；Trajectory-as-Exemplar Prompting with Memory）*


**基本信息 / Provenance**

- **年份**: 2023（arXiv 预印本 2306.07863，2023-06-13 首次公开；v3 于 2024-01-19 更新）
- **作者/机构**: Longtao Zheng（郑龙韬，第一作者）、Rundong Wang、Xinrun Wang、Bo An（安波，通讯）；全部来自新加坡南洋理工大学（NTU, Singapore）。第一作者邮箱 longtao001@e.ntu.edu.sg。
- **发表venue**: ICLR 2024（The Twelfth International Conference on Learning Representations，正式会议论文，DBLP: conf/iclr/ZhengWW024）；arXiv 首发于 2023 年 6 月。属于学术界成果并开源代码。
- **论文链接**: https://arxiv.org/abs/2306.07863（OpenReview: https://openreview.net/forum?id=Pc8AU1aF5e）
- **代码链接**: https://github.com/ltzheng/Synapse（官方开源，约 69 stars、12 forks，主语言 HTML/JavaScript/Python，Python 3.10，NOASSERTION 许可；项目主页 https://ltzheng.github.io/Synapse/，截至 2026-01 仍有更新）

**记忆分类 / Taxonomy**

- **记忆类型**: 情景性记忆（episodic）为主：存储过去成功完成任务的完整交互轨迹（trajectory）作为可检索的少样本范例（exemplar）。这些轨迹同时隐含程序性（procedural）知识——即「如何一步步完成某类任务」的操作流程，但 Synapse 并不把它抽象成显式技能/规则，而是以轨迹原样保存。属于 CoALA 框架中的情景记忆，可经检索复用以支持决策。
- **记忆结构**: 键值对范例记忆 D=(K,V)：K 为任务元数据（task metadata）的嵌入向量构成的定长数组，V 为对应的状态抽象提示（state abstraction prompts）与范例轨迹（exemplary trajectories）。整体是扁平的「元数据→范例」映射的向量索引池，无层级、无图结构；每条轨迹被格式化为 ⟨task, observation, action, …, observation, action⟩ 的序列。
- **存储后端**: 外部向量数据库，使用 Faiss（Johnson et al. 2019）存储嵌入并执行相似度检索；嵌入模型为 OpenAI text-embedding-ada-002。记忆通过 build_memory.py 离线构建为 index.faiss 与 index.pkl 两个文件。检索到的轨迹在运行时以少样本范例形式注入 LLM 上下文（in-context）。
- **持久化**: 外部持久化存储（durable external store）：范例记忆在部署前由人类示范轨迹预先构建并固定保存于 Faiss 向量库，跨任务复用；不修改模型参数（非参数化），检索结果以临时上下文注入。MiniWoB++ 中由 48 个任务的人类示范构成，Mind2Web 中由训练集构成。

**核心机制 / Mechanisms**

- **写入/编码**: 以「逐字保存完整成功轨迹」为核心的编码方式（verbatim trajectory），但配合状态抽象做压缩。流程：（1）状态抽象（state abstraction）——先用 LLM 的少样本能力把原始计算机状态（如网页 HTML）过滤为「干净、任务相关」的简洁观察，去除无关元素，从而显著降低每个状态的 token 数；抽象有两种形式：显式抽象（explicit，以 ⟨state, observation⟩ 对作范例，适用于 email-inbox 等状态较短场景）与隐式抽象（implicit，以 ⟨task, code⟩ 对让 LLM 生成解析代码、再执行代码得到干净观察，适用于 book-flight 等超长复杂状态；代码执行报错时退化为零样本抽象）。（2）将抽象后的「任务描述 + 干净观察-动作序列」打包成完整轨迹范例。（3）写入记忆时，把任务元数据（MiniWoB++：任务描述拼接 5 个随机种子的初始状态；Mind2Web：网站名+域名+任务描述）经 text-embedding-ada-002 编码为键 K，对应轨迹与状态抽象提示作为值 V，存入 Faiss。范例来自人类示范，可直接转换为 TaE 格式，无需人工编写计划或 MCQ。
- **检索机制**: 基于嵌入向量的相似度检索（similarity search）。新任务到来时，先编码其任务元数据为查询向量 q，在向量库中按公式 arg top-n_{d∈D} sim(q, d) 检索最相关的 n 条范例，其中 sim 为嵌入空间中的欧氏距离（Euclidean distance）。在 MiniWoB++ 中采取「投票式」检索：先取 top-3 范例，用其中出现最多的任务类别（如 enter-date）再取回该类别全部范例，因为任务边界清晰。在 Mind2Web 中直接检索匹配元数据对应的范例。检索得到的轨迹随后作为少样本范例注入 LLM，配合当前轨迹历史生成下一动作（Trajectory-as-Exemplar 提示）。检索仅靠语义相似度，无重排序、无学习型路由。论文报告 Mind2Web 三个测试集的平均检索距离分别为 cross-task 17.0、cross-website 24.3、cross-domain 32.9（距离越大相似度越低），解释了为何记忆在 cross-domain 几乎无增益。
- **反思/巩固**: 无显式的反思/巩固/抽象机制。Synapse 不把原始经验转化为更高层的洞见、规则或技能——它直接以完整轨迹原样存储与复用，这正是其与后续工作（如 ExpeL、ReasoningBank 蒸馏出抽象推理策略）的关键区别。唯一接近「抽象」的环节是状态抽象（把原始状态压缩为任务相关观察）和 TaE 提示隐含的「时间抽象动作」（temporally abstracted actions，即一次生成多步动作、仅在必要时查询新状态），但二者都发生在写入/推理阶段，而非对入库经验做事后反思总结。记忆内容也不随使用而演化。
- **遗忘/更新**: 基本无遗忘/更新机制。范例记忆为预构建的定长键值数组，部署期间通常不增删改、无衰减（无 Ebbinghaus 类遗忘），也无去重/合并/冲突消解。这是其作为静态范例库的设计取舍；论文将「记忆结构与检索过程的进一步研究」及「结合人类干预做用户定制与任务适配」列为未来方向。
- **经验回放 (核心主题)**: 本论文核心主题。Synapse 的「经验回放」即「轨迹即范例（Trajectory-as-Exemplar, TaE）提示」：把过去成功完成的完整交互轨迹（抽象状态与动作交替的序列 ⟨task, obs, act, …, obs, act⟩）整条作为少样本范例喂给 LLM，再附上当前轨迹历史，让 LLM 生成下一步动作。相比此前方法仅用高层计划（RCI）或多选题 MCQ（MindAct）作范例——它们无法表示完整轨迹、需逐步查询 LLM 导致误差累积——TaE 提供一致且信息更丰富的交互格式，使 LLM 能生成「时间抽象动作」（连续多动作、仅在需要新状态时暂停），降低成本与延迟，提升长程任务（如 use-autocomplete、use-spinner、book-flight）的动作准确率。配合范例记忆的相似度检索，过去轨迹能自动泛化复用到新任务（无需任务专属范例），从而以 48 个任务的示范解决 64 个任务。它复用的是「原始成功轨迹」本身，而非蒸馏后的抽象策略——这是它在「经验回放」谱系中的定位（后续 AWM 复用成功工作流、ReasoningBank 复用抽象推理策略均以其为对比基线）。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric）/ 提示层（prompt-level）的上下文学习（in-context learning, ICL）为主：不更新 LLM 参数，仅靠检索范例注入上下文实现跨任务泛化。同时论文额外探索了一条参数化分支（混合 hybrid）：在 Mind2Web 上用 Synapse 风格的数据对 CodeLlama-7B 做 LoRA 微调（finetune_mind2web.py），证明其提示技术也能提升微调性能，但主体框架与主结果均为非参数化 ICL。
- **失败学习 (核心主题)**: 几乎不涉及从失败中学习。Synapse 的范例记忆只存储「成功轨迹」（successful trajectories），明确不收集失败轨迹、不蒸馏负例教训、也无失败模式记忆。它有意不依赖自我纠错（self-correction）——这正是其相对 RCI、AdaPlanner（二者靠递归自纠错）的卖点：用更优的提示结构与完整轨迹范例减少误差累积，从源头避免出错，而非事后纠错。论文分析的失败案例（如 count-shape 计数错误、text-transform 字符误识）归因于 LLM 推理本身，未被回收为学习信号。后续工作 ReasoningBank 的消融正是指出 Synapse「仅用成功轨迹、无法有效利用失败」（其报告 Synapse 加入失败后从 40.6 仅升至 41.7）。
- **技能/程序归纳**: 不进行显式技能/流程归纳。Synapse 以完整轨迹原样作为可复用单元，而非从经验中提炼出命名的、可组合的技能或工作流（与 Voyager 的技能库、AWM 的工作流记忆不同）。其「程序性」体现为轨迹隐含的操作序列和 TaE 提示产生的时间抽象动作，调用方式是相似度检索后整条注入上下文，而非按技能 API 显式调用。论文将 Voyager 式技能库视为与本框架互补的正交方向。
- **在线 vs 离线**: 离线（offline）构建为主：范例记忆由人类示范轨迹在部署前批量构建并固定（MiniWoB++ 用 48 任务示范、Mind2Web 用训练集），运行时只读检索、不在线增量写入新经验。属于「先离线建库、后在线检索复用」的范式，区别于 ReasoningBank/AWM 等测试时在线累积记忆的方法。

**评测 / Evaluation**

- **任务领域**: 计算机控制 / 网页导航（computer control, web navigation）。涵盖两类：（1）MiniWoB++ 标准化网页交互任务套件（如终端文件操作、订机票 book-flight、邮件收件箱、点击复选框、自动补全、数字猜测、井字棋等 64 个任务）；（2）Mind2Web 真实世界网站的开放域任务（涉及 Airbnb、Twitter 等多领域真实网站）。纯文本 HTML 状态、键鼠动作空间。
- **基准**: （1）MiniWoB++（Shi et al. 2017 / Liu et al. 2018，标准研究任务套件，按 RCI 配置评测 64 个任务，每任务 50 episodes）；（2）Mind2Web（Deng et al. 2023，真实网站基准，分 Cross-Task / Cross-Website / Cross-Domain 三级泛化测试集）。指标含成功率 SR、步成功率 Step SR、元素选择准确率 Ele. Acc。
- **报告增益**: MiniWoB++：64 个任务平均成功率 99.2%（仅用 48 个任务的示范），为首个达到人类水平的 ICL 方法；相对此前 ICL SOTA 有约 10% 相对提升，且不依赖自我纠错。具体对比：Synapse 0.992 > Pix2Act 0.962 > CC-Net 0.935 ≈ Human 0.935 > AdaPlanner 0.929 > RCI 0.906 > WebGUM 0.803 > Pix2Act(BC) 0.665 > WebN-T5 0.484 > CC-Net(BC) 0.305。Synapse 是首个解出超长状态 book-flight 任务的 ICL 方法；对 RCI 表现差的长程任务大幅提升：guess-number 20%→100%、use-spinner 88%→100%、use-autocomplete 58%→98%；email-inbox-nl-turk 经状态抽象 52%→100%；16 个未见任务平均成功率近 100%。Mind2Web（GPT-3.5，平均 Step SR 相对 MindAct 的相对提升）：逐步叠加三组件分别提升 32%、50%、56%。GPT-3.5 完整 Synapse（state abstraction+TaE+memory）成绩：Cross-Task Ele.Acc 34.0 / Step SR 30.6 / SR 2.4；Cross-Website 29.1 / 24.2 / 0.6；Cross-Domain 29.6 / 26.4 / 1.5（对照 MindAct 分别为 20.3/17.4/0.8、19.3/16.2/0.6、21.6/18.6/1.0）。CodeLlama-7B 上 Synapse 平均 Step SR 达 MindAct 的约 2.5 倍。记忆模块在 cross-task/cross-website 各带来约 6% Step SR 提升，但 cross-domain 几乎无增益（甚至 CodeLlama 上 -0.2）。
- **对比基线**: MiniWoB++：BC+RL 类（CC-Net、Pix2Act）、微调类（WebGUM、WebN-T5）、ICL 类（RCI、AdaPlanner，均带自我纠错），并附人类得分。Mind2Web：MindAct（该基准当时 ICL SOTA，基于 top-50 元素排序 + MCQ 递归查询）。消融对比为「无记忆/无轨迹/无状态抽象」的逐步剥离变体。骨干 LLM 为 gpt-3.5-turbo-0301（MiniWoB++）、gpt-3.5-turbo-16k-0613（Mind2Web）及 CodeLlama-7B；嵌入用 text-embedding-ada-002，温度 0 贪心解码。

**分析 / Analysis**

- **关键创新**: 提出三位一体的 ICL 计算机控制框架，核心创新是「轨迹即范例（TaE）提示」——首次用完整成功交互轨迹（而非高层计划或 MCQ）作少样本范例以改善多步决策并产生时间抽象动作；并配合「状态抽象」压缩超长原始状态以容纳更多范例、用 Faiss「范例记忆」做相似度检索实现跨任务泛化（摆脱任务专属硬编码范例）。三者使其成为首个在 MiniWoB++ 达到人类水平、且无需自我纠错、样本效率极高的 ICL 智能体。
- **局限**: （1）LLM 带来的高推理延迟（论文自陈为主要顾虑，建议用其提示蒸馏更轻量的任务专属智能体）；（2）严重依赖范例（人类示范）的质量；（3）记忆只存成功轨迹、不学失败、无反思/巩固/遗忘/更新机制，且 cross-domain 泛化时不相关范例可能干扰 LLM（记忆几乎无增益甚至轻微负作用）；（4）记忆结构与检索过程较简单（仅欧氏距离相似度），有待深化；（5）仅处理文本 HTML 状态，未支持多模态/视觉/视频（如像素级 Android 控制）；（6）失败案例多源于 LLM 推理本身（计数、字符识别错误），框架无从纠正。
- **与其他工作关系**: 属于「C. 经验回放 与 技能/程序化（Experience replay & skill/procedural）」簇，是该方向的早期奠基性工作（2023 年），后续多篇均以其为关键基线与对比对象。与同簇/相邻工作的关系：（a）相对 RCI、AdaPlanner（靠递归自我纠错），Synapse 用更优提示结构从源头减少误差、无需自纠错；（b）相对 MindAct（MCQ 提示），TaE 用完整轨迹范例，Mind2Web 上相对提升 56%；（c）与 Voyager（A 簇/技能库）互补：Voyager 归纳可组合技能、Synapse 复用原始轨迹，论文称二者正交；（d）作为后续「经验回放」工作的对照基线被反复引用——AWM/Agent Workflow Memory 复用「成功工作流」、ReasoningBank（本研究 A6）复用从成功+失败蒸馏的「抽象推理策略」，二者都批评 Synapse「只存原始成功轨迹、无法利用失败」（ReasoningBank 消融中 Synapse 加入失败仅 40.6→41.7）；（e）与 Reflexion（A 簇，自反思）理念不同：Synapse 不做语言反思而是直接轨迹复用。可与 CoT、ReAct、Reflexion、工具调用、代码即策略等正交叠加。
- **可复现性**: 可复现性较好：官方开源（github.com/ltzheng/Synapse，约 69 stars、Python 3.10，提供 build_memory.py / run_miniwob.py / run_mind2web.py / 微调脚本，并放出全部实验轨迹的 Google Drive 下载链接与项目主页）；基准为公开数据集（MiniWoB++、Mind2Web 含官方元素排序）；嵌入与检索用公开的 text-embedding-ada-002 + Faiss。主要复现障碍：依赖闭源商用 LLM 的特定版本（gpt-3.5-turbo-0301、gpt-3.5-turbo-16k-0613），结果对模型版本敏感且这些快照已逐步弃用；许可为 NOASSERTION（未明确开源许可）。社区采用度高（ICLR 2024，约 147 次引用，被列入多个 Agent Memory 综述与 awesome 清单）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式流程）。记忆的存（离线预构建）、取（欧氏距离相似度检索 + MiniWoB++ 上的多数投票选类）均为固定的非学习型流程，不用 RL/训练去学习「何时/存什么/取什么」的记忆管理策略。处于 2025-26「学习型记忆控制」分水岭中的启发式、早期一侧（与 Memory-R1、Mem-α 等学习型方法相对）。
- **记忆主体**: 智能体中心（agent-centric）：记忆的是智能体自身完成任务的成功交互经验（轨迹），用于跨任务复用与泛化、提升任务完成能力，而非记住用户个人信息做个性化。与 Voyager、ReasoningBank 同类，区别于 Mem0/Zep/LongMemEval 等用户中心记忆。
- **多智能体记忆**: 单智能体（single-agent）。范例记忆服务于单个计算机控制智能体，不涉及多智能体间共享/路由记忆。论文提及记忆可与人类干预结合做用户定制，但无多智能体记忆设计。
- **时序推理支持**: 否。不显式建模时间有效性、事件先后或事实时效窗口。范例记忆是与时间无关的成功轨迹集合，无时间衰减/刷新。任务内的「时间抽象动作」（TaE 一次生成多步动作）属于动作粒度的时序处理，而非对记忆事实做时间推理。
- **模态**: 纯文本（text-only）。状态为网页 HTML（经状态抽象转为文本观察），动作为键鼠操作的代码/文本。论文明确将多模态、视觉与视频理解（如像素级 Android 控制）列为未来方向，本身未实现。
- **过度个性化/记忆安全风险**: 未涉及。论文不做个性化，也未讨论有害/过时/侵入性记忆、隐私治理或过度个性化风险（这类属用户中心记忆的安全维度，超出本工作范围）。
- **冲突/矛盾处理**: 未处理。记忆只存成功轨迹且为静态预构建库，无入库后的冲突/矛盾检测与合并机制；不同范例间若相互矛盾，仅靠相似度检索取最相关者，未做显式消解（cross-domain 上不相关范例的干扰即未被解决的体现）。
- **token成本/延迟证据**: 定性而非系统量化。状态抽象通过过滤任务无关元素显著降低每个状态的 token 数（如 book-flight 超长网页被压缩至可放入上下文），从而在有限上下文内容纳更多范例；TaE 提示产生「时间抽象动作」（一次返回多步、仅在需要新状态时再查询 LLM），论文指出这带来「更低的成本与延迟」。但论文未给出 token/延迟节省的具体百分比数字（与 Mem0/Zep 等给出 -90% 量化值不同）；同时自陈 LLM 推理高延迟是主要局限。Mind2Web 中把元素 top-k 从 50 降到 3/5 也大幅减少输入长度（代价是召回率从 86% 降到 53%，但 Step SR 反而更高）。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)


<a id="c4-jarvis-1开放世界-minecraft-多任务智能体基于记忆增强的多模态语言模型-mlm"></a>

### C4 JARVIS-1

*JARVIS-1（开放世界 Minecraft 多任务智能体，基于记忆增强的多模态语言模型 MLM）*


**基本信息 / Provenance**

- **年份**: 2023（arXiv 预印本 2023 年 11 月 10 日 v1，2311.05997）
- **作者/机构**: Zihao Wang（王子昭，第一作者）、Shaofei Cai（蔡少斐）、Anji Liu、Yonggang Jin、Jinbing Hou、Bowei Zhang、Haowei Lin、Zhaofeng He、Zilong Zheng、Yaodong Yang（杨耀东）、Xiaojian Ma（马晓健，通讯）、Yitao Liang（梁一韬，通讯）；隶属北京大学（PKU）、北京通用人工智能研究院（BIGAI）、UCLA、北京邮电大学（BUPT）。CraftJarvis 团队。
- **发表venue**: IEEE TPAMI 2024（IEEE Transactions on Pattern Analysis and Machine Intelligence，第 47 卷 1894–1907 页，DOI 10.1109/TPAMI.2024.3511593）；早期版本曾在 NeurIPS 2023 ALOE/FMDM 等 workshop 展示。
- **论文链接**: https://arxiv.org/abs/2311.05997（HTML 全文 https://arxiv.org/html/2311.05997；OpenReview https://openreview.net/forum?id=xzPkZyHlOW）
- **代码链接**: https://github.com/CraftJarvis/JARVIS-1（官方实现，约 396 stars / 29 forks，截至 2026 年 6 月；仓库未声明明确开源许可；项目主页 https://craftjarvis.org/JARVIS-1）
- **引用数**: 约 189 次（Semantic Scholar 实时数据，截至 2026 年 6 月；属高影响力的开放世界具身智能体代表作）。

**记忆分类 / Taxonomy**

- **记忆类型**: 以情景记忆（episodic memory）为核心——存储过去成功规划经验（任务+情境观测+成功执行的计划）作为可检索示例；其计划本身（短期子目标序列）可视作程序性知识，被反复复用相当于隐式的程序性记忆；规划过程中的交互推理（自检/自释）使用上下文工作记忆。不显式维护语义知识库。
- **记忆结构**: 多模态键值记忆（key-value memory）：键为多模态（任务文本 + 创建该条目时的观测/情境），值为成功执行过的计划（子目标序列）。同一任务因情境不同可有多条目（situated plans）。整体为‘随游戏进行不断增长的轨迹库’，是一种以经验轨迹为单元的外部记忆，而非知识图谱或 Zettelkasten 笔记图。
- **存储后端**: 外部多模态向量存储：用 CLIP/MineCLIP 文本编码器对任务键编码、用视觉编码器对状态（视觉观测）编码以支持相似度检索；条目内含文本（任务/计划/符号信息如物品栏、坐标）与视觉嵌入。分布式自我改进时使用‘共享集中式记忆’（shared centralized memory）供多个并行 agent 读写。不写入 MLM 参数。
- **持久化**: 外部持久化（durable external store）：多模态记忆在探索/学习阶段被填充并落盘，推理时被检索注入上下文；MLM（GPT-3.5/GPT-4/微调 LLaMA2）参数保持冻结，记忆作为‘in-context 终身学习’载体，不依赖梯度更新（non-parametric 持久记忆）。

**核心机制 / Mechanisms**

- **写入/编码**: 采用‘成功经验的 verbatim 轨迹’编码：当 JARVIS-1 对某任务成功执行一个计划后，把该计划连同任务描述与规划时的智能体情境（多模态观测：视觉画面 + 符号信息如物品栏/位置）整体作为一条多模态记忆条目写入记忆库。写入触发条件是‘计划被成功执行’（论文：Once the plan is successfully executed, it will be stored in the memory along with the task and the agent situation when it was planned）。经验来源于交互式规划过程中‘产生计划、与环境交互、拥抱错误、存储经验’的闭环。键侧用 CLIP 文本/视觉编码器编码以便后续相似度检索。整个写入无梯度更新——这是其相对 RL/IL 智能体的关键效率优势。
- **检索机制**: 检索增强规划（RAG-style），形式化为 p(y|x)≈Σ_{z∈top-k} p_η(z|x)·p_θ(y|x,z)，其中 x=指令、y=计划、z=检索到的记忆条目、p_η/p_θ 为检索/规划模型。分两阶段：(1) 经推理生成查询（query generation via reasoning）——MLM 以‘反向搜索（backward search）’把主任务分解为所需中间子目标（如 craft enchanting table → obtain book/diamond/obsidian），搜索深度受限以保证效率；记忆中存在的子目标连同当前视觉观测组成最终的多模态查询。(2) 多模态检索——先用 CLIP 文本编码器算查询与各条目任务键的文本相似度，选出高于置信阈值的候选；再用视觉嵌入按 p_η(z|x) ∝ CLIP_v(s_z)^T·CLIP_v(s_x)（状态视觉嵌入内积）对候选排序，每个子目标只取 top 条目，最终取 top-k 计划作参考提示。消融（图 8）证明：先推理再检索 > 纯文本检索；多模态（视觉+符号）检索 > 纯文本嵌入检索。
- **反思/巩固**: JARVIS-1 的‘原始→洞见’转化主要体现在交互式规划中的两类自我推理，而非离线知识抽象：(1) 自检（self-check）——产生初始计划后主动核验、修正潜在 bug（如未备足木头就下挖的隐患），属事前验证以减少代价高昂的事后重规划；(2) 自释（self-explain，借鉴 Reflexion/Shinn 2023）——执行中遇到环境失败反馈时，让 MLM 解释错误、定位原计划 bug（error explanation），再据外部环境反馈与内部回溯产出改进计划，形成闭环再规划。注意：JARVIS-1 不像 ExpeL 那样把经验蒸馏成自然语言规则/insight，也不做跨轨迹摘要抽象；它直接保存成功的具体计划轨迹供检索复用，‘巩固’以‘成功才写入’的筛选方式实现。
- **遗忘/更新**: 无显式遗忘/衰减/去重/冲突消解机制。记忆库随探索单调增长（4 个 epoch 后累计约 425 条成功轨迹）；同一任务的多条情境化条目并存而非合并，靠检索时‘视觉相似度排序 + 每子目标取 top 条目’来挑选最贴合当前情境者，相当于用检索消歧代替更新。无 Ebbinghaus 衰减、无 ADD/UPDATE/DELETE 算子。
- **经验回放 (核心主题)**: 本系统的核心主题。JARVIS-1 把过去成功规划经验当作可复用经验，在面对新任务（尤其长程任务）时按‘任务文本相似度 + 视觉情境相似度’检索 top-k 条历史成功计划，作为 in-context few-shot 示例注入 MLM 规划提示，从而把相关任务上的经验迁移到当前任务（论文图 2 右展示 ObtainDiamondPickaxe 与 ObtainDiamondAxe 因材料近似而相互助益）。论文明确把这一机制类比为开放/封闭世界强化学习中的终身学习与经验复用，但实现为‘无梯度、in-context 的经验复用’（in-context life-long learning），不更新模型权重。随记忆增长（不同学习阶段），中间里程碑物品与最终长程任务成功率持续上升（图 7、图 9），直接证明经验复用带来的自我改进。

**学习维度 / Learning**

- **学习范式**: 以非参数化（non-parametric，in-context/prompt 层）为主：通过外部多模态记忆 + 检索增强规划实现‘学习’，MLM 主体参数冻结（兼容 GPT-3.5/GPT-4 等闭源 API）。另含一处可选的参数化成分——为补足开源模型的 Minecraft 知识，作者对 LLaMA2-13B 在网络收集的 Minecraft 文本上做了微调（使其逼近 ChatGPT 水平），但该微调是补领域知识、非记忆机制本身。故整体可视为‘以非参数记忆为主、可选离线领域微调’的混合。
- **失败学习 (核心主题)**: 失败被主动利用但以‘任务内闭环修复’为主、而非长期失败记忆。机制有二：(1) 自释（self-explain，源自 Reflexion）——执行失败时让 MLM 解释错误原因、定位计划 bug 并再规划，使智能体能从环境反馈中恢复（如挖掘工具损坏后据当前物品栏动态重做并重造工具）；(2) 探索阶段‘拥抱错误（embraces the errors）’，在 trial-and-error 中产生经验。但与 ExpeL/ReasoningBank 不同，JARVIS-1 只把‘成功执行的计划’写入长期记忆，失败轨迹本身不作为负例长期存储或抽象成失败模式规则；失败主要服务于即时修复与探索数据生成。
- **技能/程序归纳**: 部分支持：被检索复用的‘成功计划（子目标序列）’实质上是可复用的、情境化的程序/工作流，随终身学习自动积累（约 425 条），并通过子目标级检索按需调用——形成沿 Minecraft 科技树（Wood→Stone→Iron→Diamond）渐进生长的技能集合。但它不像 Voyager 那样把技能固化为可命名调用的可执行代码函数库，而是以自然语言/结构化计划形式存储于记忆条目中。
- **在线 vs 离线**: 两者兼有但以‘在线/探索期填充 + 评测期复用’为默认。学习阶段：用 self-instruct 生成动态课程（curriculum），让 JARVIS-1 在环境中自主提任务、探索并把成功经验存入记忆；为加速采用‘分布式 + 共享记忆 + 投机执行（speculative execution）’让多 agent 并行收集，每轮直至记忆达到一定容量。评测阶段：用积累的记忆做 in-context 规划。论文亦展示该过程可贯穿整个游戏过程持续进行（life-long，边玩边学）。

**评测 / Evaluation**

- **任务领域**: 开放世界具身决策（embodied open-world），具体为 Minecraft 生存模式下的长程多任务规划与控制：采集、合成、冶炼、装备、战斗等。观测为第一人称视觉画面 + 人类同款鼠标键盘动作空间（20 fps），强调与人类相同的观测/动作接口，无特权信息。
- **基准**: Minecraft Universe Benchmark（Lin et al. 2023a）中 200+ 任务，按 Minecraft 推荐分类归为 11 组（Wood、Wood-Variants、Stone、Iron、Gold、Diamond、Redstone、Blocks、Armor、Decoration、Food）；标志性长程任务为 ObtainDiamondPickaxe。每任务至少 30 个不同随机种子评测；不提供示范。
- **报告增益**: 主结果（表 2，组平均成功率，对比 GPT/ReAct/Inner Monologue/DEPS）：Wood 88.84% vs DEPS 80.23；Stone 88.69% vs 69.27；Iron 34.63% vs 16.92；Diamond 组平均 8.99% vs DEPS 2.42%（约提升近 3 倍）；Redstone 17.51% vs 6.02；Food 46.75% vs 22.85——全组均居首。长程旗舰任务 ObtainDiamondPickaxe：20 分钟内成功率 6.22%，对比 SOTA 的 RL 微调 VPT 约 2.5%（提升约 2 倍以上，论文摘要称‘可靠性较 SOTA 提升约 5 倍’，相对早期记录如 DEPS≈0.59%）；延长到 60 分钟（72000 步）成功率升至 12.5%，而 VPT 从 2.5% 仅升到 3%。效率：JARVIS-1 通常只需 2–3 轮再规划即得可执行计划，而 DEPS 需 6+ 轮，显著节省 LLM token 与思考时间。自我改进：4 个 epoch 学习后累计约 425 条成功轨迹，随记忆增长各中间物品与最终任务成功率单调上升（图 7、9）。LM 消融：装备记忆后 ChatGPT 与 GPT-4 成功率近乎持平；微调后的 LLaMA2-13B 接近 ChatGPT 水平。
- **对比基线**: 无记忆/无跨任务学习的 LLM 规划基线：Instruct GPT、ReAct、Inner Monologue、DEPS（均按 Minecraft 重新实现，经 OpenAI API）；长程任务另对比 VPT（RL 微调，1.4M episodes）、STEVE-1、人类玩家 10 分钟成绩；记忆消融对比 JARVIS-1 w/o memory，及三种检索法（纯文本 / 文本+推理 / 多模态+推理）。

**分析 / Analysis**

- **关键创新**: 首个面向开放世界 Minecraft、用‘记忆增强多模态语言模型（MLM）+ 多模态检索增强规划’实现无梯度 in-context 终身学习的通用具身智能体：把成功规划经验连同多模态情境存为键值记忆，经‘推理生成多模态查询 + CLIP 文本/视觉相似度检索’复用历史成功计划，并以 self-instruct 自主探索持续扩充记忆从而自我改进；用人类同款视觉观测/动作接口完成 200+ 任务，将长程 ObtainDiamondPickaxe 可靠性较 SOTA 大幅提升。
- **局限**: (1) 主要瓶颈在低层控制器（STEVE-1 类）无法完美执行 LLM 生成的短程文本指令，限制了 Diamond 类任务上限；(2) 记忆库只增不删、无遗忘/去重/冲突消解，长期可能膨胀且检索成本上升；(3) 仅在 Minecraft 单一开放世界验证，泛化到其他领域未证；(4) 失败轨迹不作长期负例记忆，缺乏显式失败模式沉淀；(5) 依赖闭源 GPT-3.5/GPT-4 API（开源 LLaMA2 需额外领域微调才可用）；(6) 长程任务整体成功率仍偏低（钻石镐 60 分钟 12.5%）。
- **与其他工作关系**: 属 C 类‘经验复用与技能/程序记忆’簇。底层规划继承 ReAct 推理、Inner Monologue/DEPS 的交互式（再）规划与环境反馈，自释机制借鉴 Reflexion（A1，Shinn 2023）；多模态感知用 MineCLIP（Fan 2022），低层控制器用 STEVE-1（Lifshitz 2023）/在 VPT（Baker 2022）基础上的目标条件策略。与同期 Minecraft LLM 智能体相比：Voyager（Wang 2023a）把技能固化为可调用代码并自动课程，JARVIS-1 则以多模态情境化轨迹记忆 + 检索复用、且强调视觉观测；GITM（Zhu 2023）用文本知识+记忆但仅文本环境，JARVIS-1 在部分可观测视觉开放世界中运行。与 ExpeL（A5）同为‘智能体中心、无梯度经验记忆’，但 ExpeL 蒸馏跨任务自然语言 insight 且为纯文本，JARVIS-1 直接复用具体成功计划且为多模态具身。是后续具身/多模态记忆与自我改进智能体（如 ReasoningBank、Optimus 系列）的早期代表。
- **可复现性**: 中等偏可复现：官方 GitHub 开源（CraftJarvis/JARVIS-1，约 396 stars），提供基础规划与控制代码；但仓库 README 标注 multimodal descriptor、多模态记忆（拟上传 HuggingFace）与 learning.py（自我改进脚本）为待发布的 TODO，且未声明明确开源许可；完整复现还需 Minecraft/MineRL 环境、STEVE-1 控制器、MineCLIP 权重及 OpenAI GPT-3.5/GPT-4 API，门槛较高。社区采用信号良好（CraftJarvis 系列后续工作活跃）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否。记忆的写入（成功即存）、查询生成（LLM 反向推理分解）与检索（CLIP 文本阈值 + 视觉相似度排序、每子目标取 top）均为启发式流水线，未用 RL/训练去学习‘何时存/取/更新’的记忆管理策略。属 2025–26 代际划分中‘启发式（pre-learned-control）’一侧，是后续可学习记忆控制工作的前身参照。
- **记忆主体**: 智能体中心（agent-centric）：记忆的是智能体自身在 Minecraft 中的生存/规划经验（成功计划 + 多模态情境），目的为自我改进任务完成能力与终身学习，而非记住用户信息做个性化（区别于 Mem0/Zep 等 user-centric 系统）。
- **多智能体记忆**: 存在多智能体共享记忆的工程化用法（但非语义分层路由）：自我改进的分布式学习阶段，多个 JARVIS-1 在不同环境中并行探索，经验汇入‘共享集中式记忆（shared centralized memory）’，并用投机执行加速。推理评测时为单智能体使用该记忆。无 G-Memory/MIRIX 式跨 agent 的洞见/查询/交互分层路由设计。
- **模态**: 多模态（multimodal/embodied）：感知第一人称视觉观测 + 文本指令 + 符号信息（物品栏、坐标），记忆键含视觉嵌入，检索为多模态。是本研究中少数原生具身多模态记忆系统之一。

**不确定字段 / Uncertain**

- 冲突/矛盾处理 (`conflict_contradiction_handling`)
- 过度个性化/记忆安全风险 (`over_personalization_risk`)
- 时序推理支持 (`temporal_reasoning_support`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="c5-automanual由-llm-智能体通过交互式环境学习自动构建指令手册的框架"></a>

### C5 AutoManual

*AutoManual（由 LLM 智能体通过交互式环境学习自动构建指令手册的框架）*


**基本信息 / Provenance**

- **年份**: 2024（arXiv 预印本 2024 年 5 月 25 日，2405.16247；NeurIPS 2024 正式发表）
- **作者/机构**: Minghao Chen（陈明昊，第一作者）、Yihang Li、Yanting Yang、Shiyu Yu、Binbin Lin（林彬彬，通讯作者）、Xiaofei He（何晓飞）；隶属杭州电子科技大学计算机学院、浙江大学 CAD&CG 国家重点实验室与软件学院、Fullong Inc.、宁波港集团。
- **发表venue**: NeurIPS 2024（第 38 届神经信息处理系统大会，Poster；论文集 Advances in Neural Information Processing Systems 37，DOI 10.52202/079017-0019）。
- **论文链接**: https://arxiv.org/abs/2405.16247（OpenReview 正式版 https://openreview.net/forum?id=Pwl9n4zlf5）
- **代码链接**: https://github.com/minghchen/automanual（官方实现，约 52 stars / 5 forks，含 ALFWorld、MiniWoB++、WebArena 三套环境代码与已生成手册；提供 WebArena 测试 trace）。
- **引用数**: 约 38 次（Semantic Scholar 实时数据，NeurIPS 版本计数；arXiv 版本另计约 16 次，截至 2026 年 6 月，属新兴有影响力工作）。

**记忆分类 / Taxonomy**

- **记忆类型**: 以程序性记忆（procedural memory）为核心——把环境知识抽象为可复用的‘规则（rules）’与最终编纂的指令手册（manual），指导未来规划；并辅以情景记忆（episodic memory，技能库 Skill Library 保存成功代码块、反思库 Reflection Library 保存失败反思）。规划本身基于上下文窗口的工作记忆。属 CoALA 中的 procedural + episodic 组合。
- **记忆结构**: 分层文本记忆：(1) 在线规则系统（rule system），最多 12 条规则，每条规则带四个属性（类型/内容/示例/验证日志），可相互依赖、构成层级；(2) 技能库与反思库，按 ALFWorld 6 种任务类型各存一条成功代码或失败反思；(3) 最终由 Formulator 编纂成 Markdown 格式、人类可读的结构化手册（按应用场景分类）。均为非参数化文本结构，无向量库/图谱。
- **存储后端**: 全部为文本/上下文存储：规则系统、技能库、反思库与 Markdown 手册以文件/上下文形式保存；构建阶段使用 OpenAI Assistant API 保存各智能体历史以避免重复输入。无独立向量数据库或图数据库，规则在测试时直接放入 LLM 上下文（max context length 16000）。不写入模型参数。
- **持久化**: 外部持久化（durable external store）：规则、技能/反思库与手册在构建（Building）+ 编纂（Formulating）阶段离线生成并落盘，测试（Testing）阶段被调用注入上下文指导规划；模型参数保持冻结（兼容 GPT-4-turbo / GPT-3.5-turbo 等闭源 API）。不属于纯上下文临时记忆，也不写入权重。

**核心机制 / Mechanisms**

- **写入/编码**: 采用 summarized rule（抽象规则）+ verbatim/summarized 代码经验两路并存的编码。写入由两类智能体协作完成：(1) Planner 在一个 episode 内用自由形式 Python 代码（free-form code）与环境交互，输出结构化的‘Analysis / Related Rules / Overall Plan / Code’四段，并依据二元奖励 r∈{−1,1} 把结果分类为 Direct Success / Indirect Success / Failure，生成 conclusion（成功则整理代码块，失败则反思错因并指出问题代码段）；成功的代码块按任务类型存入技能库，失败 conclusion 存入反思库。(2) Builder 收到完整轨迹 τ 后，通过规则系统的 write_rule/update_rule/stop_generating 函数把环境知识编码为规则，每条规则含‘Rule Type（六类之一）、Rule Content（含适用范围）、Example（来自轨迹的示例/代码与易错备注）、Validation Logs（追踪应用与演化的 episode/rule ID）’四属性，并可在已有规则之上归纳更一般/更深的规则。整个写入无梯度更新，把传统样本低效的梯度上升替换为基于文本的规则管理。
- **检索机制**: 读取分两路：(1) 规则与手册——构建阶段把当前全部规则 Θ 直接注入 Planner 上下文；测试阶段把 Formulator 编纂的 Markdown 手册整体提供给 test-time Planner，无相似度打分（小/中规模环境下全量注入，规则上限 12 条）。(2) 技能/反思库检索——新任务到来时，从技能库按‘最相似任务’检索对应任务类型的代码块作为示例；若该任务类型尚无技能，则返回该类型的反思（reflection）。Consolidator 用 get_trajectory(episode_id) 按需调取历史轨迹辅助合并判断。检索机制为按任务类型匹配，非 recency/importance/relevance 复合打分；作者将大规模场景下改用 RAG 动态检索规则列为未来工作（类比 AutoGuide）。
- **反思/巩固**: 这是 AutoManual 的核心机制，体现在三层 raw→insight 抽象：(1) Planner 层——对 Indirect Success 总结导致错误的误解、对 Failure 仔细反思错因并提出修正，把经验蒸馏为 conclusion；(2) Builder 层——在线把轨迹抽象为六类规则（Special Phenomenon / Special Mechanism / Useful Helper Method / Success Process / Corrected Error / Unsolved Error），并借鉴 Generative Agents 的层级反思，允许在已有规则上归纳更一般或更深的规则；为防止‘从失败轨迹错误地导出成功规则’等幻觉，引入 case-conditioned prompting（案例条件提示）：Builder 先判定主要错误源于‘Imperfect Rules’还是‘Imperfect Agent’，再用对应的环境无关提示引导规则管理。(3) Consolidator 层——当规则数超过 N_max=12 时，合并相关规则、删除冗余规则并保留细节与示例。最终 Formulator 把规则按场景分类、加引言、编纂为 Markdown 手册。整个 reflection/abstraction 在线触发（每个 episode 后），区别于 ExpeL/AutoGuide 的离线抽取。
- **遗忘/更新**: 在线更新与合并：Builder 用 update_rule 改写已有规则属性、用层级依赖记录规则演化；当规则数超过 12 条时由 Consolidator 调用 update_rule + delete_rule 合并/删除冗余规则（合并时强制保留示例与细节）。作者明确放弃 ExpeL/AutoGuide 的‘规则打分删低分’做法（因发现 Builder 倾向给所有规则过高分、分数不可靠），改用专门的 Consolidator 智能体。无 Ebbinghaus 式时间衰减；验证日志记录规则被应用/更新的历史。
- **经验回放 (核心主题)**: 这是其核心主题。过去轨迹以两种形式被复用以改进未来行为：(1) 在线规则优化——AutoManual 将规则学习显式建模为在线强化学习式优化问题 max_Θ E_{s0,g} E_{ρ(·|Θ)} r(τ_ρ)，把 REINFORCE 式梯度上升替换为‘Planner 实践规则↔Builder 基于轨迹更新规则’的交替文本优化，使更高质量的成功过程能在迭代中涌现；(2) 技能/反思库——成功代码块作为可复用技能、失败反思作为告诫，在新任务按任务类型检索复用。作者论证：仅存成功路径作技能（如 Voyager/AdaPlanner/Planner+Lib.）会陷入‘路径依赖（Path Dependence）’问题，无法表达环境背后的规则，故规则系统比纯技能复用更具泛化性。学习曲线显示跨任务类型共享规则（Cross-task）优于单任务类型单独建规则。

**学习维度 / Learning**

- **学习范式**: 纯非参数化（non-parametric，prompt/in-context 层面），但以‘在线强化学习范式’为灵感：完全不更新 LLM 参数，通过 Planner-Builder 交替的文本化规则管理替代梯度上升，在外部规则/技能记忆中实现‘学习’，兼容仅 API 可用的闭源模型（GPT-4-turbo、GPT-3.5-turbo）。
- **失败学习 (核心主题)**: 这是其核心主题之一。失败被显式检测与利用：(1) Planner 依奖励把每个 episode 分为 Direct Success / Indirect Success / Failure；对 Indirect Success 总结造成错误的误解、对 Failure 仔细反思错因、提出修正并指出问题代码段，存入反思库；(2) 规则系统专设‘Corrected Error’与‘Unsolved Error’两类规则类型沉淀失败知识，并在规则 Example 中标注易错点（如‘目标物体可能出现在非常规位置’）；(3) case-conditioned prompting 通过判定错误源于‘Imperfect Rules’还是‘Imperfect Agent’来分流处理，避免从失败轨迹错误归纳成功规则。消融显示降低平均错误步数（Avg. Error Steps）是性能提升的关键指标（完整框架降至 0.3 步）。
- **技能/程序归纳**: 强支持且为核心：(1) 显式归纳可复用‘技能’——把成功代码块（含 Planner 自定义的可复用辅助函数）按任务类型存入技能库，规则系统亦含‘Useful Helper Method’类型；(2) 更高层地把环境知识归纳为六类规则并最终编纂为人类可读的 Markdown 指令手册（procedural knowledge）。规则/技能以文本+代码表示，测试时整体注入上下文供 Planner 调用；手册还能指导更小 LLM 的规划。
- **在线 vs 离线**: 以在线（online）为核心卖点：构建阶段交替进行‘Planner 实践规则’与‘Builder 基于该轨迹即时更新规则’，及时验证规则的可靠性与适用性，避免规则‘纸上谈兵（armchair general）’。论文同时实现离线 AutoManual（先收集全部轨迹再统一管理规则）作对照，消融证明去掉在线管理后手册仅微弱提升（88.1%→90.7%），凸显在线优势。

**评测 / Evaluation**

- **任务领域**: 三类交互式决策环境：具身/家务文本模拟（ALFWorld，家庭机器人）、模拟网页操作（MiniWoB++，键鼠完成网页任务）、真实网页导航（WebArena 的 Reddit 域，长规划视野、大观测/动作空间）。均为文本/HTML 观测的智能体任务（agent-centric 自我提升类，非用户个性化）。
- **基准**: ALFWorld（6 种任务类型测试集，含 Put/Clean/Heat/Cool/Examine/Put two）、MiniWoB++（9 种带反馈任务类型 + 全部 53 种任务类型）、WebArena（Reddit 域，沿用 AutoGuide 设定）。每实验跑 3 次取平均；构建阶段 ALFWorld 默认每类型 6 任务共 36 任务，规则上限 12，replan 上限 3（ALFWorld）/6（MiniWoB++）。
- **报告增益**: 仅用 1 个最简任务的人工示例即取得领先（构建/编纂用 GPT-4-turbo，测试用 GPT-4-turbo 或 GPT-3.5-turbo）。ALFWorld 总成功率：GPT-3.5-turbo 达 86.2%（vs ReAct 41.9% / Reflexion 59.8% / ExpeL 52.2% / AdaPlanner 63.3% / Planner+Lib. 66.5%，均用更多人工示例），GPT-4-turbo 达 97.4%（vs ReAct 76.8% / Reflexion 85.9% / ExpeL 79.2% / AdaPlanner 76.4% / Planner+Lib. 88.1%）。MiniWoB++：9 类带反馈任务 GPT-3.5 82.2% / GPT-4 94.5%（vs RCI 45.6%/60.4%、AdaPlanner 71.6%/74.1%、Planner+Lib. 63.6%/80.2%）；全 53 类 GPT-3.5 92.7% / GPT-4 98.3%。WebArena(Reddit)：GPT-4-turbo 65.1%（vs ReAct 6.0% / AutoGuide 43.7% / SteP 55.0% / 纯 Planner 51.1%）。消融（ALFWorld，GPT-4-turbo）：基线 77.6%（Avg. Error Steps 2.3）→加在线规则 88.1%（1.5）→完整框架（在线+技能反思库+case prompt+formulation）97.4%（错误步数降至 0.3）。
- **对比基线**: 对比无记忆/无跨任务学习与已有自改进/技能复用方法：ReAct、Reflexion、ExpeL、RCI、AdaPlanner、AutoGuide、SteP，以及自身变体 Planner+Lib.（仅技能&反思库、无规则）；并对所有先验方法以相同 GPT-3.5/GPT-4 版本重新实现以公平对比。

**分析 / Analysis**

- **关键创新**: 首个由 LLM 智能体在线、交互式地把环境经验抽象为‘结构化规则系统’并自动编纂为人类可读指令手册的框架：以 Planner-Builder(-Consolidator)-Formulator 多智能体协作、四属性规则系统、案例条件提示（case-conditioned prompting）抑制规则幻觉、在线规则管理替代梯度上升，仅凭单个示例显著超越离线抽规则（ExpeL/AutoGuide）与纯技能复用（Voyager/AdaPlanner），并解决其‘路径依赖’问题。
- **局限**: 作者明确列出：(1) 严重依赖 GPT-4-turbo 生成可靠规则，对较弱模型适用性受限；(2) 当前把全部规则直接放入上下文，难以扩展到更大/更动态的环境，需结合 RAG 动态检索规则；(3) 对复杂困难任务探索不足（仅基于现有知识尝试），需引入好奇心或树搜索；(4) 难以保证 Planner 始终遵循手册规则（可能忽略规则或对观测产生幻觉），需更强的规则遵从/验证机制。其它隐含局限：规则上限仅 12、仅文本/HTML 模态、评测规模有限、依赖闭源 API 模型快照。
- **与其他工作关系**: 属 C 类‘经验回放与技能/程序性记忆’簇。直接对标并区别于同簇离线抽规则方法 A5 ExpeL（Zhao et al. 2023）与 AutoGuide：二者从 Reflexion 离线轨迹抽取规则、每条规则仅含‘内容+分数’，AutoManual 则在线更新、规则含四属性且可层级依赖、用 Consolidator 取代不可靠的分数删除。借鉴 Reflexion（A1，任务内失败反思）但扩展为跨任务在线规则；规划用 ReAct + 自由形式代码（区别于 Voyager/AdaPlanner 的整函数式技能，后者陷入 Path Dependence）。规则的层级归纳与‘按相关性管理’借鉴 Generative Agents（B1）的层级反思，但面向任务导向规则而非开放式模拟；记忆管理视角上与 CLIN（A3）、RAP、MemGPT（B3）相关。属 agent-centric 自我改进路线（与 ReasoningBank/Voyager 同类，区别于 Mem0/Zep 等 user-centric 个性化记忆）。
- **可复现性**: 可复现性较好：官方代码开源（含 ALFWorld/MiniWoB++/WebArena 三环境的 build/formulate/test 三阶段脚本、已生成手册与 WebArena 测试 trace），附录给出超参（规则上限 12、temperature 0、context 16000、replan 次数等）与 ~$14 的构建+编纂 API 成本；但需自行安装 ALFWorld、MiniWoB++、修订版 WebArena 等环境，且依赖 gpt-4-1106-preview / gpt-3.5-turbo-1106 等已演进的闭源模型快照，精确复现受模型版本影响。社区采用信号中等（约 52 stars）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否。AutoManual 采用启发式/提示驱动的记忆管理流水线：规则的增删改靠 Builder/Consolidator 多智能体调用 write_rule/update_rule/delete_rule 函数 + case-conditioned 提示规则，技能检索按任务类型匹配——无 RL/训练去学习‘何时存/取/更新’的记忆管理策略。属 2025–26 代际划分中的‘启发式（pre-learned-control）’一侧，是 Memory-R1/Mem-α 等可学习记忆控制工作的前身参照。
- **记忆主体**: 智能体中心（agent-centric）：记忆的是智能体自身在环境交互中的经验（规则、成功代码技能、失败反思），目的是自我改进任务规划与对新环境的适应，而非记住用户信息做个性化（区别于 Mem0/Zep 等 user-centric 系统）。
- **多智能体记忆**: 单智能体记忆，但采用多智能体角色协作管理同一份记忆：Planner（规划/反思）、Builder（管规则）、Consolidator（合并删冗）、Formulator（编纂手册）分工协作维护共享的规则系统。这并非 G-Memory/MIRIX 式的跨智能体共享/路由记忆，而是单一智能体记忆的多角色编排；记忆最终以共享手册形式可被不同规模 LLM 复用。
- **模态**: 纯文本（text-only）：观测为 ALFWorld 文本、MiniWoB++/WebArena 的 HTML/文本，规则与手册均为文本/代码。无图像/视觉/具身多模态记忆。
- **冲突/矛盾处理**: 通过在线规则管理隐式处理冲突：Builder 用 update_rule 改写过时/相悖规则、记录规则间依赖关系，Consolidator 合并相关或重叠规则并删除冗余（强制保留细节与示例）；case-conditioned prompting 通过区分‘Imperfect Rules vs Imperfect Agent’避免引入相互矛盾的规则。但无显式的事实级冲突检测/版本化机制（不及 Memory-R1 的 UPDATE 或 MEMTRACK 精细），冲突解决依赖 LLM 判断与 Validation Logs。

**不确定字段 / Uncertain**

- 过度个性化/记忆安全风险 (`over_personalization_risk`)
- 时序推理支持 (`temporal_reasoning_support`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="c6-expgraphmodel-agnostic-experience-learning-with-graph-structured-memory"></a>

### C6 ExpGraph

*ExpGraph（Model-Agnostic Experience Learning with Graph-Structured Memory）*


**基本信息 / Provenance**

- **年份**: 2026（arXiv 预印本，2026-05-29 首次公开）
- **作者/机构**: Tao Feng（冯涛，第一作者）、Chongrui Ye、Tianyang Luo、Jingjun Xu、Xueqiang Xu、Haozhen Zhang、Zhigang Hua、Yan Xie、Shuang Yang、Ge Liu、Jiaxuan You（游嘉轩，通讯/资深作者）等 11 人；主要单位为伊利诺伊大学厄巴纳-香槟分校（UIUC），合作单位包括南洋理工大学（NTU）与 Meta Monetization AI。
- **发表venue**: arXiv 预印本（cs.CL，截至公开时未经同行评审）
- **论文链接**: https://arxiv.org/abs/2605.30712
- **引用数**: 约 0 次（Semantic Scholar 实时查询，论文 2026-05-29 刚公开，尚无引用）

**记忆分类 / Taxonomy**

- **记忆类型**: 以程序性记忆（procedural）为主——把历史轨迹蒸馏为可复用的“技能（skills，成功推理/规划模式）”与“失败教训（lessons，失败模式/需规避的约束）”；同时带有情景记忆（episodic）属性，因为每个经验节点源自一次完整轨迹 τ=(x,ξ,y,s)，并辅以语义层面的可复用知识。
- **记忆结构**: 自演化的经验图（self-evolving experience graph）：稀疏无向图 G=(V,E)，每个节点 v=(e_v, h_v, u_v, n_v) 存储经验文本、嵌入、估计效用与检索计数；新节点按 top-K 最近邻且余弦相似度 ≥ θ 与已有节点连边，形成关系型记忆（区别于扁平最近邻匹配）。
- **持久化**: 外部持久化（durable external store）：经验图独立于执行器，跨任务/跨回合持续累积并在线更新；执行器 LLM（executor）全程冻结、不写入任何参数，因此记忆完全是非参数化外部记忆，可随执行器更换而迁移复用。

**核心机制 / Mechanisms**

- **写入/编码**: 对每条历史轨迹 τ=(x,ξ,y,s)（任务输入、中间执行过程、最终响应/动作序列、任务分数）调用摘要器 e=Summarize(τ)，蒸馏为紧凑的自然语言经验单元，而非保留完整轨迹。高分轨迹蒸馏为“技能”（成功推理模式、规划策略、任务启发式），低分轨迹蒸馏为“失败教训”（失败模式、无效动作、需规避约束）。每个经验单元成为图节点 v=(e_v, h_v, u_v, n_v)，并按公式 (3)（top-K 最近邻且 cos(h_vi,h_vj)≥θ）连边并入图中，保持图稀疏。
- **检索机制**: 三步“效用引导的图检索”：(a) 语义播种 Semantic Seeding——按任务嵌入 h_x 取 top-m 余弦相似节点构成种子集 S_0；(b) 图扩散 Graph Diffusion——以 S_0 为重启分布做个性化 PageRank：p_{t+1}=α(ρ)·q+(1-α(ρ))·A_norm^T·p_t，重启概率 α(ρ) 随 ρ 单调增（ρ 大→扩散收窄于种子邻域，ρ 小→扩散更广），收敛后取 top-L 为候选集 C；(c) 效用感知排序 Utility-Aware Ranking——对候选用 UCB 式效用分 b_v=u_v+c·sqrt(log(N+1)/max(n_v,1))，最终得分 score(v|x)=(1-λ)·sim̂(x,v)+λ·b̂_v，取 top-K 作为检索结果 E。其中 ρ 与 λ 由轻量级“检索副驾（retrieval copilot）”π_ret 针对每个任务自适应预测（输出离散控制变量 R、W∈{0..100}，再归一化为 ρ=R/100、λ=W/100）。
- **反思/巩固**: 通过摘要器 Summarize 把原始轨迹抽象为更高层的可复用知识（技能/失败教训），这是核心的“原始→洞见”转化。触发时机为离线构图与在线演化两处：在线时，每完成一次带检索的任务，新轨迹 τ′=(x,ξ′,y_with,s_with) 被摘要为新候选经验节点，经近重复过滤后按公式 (3) 插入图中。无需对执行器做参数级反思训练，反思发生在外部记忆层。
- **遗忘/更新**: 在线效用更新：对被检索节点更新计数 n_v←n_v+1 与效用 u_v←(1-β)u_v+β·r（β∈(0,1] 为更新率，r 为效用奖励的滑动平均）；插入新节点时做近重复过滤；当图超过容量预算时，淘汰“低效用且低检索频率”的节点（capacity-budgeted eviction/pruning）。无 Ebbinghaus 时间衰减，遗忘由效用与使用频率驱动。
- **经验回放 (核心主题)**: 这是论文的核心主题：把过去的成功轨迹（技能）与失败轨迹（教训）作为外部经验复用来改进未来行为，而不更新执行器参数。复用方式是“检索增强”——选出的经验文本拼接进执行器输入上下文：y_with=π_exec(x, E_{R,W}(x))。关键在于复用的不是“最相似”而是“历史上对该冻结执行器最有用”的经验：检索副驾用 RL 学习检索策略，奖励直接来自下游任务表现差（带/不带经验），从而偏向真正提升执行器表现的经验复用。论文用对比表（Table 1）说明 ExpGraph 是唯一同时支持图结构经验、图扩散、效用感知排序、自适应检索四要素的复用框架。

**学习维度 / Learning**

- **学习范式**: 混合（hybrid），但以非参数化为主：执行器 LLM 全程冻结、模型无关（model-agnostic），仅通过输入上下文改进，属非参数/提示层经验复用；唯一被训练的参数化组件是轻量级检索副驾 π_ret，用 PPO 强化学习优化检索控制策略（ρ、λ），并不更新执行器。经验图本身在线统计更新（计数/效用）属非梯度更新。
- **失败学习 (核心主题)**: 这是论文的核心主题之一：低分轨迹被 Summarize 蒸馏为“失败教训（lessons）”——失败模式、无效动作、需规避的约束——作为负向经验节点存入图中，与成功“技能”节点共存。检索时这些教训可与技能一并被图扩散召回并注入上下文，引导执行器规避先前错误、减少不必要探索（agentic 环境中平均交互步数下降 12.7%/21.6% 即与此相关）。失败检测依赖任务分数 s（环境/评估器返回的低分判定为失败轨迹）。
- **技能/程序归纳**: 是。从高分轨迹归纳出可复用“技能”（成功推理模式、规划策略、任务启发式），表示为自然语言经验节点；通过语义播种+图扩散+效用排序被检索并以文本形式注入执行器上下文调用。迁移实验显示这些技能编码了可跨执行器泛化的“可复用程序性知识”（小→大、大→小、非推理→推理模型均可迁移）。
- **在线 vs 离线**: 二者兼具。离线：先用历史轨迹批量构建初始经验图；在线：部署期每完成一次任务即用下游反馈更新副驾（PPO）与经验图（效用/计数更新、新节点插入、低质节点淘汰），形成闭环协同演化（co-evolution）。

**评测 / Evaluation**

- **任务领域**: 覆盖问答（QA）、数学推理、代码生成（统称 ExpSuite-Static 静态单轮任务），以及多步交互式具身/智能体环境 ExpSuite-Agentic（ALFWorld 家居具身任务、AppWorld 应用操作任务）。
- **基准**: 作者自建评测套件 ExpSuite。ExpSuite-Static（10 个基准）：QA—ARC-C、CommonsenseQA、GPQA、MMLU、OBQA；数学—GSM8K、GSM-Symbolic、MATH；代码—HumanEval+、MBPP+。ExpSuite-Agentic：ALFWorld（Seen/Unseen，报告成功率 SR 与步数）、AppWorld（Test-Normal/Test-Challenge，报告通过率 PR 与步数）。执行器：静态任务用 Llama-3.2-3B-Instruct（小）与 Llama-3.1-8B-Instruct（大）；智能体任务用 Qwen3-32B（小）与 Gemini-3.1-Flash-Lite（大）。
- **报告增益**: 相对“最强基线”：ExpSuite-Static 平均分小执行器 +12.2%、大执行器 +4.7%（如 Llama-3.2-3B 平均 51.91→69.57；Llama-3.1-8B 平均 60.25→78.75，均为各表最高且大幅超过 ReasoningBank/Mem0/MemRL/S3 等）。ExpSuite-Agentic 加权平均分小执行器 +21.4%、大执行器 +12.7%（Qwen3-32B 平均 0.534、Gemini-3.1-Flash-Lite 平均 0.623，均为最高），同时相对“最高效基线”平均交互步数分别下降 12.7% 与 21.6%（如 Qwen3-32B 智能体平均步数降至 14.4，Gemini 降至 15.2）。趋势：较弱执行器与智能体任务获益更大。
- **对比基线**: 四组：(i) No-Memory（冻结执行器无外部经验）；(ii) 检索中心的经验学习基线——ReasoningBank、ExpeL、LightMem、Mem0、AWM、MemRL；(iii) LLM 中心基线——IRCoT、Search-o1、S3；(iv) 提示式智能体基线（仅 Agentic）——ReAct、Reflexion。为公平比较，所有基于经验的基线获得与 ExpGraph 相同数量的经验。

**分析 / Analysis**

- **关键创新**: 首个同时统一“图结构经验 + 图扩散 + 效用感知排序 + 自适应检索”四要素的、模型无关（执行器冻结可替换）的经验学习框架：用可训练检索副驾（PPO）以“带/不带经验的执行器表现差”为效用奖励来学习检索策略，使复用偏向真正有用而非仅语义相似的经验，并让经验图与副驾在线协同演化。
- **局限**: 作者自述：(1) 评测虽多样但仍局限于代表性静态推理基准与 ALFWorld/AppWorld，未覆盖更长程真实应用（网页浏览、科学发现、多智能体协作）；(2) 经验图基于嵌入语义相似度且使用固定超参（邻居数、相似度阈值 θ、图容量、效用更新率 β），更自适应的构图/剪枝策略可能更鲁棒；(3) 检索副驾仅用标量效用奖励，缺乏过程级（中间里程碑）信用分配。此外评测基于作者自建的 ExpSuite、且为未经同行评审的预印本，独立复现与可比性有待验证。
- **与其他工作关系**: 属于 C 类（经验回放与技能/程序性记忆）智能体自我改进路线。直接对比并超越同簇方法：ExpeL（文本经验蒸馏，无图/无效用/无自适应）、Mem0（有图结构但无图扩散/效用/自适应）、MemRL（有效用感知但无图/无自适应）、S3（有自适应检索但无图/无效用）、AWM（agent workflow memory）、ReasoningBank、LightMem。机制上与 ReasoningBank/Voyager 同属“agent-centric 记忆自我改进”范畴；个性化 PageRank 图扩散检索思路与同期 GraSP、Graph-of-Skills（GoS）、SkillGraph、EXG 等技能图工作相呼应；其“效用引导检索 + 冻结执行器 + RL 学习检索策略”延续了 Memory-R1/Mem-α 这一“学习记忆控制策略”的 2025–26 代际方向，但创新在于不训练执行器、检索副驾用执行器表现差作为奖励。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 是——属 2025–26“学习记忆控制策略”代际：用 PPO 训练轻量级检索副驾 π_ret 来学习“检索策略本身”（自适应预测图扩散广度 ρ 与相似度-效用权衡 λ），而非纯启发式管线；但学习的是检索/读取策略，写入与遗忘仍由在线效用统计与启发式（相似度构图、容量淘汰）驱动。
- **记忆主体**: 智能体中心（agent-centric）：记忆的是智能体自身的历史经验（成功技能与失败教训）以自我改进任务表现，而非记住用户信息做个性化；与 ReasoningBank/Voyager 一脉，区别于 Mem0/Zep/LongMemEval 的用户中心记忆。
- **多智能体记忆**: 单智能体设定。当前仅在单执行器上构建与复用经验图；作者将“协作式多智能体工作流”列为未来工作，未实现跨智能体共享/路由记忆。
- **时序推理支持**: 无显式时间建模。不维护事实有效期窗口或事件时序日历（区别于 Zep/Graphiti）；节点带“检索计数 n_v”与在线“效用 u_v”等使用统计，属使用频次/新鲜度意义上的隐式时序信号，而非显式时间有效性推理。
- **模态**: 纯文本（text-only）。经验单元为自然语言文本+嵌入；ALFWorld/AppWorld 通过文本化的交互轨迹处理，无视觉/多模态记忆。
- **冲突/矛盾处理**: 无专门的冲突/矛盾事实消解机制（非事实型用户记忆）。处理方式偏向“效用竞争”：相互矛盾或低效经验通过在线效用 u_v 衰减、检索计数与容量预算淘汰被逐步抑制；插入时做近重复过滤避免冗余，但不显式做 UPDATE/合并矛盾事实（区别于 Memory-R1 的 UPDATE、MEMTRACK）。

**不确定字段 / Uncertain**

- 代码链接 (`code_url`)
- 过度个性化/记忆安全风险 (`over_personalization_risk`)
- 可复现性 (`reproducibility`)
- 存储后端 (`storage_backend`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="c7-webcoachself-evolving-web-agents-with-cross-session-memory-guidance模型无关的网页智能体跨会话记忆教练框架"></a>

### C7 WebCoach

*WebCoach（Self-Evolving Web Agents with Cross-Session Memory Guidance；模型无关的网页智能体跨会话记忆教练框架）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本，2025-11-17 首次公开 v1）
- **作者/机构**: Genglin Liu（刘庚林，第一作者，加州大学洛杉矶分校 UCLA，在 Amazon 实习期间完成本工作）、Shijie Geng、Sha Li、Hejie Cui、Sarah Zhang、Xin Liu、Tianyi Liu（后六位均隶属 Amazon）；共 7 位作者，主要单位为 Amazon（合作单位 UCLA）。
- **发表venue**: arXiv 预印本（cs.AI / cs.CL；论文标注“18 pages; work in progress”，截至公开时未经同行评审）
- **论文链接**: https://arxiv.org/abs/2511.12997
- **代码链接**: https://github.com/genglinliu/WebCoach（官方代码仓库，创建于 2025-09-22，截至查询约 4 星标，影响力尚低）
- **引用数**: 约 6 次（Semantic Scholar 实时查询，2025-11-17 刚公开，引用尚少，处于早期阶段）

**记忆分类 / Taxonomy**

- **记忆类型**: 以情景记忆（episodic）为主——外部记忆库（EMS）将每次完整的网页浏览轨迹（observation/action/reward 序列）作为一段“情景经验”存储；同时具备程序性（procedural）属性，因为成功轨迹会被蒸馏为可复用的“成功工作流（success_workflows，关键步骤序列）”、失败轨迹蒸馏为“失败模式（fail_modes）”，可被检索并以建议形式注入引导后续决策。属 CoALA 的 episodic + procedural 类别。
- **记忆结构**: 外部向量化记忆库 EMS：每条记录为 ⟨embedding, summary_text, meta⟩ 三元组；底层是带 HNSW 索引的扁平向量存储（语义嵌入最近邻检索），并非知识图谱或 Zettelkasten 笔记图；记忆为“完整轨迹级”的情景条目，按相似度+新近度组织，可跨会话累积扩张（self-evolving）。
- **存储后端**: 外部持久化向量数据库：以 FAISS + HNSW-128 索引实现 top-K 近似最近邻相似检索（论文称可扩展至数百万条情景）；嵌入向量为 1536 维 OpenAI embedding（可替换为其他高质量嵌入模型，只需保持库内一致性）；记录另含 episode_id、域名/URL 根、用户目标、模型名、总步数、时间戳等元数据。轨迹原始日志以 JSON 文件形式由 actor 在每步写出。
- **持久化**: 外部持久化（durable external store）、跨会话长期记忆：EMS 独立于 actor 智能体并超越其原生上下文窗口，跨任务/跨会话持续累积；actor 基础策略全程冻结、不做梯度更新，因此记忆完全是非参数化外部记忆，可跨 actor、跨任务、跨域共享与迁移（cross-actor / cross-task 知识库）。

**核心机制 / Mechanisms**

- **写入/编码**: 由 WebCondenser（≤8B 小型 LLM，论文实验用 Qwen3-8B）把原始底层环境轨迹 T_{1:t}={(o_i,a_i,r_i)} 标准化压缩为固定 schema：summary_text（3-5 句概述当前结果）、embedding（1536 维嵌入）、final_success（true/false/null）、以及 fail_modes 或 success_workflows（带关键步骤的错误分析/成功工作流证据）。WebCondenser 仅做模式归一化的轻量过滤，不做推理或干预。关键路由规则：部分轨迹（任务进行中）只流式送给 Coach 做实时决策、不写入记忆；只有到达自然终止点的完整轨迹才被持久化到 EMS——以防半成品噪声污染记忆。reward 不一定是数值分数，可为对任务完成状态的自评。
- **检索机制**: 基于相似度+新近度的 top-K 检索（K=5）。EMS 用 FAISS HNSW-128 多层可导航小世界图做对数时间近似最近邻搜索；候选记忆按与当前上下文嵌入的相似度排序，相似度为归一化点积（余弦）score(e_t,e_i)=e_t·e_i/(‖e_t‖₂‖e_i‖₂)。给定 Condenser 对当前部分轨迹的嵌入 e_t，返回 top-K 历史经验。论文实测：600 条轨迹时余弦检索约 10ms/查询且与 K 几乎无关；K=5 是因为 5 条最近经验既能让 Coach 发现模式又不淹没上下文。Coach 同时利用经验的成功/失败结果标签做证据接地。评测时显式排除与当前任务 ID 相同的情景以防泄漏（leakage control）。
- **反思/巩固**: 存在两层“原始→洞见”转化：(1) WebCondenser 把低层轨迹抽象为高层自然语言摘要、并把成功轨迹蒸馏为 success_workflows、失败轨迹蒸馏为 fail_modes（错误/挑战分析）；(2) Coach（8B LLM）在运行时对“当前部分轨迹摘要 + top-K 检索经验”进行反思推理，决定是否干预并生成具体建议（如“别点 Next，之前的智能体在此陷入循环”）。论文称这是“基于检索的反思（retrieval-based reflection）”：动态模式下 Coach 常引用智能体自己早先的轨迹来预警陷阱。整个反思在外部记忆/教练层完成，不对 actor 做参数级反思训练。触发时机：Condenser 每步触发摘要；Coach 每步被调用判断是否干预。
- **遗忘/更新**: 无显式遗忘/衰减机制（无 Ebbinghaus 时间衰减、无 ADD/UPDATE/DELETE 三元操作）。记忆主要是“只增”累积：完整轨迹持续写入 EMS（self-evolving 扩张）。质量控制靠写入端的路由规则（仅持久化完整且终止的轨迹，过滤半成品噪声）与检索端的相似度/新近度排序及同任务 ID 排除，而非事后编辑或删除既有条目。
- **经验回放 (核心主题)**: 这是论文的核心主题：把过去完整浏览轨迹（成功与失败的情景经验）复用以改进未来行为，且不重训 actor 基础策略。复用链路为“检索→教练建议注入”：Coach 检索 top-5 相关历史经验，结合其成功/失败标签，在运行时（mid-episode）通过 runtime hooks 把简短的任务专属建议作为 system message 同步追加进 actor 的下一步提示（intervene=true 时），actor 策略网络不变、无梯度回传。复用的不是原始动作序列回放，而是从经验中提炼的“证据接地的建议/工作流”。论文强调“动态自我经验记忆（agent 迭代扩张自己的记忆库）”优于外部种子记忆——智能体从自己的轨迹学习最有效，因为检索到的嵌入更贴近其自身表征空间与归纳偏置，推理连续性更顺。

**学习维度 / Learning**

- **学习范式**: 非参数化（in-context / 提示层经验复用）为主：actor 基础策略全程冻结、模型无关，仅通过注入 system message 改进；WebCondenser 与 Coach 在论文实验中均为零样本提示的现成 LLM（Qwen3-8B），并未做梯度训练（作者尝试过对 Qwen3 做 DPO 微调但因 GPT-4o 教练并不稳定占优而放弃）。EMS 为非梯度的在线累积。整体属典型的非参数、检索增强的经验学习；Coach 设计上“可被训练或提示”，留有参数化空间但本文未启用。
- **失败学习 (核心主题)**: 这是论文的核心主题之一：失败检测依赖 WebCondenser 对完整轨迹的 final_success 标签判定与 fail_modes 蒸馏（总结导致失败的错误/挑战，如循环、CAPTCHA、死胡同、登录门、HTTP 4xx）。这些失败经验与成功工作流一同存入 EMS，并带结果标签。Coach 的干预决策规则即显式以“检测到高失败概率”为触发条件之一（编码循环/CAPTCHA/4xx），据此发出规避建议（负向示例引导，例如“别点 Next，之前在此陷入循环”）。行为分析显示被教练的智能体学会跳过先前导致死锁的冗余登录/以旧换新提示，减少重复页面访问。
- **技能/程序归纳**: 部分支持：从成功轨迹中归纳 success_workflows（带名称与描述的关键步骤序列，如“导航到产品页”“识别颜色选项”），从失败轨迹归纳 fail_modes。这些工作流以自然语言形式存于 EMS，并经 Coach 检索后以建议文本被复用调用。但它不像 Voyager/AWM 那样把技能固化为可调用的代码/可执行 API 库，而是“经验性工作流证据”，由 Coach 转译为运行时建议注入。
- **在线 vs 离线**: 二者兼具，且强调在线自演化。冷启动（offline）：EMS 可用先前训练好的网页智能体的高质量轨迹做种子（Frozen EMS）以保证首个在线回合即有可用经验。在线（online）：动态 EMS 模式下，每个 actor 在部署时把自己新产生的完整轨迹迭代写回记忆库、持续扩张并自我改进（self-evolution），无需重训。论文结论：动态自我经验（online）优于外部冻结种子记忆（offline 借来的经验）。

**评测 / Evaluation**

- **任务领域**: 网页导航（web navigation / 真实浏览器在线浏览任务），覆盖电商、新闻、学术、地图、航班、词典、视频/体育、代码托管等 15 个真实网站子域（Amazon、Apple、ArXiv、BBC News、Booking、Cambridge Dictionary、Coursera、ESPN、GitHub、Google Flights、Google Map、Google Search、Huggingface、Wolfram Alpha、Allrecipes）。属多模态（视觉-语言）GUI/Web 智能体场景，非具身/QA/对话。
- **基准**: WebVoyager（he2024webvoyager）——真实在线浏览基准，含 643 个跨 15 个网站子域的实时浏览任务（每子域 30-50 任务），在真实 Chromium 浏览器（Docker 容器、A100 机器）中在线评测，区别于 Mind2Web/WebArena/VisualWebArena 等缓存快照或沙盒环境。基础 actor 为 browser-use 智能体；约束为每动作 30s 超时、每任务最多 50 步。
- **报告增益**: 在 WebVoyager 643 任务上，WebCoach 在三种开源 VLM 骨干上均一致提升成功率且步数持平或下降。最大增益为 Skywork-r1v3-38B：成功率 47.3%→61.4%（动态自我 EMS + Qwen3-8B 教练），+14.1（论文称约 +14.4）个百分点，接近 GPT-4o 上限基线 65.3%；平均步数 10.7→10.2、未增加步数（平均时间因 Coach/Condenser 推理开销升至约 395s，相对基线动态模式约增 +150s，但减少 1-2 步冗余动作）。Qwen2.5-VL-32B：49.5%→57.1%（+7.6 点，动态 EMS），达到与 GPT-4o 相当的开源水平。趋势：存在“认知阈值”——7B 小模型不获益甚至略降（32.8%→31.1%），32B/38B 大模型显著获益，记忆引导在“部分胜任边界”最有价值。子域上 Apple/ArXiv/BBC News 等语义复杂站点增益最大，Booking/Google Flights 等增益小或为负。
- **对比基线**: 四组配置对比：(1) Baseline（无教练、actor 单跑）；(2) Frozen EMS + GPT-4o 教练（GPT-4o 轨迹种子记忆、GPT-4o 当教练）；(3) Frozen EMS + Qwen3-8B 教练（GPT-4o 种子记忆、Qwen3-8B 当教练）；(4) Dynamic EMS + Qwen3-8B 教练（actor 在线迭代扩张自身记忆）。另设 GPT-4o（无记忆）作为上限/天花板参照基线（65.3%）。即对比 no-memory、外部冻结借来记忆 vs 自我动态记忆、不同教练 LLM 与不同 actor 骨干。

**分析 / Analysis**

- **关键创新**: 提出一个模型无关、框架无关、即插即用的“记忆教练（Coach）”层，通过简单的轨迹 runtime hooks 包裹任意现有网页智能体（如 browser-use），用三模块解耦设计——WebCondenser（轨迹标准化压缩）+ External Memory Store（跨会话情景记忆向量库）+ Coach（运行时选择性干预的 LLM 教练）——使网页智能体获得超越原生上下文窗口的持久跨会话记忆，并通过持续把自身新轨迹写回记忆实现“自演化、不重训”的持续改进；首次系统证明在真实在线 WebVoyager 上，自生成（动态）经验记忆优于外部借来的种子记忆，并能让开源骨干逼近 GPT-4o。
- **局限**: 作者自述未来工作隐含的局限：(1) 依赖多 LLM 协作（Condenser+Coach+actor）带来额外推理开销与平均完成时间上升（动态模式约 +150s/任务），尚未把外部记忆直接整合进 actor 内部策略以消除多 LLM 依赖；(2) 仅在 WebVoyager 单一基准评测，未跨更多网页/具身/多智能体环境验证；(3) 存在“认知阈值”——7B 等小模型无法利用跨会话记忆、甚至略降，方法对小模型不友好；(4) 无显式遗忘/冲突消解机制，记忆只增长，长期可扩展性与噪声治理未充分验证；(5) 未对 Coach 做强化学习/长期奖励优化（列为未来工作）；(6) 为未经同行评审的 work-in-progress 预印本，代码星标少、独立复现信号弱。
- **与其他工作关系**: 属本研究 C 簇（经验回放与技能/程序性记忆）的网页智能体自我改进路线。机制上与 C1 Voyager、A6 ReasoningBank 同属“agent-centric 记忆自我改进”（记忆自身经验而非用户信息）；其“成功工作流/失败模式蒸馏 + 检索注入”与 A5 ExpeL、C2 Agent Workflow Memory（AWM）、C3 Synapse 的经验/工作流复用思路相近，但 WebCoach 不固化可执行技能库、改以运行时教练建议注入。它显式借鉴 Reflexion（shinn2023reflexion）的情景反思与上下文回放（liu2025contextual contextual replay），并将其整合进主流网页导航管线。区别于把记忆做成知识图谱（如 C6 ExpGraph 的经验图、B9 MIRIX）或 OS 分层（B3 MemGPT、B7 Memory-OS）：WebCoach 用扁平 FAISS/HNSW 向量库 + 独立 Coach 层。与 2025-26“学习记忆控制策略”代际（Memory-R1、Mem-α、ExpGraph 的 PPO 检索副驾）不同，WebCoach 的 Coach 本文为零样本提示、未训练记忆控制策略（虽设计上可训练）。其“模型无关、不重训 actor”定位与 ExpGraph 的冻结执行器理念一致，但聚焦真实在线网页域。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式管线为主）。写入由固定路由规则（仅持久化完整终止轨迹）驱动，检索由相似度+新近度的固定 top-K（K=5）启发式驱动，干预由 Coach 的提示/规则式决策（检测高失败概率或更优工作流则干预）驱动。Coach 设计上“可被训练或提示（trainable or prompted）”，留有学习记忆/干预策略的空间，但本文实验为零样本提示、并明确放弃了对 Qwen3 的 DPO 微调，因此未真正学习记忆管理策略（区别于 Memory-R1/Mem-α/ExpGraph 等用 RL 学控制策略的代际）。
- **记忆主体**: 智能体中心（agent-centric）：记忆的是网页智能体自身的浏览经验（成功工作流与失败模式）以自我改进任务表现，而非记住用户个人信息做个性化；与 ReasoningBank/Voyager 同范畴，区别于 Mem0/Zep/LongMemEval 的用户中心记忆。论文亦强调跨 actor 共享：EMS 可被不同模型/任务复用，但复用的仍是“智能体经验”。
- **多智能体记忆**: 本质单智能体改进，但记忆库具备跨智能体共享属性：EMS 被设计为“cross-actor、cross-task 知识库”，可由不同 actor 写入与读取，一个 actor 也能用其他模型（如 GPT-4o）产生的种子经验（Frozen EMS 实验）。不过论文未实现多智能体间的记忆路由/分层协作（无 G-Memory/MIRIX 式的多智能体记忆路由架构），且结论是自我经验优于借来的外部经验。
- **时序推理支持**: 无显式时间有效性建模。不维护事实有效期窗口或事件时序日历（区别于 Zep/Graphiti）。检索使用“新近度（recency）”作为排序信号之一、记录带 timestamp 元数据，属隐式时间偏好，而非显式事件排序或事实时效推理。
- **模态**: 多模态（视觉-语言）：actor 为多模态 VLM（Qwen2.5-VL / Skywork-r1v3 / GPT-4o），在真实浏览器中处理网页（截图/视觉 + 文本动作）；但 EMS 中存储的记忆条目本身是文本摘要 + 文本嵌入（summary_text + embedding），即“多模态智能体、文本化记忆”。

**不确定字段 / Uncertain**

- 冲突/矛盾处理 (`conflict_contradiction_handling`)
- 过度个性化/记忆安全风险 (`over_personalization_risk`)
- 可复现性 (`reproducibility`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="c8-ui-mem全称self-evolving-experience-memory-for-online-reinforcement-learning-in-mobile-gui-agents面向移动-gui-智能体在线强化学习的自演化经验记忆论文中亦写作-uimem"></a>

### C8 UI-Mem

*UI-Mem（全称：Self-Evolving Experience Memory for Online Reinforcement Learning in Mobile GUI Agents，面向移动 GUI 智能体在线强化学习的自演化经验记忆；论文中亦写作 UIMem）*


**基本信息 / Provenance**

- **年份**: 2026（arXiv 预印本，2026-02-05 首次公开，版本 v1）
- **作者/机构**: 第一/共同第一作者为 Han Xiao（肖涵，香港中文大学 CUHK MMLab）、Guozhi Wang、Hao Wang（均为 vivo AI Lab，三人并列共同第一作者）；其他作者含 Shilong Liu（普林斯顿大学 Princeton）、Yuxiang Chai（CUHK MMLab）、Yue Pan、Yufeng Zhou、Xiaoxin Chen、Yafei Wen（vivo AI Lab），通讯作者 Hongsheng Li（李鸿升，CUHK MMLab / 深圳河套研究院 / 上海人工智能实验室）。主要单位为香港中文大学多媒体实验室（CUHK MMLab）与 vivo AI Lab 的产学合作。
- **发表venue**: arXiv 预印本（cs.AI/cs.LG，2026-02），截至调研日尚无正式会议/期刊发表记录；属产学合作研究成果（vivo AI Lab × CUHK），项目主页 https://ui-mem.github.io 标注代码、模型、数据「Coming Soon」。
- **论文链接**: https://arxiv.org/abs/2602.05832
- **代码链接**: https://ui-mem.github.io （官方项目主页）；代码/模型/数据截至调研日均标注「Coming Soon」，尚未开源发布，无可用 GitHub 仓库与 star 数。
- **引用数**: 约 7 次引用（Semantic Scholar，截至调研日；paperId b0973b59d344facb9efdf3d4f108fd2b35d198e3，CorpusId 285304322；属 2026 年新近预印本，引用量低、成熟度尚早）。

**记忆分类 / Taxonomy**

- **记忆类型**: 以程序性记忆（procedural）为主、含情景蒸馏与「负向/失败」记忆的复合体。三层结构分别对应：高层工作流（任务规划级程序性策略）、中层子任务技能（执行级原子技能库）、失败模式（从失败轨迹提炼的避坑教训）。本质是把原始经验抽象为可复用、可参数化的程序性知识，明确区别于存储原始状态-动作对的传统回放缓冲（replay buffer）。
- **记忆结构**: 分层经验记忆（Hierarchical Experience Memory），三级层次：(1) 高层工作流 W —— 子任务的有序序列（如「发邮件」=[打开邮件App, 选择收件人, 输入内容, 发送]）；(2) 中层子任务技能 Σ —— 以自然语言概括的原子动作序列技能库（如「点击搜索图标, 输入{{query}}, 选第一个结果」）；(3) 失败模式 F —— 从失败轨迹提取的常见错误诊断（如「输入文件名前勿点保存图标」）。所有条目经抽象转为参数化模板（parameterized templates），将具体值（文件名、日期、电话、UI 元素名）替换为语义占位符（如 report.pdf→{{filename}}），以最大化跨任务迁移并压缩冗余。
- **存储后端**: 向量数据库（vector database），以任务与子任务描述的语义嵌入（embedding）为索引，支持大规模相似度检索。各条目附带统计量（成功计数 N_succ、使用计数 N_used、时间戳）。检索到的模板在运行时实例化后以文本形式注入 rollout 提示（in-context 注入），而记忆库本身为外部持久化向量存储。
- **持久化**: 外部持久化存储（durable external store）+ 参数内化（parametric internalization）的混合。记忆以向量库形式跨任务、跨训练迭代持久保留并随训练自演化；同时其设计目标是通过在线 RL（GRPO）训练把外部记忆中的知识逐步「内化」进策略模型参数，使无引导（No-Guidance）策略最终也能复现引导行为。因此既有外部可检索记忆，又有向模型权重的参数化迁移。

**核心机制 / Mechanisms**

- **写入/编码**: 在每次训练迭代结束时由自演化循环写入。先用奖励模型评估每条轨迹（判定全局成功与已完成子任务），再由 LLM 经验抽取模块（采用 Seed1.8）抽取结构化经验：对成功轨迹（R(τ)=1）抽取高层工作流 W 与每个已完成子任务的执行计划 Σ；对失败轨迹（R(τ)=0）抽取已成功完成子任务的有效计划，并针对「第一个失败子任务」生成失败诊断 F。随后通过抽象机制把原始抽取物转为参数化模板（具体值替换为语义占位符，如「report.pdf」→{{filename}}），以最大化可迁移性并合并语义等价经验、减少冗余。与逐字保存轨迹的回放缓冲不同，写入的是抽象化、结构化的高层知识。
- **检索机制**: 基于向量库的语义匹配 + UCB 启发式打分。给定新任务指令 q，先计算其嵌入并与存储的任务模板做语义匹配；对匹配到的模板抽取指令专属的变量绑定，把占位符替换为实际值以实例化具体计划。一个任务模板可关联多条历史经验条目，为平衡利用与探索，采用受置信上界（UCB）启发的检索打分：S(p)=N_succ(p)/Σ_{p'}N_succ(p') + λ_ucb·sqrt( ln(Σ_{p'}N_used(p')) / (N_used(p)+1) )，第一项偏好历史成功率高的计划，第二项给被尝试较少的计划探索加成。对失败模式 F 则采用「近因偏置（recency bias）」优先选最近的错误诊断，使引导反映智能体最新策略。检索是自适应的，随记忆在训练中精炼而动态演化。
- **反思/巩固**: 自演化循环（Self-Evolving Loop）即「原始轨迹→高层洞见」的反思巩固机制。每个训练迭代末：先用两阶段文本化验证流水线打分（Qwen2.5-VL-72B 把屏幕状态与动作转为文本描述，再由 DeepSeek-V3 做基于规则的状态校验，准确率 0.900 显著优于直接 MLLM 打分的 0.724），再由 LLM 抽取模块把成功轨迹提炼为工作流 W 与子任务计划 Σ、把失败轨迹提炼为失败诊断 F，并经模板抽象（具体值→占位符）泛化。该过程持续把原始经验抽象为更高层、可迁移的知识，使记忆与不断演化的策略保持对齐，形成进步式自我改进与跨任务迁移闭环。
- **遗忘/更新**: 增量式合并更新为主：对新抽取条目计算语义嵌入，在向量库中查询高余弦相似度的既有对应项——若存在相似项则合并（成功计划增加成功计数 N_succ，失败模式更新时间戳），若不存在则新建条目并初始化统计量（N_succ=1, N_used=0）。失败模式以近因偏置体现「软遗忘」（优先近期诊断）。论文未实现显式的删除/衰减/容量上限等强遗忘机制，更新核心是「合并 + 统计累积 + 时间戳刷新」，使记忆库与策略协同演化。
- **经验回放 (核心主题)**: 本文核心主题，但刻意区别于传统经验回放。作者明确指出传统回放缓冲存储原始状态-动作序列、方差高且对 UI 微小变化泛化差，难以跨任务/跨应用迁移。UI-Mem 不重放原始轨迹，而是把过往（成功与失败）经验蒸馏为分层、抽象、参数化的工作流/子任务技能/失败模式，并在后续 rollout 中通过分层检索实例化、以「分层引导」主动注入在线 RL 采样，引导生成高质量新轨迹而非被动重放静态历史。消融与对比显示：相比 GRPO+Experience Replay（混入 top-k 历史成功轨迹，56.5%）和静态复用基线 MobileRL（静态经验回放，44.9% vs 43.5），UI-Mem 用「主动记忆引导探索」显著胜出，因其引导生成新轨迹并配以多样化进度奖励，能更高效探索状态空间、适应动态 UI，而非仅记忆既有路径。

**学习维度 / Learning**

- **学习范式**: 混合（hybrid），且偏参数化。核心是把非参数的外部分层记忆与在线 RL 的梯度更新耦合：用 Group Relative Policy Optimization（GRPO）对策略模型（Qwen3-VL-4B/8B）做梯度训练，把记忆中的知识内化进参数；同时记忆库本身以非参数方式（向量检索 + 模板实例化）演化并在推理时可选地注入上下文。属于「2025-26 学习型记忆控制」分水岭中偏向用训练耦合记忆的一侧。
- **失败学习 (核心主题)**: 本文核心主题之一，机制明确且系统。失败学习贯穿三处：(1) 失败模式记忆 F —— 对失败轨迹专门抽取「第一个失败子任务」的失败诊断并存为参数化模板（如「输入文件名前避免点击保存图标」），检索时以近因偏置优先注入，使智能体主动规避已知错误而非反复试错；(2) 失败诊断驱动的纠错 —— 定性分析（论文 Figure，Error Correction via Failure Diagnosis）显示系统能在第一次 rollout 识别导航错误（如该进入列表却返回上一页），生成具体「纠正指引（Correction Guideline）」，使第二轮 rollout 成功；(3) 失败轨迹中已成功的子任务计划仍被保留复用。失败模式与工作流、子任务技能构成互补的三类知识（消融显示三者缺一均显著掉点）。
- **技能/程序归纳**: 是，且为核心贡献。显式从轨迹归纳三层可复用程序性知识：高层工作流（子任务有序序列）、中层子任务技能（跨任务复现的原子能力，如「搜索某项」「填表单字段」「导航到某文件夹」，以自然语言概括动作序列表示）、失败模式。技能以参数化模板存储、经语义检索匹配并用实例变量实例化为具体计划后调用；并可跨应用复用（交互范式相似的应用间高度可迁移）。
- **在线 vs 离线**: 在线为主（online，深度耦合在线 RL）。记忆在部署期/训练期的在线 rollout 中逐迭代累积与精炼（自演化循环），与策略协同演化；同时训练数据集是离线构建的（从 AMEX、AndroidLab、UI-Genie 收集任务指令并经 GPT-4o 增广，共 256 条训练 query，初始子任务定义由标注轨迹生成）。整体属「在线自演化 + 离线种子构建」的组合，强调随训练在线演化。

**评测 / Evaluation**

- **任务领域**: 移动 GUI 智能体（mobile GUI / Android 设备控制），需与真实/模拟 Android 环境动态交互的长程任务，覆盖邮件、联系人、笔记、日程、地图、记账、音乐、会议等日常应用操作。
- **基准**: 两大在线交互式 GUI 基准：(1) AndroidWorld（Rawles et al. 2024）—— 20 个真实应用上的 116 个程序化任务，长程依赖（常 >10 步）、动态任务参数化，奖励直接由系统状态导出；(2) AndroidLab（Xu et al. 2024）—— 9 个常用离线应用上的 138 个任务，提供子目标成功率（Sub-SR）、反向冗余比（RRR）、合理操作比（ROR）、总体成功率（SR）等细粒度指标。另在 AndroidLab 中划出 5 个训练未见的留出应用（Bluecoins、Cantook、Maps.me、Pi-Music、Zoom）做跨应用泛化测试。
- **报告增益**: AndroidWorld 成功率（SR）：UI-Mem-4B 达 58.2%，显著超过基座 Qwen3-VL-4B（45.3%，+12.9 绝对点）并超过更大的 UI-Venus-7B（49.1%）；加推理时记忆检索的 UI-Mem-4B⋆ 进一步达 62.5%。UI-Mem-8B 达 66.8%（基座 Qwen3-VL-8B 47.6%，+19.2 绝对点），UI-Mem-8B⋆ 达 71.1%，达到 SOTA，超过闭源商用 API Gemini-2.5-Pro（69.7%）与 Seed1.8（70.7%）。AndroidLab：UI-Mem-8B SR 43.5%、Sub-SR 52.7（基座 Qwen3-VL-8B SR 34.8、Sub-SR 45.3），UI-Mem-8B⋆ SR 44.9、Sub-SR 56.0、ROR 94.9，超过静态经验回放强基线 MobileRL（7B，42.5）与 UI-TARS-1.5-7B（40.6）；UI-Mem-4B⋆ SR 39.9、Sub-SR 51.9、ROR 94.6。RL 范式消融（AndroidWorld，8B）：Vanilla GRPO 53.0、GRPO+Progress Reward 57.3、Inference-time Prompting（仅 RAG 不训练）52.6、GRPO+Experience Replay 56.5、GRPO+Inference-time Prompting 55.2，而 UI-Mem 66.8——较 vanilla GRPO +13.8 点、较最佳消融变体 +9.5 点。组件消融：用原始经验替代抽象模板降至 58.2%，关闭记忆更新降至 62.9%，去除分层结构（仅用单一类型）或去掉分层采样均显著掉点。文本化验证流水线准确率 0.900 vs 直接 MLLM 打分 0.724。
- **对比基线**: (1) 闭源商用 API：Gemini-1.0/1.5-Pro、Gemini-2.5-Pro、Claude-3-Opus/3.5-Sonnet、GPT-4o、Seed1.5-VL、Seed1.8、UI-Tars-1.5、AutoGLM；(2) 开源 GUI 基座/智能体：Qwen3-VL-2B/4B/8B（基座）、MAI-UI-2B、ScaleCUA-3B、Ferret-UI Lite-3B、UI-Genie-Agent-3B/7B、UI-Venus-7B、GUI-Owl-7B、Step-GUI-8B、UI-TARS-1.5-7B、MobileRL（7B，静态经验回放强基线）；(3) RL 范式消融基线：Vanilla GRPO、GRPO+Progress Reward（密集奖励）、Inference-time Prompting（仅推理检索/RAG 无训练）、GRPO+Experience Replay（混入历史成功轨迹）、GRPO+Inference-time Prompting。核心对照是「主动记忆引导的在线 RL」对「无记忆 RL / 密集奖励 / 静态经验回放 / 纯推理时 RAG」的优越性。

**分析 / Analysis**

- **关键创新**: 把分层、自演化的结构化经验记忆（高层工作流 + 中层子任务技能 + 失败模式，均存为参数化模板）深度嵌入在线 GUI RL 的训练回路，并提出「分层组采样（Stratified Group Sampling）」——在同一 GRPO rollout 组内注入强/弱/无三档记忆引导，从而维持组内结果多样性、避免引导依赖与零优势方差，驱动无引导策略把外部记忆内化进参数。它把「经验复用」从被动重放原始轨迹（传统 replay buffer）升级为「主动记忆引导探索 + 在线自演化」，同时解决长程信用分配与跨任务/跨应用经验迁移两大瓶颈。
- **局限**: (1) 安全性：在线 RL 需与真实环境交互，探索阶段可能执行误删数据、误触发金融交易等非预期动作，作者承认需在真实部署前建立安全探索协议；(2) 隐私：处理含个人数据的截图存在隐私风险，模板抽象虽用 {{password}} 等占位符替换敏感文本，但仍需严格处理流程；(3) 可复现性：代码/模型/数据截至调研日均「Coming Soon」未开源，且依赖多个外部大模型（Qwen2.5-VL-72B、DeepSeek-V3、Seed1.8、GPT-4o 增广），复现门槛高；(4) 评测仅覆盖移动 Android GUI 单一大类，未涉具身、网页桌面、对话等域；(5) 训练数据规模较小（仅 256 条 query），记忆库无显式遗忘/容量上限，长期大规模运行下记忆膨胀与陈旧问题未充分评估；(6) 依赖奖励模型与文本化验证流水线的判定准确性。
- **与其他工作关系**: 属「C. 经验回放 & 技能/程序性记忆」簇。它把 Voyager（C1，技能库自我演化）、AWM（C2，从轨迹归纳可复用工作流）、CLIN/ExpeL（轨迹转洞见/经验抽取）等「记忆即可复用技能」思路迁移到移动 GUI 并与在线 RL 深度耦合，是该思路在 GUI 域的代表实现。与 AWM（C2）相比，UI-Mem 不仅归纳工作流，还显式建模子任务技能与失败模式三层结构，且把记忆用于「引导在线 RL 采样并内化进参数」而非仅 prompt 注入；与 ReasoningBank（A6）一样强调从失败学习（失败模式/失败诊断），但落点在 GUI RL 训练而非纯推理时记忆。与同期 GUI 记忆工作（如 MAGNET 的程序记忆+UI 元素记忆、HyMEM 的图结构自演化记忆）同属「自演化 GUI 记忆」前沿，但 UI-Mem 独特在于把记忆嵌入 GRPO 训练并提出分层组采样。与传统 RL 经验回放（DigiRL、MobileRL 等静态 replay）明确区分——它存抽象知识而非原始轨迹、主动引导而非被动重放。属智能体中心（agent-centric）自我改进记忆，区别于 Mem0/Zep/LongMemEval 等用户中心记忆。在 fields.yaml 中被列为「学习型记忆控制」分水岭代表之一（与 Memory-R1、Mem-α、SkillOS、CODESKILL 并列）。
- **可复现性**: 可复现性偏低（截至调研日）：官方项目主页（ui-mem.github.io）标注代码、模型、数据「Coming Soon」，尚无开源仓库与可下载权重；所用基准 AndroidWorld、AndroidLab 为公开在线基准（可复现的部分），但完整流水线依赖多个大模型（基座 Qwen3-VL-4B/8B、验证用 Qwen2.5-VL-72B + DeepSeek-V3、抽取用 Seed1.8、增广用 GPT-4o）与分布式 Android 模拟器集群，工程复杂、复现成本高。作为 2026 新近预印本，引用约 7 次、社区采用信号尚弱。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 部分是（混合）。记忆的检索打分（UCB 启发式 + 近因偏置）、抽取与合并更新本身为启发式流程，并非用 RL 直接学习「存什么/何时存/取什么」的元策略；但整体框架用在线 RL（GRPO）训练策略去「学会利用并内化记忆引导」，并通过分层组采样 + 动态 dropout 课程 + 引导感知奖励塑形（对无引导成功给额外 bonus）来端到端优化记忆引导的使用方式。因此其记忆「控制」是启发式管线与学习型策略内化的混合，处于 2025-26 学习型记忆控制分水岭的过渡地带。
- **记忆主体**: 智能体中心（agent-centric）：记忆的是智能体自身在 GUI 交互中归纳的可复用工作流、子任务技能与失败教训，目的是让 GUI 智能体跨任务/跨应用自我改进、提升长程任务成功率与样本效率，而非记住用户个人信息做个性化。与 Voyager、AWM、ReasoningBank 同类，区别于 Mem0/Zep/LongMemEval 等用户中心记忆。
- **多智能体记忆**: 单智能体（single-agent）。分层经验记忆服务于单个 GUI 策略智能体的在线 RL 训练与推理，未涉及多智能体间共享、路由或分层记忆（区别于 G-Memory、MIRIX 等多智能体记忆）。
- **时序推理支持**: 否（基本不涉及）。不显式建模事实时效窗口、事件先后或时间有效性（区别于 Zep/Graphiti 的时间知识图谱）。仅在失败模式检索上用「近因偏置」与时间戳刷新体现轻量的时间因素（优先近期错误诊断以对齐最新策略），但这属记忆新鲜度管理而非时序推理能力。
- **模态**: 多模态/视觉具身（multimodal，视觉 GUI）。智能体基于视觉语言模型（Qwen3-VL）感知移动设备屏幕截图并执行 UI 动作；记忆抽取时用 Qwen2.5-VL-72B 把屏幕状态文本化，记忆条目以文本/参数化模板形式存储。属图形界面视觉交互场景，区别于纯文本记忆。
- **过度个性化/记忆安全风险**: 未直接处理用户个性化风险，但论文「Impact Statement」明确讨论了相关安全与隐私维度：在线 RL 探索阶段可能执行误删数据、误触发金融交易等有害动作，需安全探索协议；处理含个人数据的截图存在隐私风险，记忆抽象虽用 {{password}} 等占位符替换敏感文本但仍需严格处理流程。不涉及 sycophantic/过度个性化等用户中心记忆安全维度。

**不确定字段 / Uncertain**

- 冲突/矛盾处理 (`conflict_contradiction_handling`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="c9-ell--stulife经验驱动终身学习框架--stulife-基准ell--experience-driven-lifelong-learning论文题为building-self-evolving-agents-via-experience-driven-lifelong-learning-a-framework-and-benchmark注意本文主要是一个概念性框架--评测基准而非单一可运行的记忆系统"></a>

### C9 ELL / StuLife

*ELL / StuLife（经验驱动终身学习框架 + StuLife 基准）。ELL = Experience-driven Lifelong Learning；论文题为《Building Self-Evolving Agents via Experience-Driven Lifelong Learning: A Framework and Benchmark》。注意：本文主要是一个概念性框架 + 评测基准，而非单一可运行的记忆系统。*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 首次提交 2025-08-26 v1；最新修订 v6 为 2026-01-26）
- **作者/机构**: Yuxuan Cai、Yipeng Hao、Jie Zhou、Hang Yan、Zhikai Lei、Rui Zheng、Zhenhua Han、Yutao Yang、Junsong Li、Qianjun Pan、Tianyu Huai、Qin Chen、Xin Li、Kai Chen、Bo Zhang、Xipeng Qiu、Liang He 等 17 位作者。第一/通讯单位为华东师范大学（ECNU）计算机科学与技术学院，合作单位包括上海人工智能实验室（Shanghai AI Lab）、香港中文大学（CUHK）、复旦大学（Fudan）。
- **论文链接**: https://arxiv.org/abs/2508.19005

**记忆分类 / Taxonomy**

- **记忆类型**: 情景（episodic/Trajectory Memory）+ 语义（semantic：声明性知识 M_decl、结构性知识 M_struct）+ 程序性（procedural：技能 F_proce、启发式 F_heur、元知识 F_meta）+ 工作记忆（STM/Working Memory）。论文在形式化定义中把 Knowledge K=(Memory M, Skills F) 显式拆为记忆与技能两大类，覆盖 CoALA 全部四类记忆。
- **记忆结构**: 分层/混合：短期记忆(STM/工作记忆)保存即时上下文 + 长期记忆(LTM)保存蒸馏经验。LTM 内部进一步细分为：轨迹记忆(原始或摘要的 ξ)、声明性知识(事实/概念)、结构性知识(语义网络/知识图谱表示概念间关系)、程序性/启发式/元知识技能库。框架层面是抽象的‘记忆 + 技能’双库结构，并非单一固定数据结构；实验中以不同具体实现替代（RAG 向量库、GraphRAG 图、MemGPT 分层、MemoryBank）。
- **存储后端**: 框架本身不绑定具体后端（抽象记忆-技能库）。StuLife 环境用单一持续 Python 对象 CampusEnvironment 维护世界状态；智能体跨任务记忆依赖环境提供的工具（日历/calendar、草稿/draft、邮件、预约系统等）做外部化存储。在记忆增强实验中分别接入向量检索(Vanilla RAG)、图(GraphRAG/Neo4j 类)、MemGPT(分层上下文+外部存储)、MemoryBank 等后端。
- **持久化**: 外部持久化为主：长期记忆/技能保存在持久化的知识库 K 中并跨任务传递（任务 T(i) 结束后的知识库成为 T(i+1) 的初始知识库）。同时提出可选的‘知识内化(Knowledge Internalization)’——通过 SFT/知识蒸馏把显式经验固化进模型参数(parametric)，即从 in-context 显式记忆向参数化隐式能力迁移。默认评测设定下 LLM 本身是无状态(stateless)的，记忆完全靠外部工具外化。

**核心机制 / Mechanisms**

- **写入/编码**: 经验以轨迹 ξ=⟨o0,a0,r0,...,oT,aT,rT⟩ 的形式产生，经‘知识抽象(Knowledge Abstraction)’把原始经验转化为结构化的记忆+技能：轨迹被原样或摘要存为 Trajectory Memory；从中抽取事实/概念形成声明性知识、抽取概念关系形成结构性知识(语义网络/KG)；从重复模式抽象出可复用技能(程序性规则、启发式、元知识)。学习函数 Φ_learn(K, ξ, g) 在每次试验后更新知识库，支持 Add/Update/Delete/Combine 四类操作。可选地把高质量经验/lesson 用于 SFT 完成参数化内化。
- **检索机制**: 框架层面强调长跨度的‘联想式回忆(associative recall)’与语义/时间/因果索引，但 ELL 本身未规定单一检索算法。论文用 Memory Utilization Score（受 GoodAI LTM Score 启发）评估检索质量：对每个需回忆既往事实的任务给出检索准确率，并按‘记忆距离’(编码到检索之间的时间步/episode/token 数)加权。实验对比多种具体检索：向量相似度(Vanilla RAG)、图遍历(GraphRAG)、MemGPT 的分层管理检索、MemoryBank。结论：朴素 RAG 直接把原始轨迹按相似度注入上下文反而引入噪声、损害性能(StuGPA 降至 10.98，LTRR 低于基线)；结构化管理的 MemGPT 检索最佳(StuGPA 19.99，考试成功率升至 23.75%)。
- **反思/巩固**: 是核心机制之一。‘学习(Learning)’模块为元认知反思架构：把同一任务多条轨迹(观测序列、动作、决策理由、奖励)聚合为统一上下文，由带元提示(meta-prompt)的反思模块做结构化回溯分析——典型问题如‘哪些策略带来更高累计奖励？哪些动作导致失败/次优？是否存在可泛化模式？下一步应如何调整？’。提炼出的 lessons 被显式追加进系统提示，或存入可检索的动态 lesson 库，用于后续任务的上下文增强或知识蒸馏。框架还提出二阶段的‘知识精炼(Knowledge Refinement)’(Add/Update/Delete/Combine)保持知识库最优、最新，以及最终的‘知识内化’把显式规则蒸馏为模型直觉。实验中以 Reflexion、AWM 等具体反思/工作流记忆方法实例化。
- **遗忘/更新**: 通过统一学习函数 Φ_learn 的四类操作实现：Add(新增)、Update(更新)、Delete(删除过时信息)、Combine(合并/巩固相似记忆或技能)。引入‘知识验证(Knowledge Validation)’：用性能增益 V(K(i-1),T(i))=J(带知识策略)−J(基线策略 π0) 衡量历史知识效用，V<0 表示知识过时/无关，触发由 Φ_learn 执行的精炼或剪枝。框架目标显式声明要‘防止灾难性遗忘并促进前向迁移’，但论文也承认现有方法在长期保持(LTRR)上仍极差(最优模型约 90% 失败)。
- **经验回放 (核心主题)**: 核心主题。整体范式即从第一人称经验中反复试验、复用：对每个任务执行多次试验 k=1..Ki，用当前知识 K(i,k-1) 生成新轨迹，再以成功/失败经验更新知识库供后续试验与后续任务复用。具体复用形式包括：(1) 训练式——RFT(拒绝采样微调)，每个 prompt 采样 8 条 rollout、仅用成功轨迹微调(把经验蒸馏进 8B 小模型)；(2) 推理式——AWM(Agent Workflow Memory) 把成功工作流抽象存储并在新任务检索复用、Reflexion 把失败反思复用为上下文；(3) 提示式——Skill-Augmented Prompt 提供 step-by-step ‘recipe’、Memory-Augmented Prompt 注入历史。论文证明 AWM 复用成功工作流既提升成功率(Total Success 8.52%→10.12%)又降低冗余步数(Total AvgTurn 16.95→13.96)。

**学习维度 / Learning**

- **学习范式**: 混合(hybrid)。框架同时容纳非参数(in-context/提示级，如 Reflexion、AWM、各类记忆增强提示)与参数化(梯度更新，如 RFT 微调，以及‘知识内化’阶段的 SFT/知识蒸馏)。论文核心论点之一是：从显式 in-context 学习逐步过渡到把经验内化进模型参数(‘第二天性’)。实验同时给出训练式(8B+RFT)与推理式(235B+AWM/Reflexion)两条演化路径的对比。
- **失败学习 (核心主题)**: 核心主题。反思模块显式分析失败：元提示要求识别‘哪些动作导致失败或次优结果’，并据此调整后续策略；失败经验与成功经验一并进入轨迹记忆与 lesson 库。Reflexion 即典型的失败自反思机制(把失败反馈转为语言反思复用)。RFT 则反向用‘仅保留成功轨迹’做拒绝采样过滤(隐式地丢弃失败)。论文还诊断了一类突出失败模式：到点(如 8:00)即便被告知当前时间，智能体仍不会主动查日历发起对应动作——由 PIS(主动性得分)量化，最优模型 GPT-5 仅 4.68%，远低于人类 88.13%。
- **技能/程序归纳**: 是。ELL 第三大原则即 Skill Learning：从经验中抽象出可复用技能(决策规则、功能模块、问题求解启发式)，并通过反思显式构造、在新任务中验证，主动对技能库做增加/精炼/组合/弃用管理。技能形式化为 F_proce(程序性‘怎么做’)、F_meta(元知识)、F_heur(经验性决策规则)。框架讨论了技能粒度(低层动作 vs 高层策略)、抽取、验证、选择与失败检测等技能生命周期问题。实验以 Skill-Augmented Prompt(提供 step-by-step recipe)和 AWM(工作流记忆)实例化技能复用。
- **在线 vs 离线**: 两者兼有。在线：StuLife 为单一连续、有状态轨迹，任务串行呈现，智能体在部署中按 episode 在线积累知识(K(i) 传给 T(i+1))。离线：RFT 在探索后批量收集成功轨迹再做微调；‘知识内化’在积累足够高质量 lessons 后批量 SFT。框架明确把内化设想为可在‘空闲期/离线巩固’(类比人类睡眠中的离线巩固)进行。

**评测 / Evaluation**

- **任务领域**: 面向具身式/智能体长程决策的‘虚拟大学校园生活’模拟：包含课堂学习(In-Class)、日常校园活动(Daily Campus：校园探索/选课/规划/学术活动/图书馆自习/社团活动)、考试(Examination：期中/期末)。涉及时间管理、空间导航(地图/路径规划)、选课与资源预约、邮件/信息查询、长期规划与社交。属 agent-centric 的长期记忆/技能/主动性综合评测，而非单纯 QA 或多轮对话。
- **基准**: 自建 StuLife 基准(论文主贡献)：1,284 个任务实例、10 个互联子场景、覆盖模拟一学年；其中需长期记忆的样本 554 个、需自我驱动(self-motivation)的样本 628 个；基于 LifelongAgentBench(LAB) 框架构建。对比的既有方法/系统作为基线，包括 Vanilla RAG、GraphRAG、MemoryBank、MemGPT、Reflexion、AWM、RFT。基准与现有 Lifelong-CIFAR10/ImageNet、CGLB、EgoThink、EmbodiedBench 做了能力维度对比(顺序性/技能学习/长期记忆/自我驱动/交互性/从经验学习)。
- **报告增益**: 核心结论是‘现有系统全面失败、提升有限’。默认无状态设定下所有 SOTA LLM 表现极差：最强的 GPT-5 StuGPA 仅 17.90/100，人类为 85.24；主动性 PIS：GPT-5 仅 4.68% vs 人类 88.13%；长期保持 LTRR：最高 Grok4 仅 10.65% vs 人类 84.91%(即最优模型约 90% 任务失败)。自演化机制(基于 Qwen3-235B-A22B 与 Qwen3-8B)：RFT 把 8B 的 StuGPA 13.31→15.43、Total Success 6.71%→8.63%；AWM 把 235B 的 StuGPA 16.03→17.81、Total Success 8.52%→10.12%、并把 Total AvgTurn 16.95→13.96(同时提效)；Reflexion 仅边际提升(StuGPA 16.18)。上下文工程(基于 Qwen3-235B-A22B)：Proactive 提示把 In-Class success 2.10%→5.09%、PIS 1.80%→3.06%；Skill 提示把 Daily Campus 10.34%→15.28% 但 PIS 反降至 0.90%；朴素 RAG 反而有害(StuGPA 降至 10.98，LTRR 5.42%→4.69%)；MemGPT 最佳单项(StuGPA 19.99，考试 23.75%)；All-in-One 综合最优(StuGPA 21.07 超过 GPT-5、PIS 3.76%、LTRR 升至 9.39%)。
- **对比基线**: 对比对象包括：(1) 无记忆/无状态原版 LLM(默认设定，13 个模型从 Llama-3.1-8B 到 GPT-5/Gemini-2.5-Pro/Grok4)；(2) 人类基线(招募本科/研究生)；(3) 自演化方法：RFT(训练式)、Reflexion、AWM(推理式)；(4) 记忆/检索系统：Vanilla RAG、GraphRAG、MemGPT、MemoryBank；(5) 提示策略：Vanilla/Proactive/Skill/Memory/All-in-One。

**分析 / Analysis**

- **关键创新**: 首次提出‘经验驱动终身学习(ELL)’的统一形式化框架(四原则：经验探索/长期记忆/技能学习/知识内化；以 POMDP+知识库 K=(M,F)+学习函数 Φ_learn 形式化)，并配套发布首个面向 ELL、有状态持续演化的‘虚拟大学生涯’长程基准 StuLife(1,284 任务、新指标 StuGPA/LTRR/PIS)，把记忆、技能、主动性三者整合评测，揭示当前 SOTA LLM 与人类的巨大差距。
- **局限**: (1) 主体是框架+基准，未提出可直接落地的单一新记忆系统，‘知识内化’等机制多停留在概念/愿景层面；(2) 评测显示所有现有方法提升有限，长期记忆(LTRR)与主动性(PIS)仍远未解决；(3) 任务域较窄(单一校园生活模拟)，泛化性待验证；(4) 奖励稀疏/难定义、内化的触发时机与可解释性等开放问题未解决；(5) 朴素 RAG 反而有害，说明记忆系统设计高度敏感、缺乏稳健通用方案；(6) 默认设定把 LLM 当无状态、记忆靠工具外化，与真正具备原生记忆的系统评测仍有距离。
- **与其他工作关系**: 本研究将 ELL/StuLife 归入 C 类(经验回放与技能/程序性记忆)。它把多条已有记忆/经验复用路线放在同一长程基准下对比：直接评测并复用了 Reflexion(失败自反思，对应 A 类反思系列)、AWM(Agent Workflow Memory，技能/工作流复用)、MemGPT(分层上下文记忆)、MemoryBank(带遗忘曲线的记忆)、Vanilla RAG / GraphRAG(检索增强)。在‘从经验自我改进’这一 agent-centric 取向上与 Voyager/ReasoningBank 同源；与偏 user-centric 的 Mem0/Zep/LongMemEval 不同(后者重在记住用户信息做个性化)。后续同组工作 ProactAgent(arXiv 2604.20572)在 StuLife 等基准上进一步引入‘主动检索 RL 策略’，可视为对本文 PIS/主动性短板的延伸改进。
- **可复现性**: 较好。代码与基准开源(GitHub ECNU-ICALK/ELL-StuLife，约 70 stars，Python，基于 LAB 框架)，提供环境搭建、配置与运行脚本、tools.json 工具清单与评测协议；项目主页含公开 leaderboard 与结果提交流程(邮件/PR，每周一人工复核)。数据集 1,284 任务公开。社区采纳尚处早期(引用约 20)，但有持续维护(2026-01 仍有提交)。

**补充维度 / Supplemented (2025-26 frontier)**

- **记忆主体**: 以 agent-centric(智能体记住自身经验以自我改进)为主——核心是从自身轨迹抽象技能、积累 lessons、内化为能力。同时也包含部分‘个人化/世界状态’信息(校园中的承诺、日程、关系、GPA 等需被记住)，但目的仍是支撑智能体自身的长程决策，而非为外部用户做个性化。
- **多智能体记忆**: 单智能体设定。StuLife 评测单个智能体在持续校园环境中的终身学习，未涉及多智能体共享/路由记忆。
- **时序推理支持**: 强。StuLife 是时间驱动、有状态的世界：智能体按模拟时钟运行，必须主动查日历理解‘下一步该做什么’，按正确时间/地点完成课程与会议。专门设计 PIS(主动主动性/前瞻记忆 prospective memory)与 LTRR(跨长时距回忆)评估时间承诺的保持与按时执行；课程热度、座位可用性、位置等状态随时间动态变化，要求实时查询最新状态而非依赖过时信息。
- **模态**: 以文本为主(LLM 智能体 + 文本工具)。形式化定义允许多模态状态/观测(文本/图像/结构化数据)，但 StuLife 基准实现与评测为纯文本环境，未涉及视觉/具身实拍。
- **过度个性化/记忆安全风险**: 未直接研究记忆安全/过度个性化/隐私治理等负面维度。相关的‘记忆有害性’以另一形式出现：实验发现朴素 RAG 把未过滤的原始轨迹注入上下文会引入噪声、加剧长上下文退化、反而降低性能(StuGPA、LTRR 双降)，间接印证‘更多记忆并非总是更好、记忆系统设计至关重要’。‘个人责任分(Personal Responsibility)’指标会因资源浪费(占座不用)、违背承诺(错过自定会议)而扣分，含一定行为约束意味。
- **token成本/延迟证据**: 有部分效率证据但非 token/延迟的直接量化。主要用 AvgTurn(平均交互轮数)衡量效率：AWM 把 Total AvgTurn 从 16.95 降至 13.96(复用成功工作流减少冗余步骤)；MemGPT 则在部分模块增加轮数(考试 AvgTurn 升至 23.75)。论文亦指出朴素 RAG 注入原始轨迹会加剧长上下文退化(隐含上下文/算力成本)，但未报告具体 token 数或 p95 延迟等绝对数值。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)
- 代码链接 (`code_url`)
- 冲突/矛盾处理 (`conflict_contradiction_handling`)
- 学习型记忆控制 (`learned_memory_control`)
- 发表venue (`venue`)


<a id="c10-memp写作-memp即-memory-procedural--智能体程序性记忆框架"></a>

### C10 Memp

*Memp（写作 $Mem^p$，即 Memory-procedural / 智能体程序性记忆框架）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本，2025-08-08 首次公开）
- **作者/机构**: Runnan Fang（方润楠）、Yuan Liang、Xiaobin Wang、Jialong Wu、Shuofei Qiao、Pengjun Xie、Fei Huang、Huajun Chen、Ningyu Zhang（张宁豫，通讯作者）等；主要单位为浙江大学与阿里巴巴集团（部分作者关联南京大学软件新技术国家重点实验室）。Runnan Fang 与 Yuan Liang 为共同核心一作。
- **论文链接**: https://arxiv.org/abs/2508.06433
- **代码链接**: https://github.com/zjunlp/MemP（官方开源，Python，约 26 stars，截至 2026-06）
- **引用数**: 约 39 次（Semantic Scholar 实时数据，CorpusId 280561810）

**记忆分类 / Taxonomy**

- **记忆类型**: 程序性记忆（procedural memory）为核心，显式以人类程序性记忆（Cohen & Squire 1980）为类比；同时涉及由轨迹抽取的情景性内容（episodic 轨迹原文）与抽象化的脚本式知识（偏 semantic/procedural），但论文明确定位为对 CoALA 分类中 procedural memory 的探索与系统化。
- **记忆结构**: 外部可编辑的程序性记忆库（procedural memory repository / library），每条记忆是 key-value 形式：key 为任务查询向量或关键词向量，value 为两种粒度的程序性知识——(1) 细粒度逐步指令（verbatim 轨迹）与 (2) 高层脚本式抽象（script-like abstraction）；通过向量检索访问，可增删改。
- **存储后端**: 向量存储（基于嵌入模型 φ 的向量库，使用余弦相似度检索）；记忆条目存放于外部记忆库中，检索后以上下文 prompt 形式拼接进入 LLM。论文未指定具体向量数据库实现，属轻量外部向量索引。
- **持久化**: 外部持久化存储（external durable store）。程序性记忆独立于模型参数，可跨任务、跨 episode 持续累积、更新与裁剪（lifelong/continual），并可离线构建后迁移到其他模型；非参数化、不写入权重。

**核心机制 / Mechanisms**

- **写入/编码**: 通过 Builder B 将历史轨迹 τ 及其奖励 r 编码为程序性记忆 m^p（公式 Mem=Σ B(τ_t, r_t)）。论文系统比较了三种构建（Build）粒度策略：(1) Trajectory——筛选训练集中成功的黄金轨迹，逐回合 verbatim 原样存储；(2) Script——由 LLM 对黄金轨迹做分析与总结，蒸馏成抽象的脚本式程序性知识（高层指南）；(3) Proceduralization——将检索到的完整轨迹与模型生成的高层脚本结合，兼具具体范例与抽象指导，实验中综合表现最优。仅对成功/黄金轨迹进行编码以保证质量。
- **检索机制**: 面向新任务 t_new，检索与其最相似的历史任务对应的记忆：m_retrieved = argmax S(t_new, t_i)，其中 S 为嵌入向量的余弦相似度 φ(t_new)·φ(t_i)/(‖φ(t_new)‖‖φ(t_i)‖)（公式 4-5），取 top-k。论文比较了三种 key 构建策略：Random Sample（不用 key，随机抽取）、Key=Query（用任务查询描述作为 key，靠查询语义相似度检索）、Key=AveFact（用大模型从查询中抽取关键词，对匹配关键词取平均相似度检索）。结果显示 Query 与 AveFact 均显著优于随机抽样，AveFact 在 GPT-4o 上 #CS 最高（76.02）。检索量存在最优区间：检索记忆数增加先升后平台，过多会因上下文变长与引入低质量记忆而下降。
- **反思/巩固**: 核心机制之一。构建阶段的 Script/Proceduralization 本身即把原始轨迹抽象为更高层的脚本式知识（raw→insight）。更新阶段提供多种巩固策略，其中 Validation 做选择性巩固：每 t 个任务后只抽取成功完成的轨迹并抽象为紧凑符号化程序性记忆，丢弃失败、冗余与噪声数据；Adjustment（基于 reflexion 的纠错更新）在检索记忆导致失败时，将错误轨迹与原记忆结合并就地修订，生成更新后的记忆。论文报告 reflexion-based update 是最有效策略（末组任务上比次优策略 +0.7 分并减少约 14 步）。
- **遗忘/更新**: 提供显式的动态更新生命周期：把更新建模为 U = Add(M_new) ⊖ Del(M_obs) ⊕ Update(M_est)（公式 7），含新增、删除（弃用过时/冗余记忆）、就地修改三类操作；每隔 t 个任务刷新一次。具体策略含 Vanilla（直接 append 合并）、Validation（只保留成功轨迹并抽象、丢弃失败/噪声）、Adjustment（reflexion 纠错修订）。即论文强调的「持续更新、纠正、弃用（deprecate）」机制，使记忆库与新经验同步演化。
- **经验回放 (核心主题)**: 核心主题：将过去轨迹蒸馏为可复用模板（推理模式、工具调用序列、恢复策略），在面对相似新任务时检索并以上下文形式复用，从而避免从零开始的重复探索。论文将范式从「并行独立完成任务」转为「顺序完成」，使 agent 能从早期任务蒸馏经验、减少重复试错。实验显示随记忆库精炼，相似任务上成功率与效率持续近似线性提升；与 ReAct、ExpeL、AWM（Agent Workflow Memory）等经验复用基线相比，Memp 在 ALFWorld 上取得最高 dev/test 成功率且步数最少。被本研究归入「经验回放与技能/程序性记忆」类（C 类）。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric / in-context、prompt-level）。程序性记忆以自然语言形式存于外部库并通过检索注入上下文，不做梯度更新；显式对比并定位为不同于「将程序性知识纠缠在难以更新的模型参数中」的方案，因此构建成本低、可迁移、可终身更新。
- **失败学习 (核心主题)**: 核心主题：通过 Adjustment / reflexion-based 更新利用失败——当检索到的程序性记忆导致执行失败时，将错误轨迹与原记忆结合并就地修订（error-correction 嵌入 reflexion 过程），得到改进的记忆。Validation 策略则在巩固时显式丢弃失败轨迹与噪声数据。论文将 reflexion 纠错更新认定为最有效的更新机制（末组任务 +0.7 分、减少约 14 步）。但构建阶段主要存成功/黄金轨迹，失败主要在更新闭环中被用作纠错信号而非显式负样本记忆。
- **技能/程序归纳**: 是。框架核心即从经验中归纳可复用技能/工作流/程序——把轨迹蒸馏为细粒度逐步指令与高层脚本式抽象（Proceduralization），表示为外部记忆库条目，通过向量检索调用并拼接进上下文供策略 π_{m^p}(a_t|s_t) 使用；定位为「将程序性记忆作为一等优化对象」。
- **在线 vs 离线**: 两者兼有。离线：在训练集上批量构建程序性记忆库（并支持离线库迁移到弱模型）；在线：更新阶段允许 agent 在测试环境中边执行边构建与精炼记忆（每 t 个任务刷新一次），实现近似线性的持续掌握。

**评测 / Evaluation**

- **任务领域**: 长时序工具使用与复杂规划（信息检索/旅行规划）以及长时序具身家务任务。具体为 TravelPlanner（约束下的复杂工具调用与规划）和 ALFWorld（文本具身家务）。
- **基准**: TravelPlanner（Xie et al. 2024，指标 Commonsense #CS、Hard Constraint #HC、Steps）与 ALFWorld（Shridhar et al. 2021，含 dev/test split 衡量泛化，指标成功率与 Steps）。
- **报告增益**: 所有记忆构建方法均超过 No Memory 基线，且步数更少。以 GPT-4o 为例：ALFWorld 测试集成功率从 No Memory 的 42.14 提升到 Proceduralization 的 77.86（dev 从 39.28 升至 87.14），步数从 23.76 降至 15.01；TravelPlanner #CS 从 71.93 升至 79.94（Proceduralization），步数从 17.84 降至 14.62。Claude-3.5-sonnet：ALFWorld test 34.97→74.72，步数 24.12→15.79。Qwen2.5-72B：ALFWorld test 41.25→77.19，步数 21.38→15.32。检索策略上 AveFact 优于随机（GPT-4o #CS 74.59→76.02）。更新策略中 reflexion 纠错最优，末组任务比次优 +0.7 分、减少约 14 步。迁移：GPT-4o 构建的记忆用于 Qwen2.5-14B-Instruct，TravelPlanner 完成率提升约 5%、平均步数减少约 1.6（ALFWorld 亦有类似收益）。案例研究显示单任务可缩短 9 步、节省 685 tokens。综合对比基线（GPT-4o/ALFWorld）：ReAct 最差，ExpeL、AWM 渐好，Memp 最优。
- **对比基线**: No Memory（ReAct 无外部记忆）、Trajectory（仅存原始轨迹）、Script（仅存抽象脚本）等内部消融，以及外部经验复用方法 ReAct、ExpeL、AWM（Agent Workflow Memory）；检索消融对比 Random Sample / Key=Query / Key=AveFact；更新消融对比 Vanilla / Validation / Adjustment(reflexion)。

**分析 / Analysis**

- **关键创新**: 首次把程序性记忆当作「一等优化对象」，系统性拆解并对比其构建（Build）、检索（Retrieve）、更新（Update）三大生命周期模块的策略，提出 Proceduralization（轨迹+脚本融合）构建、AveFact 检索与 reflexion 纠错更新，并验证可终身更新与「强模型→弱模型」记忆迁移，填补了既有框架（Voyager/AWM/AutoManual 等）缺乏对程序性记忆生命周期系统分析的空白。
- **局限**: 作者自陈两点：(1) 检索仅限于带人工设计 key 的向量相似度搜索，未纳入 BM25 等其他经典/更精确方法；(2) 框架依赖 benchmark 提供的显式奖励信号，在奖励稀疏或缺失的真实场景中无法自行判定任务成功（未来拟用 LLM-as-judge）。此外检索过多记忆会因上下文变长与引入低质记忆而损害性能；评测仅限两个领域（TravelPlanner、ALFWorld）。
- **与其他工作关系**: 建立在并系统化了一系列经验/程序性记忆工作之上：扩展 Voyager（Wang 2023，技能库）、AWM/Agent Workflow Memory（Wang 2024b）、AutoManual（Chen 2024）等程序性记忆方法，但首次对 Build/Retrieve/Update 三模块做系统比较；更新机制借鉴 Reflexion（自我反思纠错）思想；与 ExpeL（经验抽取规则）、Synapse（轨迹即范例提示）同属经验复用谱系。属本研究 C 类（经验回放与技能/程序性记忆），与同类 agent-centric 自我改进系统（如 ReasoningBank、Voyager、Contextual Experience Replay）思路相通，区别在于聚焦程序性记忆生命周期的非参数化优化与跨模型迁移。
- **可复现性**: 代码官方开源于 https://github.com/zjunlp/MemP （zjunlp 课题组，Python，约 26 stars，含 AI assistant 使用与可复现性声明）；评测基准 TravelPlanner、ALFWorld 均为公开数据集；骨干模型为 GPT-4o、Claude-3.5-sonnet、Qwen2.5-72B/14B-Instruct（含开源模型，便于复现）。整体可复现性较好，社区采用信号中等（仓库较新、star 数尚少）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式流水线为主）。记忆的构建/检索/更新策略均为人工设计并经实验比较选优，未用 RL/训练去学习记忆管理策略本身。它处于「精心设计的启发式生命周期」一侧，而非 Memory-R1/Mem-α 那类以 RL 学习记忆管理 POLICY 的方向；其学习性体现在内容的持续更新与 reflexion 纠错，而非可学习的控制策略。
- **记忆主体**: 智能体中心（agent-centric）。记忆的是 agent 自身的任务执行经验/程序性知识以实现自我改进与跨任务复用，而非记忆用户信息做个性化；与 Voyager、ReasoningBank 同类，区别于 Mem0/Zep/LongMemEval 等 user-centric 个性化记忆。
- **多智能体记忆**: 单智能体。框架面向单个 agent 的跨轨迹程序性记忆；但验证了记忆可在模型间迁移（强模型 GPT-4o 构建→弱模型 Qwen2.5-14B 复用），具备一定可共享/可移植性，未涉及多智能体共享或记忆路由（如 G-Memory、MIRIX）。
- **时序推理支持**: 否。不显式建模时间有效性窗口或事件时序（不同于 Zep/Graphiti 的 fact-validity/event timeline）；更新中的「弃用过时记忆（deprecate）」属生命周期管理而非时间有效性推理。任务本身（旅行规划/家务）涉及多步时序执行，但记忆系统不专门做时间有效性推理。
- **模态**: 纯文本（text-only）。轨迹、脚本、观测反馈均为文本；ALFWorld 为文本具身环境，无视觉/多模态记忆。
- **过度个性化/记忆安全风险**: 未涉及（不适用/未讨论）。论文为 agent-centric 自我改进，不涉及用户个性化记忆，也未讨论有害/过时/谄媚记忆或隐私治理（OP-Bench/Causal-LoCoMo 类问题）。相关的「负面」考量体现在：检索过多记忆会损害性能，以及通过 Validation/Adjustment 丢弃失败与噪声记忆以防低质记忆累积危害表现。
- **冲突/矛盾处理**: 部分处理（偏向就地修订而非冲突合并）。更新机制含 Del（删除过时/冗余记忆）与 Update（基于执行反馈就地修订）；Adjustment 在记忆导致失败时将错误轨迹与原记忆结合并修订。这间接处理了「记忆与新经验不一致」的情形，但论文未把矛盾事实解析作为独立模块讨论（不同于 MEMTRACK/Memory-R1 的显式 UPDATE 冲突解析）。
- **token成本/延迟证据**: 有定量效率证据。程序性记忆显著减少步数与 token 消耗：GPT-4o/ALFWorld 步数 23.76→15.01；TravelPlanner 17.84→14.62；案例研究中单任务缩短 9 步、节省 685 tokens；迁移到 Qwen2.5-14B 平均步数减少约 1.6。论文强调相比逐题独立求解可消除大部分无谓探索，带来步数与 token 的大幅下降（未给出统一百分比或 p95 延迟等指标）。

**不确定字段 / Uncertain**

- 发表venue (`venue`)


## D. 图结构/神经启发/生产级记忆 (Graph / neuro-inspired / production)


<a id="d1-hipporag受海马体启发的-llm-长期记忆检索框架"></a>

### D1 HippoRAG

*HippoRAG（受海马体启发的 LLM 长期记忆检索框架）*


**基本信息 / Provenance**

- **年份**: 2024（arXiv 预印本 2024-05-23 首次公开；NeurIPS 2024 正式发表）
- **作者/机构**: Bernal Jiménez Gutiérrez（第一作者）、Yiheng Shu、Yu Gu、Michihiro Yasunaga、Yu Su（通讯/资深作者）。主要单位为俄亥俄州立大学（The Ohio State University, OSU NLP Group），其中 Michihiro Yasunaga 来自斯坦福大学（Stanford University）。
- **发表venue**: NeurIPS 2024（第 38 届神经信息处理系统大会，OpenReview id=hkujvAPVsg）；预印本见 arXiv 2405.14831（cs.CL）。
- **论文链接**: https://arxiv.org/abs/2405.14831
- **代码链接**: https://github.com/OSU-NLP-Group/HippoRAG （MIT 许可证；约 3.6k stars、364 forks；当前主分支已升级为 HippoRAG 2，初版 HippoRAG 1 代码见 legacy 分支）
- **引用数**: 约 223 次（Semantic Scholar 实时查询，2024 年发表后引用快速增长，影响力较高）

**记忆分类 / Taxonomy**

- **记忆类型**: 语义记忆（semantic）为主——将语料库蒸馏为开放知识图谱（KG）三元组，作为可持续整合的世界知识/事实型长期记忆，对应人脑“新皮层+海马索引”模型。本质是知识/事实记忆而非情景或程序性记忆；属 CoALA 中的语义记忆范畴，并具备“持续整合新经验”的长期记忆属性。
- **记忆结构**: 无模式（schemaless）开放知识图谱（KG）作为“人工海马索引”：节点 N 为从语料 OpenIE 抽取的名词短语/命名实体，边 E 为关系三元组；另由检索编码器在余弦相似度超过阈值 τ 的相似实体间补充“同义边”E′。这是关系型 KG/note-graph 式结构，区别于扁平向量库或原始缓冲。
- **持久化**: 外部持久化记忆（durable external store）：KG 索引独立于 LLM，跨查询持续存在并可“仅通过向 KG 增删边”增量整合新知识，无需重训模型、避免灾难性遗忘。属非参数化外部记忆（LLM 参数静态冻结）。

**核心机制 / Mechanisms**

- **写入/编码**: 离线索引（类比记忆编码/模式分离）：用强指令微调 LLM L（默认 GPT-3.5-turbo-1106）对语料中每段 passage 做两步 1-shot OpenIE——先抽命名实体，再据此抽取最终三元组（含超出命名实体的一般概念名词短语），从而把段落编码为离散名词短语节点 N 与关系边 E（而非孤立的稠密向量），实现细粒度模式分离。再用检索编码器 M（Contriever/ColBERTv2）在余弦相似度 > 阈值 τ=0.8 的相似实体间补充“同义边”E′（类比海马旁回 PHR 的连接）。同时构建 |N|×|P| 矩阵 P 记录每个名词短语在各 passage 中出现次数，供后续 passage 排序。新知识可通过向 KG 增添边持续整合，无需像 RAPTOR/GraphRAG 那样重做摘要。
- **检索机制**: 在线检索（类比模式完成）三步：(1) 用 L 对查询 q 做 1-shot 抽取“查询命名实体”C_q={c_1..c_n}（如 Stanford、Alzheimer’s）；(2) 用编码器 M 将其与 KG 节点按最高余弦相似度链接为“查询节点”R_q（r_i=argmax_j cos(M(c_i),M(e_j))）；(3) 以 R_q 为种子在 KG 上运行个性化 PageRank（PPR，阻尼/重启因子 0.5）：将概率质量仅从查询节点出发沿图扩散到联合邻域（如 Professor Thomas），得到节点概率分布 n′，再与计数矩阵 P 相乘得到每个 passage 的排序分 p。关键创新“节点特异性（node specificity）”：以 s_i=|P_i|^{-1}（节点 i 出现的 passage 数的倒数，类似仅用局部信号的 IDF）在 PPR 前调制查询节点概率，提升稀有概念权重。整个多跳推理在单步检索中完成。
- **反思/巩固**: 无显式“原始→洞见”的反思/总结式抽象。与 RAPTOR/GraphRAG 不同，HippoRAG 不对知识做摘要式压缩，而是通过 OpenIE 把段落转化为结构化三元组并入图，知识整合发生在“图结构层”——新信息通过向 KG 添加节点/边被持续整合（区别于需反复重做摘要的方法）。检索时 PPR 的图扩散可视为一种隐式联想/模式完成，但不产生更高层的反思性洞见或经验总结。
- **遗忘/更新**: 无 Ebbinghaus 时间衰减或显式遗忘机制。更新方式是“增量加边”：新知识只需向 KG 增添三元组/同义边即可持续整合，无需重训或重做摘要，从而避免灾难性遗忘。同义边 E′ 起到松散的实体标准化/去重作用，但论文未实现显式的合并/失效/冲突删除（开源代码后续才补充 document delete API）。
- **经验回放 (核心主题)**: 不适用（非智能体经验回放型方法）。HippoRAG 复用的是外部“知识”而非“智能体自身的过往轨迹/技能”：它把语料知识沉淀进 KG 并在每次查询时通过 PPR 联想式调用相关子图，属“知识记忆检索”而非“失败/成功经验复用”。与 ReasoningBank/Voyager 等以重用过去任务轨迹改进未来决策的范式正交——HippoRAG 是 RAG/知识整合方向，不维护任务轨迹回放缓冲，也不做技能复用。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric / in-context, retrieval-level）：所有组件（OpenIE 的 LLM、检索编码器、PPR）均开箱即用、无需任何额外训练；LLM 参数保持静态，知识更新完全在外部 KG 上通过增删边完成。作者将“对各组件做针对性微调”列为可提升实用性的未来方向，但本身不含梯度更新。
- **失败学习 (核心主题)**: 不适用 / 无失败学习机制。HippoRAG 不检测或记忆任务失败、不构建负例或错误规则；它属知识检索框架，不涉及对失败轨迹的自反思（与 Reflexion/Retroformer 等正交）。论文的“错误分析”（Appendix F）是研究者对系统检索误差来源（NER/OpenIE 错误、图搜索错误）的离线人工剖析，用于指导未来改进，而非系统在线从失败中学习。
- **技能/程序归纳**: 不归纳可复用技能/工作流/程序。HippoRAG 沉淀的是事实性知识图谱（语义记忆），不从经验中提炼程序性技能；与 Voyager/AWM/Synapse 等技能/工作流诱导方法属不同范畴。
- **在线 vs 离线**: 二者结合但以离线索引为主：离线阶段对整个语料库逐段做 OpenIE 构建 KG 索引（类比记忆编码）；在线阶段对每个查询做命名实体抽取+实体链接+PPR 检索（类比模式完成）。新知识可通过增量加边在线持续整合，但实验主要在静态语料上离线建图后在线检索。

**评测 / Evaluation**

- **任务领域**: 知识密集型问答（QA），尤以多跳问答（multi-hop QA）/知识整合任务为核心；并提出“path-finding（寻路型）多跳问答”这一现有检索器难以解决的新场景（如“哪位斯坦福教授研究阿尔茨海默症神经科学”）。不涉及网页导航、具身、游戏、对话或编码等智能体任务。
- **基准**: 三个多跳 QA 基准（各取 1000 题验证集并合并支撑/干扰段落构成检索语料）：MuSiQue（answerable）、2WikiMultiHopQA（2Wiki，实体中心、最契合 HippoRAG）、HotpotQA（被指多跳信号较弱、用于完整性对比）。检索指标 Recall@2/@5 与 All-Recall@2/@5（AR，全部支撑段落均被召回的比例）；QA 指标 EM 与 F1。另用 CaRB 框架对 OpenIE 质量做小规模内在评测（239 条 gold 三元组）。
- **报告增益**: 检索（单步，Table 2，ColBERTv2 骨干）平均 R@2/R@5：HippoRAG 57.4/72.9，显著超 ColBERTv2 基线 53.9/65.6；在 2Wiki 上 R@2 70.7 vs 59.2（+约11点）、R@5 89.1 vs 68.2（+约20点）；MuSiQue R@2 40.9 vs 37.9、R@5 51.9 vs 49.2（约+3点）；HotpotQA 与最强基线相当（R@5 77.7 vs 79.3）。All-Recall（Table 6）增益更大：2Wiki AR@5 75.7 vs 37.1（+约38点）、MuSiQue AR@5 22.4 vs 16.1。QA（Table 4，同一阅读器）平均 EM/F1：HippoRAG 35.9/48.1 vs ColBERTv2 30.8/42.5，相对无检索（24.6/35.5）大幅提升；F1 在 2Wiki 提升约 17 点、MuSiQue 约 3 点、HotpotQA 约 1 点。与迭代检索 IRCoT 相当或更优，但在线检索便宜 10–30 倍、快 6–13 倍（Appendix G）。HippoRAG 与 IRCoT 互补：IRCoT+HippoRAG 平均 R@5 升至 78.2、QA F1 升至 51.7（2Wiki R@5 93.9、F1 62.7），相对 IRCoT 再提升 R@5 约 4%（MuSiQue）/18%（2Wiki）/1%（HotpotQA）。摘要中“up to 20%”即指 2Wiki 上约 20 点检索增益。
- **对比基线**: 无检索（None）；单步检索基线 BM25、Contriever、GTR、ColBERTv2；LLM 增强检索基线 Propositionizer（重写为命题）与 RAPTOR（构建摘要节点）；多步/迭代检索基线 IRCoT。消融对照包括 OpenIE 替代（REBEL、Llama-3.1-8B/70B-Instruct）与 PPR 替代（仅查询节点、查询节点+邻居）。

**分析 / Analysis**

- **关键创新**: 首个把“LLM 做 OpenIE 建无模式知识图谱（人工海马索引）+ 个性化 PageRank 图扩散检索”协同起来、用单步检索完成多跳推理的神经生物学启发式 RAG 长期记忆框架：以海马索引理论的“模式分离/模式完成”为蓝本，无需训练即可一次性整合跨段落知识，并提出局部 IDF 式“节点特异性”改进检索；相对迭代检索（IRCoT）在保持或超越精度的同时在线检索便宜 10–30 倍、快 6–13 倍，并能处理现有方法不可及的 path-finding 寻路型多跳问题。
- **局限**: 作者自述：(1) 所有组件均开箱即用、未微调，错误分析（Appendix F）显示多数误差源自 NER 与 OpenIE，针对性微调有改进空间；(2) 图搜索仅用简单 PPR，可引入关系引导的遍历；(3) OpenIE 在长文档上一致性差于短文档；(4) 最关键——可扩展性有待验证，尚未在远超现有基准规模的索引上证明效率与效果。其它隐含局限：不做反思/总结、不学习记忆控制策略、无显式遗忘/冲突消解、无时间推理、仅文本单语义记忆、性能受底层 OpenIE 抽取质量与实体重叠度影响（HotpotQA 上仅与基线持平）。
- **与其他工作关系**: 属本研究 D 类“图/神经启发/生产级”记忆方向，与同簇关注图结构+生物启发的工作相关。受 Teyler & Discenna 海马索引理论与互补学习系统启发；明确区别于需反复摘要的 RAPTOR、MemWalker、GraphRAG（HippoRAG 仅靠加边即可持续整合，离线索引成本低于这些图法）。检索骨干用 ColBERTv2/Contriever，迭代检索对照 IRCoT 并与之互补。与 B 类以智能体经验/对话记忆为主（如 Generative Agents、MemoryBank、MemGPT、A-MEM）以及 A/C 类经验回放/技能诱导（Reflexion、ExpeL、Voyager、AWM、ReasoningBank）正交——后者复用“智能体自身轨迹/技能”，而 HippoRAG 复用“外部知识”。其个性化 PageRank 图扩散思路与后续 ExpGraph（C6）等“图扩散经验检索”工作相呼应（ExpGraph 把类似 PPR 思想用于经验图）。后续工作 HippoRAG 2（From RAG to Memory, ICML 2025, arXiv 2502.14802）在其基础上增强联想性与 sense-making，进一步推进“RAG→记忆”的连续学习。
- **可复现性**: 复现性强：代码与数据开源于 https://github.com/OSU-NLP-Group/HippoRAG（MIT 许可，约 3.6k stars，11 位贡献者，141 次提交），提供 pip 包 `hipporag`、Colab、OpenAI/vLLM/Azure/Bedrock 多后端 demo 与测试脚本，数据集托管于 HuggingFace（osunlp/HippoRAG）。所用基准 MuSiQue/2Wiki/HotpotQA 均为公开数据；NeurIPS 2024 经同行评审。社区采用信号高（高引用、高星标、后续 HippoRAG 2 持续维护）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式管线）：记忆的写入（OpenIE 加边）、检索（PPR + 节点特异性 + 实体链接）与更新（增量加边）全部基于固定启发式与开箱即用组件，不用 RL/训练去学习“何时/写什么/如何检索”的记忆管理策略。属 2025–26“学习记忆控制”代际之前的启发式范式（与 Memory-R1/Mem-α 形成对比）。
- **记忆主体**: 知识中心 / 文档中心（区别于用户中心与智能体中心）：记忆的是外部语料的世界知识/事实（供跨段落整合与多跳问答），既非记住用户偏好做个性化（Mem0/Zep/LongMemEval），也非记住智能体自身经验做自我改进（ReasoningBank/Voyager）。它是一个面向知识整合的长期记忆/检索层。
- **多智能体记忆**: 单智能体/单系统设定。HippoRAG 是面向单一 LLM 的外部知识记忆层，不涉及多智能体共享或跨智能体记忆路由（与 G-Memory、MIRIX 等正交）。
- **时序推理支持**: 无显式时间建模。不维护事实有效期窗口或事件时序日历（区别于 Zep/Graphiti）；KG 三元组与节点不带时间戳或有效性区间，无法做事实随时间失效的时序推理。
- **模态**: 纯文本（text-only）。知识图谱、三元组、查询实体与 passage 均为文本，无视觉/具身/多模态记忆。
- **冲突/矛盾处理**: 无专门的冲突/矛盾事实消解机制。新知识仅通过向 KG 增添三元组/边整合，相似实体由同义边 E′ 做松散标准化，但不显式检测或调和相互矛盾的事实（无 Memory-R1 式 UPDATE 或 MEMTRACK 式冲突跟踪）；矛盾三元组可能并存，由下游 PPR 排序与节点特异性间接调制其影响。
- **token成本/延迟证据**: 有明确效率证据（Appendix G）：单步 HippoRAG 在线检索相对迭代检索 IRCoT 便宜 10–30 倍、快 6–13 倍，同时达到相当或更优的检索/QA 表现——核心论点之一是“在线检索效率是面向终端用户服务时最重要的因素”。此外，相对 RAPTOR/GraphRAG/LightRAG 等基于摘要的图法，其离线索引仅靠增量加边、成本更低（HippoRAG 2 README 进一步强调此点）。论文未给出 p95 延迟或 token 数的具体百分比绝对值（不同于 Mem0/Zep/MemMachine 的精确 token/延迟百分比），以“倍率”形式报告。

**不确定字段 / Uncertain**

- 过度个性化/记忆安全风险 (`over_personalization_risk`)
- 存储后端 (`storage_backend`)


<a id="d2-hipporag-2神经生物学启发的大语言模型长期记忆框架论文标题from-rag-to-memory-non-parametric-continual-learning-for-large-language-models为-hipporag-的升级版别名-hipporag-v2"></a>

### D2 HippoRAG 2

*HippoRAG 2（神经生物学启发的大语言模型长期记忆框架；论文标题：From RAG to Memory: Non-Parametric Continual Learning for Large Language Models；为 HippoRAG 的升级版，别名 HippoRAG v2）*


**基本信息 / Provenance**

- **年份**: 2025年（arXiv 预印本首次公开于 2025-02-20，arXiv:2502.14802；正式发表于 ICML 2025）
- **作者/机构**: 第一作者 Bernal Jiménez Gutiérrez，合作者包括 Yiheng Shu、Weijian Qi、Sizhe Zhou，通讯作者 Yu Su，全部隶属美国俄亥俄州立大学自然语言处理组（The Ohio State University, OSU-NLP-Group）。
- **发表venue**: ICML 2025（第 42 届国际机器学习大会，Poster；正式收录于 Proceedings of Machine Learning Research PMLR 第 267 卷，pp. 21497-21515）；预印本为 arXiv 2025。
- **论文链接**: https://arxiv.org/abs/2502.14802 （ICML/PMLR 正式版：https://proceedings.mlr.press/v267/gutierrez25a.html ；OpenReview: https://openreview.net/forum?id=LWH8yn4HS2 ）
- **代码链接**: https://github.com/OSU-NLP-Group/HippoRAG （官方开源代码，MIT 协议，约 3.6k star、364 fork；可 pip install hipporag；配套 HuggingFace 数据集 osunlp/HippoRAG_2）。
- **引用数**: 约 151 次引用（Semantic Scholar 实时数据，与任务备注 ~151 一致），影响力快速上升、属图/神经启发 RAG 记忆方向高被引代表作。

**记忆分类 / Taxonomy**

- **记忆类型**: 以语义记忆（semantic）为主：把外部文档语料组织成开放式知识图谱（OpenIE 三元组）作为长期记忆/“人工海马索引”，对应人类长期记忆中的事实记忆、联想记忆与意义建构（sense-making）。检索过程中查询拼接的工作上下文构成工作记忆（working）；作者把全文语料的非参数化持续吸收类比为情景/语义记忆的连续整合，但不显式建模个体经历的情景记忆（论文结论指出未来可向长对话情景记忆扩展）。
- **记忆结构**: 知识图谱（KG）+ 稠密向量混合结构。核心是一个无 schema 的开放式 KG（“海马索引”）：节点包含 phrase 节点（OpenIE 抽取的实体/概念，视为稀疏编码 sparse coding）与 passage 节点（原始段落，视为稠密编码 dense coding）；边包含 relation edge（三元组关系边）、synonym edge（同义词边，按向量相似度阈值连接）、context edge（标注为 contains，连接段落与其衍生的所有 phrase）。在该图上运行 Personalized PageRank（PPR）做上下文感知检索。区别于 GraphRAG/RAPTOR/LightRAG，HippoRAG 2 的 KG 用于辅助检索而非扩充语料，引入更少 LLM 生成噪声。
- **存储后端**: 外部持久化存储：嵌入向量库（embedding_store，分别存储段落、实体/phrase、三元组/fact 的 embedding，默认编码器为 nvidia/NV-Embed-v2，也支持 GritLM-7B、Contriever）；图结构以本地图对象/邻接结构保存（PPR 在内存图上运行）；OpenIE 三元组与 KG 落盘为本地文件（outputs 目录）。LLM（Llama-3.3-70B-Instruct 或 GPT-4o-mini）通过 vLLM 本地服务或 OpenAI/Azure/Bedrock API 调用，不修改 LLM 参数。
- **持久化**: 外部持久化（durable external store）+ 非参数化。记忆完全存放在外部 KG 与向量库中，跨查询/跨语料增量持续，不写入 LLM 权重（non-parametric continual learning）；新知识通过离线索引把新段落抽取为三元组并入图、由编码器检测同义词把新旧知识互联来吸收。在线检索时的查询上下文为临时（ephemeral）。代码提供 delete API 与增量更新支持。

**核心机制 / Mechanisms**

- **写入/编码**: 采用两阶段离线索引把原始经验编码进记忆：1) OpenIE 三元组抽取——用 LLM（Llama-3.3-70B-Instruct）对每个段落做开放式信息抽取，无 schema 约束地生成 (主语, 关系, 宾语) 三元组，主语/宾语称为 phrase、连接边称为 relation edge，构成无 schema 的 KG/海马索引；2) 同义词检测——用检索编码器评估 KG 内 phrase 对，向量相似度超过预设阈值者加一条 synonym edge，从而跨段落连接同义概念、整合新旧知识；3) 稠密-稀疏整合（Dense-Sparse Integration，§3.2）——把 phrase 节点视为概念的稀疏编码，同时为每个段落新增 passage 节点（视为稠密编码以保留上下文），并以标注 contains 的 context edge 把段落连到其衍生的所有 phrase，使最终开放 KG 同时承载概念信息与上下文信息。该编码方式相比 HippoRAG 仅靠实体中心索引，显著减少了索引与推理阶段的上下文丢失。
- **检索机制**: 基于图的上下文感知检索，核心是 Personalized PageRank（PPR）。在线检索流程：1) 深度上下文化的查询链接（Query-to-Triple，§3.3）——不再用 HippoRAG 的 NER-to-Node（从查询抽实体再匹配节点），而是用编码器把整条查询直接匹配到图中的三元组（triples 封装了概念间基本上下文关系），并消融对比了 NER-to-node、Query-to-node、Query-to-triple 三种方案，Query-to-triple 平均 Recall@5 比 NER-to-node 高 12.5%，默认采用；2) 识别记忆过滤（Recognition Memory，§3.4）——把 query-to-triple 建模为“召回+识别”两步：先用编码器取 top-k（默认 top-5）三元组 T，再用 LLM（经 DSPy MIPROv2 优化提示）过滤出相关子集 T'⊆T 作为最终种子；3) 种子节点与重置概率——从过滤后三元组取最多 k 个 phrase 节点（按其在过滤三元组中的平均排名分），所有 passage 节点也作为种子；phrase 节点按排名分、passage 节点按嵌入相似度乘以权重因子（默认 0.05，§6.2 消融显示该因子关键）分配 PPR 重置概率，以平衡两类节点；4) 执行 PPR 在 KG 上扩散概率质量，按 PageRank 分数对段落排序，取 top-5 段落作为下游 QA 上下文。若无可用三元组，则直接退化为用编码器取 top 段落。
- **反思/巩固**: 不做传统意义上的“经验反思/抽象成更高层洞见”。其知识整合发生在离线索引阶段：通过 OpenIE 把段落原始文本巩固为结构化三元组、通过同义词检测把跨段落的同义概念互联、通过稠密-稀疏整合把概念与上下文在 KG 中交织，从而把分散知识组织成可多跳关联的长期记忆索引。与 RAPTOR/GraphRAG 用 LLM 生成层级摘要/社区摘要不同，HippoRAG 2 刻意不让 LLM 生成摘要扩充语料（避免引入噪声导致简单/多跳 QA 退化），而是用 KG 仅辅助检索。在线阶段的“整合”体现为 PPR 在图上做上下文感知扩散，把种子节点邻域的相关段落激活汇聚。触发时机：索引在新语料加入时触发；图扩散在每次查询时在线触发。
- **遗忘/更新**: 以增量添加（ADD）为主：新段落到来时离线抽取三元组并入图、由同义词检测把新旧知识互联，实现持续学习下语料增长的鲁棒吸收（§6.3 验证将 NQ/MuSiQue 分 4 段递增加入时，相对 NV-Embed-v2 的优势保持稳定）。代码层面提供 delete API 支持文档删除与图更新。论文未提出 Ebbinghaus 式衰减或显式记忆遗忘机制，也无显式冲突消解流程。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric / in-context）。这是论文核心定位——“从 RAG 到记忆的非参数化持续学习”：不更新 LLM 权重、不做梯度微调，全部通过外部 KG + 向量库 + PPR 在推理时引入与组织新知识，规避了持续微调的灾难性遗忘与模型编辑的局部性失效问题。LLM 仅作为冻结的“人工新皮层”用于 OpenIE 抽取与三元组过滤。
- **失败学习 (核心主题)**: 不涉及。HippoRAG 2 是用户/语料知识型的非参数化检索记忆框架，不检测或利用智能体自身的失败经验（无自我反思、失败模式记忆、负样本或错误驱动规则）。该维度（属智能体自我改进型工作的核心主题）对本工作不适用。
- **技能/程序归纳**: 不涉及。系统不从经验中归纳可复用的技能/工作流/程序；它归纳的是事实性知识结构（KG 三元组与同义词关联），而非可调用的过程性技能。
- **在线 vs 离线**: 两者兼有，但与“批量训练轨迹”无关：离线索引——批量对语料做 OpenIE 三元组抽取、同义词检测、构建开放 KG（含 vLLM offline 批处理模式，比在线服务快 3 倍以上）；在线检索——每次查询时做 query-to-triple 链接、识别记忆过滤、PPR 图扩散与 QA。记忆随部署期语料增长持续在线增量更新（持续学习设置）。

**评测 / Evaluation**

- **任务领域**: 三类知识密集型 QA 任务：1) 简单 QA / 事实记忆（单跳、实体中心）；2) 多跳 QA / 联想记忆（multi-hop、需跨段落推理）；3) 大规模篇章理解 / 意义建构（discourse understanding，需理解整本小说级长文）。属文本检索增强问答领域，非具身/游戏/GUI/编码。
- **基准**: 事实记忆：NaturalQuestions（NQ，1000 查询）、PopQA（1000 查询）；联想/多跳记忆：MuSiQue、2WikiMultihopQA（2Wiki）、HotpotQA（各 1000 查询）、LV-Eval（hotpotwikiqa-mixup 256k，124 查询，含关键词/短语替换以减少知识泄漏）；意义建构/篇章理解：NarrativeQA（10 篇长文、293 查询）。检索用 passage recall@5、QA 用基于 token 的 F1（沿用 MuSiQue 评测）。
- **报告增益**: 以 Llama-3.3-70B-Instruct 作 QA reader、NV-Embed-v2 作检索器的设置下：QA F1（表 2，7 数据集平均）HippoRAG 2 达 59.8，为全场最高，优于最强基线 NV-Embed-v2 的 57.0、HippoRAG 的 53.1、GraphRAG 49.6、RAPTOR 48.8、LightRAG 仅 6.6；分项中 HippoRAG 2 在 NQ 63.3†、MuSiQue 48.6†、HotpotQA 75.5、LV-Eval 12.9†、NarrativeQA 25.9 均为最佳（† 表 bootstrap 检验 p<0.05 显著优于 NV-Embed-v2），在 2Wiki 71.0† 仅次于 HippoRAG 71.8。摘要强调相对标准 RAG 在联想任务平均提升约 7 个百分点（F1），同时在事实记忆与意义建构任务无退化甚至略升；具体 HippoRAG 2 在 2Wiki 上比 NV-Embed-v2 高 9.5% F1、在 LV-Eval 上高 3.1%。检索 Recall@5（表 3，5 数据集平均）HippoRAG 2 达 78.2，显著高于 NV-Embed-v2 的 73.4；其中 MuSiQue 74.7（+5.0%）、2Wiki 90.4（+13.9%）、HotpotQA 96.3、NQ 78.0 均为最佳。消融（表 4）：Query-to-triple 相对 NER-to-node 平均 Recall@5 +12.5%（87.1 vs 74.6），去掉 passage 节点降至 81.0，去掉 LLM 过滤降至 86.4。鲁棒性（表 7）：换不同检索器 HippoRAG 2 一致超过直接稠密检索（如 NV-Embed-v2 上 MuSiQue 74.7 vs 69.7、GritLM-7B 71.6 vs 66.0、GTE-Qwen2 68.8 vs 63.6）。论文未报告精确的 token/延迟数字，但 README 与附录 F 称其在线过程成本与延迟高效、离线索引资源显著少于 GraphRAG/RAPTOR/LightRAG。
- **对比基线**: 三类基线对比：1) 简单基线——BM25、Contriever、GTR(T5-base)；2) 大型嵌入模型（7B）——GTE-Qwen2-7B-Instruct、GritLM-7B、nvidia/NV-Embed-v2（NV-Embed-v2 为主要对照的最强稠密检索基线）；3) 结构增强 RAG——RAPTOR（层级摘要）、GraphRAG（社区摘要）、LightRAG（双层 KG 检索）、HippoRAG（KG+PPR 前身）。还包含“无检索”（仅 reader 参数化知识）对照。所有结构增强基线均用与 HippoRAG 2 相同的 LLM（Llama-3.3-70B-Instruct）和检索器（NV-Embed-v2）复现以公平比较。

**分析 / Analysis**

- **关键创新**: 在前作 HippoRAG（KG+PPR）基础上，通过三项关键改进解决“结构增强 RAG 在简单事实任务上反而退化”的痛点，首次让一个图/神经启发的 RAG 系统在事实记忆、意义建构、联想记忆三类任务上全面超越最强标准 RAG：1) 稠密-稀疏整合——把 passage 节点（稠密上下文编码）与 phrase 节点（稀疏概念编码）以 contains 边无缝整合进同一 KG；2) 更深的上下文化——用 Query-to-Triple 取代实体中心的 NER-to-Node 链接；3) 识别记忆——用 LLM 在线过滤检索三元组以改进 PPR 种子节点选择。由此把 RAG 推近人类长期记忆，开辟大模型非参数化持续学习路径。
- **局限**: 1) 仍以语义/事实记忆为主，缺乏对长对话情景记忆的支持，论文结论明确指出未来需用图检索增强情景记忆能力；2) 复杂联想任务在持续学习语料增长时性能仍会随信息增多而退化（与 NV-Embed-v2 同速下滑，§6.3），说明对真正持续学习的鲁棒性有限；3) 依赖较强 LLM（Llama-3.3-70B-Instruct）做 OpenIE 与三元组过滤，离线索引需较大算力（虽少于 GraphRAG 等）；4) 重置概率权重因子（默认 0.05）等超参对 PPR 结果敏感、需调；5) 无显式遗忘/事实失效/冲突消解机制；6) 评测仅限文本知识型 QA，未覆盖具身、GUI、编码等智能体任务，也不涉及记忆安全/隐私治理。
- **与其他工作关系**: 直接扩展并改进本研究 D 类前身 HippoRAG（Gutiérrez et al., NeurIPS 2024，arXiv:2405.14831）：HippoRAG 首创用 PPR 在 OpenIE 构建的开放 KG 上做多跳检索（实体中心、NER-to-Node 链接），但在简单 QA 与篇章理解上退化；HippoRAG 2 通过加入 passage 节点（稠密-稀疏整合）、Query-to-Triple 链接、识别记忆 LLM 过滤三项改进修复退化，可表述为“在 HippoRAG 上增加段落集成与更深查询上下文化”。与同为图/摘要型的 GraphRAG（Edge et al. 2024）、RAPTOR（Sarthi et al. 2024）、LightRAG（Guo et al. 2024）对比——后三者用 LLM 生成摘要扩充语料引入噪声、在单跳/多跳 QA 上退化，而 HippoRAG 2 的 KG 仅辅助检索、噪声更小。借用 Haveliwala (2002) 的 Personalized PageRank 与 NV-Embed-v2/GritLM/GTR/Contriever 等检索编码器。在本研究记忆体系中属“图/神经启发/生产级”聚类，与 B 类 EM-LLM、MemGPT 等智能体记忆形成对照（HippoRAG 2 为非参数化外部知识检索记忆，非智能体自我经验记忆）。
- **可复现性**: 可复现性强：官方开源（GitHub OSU-NLP-Group/HippoRAG，MIT 协议，约 3.6k star、364 fork、11 名贡献者、141 次提交），提供 pip 安装包 hipporag、Colab、OpenAI/Azure/Bedrock/本地 vLLM 多种部署、完整 reproduce 脚本（main.py）、自定义数据集格式说明与 OpenIE 中间结果。配套 HuggingFace 数据集 osunlp/HippoRAG_2 公开全部采样基准；所用 NQ/PopQA/MuSiQue/2Wiki/HotpotQA/LV-Eval/NarrativeQA 均为公开数据集。ICML 2025 收录、约 151 引用，社区采纳度高。需自部署 70B 级 LLM（或用 OpenAI API），算力门槛中等。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否。HippoRAG 2 采用启发式管线管理记忆（固定的 OpenIE 抽取→同义词检测→Query-to-Triple→识别记忆过滤→PPR 流程），未用 RL/训练去学习“何时/存什么/如何检索更新”的记忆管理策略本身；其唯一的学习成分是用 DSPy MIPROv2 优化器自动调优三元组过滤的提示（prompt 优化），并非对记忆控制策略的端到端训练。属 2025-26“学习型记忆控制”代际划分中的非学习型/启发式管线一侧。
- **记忆主体**: 用户/语料知识型（偏 user-centric/knowledge-centric）：记忆对象是外部文档语料中的事实与概念关联，目标是让 LLM 持续吸收并多跳检索新外部知识来正确回答问题，而非记住智能体自身的行动经验做自我改进（区别于 ReasoningBank/Voyager 的 agent-centric）。但它也不专注个体用户画像个性化（不同于 Mem0/Zep 的长对话用户记忆），更准确说是面向通用知识语料的非参数化持续学习记忆。
- **多智能体记忆**: 单智能体/单系统框架。不涉及多智能体共享或路由记忆；记忆是单一全局开放 KG，由单个检索-QA 管线读写，无跨智能体的洞见/查询/交互分层。
- **时序推理支持**: 不显式支持时间推理。KG 三元组为无 schema 的事实关系，不建模事实有效期窗口、事件排序或时间日历（区别于 Zep/Graphiti）。仅在持续学习实验中模拟语料随时间递增加入，但不对时间有效性做显式建模；论文将长对话/时序情景记忆列为未来工作。
- **模态**: 纯文本（text-only）。所有语料、KG 三元组、查询与 QA 均为文本，无视觉/多模态/具身记忆。
- **过度个性化/记忆安全风险**: 未涉及。该工作不处理记忆安全、过度个性化、有害/过时/谄媚记忆或隐私治理等负面维度（属外部知识检索记忆而非用户个性化场景），仅在 Impact Statement 中泛泛指出未发现超出一般 LLM 与信息检索系统的特别社会风险。
- **冲突/矛盾处理**: 无专门的冲突/矛盾事实消解或合并机制。新知识以增量添加并入 KG，同义词检测仅做概念互联而不裁决矛盾事实；在线阶段靠识别记忆（LLM 过滤无关三元组）与 PPR 上下文扩散间接抑制不相关信息，但不存在显式的 UPDATE/合并矛盾事实操作（区别于 Memory-R1/MEMTRACK）。

**不确定字段 / Uncertain**

- 经验回放 (核心主题) (`experience_replay`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="d3-zep--graphitizep-为面向-ai-智能体的记忆层服务graphiti-为其核心的时序感知动态知识图谱引擎亦作开源框架单独发布论文标题zep-a-temporal-knowledge-graph-architecture-for-agent-memory"></a>

### D3 Zep / Graphiti

*Zep / Graphiti（Zep 为面向 AI 智能体的记忆层服务；Graphiti 为其核心的时序感知动态知识图谱引擎，亦作开源框架单独发布；论文标题：Zep: A Temporal Knowledge Graph Architecture for Agent Memory）*


**基本信息 / Provenance**

- **年份**: 2025年（arXiv 预印本首次公开于 2025-01-20，arXiv:2501.13956；Graphiti 开源仓库早于论文，于 2024-08 创建）。
- **作者/机构**: 全部来自 Zep AI（商业公司 Zep Software, Inc.）：Preston Rasmussen、Pavlo Paliychuk、Travis Beauvais、Jack Ryan、Daniel Chalef（CEO/创始人）。属工业界团队而非学术机构。
- **发表venue**: 工业界 / 开源系统。论文以 arXiv 预印本形式发布（2025，arXiv:2501.13956），无正式会议/期刊收录记录；Zep 为商业化记忆层服务，Graphiti 为 Apache-2.0 开源框架。
- **论文链接**: https://arxiv.org/abs/2501.13956 （DOI: 10.48550/arXiv.2501.13956）。
- **代码链接**: https://github.com/getzep/graphiti （官方开源核心引擎 Graphiti，Apache-2.0，Python，约 27.2k star、约 2.7k fork、50+ 贡献者、195+ release；PyPI 包名 graphiti-core，pip install graphiti-core，需 Python>=3.10；内置 MCP server 供 Claude/Cursor 等使用）。Zep 商业服务官网 https://www.getzep.com 。
- **引用数**: 约 197 次引用（Semantic Scholar 实时数据，与任务备注一致），在 2025 年生产级智能体记忆/知识图谱方向属高被引代表作，工业影响力显著（Graphiti 仓库约 27.2k star）。

**记忆分类 / Taxonomy**

- **记忆类型**: 情景记忆（episodic）+ 语义记忆（semantic）双重显式建模，并以社区子图提供更高层抽象。检索时拼接进上下文的 facts/entities 构成工作记忆（working）。论文明确指出：episodic 子图（episode 节点存原始消息/文本/JSON 的无损数据）对应人类情景记忆中的离散事件，semantic 子图（实体节点与事实边）对应概念间关联的语义记忆，这种双存储刻意镜像人类记忆的心理学模型。不涉及程序性（procedural）技能记忆。
- **记忆结构**: 时序感知的动态知识图谱（temporally-aware dynamic KG），形式化为 G=(N,E,φ)，含三层层级子图：1) 情景子图 G_e（episode 节点，存原始消息/文本/JSON 的无损记录，episodic 边连接 episode 到其抽取出的实体）；2) 语义实体子图 G_s（entity 节点为从 episode 抽取并消歧后的实体，semantic/entity 边为实体间关系即“事实 fact”，可通过同一事实在不同实体间多次抽取实现超边 hyper-edge 建模多实体复杂事实）；3) 社区子图 G_c（community 节点为强连通实体簇，含高层摘要，借鉴 GraphRAG 但用标签传播算法 label propagation 而非 Leiden 以支持动态增量扩展）。整体层级为 episodes→facts→entities→communities。区别于静态 KG/常规 GraphRAG 的核心是双时序（bi-temporal）建模与边失效机制。
- **存储后端**: 外部持久化图数据库为主：实验与默认实现使用 Neo4j（节点/边存储，并借 Neo4j 内置的 Lucene 实现余弦语义相似度搜索 φ_cos 与 Okapi BM25 全文搜索 φ_bm25）；开源 Graphiti 现亦支持 FalkorDB、Kuzu、Amazon Neptune 等多种图后端。实体名嵌入为 1024 维向量空间做余弦近邻检索。LLM 经 API 调用（实验中图构建用 gpt-4o-mini-2024-07-18，对话生成用 gpt-4o-mini 与 gpt-4o-2024-11-20，并为与 MemGPT 对齐另用 gpt-4-turbo-2024-04-09），嵌入与重排用 BAAI 的 BGE-m3 模型。图写入用预定义 Cypher 查询（而非 LLM 生成查询）以保证 schema 一致、减少幻觉。不修改 LLM 权重。
- **持久化**: 外部持久化（durable external store）+ 非参数化。所有记忆存放于外部时序知识图谱（Neo4j 等图数据库）中，跨会话、跨数据源持久存在并随对话演进动态增量更新；不写入 LLM 权重。借助 episodic 子图的无损存储，语义产物可双向回溯到源 episode（用于引用/溯源）。作为生产系统，Zep 强调记忆检索的准确性、低延迟与可扩展性。

**核心机制 / Mechanisms**

- **写入/编码**: 采用增量式、时序感知的图构建管线把原始经验编码进知识图谱：1) Episode 摄入——原始数据（message/text/JSON）作为 episode 节点无损存储，每条 message 带参考时间戳 t_ref；2) 实体抽取——结合当前消息与前 n=4 条消息（约两轮对话）做命名实体识别，自动把说话者抽为首个实体，并借鉴 Reflexion 的反思（reflection）技术二次校验以减少幻觉、提升覆盖；同时抽取实体摘要便于后续消歧与检索；3) 实体消歧/解析——把实体名嵌入 1024 维向量做余弦近邻 + 全文检索找候选节点，再用 LLM（实体解析提示）判定是否重复，重复则合并并生成更完整的名称与摘要；4) 事实抽取——抽取实体间关系作为事实边（关系类型用全大写简洁谓词如 LOVES/WORKS_FOR，并附更详细的事实描述），事实嵌入后做边去重（去重搜索仅限同一实体对之间的边，既防错误合并又降复杂度）；5) 时序抽取（见 forgetting_update）；6) 入图用预定义 Cypher 查询写入。所有抽取产物与源 episode 维护双向索引以保证无损与可溯源。
- **检索机制**: 记忆检索是一个函数 f:S→S，把文本查询 α 映射为格式化上下文字符串 β，由三步复合 f(α)=χ(ρ(φ(α)))：1) 搜索 φ——用三种互补方法在事实边、实体节点、社区节点上召回候选：余弦语义相似度搜索 φ_cos（捕捉语义相似，基于 1024 维嵌入与 Neo4j/Lucene）、Okapi BM25 全文搜索 φ_bm25（捕捉词面相似）、广度优先搜索 φ_bfs（在 n 跳内捕捉图上的上下文相似，可用最近 episode 作为种子节点引入近期提及的实体/关系，这是 RAG 领域较少用的图遍历检索，借鉴 AriGraph/Distill-SynthKG）；搜索字段：事实边搜 fact 字段、实体节点搜 name、社区节点搜 community name；2) 重排 ρ——支持 Reciprocal Rank Fusion（RRF）、Maximal Marginal Relevance（MMR），并自研基于图的 episode-mentions 重排器（按实体/事实在对话中被提及频率排序，使高频信息更易被取回）、node-distance 重排器（按与指定中心节点的图距离排序做局部化）、以及成本最高但最精的 cross-encoder（LLM 交叉注意力打分）重排；3) 构造 χ——把选中的边/节点转为文本：事实边返回 fact 及其有效期 t_valid/t_invalid，实体节点返回 name+summary，社区节点返回 summary。实验中取 top-10～top-20 最相关边与节点拼成约 1.6k token 的上下文。
- **反思/巩固**: 在写入与社区构建两处体现“原始→更高层知识”的整合：1) 实体抽取阶段借鉴 Reflexion 的反思（reflection）技术对初次抽取结果二次校验，减少幻觉并提升实体覆盖；2) 实体消歧/事实去重把跨 episode 的同一实体/事实归并为统一节点/边，并生成更完整的名称与摘要（增量巩固）；3) 社区子图——通过社区检测（标签传播算法）把强连通实体聚成社区，并用迭代式 map-reduce 摘要生成社区高层摘要与社区名（借鉴 GraphRAG），形成对整个图结构更全局、互联的概览。社区采用动态单步标签传播增量更新（新节点归入邻居中占多数的社区并更新摘要），辅以周期性完整刷新——这是显著降低延迟与 LLM 推理成本的实用启发式。触发时机：实体/事实整合在每次 episode 摄入时在线触发；社区动态扩展在新实体加入时触发，完整刷新周期性触发。
- **遗忘/更新**: 核心机制为基于双时序的边失效（temporal edge invalidation）而非物理删除：系统对每条事实边追踪四个时间戳——t'_created/t'_expired∈T'（事务时间线 T'，记录事实在系统中被创建/失效的时间，用于数据库审计）与 t_valid/t_invalid∈T（事件时间线 T，记录事实在现实中成立的真实区间）。新边引入时，LLM 把新边与语义相关的现有边比对以识别矛盾；当发现时间上重叠的矛盾时，把被取代的旧边的 t_invalid 置为使其失效的新边的 t_valid，从而使旧事实“过期”但仍以历史记录形式保留（非无损丢弃）。沿事务时间线 T' 一贯优先采纳更新的信息。事实/实体去重亦在写入时合并冗余。该 bi-temporal 边失效是 Graphiti 区别于其它 KG 引擎的关键差异化特性。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric / in-context）。不更新 LLM 权重、不做梯度训练；全部通过外部时序知识图谱的构建、维护与推理时检索来引入与组织新知识。LLM 仅作为冻结组件用于实体/事实抽取、消歧、时序抽取与边失效判断、社区摘要及最终回答生成。论文将来自微调 GraphRAG 抽取模型（如 Triplex/Distill-SynthKG）列为可降低成本/延迟的未来方向，但本工作本身不含参数学习。
- **失败学习 (核心主题)**: 不涉及。Zep/Graphiti 是用户/对话知识型的非参数化记忆系统，不检测或利用智能体自身的失败经验（无失败轨迹自反思、失败模式记忆、负样本或错误驱动规则）。论文中借鉴的 Reflexion 仅被用作减少实体抽取幻觉的“反思校验”技术，与从任务失败中学习无关。该维度（属智能体自我改进型工作的核心主题）对本工作不适用。
- **技能/程序归纳**: 不涉及。系统不从经验中归纳可复用的技能/工作流/程序；它构建并维护的是事实性/语义性的时序知识结构（实体、带有效期的事实边、社区摘要），而非可调用的过程性技能。
- **在线 vs 离线**: 以在线（online）增量为主：随对话/业务数据流式到来，逐 episode 在线完成实体与事实抽取、消歧去重、时序抽取与边失效、以及社区的动态增量扩展，使图持续演进——契合其生产部署定位。离线成分仅为周期性的社区完整刷新（重跑标签传播以纠正动态扩展的漂移）。不存在“对训练轨迹语料的批量离线学习”。

**评测 / Evaluation**

- **任务领域**: 面向企业级的长期对话记忆与动态知识整合：多会话对话记忆（multi-session dialogue / 聊天助手长期记忆）、跨会话信息综合、长期上下文维持、时序推理、知识更新等。论文定位为企业应用（如客户体验、对话+结构化业务数据融合），非具身/游戏/GUI/编码领域。
- **基准**: 两个长期记忆基准：1) Deep Memory Retrieval（DMR，来自 MemGPT，500 段多会话对话、每段 5 个会话每会话至多 12 条消息、含一对问答）；2) LongMemEval（具体用 LongMemEval_s 子集，对话平均约 115,000 token，含 single-session-user / single-session-assistant / single-session-preference / multi-session / knowledge-update / temporal-reasoning 六类问题）。
- **报告增益**: DMR（表 1）：Zep 用 gpt-4-turbo 达 94.8% 准确率，超过 MemGPT 的 93.4%（提升约 1.4 个百分点），亦略高于全对话上下文基线 94.4%；用 gpt-4o-mini Zep 达 98.2%，略高于全对话基线 98.0%。但作者自评 DMR 规模小（每段仅 60 条消息可完全装入上下文窗口）、问题为单轮事实检索且措辞含糊，不足以区分记忆系统。LongMemEval_s（表 2）：相比 115k token 全上下文基线，Zep 用 gpt-4o-mini 达 63.8%（vs 基线 55.4%，绝对 +8.4pp，作者表述为约 +15.2% 相对提升），用 gpt-4o 达 71.2%（vs 基线 60.2%，绝对 +11.0pp，即标题所称 +18.5% 相对提升）；同时延迟从约 28.9–31.3s 降至约 2.58–3.20s（降低约 90%），平均上下文 token 从 115k 降至仅 1.6k。分项（表 3）增益最大者为 single-session-preference（gpt-4o：20.0%→56.7%，相对 +184%）、temporal-reasoning（gpt-4o：45.1%→62.4%，+38.4%）、multi-session（gpt-4o：44.3%→57.9%，+30.7%）、knowledge-update（gpt-4o：78.2%→83.3%）；唯一明显退化为 single-session-assistant（gpt-4o：94.6%→80.4%，-17.7%；gpt-4o-mini -9.06%）。无法用 LongMemEval 复现 MemGPT 结果（其框架不支持直接摄入既有消息历史）。
- **对比基线**: DMR 上对比：MemGPT（93.4%，前 SOTA）、递归摘要 Recursive Summarization（35.3%）、会话摘要 Conversation Summaries（gpt-4-turbo 78.6% / gpt-4o-mini 88.0%）、全对话上下文 Full-conversation（gpt-4-turbo 94.4% / gpt-4o-mini 98.0%）。LongMemEval 上主要对比 115k token 全上下文基线（Full-context，gpt-4o-mini 55.4% / gpt-4o 60.2%）；尝试对比 MemGPT 但因其不支持摄入既有历史而未能成功。整体即与 无记忆/全上下文、摘要式记忆、以及前 SOTA 记忆系统 MemGPT 对比。

**分析 / Analysis**

- **关键创新**: 首个面向生产、以双时序感知动态知识图谱（Graphiti）作为智能体记忆层的系统：核心创新是 bi-temporal 模型（同时追踪事件真实有效区间 T 与系统事务时间 T'）配合 LLM 驱动的边失效机制，能在不丢失历史的前提下动态融合非结构化对话与结构化业务数据、显式管理事实随时间的演变与矛盾；并把 episodic/semantic/community 三层子图与向量+BM25+图遍历混合检索结合，在保持 SOTA 准确率的同时把延迟降约 90%、上下文 token 从 115k 压到约 1.6k，证明图式记忆可在生产规模低延迟运行。
- **局限**: 1) DMR 基准本身被作者批评规模小、为单轮事实检索、措辞含糊，难以充分区分记忆系统，故 DMR 上对 MemGPT 的优势（+1.4pp）意义有限；2) single-session-assistant 类问题上 Zep 明显退化（gpt-4o -17.7%），即对“助手自身先前发言”的记忆较弱；3) 较弱模型（gpt-4o-mini）对 Zep 的时序数据理解不足，knowledge-update 等类别未受益甚至略降；4) 图构建依赖多次 LLM 调用（抽取/消歧/时序/边失效/社区摘要），存在抽取幻觉与成本，作者建议未来用微调抽取模型缓解；5) 社区动态扩展会逐渐偏离完整标签传播结果，需周期性完整刷新；6) 缺乏正式本体（ontology），论文将领域本体列为未来方向；7) 评测仅限文本对话记忆，未覆盖 Zep 宣称的“对话+结构化业务数据综合”能力（无合适基准），也未做传统 RAG 基准对比；8) 网络架构引入额外延迟（实验从波士顿连 AWS us-west-2）。
- **与其他工作关系**: 属本研究 D 类“图/神经启发/生产级”聚类。其双子图（episodic+semantic）设计借鉴 AriGraph（Anokhin et al. 2024，KG 世界模型+情景记忆）；社区节点与 map-reduce 摘要借鉴 GraphRAG（Edge et al. 2024，但用标签传播替代 Leiden 以支持动态增量）；社区高层关键词检索方法与 LightRAG（Guo et al. 2024）并行独立提出并被认为可融合；广度优先图检索借鉴 AriGraph 与 Distill-SynthKG；实体抽取的反思校验借鉴 Reflexion（A 类 Shinn et al. 2023）；混合检索沿用 RRF/MMR/cross-encoder 等成熟 IR 技术。直接对标并超越 MemGPT（B3，Packer et al.，分层记忆/LLM-as-OS）——Zep 用时序 KG 取代 MemGPT 的分层缓冲。与同处 D 类的 HippoRAG/HippoRAG 2（OpenIE 开放 KG + PPR 的外部知识检索记忆）相比：Zep 同为图式记忆但聚焦动态对话/用户记忆并显式建模时间有效性与边失效，而 HippoRAG 系列聚焦静态文档语料的多跳事实检索且不建模时间；二者均非智能体自我经验记忆。Zep 亦常与 Mem0（用户中心记忆层）并列作为生产级记忆系统对比对象。
- **可复现性**: 可复现性较强且社区采纳度高：核心引擎 Graphiti 已开源（GitHub getzep/graphiti，Apache-2.0，Python，约 27.2k star、约 2.7k fork、50+ 贡献者、195+ release，PyPI 包 graphiti-core，pip 可装，支持 Neo4j/FalkorDB/Kuzu/Neptune 多后端，内置 MCP server），实验所用 DMR、LongMemEval、Multi-Session Chat 均为公开数据集，论文承诺通过 GitHub 公开实验 notebook 与提示词并在附录给出全部图构建提示。Zep 本身为闭源商业托管服务（提供 SLA、Dashboard、Python/TypeScript/Go SDK，宣称生产规模 sub-200ms 检索）。需自备 LLM/嵌入 API（gpt-4o-mini、BGE-m3 等）与图数据库，部署门槛中等。约 197 引用，工业影响力大。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否。Zep/Graphiti 采用启发式/规则化的记忆管理管线（固定流程：抽取→消歧→事实去重→时序抽取→LLM 边失效判定→社区标签传播→混合检索→重排），未用 RL/端到端训练去学习“何时存什么、何时更新/失效、如何检索”的记忆控制策略本身。其中 LLM 用于若干局部判断（实体是否重复、事实是否矛盾失效），但属推理时的提示驱动判定而非可训练的记忆控制策略。属 2025-26“学习型记忆控制”代际划分中的非学习型/启发式管线一侧。
- **记忆主体**: 用户/对话中心（user-centric）：记忆对象是用户与智能体的对话历史及相关业务/世界数据，目标是让智能体跨会话记住用户信息、偏好与事实演变以提供个性化、连贯的长期服务（与 Mem0/LongMemEval 同类）；并非记住智能体自身的行动经验做自我改进（区别于 ReasoningBank/Voyager 的 agent-centric）。Zep 在生产中按 per-user/per-entity 管理大量独立上下文图。
- **时序推理支持**: 强支持，且为本系统最核心的差异化能力。Graphiti 实现 bi-temporal 双时序模型：事件时间线 T（事实在现实中成立的真实区间，由 t_valid/t_invalid 标注）与事务时间线 T'（事实在系统中被创建/失效的时间，由 t'_created/t'_expired 标注），能解析绝对时间戳（如出生日期）与相对时间表达（如“两周前”“下周四”，基于消息参考时间戳 t_ref 推算）；并通过 LLM 比对+边失效机制显式管理事实随时间的有效与失效、保留历史关系演变。在 LongMemEval 的 temporal-reasoning 类问题上 Zep 取得显著提升（gpt-4o 45.1%→62.4%）。属同类系统中对时间建模最显式、最完整者之一（与 Graphiti 同义）。
- **模态**: 纯文本（text-only）。论文实验聚焦 message 类型 episode 的对话记忆；Graphiti 摄入支持 message/text/JSON（结构化业务数据），但均为文本/结构化文本，不涉及视觉/多模态/具身记忆。
- **冲突/矛盾处理**: 有显式的矛盾处理机制（与单纯遗忘不同）：新事实边引入时，系统用 LLM 把新边与语义相关的现有边比对以识别潜在矛盾；当发现时间上重叠的矛盾事实时，把被取代旧边的 t_invalid 置为新边的 t_valid，从而让旧事实“失效但保留历史”，并沿事务时间线 T' 一贯优先采纳更新信息。这相当于面向时序事实的 UPDATE/失效式冲突消解（概念上接近 Memory-R1 的 UPDATE、MEMTRACK 的矛盾追踪），且能区分“当前有效事实”与“历史失效事实”而非简单覆盖。
- **token成本/延迟证据**: 有明确的效率量化证据：在 LongMemEval_s 上，相比 115k token 的全上下文基线，Zep 把平均上下文 token 从 115k 压缩到约 1.6k（约 -98.6%），同时把响应延迟从约 28.9–31.3s 降到约 2.58–3.20s（降低约 90%，且延迟 IQR 同步收窄），并在准确率更高的前提下实现（gpt-4o：71.2% vs 60.2%）。社区动态增量更新（替代每次完整标签传播）被指为显著降低延迟与 LLM 推理成本的实用启发式。Zep 商业服务另宣称生产规模检索 sub-200ms。这是论文相对全上下文记忆方案最突出的效率卖点之一（与 Mem0 的 -90% token、MemMachine 的 -80% 输入 token 同属该代际效率主张）。

**不确定字段 / Uncertain**

- 经验回放 (核心主题) (`experience_replay`)
- 多智能体记忆 (`multi_agent_memory`)
- 过度个性化/记忆安全风险 (`over_personalization_risk`)


<a id="d4-mem0--mem0gmem0面向生产可扩展长期记忆层mem0g-为其图记忆增强变体发音-mem-zero"></a>

### D4 Mem0 / Mem0^g

*Mem0 / Mem0^g（Mem0：面向生产、可扩展长期记忆层；Mem0^g 为其图记忆增强变体，发音 mem-zero）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本 2025-04-28 首次公开，arXiv:2504.19413；Semantic Scholar 将其归入 ECAI 2025 / European Conference on Artificial Intelligence，但本质为业界开源团队的 arXiv 论文）
- **作者/机构**: Prateek Chhikara（第一作者）、Dev Khant、Saket Aryan、Taranjeet Singh、Deshraj Yadav。作者均来自 Mem0（Mem0.ai，YC S24 创业公司），通讯邮箱 research@mem0.ai；第一作者 Prateek Chhikara 同时关联美国南加州大学（University of Southern California, USC）。属业界/开源团队工作。
- **发表venue**: arXiv 预印本（cs.AI / cs.CL，2025）；Semantic Scholar BibTeX 标注为 European Conference on Artificial Intelligence (ECAI 2025)。整体定位为业界/开源系统（industry/open-source），论文为其技术报告。
- **论文链接**: https://arxiv.org/abs/2504.19413
- **代码链接**: https://github.com/mem0ai/mem0 （Apache-2.0 许可；约 58k stars、6.6k forks、324 位贡献者、2200+ commits；pip 包 mem0ai、npm 包 mem0ai；评测代码见仓库 evaluation/ 目录与 mem0ai/memory-benchmarks。论文中代码链接写为 https://mem0.ai/research）

**记忆分类 / Taxonomy**

- **记忆类型**: 以语义记忆（semantic）为主、兼具情景记忆（episodic）属性：Mem0 从多轮多会话对话中动态抽取“显著事实/用户偏好”（如用户是素食者、不吃乳制品）作为自然语言长期记忆；Mem0^g 进一步以实体-关系三元组（带创建时间戳）建图。属用户中心的事实/偏好型长期记忆，对应 CoALA 的语义记忆范畴；不实现程序性技能记忆，工作记忆即 LLM 自身上下文窗口。
- **记忆结构**: 双形态：(1) Mem0——扁平的“自然语言记忆条目”集合，存于向量库（稠密嵌入索引）；(2) Mem0^g——有向带标签知识图谱 G=(V,E,L)，节点 V 为实体（含实体类型、嵌入向量 e_v、创建时间戳 t_v），边 E 为关系三元组 (v_s, r, v_d)，标签 L 赋予节点语义类型。区别于原始缓冲与纯摘要法，是“显著事实条目 + 可选关系图”的混合结构。
- **存储后端**: 外部持久化存储：Mem0 用稠密向量数据库做相似度检索；Mem0^g 用 Neo4j 图数据库存储实体节点与关系三元组并保留时间戳元数据。LLM 推理引擎为 GPT-4o-mini（抽取/更新/工具调用），嵌入用 OpenAI text-embedding-3-small。LLM 仅作处理器，不存储记忆。（注：仓库 2026 年 4 月新版算法已转向 Qdrant + 多信号检索 + Qwen 嵌入，但与本论文版本不同。）
- **持久化**: 外部持久化记忆（durable external store）：记忆独立于 LLM，跨会话、跨天/周/月持续存在，解决固定上下文窗口“情境溢出即遗忘”的问题。属非参数化外部记忆（LLM 参数静态冻结），通过对外部库/图的增删改实现增量演进。

**核心机制 / Mechanisms**

- **写入/编码**: 两阶段“抽取-更新”增量管线（incremental processing）。抽取阶段：每来一对新消息 (m_{t-1}, m_t)，构造提示 P=(S, {m_{t-m},...,m_{t-2}}, m_{t-1}, m_t)，其中 S 是由异步摘要模块周期性刷新的全局对话摘要、{...} 是近 m=10 条消息的近期上下文窗口；由 LLM 实现的抽取函数 φ(P) 从该交换中提炼出一组显著记忆 Ω={ω_1,...,ω_n}（自然语言事实/偏好），形成候选事实。Mem0^g 的抽取为两段式：先由实体抽取器识别实体及其类型（人物、地点、事件、属性等），再由关系生成器在实体对间推断带标签关系三元组（如 lives_in、prefers、owns、happened_on），并为源/目标实体计算嵌入、附加创建时间戳。整体把对话历史压缩为“紧凑结构化表征”而非保存逐字轨迹或大块原文。
- **检索机制**: Mem0：查询时用稠密嵌入在向量库中做语义相似度检索，返回最相关的少量记忆条目作为上下文（论文 Mem0 平均仅占约 7k tokens/对话，故检索 token 远低于全上下文 26k）。Mem0^g 采用双路检索：(1) 实体中心法——先在查询中识别关键实体，经语义相似度链接到图中锚节点，再系统遍历其入边与出边构造相关子图；(2) 语义三元组法——将整条查询编码为稠密向量，与图中每条关系三元组的文本编码计算细粒度相似度，返回超过可配置阈值并按相似度降序排列的三元组。两路结合兼顾“定向实体型问题”与“宽泛概念型问题”。检索延迟极低：Mem0 搜索 p50=0.148s、p95=0.200s（全方法最低）；Mem0^g 搜索 p50=0.476s。
- **反思/巩固**: 有显式整合/巩固（consolidation）但非长篇反思总结。两条路径：(1) 异步摘要模块周期性刷新全局对话摘要 S，作为抽取时的全局主题上下文（独立于主管线、不引入延迟）；(2) 更新阶段对每条候选事实 ω_i 先检索 top-s=10 条语义相似的既有记忆，再经“工具调用（tool call）”由 LLM 自身推理在 ADD / UPDATE / DELETE / NOOP 四种操作中择一，从而把分散信息巩固/去冗、保持知识库一致与时间一致性。这是“原始对话 → 精炼显著事实/关系”的抽象，但不产生 Reflexion 式的经验性反思洞见。
- **遗忘/更新**: 无 Ebbinghaus 时间衰减；以 LLM 决定的离散记忆管理操作实现更新与去冗：ADD（无等价记忆时新建）、UPDATE（用互补信息增补既有记忆）、DELETE（删除被新信息矛盾的记忆）、NOOP（无需改动）。Mem0^g 不物理删除而是把过时关系“标记为失效（invalid）”以保留时间推理能力，并有冲突检测+LLM 更新解析器（update resolver）。不直接用 LLM 分类器而借其推理选择操作。（仓库 2026 版新算法改为“单遍仅 ADD、不做 UPDATE/DELETE”的累积式，与本论文不同。）
- **经验回放 (核心主题)**: 不适用（非智能体经验回放型方法）。Mem0 复用的是“用户信息/对话事实”（用户中心记忆），而非智能体自身过往任务轨迹/技能。它不维护成败轨迹回放缓冲、不做范例提示或技能复用，目标是跨会话保持对用户偏好与既述事实的连贯性，与 ReasoningBank/Voyager 等“以重用自身经验改进未来决策”的范式正交。其“复用”体现为：把历史对话沉淀为紧凑记忆并在后续会话相关查询时检索回用，避免重复提问、避免与先前事实矛盾。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric / in-context, prompt-level）：所有组件（GPT-4o-mini 做抽取/更新/工具调用、嵌入模型、向量/图检索）均开箱即用，无任何梯度更新；LLM 参数保持冻结，学习完全发生在外部记忆库/图上（增删改条目与边）。属启发式管线驱动的非参数化记忆。
- **失败学习 (核心主题)**: 不适用 / 无失败学习机制。Mem0 不检测任务失败、不构建负例或错误规则、不对失败轨迹做自反思（与 Reflexion/Retroformer/ExpeL 正交）。其“纠错”仅限于事实层面的一致性维护——当新信息与既有记忆矛盾时通过 DELETE/UPDATE 操作（Mem0^g 通过冲突检测与失效标记）消解，但这是事实冲突消解，而非从“任务执行失败”中学习改进策略。
- **技能/程序归纳**: 不归纳可复用技能/工作流/程序。Mem0 沉淀的是用户事实与偏好（语义记忆）及实体关系（Mem0^g 图），不从经验中提炼程序性技能或可复用工作流（与 Voyager/AWM/Synapse 等技能诱导方法属不同范畴）。
- **在线 vs 离线**: 在线（online）为主、增量式：记忆在部署/对话过程中按消息对实时构建——每来一对新消息即触发抽取与更新，即用即存（Mem0^g 图构建在最坏情况下也可在 1 分钟内完成，新增记忆可立即用于回答，作者以此对比 Zep 需数小时异步处理的缺陷）。无需对轨迹语料做离线批量训练。

**评测 / Evaluation**

- **任务领域**: 长期多会话对话记忆（multi-session dialogue / 个性化会话 QA）。面向客服机器人、AI 助手、医疗/教育/企业支持等需跨会话保持连贯与用户画像的场景。不评测网页导航、具身、游戏、编码等智能体任务（作者把过程性推理与多模态列为未来方向）。
- **基准**: LOCOMO（Maharana et al., 2024）：10 段超长对话，平均每段约 600 轮对话、约 26000 tokens、跨多会话，每段约 200 个带标准答案的问题，分为 single-hop / multi-hop / temporal / open-domain 四类（原 adversarial 类因无标准答案而被排除）。指标：F1、BLEU-1（B1）与 LLM-as-a-Judge（J，跑 10 次独立实验取均值±1 标准差），外加部署指标 Token 消耗（tiktoken cl100k_base）与延迟（搜索 p50/p95、总响应 p50/p95）。（注：仓库后续在 LongMemEval、BEAM 上也有更新数据，但非本论文。）
- **报告增益**: 摘要级头条：Mem0 在 LLM-as-a-Judge（J）上相对 OpenAI 记忆方案取得 26% 相对提升；Mem0^g 总体得分比 Mem0 高约 2%；相对全上下文（full-context）方案 p95 延迟降低 91%、token 成本节省 >90%。结论级：相对各题型最佳基线，single-hop / temporal / multi-hop 分别取得约 5% / 11% / 7% 相对提升。
表1（按题型 J 分）：single-hop Mem0=67.13（>OpenAI 63.79、Zep 61.70、LangMem 62.23；A-Mem* 仅 39.79）；multi-hop Mem0=51.15（>OpenAI 42.92、LangMem 47.92、Zep 41.35）；open-domain Mem0^g=75.71、Mem0=72.93（略低于 Zep 76.60）；temporal Mem0^g=58.13、Mem0=55.51（远超 OpenAI 21.71、LangMem 23.43）。F1 上 temporal Mem0^g=51.55、Mem0=48.93 为最佳。
表2（整段 LOCOMO 总体 J 与延迟）：Mem0 总体 J=66.88%、Mem0^g=68.44%（全方法最高，仅次于计算昂贵的 full-context 72.90%）；最佳 RAG 仅约 60–61%，故 Mem0 约 +10% 相对、Mem0^g 约 +12% 相对优于最强 RAG。延迟：Mem0 搜索 p50=0.148s/p95=0.200s、总 p50=0.708s/p95=1.440s（相对 full-context 总 p95 17.117s 降约 92%）；Mem0^g 搜索 p50=0.476s、总 p50=1.091s/p95=2.590s（相对 full-context 降约 85%）。Token 占用：Mem0 约 7k/对话、Mem0^g 约 14k，而 Zep 图记忆 >600k tokens（约 20 倍于全原文 26k），凸显 Mem0 记忆压缩与构建速度优势。
- **对比基线**: 六大类基线：(1) 既有 LOCOMO 记忆方法——LoCoMo、ReadAgent、MemoryBank、MemGPT、A-Mem；(2) 开源记忆方案——LangMem（Hot Path）；(3) RAG——分块 128–8192 tokens、k∈{1,2}；(4) 全上下文 full-context（约 26k tokens 整段入窗）；(5) 专有模型——OpenAI ChatGPT 记忆功能（gpt-4o-mini）；(6) 记忆平台/提供商——Zep。所有可控实验温度设为 0 以求可复现。

**分析 / Analysis**

- **关键创新**: 提出面向生产部署、可扩展的“记忆中心”架构：用两阶段“抽取-更新”管线，由 LLM 经函数调用（tool call）自主在 ADD/UPDATE/DELETE/NOOP 间决策，将多会话对话动态压缩为紧凑的自然语言显著事实记忆（Mem0），并提供图增强变体 Mem0^g（Neo4j 实体-关系三元组+失效标记的时间推理）。核心价值在于在接近全上下文质量的同时，把 p95 延迟降 ~91%、token 成本降 >90%，且记忆占用仅为 Zep 图记忆的约 1/85（7k vs 600k+ tokens），构建快至分钟级——一个真正可在生产规模低延迟运行的长期记忆层，并形成 58k 星的高采用开源实现。
- **局限**: 作者自述/可见局限：(1) Mem0^g 图操作带来额外延迟开销（未来需优化）；(2) 全上下文方案仍有约 73% 的略高 J（Mem0/Mem0^g 用质量微降换大幅效率提升）；(3) 仅在对话场景验证，未覆盖过程性推理与多模态（列为未来工作）；(4) 仅评测 LOCOMO 单一基准（约 26k tokens 量级），更大规模可扩展性证据有限。其它：图记忆在 multi-hop 上反而不及纯自然语言记忆（存在冗余/开销）；记忆质量依赖 GPT-4o-mini 抽取与提示工程；为启发式管线、未学习记忆控制策略；无真正的时间衰减遗忘；论文版本与仓库后续大改的算法/后端存在差异（复现需对齐版本）。
- **与其他工作关系**: 属本研究 D 类“图/神经启发/生产级”记忆方向，是其中最具生产落地与采用度的代表（用户中心、可扩展记忆层）。直接对比并超越同类记忆系统：MemGPT（B3，分层记忆/虚拟上下文）、A-Mem（B4，Agentic Memory，被本文作为 LOCOMO 基线 A-Mem*）、MemoryBank（B2）、Think-in-Memory 等；与 Zep/Graphiti（时间知识图谱记忆，本文将 Zep 作为最强商业基线，在 open-domain 上略胜 Mem0、但 token 占用高 ~85 倍）形成核心对照。Mem0^g 的实体-关系图与时间戳/失效标记思路同 D 类图记忆（HippoRAG、Zep）一脉但更轻量。与 A/C 类经验回放/技能诱导（Reflexion、ExpeL、Voyager、AWM、ReasoningBank）正交——后者复用智能体自身轨迹/技能，Mem0 复用用户事实。后续大量工作（Memori、MemMachine、ENGRAM、AuroraMem 等）引用并在其基础上扩展，本文已成为生产级记忆层的事实基准之一。
- **可复现性**: 复现性强、社区采用度极高：代码 Apache-2.0 开源于 github.com/mem0ai/mem0（约 58k stars、6.6k forks、324 贡献者、2200+ commits），提供 pip(mem0ai)/npm 包、自托管 Docker、托管云平台与 CLI；评测脚本与数据见仓库 evaluation/ 目录及独立的 mem0ai/memory-benchmarks，基准 LOCOMO 为公开数据集。论文报告多次独立运行均值±标准差、温度 0。需注意：论文所述算法（GPT-4o-mini、向量库/Neo4j、ADD/UPDATE/DELETE/NOOP）与仓库 2026 年 4 月后的新算法（单遍 ADD-only、Qdrant、多信号检索）已不同，逐字复现论文数字需锁定对应历史版本。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式管线）：记忆管理虽由 LLM“推理”在 ADD/UPDATE/DELETE/NOOP 中选择操作（而非固定规则分类器），但这是基于提示与语义关系的零样本推理，并未用 RL/训练去学习“何时/写什么/如何检索/如何更新”的记忆管理策略本身。属 2025–26“学习型记忆控制”代际之前的启发式 LLM 驱动范式（与 Memory-R1、Mem-α 等用 RL 学习记忆策略的工作形成对比）。
- **记忆主体**: 用户中心（user-centric）：记忆用户的偏好、属性与既述事实以实现个性化、跨会话连贯（典型场景：素食/忌乳制品偏好需跨会话保持）。与 Mem0/Zep/LongMemEval 同属“记住用户信息做个性化”的阵营，区别于 ReasoningBank/Voyager 等“记住智能体自身经验做自我改进”的智能体中心记忆。
- **多智能体记忆**: 以单智能体/单助手为主：提供 User、Session、Agent 多级记忆（multi-level memory，仓库特性），可保留 agent 自身状态，但并非 G-Memory/MIRIX 式的多智能体共享/路由记忆架构（无跨智能体洞见/查询/交互分层路由）。论文实验为单系统对话记忆设定。
- **时序推理支持**: 显式支持时间推理（这是其相对其他记忆系统的强项之一）：Mem0^g 为每个实体节点附加创建时间戳 t_v，更新时不物理删除过时关系而“标记为失效（invalid）”以保留时间推理能力。在 LOCOMO temporal 题型上 Mem0^g（J=58.13、F1=51.55）大幅领先 OpenAI（21.71）等基线，作者强调结构化关系图对捕捉事件时序/顺序/持续时间尤为有效。但其时间建模不及 Zep/Graphiti 的双时间(bi-temporal)有效期窗口精细。
- **模态**: 纯文本（text-only）。记忆条目、实体、关系三元组与对话均为文本，无视觉/具身/多模态记忆（作者将多模态交互列为未来工作）。
- **冲突/矛盾处理**: 有专门的冲突/矛盾处理：Mem0 在更新阶段，当候选事实与既有记忆矛盾时由 LLM 选择 DELETE（删除被矛盾的旧记忆）或 UPDATE（增补/修正），以维持知识库一致与时间一致性；Mem0^g 配有冲突检测机制与“LLM 更新解析器（update resolver）”，识别潜在冲突关系并将过时关系标记为失效（而非物理删除）以兼顾一致性与时间推理。区别于单纯遗忘，是显式的语义级矛盾消解（与 Memory-R1 UPDATE、MEMTRACK 冲突跟踪思路相近但为启发式 LLM 驱动）。
- **token成本/延迟证据**: 效率证据充分且为本文核心卖点：相对全上下文（full-context）方案，Mem0 p95 总延迟降约 91–92%（1.440s vs 17.117s）、token 成本节省 >90%；Mem0^g 总 p95 降约 85%（2.590s）。搜索延迟 Mem0 p50=0.148s/p95=0.200s 为全方法最低。记忆 token 占用：Mem0 约 7k/对话、Mem0^g 约 14k，而 Zep 图记忆 >600k tokens（约 20 倍于全原文 26k、约 85 倍于 Mem0），且 Mem0 图构建分钟级 vs Zep 需数小时异步处理。是同类中 token/延迟效率对比最明确（与 Zep -90% 延迟、MemMachine -80% 输入 token 同属高效记忆层阵营）。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)
- 过度个性化/记忆安全风险 (`over_personalization_risk`)


<a id="d5-g-memorygraph-based-agentic-memory-for-llm-based-multi-agent-systems面向多智能体系统的图式分层记忆受组织记忆理论启发由-insightqueryinteraction-三层图构成的即插即用记忆模块"></a>

### D5 G-Memory

*G-Memory（Graph-based Agentic Memory for LLM-based Multi-Agent Systems；面向多智能体系统的图式分层记忆，受组织记忆理论启发，由 insight/query/interaction 三层图构成的即插即用记忆模块）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本 2506.07398 于 2025-06-09 首次公开 v1）
- **作者/机构**: Guibin Zhang（张贵彬，新加坡国立大学 NUS，与 Muxin Fu 并列第一作者）、Muxin Fu（同济大学 Tongji）、Guancheng Wan（加州大学洛杉矶分校 UCLA）、Miao Yu（新加坡科技研究局 A*STAR）、Kun Wang（南洋理工大学 NTU，通讯作者，wang.kun@ntu.edu.sg）、Shuicheng Yan（颜水成，新加坡国立大学 NUS，通讯作者，yansc@comp.nus.edu.sg）。第一作者 Guibin Zhang 是多智能体系统拓扑优化（AgentPrune、GPTSwarm 相关）方向的活跃研究者。
- **发表venue**: arXiv 预印本（卷 abs/2506.07398，DOI 10.48550/arXiv.2506.07398，DBLP journals/corr/abs-2506-07398，CorpusId 279250852，标注 Preprint）。截至调研日尚无正式会议/期刊收录记录；属带开源代码的学术系统。
- **论文链接**: https://arxiv.org/abs/2506.07398
- **代码链接**: https://github.com/bingreeky/GMemory（官方实现，约 241 stars、33 forks、2 位贡献者，截至 2026-06；语言以 SAS/PDDL/Python 为主，含 ALFWorld/PDDL/FEVER/SciWorld 任务环境与 AutoGen/DyLAN/MacNet 三套 MAS 集成、各记忆基线实现；2026 年仍有更新如新增 SciWorld 环境）。
- **引用数**: 约 62 次引用（Semantic Scholar，CorpusId 279250852，截至调研日 2026-06）；发布约一年即被高频引用，是 2025 年多智能体记忆方向的代表性高关注度工作。

**记忆分类 / Taxonomy**

- **记忆类型**: 以程序性与语义记忆为主、含情景成分的跨试验（cross-trial）协作记忆。三层图分别承载不同 CoALA 类别：交互图（Interaction/Utterance Graph）保存逐字的智能体多轮对话轨迹，偏情景记忆（具体协作 episode）；查询图（Query Graph）保存历史任务查询、其成败状态与拓扑关联，偏情景/索引层；洞见图（Insight Graph）保存从经验抽象出的可泛化经验法则（如分工、任务分解、失败教训），偏语义/程序性记忆。整体面向「跨任务积累的集体协作经验」，区别于单纯的事实/用户画像记忆。
- **记忆结构**: 三层分层图（three-tier hierarchical graph）结构，是其核心创新。①交互图 G_inter=⟨U,E_inter⟩：节点为语义话语 u_i=(A_i,m_i)（发言智能体 A_i + 文本内容 m_i），有向边按时序「u_j 传递并启发 u_k」连接。②查询图 G_query=(Q,E_n)：节点 q_i=(Q_i,ψ_i,G_inter^(Qi)) 含原始查询、任务状态 ψ∈{Failed,Resolved} 及其交互图，边编码查询间语义关系（一查询轨迹对另一查询有指导价值）。③洞见图 G_insight=(I,E_i)：节点为蒸馏出的洞见 ι_k=(κ_k,Ω_k)（洞见内容 + 支撑查询集 Ω_k），超边 (ι_m,ι_n,q_j) 表示洞见 m 通过查询 j 语境化洞见 n。三层通过 query↔insight、query↔interaction 的上/下连接耦合，是一种 note-graph/知识图层级化变体。
- **存储后端**: 外部结构化图存储（自实现的内存中图对象 + 嵌入索引），非商用图数据库（未用 Neo4j）。查询的相似检索用句向量嵌入：embedding 函数 v(·) 采用 ALL-MiniLM-L6-v2（论文 Eq.4，引用 MiniLM）。代码以 Python 实现三层图与检索/更新逻辑，骨干 LLM 经 OpenAI API（gpt-4o-mini）或本地 Ollama（Qwen-2.5-7b/14b）调用。无向量数据库（FAISS/Chroma）或专用图数据库依赖。
- **持久化**: 外部持久化（durable external store）+ 完全非参数化。三层图作为外部记忆跨任务/跨试验持续存在并随每次任务完成而增长演化（institutionalization of group knowledge）；不修改任何模型权重，骨干 LLM 全程冻结。检索到的记忆在求解某查询时被注入各智能体的内部记忆状态 Mem_i（接近 in-context 注入），但底层三层图是持久外部存储。作为「即插即用（plug-and-play）」模块嵌入既有 MAS，不改动原框架。

**核心机制 / Mechanisms**

- **写入/编码**: 写入由 LLM 驱动、agentic 地在三层逐级完成（任务完成后触发，4.3 节）。①交互级：G-Memory 追踪每个智能体在本次任务中的全部话语，按时序连接构造该查询的交互图 G_inter 并存储——这是逐字（verbatim）的协作轨迹，而非摘要。②查询级：实例化新查询节点 q_new=(Q,ψ,G_inter)（含成败状态 ψ）并加入查询图，与语义相关的历史查询建边。③洞见级：用一个 LLM 摘要函数 J(G_inter,ψ) 从本次（成功或失败的）交互轨迹中蒸馏出可能的新洞见 ι_new，并把它与本次实际被使用过的旧洞见 I^S 经超边 (ι_k,ι_new,q_new) 结构化关联；同时把 q_new 追加进被使用洞见的支撑查询集 Ω_k（Eq.10-11）。整体是「原始轨迹逐字存于底层 + 经验抽象为洞见存于顶层」的双粒度编码。
- **检索机制**: 新查询到来时执行「先粗检索、再双向遍历」的多粒度检索（4.1-4.2 节，任务开始时触发）。①粗粒度检索（Eq.4）：在查询图上用 MiniLM 嵌入做相似度检索，取 top-k 个语义最相似历史查询得到草图集 Q^S（k∈{1,2}）。②拓扑跳扩展（Eq.5）：用查询图的 1-hop 邻居扩充 Q^S（1-hop 最优，2/3-hop 反而引入噪声降性能），这是「超越纯嵌入相似度」的关键——利用图拓扑捕捉表面不相似但有指导价值的查询。③双向遍历（Eq.6-8）：向上遍历 query→insight，用 query-to-insight 投影器 Π（取支撑查询集与 Q^S 相交的洞见节点）取回高层泛化洞见 I^S；向下遍历 query→interaction，用 LLM 相关性打分 R_LLM 选 top-M（M∈{2,3,4,5}）个最相关历史查询，再用 LLM 图稀疏器 S_LLM 把其交互图压缩为只保留核心对话元素的稀疏子图。④角色化注入（Eq.8）：算子 Φ 依据每个智能体的角色 Role_i 与当前任务 Q 评估各洞见/稀疏交互子图的效用，过滤后初始化该智能体的记忆状态 Mem_i，实现「角色定制化记忆」。论文未采用 recency·importance·relevance 三因子打分公式，而是图拓扑 + LLM 评分。
- **反思/巩固**: 存在显著的「反思/抽象/巩固」机制，是其相对原始轨迹堆叠的关键优势。①任务完成后，LLM 摘要函数 J(G_inter,ψ) 把整段（成功或失败的）多智能体协作轨迹蒸馏为可泛化洞见（如分工策略、任务分解、失败教训），写入洞见图——这是从「raw 协作 episode → 高层经验法则」的抽象。②检索阶段用 LLM 图稀疏器 S_LLM 把冗长交互轨迹压缩（trajectory condensation）为只含核心协作要素的稀疏子图，缓解信息过载（MAS 轨迹比单智能体长达 10× token）。③洞见图通过超边把新洞见与既有洞见结构化关联，形成可跨任务/跨类别迁移的知识网络（案例研究显示同类任务洞见密集互连、跨类任务间出现可迁移连接）。触发时机为「每次任务完成时更新三层、每次新查询时检索抽象」，类似 ExpeL/Generative Agents 的反思但落到图结构与多智能体协作粒度。
- **遗忘/更新**: 更新为「持续增量同化（assimilation）」式：每次任务完成后三层图联合更新——新增交互图、新增查询节点与语义边、蒸馏并关联新洞见、刷新被用洞见的支撑查询集（Eq.10-11）。被使用过的旧洞见会被「更新版本」替换（I\I_ret 后并入更新后的洞见）。但论文未实现真正的遗忘/衰减（无 Ebbinghaus 曲线，与其 MemoryBank 基线相对）、未提供系统化的显式删除/合并去重/失效（invalidate）算子或冲突消解流程，记忆随任务单调增长，属相对弱项。
- **经验回放 (核心主题)**: 强经验复用，且是 agent-centric（智能体自身协作经验）而非 user-centric——这是其核心主题。复用通过双向遍历实现：对新任务，向下取回与之最相关的历史协作轨迹稀疏子图（编码了过去成功/失败的协作模式与推理路径），向上取回从历史经验蒸馏的可泛化洞见，二者作为「可执行指导」（分工、任务分解、失败教训）注入当前 MAS 各智能体。案例：ALFWorld 中对新任务「put a clean cloth in countertop」检索到高度类似的历史查询「put a clean egg in microwave」及其中「先放入微波炉再清洗」的失败轨迹片段，从而指导当前任务避免同样错误。这是典型的「跨试验经验回放 + 范例式协作轨迹复用」，使 agent team 随任务暴露增多而逐步自我进化（self-evolving MAS）。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric）/ 提示层 + 外部图记忆。不做任何梯度更新，纯靠对三层外部图的 agentic 读写、抽象与角色化注入实现 MAS 的自我进化；完全依赖冻结的骨干 LLM（gpt-4o-mini / Qwen-2.5-7b/14b）。属运行时（inference-time）跨试验有状态记忆，而非训练式学习。
- **失败学习 (核心主题)**: 明确且系统地「从失败中学习」，是其设计要点之一。①查询图节点显式记录任务成败状态 ψ∈{Failed,Resolved}，失败轨迹与成功轨迹一同被保存与索引。②洞见蒸馏函数 J(G_inter,ψ) 同时从成功与失败的协作轨迹中提炼经验——失败轨迹被抽象为「失败教训（lessons from past failures）」类洞见。③检索时既会取回成功协作模式也会取回失败教训，作为对当前 MAS 的告诫式指导。案例：HotpotQA 网页搜索中取回的洞见警告「不要错误介入/不要基于同名人物错误作答」，正是从既往失败中提炼的负面经验，帮助当前智能体避免重蹈覆辙。这与 A 簇（Reflexion/ExpeL/CLIN）的失败反思一脉相承，但落到多智能体协作轨迹与图结构上。
- **技能/程序归纳**: 部分/中等。G-Memory 从协作经验中归纳出的洞见包含可复用的「程序性/策略性」知识——如分工方式（division of labor）、任务分解（task decomposition）模式与具体协作步骤，这些经洞见图跨任务迁移并在新任务中作为指导被调用。但它归纳的是「多智能体协作策略/经验法则」而非 Voyager 式可执行的独立技能/代码模块库；程序性知识以自然语言洞见 + 稀疏交互轨迹的形式表示并经角色化注入调用，而非显式的命名技能 API。
- **在线 vs 离线**: 在线（online）/ 增量跨试验为主。记忆在部署、逐任务求解过程中实时增量构建与演化：每完成一个查询即触发三层图联合更新，agent team 随任务序列（如 ALFWorld 多次 trial）逐步进化，论文展示随 trial 增多成功率曲线上升且达更高性能上限。不依赖离线批量训练语料；属典型的跨任务在线持续学习。

**评测 / Evaluation**

- **任务领域**: 三大域、五个基准：①具身行动（embodied action）——ALFWorld（文本家务环境，成功率）、SciWorld/ScienceWorld（交互式科学实验环境，进度率）；②知识推理（knowledge reasoning / 多跳问答与事实核查）——HotpotQA（多跳 QA，精确匹配，含网页搜索工具）、FEVER（事实验证，精确匹配，含网页搜索 API）；③博弈/游戏（game）——PDDL（来自 AgentBoard 的策略博弈，用 PDDL 表达式完成任务，进度率）。
- **基准**: ALFWorld（成功率）、SciWorld/ScienceWorld（进度率）、PDDL（来自 AgentBoard，进度率）、HotpotQA（精确匹配准确率）、FEVER（精确匹配准确率），共五个基准。评测覆盖三种 LLM 骨干（gpt-4o-mini、Qwen-2.5-7b、Qwen-2.5-14b）× 三种 MAS 框架（AutoGen、DyLAN、MacNet）。
- **报告增益**: 核心结论：作为即插即用模块嵌入 SOTA MAS，具身行动成功率最高提升 20.89%、知识 QA 准确率最高提升 10.12%，且 token 开销持平或更低。具体：①最大单点增益——Qwen-2.5-14b 骨干下，G-Memory 把 MacNet 在 ALFWorld 的性能从 58.21% 提升到 79.10%（+20.89%，Table 3）。②平均增益——Qwen-2.5-7b 下，集成 AutoGen / MacNet 时分别较最佳单/多智能体记忆基线平均高 6.8% / 5.5%（Table 2）。③gpt-4o-mini 骨干（Table 1，五基准均值 Avg）：AutoGen+G-Memory 等显著优于 No-memory（AutoGen 无记忆 Avg 48.32、最佳单智能体基线 Voyager Avg 53.52；G-Memory 为各表最佳/次佳）。④敏感性峰值（Qwen-2.5-14b+AutoGen，1-hop）：ALFWorld 85.82%、PDDL 55.24%。⑤成本：PDDL+AutoGen 上较 No-memory 提升 10.32% 而仅增加约 1.4×10^6 token，远低于 MetaGPT-M 等基线的额外开销；消融显示去掉高层洞见或细粒度交互任一部分，AutoGen 平均降 4.47%、DyLAN 降 3.82%（仅保留细粒度交互时）。⑥对比基线之失败案例：Voyager/MemoryBank 反而使 AutoGen 在 PDDL 上分别降 4.17%/1.34%，ChatDev-M 使 MacNet+SciWorld 降 2.32%，凸显 G-Memory 角色化记忆的必要性。
- **对比基线**: 横向对比覆盖：①无记忆（No-memory）；②单智能体记忆移植到 MAS——Voyager、MemoryBank（含 Ebbinghaus 遗忘曲线）、Generative Agents（含反思层）；③多智能体记忆实现——MetaGPT-M（仅试验内记忆）、ChatDev-M（试验内 + 简单跨试验，仅存历史解）、MacNet-M（仅存上一轮最终答案，丢弃协作轨迹）。所有基线在 AutoGen/DyLAN/MacNet 三框架 × 三 LLM 骨干上统一复现并与 G-Memory 比较。

**分析 / Analysis**

- **关键创新**: 首个面向多智能体系统（MAS）、受组织记忆理论启发的分层图式 agentic 记忆：用 insight / query / interaction 三层图同时捕捉「细粒度智能体间协作轨迹」与「跨试验、角色定制化的泛化洞见」，并通过查询图拓扑 + 双向遍历（向上取洞见、向下取压缩协作子图）实现多粒度、角色化记忆支持。它填补了「现有 MAS 记忆过简（无视协作轨迹、缺跨试验定制）」与「单智能体记忆难以直接迁移到长达 10× token 的 MAS 轨迹」之间的空白，且作为即插即用模块在不改动 AutoGen/DyLAN/MacNet 原框架的前提下显著提升其自我进化能力。
- **局限**: ①评测域有限——仅三域五基准，作者自承缺乏更多样任务（如医疗 QA）验证；②无真正遗忘/衰减、缺系统化显式删除/合并去重/失效与冲突消解，记忆随任务单调增长，长期可能累积过时或矛盾洞见；③性能对超参敏感——hop 扩展须为 1-hop、检索查询数 k 须取 {1,2}，更大值会因引入无关噪声而显著降性能（如 k=5 时 ALFWorld+AutoGen 降 7.71%）；④依赖骨干 LLM 做洞见蒸馏、轨迹稀疏与相关性打分，质量与成本受 LLM 能力制约；⑤洞见/轨迹以自然语言形式注入，非显式可执行技能，规模化下检索与注入开销随记忆增长；⑥安全面——作者指出若骨干模型被对抗操纵，记忆机制可能放大错误推理，呼吁部署时加入持续校验与对抗鲁棒性检查；⑦尚未正式同行评审发表。
- **与其他工作关系**: 属「D. 图式/神经启发/生产化」簇，并与 B9 MIRIX 同为多智能体记忆（multi-agent memory）的代表性工作（二者常被并列）。技术谱系：受 Walsh & Ungson(1991) 组织记忆理论启发；记忆图层级化思路与 D 簇图式记忆（HippoRAG 系列、A-MEM 的 Zettelkasten 笔记图）相通，但 G-Memory 的图建模对象是「多智能体协作轨迹/查询/洞见」而非知识三元组或用户事实。与单智能体跨试验记忆 ExpeL（C/A 簇，README 致谢其 prompt 设计）、Voyager（C1，技能/经验驱动自进化）、Generative Agents（B1，反思 + 观察记忆）一脉相承并将它们移植为基线对比；相对 Mem0/A-Mem/MemInsight 等强调抽象摘要的单智能体记忆，G-Memory 聚焦 MAS 协作粒度。与 MAS 框架/拓扑优化工作（AutoGen、MetaGPT、ChatDev、MacNet、DyLAN、AFlow、GPTSwarm、AgentPrune）正交互补——后者优化「拓扑/工作流」，G-Memory 提供跨试验「记忆/经验」层并即插即用嵌入其中。与第一作者 Guibin Zhang 的 MAS 拓扑工作（AgentPrune 等）形成记忆-拓扑的互补线。与 A 簇 ReasoningBank、Retroformer、CLIN 同属 agent-centric 自进化记忆，但 G-Memory 专攻多智能体协作经验而非单智能体。
- **可复现性**: 可复现性较好：官方完整开源（github.com/bingreeky/GMemory，约 241 stars、33 forks），提供 conda 环境、requirements、run_mas.sh 与命令行入口，集成 AutoGen/DyLAN/MacNet 三套 MAS 及 Empty/ChatDev/MetaGPT/Voyager/Generative/MemoryBank/G-Memory 七种记忆，并附 ALFWorld/PDDL/FEVER/SciWorld 任务环境与 Qwen7B 等实验日志；所用数据集（ALFWorld、ScienceWorld、PDDL/AgentBoard、HotpotQA、FEVER）均公开且许可明确。仓库 2026 年仍活跃维护（新增 SciWorld 环境、更新配置）。局限：贡献者较少（2 人）、部分实验依赖闭源 gpt-4o-mini、结果对超参与 LLM 版本敏感，且尚未经同行评审。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式管线 + LLM 辅助，但非学习型策略）。记忆管理策略由人工设定规则（1-hop 扩展、检索 k∈{1,2}、向下选 top-M∈{2,3,4,5}、MiniLM 相似度粗检索、任务完成即更新三层）+ LLM 在提示引导下执行洞见蒸馏 J、相关性打分 R_LLM、图稀疏 S_LLM 与角色化注入 Φ 共同构成；不使用 RL/训练去学习「何时存/取/更新」的记忆管理策略本身。因此处于 2025-26「学习型记忆控制」分水岭的启发式/提示驱动一侧（与 Memory-R1、Mem-α 等用 RL 学习记忆策略的方法相对），但其图遍历与多智能体角色化注入为学习型控制提供了清晰的动作空间雏形。
- **记忆主体**: 智能体中心（agent-centric）。G-Memory 记忆的是多智能体系统自身的协作经验（交互轨迹、查询成败、协作洞见）以实现 MAS 的自我进化与协作能力提升，而非记住用户个人信息做个性化（区别于 B9 MIRIX、Mem0、Zep 等 user-centric 系统）。其目标、机制（三层协作经验图、跨试验复用、失败教训蒸馏）与评测（ALFWorld/PDDL/HotpotQA 等任务求解成功率）都围绕「智能体团队从经验中自我改进」展开。
- **多智能体记忆**: 多智能体（multi-agent）记忆——这是其核心定位与主要标签。G-Memory 专为 MAS 设计，显式建模「智能体间协作轨迹」：交互图按发言智能体与时序记录多轮对话，洞见图蒸馏分工/协作经验，并通过算子 Φ 依据各智能体角色 Role_i 提供「角色定制化（agent-specific）」的差异化记忆注入——不同智能体在同一任务中获得为其功能量身定制的洞见与协作片段。它批评既有 MAS 记忆（MetaGPT-M/ChatDev-M/MacNet-M）只存最终解或试验内上下文而无视协作轨迹，并将单智能体记忆（Voyager/MemoryBank/Generative）移植为对照，证明 MAS 需专门的角色化、协作感知记忆。与 B9 MIRIX（多智能体管理用户记忆）并列为多智能体记忆代表，但 G-Memory 是「为多智能体协作本身建模记忆」。
- **时序推理支持**: 弱/隐式。交互图的边按时序关系（u_j 传递并启发 u_k）连接，编码了单次协作内的话语先后顺序；查询图隐含任务的累积时间序（随 trial 增多演化）。但 G-Memory 不显式建模事实有效性窗口（fact-validity window）、事件双时间区间或事件日历（区别于 Zep/Graphiti），也未做时序推理专项评测。时间信息主要服务于「重建协作轨迹的因果/对话顺序」，而非对随时间变化的事实做版本化推理。
- **模态**: 纯文本（text-only）。三层图存储的均为文本（智能体话语、查询文本、文本洞见），评测环境（ALFWorld/SciWorld 为文本具身环境、PDDL 文本博弈、HotpotQA/FEVER 文本 QA）皆文本化；不涉及图像/截图/视觉或音频等多模态输入（区别于 B9 MIRIX 的多模态截图记忆）。
- **过度个性化/记忆安全风险**: 基本不适用 / 仅泛泛涉及安全声明。G-Memory 是 agent-centric 协作记忆，不存储用户个人信息，因此过度个性化/谄媚/隐私侵入等 user-centric 安全风险大体超出其范围。论文仅在影响声明中提及：若骨干 LLM 被对抗操纵，记忆机制可能放大错误推理，并呼吁部署时加入持续校验、对抗鲁棒性检查与价值对齐等保障；但未做相关安全基准（如 OP-Bench/Causal-LoCoMo）评估，也未处理有害/过时记忆的治理（加之缺乏遗忘/冲突消解，长期可能累积过时或矛盾洞见）。
- **冲突/矛盾处理**: 较弱、未系统化。更新时被使用的旧洞见会被「更新版本」替换、并刷新其支撑查询集（Eq.11），可视为一种隐式的信息修订；但论文未提供显式的矛盾洞见/事实检测、合并或失效算子，也未做冲突消解评测。记忆随任务单调增长，不同 trial 蒸馏出的洞见之间若存在矛盾缺乏明确仲裁机制，整体弱于 Memory-R1 的 UPDATE 算子或 MEMTRACK 的冲突追踪。
- **token成本/延迟证据**: 重点量化 token 成本（而非延迟），且效率是其卖点之一（resource-friendly）。①核心证据：在 PDDL+AutoGen 上，G-Memory 较 No-memory 提升 10.32% 而仅增加约 1.4×10^6 token，是各记忆基线中「性能提升最高、token 增量最小」的方案；相比之下 MetaGPT-M 等基线带来明显更高的额外 token 开销。②整体上 G-Memory 相对 Generative、MetaGPT-M 等经典基线仅带来边际或近乎零的 token 增量，却持续给出最大性能增益（Figure 3/7 的性能-token 权衡）。③设计动机即针对 MAS 轨迹比单智能体长达 10× token 的问题，通过 LLM 图稀疏器对交互轨迹做 trajectory condensation 来压缩注入上下文，避免朴素长上下文喂入。④论文未给出绝对延迟（秒级）数字或相对全上下文的 p95 延迟节省百分比（区别于 Mem0/Zep 的延迟口径），效率证据主要落在 token 维度。


<a id="d6-letta前身为-memgptletta-是将-memgpt-研究arxiv-231008560产品化的有状态智能体运行时平台公司由原-memgpt-团队创立别名相关memgpt现指论文中具备自编辑记忆工具的-llm-os-智能体设计范式letta-框架开源智能体框架原-memgpt-仓库改名而来letta-code记忆优先的编码智能体-cliappletta-cloud托管-api-平台adeagent-development-environment-可视化调试环境"></a>

### D6 Letta

*Letta（前身为 MemGPT）。Letta 是将 MemGPT 研究（arXiv 2310.08560）产品化的有状态智能体运行时/平台；公司由原 MemGPT 团队创立。别名/相关：MemGPT（现指论文中“具备自编辑记忆工具的 LLM-OS 智能体设计范式”）、Letta 框架（开源智能体框架，原 MemGPT 仓库改名而来）、Letta Code（记忆优先的编码智能体 CLI/App）、Letta Cloud（托管 API 平台）、ADE（Agent Development Environment 可视化调试环境）。*


**基本信息 / Provenance**

- **年份**: 2024 年（Letta 公司/平台首次公开：2024-09-23 携 1000 万美元种子轮从隐身状态发布，同时把开源仓库与 PyPI 包从 MemGPT 更名为 letta）。底层 MemGPT 研究论文最早于 2023-10-12 以 arXiv 预印本公开（arXiv:2310.08560），后被 ICML 收录；GitHub 仓库 letta-ai/letta 创建于 2023-10-11。
- **作者/机构**: 核心团队为 UC Berkeley Sky Computing Lab（前 RISELab/BAIR）的 MemGPT 作者：联合创始人 Charles Packer（CEO，PhD）与 Sarah Wooders（CTO，PhD），以及导师/合作者 Joseph E. Gonzalez、Ion Stoica（均为 Berkeley 教授）；MemGPT 论文其他作者含 Kevin Lin、Vivian Fang、Shishir G. Patil。公司名 Letta（Letta AI / Letta, Inc.），为 UC Berkeley AI 研究实验室 spinout。属工业界（创业公司）+ 学术血统混合。
- **发表venue**: 工业界 / 开源系统（产品化运行时）。底层研究 MemGPT 以 arXiv 预印本（2023，arXiv:2310.08560）发布并被 ICML 收录；Letta 本身为商业公司的开源框架（Apache-2.0）+ 托管云服务，无单独的学术会议/期刊论文。其后续量化研究以独立论文形式发表，如《Sleep-time Compute》（arXiv:2504.13171, 2025）。
- **论文链接**: 底层 MemGPT 论文：https://arxiv.org/abs/2310.08560 （DOI: 10.48550/arXiv.2310.08560）。Letta 官网 https://www.letta.com ；文档 https://docs.letta.com ；更名公告 https://www.letta.com/blog/memgpt-and-letta 。相关后续研究《Sleep-time Compute》：https://arxiv.org/abs/2504.13171 。
- **代码链接**: https://github.com/letta-ai/letta （官方开源框架，原 MemGPT 仓库改名而来，Apache-2.0，Python，约 23.2k star、约 2.5k fork、约 150+ 贡献者、139 watchers，创建于 2023-10-11；PyPI 包名 letta，Docker 镜像 letta/letta-server）。另有 https://github.com/letta-ai/letta-code （Letta Code，记忆优先编码智能体，Apache-2.0，约 2.7k star，创建于 2025-10-25，npm 包 @letta-ai/letta-code）。SDK：letta-client（Python/TypeScript）。
- **引用数**: 底层 MemGPT 论文约 767 次引用（Semantic Scholar 实时数据，截至 2025-2026），属智能体记忆方向的高被引奠基性工作之一；其衍生《Sleep-time Compute》论文约 30 次引用。开源框架 letta 约 23.2k star，工业界影响力与采用度极高，与 Mem0、Zep 并称 2026 年“三大生产级智能体记忆层”。

**记忆分类 / Taxonomy**

- **记忆类型**: 覆盖多种 CoALA 记忆类型：1) 工作记忆/核心记忆（working / core memory）——常驻上下文窗口的 memory blocks（如 persona、human），始终可见、无需检索；2) 情景记忆（episodic）——recall memory（对话/消息历史数据库，可检索过往会话）；3) 语义记忆（semantic）——archival memory（语义可搜索的长期知识库/向量库，存事实、文档、知识）；4) 程序性记忆（procedural，Letta Code 阶段）——通过 skills（技能学习）与 MemFS/system 目录中的可复用规程/工作流实现。整体把 LLM 上下文当作分层记忆资源管理（LLM-OS 范式）。
- **记忆结构**: 分层记忆层级（hierarchical memory tiers，借鉴操作系统虚拟内存/分页）：核心是 in-context（main context）与 external context 的两级划分，进一步细分为三/四层：核心记忆 core memory（结构化的 memory blocks，常驻系统提示，以类 XML 块格式 <persona>/<human> 等注入，每块有 label/description/value/limit 字符上限，可被多智能体共享）；recall memory（消息/对话历史数据库，FIFO 队列溢出后的可检索存储）；archival memory（通用向量数据库，无大小上限的语义检索 passages，可带 tags）。Letta Code 阶段引入 MemFS（git-backed 记忆文件系统 / context repository）：以 markdown 文件目录组织记忆，system/ 目录文件全量载入系统提示，其余文件仅暴露文件名+描述（记忆树），按需读取以保持上下文精简。底层非单一数据结构，而是“原始缓冲(FIFO)+结构化块+向量库(+文件系统/git)”的混合分层结构。
- **存储后端**: 外部持久化数据库 + 上下文窗口混合。所有状态（memory blocks、消息、推理、工具调用）均持久化于数据库（默认/自托管常用 PostgreSQL，archival memory 用 pgvector 扩展做向量相似度搜索，HNSW 索引实现亚秒级近似查询；MemGPT 原论文即用 PostgreSQL + pgvector，并预计算 OpenAI text-embedding-ada-002 嵌入，20M Wikipedia 文章）；核心 memory blocks 注入上下文窗口（LLM 物理内存/RAM）。嵌入模型可配置（OpenAI text-embedding-3-small 等，亦支持 Azure/Ollama/Anthropic/Vertex/vLLM/Google 等多提供商）。Letta Code 阶段记忆后端为 git-backed 文件系统 MemFS（本地路径 ~/.letta/agents/<id>/memory，API 模式推送提交同步至 Letta Cloud 并保留完整版本历史）。模型完全 model-agnostic，不修改 LLM 权重；服务端工具在沙箱执行。
- **持久化**: 外部持久化（durable external store）+ 非参数化。Letta 的核心定位是“有状态智能体（stateful agents）”：所有记忆、消息、推理、工具调用都持久化到数据库，即使被逐出上下文窗口也永不丢失，可通过 API（开发者）与检索工具（智能体）随时取回。核心 memory blocks 常驻上下文（临时可见层），recall/archival/文件系统为外部持久层。不写入 LLM 权重（学习发生在 token 空间/外部记忆，而非梯度更新）。Letta Code 进一步用 git 提交历史持久化“智能体学到的一切”，并强调记忆可跨模型代际迁移（continual learning in token space）。

**核心机制 / Mechanisms**

- **写入/编码**: 记忆写入以智能体“自编辑（self-editing）”为核心、辅以可编程/外部写入：1) 核心记忆——智能体在推理循环中调用内建记忆工具（如 core_memory_append/replace、memory blocks 编辑，或新模型上的 memory omni-tool）来增改 memory blocks 的纯文本 value（替换式更新，受字符上限约束）；开发者亦可经 SDK 直接 update/attach/detach 块。2) recall memory——队列管理器把每条新消息与 LLM 输出自动写入消息数据库（无损存储原始对话），上下文溢出时按 FIFO 逐出但仍存于 recall。3) archival memory——智能体调用 archival_memory_insert(content, tags) 把判定值得长期保存的事实/知识编码为带标签的 passage 并嵌入向量库；开发者可经 passages.insert 程序化写入。4) 上下文压力管理——当 prompt tokens 超过“警告阈值”（如上下文 70%）时插入 memory pressure 系统消息提示智能体把 FIFO 中重要信息转存至 working/archival；超过“flush 阈值”（如 100%）时逐出约 50% 消息并用旧递归摘要+被逐消息生成新的递归摘要（recursive summary）写回队列首。5) Letta Code/MemFS——智能体用 bash 工具直接编辑 markdown 记忆文件并 git commit 保存。编码形态涵盖逐字轨迹（recall）、提炼洞见/摘要（blocks、递归摘要、sleep-time 学到的 learned context）、抽取事实（archival passages）、向量嵌入。
- **检索机制**: 记忆读回为“总是可见 + 按需工具调用”的混合智能体式检索：1) 核心 memory blocks 始终拼接进上下文（无需检索，零检索开销，以类 XML 块呈现并带 chars_current/limit 元数据）。2) recall memory——智能体调用 conversation_search（分页）对过往消息做检索召回，把结果重新追加到 FIFO 队列尾以回插上下文。3) archival memory——智能体调用 archival_memory_search(query, tags, page) 做语义（向量余弦相似度）搜索，理解概念而非仅精确关键词（如“artificial memories”可命中“implanted memories”），支持 tag 过滤与分页；底层为 pgvector + HNSW 近似最近邻。4) 函数链（function chaining）——通过 request_heartbeat=true 标志请求紧随的 LLM 推理，把多次检索/工具调用串联，实现多跳检索（如先 archival 取知识再 recall 取近期对话）。整体检索质量取决于 LLM 自身决定调用哪个记忆层与查询措辞的“智能体式（agentic）”能力（这是与 Mem0 被动语义检索的关键哲学差异）。Letta Code/MemFS 阶段：记忆树暴露文件名+描述，智能体用文件读取/grep 等 bash 工具按需载入相关 markdown 记忆文件。论文未采用 recency·importance·relevance 加权打分公式（区别于 Generative Agents），也无 cross-encoder 重排（被第三方指为其与 Mem0 共同的检索局限）。
- **反思/巩固**: 有显式的“原始→更高层知识”整合，主要通过 sleep-time（睡眠时/后台反思）机制实现，这是 Letta 的标志性贡献：1) Sleep-time agents（睡眠时智能体）——可为主智能体创建共享其记忆块的后台智能体，异步反思原始上下文（对话历史/文件/数据源）并迭代提炼出“learned context（学到的上下文）”写入共享 memory blocks，把最重要的信息/洞见沉淀下来。触发：主智能体每 N 步（默认 5）调用一次睡眠时智能体；可经 sleeptime_agent_frequency 配置。2) Letta Code 的 dreaming/reflection——周期性启动 sleep-time（dream）子智能体对近期对话反思以主动创建/巩固记忆；触发器可设为 Off / 每 N 条用户消息（step count）/ 上下文压缩事件（compaction，推荐）。3) 递归摘要（recursive summarization）——FIFO 队列逐出时把被逐消息压缩进滚动递归摘要，是上下文巩固的轻量形式。4) /init、/doctor、/remember 等命令支持交互式记忆初始化、记忆布局审计与定向记忆写入。其衍生论文《Sleep-time Compute》形式化了“在查询到来前用空闲算力离线预计算/思考上下文”的范式，并给出量化收益。整体属“在线 + 后台异步反思巩固”，但官方提示睡眠时调用过频代价高（高 token 用量）、边际收益递减，建议频率保持 5~10。
- **遗忘/更新**: 更新以“自编辑覆盖”为主，无真正的生物式遗忘衰减：核心 memory blocks 通过智能体工具做替换式更新（core_memory_replace 覆盖旧值、受字符上限约束），超限即逼迫智能体精炼/取舍内容（隐式遗忘）；上下文溢出时 FIFO 逐出旧消息（从上下文“遗忘”但仍存于 recall 数据库，可检索，非物理删除）并生成递归摘要。archival passages 对智能体基本不可变（智能体难以修改/删除，开发者可经 SDK 删改）。块/消息可经 API 删除、detach（从上下文移除但保留块本身）。MemFS 用 git 版本历史记录所有变更（可追溯/回溯）。无 Ebbinghaus 衰减曲线、无自动去重/冲突合并的内建算法（区别于 Mem0 的 ADD/UPDATE/DELETE 与 Zep 的时序边失效）；冲突解决依赖 LLM 在自编辑时的判断。
- **经验回放 (核心主题)**: 经验复用是 Letta（尤其 Letta Code）后期的核心主题，定位为“能从自身经历与工作中持续改进的智能体”。具体形态：1) 跨会话同一智能体——用户跨 session/天/月复用同一个有状态智能体，使其随时间变好；过往交互、偏好、洞见沉淀在 memory blocks / MemFS 中，在后续任务中作为常驻上下文或按需检索被反复复用。2) Skill Learning（技能学习，2025 起）——智能体把过往经验动态归纳为可复用的 skills（技能），在后续任务中发现并加载相关技能完成工作（在 Context-Bench 的 Skill Use 评测中度量），实现“经验→可复用规程”的回放式提升而非随时间退化。3) Sleep-time/dreaming——把原始对话经验离线提炼为 learned context 供未来复用（《Sleep-time Compute》中通过 Multi-Query GSM-Symbolic 把同一上下文的睡眠时算力摊销到多个相关查询，平均每查询成本降 2.5x，即一种“经验预计算+跨查询复用”）。4) recall/archival——逐字对话轨迹与抽取事实持久存储，构成可重复检索的“经验缓冲”。与 Voyager/Reflexion 等智能体自我改进系统同属 agent-centric 经验复用，但 Letta 提供的是通用运行时基础设施而非单一算法。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric / in-context，“在 token 空间学习”）为主。Letta 明确主张“continual learning in token space（在 token 空间持续学习）”——智能体的改进发生在外部记忆/上下文（memory blocks、archival、MemFS、skills）而非 LLM 梯度更新，从而使学到的记忆可跨基础模型代际迁移、超越任何单一模型。模型完全 model-agnostic、权重冻结。属非参数化 + 提示/上下文级学习；不做参数微调（无 hybrid 的梯度训练成分）。其研究分支《Sleep-time Compute》探索“离线预计算”作为推理时算力的补充，仍属非参数化推理范式。
- **失败学习 (核心主题)**: 失败学习以“自编辑记忆 + 定向纠正 + 后台反思”实现，但非内建的算法化失败模式库：1) Letta Code 的 /remember 命令——当智能体犯了易避免的错误时，用户可直接指令“/remember not to make that mistake again”，智能体据此写入记忆以避免重犯；亦可无显式提示让智能体从上下文推断意图做记忆编辑。2) memory blocks 显式支持“存储工具使用指南以避免过去的错误（avoid past mistakes）”这一用法。3) sleep-time/dreaming 子智能体反思近期交互（可包含失败）以巩固/修正记忆。4) Letta 研究侧发布 Recovery-Bench（2025-08）专门评测智能体“从错误与损坏状态中恢复”的能力，显示其对失败恢复的关注。整体是把失败经验编码进可复用记忆/规程，而非 Reflexion 式的结构化“失败轨迹自反思→言语反馈缓冲”或负样本机制；失败检测与转化依赖 LLM 判断与用户纠偏。
- **技能/程序归纳**: 是（尤其 Letta Code 阶段）。通过 Skill Learning（2025-12 发布）让智能体“通过经验动态学习技能”，把过往经历归纳为可复用 skills；Letta Code 内置预制 skills/subagents 支持高级记忆与持续学习，并支持自定义 skills。技能表示为可被发现/加载的能力单元（在 Context-Bench 的 Skill Use 评测中度量“从技能库中发现并加载相关技能”的能力），在 MemFS 中以文件/规程形式组织、由智能体按需调用。这是 Letta“记忆优先、随经验改进而非退化”的核心机制之一。
- **在线 vs 离线**: 在线（online）为主 + 后台异步/睡眠时离线巩固。智能体在部署中实时自编辑记忆（每次交互、每个 episode 即时更新 memory blocks、写 recall/archival）；同时由 sleep-time/dreaming 子智能体在后台异步（每 N 步或上下文压缩事件触发）对近期经验做离线反思巩固。《Sleep-time Compute》更将“在查询到来前的离线预计算”形式化为一种新的离线算力维度。不存在对固定训练轨迹语料的批量离线梯度训练（因属非参数化）。整体为 both（在线自编辑 + 后台离线反思），偏在线/部署期持续学习。

**评测 / Evaluation**

- **任务领域**: 通用、跨域的有状态智能体平台：多会话长期对话/个性化助手与虚拟伴侣（MemGPT 原始动机即改进 Discord 聊天机器人记忆）、长文档分析与多文档问答（document analysis / multi-document QA，处理远超上下文窗口的文档）、客服/支持（积累机构知识）、个人知识管理（30k+ archival 记忆）、社交媒体智能体、企业应用；Letta Code 阶段扩展到编码/终端使用（coding / terminal-use / GUI-免，本地计算机操作）与软件工程任务。覆盖 QA、多会话对话、文档分析、编码/终端、个性化等多域，是通用运行时而非单一任务系统。
- **基准**: MemGPT 原论文：Multi-Session Chat（MSC，Xu et al. 2021，扩展版）上的 Deep Memory Retrieval（DMR，作者新提，5 会话后单问答一致性测试）与 Conversation Opener（互动性，CSIM 分）；文档分析用 retriever-reader 多文档 QA（基于 NaturalQuestions-Open + Wikipedia，Liu et al. 2023a）、key-value 检索、及作者新提的 nested key-value retrieval（多源/多跳）。后续 Letta 研究自建/采用多套基准：Letta Leaderboard（智能体记忆，2025-05）、Context-Bench（agentic context engineering，2025-10）、Skill Use（2025-11）、Recovery-Bench（错误恢复，2025-08）、Terminal-Bench（终端使用，2025-08）；并用 LoCoMo 评测 Letta Filesystem（2025-08）。注意：Letta/MemGPT 官方未发布 LongMemEval 成绩（第三方亦指出此点）；Zep 论文用 DMR（源自 MemGPT）对标并超越 MemGPT。
- **报告增益**: MemGPT 原论文 DMR（表 2，对比无 MemGPT 的同模型固定上下文基线，基线只见过去 5 会话的有损摘要）：GPT-4 Turbo+MemGPT 准确率 93.4% / ROUGE-L 0.827（基线 35.3% / 0.359，绝对 +58.1pp）；GPT-4+MemGPT 92.5% / 0.814（基线 32.1% / 0.296）；GPT-3.5 Turbo+MemGPT 66.9% / 0.629（基线 38.7% / 0.394）——MemGPT 显著优于固定上下文基线。文档 QA（图 5）：MemGPT 性能不随上下文长度/文档数增加而退化（固定上下文基线性能受检索器上限与“中间遗忘”制约）。Conversation Opener：MemGPT 生成的开场白与人工 gold 开场相当、偶有超越且更全面。后续研究《Sleep-time Compute》（arXiv:2504.13171）：在 Stateful GSM-Symbolic 与 Stateful AIME 上，睡眠时算力可在达到同等准确率时把测试时算力需求降低约 5x；进一步扩大睡眠时算力可把准确率再提升至多 13%（GSM-Symbolic）/ 18%（AIME）；用 Multi-Query GSM-Symbolic 把睡眠时算力摊销到同上下文的多个相关查询，平均每查询成本降 2.5x。Letta Filesystem 在 LoCoMo 上仅靠把对话历史存为文件即得 74.0%，超过若干专用记忆工具库。Letta Code 在 Terminal-Bench 上 42.5% 总分（排名第 4、Claude 4 Sonnet 类智能体中第 2），并被官方称为该榜单第 1 的 model-agnostic 开源智能体。
- **对比基线**: MemGPT 原论文主要对比“同底层 LLM 但无 MemGPT 的固定上下文基线”（GPT-4 / GPT-4 Turbo / GPT-3.5 Turbo），对话任务基线可见过去会话的有损递归摘要；文档 QA 对比共用同一检索器的 retriever-reader 固定上下文基线（含 truncation 压缩法）。《Sleep-time Compute》对比标准“测试时算力扩展（test-time compute scaling）”范式（即无睡眠时预计算、把全部算力放在查询到来后）。Letta 后续工作横向对标其它生产级记忆层（Mem0、Zep 等专用记忆工具库）与 Claude Code / Codex 类编码智能体。整体涵盖：无记忆/固定上下文、摘要式压缩、传统 RAG（官方专文《RAG is not Agent Memory》区隔）、专用记忆库及竞品平台。

**分析 / Analysis**

- **关键创新**: 把 MemGPT 的“LLM 作为操作系统（LLM-OS）/虚拟上下文管理”研究范式产品化为业界领先的有状态智能体运行时：核心创新是让 LLM 通过工具调用自编辑、自管理一套分层记忆（常驻 core memory blocks + 可检索 recall + 语义 archival），用 OS 分页/中断式的事件驱动控制流在有限上下文窗口内提供“无限上下文”的错觉，并把所有状态持久化为可部署的有状态服务。相对 MemGPT 论文的关键产品化增量包括：结构化可共享的 memory blocks 抽象、sleep-time agents（后台异步反思巩固，及随附的 Sleep-time Compute 离线预计算范式）、MemFS（git-backed 记忆文件系统 / context repository）与 Skill Learning（在 token 空间持续学习、跨模型代际迁移）。是“记忆即智能体核心抽象、智能体在运行时持续学习”理念的旗舰实现。
- **局限**: 1) 记忆质量完全依赖 LLM 自编辑判断：若模型未能保存某信息则永久丢失，且每次记忆操作都消耗推理 token（与 Mem0 被动抽取相比可预测性差、token 开销大）；2) 检索为智能体式工具调用，质量随底层模型与提示工程显著波动，且未用多策略检索/cross-encoder 重排（第三方指为与 Mem0 共同的检索深度局限）；3) 官方未发布 LongMemEval 等通用记忆基准成绩，难做直接量化横比；4) 作为“完整运行时/平台”而非可插拔记忆层，架构耦合带来较高锁定成本（迁移需重写整个智能体基础设施）、学习曲线陡、概念多；5) sleep-time/dreaming 频繁触发代价高（高 token 用量）、边际收益递减；6) 无真正的遗忘衰减、无内建自动去重/时序冲突消解（不及 Zep 的双时序边失效、Mem0 的 ADD/UPDATE/DELETE）；7) DMR 等早期基准规模小（被 Zep 等批评不足以区分记忆系统）。
- **与其他工作关系**: 属本研究 D 类“图/神经启发/生产级”聚类，是 2026 年“三大生产级记忆层”之一（与 D4 Mem0、D3 Zep/Graphiti 并列）。直接产品化并扩展 B3 MemGPT（Packer et al. 2023，本研究中 MemGPT 单列为 B3；Letta 即同一团队把该 LLM-OS/分层记忆范式工程化为运行时），两者共享 letta-ai/letta 仓库谱系。与 D4 Mem0 的核心区别：Mem0 是可插拔“记忆层”（被动抽取+语义检索，框架无关，库），Letta 是“智能体运行时/平台”（智能体在其内运行、自编辑分层 core/recall/archival 记忆，主动 agentic 检索）——哲学差异为“可预测性 vs 智能性”“模块化 vs 紧耦合”。与 D3 Zep/Graphiti 区别：Zep 用双时序知识图谱做用户/对话事实记忆并显式建模时间有效性，Letta 用分层记忆块+向量/文件系统且记忆管理为 LLM 自编辑（启发式而非图式/时序失效）；Zep 论文曾用源自 MemGPT 的 DMR 对标并超越 MemGPT。其 sleep-time/反思与 A 类 Reflexion、B1 Generative Agents 的反思巩固同源但实现为后台共享记忆的多智能体；MemFS/Skill Learning 与 C 类技能/工作流归纳（如 C1 Voyager、C2 AWM）及 G 类持续学习相呼应，但定位为通用运行时基础设施而非单一算法。与 RAG 明确区隔（官方《RAG is not Agent Memory》）。
- **可复现性**: 可复现性与社区采纳度极高：开源框架 letta-ai/letta（Apache-2.0，Python，约 23.2k star、约 2.5k fork、150+ 贡献者）可自托管（PyPI: pip install letta-client；Docker: letta/letta-server），含 Python/TypeScript SDK、可视化 ADE 调试环境、Agent File(.af) 序列化格式；Letta Code 亦开源（约 2.7k star，npm @letta-ai/letta-code）。多个评测套件开源（Letta Leaderboard、Context-Bench、Recovery-Bench、Letta Evals）。底层 MemGPT 论文公开代码、增广 MSC、nested-KV 数据集及 20M Wikipedia 嵌入（research.memgpt.ai）。商业 Letta Cloud 为托管服务（$20–200/月档），自托管免费且全功能。DeepLearning.AI 与 Letta 合作开设智能体记忆课程，进一步降低上手门槛。需自备 LLM/嵌入 API 与数据库（PostgreSQL/pgvector）。整体部署门槛中等、生态成熟。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（以启发式/提示驱动 + LLM 自编辑为主，而非用 RL/训练学习记忆控制策略本身）。Letta 的记忆管理策略（何时存什么、何时检索/更新、逐出/巩固）由冻结 LLM 在推理循环中经提示与工具描述自主决定（agentic self-editing），并由固定的 OS 式启发式（警告/flush 阈值、FIFO 逐出、每 N 步触发 sleep-time、递归摘要）调度——属“在 token 空间学习”而非端到端训练记忆控制器。这与 2025-26“学习型记忆控制”代际（Memory-R1、Mem-α 等用 RL 学习记忆策略）形成对照：Letta 处于非训练型/智能体自编辑一侧。其研究虽探索 Sleep-time Compute（离线预计算）与 Skill Learning，但仍为非参数化、非 RL 的记忆控制。
- **记忆主体**: 兼具用户中心与智能体中心，但整体偏“智能体中心（agent-centric）”定位。一方面（用户中心）：标准 persona/human memory blocks、recall 对话历史、个性化助手/客服用例服务于记住用户信息以个性化（与 Mem0/Zep 同类）。另一方面（智能体中心，且为其核心叙事）：Letta 强调智能体记住并复用“自身经历与工作”以自我改进（self-improving），Letta Code 的同一智能体跨会话变好、Skill Learning、sleep-time 反思、continual learning in token space 均指向 agent-centric 自我提升（与 ReasoningBank/Voyager 同向）。可个性化为“unique to you”的智能体，模糊了两类边界。
- **多智能体记忆**: 支持多智能体共享/路由记忆。memory blocks 可被多个智能体同时 attach（共享记忆块：一处更新、处处即时可见），是显式的多智能体协调原语（如父智能体实时观察子智能体结果块更新、跨智能体共享只读组织策略）。Sleep-time agents 本身即被描述为“共享一个或多个 memory blocks 的特殊多智能体架构”。Letta 还提供 groups（多智能体组）、subagents（子智能体）、Conversations API（跨并发体验的共享智能体记忆，2026-01）。属支持洞见/规程跨智能体共享与协调的多智能体记忆，但其分层机制不同于 G-Memory 的三层图或 MIRIX 的查询/交互分层。
- **模态**: 以文本为主，逐步走向多模态/具身（embodied 计算机使用）。核心记忆抽象（blocks/recall/archival/MemFS）为文本/结构化文本；Letta Filesystem 支持 PDF、转录、文档等内容的组织引用（仍以文本提取为主）。Letta Code 让智能体获得“real computer access（真实计算机访问）”在本地机器上工作（终端/文件操作，偏具身-of-computer），但记忆表示仍主要是文本/文件。非以视觉/视频记忆为核心（区别于 MIRIX 截图记忆）。
- **过度个性化/记忆安全风险**: Letta 在该负面/安全维度有相对积极的近期工作（不同于多数系统的忽视）：1) 提供 read-only memory blocks（只读块）以防智能体对共享/组织信息做破坏性修改；archival passages 对智能体基本不可变（防误删/篡改）；记忆可经 API 删除以支持“被遗忘权”类治理（自托管/企业合规）。2) 研究侧发布《Context Constitution》（2026-04，治理 AI 智能体如何管理上下文以从经验学习的原则集）与《Red-teaming the Context Constitution》（2026-05，红队审计具备身份/长期经验/自演化能力的智能体模型），直面长期记忆智能体的安全与对齐风险。但官方未提供 OP-Bench/Causal-LoCoMo 类“过度个性化/谄媚/陈旧记忆”的标准化量化评测，相关保障更多在工程与原则层面。
- **token成本/延迟证据**: 有明确的效率量化证据，且多源于其睡眠时算力研究：《Sleep-time Compute》（arXiv:2504.13171）显示，在 Stateful GSM-Symbolic 与 Stateful AIME 上，达同等准确率时测试时算力需求降低约 5x；用 Multi-Query GSM-Symbolic 把睡眠时算力摊销到同一上下文的多个相关查询，平均每查询成本降 2.5x（核心机制：在查询到来前的空闲/睡眠时间预计算有用量，从而显著削减测试时延迟与推理成本）。常驻 memory blocks 本身“始终可见、无需检索”，省去检索往返；MemFS 仅把 system/ 文件全量入提示、其余仅暴露文件名+描述以保持上下文精简、降 token。但官方提醒 sleep-time 频繁触发会推高 token 用量（建议频率 5~10）；第三方亦指出 Letta 的每次自编辑记忆操作都额外消耗推理 token（与 Mem0 被动抽取相比可能更贵）。整体效率叙事偏“离线预计算/睡眠时算力换取测试时低延迟低成本”，而非如 Mem0(-90% token)/Zep(-90% 延迟) 那样给出针对长上下文记忆任务的统一压缩比。

**不确定字段 / Uncertain**

- 冲突/矛盾处理 (`conflict_contradiction_handling`)
- 时序推理支持 (`temporal_reasoning_support`)


<a id="d7-memmachine别名memmachine-记忆层--memverge-开源记忆系统论文标题memmachine-a-ground-truth-preserving-memory-system-for-personalized-ai-agents"></a>

### D7 MemMachine

*MemMachine（别名：MemMachine 记忆层 / MemVerge 开源记忆系统；论文标题《MemMachine: A Ground-Truth-Preserving Memory System for Personalized AI Agents》）*


**基本信息 / Provenance**

- **作者/机构**: Shu Wang、Edwin Yu、Oscar Love、Tom Zhang、Tom Wong、Steve Scargall、Charles Fan（全部隶属 MemVerge, Inc.，一家专注内存/AI 基础设施的公司；Charles Fan 为 MemVerge 联合创始人兼 CEO）。任务给定的 Wang et al. 与第一作者 Shu Wang 一致。
- **论文链接**: https://arxiv.org/abs/2604.04853 （HTML 版：https://arxiv.org/html/2604.04853v1 ；DOI: 10.48550/arXiv.2604.04853）
- **代码链接**: https://github.com/MemMachine/MemMachine （Apache-2.0 许可，Python；GitHub API 实时数据约 3,107 stars、180 forks、最新版本 v0.3.9；另有 PyPI 包 memmachine-server / memmachine-client，官网 https://memmachine.ai）

**记忆分类 / Taxonomy**

- **记忆类型**: 情景记忆（episodic，存储原始对话回合，作为事实“ground truth”）+ 语义记忆（semantic，以 Profile Memory 形式抽取用户事实/偏好）+ 工作记忆（working/short-term，最近若干回合的即时上下文窗口）。明确不实现程序性记忆（procedural），仅在“未来工作”中提出可扩展。对应 CoALA 中的 episodic + semantic + working 三类。
- **记忆结构**: 分层多层（hierarchical tiers）+ 句级向量索引 + 图结构混合。三层：短期记忆（STM，最近回合缓冲 + LLM 摘要）、长期情景记忆（LTM，原始 episode 经 NLTK Punkt 切分为句子，每句生成嵌入并保留到源 episode 的关系映射）、Profile 语义记忆（结构化用户画像）。LTM 同时使用图数据库（Neo4j，关系遍历）与向量库（PostgreSQL+pgvector）。
- **存储后端**: 外部持久化多后端：PostgreSQL（含 pgvector，向量相似检索）、Neo4j（图结构 LTM）、SQLite（轻量/Profile）；Profile Memory 存于 SQL（PostgreSQL 或 SQLite）。可选向量后端：Qdrant、hnswlib、usearch、NebulaGraph；reranker 默认 AWS Cohere rerank-v3.5（也支持 BM25）。嵌入默认 OpenAI text-embedding-3-small，可配置本地模型（Ollama/vLLM）。
- **持久化**: 外部持久化（durable，存于数据库，跨会话/重启/换模型均保留）。STM 为相对短暂的上下文工作区，溢出后压缩并迁移到持久化 LTM。无参数化（parametric）记忆——刻意保持应用层、文本 API 接口，可对接任意闭源/开源 LLM。

**核心机制 / Mechanisms**

- **写入/编码**: 采用“ground-truth-preserving（保真）”写入：原始对话被组织为 Episode（每个对话回合一个 episode，附带 producer/timestamp/session_id/自定义元数据），原文逐字存入中央数据库作为原始仓库，刻意最小化对 LLM 的依赖。索引阶段：LTM 通过 NLTK Punkt 分词器将每个 episode 切分为句子（sentence-level chunking），每句继承父 episode 元数据并获得唯一 ID、保留到源 episode 的关系映射，再为每句生成语义嵌入。关键区别于 Mem0/Zep：不做逐条消息的 LLM 事实抽取（per-message extraction），LLM 仅用于 STM 摘要和 Profile 抽取两处高层抽象，从而避免概率抽取带来的事实漂移与高成本。
- **检索机制**: 分阶段召回流水线：用户查询(+过滤器) → STM 检索 → LTM 向量相似检索（ANN 近似最近邻或精确匹配，针对句级嵌入）→ 上下文化（Contextualization）→ episode 去重 → cluster 重排序（cross-encoder/reranker）→ 按时间顺序排序 → 返回。核心创新为“上下文化检索（contextualized retrieval）”：先用嵌入定位 nucleus 核心 episode，再取邻接 episode（前 1 条、后 2 条）形成 episode cluster，以解决会话数据中相关 episode 嵌入与查询差异大的问题；随后用 cross-encoder 等重排序模型对 cluster 重排，取 top-k 个 cluster 供 LLM。检索阶段可调参数 top_k（消融显示从 20→30 提升最显著 +4.2%，但 50 反而下降，呈非单调“lost in the middle”效应）。另有可选 Retrieval Agent（agent_mode），由 ToolSelectAgent 用单次 LLM 调用将查询分类后路由到三种策略：MemMachine 直接检索（单跳）、SplitQuery 并行分解（多实体扇出，分 2–6 个子查询并发执行）、ChainOfQuery 迭代链式查询（多跳，最多 3 轮，含充分性判断+查询改写，置信度≥0.8 早停），并采用 multi-query reranking（重排时拼接所有查询/改写）。
- **反思/巩固**: 存在两类“原始→高层知识”的转化，但均刻意轻量化：(1) STM 摘要——当短期记忆窗口溢出时，调用 LLM 对会话级交互生成压缩摘要，episode 与摘要一并压缩后迁入 LTM，使窗口内仍保留旧上下文的要点；(2) Profile 抽取——LLM 从对话中抽取并维护结构化用户画像（人口学信息、偏好、跨会话行为模式、专业背景），形成语义记忆。与 Generative Agents 的反思/Mem0 的逐条抽取不同，MemMachine 不对每条消息做抽取与图谱构建，主张“检索阶段优化”比“摄取阶段优化”对最终精度贡献更大（消融证据：检索侧累计提升远超摄取侧句切分的 +0.8%）。记忆巩固/遗忘机制（认知启发式的优先级保留与陈旧信息退役）列为未来工作，当前未实现。
- **遗忘/更新**: 当前无真正的遗忘/衰减机制（无 Ebbinghaus 曲线、无 ADD/UPDATE/DELETE 三元操作），原始 episode 永久保真存储。更新仅发生在 Profile（语义）层：当新信息与已有画像冲突时，系统可更新画像以反映最新状态，从而支持 LongMemEval 的“knowledge update”能力。记忆巩固与优雅退役陈旧信息明确列为未来工作。
- **经验回放 (核心主题)**: 不是面向 agent 自身经验复用/技能复用的“经验回放”系统，而是面向用户的事实召回系统（user-centric）。其“经验复用”体现为：将过去全部对话原始 episode 永久保真存储，在后续会话中通过上下文化检索按需取回原始片段供 LLM 推理（保证可审计、合规、多跳推理所需的逐字事实），而非将轨迹蒸馏为可复用策略或 exemplar prompting。STM 摘要提供高层连续性，检索取回未压缩原始 episode 提供事实根基——论文称这是相对“仅摘要(compaction)”和“全上下文”的第三种折中。不实现技能/轨迹蒸馏、replay buffer 或 exemplar 复用。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric，prompt/in-context 层面）。完全在应用层通过外部数据库存取与检索实现，不对 LLM 权重做梯度更新；明确强调 LLM-agnostic、可换任意模型而记忆系统不变（gpt-4o-mini→gpt-4.1-mini 可直接带来 3–4 个百分点提升）。提示词曾用 Agent Lightning 的 APO 自动提示优化离线调优（仅调答案提示 +约4%，全部 agent 提示联调 +约6%），但属离线提示工程而非在线参数学习；RL 优化检索策略列为未来工作。
- **失败学习 (核心主题)**: 不是面向失败学习/自我反思的系统。论文未实现对失败 episode 的自我反思、失败模式记忆、负样本 exemplar 或错误驱动规则。其稳健性来自“保真存储+检索质量”而非从失败中学习。Retrieval Agent 的 ChainOfQuery 含“充分性判断（sufficiency judgment）”，在证据不足时改写查询继续检索（最多 3 轮、置信度<0.8 才继续），可视为对“检索不足”这一即时失败的局部纠错，但并非跨 episode 的失败经验积累。
- **技能/程序归纳**: 否。MemMachine 当前不归纳可复用技能/工作流/程序（不实现程序性记忆）。论文将“程序性记忆——存储与检索已学动作模式、工具使用策略、工作流配方”明确列为未来工作方向，架构可扩展但尚未支持。
- **在线 vs 离线**: 在线构建（online）为主：部署期间逐回合摄取 episode、即时索引（号称比旧版本快约 75% 的写入速度），跨会话持续累积。提示词的 APO 优化是离线一次性进行、推理时不做在线调优（不增加运行时 token/延迟）。基准评测中按会话顺序逐条摄取。

**评测 / Evaluation**

- **任务领域**: 多会话个性化对话/长期会话记忆为主；多跳问答（QA）；时间推理；知识更新；指代消解。覆盖企业级个性化场景（论文示例：CRM 销售、医疗导航、个人理财顾问、写作助手）。不涉及网页导航、具身（ALFWorld/Minecraft）、游戏、GUI、编程等 agent 自我提升类领域。
- **基准**: LoCoMo（很长期会话记忆，1,540 道计分题，4 类：单跳/多跳/时间/开放域）、LongMemEval-S（ICLR 2025，500 题，约 115k token/题，6 类能力）、HotpotQA hard（多跳，500 题）、WikiMultiHop（2WikiMultiHopQA，500 题，含随机噪声设置）、MRCR（多轮指代消解，300 题）、EpBench（情景记忆，546 题）。
- **报告增益**: LoCoMo：gpt-4.1-mini 总分 0.9169（agent 模式；memory 模式 0.9123），为开源记忆框架最强结果之一，超过 Memobase 0.7578、Zep 0.7514、Mem0 0.6688、LangMem 0.5810、OpenAI 原生记忆 0.5290（同口径 gpt-4o-mini 下 MemMachine 为 0.8747，较次优 Memobase 高 +9.7 个百分点）。LongMemEval-S：六维度系统消融最佳配置（C15，GPT-5-mini，top_k=100）达 93.0% 总准确率；检索侧优化贡献远超摄取侧——检索深度 k:20→30(+4.2%)、上下文格式化(+2.0%)、搜索提示设计 Edwin1→3(+1.8%)、去 CoT 改简洁提示(+1.6%)、用户查询偏置纠正(+1.4%)，而摄取侧句切分仅 +0.8%；意外发现 GPT-5-mini 作答比 GPT-5 高 +2.6% 且更省钱（Pareto 最优为 C12：GPT-5-mini, k=20, 0.922, 仅 2.58M 输入 token）。Retrieval Agent：HotpotQA hard 准确率 93.2%、gold 支撑事实召回 92.31%（较基线 MemMachine 91.2%/90.98% 提升 +2.0/+1.3 个百分点，其中 ChainOfQuery 多跳召回最高 95.31%）；WikiMultiHop 随机噪声下 92.6%（基线 87.4%，+5.2 点）；MRCR 81.4%（vs 基线 79.6%，无记忆 LLM 仅 32.3%）。效率：LoCoMo 上 memory 模式输入 token 4.20M vs Mem0 main/HEAD 19.21M——约少 78%（论文综述口径约 80% fewer input tokens）；写入与检索速度较旧版快约 75%。
- **对比基线**: 无记忆全上下文基线（LLM with full context）；RAG 范式（隐式对比）；以及主流记忆系统：Mem0（重跑 main/HEAD 并以 gpt-4.1-mini 重测）、Zep、Memobase（v0.0.37）、LangMem、OpenAI ChatGPT 原生记忆；架构对比还涉及 MemGPT、Mastra 观测记忆、MemOS。

**分析 / Analysis**

- **关键创新**: “保真（ground-truth-preserving）”架构 + “上下文化检索”：存储原始对话 episode 并做句级索引、刻意把 LLM 仅用于摘要/画像两处高层抽象（不做逐条消息抽取/去重/图谱构建），从而既保留事实完整性与可审计性、又把输入 token 较 Mem0 降低约 80%；其上叠加 nucleus+邻接 episode 聚类的上下文化检索解决会话嵌入差异问题。论文核心论点：对记忆系统而言，“如何召回”比“如何存储”更重要（检索侧优化主导精度）。
- **局限**: (1) 时间推理偏弱：LoCoMo 时间类 0.7352 落后 Memobase 0.8505，依赖更强 eval 模型才提升；(2) 无真正遗忘/记忆巩固机制，原始数据永久累积存在长期可扩展性与隐私顾虑；(3) 不支持程序性记忆/多模态/多语言/严格实时(<200ms) 场景；(4) 检索式架构每轮检索使 prompt 缓存失效、增加延迟（仅 STM 摘要为半稳定前缀，可缓存性 Partial）；(5) 需要专用数据库基础设施（PostgreSQL+Neo4j），部署较重；(6) 跨系统对比混用重跑结果与公开数字，预处理/提示设置可能不一致；token 效率为工作负载相关、应视为方向性结论；消融按维度独立处理、未探索维度间交互效应，且 C1–C4 配置曾用部分题目子集。
- **与其他工作关系**: 属于 D 类（图/类神经/生产级）开源生产系统。直接对标并声称超越 Mem0（A 类，逐条 LLM 抽取+向量/图混合存储，被指成本高、事实漂移）与 Zep（时间知识图谱，关系建模/时间推理强但部署复杂）；与 Memobase、LangMem、OpenAI 原生记忆同台比较。架构定位上：继承 MemGPT 的 OS 式分层记忆思想（但去掉复杂 LLM 驱动的换页决策）、采用 Generative Agents/CoALA 的情景-语义-工作记忆划分与认知科学（Tulving 情景/语义、Atkinson-Shiffrin 多存储模型）；在“压缩 vs 保真”“检索 vs 稳定上下文”张力中，与 Mastra 观测记忆（in-context 压缩、可缓存但不可检索原文，LongMemEval 94.87%）和 MemOS（含参数化/KV-cache 记忆的“记忆操作系统”，LoCoMo 75.80）形成对照，MemMachine 选择应用层、文本 API、保真检索一侧。被同期 2026-04/05 论文（Synthius-Mem 报 LoCoMo 94.37% 超过 MemMachine 91.69% 且指其未报对抗鲁棒性；MemMark 水印、Memory as Metabolism 综述）作为对比基线引用。
- **可复现性**: 可复现性较强：Apache-2.0 开源，代码在 https://github.com/MemMachine/MemMachine （约 3,107 stars、180 forks、30 个发布版本、50 位贡献者），基准脚本/配置/运行说明在仓库 evaluation/ 目录（retrieval_agent 为当前流水线，episodic_memory 为旧版 LoCoMo 流程）。提供 PyPI 包（memmachine-server / memmachine-client）、Docker、MCP server、多框架集成（LangChain/LangGraph/CrewAI/LlamaIndex/AWS Strands/n8n/Dify/FastGPT）。所用数据集 LoCoMo、LongMemEval-S、HotpotQA、WikiMultiHop、EpBench 均公开。论文提示需固定仓库 tag/commit、记录模型版本与 API 设置；评测可在 CPU-only 环境复现（无需 GPU），但需自备 OpenAI/Cohere API key，LongMemEval 摄取约 1.5 小时、需约 50GB 磁盘。社区采纳信号良好（持续活跃、月 PyPI 下载约 1,351 次、有 Discord/DeepWiki）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式流水线为主）。记忆的写/读/更新策略均为规则/流水线式，不用 RL 训练学习记忆管理策略本身。提示词曾用 Agent Lightning 的 APO 做离线自动提示优化，Retrieval Agent 的路由由单次 LLM 分类（启发式 prompt+校准样例+倾向多跳的 tie-breaker），但并非端到端学习的记忆控制策略；将 RL 学习检索策略、预算感知路由列为未来工作。属于 2025–26“启发式管线”一侧，而非 Memory-R1/Mem-α 那种学习记忆管理策略的一代。
- **记忆主体**: 用户中心（user-centric）：核心目标是记住用户信息以实现个性化（情景记忆=用户说过/发生过什么的事实根基，Profile 记忆=用户是谁的画像），评测以 LoCoMo/LongMemEval 等用户对话个性化基准为主。不属于 agent-centric（记住 agent 自身经验以自我提升，如 ReasoningBank/Voyager）。
- **多智能体记忆**: 以单 agent 个性化为主，但架构通过项目级命名空间隔离（org_id/project_id）与会话级隔离（user_id/agent_id/session_id）天然支持多租户/多 agent 部署：多个 agent 可共享项目级记忆、同时在会话级保持隔离，支持 agent 间交接而不丢上下文、共享而非重复生成上下文以降 token。但未实现 G-Memory/MIRIX 式的 insight/query/interaction 分层或显式跨 agent 记忆路由（Retrieval Agent 的路由是查询→策略的路由，非 agent 间记忆路由）。
- **时序推理支持**: 部分支持：为所有 episode 打时间戳，检索时支持时间过滤与按时间顺序排序，可推理事件次序/新近度/时长——属轻量级时间感知，无专用时间记忆模块（区别于 Zep/Graphiti 的事实有效期窗口/双时态边）。实测时间推理为相对弱项（LoCoMo 时间类 0.7352 落后 Memobase；LongMemEval TR 随检索深度提升至 0.932）。增强时间索引/查询扩展列为未来工作。
- **模态**: 纯文本（text-only）。当前仅支持会话文本记忆；多模态（图像、音频、结构化数据、视频）明确列为未来工作。
- **过度个性化/记忆安全风险**: 论文对记忆安全/过度个性化风险讨论有限：主要从隐私与数据主权角度（开源自托管、可用本地 Ollama/vLLM、混合方案保留本地存储）给予用户对数据管道的完全控制；项目/会话隔离支持多租户数据隔离。但未提供针对有害/陈旧/侵入/谄媚记忆的治理机制，也未在 OP-Bench/Causal-LoCoMo 等记忆安全基准上评测。值得注意：同期论文 Synthius-Mem 专门指出 MemMachine “未报告对抗鲁棒性（拒答用户从未透露事实的能力）”，凸显这是其安全维度的空白。
- **冲突/矛盾处理**: 仅在 Profile（语义）层处理冲突：当新信息与已有用户画像矛盾时，系统可更新画像以反映最新状态，支撑 LongMemEval 的 knowledge-update 能力。情景记忆层不做冲突消解（原始 episode 保真并存，由检索+时间排序在读取时呈现），不具备 MEMTRACK/Memory-R1 UPDATE 那样在情景层显式合并矛盾事实的机制。
- **token成本/延迟证据**: 量化效率证据充分：LoCoMo（gpt-4.1-mini）memory 模式输入 token 4.20M vs Mem0 main/HEAD 19.21M——约少 78%（综述口径约 80% fewer input tokens），直接降低推理成本与首 token 时延；agent 模式输入 8.57M token。写入与检索操作较旧版快约 75%。LongMemEval token-精度权衡：Pareto 最优 C12（GPT-5-mini, k=20）仅 2.58M 输入 token 即达 0.922，优于 C7（GPT-5, k=30, 4.03M, 0.912）；达峰值 0.930（C15）需 9.79M token（C12 的 3.8 倍）仅换 +0.8%。Retrieval Agent 单题 token：路由 ToolSelect 约 1,244、直接 MemMachine 路径仅路由开销、ChainOfQuery 多跳约 5,732（受 3 轮上限约束）。可缓存性为 Partial（仅 STM 摘要为半稳定前缀，检索 episode 每查询变化使 prompt 缓存失效）。

**其他信息 / Other**

- **compute_cost**: 无需 GPU：基准测试环境为 Ubuntu 24.04、8 vCPU、16 GiB RAM、CPU-only、Python 3.11/3.12、PostgreSQL+Neo4j；嵌入用 OpenAI text-embedding-3-small、reranker 用 AWS Cohere rerank-v3.5、eval/judge LLM 为 OpenAI gpt-4o-mini/gpt-4.1-mini/gpt-5/gpt-5-mini（均经 API 调用，不本地训练）。不涉及模型训练/微调的 GPU-hours。LongMemEval 摄取约需 50GB 磁盘、约 1.5 小时（关闭句切分可减约 5 倍）。
- **scalability_evidence**: 规模化证据中等：LoCoMo 1,540 题、LongMemEval-S 约 115k token/题×500 题、可处理超出上下文窗口的多会话历史；项目/会话级隔离支持多租户多用户。论文承认更大规模（LongMemEval-M：500 会话、约 1.5M token/题）尚未评测，列为未来工作；原始数据永久累积的长期可扩展性、prompt 缓存优化亦待解决。生产采用信号：仓库约 3,107 stars、30 个发布版本、月 PyPI 下载约 1,351 次。
- **theoretical_grounding**: 弱（无形式化理论分析）。无收敛性保证、无决策过程(MDP/POMDP)形式化、无信息论密度证明。其论证为经验性：基于认知科学（Tulving 情景/语义记忆、Atkinson-Shiffrin 多存储模型）的设计动机，以及在多基准上的系统消融实证（明确将结果定性为“评测设定内的强经验证据，而非普适性能保证”）。
- **biological_inspiration_detail**: 借鉴认知科学记忆分类作为设计动机但承认仅为近似：引用 Tulving（1972）区分情景记忆（绑定时间地点的具体经历）与语义记忆（从经验抽象的通用知识）、Atkinson-Shiffrin（1968）多存储模型（感觉/短期/长期）。据此映射为 STM（短期工作区）、LTM 情景记忆（事实根基）、Profile 语义记忆。论文明确表示“当前实现只是近似而非复制人类记忆过程”，并将认知启发的记忆巩固/遗忘机制列为未来工作；非严格的海马体/互补学习系统/Ebbinghaus 曲线实现。
- **information_density**: 保真优先、低压缩：刻意以未压缩原始 episode 存储换取事实完整性，主要量化对比为 token 数（LoCoMo 输入 token 较 Mem0 少约 78%–80%）。论文将“压缩 vs 保真”列为核心架构张力，并对比 Mastra 观测记忆的 3–40× 压缩（牺牲可检索原文）；未给出 PlugMem 式的信息论压缩比/信息密度分析，其优势来自检索的选择性而非存储压缩。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)
- 发表venue (`venue`)
- 年份 (`year`)


<a id="d8-plugmem任务无关的即插即用插件式记忆模块将情景记忆结构化为以知识为单元的知识中心记忆图"></a>

### D8 PlugMem

*PlugMem（任务无关的即插即用插件式记忆模块；将情景记忆结构化为以“知识”为单元的知识中心记忆图）*


**基本信息 / Provenance**

- **年份**: 2026（arXiv 预印本 2026-02-06 首次公开，v1；已被 ICML 2026 接收）
- **作者/机构**: Ke Yang（杨可，第一作者）、Zixi Chen、Xuan He、Jize Jiang、Michel Galley、Chenglong Wang、Jianfeng Gao、Jiawei Han（韩家炜）、ChengXiang Zhai（翟成祥，通讯/资深作者）。主要单位为伊利诺伊大学厄巴纳-香槟分校（UIUC，TIMAN Group / DAIS 实验室，Han 与 Zhai 课题组）；Michel Galley、Chenglong Wang、Jianfeng Gao 来自微软研究院（Microsoft Research）。属学术界与工业研究院合作。
- **发表venue**: ICML 2026（国际机器学习大会，README 标注已接收）；预印本见 arXiv 2603.03296（cs.CL，兼 cs.AI / cs.IR）。
- **论文链接**: https://arxiv.org/abs/2603.03296
- **代码链接**: https://github.com/TIMAN-group/PlugMem （Apache-2.0 许可证；约 142 stars、14 forks、5 位贡献者、32 次提交；含 OpenClaw / Claude Code 原生插件与 Memory Inspector 可视化 UI；数据与轨迹托管于 Google Drive）

**记忆分类 / Taxonomy**

- **记忆类型**: 三类记忆统一支持（首次在单一任务无关框架中同时覆盖 CoALA 三型）：情景记忆（episodic，原始观测-动作轨迹，作为可验证证据/锚点层）、语义记忆（semantic，事实性“命题/Proposition”，对应“knowing that”/命题型知识）、程序性记忆（procedural，可复用“处方/Prescription”工作流，对应“knowing how”/规范型知识）。核心主张：决策相关信息集中于抽象“知识”而非原始经验，故以语义+程序性知识为主要检索单元，情景层仅作 provenance 证据。
- **记忆结构**: 知识中心记忆图（knowledge-centric memory graph）——以“知识单元（命题/处方）”而非实体或文本块为节点，明确区别于 GraphRAG（实体/文本块为单元）。由三个互联子图构成：情景图 G^E、语义图 G^S、程序图 G^P。语义图：概念（Concept，轻量索引）—mentions→命题（Proposition，重载荷事实块）；程序图：意图（Intent，用户目标键）—solves→处方（Prescription，完整动作工作流），并含意图→处方的层级边；G^S、G^P 均通过 provenance 边 Knowledge—proves→Source 链回情景图 G^E。整体为分层、带溯源的知识图谱（note-graph/KG 变体），而非扁平向量库或原始缓冲。
- **存储后端**: 外部持久化存储。情景记忆以大体量长序列存于磁盘、按 ID 引用；语义/程序节点为带缓存稠密嵌入（默认 NV-Embed-v2）的图节点，开源实现用 ChromaDB 作向量后端、FastAPI 提供 11 个 REST 端点（graph CRUD、trajectory/structured 两种插入模式、retrieve/reason/consolidate/health）。本地推理用 vLLM 托管 Qwen + NV-Embed-v2 嵌入服务。结构化、检索与推理由 LLM 驱动（Qwen2.5-32B/72B-Instruct、GPT-4o）。论文未指定专用图数据库（如 Neo4j），图以应用层数据结构+向量库实现。
- **持久化**: 外部、跨智能体生命周期持久化（durable external store），非参数化：记忆图独立于基座智能体存在，可被任意 LLM 智能体“即插”挂载并跨会话/跨任务复用；支持持久化加载（--load_memory_graph）、只读复用（--read-only-memory）、刷新嵌入等。基座模型权重不更新（无参数化记忆）。RQ3 中 offline 阶段的“新智能体”可直接继承预建记忆图以缓解冷启动。

**核心机制 / Mechanisms**

- **写入/编码**: 结构化模块分两阶段“原始→知识”抽象。①标准化（Standardize）：将异构原始轨迹表示为观测-动作对序列 τ=[(o_t,a_t)]，并由 LLM 信息抽取把每步扩展为结构化五元组 e_t=(o_t, s_t, a_t, r_t, g_t)——状态 s_t 由 (s_{t-1},a_{t-1},o_t) 推导，子目标 g_t 与相对该子目标的奖励 r_t 由 LLM 依据任务指令与局部上下文标注，聚合得情景序列 M_epi=[e_t]。②知识抽取（Extract Knowledge）：语义侧由 LLM 从每个情景单元抽取一组原子命题（如“Tam Sventon，瑞典语 Ture Sventon，是设定于斯德哥尔摩的虚构私家侦探”）并附带概念集做语义标签，施加共指消解、命题去重、长度控制等约束，存入语义图 G^S（命题/概念节点带缓存嵌入，membership 边连概念、provenance 边连情景源）；程序侧先按相邻子目标 g_{t-1}、g_t 相似度低于阈值处切分子轨迹，再由 LLM 为每段诱导紧凑的 (意图, 处方) 对——处方是环境无关的动作工作流（如“要找最低价：用搜索栏搜索→按价格排序→在各变体间核验最小值”），并由 LLM 评估器赋予标量 return 分（评判意图是否达成及执行优劣），存入程序图 G^P（层级边+provenance 边）。
- **检索机制**: 抽象-具体交替（interleaved abstraction-specificity）多跳图检索。给定任务/查询 Q，先由 LLM 检索器判定主导记忆类型（episodic/semantic/procedural）；主要在 G^S、G^P 上检索。流程：将 Q 编码为嵌入 q，对所有低层节点（命题/处方）打分初始化候选集 C_0；在第 t 跳，检索器以 (Q, C_t) 为条件生成抽象查询 q^a_t（语义图中为一组概念、程序图中为一组意图）；q^a_t 与高层节点（概念/意图）匹配，高层节点仅作“路由信号”激活相邻低层节点并并入 C_{t+1}（只保留低层节点为候选，高层不进入结果）；当 |C_t| 超出预算（如 top-K）时按相关性与重要性重排剪枝；多跳迭代直至证据充分或达最大跳数。若优先情景记忆，则用同一流程但最终返回 provenance 关联的 G^E 情景节点。检索预算在各方法间固定以保证公平比较。
- **反思/巩固**: 有显式“原始→抽象知识”的转换（结构化模块本身即一种离线反思/抽象），以及一个测试时（test-time running）推理模块（Reasoning Module）。结构化阶段把冗长情景经验蒸馏为命题与处方（知识级抽象），是 PlugMem 区别于纯检索/RAG 的核心。检索后推理模块再由 LLM 对多个相关但整体冗余的检索结果做聚合与压缩，蒸馏出单一、任务对齐、可直接行动的简洁摘要交给基座智能体——该模块把记忆 token 用量降低一到两个数量级（任务自适应压缩）。此外支持记忆图的 update/consolidate 操作做语义节点合并（见 conflict 字段）。
- **遗忘/更新**: 无 Ebbinghaus 时间衰减；提供 create/retrieve/update/delete 四类记忆图基本操作（正文主要评测 create+retrieve，附录 C.5 评测 update/delete）。update 通过“候选发现+相似度合并”实现软删除/失效：对某语义节点，收集共享至少一个概念/标签的邻居，按嵌入相似度排序、取相似度超阈值 τ 的 top-m（默认 m=1）候选触发 merge——由 LLM 合成一条更优汇总语义并依规则决定是否停用（软删除）原节点（UPDATE_SAME_FACT 停用两者 / SAME_TOPIC_MERGE_WELL 等）。实验（HotpotQA 子图，3413 个语义节点）：τ=0.6 触发 477 次合并、τ=0.7 触发 171 次；τ=0.7 时活跃语义节点 -5.0%、标签 -5.5%、二部图边 -11.3%、标签共现对 -26.5%，下游 QA 性能在波动范围内（EM/F1 61.0/74.39→62.0/74.65），表明合并/更新提升图紧凑度、减少冗余而不损性能。
- **经验回放 (核心主题)**: 核心：以“知识复用”而非原始轨迹回放来改进未来行为。PlugMem 不直接重放原始 episodic 轨迹，而是把过去经验蒸馏为可复用的命题（事实）与处方（技能型工作流），在新任务/新环境中通过抽象-具体检索调用相关知识子图，再经推理模块压缩为可执行指导注入基座智能体。RQ3（WebArena 在线/离线协议）专门验证经验复用：先在 online 集允许插入+检索记忆，再注入少量高质量人类示范（23/18/5 条，类比教程/经验分享），最后在 offline 集仅允许检索——offline 成功率显著提升（尤其多站点组合任务），证明累积的程序性+语义知识能跨任务实例复用、缓解冷启动、支持组合泛化。情景层保留作可验证证据，可在需要时回溯原始轨迹。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric / in-context、检索与提示层面）。所有结构化、检索、推理均由现成 LLM（Qwen2.5-32B/72B、GPT-4o）与嵌入模型（NV-Embed-v2）驱动，基座智能体权重静态、不做梯度更新；知识更新完全在外部记忆图上以增删/合并节点边完成。属“即插即用、零训练挂载”范式。
- **失败学习 (核心主题)**: 有隐式的质量感知机制，但非以“失败专项学习”为核心。程序性处方在抽取时由 LLM 评估器赋予标量 return 分（评判意图是否达成与执行优劣），从而实现“质量感知复用”——低质/未成功的处方可被降权或在更新中淘汰；情景五元组显式记录每步奖励 r_t（相对子目标），蕴含成功/失败信号。但 PlugMem 不像 Reflexion/Retroformer 那样维护专门的失败反思缓冲或负例规则库，也不构建显式失败模式记忆；其改进主要来自“正向知识抽象+质量评分+合并淘汰”，而非对失败轨迹的针对性自反思。
- **技能/程序归纳**: 是——程序性记忆即可复用技能/工作流诱导：把情景轨迹按子目标边界切分为子轨迹，由 LLM 诱导“(意图, 处方)”对，处方为环境无关的动作工作流（关键步骤+因果模式），带 return 评分；以意图为键、处方为解块组织，检索时按意图路由调用，并经推理模块适配为可执行指导。这是其跨任务（尤其 WebArena）泛化的关键载体。
- **在线 vs 离线**: 二者兼具。情景经验可在线（部署/逐 episode）插入并即时检索（WebArena online 阶段、create 操作）；知识抽取（结构化）与图构建对 HotpotQA 等可离线批量进行（build.py 对语料离线建图，README 注“可能需数小时”）；update/merge 既可在线也可离线运行。RQ3 明确区分 online（插入+检索）与 offline（主要只读检索）两阶段以评测知识沉淀与复用。

**评测 / Evaluation**

- **任务领域**: 三个高度异构的智能体任务域，且“同一记忆模块实现不改动”跨域评测：①长程对话问答/多会话长期交互记忆（LongMemEval）；②多跳知识检索与推理（HotpotQA，维基百科多跳 QA）；③交互式网页智能体决策（WebArena，Shopping/GitLab/Multi-site 子集）。覆盖对话、知识 QA、网页 GUI 导航三类。
- **基准**: LongMemEval（长程对话长期记忆）、HotpotQA（多跳知识 QA）、WebArena（真实网页智能体，含 Shopping/GitLab/Multi-site）。基座/嵌入：NV-Embed-v2 检索嵌入；Qwen2.5-32B(Q32)/72B(Q72)、GPT-4o(4o)、对照含 Llama3.3-70B 作结构化/推理与基座 LLM；WebArena 基座智能体为 AgentOccam。README 报告经“轻量任务适配”后在 LongMemEval、HotpotQA 取得 SOTA。
- **报告增益**: （同一模块不改动）①LongMemEval（Table 3，Q72 基座）：PlugMem Acc 75.1，#TokAvg 仅 362.58，Info.Density 1.6e-2；显著超任务无关基线 Vanilla Retrieval 63.6（3742 tok）、A-Mem 61.0；超任务特定 Zep 71.2 与 LiCoMemory 73.0（5915 tok）；远超 No Context 14.8、All Context 62.4（107K tok，密度仅 4.2e-5）。②HotpotQA（Table 4，Q32）：PlugMem EM/F1 61.4/74.1，#TokAvg 仅 81.6，Density 1.4e-1；超任务无关 Vanilla Retrieval 51.7/62.7、A-Mem 43.8/53.6，超任务特定 GraphRAG 55.2/68.6、RAPTOR 56.7/69.7、PropRAG 57.8/72.1、HippoRAG2 60.0/73.3，逼近 Gold Context 上界 69.2/82.1；token 数比这些图法少约一个数量级（81.6 vs 595~806）。③WebArena（Table 5，Q32+4o，on/off 成功率%）：PlugMem Shopping 52.6/58.4、GitLab 51.4/55.2、Multi-site 20.0/21.6，#TokAvg 仅 301，Density 1.4e-3；显著超 AgentOccam 基线（42.1/43.6、37.8/39.2、20.0/15.8）、Vanilla Retrieval（8733 tok）、A-Mem（20516 tok），并大幅超任务特定 AWM（26.3/28.2、27.0/27.3，且其密度为负 -7.9e-4）。④README“轻量任务适配”后：LongMemEval 90.2 Acc、HotpotQA 79.1 F1 / 91.1% LLM-Judge Acc，均为 SOTA。⑤推理模块把记忆 token 用量降低一到两个数量级（如 LongMemEval 去掉推理 #Tok 由 362→9478）。各任务 PlugMem 均取得最高信息密度（bit/token）。
- **对比基线**: 三类：①Vanilla（无外部记忆）：No Context、All/Gold Context、AgentOccam（WebArena 基座）；②任务无关：Vanilla Retrieval、Vanilla RAG、A-Mem；③任务特定：LongMemEval 上 Zep、LiCoMemory；HotpotQA 上 GraphRAG、RAPTOR、PropRAG、HippoRAG2；WebArena 上 AWM。消融对照：No Structuring / No Retrieval / No Reasoning / No Human Demo。

**分析 / Analysis**

- **关键创新**: 提出首个真正“任务无关、即插即用”的插件式长期记忆模块，把异构情景经验抽象为以“知识单元（命题+处方）”为节点的知识中心记忆图——明确将“知识”而非实体/文本块作为记忆访问与组织的基本单元（区别于 GraphRAG），并在认知科学（情景/语义/程序记忆三分）指导下统一支持三型记忆；配套提出统一的信息论“记忆信息密度（bit/token）”评估框架（基于 PMI 的决策信息增益 / 记忆 token），首次让不同记忆设计可跨任务公平比较“效用-成本”权衡。单一实现不改动即在对话、多跳 QA、网页三类异构任务上同时超越任务无关与任务特定基线，并取得最高信息密度。
- **局限**: ①重度依赖 LLM 做结构化抽取、检索路由、推理压缩与合并判定，结构化与推理推理成本与延迟较高（HotpotQA 建图“可能需数小时”），且抽取/评分质量受底层 LLM 影响；②无真正的时间衰减/遗忘机制，update/delete 为软删除合并，且 update/delete 仅在 HotpotQA 语义子图上小规模验证（作者称选其因事实稳定、子图轻量便于快速迭代）；③消融显示检索是决定性瓶颈，记忆只有被有效检索才有用，结构化/推理的收益依赖检索质量；④主要为纯文本，多模态/视觉/具身记忆未涉及；⑤过度个性化、记忆安全、隐私治理等负面维度未讨论；⑥SOTA（90.2/79.1）需“轻量任务适配”，纯任务无关配置略低。
- **与其他工作关系**: 属本研究 D 类“图/神经启发/生产级”记忆方向。理论上承接 Tulving、Squire、Atkinson-Shiffrin 的情景/语义/程序记忆三分与互补学习系统思想，将“命题型/规范型知识”落到图节点。明确自定位为“面向记忆管理的知识中心 GraphRAG”，与 GraphRAG（Edge 2025，实体/文本块为单元）、RAPTOR、PropRAG、HippoRAG2（本研究 D1，均以情景/文本为主要单元、PPR 检索）形成对比并在 HotpotQA 上超越之。其程序性“(意图,处方)”技能诱导与 AWM（本研究 C2，Agent Workflow Memory）、ReasoningBank（本研究 A6）同源但更通用、任务无关——论文 Table 1 用 M2K（经验转可复用抽象知识）、KaU（知识作记忆单元）等维度系统对比 Vanilla Retrieval/RAG、GraphRAG、A-Mem（本研究 B4）、Zep、MemoryOS（本研究 B7）、HippoRAG2、AWM、ReasoningBank，主张唯有“把经验转为知识”才支持跨任务泛化。语义提取的“原子命题+概念”做法与 A-Mem 的 Zettelkasten 笔记、PropRAG 的命题路径相呼应；情景标准化的五元组(o,s,a,r,g)借鉴智能体轨迹形式化。已被 CommitDistill、AutoMEM（跨场景泛化诊断）、Auto-Dreamer 等 2026 后续工作作为“类型化蒸馏知识记忆”代表引用。
- **可复现性**: 复现性较好：代码开源于 GitHub（TIMAN-group/PlugMem，Apache-2.0，约 142 stars、14 forks、5 贡献者），提供 6 行代码挂载接口（MemoryGraph + Memory + insert + retrieve_and_reason）、FastAPI/REST 层、FakeLLM/FakeEmbedder 测试套件、OpenClaw 与 Claude Code 原生插件及 Memory Inspector 可视化 UI（图视图/浏览视图/会话回放）；提供三任务（WebArena/LongMemEval/HotpotQA）的逐步复现脚本与本地推理（vLLM Qwen + NV-Embed-v2）部署脚本；释放全部三任务的智能体轨迹、记忆图工件与 WebArena 人类示范（数据托管 Google Drive，CC BY 4.0）。所用基准均公开；已被 ICML 2026 同行评审接收。社区采用信号上升（已被多篇后续工作引用、提供编码智能体插件）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式/LLM 提示管线，非 RL 学习记忆策略）。何时/写什么/如何检索/是否合并，均由固定流程+LLM 提示判定（如检索类型由 LLM 判定、合并由相似度阈值 τ+LLM 规则判定、处方 return 由 LLM 评估器打分），不用强化学习训练记忆管理策略本身。属 2025–26“学习记忆控制”代际之前的（LLM 驱动的）启发式范式，与 Memory-R1、Mem-α、Mem-π、Auto-Dreamer（用 RL/GRPO 学记忆策略）形成对比；但其 LLM 化的路由/评分/合并比纯固定规则管线更灵活。
- **记忆主体**: 兼具用户中心与智能体中心，且额外覆盖知识/文档中心，是其“任务无关”的体现：LongMemEval 上记住用户事实/偏好做个性化对话（用户中心，semantic）；WebArena 上记住智能体自身的网页操作经验/技能做自我改进（智能体中心，procedural）；HotpotQA 上把维基语料蒸馏为命题做知识检索（知识中心，semantic）。同一框架以统一的命题/处方知识单元服务三种主体。
- **时序推理支持**: 弱/有限。情景标准化保留轨迹的步级时序（observation-action 序列、状态演化），LongMemEval 评测含时序型对话记忆；update/merge 在冲突时“偏好更新者 Information2（更新近的版本）”体现一定时效优先。但 PlugMem 不像 Zep/Graphiti 那样维护显式事实有效期窗口、事件时序日历或双时间轴；命题节点不带显式有效性区间，时序推理非其主打能力。
- **模态**: 纯文本（text-only）。命题、概念、意图、处方、情景轨迹均为文本；不涉及视觉/截图/视频/具身多模态记忆（README 提到的 vision-language 集成属外部能力，论文实验为文本）。
- **冲突/矛盾处理**: 有专门的更新/合并冲突处理（区别于纯遗忘）。update 例程对相似度超阈值 τ 的语义节点对触发 LLM 合并，提示中显式编码冲突规则：CaseA「UPDATE_SAME_FACT」——两条本质同一事实且后者更正/细化前者，则合成一条并软删除两个原节点；CaseB「SAME_TOPIC_MERGE_WELL」——同主题强相关可自然统一为汇总，且“若两记忆冲突，偏好更新近的 Information2”；另设「WEAK_RELATED_STITCH_RISK」避免牵强拼接。合并不臆造新事实。与 Memory-R1 的 UPDATE、MEMTRACK 的冲突跟踪同类，但实现为相似度+LLM 规则的离线/按需合并，且仅在 HotpotQA 语义子图小规模验证。
- **token成本/延迟证据**: 有明确 token 效率证据（核心卖点之一），但以 token 用量与信息密度而非延迟百分比报告。注入基座智能体上下文的平均记忆 token #TokAvg 远低于基线：LongMemEval 362.58（vs All Context 107K、Vanilla Retrieval 3742、LiCoMemory 5915，约低一到两个数量级）；HotpotQA 81.6（vs 595~806，低约一个数量级）；WebArena 301（vs Vanilla Retrieval 8733、A-Mem 20516，低 1~2 个数量级）。推理模块去除后 token 暴涨（LongMemEval 362→9478），证实其压缩贡献。信息密度（bit/token）各任务均为最高（如 LongMemEval 1.6e-2 vs 1e-3 级、HotpotQA 1.4e-1 vs 1e-2 级、WebArena 1.4e-3 vs 1e-6/1e-7 级）。论文未给出 p95 延迟或绝对 token 节省百分比（不同于 Mem0/Zep 的延迟百分比口径）。仓库另含 scripts/bench 的 token 成本基准（按 EXPOSED/INTERNAL 区分智能体热路径 token）。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)
- 多智能体记忆 (`multi_agent_memory`)
- 过度个性化/记忆安全风险 (`over_personalization_risk`)


## E. 认知架构框架 (Cognitive-architecture frameworks)


<a id="e1-coala面向语言智能体的认知架构--cognitive-architectures-for-language-agents论文标题cognitive-architectures-for-language-agents非系统方法而是一个概念性蓝图框架配套资源别名coalaawesome-language-agents基于-coala-框架的语言智能体清单仓库"></a>

### E1 CoALA

*CoALA（面向语言智能体的认知架构 / Cognitive Architectures for Language Agents）；论文标题《Cognitive Architectures for Language Agents》；非系统/方法，而是一个概念性蓝图框架；配套资源别名：🐨CoALA、awesome-language-agents（基于 CoALA 框架的语言智能体清单仓库）。*


**基本信息 / Provenance**

- **年份**: 2023（arXiv 预印本 v1 2023-09-05；经 TMLR 期刊接收并发表于 2024）
- **作者/机构**: Theodore R. Sumers、Shunyu Yao（姚顺雨，二人并列第一作者，署名顺序掷硬币决定）、Karthik Narasimhan、Thomas L. Griffiths，均来自普林斯顿大学（Princeton University）。Thomas L. Griffiths 为资深作者（认知科学方向）。
- **发表venue**: TMLR 2024（Transactions on Machine Learning Research，期刊；Semantic Scholar 记为 Trans. Mach. Learn. Res. 2024）；属于综述/立场性质论文（publicationTypes: JournalArticle, Review）。arXiv DOI: 10.48550/arXiv.2309.02427。
- **论文链接**: https://arxiv.org/abs/2309.02427 （TMLR OpenReview 公开版；arXiv DOI: https://doi.org/10.48550/arXiv.2309.02427）
- **代码链接**: https://github.com/ysymyth/awesome-language-agents （官方配套仓库，约 1,000+ stars，MIT 许可，TeX；为基于 CoALA 框架整理的语言智能体清单与 300+ 条相关文献 BibTex 集合，而非 CoALA 框架本身的可运行参考实现——CoALA 是概念蓝图，无独立代码实现）
- **引用数**: 约 391 次（Semantic Scholar 实时数据，2026-06 核实，CorpusId 261556862；referenceCount=235）。作为 LLM 智能体记忆与认知架构领域的主要分类学参照框架，影响力高。

**记忆分类 / Taxonomy**

- **记忆类型**: 四类记忆的统一分类学（本研究 memory_type 字段的主分类透镜来源）：(1) 工作记忆 (working memory)——当前决策周期内的活跃符号变量；(2) 情景记忆 (episodic memory)——过往决策周期的经历/轨迹；(3) 语义记忆 (semantic memory)——关于世界与自身的知识/事实；(4) 程序记忆 (procedural memory)——分为隐式（LLM 权重）与显式（智能体代码）两种形式。该框架借鉴 Soar 认知架构与 Atkinson-Shiffrin、Baddeley-Hitch 等心理学记忆理论，将上述类别系统化用于刻画语言智能体。
- **记忆结构**: 模块化的多记忆体系结构（非具体数据结构，而是抽象蓝图）：工作记忆是『跨 LLM 调用持久的数据结构』（一组符号变量），充当连接 LLM、长期记忆与接地接口的中枢；长期记忆按情景/语义/程序三类组织。框架对每类记忆的底层实现（原始缓冲、向量库、知识库、代码库、模型权重）持开放态度——明确指出工作记忆模块本身的设计是未来研究方向。
- **存储后端**: 框架层面不规定具体后端，而是给出实现范畴：情景记忆可为输入输出对/事件流/游戏轨迹；语义记忆可为非结构化文本库（如维基百科）或结构化知识；程序记忆=LLM 参数（隐式）+ 智能体源代码（显式）。论文以 Voyager（代码技能库经稠密检索）、Generative Agents（事件记忆流）、DocPrompting（文档语义记忆）等具体系统举例说明可能的后端。
- **持久化**: 三种持久化形态并存且被显式区分：工作记忆为短期/易失（仅维持当前决策周期与跨调用的活跃状态）；情景与语义记忆为外部持久长期存储（可读写、可初始为空）；程序记忆中 LLM 权重为参数化持久存储、代码为显式持久存储。论文强调 LLM 本身是无状态 (stateless) 的，智能体通过这些记忆模块获得跨步骤、跨生命周期的状态维持。

**核心机制 / Mechanisms**

- **写入/编码**: 框架将『写入记忆』统一定义为一类内部学习动作 (learning action)，并系统枚举其谱系：(1) 写情景记忆——将工作记忆中的新经历/轨迹存入情景记忆（如 RL 的轨迹存储、Generative Agents 的事件流）；(2) 写语义记忆——用 LLM 对原始经历进行推理 (reasoning) 后将所得推断/知识写入（如 Reflexion 反思失败后存『厨房里没有洗碗机』之类语义知识，机器人用 VLM 构建语义地图）；(3) 写程序记忆之 LLM 参数——通过有监督/模仿/RL/RLHF/AI 反馈等微调（如 XTX 周期性在高分轨迹上微调小模型）；(4) 写程序记忆之代码——更新推理过程（如 APE 从样例归纳提示指令）、更新接地技能（如 Voyager 维护代码技能课程库）、原则上还可更新检索/学习/决策过程本身。论文区分『推理』(读写工作记忆、蒸馏洞见) 与『学习』(把结果写入长期记忆)，二者配合完成编码。
- **检索机制**: 框架将检索 (retrieval) 定义为一类内部动作：『从长期记忆读信息到工作记忆』，与『推理』(在工作记忆内读写) 和『接地』(与外部环境交互) 并列。具体实现按记忆与信息类型而定，可为基于规则、稀疏 (如 BM25) 或稠密 (embedding) 检索。论文给出范式实例：Voyager 用稠密检索从程序记忆加载代码技能；Generative Agents 用近因性 (recency, 规则)·重要性 (importance, 推理)·相关性 (relevance, 嵌入) 三因子组合从情景记忆检索事件；DocPrompting 从语义记忆检索库文档辅助代码生成。框架并指出『自适应、情境特定的回忆 (adaptive, context-specific recall)』在语言智能体中仍研究不足，并在第 6 节倡议将决策与检索做原则性整合（借鉴人类记忆-决策耦合的心理学模型）作为重要未来方向。
- **反思/巩固**: 框架通过『推理动作 (reasoning)』刻画原始经历→高层洞见的转化：推理同时读写工作记忆，可对最近观察、最近轨迹或检索回的长期记忆进行总结与蒸馏 (summarize and distill insights)；其产物既可用于决策（作为后续 LLM 调用的上下文），也可用于学习（写入长期记忆，尤其写入语义记忆）。论文以 Reflexion 对失败轨迹的反思、Generative Agents 的反思综合为典型『写语义记忆以增量构建世界知识』的反思-巩固范例。CoALA 本身不规定固定触发机制，而是把何时反思/学习交由决策过程编排（如可将学习推迟到交互结束时再总结存储）。
- **遗忘/更新**: 框架明确承认这是欠研究的薄弱环节：论文坦言『迄今讨论多聚焦于向记忆添加 (adding)，而修改 (modifying) 与删除 (deleting，即一种 unlearning) 在近期语言智能体中研究甚少』，并将其列为未来方向。提及删除无用记忆条目以实现『遗忘/反学习』(引 Nguyen et al. 2022)，但 CoALA 不提供具体的衰减/合并/去重/冲突消解机制——它只是把这些操作纳入可能的学习动作空间。
- **经验回放 (核心主题)**: 框架把经验复用统一纳入『情景记忆的写入与检索』循环：智能体把过往决策周期的经历/轨迹写入情景记忆，之后在新决策周期的规划阶段将相关情景检索回工作记忆，作为推理或决策的范例与依据 (retrieved later as examples and bases for reasoning or decision-making)。论文区分多种复用形态：非参数式（直接把轨迹/范例检索进上下文，如 Generative Agents、ReAct 的 in-context 示例）与参数式（在高分轨迹上微调，如 XTX 把情景记忆中高分轨迹周期性蒸馏进小模型策略）。框架强调语言智能体相比传统 RL 的独特优势在于可『以语言形式存储任务相关经验』，比参数更新更廉价快速，并能复合多种学习形式实现自我改进（以 Generative Agents 为复合学习范例）。

**学习维度 / Learning**

- **学习范式**: 混合 (hybrid)：CoALA 的『学习』定义同时涵盖非参数（写情景/语义记忆、改提示与代码技能等 in-context / prompt-level / 代码层学习）与参数（微调 LLM 权重的程序记忆）两类，并明确指出语言智能体的独特之处在于可在多样的学习过程中选择，而不像传统 RL 固定一种（Q-learning/PPO/A3C）。论文强调存语言比更新参数更便宜更快，且可复合多种学习方式。
- **失败学习 (核心主题)**: 框架以 Reflexion (Shinn et al. 2023) 为典型范例纳入失败学习：用 LLM 对失败的回合 (failed episodes) 进行反思，把所得推断（如『厨房里没有洗碗机』）作为语义知识存入语义记忆，并附加到后续回合的 LLM 上下文以改进求解。在 CoALA 抽象层，这被表述为『以推理处理失败轨迹 → 将洞见写入语义记忆』的学习动作链路。CoALA 本身不提出新的失败检测算法，而是把 Reflexion 式的失败反思作为『写语义记忆』学习谱系中的一个实例归类。
- **技能/程序归纳**: 支持，纳入『更新程序记忆中的代码（接地技能）』这一学习动作：智能体可创建新的代码化技能写入程序记忆并经检索调用。典型范例为 Voyager——维护一个代码技能课程库 (curriculum library)，通过稠密检索加载技能与 Minecraft 交互。论文指出当前方法局限于创建与外部环境交互的新代码技能，而更新检索/推理/学习/决策等其他程序仍属未充分探索（甚至无已知工作）。
- **在线 vs 离线**: 框架两者兼容、不作硬性规定：把学习视为决策周期内可随时执行的内部动作，可在部署中在线逐回合写入记忆（如 Generative Agents、Reflexion 的回合级反思），也可批量离线微调（如 XTX 周期性在高分轨迹上微调小模型）。论文指出当前微调成本高故多采用预设的『学习时间表 (learning schedules)』，并设想未来训练更高效时智能体可自主决定何时、如何微调。

**评测 / Evaluation**

- **任务领域**: 作为统一框架横跨多领域（通过回顾性综述覆盖），论文举例涵盖：机器人/具身（SayCan 厨房机器人，551 个接地技能）、纯推理（ReAct、Tree of Thoughts、RAP 的 24 点游戏/积木搭建）、数字环境与网页/工具使用（ReAct 的维基百科 API、文本游戏、网站；工具调用）、Minecraft 开放世界（Voyager）、社会行为模拟（Generative Agents）、检索增强 QA（NLP 检索方法）、个性化零售助手（设想的应用示例）。本身非针对单一领域的系统。
- **基准**: 不适用（N/A）：CoALA 是概念框架与综述，不报告自身在标准基准上的实验。论文以表格（Table 2）将 SayCan、ReAct、Voyager、Generative Agents、Tree of Thoughts 等代表性智能体按记忆模块/动作空间/决策过程三维度归类对照，但不进行新的基准评测。所引系统各自的基准（如 ALFWorld、Minecraft、24 点游戏等）仅作举例说明。
- **报告增益**: 不适用（N/A）：作为分类学/蓝图框架与综述，CoALA 不提出可量化的性能增益、不与基线对比分数，也不报告 token/延迟数据。其『贡献』是定性的——提供统一术语与三维度（记忆-动作-决策）结构，用以回顾性组织既有智能体并前瞻性识别未充分探索的方向（如检索与决策的原则性整合、修改/删除记忆的反学习、决策过程从单步生成走向 propose-evaluate-select、元学习改写智能体代码等）。论文定性指出更强智能体（Voyager、Generative Agents）拥有更大动作空间因而面临更复杂的决策问题、依赖更定制化的决策过程。
- **对比基线**: 不适用（N/A）：无实验基线对比。在框架层面，CoALA 把语言智能体置于历史脉络中作对照——与符号 AI 的产生式系统 (production systems)、Soar 等经典认知架构对比（指出 LLM 可视为对文本的概率化产生式系统），并相对于既有的纯检索增强语言模型 (RAG，只读人写语料) 强调记忆增强智能体可自主读写自生成内容。

**分析 / Analysis**

- **关键创新**: 提出 CoALA 概念框架：借鉴产生式系统与 Soar 等认知架构，用三个维度统一刻画与设计通用语言智能体——(1) 模块化记忆（工作 + 情景/语义/程序长期记忆）；(2) 结构化动作空间（内部动作：推理、检索、学习；外部动作：接地）；(3) 广义决策过程（规划阶段的 propose-evaluate-select + 执行阶段的决策周期循环）。其最重要贡献是为本领域提供了被广泛采用的记忆四分类学 (episodic/semantic/procedural/working) 与统一术语，从而能简洁、结构化地比较异质智能体并识别未充分探索的方向。
- **局限**: (1) 纯概念性框架、无可运行实现与实验验证，落地需逐组件具体化；(2) 自承多处机制欠研究：记忆的修改/删除/反学习、自适应检索、元学习改写检索/学习/决策过程等基本无现成工作；(3) 改写程序记忆（代码/学习/决策过程）风险高，可能引入 bug 或破坏对齐，论文反复警示其安全性；(4) 代码 vs LLM 两类程序记忆各有脆弱性（代码僵硬、LLM 不可解释）；(5) 决策过程复杂度与动作空间规模存在权衡，复杂智能体依赖手工定制决策过程，泛化性受限；(6) LLM 调用慢且算力昂贵，元推理 (metareasoning) 控制成本仍是开放问题；(7) 作为 2023 年的综述，未覆盖 2024-2026 的学习式记忆控制、时序记忆图谱、多智能体记忆路由等新进展（这些恰是后续工作填补的方向）。
- **与其他工作关系**: 本研究 E 类（认知架构框架）的核心条目，是整个研究 memory_type 字段的主分类透镜来源——其 episodic/semantic/procedural/working 四分类被本库几乎所有条目沿用。它向上承接符号 AI 的 Soar 认知架构与心理学记忆理论；向下作为综述把本库多个系统纳入统一坐标：明确把 Generative Agents (B1) 归为情景记忆 + 反思写语义记忆 + 复合学习的范例、把 Reflexion (A 类自我反思失败学习) 归为『以推理处理失败轨迹→写语义记忆』、把 Voyager (C1) 归为程序记忆中代码技能库的典型、把 ReAct/SayCan/Tree of Thoughts/RAP 等按动作空间与决策过程归类。相对纯检索增强 RAG（如 D 类 HippoRAG）它强调智能体可自主读写自生成内容而非只读人写语料。后续 2024-2026 的 Memory-R1、Mem-α（学习式记忆控制）、Zep/Graphiti（时序记忆）、G-Memory/MIRIX（多智能体记忆路由）等正是在 CoALA 指出的『检索-决策整合、修改/删除记忆、元学习记忆管理』等空白方向上的推进。
- **可复现性**: 可复现性概念高、实现复现不适用：作为框架/综述无需复现实验。官方配套 awesome-language-agents 仓库（约 1,000+ stars，MIT，含 CoALA.bib 300+ 条文献）公开可用，社区采用度高，CoALA 术语被大量后续论文引用为标准分类学（约 391 次引用）。框架本身的『可复现性』体现为其抽象易于被各系统映射套用，但它不附带可运行的 CoALA 智能体代码。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式/设计者指定为主）：CoALA 的记忆管理（何时写/读/学习、检索与决策过程）在框架中由设计者编写的智能体代码（程序记忆的显式部分）与决策过程编排，属确定性规则 + LLM 推理。论文将『让智能体元学习改写自己的检索/学习/决策过程』（即学习记忆管理策略本身）明确列为理论上可能但风险高、迄今无已知实现的未来方向——这正是 2025-26 代学习式记忆控制工作（Memory-R1、Mem-α 等）相对 CoALA 的代际推进。
- **记忆主体**: 智能体中心 (agent-centric) 为主：CoALA 关注智能体记忆自身的经历（情景）、世界与自身知识（语义）、可复用程序与技能（程序），以支撑自我改进与长期决策。也兼容用户中心的个性化应用（论文以个性化零售助手为设计示例，用情景记忆存每位顾客的历史交互、用语义记忆存商品库），但框架本身不偏向用户个性化范式。
- **多智能体记忆**: 以单智能体为默认建模对象：CoALA 的记忆-动作-决策三维度针对单个语言智能体定义。论文在工业落地讨论中设想维护『公司级语言智能体库』以复用组件、统一客户体验，但未提出跨智能体共享/路由记忆的分层机制（这区别于 G-Memory、MIRIX 等后续多智能体记忆架构）。
- **模态**: 以文本为主、显式兼容多模态：框架强调信息『主要是文本但也允许其他模态』，并通过接地 (grounding) 过程把视觉/音频/触觉等感知输入经描述/captioning 或视觉语言模型 (VLM) 转译为文本观察进入工作记忆；举例包含具身机器人（物理环境感知）与用 VLM 构建语义地图。故为文本中心的多模态接入框架。
- **过度个性化/记忆安全风险**: 未作为专门主题处理（早于 OP-Bench 等记忆安全基准）：论文未讨论有害/陈旧/侵入式/谄媚式个性化记忆的治理或隐私基准。但其从动作空间安全角度提出相关警示：『学习』动作（尤其程序记忆的删除与修改）可致内部损害，『接地』动作（如 bash 的 rm、有害言论、机器人持刀）可致外部损害；当前安全措施多为任务特定启发式（过滤关键词、限制环境等），并呼吁对动作空间作最坏情况预测与防护。
- **冲突/矛盾处理**: 未提供专门机制：CoALA 把矛盾事实的处理隐含在『修改/更新长期记忆』这一被自承欠研究的学习动作中，未给出显式的冲突消解/合并/失效流程（区别于后续 Memory-R1 的 UPDATE、MEMTRACK）。论文将记忆的修改与删除整体列为未来工作。
- **token成本/延迟证据**: 无量化数据：作为框架/综述不报告自身或对比系统的 token/延迟节省数字。仅作定性论述：以语言形式存储经验比更新参数更廉价快速；LLM 调用慢且算力昂贵，因而倡议用元推理 (metareasoning) 自适应分配推理/搜索预算以平衡计算成本与改进收益（多数方法固定搜索深度，而人类似乎自适应分配）。

**其他信息 / Other**

- **cluster**: E. 认知架构框架 (Cognitive-architecture frameworks)

**不确定字段 / Uncertain**

- 时序推理支持 (`temporal_reasoning_support`)


<a id="e2-memorag全局记忆增强检索的下一代-rag-框架全称-memorag-boosting-long-context-processing-with-global-memory-enhanced-retrieval-augmentation"></a>

### E2 MemoRAG

*MemoRAG（全局记忆增强检索的下一代 RAG 框架；全称 MemoRAG: Boosting Long Context Processing with Global Memory-Enhanced Retrieval Augmentation）*


**基本信息 / Provenance**

- **年份**: 2024（arXiv 预印本 2024-09-09 首次公开 v1；正式发表于 WWW 2025）
- **作者/机构**: Hongjin Qian（钱泓锦，第一作者，与 Peitian Zhang 并列共同一作）、Zheng Liu（刘政，通讯作者）、Peitian Zhang、Kelong Mao（毛科龙）、Defu Lian（连德富）、Zhicheng Dou（窦志成）、Tiejun Huang（黄铁军）。主要单位：北京智源人工智能研究院（BAAI，第一作者与通讯作者所属）、北京大学（PKU，Hongjin Qian 与 Tiejun Huang）、香港理工大学（HK PolyU，Zheng Liu）、中国人民大学高瓴人工智能学院（RUC，Kelong Mao、Zhicheng Dou）、中国科学技术大学（USTC，Defu Lian）。
- **发表venue**: WWW 2025（The ACM Web Conference 2025 / TheWebConf 2025，2025 年 4 月 28 日–5 月 2 日，澳大利亚悉尼）；正式收录于 Proceedings of the ACM on Web Conference 2025，DOI 10.1145/3696410.3714805，ISBN 979-8-4007-1274-6/25/04；DBLP 索引 conf/www/Qian0ZMLD025。预印本见 arXiv 2409.05591（cs.CL）。
- **论文链接**: https://arxiv.org/abs/2409.05591 （正式版 DOI https://doi.org/10.1145/3696410.3714805）
- **代码链接**: https://github.com/qhjqhj00/MemoRAG （官方开源，Apache-2.0 许可；约 2.0k stars。后续作者推出重构版 MemoRAG v2 / memorag pip 包，README 标注 “Empowering RAG with a memory-based data interface for all-purpose applications”）
- **引用数**: 约 104 次（Semantic Scholar 实时查询，CorpusId 272525276；发表后引用增长较快，在 2024–25 长上下文/记忆增强 RAG 方向影响力中上）

**记忆分类 / Taxonomy**

- **记忆类型**: 语义记忆 / 工作记忆混合，但本质是“全局上下文记忆（global context memory）”：MemoRAG 用一个轻量长程记忆模型把整段长上下文（如整本书、多年财报）压缩成全局记忆表征 θ_mem，类似人类先通读全文形成的高层语义概览。它存储的是“当前长输入文档/数据库的全局语义”而非智能体自身经验轨迹，故偏 CoALA 中的语义/工作记忆范畴；不实现情景记忆或程序性技能记忆。
- **记忆结构**: 参数化/激活态的“压缩 KV 缓存”记忆，而非外部向量库或知识图谱：记忆以 Transformer 各层的键值缓存（Key-Value cache）形式存在。具体做法是在原始 token 序列中每隔一个工作窗口 l 插入 k 个特殊“记忆 token（memory tokens x^m，k≪l）”，并为其新建独立的 Q/K/V 投影矩阵，将长上下文渐进压缩进记忆 token 的 KV 缓存（K^m_cache, V^m_cache），普通 token 的 KV 缓存随后丢弃。θ_mem = [K^m_cache, V^m_cache]。这是一种“紧凑全局记忆（compact global memory）”结构，区别于扁平缓冲、向量库与图。
- **存储后端**: 模型激活态/KV 缓存（在 GPU 显存中，可选 offload 落盘复用）：记忆即压缩后的记忆 token KV 缓存，随记忆模型（基于 Mistral-7B-Instruct-v0.2-32K 或 Qwen2-7B-Instruct 训练）一同存在。检索阶段使用标准稠密检索器+向量库（论文 baseline 用 BGE-M3、Stella-v5、Jina-emb-v3；效率分析用 FAISS）在原始长上下文上做证据检索。生成器默认复用记忆模型底座 LLM，长上下文实验中改用 Phi-3-mini-128K-instruct 以避免截断。算法第 11 行明确支持把 θ_mem“Memory Offload 落盘”以便未来任务复用。
- **持久化**: 以“会话/文档级”持久化为主，介于 in-context 与参数化之间：记忆是对“当前长输入上下文”的压缩缓存（针对一篇长文档/一个数据库形成一次），可在该上下文下跨多个查询复用，并可 offload 落盘供未来对同一上下文的任务重用；但它不是跨任务累积智能体经验的长期记忆，也不直接改写底座 LLM 的预训练权重（记忆参数仅为新初始化的记忆 token 投影矩阵，经训练得到）。属“紧凑激活态记忆 + 可选磁盘缓存”。

**核心机制 / Mechanisms**

- **写入/编码**: 通过“记忆形成（Memory Formation）”在 prefill 阶段把长上下文 X=C+辅助文本编码进记忆 token 的 KV 缓存：LLM 工作窗口长 l，每个窗口后插入 k 个记忆 token（k≪l，压缩比 β=l/k∈{4,8,16,32,64}），用专门新建的 W_{Q^m}/W_{K^m}/W_{V^m} 计算记忆 token 的 Q/K/V，注意力对 [K^m_cache; K; K^m] 与 [V^m_cache; V; V^m] 做软压缩，处理完每个窗口即更新记忆 token 缓存 K^m_cache/V^m_cache 并丢弃普通 token 的 KV 缓存以省显存，最终 θ_mem=[K^m_cache, V^m_cache]，约 β× 降低显存。例如 128K 上下文 LLM 在 β=64 时可处理约 8M token。记忆模型经三阶段训练学会“写入”：①预训练（仅优化新初始化的记忆投影矩阵、冻结底座 LLM，目标为基于记忆 token + 当前上下文预测下一 token 的交叉熵 L_pre）；②SFT（用强 LLM 生成、人工复核精修的任务特定数据，学会基于全局记忆生成任务线索 y，损失 L_SFT）；③RLGF 偏好对齐。区别于把经验抽成三元组/笔记的方法，MemoRAG 把“整段上下文语义”压缩进激活态记忆。
- **检索机制**: “记忆生成线索→线索驱动检索”的两段式记忆增强检索（核心创新）：不直接用原始查询 q 检索，而是先用记忆模型基于全局记忆生成“草稿答案/线索 y=Θ_mem(q|θ_mem)”——这些 clue 是基于压缩记忆草拟的、可能不精确但揭示真实信息需求并可直接对齐源文本的中间产物（如更显式的代理查询、推理中间步、关键概念点）。再以 y 为查询用检索器 Γ 从原始长上下文 C 中定位精确证据 E=Γ(y,C)，最后生成器 Θ 据 (q,E) 产出最终答案 Y=Θ(q,E|θ)。形式化（式 2）：Y=Θ(q,E|θ), E=Γ(y,C), y=Θ_mem(q|θ_mem)。其价值在于跨越查询与证据之间的语义鸿沟，尤其适用于查询无显式检索意图（如“总结主要人物关系”“按全局信息聚合”）或上下文非结构化的场景。检索本身用标准稠密相似度检索（无图遍历/PPR/惊奇分段等特殊机制）。
- **反思/巩固**: 存在“记忆形成 = 全局语义巩固（consolidation/abstraction）”：记忆模型把整段长上下文压缩抽象成高层全局语义表征 θ_mem（类比人类通读后形成的整体认知），这是一种把“原始长文本→高层概要记忆”的抽象。但它不是 Reflexion 式“对过往轨迹做事后反思生成经验教训”的反思机制；线索生成 y 也是从已巩固的全局记忆中“回忆/recall”草稿答案，而非对失败经验的反思。巩固发生在编码阶段（一次性压缩），随后稳定复用；不随更多查询持续重写记忆。
- **遗忘/更新**: 无 Ebbinghaus 时间衰减、无显式 ADD/UPDATE/DELETE 编辑或去重/冲突消解机制。记忆是针对某一长上下文一次性形成的压缩缓存；“遗忘”仅以隐式形式体现为压缩损失——压缩比 β 越大保留语义越少（普通 token 的 KV 缓存在 prefill 中被主动丢弃），消融显示 β 增大性能下降但在 β=32 趋稳。无随时间的记忆衰减或对旧事实的失效/合并/矛盾处理；新上下文需重新形成记忆。
- **经验回放 (核心主题)**: 不适用（非智能体经验回放范式）。MemoRAG 复用的是“当前长上下文的全局记忆”而非“智能体自身过往任务轨迹/技能”：它不维护跨任务的轨迹回放缓冲，不做技能复用，也不用过去 episode 改进未来决策。它属于长上下文处理 / 记忆增强 RAG 方向，与 ReasoningBank/Voyager/ExpeL 等以重用经验轨迹自我改进的范式正交。唯一的“复用”是把同一上下文的 θ_mem offload 落盘，供该上下文上的后续查询重复使用（缓存复用，而非经验回放）。

**学习维度 / Learning**

- **学习范式**: 混合（hybrid），但参数化训练限于新增的记忆参数：底座 LLM 主体权重冻结，仅训练新初始化的记忆 token 投影矩阵 W_{Q^m}/W_{K^m}/W_{V^m}，经预训练→SFT→RLGF 三阶段梯度更新得到“会压缩记忆、会生成线索”的记忆模型（参数化、离线训练）。部署时对每个长上下文形成记忆、生成线索、检索证据则是非参数化的 in-context 推理过程。整体是“离线参数化训练记忆模块 + 在线非参数化记忆增强检索”的混合范式。
- **失败学习 (核心主题)**: 不直接做“失败检测/失败经验记忆”，但 RLGF 以端到端生成质量为反馈隐式区分好/坏线索：RLGF（Reinforcement Learning with Generation Feedback）用偏好排序损失 L_RLGF=Σ max(0, 1−R(y^+)+R(y^-)) 训练记忆模型，使其偏好那些“能支撑高质量最终答案”的线索 y^+ 而抑制低质量线索 y^-（奖励由其对端到端表现的贡献决定）。这是一种以最终生成质量为信号的偏好学习，可视为对“无效线索”的弱负反馈，但不是 Reflexion/Retroformer 式的失败轨迹自反思，也不维护失败模式记忆或负例库。
- **技能/程序归纳**: 否。MemoRAG 不从经验中归纳可复用技能/工作流/程序（与 Voyager/AWM/Synapse 不同）。其唯一“可学习产物”是压缩全局记忆与线索生成能力，针对长文档理解任务，而非程序性技能库。
- **在线 vs 离线**: 二者结合：离线阶段批量训练记忆模型（预训练+SFT+RLGF，需在 SFT/RLGF 数据上训练，文档称仅需数小时额外训练即可适配新任务）；在线阶段对每个新长上下文实时形成全局记忆、生成线索并检索作答。记忆形成是“按上下文在线进行”，但记忆模块本身的能力来自离线训练。

**评测 / Evaluation**

- **任务领域**: 长上下文处理：长文档/长篇 QA（单文档 QA、多跳 QA、长书 QA）、查询聚焦摘要与无显式查询的摘要等非 QA 任务，覆盖法律、金融、物理、编程等 20 个专业领域；不涉及网页导航、具身、游戏、GUI 或多智能体协作。强调对“需全局信息聚合的高层查询”和“非结构化超长上下文（最高约百万 token）”的处理能力。
- **基准**: 三大评测：①LongBench（NarrativeQA/nar、Qasper/qas、MultiFieldQA/mul、HotpotQA/hot、2WikiMQA/2wiki、MuSiQue/mus、GovReport/gov、MultiNews/news 等）；②InfiniteBench（En.SUM、En.QA）；③作者自建的 UltraDomain 基准（20 个数据集、上下文最高约 100 万 token、覆盖法律/金融/物理/编程等高层查询领域）。指标按各数据集标准（QA 用 F1/相应指标、摘要用 ROUGE 等，见原文 Appendix B）。
- **报告增益**: 主实验（Table 1，13 数据集平均分 ave.）：MemoRAG 平均 40.2，全面超越所有基线——长上下文全量输入 Full 35.0、MInference 33.3、SelfExtend 30.1、标准 RAG 检索器 BGE-M3 29.7 / Stella-v5 29.0 / Jina-emb-v3 29.7、先进 RAG 方法 GraphRAG 29.4 / RQ-RAG 30.1。逐数据集（MemoRAG vs Full）：nar 27.5 vs 21.4、qas 43.9 vs 39.4、mul 52.2 vs 51.5、mus(MuSiQue) 33.9 vs 28.2、2wiki 54.1 vs 38.1（+约 16 点，提升最大之一）、hot(HotpotQA) 54.8 vs 48.1、news 26.3 vs 24.9、gov 32.9 vs 32.6、en.sum 15.7 vs 13.0、en.qa 22.9 vs 15.2（+约 7.7 点）、fin 51.5 vs 47.8、legal 51.0 vs 46.5、misc 55.6 vs 48.7。所有 13 项均带 † 标记（t 检验 p<0.05 显著超过全部基线）。UltraDomain（Fig.3，含其余 18 个数据集，上下文多超生成器 128K 上限）：MemoRAG 在全部数据集上一致超越所有基线，并超过“直接喂全量上下文”的 Full，证明其处理超长上下文与高层任务的能力。消融（Fig.4）：去掉任一技术设计（紧凑记忆 vs 轻量记忆、预训练/SFT/RLGF 各阶段）均导致性能下降；压缩比 β 增大性能下降但在 β=32 趋稳，且各 β 下均优于标准 RAG。
- **对比基线**: 三类基线：①全量上下文长 LLM（Full，128K 上下文）及其加速/扩展变体 MInference（稀疏注意力加速 prefill）、SelfExtend（双层层次注意力扩展窗口）；②标准 RAG 配不同检索器 BGE-M3、Stella-en-1.5B-v5、Jina-emb-v3；③先进 RAG 方法 HyDE（生成假设文档再检索）、RQ-RAG（查询改写/分解/消歧）、GraphRAG（构建知识图谱辅助检索，用 GPT-4o）。生成器统一为 Phi-3-mini-128K-instruct（SelfExtend 除外）。

**分析 / Analysis**

- **关键创新**: 提出“全局记忆增强检索（global memory-enhanced retrieval）”的双系统 RAG：先用一个轻量长程记忆模型把整段长上下文压缩成紧凑全局记忆（KV 压缩 + 记忆 token，β 倍降显存、单上下文可达约百万 token），再让记忆模型基于全局记忆生成“草稿答案线索”作为检索查询，从而跨越“查询无显式检索意图 / 上下文非结构化”这两大传统 RAG 失效条件；并首创 RLGF（以端到端生成质量反馈强化记忆模型的记忆与产线索能力），使 RAG 在传统 QA 之外的非 QA、高层聚合、超长上下文任务上同样可用且显著领先。
- **局限**: 作者承认/可见：①索引（记忆形成）延迟高于标准 RAG（因需对全文做全局记忆压缩），检索延迟也因需先生成线索而高于普通向量检索（但快于 GraphRAG、优于长 LLM 全量 prefill）；②记忆为有损压缩，β 增大语义保留减少、性能下降（β>32 后才趋稳），存在效率-效果折衷；③记忆模型需离线预训练+SFT+RLGF（SFT 数据由强 LLM 生成并人工精修，构建有成本），不是开箱即用；④无显式遗忘/更新/冲突消解、无时间推理、纯文本单语义记忆、不学习记忆“何时存/取”的控制策略；⑤记忆是针对单一长上下文形成，不累积跨任务智能体经验。
- **与其他工作关系**: 属本研究 E 类“认知架构启发框架（Cognitive-architecture frameworks）”，以“类人通读→形成全局记忆→回忆线索→检索细节→作答”的认知流程为蓝本。与 D 类图/生产级记忆-RAG（HippoRAG D1 用 OpenIE+知识图谱+个性化 PageRank、HippoRAG2 D2、Zep/Graphiti D3、Mem0 D4）走不同技术路线：MemoRAG 用“激活态 KV 压缩记忆 + 记忆生成线索”而非外部知识图谱或向量笔记。明确以 HyDE（生成假设文档）、RQ-RAG（查询改写）、GraphRAG（静态知识图谱）为直接对照并指出其只依赖模型内部知识或静态图、难全局理解上下文之不足。与 B 类对话/经验记忆（MemoryBank、MemGPT、A-MEM、EM-LLM 等）及 A/C 类经验回放/技能诱导（Reflexion、ExpeL、Voyager、AWM、ReasoningBank）正交——后者记忆“智能体自身轨迹/技能”，MemoRAG 记忆“当前长上下文的全局语义”。其 KV 压缩记忆思路承接长上下文压缩/记忆 token 类工作；同期/后续认知启发记忆-RAG（如 ComoRAG）亦延续“记忆组织 + 迭代检索”脉络。
- **可复现性**: 复现性较好：官方代码开源于 https://github.com/qhjqhj00/MemoRAG （约 2.0k stars，Apache-2.0），提供 memorag pip 包、记忆模型权重、Memory-Augmented Retrieval 接口、demo 与 BibTeX；评测基准 LongBench、InfiniteBench 公开，UltraDomain 由作者发布。WWW 2025 经同行评审。需注意：记忆模型依赖离线训练（预训练+SFT+RLGF），完整复现需相应训练数据与算力；线索生成与检索流程清晰、易上手使用。社区采用信号中上（约 2k stars、百余引用、作者持续维护 v2）。

**补充维度 / Supplemented (2025-26 frontier)**

- **记忆主体**: 上下文/文档中心（context-centric）：记忆的是“当前长输入文档/数据库的全局语义”，用于更好地理解与检索该上下文以作答；既非用户中心（不为个性化记住用户偏好，区别于 Mem0/Zep/LongMemEval），也非智能体中心（不记住智能体自身经验做自我改进，区别于 ReasoningBank/Voyager）。本质是面向长上下文处理的全局记忆增强检索层。
- **多智能体记忆**: 单智能体/单系统设定。MemoRAG 是面向单一 LLM 流水线的“记忆模型+检索器+生成器”双系统，不涉及多智能体共享/路由记忆（与 G-Memory、MIRIX 正交）；其“双系统”指轻量记忆模型与重量生成模型的分工，而非多个智能体。
- **时序推理支持**: 无显式时间建模。记忆为压缩 KV 缓存，不维护事实有效期窗口或事件时序日历（区别于 Zep/Graphiti），不做事实随时间失效的时序推理；上下文中的时间信息只能被隐式压缩进全局记忆。
- **模态**: 纯文本（text-only）。全局记忆、线索、证据与最终答案均为文本，无视觉/具身/多模态记忆。
- **过度个性化/记忆安全风险**: 不适用 / 论文未讨论用户记忆安全。MemoRAG 为上下文中心记忆，不构建用户画像，故不涉及过度个性化、谄媚或侵入性记忆等问题，无 OP-Bench/Causal-LoCoMo 类安全评测。潜在风险偏“信息保真”层面：有损压缩（β 较大时）可能丢失关键细节、生成的草稿线索可能含不准确信息从而误导检索，但论文未做隐私治理或记忆安全分析。
- **冲突/矛盾处理**: 无专门的冲突/矛盾事实消解机制。记忆是对单一长上下文的一次性压缩，不做跨记忆条目的更新或矛盾调和（无 Memory-R1 式 UPDATE 或 MEMTRACK 式冲突跟踪）；若上下文内部存在矛盾信息，由全局记忆压缩与下游检索/生成隐式处理，论文未显式建模。

**不确定字段 / Uncertain**

- 学习型记忆控制 (`learned_memory_control`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


## F. 记忆评测基准 (Memory-evaluation benchmarks)


<a id="f1-longmemeval聊天助手长期交互记忆评测基准含-longmemevals-约-115k-tokens-与-longmemevalm-约-500-会话约-150-万-tokens-两个标准设置外加-longmemeval_oracle-理想检索设置202509-发布去干扰的-cleaned-版202605-推出后续-longmemeval-v2-面向智能体场景"></a>

### F1 LongMemEval

*LongMemEval（聊天助手长期交互记忆评测基准；含 LongMemEvalS 约 115k tokens 与 LongMemEvalM 约 500 会话/约 150 万 tokens 两个标准设置，外加 LongMemEval_oracle 理想检索设置；2025/09 发布去干扰的 cleaned 版，2026/05 推出后续 LongMemEval-V2 面向智能体场景）*


**基本信息 / Provenance**

- **年份**: 2024（arXiv 预印本 2024-10-14 首次公开，arXiv:2410.10813；正式被 ICLR 2025 接收，2025/02 公布）
- **作者/机构**: 第一作者 Di Wu（吴頔，加州大学洛杉矶分校 UCLA，工作于腾讯 AI Lab 西雅图实习期间完成，导师 Hongwei Wang 与 Wenhao Yu）；合作者 Hongwei Wang、Wenhao Yu、Dong Yu（均腾讯 AI Lab 西雅图 Tencent AI Lab Seattle）、Yuwei Zhang（加州大学圣地亚哥分校 UC San Diego）、Kai-Wei Chang（UCLA）。属学术界（UCLA + 腾讯 AI Lab + UCSD）合作工作。
- **发表venue**: ICLR 2025（International Conference on Learning Representations，2025 年正式接收并发表）；arXiv 预印本最早见于 2024-10（cs.CL）。
- **论文链接**: https://arxiv.org/abs/2410.10813 （OpenReview: https://openreview.net/forum?id=pZiyCaVuti ；项目主页 https://xiaowu0162.github.io/long-mem-eval/）
- **代码链接**: https://github.com/xiaowu0162/LongMemEval （MIT 许可；约 837 stars、65 forks、40 commits、主要贡献者为第一作者 Di Wu；数据集托管于 HuggingFace: xiaowu0162/longmemeval-cleaned。配套发布 longmemeval_s.json / longmemeval_m.json / longmemeval_oracle.json 三套数据与检索/生成/索引扩展代码）

**记忆分类 / Taxonomy**

- **记忆类型**: 本身是"评测基准"而非记忆系统，但其考核与示例实现聚焦语义记忆（semantic，用户事实/偏好/属性）与情景记忆（episodic，跨会话的对话事件及其时间戳）；不涉及程序性技能记忆。被测对象的工作记忆即 LLM 上下文窗口。基准定义的五项核心能力对应不同记忆维度：信息抽取(IE)≈情景/语义回忆、多会话推理(MR)≈跨情景聚合、知识更新(KU)≈语义记忆的时序更新、时间推理(TR)≈带时间元数据的情景记忆、弃答(ABS)≈对记忆缺失的元认知。
- **记忆结构**: 基准把长期记忆抽象为统一的"键-值数据存储" [(k1,v1),(k2,v2),...]：键可异构（离散的句子/段落/事实/实体，或连续的模型内部表征），值可重复。论文示例最优设计采用扁平索引(flat index)的稠密向量存储，值粒度为"轮(round)"，键为"值+抽取用户事实(K=V+fact)"。数据本身是带时间戳的多会话对话历史（haystack 形式），每条问题嵌入若干"证据会话(evidence session)"于大量无关填充会话中。
- **存储后端**: 评测层面与具体后端无关（可评测任意记忆系统）。论文示例系统采用：稠密向量检索器 Stella V5 1.5B（MTEB 高分）做相似度索引/检索；索引阶段用 Llama 3.1 8B Instruct 抽取摘要/关键短语/用户事实/带时间戳事件；阅读/生成阶段用 GPT-4o、Llama 3.1 70B/8B Instruct；QA 自动评测用 gpt-4o-2024-08-06 作为 LLM 裁判（与人工一致率 >97%）。数据集以 JSON 形式发布（HuggingFace + Google Drive）。
- **持久化**: 面向"外部持久化记忆(durable external store)"的在线增量评测：测试时把 N 个带时间戳的历史会话 S=[(t1,S1),...,(tN,SN)] 逐个喂给被测系统，要求其在线解析、记忆、更新，在所有会话之后再回答问题 q（提问时间 t_q>t_N）。基准本身不规定记忆驻留方式，但其设定（不断增长的交互历史、超出上下文窗口）正是为暴露"纯长上下文(in-context)"方案的不足而设计——LongMemEvalS 约 115k tokens、LongMemEvalM 约 150 万 tokens，迫使系统采用外部记忆。

**核心机制 / Mechanisms**

- **写入/编码**: 作为基准，LongMemEval 不规定写入编码，而是系统性地比较"值(value)"这一控制点(CP1)的三种编码粒度：(1) 整段会话(session)；(2) 分解为轮(round，一条用户消息+一条助手回复)；(3) 进一步压缩为摘要/抽取用户事实(summary/fact)。关键发现：把会话分解为"轮"显著提升 GPT-4o 阅读 QA 表现；进一步压缩为摘要/事实虽省 token 但因信息损失整体损害性能——唯一例外是多会话推理(MR)题型，事实分解因把跨会话信息统一为简化格式而持续提升表现。论文最优设计的写入编码为"轮级值 + 抽取用户事实做键扩展"。其示例索引管线用 Llama 3.1 8B Instruct 从会话中抽取摘要、关键短语、用户事实与"带时间戳事件"四类辅助表征。
- **检索机制**: 基准把检索拆为"键(CP2)/查询(CP3)/检索策略"三个控制点并实证比较。键扩展(CP2)：直接用值做键(K=V)是强基线；单独用压缩形式(摘要/关键短语)做键反而不如 K=V（因检索器已能处理长文本语义）；采用"文档扩展"把压缩信息与原值拼接做键(K=V+fact)效果最佳——平均 Recall@k 提升 9.4%、下游准确率提升 5.4%（在 LongMemEvalM、轮粒度下 Recall@10 从 0.692→0.784，会话粒度下 0.783→0.862）。时间感知查询扩展(CP3)：对含时间引用的查询，用强 LLM(M_T=GPT-4o)抽取时间范围以裁剪检索空间，在时间推理子集上轮粒度 Recall 平均提升 11.3%、会话粒度提升 6.8%（K=V+fact 时 Recall@10 由 0.550→0.722）；弱模型(Llama 8B)做时间范围推断会幻觉/漏掉时间线索反致性能下降。默认采用扁平稠密检索(Stella V5)，并把检索到的记忆项按时间戳排序以维持时序一致。基准统一接口可计算 Recall@k 与 NDCG@k（利用人工标注的答案位置标签，含轮级 has_answer 与会话级 answer_session_ids 两级召回指标）。
- **反思/巩固**: 基准不强制反思机制，但其"阅读策略(CP4)"实证了类反思的"读时整合"：即便理想检索(oracle)下，朴素阅读策略相比最优策略仍最多损失约 10 个绝对百分点。论文采用 Chain-of-Note(CoN，先逐条记忆项抽取要点笔记再据笔记推理) + 结构化 JSON 格式呈现检索结果，把长上下文阅读分解为"复制关键细节"与"基于精简笔记推理"两个更简单子任务；实验显示 CoN+JSON 在多种能力 LLM 上一致优于其它组合。索引阶段对会话做摘要/事实抽取也是一种"原始对话→精炼表征"的离线整合，但基准本身不产生 Reflexion 式经验洞见。
- **遗忘/更新**: 知识更新(KU)是其五大核心能力之一，专门考核"识别用户个人信息随时间发生变化并动态更新记忆"的能力（如用户搬家、换工作、偏好改变），这是先前基准(LoCoMo、MemoryBank、PerLTQA)普遍缺失的维度。基准本身不实现遗忘/衰减算法，而是评测被测系统能否正确以最新信息覆盖过时事实；论文亦在伦理声明中指出当前示例系统缺乏"记忆删除算子"是可信度隐患。商用系统试点发现 ChatGPT 倾向于在对话延续中"覆盖关键信息"导致出错。
- **经验回放 (核心主题)**: 不适用（本工作是用户中心的对话记忆评测基准，而非智能体经验回放方法）。LongMemEval 复用的是"用户在历史对话中透露的信息/偏好/事实"，而非智能体自身过往任务轨迹/技能。它不维护成败轨迹缓冲、不做范例提示或技能复用，与 ReasoningBank/Voyager/ExpeL 等"重用自身经验改进未来决策"的智能体中心范式正交。其"复用"体现为：要求系统跨多会话、跨长时间正确回忆并综合先前对话内容来回答新问题，以避免重复提问、保持个性化连贯。基准设计的"证据会话间接表达信息"(self-chatting，如用户不直说"上月买了新车"而是借咨询车险顺带透露)进一步增大了记忆与回忆难度。

**学习维度 / Learning**

- **学习范式**: 非参数化(non-parametric / in-context, prompt-level)为主：基准把长期记忆视为"在线上下文压缩"——逐会话处理、存储、按需索引检索，被测的示例系统全程冻结 LLM 参数、无梯度更新，学习/记忆完全发生在外部键值存储上。基准在相关工作中也讨论了"可微记忆模块(参数化)"路线，但其评测形式定位于即插即用、可集成现有助手系统的非参数化记忆。
- **失败学习 (核心主题)**: 不适用 / 非失败学习型方法。LongMemEval 不检测任务执行失败、不构建负例或错误规则、不对失败轨迹做自反思（与 Reflexion/Retroformer/ExpeL 正交）。它评测的是"记忆-回忆-推理"的正确性而非"从失败中改进策略"。其唯一的"知道自己不知道"维度是弃答(ABS)能力：从其它题型改写出 30 道"虚假前提(false premise)"问题，考核系统能否识别历史中并未提及的信息并正确回答"我不知道"，这是对记忆缺失的元认知，而非失败经验学习。
- **技能/程序归纳**: 不归纳可复用技能/工作流/程序。LongMemEval 聚焦用户事实/偏好的语义与情景记忆，不从经验中提炼程序性技能或可复用工作流（与 Voyager/AWM/Synapse 等技能诱导方法属不同范畴）。后续工作 LongMemEval-V2(2026/05) 才将长期记忆扩展到"智能体上下文(agentic context)"，但本论文不涉及。
- **在线 vs 离线**: 在线评测(online)：测试时把带时间戳的历史会话逐个流式喂入，要求被测系统"在线"解析、记忆、更新，待全部会话结束后再回答提问——模拟真实助手在持续交互中动态积累用户信息。基准数据的构建则是离线的（人工构造问题/证据语句、LLM 模拟+人工编辑证据会话、采样无关填充会话拼接成可任意扩展长度的历史）。示例系统的索引可在线（逐会话）进行，键/事件抽取在论文中以离线缓存形式提供以便复现。

**评测 / Evaluation**

- **任务领域**: 长期多会话对话记忆(multi-session dialogue)/个性化对话 QA。领域定位为"Personal"个人化任务型对话，覆盖如心理咨询、秘书事务等高度依赖累积个人知识的场景；强调任务导向(task-oriented)对话而非纯闲聊。不评测网页导航、具身、游戏、编码等智能体任务（后续 V2 才扩展到智能体场景）。
- **基准**: 本工作即提出 LongMemEval 这一新基准，含 500 道人工精心构造问题，覆盖七种题型(single-session-user / single-session-assistant / single-session-preference / multi-session / knowledge-update / temporal-reasoning，外加从前六类改写的 30 道 abstention)与五大核心能力(IE/MR/KU/TR/ABS)；三套数据：LongMemEvalS(约 115k tokens/约 40-50 会话)、LongMemEvalM(约 500 会话/约 150 万 tokens)、LongMemEval_oracle(仅证据会话的理想检索设置)。对比的既有基准包括 MSC、DuLeMon、MemoryBank、PerLTQA、LoCoMo、DialSim——LongMemEval 在历史长度可自由扩展与五大能力全覆盖(尤其首次纳入 KU 知识更新与助手侧信息回忆)上优于它们。指标：QA 用 GPT-4o LLM-as-a-Judge(与人工 >97% 一致)，检索用 Recall@k 与 NDCG@k。
- **报告增益**: 头条难度证据：在比 LongMemEvalS 简单约 10 倍的设置下人工评测显示，商用记忆助手 ChatGPT(GPT-4o)准确率 0.5773、Coze(GPT-4o)仅 0.3299，相比同模型离线读全文(Offline Reading GPT-4o=0.9184)分别下降约 37%、64%；长上下文 LLM 在 LongMemEvalS(约 115k tokens)上相比 oracle 检索设置普遍下降 30%~60%（GPT-4o：oracle 0.870→S 0.606，降 30.3%；Llama 3.1 70B：0.744→0.334，降 55.1%；Phi-3 14B：0.702→0.380，降 45.9%；含 Chain-of-Note 时 Llama 3.1 70B 降幅高达 66.3%）。
所提记忆优化的增益(LongMemEvalM)：(1) 值分解——把会话分解为"轮"显著提升 GPT-4o 阅读 QA；事实分解仅在多会话推理上持续增益。(2) 键扩展 K=V+fact——平均 Recall@k +9.4%、最终准确率 +5.4%（轮粒度 Recall@10 0.692→0.784，GPT-4o Top-10 准确率 0.670→0.720；会话粒度 Recall@10 0.783→0.862，GPT-4o Top-5 0.670→0.714）。(3) 时间感知查询扩展——时间推理子集上轮粒度 Recall 平均 +11.3%、会话粒度 +6.8%（K=V+fact 下 Recall@10 0.550→0.722）。(4) 阅读策略 Chain-of-Note+JSON——理想检索下相比次优阅读策略最多提升约 10 个绝对百分点。
- **对比基线**: (1) 商用记忆助手：ChatGPT(OpenAI 记忆功能，GPT-4o / GPT-4o-mini)、Coze(GPT-4o / GPT-3.5-turbo)；(2) 长上下文 LLM 直接读全文：GPT-4o、Llama 3.1 Instruct 70B/8B、Phi-3 128k 14B、Phi-3.5 Mini 4B（对照 oracle 仅证据会话设置）；(3) 记忆系统设计基线（统一框架视角下的九种方法）：In-context RAG、MemoryBank、LD-Agent、CoN、RAPTOR、MemWalker、HippoRAG 等；(4) 检索器对比：BM25、Contriever、Stella V5 1.5B、gte-Qwen2-7B-instruct。可控实验对照"是否分解值/是否键扩展/是否时间感知查询扩展/是否 CoN+JSON"等设计点。

**分析 / Analysis**

- **关键创新**: 提出首个全面、可任意扩展长度、覆盖五大核心长期记忆能力(信息抽取、多会话推理、时间推理、知识更新、弃答)的聊天助手长期交互记忆评测基准——尤其首次系统纳入"知识更新"与"助手侧信息回忆"两个先前基准缺失的维度，并用"针海(needle-in-a-haystack)"式可控管线把人工构造的证据会话嵌入大量填充会话、配以时间戳，构建难度可调的真实任务型对话历史。同时提出"索引-检索-阅读"三阶段 + 值/键/查询/阅读策略四控制点的统一框架，把九种现有记忆系统纳入同一视角，并据实验提出会话分解(轮粒度)、事实增强键扩展、时间感知查询扩展三项简单有效的记忆设计优化，显著提升召回与下游 QA。
- **局限**: (1) 仅覆盖个人化任务型对话单一领域，不含网页/具身/编码等智能体任务(后续 V2 才扩展)；(2) 纯文本、无多模态；(3) QA 评测依赖 GPT-4o LLM-as-a-Judge，存在裁判模型偏差与成本(虽报告 >97% 人工一致)；(4) 商用系统试点仅 97 题、3-6 会话(规模远小于 LongMemEvalS)且为人工逐轮交互，覆盖有限、时间点为 2024 年 8 月初(结果随系统迭代会变)；(5) 所提优化为启发式管线，未学习记忆控制策略；时间感知查询扩展依赖强 LLM(弱模型反而有害)；(6) 数据构造耗费约 400 人工时构建+150 人工时商用系统研究，难以低成本复制规模；(7) 伦理上指出示例系统缺记忆删除算子、存在隐私泄露与记忆投毒/越狱风险。作者于 2025/09 发布 cleaned 版修正部分历史会话对答案正确性的干扰。
- **与其他工作关系**: 属本研究 F 类"记忆评测基准"的代表作之一。它定位为先前长期对话记忆基准(MemoryBank=B2、PerLTQA、LoCoMo、DialSim、MSC、DuLeMon)的升级，强调更长可扩展历史与五大能力全覆盖；其与 LoCoMo 互为该领域两大主流记忆评测基准(LoCoMo 偏人-人对话、约 26k tokens；LongMemEval 偏人-AI 任务型、115k~150 万 tokens 且含知识更新)。统一框架中纳入并对照 MemoryBank(B2)、MemGPT(B3)思路、HippoRAG(D1，作为 PPR 实体检索基线)、RAPTOR、MemWalker、In-context RAG、Chain-of-Note 等。被大量后续记忆系统(Mem0=D4、Zep/Graphiti=D3、MIRIX=B9、A-MEM=B4、MemoryOS=B7、EM-LLM=B8 等)用作核心评测基准之一(与 LoCoMo 并列)，是衡量用户中心长期记忆质量(尤其时间推理与知识更新)的事实标准；与智能体中心经验回放/技能诱导(A 类 Reflexion/ExpeL/ReasoningBank、C 类 Voyager/AWM)正交。后续 LongMemEval-V2(2026/05) 将其扩展到智能体上下文。
- **可复现性**: 复现性强、社区采用度高：基准与代码 MIT 许可开源于 github.com/xiaowu0162/LongMemEval(约 837 stars、65 forks)，数据集托管 HuggingFace(xiaowu0162/longmemeval-cleaned，提供 longmemeval_s/m/oracle 三套 JSON)，并公开历史拼接算法、属性本体(164 个用户属性)、源数据混合(ShareGPT Apache-2.0、UltraChat MIT)与索引扩展/时间事件等中间缓存(Google Drive)。提供 lite(仅评测)与 full(运行记忆系统)两套环境、评测脚本 evaluate_qa.py、检索/生成/索引扩展完整代码。论文含详尽复现声明与附录实现细节；2025/09 发布 cleaned 版并附变更日志(Google Sheet)，复现需注意对齐数据版本(原版 vs cleaned)。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式管线）。LongMemEval 提出的记忆优化(值分解、事实键扩展、时间感知查询扩展、CoN+JSON 阅读)均为基于提示与检索的启发式设计，不用 RL/训练去学习"何时/写什么/如何检索/如何更新"的记忆管理策略本身。作为评测基准，它为后续"学习型记忆控制"工作(Memory-R1、Mem-α 等)提供了考核场地，但其自身属 2025-26 学习型记忆控制代际之前的启发式范式。
- **记忆主体**: 用户中心(user-centric)：核心目标是记住用户在历史对话中透露的信息、偏好与属性以实现个性化、跨会话连贯响应(典型场景如心理咨询、秘书事务)。与 Mem0/Zep 同属"记住用户信息做个性化"阵营，区别于 ReasoningBank/Voyager 等"记住智能体自身经验做自我改进"的智能体中心记忆。值得注意的是它还特别考核"助手侧信息回忆"(single-session-assistant)，即记住助手自己说过的内容，这是先前用户中心基准忽略的维度。
- **多智能体记忆**: 单智能体/单助手设定。LongMemEval 评测的是单个聊天助手在与单一用户长期一对一交互中的记忆能力，不涉及 G-Memory(D5)/MIRIX(B9) 式的多智能体共享/路由记忆架构(无跨智能体洞见/查询/交互分层)。
- **时序推理支持**: 显式且重点支持。时间推理(TR)是其五大核心能力之一，专门考核利用元数据时间戳与用户话语中显式时间引用回答时间敏感问题的能力(如"上周末你推荐的那家餐厅")。每个会话与问题都带时间戳(t_i 与提问时间 t_q>t_N)，含时间元数据推理与显式时间引用两类。论文证明朴素时间无关设计在 TR 上表现差，并提出"时间感知索引+查询扩展"——用带时间戳事件索引值、用强 LLM 从查询抽取时间范围裁剪检索空间，TR 召回提升 6.8%~11.3%。时间推理是基准中最难能力之一，也是后续记忆系统(Zep/Graphiti 双时间模型、Mem0^g 时间戳)重点攻关的维度。
- **模态**: 纯文本(text-only)。所有对话历史、问题、答案均为文本，无视觉/具身/多模态记忆。
- **冲突/矛盾处理**: 通过"知识更新(KU)"能力间接考核冲突/矛盾处理：KU 题型专门测试系统能否识别用户个人信息随时间发生变化(如旧偏好被新偏好取代)并以最新信息更新记忆、覆盖过时事实，从而在提问时给出与最新状态一致的答案。基准本身不实现冲突消解算法，而是评测被测系统的更新/覆盖正确性；试点发现 ChatGPT 在长对话中倾向于"覆盖关键信息"导致出错，说明朴素覆盖式更新易引发冲突处理失败。与显式遗忘/合并算子(Memory-R1 UPDATE、MEMTRACK)不同，这里是以"最新答案是否正确"为外部度量。
- **token成本/延迟证据**: 本工作为评测基准，非记忆层产品，故不以"省 token/降延迟"为卖点，未报告系统级 token/延迟节省百分比。但它在难度设计上量化了上下文规模与成本压力：LongMemEvalS 约 115k tokens/问题、LongMemEvalM 约 150 万 tokens/问题，证明纯长上下文(in-context)方案在 115k tokens 下已下降 30%~60%、且随历史增长会进一步恶化、并受"lost-in-the-middle"影响，从而论证外部记忆/检索相对全上下文在效率与质量上的必要性。值得注意是"值进一步压缩为摘要/事实"虽省 token 但损害整体 QA(信息损失)，提示 token 效率与记忆质量需权衡。基准提供的 Recall@k/NDCG@k 与按 token 预算的 QA 曲线(如 Llama 8B 超 3k 检索 token 性能骤降、GPT-4o 超 20k token 仍提升)是其效率相关证据。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)
- 过度个性化/记忆安全风险 (`over_personalization_risk`)


<a id="f2-locomolong-conversational-memory超长期对话记忆评测基准与数据集maharana-等人-2024-提出含问答事件摘要多模态对话生成三任务"></a>

### F2 LoCoMo

*LoCoMo（Long Conversational Memory；超长期对话记忆评测基准与数据集，Maharana 等人 2024 提出；含问答/事件摘要/多模态对话生成三任务）*


**基本信息 / Provenance**

- **年份**: 2024（arXiv 预印本 2402.17753 于 2024-02-27 首次公开 v1；正式发表于 ACL 2024）
- **作者/机构**: Adyasha Maharana（北卡罗来纳大学教堂山分校 UNC Chapel Hill）、Dong-Ho Lee（南加州大学 USC）、Sergey Tulyakov（Snap Inc.）、Mohit Bansal（UNC，共同指导）、Francesco Barbieri（Snap Inc.，共同指导）、Yuwei Fang（Snap Inc.，共同指导）。主要单位为 UNC Chapel Hill、USC 与 Snap Inc. 三方合作；该工作主要在 Snap 实习/合作期间完成。
- **发表venue**: ACL 2024（第 62 届计算语言学协会年会，长论文卷 Volume 1: Long Papers，泰国曼谷）。ACL Anthology 编号 2024.acl-long.747，页码 13851–13870，DOI 10.18653/v1/2024.acl-long.747；属正式主会长论文（非 Findings）。亦有 arXiv 预印本 abs/2402.17753（DOI 10.48550/arXiv.2402.17753，Corpus ID 268041615）。
- **论文链接**: https://arxiv.org/abs/2402.17753 （ACL Anthology 正式版：https://aclanthology.org/2024.acl-long.747/ ；项目主页 https://snap-research.github.io/locomo/ ）
- **代码链接**: https://github.com/snap-research/locomo （Snap Research 官方仓库，数据集文件 data/locomo10.json；约 929 stars、92 forks、18 commits、2 贡献者；含对话生成、QA/RAG/事件摘要评测脚本；不发布图片本体，仅提供图片 URL/BLIP 字幕/检索查询）
- **引用数**: 约 524 次（Semantic Scholar 实时查询，Corpus ID 268041615；与任务锚点“约 524”一致）。是引用量最高的智能体/对话记忆评测基准之一，被 Mem0、Zep、A-Mem、MemoryOS、ReasoningBank 等大量记忆系统作为标准评测集。

**记忆分类 / Taxonomy**

- **记忆类型**: 本身为评测基准而非记忆系统：LoCoMo 评估被测智能体的情景记忆（episodic，回忆过去会话中的具体事件/对话）与语义记忆（semantic，结合说话人信息与世界知识）能力，重点考查情景记忆的长程检索、时间记忆与因果记忆。它不实现也不规定特定记忆类型，而是提供让各类记忆机制（长上下文、RAG、外部记忆库）一较高下的检验场。所附生成管线中的虚拟智能体则用到短期记忆（会话摘要）与长期记忆（逐轮观察 observation）。
- **记忆结构**: 作为基准，其数据结构为带时间戳的多会话对话 JSON：每个样本含 conversation（按时序排列的 session_<num> 列表 + 各 session 的 date_time + speaker_a/speaker_b 双说话人，每个 turn 含 speaker、dia_id、text，多模态轮还含 img_url/blip_caption/检索 query），observation（逐会话生成的“关于说话人的断言/观察”，供 RAG 用），session_summary（逐会话摘要，供 RAG 用），event_summary（按说话人/会话标注的重大事件，事件摘要任务的标准答案），qa（问答标注：question/answer/category/evidence 证据轮 ID）。底层支撑数据为每位说话人的 persona（人设）与 temporal event graph（时序因果事件图）。
- **存储后端**: 数据集以静态 JSON 文件发布（data/locomo10.json），可加载入任意记忆后端进行评测。评测时被测系统可选用：被截断的上下文窗口（base/long-context LLM）、或基于 DRAGON 检索器构建的对话/观察/摘要数据库（向量检索 RAG）。后续第三方多用向量库（FAISS/Chroma/Qdrant）、图数据库（Neo4j，如 Mem0^g/Zep）等承载从 LoCoMo 抽取的记忆。基准本身不绑定特定存储后端。
- **持久化**: 对被测系统而言为外部持久化记忆评测场景：对话跨“数月”时间跨度、最多 32 个会话，远超固定上下文窗口，迫使系统使用外部/持久记忆（RAG、外部记忆库）而非纯 in-context。基准数据本身为持久静态文件。其生成管线中的虚拟智能体把短期摘要存入 H_s、把逐轮观察存入长期记忆 H_l（外部记忆）。

**核心机制 / Mechanisms**

- **写入/编码**: （基准侧）LoCoMo 不规定写入编码，但提供两类“写入”产物供 RAG 基线使用：(1) observation 观察——把每一轮对话 h_{k_j} 经 gpt-3.5-turbo 转写为关于说话人生活/人设的断言 o_{k_j}（事实抽取式编码）；(2) session_summary 会话摘要——逐会话的增量式摘要（摘要式编码）。原始对话则可逐字（verbatim）入库。
（生成管线侧）创建数据时采用 Park 等人（Generative Agents）的智能体架构：每个虚拟智能体 L_i 被赋予 persona p 与时序事件图 G；每个会话 k 结束后由 LLM M（gpt-3.5-turbo）基于本会话历史 h_k 与上一摘要 w_{k-1} 生成会话摘要 w_k 存入短期记忆 H_s；会话内每一轮 h_{k_j} 转为观察 o_{k_j} 存入长期记忆 H_l。响应时再以最新摘要 w_k、检索到的相关观察、当前会话历史 h_{k+1}、人设 p、以及落在两会话日期之间的事件子集为条件生成下一会话内容，从而把长程时间线注入对话。多模态轮通过 image-sharing/​image-reaction 函数：用 LLM 生成图片字幕 c→关键词 w→icrawler 网络检索取图；收图方用 BLIP-2 生成字幕再生成反应。
- **检索机制**: （基准侧）LoCoMo 评测三种读取范式并量化检索效果：(1) base LLM——截断上下文，丢弃较早对话；(2) long-context LLM——以更大窗口（gpt-3.5-turbo-16k，4K/8K/12K/16K）整体读取；(3) RAG——用 DRAGON 检索器、gpt-3.5-turbo-16k 阅读器，从“对话/观察/会话摘要”三种数据库中检索 top-k（k∈{5,10,25,50}）相关单元。基准为每个 QA 标注证据轮 ID，并以 recall@k 报告检索召回准确率。核心实证：以“观察 observation”为数据库、top-5 时检索效果最佳（信噪比 SNR 关键，检索过多反而下降）；以会话摘要为库虽召回高但因摘要丢信息而无明显增益。被测记忆系统可自由实现相似度检索、图遍历等任意读取机制——LoCoMo 仅提供问答正确性与召回率的统一裁判。
- **反思/巩固**: （基准侧）LoCoMo 不强制反思，但其设计专门考查“原始对话→高层理解”的整合能力：事件摘要任务要求被测系统把跨会话、含时间与因果共指的密集事件“消化”为结构化事件列表（用 FactScore 衡量原子事实精确率/召回率），实证发现长上下文模型在此任务上反而不及 4K 基础模型（gpt-3.5-turbo-16k 精确率降 3.0%、召回降 8.7%），说明长上下文模型未必善用其窗口、整合能力有限。
（生成管线侧）虚拟智能体含显式 reflect & respond 机制：逐会话生成摘要 w_k（巩固近期），逐轮蒸馏为观察 o（抽象为长期断言），响应时检索相关观察做反思——直接沿用 Generative Agents 的反思范式。
- **遗忘/更新**: （基准侧）LoCoMo 不实现遗忘/更新机制，但被广泛批评为“缺乏对知识更新（knowledge update）的考查”——即缺少“用户信息随时间变化（如换工作）”这类需要 UPDATE/失效的题型（Zep 团队 2025 年明确指出此短板）。其 base LLM 基线则以“截断早期对话”作为被动遗忘的代理。后续系统（Mem0 的 ADD/UPDATE/DELETE、Memory-R1 的 UPDATE）正是为补这一缺口而设计。
- **经验回放 (核心主题)**: 不适用（LoCoMo 是用户中心的对话记忆评测，不涉及智能体经验回放）。它评估的是“记住对话中关于用户/说话人的信息并在后续问答中正确回忆”，而非“复用智能体自身过往任务轨迹/技能改进未来决策”。与 ReasoningBank/Voyager/ExpeL 等经验回放、技能复用范式正交。生成数据时虚拟智能体会“复用”历史摘要与观察来保持对话连贯，但这属对话一致性维护而非任务级经验回放。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric / in-context, prompt-level）为主：LoCoMo 评测的主流方法（长上下文、RAG、外部记忆库）均不更新被测 LLM 参数，学习发生在外部上下文/记忆库层面。唯一例外是多模态对话生成任务中对 MiniGPT-5 做了参数化微调（base/+summary/+observation 三版本，均从 MMDialog 微调的检查点初始化），属混合中的参数化分支。整体基准本身无学习范式约束，可同时承载非参数化与参数化方案的对比。
- **失败学习 (核心主题)**: 不适用 / 非该基准目标。LoCoMo 不检测任务失败、不构建失败记忆或负例规则，与 Reflexion/Retroformer/ExpeL 等失败学习方法正交。它通过“adversarial 对抗题型”间接考查模型识别“不可回答/被误导”问题的能力（期望模型答“无法回答”），并发现长上下文 LLM 在对抗题上崩塌（gpt-3.5-turbo-16k 16K 时仅 2.1%，而 GPT-4-turbo 4K 达 70.2%），揭示长上下文易被诱导幻觉、易把对话/事件错配给错误说话人——这是对“失败模式”的诊断，而非让系统从失败中学习。
- **技能/程序归纳**: 不诱导可复用技能/工作流。LoCoMo 既不评测也不实现程序性技能归纳，与 Voyager/AWM/Synapse 等技能诱导工作属不同范畴。其考查对象限于对话记忆的回忆、时间推理、因果/事件理解与一致性。
- **在线 vs 离线**: （基准侧）离线静态评测集：50（v1）/10（正式版）段对话一次性生成并人工校验后固定发布，供离线批量评测。（生成管线侧）数据构造为在线增量式——两个虚拟智能体跨数月、最多 32 会话逐会话在线生成对话，每会话即时写入摘要与观察记忆，再供后续会话条件生成。

**评测 / Evaluation**

- **任务领域**: 多会话长期开放域对话记忆（multi-session dialogue / 个性化对话）。具体为：超长期开放域多模态对话理解，覆盖问答（QA）、事件图摘要、多模态对话生成三类任务；面向需要跨数月、数十会话保持记忆一致性的对话智能体/聊天机器人评测。不涉及网页导航、具身、游戏、GUI、编码等智能体任务。
- **基准**: LoCoMo 自身即为基准/数据集（非在其它基准上评测）。规模（正式 ACL 版 / data/locomo10.json）：10 段超长对话，每段平均约 600 轮、约 16K tokens、跨最多 32 会话、时间跨度约数月、双说话人。共约 1,540 道问答题，分五类推理：single-hop（单跳）、multi-hop（多跳）、temporal（时间推理）、open-domain（开放域/常识世界知识）、adversarial（对抗，期望答不可回答）。三任务指标：QA 用 F1（部分匹配）+ recall@k（检索召回，仅 RAG）；事件摘要用 ROUGE + FactScore（原子事实精确率/召回率/F1）；多模态对话生成用 MMRelevance + 常规 NLG 指标。注：arXiv v1（2024-03）原含 50 段对话（平均约 300 轮、9K tokens、最多 35 会话），正式版精简为 10 段最长、标注质量最高的对话以降低闭源模型评测成本，故不同来源统计数字存在差异。
- **报告增益**: LoCoMo 给出的是被测方法的基线表现而非自身“增益”，关键数字（QA 总体 F1，越高越好）：人类 87.9（single-hop 95.1 / multi-hop 85.8 / temporal 92.6 / open-domain 75.4 / adversarial 89.4）；最佳模型 GPT-4-turbo(4K) 总体 32.1（论文 §6.1 另称 32.4），远落后人类约 56 点。长上下文 gpt-3.5-turbo-16k 随窗口增大：4K→24.1、8K→25.2、12K→33.5、16K→37.8，但 adversarial 从 4K 时退化至 16K 的 2.1%（GPT-4-turbo 4K 为 70.2%、Llama-2-70B 4K 为 22.1%），即长上下文使对抗题下降约 83%、并在事件理解上落后 base 约 14%。RAG（DRAGON+gpt-3.5-16k）：以 observation 作库、top-5 时相对纯对话日志约 +5%（gpt-3.5-turbo 22.4→约 27），增益随检索数增多而衰减（信噪比问题）；以 summary 作库召回高（R@k 可达 90%+）但 F1 无显著提升。项目主页综述：长上下文与 RAG 使“记忆”能力提升约 22–66%，但仍落后人类约 56%，时间推理落后约 73%。事件摘要（FactScore F1）：gpt-3.5-turbo 45.9（最高，含增量摘要）、GPT-4-turbo 45.1、long-context 16k 仅 39.9，开源模型显著更低。核心结论：时间推理与开放域知识题最难；长上下文≠会用上下文。
- **对比基线**: LoCoMo 内置三大类被测“基线”：(1) base LLM（受限上下文，丢弃早期对话）——Mistral-Instruct-7B(8K)、Llama-2-Chat-70B(4K)、gpt-3.5-turbo(4K)、gpt-4-turbo(4K)；(2) long-context LLM——gpt-3.5-turbo-16k（4K/8K/12K/16K 多档）；(3) RAG——DRAGON 检索器 + gpt-3.5-turbo-16k 阅读器，数据库为 对话/观察(observation)/会话摘要(summary) 三选一、top-k∈{5,10,25,50}。另设人类上界（Human）作为天花板对照，以及多模态对话生成任务的 MiniGPT-5（base / +summary / +observation）。

**分析 / Analysis**

- **关键创新**: 首个“超长期（数月、最多 32 会话、约 600 轮/16K tokens）”开放域多模态对话记忆评测基准。两大核心创新：(1) 人机协同生成管线——用赋予人设 persona + 时序因果事件图 G 的 LLM 生成式智能体（Generative Agents 架构）产出长程一致对话，再由人工标注者修订长程不一致（约编辑 15% 对话轮、替换/删除约 19% 图片）保证质量与事件图对齐；(2) 三任务整体评测框架——QA（五类推理：单跳/多跳/时间/开放域/对抗）+ 事件图摘要（FactScore 原子事实）+ 多模态对话生成（MMRelevance），首次系统量化长上下文 LLM 与 RAG 在真正超长对话上的记忆短板（尤其时间推理、对抗幻觉、长程因果理解），且全面落后人类。
- **局限**: (1) 长度/复杂度不足：对话仅约 16K–26K tokens，落在现代 LLM 上下文窗口内，full-context 全量输入往往胜过专门记忆系统（Mem0 自报 full-context J≈73% 高于其最佳记忆方案 ≈68%），故未真正在“压力下”考验长程检索（Zep 团队 2025 批评）。(2) 缺知识更新题型：不考查随时间变化的信息更新（如换工作），漏掉 agent 记忆的关键功能。(3) 标注质量缺陷：2026 年 Penfield Labs / dial481 独立审计发现 1,540 题中约 99 个“损分错误”（约 6.4% 标准答案错误，含答案键幻觉、时间推理错误、说话人归属错误），理论满分上限仅约 93.6%，使 >93.6% 的得分在数学上不可能（系受益于错误标注）；snap-research/locomo#27 早已报告 29 处错误。(4) 裁判过宽：用 gpt-4o-mini 作 LLM 裁判，对“话题相近但细节全错”的答案接受率高达约 62.8%，奖励弱检索的典型失败模式，使约 6 点以内的得分差落在裁判噪声内。(5) 各题型样本量悬殊（96–841，约 8.8 倍），Wilson 区间下大量相邻系统差异统计上不可区分（open-domain n=96 需 15+ 点差才显著）。(6) 复现性差：多家第三方无法复现基于 LoCoMo 的厂商宣称分数（如 mem0 #2800/#3944、时间戳错用导致温度时间记忆错误致 ~20% 分）。(7) 不发布图片本体（仅 URL/字幕/query），多模态可复现性受限；v1（50 段）与正式版（10 段）统计不一致易致误用。
- **与其他工作关系**: 属本研究 F 类“记忆评测基准”，且为其中引用最高、最被广泛采用者（约 524 次）。它是 D 类生产级用户中心记忆系统的事实标准评测场：Mem0 / Mem0^g（D4）、Zep / Graphiti（D3）、A-Mem（B4）、MemoryOS（B7）、MIRIX（B9）等均在 LoCoMo 上报告分数并相互比拼（Mem0 vs Zep 之争即围绕 LoCoMo 的实现/裁判分歧展开）。其生成管线直接构建于 Generative Agents（B1，Park 等人 2023）的 reflect & respond + 观察/反思记忆架构之上，并用 DRAGON 检索器实现 RAG 基线、MiniGPT-5 做多模态。常被与 LongMemEval（F 类另一基准，约 115K tokens/题、被批更像“上下文窗口测试”而非记忆测试）对照讨论；近年 dial481/locomo-audit、Penfield Labs 审计与 EverMemOS、Emergence AI 等围绕其标注与裁判缺陷的争议，催生对更严格记忆基准的需求。
- **可复现性**: 代码与数据公开（github.com/snap-research/locomo，约 929 stars、92 forks），数据集 data/locomo10.json 直接可下载，附对话生成、QA/RAG 评测脚本（evaluate_gpts/claude/gemini/hf、evaluate_rag_gpts、generate_observations/session_summaries），引用便利（ACL Anthology 正式版 + arXiv）。社区采用度极高，是记忆系统标配评测集。但复现性存重大隐患：(1) 不发布图片本体，多模态任务难完全复现；事件摘要与多模态训练脚本 README 标注“Coming soon”；(2) v1（50 段对话）与正式版（10 段）数据规模不同，跨论文数字不可直接比；(3) 标准答案约 6.4% 错误 + LLM 裁判过宽，导致小幅分差不可解释、且多家厂商分数无法被第三方复现（mem0 #2800、EverMemOS#73 报告 38.38% vs 宣称 92.32% 等）。结论：基准本身可获取、易上手，但评测协议（裁判/标注/数据版本/被测模型配置）差异使“可比性”与“可复现性”面临系统性挑战。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否。LoCoMo 是评测基准，本身不含记忆管理策略，更不用 RL/训练学习“何时/写什么/如何检索/更新”的记忆控制策略。它评测的基线（长上下文、启发式 RAG）均为固定/启发式管线。它反而是 2025–26 学习型记忆控制工作（Memory-R1、Mem-α 等）用来验证其学得策略优劣的评测场之一。
- **记忆主体**: 用户中心（user-centric）。LoCoMo 考查的是“记住对话中关于说话人/用户的信息（人设、生活事件、偏好）以在后续问答中正确回忆并保持一致”，属个性化对话记忆评测，与 Mem0/Zep/LongMemEval 同阵营。与 ReasoningBank/Voyager 等“记住智能体自身任务经验做自我改进”的 agent-centric 记忆正交——这也是 LoCoMo 不评测经验回放/技能诱导的根本原因。
- **多智能体记忆**: 单智能体/单系统评测设定（对话双方为两位说话人，但被测的是单个记忆系统对该对话的记忆能力）。不涉及 G-Memory/MIRIX 式的多智能体共享/路由记忆架构，无跨智能体洞见/查询/交互分层。数据生成阶段虽用两个虚拟智能体对话，但各自独立维护记忆，非协作共享记忆。
- **时序推理支持**: 显式且重点考查时间推理——这是 LoCoMo 的核心难点维度之一。每个会话带 date_time 时间戳，QA 含专门的 temporal（时间推理）题型，事件摘要任务需理解跨会话的时间与因果共指。实证显示时间推理是最难场景之一：模型在 temporal 题上远落后人类（人类 92.6 vs GPT-4-turbo 4K 仅 10.4 F1），长上下文/RAG 仍落后人类约 73%。这也使后续支持双时态/有效期窗口的系统（Zep/Graphiti、Mem0^g 时间戳+失效标记）能在 LoCoMo temporal 题上拉开差距。但审计指出部分 temporal 标准答案本身存在时间推理错误。
- **模态**: 多模态（multimodal，文本+图像）。对话含 image-sharing / image-reaction 行为，轮次可带 img_url 与 BLIP 生成的字幕；专设多模态对话生成任务（MiniGPT-5 + MMRelevance）。但 QA 与事件摘要任务中图片以字幕替代为纯文本处理；且仓库不发布图片本体（仅 URL/字幕/检索 query），多模态实测受限。
- **冲突/矛盾处理**: 基准本身不考查冲突/矛盾事实的更新解决——这正是其被批评的缺口之一（缺知识更新题型，未测“信息随时间变化后的覆盖/失效”）。其 adversarial 题更接近“识别不可回答/被误导”而非“消解两条矛盾事实”。后续 Memory-R1（UPDATE）、MEMTRACK 等正是针对此空白设计冲突跟踪与更新机制。
- **token成本/延迟证据**: LoCoMo 本身不报告 token/延迟效率（其关注准确性指标 F1/FactScore/MMRelevance/recall@k）。但它催生并成为后续记忆系统效率对比的舞台：在 LoCoMo 上，Mem0 相对 full-context 报告 p95 延迟降约 91%、token 成本省 >90%，Mem0 记忆约 7k tokens/对话 vs Zep 图记忆 >600k；Zep 报告相对全量上下文延迟降约 90%。另一关键点：因 LoCoMo 对话仅约 16–26K tokens，full-context 全量输入成本可控且准确率反超部分记忆系统，使“在 LoCoMo 上的效率优势”被质疑不代表真实长程压力场景。

**不确定字段 / Uncertain**

- 过度个性化/记忆安全风险 (`over_personalization_risk`)


<a id="f3-membench面向-llm-智能体记忆能力的更全面评测基准其数据集别名亦写作-membench--membench基于-memsimmemengine-生态扩展引入事实记忆--反思记忆两个记忆层级与参与--观察两种交互场景并提供-effectivenessefficiencycapacity-多维度指标"></a>

### F3 MemBench

*MemBench（面向 LLM 智能体记忆能力的更全面评测基准；其数据集别名亦写作 Membench / MemBench；基于 MemSim/MemEngine 生态扩展，引入「事实记忆 × 反思记忆」两个记忆层级与「参与 × 观察」两种交互场景，并提供 effectiveness/efficiency/capacity 多维度指标）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本 2506.21605 于 2025-06-20 首次公开 v1；同期被 ACL 2025 Findings 录用）
- **作者/机构**: Haoran Tan（谭浩然，并列第一作者）、Zeyu Zhang（张泽宇，并列第一作者）、Chen Ma（马晨）、Xu Chen（陈旭，通讯作者）均来自中国人民大学高瓴人工智能学院（Gaoling School of Artificial Intelligence, Renmin University of China，北京大模型与智能治理重点实验室、教育部下一代智能搜索与推荐工程研究中心）；Quanyu Dai（戴权煜，通讯作者）、Zhenhua Dong（董振华）来自华为诺亚方舟实验室（Huawei Noah's Ark Lab）。通讯作者邮箱 xu.chen@ruc.edu.cn、daiquanyu@huawei.com。该团队与 LLM 智能体记忆综述（Zhang et al. 2024《A Survey on the Memory Mechanism of LLM-based Agents》）、MemSim、MemEngine 同源。
- **发表venue**: ACL 2025 Findings（Findings of the Association for Computational Linguistics: ACL 2025，2025 年 7 月，奥地利维也纳；Anthology ID 2025.findings-acl.989，pages 19336–19352，DOI 10.18653/v1/2025.findings-acl.989）。arXiv 同步预印本 2506.21605（DOI 10.48550/arXiv.2506.21605，DBLP journals/corr/abs-2506-21605，CorpusId 280011015）。属带开源数据/代码的学术评测基准（benchmark/dataset 论文，而非记忆系统/方法）。
- **论文链接**: https://arxiv.org/abs/2506.21605
- **代码链接**: https://github.com/import-myself/Membench（官方仓库，约 52 stars、2 forks、1 位主要贡献者，MIT 许可，截至调研日 2026-06；以 Python 为主，含 data2test 采样数据集、NoiseData 噪声数据、makenoise.py 噪声生成脚本；完整数据集通过百度网盘/Google Drive 分发；2025-11 仍有更新。评测实现基于同团队的 MemEngine 库）。
- **引用数**: 约 55 次引用（Semantic Scholar，CorpusId 280011015，截至调研日 2026-06）；作为 2025 年 LLM 智能体记忆评测方向的代表性基准，发布约半年内已被高频引用（如 EvolveMem 等记忆系统将其与 LoCoMo 并列为标准评测台）。

**记忆分类 / Taxonomy**

- **记忆类型**: 本身是评测基准而非记忆系统，但其数据集按记忆「层级」显式区分两类待评测能力：①事实记忆（Factual Memory）——用户或关联实体被明确陈述的具体属性（亲属年龄/职业、事件时间细节等），偏低层、对应可被直接抽取的语义/情景事实；②反思记忆（Reflective Memory）——未被明确陈述、需从大量低层偏好/属性表达中归纳总结出的高层偏好（如由「喜欢多部某类电影」推断出「电影类型偏好」、由对多道菜的喜好推断「口味偏好」），对应需经反思/抽象生成的高阶语义记忆。评测覆盖信息抽取、跨会话推理、知识更新、时间推理、反思式总结等子能力。所评测的记忆机制（FullMemory/Retrieval/Recent/GenerativeAgent/MemoryBank/MemGPT/SCMemory）跨越工作记忆（上下文窗口）与外部长期记忆。
- **记忆结构**: 作为基准，其「结构」体现在数据集的组织而非单一记忆数据结构：底层是 500 个用户关系图（user relation graph，每个由用户 profile + 关联实体——人物、事件、地点、物品——构成），由此生成多会话对话（参与场景）与用户消息流（观察场景）及对应多选题。所被评测的七种记忆机制各自采用不同结构（原始全量缓冲 FullMemory、近期窗口 RecentMemory、向量检索库 RetrievalMemory、Generative Agents 反思记忆流、MemoryBank 含遗忘曲线、MemGPT 分层操作系统式记忆、SCMemory 自控记忆），基准本身对这些结构保持中立、统一评测。
- **存储后端**: 基准自身不绑定特定后端；评测框架基于 MemEngine（Zhang et al. 2025，统一模块化记忆库）实现七种记忆机制，统一以 Qwen2.5-7B 为智能体基座模型，所有涉及检索的机制统一采用 multilingual-e5-small 嵌入模型做检索。数据集以 JSON 形式分发（百度网盘/Google Drive + 仓库 data2test 目录）。被评测后端因机制而异（上下文窗口、向量检索索引、外部记忆库等）。
- **持久化**: 基准评测的是跨多会话/长时间消息流的外部长期记忆与上下文记忆能力。其时间感知评测协议模拟用户-智能体随时间的交互流：在第 t 轮输入当前发言，t-1 轮及更早内容只能通过记忆机制召回——以此考察记忆的持久化与跨会话保持能力。被评测机制覆盖 in-context（FullMemory/RecentMemory）与外部持久存储（Retrieval/MemoryBank/MemGPT 等）两类持久性，但基准聚焦非参数化记忆，不涉及参数化（权重内）记忆。

**核心机制 / Mechanisms**

- **写入/编码**: （基准视角）数据构造与「写入」协议：①数据生成借鉴 MemSim（Zhang et al. 2024b，贝叶斯式记忆评测模拟器）流程，先采样用户关系图（profile + 关联人物/事件/地点/物品），并基于 MovieLens、Food、Goodreads 三个推荐数据集抽取真实世界高层偏好分布、用 GPT-4o-mini 总结高层偏好、构建「高层偏好↔低层事实属性」的一对多映射；②再由属性（含间接时间引用如「next Monday」）生成证据对话/消息及多选题，并把无关 News 数据（twitter-news）作为噪声会话按比例随机插入相邻会话，构造平均 100k token 以上的长测试样本以调节难度。③评测时各记忆机制按自身策略对每一轮消息执行写入（write）操作，基准记录其每次写入耗时（WT，秒/次操作）作为效率指标——这是基准对「写入编码」机制的量化考察维度，而非自身提出新的编码方法。参与场景下记忆还需存储智能体预定义的回复（以剥离推理模块影响）。
- **检索机制**: （基准视角）基准统一为涉及检索的机制配置 multilingual-e5-small 嵌入做相似度检索，并把「检索质量」本身列为核心评测维度：通过在构造对话时预先标注用于回答问题的「关键证据对话轮」（key evidence dialogue turn），可量化 Memory Recall@10（检索召回率，参与与观察场景分别报告）。同时记录每种机制的读取耗时 RT（秒/次操作）作为效率指标。被评测机制各自采用不同读取策略（全量、近期窗口、向量相似检索、Generative Agents 的 recency·importance·relevance 三因子打分、MemGPT 的分层调页、SCMemory 的自控召回），基准对其检索有效性与效率统一打分，而非提出新的检索算法。
- **反思/巩固**: 「反思记忆」是 MemBench 区别于既有基准（LoCoMo/LongMemEval/PerLTQA 仅评事实记忆）的核心贡献之一：它专门评测记忆机制能否从用户对大量低层偏好/事实的零散表达中归纳、总结出未被明确陈述的高层偏好（reflective summarization），即考察机制的「原始观察→高层洞见」抽象与巩固能力。为提高答案可信度，数据中通过对不同事实偏好/属性的多次表达来强化对该高层偏好的支撑。实验发现 GenerativeAgent、MemGPT、MemoryBank 这类带反思/总结机制的方法在小规模子数据集 1 上反思记忆表现很好，但在 100k 长上下文子数据集 2 上显著下降（疑因上下文窗口受限或遗忘机制丢失关键记忆），仅基于检索的 RetrievalMemory 仍保持较好结果——揭示「长期交互后如何维持反思能力」是开放难题。基准本身不实现反思机制，而是为反思能力提供专门评测台。
- **遗忘/更新**: 基准把「知识更新（knowledge updating）」列为事实记忆的一个子评测能力：通过设计同一属性在不同时间被不同表达的题目，考察记忆机制能否随时间正确更新知识、给出最新值。对「遗忘」的处理体现在分析层——观察到 MemoryBank（含 Ebbinghaus 式遗忘曲线）等带遗忘机制的方法在 100k 长上下文下因丢失关键记忆而性能下降，从而揭示遗忘机制的副作用。基准自身不提供 ADD/UPDATE/DELETE 算子，而是评测被测机制的更新/遗忘行为后果。
- **经验回放 (核心主题)**: 基本不适用。MemBench 是 user-centric 的个人助理记忆评测基准，关注「记住用户信息以正确作答」，而非智能体复用自身历史轨迹来自我改进。其参与场景虽会存储智能体的（预定义）回复并要求记忆，但目的是评测记忆保真度，不涉及把过去成功/失败轨迹作为范例/技能/回放缓冲来提升后续行为。因此本基准与「经验回放/轨迹复用」这一 agent-centric 主题关系很弱，更适合评测 Mem0/Zep/MemoryBank 类用户记忆系统。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric）评测设定。所有被评测记忆机制均在冻结基座模型（默认 Qwen2.5-7B-Instruct，另比较 GPT-4o-mini、Meta-Llama-3.1-8B-Instruct、glm-4-9b-chat）之上以提示/外部记忆方式运行，不做任何梯度更新；基准本身评测的是 inference-time、非参数化的记忆读写与抽象能力，不涉及参数化学习或训练。
- **失败学习 (核心主题)**: 不适用。作为 user-centric 记忆评测基准，MemBench 不涉及「检测失败轨迹并据此改进」的 agent-centric 失败学习主题。它通过多选题的对错来量化记忆机制是否正确召回/抽象信息，但这是对记忆有效性的评分，而非让智能体从自身失败经验中学习负例或失败模式。基准未设计失败反思、负面范例或错误驱动规则等机制评测。
- **技能/程序归纳**: 不适用。基准聚焦事实记忆与反思（偏好总结）记忆，不评测从经验中归纳可复用技能/工作流/程序（procedural induction）的能力；也不涉及技能表示与调用。其「反思记忆」是对用户高层偏好的归纳总结，而非可执行技能的诱导。
- **在线 vs 离线**: 评测协议为在线/流式（online，模拟时间流）：逐轮输入用户当前发言，历史只能经记忆召回，模拟智能体在真实部署中随时间累积记忆的过程。数据集构造本身是离线批量生成（基于关系图采样 + LLM 扩展 + 噪声注入）。即数据离线构建、记忆能力以在线时间感知方式评测。

**评测 / Evaluation**

- **任务领域**: 个人助理/多会话对话记忆评测（personal assistant / long-term conversational memory）。围绕用户画像与关联实体（人物、事件、地点、物品）的个性化记忆问答，覆盖两类交互场景：①参与场景（Participation，第一人称，智能体与用户对话交互）；②观察场景（Observation，第三人称，智能体被动记录用户消息流）。任务形式统一为多选题问答，涵盖信息抽取、（间接表达的）时间推理、知识更新、单/多会话跨会话推理、反思式偏好总结。不涉及网页导航/具身/编码/GUI 等智能体行动域。
- **基准**: MemBench 本身即为新提出的基准与数据集。其数据集规模（Table 2）：参与-反思 PS-RM 3.5k 会话/题/轨迹（平均 2,195 token/轨迹 TPT）、参与-事实 PS-FM 51k 会话·39k 题·8k 轨迹（TPT 10,285）、观察-反思 OS-RM 2k（TPT 745）、观察-事实 OS-FM 8.5k（TPT 617）；含 500 个用户关系图。论文中实际采样两套子集：子数据集 1（普通规模，参与每会话约 10K token：120 反思 + 360 事实；观察每消息列约 1K token：60 反思 + 280 事实）与子数据集 2（参与约 100K token：30 反思 + 90 事实；观察约 10K token：15 反思 + 84 事实）。相关/对比的既有数据集：LoCoMo、LongMemEval、PerLTQA、PersonaChat、LongBench、L-Eval（均被指出仅含事实记忆、仅参与场景）；构造借鉴 MemSim，评测实现基于 MemEngine；噪声数据来自 twitter-news（News）数据集。
- **报告增益**: 本文是评测基准，核心产出是对七种现有记忆机制的横向测评结果（非提出新方法的增益）。关键数字（基座 Qwen2.5-7B，Table 3/4，Accuracy 满分 1.0；RT/WT 单位秒/次操作）：①事实记忆-参与场景准确率：FullMemory 0.647(10k)/0.489(100k)、RecentMemory 0.639/0.422、RetrievalMemory 0.692/0.833、GenerativeAgent 0.478/0.455、MemoryBank 0.442/0.456、MemGPT 0.455/0.411、SCMemory 0.355/0.444；②事实记忆-观察场景准确率（1k/100k）：FullMemory 0.786/0.631、RecentMemory 0.800/0.512、RetrievalMemory 0.883/0.933 等；③事实记忆 Recall@10（仅 RetrievalMemory 有意义）：参与 0.776(10k)/0.749(100k)、观察 0.847/0.769；④反思记忆-参与准确率（10k/100k）：FullMemory 0.733/0.533、GenerativeAgent 0.742/0.333、RetrievalMemory 0.692/0.833、MemoryBank 0.692/0.400 等；⑤效率对比：MemGPT 读取耗时极高（参与 RT 高达 4.549s，远超其它机制 ~0.001–0.045s），MemoryBank 写入耗时极高（观察 WT 高达 18.243s，因其写入需 LLM 总结）。核心发现：FullMemory/RetrievalMemory/RecentMemory 在短上下文（子集 1）表现最好；上下文增至 100k 后 FullMemory/RecentMemory 因目标信息落出窗口而下降（RecentMemory 因窗口更小下降更明显），唯有检索式 RetrievalMemory 在长上下文下仍稳健甚至更优；带反思/遗忘机制的 GenerativeAgent/MemGPT/MemoryBank 在长上下文下显著退化。跨模型对比（Table 5，子集 1）：GPT-4o-mini 多数情况下为最佳基座（事实-参与 FullMemory 达 0.736），Llama-3.1-8B 事实记忆较弱但反思记忆尚可。容量测试（Fig.5，观察 100k）：MemGPT 与 SCMemory 随 token 增长准确率出现急剧下降，暴露其容量上限。
- **对比基线**: 本文横向评测的七种记忆机制即为对比对象（均基于 MemEngine 实现、Qwen2.5-7B 基座）：①FullMemory（全量上下文记忆）；②RecentMemory（近期窗口记忆）；③RetrievalMemory（向量检索式记忆，multilingual-e5-small）；④GenerativeAgent（Park et al. 2023 生成式智能体记忆流，含反思）；⑤MemoryBank（Zhong et al. 2024，含 Ebbinghaus 遗忘曲线）；⑥MemGPT（Packer et al. 2023，分层操作系统式记忆）；⑦Self-Controlled Memory / SCMemory（Wang et al. 2023，自控记忆框架）。FullMemory/RecentMemory 实质充当「全上下文 / 近期窗口」对照，RetrievalMemory 充当 RAG 对照。

**分析 / Analysis**

- **关键创新**: 首个同时在「记忆层级」与「交互场景」两个维度上扩展、并配套「有效性 + 效率 + 容量」多指标的 LLM 智能体记忆评测基准：①首次显式强调并系统评测反思记忆（reflective memory，高层偏好归纳）而不止于事实记忆；②首次提出观察场景（observation，第三人称被动记录）与参与场景（participation，第一人称交互）并存的评测；③提供四类指标——准确率 Accuracy、检索召回 Recall、容量 Capacity、时间效率 Efficiency（读/写耗时 RT/WT）；④采用时间感知、模拟记忆流的评测协议（逐轮输入、历史只能经记忆召回），并用可控比例的 News 噪声把样本扩展至 100k+ token 以调节难度与考察容量上限，比 LoCoMo/LongMemEval 等更贴合智能体实际记忆过程而非朴素长上下文评测。
- **局限**: ①作者自承评测受限于「对结构化数据的记忆评测」——数据集源于用户与关联实体 profile 构成的关系图，主要考察对结构化属性的记忆/结构化能力，覆盖面受此限制；②反思记忆仍有大量未探索方向（如用户情感记忆 emotional memory 未涵盖）；③评测题型统一为多选题（以规避自由表达带来的判分偏差），可能与开放式生成的真实记忆使用存在差距；④实验中实际仅对子集做均匀采样测评（子集 1 数百条、子集 2 仅数十条/类），样本量偏小，统计稳健性有限；⑤主要以 Qwen2.5-7B 为基座（跨模型对比也仅 7B–9B 级小模型 + GPT-4o-mini），未覆盖更大/更新模型；⑥被评测的部分先进记忆机制表现不佳，作者归因为「这些机制本身的缺陷」，但也可能与统一实现/适配（基于 MemEngine）方式有关；⑦数据生成依赖 LLM（GPT-4o-mini）扩展，可能引入偏差或风格化痕迹。
- **与其他工作关系**: 属「F. 记忆评测基准」簇，与本研究中其它记忆系统形成「评测台 vs 被评测对象」的关系。直接评测了本研究覆盖的多个系统/机制：B1 Generative Agents（GenerativeAgent 反思记忆流）、B2 MemoryBank（含 Ebbinghaus 遗忘曲线）、B3 MemGPT（分层 OS 式记忆），以及 RetrievalMemory（RAG 基线，关联 D 簇检索式记忆）、SCMemory 等。技术血缘：数据构造直接扩展同团队的 MemSim（Zhang et al. 2024b，贝叶斯记忆评测模拟器），把其仅有的观察场景与事实记忆扩展到参与场景 + 反思记忆；评测实现基于同团队 MemEngine（Zhang et al. 2025，统一模块化记忆库）；与同团队综述《A Survey on the Memory Mechanism of LLM-based Agents》（Zhang et al. 2024）一脉相承。定位上与 LoCoMo、LongMemEval、PerLTQA 同为 user-centric 长期对话/个人助理记忆基准，但宣称「首个强调反思记忆、首个提出观察场景、采用更贴合智能体记忆过程的评测方法与更全面指标」，是对这些前作的直接超越。被后续记忆系统（如 EvolveMem）用作与 LoCoMo 并列的标准评测基准。属 user-centric 评测，与 A 簇 ReasoningBank / D5 G-Memory 等 agent-centric 自进化记忆评测正交。
- **可复现性**: 可复现性中等偏好：官方开源（github.com/import-myself/Membench，约 52 stars，MIT 许可），提供采样后的 data2test（0-10k 与 100k）数据集、NoiseData 噪声数据与 makenoise.py 噪声生成脚本，完整数据集经百度网盘/Google Drive 分发；评测基于公开的 MemEngine 库与公开基座模型（Qwen2.5-7B、GPT-4o-mini 等）、公开嵌入模型 multilingual-e5-small。所用第三方数据（MovieLens、Food、Goodreads、twitter-news）均公开。局限：仓库以数据/脚本为主、贡献者仅 1 人、缺少端到端一键复现脚本与详尽实验配置说明，部分构造依赖闭源 GPT-4o-mini；论文实际测评样本量偏小。整体作为标准评测基准已被社区采用引用。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否，且不适用。MemBench 是评测基准，不学习记忆管理策略；它评测的七种机制均为启发式/规则式管线（全量、近期窗口、向量检索、生成式反思、遗忘曲线、分层调页、自控），无一采用 RL/训练来学习「何时存/取/更新」的记忆策略。基准本身亦未设计针对「学习型记忆控制」（Memory-R1/Mem-α 类）的专门评测任务。
- **记忆主体**: 用户中心（user-centric）。MemBench 评测「智能体记住用户及其关联实体信息以正确作答/个性化」的能力——用户画像、亲友/事件/地点/物品属性、高层偏好等，与 Mem0/Zep/LongMemEval/LoCoMo 同属用户记忆评测范畴；不评测智能体记忆自身经验以自我改进的 agent-centric 能力。其反思记忆（用户高层偏好归纳）尤其强调个性化服务质量提升。
- **多智能体记忆**: 单智能体（single-agent）。评测设定为单个个人助理智能体与单一用户的交互（参与或观察），不涉及多智能体间共享/路由记忆，也不评测 G-Memory/MIRIX 式多智能体记忆分层。
- **时序推理支持**: 是，显式支持且为重点评测维度之一。①数据中事件时间常以间接/相对引用表达（如「next Monday」「the week after next Sat 9:00 AM」），并提供绝对时间标签（如「2024-10-07 Monday 19:00」），考察智能体即时转换/推理具体时间的能力（temporal reasoning，事实记忆子能力）；②会话采用基于时间的划分——同一会话内逐轮时间戳连续（间隔约 1 分钟），跨会话保持时序且相邻会话间隔较长（如 1 天）；③整体评测协议为时间感知的记忆流模拟（逐轮按时间输入、历史经记忆召回）。但其时间建模偏「事件时间的抽取与换算」，未显式建模事实有效性窗口/双时间区间或事件日历（区别于 Zep/Graphiti 的时序知识图）。
- **模态**: 纯文本（text-only）。数据集为文本化的用户-智能体对话、用户消息流、文本化用户/实体 profile 与多选题；不涉及图像/截图/视觉或音频等多模态记忆评测。
- **过度个性化/记忆安全风险**: 基本未涉及该负面安全维度。MemBench 关注记忆的有效性/效率/容量与正确个性化，未设计针对有害/过时/侵入式/谄媚记忆、隐私治理（如 OP-Bench/Causal-LoCoMo）的评测任务；伦理声明仅泛泛提及所用公开数据合规、LLM 生成内容可能带偏见/有害输出并呼吁负责任使用。值得注意的是，其用户画像数据含 SSN、护照号、银行账号、驾照等高度敏感的合成 PII 字段，凸显此类个人记忆数据的隐私治理重要性，但论文未就此展开安全分析。
- **冲突/矛盾处理**: 部分涉及（以「知识更新」形式）。基准设计了「同一属性随时间被不同表达」的知识更新题，考察记忆机制能否正确采纳最新值、解决新旧值之间的更新；构造噪声时也刻意保证噪声内容不与评测记忆产生事实冲突。但它未提供针对显式矛盾事实检测/合并/仲裁（如 Memory-R1 UPDATE、MEMTRACK 冲突追踪）的专门评测维度，冲突处理主要隐含在「随时间更新取最新」这一能力点中。
- **token成本/延迟证据**: 效率是 MemBench 的核心评测维度之一，但量化的是各被测机制的读/写延迟（时间效率）而非自身系统的成本节省。具体（秒/次操作，Qwen2.5-7B）：多数机制读写耗时极低（FullMemory/RecentMemory RT/WT 约 0.001s 或 <0.001s；Retrieval RT ~0.024–0.041s、WT ~0.026–0.058s）；而 MemGPT 读取显著偏慢（参与场景 RT 高达 4.549s、观察 1.541s，因频繁分层调页/LLM 调用），MemoryBank 写入显著偏慢（观察场景 WT 高达 18.243s、参与 8.047s，因写入需 LLM 总结），GenerativeAgent 写入也较慢（WT ~6s）。容量维度（Fig.5）显示 MemGPT、SCMemory 在 token 增长至 100k 时准确率急剧下降，暴露记忆容量上限。基准用「平均每单位噪声长度约增加 1k token」来刻画上下文规模与难度。本基准不报告自身相对全上下文的延迟/token 节省百分比（区别于 Mem0/Zep 类系统口径）。


<a id="f4-evo-memory自演化记忆流式基准与框架配套基线方法-exprag-与提出的-remem行动-思考-记忆精炼流水线"></a>

### F4 Evo-Memory

*Evo-Memory（自演化记忆流式基准与框架；配套基线方法 ExpRAG 与提出的 ReMem“行动-思考-记忆精炼”流水线）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本 2511.20857，首次公开 2025-11-25，v1）
- **作者/机构**: Tianxin Wei（魏天欣，第一作者，通讯 twei10@illinois.edu，工作完成于在 Google DeepMind 期间）、Noveen Sachdeva、Benjamin Coleman、Zhankui He、Yuanchen Bei、Xuying Ning、Mengting Ai、Yunzhe Li、Jingrui He（何静蕊）、Ed H. Chi、Chi Wang、Shuo Chen、Fernando Pereira、Wang-Cheng Kang、Derek Zhiyuan Cheng。主要单位为伊利诺伊大学厄巴纳-香槟分校（UIUC）与 Google DeepMind（多数作者来自 DeepMind，第一作者与若干 UIUC 学生合作）。属学术界与工业研究院（Google DeepMind）合作。
- **论文链接**: https://arxiv.org/abs/2511.20857
- **代码链接**: https://github.com/zhaosnw/evo_mem （MIT 许可证；约 16 stars、4 forks、2 位贡献者（含 Claude 协作）、6 次提交，2025-12 发布；含 ExpRAG/ReMem/ReAct/A-mem/Self-RAG/Mem0/LangMem/Dynamic Cheatsheet/AWM 等代理实现、单/多轮数据集与评测流水线；数据托管 HuggingFace）

**记忆分类 / Taxonomy**

- **记忆类型**: 作为基准/框架，Evo-Memory 不绑定单一记忆类型，而是统一评测覆盖多类记忆的代理。其核心关注“经验/程序性记忆”（procedural / experiential，即从过往任务轨迹中抽象可复用的推理策略与工作流），区别于以往聚焦“情景/语义对话记忆”（episodic/semantic 的会话事实回忆）的评测。被统一的 10+ 模块跨越情景记忆（如 A-mem 缓存近期观测与反思）、语义/事实检索记忆（如 Mem0、MemOS、LangMem 的读写更新）、程序性工作流记忆（AWM、Dynamic Cheatsheet）。提出的 ReMem 同时操作工作记忆（当前推理轨迹）与外部经验记忆，强调“记忆推理”这一新维度。
- **记忆结构**: 框架本身以统一抽象 (F, U, R, C) 容纳多种底层结构：原始经验缓冲（ExpRecent 的近期轨迹）、向量化经验库（ExpRAG/Self-RAG 的可检索经验文本）、分层/结构化记忆（Mem0、MemOS）、工作流/“小抄”式程序记忆（AWM、Dynamic Cheatsheet 的累积/合成两种变体）。ExpRAG 将每条交互编码为结构化经验文本条目 m_i=S(x_i, ŷ_i, f_i)（输入、模型输出、反馈/正确性信号）；ReMem 在此之上维护可被代理主动“检索-剪枝-重组”的经验记忆对象。
- **存储后端**: 外部可检索记忆存储为主：用 BAAI/bge-base-en-v1.5 编码器对查询与记忆条目做嵌入，存入嵌入索引并做 top-k 相似度检索（默认 k=4），保证各方法统一检索预算；基座模型权重不更新（无参数化记忆）。开源实现以 Python 内存数据结构 + 嵌入检索实现记忆库，支持 OpenAI/Anthropic/Google 多后端 LLM；数据集托管于 HuggingFace。未使用专用图数据库或外部向量数据库服务（轻量自实现）。
- **持久化**: 外部、跨任务流持久化（durable external store，非参数化）：记忆状态 M_t 随任务流逐步演化并在整段流式部署中持久保存与复用，独立于基座 LLM 权重；模型本身静态、不做梯度更新。记忆在“流”内跨任务累积（在线、逐 episode 写入与检索），但不写回模型参数。

**核心机制 / Mechanisms**

- **写入/编码**: Evo-Memory 的统一“演化（Evolve）”步骤：在产生输出 ŷ_t 后，代理构造新记忆条目 m_t = h(x_t, ŷ_t, f_t)，把当前步的经验连同反馈 f_t（如任务是否完成的正确性信号）一并编码，再经更新算子 M_{t+1}=U(M_t, m_t) 写入记忆。不同模块对 U 的实现不同：检索类记忆为直接追加（append），长期存储为摘要/压缩（summarization/compression），有界容量存储为替换（replacement）。在 ExpRAG 中，写编码即把 (x_t, ŷ_t, f_t) 用模板 S 结构化为经验文本并并入记忆 M_{t+1}=M_t∪{(x_t, ŷ_t, f_t)}；ExpRecent 维护近期任务轨迹的精简情景痕迹。在 ReMem 中，写编码与代理的 Think/Act/Refine 决策耦合——Refine 操作显式对记忆做精炼后写入，使写入不再是被动追加而是经过元推理筛选/重组。统一基准把所有方法都纳入“search→synthesis/predict→evolve”三步循环以隔离记忆设计差异。
- **检索机制**: 统一“搜索（Search）”步骤：给定当前输入 x_t，先从演化中的记忆检索相关条目 R_t = R(M_t, x_t)，R 可实例化为相似度检索、索引查表或对存储嵌入的注意力。基准统一用 bge-base-en-v1.5 把查询与记忆项嵌入，按相似度取 top-k（默认 k=4）注入上下文，所有方法共享同一检索池与预算以保证可比。ExpRAG 形式化为 R_t = Top-k_{m_i∈M_t} φ(x_t, m_i)（按检索分 φ 取前 k 条相似经验）。部分方法（Self-RAG、ReMem）在同一检索池之上叠加自适应的“是否检索/检索什么”的推理判断：ReMem 通过 Refine 操作主动检索-剪枝-重组记忆，实现“记忆推理”而非被动拼接。检索结果按相关性从高到低顺序追加到提示，并与任务输入一同截断到统一长度约束。
- **反思/巩固**: 框架的“合成（Synthesis）”与 ReMem 的 Refine 共同承担原始→抽象的整合。Synthesis 把检索到的 R_t 重组为面向当前输入的工作上下文 C̃_t=C(x_t, R_t)，可表现为构造结构化提示（AWM 工作流）、选择关键记忆项（Mem0）、或把检索内容合并成简短摘要（Dynamic Cheatsheet）。ReMem 引入显式“记忆推理”维度：Refine 操作对记忆做元推理——挖掘有用经验、剪除噪声、重组 M_t——可在单步内多次执行 Think 与 Refine，直到选择 Act 才结束该步，从而把“反思/巩固”从离线后处理变为推理过程中实时进行的记忆演化。论文强调正是这种持续反思与记忆精炼显著提升了程序性知识的累积（多轮任务增益尤为明显）。
- **遗忘/更新**: 更新算子 U 因方法而异：检索类为追加；长期存储为摘要/压缩；有界容量存储为替换；ReMem 的 Refine 显式做剪枝（pruning）与重组以去噪。论文报告 ReMem 在含失败经验时仍稳健（RQ4），通过主动精炼避免“未过滤失败”污染记忆；附录另有记忆剪枝率分析（Appendix B.2）。框架不内置 Ebbinghaus 时间衰减或显式失效机制，遗忘主要由替换/剪枝/重组体现。
- **经验回放 (核心主题)**: 核心主题。Evo-Memory 把传统静态数据集重构为顺序任务流（streaming task stream），其中早期任务携带对后续任务有用的策略；代理须在每次交互后检索、整合并演化记忆，从而显式考核“经验复用（experience reuse）”而非“会话回忆（conversational recall）”。论文用方程对比：会话回忆检索过去事实（如方程 2x²+3x−1=0 的解），而经验复用回忆推理策略（如使用求根公式）。ExpRAG 即“一次性经验复用”——把过往任务-输出-反馈结构化为经验文本，对新任务检索 top-k 相似经验作为 in-context 范例。ReMem 进一步在推理回路中交错使用并实时精炼经验。RQ2 表明 ReMem 的增益与“数据集内任务相似度”强相关（Gemini 2.5 Flash 上 Pearson r=0.717、Claude 3.7 Sonnet 上 r=0.563）：结构重复度高的 PDDL/AlfWorld 复用增益最大，AIME-25/GPQA 等多样任务增益较小，说明可迁移经验的存在与否决定复用价值。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric / 测试时、in-context、检索-提示层面）。所有方法均在不更新基座 LLM 权重的前提下，通过外部记忆的检索、合成与演化实现“测试时学习/测试时演化（test-time learning / test-time evolution）”。论文明确把基座能力差异排除在研究焦点之外，只隔离记忆架构与更新策略的影响；提出的 ReMem 是“轻量却强力”的非参数化持续适应范式，无需任何重训即让小模型表现得更强。
- **失败学习 (核心主题)**: 核心主题且为 RQ4 专门研究。反馈 f_t 被设为正确性信号，成功与失败经验均可写入记忆。RQ4（Table 3，AlfWorld 与 ScienceWorld 同时存入成功与失败经验）发现：朴素累积记忆的基线在“未过滤失败”下显著退化，说明天真地累积失败会引入噪声、干扰检索；而自演化方法尤其 ReMem 通过主动精炼/筛选存储经验保持稳健，取得最佳成功率与进度率（Claude 3.7 上 0.81/0.94，Gemini 2.5 Flash 上 0.54/0.76）。论文据此强调“选择性利用与记忆精炼”对稳定测试时适应的重要性，并明确呼吁未来研究“失败感知的记忆演化（failure-aware memory evolution）”。此外 RQ3 发现来自更难任务的成功经验更可迁移（Hard→Easy 序列 ReMem 达 0.94/0.97），间接体现从困难/失败中提炼可迁移知识的价值。Evo-Memory 本身不规定固定的失败学习算法，而是提供评测失败-鲁棒性的协议。
- **技能/程序归纳**: 是——这是 Evo-Memory 区别于会话记忆基准的关键考核点：考核代理能否从过往轨迹中抽象“可复用的推理策略/工作流/程序”。被统一的模块中 AWM（Agent Workflow Memory）与 Dynamic Cheatsheet（累积 Cu / 合成 RS 两变体）专门做程序性“how-to”知识复用；ReMem 通过 Refine 把经验重组为可复用知识。论文发现程序性记忆方法在结构化领域（如 AIME）表现尚可，但在科学推理与工具使用上灵活性受限；轻量的 ExpRAG/ExpRecent 反而常优于更复杂设计，提示“任务级显式利用”被低估。
- **在线 vs 离线**: 在线（online）为主：记忆在流式部署中逐任务（per-step / per-episode）在线写入、检索与演化，模拟真实连续任务流。基准将静态数据集离线重构为固定顺序的任务流（统一序列排序以保证演化动态一致与公平），但记忆的累积-适应过程发生在线。RQ3 通过构造 Easy→Hard 与 Hard→Easy 两种任务序列方向考核序列鲁棒性。

**评测 / Evaluation**

- **任务领域**: 覆盖两大类、跨多领域：①单轮推理与问答（事实知识、研究生级科学推理、奥数符号推理、工具/API 使用）；②多轮目标导向的具身/交互环境（家务指令跟随、网格导航与组合推理、开放式科学实验、符号规划）。强调随任务时域变长，持续适应价值越大（多轮增益显著大于单轮）。
- **基准**: 共 10 个数据集。单轮：MMLU-Pro（多学科推理，含经济/工程/哲学子域）、GPQA-Diamond（研究生级“Google-proof”科学题）、AIME-24 与 AIME-25（奥数，精确匹配）、ToolBench（工具/API 调用，来自 Gorilla/patil2023）。多轮（具身/交互，部分取自 AgentBoard）：AlfWorld（家务指令跟随）、BabyAI（网格导航/组合推理）、ScienceWorld（开放式科学实验）、PDDL（符号规划）。检索编码器：BAAI/bge-base-en-v1.5。基座 LLM：Gemini-2.5 系列（Flash、Flash-Lite、Pro）与 Claude 系列（3.5-Haiku、3.7-Sonnet）。
- **报告增益**: 统一“search–predict–evolve”协议下（隔离记忆设计）。单轮（Table 1a）：Gemini 2.5 Flash 上 ReMem 平均精确匹配 0.65、ToolBench API/Acc 0.85/0.71，ExpRAG 平均 0.60（均优于 AWM 0.56、Dynamic Cheatsheet 变体约 0.56–0.58、Mem0/MemOS/SelfRAG 约 0.59 等更复杂设计），相对 Baseline 0.59、History 0.58 提升；Claude 3.7 Sonnet 上 ExpRAG 平均 0.59、ReMem 0.58（GPQA 0.70/0.67、ToolBench 0.88/0.72 等显著高于基线 0.54）。多轮（Table 1b，S=成功率/P=进度率）：Claude 3.7 Sonnet 上 ReMem 大幅领先——AlfWorld 0.92/0.96、BabyAI 0.73/0.83、PDDL 0.83/0.95、ScienceWorld 0.62/0.89，平均 0.78/0.91（对比 Baseline 0.24/0.52、History 0.49/0.74、ReAct 0.57/0.79、Mem0 0.50/0.75、AWM 0.49/0.74）；Gemini 2.5 Flash 上 ReMem 平均 0.50/0.64、ExpRAG 0.46/0.63（均优于 Baseline 0.27/0.46）。步效率（Figure 4）：ReMem 在 AlfWorld 把平均完成步数从 History 的 22.6 降至 11.5。序列鲁棒性（RQ3, Table 2）：ReMem 在 Hard→Easy 达 0.94/0.97（AlfWorld+ScienceWorld 平均 0.81/0.94），显著优于基线。失败经验鲁棒（RQ4, Table 3）：含失败经验时 ReMem 仍取 0.81/0.94（Claude）。论文指出小模型受益最大——测试时精炼是增强轻量 LLM 的有效途径。
- **对比基线**: ①无外部/程序记忆：Baseline（无记忆）、History（全历史上下文）、ReAct、A-mem（轻量缓存近期观测与反思）；②自适应代理记忆：SelfRAG、MemOS、Mem0、LangMem（动态检索与读写更新）；③程序性记忆：Dynamic Cheatsheet（DC-Cu 累积 / DC-RS 合成）、AWM（Agent Workflow Memory）；④本文提出的自演化框架：ExpRecent、ExpRAG、ReMem。其中 ExpRAG 作为“最小经验复用基线”用以把增益归因于任务级经验检索本身。

**分析 / Analysis**

- **关键创新**: 首次提出面向“自演化记忆/测试时演化”的统一流式基准与框架：把静态数据集重构为顺序任务流，强制代理在每次交互后检索-整合-演化记忆，从而把评测焦点从“会话回忆”转向“经验复用”——考核代理能否跨任务流累积并复用推理策略而非仅回放上下文。框架以统一抽象 (F, U, R, C) 与 Search–Synthesize–Evolve 循环容纳并公平比较 10+ 代表性记忆模块（检索式/工作流/分层），跨 10 个单轮与多轮数据集评测，并提供四类记忆中心指标（答案准确率、成功率、步效率、序列鲁棒性）。同时给出最小基线 ExpRAG 与提出 ReMem（行动-思考-记忆精炼流水线），ReMem 把“记忆推理”作为与 Think/Act 并列的第三维操作，将记忆从被动上下文升级为代理可实时检索-剪枝-重组的对象，建立测试时持续自我改进的新范式。
- **局限**: ①作为基准其核心贡献是评测协议而非新记忆算法，记忆方法本身的创新（ReMem）仍以现成 LLM 与启发式/提示驱动的 Think-Act-Refine 实现，非用 RL 学习记忆管理策略；②增益高度依赖“数据集内任务相似度”（RQ2），对多样/低相似任务（AIME-25、GPQA）复用收益有限，泛化到真正异质流尚未验证；③静态数据集“折叠/重构”为任务流近似在线学习，并非真实分布漂移下的持续反馈流（后续 Live-Evo 等正指出此局限）；④仅评测两族基座（Gemini-2.5、Claude-3.5/3.7），且部分方法（MemOS、LangMem）因与具身环境不兼容被排除在多轮评测之外，覆盖不完全；⑤纯文本设定，未涉及多模态/视觉/真实工具执行的延迟与成本；⑥未讨论记忆安全、过度个性化、隐私治理等负面维度；⑦未报告显式的 token/延迟成本百分比（仅以步效率间接衡量效率）。
- **与其他工作关系**: 属本研究 F 类“记忆评测基准”。定位为弥合“会话回忆类基准”与“经验复用/自演化”之间的空白，明确区别并超越 StreamBench（仅测序列学习/事实保持，缺推理与轨迹复用）、LifelongAgentBench（重保持、不建模记忆结构与更新）、LongMemEval / LoCoMo 类长程对话一致性评测（不测部署中记忆演化）。其统一抽象 (F,U,R,C) 与 search–predict–evolve 循环把本研究中众多系统纳入同一框架公平比较：复用并对照 A-mem（B4 A-MEM，xu2025mem）、Mem0（D4）、MemOS（B7 Memory-OS 同系）、AWM（C2 Agent Workflow Memory）、Dynamic Cheatsheet、Self-RAG、LangMem 等。提出的 ReMem 在思想上扩展 ReAct（yao2023react）的动作空间（新增 Refine 记忆操作），并与 Reflexion（A1）、Voyager（C1）、Generative Agents（B1）等“反思/技能复用/自演化”工作同脉络，但以基准化方式统一评测。其“经验复用考核”与 ReasoningBank（A6）、ExpeL（A5）等智能体经验记忆方法关切一致。后续工作 Live-Evo（2026，2602.02369）明确以 Evo-Memory 为对照，指出其“折叠静态基准近似在线”的局限并推进真正在线演化。
- **可复现性**: 复现性较好但仓库较新、规模有限：代码开源于 GitHub（zhaosnw/evo_mem，MIT 许可，约 16 stars、4 forks、2 贡献者、6 次提交，2025-12），提供完整框架——记忆模块（嵌入/近期两种检索、上下文构造器）、十类代理实现（ExpRAG、ReMem 的 Think-Act-Refine、ReAct、A-mem、Self-RAG、Mem0、LangMem、Dynamic Cheatsheet、AWM、ExpRecent）、OpenAI/Anthropic/Google 多 LLM 后端、单/多轮数据集与下载脚本（HuggingFace）、评测指标（准确率/成功率/进度率/步效率）、CLI 与批量运行器、各代理提示模板，并附论文 PDF。所用基准（MMLU-Pro、GPQA、AIME、ToolBench、AlfWorld、BabyAI、ScienceWorld、PDDL）均公开。论文承诺释放全部代码与配置。社区采用信号处于早期上升阶段（已被 Live-Evo 等引用）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式/LLM 提示驱动，非 RL 学习记忆策略）。ReMem 把记忆管理建模为一个马尔可夫决策过程（状态 s_t^n=(x_t, M_t, o^{1:n-1}_t)、动作空间 {Think, Act, Refine}），但其策略由现成 LLM 经提示选择操作，并非用强化学习训练记忆管理策略本身。论文在相关工作中明确把“策略驱动控制（policy-driven control，显式优化决定存/取/覆盖）”列为另一支线（如 MemAgent、Mem-π、Mem1、MemoryLLM、MemOS 等），Evo-Memory 自身定位为评测这些设计的统一基准。因此属于 2025–26“学习记忆控制”代际之外的（LLM 提示化但更灵活的）启发式范式；但其 MDP 形式化为未来用 RL 学习 Think-Act-Refine 策略留出接口。
- **记忆主体**: 智能体中心（agent-centric）为主：记忆的对象是代理自身的过往任务经验与推理策略，用于跨任务流自我改进（experience reuse / test-time self-improvement），明确区别于以记住用户信息做个性化的“用户中心”会话记忆（如 Mem0/Zep/LongMemEval 路线）。论文开篇即对比“记住别人说了什么（会话回忆）”与“记住自己学到了什么（经验复用）”，并把后者作为核心考核目标。
- **多智能体记忆**: 单智能体设定。Evo-Memory 评测单个有状态代理在任务流上的记忆演化，未涉及 G-Memory/MIRIX 式的多智能体共享/路由记忆、洞见-查询-交互分层或跨代理记忆传递。
- **时序推理支持**: 弱/有限。基准的“流式任务序列”本身蕴含时间/顺序结构（早期任务为后续提供策略，记忆随历史演化 M_t），RQ3 专门研究任务序列方向（Easy→Hard / Hard→Easy）对适应与泛化的影响、考核序列鲁棒性；但 Evo-Memory 不像 Zep/Graphiti 那样建模事实有效期窗口、事件双时间轴或时序事实更新，时序推理非其主打能力。
- **模态**: 纯文本（text-only）。所有任务输入、经验文本、记忆条目与具身环境观测均以文本表示；不涉及视觉/截图/视频/真实多模态记忆。
- **过度个性化/记忆安全风险**: 论文未讨论。Evo-Memory 不涉及过度个性化、谄媚、侵入性/过时记忆的安全治理，也无 OP-Bench/Causal-LoCoMo 类记忆安全或隐私评测。其相关“负面”讨论偏效用层面：RQ4 表明天真累积失败/噪声经验会损害检索与性能，强调“选择性利用与记忆精炼”，即“更多记忆并非总是更好”，但属噪声/干扰而非用户隐私/安全维度。
- **token成本/延迟证据**: 以“步效率（step efficiency）”而非 token/延迟百分比量化效率：统一检索预算（top-k=4、bge-base-en-v1.5、提示长度截断一致）以保证公平；Figure 4 显示自演化方法持续以更少步数完成任务，ReMem 减步最大且最稳（如 AlfWorld 平均完成步数从 History 的 22.6 降至 11.5），ExpRAG/ExpRecent 亦显著减步。论文未报告 p95 延迟或绝对 token 节省百分比（口径不同于 Mem0/Zep 的延迟百分比），但更少步数间接意味更低的累积推理 token 与时延。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)
- 冲突/矛盾处理 (`conflict_contradiction_handling`)
- 发表venue (`venue`)


<a id="f5-memtrack多平台动态智能体环境下的长期记忆与状态追踪评测基准全称-memtrack-evaluating-long-term-memory-and-state-tracking-in-multi-platform-dynamic-agent-environmentspatronus-ai-出品它不是记忆系统而是一个面向企业级-swe-工作流的容器化记忆评测基准环境跨-slacklineargitgitea-三平台模拟异步事件时间线考核记忆的获取选择冲突消解能力"></a>

### F5 MEMTRACK

*MEMTRACK（多平台动态智能体环境下的长期记忆与状态追踪评测基准；全称 MEMTRACK: Evaluating Long-Term Memory and State Tracking in Multi-Platform Dynamic Agent Environments；Patronus AI 出品。它不是记忆系统，而是一个面向企业级 SWE 工作流的容器化记忆评测基准/环境，跨 Slack、Linear、Git/Gitea 三平台模拟异步事件时间线，考核记忆的获取/选择/冲突消解能力）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本 2510.01353 首次公开于 2025-10-01，cs.* 领域；同年被 NeurIPS 2025 工作坊 SEA(Scaling Environments for Agents) 接收，OpenReview id mVxmbMng4B，2025-10 公布）
- **作者/机构**: Darshan Deshpande 与 Varun Gangal 为共同第一作者（标注 ∗ 同等贡献）；其余作者 Hersh Mehta、Anand Kannappan、Rebecca Qian、Peng Wang。全部隶属工业界初创公司 Patronus AI（AI 评测/护栏公司，邮箱后缀 @patronus.ai）。属工业界（industry）出品工作。
- **发表venue**: NeurIPS 2025 工作坊 SEA（Scaling Environments for Agents，扩展智能体环境工作坊），OpenReview forum id mVxmbMng4B，有 NeurIPS 2025 现场幻灯片；arXiv 预印本最早见于 2025-10（arXiv:2510.01353）。属工作坊论文（非主会正刊）。
- **论文链接**: https://arxiv.org/abs/2510.01353 （OpenReview: https://openreview.net/forum?id=mVxmbMng4B ；HuggingFace 论文页 https://huggingface.co/papers/2510.01353 ；官方博客 https://www.patronus.ai/blog/memtrack ；NeurIPS 幻灯片 https://neurips.cc/media/neurips-2025/Slides/124523.pdf）
- **引用数**: 约 9 次（Semantic Scholar 实时查询，Corpus ID 281724947 / paperId 164188a2e6326e187b4d1699f17adf413f0d675d，2025-10 新近发表，引用数较低，影响力/成熟度处于早期）

**记忆分类 / Taxonomy**

- **记忆类型**: 本身是"记忆评测基准/环境"而非记忆系统，对记忆后端保持中立(backend-agnostic)。它考核的记忆维度主要为情景记忆（episodic，跨 Slack/Linear/Git 平台、带时间戳的异步组织事件时间线）与语义记忆（semantic，从对话/工单/代码中抽取并维护的事实、决策、状态），并强调状态追踪(state tracking)；被测对象的工作记忆即 LLM 上下文窗口。按 Shan/Du 等综述的记忆三分法，MEMTRACK 同时覆盖获取(acquisition)、利用/选择(selection/utilization)与维护(maintenance，含冲突消解)三个环节，而非仅检索准确性。
- **记忆结构**: 对底层记忆数据结构不作规定（可评测任意结构：原始缓冲、向量库、知识图谱等）。基准自身的核心结构是"按时间顺序、跨平台交错的事件时间线" T=(E_1,...,E_n)：每个事件 E_i 带元信息（时间戳 τ、平台类型 P∈{Slack, Linear, Git}）与内容（Slack 消息/Linear 工单/Git 提交与文件系统）。时间线被加载到对应平台服务器，仅能通过各平台工具/通知按需访问，不向智能体整体暴露，从而强制跨平台上下文切换与信息保持。实验中接入的两类外部记忆代表两种结构：Mem0 为向量库(ChromaDB)结构，Zep 为带知识图谱的时序记忆结构。
- **存储后端**: 评测层面与后端无关。环境后端：Linear(及竞品 Jira 风格)做工单/项目跟踪、Slack 做沟通与实时通知服务器、Git 用 Gitea 自建、外加一个 Docker 化文件系统供智能体 git clone 后浏览代码。被测记忆组件后端：LLM+NoMem（无内置记忆）；LLM+Mem0（向量存储用 ChromaDB，嵌入用 gpt-4o-mini 的 LLM 嵌入，暴露 search_memory/store_memory/get_memories 工具）；LLM+Zep（知识图谱式时序记忆，用其默认 API 设置）。被测 LLM 为 gpt-5 与 gemini-2.5-pro（部分附录含 claude）；事件合成用 Claude-4-Sonnet；正确性裁判用 gpt-4o。数据以 JSON 实例形式发布（Google Drive）。
- **持久化**: 面向"外部持久化记忆(durable external store)"的在线交互式评测：智能体在一个实例内需在线解析、保持、更新跨平台事件信息，并被严格按顺序逐题(q_i)提问（智能体事先不知道一个实例共有几道问题，以防预先规划解答）。基准本身不规定记忆驻留方式——既可由长上下文模型纯 in-context 处理，也可挂接 Mem0/Zep 等外部持久存储。其设定（时间线平均 39.9 事件/约 4.01K tokens、最长 115 事件/11.1K tokens，时间跨度平均 878 小时、最长 3049 小时）正是为暴露跨长时程、跨平台记忆保持的不足而设计。

**核心机制 / Mechanisms**

- **写入/编码**: 作为基准，MEMTRACK 不规定写入编码，而是把"如何把跨平台异步事件编码进记忆"留给被测系统并加以考核。其数据构建端的"写入"逻辑由三套数据生成法决定：(1) 自底向上(Bottom-Up)——类似 SWE-Bench，先用带 web 搜索与 bash 的探索智能体从热门开源 GitHub 仓库挑选有已合并 PR(优先 <3 个文件改动、可确定性验证)的已关闭 issue 及讨论，经两名人工标注者删除无确定答案者后，再由另一智能体把 issue/解决方案/讨论改写成自然的 Slack 对话与 Linear 工单，并注入情境化反事实数据与无关干扰项(distractor)；(2) 自顶向下(Top-Down)——4 名产品/工程资深专家用自身经验描述真实跨团队问题如何被定义、沟通、迭代解决，再用 LLM 据描述生成时间线、专家手工校订并人工撰写问答对；(3) 混合(Hybrid)——专家高层构思+LLM 迭代细化，标注者逐步增加事件历史复杂度并据需细调问题。被测智能体侧的"写入"则体现在挂接的记忆组件：Mem0/Zep 通过 store_memory(content, metadata) 把读到的工单/对话/代码要点写入向量库/知识图谱。实验观察：被测 LLM 倾向于反复直接读取信息源而非主动调用 store_memory 写入记忆。
- **检索机制**: 基准不规定具体检索算法，而是把记忆访问映射为"工具调用"并系统度量其行为，借鉴工具学习中规划/选择/调用/响应生成的评估视角(因记忆常实现为工具的封装：向量库或图)。访问通道为各平台工具：Slack(get_unread_messages、get_channel_messages(channel,limit,after_id)、get_direct_messages、list_channels、list_users 等)、Linear(get_ticket(id)、list_tickets(team,status,...)、list_milestones 等)、Git(list_remote_git_repositories 及文件系统 list_directory/read_file/search_file_content/run_shell_command(git clone))；外接记忆组件额外暴露 search_memory(query,limit) 做混合语义+关键词检索、get_memories(limit) 取近期记忆。基准定义三项度量来刻画检索/访问行为：①Efficiency（工具调用经济性，公式 Efficiency=exp(-(|TC|-TC_min)/TC_min)，当 TC≥TC_min，否则为 1，TC 为访问时间线元素的工具调用数）；②Redundancy（冗余度，由 LLM 裁判判定后续工具调用是否被既有调用语义包含/重复，取冗余调用占比，需软匹配并识别"被包含"关系如 get_ticket(id) 与 list_all_tickets()）；③工具调用熵 H(TC_i)（跨平台访问的多样性，越高越好）。无内置相似度/PPR 算法，检索能力完全取决于被测系统及其挂接的记忆后端。
- **反思/巩固**: MEMTRACK 不实现反思/整合算法，而是"考核"被测系统能否把原始跨平台事件提炼、保持并复用为可回答问题的高层认知，并用度量暴露其缺失。相关工作部分把记忆获取归因于 Reflexion/MemInsight 式的 LLM 自反思以判定哪些记忆有用——但实验结论是：当前 SoTA LLM 既不擅长把信息整合进外部记忆、也不擅长复用——它们倾向于在隔了≥3 个回合后重复读取同一工单/Slack 消息/代码文件（"间隔后重复访问"模式），而非依赖已整合的记忆，导致工具调用冗余≥20%。其裁判式度量（正确性、冗余、效率）即是对"是否有效整合并复用记忆"的间接评估。数据构建端用 LLM 把 GitHub issue/PR 讨论"整合"成连贯多平台时间线，是一种离线的原始材料→情境化场景的整合，但与运行期 Reflexion 式经验洞见无关。
- **遗忘/更新**: 把"维护(maintenance)"——含遗忘/保留与冲突消解——作为核心考核维度之一，但不规定具体遗忘算法。基准通过在事件历史中注入噪声、反事实(counterfactual)、跨引用与会被后续事件修订/推翻的决策与状态变化，考核被测系统能否随时间正确更新状态、用最新信息覆盖过时事实、并消解矛盾(冲突消解 conflict resolution 是其三大记忆能力之一)。论文亦指出记忆框架"按需遗忘或保留"的能力当前研究不足。实验显示当前模型在此环节表现欠佳（后续问题正确率下降、跨平台矛盾难消解）。
- **经验回放 (核心主题)**: 不适用（本工作是评测基准而非智能体经验回放方法）。MEMTRACK 不维护成败轨迹缓冲、不做范例提示或技能复用来改进未来决策（与 ReasoningBank/Voyager/ExpeL 等智能体中心经验回放范式正交）。其"复用"含义是评测层面的：要求智能体在一个实例的多道顺序问题中，最大化复用先前回答时已获取的信息（避免重复工具调用），这正是 Redundancy 度量考核的"信息复用效率"——理想行为是答完早题后保留并复用已得信息以减少后续访问。它复用的是同一会话内跨平台读取到的组织事件信息，而非智能体自身过往任务轨迹/技能。

**学习维度 / Learning**

- **学习范式**: 非参数化(non-parametric / in-context, prompt-level / 工具增强)：被测系统全程冻结 LLM 参数、无梯度更新；记忆能力体现在上下文窗口 + 可选外接的非参数化记忆组件(Mem0 向量库 / Zep 知识图谱)上，纯靠提示与工具调用读写。基准定位为即插即用、对长上下文模型与挂接记忆组件的智能体系统皆适用，不涉及参数化记忆训练。其相关工作讨论了用强化环境/对齐技术提升智能体的方向，并把 MEMTRACK 视为可为此类训练提供数据/考核的环境，但本工作自身不做训练。
- **失败学习 (核心主题)**: 不适用 / 非失败学习型方法。MEMTRACK 不检测任务执行失败、不构建负例或错误规则、不对失败轨迹做自反思以改进策略（与 Reflexion/Retroformer/ExpeL 正交）。它评测的是"跨平台记忆-状态追踪-推理"的正确性与工具使用效率，而非"从失败中学习"。不过其数据生成协议含一个对抗式难度提升回路：自底向上生成时要求脚本反复迭代，"必须验证模型在你的数据集上失败"，若模型通过(数据太易)则继续加大问题复杂度或事件历史复杂度——这是用模型失败信号来加固基准难度，而非让智能体从失败中学习。论文用大量错误模式定性分析（如"先泛后精"式冗余访问、间隔后重复访问、渐进扩大探索）刻画被测模型的失败方式。
- **技能/程序归纳**: 不归纳可复用技能/工作流/程序。MEMTRACK 聚焦跨平台情景/语义记忆与状态追踪，不从经验中提炼程序性技能或可复用工作流（与 Voyager/AWM/Synapse 等技能诱导方法属不同范畴）。论文展望的未来工作是让智能体能在时间线中"主动行动"(创建 Linear 工单、发送 Slack 消息、参与组织时间线的后续演化)，但本初始工作刻意回避这一更难设定以先打磨生态有效的评测底座，故不涉及技能诱导。
- **在线 vs 离线**: 评测过程为在线交互式(online)：智能体在一个实例内实时通过工具访问跨平台时间线、在线保持/更新记忆，并被严格按顺序逐题提问（不预先告知问题总数以杜绝预先规划），平均每实例 3.2 题(最多 5 题，含初始题+可选追问)。数据集构建则是离线的：先离线从 GitHub 抓取已关闭 issue/PR 与讨论、专家离线撰写场景、LLM 离线合成时间线并人工校订、再离线生成问答对并人工求解验证完整性。事件注入与服务器加载在实例实例化(instantiation)时离线完成，运行评测时则在线交互。

**评测 / Evaluation**

- **任务领域**: 企业级软件工程(enterprise SWE)多平台组织工作流——跨 Slack(沟通)、Linear(工单/项目跟踪)、Git/Gitea(代码与文件系统)的异步协作场景，融合多跳推理、跨知识库检索、代码库/文件系统理解与探索。明确区别于既有以"单线程对话"为主的记忆评测(LoCoMo、LongMemEval)，强调真实组织环境中的实时上下文切换。论文展望可扩展到营销、销售等含大量内外部沟通重叠的领域。不涉及网页导航、具身、游戏等智能体任务。
- **基准**: 本工作即提出 MEMTRACK 这一新基准：47 个精心策划的长上下文实例，模拟真实企业 SWE 工作流；每实例含一条跨平台交错事件时间线 T 与若干顺序提问的问答对。七项数据集质量度量(均值/最大)：每实例事件数 39.9/115、事件 token 数 4.01K/11.1K、平台熵 0.668/0.989、跨平台引用数 2.1/19(LLM 裁判软标签阈值 0.3)、时序异配 Chronological Heterophily 0.364/0.714、时间线跨度(小时) 878/3049、每实例问题数 3.2/5。评测指标：Correctness(gpt-4o LLM-as-judge，按实例对问题取平均)、Efficiency(工具调用经济性，指数衰减公式)、Redundancy(冗余工具调用占比，LLM 裁判)，外加工具调用量 TC(mean/max) 与工具调用熵 H(TC)。对比/参照的既有基准/数据集理念包括 SWE-Bench(自底向上选仓库)、LoCoMo、LongMemEval、MemoryAgentBench(基于 ∞Bench，被指出非内聚且受 MCQ 输出偏差影响)。为降低 MCQ 输出偏差与不确定性，MEMTRACK 用简短短语答案 + 直接/近似匹配的 LLM 裁判评测。
- **报告增益**: 本工作为基准，"头条结果"是揭示当前 SoTA 模型与记忆后端的不足而非提升某指标。主结果(Table 3，5 次运行均值)：gpt-5+NoMem Correctness 0.601 / Eff 0.667 / Red 0.206 / TC 13.22(mean)·45.4(max) / H(TC) 0.978；gpt-5+Mem0 0.610 / 0.656 / 0.214；gpt-5+Zep 0.601 / 0.660 / 0.214；gemini-2.5-pro+NoMem 仅 0.144 / 0.638 / 0.237；gemini-2.5-pro+Mem0 0.118（引入 Mem0 后反而略降）；gemini-2.5-pro+Zep 0.140。即最佳的 gpt-5 仅约 60% 正确率，且 gpt-5 系约为 gemini 系的约 4 倍。关键结论：①外接记忆 Mem0/Zep 对 gpt-5、gemini-2.5-pro 均无显著提升，gemini+Mem0 反而略降——因 LLM 不擅长有效调用记忆工具且引入记忆后冗余上升；②追问性能下降(Table 4)：gpt-5 整体 0.601→追问 0.571、Mem0 0.588→0.553、Zep 0.604→0.585；gemini 0.144→0.121、Mem0 0.118→0.094、Zep 0.140→0.113，趋势在所有方法一致；③工具调用成功率高(均值约 0.91–0.95)说明调用本身流畅，瓶颈在记忆保持与跨平台综合；④≥20% 的工具调用冗余佐证跨平台信息保持欠佳。标准差见 Table 6(gpt-5+NoMem σ=0.0594 等)。温度统一设为 1。
- **对比基线**: 主要对照三种被测方法(同一智能体架构下变更记忆组件)：LLM+NoMem(无内置记忆，纯长上下文/工具访问)、LLM+Mem0(向量库式记忆，ChromaDB + gpt-4o-mini 嵌入)、LLM+Zep(知识图谱式时序记忆)。被测主干 LLM 为 gpt-5 与 gemini-2.5-pro(附录工具调用成功率表另含 claude 系结果)。因此核心比较是"无记忆 vs 两类主流外接记忆后端"以及"两大前沿 LLM 家族"的横向对比。基准设计上还在动机层面对照了 LoCoMo、LongMemEval(单线程对话)、MemoryAgentBench(基于 ∞Bench、MCQ 偏差)、MemGPT/archival memory(依赖特定实现)等既有评测的局限。

**分析 / Analysis**

- **关键创新**: 首个面向"动态企业多平台环境"的长期记忆与状态追踪评测基准——把记忆评测从既有的单线程对话(LoCoMo/LongMemEval)推进到跨 Slack+Linear+Git 三平台、含异步事件、噪声、反事实、跨引用、代码库探索的容器化生态有效(ecologically valid)组织工作流，显式考核记忆的获取/选择/冲突消解三能力且对记忆后端中立。配套两项创新：①可扩展的"自底向上"数据合成法——类 SWE-Bench 地从开源仓库已关闭 PR/issue 反推、用智能体把真实解决过程改写为带干扰项的多平台时间线，配以自顶向下专家法与混合法，三法并用兼顾真实性与规模；②超越简单 QA 的三类记忆度量(Correctness、Efficiency 指数衰减公式、LLM 裁判式 Redundancy)外加工具调用量与跨平台熵，以及七项数据集质量度量(平台熵、跨平台引用、时序异配等)，并用简短短语答案+近似匹配规避 MCQ 输出偏差。核心发现：即便最强 gpt-5 也仅约 60% 正确率，外接记忆几无增益且增加冗余，模型宁可反复读源也不善用记忆工具。
- **局限**: (1) 规模小：仅 47 个实例、每实例平均 3.2 题(最多 5)，统计功效有限；(2) 域窄：仅企业 SWE 工作流，未含营销/销售等其它企业域(论文列为未来工作)；(3) 被动设定：智能体只读取时间线回答问题，尚不能主动行动(创建工单/发消息/参与时间线演化)，论文刻意回避更难的主动设定；(4) 重度依赖 LLM：数据合成用 Claude-4-Sonnet、事件/干扰项由 LLM 生成可能引入偏差，正确性/冗余/跨引用均由 LLM 裁判(gpt-4o)评判，存在裁判偏差与成本；(5) 仅评测两类外接记忆(Mem0/Zep)且用其默认设置，未做记忆超参/配置调优，可能低估记忆组件潜力；(6) 温度=1 引入非确定性(虽 5 次平均并报告 σ)；(7) 代码/环境完整开源状态不明，仅数据集经 Google Drive 公开，复现门槛(Docker 多服务器、端口/网络要求)较高。
- **与其他工作关系**: 属本研究 F 类"记忆评测基准"中聚焦"智能体/企业多平台、状态追踪与冲突消解"的代表作。它明确定位为单线程对话记忆基准 LongMemEval(F1)、LoCoMo(F2) 的升级与补充——把评测从人-AI/人-人对话推进到跨平台异步组织工作流，并批评 MemoryAgentBench(基于 ∞Bench、受 MCQ 偏差)的非内聚性。在记忆能力分类上引用 Shan et al.(认知记忆综述)与 Du et al.(记忆分类/操作综述)的获取/利用/维护三分法。被测/引用的记忆系统包括 Mem0(对应本研究 D 类)、Zep/Graphiti(D 类，知识图谱时序记忆)、MemGPT(B3，文件系统 archival memory)、A-MEM(B4，链接生成与记忆演化)、MemoryBank(B2)、MemoryOS(B7)、MIRIX(B9，多智能体记忆)、Reflexion(A 类，自反思获取记忆)、MemInsight 等；数据合成借鉴 SWE-Bench。其"冲突/矛盾处理"维度与 Memory-R1 的 UPDATE、A-MEM 的演化等"学习型/算子式记忆维护"互补——MEMTRACK 提供考核场地，后者提供机制。与智能体中心经验回放/技能诱导(A 类 ReasoningBank/ExpeL、C 类 Voyager/AWM)正交：MEMTRACK 不教智能体从经验自我改进，而是诊断其跨平台记忆保持与状态追踪。论文展望多智能体、多平台记忆基准方向，可与 G-Memory/MIRIX 衔接。
- **可复现性**: 中等偏弱、社区采用尚早(约 9 引用)。数据集：47 个 MEMTRACK 实例经 Google Drive 公开下载，论文附录详列三套数据生成提示词(自底向上完整协议、自顶向下系统提示、混合法)、评测提示词(Correctness/Redundancy 裁判)、统计度量提示词(跨引用打分)、平台工具签名(Slack/Linear/Git/记忆组件接口)与系统资源要求(每并发实例 2GB、端口 3000-3999、10+ Docker 网络)，复现路径较清晰。但未发现独立公开的环境/评测代码仓库(GitHub patronus-ai 下未见 memtrack repo)，Docker 化多平台服务器(Gitea/Slack/Linear 模拟)的搭建依赖论文描述，工程复现门槛较高。被测记忆组件 Mem0/Zep 用其默认 API 设置，便于对齐。整体属"数据+协议公开、完整环境代码状态不明"的可复现性。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（被测对象为启发式/即插即用管线，基准本身亦不训练记忆控制策略）。MEMTRACK 评测的 LLM+Mem0/Zep 均冻结参数、靠提示与默认 API 启发式地决定何时调用 store_memory/search_memory，不用 RL/训练学习"何时/写什么/如何检索/如何更新"的记忆管理策略。它恰恰为 2025-26 代际的"学习型记忆控制"工作(Memory-R1、Mem-α 等)提供了一个困难的考核场地——其核心负面发现(模型不会有效用记忆工具、引入记忆反增冗余)正凸显了需要学习型记忆控制策略。属学习型记忆控制代际"之前/之外"的诊断性基准。
- **记忆主体**: 偏"组织/任务中心"，介于用户中心与智能体中心之间但都不完全等同。MEMTRACK 记忆的对象是组织事件时间线(跨 Slack/Linear/Git 的工单、对话、代码、决策与状态)，目标是智能体能跨平台准确追踪企业 SWE 工作流的状态并回答信息需求——既非 Mem0/Zep 式"记住单一用户偏好做个性化"，也非 Voyager/ReasoningBank 式"记住智能体自身任务经验做自我改进"。它更接近"记住一个组织/项目的共享上下文与状态"的企业知识记忆，问答模拟同事询问上下文/会议准备/新人 onboarding 等真实工作场景。
- **多智能体记忆**: 当前为单智能体设定：单个 LLM 驱动的智能体跨多平台访问时间线、保持记忆并回答问题，记忆组件(Mem0/Zep)亦服务于该单智能体，不涉及 G-Memory/MIRIX 式多智能体共享/路由记忆(无跨智能体洞见/查询/交互分层)。但论文明确把"多智能体、多平台记忆基准"列为其铺垫的未来方向，并展望让智能体主动参与组织时间线演化，可衔接多智能体记忆研究。
- **时序推理支持**: 显式且重点支持。每个事件带时间戳 τ、平台类型等元信息，时间线按时间顺序跨平台交错；基准专设"时间线跨度(小时)"(均值 878、最大 3049)与"时序异配 Chronological Heterophily"(均值 0.364，衡量相邻事件跨平台交替的概率，防止各平台事件被整齐分段而平凡化)两项度量。问题设计含大量时间型模糊引用(如"今天/上月/这个下午/几周前那个 issue")，需解析并与事件时间线关联(示例题:据模糊日期引用确定 Sarah 首次报告问题的日期=20250515)。状态随时间被修订/推翻的决策亦考核时序状态追踪。时间推理是其暴露模型短板的关键难点之一。
- **模态**: 以文本为主，含代码/文件系统模态。事件(Slack 消息、Linear 工单)为文本；Git 部分要求智能体 git clone 真实开源仓库并在 Docker 文件系统中浏览、grep、读代码回答(如统计 import、最长函数所在文件名、def 数量、某数字首次出现行号),涉及代码库/文件系统理解。无图像/视频/具身视觉记忆。可大致归为"文本+代码(text+code/filesystem)"，非多模态视觉。
- **冲突/矛盾处理**: 核心考核维度之一(冲突消解 conflict resolution 是 MEMTRACK 明列的三大记忆能力"获取/选择/冲突消解"之一)。基准在跨平台时间线中刻意注入"噪声、矛盾、跨引用"信息与会被后续事件修订/推翻的决策、状态变化与情境化反事实(counterfactual)干扰，考核智能体能否跨平台关联并消解矛盾、用最新/正确信息作答(如多个相似命名工单、被否决的方案、随时间改变的优先级)。这是 MEMTRACK 区别于纯 needle-in-haystack 检索基准的关键。实验结论:当前模型在跨平台依赖与矛盾消解上表现欠佳，是 gpt-5 仅约 60% 正确率的重要原因之一。与显式记忆更新算子(Memory-R1 UPDATE)不同，MEMTRACK 以"最终答案是否正确"为外部度量来评判冲突处理能力。
- **token成本/延迟证据**: 本工作为评测基准而非记忆层产品，未报告"省 token/降延迟"百分比，反而用度量量化了"效率"与"冗余"成本：Efficiency=exp(-(|TC|-TC_min)/TC_min) 惩罚超出必要的工具调用；Redundancy 度量冗余工具调用占比(各方法约 0.206–0.240，即≥20% 工具调用冗余)；并报告工具调用量 TC(gpt-5 系 mean 约 13.2、max 45–51；gemini 系 mean 约 12.5–13.7)。关键效率发现:挂接 Mem0/Zep 后冗余不降反升(gpt-5+NoMem 0.206→+Mem0 0.214→+Zep 0.214)、效率略降(0.667→0.656/0.660)，且模型偏好反复读源(隔≥3 回合重复访问同一工单/消息/文件)，说明外接记忆未带来 token/调用效率收益。数据规模本身(事件 token 均值 4.01K/最大 11.1K、时间跨度最长 3049 小时)构成长时程记忆的成本压力证据。无系统级延迟数字。

**不确定字段 / Uncertain**

- 代码链接 (`code_url`)
- 过度个性化/记忆安全风险 (`over_personalization_risk`)


<a id="f6-locomo-plus全称-locomo-plus-beyond-factual-cognitive-memory-evaluation-framework-for-llm-agents超越事实的认知记忆评测基准与框架在-locomo-原有五类问题单跳多跳时序常识对抗之上新增第六类认知记忆-cognitive任务并配套提出基于约束一致性-constraint-consistency的统一评测范式亦写作-locomo-plus"></a>

### F6 LoCoMo-Plus

*LoCoMo-Plus（全称 Locomo-Plus: Beyond-Factual Cognitive Memory Evaluation Framework for LLM Agents；超越事实的认知记忆评测基准与框架。在 LoCoMo 原有五类问题（单跳/多跳/时序/常识/对抗）之上新增第六类「认知记忆 Cognitive」任务，并配套提出基于「约束一致性 constraint consistency」的统一评测范式。亦写作 Locomo-Plus）*


**基本信息 / Provenance**

- **年份**: 2026（arXiv 预印本 2602.10715 于 2026-02-11 首次公开 v1，cs.CL；GitHub 仓库最早 2026-01-06 创建）
- **作者/机构**: 第一作者 Yifei Li（李一非，西安交通大学 Xi'an Jiaotong University，邮箱 yifeilee@stu.xjtu.edu.cn）；合作者 Lingling Zhang、Rongman Xu、Muye Huang、Jun Liu（均西安交通大学）以及 Weidong Guo、Hui Liu、Lijiao Xu、Yu Xu（均腾讯 Tencent）。属学术界（西安交通大学）与工业界（腾讯）合作工作。
- **论文链接**: https://arxiv.org/abs/2602.10715 （HTML 全文 https://arxiv.org/html/2602.10715v1 ；OpenReview https://openreview.net/forum?id=QWVKrMGdah ；DOI 10.48550/arXiv.2602.10715，Semantic Scholar Corpus ID 285470135）
- **代码链接**: https://github.com/xjtuleeyf/Locomo-Plus （官方仓库，约 17 stars、1 fork、3 open issues，主语言 Python(94.7%)+Shell(5.3%)，单一贡献者 xjtuleeyf；含数据构建管线 data/、生成管线 generation_pipeline/、评测框架 evaluation_framework/，发布 locomo_plus.json 样本与 locomo10.json 原 LoCoMo 对话；评审版另有匿名仓库 anonymous.4open.science/r/Locomo-Plus）

**记忆分类 / Taxonomy**

- **记忆类型**: 本身是「评测基准+评测框架」而非记忆系统。其考核对象分两层：Level-1 事实记忆（factual，含 object 对象级局部事实与 episodic 情景/事件信息）与 Level-2 认知记忆（cognitive，本文核心新增）。认知记忆指保留并应用从历史交互中推断出的隐式约束（用户状态 state、目标 goal、价值/偏好 value、因果情境 causal），属语义/情景记忆之上的「行为约束型」高阶记忆，对应 CoALA 语义记忆但强调隐式约束应用而非显式回忆；不涉及程序性技能记忆。被测对象的工作记忆即 LLM 上下文窗口。
- **记忆结构**: 作为基准其数据结构为带时间戳的多会话对话（在 LoCoMo 原 conversation 轨迹中嵌入构造好的 cue-trigger 对）。新增认知记忆样本由「cue 隐式线索对话 + trigger 触发查询 + 时间间隔指示 t」三元组构成，嵌入长对话后形成需跨多轮、跨时间保留并应用隐式约束的实例。认知记忆被显式分解为四类潜在约束：causal（因果，X 发生→Y 改变）、state（状态，当前处境为 Z）、goal（目标，正朝 X 努力）、value（价值/信念，相信或偏好 X）。基准不规定被测系统的内部记忆结构（可为原始缓冲/向量库/图等）。
- **存储后端**: 评测层面与具体后端无关，可评测任意记忆方案。被测方法覆盖四类：(1) 纯上下文开源 LLM（Qwen2.5-3B/7B/14B-Instruct、Qwen3-4B/8B/14B，全历史入上下文）；(2) 纯上下文闭源 LLM（gpt-5-nano、gpt-4.1、gpt-4o、gemini-2.5-flash、gemini-2.5-pro）；(3) RAG 基线（GPT-4o 生成 + OpenAI 嵌入 text-ada-embedding-002 / text-embedding-small / text-embedding-large，取 Top-5 段落）；(4) 记忆系统（A-Mem、Mem0、SeCom，均以 GPT-4o 为骨干）。数据集以 JSON 发布于 GitHub。构建/裁判侧用 OpenAI 接口 LLM、BM25、MPNet 与 BGE 句向量模型。
- **持久化**: 针对「外部持久化记忆（durable external store）」的长上下文在线评测：把含 cue 的多会话历史 H={u1,a1,…,ut,at} 逐步呈现，被测系统须跨越大量中间无关轮次与干扰内容保留 cue 信息，待后续 trigger 查询 q_{t+1} 到来时应用之。基准设定（长对话、cue 与 trigger 之间存在两周到一年以上的时间间隔、超长上下文干扰）正是为暴露纯上下文（in-context）方案在长程隐式约束保持上的不足而设计——即使 Gemini-2.5-Pro 这类百万级上下文窗口能装下全部历史，认知记忆仍随上下文增长而迅速崩溃。

**核心机制 / Mechanisms**

- **写入/编码**: 作为基准本身不规定写入编码，而是通过统一输入协议（unified input）把构造实例嵌入长对话供被测系统按其各自机制写入。其核心方法学贡献在数据「构造侧」而非记忆写入算子：通过六阶段管线从零生成认知记忆实例——(1) 隐式 cue 对话生成：提示 LLM 生成以自然对话形式（而非显式事实陈述）隐含传达某参与者状态/目标/偏好/价值的短对话片段，产出候选 cue 池 c0；(2) 价值性人工验证：人工筛除不具持久/行为约束性、可由局部上下文平凡推断的 cue，得 c1；(3) cue-trigger 查询构造：提示 LLM 生成其正确作答依赖 cue、但表层语义相似度低、单看欠定（无 cue 时多种回答都看似合理）的下游 trigger 查询 q，并赋予时间间隔指示 t；(4) 语义过滤：用 BM25 + MPNet 相似度打分剔除 cue 被复述/改写/可直接从查询恢复的捷径对；(5) cue 记忆触发性人工验证：人工确认每对确实需回忆并应用 cue 的隐式约束、而非靠表层相似命中；(6) 嵌入 LoCoMo 长对话：按 t 指定的间隔把 cue 与 trigger 插入选定长对话轨迹。被测的示例记忆系统（A-Mem/Mem0/SeCom）则各自做抽取/摘要/分段压缩式写入。
- **检索机制**: 基准的关键论点之一是认知记忆下「检索本身失效」：因 cue（如『准备重要考试，想减少干扰』）与 trigger（如『要不要追那部新剧』）之间被刻意构造为「cue–trigger 语义脱节（semantic disconnect）」——表层词汇与语义相似度都低——基于相似度的 RAG（OpenAI 三款嵌入 Top-5）与专用记忆系统（A-Mem/Mem0/SeCom 的检索）都因无法用 trigger 召回相关 cue 而大幅失败。数据构造阶段反向使用检索器（BM25 + MPNet）来「过滤掉」表层相似的简单对，确保留存样本无法靠相似检索捷径解决。RAG 基线检索 Top-5 段落拼接入提示；记忆系统按各自机制（A-Mem 自适应构建+检索、Mem0 生产级长期存储检索、SeCom 段级记忆单元+压缩去噪）检索。基准用「约束一致性」而非召回率作为最终度量，但其设计本质上揭示了语义脱节对所有检索式方法的系统性挑战。
- **反思/巩固**: 本基准不引入也不要求反思/整合机制（不产生 Reflexion 式经验洞见），属评测基准而非记忆系统。其「整合」体现在数据构造侧的离线生成-过滤-验证流程，把原始 cue 提炼为带隐式约束的可评测实例。被测的记忆系统（A-Mem 自适应记忆构建、SeCom 压缩去噪、Mem0 长期抽象）各自含整合/压缩步骤，但论文发现这些整合对认知记忆并无帮助、甚至因丢失隐式约束信息而无济于事。基准强调：对认知记忆而言，单纯把对话摘要/抽取为事实会丢失行为约束信号，整合并不能弥补语义脱节带来的检索失败。
- **遗忘/更新**: 不实现遗忘/衰减/编辑算子（非记忆系统）。基准聚焦于「长程保持并应用隐式约束」的能力评测，不涉及 Ebbinghaus 衰减、ADD/UPDATE/DELETE 操作或冲突消解算法。其长上下文敏感性分析（图 7）显示认知记忆随对话长度增加而急剧崩溃，间接反映长程「无意遗忘/被干扰覆盖」是当前系统的核心失败模式，但本文不提供遗忘机制本身。
- **经验回放 (核心主题)**: 不适用（本工作是用户中心的对话认知记忆评测基准，而非智能体经验回放方法）。LoCoMo-Plus 复用的是「用户在历史 cue 对话中隐式透露的状态/目标/价值约束」，而非智能体自身过往任务轨迹/技能。它不维护成败轨迹缓冲、不做范例提示或技能复用，与 ReasoningBank/Voyager/ExpeL 等「重用自身经验改进未来决策」的智能体中心范式正交。其「复用」体现为：要求被测系统在后续 trigger 查询时正确回忆并应用先前 cue 所诱导的隐式行为约束（如察觉新请求与早前目标冲突并据此作答），以保持跨长对话的行为一致性与个性化连贯。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric / in-context, prompt-level）评测：被测的示例方案（开源/闭源 LLM 全上下文、RAG、A-Mem/Mem0/SeCom）全程冻结 LLM 参数、无梯度更新，记忆与作答完全发生在上下文/外部存储层面。基准本身为纯推理评测（inference-only），不训练任何模型。论文明确指出数据集规模有限、刻意优先诊断价值而非规模，因此不适合用于训练或微调 LLM。
- **失败学习 (核心主题)**: 不适用 / 非失败学习型方法。LoCoMo-Plus 不检测任务执行失败、不构建负例或错误规则、不对失败轨迹做自反思（与 Reflexion/Retroformer/ExpeL 正交）。它评测的是「隐式约束保持-应用」的正确性而非「从失败中改进策略」。不过基准的诊断价值恰在于「暴露失败」：它揭示了现有 LLM 与记忆系统在认知记忆上普遍且严重的失败模式（从事实记忆到认知记忆出现 20–40% 的性能骤降，且性能差异在 LoCoMo-Plus 下被压缩、收敛到一致的低水平），为后续方法改进提供失败诊断信号；但「从失败中学习」属被测方法的职责而非本基准的机制。
- **技能/程序归纳**: 不归纳可复用技能/工作流/程序。LoCoMo-Plus 聚焦用户隐式约束（状态/目标/价值/因果）的认知记忆评测，不从经验中提炼程序性技能或可复用工作流（与 Voyager/AWM/Synapse 等技能诱导方法属不同范畴）。
- **在线 vs 离线**: 评测为在线式（online）：把含 cue 的多会话历史按时序呈现给被测系统，要求其跨中间干扰轮次在线保持 cue 隐式约束，待后续 trigger 查询到来时应用。基准数据的构建则是离线的（LLM 生成 cue/trigger + 人工两轮验证 + BM25/MPNet 相似度过滤 + 按时间间隔嵌入 LoCoMo 长对话）。因生成与人工验证成本高，认知记忆样本规模刻意保持精简（约 401 条测试实例，覆盖四类约束、时间间隔从两周到一年以上）。

**评测 / Evaluation**

- **任务领域**: 长期多会话对话记忆（multi-session dialogue）/个性化对话场景下的「认知记忆」评测。任务形式为长上下文对话续写/作答：被测系统须在多会话历史中保留并应用隐式约束。不评测网页导航、具身、游戏、编码、GUI 等智能体任务，亦不评测企业场景；纯文本对话域。
- **基准**: 本工作即提出 LoCoMo-Plus 这一新基准。它在 LoCoMo（Maharana 等 2024）十段多会话对话之上：(a) 沿用 LoCoMo 原五类问题（single-hop 单跳、multi-hop 多跳、temporal 时序、commonsense 常识、adversarial 对抗）；(b) 新增第六类 Cognitive 认知记忆任务（约 401 条，四类约束 causal/state/goal/value，时间间隔两周至一年以上）；(c) 用统一格式整合六类。对比/复用的既有基准为 LoCoMo（事实记忆 Level-1）。裁判默认采用 LLM-as-a-judge（论文用 gemini-2.5-flash；仓库默认 gpt-4o-mini），三档计分（正确=1/部分=0.5/错误=0）。
- **报告增益**: 本工作为「揭示挑战」型基准，核心是「性能骤降」而非「方法增益」。主结果（表 1，认知记忆 LoCoMo-Plus 列）显示所有方法从事实记忆到认知记忆出现巨大 Gap：开源 LLM——Qwen2.5-3B/7B/14B 认知得分仅 10.82/9.57/19.24（Gap 31.38/35.74/44.21），Qwen3-4B/8B/14B 为 15.70/17.68/19.09；闭源 LLM——gpt-5-nano 14.84、gpt-4.1 18.63、gpt-4o 21.05、gemini-2.5-flash 24.67、gemini-2.5-pro 26.06（即便最强模型认知得分也仅约 26，Gap 高达约 45.72）；RAG（GPT-4o）——三款嵌入认知得分仅 13.91/12.29/15.55（Gap 23.47/24.94/29.77，因语义脱节检索失败）；记忆系统（GPT-4o）——Mem0 15.80、SeCom 14.90、A-Mem 17.20（Gap 41.44/42.63/42.44，专用记忆系统同样大幅失败）。总体观察：(1) 全体方法从 LoCoMo 到 LoCoMo-Plus 出现约 20–40% 的性能骤降；(2) 各方法在 LoCoMo 上的差异在 LoCoMo-Plus 下被显著压缩、收敛到统一低水平；(3) 长上下文敏感性分析（图 7）：object 记忆随长度稳健、episodic 稳步退化、cognitive 随上下文增长「迅速崩溃」。评测偏差分析另证明：任务披露使 temporal/adversarial 任务得分被不当抬高；BLEU/ROUGE/EM/F1 等表层指标随生成长度系统性偏移（在真值平均长度约 5.18 tokens 附近达峰）。
- **对比基线**: (1) 纯上下文开源 LLM：Qwen2.5-3B/7B/14B-Instruct、Qwen3-4B/8B/14B（全历史入上下文，无外部检索/记忆）；(2) 纯上下文闭源 LLM：gpt-5-nano、gpt-4.1、gpt-4o、gemini-2.5-flash、gemini-2.5-pro（强参照基线）；(3) RAG 检索增强（GPT-4o 生成）：OpenAI text-ada-embedding-002 / text-embedding-small / text-embedding-large，取 Top-5 段落；(4) 专用记忆系统（GPT-4o 骨干）：A-Mem、Mem0、SeCom。对照维度还包括：事实记忆 LoCoMo vs 认知记忆 LoCoMo-Plus、任务披露 vs 统一输入、表层匹配指标 vs 约束一致性裁判、不同裁判骨干（Gemini-2.5-Flash vs GPT-4o）。

**分析 / Analysis**

- **关键创新**: 首次区分并系统评测 LLM 智能体的「Level-2 认知记忆」——即跨长对话保留并应用隐式约束（状态/目标/价值/因果）的能力，相对既有基准仅考核「Level-1 事实记忆（显式回忆）」是重要概念跃迁。两项核心创新：(1) 提出「cue–trigger 语义脱节」构造范式：让触发查询与记忆线索在表层词汇/语义上刻意不重叠（如把『刚领养了救助犬』的 cue 连到『该买什么宠物粮』的 trigger），迫使系统应用隐式约束而非靠相似检索走捷径，从根本上击穿 RAG/记忆系统的检索假设；(2) 提出「约束一致性（constraint consistency）」统一评测框架：以「统一输入、差异化裁判」取代「任务披露式提示 + 表层字符串匹配」，把正确性定义为响应落入由 cue 诱导的有效行为空间 A_c={a | a 与约束 c 一致}（容许多种合法实现），而非匹配单一参考答案。论文还实证揭示了既有评测的两类系统性偏差（任务披露偏差、生成长度偏差），论证表层匹配指标即使对事实记忆也具误导性。
- **局限**: (1) 规模有限：认知记忆样本因生成+人工验证成本高而刻意精简（约 401 条），优先诊断价值而非规模，不适合训练/微调；(2) 依赖专有 LLM 裁判（Gemini-2.5-Flash/GPT-4o，仓库默认 gpt-4o-mini）评判约束一致性，结果可能对裁判模型与提示设计敏感（虽报告了人工-裁判一致性 0.80–0.82 与跨裁判稳定性）；(3) 仅英文对话、特定骨干模型/检索管线/记忆系统集合，泛化性待验证；(4) 仅建模隐式约束影响行为的对话设定，不覆盖长期信念修正、情绪动态、多智能体记忆交互；(5) 聚焦长程约束的「单轮解析」，不含多轮约束协商；(6) 第三方审计（dial481/locomo-audit、AI-Navigate）指出：它原封继承 LoCoMo 全部 1,540 道原始问题（含约 6.4%/99 道有缺陷的标准答案），改进后的裁判方法学（任务专属提示、三档计分、0.80+ 一致率）仅在新增认知问题上验证、原五类未重新校验；且仍缺乏标准化端到端管线（各系统自带摄取/提示/模型）。
- **与其他工作关系**: 属本研究 F 类「记忆评测基准」（编号 F6），是 F2 LoCoMo（Maharana 等 2024，ACL）的直接「下一代/扩展」：在 LoCoMo 十段对话与五类问题之上新增第六类认知记忆并重构评测协议，故强依赖并复用 LoCoMo 数据（locomo10.json）。与同类基准的关系：相对 F1 LongMemEval（用户中心事实记忆、含知识更新维度）与 F2 LoCoMo（事实回忆），LoCoMo-Plus 把评测轴从「显式事实回忆」上移到「隐式约束应用（认知记忆）」，是对二者「记忆=事实检索」假设的批判性补全；与 F8 Causal-LoCoMo（聚焦因果，亦衍生自 LoCoMo）在「超越表层事实」方向相邻，但 LoCoMo-Plus 覆盖 causal/state/goal/value 四类约束更广；与 F5 MEMTRACK、F7 OP-Bench 在「记忆评测前沿维度」上互补。它把 A-Mem（B4）、Mem0（D4）、SeCom 作为被测记忆系统基线，证明这些用户中心记忆系统在语义脱节认知任务上同样失败；与智能体中心经验回放/技能诱导（A 类 Reflexion/ExpeL/ReasoningBank、C 类 Voyager/AWM）正交（被测对象与记忆主体不同）。已被第三方图原生记忆产品（Kumiho 宣称在其上达 93.3%）与 LoCoMo 审计工作引用复现。
- **可复现性**: 复现性中等偏上、采用度尚早期。代码与数据管线开源于 github.com/xjtuleeyf/Locomo-Plus（约 17 stars、1 fork、3 open issues，Python，单一贡献者），含三大模块：data/（build_conv.py、unified_input.py 与样本 locomo_plus.json、原 LoCoMo 对话 locomo10.json）、generation_pipeline/（cue 生成、trigger 生成、相似度排序；其中第 2、5 步为人工）、evaluation_framework/（运行模型预测脚本 evaluate.sh 与 LLM-as-judge 脚本 judge.sh）。所有 API key/路径经环境变量与本地配置（env.local.sh）管理，仓库不含密钥；裁判默认 gpt-4o-mini。论文附录提供裁判提示、标注指南与一致性统计。复现注意：数据生成第 2、5 步需人工介入；大体量生成 JSON 被 gitignore 需本地运行产出；继承自 LoCoMo 的原始问题含已知标注缺陷（见第三方审计），需对齐数据版本。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式/无学习记忆控制）。LoCoMo-Plus 是评测基准，不训练记忆管理策略；被测方法亦均为启发式管线（全上下文、相似度 RAG、A-Mem/Mem0/SeCom 的启发式抽取-检索），不用 RL/训练去学习「何时/写什么/如何检索/如何更新」的记忆控制策略本身。它属 2025–26「学习型记忆控制」（Memory-R1、Mem-α 等）代际之前/之外，但为该方向提供了一个专门考核「隐式约束保持」的高难度评测场。
- **记忆主体**: 用户中心（user-centric）：核心是记住用户在历史 cue 对话中隐式透露的状态/目标/价值/因果约束，并在后续作答时应用之以保持个性化行为一致性。与 Mem0/Zep/LongMemEval 同属「记住用户信息做个性化」阵营，区别于 ReasoningBank/Voyager 等「记住智能体自身经验做自我改进」的智能体中心记忆。其独特之处在于：考核的不是「记住用户事实并回答」，而是「记住用户隐式约束并据此约束行为」（如察觉新请求与早前目标冲突）。
- **多智能体记忆**: 单智能体/单助手设定。LoCoMo-Plus 评测单个对话系统在与用户长期一对一多会话交互中的认知记忆能力，不涉及 G-Memory/MIRIX 式多智能体共享/路由记忆架构（无跨智能体洞见/查询/交互分层）。论文「局限」明确将多智能体记忆交互列为不覆盖范围。
- **模态**: 纯文本（text-only）。所有 cue 对话、trigger 查询、历史与作答均为文本，无视觉/具身/多模态记忆（虽嵌入的 LoCoMo 原始对话含多模态轮，但 LoCoMo-Plus 认知任务本身为文本）。
- **过度个性化/记忆安全风险**: 未提供专门的过度个性化/记忆安全评测（无 OP-Bench、Causal-LoCoMo 类有害/陈旧/侵入记忆度量），但在伦理声明中触及相关负面维度：指出增强对话记忆会引发隐私与用户控制顾虑，并强调本工作仅在受控评测设定下评估记忆使用、不主张持久存储用户数据，且明确把「记忆应如何存储/遗忘/治理」列为本文范围之外。其认知记忆评测的潜在价值之一在于：正确应用隐式约束（如尊重用户目标）本身关乎避免不当/冲突性回应，但本基准不直接度量谄媚/侵入/有害记忆。属本条专门评测证据有限。
- **冲突/矛盾处理**: 未将冲突/矛盾事实消解作为独立评测维度（区别于 MEMTRACK、Memory-R1 UPDATE 的显式冲突处理）。但其核心场景隐含一种「约束冲突」考核：当后续 trigger 请求与早前 cue 所诱导的隐式约束相冲突时（如用户目标是『备考减少干扰』而 trigger 问『要不要追新剧』），系统应识别冲突并据约束作答——这是「行为约束一致性」而非「事实矛盾合并」。论文不实现冲突消解算法，而以「响应是否落入约束有效空间 A_c」为外部度量。
- **token成本/延迟证据**: 本工作为评测基准，非记忆层产品，不以「省 token/降延迟」为卖点，未报告系统级 token 成本或延迟节省百分比。其相关「成本/长度」证据为：(a) 长上下文敏感性分析（图 7）表明认知记忆随对话长度（token 数）增长而迅速崩溃，揭示纯长上下文方案在长程隐式约束保持上的低效——即便 Gemini-2.5-Pro 百万级上下文能装下全部历史，认知得分仍仅约 26（45.7% 量级，据第三方复现）；(b) 生成长度偏差分析显示 EM/F1/BLEU/ROUGE 随生成 token 数系统性偏移（真值平均长度约 5.18 tokens，过短或过长均被惩罚），论证表层指标的长度敏感性。认知记忆样本规模有限亦源于生成+人工验证的高成本权衡。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)
- 时序推理支持 (`temporal_reasoning_support`)
- 发表venue (`venue`)


<a id="f7-op-bench过度个性化基准配套提出-self-recheck-记忆过滤方法"></a>

### F7 OP-Bench

*OP-Bench（过度个性化基准；配套提出 Self-ReCheck 记忆过滤方法）*


**基本信息 / Provenance**

- **年份**: 2026（arXiv 预印本，2026-01-20）
- **作者/机构**: Yulin Hu（胡玉林）、Zimo Long、Jiahe Guo、Xingyu Sui、Xing Fu、Weixiang Zhao、Yanyan Zhao（赵妍妍）、Bing Qin（秦兵）；哈尔滨工业大学（Harbin Institute of Technology，SCIR/社会计算与信息检索研究中心）。通讯邮箱 {ylhu, yyzhao}@ir.hit.edu.cn。
- **发表venue**: arXiv 预印本（cs.CL，arXiv:2601.13722v1，2026-01-20）；同时投稿于 OpenReview（疑似 ACL 2026，匿名提交，forum id=epSwZTwUmP），截至调研时尚未确认正式录用。
- **论文链接**: https://arxiv.org/abs/2601.13722 （HTML: https://arxiv.org/html/2601.13722v1 ；OpenReview: https://openreview.net/forum?id=epSwZTwUmP ）
- **引用数**: 约 0（2026-01 极新预印本，Semantic Scholar/OpenAlex 暂未索引引用数）。

**记忆分类 / Taxonomy**

- **记忆类型**: 本身不实现记忆机制，而是面向‘记忆增强个性化对话智能体’的评测基准。被评测对象覆盖语义记忆（用户画像/偏好）与情景记忆（多会话长程对话历史）。所提 Self-ReCheck 方法作用于检索到的外部用户记忆（语义/情景型），属工作记忆层的过滤。
- **记忆结构**: 作为基准本身无记忆结构。被评测的 6 种记忆方案涵盖多种结构：原始检索（RAG，向量检索）、分层记忆（LDAgent 层次化记忆）、稠密自然语言记忆库（Mem0）、智能体中心文件式抽象（MemU）、记忆操作系统式结构化索引（MEMOS/MemOS）。
- **存储后端**: 基准不规定后端。被评测方法分别使用：向量检索（RAG，OpenAI text-embedding-3-small 稠密检索）、各记忆系统自带的存储（Mem0 自然语言记忆库、MemU 文件抽象、MEMOS 统一记忆操作系统、LDAgent 层次记忆）。基准数据本身基于 LoCoMo 多会话对话语料构建。
- **持久化**: 评测的是‘外部持久化’记忆（跨会话长期存储用户偏好与历史）在生成时被检索注入上下文。Self-ReCheck 在检索后、生成前对记忆做即时（in-context）过滤，不改变底层持久化存储。

**核心机制 / Mechanisms**

- **写入/编码**: OP-Bench 不做记忆写入，而是构造测试问题：采用三阶段自动化流水线——阶段1从 LoCoMo 初始化对话中用 LLM 抽取每个用户的精简画像（profile）与偏好主题（speaker_a 视为用户、speaker_b 视为助手）；阶段2针对三类过度个性化失败模式生成查询（Irrelevance：采样画像外主题生成‘完全无关’题、用人类混淆类型模板生成‘诱饵/Baiting’题；Sycophancy：基于画像构造含事实错误或夸大价值主张的诱导问、以及伪造的不存在记忆变体；Repetition：对每个用户兴趣生成多个子主题及若干开放式非引导问题，产出语义相异但话题相关的查询）；阶段3双人独立人工复核、需两位标注者达成一致方可入库，否则升级资深标注员裁决。
- **检索机制**: 基准评测被测系统的检索行为，并诊断其检索病理。所提 Self-ReCheck 是一个轻量、即插即用、模型无关的记忆过滤器：给定查询 q 与检索得到的记忆集合 M={m_1,…,m_k}，用一个轻量级 LLM 推理函数 f_θ(q,m_i)∈{0,1} 对每条记忆做二元判断（该记忆是否‘有助于’回答当前查询），保留 M′={m_i∈M | f_θ(q,m_i)=1} 子集传给对话模型，检索与生成模块保持不变。论文指出简单的相似度阈值过滤无法奏效——‘诱饵（baiting）’查询会检索到语义高度相似但实际无关的记忆，固定阈值无法区分欺骗性与真正有用的记忆，降低阈值又会误删正常查询所需记忆、损害个性化性能（LoCoMo）。
- **反思/巩固**: 基准本身不做反思/巩固。诊断分析（RQ2）通过注意力与 token 归因发现‘记忆劫持（memory hijacking）’：模型对记忆 token 的注意力显著高于对用户查询（长度归一化后记忆/查询注意力比值持续 >2×），把检索记忆当作特权信号而非条件线索，从而导致过度个性化、损害连贯性、事实准确性与回答多样性。Self-ReCheck 本质是一种‘自检式’相关性再判断（self re-check），通过过滤降低对检索记忆的依赖与记忆-查询注意力比。
- **遗忘/更新**: 基准不涉及记忆遗忘/更新；Self-ReCheck 不删除底层记忆，仅在使用时按查询相关性逐条筛除（in-context 抑制），属于‘按需抑制’而非真正遗忘。
- **经验回放 (核心主题)**: 不适用（N/A）。该工作面向以用户为中心的个性化对话评测，不涉及以自身轨迹/经验回放来自我改进的‘智能体经验复用’机制。其与‘经验回放’主题的唯一交集是：揭示了盲目复用长期用户记忆（不加选择地注入历史偏好）所带来的负面副作用（过度个性化）。

**学习维度 / Learning**

- **学习范式**: 非参数（non-parametric）。OP-Bench 是评测基准；Self-ReCheck 是无需训练的提示式/推理式过滤器（用冻结的轻量 LLM 做二元相关性判断），不涉及梯度更新。
- **失败学习 (核心主题)**: 核心贡献即‘失败诊断’而非‘从失败中学习’：将过度个性化系统化为三类失败模式——Irrelevance（无关：在查询不需要个性化时仍注入用户记忆/画像，含 Fully Irrelevant 与 Baiting 两子型）、Sycophancy（谄媚：为迎合用户而牺牲事实，含 Fact-level/Memory-level/Value-level 三子型）、Repetition（重复：对语义相异查询反复套用相同用户记忆，产出高度相似回答）。诊断出根因为‘记忆劫持’与‘即使完全无关也强行检索（追求召回率而非弃权）’。Self-ReCheck 据此用相关性自检过滤来缓解这些失败。
- **技能/程序归纳**: 不适用。不涉及从经验中归纳可复用技能/工作流。
- **在线 vs 离线**: 基准数据为离线（offline）批量构建（基于 LoCoMo 语料经三阶段流水线+人工复核生成 1700 条实例）。Self-ReCheck 过滤在在线推理时（per-query）执行。

**评测 / Evaluation**

- **任务领域**: 以用户为中心的多会话长程个性化对话（multi-session personalized dialogue）。OP-Bench 评测过度个性化（无关/谄媚/重复）；并联合 LoCoMo 评测正常的个性化记忆能力（多跳/时序/开放域/单跳问答），以区分‘有效用记忆’与‘过度/误用记忆’。
- **基准**: 本工作即提出 OP-Bench（1700 条人工核验实例、20 个用户、3 大类 6 子类；Repetition 占 51.9%/882 条、Irrelevance 占 24.6%（Fully 318、Baiting 100）、Sycophancy 占 23.5%（Fact 100、Value 100、Memory 200））；联合使用 LoCoMo（多会话长程对话）评测个性化能力。相关基准对比：PrefEval、PersonaMem/PersonaMem-v2、MemoryAgentBench、Persona-Chat、PersonalDialog（均不专门评测过度个性化）。
- **报告增益**: （1）过度个性化普遍存在：相对无记忆 BASE，各记忆方法在 OP-Bench 上相对性能下降 26.2%~61.1%（共评测 36 个配置=6 LLM×6 记忆方法）。例：GPT-4o-mini BASE 平均分 83.10，RAG 降至 55.96（↓32.7%）、LDAgent 43.09（↓48.1%）、Mem0 46.32（↓44.3%）、MemU 40.46（↓51.3%）、MEMOS 41.86（↓49.6%）；Gemini-2.5-flash BASE 70.55，MemU 33.05（↓53.2%，最严重）；Qwen3-32B BASE 72.91，LDAgent↓54.2%；Qwen3-8B BASE 73.80。规律：越复杂/精细的记忆系统下降越严重。（2）注意力诊断：记忆/查询注意力比持续 >2×（记忆劫持）。（3）Self-ReCheck（在 Qwen3-8B、跨 5 种记忆方法上）：平均把过度个性化降低 +29%（OP-Bench 上提升），同时个性化性能（LoCoMo）平均提升约 +3%（因过滤噪声减少干扰），实现帕累托改进。LoCoMo 记忆能力对比（GPT-4o-mini，F1/BLEU-1，Overall）：MEMOS 44.14/35.47（最优）、MemU 34.68/28.15、RAG 34.10/28.13、Mem0 11.62/9.24、LDAgent 10.12/7.54。
- **对比基线**: 无记忆基线 BASE（标准 LLM 无检索/记忆）；以及 5 种代表性记忆增强方法：RAG（Lewis 2020，向量检索+text-embedding-3-small）、LDAgent（Li 2025a，层次化长期个性化记忆）、Mem0（Chhikara 2025，稠密自然语言记忆库）、MemU（NevaMind-AI 2025，用户/智能体导向文件式记忆）、MEMOS/MemOS（Li 2025b，记忆操作系统）。被评测的基座 LLM：闭源 GPT-4o-mini、Gemini-2.5-flash；开源 DeepSeek-v3.2、Qwen3-235B-A22B-Instruct-2507、Qwen3-32B、Qwen3-8B。评判模型（LLM-as-Judge）选用 GPT-4o-mini。

**分析 / Analysis**

- **关键创新**: 首个系统化定义并量化‘过度个性化（over-personalization）’的基准 OP-Bench——将其形式化为 Irrelevance / Sycophancy / Repetition 三类理论有据的失败模式，揭示‘加记忆反而有害’这一被忽视的失败模式，并通过注意力/归因分析提出‘记忆劫持’机理，配套提出无需训练、模型无关的 Self-ReCheck 记忆相关性自检过滤器以实现帕累托改进。
- **局限**: (1) 合成数据：OP-Bench 依赖合成提示与 LLM 生成的用户交互来模拟长期个性化对话，尽管经人工复核，仍可能无法完全覆盖真实用户行为的多样性与不可预测性。(2) LLM-as-Judge 偏差：评测高度依赖 LLM 评判（GPT-4o-mini），虽引入人工专家评估缓解，但因成本所限人工评估规模有限。(3) Self-ReCheck 仅在 Qwen3-8B 上验证缓解效果（覆盖面有限）。(4) 数据/代码截至调研尚未公开（称录用后发布），复现性待验证。(5) 基于 LoCoMo（仅 20 用户）构建，用户规模较小。
- **与其他工作关系**: 属于 F 类‘记忆评测基准’，与同期 2025-26 一批‘记忆负面/安全维度’工作高度互补、共同构成‘更多记忆并非总是更好’的研究浪潮：与 fields.yaml 提及的 Causal-LoCoMo 同样关注 LoCoMo 衍生的负面诊断；与 STALE（陈旧记忆失效检测，arXiv:2605.06527）、BenchPreS（偏好误用 MR/AAR，arXiv:2603.16557）、RPEval/RP-REASONER（非理性个性化，arXiv:2601.16621）、PersonalBench、AlpsBench（个性化偏见/过度依赖记忆）形成‘过度/误用个性化’主题群；与同一哈工大团队的 PS-Bench（意图合法化安全失效，arXiv:2601.17887，作者高度重叠：Guo/Hu/Long/Sui/Zhao/Qin）、MEMDRIFT（记忆诱导工具漂移，arXiv:2605.24941）共属‘记忆安全’谱系。被评测对象 Mem0/MemU/MEMOS/LDAgent/RAG 即为本研究语料中其它‘用户中心个性化记忆系统’条目。Self-ReCheck 的相关性自检过滤思路与 RP-REASONER 的语用推理选择性整合、PerMem-Bench 的会话级存储门控属同类‘选择性使用记忆’方向。
- **可复现性**: 代码与数据截至调研未公开（伦理声明承诺录用后发布全部数据）；评测流程描述较完整（基座模型、6 记忆方法、判分提示模板、人工复核协议均在附录给出），但因缺少公开工件且依赖 LLM-as-Judge，独立复现存在一定门槛。社区采用信号尚弱（极新预印本、引用约 0、HuggingFace 有论文页但无数据集）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（启发式而非学习式）。Self-ReCheck 用冻结的轻量 LLM 做提示式二元相关性判断来控制‘哪些记忆参与生成’，不通过 RL/训练学习记忆管理策略，属无训练的启发/推理式过滤；论文核心在于揭示当前记忆系统缺乏‘是否应让某条记忆影响回答’的判定机制。
- **记忆主体**: 以用户为中心（user-centric）：记忆内容为用户画像、偏好、长期个性化历史，目标是个性化对话；本工作专门评测‘记住用户信息’被过度/不当使用的负面效应（区别于以智能体自身经验自我改进的 agent-centric 路线）。
- **多智能体记忆**: 单智能体（single-agent）。不涉及多智能体共享/路由记忆。
- **时序推理支持**: OP-Bench 本身不专门建模时间有效性；但联合评测的 LoCoMo 含时序（Temporal）问答类别，用以衡量被测系统的时序记忆能力（如 MEMOS 在 Temporal 上 F1 45.71 明显领先）。该工作关注的‘过度个性化’与‘陈旧偏好’相关，但时间有效性建模非其核心。
- **模态**: 纯文本（text-only）。
- **过度个性化/记忆安全风险**: 本工作即专门面向该负面维度的旗舰基准：系统化定义并量化过度个性化（侵入式/无关/谄媚/重复 的记忆使用），实证‘引入记忆普遍加剧过度个性化（↓26.2%~61.1%）’，揭示‘记忆劫持’机理，并提出 Self-ReCheck 缓解（↓29% 过度个性化且保持/略升个性化），直接呼应 fields.yaml 中‘更多记忆并非总是更好’、与 OP-Bench/Causal-LoCoMo 同列的记忆安全主题。
- **冲突/矛盾处理**: 不直接处理事实冲突/矛盾的合并更新；但 Sycophancy 任务中的 Memory-level 子型专门测试模型是否会认可‘记忆库中并不存在的伪造记忆’（即对虚假/冲突记忆的抵抗力），属与冲突处理相关的诊断维度。Self-ReCheck 通过相关性过滤间接降低对欺骗性（baiting）记忆的误用。

**不确定字段 / Uncertain**

- 代码链接 (`code_url`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="f8-causal-locomo--因果记忆干预-cmi别名causal-memory-intervention论文题为causal-intervention-based-memory-selection-for-long-horizon-llm-agentscausal-locomo-是其配套基准cmi-是其配套方法"></a>

### F8 Causal-LoCoMo / 因果记忆干预

*Causal-LoCoMo / 因果记忆干预 (CMI)（别名：Causal Memory Intervention；论文题为《Causal Intervention-Based Memory Selection for Long-Horizon LLM Agents》。Causal-LoCoMo 是其配套基准，CMI 是其配套方法）*


**基本信息 / Provenance**

- **年份**: 2026（arXiv 预印本，发布于 2026-05-17）
- **发表venue**: arXiv 预印本（截至调研未见正式会议/期刊收录）
- **论文链接**: https://arxiv.org/abs/2605.17641
- **代码链接**: https://github.com/Saksham4796/causal-memory-intervention（Python 98.6%/TeX 1.4%，公开；截至调研 0 star、0 fork，仅 1 次提交，2026-05-09）
- **引用数**: 约 0（Semantic Scholar 实时，CorpusId 288650056；论文极新，2026-05 发布）

**记忆分类 / Taxonomy**

- **记忆类型**: 面向情景式/语义式持久记忆（episodic/semantic）——记忆库由历史多会话对话中抽取的自然语言事实片段构成，覆盖时间型、多证据型、推断型与事实型记忆条目；属于评测基准与记忆选择方法，而非完整记忆系统
- **记忆结构**: 结构化记忆库（structured memory bank）：每个样例包含一个用户请求 + 一组带角色标注的记忆条目（有用 useful / 无关 irrelevant / 有害 harmful）。基准从 LoCoMo 长对话改造而来；基线方法分别用向量库、轻量词法-范围图、反思串、摘要等结构表示同一记忆库
- **存储后端**: 记忆库以 JSONL 文件存储（causal_locomo.jsonl 等）。检索阶段：向量基线用 text-embedding-3-large 嵌入做语义相似检索；图基线用记忆文本/范围构成的轻量词法图；CMI 用混合检索器（hybrid retriever）提候选。无专用向量数据库或图数据库（如 FAISS/Neo4j）
- **持久化**: 外部持久记忆（external durable store）——记忆作为跨会话持久存储的离线记忆库供智能体在推理时选择性读取；非参数化、非上下文内临时记忆

**核心机制 / Mechanisms**

- **写入/编码**: 本文聚焦记忆选择（读取）而非在线写入。基准构造侧的写入由 LLM 辅助流水线完成：以 GPT-5 作为数据构造模型，从 LoCoMo 对话/问答的证据标注出发，生成 100 个候选样例；将支持金标答案的证据改写为清晰自洽的『有用记忆』，从同一对话采样真实但不支持答案的片段作为『无关干扰项』，并对部分样例插入合成『有害记忆』（这些有害记忆与金标答案矛盾、调换实体、引入错误时间信息或诱导无依据推断）；时间型问题将『昨天/上周/去年』等相对表达按会话时间戳解析为显式可审计的时间事实。经确定性过滤与 schema 校验后保留 87 个评测样例（432 个历史会话、491 条记忆：89 有用 / 348 无关 / 54 有害）
- **检索机制**: 核心创新在于以『因果有用性』替代『相似度』做记忆选择。CMI（因果记忆干预）流程：(1) 用混合检索器从记忆库 M 提出宽泛候选集 P_K(x,M)；(2) 对每个候选记忆 m_i 做三条件干预对照——无记忆 y_no~p_θ(y|x,∅)、含该记忆 y_with~p_θ(y|x,{m_i})、含扰动版 y_pert~p_θ(y|x,{m̃_i})；(3) 用确定性打分器 s(·) 计算 因果效用 Utility(m_i)=s_with^(i)−s_no 与 稳定性 Stability(m_i)=s_with^(i)−s_pert^(i)；(4) 选择规则：当且仅当记忆相对无记忆基线提升任务分数、且在扰动下保持效果（满足记忆预算 k 与风险过滤约束）时纳入 Ŝ_CMI，再由 y_CMI~p_θ(y|x,Ŝ_CMI) 生成最终答案。形式化为干预 do(C=S)、目标 S*=argmax_{|S|≤k} U(S)（组合不可解，故以逐条目干预近似）。对比标准检索 Ŝ_retr=TopK sim(x,m_i)。基线检索机制：向量（嵌入相似）、图（词法-范围图邻近）、反思（关键词匹配反思串）、摘要、全历史、无记忆
- **反思/巩固**: 本方法本身不做经验到洞见的反思/抽象（CMI 是选择层而非巩固层）。但有一个『反思记忆』基线（受 Reflexion / Shinn et al. 2023 启发）：将记忆条目按其标签确定性地构造成反思串（把有用记忆转为『教训式』反思、把有害记忆转为『安全导向』反思）后再做关键词检索——注意这是标签感知的反思式表示，而非真正的 LLM 自我反思。CMI 的『干预诊断』可视为对每条记忆因果效用/稳定性的元级总结，但不回写记忆库
- **遗忘/更新**: 无显式的遗忘/衰减/合并/去重机制（非时间衰减、无 ADD/UPDATE/DELETE 操作）。CMI 通过在选择时『抑制/拒绝』因果效用为负或扰动下不稳定的记忆来实现功能性遗忘——即每次查询动态过滤记忆库（平均仅选 <1 条记忆），但记忆库本身不被编辑。冲突/矛盾记忆作为『有害记忆』被干预打分识别并排除
- **经验回放 (核心主题)**: 不属于经验回放范式。CMI 不复用过去轨迹改进未来行为，而是在回答前对候选记忆做『反事实式干预重放』：对同一查询分别在无记忆/含记忆/含扰动记忆三种上下文下让模型作答并打分，以此估计单条记忆对当前答案的因果贡献。这是一种『推理时干预探测』而非『跨回合策略蒸馏/技能复用/重放缓冲』。基准侧保留原 LoCoMo 历史会话供全历史基线回放整段对话

**学习维度 / Learning**

- **学习范式**: 非参数化、提示层（non-parametric / in-context）。无梯度更新；CMI 通过推理时干预对照与确定性打分在提示中做记忆选择。响应模型 GPT-4.1、判分模型 GPT-5 均冻结。属于纯推理时方法
- **失败学习 (核心主题)**: 以『有害/无关记忆』作为负样本进行鲁棒性评测，而非从智能体自身失败轨迹中学习。CMI 的稳定性检验（Stability=s_with−s_pert）专门用于发现『仅因脆弱/误导性措辞而看似有益』的记忆并予以拒绝：效用为正但稳定性为负的记忆被判为不可靠。诊断显示有用记忆平均效用 +0.307、无关记忆 −0.009、有害记忆 −0.033，证明干预分数能区分会拉低表现的有害记忆。基准引入合成有害记忆（矛盾、实体调换、错误时间、诱导无依据推断）专门测试对误导性检索的抵抗
- **技能/程序归纳**: 不诱导可复用技能/工作流；基准含『程序性记忆 procedural memory』等任务族（在配套 CausalMemBench 合成数据中），但 CMI 方法本身不归纳或调用技能/过程
- **在线 vs 离线**: 离线构建 + 推理时在线选择：记忆库与基准为离线批量构造（GPT-5 生成 + 确定性过滤）；CMI 的记忆选择在部署/推理时按每个查询在线执行干预打分。不在部署中持续写入新记忆

**评测 / Evaluation**

- **任务领域**: 多会话长程对话记忆问答（multi-session long-term conversational memory QA）；细分为时间推理、多证据、推断、事实四类记忆问答。还有配套合成基准 CausalMemBench 覆盖偏好更新、情境偏好、程序性记忆、虚假语义陷阱、冲突记忆、投毒记忆、多跳记忆组合、弃答等八个任务族
- **基准**: Causal-LoCoMo（本文提出，从 LoCoMo (Maharana et al., 2024) 改造，87 个过滤后样例：46 时间型 / 25 多证据 / 14 推断型 / 2 事实型；491 条记忆）；以及配套合成基准 CausalMemBench。基础数据源为 LoCoMo（多会话长对话）
- **报告增益**: 在 87 个过滤后 Causal-LoCoMo 样例上（响应模型 GPT-4.1，判分 GPT-5，温度0），CMI 在主表全面领先：任务分数 0.846（最高）、成功率 0.816（成功=混合分≥0.7）、有用记忆 F1 0.875、坏记忆拒绝率 0.990、投毒记忆采纳率 0.000、平均选记忆数仅 0.943 条。对比关键基线：反思记忆 任务分 0.845/F1 0.486/拒绝 0.557/投毒采纳 0.540/选3条；向量记忆 0.839/0.501/0.566/0.609/选3条；图记忆 0.824/0.469/0.550/0.586/选3条；摘要记忆 0.723/0.308/0.000/0.621/选5.644条；全历史 0.515/0.308/0.000/0.621/选5.644条；无记忆 0.429/0.000/拒绝1.000/投毒0.000/0条。即 CMI 任务分≈最强基线但 F1 高出约 0.37、投毒采纳从约 0.54-0.61 降到 0；任务族上 CMI 在多证据QA最佳(0.754)、事实QA并列最佳(0.985)，时间QA(0.901)略逊于向量记忆(0.925)，推断QA(0.808)略逊于反思记忆(0.823)。混合任务分 s=0.7·s_det+0.3·s_judge
- **对比基线**: 无记忆(No Memory)、全历史提示(Full History，长上下文)、摘要记忆(Summary Memory)、向量记忆(Vector Memory，RAG/稠密检索)、图记忆(Graph Memory，词法-范围图)、反思记忆(Reflection Memory，受 Reflexion 启发)。覆盖无记忆/全上下文/RAG/图/反思/摘要等代表性记忆策略

**分析 / Analysis**

- **关键创新**: 把记忆选择从『相似度检索问题』重构为『因果决策问题』（Pearl 干预 do-演算）：一条记忆被选中不是因为与查询语义相似，而是因为在受控干预下它能稳定地改善当前答案；并配套提出首个带因果角色标注（有用/无关/有害）的对话记忆选择基准 Causal-LoCoMo，可同时度量答案质量与记忆选择安全性（投毒采纳率）
- **局限**: (1) 受控标注设定：实现中记忆带 useful/irrelevant/harmful 角色标注且部分方法直接使用，故验证的是『显式因果结构的价值』而非智能体能否从原始记忆自行推断因果角色——可部署版需用预测的因果角色估计替代金标标签；(2) 基准小：仅 87 个样例，集中于时间/多证据/推断型，未覆盖陈旧偏好、错误摘要、实体链接错误、冲突更新、长期漂移等真实记忆失败；(3) 有害记忆为合成对抗插入，未必反映自然/人工的记忆腐化；(4) 评分依赖 GPT-5 判分器，存在校准/提示敏感/偏置风险；(5) 计算开销大：CMI 每条候选记忆需额外做无记忆/含记忆/扰动三次干预调用，增加延迟与 token 用量（建议缓存效用、仅对高风险查询干预、或用小型验证模型）
- **与其他工作关系**: 属于 F 类『记忆评测基准』。直接派生自 LoCoMo (Maharana et al., 2024) 长对话记忆基准，与同样改造 LoCoMo 的 LoCoMo-Plus（侧重隐式约束认知记忆）平行。安全维度上延续作者前作 MemoryGraft (Srivastava & He, 2025) 关于投毒经验检索持久危害的发现，并借鉴 RAG 投毒/提示注入研究（PoisonedRAG, Zou et al. 2025；Liu et al. 2023）。方法上把 NLP 因果中介分析/因果追踪/模型编辑（Vig 2020；Meng 2022；Geiger 2025）从『干预神经元/权重』迁移到『干预外部记忆上下文』。基线对接 Reflexion (Shinn 2023)、向量 RAG (Lewis 2020; Karpukhin 2020)、全历史 (LoCoMo)；与 Generative Agents、MemoryBank、MemGPT、Voyager 等『把记忆当作检索后即有用资源』的工作形成对照——本文强调被检索记忆并非一律有用。与本研究中『过度个性化/有害记忆』维度（如 OP-Bench、over-personalization）主题同源
- **可复现性**: 可复现性较好：代码完全开源（GitHub: Saksham4796/causal-memory-intervention，含基准构造脚本 build_causal_locomo.py、实验运行器、生成的 causal_locomo.jsonl 数据、配置与论文产物脚本）；流水线保存配置/提示/响应/分数/所选记忆/诊断/成本/日志，数据生成与本地打分默认设种子且确定性。但社区采纳信号弱（0 star/0 fork，仅 1 次提交）；实验依赖 OpenAI API（GPT-4.1/GPT-5/text-embedding-3-large），全量复现需付费 API 且 CMI 多次干预调用成本较高

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否——采用启发式+干预对照的固定流水线（混合检索→风险过滤→三条件干预打分→规则选择），不使用 RL/训练去学习『何时/存什么/取什么』的记忆管理策略。作者指出可部署版应将金标角色标签替换为『学习或提示得到的因果角色预测器』，但本文未实现
- **记忆主体**: 兼具用户中心与任务中心，偏用户中心：记忆是从多会话对话中关于用户/说话人的事实（偏好、事件、时间等），用于跨会话回答用户请求；但评测核心是『智能体应选哪些记忆』这一选择能力（含对自身记忆库的安全过滤），不是自我经验改进型（非 ReasoningBank/Voyager 式 agent-centric）
- **多智能体记忆**: 单智能体记忆。基准/方法均针对单个长程智能体的记忆库选择，无多智能体共享或记忆路由（非 G-Memory/MIRIX 式）
- **时序推理支持**: 显式支持时间推理：基准含 46 个时间型记忆问答（最大类别），构造时将『昨天/上周/去年』等相对时间表达按会话时间戳解析为显式可审计的绝对时间事实。但不建模事实有效期窗口/事件日历（非 Zep/Graphiti 式双时态图）；CMI 在时间QA(0.901)上略逊于向量记忆(0.925)
- **模态**: 纯文本（text-only）。基准源 LoCoMo 含多模态，但 Causal-LoCoMo 仅用其文本对话与问答，无视觉/具身记忆
- **过度个性化/记忆安全风险**: 核心关切维度。本文正是为应对『更多记忆未必更好』而设计：记忆库含合成有害记忆（矛盾、实体调换、错误时间、诱导无依据推断），并以『投毒记忆采纳率』『坏记忆拒绝率』量化记忆安全。CMI 实现近乎完美的有害记忆拒绝(0.990)与零投毒采纳(0.000)，而向量/图/反思基线投毒采纳达 0.54-0.61。伦理章节强调记忆含敏感个人属性，需数据最小化、用户同意、记忆过期、来源追踪、隐私审计；并提示因果过滤本身有权衡（过严会抑制敏感但相关上下文，过松会强化陈旧/错误/偏见记忆）
- **冲突/矛盾处理**: 通过干预打分隐式处理：矛盾/投毒记忆被建模为『有害记忆』，其因果效用为负（平均 −0.033）且/或扰动下不稳定，从而被 CMI 选择规则排除；时间型相对表达解析为显式时间事实以避免时间冲突。无显式的 UPDATE/合并式冲突解析操作（区别于 Memory-R1 的 UPDATE），而是『选择时拒绝』

**不确定字段 / Uncertain**

- 作者/机构 (`authors_institution`)
- token成本/延迟证据 (`token_cost_latency_evidence`)


<a id="f9-memoryagentbench论文标题evaluating-memory-in-llm-agents-via-incremental-multi-turn-interactions面向记忆智能体memory-agents的统一评测基准提出按四项核心记忆能力精确检索-accurate-retrieval--测试时学习-test-time-learning--长程理解-long-range-understanding--选择性遗忘-selective-forgetting评测并以增量多轮交互incremental-multi-turn协议把长上下文数据集改造为逐块顺序注入自建-eventqa-与-factconsolidation-两个新数据集官方-github-仓库与-huggingface-数据集名亦作-memoryagentbench注意与同处f-记忆评测基准簇的-f3-membencharxiv-250621605人民大学华为为两个不同基准本条-arxiv-250705257由-ucsd-出品二者无血缘关系"></a>

### F9 MemoryAgentBench

*MemoryAgentBench（论文标题《Evaluating Memory in LLM Agents via Incremental Multi-Turn Interactions》；面向「记忆智能体（memory agents）」的统一评测基准；提出按四项核心记忆能力——精确检索 Accurate Retrieval / 测试时学习 Test-Time Learning / 长程理解 Long-Range Understanding / 选择性遗忘 Selective Forgetting——评测，并以增量多轮交互（incremental multi-turn）协议把长上下文数据集改造为逐块顺序注入；自建 EventQA 与 FactConsolidation 两个新数据集；官方 GitHub 仓库与 HuggingFace 数据集名亦作 MemoryAgentBench。注意：与同处「F. 记忆评测基准」簇的 F3 MemBench（arXiv 2506.21605，人民大学/华为）为两个不同基准，本条 arXiv 2507.05257、由 UCSD 出品，二者无血缘关系。）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本 2507.05257 于 2025-07-07 首次公开 v1；后被 ICLR 2026 录用，仓库标注 ‘ICLR 2026 Paper’；HuggingFace 数据集亦标注 Published Jul 7, 2025）
- **作者/机构**: Yuanzhe Hu（胡远哲，并列第一作者、并列通讯）、Yu Wang（王宇，并列第一作者、并列通讯）、Julian McAuley（朱利安·麦考利，资深作者）。三人均来自美国加州大学圣地亚哥分校（University of California, San Diego, UCSD），邮箱 {yuh127, yuw164, jmcauley}@ucsd.edu。McAuley 课题组长期从事推荐系统与序列建模研究。
- **发表venue**: ICLR 2026（International Conference on Learning Representations 2026；官方代码仓库与多个 Awesome-list 均标注为 ‘ICLR 2026 Paper’）。arXiv 同步预印本 2507.05257（DOI 10.48550/arXiv.2507.05257，DBLP journals/corr/abs-2507-05257，CorpusId 280136659）。属带开源代码/数据集的学术评测基准（benchmark/dataset 论文，而非记忆系统/方法）。
- **论文链接**: https://arxiv.org/abs/2507.05257
- **代码链接**: https://github.com/HUST-AI-HYZ/MemoryAgentBench（官方代码仓库，约 300–345 stars、约 53 forks、以 Python 为主，MIT 许可，截至调研日 2026-06，最后推送约 2026-01-27；含 main.py 评测主循环、各类 memory agent 实现、llm_based_eval 评测脚本等）；配套数据集托管于 HuggingFace：https://huggingface.co/datasets/ai-hyz/MemoryAgentBench（约 76.6 MB，parquet 格式，含 RULER QA / EventQA / FactConsolidation / ICL 系列 / Recsys / LongMemEval / InfBench 等子集，许可证 MIT，数据约 38 likes）。
- **引用数**: 约 108 次引用（Semantic Scholar，CorpusId 280136659，截至调研日 2026-06）；作为 2025 下半年发布的 LLM 记忆智能体评测基准，半年内即被高频引用，被多个智能体记忆 Awesome-list（TsinghuaC3I、TeleAI-UAGI、IAAR-Shanghai 等）收录为代表性 benchmark。

**记忆分类 / Taxonomy**

- **记忆类型**: 本身是评测基准而非记忆系统；它从认知/记忆科学经典理论出发，把待评测的记忆能力划分为四项正交核心能力：①精确检索 AR（从长历史中取回正确片段，含单跳/多跳）；②测试时学习 TTL（部署期无需训练即吸收新行为/技能，对应 in-context learning 式学习）；③长程理解 LRU（整合分散在 ≥100k token 上下文中的信息做全局理解）；④选择性遗忘 SF（面对矛盾证据时修订/覆盖/删除旧记忆，对接 model editing 与 knowledge unlearning）。被评测的记忆机制覆盖工作记忆（上下文窗口）、外部情景/语义记忆（向量库、知识图谱、事件时间线）等，但论文聚焦文本历史与外部数据库式记忆，明确把参数化（权重内）记忆排除在主要评测范围之外。
- **记忆结构**: 作为基准，其「结构」体现在数据集组织与被评测智能体的三大类记忆架构，而非单一数据结构：①长上下文智能体（Long-Context Agents）——以 FIFO 上下文缓冲区为记忆；②RAG 智能体——分为简单 RAG（原始文本块 + BM25/TF-IDF 等字符串匹配）、嵌入式 RAG（dense 向量 + 余弦相似检索）、结构增强 RAG（构建知识图谱/树/事件时间线，如 GraphRAG/RAPTOR/HippoRAG-v2/Cognee/Zep/MemoRAG/Mem0）；③智能体式记忆（Agentic Memory）——带 agentic loop 的迭代检索-反思-更新（MemGPT/Self-RAG/MIRIX）。数据层面：所有数据被切成 chunk（块大小 512 或 4096）以模拟多轮交互，按时间顺序逐块注入；基准对各机制结构保持中立、统一评测。
- **存储后端**: 基准自身不绑定特定后端；为各被评测机制配置常见后端：长上下文智能体用模型上下文窗口；嵌入式 RAG 用 Contriever / OpenAI text-embedding-3-small / text-embedding-3-large / Qwen3-Embedding-4B 等嵌入器 + 向量检索；简单 RAG 用 BM25；结构增强 RAG 用知识图谱/树（GraphRAG、RAPTOR、HippoRAG-v2）或商用记忆服务后端（Mem0、Cognee、Zep）；智能体式记忆用 MemGPT、MIRIX、Self-RAG。基座 LLM 默认 GPT-4o-mini（RAG 与商用记忆智能体统一以其为骨干），另对比 GPT-4o、GPT-4.1-mini、Gemini-2.0-Flash、Claude-3.7-Sonnet，并用 o4-mini 做数据集可解性验证。数据以 parquet/JSON 形式经 HuggingFace 分发。
- **持久化**: 基准评测的是「跨增量多轮交互、长历史」下的外部持久记忆与上下文记忆能力。评测协议为：把数据切块后逐块以「用户-助手」对话形式按时间顺序注入，每块都附带「请记住此内容、稍后会提问」的记忆指令；待全部块注入完毕再提问，历史仅能通过各机制的记忆召回。被评测机制覆盖 in-context（长上下文 FIFO）与外部持久存储（向量库/图谱/商用记忆服务）两类持久性。论文明确说明主要聚焦文本历史与外部数据库式记忆，参数化（权重内、如 MemoryLLM/M+/SELF-PARAM）记忆因多停留在学术研究、能力通常弱于闭源 API 记忆系统而未纳入主要评测。

**核心机制 / Mechanisms**

- **写入/编码**: （基准视角）数据构造与「写入」协议：①数据来源——重构既有长上下文数据集（NIAH 式单/多跳文档 QA、LongMemEval 对话、∞-Bench 摘要 En.Sum、DetectiveQA、BANKING77/CLINC150/NLU/TREC 分类、电影推荐对话），并自建两个新数据集 EventQA（基于小说自动化流水线，给定最多 5 个先前事件后从候选中选下一事件，考察时序推理）与 FactConsolidation（基于 MQUAKE 反事实编辑对构造，把「真事实→改写后矛盾的新事实」按序拼接成 6K/32K/64K/262K 长上下文，模拟知识更新）；②写入协议——所有智能体被要求逐块吸收 chunk 并增量更新记忆；每块切分为 512（文档 QA、LongMemEval、SF 任务）或 4096（其余任务及 Mem0/Cognee/Zep/MIRIX 统一）token。各机制按自身策略执行写入编码：长上下文直接拼接；RAG 把块存为原始文本或嵌入向量；结构增强 RAG 在全部摄入后构建知识图谱/树/事件时间线；智能体式记忆经 agentic loop 抽取并写入工作记忆。基准本身不提出新的编码方法，而是统一注入协议并量化各机制写入行为的后果（与延迟，见 Appendix E.5）。
- **检索机制**: （基准视角）基准统一注入协议后，按机制类别评测不同读取策略：①长上下文智能体——不显式检索，直接让模型在当前窗口内注意（依赖位置近期性，超窗即 FIFO 丢弃最早块）；②简单 RAG——BM25/TF-IDF/BMX 字符串匹配取 top-k；③嵌入式 RAG——query 嵌入 + 余弦相似度取 top-k（默认 top-k=10，并做 k=2/5/10 消融）；④结构增强 RAG——基于已构建的知识图谱/树/事件时间线检索（GraphRAG、RAPTOR、HippoRAG-v2 个性化 PageRank 等）；⑤智能体式记忆——agentic loop 迭代重写问题、多次记忆查找、更新工作记忆（MemGPT 分层调页、Self-RAG 自反思检索、MIRIX）。基准把检索质量直接反映在端到端任务准确率上，并做块大小（更小块对 AR 更有利、对 LRU 有害）与 top-k（增大一般更好，但 4096×10≈40k token 已逼近容量上限，故不评 20 块）两类消融；本基准不提出新检索算法，而是横向评测各检索范式在四能力上的迁移效果。
- **反思/巩固**: （基准视角）基准并不实现反思机制，而是把「需要把分散历史抽象/巩固为全局理解或新规则的能力」作为重点评测维度，尤其体现在长程理解 LRU（小说摘要 ∞-Bench-Sum 需梳理全书情节与人物、DetectiveQA 需跨长叙事推理）与测试时学习 TTL（从大量上下文示例中归纳分类规则/推荐偏好）。论文据此揭示一个核心论点：记忆不是对历史的逐字存储，而是「压缩、蒸馏」的表征——理应抽取要点、剔除无关、并据历史生成新推断；正因如此，把整段上下文一次性给出的长上下文数据集不适合评测记忆智能体，必须改为增量注入、考察机制能否在线巩固。实验发现：RAG 与商用记忆智能体只能取回片段、缺乏全局整合/巩固能力，故在 TTL、LRU 上反而不如直接读全文的长上下文模型——暴露「检索式记忆难以做反思/巩固式整合」这一开放难题。
- **遗忘/更新**: 选择性遗忘 SF 是 MemoryAgentBench 区别于既有记忆基准（LoCoMo/LongMemEval/RealTalk/StoryBench 均不覆盖 SF）的核心新增维度。基准用自建 FactConsolidation 数据集评测：基于 MQUAKE 反事实编辑对，把「原始真事实」与「改写后矛盾的新事实」按时间先后拼接（新事实序号更大、出现更晚），并在 prompt 中显式加 guardrail——告知「事实按序号索引、序号越大越新，冲突时须采用最新事实」，从而考察机制能否覆盖/修订/删除过时记忆。GitHub 仓库中该维度被重命名为「Conflict Resolution（冲突消解）」，对应 fact_sh/fact_mh 子集。核心发现：所有方法在多跳 SF 上几乎全部失败（最高仅约 7% 准确率），仅长上下文智能体在单跳 SF 上能取得尚可结果；即便换用更强推理模型 o4-mini，6K 短上下文可解、扩到 32K 即骤降——说明真正的「随时间选择性遗忘/更新」仍是未解难题。基准自身不提供 ADD/UPDATE/DELETE 算子，而是评测被测机制的更新/遗忘后果。
- **经验回放 (核心主题)**: 基本不适用（与「经验回放/轨迹复用」主题关系很弱）。MemoryAgentBench 评测的是「记住注入的历史信息并据此正确作答」，而非智能体把自身过去成功/失败轨迹作为范例/技能/回放缓冲来自我改进。最接近的维度是测试时学习 TTL：智能体需利用上下文中已见的带标签示例（如 BANKING77/CLINC150 的 in-context 示例、电影推荐对话历史）来对新输入分类/推荐——这是 in-context 示例驱动的学习，可视为对「历史示例的复用」，但并非 agent-centric 的轨迹回放或技能蒸馏。基准未设计成功/失败轨迹回放、技能复用或 replay buffer 等经验回放机制的评测任务，更适合评测 user/history-centric 的记忆系统而非 ReasoningBank/Voyager 类自进化智能体。

**学习维度 / Learning**

- **学习范式**: 非参数化（non-parametric）评测设定。所有被评测记忆智能体均在冻结基座 LLM（默认 GPT-4o-mini，另含 GPT-4o、GPT-4.1-mini、Gemini-2.0-Flash、Claude-3.7-Sonnet、o4-mini 等）之上，以上下文/外部记忆（向量库、图谱、agentic loop）方式运行，不做任何梯度更新。基准本身评测 inference-time、非参数化的记忆读写、整合与更新能力；论文明确把参数化（权重内）记忆排除在主要评测之外，认为其多停留在学术研究、能力通常弱于闭源 API 记忆系统。其中测试时学习 TTL 维度评测的是「无需训练、在部署期通过上下文示例获取新行为」的能力，仍属非参数化范畴。
- **失败学习 (核心主题)**: 不适用。作为 history/user-centric 记忆评测基准，MemoryAgentBench 不涉及「检测失败轨迹并据此改进」的 agent-centric 失败学习主题。它通过任务准确率（substring/exact match、Recall@5、LLM-as-judge F1）量化记忆机制是否正确检索/整合/更新信息，这是对记忆有效性的评分，而非让智能体从自身失败经验中学习负例、失败模式或错误驱动规则。基准未设计失败反思、负面范例等机制的评测任务。
- **技能/程序归纳**: 基本不适用。基准聚焦四类记忆能力（精确检索、测试时学习、长程理解、选择性遗忘），不评测从经验中归纳可复用技能/工作流/程序（procedural skill induction）的能力，也不涉及技能表示与调用。其测试时学习 TTL 维度考察的是「依据上下文示例做分类/推荐」的 in-context 适应，而非诱导并存储可执行的程序性技能；论文亦把「学习新技能 acquire new skills」笼统归入 TTL 的能力描述，但评测形式仍是分类/推荐准确率，而非技能库构建与复用。
- **在线 vs 离线**: 评测协议为在线/增量（online / incremental，模拟时间流）：把数据切块后逐块按时间顺序注入、智能体增量吸收并更新记忆，全部注入后再提问，历史仅能经记忆召回——以此模拟真实部署中随交互累积记忆的过程，刻意区别于把整段上下文一次性给出的静态长上下文评测。数据集构造本身是离线批量完成（重构既有数据集 + 自动化流水线生成 EventQA/FactConsolidation + 噪声/干扰拼接）。即数据离线构建、记忆能力以在线增量方式评测。为提高利用率，刻意设计「单段长上下文配多道题」（如 LongMemEval(S*) 用 5 段上下文配 300 题），避免为每题重建百万 token 记忆。

**评测 / Evaluation**

- **任务领域**: 长期记忆问答与长上下文理解评测，覆盖四类能力对应的多域任务：①精确检索 AR——NIAH 式单/多跳文档 QA、长对话 QA（LongMemEval(S*)）、小说事件时序 QA（EventQA）；②测试时学习 TTL——多类意图/问题分类（BANKING77 银行意图 77 类、CLINC150 151 类、NLU 68 类、TREC-Coarse 6 类、TREC-Fine 50 类）与电影推荐；③长程理解 LRU——小说摘要（∞-Bench-Sum）、侦探小说长程推理 QA（DetectiveQA）；④选择性遗忘 SF——基于反事实编辑对的事实判定（FactConsolidation 单跳/多跳）。上下文深度 103K–1.44M token。不涉及网页导航/具身/编码/GUI 等智能体行动域，属 QA/对话/分类/摘要式记忆评测。
- **基准**: MemoryAgentBench 本身即为新提出的统一基准（共 2071 道题、上下文深度 103K–1.44M token），由 13 个子数据集组成：精确检索类——SH-Doc QA（单跳，AvgL≈197K）、MH-Doc QA（多跳，≈421K）、LongMemEval(S*)（对话 QA，≈355K，5 段上下文配 300 题）、EventQA（自建，≈534K）；测试时学习类——BANKING77（≈103K）、CLINC150、NLU、TREC-Coarse、TREC-Fine、Movie Recommendation（Recall@5，≈1.44M）；长程理解类——∞-Bench-Sum（F1/LLM-as-judge，≈172K）、DetectiveQA（≈124K，10 部小说 71 题）；选择性遗忘类——FactConsolidation-SH 与 FactConsolidation-MH（自建，基于 MQUAKE，长度 6K/32K/64K/262K）。论文中对比的既有记忆/长上下文基准：MemoryBank(194 题/5K)、LoCoMo(7512 题/10K)、PerLTQA(8593 题/≈1M)、RealTalk(728 题/≈375K)、LongMemEval(500 题/115K–1.5M)、StoryBench(86 题)，并指出它们均不能同时覆盖 AR/TTL/LRU/SF 四能力。
- **报告增益**: 本文是评测基准，核心产出是对多类记忆智能体的横向测评（非提出新方法的增益）。关键数字（Table 3，准确率/分数满分 100；除注明外 RAG 与商用记忆智能体均以 GPT-4o-mini 为骨干）：【总分 Overall】GPT-4o 48.8、Claude-3.7-Sonnet 49.6（最高）、GPT-4.1-mini 46.9、Gemini-2.0-Flash 42.4、GPT-4o-mini 42.2/42.3；BM25 41.5、HippoRAG-v2 41.6、Text-Embed-3-Large 38.0、Qwen3-Embedding-4B 38.2、Text-Embed-3-Small 37.1、MIRIX(GPT-4.1-mini) 37.7、Contriever 29.8、MemoRAG 30.9、MemGPT 28.3、MIRIX(GPT-4o-mini) 26.2、Zep 24.0、GraphRAG 23.4、Mem0 21.1、Cognee 20.6、Self-RAG 18.7。【分能力规律】①AR：多数 RAG 智能体优于 GPT-4o-mini 骨干（如 HippoRAG-v2 AR 均分 65.1、BM25 60.5，对比 GPT-4o-mini 49.2），印证 RAG 擅长取片段；②TTL+LRU：长上下文模型最佳（Claude-3.7-Sonnet TTL 53.9、LRU 62.2；GPT-4o LRU 54.9），而 RAG/商用记忆智能体因只取部分信息、缺乏全局整合而显著落后（Mem0 LRU 仅 20.7、GraphRAG 仅 19.9、∞-Bench-Sum 上 GraphRAG 仅 0.4）；③SF：几乎全军覆没——多跳 FC-MH 所有方法 ≤7%，单跳 FC-SH 仅长上下文模型尚可（GPT-4o 60.0、HippoRAG-v2 54.0）。【骨干影响】(Table 4) RAG 对骨干不敏感（升级到 GPT-4.1-mini 仅微增，如 BM25 38.8→40.2），但 Agentic Memory 对骨干高度敏感——MIRIX 由 GPT-4o-mini 换 GPT-4.1-mini，EventQA +23.2、∞-Bench-Sum +9.0、FC-SH +6.0、四项均分 15.9→25.6（+9.7）。【数据集验证】(Table 5) o4-mini 在 FC-SH 6K 达 100、扩到 32K 降至 61；FC-MH 6K 达 80、32K 骤降至 14；GPT-4o 同维度 92/88、28/10——证明数据集短上下文可解、长上下文下现有记忆智能体仍力不从心。
- **对比基线**: 本文横向评测的三大类记忆智能体即为对比对象：①长上下文智能体（FIFO 全文缓冲）——GPT-4o、GPT-4o-mini、GPT-4.1-mini、Gemini-2.0-Flash、Claude-3.7-Sonnet；②RAG 智能体——简单 RAG（BM25）、嵌入式 RAG（Contriever、OpenAI text-embedding-3-small、text-embedding-3-large、Qwen3-Embedding-4B）、结构增强 RAG（RAPTOR、GraphRAG、MemoRAG、HippoRAG-v2、Mem0、Cognee、Zep）；③智能体式记忆——Self-RAG、MemGPT、MIRIX（含 GPT-4o-mini 与 GPT-4.1-mini 两骨干）。其中长上下文模型直接读全文，充当「full-context 上限对照」；BM25/嵌入式 RAG 充当经典 RAG 对照；商用/结构化记忆系统（Mem0、Zep、Cognee、MIRIX、MemGPT）则代表最新外部记忆方案。

**分析 / Analysis**

- **关键创新**: 首个同时覆盖「精确检索 + 测试时学习 + 长程理解 + 选择性遗忘」四项核心记忆能力、并采用增量多轮交互注入协议的 LLM 记忆智能体统一评测基准：①从认知/记忆科学经典理论提炼四项互补能力，并指出既有基准（LoCoMo/LongMemEval/RealTalk/StoryBench 等）均无法同时覆盖（尤其普遍缺失 TTL 与 SF）；②揭示「记忆 ≠ 长上下文」——记忆应是历史的压缩蒸馏表征，故把一次性给出的长上下文数据集改造为「逐块按时间顺序注入、附记忆指令、注入完毕再提问」的增量协议，更贴合记忆智能体真实工作方式；③自建两个新数据集 EventQA（小说事件时序、全自动可扩展流水线）与 FactConsolidation（基于 MQUAKE 反事实编辑对，专测随时间更新/选择性遗忘）；④在统一协议下横向评测长上下文、三类 RAG 与商用 Agentic Memory 共约 20 个系统，得出「无单一方法精通全部四能力、SF 多跳几乎全部失败、Agentic Memory 强烈依赖骨干强度」等系统性结论。
- **局限**: ①作者自承因预算约束，只能在「相对有代表性」的部分记忆智能体上做实验，未能覆盖更多/更新系统（如未纳入参数化记忆 MemoryLLM/M+、A-MEM、MemAgent、Mem1 等）；②评测主要以 GPT-4o-mini 为统一骨干，部分商用记忆系统（Mem0/Cognee/Zep/MIRIX）的低分可能部分源于统一 chunk_size=4096 的适配设置而非系统本身缺陷；③对话/对话历史的真实性仍有限，作者将「收集更真实的真实世界对话数据以丰富多样性与真实性」列为未来工作；④部分任务为节省成本采用「单段上下文配多题」，与逐题重建记忆的真实场景存在差距；⑤选择性遗忘任务在长上下文下几乎全员失败，基准能区分但尚无方法能解，揭示评测领先于方法；⑥摘要/LongMemEval 等子任务依赖 LLM-as-judge 评分，存在判分噪声。
- **与其他工作关系**: 属「F. 记忆评测基准」簇，与本研究其它系统形成「评测台 vs 被评测对象」关系，并直接测评了本研究覆盖的多个系统：B3 MemGPT、B9 MIRIX、D1/D2 HippoRAG(-v2)、D3 Zep/Graphiti、D4 Mem0、E2 MemoRAG，以及 GraphRAG、RAPTOR、Cognee、Self-RAG、BM25/Contriever 等。与同簇 F1 LongMemEval、F2 LoCoMo 同属对话/长期记忆 QA 基准，但本基准明确批评二者上下文偏短（LoCoMo ≈9–10K）或合成对话主题多样性不足（LongMemEval），并自我定位为「首个同时覆盖 AR/TTL/LRU/SF 四能力、采用增量多轮注入协议」的超集；其 LongMemEval(S*) 子集即在 F1 基础上重构而来。与 F3 MemBench（arXiv 2506.21605，人民大学/华为）是名称相近但完全独立的两个基准——后者强调「事实记忆×反思记忆 + 参与/观察场景」、本条强调「四能力 + 增量注入 + 选择性遗忘」，无作者/血缘重叠（本条 dedup 已确认）。整体属 history/user-centric 长上下文记忆评测，与 A 簇 ReasoningBank、D5 G-Memory 等 agent-centric 自进化记忆评测正交。
- **可复现性**: 可复现性较好：官方完整开源——代码 github.com/HUST-AI-HYZ/MemoryAgentBench（约 300–345 stars、约 53 forks，MIT 许可，含 main.py 评测主循环、各类 memory agent 实现、llm_based_eval 评测脚本、各任务指标说明）；数据集托管 HuggingFace ai-hyz/MemoryAgentBench（parquet，约 76.6 MB，含全部 13 个子集，可一行 load_dataset 加载，许可证 MIT，数据集另声明 CC BY 4.0）。可复现性声明承诺提供训练/评测脚本、配置、精确 prompt、含随机种子的数据生成脚本、端到端运行配方、容器化环境（Dockerfile + conda/requirements）与硬件/CUDA 细节。局限：部分骨干（GPT-4o-mini/4.1-mini、Gemini、Claude、o4-mini）与商用记忆服务（Mem0/Zep/Cognee/MIRIX）为闭源 API，复现需付费且结果可能随 API 版本漂移；LLM-as-judge 评分引入随机性。整体已被社区广泛收录、采用为标准记忆评测台。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否，且不适用。MemoryAgentBench 是评测基准，本身不学习记忆管理策略；其评测的所有机制（长上下文 FIFO、BM25/嵌入式/结构增强 RAG、MemGPT/MIRIX/Self-RAG）均为启发式/规则式或预置 agentic-loop 管线，无一通过 RL/训练学习「何时存/取/更新」的记忆策略。基准也未针对「学习型记忆控制」（Memory-R1/Mem-α 类）设计专门评测任务。其揭示的「Agentic Memory 强烈依赖骨干强度（MIRIX 换强骨干四项均分 +9.7）」恰反向说明：这些系统的记忆控制由 LLM 推理隐式驱动，而非学得的显式策略。
- **记忆主体**: 以历史/用户为中心（history- & user-centric）。MemoryAgentBench 评测「智能体记住注入的历史信息（文档、对话、事件、事实、用户偏好/示例）以正确检索、整合、更新作答」的能力，与 LongMemEval/LoCoMo/Mem0/Zep 同属此范畴；不评测智能体记忆自身行动经验以自我改进的 agent-centric 能力。其测试时学习 TTL 维度（从上下文示例学分类/推荐）虽涉及「从历史学习」，但仍是对外部输入的适应，而非复用自身成功/失败轨迹。
- **多智能体记忆**: 单智能体（single-agent）。评测设定为单个记忆智能体与单一模拟用户的增量多轮交互，不涉及多智能体间共享/路由记忆，也不评测 G-Memory/MIRIX 式多智能体记忆分层。（注：被评测的 MIRIX 系统内部含多记忆组件，但在本基准中作为单智能体记忆方案被整体评测，基准本身不考察跨智能体记忆路由。）
- **时序推理支持**: 部分支持，且为新增重点之一。①EventQA 专门考察「事件时序推理」——给定最多 5 个先前事件后从候选中选出正确的下一事件，需对长篇叙事中的时间顺序推理；②整个评测协议为时间感知的增量注入（数据按时间顺序逐块送入、历史经记忆召回），并对 LongMemEval 等对话保留会话时序；③FactConsolidation 通过「序号越大越新、冲突取最新」显式建模事实随时间的更新顺序。但其时间建模偏「事件顺序/更新先后」，并未像 Zep/Graphiti 那样显式建模事实有效性窗口（valid-from/valid-to 双时间区间）或事件日历。整体把时序能力分散在 AR（EventQA 时序）与 SF（FactConsolidation 更新顺序）两维中评测。
- **模态**: 纯文本（text-only）。所有子数据集均为文本——长文档、长对话、小说叙事、分类句子、推荐对话、事实编辑对；不涉及图像/截图/视觉或音频等多模态记忆评测（即便被评测的 MIRIX 本身支持截图记忆，在本基准中也仅以文本输入评测）。
- **过度个性化/记忆安全风险**: 基本未涉及该负面安全维度。MemoryAgentBench 关注记忆的有效性（四能力准确率）而非有害/过时/侵入/谄媚记忆与隐私治理，未设计 OP-Bench/Causal-LoCoMo 式安全评测。伦理声明仅泛述：遵循 ICLR 行为准则、所用对话与语料合规且不含 PII 或未成年人数据、为降低双重用途风险仅发布经安全筛查的 prompt 并附「不鼓励监控类应用」的使用说明、代码以 MIT、数据以 CC BY 4.0 发布。值得注意：其选择性遗忘（FactConsolidation）虽以「冲突取最新」隐含「应淘汰过时记忆」，但出发点是事实正确性而非记忆安全治理。
- **冲突/矛盾处理**: 显式涉及，且是核心新增维度之一（选择性遗忘 SF / 仓库中称 Conflict Resolution 冲突消解）。基准用自建 FactConsolidation 专测矛盾事实处理：基于 MQUAKE 反事实编辑对，把「原始真事实」与「改写后与之矛盾的新事实」按时间先后（新事实序号更大、出现更晚）拼接成长上下文，prompt 中显式指示「事实按序号索引、序号越大越新，遇冲突须采用最新事实、基于最终记忆状态推理」，分单跳 FC-SH（直接事实召回）与多跳 FC-MH（跨多事实推理）。核心发现：现有记忆机制普遍无法稳健消解长上下文中的矛盾——多跳几乎全部 ≤7%、单跳仅长上下文模型尚可，o4-mini 也在 32K 处骤降，说明「随时间更新并淘汰旧值」仍是开放难题。这与 Memory-R1 的 UPDATE、MEMTRACK 的冲突追踪关注点一致，但本基准是评测台而非提供消解算子。
- **token成本/延迟证据**: 基准在附录中报告计算开销而非自身的成本节省口径：Appendix E.5 给出各记忆智能体的计算延迟（computational latency）对比、E.6 给出 GPU 显存占用对比、附录 I 给出成本-性能（Cost-Performance）估计。论文正文亦指出效率约束影响设计选择：因 chunk_size=4096 时检索 10 块即约 40k token、对模型容量压力大，故不评 top-k=20；对 Mem0/Cognee/Zep/MIRIX 统一用 4096 块也是出于计算开销与 API 成本考量；并强调「单段长上下文配多题」可避免为单题注入 1M token 的资源浪费。基准未像 Mem0/Zep 那样报告「相对全上下文的 p95 延迟/token 节省百分比」这类系统级口径，其效率证据以横向延迟/显存/成本对比为主（具体数值见原文附录表，本调研未逐一抽取）。


## G. 学习/RL驱动的记忆控制 (Learned / RL-based memory control)


<a id="g1-memory-r1基于强化学习的-llm-外部记忆管理框架双智能体-memory-manager--answer-agent"></a>

### G1 Memory-R1

*Memory-R1（基于强化学习的 LLM 外部记忆管理框架；双智能体 Memory Manager + Answer Agent）*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本，v1 首次公开于 2025-08-27；最新 v5 修订于 2026-01-14）
- **作者/机构**: Sikuan Yan*、Xiufeng Yang*（*共同一作）、Zuchao Huang、Ercong Nie、Zifeng Ding、Zonggen Li、Xiaowen Ma、Jinhe Bi、Kristian Kersting、Jeff Z. Pan、Hinrich Schütze、Volker Tresp、Yunpu Ma†（†通讯，cognitive.yunpu@gmail.com）共 13 位作者。主要单位：慕尼黑大学（LMU Munich）与慕尼黑机器学习中心（MCML），合作单位包括慕尼黑工业大学（TUM）、剑桥大学、香港大学、达姆施塔特工业大学（TU Darmstadt）、爱丁堡大学。第一作者 Sikuan Yan 来自 LMU。
- **发表venue**: ACL 2026（第 64 届 ACL 年会，美国圣地亚哥，2026-07-02 至 07-07，主会论文，To be published；据 MCML 官方出版页 YYH+26）。arXiv 预印本归类于 cs.CL / cs.MA。属学术界研究成果并开源代码。
- **论文链接**: https://arxiv.org/abs/2508.19828

**记忆分类 / Taxonomy**

- **记忆类型**: 情景性/语义性记忆（episodic + semantic）：以「记忆条目（memory entries）」形式存储多会话对话中抽取的事实与事件。同时具备程序性意味——但程序性体现在「记忆管理策略本身被 RL 学到」（何时 ADD/UPDATE/DELETE/NOOP），而非显式存储可复用技能。本质是面向长程多会话 QA 的事实型外部记忆库（temporal memory bank）。
- **记忆结构**: 外部「记忆库（memory bank）」——由自然语言记忆条目组成的可演化条目集合（非图、非层级），每条记忆是一句话级别的事实陈述；通过基于嵌入向量的相似度检索（RAG）索引。记忆库随对话推进由 Memory Manager 增量维护与演化（structured CRUD 操作）。
- **存储后端**: 外部记忆库 + 相似度检索（embedding-based RAG），沿用 Mem0（Chhikara et al. 2025）的检索设定，每个问题召回最多 60 条候选记忆。论文未指定具体向量数据库实现；检索结果在运行时注入 Answer Agent 上下文。
- **持久化**: 外部持久化存储（durable external store）。记忆库跨多会话持续累积与更新；记忆管理与利用的「策略」则参数化固化进两个被 RL 微调的智能体权重中（即记忆内容是非参数外部存储，而控制策略是参数化的——这是与纯启发式管线的关键区别）。

**核心机制 / Mechanisms**

- **写入/编码**: 两阶段写入。先由一个 LLM 抽取器（LLMExtract）从每个对话轮中抽取并概括「值得记住」的信息 x；随后 Memory Manager（被 PPO/GRPO 微调的策略 π_θ）以抽取信息 x 与检索到的现有记忆 M_old 为输入，输出一个结构化操作 o∈{ADD, UPDATE, DELETE, NOOP} 及更新后的内容 m′，即 (o, m′) ∼ π_θ(·|x, M_old)，据此维护并演化记忆库。关键创新在于「写什么、是新增还是合并/删除」由 RL 学到的策略决定，而非固定脚本：相较 vanilla 管线遇到多会话相互关联事实时常错误地发出 DELETE+ADD（导致碎片化/信息丢失），Memory-R1 学会用 UPDATE 将互补信息合并巩固到同一条记忆。Memory Manager 的训练样本为「(对话轮, 时间性记忆库=前 50 轮构成, 对应 QA 对)」三元组。
- **检索机制**: 面对一个问题 q 时，先用基于相似度的 RAG 召回最多 60 条候选记忆 M_ret（沿用 Mem0 设定）；随后 Answer Agent（同样被 RL 微调的策略）执行「记忆蒸馏（Memory Distillation）」：从这 60 条噪声候选中筛选/挑出最相关条目，再在蒸馏后的上下文上进行推理生成答案 y ∼ π_θ(·|q, M_ret)。即检索本身是固定的相似度召回（检索器未学习），但「读后如何过滤与使用」是学习型的蒸馏策略。论文对比表明：相比额外加 reranker 的管线，学习型蒸馏在更低的中位/尾部延迟下取得更高准确率（Pareto 改进）。
- **反思/巩固**: 「巩固」体现在 Memory Manager 的 UPDATE 操作上：将语义重叠或互补的多会话信息合并进单条记忆，避免碎片化与上下文丢失——这是由 RL（以下游答案正确性为奖励）学到的巩固行为，而非显式反思蒸馏出高层洞见。Answer Agent 侧的「记忆蒸馏」则是读取时对召回噪声的过滤与精炼。本文不做 Generative Agents 式的周期性反思总结；其「raw→有用记忆」的转化由可学习的写入/读取策略隐式承担。
- **遗忘/更新**: 提供显式的结构化更新与遗忘原语：ADD（新增）、UPDATE（合并/修订已有条目）、DELETE（删除过时/错误条目）、NOOP（不操作）。这四种操作的选择由 RL 学习的 Memory Manager 策略决定，是论文的核心机制之一；相比启发式管线无差别 DELETE+ADD，学到的 UPDATE 能保留并整合跨会话证据。无 Ebbinghaus 式时间衰减，遗忘是「策略驱动的主动删除」而非自动衰减。
- **经验回放 (核心主题)**: 不属于「重放过去轨迹以指导未来动作」式的经验复用框架（与 Voyager/ExpeL/ReasoningBank 不同）。其「经验」体现在 RL 训练阶段：智能体在 152 条训练 QA 上反复 rollout 记忆操作/答案，由结果奖励（exact-match）驱动策略更新，从而把「如何管理与使用记忆」内化为参数；部署时不再重放训练轨迹，而是用学到的策略实时管理一个跨会话演化的记忆库。可视为「把记忆管理建模为序列决策过程并用 RL 学习最优存/取/更新策略」——这正是 2025-26「学习型记忆控制」范式的代表。

**学习维度 / Learning**

- **学习范式**: 参数化（parametric）/ 基于梯度的强化学习微调。两个智能体（Memory Manager 与 Answer Agent）均用结果驱动 RL（PPO 与 GRPO 两种实现）微调；为在稀疏奖励下保持稳定，二者分开训练（训练 Manager 时冻结 Answer Agent，反之亦然）。这与多数仅靠提示/外部存储累积的非参数记忆系统（Mem0、A-Mem、MemoryOS）形成对照——记忆内容仍外部存储，但控制策略被学进权重。
- **失败学习 (核心主题)**: 无专门的「失败轨迹反思/负样本入库」机制（区别于 Reflexion/ReasoningBank）。其对「错误记忆操作」的纠正是通过 RL 的结果奖励隐式实现：若某次 ADD/UPDATE/DELETE 导致下游 Answer Agent 答错（EM=0），该动作获得低优势值并被抑制，从而学会避免如「错误删除关键记忆」「无差别 DELETE+ADD 造成碎片化」等失败模式。GRPO 以组内相对优势 A_i=(r_i−mean(r))/std(r) 区分组内更优/更差动作，本质上是从同一状态的成败对比中学习。论文指出：由于 Manager 仅由下游 EM 奖励，存在潜在「奖励作弊/不忠实写入」风险（写出对答题方便但与原对话不符的内容），未加事实一致性约束。
- **技能/程序归纳**: 不显式归纳可复用的技能库/工作流（无 skill library，区别于 Voyager/AWM）。它归纳的是「记忆管理的程序性策略」本身——即把何时执行哪种 CRUD 操作、如何蒸馏记忆的决策内化为 RL 策略，可跨问题类型与跨数据集（零样本迁移到 MSC/LongMemEval）复用。
- **在线 vs 离线**: 训练为离线（offline）：在 LoCoMo 训练集（152 QA + 对应时间性记忆库）上用 RL 批量微调两个智能体。部署/推理为在线（online）：记忆库随多会话对话流增量构建与演化，但策略权重固定、不再在线更新。

**评测 / Evaluation**

- **任务领域**: 长程多会话对话记忆与问答（multi-session dialogue QA）。覆盖单跳（single-hop）、多跳（multi-hop）、开放域（open-domain）与时间推理（temporal）四类问题。纯文本对话场景，未涉及网页/具身/GUI/编码。
- **基准**: 三个基准：LoCoMo（主训练+评测；约 600 轮/26k tokens 的多会话对话，10 段对话，排除对抗子集，按 1:1:8 划分为 152/81/1307 题）、MSC（Multi-Session Chat，零样本迁移评测）、LongMemEval（零样本迁移评测）。评测指标：token 级 F1、BLEU-1（B1）、LLM-as-a-Judge（J，语义正确性）。
- **报告增益**: 仅用 152 条训练 QA 即在 LoCoMo 上达到 SOTA，并跨模型规模（Qwen-2.5 3B/7B/14B）与跨数据集稳健。【当前 v5 相对 MemoryOS（最强基线）】LLaMA-3.1-8B 上 Memory-R1-GRPO 相对提升 F1 +28.5%、B1 +34.0%、J +30.2%（绝对值约 F1=45.0、B1=37.5、J=62.7）；Memory-R1-PPO 相对提升 F1 +17.2%、B1 +17.6%、J +19.4%（绝对约 F1=41.0、B1=32.9、J=57.5）。Qwen-2.5-7B 上 GRPO 相对 MemoryOS 提升 F1 +24.5%、B1 +24.1%、J +20.0%。【早期版本相对 Mem0 的口径（仓库 README / emergentmind v2）】GRPO 绝对 F1=45.02、B1=37.51、J=62.74，相对 Mem0 提升 F1 +68.9%、B1 +48.3%、J +37.1%（README 概述为「+48% F1、+69% BLEU-1、+37% LLM-as-a-Judge」，系相对当时 prior best 的口径，与版本/基线选择有关，需注意基线差异）。零样本迁移：在 MSC、LongMemEval 上 PPO/GRPO 均一致超过基础模型。消融可见各组件贡献：去掉 RL Memory Manager 时 PPO 的 F1/B1/J 从 41.0/32.9/57.5 降到 34.5/28.1/49.0；记忆蒸馏使 GRPO 从 41.0/34.4/60.1 升到 45.0/37.5/62.7。
- **对比基线**: 无记忆/弱记忆与多种记忆系统：(1) LoCoMo（RAG 式分块检索基线）、(2) A-Mem（动态智能体记忆）、(3) Mem0（模块化记忆，沿用其 60 条检索设定）、(4) MemoryOS（记忆操作系统抽象，v5 的最强基线）、(5) Memory-SFT（同架构同数据但用 GPT-5 生成轨迹做行为克隆的监督微调变体，用于隔离 RL 的增益）。早期版本还对比过 Zep、LangMem。所有基线在 LLaMA-3.1-8B-Instruct 与 Qwen-2.5-7B-Instruct 上重新实现（temperature=0，max tokens=2048）以保证公平。

**分析 / Analysis**

- **关键创新**: 首次把「LLM 智能体的外部记忆管理与利用」建模为序列决策问题并用结果驱动 RL（PPO/GRPO）端到端学习：用一个 RL 微调的 Memory Manager 学习结构化 CRUD 操作（ADD/UPDATE/DELETE/NOOP），用一个 RL 微调的 Answer Agent 学习记忆蒸馏与推理；仅需 152 条 QA、仅用 exact-match 这种稀疏结果奖励（无中间人工标注）即可教会复杂的记忆巩固与利用行为并达到 SOTA。证明了「学习型记忆控制策略」显著优于静态启发式管线。
- **局限**: (1) 仅在对话型数据集上评测，扩展到多模态记忆存在挑战；(2) 为在稀疏奖励下稳定训练，Manager 与 Answer Agent 分开训练（非端到端），流程不够简洁，作者视端到端多智能体 RL 为未来方向；(3) Memory Manager 仅由下游答案 EM 奖励，存在「奖励作弊/不忠实写入」风险（缺乏对原对话事实一致性的显式约束、无记忆来源溯源/版本回滚）；(4) EM 奖励对改写/长答案脆弱并偏向短答案；MM 多步先于单次 QA 奖励，存在严重的延迟信用分配问题；(5) RL 训练计算成本高（4×H100）；检索器固定未学习、UPDATE/合并机制规格化不足、冲突消解未明确定义；(6) 仅 LoCoMo 单基准的 10 段对话、排除对抗子集，外部效度与对记忆投毒/提示注入的鲁棒性、隐私/遗忘合规均未评估；代码/checkpoint 尚未完整释出，复现性受限。
- **与其他工作关系**: 属于「G. 学习型/基于 RL 的记忆控制」簇，是该范式的旗舰代表。与同期 Mem-α（同样用 RL 学习记忆构建）、MemAgent（端到端 RL 长上下文记忆）、RMM、DeltaMem、MemSearcher 同源，均把记忆管理当作可学习策略，区别于启发式管线。它直接对照并超越用户中心/启发式记忆系统 Mem0（D4）、A-Mem（B4）、MemoryOS（B7）——这些系统的存/取/更新是固定流程，Memory-R1 则用 RL 学习该策略本身，是 2025-26「学习型 vs 启发式记忆控制」分水岭中的学习型一侧（与 ReasoningBank A6、ExpeL A5 等仍属启发式管线者相对）。其 CRUD 操作集与 Mem0/A-Mem 类似，但「何时用哪种操作」由 GRPO/PPO 学到。任务与评测对标 LoCoMo（F2）、LongMemEval（F1）、MSC 等长程对话记忆基准。后续 MemFactory（2026）将其抽象为标准模块（MemoryR1Agent）作为 RL 记忆训练的代表范式之一。
- **可复现性**: 中等偏弱。官方仓库 github.com/yansikuan/memory-r1（Apache-2.0，约 107 stars）截至调研日仍标「Code coming soon」、最近提交在 2025-09-10，完整训练/推理代码与 checkpoint 尚未释出；所用基准（LoCoMo/MSC/LongMemEval）与骨干（LLaMA-3.1-8B、Qwen-2.5 3B/7B/14B）均公开。论文给出较完整的超参（4×H100 80GB、batch 128、micro-batch 2/GPU、prompt 4096/response 2048、PPO 需 actor+critic、GRPO 仅 actor）、奖励设计与 prompt（附录 C/D），有助复现；但检索器/embedding 模型与索引设置未完全指定、依赖 LLM-as-a-Judge 评测，结果对实现细节敏感。社区关注度高（约 123 引用、被多篇后续 RL 记忆工作直接引用与复刻）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 是——本文是「学习型记忆控制」范式的核心代表。它明确用 RL（PPO 与 GRPO）训练记忆管理「策略本身」：Memory Manager 学习何时 ADD/UPDATE/DELETE/NOOP，Answer Agent 学习如何蒸馏与利用记忆，奖励为下游答案 exact-match。这与启发式/静态管线（Mem0、A-Mem、MemoryOS、ReasoningBank）形成 2025-26 代际分水岭的两端；论文正是为「现有管线缺乏决定存什么/更新什么/检索什么的学习机制」而提出。
- **记忆主体**: 用户/对话中心（user-centric）为主：记忆的是多会话对话中的用户事实与事件，服务于跨会话个性化问答，与 Mem0/Zep/LongMemEval 同类评测谱系。但其「自我改进」体现在记忆管理策略层（智能体学会更好地管理记忆），故兼具弱的 agent-centric 特征（学习的是智能体自身的记忆操作能力，而非记住自身解题经验）。
- **多智能体记忆**: 单一记忆库、但采用「双智能体分工」架构（Memory Manager 负责写/维护，Answer Agent 负责读/蒸馏/推理）；arXiv 归类含 cs.MA。两个智能体分开 RL 训练（一方训练时冻结另一方）。并非多个对等智能体之间的共享/路由记忆（区别于 G-Memory、MIRIX）；作者将端到端多智能体协同 RL 列为未来方向。
- **时序推理支持**: 评测层面覆盖时间推理（LoCoMo 含 temporal 问题类型，且训练用「时间性记忆库 temporal memory bank」），并报告在该类问题上的提升；但机制层面不显式建模时间有效性/事件时序/事实有效窗口（记忆条目无时间戳/有效区间/衰减），多次 UPDATE 后的时间一致性未做评估，被列为已知不足。
- **模态**: 纯文本（text-only）。仅处理对话文本记忆；多模态记忆被明确列为未来工作。
- **过度个性化/记忆安全风险**: 论文未处理过度个性化/有害·过时·侵入性记忆的安全维度，也未做隐私/遗忘合规（GDPR「被遗忘权」）评估。emergentmind 的知识缺口分析指出其对记忆投毒、提示注入、矛盾/恶意输入的鲁棒性，以及保留/脱敏/选择性遗忘策略均缺失；并存在 Memory Manager「写入对答题方便但不忠实于原对话」的奖励作弊隐患（无事实一致性约束、无来源溯源/回滚）。
- **冲突/矛盾处理**: 通过 RL 学到的 UPDATE 操作处理跨会话的互补/更新型信息：将语义重叠或演进的事实合并巩固进同一条记忆，而非像 vanilla 管线那样错误地 DELETE+ADD 造成碎片化——这是论文动机图（Figure 1）的核心卖点。但 UPDATE/Merge 的合并粒度、真正矛盾事实（非演进而是互斥）的消解规则未显式定义或评估，被列为不足。
- **token成本/延迟证据**: 提供专门的延迟分析（附录 G），按中位 p50 与尾部 p95 分别测三个组件（Memory Manager / Memory Search / Answer Agent）。总体：Memory-R1 不引入显著延迟开销，且 GRPO 变体常获得比 base 与 PPO 更低的尾部延迟（accuracy-latency 上呈 Pareto 改进而非权衡）。具体在 LLaMA-3.1-8B 上，Answer Agent 组件 GRPO 的 p50/p95 = 0.34s/0.67s，显著低于 base 的 0.65s/3.07s 与 PPO 的 0.91s/4.67s；Qwen-2.5-7B 上 GRPO 把 p95 降到 0.83s（base 1.06s、PPO 2.60s）。Memory Manager p50 在 LLaMA 上约 1.98–2.17s（p95 约 3.4–3.6s），Memory Search p50<0.35s、p95<0.65s。相比加 reranker 的管线，学习型记忆蒸馏在更低延迟下取得更高准确率。论文未直接报告输入 token 成本的百分比节省，效率证据以墙钟延迟为主。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)
- 代码链接 (`code_url`)


<a id="g2-mem-α-mem-alpha论文标题mem-α-learning-memory-construction-via-reinforcement-learning模型权重发布名-memalpha-4b基于-qwen3-4b-训练"></a>

### G2 Mem-α

*Mem-α (Mem-alpha)，论文标题《Mem-α: Learning Memory Construction via Reinforcement Learning》；模型权重发布名 Memalpha-4B（基于 Qwen3-4B 训练）。*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本于 2025-09-30 首次公开；同期投稿 ICLR 2026）。
- **作者/机构**: Yu Wang（王宇，UCSD/Anuttacon，第一作者与通讯）、Ryuichi Takanobu、Zhiqi Liang、Yuzhen Mao（Stanford）、Yuanzhe Hu、Julian McAuley（UCSD）、Xiaojian Wu。主要机构为 Anuttacon、加州大学圣地亚哥分校（UCSD）与斯坦福大学；工作于 Anuttacon 实习期间完成。
- **论文链接**: https://arxiv.org/abs/2509.25911
- **代码链接**: https://github.com/wangyu-ustc/Mem-alpha （官方实现，约 210 stars / 17 forks，Apache-2.0；模型 https://huggingface.co/YuWangX/Memalpha-4B，数据集 https://huggingface.co/datasets/YuWangX/Memalpha）。

**记忆分类 / Taxonomy**

- **记忆类型**: 组合式多类型记忆：核心记忆（Core，用户画像/偏好等浓缩摘要，属语义+程序性混合）、情景记忆（Episodic，带时间与情境的个人经历事件）、语义记忆（Semantic，事实/知识/概念/how-to 信息）。覆盖 CoALA 中的语义与情景记忆，核心记忆兼具工作记忆式的浓缩上下文功能。
- **记忆结构**: 分层多组件外部记忆：核心记忆为单段浓缩摘要（整体重写维护），情景记忆与语义记忆为可逐条增删改的结构化条目池（带 record id）。属于'分层分类型条目库'，不同于扁平 buffer 或知识图谱。
- **存储后端**: 外部记忆库（条目集合 + record id），检索阶段对情景/语义记忆使用 BM25 检索器从对应记忆池取 top-k 条目；不依赖向量库或图数据库，记忆内容以自然语言文本条目形式存储；策略模型为 Qwen3-4B（用 verl/GRPO 训练），评测 RAG 生成器固定为 Qwen3-32B。
- **持久化**: 外部持久化记忆（durable external store），跨整段交互历史持续存在并可更新；记忆构建策略本身被固化进模型参数（RL 训练后的 Qwen3-4B 权重），但被记忆的内容存于外部记忆库而非参数。属外部存储 + 参数化策略的组合。

**核心机制 / Mechanisms**

- **写入/编码**: 智能体逐块（chunk）处理顺序输入的对话/文档流，在每个 chunk 上自主决定发出一串写操作 a_t=(a_t^(1),...,a_t^(K_t))，每个操作属于 {memory_insert, memory_update, memory_delete} 的结构化函数调用（带参数：record id、记忆类型、文本内容）。核心记忆只支持 update（整体重写）；情景与语义记忆支持 insert/update/delete 三种细粒度操作。编码方式为模型自主抽取并改写为精炼自然语言条目，而非逐字保存——压缩奖励 r3=1-l_m/l_c 显式鼓励压缩，内容奖励 r4 用 Qwen3-32B 判定条目是否被写入正确的记忆类型且语义有效。关键点在于'写什么/写到哪类记忆/何时更新'由 RL 策略学得，而非预定义指令。
- **检索机制**: 采用解耦的 RAG 评测/读取流程，检索器与生成器均固定、仅写策略可学。处理完所有 chunk 得到终态记忆 M_n 后，对每个问题 q：(1) 检索——对语义记忆与情景记忆分别用固定 BM25 检索器 φ 选 top-k 条目（核心记忆通常全量纳入上下文）；(2) 生成——冻结的生成器 g（Qwen3-32B）基于 q 与检索到的支撑集生成答案；(3) 打分——与参考答案比较得到正确性指标，回灌为正确性奖励 r1。检索本身不被训练，性能提升来自上游记忆构建质量的改善。
- **反思/巩固**: 记忆构建过程本身即一种持续的反思/巩固：智能体并非逐字堆积，而是通过 insert/update/delete 不断把原始 chunk 抽象、归并、改写为三类高层记忆条目；核心记忆通过整体重写持续提炼用户画像类浓缩摘要。这种'抽象-归并'由四项奖励驱动学习（正确性 r1、工具调用格式 r2、压缩 r3、内容质量 r4 由 Qwen3-32B 作为 LLM-judge 评估条目是否归到正确记忆类型且非占位/空洞），而非靠固定的 reflection 触发规则；属于学习到的、随每个 chunk 触发的隐式巩固机制。
- **遗忘/更新**: 通过显式 memory_update（修改已有条目，核心记忆为整体重写）与 memory_delete（删除条目）实现编辑、合并与去冗；压缩奖励 r3 进一步压制冗余、鼓励紧凑表示。无 Ebbinghaus 式时间衰减；删除/更新由 RL 学得的策略按需触发。论文明确未覆盖 MemoryAgentBench 的 Conflict Resolution（冲突消解）维度（因现有该维度数据集多为合成、不够真实）。
- **经验回放 (核心主题)**: 本工作属用户/信息中心记忆，而非智能体自我经验回放范式：它不复用过去成功轨迹做技能复用或示例提示，而是把整段交互历史（对话/文档/分类样例/叙事）压缩进可被后续 QA 复用的结构化长期记忆。训练层面用 GRPO 采样多条记忆构建轨迹、以组内相对优势优化写策略，可视为对'记忆构建动作序列'的强化采样而非经典经验回放缓冲；记忆被'复用'的方式是检索增强生成在评测/部署时调用所构建的记忆。

**学习维度 / Learning**

- **学习范式**: 混合型，但核心是参数化学习：用强化学习（GRPO，基于 verl 框架）对记忆构建写策略做梯度更新，将一个相对弱的 Qwen3-4B 训练成强记忆智能体；记忆内容的存取本身是非参数（外部条目库 + BM25 检索）。即'参数化学习的记忆管理策略 + 非参数化的记忆存储/检索'。
- **失败学习 (核心主题)**: 无显式失败/负例记忆机制：不维护失败模式库、不做对失败轨迹的自反思、不存负例示例。失败信号以隐式方式体现在奖励中——若记忆构建不当导致下游 QA 答错，正确性奖励 r1 降低，GRPO 通过组内相对优势惩罚该类记忆构建动作，从而'试错'式地学到更好的存储/更新策略（论文强调 RL 相对 SFT 的优势正是无需 ground-truth 构建轨迹、可通过 trial-and-error 发现最优记忆策略）。属过程奖励驱动的间接错误学习，而非本研究中 ReasoningBank/Reflexion 式的显式失败学习。
- **技能/程序归纳**: 不诱导可复用的技能/工作流/过程脚本。核心记忆可保存用户偏好、目标乃至 how-to/规则类语义信息，但系统主要诱导的是'记忆管理策略'（何时 insert/update/delete、信息归到哪类记忆），而非面向任务执行的程序性技能库。可视为对'记忆操作过程'的程序性能力学习，但不属于经验技能归纳范式。
- **在线 vs 离线**: 两者结合但偏离线训练：记忆管理策略通过离线 RL 在 4139 实例（分层采样平衡后 562 实例）的训练语料上批量训练；训练完成后，记忆本身在部署/评测时在线、逐 chunk 顺序构建（每条新信息即时触发写操作）。训练长度上限 30k token，部署可在线泛化到 400k+。

**评测 / Evaluation**

- **任务领域**: 多轮对话记忆、长文档理解、问答（单跳/多跳 QA）、测试时学习（分类任务的 in-context 学习，如意图分类）、长篇叙事/书籍摘要理解。覆盖个性化对话记忆与长上下文信息保持；非具身、非 GUI、非网页导航；论文称架构可扩展到多模态但本文为纯文本评测。
- **基准**: 训练/同分布测试：自建 Memalpha 数据集（来源含 SQuAD、HotpotQA、PerLTQA、TREC-C、NLU(banking) 、PubMed-RCT、BookSum）。分布外测试：MemoryAgentBench（Hu et al. 2025, arXiv:2507.05257）的三大类——准确检索 AR（Single-Doc/Multi-Doc/LME(S)）、测试时学习 TTL（TREC-C/NLU/TREC-F/CLINIC/Banking77）、长程理解 LRU（InfBench-Sum）；未评 Conflict Resolution。最长上下文达 474K token（MemoryAgentBench Multi-Doc）。
- **报告增益**: 在自建 Memalpha 测试集（Table 1）：Mem-α 平均 0.642，显著高于 Long-Context 0.588、RAG-Top2 0.567、MemAgent 0.236、MEM1 0.111；同时记忆占用更省（平均 7.9K token vs Long-Context 10.8K / RAG-Top2 11.3K，约省 25-30%）。在分布外 MemoryAgentBench（Table 2，主配置 β=0.05,γ=0.1 的 Mem-α-4B）：平均 0.592，超过 RAG-Top2 0.502、Long-Context 0.461、MemAgent 0.198、MEM1 0.071；AR 与 LRU 维度提升尤为明显（如 Single-Doc 0.740 vs RAG-Top2 0.690、Long-Context 0.280；Multi-Doc 0.680 vs 0.450/0.270；InfBench-Sum 0.129 vs 0.065/0.125）。记忆占用约 129K token，远低于 RAG-Top2 的约 207K（约省 ~38%）。RL 增益消融（Table 3，验证集）：基座 Qwen3-4B + 同一记忆框架仅 0.389，RL 训练后 Mem-α 达 0.642，提升 +0.253（约 +65% 相对），且超过 gpt-4.1-mini 同框架结果，证明增益来自 RL 而非记忆结构本身。长度泛化：仅在 ≤30k（平均<20k）token 实例上训练，却能泛化到 400k+ token（最高 474k），超训练长度 13×以上。
- **对比基线**: Long-Context（Qwen3-32B，32k 上下文，超长则截取末 32k）、RAG-Top2（BM25 检索 top-2 chunk + Qwen3-32B）、gpt-4o-mini/gpt-4.1-mini（长上下文/同记忆框架）、MemAgent（RL-MemoryAgent-14B，扁平记忆改写）、MEM1（Qwen2.5-7B-RL-RAG，扁平单段记忆）；RL 增益对照含基座 Qwen3-4B + 本记忆框架、gpt-4.1-mini + 本记忆框架。即对照覆盖：无记忆/全上下文、RAG、专有大模型、以及同属 RL 学习记忆的扁平结构方法（MEM1/MemAgent）。

**分析 / Analysis**

- **关键创新**: 首个用强化学习（GRPO）训练智能体去管理'复杂多组件（核心/情景/语义）记忆架构'的框架：把记忆构建建模为序列决策问题，仅以下游 QA 正确性等四项奖励（r1 正确性 + r2 工具调用格式 + β·r3 压缩 + γ·r4 内容质量）作监督、无需 ground-truth 记忆构建轨迹，直接学得'存什么/归哪类/何时更新'的记忆管理策略。相对 MEM1/MemAgent/Memory-R1 等仅在扁平/简单记忆上做 RL 的前作，首次把 RL 扩展到富结构记忆并展现 13×+ 的极强长度泛化。
- **局限**: (1) 未覆盖冲突消解维度（因缺乏真实评测基准）；(2) RL 训练计算开销大（4 节点、需 GRPO 多轨迹采样，故训练样本压缩到 562 实例），训练长度受限于 30k token；(3) 检索仍用固定 BM25、生成器为冻结 Qwen3-32B，检索/生成未联合优化，性能受限于检索质量；(4) 评测为纯文本，多模态仅为设计可能性、未实证；(5) 内容奖励 r4 依赖 Qwen3-32B 作 LLM-judge，引入评判模型偏差与额外成本；(6) 记忆操作仅 insert/update/delete，无显式时间衰减/遗忘；(7) 基座局限于开源中等规模模型（Qwen3-4B/8B）。
- **与其他工作关系**: 属本研究 G 类（学习/RL 驱动的记忆控制）。与同类 RL 记忆控制方法对比：MEM1（Zhou 2025）、MemAgent（Yu 2025）训练简单纯文本记忆改写；Memory-R1（Yan 2025）、Learn-to-Memorize、REMEMBER 引入略丰富记忆与简化工具调用但局限于 LoCoMo（上下文<~26k）且训练/测试同分布——Mem-α 明确以这些为对照并主张其记忆结构更复杂、长度泛化更强。记忆架构（核心/情景/语义 + 多工具）直接借鉴并对标 MIRIX（Wang & Chen 2025）、MemGPT（Packer 2023）、Mem0（Chhikara 2025）等启发式 pipeline 系统，但区别在于用 RL 训练而非纯 prompt 驱动其工具使用。评测基准复用同组工作 MemoryAgentBench（Hu, Wang & McAuley 2025, arXiv:2507.05257）。GRPO 来自 DeepSeek（Shao 2024），训练基于 verl 框架。
- **可复现性**: 可复现性较好：官方开源代码（github.com/wangyu-ustc/Mem-alpha，Apache-2.0，约 210 stars）含训练/评测/基线脚本与数据处理流程；发布训练后模型权重 Memalpha-4B（HuggingFace YuWangX/Memalpha-4B）、训练数据集（YuWangX/Memalpha，约 62MB，Git LFS）及处理后的 MemoryAgentBench 评测集（YuWangX/Memalpha-Memoryagentbench）。提供主模型与多组消融（不同 β/γ）的训练脚本。需自行部署 Qwen3-32B（或 OpenRouter）作记忆服务/评测生成器；MemoryAgentBench 原始数据持续更新，作者建议用其处理版以保证复现一致。社区采用信号中等（新作，引用约 15）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 是——核心定位即'用 RL 学习记忆管理策略本身'。智能体通过 GRPO 学得何时及如何对核心/情景/语义记忆执行 insert/update/delete，奖励直接来自下游 QA 正确性（+格式/压缩/内容质量），而非依赖人工预定义的启发式 pipeline。是 2025-26 代际分水岭（与 Memory-R1、SkillOS、UI-Mem 等同列）的代表性 RL 记忆控制工作，且将 RL 控制扩展到复杂多组件记忆。
- **记忆主体**: 以用户/信息为中心（user/information-centric）：目标是记住用户对话、文档、知识、经历等长期信息以支撑后续问答与个性化（核心记忆=用户偏好/角色/目标，情景记忆=用户经历，语义记忆=知识）。不属于'记住自身经验以自我改进'的智能体中心范式（如 Voyager/ReasoningBank）。其'自我改进'仅体现在记忆构建策略经 RL 训练后变强，而非积累任务执行经验。
- **多智能体记忆**: 单智能体记忆系统，无多智能体共享/路由记忆设计（不同于 G-Memory、MIRIX 的多智能体分层/路由）。训练与部署均为单一记忆构建智能体维护单一记忆库；评测时记忆构建智能体（Qwen3-4B）与问答生成器（Qwen3-32B）解耦，但属同一记忆流水线的角色分工而非多智能体协作记忆。
- **时序推理支持**: 部分支持但非显式时序模型：情景记忆条目要求包含明确的时间信息（何时发生）与情境细节，内容奖励 r4 会校验其含时间维度；但系统不构建事实有效期窗口、事件日历或显式事件排序结构（不同于 Zep/Graphiti 的时间有效性建模）。时序更多以自然语言条目内容承载，而非结构化时间索引。
- **模态**: 纯文本（text-only）。论文称记忆架构可扩展到多模态信息，并将多模态列为未来方向，但本文实验与评测均为文本，无视觉/具身/视频记忆实现。
- **过度个性化/记忆安全风险**: 未涉及。论文未讨论有害/过期/侵入性/谄媚记忆的过度个性化风险，也无隐私治理或记忆安全机制；压缩奖励 r3 仅出于效率与紧凑性目的（防记忆膨胀），并非安全/防过度记忆设计。未在 OP-Bench、Causal-LoCoMo 等记忆安全/负向维度上评测。
- **冲突/矛盾处理**: 有限：提供 memory_update（修改/重写已有条目，核心记忆整体重写）与 memory_delete（删除）作为更新原语，理论上可在遇到新信息时覆盖旧条目，但论文明确未针对 MemoryAgentBench 的 Conflict Resolution（冲突消解）维度做训练或评测（因缺乏真实基准），故对矛盾事实的系统化消解能力未被验证，是区别于 Memory-R1（显式 UPDATE 冲突处理）/MEMTRACK 的弱项。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)
- token成本/延迟证据 (`token_cost_latency_evidence`)
- 发表venue (`venue`)


<a id="g3-mem-π-mem-pi全称-adaptive-memory-through-learning-when-and-what-to-generate将记忆建模为生成式策略-π_mem-而非检索库"></a>

### G3 Mem-π

*Mem-π (Mem-pi)；全称 Adaptive Memory through Learning When and What to Generate；将记忆建模为生成式策略 π_mem 而非检索库*


**基本信息 / Provenance**

- **年份**: 2026（arXiv 预印本，2026-05-20）
- **作者/机构**: Xiaoqiang Wang、Chao Wang、Hadi Nekoei、Christopher Pal、Alexandre Lacoste、Spandana Gella、Bang Liu（通讯）、Perouz Taslakian（通讯）；主要机构为 ServiceNow AI Research 与 Mila（魁北克 AI 研究所），并涉及蒙特利尔大学、蒙特利尔理工学院、麦吉尔大学、CIFAR AI Chair
- **发表venue**: arXiv 预印本（2026），截至调研时未见正式会议/期刊收录
- **论文链接**: https://arxiv.org/abs/2605.21463
- **引用数**: 约 0 次（Semantic Scholar 实时，截至 2026-06；属于刚发布的新论文）

**记忆分类 / Taxonomy**

- **记忆类型**: 程序性/语义性记忆为主：把可复用的任务流程与策略性提示（procedural hints）内化为模型参数；从经验中蒸馏的行为知识带有情景经验来源，但最终以参数化的程序性指导形式呈现，而非情景片段回放
- **记忆结构**: 参数化记忆（parametric weights）：记忆不是缓冲区/向量库/知识图谱，而是一个独立的生成式策略模型 π_mem（语言或视觉-语言模型）的参数；记忆条目（context-guidance 对）在离线阶段被蒸馏进权重，推理时按需生成文本提示
- **存储后端**: 模型参数（Qwen2.5-7B-Instruct 作为 π_mem 文本骨干；视觉变体用 Qwen2.5-VL-7B-Instruct）。离线经验库 E 仅用于训练（由 JEF-Hinter 从轨迹蒸馏的 hint 构成），推理时不再做外部检索
- **持久化**: 参数化（baked into weights）：经验持久化于 π_mem 的权重中，与下游 agent 模型参数分离；推理时生成的提示注入 agent 上下文属临时性（in-context）。记忆占用受模型规模上限约束，而非随经验数量线性增长

**核心机制 / Mechanisms**

- **写入/编码**: 两阶段蒸馏将经验写入参数。第一阶段「经验蒸馏」(experience distillation)：用监督学习把离线经验库 E（context-guidance 对 (x,m)，x=(任务 q, 观测 o)，m 为文本提示）压缩进 π_mem 参数，学习映射 (q,o)→m（自回归 SFT，受 context-supervised pretraining 启发）。经验库由 JEF-Hinter 从原始交互轨迹中识别「决定性步骤」并蒸馏为紧凑可复用 hint（每任务采样 5 条 hint），框架对来源不可知（可用人类示范/agent 轨迹/文档）。第二阶段「适应蒸馏」(adaptation distillation) 用 RL 以下游任务成功率为奖励进一步对齐生成行为。新增 [ABSTAIN]/[GENERATE] 两个特殊 token，用语义近义词嵌入均值对称初始化（≈50% 弃权概率冷启动）
- **检索机制**: 无外部相似度检索——这是核心区别。记忆作为「生成式策略」按需合成：以当前 agent 上下文（任务指令+环境观测，含可用截图）为条件，π_mem 联合决定「是否生成」(when) 与「生成什么」(what)，输出 y=d⊕m，其中 d∈{[ABSTAIN],[GENERATE]}。生成时把多条过去经验融合为一条上下文特定提示（不同于 top-k 检索可能割裂或截断信息），上限 L_max=256 token，再注入下游 agent 上下文。检索式基线（RAG/Mem0）用 BM25 取 top-k=1 作为对照
- **反思/巩固**: 经验抽象在「离线」完成而非在线反思：JEF-Hinter 先把长轨迹蒸馏为高层可复用 hint（程序化总结），再由经验蒸馏阶段把这些抽象写入参数。推理时 π_mem 进一步把参数中的多条经验「构造式」融合为当前情景的简明指导——借鉴认知科学「记忆是建构过程而非字面重放」(Bartlett/Schacter) 的观点。未实现持续在线巩固（作者将闭环记忆学习列为未来工作）
- **遗忘/更新**: 无显式遗忘/合并/去重管线——记忆容量由模型规模上限界定，从而规避检索式系统随经验增长的合并(merge)与遗忘(forgetting)管理开销；记忆更新通过对 π_mem 参数的再训练完成，而非条目级 ADD/UPDATE/DELETE。可视为用「参数覆盖+重训」隐式替代显式遗忘
- **经验回放 (核心主题)**: 把离线经验轨迹复用为可生成的程序化指导，而非检索回放。流程：收集 agent 轨迹→JEF-Hinter 蒸馏为 hint（识别决定性步骤）→经验蒸馏把 hint 内化进 π_mem 参数→RL 用下游任务成功率对齐→推理时按需生成上下文特定提示注入 agent。相比检索式技能库/情景库（返回静态条目），Mem-π 能融合多条经验产出单条贴合当前上下文的提示；token 效率更优（平均 138 token/任务，比 Stage1 的 200 少 31%、比 Memory-R1 的 225 少 38%）

**学习维度 / Learning**

- **学习范式**: 混合(hybrid)：以参数化学习为核心（两阶段——SFT 经验蒸馏 + GRPO 强化学习适应蒸馏，均更新 π_mem 梯度），但产物是注入 agent 上下文的非参数化文本提示；下游 agent 模型保持冻结。属「学习型记忆控制」范式（用 RL 学习记忆策略本身）
- **失败学习 (核心主题)**: 通过下游任务成功率(SR)作为 0/1 奖励间接吸收失败信号：RL 阶段用结构化反事实 rollout（1 条 [ABSTAIN] + G-1=3 条 [GENERATE] 分支）对比「生成 vs 弃权」的相对价值 Δ=V_abs−V_gen；当生成不如弃权（Δ>0）时决策优势惩罚生成、奖励弃权，从而学会在生成会有害/无益的情景（模糊、弱接地、分布外）主动弃权，防止把幻觉提示传给 agent。这是对「生成式记忆可能有害」失败模式的针对性防护。经验来源（JEF-Hinter）也可纳入成功轨迹的决定性步骤；论文未强调专门的负例/失败模式记忆库
- **技能/程序归纳**: 是：从轨迹中蒸馏出程序化、可复用的任务流程提示（如「先打开存储位置再放置物品」「用 Reports>Products>Bestsellers 并设日期过滤」），内化进 π_mem 参数，推理时按需生成并注入 agent，作为执行复杂任务的「how-to」线索
- **在线 vs 离线**: 记忆构建以离线为主（基于轨迹/经验库的批量两阶段蒸馏）；推理时是按需在线生成提示但不更新参数。作者将「持续在线收集经验并更新参数记忆」的闭环学习列为未来方向

**评测 / Evaluation**

- **任务领域**: 网页导航(web navigation)、终端工具使用(terminal/tool use)、文本化具身交互(text-based embodied)；其中网页导航含企业软件场景(ServiceNow/WorkArena)
- **基准**: WebArena（812 个多步浏览器任务，5 域：Shopping/CMS/GitLab/Reddit/Maps，647/165 训练/测试划分）、WorkArena（ServiceNow 企业网页导航，33 模板/4 类）、LifelongAgentBench/LAB（终端，DB 22 个 SQL 技能 + OS 29 个 Bash 技能）、ALFWorld（文本具身家务，3553 训练/134 未见测试）。评测用 BrowserGym 官方校验器、SQL 执行/OS 状态检查、环境终止条件，结果取 3 个随机种子均值
- **报告增益**: 以 gpt-5.4-mini 为基础 agent（表1，SR%）：四基准平均 45.3→55.4（Mem-π），即 +10.1 绝对点。WebArena 平均 27.1→43.1（+16.0 pp，相对约 +59%，论文称网页导航相对提升「超 30%」、WebArena「接近 50%」，与摘要不同版本表述一致）；最大子域增益 CMS 14.6→42.8（+28.2pp）、Reddit 28.8→52.6（+23.8pp）。WorkArena 42.0→50.3（+8.3pp，Form +14.9pp）；ALFWorld 78.8→86.7（+7.9pp，6 类家务）；LAB 26.8→36.7（+9.9pp）。优于检索式 RAG(47.4 avg)/Mem0(48.4) 与学习式 Memory-R1(49.2)/MemRL(50.0)。仅 Stage1（无 RL）即达 51.4 avg，RL 再加约 +4pp。跨 agent 迁移（表3）：在训练用 Qwen2.5-7B 上 WebArena +18.2pp(vs RAG +4.2)、ALFWorld +11.8pp；未见 GPT-5.4-mini 上 WebArena +16.0、ALFWorld +6.3，整体比 RAG 大 3–5 倍增益。Token 效率：平均 138 记忆 token/任务，较 Stage1(200) 省 31%、较 Memory-R1(225) 省 38%，同时 SR 更高。视觉变体在 WebArena 较纯文本再 +2.7pp（CMS +3.8、Shopping +3.3）
- **对比基线**: 无记忆基础 agent（Base Agent）；检索/工作流式记忆 RAG（BM25 top-k=1 检索 JEF-Hinter 库）与 Mem0（RAG+规则化管理）；学习式记忆 Memory-R1（RL 训练结构化记忆操作管理器）与 MemRL（学习记忆效用的 Q 值检索）。为公平起见记忆骨干统一用 Qwen2.5-7B-Instruct

**分析 / Analysis**

- **关键创新**: 把 agent 记忆从「检索静态条目」重构为「生成式策略」π_mem：用一个与下游 agent 参数分离的专用(V)LM，联合学习「何时生成(when)」与「生成什么(what)」，并提出「决策-内容解耦」(decision-content decoupled) 的 GRPO 强化学习目标——通过结构化反事实 rollout 把优势分解为跨分支决策优势 A_d 与分支内内容优势 A_c，并用 Δ-门控做 token 级信用分配（仅当生成优于弃权 Δ<0 时才更新内容 token），使记忆能在无益时主动弃权、有益时产出简明指导
- **局限**: 未公开官方代码（复现门槛）；记忆更新依赖离线重训、无真正的在线/增量遗忘与持续学习（作者列为未来工作）；增益随基础 agent 变强而收缩（接近能力上限的 GPT-5.4-mini 上 ALFWorld 仅 +6.3pp）；存在 Mem-π 回归区(Region 110)——少数任务上生成提示反而把 Base/RAG 能解的任务做坏、或把检索到的正确流程重排错误(Region 010)；生成提示缺乏可溯源/可归因性（作者列为未来方向 grounded & attributable memory）；用更弱 agent 训练虽提升可解释性但会使 Stage2 RL 奖励更稀疏；训练成本不低（8×H100）
- **与其他工作关系**: 属本研究 G 类（学习型/基于 RL 的记忆控制）。直接对标并超越 Memory-R1（用 RL 训练结构化记忆操作管理器）与 MemRL（学习效用感知检索的 Q 值）等同类学习式记忆——二者仍是检索中心(retrieval-centric)，只改进「何时/如何访问」而记忆内容在写入时即固定；Mem-π 改为生成中心，动态构造内容。与生成式记忆同期工作关系：ParamMem（把跨样本反思模式编码进参数）、MemGen（学习触发器+weaver 生成潜在记忆 token，ICLR 2026）、SEAM（用 GRPO 训练经验适配器为冻结执行器生成效用优化经验）、R3Mem（可逆上下文压缩）——但这些多把生成当作「检索模仿」或「常开辅助步骤」，Mem-π 的关键差异是显式学习「按需弃权」的自适应生成策略。经验来源工具 JEF-Hinter（Nekoei 等）用于蒸馏 hint；下游网页 agent 训练沿用 WebAgent-R1/WebRL 设置。与同期 AdaMEM、CoMem（解耦异步记忆模型）思路相近（解耦记忆与执行）但机制不同
- **可复现性**: 中等偏低：方法、训练超参（8×H100、GRPO G=4、L_max=256、lr、200 步等）与基准划分披露详尽，实现栈明确（PyTorch/HuggingFace/TRL/vLLM）；但未见官方开源代码与训练好的 π_mem 权重；所用基准(WebArena/WorkArena/LAB/ALFWorld)与 JEF-Hinter 多为公开资源。新论文(2026-05)社区采用信号尚少（引用≈0）

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 是——核心特征：用 RL（决策-内容解耦的 GRPO）学习记忆「策略」本身，即联合学习何时生成(when, 决策 token)与生成什么(what, 内容 token)，而非启发式管线；属 2025–26「学习型记忆控制」代际（与 Memory-R1、Mem-α 等同代但更进一步学到自适应弃权）
- **记忆主体**: Agent-centric（以 agent 自身经验自我改进为主）：记忆内容是从 agent 任务轨迹蒸馏的可复用程序化指导，用于提升 agent 在新任务上的执行能力，而非记忆用户信息做个性化
- **多智能体记忆**: 单 agent 设置；但架构上把「记忆模型 π_mem」与「下游执行 agent」解耦为两套独立参数，可即插即用地为更大/前沿 agent 提供指导（已验证从弱 agent 训练的记忆可迁移到未见的 GPT-5.4-mini）。未涉及多 agent 间共享/路由记忆
- **模态**: 文本为主，支持多模态：默认文本骨干 Qwen2.5-7B；视觉-语言变体 Qwen2.5-VL-7B 可接收网页初始截图与视觉接地信息（由 gemini-2.5-flash 抽取），在 WebArena 较纯文本再 +2.7pp。LAB/ALFWorld 为纯文本无视觉通道
- **过度个性化/记忆安全风险**: 正面应对生成式记忆的「有害/无益记忆」风险：通过自适应弃权机制，在模糊、弱接地、分布外或基础 agent 已能成功的情景主动不生成（弃权率随任务难度升高而下降：最易任务约 71%、最难约 13%），避免把幻觉/噪声提示传给 agent——体现「更多记忆未必更好」。案例区 Region 100/110 显示「任何记忆都有害」时弃权可恢复 Base agent 表现。未涉及隐私治理/用户数据安全维度（因属 agent-centric）
- **冲突/矛盾处理**: 无显式的事实冲突合并/失效机制（不存储可冲突的事实条目）。冲突在「生成 vs 检索/弃权」层面通过 RL 决策优势隐式处理：当检索提示与当前查询不符（如继承来源的「top-2」规格而查询要「top-3」）时，生成式记忆可改写以贴合当前上下文，或直接弃权恢复 base agent 的开放搜索（Region 001/101 案例）
- **token成本/延迟证据**: 量化 token 效率：Mem-π 平均仅用 138 记忆 token/任务，较 Stage1(200) 省 31%、较 Memory-R1(225) 省 38%，且 SR 最高（弃权时贡献 0 token）；同时获得最佳「性能-效率」权衡（Stage1 35.0%→Mem-π 43.1% on WebArena）。记忆容量由模型规模上限界定而非随经验线性增长。论文未报告端到端延迟数字（额外推理一个 7B 记忆模型会带来计算开销，但被弃权机制部分抵消）

**不确定字段 / Uncertain**

- 代码链接 (`code_url`)
- 时序推理支持 (`temporal_reasoning_support`)


<a id="g4-skillos全称-skillos-learning-skill-curation-for-self-evolving-agents一种用强化学习训练技能策展skill-curation策略的自进化智能体训练配方架构为冻结的-agent-executor--可训练的-skill-curator双模块外接一个可演化的技能仓库-skillrepo"></a>

### G4 SkillOS

*SkillOS（全称 SkillOS: Learning Skill Curation for Self-Evolving Agents；一种用强化学习训练「技能策展（skill curation）」策略的自进化智能体训练配方。架构为「冻结的 Agent Executor + 可训练的 Skill Curator」双模块，外接一个可演化的技能仓库 SkillRepo）*


**基本信息 / Provenance**

- **年份**: 2026（arXiv 预印本，v1 首次公开于 2026-05-07，cs.AI / cs.CL）
- **作者/机构**: Siru Ouyang（欧阳思蛞，第一作者，UIUC，通讯 siruo2@illinois.edu）、Jun Yan、Yanfei Chen、Rujun Han、Zifeng Wang、Bhavana Dalvi Mishra、Rui Meng、Chun-Liang Li、Yizhu Jiao、Kaiwen Zha、Maohao Shen、Vishy Tirumalashetty、George Lee、Jiawei Han（韩家炜）、Tomas Pfister、Chen-Yu Lee（通讯 junyann/chenyulee@google.com）共 16 位作者。主要单位：Google Cloud AI Research（多数作者）；合作单位：伊利诺伊大学厄巴纳-香槟分校（UIUC，Siru Ouyang、Yizhu Jiao、Jiawei Han）、麻省理工学院（MIT，Kaiwen Zha、Maohao Shen）。属工业界主导（Google Cloud AI Research）的学术研究成果。
- **发表venue**: arXiv 预印本（2026-05；cs.AI 主类，附 cs.CL）。截至调研日（2026-06-07）尚未见正式会议/期刊发表记录，归类为预印本。11 页、6 图、3 表。
- **论文链接**: https://arxiv.org/abs/2605.06614
- **引用数**: 约 3 次引用（Semantic Scholar，paperId 2653e4c7fa3be13d3db4b7a6ff3032ac17f7f467 / CorpusId 288014414，citationCount=3，referenceCount=62，isOpenAccess=false，截至 2026-06-07）。因论文发布仅约一个月，引用量低属正常，反映其为 2026 前沿新作。

**记忆分类 / Taxonomy**

- **记忆类型**: 程序性记忆（procedural memory）为主：以「可复用技能（reusable skills）」形式从过往交互中蒸馏并累积，作为自进化智能体的核心载体。每个技能本质是「何时用、怎么做」的可执行知识/工作流/启发式，随训练演化出更高层的「元技能（meta-skills）」。同时隐含情景/语义成分（技能内容由具体轨迹经验抽象而来），但论文明确将其定位在 CoALA 的程序性记忆范畴。
- **记忆结构**: 外部「技能仓库 SkillRepo」——由一组可复用技能 S_t={s_t^1,…,s_t^{N_t}} 组成的可编辑、可审计、可组合的集合（非向量库、非知识图谱）。沿用 Anthropic 的 SKILL.md 格式：每个技能是一个 Markdown 文件，含 (i) YAML frontmatter（技能名 + 何时使用的自然语言描述）与 (ii) Markdown 正文（可执行知识、工作流、约束、可复用启发式）。仓库通过类操作系统的文件 I/O 操作（insert/update/delete）增量演化；分析显示技能文件随训练发展出更丰富的内部结构（如失败处理逻辑、条件分支）与跨技能的元策略组织。
- **存储后端**: 外部文件型存储（Markdown 技能文件构成的 SkillRepo，类比操作系统的文件系统，故名 OS）。检索后端为 BM25 关键词稀疏检索（Lucene/Okapi BM25 设定）。技能控制「策略」则参数化固化进被 GRPO 微调的 Skill Curator 权重（Qwen3-8B）。论文未指定具体的文件/索引底层实现细节。
- **持久化**: 外部持久化存储（durable external store）：SkillRepo 跨流式任务持续累积、更新与删除，初始可为空（每个训练 group 从空仓库起步）。同时存在参数化成分——「如何策展（增/改/删/检索使用）」的策略被 RL 学进 Skill Curator 的权重；而 Agent Executor 全程冻结、不更新权重。即「技能内容=外部非参数存储 + 策展策略=参数化」，这是与纯启发式管线的关键分界。

**核心机制 / Mechanisms**

- **写入/编码**: 由可训练的 Skill Curator（策略 π_S，基于 Qwen3-8B）在 Executor 完成每个任务后执行写入。Curator 观察三项输入：执行轨迹 ξ_t（观察-动作序列）、对答案/交互正确性的自判信号 1_{ξ_t}（由冻结 Executor 作 LLM-as-a-judge 得到）、以及 BM25 检索到的相关技能子集 S̃_t；据此生成一串结构化策展操作 c_t=(u_t^1,…,u_t^{M_t})，每个操作 ∈ {insert_skill, update_skill, delete_skill}，以函数调用（tool call）形式作用于仓库，得 S_{t+1}=ApplyOps(S_t,c_t)。编码目标不是逐字复制轨迹，而是把经验蒸馏为简洁、可复用、Markdown 结构化的技能（由 compression reward 显式惩罚冗长复制、由 content-quality reward 鼓励语义有用）。分析（图 5）显示：训练早期 Curator 倾向插入泛化的「tips/建议」段落（冗长但价值低），随训练逐步转向「失败处理逻辑、条件分支」等可执行结构，并涌现跨任务元技能。
- **检索机制**: 采用固定（未学习）的 BM25 稀疏关键词检索：对每个新任务 x_t，从当前仓库 S_t 中召回相关技能子集 S̃_t⊆S_t（Algorithm 1 第 4 行 S̃←BM25(x_i,S)），注入冻结 Executor 的上下文，Executor 据此采样动作 a∼π_L(·|x_t,o_t,S̃_t)（多轮任务用 ReAct，推理任务用 CoT）。Curator 侧也用 BM25 召回相关技能作为策展上下文。论文有意保留简单检索器以隔离「策展」这一研究焦点，并在 Limitations 明确指出 dense/hybrid/learned retriever 可进一步提升，将「策展与检索的联合优化」留作未来工作。无 recency·importance·relevance 三因子打分、无图遍历/PPR、无 surprise 分段。检索的「读后使用」由 Executor 隐式承担（分析显示 SkillOS 每例使用更少但更精准的技能）。
- **反思/巩固**: 「巩固/反思」由 RL 学到的 update_skill 操作承担，并由 Curator 行为的训练演化体现。关键发现（图 4）：训练初期 insert 操作占绝对主导（快速填充仓库），随训练推进 update 逐渐增多、insert 下降，delete 始终是小比例但略增——说明 Curator 自发从「单纯扩充」转向「修订与合并已有技能」，即把跨任务经验巩固进现有技能而非无限新增。图 5 进一步显示技能文件从「泛化补充段落」演化为「可执行的失败处理/条件分支」，并在仓库层面从「狭窄任务专用技能」演化为「验证、回退规划、系统性搜索、策略调整」等可组合的跨任务元技能。这是「由结果奖励驱动的隐式反思/巩固」，而非 Generative Agents 式周期性显式反思总结。
- **遗忘/更新**: 提供显式的三种结构化操作原语：insert_skill（新增）、update_skill（修订/合并/巩固已有技能）、delete_skill（删除过时/有害技能），均以函数调用实现并作用于 SkillRepo。三者的选择由 RL 学习的 Curator 策略决定（非固定规则）。compression reward（奖励仓库相对输入上下文的简洁度）间接抑制冗余、鼓励删除/不囤积。无 Ebbinghaus 式时间衰减——「遗忘」是策略驱动的主动 delete，且分析显示 delete 占比在训练全程偏低（以 update 巩固为主导适应方式）。
- **经验回放 (核心主题)**: 核心主题——属「把流式经验蒸馏为可复用技能并在后续相关任务中复用」的经验复用范式（与 Voyager/ExpeL/ReasoningBank 同谱系，但用 RL 学策展）。机制为闭环流式工作流：对每个新任务，Executor 选取相关技能→用其指导执行→Curator 基于结果轨迹更新技能集；早先经验蒸馏出的技能由「能否帮助解决后续相关任务」来评判其价值。训练侧的创新在于「分组任务流（grouped task streams）」：依据技能相关的任务依赖把任务打包成 group，组内顺序求解——组内第一个任务用空仓库执行（无策展），其后任务的成功率作为对前序策展决策的延迟奖励信号，从而把「间接、延迟的监督」转化为可学习信号。每个训练 group 采样 N 条独立 rollout（整条策展序列），不同 rollout 演化出不同的仓库历史，用 GRPO 组内相对优势学习。部署时直接用学到的 Curator 实时维护 SkillRepo，不重放训练轨迹。

**学习维度 / Learning**

- **学习范式**: 混合（hybrid），偏参数化 RL：Skill Curator（Qwen3-8B）用基于梯度的强化学习（GRPO，去掉 KL 项以鼓励探索）端到端微调，学习「技能策展策略」；Agent Executor 全程冻结（非参数，仅在上下文层使用技能）。技能内容本身是非参数的外部 Markdown 存储。整体可概括为「冻结执行器 + 可训练策展器」的模块化解耦：无需重训底层执行器即可获得自进化能力。这与纯非参数提示/启发式记忆系统（Mem0、A-MEM、ReasoningBank）相对，也与端到端微调执行器的方法相对。
- **失败学习 (核心主题)**: 失败学习以两种方式体现，但非显式「负样本入库」。其一（训练层，隐式）：Curator 仅由下游 task outcome reward（后续任务平均成功率）等组合奖励驱动，若某次策展（如插入/修订出无用或误导性技能）导致后续 Executor 失败，则该策展序列在 GRPO 中获得低组内相对优势 A^n=r^n−mean(r) 而被抑制，从而学会规避有害策展；这正是论文相对「短视域」前作的卖点——能学到复杂的 update/delete 而不仅是 insert。其二（技能内容层，显式涌现）：分析（图 5、附录案例 17/19）显示训练后 Curator 在技能 Markdown 中主动写入「失败处理逻辑、条件分支、何时偏离默认工作流」，并蒸馏出失败恢复元策略（如「穷举搜索→确认不可得→寻找替代→用替代继续」）；ALFWorld 案例显示这类技能帮助 Executor 避免在错误容器上的低效搜索。但论文未设专门的失败轨迹反思模块或负样本记忆库。
- **技能/程序归纳**: 是——这是本文核心：显式从经验归纳可复用技能/工作流，并以 Anthropic SKILL.md（Markdown + YAML frontmatter）表示，由 Executor 经 BM25 检索后在上下文中调用。归纳过程不是手工或固定启发式，而是由 RL 学到的 Curator 通过 insert/update/delete 自动完成；训练中技能从狭窄任务专用过程演化为跨任务可组合的高层元技能（验证、回退规划、系统性搜索、策略调整等）。同时也归纳了「策展策略本身」这一程序性能力。
- **在线 vs 离线**: 两者兼有。训练为离线（offline）：在各基准训练集（ALFWorld 训练分割、WebShop 训练分割、以及从 DeepMath-103K 随机抽 33,000 条构造的推理数据）上，用 GRPO 在「分组任务流」上批量微调 Curator（16×H100，ALFWorld 约 3 天、推理约 2.5 天、WebShop 约 5 天）。部署/推理为在线（online）：SkillRepo 随测试任务流增量演化，但 Curator 策略权重固定、不再在线更新。

**评测 / Evaluation**

- **任务领域**: 两大类：(1) 多轮智能体/具身-交互任务——ALFWorld（文本化具身家务，对齐 ALFRED，6 个子类 Pick/Look/Clean/Heat/Cool/Pick2）与 WebShop（模拟在线购物网页导航与购买）；(2) 单轮推理任务——数学竞赛与研究生级科学问答。覆盖具身文本交互、网页购物、数学/科学推理；未涉及多会话对话记忆、GUI 截图、长文档 QA。
- **基准**: 智能体任务：ALFWorld（主表，6 子集共 140 例，报告成功率 SR↑ 与步数 Steps↓）、WebShop（报告 Score、SR↑、Steps↓）。推理任务：AIME24、AIME25、GPQA-Diamond（报告 exact-match 准确率 Acc↑ 及三者平均 Avg.Acc）。推理训练数据来自 DeepMath-103K（抽样 33k）。附加：附录 C.1 在 ALFWorld 上用更新的 Gemini-3.1-Flash-Lite 作执行器复验。AIME 用 HuggingFace Math-Verify 评测，GPQA 用选项字母精确匹配。
- **报告增益**: 总体：相对最强基线最高 +9.8% 相对成功率提升、−6.0% 更少交互步数；RL 训练的 8B Curator 甚至超过用 Gemini-2.5-Pro 直接做 Curator 的 SkillOS-gemini。【ALFWorld，平均 SR，Qwen3-8B 执行器】No Memory 47.9 → ReasoningBank 55.7（最强外部基线）→ MemP 49.7 → SkillOS-base 53.1 → SkillOS-gemini 50.7 → SkillOS 61.2；即 SkillOS 相对 ReasoningBank +5.5 绝对/约 +9.8% 相对，相对未训练的 SkillOS-base 约 +8.1（论文表述 +7.9）绝对；步数 18.9 vs No Memory 21.1（−2.2 步）。【ALFWorld，Qwen3-32B 执行器】SkillOS 68.6（步 17.3）vs ReasoningBank 61.4、No Memory 54.5。【ALFWorld，Gemini-2.5-Pro 执行器（训练未见）】SkillOS 80.2（步 14.8）vs No Memory 66.4（+13.8 绝对），相对 SkillOS-base 提升 +9.5（vs Qwen3-8B 上的 +7.9，说明与执行器能力复合放大）。论文亦称 3 个执行器上分别减少 2.2/3.0/3.1 步。【WebShop，Score / SR】Qwen3-8B：SkillOS 40.6/16.5（步 19.4）vs ReasoningBank 35.4/11.4、No Memory 33.3/9.8；Qwen3-32B：49.2/16.5 vs No Memory 41.5/12.2；Gemini-2.5-Pro：56.0/41.3 vs No Memory 48.6/38.4。【推理，Avg.Acc（AIME24/AIME25/GPQA）】Qwen3-8B：SkillOS 73.8（80.0/76.7/64.6）vs No Memory 69.6（76.0/71.1/61.8）；Qwen3-32B：79.7（85.6/81.1/72.4）vs No Memory 74.0；Gemini-2.5-Pro：88.6（92.2/86.7/86.8）vs No Memory 81.8。论文指出智能体任务增益普遍大于单轮推理任务（因前者过程性规律更易复用）。【Gemini-3.1-Flash-Lite 执行器，ALFWorld 平均 SR（附录 C.1）】SkillOS 73.1% vs ReasoningBank 66.0%（+7.1）、No Memory 61.2%（+11.9），步数 15.5 vs 18.5；其中 MemP（58.6%）甚至低于 No Memory，显示手工启发式策展在弱执行器下脆弱而学习型策略稳健。
- **对比基线**: 三类：(i) 无记忆智能体 No Memory；(ii) 现有记忆/技能方法——ReasoningBank（从过往经验蒸馏可复用洞见；本研究中最强外部基线）与 MemP（用先进记忆管理策略归纳程序性记忆）；(iii) 本框架内部变体——SkillOS-base（用未经 RL 训练的初始 Curator）与 SkillOS-gemini（用 Gemini-2.5-Pro 直接做策展、不学习 Curator）。所有方法在统一的冻结执行器（Qwen3-8B / Qwen3-32B / Gemini-2.5-Pro，附加 Gemini-3.1-Flash-Lite）下评测，3 次运行报告均值±标准差。

**分析 / Analysis**

- **关键创新**: 首次把「自进化智能体的技能策展（skill curation）」表述为一个长程、以执行器为根基（executor-grounded）的 RL 学习问题，并解耦为「冻结的 Agent Executor + 可训练的 Skill Curator」模块化双智能体框架——无需重训执行器即可学习「何时/如何 insert/update/delete 技能」的复杂长期策略。两项关键设计：(1)「分组任务流」训练实例构造——按技能相关任务依赖打包任务组，让早先策展由后续相关任务的成功率来评判，把延迟/间接反馈变为密集学习信号（解决前作仅在短任务流上只学会 insert、难学 update/delete 的问题）；(2) 复合奖励 r=r^task + λ_f·r^fc + λ_u·r^cnt + λ_c·r^comp（任务结果 + 函数调用有效性 + 技能内容质量[Qwen3-32B 评判] + 仓库压缩度，λ_f=1.0、λ_u=0.1、λ_c=0.05），更好地把下游执行反馈归因到策展决策。结果证明：一个经针对性 RL 训练的小（8B）策展器可胜过零样本使用前沿大模型（Gemini-2.5-Pro）做策展，即「策展是可学习且执行器对齐的能力，原始模型规模并非决定因素」。
- **局限**: (1) 检索器为简单 BM25 关键词检索且未学习（附录 D 明确承认），dense/hybrid/learned 检索可进一步提升，策展与检索的联合优化留作未来工作；(2) 训练计算成本高（16×H100，单任务 2.5–5 天），WebShop 尤甚；(3) 评测域较窄——具身文本交互、网页购物、数学/科学推理，未涉及多会话对话个性化、GUI 视觉、长文档、多模态；(4) 任务结果信号 1_{ξ_t} 由冻结执行器自判（LLM-as-a-judge），可能引入判定噪声/偏差；内容质量奖励依赖外部 Qwen3-32B 评判；(5) 推理任务增益明显小于智能体任务（可复用知识更抽象，难以直接落为动作过程）；(6) delete 操作占比偏低，真正的「有害/过时技能」清理与冲突消解规格化不足；安全/过度个性化/记忆投毒等维度未评估；(7) 无官方代码/checkpoint 释出（截至调研日），复现性受限。
- **与其他工作关系**: 属本研究「G. 学习型/基于 RL 的记忆控制」簇，与 G1 Memory-R1（RL 学习记忆 CRUD 操作）、Mem-α、CODESKILL、UI-Mem（C8）同代际——共同特征是「把记忆/技能管理策略本身用 RL 学到」，区别于启发式/静态管线。在技能/经验复用谱系上：它沿用并对照 ReasoningBank（A6，蒸馏可复用洞见，本文最强外部基线）、MemP（C10，程序性记忆 + 先进管理启发式，本文另一基线），并与 Voyager（C1，技能库自进化）、Agent Workflow Memory（C2）、ExpeL（A5）、AutoManual（C5）等技能/工作流归纳工作同源——但这些前作多为手工/提示/启发式策展，SkillOS 用 RL 端到端学策展。技能表示直接采用 Anthropic 的 SKILL.md（Markdown + YAML）设计，简化为单文件以便研究。相对前作（如仅在短任务流上训练记忆/技能操作的工作）的核心差异是「分组任务流 + 复合奖励」带来的长程、密集、执行器对齐的学习信号，使其能掌握 update/delete 等复杂操作而非只会 insert。与同名但无关的开源项目 ynulihao/AgentSkillOS（技能检索/编排 OS）须明确区分。
- **可复现性**: 中等偏弱。论文给出较完整的方法、算法（Algorithm 1）、奖励设计与超参（Qwen3-8B Curator，GRPO，lr=1e-6，batch=32，group size=8，16×H100，verl 框架；奖励权重 λ_f=1.0/λ_u=0.1/λ_c=0.05；ReAct + CoT；3 次运行均值±std；附录含 prompt、tool 签名、分组算法两阶段流程、超参表），所用基准（ALFWorld、WebShop、AIME24/25、GPQA、DeepMath-103K）与骨干（Qwen3-8B/32B、Gemini-2.5-Pro、Gemini-3.1-Flash-Lite）均公开。但截至调研日（2026-06-07）未发现官方开源代码仓库或 checkpoint（仅引用第三方 verl 与 Math-Verify），分组数据构造与权重需在 200 例 held-out 上人工调，且评测含 LLM-as-a-judge，结果对实现细节敏感；论文新（约 1 个月），社区复现/采用信号尚少（约 3 引用）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 是——本文是「学习型记忆/技能控制」范式的代表之一。明确用 RL（GRPO）训练「技能策展策略本身」：学习何时及如何 insert/update/delete 技能、以及如何组织演化 SkillRepo，奖励为下游任务成功率等复合信号。论文正是针对「现有方法依赖人工策展、固定启发式或只学短期/插入型操作，难以从间接延迟反馈中学到复杂长期策展策略」而提出，处于 2025-26「学习型 vs 启发式记忆控制」代际分水岭的学习型一侧（与 ReasoningBank、MemP 等启发式管线相对）。note 中点名 SkillOS 为该范式样例。
- **记忆主体**: 智能体中心（agent-centric）：记忆/技能是智能体从自身解题经验中蒸馏的可复用过程性知识，用于自我改进（self-evolution），与 ReasoningBank、Voyager 同类；而非记住用户信息做个性化（区别于 Mem0/Zep/LongMemEval 的 user-centric 谱系）。同时其学习目标兼含「策展能力」这一智能体自身元能力的提升。
- **多智能体记忆**: 单一共享技能仓库 SkillRepo，但采用「模块化双智能体」架构：Agent Executor（冻结，负责检索+应用技能解题）与 Skill Curator（可训练，负责基于轨迹更新仓库），二者通过 SkillRepo 与轨迹/正确性信号解耦协作。并非多个对等智能体之间的共享/路由记忆或洞见-查询-交互分层（区别于 G-Memory D5、MIRIX B9）。SkillRepo 可在不同执行器骨干间迁移共享（学到的 Curator 可配不同 Executor）。
- **时序推理支持**: 不显式建模时间有效性/事件时序/事实有效窗口（无时间戳、有效区间、事件日历——区别于 Zep/Graphiti D3）。其「时间性」仅体现在流式任务序列（task stream）的时间推进与「早先策展由后续任务评判」的时序信用分配上，属任务流层面的时序，而非记忆内容的时间推理。
- **模态**: 纯文本（text-only）。ALFWorld 为文本化具身环境，WebShop 为文本化网页接口，推理任务为文本数学/科学题；技能为 Markdown 文本文件。未涉及视觉/截图/视频/多模态记忆。
- **过度个性化/记忆安全风险**: 论文未处理过度个性化/有害·过时·侵入·谄媚记忆的安全维度，也未做隐私/遗忘合规评估。但与该主题相关的安全切面是「有害技能清理」：框架提供 delete_skill 原语且 compression reward 抑制囤积；附录 C.1 还观察到手工启发式策展（MemP）在弱执行器下会注入有害/误导技能反而低于 No Memory，而学习型 SkillOS 更稳健——侧面说明「更多技能并非总更好」、策展质量至关重要。但无显式的有害记忆检测/隐私治理/OP-Bench 类评测。
- **冲突/矛盾处理**: 通过 RL 学到的 update_skill 操作进行技能巩固/修订来隐式处理跨任务的互补/演进信息：分析显示训练后 Curator 以 update 为主导适应方式，把重叠或演化的经验合并进现有技能（而非无限新增或盲目删除），并写入条件分支/失败处理逻辑以覆盖不同情形。但论文未显式定义或评估「真正互斥/矛盾技能」的冲突消解规则与合并粒度（区别于 MEMTRACK、Memory-R1 的显式 UPDATE 冲突处理评测），属隐式、未单独度量。
- **token成本/延迟证据**: 以「效率」为第二评测维度，但量化口径是交互步数与 token 数，而非墙钟延迟或百分比成本节省。智能体任务用「每任务执行步数 Steps↓」：ALFWorld 上 SkillOS 相对 No Memory 在 3 个执行器分别减少 2.2/3.0/3.1 步（如 Qwen3-8B 18.9 vs 21.1；Gemini-2.5-Pro 14.8 vs 17.7），并低于所有记忆基线；WebShop 亦以更少步数取得更高成功率（如 Qwen3-8B 步 19.4，Gemini-2.5-Pro 步 18.3 优于多数基线）；Gemini-3.1-Flash-Lite 上 15.5 vs No Memory 18.5。推理任务用「每题 token 数」衡量效率（论文称更高效但正文未逐一列出 token 数值）。分析（图 6）显示 SkillOS 每例使用更少但更精准的技能，增益来自更精准的技能选择而非堆更多上下文。论文未给出如 Mem0/Zep 式的「−90% tokens / −90% latency」类百分比数字。

**不确定字段 / Uncertain**

- 代码链接 (`code_url`)


<a id="g5-codeskill全称-codeskill-learning-self-evolving-skills-for-coding-agents为编码智能体学习自演化技能"></a>

### G5 CODESKILL

*CODESKILL（全称 CODESKILL: Learning Self-Evolving Skills for Coding Agents；为编码智能体学习自演化技能）*


**基本信息 / Provenance**

- **年份**: 2026（arXiv 预印本，提交日期 2026-05-25）
- **作者/机构**: Yanzhou Li、Yiran Zhang、Xiaoyu Zhang、Yang Liu（南洋理工大学 NTU），Xiaoxia Liu（浙江大学）。通讯/主要单位为新加坡南洋理工大学。
- **发表venue**: arXiv 预印本（arXiv:2605.25430），截至调研时尚未见正式会议/期刊发表记录。
- **论文链接**: https://arxiv.org/abs/2605.25430
- **引用数**: 约 0 次（Semantic Scholar 实时查询，CorpusId 288670101；论文极新，2026-05 发布，尚未积累引用）。

**记忆分类 / Taxonomy**

- **记忆类型**: 程序性记忆（procedural memory）为主：将编码智能体轨迹蒸馏为可复用的程序性技能（skills）。技能分两种粒度——任务级技能（高层任务策略/工作流）与事件驱动技能（针对命令失败、报错、测试输出等局部执行事件的触发-响应规则），后者隐含了对失败/情景模式的情景性记忆成分。
- **记忆结构**: 结构化技能库（skill bank）B={s_i}。每个技能表示为一个 Markdown 指令文件，含标题 title、触发条件 when_to_apply、可执行规则 rules（以及粒度 granularity 字段）。技能库按两种粒度组织（任务级 general / 事件驱动 event-driven），并为两种粒度分别建立独立的稠密检索索引。
- **存储后端**: 外部稠密向量检索索引 + Markdown 技能文件。检索文档由 title+when_to_apply+rules 拼接，用 sentence-transformers/all-MiniLM-L6-v2(Sentence-BERT) 编码，按 benchmark 与技能类型分别建立稠密索引；技能本身以自然语言 Markdown 指令存储，检索后注入冻结编码智能体的提示词。
- **持久化**: 外部持久化技能库（durable external store），跨任务/跨回合持续累积与演化；下游编码策略 π 的参数始终冻结、不更新（记忆不写入下游模型权重）。管理策略模型 M_θ 的能力通过训练固化进自身参数，但管理的“技能记忆”本体是外部可持久化对象。

**核心机制 / Mechanisms**

- **写入/编码**: 由可学习的管理策略 M_θ（基于 Qwen3.5-4B）从规范化后的编码智能体轨迹（reasoning-action-observation 统一格式）中抽取可复用程序性技能。写入是一个被建模为 (a,z) 的操作：a 为操作类型（生成 generation/演化 evolution/维护 maintenance），z 为操作内容（生成的技能或维护决策）。抽取分两种粒度：任务级抽取从多条相关轨迹中抽象高层工作流；事件驱动抽取从单条轨迹中捕获局部执行模式（命令失败、报错、测试输出模式等）；当轨迹证据不支持可复用知识时输出 skip。技能不含仓库名/文件路径/函数名等一次性实例细节（写入即去任务特异化）。
- **检索机制**: 稠密语义相似度检索（Sentence-BERT all-MiniLM-L6-v2 编码 + 稠密索引，按 benchmark 与粒度分桶，过滤同实例技能以防泄漏）。任务级技能：查询由任务目标/问题陈述/仓库上下文构成，在任务求解前一次性检索并拼接到下游策略初始提示。事件驱动技能：查询在线由局部执行信号构造（当前任务上下文、近期 reasoning、已执行动作、观察、报错信息、命令输出、测试输出片段），按其触发条件匹配。训练侧还使用‘反向检索’ x_u~TopK(s_u, D_task)，用技能内容检索其适用的评测实例以计算执行奖励。
- **反思/巩固**: 通过技能抽取与‘技能演化（skill evolution）’实现原始经验→高层知识的转化。演化算子：输入为已有技能 + 新的或失败的轨迹证据，输出修订后的候选技能或 skip，用以更新技能的适用条件或程序性指导（补充缺失情形、更优流程、失败模式）。整个流程不依赖固定提示/启发式规则，而是由 RL 训练出的管理策略决定‘抽取什么、抽象到何种程度、如何修订’。维护阶段进一步把候选与库内相似技能合并（merge）以巩固互补知识。每个评测实例平均产生约 1 条任务级 + 3 条事件驱动候选技能再进入维护。
- **遗忘/更新**: 通过‘技能库维护（maintenance）’显式管理遗忘/更新：对每个新抽取或演化的候选技能，先检索库内相似技能，再由 M_θ 输出 add（新增）/merge（与一条已有技能合并去重）/drop（丢弃冗余、弱证据、过于局部或不可迁移者）三选一操作。drop 即等同遗忘/淘汰；merge 实现去重合并；evolution 实现就地更新。实验证明维护能把技能库从 1252 条压缩到 676 条（约减半）而仅损失约 2% 平均通过率，使库规模趋于稳定而非无界增长。
- **经验回放 (核心主题)**: 核心主题：将过去编码轨迹蒸馏为可复用程序性技能，作为先验知识在未来任务中‘技能复用（skill reuse）’而非逐字回放原始轨迹。复用方式：检索到的任务级技能注入初始提示提供整体策略，事件驱动技能在执行中按局部信号触发提供反应式指导。技能在长时程 SWE 任务中被检索→注入→遵循，从而提升下游冻结智能体的成功率与效率（平均推理步数从 44.12 降至 35.15）。训练时还用反向检索把生成的技能映射到适用实例，运行下游 rollout 度量增益（一种以执行结果为信号的经验复用闭环）。

**学习维度 / Learning**

- **学习范式**: 混合（hybrid）。下游编码策略 π 始终非参数化复用（冻结、仅靠提示注入技能，属 in-context/prompt 级）；但 CODESKILL 的创新在于对‘记忆管理策略 M_θ’本身进行参数化梯度训练：先用教师生成数据做 SFT 预热（LoRA 微调 Qwen3.5-4B-Instruct），再用 GRPO 强化学习优化。即记忆‘内容’非参数注入，记忆‘管理策略’参数化学习。
- **失败学习 (核心主题)**: 核心主题：显式利用失败经验。事件驱动技能专门捕获命令失败、报错信息、测试失败模式、特定动作后的重复失败模式，并规定触发时智能体应如何反应。技能演化的训练数据通过把已有技能与‘失败或部分成功’的相关轨迹配对来合成，使更新暴露缺失条件、更优流程与失败模式；‘失败响应性（failure_responsiveness）’是演化质量评判 rubric 的明确维度（新证据是否揭示失败/限制/矛盾/缺失情形，更新是否直接应对并改变未来决策）。执行奖励 R_E 基于相对无技能基线的验证器分数改进，隐含从失败 rollout 中学习。
- **技能/程序归纳**: 是，且为系统核心。从轨迹归纳可复用程序性技能/工作流，分两种粒度：任务级（多步工作流模式，如仓库探查、问题定位、修复验证）与事件驱动（单一局部触发-响应规则）。技能表示为 Markdown（title + when_to_apply + rules + granularity），通过稠密检索按相关性被调用并注入冻结编码策略提示词。
- **在线 vs 离线**: 两者兼具。离线：用 SWE-Bench Verified、SWE-smith、EnvBench 的批量轨迹做 SFT 预热与 GRPO 强化训练，得到管理策略 M_θ。在线：评测时 CODESKILL 随评测数据流在线维护技能库（逐实例收集无技能基线 rollout、抽取候选、执行 add/merge/drop 维护），实现部署期持续构建。

**评测 / Evaluation**

- **任务领域**: 软件工程/编码智能体领域：环境搭建与依赖修复（EnvBench，含 Python 与 Java/JVM 仓库）、仓库级 GitHub issue 解决（SWE-Bench Verified）、基于终端命令行的问题求解（Terminal-Bench 2，作为分布外泛化测试）。所有智能体均通过 bash 命令动作与环境交互。
- **基准**: EnvBench（环境搭建，994 仓库中抽 150 评测：107 JVM + 43 Python）、SWE-Bench Verified（500 中抽 150 评测、350 训练）、Terminal-Bench 2（完全留作分布外评测，不用于训练）。下游冻结编码策略：Qwen3.5-35B-A3B 与 GPT-5.4-mini；管理策略主干 Qwen3.5-4B。
- **报告增益**: 在 Qwen3.5-35B-A3B 冻结策略下，平均任务通过率从无技能基线 29.57 提升到 39.26（+9.69，相对约 +33%），比最强提示式技能管理/记忆基线（GPT-5.4-mini 提示式 35.25）高 +4.01（相对约 +11.4%）。分项（Succ.↑）：EnvBench-Python 6.98→18.60；EnvBench-Java 27.10→38.32；SWE-Bench Verified 57.33→66.00；Terminal-Bench 2(OOD) 25.88→34.12。效率：平均推理步数 44.12→35.15（最低）。换用 GPT-5.4-mini 作冻结策略时仍最优：平均较无技能 +8.93（相对 +41.0%），较最强基线 +2.87（相对 +10.3%）。消融：抽取+演化达 40.75 平均；加全周期维护降约 2% 但技能库 1252→676 条。RL 阶段-3 执行奖励 20 步均值从 0.004 升至 0.158。
- **对比基线**: 无技能基线（no-skill，下游策略不加任何先验知识）；提示式技能管理（Prompt Skill Mgmt.，与 CODESKILL 同操作空间但用固定提示+启发式决策，分别用 Qwen3.5-4B 与 GPT-5.4-mini 主干，是对 SkillRL、AutoSkill 等的 SWE 适配复现）；子任务级记忆（Subtask Memory，对 Shen et al. 2026 推理导向子任务记忆的高层复现，GPT-5.4-mini 主干）；以及消融对照 SFT-CODESKILL（仅监督预热、无 RL）。

**分析 / Analysis**

- **关键创新**: 首次把编码智能体的‘技能抽取 + 技能库维护’本身重构为一个可学习的管理策略（learnable management policy），并用强化学习（GRPO）+ 混合奖励（稠密 rubric 式 LLM-as-judge 技能质量奖励 R_Q，结合来自冻结下游智能体的稀疏可验证执行反馈 R_E，并用对齐因子 R_A 改进信用分配）来优化它，取代以往依赖固定提示和启发式更新规则的做法；同时引入多粒度技能（任务级/事件驱动）与三阶段课程训练。
- **局限**: 1) 仅支持自然语言指令型技能，不含可执行脚本/API/工具定义等结构化资源，限制了技能库表达力（在可扩展工具集的场景下指导力受限）。2) 动作空间受限为‘一次对一个候选技能执行一个操作’，无法直接表达更复杂维护（如联合修订多个相关技能、拆分过宽技能、单步协调多个 add/merge/drop）。3) 全周期维护会带来约 2% 平均通过率下降（以压缩库规模换取）。4) 环境奖励噪声大、稀疏（依赖长时程下游 rollout）。5) 训练依赖 GPT-5.4-mini 作教师与裁判，成本/可复现性存疑；6) 未见官方开源代码。
- **与其他工作关系**: 属本研究 G 类（学习型/RL 驱动的记忆控制）。与 Memory-R1、Mem-α、SkillOS、UI-Mem 同属 2025-26 ‘学习记忆管理策略’这一代际分水岭，但聚焦编码/SWE 领域且专注‘技能管理本身’而非求解策略。区别于 Voyager（Wang 2024，技能库但启发式构建）、Reflexion（Shinn 2023，言语反思）、ExpeL（Zhao 2024）、ReasoningBank（Ouyang 2026，推理记忆）、Agent Workflow Memory（Wang 2025b）、MemP（Fang 2025，程序性记忆）等以检索/总结或固定规则构建记忆的方法——CODESKILL 用下游反馈学习抽取/演化/维护。区别于 SkillRL（Xia 2026）、AutoSkill（Yang 2026）等‘训练求解器用技能’或‘技能与求解器协同进化’的工作——其冻结下游策略、只学技能管理。基线采用了对 SkillRL/AutoSkill 的提示式适配复现，以及子任务级记忆（Shen 2026）。
- **可复现性**: 中等偏低。论文给出较完整的算法（GRPO + 混合奖励公式）、训练超参（LoRA、SFT 2 epoch、lr 1e-4 cosine；GRPO 500 步分 130/120/250 三阶段、组大小 6、质量奖励权重 λ=0.25、KL 系数 0.02、rollout 温度 0.7、batch 4）、检索实现（all-MiniLM-L6-v2）、benchmark 划分与全套提示/rubric（附录 B/D/E）。但截至调研未见 CODESKILL 自身的官方开源代码与技能库发布；子任务记忆基线为自行高层复现（原实现不公开）；强依赖闭源 GPT-5.4-mini 作教师/裁判，复现门槛较高。论文极新（2026-05），暂无社区采用信号（0 引用）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 是——这正是其核心定位与卖点。明确把‘记忆/技能管理策略’（何时/如何抽取、抽象到何粒度、如何 add/merge/drop 维护）从启发式流水线改造为用 RL（GRPO）训练的可学习策略 M_θ，是 fields.yaml 所列 2025-26 ‘学习型记忆控制’代际（与 Memory-R1、Mem-α、SkillOS、UI-Mem 同类）在编码智能体领域的代表。
- **记忆主体**: 智能体中心（agent-centric）。记忆的是智能体自身在 SWE 任务中的程序性经验（如何探查仓库、修复 issue、应对命令失败/报错）以自我改进，而非记忆用户个性化信息。与 Voyager/ReasoningBank 同属 agent-centric 自我演化谱系，区别于 Mem0/Zep/LongMemEval 等 user-centric 个性化记忆。
- **多智能体记忆**: 单智能体设置为主：技能库服务于单一冻结下游编码策略 π。但技能被设计为去实例特异化、可跨任务/跨不同下游模型迁移（实验证明从 Qwen3.5-35B-A3B 训练得到的管理策略迁移到 GPT-5.4-mini 仍有效），具备一定跨策略复用性；不涉及 G-Memory/MIRIX 式的多智能体共享/路由记忆分层。
- **模态**: 纯文本（text-only）。技能为自然语言 Markdown 指令，轨迹规范化为文本化的 reasoning-action-observation 序列；不涉及视觉/截图/视频等多模态记忆。
- **冲突/矛盾处理**: 通过 merge 与 evolution 处理冲突/重叠：merge 把候选与一条已有相似技能合并为更强的单一技能（去重、整合互补规则、收紧适用条件、避免‘过宽伞型’技能），merge 裁判 rubric 专门检查 target_choice_correct、overlap_substantive、no_critical_loss 等；evolution 在新证据揭示矛盾/失败时就地修订规则与适用条件。属于‘合并/修订式’冲突处理，而非简单覆盖。
- **token成本/延迟证据**: 以‘平均推理步数’作为效率指标而非 token/延迟直接计量：在 Qwen3.5-35B-A3B 下，CODESKILL 把已解实例平均推理步数从无技能基线 44.12 降至 35.15（最低，约 -20%），表明检索到的程序性技能能让智能体更直接地达成解。维护机制把技能库从 1252 压缩到 676 条以控制存储/检索开销。论文未报告具体 token 数或 p95 延迟节省（无 Mem0/Zep/MemMachine 式百分比 token/延迟数据）。

**不确定字段 / Uncertain**

- 代码链接 (`code_url`)
- 过度个性化/记忆安全风险 (`over_personalization_risk`)
- 时序推理支持 (`temporal_reasoning_support`)


## H. 综述 (Surveys)


<a id="h1-a-survey-on-the-memory-mechanism-of-large-language-model-based-agents基于大语言模型智能体的记忆机制综述这是一篇综述survey而非具体系统方法别名配套资源llm_agent_memory_survey官方-github-论文清单仓库约-495-stars被公认为该领域最早最权威的系统性综述之一提出记忆来源sources记忆形式forms记忆操作operations设计三维度--直接评估direct间接评估indirect评估二分法"></a>

### H1 《A Survey on the Memory Mechanism of Large Language Model based Agents》

*《A Survey on the Memory Mechanism of Large Language Model based Agents》（基于大语言模型智能体的记忆机制综述）；这是一篇综述（survey），而非具体系统/方法；别名/配套资源：LLM_Agent_Memory_Survey（官方 GitHub 论文清单仓库，约 495 stars）。被公认为该领域最早、最权威的系统性综述之一，提出『记忆来源(sources)/记忆形式(forms)/记忆操作(operations)』设计三维度 + 『直接评估(direct)/间接评估(indirect)』评估二分法。*


**基本信息 / Provenance**

- **年份**: 2024（arXiv 预印本 v1 2024-04-21）；后经期刊扩充版接收并发表于 ACM TOIS（2025-07-02 录用，正式刊出标题改为带连字符的 LLM-based Agents）
- **作者/机构**: Zeyu Zhang（张泽宇，第一作者）、Xiaohe Bo、Chen Ma、Rui Li、Xu Chen（陈旭，通讯作者）、Ji-Rong Wen（文继荣），来自中国人民大学高瓴人工智能学院（Gaoling School of Artificial Intelligence, Renmin University of China）；Quanyu Dai、Jieming Zhu、Zhenhua Dong 来自华为诺亚方舟实验室（Huawei Noah's Ark Lab）。联系邮箱 zeyuzhang@ruc.edu.cn / xu.chen@ruc.edu.cn。
- **发表venue**: ACM Transactions on Information Systems (ACM TOIS)，期刊，2025 年录用（DOI: 10.1145/3748302）；arXiv 预印本属 cs.AI（2024-04，CC BY 4.0）。论文类型为综述（Survey/Review）。Semantic Scholar 标注 venue=ACM Transactions on Information Systems，year=2024（以 arXiv 计）。
- **论文链接**: https://arxiv.org/abs/2404.13501 （arXiv 摘要页，DOI: https://doi.org/10.48550/arXiv.2404.13501）；期刊正式版 https://dl.acm.org/doi/10.1145/3748302
- **代码链接**: https://github.com/nuster1128/LLM_Agent_Memory_Survey （官方配套论文清单仓库，约 495 stars / 12 forks，维护者为第一作者 Zeyu Zhang 与通讯作者 Xu Chen；内容为按综述分类整理的相关论文清单与示意图，而非可运行记忆系统实现——本条目是综述，无独立算法代码）
- **引用数**: 约 568 次（Semantic Scholar 实时数据，2026-06 核实，CorpusId b6ab16c8eade03a39830493071d99fc48a736fac）。作为该领域最早、被引最高的系统性记忆综述之一，影响力很高，被大量后续记忆系统与综述引用为标准分类学参照。

**记忆分类 / Taxonomy**

- **记忆类型**: 本综述不直接采用 CoALA 的 episodic/semantic/procedural/working 四分类，而是提出自有分类透镜。在『是什么(what)』部分给出窄义(narrow)与广义(broad)两种记忆定义：窄义记忆仅指『单次交互试验内(inside-trial)』的历史信息（即当前任务上下文）；广义记忆进一步纳入『跨试验(cross-trial)』的历史信息与『外部知识(external knowledge)』。从认知心理学视角对应人类的短期记忆（短时/上下文窗口）与长期记忆（外部存储/参数）。综述将各类记忆映射到智能体的经验积累、环境探索与知识抽象三种功能。
- **记忆结构**: 综述以『记忆形式(memory forms)』维度统揽底层结构，分两大类：(1) 文本形式(textual memory)——以自然语言存储，可细分为存于上下文窗口内(within the context window)的短期记忆与存于外部存储(external storage，如数据库、文本日志、向量库)的长期记忆；(2) 参数形式(parametric memory)——将信息编码进模型参数（通过知识编辑/微调）。综述未规定单一数据结构，而是横向归纳被调研系统所用的原始缓冲、键值存储、向量数据库、文本日志等多种结构。
- **存储后端**: 作为综述横向覆盖多种后端而非自身实现：文本记忆的外部存储后端包括向量数据库(vector DB)/键值存储(key-value store)/文本日志(text log)/普通数据库，配合检索器(向量检索、关键词匹配)读取；上下文窗口内记忆为 in-context（易失）；参数记忆后端为模型权重本身（经知识编辑或微调写入）。综述以 Reflexion、MemoryBank、RET-LLM、Generative Agents 等具体系统举例说明各类后端的典型实现。
- **持久化**: 综述显式区分三种持久化层次：(1) 短期/上下文记忆——驻留于 LLM 上下文窗口，易失、随会话结束消失；(2) 长期外部记忆——持久写入外部存储（数据库/向量库/文件），可跨试验、跨会话读写；(3) 参数化记忆——通过知识编辑或微调baked进模型权重，最持久但更新成本高且有灾难性遗忘风险。综述指出当前系统以文本型外部记忆为主流，参数记忆因成本与遗忘问题应用较少。

**核心机制 / Mechanisms**

- **写入/编码**: 综述在『记忆操作(memory operations)』维度将『写入(memory writing)』作为独立操作系统化归纳：原始经验（来自环境观察、用户反馈、智能体自身思考三类来源）经写入器(Memory Writer)编码进存储，编码方式横跨被调研系统的多种范式——逐字保存交互记录/轨迹(verbatim log)、用 LLM 进行总结后存自然语言洞见(LLM summarization)、抽取结构化事实、生成嵌入向量、或通过知识编辑/微调写入模型参数。综述强调记忆写入需处理两个子问题：记忆重复(memory duplicated，相似信息的整合)与记忆溢出(memory overflow，存储满时如何取舍)，并归纳出相应的合并(merge)与覆盖(overwrite)策略。综述将『写什么/如何写』与来源(sources)、形式(forms)三维度交叉刻画各系统设计。
- **检索机制**: 综述将『读取(memory reading)』(亦即检索)作为核心记忆操作之一系统归纳：由读取器(Memory Reader)从存储中取回与当前决策相关的信息送入 LLM 上下文。综述总结被调研系统常用的检索打分维度为三类信号——近因性(recency)、相关性(relevance)、重要性(importance)（典型如 Generative Agents 用三者加权组合），实现手段包括向量相似度检索(vector search/dense retrieval)、关键词/稀疏匹配(keyword match)等。综述指出检索质量是记忆系统效果的关键，并把『检索不准确、易被近因性偏置(recency bias)』列为现有工作的主要局限。
- **反思/巩固**: 综述在记忆操作中专设『记忆管理(memory management/管理性操作)』来刻画原始经验→高层知识的转化，涵盖：反思(reflection)、总结/摘要(summarization)、抽象(abstraction)、合并去重(merging)等。综述以 Reflexion（对失败试验自我批判生成改进计划）与 Generative Agents（周期性反思综合高层洞见）为典型范例，归类为『管理操作中的反思』。综述指出知识抽象(knowledge abstraction)——从原始观察总结高层信息——是智能体得以适应、泛化到未见环境的基础，并把它列为记忆相对于无记忆 LLM 的关键增益来源之一。
- **遗忘/更新**: 综述将遗忘/更新纳入『记忆管理』操作并指出现状不足：归纳出的处理手段包括基于近因/重要性的记忆淘汰与合并去重(应对 memory duplicated 与 memory overflow)，并提及借鉴艾宾浩斯遗忘曲线式的衰减机制(如 MemoryBank)。综述将『更好的遗忘机制以管理信息过载』列为重要未来方向，认为现有系统的遗忘/更新仍较粗糙、缺乏标准化处理。
- **经验回放 (核心主题)**: 作为该研究 H 类（综述）核心条目，本综述把『经验复用』确立为记忆存在的根本理由之一——在『为何需要记忆(why)』的自我演化(self-evolution)视角下，明确把记忆的首要功能定义为『经验积累(experience accumulation)：记住过去的错误规划、不当行为或失败经历，从而在未来相似任务上更高效』。综述横向归纳被调研系统的复用形态：非参数式（把过往轨迹/洞见检索回上下文作为后续决策依据，如 Reflexion 的跨试验反思、Generative Agents 的事件检索）与参数式（把经验微调进权重）。综述将『经验积累 / 环境探索 / 知识抽象』三者列为记忆驱动自我演化的三大机制，构成本研究『经验复用』主题的概念框架来源。

**学习维度 / Learning**

- **学习范式**: 综述同时覆盖两类范式并以记忆形式区分：非参数(non-parametric)——把经验以自然语言存于上下文或外部存储，靠 in-context 学习与检索复用（主流）；参数(parametric)——通过知识编辑/微调把信息写入模型权重。综述指出参数记忆能缓解文本记忆的检索低效问题、提供更强知识管理能力，但因训练成本与灾难性遗忘风险而应用较少，并呼吁未来深入研究参数记忆。属对两种范式的系统综述（hybrid coverage）。
- **失败学习 (核心主题)**: 综述把失败学习作为记忆驱动自我演化的核心论据：在自我演化视角下首要列出『记住过去的错误规划、不当行为或失败经历(past error plannings, inappropriate behaviors, or failed experiences)以在未来相似任务上更有效』，并指出这对提升智能体在自我演化过程中的学习效率极为重要。综述将 Reflexion 作为失败学习的代表性记忆操作实例（智能体对自身过往行动进行批判式反思，生成改进后的后续试验计划），归类为『跨试验信息 + 反思管理操作』；但综述本身不提出新的失败检测算法，而是对该类机制做分类归纳。
- **技能/程序归纳**: 综述在记忆来源/形式层面涵盖技能与流程类记忆（如开放世界游戏中积累的可复用技能、代码生成场景中的经验），并在应用部分专门讨论开放世界游戏(open-world games)与代码生成(code generation)等需要程序性/技能记忆的场景；但作为综述不自行提出技能归纳方法，而是把诸如 Voyager 式技能库归入其『跨试验记忆 + 文本形式』的分类坐标中加以对照。
- **在线 vs 离线**: 综述对两者均有覆盖且不偏废：在线——多数被调研系统在部署/交互过程中逐步写入与更新记忆（环境探索、经验积累为在线进行）；离线/批量——参数记忆的微调通常以批量方式离线进行。综述把『记忆采集→存储→检索→演化(反思/遗忘)』刻画为持续运行的循环管线，兼容在线与离线两种记忆构建方式。

**评测 / Evaluation**

- **任务领域**: 作为综述横跨多领域，在『记忆增强的智能体应用(memory-enhanced agent applications)』章节系统列举七类应用领域：角色扮演(role-playing)、社会模拟(social simulation)、个人助理(personal assistant)、开放世界游戏(open-world games)、代码生成(code generation)、推荐(recommendation)、专家系统(expert systems，含医疗诊断等)。每类应用说明其特定的记忆需求与实现方式。本身非针对单一领域。
- **基准**: 不适用于自身实验（N/A）：作为综述不报告自有基准实验。在『如何评估(how to evaluate)』章节，综述把评估方法归纳为两大范式——直接评估(direct evaluation)与间接评估(indirect evaluation)，而非列举单一基准。它指出被调研工作所用的评估指标包括检索准确率(retrieval accuracy)、任务成功率(task success rate)、用户满意度(user satisfaction)以及直接评估中的主观指标(记忆召回的连贯性 coherence、合理性 rationality)等。综述明确把『缺乏专门面向记忆模块的标准化基准』列为现有工作局限。
- **报告增益**: 不适用（N/A）：作为综述/分类学论文，本文不提出可量化的性能增益、不报告与基线对比的具体分数，也不给出 token/延迟数字。其『贡献』是定性的：提供统一术语与三维度设计分类(来源-形式-操作) + 评估二分法(直接-间接)，用以系统化组织既有记忆机制并抽象通用设计模式。综述定性总结：记忆使智能体无需重训权重即可随时间自我改进；当前以文本/向量型外部记忆为主流，参数记忆因成本与灾难性遗忘风险应用较少；评估呈现『直接测召回 vs 间接看下游任务表现』的二分格局。
- **对比基线**: 不适用（N/A）：综述无实验基线对比。在概念层面，它将『记忆增强的 LLM 智能体』与『无记忆的原始 LLM(输入→LLM→输出)』对照，强调后者无法解决需长期上下文的真实任务；并相对既有的 RAG（仅读取人写外部文档）强调智能体记忆可读写自生成、跨试验经验。

**分析 / Analysis**

- **关键创新**: 提出该领域首个专门针对『记忆机制』的系统性综述，建立两套被广泛沿用的分类框架：(1) 记忆设计三维度——记忆来源(sources：试验内/跨试验/外部知识)、记忆形式(forms：文本型[上下文内 vs 外部存储] / 参数型)、记忆操作(operations：写入/管理[反思·总结·合并·遗忘]/读取)；(2) 记忆评估二分法——直接评估(direct，主观+客观地独立测量记忆质量)与间接评估(indirect，经下游智能体任务表现间接衡量)。并从认知心理学、自我演化、应用三视角论证记忆的必要性，把『经验积累/环境探索/知识抽象』确立为记忆驱动自我演化的核心机制。
- **局限**: 综述自身/对现有工作指出的局限：(1) 记忆检索常不准确或被近因性(recency)偏置；(2) 记忆存储容量受检索延迟与成本制约，难以无限扩展；(3) 缺乏专门面向记忆模块的标准化评估基准；(4) 在长期记忆中存储用户交互引发隐私问题；(5) 参数记忆研究不足（成本高、灾难性遗忘风险）；(6) 多智能体系统中的记忆整合具挑战。作为 2024 年综述，其覆盖截至该时点，未涵盖 2025-26 的学习式记忆控制、时序记忆图谱、多智能体记忆路由、记忆安全等新进展。
- **与其他工作关系**: 本研究 H 类（综述）的核心权威条目，是整个研究库的概念骨架来源之一：其『来源/形式/操作』三维度与『直接/间接评估』二分法为本库众多系统条目提供横向归类坐标。它把本库多个系统作为分类实例纳入：Reflexion（A 类失败反思）被归为『跨试验信息 + 反思管理操作』、Generative Agents（B1）被归为『文本形式 + 用户/智能体来源 + 检索操作 + 反思』、MemoryBank（B2，自然语言存储+艾宾浩斯式遗忘)、RET-LLM（读写操作设计）、Voyager(C1，开放世界技能记忆)等。相对 CoALA（E1，认知架构四分类透镜）它采用不同的分类切面(来源-形式-操作)，二者互补；相对纯 RAG（D 类）它强调智能体可读写自生成的跨试验经验。后续 2025-26 工作(Mem0/Zep/MIRIX/Memory-R1 等)正是在其指出的检索准确性、遗忘机制、参数记忆、多智能体记忆整合等空白方向上的推进。
- **可复现性**: 可复现性概念高、实验复现不适用：作为综述无需复现算法实验。官方配套仓库 LLM_Agent_Memory_Survey（约 495 stars / 12 forks）公开可用，含分类示意图与按综述结构整理的相关论文清单，社区采用度高；综述的术语与分类被大量后续论文引用(约 568 次)。其『可复现性』体现为分类框架易被各系统映射套用，但不附带可运行的记忆系统参考实现。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（以启发式/设计者指定为主）：本综述（2024 年）归纳的记忆管理操作（写入/管理/读取的何时与如何）均为启发式规则 + LLM 推理驱动，未把『用 RL/训练学习记忆管理策略本身』作为独立类别。它指出的未来方向(更高效记忆结构、更好遗忘机制、参数记忆)隐含了对更自动化记忆管理的需求，但学习式记忆控制(如后续 Memory-R1、Mem-α)属于该综述发表之后的代际推进，未被本综述覆盖。
- **记忆主体**: 兼顾用户中心与智能体中心，但综述视角更偏智能体中心：核心论点是记忆服务于智能体的『自我演化(self-evolution)』——经验积累/环境探索/知识抽象(agent-centric)。同时在应用层面充分覆盖用户中心的个性化场景(个人助理记住用户偏好、角色扮演保持人设一致)。记忆来源分类(环境/用户/智能体自身)即同时容纳两种主体。
- **多智能体记忆**: 以单智能体记忆为主要建模对象：综述的来源-形式-操作三维度针对单个智能体的记忆模块定义。它把『多智能体系统中的记忆整合(memory integration in multi-agent systems)』列为现有工作的挑战与未来方向，但未提出跨智能体共享/路由记忆的具体分层机制(区别于后续 G-Memory、MIRIX 等多智能体记忆架构)。
- **模态**: 以文本为主：综述聚焦自然语言文本型记忆与参数型记忆，主体讨论文本记忆；虽涉及开放世界游戏、个人助理等可能含多模态感知的应用，但未将多模态/视觉记忆作为独立分类维度系统展开（多模态记忆为后续工作如 MIRIX 的重点）。
- **过度个性化/记忆安全风险**: 部分触及但非专门主题（早于 OP-Bench 等记忆安全基准）：综述明确把『在长期记忆中存储用户交互引发的隐私问题(privacy concerns)』列为现有工作局限之一，体现对记忆负面维度的关注；但未系统讨论有害/陈旧/侵入式/谄媚式个性化记忆的治理，也未提出相关安全基准——这些是 2025-26 记忆安全工作(OP-Bench、Causal-LoCoMo 等)的范畴。
- **token成本/延迟证据**: 无量化数据：作为综述不报告具体 token/延迟节省数字。仅作定性论述：指出记忆存储容量受检索延迟(retrieval latency)与计算成本(cost)制约，存在『检索准确性 vs 维护大型外部记忆库的计算开销』之间的权衡；并以此论证更高效记忆结构(分层/图结构)为重要未来方向。

**其他信息 / Other**

- **cluster**: H. 综述 (Surveys)

**不确定字段 / Uncertain**

- 冲突/矛盾处理 (`conflict_contradiction_handling`)
- 时序推理支持 (`temporal_reasoning_support`)


<a id="h2-rethinking-memory-in-ai-taxonomy-operations-topics-and-future-directionsai-中的记忆再思考分类学操作主题与未来方向最新修订版huggingfaceads-收录改题为rethinking-memory-in-llm-based-agents-representations-operations-and-emerging-topics即任务锚点中的survey-rethinking-memory-in-llm-based-agents之由来配套资源别名memory-compassgithub-仓库-survey_memory_in_ai这是一篇综述立场论文非具体记忆系统"></a>

### H2 《Rethinking Memory in AI

*《Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions》（AI 中的记忆再思考：分类学、操作、主题与未来方向）。最新修订版（HuggingFace/ADS 收录）改题为《Rethinking Memory in LLM based Agents: Representations, Operations, and Emerging Topics》（即任务锚点中的『Survey: Rethinking Memory in LLM-based Agents』之由来）。配套资源别名：『Memory Compass』（GitHub 仓库 Survey_Memory_in_AI）。这是一篇综述/立场论文，非具体记忆系统。*


**基本信息 / Provenance**

- **年份**: 2025（arXiv 预印本 v1：2025-05-01；存在 v2 修订版；属 cs.CL）
- **作者/机构**: Yiming Du（杜一鸣，第一作者，香港中文大学 CUHK，研究期间访学于爱丁堡大学）、Wenyu Huang、Danna Zheng（三人并列共同第一作者，爱丁堡大学）、Zhaowei Wang（HKUST 香港科技大学）、Sebastien Montella（华为英国 R&D Poisson Lab）、Mirella Lapata（爱丁堡大学，资深作者）、Kam-Fai Wong（黄锦辉，CUHK）、Jeff Z. Pan（潘宇，爱丁堡大学 / 华为英国，通讯/资深作者）。主要机构：香港中文大学、爱丁堡大学、香港科技大学、华为英国 R&D（Poisson Lab, CSI）。
- **发表venue**: arXiv 预印本（cs.CL，Computation and Language），2025 年 5 月；ADS Bibcode 2025arXiv250500675D。未见正式会议/期刊发表记录（截至核实时为 arXiv preprint）。
- **论文链接**: https://arxiv.org/abs/2505.00675 （DOI: https://doi.org/10.48550/arXiv.2505.00675）
- **代码链接**: https://github.com/Elvin-Yiming-Du/Survey_Memory_in_AI （仓库名『Memory Compass』；约 351 stars、17 forks，MIT 许可，创建于 2025-04-25，持续维护至 2025-12；非可运行系统，而是综述配套的论文清单/数据集/方法/工具汇编与持续更新的 awesome-list）

**记忆分类 / Taxonomy**

- **记忆类型**: 作为综述，本文不实现单一记忆类型，而是提出一个跨越四类 CoALA 心理学记忆维度的统一框架；其核心新分类是按『表示形态』而非功能划分：参数化记忆（parametric，隐含于模型权重，对应程序/语义知识的隐式形态）与上下文记忆（contextual，外部显式信息），后者再分为非结构化（unstructured：文本/图像/音频/视频等异质模态，按时间范围分短期与长期）与结构化（structured：知识图谱、关系表、本体等可查询模式）。论文第 5 节专门对照人类记忆（工作记忆 + 情景/语义长期记忆）与智能体记忆（短期上下文窗口 + 持久化外部/参数化模块）的功能异同。
- **记忆结构**: 综述层面覆盖全谱系数据结构：参数化=模型权重；上下文非结构化=原始对话历史/向量库/多模态缓冲；上下文结构化=知识图谱（Neo4j 类）/关系表/本体；并涵盖长上下文场景的 KV cache。论文以『分类学 × 6 操作 × 4 主题』三维框架（Figure 1、Table 1）组织，不规定单一底层结构，而是把各类结构映射到操作与主题的二维对齐表（Table 1：操作 × 参数化/结构化/非结构化）。
- **存储后端**: 综述明确枚举工具生态三层后端：(1) 基础组件——向量库 FAISS、图数据库 Neo4j、LLM（Llama/GPT-4/DeepSeek）、检索器（BM25、Contriever、OpenAI embeddings）；(2) 框架层——Graphiti、LlamaIndex、LangChain、LangGraph、EasyEdit、CrewAI、Letta；(3) 记忆层系统——Mem0、Zep、Memary、Memobase（提供编排、持久化与生命周期管理）。本身不绑定特定后端。
- **持久化**: 综述系统区分三种持久形态：参数化记忆（baked into weights，即时、长期、持久，但难选择性更新、不透明）；上下文记忆（外部显式、可读写、可结构化或非结构化，按时间分短期会话上下文与跨会话长期记录）；并在人类 vs 智能体对照表（Table 2）中刻画智能体记忆相对人类的特性：可共享/复制/广播、容量仅受存储与算力限制、支持回滚/反学习。

**核心机制 / Mechanisms**

- **写入/编码**: 综述以形式化的『巩固 (Consolidation)』操作刻画写入编码：将 t 到 t+Δt 间的 m 条短期经历 E=(ε₁,…,ε_m) 转化为持久记忆 M_t，记为 M_{t+Δt}=Consolidate(M_t, E)（式 1）。编码目标形态包括模型参数、图、知识库；服务于持续学习、个性化、外部 MemoryBank 构建与知识图谱构建。论文还区分『索引 (Indexing)』操作 I_t=Index(M_t, φ)（式 2）：构造实体/属性/内容表示等辅助编码 φ 作为访问点，并编码时序与关系结构以支持可遍历的索引路径，跨符号/神经/混合记忆系统实现可扩展检索。综述强调与编码相关的安全风险：记忆内容可被投毒/篡改，损坏片段可潜伏并触发恶意行为。
- **检索机制**: 综述将检索 (Retrieval) 形式化为记忆利用 (utilization) 的核心操作：在输入 Q（可为简单查询、多轮对话上下文、纯文本或视觉等多模态）下，从 M_t 中识别相关片段 m_Q，用相似度函数 sim() 打分并以阈值 τ 筛选：Retrieve(M_t, Q)=m_Q ∈ M_t，s.t. sim(Q, m_Q) ≥ τ（式 5）。检索目标可跨多源、多模态甚至模型内参数化表示（『检索参数化知识』被列为重要未来方向）。综述还引入 RCI（Relative Citation Index，相对引用指数，受 RCR 启发的时间归一化引用度量）来在 30K 论文中突出高影响检索/记忆工作；并指出『检索-生成失配』（retrieved content 过时/不相关/未对齐）是长期记忆的核心挑战，需时序推理、结构感知生成与检索鲁棒性。
- **反思/巩固**: 综述把『原始经历→持久知识』的转化统一为巩固 (Consolidation) 与压缩/凝练 (Compression/Condensation) 两个操作。巩固=在记忆构建时把交互历史（对话、轨迹）编码进持久形态（式 1），区别于压缩在推理时减小记忆（式 6）。论文以 MyAgent、MemoChat 等为长期记忆巩固代表，COMEDY/MEMORAG/ReadAgent 为生成-利用代表。综述并未提出新的反思算法，而是把反思/总结/抽象归入巩固与更新操作族，并在人类 vs 智能体对照中指出：智能体巩固是『快速、显式、策略驱动且有选择性的』，而人类巩固是『缓慢、生物驱动、被动的』。
- **遗忘/更新**: 综述把遗忘 (Forgetting) 与更新 (Updating) 列为两个独立的管理操作并给出形式化：更新 M_{t+Δt}=Update(M_t, K_{t+Δt})（式 3）——重新激活并临时修改既有记忆以纳入新知识 K；参数化更新用 locate-and-edit 机制（如 AlphaEdit/ROME/MEMIT），上下文更新用摘要/剪枝/精炼替换过时内容。遗忘 M_{t+Δt}=Forget(M_t, F)（式 4）——选择性抑制过时/无关/有害内容 F；参数化遗忘=机器反学习 (unlearning)，上下文遗忘=基于时间的删除或语义过滤。论文坦言当前系统多依赖启发式解决新旧冲突、缺乏显式仲裁机制，并警示反学习方法易受恶意攻击。
- **经验回放 (核心主题)**: 综述把经验复用纳入巩固（把交互历史/轨迹编码为持久记忆供后续检索）与检索（取回相关记忆支持下游生成/规划）的循环；在生物启发未来方向中专门讨论『经验回放 (experience replay)』——借鉴互补学习系统（海马快编码情景、皮层慢整合长期记忆）以缓解遗忘，并列举 dual-memory 架构、突触巩固、经验回放（Ritter et al. 2018; Wang et al. 2021）为对抗灾难性遗忘的策略。作为综述，它分类与对比既有复用范式（参数化 vs 非参数化、长期记忆个性化复用、任务导向 agent 的工作流/KV 复用如 A-MEM、Optimus-1），而非提出新的复用机制。

**学习维度 / Learning**

- **学习范式**: 综述同时覆盖参数化与非参数化两大范式并以表示形态统一之：参数化范式=知识编辑 (model editing：ROME/MEMIT/AlphaEdit、MEND、SERAC)、反学习 (unlearning)、持续/终身学习，直接改写权重；非参数化范式=上下文/外部记忆的巩固、索引、更新、遗忘、检索、压缩。论文倡议未来把编辑、反学习与持续学习统一进一个内聚框架，并探索『统一记忆表示』（联合参数化与外部记忆的表示空间与索引机制）。
- **失败学习 (核心主题)**: 综述未把失败学习作为独立章节专门处理（这更多见于 A 类自我反思系统如 Reflexion）。它从记忆操作角度间接覆盖：更新与遗忘操作用于移除过时/错误内容、修正记忆；并在挑战中指出当前系统缺乏对新旧矛盾的显式仲裁、依赖启发式。安全性章节（Memory Threats & Safety）讨论错误/有害记忆的处理与反学习方法的脆弱性。整体上，本综述对失败/错误的处理是分类性的（归入更新/遗忘/反学习操作族），而非提出新的失败检测与负样本学习机制。
- **技能/程序归纳**: 综述将技能/程序的归纳归入参数化记忆（程序知识隐含于权重）与任务导向 agent 的结构化记忆（工作流图、key-value 存储维护会话连续与长程推理，如 A-MEM、Du et al. 2025）；并在『从操作到研究主题』中以 Optimus-1、AutoManual 类工作流记忆为多源整合代表。作为综述不提出新的技能归纳算法，而是把技能/工作流记忆映射到巩固（构建）与检索（调用）操作及长期/多源主题。
- **在线 vs 离线**: 综述两者兼论：在线——多会话对话、个性化 agent 在部署中持续巩固/更新/遗忘记忆（终身学习方向强调稳定性-可塑性平衡、在线整合新信息同时保留旧知识）；离线——参数化记忆的批量编辑/反学习/持续学习训练，以及本综述自身的离线大规模文献分析（用 GPT-4o-mini 对 2022–2025 NeurIPS/ICLR/ICML/ACL/EMNLP/NAACL 的 30K+ 论文打相关性分，保留 ≥8 分的 3,923 篇）。

**评测 / Evaluation**

- **任务领域**: 综述横跨多领域（回顾性覆盖）：多会话对话与个性化（多轮对话、个性化助教、AI 陪伴如 Replika）、检索增强 QA（RAG、HippoRAG、LongMemEval、LoCoMo）、长上下文处理（KV cache 优化、上下文压缩）、知识编辑/反学习（ROME/MEMIT/TOFU）、任务导向 agent（项目管理、虚拟助手、代码生成 GitHub Copilot/Coze/CodeBuddy）、多模态系统（自动驾驶、医疗决策）。映射到四大研究主题：长期记忆（temporal）、长上下文记忆（contextual）、参数化记忆修改（model-internal）、多源记忆（modality/integration）。
- **基准**: 作为综述不报告自身实验，但系统编目相关基准/数据集：长期记忆评测 LoCoMo、LongMemEval；RAG/QA 相关 HippoRAG 等；反学习基准 TOFU（论文批评其过于简单，难以暴露真实局限）；以及附录表（Tables 4–20）汇总各主题方法、数据集与工具的适用记忆类型/操作/输入输出/功能/场景/来源。论文呼吁建立统一评测以覆盖一致性、个性化与时序推理，并评估巩固/更新/检索/遗忘等核心操作在动态多会话设置下的表现。
- **报告增益**: 不适用（N/A）：作为分类学综述/立场论文，本文不提出可量化性能增益、不与基线对比分数、不报告自身 token/延迟数据。其量化贡献为文献计量层面：从 6 个顶会（NeurIPS/ICLR/ICML/ACL/EMNLP/NAACL，2022–2025）采集 30K+ 论文，用 GPT-4o-mini 零样本打分，保留相关性 ≥8/10 的 3,923 篇高相关论文（经人工抽样校验阈值），并用 RCI（相对引用指数，按发表年龄归一化引用）排序突出高影响工作。贡献是定性框架而非实验增益。
- **对比基线**: 不适用（N/A）：无实验基线对比。在综述论证层面，本文以『现有综述』为对照对象——批评 Zhang et al. (2024f) 等先前综述仅覆盖写/管理/读等高层操作而遗漏索引等原子操作，且多局限于长上下文、长期记忆、个性化或知识编辑等单一子主题，缺乏统一的操作化框架与明确的文献范围；本综述以参数化/上下文分类 + 6 原子操作 + RCI 文献计量来填补这些空白。

**分析 / Analysis**

- **关键创新**: 提出以『记忆表示形态 × 原子操作』为核心的统一框架：(1) 按表示把记忆分为参数化 (parametric) 与上下文 (contextual，再分结构化/非结构化)，区别于以往按功能分类的综述；(2) 定义 6 个基础原子操作并给出形式化定义（管理：Consolidation 巩固、Indexing 索引、Updating 更新、Forgetting 遗忘；利用：Retrieval 检索、Compression/Condensation 压缩/凝练），各配公式（式 1–6）；(3) 将操作映射到四大研究主题（长期/长上下文/参数化修改/多源），并以 RCI 文献计量从 30K+ 顶会论文中筛出 3,923 篇做系统分析；(4) 提供工具生态三层图谱与人类-智能体记忆对照。最大贡献是把碎片化的记忆研究重构为『原子操作 + 表示形态』的动态结构化透镜。
- **局限**: (1) 纯综述/分类学框架，无可运行实现与实验验证；(2) 文献筛选依赖 GPT-4o-mini 自动打分（阈值 ≥8），存在自动标注偏差，仅以抽样人工校验；(3) RCI 等引用计量对新近论文有时间偏差；(4) 论文自承多处研究空白尚无成熟方案：统一评测缺失、检索-生成失配、参数化编辑缺乏特异性且难以规模化（多数方法不超过数千次编辑、不支持 >20B 模型）、终身学习与多源冲突消解欠成熟、记忆安全（反学习易受攻击、投毒）；(5) 截至 arXiv 预印本未正式发表，且存在版本间术语漂移（第 6 操作 Compression↔Condensation、标题在『AI』与『LLM-based Agents』间变动）。
- **与其他工作关系**: 本研究 H 类（综述）的核心条目之一。其分类透镜与 E1 CoALA（episodic/semantic/procedural/working 四分类）互补：CoALA 按功能/认知角色分类并刻画单智能体的记忆-动作-决策，而本综述按『表示形态 (参数化/上下文)』分类并以 6 原子操作 + 4 主题 + RCI 文献计量重构整个领域，覆盖面更广（含知识编辑/反学习/长上下文 KV cache 等 CoALA 未涉主题）。它在编目中直接引用并归类本库多个条目：把 HippoRAG (D1) 归为长期记忆的索引/检索代表、LongMemEval (F1) 与 LoCoMo (F2) 归为长期记忆基准、A-MEM (B4) 归为多源整合、MemoryBank (B2)/MemGPT-Letta (B3) 归为遗忘/工具层、Mem0 (D4)/Zep-Graphiti (D3) 归为记忆层系统与框架。其未来方向（时空记忆、检索参数化知识、终身学习、多智能体记忆、统一记忆表示、记忆安全）与本库 G 类（学习式记忆控制 Memory-R1）、D3 Zep（时序）、D5 G-Memory/B9 MIRIX（多智能体）、F7 OP-Bench/F8 Causal-LoCoMo（记忆安全）等条目高度呼应。
- **可复现性**: 高（综述层面）：官方配套 GitHub 仓库『Memory Compass』(Survey_Memory_in_AI，约 351 stars、MIT、7 位贡献者，持续更新至 2025-12) 公开论文清单、数据集、方法与工具汇编（含 awesome-list 与 Notions）。文献筛选流程（30K→3,923，GPT-4o-mini 打分、≥8 阈值、人工抽样校验）在附录 A/B 描述，原则上可复现，但完整 30K 标注数据是否全部开放未明确。作为综述无需复现实验。社区采用度通过仓库 star 与持续维护体现。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（综述本身不实现学习式记忆控制策略）。但本文在分类与未来方向上为该方向提供框架与动机：它把记忆管理 (consolidation/indexing/updating/forgetting) 与利用 (retrieval/compression) 形式化为可被策略控制的操作，并指出当前系统多依赖启发式仲裁、缺乏显式策略；其未来方向（自适应记忆管理、统一编辑/反学习/持续学习框架）正对应 2025-26 代的学习式记忆控制工作（Memory-R1、Mem-α）。综述定位为该代际划分的分类学坐标，而非实现者。
- **记忆主体**: 兼顾两者（综述全景覆盖）：既覆盖用户中心 (user-centric)——个性化对话、用户偏好建模、AI 陪伴（Replika）、推荐（Amazon）、虚拟助手（Me.bot、ima.copilot），也覆盖智能体中心/知识中心 (agent/knowledge-centric)——参数化知识 agent（编程、医疗、金融、法律）、任务导向 agent 的工作流/会话记忆、多智能体分布式记忆。论文将应用按主导记忆模态分为知识中心、用户中心、任务导向、多模态四类。
- **多智能体记忆**: 明确讨论并列为重要未来方向：第 6.2 节『Memory in Multi-agent Systems』指出多智能体中记忆既是个体的也是分布式的，智能体须管理自身内部记忆同时与他者交互学习，带来记忆共享、对齐、冲突消解与跨智能体一致性等独特挑战；倡议去中心化记忆架构、跨智能体记忆同步与集体记忆巩固，以支持协作规划与长期协调。综述编目中把 G-Memory、MIRIX 类多智能体记忆系统纳入相关工作。
- **时序推理支持**: 显式强调：综述指出操作『自然蕴含记忆的时间性』（信息随时间演化），索引操作可编码时序与关系结构；专设未来方向『时空记忆 (Spatio-temporal Memory)』——既捕捉信息间结构关系也捕捉其时间演化，使智能体能在保留历史脉络的同时自适应更新知识（例：记录用户曾不喜欢西兰花、后据近期购买调整），支持时序知情推理与细粒度个性化。论文将时序推理列为长期记忆评测中最难且当前基准罕能评估的能力之一，并把 Zep/Graphiti 类时序记忆图谱纳入框架与工具层。
- **模态**: 多模态（综述层面全覆盖）：上下文非结构化记忆被定义为『模态通用 (modality-general)』系统，跨文本、图像、音频、视频存取；多源记忆 (Multi-Source) 是四大主题之一，强调跨异质文本源整合并扩展到多模态输入以支持场景感知推理；多模态系统（语言+视觉+音频）应用于自动驾驶、医疗决策。
- **过度个性化/记忆安全风险**: 有专门讨论（记忆安全维度）：第 6.2 节『Memory Threats & Safety』指出记忆常存储敏感/机密数据，增删操作非平凡；多项研究（Liu et al. 2025b; Barez et al. 2025）证明机器反学习方法易受恶意攻击，强调需更安全可靠的记忆操作。论文在人类-智能体对照中也警示：内部记忆痕迹的反复复用可能逐渐使智能体偏向特定行为轨迹、隐式塑造身份；优化驱动的遗忘/压缩可能删除低频但情感/社会显著的数据，尤其在交互或安全关键场景。指出『更多记忆并非总是更好』的隐患。
- **冲突/矛盾处理**: 明确指出为开放挑战：综述坦言『当前多数系统依赖启发式解决新输入与既有记忆间的冲突，缺乏显式仲裁机制 (explicit arbitration mechanisms)』；在多源整合主题与未来方向中强调需要冲突消解 (conflict resolution)、时序接地 (temporal grounding) 与来源溯源 (provenance tracking) 来应对冗余、不一致与来源歧义（源于时间范围错配、语义冲突、缺失归属，尤其跨模态）。更新操作 (式 3) 涵盖据新知识修改记忆，但综述把矛盾的显式仲裁列为待解难题。
- **token成本/延迟证据**: 无自身量化数据（综述不做实验）：本文定性讨论效率-表达力权衡——长上下文场景的 KV cache 压缩/丢弃 (H2O、StreamingLLM、SnapKV)、KV cache 存储/选择优化 (KVQuant、KIVI、QUEST)、上下文压缩 (RECOMP、xRAG、LongLLMLingua) 提供效率但有信息损失/不稳定风险；压缩操作 (式 6) 以压缩比 α 在推理时减小记忆。论文把『效率 vs 表达力』列为长上下文处理的核心未解权衡，但不报告具体 token/延迟节省数字（这些见于被其编目的 Mem0/Zep 等记忆层系统）。

**其他信息 / Other**

- **cluster**: H. 综述 (Surveys)

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)


<a id="h3-memory-for-autonomous-llm-agents机制评测与新兴前沿综述别名survey-memory-for-autonomous-llm-agents"></a>

### H3 Memory for Autonomous LLM Agents

*Memory for Autonomous LLM Agents：机制、评测与新兴前沿（综述）（别名：Survey: Memory for Autonomous LLM Agents）*


**基本信息 / Provenance**

- **年份**: 2026（arXiv 预印本，2026-03-08 首次公开）
- **作者/机构**: 单一作者 Pengfei Du（杜鹏飞），香港研究科技学院（Hong Kong Research Institute of Technology, 香港）。手稿声明拟投期刊《Advanced Intelligent Systems》。
- **论文链接**: https://arxiv.org/abs/2603.07670
- **代码链接**: 无官方代码仓库（综述论文，不含代码/数据；数据可用性声明称未生成原始数据，所有引用文献均公开可得）。

**记忆分类 / Taxonomy**

- **记忆类型**: 综述覆盖全部四类：工作记忆（working，当前上下文窗口）、情景记忆（episodic，具体事件/工具调用/对话轮记录）、语义记忆（semantic，去情景化的抽象知识如用户偏好）、程序记忆（procedural，可复用技能与可执行计划，如 Voyager 技能库）。在三维分类法的“时间尺度（temporal scope）”维度沿用 CoALA/认知科学（Atkinson-Shiffrin、Tulving、Baddeley、Squire）的四分法。
- **记忆结构**: 作为综述，归纳全部底层结构形态，并以“表征基质（representational substrate）”维度统一：上下文驻留文本（摘要/草稿/思维链）、向量索引存储（稠密 embedding + 近似最近邻）、结构化存储（SQL/键值/知识图谱）、可执行仓库（代码库/工具定义/计划模板）、以及生产中常见的混合分层存储（如 MemGPT 主上下文+召回库+归档向量库）。还涉及参数化记忆（权重内的微调/adapter）。
- **存储后端**: 综述列举多种具体后端：上下文窗口（in-context）、FAISS 式向量库 + 稠密检索（DPR）、SQL 数据库（ChatDB）、键值映射、知识图谱、文件、代码库（Voyager JavaScript 技能），以及模型参数（MemLLM、LongMem 侧网络）。MemGPT 采用 OS 虚拟内存式三级后端（主上下文/召回 DB/归档向量库）。
- **持久化**: 三种持久化形态并存并被系统比较：上下文内（瞬时、容量受限）、外部存储（持久、可检索/治理/删除）、参数化（烘焙进权重，便于无缝整合但难定向删除与审计）。综述强调长上下文（200k+ token）只扩大工作记忆，无法替代跨会话持久存储、结构化组织、选择性检索与治理机制。

**核心机制 / Mechanisms**

- **写入/编码**: 作为综述系统梳理了“写入路径（write path）”：原始经验既可逐字记录情景轨迹（verbatim trajectory），也可经摘要提炼为洞见、抽取为事实/结构化三元组（RET-LLM 写入时写结构化三元组、读取时用自然语言查询）、编码为稠密向量，或固化为可执行技能（Voyager）。综述明确反对“全量逐字存储”，主张写路径应包含：低信号过滤（filtering）、规范化（canonicalization，统一日期/姓名/数量）、去重（deduplication）、优先级打分（按任务相关性与新颖度排序）、元数据标注（时间戳/来源/任务标签/置信度）。最优过滤阈值与领域风险相关（医疗代理不能漏写药物过敏；闲聊助手可容忍漏写）。情景→语义的巩固转换是最薄弱环节，多依赖开发者规则或周期性 LLM 摘要，脆弱且难验证。
- **检索机制**: 综述形式化读取算子 R(M_t, x_t) 并归纳多种读路径机制：(1) 相似度检索——稠密向量检索（DPR）+ FAISS 近似最近邻，常配合稀疏 BM25 与元数据过滤；(2) Generative Agents 的多信号加权打分——近因性（recency，指数衰减）×相关性（relevance，embedding 相似度）×重要性（importance，自评整数），优于纯余弦相似度；(3) 查询重构（LLM 改写查询、多查询扇出 + 结果融合、以当前子目标作为额外检索信号），因为当前输入 x_t 常是糟糕的检索查询；(4) 检索门控（Self-RAG 学习判断是否需要检索）；(5) 两阶段检索（快速 BM25/元数据过滤 → 慢速 cross-encoder 重排）；(6) 多粒度索引（自适应选择粒度，避免细粒度碎片化或粗粒度淹没信号）；(7) 学习式检索/策略化检索（Agentic Memory 将 retrieve 作为 RL 可调动作）。指出瓶颈已从存储转向“相关性”——返回最有用而非最相似的记录。
- **反思/巩固**: 综述将“反思与自改进记忆”列为五大机制家族之一并深入剖析：Reflexion 在失败后写自然语言事后复盘并前置到下次提示（无梯度更新，HumanEval pass@1 91% vs GPT-4 80%）；Generative Agents 周期性聚类相关观察、综合更高阶反思（如“Klaus 最近独自吃饭、显得孤僻”）；ExpeL 系统性对比成功/失败轨迹抽取判别性“经验法则”；Think-in-Memory 将检索与推理分离。巩固（情景→语义）一般由显式提示或启发式触发，缺乏自动化。核心风险：自我强化的错误（self-reinforcing error，错误反思被永久保留并放大，危害随智能体寿命增长）与过度泛化。缓解手段：反思接地（reflection grounding，要求为每条反思引用具体情景证据）、置信度打分、与其它记忆做矛盾检查、周期性过期。开放挑战章节进一步提出可信反思需外部校验、不确定性量化、对抗性探测与过期策略；并将类脑“离线巩固”（睡眠期海马回放）与“双缓冲巩固”（热缓冲试用期→通过质检后晋升长期存储）列为前沿方向。
- **遗忘/更新**: 综述系统讨论遗忘/更新/去重/失效：时间版本化（优先最新记录）、来源归属（用户陈述 > 智能体推断）、矛盾检测（冲突标记待解决）、周期性巩固（定时合并重复、淘汰陈旧）；MemoryBank 用 Ebbinghaus 遗忘曲线建模衰减（高频高重要性记忆被强化、被忽略者淡出）；Agentic Memory 将 update/discard/summarize 作为可学习 RL 动作。指出当前系统对遗忘处理粗糙（硬时间过期、容量驱逐或完全不处理），“学会遗忘”是开放难题；记忆若已渗入微调权重则需机器遗忘（machine unlearning），尚远未生产可用。
- **经验回放 (核心主题)**: 作为本研究核心主题，综述以多个系统佐证经验复用：Reflexion 把过去失败的口头复盘前置复用（exemplar/text journal）；Voyager 不断增长的程序技能库（已验证 Minecraft 例程存为可运行 JavaScript、按自然语言描述索引、动态组合解决新任务）——消融显示去掉技能库导致里程碑速度损失 15.3×；ExpeL 从成功/失败轨迹蒸馏“经验法则”作为可复用启发式；JARVIS-1 多模态记忆复用视觉观察与计划。开放挑战章节将类脑“离线巩固/海马回放”作为更原则化的经验整合方向。综述同时把跨域记忆迁移（如 Python 调试启发式复用到 Java）列为新兴方向，开放问题是“哪些记忆可泛化、哪些不可”。

**学习维度 / Learning**

- **学习范式**: 综述梳理三代/三范式并存：非参数（in-context/prompt 级，如 Reflexion 文本复盘、RAG 外部存储）、参数化（梯度更新，如 MemLLM 微调读写模块、检索与生成联合训练、LongMem 冻结主干+残差侧网络）、以及混合。并指出参数化与非参数记忆有不同失败画像：参数化擅长无缝知识整合但难定向删除与审计；非参数支持检视与治理但可能“外挂感”、被智能体忽略或不一致使用——两者最优结合仍是开放经验问题。
- **失败学习 (核心主题)**: 作为本研究核心主题，综述强调从失败中学习是反思式记忆的核心：Reflexion 在任务失败后写自然语言事后复盘（post-mortem）并复用，显著提升下一次尝试成功率；ExpeL 通过系统性对比成功 vs 失败轨迹抽取判别性“经验法则”（既含正例也含负例/失败模式），形成可复用启发式以避免重蹈覆辙。引言以调试助手为例——“最糟的是周一早晨重试周五崩溃过的修复”，有记忆后即可跳过死胡同。同时警示失败学习的反面：错误的失败归因会自我强化（如误判“API X 总报错”后永久回避，无法纠偏），需引用具体证据接地、矛盾检查与过期机制；危害随智能体寿命增长而放大。
- **技能/程序归纳**: 是。综述将程序记忆与可执行仓库作为独立维度处理：Voyager 从经验归纳可复用技能（已验证的 Minecraft 例程存为可运行 JavaScript 代码、以自然语言描述索引、面对新任务时动态组合调用），实现终身学习（独特物品 3.3×、科技树进度 15.3×）；MetaGPT/ChatDev 以标准化文档（PRD/设计规范/代码模块）作为共享程序性记忆持续演化；工具使用场景需维护“带版本的工具能力目录”应对 schema drift。综述指出软件工程代理最依赖程序记忆、游戏代理需情景+程序记忆紧密结合。
- **在线 vs 离线**: 综述涵盖两种构建方式并加以对比：在线（部署中逐情景增量更新，如对话/游戏代理实时写入与反思）与离线（对轨迹语料批量处理，如 ExpeL 对比成功/失败轨迹离线抽取规则、Agentic Memory 的离线 RL 训练阶段）。开放挑战章节倡导“离线巩固”（类比睡眠期海马回放，在空闲期调度的离线整合过程）以平衡囤积与遗忘。

**评测 / Evaluation**

- **任务领域**: 综述覆盖广泛领域：多会话对话/个人助理、软件工程/编码代理（百万行代码库）、开放世界游戏（Minecraft 等沙盒）、科学推理与发现（假设账本/证据累积）、多智能体协作、工具使用与 API 编排、跨域记忆迁移；以及评测侧的网页导航、偏好约束规划、渐进式信息检索、序列化形式推理。
- **基准**: 重点分析四个近期基准：LoCoMo（最长 35 会话、300+ 轮、每段对话 9k–16k token；事实 QA/事件摘要/对话生成）、MemBench（区分事实型 vs 反思型记忆，参与 vs 观察模式；ACL 2025 Findings）、MemoryAgentBench（基于认知科学的四种能力：精确检索/测试时学习/长程理解/选择性遗忘）、MemoryArena（多会话相互依赖的智能体任务，覆盖四个领域）。还提及历史系统涉及的 HumanEval、ALFWorld、AgentBench 等。并提出四层评测指标栈（任务效果/记忆质量/效率/治理）。
- **报告增益**: 作为综述汇总各系统头条结果（非自身实验）：MemoryArena 中将主动记忆代理换成纯长上下文基线，相互依赖多会话任务完成率从 >80% 跌至约 45%（约 -35 个百分点，即主动记忆 +35pp）；在 LoCoMo 上接近饱和的模型到 MemoryArena 仅 40–60%。Voyager（vs 无技能库基线）：独特物品 3.3×、科技树里程碑速度 15.3×。Reflexion：HumanEval pass@1 91% vs GPT-4 80%（+11pp）。ReAct：ALFWorld 绝对提升 34%。RETRO：从 2 万亿 token 语料检索，7.5B 参数模型在 16 项中的 10 项上比肩 175B 的 Jurassic-1。Generative Agents：去掉反思组件后多日连贯规划在 48 模拟小时内退化为重复无上下文响应。
- **对比基线**: 综述对比的基线类型包括：无记忆代理、纯长上下文（long-context-only，如 200k token 窗口）、RAG 增强 LLM、以及不同记忆系统之间互比（如 Agentic Memory 在五个基准上胜过所有记忆增强基线；Voyager vs 无技能库消融）。核心论断：长上下文不是记忆——纯长上下文在需选择性检索与主动管理的任务上持续逊于专用记忆系统。

**分析 / Analysis**

- **关键创新**: 最重要贡献：(1) 将智能体记忆形式化为嵌入 POMDP 智能体循环的“写-管理-读（write–manage–read）”闭环（a_t=π_θ(x_t, R(M_t,x_t), g_t)；M_{t+1}=U(M_t,x_t,a_t,o_t,r_t)）；(2) 提出统一的三维分类法（时间尺度 / 表征基质 / 控制策略），其中“控制策略”维度（启发式 / 提示自控 / 学习式控制）是相对早期综述最具区分度的更新；(3) 深入剖析五大机制家族（上下文驻留压缩、检索增强存储、反思自改进、分层虚拟上下文、策略学习式管理）并配实证对比与 2025–2026 新系统/新基准。
- **局限**: 作为综述自身局限：无新实验/无代码，结论依赖二手报告；覆盖 2022–2026 早期，时效性会随领域快速演进而衰减。它所综述领域的共性局限亦被指出：无人很好评测遗忘；跨会话一致性研究不足；评测普遍不报告成本（token/延迟）；缺乏社区标准评测基准（各基准数据/指标/协议各异，难横向比较）；情景→语义巩固脆弱；反思易自我强化错误；多智能体记忆治理（访问控制/并发写一致性）几乎无满意方案；记忆系统可观测性/调试基础设施在论文中鲜被讨论却是落地关键。
- **与其他工作关系**: 本条目为本研究的“H. Surveys（综述）”簇成员，是覆盖其它条目的伞状综述。明确定位相对前序综述：更新 Zhang et al.（2024，同样围绕 write–manage–read 操作组织）至 2025–2026 系统（Agentic Memory/AgeMem、MemBench、MemoryAgentBench、MemoryArena），新增 POMDP 接地的形式化，并扩展到应用、工程模式与治理；与 Gao et al.（RAG 综述，范围限于检索-生成管线而非智能体记忆）、Sumers et al.（CoALA 认知架构蓝图，本分类法与之互补、共享认知科学术语但扩展到表征基质与控制策略）、Xi et al./Wang et al.（宽泛 agent 综述、记忆仅为其一模块）相区分。直接关联本研究其它条目对应的代表系统：Reflexion、Generative Agents、Voyager、ExpeL、MemGPT、MemoryBank、ChatDB、LongMem、RAG/RETRO、Self-RAG、MemLLM，以及策略学习式 Agentic Memory（GRPO 强化学习记忆管理）。需注意与同期其它综述（如 arXiv 2602.06052“Rethinking Memory…Second Half”、2505.00675、2602.19320 等）属并行工作而非引用关系。
- **可复现性**: 综述类论文，无代码与数据需复现；数据可用性声明称未生成原始数据、所有引用文献公开可得。可复现性更多体现于其方法论框架（写-管理-读闭环、三维分类法、四层评测栈）可被他人采用；其引用的具体系统（Voyager、MemGPT、Reflexion 等）各自有公开代码。综述本身致谢段尚标注“作者/资助/伦理元数据需在投稿前敲定”，显示其为预印/在投阶段，社区采纳信号仍早期（约 26 次引用）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 重点覆盖。综述把“策略学习式记忆管理（policy-learned management）”列为五大机制家族之一，并将“控制策略”作为分类法第三维（启发式 / 提示自控（如 MemGPT 暴露 core_memory_append、archival_memory_search 工具由 LLM 决定调用）/ 学习式控制）。代表系统 Agentic Memory（AgeMem, Yu et al. 2026）把 store/retrieve/update/summarize/discard 五种操作作为可调用工具，用三阶段 RL（监督预热 → 任务级结果奖励 RL → 步级 GRPO 稠密信用分配）端到端优化，发现非显然策略（上下文将满前预先摘要、丢弃语义冗余记录）。视为 2025–2026 的代际分水岭，但训练成本高、可能误删安全关键信息、跨任务迁移与可解释性存疑。
- **记忆主体**: 综述同时覆盖用户中心（记住用户信息以个性化——个人助理/对话代理，MemGPT 演化用户模型、MemoryBank 个性化衰减）与智能体中心（记住自身经验以自我改进——Voyager 技能库、Reflexion 自我复盘、ExpeL 经验法则）。第 6.8 节明确不同领域侧重不同记忆类型：个人助理最依赖语义记忆（用户偏好画像），软件/游戏/科学代理更依赖程序/情景记忆（自身经验）。
- **多智能体记忆**: 覆盖单智能体与多智能体共享/路由记忆的对比。多智能体场景（AutoGen 共享上下文、CAMEL 角色感知通信、ProAgent 主动队友、ChatDev/MetaGPT 共享文档记忆）将记忆作为协调机制，核心难题为“共享 vs 私有记忆边界（谁可见什么）”与“并发写一致性”。综述批评当前两极方案（全共享会泄私 / 完全隔离阻断知识迁移）均不理想，建议基于角色的访问控制（ACL 式、适配自然语言记录）作为中间路径；并把多智能体记忆治理（访问控制、并发写共识协议、跨智能体知识迁移、分层共享记忆+按智能体缓存）列为开放挑战。
- **时序推理支持**: 综述明确强调时间推理为最难评测能力之一：LoCoMo 头条结论是 RAG 增强 LLM 在时间与因果动态上远逊于人类；工程章节要求时间版本化（优先最新记录）以区分 2024 与 2022 的地址等陈旧信息；并提出“因果接地检索（causally grounded retrieval）”为前沿方向——语义相似回答“像什么”，但调试场景需检索“什么导致了此”，需融合语义相似、时间排序、因果图遍历与反事实相关性，建议在向量索引上加轻量因果元数据层（写入时由 LLM 标注因果父节点）。
- **模态**: 以文本为主，同时讨论多模态/具身记忆：JARVIS-1 在 Minecraft 中融合视觉观察、文本计划与可执行技能；开放挑战章节将多模态与具身记忆（融合文本/视觉/音频/本体感觉/工具状态，新增空间记忆、实时延迟约束、跨模态检索）列为重要前沿。
- **冲突/矛盾处理**: 专门讨论矛盾处理（区别于遗忘）：工程章节列出陈旧/矛盾/漂移（staleness, contradictions, drift）应对四件套——时间版本化、来源归属（用户陈述 > 智能体推断）、矛盾检测（标记冲突待解决）、周期性巩固（合并重复、淘汰陈旧）；写-管理-读中的管理算子 U 显式包含“解决矛盾”；Agentic Memory 的 update 操作承担冲突更新。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)
- 过度个性化/记忆安全风险 (`over_personalization_risk`)
- token成本/延迟证据 (`token_cost_latency_evidence`)
- 发表venue (`venue`)


<a id="h4-from-storage-to-experience从存储到经验llm-智能体记忆机制演化综述提出-storagereflectionexperience-三阶段演化框架"></a>

### H4 From Storage to Experience

*From Storage to Experience（《从存储到经验》——LLM 智能体记忆机制演化综述；提出 Storage→Reflection→Experience 三阶段演化框架）*


**基本信息 / Provenance**

- **年份**: 2026（arXiv 预印本 2605.06716，首次公开日 2026-05-07；另有 Preprints.org 版本 2026-01）
- **作者/机构**: Jinghao Luo（罗景昊，华南师范大学，共同一作）、Yuchen Tian（田雨辰，香港浸会大学，共同一作兼项目负责人）、Chuxue Cao（香港科技大学）、Ziyang Luo、Hongzhan Lin、Chuyi Kong、Jing Ma（马晶，香港浸会大学，通讯作者）、Kaixin Li（新加坡国立大学）、Ruichao Yang（北京科技大学）等 9 位作者；主要单位为香港浸会大学（HKBU）计算机系 Jing Ma 课题组，合作单位含华南师大、港科大、新国大、北京科技大。
- **发表venue**: ACL 2026 Findings（已录用）；同时入选 ICLR 2026 Workshop on Memory for LLM-Based Agentic Systems（MemAgents，2026-04 海报）。属学术综述（Review）。arXiv 预印本（cs.AI/cs.CL）。
- **论文链接**: https://arxiv.org/abs/2605.06716
- **代码链接**: https://github.com/FeishuLuo/Evolving-LLM-Agent-Memory-Survey（官方论文资源/paper-list 仓库，收录 140+ 篇论文与 40+ 个 benchmark，MIT 许可，约 25 stars，截至 2026-06）
- **引用数**: 约 6 次引用（Semantic Scholar，CorpusId 288111157，截至调研日）；参考文献 216 篇。

**记忆分类 / Taxonomy**

- **记忆类型**: 作为综述，覆盖 CoALA 全部记忆类型。其分类不直接套用 episodic/semantic/procedural/working，而是按「信息抽象与认知加工程度」分三层：Storage（原始轨迹保存，近似情景/工作记忆的逐字记录）、Reflection（轨迹精炼，反思得到的语义化记忆单元）、Experience（跨轨迹抽象出的通用规则/技能，近似程序性+语义记忆的高层归纳）。论文还在未来方向中专门强调「工作记忆（working memory）」的组织是核心瓶颈。
- **记忆结构**: 综述层面归纳了三类存储结构（Storage 阶段）：线性（Linear，按时间排序的 token 流，FIFO）、向量（Vector，高维嵌入空间）、结构化（Structured，含关系表/层级分层/实体-关系图三种子型）。Reflection 阶段产出独立的精炼记忆单元 m′；Experience 阶段产出与具体场景脱钩的通用规则集 K（可为自然语言策略、可执行实体/代码、可演化技能库，或内化进模型参数/隐变量）。
- **存储后端**: 综述覆盖各类后端：上下文窗口（线性/注意力机制扩展，如 StreamingLLM、InfLLM）、向量库（语义检索/加权检索，如 Generative Agents、MemoryBank）、结构化后端（关系数据库 ChatDB/DB-GPT、分层架构 MemGPT/MemOS、知识图谱 AriGraph/GraphReader），以及参数化后端（将经验内化进模型权重，如 Titans、AgentRefine、Agent Lightning）。本身为综述无单一实现。
- **持久化**: 综述同时讨论三种持久化形态并将其映射到演化阶段：in-context（短暂上下文，Storage 早期线性方案）、external（外部持久化存储，Storage 向量/结构化与 Reflection 阶段主流）、parametric（参数化，Experience 阶段的 Implicit/Hybrid 经验，把交互历史内化进模型权重或隐层潜变量）。论文将 LLM 记忆 M 形式化定义为「连接冻结参数知识 θ 与不断演化的环境动态的外部化仓库」。

**核心机制 / Mechanisms**

- **写入/编码**: 综述按三阶段刻画写入/编码的演化：Storage 阶段以最小变换保真记录轨迹 τ=⟨(o_1,a_1),…,(o_T,a_T)⟩，原始仓库 M_raw={τ_i}，保持记忆条目与执行轨迹一一对应（线性逐字流、向量嵌入、结构化三元组）。Reflection 阶段把写入建模为语义变换映射 F_ref:T→S，对完成的轨迹 τ_i 生成精炼单元 m′_i=F_ref(τ_i|φ)（φ 为评估准则），即注入批判/纠错洞见，注重「质量密度」而非原始保真度，并以 M←M∪{m′_i} 并回仓库。Experience 阶段把写入建模为跨轨迹归纳算子 F_exp：先选出拓扑相似轨迹子集 T_batch={τ_i|Sim(τ_i,τ_j)>ε}，再压缩为通用规则集 K=F_exp(T_batch) 且 |K|≪Σ|τ|，遵循最小描述长度（MDL）原则。
- **检索机制**: 综述系统梳理检索机制随阶段的演化：Storage 阶段从语义近邻检索（基于嵌入相似度，Larimar/MemLong）发展到「加权检索」——综合时间衰减与重要性评分（Generative Agents 的 recency·importance·relevance 评分、MemoryBank 借鉴 Ebbinghaus 遗忘曲线），结构化阶段支持多跳精确检索与图遍历。论文在未来方向「主动记忆感知（Active Memory Perception）」中批评当前多为被动触发式无差别检索，会引入无关/过时记忆干扰推理，倡导用自主检索控制器按需调用、自主判断是否需要以及需要何种记忆。Experience 阶段的关键转变是：抽象出的规则 K 作为「策略先验（policy prior）」可直接用于未见场景，无需轨迹级匹配检索，从而大幅降低检索/推理开销。
- **反思/巩固**: 这是综述三阶段框架的核心枢纽（Reflection 阶段，§4.2）。论文将反思定义为把被动记录者转变为主动批判者的过程，利用反馈信号对历史轨迹做纠错与去噪，提升记忆仓库质量密度。归纳为三种范式：（1）Introspection（内省）——把智能体当作自主批判者，仅用模型内部知识纠错（轨迹纠错、记忆生命周期维护、长轨迹压缩蒸馏，代表 Reflexion、Think-in-Memory），优点是无需外部反馈，风险是强化自身偏见；（2）Environment（环境）——以执行结果/真实世界反馈为锚点校准世界模型、优化行为策略（CLIN、DEAL），适应动态环境但受奖励稀疏/设定模糊困扰；（3）Coordination（协调）——多智能体集体反思，借角色分工与共识降低幻觉、丰富视角（MIRIX、多智能体 Reflexion），代价是通信开销与记忆冲突。反思与经验的本质区别：反思是轨迹内变换 F_ref(τ_i|φ)=m′_i 仍绑定原任务上下文，经验是跨轨迹归纳 F_exp(T_batch)=K 与场景脱钩。
- **遗忘/更新**: 综述把「主动管理/遗忘/更新」识别为 Storage→Reflection 演化的关键驱动力之一（由动态环境中知识的时间有效性催生）：过时知识常无明显征兆地失效，故需引入时间感知、衰减策略与更灵活检索（Zep 时间知识图谱、MemoryBank 的 Ebbinghaus 衰减）。论文同时援引近期研究指出「记忆无限制扩张反而有害」——错误会在记忆系统中传播污染学习，因此需要更策略化的增删（ADD/DELETE）策略；收录了 OBLIVION（衰减驱动激活）、Adaptive Memory Admission Control、SSGM 等专门的遗忘/准入治理工作。
- **经验回放 (核心主题)**: 这是综述的核心主题（Experience 阶段 §4.3 与 §5「Transformative Experience」）。论文指出 LLM 智能体常有「过度跟随成功轨迹」的倾向，未经抽象的纠错轨迹仍会因上下文微小偏移而出错；因此 Experience 阶段不重放原始轨迹，而是隔离相似轨迹、剥离具体上下文，抽取「通用启发式智慧（universal heuristic wisdom）」并压缩仓库以泛化到未知环境。复用形态分三类：Explicit（显式，人类可读可编辑的自然语言策略/可执行实体/可演化技能库，代表 Agent Workflow Memory、ReasoningBank、Voyager 式技能）、Implicit（隐式，内化进模型参数或隐层潜变量，近零检索开销）、Hybrid（混合，「积累-内化」动态循环，显式经验作为高容量缓存周期性蒸馏进权重）。§5 进一步把「主动探索↔跨轨迹抽象」的反馈闭环确立为驱动智能体自主持续进化的中央引擎。

**学习维度 / Learning**

- **学习范式**: 综述同时覆盖三种范式并将其映射到 Experience 阶段的三种经验形态：非参数化（non-parametric，提示层/上下文注入，对应 Explicit 显式经验）、参数化（parametric，梯度更新/微调/RL，对应 Implicit 隐式经验，将经验内化进权重）、以及混合（hybrid，对应 Hybrid 经验，显式缓存周期性内化进参数）。论文在 Limitations 中明确：Experience 阶段（尤其 Implicit）在技术上与微调、强化学习、元学习交叉，但本框架不把经验定位为全新学习范式，而强调这些既有技术如何被部署于「以记忆为中心」的智能体架构中，充当交互轨迹与参数更新之间的关键中介。
- **失败学习 (核心主题)**: 综述将「从失败中学习」贯穿于 Reflection 与 Experience 两阶段。Reflection 阶段的 Introspection 范式以轨迹纠错（error rectification）为核心，环境反思用真实执行失败信号校准策略。Experience 阶段的「跨轨迹抽象」机制中专门列出 contrastive induction（对比归纳）——利用成功与失败轨迹的对立来精确刻画策略边界（代表工作如 He et al. 2024 等）。论文 §5.1「主动探索」也强调以失败/反馈作为内在驱动信号。作为综述其本身不报告失败学习的定量收益，而是综述并归类各代表系统（如 Reflexion、ReasoningBank 等）的失败利用方式。
- **技能/程序归纳**: 是，且为综述重点。Experience 阶段的 Explicit 子类专门讨论「Procedural Primitives（程序性原语）」与可演化技能库：把累积轨迹蒸馏为可执行实体或程序函数，耦合「归纳-复用-精炼」的生命周期（代表 Agent Workflow Memory、Inducing Programmatic Skills、MemP、PolySkill、CASCADE 等）。跨轨迹抽象机制中亦含「将复发行为模式封装为可复用程序函数（利用代码组合性）」这一路径。
- **在线 vs 离线**: 综述两者皆覆盖。Storage/Reflection 多为在线（部署时按轨迹增量构建与精炼）；Experience 阶段既含在线测试时学习/自我进化（test-time learning，如 ReasoningBank、Evo-Memory），也含离线批量的参数内化（对成功轨迹批量微调/RL，如 AgentRefine、Agent Lightning）。论文将「持续学习（continual learning）」列为推动演化的终极驱动力，强调把原始记忆簇转化为可跨场景复用的经验。

**评测 / Evaluation**

- **任务领域**: 综述横跨多领域：网页导航/网页智能体（WebArena、Mind2Web、WebChoreArena）、长程对话/多会话个性化对话（LoCoMo、PerLTQA、多会话个性化）、多跳问答与长上下文理解（HotpotQA、LongBench、RULER、BABILong）、软件工程/代码（SWE-Exp、LoCoBench-Agent）、多智能体系统、具身/多模态（多模态大海捞针、视频推理），以及终身学习与游戏。综述本身不在单一领域评测。
- **基准**: 综述系统整理了 40+ 个 benchmark 并按三阶段归类。Storage 阶段：HotpotQA、LongBench/v2、RULER、BABILong、HELMET、MemoryBank、LoCoBench-Agent、AgentLongBench 等。Reflection 阶段：LoCoMo（Very Long-Term Conversational Memory）、PerLTQA、Minerva、HaluMem、ConvoMem、StoryBench、PersonaMem-v2 等。Experience 阶段：StreamBench、LifelongAgentBench、MEMTRACK、MemoryBench、Evo-Memory、MemoryArena、AMA-Bench 等。论文特别指出 Experience 阶段（抽象/泛化能力评测）的数据集严重不足，是其呼吁的未来方向。
- **对比基线**: 不适用（综述类工作）。论文的「对比对象」是既有综述：批评 Zhang et al. 2024 仅做工程模块分类、未阐明技术变革逻辑；批评 Hu et al. 2025 局限于静态功能分类、未揭示动态演化原理；并区别于其他记忆综述（Wu et al.、Du et al.、Cao et al.）。其差异化定位是首个以「动态演化」视角统一「操作系统工程派」与「认知科学派」两套割裂范式。

**分析 / Analysis**

- **关键创新**: 提出首个以「动态演化」为中心、统一弥合「操作系统工程」与「认知科学」两大割裂范式的 LLM 智能体记忆框架，将记忆机制发展形式化为三个递进阶段——Storage（轨迹保存）→Reflection（轨迹精炼）→Experience（轨迹抽象），并以「Why（演化驱动力）-How（演化路径）-What（Experience 带来的变革）」三段论组织；尤其首次系统刻画 Experience 阶段的两大变革性机制：主动探索（active/proactive exploration）与跨轨迹抽象（cross-trajectory abstraction），给出从原始轨迹到通用策略先验 K 的形式化（MDL 压缩）。
- **局限**: 论文自陈三点局限：（1）缺乏直接定量比较——采用定性分析框架，无跨阶段统一 benchmark，且基础模型/环境/prompt 差异使数值横评易误导；（2）与既有学习范式的关系——Experience（尤其 Implicit）与微调/RL/元学习技术交叉，本框架不主张其为全新范式，仅强调其在记忆中心架构中的中介作用；（3）时间覆盖与近因偏差——领域 2024–2025 爆发、Experience 阶段 2025 下半年才成形，早期奠基工作可能未获相称关注，部分纳入的近期预印本尚未经同行评审。此外作为综述无原创实现与可复现实验。
- **与其他工作关系**: 属本研究「H. 综述（Surveys）」簇，与本主题高度契合：其 Storage→Reflection→Experience 演化框架几乎等价于本研究关注的「记忆→反思→经验复用」主线。它把本库内多个被独立调研的系统精确归位到框架坐标中——例如把 Reflexion（A1）、CLIN（A3）归为 Reflection/Introspection 与 Environment；把 ReasoningBank（A6）同时列入 Explicit Experience 与 Hybrid Experience 的代表，并在 Reflection-vs-Experience 对比表中以 Reflexion/CLIN/AgentFold 代表反思、以 FLEX/MemSkill/SkillRL 代表经验；把 Generative Agents（B1）、MemoryBank（B2）归为 Storage 的加权检索，MemGPT（B3）归为分层结构化存储。与 H 簇内其它综述（如 Graph-based Agent Memory 综述，2602.05665）的区别在于：后者按图结构生命周期组织，本文按「认知抽象层级的动态演化」组织，更强调从存储到经验的能力跃迁与 Experience 阶段的探索-抽象闭环。
- **可复现性**: 作为综述无实验可复现性问题，但配套资源开放度好：官方 GitHub 仓库 FeishuLuo/Evolving-LLM-Agent-Memory-Survey 持续维护，分三阶段整理 140+ 篇论文与 40+ 个 benchmark 并附 arXiv 链接，MIT 许可，欢迎社区 PR 贡献；论文已被 ACL 2026 Findings 录用并入选 ICLR 2026 MemAgents Workshop。社区采用信号尚早（约 25 stars、约 6 次引用，因新近发表）。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 综述明确将「学习型记忆控制 vs 启发式流程」作为前沿分水岭来讨论：在 Reflection/Environment 的「Decision Optimization」与 Experience 的「Parameter Internalization」中专门收录用 RL 学习记忆管理策略本身的工作（Memory-R1 用 RL 管理与利用记忆、Memory-Driven Self-Improvement、Agent Lightning、Group-in-Group Policy Optimization 等）。在未来方向「主动记忆感知」中倡导从被动触发式检索转向自主检索控制器（learned controller），即学习「何时/何种记忆需被引入」的策略，呼应 2025–26 这一代际转变。
- **记忆主体**: 综述同时覆盖「用户中心」与「智能体中心」两类记忆并不偏废，但其演化框架（尤其 Experience 阶段「让智能体从自身经验中自我进化」）更偏智能体中心（agent-centric，记忆自身交互经验以自我提升，代表 ReasoningBank/Voyager 式技能）；同时也综述大量用户中心记忆（user-centric，记住用户信息做个性化，代表 Mem0/Zep 及 LoCoMo/PerLTQA/PersonaMem 等个性化对话 benchmark）。Reflection 阶段的个性化对话数据集与 Experience 阶段的自我进化系统分别对应这两条线。
- **多智能体记忆**: 综述专门覆盖多智能体记忆，并将「分布式共享记忆（Distributed Shared Memory）」列为重点未来方向。Reflection 阶段的 Coordination 范式即多智能体集体反思（含记忆冲突、通信开销问题，代表 MIRIX、G-Memory、多智能体 Reflexion、LEGOMem 模块化程序记忆）；论文指出当前共享记忆多依赖显式对话通信，受带宽瓶颈与噪声困扰，倡导构建「共识记忆系统（consensus memory）」实现个体视角与集体知识的高效同步，迈向「组织（Organizations）」级智能体社会。
- **时序推理支持**: 综述将「知识的时间有效性（Temporal Validity）」与「环境的因果结构（Causal Structure）」作为『动态环境』驱动力的两大支柱专章论述（§3.2，附 Figure 2）：动态环境中知识多为条件性而非永恒有效，过时知识常无征兆地失效，催生时间感知、衰减策略与因果依赖建模（跨时间步构建复杂因果依赖、构建因果一致的内部世界）。收录 Zep（时间知识图谱）等显式时间建模工作，并将相关评测（如 MEMTRACK 状态跟踪）归入 Experience 阶段。
- **模态**: 综述以文本为主，但将「多模态记忆（Multimodal Memory）」列为关键未来突破方向（§6，附录 §C 详述）：倡导把视觉感知状态、语言推理过程等多种模态融合进具备统一时序与语义的记忆单元，尤其面向具身智能的世界模型完整性。收录 MIRIX（截图）、ToolMem、GEMS（智能体原生多模态生成）、ReMem-VLA（视觉-语言-动作）、视频推理记忆等多模态工作，以及『多模态大海捞针』等多模态 benchmark。
- **过度个性化/记忆安全风险**: 综述触及记忆安全/负面维度但着墨有限。它援引近期研究强调「更多记忆并非总是更好」——记忆无限制扩张会让错误在系统内传播、污染学习效能（Xiong et al. 2025、Srivastava & He 2025），并在『主动记忆感知』方向指出持续检索无关/过时记忆会破坏推理连贯性。收录 HaluMem（记忆系统幻觉）等评测，但对隐私治理、谄媚/侵入性记忆、过度个性化等专门安全维度（如 OP-Bench 类）未做系统展开。
- **冲突/矛盾处理**: 综述将冲突/矛盾处理纳入 Reflection 阶段的「动态维护（Dynamic Maintenance）」与遗忘/更新讨论：动态环境下过时且语义仍相关的知识需被识别与更新（Zep、Mem0 的记忆生命周期管理），多智能体协调反思中明确指出「记忆冲突（memory conflicts）」是其代价之一。收录 MEMTRACK（多平台状态跟踪）等评测矛盾/状态一致性的工作。区别于单纯遗忘，强调入库后对相互矛盾事实的检测与合并仍是开放挑战。
- **token成本/延迟证据**: 综述把效率/成本作为演化驱动力（『记忆存储约束』『连续学习』）与 Storage 阶段技术主线之一来梳理：线性存储阶段大量工作做信息稀疏化/prompt 压缩以加速推理（H2O、LLMLingua、Quest、InfLLM、LightThinker++、MemBoost 等），结构化分层架构（MemGPT/MemOS）权衡容量与检索速度。Experience 阶段抽象出的通用规则 K 满足 |K|≪Σ|τ|（MDL 压缩），从根本上降低检索与推理开销（隐式经验近零检索开销）。收录『Beyond the Context Window: Cost-Performance Analysis of Fact-Based Memory vs Long-Context』等专门成本-性能权衡 benchmark。作为综述其本身不给出统一的 token/延迟节省百分比。

**不确定字段 / Uncertain**

- 报告增益 (`reported_gains`)


<a id="h5-graph-based-agent-memory-taxonomy-techniques-and-applications基于图的智能体记忆综述配套开源资源库-awesome-graphmemory"></a>

### H5 Graph-based Agent Memory

*Graph-based Agent Memory: Taxonomy, Techniques, and Applications（基于图的智能体记忆综述；配套开源资源库 Awesome-GraphMemory）*


**基本信息 / Provenance**

- **年份**: 2026（arXiv 预印本，2026-02-05 首次公开，v1）
- **作者/机构**: Chang Yang、Chuang Zhou、Yilin Xiao（三人共同一作）等，共 18 位作者；通讯作者为 Qinggang Zhang、Huachi Zhou、Shengyuan Chen。主要单位为香港理工大学（The Hong Kong Polytechnic University，DEEP-PolyU 课题组），合作单位包括厦门大学（Jinsong Su 等）、新加坡管理大学（Xinrun Wang）、吉林大学（Yi Chang），并有佐治亚大学（Ninghao Liu）参与。
- **论文链接**: https://arxiv.org/abs/2602.05665
- **代码链接**: https://github.com/DEEP-PolyU/Awesome-GraphMemory（官方维护的 Awesome 资源清单，收录论文/基准/开源项目，约 284 stars、16 forks，截至 2026-06；持续更新）

**记忆分类 / Taxonomy**

- **记忆类型**: 作为综述，覆盖 CoALA 全部记忆类型并提出自身分类：按时间分为短期（short-term：当前对话上下文、活动推理轨迹）与长期（long-term：跨会话持久知识/历史/用户偏好）；按认知结构分为语义（semantic）、程序（procedural）、联想（associative）记忆；按功能角色提出核心二分法——知识记忆（Knowledge Memory，被动静态，类似内部参考书/教科书，存事实、规则、流程）与经验记忆（Experience Memory，主动动态，类似个人日志，存交互历史、执行日志、试错轨迹与反馈）。
- **记忆结构**: 核心立场是「图结构记忆」作为统一视角，对比传统的非结构化记忆（线性/缓冲式、向量式、键值/日志式——被批评为扁平、缺乏关系上下文与层级）。在存储（Section V）上系统归纳五类图结构：(1) 知识图谱结构（KG，实体-关系三元组）；(2) 层级记忆结构（hierarchical，多粒度/树状，如 G-Memory、SGMem、GraphRAG 社区层级）；(3) 时序图结构（temporal graph，带双时间戳/事件序，如 Zep、MemoTime）；(4) 超图结构（hypergraph，HyperGraphRAG/HyperG，建模高阶 n 元关系）；(5) 混合图架构（hybrid，如 MAGMA 多图、Optimus-1、KG-Agent）。
- **存储后端**: 综述层面不绑定单一后端，梳理了图数据库（Neo4j 等）、向量库、混合检索运行时等实现。开源项目章节列举典型后端/框架：Graphiti（时序感知 KG）、LangMem（LangChain 长期记忆 SDK）、Mem0、MemMachine（多层记忆）、O-Mem、OpenMemory、Memori、Memary、Omnigraph（S3-native、Rust，融合图遍历+向量+BM25）、taOSmd（append-only 转录+时序 KG+混合向量/BM25）等。
- **持久化**: 覆盖全谱：以外部持久化的图记忆（durable external graph store，支持跨会话/跨任务持久保留）为主线，同时讨论短期/上下文内（ephemeral，工作记忆）与可被巩固进长期存储的过渡；亦提及参数化方法（如 MemLLM 微调读写记忆）。综述强调长期图记忆支撑跨情景连续性、迁移学习与个性化。

**核心机制 / Mechanisms**

- **写入/编码**: 对应「记忆提取（Memory Extraction，Section IV）」阶段，定义为把原始非结构化观察 o_t 转化为结构化记忆单元 m。按输入数据模态归纳三类提取：(1) 文本数据（textual，从对话/文档抽取实体-关系三元组、社区感知 KG，如 PersonaAgent-GraphRAG、HiAgent 层级工作记忆）；(2) 序列数据（sequential，从轨迹/交互序列提取，如 Reflexion 语言化反思、Mem-α 用 RL 学习记忆构建）；(3) 多模态数据（multimodal，从视觉/音频/动作提取，如 MemoryVLA、Optimus-1 混合多模态记忆）。提取手段涵盖 LLM 抽取三元组、摘要式蒸馏、结构化解析等，强调从扁平文本到图节点/边的结构化转换。形式上记忆有 Write/Read/Update/Delete 四个原子操作。
- **检索机制**: 对应「记忆检索（Memory Retrieval，Section VI）」，系统化为「六类基础算子 + 三类增强策略」的检索流水线（查询预处理→检索→剪枝/选择→注入下游推理）。六类基础算子归入三种范式：语义检索（semantic：①相似度算子 similarity-based，对文本/多模态嵌入做模糊匹配，常作候选生成器）；结构化检索（structured：②规则算子 rule-based，③时序算子 temporal-based，④图算子 graph-based——在 KG/层级/时序图/超图上做可验证、可解释、可追溯的图遍历/约束查询，含 N-hop 邻居扩展、PPR 个性化 PageRank 等）；策略式检索（policy-based：⑤强化学习算子 RL-based，⑥智能体算子 agent-based，把检索建模为序贯决策——决定查询哪类记忆、用哪些算子、分配计算预算、何时停止）。三类增强策略：多轮检索（multi-round）、后检索（post-retrieval 重排/精炼）、混合源检索（hybrid-source，协调内部记忆与外部资源）。常见组合：语义锚定→结构化扩展→策略式停止与剪枝。
- **反思/巩固**: 对应「记忆演化（Memory Evolution，Section VII）」，定义为对记忆进行精炼的后处理阶段，分两条路径：(1) 内部自演化（Internal Self-Evolving）——巩固（consolidation）、抽象（abstraction）、自组织等，把原始经验提炼为更高层知识，代表工作如 Zep、Nemori（受认知科学启发的自组织记忆）、GraphRAG（local→global 摘要）、Reflexion、Think-on-Graph、MemoryBank、MemGPT/Memory OS（操作系统式分层管理）、HippoRAG（从 RAG 到非参数持续学习）；(2) 外部自探索（External Self-Exploration）——通过新的环境反馈驱动记忆更新，如 ExpeL（经验学习者）、Memory-R1（用 RL 管理与利用记忆）、MemEvolve（记忆系统的元演化/meta-evolution）、AgentEvolver、Beyond Static Summarization（主动记忆提取）。综述强调演化使记忆保持时效、相关并结构化最优。
- **遗忘/更新**: 综述将更新/遗忘纳入「演化」阶段与原子操作（Update/Delete）；讨论图记忆的去重、合并、冲突消解与失效（invalidation，如 taOSmd 让纠正后的事实通过失效机制取代旧事实、Zep 的双时间事实失效）。同时把「真正的遗忘/可学习的遗忘」列为开放挑战。
- **经验回放 (核心主题)**: 作为综述的核心主题之一（经验记忆这一支）。系统梳理智能体如何复用过去轨迹/经验以改进未来行为：从原始轨迹保存，到反思蒸馏（Reflexion）、经验学习（ExpeL、Agent KB 跨域经验复用）、技能/工作流归纳，再到前沿的跨轨迹抽象与自演化（From Storage to Experience 三阶段：Storage→Reflection→Experience）。图结构使经验以情景节点+语义关系组织，支持多跳复用与策略迁移。综述将经验记忆定位为「主动、动态、个性化」的学习基础，区别于被动静态的知识记忆。

**学习维度 / Learning**

- **学习范式**: 综述同时覆盖非参数化（in-context/prompt 层，绝大多数图记忆系统）、参数化（gradient/微调，如 MemLLM 微调读写记忆、Mem-α/Memory-R1 用 RL 训练记忆策略）与混合范式；并以「记忆生命周期（提取-存储-检索-演化）」为统一框架横向组织这些范式，强调自演化（self-evolving）智能体记忆是发展方向。
- **失败学习 (核心主题)**: 在经验记忆与外部自探索（External Self-Exploration）部分讨论从试错/失败反馈中学习：经验记忆显式记录「试错轨迹与反馈」（如某抓取流程在湿杯子上失败、某均值回归交易亏损）；Reflexion 的语言化强化、ExpeL 的经验学习、Memory-R1 的 RL 反馈、AgentEvolver 等均涉及从结果（含失败）中更新记忆与策略。综述把「从环境反馈/失败中持续演化」列为驱动记忆演化的关键动力，但作为综述未做单独的失败学习消融实验。
- **技能/程序归纳**: 是（在分类与应用层面覆盖）。程序记忆（Procedural Memory）被定义为编码技能、例程与不可变规则的「how-to」知识（如标准操作流程、游戏规则），使智能体在标准条件下自动执行复杂任务；并在游戏、机器人/具身、代码等应用中讨论从经验归纳可复用技能/工作流，以及在图中以节点/子图表示并被检索调用。
- **在线 vs 离线**: 两者皆有。综述以「记忆生命周期」组织在线（部署期逐情景增量构建与演化，如对话/游戏/具身）与离线（对轨迹语料批量构建图、社区检测、KG 抽取，如 GraphRAG）两类构建方式，并讨论持续学习（Continual）基准下的在线/测试时适应。

**评测 / Evaluation**

- **任务领域**: 覆盖广泛：对话/多会话交互（conversational）、代码/软件工程（code agents）、推荐系统（recommender）、金融交易（financial）、游戏（game，开放世界如 Minecraft）、机器人与具身（robotics & embodied）、医疗健康（medical/health）、科学发现（science agents）。基准章节进一步按场景分七类：Interaction（多轮/跨会话对话）、Personalization（用户画像/偏好）、Web（长程浏览/多步在线任务）、LongContext（长文档理解与检索）、Continual（终身/测试时学习）、Environments（具身与交互世界）、Tool/Gen（工具使用与工作流执行）。
- **基准**: 汇总数十个基准（Table I），按七类组织：交互类 LoCoMo、LongMemEval、MemoryAgentBench、MEMTRACK、MADial-Bench、MemSim、MSC、MMRC、MemBench、DialSim、RealMem 等；个性化类 PersonaMem、PerLTQA、MemoryBank、MPR、PrefEval、LOCCO；Web 类 WebArena、WebShop、MT-Mind2Web、WebChoreArena、MMInA；长上下文类 NQ、TriviaQA、PopQA、HotpotQA、2WikiMultihopQA、MuSiQue、LongBench(v2)、RULER、BABILong、MM-Needle；持续学习类 LifelongAgentBench、StreamBench、Evo-Memory、MemoryBench；环境/具身类 Ego4D、EgoLife、ALFWorld、BabyAI、ScienceWorld、AgentGym、AgentBoard；工具/生成类 SWE-Bench、GAIA、xBench-DS、ToolBench、GenAI-Bench。
- **报告增益**: 本文为综述（survey），不提出新方法、不报告自有实验的统一量化增益；它系统整理了各代表性图记忆系统在上述基准上相对无记忆/全上下文/RAG/向量记忆等基线的提升与效率优势（如 Zep、Mem0、G-Memory 等在 LoCoMo/LongMemEval 上的准确率提升与 token/延迟节省），但不给出单一「头条数字」。其量化贡献体现在：覆盖 201 篇参考文献、提出多维分类、整理 40+ 基准与十余个开源项目的对照表。
- **对比基线**: 作为综述的「对照对象」是传统/非图记忆范式：线性/缓冲式记忆、向量式记忆（dense embedding + 向量库相似度检索）、键值/日志式记忆，以及标准 RAG。综述论证图结构记忆相对这些扁平/隐式结构在关系建模、层级组织、可验证多跳检索与可解释性上的优势。

**分析 / Analysis**

- **关键创新**: 首个从「图结构」统一视角系统综述智能体记忆的工作：(1) 提出多维分类（短期/长期、知识/经验、非结构化/结构化），并以图记忆作为统一实现视角；(2) 按记忆生命周期（提取→存储→检索→演化）系统拆解关键技术，尤其把检索归纳为「六算子三策略」、把存储归纳为「KG/层级/时序/超图/混合」五类图结构、把演化分为内部自演化与外部自探索；(3) 系统整理支持自演化记忆的开源库与基准，并覆盖八大应用域；(4) 提炼七大挑战与未来方向。配套维护持续更新的 Awesome-GraphMemory 资源库。
- **局限**: （综述自陈的领域挑战即其讨论的局限）：(1) 记忆图质量——缺乏显式评估图记忆内在质量（结构/语义/时序/操作多维）的指标；(2) 可扩展性与效率——图操作常呈二次或更差复杂度，长期累积形成计算瓶颈，需压缩/增量更新/近似检索/硬件加速；(3) 隐私与安全——关系结构易经推理攻击泄露隐私，且面临记忆投毒/对抗攻击；(4) 动态 schema 学习与知识迁移——现有图 schema 多为领域专用、复用性差；(5) 可解释性与可信；(6) 理论基础薄弱（缺完整性/一致性保证、复杂度界、记忆扩展定律）；(7) 多智能体记忆协调。作为综述本身亦受限于：以图视角组织、对非图/参数化记忆覆盖相对较浅，且为快速演进领域的即时快照（截至 2026 初）。
- **与其他工作关系**: 属于本研究「H. 综述」簇，是少数以「图结构」为核心组织维度的智能体记忆综述。它继承并细化 CoALA（语义/情景/程序/工作记忆认知分类）与既往记忆综述（如 A Survey on the Memory Mechanism of LLM-based Agents (TOIS'25)、From Human Memory to AI Memory、The AI Hippocampus、Memory in the Age of AI Agents）。同期/后续综述常将其作为「图导向生命周期（Graph-oriented lifecycle）」的代表加以对照：如 Anatomy of Agentic Memory（2602.19320）在分类对照表中标注本文为 graph-oriented lifecycle、并指出其偏「理论到实践」组织而实证分析较少；From Storage to Experience（2605.06716）提出 Storage→Reflection→Experience 演化框架与之互补；Toward a Theory of Hierarchical Memory（2603.21564）从层级记忆理论角度延伸。其覆盖的代表性系统横跨本研究多个簇：用户中心记忆 Mem0/Zep/Graphiti、智能体中心经验记忆 Reflexion/ExpeL/Voyager 式技能、学习型记忆控制 Mem-α/Memory-R1、多智能体记忆 G-Memory、时序图 Zep/MemoTime、超图 HyperGraphRAG。
- **可复现性**: 作为综述，「可复现性」体现为资源可得性与社区采用：官方维护开源资源库 Awesome-GraphMemory（约 284 stars、16 forks，持续更新），系统收录论文、40+ 公开基准（多数附 Paper+Repo/Website 链接）与十余个开源记忆项目（LangMem、Graphiti、Mem0、MemMachine 等），便于研究者复现所综述系统并选型；论文本身为开放获取（arXiv，含 HTML 版）。短期引用约 10 次，已被多篇 2026 综述/方法引用，呈现一定影响力。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 综述明确把「学习型记忆控制」作为前沿分水岭加以覆盖：在检索层提出策略式检索范式（RL 算子、智能体算子，把检索/停止/预算建模为序贯决策），在演化层与提取层收录用 RL/训练学习记忆管理策略的代表工作 Memory-R1（RL 管理与利用记忆）、Mem-α（RL 学习记忆构建）、MemEvolve（记忆系统元演化）、AgentEvolver。与启发式流水线方法对照呈现，体现 2025-26 从启发式到可学习记忆控制的代际转变。
- **记忆主体**: 兼顾两端，并以「知识记忆 vs 经验记忆」二分法作为核心组织轴：用户中心（user-centric，记住用户信息做个性化——对应 Personalization 基准与 Mem0/Zep/PersonaMem/对话与推荐应用）与智能体中心（agent-centric，记住自身经验做自我进化——对应 Experience Memory、ExpeL/Reflexion/Voyager 式技能、游戏/具身/科学应用）。综述指出静态知识记忆与动态经验记忆的协同使智能体既理解世界规则又适应个性化交互。
- **多智能体记忆**: 覆盖。综述讨论多智能体共享/路由记忆（如 G-Memory: Tracing Hierarchical Memory for Multi-Agent Systems 的层级洞见/查询/交互分层、去中心化生成智能体的自适应层级 KG 协同规划），并把「多智能体系统中的记忆协调」（记忆同步、角色感知访问、通信受限下的可扩展协调、避免不一致更新导致的冲突决策）列为关键未来挑战。
- **时序推理支持**: 覆盖且专设一类。综述将「时序图结构」列为五大存储结构之一，并设「时序算子（temporal-based operator）」检索范式，讨论显式建模时间有效性/事件先后/事实时效窗口的系统：Zep（双时间知识图谱，episodic+semantic 层）、TReMu（神经符号时序推理）、MemoTime（记忆增强时序知识图谱）、RecallM（带时序理解的记忆）。时序推理被视为最难评估的能力之一。
- **模态**: 覆盖多模态。提取章节专列多模态数据（视觉/音频/动作，如 MemoryVLA、Optimus-1、Lip-Audio），应用涉及机器人/具身（视频+动作）与多模态对话；基准含 Text+Image（LoCoMo、MMRC、MMInA、MM-Needle、WebArena/WebShop 截图）与 Video+Audio（Ego4D、EgoLife）。但多数图记忆系统仍以文本为主。
- **过度个性化/记忆安全风险**: 部分覆盖（置于隐私与安全挑战下）。综述讨论个人助理需在个性化与敏感信息保护间权衡，图结构的关系模式可能经推理攻击泄露隐私，并面临记忆投毒/对抗注入污染智能体行为；提出差分隐私、联邦/端侧处理、安全多方计算、记忆内容校验/异常检测/审计等防护方向。对「更多记忆不一定更好」的过度个性化/谄媚/陈旧记忆等专门安全维度（如 OP-Bench 类）覆盖相对较浅。
- **冲突/矛盾处理**: 覆盖（在演化与更新讨论中）。综述涉及图记忆更新时的冲突/矛盾事实消解与去重合并，如 taOSmd 通过失效（invalidation）让纠正后的事实取代旧事实、Zep 的双时间失效机制处理过期事实，以及 Memory-R1 的 UPDATE 操作；MEMTRACK 基准被收录用于评测长期交互中的记忆一致性追踪。综述将一致性/完整性保证列为理论基础挑战。
- **token成本/延迟证据**: 作为综述未给出统一的自有效率数字，但整理并强调图记忆相对全上下文/向量记忆的效率优势（典型如 Mem0/Zep 报告的大幅 token 与 p95/延迟节省、层级与剪枝检索降低 token 成本），并在挑战部分指出图操作的二次或更差复杂度是主要效率瓶颈，呼吁记忆压缩、增量更新、近似检索与硬件/分布式加速以支撑百万级节点的快速访问。

**不确定字段 / Uncertain**

- 引用数 (`citations_approx`)
- 发表venue (`venue`)


<a id="h6-anatomy-of-agentic-memory-taxonomy-and-empirical-analysis-of-evaluation-and-system-limitations智能体记忆解剖评测与系统局限的分类学与实证分析这是一篇综述实证分析论文非具体记忆系统其核心立场是从评测有效性--系统局限的实证透镜审视智能体记忆提出-memory-augmented-generation-mag-记忆增强生成-四结构分类学并系统暴露基准饱和指标失配骨干模型敏感性与系统级开销四大痛点配套资源别名github-awesome-list-仓库-fredjiang0324anatomy-of-agentic-memory"></a>

### H6 《Anatomy of Agentic Memory

*《Anatomy of Agentic Memory: Taxonomy and Empirical Analysis of Evaluation and System Limitations》（智能体记忆解剖：评测与系统局限的分类学与实证分析）。这是一篇综述/实证分析论文（非具体记忆系统），其核心立场是从『评测有效性 + 系统局限』的实证透镜审视智能体记忆，提出 Memory-Augmented Generation (MAG, 记忆增强生成) 四结构分类学，并系统暴露基准饱和、指标失配、骨干模型敏感性与系统级开销四大『痛点』。配套资源别名：GitHub awesome-list 仓库 FredJiang0324/Anatomy-of-Agentic-Memory。*


**基本信息 / Provenance**

- **年份**: 2026（arXiv 预印本 v1：2026-02-22；存在 v2 修订版；属 cs.CL）
- **作者/机构**: Dongming Jiang（蒋东明，第一作者，德州大学达拉斯分校 UT Dallas）、Yi Li、Songtao Wei、Jinxin Yang、Dingyi Kang、Xu Hu、Feng Chen（均 UT Dallas）、Ayushi Kishore、Qiannan Li（加州大学戴维斯分校 UC Davis）、Alysa Zhao（德州农工大学 Texas A&M）、Bingzhe Li（李冰哲，通讯/资深作者，UT Dallas）。共 11 位作者。主要机构：德州大学达拉斯分校（主导）、加州大学戴维斯分校、德州农工大学。
- **发表venue**: arXiv 预印本（cs.CL，Computation and Language），2026 年 2 月；DOI: 10.48550/arXiv.2602.19320；DBLP: journals/corr/abs-2602-19320；Semantic Scholar 将其标注为 JournalArticle/Review 类型。截至核实未见正式会议/期刊发表记录。
- **论文链接**: https://arxiv.org/abs/2602.19320 （HTML 版 https://arxiv.org/html/2602.19320v2 ；DOI: https://doi.org/10.48550/arXiv.2602.19320）
- **代码链接**: https://github.com/FredJiang0324/Anatomy-of-Agentic-Memory （综述配套 awesome-list 论文清单仓库，按文中四结构分类学组织；核实时约 18 stars、0 forks、2 位贡献者，创建于 2026-02-20，持续更新至 2026-02；非可运行系统，而是持续维护的智能体记忆论文/分类编目）
- **引用数**: 约 10 次引用（Semantic Scholar 实时 API 核实值，paperId b11a1d01...；CorpusId 285973876；参考文献数 84）。作为 2026 年 2 月新近综述，影响力处于早期累积阶段。

**记忆分类 / Taxonomy**

- **记忆类型**: 作为综述/实证分析论文，本文不实现单一记忆类型，而是提出按『结构』而非 CoALA 功能维度划分的四类 MAG 分类学：(1) Lightweight Semantic（轻量语义记忆，独立文本单元向量化，top-k 相似检索）；(2) Entity-Centric and Personalized（实体中心与个性化记忆，围绕用户/任务/偏好的结构化记录）；(3) Episodic and Reflective（情景与反思记忆，按 episode 组织 + 周期性反思巩固，对应 CoALA episodic）；(4) Structured and Hierarchical（结构化与层级记忆，多层级或图结构）。文中亦覆盖 procedural/skill 记忆线（如 MemP、LEGOMem、MemSkill、TokMem 的程序记忆/技能复用），呼应任务锚点提及的『procedural-memory line』。
- **记忆结构**: 综述层面覆盖全谱系结构，并以『结构优先 (structure-first)』分类组织：轻量语义=向量空间中独立文本单元（append-only 或最小过滤，无显式结构关系）；实体中心=结构化记录/属性-值对/预定义 schema；情景反思=有界 episodic buffer + 反思总结产出的紧凑表示；结构化层级=知识图谱（节点/边捕捉语义、时序、因果、实体关系）与多层级存储（短期/情景/长期分层）。还细分子类如 token 级语义记忆、OS 启发分层记忆、策略优化记忆管理等。
- **存储后端**: 综述不绑定单一后端，编目覆盖：向量库（轻量语义记忆 top-k 检索）、图数据库/知识图谱（MAGMA 多图、Zep 双时序知识图谱、SGMem 句级图）、OS 启发的多层级分页存储（MemGPT、MemoryOS 三级层次、EverMemOS）、上下文窗口工作区（AgentFold、Context-Folding）、token 级潜在记忆/记忆 token（MemGen、TokMem）。实证实验中评测的六个系统为 AMem、MemoryOS、Nemori、MAGMA、SimpleMEM、MemSkill，骨干模型用 gpt-4o-mini 与 Qwen-2.5-3B。
- **持久化**: 综述系统刻画外部、非参数化、可写的持久记忆：定义 MAG 为在固定上下文窗口之外引入持久可写记忆 M_t，跨交互演化（存储/检索/更新）。明确区分本研究关注的『外部显式读写记忆 (external, non-parametric)』与参数化学习（修改权重 θ）——核心观点是记忆通过显式 read-write 操作 ψ(M_t; q_t) 而非更新 θ 影响行为（式 1、式 5）。持久形态按结构分为短期上下文工作区（单任务内）与跨会话长期外部存储（情景/语义/层级）。

**核心机制 / Mechanisms**

- **写入/编码**: 综述把写入编码形式化为记忆更新/写函数 M_{t+1}=Write(M_t, o_t, a_t, s_t)（式 8），并进一步显式化为记忆动作 u_t∈{store, update, summarize, link, evict, delete}，u_t=g(o_t,a_t,s_t)，M_{t+1}=T(M_t,u_t)（式 9），其中 g 可为规则、模型驱动或学习式策略（连接 RL 引导的记忆管理）。按四结构编目编码方式：轻量语义=embedding 向量化 append；实体中心=抽取实体-属性-值结构化记录（A-MEM 构建带 LLM 生成链接的知识 note）；情景反思=把交互记录入 buffer + 反思总结/蒸馏（MemP 蒸馏轨迹为程序抽象、Nemori 自组织情景）；结构化=抽取实体构建关系图（MAGMA 跨语义/时序/因果/实体图、Zep 双时序图）。实证关键发现：图/情景架构的结构化输出生成在弱骨干模型下格式错误率显著升高（Nemori 在 gpt-4o-mini 上 17.91%、Qwen-2.5-3B 上 30.38%），会『静默腐蚀 (silent failure)』长期记忆。
- **检索机制**: 综述把检索形式化为推理时召回：q_t=Query(o_t,s_t)（式 3），r_t=Read(M_t,q_t)（式 4），常见实例为 top-k 召回 r_t=TopK({m_i}; score(q_t,m_i), k)（式 6）。核心论点：智能体场景下理想『相关性』不是纯语义而是决策条件式 (decision-conditional)——提出理想检索目标 r_t*=argmax_r E[U(a_t|o_t,r,s_t)]（式 7），即选取最大化下游效用 U（任务成功/效率/鲁棒）的记忆；实际系统用相似搜索、学习式 reranker、多跳检索、planner 引导召回或检索策略来逼近。编目覆盖各结构检索方式：轻量=dense 相似 top-k；图结构=子图遍历/多跳推理；情景=retrieve-reflect-answer 闭环（MemR3）、时序感知检索策略（Memory-T1 用 GRPO 优化准确性/接地/时序一致）。
- **反思/巩固**: 综述把『原始经历→高层知识』的转化统一归入情景反思与巩固操作族，并将其作为四结构之一（Episodic and Reflective）专门刻画：周期性把 episode 经验通过总结/反思巩固为紧凑表示，平衡记忆容量与长程推理效用。编目代表：MemP 把轨迹蒸馏为可持续精炼/迁移的程序抽象；LEGOMem 为多智能体协调构建模块化角色感知程序记忆；TiMem 引入时序-层级记忆树做结构化巩固；ReasoningBank 把推理经验巩固为可复用推理记忆；ReadAgent 用 gist memory 反思超长上下文。综述作为分析者不提出新反思算法，而是分类对比既有反思/巩固/抽象范式，并在实证中指出巩固/维护开销是被忽视的系统级瓶颈（异步维护若滞后于用户交互会导致记忆陈旧/throughput collapse）。
- **遗忘/更新**: 综述把遗忘与更新归入显式记忆动作（store/update/summarize/link/evict/delete，式 9）。编目层面：OS 启发分层记忆做自适应遗忘与跨层巩固（MemGPT 分页、MemoryOS STM→LTM）；个性化记忆做冲突感知更新（EgoMem conflict-aware update、PAMU 用滑动窗口+移动平均跟踪演化偏好）；策略优化记忆把 store/update/consolidate/delete 当作可学习决策（AtomMem 分解为 CRUD 原子动作学习任务对齐控制、Memory-R1 用 RL 管理实体-事实库）。实证警示：结构化架构（MAGMA、AMem）的图重构与 LLM 驱动巩固带来高维护成本，且弱骨干下更新易产生格式错误致记忆腐蚀。
- **经验回放 (核心主题)**: 作为综述/实证分析论文，本文不提出新的经验复用机制，而是把经验复用作为多个结构子类的核心主题加以分类与对比。编目代表：情景记忆用于探索与信用分配（EMU 用大容量情景记忆加速协作 MARL 探索、SAM2RL 用视觉记忆库作 episodic buffer + PPO 管理替换）；情景效用学习把 utility/Q-value 与意图-经验对绑定在线更新以选择性保留复用（MemRL 在线更新 Q 值平衡稳定性-可塑性、无需微调）；程序记忆把轨迹蒸馏为可复用技能/工作流供后续调用（MemP、LEGOMem、ReasoningBank 把推理经验复用以自进化、WebCoach 跨会话记忆指导）；token 级技能复用（TokMem 用可训练记忆 token 替代冗长程序提示实现常量上下文的技能复用）。综述在实证侧并不单独评测复用增益，而是把复用纳入巩固→检索循环。

**学习维度 / Learning**

- **学习范式**: 综述明确把本研究关注对象界定为非参数化 (non-parametric, in-context/external) 记忆——记忆通过显式 read-write 操作而非更新权重 θ 影响行为（与参数化学习对立）。但分类学中大量编目混合范式：策略优化记忆管理用 RL/混合训练学习存储/更新/巩固/删除决策（MEM1、Mem-α、AtomMem、Memory-R1）；token 级记忆用 RL 触发 + LoRA 编织潜在记忆（MemGen）。综述自身定位为非参数化外部记忆的结构化分析，但系统覆盖了从纯启发式管线到 RL 学习式控制策略的代际谱系（hybrid）。
- **失败学习 (核心主题)**: 综述未把失败学习作为独立章节专门处理（这更多见于 A 类自我反思系统）。但其实证分析从『系统失败』角度提供独特透镜：定义并量化『骨干敏感性 (Backbone Sensitivity)』与『静默失败 (Silent Failure)』——弱骨干模型（Qwen-2.5-3B）在记忆维护时产生无效结构化输出（malformed JSON、幻觉键），格式错误率显著高于 gpt-4o-mini（Nemori 17.91%→30.38%；SimpleMem 1.20%→4.82%），导致写操作失败、长期记忆腐蚀；编目层面把反思/失败经验复用（MemOrb 存反思记忆持续改进、MemP 从轨迹学习）纳入情景反思族。整体上本文对失败的处理是『系统可靠性失败 (format/maintenance failure)』视角，而非提出新的负样本/错误驱动学习机制。
- **技能/程序归纳**: 综述系统编目程序/技能归纳线（呼应任务锚点 procedural-memory line）：MemP 把轨迹蒸馏为程序抽象 (procedural abstraction) 供持续精炼与迁移；LEGOMem 构建模块化角色感知程序记忆用于多智能体工作流自动化；TokMem 用可训练记忆 token 替代冗长程序提示实现可扩展技能复用；MemSkill（实证评测系统之一）与『Remember Me, Refine Me』(Cao et al.) 的动态程序记忆框架做经验驱动智能体进化；ReasoningBank 归纳可复用推理记忆。综述把这些归入轻量语义（token 级）、情景反思（巩固/蒸馏）与结构化子类，作为分类者刻画其表示（程序抽象/记忆 token/工作流图）与调用（检索后注入上下文）方式，不提出新归纳算法。
- **在线 vs 离线**: 综述两者兼论并在实证中量化离线成本：在线——多会话对话/个性化 agent 在部署中按 turn 检索 + 写入/巩固（user-facing latency T_read+T_gen 是交互体验关键）；离线——记忆索引的批量构建（Construction Cost）。实证 Table 5 量化离线时间/token 经济学：AMem 构建约需 15 小时（远慢于其他基线，暗示超线性 pairwise 巩固复杂度）；Nemori 构建消耗 >7.04M tokens（约为 SimpleMem 1.3M 的 5 倍，反映『intelligence tax』）；MAGMA 以 2.7M tokens 取得更优 Pareto 平衡。

**评测 / Evaluation**

- **任务领域**: 综述实证聚焦长期对话记忆领域（LoCoMo），并在分类编目中横跨多域：多会话对话与个性化（personal assistant、AI 陪伴）、长上下文处理与网页/长程任务 agent（AgentFold、Context-Folding、WebCoach、IterResearch）、QA（HotpotQA、多跳推理）、嵌入式/具身（KARMA embodied、EgoMem 多模态）、编程/工作流自动化（LEGOMem、低代码 agent）、金融（StockMem 股票预测）、多智能体系统（G-Memory、MIRIX、EvoMem）。实证实验主要在 LoCoMo 多会话对话基准上对六系统做评测。
- **基准**: 综述实证使用并系统分析五个记忆基准的结构饱和风险（Table 2）：HotpotQA（~1k tokens，单轮，低实体多样性，饱和风险高）、LoCoMo（~20k tokens，35 sessions，高实体多样性，中等饱和风险，实证主战场）、LongMemEval-S（103k tokens，5 核心能力，临界饱和）、LongMemEval-M（>1M tokens，需外部记忆，低饱和风险）、MemBench（~100k tokens，事实/反思，可塞入 128k 窗口故饱和风险高）。实证评测在 LoCoMo 上比较 F1 与三套 LLM-judge prompt 的排序（来自 MAGMA/Nemori/SimpleMem 三源）。
- **报告增益**: 本文为综述/实证分析，不提出新系统的『增益』，而是报告诊断性实证结果。关键量化发现：(1) 指标失配——F1 与 LLM-judge 语义排序显著背离：AMem F1 仅 0.116（排第 5）但语义判分排第 4（0.480–0.512），因其抽象式答案不依赖逐词重叠被 F1 惩罚（Paraphrase Penalty）；SimpleMem F1 较高 0.268 但语义合成能力弱（语义分<0.30）。Nemori 在 Nemori-prompt 下语义分 0.781（最高）。(2) 语义判分跨三套 prompt 排序高度一致（鲁棒），而 F1 排序不可靠。(3) 骨干敏感性——格式错误率：gpt-4o-mini 上 SimpleMem 1.20% / Nemori 17.91%；Qwen-2.5-3B 上 SimpleMem 4.82% / Nemori 30.38%，且答案分从 0.781 跌至 0.447。(4) 系统延迟（per turn 总秒数）：Full Context 1.726s、AMem 1.181s、MemoryOS 高达 32.372s（检索 31.247s 成瓶颈）、MAGMA 1.462s、SimpleMem 1.057s、MemSkill 0.306s。(5) 离线构建：AMem~15h、Nemori~7.04M tokens、MAGMA 2.7M tokens（Pareto 较优）。提出 Context Saturation Gap Δ=Score_MAG − Score_FullContext（式 2）作为诊断指标。
- **对比基线**: 实证对比的关键基线是 Full-Context（暴力全文入 prompt 基线，用于度量 Context Saturation Gap Δ，检验外部记忆是否真有超越『把全部证据塞进 prompt』的优势）；以及 LOCOMO 自带检索基线。被评测的六个 MAG 系统互为对照：AMem、MemoryOS、Nemori、MAGMA、SimpleMEM、MemSkill（跨四结构分类）。在评测协议层面，本文以『现有六篇相关综述』为对照对象（Table 1：AI Hippocampus、Memory in the Age of AI Agents、Toward Efficient Agents、Rethinking Memory Mechanisms、From Storage to Experience、Graph-based Agent Memory），指出唯有本文同时系统覆盖基准饱和、指标有效性、骨干敏感性与系统成本四项。

**分析 / Analysis**

- **关键创新**: 首个把智能体记忆从『理论分类』推进到『系统化实证分析』的综述：(1) 提出结构优先 (structure-first) 的四类 MAG 分类学（轻量语义/实体个性化/情景反思/结构化层级），把记忆管理框架与优化策略也纳入；(2) 提出 Context Saturation Gap Δ=Score_MAG − Score_FullContext（式 2）作为诊断基准是否真需外部记忆的实证信号，并按 volume/interaction depth/entity diversity 三轴评估基准的结构饱和风险；(3) 实证揭示四大被忽视痛点——基准饱和（多数基准可塞进 128k+ 窗口使外部记忆结构性多余）、指标失配（F1 等词汇指标与语义正确性背离，存在 Paraphrase Penalty 与 Negation Trap 两失败模式）、骨干敏感性（弱开源骨干因结构化输出格式错误致记忆静默腐蚀）、系统级 Agency Tax（写-巩固维护带来延迟/throughput/token 开销）。最大贡献是把『为何 MAG 实际表现常低于理论承诺』转化为可量化诊断框架。
- **局限**: 论文自承局限：(1) 智能体记忆演进极快，分类学可能漏掉并发/最新系统；(2) 实证仅覆盖六个代表性 MAG 架构与选定基准（主要 LoCoMo），结果随实现、prompt、模型版本与 API 行为而变，非穷尽评测，呼吁更广泛标准化评测；(3) Table 2 报告的是结构饱和风险（基于数据集内在统计的启发式估计）而非实证饱和测试；(4) Δ 应视为诊断而非严格 pass/fail 判据（基准仍可通过效率/可更新性/鲁棒性/证据忠实度评测记忆）；(5) LLM-as-a-judge 虽更可靠但需仔细 prompt 校准；(6) 作为 arXiv 预印本（2026-02）尚未正式发表，存在 v1/v2 版本差异。
- **与其他工作关系**: 本研究为 H 类（综述）条目，与本库其他综述形成『实证 vs 理论』互补：H1（《A Survey on the Memory Mechanism of LLM-based Agents》，2024）与 H2（《Rethinking Memory in LLM-based Agents》，2025）侧重理论分类学/原子操作框架，本文则用六系统实证暴露评测与系统局限——其 Table 1 直接把 H2 类『Rethinking Memory Mechanisms (Huang et al. 2026)』『From Storage to Experience (Luo et al. 2026)』『Graph-based Agent Memory』等列为对照并指出它们未覆盖基准饱和/指标有效性/骨干敏感性/系统成本。它在实证与编目中直接引用并归类本库多个系统条目：把 B4 A-MEM（实体中心，实证系统 AMem）、B3 MemGPT（OS 启发分层）、B7 MemoryOS（实证系统，揭示其 32s 延迟瓶颈）、B9 MIRIX（多智能体层级）归入对应结构；把 D3 Zep/Graphiti（图结构双时序）、D4 Mem0、A6 ReasoningBank（情景反思自进化，呼应 procedural-memory line）、C10 MemP（程序记忆）、G 类 Memory-R1/Mem-α（策略优化/学习式记忆控制）等纳入分类。其骨干敏感性与系统成本分析为 G 类学习式记忆控制与 D 类图记忆提供了实证警示。
- **可复现性**: 中等偏上（综述层面）：官方配套 GitHub 仓库 FredJiang0324/Anatomy-of-Agentic-Memory 公开按四结构分类的论文清单（awesome-list，持续更新，欢迎社区 PR），便于追溯编目。实证部分附录提供完整 Prompt Library（Appendix D：记忆构建/检索 prompt、响应生成 prompt、三套 LLM-judge 评测协议含文献衍生基线）与 Baseline Configurations（Appendix E，六系统配置 Table 8），原则上可复现 LoCoMo 上的实验；但论文本身未提供端到端可运行代码库（仓库为论文清单非实验代码），骨干模型用 gpt-4o-mini（API）增加复现的成本/版本不确定性。作为综述无需复现新系统。社区采用通过仓库 star 与持续维护体现，处早期阶段。

**补充维度 / Supplemented (2025-26 frontier)**

- **学习型记忆控制**: 否（综述本身不实现学习式记忆控制策略）。但本文把学习式记忆控制作为分类学的重要维度并提供实证视角：专设 Policy-Optimized Memory Management 子类，把 store/update/consolidate/delete 当作可学习决策（编目 MEM1 学习常量记忆操作、Mem-α 用 RL 管理多组件外部记忆、AtomMem 把记忆分解为 CRUD 原子动作学习任务对齐控制、Memory-R1 RL 管理实体库、Memory-T1 用 GRPO）；并在背景形式化中指出 g(·) 可被优化为记忆动作上的策略（式 9 连接 RL 引导记忆）。其骨干敏感性分析（结构化记忆操作在弱骨干下崩溃）正是对 2025-26 代学习式/结构化记忆控制的实证警示。本文定位为该代际划分的实证坐标，而非实现者。
- **记忆主体**: 兼顾两者（综述全景覆盖并以结构分类）：用户中心 (user-centric)——Entity-Centric and Personalized 结构专门刻画用户画像/偏好建模（个性化对话、PAMU 偏好更新、EgoMem 终身多模态画像、MemoryBank、个性化助手），实证主基准 LoCoMo 即多会话个性化对话；智能体中心 (agent-centric)——情景反思与程序记忆刻画 agent 自身经验自进化（ReasoningBank、MemP、Voyager 类技能、WebCoach 跨会话自进化、多智能体 G-Memory/MIRIX）。综述指出不同 subject 对应不同机制与评测。
- **多智能体记忆**: 明确讨论并编目（设有相关子类与条目）：结构化/层级记忆下编目多智能体共享/路由记忆——G-Memory 为多智能体系统追踪层级记忆（insight/query/interaction 分层）、MIRIX 多智能体记忆系统、LEGOMem 为多智能体工作流自动化构建模块化角色感知程序记忆、EvoMem 用双演化记忆改进多智能体规划、MeMAD 存结构化辩论经验。综述把记忆路由 (memory routing) 与跨智能体协调列为结构化记忆的能力维度，但实证评测聚焦单智能体场景。
- **时序推理支持**: 显式覆盖：图结构记忆子类强调捕捉时序关系——Zep 构建双时序 (bi-temporal) 知识图谱（episodic + semantic 层）、MAGMA 含专门的时序图、TiMem 时序-层级记忆树、『Beyond Dialogue Time: Temporal Semantic Memory』；情景效用学习中 Memory-T1 用 GRPO 学习时序感知检索策略以优化时序一致性，Memory-T1（RL for Temporal Reasoning in Multi-Session Agents）专攻多会话时序推理。综述在基准分析中把 interaction depth（temporal structure，如 LoCoMo 35 sessions 的纵向推理）作为评估饱和风险的核心轴之一，并将时序推理视为最难评测能力之一。
- **模态**: 以文本为主、覆盖多模态（综述层面）：实证评测均为文本（LoCoMo 对话）；分类编目含多模态/具身条目——EgoMem 构建终身多模态画像、KARMA 增强具身 AI 的长短期记忆、SAM2RL 用视觉记忆库（Segment Anything Model 2）、『Full-Duplex Omnimodal Models』。整体上属文本为主、附带视觉/具身记忆编目的综述。
- **过度个性化/记忆安全风险**: 部分覆盖（系统可靠性/安全维度，非传统过度个性化）：本文的独特安全透镜是『记忆腐蚀 (memory corruption) 与静默失败』——弱骨干模型在记忆更新时产生无效结构化输出致长期记忆被错误写入而静默腐蚀（4.4 节实证）；并指出过度维护开销会致 throughput collapse、记忆陈旧 (stale memory)。论文亦警示『更多/更复杂记忆并非总更好』：复杂结构化记忆在饱和基准上常被 Full-Context 基线追平甚至超越，引入不必要成本。但本文未深入隐私治理/有害-谄媚记忆等传统过度个性化主题（这更多见于 F 类 OP-Bench/Causal-LoCoMo），在编目中提及 EgoMem 的 conflict-aware 个性化更新。
- **冲突/矛盾处理**: 编目并实证涉及：把冲突/矛盾处理归入更新与策略优化记忆操作——EgoMem 做 conflict-aware updates（冲突感知更新构建终身画像）、Memory-R1 的 UPDATE 操作、AtomMem 的 CRUD（含 update/delete）逻辑去重。实证侧指出结构化/图记忆在更新时需『提取实体、构建关系、执行逻辑去重 (logical deduplication)』，这些操作在弱骨干下格式错误率剧增、易致结构不稳定或崩溃。综述把矛盾的可靠仲裁视为依赖骨干结构化输出能力的系统级挑战，区别于纯遗忘。
- **token成本/延迟证据**: 有自身实证量化数据（本文核心贡献之一，『Agency Tax』效率剖析 Table 5）：用户侧每轮延迟（T_read+T_gen，秒）——Full Context 1.726、LOCOMO 0.783、AMem 1.181、MemoryOS 32.372（检索 31.247 成瓶颈，严格分层分页对交互式场景不可行）、Nemori 1.129、MAGMA 1.462、SimpleMem 1.057、MemSkill 0.306；离线构建——时间(h)：LOCOMO 0.86、AMem 15.00（超线性 pairwise 巩固）、MemoryOS 7.83、Nemori 3.25、MAGMA 7.28、SimpleMem 3.45、MemSkill 0.60；token(k)：LOCOMO 1623、AMem 1486、MemoryOS 4043、Nemori 7044（约 SimpleMem 1308 的 5 倍 intelligence tax）、MAGMA 2725、SimpleMem 1308、MemSkill 1796。核心论点：MAG 相对只读 RAG 多出 write-consolidate 维护开销（Agency Tax），但延迟/构建时间/token 三者权衡在文献中极少被报告，对部署却至关重要。

**其他信息 / Other**

- **cluster**: H. 综述 (Surveys)
