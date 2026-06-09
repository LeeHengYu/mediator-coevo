I now have comprehensive detail on all key items, including the LbMAS implementation specifics (control unit as LLM, public/private blackboard, planner/decider/critic/cleaner/conflict-resolver agents). I have thoroughly covered classical roots and modern works. Let me compile the final report.

I have gathered enough evidence across 15+ searches and primary-source reads. Here is my research synthesis.

---

# Blackboard Architecture for Multi-Agent Systems — Research Report

The blackboard is the canonical "shared-memory mediated coordination" paradigm: independent knowledge sources (KSs) never talk to each other directly; they only read from and write to a shared, structured blackboard, and a **separate control/scheduling component** decides which KS gets to act next. That control shell is itself a mediator — it gates write access, prioritizes contributions, and arbitrates conflicts. This maps directly onto mediator-coevo: the blackboard control unit is a learnable/adaptive mediator over a shared coordination medium. The classical roots (Hearsay-II, BB1) are foundational and the paradigm has had a strong 2025–2026 revival in LLM-MAS.

## 补充Items

### Classical roots
- **Hearsay-II Speech Understanding System** (1980, Erman, F. Hayes-Roth, Lesser, Reddy): The origin blackboard system; independent KSs (acoustic/phonetic/lexical/syntactic/semantic) post hypotheses at multiple abstraction levels onto a shared blackboard, mediated by a focus-of-control scheduler that allocates limited compute to highest-value actions. ACM Computing Surveys 12(2):213-253. https://doi.org/10.1145/356810.356816 (PDF: http://faculty.chas.uni.edu/~wallingf/teaching/162/readings/hearsay-ii.pdf)
- **A Retrospective View of the Hearsay-II Architecture** (1977, Lesser & Erman): Earlier IJCAI retrospective detailing the KS/blackboard/scheduler separation. https://www.ijcai.org/Proceedings/77-2/Papers/055.pdf
- **BB1 / A Blackboard Architecture for Control** (1984–1985, B. Hayes-Roth): Makes control itself a blackboard problem — a **separate control blackboard** plus control KSs dynamically build/modify a control plan at runtime (opportunistic reasoning). The key "mediator-as-first-class-object" reference. Artificial Intelligence 26(3):251-321. https://doi.org/10.1016/0004-3702(85)90063-3 (TR: http://i.stanford.edu/pub/cstr/reports/cs/tr/84/1034/CS-TR-84-1034.pdf)
- **Blackboard Systems, Part One** (1986, H. Penny Nii): The definitive survey of the blackboard model and evolution of architectures (HASP/SIAP, CRYSALIS, HEARSAY). AI Magazine 7(2):38-53. https://doi.org/10.1609/AIMAG.V7I2.537
- **The Evolution of Blackboard Control Architectures** (1992, Carver & Lesser): Traces control mechanisms from HSII's local-only value estimation through goal-directed (DVMT) and RESUN; central theme = how to *estimate expected value of actions* (the mediation/filtering problem). Expert Systems with Applications 7(1):1-30. https://doi.org/10.1016/0957-4174(94)90023-X
- **Blackboard Systems** (1991, D. Corkill) and (1988, I.D. Craig, AI Review 2(2):103-118): Standard reference surveys; Corkill is the GBB (Generic Blackboard) author cited by the modern works. https://doi.org/10.1016/0950-7051(89)90039-7

### Modern LLM-blackboard (2025–2026) — primary new items
- **bMAS / LbMAS** (2025, Bochen Han, Songmao Zhang): The flagship "blackboard architecture for LLM-MAS" paper. Three components: control unit (an LLM that selects which agents act each round given query + blackboard content + agent ability descriptors), the blackboard (public + private spaces, replaces per-agent memory), and role agents (planner, decider, critic, cleaner, conflict-resolver). Iterates until consensus; matches SOTA at lower token cost. arXiv:2507.01701. https://arxiv.org/abs/2507.01701
- **LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science** (2025, Salemi, Parmar, et al. — Google/UMass): Central agent posts a *request* to a shared blackboard (no addressee); autonomous helper agents (data-lake partitions / web retrieval) *self-select / volunteer* to respond; append-only board. Removes the master-slave coordinator's need to know each sub-agent's capabilities. 13–57% relative gain over RAG and master-slave. arXiv:2510.01285 / ICLR 2026. https://arxiv.org/abs/2510.01285
- **Terrarium: Revisiting the Blackboard for Multi-Agent Safety, Privacy, and Security** (2025): Explicitly revives Erman-1980 blackboard as a modular, configurable, *centralized communication proxy* for studying MAS attacks (exfiltration, poisoning, spoofing, DoS, collusion). Blackboard topology set by a factor graph; append-only board with write/erase/reference/read/highlight actions over MCP. The mediator-as-observability-layer framing. arXiv:2510.14312. https://arxiv.org/abs/2510.14312
- **PatchBoard: Schema-Grounded State Mutation for Reliable and Auditable LLM Multi-Agent Collaboration** (2026, Shuyu Zhang et al.): Replaces dialogue with validated JSON-Patch mutations over shared structured state. An Architect agent builds task schema + role-specific *write contracts*; a deterministic kernel validates every mutation against schema/role/invariants before transactional commit. 84.6% vs 30.8% (LangGraph) on ALFWorld at far lower token cost. The strongest example of write-arbitration/conflict-resolution as the mediator. arXiv:2605.29313. https://arxiv.org/abs/2605.29313
- **MDTeamGPT** (2025, Kai Chen et al., Nanjing Univ.): Medical MDT consultation; shared "Historical Shared Pool" + Lead Physician that aggregates each round into Consistency/Conflict/Independence/Integration categories (residual discussion structure) to reduce "information pollution" and context collapse. Dual knowledge bases (CorrectKB / ChainKB). arXiv:2503.13856. https://arxiv.org/abs/2503.13856
- **MegaAgent** (2025, ACL Findings): Large-scale autonomous MAS without predefined SOPs; hierarchical shared task/monitoring system functioning as a multi-level blackboard. https://aclanthology.org/2025.findings-acl.259.pdf

### Modern global-workspace-theory (GWT) inspired — the "neuro" branch of blackboard
- **"Theater of Mind" / Global Workspace Agents (GWA)** (2026, Wenlong Shang): Explicitly evolves *from blackboard to global workspace*. Replaces passive shared memory with an active event-driven broadcast hub; 4-phase Cognitive Tick (Perceive→Think→Arbitrate→Update); heterogeneous agents (Attention/Generator/Critic/Meta-arbitrator/Response); entropy-based intrinsic drive regulates temperature to break deadlocks. The Meta Agent = metacognitive arbitrator (mediator). arXiv:2604.08206; code https://github.com/giansha/Global-Workspace-Agents
- **BIGMAS: Brain-Inspired Graph Multi-Agent Systems for LLM Reasoning** (2026, Hao, Dai, et al., CASIA): GWT-grounded; a GraphDesigner builds per-problem agent topology, agents coordinate *exclusively* through a centralized shared Workspace, and a global Orchestrator uses the complete shared state for routing (overcoming local-view bottleneck). arXiv:2603.15371. https://arxiv.org/abs/2603.15371
- **Beyond Text-Passing: Shared Cognitive Substrates for Multi-Agent LLM Coordination** (2026, ICLR MALGAI workshop): Argues for *typed, schema-bound* substrates (not text buffers) with causal provenance and **budget-aware arbitration**; introduces "batons" that govern only privileged writes/commits while reads proceed concurrently. Directly extends Hayes-Roth-1985 with auditable, budgeted mediation. https://openreview.net/pdf?id=RRIw2L4Z1g

### Related shared-medium coordination (not labeled "blackboard" but mechanistically equivalent)
- **MetaGPT — Shared Message Pool** (2023, Hong et al.): The widely-cited publish/subscribe shared message pool; agents publish structured messages and subscribe by profile — a blackboard with a subscription-based read filter. Cited as the canonical blackboard exemplar by the Beyond-Self-Talk survey. ICLR 2024.
- **AutoGen GroupChat / SelectorGroupChat** (2023–2024, Microsoft): Shared conversation context + a GroupChatManager that does *speaker selection* (round-robin, LLM selector, or FSM transition graph) and termination — i.e. the control-unit-as-mediator pattern in production. https://microsoft.github.io/autogen/
- **INMS: Interactive Memory Sharing** (2024): Shared memory pool of Prompt-Answer pairs filtered by an LLM "dialogue moderator" scorer + an adaptive retriever — explicit information-filtering mediator over a shared medium. arXiv:2404.09982

### Surveys to anchor the taxonomy
- **Beyond Self-Talk: A Communication-Centric Survey of LLM-Based MAS** (2025): Has a dedicated §4.2.3 "Blackboard" paradigm (centralized repository, shared communal workspace); names MetaGPT and MDTeamGPT as exemplars and flags risks (bottlenecks, misinformation injection). arXiv:2502.14321. https://arxiv.org/html/2502.14321v2
- **LLM-based Multi-Agents: A Survey of Progress and Challenges** (2024, Guo et al.): Defines "Shared Message Pool" as a distinct communication structure vs centralized/decentralized/layered. arXiv:2402.01680

## 推荐补充字段

- **write_arbitration**: Mechanism deciding *which* KS/agent may commit to the blackboard and in what order (HSII focus-of-control scheduler; bMAS control unit; PatchBoard deterministic kernel + role write-contracts; "Beyond Text-Passing" batons). The core mediator action; currently folded into mediation_and_information_filtering but deserves its own column because modern systems differ sharply (LLM-selector vs deterministic-validator vs lock/baton).
- **read_selection / context_gating**: How agents decide *what subset* of the board to read as their prompt (MetaGPT subscribe-by-profile; Terrarium factor-graph blackboard membership; AutoGen candidate_func filter). Matters because feeding the whole board into context degrades LLM performance — the filter is a load-bearing design choice.
- **conflict_resolution**: Explicit handling of contradictory contributions (bMAS conflict-resolver agent; MDTeamGPT Consistency/Conflict/Integration categorization; GWA Critic+Meta arbitration). Distinct from write-arbitration (ordering) — this is reconciling *semantically* clashing entries.
- **board_persistence_and_GC**: Append-only vs mutable, and garbage-collection/compaction (bMAS cleaner agent; GWA dual-layer STM/LTM bifurcation at token threshold; Terrarium append-only with erase action). Critical for long-horizon runs and token cost.
- **board_structure_typing**: Free-text vs typed/schema-bound vs leveled-abstraction (HSII multi-level hypothesis hierarchy; PatchBoard JSON schema; "Beyond Text-Passing" typed substrate with invariants). Determines what validation/audit is possible.
- **control_locus**: Where coordination authority lives — centralized control unit, distributed self-selection/volunteering, or broadcast+competition (bMAS central LLM; Salemi distributed volunteering; GWA spotlight-competition). The crux of "is the mediator a single chokepoint or emergent."
- **termination_consensus_mechanism**: How the system decides it's done (bMAS "consensus on blackboard"; AutoGen termination keyword/manager; GWA [RESPONSE] tag). The mediator's stop condition.
- **auditability_provenance**: Whether contributions are attributable/replayable (PatchBoard transactional + replayable; Terrarium logged transcripts; "Beyond Text-Passing" causal provenance). Increasingly a primary motivation for the 2026 blackboard revival.

## 信息来源
- [Hearsay-II (Erman et al. 1980, ACM Computing Surveys)](https://doi.org/10.1145/356810.356816) / [PDF](http://faculty.chas.uni.edu/~wallingf/teaching/162/readings/hearsay-ii.pdf)
- [Lesser & Erman 1977, Retrospective View of Hearsay-II](https://www.ijcai.org/Proceedings/77-2/Papers/055.pdf)
- [B. Hayes-Roth 1985, A Blackboard Architecture for Control](https://doi.org/10.1016/0004-3702(85)90063-3) / [BB1 TR CS-TR-84-1034](http://infolab.stanford.edu/TR/CS-TR-84-1034.html)
- [Nii 1986, Blackboard Systems Part One](https://doi.org/10.1609/AIMAG.V7I2.537)
- [Carver & Lesser 1992, Evolution of Blackboard Control Architectures](https://doi.org/10.1016/0957-4174(94)90023-X)
- [Han & Zhang 2025, bMAS/LbMAS, arXiv:2507.01701](https://arxiv.org/abs/2507.01701) / [emergentmind summary](https://www.emergentmind.com/topics/blackboard-based-llm-multi-agent-system-bmas)
- [Salemi et al. 2025, Multi-Agent Blackboard for Data Science, arXiv:2510.01285](https://arxiv.org/abs/2510.01285) / [OpenReview](https://openreview.net/pdf?id=egTQgf89Lm)
- [Terrarium 2025, arXiv:2510.14312](https://arxiv.org/html/2510.14312)
- [PatchBoard 2026, arXiv:2605.29313](https://arxiv.org/abs/2605.29313)
- [MDTeamGPT 2025, arXiv:2503.13856](https://arxiv.org/abs/2503.13856) / [OpenReview](https://openreview.net/forum?id=51DKw1vQ5p)
- [Shang 2026, Theater of Mind / GWA, arXiv:2604.08206](https://arxiv.org/pdf/2604.08206) / [code](https://github.com/giansha/Global-Workspace-Agents)
- [BIGMAS 2026, arXiv:2603.15371](https://arxiv.org/abs/2603.15371)
- [Beyond Text-Passing: Shared Cognitive Substrates, ICLR 2026 MALGAI](https://openreview.net/pdf?id=RRIw2L4Z1g)
- [Beyond Self-Talk: Communication-Centric Survey, arXiv:2502.14321 §4.2.3](https://arxiv.org/html/2502.14321v2)
- [Guo et al. 2024, LLM-MA Survey (Shared Message Pool), arXiv:2402.01680](https://arxiv.org/html/2402.01680)
- [AutoGen GroupChat / SelectorGroupChat docs](https://microsoft.github.io/autogen/0.6.4/user-guide/agentchat-user-guide/selector-group-chat.html)
- [INMS Memory Sharing 2024, arXiv:2404.09982](https://arxiv.org/pdf/2404.09982)

**Key cross-cutting finding for mediator-coevo**: Across both eras, the load-bearing component is *not* the board itself but the **control/scheduling shell** — HSII's focus-of-control, BB1's control blackboard, bMAS's control unit, PatchBoard's validation kernel, GWA's Meta-arbitrator. This shell is exactly a mediator over a shared medium, and the field is converging on the same open problems your framework targets: how to *estimate the value of admitting a contribution* (Carver & Lesser's central theme, reborn as LLM agent-selection and budget-aware arbitration), and how to make that mediation auditable/learnable. Two distinct mediator designs have emerged in 2025–2026: **LLM-as-controller** (bMAS, AutoGen selector, GWA Meta) which is flexible but stochastic, and **deterministic-validator-as-controller** (PatchBoard kernel, typed substrates with batons) which trades flexibility for safety/auditability — a tension worth surfacing in the framework.
