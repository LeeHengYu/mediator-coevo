I found another important academic item — "Emergent Collective Memory in Decentralized Multi-Agent AI Systems" (arXiv 2512.10166), which cites a "boldini2024stigmergy" survey. I now have exhaustive coverage. Let me compile the final report.

I have completed comprehensive research across the full stigmergy lineage. Here is my synthesis.

---

# Stigmergy — Indirect, Environment-Mediated Coordination (classical roots → LLM/agent reincarnations)

研究切片完成。Below: validated/extended seed items, new items spanning classical → LLM-era, recommended new fields, and sources. **Key finding on the LLM thread:** contrary to the widely-circulated early-2026 claim (de Beer essay) that "stigmergy + LLM agents returns zero results," there is in fact a fast-growing 2025–2026 cluster — but it splits sharply into (a) **academic, peer-reviewed work** that mostly uses the *blackboard* lineage and rarely says "stigmergy" explicitly, and (b) a **practitioner/open-source explosion** that uses the term "stigmergy" heavily but is largely un-peer-reviewed (GitHub repos, dev.to, independent arXiv preprints). True trace-based, environment-mediated coordination (vs. direct messaging) is the discriminating axis throughout.

---

## 补充Items

### Classical foundations (seed #23 lineage)
- **Grassé stigmergy (1959, Pierre-Paul Grassé)**: the origin — coined "stigmergy" (Gk *stigma* "mark/goad" + *ergon* "work") for *Bellicositermes natalensis*/*Cubitermes* termite nest reconstruction; "stimulation of workers by the very performances they have achieved." https://doi.org/10.1007/BF02223791
- **Deneubourg double-bridge / Argentine ant shortcut (Goss, Aron, Deneubourg, Pasteels 1989)**: the canonical empirical proof that trail-pheromone + evaporation alone selects shortest paths — direct mechanistic ancestor of ACO. https://dipot.ulb.ac.be/dspace/bitstream/2013/19271/1/042GossNaturwissenschaften89.pdf
- **Ant System (Dorigo, Maniezzo, Colorni 1996, IEEE SMC-B)**: first ACO algorithm; "stigmergic variables" / pheromone as distributed long-term memory on graph edges. https://iridia.ulb.ac.be (also Ant Colony System, Dorigo & Gambardella 1997, IEEE TEC)
- **"A Brief History of Stigmergy" (Theraulaz & Bonabeau 1999, Artificial Life 5(2):97-116)**: defines the **quantitative vs. qualitative stigmergy** distinction (load-bearing for your framework). https://doi.org/10.1162/106454699568700
- **Swarm Intelligence: From Natural to Artificial Systems (Bonabeau, Dorigo, Theraulaz 1999, OUP/Santa Fe)**: the canonical bridge text from biological stigmergy to engineered swarm algorithms. ISBN 0195131592
- **"Ant algorithms and stigmergy" (Dorigo, Bonabeau, Theraulaz 2000, Future Gen. Computer Systems)**: formalizes "artificial/synthetic stigmergy" as the design principle. https://www.sciencedirect.com/science/article/abs/pii/S0167739X0000042X

### Digital / robotic stigmergy (seed #24 lineage)
- **Digital Pheromones for swarming UAVs / unmanned vehicles (Parunak, Brueckner, Sauter; AAMAS 2002, AIAA 2002, E4MAS 2004)**: the foundational *digital* pheromone work (ADAPTIV); potential-field deposit/evaporate/propagate dynamics scaled to 50k virtual UAVs. https://link.springer.com/chapter/10.1007/978-3-540-32259-7_13
- **"A survey of environments and mechanisms for human-human stigmergy" (Parunak 2006, E4MAS)**: introduces the **sematectonic vs. marker-based** and quantitative/qualitative cross-classification applied to humans. https://dl.acm.org/doi/10.1007/11678809_10
- **S-MADRL — Stigmergic Multi-Agent Deep RL (Aina & Ha 2025, arXiv 2510.03592)**: virtual-pheromone stigmergy fused with deep RL (DQN); decentralized congestion-avoidance for ≤8 robots. https://arxiv.org/abs/2510.03592

### Human stigmergy
- **"Stigmergy as a universal coordination mechanism I & II" (Heylighen 2016, Cognitive Systems Research 38)**: THE generalizing theory — defines stigmergy abstractly (action→trace-in-medium→subsequent action), classifies by #agents/scope/persistence/sematectonic-vs-marker/quantitative-vs-qualitative; extends to Wikipedia, cognition, the web. https://pespmc1.vub.ac.be/Papers/Stigmergy-Springer.pdf
- **"Stigmergy in Open Collaboration: ... Wikipedia" (Zheng, Mai, Yan, Nickerson 2023, JMIS 40(3):983-1008)**: empirical human stigmergy — "collective modification + collective excitation"; a spatial-temporal-clustering measure of stigmergy correlates with knowledge quality. https://www.jmis-web.org/articles/1636
- **Heylighen, "Why is Open Access Development so Successful? Stigmergic organization..." (2007)** + **Crowston/Howison/Østerlund/Bolici stigmergic-coordination-in-FLOSS** work: open-source software as human stigmergy (the code artifact is the medium). https://crowston.syr.edu/sites/default/files/stigmergy.pdf

### LLM-era — academic / peer-reviewed (seed #25 lineage)
- **GPTSwarm: Language Agents as Optimizable Graphs (Zhuge, Wang, Kirsch, Faccio, Khizbullin, Schmidhuber 2024, ICML Oral, arXiv 2402.16823)**: the key LLM precursor — agents as computational graphs with REINFORCE edge-optimization; explicitly described elsewhere as "stigmergic feedback." Most-cited LLM-MAS work adjacent to stigmergy. https://arxiv.org/abs/2402.16823 / code https://github.com/metauto-ai/GPTSwarm
- **Generative Agents / Smallville (Park et al. 2023, UIST, arXiv 2304.03442)**: 25 LLM agents coordinating through a *shared simulated environment* (observe each other's actions/spaces, self-activate) — the strongest empirical precedent for environment-mediated LLM coordination, though a simulation, not infrastructure.
- **LLM-based Multi-Agent Blackboard System for Information Discovery (Salemi, Parmar, Goyal, ... Zamani 2025, arXiv 2510.01285 / OpenReview)**: central agent *posts requests to a shared blackboard*; subordinate agents self-select to respond (no task assignment). +13–57% task success vs. master-slave/RAG. Blackboard = stigmergic medium. https://arxiv.org/abs/2510.01285
- **Exploring Advanced LLM MAS Based on Blackboard Architecture / bMAS (Han & Zhang 2025, arXiv 2507.01701)**: agents communicate *solely* through a blackboard, no direct contact; agent selection driven by current blackboard content; competitive with SOTA at lower token cost. https://arxiv.org/abs/2507.01701
- **MACOG — Multi-Agent Code-Orchestrated Generation (Khan et al. 2025, arXiv 2510.03902)**: typed shared blackboard (I-IR versions, validator traces, deploy logs, cost sheets, policy proofs, content-hashed) for Infrastructure-as-Code; *but* a centralized finite-state orchestrator (so partial/hybrid stigmergy). GPT-5 54.9→74.0 on IaC-Eval. https://arxiv.org/abs/2510.03902
- **SwarmSys: Decentralized Swarm-Inspired Agents (2025, arXiv 2510.10047)**: Explorer/Worker/Validator roles; **pheromone-inspired reinforcement** — validated traces strengthen, ineffective decay — as a decentralized optimization loop over agent/event profiles. https://arxiv.org/pdf/2510.10047
- **SwarmAgentic (Zhang, Lin, Tang, ... Tresp 2025, EMNLP main)**: Particle-Swarm-Optimization reformulated in *language space* to generate+evolve whole agentic systems; +261.8% over ADAS on TravelPlanner. Swarm-intelligence, less purely stigmergic. https://aclanthology.org/2025.emnlp-main.93/
- **Emergent Coordination via Pressure Fields and Temporal Decay (Rodriguez 2026, arXiv 2601.08129)**: most explicit academic stigmergy-for-LLM paper — agents edit a *shared artifact* guided only by local "pressure gradients" (= pheromone concentration) with **temporal decay** (= evaporation); proves convergence via potential games. 48.5% solve vs 12.6% conversation / 1.5% hierarchical on meeting-room scheduling; decay ablation ~10pp. Code: github.com/Govcraft/pressure-field-experiment. https://arxiv.org/html/2601.08129v3
- **Emergent Collective Memory in Decentralized Multi-Agent AI (2025, arXiv 2512.10166)**: agents deposit *persistent multi-category environmental traces* (food/danger/social/exploration) with category-specific exponential decay → spatially distributed collective memory; explicitly grounds in stigmergy. Cites a **"boldini2024stigmergy"** survey worth chasing. https://arxiv.org/html/2512.10166
- **Ledger-State Stigmergy (Mireles Garcia 2026, arXiv 2604.03997)**: formal framework mapping Grassé's 4 components onto blockchain state; on-chain agents (keepers, arb bots) coordinate by reading shared ledger state. Patterns: State-Flag / Event-Signal / Threshold-Trigger + Commit-Reveal. https://arxiv.org/abs/2604.03997

### LLM-era — practitioner / framework / open-source (term used explicitly; mostly non-peer-reviewed)
- **SBP — Stigmergic Blackboard Protocol (AdviceNXT 2026)**: a spec'd protocol — digital pheromones (intensity+decay+payload) for ephemeral signals + durable "traces"; Emit/Sniff/Scent/Evaporate verbs; positioned as complementary to MCP. https://github.com/AdviceNXT/sbp
- **markspace / hyperspace (opinionated-systems 2026)**: stigmergic coordination protocol for agent fleets with a *deterministic guard layer* enforcing identity/scope/conflict at the environment boundary (agent cannot bypass) — typed marks (plan/fact/belief/warning/escalation) with purpose-driven lifecycles (facts permanent, beliefs decay, plans expire). Strong relevance to mediator-as-guard. https://github.com/opinionated-systems/markspace
- **PheroPath (2026)**: digital stigmergy on the *filesystem* — agents leave DANGER/TODO/SAFE/INSIGHT "scent" metadata on files; "context attached to the file, not the prompt." Planned MCP server. https://github.com/starpig1129/PheroPath
- **pssah4/stigmergy (2026)**: stigmergy as a *recall layer* over an agent loop — accepted capability-transitions deposit pheromone (weighted by quality+token-efficiency, decayed); surfaces proven paths for similar future tasks. Very close to your diffusion/skill-consolidation thesis. https://github.com/pssah4/stigmergy
- **Many Tems / temm1e (2026)**: scent-based stigmergic coordination over a shared SQLite "Den"; reports 5.86× speedup, 3.4× lower token cost vs LLM-to-LLM chat ("zero coordination tokens"). https://github.com/nagisanzenin/temm1e
- **Production-Grade/stigmergy, geokaralis/collective-intelligence, mandible-ai/mandible, Stigmera** (2025–26): pressure-field schedulers / minimal stigmergy libs (env.strongest/freshest/reinforce/invalidate) / filesystem-or-GitHub substrate frameworks. https://github.com/Production-Grade/stigmergy
- **"Stigmergy Pattern for Multi-Agent LLM Systems" (dev.to, 2026)** & **de Beer "Sync is not ReAct" (2026)**: the conceptual framing essays. De Beer's three-axis taxonomy (declarative self-activation × environment-mediated coordination × persistent shared state) and his blackboard/Linda/CALM/affordance genealogy are the best available conceptual scaffolding — even though his "zero results" literature claim is already outdated. https://www.christopherdebeer.com/isnt-this-just-react.html

---

## 推荐补充字段 (new fields for the stigmergy slice)

The seed's `mediation_and_information_filtering_mechanism` field is well-chosen and confirmed central. Recommended additions/refinements:

- **trace_persistence_and_decay**: Does the medium have evaporation/decay (pheromone, half-life, TTL) vs permanent traces? How is it parameterized (decay rate, half-life, evaporation threshold)? *Why:* This is THE mechanism that filters stale info and prevents premature convergence — the empirically load-bearing knob (pressure-field decay ablation ≈10pp; SBP "stale-by-default"; markspace lifecycles). Distinguishes negative-feedback (forgetting) from pure accumulation.
- **indirect_vs_direct_coordination** (purity axis): Is coordination *purely* trace-based (no agent ever addresses another) or hybrid (e.g., a central orchestrator/FSM still sequences agents)? *Why:* The single discriminating criterion separating "true stigmergy" from "blackboard + scheduler" (MACOG, Salemi blackboard, ChatDev-puppeteer are hybrids; pressure-field, SBP, Smallville are pure). Maps to de Beer's coordination column.
- **medium_modifiability / trace_type**: Sematectonic (the work product *itself* is the signal — code, document, artifact state) vs marker-based (separate deposited signals — pheromones/scents/flags over the work). *Why:* directly imports Wilson/Parunak's classic distinction; in LLM systems this is the difference between editing a shared artifact (Wikipedia/pressure-field/Smallville) vs annotating it with metadata pheromones (PheroPath/SBP).
- **signal_granularity** (quantitative vs qualitative): Scalar intensity/gradient (quantitative — pheromone concentration, pressure) vs discrete typed signals (qualitative — DANGER/TODO/INSIGHT, plan/fact/belief). *Why:* the Theraulaz-Bonabeau axis; predicts whether amplification dynamics or discrete pattern-matching drives coordination.
- **activation_mechanism**: How agents are triggered by the medium — agent-driven polling (ReAct-style), declarative predicate/scent-threshold ("wake me when condition X"), clock-tick, or external scheduler. *Why:* de Beer shows this is orthogonal to coordination and is the axis where LLMs "solve the blackboard control problem" (the 40-year-old scheduler problem). Highly relevant to whether your mediator pushes vs agents pull.
- **boundary_enforcement / governance_layer**: Is there a deterministic guard between agents and the medium enforcing scope/identity/conflict (markspace), schema/typing/auth (MACOG, PatchBoard), or content-hashing/provenance? *Why:* directly maps to your `mediator` role and your "validity-aware observability" (provenance, leakage, token budgets) — the medium-as-mediator becomes a *governed* mediator.
- **convergence_dynamics**: positive feedback (amplification/autocatalysis) + negative feedback (evaporation/repellers); any formal guarantee (ACO convergence proofs; pressure-field potential-game proof; CALM monotonicity). *Why:* lets the report make precise claims about *why* graph-aware diffusion can beat broadcast/random — same amplify-good / suppress-error logic as pheromone selection.

---

## Relevance to mediator-coevo (load-bearing connection)

The project's **experience-diffusion mechanism is structurally a stigmergic system**: source tasks emit compact artifacts → a medium holds them → a diffusion policy filters/routes them into later planning contexts, with skills held fixed. This maps cleanly:
- The **diffusion artifact store = the medium/blackboard**; artifacts (debug hints, mediator summaries, regression warnings) = **marker-based qualitative traces**.
- **top-k similarity vs broadcast vs random vs off** diffusion policies = **information-filtering on the medium** — exactly the deposition/evaporation selectivity that defines stigmergy. Broadcast ≈ no filtering; top-k similarity ≈ gradient-following (climb toward relevant traces); decay = forgetting stale artifacts.
- The **mediator = the governance/guard layer** (markspace's deterministic boundary; MACOG's typed blackboard with provenance/content-hashes) — your provenance/token-budget/leakage tracking is precisely a stigmergic-medium governance layer.
- The closest single analogue to your thesis is **pssah4/stigmergy** (pheromone over capability-paths, reinforced by graded outcomes, surfacing proven paths for similar future tasks) and **SwarmSys** (validated traces strengthen / ineffective decay). Both are "fixed-policy, context-routing" in spirit.
- Useful framing leverage: position diffusion as **quantitative-vs-qualitative** and **sematectonic-vs-marker-based** stigmergy, and cite the **decay/evaporation = forgetting** result (pressure-field ablation) to motivate artifact TTL; cite **CALM monotonicity** (de Beer) for why append-only artifact accumulation needs no coordination.

---

## 信息来源 (verified)
- Grassé 1959 (Insectes Sociaux, DOI 10.1007/BF02223791) — via [Wikipedia/Stigmergy](https://en.wikipedia.org/wiki/Stigmergy), [Heylighen Springer PDF](https://pespmc1.vub.ac.be/Papers/Stigmergy-Springer.pdf)
- [Heylighen 2016, Stigmergy as universal coordination mechanism I (CSR 38)](https://www.sciencedirect.com/science/article/abs/pii/S1389041715000327)
- [Theraulaz & Bonabeau 1999, A Brief History of Stigmergy (Artificial Life)](https://pubmed.ncbi.nlm.nih.gov/10633572/) ; [full PDF](https://crca.cbi-toulouse.fr/wp-content/uploads/2018/06/29.pdf)
- [Dorigo et al. 1996 Ant System (IEEE SMC-B)](https://sci2s.ugr.es/sites/default/files/files/Teaching/GraduatesCourses/Metaheuristicas/Bibliography/Ant_System_Dorigo_IEEE_SMC_1996.pdf) ; [Ant algorithms and stigmergy 2000](https://lia.disi.unibo.it/courses/2006-2007/PSI-LS/pdf/roli/dorigo2000-ant_algorithms_and_stigmergy.pdf)
- [Goss/Deneubourg 1989, Argentine ant shortcut](https://dipot.ulb.ac.be/dspace/bitstream/2013/19271/1/042GossNaturwissenschaften89.pdf)
- [Bonabeau, Dorigo, Theraulaz 1999, Swarm Intelligence (OUP)](https://jmvidal.cse.sc.edu/lib/bonabeau99a.html)
- [Parunak 2006, human-human stigmergy survey (E4MAS)](https://dl.acm.org/doi/10.1007/11678809_10) ; [Parunak digital pheromones / UAVs](https://link.springer.com/chapter/10.1007/978-3-540-32259-7_13)
- [Zheng, Mai, Yan, Nickerson 2023, Stigmergy in Open Collaboration: Wikipedia (JMIS)](https://www.jmis-web.org/articles/1636) ; [PDF](https://fengmai.net/wp-content/uploads/2024/09/ZhengMaiYanNickerson2023-Stigmergy-in-Open-Collaboration-An-Empirical-Investigation-Based-on-Wikipedia-JMIS.pdf)
- [GPTSwarm, Zhuge et al. 2024 ICML (arXiv 2402.16823)](https://arxiv.org/abs/2402.16823)
- [Salemi et al. 2025, Blackboard MAS (arXiv 2510.01285)](https://arxiv.org/html/2510.01285v1) ; [Han & Zhang 2025, bMAS (arXiv 2507.01701)](https://arxiv.org/abs/2507.01701v1)
- [MACOG, Khan et al. 2025 (arXiv 2510.03902)](https://arxiv.org/html/2510.03902v1)
- [SwarmSys (arXiv 2510.10047)](https://arxiv.org/pdf/2510.10047) ; [SwarmAgentic, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.93/)
- [Pressure Fields & Temporal Decay, Rodriguez 2026 (arXiv 2601.08129)](https://arxiv.org/html/2601.08129v3) ; [code](https://github.com/Govcraft/pressure-field-experiment)
- [Emergent Collective Memory (arXiv 2512.10166)](https://arxiv.org/html/2512.10166) ; [Ledger-State Stigmergy (arXiv 2604.03997)](https://arxiv.org/abs/2604.03997)
- [S-MADRL (arXiv 2510.03592)](https://arxiv.org/abs/2510.03592)
- [de Beer 2026, "Sync is not ReAct"](https://www.christopherdebeer.com/isnt-this-just-react.html) ; [SBP](https://github.com/AdviceNXT/sbp) ; [markspace](https://github.com/opinionated-systems/markspace) ; [PheroPath](https://github.com/starpig1129/PheroPath) ; [pssah4/stigmergy](https://github.com/pssah4/stigmergy)

**Caveats / gaps:** (1) Semantic Scholar's API returned near-zero for these queries (its indexing of 2025-26 arXiv + the exact term "stigmergy"+"LLM" is sparse) — Exa/Firecrawl + direct arXiv were the productive channels; the de Beer "zero results" claim reflects this indexing gap, not reality. (2) Much of the *explicitly-labeled* "stigmergy + LLM" corpus is practitioner/preprint, not peer-reviewed; the peer-reviewed core (Salemi, Han&Zhang, MACOG, GPTSwarm) tends to use *blackboard* framing instead. (3) One cited survey, "boldini2024stigmergy," and the "de2020multi" stigmergy reference (both cited by arXiv 2512.10166) I did not independently fetch — worth chasing if you need a recent stigmergy survey anchor.
