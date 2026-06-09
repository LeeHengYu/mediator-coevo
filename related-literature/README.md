# Related Literature

This folder collects the literature reports and supporting resources used to motivate the project positioning:

> **Graph-Aware Experience Diffusion for LLM Agent Co-Evolution**

The reports are organized by research cluster. They support the diffusion-first framing: prior agent experience should first move as bounded, provenance-tracked context across related task/agent neighborhoods, while durable skill updates are treated as a later consolidation mechanism.

## Report clusters

| Folder | Focus | Why it matters |
|---|---|---|
| `agent-reflection-learning/` | Reflection, verbal feedback, self-correction, reflective memories | Supports feedback-as-context while warning against unsupported self-correction claims. |
| `cross-task-transfer-llm-agents/` | Cross-task transfer, task vectors, workflow reuse, agent transfer | Connects diffusion to cross-task in-context transfer and transferable execution evidence. |
| `graph-similarity-task-retrieval/` | Task similarity, graph retrieval, case-based reasoning, Task2Vec/Taskonomy-style transfer maps | Supports graph-aware routing instead of random or broadcast sharing. |
| `icl-test-time-adaptation/` | In-context learning and test-time adaptation | Supports the phase-1 fixed-policy control: isolate context-only adaptation before skill consolidation. |
| `llm-agent-eval-rigor/` | Agent benchmark design, leakage, ablations, error bars, rollout cards | Defines the evidence standard before claiming diffusion improves agent performance. |
| `llm-agent-memory/` | Agent memory, experience replay, graph memory, memory benchmarks | Supports externalized experience stores and selective retrieval into context. |
| `llm-longcontext-degradation/` | Long-context failure modes, distractors, context placement, over-retrieval | Motivates token-bounded top-k diffusion instead of dumping all prior artifacts. |
| `multi-agent-llm-mediator/` | Mediators, multi-agent communication, sparse debate, blackboards, topology | Frames the mediator as a controlled information bottleneck for experience diffusion. |
| `retrieval-augmented-llm-agents/` | RAG for agents, trajectory/case retrieval, memory-augmented agents | Provides direct precedents for routing trajectories, cases, workflows, and memories into agent context. |
| `self-improving-llm-agents/` | Self-improving agents, prompt evolution, skill/policy consolidation | Supports skill updates as a slower consolidation layer after diffusion identifies reusable evidence. |

## How to read this folder

1. Start with the root synthesis: [`../diffusion-first-literature-synthesis.md`](../diffusion-first-literature-synthesis.md).
2. Use the table above to inspect the report cluster relevant to a claim.
3. Treat 2025-2026 preprints and generated metadata as verify-before-citation sources.
4. Keep empirical claims separate from target positioning until matched controls finish.

## Claim discipline

- Do **not** claim that diffusion already improves agents generally.
- Do **not** claim durable skill learning when skill updates are disabled.
- Do frame fixed-policy runs as a **phase-1 control** to isolate diffusion.
- Do frame skill updates as **phase-2 consolidation** of repeatedly useful diffused evidence.
