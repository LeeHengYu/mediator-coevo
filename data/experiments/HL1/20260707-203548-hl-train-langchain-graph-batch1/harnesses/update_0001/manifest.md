# update_0001

Training run: `20260707-203548-hl-train-langchain-graph-batch1`

Targeted failure mode:

- The LangChain diffusion agent sometimes misclassified successful artifacts as
  `avoid_recheck`, which rendered verified success records under the warning
  channel.
- For new sibling tasks, the agent often selected a high-level run outcome plus
  a report summary and left budget unused, omitting shorter actionable artifacts
  from the same graph source.

Changed files:

- `overlay/src/mediated_coevo/diffusion/langchain_graph.py`
- `overlay/tests/test_langchain_graph_policy.py`

Expected effect:

- Context sections now follow source verifier outcomes, so success artifacts are
  rendered as reusable context and failed artifacts are rendered as avoid/recheck
  warnings regardless of the LLM-selected channel.
- Unused artifact budget is deterministically filled from eligible graph sources,
  preferring method summaries and debug hints before bare run outcomes.
