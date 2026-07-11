# update_0001

- target failure mode: graph-off-prior cross-family success artifacts overriding the incoming graph priors during direct agent selection
- source evidence: HL3 train batch 2 run 20260710-222312-hl3-train-langchain-graph-batch2
- overlay changes:
  - overlay/src/mediated_coevo/diffusion/langchain_graph.py
  - overlay/tests/test_langchain_graph_policy.py
- carried state:
  - state/diffusion/graph_snapshots/
  - state/diffusion/diffused_records.jsonl
