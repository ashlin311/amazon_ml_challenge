"""
src.member3 — DeBERTa cross-encoder matching pipeline (Member 3).

Key modules:
  data_loader      — load source TSVs and ground truth
  candidate_loader — load Member 2's candidate_pairs.tsv
  pair_dataset     — build labeled / inference pair records
  model            — DeBERTa model init, checkpointing
  train            — training loop, validation, checkpoint saving
  predict          — inference: score candidate pairs → probabilities
  metrics          — entity-level macro F0.5 evaluation
  threshold        — optimal threshold selection
  postprocess      — singleton / confidence post-processing
  baseline         — cheap exact-name baseline strategies
  candidate_eval   — blocking-output diagnostics
"""
