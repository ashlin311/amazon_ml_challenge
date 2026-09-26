"""
predict.py — Inference pipeline and final submission generation.

Responsibilities:
  - Load a fine-tuned DeBERTa checkpoint.
  - Score all candidate pairs and produce match probabilities.
  - Apply optimized threshold to convert probabilities to match decisions.
  - Invoke post-processing (singletons, low-confidence handling).
  - Write matching_results.tsv in the official submission format:
      source1_entity_id    matched_entity_ids
"""
