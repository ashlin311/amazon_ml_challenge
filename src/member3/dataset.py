"""
dataset.py — PyTorch Dataset for DeBERTa-based pairwise entity matching.

Responsibilities:
  - Tokenize entity-pair text for the DeBERTa cross-encoder.
  - Return input_ids, attention_mask, and (optionally) labels.
  - Support both training and inference modes.
"""
