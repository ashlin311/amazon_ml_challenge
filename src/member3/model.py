"""
model.py — DeBERTa cross-encoder model definition for pairwise entity matching.

Responsibilities:
  - Wrap a pre-trained DeBERTa model with a binary classification head.
  - Provide forward pass returning match logits / probabilities.
  - Support loading and saving fine-tuned checkpoints.
"""
