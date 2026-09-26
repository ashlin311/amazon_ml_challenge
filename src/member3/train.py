"""
train.py — Training loop for the DeBERTa cross-encoder matching model.

Responsibilities:
  - Set up optimizer, scheduler, and loss function.
  - Run training and validation epochs.
  - Log metrics and save best checkpoints to models/deberta/.
  - Support early stopping and mixed-precision training.
"""
