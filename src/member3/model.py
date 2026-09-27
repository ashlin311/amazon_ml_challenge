"""
model.py — DeBERTa cross-encoder model definition for pairwise entity matching.

Responsibilities:
  - Wrap a pre-trained DeBERTa model with a binary classification head.
  - Provide forward pass returning match logits / probabilities.
  - Support loading and saving fine-tuned checkpoints.
  - Support CPU and CUDA.
"""

import os
from typing import Optional

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL_NAME = "microsoft/deberta-v3-small"
NUM_LABELS = 2
DEFAULT_MAX_LENGTH = 256


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_tokenizer(model_name_or_path: str = DEFAULT_MODEL_NAME):
    """Load the tokenizer for the given model."""
    return AutoTokenizer.from_pretrained(model_name_or_path)


def load_model(
    model_name_or_path: str = DEFAULT_MODEL_NAME,
    num_labels: int = NUM_LABELS,
):
    """Load a DeBERTa model with a sequence-classification head.

    Parameters
    ----------
    model_name_or_path : str
        HuggingFace model hub name or local checkpoint directory.
    num_labels : int
        Number of classification labels (default 2: match / non-match).

    Returns
    -------
    AutoModelForSequenceClassification
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        num_labels=num_labels,
    )
    return model


def get_device() -> torch.device:
    """Return the best available device (CUDA if available, else CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(model, tokenizer, output_dir: str) -> None:
    """Save model and tokenizer to a directory."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def load_checkpoint(checkpoint_dir: str):
    """Load a fine-tuned model and tokenizer from a checkpoint directory.

    Returns
    -------
    (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint_dir)
    return model, tokenizer
