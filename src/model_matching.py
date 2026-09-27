"""
src/model_matching.py — Public access point for the Member 3 matching layer.

This module is the single import surface for the DeBERTa-based entity matching pipeline.
All implementation lives in src/member3/.

Typical usage::

    from src.model_matching import (
        load_model, load_tokenizer, load_checkpoint, get_device,
        load_train_data, load_candidate_pairs,
        build_inference_pairs, PairDataset,
        evaluate_predictions,
    )

Pipeline entry points (run as CLI modules)::

    python -m src.member3.train       # train DeBERTa cross-encoder
    python -m src.member3.predict     # score candidate pairs
    python -m src.member3.run_baseline   # cheap baseline evaluation
    python -m src.member3.candidate_eval # blocking diagnostics
"""

# --- Model layer ---
from src.member3.model import (
    DEFAULT_MODEL_NAME,
    DEFAULT_MAX_LENGTH,
    load_model,
    load_tokenizer,
    load_checkpoint,
    save_checkpoint,
    get_device,
)

# --- Data loading ---
from src.member3.data_loader import (
    load_source_tsv,
    load_ground_truth,
    load_train_data,
    split_s1_ids,
)

# --- Candidate loading ---
from src.member3.candidate_loader import load_candidate_pairs

# --- Pair construction ---
from src.member3.pair_dataset import (
    PairRecord,
    PairDataset,
    build_labeled_pairs,
    build_inference_pairs,
)

# --- Evaluation ---
from src.member3.metrics import entity_f05, evaluate_predictions

# --- Post-processing (stubs — implement in src/member3/threshold.py and postprocess.py) ---
# from src.member3.threshold import find_optimal_threshold
# from src.member3.postprocess import apply_postprocessing

__all__ = [
    # model
    "DEFAULT_MODEL_NAME",
    "DEFAULT_MAX_LENGTH",
    "load_model",
    "load_tokenizer",
    "load_checkpoint",
    "save_checkpoint",
    "get_device",
    # data
    "load_source_tsv",
    "load_ground_truth",
    "load_train_data",
    "split_s1_ids",
    # candidates
    "load_candidate_pairs",
    # pairs
    "PairRecord",
    "PairDataset",
    "build_labeled_pairs",
    "build_inference_pairs",
    # evaluation
    "entity_f05",
    "evaluate_predictions",
]
