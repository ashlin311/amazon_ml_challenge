"""
src.member2 — High-Recall Hybrid Blocking & Candidate Retrieval Engine (Member 2).

Public API re-exported for convenience:
"""

from .blocking_engine import (
    normalize_text,
    clean_name,
    clean_address,
    HybridBlocker,
    merge_candidates,
    evaluate_blocking_recall,
    export_candidates,
    run_submission_validation,
)

__all__ = [
    "normalize_text",
    "clean_name",
    "clean_address",
    "HybridBlocker",
    "merge_candidates",
    "evaluate_blocking_recall",
    "export_candidates",
    "run_submission_validation",
]
