"""
src/blocking_engine.py — Public access point for the Member 2 blocking layer.

This module is the single import surface for the candidate-generation pipeline.
All implementation lives in src/member2/blocking_engine.py and src/member2/phonetics.py.

Typical usage::

    from src.blocking_engine import HybridBlocker, export_candidates, evaluate_blocking_recall

    blocker = HybridBlocker(top_k_sparse=40, top_k_dense=35)
    blocker.fit_targets(target_ids, target_names, target_addresses)
    candidates = blocker.block_all(query_ids, query_names, query_addresses)
    export_candidates(candidates, "output/candidate_pairs.tsv", query_ids)
"""

from src.member2.blocking_engine import (
    normalize_text,
    clean_name,
    clean_address,
    extract_numeric_tokens,
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
    "extract_numeric_tokens",
    "HybridBlocker",
    "merge_candidates",
    "evaluate_blocking_recall",
    "export_candidates",
    "run_submission_validation",
]
