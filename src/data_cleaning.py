"""
src/data_cleaning.py — Public access point for the Member 1 data cleaning layer.

This module is the single import surface for data normalization, cleaning,
and token-level feature extraction. All implementation lives in src/member1/.

Typical usage:

    from src.data_cleaning import clean_name, clean_address, normalize_text
    from src.data_cleaning import clean_dataframe, clean_source_tsv
"""

from src.member1.data_cleaning import (
    ABBREVIATIONS,
    clean_address,
    clean_dataframe,
    clean_name,
    clean_source_tsv,
    evaluate_crf_model,
    extract_entities_from_tokens,
    normalize_text,
    sent2features,
    sent2labels,
    train_crf_model,
    word2features,
)

__all__ = [
    "ABBREVIATIONS",
    "clean_name",
    "clean_address",
    "normalize_text",
    "clean_dataframe",
    "clean_source_tsv",
    "word2features",
    "sent2features",
    "sent2labels",
    "train_crf_model",
    "evaluate_crf_model",
    "extract_entities_from_tokens",
]
