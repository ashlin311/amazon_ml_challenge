"""
src — Amazon ML Challenge 2026 source package.

Facades / Entry points:
  src.data_cleaning    — Data Cleaning facade (Member 1: normalization, cleaning, CRF feature extraction)
  src.blocking_engine  — Blocking layer facade (Member 2: HybridBlocker, TF-IDF + FAISS + phonetic)
  src.model_matching   — Matching layer facade (Member 3: DeBERTa cross-encoder matching pipeline)

Sub-packages:
  src.member1          — Data normalization, cleaning, and token feature engineering
  src.member2          — High-recall hybrid blocking engine implementation
  src.member3          — Full DeBERTa matching pipeline implementation
"""
