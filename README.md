# Amazon ML Challenge 2026 — Business Entity Resolution

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

---

## 1. Project Goal & Overview

### Goal
The goal of this project is to solve the **Large-Scale Multi-Source Business Entity Resolution** challenge in the Amazon ML Challenge 2026. The objective is to identify and link identical real-world business entities across three distinct, heterogeneous datasets (`Source 1`, `Source 2`, and `Source 3`). Specifically, for each entity in **Source 1** (query set), the system must discover all corresponding records in **Source 2** and **Source 3** (target pool).

### Problem Description & Challenges
In multi-source business catalogs, entity records exhibit severe real-world noise:
* **Typographical and phonetic variations** (e.g., *"Apex Tech Solutions"* vs. *"Apeks Technologies Solns"*).
* **Syntactic formatting differences** in addresses (e.g., *"100 North Main Street, Suite 400"* vs. *"100 N Main St Ste 400"*).
* **Multilingual and international entries** containing varied scripts, diacritics, and regional abbreviations.
* **Extreme candidate space complexity**: Comparing every Source 1 entity against all records in Source 2 and Source 3 results in an intractable $\mathcal{O}(N_1 \times (N_2 + N_3))$ search space.
* **Class imbalance & Metric optimization**: True matches represent less than $0.01\%$ of all possible pairs. The competition evaluates systems using the **Micro/Macro $F_{0.5}$ score**, which weighs precision twice as heavily as recall ($\beta = 0.5$). False positives incur double the penalty of false negatives, necessitating high-precision decision boundaries.

---

## 2. Dataset Description & File Formats

The competition dataset is organized under `student_resource/dataset/` with separate `train/` and `test/` splits:

### 2.1 Raw Source Files (`train_source{1,2,3}.tsv` / `test_source{1,2,3}.tsv`)
Tab-separated values (TSV) containing entity attributes across sources:
```tsv
entity_id	business_name	business_address	country
S1-00001	Acme Corp	123 Main St, Suite 100	US
S2-00047	Acme Corporation	123 Main Street	US
S3-00812	Acme Corp.	123 Main St STE 100	US
```

| Column | Type | Description |
|---|---|---|
| `entity_id` | String | Unique record identifier (`S1-xxxxx`, `S2-xxxxx`, or `S3-xxxxx`). |
| `business_name` | String | Raw name of the business entity. |
| `business_address` | String | Raw physical address (street, unit, city, state, postal code). |
| `country` | String | ISO two-letter country code or territory identifier. |

### 2.2 Ground Truth (`train_ground_truth.tsv`)
Mapping from Source 1 entity IDs to comma-separated matching target IDs (or empty for singletons):
```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S3-00812
S1-00002	S3-00004
S1-00003	
```
* **Singletons**: Entities with no counterpart in Source 2 or Source 3. Preserving precision on singletons is vital for $F_{0.5}$.

### 2.3 Candidate Pairs (`output/candidate_pairs.tsv`)
Generated during the blocking stage. Contains up to 100 candidate target IDs per Source 1 query:
```tsv
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

### 2.4 Final Submission (`output/matching_results.tsv`)
The final output file validated by `student_resource/utils/validate_submission.py`:
```tsv
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S3-00812
S1-00002	S3-00004
S1-00003	
```

---

## 3. Execution Pipeline Traversal

The end-to-end data traversal flows through four decoupled stages:

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Raw Data: Source 1, Source 2, Source 3               │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Data Normalization & Cleaning (Member 1)                      │
│ - Unicode NFKC, lowercasing, URL/noise stripping                       │
│ - Standardized abbreviations (legal suffixes, street terms)            │
│ - Token-level feature engineering & CRF NER sequence tagging           │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Cleaned & Normalized Records
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: High-Recall Hybrid Blocking & Candidate Retrieval (Member 2)  │
│ - Sparse TF-IDF (character 3-4 n-grams, sublinear frequency)           │
│ - Dense Semantic Embeddings (FAISS inner product search)               │
│ - Phonetic Double Metaphone inverted index                             │
│ - Candidate fusion, deduping, and top-K pruning                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ candidate_pairs.tsv (Recall > 95%)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Pairwise Matching & Cross-Encoder Classification (Member 3)   │
│ - Structured pair representation: Name: ... | Address: ... [SEP] ...   │
│ - DeBERTa cross-encoder scoring & baseline string matching             │
│ - Threshold optimization on validation set for Micro/Macro F0.5        │
│ - Singleton filtering and confidence post-processing                   │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ matching_results.tsv
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│ STAGE 4: Submission Validation & Evaluation                            │
│ - student_resource/utils/validate_submission.py                        │
│ - Schema, candidate compliance, singleton, and F0.5 verification       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Detailed Methodology by Module

The repository is architected into three specialized member modules, exposed via unified top-level facades in `src/`:

```
src/
├── data_cleaning.py       ← Facade for Member 1
├── blocking_engine.py     ← Facade for Member 2
└── model_matching.py      ← Facade for Member 3
```

---

### Module 1: Data Normalization, Cleaning & Token Feature Engineering (Member 1)
**Package**: `src/member1/` | **Facade**: `src/data_cleaning.py`

Member 1 handles input sanitation, semantic normalization, and token-level feature extraction:

1. **Text Normalization Engine (`clean_name`, `clean_address`, `normalize_text`)**:
   * **Unicode NFKC normalization**: Standardizes accents, ligatures, full-width characters, and multilingual diacritics.
   * **Comprehensive Abbreviation Canonicalization**: Normalizes legal entity types (`"corporation"` $\to$ `"corp"`, `"limited liability company"` $\to$ `"llc"`, `"aktiengesellschaft"` $\to$ `"ag"`) and street/address terms (`"boulevard"` $\to$ `"blvd"`, `"apartment"` $\to$ `"apt"`, `"highway"` $\to$ `"hwy"`).
   * **Noise Removal with Information Preservation**: Strips URLs and extraneous punctuation while preserving digits (building numbers, postal codes) and Indic/multilingual script ranges (`\u0900-\u0D7F`).
2. **DataFrame & File Sanitation (`clean_dataframe`, `clean_source_tsv`)**:
   * Batch processes source TSVs, handles missing values, and creates standard `cleaned_business_name`, `cleaned_business_address`, and `normalized_text` columns.
3. **CRF Sequence Tagging & Token Feature Engineering (`word2features`, `train_crf_model`, `evaluate_crf_model`)**:
   * Implements token feature extractors capturing capitalization, prefix/suffix n-grams, digit flags, POS tags, and left/right contextual windows (`-1` and `+1` tokens).
   * Trains a Conditional Random Field (CRF) with L-BFGS optimization ($L_1/L_2$ regularization) for extracting named entities (Organizations, Locations, Persons).

---

### Module 2: High-Recall Hybrid Blocking & Candidate Retrieval (Member 2)
**Package**: `src/member2/` | **Facade**: `src/blocking_engine.py`

Member 2 solves the $\mathcal{O}(N^2)$ candidate generation problem, reducing millions of candidate combinations down to $\le 100$ high-quality candidates per Source 1 entity while achieving $>95\%$ recall.

1. **Tri-Hybrid Retrieval Engine (`HybridBlocker`)**:
   * **Branch 1: Sparse Sublinear TF-IDF**:
     * Character n-grams (range 3–4) for robust handling of misspellings, permutations, and compound words.
     * Batched sparse matrix multiplication with automatic memory chunking to prevent out-of-memory (OOM) errors.
   * **Branch 2: Dense Semantic Vector Retrieval (FAISS)**:
     * Dense embeddings via sentence-transformer encoders to capture semantic paraphrasing and conceptual synonyms.
     * Normalized cosine similarity indexing using FAISS for rapid approximate nearest neighbor retrieval.
   * **Branch 3: Phonetic Blocking (`phonetics.py`)**:
     * Inverted index utilizing the Double Metaphone algorithm.
     * Encodes primary name tokens into phonetic keys, ensuring sound-alike names (e.g., *"Smit"* vs *"Smith"*) match regardless of English or foreign phonetic spellings.
2. **Candidate Fusion & Pruning (`merge_candidates`, `export_candidates`)**:
   * Combines candidates across all three branches using reciprocal rank scoring.
   * Enforces the hard competition ceiling (maximum 100 candidates per query entity).
   * Verifies that every single Source 1 entity is present in the final candidate dictionary (with empty candidate lists for singletons).

---

### Module 3: Pairwise Cross-Encoder Matching & Optimization (Member 3)
**Package**: `src/member3/` | **Facade**: `src/model_matching.py`

Member 3 performs deep semantic comparison, probability scoring, threshold tuning, and final decision-making:

1. **Structured Pair Construction (`pair_dataset.py`)**:
   * Pairs Source 1 entities with candidate targets retrieved by Member 2.
   * Formulates rich structured textual sequences:
     `Name: <S1_name> | Address: <S1_address> | Country: <S1_country> [SEP] Name: <Target_name> | Address: <Target_address> | Country: <Target_country>`
   * Assigns binary ground-truth labels ($1$ for true match, $0$ for non-match) with strict entity-level train/validation splitting (no leakage).
2. **DeBERTa Cross-Encoder Matching (`model.py`, `train.py`, `predict.py`)**:
   * Fine-tunes pre-trained DeBERTa (`microsoft/deberta-v3-small` / `base`) with full cross-attention across the query and target pairs.
   * Employs binary cross-entropy loss with hard-negative mining from blocking candidates.
   * Exports pair-level probabilities to `output/pair_predictions.tsv`.
3. **Fast Baseline Matching Engine (`baseline.py`, `run_baseline.py`)**:
   * High-speed heuristic matcher utilizing exact name overlap, Jaro-Winkler string similarity, and token set ratios.
   * Generates benchmark submissions without requiring GPU resources.
4. **Evaluation & Threshold Optimization (`metrics.py`, `threshold.py`, `candidate_eval.py`)**:
   * Evaluates entity-level Micro and Macro $F_{0.5}$ scores.
   * Analyzes candidate generation recall, coverage fractions, and volume distributions.
   * Sweeps classification decision thresholds $\tau \in [0.1, 0.9]$ to maximize the precision-heavy $F_{0.5}$ objective.

---

### Integration Logic & Inter-Module Communication

1. **Facade Layer**: Top-level scripts (`run_pipeline.py`, `run_blocking.py`) interact exclusively with `src/data_cleaning.py`, `src/blocking_engine.py`, and `src/model_matching.py`.
2. **Member 1 $\to$ Member 2**: `src/member2/blocking_engine.py` imports `clean_name`, `clean_address`, and `normalize_text` directly from `src.member1.data_cleaning`, eliminating redundant duplication of normalizer rules.
3. **Member 2 $\to$ Member 3**: Member 2 outputs `output/candidate_pairs.tsv`. Member 3's `candidate_loader.py` and `pair_dataset.py` ingest this candidate TSV and merge it with raw entity data to generate training/inference pairs.
4. **Orchestration**: `run_pipeline.py` unifies all stages into a single configurable CLI runner.

---

## 5. Directory Layout

```
amazon_ml_challenge/
├── student_resource/
│   ├── dataset/
│   │   ├── train/
│   │   │   ├── train_source1.tsv
│   │   │   ├── train_source2.tsv
│   │   │   ├── train_source3.tsv
│   │   │   └── train_ground_truth.tsv
│   │   └── test/
│   │       ├── test_source1.tsv
│   │       ├── test_source2.tsv
│   │       └── test_source3.tsv
│   └── utils/
│       └── validate_submission.py   ← Official format validator
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py             ← Facade: Member 1 data cleaning & normalization
│   ├── blocking_engine.py           ← Facade: Member 2 hybrid blocking engine
│   ├── model_matching.py            ← Facade: Member 3 DeBERTa matching pipeline
│   ├── member1/                     ← Member 1: Normalization & feature engineering
│   │   ├── __init__.py
│   │   └── data_cleaning.py
│   ├── member2/                     ← Member 2: Candidate retrieval & blocking
│   │   ├── __init__.py
│   │   ├── blocking_engine.py
│   │   └── phonetics.py
│   └── member3/                     ← Member 3: Matching & evaluation pipeline
│       ├── __init__.py
│       ├── baseline.py
│       ├── candidate_eval.py
│       ├── candidate_loader.py
│       ├── data_loader.py
│       ├── metrics.py
│       ├── model.py
│       ├── pair_dataset.py
│       ├── postprocess.py
│       ├── predict.py
│       ├── run_baseline.py
│       ├── threshold.py
│       └── train.py
├── tests/
│   ├── test_data_cleaning.py        ← Member 1 unit tests
│   ├── test_blocking.py             ← Member 2 unit tests
│   ├── test_candidate_eval.py       ← Member 3 diagnostic tests
│   ├── test_metrics.py              ← F0.5 metric calculation tests
│   ├── test_model.py                ← DeBERTa architecture tests
│   ├── test_pair_dataset.py         ← Pair construction tests
│   └── test_train.py                ← Training loop tests
├── models/
│   └── deberta/                     ← Fine-tuned model checkpoints
├── output/                          ← Generated candidate and submission TSVs
├── run_blocking.py                  ← Standalone blocking launcher (Member 2)
├── run_pipeline.py                  ← Master end-to-end execution pipeline
├── requirements.txt
└── README.md
```

---

## 6. Steps to Execute

### 6.1 Setup Environment

```bash
# 1. Clone repository
git clone https://github.com/AsherWood39/amazon_ml_challenge.git
cd amazon_ml_challenge

# 2. Create and activate virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

### 6.2 End-to-End Pipeline Execution (`run_pipeline.py`)

The master runner orchestrates all stages from cleaning to candidate retrieval, scoring, and validation:

```bash
# Run full end-to-end pipeline on training dataset:
python run_pipeline.py --mode all --data_dir student_resource/dataset/train --output_dir output

# Run on test dataset to generate final submission:
python run_pipeline.py --mode all --data_dir student_resource/dataset/test --output_dir output --validate

# Fast benchmark on 500 samples (CPU/laptop testing):
python run_pipeline.py --mode all --data_dir student_resource/dataset/train --sample_size 500 --skip_dense
```

---

### 6.3 Executing Individual Stages

#### Stage 1: Data Cleaning (Member 1)
```bash
# Clean and normalize TSV dataset:
python run_pipeline.py --mode clean --data_dir student_resource/dataset/train --output_dir output/cleaned

# Run Member 1 normalization CLI demo:
python -m src.member1.data_cleaning --mode normalize --name "Acme International Solutions Ltd." --address "123 N Main St, Suite 400"
```

#### Stage 2: Candidate Generation & Blocking (Member 2)
```bash
# Run hybrid blocking engine:
python run_blocking.py \
    --data_dir student_resource/dataset/train \
    --output_dir output \
    --top_k_sparse 40 \
    --top_k_dense 35 \
    --max_candidates 80

# Evaluate candidate recall and diagnostics:
python -m src.member3.candidate_eval \
    --candidate-file output/candidate_pairs.tsv \
    --train-dir student_resource/dataset/train
```

#### Stage 3: Model Training & Inference (Member 3)
```bash
# Train DeBERTa cross-encoder:
python -m src.member3.train \
    --train-dir student_resource/dataset/train \
    --candidate-file output/candidate_pairs.tsv \
    --output-dir models/deberta \
    --epochs 3 \
    --batch-size 16

# Run inference to produce match probabilities:
python -m src.member3.predict \
    --model-dir models/deberta \
    --candidate-file output/candidate_pairs.tsv \
    --train-dir student_resource/dataset/train \
    --output-file output/pair_predictions.tsv

# Or run fast baseline matching:
python run_pipeline.py --mode baseline --data_dir student_resource/dataset/train --output_dir output
```

---

### 6.4 Submission Validation

Verify that `output/matching_results.tsv` strictly adheres to competition constraints:

```bash
cd student_resource
python utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir dataset/test
```

---

### 6.5 Running the Test Suite

Run all unit tests across all member modules:

```bash
# Run all tests via unittest:
python -m unittest discover -s tests -p "test_*.py"

# Or run individual module tests:
python -m unittest tests/test_data_cleaning.py
python -m unittest tests/test_blocking.py
python -m unittest tests/test_model.py
python -m unittest tests/test_metrics.py
```
