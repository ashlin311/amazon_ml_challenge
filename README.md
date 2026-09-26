# Amazon ML Challenge 2026 — Business Entity Resolution

## Team Structure

| Member | Responsibilities |
|--------|-----------------|
| Member 1 | Data normalization, feature engineering |
| Member 2 | Blocking / candidate generation |
| Member 3 | DeBERTa matching model, pairwise classification, F0.5 evaluation, threshold optimization, post-processing, final predictions |

## Pipeline Overview

```
candidate_pairs.tsv  (from Member 2)
        ↓
pair construction / label assignment      ← pair_builder.py
        ↓
DeBERTa cross-encoder scoring            ← dataset.py, model.py, train.py
        ↓
match probabilities                       ← predict.py
        ↓
F0.5 evaluation                           ← metrics.py
        ↓
threshold optimization                    ← threshold.py
        ↓
singleton / confidence post-processing    ← postprocess.py
        ↓
matching_results.tsv                      ← predict.py (final output)
```

## Data Format

### candidate_pairs.tsv (input from Member 2)

```
source1_entity_id    candidate_entity_ids
S1-00001             S2-00047,S2-00193,S3-00812
S1-00002             S3-00004
S1-00003
```

### matching_results.tsv (final submission)

```
source1_entity_id    matched_entity_ids
S1-00001             S2-00047,S3-00812
S1-00002             S3-00004
S1-00003
```

## Directory Layout

```
amazon_ml_challenge/
├── data/          — Raw train/test TSV files
├── src/member3/   — Member 3 source code
├── models/deberta — Fine-tuned model checkpoints
├── outputs/       — Pipeline outputs
├── notebooks/     — Experiment notebooks
├── requirements.txt
└── README.md
```
