# Amazon ML Challenge 2026 — Business Entity Resolution

## Team Structure

| Member | Responsibilities |
|--------|-----------------|
| Member 1 | Data normalization, feature engineering |
| Member 2 | Blocking / candidate generation |
| Member 3 | DeBERTa matching model, pairwise classification, F0.5 evaluation, threshold optimization, post-processing, final predictions |

## Pipeline Overview

```
student_resource/dataset/train/   (source TSV files)
        ↓
run_blocking.py  (src/blocking_engine.py)       ← hybrid TF-IDF + FAISS + phonetic blocking
        ↓
output/candidate_pairs.tsv
        ↓
pair construction / label assignment            ← src/member3/pair_builder.py
        ↓
DeBERTa cross-encoder scoring                   ← src/member3/dataset.py, model.py, train.py
        ↓
match probabilities                             ← src/member3/predict.py
        ↓
F0.5 evaluation                                 ← src/member3/metrics.py
        ↓
threshold optimization                          ← src/member3/threshold.py
        ↓
singleton / confidence post-processing          ← src/member3/postprocess.py
        ↓
output/matching_results.tsv
```

## Data Format

### train_source{1,2,3}.tsv

```
entity_id    business_name    business_address    country
S1-00001     Acme Corp        123 Main St         US
```

### candidate_pairs.tsv (output from blocking)

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
│       └── validate_submission.py
├── src/
│   ├── __init__.py
│   ├── data_cleaning.py            ← Facade: Member 1 data cleaning & normalization
│   ├── blocking_engine.py          ← Facade: Member 2 hybrid blocking engine
│   ├── model_matching.py           ← Facade: Member 3 DeBERTa matching pipeline
│   ├── member1/                    ← Member 1: Data cleaning & feature engineering
│   │   ├── __init__.py
│   │   └── data_cleaning.py
│   ├── member2/                    ← Member 2: Candidate retrieval & blocking
│   │   ├── __init__.py
│   │   ├── blocking_engine.py
│   │   └── phonetics.py
│   └── member3/                    ← Member 3: Matching pipeline
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
│   ├── test_data_cleaning.py
│   ├── test_blocking.py
│   ├── test_candidate_eval.py
│   ├── test_metrics.py
│   ├── test_model.py
│   ├── test_pair_dataset.py
│   └── test_train.py
├── models/
│   └── deberta/                    ← fine-tuned model checkpoints
├── output/                         ← pipeline outputs
├── run_blocking.py                 ← Blocking stage launcher
├── run_pipeline.py                 ← Master end-to-end pipeline launcher
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Place dataset files

Copy the competition TSV files into:
- `student_resource/dataset/train/` — train_source1.tsv, train_source2.tsv, train_source3.tsv, train_ground_truth.tsv
- `student_resource/dataset/test/` — test_source1.tsv, test_source2.tsv, test_source3.tsv

### 3. Run blocking (candidate generation)

```bash
python run_blocking.py --data_dir student_resource/dataset/train --output_dir output
```

Or via the pipeline launcher:

```bash
python run_pipeline.py --mode blocking --data_dir student_resource/dataset/train
```

### 4. Train DeBERTa model

```bash
python -m src.member3.train \
    --train-dir student_resource/dataset/train \
    --candidate-file output/candidate_pairs.tsv \
    --output-dir models/deberta
```

### 5. Run inference

```bash
python -m src.member3.predict \
    --model-dir models/deberta \
    --candidate-file output/candidate_pairs.tsv \
    --train-dir student_resource/dataset/train \
    --output-file output/pair_predictions.tsv
```

### 6. Validate submission

```bash
cd student_resource
python3 utils/validate_submission.py \
    --matching ../output/matching_results.tsv \
    --candidate ../output/candidate_pairs.tsv \
    --test-dir dataset/test
```
