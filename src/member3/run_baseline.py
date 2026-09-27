"""
run_baseline.py — CLI to evaluate cheap baselines on a validation split.

Usage::

    python -m src.member3.run_baseline \\
        --train-dir student_resource/dataset/train \\
        --validation-fraction 0.01 \\
        --seed 42 \\
        --strategy exact_name

Pipeline::

    load data
    → create S1 validation split
    → build S2/S3 exact-name index
    → generate predictions
    → evaluate with metrics.py
    → print results
    → optionally write output/validation_predictions.tsv
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, FrozenSet

from src.member3.baseline import STRATEGIES, build_name_index, predict_batch
from src.member3.data_loader import load_train_data, split_s1_ids
from src.member3.metrics import evaluate_predictions


def write_predictions_tsv(
    predictions: Dict[str, FrozenSet[str]],
    path: str,
) -> None:
    """Write predictions in the official submission format."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["source1_entity_id", "matched_entity_ids"])
        for s1_id in sorted(predictions):
            matched = ",".join(sorted(predictions[s1_id]))
            writer.writerow([s1_id, matched])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run a cheap baseline on a validation split and report F0.5.",
    )
    parser.add_argument(
        "--train-dir",
        default="student_resource/dataset/train",
        help="Directory with train_source{1,2,3}.tsv and train_ground_truth.tsv.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.01,
        help="Fraction of S1 entities for validation (default 0.01 = 1%%).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split.",
    )
    parser.add_argument(
        "--strategy",
        choices=sorted(STRATEGIES),
        default="exact_name",
        help="Matching strategy to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for validation_predictions.tsv (default: output/).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Skip writing validation_predictions.tsv.",
    )

    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print(f"[1/5] Loading data from {args.train_dir} ...")
    source1, source2, source3, ground_truth = load_train_data(args.train_dir)
    t_load = time.perf_counter() - t0
    print(
        f"       source1={len(source1):,}  source2={len(source2):,}  "
        f"source3={len(source3):,}  ground_truth={len(ground_truth):,}  "
        f"({t_load:.1f}s)"
    )

    # ------------------------------------------------------------------
    # 2. Validation split
    # ------------------------------------------------------------------
    t1 = time.perf_counter()
    print(
        f"[2/5] Splitting S1 entities (val_frac={args.validation_fraction}, "
        f"seed={args.seed}) ..."
    )
    all_s1_ids = list(ground_truth.keys())
    _train_ids, val_ids = split_s1_ids(
        all_s1_ids,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    t_split = time.perf_counter() - t1
    print(
        f"       train={len(_train_ids):,}  val={len(val_ids):,}  ({t_split:.1f}s)"
    )

    # ------------------------------------------------------------------
    # 3. Build index
    # ------------------------------------------------------------------
    t2 = time.perf_counter()
    print(f"[3/5] Building name index over S2 + S3 ...")
    s2_index = build_name_index(source2)
    s3_index = build_name_index(source3)
    t_index = time.perf_counter() - t2
    print(
        f"       s2_index={len(s2_index):,} names  "
        f"s3_index={len(s3_index):,} names  ({t_index:.1f}s)"
    )

    # ------------------------------------------------------------------
    # 4. Predict
    # ------------------------------------------------------------------
    t3 = time.perf_counter()
    print(f"[4/5] Running strategy={args.strategy!r} on {len(val_ids):,} val entities ...")
    predictions = predict_batch(
        val_ids, source1, s2_index, s3_index, strategy=args.strategy
    )
    t_pred = time.perf_counter() - t3
    print(f"       predictions generated ({t_pred:.1f}s)")

    # ------------------------------------------------------------------
    # 5. Evaluate
    # ------------------------------------------------------------------
    t4 = time.perf_counter()
    print("[5/5] Evaluating ...")
    val_gt = {s1: ground_truth[s1] for s1 in val_ids}
    result = evaluate_predictions(val_gt, predictions)
    t_eval = time.perf_counter() - t4

    total_time = time.perf_counter() - t0

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print()
    print("=" * 60)
    print(f" Strategy          : {args.strategy}")
    print(f" Validation entities: {result.entity_count:,}")
    print(f" Macro Precision   : {result.macro_precision:.6f}")
    print(f" Macro Recall      : {result.macro_recall:.6f}")
    print(f" Macro F0.5        : {result.macro_f05:.6f}")
    print(f" Eval time         : {t_eval:.1f}s")
    print(f" Total time        : {total_time:.1f}s")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Optionally write predictions
    # ------------------------------------------------------------------
    if not args.no_write:
        out_path = os.path.join(args.output_dir, "validation_predictions.tsv")
        write_predictions_tsv(predictions, out_path)
        print(f"\nValidation predictions written to: {out_path}")

    return result


if __name__ == "__main__":
    main()
