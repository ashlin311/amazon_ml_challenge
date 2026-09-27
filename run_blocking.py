#!/usr/bin/env python3
"""
CLI Launcher Script for High-Recall Hybrid Blocking & Candidate Retrieval.

Usage:
    # Run on training data with validation against ground truth:
    python run_blocking.py --data_dir student_resource/dataset/train --output_dir output --top_k_sparse 40 --top_k_dense 35

    # Run on test data for submission generation:
    python run_blocking.py --data_dir student_resource/dataset/test --output_dir output --top_k_sparse 40 --top_k_dense 35

    # Fast benchmark on sample of 1000 records:
    python run_blocking.py --data_dir student_resource/dataset/train --output_dir output --sample_size 1000
"""

import argparse
import gc
import glob
import logging
import os
import sys
import time
from typing import Optional

import pandas as pd

# Add workspace and src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.blocking_engine import (
    HybridBlocker,
    evaluate_blocking_recall,
    export_candidates,
    run_submission_validation,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("RunBlocking")


def find_file(directory: str, patterns: list) -> Optional[str]:
    """Find a file matching any of the glob patterns inside directory."""
    for pattern in patterns:
        matches = glob.glob(os.path.join(directory, pattern))
        if matches:
            return matches[0]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="High-Recall Hybrid Blocking & Candidate Retrieval Engine CLI"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="student_resource/dataset/train",
        help="Path to folder containing source1/2/3 and ground_truth TSV files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save output/candidate_pairs.tsv",
    )
    parser.add_argument(
        "--top_k_sparse",
        type=int,
        default=40,
        help="Top-K candidates per entity from Blocker 1 (Sparse TF-IDF)",
    )
    parser.add_argument(
        "--top_k_dense",
        type=int,
        default=35,
        help="Top-K candidates per entity from Blocker 2 (Dense FAISS)",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=80,
        help="Maximum combined candidates ceiling per Source 1 entity (default: 80)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2000,
        help="Batch size for vector search and sparse multiplication (OOM prevention)",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Optional: sample first N Source 1 entities for fast evaluation/testing",
    )
    parser.add_argument(
        "--sample_targets",
        type=int,
        default=None,
        help="Optional: sample first N target records from Source 2 and Source 3 for lightweight laptop testing",
    )
    parser.add_argument(
        "--skip_dense",
        action="store_true",
        help="Disable dense semantic vector retrieval (useful for fast CPU-only runs)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for dense encoder ('cuda' or 'cpu')",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validate_submission.py against generated candidate_pairs.tsv",
    )

    args = parser.parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    candidate_output_path = os.path.join(output_dir, "candidate_pairs.tsv")

    # Locate source files
    s1_path = find_file(data_dir, ["*source1*.tsv"])
    s2_path = find_file(data_dir, ["*source2*.tsv"])
    s3_path = find_file(data_dir, ["*source3*.tsv"])
    gt_path = find_file(data_dir, ["*ground_truth*.tsv"])

    if not s1_path or not s2_path or not s3_path:
        logger.error(
            f"Could not find source1/2/3 files in {data_dir}. "
            f"Found: s1={s1_path}, s2={s2_path}, s3={s3_path}"
        )
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("BUSINESS ENTITY RESOLUTION - HYBRID BLOCKING PIPELINE")
    logger.info(f"  Data Directory:      {data_dir}")
    logger.info(f"  Source 1 Path:       {s1_path}")
    logger.info(f"  Source 2 Path:       {s2_path}")
    logger.info(f"  Source 3 Path:       {s3_path}")
    logger.info(f"  Ground Truth Path:   {gt_path}")
    logger.info(f"  Output Path:         {candidate_output_path}")
    logger.info(f"  Top-K Sparse:        {args.top_k_sparse}")
    logger.info(f"  Top-K Dense:         {args.top_k_dense}")
    logger.info(f"  Max Candidates:      {args.max_candidates}")
    logger.info(f"  Batch Size:          {args.batch_size}")
    logger.info(f"  Sample Size (S1):    {args.sample_size}")
    logger.info(f"  Sample Targets (S2/3): {args.sample_targets}")
    logger.info("=" * 60)

    # Load Source 1
    logger.info(f"Loading Source 1 records from {s1_path}...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    if args.sample_size is not None and len(df_s1) > args.sample_size:
        logger.info(f"Subsampling Source 1 to {args.sample_size} records for testing...")
        df_s1 = df_s1.iloc[: args.sample_size].copy()

    s1_ids = df_s1["entity_id"].fillna("").tolist()
    s1_names = df_s1["business_name"].fillna("").tolist()
    s1_addresses = df_s1["business_address"].fillna("").tolist()
    logger.info(f"Loaded {len(s1_ids)} Source 1 records.")
    del df_s1
    gc.collect()

    # Load Source 2 & Source 3 targets
    logger.info(f"Loading Source 2 records from {s2_path}...")
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str, nrows=args.sample_targets)
    logger.info(f"Loaded {len(df_s2)} Source 2 records.")

    logger.info(f"Loading Source 3 records from {s3_path}...")
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str, nrows=args.sample_targets)
    logger.info(f"Loaded {len(df_s3)} Source 3 records.")

    # Target records pool
    target_ids = df_s2["entity_id"].fillna("").tolist() + df_s3["entity_id"].fillna("").tolist()
    target_names = df_s2["business_name"].fillna("").tolist() + df_s3["business_name"].fillna("").tolist()
    target_addresses = df_s2["business_address"].fillna("").tolist() + df_s3["business_address"].fillna("").tolist()
    logger.info(f"Total Target records (Source 2 + Source 3): {len(target_ids)}")

    # Free large DataFrames to recover 5-6 GB of memory before indexing
    del df_s2, df_s3
    gc.collect()

    # Initialize Hybrid Blocker
    blocker = HybridBlocker(
        top_k_sparse=args.top_k_sparse,
        top_k_dense=args.top_k_dense,
        batch_size=args.batch_size,
        device=args.device,
        enable_sparse=True,
        enable_dense=not args.skip_dense,
        enable_phonetic=True,
    )

    # Fit targets
    start_fit = time.time()
    blocker.fit_targets(
        target_ids=target_ids,
        target_names=target_names,
        target_addresses=target_addresses,
    )
    logger.info(f"Target indexing completed in {time.time() - start_fit:.2f}s")

    # Run blocking on queries
    start_block = time.time()
    candidate_dict = blocker.block_all(
        query_ids=s1_ids,
        query_names=s1_names,
        query_addresses=s1_addresses,
        max_candidates=args.max_candidates,
    )
    logger.info(f"Blocking queries completed in {time.time() - start_block:.2f}s")

    # Export candidate pairs TSV
    export_candidates(
        candidate_dict=candidate_dict,
        output_path=candidate_output_path,
        required_s1_ids=s1_ids,
    )

    # Evaluate Recall if ground truth is present
    if gt_path and os.path.isfile(gt_path):
        logger.info(f"Evaluating Blocking Recall against ground truth: {gt_path}...")
        metrics = evaluate_blocking_recall(
            ground_truth_path_or_dict=gt_path,
            candidate_dict=candidate_dict,
            target_count=len(target_ids),
        )
        print("\n" + "=" * 50)
        print(f"FINAL BLOCKING RECALL: {metrics['blocking_recall']:.4%}")
        print(f"AVG CANDIDATES / ENTITY: {metrics['avg_candidates_per_entity']:.2f}")
        print(f"REDUCTION RATIO: {metrics['reduction_ratio']:.6f}")
        print("=" * 50 + "\n")

    # Validate output format if requested
    if args.validate:
        logger.info("Running official submission validation script...")
        val_success = run_submission_validation(
            candidate_tsv_path=candidate_output_path,
            test_dir=data_dir,
        )
        if val_success:
            logger.info("Validation PASSED! Output candidate_pairs.tsv is 100% compliant.")
        else:
            logger.warning("Validation returned non-zero status. Check logs above.")


if __name__ == "__main__":
    main()
