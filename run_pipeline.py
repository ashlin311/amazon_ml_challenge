#!/usr/bin/env python3
"""
run_pipeline.py — Master End-to-End Pipeline for Business Entity Resolution.

Integrates all team members' contributions into a unified execution flow:
  1. Data Cleaning & Normalization (Member 1: src.data_cleaning)
  2. Candidate Retrieval / Hybrid Blocking (Member 2: src.blocking_engine)
  3. Model Matching / Cross-Encoder Scoring (Member 3: src.model_matching)
  4. Submission Validation (student_resource/utils/validate_submission.py)

Usage Examples:
  # Run full end-to-end pipeline on training dataset:
  python run_pipeline.py --mode all --data_dir student_resource/dataset/train --output_dir output

  # Run only data cleaning stage (Member 1):
  python run_pipeline.py --mode clean --data_dir student_resource/dataset/train --output_dir output/cleaned

  # Run blocking stage (Member 2):
  python run_pipeline.py --mode blocking --data_dir student_resource/dataset/train --output_dir output

  # Run baseline matching stage (Member 3):
  python run_pipeline.py --mode baseline --data_dir student_resource/dataset/train --output_dir output

  # Run end-to-end test on small sample (fast laptop benchmark):
  python run_pipeline.py --mode all --data_dir student_resource/dataset/train --sample_size 100 --skip_dense
"""

import argparse
import glob
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import pandas as pd

# Add repo root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from src.data_cleaning import clean_dataframe, clean_source_tsv, normalize_text
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
logger = logging.getLogger("Pipeline")


def find_file(directory: str, patterns: List[str]) -> Optional[str]:
    """Find a file matching any of the glob patterns inside directory."""
    for pattern in patterns:
        matches = glob.glob(os.path.join(directory, pattern))
        if matches:
            return matches[0]
    return None


def run_stage_cleaning(data_dir: str, cleaned_dir: str) -> Dict[str, str]:
    """Stage 1: Clean and normalize source TSV files using Member 1's normalizer."""
    logger.info("=" * 60)
    logger.info("STAGE 1: DATA CLEANING & NORMALIZATION (MEMBER 1)")
    logger.info("=" * 60)

    os.makedirs(cleaned_dir, exist_ok=True)
    cleaned_paths = {}

    for src_name in ["source1", "source2", "source3"]:
        raw_path = find_file(data_dir, [f"*{src_name}*.tsv"])
        if not raw_path:
            logger.warning(f"Could not find {src_name} in {data_dir}")
            continue

        out_path = os.path.join(cleaned_dir, f"{src_name}_cleaned.tsv")
        logger.info(f"Cleaning {raw_path} -> {out_path}...")
        t0 = time.time()
        clean_source_tsv(raw_path, out_path)
        logger.info(f"Cleaned {src_name} in {time.time() - t0:.2f}s")
        cleaned_paths[src_name] = out_path

    logger.info("Stage 1 (Data Cleaning) completed successfully.")
    return cleaned_paths


def run_stage_blocking(
    data_dir: str,
    output_dir: str,
    top_k_sparse: int = 40,
    top_k_dense: int = 35,
    max_candidates: int = 80,
    batch_size: int = 2000,
    sample_size: Optional[int] = None,
    sample_targets: Optional[int] = None,
    skip_dense: bool = False,
    device: Optional[str] = None,
) -> str:
    """Stage 2: Generate candidate pairs using Member 2's hybrid blocking engine."""
    logger.info("=" * 60)
    logger.info("STAGE 2: HYBRID BLOCKING & CANDIDATE RETRIEVAL (MEMBER 2)")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    candidate_output_path = os.path.join(output_dir, "candidate_pairs.tsv")

    s1_path = find_file(data_dir, ["*source1*.tsv"])
    s2_path = find_file(data_dir, ["*source2*.tsv"])
    s3_path = find_file(data_dir, ["*source3*.tsv"])
    gt_path = find_file(data_dir, ["*ground_truth*.tsv"])

    if not s1_path or not s2_path or not s3_path:
        raise FileNotFoundError(f"Missing required source TSV files in {data_dir}")

    # Load Source 1
    logger.info(f"Loading Source 1 records from {s1_path}...")
    df_s1 = pd.read_csv(s1_path, sep="\t", dtype=str)
    if sample_size is not None and len(df_s1) > sample_size:
        logger.info(f"Sampling Source 1 to first {sample_size} records...")
        df_s1 = df_s1.iloc[:sample_size].copy()

    s1_ids = df_s1["entity_id"].fillna("").tolist()
    s1_names = df_s1["business_name"].fillna("").tolist()
    s1_addresses = df_s1["business_address"].fillna("").tolist()

    # Load Targets
    logger.info(f"Loading Source 2 & 3 targets...")
    df_s2 = pd.read_csv(s2_path, sep="\t", dtype=str, nrows=sample_targets)
    df_s3 = pd.read_csv(s3_path, sep="\t", dtype=str, nrows=sample_targets)

    target_ids = df_s2["entity_id"].fillna("").tolist() + df_s3["entity_id"].fillna("").tolist()
    target_names = df_s2["business_name"].fillna("").tolist() + df_s3["business_name"].fillna("").tolist()
    target_addresses = df_s2["business_address"].fillna("").tolist() + df_s3["business_address"].fillna("").tolist()

    blocker = HybridBlocker(
        top_k_sparse=top_k_sparse,
        top_k_dense=top_k_dense,
        batch_size=batch_size,
        device=device,
        enable_sparse=True,
        enable_dense=not skip_dense,
        enable_phonetic=True,
    )

    t0 = time.time()
    blocker.fit_targets(target_ids=target_ids, target_names=target_names, target_addresses=target_addresses)
    logger.info(f"Fitted target index with {len(target_ids)} records in {time.time() - t0:.2f}s")

    t1 = time.time()
    candidate_dict = blocker.block_all(
        query_ids=s1_ids,
        query_names=s1_names,
        query_addresses=s1_addresses,
        max_candidates=max_candidates,
    )
    logger.info(f"Blocked {len(s1_ids)} queries in {time.time() - t1:.2f}s")

    export_candidates(candidate_dict=candidate_dict, output_path=candidate_output_path, required_s1_ids=s1_ids)
    logger.info(f"Exported candidates to {candidate_output_path}")

    if gt_path and os.path.isfile(gt_path):
        metrics = evaluate_blocking_recall(
            ground_truth_path_or_dict=gt_path,
            candidate_dict=candidate_dict,
            target_count=len(target_ids),
        )
        logger.info(f"Blocking Recall: {metrics['blocking_recall']:.4%}")
        logger.info(f"Avg Candidates / Entity: {metrics['avg_candidates_per_entity']:.2f}")

    return candidate_output_path


def run_stage_matching(
    data_dir: str,
    candidate_file: str,
    output_dir: str,
    use_baseline: bool = True,
) -> str:
    """Stage 3: Run entity matching scoring (Member 3)."""
    logger.info("=" * 60)
    logger.info("STAGE 3: ENTITY MATCHING & PREDICTION (MEMBER 3)")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    matching_output = os.path.join(output_dir, "matching_results.tsv")

    if use_baseline:
        from src.member3.baseline import run_baseline_matching
        from src.member3.candidate_loader import load_candidate_pairs
        from src.member3.data_loader import load_source_tsv

        logger.info("Running Member 3 baseline string matching algorithm...")
        s1_path = find_file(data_dir, ["*source1*.tsv"])
        s2_path = find_file(data_dir, ["*source2*.tsv"])
        s3_path = find_file(data_dir, ["*source3*.tsv"])

        entities = {}
        entities.update(load_source_tsv(s1_path))
        entities.update(load_source_tsv(s2_path))
        entities.update(load_source_tsv(s3_path))

        candidates = load_candidate_pairs(candidate_file)
        results = run_baseline_matching(candidates=candidates, entities=entities)

        # Write matching_results.tsv
        with open(matching_output, "w", encoding="utf-8") as fh:
            fh.write("source1_entity_id\tmatched_entity_ids\n")
            for s1_id, matches in sorted(results.items()):
                fh.write(f"{s1_id}\t{','.join(sorted(matches))}\n")

        logger.info(f"Wrote matching results to {matching_output}")
    else:
        logger.info("DeBERTa inference can be run via: python -m src.member3.predict")

    return matching_output


def main():
    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge 2026 — Master Pipeline Runner"
    )
    parser.add_argument(
        "--mode",
        choices=["all", "clean", "blocking", "match", "baseline", "eval"],
        default="all",
        help="Pipeline mode to execute.",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="student_resource/dataset/train",
        help="Input dataset directory containing source TSVs.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Output directory for generated artifacts.",
    )
    parser.add_argument(
        "--top_k_sparse",
        type=int,
        default=40,
        help="Top-K sparse candidates per entity.",
    )
    parser.add_argument(
        "--top_k_dense",
        type=int,
        default=35,
        help="Top-K dense candidates per entity.",
    )
    parser.add_argument(
        "--max_candidates",
        type=int,
        default=80,
        help="Max candidate pairs per entity.",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Sample first N Source 1 entities (for fast testing).",
    )
    parser.add_argument(
        "--sample_targets",
        type=int,
        default=None,
        help="Sample first N Source 2/3 records (for fast testing).",
    )
    parser.add_argument(
        "--skip_dense",
        action="store_true",
        help="Skip dense embedding computation in blocking.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run submission format validation script on output.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    candidate_file = os.path.join(args.output_dir, "candidate_pairs.tsv")

    if args.mode in ("all", "clean"):
        cleaned_dir = os.path.join(args.output_dir, "cleaned")
        run_stage_cleaning(args.data_dir, cleaned_dir)

    if args.mode in ("all", "blocking"):
        candidate_file = run_stage_blocking(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            top_k_sparse=args.top_k_sparse,
            top_k_dense=args.top_k_dense,
            max_candidates=args.max_candidates,
            sample_size=args.sample_size,
            sample_targets=args.sample_targets,
            skip_dense=args.skip_dense,
            device=args.device,
        )

    if args.mode in ("all", "match", "baseline"):
        if not os.path.isfile(candidate_file):
            logger.error(f"Candidate file {candidate_file} not found. Run blocking first.")
            sys.exit(1)
        run_stage_matching(
            data_dir=args.data_dir,
            candidate_file=candidate_file,
            output_dir=args.output_dir,
            use_baseline=True,
        )

    if args.validate or args.mode in ("all", "eval"):
        matching_file = os.path.join(args.output_dir, "matching_results.tsv")
        if os.path.isfile(matching_file):
            logger.info("Validating submission format...")
            run_submission_validation(
                candidate_tsv_path=matching_file,
                test_dir=args.data_dir,
            )

    logger.info("Pipeline execution finished successfully.")


if __name__ == "__main__":
    main()
