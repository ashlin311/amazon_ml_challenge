"""
train.py — Training loop for the DeBERTa cross-encoder matching model.

CLI usage::

    python -m src.member3.train \\
        --train-dir student_resource/dataset/train \\
        --candidate-file output/candidate_pairs.tsv \\
        --output-dir models/deberta \\
        --epochs 3 --batch-size 32 --max-length 256

Pipeline::

    load source data + ground truth
    → load candidate pairs
    → build labeled pairs (S1-level split)
    → tokenize with DeBERTa tokenizer
    → train with cross-entropy loss
    → evaluate on validation split
    → save best checkpoint
"""

import argparse
import os
import sys
import time
from collections import defaultdict
from typing import Dict, FrozenSet, List, Sequence

import torch
from torch.utils.data import DataLoader

from src.member3.candidate_loader import load_candidate_pairs
from src.member3.data_loader import (
    GroundTruth,
    load_ground_truth,
    load_source_tsv,
    split_s1_ids,
)
from src.member3.metrics import evaluate_predictions
from src.member3.model import (
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_NAME,
    get_device,
    load_model,
    load_tokenizer,
    save_checkpoint,
)
from src.member3.pair_dataset import (
    PairDataset,
    PairRecord,
    build_labeled_pairs,
)


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_one_epoch(model, dataloader, optimizer, device) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_validation(
    model,
    dataloader,
    pairs: List[PairRecord],
    ground_truth: GroundTruth,
    val_s1: Sequence[str],
    device,
    threshold: float = 0.5,
) -> dict:
    """Evaluate on validation set. Returns loss + pair/entity metrics."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_probs = []
    all_labels = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
        n_batches += 1

        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / max(n_batches, 1)

    # Pair-level metrics
    tp = fp = fn = tn = 0
    for prob, label in zip(all_probs, all_labels):
        pred = 1 if prob >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 1:
            fn += 1
        else:
            tn += 1

    pair_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    pair_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    pair_f05 = (
        (1.25 * pair_precision * pair_recall)
        / (0.25 * pair_precision + pair_recall)
        if (pair_precision + pair_recall) > 0
        else 0.0
    )

    # Entity-level metrics: aggregate pair predictions into S1 -> set(pred_ids)
    entity_preds: Dict[str, set] = defaultdict(set)
    for i, rec in enumerate(pairs):
        if all_probs[i] >= threshold:
            entity_preds[rec.s1_id].add(rec.candidate_id)

    # Build val ground truth subset explicitly from val_s1
    val_gt = {s1: ground_truth.get(s1, frozenset()) for s1 in val_s1}
    val_preds = {s1: frozenset(entity_preds.get(s1, set())) for s1 in val_s1}

    entity_result = evaluate_predictions(val_gt, val_preds)

    return {
        "val_loss": avg_loss,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_f05": pair_f05,
        "pair_tp": tp,
        "pair_fp": fp,
        "pair_fn": fn,
        "pair_tn": tn,
        "entity_macro_precision": entity_result.macro_precision,
        "entity_macro_recall": entity_result.macro_recall,
        "entity_macro_f05": entity_result.macro_f05,
        "entity_count": entity_result.entity_count,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train DeBERTa cross-encoder on candidate pairs.",
    )
    parser.add_argument("--train-dir", default="student_resource/dataset/train")
    parser.add_argument("--candidate-file", default="output/candidate_pairs.tsv")
    parser.add_argument("--output-dir", default="models/deberta")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument(
        "--negative-sampling-ratio",
        type=float,
        default=1.0,
        help="Fraction of negatives to keep (1.0 = all).",
    )
    parser.add_argument(
        "--max-s1",
        type=int,
        default=None,
        help="Limit number of S1 entities for smoke-testing.",
    )

    args = parser.parse_args(argv)
    torch.manual_seed(args.seed)

    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print(f"[1/6] Loading source data from {args.train_dir} ...")

    source1 = load_source_tsv(
        os.path.join(args.train_dir, "train_source1.tsv")
    )
    source2 = load_source_tsv(
        os.path.join(args.train_dir, "train_source2.tsv")
    )
    source3 = load_source_tsv(
        os.path.join(args.train_dir, "train_source3.tsv")
    )
    ground_truth = load_ground_truth(
        os.path.join(args.train_dir, "train_ground_truth.tsv")
    )
    print(
        f"       source1={len(source1):,}  source2={len(source2):,}  "
        f"source3={len(source3):,}  gt={len(ground_truth):,}  "
        f"({time.perf_counter() - t0:.1f}s)"
    )

    # ------------------------------------------------------------------
    # 2. Load candidate pairs
    # ------------------------------------------------------------------
    t1 = time.perf_counter()
    print(f"[2/6] Loading candidate pairs from {args.candidate_file} ...")
    candidate_pairs, dup_count = load_candidate_pairs(args.candidate_file)
    print(f"       {len(candidate_pairs):,} S1 entries ({time.perf_counter() - t1:.1f}s)")

    # ------------------------------------------------------------------
    # 3. S1-level train/val split
    # ------------------------------------------------------------------
    t2 = time.perf_counter()
    print("[3/6] Splitting S1 entities ...")

    # Split across all ground truth S1 entities so S1 entities with zero
    # candidate pairs can also be represented in validation.
    available_s1 = sorted(ground_truth.keys())
    if args.max_s1 is not None:
        available_s1 = available_s1[: args.max_s1]
        print(f"       [smoke-test] limited to {len(available_s1)} S1 entities")

    train_s1, val_s1 = split_s1_ids(
        available_s1,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    print(
        f"       train_s1={len(train_s1):,}  val_s1={len(val_s1):,}  "
        f"({time.perf_counter() - t2:.1f}s)"
    )

    # ------------------------------------------------------------------
    # 4. Build labeled pairs
    # ------------------------------------------------------------------
    t3 = time.perf_counter()
    print("[4/6] Building labeled pairs ...")

    train_pairs, train_stats = build_labeled_pairs(
        train_s1, source1, source2, source3,
        ground_truth, candidate_pairs,
        negative_sampling_ratio=args.negative_sampling_ratio,
        seed=args.seed,
    )
    val_pairs, val_stats = build_labeled_pairs(
        val_s1, source1, source2, source3,
        ground_truth, candidate_pairs,
        negative_sampling_ratio=1.0,  # keep all val pairs
        seed=args.seed,
    )

    print(f"       Train: {train_stats}")
    print(f"       Val:   {val_stats}")
    print(f"       ({time.perf_counter() - t3:.1f}s)")

    if not train_pairs:
        print("[ERROR] No training pairs. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # 5. Tokenize and create dataloaders
    # ------------------------------------------------------------------
    t4 = time.perf_counter()
    print(f"[5/6] Loading tokenizer + model ({args.model_name}) ...")
    tokenizer = load_tokenizer(args.model_name)
    model = load_model(args.model_name)
    model.to(device)

    train_dataset = PairDataset(train_pairs, tokenizer, max_length=args.max_length)
    val_dataset = PairDataset(val_pairs, tokenizer, max_length=args.max_length)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False
    )
    print(f"       Tokenizer + model loaded ({time.perf_counter() - t4:.1f}s)")

    # ------------------------------------------------------------------
    # 6. Training loop
    # ------------------------------------------------------------------
    print(f"[6/6] Training for {args.epochs} epoch(s) ...")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate
    )

    best_val_f05 = -1.0

    for epoch in range(1, args.epochs + 1):
        t_ep = time.perf_counter()
        train_loss = train_one_epoch(model, train_loader, optimizer, device)

        val_metrics = {}
        if val_s1:
            val_metrics = evaluate_validation(
                model, val_loader, val_pairs, ground_truth, val_s1, device
            )

        elapsed = time.perf_counter() - t_ep
        print(f"\n  Epoch {epoch}/{args.epochs}  ({elapsed:.1f}s)")
        print(f"    Train loss        : {train_loss:.4f}")

        if val_metrics:
            print(f"    Val loss          : {val_metrics['val_loss']:.4f}")
            print(f"    Pair P/R/F0.5     : "
                  f"{val_metrics['pair_precision']:.4f} / "
                  f"{val_metrics['pair_recall']:.4f} / "
                  f"{val_metrics['pair_f05']:.4f}  "
                  f"(TP={val_metrics['pair_tp']} FP={val_metrics['pair_fp']} "
                  f"FN={val_metrics['pair_fn']} TN={val_metrics['pair_tn']})")
            print(f"    Entity F0.5       : "
                  f"{val_metrics['entity_macro_f05']:.4f}  "
                  f"(P={val_metrics['entity_macro_precision']:.4f} "
                  f"R={val_metrics['entity_macro_recall']:.4f} "
                  f"n={val_metrics['entity_count']})")

            # Save best checkpoint
            entity_f05 = val_metrics["entity_macro_f05"]
            if entity_f05 > best_val_f05:
                best_val_f05 = entity_f05
                save_checkpoint(model, tokenizer, args.output_dir)
                print(f"    ✓ Best model saved to {args.output_dir}")

    # Always save final checkpoint if no val data
    if not val_s1:
        save_checkpoint(model, tokenizer, args.output_dir)
        print(f"\n  Model saved to {args.output_dir}")

    total = time.perf_counter() - t0
    print(f"\nTotal training time: {total:.1f}s")
    print(f"Best validation entity F0.5: {best_val_f05:.4f}")


if __name__ == "__main__":
    main()
