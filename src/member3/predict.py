"""
predict.py — Inference pipeline for scoring candidate pairs.

Produces an intermediate pair-level probability file::

    output/pair_predictions.tsv

with columns: source1_entity_id, candidate_entity_id, probability

Does NOT produce matching_results.tsv (deferred to threshold/postprocess).

CLI usage::

    python -m src.member3.predict \\
        --model-dir models/deberta \\
        --candidate-file output/candidate_pairs.tsv \\
        --train-dir student_resource/dataset/train \\
        --output-file output/pair_predictions.tsv
"""

import argparse
import csv
import os
import sys
import time
from typing import Dict, FrozenSet, List

import torch
from torch.utils.data import DataLoader

from src.member3.candidate_loader import load_candidate_pairs
from src.member3.data_loader import load_source_tsv
from src.member3.model import get_device, load_checkpoint
from src.member3.pair_dataset import PairDataset, PairRecord, build_inference_pairs


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(
    model,
    dataloader: DataLoader,
    device: torch.device,
) -> List[float]:
    """Score all pairs and return match probabilities."""
    model.eval()
    all_probs: List[float] = []

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1)[:, 1]
        all_probs.extend(probs.cpu().tolist())

    return all_probs


def write_pair_predictions(
    pairs: List[PairRecord],
    probabilities: List[float],
    output_path: str,
) -> None:
    """Write pair-level predictions to TSV."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, delimiter="\t")
        writer.writerow(["source1_entity_id", "candidate_entity_id", "probability"])
        for rec, prob in zip(pairs, probabilities):
            writer.writerow([rec.s1_id, rec.candidate_id, f"{prob:.6f}"])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run DeBERTa inference on candidate pairs.",
    )
    parser.add_argument("--model-dir", default="models/deberta",
                        help="Directory with fine-tuned model checkpoint.")
    parser.add_argument("--candidate-file", default="output/candidate_pairs.tsv")
    parser.add_argument("--train-dir", default="student_resource/dataset/train",
                        help="Source data directory (for entity text lookup).")
    parser.add_argument("--output-file", default="output/pair_predictions.tsv")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=256)

    args = parser.parse_args(argv)
    device = get_device()
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    print(f"[1/4] Loading model from {args.model_dir} ...")
    model, tokenizer = load_checkpoint(args.model_dir)
    model.to(device)
    print(f"       Loaded ({time.perf_counter() - t0:.1f}s)")

    # ------------------------------------------------------------------
    # 2. Load source data + candidates
    # ------------------------------------------------------------------
    t1 = time.perf_counter()
    print(f"[2/4] Loading source data from {args.train_dir} ...")
    source1 = load_source_tsv(
        os.path.join(args.train_dir, "train_source1.tsv")
    )
    source2 = load_source_tsv(
        os.path.join(args.train_dir, "train_source2.tsv")
    )
    source3 = load_source_tsv(
        os.path.join(args.train_dir, "train_source3.tsv")
    )
    print(f"       source1={len(source1):,}  source2={len(source2):,}  "
          f"source3={len(source3):,}  ({time.perf_counter() - t1:.1f}s)")

    t2 = time.perf_counter()
    print(f"[3/4] Loading candidates from {args.candidate_file} ...")
    candidate_pairs, _ = load_candidate_pairs(args.candidate_file)
    s1_ids = sorted(candidate_pairs.keys())
    print(f"       {len(s1_ids):,} S1 entities ({time.perf_counter() - t2:.1f}s)")

    # ------------------------------------------------------------------
    # 3. Build inference pairs
    # ------------------------------------------------------------------
    pairs = build_inference_pairs(
        s1_ids, source1, source2, source3, candidate_pairs
    )
    print(f"       {len(pairs):,} total pairs to score")

    if not pairs:
        print("[WARNING] No pairs to score. Writing empty output.")
        write_pair_predictions([], [], args.output_file)
        return

    dataset = PairDataset(pairs, tokenizer, max_length=args.max_length)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # ------------------------------------------------------------------
    # 4. Inference
    # ------------------------------------------------------------------
    t3 = time.perf_counter()
    print(f"[4/4] Running inference ({len(pairs):,} pairs, "
          f"batch_size={args.batch_size}) ...")
    probabilities = run_inference(model, loader, device)
    t_inf = time.perf_counter() - t3
    print(f"       Done ({t_inf:.1f}s, {len(pairs)/max(t_inf,0.001):.0f} pairs/sec)")

    # ------------------------------------------------------------------
    # Write output
    # ------------------------------------------------------------------
    write_pair_predictions(pairs, probabilities, args.output_file)
    total = time.perf_counter() - t0
    print(f"\nPair predictions written to: {args.output_file}")
    print(f"Total time: {total:.1f}s")


if __name__ == "__main__":
    main()
