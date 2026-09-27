"""
pair_dataset.py — Build labeled pairwise examples from candidate pairs.

Constructs (S1, candidate) pairs with binary labels for DeBERTa
cross-encoder training and inference.

Input representation for each pair::

    Name: <S1 business_name> | Address: <S1 address> | Country: <S1 country>
    [SEP]
    Name: <candidate business_name> | Address: <candidate address> | Country: <candidate country>

Labels:
    1 = candidate entity ID is in the ground-truth matches for this S1
    0 = candidate entity ID is NOT in the ground-truth matches

Split is at the S1-entity level (no leakage).
"""

import random
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

import torch
from torch.utils.data import Dataset

from src.member3.data_loader import EntityRow, GroundTruth, split_s1_ids

# ---------------------------------------------------------------------------
# Pair record
# ---------------------------------------------------------------------------

class PairRecord(NamedTuple):
    """One candidate pair with metadata."""
    s1_id: str
    candidate_id: str
    text_a: str  # S1 entity text
    text_b: str  # Candidate entity text
    label: int   # 1 = match, 0 = non-match, -1 = unknown (inference)


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------

def format_entity_text(row: Optional[EntityRow]) -> str:
    """Build a field-labeled text representation for one entity.

    Returns a string like::

        Name: Acme Corp | Address: 123 Main St | Country: US

    Missing/empty fields are included as empty strings so the model
    always sees the same field structure.
    """
    if row is None:
        return "Name:  | Address:  | Country: "

    name = (row.get("business_name") or "").strip()
    address = (row.get("business_address") or "").strip()
    country = (row.get("country") or "").strip()

    return f"Name: {name} | Address: {address} | Country: {country}"


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

class PairStats(NamedTuple):
    total_pairs: int
    positive_pairs: int
    negative_pairs: int
    positive_ratio: float
    unique_s1: int
    unique_candidates: int


def build_labeled_pairs(
    s1_ids: List[str],
    source1: Dict[str, EntityRow],
    source2: Dict[str, EntityRow],
    source3: Dict[str, EntityRow],
    ground_truth: GroundTruth,
    candidate_pairs: Dict[str, FrozenSet[str]],
    *,
    negative_sampling_ratio: float = 1.0,
    seed: int = 42,
) -> Tuple[List[PairRecord], PairStats]:
    """Build labeled pairs from candidate sets for a given list of S1 IDs.

    Parameters
    ----------
    s1_ids : list[str]
        S1 entity IDs to process (train or val subset).
    source1, source2, source3 : dict
        Entity data keyed by entity_id.
    ground_truth : GroundTruth
        ``{s1_id: frozenset(matched_ids)}``.
    candidate_pairs : dict
        ``{s1_id: frozenset(candidate_ids)}`` from Member 2.
    negative_sampling_ratio : float
        Fraction of negative pairs to keep. 1.0 = keep all.
        Positives are never removed.
    seed : int
        RNG seed for negative sampling reproducibility.

    Returns
    -------
    (pairs, stats)
    """
    rng = random.Random(seed)
    all_positives: List[PairRecord] = []
    all_negatives: List[PairRecord] = []
    unique_s1 = set()
    unique_cands = set()

    for s1_id in s1_ids:
        cands = candidate_pairs.get(s1_id, frozenset())
        if not cands:
            continue

        true_matches = ground_truth.get(s1_id, frozenset())
        s1_row = source1.get(s1_id)
        text_a = format_entity_text(s1_row)
        unique_s1.add(s1_id)

        for cid in sorted(cands):  # sorted for determinism
            # Look up candidate in S2 or S3
            cand_row = source2.get(cid) or source3.get(cid)
            text_b = format_entity_text(cand_row)
            label = 1 if cid in true_matches else 0
            unique_cands.add(cid)

            rec = PairRecord(
                s1_id=s1_id,
                candidate_id=cid,
                text_a=text_a,
                text_b=text_b,
                label=label,
            )
            if label == 1:
                all_positives.append(rec)
            else:
                all_negatives.append(rec)

    # Negative sampling
    if negative_sampling_ratio < 1.0 and all_negatives:
        k = max(1, int(len(all_negatives) * negative_sampling_ratio))
        rng.shuffle(all_negatives)
        all_negatives = all_negatives[:k]

    pairs = all_positives + all_negatives
    # Shuffle to mix positives and negatives
    rng2 = random.Random(seed + 1)
    rng2.shuffle(pairs)

    n_pos = len(all_positives)
    n_neg = len(all_negatives)
    total = n_pos + n_neg

    stats = PairStats(
        total_pairs=total,
        positive_pairs=n_pos,
        negative_pairs=n_neg,
        positive_ratio=n_pos / total if total else 0.0,
        unique_s1=len(unique_s1),
        unique_candidates=len(unique_cands),
    )

    return pairs, stats


def build_inference_pairs(
    s1_ids: List[str],
    source1: Dict[str, EntityRow],
    source2: Dict[str, EntityRow],
    source3: Dict[str, EntityRow],
    candidate_pairs: Dict[str, FrozenSet[str]],
) -> List[PairRecord]:
    """Build unlabeled pairs for inference (label=-1)."""
    pairs: List[PairRecord] = []

    for s1_id in s1_ids:
        cands = candidate_pairs.get(s1_id, frozenset())
        if not cands:
            continue

        s1_row = source1.get(s1_id)
        text_a = format_entity_text(s1_row)

        for cid in sorted(cands):
            cand_row = source2.get(cid) or source3.get(cid)
            text_b = format_entity_text(cand_row)

            pairs.append(PairRecord(
                s1_id=s1_id,
                candidate_id=cid,
                text_a=text_a,
                text_b=text_b,
                label=-1,
            ))

    return pairs


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class PairDataset(Dataset):
    """PyTorch Dataset wrapping a list of PairRecords with tokenization.

    Parameters
    ----------
    pairs : list[PairRecord]
        Pre-built pair records.
    tokenizer : PreTrainedTokenizer
        HuggingFace tokenizer (e.g. DeBERTa).
    max_length : int
        Maximum token sequence length.
    """

    def __init__(self, pairs: List[PairRecord], tokenizer, max_length: int = 256):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        rec = self.pairs[idx]
        encoding = self.tokenizer(
            rec.text_a,
            rec.text_b,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        item = {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
        }
        if rec.label >= 0:
            item["labels"] = torch.tensor(rec.label, dtype=torch.long)
        return item
