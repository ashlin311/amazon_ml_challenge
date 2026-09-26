"""
data_loader.py — Load and merge entity data from source TSV files.

Responsibilities:
  - Read train/test source TSV files (source1, source2, source3).
  - Read ground-truth labels for training.
  - Read candidate_pairs.tsv produced by Member 2's blocking pipeline.
  - Provide unified entity lookup by entity ID across all sources.
"""

import csv
import hashlib
import os
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_SOURCE_COLUMNS = {"entity_id", "business_name", "business_address", "country"}
GT_COLUMNS = {"source1_entity_id", "matched_entity_ids"}


# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# Each entity row is stored as a plain dict to avoid duplicating a DataFrame.
EntityRow = Dict[str, str]

# Ground truth: S1 ID -> frozenset of matched S2/S3 IDs (empty for singletons).
GroundTruth = Dict[str, FrozenSet[str]]


# ---------------------------------------------------------------------------
# Source loading
# ---------------------------------------------------------------------------

def load_source_tsv(
    path: str,
    *,
    required_columns: Optional[Set[str]] = None,
) -> Dict[str, EntityRow]:
    """Load a source TSV into a dict keyed by ``entity_id``.

    Parameters
    ----------
    path : str
        Absolute or relative path to a ``*.tsv`` file with a header row.
    required_columns : set[str] | None
        If provided, raise ``ValueError`` when any column is missing.

    Returns
    -------
    dict[str, EntityRow]
        Mapping from ``entity_id`` to the remaining column values.
    """
    if required_columns is None:
        required_columns = REQUIRED_SOURCE_COLUMNS

    entities: Dict[str, EntityRow] = {}

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")

        if reader.fieldnames is None:
            raise ValueError(f"Empty or header-less TSV: {path}")

        header_set = set(reader.fieldnames)
        missing = required_columns - header_set
        if missing:
            raise ValueError(
                f"{path}: missing required columns {sorted(missing)}. "
                f"Found: {sorted(header_set)}"
            )

        for row in reader:
            eid = row["entity_id"]
            if eid in entities:
                raise ValueError(f"{path}: duplicate entity_id {eid!r}")
            entities[eid] = row

    return entities


# ---------------------------------------------------------------------------
# Ground-truth loading
# ---------------------------------------------------------------------------

def load_ground_truth(path: str) -> GroundTruth:
    """Parse the official ground-truth TSV.

    Expected format (tab-separated)::

        source1_entity_id   matched_entity_ids
        S1-00001            S2-00047,S3-00812
        S1-00002            S3-00004
        S1-00003

    Parameters
    ----------
    path : str
        Path to ``train_ground_truth.tsv``.

    Returns
    -------
    GroundTruth
        ``{s1_id: frozenset(matched_ids)}``; singletons map to
        ``frozenset()``.

    Raises
    ------
    ValueError
        On missing header columns or duplicate S1 rows.
    """
    gt: GroundTruth = {}
    duplicates: List[str] = []

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")

        if reader.fieldnames is None:
            raise ValueError(f"Empty or header-less TSV: {path}")

        header_set = set(reader.fieldnames)
        missing = GT_COLUMNS - header_set
        if missing:
            raise ValueError(
                f"{path}: missing required columns {sorted(missing)}. "
                f"Found: {sorted(header_set)}"
            )

        for row in reader:
            s1_id = row["source1_entity_id"]
            raw_matches = row.get("matched_entity_ids", "").strip()
            matched = (
                frozenset(raw_matches.split(",")) if raw_matches else frozenset()
            )

            if s1_id in gt:
                duplicates.append(s1_id)
            gt[s1_id] = matched

    if duplicates:
        raise ValueError(
            f"{path}: {len(duplicates)} duplicate source1_entity_id(s). "
            f"First 5: {duplicates[:5]}"
        )

    return gt


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def split_s1_ids(
    s1_ids: List[str],
    *,
    validation_fraction: float = 0.01,
    seed: int = 42,
) -> Tuple[List[str], List[str]]:
    """Deterministic, hash-based split of S1 entity IDs.

    Uses MD5 of ``f"{seed}:{s1_id}"`` so the split is stable regardless
    of input order and does not require shuffling the full list.

    Parameters
    ----------
    s1_ids : list[str]
        All source-1 entity IDs.
    validation_fraction : float
        Fraction of IDs to assign to the validation set (0 < frac < 1).
    seed : int
        Integer mixed into the hash to allow reproducible but adjustable
        splits.

    Returns
    -------
    (train_ids, val_ids)
        Two disjoint sorted lists.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")

    # 2**128 normalization constant for MD5 digest → [0, 1).
    _MAX = 2 ** 128
    threshold = validation_fraction

    train_ids: List[str] = []
    val_ids: List[str] = []

    for s1 in s1_ids:
        h = int(hashlib.md5(f"{seed}:{s1}".encode()).hexdigest(), 16) / _MAX
        if h < threshold:
            val_ids.append(s1)
        else:
            train_ids.append(s1)

    train_ids.sort()
    val_ids.sort()
    return train_ids, val_ids


# ---------------------------------------------------------------------------
# Convenience: load everything needed for validation
# ---------------------------------------------------------------------------

def load_train_data(
    train_dir: str,
) -> Tuple[Dict[str, EntityRow], Dict[str, EntityRow], Dict[str, EntityRow], GroundTruth]:
    """Load all three source files and the ground truth from *train_dir*.

    Parameters
    ----------
    train_dir : str
        Directory containing ``train_source1.tsv``, ``train_source2.tsv``,
        ``train_source3.tsv``, and ``train_ground_truth.tsv``.

    Returns
    -------
    (source1, source2, source3, ground_truth)
    """
    source1 = load_source_tsv(os.path.join(train_dir, "train_source1.tsv"))
    source2 = load_source_tsv(os.path.join(train_dir, "train_source2.tsv"))
    source3 = load_source_tsv(os.path.join(train_dir, "train_source3.tsv"))
    ground_truth = load_ground_truth(
        os.path.join(train_dir, "train_ground_truth.tsv")
    )
    return source1, source2, source3, ground_truth
