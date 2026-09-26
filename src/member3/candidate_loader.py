"""
candidate_loader.py — Load and validate Member 2's candidate_pairs.tsv.

This is the interface between Member 2's blocking/candidate-generation
pipeline and Member 3's matching pipeline.

Expected file format (tab-separated)::

    source1_entity_id    candidate_entity_ids
    S1-000001            S2-000123,S3-004521,S2-001892
    S1-000002            S3-000812
    S1-000003

Responsibilities:
  - Parse candidate_pairs.tsv with strict column validation.
  - Reject duplicate S1 rows.
  - Validate that candidate IDs have S2-/S3- prefixes.
  - Normalize duplicate candidate IDs within a row (report count).
  - Return Dict[str, FrozenSet[str]] for downstream consumption.
"""

import csv
from typing import Dict, FrozenSet, List, Tuple

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CANDIDATE_COLUMNS = {"source1_entity_id", "candidate_entity_ids"}
VALID_CANDIDATE_PREFIXES = ("S2-", "S3-")


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def load_candidate_pairs(
    path: str,
) -> Tuple[Dict[str, FrozenSet[str]], int]:
    """Load and validate a candidate-pairs TSV.

    Parameters
    ----------
    path : str
        Path to ``candidate_pairs.tsv``.

    Returns
    -------
    (candidates, duplicate_id_count)
        ``candidates``: ``{s1_id: frozenset(candidate_ids)}``.
        ``duplicate_id_count``: total number of duplicate candidate IDs
        that were encountered and normalized away (0 if clean).

    Raises
    ------
    ValueError
        On missing required columns, duplicate S1 rows, invalid candidate
        ID prefixes, or malformed IDs.
    """
    candidates: Dict[str, FrozenSet[str]] = {}
    duplicates_s1: List[str] = []
    total_dup_ids = 0

    with open(path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter="\t")

        # --- Column validation ---
        if reader.fieldnames is None:
            raise ValueError(f"Empty or header-less TSV: {path}")

        header_set = set(reader.fieldnames)
        missing = CANDIDATE_COLUMNS - header_set
        if missing:
            raise ValueError(
                f"{path}: missing required columns {sorted(missing)}. "
                f"Found: {sorted(header_set)}"
            )

        # --- Row parsing ---
        for line_num, row in enumerate(reader, start=2):
            s1_id = row["source1_entity_id"].strip()
            if not s1_id:
                continue  # skip blank rows

            # Duplicate S1 check.
            if s1_id in candidates:
                duplicates_s1.append(s1_id)

            raw_cands = row.get("candidate_entity_ids", "")
            if raw_cands is None:
                raw_cands = ""
            raw_cands = raw_cands.strip()

            if not raw_cands:
                candidates[s1_id] = frozenset()
                continue

            # Parse comma-separated candidate IDs.
            raw_ids = [cid.strip() for cid in raw_cands.split(",")]

            # Validate each candidate ID.
            valid_ids: List[str] = []
            for cid in raw_ids:
                if not cid:
                    raise ValueError(
                        f"{path} line {line_num}: empty candidate ID in "
                        f"list for {s1_id!r}"
                    )
                if cid.startswith("S1-"):
                    raise ValueError(
                        f"{path} line {line_num}: candidate ID {cid!r} has "
                        f"S1- prefix (self-source error) for {s1_id!r}"
                    )
                if not cid.startswith(VALID_CANDIDATE_PREFIXES):
                    raise ValueError(
                        f"{path} line {line_num}: candidate ID {cid!r} has "
                        f"invalid prefix for {s1_id!r}. "
                        f"Expected S2- or S3-."
                    )
                valid_ids.append(cid)

            # Normalize duplicates within a candidate list.
            id_set = frozenset(valid_ids)
            n_dups = len(valid_ids) - len(id_set)
            total_dup_ids += n_dups

            candidates[s1_id] = id_set

    if duplicates_s1:
        raise ValueError(
            f"{path}: {len(duplicates_s1)} duplicate source1_entity_id(s). "
            f"First 5: {duplicates_s1[:5]}"
        )

    return candidates, total_dup_ids
