"""
baseline.py — Cheap, transparent baselines for entity matching.

Supports two strategies:
  Baseline A  — exact normalized business-name matching
  Baseline B  — exact normalized business-name + country agreement

Normalization is deterministic and conservative:
  - lowercase
  - Unicode NFKD normalization
  - whitespace collapse / strip
  - punctuation / separator removal
  - safe handling of missing / NaN values

Builds an inverted index ``normalized_name -> [entity_ids]`` over S2/S3
so that matching is O(|S1_val|) index lookups, not O(N²).
"""

import re
import unicodedata
from collections import defaultdict
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

from src.member3.data_loader import EntityRow

# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

# Characters to strip: common punctuation and separators found in business names.
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_name(raw: Optional[str]) -> str:
    """Return a deterministic, lowercased, whitespace-collapsed form.

    Returns the empty string for None / blank / NaN-like values.
    """
    if not raw or not isinstance(raw, str):
        return ""

    text = raw.strip()
    if not text or text.lower() == "nan":
        return ""

    # Unicode NFKD: decompose compatibility characters.
    text = unicodedata.normalize("NFKD", text)

    # Lowercase.
    text = text.lower()

    # Remove punctuation / separators.
    text = _PUNCT_RE.sub(" ", text)

    # Collapse whitespace.
    text = _MULTI_SPACE_RE.sub(" ", text).strip()

    return text


def normalize_country(raw: Optional[str]) -> str:
    """Normalise a country string (lowercase, stripped).

    Returns empty string for missing values.
    """
    if not raw or not isinstance(raw, str):
        return ""
    text = raw.strip().lower()
    if text == "nan":
        return ""
    return text


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

# Index value: list of (entity_id, country_normalised) tuples so we can
# optionally filter by country without rebuilding the index.
IndexEntry = List[Tuple[str, str]]
NameIndex = Dict[str, IndexEntry]


def build_name_index(entities: Dict[str, EntityRow]) -> NameIndex:
    """Build ``{normalised_name: [(entity_id, normalised_country), ...]}``."""
    index: NameIndex = defaultdict(list)

    for eid, row in entities.items():
        nname = normalize_name(row.get("business_name"))
        if not nname:
            continue
        ncountry = normalize_country(row.get("country"))
        index[nname].append((eid, ncountry))

    return dict(index)


# ---------------------------------------------------------------------------
# Matching strategies
# ---------------------------------------------------------------------------

def match_exact_name(
    s1_id: str,
    s1_row: EntityRow,
    s2_index: NameIndex,
    s3_index: NameIndex,
) -> FrozenSet[str]:
    """Baseline A — return all S2/S3 IDs whose normalised name matches S1."""
    nname = normalize_name(s1_row.get("business_name"))
    if not nname:
        return frozenset()

    matched: Set[str] = set()
    for eid, _country in s2_index.get(nname, []):
        matched.add(eid)
    for eid, _country in s3_index.get(nname, []):
        matched.add(eid)

    return frozenset(matched)


def match_exact_name_country(
    s1_id: str,
    s1_row: EntityRow,
    s2_index: NameIndex,
    s3_index: NameIndex,
) -> FrozenSet[str]:
    """Baseline B — exact normalised name AND matching country."""
    nname = normalize_name(s1_row.get("business_name"))
    if not nname:
        return frozenset()

    s1_country = normalize_country(s1_row.get("country"))

    matched: Set[str] = set()
    for eid, ecountry in s2_index.get(nname, []):
        if s1_country and ecountry and s1_country == ecountry:
            matched.add(eid)
    for eid, ecountry in s3_index.get(nname, []):
        if s1_country and ecountry and s1_country == ecountry:
            matched.add(eid)

    return frozenset(matched)


# Strategy registry (for CLI selection).
STRATEGIES = {
    "exact_name": match_exact_name,
    "exact_name_country": match_exact_name_country,
}


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_batch(
    val_s1_ids: List[str],
    source1: Dict[str, EntityRow],
    s2_index: NameIndex,
    s3_index: NameIndex,
    *,
    strategy: str = "exact_name",
) -> Dict[str, FrozenSet[str]]:
    """Run a baseline strategy over a list of S1 validation IDs.

    Parameters
    ----------
    val_s1_ids : list[str]
        Validation S1 entity IDs.
    source1 : dict
        Full source-1 entity dict.
    s2_index, s3_index : NameIndex
        Pre-built normalised-name indices for source 2 and source 3.
    strategy : str
        One of ``STRATEGIES`` keys.

    Returns
    -------
    dict[str, frozenset[str]]
        ``{s1_id: frozenset(predicted_matched_ids)}``.
    """
    match_fn = STRATEGIES.get(strategy)
    if match_fn is None:
        raise ValueError(
            f"Unknown strategy {strategy!r}. Choose from: {sorted(STRATEGIES)}"
        )

    predictions: Dict[str, FrozenSet[str]] = {}
    for s1_id in val_s1_ids:
        s1_row = source1.get(s1_id)
        if s1_row is None:
            predictions[s1_id] = frozenset()
            continue
        predictions[s1_id] = match_fn(s1_id, s1_row, s2_index, s3_index)

    return predictions
