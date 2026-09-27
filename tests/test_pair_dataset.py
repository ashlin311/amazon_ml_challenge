"""
test_pair_dataset.py — Unit tests for pair_dataset.py.

Uses tiny synthetic data only.
"""

import pytest

from src.member3.pair_dataset import (
    PairRecord,
    PairStats,
    build_labeled_pairs,
    build_inference_pairs,
    format_entity_text,
)
from src.member3.data_loader import split_s1_ids


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SOURCE1 = {
    "S1-A": {"entity_id": "S1-A", "business_name": "Acme Corp", "business_address": "123 Main St", "country": "US"},
    "S1-B": {"entity_id": "S1-B", "business_name": "Beta LLC", "business_address": "456 Oak Ave", "country": "US"},
    "S1-C": {"entity_id": "S1-C", "business_name": "Gamma Inc", "business_address": "789 Pine Rd", "country": "India"},
}

SOURCE2 = {
    "S2-1": {"entity_id": "S2-1", "business_name": "Acme Corporation", "business_address": "123 Main Street", "country": "US"},
    "S2-2": {"entity_id": "S2-2", "business_name": "Delta Co", "business_address": "999 Elm St", "country": "US"},
}

SOURCE3 = {
    "S3-1": {"entity_id": "S3-1", "business_name": "ACME CORP", "business_address": "123 Main", "country": "US"},
    "S3-2": {"entity_id": "S3-2", "business_name": "Epsilon Ltd", "business_address": "555 Birch", "country": "UK"},
}

GT = {
    "S1-A": frozenset({"S2-1", "S3-1"}),  # 2 true matches
    "S1-B": frozenset({"S2-2"}),           # 1 true match
    "S1-C": frozenset(),                    # singleton
}

CANDS = {
    "S1-A": frozenset({"S2-1", "S3-1", "S3-2"}),   # 2 positives, 1 negative
    "S1-B": frozenset({"S2-2", "S2-1"}),              # 1 positive, 1 negative
    "S1-C": frozenset(),                               # empty candidate set
}


# ---------------------------------------------------------------------------
# format_entity_text
# ---------------------------------------------------------------------------

class TestFormatEntityText:
    def test_normal_row(self):
        txt = format_entity_text(SOURCE1["S1-A"])
        assert "Name: Acme Corp" in txt
        assert "Address: 123 Main St" in txt
        assert "Country: US" in txt

    def test_none_row(self):
        txt = format_entity_text(None)
        assert "Name:" in txt
        assert "Address:" in txt
        assert "Country:" in txt


# ---------------------------------------------------------------------------
# Positive/negative label creation
# ---------------------------------------------------------------------------

class TestBuildLabeledPairs:
    def test_positive_label(self):
        """Candidates that are in GT get label=1."""
        pairs, stats = build_labeled_pairs(
            ["S1-A"], SOURCE1, SOURCE2, SOURCE3, GT, CANDS
        )
        pos_ids = {p.candidate_id for p in pairs if p.label == 1}
        assert "S2-1" in pos_ids
        assert "S3-1" in pos_ids

    def test_negative_label(self):
        """Candidates NOT in GT get label=0."""
        pairs, _ = build_labeled_pairs(
            ["S1-A"], SOURCE1, SOURCE2, SOURCE3, GT, CANDS
        )
        neg_ids = {p.candidate_id for p in pairs if p.label == 0}
        assert "S3-2" in neg_ids

    def test_missing_gt_match(self):
        """S1 with no ground-truth entry → all candidates are negative."""
        gt_missing = {"S1-A": frozenset()}  # no matches
        pairs, _ = build_labeled_pairs(
            ["S1-A"], SOURCE1, SOURCE2, SOURCE3, gt_missing, CANDS
        )
        assert all(p.label == 0 for p in pairs)

    def test_multiple_true_matches(self):
        """S1-A has 2 true matches among 3 candidates."""
        pairs, stats = build_labeled_pairs(
            ["S1-A"], SOURCE1, SOURCE2, SOURCE3, GT, CANDS
        )
        assert stats.positive_pairs == 2
        assert stats.negative_pairs == 1
        assert stats.total_pairs == 3

    def test_s2_candidate(self):
        """S2 candidate IDs are resolved from source2."""
        pairs, _ = build_labeled_pairs(
            ["S1-B"], SOURCE1, SOURCE2, SOURCE3, GT, CANDS
        )
        s2_pair = [p for p in pairs if p.candidate_id == "S2-2"][0]
        assert "Delta Co" in s2_pair.text_b

    def test_s3_candidate(self):
        """S3 candidate IDs are resolved from source3."""
        pairs, _ = build_labeled_pairs(
            ["S1-A"], SOURCE1, SOURCE2, SOURCE3, GT, CANDS
        )
        s3_pair = [p for p in pairs if p.candidate_id == "S3-1"][0]
        assert "ACME CORP" in s3_pair.text_b

    def test_empty_candidate_set(self):
        """S1-C has zero candidates → no pairs produced."""
        pairs, stats = build_labeled_pairs(
            ["S1-C"], SOURCE1, SOURCE2, SOURCE3, GT, CANDS
        )
        assert stats.total_pairs == 0
        assert len(pairs) == 0

    def test_stats_reporting(self):
        """Stats correctly report unique counts."""
        pairs, stats = build_labeled_pairs(
            ["S1-A", "S1-B"], SOURCE1, SOURCE2, SOURCE3, GT, CANDS
        )
        assert stats.unique_s1 == 2
        assert stats.unique_candidates >= 3
        assert stats.positive_ratio > 0.0

    def test_negative_sampling_ratio(self):
        """Negative sampling reduces negatives but keeps all positives."""
        pairs_full, stats_full = build_labeled_pairs(
            ["S1-A", "S1-B"], SOURCE1, SOURCE2, SOURCE3, GT, CANDS,
            negative_sampling_ratio=1.0,
        )
        pairs_half, stats_half = build_labeled_pairs(
            ["S1-A", "S1-B"], SOURCE1, SOURCE2, SOURCE3, GT, CANDS,
            negative_sampling_ratio=0.5,
        )
        assert stats_half.positive_pairs == stats_full.positive_pairs
        assert stats_half.negative_pairs <= stats_full.negative_pairs


# ---------------------------------------------------------------------------
# S1-level train/validation split
# ---------------------------------------------------------------------------

class TestS1LevelSplit:
    def test_no_s1_leakage(self):
        """Train and val S1 IDs must be disjoint."""
        all_s1 = list(SOURCE1.keys())
        train_s1, val_s1 = split_s1_ids(
            all_s1, validation_fraction=0.5, seed=42
        )
        assert set(train_s1) & set(val_s1) == set()

    def test_all_s1_covered(self):
        """Union of train + val = all S1 IDs."""
        all_s1 = list(SOURCE1.keys())
        train_s1, val_s1 = split_s1_ids(
            all_s1, validation_fraction=0.5, seed=42
        )
        assert set(train_s1) | set(val_s1) == set(all_s1)

    def test_pairs_respect_split(self):
        """Pairs built from train_s1 should not contain val S1 IDs."""
        all_s1 = list(SOURCE1.keys())
        train_s1, val_s1 = split_s1_ids(
            all_s1, validation_fraction=0.5, seed=42
        )
        train_pairs, _ = build_labeled_pairs(
            train_s1, SOURCE1, SOURCE2, SOURCE3, GT, CANDS
        )
        val_pairs, _ = build_labeled_pairs(
            val_s1, SOURCE1, SOURCE2, SOURCE3, GT, CANDS
        )
        train_pair_s1 = {p.s1_id for p in train_pairs}
        val_pair_s1 = {p.s1_id for p in val_pairs}
        assert train_pair_s1 & val_pair_s1 == set()


# ---------------------------------------------------------------------------
# Inference pairs
# ---------------------------------------------------------------------------

class TestBuildInferencePairs:
    def test_inference_labels(self):
        """Inference pairs should have label=-1."""
        pairs = build_inference_pairs(
            ["S1-A"], SOURCE1, SOURCE2, SOURCE3, CANDS
        )
        assert all(p.label == -1 for p in pairs)
        assert len(pairs) == 3  # S1-A has 3 candidates
