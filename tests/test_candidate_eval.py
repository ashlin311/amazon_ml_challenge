"""
test_candidate_eval.py — Unit tests for candidate_loader and candidate_eval.

Uses tiny synthetic data only; no full competition dataset.
"""

import os
import tempfile
import textwrap

import pytest

from src.member3.candidate_loader import load_candidate_pairs
from src.member3.candidate_eval import (
    compute_coverage,
    compute_recall,
    compute_singleton_stats,
    compute_volume,
    compute_recall_by_match_count,
    evaluate_candidates,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_tsv(tmp_dir: str, name: str, content: str) -> str:
    """Write a TSV string to a temp file and return its path."""
    path = os.path.join(tmp_dir, name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(textwrap.dedent(content))
    return path


# ===========================================================================
# candidate_loader tests
# ===========================================================================

class TestLoadCandidatePairs:
    """Tests for load_candidate_pairs."""

    def test_valid_file(self, tmp_path):
        path = _write_tsv(str(tmp_path), "cands.tsv",
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-001\tS2-010,S3-020\n"
            "S1-002\tS3-030\n"
            "S1-003\t\n"
        )
        cands, dups = load_candidate_pairs(path)
        assert cands["S1-001"] == frozenset({"S2-010", "S3-020"})
        assert cands["S1-002"] == frozenset({"S3-030"})
        assert cands["S1-003"] == frozenset()
        assert dups == 0

    def test_missing_required_column(self, tmp_path):
        path = _write_tsv(str(tmp_path), "bad.tsv",
            "source1_entity_id\twrong_column\n"
            "S1-001\tS2-010\n"
        )
        with pytest.raises(ValueError, match="missing required columns"):
            load_candidate_pairs(path)

    def test_duplicate_s1_row(self, tmp_path):
        path = _write_tsv(str(tmp_path), "dup.tsv",
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-001\tS2-010\n"
            "S1-001\tS3-020\n"
        )
        with pytest.raises(ValueError, match="duplicate"):
            load_candidate_pairs(path)

    def test_empty_candidate_list(self, tmp_path):
        path = _write_tsv(str(tmp_path), "empty.tsv",
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-001\t\n"
        )
        cands, _ = load_candidate_pairs(path)
        assert cands["S1-001"] == frozenset()

    def test_s2_s3_ids_accepted(self, tmp_path):
        path = _write_tsv(str(tmp_path), "ok.tsv",
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-001\tS2-100,S3-200\n"
        )
        cands, _ = load_candidate_pairs(path)
        assert "S2-100" in cands["S1-001"]
        assert "S3-200" in cands["S1-001"]

    def test_s1_candidate_id_rejected(self, tmp_path):
        path = _write_tsv(str(tmp_path), "s1bad.tsv",
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-001\tS1-999\n"
        )
        with pytest.raises(ValueError, match="S1- prefix"):
            load_candidate_pairs(path)

    def test_malformed_candidate_id_rejected(self, tmp_path):
        path = _write_tsv(str(tmp_path), "malformed.tsv",
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-001\tXYZ-123\n"
        )
        with pytest.raises(ValueError, match="invalid prefix"):
            load_candidate_pairs(path)

    def test_duplicate_candidate_id_handling(self, tmp_path):
        path = _write_tsv(str(tmp_path), "dupcand.tsv",
            "source1_entity_id\tcandidate_entity_ids\n"
            "S1-001\tS2-010,S2-010,S3-020\n"
        )
        cands, dups = load_candidate_pairs(path)
        assert cands["S1-001"] == frozenset({"S2-010", "S3-020"})
        assert dups == 1  # one duplicate normalised away


# ===========================================================================
# candidate_eval tests
# ===========================================================================

# Standard test fixture:
#   GT:         S1-A -> {S2-1, S3-1}    (2 matches)
#               S1-B -> {S2-2}          (1 match)
#               S1-C -> {}              (singleton)
#
#   Candidates: S1-A -> {S2-1, S3-9}    (1 of 2 covered)
#               S1-B -> {}              (0 of 1 covered)
#               S1-C -> {S2-8}          (singleton with spurious candidate)

GT_FIXTURE = {
    "S1-A": frozenset({"S2-1", "S3-1"}),
    "S1-B": frozenset({"S2-2"}),
    "S1-C": frozenset(),
}

CAND_FIXTURE = {
    "S1-A": frozenset({"S2-1", "S3-9"}),
    "S1-B": frozenset(),
    "S1-C": frozenset({"S2-8"}),
}


class TestCandidateRecall:
    """Test candidate-generation recall computation."""

    def test_fixture_recall(self):
        """Spec example: true=3, covered=1, recall=1/3."""
        r = compute_recall(GT_FIXTURE, CAND_FIXTURE)
        assert r.total_true_matches == 3
        assert r.covered_true_matches == 1
        assert abs(r.candidate_recall - 1 / 3) < 1e-9

    def test_perfect_recall(self):
        gt = {"S1-X": frozenset({"S2-1", "S3-1"})}
        cands = {"S1-X": frozenset({"S2-1", "S3-1", "S2-99"})}
        r = compute_recall(gt, cands)
        assert r.candidate_recall == 1.0
        assert r.perfect_recall_s1_count == 1

    def test_zero_recall(self):
        gt = {"S1-X": frozenset({"S2-1"})}
        cands = {"S1-X": frozenset({"S3-99"})}
        r = compute_recall(gt, cands)
        assert r.candidate_recall == 0.0
        assert r.perfect_recall_s1_count == 0

    def test_s1_missing_from_candidate_file(self):
        gt = {"S1-X": frozenset({"S2-1"})}
        cands = {}  # S1-X not present
        r = compute_recall(gt, cands)
        assert r.covered_true_matches == 0
        assert r.candidate_recall == 0.0

    def test_singleton_behavior(self):
        gt = {"S1-X": frozenset()}
        cands = {"S1-X": frozenset({"S2-1"})}
        s = compute_singleton_stats(gt, cands)
        assert s.true_singleton_count == 1
        assert s.singleton_with_candidates == 1
        assert s.singleton_zero_candidates == 0

    def test_singleton_zero_candidates(self):
        gt = {"S1-X": frozenset()}
        cands = {"S1-X": frozenset()}
        s = compute_singleton_stats(gt, cands)
        assert s.singleton_zero_candidates == 1
        assert s.singleton_with_candidates == 0

    def test_candidate_with_empty_set(self):
        """S1 in candidates with empty set → 0 candidates."""
        gt = {"S1-X": frozenset({"S2-1"})}
        cands = {"S1-X": frozenset()}
        r = compute_recall(gt, cands)
        assert r.covered_true_matches == 0

    def test_multiple_candidates_for_one_s1(self):
        gt = {"S1-X": frozenset({"S2-1", "S2-2", "S3-1"})}
        cands = {"S1-X": frozenset({"S2-1", "S2-2", "S3-1", "S3-99"})}
        r = compute_recall(gt, cands)
        assert r.candidate_recall == 1.0
        assert r.perfect_recall_s1_count == 1


class TestCoverage:
    """Test S1 coverage computation."""

    def test_full_coverage(self):
        c = compute_coverage(GT_FIXTURE, CAND_FIXTURE)
        assert c.gt_s1_count == 3
        assert c.covered_s1_count == 3
        assert c.missing_s1_count == 0
        assert c.coverage_fraction == 1.0

    def test_partial_coverage(self):
        gt = {"S1-A": frozenset(), "S1-B": frozenset()}
        cands = {"S1-A": frozenset()}
        c = compute_coverage(gt, cands)
        assert c.covered_s1_count == 1
        assert c.missing_s1_count == 1
        assert abs(c.coverage_fraction - 0.5) < 1e-9


class TestVolume:
    """Test candidate-volume statistics."""

    def test_basic_volume(self):
        v = compute_volume(GT_FIXTURE, CAND_FIXTURE)
        # S1-A: 2 candidates, S1-B: 0, S1-C: 1 → total=3
        assert v.total_candidates == 3
        assert v.min_all == 0
        assert v.max_all == 2
        assert v.nonempty_count == 2

    def test_empty_candidates(self):
        gt = {"S1-A": frozenset()}
        cands = {"S1-A": frozenset()}
        v = compute_volume(gt, cands)
        assert v.total_candidates == 0
        assert v.nonempty_count == 0


class TestRecallByMatchCount:
    """Test recall breakdown by number of true matches."""

    def test_fixture_breakdown(self):
        groups = compute_recall_by_match_count(GT_FIXTURE, CAND_FIXTURE)
        # match_count=0 (S1-C singleton), =1 (S1-B), =2 (S1-A)
        mc_map = {g.match_count: g for g in groups}

        assert 0 in mc_map
        assert mc_map[0].s1_count == 1
        assert mc_map[0].recall == 1.0  # no true matches → recall=1.0 by convention

        assert 1 in mc_map
        assert mc_map[1].s1_count == 1
        assert mc_map[1].covered_true == 0
        assert mc_map[1].recall == 0.0

        assert 2 in mc_map
        assert mc_map[2].s1_count == 1
        assert mc_map[2].covered_true == 1
        assert abs(mc_map[2].recall - 0.5) < 1e-9


class TestEvaluateCandidates:
    """Test the full report."""

    def test_full_report_returns_all_fields(self):
        report = evaluate_candidates(GT_FIXTURE, CAND_FIXTURE, 0)
        assert report.coverage.gt_s1_count == 3
        assert report.recall.total_true_matches == 3
        assert report.singletons.true_singleton_count == 1
        assert len(report.recall_by_match_count) > 0
