"""
test_metrics.py — Unit tests for src.member3.metrics (entity-level macro F0.5).
"""

import pytest

from src.member3.metrics import entity_f05, evaluate_predictions


# ---- per-entity tests ----

class TestEntityF05:
    """Test the per-entity F0.5 calculation."""

    def test_exact_match(self):
        """Predicted set exactly equals truth → F0.5 = 1.0."""
        s = entity_f05({"A", "B"}, {"A", "B"})
        assert s.precision == 1.0
        assert s.recall == 1.0
        assert s.f05 == 1.0

    def test_correct_singleton(self):
        """True singleton, predicted singleton → F0.5 = 1.0."""
        s = entity_f05(set(), set())
        assert s.precision == 1.0
        assert s.recall == 1.0
        assert s.f05 == 1.0

    def test_false_positive_singleton(self):
        """True singleton but predictions made → F0.5 = 0.0."""
        s = entity_f05(set(), {"X"})
        assert s.precision == 0.0
        assert s.f05 == 0.0

    def test_missed_match(self):
        """True matches exist, nothing predicted → F0.5 = 0.0."""
        s = entity_f05({"A", "B"}, set())
        assert s.recall == 0.0
        assert s.f05 == 0.0

    def test_partial_match(self):
        """Predict one of two true matches."""
        s = entity_f05({"A", "B"}, {"A"})
        assert s.precision == 1.0
        assert s.recall == 0.5
        # F0.5 = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5) = 0.625 / 0.75
        expected = 1.25 * 1.0 * 0.5 / (0.25 * 1.0 + 0.5)
        assert abs(s.f05 - expected) < 1e-9

    def test_extra_false_positive(self):
        """Predict the true match plus one extra → precision < 1."""
        s = entity_f05({"A"}, {"A", "X"})
        assert s.precision == 0.5
        assert s.recall == 1.0
        expected = 1.25 * 0.5 * 1.0 / (0.25 * 0.5 + 1.0)
        assert abs(s.f05 - expected) < 1e-9

    def test_frozenset_inputs(self):
        """Accepts frozenset as well as set."""
        s = entity_f05(frozenset({"A"}), frozenset({"A"}))
        assert s.f05 == 1.0


# ---- aggregate tests ----

class TestEvaluatePredictions:
    """Test the macro-averaged evaluate_predictions."""

    def test_empty_input(self):
        """No entities → zeroes everywhere."""
        r = evaluate_predictions({}, {})
        assert r.entity_count == 0
        assert r.macro_f05 == 0.0

    def test_single_perfect(self):
        gt = {"S1-001": frozenset({"S2-001"})}
        preds = {"S1-001": frozenset({"S2-001"})}
        r = evaluate_predictions(gt, preds)
        assert r.entity_count == 1
        assert r.macro_f05 == 1.0

    def test_single_singleton(self):
        gt = {"S1-001": frozenset()}
        preds = {"S1-001": frozenset()}
        r = evaluate_predictions(gt, preds)
        assert r.macro_f05 == 1.0

    def test_macro_average(self):
        """Average of one perfect (1.0) and one complete miss (0.0) → 0.5."""
        gt = {
            "S1-001": frozenset({"S2-001"}),
            "S1-002": frozenset({"S2-002"}),
        }
        preds = {
            "S1-001": frozenset({"S2-001"}),
            # S1-002 missing from predictions → empty set → 0.0
        }
        r = evaluate_predictions(gt, preds)
        assert r.entity_count == 2
        assert abs(r.macro_f05 - 0.5) < 1e-9

    def test_missing_prediction_key_treated_as_empty(self):
        """S1 in ground truth but not in predictions → predicted singleton."""
        gt = {"S1-001": frozenset({"S2-001"})}
        preds = {}
        r = evaluate_predictions(gt, preds)
        assert r.macro_f05 == 0.0

    def test_precision_heavy_f05(self):
        """F0.5 should penalise false positives more than false negatives."""
        # Extra FP: predict {A, X} for true {A}
        gt_fp = {"S1-001": frozenset({"A"})}
        pred_fp = {"S1-001": frozenset({"A", "X"})}
        r_fp = evaluate_predictions(gt_fp, pred_fp)

        # Missed recall: predict {A} for true {A, B}
        gt_fn = {"S1-001": frozenset({"A", "B"})}
        pred_fn = {"S1-001": frozenset({"A"})}
        r_fn = evaluate_predictions(gt_fn, pred_fn)

        # F0.5 should be higher when precision is higher (partial recall)
        assert r_fn.macro_f05 > r_fp.macro_f05
