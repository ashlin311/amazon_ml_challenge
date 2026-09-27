"""
test_train.py — Unit and regression tests for train.py.
"""

from collections import namedtuple
from typing import Dict, FrozenSet, List

import pytest
import torch
import torch.nn as nn

from src.member3.pair_dataset import PairRecord
from src.member3.train import evaluate_validation


# ---------------------------------------------------------------------------
# Mock / Stub Model & Dataloader for Fast Testing (no weights download)
# ---------------------------------------------------------------------------

ModelOutput = namedtuple("ModelOutput", ["logits", "loss"])


class StubModel(nn.Module):
    """Stub model that returns predetermined logits for testing."""

    def __init__(self, logits_sequence: List[torch.Tensor], loss_val: float = 0.25):
        super().__init__()
        self.logits_sequence = logits_sequence
        self.idx = 0
        self.loss_val = loss_val

    def forward(self, input_ids=None, attention_mask=None, labels=None):
        logits = self.logits_sequence[self.idx]
        self.idx += 1
        loss = torch.tensor(self.loss_val)
        return ModelOutput(logits=logits, loss=loss)


def create_mock_batch(labels: List[int]):
    n_samples = len(labels)
    return {
        "input_ids": torch.zeros((n_samples, 8), dtype=torch.long),
        "attention_mask": torch.ones((n_samples, 8), dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# Regression Tests for S1 with Zero Candidate Pairs
# ---------------------------------------------------------------------------

class TestEvaluateValidationZeroCandidates:
    """Regression tests ensuring S1 entities with zero candidate pairs

    are included in entity-level evaluation.
    """

    def test_val_s1_with_zero_candidate_pairs_included_in_metrics(self):
        """Regression test: S1 entities in val_s1 with zero candidate pairs

        must be included in entity_count and evaluated as empty predictions.

        Scenario:
          - S1-MATCH: has candidate S2-001 (prob=0.9 -> predicted match). GT={S2-001}.
                      Precision=1.0, Recall=1.0, F0.5=1.0.
          - S1-MISSED: has 0 candidate pairs. GT={S2-002} (true match missed).
                       Precision=0.0, Recall=0.0, F0.5=0.0.
          - S1-SINGLETON: has 0 candidate pairs. GT=empty (singleton entity).
                          Precision=1.0, Recall=1.0, F0.5=1.0.

        Total val_s1: 3 entities.
        Macro F0.5 = (1.0 + 0.0 + 1.0) / 3 = 2/3 ≈ 0.6667.
        (Under the old bug where val_s1 was derived from pairs, only S1-MATCH
         would be counted, yielding an inflated Macro F0.5 of 1.0).
        """
        device = torch.device("cpu")

        val_s1 = ["S1-MATCH", "S1-MISSED", "S1-SINGLETON"]
        ground_truth: Dict[str, FrozenSet[str]] = {
            "S1-MATCH": frozenset({"S2-001"}),
            "S1-MISSED": frozenset({"S2-002"}),
            "S1-SINGLETON": frozenset(),
        }

        # Only S1-MATCH has candidate pairs:
        pairs = [
            PairRecord(
                s1_id="S1-MATCH",
                candidate_id="S2-001",
                text_a="Name: A | Address: 1 | Country: US",
                text_b="Name: A | Address: 1 | Country: US",
                label=1,
            )
        ]

        # Logits that yield prob ~0.9 for class 1 (match)
        # softmax([-1.0, 1.197]) -> class 1 prob ~0.9
        logits = torch.tensor([[-1.0, 1.197]])
        stub_model = StubModel(logits_sequence=[logits], loss_val=0.15)
        dataloader = [create_mock_batch([1])]

        result = evaluate_validation(
            model=stub_model,
            dataloader=dataloader,
            pairs=pairs,
            ground_truth=ground_truth,
            val_s1=val_s1,
            device=device,
            threshold=0.5,
        )

        # Entity count must match val_s1 length (3), not pair count (1)
        assert result["entity_count"] == 3

        # Macro precision: all three have precision=1.0 (under official metric,
        # true matches + empty prediction has precision=1.0, recall=0.0, f05=0.0).
        # Macro recall and F0.5 are (1.0 + 0.0 + 1.0) / 3 = 2/3.
        assert result["entity_macro_precision"] == pytest.approx(1.0, abs=1e-4)
        assert result["entity_macro_recall"] == pytest.approx(2.0 / 3.0, abs=1e-4)
        assert result["entity_macro_f05"] == pytest.approx(2.0 / 3.0, abs=1e-4)

        # Pair metrics should only reflect the 1 evaluated pair
        assert result["pair_tp"] == 1
        assert result["pair_fp"] == 0
        assert result["pair_fn"] == 0
        assert result["pair_precision"] == 1.0
        assert result["pair_recall"] == 1.0
        assert result["pair_f05"] == 1.0

    def test_val_s1_completely_empty_pairs(self):
        """When no candidate pairs exist at all in validation,

        entity metrics should still evaluate across all val_s1 entities.
        """
        device = torch.device("cpu")

        val_s1 = ["S1-001", "S1-002"]
        ground_truth: Dict[str, FrozenSet[str]] = {
            "S1-001": frozenset({"S2-001"}),  # Missed (F0.5 = 0)
            "S1-002": frozenset(),             # Singleton (F0.5 = 1)
        }

        stub_model = StubModel(logits_sequence=[])
        dataloader: List[dict] = []
        pairs: List[PairRecord] = []

        result = evaluate_validation(
            model=stub_model,
            dataloader=dataloader,
            pairs=pairs,
            ground_truth=ground_truth,
            val_s1=val_s1,
            device=device,
            threshold=0.5,
        )

        assert result["entity_count"] == 2
        assert result["entity_macro_f05"] == pytest.approx(0.5, abs=1e-4)
        assert result["val_loss"] == 0.0
        assert result["pair_tp"] == 0

    def test_contrast_with_deriving_s1_from_pairs(self):
        """Demonstrate that passing explicit val_s1 prevents metric inflation.

        If val_s1 were derived only from pair records:
          - S1-MISSED (which had 0 candidates) would be omitted, resulting in
            entity_count = 1 and Macro F0.5 = 1.0.
        With the fix:
          - S1-MISSED is included with an empty prediction, yielding entity_count = 2
            and Macro F0.5 = 0.5.
        """
        device = torch.device("cpu")

        val_s1 = ["S1-MATCH", "S1-MISSED"]
        ground_truth: Dict[str, FrozenSet[str]] = {
            "S1-MATCH": frozenset({"S2-001"}),
            "S1-MISSED": frozenset({"S2-002"}),
        }
        pairs = [
            PairRecord(
                s1_id="S1-MATCH",
                candidate_id="S2-001",
                text_a="a",
                text_b="b",
                label=1,
            )
        ]

        logits = torch.tensor([[-1.0, 1.197]])
        stub_model = StubModel(logits_sequence=[logits], loss_val=0.1)
        dataloader = [create_mock_batch([1])]

        result = evaluate_validation(
            model=stub_model,
            dataloader=dataloader,
            pairs=pairs,
            ground_truth=ground_truth,
            val_s1=val_s1,
            device=device,
            threshold=0.5,
        )

        # Must be 2 entities, macro F0.5 = 0.5 (not inflated to 1.0)
        assert result["entity_count"] == 2
        assert result["entity_macro_f05"] == pytest.approx(0.5, abs=1e-4)
