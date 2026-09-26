"""
metrics.py — F0.5 metric calculation for entity matching evaluation.

Responsibilities:
  - Compute precision, recall, and F0.5 score at the entity-set level.
  - Support per-entity and aggregate evaluation.
  - Compare predicted matched_entity_ids against ground-truth.

Official formula:
    F0.5 = (1.25 * P * R) / (0.25 * P + R)

Singleton handling:
    - true singleton + predicted singleton → F0.5 = 1.0
    - true singleton + false prediction   → F0.5 = 0.0
    - true match    + predicted singleton  → F0.5 = 0.0
"""

from typing import Dict, FrozenSet, List, NamedTuple, Set, Union

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class EntityScore(NamedTuple):
    """Per-entity evaluation result."""
    entity_id: str
    precision: float
    recall: float
    f05: float


class EvalResult(NamedTuple):
    """Aggregate evaluation result."""
    macro_precision: float
    macro_recall: float
    macro_f05: float
    entity_count: int
    per_entity: List[EntityScore]


# ---------------------------------------------------------------------------
# Core metric
# ---------------------------------------------------------------------------

BETA = 0.5
BETA_SQ = BETA ** 2  # 0.25


def _f_beta(precision: float, recall: float, *, beta_sq: float = BETA_SQ) -> float:
    """Compute F_β given pre-calculated precision and recall."""
    if precision + recall == 0.0:
        return 0.0
    return ((1 + beta_sq) * precision * recall) / (beta_sq * precision + recall)


def entity_f05(
    true_ids: Union[Set[str], FrozenSet[str]],
    pred_ids: Union[Set[str], FrozenSet[str]],
) -> EntityScore:
    """Compute precision, recall, F0.5 for a single S1 entity.

    Parameters
    ----------
    true_ids : set[str]
        Ground-truth matched IDs (empty set for a true singleton).
    pred_ids : set[str]
        Predicted matched IDs (empty set if predicted singleton).

    Returns
    -------
    EntityScore
        Named tuple with precision, recall, f05 (entity_id left blank).
    """
    true_set = set(true_ids)
    pred_set = set(pred_ids)

    # Both singleton → perfect score.
    if not true_set and not pred_set:
        return EntityScore(entity_id="", precision=1.0, recall=1.0, f05=1.0)

    # True singleton, but predictions made → all false positives.
    if not true_set and pred_set:
        return EntityScore(entity_id="", precision=0.0, recall=1.0, f05=0.0)

    # True matches exist, but nothing predicted → complete miss.
    if true_set and not pred_set:
        return EntityScore(entity_id="", precision=1.0, recall=0.0, f05=0.0)

    # General case: both sets non-empty.
    tp = len(true_set & pred_set)
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    f05 = _f_beta(precision, recall)

    return EntityScore(entity_id="", precision=precision, recall=recall, f05=f05)


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------

def evaluate_predictions(
    ground_truth: Dict[str, Union[Set[str], FrozenSet[str]]],
    predictions: Dict[str, Union[Set[str], FrozenSet[str]]],
) -> EvalResult:
    """Entity-level macro-averaged F0.5 evaluation.

    Iterates over every S1 entity in *ground_truth*, looks up its
    prediction in *predictions* (defaulting to empty set if absent),
    computes per-entity scores, and returns macro averages.

    Parameters
    ----------
    ground_truth : dict
        ``{s1_id: set(matched_ids)}``  — from ``data_loader.load_ground_truth``.
    predictions : dict
        ``{s1_id: set(predicted_ids)}``  — from the matching pipeline.

    Returns
    -------
    EvalResult
        Macro precision, macro recall, macro F0.5, entity count, and
        the full per-entity score list.
    """
    per_entity: List[EntityScore] = []

    for s1_id in sorted(ground_truth):
        true_ids = ground_truth[s1_id]
        pred_ids = predictions.get(s1_id, frozenset())

        score = entity_f05(true_ids, pred_ids)
        per_entity.append(score._replace(entity_id=s1_id))

    n = len(per_entity)
    if n == 0:
        return EvalResult(
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f05=0.0,
            entity_count=0,
            per_entity=[],
        )

    macro_p = sum(s.precision for s in per_entity) / n
    macro_r = sum(s.recall for s in per_entity) / n
    macro_f = sum(s.f05 for s in per_entity) / n

    return EvalResult(
        macro_precision=macro_p,
        macro_recall=macro_r,
        macro_f05=macro_f,
        entity_count=n,
        per_entity=per_entity,
    )
