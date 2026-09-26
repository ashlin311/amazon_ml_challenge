"""
candidate_eval.py — Candidate-set diagnostics for Member 2's blocking output.

Compares candidate_pairs against training ground truth to measure:
  - S1 coverage
  - Candidate volume statistics
  - Candidate-generation recall (NOT model recall)
  - Singleton/no-match diagnostics
  - Candidate recall broken down by number of true matches

CLI usage::

    python -m src.member3.candidate_eval \\
        --candidate-file output/candidate_pairs.tsv \\
        --train-dir data/train
"""

import argparse
import os
import statistics
import sys
import time
from collections import Counter
from typing import Dict, FrozenSet, List, NamedTuple, Tuple

from src.member3.candidate_loader import load_candidate_pairs
from src.member3.data_loader import GroundTruth, load_ground_truth


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

class CoverageStats(NamedTuple):
    gt_s1_count: int
    candidate_s1_count: int
    covered_s1_count: int
    missing_s1_count: int
    coverage_fraction: float


class VolumeStats(NamedTuple):
    total_candidates: int
    # Including S1 entities with zero candidates:
    mean_all: float
    median_all: float
    p95_all: float
    max_all: int
    min_all: int
    # Excluding S1 entities with zero candidates:
    nonempty_count: int
    mean_nonempty: float
    median_nonempty: float
    p95_nonempty: float
    max_nonempty: int
    min_nonempty: int


class RecallStats(NamedTuple):
    total_true_matches: int
    covered_true_matches: int
    candidate_recall: float
    matched_s1_count: int  # S1 entities with >=1 true match
    perfect_recall_s1_count: int  # matched S1 with all true matches covered
    perfect_recall_fraction: float


class SingletonStats(NamedTuple):
    true_singleton_count: int
    singleton_zero_candidates: int
    singleton_with_candidates: int


class RecallByMatchCount(NamedTuple):
    match_count: int
    s1_count: int
    total_true: int
    covered_true: int
    recall: float
    perfect_count: int
    perfect_fraction: float


class CandidateReport(NamedTuple):
    coverage: CoverageStats
    volume: VolumeStats
    recall: RecallStats
    singletons: SingletonStats
    recall_by_match_count: List[RecallByMatchCount]
    duplicate_candidate_ids: int


# ---------------------------------------------------------------------------
# Core diagnostics
# ---------------------------------------------------------------------------

def _percentile(sorted_values: List[int], p: float) -> float:
    """Calculate the p-th percentile (0-100) of a sorted list."""
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    k = (p / 100.0) * (n - 1)
    f = int(k)
    c = f + 1
    if c >= n:
        return float(sorted_values[-1])
    d = k - f
    return sorted_values[f] + d * (sorted_values[c] - sorted_values[f])


def compute_coverage(
    ground_truth: GroundTruth,
    candidates: Dict[str, FrozenSet[str]],
) -> CoverageStats:
    """Measure how many ground-truth S1 entities appear in the candidate file."""
    gt_ids = set(ground_truth.keys())
    cand_ids = set(candidates.keys())
    covered = gt_ids & cand_ids
    missing = gt_ids - cand_ids
    gt_n = len(gt_ids)
    return CoverageStats(
        gt_s1_count=gt_n,
        candidate_s1_count=len(cand_ids),
        covered_s1_count=len(covered),
        missing_s1_count=len(missing),
        coverage_fraction=len(covered) / gt_n if gt_n else 0.0,
    )


def compute_volume(
    ground_truth: GroundTruth,
    candidates: Dict[str, FrozenSet[str]],
) -> VolumeStats:
    """Candidate-volume distribution across covered GT S1 entities."""
    gt_ids = sorted(ground_truth.keys())
    sizes_all: List[int] = []
    sizes_nonempty: List[int] = []

    for s1 in gt_ids:
        cands = candidates.get(s1, frozenset())
        n = len(cands)
        sizes_all.append(n)
        if n > 0:
            sizes_nonempty.append(n)

    sizes_all.sort()
    sizes_nonempty.sort()

    total = sum(sizes_all)

    def _stats(vals: List[int]):
        if not vals:
            return 0.0, 0.0, 0.0, 0, 0
        return (
            statistics.mean(vals),
            statistics.median(vals),
            _percentile(vals, 95),
            vals[-1],
            vals[0],
        )

    m_all, med_all, p95_all, max_all, min_all = _stats(sizes_all)
    m_ne, med_ne, p95_ne, max_ne, min_ne = _stats(sizes_nonempty)

    return VolumeStats(
        total_candidates=total,
        mean_all=m_all,
        median_all=med_all,
        p95_all=p95_all,
        max_all=max_all,
        min_all=min_all,
        nonempty_count=len(sizes_nonempty),
        mean_nonempty=m_ne,
        median_nonempty=med_ne,
        p95_nonempty=p95_ne,
        max_nonempty=max_ne,
        min_nonempty=min_ne,
    )


def compute_recall(
    ground_truth: GroundTruth,
    candidates: Dict[str, FrozenSet[str]],
) -> RecallStats:
    """Candidate-generation recall: what fraction of true matches are in the
    candidate set?"""
    total_true = 0
    covered_true = 0
    matched_s1 = 0
    perfect_s1 = 0

    for s1, true_ids in ground_truth.items():
        if not true_ids:
            continue  # singletons handled separately
        matched_s1 += 1
        cands = candidates.get(s1, frozenset())
        n_true = len(true_ids)
        n_covered = len(true_ids & cands)
        total_true += n_true
        covered_true += n_covered
        if n_covered == n_true:
            perfect_s1 += 1

    return RecallStats(
        total_true_matches=total_true,
        covered_true_matches=covered_true,
        candidate_recall=covered_true / total_true if total_true else 0.0,
        matched_s1_count=matched_s1,
        perfect_recall_s1_count=perfect_s1,
        perfect_recall_fraction=perfect_s1 / matched_s1 if matched_s1 else 0.0,
    )


def compute_singleton_stats(
    ground_truth: GroundTruth,
    candidates: Dict[str, FrozenSet[str]],
) -> SingletonStats:
    """Diagnostics for true singleton entities (no ground-truth matches)."""
    n_singleton = 0
    n_zero = 0
    n_with = 0

    for s1, true_ids in ground_truth.items():
        if true_ids:
            continue
        n_singleton += 1
        cands = candidates.get(s1, frozenset())
        if len(cands) == 0:
            n_zero += 1
        else:
            n_with += 1

    return SingletonStats(
        true_singleton_count=n_singleton,
        singleton_zero_candidates=n_zero,
        singleton_with_candidates=n_with,
    )


def compute_recall_by_match_count(
    ground_truth: GroundTruth,
    candidates: Dict[str, FrozenSet[str]],
) -> List[RecallByMatchCount]:
    """Candidate recall grouped by number of true matches per S1."""
    # Group S1 entities by their ground-truth match count.
    groups: Dict[int, List[str]] = {}
    for s1, true_ids in ground_truth.items():
        mc = len(true_ids)
        groups.setdefault(mc, []).append(s1)

    results: List[RecallByMatchCount] = []
    for mc in sorted(groups):
        s1_list = groups[mc]
        total_t = 0
        covered_t = 0
        perfect = 0
        for s1 in s1_list:
            true_ids = ground_truth[s1]
            cands = candidates.get(s1, frozenset())
            n_true = len(true_ids)
            n_cov = len(true_ids & cands)
            total_t += n_true
            covered_t += n_cov
            if n_cov == n_true:
                perfect += 1

        results.append(RecallByMatchCount(
            match_count=mc,
            s1_count=len(s1_list),
            total_true=total_t,
            covered_true=covered_t,
            recall=covered_t / total_t if total_t else 1.0,
            perfect_count=perfect,
            perfect_fraction=perfect / len(s1_list) if s1_list else 0.0,
        ))

    return results


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def evaluate_candidates(
    ground_truth: GroundTruth,
    candidates: Dict[str, FrozenSet[str]],
    duplicate_candidate_ids: int = 0,
) -> CandidateReport:
    """Run all candidate diagnostics and return a structured report."""
    return CandidateReport(
        coverage=compute_coverage(ground_truth, candidates),
        volume=compute_volume(ground_truth, candidates),
        recall=compute_recall(ground_truth, candidates),
        singletons=compute_singleton_stats(ground_truth, candidates),
        recall_by_match_count=compute_recall_by_match_count(
            ground_truth, candidates
        ),
        duplicate_candidate_ids=duplicate_candidate_ids,
    )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def print_report(report: CandidateReport) -> None:
    """Print a human-readable diagnostic summary to stdout."""
    c = report.coverage
    v = report.volume
    r = report.recall
    s = report.singletons

    print()
    print("=" * 68)
    print(" CANDIDATE-SET DIAGNOSTICS")
    print("=" * 68)

    # Coverage
    print()
    print("--- S1 Coverage ---")
    print(f"  Ground-truth S1 count       : {c.gt_s1_count:,}")
    print(f"  Candidate-file S1 count     : {c.candidate_s1_count:,}")
    print(f"  Covered S1 (in both)        : {c.covered_s1_count:,}")
    print(f"  Missing S1 (GT only)        : {c.missing_s1_count:,}")
    print(f"  Candidate S1 coverage       : {c.coverage_fraction:.6f}")

    # Volume
    print()
    print("--- Candidate Volume (all GT S1 entities) ---")
    print(f"  Total candidate pairs       : {v.total_candidates:,}")
    print(f"  Average candidates/S1       : {v.mean_all:.2f}")
    print(f"  Median candidates/S1        : {v.median_all:.1f}")
    print(f"  P95 candidates/S1           : {v.p95_all:.1f}")
    print(f"  Maximum candidates/S1       : {v.max_all:,}")
    print(f"  Minimum candidates/S1       : {v.min_all:,}")

    print()
    print(f"--- Candidate Volume (non-empty only, n={v.nonempty_count:,}) ---")
    if v.nonempty_count > 0:
        print(f"  Average candidates/S1       : {v.mean_nonempty:.2f}")
        print(f"  Median candidates/S1        : {v.median_nonempty:.1f}")
        print(f"  P95 candidates/S1           : {v.p95_nonempty:.1f}")
        print(f"  Maximum candidates/S1       : {v.max_nonempty:,}")
        print(f"  Minimum candidates/S1       : {v.min_nonempty:,}")
    else:
        print("  (no non-empty candidate sets)")

    # Recall
    print()
    print("--- Candidate-Generation Recall ---")
    print(f"  Total true matches          : {r.total_true_matches:,}")
    print(f"  Covered true matches        : {r.covered_true_matches:,}")
    print(f"  Candidate recall            : {r.candidate_recall:.6f}")
    print()
    print(f"  Matched S1 entities (>=1 true match) : {r.matched_s1_count:,}")
    print(f"  S1 with all true matches covered     : {r.perfect_recall_s1_count:,}")
    print(f"  Fraction with perfect recall         : {r.perfect_recall_fraction:.6f}")

    # Singletons
    print()
    print("--- Singleton / No-Match Diagnostics ---")
    print(f"  True singleton S1 count     : {s.true_singleton_count:,}")
    print(f"    zero candidates           : {s.singleton_zero_candidates:,}")
    print(f"    >=1 candidates            : {s.singleton_with_candidates:,}")

    # Duplicate candidate IDs
    if report.duplicate_candidate_ids > 0:
        print()
        print(f"  [WARNING] Duplicate candidate IDs normalized: "
              f"{report.duplicate_candidate_ids:,}")

    # Recall by match count
    print()
    print("--- Candidate Recall by Number of True Matches ---")
    print(f"  {'Matches':>8}  {'S1 count':>10}  {'True':>8}  {'Covered':>8}"
          f"  {'Recall':>8}  {'Perfect':>8}  {'Perf%':>7}")
    print(f"  {'--------':>8}  {'--------':>10}  {'-----':>8}  {'-------':>8}"
          f"  {'------':>8}  {'-------':>8}  {'-----':>7}")
    for g in report.recall_by_match_count:
        print(
            f"  {g.match_count:>8}  {g.s1_count:>10,}  {g.total_true:>8,}"
            f"  {g.covered_true:>8,}  {g.recall:>8.4f}"
            f"  {g.perfect_count:>8,}  {g.perfect_fraction:>6.2%}"
        )

    print()
    print("=" * 68)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate candidate-generation quality against ground truth.",
    )
    parser.add_argument(
        "--candidate-file",
        default="output/candidate_pairs.tsv",
        help="Path to Member 2's candidate_pairs.tsv.",
    )
    parser.add_argument(
        "--train-dir",
        default="data/train",
        help="Directory containing train_ground_truth.tsv.",
    )

    args = parser.parse_args(argv)

    # --- Load ground truth ---
    t0 = time.perf_counter()
    gt_path = os.path.join(args.train_dir, "train_ground_truth.tsv")
    print(f"Loading ground truth from {gt_path} ...")
    ground_truth = load_ground_truth(gt_path)
    t_gt = time.perf_counter() - t0
    print(f"  {len(ground_truth):,} S1 entities loaded ({t_gt:.1f}s)")

    # --- Load candidate pairs ---
    t1 = time.perf_counter()
    print(f"Loading candidate pairs from {args.candidate_file} ...")
    try:
        candidates, dup_ids = load_candidate_pairs(args.candidate_file)
    except (ValueError, FileNotFoundError) as exc:
        print(f"\n[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)
    t_cand = time.perf_counter() - t1
    print(f"  {len(candidates):,} S1 entries loaded ({t_cand:.1f}s)")
    if dup_ids:
        print(f"  [WARNING] {dup_ids:,} duplicate candidate IDs were normalized")

    # --- Evaluate ---
    t2 = time.perf_counter()
    print("Computing diagnostics ...")
    report = evaluate_candidates(ground_truth, candidates, dup_ids)
    t_eval = time.perf_counter() - t2
    print(f"  Done ({t_eval:.1f}s)")

    # --- Print ---
    print_report(report)

    total_time = time.perf_counter() - t0
    print(f"Total wall time: {total_time:.1f}s")

    return report


if __name__ == "__main__":
    main()
