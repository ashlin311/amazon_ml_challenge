"""
postprocess.py — Singleton and low-confidence post-processing.

Responsibilities:
  - Handle source1 entities with no matched candidates (singletons).
  - Filter or adjust low-confidence matches below a secondary threshold.
  - Ensure transitive consistency of match decisions if required.
  - Produce the final cleaned set of match predictions.
"""
