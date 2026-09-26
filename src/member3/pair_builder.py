"""
pair_builder.py — Construct labeled entity pairs for training and unlabeled pairs for inference.

Responsibilities:
  - Parse candidate_pairs.tsv (from Member 2) into (entity_a, entity_b) tuples.
  - Join with ground-truth to assign match/non-match labels for training pairs.
  - Handle singletons (source1 entities with no candidates).
  - Produce pair DataFrames ready for the Dataset class.

Expected candidate_pairs.tsv format:
  source1_entity_id    candidate_entity_ids
  S1-00001             S2-00047,S2-00193,S3-00812
  S1-00002             S3-00004
  S1-00003
"""
