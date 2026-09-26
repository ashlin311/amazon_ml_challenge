"""
data_loader.py — Load and merge entity data from source TSV files.

Responsibilities:
  - Read train/test source TSV files (source1, source2, source3).
  - Read ground-truth labels for training.
  - Read candidate_pairs.tsv produced by Member 2's blocking pipeline.
  - Provide unified entity lookup by entity ID across all sources.
"""
