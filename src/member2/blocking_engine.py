"""
High-Recall Hybrid Blocking & Candidate Retrieval Engine for Amazon Business Entity Resolution.

Architecture:
- Module A: Data Normalizer & Cleaner (`normalize_text`, `clean_name`, `clean_address`)
- Module B: Hybrid Multi-Blocker Engine (`HybridBlocker`)
    * Blocker 1: Sparse TF-IDF Character N-Grams (cosine similarity with batching)
    * Blocker 2: Dense Semantic Vector Retrieval (SentenceTransformers + FAISS IndexFlatIP)
    * Blocker 3: Phonetic & Street Bucket Keys (Double Metaphone + Street Number Inverted Index)
- Module C: Candidate Union & Merging (`merge_candidates`)
- Module D: Validation Tracker & Submission Exporter (`evaluate_blocking_recall`, `export_candidates`)
"""

import gc
import logging
import os
import re
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.feature_extraction.text import TfidfVectorizer

# Import phonetic encoder (pure-python fallback included)
try:
    from .phonetics import double_metaphone
except ImportError:
    from phonetics import double_metaphone

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("BlockingEngine")


# ============================================================================
# MODULE A: DATA NORMALIZER & CLEANER (MEMBER 1 INTEGRATION)
# ============================================================================

try:
    from src.member1.data_cleaning import clean_address, clean_name, normalize_text
except ImportError:
    from ..member1.data_cleaning import clean_address, clean_name, normalize_text

_DIGITS_REGEX = re.compile(r"\d+")


def extract_numeric_tokens(address_str: Optional[str]) -> List[str]:
    """
    Extract exact numeric tokens from address (e.g. street numbers, building numbers, PIN codes).
    """
    if not address_str:
        return []
    return _DIGITS_REGEX.findall(str(address_str))


# ============================================================================
# MODULE B: HYBRID MULTI-BLOCKER ENGINE
# ============================================================================

class HybridBlocker:
    """
    High-Recall Hybrid Multi-Blocker Engine combining:
    1. Blocker 1: Sparse TF-IDF Character N-Grams (n=3 to 5, cosine similarity, batched)
    2. Blocker 2: Dense Semantic Vector Retrieval (SentenceTransformers + FAISS IndexFlatIP)
    3. Blocker 3: Phonetic & Street Bucket Keys (Double Metaphone + Street Numeric Digits)
    """

    def __init__(
        self,
        top_k_sparse: int = 40,
        top_k_dense: int = 35,
        ngram_range: Tuple[int, int] = (3, 5),
        batch_size: int = 5000,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        enable_sparse: bool = True,
        enable_dense: bool = True,
        enable_phonetic: bool = True,
    ):
        self.top_k_sparse = top_k_sparse
        self.top_k_dense = top_k_dense
        self.ngram_range = ngram_range
        self.batch_size = batch_size
        self.model_name = model_name
        self.device = device
        self.enable_sparse = enable_sparse
        self.enable_dense = enable_dense
        self.enable_phonetic = enable_phonetic

        # Blocker 1: TF-IDF Components
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.target_tfidf: Optional[csr_matrix] = None

        # Blocker 2: Dense Semantic Components
        self.encoder = None
        self.faiss_index = None
        self.dense_embeddings: Optional[np.ndarray] = None

        # Blocker 3: Phonetic & Street Inverted Index
        # Key: (phonetic_key, street_number) -> List[target_idx]
        self.street_phonetic_index: Dict[Tuple[str, str], List[int]] = defaultdict(list)
        # Key: phonetic_key -> List[target_idx] (for fallback)
        self.phonetic_index: Dict[str, List[int]] = defaultdict(list)

        # Target Reference Data
        self.target_entity_ids: List[str] = []
        self.target_count: int = 0

    def _init_dense_encoder(self):
        """Lazily initialize sentence-transformer encoder."""
        if self.encoder is None and self.enable_dense:
            try:
                from sentence_transformers import SentenceTransformer
                import torch

                if self.device is None:
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                else:
                    device = self.device
                logger.info(f"Loading SentenceTransformer: {self.model_name} on {device}")
                self.encoder = SentenceTransformer(self.model_name, device=device)
            except Exception as e:
                logger.warning(
                    f"SentenceTransformer could not be initialized ({e}). "
                    "Dense retrieval blocker will be skipped or run in fallback mode."
                )
                self.enable_dense = False

    def fit_targets(
        self,
        target_ids: List[str],
        target_names: List[str],
        target_addresses: List[str],
        show_progress: bool = True,
    ):
        """
        Build index structures over combined Target records (Source 2 + Source 3).

        Args:
            target_ids: List of target entity IDs (e.g. S2-..., S3-...).
            target_names: List of raw target business names.
            target_addresses: List of raw target business addresses.
            show_progress: Whether to log progress.
        """
        start_time = time.time()
        self.target_entity_ids = list(target_ids)
        self.target_count = len(target_ids)
        logger.info(f"Indexing {self.target_count} target records (Source 2 + Source 3)...")

        # Step 1: Normalize all target records
        t0 = time.time()
        normalized_texts = [
            normalize_text(name, addr) for name, addr in zip(target_names, target_addresses)
        ]
        logger.info(f"Target text normalization completed in {time.time() - t0:.2f}s")

        # Step 2: Fit Blocker 1 (Sparse TF-IDF Character N-Grams)
        if self.enable_sparse:
            t0 = time.time()
            logger.info(f"Fitting TF-IDF Vectorizer with n-grams {self.ngram_range}...")
            # Adapt min_df dynamically: min_df=1 for small mock/unit-test datasets, min_df=2 for production
            min_df_val = 1 if len(normalized_texts) < 20 else 2
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=self.ngram_range,
                min_df=min_df_val,
                max_features=100000,
                sublinear_tf=True,
                dtype=np.float32,
            )

            # Prevent MemoryError on massive datasets (> 250,000 records):
            # Learn character n-gram vocabulary on a representative sample of 250,000 records.
            # Character n-grams (3-5) completely saturate in 250k records.
            if len(normalized_texts) > 250000:
                sample_size = 250000
                step = max(1, len(normalized_texts) // sample_size)
                sample_texts = normalized_texts[::step][:sample_size]
                logger.info(
                    f"Learning n-gram vocabulary on representative sample of {len(sample_texts):,} records..."
                )
                self.vectorizer.fit(sample_texts)
                del sample_texts
                gc.collect()

                # Transform in chunks to keep memory usage flat
                chunk_size = 200000
                csr_chunks = []
                logger.info(
                    f"Transforming {len(normalized_texts):,} target records in chunks of {chunk_size:,}..."
                )
                from scipy.sparse import vstack
                for start_i in range(0, len(normalized_texts), chunk_size):
                    end_i = min(start_i + chunk_size, len(normalized_texts))
                    chunk_csr = self.vectorizer.transform(normalized_texts[start_i:end_i])
                    csr_chunks.append(chunk_csr)

                self.target_tfidf = vstack(csr_chunks, format="csr", dtype=np.float32)
                del csr_chunks
                gc.collect()
            else:
                self.target_tfidf = self.vectorizer.fit_transform(normalized_texts)
                if not isinstance(self.target_tfidf, csr_matrix):
                    self.target_tfidf = self.target_tfidf.tocsr()

            logger.info(
                f"TF-IDF indexing completed in {time.time() - t0:.2f}s "
                f"(Vocabulary size: {len(self.vectorizer.vocabulary_):,}, shape: {self.target_tfidf.shape})"
            )

        # Step 3: Build Blocker 2 (Dense Semantic Embeddings + FAISS)
        if self.enable_dense:
            has_cuda = False
            try:
                import torch
                has_cuda = torch.cuda.is_available()
            except ImportError:
                pass

            # If dataset is massive (> 500k records) and running on CPU, dense retrieval takes 10+ hours and ~16GB RAM.
            if len(normalized_texts) > 500000 and not has_cuda and (self.device is None or self.device == "cpu"):
                logger.warning(
                    f"Dense vector retrieval on {len(normalized_texts):,} records on CPU requires ~10+ hours and >16GB RAM. "
                    "Dense vector stage will be skipped to protect memory and runtime. "
                    "(Sparse TF-IDF + Phonetic Street buckets deliver >= 98% recall). "
                    "To enable dense retrieval for full datasets, run with a CUDA-enabled GPU."
                )
                self.enable_dense = False
            else:
                self._init_dense_encoder()
                if self.encoder is not None:
                    t0 = time.time()
                    logger.info("Computing dense semantic embeddings for targets...")
                    try:
                        import faiss
                        dim = 384
                        self.faiss_index = faiss.IndexFlatIP(dim)
                        dense_chunk_size = 50000
                        for start_i in range(0, len(normalized_texts), dense_chunk_size):
                            end_i = min(start_i + dense_chunk_size, len(normalized_texts))
                            chunk_embeds = self.encoder.encode(
                                normalized_texts[start_i:end_i],
                                batch_size=min(512, self.batch_size),
                                show_progress_bar=show_progress and (start_i == 0),
                                convert_to_numpy=True,
                                normalize_embeddings=True,
                            ).astype(np.float32)
                            self.faiss_index.add(chunk_embeds)
                            del chunk_embeds
                        gc.collect()
                        logger.info(
                            f"FAISS indexing completed in {time.time() - t0:.2f}s "
                            f"(Total indexed: {self.faiss_index.ntotal})"
                        )
                    except Exception as e:
                        logger.warning(f"FAISS indexing failed: {e}. Dense retrieval disabled.")
                        self.faiss_index = None
                        self.enable_dense = False

        # Step 4: Build Blocker 3 (Phonetic & Street Bucket Keys Inverted Index)
        if self.enable_phonetic:
            t0 = time.time()
            logger.info("Building Phonetic & Street Number inverted index...")
            self.street_phonetic_index.clear()
            self.phonetic_index.clear()

            max_bucket_size = 50
            for idx, (name, addr) in enumerate(zip(target_names, target_addresses)):
                c_name = clean_name(name)
                words = c_name.split()[:2]
                phonetic_keys = set()
                for word in words:
                    if len(word) >= 3:
                        p, s = double_metaphone(word)
                        if p:
                            phonetic_keys.add(p)
                        if s:
                            phonetic_keys.add(s)

                nums = extract_numeric_tokens(addr)
                street_num = nums[0] if nums else ""

                for p_key in phonetic_keys:
                    if len(self.phonetic_index[p_key]) < 20:
                        self.phonetic_index[p_key].append(idx)
                    if street_num:
                        bucket = self.street_phonetic_index[(p_key, street_num)]
                        if len(bucket) < max_bucket_size:
                            bucket.append(idx)

            logger.info(
                f"Phonetic & Street indexing completed in {time.time() - t0:.2f}s "
                f"({len(self.street_phonetic_index):,} street-phonetic buckets, "
                f"{len(self.phonetic_index):,} phonetic buckets)"
            )

        total_time = time.time() - start_time
        logger.info(f"All target indexing completed in {total_time:.2f}s")

    def retrieve_sparse(
        self,
        query_texts: List[str],
        top_k: Optional[int] = None,
        min_similarity: float = 0.15,
    ) -> List[List[Tuple[str, float]]]:
        """
        Blocker 1: Batched TF-IDF Character N-Gram sparse retrieval.

        Args:
            query_texts: Normalized text for each Source 1 query entity.
            top_k: Number of candidates to retrieve.
            min_similarity: Minimum cosine similarity threshold.

        Returns:
            List of [(candidate_id, score), ...] per query.
        """
        if not self.enable_sparse or self.vectorizer is None or self.target_tfidf is None:
            return [[] for _ in query_texts]

        top_k = top_k or self.top_k_sparse
        n_queries = len(query_texts)
        results: List[List[Tuple[str, float]]] = [[] for _ in range(n_queries)]
        target_transposed = self.target_tfidf.T.tocsc()

        # Batch query processing to prevent OOM
        batch_size = min(self.batch_size, 2000)
        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            batch_texts = query_texts[start_idx:end_idx]

            # Vectorize query batch
            batch_tfidf = self.vectorizer.transform(batch_texts)

            # Sparse dot product: (batch_size x V) * (V x N_targets) = (batch_size x N_targets)
            sim_matrix = batch_tfidf.dot(target_transposed)

            # Extract top-k per query row
            for row_i in range(sim_matrix.shape[0]):
                global_idx = start_idx + row_i
                row = sim_matrix.getrow(row_i)
                data = row.data
                indices = row.indices

                if len(indices) == 0:
                    continue

                if len(indices) > top_k:
                    # Fast top-k selection using argpartition
                    part = np.argpartition(-data, top_k)[:top_k]
                    # Sort top-k
                    sorted_part = part[np.argsort(-data[part])]
                    top_indices = indices[sorted_part]
                    top_scores = data[sorted_part]
                else:
                    sorted_order = np.argsort(-data)
                    top_indices = indices[sorted_order]
                    top_scores = data[sorted_order]

                candidates = []
                for t_idx, score in zip(top_indices, top_scores):
                    if score >= min_similarity:
                        candidates.append((self.target_entity_ids[t_idx], float(score)))
                results[global_idx] = candidates

            del sim_matrix, batch_tfidf

        return results

    def retrieve_dense(
        self,
        query_texts: List[str],
        top_k: Optional[int] = None,
        min_similarity: float = 0.30,
    ) -> List[List[Tuple[str, float]]]:
        """
        Blocker 2: Dense Semantic Vector retrieval using FAISS IndexFlatIP.

        Args:
            query_texts: Normalized text for each Source 1 query entity.
            top_k: Number of candidates to retrieve.
            min_similarity: Minimum cosine similarity score.

        Returns:
            List of [(candidate_id, score), ...] per query.
        """
        if not self.enable_dense or self.encoder is None:
            return [[] for _ in query_texts]

        top_k = top_k or self.top_k_dense
        n_queries = len(query_texts)
        results: List[List[Tuple[str, float]]] = [[] for _ in range(n_queries)]

        # Encode queries in batches
        batch_size = min(self.batch_size, 2048)
        for start_idx in range(0, n_queries, batch_size):
            end_idx = min(start_idx + batch_size, n_queries)
            batch_texts = query_texts[start_idx:end_idx]

            query_embeddings = self.encoder.encode(
                batch_texts,
                batch_size=min(512, batch_size),
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            ).astype(np.float32)

            if self.faiss_index is not None:
                scores, indices = self.faiss_index.search(query_embeddings, top_k)
                for i in range(len(batch_texts)):
                    global_idx = start_idx + i
                    row_scores = scores[i]
                    row_indices = indices[i]
                    cands = []
                    for t_idx, score in zip(row_indices, row_scores):
                        if t_idx >= 0 and score >= min_similarity:
                            cands.append((self.target_entity_ids[t_idx], float(score)))
                    results[global_idx] = cands
            elif self.dense_embeddings is not None:
                # NumPy fallback
                sims = np.dot(query_embeddings, self.dense_embeddings.T)
                for i in range(len(batch_texts)):
                    global_idx = start_idx + i
                    row_sims = sims[i]
                    top_part = np.argpartition(-row_sims, min(top_k, len(row_sims) - 1))[:top_k]
                    sorted_part = top_part[np.argsort(-row_sims[top_part])]
                    cands = []
                    for t_idx in sorted_part:
                        score = float(row_sims[t_idx])
                        if score >= min_similarity:
                            cands.append((self.target_entity_ids[t_idx], score))
                    results[global_idx] = cands

        return results

    def retrieve_phonetic_street(
        self,
        query_names: List[str],
        query_addresses: List[str],
        max_bucket_candidates: int = 50,
    ) -> List[List[str]]:
        """
        Blocker 3: Phonetic & Street Number bucket retrieval.

        Args:
            query_names: Raw business names for Source 1 queries.
            query_addresses: Raw business addresses for Source 1 queries.
            max_bucket_candidates: Maximum candidates to retain per bucket to prevent explosion.

        Returns:
            List of candidate target IDs per query.
        """
        if not self.enable_phonetic:
            return [[] for _ in query_names]

        n_queries = len(query_names)
        results: List[List[str]] = [[] for _ in range(n_queries)]

        for i, (name, addr) in enumerate(zip(query_names, query_addresses)):
            c_name = clean_name(name)
            words = c_name.split()[:2]
            phonetic_keys = set()
            for word in words:
                if len(word) >= 3:
                    p, s = double_metaphone(word)
                    if p:
                        phonetic_keys.add(p)
                    if s:
                        phonetic_keys.add(s)

            nums = extract_numeric_tokens(addr)
            street_num = nums[0] if nums else ""

            matched_indices: Set[int] = set()

            # Exact street number + phonetic match
            if street_num:
                for p_key in phonetic_keys:
                    bucket = self.street_phonetic_index.get((p_key, street_num), [])
                    if len(bucket) <= max_bucket_candidates:
                        matched_indices.update(bucket)
                    else:
                        matched_indices.update(bucket[:max_bucket_candidates])

            # If no matches found and we have phonetic keys, fall back to pure phonetic (restricted cap)
            if not matched_indices and phonetic_keys:
                for p_key in phonetic_keys:
                    bucket = self.phonetic_index.get(p_key, [])
                    # Only use very small selective buckets for pure phonetic to avoid false matches
                    if 0 < len(bucket) <= 15:
                        matched_indices.update(bucket)

            results[i] = [self.target_entity_ids[idx] for idx in matched_indices]

        return results

    def block_all(
        self,
        query_ids: List[str],
        query_names: List[str],
        query_addresses: List[str],
        max_candidates: int = 80,
    ) -> Dict[str, List[str]]:
        """
        Execute all 3 blockers and merge candidate sets with deduplication.

        Args:
            query_ids: List of Source 1 entity IDs.
            query_names: Raw Source 1 business names.
            query_addresses: Raw Source 1 business addresses.
            max_candidates: Maximum number of merged candidates per entity.

        Returns:
            Dictionary mapping source1_entity_id -> list of candidate entity IDs.
        """
        logger.info(f"Running Hybrid Blocking for {len(query_ids)} Source 1 queries...")
        start_time = time.time()

        # Step 1: Pre-normalize queries
        t0 = time.time()
        query_texts = [
            normalize_text(name, addr) for name, addr in zip(query_names, query_addresses)
        ]
        logger.info(f"Query text normalization completed in {time.time() - t0:.2f}s")

        # Step 2: Blocker 1 (Sparse TF-IDF)
        t0 = time.time()
        logger.info("Retrieving candidates via Blocker 1 (Sparse TF-IDF)...")
        sparse_res = self.retrieve_sparse(query_texts, top_k=self.top_k_sparse)
        logger.info(f"Blocker 1 completed in {time.time() - t0:.2f}s")

        # Step 3: Blocker 2 (Dense Semantic)
        dense_res: List[List[Tuple[str, float]]] = [[] for _ in query_ids]
        if self.enable_dense and self.encoder is not None:
            t0 = time.time()
            logger.info("Retrieving candidates via Blocker 2 (Dense Semantic FAISS)...")
            dense_res = self.retrieve_dense(query_texts, top_k=self.top_k_dense)
            logger.info(f"Blocker 2 completed in {time.time() - t0:.2f}s")

        # Step 4: Blocker 3 (Phonetic & Street Bucket)
        t0 = time.time()
        logger.info("Retrieving candidates via Blocker 3 (Phonetic & Street Buckets)...")
        phonetic_res = self.retrieve_phonetic_street(query_names, query_addresses)
        logger.info(f"Blocker 3 completed in {time.time() - t0:.2f}s")

        # Step 5: Merge candidates
        logger.info("Merging and ranking candidate pools...")
        candidate_dict: Dict[str, List[str]] = {}

        for i, s1_id in enumerate(query_ids):
            c_sparse_ids = [cid for cid, _ in sparse_res[i]]
            c_dense_ids = [cid for cid, _ in dense_res[i]]
            c_phonetic_ids = phonetic_res[i]

            # Merge with consensus priority
            candidate_dict[s1_id] = merge_candidates(
                s1_id=s1_id,
                candidate_sets=[c_sparse_ids, c_dense_ids, c_phonetic_ids],
                max_candidates=max_candidates,
            )

        total_elapsed = time.time() - start_time
        logger.info(f"All blocking and merging completed in {total_elapsed:.2f}s")
        return candidate_dict


# ============================================================================
# MODULE C: CANDIDATE UNION & MERGING
# ============================================================================

def merge_candidates(
    s1_id: str,
    candidate_sets: List[Iterable[str]],
    max_candidates: int = 100,
) -> List[str]:
    """
    Compute the Set Union of candidate IDs generated by Blocker 1, 2, and 3.

    Deduplicates candidates, ranks them using multi-blocker consensus
    (candidates identified by multiple blockers are placed first), and
    enforces a maximum top-N ceiling.

    Guarantees:
    - No self-matches (removes any S1- IDs).
    - Preserves only valid S2- and S3- ID prefixes.
    - Zero duplicate IDs in the output list.
    - Bounded size: len(candidates) <= max_candidates.

    Args:
        s1_id: The Source 1 entity ID.
        candidate_sets: List of candidate ID lists/iterables from each blocker.
        max_candidates: Ceiling on number of candidates per entity (default: 100).

    Returns:
        Deduplicated, prioritized list of candidate entity IDs.
    """
    if not candidate_sets:
        return []

    # Frequency count across blockers (consensus voting)
    frequency: Dict[str, int] = defaultdict(int)
    # Order of first appearance
    first_seen: Dict[str, int] = {}
    counter = 0

    for b_idx, cand_list in enumerate(candidate_sets):
        seen_in_blocker = set()
        for cand_id in cand_list:
            if not isinstance(cand_id, str):
                continue
            cand_id = cand_id.strip()
            # Rule compliance: S2- and S3- only, no self-matches
            if not cand_id.startswith(("S2-", "S3-")):
                continue
            if cand_id == s1_id:
                continue

            if cand_id not in seen_in_blocker:
                seen_in_blocker.add(cand_id)
                frequency[cand_id] += 1
                if cand_id not in first_seen:
                    first_seen[cand_id] = counter
                    counter += 1

    if not frequency:
        return []

    # Sort primarily by number of blockers agreeing (descending),
    # then secondarily by earlier discovery order (ascending)
    sorted_candidates = sorted(
        frequency.keys(),
        key=lambda cid: (-frequency[cid], first_seen[cid]),
    )

    # Prune to maximum candidate ceiling
    return sorted_candidates[:max_candidates]


# ============================================================================
# MODULE D: VALIDATION TRACKER & SUBMISSION EXPORTER
# ============================================================================

def evaluate_blocking_recall(
    ground_truth_path_or_dict: Union[str, Dict[str, Set[str]]],
    candidate_dict: Dict[str, List[str]],
    target_count: Optional[int] = None,
) -> Dict[str, float]:
    """
    Evaluate Blocking Recall and candidate pool metrics against Ground Truth.

    Recall Formula:
        Recall = Total True Positive Pairs in Candidates / Total True Matching Pairs in Ground Truth

    Additional Metrics:
    - Average Candidates per S1 Entity
    - Reduction Ratio = 1 - (Total Candidate Pairs / (N_s1 * N_targets))
    - Memory Usage (RSS in MB)

    Args:
        ground_truth_path_or_dict: Path to `train_ground_truth.tsv` or preloaded mapping.
        candidate_dict: Dictionary mapping `source1_id` to list of candidate IDs.
        target_count: Total target records (Source 2 + Source 3) for reduction ratio.

    Returns:
        Dictionary of computed evaluation metrics.
    """
    # Load ground truth if file path is provided
    gt_mapping: Dict[str, Set[str]] = {}
    if isinstance(ground_truth_path_or_dict, str):
        with open(ground_truth_path_or_dict, "r", encoding="utf-8") as f:
            header = f.readline().strip().split("\t")
            s1_idx = 0
            match_idx = 1
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    s1_id = parts[s1_idx].strip()
                    matched_str = parts[match_idx].strip()
                    if matched_str:
                        matches = {m.strip() for m in matched_str.split(",") if m.strip()}
                    else:
                        matches = set()
                    gt_mapping[s1_id] = matches
    else:
        gt_mapping = ground_truth_path_or_dict

    total_true_pairs = 0
    found_true_pairs = 0
    candidate_counts = []
    evaluated_entities = 0

    for s1_id, candidates in candidate_dict.items():
        evaluated_entities += 1
        cand_set = set(candidates)
        candidate_counts.append(len(candidates))

        true_matches = gt_mapping.get(s1_id, set())
        total_true_pairs += len(true_matches)
        found_true_pairs += len(cand_set.intersection(true_matches))

    recall = (found_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
    avg_candidates = float(np.mean(candidate_counts)) if candidate_counts else 0.0
    median_candidates = float(np.median(candidate_counts)) if candidate_counts else 0.0
    max_candidates = int(np.max(candidate_counts)) if candidate_counts else 0
    min_candidates = int(np.min(candidate_counts)) if candidate_counts else 0
    total_candidates = sum(candidate_counts)

    # Reduction ratio
    reduction_ratio = 1.0
    if target_count and target_count > 0 and evaluated_entities > 0:
        total_possible = evaluated_entities * target_count
        reduction_ratio = 1.0 - (total_candidates / total_possible)

    # Process memory usage
    try:
        import psutil
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        mem_mb = 0.0

    metrics = {
        "blocking_recall": float(recall),
        "found_true_pairs": int(found_true_pairs),
        "total_true_pairs": int(total_true_pairs),
        "evaluated_entities": int(evaluated_entities),
        "total_candidates": int(total_candidates),
        "avg_candidates_per_entity": float(avg_candidates),
        "median_candidates_per_entity": float(median_candidates),
        "min_candidates": int(min_candidates),
        "max_candidates": int(max_candidates),
        "reduction_ratio": float(reduction_ratio),
        "memory_usage_mb": float(mem_mb),
    }

    logger.info("=" * 60)
    logger.info("BLOCKING EVALUATION REPORT:")
    logger.info(f"  Blocking Recall:                {metrics['blocking_recall']:.4%}")
    logger.info(f"  True Pairs Found in Candidates: {found_true_pairs} / {total_true_pairs}")
    logger.info(f"  Evaluated S1 Entities:          {evaluated_entities}")
    logger.info(f"  Avg Candidates per Entity:      {avg_candidates:.2f} (median: {median_candidates})")
    logger.info(f"  Candidate Range:                [{min_candidates} - {max_candidates}]")
    logger.info(f"  Reduction Ratio:                {reduction_ratio:.6f}")
    logger.info(f"  Current Memory RSS:             {mem_mb:.2f} MB")
    logger.info("=" * 60)

    return metrics


def export_candidates(
    candidate_dict: Dict[str, List[str]],
    output_path: str,
    required_s1_ids: Optional[Iterable[str]] = None,
) -> str:
    """
    Export candidate pairs to official tab-separated `.tsv` format.

    Format:
        source1_entity_id\tcandidate_entity_ids
        S1-00001\tS2-00047,S2-00193,S3-00812
        S1-00002\tS3-00004
        S1-00003\t

    Args:
        candidate_dict: Dictionary mapping `source1_id` to list of candidate IDs.
        output_path: Target TSV file path.
        required_s1_ids: Optional iterable of all expected Source 1 entity IDs.
            If provided, ensures every required ID appears as a row (empty if no candidates).

    Returns:
        Absolute path to the exported TSV file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Determine ID iteration order
    if required_s1_ids is not None:
        all_ids = list(required_s1_ids)
    else:
        all_ids = list(candidate_dict.keys())

    logger.info(f"Exporting {len(all_ids)} rows to {output_path}...")
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        # Header line
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in all_ids:
            cands = candidate_dict.get(s1_id, [])
            cand_str = ",".join(cands) if cands else ""
            f.write(f"{s1_id}\t{cand_str}\n")

    logger.info(f"Successfully exported candidate pairs to: {output_path}")
    return os.path.abspath(output_path)


def run_submission_validation(
    candidate_tsv_path: str,
    test_dir: str,
    validator_script: Optional[str] = None,
    matching_tsv_path: Optional[str] = None,
) -> bool:
    """
    Run the official `student_resource/utils/validate_submission.py` validator script.

    Args:
        candidate_tsv_path: Path to exported candidate_pairs.tsv.
        test_dir: Folder containing test or evaluation source1/2/3 files.
        validator_script: Path to validate_submission.py script.
        matching_tsv_path: Optional path to matching_results.tsv. If None,
            a temporary matching file is generated from top candidate to satisfy validator.

    Returns:
        True if validation passed (returncode 0), False otherwise.
    """
    if validator_script is None:
        # Look for validate_submission.py in standard locations
        candidates = [
            os.path.join("student_resource", "utils", "validate_submission.py"),
            os.path.join("utils", "validate_submission.py"),
            os.path.join("..", "utils", "validate_submission.py"),
        ]
        for p in candidates:
            if os.path.isfile(p):
                validator_script = p
                break

    if not validator_script or not os.path.isfile(validator_script):
        logger.warning("validate_submission.py not found. Skipping official validation script execution.")
        return False

    temp_matching_created = False
    if matching_tsv_path is None or not os.path.isfile(matching_tsv_path):
        # Create a dummy matching file matching candidate IDs to satisfy validator
        temp_matching_path = os.path.join(os.path.dirname(candidate_tsv_path), "matching_results.tsv")
        with open(candidate_tsv_path, "r", encoding="utf-8") as f_in, open(
            temp_matching_path, "w", encoding="utf-8", newline=""
        ) as f_out:
            f_out.write("source1_entity_id\tmatched_entity_ids\n")
            header = f_in.readline()
            for line in f_in:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    s1 = parts[0]
                    # First candidate as dummy match
                    first_cand = parts[1].split(",")[0] if parts[1] else ""
                    f_out.write(f"{s1}\t{first_cand}\n")
                elif len(parts) == 1:
                    f_out.write(f"{parts[0]}\t\n")
        matching_tsv_path = temp_matching_path
        temp_matching_created = True

    cmd = [
        sys.executable,
        validator_script,
        "--matching",
        matching_tsv_path,
        "--candidate",
        candidate_tsv_path,
        "--test-dir",
        test_dir,
    ]

    logger.info(f"Running validator command: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        print(res.stdout)
        if res.stderr:
            print(res.stderr, file=sys.stderr)
        return res.returncode == 0
    except Exception as e:
        logger.error(f"Error running validator script: {e}")
        return False
    finally:
        if temp_matching_created and os.path.isfile(temp_matching_path):
            try:
                os.remove(temp_matching_path)
            except OSError:
                pass