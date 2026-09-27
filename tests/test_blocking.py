"""
Unit Tests for High-Recall Hybrid Blocking & Candidate Retrieval Engine.

Covers:
- Module A: Text Normalization, Noise Cleaning & Numeric Digit Preservation
- Module B: Blocker 1 (TF-IDF N-grams), Blocker 2 (Dense FAISS), Blocker 3 (Phonetic & Street)
- Module C: Candidate Union, Consensus Ranking, Ceiling Pruning & Deduplication
- Module D: Blocking Recall Evaluation & TSV Export Compliance
"""

import os
import sys
import tempfile
import unittest
import numpy as np

# Ensure src is importable
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.blocking_engine import (
    clean_name,
    clean_address,
    normalize_text,
    extract_numeric_tokens,
    HybridBlocker,
    merge_candidates,
    evaluate_blocking_recall,
    export_candidates,
)
from src.member2.phonetics import double_metaphone


class TestTextNormalization(unittest.TestCase):
    """Test Module A: Data Normalizer & Cleaner."""

    def test_standard_us_abbreviations(self):
        name = "Summit Corporation, Incorporated"
        addr = "105 Elm Street, Suite 200, Morganton, NC"
        norm = normalize_text(name, addr)

        # Standard abbreviations applied
        self.assertIn("corp", norm)
        self.assertIn("inc", norm)
        self.assertIn("st", norm)
        self.assertIn("ste", norm)
        # Numbers preserved
        self.assertIn("105", norm)
        self.assertIn("200", norm)
        # No punctuation
        self.assertNotIn(",", norm)
        self.assertNotIn(".", norm)

    def test_indian_address_and_multilingual(self):
        name = "Prabhav Business Center Private Limited"
        addr = "797, Lake Town Block A, Kolkata, Howrah, West Bengal"
        norm = normalize_text(name, addr)

        self.assertIn("pvt", norm)
        self.assertIn("ltd", norm)
        self.assertIn("797", norm)
        self.assertIn("kolkata", norm)

        # Devanagari text preservation
        hindi_name = "राम मार्केटिंग प्राइवेट लिमिटेड"
        c_hindi = clean_name(hindi_name)
        self.assertTrue(len(c_hindi) > 0)
        self.assertIn("राम", c_hindi)

    def test_french_address_and_accents(self):
        name = "ZNB Club SARL"
        addr = "175 Boulevard du Président Franklin Roosevelt, Bordeaux"
        norm = normalize_text(name, addr)

        self.assertIn("sarl", norm)
        self.assertIn("blvd", norm)
        self.assertIn("175", norm)
        self.assertIn("bordeaux", norm)

    def test_url_and_noise_removal(self):
        name = "Shivshakti Overseas Corp | www.shivshakti.com"
        addr = "H.NO 204 C Road, Punjab"
        norm = normalize_text(name, addr)

        self.assertNotIn("www", norm)
        self.assertNotIn("http", norm)
        self.assertIn("204", norm)
        self.assertIn("rd", norm)

    def test_extract_numeric_tokens(self):
        addr1 = "H.No.16-11-23/37/A, Flat No.207, Hyderabad"
        nums = extract_numeric_tokens(addr1)
        self.assertEqual(nums, ["16", "11", "23", "37", "207"])

        addr2 = "No numbers here"
        self.assertEqual(extract_numeric_tokens(addr2), [])


class TestPhoneticEncoding(unittest.TestCase):
    """Test Double Metaphone Phonetic Encoding."""

    def test_double_metaphone_english(self):
        p1, s1 = double_metaphone("Barbershop")
        self.assertTrue(len(p1) > 0)
        self.assertEqual(p1[0], "P")  # 'B' encodes to 'P' in Metaphone

        p2, s2 = double_metaphone("Orelee")
        self.assertTrue(len(p2) > 0)

    def test_double_metaphone_variations(self):
        # Similar sounding names should produce identical or overlapping phonetic keys
        p_smith, _ = double_metaphone("Smith")
        p_smyth, _ = double_metaphone("Smythe")
        self.assertEqual(p_smith, p_smyth)


class TestHybridBlockerWithMockData(unittest.TestCase):
    """Test Module B: Hybrid Multi-Blocker Engine with mock data."""

    def setUp(self):
        self.target_ids = [
            "S2-101",
            "S2-102",
            "S3-201",
            "S3-202",
            "S2-103",
        ]
        self.target_names = [
            "Orelee Barbershop Inc",
            "Prime Money Services LLC",
            "Summit Corp",
            "Christ Chapel Church",
            "Ram Marketing Private Limited",
        ]
        self.target_addresses = [
            "1795 Westchester Drive, High Point, NC",
            "17560 Ellis Road, Tahlequah, OK",
            "105 Elm Street, Morganton, NC",
            "2100 Cameron Drive, Unit G, Dundalk, MD",
            "KH NO 570/13, New Delhi, Delhi",
        ]

        self.query_ids = ["S1-001", "S1-002", "S1-003"]
        self.query_names = [
            "Orelee's Barbershop",
            "Summit Incorporated",
            "Unrelated Business LLC",
        ]
        self.query_addresses = [
            "1795 Westchester Dr, High Point, NC",
            "105 Elm St, Morganton, NC",
            "9999 Nowhere Lane, Unknown, ZZ",
        ]

    def test_sparse_tfidf_retrieval(self):
        blocker = HybridBlocker(
            top_k_sparse=3,
            top_k_dense=2,
            batch_size=10,
            enable_sparse=True,
            enable_dense=False,
            enable_phonetic=False,
        )
        blocker.fit_targets(self.target_ids, self.target_names, self.target_addresses)

        query_texts = [
            normalize_text(n, a) for n, a in zip(self.query_names, self.query_addresses)
        ]
        sparse_res = blocker.retrieve_sparse(query_texts, top_k=3)

        self.assertEqual(len(sparse_res), 3)
        # S1-001 should retrieve S2-101 as top candidate
        top_s1_cands = [cid for cid, _ in sparse_res[0]]
        self.assertIn("S2-101", top_s1_cands)

        # S1-002 should retrieve S3-201 as top candidate
        top_s2_cands = [cid for cid, _ in sparse_res[1]]
        self.assertIn("S3-201", top_s2_cands)

    def test_phonetic_street_retrieval(self):
        blocker = HybridBlocker(
            top_k_sparse=3,
            top_k_dense=2,
            enable_sparse=False,
            enable_dense=False,
            enable_phonetic=True,
        )
        blocker.fit_targets(self.target_ids, self.target_names, self.target_addresses)

        res = blocker.retrieve_phonetic_street(self.query_names, self.query_addresses)
        self.assertEqual(len(res), 3)
        # Query 1 (1795, Orelee) should match Target 0 (1795, Orelee)
        self.assertIn("S2-101", res[0])
        # Query 2 (105, Summit) should match Target 2 (105, Summit)
        self.assertIn("S3-201", res[1])

    def test_block_all_and_union(self):
        blocker = HybridBlocker(
            top_k_sparse=3,
            top_k_dense=2,
            enable_sparse=True,
            enable_dense=False,  # Skip dense in fast unit test
            enable_phonetic=True,
        )
        blocker.fit_targets(self.target_ids, self.target_names, self.target_addresses)
        cand_dict = blocker.block_all(
            self.query_ids,
            self.query_names,
            self.query_addresses,
            max_candidates=10,
        )

        self.assertIn("S1-001", cand_dict)
        self.assertIn("S1-002", cand_dict)
        self.assertIn("S1-003", cand_dict)
        self.assertIn("S2-101", cand_dict["S1-001"])
        self.assertIn("S3-201", cand_dict["S1-002"])


class TestCandidateMerging(unittest.TestCase):
    """Test Module C: Candidate Union & Merging."""

    def test_merge_candidates_consensus_and_ceiling(self):
        s1_id = "S1-999"
        # Blocker 1 found: S2-1, S3-1, S2-2
        b1 = ["S2-1", "S3-1", "S2-2"]
        # Blocker 2 found: S3-1, S2-3, S2-1
        b2 = ["S3-1", "S2-3", "S2-1"]
        # Blocker 3 found: S3-1, S2-4
        b3 = ["S3-1", "S2-4"]

        merged = merge_candidates(s1_id, [b1, b2, b3], max_candidates=3)

        # S3-1 was found by all 3 blockers -> MUST be rank 1
        self.assertEqual(merged[0], "S3-1")
        # S2-1 was found by 2 blockers -> MUST be rank 2
        self.assertEqual(merged[1], "S2-1")
        # Ceiling enforced to 3
        self.assertEqual(len(merged), 3)
        # No duplicates
        self.assertEqual(len(merged), len(set(merged)))

    def test_merge_filters_self_matches_and_invalid_ids(self):
        s1_id = "S1-123"
        b1 = ["S1-123", "S2-456", "INVALID-789", "S3-999"]
        merged = merge_candidates(s1_id, [b1], max_candidates=10)

        self.assertNotIn("S1-123", merged)
        self.assertNotIn("INVALID-789", merged)
        self.assertIn("S2-456", merged)
        self.assertIn("S3-999", merged)


class TestEvaluationAndExporter(unittest.TestCase):
    """Test Module D: Validation Tracker & Submission Exporter."""

    def test_evaluate_blocking_recall(self):
        gt = {
            "S1-1": {"S2-10", "S3-20"},
            "S1-2": {"S2-30"},
            "S1-3": set(),  # singleton
        }
        # Candidates found
        cand_dict = {
            "S1-1": ["S2-10", "S2-99"],  # found 1 of 2
            "S1-2": ["S2-30", "S3-88"],  # found 1 of 1
            "S1-3": [],                  # singleton
        }
        metrics = evaluate_blocking_recall(gt, cand_dict, target_count=100)
        # Total true pairs = 2 + 1 = 3. Found = 1 + 1 = 2.
        # Recall = 2 / 3 = 0.6666...
        self.assertAlmostEqual(metrics["blocking_recall"], 2 / 3, places=3)
        self.assertEqual(metrics["found_true_pairs"], 2)
        self.assertEqual(metrics["total_true_pairs"], 3)
        self.assertGreater(metrics["reduction_ratio"], 0.9)

    def test_export_candidates_format(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_file = os.path.join(tmpdir, "candidate_pairs.tsv")
            cand_dict = {
                "S1-001": ["S2-10", "S3-20"],
                "S1-002": ["S3-30"],
                "S1-003": [],
            }
            export_candidates(cand_dict, out_file, required_s1_ids=["S1-001", "S1-002", "S1-003"])

            self.assertTrue(os.path.isfile(out_file))
            with open(out_file, "r", encoding="utf-8") as f:
                lines = [line.rstrip("\n") for line in f]

            self.assertEqual(lines[0], "source1_entity_id\tcandidate_entity_ids")
            self.assertEqual(lines[1], "S1-001\tS2-10,S3-20")
            self.assertEqual(lines[2], "S1-002\tS3-30")
            self.assertEqual(lines[3], "S1-003\t")


if __name__ == "__main__":
    unittest.main()
