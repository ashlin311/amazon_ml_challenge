"""
tests/test_data_cleaning.py — Unit tests for Member 1 data cleaning & normalization.
"""

import os
import sys
import unittest
import pandas as pd

# Add repo root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from src.data_cleaning import (
    clean_name,
    clean_address,
    normalize_text,
    clean_dataframe,
    word2features,
    sent2features,
    sent2labels,
)


class TestDataCleaning(unittest.TestCase):
    """Test Member 1 text normalization and cleaning routines."""

    def test_clean_name_abbreviations(self):
        raw = "Apex Technologies International Incorporated"
        cleaned = clean_name(raw)
        self.assertEqual(cleaned, "apex tech intl inc")

    def test_clean_address_abbreviations(self):
        raw = "100 South Boulevard, Suite 500"
        cleaned = clean_address(raw)
        self.assertEqual(cleaned, "100 south blvd ste 500")

    def test_normalize_text_combined(self):
        norm = normalize_text("Acme Corporation", "456 Market Street")
        self.assertEqual(norm, "acme corp 456 market st")

    def test_normalize_text_empty_and_none(self):
        self.assertEqual(normalize_text("", None), "")
        self.assertEqual(normalize_text(None, "123 Main St"), "123 main st")
        self.assertEqual(normalize_text("Acme Corp", None), "acme corp")

    def test_clean_dataframe(self):
        data = {
            "entity_id": ["E-001", "E-002"],
            "business_name": ["Global Solutions Limited", "Beta LLC"],
            "business_address": ["42 Highway 1, Room 10", "77 Ocean Drive"],
            "country": ["us", "ca"],
        }
        df = pd.DataFrame(data)
        cleaned_df = clean_dataframe(df)

        self.assertIn("cleaned_business_name", cleaned_df.columns)
        self.assertIn("cleaned_business_address", cleaned_df.columns)
        self.assertIn("normalized_text", cleaned_df.columns)
        self.assertEqual(cleaned_df.loc[0, "country"], "US")
        self.assertEqual(cleaned_df.loc[0, "cleaned_business_name"], "global solns ltd")
        self.assertEqual(cleaned_df.loc[0, "cleaned_business_address"], "42 hwy 1 rm 10")


class TestCRFFeatureExtraction(unittest.TestCase):
    """Test Member 1 token-level feature extraction."""

    def test_word2features_single_token(self):
        sent = [("Amazon", "NNP", "B-ORG")]
        feat = word2features(sent, 0)
        self.assertEqual(feat["word.lower()"], "amazon")
        self.assertTrue(feat["word.istitle()"])
        self.assertTrue(feat.get("BOS"))
        self.assertTrue(feat.get("EOS"))

    def test_sent2features_and_labels(self):
        sent = [("San", "NNP", "B-LOC"), ("Francisco", "NNP", "I-LOC")]
        feats = sent2features(sent)
        labels = sent2labels(sent)

        self.assertEqual(len(feats), 2)
        self.assertEqual(labels, ["B-LOC", "I-LOC"])
        self.assertTrue(feats[0].get("BOS"))
        self.assertTrue(feats[1].get("EOS"))
        self.assertEqual(feats[1]["-1:word.lower()"], "san")


if __name__ == "__main__":
    unittest.main()
