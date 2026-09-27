"""
src/member1/data_cleaning.py — Data Normalization, Feature Engineering & Cleaning.

Responsibilities (Member 1):
  - Text normalization: Unicode NFKC, case folding, URL removal, punctuation cleaning.
  - Abbreviation standardization (business entity terms, address components).
  - DataFrame & TSV dataset cleaning and sanitation.
  - Token-level feature extraction for Named Entity Recognition (NER).
  - CRF (Conditional Random Field) model training, evaluation, and entity extraction.
"""

import argparse
import os
import re
import sys
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Common Abbreviations & Normalization Constants
# ---------------------------------------------------------------------------

ABBREVIATIONS: Dict[str, str] = {
    # Business Entity Types & Suffixes
    "limited": "ltd", "corporation": "corp", "incorporated": "inc",
    "company": "co", "societe anonyme": "sa", "aktiengesellschaft": "ag",
    "private": "pvt", "public limited company": "plc", "llc": "llc",
    "gmbh": "gmbh", "srl": "srl", "bv": "bv", "holding": "hldgs",
    "holdings": "hldgs", "enterprises": "ent", "international": "intl",
    "group": "grp", "services": "svcs", "solutions": "solns",
    "industries": "inds", "consulting": "consult", "technologies": "tech",
    "technology": "tech", "development": "dev", "management": "mgmt",

    # Street & Address Standards
    "street": "st", "avenue": "ave", "road": "rd", "boulevard": "blvd",
    "drive": "dr", "lane": "ln", "court": "ct", "place": "pl",
    "square": "sq", "suite": "ste", "apartment": "apt", "building": "bldg",
    "room": "rm", "floor": "fl", "department": "dept", "block": "blk",
    "circle": "cir", "parkway": "pkwy", "highway": "hwy", "route": "rte",
    "terrace": "ter", "crossing": "xing", "cross": "xing",
    "expressway": "expy", "expwy": "expy", "near": "nr", "opposite": "opp",
    "house no": "hno", "plot no": "plot", "flat no": "flat",
    "boulevard du": "blvd", "bd du": "blvd", "bd": "blvd", "rue": "rue",
}

_ABBREV_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(ABBREVIATIONS.keys(), key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)
_PUNCT_REGEX = re.compile(r"[^\w\s\u0900-\u0D7F\u0300-\u036F]", flags=re.UNICODE)
_WHITESPACE_REGEX = re.compile(r"\s+")
_URL_REGEX = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)


# ---------------------------------------------------------------------------
# Text Normalization & Cleaning Utilities
# ---------------------------------------------------------------------------

def clean_name(name_str: Optional[str]) -> str:
    """Clean business name by normalizing case, unicode, punctuation,
    and business entity suffixes while preserving alphanumeric tokens.
    """
    if not name_str or pd.isna(name_str):
        return ""

    text = str(name_str)
    text = unicodedata.normalize("NFKC", text).lower()
    text = _URL_REGEX.sub(" ", text)
    text = _ABBREV_REGEX.sub(lambda m: ABBREVIATIONS[m.group(0).lower()], text)
    text = _PUNCT_REGEX.sub(" ", text)
    return _WHITESPACE_REGEX.sub(" ", text).strip()


def clean_address(address_str: Optional[str]) -> str:
    """Clean business address by normalizing abbreviations (street, road, suite, etc.),
    removing noise punctuation while retaining exact digits and street numbers.
    """
    if not address_str or pd.isna(address_str):
        return ""

    text = str(address_str)
    text = unicodedata.normalize("NFKC", text).lower()
    text = _ABBREV_REGEX.sub(lambda m: ABBREVIATIONS[m.group(0).lower()], text)
    text = _PUNCT_REGEX.sub(" ", text)
    return _WHITESPACE_REGEX.sub(" ", text).strip()


def normalize_text(name_str: Optional[str], address_str: Optional[str]) -> str:
    """Normalize combined business name and address for entity resolution.

    Features:
    - Lowercase and normalize Unicode characters.
    - Strip noise punctuation while preserving exact alphanumeric tokens.
    - Standardize common business and address abbreviations.
    - Return clean, single-spaced matching representation.
    """
    c_name = clean_name(name_str)
    c_addr = clean_address(address_str)

    if c_name and c_addr:
        return f"{c_name} {c_addr}"
    elif c_name:
        return c_name
    elif c_addr:
        return c_addr
    return ""


def clean_dataframe(
    df: pd.DataFrame,
    name_col: str = "business_name",
    address_col: str = "business_address",
    country_col: str = "country",
    in_place: bool = False,
) -> pd.DataFrame:
    """Clean and standardize an entity DataFrame.

    Adds normalized text column and strips whitespace across string fields.
    """
    out_df = df if in_place else df.copy()

    if name_col in out_df.columns:
        out_df[name_col] = out_df[name_col].fillna("").astype(str).str.strip()
        out_df[f"cleaned_{name_col}"] = out_df[name_col].apply(clean_name)

    if address_col in out_df.columns:
        out_df[address_col] = out_df[address_col].fillna("").astype(str).str.strip()
        out_df[f"cleaned_{address_col}"] = out_df[address_col].apply(clean_address)

    if country_col in out_df.columns:
        out_df[country_col] = out_df[country_col].fillna("").astype(str).str.strip().str.upper()

    if name_col in out_df.columns and address_col in out_df.columns:
        out_df["normalized_text"] = [
            normalize_text(n, a) for n, a in zip(out_df[name_col], out_df[address_col])
        ]

    return out_df


def clean_source_tsv(input_path: str, output_path: str) -> pd.DataFrame:
    """Load, clean, and write back a TSV source file."""
    df = pd.read_csv(input_path, sep="\t", dtype=str)
    cleaned = clean_dataframe(df)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cleaned.to_csv(output_path, sep="\t", index=False)
    return cleaned


# ---------------------------------------------------------------------------
# Token Feature Extraction (CRF Feature Engineering from Member 1)
# ---------------------------------------------------------------------------

def word2features(sent: Sequence[Tuple[str, ...]], i: int) -> Dict[str, Any]:
    """Extract hand-engineered token features for CRF sequence tagging.

    Args:
        sent: List of tuples where each element contains at least (word, pos_tag).
        i: Current index in sentence.
    """
    word = str(sent[i][0])
    postag = str(sent[i][1]) if len(sent[i]) > 1 else "UNK"

    features: Dict[str, Any] = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "postag": postag,
        "postag[:2]": postag[:2],
    }

    # Features for previous token (Left Context)
    if i > 0:
        word1 = str(sent[i - 1][0])
        postag1 = str(sent[i - 1][1]) if len(sent[i - 1]) > 1 else "UNK"
        features.update({
            "-1:word.lower()": word1.lower(),
            "-1:word.istitle()": word1.istitle(),
            "-1:word.isupper()": word1.isupper(),
            "-1:postag": postag1,
            "-1:postag[:2]": postag1[:2],
        })
    else:
        features["BOS"] = True  # Beginning of Sentence

    # Features for next token (Right Context)
    if i < len(sent) - 1:
        word1 = str(sent[i + 1][0])
        postag1 = str(sent[i + 1][1]) if len(sent[i + 1]) > 1 else "UNK"
        features.update({
            "+1:word.lower()": word1.lower(),
            "+1:word.istitle()": word1.istitle(),
            "+1:word.isupper()": word1.isupper(),
            "+1:postag": postag1,
            "+1:postag[:2]": postag1[:2],
        })
    else:
        features["EOS"] = True  # End of Sentence

    return features


def sent2features(sent: Sequence[Tuple[str, ...]]) -> List[Dict[str, Any]]:
    """Convert an entire sentence of tokens into a list of feature dictionaries."""
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent: Sequence[Tuple[str, ...]]) -> List[str]:
    """Extract gold IOB/NER labels from an annotated sentence."""
    return [str(token[-1]) for token in sent]


# ---------------------------------------------------------------------------
# CRF Model Training, Evaluation, and Prediction
# ---------------------------------------------------------------------------

def train_crf_model(
    train_sents: Sequence[Sequence[Tuple[str, ...]]],
    c1: float = 0.1,
    c2: float = 0.1,
    max_iterations: int = 100,
    algorithm: str = "lbfgs",
    all_possible_transitions: bool = True,
):
    """Train a Conditional Random Field (CRF) model using sklearn_crfsuite.

    Args:
        train_sents: List of sentences, where each sentence is a sequence of (word, pos, label) tuples.
        c1: L1 regularization coefficient.
        c2: L2 regularization coefficient.
        max_iterations: Maximum iterations for L-BFGS optimizer.
        algorithm: Optimization algorithm.
        all_possible_transitions: Whether to learn all transitions.

    Returns:
        Fitted CRF model instance.
    """
    try:
        import sklearn_crfsuite
    except ImportError as e:
        raise ImportError(
            "sklearn-crfsuite is required for CRF model training. "
            "Please install it using: pip install sklearn-crfsuite"
        ) from e

    x_train = [sent2features(s) for s in train_sents]
    y_train = [sent2labels(s) for s in train_sents]

    crf = sklearn_crfsuite.CRF(
        algorithm=algorithm,
        c1=c1,
        c2=c2,
        max_iterations=max_iterations,
        all_possible_transitions=all_possible_transitions,
    )
    crf.fit(x_train, y_train)
    return crf


def evaluate_crf_model(
    crf: Any,
    test_sents: Sequence[Sequence[Tuple[str, ...]]],
    exclude_labels: Optional[Sequence[str]] = ("O",),
) -> str:
    """Evaluate a trained CRF model and return the classification report string."""
    try:
        from sklearn_crfsuite import metrics
    except ImportError as e:
        raise ImportError("sklearn-crfsuite is required for CRF evaluation.") from e

    x_test = [sent2features(s) for s in test_sents]
    y_test = [sent2labels(s) for s in test_sents]

    y_pred = crf.predict(x_test)
    exclude_set = set(exclude_labels or [])
    labels = sorted(
        [label for label in crf.classes_ if label not in exclude_set],
        key=lambda name: (name[1:], name[0]),
    )

    report = metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=4)
    return report


def extract_entities_from_tokens(
    crf: Any,
    tokens: Sequence[str],
    postags: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str]]:
    """Predict entity tags for an unannotated token list.

    Returns list of (token, predicted_label) tuples.
    """
    if postags is None:
        postags = ["NOUN"] * len(tokens)
    sent = list(zip(tokens, postags))
    features = sent2features(sent)
    preds = crf.predict_single(features)
    return list(zip(tokens, preds))


# ---------------------------------------------------------------------------
# CLI Entrypoint
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Member 1 — Data Cleaning, Normalization & CRF Feature Pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["normalize", "clean-file", "crf-demo"],
        default="normalize",
        help="Action to perform: normalize test text, clean a TSV file, or run the CoNLL CRF experiment.",
    )
    parser.add_argument("--input-file", help="Path to input TSV file to clean.")
    parser.add_argument("--output-file", help="Path to output cleaned TSV file.")
    parser.add_argument("--name", default="Acme International Solutions Ltd.", help="Sample name to clean.")
    parser.add_argument("--address", default="123 North Main Street, Suite 400", help="Sample address to clean.")

    args = parser.parse_args(argv)

    if args.mode == "normalize":
        cleaned_n = clean_name(args.name)
        cleaned_a = clean_address(args.address)
        norm = normalize_text(args.name, args.address)
        print("Original Name:    ", args.name)
        print("Cleaned Name:     ", cleaned_n)
        print("Original Address: ", args.address)
        print("Cleaned Address:  ", cleaned_a)
        print("Normalized Text:  ", norm)
        return 0

    elif args.mode == "clean-file":
        if not args.input_file or not args.output_file:
            print("Error: --input-file and --output-file are required for clean-file mode.")
            return 1
        clean_source_tsv(args.input_file, args.output_file)
        print(f"Cleaned {args.input_file} -> {args.output_file}")
        return 0

    elif args.mode == "crf-demo":
        try:
            import nltk
            print("Downloading CoNLL-2002 dataset...")
            nltk.download("conll2002", quiet=True)
            train_sents = list(nltk.corpus.conll2002.iob_sents("esp.train"))
            test_sents = list(nltk.corpus.conll2002.iob_sents("esp.testb"))
            print(f"Loaded {len(train_sents)} train and {len(test_sents)} test sentences.")
            print("Training CRF model...")
            crf = train_crf_model(train_sents)
            print("Evaluating CRF model...")
            report = evaluate_crf_model(crf, test_sents)
            print("\n--- Classification Report ---")
            print(report)
            return 0
        except Exception as e:
            print(f"CRF demo failed: {e}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
