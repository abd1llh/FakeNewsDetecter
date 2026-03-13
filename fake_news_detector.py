"""
Fake News Detector using Logistic Regression.

This script trains a logistic regression classifier on news article text to
distinguish real news from fake news.  It expects a CSV dataset with at least
two columns:

    text  – the article body (or headline + body concatenated)
    label – 0 for real news, 1 for fake news  (or "REAL" / "FAKE" strings)

A small built-in sample is included so the script can be run immediately
without an external dataset.  Pass a CSV file path as the first command-line
argument to use your own data instead.

Usage
-----
    # Run on the built-in sample data
    python fake_news_detector.py

    # Run on your own CSV dataset
    python fake_news_detector.py path/to/dataset.csv
"""

import sys
import re
import string

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ---------------------------------------------------------------------------
# Built-in sample data (used when no external dataset is supplied)
# ---------------------------------------------------------------------------
SAMPLE_DATA = [
    # (text, label)  label: 0 = real, 1 = fake
    ("Scientists confirm new vaccine is safe and effective after large clinical trial", 0),
    ("Government releases annual budget report showing economic growth", 0),
    ("Local council approves new public transport route for the city", 0),
    ("University study finds regular exercise improves mental health outcomes", 0),
    ("Central bank raises interest rates to curb inflation", 0),
    ("Health authorities report steady decline in seasonal flu cases", 0),
    ("New renewable energy plant begins operations in the region", 0),
    ("Court upholds ruling on data privacy regulations", 0),
    ("Weather service issues advisory ahead of expected storm", 0),
    ("Tech company releases security patch for recently discovered vulnerability", 0),
    ("SHOCKING: Secret government mind-control program exposed by whistleblower!", 1),
    ("Miracle cure discovered that doctors don't want you to know about!", 1),
    ("Alien spacecraft lands in major city – media blackout ordered!", 1),
    ("Drinking bleach cures all diseases according to suppressed research", 1),
    ("Famous celebrity secretly runs underground criminal network", 1),
    ("The moon landing was staged in a Hollywood studio, new evidence proves", 1),
    ("Microchips hidden in vaccines to track the entire population", 1),
    ("5G towers spreading disease – government cover-up confirmed", 1),
    ("Billionaire plotting to replace world leaders with clones", 1),
    ("Ancient prophecy predicts end of world next Tuesday", 1),
]


# ---------------------------------------------------------------------------
# Text pre-processing
# ---------------------------------------------------------------------------

def preprocess_text(text: str) -> str:
    """Lower-case, remove punctuation/numbers, and strip extra whitespace."""
    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_sample_data() -> pd.DataFrame:
    """Return the built-in sample dataset as a DataFrame."""
    texts, labels = zip(*SAMPLE_DATA)
    return pd.DataFrame({"text": texts, "label": labels})


def load_csv_data(filepath: str) -> pd.DataFrame:
    """
    Load a CSV file.

    The file must contain a 'text' column and a 'label' column.
    Labels may be integers (0/1) or strings ('REAL'/'FAKE', case-insensitive).
    """
    df = pd.read_csv(filepath)

    required_columns = {"text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    # Normalise string labels to integers
    if df["label"].dtype == object:
        label_map = {"real": 0, "fake": 1}
        df["label"] = df["label"].str.lower().str.strip().map(label_map)
        if df["label"].isnull().any():
            raise ValueError(
                "Unrecognised label values found. Expected 'REAL'/'FAKE' or 0/1."
            )

    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    return df


# ---------------------------------------------------------------------------
# Model training & evaluation
# ---------------------------------------------------------------------------

def build_and_train(X_train, y_train, max_features: int = 10_000):
    """Fit a TF-IDF vectorizer and a logistic regression classifier."""
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)

    return vectorizer, model


def evaluate(vectorizer, model, X_test, y_test) -> dict:
    """Return evaluation metrics for the fitted model on the test set."""
    X_test_tfidf = vectorizer.transform(X_test)
    y_pred = model.predict(X_test_tfidf)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(
            y_test, y_pred, target_names=["Real", "Fake"]
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Prediction helper
# ---------------------------------------------------------------------------

def predict(vectorizer, model, texts: list) -> list:
    """
    Predict whether each text in *texts* is real (0) or fake (1).

    Returns a list of (label_int, label_str) tuples.
    """
    cleaned = [preprocess_text(t) for t in texts]
    X = vectorizer.transform(cleaned)
    predictions = model.predict(X)
    return [(int(p), "Fake" if p == 1 else "Real") for p in predictions]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load data
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
        print(f"Loading dataset from: {filepath}")
        df = load_csv_data(filepath)
    else:
        print("No dataset path provided – using built-in sample data.")
        df = load_sample_data()

    print(f"Dataset size: {len(df)} articles  "
          f"(real: {(df['label'] == 0).sum()}, fake: {(df['label'] == 1).sum()})\n")

    # 2. Pre-process
    df["text"] = df["text"].fillna("").apply(preprocess_text)

    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    # 4. Train
    print("Training logistic regression model ...")
    vectorizer, model = build_and_train(X_train, y_train)

    # 5. Evaluate
    results = evaluate(vectorizer, model, X_test, y_test)
    print(f"Test Accuracy : {results['accuracy']:.4f}\n")
    print("Classification Report:")
    print(results["classification_report"])
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print("            Predicted Real  Predicted Fake")
    cm = results["confusion_matrix"]
    print(f"Actual Real       {cm[0][0]:<14}  {cm[0][1]}")
    print(f"Actual Fake       {cm[1][0]:<14}  {cm[1][1]}\n")

    # 6. Example predictions on new headlines
    new_headlines = [
        "Scientists develop new battery technology that doubles electric car range",
        "BREAKING: Government admits chemtrails have been poisoning water supply for decades",
    ]
    print("Example predictions on new headlines:")
    print("-" * 60)
    for text, (label_int, label_str) in zip(new_headlines, predict(vectorizer, model, new_headlines)):
        print(f"  [{label_str:4s}]  {text}")


if __name__ == "__main__":
    main()
