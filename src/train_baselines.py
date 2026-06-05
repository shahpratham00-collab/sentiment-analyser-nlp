"""
Baseline sentiment classifier training: Logistic Regression and LinearSVC.

Downloads SST-2 from HuggingFace, applies TF-IDF vectorisation, trains both
models, evaluates on the validation split, and saves all artefacts to
``models/baseline/``.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.svm import LinearSVC

from src.utils import get_logger, load_config, set_seed, timer

# ── Constants ────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}
VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"
LR_FILENAME = "logistic_regression.joblib"
SVM_FILENAME = "linear_svc.joblib"

log = get_logger(__name__)


# ── Text preprocessing ───────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Lowercase and strip non-alphanumeric characters from *text*.

    Args:
        text: Raw input string.

    Returns:
        Cleaned, lowercased string suitable for TF-IDF vectorisation.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_split(examples: list[str]) -> list[str]:
    """Apply :func:`clean_text` to every element of *examples*.

    Args:
        examples: List of raw sentence strings.

    Returns:
        List of cleaned strings.
    """
    return [clean_text(s) for s in examples]


# ── Data loading ─────────────────────────────────────────────────────────────


@timer
def load_sst2() -> tuple[list[str], list[int], list[str], list[int]]:
    """Download and return the SST-2 train and validation splits.

    Returns:
        A 4-tuple ``(train_texts, train_labels, val_texts, val_labels)``
        where texts are cleaned strings and labels are 0/1 integers.
    """
    log.info("Loading SST-2 from HuggingFace datasets ...")
    dataset = load_dataset("glue", "sst2")

    train_texts = preprocess_split(dataset["train"]["sentence"])
    train_labels: list[int] = list(dataset["train"]["label"])

    val_texts = preprocess_split(dataset["validation"]["sentence"])
    val_labels: list[int] = list(dataset["validation"]["label"])

    log.info(
        "SST-2 loaded — train: %d samples | val: %d samples",
        len(train_texts),
        len(val_texts),
    )
    return train_texts, train_labels, val_texts, val_labels


# ── Vectorisation ────────────────────────────────────────────────────────────


@timer
def build_tfidf(
    train_texts: list[str],
    val_texts: list[str],
    max_features: int,
    ngram_range: tuple[int, int],
) -> tuple[Any, Any, TfidfVectorizer]:
    """Fit a TF-IDF vectoriser on training data and transform both splits.

    Args:
        train_texts: Pre-cleaned training sentences.
        val_texts: Pre-cleaned validation sentences.
        max_features: Vocabulary size cap.
        ngram_range: ``(min_n, max_n)`` for n-gram extraction.

    Returns:
        A 3-tuple ``(X_train, X_val, vectorizer)`` where ``X_train`` and
        ``X_val`` are sparse matrices.
    """
    log.info(
        "Fitting TF-IDF vectoriser (max_features=%d, ngram_range=%s) ...",
        max_features,
        ngram_range,
    )
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\w{1,}",
        min_df=2,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    log.info("TF-IDF matrix shapes — train: %s | val: %s", X_train.shape, X_val.shape)
    return X_train, X_val, vectorizer


# ── Model training ────────────────────────────────────────────────────────────


@timer
def train_logistic_regression(X_train: Any, y_train: list[int]) -> LogisticRegression:
    """Fit a Logistic Regression classifier.

    Args:
        X_train: Sparse TF-IDF feature matrix for training.
        y_train: Integer class labels.

    Returns:
        Fitted :class:`sklearn.linear_model.LogisticRegression` instance.
    """
    log.info("Training Logistic Regression ...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    log.info("Logistic Regression training complete.")
    return model


@timer
def train_linear_svc(X_train: Any, y_train: list[int]) -> LinearSVC:
    """Fit a LinearSVC classifier.

    Args:
        X_train: Sparse TF-IDF feature matrix for training.
        y_train: Integer class labels.

    Returns:
        Fitted :class:`sklearn.svm.LinearSVC` instance.
    """
    log.info("Training LinearSVC ...")
    model = LinearSVC(
        class_weight="balanced",
        max_iter=2000,
        C=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)
    log.info("LinearSVC training complete.")
    return model


# ── Evaluation ───────────────────────────────────────────────────────────────


def evaluate_model(
    model: Any,
    X_val: Any,
    y_val: list[int],
    model_name: str,
) -> dict[str, Any]:
    """Compute classification metrics and log a full report.

    Args:
        model: Fitted sklearn classifier.
        X_val: Validation feature matrix.
        y_val: True validation labels.
        model_name: Display name used in log output.

    Returns:
        Dict with keys ``model``, ``accuracy``, ``precision``, ``recall``,
        ``f1``, ``inference_time_s``.
    """
    log.info("Evaluating %s ...", model_name)

    t0 = time.perf_counter()
    preds = model.predict(X_val)
    inference_time = time.perf_counter() - t0

    acc = accuracy_score(y_val, preds)
    prec = precision_score(y_val, preds, average="binary", zero_division=0)
    rec = recall_score(y_val, preds, average="binary", zero_division=0)
    f1 = f1_score(y_val, preds, average="binary", zero_division=0)

    log.info(
        "%s | Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | Inference: %.3fs",
        model_name,
        acc,
        prec,
        rec,
        f1,
        inference_time,
    )
    log.info(
        "\n%s",
        classification_report(y_val, preds, target_names=["NEGATIVE", "POSITIVE"]),
    )

    return {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "inference_time_s": round(inference_time, 4),
    }


# ── Persistence ───────────────────────────────────────────────────────────────


def save_artefacts(
    vectorizer: TfidfVectorizer,
    lr_model: LogisticRegression,
    svm_model: LinearSVC,
    output_dir: Path,
) -> None:
    """Persist all trained artefacts to *output_dir*.

    Args:
        vectorizer: Fitted TF-IDF vectoriser.
        lr_model: Trained Logistic Regression model.
        svm_model: Trained LinearSVC model.
        output_dir: Directory to write joblib files into.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, output_dir / VECTORIZER_FILENAME)
    joblib.dump(lr_model, output_dir / LR_FILENAME)
    joblib.dump(svm_model, output_dir / SVM_FILENAME)

    log.info("Artefacts saved to %s", output_dir)


# ── Results table ─────────────────────────────────────────────────────────────


def print_results_table(results: list[dict[str, Any]]) -> None:
    """Print a formatted comparison table for all evaluated models.

    Args:
        results: List of metric dicts returned by :func:`evaluate_model`.
    """
    df = pd.DataFrame(results).set_index("model")
    separator = "-" * 72
    print("\n" + separator)
    print("  BASELINE MODEL COMPARISON — SST-2 VALIDATION SET")
    print(separator)
    print(df.to_string())
    print(separator + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Run the full baseline training pipeline end-to-end."""
    cfg = load_config()
    set_seed(cfg["seed"])

    baseline_dir = Path(cfg["baseline_dir"])
    ngram: tuple[int, int] = tuple(cfg["tfidf_ngram_range"])  # type: ignore[assignment]

    train_texts, train_labels, val_texts, val_labels = load_sst2()

    X_train, X_val, vectorizer = build_tfidf(
        train_texts,
        val_texts,
        max_features=cfg["tfidf_max_features"],
        ngram_range=ngram,
    )

    lr_model = train_logistic_regression(X_train, train_labels)
    svm_model = train_linear_svc(X_train, train_labels)

    results = [
        evaluate_model(lr_model, X_val, val_labels, "Logistic Regression"),
        evaluate_model(svm_model, X_val, val_labels, "LinearSVC"),
    ]

    save_artefacts(vectorizer, lr_model, svm_model, baseline_dir)
    print_results_table(results)
    log.info("Baseline training pipeline complete.")


if __name__ == "__main__":
    main()
