"""
Full evaluation of all three sentiment models on the SST-2 validation set.

Generates confusion matrices, classification reports, and side-by-side bar
charts saved to ``docs/``. Prints a final comparison table at the end.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from src.utils import get_logger, load_config, set_seed, timer

# ── Constants ────────────────────────────────────────────────────────────────

CLASS_NAMES = ["NEGATIVE", "POSITIVE"]
FIGURE_DPI = 150
PLOT_STYLE = "seaborn-v0_8-whitegrid"

VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"
LR_FILENAME = "logistic_regression.joblib"
SVM_FILENAME = "linear_svc.joblib"

log = get_logger(__name__)


# ── Text helpers ──────────────────────────────────────────────────────────────


def _clean_text(text: str) -> str:
    """Lowercase and remove non-alphanumeric characters for baseline models.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Data loading ─────────────────────────────────────────────────────────────


@timer
def load_validation_set() -> tuple[list[str], list[int]]:
    """Return the SST-2 validation split as cleaned texts and integer labels.

    Returns:
        A 2-tuple ``(texts, labels)``.
    """
    log.info("Loading SST-2 validation split ...")
    dataset = load_dataset("glue", "sst2")
    val = dataset["validation"]
    texts = [_clean_text(s) for s in val["sentence"]]
    labels: list[int] = list(val["label"])
    log.info("Loaded %d validation samples.", len(texts))
    return texts, labels


# ── Model loaders ─────────────────────────────────────────────────────────────


def load_baseline_models(baseline_dir: Path) -> tuple[Any, Any, Any]:
    """Load the TF-IDF vectoriser, Logistic Regression, and LinearSVC.

    Args:
        baseline_dir: Directory containing the joblib artefacts.

    Returns:
        A 3-tuple ``(vectorizer, lr_model, svm_model)``.

    Raises:
        FileNotFoundError: If any expected artefact is missing.
    """
    for filename in (VECTORIZER_FILENAME, LR_FILENAME, SVM_FILENAME):
        path = baseline_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Baseline artefact not found: {path}\n"
                "Run: python -m src.train_baselines"
            )

    log.info("Loading baseline artefacts from %s ...", baseline_dir)
    vectorizer = joblib.load(baseline_dir / VECTORIZER_FILENAME)
    lr_model = joblib.load(baseline_dir / LR_FILENAME)
    svm_model = joblib.load(baseline_dir / SVM_FILENAME)
    return vectorizer, lr_model, svm_model


def load_distilbert_pipeline(saved_dir: Path) -> Any:
    """Load the fine-tuned DistilBERT as a HuggingFace text-classification pipeline.

    Args:
        saved_dir: Directory produced by ``train_distilbert.py``.

    Returns:
        A HuggingFace ``pipeline`` object.

    Raises:
        FileNotFoundError: If ``config.json`` is absent from *saved_dir*.
    """
    config_path = saved_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"DistilBERT model not found at {saved_dir}\n"
            "Run: python -m src.train_distilbert"
        )

    from transformers import pipeline as hf_pipeline

    log.info("Loading DistilBERT pipeline from %s ...", saved_dir)
    pipe = hf_pipeline(
        "text-classification",
        model=str(saved_dir),
        tokenizer=str(saved_dir),
        device=-1,
        truncation=True,
        max_length=128,
    )
    return pipe


# ── Inference ─────────────────────────────────────────────────────────────────


def predict_baseline(
    model: Any,
    vectorizer: Any,
    texts: list[str],
) -> tuple[list[int], float]:
    """Vectorise *texts* and predict with *model*, returning preds and timing.

    Args:
        model: Fitted sklearn classifier.
        vectorizer: Fitted TF-IDF vectoriser.
        texts: Pre-cleaned text strings.

    Returns:
        A 2-tuple ``(predictions, inference_seconds)``.
    """
    X = vectorizer.transform(texts)
    t0 = time.perf_counter()
    preds = list(model.predict(X))
    elapsed = time.perf_counter() - t0
    return preds, elapsed


def predict_distilbert(
    pipe: Any,
    texts: list[str],
    batch_size: int = 64,
) -> tuple[list[int], float]:
    """Run DistilBERT inference over *texts* in batches.

    Args:
        pipe: HuggingFace text-classification pipeline.
        texts: Raw text strings (DistilBERT handles its own tokenisation).
        batch_size: Number of samples per forward pass.

    Returns:
        A 2-tuple ``(predictions, inference_seconds)``.
    """
    from tqdm import tqdm

    raw_texts = [t.strip() if t.strip() else " " for t in texts]

    log.info("Running DistilBERT inference on %d samples (batch=%d) ...", len(raw_texts), batch_size)
    preds: list[int] = []
    t0 = time.perf_counter()

    for i in tqdm(range(0, len(raw_texts), batch_size), desc="DistilBERT inference"):
        batch = raw_texts[i : i + batch_size]
        results = pipe(batch)
        for r in results:
            raw = r["label"].upper()
            preds.append(1 if raw in ("LABEL_1", "POSITIVE", "1") else 0)

    elapsed = time.perf_counter() - t0
    return preds, elapsed


# ── Plotting ──────────────────────────────────────────────────────────────────


def plot_confusion_matrices(
    all_preds: dict[str, list[int]],
    y_true: list[int],
    docs_dir: Path,
) -> None:
    """Save a figure with one confusion matrix per model.

    Args:
        all_preds: Dict mapping model name to list of integer predictions.
        y_true: Ground-truth integer labels.
        docs_dir: Output directory for the PNG file.
    """
    n = len(all_preds)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, preds) in zip(axes, all_preds.items()):
        cm = confusion_matrix(y_true, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(name, fontsize=13, fontweight="bold")

    fig.suptitle("Confusion Matrices — SST-2 Validation Set", fontsize=15, y=1.02)
    plt.tight_layout()
    out_path = docs_dir / "confusion_matrices.png"
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Confusion matrices saved to %s", out_path)


def plot_metric_comparison(
    summary: list[dict[str, Any]],
    docs_dir: Path,
) -> None:
    """Save a side-by-side bar chart comparing accuracy and F1 across models.

    Args:
        summary: List of metric dicts (one per model) with keys
            ``model``, ``accuracy``, ``f1``.
        docs_dir: Output directory for the PNG file.
    """
    plt.style.use(PLOT_STYLE)
    df = pd.DataFrame(summary).set_index("model")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric in zip(axes, ["accuracy", "f1"]):
        colours = sns.color_palette("muted", len(df))
        bars = ax.bar(df.index, df[metric], color=colours, edgecolor="white", linewidth=0.8)
        ax.set_title(metric.capitalize(), fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", rotation=15)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    fig.suptitle("Model Comparison — SST-2 Validation Set", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out_path = docs_dir / "metric_comparison.png"
    plt.savefig(out_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    log.info("Metric comparison chart saved to %s", out_path)


# ── Results table ─────────────────────────────────────────────────────────────


def print_comparison_table(summary: list[dict[str, Any]]) -> None:
    """Print the final model comparison table to stdout.

    Args:
        summary: List of metric dicts with keys ``model``, ``accuracy``,
            ``f1``, ``inference_s``.
    """
    df = pd.DataFrame(summary).set_index("model")
    separator = "-" * 72
    print("\n" + separator)
    print("  FINAL MODEL COMPARISON — SST-2 VALIDATION SET")
    print(separator)
    print(df.to_string())
    print(separator + "\n")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Run the complete evaluation pipeline for all three models."""
    cfg = load_config()
    set_seed(cfg["seed"])

    baseline_dir = Path(cfg["baseline_dir"])
    saved_dir = Path(cfg["saved_dir"])
    docs_dir = Path(cfg["docs_dir"])
    docs_dir.mkdir(parents=True, exist_ok=True)

    texts, y_true = load_validation_set()

    # ── Load models ──────────────────────────────────────────────────────────
    errors: list[str] = []
    vectorizer = lr_model = svm_model = distilbert_pipe = None

    try:
        vectorizer, lr_model, svm_model = load_baseline_models(baseline_dir)
    except FileNotFoundError as exc:
        log.warning("Baseline models not available: %s", exc)
        errors.append("baseline")

    try:
        distilbert_pipe = load_distilbert_pipeline(saved_dir)
    except FileNotFoundError as exc:
        log.warning("DistilBERT not available: %s", exc)
        errors.append("distilbert")

    if len(errors) == 2:
        raise RuntimeError("No models found. Train at least one model first.")

    # ── Inference ────────────────────────────────────────────────────────────
    all_preds: dict[str, list[int]] = {}
    summary: list[dict[str, Any]] = []

    if lr_model is not None and vectorizer is not None:
        lr_preds, lr_time = predict_baseline(lr_model, vectorizer, texts)
        all_preds["Logistic Regression"] = lr_preds
        summary.append({
            "model": "Logistic Regression",
            "accuracy": round(accuracy_score(y_true, lr_preds), 4),
            "f1": round(f1_score(y_true, lr_preds, average="macro", zero_division=0), 4),
            "inference_s": round(lr_time, 3),
        })
        log.info("\n%s", classification_report(y_true, lr_preds, target_names=CLASS_NAMES))

    if svm_model is not None and vectorizer is not None:
        svm_preds, svm_time = predict_baseline(svm_model, vectorizer, texts)
        all_preds["LinearSVC"] = svm_preds
        summary.append({
            "model": "LinearSVC",
            "accuracy": round(accuracy_score(y_true, svm_preds), 4),
            "f1": round(f1_score(y_true, svm_preds, average="macro", zero_division=0), 4),
            "inference_s": round(svm_time, 3),
        })
        log.info("\n%s", classification_report(y_true, svm_preds, target_names=CLASS_NAMES))

    if distilbert_pipe is not None:
        db_preds, db_time = predict_distilbert(distilbert_pipe, texts)
        all_preds["DistilBERT"] = db_preds
        summary.append({
            "model": "DistilBERT",
            "accuracy": round(accuracy_score(y_true, db_preds), 4),
            "f1": round(f1_score(y_true, db_preds, average="macro", zero_division=0), 4),
            "inference_s": round(db_time, 3),
        })
        log.info("\n%s", classification_report(y_true, db_preds, target_names=CLASS_NAMES))

    # ── Plots ────────────────────────────────────────────────────────────────
    plot_confusion_matrices(all_preds, y_true, docs_dir)
    plot_metric_comparison(summary, docs_dir)

    # ── Summary ──────────────────────────────────────────────────────────────
    print_comparison_table(summary)
    log.info("Evaluation complete. Plots saved to %s", docs_dir)


if __name__ == "__main__":
    main()
