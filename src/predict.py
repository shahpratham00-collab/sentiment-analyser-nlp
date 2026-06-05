"""
Inference classes for all three trained sentiment models.

Provides:
- ``DistilBERTPredictor`` — wraps the fine-tuned HuggingFace model.
- ``BaselinePredictor``   — wraps the sklearn TF-IDF + classifier pipeline.
- ``ModelComparator``     — runs all three models and returns a unified result.
"""

from __future__ import annotations

import re
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.utils import get_logger, load_config

# ── Constants ────────────────────────────────────────────────────────────────

LABEL_MAP = {0: "NEGATIVE", 1: "POSITIVE"}

VECTORIZER_FILENAME = "tfidf_vectorizer.joblib"
LR_FILENAME = "logistic_regression.joblib"
SVM_FILENAME = "linear_svc.joblib"

log = get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _clean_text(text: str) -> str:
    """Lowercase and strip non-alphanumeric characters for baseline models.

    Args:
        text: Raw input string.

    Returns:
        Cleaned string.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _require_file(path: Path) -> None:
    """Raise ``FileNotFoundError`` with a clear message if *path* is absent.

    Args:
        path: File path to verify.

    Raises:
        FileNotFoundError: When *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run the training script first:\n"
            "  python -m src.train_baselines   # for baseline models\n"
            "  python -m src.train_distilbert  # for DistilBERT"
        )


# ── Abstract base ─────────────────────────────────────────────────────────────


class PredictorBase(ABC):
    """Abstract interface that every predictor must implement."""

    @abstractmethod
    def predict(self, text: str) -> dict[str, Any]:
        """Run inference on *text* and return a result dict.

        Args:
            text: Raw input sentence.

        Returns:
            Dict with keys:
            - ``label``      : ``"POSITIVE"`` or ``"NEGATIVE"``
            - ``confidence`` : float in ``[0, 1]``
            - ``inference_ms``: wall-clock milliseconds for the prediction
        """

    @abstractmethod
    def model_name(self) -> str:
        """Return a human-readable name for this predictor."""


# ── DistilBERT predictor ──────────────────────────────────────────────────────


class DistilBERTPredictor(PredictorBase):
    """Sentiment predictor backed by the fine-tuned DistilBERT model.

    Args:
        model_dir: Path to the directory containing the saved model and
            tokeniser (output of ``train_distilbert.py``).

    Raises:
        FileNotFoundError: If *model_dir* does not contain expected files.
        ImportError: If ``transformers`` is not installed.
    """

    def __init__(self, model_dir: str | Path | None = None) -> None:
        cfg = load_config()
        self._model_dir = Path(model_dir or cfg["saved_dir"])

        config_file = self._model_dir / "config.json"
        _require_file(config_file)

        log.info("Loading DistilBERT from %s ...", self._model_dir)
        try:
            from transformers import pipeline as hf_pipeline

            self._pipe = hf_pipeline(
                "text-classification",
                model=str(self._model_dir),
                tokenizer=str(self._model_dir),
                device=-1,
                truncation=True,
                max_length=128,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to load DistilBERT model: {exc}") from exc

        log.info("DistilBERT predictor ready.")

    def model_name(self) -> str:
        """Return the display name for this predictor."""
        return "DistilBERT"

    def predict(self, text: str) -> dict[str, Any]:
        """Run DistilBERT inference on *text*.

        Args:
            text: Input sentence (raw, unprocessed).

        Returns:
            Dict with ``label``, ``confidence``, and ``inference_ms``.

        Raises:
            ValueError: If *text* is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        t0 = time.perf_counter()
        result = self._pipe(text.strip())[0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        raw_label: str = result["label"].upper()
        # Normalise labels that may differ between model checkpoints
        if raw_label in ("LABEL_1", "POSITIVE", "1"):
            label = "POSITIVE"
        else:
            label = "NEGATIVE"

        return {
            "label": label,
            "confidence": round(float(result["score"]), 4),
            "inference_ms": round(elapsed_ms, 2),
        }


# ── Baseline predictor ────────────────────────────────────────────────────────


class BaselinePredictor(PredictorBase):
    """Sentiment predictor using a sklearn TF-IDF + classifier pipeline.

    Args:
        classifier: ``"lr"`` for Logistic Regression, ``"svm"`` for LinearSVC.
        baseline_dir: Directory containing the joblib artefacts.

    Raises:
        ValueError: If *classifier* is not ``"lr"`` or ``"svm"``.
        FileNotFoundError: If any required artefact is missing.
    """

    _CLASSIFIER_FILES = {"lr": LR_FILENAME, "svm": SVM_FILENAME}
    _DISPLAY_NAMES = {"lr": "Logistic Regression", "svm": "LinearSVC"}

    def __init__(
        self,
        classifier: str = "lr",
        baseline_dir: str | Path | None = None,
    ) -> None:
        if classifier not in self._CLASSIFIER_FILES:
            raise ValueError(f"classifier must be 'lr' or 'svm', got '{classifier}'.")

        cfg = load_config()
        self._baseline_dir = Path(baseline_dir or cfg["baseline_dir"])
        self._classifier_key = classifier

        import joblib

        vec_path = self._baseline_dir / VECTORIZER_FILENAME
        clf_path = self._baseline_dir / self._CLASSIFIER_FILES[classifier]
        _require_file(vec_path)
        _require_file(clf_path)

        log.info("Loading %s artefacts from %s ...", self._DISPLAY_NAMES[classifier], self._baseline_dir)
        self._vectorizer = joblib.load(vec_path)
        self._classifier = joblib.load(clf_path)
        log.info("%s predictor ready.", self._DISPLAY_NAMES[classifier])

    def model_name(self) -> str:
        """Return the display name for this predictor."""
        return self._DISPLAY_NAMES[self._classifier_key]

    def predict(self, text: str) -> dict[str, Any]:
        """Vectorise *text* and classify with the loaded sklearn model.

        Args:
            text: Input sentence (raw or lightly preprocessed).

        Returns:
            Dict with ``label``, ``confidence``, and ``inference_ms``.

        Raises:
            ValueError: If *text* is empty or whitespace-only.
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        cleaned = _clean_text(text)
        t0 = time.perf_counter()
        X = self._vectorizer.transform([cleaned])

        if hasattr(self._classifier, "predict_proba"):
            proba = self._classifier.predict_proba(X)[0]
            pred_idx = int(proba.argmax())
            confidence = float(proba[pred_idx])
        else:
            # LinearSVC uses decision_function; convert to a pseudo-probability
            decision = self._classifier.decision_function(X)[0]
            pred_idx = int(self._classifier.predict(X)[0])
            # Sigmoid of the decision value as a confidence proxy
            confidence = float(1 / (1 + pow(2.718281828, -abs(decision))))

        elapsed_ms = (time.perf_counter() - t0) * 1000
        label = LABEL_MAP[pred_idx]

        return {
            "label": label,
            "confidence": round(confidence, 4),
            "inference_ms": round(elapsed_ms, 2),
        }


# ── Model comparator ──────────────────────────────────────────────────────────


class ModelComparator:
    """Run all three models on the same input and return a unified comparison.

    Lazily loads each predictor on first use, and skips any that cannot be
    loaded (e.g. model files not yet trained) with a warning.

    Args:
        baseline_dir: Optional override for the baseline artefact directory.
        saved_dir: Optional override for the DistilBERT model directory.
    """

    def __init__(
        self,
        baseline_dir: str | Path | None = None,
        saved_dir: str | Path | None = None,
    ) -> None:
        self._baseline_dir = baseline_dir
        self._saved_dir = saved_dir
        self._predictors: list[PredictorBase] = []
        self._loaded = False

    def _load_predictors(self) -> None:
        """Attempt to load all three predictors, logging failures gracefully."""
        candidates: list[tuple[str, Any]] = [
            ("DistilBERT", lambda: DistilBERTPredictor(self._saved_dir)),
            ("Logistic Regression", lambda: BaselinePredictor("lr", self._baseline_dir)),
            ("LinearSVC", lambda: BaselinePredictor("svm", self._baseline_dir)),
        ]
        for name, factory in candidates:
            try:
                self._predictors.append(factory())
            except (FileNotFoundError, RuntimeError) as exc:
                log.warning("Skipping %s — %s", name, exc)

        if not self._predictors:
            raise RuntimeError(
                "No models could be loaded. Train at least one model before running comparison."
            )
        self._loaded = True

    def compare(self, text: str) -> dict[str, Any]:
        """Run inference with every loaded model and aggregate results.

        Args:
            text: Raw input sentence.

        Returns:
            Dict with:
            - ``input_text``  : The original input.
            - ``results``     : List of per-model result dicts (each has
              ``model``, ``label``, ``confidence``, ``inference_ms``).
            - ``consensus``   : Majority-vote label across all models.

        Raises:
            ValueError: If *text* is empty.
            RuntimeError: If no models are available.
        """
        if not text or not text.strip():
            raise ValueError("Input text must not be empty.")

        if not self._loaded:
            self._load_predictors()

        results = []
        for predictor in self._predictors:
            try:
                outcome = predictor.predict(text)
                outcome["model"] = predictor.model_name()
                results.append(outcome)
            except Exception as exc:
                log.error("Inference failed for %s: %s", predictor.model_name(), exc)

        if not results:
            raise RuntimeError("All models failed during inference.")

        labels = [r["label"] for r in results]
        consensus = max(set(labels), key=labels.count)

        return {
            "input_text": text,
            "results": results,
            "consensus": consensus,
        }


# ── Standalone demo ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_texts = [
        "This film was absolutely brilliant — a masterpiece of modern cinema.",
        "Terrible movie. Boring, predictable, and a complete waste of time.",
    ]

    comparator = ModelComparator()

    for text in sample_texts:
        print(f"\nInput: {text!r}")
        try:
            result = comparator.compare(text)
            print(f"Consensus: {result['consensus']}")
            for r in result["results"]:
                print(
                    f"  [{r['model']:22s}]  {r['label']:8s}  "
                    f"confidence={r['confidence']:.4f}  "
                    f"({r['inference_ms']:.1f} ms)"
                )
        except RuntimeError as e:
            print(f"Error: {e}")
