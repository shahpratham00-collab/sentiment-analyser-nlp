"""
Unit tests for src/predict.py.

These tests are designed to run with ``pytest`` from the project root:

    pytest tests/test_predict.py -v

DistilBERT tests require a trained model in ``models/saved/``.
Baseline tests require trained artefacts in ``models/baseline/``.
Tests that cannot find their model files are automatically skipped.
"""

from __future__ import annotations

import pytest

from src.utils import load_config

cfg = load_config()

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def distilbert_predictor():
    """Return a cached DistilBERTPredictor, skip if model files are missing."""
    from pathlib import Path

    saved_dir = Path(cfg["saved_dir"])
    if not (saved_dir / "config.json").exists():
        pytest.skip("DistilBERT model not found — run train_distilbert.py first.")

    from src.predict import DistilBERTPredictor

    return DistilBERTPredictor(saved_dir)


@pytest.fixture(scope="module")
def lr_predictor():
    """Return a cached Logistic Regression BaselinePredictor, skip if artefacts are missing."""
    from pathlib import Path

    baseline_dir = Path(cfg["baseline_dir"])
    if not (baseline_dir / "logistic_regression.joblib").exists():
        pytest.skip("Baseline models not found — run train_baselines.py first.")

    from src.predict import BaselinePredictor

    return BaselinePredictor("lr", baseline_dir)


@pytest.fixture(scope="module")
def svm_predictor():
    """Return a cached LinearSVC BaselinePredictor, skip if artefacts are missing."""
    from pathlib import Path

    baseline_dir = Path(cfg["baseline_dir"])
    if not (baseline_dir / "linear_svc.joblib").exists():
        pytest.skip("Baseline models not found — run train_baselines.py first.")

    from src.predict import BaselinePredictor

    return BaselinePredictor("svm", baseline_dir)


# ── DistilBERT tests ──────────────────────────────────────────────────────────


def test_distilbert_positive(distilbert_predictor):
    """Known strongly positive text should return POSITIVE."""
    result = distilbert_predictor.predict(
        "This film was an absolute masterpiece. Brilliant performances and stunning visuals."
    )
    assert result["label"] == "POSITIVE", f"Expected POSITIVE, got {result['label']}"


def test_distilbert_negative(distilbert_predictor):
    """Known strongly negative text should return NEGATIVE."""
    result = distilbert_predictor.predict(
        "Terrible movie. Boring, predictable, and a complete waste of two hours."
    )
    assert result["label"] == "NEGATIVE", f"Expected NEGATIVE, got {result['label']}"


def test_distilbert_confidence_range(distilbert_predictor):
    """Confidence score must be a float in [0, 1]."""
    result = distilbert_predictor.predict("The movie was okay.")
    assert isinstance(result["confidence"], float), "Confidence must be float"
    assert 0.0 <= result["confidence"] <= 1.0, (
        f"Confidence out of range: {result['confidence']}"
    )


def test_distilbert_inference_ms_positive(distilbert_predictor):
    """Inference time must be a non-negative number."""
    result = distilbert_predictor.predict("I loved the acting in this film.")
    assert result["inference_ms"] >= 0.0, "Inference time must be non-negative"


def test_distilbert_empty_input_raises(distilbert_predictor):
    """Empty string should raise ValueError, not crash silently."""
    with pytest.raises(ValueError, match="empty"):
        distilbert_predictor.predict("")


def test_distilbert_whitespace_input_raises(distilbert_predictor):
    """Whitespace-only string should raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        distilbert_predictor.predict("   ")


# ── Baseline predictor tests ──────────────────────────────────────────────────


def test_baseline_lr_loads(lr_predictor):
    """Logistic Regression predictor loads without error."""
    assert lr_predictor is not None
    assert lr_predictor.model_name() == "Logistic Regression"


def test_baseline_svm_loads(svm_predictor):
    """LinearSVC predictor loads without error."""
    assert svm_predictor is not None
    assert svm_predictor.model_name() == "LinearSVC"


def test_baseline_lr_positive(lr_predictor):
    """Logistic Regression should classify obviously positive text as POSITIVE."""
    result = lr_predictor.predict(
        "an absolutely wonderful and delightful experience, highly recommend"
    )
    assert result["label"] == "POSITIVE", f"Got {result['label']}"


def test_baseline_lr_negative(lr_predictor):
    """Logistic Regression should classify obviously negative text as NEGATIVE."""
    result = lr_predictor.predict(
        "awful terrible boring dreadful disaster waste of time"
    )
    assert result["label"] == "NEGATIVE", f"Got {result['label']}"


def test_baseline_svm_positive(svm_predictor):
    """LinearSVC should classify obviously positive text as POSITIVE."""
    result = svm_predictor.predict(
        "an absolutely wonderful and delightful experience, highly recommend"
    )
    assert result["label"] == "POSITIVE", f"Got {result['label']}"


def test_baseline_confidence_range(lr_predictor):
    """Confidence score from baseline predictor must be in [0, 1]."""
    result = lr_predictor.predict("The plot was somewhat interesting.")
    assert 0.0 <= result["confidence"] <= 1.0, (
        f"Confidence out of range: {result['confidence']}"
    )


def test_baseline_empty_input_raises(lr_predictor):
    """Empty input to baseline predictor must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        lr_predictor.predict("")


def test_baseline_whitespace_raises(svm_predictor):
    """Whitespace-only input to SVM predictor must raise ValueError."""
    with pytest.raises(ValueError, match="empty"):
        svm_predictor.predict("   \t  ")


def test_baseline_result_has_required_keys(lr_predictor):
    """Result dict must contain label, confidence, and inference_ms."""
    result = lr_predictor.predict("The story was gripping and well-paced.")
    assert "label" in result
    assert "confidence" in result
    assert "inference_ms" in result


# ── Invalid model key test ────────────────────────────────────────────────────


def test_invalid_classifier_key_raises():
    """Passing an unknown classifier key to BaselinePredictor must raise ValueError."""
    from src.predict import BaselinePredictor

    with pytest.raises(ValueError, match="classifier must be"):
        BaselinePredictor("random_forest")
