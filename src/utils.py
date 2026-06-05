"""
Shared utility functions for the Sentiment Analyser project.

Provides logging, seeding, JSON I/O, and a timing decorator used across
all training, evaluation, and inference modules.
"""

from __future__ import annotations

import functools
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

F = TypeVar("F", bound=Callable[..., Any])

# ── Constants ────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "data_dir": str(PROJECT_ROOT / "data"),
    "model_dir": str(PROJECT_ROOT / "models"),
    "docs_dir": str(PROJECT_ROOT / "docs"),
    "baseline_dir": str(PROJECT_ROOT / "models" / "baseline"),
    "saved_dir": str(PROJECT_ROOT / "models" / "saved"),
    "distilbert_model_name": "distilbert-base-uncased",
    "max_length": 128,
    "batch_size": 32,
    "num_epochs": 3,
    "learning_rate": 2e-5,
    "tfidf_max_features": 10_000,
    "tfidf_ngram_range": [1, 2],
}


# ── Logging ──────────────────────────────────────────────────────────────────


def get_logger(name: str) -> logging.Logger:
    """Return a consistently formatted logger.

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        A :class:`logging.Logger` instance with a stream handler attached.
        Calling this function multiple times with the same name is safe —
        handlers are not duplicated.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ── Config ───────────────────────────────────────────────────────────────────


def load_config(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the project config, optionally merged with caller-supplied overrides.

    Args:
        overrides: Key/value pairs that take precedence over ``DEFAULT_CONFIG``.

    Returns:
        A new dict containing the merged configuration.

    Example:
        >>> cfg = load_config({"seed": 7, "num_epochs": 5})
        >>> cfg["seed"]
        7
    """
    config = dict(DEFAULT_CONFIG)
    if overrides:
        config.update(overrides)
    return config


# ── Reproducibility ──────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    """Fix random seeds for reproducibility across Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value to apply globally.

    Note:
        PyTorch is seeded only if it is importable; the function degrades
        gracefully if torch is not installed.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ── JSON helpers ─────────────────────────────────────────────────────────────


def save_json(data: Any, path: str | Path) -> None:
    """Serialise *data* to a JSON file, creating parent directories as needed.

    Args:
        data: JSON-serialisable Python object.
        path: Destination file path.

    Raises:
        TypeError: If *data* contains non-serialisable objects.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load and return the JSON content of *path*.

    Args:
        path: Path to an existing JSON file.

    Returns:
        Deserialised Python object.

    Raises:
        FileNotFoundError: If *path* does not exist.
        json.JSONDecodeError: If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ── Timer decorator ───────────────────────────────────────────────────────────


def timer(func: F) -> F:
    """Decorator that logs the wall-clock execution time of *func*.

    The elapsed time is written via the module-level logger at INFO level.

    Args:
        func: The function to wrap.

    Returns:
        Wrapped function with identical signature.

    Example:
        >>> @timer
        ... def slow_fn():
        ...     time.sleep(0.1)
        >>> slow_fn()   # logs: "slow_fn finished in 0.100s"
    """
    _logger = get_logger(__name__)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        _logger.info("%s finished in %.3fs", func.__name__, elapsed)
        return result

    return wrapper  # type: ignore[return-value]


# ── Standalone smoke-test ────────────────────────────────────────────────────

if __name__ == "__main__":
    log = get_logger("utils.smoke_test")

    cfg = load_config({"seed": 99})
    log.info("Config loaded — seed=%s", cfg["seed"])

    set_seed(cfg["seed"])
    log.info("Seeds fixed.")

    @timer
    def _dummy() -> str:
        time.sleep(0.05)
        return "ok"

    result = _dummy()
    log.info("Timer decorator result: %s", result)

    tmp = PROJECT_ROOT / "docs" / "_utils_test.json"
    save_json({"hello": "world"}, tmp)
    loaded = load_json(tmp)
    assert loaded["hello"] == "world"
    tmp.unlink()
    log.info("JSON round-trip: OK")
