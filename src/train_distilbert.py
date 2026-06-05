"""
Fine-tune DistilBERT on SST-2 for binary sentiment classification.

Uses the HuggingFace Trainer API with early stopping, computes accuracy and
macro-F1 at each evaluation step, and saves the final model + tokenizer to
``models/saved/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import EvalPrediction

from src.utils import get_logger, load_config, set_seed, timer

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
LABEL2ID = {"NEGATIVE": 0, "POSITIVE": 1}
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}

VAL_SPLIT_RATIO = 0.10
EARLY_STOPPING_PATIENCE = 2
WARMUP_RATIO = 0.1
WEIGHT_DECAY = 0.01

log = get_logger(__name__)


# ── Metrics ───────────────────────────────────────────────────────────────────


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    """Compute accuracy and macro-F1 from Trainer evaluation output.

    Args:
        eval_pred: Named tuple with ``predictions`` (logits) and ``label_ids``.

    Returns:
        Dict with ``accuracy`` and ``f1`` keys.
    """
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return {"accuracy": round(float(acc), 4), "f1": round(float(f1), 4)}


# ── Tokenisation ──────────────────────────────────────────────────────────────


def tokenize_dataset(
    dataset: Any,
    tokenizer: AutoTokenizer,
    max_length: int,
    text_column: str = "sentence",
) -> Any:
    """Apply tokeniser to every example in *dataset* using batched mapping.

    Args:
        dataset: A HuggingFace ``DatasetDict`` or ``Dataset``.
        tokenizer: Pre-loaded tokeniser.
        max_length: Maximum token sequence length.
        text_column: Name of the column containing raw text.

    Returns:
        Tokenised dataset with ``input_ids``, ``attention_mask`` columns added.
    """
    def _tokenize(batch: dict[str, Any]) -> dict[str, Any]:
        return tokenizer(
            batch[text_column],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    return dataset.map(_tokenize, batched=True, desc="Tokenising")


# ── Data loading ─────────────────────────────────────────────────────────────


@timer
def load_and_split_sst2(val_ratio: float) -> tuple[Any, Any]:
    """Load SST-2 and create a train/validation split from the training data.

    The official SST-2 validation set has no public labels, so we carve a
    held-out slice from the training data.

    Args:
        val_ratio: Fraction of training data reserved for validation.

    Returns:
        A 2-tuple ``(train_dataset, val_dataset)``.
    """
    log.info("Loading SST-2 from HuggingFace datasets ...")
    dataset = load_dataset("glue", "sst2")

    split = dataset["train"].train_test_split(test_size=val_ratio, seed=42)
    train_ds = split["train"]
    val_ds = split["test"]

    log.info(
        "Split — train: %d | val: %d",
        len(train_ds),
        len(val_ds),
    )
    return train_ds, val_ds


# ── Training ──────────────────────────────────────────────────────────────────


@timer
def train(cfg: dict[str, Any]) -> None:
    """Execute the full DistilBERT fine-tuning pipeline.

    Args:
        cfg: Project configuration dict from :func:`src.utils.load_config`.
    """
    saved_dir = Path(cfg["saved_dir"])
    saved_dir.mkdir(parents=True, exist_ok=True)

    checkpoints_dir = saved_dir / "checkpoints"

    log.info("Loading tokeniser: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    log.info("Loading model: %s (%d labels)", MODEL_NAME, NUM_LABELS)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds, val_ds = load_and_split_sst2(VAL_SPLIT_RATIO)

    log.info("Tokenising datasets ...")
    train_ds = tokenize_dataset(train_ds, tokenizer, cfg["max_length"])
    val_ds = tokenize_dataset(val_ds, tokenizer, cfg["max_length"])

    train_ds = train_ds.rename_column("label", "labels")
    val_ds = val_ds.rename_column("label", "labels")
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        num_train_epochs=cfg["num_epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=WARMUP_RATIO,
        weight_decay=WEIGHT_DECAY,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=str(saved_dir / "logs"),
        logging_steps=50,
        report_to="none",
        seed=cfg["seed"],
        fp16=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    log.info("Starting fine-tuning for %d epochs ...", cfg["num_epochs"])
    trainer.train()

    log.info("Training complete. Saving model and tokeniser to %s", saved_dir)
    trainer.save_model(str(saved_dir))
    tokenizer.save_pretrained(str(saved_dir))

    final_metrics = trainer.evaluate()
    log.info("Final validation metrics: %s", final_metrics)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    """Entry point — loads config, sets seed, and launches training."""
    cfg = load_config()
    set_seed(cfg["seed"])
    train(cfg)
    log.info("DistilBERT fine-tuning pipeline complete.")


if __name__ == "__main__":
    main()
