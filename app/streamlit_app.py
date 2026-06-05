"""
Streamlit frontend for the NLP Sentiment Analyser portfolio project.

Allows the user to select a model (DistilBERT, Logistic Regression, SVM, or
all three), enter free text, and receive a colour-coded sentiment prediction
with confidence score and inference time.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

# Make sure src/ is importable when launched from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils import load_config

# ── Page config — must be first Streamlit call ────────────────────────────────

st.set_page_config(
    page_title="NLP Sentiment Analyser",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_OPTIONS = {
    "DistilBERT (fine-tuned)": "distilbert",
    "Logistic Regression (TF-IDF)": "lr",
    "Support Vector Machine (TF-IDF)": "svm",
    "All Three — Compare": "all",
}

POSITIVE_COLOR = "#2ecc71"
NEGATIVE_COLOR = "#e74c3c"
NEUTRAL_COLOR = "#95a5a6"


# ── Model loading (cached) ────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading model ...")
def load_distilbert() -> Any:
    """Load and cache the DistilBERT predictor.

    Returns:
        A :class:`src.predict.DistilBERTPredictor` instance.

    Raises:
        FileNotFoundError: If the model has not been trained yet.
    """
    from src.predict import DistilBERTPredictor

    return DistilBERTPredictor()


@st.cache_resource(show_spinner="Loading model ...")
def load_lr() -> Any:
    """Load and cache the Logistic Regression predictor.

    Returns:
        A :class:`src.predict.BaselinePredictor` instance.
    """
    from src.predict import BaselinePredictor

    return BaselinePredictor("lr")


@st.cache_resource(show_spinner="Loading model ...")
def load_svm() -> Any:
    """Load and cache the LinearSVC predictor.

    Returns:
        A :class:`src.predict.BaselinePredictor` instance.
    """
    from src.predict import BaselinePredictor

    return BaselinePredictor("svm")


# ── UI helpers ────────────────────────────────────────────────────────────────


def _render_single_result(result: dict[str, Any], model_label: str) -> None:
    """Render label, confidence bar, and inference time for a single model.

    Args:
        result: Dict with ``label``, ``confidence``, ``inference_ms``.
        model_label: Display name for the model.
    """
    label = result["label"]
    confidence = result["confidence"]
    inference_ms = result["inference_ms"]

    color = POSITIVE_COLOR if label == "POSITIVE" else NEGATIVE_COLOR
    label_html = (
        f"<span style='font-size:2rem; font-weight:700; color:{color};'>"
        f"{label}</span>"
    )

    st.markdown(f"**{model_label}**")
    st.markdown(label_html, unsafe_allow_html=True)
    st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")
    st.caption(f"Inference time: {inference_ms:.1f} ms")


def _render_comparison_table(results: list[dict[str, Any]]) -> None:
    """Render a side-by-side comparison table for all three models.

    Args:
        results: List of per-model dicts each containing ``model``,
            ``label``, ``confidence``, ``inference_ms``.
    """
    import pandas as pd

    rows = []
    for r in results:
        rows.append({
            "Model": r.get("model", "—"),
            "Sentiment": r["label"],
            "Confidence": f"{r['confidence'] * 100:.1f}%",
            "Inference (ms)": f"{r['inference_ms']:.1f}",
        })

    df = pd.DataFrame(rows)

    def _style_label(val: str) -> str:
        color = POSITIVE_COLOR if val == "POSITIVE" else NEGATIVE_COLOR
        return f"color: {color}; font-weight: bold"

    styled = df.style.applymap(_style_label, subset=["Sentiment"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    consensus_labels = [r["label"] for r in results]
    consensus = max(set(consensus_labels), key=consensus_labels.count)
    c_color = POSITIVE_COLOR if consensus == "POSITIVE" else NEGATIVE_COLOR
    st.markdown(
        f"**Consensus (majority vote):** "
        f"<span style='color:{c_color}; font-weight:700;'>{consensus}</span>",
        unsafe_allow_html=True,
    )


# ── Sidebar ───────────────────────────────────────────────────────────────────


def render_sidebar() -> str:
    """Render the sidebar and return the selected model key.

    Returns:
        One of ``"distilbert"``, ``"lr"``, ``"svm"``, ``"all"``.
    """
    st.sidebar.title("Model Selection")
    st.sidebar.markdown(
        "Choose which model(s) to use for sentiment analysis. "
        "DistilBERT is the most accurate; baseline models are much faster."
    )

    choice = st.sidebar.radio(
        "Select model:",
        options=list(MODEL_OPTIONS.keys()),
        index=0,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About the models")
    st.sidebar.markdown(
        "- **DistilBERT** — transformer fine-tuned on SST-2 (~66M params)\n"
        "- **Logistic Regression** — TF-IDF (10k features, bigrams)\n"
        "- **SVM** — LinearSVC with same TF-IDF features"
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "_Built for the MSc AI & Data Science portfolio at "
        "Nottingham Trent University._"
    )

    return MODEL_OPTIONS[choice]


# ── Main app ──────────────────────────────────────────────────────────────────


def main() -> None:
    """Render the full Streamlit application."""

    # Header
    st.title("NLP Sentiment Analyser")
    st.markdown(
        "A portfolio project comparing **DistilBERT** (transformer) against "
        "traditional ML baselines (Logistic Regression & SVM) on binary "
        "sentiment classification."
    )
    st.markdown("---")

    selected_model = render_sidebar()

    # Text input
    user_input = st.text_area(
        "Enter text to analyse:",
        placeholder="Paste any review or text here...",
        height=140,
        key="user_text",
    )

    analyse_clicked = st.button("Analyse Sentiment", type="primary", use_container_width=True)

    if not analyse_clicked:
        st.info("Enter some text above and click **Analyse Sentiment** to get started.")

    if analyse_clicked:
        # Validate input
        if not user_input or not user_input.strip():
            st.error("Please enter some text before running the analysis.")
            return

        text = user_input.strip()

        # Single model path
        if selected_model != "all":
            try:
                if selected_model == "distilbert":
                    predictor = load_distilbert()
                    display_name = "DistilBERT"
                elif selected_model == "lr":
                    predictor = load_lr()
                    display_name = "Logistic Regression"
                else:
                    predictor = load_svm()
                    display_name = "Support Vector Machine"

                with st.spinner(f"Running {display_name} inference ..."):
                    result = predictor.predict(text)

                st.markdown("### Result")
                _render_single_result(result, display_name)

            except FileNotFoundError as exc:
                st.error(
                    f"Model not found. Have you run the training script?\n\n`{exc}`"
                )
            except Exception as exc:
                st.error(f"Inference failed: {exc}")

        # All-three comparison path
        else:
            results: list[dict[str, Any]] = []
            errors: list[str] = []

            model_loaders = [
                ("DistilBERT", load_distilbert),
                ("Logistic Regression", load_lr),
                ("Support Vector Machine", load_svm),
            ]

            with st.spinner("Running all three models ..."):
                for name, loader in model_loaders:
                    try:
                        predictor = loader()
                        outcome = predictor.predict(text)
                        outcome["model"] = name
                        results.append(outcome)
                    except FileNotFoundError:
                        errors.append(f"{name} (model not trained)")
                    except Exception as exc:
                        errors.append(f"{name} ({exc})")

            if errors:
                st.warning(
                    "Some models could not be loaded:\n"
                    + "\n".join(f"- {e}" for e in errors)
                )

            if results:
                st.markdown("### Comparison")
                _render_comparison_table(results)

                st.markdown("#### Per-model confidence")
                cols = st.columns(len(results))
                for col, r in zip(cols, results):
                    with col:
                        st.markdown(f"**{r['model']}**")
                        color = POSITIVE_COLOR if r["label"] == "POSITIVE" else NEGATIVE_COLOR
                        st.markdown(
                            f"<span style='color:{color}; font-weight:700;'>{r['label']}</span>",
                            unsafe_allow_html=True,
                        )
                        st.progress(r["confidence"])
                        st.caption(f"{r['inference_ms']:.1f} ms")
            else:
                st.error(
                    "No models could produce a prediction. "
                    "Ensure training scripts have been run."
                )

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; color:#7f8c8d; font-size:0.85rem;'>"
        "Built by <strong>Pratham Shah</strong> — "
        "MSc AI &amp; Data Science, Nottingham Trent University"
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
