"""
NLP Sentiment Analyser — Cloud Deployment Version
Pratham Shah | MSc AI & Data Science | Nottingham Trent University
Loads DistilBERT from HuggingFace Hub, baseline models from repo.
"""
import time
import re
import sys
from pathlib import Path
import streamlit as st

st.set_page_config(
    page_title="NLP Sentiment Analyser | Pratham Shah",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)

POSITIVE_COLOR = "#2ecc71"
NEGATIVE_COLOR = "#e74c3c"
HF_MODEL_ID = "Shahpratham00/sentiment-distilbert-sst2"
BASELINE_DIR = Path(__file__).parent / "models" / "baseline"

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

@st.cache_resource(show_spinner="Loading DistilBERT from HuggingFace...")
def load_distilbert():
    from transformers import pipeline
    return pipeline(
        "text-classification",
        model=HF_MODEL_ID,
        tokenizer=HF_MODEL_ID,
        device=-1,
        truncation=True,
        max_length=128,
    )

@st.cache_resource(show_spinner="Loading baseline models...")
def load_baselines():
    import joblib
    vec = joblib.load(BASELINE_DIR / "tfidf_vectorizer.joblib")
    lr  = joblib.load(BASELINE_DIR / "logistic_regression.joblib")
    svm = joblib.load(BASELINE_DIR / "linear_svc.joblib")
    return vec, lr, svm

def predict_distilbert(text):
    pipe = load_distilbert()
    t0 = time.perf_counter()
    result = pipe(text.strip())[0]
    ms = (time.perf_counter() - t0) * 1000
    raw = result["label"].upper()
    label = "POSITIVE" if raw in ("LABEL_1", "POSITIVE", "1") else "NEGATIVE"
    return {"label": label, "confidence": round(float(result["score"]), 4), "inference_ms": round(ms, 2)}

def predict_baseline(text, clf_name):
    vec, lr, svm = load_baselines()
    clf = lr if clf_name == "lr" else svm
    cleaned = clean_text(text)
    t0 = time.perf_counter()
    X = vec.transform([cleaned])
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(X)[0]
        idx = int(proba.argmax())
        conf = float(proba[idx])
    else:
        decision = clf.decision_function(X)[0]
        idx = int(clf.predict(X)[0])
        conf = float(1 / (1 + pow(2.718281828, -abs(decision))))
    ms = (time.perf_counter() - t0) * 1000
    label = "POSITIVE" if idx == 1 else "NEGATIVE"
    return {"label": label, "confidence": round(conf, 4), "inference_ms": round(ms, 2)}

def render_result(result, model_label):
    color = POSITIVE_COLOR if result["label"] == "POSITIVE" else NEGATIVE_COLOR
    st.markdown(f"**{model_label}**")
    st.markdown(f"<span style='font-size:2rem;font-weight:700;color:{color};'>{result['label']}</span>", unsafe_allow_html=True)
    st.progress(result["confidence"], text=f"Confidence: {result['confidence']*100:.1f}%")
    st.caption(f"Inference time: {result['inference_ms']:.1f} ms")

# Sidebar
st.sidebar.title("Model Selection")
choice = st.sidebar.radio("Select model:", [
    "DistilBERT (fine-tuned)",
    "Logistic Regression (TF-IDF)",
    "Support Vector Machine (TF-IDF)",
    "All Three — Compare"
])
st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.markdown("""
- **DistilBERT** — transformer fine-tuned on SST-2
- **Logistic Regression** — TF-IDF + sklearn
- **SVM** — LinearSVC + TF-IDF
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Pratham Shah** | MSc AI & Data Science | NTU")

# Main
st.title("🔍 NLP Sentiment Analyser")
st.markdown("Comparing **DistilBERT** (transformer) vs **Logistic Regression** and **SVM** (classical ML) on binary sentiment classification.")
st.markdown("---")

user_input = st.text_area("Enter text to analyse:", placeholder="Paste any review or text here...", height=140)
analyse = st.button("Analyse Sentiment", type="primary", use_container_width=True)

if not analyse:
    st.info("Enter some text above and click **Analyse Sentiment** to get started.")

if analyse:
    if not user_input or not user_input.strip():
        st.error("Please enter some text.")
    else:
        text = user_input.strip()
        try:
            if choice == "DistilBERT (fine-tuned)":
                result = predict_distilbert(text)
                st.markdown("### Result")
                render_result(result, "DistilBERT")

            elif choice == "Logistic Regression (TF-IDF)":
                result = predict_baseline(text, "lr")
                st.markdown("### Result")
                render_result(result, "Logistic Regression")

            elif choice == "Support Vector Machine (TF-IDF)":
                result = predict_baseline(text, "svm")
                st.markdown("### Result")
                render_result(result, "Support Vector Machine")

            else:
                st.markdown("### Comparison")
                results = []
                with st.spinner("Running all three models..."):
                    for name, fn in [
                        ("DistilBERT", lambda: predict_distilbert(text)),
                        ("Logistic Regression", lambda: predict_baseline(text, "lr")),
                        ("SVM", lambda: predict_baseline(text, "svm")),
                    ]:
                        r = fn()
                        r["model"] = name
                        results.append(r)

                import pandas as pd
                df = pd.DataFrame([{
                    "Model": r["model"],
                    "Sentiment": r["label"],
                    "Confidence": f"{r['confidence']*100:.1f}%",
                    "Inference (ms)": f"{r['inference_ms']:.1f}"
                } for r in results])
                st.dataframe(df, use_container_width=True, hide_index=True)

                cols = st.columns(3)
                for col, r in zip(cols, results):
                    with col:
                        color = POSITIVE_COLOR if r["label"] == "POSITIVE" else NEGATIVE_COLOR
                        st.markdown(f"**{r['model']}**")
                        st.markdown(f"<span style='color:{color};font-weight:700;'>{r['label']}</span>", unsafe_allow_html=True)
                        st.progress(r["confidence"])
                        st.caption(f"{r['inference_ms']:.1f} ms")

                labels = [r["label"] for r in results]
                consensus = max(set(labels), key=labels.count)
                color = POSITIVE_COLOR if consensus == "POSITIVE" else NEGATIVE_COLOR
                st.markdown(f"**Consensus:** <span style='color:{color};font-weight:700;'>{consensus}</span>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.markdown("<div style='text-align:center;color:#7f8c8d;font-size:0.85rem;'>Built by <strong>Pratham Shah</strong> — MSc AI & Data Science, Nottingham Trent University</div>", unsafe_allow_html=True)
