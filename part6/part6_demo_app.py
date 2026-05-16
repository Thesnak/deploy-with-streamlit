"""
Part 6 Demo App — Deployment-Ready ML App
This file is the reference app used for live deployment in Part 6.

It is designed to deploy successfully on BOTH:
  - Streamlit Community Cloud (sklearn only, 1GB RAM)
  - Hugging Face Spaces (includes HF model demo, 16GB RAM)

Run locally: streamlit run part6_demo_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
import joblib

st.set_page_config(
    page_title="ML Deployment Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Deployment environment detection ────────────────────────────────────────
IS_HF_SPACES = os.environ.get("SPACE_ID") is not None
IS_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") is not None

DEPLOYMENT_ENV = (
    "🤗 Hugging Face Spaces" if IS_HF_SPACES
    else "☁️ Streamlit Community Cloud" if IS_STREAMLIT_CLOUD
    else "💻 Local development"
)

# ─── Model loading — works in all environments ────────────────────────────────
@st.cache_resource(show_spinner="Loading classifier pipeline...")
def load_sklearn_model():
    """
    Loads a pre-trained sklearn pipeline.
    Falls back to training on-the-fly if no saved model found.
    In production, replace this with hf_hub_download() for large models.
    """
    MODEL_PATH = "models/pipeline.pkl"

    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)

    # Train on-the-fly if no saved model (useful for first deploy)
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    pipeline.fit(X_tr, y_tr)

    os.makedirs("models", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    return pipeline


@st.cache_resource(show_spinner="Loading NLP model...")
def load_nlp_model():
    """
    Loads a lightweight Hugging Face model (265MB).
    Only loaded when user selects the NLP tab.
    Works on HF Spaces free tier (16GB RAM).
    For Streamlit Cloud (1GB RAM): use even smaller models.
    """
    from transformers import pipeline as hf_pipeline
    return hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )


# ─── Load sklearn model at startup ───────────────────────────────────────────
pipeline = load_sklearn_model()

from sklearn.datasets import load_breast_cancer
_data = load_breast_cancer(as_frame=True)
FEATURES     = list(_data.data.columns)
CLASSES      = list(_data.target_names)
TRAIN_STATS  = _data.data.describe().T

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🚀 Deployment Demo")
    st.markdown("---")

    st.markdown(f"**Environment**: {DEPLOYMENT_ENV}")
    st.markdown(f"**Streamlit**: {st.__version__}")

    st.markdown("---")
    st.markdown("""
**App features**
- ✅ Sklearn classifier
- ✅ Batch prediction
- ✅ HF Transformers (NLP tab)
- ✅ Graceful error handling
- ✅ Deployment-ready structure
    """)
    st.markdown("---")
    st.caption("Part 6 — Cloud Deployment")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("🚀 Production ML App")
st.markdown(f"Running on: **{DEPLOYMENT_ENV}**")

tab_predict, tab_batch, tab_nlp, tab_deploy = st.tabs([
    "🔮 Predict", "📋 Batch", "💬 NLP (HF)", "📦 Deployment Info"
])

# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.header("Single Sample Prediction")

    with st.form("predict_form"):
        st.markdown("**Enter feature values** (defaults = training set means)")
        cols = st.columns(3)
        inputs = {}
        for i, feat in enumerate(FEATURES):
            with cols[i % 3]:
                inputs[feat] = st.number_input(
                    label=feat[:30],
                    value=float(TRAIN_STATS.loc[feat, "mean"]),
                    format="%.4f",
                    key=f"feat_{i}"
                )
        submitted = st.form_submit_button("🔮 Classify", type="primary", use_container_width=True)

    if submitted:
        X_in = pd.DataFrame([inputs])

        try:
            with st.spinner("Running inference..."):
                pred  = pipeline.predict(X_in)[0]
                proba = pipeline.predict_proba(X_in)[0]

            label = CLASSES[pred]
            confidence = proba.max()

            if pred == 1:
                st.success(f"✅ **{label.upper()}** — {confidence:.1%} confidence")
            else:
                st.error(f"⚠️ **{label.upper()}** — {confidence:.1%} confidence")

            if confidence < 0.70:
                st.warning("Low-confidence prediction. Treat results with caution.")

            col1, col2 = st.columns([1, 2])
            with col1:
                for i, cls in enumerate(CLASSES):
                    st.progress(float(proba[i]), text=f"{cls.capitalize()}: {proba[i]:.1%}")

            with col2:
                fig = px.bar(
                    x=[c.capitalize() for c in CLASSES], y=proba,
                    labels={"x": "Class", "y": "Probability"},
                    title="Prediction Probabilities",
                    color=proba, color_continuous_scale="RdYlGn"
                )
                fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Inference failed: {e}")
            st.info("Check that all inputs are numeric and within expected ranges.")

# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.header("Batch Prediction")
    st.markdown("Upload a CSV with the same columns as the training data.")

    from sklearn.datasets import load_breast_cancer
    _sample = load_breast_cancer(as_frame=True).data.head(5)
    st.download_button("📥 Download sample input CSV", _sample.to_csv(index=False),
                       "sample_input.csv", "text/csv")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)

            # Validate columns
            missing_cols = [f for f in FEATURES if f not in df.columns]
            if missing_cols:
                st.error(f"Missing {len(missing_cols)} required columns. Examples: {missing_cols[:3]}")
                st.stop()

            if len(df) > 5000:
                st.warning(f"Large file ({len(df):,} rows). Processing first 5000 rows.")
                df = df.head(5000)

            st.success(f"Loaded {len(df):,} samples")

            if st.button("▶️ Run Batch Inference", type="primary"):
                with st.spinner(f"Classifying {len(df):,} rows..."):
                    preds  = pipeline.predict(df[FEATURES])
                    probas = pipeline.predict_proba(df[FEATURES])

                df["prediction"]  = [CLASSES[p] for p in preds]
                df["confidence"]  = probas.max(axis=1).round(4)

                st.success(f"Done — {len(df):,} predictions completed")

                c1, c2, c3 = st.columns(3)
                c1.metric("Benign",    (preds == 1).sum())
                c2.metric("Malignant", (preds == 0).sum())
                c3.metric("Avg confidence", f"{df['confidence'].mean():.1%}")

                st.dataframe(df[["prediction", "confidence"] + FEATURES[:5]].head(20),
                             use_container_width=True)

                st.download_button(
                    "📥 Download predictions",
                    df.to_csv(index=False),
                    "predictions.csv", "text/csv"
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")

# ══════════════════════════════════════════════════════════════════════════════
with tab_nlp:
    st.header("💬 NLP Demo — DistilBERT Sentiment")

    if IS_STREAMLIT_CLOUD and not IS_HF_SPACES:
        st.warning("""
⚠️ **On Streamlit Community Cloud**: This model (~265MB) may exceed the 1GB RAM limit.
If the app crashes, consider deploying to **Hugging Face Spaces** (16GB RAM free).
        """)

    st.info("Model loads on first use and is cached. First load takes ~20 seconds.")

    text = st.text_area("Enter text for sentiment analysis", height=120,
                        placeholder="Type or paste any text here...")

    if st.button("Analyze Sentiment", type="primary") and text:
        try:
            nlp = load_nlp_model()
            with st.spinner("Analyzing..."):
                result = nlp(text[:512])[0]

            label = result["label"]
            score = result["score"]

            if label == "POSITIVE":
                st.success(f"😊 **POSITIVE** — {score:.1%} confidence")
                st.progress(score, text=f"Positivity: {score:.1%}")
            else:
                st.error(f"😞 **NEGATIVE** — {score:.1%} confidence")
                st.progress(1 - score, text=f"Negativity: {score:.1%}")

        except MemoryError:
            st.error("Out of memory. The model is too large for this environment.")
            st.info("Deploy to Hugging Face Spaces (16GB free RAM) to run this model.")
        except Exception as e:
            st.error(f"Model error: {e}")

# ══════════════════════════════════════════════════════════════════════════════
with tab_deploy:
    st.header("📦 Deployment Reference")

    st.subheader("requirements.txt for this app")
    st.code("""
streamlit>=1.35.0,<2.0.0
scikit-learn>=1.4.0,<2.0.0
pandas>=2.0.0,<3.0.0
numpy>=1.26.0,<2.0.0
plotly>=5.20.0
joblib>=1.3.0
transformers>=4.40.0
torch
""", language="text")

    st.subheader("Deployment targets")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
**☁️ Streamlit Community Cloud**
- Free, unlimited apps
- 1 GB RAM ← watch this
- Auto-redeploy on git push
- Best for: sklearn apps, lightweight demos

```bash
# 1. Push to GitHub
git add . && git commit -m "deploy"
git push

# 2. Go to share.streamlit.io
# 3. New app → select repo
# 4. Add secrets in Settings
```
""")

    with col2:
        st.markdown("""
**🤗 Hugging Face Spaces**
- Free, 16 GB RAM ← runs Transformers
- Dedicated hardware upgrades available
- Best for: NLP, Transformers, larger models

```bash
# Clone your Space repo
git clone https://huggingface.co/spaces/user/space
cd space

# Add your files
cp -r ../my-app/* .

# Push to deploy
git add . && git commit -m "deploy" && git push
```
""")

    st.subheader("Environment detection in code")
    st.code("""
import os

IS_HF     = os.environ.get("SPACE_ID") is not None
IS_ST_CLD = os.environ.get("STREAMLIT_SHARING_MODE") is not None

# Use to conditionally load lighter models on Streamlit Cloud
if IS_ST_CLD:
    model_name = "distilbert-base-uncased"  # Small
else:
    model_name = "bert-large-uncased"        # Full
    """, language="python")

    st.subheader("Model storage decision")
    st.markdown("""
| Size | Strategy |
|------|----------|
| < 50 MB | Commit to git repo |
| 50–500 MB | Upload to HF Hub model repo |
| > 500 MB | HF Hub (private) + `hf_hub_download()` |

```python
# In your app — load from HF Hub
from huggingface_hub import hf_hub_download
import streamlit as st, joblib

@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="your-org/your-model",
        filename="pipeline.pkl",
        token=st.secrets.get("hf_api_token")
    )
    return joblib.load(path)
```
""")
