"""
Part 7 Demo App — Real-Time Streamlit App with Data + Model Monitoring
This demo shows a flexible architecture for real-time app UX, Git-friendly model loading,
and monitoring-friendly output.

Run locally: streamlit run part7_demo_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import time
from pathlib import Path
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Real-Time ML App",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

DATA_DIR = Path("data")
MODEL_DIR = Path("models")
LOG_DIR = Path("logs")

# Ensure local directories exist for datasets or artifacts.
for folder in [DATA_DIR, MODEL_DIR, LOG_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

# ─── Utility helpers ───────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    model_path = MODEL_DIR / "pipeline.pkl"
    if model_path.exists():
        return joblib.load(model_path)

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
    joblib.dump(pipeline, model_path)

    return pipeline


def log_prediction(input_data: dict, result: dict) -> None:
    log_file = LOG_DIR / "predictions.jsonl"
    event = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": input_data,
        "result": result
    }
    with log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")


def get_sample_data() -> pd.DataFrame:
    return pd.DataFrame({
        "radius_mean": [12.0, 14.2, 10.8],
        "texture_mean": [15.0, 20.1, 13.3],
        "perimeter_mean": [78.0, 90.1, 72.3],
        "area_mean": [450.0, 600.0, 380.0],
        "smoothness_mean": [0.1, 0.12, 0.09]
    })


pipeline = load_pipeline()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📡 Streaming ML Monitor")
    if st.checkbox("Show monitoring logs", key="show_logs"):
        if (LOG_DIR / "predictions.jsonl").exists():
            lines = (LOG_DIR / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
            logs = [json.loads(line) for line in lines[-10:]]
            st.write(logs)
        else:
            st.info("No logs yet. Run a prediction to generate events.")

    st.markdown("---")
    st.markdown("**App features:**\n- Batch inference\n- Prediction logging\n- Monitoring view\n- Lightweight deployment topology")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("📡 Real-Time ML Deployment")

mode = st.radio("Select mode", ["Single prediction", "Batch prediction", "Monitoring"], horizontal=True)

if mode == "Single prediction":
    st.header("Single sample inference")
    with st.form("single_form"):
        radius      = st.number_input("radius_mean", value=12.0, format="%.3f")
        texture     = st.number_input("texture_mean", value=15.0, format="%.3f")
        perimeter   = st.number_input("perimeter_mean", value=78.0, format="%.3f")
        area        = st.number_input("area_mean", value=450.0, format="%.3f")
        smoothness  = st.number_input("smoothness_mean", value=0.1, format="%.4f")
        submitted = st.form_submit_button("Run prediction")

    if submitted:
        sample = pd.DataFrame([{
            "radius_mean": radius,
            "texture_mean": texture,
            "perimeter_mean": perimeter,
            "area_mean": area,
            "smoothness_mean": smoothness
        }])

        with st.spinner("Predicting..."):
            pred = pipeline.predict(sample)[0]
            proba = pipeline.predict_proba(sample)[0].max()

        label = "Benign" if pred == 1 else "Malignant"
        st.metric("Prediction", label)
        st.metric("Confidence", f"{proba:.1%}")

        log_prediction(sample.to_dict(orient="records")[0], {"label": label, "confidence": float(proba)})
        st.success("Prediction logged")

elif mode == "Batch prediction":
    st.header("Batch inference")
    st.markdown("Upload a CSV file with the required feature columns.")

    sample_df = get_sample_data()
    st.download_button("Download sample CSV", sample_df.to_csv(index=False), "sample_input.csv", "text/csv")

    uploaded = st.file_uploader("Batch CSV", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        if not {"radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"}.issubset(df.columns):
            st.error("CSV must include radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean")
        else:
            if st.button("Run batch predictions"):
                with st.spinner("Running batch inference..."):
                    df["prediction"] = pipeline.predict(df)
                    df["confidence"] = pipeline.predict_proba(df).max(axis=1)

                st.success(f"Predicted {len(df):,} rows")
                st.dataframe(df.head(20), use_container_width=True)
                st.download_button("Download results", df.to_csv(index=False), "batch_predictions.csv", "text/csv")
                for idx, row in df.head(10).iterrows():
                    log_prediction(row.to_dict(), {"label": int(row["prediction"]), "confidence": float(row["confidence"])})

elif mode == "Monitoring":
    st.header("Monitoring & data drift")
    if (LOG_DIR / "predictions.jsonl").exists():
        logs = [json.loads(line) for line in (LOG_DIR / "predictions.jsonl").read_text(encoding="utf-8").splitlines()]
        if logs:
            df_logs = pd.DataFrame(logs)
            st.metric("Total predictions", len(df_logs))
            st.line_chart(df_logs["result"].apply(lambda r: 1 if r["label"] == "Benign" else 0))
            st.dataframe(df_logs.tail(20), use_container_width=True)
        else:
            st.info("No logs available yet.")
    else:
        st.info("No monitoring logs found. Run a prediction first.")
