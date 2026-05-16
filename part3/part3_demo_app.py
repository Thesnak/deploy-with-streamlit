"""
Part 3 Demo App — Classic ML Deployment with Scikit-learn
Run with: streamlit run part3_demo_app.py
pip install streamlit scikit-learn joblib pandas numpy plotly

NOTE: Run this to train and save the model first (or the app trains it automatically).
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from train import load_pipeline_and_data
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

st.set_page_config(
    page_title="Breast Cancer Classifier",
    page_icon="🔬",
    layout="wide"
)

try:
    pipeline, train_data = load_pipeline_and_data()
except FileNotFoundError as err:
    st.error(str(err))
    st.stop()

FEATURE_NAMES = train_data["feature_names"]
CLASS_NAMES   = train_data["target_names"].tolist()  # ["malignant", "benign"]
STATS         = train_data["stats"]
X_test        = train_data["X_test"]
y_test        = train_data["y_test"]

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 Breast Cancer Classifier")
    st.markdown("---")
    st.markdown("**Model**: Random Forest")
    st.markdown(f"**Features**: {len(FEATURE_NAMES)}")
    st.markdown(f"**Classes**: {', '.join(CLASS_NAMES)}")

    test_acc = pipeline.score(X_test, y_test)
    st.metric("Test Accuracy", f"{test_acc:.2%}")
    st.markdown("---")
    st.caption("Part 3 — Classic ML Deployment")
    st.caption("⚠️ For educational use only")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("🔬 Breast Cancer Classification")
st.markdown("Enter tumor measurements to get a classification prediction.")

tab_predict, tab_batch, tab_metrics, tab_model = st.tabs(
    ["🔮 Single Predict", "📋 Batch Predict", "📊 Model Metrics", "📄 Model Card"]
)

# ──────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.header("Feature Inputs")
    st.info("Default values are the mean of the training set. Adjust to match your sample.")

    with st.form("prediction_form"):
        inputs = {}
        # Display in groups of 3 columns
        cols = st.columns(3)
        for i, feature in enumerate(FEATURE_NAMES):
            with cols[i % 3]:
                inputs[feature] = st.number_input(
                    label=feature.replace(" ", "_"),
                    value=float(STATS.loc[feature, "mean"]),
                    min_value=float(STATS.loc[feature, "min"]) * 0.5,
                    max_value=float(STATS.loc[feature, "max"]) * 1.5,
                    format="%.4f",
                    help=f"Train range: [{STATS.loc[feature,'min']:.3f}, {STATS.loc[feature,'max']:.3f}]"
                )

        submitted = st.form_submit_button("🔮 Classify", type="primary", use_container_width=True)

    if submitted:
        X_input = pd.DataFrame([inputs])

        with st.spinner("Running classification..."):
            pred = pipeline.predict(X_input)[0]
            proba = pipeline.predict_proba(X_input)[0]

        label = CLASS_NAMES[pred]
        confidence = proba.max()

        # Result display
        if pred == 1:  # Benign
            st.success(f"✅ **Prediction: {label.upper()}**  —  Confidence: {confidence:.1%}")
        else:          # Malignant
            st.error(f"⚠️ **Prediction: {label.upper()}**  —  Confidence: {confidence:.1%}")

        if confidence < 0.70:
            st.warning("Low confidence prediction. Results should be interpreted with caution.")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.metric("Predicted Class", label.capitalize())
            st.metric("Confidence", f"{confidence:.1%}")
            for i, cls in enumerate(CLASS_NAMES):
                st.progress(float(proba[i]), text=f"{cls.capitalize()}: {proba[i]:.1%}")

        with col2:
            fig = px.bar(
                x=[c.capitalize() for c in CLASS_NAMES],
                y=proba,
                title="Prediction Probabilities",
                labels={"x": "Class", "y": "Probability"},
                color=proba,
                color_continuous_scale=["salmon", "lightgreen"]
            )
            fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

        # Feature importance for this prediction (simplified — global importances)
        with st.expander("🔍 Feature Importances (global)"):
            model = pipeline.named_steps["classifier"]
            importances = model.feature_importances_
            sorted_idx  = importances.argsort()[-15:]  # Top 15

            fig2 = px.bar(
                x=importances[sorted_idx],
                y=[FEATURE_NAMES[i] for i in sorted_idx],
                orientation="h",
                title="Top 15 Feature Importances",
                labels={"x": "Importance", "y": "Feature"}
            )
            st.plotly_chart(fig2, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
with tab_batch:
    st.header("Batch Prediction")
    st.markdown("Upload a CSV with the same feature columns. Get predictions for all rows.")

    # Sample CSV download
    sample_df = X_test.head(5).copy()
    st.download_button(
        "📥 Download sample input CSV",
        sample_df.to_csv(index=False),
        "sample_input.csv",
        "text/csv"
    )

    uploaded = st.file_uploader("Upload CSV for batch inference", type=["csv"])

    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            missing = [f for f in FEATURE_NAMES if f not in batch_df.columns]
            if missing:
                st.error(f"Missing columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
            else:
                st.success(f"Loaded {len(batch_df):,} samples")
                st.dataframe(batch_df.head(3), use_container_width=True)

                if st.button("▶️ Run Batch Inference", type="primary"):
                    with st.spinner(f"Classifying {len(batch_df):,} samples..."):
                        preds  = pipeline.predict(batch_df[FEATURE_NAMES])
                        probas = pipeline.predict_proba(batch_df[FEATURE_NAMES])

                        batch_df["prediction"]  = [CLASS_NAMES[p] for p in preds]
                        batch_df["confidence"]  = probas.max(axis=1).round(4)
                        for i, cls in enumerate(CLASS_NAMES):
                            batch_df[f"prob_{cls}"] = probas[:, i].round(4)

                    st.success(f"Completed {len(batch_df):,} predictions!")

                    c1, c2 = st.columns(2)
                    c1.metric("Benign",    (preds == 1).sum())
                    c2.metric("Malignant", (preds == 0).sum())

                    fig = px.pie(values=[(preds==1).sum(), (preds==0).sum()],
                                 names=["Benign", "Malignant"],
                                 title="Prediction Distribution",
                                 color_discrete_sequence=["#2ecc71", "#e74c3c"])
                    st.plotly_chart(fig, use_container_width=True)

                    st.dataframe(batch_df, use_container_width=True)
                    st.download_button(
                        "📥 Download predictions",
                        batch_df.to_csv(index=False),
                        "predictions.csv", "text/csv"
                    )
        except Exception as e:
            st.error(f"Error reading file: {e}")

# ──────────────────────────────────────────────────────────────────────────────
with tab_metrics:
    st.header("Model Performance Metrics")

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # Summary metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.2%}")
    c2.metric("Precision", f"{precision_score(y_test, y_pred):.2%}")
    c3.metric("Recall",    f"{recall_score(y_test, y_pred):.2%}")
    c4.metric("F1 Score",  f"{f1_score(y_test, y_pred):.2%}")

    col_cm, col_roc = st.columns(2)

    with col_cm:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm, text_auto=True,
            labels=dict(x="Predicted", y="Actual"),
            x=[c.capitalize() for c in CLASS_NAMES],
            y=[c.capitalize() for c in CLASS_NAMES],
            color_continuous_scale="Blues",
            title="Confusion Matrix"
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_roc:
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC = {roc_auc:.3f}",
                                     line=dict(color="#1f77b4", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Random",
                                     line=dict(dash="dash", color="gray")))
        fig_roc.update_layout(
            title="ROC Curve",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate"
        )
        st.plotly_chart(fig_roc, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
with tab_model:
    st.header("📄 Model Card")
    st.markdown(f"""
**Model name**: Breast Cancer Classifier v1.0
**Algorithm**: Random Forest Classifier
**Training date**: Auto-trained on first run

**Dataset**: Breast Cancer Wisconsin (Diagnostic) — UCI ML Repository
- {len(FEATURE_NAMES)} features computed from digitized images of fine needle aspirates
- 569 total samples (80% train / 20% test)

**Performance (held-out test set)**:
- Accuracy: {pipeline.score(X_test, y_test):.2%}
- AUC-ROC: 0.99+

**Preprocessing**: StandardScaler (zero mean, unit variance)

**Classes**:
- `malignant` (0): Cancerous tumor
- `benign` (1): Non-cancerous tumor

---

**⚠️ Important disclaimer**

This application is built for **educational and demonstration purposes only**.
It is NOT intended for clinical use and should NOT be used to make medical decisions.
Always consult qualified medical professionals for diagnosis and treatment.
    """)
