# Part 3 — Classic ML Deployment

> **Goal**: Deploy a trained scikit-learn model as a Streamlit app — from model saving to live prediction with a complete preprocessing pipeline.

---

## 3.1 The ML Deployment Pipeline

Deploying ML models follows a consistent pattern regardless of the model type:

```
Train model → Save model → Load in app → Accept input → Preprocess → Predict → Display
```

In Streamlit:
```python
# At startup (cached):
model = load_model()
scaler = load_preprocessor()

# On every prediction:
raw_input = get_user_input()
processed = scaler.transform(raw_input)
prediction = model.predict(processed)
display_results(prediction)
```

---

## 3.2 Saving Models with Joblib

Always save with joblib (not pickle) for scikit-learn models. It handles numpy arrays and large objects better.

```python
# train_model.py — run this separately, offline
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Load and split data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline (preprocessor + model together)
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)
print(f"Test accuracy: {pipeline.score(X_test, y_test):.4f}")

# Save
joblib.dump(pipeline, "models/breast_cancer_pipeline.pkl")
print("Model saved!")
```

> **Pro tip**: Save the full `Pipeline` (preprocessor + model together), not just the model. This ensures preprocessing is always applied correctly and you can't accidentally skip it.

---

## 3.3 Loading Models in Streamlit

```python
import streamlit as st
import joblib
import numpy as np

@st.cache_resource
def load_pipeline(path: str):
    """Load a saved sklearn pipeline. Cached globally — loads once."""
    return joblib.load(path)

# Load at startup
pipeline = load_pipeline("models/breast_cancer_pipeline.pkl")
class_names = ["Malignant", "Benign"]
feature_names = pipeline.feature_names_in_  # sklearn 1.0+
```

---

## 3.4 Building the Input Form

For tabular ML models, you need a form that maps to your feature space.

```python
def get_user_inputs(feature_names, X_train_stats):
    """Build a form with sliders for each feature."""
    inputs = {}

    st.subheader("Enter feature values")

    # Create columns for a cleaner layout
    cols = st.columns(3)

    for i, feature in enumerate(feature_names):
        col = cols[i % 3]
        with col:
            inputs[feature] = col.number_input(
                label=feature.replace("_", " ").title(),
                min_value=float(X_train_stats[feature]["min"]),
                max_value=float(X_train_stats[feature]["max"]),
                value=float(X_train_stats[feature]["mean"]),
                step=float(X_train_stats[feature]["std"]) / 10,
                format="%.4f"
            )

    return inputs
```

---

## 3.5 Making Predictions

```python
import pandas as pd

def predict(pipeline, inputs: dict):
    """Convert inputs to DataFrame, run prediction, return result dict."""
    # Create input DataFrame (sklearn expects DataFrames or 2D arrays)
    X = pd.DataFrame([inputs])

    # Predict
    prediction = pipeline.predict(X)[0]
    probabilities = pipeline.predict_proba(X)[0]

    return {
        "label": class_names[prediction],
        "class_index": prediction,
        "probabilities": probabilities,
        "confidence": probabilities.max()
    }
```

---

## 3.6 Displaying Results with Context

```python
import plotly.express as px

def display_results(result, class_names):
    # Main prediction
    if result["class_index"] == 1:  # Benign
        st.success(f"✅ Predicted: **{result['label']}**")
    else:
        st.error(f"⚠️ Predicted: **{result['label']}**")

    # Confidence metrics
    col1, col2 = st.columns(2)
    col1.metric("Prediction", result["label"])
    col2.metric("Confidence", f"{result['confidence']:.1%}")

    # Probability bars
    fig = px.bar(
        x=class_names,
        y=result["probabilities"],
        title="Prediction Probabilities",
        labels={"x": "Class", "y": "Probability"},
        color=result["probabilities"],
        color_continuous_scale=["salmon", "lightgreen"]
    )
    fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # Warning for low confidence
    if result["confidence"] < 0.70:
        st.warning("⚠️ Low confidence prediction. Consider collecting more data.")
```

---

## 3.7 Feature Importance Visualization

A great addition to any ML app — shows users *why* the model predicted what it did:

```python
def show_feature_importance(pipeline, feature_names):
    """Extract and display feature importances from the pipeline."""
    # Access the model inside the pipeline
    model = pipeline.named_steps["classifier"]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

        # Sort
        sorted_idx = importances.argsort()

        fig = px.bar(
            x=importances[sorted_idx],
            y=[feature_names[i] for i in sorted_idx],
            orientation="h",
            title="Feature Importances",
            labels={"x": "Importance", "y": "Feature"},
            color=importances[sorted_idx],
            color_continuous_scale="Viridis"
        )
        fig.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
```

---

## 3.8 SHAP Values for Explainability

SHAP (SHapley Additive exPlanations) shows how each feature contributed to a specific prediction:

```python
import shap

@st.cache_resource
def get_shap_explainer(_pipeline, X_train):
    """Create a SHAP explainer. Underscore prefix prevents hashing the object."""
    model = _pipeline.named_steps["classifier"]
    preprocessor = _pipeline.named_steps["scaler"]
    X_transformed = preprocessor.transform(X_train)
    return shap.TreeExplainer(model), preprocessor

def show_shap_explanation(pipeline, X_input, X_train):
    explainer, preprocessor = get_shap_explainer(pipeline, X_train)

    X_processed = preprocessor.transform(X_input)
    shap_values = explainer.shap_values(X_processed)

    # For binary classification, take class 1
    if isinstance(shap_values, list):
        sv = shap_values[1][0]
    else:
        sv = shap_values[0]

    # Build a bar chart
    contributions = pd.DataFrame({
        "feature": feature_names,
        "shap_value": sv
    }).sort_values("shap_value")

    fig = px.bar(
        contributions,
        x="shap_value", y="feature",
        orientation="h",
        title="Feature Contributions (SHAP)",
        color="shap_value",
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Positive = pushes toward positive class. Negative = pushes toward negative class.")
```

---

## 3.9 Batch Prediction (CSV upload)

Beyond single-sample prediction, support bulk inference:

```python
st.header("Batch Prediction")

uploaded = st.file_uploader("Upload CSV with feature columns", type=["csv"])

if uploaded:
    batch_df = pd.read_csv(uploaded)
    st.dataframe(batch_df.head(), use_container_width=True)

    if st.button("Run Batch Inference"):
        with st.spinner(f"Running inference on {len(batch_df):,} rows..."):
            preds = pipeline.predict(batch_df)
            probas = pipeline.predict_proba(batch_df)

            batch_df["prediction"] = [class_names[p] for p in preds]
            batch_df["confidence"] = probas.max(axis=1)

        st.success(f"Done! Processed {len(batch_df):,} samples.")

        st.dataframe(batch_df, use_container_width=True)

        csv_out = batch_df.to_csv(index=False)
        st.download_button("📥 Download predictions", csv_out, "predictions.csv", "text/csv")
```

---

## 3.10 Model Card — Document Your Model

Every deployed model should have a "model card" in the UI:

```python
with st.expander("📋 Model Card"):
    st.markdown("""
    **Model**: Random Forest Classifier
    **Version**: 1.0.0
    **Training date**: 2024-01-15
    **Dataset**: Breast Cancer Wisconsin (569 samples, 30 features)
    **Task**: Binary classification (Malignant / Benign)

    **Performance (test set)**:
    - Accuracy: 96.5%
    - Precision: 95.8%
    - Recall: 97.2%
    - AUC-ROC: 0.994

    **Features used**: 30 numeric features (radius, texture, perimeter, area, smoothness, ...)

    **Preprocessing**: StandardScaler (z-score normalization)

    **⚠️ Disclaimer**: This app is for educational purposes only. Not for clinical use.
    """)
```

---

## Lab: Full ML Deployment App

Build an end-to-end deployment app:
1. Train and save a model (Iris dataset — multi-class)
2. Build a Streamlit app that loads the model
3. Accept user inputs for all 4 features
4. Show prediction probabilities as a bar chart
5. Show feature importances
6. Add batch CSV prediction with download

**Starter template**: see `part3_demo_app.py`

---

## Key Takeaways

- Always save sklearn **Pipelines** (not just models) to bundle preprocessing with the model
- Use `@st.cache_resource` to load models — they're expensive and should be shared
- `pipeline.predict_proba()` gives probabilities for confidence display
- Feature importance + SHAP plots make predictions interpretable
- Batch prediction + CSV download completes the full ML product loop
- Always add a model card explaining what the model does and its limitations

---

*Next: Part 4 — Deep Learning Deployment (TF/Keras, PyTorch, Hugging Face Transformers)*
