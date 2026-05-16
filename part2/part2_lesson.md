# Part 2 — Data & Visualization

> **Goal**: Display real data, build charts, handle file uploads, and use caching so your ML app is fast.

---

## 2.1 Working with DataFrames

Streamlit has native support for pandas DataFrames.

```python
import streamlit as st
import pandas as pd

df = pd.read_csv("data.csv")

# Interactive sortable table (best for exploration)
st.dataframe(df, use_container_width=True)

# Static table (best for small, formatted outputs)
st.table(df.describe())

# Editable dataframe (users can modify cells!)
edited_df = st.data_editor(df, use_container_width=True)
```

### Displaying with styling
```python
st.dataframe(
    df.style
      .highlight_max(color="lightgreen")
      .highlight_min(color="salmon")
      .format({"accuracy": "{:.2%}", "loss": "{:.4f}"})
)
```

---

## 2.2 Caching — The Most Important Performance Tool

Without caching, your model **reloads every time a user moves a slider**. This kills performance.

### `@st.cache_data` — for data
```python
@st.cache_data
def load_data(path):
    return pd.read_csv(path)

df = load_data("large_dataset.csv")  # Loads once, cached forever (per session)
```

### `@st.cache_resource` — for models and connections
```python
@st.cache_resource
def load_model():
    import joblib
    return joblib.load("model.pkl")

model = load_model()  # Loaded once, shared across all users
```

### Key difference

| Decorator | Use for | Shared across users? | Copied on return? |
|-----------|---------|----------------------|-------------------|
| `@st.cache_data` | DataFrames, API results, preprocessed data | No (per-session) | Yes (safe) |
| `@st.cache_resource` | ML models, DB connections, tokenizers | Yes (global) | No (shared reference) |

> **Rule of thumb**: Models and connections → `cache_resource`. Data and computed results → `cache_data`.

### Cache invalidation
```python
@st.cache_data(ttl=3600)    # Expire after 1 hour
def get_live_data():
    return fetch_from_api()

@st.cache_data(max_entries=10)  # Keep only 10 cached versions
def load_data(path):
    return pd.read_csv(path)

# Manual clear
load_data.clear()             # Clear this function's cache
st.cache_data.clear()         # Clear all cache_data
```

---

## 2.3 Charts with Streamlit's Built-in API

Quick charts — great for prototyping:

```python
import numpy as np

data = pd.DataFrame({
    "x": range(100),
    "train_loss": np.exp(-np.linspace(0, 3, 100)) + np.random.normal(0, 0.02, 100),
    "val_loss":   np.exp(-np.linspace(0, 2.5, 100)) + np.random.normal(0, 0.03, 100),
})

st.line_chart(data.set_index("x"))
st.area_chart(data.set_index("x"))
st.bar_chart(data[["train_loss"]].head(20))
st.scatter_chart(data, x="train_loss", y="val_loss")
```

---

## 2.4 Plotly — Production-Grade Charts

For ML apps, Plotly gives you interactive, professional charts:

```python
import plotly.express as px
import plotly.graph_objects as go

# ROC Curve
fig = px.line(
    x=fpr, y=tpr,
    labels={"x": "False Positive Rate", "y": "True Positive Rate"},
    title=f"ROC Curve (AUC = {auc:.3f})"
)
fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash"))
st.plotly_chart(fig, use_container_width=True)

# Confusion matrix heatmap
fig2 = px.imshow(
    cm,
    labels=dict(x="Predicted", y="Actual", color="Count"),
    x=class_names, y=class_names,
    text_auto=True, color_continuous_scale="Blues"
)
st.plotly_chart(fig2, use_container_width=True)

# Feature importance bar chart
fig3 = px.bar(
    x=importances, y=feature_names,
    orientation="h",
    title="Feature Importances",
    labels={"x": "Importance", "y": "Feature"}
)
st.plotly_chart(fig3, use_container_width=True)
```

---

## 2.5 Matplotlib & Seaborn

When you already have matplotlib code from a notebook:

```python
import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Confusion matrix
sns.heatmap(cm, annot=True, fmt="d", ax=axes[0], cmap="Blues")
axes[0].set_title("Confusion Matrix")

# Distribution plot
axes[1].hist(predictions_proba, bins=30, color="#1f77b4", edgecolor="white")
axes[1].set_title("Prediction Confidence Distribution")

st.pyplot(fig)
plt.close(fig)  # Always close to free memory
```

---

## 2.6 File Upload Pipeline

The core pattern for any ML app that accepts user data:

```python
import streamlit as st
import pandas as pd

st.header("Upload your data for inference")

uploaded_file = st.file_uploader(
    "Upload CSV file",
    type=["csv"],
    help="File must have columns: age, income, score"
)

if uploaded_file is not None:
    # Read the file
    df = pd.read_csv(uploaded_file)

    # Validate
    required_cols = ["age", "income", "score"]
    missing = [c for c in required_cols if c not in df.columns]

    if missing:
        st.error(f"Missing required columns: {missing}")
    else:
        st.success(f"Loaded {len(df):,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(), use_container_width=True)

        # Run predictions
        if st.button("Run Inference"):
            with st.spinner("Processing..."):
                predictions = model.predict(df[required_cols])
                df["prediction"] = predictions

            st.success("Done!")
            st.dataframe(df, use_container_width=True)

            # Download button
            csv_output = df.to_csv(index=False)
            st.download_button(
                "📥 Download predictions",
                data=csv_output,
                file_name="predictions.csv",
                mime="text/csv"
            )
```

---

## 2.7 Real-time Data Updates

For live dashboards or streaming inference:

```python
import time
import random

placeholder = st.empty()
chart_data = pd.DataFrame(columns=["loss"])

for step in range(100):
    new_row = pd.DataFrame({"loss": [1 / (step + 1) + random.gauss(0, 0.02)]})
    chart_data = pd.concat([chart_data, new_row], ignore_index=True)

    with placeholder.container():
        st.metric("Current loss", f"{chart_data['loss'].iloc[-1]:.4f}")
        st.line_chart(chart_data)

    time.sleep(0.1)
```

---

## 2.8 Displaying Model Outputs

### Probability distributions (classification)
```python
proba = model.predict_proba(X)[0]
classes = ["Cat", "Dog", "Bird"]

fig = px.bar(
    x=classes, y=proba,
    labels={"x": "Class", "y": "Confidence"},
    title="Prediction Probabilities",
    color=proba, color_continuous_scale="Bluered"
)
fig.update_layout(coloraxis_showscale=False)
st.plotly_chart(fig, use_container_width=True)
```

---

*Next: Part 3 — Classic ML Deployment (Scikit-learn models, joblib, preprocessing pipelines)*
