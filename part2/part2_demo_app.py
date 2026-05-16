"""
Part 2 Demo App — Data & Visualization
Run with: streamlit run part2_demo_app.py
pip install streamlit pandas numpy plotly seaborn scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Data Explorer", page_icon="📊", layout="wide")

# ─── Caching examples ─────────────────────────────────────────────────────────
@st.cache_data
def load_builtin_dataset(name: str) -> pd.DataFrame:
    """Loads a seaborn dataset — cached so it only loads once."""
    return sns.load_dataset(name)

@st.cache_data
def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    return df.select_dtypes(include="number").corr()

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📊 Data Explorer")
    data_source = st.radio("Data source", ["Built-in dataset", "Upload CSV"])

    if data_source == "Built-in dataset":
        dataset_name = st.selectbox(
            "Dataset",
            ["titanic", "iris", "penguins", "diamonds", "tips"],
            index=0
        )

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("📊 Data Explorer & Visualizer")
st.markdown("Upload a CSV or pick a built-in dataset. Explore, visualize, and download.")

# ─── Load data ────────────────────────────────────────────────────────────────
df = None

if data_source == "Built-in dataset":
    df = load_builtin_dataset(dataset_name)
    st.success(f"Loaded built-in dataset: **{dataset_name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")

else:
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Loaded: **{uploaded.name}** — {df.shape[0]:,} rows × {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Could not read file: {e}")

if df is None:
    st.info("Select a dataset or upload a CSV to get started.")
    st.stop()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_data, tab_charts, tab_corr, tab_export = st.tabs(
    ["📋 Data", "📈 Charts", "🔗 Correlations", "📥 Export"]
)

# ──────────────────────────────────────────────────────────────────────────────
with tab_data:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    col3.metric("Numeric cols", df.select_dtypes(include="number").shape[1])
    col4.metric("Missing values", f"{df.isna().sum().sum():,}")

    show_stats = st.toggle("Show descriptive statistics", value=True)

    if show_stats:
        st.subheader("Descriptive Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    st.subheader("Data Preview")
    n_rows = st.slider("Rows to display", 5, min(100, len(df)), 10)
    st.dataframe(df.head(n_rows), use_container_width=True)

    with st.expander("Column info"):
        info_df = pd.DataFrame({
            "dtype": df.dtypes,
            "non-null count": df.notna().sum(),
            "null count": df.isna().sum(),
            "unique values": df.nunique(),
        })
        st.dataframe(info_df, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
with tab_charts:
    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if not num_cols:
        st.warning("No numeric columns found for charting.")
    else:
        chart_type = st.selectbox("Chart type", ["Scatter", "Histogram", "Box plot", "Line"]) 

        if chart_type == "Scatter":
            c1, c2, c3 = st.columns(3)
            x_col     = c1.selectbox("X axis", num_cols, index=0)
            y_col     = c2.selectbox("Y axis", num_cols, index=min(1, len(num_cols)-1))
            color_col = c3.selectbox("Color by", ["None"] + cat_cols + num_cols)

            fig = px.scatter(
                df, x=x_col, y=y_col,
                color=None if color_col == "None" else color_col,
                title=f"{x_col} vs {y_col}",
                opacity=0.7
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Histogram":
            col_hist = st.selectbox("Column", num_cols)
            bins     = st.slider("Number of bins", 10, 100, 30)
            fig = px.histogram(df, x=col_hist, nbins=bins, title=f"Distribution of {col_hist}")
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Box plot":
            c1, c2 = st.columns(2)
            y_box = c1.selectbox("Numeric column", num_cols)
            x_box = c2.selectbox("Group by", ["None"] + cat_cols)
            fig = px.box(
                df, x=None if x_box == "None" else x_box, y=y_box,
                title=f"Box plot: {y_box}" + (f" by {x_box}" if x_box != "None" else "")
            )
            st.plotly_chart(fig, use_container_width=True)

        elif chart_type == "Line":
            col_line = st.selectbox("Column", num_cols)
            fig = px.line(df.reset_index(), x="index", y=col_line, title=f"{col_line} over rows")
            st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
with tab_corr:
    num_df = df.select_dtypes(include="number")
    if num_df.shape[1] < 2:
        st.warning("Need at least 2 numeric columns for a correlation matrix.")
    else:
        corr = compute_correlation(df)

        st.subheader("Correlation Matrix")
        fig = px.imshow(
            corr,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Pearson Correlation"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Top Correlated Pairs")
        # Extract upper triangle
        mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
        pairs = corr.where(mask).stack().reset_index()
        pairs.columns = ["Feature A", "Feature B", "Correlation"]
        pairs["abs_corr"] = pairs["Correlation"].abs()
        pairs = pairs.sort_values("abs_corr", ascending=False).drop("abs_corr", axis=1)
        st.dataframe(pairs.head(10), use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
with tab_export:
    st.subheader("Export Data")

    col_filter = st.multiselect("Select columns to export", df.columns.tolist(), default=df.columns.tolist())
    filtered_df = df[col_filter] if col_filter else df

    st.dataframe(filtered_df.head(), use_container_width=True)
    st.caption(f"Will export {len(filtered_df):,} rows × {len(col_filter)} columns")

    csv = filtered_df.to_csv(index=False)
    st.download_button(
        "📥 Download as CSV",
        data=csv,
        file_name="export.csv",
        mime="text/csv"
    )

    # Also show JSON preview
    with st.expander("Preview as JSON (first 5 rows)"):
        st.json(filtered_df.head(5).to_dict(orient="records"))
