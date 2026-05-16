"""
Part 5 Demo App — Advanced Streamlit Features
Run with: streamlit run part5_demo_app.py
Demonstrates: multi-page structure (simulated), secrets, caching, fragments, forms, auth

Note: True multi-page requires the pages/ directory. This file simulates it in tabs
so the demo is self-contained and runnable as a single file.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.express as px
import joblib, os

st.set_page_config(
    page_title="Advanced Streamlit Demo",
    page_icon="⚙️",
    layout="wide"
)

# ─── Simulated secrets (in production: st.secrets["key"]) ─────────────────────
DEMO_PASSWORD = "streamlit123"

# ─── Model loader (cache_resource — shared across all "pages") ───────────────
@st.cache_resource(show_spinner="Loading model...")
def get_model():
    from sklearn.datasets import load_breast_cancer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split

    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    pl = Pipeline([("sc", StandardScaler()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
    pl.fit(X_tr, y_tr)

    return pl, X_tr, X_te, y_tr, y_te, list(X.columns), ["malignant", "benign"]

pipeline, X_train, X_test, y_train, y_test, FEATURES, CLASSES = get_model()

# ─── Auth helper ──────────────────────────────────────────────────────────────
def check_password(password: str) -> bool:
    return password == DEMO_PASSWORD

# ─── Session state init ───────────────────────────────────────────────────────
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None
if "batch_results" not in st.session_state:
    st.session_state.batch_results = None

# ─── Sidebar nav ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ ML Platform")
    page = st.radio("Navigate", [
        "🏠 Home",
        "📊 EDA (with shared state)",
        "🔮 Predict (fragments demo)",
        "📋 Batch (forms demo)",
        "🔒 Metrics (auth demo)",
        "🎨 Theme & Secrets",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("Part 5 — Advanced Streamlit")

st.title("⚙️ Advanced Streamlit Features Demo")

# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
## What this demo covers

| Feature | Page |
|---------|------|
| Shared session state across pages | 📊 EDA |
| `@st.fragment` for partial reruns | 🔮 Predict |
| `st.form` for batched submissions | 📋 Batch |
| Password authentication + `st.stop()` | 🔒 Metrics |
| Secrets & custom config | 🎨 Theme & Secrets |

**Navigate using the sidebar →**
    """)

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA (with shared state)":
    st.header("📊 EDA — Shared State Demo")
    st.info("Data uploaded here will be available on the **Predict** and **Batch** pages via `st.session_state`.")

    source = st.radio("Data source", ["Use sample test data", "Upload CSV"])

    if source == "Use sample test data":
        if st.button("Load sample data (X_test, 114 rows)"):
            st.session_state.uploaded_df = X_test.copy()
            st.success("Sample data loaded into session state!")

    else:
        uploaded = st.file_uploader("Upload CSV with feature columns", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            missing = [f for f in FEATURES if f not in df.columns]
            if missing:
                st.error(f"Missing {len(missing)} required columns. First few: {missing[:3]}")
            else:
                st.session_state.uploaded_df = df[FEATURES]
                st.success(f"Loaded {len(df):,} rows into session state!")

    if st.session_state.uploaded_df is not None:
        df = st.session_state.uploaded_df
        st.markdown(f"**Current dataset**: {df.shape[0]:,} rows × {df.shape[1]} columns")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Statistics")
            st.dataframe(df.describe().round(3), use_container_width=True)
        with col2:
            st.subheader("Distribution")
            feat = st.selectbox("Feature to plot", FEATURES[:10])
            fig = px.histogram(df, x=feat, nbins=30, title=f"Distribution: {feat}")
            st.plotly_chart(fig, use_container_width=True)

        if st.button("🗑️ Clear session data"):
            st.session_state.uploaded_df = None
            st.session_state.batch_results = None
            st.rerun()
    else:
        st.warning("No data loaded. Choose a source above.")

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predict (fragments demo)":
    st.header("🔮 Predict — Fragment Demo")
    st.info("The feature importance chart below is **outside the fragment** — it renders once. The prediction panel is a **fragment** — only it reruns when inputs change.")

    # ── This section renders ONCE (not inside a fragment) ──────────────────────
    with st.expander("📊 Global Feature Importances (renders once — not in fragment)", expanded=False):
        importances = pipeline.named_steps["clf"].feature_importances_
        sorted_idx  = importances.argsort()[-10:]
        fig = px.bar(
            x=importances[sorted_idx],
            y=[FEATURES[i] for i in sorted_idx],
            orientation="h", title="Top 10 Feature Importances"
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("⬆️ This plot doesn't rerun when you adjust the inputs below.")

    st.divider()

    # ── Fragment: only this reruns on input change ─────────────────────────────
    @st.fragment
    def prediction_fragment():
        st.subheader("🎛️ Input Controls (inside @st.fragment)")
        st.caption("Move these sliders — only this section reruns, not the chart above.")

        STATS = X_train.describe().T

        c1, c2, c3 = st.columns(3)
        vals = {}
        for i, feat in enumerate(FEATURES[:9]):
            col = [c1, c2, c3][i % 3]
            vals[feat] = col.number_input(
                feat[:25],
                value=float(STATS.loc[feat, "mean"]),
                format="%.3f",
                key=f"frag_{feat}"
            )

        # Fill remaining features with mean
        for feat in FEATURES[9:]:
            vals[feat] = float(STATS.loc[feat, "mean"])

        if st.button("⚡ Predict (instant — fragment only reruns)", type="primary"):
            X_in = pd.DataFrame([vals])
            pred  = pipeline.predict(X_in)[0]
            proba = pipeline.predict_proba(X_in)[0]
            label = CLASSES[pred]

            if pred == 1:
                st.success(f"✅ {label.upper()} — {proba[1]:.1%} confidence")
            else:
                st.error(f"⚠️ {label.upper()} — {proba[0]:.1%} confidence")

            for i, cls in enumerate(CLASSES):
                st.progress(float(proba[i]), text=f"{cls.capitalize()}: {proba[i]:.1%}")

    prediction_fragment()

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📋 Batch (forms demo)":
    st.header("📋 Batch Prediction — st.form Demo")

    if st.session_state.uploaded_df is None:
        st.warning("No data loaded. Go to the 📊 EDA page to load data first.")
        st.stop()

    df = st.session_state.uploaded_df
    st.success(f"Using dataset from session state: {len(df):,} rows")

    # ── Form: all settings submitted at once ───────────────────────────────────
    st.subheader("Inference Settings")
    st.caption("With `st.form`, the model doesn't run until you click Submit.")

    with st.form("batch_settings"):
        col1, col2 = st.columns(2)

        threshold   = col1.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)
        max_rows    = col2.number_input("Max rows to process", 10, len(df), min(100, len(df)))
        output_cols = st.multiselect(
            "Extra columns to include in output",
            FEATURES, default=FEATURES[:3]
        )
        add_proba   = st.checkbox("Include probability columns", value=True)

        submitted = st.form_submit_button("▶️ Run Batch Inference", type="primary", use_container_width=True)

    if submitted:
        batch = df.head(max_rows)
        with st.spinner(f"Classifying {len(batch):,} rows..."):
            progress = st.progress(0)
            preds, probas = [], []
            chunk = 10
            for i in range(0, len(batch), chunk):
                c = batch.iloc[i:i+chunk]
                preds.extend(pipeline.predict(c).tolist())
                probas.extend(pipeline.predict_proba(c).tolist())
                progress.progress(min(i + chunk, len(batch)) / len(batch))
            time.sleep(0.3)

        result_df = batch[output_cols].copy() if output_cols else pd.DataFrame(index=batch.index)
        result_df["prediction"]  = [CLASSES[p] for p in preds]
        result_df["confidence"]  = [max(p) for p in probas]
        if add_proba:
            for i, cls in enumerate(CLASSES):
                result_df[f"prob_{cls}"] = [p[i] for p in probas]

        st.session_state.batch_results = result_df
        st.success(f"Done! Processed {len(result_df):,} rows.")

        c1, c2, c3 = st.columns(3)
        benign_n = sum(1 for p in preds if p == 1)
        malig_n  = len(preds) - benign_n
        low_conf = sum(1 for p in result_df["confidence"] if p < threshold)
        c1.metric("Benign", benign_n)
        c2.metric("Malignant", malig_n)
        c3.metric("Low confidence", low_conf, help=f"Confidence < {threshold:.0%}")

        st.dataframe(result_df.head(20), use_container_width=True)
        st.download_button("📥 Download results", result_df.to_csv(index=False), "batch_results.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔒 Metrics (auth demo)":
    st.header("🔒 Protected Metrics Page")

    if not st.session_state.authenticated:
        st.info("This page requires a password. (Demo password: `streamlit123`)")

        with st.form("login_form"):
            pw = st.text_input("Password", type="password", placeholder="Enter password")
            login_btn = st.form_submit_button("Login")

        if login_btn:
            if check_password(pw):
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("Incorrect password.")

        st.stop()  # ← Halts execution; nothing below renders if not authenticated

    # ── Only rendered after authentication ────────────────────────────────────
    col_logout, _ = st.columns([1, 5])
    if col_logout.button("Logout"):
        st.session_state.authenticated = False
        st.rerun()

    st.success("Authenticated ✅")
    st.subheader("Model Performance on Test Set")

    from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    c1, c2, c3, c4 = st.columns(4)
    from sklearn.metrics import accuracy_score, f1_score
    c1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred):.2%}")
    c2.metric("F1",        f"{f1_score(y_test, y_pred):.2%}")
    c3.metric("AUC",       f"{auc(*roc_curve(y_test, y_proba)[:2]):.3f}")
    c4.metric("Test size", len(y_test))

    col_cm, col_roc = st.columns(2)
    with col_cm:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, x=[c.capitalize() for c in CLASSES],
                           y=[c.capitalize() for c in CLASSES],
                           color_continuous_scale="Blues", title="Confusion Matrix")
        st.plotly_chart(fig_cm, use_container_width=True)
    with col_roc:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        import plotly.graph_objects as go
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"AUC={roc_auc:.3f}",
                                     line=dict(color="#1f77b4", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], line=dict(dash="dash", color="gray"), name="Random"))
        fig_roc.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(fig_roc, use_container_width=True)

elif page == "🎨 Theme & Secrets":
    st.header("🎨 Themes, Config & Secrets")

    tab1, tab2, tab3 = st.tabs(["Secrets", "Config", "Cache Control"])

    with tab1:
        st.subheader("st.secrets pattern")
        st.code("""
# .streamlit/secrets.toml
openai_api_key = "sk-..."
hf_api_token   = "hf_..."

[database]
host = "db.company.com"
password = "..."
""", language="toml")

        st.code("""
# In your app:
import streamlit as st
api_key = st.secrets["openai_api_key"]
db_host = st.secrets["database"]["host"]
""", language="python")

        st.info("On Streamlit Cloud: add secrets in the app settings UI — same TOML format, stored securely.")

    with tab2:
        st.subheader(".streamlit/config.toml")
        st.code("""
[theme]
primaryColor         = "#00D4FF"
backgroundColor      = "#0E1117"
secondaryBackgroundColor = "#1A1D23"
textColor            = "#FAFAFA"
font                 = "sans serif"

[server]
maxUploadSize = 200    # MB
enableCORS    = false

[runner]
fastReruns = true
""", language="toml")

    with tab3:
        st.subheader("Cache management")
        col1, col2, col3 = st.columns(3)

        if col1.button("Clear cache_data"):
            st.cache_data.clear()
            st.success("cache_data cleared")

        if col2.button("Clear cache_resource"):
            st.cache_resource.clear()
            st.success("cache_resource cleared (models will reload)")

        if col3.button("Force rerun"):
            st.rerun()

        st.code("""
# In production, clear cache programmatically:
st.cache_data.clear()         # All cached data
load_dataset.clear()          # Just one function

# Or with TTL:
@st.cache_data(ttl=3600)      # Auto-expires after 1 hour
def fetch_live_data():
    return api.get_latest()
""", language="python")
