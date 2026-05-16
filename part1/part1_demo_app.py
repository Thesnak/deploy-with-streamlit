"""
Part 1 Demo App — Streamlit Fundamentals
Run with: streamlit run part1_demo_app.py
"""

import streamlit as st
import time

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Streamlit Fundamentals Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Session state initialization ─────────────────────────────────────────────
if "submission_count" not in st.session_state:
    st.session_state.submission_count = 0
if "history" not in st.session_state:
    st.session_state.history = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ App Settings")
    st.markdown("---")

    app_theme = st.radio("Color accent", ["Blue", "Green", "Orange"])
    show_history = st.toggle("Show submission history", value=True)
    simulate_delay = st.checkbox("Simulate model latency (2s)", value=False)

    st.markdown("---")
    st.caption("Part 1 — Streamlit Fundamentals")
    st.caption(f"Total submissions: {st.session_state.submission_count}")

# ─── Main area ────────────────────────────────────────────────────────────────
st.title("🧠 Streamlit Fundamentals — Demo App")
st.markdown("This app demonstrates **widgets, layout, session state, and spinners**.")

tab_predict, tab_widgets, tab_layout = st.tabs(["🔮 Predict", "🎛️ All Widgets", "📐 Layout Demo"])

# ──────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.header("Mini Personality Predictor")
    st.info("Enter your details below. The 'model' will analyze your inputs.")

    col1, col2 = st.columns(2)

    with col1:
        name = st.text_input("Your name", placeholder="e.g. Alex")
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        mood = st.slider("Current mood (1 = grumpy, 10 = ecstatic)", 1, 10, 7)

    with col2:
        personality = st.radio("Describe yourself", ["Introvert", "Ambivert", "Extrovert"])
        hobbies = st.multiselect(
            "Hobbies",
            ["Coding", "Reading", "Gaming", "Hiking", "Cooking", "Music", "Travel"],
            default=["Coding"]
        )

    agree = st.checkbox("I agree to having my personality analyzed 🤖")
    submitted = st.button("🔮 Analyze", type="primary", disabled=not agree)

    if submitted and name:
        if simulate_delay:
            with st.spinner("Running personality model inference..."):
                time.sleep(2)
        else:
            time.sleep(0.1)

        # Fake "model" output
        score = (mood * 10) + (len(hobbies) * 5) + (30 if personality == "Extrovert" else 0)
        label = "Social Butterfly 🦋" if score > 100 else "Thoughtful Explorer 🧭" if score > 60 else "Quiet Creator 🎨"

        st.success(f"Analysis complete for **{name}**!")

        m1, m2, m3 = st.columns(3)
        m1.metric("Personality Type", label)
        m2.metric("Social Score", f"{score}/130")
        m3.metric("Mood Index", f"{mood}/10", f"{mood - 5:+d} vs neutral")

        with st.expander("🔍 Detailed breakdown"):
            st.write(f"- **Name**: {name}")
            st.write(f"- **Age**: {age}")
            st.write(f"- **Mood score**: {mood} × 10 = {mood * 10} points")
            st.write(f"- **Hobbies bonus**: {len(hobbies)} hobbies × 5 = {len(hobbies)*5} points")
            st.write(f"- **Personality bonus**: {30 if personality=='Extrovert' else 0} points")
            st.write(f"- **Total**: {score}/130")

        # Update session state
        st.session_state.submission_count += 1
        st.session_state.history.append({
            "name": name, "mood": mood, "type": label, "score": score
        })

    elif submitted and not name:
        st.error("Please enter your name before submitting.")

    if show_history and st.session_state.history:
        st.markdown("---")
        st.subheader("📋 Submission History")
        import pandas as pd
        st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)

        if st.button("Clear history"):
            st.session_state.history = []
            st.session_state.submission_count = 0
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
with tab_widgets:
    st.header("Widget Gallery")
    st.markdown("Every widget shown here — run this app and interact with all of them.")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Text Inputs")
        ti = st.text_input("text_input", placeholder="Type something")
        ta = st.text_area("text_area", height=80, placeholder="Longer text...")
        ni = st.number_input("number_input", 0, 100, 42)
        st.caption(f"Values: `{ti!r}`, `{ta!r}`, `{ni}`")

        st.subheader("Selection")
        sb = st.selectbox("selectbox", ["Option A", "Option B", "Option C"])
        ms = st.multiselect("multiselect", ["X", "Y", "Z"], default=["X"])
        rb = st.radio("radio", ["Choice 1", "Choice 2"])
        st.caption(f"Values: `{sb}`, `{ms}`, `{rb}`")

    with col_b:
        st.subheader("Sliders & Toggles")
        sl = st.slider("slider (float)", 0.0, 1.0, 0.5, 0.01)
        sl2 = st.slider("slider (range)", 0, 100, (20, 80))
        tog = st.toggle("toggle")
        cb = st.checkbox("checkbox")
        st.caption(f"Values: `{sl}`, `{sl2}`, `{tog}`, `{cb}`")

        st.subheader("Date & Color")
        import datetime
        dt = st.date_input("date_input")
        col = st.color_picker("color_picker", "#1f77b4")
        st.caption(f"Values: `{dt}`, `{col}`")

    st.subheader("File Upload")
    uploaded = st.file_uploader("Upload any file to see its info", type=["csv", "txt", "png", "jpg"])
    if uploaded:
        st.success(f"File received: **{uploaded.name}** ({uploaded.size:,} bytes, type: `{uploaded.type}`)")

# ──────────────────────────────────────────────────────────────────────────────
with tab_layout:
    st.header("Layout Tools")

    st.subheader("Columns")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", "92.4%", "+1.2%")
    c2.metric("Precision", "91.0%", "-0.5%")
    c3.metric("Recall", "93.1%", "+2.1%")
    c4.metric("F1 Score", "92.0%", "+0.8%")

    st.subheader("Callout Boxes")
    st.info("ℹ️ This is an info box — good for neutral model notes")
    st.success("✅ This is a success box — show when prediction is done")
    st.warning("⚠️ This is a warning box — show when confidence is low")
    st.error("❌ This is an error box — show when input is invalid")

    st.subheader("Expander")
    with st.expander("Click to see model config"):
        st.json({
            "model": "RandomForestClassifier",
            "n_estimators": 100,
            "max_depth": 5,
            "trained_on": "2024-01-15",
            "features": ["age", "income", "score"]
        })

    st.subheader("Progress & Spinner")
    if st.button("Simulate batch inference"):
        bar = st.progress(0, text="Processing batches...")
        for i in range(10):
            time.sleep(0.15)
            bar.progress((i + 1) / 10, text=f"Processing batch {i+1}/10...")
        st.success("All 10 batches processed!")

    with st.container(border=True):
        st.subheader("Bordered Container")
        st.write("Group related content inside `st.container(border=True)`.")
        st.write("Great for model input forms.")
