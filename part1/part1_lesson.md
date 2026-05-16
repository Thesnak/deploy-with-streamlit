# Part 1 — Streamlit Fundamentals

> **Goal**: Understand what Streamlit is, install it, and build your first interactive app from scratch.

---

## 1.1 What Is Streamlit?

Streamlit is an open-source Python library that turns Python scripts into shareable web apps — **without any HTML, CSS, or JavaScript**.

You write Python. Streamlit renders the UI.

```
Your Python script
        ↓
   Streamlit runs it top to bottom
        ↓
   Browser renders the result
        ↓
   User interacts → script reruns
```

### Why Streamlit for ML?

| Tool | Good for | Not ideal for |
|------|----------|---------------|
| Streamlit | Fast prototyping, ML demos, internal tools | High-traffic production apps |
| Flask/FastAPI | REST APIs, backend services | Interactive UIs |
| Dash | Complex dashboards | Quick demos |
| Gradio | Simple model demos | Full-featured apps |

Streamlit's killer feature for ML engineers: **zero frontend code, full Python, hot-reloads on save**.

---

## 1.2 Installation & First Run

```bash
pip install streamlit
streamlit hello        # Opens the demo gallery
```

To run your own app:
```bash
streamlit run my_app.py
```

Streamlit starts a local server at `http://localhost:8501`.

---

## 1.3 The Execution Model

This is the most important concept in Streamlit:

> **Every time a user interacts with a widget, the entire script reruns from top to bottom.**

```python
# app.py
import streamlit as st

st.title("My first app")
name = st.text_input("Your name")
st.write(f"Hello, {name}!")
```

When the user types in the text box → the script reruns → `st.write` outputs the updated greeting.

This is called the **execution model**. It's simple but it means:
- No callbacks (usually)
- No manual DOM updates
- State management needs special handling (covered in 1.7)

---

## 1.4 Text & Markdown Elements

```python
import streamlit as st

st.title("App Title")                          # Large heading
st.header("Section Header")                    # Medium heading
st.subheader("Sub-section")                    # Smaller heading
st.text("Plain text")                          # Monospace-friendly
st.markdown("**Bold**, *italic*, `code`")      # Full Markdown support
st.caption("Small caption text")               # Small gray text
st.code("print('hello')", language="python")   # Syntax-highlighted code block
st.latex(r"\frac{1}{\sqrt{2\pi}} e^{-x^2/2}") # LaTeX math
```

### Callout boxes (great for ML apps)
```python
st.info("ℹ️ Model was last trained on 2024-01-01")
st.success("✅ Prediction complete!")
st.warning("⚠️ Confidence below 70% — interpret with caution")
st.error("❌ Input is out of the training distribution")
```

---

## 1.5 Input Widgets

Widgets are the heart of Streamlit interactivity. Every widget returns a Python value.

### Text inputs
```python
name    = st.text_input("Enter your name", placeholder="e.g. Alex")
message = st.text_area("Long text", height=150)
number  = st.number_input("Pick a number", min_value=0, max_value=100, value=50)
password = st.text_input("Password", type="password")
```

### Selection widgets
```python
option    = st.selectbox("Choose a model", ["Logistic Regression", "Random Forest", "XGBoost"])
multi     = st.multiselect("Select features", ["age", "income", "score"])
radio_val = st.radio("Task type", ["Classification", "Regression"])
```

### Sliders and toggles
```python
threshold = st.slider("Decision threshold", 0.0, 1.0, 0.5, step=0.01)
n_trees   = st.slider("Number of trees", 10, 500, 100)
show_raw  = st.toggle("Show raw data")
agree     = st.checkbox("I agree to terms")
```

### Date, time, and color
```python
date  = st.date_input("Training cutoff date")
time  = st.time_input("Run inference at")
color = st.color_picker("Pick a chart color", "#1f77b4")
```

### File and media
```python
file  = st.file_uploader("Upload a CSV", type=["csv"])
image = st.file_uploader("Upload an image", type=["jpg", "png"]) 
camera_photo = st.camera_input("Take a photo")
```

### Buttons
```python
clicked = st.button("Run Model")
if clicked:
    st.write("Running...")

# Download button
st.download_button("Download results", data="col1,col2\n1,2", file_name="results.csv")
```

---

## 1.6 Layout & Containers

### Columns
```python
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Accuracy", "92.4%", "+1.2%")

with col2:
    st.metric("F1 Score", "0.91", "-0.01")

with col3:
    st.metric("AUC", "0.97", "+0.03")
```

with col2:
    st.subheader("Distribution")

---

## 1.7 Session State

Because Streamlit reruns the script on every interaction, variables are reset on each run. **Session state** persists values across reruns.

```python
import streamlit as st

# Initialize state
if "count" not in st.session_state:
    st.session_state.count = 0

# Increment on button click
if st.button("Click me"):
    st.session_state.count += 1

st.write(f"Button clicked {st.session_state.count} times")
```

### Typical ML use: storing predictions
```python
if "predictions" not in st.session_state:
    st.session_state.predictions = []

if st.button("Predict"):
    result = model.predict(input_data)
    st.session_state.predictions.append(result)

st.write(f"Total predictions made: {len(st.session_state.predictions)}")
```

### Key rules for session state
- Access with `st.session_state["key"]` or `st.session_state.key`
- Always initialize with `if "key" not in st.session_state`
- State survives reruns but **not** page refreshes (unless you use a database or file)

---

## 1.8 Displaying Data & Media

```python
import pandas as pd

df = pd.read_csv("data.csv")

st.dataframe(df)                          # Interactive, sortable table
st.table(df.head(5))                      # Static, non-interactive table
st.json({"key": "value", "score": 0.92})  # Formatted JSON
st.metric("Accuracy", "92.4%", "+1.2%")  # KPI metric with delta

# Images
from PIL import Image
img = Image.open("photo.jpg")
st.image(img, caption="Input image", use_container_width=True)

# Video
st.video("demo.mp4")

# Audio
st.audio("output.wav")
```

---

## 1.9 Progress and Spinners

Essential for ML inference (which takes time):

```python
import time

with st.spinner("Running inference..."):
    time.sleep(3)  # Replace with actual model call
    result = model.predict(X)

st.success("Done!")

# Progress bar for batch processing
progress = st.progress(0)
for i, batch in enumerate(batches):
    process(batch)
    progress.progress((i + 1) / len(batches))
```

---

## Lab: Build Your First Streamlit App

Build an app that:
1. Takes a user's name, age, and a mood slider (1–10)
2. Displays a personalized greeting with the inputs
3. Shows a "personality report" in an expander
4. Has a sidebar for app settings (theme color picker)
5. Counts how many times the user has submitted (session state)

**Starter template**: see `part1_demo_app.py`

---

## Key Takeaways

- Streamlit reruns your script top-to-bottom on every interaction
- Widgets return Python values — use them like any variable
- Layout tools: `st.columns`, `st.sidebar`, `st.tabs`, `st.expander`
- Session state (`st.session_state`) persists values across reruns
- `st.spinner` and `st.progress` are essential for ML apps with slow inference

---

*Next: Part 2 — Data & Visualization (DataFrames, charts, file uploads, caching)*
