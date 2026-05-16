# Part 1 — Instructor Script
## Streamlit Fundamentals

**Total estimated time**: 90 minutes
**Format**: Live coding + slides + demo
**Prerequisites confirmed**: Python installed, venv set up, streamlit installed

---

## Opening (5 min)

> "Before I show you a single line of code, let me ask: how many of you have built a web app before?"

[Pause for hands]

> "Whether you have or haven't — that experience doesn't matter here. Streamlit is for Python people. It removes the entire frontend layer. You write Python, the browser shows a UI. That's it."

**Hook demo** — run `part1_demo_app.py` immediately. Let students see a working ML app before the lesson starts. Don't explain it yet. Just say:

> "This entire app — inputs, charts, predictions — is about 80 lines of Python. No HTML. No JavaScript. No CSS. Let's learn how to build this."

---

## Section 1.1 — What Is Streamlit? (10 min)

**Talk about the execution model FIRST** — it's the most common source of confusion.

> "Streamlit has a mental model that's different from React, different from Flask, different from anything else. Every time a user clicks a button or moves a slider, Streamlit runs your entire Python script from the top. Every single time."

Draw this on the board or paste this diagram in chat:
```
User moves slider
       ↓
Script runs: line 1, line 2, line 3 ... all the way down
       ↓
Browser shows updated result
```

> "This is weird at first. But it's actually genius for data apps — you always have fresh state, no stale callbacks, and your Python logic is always in control."

**Comparison table**: Spend 2 minutes on the comparison table. Students often ask "why not Flask?" or "why not Dash?"

Key message:
> "Streamlit is optimized for speed of development. You can go from model training to working demo in 20 minutes. That's the trade-off."

---

## Section 1.2 — Installation (5 min)

**Live terminal demo**:
```bash
pip install streamlit
streamlit hello
```

> "Notice what just happened. You installed a library and a web server opened in your browser. No server.py, no app factory, no routes. Streamlit handles all of that."

Common issue: students on Windows may need `python -m streamlit hello` if the `streamlit` command isn't on PATH.

---

## Section 1.3 — First App (10 min)

**Live code** — type this from scratch (don't paste):
```python
import streamlit as st

st.title("My first app")
name = st.text_input("Your name")
st.write(f"Hello, {name}!")
```

Run it. Show that:
1. The app appears instantly
2. Typing in the box immediately updates the greeting
3. Ask: "Where is the event listener? Where is the callback?" Answer: there isn't one. Streamlit handles it.

> "Every widget is just a Python variable. `st.text_input` returns whatever the user typed. That's it."

---

## Section 1.4 — Text Elements (5 min)

Quick fire live demo. Type each line and show the output:
```python
st.title, st.header, st.markdown("**bold**"), st.info("info box"), st.error("error box")
```

**Teaching moment**:
> "In an ML app, these callout boxes — info, success, warning, error — are incredibly useful. When model confidence is low, show a warning. When the prediction is done, show success. You get professional-looking UX for free."

---

## Section 1.5 — Widgets (15 min)

This is the longest section. Go through widgets in groups:

**Group 1: Text inputs** (3 min)
- `text_input`, `text_area`, `number_input`
- Show that they all return values

**Group 2: Selection** (4 min)
- `selectbox`, `multiselect`, `radio`
- Point out: multiselect returns a **list**

**Group 3: Sliders** (3 min)
- `slider` is the most-used widget in ML apps
- Show the `step` parameter — useful for threshold tuning

**Group 4: File uploader** (5 min)
- **This is critical for ML deployment**
- Show a file uploader that reads a CSV
```python
file = st.file_uploader("Upload CSV", type=["csv"])
if file:
    import pandas as pd
    df = pd.read_csv(file)
    st.dataframe(df)
```
> "If the file is None (nothing uploaded), the rest of the code doesn't run. You always have to guard with `if file:`"

---

## Section 1.6 — Layout (10 min)

**Columns demo** — most impactful layout tool:
```python
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", "92%")
with col2:
    st.metric("F1", "0.89")
```

**Sidebar demo** — show how settings naturally belong in the sidebar:
> "The sidebar is where you put app settings, model selection, hyperparameter controls. The main area is for inputs and results."

**Tabs demo** — show a 3-tab layout: Predict / Metrics / Data
> "Tabs are great for organizing complex ML apps. One tab for prediction, one for model performance, one for raw data."

---

## Section 1.7 — Session State (15 min)

⚠️ **This is the hardest concept in Part 1. Slow down here.**

Start with the problem:
```python
# This is broken:
count = 0
if st.button("Click"):
    count += 1
st.write(count)  # Always shows 0 or 1. Why?
```

Ask students: "Why does this never go above 1?"

Answer: Every click reruns the script. `count = 0` on line 1 resets it every time.

Solution:
```python
if "count" not in st.session_state:
    st.session_state.count = 0

if st.button("Click"):
    st.session_state.count += 1

st.write(st.session_state.count)
```

> "Session state lives outside the script execution cycle. It's a dictionary that survives reruns. Think of it like a global variable that Streamlit manages for you."

**ML use case**: storing predictions, conversation history for chatbots, uploaded file between reruns.

---

## Section 1.8 — Progress & Spinners (5 min)

```python
with st.spinner("Running model inference..."):
    time.sleep(2)
    result = "Cat"
st.success(f"Prediction: {result}")
```

> "In real ML apps, inference might take 1–30 seconds. Without a spinner, users think the app is broken. This one pattern will save you a lot of support questions."

---

## Lab Introduction (10 min)

Point to `part1_demo_app.py`:
> "Open this file and run it. Then look at the code. Your task is to extend it: add a second tab that shows a summary of all the inputs the user has entered across multiple submissions. You'll need session state."

Give students 10 minutes. Circulate and help.

Common issues:
- Forgetting to initialize session state before using it
- Putting `st.set_page_config` anywhere other than line 1 (it must be first)
- Confusion between `st.columns` and `st.sidebar`

---

## Closing & Q&A (5 min)

**Key questions to prompt discussion**:
1. "When would Streamlit be the wrong choice?" (High-traffic production, mobile-first apps, complex state management)
2. "What's the difference between session state and a database?" (Session state resets on page refresh; a database persists forever)
3. "What happens if two users use the app at the same time?" (Each user gets their own session — session state is per-user)

**Preview of Part 2**:
> "Next we'll add charts, handle real data, and learn about caching — which is how you stop your model from reloading every time someone moves a slider."

---

## Timing Checklist

| Section | Target Time |
|---------|-------------|
| Opening demo + hook | 5 min |
| What is Streamlit | 10 min |
| Installation | 5 min |
| First app | 10 min |
| Text elements | 5 min |
| Widgets | 15 min |
| Layout | 10 min |
| Session state | 15 min |
| Spinners | 5 min |
| Lab + Q&A | 10 min |
| **Total** | **90 min** |

---

## Common Student Questions

**Q: Can I use Streamlit with FastAPI?**
A: Yes. You can have Streamlit call a FastAPI backend. But for simple ML demos, keep everything in Streamlit.

**Q: Does Streamlit work with Jupyter notebooks?**
A: You can convert notebooks to scripts, but Streamlit doesn't run inside Jupyter. Use VS Code or a terminal.

**Q: Is Streamlit free?**
A: Yes, both the library and Streamlit Community Cloud (for deployment) have free tiers.

**Q: Can I style it with custom CSS?**
A: Yes, with `st.markdown("<style>...</style>", unsafe_allow_html=True)`. But try to avoid it — the default theme is clean and professional.
