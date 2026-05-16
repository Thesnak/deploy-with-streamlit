# Part 2 — Instructor Script
## Data & Visualization

**Total estimated time**: 60 minutes
**Format**: Live coding + interactive demo
**Assumed**: Students completed Part 1 and understand widgets, layout, session state

---

## Opening (3 min)

> "In Part 1 we built UIs. In Part 2 we connect those UIs to real data and real charts. By the end of this session you'll have built a data explorer that can accept any CSV, profile it, and visualize any columns — the kind of tool you'd actually use in a real ML project."

Quick check: Run `part2_demo_app.py` and show the finished result. Let it sink in for 30 seconds, then say:

> "Every chart here is one or two lines of Python. Let's build it."

---

## Section 2.1 — DataFrames (8 min)

Live demo with a small dataset. Use the Iris or Titanic dataset (built-in via seaborn):

```python
import seaborn as sns
df = sns.load_dataset("titanic")
st.dataframe(df)
```

Show:
1. `st.dataframe` — sortable, filterable, resizable
2. `st.table` — static, clean for small outputs
3. `st.data_editor` — editable cells (great for annotation tools)

**Key moment**: show `use_container_width=True`. Without it, tables are cramped.

> "Notice you can click column headers to sort. You can resize columns. This is completely free — zero configuration."

---

## Section 2.2 — Caching (12 min)

⚠️ **This section determines whether students' ML apps are fast or unusable. Don't skip it.**

**Start with the problem** — add a print statement to show reruns:

```python
def load_data():
    print("Loading data...")         # Watch the terminal
    return pd.read_csv("data.csv")

df = load_data()
name = st.text_input("Your name")   # Move slider → "Loading data..." prints again
st.write(name)
```

Show the terminal. Every widget interaction triggers the load. For a model, this means:
- Random Forest: reloads in ~500ms
- BERT: reloads in ~8 seconds
- Users leave after 3 seconds of nothing happening

**Fix it**:
```python
@st.cache_resource
def load_model():
    print("Loading model...")    # Prints ONCE then never again
    return joblib.load("model.pkl")
```

**The two decorators — draw this on the board**:

```
@st.cache_data         @st.cache_resource
───────────────────    ──────────────────────────
DataFrames             ML models
API responses          Database connections
Preprocessing results  Tokenizers / embeddings
─ per-session          ─ shared across ALL users
─ safe to mutate       ─ don't mutate (shared ref)
```

> "The way to remember: `cache_resource` for anything that's expensive to create and can be shared. `cache_data` for anything that's specific to a user or a computation."

**Common mistake to mention**:
> "A lot of beginners use `cache_data` for models. It works but it's wasteful — it copies the model object for each cache hit. For a 500MB model, that's 500MB per cache hit. Use `cache_resource`."

---

## Section 2.3 — Built-in Charts (5 min)

Quick tour:
```python
st.line_chart(df[["train_loss", "val_loss"]])
st.area_chart(...)
st.bar_chart(...)
```

> "These are quick-and-dirty. Great for notebook-to-demo in 5 minutes. But for production apps, switch to Plotly."

---

## Section 2.4 — Plotly (15 min)

This is the core of the section. Cover three patterns:

**Pattern 1: ROC Curve**

> "Every classification model app should have a ROC curve. Here's how to build one in 8 lines."

Live code the ROC curve. Show that `st.plotly_chart` just takes a figure object — no special Streamlit knowledge needed.

**Pattern 2: Confusion matrix heatmap**

> "Heatmaps are one line with Plotly Express — `px.imshow`. This pattern works for any 2D array."

**Pattern 3: Feature importance bar chart**

> "Horizontal bar charts for feature importance are standard. Always sort descending."

```python
sorted_idx = importances.argsort()
fig = px.bar(x=importances[sorted_idx], y=feature_names[sorted_idx], orientation="h")
```

**Teaching moment** — show the Plotly hover interactivity:
> "Your users get hover tooltips, zoom, pan, download as PNG — all for free. Compare that to matplotlib where you'd need a week of work."

---

## Section 2.5 — Matplotlib (5 min)

> "If you have existing matplotlib code from a notebook, you don't need to rewrite it. Just use `st.pyplot(fig)`."

Show one quick seaborn heatmap example. Then:

> "One critical thing: always call `plt.close(fig)` after `st.pyplot`. Otherwise matplotlib keeps a reference to every figure and your app leaks memory. This is a real production issue."

---

## Section 2.6 — File Upload Pipeline (10 min)

This is the most important pattern in Part 2. Type it from scratch:

```python
file = st.file_uploader("Upload CSV", type=["csv"])

if file is not None:
    df = pd.read_csv(file)
    st.success(f"Loaded {len(df)} rows")
    st.dataframe(df)
```

Then extend it:
1. Add column validation
2. Add the predict button
3. Add the download button

> "This is the full ML app loop: upload → validate → predict → download. Once you have this pattern, you can swap out any model in the middle."

**Common mistake**: accessing `file` without the `if file is not None` guard. The app crashes with `AttributeError: 'NoneType' object has no attribute 'read'`.

---

## Lab (10 min)

Open `part2_demo_app.py`. Students run it first, then extend:

> "Your challenge: add a column selector that lets the user pick which columns to use as X and Y for a scatter plot. Then add color-coding by a categorical column of their choice."

Hint: `st.selectbox` returning a column name, then passing to `px.scatter(x=col_x, y=col_y, color=col_color)`.

---

## Q&A Prompts

1. "What happens if a user uploads a 500MB CSV?" → The file is held in memory. Add a file size check: `if uploaded_file.size > 50 * 1024 * 1024: st.error("File too large")`

2. "Can I use Altair or Bokeh instead of Plotly?" → Yes. `st.altair_chart`, `st.bokeh_chart`. Plotly is the most feature-rich.

3. "Does caching persist when I redeploy the app?" → No. Cache is in-memory, reset on restart. For persistent caching, use a database or Redis.

---

## Timing Checklist

| Section | Target Time |
|---------|-------------|
| Opening + demo | 3 min |
| DataFrames | 8 min |
| Caching | 12 min |
| Built-in charts | 5 min |
| Plotly | 15 min |
| Matplotlib | 5 min |
| File upload pipeline | 10 min |
| Lab + Q&A | 2 min buffer |
| **Total** | **60 min** |
