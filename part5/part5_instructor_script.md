# Part 5 — Instructor Script
## Advanced Streamlit Features

**Total estimated time**: 60 minutes
**Audience note**: ML practitioners — skip motivational framing, go straight to patterns
**Assumed**: Parts 1–4. Students have working apps and want to productionize them.

---

## Opening (3 min)

No demo warmup needed — this audience moves fast.

> "Parts 1–4 got you to a working app. Part 5 gets you to a *shippable* app. We're talking multi-page navigation, secret management, auth, and the performance patterns that prevent your app from melting when someone uploads a 50k-row CSV."

Write these three problems on the board — you'll solve each one:
1. "My app is getting too big for one file"
2. "I have API keys in my code — that's bad"
3. "Every slider move reruns the entire model — it's too slow"

---

## Section 5.1 — Multi-Page Apps (12 min)

Live demo: create a `pages/` directory in the current project. Add one file. Show that Streamlit auto-detects it and adds it to the sidebar.

```bash
mkdir pages
touch "pages/1_📊_Data.py"
```

Just add one line to the new file:
```python
import streamlit as st
st.write("This is the data page")
```

Refresh the app. The sidebar now shows the new page.

> "That's it. No routing, no imports, no decorators. Streamlit reads the `pages/` directory and builds the nav automatically."

**File naming convention**:
> "The number prefix controls order. The emoji becomes the icon. Replace underscores with spaces. `1_📊_EDA.py` becomes '📊 EDA' in the sidebar."

**Session state across pages** — this is the key pattern for ML apps:
> "Session state is your shared memory between pages. User uploads a dataset on the EDA page, stores it in session state. The Predict page reads it. This replaces the need for a database for simple workflows."

Show the guard pattern:
```python
if "dataset" not in st.session_state:
    st.warning("Please upload data on the EDA page first.")
    st.stop()
```

> "`st.stop()` is one of the most underused functions in Streamlit. It halts execution immediately. Use it everywhere you need to gate access to functionality."

---

## Section 5.2 — Secrets (10 min)

Show the problem first:
```python
# This is in your GitHub repo. This is bad.
api_key = "sk-abc123..."
```

> "This is how API keys get rotated at 2am on a Friday. Someone commits a key, GitHub scans it, the provider revokes it. Let's not do this."

Live demo:
```bash
mkdir -p .streamlit
touch .streamlit/secrets.toml
echo ".streamlit/secrets.toml" >> .gitignore
```

Open secrets.toml, add:
```toml
hf_api_token = "hf_test_token_here"
```

In the app:
```python
token = st.secrets["hf_api_token"]
st.write(f"Token loaded: {token[:8]}...")
```

**Preview Streamlit Cloud secrets** — show a screenshot or describe:
> "On Streamlit Community Cloud, you add secrets in the app settings UI — same TOML format, but stored securely in their system. Your code doesn't change between local and cloud."

---

## Section 5.3 — Caching Deep Dive (8 min)

Students at this level often already know caching basics. Focus on the two patterns they haven't seen:

**1. The underscore prefix for unhashable args:**
```python
@st.cache_resource
def build_shap_explainer(_model, X_train):  # _model not hashed
    return shap.TreeExplainer(_model)
```

> "Streamlit caches based on argument values. ML model objects can't be hashed — they're complex C++ objects. Prefix with underscore to tell Streamlit 'don't try to hash this, trust me it's the same.'"

**2. TTL for live data:**
```python
@st.cache_data(ttl=300)  # 5-minute cache
def get_model_predictions_from_api():
    return requests.get("https://api.mymodel.com/latest").json()
```

> "If you're pulling from an API or database, TTL prevents stale data without requiring a full app restart."

---

## Section 5.4 — Performance: Fragments (10 min)

This is new-ish (added in Streamlit 1.33) and very powerful. Start by showing the problem:

```python
# Without fragments:
# User moves the threshold slider → ENTIRE PAGE reruns
# This includes the 3-second SHAP computation above it
threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
```

Add the fragment decorator:
```python
@st.fragment
def prediction_controls():
    threshold = st.slider("Threshold", 0.0, 1.0, 0.5)
    st.metric("Prediction", "Positive" if threshold > 0.5 else "Negative")

# This runs once at startup, not again when threshold changes:
expensive_shap_plot()

# This only reruns when the slider moves:
prediction_controls()
```

> "Fragments scope the rerun. The SHAP plot above it stays frozen. The slider reruns only the fragment. For DL apps where model loading takes 10 seconds, this is the difference between usable and unusable."

---

## Section 5.5 — Authentication (8 min)

Show the simple password approach first — appropriate for internal tools:

```python
def check_password():
    if st.session_state.get("authenticated"):
        return True
    with st.form("login"):
        st.text_input("Password", type="password", key="password")
        submitted = st.form_submit_button("Login")
    if submitted:
        if st.session_state["password"] == st.secrets["app_password"]:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Wrong password")
    return False

if not check_password():
    st.stop()

# Everything below only runs if authenticated
st.title("Internal ML Dashboard")
```

> "For an internal tool used by your data team — 10 people, all trusted — this is plenty. For external users, use `streamlit-authenticator` or offload auth to a proper identity provider."

Quick mention of `streamlit-authenticator`:
> "The package handles hashed passwords, session cookies, logout, and user roles. See the lesson notes for the full integration. Takes about 30 minutes to add."

---

## Section 5.6 — Forms (5 min)

Quick but important:

```python
# Without form: model reruns on EVERY keystroke/slider move
age    = st.number_input("Age", 18, 100)
income = st.slider("Income", 0, 500_000)
# ... model predicts here, reruns constantly

# With form: model only runs on Submit
with st.form("inputs"):
    age    = st.number_input("Age", 18, 100)
    income = st.slider("Income", 0, 500_000)
    submitted = st.form_submit_button("Predict")

if submitted:
    with st.spinner("Running model..."):
        result = model.predict([[age, income, score, ["Home","Car","Education","Business"].index(purpose)]])
    st.success(f"Decision: **{'Approved' if result[0] == 1 else 'Denied'}**")
```

> "Any time you have a TF/HF model and 4+ inputs, wrap them in a form. Your users will thank you."

---

## Lab Introduction (4 min)

> "You have a working classifier from Part 3. Your task: split it into a multi-page app. App.py as the home. A predict page. A metrics page that requires a password. Pass the uploaded data between pages via session state."

Key challenge: the metrics page needs access to the model, which is defined in utils/. Point students toward the centralized model loader pattern in section 5.9.

---

## Timing Checklist

| Section | Target |
|---------|--------|
| Opening | 3 min |
| Multi-page apps | 12 min |
| Secrets management | 10 min |
| Caching deep dive | 8 min |
| Fragments (performance) | 10 min |
| Authentication | 8 min |
| Forms | 5 min |
| Lab intro + Q&A | 4 min |
| **Total** | **60 min** |

---

## Questions to Expect

**Q: Can I have nested pages (sub-navigation)?**
A: Not natively. You can fake it with `st.selectbox` inside a page, or use a third-party package like `streamlit-option-menu`.

**Q: How do I handle database connections across pages?**
A: `@st.cache_resource` — the connection is initialized once and shared across all users and pages. Use the secrets pattern for credentials.

**Q: Is `st.fragment` stable for production?**
A: Yes as of Streamlit 1.33+. Check your version: `streamlit version`. If on an older version, use session state + form patterns instead.

**Q: What if I want to pre-load models before the user even visits the page?**
A: Put the `@st.cache_resource` function call in `app.py` at module level. It runs on startup and is cached before any page is visited.
