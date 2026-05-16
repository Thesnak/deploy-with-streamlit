# Part 5 — Advanced Streamlit Features

> **Goal**: Go beyond basic apps — multi-page navigation, secrets management, performance optimization, authentication, and production patterns for ML practitioners.

---

## 5.1 Multi-Page Apps

As your ML app grows, splitting into multiple pages keeps it maintainable. Streamlit has native multi-page support.

### Folder structure

```
my_ml_app/
├── app.py                    ← Entry point (home page)
├── pages/
│   ├── 1_📊_EDA.py           ← Page 1
│   ├── 2_🔮_Predict.py       ← Page 2
│   ├── 3_📈_Model_Metrics.py ← Page 3
│   └── 4_⚙️_Settings.py     ← Page 4
├── utils/
│   ├── model_utils.py
│   └── data_utils.py
├── models/
│   └── pipeline.pkl
└── requirements.txt
```

- Streamlit auto-detects files in `pages/` and adds them to the sidebar
- File name prefixes (`1_`, `2_`) control ordering
- Emojis in file names render in the nav sidebar
- Each page is a normal Python script

### app.py (home page)
```python
import streamlit as st

st.set_page_config(page_title="ML Platform", page_icon="🤖", layout="wide")

st.title("🤖 ML Deployment Platform")
st.markdown("""
Welcome! Navigate using the sidebar.

| Page | Purpose |
|------|---------|
| 📊 EDA | Explore your data |
| 🔮 Predict | Run inference |
| 📈 Model Metrics | Review performance |
| ⚙️ Settings | Configure the app |
""")
```

### Sharing state across pages

Session state persists across page navigation:

```python
# pages/1_📊_EDA.py — user uploads data
uploaded = st.file_uploader("Upload training data")
if uploaded:
    st.session_state["dataset"] = pd.read_csv(uploaded)

# pages/2_🔮_Predict.py — uses the uploaded data
if "dataset" not in st.session_state:
    st.warning("Please upload data on the EDA page first.")
    st.stop()

df = st.session_state["dataset"]
```

---

## 5.2 Secrets Management

Never hardcode API keys, database passwords, or credentials in your code.

### Local development — `.streamlit/secrets.toml`

```toml
# .streamlit/secrets.toml  ← NEVER commit this file!

# API Keys
openai_api_key = "sk-..."
hf_api_token   = "hf_..."

# Database
[database]
host     = "localhost"
port     = 5432
name     = "mldb"
user     = "admin"
password = "secret"

# Feature flags
[settings]
max_batch_size = 100
enable_logging = true
```

Add `.streamlit/secrets.toml` to `.gitignore`:
```bash
echo ".streamlit/secrets.toml" >> .gitignore
```

### Accessing secrets in code

```python
import streamlit as st

# Single values
api_key  = st.secrets["openai_api_key"]
hf_token = st.secrets["hf_api_token"]

# Nested (dict-style)
db_host  = st.secrets["database"]["host"]
```

### Practical patterns for ML apps

```python
# Hugging Face private model access
from transformers import pipeline

@st.cache_resource
def load_private_model():
    return pipeline(
        "text-classification",
        model="your-org/private-model",
        use_auth_token=st.secrets["hf_api_token"]
    )

# OpenAI API
import openai

def call_gpt(prompt: str) -> str:
    openai.api_key = st.secrets["openai_api_key"]
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content
```

---

## 5.3 Caching — Advanced Patterns

### Cache with parameters

```python
@st.cache_data
def load_and_filter(filepath: str, min_date: str, max_rows: int) -> pd.DataFrame:
    """Cache key includes all parameters — different args = different cache entry."""
    df = pd.read_csv(filepath)
    df = df[df["date"] >= min_date]
    return df.head(max_rows)

# Each unique combination of arguments gets its own cache entry
df = load_and_filter("data.csv", "2024-01-01", 1000)
```

### Excluding unhashable arguments

```python
# Underscore prefix = excluded from cache key (won't be hashed)
@st.cache_resource
def build_explainer(_model, X_train: pd.DataFrame):
    """Model object is excluded from key; DataFrame is included."""
    import shap
    return shap.TreeExplainer(_model)

explainer = build_explainer(pipeline.named_steps["classifier"], X_train)
```

### `st.cache_data` vs `st.cache_resource` — quick guide for ML

```python
# Always cache_resource:
@st.cache_resource
def load_tf_model(): ...

@st.cache_resource
def load_hf_pipeline(): ...

@st.cache_resource
def get_db_connection(): ...

# Always cache_data:
@st.cache_data
def load_dataset(path): ...

@st.cache_data(ttl=3600)  # Refresh hourly
def fetch_live_predictions(): ...

@st.cache_data
def compute_metrics(y_true, y_pred): ...
```

---

## 5.4 Performance Optimization

### Lazy loading — don't load all models at startup

```python
# Bad: loads ALL models when app starts
sentiment_model = load_sentiment()
vision_model    = load_vision()
nlp_model       = load_nlp()

# Good: load only when the user selects that task
task = st.selectbox("Task", ["Sentiment", "Vision", "NLP"])

if task == "Sentiment":
    model = load_sentiment()  # Loads on first select, cached after
elif task == "Vision":
    model = load_vision()
```

### Fragment rerunning — only rerun part of the page

```python
@st.fragment
def prediction_panel():
    """Only this fragment reruns when user changes inputs."""
    threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5)
    if st.button("Predict"):
        run_prediction(threshold)

# The chart below doesn't rerun when threshold changes
prediction_panel()
st.plotly_chart(expensive_global_chart)  # Runs once, not on every slider move
```

### Avoid recomputing heavy operations

```python
# Without optimization: SHAP values recomputed on every widget change
shap_values = compute_shap(model, X_input)  # 2 seconds every time

# With optimization: compute only when input changes
input_hash = hash(str(X_input.values.tolist()))
if st.session_state.get("last_input_hash") != input_hash:
    st.session_state["shap_values"] = compute_shap(model, X_input)
    st.session_state["last_input_hash"] = input_hash

shap_values = st.session_state["shap_values"]
```

---

## 5.5 Authentication

### Simple password protection (built-in)

For internal tools, Streamlit has a built-in auth mechanism via secrets:

```toml
# .streamlit/secrets.toml
[passwords]
# sha256 of the password
analyst_team = "8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918"
```

```python
import hashlib

def check_password():
    def verify():
        pw_hash = hashlib.sha256(st.session_state["password"].encode()).hexdigest()
        if pw_hash in st.secrets["passwords"].values():
            st.session_state["authenticated"] = True
        else:
            st.session_state["authenticated"] = False
            st.error("Incorrect password")

    if st.session_state.get("authenticated"):
        return True

    with st.form("login"):
        st.text_input("Password", type="password", key="password")
        st.form_submit_button("Login", on_click=verify)

    if not check_password():
        st.stop()

# Everything below only runs if authenticated
st.title("Internal ML Dashboard")
```

### OAuth / SSO via `streamlit-authenticator` package

```bash
pip install streamlit-authenticator
```

```python
import streamlit_authenticator as stauth

credentials = {
    "usernames": {
        "analyst1": {
            "name": "Alice Smith",
            "password": stauth.Hasher(["plaintext_pass"]).generate()[0],
            "email": "alice@company.com"
        }
    }
}

auth = stauth.Authenticate(credentials, cookie_name="ml_app", key="secret_key", cookie_expiry_days=7)
name, authenticated, username = auth.login("Login", "main")

if not authenticated:
    st.stop()

auth.logout("Logout", "sidebar")
st.write(f"Welcome, {name}!")
```

---

## 5.6 Forms — Batch Submit All Inputs at Once

Without forms, every widget change triggers a full rerun. With `st.form`, changes accumulate and only trigger on submit:

```python
with st.form("model_inputs"):
    st.subheader("Model inputs")

    age     = st.number_input("Age", 18, 100, 35)
    income  = st.number_input("Annual income", 0, 1_000_000, 60_000)
    score   = st.slider("Credit score", 300, 850, 700)
    purpose = st.selectbox("Loan purpose", ["Home", "Car", "Education", "Business"])

    # Form only submits when this button is clicked
    submitted = st.form_submit_button("💳 Predict loan approval", type="primary")

if submitted:
    with st.spinner("Running model..."):
        result = model.predict([[age, income, score, ["Home","Car","Education","Business"].index(purpose)]])
    st.success(f"Decision: **{'Approved' if result[0] == 1 else 'Denied'}**")
```

> **When to use forms**: When you have 5+ inputs and don't want the model to re-run on every single keystroke. Essential for heavy DL models.

---

## 5.7 Custom Themes

In `.streamlit/config.toml`:

```toml
[theme]
primaryColor         = "#FF4B4B"     # Button, slider, checkbox color
backgroundColor      = "#FFFFFF"     # Main area background
secondaryBackgroundColor = "#F0F2F6" # Sidebar background
textColor            = "#262730"     # Primary text
font                 = "sans serif"  # "sans serif", "serif", or "monospace"
```

For a dark data-science theme:
```toml
[theme]
primaryColor         = "#00D4FF"
backgroundColor      = "#0E1117"
secondaryBackgroundColor = "#1A1D23"
textColor            = "#FAFAFA"
```

---

## 5.8 Status and Progress Patterns

For long-running ML operations:

```python
# st.status — collapsible progress log
with st.status("Running experiment...", expanded=True) as status:
    st.write("Loading data...")
    df = load_data()

    st.write("Training model...")
    model = train_model(df)

    st.write("Evaluating...")
    metrics = evaluate(model)

    status.update(label="Experiment complete!", state="complete", expanded=False)

st.success(f"Final accuracy: {metrics['accuracy']:.2%}")
```

```python
# Multi-step progress for batch inference
```