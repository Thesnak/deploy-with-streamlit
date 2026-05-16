# Part 3 — Instructor Script
## Classic ML Deployment

**Total estimated time**: 90 minutes
**Format**: Live coding, train → deploy pipeline
**Assumed**: Parts 1 & 2 complete. Scikit-learn familiarity assumed.

---

## Opening (5 min)

> "This is where everything clicks. In Parts 1 and 2 we learned UI and data. Now we connect a real trained model to a real UI and let someone use it in a browser. In 90 minutes you'll have deployed your first ML model."

Show `part3_demo_app.py` running — particularly the batch prediction tab. Let students see predictions updating in real time.

---

## Section 3.1 — Pipeline Pattern (10 min)

Draw the deployment pipeline on the board:

```
OFFLINE:                       ONLINE (Streamlit):
Train → Evaluate → Save   →   Load → Input Form → Preprocess → Predict → Display
```

Key message:
> "The offline part (training) and the online part (serving) are completely separate. When you deploy, you never retrain — you just load the saved artifact."

**Why Pipelines?**
> "Here's the most common bug in ML deployment: you train with StandardScaler on the training data, then you forget to apply it in production. The model gets unscaled inputs and returns garbage. sklearn Pipelines solve this — the scaler and model are bundled together. One object, one `.predict()` call."

Live demo: show two approaches side-by-side.

**Bad approach**:
```python
scaler = StandardScaler()
model = RandomForestClassifier()
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled, y)
joblib.dump(model, "model.pkl")   # Forgot to save the scaler!
```

**Good approach**:
```python
pipeline = Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier())])
pipeline.fit(X, y)
joblib.dump(pipeline, "pipeline.pkl")  # Everything in one file
```

---

## Section 3.2 — Training Script (10 min)

Open a new terminal. Create `train_model.py`. Type it live (don't paste) to show it's just regular sklearn:

```python
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib, os

X, y = load_breast_cancer(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
])
pipeline.fit(X_train, y_train)
print(f"Accuracy: {pipeline.score(X_test, y_test):.4f}")

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/pipeline.pkl")
```

Run it: `python train_model.py`. Show the output.

> "That's our offline step. Takes about 2 seconds. Now we have a .pkl file. The rest of today is about building the Streamlit frontend that loads this file."

---

## Section 3.3 — Loading in Streamlit (8 min)

Show `@st.cache_resource` in context:

```python
@st.cache_resource
def load_pipeline():
    return joblib.load("models/pipeline.pkl")

pipeline = load_pipeline()
```

Show the terminal output. Run the app. Show that "Loading..." only prints once.

> "This single decorator is the difference between an app that takes 0.1 seconds to respond and one that takes 3 seconds every time a user moves a slider."

---

## Section 3.4 — Input Form (15 min)

This is where students struggle most. The challenge: breast cancer dataset has 30 features. We need 30 inputs.

Show the naive approach first (30 individual `st.number_input` calls):
> "You could write 30 number_input calls. That works but it's not maintainable. Here's a better pattern."

Show the loop-based form generation:
```python
inputs = {}
cols = st.columns(3)
for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        inputs[feature] = st.number_input(feature, value=default_values[feature])
```

**Key insight**:
> "We're generating the UI from data — the feature names list. If the model changes and gets more features, the UI updates automatically. This is a fundamental pattern in ML apps."

Show `X.describe()` to get min/max/mean values to use as slider defaults.

---

## Section 3.5 — Predictions & Display (12 min)

Walk through:
1. `pipeline.predict()` → class label
2. `pipeline.predict_proba()` → probabilities
3. Display with `st.metric` + plotly bar chart

**Common confusion**: `predict_proba()` returns shape `(n_samples, n_classes)`. For a single sample:
```python
probas = pipeline.predict_proba(X_input)[0]   # [0] gets the first (only) row
```

Show the confidence warning pattern:
```python
if probas.max() < 0.7:
    st.warning("Low confidence — be cautious with this prediction")
```

> "This kind of UX detail — flagging low-confidence predictions — is what separates a research demo from a production tool. Your users will trust the app more if it tells them when it's uncertain."

---

## Section 3.6 — Feature Importance (8 min)

```python
model = pipeline.named_steps["clf"]
importances = model.feature_importances_
```

Show the sorted horizontal bar chart. Ask students:
> "Why do we sort descending? Because the most important feature should be at the top, where the eye goes first."

Brief mention of SHAP:
> "SHAP takes this further — instead of global importance, it shows per-prediction importance. 'For this specific patient, *these* features drove the prediction.' That's much more clinically useful. See the lesson notes for the full SHAP implementation."

---

## Section 3.7 — Batch Prediction (10 min)

This is the pattern that makes ML apps actually useful for business users:

> "Most business use cases involve batch prediction. The user has a spreadsheet of 500 customers. They upload it, get predictions, download the results. Let's build that."

Walk through the file upload → predict → download loop. Emphasize:
1. Validate columns before running prediction
2. Add a progress spinner
3. The download button with `st.download_button`

**Demo moment**: Upload a real CSV, watch the predictions appear, download the result. This is always impressive for students.

---

## Lab (12 min)

> "Your task: adapt the demo app to use the Iris dataset instead of breast cancer. The Iris dataset has 3 classes (setosa, versicolor, virginica) and 4 features. You'll need to change the class names and the feature form."

Hint: `sklearn.datasets.load_iris(return_X_y=True, as_frame=True)`

Common issues:
- `predict_proba` now returns 3 columns, not 2 — the chart code needs to handle 3 classes
- Feature name formatting (underscores vs spaces)

---

## Timing Checklist

| Section | Target |
|---------|--------|
| Opening | 5 min |
| Pipeline pattern | 10 min |
| Training script | 10 min |
| Loading in Streamlit | 8 min |
| Input form | 15 min |
| Predictions & display | 12 min |
| Feature importance | 8 min |
| Batch prediction | 10 min |
| Lab + Q&A | 12 min |
| **Total** | **90 min** |

---

## Common Questions

**Q: Should I save the training data too?**
A: Yes — save `X_train` as a CSV for SHAP explanations and for computing feature min/max/mean defaults in the UI.

**Q: What if my model is too big to load quickly?**
A: Cache it with `@st.cache_resource`. If it's still slow (>5s), consider running inference in a separate FastAPI service and calling it from Streamlit.

**Q: Can I update the model without redeploying?**
A: You can, but it's better practice to deploy a new version. On Streamlit Cloud, pushing a new commit redeploys automatically.

**Q: What's the difference between joblib and pickle?**
A: joblib is faster for large numpy arrays (uses memory-mapped files). For sklearn models, always prefer joblib.
