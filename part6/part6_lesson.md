# Part 6 — Deployment and Production

In Part 6, we move from local demo apps to deployment-ready ML applications. The focus is on packaging, environment-aware model loading, and choosing the right hostess for your app.

---

## Goals
- Learn what differs between Streamlit Community Cloud and Hugging Face Spaces
- Build apps that can conditionally load lighter or heavier models
- Keep large model files out of git and use Hugging Face Hub or cloud storage
- Prepare an app for auto-deploy from git

---

## 6.1 Deployment targets

### Streamlit Community Cloud
- Free and easy to set up
- 1 GB RAM limit
- Ideal for lightweight sklearn apps and small HF models
- Auto-redeploys on git push

### Hugging Face Spaces
- Free tier includes 16 GB RAM
- Better for Transformers and larger inference workloads
- Works well with `transformers`, `diffusers`, and model repos

---

## 6.2 Environment-aware model loading

Your app can detect where it is running and adjust accordingly.

```python
import os

IS_HF_SPACES = os.environ.get("SPACE_ID") is not None
IS_STREAMLIT_CLOUD = os.environ.get("STREAMLIT_SHARING_MODE") is not None

if IS_STREAMLIT_CLOUD:
    model_name = "distilbert-base-uncased"
else:
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
```

This keeps your app stable on lower-memory hosts while still supporting more capable runtimes.

---

## 6.3 Minimal `requirements.txt`

Best practice: pin dependencies and keep them minimal.

```text
streamlit>=1.35.0,<2.0.0
scikit-learn>=1.4.0,<2.0.0
pandas>=2.0.0,<3.0.0
numpy>=1.26.0,<2.0.0
plotly>=5.20.0
joblib>=1.3.0
transformers>=4.40.0
torch
```

Only add `torch` if you actually use it. For purely sklearn apps, omit it.

---

## 6.4 Managing large models

- Small models (<50 MB): commit to repo
- Medium models (50–500 MB): store on Hugging Face Hub
- Large models (>500 MB): use `hf_hub_download()` or a cloud artifact store

```python
from huggingface_hub import hf_hub_download

@st.cache_resource
def load_model():
    path = hf_hub_download(
        repo_id="my-org/my-model",
        filename="pipeline.pkl",
        token=st.secrets["hf_api_token"]
    )
    return joblib.load(path)
```

---

## 6.5 Key deployment patterns

- Use `st.cache_resource` for models, database connections, and heavy initialization
- Use `st.cache_data` for datasets and computed metrics
- Use `st.stop()` to gate access until required resources are available
- Avoid loading all models at once on startup

---

## 6.6 Lab prompt

> Deploy the app to a hosted platform. If you choose Streamlit Cloud, ensure the app only loads the sklearn pipeline. If you choose HF Spaces, allow the NLP model to run as well.

### Checklist
- [ ] `requirements.txt` is pinned
- [ ] App detects the deployment environment
- [ ] Heavy model is loaded lazily
- [ ] The app runs with the selected cloud provider
