# 🤗 Hugging Face Spaces — Quick Reference Cheatsheet

---

## Create & Deploy

```bash
# Install & login (once)
pip install huggingface_hub
huggingface-cli login

# Create a Space
git clone https://huggingface.co/spaces/YOUR_USER/my-app
cd my-app
cp /your/app.py .
cp /your/requirements.txt .
git add . && git commit -m "deploy" && git push

# Update after changes
git add . && git commit -m "update" && git push

# Pause / resume (stops billing on paid tiers)
huggingface-cli space pause  YOUR_USER/my-app
huggingface-cli space resume YOUR_USER/my-app
```

---

## Minimum Required Files

```
my-app/
├── app.py            ← entry point (must be app.py)
├── requirements.txt  ← all pip dependencies
└── README.md         ← Space card (title, emoji, sdk)
```

---

## README Front-Matter (Space Card)

```markdown
---
title: My ML App
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
license: mit
tags:
  - machine-learning
  - scikit-learn
---
# My ML App
Short description here.
```

---

## Secrets

```bash
HF Spaces UI: Space → Settings → Repository secrets → New secret
Key: HF_TOKEN    Value: hf_abc123...
Key: OPENAI_KEY  Value: sk-...
```

```python
# In app.py — portable, works locally and on HF Spaces
import os, streamlit as st

def get_secret(key: str, default: str = "") -> str:
    try:
        return st.secrets[key]           # local: secrets.toml
    except Exception:
        return os.environ.get(key, default)  # HF Spaces: env var

HF_TOKEN = get_secret("HF_TOKEN")
```

---

## Load Models from the Hub

```python
# Public model — no token needed
from transformers import pipeline
model = pipeline("sentiment-analysis",
                 model="distilbert-base-uncased-finetuned-sst-2-english")

# Private model — token required
from huggingface_hub import hf_hub_download
import joblib, os

path = hf_hub_download(
    repo_id="your-org/your-model",
    filename="pipeline.pkl",
    token=os.environ.get("HF_TOKEN")
)
model = joblib.load(path)
```

---

## Upload Your Model to the Hub

```python
from huggingface_hub import HfApi
api = HfApi()

# Create repo (once)
api.create_repo("your-user/my-model", repo_type="model", private=True)

# Upload file
api.upload_file(
    path_or_fileobj="models/pipeline.pkl",
    path_in_repo="pipeline.pkl",
    repo_id="your-user/my-model"
)
```

---

## requirements.txt Tips

```text
# Use tensorflow-cpu (not tensorflow) on CPU Spaces
tensorflow-cpu>=2.15.0,<3.0.0

# Pin major versions, allow minor
streamlit>=1.35.0,<2.0.0
scikit-learn>=1.4.0,<2.0.0
transformers>=4.40.0
torch
huggingface_hub>=0.22.0

# Generate with pipreqs (only what you import):
# pip install pipreqs && pipreqs . --force
```

---

## Hardware Tiers

| Tier | RAM | Cost/hr | Use case |
|------|-----|---------|---------|
| CPU Basic | 16 GB | **Free** | sklearn, DistilBERT, EfficientNet |
| CPU Upgrade | 32 GB | ~$0.03 | BART, Whisper |
| T4 Small | 15 GB VRAM | ~$0.60 | Stable Diffusion, LLaMA 7B |
| A10G Small | 24 GB VRAM | ~$1.05 | LLaMA 13B |

---

## GitHub → HF Spaces CI/CD

```yaml
# .github/workflows/deploy_to_hf.yml
name: Deploy to HF Spaces
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with: { fetch-depth: 0 }
      - name: Sync to Space
        env: { HF_TOKEN: ${{ secrets.HF_TOKEN }} }
        run: |
          git remote add space \
            https://YOUR_USER:$HF_TOKEN@huggingface.co/spaces/YOUR_USER/YOUR_SPACE
          git push space main --force
```

Add `HF_TOKEN` as a GitHub repo secret: Settings → Secrets → Actions.

---

## Docker Space (when you need system packages)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 7860
CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", "--server.address=0.0.0.0"]
```

README front-matter for Docker:
```yaml
sdk: docker
app_port: 7860
```

---

## Common Errors & Fixes

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError` | Add package to requirements.txt |
| `HTTPError 401` on model load | Add `HF_TOKEN` to Space secrets |
| `OSError: No space left` | Use `TRANSFORMERS_CACHE=/tmp/hf` or persistent storage |
| `MemoryError` | Use smaller model or upgrade hardware tier |
| Blank page on load | Check Logs tab for Python traceback |

---

## Useful Links

| Resource | URL |
|----------|-----|
| Create a Space | huggingface.co/new-space |
| HF Spaces docs | huggingface.co/docs/hub/spaces |
| Model Hub | huggingface.co/models |
| Your tokens | huggingface.co/settings/tokens |
| Space logs | huggingface.co/spaces/YOU/APP/logs |
| Hardware pricing | huggingface.co/docs/hub/spaces-gpus |
