# Part 4 — Instructor Script
## Deep Learning Deployment

**Total estimated time**: 120 minutes
**Format**: Live demo-heavy (large model downloads take time — warn students in advance)
**Assumed**: Parts 1–3 complete. Basic familiarity with neural networks.

---

## Pre-class Setup (CRITICAL)

⚠️ **Tell students to download models before class starts.**

In chat or email beforehand:
```
Before class: run these commands to pre-download models (takes 5–15 min):
python -c "import tensorflow as tf; tf.keras.applications.MobileNetV2(weights='imagenet')"
python -c "from transformers import pipeline; pipeline('sentiment-analysis')"
```

Without this, the first live demo will have an 8-minute download pause that kills momentum.

---

## Opening (5 min)

> "Classic ML models — scikit-learn random forests, SVMs — are maybe a few megabytes. The model we'll load today is 265 megabytes, and BERT-large is 1.3 gigabytes. This changes everything about how you build the app."

Key message:
> "In DL deployment, 80% of the engineering effort is around the model loading — not the inference. Loading once, sharing across users, managing memory. That's what this session is about."

Show the table from the lesson (Classic ML vs DL comparison) — spend 3 minutes on it.

---

## Section 4.2 — Image Classification with TF/Keras (25 min)

**Start with the model card for MobileNetV2**:
> "MobileNetV2 is trained on 1.4 million images across 1000 categories. 14MB. Runs on CPU. This is the go-to model for demos — lightweight but surprisingly good."

**Live demo** — type from scratch:

```python
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights="imagenet")

model = load_model()

uploaded = st.file_uploader("Upload image", type=["jpg", "png", "jpeg"]) 
if uploaded:
    image = Image.open(uploaded)
    st.image(image, width=300)
    
    with st.spinner("Classifying..."):
        img = image.resize((224, 224)).convert("RGB")
        arr = tf.keras.applications.mobilenet_v2.preprocess_input(
            np.expand_dims(np.array(img), 0)
        )
        preds = model.predict(arr, verbose=0)
        results = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=5)[0]
    
    for _, label, score in results:
        st.progress(float(score), text=f"{label}: {score:.1%}")
```

Upload a photo of a dog. Watch the classification happen.

**Teaching moment — preprocessing is CRITICAL**:
> "Watch what happens if I don't apply the preprocessing function."

Comment out `preprocess_input`. Show the garbage output. Uncomment it. Show correct output again.

> "The model was trained on inputs with specific normalization — pixel values transformed to roughly [-1, 1]. If you feed raw [0, 255] pixels, it gets confused. This is the single most common bug in DL deployment."

---

## Section 4.3 — PyTorch (20 min)

> "PyTorch users: the pattern is the same, the syntax is different. The one thing you CANNOT forget is model.eval()."

Show what happens without `.eval()` — the model behaves differently (dropout active, batch norm uses batch stats instead of running stats). This is a real production bug.

```python
model = models.resnet50(weights=...)
# model.eval()  ← Comment this out to show the bug
```

The predictions won't be stable — each call gives slightly different results.

> "Dropout randomly zeros activations during training. In eval mode it's disabled. Without this line, your predictions are non-deterministic. You'll get different results every time you call the model."

**Also show `torch.no_grad()`**:
> "This tells PyTorch not to track gradients during inference. Gradients are needed for training (backprop) but are wasted memory in deployment. This can 2–3x your memory efficiency."

---

## Section 4.4 — Hugging Face (35 min)

This is the highlight of Part 4. Hugging Face transforms what used to be a multi-week ML engineering task into 3 lines.

**Start with sentiment analysis** (easiest, fastest):

```python
from transformers import pipeline

@st.cache_resource
def load_sentiment():
    return pipeline("sentiment-analysis")

model = load_sentiment()
text = st.text_area("Enter text")
if st.button("Analyze"):
    with st.spinner():
        result = model(text)[0]
    st.write(result)
```

Run it live. Type "This is absolutely incredible, I love this course!" Watch the output.

Then type "This is the worst product I've ever used." Pause for effect.

> "Three lines of code. State of the art NLP. Pre-trained on millions of sentences. This is why Hugging Face changed the ML industry."

**Zero-shot classification** (most impressive demo):

> "Here's the one that usually gets a 'wow' in the room. Zero-shot means you don't need to have trained the model on your specific categories."

```python
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
result = classifier(
    "Apple announced a new MacBook with M3 chips",
    candidate_labels=["technology", "politics", "sports", "cooking"]
)
```

Ask students: "What label do you think it will pick?" Show the output.

Then change the labels to something completely different:
```python
candidate_labels=["good news", "bad news", "neutral"]
```

> "Same model. Different labels. No retraining. This is zero-shot classification — the model understands language well enough to classify into any categories you give it."

**NER** (very visual, students love it):

After running NER, show the entity table. Then show how to highlight entities in the original text using colored text.

---

## Section 4.5 — Memory & Performance (10 min)

> "Here's a realistic scenario: you deploy your BERT app on Streamlit Community Cloud. They give you 1GB of RAM. BERT-base needs 440MB. BERT-large needs 1.3GB. Your app crashes."

Three solutions to discuss:
1. **Use distilled models**: DistilBERT = 40% smaller, 97% of BERT's performance
2. **Use quantized models**: `AutoModelForSequenceClassification` with 8-bit quantization via `bitsandbytes`
3. **Use API calls**: Call Hugging Face Inference API instead of loading the model locally

> "For demos and teaching, option 1 (distilled) is almost always the right choice."

---

## Lab (15 min)

> "Build the multi-modal app. Tab 1: image classification. Tab 2: sentiment + NER. The challenge: both models are loaded at startup. Make sure both are cached with `@st.cache_resource`."

Point to `part4_demo_app.py` as the reference.

Common issues:
- Forgetting `verbose=0` in `model.predict()` — TF prints a progress bar to terminal every inference call
- File type not being in the `type=[]` list in `st.file_uploader`
- PIL image mode issues (RGBA → RGB conversion)

---

## Timing Checklist

| Section | Target |
|---------|--------|
| Opening | 5 min |
| TF/Keras image classifier | 25 min |
| PyTorch image classifier | 20 min |
| HF Transformers (all) | 35 min |
| Memory & performance | 10 min |
| Lab | 15 min |
| Overflow / Q&A | 10 min |
| **Total** | **120 min** |

---

## Common Questions

**Q: How do I use a GPU in Streamlit?**
A: Streamlit Community Cloud and most hosting services don't provide GPUs. For GPU inference, deploy to a cloud VM with GPU (AWS g4dn.xlarge, ~$0.50/hr) or use a model serving API. For demos, CPU-only inference is usually fast enough.

**Q: Can I use Ollama / local LLMs?**
A: Yes — call `ollama.chat()` from Streamlit. But Ollama runs as a separate process. The Streamlit app calls it via HTTP. Works great locally; deployment requires Ollama to be running on the same server.

**Q: What about OpenAI / Anthropic APIs?**
A: Perfect fit for Streamlit. Use `st.secrets["OPENAI_API_KEY"]` to store the key securely (covered in Part 5). Then call the API in a button handler, show results in the UI.

**Q: Model loads take 20 seconds — can I show a loading screen?**
A: Yes, use `st.status()` which shows a live spinner with custom messages. Or preload the model before `st.set_page_config` (at module level) so it loads in the background during the initial page render.
