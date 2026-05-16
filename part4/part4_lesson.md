# Part 4 — Deep Learning Deployment

> **Goal**: Deploy image classifiers (TF/Keras, PyTorch), NLP models (Hugging Face Transformers), and handle the unique challenges of DL inference in Streamlit.

---

## 4.1 DL vs Classic ML: Key Differences for Deployment

| Aspect | Scikit-learn | Deep Learning |
|--------|-------------|---------------|
| Model size | KBs–MBs | MBs–GBs |
| Load time | < 1 sec | 5–60 seconds |
| Inference time | Milliseconds | 0.1–10 seconds |
| Preprocessing | sklearn transformers | Custom (resize, tokenize, normalize) |
| GPU required? | Never | Sometimes (CPU is fine for inference) |
| Memory usage | Low | High (1–15 GB) |

This changes how we design Streamlit apps:
- **Caching is non-negotiable** — models take too long to load
- **Spinners are essential** — inference takes real time
- **Batch limits** — large batches may crash the app
- **Image/text preprocessing** requires custom code

---

## 4.2 Image Classification with TensorFlow/Keras

### Loading a pre-trained model

```python
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

@st.cache_resource
def load_model():
    """Load MobileNetV2 pre-trained on ImageNet. ~14MB, loads in ~3s."""
    model = tf.keras.applications.MobileNetV2(weights="imagenet")
    return model

model = load_model()
```

### Preprocessing for ImageNet models

```python
def preprocess_image(image: Image.Image, target_size=(224, 224)) -> np.ndarray:
    """Resize, convert, and normalize image for MobileNetV2."""
    # Ensure RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize
    image = image.resize(target_size)

    # Convert to array and add batch dimension
    img_array = np.array(image)          # Shape: (224, 224, 3)
    img_array = np.expand_dims(img_array, 0)  # Shape: (1, 224, 224, 3)

    # Normalize for MobileNetV2
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)

    return img_array
```

### Making predictions

```python
def classify_image(model, image: Image.Image, top_k=5):
    """Run inference and decode top-K ImageNet predictions."""
    img_array = preprocess_image(image)
    predictions = model.predict(img_array, verbose=0)  # verbose=0 suppresses logs

    # Decode ImageNet class labels
    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=top_k)[0]

    return [
        {"rank": i+1, "class": label.replace("_", " ").title(), "confidence": float(score)}
        for i, (_, label, score) in enumerate(decoded)
    ]
```

### Complete image classifier app

```python
st.title("🖼️ Image Classifier")
st.markdown("Upload an image — the model will classify it using 1000 ImageNet categories.")

model = load_model()

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"])

if uploaded:
    image = Image.open(uploaded)

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded image", use_container_width=True)

    with col2:
        with st.spinner("Classifying..."):
            results = classify_image(model, image, top_k=5)

        st.subheader("Top predictions")
        for r in results:
            st.progress(r["confidence"], text=f"{r['rank']}. {r['class']} ({r['confidence']:.1%})")
```

---

## 4.3 Image Classification with PyTorch

```python
import torch
import torchvision.transforms as transforms
from torchvision import models

@st.cache_resource
def load_pytorch_model():
    """Load ResNet50 pre-trained on ImageNet."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    model.eval()  # Important! Set to evaluation mode
    return model

@st.cache_data
def load_imagenet_labels():
    import json, urllib.request
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(url) as f:
        return json.load(f)

def preprocess_pytorch(image: Image.Image) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image.convert("RGB")).unsqueeze(0)  # Add batch dim

def classify_pytorch(model, image, labels, top_k=5):
    with torch.no_grad():
        tensor = preprocess_pytorch(image)
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs[0], dim=0)

    top_probs, top_indices = probs.topk(top_k)
    return [
        {"class": labels[idx].replace("_", " ").title(), "confidence": float(prob)}
        for prob, idx in zip(top_probs, top_indices)
    ]
```

---

## 4.4 Hugging Face Transformers — NLP

Hugging Face Transformers gives access to thousands of pre-trained NLP models.

### Sentiment Analysis

```python
from transformers import pipeline as hf_pipeline

@st.cache_resource
def load_sentiment_model():
    """Loads DistilBERT fine-tuned for sentiment. ~265MB."""
    return hf_pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

sentiment_model = load_sentiment_model()

# In the app:
text = st.text_area("Enter text for sentiment analysis", height=150)

if st.button("Analyze") and text:
    with st.spinner("Analyzing..."):
        result = sentiment_model(text)[0]

    label = result["label"]  # "POSITIVE" or "NEGATIVE"
    score = result["score"]

    if label == "POSITIVE":
        st.success(f"😊 Positive — {score:.1%} confidence")
    else:
        st.error(f"😞 Negative — {score:.1%} confidence")

    st.progress(score if label == "POSITIVE" else 1 - score, text="Positivity score")
```

---

## 4.5 Handling Large Models: Memory & Performance

### Memory management
```python
# For PyTorch: always use no_grad for inference
with torch.no_grad():
    output = model(input_tensor)

# For TensorFlow: limit GPU memory growth
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

### Model quantization (reduce size/speed)
```python
# Hugging Face — use quantized/distilled models
@st.cache_resource
def load_fast_model():
    # DistilBERT is 40% smaller, 60% faster than BERT
    return hf_pipeline("sentiment-analysis",
                       model="distilbert-base-uncased-finetuned-sst-2-english")
```

---

## 4.6 Custom-Trained DL Models

If you've trained your own Keras model:

```python
@st.cache_resource
def load_custom_model():
    # For SavedModel format:
    return tf.keras.models.load_model("models/my_cnn/")

def predict_custom(model, image):
    img = image.resize((128, 128)).convert("RGB")
    arr = np.array(img) / 255.0           # Normalize to [0, 1]
    arr = np.expand_dims(arr, 0)           # Add batch dim
    probs = model.predict(arr, verbose=0)[0]
    return {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}
```

---

## Lab: Multi-Modal ML App

Build an app with 3 tabs:
1. **Image** tab: Upload an image → classify with MobileNetV2 (TF) or ResNet50 (PyTorch)
2. **Text** tab: Enter text → sentiment analysis + NER
3. **Compare** tab: Run the same image through two different models and compare results

**Starter template**: see `part4_demo_app.py`

---

## Key Takeaways

- `@st.cache_resource` for models
- Lazy-load heavy models — don't import them until needed
- Use distilled or quantized variants for cloud deployment
- SHAP & explainability are important for trust
