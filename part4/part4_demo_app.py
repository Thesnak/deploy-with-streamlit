"""
Part 4 Demo App — Deep Learning Deployment
Run with: streamlit run part4_demo_app.py

Install:
  pip install streamlit transformers torch torchvision pillow plotly pandas

Note: First run downloads models (~500MB). Use @st.cache_resource to load once.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image

st.set_page_config(
    page_title="DL Model Hub",
    page_icon="🧠",
    layout="wide"
)

# ─── Lazy model loaders (only imported when tab is selected) ───────────────────
@st.cache_resource(show_spinner="Loading sentiment model...")
def load_sentiment():
    from transformers import pipeline
    return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

@st.cache_resource(show_spinner="Loading summarizer...")
def load_summarizer():
    from transformers import pipeline
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource(show_spinner="Loading NER model...")
def load_ner():
    from transformers import pipeline
    return pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english",
                    aggregation_strategy="simple")

@st.cache_resource(show_spinner="Loading zero-shot model...")
def load_zero_shot():
    from transformers import pipeline
    return pipeline("zero-shot-classification", model="typeform/distilbert-base-uncased-mnli")

@st.cache_resource(show_spinner="Loading image classifier...")
def load_image_model():
    """Load MobileNetV2 via torchvision (or TF if available)."""
    try:
        import torchvision.models as models
        import torch
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        model.eval()
        return ("pytorch", model)
    except ImportError:
        import tensorflow as tf
        model = tf.keras.applications.MobileNetV2(weights="imagenet")
        return ("tensorflow", model)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 DL Model Hub")
    st.markdown("---")
    st.markdown("""
**Models available**:
- 🖼️ MobileNetV2 — Image classification
- 💬 DistilBERT — Sentiment analysis
- 📝 DistilBART — Text summarization
- 🏷️ BERT-NER — Named Entity Recognition
- ✨ DistilBERT — Zero-shot classification
    """)
    st.markdown("---")
    st.caption("Part 4 — Deep Learning Deployment")
    st.caption("Models load on first use and are cached.")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.title("🧠 Deep Learning Model Hub")
st.markdown("Explore NLP and computer vision models, all deployed with Streamlit.")

tab_nlp, tab_vision, tab_compare = st.tabs(
    ["💬 NLP Models", "🖼️ Image Classification", "🆚 Model Comparison"]
)

# ──────────────────────────────────────────────────────────────────────────────
with tab_nlp:
    nlp_task = st.selectbox(
        "Choose NLP task",
        ["Sentiment Analysis", "Text Summarization", "Named Entity Recognition", "Zero-Shot Classification"]
    )

    # ── Sentiment Analysis ────────────────────────────────────────────────────
    if nlp_task == "Sentiment Analysis":
        st.subheader("💬 Sentiment Analysis")
        st.caption("Model: DistilBERT fine-tuned on SST-2 (~265MB)")

        col1, col2 = st.columns([3, 1])
        with col1:
            text = st.text_area("Enter text to analyze", height=120,
                                placeholder="Type or paste any text here...")
        with col2:
            st.markdown("**Example texts**")
            if st.button("Positive example"):
                st.session_state["sentiment_text"] = "This is absolutely fantastic! I love everything about it."
            if st.button("Negative example"):
                st.session_state["sentiment_text"] = "This is terrible. I'm completely disappointed."
            if st.button("Mixed example"):
                st.session_state["sentiment_text"] = "The food was good but the service was really slow."

        if "sentiment_text" in st.session_state:
            text = st.session_state["sentiment_text"]

        if st.button("Analyze Sentiment", type="primary") and text:
            model = load_sentiment()
            with st.spinner("Analyzing..."):
                result = model(text[:512])[0]

            label = result["label"]
            score = result["score"]

            if label == "POSITIVE":
                st.success(f"😊 **POSITIVE** — {score:.1%} confidence")
                st.progress(score, text=f"Positivity: {score:.1%}")
            else:
                st.error(f"😞 **NEGATIVE** — {score:.1%} confidence")
                st.progress(1 - score, text=f"Negativity: {score:.1%}")

    # ── Summarization ─────────────────────────────────────────────────────────
    elif nlp_task == "Text Summarization":
        st.subheader("📝 Text Summarization")
        st.caption("Model: DistilBART CNN (~820MB)")

        sample = """Machine learning is a branch of artificial intelligence (AI) and computer science which focuses on the use of data and algorithms to imitate the way that humans learn, gradually improving its accuracy. Machine learning is an important component of the growing field of data science. Through the use of statistical methods, algorithms are trained to make classifications or predictions, and to uncover key insights in data mining projects. These insights subsequently drive decision making within applications and businesses, ideally impacting key growth metrics. As big data continues to expand and grow, the market demand for data scientists will increase. They will be required to help identify the most relevant business questions and the data to answer them."""

        text = st.text_area("Paste article or long text", value=sample, height=200)
        col_max, col_min = st.columns(2)
        max_len = col_max.slider("Max summary tokens", 50, 300, 130)
        min_len = col_min.slider("Min summary tokens", 10, 100, 30)

        if st.button("Summarize", type="primary") and len(text) > 100:
            summarizer = load_summarizer()
            with st.spinner("Summarizing..."):
                result = summarizer(text[:1024], max_length=max_len, min_length=min_len, do_sample=False)[0]

            st.subheader("Summary")
            st.info(result["summary_text"])

            original_words = len(text.split())
            summary_words  = len(result["summary_text"].split())
            col1, col2, col3 = st.columns(3)
            col1.metric("Original words", original_words)
            col2.metric("Summary words", summary_words)
            col3.metric("Compression", f"{summary_words/original_words:.0%}")

    # ── NER ───────────────────────────────────────────────────────────────────
    elif nlp_task == "Named Entity Recognition":
        st.subheader("🏷️ Named Entity Recognition")
        st.caption("Model: BERT-large fine-tuned on CoNLL-2003")

        sample_ner = "Apple CEO Tim Cook met with President Biden at the White House in Washington D.C. on Monday to discuss AI regulation."
        text = st.text_area("Enter text", value=sample_ner, height=100)

        if st.button("Extract Entities", type="primary") and text:
            ner_model = load_ner()
            with st.spinner("Extracting entities..."):
                entities = ner_model(text)

            if not entities:
                st.warning("No named entities found.")
            else:
                st.success(f"Found {len(entities)} entities")

                entity_df = pd.DataFrame([{
                    "Entity": e["word"],
                    "Type": e["entity_group"],
                    "Confidence": f"{e['score']:.1%}"
                } for e in entities])

                col1, col2 = st.columns([2, 3])

                with col1:
                    st.dataframe(entity_df, use_container_width=True)

                with col2:
                    type_counts = entity_df["Type"].value_counts().reset_index()
                    type_counts.columns = ["Type", "Count"]
                    fig = px.bar(type_counts, x="Type", y="Count",
                                 title="Entity Types",
                                 color="Type")
                    st.plotly_chart(fig, use_container_width=True)

    # ── Zero-shot ─────────────────────────────────────────────────────────────
    elif nlp_task == "Zero-Shot Classification":
        st.subheader("✨ Zero-Shot Classification")
        st.caption("Classify text into any categories — no training needed!")

        text = st.text_area("Text to classify",
                             value="The new electric vehicle can travel 400 miles on a single charge.",
                             height=100)
        labels_raw = st.text_input(
            "Candidate labels (comma-separated)",
            value="technology, environment, sports, politics, economy"
        )
        labels = [l.strip() for l in labels_raw.split(",") if l.strip()]

        if st.button("Classify", type="primary") and text and labels:
            zs_model = load_zero_shot()
            with st.spinner("Classifying..."):
                result = zs_model(text[:512], labels)

            df = pd.DataFrame({"Label": result["labels"], "Score": result["scores"]})
            df = df.sort_values("Score", ascending=True)

            fig = px.bar(df, x="Score", y="Label", orientation="h",
                         title="Zero-Shot Classification Scores",
                         color="Score", color_continuous_scale="Viridis")
            fig.update_layout(coloraxis_showscale=False, xaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            top = result["labels"][0]
            top_score = result["scores"][0]
            st.success(f"Top prediction: **{top}** ({top_score:.1%})")

# ──────────────────────────────────────────────────────────────────────────────
with tab_vision:
    st.subheader("🖼️ Image Classification")
    st.markdown("Upload an image — MobileNetV2 classifies it into 1000 ImageNet categories.")

    uploaded_img = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "webp"],
                                    key="vision_upload")

    top_k = st.slider("Top K predictions", 3, 10, 5)

    if uploaded_img:
        image = Image.open(uploaded_img).convert("RGB")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption=f"Uploaded: {uploaded_img.name}", use_container_width=True)
            st.caption(f"Size: {image.size[0]}×{image.size[1]} px")

        with col2:
            framework, model = load_image_model()

            with st.spinner(f"Classifying with MobileNetV2 ({framework})..."):
                if framework == "pytorch":
                    import torch
                    import torchvision.transforms as T

                    transform = T.Compose([
                        T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    tensor = transform(image).unsqueeze(0)

                    with torch.no_grad():
                        logits = model(tensor)
                        probs  = torch.nn.functional.softmax(logits[0], dim=0)

                    # Load ImageNet labels
                    try:
                        from torchvision.models import MobileNet_V2_Weights
                        categories = MobileNet_V2_Weights.IMAGENET1K_V1.meta["categories"]
                    except Exception:
                        categories = [f"class_{i}" for i in range(1000)]

                    top_probs, top_indices = probs.topk(top_k)
                    results = [
                        {"class": categories[idx].replace("_", " ").title(),
                         "confidence": float(prob)}
                        for prob, idx in zip(top_probs.tolist(), top_indices.tolist())
                    ]

                else:  # tensorflow
                    import tensorflow as tf
                    img_resized = image.resize((224, 224))
                    arr = tf.keras.applications.mobilenet_v2.preprocess_input(
                        np.expand_dims(np.array(img_resized), 0)
                    )
                    preds = model.predict(arr, verbose=0)
                    decoded = tf.keras.applications.mobilenet_v2.decode_predictions(preds, top=top_k)[0]
                    results = [{"class": l.replace("_", " ").title(), "confidence": float(s)} for _, l, s in decoded]

            st.subheader(f"Top {top_k} Predictions")
            for r in results:
                st.progress(r["confidence"], text=f"{r['class']}: {r['confidence']:.1%}")

            fig = px.bar(
                x=[r["class"] for r in results[::-1]],
                y=[r["confidence"] for r in results[::-1]],
                title="Prediction Confidence",
                labels={"x": "Class", "y": "Confidence"},
                color=[r["confidence"] for r in results[::-1]],
                color_continuous_scale="Viridis"
            )
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

# ──────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.subheader("🆚 Sentiment Model Comparison")
    st.markdown("Run the same text through two different approaches and compare outputs.")

    text_compare = st.text_area("Text to analyze",
                                 value="The product works well but the customer support was very disappointing.",
                                 height=100, key="compare_text")

    if st.button("Compare Models", type="primary") and text_compare:
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**DistilBERT (fast, 265MB)**")
            model_a = load_sentiment()
            with st.spinner():
                import time
                t0 = time.time()
                res_a = model_a(text_compare[:512])[0]
                t1 = time.time()
            st.metric("Label", res_a["label"])
            st.metric("Confidence", f"{res_a['score']:.1%}")
            st.metric("Inference time", f"{(t1-t0)*1000:.0f}ms")

        with col2:
            st.markdown("**TextBlob (rule-based, no ML)**")
            try:
                from textblob import TextBlob
                t0 = time.time()
                blob = TextBlob(text_compare)
                polarity = blob.sentiment.polarity
                t1 = time.time()
                label_tb = "POSITIVE" if polarity > 0 else "NEGATIVE" if polarity < 0 else "NEUTRAL"
                st.metric("Label", label_tb)
                st.metric("Polarity score", f"{polarity:.3f}")
                st.metric("Inference time", f"{(t1-t0)*1000:.0f}ms")
            except ImportError:
                st.info("Install textblob for comparison: `pip install textblob`")
                st.markdown("TextBlob uses rule-based sentiment (polarity -1 to 1).")
                st.markdown("No ML model required — instant but less accurate.")

        st.info("**Key insight**: DistilBERT understands context and nuance. TextBlob works on keyword matching. Neural models win on complex sentences.")
