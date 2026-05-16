# Streamlit ML/DL Deployment — Crash Course

> **Deploy machine learning and deep learning models as beautiful, interactive web apps — no frontend experience needed.**

---

## Who This Course Is For

This course is designed for **Python developers** who know the basics of Python and want to:
- Learn Streamlit from the ground up
- Deploy real ML and DL models as interactive web apps
- Publish those apps to the cloud for the world to use

**You don't need**: web development experience, React, HTML/CSS, or prior deployment knowledge.

**You do need**: Python 3.9+, familiarity with functions and libraries, and ideally some exposure to scikit-learn or similar.

---

## Course Structure

| Part | Title | Key Topics | Duration |
|------|-------|------------|----------|
| **Part 1** | Streamlit Fundamentals | Installation, widgets, layout, session state | ~90 min |
| **Part 2** | Data & Visualization | Dataframes, charts, file upload, caching | ~60 min |
| **Part 3** | Classic ML Deployment | Scikit-learn models, joblib, prediction pipelines | ~90 min |
| **Part 4** | Deep Learning Deployment | TF/Keras, Hugging Face Transformers | ~120 min |
| **Part 5** | Advanced Streamlit | Multi-page apps, auth, secrets, performance | ~60 min |
| **Part 6** | Cloud Deployment | Streamlit Community Cloud, requirements, CI/CD | ~90 min |
| **Part 7** | Hugging Face Spaces (Deep Dive) | Space creation, Hub models, GPU tiers, Docker, GitHub Actions sync | ~90 min |

**Total estimated time**: ~9.5 hours of instruction + hands-on labs

---

## Prerequisites & Setup

### Python Environment

```bash
# Create a virtual environment (recommended)
python -m venv streamlit-env

# Activate it
# macOS/Linux:
source streamlit-env/bin/activate
# Windows:
streamlit-env\Scripts\activate

# Install core dependencies
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn joblib pillow
```

### Optional (for DL parts)
```bash
pip install tensorflow torch torchvision transformers datasets
```

### Verify installation
```bash
streamlit hello
```
This opens a demo app in your browser. If you see it — you're ready.

---

## How Each Part Is Organized

Every part includes three files:

```
partN_lesson.md          ← Full lesson: concepts, code, diagrams
partN_instructor_script.md ← Talking points, timing, Q&A guide
partN_demo_app.py        ← Runnable Streamlit app for live demos
```

### How to run a demo app
```bash
streamlit run part1_demo_app.py
```

---

## Learning Outcomes

By the end of this course you will be able to:

1. Build full-featured Streamlit apps with rich UI and interactivity
2. Load and serve scikit-learn models (classification, regression, clustering)
3. Deploy image classifiers, NLP models, and generative AI via Streamlit
4. Handle file uploads, preprocessing, and real-time inference
5. Add caching, authentication, and multi-page navigation
6. Publish your apps to Streamlit Community Cloud and Hugging Face Spaces
7. Use Docker basics to containerize Streamlit apps

---

## Recommended Progression

```
Part 1 → Part 2 → Part 3 → Part 4 → Part 5 → Part 6 → Part 7
  ↑          ↑        ↑        ↑                  ↑        ↑
 Must     Must    Required  Required           Cloud    HF Spaces
                                               Intro    Deep Dive
```

Parts 1 and 2 are required before any model-deployment parts.
Parts 5, 6, and 7 can be taken after Part 3.
Part 7 (HF Spaces deep dive) extends Part 6 — take Part 6 first.

---

## Repository Structure

```
streamlit-ml-course/
├── 00_course_overview.md
├── part1_lesson.md
├── part1_instructor_script.md
├── part1_demo_app.py
├── part2_lesson.md
├── part2_instructor_script.md
├── part2_demo_app.py
├── part3_lesson.md
├── part3_instructor_script.md
├── part3_demo_app.py
├── part4_lesson.md
├── part4_instructor_script.md
├── part4_demo_app.py
├── part5_lesson.md
├── part5_instructor_script.md
├── part5_demo_app.py
├── part6_lesson.md
├── part6_instructor_script.md
├── part6_demo_app.py
├── part7_lesson.md
├── part7_instructor_script.md
├── part7_demo_app.py
├── part7_hf_spaces_cheatsheet.md  ← One-page HF Spaces quick reference
├── models/                        ← Saved models used in demos
├── data/                          ← Sample datasets
└── requirements.txt
```

---

## Quick Reference Card

| Task | Streamlit Command |
|------|-------------------|
| Run app | `streamlit run app.py` |
| Stop app | `Ctrl + C` |
| Clear cache | `Ctrl + R` in browser, or `st.cache_data.clear()` |
| Show rerun | Automatic on widget interaction |
| Set page config | `st.set_page_config(title="...", layout="wide")` |
| HF login | `huggingface-cli login` |
| Create HF Space | `huggingface-cli repo create my-app --type space --space_sdk streamlit` |
| Pause HF Space | `huggingface-cli space pause YOUR_USER/my-app` |

---

## Resources

- [Streamlit Docs](https://docs.streamlit.io)
- [Streamlit Community Forum](https://discuss.streamlit.io)
- [Hugging Face Model Hub](https://huggingface.co/models)
- [HF Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Streamlit Community Cloud](https://streamlit.io/cloud)
- [Create a HF Space](https://huggingface.co/new-space)
- [HF Hardware Pricing](https://huggingface.co/docs/hub/spaces-gpus)

---

*Course designed for ML practitioners. All demo apps are self-contained and runnable with `streamlit run`. Part 7 requires a Hugging Face account (free).*
