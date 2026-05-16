# Part 6 — Instructor Script
## Deployment and Production Patterns for ML Apps

**Duration**: 50 minutes
**Level**: Intermediate — assumes students completed Parts 1–5

---

## Opening (3 min)

Goal: get one working app into the cloud. Discuss the tradeoffs: memory, model size, infra.

Key decision: Streamlit Community Cloud (1GB) vs Hugging Face Spaces (16GB). Show example apps and when to choose which.

---

## Section 6.1 — Packaging your app (10 min)

- `requirements.txt` vs `pipfile` vs `poetry` — keep it small and pinned.
- Keep heavy model files out of git; use HF Hub for models >50MB.

Live demo: show a `requirements.txt` tuned for sklearn apps vs transformers apps.

---

## Section 6.2 — Environment detection (7 min)

Show the `IS_HF_SPACES` and `IS_STREAMLIT_CLOUD` patterns and how to conditionally load lighter models on Streamlit Cloud.

---

## Section 6.3 — Continuous deployment (10 min)

- Streamlit auto-deploys from GitHub pushes.
- HF Spaces deploy on git push to the space repo.
- Demo: push a small change and show auto redeploy.

---

## Section 6.4 — Model storage strategies (10 min)

- <50MB: commit to repo
- 50–500MB: HF Hub model repo
- >500MB: HF Hub and `hf_hub_download()` or cloud storage

---

## Section 6.5 — Monitoring and logging (5 min)

- Use simple health endpoints by returning small files
- Add minimal telemetry: request counts, average latencies
- Integrate Sentry or LogDNA for production apps

---

## Lab (5 min)

Deploy the class app to Streamlit or HF Spaces. Verify it loads, runs predictions, and that secrets are configured correctly.

---

## Wrap-up

- Emphasize keeping the repo small, decoupling model artifacts, and choosing the correct deployment target for model size.
