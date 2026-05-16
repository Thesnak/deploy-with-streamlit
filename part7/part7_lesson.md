# Part 7 — Real-Time Deployment and Monitoring

In this final part, we focus on the last mile: observability. The app should not only predict, it should also track its own behavior and surface issues.

---

## Goals
- Build a real-time app workflow with single and batch inference
- Log predictions for auditability and drift detection
- Display monitoring metrics and simple charts
- Understand why deployed models need observability

---

## 7.1 Prediction logging

A simple JSONL log is one of the easiest monitoring primitives:

```python
with open("logs/predictions.jsonl", "a") as f:
    f.write(json.dumps({
        "timestamp": datetime.utcnow().isoformat(),
        "input": input_data,
        "result": result
    }) + "\n")
```

This is easy to parse with `pandas.read_json(..., lines=True)` or any log analysis tool.

---

## 7.2 Monitoring dashboard

The monitoring page should answer:
- How many predictions have been made?
- Are low-confidence predictions increasing?
- What did the most recent requests look like?

Example chart:

```python
st.line_chart(df_logs["result"].apply(lambda r: 1 if r["label"] == "Benign" else 0))
```

---

## 7.3 Real-time app patterns

- Use session state only for UI state, not logs
- Write logs to disk or a stream sink on each prediction
- Use `st.spinner` for user feedback during inference
- Avoid reading the full log file on every rerun if it is large; limit to recent entries

---

## 7.4 Lab challenge

> Extend the monitoring page with:
- A count of low-confidence predictions
- A rolling confidence plot for the last 20 predictions
- A warning if the last prediction confidence < 0.7

---

## 7.5 Deployment note

This app can be deployed to Hugging Face Spaces or Streamlit Community Cloud if you keep the model small. The current pipeline uses sklearn and saves a local model to `models/pipeline.pkl`.
