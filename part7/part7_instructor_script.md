# Part 7 — Instructor Script
## Real-Time ML App and Monitoring

**Duration**: 55 minutes
**Target audience**: Students who want to deploy a real app with monitoring, logging, and resilience patterns.

---

## Opening (5 min)

Reinforce the narrative from earlier parts: by Part 7, the app is not just predictive — it should be observable.

Key takeaway:
- A deployed model without monitoring is just a toy.
- Data drift, prediction logs, and lightweight dashboards matter.

---

## Section 7.1 — Why monitoring matters (10 min)

Show three scenarios:
1. Input distributions shift over time
2. Users upload malformed data
3. Confidence falls but predictions continue

Explain that logging is the first step to debugging these failures.

---

## Section 7.2 — Prediction logging (10 min)

Live demo from the app:
- Single prediction logs to `logs/predictions.jsonl`
- Batch prediction logs also record the first 10 rows
- Logs are readable JSONL and easy to ship to Splunk / Datadog later

Discuss the tradeoff: file-based logging in local demos vs production log sinks.

---

## Section 7.3 — Monitoring dashboard (10 min)

Walk through the `Monitoring` tab.
- Total predictions count
- Line chart of benign/malignant prediction mix
- Recent prediction table

Explain that this is a simple real-time dashboard pattern; production systems use time windows and drift metrics.

---

## Section 7.4 — Real-time vs batch workloads (10 min)

Use the app modes to compare:
- Single prediction = low-latency, user-driven
- Batch prediction = throughput-oriented, file-driven
- Monitoring = observability layer

Talk about when to choose each architecture.

---

## Section 7.5 — Lab challenge (10 min)

> Add a simple alert for low-confidence predictions. If `confidence < 0.70`, show a red warning and count how many low-confidence predictions occurred in the log.

Ask students to add one new chart in `Monitoring` that plots rolling confidence over the last 20 predictions.

---

## Wrap-up

- Monitoring is essential for deployed AI
- Logging and dashboards keep models honest
- Small apps can still teach the right architecture
