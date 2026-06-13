# Exp 002 — Managing the AI Thinking Tax (Control Plane vs Data Plane)

*Date: 2026-04-22*  
*Status: Complete*  
*Full log entry: [`scientific_log.md` § Experiment 002](../scientific_log.md)*

---

## Observation

Unconstrained use of thinking mode (`think: true`) on local hardware risks runaway token
generation — turning efficient local tasks into thermal events and connection timeouts.

## Hypothesis

By delegating high-level orchestration to the Control Plane (agent reasoning) and reserving
the Data Plane (model weights thinking) strictly for verified tasks, throughput remains high
and system exhaustion is prevented.

## Results

| Mode | OpenClaw Reasoning | Model Think | Latency (s) | Tokens | Throughput (t/s) |
|---|---|---|---|---|---|
| Assembly Line | false | false | 8.71 | 347 | ~39.84 |
| Auditor | true | false | 46.33 | 1754 | ~37.86 |
| Architect | true | true | 53.05 | 2003 | ~37.76 |
| **Architect (Edge Case)** | true | true | **>1200** | *Saturated* | **TIMEOUT** |

**Conclusion:** Throughput is stable at ~38 t/s across all bounded modes. The failure mode
is catastrophic, not gradual: an unmoored thinking loop saturated the KV cache in 20 minutes.
Strict Control Plane guardrails are mandatory for local operations.

## Incident 002-Alpha (2026-04-24)

During initial drafting, Nestor misidentified the latency gap as "exponential" while ignoring
stable throughput and the timeout. Human engineer triggered review; analysis recalibrated to
prioritise throughput consistency and flag the Edge Case as the true failure mode.

---

## Artifacts

- [`latency_benchmark_v2.py`](./latency_benchmark_v2.py) — benchmark script (three modes + edge case)
- [`latency_benchmark_v2.jsonl`](./latency_benchmark_v2.jsonl) — raw output
- Blog post: [The Control Plane and the Data Plane](https://localfirstai.eu/posts/2026-04-22-control-plane-vs-data-plane/)
