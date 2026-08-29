# Experiment 024 — Vision Capability: gemma4:26b vs qwen3.8:27b on Pharos's Actual Task

*Pre-registered: 2026-08-29 · Status: pre-registered, execution pending*

**Follows from:** exp_023 (generation efficiency) and the exp_012/015 audit-rubric comparison — both text-only. This extends "is gemma4:26b still the best local model" into vision, the one modality those experiments don't touch.

---

## What this experiment tests

Pharos (a separate local-first agentic-compute project, `~/REPOS/pharos`) runs a real production vision task: a Telegram maintenance-intake bot where tenants photograph a property issue (leak, crack, damp, broken fixture), and a vision leaf (`bot/pipeline.py::leaf_vision`) produces structured JSON — `visual_description`, `apparent_severity` (low/medium/high/critical), `location_hint`. This is currently served by `gemma4:26b` (`VISION_MODEL` in `bot/config.py`).

`qwen3.8:27b` (pulled for exp_015/exp_023) is also confirmed multimodal — `ollama show` lists a `vision` capability with a CLIP-based vision encoder — making a direct, real-task comparison possible without pulling anything new.

---

## Hypotheses

**H1 — JSON reliability:** Both models produce parseable JSON (per Pharos's own `_parse_json`/`_safe_parse` fallback logic) on ≥4/5 test photos, at `temperature=0`, `format=json`, matching Pharos's exact production call shape.

*Null:* one or both models fall back to the `_parse_error` path on ≥2/5 photos.

**H2 — Latency:** `qwen3.8:27b`'s vision latency is not dramatically worse than `gemma4:26b`'s (within the same order of magnitude), consistent with exp_023's finding that raw model size doesn't straightforwardly predict wall-clock time.

*Null:* `qwen3.8:27b` is >3x slower per photo than `gemma4:26b`.

**H3 — Output quality (qualitative, human-scored, not automated):** for the same photo, one model's `visual_description` is not obviously more specific/accurate than the other's on a majority of the test set. This is scored by direct human read of paired outputs, not a rubric — there's no ground-truth label for the substitute photos (see Design note below), so this hypothesis is inherently softer than H1/H2 and reported as an observation, not a pass/fail.

---

## Design

**Substitution note (documented up front, not discovered mid-run):** Pharos has no maintenance-issue sample photos in its repo (checked: only two branding PNGs exist). This experiment substitutes 5 real property photos from CasaSol's `witness/photos/` corpus (the GDPR-blurred versions, matching the sanitized-for-processing versions Pharos would actually see in production) as a same-domain stand-in — both tasks are "look at a property photo and describe its condition," even though CasaSol's photos are marketing-oriented rather than maintenance-issue photos. This is not the real Pharos task; it's the closest same-domain proxy available without staging fake maintenance photos. Results should be read as "how do these models handle property-photo vision tasks broadly," not as a validated Pharos production benchmark.

**Models:** `gemma4:26b` (current Pharos production), `qwen3.8:27b` (candidate).

**Photos:** 5 fixed photos from `casasol/witness/photos/*_blurred.jpg`, same 5 for both models.

**Call shape:** identical to Pharos's `_ollama_generate` — `/api/generate`, `format: "json"`, `temperature: 0`, `num_ctx: 4096`, `num_predict: 200`, `images: [base64]`, using Pharos's exact `VISION_PROMPT` text verbatim (unmodified — this is Pharos's prompt, not a new one written for this experiment).

**Per-photo, per-model:** record `total_ms`, `prompt_tokens`, `output_tokens`, `tok_per_s` (from the API response, same fields Pharos's own harness captures), whether the output parsed as valid JSON, and the parsed `visual_description`/`apparent_severity`/`location_hint` (or the raw text + parse error if it didn't parse).

---

## Evidence artefacts

```
results/
  run_YYYYMMDDTHHMMSSZ.json
```

---

*Experiment design: Andrei + Claude Sonnet 5 · 2026-08-29*
