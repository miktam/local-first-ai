#!/usr/bin/env python3
"""
Exp 024 — Vision capability: gemma4:26b vs qwen3.8:27b on Pharos's actual
vision task (facilities-photo description + severity/location extraction).

Reuses Pharos's exact VISION_PROMPT and /api/generate payload shape
(bot/pipeline.py::leaf_vision, _ollama_generate) verbatim — this is Pharos's
production call, not a new prompt written for this experiment.

Photos: 5 fixed GDPR-blurred property photos from casasol/witness/photos/,
substituting for maintenance-issue photos which don't exist anywhere in
Pharos's repo (documented in HYPOTHESIS.md).
"""
import base64
import json
import re
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OLLAMA_BASE = "http://localhost:11434"
MODELS = ["gemma4:26b", "qwen3.8:27b"]

VISION_PROMPT = """\
You are a facilities inspection assistant. Examine the photo and identify the maintenance issue.

Respond with JSON only:
{
  "visual_description": "<one or two sentences: what the issue is and where it appears>",
  "apparent_severity": "low|medium|high|critical",
  "location_hint": "<room or area visible in the photo, or 'unknown'>"
}"""

PHOTOS_DIR = Path("/Users/miktam02/REPOS/casasol/witness/photos")
PHOTOS = sorted(PHOTOS_DIR.glob("*_blurred.jpg"))[:5]

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ollama_generate_vision(model: str, image_b64: str) -> tuple[str, dict]:
    payload = {
        "model": model,
        "prompt": VISION_PROMPT,
        "images": [image_b64],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 200},
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{OLLAMA_BASE}/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.loads(r.read())
    wall_s = time.time() - t0
    prompt_tokens = data.get("prompt_eval_count", 0)
    output_tokens = data.get("eval_count", 0)
    eval_ns = data.get("eval_duration", 0)
    meta = {
        "total_ms": round(data.get("total_duration", 0) / 1_000_000),
        "wall_clock_s": round(wall_s, 2),
        "prompt_tokens": prompt_tokens,
        "output_tokens": output_tokens,
        "tok_per_s": round(output_tokens / max(eval_ns / 1e9, 0.001), 1),
    }
    return data["response"], meta


def parse_json_pharos_style(raw: str) -> tuple[dict | None, str | None]:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?", "", cleaned).removesuffix("```").strip()
    try:
        return json.loads(cleaned), None
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group()), None
            except json.JSONDecodeError as e:
                return None, str(e)
        return None, "no JSON object found"


def main():
    if not PHOTOS:
        print(f"[ERROR] no *_blurred.jpg photos found in {PHOTOS_DIR}")
        return

    print(f"Using {len(PHOTOS)} photos: {[p.name for p in PHOTOS]}")
    all_runs = []

    for model in MODELS:
        print(f"\n=== {model} ===")
        for photo in PHOTOS:
            image_b64 = base64.b64encode(photo.read_bytes()).decode()
            try:
                raw, meta = ollama_generate_vision(model, image_b64)
                parsed, parse_error = parse_json_pharos_style(raw)
                run = {
                    "model": model, "photo": photo.name,
                    "parsed_ok": parsed is not None,
                    "parse_error": parse_error,
                    "raw_response": raw[:500],
                    "visual_description": (parsed or {}).get("visual_description"),
                    "apparent_severity": (parsed or {}).get("apparent_severity"),
                    "location_hint": (parsed or {}).get("location_hint"),
                    **meta,
                }
                print(f"  {photo.name}: parsed={run['parsed_ok']} "
                      f"total_ms={meta['total_ms']} tok/s={meta['tok_per_s']} "
                      f"severity={run['apparent_severity']}")
            except Exception as e:
                run = {"model": model, "photo": photo.name, "error": str(e)}
                print(f"  {photo.name}: ERROR: {e}")
            all_runs.append(run)

    # ── Verdicts ──────────────────────────────────────────────────────────
    verdicts = {}
    for model in MODELS:
        model_runs = [r for r in all_runs if r["model"] == model and "error" not in r]
        n_parsed = sum(1 for r in model_runs if r["parsed_ok"])
        verdicts[model] = {
            "n_parsed_ok": n_parsed,
            "n_total": len(model_runs),
            "mean_total_ms": round(sum(r["total_ms"] for r in model_runs) / len(model_runs), 1) if model_runs else None,
            "mean_tok_per_s": round(sum(r["tok_per_s"] for r in model_runs) / len(model_runs), 1) if model_runs else None,
        }

    if len(MODELS) == 2 and all(verdicts[m]["mean_total_ms"] for m in MODELS):
        a, b = MODELS
        ratio = max(verdicts[a]["mean_total_ms"], verdicts[b]["mean_total_ms"]) / \
                min(verdicts[a]["mean_total_ms"], verdicts[b]["mean_total_ms"])
        verdicts["H2_latency_ratio"] = round(ratio, 2)
        verdicts["H2"] = "CONFIRMED" if ratio <= 3.0 else "REFUTED"

    result = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "photos": [p.name for p in PHOTOS],
        "photo_source_note": (
            "Substituted from casasol/witness/photos/ (GDPR-blurred property photos) — "
            "Pharos has no maintenance-issue sample photos in its own repo. See HYPOTHESIS.md."
        ),
        "prompt": VISION_PROMPT,
        "runs": all_runs,
        "verdicts": verdicts,
    }

    out_path = RESULTS_DIR / f"run_{TS}.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out_path}")
    print("\n=== VERDICTS ===")
    print(json.dumps(verdicts, indent=2))


if __name__ == "__main__":
    main()
