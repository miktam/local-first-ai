#!/usr/bin/env python3
"""
Exp 006 — Redactor Fidelity Test: Batch Runner

Sends each fixture in fixtures/ through the redactor system prompt
and saves the output to results/output_NNN.json.

Usage:
    python3 run_batch.py
    python3 run_batch.py --model gemma4:26b
    python3 run_batch.py --fixture note_001 note_002  # run specific notes only

Pre-condition: Ollama must be running with gemma4:26b loaded.
"""

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent
FIXTURES    = BASE_DIR / "fixtures"
PROMPTS     = BASE_DIR / "prompts"
RESULTS     = BASE_DIR / "results"
OLLAMA_URL  = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4:26b"
TIMEOUT     = 300

RESULTS.mkdir(exist_ok=True)


def load_system_prompt() -> str:
    return (PROMPTS / "system.txt").read_text()


def call_ollama(system: str, note: str, model: str) -> dict:
    payload = json.dumps({
        "model": model,
        "system": system,
        "prompt": f"Redact this internal note:\n\n{note}",
        "stream": False,
        "options": {"temperature": 0.1},
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        raw = json.loads(resp.read())
    elapsed = time.time() - t0
    return {
        "response":        raw["response"].strip(),
        "wall_seconds":    round(elapsed, 1),
        "prompt_tokens":   raw.get("prompt_eval_count", None),
        "response_tokens": raw.get("eval_count", None),
    }


def run(model: str, only: list[str] | None) -> None:
    system  = load_system_prompt()
    fixtures = sorted(FIXTURES.glob("note_*.txt"))
    if only:
        fixtures = [f for f in fixtures if f.stem in only]

    print(f"\nExp 006 — Redactor Fidelity Test")
    print(f"Model: {model}  |  Fixtures: {len(fixtures)}  |  Started: {datetime.now():%H:%M:%S}\n")

    passed = 0
    failed = 0

    for fx in fixtures:
        note_id = fx.stem          # e.g. "note_001"
        note    = fx.read_text()
        out_path = RESULTS / f"output_{note_id[5:]}.json"  # output_001.json

        print(f"  [{note_id}] ", end="", flush=True)
        try:
            result = call_ollama(system, note, model)
            record = {
                "experiment":    "006",
                "fixture":       note_id,
                "model":         model,
                "timestamp":     datetime.now(timezone.utc).isoformat(),
                "wall_seconds":  result["wall_seconds"],
                "prompt_tokens": result["prompt_tokens"],
                "response_tokens": result["response_tokens"],
                "input":         note,
                "output":        result["response"],
            }
            out_path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
            status = "ok" if result["wall_seconds"] < TIMEOUT else "slow"
            print(f"{status}  ({result['wall_seconds']:.0f}s, {result['response_tokens']} tokens)")
            passed += 1
        except Exception as e:
            print(f"FAILED — {e}")
            (RESULTS / f"output_{note_id[5:]}.error.txt").write_text(str(e))
            failed += 1

    print(f"\nDone. {passed} succeeded, {failed} failed.")
    print(f"Results in: {RESULTS}")
    print("Next: python3 check_output.py\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=DEFAULT_MODEL)
    parser.add_argument("--fixture", nargs="+", metavar="NOTE_ID",
                        help="e.g. note_001 note_005 — run only these fixtures")
    args = parser.parse_args()
    run(model=args.model, only=args.fixture)
