#!/usr/bin/env python3
"""
Exp 007 — Phase D: Router/Reducer cascade, mixed corpus (Watch + CasaSol).

Runs 4 pre-registered queries × 3 repeats per machine.
Evaluates grounding, completion time, and Router kind (plan vs question).

Queries:
    Q1  RHR monthly trend, last 12 months (Watch corpus, single-slice)
    Q2  Fencing session volume by year, full history (Watch corpus, workout aggregate)
    Q3  Nota Simple vs Catastro surface-area mismatch (CasaSol — deferred if index absent)
    Q4  "What were my best fitness years?" (Watch corpus, Router clarifying-question protocol)

Usage:
    python3 bench_phase_d.py --machine mini
    python3 bench_phase_d.py --machine mbp
    python3 bench_phase_d.py --machine mini --queries Q1 Q2 Q4  # skip Q3

Pre-conditions:
    - Ollama running with gemma4:26b and gemma4:e4b available
    - exp_005 Watch index exists (see EXP005_DIR below)
    - Phase A complete (cliff threshold known; 22K ceiling in effect unless overridden)

Evidence written to: evidence/<timestamp>-phase_d-<machine>/
"""

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

OLLAMA_CHAT = "http://localhost:11434/api/chat"
OLLAMA_GENERATE = "http://localhost:11434/api/generate"

ROUTER_MODEL = "gemma4:e4b"
REDUCER_MODEL = "gemma4:26b"

ROUTER_TIMEOUT_S = 120
REDUCER_FIRST_BYTE_S = 600
REDUCER_IDLE_S = 90
REDUCER_TOTAL_S = 1200
CHARS_PER_TOKEN = 4
CONTEXT_LIMIT_TOKENS = 22_000

NUM_REPEATS = 3

BASE_DIR = Path(__file__).resolve().parent
EVIDENCE = BASE_DIR / "evidence"

EXP005_DIR = BASE_DIR.parent / "exp_005_dicer_describer"
EXP005_INDEX = EXP005_DIR / "index"

QUERIES = {
    "Q1": "What has my resting heart rate trend been over the last 12 months? Show monthly values.",
    "Q2": "How much fencing have I done each year across my full history?",
    "Q3": None,  # CasaSol: populated if CasaSol index available
    "Q4": "What were my best fitness years?",
}

Q3_TEXT = (
    "Cross-reference the Nota Simple surface area against the Catastro registered area "
    "for each property in the database. Classify each discrepancy as: "
    "less than 10% (notary-correctable), 10% or more (Article 199 mandatory amendment), "
    "or classification conflict (Urbano/Rústico mismatch). "
    "List each property by Referencia Catastral with its discrepancy category."
)


# ---------------------------------------------------------------------------
# Ollama clients (adapted from exp_005/cascade.py)
# ---------------------------------------------------------------------------

def call_ollama_blocking(model: str, system: str, user: str,
                         timeout: float = 300.0,
                         response_format_json: bool = False) -> tuple[str, dict]:
    payload: dict = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
    }
    if response_format_json:
        payload["format"] = "json"
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_CHAT, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read())
    return body.get("message", {}).get("content", ""), body


def _stream_reader(resp, q: queue.Queue) -> None:
    try:
        for raw in resp:
            line = raw.strip()
            if not line:
                continue
            try:
                q.put(json.loads(line.decode("utf-8")))
            except Exception:
                continue
        q.put(None)
    except Exception as e:
        q.put({"_error": repr(e)})
        q.put(None)


def call_ollama_streaming(model: str, system: str, user: str,
                          first_byte_s: float, idle_s: float,
                          total_s: float) -> dict:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_CHAT, data=data,
                                 headers={"Content-Type": "application/json"})
    resp = urllib.request.urlopen(req, timeout=first_byte_s)
    q_: queue.Queue = queue.Queue()
    threading.Thread(target=_stream_reader, args=(resp, q_), daemon=True).start()

    chunks: list[str] = []
    first_event_at: float | None = None
    started = time.time()
    final_obj: dict = {}

    try:
        while True:
            elapsed = time.time() - started
            remaining = total_s - elapsed
            if remaining <= 0:
                raise TimeoutError(f"total budget {total_s}s exceeded")
            wait = min(first_byte_s if first_event_at is None else idle_s, remaining)
            try:
                obj = q_.get(timeout=wait)
            except queue.Empty:
                if first_event_at is None:
                    raise TimeoutError(f"no first byte in {first_byte_s}s")
                raise TimeoutError(f"idle for {idle_s}s")
            if obj is None:
                break
            if "_error" in obj:
                raise OSError(obj["_error"])
            if first_event_at is None:
                first_event_at = time.time() - started
            c = obj.get("message", {}).get("content", "")
            if c:
                chunks.append(c)
            if obj.get("done"):
                final_obj = obj
                break
    finally:
        try:
            resp.close()
        except Exception:
            pass

    return {
        "content": "".join(chunks),
        "first_event_at": first_event_at,
        "total_seconds": round(time.time() - started, 3),
        "final": final_obj,
    }


# ---------------------------------------------------------------------------
# Router / Reducer helpers
# ---------------------------------------------------------------------------

def normalise_json(raw: str) -> str:
    s = raw.strip()
    if s.startswith("```"):
        nl = s.find("\n")
        s = s[nl + 1:] if nl != -1 else s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def run_router(query: str, router_prompt: str, dicer_view: dict) -> tuple[str, float]:
    user = (
        "Corpus inventory:\n\n```json\n"
        + json.dumps(dicer_view, ensure_ascii=False, indent=2)
        + "\n```\n\nUser question:\n\n" + query.strip()
        + "\n\nReturn JSON only."
    )
    t0 = time.time()
    raw, _ = call_ollama_blocking(ROUTER_MODEL, router_prompt, user,
                                  timeout=ROUTER_TIMEOUT_S,
                                  response_format_json=True)
    elapsed = time.time() - t0
    return normalise_json(raw), round(elapsed, 3)


def run_extractor(plan_text: str) -> tuple[int, str, str]:
    cmd = [sys.executable, str(EXP005_DIR / "extract.py"),
           "--plan", "-", "--index-dir", str(EXP005_INDEX)]
    proc = subprocess.run(cmd, input=plan_text, capture_output=True,
                          text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def estimate_tokens(s: str) -> int:
    return len(s) // CHARS_PER_TOKEN


def guard_bundle(bundle: dict, reducer_prompt: str, query: str) -> tuple[dict, dict]:
    composed = reducer_prompt + "\n\n" + query + "\n\n" + json.dumps(bundle, ensure_ascii=False)
    initial = estimate_tokens(composed)
    report = {"initial_estimated_tokens": initial, "limit": CONTEXT_LIMIT_TOKENS, "truncated": False}
    if initial <= CONTEXT_LIMIT_TOKENS or bundle.get("kind") != "slice_bundle":
        return bundle, report
    bundle = json.loads(json.dumps(bundle))
    report["truncated"] = True
    overhead = estimate_tokens(reducer_prompt + "\n\n" + query + "\n\n{}")
    budget = max(0, CONTEXT_LIMIT_TOKENS - overhead)
    for _ in range(5):
        total = sum(estimate_tokens(json.dumps(s, ensure_ascii=False))
                    for s in bundle["slices"])
        if total <= budget:
            break
        factor = budget / total if total > 0 else 1.0
        for s in bundle["slices"]:
            for key in ("records", "workouts"):
                arr = s.get(key)
                if arr:
                    s[key] = arr[:max(20, int(len(arr) * factor))]
    report["final_estimated_tokens"] = estimate_tokens(
        reducer_prompt + "\n\n" + query + "\n\n" + json.dumps(bundle, ensure_ascii=False))
    return bundle, report


def run_reducer(query: str, reducer_prompt: str, bundle: dict) -> dict:
    user = (
        "User question:\n\n" + query.strip()
        + "\n\nSlice bundle:\n\n```json\n"
        + json.dumps(bundle, ensure_ascii=False, indent=2)
        + "\n```\n\nAnswer using only the data above."
    )
    return call_ollama_streaming(REDUCER_MODEL, reducer_prompt, user,
                                 first_byte_s=REDUCER_FIRST_BYTE_S,
                                 idle_s=REDUCER_IDLE_S,
                                 total_s=REDUCER_TOTAL_S)


# ---------------------------------------------------------------------------
# Per-query runner
# ---------------------------------------------------------------------------

def run_query(query_id: str, query_text: str,
              router_prompt: str, reducer_prompt: str, dicer_view: dict,
              machine: str, rep: int, ev_dir: Path) -> dict:
    started = time.time()
    ts = datetime.now(timezone.utc).isoformat()
    record: dict = {
        "experiment": "007", "phase": "D",
        "machine": machine, "query_id": query_id, "repeat": rep,
        "timestamp": ts, "query": query_text,
        "models": {"router": ROUTER_MODEL, "reducer": REDUCER_MODEL},
    }

    # Stage 1: Router
    try:
        plan_text, router_s = run_router(query_text, router_prompt, dicer_view)
        record["router_seconds"] = router_s
    except Exception as e:
        record["error"] = f"router: {e}"
        record["total_seconds"] = round(time.time() - started, 3)
        return record

    # Parse plan kind
    try:
        plan = json.loads(plan_text)
        kind = plan.get("kind", "unknown")
    except json.JSONDecodeError:
        plan = {}
        kind = "parse_error"
    record["router_kind"] = kind

    if kind == "question":
        record["ok"] = True
        record["router_question"] = plan.get("question")
        record["router_options"] = plan.get("options", [])
        record["grounded"] = len(plan.get("options", [])) >= 2
        record["total_seconds"] = round(time.time() - started, 3)
        return record

    # Stage 2: Extractor
    try:
        rc, stdout, stderr = run_extractor(plan_text)
        record["extractor_rc"] = rc
        if rc != 0:
            record["error"] = f"extractor: {stderr.strip()[:300]}"
            record["total_seconds"] = round(time.time() - started, 3)
            return record
        bundle = json.loads(stdout)
    except Exception as e:
        record["error"] = f"extractor exception: {e}"
        record["total_seconds"] = round(time.time() - started, 3)
        return record

    # Stage 2.5: Bundle guard
    guarded, guard_report = guard_bundle(bundle, reducer_prompt, query_text)
    record["bundle_guard"] = guard_report

    # Stage 3: Reducer
    try:
        t0 = time.time()
        result = run_reducer(query_text, reducer_prompt, guarded)
        record["reducer_seconds"] = round(time.time() - t0, 3)
        record["reducer_first_event_at"] = result.get("first_event_at")
        record["answer"] = result["content"]
        record["answer_chars"] = len(result["content"])
        record["ok"] = True
    except TimeoutError as e:
        record["error"] = f"reducer timeout: {e}"
        record["reducer_seconds"] = round(time.time() - started, 3)
        record["total_seconds"] = round(time.time() - started, 3)
        return record
    except Exception as e:
        record["error"] = f"reducer: {e}"
        record["total_seconds"] = round(time.time() - started, 3)
        return record

    record["total_seconds"] = round(time.time() - started, 3)
    record["grounded"] = None  # manual review required — flag in notes

    return record


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(machine: str, query_ids: list[str]) -> None:
    if not EXP005_DIR.exists():
        print(f"ERROR: exp_005 directory not found: {EXP005_DIR}")
        sys.exit(1)

    router_prompt_path = EXP005_DIR / "dicer_prompt.md"
    reducer_prompt_path = EXP005_DIR / "describer_prompt.md"
    dicer_view_path = EXP005_INDEX / "dicer_view.json"

    for path, label in [(router_prompt_path, "router prompt"),
                        (reducer_prompt_path, "reducer prompt"),
                        (dicer_view_path, "dicer_view")]:
        if not path.exists():
            print(f"ERROR: {label} not found: {path}")
            sys.exit(1)

    router_prompt = router_prompt_path.read_text(encoding="utf-8")
    reducer_prompt = reducer_prompt_path.read_text(encoding="utf-8")
    dicer_view = json.loads(dicer_view_path.read_text(encoding="utf-8"))

    # Q3: check for CasaSol index
    active_queries = {}
    for qid in query_ids:
        if qid == "Q3":
            # CasaSol index check — deferred if not available
            # (no fixed path yet; log deferral)
            print(f"  [Q3] CasaSol index not yet available — Q3 deferred per pre-registration.")
            continue
        if qid in QUERIES and QUERIES[qid] is not None:
            active_queries[qid] = QUERIES[qid]

    if not active_queries:
        print("No queries to run.")
        sys.exit(0)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    ev_dir = EVIDENCE / f"{timestamp}-phase_d-{machine}"
    ev_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nExp 007 — Phase D: Router/Reducer cascade")
    print(f"Machine: {machine}  |  Queries: {list(active_queries)}  |  Repeats: {NUM_REPEATS}")
    print(f"Router: {ROUTER_MODEL}  |  Reducer: {REDUCER_MODEL}")
    print(f"Evidence: {ev_dir}\n")

    all_records: list[dict] = []

    for qid, qtext in active_queries.items():
        print(f"  [{qid}] {qtext[:80]}...")
        query_records = []

        for rep in range(1, NUM_REPEATS + 1):
            print(f"         rep {rep}/{NUM_REPEATS}  ", end="", flush=True)
            rec = run_query(qid, qtext, router_prompt, reducer_prompt,
                            dicer_view, machine, rep, ev_dir)
            query_records.append(rec)
            all_records.append(rec)

            if rec.get("ok"):
                kind = rec.get("router_kind", "?")
                total_s = rec.get("total_seconds", "?")
                grounded = rec.get("grounded")
                ans_chars = rec.get("answer_chars", "")
                g_str = "?" if grounded is None else ("Y" if grounded else "N")
                print(f"ok  kind={kind}  grounded={g_str}  "
                      f"total={total_s}s  chars={ans_chars}", flush=True)
            else:
                print(f"FAILED — {rec.get('error','?')}", flush=True)

        (ev_dir / f"{qid}_records.json").write_text(
            json.dumps(query_records, indent=2, ensure_ascii=False))
        print()

    (ev_dir / "all_records.json").write_text(
        json.dumps(all_records, indent=2, ensure_ascii=False))

    summary = {
        "machine": machine,
        "models": {"router": ROUTER_MODEL, "reducer": REDUCER_MODEL},
        "started": timestamp,
        "queries_run": list(active_queries.keys()),
        "queries_deferred": [q for q in query_ids if q not in active_queries],
        "total_runs": len(all_records),
        "successful": sum(1 for r in all_records if r.get("ok")),
        "note": "grounded field requires manual review against index files",
    }
    (ev_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n=== Phase D Summary ===")
    for qid in active_queries:
        recs = [r for r in all_records if r.get("query_id") == qid]
        ok = sum(1 for r in recs if r.get("ok"))
        times = [r.get("total_seconds") for r in recs if r.get("total_seconds") is not None]
        mean_t = round(sum(times) / len(times), 1) if times else None
        print(f"  {qid}: {ok}/{NUM_REPEATS} ok  mean_total={mean_t}s")

    print(f"\nManual grounding review required — see {ev_dir}/")
    print(f"Check answer values against: {EXP005_INDEX}/\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--machine", required=True, choices=["mini", "mbp"],
                   help="Machine identifier")
    p.add_argument("--queries", nargs="+", choices=["Q1", "Q2", "Q3", "Q4"],
                   default=["Q1", "Q2", "Q3", "Q4"],
                   help="Which queries to run (default: all 4)")
    args = p.parse_args()
    run(machine=args.machine, query_ids=args.queries)
