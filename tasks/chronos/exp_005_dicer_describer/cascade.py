#!/usr/bin/env python3
"""
cascade.py — Phase 0 orchestrator for Experiment 005.

End-to-end run for one user query:
    1. Send query + Dicer prompt + dicer_view.json to gemma4:e4b via Ollama.
    2. Normalise the Dicer's output (strip markdown code fences if present)
       per the ADR-001 amendment: normalisation at the protocol boundary is
       distinct from repair of plan content.
    3. Validate via extract.py (strict; ADR-001).
    4. Token-guard the bundle.
    5. STREAM query + Describer prompt + slice_bundle to gemma4:26b. Reader
       thread parses NDJSON; main thread enforces idle timeout via
       queue.get(timeout=...).
    6. Print the final answer; log everything to runs/<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

OLLAMA_URL = "http://localhost:11434/api/chat"
DICER_MODEL = "gemma4:e4b"
DESCRIBER_MODEL = "gemma4:26b"

DICER_RETRY_LIMIT = 1                    # ADR-001
DICER_TIMEOUT_S = 120                    # warm: ~3-5s; cold: ~30s
DESCRIBER_IDLE_TIMEOUT_S = 90            # max gap between any two streamed events
DESCRIBER_TOTAL_BUDGET_S = 1800          # absolute ceiling
DESCRIBER_CONTEXT_LIMIT_TOKENS = 200_000 # under 26b's 262k ceiling
CHARS_PER_TOKEN = 4


# ---------------------------------------------------------------------------
# Ollama client — non-streaming (Dicer)
# ---------------------------------------------------------------------------

def call_ollama_blocking(model: str, system: str, user: str, *,
                         response_format_json: bool = False,
                         timeout: float = 300.0) -> tuple[str, dict[str, Any]]:
    payload: dict[str, Any] = {
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
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    text = parsed.get("message", {}).get("content", "")
    return text, parsed


# ---------------------------------------------------------------------------
# Ollama client — streaming with idle timeout (Describer)
# ---------------------------------------------------------------------------

class StreamIdleTimeout(Exception):
    """No streamed events for longer than DESCRIBER_IDLE_TIMEOUT_S."""


class StreamTotalTimeout(Exception):
    """Total wall-clock exceeded DESCRIBER_TOTAL_BUDGET_S."""


def _stream_reader(resp, q: queue.Queue) -> None:
    """Read NDJSON lines from the response, parse, push events onto q.
    Sends a sentinel None on EOF or error.
    """
    try:
        for raw in resp:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            q.put(obj)
            if obj.get("done"):
                break
    except Exception as e:
        q.put({"_error": repr(e)})
    finally:
        q.put(None)


def call_ollama_streaming(model: str, system: str, user: str, *,
                          idle_timeout_s: float,
                          total_budget_s: float,
                          show_thinking: bool = False
                          ) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": True,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_URL, data=data,
        headers={"Content-Type": "application/json"},
    )
    resp = urllib.request.urlopen(req, timeout=DESCRIBER_IDLE_TIMEOUT_S)

    q: queue.Queue = queue.Queue()
    reader = threading.Thread(target=_stream_reader, args=(resp, q), daemon=True)
    reader.start()

    thinking_chunks: list[str] = []
    content_chunks: list[str] = []
    first_thinking_at: float | None = None
    first_content_at: float | None = None
    started = time.time()
    in_thinking = False
    in_answer = False
    final_obj: dict[str, Any] = {}

    try:
        while True:
            elapsed_total = time.time() - started
            remaining_total = total_budget_s - elapsed_total
            if remaining_total <= 0:
                raise StreamTotalTimeout(f"exceeded total budget {total_budget_s}s")

            wait = min(idle_timeout_s, remaining_total)
            try:
                obj = q.get(timeout=wait)
            except queue.Empty:
                raise StreamIdleTimeout(f"no events for {idle_timeout_s}s")

            if obj is None:
                break
            if "_error" in obj:
                raise OSError(f"stream reader error: {obj['_error']}")

            msg = obj.get("message", {})
            t = msg.get("thinking", "")
            c = msg.get("content", "")

            if t:
                if first_thinking_at is None:
                    first_thinking_at = time.time() - started
                thinking_chunks.append(t)
                if show_thinking:
                    if not in_thinking:
                        print("\n[thinking] ", end="", flush=True)
                        in_thinking = True
                        in_answer = False
                    sys.stdout.write(t)
                    sys.stdout.flush()
            if c:
                if first_content_at is None:
                    first_content_at = time.time() - started
                content_chunks.append(c)
                if show_thinking:
                    if not in_answer:
                        print("\n\n[answer] ", end="", flush=True)
                        in_answer = True
                        in_thinking = False
                    sys.stdout.write(c)
                    sys.stdout.flush()

            if obj.get("done"):
                final_obj = obj
                if show_thinking:
                    print()
                break
    finally:
        try:
            resp.close()
        except Exception:
            pass

    return {
        "thinking": "".join(thinking_chunks),
        "content": "".join(content_chunks),
        "first_thinking_at": first_thinking_at,
        "first_content_at": first_content_at,
        "total_seconds": round(time.time() - started, 3),
        "final": final_obj,
    }


# ---------------------------------------------------------------------------
# Dicer output normalisation (boundary fix; ADR-001 amendment)
# ---------------------------------------------------------------------------

def normalise_dicer_output(raw: str) -> str:
    """Strip markdown code fences around the JSON body if present.

    Thinking models (gemma4:e4b observed) sometimes wrap valid JSON in
    ```json ... ``` even when format=json is requested. The wrapping is a
    protocol-level deviation, not a content error. Stripping it preserves
    ADR-001's strict-validation contract on the actual plan structure.

    No content rewriting — only outer-fence removal. If the body inside the
    fences is malformed JSON, downstream validation still rejects it.
    """
    s = raw.strip()
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        else:
            s = s[3:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def call_dicer(query: str, dicer_prompt: str, dicer_view: dict[str, Any]) -> str:
    user = (
        "Here is the corpus inventory you must route against:\n\n"
        "```json\n"
        + json.dumps(dicer_view, ensure_ascii=False, indent=2)
        + "\n```\n\n"
        + "User question:\n\n"
        + query.strip()
        + "\n\nReturn JSON only."
    )
    text, _ = call_ollama_blocking(DICER_MODEL, dicer_prompt, user,
                                   response_format_json=True,
                                   timeout=DICER_TIMEOUT_S)
    return text


def run_extractor(plan_text: str, here: Path, index_dir: Path,
                  export_xml: Path | None,
                  routes_dir: Path | None) -> tuple[int, str, str]:
    cmd: list[str] = [
        sys.executable, str(here / "extract.py"),
        "--plan", "-",
        "--index-dir", str(index_dir),
    ]
    if export_xml is not None:
        cmd += ["--export", str(export_xml)]
    if routes_dir is not None:
        cmd += ["--routes", str(routes_dir)]
    proc = subprocess.run(cmd, input=plan_text, capture_output=True,
                          text=True, check=False)
    return proc.returncode, proc.stdout, proc.stderr


def estimate_tokens(s: str) -> int:
    return len(s) // CHARS_PER_TOKEN


def guard_bundle_size(bundle: dict[str, Any],
                      describer_prompt: str,
                      query: str) -> tuple[dict[str, Any], dict[str, Any]]:
    composed = (
        describer_prompt + "\n\n" + query + "\n\n"
        + json.dumps(bundle, ensure_ascii=False)
    )
    initial_tokens = estimate_tokens(composed)
    report = {
        "initial_estimated_tokens": initial_tokens,
        "limit_tokens": DESCRIBER_CONTEXT_LIMIT_TOKENS,
        "truncated": False,
        "slice_truncations": [],
    }
    if initial_tokens <= DESCRIBER_CONTEXT_LIMIT_TOKENS:
        return bundle, report
    if bundle.get("kind") != "slice_bundle":
        return bundle, report

    bundle = json.loads(json.dumps(bundle))
    report["truncated"] = True

    overhead = estimate_tokens(
        describer_prompt + "\n\n" + query + "\n\n"
        + json.dumps({"kind": "slice_bundle", "rationale": bundle.get("rationale"),
                      "slices": []}, ensure_ascii=False)
    )
    budget = max(0, DESCRIBER_CONTEXT_LIMIT_TOKENS - overhead)

    for _ in range(5):
        per_slice_costs = [
            estimate_tokens(json.dumps(s, ensure_ascii=False))
            for s in bundle["slices"]
        ]
        total = sum(per_slice_costs)
        if total <= budget:
            break
        factor = budget / total if total > 0 else 1.0
        for s in bundle["slices"]:
            for key in ("records", "workouts"):
                arr = s.get(key)
                if not arr:
                    continue
                new_len = max(20, int(len(arr) * factor))
                if new_len < len(arr):
                    s[key] = arr[:new_len]
            s["truncated"] = True
            s["row_count"] = len(s.get("records") or []) + len(s.get("workouts") or [])

    final = estimate_tokens(
        describer_prompt + "\n\n" + query + "\n\n"
        + json.dumps(bundle, ensure_ascii=False)
    )
    report["final_estimated_tokens"] = final
    for s in bundle["slices"]:
        report["slice_truncations"].append({
            "label": s.get("label"),
            "row_count": s.get("row_count"),
            "truncated": s.get("truncated", False),
        })
    bundle["_orchestrator_note"] = (
        "Slices were truncated by the orchestrator to fit the Describer's "
        "context window. Coverage is partial; do not assume the slice contains "
        "the entire requested period."
    )
    return bundle, report


def call_describer(query: str, describer_prompt: str,
                   slice_bundle: dict[str, Any], *,
                   show_thinking: bool) -> dict[str, Any]:
    user = (
        "User question:\n\n"
        + query.strip()
        + "\n\nSlice bundle from the cascade:\n\n"
        + "```json\n"
        + json.dumps(slice_bundle, ensure_ascii=False, indent=2)
        + "\n```\n\n"
        + "Answer the question using only the data above."
    )
    return call_ollama_streaming(DESCRIBER_MODEL, describer_prompt, user,
                                 idle_timeout_s=DESCRIBER_IDLE_TIMEOUT_S,
                                 total_budget_s=DESCRIBER_TOTAL_BUDGET_S,
                                 show_thinking=show_thinking)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_once(query: str, here: Path, *, export_xml: Path | None,
             routes_dir: Path | None, show_thinking: bool) -> dict[str, Any]:
    started = time.time()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = here / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    index_dir = here / "index"
    dicer_view_path = index_dir / "dicer_view.json"
    if not dicer_view_path.exists():
        return {"ok": False, "error": "io/missing_dicer_view",
                "message": f"{dicer_view_path} not found. "
                           f"Run build_dicer_view.py first."}

    dicer_prompt = (here / "dicer_prompt.md").read_text(encoding="utf-8")
    describer_prompt = (here / "describer_prompt.md").read_text(encoding="utf-8")
    dicer_view = json.loads(dicer_view_path.read_text(encoding="utf-8"))

    trace: dict[str, Any] = {
        "ok": False,
        "query": query,
        "timestamp": timestamp,
        "models": {"dicer": DICER_MODEL, "describer": DESCRIBER_MODEL},
        "stages": {},
    }

    # ---- Stage 1: Dicer ----
    attempt = 0
    plan_text = ""
    extractor_rc = 1
    extractor_stdout = ""
    extractor_stderr = ""
    attempts_made = 0

    while attempt <= DICER_RETRY_LIMIT:
        attempts_made = attempt + 1
        t0 = time.time()
        try:
            plan_text_raw = call_dicer(query, dicer_prompt, dicer_view)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            dicer_seconds = time.time() - t0
            trace["error"] = "dicer/timeout_or_network"
            trace["message"] = f"Dicer call failed after {dicer_seconds:.1f}s: {e}"
            trace["stages"][f"attempt_{attempt}"] = {
                "dicer_seconds": round(dicer_seconds, 3),
                "outcome": "exception",
            }
            (run_dir / "trace.json").write_text(
                json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
            return trace
        dicer_seconds = time.time() - t0

        # Save the RAW Dicer output for forensics, then normalise for the extractor.
        (run_dir / f"dicer_output_attempt_{attempt}.json").write_text(
            plan_text_raw, encoding="utf-8")
        plan_text = normalise_dicer_output(plan_text_raw)
        if plan_text != plan_text_raw.strip():
            (run_dir / f"dicer_output_attempt_{attempt}.normalised.json").write_text(
                plan_text, encoding="utf-8")

        t0 = time.time()
        extractor_rc, extractor_stdout, extractor_stderr = run_extractor(
            plan_text, here, index_dir, export_xml, routes_dir)
        extractor_seconds = time.time() - t0

        trace["stages"][f"attempt_{attempt}"] = {
            "dicer_seconds": round(dicer_seconds, 3),
            "extractor_seconds": round(extractor_seconds, 3),
            "extractor_returncode": extractor_rc,
            "extractor_stderr": extractor_stderr.strip() if extractor_stderr else None,
            "normalised": plan_text != plan_text_raw.strip(),
        }
        if extractor_rc == 0:
            break
        attempt += 1

    if extractor_rc != 0:
        trace["error"] = "dicer/output_invalid_after_retry"
        trace["message"] = (
            f"Dicer produced invalid output across {attempts_made} attempts. "
            f"Last extractor error: {extractor_stderr.strip()}"
        )
        (run_dir / "trace.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        return trace

    bundle = json.loads(extractor_stdout)
    (run_dir / "slice_bundle.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    if bundle["kind"] == "question":
        trace["ok"] = True
        trace["kind"] = "question"
        trace["question"] = bundle["question"]
        trace["reason"] = bundle.get("reason")
        trace["total_seconds"] = round(time.time() - started, 3)
        (run_dir / "trace.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        return trace

    # ---- Stage 2.5: Token guard ----
    guarded_bundle, guard_report = guard_bundle_size(bundle, describer_prompt, query)
    trace["stages"]["guard"] = guard_report
    if guard_report["truncated"]:
        (run_dir / "slice_bundle_guarded.json").write_text(
            json.dumps(guarded_bundle, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- Stage 3: Describer (streaming) ----
    t0 = time.time()
    try:
        record = call_describer(query, describer_prompt, guarded_bundle,
                                show_thinking=show_thinking)
    except StreamIdleTimeout as e:
        describer_seconds = time.time() - t0
        trace["error"] = "describer/idle_timeout"
        trace["message"] = f"Describer idle-timed-out after {describer_seconds:.1f}s: {e}"
        trace["stages"]["describer_seconds"] = round(describer_seconds, 3)
        (run_dir / "trace.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        return trace
    except StreamTotalTimeout as e:
        describer_seconds = time.time() - t0
        trace["error"] = "describer/total_timeout"
        trace["message"] = f"Describer hit total budget after {describer_seconds:.1f}s: {e}"
        trace["stages"]["describer_seconds"] = round(describer_seconds, 3)
        (run_dir / "trace.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        return trace
    except (urllib.error.URLError, OSError) as e:
        describer_seconds = time.time() - t0
        trace["error"] = "describer/network"
        trace["message"] = f"Describer network error after {describer_seconds:.1f}s: {e}"
        trace["stages"]["describer_seconds"] = round(describer_seconds, 3)
        (run_dir / "trace.json").write_text(
            json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
        return trace
    describer_seconds = time.time() - t0

    answer = record["content"]
    thinking = record["thinking"]

    (run_dir / "describer_output.txt").write_text(answer, encoding="utf-8")
    if thinking:
        (run_dir / "describer_thinking.txt").write_text(thinking, encoding="utf-8")

    trace["ok"] = True
    trace["kind"] = "answer"
    trace["answer"] = answer
    trace["stages"]["describer_seconds"] = round(describer_seconds, 3)
    trace["stages"]["describer_first_thinking_at"] = record.get("first_thinking_at")
    trace["stages"]["describer_first_content_at"] = record.get("first_content_at")
    trace["stages"]["thinking_chars"] = len(thinking)
    trace["stages"]["answer_chars"] = len(answer)
    trace["stages"]["slice_row_counts"] = [
        s.get("row_count", 0) for s in guarded_bundle.get("slices", [])
    ]
    trace["total_seconds"] = round(time.time() - started, 3)
    (run_dir / "trace.json").write_text(
        json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return trace


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def render(trace: dict[str, Any]) -> None:
    if not trace.get("ok"):
        print(f"\n[cascade] FAILED: {trace.get('error', 'unknown')}")
        print(f"[cascade] {trace.get('message', '')}")
        return

    if trace["kind"] == "question":
        print(f"\n[cascade asks]:\n{trace['question']}\n")
    else:
        print(f"\n=== ANSWER ===\n{trace['answer']}\n==============\n")

    stages = trace.get("stages", {})
    print(f"[cascade] total: {trace.get('total_seconds')}s  "
          f"(see runs/{trace['timestamp']}/ for full trace)")
    print(f"[cascade] models: dicer={trace['models']['dicer']}  "
          f"describer={trace['models']['describer']}")
    for k, v in stages.items():
        if isinstance(v, dict):
            print(f"           {k}: {v}")
        else:
            print(f"           {k}: {v}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query", nargs="?", help="The user question. If omitted, read from stdin.")
    p.add_argument("--export", type=Path, default=None,
                   help="Path to export.xml (only needed if a plan uses raw aggregation).")
    p.add_argument("--routes", type=Path, default=None,
                   help="Path to workout-routes/ (only needed if include_routes=true).")
    p.add_argument("--show-thinking", action="store_true",
                   help="Stream thinking and content to the terminal as they arrive.")
    args = p.parse_args()

    here = Path(__file__).resolve().parent

    if args.query:
        query = args.query
    else:
        print("> ", end="", flush=True)
        query = sys.stdin.readline().strip()
        if not query:
            print("error: no query provided", file=sys.stderr)
            return 2

    try:
        trace = run_once(query, here,
                         export_xml=args.export.expanduser().resolve() if args.export else None,
                         routes_dir=args.routes.expanduser().resolve() if args.routes else None,
                         show_thinking=args.show_thinking)
    except urllib.error.URLError as e:
        print(f"error: cannot reach Ollama at {OLLAMA_URL}: {e}", file=sys.stderr)
        print("Is Ollama running?", file=sys.stderr)
        return 3

    render(trace)
    return 0 if trace.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
