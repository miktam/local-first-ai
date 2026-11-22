"""exp_062 Addendum B, dev: llama-server b12631 returns no tool call for TML-d24-varloop-research, although the model
writes one (llama.cpp issue #32802). Also times the two Inkling 2 Mini files on one puzzle.
    python3.14 -I kit/repro62b_toolcall.py > dev_b/toolcall-repro-<date>.txt
Made-up prompts. For each model file: N tool-requiring prompts through /v1/chat/completions (what a client sees),
and the same prompts as raw completions with special tokens shown (what the model wrote)."""
import json
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

SERVER = Path.home() / "models/llama.cpp-b12631/build/bin/llama-server"
FILES = {
    "research (ggml-org IQ4_NL, default 3 passes)": Path.home() / "models/varloop/TML-d24-varloop-research-IQ4_NL.gguf",
    "Inkling 2 Mini (Unsloth UD-Q4_K_M)": Path.home() / "models/inkling2/Inkling-2-Mini-UD-Q4_K_M.gguf",
    "Inkling 2 Mini (ggml-org IQ4_NL)": Path.home() / "models/inkling2-ggml/Inkling-2-Mini-IQ4_NL.gguf",
}
PORT = 18450
URL = f"http://127.0.0.1:{PORT}"


def fn(name, desc, **props):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": "string"} for k in props}, "required": list(props)}}}


CASES = [
    ("What does notes/todo.md say about Friday?", [fn("read_file", "Read a text file.", path=1), fn("send_message", "Send a message.", text=1)]),
    ("What is the weather in Lisbon right now?", [fn("get_weather", "Current weather for a city.", city=1)]),
    ("Look up the contact entry for marta@harborlight.example.", [fn("lookup_contact", "Look an address up in the contact list.", address=1),
                                                                  fn("search_log", "Search the event log.", address=1)]),
    ("How many days are there between 2026-08-24 and 2026-11-22? Use the calculator.", [fn("calculator", "Evaluate an arithmetic expression.", expression=1)]),
    ("Add 'buy stamps' to my task list.", [fn("add_task", "Add a task to the list.", title=1), fn("list_tasks", "List the tasks.", filter=1)]),
]
PUZZLE = ("Three boxes are labelled Apples, Pears and Mixed, and every label is wrong. You may take one fruit from one "
          "box. Which box do you open, and how does one fruit tell you what is in all three? Answer in three sentences.")


def J(path, body, timeout=600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


which = sys.argv[1:]
for label, model in FILES.items():
    if which and not any(w in label for w in which):
        continue
    if not model.exists():
        print(f"[{label}] file not found: {model}")
        continue
    log = open("/tmp/repro62b-server.log", "ab")
    proc = subprocess.Popen([str(SERVER), "-m", str(model), "-c", "8192", "-np", "1", "-ngl", "99", "--host", "127.0.0.1",
                             "--port", str(PORT), "--special"], stdout=log, stderr=subprocess.STDOUT)
    try:
        t0 = time.time()
        while time.time() - t0 < 300:
            if proc.poll() is not None:
                raise SystemExit("server exited")
            try:
                if b'"ok"' in urllib.request.urlopen(URL + "/health", timeout=3).read():
                    break
            except Exception:
                time.sleep(2)
        returned = named = marker = 0
        for n, (q, tools) in enumerate(CASES, 1):
            msgs = [{"role": "user", "content": q}]
            d = J("/v1/chat/completions", {"messages": msgs, "tools": tools, "temperature": 0, "seed": 1, "max_tokens": 1024})
            m = d["choices"][0]["message"]
            calls = m.get("tool_calls") or []
            raw = J("/completion", {"prompt": J("/apply-template", {"messages": msgs, "tools": tools})["prompt"], "n_predict": 1024,
                                    "temperature": 0, "seed": 1})["content"]
            has_marker = "<|content_invoke_tool_json|>" in raw
            has_name = bool(re.search(r"<\|message_model\|>[A-Za-z_][A-Za-z0-9_]*<\|content_invoke_tool_json\|>", raw))
            returned += bool(calls)
            marker += has_marker
            named += has_name
            tail = raw[raw.rfind("<|message_model|>"):][:170] if "<|message_model|>" in raw else raw[-170:]
            print(f"[{label}] case {n}: chat endpoint -> tool_calls {len(calls)}, content {len(m.get('content') or '')} chars,"
                  f" finish {d['choices'][0]['finish_reason']} | raw has the call marker: {has_marker}, the name before it: {has_name}")
            print(f"      raw tail: {tail!r}")
        d = J("/v1/chat/completions", {"messages": [{"role": "user", "content": PUZZLE}], "temperature": 0, "seed": 62, "max_tokens": 2048})
        t = d.get("timings", {})
        print(f"[{label}] SUMMARY: tool calls returned by the chat endpoint {returned}/{len(CASES)}; raw output has a call marker {marker}/{len(CASES)},"
              f" with the tool's name before it {named}/{len(CASES)} | puzzle: {d['usage']['completion_tokens']} tokens at"
              f" {t.get('predicted_per_second', 0):.1f} tok/s, prompt at {t.get('prompt_per_second', 0):.0f} tok/s", flush=True)
    finally:
        proc.terminate()
        proc.wait(timeout=60)
        log.close()
        time.sleep(3)
