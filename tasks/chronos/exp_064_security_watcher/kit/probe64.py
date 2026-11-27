"""exp_064, the two probes the registration names before signing. One call each, made-up input.
    python3.14 -I kit/probe64.py > dev/probe-<date>.txt
1. Does qwen3.8:27b return a tool call through Ollama's chat endpoint? (the third model for tuning)
2. Does gemma4:26b-a4b-it-qat return one through llama-server b12676? (the optional arm Q)
Nothing real is looked up: the tools are definitions only and no tool result is sent back."""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OLLAMA = "http://127.0.0.1:11434"
THIRD = "qwen3.8:27b"
SERVER = Path.home() / "models/llama.cpp-b12676/build/bin/llama-server"
QAT = Path.home() / ".ollama/models/blobs/sha256-4c856523d61d77922dbc0b26753a6bf6208e5d69d80db0c04dcd776832d054c5"
PORT = 18464


def fn(name, desc, **props):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": "string"} for k in props}, "required": list(props)}}}


TOOLS = [fn("lookup_device", "Look a device id up in the list of the owner's known devices.", device_id=1),
         fn("signin_history", "List earlier sign-ins for an account.", account=1),
         fn("alert_owner", "Send the owner an alert.", text=1)]
MSGS = [{"role": "system", "content": "You watch sign-in events for one owner. Before deciding anything about a device, "
                                       "look the device up with lookup_device."},
        {"role": "user", "content": "Event: sign-in to account ops@harborlight.example from device D-4471 at 03:12."}]


def post(url, body, timeout=900):
    req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return d, time.time() - t0


def show(label, d, secs):
    m = d["choices"][0]["message"]
    calls = m.get("tool_calls") or []
    got = [(c["function"]["name"], c["function"]["arguments"]) for c in calls]
    print(f"[{label}] tool_calls returned: {len(calls)} {got} | content {len(m.get('content') or '')} chars | "
          f"finish {d['choices'][0]['finish_reason']} | {secs:.0f} s (load included)", flush=True)


which = sys.argv[1:] or ["third", "qat"]
body = {"messages": MSGS, "tools": TOOLS, "temperature": 0, "seed": 1, "max_tokens": 1024}

if "third" in which:
    try:
        d, secs = post(OLLAMA + "/v1/chat/completions", dict(body, model=THIRD))
        show(f"{THIRD} on Ollama", d, secs)
    finally:
        subprocess.run(["ollama", "stop", THIRD], capture_output=True)

if "qat" in which:
    if not QAT.exists():
        print(f"[QAT gemma on llama.cpp b12676] file not found: {QAT.name}")
    else:
        log = open("/tmp/probe64-server.log", "ab")
        proc = subprocess.Popen([str(SERVER), "-m", str(QAT), "-c", "8192", "-np", "1", "-ngl", "99", "--host", "127.0.0.1",
                                 "--port", str(PORT), "--jinja"], stdout=log, stderr=subprocess.STDOUT)
        try:
            t0 = time.time()
            ready = False
            while time.time() - t0 < 300 and proc.poll() is None:
                try:
                    if b'"ok"' in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).read():
                        ready = True
                        break
                except Exception:
                    time.sleep(2)
            if not ready:
                print("[QAT gemma on llama.cpp b12676] the server did not come up")
            else:
                d, secs = post(f"http://127.0.0.1:{PORT}/v1/chat/completions", body)
                show("gemma4:26b-a4b-it-qat on llama.cpp b12676", d, secs)
        finally:
            proc.terminate()
            proc.wait(timeout=60)
            log.close()
