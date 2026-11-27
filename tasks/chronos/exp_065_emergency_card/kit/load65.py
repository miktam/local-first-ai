"""exp_065, step one: which model files on this machine load on the pinned llama.cpp release?
    python3.14 -I kit/load65.py > dev/load-<machine>-<date>.txt
For each file: start llama-server (loopback only), wait for it to be ready, one short chat call, stop it.
Prints one line per file: size, sha256, loaded or not, the reply's first characters, seconds. No task of the test is used."""
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path.home()
SERVER = HOME / "models/llama.cpp-b12676/build/bin/llama-server"
PORT = 18465
OLLAMA_TAGS = ["gemma4:26b", "gemma4:26b-a4b-it-qat", "gemma4:31b", "qwen3.5:35b", "qwen3.6:35b", "qwen3.8:27b",
               "hf.co/bartowski/Qwen3.8-27B-GGUF:Q4_K_M", "hf.co/bartowski/NousResearch_Arete-27B-GGUF:Q4_K_M", "gemma4:e4b"]
PLAIN = {"Inkling 2 Mini (ggml-org IQ4_NL)": HOME / "models/inkling2-ggml/Inkling-2-Mini-IQ4_NL.gguf",
         "Inkling 2 Mini (Unsloth UD-Q4_K_M)": HOME / "models/inkling2/Inkling-2-Mini-UD-Q4_K_M.gguf"}


def ollama_file(tag):
    out = subprocess.run(["ollama", "show", tag, "--modelfile"], capture_output=True, text=True).stdout
    froms = [l.split(None, 1)[1].strip() for l in out.splitlines() if l.startswith("FROM ")]
    files = [Path(f) for f in froms if f.startswith("/")]
    return max(files, key=lambda p: p.stat().st_size) if files else None


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def try_load(path):
    log_path = Path("/tmp/load65-server.log")
    log = open(log_path, "wb")
    proc = subprocess.Popen([str(SERVER), "-m", str(path), "-c", "8192", "-np", "1", "-ngl", "99", "--host", "127.0.0.1",
                             "--port", str(PORT), "--jinja"], stdout=log, stderr=subprocess.STDOUT)
    t0 = time.time()
    try:
        while time.time() - t0 < 240:
            if proc.poll() is not None:
                break
            try:
                if b'"ok"' in urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=3).read():
                    body = {"messages": [{"role": "user", "content": "Reply with the single word: ready"}],
                            "temperature": 0, "seed": 1, "max_tokens": 256}
                    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", json.dumps(body).encode(),
                                                 {"Content-Type": "application/json"})
                    d = json.loads(urllib.request.urlopen(req, timeout=300).read())
                    m = d["choices"][0]["message"]
                    return True, (m.get("content") or "").strip()[:40], time.time() - t0, ""
            except urllib.error.URLError:
                time.sleep(2)
            except Exception as e:  # the server answered, but not with a usable reply
                return False, "", time.time() - t0, f"call failed: {type(e).__name__}"
        log.flush()
        tail = [l for l in log_path.read_text(errors="replace").splitlines() if "error" in l.lower() or "fail" in l.lower()]
        return False, "", time.time() - t0, (tail[-1][-160:] if tail else "server exited or did not become ready")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()
        time.sleep(2)


files = {}
for tag in OLLAMA_TAGS:
    f = ollama_file(tag)
    if f is None:
        print(f"[{tag}] not on this machine", flush=True)
    else:
        files[tag] = f
for label, f in PLAIN.items():
    if f.exists():
        files[label] = f
    else:
        print(f"[{label}] not on this machine", flush=True)

only = sys.argv[1:]
seen = {}
for label, f in files.items():
    if only and not any(o in label for o in only):
        continue
    if f in seen:
        print(f"[{label}] the same file as [{seen[f]}]", flush=True)
        continue
    seen[f] = label
    size = f.stat().st_size
    digest = sha256(f)
    ok, reply, secs, why = try_load(f)
    print(f"[{label}] {size / 1e9:.1f} GB sha256 {digest[:16]}… | {'LOADED' if ok else 'NOT LOADED'} in {secs:.0f} s"
          f" | reply {reply!r}" + (f" | {why}" if why else ""), flush=True)
