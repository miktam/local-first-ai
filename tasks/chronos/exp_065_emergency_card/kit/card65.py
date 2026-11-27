"""exp_065, the driver: for each model file on this machine, Part C (tool calls), Part B (the 48 tasks) and Part A (the
context ladder), on llama-server from the pinned release, loopback only.

    python3.14 -I kit/card65.py run --machine mini                      # a new run, every model, parts A, B and C
    python3.14 -I kit/card65.py run --machine mbp --parts A,C            # files whose tasks were run on the other machine
    python3.14 -I kit/card65.py run --machine mbp --only Ling,Nanbeige   # only models whose label contains one of these
    python3.14 -I kit/card65.py run --machine mini --resume runs/mini-20261128T080000Z   # after a stop or a power cut
    python3.14 -I kit/card65.py list --machine mini                      # which model files are on this machine

Every answer is written to disk when it is complete, so a stopped run loses at most the call in flight. Nothing is
fetched: a file that is not on the disk is reported and skipped."""
import argparse
import hashlib
import json
import os
import random
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

KIT = Path(__file__).resolve().parent
EXP = KIT.parent
HOME = Path.home()
SERVER = Path(os.environ.get("CARD65_SERVER") or HOME / "models/llama.cpp-b12676/build/bin/llama-server")
PORT = int(os.environ.get("CARD65_PORT") or 18465)
URL = f"http://127.0.0.1:{PORT}"
TASKS = Path(os.environ.get("CARD65_TASKS") or KIT / "tasks65.json")  # the env variable is for the smoke only
SIZES = [8192, 16384, 32768, 65536, 131072]
TASK_CTX = 8192
TASK_CAP = 2048
SEED = 65
FUTILITY = 12
SIZES_ARG = []


# ---------- model files ----------

def models_on_this_machine():
    """label -> (path, template or None). Ollama tags are resolved to their weight file; plain files by path."""
    out = {}
    for m in json.loads((KIT / "models65.json").read_text()):
        path = None
        if m["kind"] == "ollama":
            try:
                txt = subprocess.run(["ollama", "show", m["ref"], "--modelfile"], capture_output=True, text=True).stdout
            except FileNotFoundError:
                txt = ""
            files = [Path(l.split(None, 1)[1].strip()) for l in txt.splitlines() if l.startswith("FROM /")]
            files = [f for f in files if f.exists()]
            if files:
                path = max(files, key=lambda p: p.stat().st_size)
        else:
            p = Path(os.path.expanduser(m["ref"]))
            path = p if p.exists() else None
        out[m["label"]] = (path, (KIT / m["template"]) if m.get("template") else None)
    return out


def file_size(path: Path) -> int:
    """A split GGUF is given by its first part; the size is of all parts."""
    if "-00001-of-" in path.name:
        stem = path.name.split("-00001-of-")[0]
        return sum(p.stat().st_size for p in path.parent.glob(stem + "-*-of-*.gguf"))
    return path.stat().st_size


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def gguf_meta(path: Path) -> dict:
    """The scalar metadata of a GGUF file (arrays are skipped). Enough to estimate the memory a context needs."""
    out = {}
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return out
        _version, _tensors, n_kv = struct.unpack("<IQQ", f.read(20))

        def string():
            (n,) = struct.unpack("<Q", f.read(8))
            return f.read(n).decode("utf-8", "replace")

        scalar = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}

        def value(t):
            if t in scalar:
                return struct.unpack(scalar[t], f.read(struct.calcsize(scalar[t])))[0]
            if t == 8:
                return string()
            if t == 9:
                (it,) = struct.unpack("<I", f.read(4))
                (n,) = struct.unpack("<Q", f.read(8))
                if it in scalar:
                    f.seek(n * struct.calcsize(scalar[it]), 1)
                else:
                    for _ in range(n):
                        value(it)
                return None
            raise ValueError(f"unknown GGUF value type {t}")

        try:
            for _ in range(n_kv):
                key = string()
                (t,) = struct.unpack("<I", f.read(4))
                v = value(t)
                if v is not None and not isinstance(v, str) or key == "general.architecture":
                    out[key] = v
        except Exception:
            pass
    return out


def kv_estimate(meta: dict, n_ctx: int):
    """Bytes of key/value cache at 16 bits a number, or None when the file does not state its shape."""
    arch = meta.get("general.architecture")
    try:
        layers = meta[f"{arch}.block_count"]
        embd = meta[f"{arch}.embedding_length"]
        heads = meta[f"{arch}.attention.head_count"]
        heads_kv = meta.get(f"{arch}.attention.head_count_kv", heads)
        return 2 * layers * n_ctx * (embd // heads) * heads_kv * 2
    except Exception:
        return None


def physical_memory() -> int:
    return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip() or 0)


def swap_used_mb() -> float:
    txt = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    try:
        return float(txt.split("used = ")[1].split("M")[0])
    except Exception:
        return -1.0


# ---------- the server ----------

def post(path, body, timeout=3600):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return d, time.time() - t0


def get(path, timeout=10):
    return json.loads(urllib.request.urlopen(URL + path, timeout=timeout).read())


class Server:
    def __init__(self, path: Path, template, n_ctx: int, log_path: Path):
        self.args = [str(SERVER), "-m", str(path), "-c", str(n_ctx), "-np", "1", "-ngl", "99", "--host", "127.0.0.1",
                     "--port", str(PORT), "--jinja"] + (["--chat-template-file", str(template)] if template else [])
        self.log_path = log_path
        self.proc = None

    def __enter__(self):
        self.log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(self.args, stdout=self.log, stderr=subprocess.STDOUT)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                return None
            try:
                if b'"ok"' in urllib.request.urlopen(URL + "/health", timeout=3).read():
                    return self
            except Exception:
                time.sleep(2)
        return None

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.log.close()
        time.sleep(3)


def port_busy() -> bool:
    try:
        urllib.request.urlopen(URL + "/health", timeout=2)
        return True
    except Exception:
        return False


# ---------- rows on disk ----------

def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in path.read_text().splitlines() if l.strip()}


def append(path: Path, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


# ---------- Part C: tool calls ----------

def fn(name, desc, **props):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {k: {"type": "string"} for k in props}, "required": list(props)}}}


TOOL_CASES = [
    ("C1", "What does notes/todo.md say about Friday?", [fn("read_file", "Read a text file.", path=1), fn("send_message", "Send a message.", text=1)],
     "read_file", "todo.md"),
    ("C2", "What is the weather in Lisbon right now?", [fn("get_weather", "Current weather for a city.", city=1)], "get_weather", "lisbon"),
    ("C3", "Look up the contact entry for marta@harborlight.example.",
     [fn("lookup_contact", "Look an address up in the contact list.", address=1), fn("search_log", "Search the event log.", address=1)],
     "lookup_contact", "marta@harborlight.example"),
    ("C4", "How many days are there between 2026-08-24 and 2026-11-22? Use the calculator.",
     [fn("calculator", "Evaluate an arithmetic expression.", expression=1)], "calculator", ""),
    ("C5", "Add 'buy stamps' to my task list.", [fn("add_task", "Add a task to the list.", title=1), fn("list_tasks", "List the tasks.", filter=1)],
     "add_task", "stamps"),
]


def part_c(out: Path):
    have = done_ids(out)
    for cid, q, tools, want, arg in TOOL_CASES:
        if cid in have:
            continue
        try:
            d, secs = post("/v1/chat/completions", {"messages": [{"role": "user", "content": q}], "tools": tools,
                                                    "temperature": 0, "seed": SEED, "max_tokens": 1024})
            m = d["choices"][0]["message"]
            calls = [(c["function"]["name"], c["function"]["arguments"]) for c in (m.get("tool_calls") or [])]
            ok = any(n == want and arg in str(a).lower() for n, a in calls)
            append(out, {"id": cid, "calls": calls, "right": ok, "content_chars": len(m.get("content") or ""),
                         "finish": d["choices"][0]["finish_reason"], "seconds": round(secs, 1)})
        except Exception as e:
            append(out, {"id": cid, "calls": [], "right": False, "error": f"{type(e).__name__}: {e}"[:200]})


# ---------- Part B: the tasks ----------

def part_b(out: Path, tasks: list, only_ids=None) -> str:
    have = done_ids(out)
    rows = [json.loads(l) for l in out.read_text().splitlines()] if out.exists() else []
    for t in tasks:
        if only_ids and t["id"] not in only_ids:
            continue
        if t["id"] in have:
            continue
        if len(rows) >= FUTILITY and all(r["no_answer"] for r in rows[:FUTILITY]):
            return f"futility: the first {FUTILITY} answers were empty"
        last = None
        for attempt in (1, 2, 3):
            try:
                d, secs = post("/v1/chat/completions", {"messages": [{"role": "user", "content": t["prompt"]}], "temperature": 0,
                                                        "seed": SEED, "max_tokens": TASK_CAP, "cache_prompt": False})
                break
            except Exception as e:  # a failed call is not an answer: ask again, then stop the model
                last = f"{type(e).__name__}: {e}"[:200]
                time.sleep(10 * attempt)
        else:
            return f"stopped at {t['id']}: {last}"
        ch = d["choices"][0]
        m = ch["message"]
        content = (m.get("content") or "").strip()
        tm = d.get("timings", {})
        row = {"id": t["id"], "answer": content, "reasoning": m.get("reasoning_content") or "", "finish": ch["finish_reason"],
               "no_answer": not content,  # empty: nothing was written, or the cap fell inside the reasoning
               "cut": ch["finish_reason"] == "length",  # an answer cut at the cap is judged on what is there
               "prompt_tokens": d["usage"]["prompt_tokens"], "completion_tokens": d["usage"]["completion_tokens"],
               "seconds": round(secs, 1), "tokens_per_second": round(tm.get("predicted_per_second") or 0, 1)}
        append(out, row)
        rows.append(row)
        print(f"      {t['id']}: {row['completion_tokens']} tokens in {row['seconds']} s"
              f"{' | NO ANSWER' if row['no_answer'] else (' | cut at the cap' if row['cut'] else '')}", flush=True)
    return ""


# ---------- Part A: the context ladder ----------

WORDS = ("harbor ledger window copper gentle meadow signal paper lantern orchard river basket marble quiet winter engine "
         "garden saddle pocket velvet timber anchor button candle feather granite hollow island jacket kettle").split()
PLACES = ["Thornfield", "Marlow Quay", "Easthollow"]


def ladder_text(n_tokens_target: int, seed: int):
    """Made-up filler with three planted facts at the start, the middle and the end. Returns text and the three codes."""
    rnd = random.Random(seed)
    codes = [str(rnd.randrange(100000, 999999)) for _ in PLACES]
    facts = [f"The access code for the {p} archive is {c}." for p, c in zip(PLACES, codes)]

    def sentence():
        return " ".join(rnd.choice(WORDS) for _ in range(rnd.randrange(8, 14))).capitalize() + "."

    sample = " ".join(sentence() for _ in range(200))
    per_sentence = len(post("/tokenize", {"content": sample})[0]["tokens"]) / 200
    n = max(30, int(n_tokens_target / per_sentence))
    for _ in range(4):
        rnd2 = random.Random(seed + 1)
        body = []
        for i in range(n):
            body.append(" ".join(rnd2.choice(WORDS) for _ in range(rnd2.randrange(8, 14))).capitalize() + ".")
        body.insert(int(n * 0.97), facts[2])
        body.insert(int(n * 0.50), facts[1])
        body.insert(int(n * 0.02), facts[0])
        text = " ".join(body)
        got = len(post("/tokenize", {"content": text})[0]["tokens"])
        if abs(got - n_tokens_target) <= n_tokens_target * 0.03:
            break
        n = max(30, int(n * n_tokens_target / got))
    return text, codes, got


def part_a(out: Path, path: Path, template, log_path: Path, meta: dict, n_ctx_train):
    have = done_ids(out)
    mem = physical_memory()
    size = file_size(path)
    for n_ctx in (SIZES_ARG or SIZES):
        cid = f"A-{n_ctx}"
        if cid in have:
            continue
        if n_ctx_train and n_ctx > n_ctx_train:
            append(out, {"id": cid, "n_ctx": n_ctx, "skipped": f"above the model's trained context of {n_ctx_train}"})
            continue
        kv = kv_estimate(meta, n_ctx)
        if kv is not None and size + kv + 6e9 > mem * 0.9:
            append(out, {"id": cid, "n_ctx": n_ctx, "skipped": f"estimated memory {(size + kv) / 1e9:.0f} GB of {mem / 1e9:.0f} GB"})
            continue
        swap0 = swap_used_mb()
        row = {"id": cid, "n_ctx": n_ctx, "kv_estimate_gb": round(kv / 1e9, 1) if kv else None}
        with Server(path, template, n_ctx, log_path) as srv:
            if srv is None:
                row["loaded"] = False
                append(out, row)
                break
            row["loaded"] = True
            try:
                text, codes, got = ladder_text(int(n_ctx * 0.75), SEED + n_ctx)
                q = text + "\n\nQuestion: list the three access codes mentioned in the text above, each with the name of its archive."
                d, secs = post("/v1/chat/completions", {"messages": [{"role": "user", "content": q}], "temperature": 0, "seed": SEED,
                                                        "max_tokens": 1024, "cache_prompt": False}, timeout=7200)
                ch = d["choices"][0]
                ans = (ch["message"].get("content") or "")
                tm = d.get("timings", {})
                found = [c in ans for c in codes]
                first = (tm.get("prompt_ms") or 0) / 1000
                tps = tm.get("predicted_per_second") or 0
                row.update({"prompt_tokens": d["usage"]["prompt_tokens"], "filler_tokens": got, "seconds_to_first_token": round(first, 1),
                            "tokens_per_second": round(tps, 1), "facts_found": found, "finish": ch["finish_reason"],
                            "seconds": round(secs, 1), "swap_grew_mb": round(swap_used_mb() - swap0, 0),
                            "usable": first <= 60 and tps >= 10 and all(found)})
            except Exception as e:
                row["error"] = f"{type(e).__name__}: {e}"[:200]
        append(out, row)
        print(f"      {cid}: {json.dumps({k: v for k, v in row.items() if k not in ('id', 'n_ctx')})}", flush=True)
        if row.get("error") or row.get("swap_grew_mb", 0) > 1024:
            append(out, {"id": f"A-stop-after-{n_ctx}", "n_ctx": n_ctx, "skipped": "the ladder stops here: an error, or swap grew by more than 1 GB"})
            break


# ---------- a run ----------

def slug(label: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in label).strip("-").lower()


def machine_state() -> dict:
    def sh(*cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            return ""
    return {"thermal": sh("pmset", "-g", "therm"), "ollama_ps": sh("ollama", "ps"), "swap": sh("sysctl", "-n", "vm.swapusage"),
            "memory_bytes": physical_memory(), "chip": sh("sysctl", "-n", "machdep.cpu.brand_string"), "os": sh("sw_vers", "-productVersion"),
            "server_version": subprocess.run([str(SERVER), "--version"], capture_output=True, text=True).stderr.strip()[:300]}


def run(a):
    if not SERVER.exists():
        sys.exit(f"llama-server not found at {SERVER}")
    if port_busy():
        sys.exit(f"port {PORT} is in use: another run is going, or an old server was left behind")
    parts = set(a.parts.upper().split(","))
    tasks = json.loads(TASKS.read_text()) if TASKS.exists() else []
    if "B" in parts and not tasks:
        sys.exit(f"{TASKS.name} is missing: the tasks are not frozen yet")
    run_dir = Path(a.resume) if a.resume else EXP / "runs" / f"{a.machine}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"machine": a.machine, "models": {}, "starts": []}
    if TASKS.exists():
        digest = hashlib.sha256(TASKS.read_bytes()).hexdigest()
        if meta.get("tasks_sha256") not in (None, digest):
            sys.exit("the task file changed since this run started")
        meta["tasks_sha256"] = digest
    meta["starts"].append({"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "parts": sorted(parts), "state": machine_state()})
    only = [o.lower() for o in a.only.split(",")] if a.only else []
    check_ids = set(a.task_ids.split(",")) if a.task_ids else None
    for label, (path, template) in models_on_this_machine().items():
        if only and not any(o in label.lower() for o in only):
            continue
        if path is None:
            print(f"[{label}] not on this machine", flush=True)
            meta["models"].setdefault(label, {"present": False})
            continue
        d = run_dir / slug(label)
        d.mkdir(exist_ok=True)
        info = meta["models"].setdefault(label, {"present": True, "file": path.name, "bytes": file_size(path)})
        if "sha256" not in info:
            print(f"[{label}] hashing {info['bytes'] / 1e9:.1f} GB", flush=True)
            info["sha256"] = sha256(path)
            info["template"] = template.name if template else "the file's own"
        g = gguf_meta(path)
        arch = g.get("general.architecture")
        n_ctx_train = g.get(f"{arch}.context_length")
        info["n_ctx_train"] = n_ctx_train
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"[{label}] parts {','.join(sorted(parts))}", flush=True)
        if parts & {"B", "C"}:
            with Server(path, template, TASK_CTX, d / "server.log") as srv:
                if srv is None:
                    info["note"] = "did not start at 8k"
                    print("      the server did not start", flush=True)
                else:
                    if "C" in parts:
                        part_c(d / "tools.jsonl")
                    if "B" in parts:
                        why = part_b(d / "tasks.jsonl", tasks, check_ids)
                        if why:
                            info["note"] = why
                            print(f"      {why}", flush=True)
        if "A" in parts:
            part_a(d / "ladder.jsonl", path, template, d / "server.log", g, n_ctx_train)
        meta_path.write_text(json.dumps(meta, indent=1))
    print(f"run directory: {run_dir}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--machine", required=True, choices=["mini", "mbp"])
    r.add_argument("--parts", default="A,B,C")
    r.add_argument("--only", default="")
    r.add_argument("--task-ids", default="", help="Part B on these task ids only (the six-task check across machines)")
    r.add_argument("--resume", default="")
    r.add_argument("--sizes", default="", help="context sizes for Part A, for the smoke; the run uses all five")
    li = sub.add_parser("list")
    li.add_argument("--machine", required=True, choices=["mini", "mbp"])
    a = ap.parse_args()
    if a.cmd == "list":
        for label, (path, template) in models_on_this_machine().items():
            print(f"{'yes' if path else 'no ':3} {label}" + (f"  ({file_size(path) / 1e9:.1f} GB)" if path else ""))
        return
    if a.sizes:
        SIZES_ARG.extend(int(s) for s in a.sizes.split(","))
    run(a)


if __name__ == "__main__":
    main()
