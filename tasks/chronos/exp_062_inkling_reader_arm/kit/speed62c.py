#!/usr/bin/env python3
"""exp_062 Addendum C: the same two files of Inkling 2 Mini on three builds of llama.cpp.

    python3.14 kit/speed62c.py --all --run runs_c/<ts> [--smoke]

The driver checks every pinned input, then runs, one side server at a time on 127.0.0.1:18450:

  cells      U0 N0 U1 U2 N2 U0r   one build with one file, one slot of 8,192:
             Part 1 (speed: two workloads, six requests each) and Part 3 (eight texts, token ids kept)
  placement  U on b12631 and on the head, N on b12631: one short load with GGML_SCHED_DEBUG=2
  capacity   C-U0 C-N0 C-U2       Addendum A's Part 2 (its job runner and 12 jobs) at 1, 2 and 4 watchers

Each step writes its file when it is complete and a resume with the same --run skips complete steps. Every child
runs under exp_054's audit hook. The side server is stopped on any exit. Nothing calls the production Ollama
except `ollama ps`, read before and after. No generated text is written to the run folder: token ids only.
"""
import argparse
import atexit
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
import arm62      # noqa: E402  (port, helpers, the Unsloth file's pinned size and hash)
import watch62a   # noqa: E402  (Addendum A's job runner, tools and system prompt, unchanged)

HOST = arm62.HOST
MODELS = Path.home() / "models"
BUILDS = {
    "b12631": (MODELS / "llama.cpp-b12631", "e300919cc704ddde3a8a8346a8714f57eeeee71b"),
    "parent": (MODELS / "llama.cpp-pr32825-parent", "1834cc254c77e536930d9bebaf7f85d701a034b6"),
    "head": (MODELS / "llama.cpp-pr32825-head", "7b43c7730485cc77429485d8232c9643e8aab1f2"),
}
FILES = {
    "U": (arm62.MODEL, arm62.MODEL_BYTES, arm62.MODEL_SHA256),
    "N": (MODELS / "inkling2-ggml/Inkling-2-Mini-IQ4_NL.gguf", 17627940768,
          "cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4"),
}
CELLS = [("U0", "b12631", "U"), ("N0", "b12631", "N"), ("U1", "parent", "U"), ("U2", "head", "U"),
         ("N2", "head", "N"), ("U0r", "b12631", "U")]
PLACEMENT = [("P-U0", "b12631", "U"), ("P-U2", "head", "U"), ("P-N0", "b12631", "N")]
CAPACITY = [("C-U0", "b12631", "U"), ("C-N0", "b12631", "N"), ("C-U2", "head", "U")]
SMOKE_CELLS, SMOKE_PLACEMENT, SMOKE_CAPACITY = [("U2", "head", "U")], [("P-U0", "b12631", "U"), ("P-U2", "head", "U")], [("C-U2", "head", "U")]
# Amendment 1 (2026-11-24, ADDENDUM_C_AMENDMENT_1.md): the merged release b12676, which holds #32825 (merged as
# 51831d47) and #32833; run with --tag into runs_c2/, compared there with Addendum C's U0 and U2 by score62c2.py
BUILDS["tag"] = (MODELS / "llama.cpp-b12676", "d35dc4d9e6827db8e08426a2340d8a71ea863cd9")
TAG_CELLS, TAG_PLACEMENT, TAG_CAPACITY = [("U3", "tag", "U"), ("N3", "tag", "N")], [("P-U3", "tag", "U")], [("C-U3", "tag", "U")]
KIT_SHA256 = {  # frozen with the kit, 2026-11-23
    "text62c.txt": "3ffb77ad1f94963fa986021bff7be7aa2ab3f52f50d96be6f30686ea22c769df",
    "prompts62c.json": "692326baabe02efa8b7d9c4928ca45205c1fbfe3c64ddfb49ee60f79aae7f98c",
    "jobs62a.json": watch62a.JOBS_SHA256,
}
ONE = ["-c", "8192", "-np", "1", "-ngl", "99", "--host", "127.0.0.1", "--port", str(arm62.PORT)]
FOUR = ["-c", "32768", "-np", "4", "-ngl", "99", "--host", "127.0.0.1", "--port", str(arm62.PORT)]
LONG, SHORT = (2048, 256), (128, 512)   # (prompt tokens, generated tokens)
REPEATS, SEED, TIMEOUT = 6, 62, 900
THERMAL_WAIT_S = 600
_server = None


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)          # a step's file exists only when the step is complete


def post(path: str, body: dict) -> dict:
    req = urllib.request.Request(HOST + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.loads(r.read())


def thermal_wait(label: str) -> dict:
    """The registered rule: wait up to ten minutes for level 0, then run flagged."""
    t0, level = time.time(), arm62.thermal()
    while level != "0" and time.time() - t0 < THERMAL_WAIT_S:
        time.sleep(30)
        level = arm62.thermal()
    rec = {"level": level, "waited_s": round(time.time() - t0), "flagged": level != "0"}
    if rec["waited_s"] or rec["flagged"]:
        print(f"thermal before {label}: {rec}", flush=True)
    return rec


# ---------------------------------------------------------------- server

def stop_server():
    global _server
    if _server is not None and _server.poll() is None:
        _server.terminate()
        try:
            _server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            _server.kill()
    _server = None


atexit.register(stop_server)


def start_server(build: str, file: str, args: list, log_path: Path, env: dict | None = None, extra: list | None = None) -> bool:
    global _server
    if watch62a.port_busy():
        print("STOP: something already answers on the side port", flush=True)
        sys.exit(2)
    log = open(log_path, "ab")
    _server = subprocess.Popen([str(BUILDS[build][0] / "build/bin/llama-server"), "-m", str(FILES[file][0]), *args,
                                *(extra or [])], stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(env or {})})
    return arm62.wait_ready(_server)


def child(run: Path, name: str, *argv) -> int:
    env = {**os.environ, "PYTHONPATH": str(arm62.HOOK), "AUDIT54_LOG": str(run / f"audit-{name}.jsonl")}
    return subprocess.run([sys.executable, __file__, *argv, "--run", str(run)], env=env).returncode


# ---------------------------------------------------------------- children

def timed(tokens: list, n_predict: int) -> dict:
    """One timed request. The whole prompt must be evaluated and the full count generated, or the run stops."""
    therm = arm62.thermal()
    t0 = time.time()
    d = post("/completion", {"prompt": tokens, "n_predict": n_predict, "temperature": 0, "seed": SEED,
                             "cache_prompt": False, "ignore_eos": True, "return_tokens": True})
    t = d.get("timings") or {}
    row = {"prompt_tokens": len(tokens), "n_predict": n_predict, "wall_s": round(time.time() - t0, 3),
           "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"), "predicted_n": t.get("predicted_n"),
           "prompt_per_second": t.get("prompt_per_second"), "predicted_per_second": t.get("predicted_per_second"),
           "prompt_ms": t.get("prompt_ms"), "predicted_ms": t.get("predicted_ms"), "thermal_before": therm,
           "tokens_sha256": hashlib.sha256(json.dumps(d.get("tokens") or []).encode()).hexdigest(),
           "tokens_returned": len(d.get("tokens") or [])}
    ok = row["prompt_n"] == len(tokens) and not row["cache_n"] and row["predicted_n"] == n_predict
    if not ok:
        print("STOP: a timed request did not evaluate the whole prompt or did not generate the full count:", row, flush=True)
        sys.exit(5)
    return row


def run_cell(run: Path, name: str, smoke: bool):
    """Part 1 and Part 3 for one cell; the server for it is already up."""
    text = (KIT / ("smoke62c.txt" if smoke else "text62c.txt")).read_text(encoding="utf-8")
    prompts = json.loads((KIT / "prompts62c.json").read_text(encoding="utf-8"))["smoke" if smoke else "test"]
    ids = post("/tokenize", {"content": text, "add_special": True})["tokens"]
    if len(ids) < LONG[0]:
        print("STOP: the text is shorter than the long workload:", len(ids), flush=True)
        sys.exit(5)
    long_ids, short_ids = ids[:LONG[0]], ids[:SHORT[0]]
    cell = {"cell": name, "started_utc": utc(), "smoke": smoke, "text_tokens": len(ids),
            "long_ids_sha256": hashlib.sha256(json.dumps(long_ids).encode()).hexdigest(),
            "long": [], "short": [], "texts": []}
    reps = 1 if smoke else REPEATS
    for i in range(reps):
        cell["long"].append(timed(long_ids, LONG[1]))
        print(name, "long", i + 1, cell["long"][-1]["prompt_per_second"], cell["long"][-1]["predicted_per_second"], flush=True)
    for i in range(reps):
        cell["short"].append(timed(short_ids, SHORT[1]))
        print(name, "short", i + 1, cell["short"][-1]["prompt_per_second"], cell["short"][-1]["predicted_per_second"], flush=True)
    for p in prompts:
        d = post("/completion", {"prompt": p["text"], "n_predict": 256, "temperature": 0, "seed": SEED,
                                 "cache_prompt": False, "ignore_eos": True, "return_tokens": True, "n_probs": 5})
        probs = d.get("completion_probabilities") or []
        first = [{"id": x.get("id"), "logprob": x.get("logprob"), "prob": x.get("prob")}
                 for x in ((probs[0].get("top_logprobs") or probs[0].get("top_probs") or []) if probs else [])]
        cell["texts"].append({"id": p["id"], "tokens": d.get("tokens") or [], "first_top5": first,
                              "predicted_n": (d.get("timings") or {}).get("predicted_n")})
        print(name, "text", p["id"], len(cell["texts"][-1]["tokens"]), "tokens", flush=True)
    cell["finished_utc"] = utc()
    (run / "cells").mkdir(exist_ok=True)
    write_json(run / "cells" / f"{name}.json", cell)


def run_capacity(run: Path, name: str, smoke: bool):
    """Addendum A's Part 2 with its job runner: the fixed 12 jobs at 1, 2 and 4 watchers. Each level is kept when it
    is complete, and the thermal rule is applied before each level."""
    import score62a
    data = watch62a.load(KIT / ("smoke62a.json" if smoke else "jobs62a.json"))
    jobs = {j["id"]: j for j in data["jobs"]}
    out = run / name
    out.mkdir(parents=True, exist_ok=True)
    for n in (1, 2, 4):
        done = out / f"level-{n}.json"
        if done.exists():
            continue
        therm = thermal_wait(f"{name} level {n}")
        t0 = time.time()
        with ThreadPoolExecutor(n) as ex:
            rows = list(ex.map(lambda jid: watch62a.run_job("I", data["world"], jobs[jid])[0], data["capacity_subset"]))
        wall = time.time() - t0
        rss = arm62._run("sh", "-c", f"ps -o rss= -p $(lsof -ti tcp:{arm62.PORT} -sTCP:LISTEN | head -1)")
        s = {"at_once": n, "jobs": len(rows), "wall_s": round(wall, 1), "jobs_per_minute": round(len(rows) / wall * 60, 2),
             "median_job_s": sorted(r["seconds"] for r in rows)[len(rows) // 2], "max_job_s": max(r["seconds"] for r in rows),
             "right": sum(score62a.judge(jobs[r["id"]], r)["right"] for r in rows),
             "completion_tokens": sum(t["completion_tokens"] or 0 for r in rows for t in r["turns"]),
             "server_rss_kb": rss, "thermal": therm, "thermal_after": arm62.thermal()}
        s["tokens_per_second"] = round(s["completion_tokens"] / wall, 1)
        for r in rows:           # the final sentence a model wrote is not kept: counts and tool calls only
            r["right"] = bool(score62a.judge(jobs[r["id"]], r)["right"])
            r["final_chars"] = len(r.pop("final") or "")
            for c in r["calls"]:
                if c["name"] in ("alert_owner", "ask_owner"):
                    c["args"] = {k: f"<{len(str(v))} characters>" for k, v in c["args"].items()}
        write_json(done, {"summary": s, "rows": rows})
        print("capacity", name, s, flush=True)


# ---------------------------------------------------------------- driver

def placement(run: Path, name: str, build: str, file: str) -> dict:
    """One short load with the scheduler's debug output on; counts which backend each graph split is given."""
    log = MODELS / "exp062c" / f"{run.name}-{name}.log"       # the raw log stays outside the repository
    log.parent.mkdir(parents=True, exist_ok=True)
    log.unlink(missing_ok=True)
    rec = {"name": name, "build": build, "file": file, "raw_log": str(log).replace(str(Path.home()), "~")}
    if not start_server(build, file, ONE, log, env={"GGML_SCHED_DEBUG": "2"}, extra=["-v"]):
        stop_server()
        return {**rec, "error": "the server did not become ready"}
    try:
        ids = post("/tokenize", {"content": "The harbour clerk opened the ledger and", "add_special": True})["tokens"]
        post("/completion", {"prompt": ids, "n_predict": 8, "temperature": 0, "seed": SEED, "cache_prompt": False})
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"[:200]
    stop_server()
    splits, nodes, graphs = [], {}, 0
    node_rx = re.compile(r"node #\s*\d+ \(\s*([A-Z_0-9]+)\s*\):.*?\[\s*(\S+)\s+[^\]]*\] use=")
    for line in open(log, errors="replace"):
        if "## SPLIT #" in line:
            if "## SPLIT #0:" in line:          # a new graph: keep the last one printed
                splits, nodes, graphs = [], {}, graphs + 1
            splits.append(line.split("## SPLIT #", 1)[1].split(":", 1)[1].split("#")[0].strip())
            continue
        m = node_rx.search(line)
        if m and m.group(1) in ("MUL_MAT", "MUL_MAT_ID"):
            nodes.setdefault(m.group(1), {}).setdefault(m.group(2), 0)
            nodes[m.group(1)][m.group(2)] += 1
    by = {}
    for s in splits:
        by[s] = by.get(s, 0) + 1
    rec.update({"log_bytes": log.stat().st_size, "graphs_printed": graphs, "splits_in_last_graph": len(splits),
                "splits_by_backend_last_graph": by, "mul_mat_nodes_by_op_and_backend_last_graph": nodes})
    return rec


def pinned(smoke: bool) -> dict:
    out = {"builds": {}, "files": {}, "kit": {}}
    for b, (repo, sha) in BUILDS.items():
        got = arm62._run("git", "-C", str(repo), "rev-parse", "HEAD")
        out["builds"][b] = {"commit": got, "ok": got == sha and (repo / "build/bin/llama-server").exists()}
    for f, (path, size, sha) in FILES.items():
        print(f"hashing file {f} (about half a minute) ...", flush=True)
        got = arm62.sha256_file(path) if path.exists() else None
        out["files"][f] = {"sha256": got, "bytes": path.stat().st_size if path.exists() else None,
                           "ok": got == sha and path.stat().st_size == size}
    for k, sha in KIT_SHA256.items():
        got = arm62.sha256_file(KIT / k)
        out["kit"][k] = {"sha256": got, "ok": got == sha}
    out["ok"] = all(v["ok"] for part in ("builds", "files", "kit") for v in out[part].values())
    return out


def run_all(run: Path, smoke: bool):
    run.mkdir(parents=True, exist_ok=True)
    meta_path = run / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"starts": []}
    start = {"started_utc": utc(), "argv": sys.argv, "smoke": smoke, "python": sys.version,
             "macos": arm62._run("sw_vers", "-productVersion"), "ollama_version": arm62._run("ollama", "--version"),
             "ollama_ps_before": arm62._run("ollama", "ps"),
             "other_llama_servers": arm62._run("sh", "-c", "pgrep -fl llama-server | sed 's/ -m .*//'"),
             "thermal_start": arm62.thermal(), "repeats": REPEATS, "seed": SEED, "long": LONG, "short": SHORT}
    start["pinned"] = pinned(smoke)
    meta["starts"].append(start)
    write_json(meta_path, meta)
    if not start["pinned"]["ok"]:
        print("STOP: a pinned input does not match:", json.dumps(start["pinned"], indent=1), flush=True)
        sys.exit(2)
    meta.setdefault("steps", {})

    def note(name, **kw):
        meta["steps"].setdefault(name, {}).update(kw)
        write_json(meta_path, meta)

    sm = ["--smoke"] if smoke else []
    for name, build, file in (SMOKE_CELLS if smoke else CELLS):
        if (run / "cells" / f"{name}.json").exists():
            print("cell", name, "is complete, skipped", flush=True)
            continue
        therm = thermal_wait(f"cell {name}")
        if not start_server(build, file, ONE, run / f"server-{name}.log"):
            stop_server()
            note(name, error="the server did not become ready", thermal=therm, utc=utc())
            print(f"cell {name}: the server did not become ready; recorded, the run goes on", flush=True)
            continue
        rc = child(run, name, "--cell", name, *sm)
        stop_server()
        note(name, rc=rc, thermal=therm, build=build, file=file, utc=utc())
        if rc == 5:
            print(f"STOPPED: cell {name} failed a check on a timed request", flush=True)
            sys.exit(5)
        if rc != 0:
            print(f"cell {name} ended with code {rc}; recorded, the run goes on", flush=True)

    if not (run / "placement.json").exists():
        recs = [placement(run, n, b, f) for n, b, f in (SMOKE_PLACEMENT if smoke else PLACEMENT)]
        write_json(run / "placement.json", recs)
        for r in recs:
            print("placement", json.dumps(r), flush=True)

    for name, build, file in (SMOKE_CAPACITY if smoke else CAPACITY):
        if all((run / name / f"level-{n}.json").exists() for n in (1, 2, 4)):
            print("capacity", name, "is complete, skipped", flush=True)
            continue
        if name == "C-U2" and not (run / "cells" / "U2.json").exists():
            note(name, skipped="the head build did not complete its cell with file U")
            print("capacity C-U2 not run: the head build did not complete its cell with file U", flush=True)
            continue
        if not start_server(build, file, FOUR, run / f"server-{name}.log"):
            stop_server()
            note(name, error="the server did not become ready", utc=utc())
            continue
        rc = child(run, name, "--capacity", name, *sm)
        stop_server()
        note(name, rc=rc, build=build, file=file, utc=utc())

    meta["finished"] = {"utc": utc(), "ollama_ps_after": arm62._run("ollama", "ps"), "thermal_end": arm62.thermal()}
    write_json(meta_path, meta)
    print("all done", run, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cell")
    ap.add_argument("--capacity")
    ap.add_argument("--run", required=True)
    ap.add_argument("--tag", action="store_true", help="Amendment 1: the cells on the release b12676 only")
    a = ap.parse_args()
    if a.tag:
        CELLS[:], PLACEMENT[:], CAPACITY[:] = TAG_CELLS, TAG_PLACEMENT, TAG_CAPACITY
    run = Path(a.run).resolve()
    if a.all:
        run_all(run, a.smoke)
    elif a.cell:
        run_cell(run, a.cell, a.smoke)
    elif a.capacity:
        run_capacity(run, a.capacity, a.smoke)


if __name__ == "__main__":
    main()
