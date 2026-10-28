#!/usr/bin/env python3
"""exp_059: a Mac arm. Sends each sealed enquiry to a Kev-4B `/v1/systemone` server the way the frozen knife does
(masked state + LEAD_SCHEMA, one request per enquiry) and keeps the raw answers with full precision and the wall time.

    python3 run59_mac.py <arm> [--url http://127.0.0.1:18442] [--limit N]

<arm> is one of mini-a, mini-b, mbp (any other name is accepted for a dev run). Output: raw/<arm>.jsonl, one line per
enquiry {id, answers, usage, ms, t}; resumable; a finished arm writes raw/<arm>.done.json with hashes and the server's
/props, and a second pass is refused. --limit N is a smoke test: it writes raw/smoke_<arm>.jsonl and no done file.

The server must be llama.cpp b11711 serving Kev-4B-Q8_0.gguf (sha256 7c2ebed9…) with the gate's flags:
    llama-server -m Kev-4B-Q8_0.gguf --host 127.0.0.1 --port 18442 -c 16384 -b 8192 -ub 8192
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
EXP044 = EXP.parent / "exp_044_kev_lead_triage_rev4"
KEV_SHA = "7c2ebed90560522c2801389db482ac1dc4c36d828f201f6074c1d60e433948da"

sys.path.insert(0, str(HERE))
import render59  # noqa: E402  (the frozen knife loader and the hash checks)

TEST_SHA = render59.TEST_SHA   # the public copy of the sealed set, the same on every arm


def get_json(url: str, body: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception as e:   # noqa: BLE001
        return f"<{e}>"


def host_info() -> dict:
    return {"hostname": socket.gethostname(), "machine": platform.machine(), "macos": sh(["sw_vers", "-productVersion"]),
            "chip": sh(["sysctl", "-n", "machdep.cpu.brand_string"]), "python": platform.python_version()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("--url", default="http://127.0.0.1:18442")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N enquiries, separate output, no done file")
    ap.add_argument("--test", type=Path, default=EXP044 / "data" / "test.jsonl")
    a = ap.parse_args()

    if render59.sha256(a.test) != TEST_SHA:
        sys.exit(f"REFUSED: {a.test} is not the sealed exp_044 test set")
    K = render59.knife()
    rows = [json.loads(l) for l in a.test.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.limit:
        rows = rows[:a.limit]

    props = get_json(f"{a.url}/props")
    props.pop("chat_template", None)
    model_path = props.get("model_path", "")
    model_sha = render59.sha256(Path(model_path)) if model_path and Path(model_path).exists() else None
    if model_sha is not None and model_sha != KEV_SHA:
        sys.exit(f"REFUSED: the server's model {model_path} is not Kev-4B-Q8_0 {KEV_SHA[:12]} (got {model_sha[:12]})")
    print(f"server {a.url}: model {model_path or '?'} sha256 {(model_sha or 'unverified')[:12]}, slots {props.get('total_slots')}")

    (EXP / "raw").mkdir(exist_ok=True)
    out = EXP / "raw" / (f"smoke_{a.arm}.jsonl" if a.limit else f"{a.arm}.jsonl")
    done_file = EXP / "raw" / f"{a.arm}.done.json"
    if not a.limit and done_file.exists():
        sys.exit(f"REFUSED: arm {a.arm} has had its one pass ({done_file})")
    done = {json.loads(l)["id"] for l in out.read_text().splitlines()} if out.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    print(f"{a.arm}: {len(done)} done, {len(todo)} to go → {out}")

    t_start = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    with out.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            body = {"state": K.masked(r["text"]), "questions": K.LEAD_SCHEMA}     # exactly what kev_knife_rev4.ask() sends
            t0 = time.perf_counter()
            resp = get_json(f"{a.url}/v1/systemone", body)
            ms = round((time.perf_counter() - t0) * 1000, 1)
            f.write(json.dumps({"id": r["id"], "answers": resp["answers"], "usage": resp.get("usage"), "ms": ms,
                                "t": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds")},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"  {len(done) + i}/{len(rows)}  last {ms} ms", flush=True)

    n = sum(1 for _ in out.open())
    if a.limit:
        print(f"smoke done: {n} rows in {out}")
        return
    if n >= len(rows):
        done_file.write_text(json.dumps({
            "arm": a.arm, "n": n, "started": t_start, "finished": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "raw_sha256": render59.sha256(out), "test_sha256": TEST_SHA, "model_path": model_path, "model_sha256": model_sha,
            "server_props": props, "host": host_info(), "url": a.url,
        }, indent=1) + "\n")
        print(f"done: {n} rows, {done_file}")


if __name__ == "__main__":
    main()
