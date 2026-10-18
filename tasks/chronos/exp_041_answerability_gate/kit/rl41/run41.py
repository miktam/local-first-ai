#!/usr/bin/env python3
"""
exp_041 — run one gate on the dev set or on v3 (started by rl41/run41.sh, which owns the llama-server).

  python rl41/run41.py --host URL GATE dev       # resumable; writes runs/exp041_dev_<GATE>/probs.jsonl
  python rl41/run41.py --host URL GATE v3        # one pass per gate, only after data/gate41/thresholds.json exists

v3 rows are rendered exactly as exp_040 rendered them for B0i (prompt40.render40: same retrieval, withheld chunks
removed from the index, options after the question for categorical / yes-no), so the gate sees the material and
the question the voice saw. Output carries probabilities and ids only; no v3 text is printed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV = HERE.parent
sys.path[:0] = [str(CV), str(HERE), str(CV / "rl40"), str(CV / "rl")]
import gate41 as GT         # noqa: E402

DEV = CV / "data" / "gate41" / "dev.jsonl"
THRESHOLDS = CV / "data" / "gate41" / "thresholds.json"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def v3_rows() -> list[dict]:
    import guide_bm25 as g
    import prompt40 as P
    import run_arm as ra
    import run_arm40 as A
    ra.use_index(ra.INDEX_LIVE)
    idx = {lang: g.load_index(lang) for lang in g.LANGS}
    out = []
    for row in A.load_set("v3"):
        r = P.render40(row["q"], row["lang"], idx, A.options_for(row), withhold=row.get("withhold") or [])
        out.append({"id": row["id"], "lang": row["lang"], "material": r["context"], "question": r["shown_q"]})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("gate", choices=sorted(GT.GATES))
    ap.add_argument("set", choices=["dev", "v3"])
    ap.add_argument("--limit", type=int, default=0, help="dev only: first N rows (smoke)")
    a = ap.parse_args(argv)
    if a.limit and a.set == "v3":
        sys.exit("REFUSED: v3 is scored whole, once")
    if not sha(GT.GATES[a.gate]["gguf"]).startswith(GT.GATES[a.gate]["sha"]):
        sys.exit(f"REFUSED: {a.gate} model file does not match its pin")
    if a.set == "dev":
        if not sha(DEV).startswith((CV / "data/gate41/dev.sha256").read_text().strip()[:16]):
            sys.exit("REFUSED: dev set does not match its sha256")
        rows = [json.loads(l) for l in DEV.read_text(encoding="utf-8").splitlines() if l.strip()]
        rows = rows[: a.limit] if a.limit else rows
        out = CV / "runs" / f"exp041_dev_{a.gate}"
    else:
        if not THRESHOLDS.exists():
            sys.exit("REFUSED: no v3 pass before the thresholds are frozen (data/gate41/thresholds.json)")
        out = CV / "runs" / f"exp041_v3_{a.gate}"
        if (out / "done.json").exists():
            sys.exit(f"REFUSED: {a.gate} already has its one v3 pass")
        rows = v3_rows()
    out.mkdir(parents=True, exist_ok=True)
    path = out / "probs.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    t0 = time.time()
    with path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            res = GT.ask(a.gate, a.host, r["material"], r["question"])
            f.write(json.dumps({"id": r["id"], "p": res["p"], "tokens": res["tokens"]}) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"{a.gate}/{a.set} {len(done) + i}/{len(rows)} · {time.time() - t0:.0f}s", flush=True)
    n = sum(1 for _ in path.open())
    if n >= len(rows) and not a.limit:
        (out / "done.json").write_text(json.dumps({"gate": a.gate, "set": a.set, "n": n, "gguf_sha": sha(GT.GATES[a.gate]["gguf"])[:16],
                                                   "probs_sha256": sha(path), "seconds": round(time.time() - t0)}) + "\n")
    print(f"{a.gate}/{a.set}: {n}/{len(rows)} rows")


if __name__ == "__main__":
    main()
