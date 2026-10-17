#!/usr/bin/env python3
"""
exp_040 — the offline dynamic-sampling filter (BUILD §5; DAPO's "keep groups with variance", done once, offline).

  python rl40/dapo_filter.py --host URL run    [--n 6] [--workers 2] [--limit N]   # sample, resumable
  python rl40/dapo_filter.py select [--partial]                                    # write the trainer's data dir

Started by rl40/dapo_filter.sh, which owns the Ollama 0.33.0 side server (as rl/measure.sh). The base at step 0's
quantisation (BASE_OURS_IMAT, digest pinned in rl/run_arm.py) answers every TRAIN prompt of data/rl40/prompts n times,
with the row's own system prompt and user turn (prompt40.render40 — train = filter = serve), thinking off, and
sampling as the GRPO rollouts sample (rl/grpo_config.yaml: temperature 0.9; mlx-lm-lora's forced top_p 0.95,
top_k 20, min_p 0; no repeat or presence penalty; 512 tokens). Each sample has its own seed from (row id, k).
Every sample is scored by reward40 (rev 3) when drawn and re-scored at select; the filter uses only the
answer verdict.

Samples carry the row's prompt hash: a row whose prompt changes is redrawn on the next run, the others
keep their samples. select keeps a train row when the base is right on 1..n−1 of its n samples (0 or n: no variance in a GRPO group,
no gradient). Valid is copied unfiltered. Twin links to dropped rows are cleared. Writes data/rl40/train/
{train,valid}.jsonl (mlx-lm-lora's data dir) and filter_manifest.json. Samples stay in runs/ (gitignored).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV = HERE.parent
sys.path[:0] = [str(CV), str(HERE), str(CV / "rl")]
import reward as R      # noqa: E402
import reward40 as W    # noqa: E402
import run_arm as ra    # noqa: E402  (check_server, BASE_OURS_IMAT, DIGEST — pins, unchanged)

MODEL = ra.BASE_OURS_IMAT
PROMPTS = CV / "data" / "rl40" / "prompts"
OUT_DATA = CV / "data" / "rl40" / "train"
RUN_DIR = CV / "runs" / "exp040_dapo"
OPTIONS = {"temperature": 0.9, "top_p": 0.95, "top_k": 20, "min_p": 0.0, "repeat_penalty": 1.0,
           "presence_penalty": 0.0, "num_predict": R.CAP, "num_ctx": 4096}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def prompt_sha(row: dict) -> str:
    """A sample counts for a row only while the row's system prompt and user turn are unchanged."""
    return hashlib.sha256(f"{row['system']}\0{row['prompt']}".encode()).hexdigest()[:16]


def seed_of(rid: str, k: int) -> int:
    return int(hashlib.sha256(f"40:{rid}:{k}".encode()).hexdigest()[:8], 16)


def chat(host: str, row: dict, seed: int) -> dict:
    body = json.dumps({"model": MODEL, "stream": False, "think": False, "options": {**OPTIONS, "seed": seed},
                       "messages": [{"role": "system", "content": row["system"]},
                                    {"role": "user", "content": row["prompt"]}]}).encode()
    req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)


def score(completion: str, answer: str, ntok: int | None) -> dict:
    R.set_token_counter(lambda s, n=ntok: n if n is not None else int(len(s.split()) * 1.4))
    a = W.analyse(completion, answer)
    return {"verdict": a["facts"]["verdict"], "answer_term": a["components"]["answer"], "total": a["total"],
            "final_line": a["facts"]["final_line"]}


def cmd_run(a) -> None:
    rows = load(PROMPTS / "train.jsonl")
    if a.limit:
        rows = rows[: a.limit]
    digest = ra.check_server(a.host, MODEL)
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    samples = RUN_DIR / "samples.jsonl"
    done = {(s["id"], s["k"], s.get("prompt_sha")) for s in load(samples)} if samples.exists() else set()
    todo = [(r, k) for r in rows for k in range(a.n) if (r["id"], k, prompt_sha(r)) not in done]
    meta = RUN_DIR / "run_meta.json"
    if not meta.exists():
        meta.write_text(json.dumps({"model": MODEL, "digest": digest, "server": ra.get(a.host, "/api/version")["version"],
                                    "options": OPTIONS, "n": a.n, "prompt_sets": [sha(PROMPTS / "train.jsonl")],
                                    "reward_rev": W.REWARD_REV, "started": dt.datetime.now().isoformat(timespec="seconds")},
                                   indent=1) + "\n")
    else:
        m = json.loads(meta.read_text())
        if m["n"] != a.n or m["digest"] != digest or m["options"] != OPTIONS:
            sys.exit("REFUSED: n, sampling options or model differ from the run already in runs/exp040_dapo — move it aside first")
        if sha(PROMPTS / "train.jsonl") not in m["prompt_sets"]:       # rows whose prompt changed are redrawn
            m["prompt_sets"].append(sha(PROMPTS / "train.jsonl"))
            meta.write_text(json.dumps(m, indent=1) + "\n")
    print(f"{len(rows)} rows × {a.n} = {len(rows) * a.n} samples; {len(rows) * a.n - len(todo)} already done "
          f"for the current prompts, {len(todo)} to go", flush=True)
    t0, n_done = time.time(), 0
    with open(samples, "a", encoding="utf-8") as f, cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(chat, a.host, r, seed_of(r["id"], k)): (r, k) for r, k in todo}
        for fut in cf.as_completed(futs):
            r, k = futs[fut]
            resp = fut.result()
            text = resp["message"]["content"]
            rec = {"id": r["id"], "k": k, "seed": seed_of(r["id"], k), "prompt_sha": prompt_sha(r), "completion": text,
                   "eval_count": resp.get("eval_count"), "prompt_eval_count": resp.get("prompt_eval_count"),
                   **score(text, r["answer"], resp.get("eval_count"))}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            n_done += 1
            if n_done % 60 == 0 or n_done == len(todo):
                el = time.time() - t0
                eta = el / n_done * (len(todo) - n_done)
                print(f"{n_done}/{len(todo)}  {el / 60:.1f} min  {n_done / el:.2f}/s  eta {eta / 3600:.1f} h", flush=True)
    print("sampling done; next: dapo_filter.py select", flush=True)


def cmd_select(a) -> None:
    rows = load(PROMPTS / "train.jsonl")
    m = json.loads((RUN_DIR / "run_meta.json").read_text())
    n = m["n"]
    by_id = {r["id"]: r for r in rows}
    cur = {rid: prompt_sha(r) for rid, r in by_id.items()}
    right, seen, verdicts = Counter(), Counter(), Counter()
    stale = 0
    for s in load(RUN_DIR / "samples.jsonl"):       # re-scored here: the verdict always follows the current reward40
        if s.get("prompt_sha") != cur.get(s["id"]):  # drawn for an earlier prompt of this row (or a dropped row)
            stale += 1
            continue
        v = score(s["completion"], by_id[s["id"]]["answer"], s["eval_count"])["verdict"]
        seen[s["id"]] += 1
        right[s["id"]] += v == "right"
        verdicts[v] += 1
    incomplete = [r["id"] for r in rows if seen[r["id"]] < n]
    if incomplete and not a.partial:
        sys.exit(f"REFUSED: {len(incomplete)} rows have fewer than {n} samples (run again to resume, or --partial)")
    sampled = [r for r in rows if seen[r["id"]] >= n]
    kept = [r for r in sampled if 0 < right[r["id"]] < n]
    ids = {r["id"] for r in kept}
    cleared = 0
    for r in kept:
        if r["twin_of"] and r["twin_of"] not in ids:
            r["twin_of"] = None
            cleared += 1
    OUT_DATA.mkdir(parents=True, exist_ok=True)
    (OUT_DATA / "train.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept), encoding="utf-8")
    (OUT_DATA / "valid.jsonl").write_bytes((PROMPTS / "valid.jsonl").read_bytes())

    def by(rs, key):
        return dict(sorted(Counter(key(r) for r in rs).items()))

    hist = {f"{lt}": dict(sorted(Counter(right[r["id"]] for r in sampled if f"{r['lang']}/{r['type']}" == lt).items()))
            for lt in sorted({f"{r['lang']}/{r['type']}" for r in sampled})}
    commit = subprocess.run(["git", "-C", str(CV), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    man = {
        "experiment": "exp_040", "built_by": "rl40/dapo_filter.py select", "casasol_commit": commit, "run": m,
        "partial": bool(incomplete), "incomplete_rows": len(incomplete), "stale_samples_ignored": stale,
        "sampled": len(sampled), "kept": len(kept), "dropped_all_wrong": sum(right[r["id"]] == 0 for r in sampled),
        "dropped_all_right": sum(right[r["id"]] == n for r in sampled), "twin_links_cleared": cleared,
        "base_right_rate": round(sum(right[r["id"]] for r in sampled) / max(1, n * len(sampled)), 4),
        "verdicts": dict(verdicts), "right_count_histogram_by_lang_type": hist,
        "kept_by_lang_type": by(kept, lambda r: f"{r['lang']}/{r['type']}"),
        "kept_by_module": by(kept, lambda r: (r["tema"] or "-").split(":")[0]),
        "sha256": {"train.jsonl": sha(OUT_DATA / "train.jsonl"), "valid.jsonl": sha(OUT_DATA / "valid.jsonl"),
                   "samples.jsonl": sha(RUN_DIR / "samples.jsonl"), "prompts/train.jsonl": sha(PROMPTS / "train.jsonl"),
                   "code/reward40.py": sha(HERE / "reward40.py")},
    }
    (OUT_DATA / "filter_manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: man[k] for k in ("sampled", "kept", "dropped_all_wrong", "dropped_all_right", "base_right_rate",
                                          "verdicts", "kept_by_lang_type", "partial")}, indent=1, ensure_ascii=False))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--host", required=True)
    r.add_argument("--n", type=int, default=6)
    r.add_argument("--workers", type=int, default=2, help="= the side server's OLLAMA_NUM_PARALLEL")
    r.add_argument("--limit", type=int, default=0, help="first N rows only (smoke test)")
    s = sub.add_parser("select")
    s.add_argument("--partial", action="store_true")
    a = ap.parse_args(argv)
    cmd_run(a) if a.cmd == "run" else cmd_select(a)


if __name__ == "__main__":
    main()
