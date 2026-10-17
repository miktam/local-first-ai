#!/usr/bin/env python3
"""
exp_040 — checkpoint selection (DESIGN §7): the checkpoint with the highest exact match on the validation prompts
(data/rl40/train/valid.jsonl, M5 T42–43 bank items in EN/PL/ES + the valid arithmetic templates), one greedy answer
per prompt, thinking off. Exact match = reward40's answer verdict "right". Ties → the earlier step (less drift);
mean reward is reported, never used. No eval set is read. "base" (no adapter) is scored as a reference only.

  python rl40/select_ckpt40.py RUN_DIR [--out DIR] [--only base,25,...] [--limit N]

Generation, checkpoint discovery and the think:false check are exp_039's (rl/select_ckpt.py), imported unchanged.
Each checkpoint's answers are written as they are produced, so an interrupted run resumes where it stopped.
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
sys.path[:0] = [str(HERE), str(CV / "rl")]
import reward40 as W        # noqa: E402
import select_ckpt as S     # noqa: E402

VALID = CV / "data" / "rl40" / "train" / "valid.jsonl"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    ap.add_argument("--only", help="comma list of checkpoints, e.g. base,25,78")
    ap.add_argument("--limit", type=int, help="first N validation prompts (sanity check only)")
    a = ap.parse_args(argv)
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    run = Path(a.run_dir).resolve()
    out = Path(a.out) if a.out else run.parent / f"select40_{run.name}"
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in VALID.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[: a.limit] if a.limit else rows
    cks = S.checkpoints(run)
    if a.only:
        cks = {k: v for k, v in cks.items() if k in a.only.split(",")}

    for name, adapter in cks.items():
        path = out / f"ckpt_{name}.jsonl"
        done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        todo = [r for r in rows if r["id"] not in done]
        if not todo:
            continue
        model, tok = S.load(adapter, run)
        sampler = make_sampler(temp=0.0)
        t0 = time.time()
        with path.open("a", encoding="utf-8") as f:
            for i, r in enumerate(todo, 1):
                text = tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}],
                                               add_generation_prompt=True, tokenize=False, enable_thinking=False)
                if not text.endswith(S.THINK_OFF):
                    sys.exit("rendered prompt is not think:false; refusing")
                completion = generate(model, tok, prompt=text, max_tokens=W.CAP, sampler=sampler, verbose=False)
                x = W.analyse(completion, r["answer"])
                f.write(json.dumps({"ckpt": name, "id": r["id"], "type": r["type"], "lang": r["lang"],
                                    "verdict": x["facts"]["verdict"], "total": x["total"], "components": x["components"],
                                    "facts": x["facts"], "completion": completion}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{name}] {i}/{len(todo)} {x['facts']['verdict']} · {time.time() - t0:.0f}s", flush=True)
        del model

    summary = {}
    for name in S.checkpoints(run):
        p = out / f"ckpt_{name}.jsonl"
        if not p.exists():
            continue
        rs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        summary[name] = {"n": len(rs), "exact": round(sum(r["verdict"] == "right" for r in rs) / len(rs), 4),
                         "mean_total": round(sum(r["total"] for r in rs) / len(rs), 4),
                         "verdicts": {v: sum(r["verdict"] == v for r in rs) for v in ("right", "wrong", "hedge", "missing")},
                         "at_cap": sum(r["facts"]["at_cap"] for r in rs)}
    cands = {k: v for k, v in summary.items() if k != "base" and v["n"] == len(rows)}
    selected = max(sorted(cands, key=int), key=lambda k: cands[k]["exact"]) if cands else None
    result = {"rule": "highest exact match on the validation prompts, greedy, thinking off; ties → earlier step",
              "n_valid": len(rows), "selected": selected, "checkpoints": summary,
              "adapter_sha256": {k: hashlib.sha256(v.read_bytes()).hexdigest()
                                 for k, v in S.checkpoints(run).items() if v is not None},
              "valid_sha256": hashlib.sha256(VALID.read_bytes()).hexdigest()}
    (out / "select.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: (v["exact"], v["mean_total"], v["n"]) for k, v in summary.items()}), "→ selected:", selected)


if __name__ == "__main__":
    main()
