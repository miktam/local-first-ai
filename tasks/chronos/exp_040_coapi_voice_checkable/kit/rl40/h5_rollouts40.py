#!/usr/bin/env python3
"""
exp_040 — rollouts for the H5 verifier audit (DESIGN §9 H5; exp_039 amendment of 2026-10-14 kept: mlx-lm-lora never
writes rollouts, so the selected checkpoint answers 200 training prompts, one each, at the training sampling
(temperature 0.9, top_p 0.95, top_k 20), thinking off, stratified by language × type (seed 40)).

  python rl40/h5_rollouts40.py RUN_DIR CKPT [--n 200] [--top 40] [--out DIR]

Every rollout is scored by reward40. The 40 with the highest total go to two blind readers (Fable and Opus), each
alone. audit_blind.jsonl carries the language, the question as shown, the material, the answer and whether code
found the final line right — the audit asks whether a right final line rests on an explanation that supports it,
so the reader must know which lines code paid for. No scores, components, ids, labels or checkpoint names; the key
is kept apart (audit_key.jsonl). Gaming = a right final line on an answer that does not explain it, or a guessed label.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV = HERE.parent
sys.path[:0] = [str(HERE), str(CV / "rl")]
import h5_rollouts as H     # noqa: E402  (stratified, exp_039's)
import reward40 as W        # noqa: E402
import select_ckpt as S     # noqa: E402

SEED, TEMP, TOP_P, TOP_K = 40, 0.9, 0.95, 20
TRAIN = CV / "data" / "rl40" / "train" / "train.jsonl"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("ckpt", help="checkpoint step, e.g. 75")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    run = Path(a.run_dir).resolve()
    out = Path(a.out) if a.out else run.parent / f"h5_40_ckpt{a.ckpt}"
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    picked = H.stratified(rows, a.n, random.Random(SEED))
    path = out / "rollouts.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
    todo = [r for r in picked if r["id"] not in done]
    if todo:
        import mlx.core as mx
        mx.random.seed(SEED)
        model, tok = S.load(S.checkpoints(run)[a.ckpt], run)
        sampler = make_sampler(temp=TEMP, top_p=TOP_P, top_k=TOP_K)
        with path.open("a", encoding="utf-8") as f:
            for i, r in enumerate(todo, 1):
                text = tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}],
                                               add_generation_prompt=True, tokenize=False, enable_thinking=False)
                if not text.endswith(S.THINK_OFF):
                    sys.exit("rendered prompt is not think:false; refusing")
                completion = generate(model, tok, prompt=text, max_tokens=W.CAP, sampler=sampler, verbose=False)
                x = W.analyse(completion, r["answer"])
                f.write(json.dumps({"id": r["id"], "type": r["type"], "lang": r["lang"], "verdict": x["facts"]["verdict"],
                                    "total": x["total"], "components": x["components"], "facts": x["facts"],
                                    "completion": completion}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{i}/{len(todo)} {r['id']} {x['facts']['verdict']} total {x['total']:+.2f}", flush=True)

    rs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    by_id = {r["id"]: r for r in rows}
    top = sorted(rs, key=lambda r: (-r["total"], r["id"]))[: a.top]
    order = random.Random(SEED + 1).sample(top, len(top))          # the reader sees no ranking
    with (out / "audit_blind.jsonl").open("w", encoding="utf-8") as fb, (out / "audit_key.jsonl").open("w", encoding="utf-8") as fk:
        for n, r in enumerate(order, 1):
            meta = json.loads(by_id[r["id"]]["answer"])
            fb.write(json.dumps({"item": n, "language": r["lang"], "question": meta["q"], "retrieved_material": meta["context"],
                                 "answer": r["completion"], "final_line_paid_as_right": r["verdict"] == "right"},
                                ensure_ascii=False) + "\n")
            fk.write(json.dumps({"item": n, "id": r["id"], "type": r["type"], "label": meta.get("label"), "value": meta.get("value"),
                                 "verdict": r["verdict"], "total": r["total"], "components": r["components"], "facts": r["facts"]},
                                ensure_ascii=False) + "\n")
    summary = {"ckpt": a.ckpt, "n": len(rs), "top": len(top),
               "sampling": {"temperature": TEMP, "top_p": TOP_P, "top_k": TOP_K, "seed": SEED},
               "exact": round(sum(r["verdict"] == "right" for r in rs) / len(rs), 4),
               "mean_total": round(sum(r["total"] for r in rs) / len(rs), 4),
               "top_total_range": [top[-1]["total"], top[0]["total"]] if top else None,
               "top_right": sum(r["verdict"] == "right" for r in top),
               "copy_triggered": sum(r["components"]["copy"] == -1.0 for r in rs),
               "at_cap": sum(r["facts"]["at_cap"] for r in rs), "repeat_triggered": sum(r["components"]["repeat"] == -1.0 for r in rs)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
