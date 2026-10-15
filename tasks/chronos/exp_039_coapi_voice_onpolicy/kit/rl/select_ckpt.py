#!/usr/bin/env python3
"""
exp_039 — checkpoint selection (DESIGN §7): the checkpoint with the highest mean reward on the
validation prompts (M5 T42–43 + F25), one greedy answer per prompt, thinking off. Reward only;
never judged; no eval set is read. Ties go to the earlier step (less drift from the base).

  python rl/select_ckpt.py RUN_DIR [--out DIR] [--only base,25,...] [--limit N]

RUN_DIR holds the trainer's adapter_config.json and NNNNNNN_adapters.safetensors checkpoints;
`adapters.safetensors` is the final step. "base" (the untrained 4B, no adapter) is scored too,
as a descriptive reference, not a candidate. Each checkpoint's answers are written as they are
produced, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV = HERE.parent
os.environ.setdefault("EXP039_GLOSSARY", str(CV / "data" / "rl" / "glossary.json"))   # same glossary as training
sys.path.insert(0, str(HERE))
import reward as R  # noqa: E402

SNAPSHOT = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
THINK_OFF = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def checkpoints(run: Path) -> dict[str, Path | None]:
    found = {"base": None}
    for p in sorted(run.glob("[0-9]*_adapters.safetensors")):
        found[str(int(p.name.split("_")[0]))] = p
    final = run / "adapters.safetensors"
    iters = (run / "config_resolved.yaml").read_text(encoding="utf-8")
    n = int(re.search(r"^iters: (\d+)$", iters, re.M).group(1))
    if str(n) not in found:
        found[str(n)] = final
    return found


def load(adapter: Path | None, run: Path):
    from mlx_lm import load as mlx_load
    if adapter is None:
        return mlx_load(str(SNAPSHOT))
    tmp = Path(tempfile.mkdtemp(prefix="exp039-ckpt-"))
    (tmp / "adapter_config.json").write_text((run / "adapter_config.json").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp / "adapters.safetensors").symlink_to(adapter.resolve())
    return mlx_load(str(SNAPSHOT), adapter_path=str(tmp))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out")
    ap.add_argument("--only", help="comma list of checkpoints, e.g. base,25,105")
    ap.add_argument("--limit", type=int, help="first N validation prompts (sanity check only)")
    a = ap.parse_args(argv)
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    run = Path(a.run_dir).resolve()
    out = Path(a.out) if a.out else run.parent / f"select_{run.parent.name}"
    out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (CV / "data" / "rl" / "valid.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = rows[: a.limit] if a.limit else rows
    cks = checkpoints(run)
    if a.only:
        cks = {k: v for k, v in cks.items() if k in a.only.split(",")}

    for name, adapter in cks.items():
        path = out / f"ckpt_{name}.jsonl"
        done = {json.loads(l)["id"] for l in path.read_text(encoding="utf-8").splitlines()} if path.exists() else set()
        todo = [r for r in rows if r["id"] not in done]
        if not todo:
            continue
        model, tok = load(adapter, run)
        sampler = make_sampler(temp=0.0)
        t0 = time.time()
        with path.open("a", encoding="utf-8") as f:
            for i, r in enumerate(todo, 1):
                text = tok.apply_chat_template([{"role": "system", "content": r["system"]}, {"role": "user", "content": r["prompt"]}],
                                               add_generation_prompt=True, tokenize=False, enable_thinking=False)
                if not text.endswith(THINK_OFF):
                    sys.exit("rendered prompt is not think:false; refusing")
                completion = generate(model, tok, prompt=text, max_tokens=R.CAP, sampler=sampler, verbose=False)
                x = R.analyse(completion, r["answer"])
                f.write(json.dumps({"ckpt": name, "id": r["id"], "type": r["type"], "lang": r["lang"], "total": x["total"],
                                    "components": x["components"], "facts": x["facts"], "completion": completion},
                                   ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{name}] {i}/{len(todo)} total {x['total']:+.2f} · {time.time() - t0:.0f}s", flush=True)
        del model

    # ---- summary over whatever has been scored
    summary = {}
    for name in checkpoints(run):
        p = out / f"ckpt_{name}.jsonl"
        if not p.exists():
            continue
        rs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        comp = {c: [r["components"][c] for r in rs if r["components"][c] is not None] for c in R.COMPONENTS}
        summary[name] = {"n": len(rs), "mean_total": round(sum(r["total"] for r in rs) / len(rs), 4),
                         "components": {c: round(sum(v) / len(v), 3) for c, v in comp.items() if v},
                         "at_cap": sum(r["facts"]["at_cap"] for r in rs)}
    cands = {k: v for k, v in summary.items() if k != "base" and v["n"] == len(rows)}
    selected = max(sorted(cands, key=int), key=lambda k: cands[k]["mean_total"]) if cands else None
    result = {"rule": "highest mean reward on the validation prompts, greedy, thinking off; ties → earlier step",
              "n_valid": len(rows), "selected": selected, "checkpoints": summary,
              "adapter_sha256": {k: __import__("hashlib").sha256(v.read_bytes()).hexdigest()
                                 for k, v in checkpoints(run).items() if v is not None}}
    (out / "select.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: (v["mean_total"], v["n"]) for k, v in summary.items()}), "→ selected:", selected)


if __name__ == "__main__":
    main()
