#!/usr/bin/env python3
"""
exp_039 — run one arm on one set through the Ollama 0.33.0 side server (HYPOTHESIS.md, amendment 1).
Started by rl/measure.sh, which owns the side server; never edit a pinned file to make this pass.

  python rl/run_arm.py --host URL repro            # A12 on v1 rows 1–3, archived 22 Sep index: byte-identical
  python rl/run_arm.py --host URL ARM SET          # ARM: B0r R1 A12 R1G · SET: v1 v21

Arms
  B0r, R1   "prompt" mode: exactly what R1 was trained on and nothing else — glue's
            language-rendered system prompt (+ the advice suffix when glue's _ADVICE fires), the
            retrieved material inside the user turn (k 5, budget 1200, ES retrieves from EN),
            run_eval.OPTIONS, thinking off. No scope gate, no citation stripping, no hand-off, no scrub.
  A12, R1G  "glue" mode: glue.answer() unchanged (b1a648c), with the arm's model.

Pins checked before any row: glue / run_eval / guide_bm25 identical to b1a648c; index files present
with their hashes (live = post-audit f1e28120 / 9a422b0e; repro = archived a1793b82 / ce512559 — a
missing index would silently be rebuilt from the guide source, so absence is a refusal); server
version 0.33.0; the arm's model digest; the set's sha256. A v2.1 cell runs once: a second run of the
same arm on v2.1 is refused. Output: runs/exp039_<ARM>_<SET>_<stamp>/ in run_eval's record format,
so `run_eval.py score` and the judge read it unchanged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV = HERE.parent
sys.path.insert(0, str(CV))
import glue as G          # noqa: E402
import guide_bm25 as g    # noqa: E402
import run_eval as ev     # noqa: E402

BASE_MODEL = "hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M"     # Unsloth's quantisation: A12, the reference (ruler continuity)
BASE_OURS = "qwen3-4b-b11577:q4km"                     # the untrained snapshot through R1's exact pipeline (amendment 2)
# Arms R1 / R1G carry run 2's checkpoint 100 (amendment 4): adapter 0c57f0e7…, mlx-lm 0.32 fuse, b11577 f16 → Q4_K_M
# (GGUF 86fcc330…). Run 1's model (coapi-voice:r1, 08b5643e92c3) stopped at H5 and is never scored.
R1_MODEL = "coapi-voice:r2"
R1_Q8 = "coapi-voice:r2-q8"                            # same checkpoint at Q8_0 (GGUF 63f1e254…), D1 and amendment 5
BASE_OURS_Q8 = "qwen3-4b-b11577:q8"                    # untrained snapshot, same pipeline, Q8_0 (GGUF 391074283823…), amendment 5
DIGEST = {BASE_MODEL: "66f6d52aa56a", BASE_OURS: "8fa738d6b7b9", R1_MODEL: "0d974e5fc4a3", R1_Q8: "8478de4545b7",
          BASE_OURS_Q8: "3e8eb60ca5fa"}
ARMS = {"B0r": ("prompt", BASE_OURS), "R1": ("prompt", R1_MODEL), "A12": ("glue", BASE_MODEL), "R1G": ("glue", R1_MODEL),
        "R1q8": ("prompt", R1_Q8), "B0rq8": ("prompt", BASE_OURS_Q8)}
SETS = {"v1": (CV / "data/eval/coapi_eval_v1.jsonl", "efc16ab2"), "v21": (CV / "data/eval/coapi_eval_v2_1.jsonl", "b40a3c79"),
        "d1": (CV / "data/eval/coapi_eval_v1.jsonl", "efc16ab2")}
D1_ROWS = ["M1-01", "M1-06", "M2-04", "M2-05", "M3-03", "M3-06", "M4-01", "M4-06", "M5-02", "M6-03"]  # E35 H5, fixed 2026-09-21
INDEX_LIVE = {"en": "f1e28120", "pl": "9a422b0e"}
INDEX_ARCHIVED = {"en": "a1793b82", "pl": "ce512559"}
CODE_PIN = "b1a648c"
REF_A12 = CV / "runs/glue_Qwen3-4B-GGUF-Q4-K-M_A12_20260922-2118"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def use_index(pins: dict[str, str]) -> None:
    """Point guide_bm25 at a directory holding exactly the pinned index files; refuse otherwise."""
    if pins == INDEX_ARCHIVED:
        d = Path(tempfile.mkdtemp(prefix="exp039-index-"))
        for lang, h in pins.items():
            (d / f"guide_{lang}.json").symlink_to(CV / "index/archive" / f"guide_{lang}.{h}.json")
    else:
        d = CV / "index"
    for lang, h in pins.items():
        p = d / f"guide_{lang}.json"
        if not p.exists() or not sha(p).startswith(h):
            sys.exit(f"REFUSED: index {p} missing or not {h}")
    g.INDEX_DIR = d


def get(host: str, path: str) -> dict:
    with urllib.request.urlopen(f"{host}{path}", timeout=10) as r:
        return json.load(r)


def check_server(host: str, model: str) -> str:
    ver = get(host, "/api/version")["version"]
    if ver != "0.33.0":
        sys.exit(f"REFUSED: server is {ver}, not 0.33.0")
    tags = {m["name"]: m["digest"] for m in get(host, "/api/tags")["models"]}
    dig = tags.get(model) or tags.get(model + ":latest")
    if not dig:
        sys.exit(f"REFUSED: {model} not registered on {host}")
    if DIGEST.get(model) and not dig.startswith(DIGEST[model]):
        sys.exit(f"REFUSED: {model} digest {dig[:12]} is not {DIGEST[model]}")
    return dig


def check_code() -> None:
    r = subprocess.run(["git", "-C", str(CV.parent), "diff", "--quiet", CODE_PIN, "--",
                        "coapi_voice/glue.py", "coapi_voice/run_eval.py", "coapi_voice/guide_bm25.py"])
    if r.returncode != 0:
        sys.exit(f"REFUSED: glue / run_eval / guide_bm25 differ from {CODE_PIN}")


def prompt_answer(q: str, model: str, idx: dict) -> dict:
    lang = G.detect_lang(q)                                   # as at serve
    chunks = idx["en" if lang == "es" else lang].retrieve(q, k=G.K, budget=G.BUDGET)
    context = g.context_block(chunks)
    system = G.SYSTEM[lang] + (G.ADVICE_SUFFIX[lang] if G._ADVICE.search(q) else "")
    user = f"{G.CONTEXT_LABEL[lang]}:\n{context}\n\n{G.QUESTION_LABEL[lang]}:\n{q}"
    resp = ev.ollama_chat(model, [{"role": "system", "content": system}, {"role": "user", "content": user}], ev.OPTIONS)
    return {"lang": lang, "scope": "in", "chunk_ids": [c["id"] for c in chunks], "context": context,
            "context_tokens_est": sum(c["tokens_est"] for c in chunks), "answer": resp["message"]["content"],
            "generated": True, "advice": bool(G._ADVICE.search(q)), "system_prompt": system,
            "prompt_tokens": resp.get("prompt_eval_count"), "output_tokens": resp.get("eval_count")}


def run(arm: str, rows: list[dict], host: str, out: Path) -> Path:
    mode, model = ARMS[arm]
    idx = {lang: g.load_index(lang) for lang in g.LANGS}
    out.mkdir(parents=True)
    with (out / "outputs.jsonl").open("w", encoding="utf-8") as fh:
        for i, r in enumerate(rows, 1):
            t0 = time.time()
            res = prompt_answer(r["q"], model, idx) if mode == "prompt" else G.answer(r["q"], model=model, index=idx)
            rec = {"id": r["id"], "module": r["module"], "lang": r["lang"], "type": r["type"], "expect": r["expect"],
                   "q": r["q"], "gold": r["gold"], "chunk_ids": res["chunk_ids"], "context": res["context"],
                   "context_tokens_est": res.get("context_tokens_est", 0), "answer": res["answer"],
                   "scope": res["scope"], "advice": res.get("advice", False),
                   "citations_dropped": res.get("citations_dropped", []), "generated": res["generated"],
                   "prompt_tokens": res.get("prompt_tokens"), "output_tokens": res.get("output_tokens"),
                   "wall_s": round(time.time() - t0, 2)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            print(f"{arm} {i:3}/{len(rows)} {r['id']:9} {r['lang']} {res['scope']:3} {rec['wall_s']:5.1f}s", file=sys.stderr)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("arm", choices=["repro", *ARMS])
    ap.add_argument("set", nargs="?", choices=list(SETS))
    a = ap.parse_args(argv)
    ev.OLLAMA = a.host
    check_code()
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    if a.arm == "repro":
        use_index(INDEX_ARCHIVED)
        check_server(a.host, BASE_MODEL)
        rows = [json.loads(l) for l in SETS["v1"][0].read_text(encoding="utf-8").splitlines() if l.strip()][:3]
        out = run("A12", rows, a.host, CV / f"runs/exp039_repro_{stamp}")
        ref = [json.loads(l) for l in (REF_A12 / "outputs.jsonl").read_text(encoding="utf-8").splitlines()][:3]
        new = [json.loads(l) for l in (out / "outputs.jsonl").read_text(encoding="utf-8").splitlines()]
        bad = [f"{x['id']}: {k}" for x, y in zip(ref, new) for k in ("id", "chunk_ids", "context", "answer", "output_tokens")
               if x.get(k) != y.get(k)]
        print("REPRO PASS: 3/3 byte-identical to 2026-09-22 (archived index)" if not bad and len(new) == 3 else f"REPRO FAIL: {bad}")
        sys.exit(1 if bad or len(new) != 3 else 0)

    if not a.set:
        sys.exit("a set is required: v1 or v21")
    path, want = SETS[a.set]
    if not sha(path).startswith(want):
        sys.exit(f"REFUSED: {path.name} is not {want}")
    if a.set == "v21" and list((CV / "runs").glob(f"exp039_{a.arm}_v21_*")):
        sys.exit(f"REFUSED: {a.arm} has already run on v2.1; each cell is scored once")
    use_index(INDEX_LIVE)
    mode, model = ARMS[a.arm]
    digest = check_server(a.host, model)
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if a.set == "d1":
        rows = [r for r in rows if r["id"] in D1_ROWS]
        assert len(rows) == 10, len(rows)
    out = CV / f"runs/exp039_{a.arm}_{a.set}_{stamp}"
    run(a.arm, rows, a.host, out)
    (out / "config.json").write_text(json.dumps({
        "experiment": "exp_039", "arm": a.arm, "mode": mode, "model": model, "model_digest": digest, "set": a.set,
        "set_sha256": sha(path), "index": {l: sha(g.INDEX_DIR / f"guide_{l}.json") for l in g.LANGS},
        "options": ev.OPTIONS, "think": False, "code_pin": CODE_PIN, "server": "ollama 0.33.0",
        "finished": dt.datetime.now().isoformat(timespec="seconds")}, indent=2) + "\n", encoding="utf-8")
    ev.main(["score", str(out)])
    print(out)


if __name__ == "__main__":
    main()
