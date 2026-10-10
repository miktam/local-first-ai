"""R1: the vendored A12 glue reproduces exp_035's A12 v1 run byte for byte.

Runs exp_035's glue exactly as `run_eval.py run --mode glue` did on 2026-09-22, with
nothing changed except the Ollama base URL (a module attribute, set here). The server must be
the 0.33.0 copy that exp_035 ran on, started by harness/run_r1.sh. The vendored code is not edited.

Pass: all 60 v1 rows equal the reference on answer, chunk_ids and context. Every other
recorded field is compared and reported; wall time is not compared.
Stdlib only, so it runs under both interpreters (casasol .venv 3.14 and venv312 3.12).
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import platform
import sys
import time
import urllib.request
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
VENDOR = P38 / "vendor" / "coapi_voice"
REF = P38 / "reference" / "runs" / "glue_Qwen3-4B-GGUF-Q4-K-M_A12_20260922-2118"
MODEL = "hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M"
MODEL_DIGEST = "66f6d52aa56a"
EXPECT_VERSION = "0.33.0"
MUST_MATCH = ("answer", "chunk_ids", "context")
ALSO_COMPARED = ("scope", "advice", "citations_dropped", "flagged_claims", "repaired", "generated",
                 "prompt_tokens", "output_tokens", "context_tokens_est", "module", "lang", "type", "q", "gold", "expect")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        body = r.read()
    try:
        return json.loads(body)
    except ValueError:
        sys.exit(f"REFUSED: {url} did not answer with JSON ({len(body)} bytes); is the side server the one listening?")


def check_vendor() -> None:
    m = json.loads((P38 / "VENDOR_MANIFEST.json").read_text(encoding="utf-8"))
    for rel, e in m["files"].items():
        got = sha(P38 / rel)
        if got != e["sha256"]:
            sys.exit(f"REFUSED: {rel} sha256 {got[:12]} != manifest {e['sha256'][:12]}")
    if (VENDOR.parent / "casasol.ai").exists():
        sys.exit("REFUSED: a guide source exists under vendor/; load_index must never fall back to parse_guide")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18435")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)

    check_vendor()
    ver = get(f"{a.base_url}/api/version").get("version")
    if ver != EXPECT_VERSION:
        sys.exit(f"REFUSED: server at {a.base_url} is Ollama {ver}, R1 needs {EXPECT_VERSION}")
    tags = {t["name"]: t["digest"][:12] for t in get(f"{a.base_url}/api/tags")["models"]}
    if tags.get(MODEL) != MODEL_DIGEST:
        sys.exit(f"REFUSED: {MODEL} digest {tags.get(MODEL)} != {MODEL_DIGEST}")

    sys.path.insert(0, str(VENDOR))
    import guide_bm25 as g  # noqa: E402
    import run_eval as ev  # noqa: E402
    import glue as G  # noqa: E402
    ev.OLLAMA = a.base_url  # the only difference from exp_035's run

    ref_cfg = json.loads((REF / "config.json").read_text(encoding="utf-8"))
    if ev.OPTIONS != ref_cfg["options"]:
        sys.exit("REFUSED: vendored OPTIONS differ from the reference run's config")
    rows = [json.loads(l) for l in (VENDOR / "data/eval/coapi_eval_v1.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ref = {r["id"]: r for r in (json.loads(l) for l in (REF / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip())}
    idx = {lang: g.load_index(lang) for lang in g.LANGS}

    a.out.mkdir(parents=True, exist_ok=False)
    meta = {"check": "R1", "python": platform.python_version(), "executable": sys.executable,
            "ollama_version": ver, "base_url": a.base_url, "model": MODEL, "model_digest": MODEL_DIGEST,
            "started_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    fh = (a.out / "outputs.jsonl").open("w", encoding="utf-8")
    diffs = []
    for i, r in enumerate(rows, 1):
        t0 = time.time()
        res = G.answer(r["q"], model=MODEL, index=idx, verify=False)
        rec = {"id": r["id"], "module": r["module"], "lang": r["lang"], "type": r["type"],
               "expect": r["expect"], "q": r["q"], "gold": r["gold"],
               "chunk_ids": res["chunk_ids"], "context": res["context"],
               "context_tokens_est": res.get("context_tokens_est", 0), "answer": res["answer"],
               "scope": res["scope"], "advice": res.get("advice", False), "citations_dropped": res.get("citations_dropped", []),
               "flagged_claims": res.get("flagged_claims", []), "repaired": res.get("repaired", False), "generated": res["generated"],
               "prompt_tokens": res.get("prompt_tokens"), "output_tokens": res.get("output_tokens"),
               "wall_s": round(time.time() - t0, 2)}
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
        rr = ref.get(r["id"])
        bad = [f for f in MUST_MATCH + ALSO_COMPARED if rr is None or rr.get(f) != rec.get(f)]
        if bad:
            diffs.append({"id": r["id"], "fields": bad, "must_match_failed": [f for f in bad if f in MUST_MATCH]})
        print(f"{i:3}/{len(rows)} {r['id']:6} {r['lang']} {rec['scope']:3} {rec['wall_s']:5.1f}s {'ok' if not bad else 'DIFF ' + ','.join(bad)}", file=sys.stderr)
    fh.close()
    hard = [d for d in diffs if d["must_match_failed"]]
    meta.update({"ended_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "n_rows": len(rows),
                 "n_rows_any_diff": len(diffs), "n_rows_must_match_failed": len(hard), "diffs": diffs,
                 "outputs_sha256": sha(a.out / "outputs.jsonl"), "verdict": "PASS" if not hard and len(rows) == 60 else "FAIL"})
    (a.out / "r1_result.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("python", "ollama_version", "n_rows", "n_rows_any_diff", "n_rows_must_match_failed", "verdict")}))
    return 0 if meta["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
