"""Shared readers for the June runtime experiments (exp_007, exp_008, exp_010, exp_011): repeat-1 (cold) values only."""
import glob
import json
from pathlib import Path

C = "tasks/chronos"


def _key(sk):
    return int(sk.rstrip("k")) if sk.rstrip("k").isdigit() else sk


def prefill_curve(repo: Path, phase_a: str, phase_b: str):
    """[(prompt tokens, rep-1 prefill ms/tok)]: Phase A rep 1 from its summary (or rep-1 size_*.json records), Phase B
    cold values from its summary; where both phases measured a size, Phase B is used."""
    pts = {}
    a = repo / C / phase_a
    s = json.load(open(a / "summary.json")) if (a / "summary.json").exists() else None
    if s and "rep1_prefill_ms_per_tok" in s["sizes"][0]:
        for z in s["sizes"]:
            pts[z["size_key"]] = (z["prompt_tokens"], z["rep1_prefill_ms_per_tok"])
    else:                                            # exp_007: the summary mean includes cache-hit reps; read repeat 1
        for f in sorted(glob.glob(str(a / "size_*.json"))):
            r = next(x for x in json.load(open(f)) if x["repeat"] == 1)
            pts[r["size_key"]] = (r["metrics"]["prompt_tokens"], r["metrics"]["prefill_ms_per_tok"])
    b = json.load(open(repo / C / phase_b / "summary.json"))
    for z in b["sizes"]:
        pts[z["size_key"]] = (z["prompt_tokens"], z.get("cold_prefill_ms_per_tok", z.get("rep1_prefill_ms_per_tok")))
    return sorted(pts.values()), b.get("cliff_threshold_ms_per_tok")


def prefill_seconds(repo: Path, *dirs: str):
    """[(prompt tokens, prefill seconds)] from the repeat-1 record of each size, Ollama's prompt_eval_duration."""
    pts = {}
    for d in dirs:
        for f in sorted(glob.glob(str(repo / C / d / "size_*.json"))):
            recs = json.load(open(f))
            recs = recs if isinstance(recs, list) else [recs]
            r = next((x for x in recs if x.get("repeat") == 1), recs[0])
            pts[r["size_key"]] = (r["metrics"]["prompt_tokens"], r["ollama_raw"]["prompt_eval_duration"] / 1e9)
    return sorted(pts.values())
