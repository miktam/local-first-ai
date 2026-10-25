#!/usr/bin/env python3
"""exp_055: run one arm over every message, offline.

  P  Exposure Review's message check as built: exposure.py triage --offline (Parry's rules, the seeded registry
     facts, and its one local model step, gemma4:26b) -> band, and the join's hooks (footprint items)
  R  the same with --no-llm: Parry's rules and the registry facts only
  G  gemma4:26b alone: kit/prompt55.md with the footprint, the registry facts and the message, temperature 0,
     seed 55, Ollama's default thinking -> {"band", "used_items", "why"}

Nothing leaves the machine: P and R run --offline against a copy of the seeded cache (data/cache), and G calls
the local Ollama. Messages are named by opaque id (X001…), so no arm sees the kind.

    python3.14 kit/arms55.py --arm P --out runs/<ts>/P
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
WS = Path.home() / ".openclaw/workspace"
EXPOSURE = WS / "tools/exposure-report/exposure.py"
DEMO = WS / "tasks/exposure-report/demo"
DATA = HERE / "data"
BANDS = ("benign", "suspicious", "likely-scam")


def registry_facts(m: dict) -> str:
    return "\n".join(f"- {d}: created {c}" for d, c in sorted(m["domains"].items())) or "- (no domain in this message)"


def message_text(m: dict) -> str:
    return (DATA / "messages" / m["file"]).read_text(encoding="utf-8")


def to_label(m: dict) -> str:
    return {"C-526e13": "iris@vennholt-partners.example", "C-04c986": "hello@kestrelmoor.example",
            "C-5415d0": "her phone, +447700900142"}[m["to_item"]]


def prompt_for(m: dict) -> str:
    footprint = "\n".join((DATA / "FOOTPRINT.txt").read_text().splitlines()[1:])
    return (HERE / "kit/prompt55.md").read_text().format(footprint=footprint, registry=registry_facts(m),
                                                        channel=m["channel"], to=to_label(m), message=message_text(m))


def parse_json(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.M)
    s, e = text.find("{"), text.rfind("}")
    try:
        d = json.loads(text[s:e + 1])
    except Exception:
        return None
    return d if isinstance(d, dict) and d.get("band") in BANDS else None


def run_parry(mid, m, subject, cache, llm):
    argv = ["python3.14", str(EXPOSURE), "triage", str(subject), str(DATA / "messages" / m["file"]), "--offline",
            "--json"] + ([] if llm else ["--no-llm"])
    if m["channel"] != "email":
        argv += ["--channel", m["channel"]]
    t0 = time.time()
    p = subprocess.run(argv, capture_output=True, text=True, env={**os.environ, "TRIAGE_CACHE": str(cache)},
                       timeout=600)
    try:
        d = json.loads(p.stdout)
        band = d["verdict"]["band"]
        hooks = [{"item_id": h["item_id"], "level": h["level"]} for h in d["join"]["hooks"]]
        reach = d["join"]["reach"]
        extra = {"score": d["verdict"].get("score"), "persona_points": d["verdict"].get("persona_points"),
                 "complete": d["join"].get("complete")}
    except Exception as e:
        band, hooks, reach, extra = None, [], None, {"error": f"{type(e).__name__}: {e}", "stderr": p.stderr[-500:]}
    return {"id": mid, "band": band, "hooks": hooks, "reach": reach, "rc": p.returncode,
            "seconds": round(time.time() - t0, 2), **extra}


def run_gemma(mid, m):
    t0 = time.time()
    body = {"model": "gemma4:26b", "messages": [{"role": "user", "content": prompt_for(m)}], "stream": False,
            "options": {"temperature": 0, "seed": 55}}
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        r = json.loads(urllib.request.urlopen(req, timeout=600).read())
        content = r["message"]["content"]
        d = parse_json(content)
    except Exception as e:
        content, d = f"{type(e).__name__}: {e}", None
    return {"id": mid, "band": d["band"] if d else None, "used_items": (d or {}).get("used_items", []),
            "why": (d or {}).get("why"), "raw": None if d else content[:500], "seconds": round(time.time() - t0, 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["P", "R", "G"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "MANIFEST.json").read_text())
    if a.arm in ("P", "R"):
        subject, cache = out / "subject", out / "cache"
        if not subject.exists():
            shutil.copytree(DEMO, subject, ignore=shutil.ignore_patterns("out", ".lock"))
            subprocess.run(["python3.14", str(EXPOSURE), "build", str(subject)], capture_output=True, check=True)
        if not cache.exists():
            shutil.copytree(DATA / "cache", cache)
    done = set()
    res_path = out / "results.jsonl"
    if res_path.exists():
        done = {json.loads(l)["id"] for l in open(res_path)}
    with open(res_path, "a") as f:
        for mid in sorted(manifest):
            if mid in done:
                continue
            m = manifest[mid]
            r = run_parry(mid, m, subject, cache, a.arm == "P") if a.arm in ("P", "R") else run_gemma(mid, m)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            print(a.arm, mid, r["band"], r["seconds"], flush=True)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
