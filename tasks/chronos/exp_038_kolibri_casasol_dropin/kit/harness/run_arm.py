"""exp_038 W5: run one arm over the sets, through the vendored A12 glue, unchanged.

Arms (DESIGN rev 2 §2, §0b):
  A12      hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M on the Ollama 0.33.0 side server (:18435), as R1
  G26      gemma4:26b on the production Ollama (:11434, 0.40.2, llamacpp runner)
  K4       Kolibri 4-bit through kit/mlx_backend.py, effort none            (mbp only; binds to exp_037's gate)
  K4-MED   Kolibri 4-bit, answer call at effort medium, --reasoning-cap N  (mbp only)
  K8       Kolibri 8-bit, effort none, reference arm                         (mbp only)

Sets: probes, v1, de (German slice), twins (its English twins), dev (DE dev set; never scored). Run order per
arm: probes, v1, de, twins (§4 "Run order"). v2 is NOT RUN (§0b.13).

The harness sets the glue's model call (`run_eval.ollama_chat`) to a recorder that adds one automatic retry on a
transport error or timeout and keeps every call's raw text, tokens and wall time. German rows go through
harness/de_path.py (the additive, language-keyed A12-DE path). Records are appended to <out>/<set>.jsonl one per
row and fsync'ed; a re-run skips ids already recorded (resume). An exception in exp_038's own code is recorded on
the row (`error`) and never counted as a model result.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import socket
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
VENDOR = P38 / "vendor" / "coapi_voice"
SETS = {
    "v1": VENDOR / "data/eval/coapi_eval_v1.jsonl",
    "probes": P38 / "private/sets/probes.jsonl",
    "de": P38 / "private/sets/de_slice.jsonl",
    "twins": P38 / "private/sets/de_twins.jsonl",
    "dev": P38 / "private/sets/de_dev.jsonl",
    "step0": VENDOR / "data/step0_questions.jsonl",
}
OLLAMA_ARMS = {
    "A12": {"base_url": "http://127.0.0.1:18435", "model": "hf.co/unsloth/Qwen3-4B-GGUF:Q4_K_M", "version": "0.33.0"},
    "G26": {"base_url": "http://127.0.0.1:11434", "model": "gemma4:26b", "version": "0.40.2"},
}
MLX_ARMS = {"K4": ("K4", "none"), "K4-MED": ("K4", "medium"), "K8": ("K8", "none")}
TIMEOUT_S = 600
RENDERED_ROWS = 3  # the full rendered prompt is kept for the first rows of each arm and set


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


class Recorder:
    """Stands in for run_eval.ollama_chat: one retry on transport errors, every call kept."""

    def __init__(self, inner):
        self.inner = inner
        self.calls: list[dict] = []
        self.keep_messages = False

    def __call__(self, model, messages, options):
        kind = "gate" if int(options.get("num_predict", 0)) == 4 else "answer"
        last = None
        for attempt in (1, 2):
            t0 = time.time()
            try:
                resp = self.inner(model, messages, options)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
                last = e
                self.calls.append({"call": kind, "attempt": attempt, "error": f"{type(e).__name__}: {e}",
                                   "wall_s": round(time.time() - t0, 2)})
                continue
            rec = {"call": kind, "attempt": attempt, "wall_s": round(time.time() - t0, 2),
                   "raw": resp["message"]["content"], "prompt_tokens": resp.get("prompt_eval_count"),
                   "output_tokens": resp.get("eval_count"), "done_reason": resp.get("done_reason")}
            if "exp038" in resp:
                rec["mlx"] = resp["exp038"]
            if self.keep_messages:
                rec["messages_sha256"] = sha(json.dumps(messages, ensure_ascii=False, sort_keys=True))
                rec["messages"] = messages
            self.calls.append(rec)
            if rec["wall_s"] > TIMEOUT_S:
                rec["timeout"] = True
            return resp
        raise RuntimeError(f"failed call: both attempts failed ({last})")


def load_rows(name: str) -> list[dict]:
    p = SETS[name]
    if not p.is_file():
        sys.exit(f"REFUSED: set {name} missing at {p}")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out = set()
    for l in path.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            if not r.get("error"):
                out.add(r["id"])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(OLLAMA_ARMS) + sorted(MLX_ARMS))
    ap.add_argument("--sets", default="probes,v1,de,twins")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reasoning-cap", type=int, default=None)
    ap.add_argument("--ids", default=None, help="comma-separated row ids only (R6 re-runs)")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    sys.path.insert(0, str(VENDOR))
    import guide_bm25 as g  # noqa: E402
    import run_eval as ev  # noqa: E402
    import glue as G  # noqa: E402
    sys.path.insert(0, str(P38 / "harness"))

    header = {"arm": a.arm, "started_utc": utc(), "python": platform.python_version(), "host": socket.gethostname().split(".")[0],
              "vendor_manifest_sha256": hashlib.sha256((P38 / "VENDOR_MANIFEST.json").read_bytes()).hexdigest(), "tag": a.tag}
    if a.arm in OLLAMA_ARMS:
        cfg = OLLAMA_ARMS[a.arm]
        ver = get_json(f"{cfg['base_url']}/api/version").get("version")
        if ver != cfg["version"]:
            sys.exit(f"REFUSED: {a.arm} needs Ollama {cfg['version']} at {cfg['base_url']}, found {ver}")
        ev.OLLAMA = cfg["base_url"]
        model = cfg["model"]
        rec = Recorder(ev.ollama_chat)
        header.update(base_url=cfg["base_url"], ollama_version=ver, model=model,
                      model_digest={t["name"]: t["digest"] for t in get_json(f"{cfg['base_url']}/api/tags")["models"]}.get(model))
    else:
        sys.path.insert(0, str(P38 / "kit"))
        import mlx_backend as mb  # noqa: E402
        arm, effort = MLX_ARMS[a.arm]
        backend = mb.MLXGatedBackend(arm, effort_answer=effort, reasoning_cap=a.reasoning_cap)
        model = a.arm
        rec = Recorder(backend.chat)
        header.update(binding=backend.binding, effort_answer=effort, reasoning_cap=a.reasoning_cap)
    ev.ollama_chat = rec  # the glue's only model call, now recorded (glue.py calls ev.ollama_chat)

    idx = {lang: g.load_index(lang) for lang in g.LANGS}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"header_{header['started_utc'].replace(':', '')}.json").write_text(json.dumps(header, indent=1) + "\n")
    only = set(a.ids.split(",")) if a.ids else None

    for set_name in [s for s in a.sets.split(",") if s]:
        rows = load_rows(set_name)
        path = a.out / f"{set_name}.jsonl"
        done = done_ids(path)
        n_kept = 0
        with open(path, "a", encoding="utf-8") as fh:
            for r in rows:
                if r["id"] in done or (only is not None and r["id"] not in only):
                    continue
                rec.calls = []
                rec.keep_messages = n_kept < RENDERED_ROWS
                t0 = time.time()
                out = {"id": r["id"], "set": set_name, "arm": a.arm, "lang": r["lang"], "type": r.get("type"),
                       "module": r.get("module"), "expect": r.get("expect"), "q": r["q"], "route_q": r.get("route_q"),
                       "gold": r.get("gold"), "t_start": utc()}
                try:
                    if r["lang"] == "de":
                        import de_path  # noqa: E402  (harness/; written in W3)
                        res = de_path.answer_de(G, r["q"], r["route_q"], model=model, index=idx)
                    else:
                        res = G.answer(r["q"], model=model, index=idx, verify=False)
                    out.update({k: res.get(k) for k in ("scope", "scope_raw", "advice", "citations_dropped", "flagged_claims",
                                                        "repaired", "chunk_ids", "context", "context_tokens_est", "answer",
                                                        "generated", "prompt_tokens", "output_tokens")})
                    out["context_sha256"] = sha(res.get("context") or "")
                    out["error"] = None
                except Exception as e:  # an adapter or harness defect, never a model result
                    out["error"] = f"{type(e).__name__}: {e}"
                    out["traceback"] = traceback.format_exc(limit=6)
                out["calls"] = rec.calls
                out["failed_call"] = any(c.get("error") for c in rec.calls) and not any("raw" in c for c in rec.calls[-1:])
                out["wall_s"] = round(time.time() - t0, 2)
                fh.write(json.dumps(out, ensure_ascii=False) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
                n_kept += 1
                print(f"{a.arm} {set_name} {r['id']:10} {r['lang']} {out.get('scope') or '-':3} {out['wall_s']:6.1f}s"
                      f"{'  ERROR ' + out['error'] if out['error'] else ''}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
