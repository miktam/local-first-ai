#!/usr/bin/env python3
"""exp_062: score arm I with exp_055's metric functions, imported unchanged; C1-C4, H1-H3, the paired tables.

    python3.14 kit/score62.py runs/<ts>          writes runs/<ts>/score.json and prints a summary
"""
import json
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
EXP55 = HERE.parent / "exp_055_exposure_review_lures"
EXP60 = HERE.parent / "exp_060_reader_empty_answers"
sys.path.insert(0, str(EXP55 / "kit"))
import score55  # noqa: E402

G_RESULTS = EXP55 / "runs/20261025T095614Z/G/results.jsonl"
F_RESULTS = EXP60 / "runs/20261101T152955Z/F/results.jsonl"
PORT = 18450
F_BA, G_BA, OPUS_BA = 0.914, 0.882, 0.950
BARS = {"answered": 218, "balanced_accuracy": F_BA, "false_alarms": 11, "attribution_rate": 0.85}


def rows(path: Path) -> dict:
    """id -> last row (a resumed run may repeat an id)."""
    out = {}
    if path.exists():
        for l in open(path, encoding="utf-8"):
            if l.strip():
                r = json.loads(l)
                out[r["id"]] = r
    return out


def correct(m: dict, band) -> bool:
    flagged = band in score55.FLAGGED
    return flagged if m["kind"] == "lure" else not flagged


def paired(a: dict, b: dict, man: dict) -> dict:
    """Messages by who is right: both, neither, only the first arm, only the second."""
    t = {"both": 0, "neither": 0, "only_first": 0, "only_second": 0}
    for i, m in man.items():
        x, y = correct(m, a.get(i, {}).get("band")), correct(m, b.get(i, {}).get("band"))
        t["both" if x and y else "neither" if not x and not y else "only_first" if x else "only_second"] += 1
    t["p_first_at_least_this_good"] = round(score55.mcnemar_one_sided(t["only_first"], t["only_second"]), 4)
    return t


def audit_check(path: Path) -> dict:
    """C2: every connect of the runner goes to 127.0.0.1:18450; its only subprocess is notifyutil."""
    if not path.exists():
        return {"present": False, "pass": False, "note": "no audit log"}
    connects, resolves, spawns, bad = 0, 0, 0, []
    for l in open(path, encoding="utf-8"):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if r.get("kind") == "getaddrinfo":
            resolves += 1
            if str(r.get("host")) not in ("127.0.0.1", "localhost"):
                bad.append(f"resolve {r.get('host')}")
        elif r.get("kind") == "audit":
            ev, args = r.get("event"), [str(a) for a in (r.get("args") or [])]
            if ev == "socket.connect":
                connects += 1
                addr = args[1] if len(args) > 1 else ""
                if not re.search(rf"'127\.0\.0\.1',\s*{PORT}\)", addr):
                    bad.append(f"connect {addr[:120]}")
            elif ev in ("subprocess.Popen", "os.posix_spawn", "os.exec", "os.spawn", "os.system"):
                spawns += 1
                if "notifyutil" not in " ".join(args):
                    bad.append(f"{ev} {' '.join(args)[:120]}")
    return {"present": True, "connects": connects, "resolutions": resolves, "spawns": spawns, "violations": bad[:20],
            "pass": connects > 0 and not bad}


def server_check(path: Path) -> dict:
    """C3, second half: one slot with a context of 8,192."""
    text = path.read_text(errors="replace") if path.exists() else ""
    m = re.search(r"n_slots = (\d+), n_ctx_slot = (\d+)", text)
    slots, ctx = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    return {"n_slots": slots, "n_ctx_slot": ctx, "pass": slots == 1 and ctx == 8192}


def quantile(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def main():
    run = Path(sys.argv[1])
    man = json.loads((EXP55 / "data/MANIFEST.json").read_text())
    res, G, F = rows(run / "I" / "results.jsonl"), rows(G_RESULTS), rows(F_RESULTS)
    meta = json.loads((run / "meta.json").read_text()) if (run / "meta.json").exists() else {}
    m = score55.arm_metrics(res, man)
    answered = sum(1 for i in man if res.get(i, {}).get("band") is not None)
    vs_f, vs_g = paired(res, F, man), paired(res, G, man)
    secs = [r["seconds"] for r in res.values() if r.get("seconds") is not None]
    toks = [r["completion_tokens"] for r in res.values() if r.get("completion_tokens")]
    thermals = sorted({str(r.get("thermal_before")) for r in res.values()})
    empties = [{"id": i, "finish_reason": res.get(i, {}).get("finish_reason"), "error": res.get(i, {}).get("error"),
                "prompt_tokens": res.get(i, {}).get("prompt_tokens"),
                "completion_tokens": res.get(i, {}).get("completion_tokens"), "kind": man[i]["kind"]}
               for i in sorted(man) if res.get(i, {}).get("band") is None]

    controls = {
        "C1": {"pass": bool(meta.get("pinned", {}).get("frozen_inputs")) and len(res) == len(man),
               "frozen_inputs": meta.get("frozen_inputs"), "messages_run": len(res),
               "note": "the prompt is asserted byte for byte by the runner on every corpus message"},
        "C2": audit_check(run / "I" / "audit.jsonl"),
        "C3": {**server_check(run / "I" / "server.log"),
               "model_file": bool(meta.get("pinned", {}).get("model_file")),
               "server_commit": bool(meta.get("pinned", {}).get("server_commit"))},
        "C4": {"thermal_levels_seen": thermals, "ollama_ps_before": meta.get("ollama_ps_before"),
               "ollama_ps_after": meta.get("ollama_ps_after"), "pass": thermals == ["0"]},
    }
    controls["C3"]["pass"] = controls["C3"]["pass"] and controls["C3"]["model_file"] and controls["C3"]["server_commit"]

    h1 = answered >= BARS["answered"]
    h2 = m["balanced_accuracy"] >= BARS["balanced_accuracy"] and vs_f["only_first"] >= vs_f["only_second"]
    h3 = m["false_alarms"] <= BARS["false_alarms"] and (m["attribution_rate"] or 0) >= BARS["attribution_rate"]
    void = not (controls["C1"]["pass"] and controls["C2"]["pass"] and controls["C3"]["pass"])
    if void:
        decision = "void: a control failed (C1, C2 or C3)"
    elif m["balanced_accuracy"] < G_BA:
        decision = "the line stops: balanced accuracy below G's 0.882"
    elif h1 and h2 and h3:
        decision = "passes here; the seat changes only if this repeats on a tagged llama.cpp release"
    elif h2 and h3:
        decision = "recorded; gemma4:26b with F keeps the seat; next step registered: a retry in glue"
    else:
        decision = "recorded; gemma4:26b with F keeps the seat"

    public = {k: v for k, v in m.items() if not k.startswith("_")}
    score = {
        "run": str(run), "messages": len(man), "answered": answered, "metrics": public, "controls": controls,
        "hypotheses": {"H1": h1, "H2": h2, "H3": h3}, "bars": BARS, "decision": decision,
        "paired_vs_F": vs_f, "paired_vs_G": vs_g, "gap_to_frontier": round(OPUS_BA - m["balanced_accuracy"], 3),
        "time": {"median_s": round(statistics.median(secs), 1) if secs else None, "p95_s": quantile(secs, 0.95),
                 "total_hours": round(sum(secs) / 3600, 2) if secs else None,
                 "median_completion_tokens": statistics.median(toks) if toks else None,
                 "void_by_C4": not controls["C4"]["pass"]},
        "empties": empties,
        "breakdowns": {
            "by_channel_and_kind": score55.breakdown(res, man, lambda x: f"{x['channel']}/{x['kind']}"),
            "lure_by_ask": score55.breakdown(res, man, lambda x: x["ask"] if x["kind"] == "lure" else None)},
    }
    (run / "score.json").write_text(json.dumps(score, indent=1, default=list) + "\n")
    print(f"answered {answered}/{len(man)} | caught {m['caught']}/{m['lures']} | genuine flagged {m['false_alarms']}/{m['benign']}"
          f" | balanced accuracy {m['balanced_accuracy']} | attribution {m['attribution_rate']}")
    print(f"paired vs F: only I right {vs_f['only_first']}, only F right {vs_f['only_second']}"
          f" | vs G: only I {vs_g['only_first']}, only G {vs_g['only_second']}")
    print("controls:", {k: v["pass"] for k, v in controls.items()}, "| H1 H2 H3:", h1, h2, h3)
    print("time:", score["time"])
    print("decision:", decision)


if __name__ == "__main__":
    main()
