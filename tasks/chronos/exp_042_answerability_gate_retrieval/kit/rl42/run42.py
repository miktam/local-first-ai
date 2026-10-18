#!/usr/bin/env python3
"""
exp_042 — the runs, all from the stored rows of rl42/build42.py (started by rl42/run42.sh, which owns the servers).

  python rl42/run42.py voice --host URL           # S0: the untrained voice on v3 (0.33.0 side server, B0i digest)
  python rl42/run42.py gate  --host URL GATE SET  # GATE G2 | G4 on SET dev | v3 (llama.cpp b11711); v3 only after
                                                  # data/gate42/thresholds.json is committed
  python rl42/run42.py select                     # thresholds from dev (rule as exp_041), gates fixed: G4 primary
  python rl42/run42.py score [--json OUT]         # H1–H4 on v3, plus the source-lost report

S0 answers come from the stored system + user turn, run_eval.OPTIONS (greedy, 512 tokens), thinking off, and are
scored by reward40 (rev 3). The gates see the stored material and the question as shown. Resumable; one v3 pass per
gate and for the voice. Prints aggregates only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV = HERE.parent
sys.path[:0] = [str(CV), str(HERE), str(CV / "rl40"), str(CV / "rl41"), str(CV / "rl")]
import compare40 as C     # noqa: E402
import gate41 as GT       # noqa: E402
import reward40 as W      # noqa: E402
import select41 as SEL    # noqa: E402

D = CV / "data" / "gate42"
RUNS = CV / "runs"
GATES = ("G4", "G2")       # fixed in advance (DESIGN decision 3); G4 primary
VOICE_MODEL = "qwen3-4b-b11577:q4km-imat"
H1_RISE, H2_MAX, H3_FLOOR, H4_MAX = 0.20, 0.05, -0.02, 0.10


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(name: str) -> list[dict]:
    p = D / f"{name}.jsonl"
    if sha(p) != (D / f"{name}.sha256").read_text().strip():
        sys.exit(f"REFUSED: {p.name} does not match its sha256")
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def committed() -> bool:
    return not subprocess.run(["git", "-C", str(CV.parent), "status", "--porcelain", "--", "coapi_voice/rl42",
                               "coapi_voice/data/gate42"], capture_output=True, text=True).stdout.strip()


def resumable(out: Path, rows: list[dict], fn, label: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "rows.jsonl"
    done = {json.loads(l)["id"] for l in path.read_text().splitlines()} if path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    t0 = time.time()
    with path.open("a", encoding="utf-8") as f:
        for i, r in enumerate(todo, 1):
            f.write(json.dumps(fn(r), ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0 or i == len(todo):
                print(f"{label} {len(done) + i}/{len(rows)} · {time.time() - t0:.0f}s", flush=True)
    n = sum(1 for _ in path.open())
    if n >= len(rows):
        (out / "done.json").write_text(json.dumps({"label": label, "n": n, "rows_sha256": sha(path)}) + "\n")


def cmd_voice(a) -> None:
    import run_arm as ra
    import run_eval as ev
    if (RUNS / "exp042_S0_v3" / "done.json").exists():
        sys.exit("REFUSED: S0 already has its v3 pass")
    ev.OLLAMA = a.host
    ra.check_server(a.host, VOICE_MODEL)
    rows = load("v3_rendered")

    def one(r):
        resp = ev.ollama_chat(VOICE_MODEL, [{"role": "system", "content": r["system"]}, {"role": "user", "content": r["user"]}],
                              ev.OPTIONS)
        text = resp["message"]["content"]
        n = resp.get("eval_count")
        W.R.set_token_counter(lambda s, n=n: n if n is not None else int(len(s.split()) * 1.4))
        x = W.analyse(text, json.dumps(r["meta"], ensure_ascii=False, sort_keys=True))
        line = x["facts"]["final_line"]
        return {"id": r["id"], "verdict": x["facts"]["verdict"], "final_line": line,
                "abstained": bool(line) and W.ABSTAIN in W.candidate_hits(line, {}), "output_tokens": n, "completion": text}
    resumable(RUNS / "exp042_S0_v3", rows, one, "S0/v3")


def cmd_gate(a) -> None:
    if not sha(GT.GATES[a.gate]["gguf"]).startswith(GT.GATES[a.gate]["sha"]):
        sys.exit(f"REFUSED: {a.gate} model file does not match its pin")
    if a.set == "v3":
        if not (D / "thresholds.json").exists() or not committed():
            sys.exit("REFUSED: v3 runs only after thresholds.json is committed, from committed code")
        if (RUNS / f"exp042_v3_{a.gate}" / "done.json").exists():
            sys.exit(f"REFUSED: {a.gate} already has its v3 pass")
    rows = load("dev" if a.set == "dev" else "v3_rendered")

    def one(r):
        res = GT.ask(a.gate, a.host, r["material"], r["question"])
        return {"id": r["id"], "p": res["p"], "tokens": res["tokens"]}
    resumable(RUNS / f"exp042_{a.set}_{a.gate}", rows, one, f"{a.gate}/{a.set}")


def probs(set_: str, gate: str) -> dict:
    p = RUNS / f"exp042_{set_}_{gate}" / "rows.jsonl"
    return {json.loads(l)["id"]: json.loads(l)["p"] for l in p.read_text().splitlines() if l.strip()}


def cmd_select(a) -> None:
    rows = {r["id"]: r for r in load("dev")}
    out = {"rule": "abstain iff P(answerable) < tau; tau = largest value with dev false abstention <= 0.03; gates fixed, G4 primary",
           "dev_sha256": sha(D / "dev.jsonl"), "primary": "G4", "gates": {}}
    for gate in GATES:
        pr = probs("dev", gate)
        if set(pr) != set(rows):
            sys.exit(f"REFUSED: {gate} dev run incomplete ({len(pr)}/{len(rows)})")
        ans = [pr[i] for i, r in rows.items() if r["label"] == "answerable"]
        neg = [pr[i] for i, r in rows.items() if r["label"] == "not_answerable"]
        tau = SEL.threshold(ans)
        by = {f"src={s}": round(sum(pr[i] < tau for i, r in rows.items() if r["src"] == s) /
                                 sum(1 for r in rows.values() if r["src"] == s), 4) for s in sorted({r["src"] for r in rows.values()})}
        out["gates"][gate] = {"tau": tau, "abstain_dev": round(sum(p < tau for p in neg) / len(neg), 4),
                              "false_abstention_dev": round(sum(p < tau for p in ans) / len(ans), 4),
                              "auroc": round(SEL.auroc(ans, neg), 4),
                              "ece": round(SEL.ece([(pr[i], int(r["label"] == "answerable")) for i, r in rows.items()]), 4),
                              "blocked_share_by_src": by}
    (D / "thresholds.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out["gates"], indent=1))


def cmd_score(a) -> None:
    rows = {r["id"]: r for r in load("v3_rendered")}
    s0 = {json.loads(l)["id"]: json.loads(l) for l in (RUNS / "exp042_S0_v3/rows.jsonl").read_text().splitlines() if l.strip()}
    th = json.loads((D / "thresholds.json").read_text())
    if set(s0) != set(rows):
        sys.exit("REFUSED: S0 v3 pass incomplete")
    right = lambda r: int(r["verdict"] == "right")
    na = [i for i, r in rows.items() if r["kind"] in ("abstain", "oos")]          # the 38 abstention rows (H1)
    an = [i for i, r in rows.items() if r["label"] == "answerable"]              # source in material (H2)
    lost = [i for i, r in rows.items() if r["source_lost"]]
    res = {"n": len(rows), "answerable": len(an), "abstention_rows": len(na), "source_lost": len(lost),
           "S0": {"exact": round(sum(right(s0[i]) for i in rows) / len(rows), 4),
                  "abstain": round(sum(s0[i]["abstained"] for i in na) / len(na), 4),
                  "false_abstention": round(sum(s0[i]["abstained"] for i in an) / len(an), 4),
                  "source_lost_right": sum(right(s0[i]) for i in lost)}, "gates": {}}
    for gate, g in th["gates"].items():
        p = RUNS / f"exp042_v3_{gate}" / "rows.jsonl"
        if not p.exists():
            continue
        pr = probs("v3", gate)
        if set(pr) != set(rows):
            continue
        x = {}
        for i, r in rows.items():
            if pr[i] < g["tau"]:
                x[i] = {"abstained": True, "verdict": "right" if r["kind"] in ("abstain", "oos") else "wrong"}
            else:
                x[i] = {"abstained": s0[i]["abstained"], "verdict": s0[i]["verdict"]}
        h1 = C.paired([(rows[i]["lang"], int(x[i]["abstained"]), int(s0[i]["abstained"])) for i in na])
        ex = C.paired([(f"{rows[i]['lang']}/{rows[i]['kind']}", right(x[i]), right(s0[i])) for i in rows])
        fa = sum(x[i]["abstained"] for i in an) / len(an)
        res["gates"][gate] = {
            "tau": g["tau"], "abstain": {"S0": h1["y"], "gated": h1["x"], "rise": h1["diff"], "ci95": h1["ci95"],
                                         "mcnemar_p": h1["mcnemar_p_one_sided"]},
            "false_abstention": {"S0": res["S0"]["false_abstention"], "gated": round(fa, 4),
                                 "count": sum(x[i]["abstained"] for i in an), "of": len(an)},
            "exact": {"S0": ex["y"], "gated": ex["x"], "diff": ex["diff"], "ci95": ex["ci95"]},
            "source_lost": {"blocked": sum(pr[i] < g["tau"] for i in lost), "of": len(lost),
                            "S0_right_among_blocked": sum(right(s0[i]) for i in lost if pr[i] < g["tau"])},
            "dev_ece": g["ece"]}
    if "G4" in res["gates"]:
        m = res["gates"]["G4"]
        h = {"H1": m["abstain"]["rise"] >= H1_RISE and m["abstain"]["mcnemar_p"] < 0.05,
             "H2": m["false_abstention"]["gated"] <= H2_MAX, "H3": m["exact"]["ci95"][0] > H3_FLOOR, "H4": m["dev_ece"] <= H4_MAX}
        res["hypotheses"] = h
        res["verdict"] = ("No gate earns its place" if not (h["H1"] and h["H2"]) else
                          "Abstains, but costs answers" if not h["H3"] else "Gate adopted for the glue")
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("voice"); v.add_argument("--host", required=True)
    gp = sub.add_parser("gate"); gp.add_argument("--host", required=True); gp.add_argument("gate", choices=GATES)
    gp.add_argument("set", choices=["dev", "v3"])
    sub.add_parser("select")
    sc = sub.add_parser("score"); sc.add_argument("--json")
    a = ap.parse_args(argv)
    {"voice": cmd_voice, "gate": cmd_gate, "select": cmd_select, "score": cmd_score}[a.cmd](a)


if __name__ == "__main__":
    main()
