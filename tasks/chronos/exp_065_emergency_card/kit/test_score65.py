"""exp_065: the scorer on made-up verdicts, before any judge has run.
    python3.14 -I kit/test_score65.py
Builds two made-up runs (a mini and a MacBook) with three made-up models and made-up judges' files in a temporary
folder, runs score65's card on them, and checks the printed bars against values worked out here. No model answer
and no real verdict is used; only the ids and key sizes of the real task file."""
import contextlib
import io
import json
import os
import sys
import tempfile
from pathlib import Path

KIT = Path(__file__).resolve().parent
tmp = Path(tempfile.mkdtemp())
os.environ["SCORE65_JUDGING"] = str(tmp / "judging")
sys.path.insert(0, str(KIT))
import score65  # noqa: E402

tasks = json.loads((KIT / "tasks65.json").read_text())
ids = [t["id"] for t in tasks]


def make_run(name, machine, models):
    rd = tmp / name
    meta = {"machine": machine, "models": {}}
    for label, m in models.items():
        d = rd / score65.slug(label)
        d.mkdir(parents=True)
        meta["models"][label] = {"present": True, "template": "the file's own"}
        (d / "ladder.jsonl").write_text("".join(json.dumps({"id": f"A-{n}", "n_ctx": n, "usable": n <= m["usable"], "tokens_per_second": 50.0}) + "\n"
                                                for n in (8192, 16384, 32768, 65536)))
        (d / "tools.jsonl").write_text("".join(json.dumps({"id": f"C{i}", "right": i < m["tools"]}) + "\n" for i in range(5)))
        if "plan" in m:
            (d / "tasks.jsonl").write_text("".join(json.dumps({"id": tid, "answer": "" if m["plan"][tid] == "none" else "text",
                                                                "no_answer": m["plan"][tid] == "none", "seconds": 30.0}) + "\n" for tid in ids))
    (rd / "meta.json").write_text(json.dumps(meta))
    return rd


def plan(wrong=(), refused=(), none=()):
    return {tid: "wrong" if i in wrong else "refused" if i in refused else "none" if i in none else "right" for i, tid in enumerate(ids)}


alpha = plan(wrong=range(0, 8))                                    # 48 answered, 40 right
beta = plan(wrong=range(0, 5), refused=range(10, 20), none=(30, 31))   # 36 answered, 31 right, refusals > wrong
gamma = plan(wrong=(0, 1))                                          # 48 answered, 46 right
mini = make_run("mini-test", "mini", {"Alpha": {"usable": 32768, "tools": 5, "plan": alpha}, "Beta": {"usable": 8192, "tools": 0, "plan": beta}})
mbp = make_run("mbp-test", "mbp", {"Alpha": {"usable": 65536, "tools": 5}, "Gamma": {"usable": 16384, "tools": 4, "plan": gamma}})

# judges: both agree with the plan, except that judge 2 flips three of Alpha's answers and judge 3 restores the plan
flips = {score65.blind("mini-test", "alpha", ids[i]) for i in (20, 21, 22)}
for j in ("judge1", "judge2", "judge3"):
    (tmp / "judging" / j).mkdir(parents=True)
for t in tasks:
    out = {"judge1": [], "judge2": [], "judge3": []}
    for run, label, p in (("mini-test", "Alpha", alpha), ("mini-test", "Beta", beta), ("mbp-test", "Gamma", gamma)):
        kind = p[t["id"]]
        if kind == "none":
            continue
        b = score65.blind(run, score65.slug(label), t["id"])
        row = {"blind": b, "class": "refused" if kind == "refused" else "answered",
               "must": [kind == "right"] * len(t["must"]), "must_not": [False] * len(t.get("must_not", []))}
        out["judge1"].append(row)
        if b in flips:
            out["judge2"].append(dict(row, must=[False] * len(t["must"])))
            out["judge3"].append(row)
        else:
            out["judge2"].append(row)
    for j, rows_ in out.items():
        (tmp / "judging" / j / f"{t['id']}.json").write_text(json.dumps(rows_))

buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    score65.card([str(mini), str(mbp)])
text = buf.getvalue()
checks = {
    "Alpha: 48 answered, 40 right, emergency-usable": "Alpha" in text and "answered 48, right 40" in text and "| EMERGENCY-USABLE" in text,
    "Beta: 36 answered, 31 right": "answered 36, right 31" in text,
    "Beta: refusals are its main limit": "REFUSALS ARE THIS MODEL'S MAIN LIMIT" in text,
    "Beta: two no-answers counted apart": "'no answer': 2" in text,
    "Gamma: 46 right, carried from the MacBook run": "answered 48, right 46" in text and "on the mbp" in text,
    "Alpha's tasks shown on the MacBook card too": text.count("answered 48, right 40") == 2,
    "H1 mini passes with Alpha": "H1 on the mini: PASS: Alpha" in text,
    "H1 MacBook passes with Alpha and Gamma, Gamma opens first": "H1 on the mbp: PASS: Alpha, Gamma" in text and "the first model to open: Gamma" in text,
    "long documents on the MacBook: Alpha at 65536": "the model for long documents: Alpha (usable at 65536" in text,
    "H4 mini passes (32768), MacBook passes (65536)": "H4 on the mini (usable at 32768): PASS: Alpha" in text and "H4 on the mbp (usable at 65536): PASS: Alpha" in text,
    "H2 fails for Beta": "H2 (wrong > refused for every candidate): FAIL for: Beta" in text,
    "H3 passes: 46 against 40": "best MacBook-only 46, best that fits the mini 40: PASS" in text,
    "three disagreements, settled by the third judge": "'disagreed': 3" in text and "'undecided': 0" in text,
}
for name, ok in checks.items():
    print(("ok   " if ok else "FAIL ") + name)
if not all(checks.values()):
    print(text)
    sys.exit(1)
print(f"{len(checks)} of {len(checks)} passed")
