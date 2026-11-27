"""exp_065, the scorer.

    python3.14 -I kit/score65.py packets runs/mini-<stamp> [runs/mbp-<stamp> ...]   # blind packets for the judges
    python3.14 -I kit/score65.py card    runs/mini-<stamp> [runs/mbp-<stamp> ...]   # the tables, once the judges are in

packets: writes judging/packets/<task id>.json (the task, its key, and every non-empty answer under a blind id) and
judging/map.json (blind id -> run, model, task; never shown to a judge). Empty answers are "no answer" by code and are
not sent to a judge.

A judge returns, per task, a JSON list of {"blind": "...", "class": "answered" | "part" | "refused",
"must": [true/false per must-point], "must_not": [true/false per must-not point]}, saved by the dispatcher as
judging/judge1/<task id>.json and judging/judge2/<task id>.json; judging/judge3/<task id>.json holds the third
judge's rows for the blind ids the first two disagreed on.

card: prints, per run and model, Part A (usable context, speed), Part B (answered, right, the three ways to lose, per
type) and Part C, then the bars H1 to H4 as far as the data allows."""
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

KIT = Path(__file__).resolve().parent
EXP = KIT.parent
TASKS = KIT / "tasks65.json"
JUDGING = Path(os.environ.get("SCORE65_JUDGING") or EXP / "judging")  # the env variable is for the scorer's own test
SALT = "exp065-2026-11"


def rows(path: Path) -> list:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def answers(run_dirs: list) -> dict:
    """(run, model slug, task id) -> row, for every Part B row of every run."""
    out = {}
    for rd in run_dirs:
        rd = Path(rd)
        for d in sorted(p for p in rd.iterdir() if p.is_dir()):
            for r in rows(d / "tasks.jsonl"):
                out[(rd.name, d.name, r["id"])] = r
    return out


def blind(run, model, task) -> str:
    return hashlib.sha256(f"{SALT}|{run}|{model}|{task}".encode()).hexdigest()[:10]


def packets(run_dirs: list):
    tasks = {t["id"]: t for t in json.loads(TASKS.read_text())}
    by_task, mapping = defaultdict(list), {}
    for (run, model, tid), r in answers(run_dirs).items():
        if r["no_answer"]:
            continue
        b = blind(run, model, tid)
        mapping[b] = {"run": run, "model": model, "task": tid}
        by_task[tid].append({"blind": b, "answer": r["answer"], "cut_at_the_cap": r.get("cut", False)})
    (JUDGING / "packets").mkdir(parents=True, exist_ok=True)
    for tid, items in by_task.items():
        items.sort(key=lambda x: x["blind"])  # the order says nothing about the model
        t = tasks[tid]
        (JUDGING / "packets" / f"{tid}.json").write_text(json.dumps(
            {"task": tid, "prompt": t["prompt"], "must": t["must"], "must_not": t.get("must_not", []),
             "judge_note": t.get("judge_note", ""), "answers": items}, ensure_ascii=False, indent=1))
    (JUDGING / "map.json").write_text(json.dumps(mapping, indent=1))
    print(f"{len(mapping)} answers in {len(by_task)} packets; judging/map.json is not for the judges")


def verdict(row: dict, n_must: int, n_not: int):
    """(class, right) from one judge's row, or None when the row is malformed."""
    try:
        cls = row["class"]
        must, must_not = list(row["must"]), list(row.get("must_not", []))
        if cls not in ("answered", "part", "refused") or len(must) != n_must or len(must_not) != n_not:
            return None
        return cls, cls == "answered" and all(must) and not any(must_not)
    except Exception:
        return None


def judged() -> tuple:
    """blind id -> (class, right) by two judges and a third on disagreement; plus counts."""
    tasks = {t["id"]: t for t in json.loads(TASKS.read_text())}
    final, stats = {}, {"judged": 0, "disagreed": 0, "undecided": 0, "malformed": 0}
    for tid, t in tasks.items():
        n_must, n_not = len(t["must"]), len(t.get("must_not", []))
        js = []
        for j in ("judge1", "judge2", "judge3"):
            f = JUDGING / j / f"{tid}.json"
            js.append({r["blind"]: verdict(r, n_must, n_not) for r in json.loads(f.read_text())} if f.exists() else {})
        for b in set(js[0]) | set(js[1]):
            a, c, third = js[0].get(b), js[1].get(b), js[2].get(b)
            if a is None or c is None:
                stats["malformed"] += 1
                continue
            stats["judged"] += 1
            if a == c:
                final[b] = a
            else:
                stats["disagreed"] += 1
                if third is not None:
                    final[b] = third
                else:
                    stats["undecided"] += 1
    return final, stats


def slug(label: str) -> str:
    return "".join(c if c.isalnum() else "-" for c in label).strip("-").lower()


def collect(run_dirs: list):
    """Per run and model: Part A, Part C, and Part B judged. Part B is run once per file, so a model's task result is
    taken from whichever run holds its task rows and shown on every machine that has the file."""
    tasks = json.loads(TASKS.read_text()) if TASKS.exists() else []
    types = {t["id"]: t["type"] for t in tasks}
    final, stats = judged() if tasks and (JUDGING / "judge1").exists() else ({}, None)
    ans = answers(run_dirs)
    machines, taskres = {}, {}
    for rd in map(Path, run_dirs):
        meta = json.loads((rd / "meta.json").read_text())
        models = {}
        for label, info in meta["models"].items():
            d = rd / slug(label)
            if not info.get("present") or not d.is_dir():
                continue
            rec = {"template": info.get("template"), "note": info.get("note", "")}
            ladder = [r for r in rows(d / "ladder.jsonl") if "usable" in r]
            rec["tried"] = [r["n_ctx"] for r in ladder]
            rec["usable"] = max([r["n_ctx"] for r in ladder if r["usable"]], default=0)
            rec["tps_8k"] = next((r["tokens_per_second"] for r in ladder if r["n_ctx"] == 8192), None)
            tools = rows(d / "tools.jsonl")
            rec["tools"] = (sum(r["right"] for r in tools), len(tools)) if tools else None
            models[label] = rec
            mine = {tid: r for (run, model, tid), r in ans.items() if run == rd.name and model == d.name}
            if mine and len(mine) > len(taskres.get(label, {}).get("rows", {})):
                per_type = {}
                lost = {"refused or in part": 0, "wrong": 0, "no answer": 0, "not judged": 0}
                for tid, r in mine.items():
                    pt = per_type.setdefault(types.get(tid, 0), [0, 0, 0])  # answered, right, of
                    pt[2] += 1
                    if r["no_answer"]:
                        lost["no answer"] += 1
                        continue
                    v = final.get(blind(rd.name, d.name, tid))
                    if v is None:
                        lost["not judged"] += 1
                    elif v[0] != "answered":
                        lost["refused or in part"] += 1
                    else:
                        pt[0] += 1
                        if v[1]:
                            pt[1] += 1
                        else:
                            lost["wrong"] += 1
                secs = sorted(r["seconds"] for r in mine.values())
                taskres[label] = {"rows": mine, "run": rd.name, "machine": meta["machine"], "per_type": per_type, "lost": lost,
                                  "answered": sum(v[0] for v in per_type.values()), "right": sum(v[1] for v in per_type.values()),
                                  "median_seconds": secs[len(secs) // 2]}
        machines.setdefault(meta["machine"], {}).update(models)
    return machines, taskres, stats, bool(final), len(tasks)


def emergency_usable(rec: dict, tr) -> bool:
    return bool(tr and tr["answered"] >= 46 and all(v[0] >= 7 for v in tr["per_type"].values()) and tr["right"] >= 34
                and rec["usable"] >= 16384 and rec["tools"] and rec["tools"][0] >= 4)


def card(run_dirs: list):
    machines, taskres, stats, have_judges, n_tasks = collect(run_dirs)
    for machine, models in machines.items():
        print(f"\n== {machine} ==")
        for label, rec in models.items():
            line = [label, f"template: {rec['template']}",
                    f"usable context: {rec['usable'] or 'none'} (tried {rec['tried']})" if rec["tried"] else "no ladder yet",
                    f"{rec['tps_8k']} tok/s at 8k" if rec["tps_8k"] else "", f"tool calls {rec['tools'][0]} of {rec['tools'][1]}" if rec["tools"] else ""]
            tr = taskres.get(label)
            if tr:
                line.append(f"tasks run {len(tr['rows'])} on the {tr['machine']}, median {tr['median_seconds']} s, no answer {tr['lost']['no answer']}")
                if have_judges:
                    line.append(f"answered {tr['answered']}, right {tr['right']}, lost: {tr['lost']}")
                    line.append("per type (answered/right/of): " + ", ".join(f"{k}: {v[0]}/{v[1]}/{v[2]}" for k, v in sorted(tr["per_type"].items())))
                    w, r = tr["lost"]["wrong"], tr["lost"]["refused or in part"]
                    line.append("H2: " + ("wrong answers cost more than refusals" if w > r else
                                          "REFUSALS ARE THIS MODEL'S MAIN LIMIT" if r > w else "wrong = refused"))
                    line.append("EMERGENCY-USABLE" if emergency_usable(rec, tr) else "not emergency-usable")
            if rec["note"]:
                line.append(f"note: {rec['note']}")
            print(" | ".join(x for x in line if x))
    if not have_judges:
        print("\nno judges' files yet: Part B shows counts by code only")
        return
    print("\n== the bars ==")
    full = {l: tr for l, tr in taskres.items() if len(tr["rows"]) == n_tasks}
    for machine, models in machines.items():
        ok = [l for l, rec in models.items() if emergency_usable(rec, full.get(l))]
        print(f"H1 on the {machine}: {'PASS: ' + ', '.join(ok) if ok else 'FAIL: no model is emergency-usable'}")
        if ok:
            first = max(ok, key=lambda l: full[l]["right"])
            print(f"   the first model to open: {first} (right on {full[first]['right']})")
        longdoc = [l for l, rec in models.items() if l in full and full[l]["right"] >= 34 and rec["usable"]]
        if longdoc:
            best = max(longdoc, key=lambda l: (models[l]["usable"], full[l]["right"]))
            print(f"   the model for long documents: {best} (usable at {models[best]['usable']}, right on {full[best]['right']})")
        need = 32768 if machine == "mini" else 65536
        got = [l for l, rec in models.items() if rec["usable"] >= need]
        print(f"H4 on the {machine} (usable at {need}): {'PASS: ' + ', '.join(got) if got else 'FAIL'}")
    rev = [l for l, tr in full.items() if tr["lost"]["refused or in part"] >= tr["lost"]["wrong"]]
    print(f"H2 (wrong > refused for every candidate): {'PASS' if full and not rev else 'FAIL for: ' + ', '.join(rev) if rev else 'no complete model'}")
    if "mini" in machines and "mbp" in machines:
        only_mbp = [l for l in machines["mbp"] if l not in machines["mini"] and l in full]
        fits = [l for l in machines["mini"] if l in full]
        if only_mbp and fits:
            a, b = max(full[l]["right"] for l in only_mbp), max(full[l]["right"] for l in fits)
            print(f"H3 (the larger machine buys better answers): best MacBook-only {a}, best that fits the mini {b}: "
                  f"{'PASS' if a - b >= 5 else 'FAIL: the larger machine buys context and speed'}")
    if full:
        rights = [tr["right"] for tr in full.values()]
        if min(rights) >= 44 or max(rights) <= 10:
            print("DECLARED: the tasks did not separate the models (every candidate at 44 or more, or at 10 or fewer)")
    rate = stats["disagreed"] / stats["judged"] if stats["judged"] else 0
    print(f"judges: {stats} | disagreement {rate:.1%}" + (" | ABOVE 15 %: the right-or-wrong column is unreliable" if rate > 0.15 else ""))


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("packets", "card"):
        sys.exit(__doc__)
    (packets if sys.argv[1] == "packets" else card)(sys.argv[2:])
