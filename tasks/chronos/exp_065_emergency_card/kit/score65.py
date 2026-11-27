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
import sys
from collections import defaultdict
from pathlib import Path

KIT = Path(__file__).resolve().parent
EXP = KIT.parent
TASKS = KIT / "tasks65.json"
JUDGING = EXP / "judging"
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


def card(run_dirs: list):
    tasks = json.loads(TASKS.read_text()) if TASKS.exists() else []
    types = {t["id"]: t["type"] for t in tasks}
    final, stats = judged() if tasks and (JUDGING / "judge1").exists() else ({}, None)
    ans = answers(run_dirs)
    for rd in run_dirs:
        rd = Path(rd)
        meta = json.loads((rd / "meta.json").read_text())
        print(f"\n== {rd.name} ({meta['machine']}) ==")
        for label, info in meta["models"].items():
            if not info.get("present"):
                print(f"{label}: not on this machine")
                continue
            d = rd / "".join(c if c.isalnum() else "-" for c in label).strip("-").lower()
            if not d.is_dir():
                continue
            line = [label, f"template: {info.get('template')}"]
            ladder = [r for r in rows(d / "ladder.jsonl") if "usable" in r]
            if ladder:
                usable = [r["n_ctx"] for r in ladder if r["usable"]]
                at8 = next((r for r in ladder if r["n_ctx"] == 8192), None)
                line.append(f"usable context: {max(usable) if usable else 'none'} (tried {[r['n_ctx'] for r in ladder]})")
                if at8:
                    line.append(f"{at8['tokens_per_second']} tok/s at 8k")
            tools = rows(d / "tools.jsonl")
            if tools:
                line.append(f"tool calls {sum(r['right'] for r in tools)} of {len(tools)}")
            mine = {tid: r for (run, model, tid), r in ans.items() if run == rd.name and model == d.name}
            if mine:
                no_answer = sum(r["no_answer"] for r in mine.values())
                line.append(f"tasks run {len(mine)}, no answer {no_answer}")
                if final:
                    per_type = defaultdict(lambda: [0, 0, 0])  # answered, right, total
                    lost = {"refused or in part": 0, "wrong": 0, "no answer": no_answer, "not judged": 0}
                    for tid, r in mine.items():
                        ty = types.get(tid, 0)
                        per_type[ty][2] += 1
                        if r["no_answer"]:
                            continue
                        v = final.get(blind(rd.name, d.name, tid))
                        if v is None:
                            lost["not judged"] += 1
                        elif v[0] != "answered":
                            lost["refused or in part"] += 1
                        else:
                            per_type[ty][0] += 1
                            if v[1]:
                                per_type[ty][1] += 1
                            else:
                                lost["wrong"] += 1
                    answered = sum(v[0] for v in per_type.values())
                    right = sum(v[1] for v in per_type.values())
                    line.append(f"answered {answered}, right {right}, lost: {lost}")
                    line.append("per type (answered/right/of): " + ", ".join(f"{k}: {v[0]}/{v[1]}/{v[2]}" for k, v in sorted(per_type.items())))
                    line.append("H2 " + ("wrong > refused" if lost["wrong"] > lost["refused or in part"] else "REFUSALS ARE THE MAIN LIMIT"
                                         if lost["refused or in part"] > lost["wrong"] else "wrong = refused"))
            print(" | ".join(str(x) for x in line))
    if stats:
        rate = stats["disagreed"] / stats["judged"] if stats["judged"] else 0
        print(f"\njudges: {stats} | disagreement {rate:.1%}" + (" | ABOVE 15 %: the right-or-wrong column is unreliable" if rate > 0.15 else ""))


if __name__ == "__main__":
    if len(sys.argv) < 3 or sys.argv[1] not in ("packets", "card"):
        sys.exit(__doc__)
    (packets if sys.argv[1] == "packets" else card)(sys.argv[2:])
