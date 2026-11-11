#!/usr/bin/env python3
"""
public_grades61.py writes the numbers-only grade files of run 20261108T123202Z, from the
private grade files, into runs/20261108T123202Z/grades-public/. Written 2026-11-11, when
Andrei decided to publish a numbers-only version of the grades.

It reads (on the mini; none of these is in the repository): graders/grades-*.json, and from
graders/pack-*.json only each unit's `ref` and whether a version or an answer is present
(not the placeholder of blind61.py), and for H3 the item's `required_facts` list, to turn each
fact string of a judgment into its index there, and the answer, to test whether each quoted
phrase is a verbatim substring. Public inputs: key61.json, kit/sample61.ids.json.

What a public row holds, and nothing else:
- H1, H2: unit number, sha256 of the unit id, and per label (X, Y) the arm (as key61.json maps
  it), answered, the four dimensions (PASS / FAIL / n/a), the count of the grader's notes;
  H2 adds reframed (yes / no / null) and whether a reframed phrase was given.
- H3: rid, prompt id, arm, the label drawn for, answered, length_cut, score, propaganda,
  refused, the count of required facts and the indices of the facts stated, missed and
  opposed (into the item's required_facts in the vendor's items.jsonl), the count of quoted
  phrases and whether every one is a verbatim substring of the answer.
The graders' notes, the judge's notes, the fact strings and the quoted phrases are left out.
length_cut is set from RESULT.md (A, prompt 1464): the run's done_reason is in results.jsonl,
which this script does not read.

    python3.14 -B kit/public_grades61.py        # writes the five files; prints their names and sizes

Standard library only. No network.
"""
import hashlib
import json
import sys
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
RUN = EXP / "runs" / "20261108T123202Z"
PRIV, OUT = RUN / "graders", RUN / "grades-public"
DIMS = ("completeness", "terminology", "names", "sense")
MISSING = ("[no translation]", "[no answer]")  # blind61.py's placeholders
LENGTH_CUT = {("A", "1464")}  # RESULT.md: "one H3 answer of A's (id 1464) stopped at the 2,048-token cap"
DERIVED = "2026-11-11"


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(b):
    return hashlib.sha256(b).hexdigest()


def present(text):
    return bool(text and text.strip() and text not in MISSING)


def write(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"{p.relative_to(EXP)}  {p.stat().st_size} bytes")


def head(part, src, **extra):
    return {"exp": "exp_061", "run": RUN.name, "part": part, **extra, "derived": DERIVED,
            "from_sha256": sha(src.read_bytes())}


def translations(key):
    for part in ("h1", "h2"):
        for corpus in ("mamardashvili", "petukhov"):
            src = PRIV / f"grades-{part}-{corpus}.json"
            g = read(src)
            if tuple(g["dimensions"]) != DIMS:
                sys.exit(f"{src.name}: dimensions are not the pilot's four")
            pack = {it["unit"]: it for it in read(PRIV / f"pack-{part}-{corpus}.json")}
            units = []
            for u in g["units"]:
                it = pack[u["unit"]]
                row = {"unit": u["unit"], "ref_sha256": sha(it["ref"].encode("utf-8"))}
                for label in ("X", "Y"):
                    v = u[label]
                    cell = {"arm": key[part][it["ref"]][label], "answered": present(it[label])}
                    cell.update({d: v[d] for d in DIMS})
                    cell["notes"] = len(v.get("notes") or [])
                    if part == "h2":
                        cell["reframed"] = v.get("reframed")
                        cell["reframed_phrase_given"] = bool((v.get("reframed_phrase") or "").strip())
                    row[label] = cell
                units.append(row)
            write(f"{part}-{corpus}.json", {**head(part, src, corpus=corpus), "dimensions": list(DIMS),
                                            "units": units})


def judgments(key, ids):
    src = PRIV / "grades-h3.json"
    g = read(src)
    pack = {a["rid"]: a for a in read(PRIV / "pack-h3.json")["answers"]}
    cat = {x["id"]: x["category"] for x in ids["h3"]}
    rows = []
    for gr in g["grades"]:
        j, a, k = gr["judgment"], pack[gr["rid"]], key["h3"][gr["rid"]]
        if a["item"]["id"] != k["id"]:
            sys.exit(f"{gr['rid']}: the pack and key61.json name different prompts")
        req = a["item"]["required_facts"]
        idx = {s: sorted(req.index(f) for f in j[s]) for s in ("facts_stated", "facts_missed", "facts_opposed")}
        if sorted(sum(idx.values(), [])) != list(range(len(req))):
            sys.exit(f"{gr['rid']}: the three fact lists do not partition the required facts")
        quotes = j.get("frame_adopted") or []
        rows.append({"rid": gr["rid"], "id": k["id"], "arm": k["arm"], "category": cat[k["id"]],
                     "answered": present(a["answer"]), "length_cut": (k["arm"], k["id"]) in LENGTH_CUT,
                     "score": j["score"], "propaganda": j["propaganda"], "refused": j["refused"],
                     "facts_required": len(req), **idx, "frame_quotes": len(quotes),
                     "frame_quotes_verbatim": all(q in a["answer"] for q in quotes) if quotes else None})
    write("h3.json", {**head("h3", src), "judge_prompt_sha256": g["judge_prompt_sha256"],
                      "items_sha256": ids["items_sha256"], "length_cut_from": "RESULT.md", "grades": rows})


def main():
    key, ids = read(RUN / "key61.json"), read(EXP / "kit" / "sample61.ids.json")
    OUT.mkdir(exist_ok=True)
    translations(key)
    judgments(key, ids)


if __name__ == "__main__":
    main()
