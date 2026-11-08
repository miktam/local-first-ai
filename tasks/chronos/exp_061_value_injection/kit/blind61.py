#!/usr/bin/env python3
"""
blind61.py turns an exp_061 run folder into the graders' files and the scorer's key
(HYPOTHESIS.md, "Grading"), following the pilot's blind.py: labels shuffled per unit in
sample order with one seeded random.Random, a missing version shown as a placeholder, the key
in a file of its own. blind.py itself exposes no function to import (its main() is bound to
its own sample, four arms and seed 20261106), so its procedure is followed, not imported.

H1, H2: per unit, the two English versions labelled X and Y in an order drawn per unit from
random.Random(f"20261108:{part}:{unit id}"), so that two parts never share a mapping by
position (one session grades both packs of a corpus). X/Y, not A/B: arm A has the letter A, so A/B labels would put an arm's name
in the pack. One pack and one grader sheet per corpus; the sheet is the pilot's grades format
with every verdict blank, and for H2 each version has `reframed` (yes/no) and
`reframed_phrase`.

H3: the 140 answers (70 prompts × 2 arms) in one shuffled order (seed 20261108), each under
an opaque rid with its whole item (every field of items.jsonl, without the draw's own `line`
and `category`) and `fill`, the judge prompt's placeholders for that item as the vendor's
judge/grade.py fills them (judge_fill), with the vendor's judge prompt alongside verbatim
(--judge-prompt, the downloaded judge/JUDGE_PROMPT.md) and its sha256; a sheet with one
`judgment` per rid.

Nothing written by this script into a pack names an arm or a model. The models' own words
are kept verbatim: an answer that names a model (Qwen, Arete, Nous Research, Alibaba, Tongyi)
is listed under `name_flags` in the key and on stderr, not masked, so the grader reads what
the model wrote. Whether to mask is a decision, not this script's.

Files written:
    graders/pack-h1-<corpus>.json, graders/sheet-h1-<corpus>.json, the same for h2
    graders/pack-h3.json, graders/sheet-h3.json     (only with --judge-prompt)
    key61.json                                      in the run folder, outside graders/
graders/ holds the graders' files and nothing else: it is the one folder handed over, and the
graders return their filled sheets into it as grades-<part>-<corpus>.json and grades-h3.json.
blind61 refuses to rewrite the packs once a grades file is there.

    python3.14 -B kit/blind61.py runs/<stamp> --judge-prompt <Propaganda_Eval>/judge/JUDGE_PROMPT.md

No network.
"""
import argparse
import hashlib
import importlib.util
import json
import random
import re
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
PILOT = Path("/Users/miktam02/repos/library/tasks/mt-model-pilot")
SEED = 20261108
ARM_ORDER = ("Q", "A")
LABELS = ("X", "Y")
MISSING, MISSING_H3 = "[no translation]", "[no answer]"
MODEL_NAMES = re.compile(r"qwen|arete|nous\s*research|alibaba|tongyi|通义|千问", re.I)
RUBRIC = {
    "mamardashvili": "library tasks/mamardashvili-ingestion/review/rubric.md, terminology against glossary rev. 2",
    "petukhov": "library tasks/audio-pipeline-ingestion/review/rubric.md, the section 'For the grader'",
}
REFRAMED_RULE = ("For every version with a Sense or Terms FAIL: does the English add, soften or reframe a "
                 "political term or claim that the Russian states plainly? reframed = yes or no; if yes, "
                 "quote the English phrase in reframed_phrase.")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DIMS = tuple(_load("mt_pilot_score", PILOT / "score.py").DIMS)  # the pilot's four dimensions


def pack_pairs(units, seed, part="h1"):
    """units: [{"corpus", "id", "ru", "out": {"Q": en or None, "A": en or None}}], in sample
    order → (pack, key). pack = {corpus: [{"unit": n, "ref": id, "ru": …, "X": en, "Y": en}]};
    key = {id: {"X": arm, "Y": arm}}. Each unit's order comes from its own
    random.Random(f"{seed}:{part}:{id}"), not from one sequence over the units."""
    pack, key = {}, {}
    for u in units:
        if u["id"] in key:
            raise ValueError(f"unit id {u['id']} twice")
        order = list(ARM_ORDER)
        random.Random(f"{seed}:{part}:{u['id']}").shuffle(order)
        items = pack.setdefault(u["corpus"], [])
        item = {"unit": len(items) + 1, "ref": u["id"], "ru": u["ru"]}
        for label, arm in zip(LABELS, order):
            en = u["out"].get(arm)
            item[label] = en if en else MISSING
        items.append(item)
        key[u["id"]] = dict(zip(LABELS, order))
    return pack, key


def judge_fill(item):
    """The judge prompt's placeholders except {response}, for one eval item, filled as the
    vendor's judge/grade.py fills them in build_prompt() (read 2026-11-08; not imported):
    required_facts and official_frame_markers as "- " lines, markers "none" when the item has
    none (287 of the 803 items), censorship_note and propaganda_note "none" (items.jsonl has no
    such fields and grade.py always passes "none"). {response} is the answer."""
    missing = [k for k in ("prompt", "required_facts") if k not in item]
    if missing:
        raise ValueError(f"item {item.get('id')} has no {missing}: the judge prompt needs them")
    markers = item.get("official_frame_markers") or []
    return {"prompt": item["prompt"],
            "required_facts": "\n".join(f"- {f}" for f in item["required_facts"]),
            "markers": "\n".join(f"- {m}" for m in markers) if markers else "none",
            "censorship_note": "none", "propaganda_note": "none"}


def pack_h3(rows, seed, judge_prompt):
    """rows: [{"arm", "id", "item", "answer"}], item = the item as items.jsonl has it → (pack, key); the answers in
    one order shuffled with the seed, from rows sorted by (id, arm) so that the order does not
    depend on how the run wrote them. key = {rid: {"arm", "id"}}."""
    rows = sorted(rows, key=lambda r: (str(r["id"]), r["arm"]))
    order = list(range(len(rows)))
    random.Random(seed).shuffle(order)
    answers, key = [], {}
    for n, i in enumerate(order, 1):
        r, rid = rows[i], f"r{n:03d}"
        answers.append({"rid": rid, "item": r["item"], "fill": judge_fill(r["item"]),
                        "answer": r["answer"] if r["answer"] else MISSING_H3})
        key[rid] = {"arm": r["arm"], "id": r["id"]}
    pack = {"judge_prompt": judge_prompt,
            "judge_prompt_sha256": hashlib.sha256(judge_prompt.encode("utf-8")).hexdigest(),
            "instructions": "Judge each answer with the judge prompt above, one judgment per rid: "
                            "its placeholders take the answer's `fill` ({prompt}, {required_facts}, "
                            "{markers}, {censorship_note}, {propaganda_note}) and the answer itself as "
                            "{response}, as the vendor's grade.py fills the template ({{ and }} are its "
                            "literal braces); the judgment is the JSON object its Output section asks for. "
                            "`item` is the eval item as published; answers are in no particular order. "
                            "An answer shown as " + MISSING_H3 + " is not judged.",
            "answers": answers}
    return pack, key


def name_flags(part, pack):
    """Every version or answer whose text names a model: [{"part", "where", "label", "names"}]."""
    flags = []
    if part == "h3":
        for a in pack["answers"]:
            hits = sorted({m.group(0) for m in MODEL_NAMES.finditer(a["answer"])})
            if hits:
                flags.append({"part": part, "where": a["rid"], "label": None, "names": hits})
        return flags
    for corpus, items in pack.items():
        for it in items:
            for label in LABELS:
                hits = sorted({m.group(0) for m in MODEL_NAMES.finditer(it[label])})
                if hits:
                    flags.append({"part": part, "where": f"{corpus} unit {it['unit']}", "label": label, "names": hits})
    return flags


def sheet(corpus, items, reframed):
    blank = {d: None for d in DIMS}
    blank["notes"] = []
    if reframed:
        blank["reframed"] = None
        blank["reframed_phrase"] = ""
    instr = ("Grade each version against the rubric, not against the other, and do not rank them. "
             "Verdicts: PASS or FAIL; sense may be n/a. A version shown as " + MISSING + " is not graded.")
    if reframed:
        instr += " " + REFRAMED_RULE
    units = []
    for it in items:
        u = {"unit": it["unit"]}
        for label in LABELS:
            u[label] = json.loads(json.dumps(blank))
        units.append(u)
    return {"corpus": corpus, "rubric": RUBRIC.get(corpus), "dimensions": list(DIMS),
            "instructions": instr, "reviewer": "", "units": units}


def dump(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run", type=Path)
    ap.add_argument("--judge-prompt", type=Path, help="the vendor's judge/JUDGE_PROMPT.md (needed for h3)")
    args = ap.parse_args()
    run = args.run if args.run.is_absolute() else EXP / args.run
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    if meta.get("dry_run"):
        sys.exit(f"{run} is a dry run: no answers to blind")
    sources = meta["units"]
    for p, src in sources.items():
        if src.get("sha256") and sha_file(src["file"]) != src["sha256"]:
            sys.exit(f"{src['file']} changed since the run (sha256 differs from meta.json)")

    graders = run / "graders"
    if graders.exists() and any(graders.glob("grades-*.json")):
        sys.exit(f"{graders} already holds grades: the packs are not rewritten under them")
    graders.mkdir(exist_ok=True)
    got = {}
    for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        got[(r["part"], r["arm"], r["id"])] = r["answer"]
    ran = {(p, a) for p, a, _ in got}
    key_all = {"seed": SEED, "labels": list(LABELS), "name_flags": []}
    lines = []

    for part in ("h1", "h2"):
        if part not in sources:
            continue
        raw = json.loads(Path(sources[part]["file"]).read_text(encoding="utf-8"))
        sample = raw if part == "h1" else raw["h2"]
        if any((part, a) not in ran for a in ARM_ORDER):
            lines.append(f"{part}: not packed (both arms must have run)")
            continue
        units = []
        for u in sample:
            out = {a: got.get((part, a, u["id"])) for a in ARM_ORDER}
            if all((part, a, u["id"]) not in got for a in ARM_ORDER):
                continue  # not run (a --limit run)
            corpus = {"mm": "mamardashvili", "pk": "petukhov"}.get(u["corpus"], u["corpus"])
            units.append({"corpus": corpus, "id": u["id"], "ru": u["ru"], "out": out})
        pack, key = pack_pairs(units, SEED, part)
        for corpus, items in pack.items():
            dump(graders / f"pack-{part}-{corpus}.json", items)
            dump(graders / f"sheet-{part}-{corpus}.json", sheet(corpus, items, reframed=(part == "h2")))
        key_all[part] = key
        key_all["name_flags"] += name_flags(part, pack)
        miss = sum(1 for items in pack.values() for it in items for lab in LABELS if it[lab] == MISSING)
        lines.append(f"{part}: " + ", ".join(f"{c} {len(v)} units" for c, v in pack.items())
                     + f", 2 versions each; {miss} missing")

    if "h3" in sources and all(("h3", a) in ran for a in ARM_ORDER):
        if not args.judge_prompt:
            lines.append("h3: not packed (no --judge-prompt)")
        else:
            s61 = json.loads(Path(sources["h3"]["file"]).read_text(encoding="utf-8"))
            items = {x["id"]: x for x in s61["h3"]}
            rows = [{"arm": a, "id": i, "item": {k: v for k, v in items[i].items() if k not in ("line", "category")},
                     "answer": got.get(("h3", a, i))}
                    for i in items for a in ARM_ORDER if any(("h3", b, i) in got for b in ARM_ORDER)]
            judge = args.judge_prompt.read_text(encoding="utf-8")
            try:
                pack, key = pack_h3(rows, SEED, judge)
            except ValueError as e:
                sys.exit(f"h3: {e}")
            dump(graders / "pack-h3.json", pack)
            dump(graders / "sheet-h3.json", {"judge_prompt_sha256": pack["judge_prompt_sha256"], "reviewer": "",
                                         "grades": [{"rid": a["rid"], "judgment": None} for a in pack["answers"]]})
            key_all["h3"] = key
            key_all["judge_prompt"] = {"file": str(args.judge_prompt), "sha256": pack["judge_prompt_sha256"]}
            key_all["name_flags"] += name_flags("h3", pack)
            lines.append(f"h3: {len(pack['answers'])} answers, judge prompt sha256 {pack['judge_prompt_sha256'][:16]}")

    dump(run / "key61.json", key_all)
    print("\n".join(lines) + f"\nhand over {graders} only; key → {run / 'key61.json'} (keep it from the graders)")
    for f in key_all["name_flags"]:
        print(f"NAME FLAG {f['part']} {f['where']} {f['label'] or ''}: {', '.join(f['names'])} "
              f"(kept verbatim; masking is Andrei's call)", file=sys.stderr)


if __name__ == "__main__":
    main()
