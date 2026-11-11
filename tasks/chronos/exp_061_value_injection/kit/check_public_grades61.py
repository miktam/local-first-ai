#!/usr/bin/env python3
"""
check_public_grades61.py guards the numbers-only grade files against leaks. Two checks, counts
only; it prints no text of a lecture, an answer, a note or a quotation.

1. Every string in runs/20261108T123202Z/grades-public/*.json, keys and values, is at most 64
   characters, holds no Cyrillic character, and is an id (r001, 1464, the run stamp, a date),
   a sha256, or a label from a closed list (the arms, the blind labels, PASS/FAIL/n/a, yes/no,
   the parts and corpora, the rubric's four dimensions, the vendor's seven category labels as
   kit/sample61.ids.json lists them, the field names).
2. No 30-character window of the private text occurs in any file of the experiment folder that
   git tracks or would add (`git ls-files`, tracked and untracked-not-ignored). The private text
   is read in code from runs/20261108T123202Z/graders/ (on the mini only): the `ru`, `X` and `Y`
   of the H1 and H2 packs, the `answer` of the H3 pack, the graders' notes, the judge's notes
   and the phrases the judge quoted (`frame_adopted`). The fact strings of the judgments are
   left out: each is a `required_facts` entry of the vendor's published items.jsonl, verbatim.
   Whitespace is collapsed on both sides, so a line break does not hide a window.

    python3.14 -B kit/check_public_grades61.py              # strict: any window fails
    python3.14 -B kit/check_public_grades61.py --new-only   # windows not already in the 2026-11-08 text fail

Exit 0 when both checks pass; 1 on any string that fails check 1 or any window found by
check 2, listed by file and line number with the kind of private text it matches, and split
into windows already in that file as published on 2026-11-08 (commit 5514c31, which brought
RESULT.md's quotations) and new ones. With --new-only only the new ones fail. Standard
library only. No network.
"""
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent
RUN = EXP / "runs" / "20261108T123202Z"
PUB, PRIV = RUN / "grades-public", RUN / "graders"
W = 30
BASE = "5514c31"  # the result as published on 2026-11-08
CYR = re.compile(r"[Ѐ-ӿԀ-ԯ]")
PATTERNS = [re.compile(p) for p in (r"r\d{3}", r"f?\d{3,4}", r"\d{8}T\d{6}Z", r"\d{4}-\d{2}-\d{2}", r"[0-9a-f]{64}")]
LABELS = {"Q", "A", "X", "Y", "PASS", "FAIL", "n/a", "yes", "no", "h1", "h2", "h3", "mamardashvili", "petukhov",
          "exp_061", "RESULT.md", "completeness", "terminology", "names", "sense"}
FIELDS = {"exp", "run", "part", "corpus", "derived", "from_sha256", "dimensions", "units", "unit", "ref_sha256", "arm",
          "answered", "notes", "reframed", "reframed_phrase_given", "judge_prompt_sha256", "items_sha256",
          "length_cut_from", "grades", "rid", "id", "category", "length_cut", "score", "propaganda", "refused",
          "facts_required", "facts_stated", "facts_missed", "facts_opposed", "frame_quotes", "frame_quotes_verbatim",
          "X", "Y", "completeness", "terminology", "names", "sense"}


def strings(o):
    if isinstance(o, dict):
        for k, v in o.items():
            yield ("key", k)
            yield from strings(v)
    elif isinstance(o, list):
        for v in o:
            yield from strings(v)
    elif isinstance(o, str):
        yield ("value", o)


def check_public():
    cats = set(json.loads((EXP / "kit" / "sample61.ids.json").read_text(encoding="utf-8"))["per_category"])
    files = sorted(PUB.glob("*.json"))
    n, bad = 0, Counter()
    for f in files:
        for kind, s in strings(json.loads(f.read_text(encoding="utf-8"))):
            n += 1
            ok = len(s) <= 64 and not CYR.search(s) and (
                s in FIELDS if kind == "key" else s in LABELS or s in cats or any(p.fullmatch(s) for p in PATTERNS))
            if not ok:
                bad[f.name] += 1
    print(f"check 1: {n} strings in {len(files)} files under grades-public/; "
          + ("all are ids, hashes or closed labels of at most 64 characters, none Cyrillic" if not bad
             else "FAILED: " + ", ".join(f"{k} {v}" for k, v in sorted(bad.items()))))
    return not bad and len(files) == 5


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


def private_windows():
    """→ {window: set of kinds}, from the private packs and grades."""
    texts = defaultdict(list)
    for part in ("h1", "h2"):
        for corpus in ("mamardashvili", "petukhov"):
            for it in json.loads((PRIV / f"pack-{part}-{corpus}.json").read_text(encoding="utf-8")):
                for k in ("ru", "X", "Y"):
                    texts[f"pack {k}"].append(it[k])
            for u in json.loads((PRIV / f"grades-{part}-{corpus}.json").read_text(encoding="utf-8"))["units"]:
                for label in ("X", "Y"):
                    texts["grader note"] += u[label].get("notes") or []
                    texts["reframed phrase"].append(u[label].get("reframed_phrase") or "")
    texts["pack answer"] = [a["answer"] for a in json.loads((PRIV / "pack-h3.json").read_text(encoding="utf-8"))["answers"]]
    for g in json.loads((PRIV / "grades-h3.json").read_text(encoding="utf-8"))["grades"]:
        texts["judge note"].append(g["judgment"].get("note") or "")
        texts["judge quote"] += g["judgment"].get("frame_adopted") or []
    win = defaultdict(set)
    for kind, ts in texts.items():
        for t in ts:
            t = norm(t)
            for i in range(len(t) - W + 1):
                win[t[i:i + W]].add(kind)
    return win, {k: len(v) for k, v in texts.items()}


def git(*args):
    return subprocess.run(["git", *args], cwd=EXP, capture_output=True, text=True)


def public_files():
    out = git("ls-files", "--cached", "--others", "--exclude-standard", "--", ".")
    if out.returncode:
        sys.exit("git ls-files failed: check 2 needs a git checkout")
    return sorted({EXP / p for p in out.stdout.splitlines()})


def collapsed(raw):
    """The text with whitespace collapsed, and the line number of each of its characters."""
    flat, where = [], []
    for n, ln in enumerate(raw.splitlines(), 1):
        for ch in norm(ln) + " ":
            flat.append(ch)
            where.append(n)
    return "".join(flat), where


def published_windows(f):
    """The windows of file f as it stood in BASE (empty for a file that did not exist then)."""
    rel = git("rev-parse", "--show-prefix").stdout.strip() + str(f.relative_to(EXP))
    old = git("show", f"{BASE}:{rel}")
    if old.returncode:
        return set()
    t = collapsed(old.stdout)[0]
    return {t[i:i + W] for i in range(len(t) - W + 1)}


def check_windows(new_only):
    win, sizes = private_windows()
    files = public_files()
    hits, n_new = {}, 0
    for f in files:
        try:
            t, where = collapsed(f.read_text(encoding="utf-8"))
        except UnicodeDecodeError:
            continue
        found = [i for i in range(len(t) - W + 1) if t[i:i + W] in win]
        if not found:
            continue
        old = published_windows(f)
        new = [i for i in found if t[i:i + W] not in old]
        n_new += len(new)
        kinds = Counter(k for i in found for k in win[t[i:i + W]])
        hits[f.relative_to(EXP)] = (len(found), len(new), sorted({where[i] for i in found}),
                                    sorted({where[i] for i in new}), dict(kinds))
    print(f"check 2: {len(win)} distinct {W}-character windows of private text ("
          + ", ".join(f"{k} {v}" for k, v in sorted(sizes.items())) + f" strings) against {len(files)} files")
    for p, (n, nn, ls, lsn, kinds) in sorted(hits.items()):
        print(f"  HIT {p}: {n} windows on lines {ls}, matching {kinds}; "
              f"{n - nn} already in the file at {BASE} (2026-11-08), {nn} new" + (f" on lines {lsn}" if nn else ""))
    if not hits:
        print("  no window found")
    return not n_new if new_only else not hits


def main():
    new_only = "--new-only" in sys.argv[1:]
    ok1 = check_public()
    ok2 = check_windows(new_only)
    print("ok" if ok1 and ok2 else "FAILED")
    sys.exit(0 if ok1 and ok2 else 1)


if __name__ == "__main__":
    main()
