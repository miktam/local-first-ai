#!/usr/bin/env python3
"""exp_055: assemble the writers' records into messages, a seeded lookup cache and a manifest.

Ground truth is by construction: a lure is a lure because it was written as one, a genuine message because it was
written as one. No model filters the set, so no arm is scored on messages chosen by its own family. Checks here
are by code only: fields, ids, footprint items, domains listed, fictional phone range, no real-looking secrets.

    python3.14 kit/build55.py data
"""
import json
import random
import re
import sys
from pathlib import Path

FIELDS = {"id", "kind", "channel", "to_item", "used_items", "tactic", "ask", "from_name", "from_address", "subject",
          "body", "domains"}
OWN = {"vennholt-partners.example", "kestrelmoor.example", "profiles.ringfold.example", "ringfold.example"}
TO = {"C-526e13": "iris@vennholt-partners.example", "C-04c986": "hello@kestrelmoor.example",
      "C-5415d0": "+447700900142"}
DATE = "Sun, 25 Oct 2026 09:00:00 +0000"
TLDS = "com|net|org|co|io|info|online|events|uk|eu|site|app|dev|biz|xyz|top|me|ai|cloud|email|live|news|tech|today"
URL = re.compile(r"(?:https?://)?((?:[a-z0-9-]+\.)+(?:" + TLDS + r"))(?![a-z0-9-])(?:[/:?#][^\s<>\"')]*)?", re.I)
EMAIL = re.compile(r"[\w.+-]+@((?:[a-z0-9-]+\.)+[a-z]{2,})", re.I)
PHONE = re.compile(r"(?:\+\d{1,3}[\s.-]?\(?0?\)?[\s.-]?\d{2,4}|\b0\d{3,4})[\s.-]?\d{3}[\s.-]?\d{3,4}\b")


def registrable(host: str) -> str:
    parts = host.lower().strip(".").split(".")
    two = {"co.uk", "org.uk", "ac.uk", "gov.uk", "com.au"}
    return ".".join(parts[-3:]) if ".".join(parts[-2:]) in two else ".".join(parts[-2:])


def domains_in(text: str) -> set:
    found = {registrable(m.group(1)) for m in URL.finditer(text) if "@" not in text[max(0, m.start() - 1):m.start()]
             and not re.match(r"^[a-z]+\.[a-z]+$", m.group(1), re.I) or "/" in m.group(0)} | \
            {registrable(m.group(1)) for m in EMAIL.finditer(text)}
    return {d for d in found if "." in d and not d.replace(".", "").isdigit()}


def main():
    root = Path(sys.argv[1])
    items = {ln.split(" | ")[0] for ln in (root / "FOOTPRINT.txt").read_text().splitlines()[1:] if " | " in ln}
    recs, problems = [], []
    for f in sorted((root / "raw").glob("*.jsonl")):
        for n, line in enumerate(open(f, encoding="utf-8"), 1):
            if not line.strip():
                continue
            r = json.loads(line)
            if set(r) != FIELDS:
                problems.append(f"{f.name}:{n} fields {sorted(set(r) ^ FIELDS)}")
            recs.append(r)
    ids = [r["id"] for r in recs]
    problems += [f"duplicate id {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    cache, conflicts, manifest = {}, [], {}
    msgdir = root / "messages"
    msgdir.mkdir(exist_ok=True)
    order = sorted(recs, key=lambda r: r["id"])
    random.Random(55).shuffle(order)                     # opaque ids: the writers' ids (LA.., BA..) name the kind
    opaque = {r["id"]: f"X{n:03d}" for n, r in enumerate(order, 1)}
    for r in recs:
        rid = opaque[r["id"]]
        bad = [i for i in r["used_items"] if i not in items]
        if bad:
            problems.append(f"{r['id']} unknown items {bad}")
        if r["to_item"] not in TO:
            problems.append(f"{r['id']} to_item {r['to_item']}")
        text = " ".join([r["from_address"], r["subject"] or "", r["body"]])
        listed = {registrable(d["domain"]) for d in r["domains"]}
        missing = sorted(domains_in(text) - listed - OWN - {registrable(o) for o in OWN})
        if missing:
            problems.append(f"{r['id']} domains not listed {missing}")
        for p in PHONE.findall(r["body"] + " " + r["from_address"]):
            digits = re.sub(r"\D", "", p)
            if len(digits) >= 10 and not digits.startswith("447700900"):
                problems.append(f"{r['id']} phone outside the drama range: {p}")
        for d in r["domains"]:
            dom = registrable(d["domain"])
            entry = {"abuse_email": None, "abuse_tel": None, "created": d["created"] + "T00:00:00Z", "domain": dom,
                     "error": None, "fetched_at": "2026-10-25T00:00:00+00:00", "nameservers": [],
                     "registrar": "Seeded registrar (exp_055 fixture)", "registrar_iana_id": None,
                     "source": "rdap:seeded.invalid", "status": [], "v": 2}
            key = "age:" + dom
            if key in cache and cache[key]["created"] != entry["created"]:
                conflicts.append(f"{dom}: {cache[key]['created'][:10]} vs {d['created']} ({r['id']})")
            cache.setdefault(key, entry)
        if r["channel"] == "email":
            frm = f"{r['from_name']} <{r['from_address']}>"
            raw = (f"From: {frm}\nTo: Iris Kestrelmoor <{TO[r['to_item']]}>\nSubject: {r['subject']}\nDate: {DATE}\n"
                   f"Message-ID: <{rid.lower()}@corpus.invalid>\nMIME-Version: 1.0\n"
                   f"Content-Type: text/plain; charset=utf-8\n\n{r['body']}\n")
            name = f"{rid}.eml"
        else:
            raw = r["body"] + "\n"
            name = f"{rid}.txt"
        (msgdir / name).write_text(raw, encoding="utf-8")
        manifest[rid] = {"file": name, "writer_id": r["id"], "kind": r["kind"], "channel": r["channel"], "to_item": r["to_item"],
                         "used_items": r["used_items"], "tactic": r["tactic"], "ask": r["ask"],
                         "from": r["from_address"],
                         "domains": {registrable(d["domain"]): cache["age:" + registrable(d["domain"])]["created"][:10]
                                     for d in r["domains"]}}       # one domain, one age: the first record's
    (root / "cache").mkdir(exist_ok=True)
    (root / "cache/lookups.json").write_text(json.dumps(cache, indent=1, sort_keys=True) + "\n")
    (root / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    kinds = {k: sum(1 for m in manifest.values() if m["kind"] == k) for k in ("lure", "benign")}
    print(f"{len(manifest)} messages {kinds}; {len(cache)} seeded domains; {len(conflicts)} date conflicts; "
          f"{len(problems)} problems")
    for p in problems + conflicts:
        print("  ", p)


if __name__ == "__main__":
    main()
