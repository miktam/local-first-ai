"""exp_038 seal (DESIGN §4.3 step 5). Run by Andrei, after he has read the sets. Code only.

  python3 tools/seal.py --prepare   applies the German language check's rewrites (q only) to the reviewed slice,
                                    re-runs the gates, merges the substance flags, writes the run files
                                    (de_slice.jsonl, de_twins.jsonl) and SEAL_PREVIEW.json. Seals nothing.
  python3 tools/seal.py --seal --by Andrei
                                    refuses on any gate failure, any key-rule gate fail, any blindness audit that is
                                    neither CLEAN nor reviewed in blindness/REVIEWS.md, or a run file that changed
                                    since --prepare; then writes private/sets/SEAL.json: sha256 of every sealed file,
                                    the provenance chain, the audits, the gate outputs, and the seal line with UTC.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
SETS, W4, BL = P38 / "private/sets", P38 / "private/w4", P38 / "blindness"
CHAIN = [
    ("strings author (Fable)", "w3_strings_author"), ("German editor (Fable)", "w3_german_editor"),
    ("v1 checker (Fable)", "w4_v1_checker"), ("slice writer (Fable)", "w4_slice_writer"),
    ("dev-set writer (Fable)", "w4_dev_writer"), ("probe writer (Fable)", "w4_probe_writer"),
    ("adversarial reviewer (Fable)", "w4_reviewer"), ("substance flagger (Fable)", "w4_flagger"),
    ("German language checker (Fable)", "w4_native_checker"), ("key-rule checker (Fable)", "w4_keyrule_checker"),
    ("replacement writer (Fable)", "w4_replacement_writer"), ("replacement reviewer (Fable)", "w4_repl_reviewer"),
    ("replacement flagger (Fable)", "w4_repl_flagger"), ("replacement language checker (Fable)", "w4_repl_native"),
    ("replacement key-rule checker (Fable)", "w4_repl_keyrule"),
]


def merge_final() -> None:
    """The two replacement rows' checks replace the failed rows' entries: *_final.json files."""
    for base in ("substance_flags", "native_check", "keyrule_gate"):
        main = json.loads((W4 / f"{base}.json").read_text())
        repl = {x["id"]: x for x in json.loads((W4 / f"{base}_repl.json").read_text())["rows"]}
        main["rows"] = [repl.get(x["id"], x) for x in main["rows"]]
        main["replaced"] = sorted(repl)
        if base == "keyrule_gate":
            main["summary"] = {"pass": sum(x["key_rule_gate"] == "pass" for x in main["rows"]),
                               "fail": sum(x["key_rule_gate"] != "pass" for x in main["rows"]),
                               "fresh_a10": sum(bool(x.get("fresh_a10")) for x in main["rows"])}
        (W4 / f"{base}_final.json").write_text(json.dumps(main, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
SEALED = ["de_slice.jsonl", "de_twins.jsonl", "probes.jsonl", "de_dev.jsonl", "grid_de.jsonl"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def prepare() -> int:
    merge_final()
    rows = jl(SETS / "de_slice_reviewed3.jsonl")
    nat = {x["id"]: x for x in json.loads((W4 / "native_check_final.json").read_text())["rows"]}
    for r in rows:
        n = nat.get(r["id"])
        if n and n["verdict"] == "rewrite":
            r["q"] = n["q"]
            r["native_rewritten"] = True
    src = SETS / "de_slice_final_src.jsonl"
    src.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    g = subprocess.run([sys.executable, str(P38 / "tools/de_gates.py"), "check", str(src)], capture_output=True, text=True)
    if g.returncode != 0:
        print(g.stdout[-2000:])
        return 1
    subprocess.run([sys.executable, str(P38 / "tools/slice_steps.py"), "finalise", str(src), str(W4 / "substance_flags_final.json")], check=True)
    preview = {"files": {f: sha(SETS / f) for f in SEALED}, "gates": json.loads((SETS / "de_slice_final_src.gates.json").read_text())["n_failed_rows"],
               "native_rewrites": sum(1 for r in rows if r.get("native_rewritten"))}
    (SETS / "SEAL_PREVIEW.json").write_text(json.dumps(preview, indent=1) + "\n")
    print(json.dumps(preview))
    return 0


def seal(by: str, write: bool = True) -> int:
    problems = []
    preview = json.loads((SETS / "SEAL_PREVIEW.json").read_text())
    for f, h in preview["files"].items():
        if sha(SETS / f) != h:
            problems.append(f"{f} changed since --prepare")
    if preview["gates"] != 0:
        problems.append("gates failed")
    kr = json.loads((W4 / "keyrule_gate_final.json").read_text())
    fails = [x["id"] for x in kr["rows"] if x["key_rule_gate"] != "pass"]
    if fails:
        problems.append(f"key-rule gate fails: {fails}")
    reviews = (BL / "REVIEWS.md").read_text() if (BL / "REVIEWS.md").is_file() else ""
    audits = {}
    for role, label in CHAIN:
        f = BL / f"{label}.json"
        if not f.is_file():
            problems.append(f"no blindness audit for {role}")
            continue
        a = json.loads(f.read_text())
        audits[label] = {"role": role, "verdict": a["verdict"], "reads": a["reads"], "flags": a["flags"]}
        if a["verdict"] != "CLEAN" and f"| {label} |" not in reviews:
            problems.append(f"{label} audit {a['verdict']} without a recorded review")
    if problems:
        print(json.dumps({"refused": problems}, indent=1))
        return 1
    if not write:
        print(json.dumps({"check": "all seal conditions hold", "files": len(preview["files"]), "audits": len(audits)}))
        return 0
    rec = {"sealed_by": by, "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "files": {f: sha(SETS / f) for f in SEALED},
           "provenance": {"design": "workspace tasks/exp038/DESIGN.md rev 2", "chain": [r for r, _ in CHAIN],
                          "review": {"file": "private/w4/de_slice_review.json", "sha256": sha(W4 / "de_slice_review.json"), "replacement_review_sha256": sha(W4 / "repl_review.json")},
                          "substance_flags": {"file": "private/w4/substance_flags_final.json", "sha256": sha(W4 / "substance_flags_final.json")},
                          "native_check": {"file": "private/w4/native_check_final.json", "sha256": sha(W4 / "native_check_final.json"),
                                           "limitation": "LLM-only German check (decision 6 fallback)"},
                          "keyrule_gate": {"file": "private/w4/keyrule_gate_final.json", "sha256": sha(W4 / "keyrule_gate_final.json"),
                                           "fresh_a10": kr["summary"].get("fresh_a10")},
                          "m0_exclusions": sha(P38 / "rights/m0_exclusions.json"),
                          "r4_tree": sorted((P38 / "runs").glob("r4_*.json"))[-1].name},
           "audits": audits}
    (SETS / "SEAL.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"sealed": rec["files"], "utc": rec["utc"], "by": by}, indent=1))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--seal", action="store_true")
    ap.add_argument("--by", default=None)
    ap.add_argument("--check", action="store_true", help="run every seal check, write nothing")
    a = ap.parse_args()
    if a.prepare:
        sys.exit(prepare())
    if a.check:
        sys.exit(seal("check", write=False))
    if a.seal and a.by:
        sys.exit(seal(a.by))
    sys.exit("use --prepare, or --seal --by <name>")
