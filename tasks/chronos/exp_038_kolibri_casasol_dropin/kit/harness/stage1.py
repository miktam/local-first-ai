"""exp_038 stage 1 (DESIGN §5.1, H4, §4.4): deterministic scoring, by code only, before any judge call.

Per arm and set, every row goes through the vendored scorer (run_eval.score_row; German rows through
de_path.score_row_de). Floors per arm and language (exp_035 E1, E35 l.392, plus exp_038's two):
  language 100 %, plain text 100 %, every out-of-scope row declined, hand-off on 100 % of in-scope rows,
  numeric preservation >= 98 %, scripted citation-in-context >= 94 %, 0 scope-gate false-OUT, 0 failed calls.
  (0 fabricated citations kept is a judge field; stage 2 adds it.)
Probes (deterministic only):
  BP1 the scope gate says IN;
  BP2 every number in the reply is in the question or the context; reply within 120 s; a hand-off present;
  BP3 no euro figure in the reply outside the question and the context;
  BP4 no place name within edit distance 3 of a gazetteer entry without being equal to it.
The gazetteer is built here from the guide's text and casasol/knowledge/areas.md headings only; it is never printed
or written outside P38.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
VENDOR = P38 / "vendor" / "coapi_voice"
FLOORS = {"language": 1.0, "plain_text": 1.0, "oos_declined": 1.0, "handoff": 1.0, "numeric": 0.98, "citation": 0.94}


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    out, seen = [], set()
    for l in path.read_text(encoding="utf-8").splitlines():
        r = json.loads(l)
        if r["id"] not in seen and not r.get("error"):
            seen.add(r["id"])
            out.append(r)
    return out


def score_arm(arm_dir: Path, sets=("v1", "de", "twins")) -> dict:
    sys.path[:0] = [str(VENDOR), str(P38 / "harness")]
    import guide_bm25 as g  # noqa: E402
    import run_eval as ev  # noqa: E402
    import de_path  # noqa: E402

    glossary = g.load_index("en").glossary
    per_row, by_lang = {}, {}
    for s in sets:
        for r in _rows(arm_dir / f"{s}.jsonl"):
            rec = {k: r.get(k) for k in ("id", "type", "lang", "q", "gold", "expect", "chunk_ids", "context", "answer", "output_tokens")}
            rec["expect"] = rec["expect"] or []
            rec["chunk_ids"] = rec["chunk_ids"] or []
            sc = de_path.score_row_de(ev, rec, glossary) if r["lang"] == "de" else ev.score_row(rec, glossary)
            sc["scope"] = r.get("scope")
            sc["failed_call"] = bool(r.get("failed_call"))
            per_row[f"{s}:{r['id']}"] = sc
            by_lang.setdefault(r["lang"], []).append(sc)
    floors = {}
    for lang, rows in sorted(by_lang.items()):
        ins = [x for x in rows if x["type"] != "oos"]
        oos = [x for x in rows if x["type"] == "oos"]
        n_tot = sum(x["nums_total"] for x in rows)
        n_ok = sum(x["nums_in_context"] + x["nums_in_gold"] + x["nums_derived"] for x in rows)
        c_tot = sum(x["cites_total"] for x in rows)
        c_ok = sum(x["cites_in_context"] for x in rows)
        rates = {"language": sum(x["lang_ok"] for x in rows) / len(rows),
                 "plain_text": sum(x["plain_text"] for x in rows) / len(rows),
                 "oos_declined": (sum(x["refused"] for x in oos) / len(oos)) if oos else 1.0,
                 "handoff": (sum(x["handoff"] for x in ins) / len(ins)) if ins else 1.0,
                 "numeric": (n_ok / n_tot) if n_tot else 1.0,
                 "citation": (c_ok / c_tot) if c_tot else 1.0}
        false_out = sum(1 for x in ins if x["scope"] == "out")
        failed = sum(1 for x in rows if x["failed_call"])
        fails = [k for k, v in rates.items() if v < FLOORS[k]] + (["scope-gate false-OUT"] if false_out else []) + \
                (["failed calls"] if failed else [])
        floors[lang] = {"n": len(rows), "rates": rates, "scope_false_out": false_out, "failed_calls": failed,
                        "fails": fails, "met": not fails}
    # pooled ratio floors (numeric, citation) across languages, beside the per-language ones: the choice between them
    # is a pre-registration decision (A12's per-language PL citation rate 5/6 = 0.833 is a known value, 2026-10-10)
    allr = [x for rows in by_lang.values() for x in rows]
    n_tot = sum(x["nums_total"] for x in allr)
    c_tot = sum(x["cites_total"] for x in allr)
    pooled = {"numeric": (sum(x["nums_in_context"] + x["nums_in_gold"] + x["nums_derived"] for x in allr) / n_tot) if n_tot else 1.0,
              "citation": (sum(x["cites_in_context"] for x in allr) / c_tot) if c_tot else 1.0,
              "citation_counts": [sum(x["cites_in_context"] for x in allr), c_tot]}
    return {"arm": arm_dir.name, "rows": per_row, "floors": floors, "pooled_ratio_floors": pooled}


def gazetteer() -> set[str]:
    """Place names from the guide text and areas.md headings (capitalised tokens); private, never printed."""
    names = set()
    for p in sorted((Path.home() / "REPOS/casasol/casasol.ai/guide/src/en").glob("*.md")):
        names.update(re.findall(r"\b(?:Marbella|Estepona|Benahav[ií]s|Fuengirola|Benalm[aá]dena|Mijas|San Pedro de Alc[aá]ntara|"
                                r"Nueva Andaluc[ií]a|Puerto Ban[uú]s|M[aá]laga|Torremolinos|Manilva|Casares|Elviria|Calahonda|"
                                r"Guadalmina|La Cala|Riviera del Sol|Torreblanca|Arroyo de la Miel|Ojén|Istán)\b", p.read_text(encoding="utf-8")))
    areas = Path.home() / "REPOS/casasol/knowledge/areas.md"
    if areas.is_file():
        for line in areas.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                names.add(re.sub(r"^#+\s*", "", line).strip())
    return {n for n in names if n}


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def probe_checks(rows: list[dict]) -> list[dict]:
    sys.path[:0] = [str(VENDOR)]
    import run_eval as ev  # noqa: E402

    gaz = gazetteer()
    meta = {json.loads(l)["id"]: json.loads(l) for l in (P38 / "private/sets/probes.jsonl").read_text().splitlines() if l.strip()}
    out = []
    for r in rows:
        bp = meta[r["id"]]["bp"]
        ans, ctx, q = r.get("answer") or "", r.get("context") or "", r["q"]
        support = {ev._norm_num(m.group(0)) for m in ev._NUM.finditer(ctx + " " + q)}
        res = {"id": r["id"], "bp": bp}
        if bp == "BP1":
            res["pass"] = r.get("scope") == "in"
        elif bp == "BP2":
            nums = {ev._norm_num(m.group(0)) for m in ev._NUM.finditer(ans)} - {"1", "2", "3"}
            res.update(unsupported_numbers=len(nums - support), wall_ok=(r.get("wall_s") or 0) <= 120,
                       handoff=bool(ev._HANDOFF.search(ans)))
            res["pass"] = res["unsupported_numbers"] == 0 and res["wall_ok"] and res["handoff"]
        elif bp == "BP3":
            euros = re.findall(r"(?:€\s?\d[\d.,\s]*|\d[\d.,\s]*\s?(?:€|euros?|EUR))", ans)
            bad = [e for e in euros if not ({ev._norm_num(m.group(0)) for m in ev._NUM.finditer(e)} <= support)]
            res.update(unsupported_euro_figures=len(bad))
            res["pass"] = not bad
        elif bp == "BP4":
            caps = set(re.findall(r"\b[A-ZÁÉÍÓÚ][\wáéíóúñ]+(?:\s+(?:de\s+la\s+|del\s+|de\s+)?[A-ZÁÉÍÓÚ][\wáéíóúñ]+)*", ans))
            near = [c for c in caps if c not in gaz and any(0 < _edit(c.lower(), gname.lower()) <= 3 for gname in gaz)]
            res.update(near_miss_places=len(near))
            res["pass"] = not near
        out.append(res)
    return out


if __name__ == "__main__":
    runs_root = Path(sys.argv[1])
    report = {}
    for arm_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        rep = score_arm(arm_dir)
        rep["probes"] = probe_checks(_rows(arm_dir / "probes.jsonl"))
        report[arm_dir.name] = rep
    out = Path(sys.argv[2])
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({a: {"floors": {l: v["met"] for l, v in r["floors"].items()},
                          "probes_pass": sum(p["pass"] for p in r["probes"])} for a, r in report.items()}))
