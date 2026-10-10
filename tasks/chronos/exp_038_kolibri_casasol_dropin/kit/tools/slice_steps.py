"""exp_038 DE slice: the mechanical steps between the Fable instances (DESIGN §4.3). Code only.

merge <draft> <review> <out>    Apply the reviewer's "fix" values field by field (exact replacement), keep "ok" rows,
                                refuse while any row is "replace" (a replacement row needs a new writer pass).
flags-input <slice> <out>       For every in-scope row: its gold and the context the glue retrieves for its English
                                twin (the EN index, k = glue.K, budget = glue.BUDGET), with the M0-flagged lines
                                replaced by "[withheld]". The flagger sees nothing else (no question, no twin).
finalise <slice> <flags>        Write the run files: de_slice.jsonl (q German, route_q = q_en) and de_twins.jsonl
                                (q = q_en, lang en), each row carrying substance_in_context, plus sha256 manifest.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
VENDOR = P38 / "vendor" / "coapi_voice"
FIX_FIELDS = {"gold", "q", "q_en", "key_rule", "expect", "hand_off", "numbers", "citations", "guide_evidence", "arithmetic", "type"}


def jl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jl(path: Path, rows: list[dict]) -> str:
    data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    path.write_text(data, encoding="utf-8")
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def merge(draft: Path, review: Path, out: Path) -> int:
    rows = {r["id"]: r for r in jl(draft)}
    rv = json.loads(review.read_text(encoding="utf-8"))
    repl = [x["id"] for x in rv["rows"] if x["verdict"] == "replace"]
    if repl:
        print(json.dumps({"refused": "rows marked replace need a new writer pass", "rows": repl}))
        return 2
    log = []
    for x in rv["rows"]:
        if x["verdict"] != "fix":
            continue
        bad = set(x.get("fixes", {})) - FIX_FIELDS
        if bad:
            raise SystemExit(f"{x['id']}: unknown fix fields {bad}")
        for k, v in x["fixes"].items():
            log.append({"id": x["id"], "field": k})
            rows[x["id"]][k] = v
    sha = write_jl(out, [rows[r["id"]] for r in jl(draft)])
    print(json.dumps({"out": out.name, "sha256": sha, "fixed_fields": len(log),
                      "fixed_rows": len({e["id"] for e in log})}))
    return 0


def flags_input(slice_path: Path, out: Path) -> None:
    sys.path[:0] = [str(VENDOR), str(P38 / "harness")]
    import glue as G  # noqa: E402
    import guide_bm25 as g  # noqa: E402
    from judge_kit import scrub_context, withheld_lines  # noqa: E402

    idx = g.load_index("en")
    flagged = withheld_lines()
    items = []
    for r in jl(slice_path):
        if r["type"] == "oos":
            continue
        chunks = idx.retrieve(r["q_en"], k=G.K, budget=G.BUDGET)
        ctx = scrub_context(g.context_block(chunks), flagged)
        items.append({"id": r["id"], "gold": r["gold"], "context": ctx, "chunk_ids": [c["id"] for c in chunks]})
    out.write_text(json.dumps({"n": len(items), "items": items}, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"out": out.name, "n": len(items)}))


def finalise(slice_path: Path, flags_path: Path) -> None:
    rows = jl(slice_path)
    flags = {x["id"]: bool(x["substance_in_context"]) for x in json.loads(flags_path.read_text())["rows"]}
    de, tw = [], []
    for r in rows:
        f = None if r["type"] == "oos" else flags[r["id"]]
        common = {k: r[k] for k in ("module", "type", "expect", "gold", "hand_off", "key_rule", "numbers", "citations")}
        de.append({"id": r["id"], "lang": "de", "q": r["q"], "route_q": r["q_en"], "twin_id": r["twin_id"],
                   "substance_in_context": f, **common})
        tw.append({"id": r["twin_id"], "lang": "en", "q": r["q_en"], "twin_of": r["id"], "substance_in_context": f, **common})
    sets = P38 / "private/sets"
    m = {"de_slice.jsonl": write_jl(sets / "de_slice.jsonl", de), "de_twins.jsonl": write_jl(sets / "de_twins.jsonl", tw),
         "source": slice_path.name, "source_sha256": hashlib.sha256(slice_path.read_bytes()).hexdigest()}
    (sets / "DE_MANIFEST.json").write_text(json.dumps(m, indent=1) + "\n")
    print(json.dumps(m))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "merge":
        sys.exit(merge(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])))
    if c == "flags-input":
        flags_input(Path(sys.argv[2]), Path(sys.argv[3]))
    elif c == "finalise":
        finalise(Path(sys.argv[2]), Path(sys.argv[3]))
