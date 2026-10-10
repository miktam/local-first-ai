"""exp_038 writer inputs (DESIGN §4.3, §8.0), made by code; no agent reads the flagged lines.

1. private/writer_inputs/s1_en/study_guide_modulo{1..6}.md: S1 (EN guide) at its M0-time hashes, re-verified,
   with every M0-flagged line replaced by "[withheld]". Writers, reviewers and flaggers read only this copy.
2. private/w3/glue_strings.json: the vendored glue's user-facing strings and the regex sources the German path
   extends (glue.py, run_eval.py), for the W3 strings author. No guide text, no eval rows.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
CASASOL = Path.home() / "REPOS/casasol"
VENDOR = P38 / "vendor" / "coapi_voice"


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def writer_s1() -> dict:
    ex = json.loads((P38 / "rights/m0_exclusions.json").read_text(encoding="utf-8"))
    m0 = json.loads((P38 / "runs" / ex["source"]).read_text(encoding="utf-8"))
    out = P38 / "private/writer_inputs/s1_en"
    out.mkdir(parents=True, exist_ok=True)
    rec = {"rule": "M0-flagged lines replaced by [withheld]; S1 itself unchanged", "files": {}}
    for m in range(1, 7):
        rel = f"casasol.ai/guide/src/en/study_guide_modulo{m}.md"
        raw = (CASASOL / rel).read_bytes()
        if sha(raw) != m0["guide_sha256"][rel]:
            sys.exit(f"REFUSED: {rel} changed since M0 ({sha(raw)[:12]} != {m0['guide_sha256'][rel][:12]})")
        lines = raw.decode("utf-8").split("\n")
        n = 0
        for a, b in ex["lines"].get(rel, []):
            for i in range(a, b + 1):
                lines[i - 1] = "[withheld]"
                n += 1
        data = "\n".join(lines).encode("utf-8")
        (out / f"study_guide_modulo{m}.md").write_bytes(data)
        rec["files"][f"study_guide_modulo{m}.md"] = {"source_sha256": sha(raw), "withheld_lines": n, "sha256": sha(data)}
    (out / "MANIFEST.json").write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    return rec


def glue_strings() -> dict:
    sys.path.insert(0, str(VENDOR))
    import glue as G  # noqa: E402
    import run_eval as ev  # noqa: E402

    names = ("SYSTEM", "ADVICE_SUFFIX", "ADVICE_LEAD", "ADVICE_LEAD_DRAFT", "ADVICE_LEAD_PRICE",
             "DECLINE", "HANDOFF", "HANDOFF_SOFT", "QUESTION_LABEL", "CONTEXT_LABEL", "LANGNAME")
    strings = {k: getattr(G, k) for k in names}
    regex = {k: getattr(G, k).pattern for k in ("_ROUTES", "_CONSEQUENCE", "_LAW_TOKEN", "_ADVICE", "_DRAFTING", "_PRICING", "_DECISION")}
    regex.update({f"run_eval.{k}": getattr(ev, k).pattern for k in ("_REFUSAL", "_HANDOFF", "_CITE")})
    out = {"source": "P38 vendor/coapi_voice (casasol b1a648c)", "strings": strings, "regex_sources": regex,
           "seams": getattr(G, "_SEAMS", None), "stop_words": {k: sorted(v) for k, v in ev._STOP.items()}}
    p = P38 / "private/w3/glue_strings.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return {"path": str(p.relative_to(P38)), "sha256": sha(p.read_bytes()), "n_strings": len(strings), "n_regex": len(regex)}


if __name__ == "__main__":
    w = writer_s1()
    print(json.dumps({"withheld_lines": {k: v["withheld_lines"] for k, v in w["files"].items()},
                      "glue_strings": glue_strings()}, indent=1))
