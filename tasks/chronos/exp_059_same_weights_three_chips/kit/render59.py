#!/usr/bin/env python3
"""exp_059: render the 12 triage prompts per enquiry exactly as llama.cpp b11711 renders them for a `kev` decision model.

The phone cannot run `llama-server`, so it is handed the finished prompt strings and does only the model mathematics.
This script is the server's `render()` for the kev type (tools/server/server-decision.cpp at b11711) applied to the
frozen rev 4 knife's masked state and LEAD_SCHEMA, with the model's own `systemone` template:

  <|fim_prefix|>{{ state }}<|fim_middle|>{{ instructions }}
  {% for o in options %}<|box_start|>{{ o.key }}{% if o.description %}: {{ o.description }}{% endif %}<|box_end|>{% endfor %}
  <|fim_suffix|>

(for `noul` the option text is `no` / `yes`, options in the order false, true; the template is one line, shown here
wrapped). State, instructions, keys and descriptions pass through the server's special-token escape: `<|name|>` becomes
`<¦name¦>` (U+00A6), so text from an enquiry can never be parsed as a control token.

    python3 render59.py [--test PATH] [--out DIR]

Writes <out>/prompts59.jsonl (one line per enquiry: id, lang, questions[qid, type, keys, prompt]), prompts59.sha256 and
prompts59.meta.json (counts, the template, the temperature the model declares, the test-set hash).
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
EXP044 = EXP.parent / "exp_044_kev_lead_triage_rev4"
# The input is exp_044's PUBLIC copy of the sealed test set. The as-run set (bf54b4d0…, data/test.sha256) had its
# phone numbers moved to fiction ranges after the run (local-first-ai f1ad329, 9d87aa7); nothing else differs, and
# masked() turns every phone number into "[phone]" before the model sees it. Every arm of exp_059 reads this copy.
TEST_SHA = "514b6386dfba45464089a472d579097df58c3380964c1203aab9a39deddae808"
TEST_SHA_AS_RUN = "bf54b4d0cb017f266fa094afdd40802602878d0b259992b9c8cf6b4bba3ccd0b"
KNIFE_SHA = "2cdfbdf794ff50b52bffa88230645988b28da599783c19ba9e6a3983a43020f1"   # exp_044's public copy of rev 4

# read from Kev-4B-Q8_0.gguf (7c2ebed9…) on 2026-10-28: qwen35.decision.temperature.{choice,score,noul}
TEMPERATURE = 2.406049966812134
TEMPLATE = ("<|fim_prefix|>{{ state }}<|fim_middle|>{{ instructions }}{% for o in options %}<|box_start|>"
            "{% if type == 'score' %}{% if o.description %}{{ o.description }}{% endif %}{% else %}"
            "{% if type != 'noul' %}{{ o.key }}{% elif o.key == 'true' %}yes{% else %}no{% endif %}"
            "{% if o.description %}: {{ o.description }}{% endif %}{% endif %}<|box_end|>{% endfor %}<|fim_suffix|>")

_SPECIAL = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def knife():
    path = HERE / "kev_knife_rev4.py"
    if sha256(path) != KNIFE_SHA:
        sys.exit(f"REFUSED: {path.name} is not the frozen rev 4 copy ({KNIFE_SHA[:12]})")
    spec = importlib.util.spec_from_file_location("kev_knife_rev4", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def kev_text(val) -> str:
    """server-decision.cpp: decision_kev_render (a string is itself) then server_decision_escape_special_tokens."""
    if val is None:
        return ""
    if isinstance(val, bool):
        return "True" if val else "False"
    if not isinstance(val, str):
        raise TypeError("exp_059 only renders string states, instructions and descriptions")
    return _SPECIAL.sub("<¦\\1¦>", val)


def options_of(q: dict) -> list[tuple[str, str | None]]:
    """parse_questions() for kev: choice keeps the criteria order (common_json keeps insertion order);
    noul is [false, true] with the descriptions under those keys, if any."""
    if q["type"] == "choice":
        return list(q["criteria"].items())
    if q["type"] == "noul":
        crit = q.get("criteria") or {}
        return [("false", crit.get("false")), ("true", crit.get("true"))]
    raise ValueError("exp_059 renders choice and noul questions only")


def render(state: str, qtype: str, instructions: str, options: list[tuple[str, str | None]]) -> str:
    out = "<|fim_prefix|>" + kev_text(state) + "<|fim_middle|>" + kev_text(instructions)
    for key, desc in options:
        out += "<|box_start|>"
        out += ("yes" if key == "true" else "no") if qtype == "noul" else kev_text(key)
        if desc:                                   # jinja truthiness: null and "" print nothing
            out += ": " + kev_text(desc)
        out += "<|box_end|>"
    return out + "<|fim_suffix|>"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", type=Path, default=EXP044 / "data" / "test.jsonl")
    ap.add_argument("--out", type=Path, default=EXP / "data")
    a = ap.parse_args()

    if sha256(a.test) != TEST_SHA:
        sys.exit(f"REFUSED: {a.test} is not the sealed exp_044 test set ({TEST_SHA[:12]})")
    K = knife()
    rows = [json.loads(l) for l in a.test.read_text(encoding="utf-8").splitlines() if l.strip()]

    a.out.mkdir(parents=True, exist_ok=True)
    out = a.out / "prompts59.jsonl"
    n_q = 0
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            state = K.masked(r["text"])
            qs = []
            for qid, q in K.LEAD_SCHEMA.items():
                opts = options_of(q)
                qs.append({"qid": qid, "type": q["type"], "keys": [k for k, _ in opts],
                           "prompt": render(state, q["type"], q["instructions"], opts)})
            n_q += len(qs)
            f.write(json.dumps({"id": r["id"], "lang": r["lang"], "questions": qs}, ensure_ascii=False) + "\n")
    h = sha256(out)
    (a.out / "prompts59.sha256").write_text(h + "\n")
    (a.out / "prompts59.meta.json").write_text(json.dumps({
        "enquiries": len(rows), "questions": n_q, "questions_per_enquiry": len(K.LEAD_SCHEMA),
        "test_sha256": TEST_SHA, "test_sha256_as_run": TEST_SHA_AS_RUN, "knife_sha256": KNIFE_SHA, "prompts_sha256": h,
        "temperature": TEMPERATURE, "temperature_source": "qwen35.decision.temperature.* in Kev-4B-Q8_0.gguf 7c2ebed9",
        "marker": "<|box_end|>", "template": TEMPLATE, "engine": "llama.cpp b11711 (9d1e036), tools/server/server-decision.cpp kev type",
    }, indent=1, ensure_ascii=False) + "\n")
    print(f"{out}: {len(rows)} enquiries, {n_q} prompts, sha256 {h[:12]}")


if __name__ == "__main__":
    main()
