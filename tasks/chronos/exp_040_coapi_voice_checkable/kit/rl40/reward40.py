#!/usr/bin/env python3
"""
exp_040 — the correctness-checkable reward (DESIGN §6, BUILD §2). Reward rev 3. No model call of any kind.

exp_039's reward paid for form; this one pays mainly for the answer. Every prompt carries a label that
code can check against the completion's final line ("Answer: …" / "Odpowiedź: …" / "Respuesta: …").

  component  check                                                              value
  answer     final line names exactly the label                                 +3
             final line names a wrong candidate, or two or more (a hedge)       −2
             no final line, or none of the candidates found on it               −1
  lang       answer language == question language                               +0.5 / −0.5
  plain      no markdown                                                        +0.25 / −0.25
  cite       ≥1 citation and all of them in the context                         +0.25 (capped)
             −0.5 per citation not in the context, floor −1; no citation: 0
  num        every number in context ∪ question ∪ gold, or derivable from them  0; −0.5 per unsupported, floor −1
  length     at the token cap                                                   −1 / 0
  repeat     any word 8-gram three or more times                                −1 / 0
  copy       longest common substring with the context > COPY_T chars           −1 / 0

The answer term spans −2…+3; all form terms together span about −4…+1, so a well-cited wrong answer
always scores below a plain right one (DESIGN §6).

Matching rules (fixed here; a change is a re-registration):
  • the final line is the rest of the line after the LAST Answer:/Odpowiedź:/Respuesta: key (case-insensitive,
    Odpowiedz accepted) that opens a line or follows a sentence end (". Answer: …" — Qwen3-4B under the glue's
    "plain text, 80–200 words" writes one paragraph and puts the answer last; smoke run 2026-10-15). A key
    inside a sentence ("the answer: …") is not a final line;
  • categorical / yes-no / abstain: every candidate's forms in all three languages, plus the candidate's
    key, are searched as whole words (case- and accent-folded). A span inside a longer span of another
    candidate is dropped ("built area" inside "built area with common elements"; "no" inside "no está en
    el material"). ABSTAIN ("not in the material" and its PL/ES forms) is a candidate on every prompt;
  • numeric: the numbers on the final line (exp_039's pattern, sign-aware, EN and ES/PL separators). One
    number within the item's relative tolerance → right; one wrong number → wrong; two or more different
    numbers, or a number together with ABSTAIN → hedge (−2); ABSTAIN alone → wrong unless the label is ABSTAIN;
  • abstain rows (label ABSTAIN, no candidates): a final line that abstains → +3; any other final line → wrong (−2),
    so guessing on an abstention row costs what abstaining on an answerable row costs (filter run 3, 2026-10-16:
    708 such answers had scored "missing", −1);
  • the reward never sees the explanation's correctness — H2 (judged prose) and H5 (audit) guard that.

Dataset contract: `answer` = encode_meta40(...) JSON string (mlx-lm-lora passes str(item["answer"])).
"""
from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV = HERE.parent
sys.path[:0] = [str(CV), str(CV / "rl")]
import reward as R      # noqa: E402  exp_039's helpers (number pattern, tokenizer, LCS, n-grams, COPY_T)
import run_eval as rv   # noqa: E402

try:
    from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function
except ImportError:
    def register_reward_function(name=None):
        return lambda f: f

REWARD_REV = 3
COMPONENTS = ("answer", "lang", "plain", "cite", "num", "length", "repeat", "copy")
KINDS = {"categorical", "yesno", "numeric", "abstain", "arith"}
LANGS = {"en", "pl", "es"}
ABSTAIN = "ABSTAIN"
ABSTAIN_FORMS = ["not in the material", "not covered by the material", "the material does not say",
                 "brak w materiale", "nie ma w materiale", "materiał tego nie obejmuje",
                 "no está en el material", "no figura en el material", "el material no lo dice"]
ANSWER_KEY = {"en": "Answer", "pl": "Odpowiedź", "es": "Respuesta"}
ABSTAIN_LINE = {"en": "not in the material", "pl": "brak w materiale", "es": "no está en el material"}
FINAL_LINE = {   # appended to glue.SYSTEM[lang] for every exp_040 arm (train = serve; A13 = A12 + this sentence)
    "en": ' End with one final line that gives only the answer, in the form "Answer: …": a single number with its unit, '
          'or the short name of the rule, tax, deadline, party or option. If the material does not contain the answer, '
          'the final line is "Answer: not in the material".',
    "pl": ' Zakończ jedną ostatnią linią, która podaje wyłącznie odpowiedź, w formie "Odpowiedź: …": jedną liczbę z jednostką '
          'albo krótką nazwę przepisu, podatku, terminu, strony lub opcji. Jeśli materiał nie zawiera odpowiedzi, ostatnia '
          'linia brzmi "Odpowiedź: brak w materiale".',
    "es": ' Termina con una última línea que dé solo la respuesta, con la forma "Respuesta: …": un único número con su unidad, '
          'o el nombre breve de la norma, impuesto, plazo, parte u opción. Si el material no contiene la respuesta, la última '
          'línea es "Respuesta: no está en el material".',
}
CAP, CAP_SLACK = R.CAP, R.CAP_SLACK
_KEY = re.compile(r"(?<!\w)(?:answer|respuesta|odpowied[zź])[ \t]*[*_`]*[ \t]*[:：]", re.I)
_OPENS = ".!?;)]»”\"\n"   # what may precede the key (after markdown and spaces): a line start or a sentence end


def fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s.casefold()) if not unicodedata.combining(c))


# ---------------------------------------------------------------- metadata
def encode_meta40(*, lang: str, kind: str, q: str, context: str, label: str | None = None,
                  candidates: dict | None = None, value: float | None = None, tol: float = 0.005,
                  item: str | None = None) -> str:
    """The `answer` field of an exp_040 GRPO row. `candidates` = {name: {"en": [...], "pl": [...], "es": [...]}}."""
    flat = None
    if candidates is not None:
        flat = {k: sorted({f for fs in v.values() for f in fs} | {k}) for k, v in candidates.items()}
    meta = {"v": 40, "lang": lang, "kind": kind, "q": q, "context": context, "label": label,
            "candidates": flat, "value": value, "tol": tol, "item": item}
    _check(meta)
    return json.dumps(meta, ensure_ascii=False, sort_keys=True)


def _check(m: dict) -> dict:
    if m.get("v") != 40 or m.get("lang") not in LANGS or m.get("kind") not in KINDS:
        raise ValueError(f"bad exp_040 meta: v/lang/kind = {m.get('v')}/{m.get('lang')}/{m.get('kind')}")
    if not isinstance(m.get("q"), str) or not isinstance(m.get("context"), str):
        raise ValueError("bad exp_040 meta: q and context must be strings")
    k = m["kind"]
    if k in ("numeric", "arith"):
        if m.get("value") is None or not math.isfinite(float(m["value"])):
            raise ValueError("bad exp_040 meta: numeric/arith needs a finite value")
    elif k == "abstain":
        if m.get("label") != ABSTAIN:
            raise ValueError("bad exp_040 meta: abstain rows carry label ABSTAIN")
    else:
        if not m.get("candidates") or m.get("label") not in m["candidates"]:
            raise ValueError("bad exp_040 meta: categorical/yesno need candidates containing the label")
        if ABSTAIN in m["candidates"]:
            raise ValueError("bad exp_040 meta: ABSTAIN is implicit, not a listed candidate")
    return m


@lru_cache(maxsize=4096)
def _meta(answer: str) -> dict:
    return _check(json.loads(answer))


# ---------------------------------------------------------------- final line
def final_line(ans: str) -> str | None:
    for m in reversed(list(_KEY.finditer(ans))):
        before = ans[:m.start()].rstrip(" \t*_`\"'«„")
        if not before or before[-1] in _OPENS:
            end = ans.find("\n", m.end())
            return ans[m.end():end if end >= 0 else None].strip().strip("*_`\"'»”").strip()
    return None


def candidate_hits(line: str, candidates: dict) -> set[str]:
    """Candidates named on the line; spans nested inside a longer span of another candidate are dropped."""
    f = fold(line)
    spans = []
    for name, forms in {**(candidates or {}), ABSTAIN: ABSTAIN_FORMS}.items():
        for form in forms:
            ff = fold(form).strip()
            if not ff:
                continue
            for m in re.finditer(r"(?<!\w)" + re.escape(ff) + r"(?!\w)", f):
                spans.append((m.start(), m.end(), name))
    keep = {n for s, e, n in spans
            if not any(s2 <= s and e <= e2 and (e2 - s2) > (e - s) and n2 != n for s2, e2, n2 in spans)}
    return keep


def numbers_on(line: str) -> list[set[float]]:
    out = []
    clean = rv._CITE.sub(" ", line)
    for m in R._NUMBER.finditer(clean):
        sign = -1.0 if m.start() and clean[m.start() - 1] in "-−" else 1.0
        out.append({sign * v for v in R._amount_values(m.group(0))})
    return out


def judge_answer(ans: str, m: dict) -> tuple[float, str]:
    """(value, verdict) for the answer component. verdict ∈ right / wrong / hedge / missing."""
    line = final_line(ans)
    if line is None:
        return -1.0, "missing"
    kind = m["kind"]
    if kind in ("numeric", "arith"):
        abstained = ABSTAIN in candidate_hits(line, {})
        nums = numbers_on(line)
        distinct = []
        for vs in nums:
            if not any(vs & d or any(abs(a - b) < 1e-9 for a in vs for b in d) for d in distinct):
                distinct.append(vs)
        if abstained and distinct:
            return -2.0, "hedge"
        if abstained:
            return -2.0, "wrong"
        if not distinct:
            return -1.0, "missing"
        if len(distinct) > 1:
            return -2.0, "hedge"
        gold = float(m["value"])
        tol = max(abs(gold) * float(m["tol"]), 1e-6)
        return (3.0, "right") if any(abs(v - gold) <= tol for v in distinct[0]) else (-2.0, "wrong")
    if kind == "abstain":   # any final line that does not abstain is an answer where none was due: wrong, not missing
        return (3.0, "right") if ABSTAIN in candidate_hits(line, {}) else (-2.0, "wrong")
    hits = candidate_hits(line, m["candidates"])
    if not hits:
        return -1.0, "missing"
    if len(hits) > 1:
        return -2.0, "hedge"
    return (3.0, "right") if hits == {m["label"]} else (-2.0, "wrong")


# ---------------------------------------------------------------- the reward
@lru_cache(maxsize=8192)
def analyse(completion: str, answer: str) -> dict:
    m = _meta(answer)
    ans = R._THINK.sub("", completion).strip()
    ctx, q, lang = m["context"], m["q"], m["lang"]
    ntok = R.token_count(completion)
    at_cap = ntok >= CAP - CAP_SLACK
    ans_lang = rv.detect_answer_lang(ans)
    has_md = bool(rv._MARKDOWN.search(ans))
    cites = rv._cites(ans)
    bad_cites = [c for c in cites if not (ctx and rv.cite_ok(c, ctx))]
    support = R._numbers(ctx + " " + q)
    if m.get("value") is not None:
        support |= {R._canon(f"{float(m['value']):g}"), R._canon(f"{float(m['value']):.2f}")}
    nums = R._numbers(rv._CITE.sub(" ", ans)) - R.SMALL
    bad_nums = sorted(n for n in nums if n not in support and not rv.derivable(n, support))
    ngram_max = R._max_ngram_count(ans)
    lcs = R.longest_common_substring(ans, ctx) if ctx else 0
    a_val, verdict = judge_answer(ans, m)

    r = dict.fromkeys(COMPONENTS)
    r["answer"] = a_val
    r["lang"] = 0.5 if ans_lang == lang else -0.5
    r["plain"] = -0.25 if has_md else 0.25
    r["cite"] = 0.0 if not cites else (0.25 if not bad_cites else max(-1.0, -0.5 * len(bad_cites)))
    r["num"] = 0.0 if not bad_nums else max(-1.0, -0.5 * len(bad_nums))
    r["length"] = -1.0 if at_cap else 0.0
    r["repeat"] = -1.0 if ngram_max >= R.NGRAM_REPEATS else 0.0
    r["copy"] = -1.0 if lcs > R.COPY_T else 0.0
    facts = {"verdict": verdict, "final_line": final_line(ans), "answer_lang": ans_lang, "markdown": has_md,
             "cites": cites, "bad_cites": bad_cites, "bad_nums": bad_nums, "tokens": ntok, "at_cap": at_cap,
             "ngram_max": ngram_max, "lcs": lcs, "words": len(ans.split())}
    return {"components": r, "total": sum(r.values()), "facts": facts}


def _component(name: str):
    def fn(prompts, completions, answer, types=None):
        return [analyse(c, a)["components"][name] for c, a in zip(completions, answer, strict=True)]
    fn.__name__ = f"coapi40_{name}"
    return register_reward_function(fn.__name__)(fn)


REWARD_FUNCTIONS = [_component(n) for n in COMPONENTS]
REWARD_FUNCTION_NAMES = ",".join(f.__name__ for f in REWARD_FUNCTIONS)


def total_reward(completion: str, answer: str) -> float:
    return analyse(completion, answer)["total"]
