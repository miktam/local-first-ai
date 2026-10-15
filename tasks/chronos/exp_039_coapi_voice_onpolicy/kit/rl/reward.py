#!/usr/bin/env python3
"""
exp_039 — the deterministic verifier reward (DESIGN §5). No model call of any kind.

Each rollout is scored by code, from run_eval.py's own primitives, into eleven
components. Each component is registered as its own mlx-lm-lora reward function,
so the trainer logs `<name>_mean / _std / _coverage` per step (DESIGN §9) and
sums them with weight 1 — the values below already carry the design's weights.
`None` means "not applicable to this prompt" (the trainer treats it as 0 and
excludes it from the component's mean).

  component  check                                                        value
  lang       answer language == question language                         +1 / −1
  plain      no markdown                                                  +0.5 / −0.5
  cite       in-scope: ≥1 citation, all pass cite_ok against the context   +1; −1 per unsupported (floor −2); 0 if none
  num        every number in context ∪ question (∪ arith gold), or derivable  +1; −1 per unsupported (floor −2)
  arith      arithmetic prompts: code-computed gold present within 0.5 %   +2 / −1
  term       in-scope EN/PL: a glossary Spanish term (≥ 4 chars) present    +0.5 / 0
  handoff    advice: professional route, no invented figure, no drafted text  +1 / −2
  oos        out of scope: decline, no citation, < 80 words                +2 / −2
  length     in-scope: over 220 words −0.5; any row at the cap −1 (no floor since rev 2)
  repeat     any word 8-gram three or more times                           −1 / 0
  copy       longest common substring with the context > COPY_T chars      −1 / 0

Interpretations of the design table, fixed here (a change is a re-registration):
  • term counts glossary entries only — no acronym fallback as in score_row, since
    "ITP" is the cheapest string to stuff — and only where the §8 floor measures it
    (in-scope rows whose question is EN or PL).
  • rev 2 (2026-10-14, amendment 4): the 80-word floor is removed. The H5 audit of run 1 found the
    gamed answers padded past it where the material was thin (5/40, Fable). Nothing in the reward
    pays for length any more; only running long (> 220 words) or hitting the cap costs.
  • length's ceiling applies to in-scope rows; out-of-scope rows are held to < 80
    words by `oos`, so only the cap applies to them.
  • num's support includes the arith gold, so the correct computed answer is not
    charged as an unsupported number.
  • num and arith read numbers with a broader pattern than score_row's _NUM, which
    cannot see a 4+ digit number written without separators ("48000", "€1158") — a
    policy could learn to drop separators to escape the check. Numbers inside a
    citation span (art. 9.1.e, Ley 22/2021, Tema 15) are removed first: they are cite's.
  • handoff's "figure" = a money or percent amount whose number is not in the
    question or the context (no derivation: computing an offer is advising one);
    "drafted text" = a clause / salutation heading or a quotation of ≥ 12 words.
  • the cap is detected by re-tokenising the completion with the base's tokenizer
    (the trainer strips EOS and passes text only): ≥ CAP − CAP_SLACK tokens.
  • a closed <think>…</think> block is removed before scoring (thinking is off at
    rollout and serve; the block, if any, is not the answer).

Dataset contract (mlx-lm-lora GRPO rows: prompt / answer / system / type):
`answer` must be the JSON string from `encode_meta(...)` — the trainer passes
str(item["answer"]), so a dict would arrive as a Python repr and is refused.

What this reward cannot see (DESIGN §5, [[evals-first]]): a rule inverted with the
right numbers; an invented statute name on a real article number; confident filler
that cites once and says nothing; native fluency. H2a and the judge guard those.

  python rl/reward.py calibrate RUN_DIR   # p95 of A12's v1 LCS → COPY_T (step 1)
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_eval as rv  # noqa: E402  (pulls guide_bm25; no index is loaded at import)

try:
    from mlx_lm_lora.trainer.grpo_reward_functions import register_reward_function
except ImportError:                                  # the scorer's venv: tests, calibration
    def register_reward_function(name=None):
        return lambda f: f

# --- frozen constants (recorded in the exp_039 manifest by sha256 of this file)
COPY_T = 109             # calibrated 2026-10-13: p95 of A12 v1 (glue_…_A12_20260922-2118), n=60, median 44, max 264
CAP = 512                # = max_completion_length in the GRPO config = run_eval.OPTIONS["num_predict"]
CAP_SLACK = 2            # re-tokenised decoded text can differ from the sampled ids by a token or two
REWARD_REV = 2                # rev 1 had a floor of 80 words (WORDS_LO), removed by amendment 4
WORDS_HI, OOS_WORDS = 220, 80
NGRAM, NGRAM_REPEATS = 8, 3
ARITH_TOL = 0.005
DRAFT_QUOTE_WORDS = 12
TOKENIZER_REPO, TOKENIZER_REV = "Qwen/Qwen3-4B", "1cfa9a7208912126459214e8b04321603b3df60c"
TYPES = {"numeric", "simple", "multi", "advice", "oos", "arith"}
LANGS = {"en", "pl", "es"}
COMPONENTS = ("lang", "plain", "cite", "num", "arith", "term", "handoff", "oos", "length", "repeat", "copy")
SMALL = {"1", "2", "3"}  # tiny ordinals are noise, as in score_row

_NUMBER = re.compile(r"(?<![\w/])\d+(?:[.,]\d+)*(?:\s\d{3}(?!\d))*(?:[.,]\d+)?(?:\s?[kK](?!\w))?")
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_MONEY = re.compile(r"(?:€|eur\b|euros?\b|zł|pln\b)\s*(\d[\d.,\s]*\d|\d)|(\d[\d.,\s]*\d|\d)\s*(?:€|eur\b|euros?\b|zł|pln\b|%|por ?ciento|per ?cent|procent)", re.I)
_DRAFT_HEAD = re.compile(r"(?:^|\n)\s*(?:clause|cl[aá]usula|klauzula|asunto|subject|temat|dear|estimad[oa]s?|muy se[ñn]or|szanown[aiy]+|drog[aiy])\b", re.I)
_QUOTE = re.compile(r"[\"“«„]([^\"”»“]+)[\"”»“]")


# ---------------------------------------------------------------- metadata
def encode_meta(*, lang: str, type: str, q: str, context: str, arith: float | None = None) -> str:
    """The `answer` field of a GRPO row. The prompt builder must use this."""
    meta = {"v": 1, "lang": lang, "type": type, "q": q, "context": context, "arith": arith}
    _check_meta(meta)
    return json.dumps(meta, ensure_ascii=False)


def _check_meta(m: dict) -> dict:
    if m.get("v") != 1 or m.get("lang") not in LANGS or m.get("type") not in TYPES:
        raise ValueError(f"bad exp_039 meta: v/lang/type = {m.get('v')}/{m.get('lang')}/{m.get('type')}")
    if not isinstance(m.get("q"), str) or not isinstance(m.get("context"), str):
        raise ValueError("bad exp_039 meta: q and context must be strings")
    if (m["type"] == "arith") != (m.get("arith") is not None):
        raise ValueError("bad exp_039 meta: arith gold is required for, and only for, type 'arith'")
    if m["type"] == "arith" and not math.isfinite(float(m["arith"])):
        raise ValueError("bad exp_039 meta: arith gold must be finite")
    return m


@lru_cache(maxsize=4096)
def _meta(answer: str) -> dict:
    return _check_meta(json.loads(answer))


# ---------------------------------------------------------------- glossary, tokenizer
_glossary: tuple[str, ...] | None = None
_token_count = None


def set_glossary(terms) -> None:
    global _glossary
    _glossary = tuple(sorted({t for t in terms if len(t) > 3}))
    analyse.cache_clear()


def glossary() -> tuple[str, ...]:
    """EXP039_GLOSSARY (a JSON list, shipped in the private bundle) or the guide index."""
    if _glossary is None:
        p = os.environ.get("EXP039_GLOSSARY")
        set_glossary(json.loads(Path(p).read_text(encoding="utf-8")) if p else rv.g.load_index("en").glossary)
    return _glossary


def set_token_counter(fn) -> None:
    global _token_count
    _token_count = fn
    analyse.cache_clear()


def token_count(text: str) -> int:
    """Tokens of the completion under the base's own tokenizer, pinned revision, local files only."""
    global _token_count
    if _token_count is None:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REV,
                                                  local_files_only=True))
        _token_count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)  # noqa: E731
    return _token_count(text)


# ---------------------------------------------------------------- primitives
def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", rv._fold(s)).strip()


def longest_common_substring(a: str, b: str) -> int:
    """Length of the longest common substring of a and b after folding case, diacritics and
    whitespace. Binary search on length over hashed slices: O((|a|+|b|)·L·log L)."""
    a, b = _norm_ws(a), _norm_ws(b)
    if not a or not b:
        return 0
    if len(a) > len(b):
        a, b = b, a

    def common(n: int) -> bool:
        subs = {b[i:i + n] for i in range(len(b) - n + 1)}
        return any(a[i:i + n] in subs for i in range(len(a) - n + 1))

    lo, hi = 0, len(a)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if common(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def _canon(tok: str) -> str:
    """One canonical string per number: '48.000' / '48,000' / '48 000' / '48000' → '48000';
    '1.234,56' / '1,234.56' → '1234.56'; '2,5' → '2.5'; '150k' → '150000'."""
    t = tok.replace("\u00a0", " ").strip()
    if t[-1:] in "kK":
        return rv._norm_num(t)
    t = t.replace(" ", "")
    if "." in t and "," in t:
        dec = max(t.rfind("."), t.rfind(","))
        return re.sub(r"[.,]", "", t[:dec]) + "." + t[dec + 1:]
    return rv._norm_num(t)


def _numbers(text: str) -> set[str]:
    return {_canon(m.group(0)) for m in _NUMBER.finditer(text)}


def _amount_values(tok: str) -> set[float]:
    """Every plausible value of a numeric token under EN and ES/PL conventions:
    '1.234,56' → 1234.56; '1.234' → {1.234, 1234}; '12 500' → 12500; '150k' → 150000."""
    t = tok.replace(" ", " ").strip()
    mult = 1000.0 if t[-1:] in "kK" else 1.0
    t = t.rstrip("kK").strip().replace(" ", "")
    out = set()
    for thou, dec in ((",", "."), (".", ",")):
        s = t.replace(thou, "").replace(dec, ".")
        try:
            out.add(float(s) * mult)
        except ValueError:
            pass
    return out


def _arith_hit(ans: str, gold: float) -> bool:
    tol = max(abs(gold) * ARITH_TOL, 0.01)
    for m in _NUMBER.finditer(ans):
        sign = -1.0 if m.start() and ans[m.start() - 1] in "-−" else 1.0     # NPS can be negative
        if any(abs(sign * v - gold) <= tol for v in _amount_values(m.group(0))):
            return True
    return False


def _max_ngram_count(ans: str) -> int:
    w = re.findall(r"\w+", ans.lower())
    if len(w) < NGRAM:
        return 0
    counts: dict[tuple, int] = {}
    for i in range(len(w) - NGRAM + 1):
        k = tuple(w[i:i + NGRAM])
        counts[k] = counts.get(k, 0) + 1
    return max(counts.values())


def _invented_figure(ans: str, support_nums: set[str]) -> list[str]:
    figs = []
    for m in _MONEY.finditer(ans):
        n = _canon((m.group(1) or m.group(2)).strip())
        if n not in support_nums:
            figs.append(n)
    return figs


def _drafted(ans: str) -> bool:
    return bool(_DRAFT_HEAD.search(ans)) or any(len(q.split()) >= DRAFT_QUOTE_WORDS for q in _QUOTE.findall(ans))


# ---------------------------------------------------------------- the reward
@lru_cache(maxsize=8192)
def analyse(completion: str, answer: str) -> dict:
    """All components for one rollout, plus the raw facts behind them (for logs and the H5 audit)."""
    m = _meta(answer)
    ans = _THINK.sub("", completion).strip()
    ctx, q, typ, lang = m["context"], m["q"], m["type"], m["lang"]
    inscope = typ != "oos"
    words = len(ans.split())
    ntok = token_count(completion)
    at_cap = ntok >= CAP - CAP_SLACK

    ans_lang = rv.detect_answer_lang(ans)
    has_md = bool(rv._MARKDOWN.search(ans))

    cites = rv._cites(ans)
    bad_cites = [c for c in cites if not (ctx and rv.cite_ok(c, ctx))]

    support = _numbers(ctx + " " + q)
    if m["arith"] is not None:
        support |= {_canon(f"{m['arith']:g}"), _canon(f"{m['arith']:.2f}")}
    nums = _numbers(rv._CITE.sub(" ", ans)) - SMALL
    bad_nums = sorted(n for n in nums if n not in support and not rv.derivable(n, support))

    terms = [t for t in glossary() if t in rv._fold(ans)]
    refused = bool(rv._REFUSAL.search(ans))
    handoff = bool(rv._HANDOFF.search(ans))
    figures = _invented_figure(ans, support) if typ == "advice" else []
    drafted = _drafted(ans) if typ == "advice" else False
    ngram_max = _max_ngram_count(ans)
    lcs = longest_common_substring(ans, ctx) if ctx else 0
    if COPY_T is None:
        raise RuntimeError("COPY_T is not set: run `reward.py calibrate` on A12's v1 run and freeze it")

    r = dict.fromkeys(COMPONENTS)
    r["lang"] = 1.0 if ans_lang == lang else -1.0
    r["plain"] = -0.5 if has_md else 0.5
    if inscope:
        r["cite"] = 0.0 if not cites else (1.0 if not bad_cites else max(-2.0, -float(len(bad_cites))))
    r["num"] = 1.0 if not bad_nums else max(-2.0, -float(len(bad_nums)))
    if typ == "arith":
        r["arith"] = 2.0 if _arith_hit(ans, float(m["arith"])) else -1.0
    if inscope and lang in ("en", "pl"):
        r["term"] = 0.5 if terms else 0.0
    if typ == "advice":
        r["handoff"] = 1.0 if handoff and not figures and not drafted else -2.0
    if typ == "oos":
        r["oos"] = 2.0 if refused and not cites and words < OOS_WORDS else -2.0
    r["length"] = -1.0 if at_cap else (-0.5 if inscope and words > WORDS_HI else 0.0)
    r["repeat"] = -1.0 if ngram_max >= NGRAM_REPEATS else 0.0
    r["copy"] = -1.0 if lcs > COPY_T else 0.0

    total = sum(v for v in r.values() if v is not None)
    facts = {"answer_lang": ans_lang, "markdown": has_md, "cites": cites, "bad_cites": bad_cites,
             "bad_nums": bad_nums, "terms": terms[:6], "refused": refused, "handoff_route": handoff,
             "figures": figures, "drafted": drafted, "words": words, "tokens": ntok, "at_cap": at_cap,
             "ngram_max": ngram_max, "lcs": lcs}
    return {"components": r, "total": total, "facts": facts}


def _component(name: str):
    def fn(prompts, completions, answer, types=None):
        return [analyse(c, a)["components"][name] for c, a in zip(completions, answer, strict=True)]
    fn.__name__ = f"coapi_{name}"
    return register_reward_function(fn.__name__)(fn)


REWARD_FUNCTIONS = [_component(n) for n in COMPONENTS]
REWARD_FUNCTION_NAMES = ",".join(f.__name__ for f in REWARD_FUNCTIONS)   # → --reward-functions


def total_reward(completion: str, answer: str) -> float:
    return analyse(completion, answer)["total"]


# ---------------------------------------------------------------- calibration (DESIGN §5, step 1)
def calibrate(run_dir: Path) -> dict:
    """COPY_T = nearest-rank 95th percentile of LCS(answer, context) over a run's answers."""
    recs = [json.loads(l) for l in (run_dir / "outputs.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    lcs = sorted(longest_common_substring(r["answer"], r.get("context", "")) for r in recs)
    p95 = lcs[math.ceil(0.95 * len(lcs)) - 1]
    return {"run": run_dir.name, "n": len(lcs), "p95": p95, "max": lcs[-1], "median": lcs[len(lcs) // 2],
            "above_p95": sum(x > p95 for x in lcs)}


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "calibrate":
        print(json.dumps(calibrate(Path(sys.argv[2])), indent=2))
    else:
        sys.exit(__doc__)
