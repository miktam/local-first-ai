"""Synthetic score sets and bench records for the analysis tests (exp_036, BUILD_SPEC §5.8).

Everything here is invented: item ids like "mm_03_007", category names
"cat00".."cat13", and counts chosen so that each hypothesis' true verdict is
known by construction. No dataset text, no gold and no model output appears.

Other test_analysis_*.py modules import World and helpers from here; the tests
at the bottom check the builder itself.
"""

from __future__ import annotations

import copy
import json
import math
import zlib
from pathlib import Path

import numpy as np
import pytest

from analysis import verdicts as V

CATS = [f"cat{c:02d}" for c in range(14)]
# Unbalanced full-split counts (the MMLU-Pro shares) for post-stratification.
COUNTS = [1351, 1299, 1132, 1101, 969, 924, 844, 818, 798, 789, 717, 499, 410, 381]
POST_STRAT = {"en": dict(zip(CATS, COUNTS)), "de": dict(zip(CATS, reversed(COUNTS)))}
# Amendment 4 (2026-10-04): the Lite sample is not 42 per category. The world allocates n_M by the kit's own Webster
# rule (tasks/mmlu_prox.py) over the real MMLU-ProX-Lite counts, carried onto the invented category names in the
# registered subject order; the post-stratification weights above stay as they were, so in this world a category's
# share of the sample and its weight differ, which is what the post-stratification has to undo.
LITE_COUNTS = dict(zip(CATS, (48, 65, 40, 56, 36, 48, 25, 20, 46, 42, 40, 19, 68, 35)))


def mmlu_alloc(n_M):
    """Items per category (in CATS order) at n_M: the Webster allocation of Amendment 4."""
    from tasks.mmlu_prox import allocation

    a = allocation(LITE_COUNTS, n_M)
    return [a[c] for c in CATS]
GIB = 2 ** 30

N_TASK = {"gpqa_en": 198, "gpqa_de": 198, "ifbench": 300, "rgb_cb": 400, "rgb_forced": 400,
          "rgb_neg": 300, "rgb_fact": 100, "aime_en": 30, "aime_de": 30}
# Amendment 3 (2026-10-04): eval-framework's over-long filter excludes no GPQA Diamond item, so the recorded
# exclusion list is empty and the primary GPQA EN set is all 198 items.
GPQA_EN_EXCLUDED: list = []

# Vendor values (fractions) of the rows, per arm family, from analysis/vendor_values.json.
VENDOR = V.load_vendor()


def vend(row, arm):
    return V.vendor_value(VENDOR, row, arm)


def item_ids(task, n_M=588):
    if task.startswith("mmlu_"):
        alloc = mmlu_alloc(n_M)
        return [(f"mm_{c:02d}_{j:03d}", CATS[c]) for c in range(14) for j in range(alloc[c])]
    if task == "rgb_fact":       # the en_fact questions: the same ids as in the closed-book set
        return [(f"rgb_{300 + j:03d}", None) for j in range(N_TASK[task])]
    prefix = {"gpqa_en": "gq_en", "gpqa_de": "gq_de", "ifbench": "if", "rgb_cb": "rgb",
              "rgb_forced": "rgb", "rgb_neg": "rgb", "aime_en": "ai", "aime_de": "ai"}[task]
    return [(f"{prefix}_{j:03d}", None) for j in range(N_TASK[task])]


def pattern(n, k, salt):
    """Exactly k ones among n, at positions fixed by the salt string (deterministic)."""
    k = int(max(0, min(n, k)))
    perm = np.random.default_rng(zlib.crc32(salt.encode())).permutation(n)
    v = np.zeros(n)
    v[perm[:k]] = 1.0
    return v


def rec(arm, task, item, correct, truncated=False, category=None, effort="high", pass_=0, **extra):
    r = {"key": {"arm": arm, "task": task, "effort": effort, "item": item, "pass": pass_},
         "item_sha256": "0" * 64, "extracted": None, "category": category,
         "correct": bool(correct) and not truncated, "truncated": bool(truncated),
         "parse_status": "ok"}
    r.update(extra)
    return r


class World:
    """A full synthetic run: scores, bench records and a plan.

    rates[(arm, task)] = accuracy (MMLU: the same rate in every category unless a
    per-category list is given; each category's n is its Webster allocation at n_M,
    so a rate lands to within half an item per category). trunc[(arm, task)] = item indices truncated
    (scored 0; the rest of the pattern is unchanged).
    """

    def __init__(self, n_M=588):
        self.n_M = n_M
        self.rates: dict = {}
        self.trunc: dict = {}
        self.categories_rgb: dict = {}
        self.extra_records: list = []
        self.speed = None
        self.speed_desc = None
        self.fit = None
        self.tokenizer = None
        self.kl = None
        self.c1 = None
        self.c1_summary = None
        self.ladder = None
        self.gate = None
        self.plan = {"plan": "P0", "n_M": n_M, "gpqa_en_excluded": list(GPQA_EN_EXCLUDED),
                     "post_strat_counts": POST_STRAT, "peers": ["G8", "Q36-8"], "tier_b": []}

    # ---- scores
    def set(self, arm, task, rate, effort="high", pass_=0):
        self.rates[(arm, task, effort, pass_)] = rate
        return self

    def records(self):
        """Score records; a rate of None means "a copy of K8's records on the same cell"."""
        out = []
        copies = []
        for (arm, task, effort, pass_), rate in sorted(self.rates.items(), key=lambda kv: str(kv[0])):
            if rate is None:
                copies.append((arm, task, effort, pass_))
                continue
            ids = item_ids(task, self.n_M)
            salt = f"{arm}|{task}|{effort}|{pass_}"
            tr = self.trunc.get((arm, task), set())
            if task.startswith("mmlu_"):
                v = np.zeros(len(ids))
                start = 0
                for c, nc in enumerate(mmlu_alloc(self.n_M)):
                    rc = rate[c] if isinstance(rate, (list, tuple)) else rate
                    v[start:start + nc] = pattern(nc, round(rc * nc), f"{salt}|{c}")
                    start += nc
            else:
                v = pattern(len(ids), round(rate * len(ids)), salt)
            cats = self.categories_rgb.get((arm, task))
            for j, ((iid, cat), x) in enumerate(zip(ids, v)):
                extra = {}
                category = cat
                if task in ("rgb_cb", "rgb_forced"):
                    category = "correct" if x else ("abstain" if (cats and j % cats == 0) else "wrong")
                    extra["subset"] = "en_fact" if j >= 300 else "en"
                if task == "rgb_neg":
                    category = "rejected" if x else "not"
                if task == "rgb_fact":
                    category = "corrected" if x else ("deferred" if j % 2 else "detected")
                if task.endswith("_de") and arm.startswith("K"):
                    extra["reasoning_lang"] = "de" if j % 20 else "mixed"
                if task == "ifbench":
                    extra["correct_loose"] = bool(x) and j not in tr
                    extra["correct_strict"] = bool(x) and j % 7 != 0 and j not in tr
                out.append(rec(arm, task, iid, x, truncated=j in tr, category=category,
                               effort=effort, pass_=pass_, **extra))
        for (arm, task, effort, pass_) in copies:
            for r in [r for r in out if r["key"]["arm"] == "K8" and r["key"]["task"] == task
                      and r["key"]["effort"] == effort and r["key"]["pass"] == pass_]:
                r2 = copy.deepcopy(r)
                r2["key"]["arm"] = arm
                out.append(r2)
        return out + list(self.extra_records)

    def scores(self):
        return V.Scores(self.records())

    def bench(self):
        b = {"_incomplete": []}
        for k in ("speed", "speed_desc", "fit", "tokenizer", "kl", "c1", "ladder", "gate"):
            if getattr(self, k) is not None:
                b[k] = copy.deepcopy(getattr(self, k))
        if self.c1_summary is not None:
            b["c1_summary"] = copy.deepcopy(self.c1_summary)
        return b

    def ctx(self, margins=None):
        return V.Context(None, self.plan, self.scores(), self.bench(), margins or V.load_margins(), VENDOR)

    def compute(self, margins=None, **kw):
        kw.setdefault("require_rescore", False)
        return V.compute(None, self.plan, self.scores(), bench=self.bench(), margins=margins,
                         vendor=VENDOR, **kw)

    # ---- writing a results/ tree (for the rescore and file round trip)
    def write_results(self, results: Path, rescore=True, tamper=None):
        results.mkdir(parents=True, exist_ok=True)
        by_file: dict = {}
        for r in self.records():
            k = r["key"]
            by_file.setdefault((k["arm"], f"{k['task']}_{k['effort']}"), []).append(r)
        entries = []
        for (arm, stem), recs in sorted(by_file.items()):
            p = results / "scores" / arm / f"{stem}.jsonl"
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(json.dumps({"type": "header", "arm": arm}, sort_keys=True) + "\n")
                for r in recs:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
            sha = V.sha256_file(p)
            mini_only = stem.startswith("ifbench")
            entries.append({"path": p.relative_to(results).as_posix(), "sha256_mini": sha,
                            "sha256_mbp": None if mini_only else sha, "mini_only": mini_only, "match": True})
        if tamper:
            tamper(entries)
        if rescore:
            with open(results / "rescore_mini_20261010T000000Z.json", "w", encoding="utf-8") as f:
                json.dump({"files": entries}, f, sort_keys=True)
        bench_dir = results / "bench"
        bench_dir.mkdir(exist_ok=True)
        for kind in ("speed", "speed_desc", "fit", "c1", "ladder"):
            data = getattr(self, kind)
            if data is not None:
                summary = self.c1_summary if kind == "c1" else {}
                with open(bench_dir / f"{kind}_20261005T000000Z.jsonl", "w", encoding="utf-8") as f:
                    f.write(json.dumps({"type": "header", "cell": kind}) + "\n")
                    for r in data:
                        f.write(json.dumps(r, sort_keys=True) + "\n")
                    f.write(json.dumps({"type": "end", "complete": True, "summary": summary}) + "\n")
        if self.tokenizer is not None:
            (results / "tokenizer_20261005T000000Z.json").write_text(json.dumps(self.tokenizer), encoding="utf-8")
        if self.kl is not None:
            (results / "kl_8v4_20261005T000000Z.json").write_text(json.dumps(self.kl), encoding="utf-8")
        return results


# --------------------------------------------------------------------------
# Bench builders
# --------------------------------------------------------------------------

def speed_blocks(ratio, noise_sd=0.01, n=10, g_tps=100.0, salt="speed"):
    """bench/speed.py layout: warm-ups, one "run" per arm and block, a "block" summary, prefill reps."""
    rng = np.random.default_rng(zlib.crc32(salt.encode()))
    out = [{"kind": "warmup", "arm": "K4", "generation_tps": 1.0, "prompt_tps": 1.0},
           {"kind": "warmup", "arm": "G4", "generation_tps": 1.0, "prompt_tps": 1.0}]
    eps = rng.normal(0.0, noise_sd, size=n)
    eps = eps - eps.mean()        # the geometric mean ratio is exactly `ratio`
    for b in range(n):
        k = g_tps * ratio * math.exp(eps[b])
        out.append({"kind": "run", "block": b, "position": 0, "arm": "G4", "generation_tps": g_tps, "prompt_tps": 1500.0})
        out.append({"kind": "run", "block": b, "position": 1, "arm": "K4", "generation_tps": k, "prompt_tps": 1400.0})
        out.append({"kind": "block", "block": b, "order": ["G4", "K4"], "generation_tps": {"G4": g_tps, "K4": k}})
    for rep in range(2):
        for arm, tps in (("K4", 1400.0), ("G4", 1500.0)):
            out.append({"kind": "prefill", "rep": rep, "arm": arm, "prompt_tps": tps})
    return out


def speed_desc():
    """bench/speed.py speed_desc layout: decode_b1, prefill and batch records per arm and rep."""
    out = []
    for arm, tps in (("K8", 70.0), ("G8", 80.0), ("Q36-8", 90.0), ("Q36-4", 110.0), ("Q38-8", 25.0), ("Q38-4", 40.0)):
        out.append({"kind": "arm", "arm": arm})
        out.append({"kind": "warmup", "arm": arm, "generation_tps": 1.0})
        for rep in range(5):
            out.append({"kind": "prefill", "arm": arm, "rep": rep, "prompt_tps": 10 * tps})
            out.append({"kind": "decode_b1", "arm": arm, "rep": rep, "generation_tps": tps})
            for B in (1, 2, 4):
                out.append({"kind": "batch", "arm": arm, "rep": rep, "B": B, "generation_tps": tps * (1 + 0.8 * (B - 1))})
    return out


def fit_records(peaks_gib_64k, peak_gib_32k=40.0):
    """bench/fit.py layout: K4 "run" records with n_context and peak_bytes."""
    out = [{"kind": "run", "n_context": 32768, "rep": r, "peak_bytes": int(peak_gib_32k * GIB),
            "prompt_tps": 1000.0} for r in range(3)]
    out += [{"kind": "run", "n_context": 65536, "rep": r, "peak_bytes": int(p * GIB), "prompt_tps": 800.0}
            for r, p in enumerate(peaks_gib_64k)]
    return out


TOK_COLUMNS = ["index", "bytes", "chars", "digits", "tokens_kolibri", "tokens_gemma4", "tokens_qwen3_6", "tokens_qwen3_8"]


def tokenizer_docs(bpt_k=4.90, bpt_g=4.13, bpt_q=4.17, n=400, noise=0.01, salt="tok"):
    """bench/tokenizer_ratio.py layout: {"de": {"columns", "docs": rows}, "en_descriptive", "complete"}."""
    rng = np.random.default_rng(zlib.crc32(salt.encode()))
    rows = []
    for j in range(n):
        nbytes = int(2000 + 37 * (j % 50))

        def toks(bpt):
            return max(1, int(round(nbytes / (bpt * math.exp(rng.normal(0.0, noise))))))

        tq = toks(bpt_q)
        rows.append([5000 + j, nbytes, int(nbytes * 0.97), int(nbytes * (0.08 if j % 10 == 0 else 0.01)),
                     toks(bpt_k), toks(bpt_g), tq, tq])
    return {"cell": "tokenizer", "complete": True,
            "de": {"start": 0, "n_docs": n, "columns": TOK_COLUMNS, "docs": rows,
                   "tokenizer_identity": {"qwen3_8_vs_qwen3_6": {"identical": True, "n_docs_differing": 0}}},
            "en_descriptive": {"gate_T1": {"status": "ok", "label": "not FineWeb"}}}


def kl_blob(k_ratio_per_text=(1.0,) * 6, noise=0.05, peers=("gemma4", "qwen3_6", "qwen3_8"), salt="kl"):
    """bench/kl_8v4.py layout. KL per block = ratio x 0.01 nats/byte x bytes, multiplicative noise."""
    rng = np.random.default_rng(zlib.crc32(salt.encode()))
    fams = {}
    texts = [f"T{t + 1}" for t in range(6)]
    for mdl, scale in [("kolibri", None)] + [(p, 1.0) for p in peers]:
        fams[mdl] = {"arms": [], "texts": {}}
        for t, name in enumerate(texts):
            blocks = []
            for b in range(8):
                nbytes = 900 + 13 * b + 7 * t
                ratio = k_ratio_per_text[t] if scale is None else scale
                kl = 0.01 * ratio * nbytes * math.exp(rng.normal(0.0, noise))
                blk = {"block": b, "bytes": nbytes, "tokens": int(nbytes / (4.5 if scale is None else 4.0)),
                       "kl_sum": kl, "nll8_sum": 1.0, "nll4_sum": 1.0, "agree_count": 1.0}
                if scale is None:
                    blk["kl_fp32_sum"] = kl * 1.01
                blocks.append(blk)
            fams[mdl]["texts"][name] = {"blocks": blocks}
    return {"complete": True, "families": fams}


def all_confirmed_world(n_M=588):
    """Every confirmatory hypothesis CONFIRMED by construction (see the module docstring)."""
    w = World(n_M)
    # H2: K8 at the vendor values on every row (D-bar about 0, well above -4 pp).
    w.set("K8", "gpqa_en", vend("gpqa_en", "K8"))
    w.set("K8", "gpqa_de", vend("gpqa_de", "K8"))
    w.set("K8", "mmlu_en", vend("mmlu_en", "K8"))
    w.set("K8", "mmlu_de", vend("mmlu_de", "K8"))
    w.set("K8", "ifbench", vend("ifbench", "K8"))
    # Peers at their vendor values (E8 control passes; H3 gap -5.2 pp; H4 +12 pp).
    for arm in ("G8", "Q36-8"):
        for row in ("mmlu_en", "mmlu_de", "ifbench", "rgb_cb"):
            w.set(arm, row, vend(row, arm))
    # H6: RGB closed-book K8 51 % vs 79 / 79 (-28 pp).
    w.set("K8", "rgb_cb", vend("rgb_cb", "K8"))
    w.categories_rgb[("K8", "rgb_cb")] = 2      # every 2nd non-correct item is an abstention
    # H7: K4 identical to K8 on every H7 row (D-bar = 0, no variance).
    for row in ("mmlu_en", "mmlu_de", "ifbench", "rgb_cb"):
        w.rates[("K4", row, "high", 0)] = None
    w.speed = speed_blocks(0.90)
    w.speed_desc = speed_desc()
    w.fit = fit_records([43.5, 43.7, 43.6])
    w.tokenizer = tokenizer_docs()
    w.kl = kl_blob()
    return w


# --------------------------------------------------------------------------
# Tests of the builder itself
# --------------------------------------------------------------------------

def test_pattern_exact_and_deterministic():
    v = pattern(300, 234, "x")
    assert v.sum() == 234
    assert np.array_equal(v, pattern(300, 234, "x"))
    assert not np.array_equal(v, pattern(300, 234, "y"))


def test_world_rates_land_exactly():
    w = all_confirmed_world()
    ctx = w.ctx()
    assert len(ctx.cell("K8", "gpqa_en")) == 198
    # MMLU rows: every category at the same rate, so the post-stratified mean equals the rate to within half an item
    # of the smallest category (Amendment 4: unequal n per category, 19 to 68 at n_M = 588; was 1/42 at 42 each).
    pt = V.row_point(ctx, "K8", "mmlu_en")
    assert abs(pt - vend("mmlu_en", "K8")) <= 1.0 / (2 * min(mmlu_alloc(588)))
    assert mmlu_alloc(588) == list(LITE_COUNTS.values()) and sum(mmlu_alloc(154)) == 154
    # K4 is a copy of K8 on the H7 rows.
    c4, c8 = ctx.cell("K4", "ifbench"), ctx.cell("K8", "ifbench")
    assert all(c4[i].score == c8[i].score for i in c8)


def test_world_has_no_text_fields():
    for r in all_confirmed_world().records():
        assert "text" not in r and "gold" not in r and "prompt" not in r
