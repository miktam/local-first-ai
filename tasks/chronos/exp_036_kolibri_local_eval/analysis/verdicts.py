"""exp_036 verdicts (Tier 1: H1-H8 and D1; Tier 2 E1-E12 and C1 are delegated to exploratory.py).

BUILD_SPEC §5.8; HYPOTHESIS "Confirmatory hypotheses (H1-H8, Holm family)", D1,
"What counts as evidence", "Plain answers". Every threshold is read from
analysis/margins.json and every vendor number from analysis/vendor_values.json.

Decision rules (HYPOTHESIS, Common rules):
- Two Holm families over the family (H1-H8, m = 8 frozen at the plan
  amendment): the eight one-sided p for the CONFIRMED side and, separately, the
  eight p_rev for the reverse nulls, each at family-wise alpha = 0.05.
- CONFIRMED iff the Holm-adjusted p <= 0.05 (H8 also needs its per-text
  condition). REFUTED iff not CONFIRMED and the Holm-adjusted p_rev <= 0.05.
  Otherwise INCONCLUSIVE, published as "INCONCLUSIVE (leans refuted, nominal
  95 %)" if the unadjusted p_rev <= 0.025.
- A NOT RUN hypothesis keeps its slot with p = p_rev = 1.
- B = 10,000 per hypothesis from its own seed; any H whose raw p or p_rev lies
  within a factor of 2 of its Holm threshold is recomputed with B = 100,000
  before Holm is final.
- Paired truncation sensitivity for H3, H4, H6, H7 (headline only if both
  verdicts agree); E8 peer-control flag for H3, H4, H6; H2 protocol control and
  DiD; H8 per-text condition and per-token sensitivity; H6 wording state; the
  plain-answer state per question.

INPUT CONTRACTS (paths relative to results/; the newest <UTC> file wins):

scores/<arm>/<task>_<effort>.jsonl   one record per item (scorers/score_all.py):
    {"key": {"arm", "task", "effort", "item", "pass"}, "item_sha256",
     "extracted", "category", "correct", "truncated", "parse_status"}
    + IFBench: "correct_loose" (used for H2/H4 when present), "correct_strict"
    + RGB: "category" is correct | abstain | wrong (closed-book, forced),
      rejected | not (negative), corrected | deferred | detected | other (fact);
      optional "subset" ("en" | "en_fact") for the H6 en_fact descriptive row.
    Lines with "type": "header" are skipped. The first record per key counts.
    A truncated item scores 0 whatever "correct" says (HYPOTHESIS Common rules).
    Task ids: gpqa_en gpqa_de mmlu_en mmlu_de ifbench rgb_cb rgb_forced
    rgb_neg rgb_fact aime_en aime_de (TASKS below).
bench/*_<UTC>.jsonl            bench/common.py layout: a header record, measurement records, and
                                an end record {"type": "end", "complete": true, "summary"}; only
                                complete files are read (the newest complete one per kind).
bench/speed_<UTC>.jsonl (H1)   kind "run": {"block", "arm": "K4"|"G4", "generation_tps"}
                                (kind "warmup", "block", "prefill" records are not H1 runs)
bench/speed_desc_<UTC>.jsonl   kind "decode_b1" {"arm", "generation_tps"}, "prefill" {"arm",
                                "prompt_tps"}, "batch" {"arm", "B", "generation_tps"}
bench/fit_<UTC>.jsonl (D1)     kind "run": {"n_context", "rep", "peak_bytes", "prompt_tps"} (K4)
bench/ladder_<UTC>.jsonl (E6)  kind "run": {"arm", "rung", "n_context", "prompt_tps", "peak_bytes"}
bench/c1_<UTC>.jsonl (C1)      per item {"flipped": bool}; end summary {"flip_rate", "B"}
tokenizer_<UTC>.json (H5)      {"de": {"columns": ["index", "bytes", "chars", "digits",
                                "tokens_kolibri", "tokens_gemma4", "tokens_qwen3_6", ...],
                                "docs": [[...], ...], "tokenizer_identity": {...}},
                                "en_descriptive": {...}, "complete": true}
kl_8v4_<UTC>.json (H8)         {"families": {"kolibri"|"gemma4"|"qwen3_6"|"qwen3_8": {"texts":
                                {"T1": {"blocks": [{"bytes", "tokens", "kl_sum", "kl_fp32_sum"
                                (Kolibri only)}, x8]}}}}, "complete": true}; kl_sum = sum of
                                KL(p8 || p4) in nats over the block's positions, from bf16-rounded
                                logits upcast to fp32.
rescore_mini_<UTC>.json        {"files": [{"path": "scores/K8/gpqa_en_high.jsonl",
                                 "sha256_mbp": hex | null, "sha256_mini": hex,
                                 "mini_only": bool, "match": bool}, ...]}
plan (plan_fixed_<UTC>.json), keys read by normalise_plan(); defaults in brackets:
    plan ["P?"], n_M, m [8], family [H1..H8], kolibri_primary ["K8"; "K4" if K8
    cannot run], peers [["G8", "Q36-8"]], h2_rows, h2_secondary_rows, h7_rows,
    tier_b [[]], forced_cells, b10, efforts [{arm: "high"}], not_run [{H: reason}],
    cells [[{"arm", "task", "effort", "n"}]] for completeness checks,
    post_strat_counts [{"en": {category: n}, "de": {...}}; else the plan's
    category_counts (runner/plan_fix.py's frozen copy); else
    tasks/manifests/mmlu_prox_category_counts.json], gpqa_en_excluded [the
    in_primary false ids of tasks/manifests/gpqa_diamond_en.json; else none]:
    Amendment 3 makes the GPQA EN primary set every Diamond item, so it must be
    empty (any id is a ConfigError), h8_texts [T1..T6], h8_peers [gemma4,
    qwen3_6, qwen3_8].

CLI: python analysis/verdicts.py [--results DIR] [--plan FILE] [--no-exploratory]
writes results/verdicts_<UTC>.json and results/verdicts_<UTC>.md and never
overwrites a file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Mapping

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
EXP_DIR = ANALYSIS_DIR.parent
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

from analysis import stats  # noqa: E402

VERSION = "exp036-verdicts-1"
HOST_LABEL = "mbp (M5 Max, 128 GB)"

FAMILY = ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8")
CONFIRMED = "CONFIRMED"
REFUTED = "REFUTED"
INCONCLUSIVE = "INCONCLUSIVE"
NOT_RUN = "NOT RUN"
LEANS_REFUTED = "INCONCLUSIVE (leans refuted, nominal 95 %)"
TRUNCATION_SENSITIVE = "INCONCLUSIVE (truncation-sensitive)"

ARM_FAMILY = {
    "K8": "kolibri", "K4": "kolibri",
    "G8": "gemma4", "G4": "gemma4",
    "Q36-8": "qwen3_6", "Q36-4": "qwen3_6",
    "Q38-8": "qwen3_8", "Q38-4": "qwen3_8",
}
TASKS = ("gpqa_en", "gpqa_de", "mmlu_en", "mmlu_de", "ifbench", "rgb_cb", "rgb_forced",
         "rgb_neg", "rgb_fact", "aime_en", "aime_de")
H2_ROWS_FULL = ("gpqa_en", "gpqa_de", "mmlu_en", "mmlu_de", "ifbench")
H2_SECONDARY = ("aime_en", "aime_de")
H7_ROWS_BASE = ("mmlu_en", "mmlu_de", "ifbench", "rgb_cb")
H2_SHARED_ROWS = ("mmlu_en", "mmlu_de", "ifbench")
FORCED_PLANS = ("P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7")
TRUNCATION_SENSITIVITY = ("H3", "H4", "H6", "H7")
GPQA_ROWS = ("gpqa_en", "gpqa_de")
GIB = float(2 ** 30)
# Threshold comparisons on fractions carry floating-point error (e.g. 0.71 - 0.79 = -0.08000000000000007). Every
# rule compares with this tolerance in the direction the rule's own wording gives: "> 8 pp" is not met by exactly
# 8 pp, "within +/- 2 pp" is met by exactly 2 pp (review fix 2026-10-03).
EPS = 1e-9

# Which side of the bootstrap distribution bounds each claim (confirm, refute).
BOUND_SIDES = {
    "H1": ("lower", "upper"), "H2": ("lower", "upper"), "H3": ("upper", "lower"),
    "H4": ("lower", "upper"), "H5": ("lower", "upper"), "H6": ("upper", "lower"),
    "H7": ("lower", "upper"), "H8": ("upper", "lower"),
}

# Pre-registered plain-answer wording (HYPOTHESIS "Plain answers"), verbatim.
PLAIN_ANSWERS = {
    "Q1_always": "Kolibri runs on an Apple M5 Max (128 GB) through our gated MLX port: <K8 tok/s> tok/s at 8-bit and <K4 tok/s> at 4-bit, batch 1 (M5 Max, MLX 0.31.2).",
    "Q1_H1": "At 4-bit it decodes at <r> × Gemma 4's speed [CI]: <H1 phrase> three quarters of it.",
    "Q1_H1_phrases": {"CONFIRMED": "at least", "REFUTED": "below", "INCONCLUSIVE": "we cannot tell whether it reaches"},
    "Q1_D1": "On a 64 GB M4 Pro node the 4-bit build <D1 phrase>, if the node runs nothing else. We did not measure speed there; the 4-bit-vs-Gemma ratio is the number that transfers.",
    "Q1_D1_phrases": {"CONFIRMED": "fits a 64k context with 10 % headroom", "INCONCLUSIVE": "fits only without headroom", "REFUTED": "does not fit"},
    "Q2_confirmed": "On average across the five public rows we could check, at MLX 8-bit and k = 1, Kolibri scores no more than 4 pp below its own scorecard (D̄ = <x> [CI]); three of the five rows are our reconstructions.",
    "Q2_confirmed_named_rows": "Any row whose 95 % upper bound is < −8 pp is named in the same sentence.",
    "Q2_refuted": "On the public rows we could check, at MLX 8-bit, Kolibri scores <x> pp below its own table while Gemma 4 and Qwen3.6 land within <y> pp of theirs. The port passed its gate; grid and protocol effects are bounded by the peers.",
    "Q2_peers_outside": "Our local protocol scores every model <lower/higher> than the vendor's table: Kolibri by <x> pp, the peers by <y> pp; Kolibri's gap relative to the peers is <DiD> [CI].",
    "Q2_inconclusive": "We can't tell at this sample size: Kolibri is <x> pp [CI] from its own table, on average across the public rows.",
    "Q2_peer_control_not_run": "Kolibri is <x> pp [CI] from its own table on the public rows (H2 <verdict>), but the peer control did not run (<reason>), so we cannot separate Kolibri from our local protocol; this is not a headline.",
    "Q3_always": "In English and German on MMLU-ProX-Lite, Kolibri 8-bit scores <x> pp [CI] against the mean of Gemma 4 and Qwen3.6 (EN <a>, DE <b>) — <H3 verdict>. On IFBench it is <y> pp ahead of Qwen3.6 (<H4 verdict>) and <z> pp against Gemma 4. Its tokenizer packs German <r> × denser than both (<H5 verdict>). The German comparison rests on <MMLU-ProX DE / plus GPQA-D DE and AIME DE>.",
    "Q3_e1_label": "excludes the rows where the vendor reports Kolibri ahead of these peers (AIME, GPQA)",
    "Q4_knows_less": "Kolibri knows less without documents: it answers <x> pp fewer RGB questions correctly than the peers, and still <y> pp fewer when told to always answer. Of the questions it gets wrong, <A> are abstentions: it <says so / guesses / unclear>.",
    "Q4_confirmed_rate": "Without documents, Kolibri's closed-book correct-answer rate is <x> pp below the peers'; <A> of its non-correct answers are abstentions (<E2 reading>).",
    "Q4_refuted": "We did not reproduce the vendor's closed-book gap: <x> pp [CI].",
    "Q4_inconclusive": "We can't tell: <x> pp [CI].",
    "Q5_both": "Squeezing Kolibri to 4-bit costs <x> pp on the task rows (<H7 verdict>), and its logit fidelity loss is <r> × the peers' on our six gate texts (<H8 verdict>).",
    "Q5_not_run": "We could not measure the task cost / fidelity loss of 4-bit (<reason>); D1 says only whether it fits.",
    "not_run": "<H> was not run (<reason>); we make no claim about it.",
}


class NotRun(Exception):
    """A hypothesis cannot be computed from the data present (missing or incomplete cell)."""


class ConfigError(Exception):
    """The plan or the analysis configuration is missing something the rules need."""


class RefuseError(Exception):
    """verdicts.py refuses to run (e.g. no matching mini re-score)."""


# --------------------------------------------------------------------------
# Small I/O helpers
# --------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    """Records of a JSONL file; header lines are kept (callers filter); an
    unparsable (torn) line is an error here, because scores are complete files."""
    out = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ConfigError(f"{path}: line {n} is not JSON ({e})") from None
    return out


def newest(paths: Iterable[Path]) -> Path | None:
    """The newest <UTC>-stamped file (UTC stamps sort lexicographically)."""
    ps = sorted(paths, key=lambda p: p.name)
    return ps[-1] if ps else None


def load_margins(path: Path | None = None) -> dict:
    return read_json(path or ANALYSIS_DIR / "margins.json")


def load_vendor(path: Path | None = None) -> dict:
    return read_json(path or ANALYSIS_DIR / "vendor_values.json")


def vendor_value(vendor: dict, row: str, arm: str) -> float:
    """Vendor value as a fraction for the arm's model family on a row."""
    return float(vendor["rows"][row]["values"][ARM_FAMILY[arm]]) / 100.0


def vendor_sd(vendor: dict, row: str, arm: str) -> float:
    """sd of the vendor's avg@N term: sqrt(p_v (1 - p_v) / (N_v n_v)); 0 where the vendor
    ran a set once (N_v = 1; HYPOTHESIS H2: "it is 0 for MMLU")."""
    r = vendor["rows"][row]
    N_v, n_v = int(r.get("N_v") or 1), r.get("n_v")
    if N_v <= 1 or not n_v:
        return 0.0
    p = vendor_value(vendor, row, arm)
    return math.sqrt(p * (1.0 - p) / (N_v * int(n_v)))


# --------------------------------------------------------------------------
# Scores
# --------------------------------------------------------------------------

class Item:
    __slots__ = ("score", "truncated", "category", "parse_ok", "strict", "subset", "reasoning_lang")

    def __init__(self, score, truncated, category, parse_ok, strict, subset, reasoning_lang=None):
        self.score = score
        self.truncated = truncated
        self.category = category
        self.parse_ok = parse_ok
        self.strict = strict
        self.subset = subset
        self.reasoning_lang = reasoning_lang


def _to_item(rec: dict, task: str) -> Item:
    truncated = bool(rec.get("truncated", False))
    base = rec.get("correct_loose", rec.get("correct")) if task == "ifbench" else rec.get("correct")
    correct = bool(base) and not truncated
    strict = rec.get("correct_strict")
    strict = None if strict is None else float(bool(strict) and not truncated)
    status = rec.get("parse_status", "ok")
    return Item(1.0 if correct else 0.0, truncated, rec.get("category"),
                status in (None, "ok"), strict, rec.get("subset"), rec.get("reasoning_lang"))


class Scores:
    """Score records indexed by (arm, task, effort, pass) -> {item_id: Item}."""

    def __init__(self, records: Iterable[dict] = ()):
        self._cells: dict[tuple, dict[str, Item]] = {}
        self.duplicates = 0
        for rec in records:
            self.add(rec)

    def add(self, rec: dict) -> None:
        if rec.get("type") == "header":
            return
        key = rec.get("key")
        if not isinstance(key, dict):
            raise ConfigError(f"score record without a key: {sorted(rec)}")
        ck = (str(key["arm"]), str(key["task"]), str(key.get("effort", "high")), int(key.get("pass", 0)))
        item = str(key["item"])
        cell = self._cells.setdefault(ck, {})
        if item in cell:
            self.duplicates += 1   # BUILD_SPEC §5.4: only the first complete record counts
            return
        cell[item] = _to_item(rec, ck[1])

    @classmethod
    def from_dir(cls, scores_dir: Path) -> tuple["Scores", dict[str, str]]:
        s = cls()
        shas: dict[str, str] = {}
        for p in sorted(Path(scores_dir).rglob("*.jsonl")):
            shas[p.relative_to(Path(scores_dir).parent).as_posix()] = sha256_file(p)
            for rec in read_jsonl(p):
                s.add(rec)
        return s, shas

    def cell(self, arm: str, task: str, effort: str = "high", pass_: int = 0) -> dict[str, Item] | None:
        return self._cells.get((arm, task, effort, pass_))

    def keys(self) -> list[tuple]:
        return sorted(self._cells)


# --------------------------------------------------------------------------
# Bench and other inputs
# --------------------------------------------------------------------------

BENCH_FILES = {
    "speed": ("bench", "speed_*.jsonl"),
    "speed_desc": ("bench", "speed_desc_*.jsonl"),
    "fit": ("bench", "fit_*.jsonl"),
    "c1": ("bench", "c1_*.jsonl"),
    "ladder": ("bench", "ladder_*.jsonl"),
    "tokenizer": (".", "tokenizer_*.json"),
    "kl": (".", "kl_8v4_*.json"),
    "gate": ("gate", "gate_*.json"),
}


def _complete(kind_path: Path, content) -> bool:
    """bench/common.py: a JSONL cell is complete iff its last record is {"type": "end", "complete": true};
    a JSON output iff its top-level "complete" is true (a gate record has no such flag)."""
    if kind_path.suffix == ".jsonl":
        return bool(content) and content[-1].get("type") == "end" and bool(content[-1].get("complete"))
    return bool(content.get("complete", True)) if isinstance(content, dict) else False


def load_bench(results_dir: Path) -> tuple[dict, dict[str, str]]:
    """Newest *complete* file of each bench kind -> parsed content (header and end records
    dropped); also {relpath: sha256}. A kind whose files are all incomplete is listed in
    bench["_incomplete"], and the hypotheses that need it are NOT RUN."""
    results_dir = Path(results_dir)
    out: dict = {"_incomplete": []}
    shas: dict[str, str] = {}
    for kind, (sub, pattern) in BENCH_FILES.items():
        cands = [p for p in (results_dir / sub).glob(pattern) if p.is_file()]
        if kind == "speed":   # speed_*.jsonl must not pick up speed_desc_*.jsonl
            cands = [p for p in cands if not p.name.startswith("speed_desc_")]
        if kind == "gate":
            cands = [p for p in cands if not p.name.endswith(("_layers.csv", "_mutants.json"))]
        chosen = None
        for p in sorted(cands, key=lambda q: q.name, reverse=True):
            content = read_jsonl(p) if p.suffix == ".jsonl" else read_json(p)
            if _complete(p, content):
                chosen = (p, content)
                break
        if chosen is None:
            if cands:
                out["_incomplete"].append(kind)
            continue
        p, content = chosen
        shas[p.relative_to(results_dir).as_posix()] = sha256_file(p)
        if p.suffix == ".jsonl":
            out[kind] = [r for r in content if r.get("type") not in ("header", "end")]
            out[f"{kind}_summary"] = content[-1].get("summary")
        else:
            out[kind] = content
    return out, shas


def _require_bench(ctx: "Context", kind: str, what: str):
    if kind in ctx.bench.get("_incomplete", []):
        raise NotRun(f"{what}: the {kind} bench file is incomplete (no end record with complete: true)")
    data = ctx.bench.get(kind)
    if not data:
        raise NotRun(f"{what}: no {kind} bench file")
    return data


def _is_run(r: dict) -> bool:
    """A measurement record: kind "run" (bench/common.py layout); warm-ups and summaries are not."""
    if r.get("type") in ("header", "end"):
        return False
    k = r.get("kind")
    return k == "run" if k is not None else not r.get("warmup")


def speed_runs(recs: list[dict]) -> dict[int, dict[str, float]]:
    """H1 runs: block -> {arm: generation_tps} (bench/speed.py "run" records; the first per block and arm)."""
    blocks: dict[int, dict[str, float]] = {}
    for r in recs:
        if not _is_run(r) or r.get("arm") not in ("K4", "G4") or r.get("block") is None:
            continue
        blocks.setdefault(int(r["block"]), {}).setdefault(r["arm"], float(r["generation_tps"]))
    return blocks


def fit_runs(recs: list[dict]) -> list[dict]:
    """D1 runs: [{"arm", "n_tokens", "peak_bytes", "prefill_tps"}] (bench/fit.py: n_context, peak_bytes, prompt_tps)."""
    out = []
    for r in recs:
        if not _is_run(r):
            continue
        n = r.get("n_context", r.get("n_tokens"))
        peak = r.get("peak_bytes", r.get("peak_memory_bytes"))
        if n is None or peak is None:
            continue
        out.append({"arm": r.get("arm", "K4"), "n_tokens": int(n), "peak_bytes": int(peak),
                    "prefill_tps": r.get("prompt_tps", r.get("prefill_tps")), "rep": r.get("rep")})
    return out


def tokenizer_docs(tok: dict) -> dict:
    """H5 per-document columns -> numpy arrays {"bytes", "chars", "digits", "tokens": {name: array}}.

    bench/tokenizer_ratio.py writes {"de": {"columns": [index, bytes, chars, digits, tokens_<name>...],
    "docs": [[...], ...], "tokenizer_identity": {...}}, "en_descriptive": {...}}; a list of
    {"bytes", "chars", "digits", "tokens": {name: n}} dicts is accepted too.
    """
    de = tok.get("de", tok)
    docs = de.get("docs") or []
    if not docs:
        raise NotRun("H5: the tokenizer record holds no documents")
    if isinstance(docs[0], (list, tuple)):
        col = {c: i for i, c in enumerate(de["columns"])}
        arr = np.array(docs, dtype=np.float64)
        names = [c[len("tokens_"):] for c in de["columns"] if c.startswith("tokens_")]
        return {"bytes": arr[:, col["bytes"]], "chars": arr[:, col["chars"]],
                "digits": arr[:, col["digits"]] if "digits" in col else None,
                "tokens": {n: arr[:, col[f"tokens_{n}"]] for n in names}}
    names = sorted(docs[0]["tokens"])
    return {"bytes": np.array([d["bytes"] for d in docs], dtype=np.float64),
            "chars": np.array([d.get("chars", 0) for d in docs], dtype=np.float64),
            "digits": (np.array([d["digits"] for d in docs], dtype=np.float64)
                       if all("digits" in d for d in docs) else None),
            "tokens": {n: np.array([d["tokens"][n] for d in docs], dtype=np.float64) for n in names}}


def kl_models(kl: dict, fp32: bool = False) -> dict:
    """H8 blocks -> {family: {text: [{"kl", "bytes", "tokens"}]}}.

    bench/kl_8v4.py writes {"families": {family: {"texts": {T: {"blocks": [{"block", "bytes",
    "tokens", "kl_sum", "kl_fp32_sum" (Kolibri), ...}]}}}}}; {"models": {family: {T: [{"kl",
    "bytes", "tokens"}]}}} is accepted too. fp32=True reads Kolibri's fp32-logit KL.
    """
    key = "kl_fp32_sum" if fp32 else "kl_sum"
    if "families" in kl:
        out = {}
        for fam, f in kl["families"].items():
            texts = {}
            for t, rec in (f.get("texts") or {}).items():
                blocks = rec.get("blocks") or []
                if blocks and all(key in b for b in blocks):
                    texts[t] = [{"kl": b[key], "bytes": b["bytes"], "tokens": b["tokens"]} for b in blocks]
            if texts:
                out[fam] = texts
        return out
    if fp32:
        return {"kolibri": (kl.get("models_fp32") or {}).get("kolibri")} if (kl.get("models_fp32") or {}).get("kolibri") else {}
    return kl.get("models") or {}


def check_rescore(results_dir: Path) -> dict:
    """Refuse unless the newest rescore_mini_<UTC>.json covers every scores file byte for byte.

    HYPOTHESIS "What counts as evidence": the mini re-scores every set and its
    scores must match the mbp's byte for byte; verdicts.py refuses otherwise.
    IFBench is scored on the mini only ("mini_only": true, sha256_mbp null).
    """
    results_dir = Path(results_dir)
    files = sorted((results_dir / "scores").rglob("*.jsonl"))
    if not files:
        raise RefuseError("no results/scores/**/*.jsonl to compute verdicts from")
    rp = newest(results_dir.glob("rescore_mini_*.json"))
    if rp is None:
        raise RefuseError("no results/rescore_mini_<UTC>.json: the mini has not re-scored the sets")
    data = read_json(rp)
    entries = data.get("files", [])
    if isinstance(entries, dict):
        entries = [dict(v, path=k) for k, v in entries.items()]
    by_path = {e["path"]: e for e in entries}
    problems = []
    for f in files:
        rel = f.relative_to(results_dir).as_posix()
        e = by_path.get(rel)
        if e is None:
            problems.append(f"{rel}: not in {rp.name}")
            continue
        sha = sha256_file(f)
        if e.get("match") is False:
            problems.append(f"{rel}: re-score reports a mismatch")
        if e.get("sha256_mini") != sha:
            problems.append(f"{rel}: mini sha256 differs from the file")
        mbp = e.get("sha256_mbp")
        if mbp is None and not e.get("mini_only"):
            problems.append(f"{rel}: no mbp sha256 and not marked mini_only")
        if mbp is not None and mbp != sha:
            problems.append(f"{rel}: mbp sha256 differs from the mini re-score")
    if problems:
        raise RefuseError("rescore check failed:\n  " + "\n  ".join(problems))
    return {"file": rp.relative_to(results_dir).as_posix(), "sha256": sha256_file(rp), "n_files": len(files)}


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------

def normalise_plan(plan: Mapping | None) -> dict:
    """Fill the plan keys the verdict code reads with the defaults HYPOTHESIS implies.

    Reads runner/plan_fix.py's plan_fixed_<UTC>.json directly: "rows" {"H2", "H2_arm",
    "H2_secondary_aime", "H7"} and "queue" [{arm, task, effort, pass, n, ...}] map onto
    h2_rows, kolibri_primary, h2_secondary_rows, h7_rows and cells. Explicit keys win.
    """
    p = dict(plan or {})
    name = p.get("plan")
    tier_b = list(p.get("tier_b") or [])
    p["tier_b"] = tier_b
    rows = p.get("rows") if isinstance(p.get("rows"), dict) else {}
    if "H2" in rows:
        p.setdefault("h2_rows", list(rows["H2"]))
    if rows.get("H2_arm"):
        p.setdefault("kolibri_primary", rows["H2_arm"])
    if "H2_secondary_aime" in rows:
        p.setdefault("h2_secondary_rows", list(rows["H2_secondary_aime"]))
    if "H7" in rows and rows["H7"]:
        p.setdefault("h7_rows", list(rows["H7"]))
    # runner/plan_fix.py freezes the post-stratification counts (HYPOTHESIS rule 5) under "category_counts"
    # (a copy of tasks/manifests/mmlu_prox_category_counts.json: {"en": {...}, "de": {...}, "_meta": {...}}).
    cc = p.get("category_counts")
    if "post_strat_counts" not in p and isinstance(cc, Mapping) and any(k in cc for k in ("en", "de")):
        p["post_strat_counts"] = {k: v for k, v in cc.items() if k in ("en", "de")}
    queue = [c for c in (p.get("queue") or []) if c.get("arm") in ARM_FAMILY]
    if "cells" not in p and queue:
        p["cells"] = [{"arm": c["arm"], "task": c["task"], "effort": c.get("effort", "high"),
                       "pass": int(c.get("pass", 0)), "n": c.get("n")} for c in queue]
    if "peers" not in p and queue:
        # runner/plan_fix.py writes "peers" (the MoE peers left after any drop, review fix 2026-10-03); an older
        # plan without the key falls back to the peers that have Tier-A MMLU cells in the queue.
        p["peers"] = [a for a in ("G8", "Q36-8") if any(c["arm"] == a and c["task"].startswith("mmlu_")
                                                        for c in queue)]
    p.setdefault("family", list(FAMILY))
    p["family"] = list(p["family"])
    p.setdefault("m", len(p["family"]))
    if int(p["m"]) != len(p["family"]):
        raise ConfigError(f"plan m = {p['m']} but the family has {len(p['family'])} hypotheses")
    p.setdefault("kolibri_primary", "K8")
    p.setdefault("peers", ["G8", "Q36-8"])
    if "h2_rows" not in p:
        p["h2_rows"] = [r for r in H2_ROWS_FULL if not (name == "P10" and r == "gpqa_de")]
    if "h2_secondary_rows" not in p:
        p["h2_secondary_rows"] = list(H2_SECONDARY) if "B1" in tier_b else []
    if "h7_rows" not in p:
        b8 = [r for r in GPQA_ROWS if not (name == "P10" and r == "gpqa_de")]   # P10: no K8 GPQA-D DE cell
        p["h7_rows"] = list(H7_ROWS_BASE) + (b8 if "B8" in tier_b else [])
    if "forced_cells" not in p:
        p["forced_cells"] = name in FORCED_PLANS
    p.setdefault("negfc_cells", p.get("neg_fc_cells", name == "P0"))
    p.setdefault("b10", "B10" in tier_b)
    p.setdefault("efforts", {})
    p.setdefault("not_run", {})
    p.setdefault("cells", [])
    p.setdefault("h8_texts", ["T1", "T2", "T3", "T4", "T5", "T6"])
    p.setdefault("h8_peers", ["gemma4", "qwen3_6", "qwen3_8"])
    return p


# --------------------------------------------------------------------------
# Context: the inputs every hypothesis reads
# --------------------------------------------------------------------------

class Context:
    def __init__(self, results_dir, plan, scores: Scores, bench: dict, margins: dict, vendor: dict):
        self.results_dir = Path(results_dir) if results_dir is not None else None
        self.plan = normalise_plan(plan)
        self.scores = scores
        self.bench = bench or {}
        self.margins = margins
        self.vendor = vendor
        self._post_strat = None

    # -- arms
    @property
    def K(self) -> str:
        return self.plan["kolibri_primary"]

    @property
    def peers(self) -> list[str]:
        return list(self.plan["peers"])

    def effort(self, arm: str) -> str:
        return self.plan["efforts"].get(arm, "high")

    def B(self) -> int:
        return int(self.margins.get("B", stats.B_DEFAULT))

    # -- cells
    def expected_n(self, arm: str, task: str, pass_: int = 0) -> int | None:
        for c in self.plan["cells"]:
            if (c.get("arm") == arm and c.get("task") == task and int(c.get("pass", 0)) == pass_
                    and c.get("effort", "high") == self.effort(arm)):
                return int(c["n"]) if c.get("n") is not None else None
        return None

    def cell(self, arm: str, task: str, pass_: int = 0) -> dict[str, Item]:
        c = self.scores.cell(arm, task, self.effort(arm), pass_)
        if not c:
            raise NotRun(f"no scores for {arm} {task} (effort {self.effort(arm)}, pass {pass_})")
        n = self.expected_n(arm, task, pass_)
        if n is not None and len(c) < n:
            raise NotRun(f"{arm} {task}: {len(c)} of {n} items scored")
        return c

    def has_cell(self, arm: str, task: str, pass_: int = 0) -> bool:
        try:
            self.cell(arm, task, pass_)
            return True
        except NotRun:
            return False

    # -- item sets
    def primary_items(self, task: str, items: Iterable[str]) -> list[str]:
        """GPQA EN primary = every Diamond item (Amendment 3, 2026-10-04): eval-framework's over-long question is
        not in gpqa_diamond.csv, so its filter excludes nothing and the vendor ran all 198 too. The pre-registered
        197 primary / 198 sensitivity split (HYPOTHESIS C14) is gone. The exclusion record (plan gpqa_en_excluded,
        else the manifest's in_primary flags) is still read: empty is the expected record, and any excluded id
        contradicts the amended rule, so it is a ConfigError rather than a silent drop."""
        items = sorted(items)
        if task != "gpqa_en":
            return items
        excluded = self.gpqa_en_excluded()
        if excluded:
            raise ConfigError(f"GPQA EN exclusions {sorted(excluded)[:5]} ({len(excluded)}): Amendment 3 makes the "
                              "primary set every Diamond item (eval-framework's over-long filter excludes none)")
        return items

    def gpqa_en_excluded(self) -> list[str]:
        """The recorded GPQA EN exclusions: plan gpqa_en_excluded, else the manifest's, else none."""
        excluded = self.plan.get("gpqa_en_excluded")
        if excluded is None:
            excluded = self._gpqa_manifest_excluded()
        return [str(x) for x in (excluded or [])]

    def _gpqa_manifest_excluded(self) -> list[str] | None:
        """Item ids with in_primary == false in the GPQA EN manifest (tasks/build_manifests.py)."""
        base = Path(self.plan["manifests_dir"]) if self.plan.get("manifests_dir") else EXP_DIR / "tasks" / "manifests"
        f = base / "gpqa_diamond_en.json"
        if not f.is_file():
            return None
        doc = read_json(f)
        items = doc.get("items", [])
        if not items or any("in_primary" not in e for e in items):
            return None
        return [str(e["id"]) for e in items if not e["in_primary"]]

    def common_items(self, arms: Iterable[str], task: str, pass_: int = 0) -> list[str]:
        sets = [set(self.cell(a, task, pass_)) for a in arms]
        common = set.intersection(*sets) if sets else set()
        return self.primary_items(task, common)

    # -- post-stratification weights (HYPOTHESIS H2 Measurement)
    def post_strat(self, lang: str) -> dict[str, float]:
        if self._post_strat is None:
            counts = self.plan.get("post_strat_counts")
            if counts is None:
                f = EXP_DIR / "tasks" / "manifests" / "mmlu_prox_category_counts.json"
                if not f.is_file():
                    raise ConfigError("no post-stratification counts: plan post_strat_counts or "
                                      "tasks/manifests/mmlu_prox_category_counts.json")
                counts = read_json(f)
            self._post_strat = counts
        if lang not in self._post_strat:
            raise ConfigError(f"no post-stratification counts for language {lang!r}")
        return {str(k): float(v) for k, v in self._post_strat[lang].items()}


def _lang(task: str) -> str:
    return task.rsplit("_", 1)[-1]


def _is_mmlu(task: str) -> bool:
    return task.startswith("mmlu_")


def _scores_vec(cell: dict[str, Item], items: list[str]) -> np.ndarray:
    return np.array([cell[i].score for i in items], dtype=np.float64)


def _strata(cell_for_category: dict[str, Item], items: list[str], values: np.ndarray,
            stratify: bool, prefix: str = "") -> dict[str, np.ndarray]:
    """Group per-item values (rows aligned with items) by category, or one stratum."""
    if not stratify:
        return {prefix + "all": values}
    groups: dict[str, list[int]] = {}
    for idx, i in enumerate(items):
        cat = cell_for_category[i].category
        if cat is None:
            raise ConfigError(f"item {i} has no category for a stratified row")
        groups.setdefault(prefix + str(cat), []).append(idx)
    return {k: values[np.array(v)] for k, v in groups.items()}


def row_point(ctx: Context, arm: str, task: str, items: list[str] | None = None,
              cell: dict[str, Item] | None = None) -> float:
    """Our score on a row: post-stratified for the MMLU rows, the plain mean otherwise."""
    cell = cell if cell is not None else ctx.cell(arm, task)
    items = items if items is not None else ctx.primary_items(task, cell)
    v = _scores_vec(cell, items)
    if _is_mmlu(task):
        st = _strata(cell, items, v, True)
        return stats.post_stratified_mean(st, ctx.post_strat(_lang(task)))
    return float(np.mean(v))


def truncation_rate(cell: dict[str, Item], items: list[str]) -> float:
    return float(np.mean([cell[i].truncated for i in items])) if items else 0.0


def peer_D(ctx: Context, arm: str, row: str) -> float:
    """D_peer = local - vendor on a row, computed as H2's rows (post-stratified MMLU; E8)."""
    cell = ctx.cell(arm, row)
    items = ctx.primary_items(row, cell)
    return row_point(ctx, arm, row, items, cell) - vendor_value(ctx.vendor, row, arm)


def e8_flags(ctx: Context, arms: Iterable[str], rows: Iterable[str]) -> tuple[list[dict], list[dict]]:
    """(flags, all D_peer): |D_peer| > margins E8_flag on a row the H uses (HYPOTHESIS Common rules)."""
    thr = float(ctx.margins["E8_flag"])
    flags, values = [], []
    for arm in arms:
        for row in rows:
            try:
                d = peer_D(ctx, arm, row)
            except NotRun:
                values.append({"arm": arm, "row": row, "D_peer": None})
                continue
            values.append({"arm": arm, "row": row, "D_peer": d})
            if abs(d) > thr + EPS:
                flags.append({"arm": arm, "row": row, "D_peer": d})
    return flags, values


# --------------------------------------------------------------------------
# Result helpers
# --------------------------------------------------------------------------

def not_run(reason: str) -> dict:
    return {"status": NOT_RUN, "reason": reason, "p": 1.0, "p_rev": 1.0, "B": 0}


def _boot_result(samples: np.ndarray, estimate, theta0: float, null_side: str,
                 theta_rev: float, rev_null_side: str, B: int, level: float) -> dict:
    p = stats.p_one_sided(samples, theta0, null_side)
    p_rev = stats.p_one_sided(samples, theta_rev, rev_null_side)
    lo, hi = stats.percentile_ci(samples, level)
    return {
        "status": "RUN",
        "estimate": float(estimate),
        "ci95": [lo, hi],
        "p": p,
        "p_rev": p_rev,
        "B": B,
        "mc_se_p": stats.mc_se(p, B),
        "mc_se_p_rev": stats.mc_se(p_rev, B),
        "_samples": samples,
        "_bound": lambda a, side, s=samples: stats.one_sided_bound(s, a, side),
    }


# --------------------------------------------------------------------------
# H1 — speed class (t-test)
# --------------------------------------------------------------------------

def h1(ctx: Context, B: int) -> dict:
    blocks = speed_runs(_require_bench(ctx, "speed", "H1"))
    used = sorted(b for b, d in blocks.items() if "K4" in d and "G4" in d)
    want = int(ctx.margins.get("H1_blocks", 10))
    if len(used) != want:
        # HYPOTHESIS H1: 10 blocks, df = 9 (review fix 2026-10-03: another count is not the pre-registered test).
        raise NotRun(f"H1 needs {want} blocks with both K4 and G4 (df = {want - 1}); found {len(used)}")
    x = [math.log(blocks[b]["K4"] / blocks[b]["G4"]) for b in used]
    theta0 = math.log(float(ctx.margins["H1"]))
    level = float(ctx.margins["ci_level"])
    p, p_rev, (lo, hi) = stats.t_test_one_sample(x, theta0, level)
    ts = stats.t_stats(x, theta0)
    detail = {
        "statistic": "geometric-mean ratio exp(mean ln(tps K4 / tps G4)) over blocks",
        "n_blocks": len(used), "df": ts["df"], "t": ts["t"] if math.isfinite(ts["t"]) else None,
        "mean_log_ratio": ts["mean"], "sd_log_ratio": ts["sd"],
        "block_log_ratios": {str(b): v for b, v in zip(used, x)},
        "tps": {str(b): blocks[b] for b in used},
        "descriptive": _speed_descriptive(ctx),
    }
    return {
        "status": "RUN", "estimate": math.exp(ts["mean"]), "ci95": [math.exp(lo), math.exp(hi)],
        "p": p, "p_rev": p_rev, "B": 0, "method": "one-sample t-test, df = n - 1",
        "mc_se_p": 0.0, "mc_se_p_rev": 0.0, "detail": detail,
        "_bound": lambda a, side: math.exp(stats.t_bound(x, a, side)),
    }


def _speed_descriptive(ctx: Context) -> dict:
    """Prefill ratio, K8/G8, Qwen arms, aggregate decode at B in {1, 2, 4} (HYPOTHESIS H1 Descriptive).

    bench/speed.py speed_desc records: kind "decode_b1" (generation_tps), "prefill" (prompt_tps),
    "batch" (B, generation_tps = aggregate). Medians per arm.
    """
    recs = ctx.bench.get("speed_desc") or []
    by: dict[tuple[str, int], dict[str, list[float]]] = {}

    def slot(arm, b):
        return by.setdefault((str(arm), int(b)), {"generation_tps": [], "prompt_tps": [], "aggregate_tps": []})

    for r in recs:
        kind = r.get("kind")
        if r.get("type") in ("header", "end") or r.get("arm") is None:
            continue
        if kind == "decode_b1":
            slot(r["arm"], 1)["generation_tps"].append(float(r["generation_tps"]))
        elif kind == "prefill":
            slot(r["arm"], 1)["prompt_tps"].append(float(r["prompt_tps"]))
        elif kind == "batch":
            slot(r["arm"], r["B"])["aggregate_tps"].append(float(r["generation_tps"]))
    # K4 batch-1 decode comes from the H1 runs (speed_desc covers K8, G8, Q36-4/8, Q38-4/8).
    if ("K4", 1) not in by:
        k4 = [v["K4"] for v in speed_runs(ctx.bench.get("speed") or []).values() if "K4" in v]
        if k4:
            slot("K4", 1)["generation_tps"].extend(k4)
    out: dict = {"label": "M5 Max, MLX 0.31.2", "arms": {}}
    for (arm, b), d in sorted(by.items()):
        out["arms"].setdefault(arm, {})[f"batch_{b}"] = {
            f: (float(np.median(v)) if v else None) for f, v in d.items()} | {"n": max(len(v) for v in d.values())}
    for (a, g) in (("K8", "G8"), ("K4", "G4")):
        try:
            ka = out["arms"][a]["batch_1"]
            ga = out["arms"][g]["batch_1"]
            out[f"{a}_over_{g}_decode"] = ka["generation_tps"] / ga["generation_tps"]
            if ka.get("prompt_tps") and ga.get("prompt_tps"):
                out[f"{a}_over_{g}_prefill"] = ka["prompt_tps"] / ga["prompt_tps"]
        except (KeyError, TypeError, ZeroDivisionError):
            pass
    summ = ctx.bench.get("speed_summary") or {}
    if summ.get("prefill_tps_median"):
        pm = summ["prefill_tps_median"]
        out["h1_cell_prefill_tps_median"] = pm
        if pm.get("K4") and pm.get("G4"):
            out["K4_over_G4_prefill_4096"] = pm["K4"] / pm["G4"]
    return out


# --------------------------------------------------------------------------
# D1 — fit on a 64 GB node (deterministic, outside Holm)
# --------------------------------------------------------------------------

def d1(ctx: Context) -> dict:
    if "D1" in ctx.plan["not_run"]:
        return {"verdict": NOT_RUN, "reason": ctx.plan["not_run"]["D1"]}
    if "fit" in ctx.bench.get("_incomplete", []):
        return {"verdict": NOT_RUN, "reason": "the fit bench file is incomplete"}
    recs = [r for r in fit_runs(ctx.bench.get("fit") or []) if r["arm"] == "K4"]
    n64 = int(ctx.margins["D1_n_tokens"])
    n32 = int(ctx.margins["D1_descriptive_n_tokens"])
    peaks64 = [r["peak_bytes"] for r in recs if r["n_tokens"] == n64]
    if not peaks64:
        return {"verdict": NOT_RUN, "reason": "no K4 fit records at 64k (results/bench/fit_<UTC>.jsonl)"}
    reps = int(ctx.margins.get("D1_reps", 3))
    if len(peaks64) != reps:
        # HYPOTHESIS D1: 3 reps at 64k (review fix 2026-10-03).
        return {"verdict": NOT_RUN, "reason": f"{len(peaks64)} K4 fit records at 64k; the rule needs {reps}"}
    med = float(np.median(peaks64)) / GIB
    conf, ref = float(ctx.margins["D1_confirm_gib"]), float(ctx.margins["D1_refute_gib"])
    if med <= conf:
        v = CONFIRMED
    elif med > ref:
        v = REFUTED
    else:
        v = INCONCLUSIVE
    peaks32 = [r["peak_bytes"] for r in recs if r["n_tokens"] == n32]
    return {
        "verdict": v,
        "label": v if v != INCONCLUSIVE else "INCONCLUSIVE (fits at 64k only without the 10 % headroom)",
        "median_peak_gib_64k": med,
        "peaks_gib_64k": [x / GIB for x in peaks64],
        "median_peak_gib_32k": float(np.median(peaks32)) / GIB if peaks32 else None,
        "prefill_tps": {str(n): [r["prefill_tps"] for r in recs if r["n_tokens"] == n]
                        for n in sorted({r["n_tokens"] for r in recs})},
        "thresholds_gib": {"confirm_le": conf, "refute_gt": ref},
        "n_reps_64k": len(peaks64),
        "label_hardware": "M5 Max, MLX 0.31.2; transfer to the M4 Pro is an assumption (C18)",
    }


# --------------------------------------------------------------------------
# H2 — scorecard
# --------------------------------------------------------------------------

def _h2_row_spec(ctx: Context, arm: str, row: str, exclude_truncated: bool = False) -> dict:
    cell = ctx.cell(arm, row)
    items = ctx.primary_items(row, cell)
    if exclude_truncated:
        items = [i for i in items if not cell[i].truncated]
    if not items:
        raise NotRun(f"{arm} {row}: no items")
    spec = {
        "name": row,
        "vendor": vendor_value(ctx.vendor, row, arm),
        "vendor_sd": vendor_sd(ctx.vendor, row, arm),
        "weights": ctx.post_strat(_lang(row)) if _is_mmlu(row) else None,
    }
    # Two passes iff the B10 pass-1 cell of this row is complete (HYPOTHESIS H2: "If B10 runs"), decided per row
    # from what ran, not from the plan: an unstarted or aborted pass 1 leaves the row single-pass, recorded
    # (review fix 2026-10-03).
    cell2 = ctx.cell(arm, row, pass_=1) if row in GPQA_ROWS and ctx.has_cell(arm, row, 1) else None
    if row in GPQA_ROWS and ctx.plan["b10"] and cell2 is None:
        spec["two_pass_note"] = f"B10 pass 1 of {arm} {row} is missing or incomplete: single pass"
    all_items = ctx.primary_items(row, cell)
    if cell2 is not None:
        both = [i for i in items if i in cell2]
        if exclude_truncated:
            both = [i for i in both if not cell2[i].truncated]
            if not both:
                raise NotRun(f"{arm} {row}: no items left in both passes")
        x1, x2 = _scores_vec(cell, both), _scores_vec(cell2, both)
        spec["two_pass"] = (float(np.mean((x1 + x2) / 2.0)), stats.two_pass_row_var(x1, x2))
        spec["point"] = spec["two_pass"][0]
        spec["n"] = len(both)
        both_all = [i for i in all_items if i in cell2]
        spec["truncation_rate"] = (truncation_rate(cell, both_all) + truncation_rate(cell2, both_all)) / 2.0
        if exclude_truncated:
            spec.pop("two_pass")   # the descriptive excluding-truncated point needs no variance
        return spec
    v = _scores_vec(cell, items)
    spec["strata"] = _strata(cell, items, v, _is_mmlu(row))
    spec["point"] = (stats.post_stratified_mean(spec["strata"], spec["weights"])
                     if _is_mmlu(row) else float(np.mean(v)))
    spec["n"] = len(items)
    spec["truncation_rate"] = truncation_rate(cell, all_items)
    return spec


def h2_like(ctx: Context, arm: str, rows: list[str], B: int, h_id: str,
            exclude_truncated: bool = False) -> dict:
    """D-bar over rows for one arm, with the H2 bootstrap (also used for sensitivities and E8)."""
    specs = [_h2_row_spec(ctx, arm, r, exclude_truncated) for r in rows]
    boot = stats.boot_mean_rows(specs, B, stats.rng(h_id))
    point_rows = {s["name"]: s["point"] - s["vendor"] for s in specs}
    point = float(np.mean(list(point_rows.values())))
    level = float(ctx.margins["ci_level"])
    rows_out = {}
    for s in specs:
        lo, hi = stats.percentile_ci(boot["rows"][s["name"]], level)
        rows_out[s["name"]] = {
            "name": ctx.vendor["rows"][s["name"]]["name"],
            "ours": s["point"], "vendor": s["vendor"], "D": point_rows[s["name"]], "ci95": [lo, hi],
            "n": s["n"], "label": ctx.vendor["rows"][s["name"]]["label"],
            "vendor_page": ctx.vendor["rows"][s["name"]]["page"],
            "truncation_rate": s["truncation_rate"],
            "two_pass": s.get("two_pass") is not None,
            **({"two_pass_note": s["two_pass_note"]} if s.get("two_pass_note") else {}),
        }
    return {"point": point, "samples": boot["mean"], "rows": rows_out}


def h2(ctx: Context, B: int) -> dict:
    m = ctx.margins
    rows = list(ctx.plan["h2_rows"])
    core = h2_like(ctx, ctx.K, rows, B, "H2")
    res = _boot_result(core["samples"], core["point"], float(m["H2"]), "le", float(m["H2"]), "ge",
                       B, float(m["ci_level"]))
    detail: dict = {"rows": core["rows"], "margin": float(m["H2"]), "arm": ctx.K,
                    "title": "at MLX 4-bit" if ctx.K == "K4" else "at MLX 8-bit", "k": 1,
                    "n_rows": len(rows),
                    "reconstructed_rows": [r for r in rows if ctx.vendor["rows"][r]["label"] == "R"]}
    # Per-row result excluding truncated items (C7; key-numbers table).
    for r in rows:
        try:
            spec = _h2_row_spec(ctx, ctx.K, r, exclude_truncated=True)
            detail["rows"][r]["D_excluding_truncated"] = spec["point"] - spec["vendor"]
            detail["rows"][r]["n_excluding_truncated"] = spec["n"]
        except (NotRun, ConfigError, ValueError) as e:
            # e.g. every item of one weighted MMLU category truncated: the descriptive number is not computable;
            # the verdicts still are (review fix 2026-10-03).
            detail["rows"][r]["D_excluding_truncated"] = None
            detail["rows"][r]["D_excluding_truncated_reason"] = f"{type(e).__name__}: {e}"
    named = float(m["H2_row_named_abs"])
    detail["rows_abs_D_gt_5pp"] = [r for r in rows if abs(detail["rows"][r]["D"]) > named + EPS]
    upper = float(m["H2_row_upper_named"])
    detail["rows_upper_bound_below_minus_8pp"] = [r for r in rows if detail["rows"][r]["ci95"][1] < upper - EPS]
    # GPQA EN primary set (Amendment 3): every Diamond item; the pre-registered GPQA-198 sensitivity (C14) is
    # dropped, because eval-framework's over-long filter excludes no Diamond item (197 and 198 are the same set).
    if "gpqa_en" in rows:
        detail["gpqa_en_primary"] = {
            "n": detail["rows"]["gpqa_en"]["n"], "excluded": ctx.gpqa_en_excluded(),
            "rule": "every GPQA Diamond EN item (Amendment 3); the 197-vs-198 sensitivity (C14) is dropped"}
    detail["protocol_control"] = protocol_control(ctx, B)
    # Secondary 7-row estimate with AIME (B1): estimate and 95 % CI only, no verdict.
    sec = list(ctx.plan["h2_secondary_rows"])
    if sec:
        try:
            s = h2_like(ctx, ctx.K, rows + sec, B, "H2|secondary")
            lo, hi = stats.percentile_ci(s["samples"], float(m["ci_level"]))
            detail["secondary_with_aime"] = {"estimate": s["point"], "ci95": [lo, hi],
                                             "rows": {r: s["rows"][r] for r in sec},
                                             "note": "estimate and 95 % CI only; no verdict"}
        except NotRun as e:
            detail["secondary_with_aime"] = {"status": NOT_RUN, "reason": str(e)}
    res["detail"] = detail
    return res


def protocol_control(ctx: Context, B: int) -> dict:
    """H2_detail protocol control (from E8): peers' mean D on the shared rows and DiD, joint bootstrap."""
    if not ctx.peers:
        return {"status": NOT_RUN, "reason": "no MoE peer left (peer check)"}   # review fix 2026-10-03
    arms = [ctx.K] + ctx.peers
    level = float(ctx.margins["ci_level"])
    gen = stats.rng("H2|protocol")
    try:
        row_boot, row_point_d = {}, {}
        for row in H2_SHARED_ROWS:
            items = ctx.common_items(arms, row)
            cells = {a: ctx.cell(a, row) for a in arms}
            mat = np.stack([_scores_vec(cells[a], items) for a in arms], axis=1)   # (n, k)
            st = _strata(cells[ctx.K], items, mat, _is_mmlu(row))
            w = ctx.post_strat(_lang(row)) if _is_mmlu(row) else None
            row_boot[row] = stats.boot_strata(st, B, gen, weights=w)        # (B, k)
            if _is_mmlu(row):
                pts = [stats.post_stratified_mean({k: v[:, j] for k, v in st.items()}, w) for j in range(len(arms))]
            else:
                pts = [float(np.mean(mat[:, j])) for j in range(len(arms))]
            row_point_d[row] = [pts[j] - vendor_value(ctx.vendor, row, a) for j, a in enumerate(arms)]
        d_samples = {}
        for row in H2_SHARED_ROWS:
            cols = []
            for j, a in enumerate(arms):
                sd = vendor_sd(ctx.vendor, row, a)
                vend = vendor_value(ctx.vendor, row, a) + (gen.normal(0.0, sd, size=B) if sd > 0 else 0.0)
                cols.append(row_boot[row][:, j] - vend)
            d_samples[row] = np.stack(cols, axis=1)
    except (NotRun, ConfigError) as e:
        return {"status": NOT_RUN, "reason": str(e)}
    stack = np.stack([d_samples[r] for r in H2_SHARED_ROWS], axis=0)   # (rows, B, k)
    k_shared = stack[:, :, 0].mean(axis=0)
    peers_shared = stack[:, :, 1:].mean(axis=(0, 2))
    did = k_shared - peers_shared
    pt = np.array([row_point_d[r] for r in H2_SHARED_ROWS])            # (rows, k)
    pk, pp = float(pt[:, 0].mean()), float(pt[:, 1:].mean())
    band = float(ctx.margins["headline_peer_band"])
    trunc = {r: {a: truncation_rate(ctx.cell(a, r), sorted(ctx.cell(a, r))) for a in arms} for r in H2_SHARED_ROWS}
    return {
        "rows": list(H2_SHARED_ROWS), "arms": arms, "truncation_rate": trunc,
        "kolibri_mean_D_shared": pk, "kolibri_ci95": list(stats.percentile_ci(k_shared, level)),
        "peers_mean_D_shared": pp, "peers_ci95": list(stats.percentile_ci(peers_shared, level)),
        "DiD": pk - pp, "DiD_ci95": list(stats.percentile_ci(did, level)),
        "per_row_D": {r: {a: row_point_d[r][j] for j, a in enumerate(arms)} for r in H2_SHARED_ROWS},
        "peers_within_band": abs(pp) <= band + EPS, "band": band,
    }


# --------------------------------------------------------------------------
# H3 — knowledge gap, EN and DE pooled
# --------------------------------------------------------------------------

def _h3_strata(ctx: Context, exclude_truncated: bool):
    K, peers = ctx.K, ctx.peers
    if not peers:
        raise NotRun("no MoE peer left")
    strata: dict[str, list[float]] = {}
    per_lang: dict[str, list[float]] = {}
    n_excl = 0
    for task in ("mmlu_en", "mmlu_de"):
        cells = {a: ctx.cell(a, task) for a in [K] + peers}
        items = ctx.common_items([K] + peers, task)
        for i in items:
            if exclude_truncated and any(cells[a][i].truncated for a in cells):
                n_excl += 1
                continue
            d = cells[K][i].score - float(np.mean([cells[a][i].score for a in peers]))
            cat = cells[K][i].category
            strata.setdefault(f"{_lang(task)}|{cat}", []).append(d)
            per_lang.setdefault(_lang(task), []).append(d)
    if not strata:
        raise NotRun("H3: no items left")
    return {k: np.array(v) for k, v in strata.items()}, per_lang, n_excl


def h3(ctx: Context, B: int, exclude_truncated: bool = False, h_id: str = "H3") -> dict:
    m = ctx.margins
    strata, per_lang, n_excl = _h3_strata(ctx, exclude_truncated)
    point = float(np.mean(np.concatenate(list(strata.values()))))
    samples = stats.boot_paired(strata, B, stats.rng(h_id))
    res = _boot_result(samples, point, float(m["H3"]), "ge", float(m["H3_refute"]), "le", B, float(m["ci_level"]))
    res["n_excluded_truncated"] = n_excl
    if exclude_truncated:
        return res
    K, peers = ctx.K, ctx.peers
    detail: dict = {"arm": K, "peers": peers, "n_items": int(sum(len(v) for v in strata.values())),
                    "per_language_gap": {k: float(np.mean(v)) for k, v in per_lang.items()}}
    ps = {}
    drops = {}
    for a in [K] + peers:
        en = row_point(ctx, a, "mmlu_en", ctx.common_items([K] + peers, "mmlu_en"))
        de = row_point(ctx, a, "mmlu_de", ctx.common_items([K] + peers, "mmlu_de"))
        ps[a] = {"en": en, "de": de}
        drops[a] = en - de
    gap_ps = {lang: ps[K][lang] - float(np.mean([ps[a][lang] for a in peers])) for lang in ("en", "de")}
    detail["post_stratified_gap"] = dict(gap_ps, pooled=(gap_ps["en"] + gap_ps["de"]) / 2.0)
    detail["vendor_gap"] = ctx.vendor["vendor_gaps"]["H3_mmlu_kolibri_minus_peer_mean"]
    detail["en_to_de_drop_post_stratified"] = drops
    detail["vendor_en_to_de_drop"] = ctx.vendor["vendor_gaps"]["H3_en_to_de_drop"]
    detail["truncation_rate"] = {a: {t: truncation_rate(ctx.cell(a, t), sorted(ctx.cell(a, t)))
                                     for t in ("mmlu_en", "mmlu_de")} for a in [K] + peers}
    flags, vals = e8_flags(ctx, peers, ("mmlu_en", "mmlu_de"))
    res["e8_flags"] = flags
    detail["e8_D_peer"] = vals
    res["detail"] = detail
    return res


# --------------------------------------------------------------------------
# H4 — instruction following against Qwen3.6
# --------------------------------------------------------------------------

def h4(ctx: Context, B: int, exclude_truncated: bool = False, h_id: str = "H4") -> dict:
    m = ctx.margins
    K, Q = ctx.K, "Q36-8"
    if Q not in ctx.peers:
        raise NotRun("Qwen3.6 was dropped by the peer check (HYPOTHESIS: H4 is NOT RUN)")
    ck, cq = ctx.cell(K, "ifbench"), ctx.cell(Q, "ifbench")
    items = ctx.common_items([K, Q], "ifbench")
    n_excl = 0
    if exclude_truncated:
        keep = [i for i in items if not (ck[i].truncated or cq[i].truncated)]
        n_excl = len(items) - len(keep)
        items = keep
    if not items:
        raise NotRun("H4: no items left")
    d = _scores_vec(ck, items) - _scores_vec(cq, items)
    samples = stats.boot_paired({"all": d}, B, stats.rng(h_id))
    res = _boot_result(samples, float(np.mean(d)), float(m["H4"]), "le", float(m["H4_refute"]), "ge",
                       B, float(m["ci_level"]))
    res["n_excluded_truncated"] = n_excl
    if exclude_truncated:
        return res
    detail: dict = {"arm": K, "peer": Q, "n_items": len(items),
                    "kolibri_loose": float(np.mean(_scores_vec(ck, items))),
                    "qwen3_6_loose": float(np.mean(_scores_vec(cq, items))),
                    "vendor_difference": ctx.vendor["vendor_gaps"]["H4_ifbench_kolibri_minus_qwen3_6"]}
    if all(ck[i].strict is not None and cq[i].strict is not None for i in items):
        detail["strict_difference"] = float(np.mean([ck[i].strict - cq[i].strict for i in items]))
    if "G8" in ctx.peers and ctx.has_cell("G8", "ifbench"):
        cg = ctx.cell("G8", "ifbench")
        it = ctx.common_items([K, "G8"], "ifbench")
        dg = _scores_vec(ck, it) - _scores_vec(cg, it)
        sg = stats.boot_paired({"all": dg}, B, stats.rng("H4|gemma4"))
        detail["kolibri_vs_gemma4"] = {"D": float(np.mean(dg)), "ci95": list(stats.percentile_ci(sg)),
                                       "vendor_difference": ctx.vendor["vendor_gaps"]["H4_ifbench_kolibri_minus_gemma4"]}
    detail["truncation_rate"] = {K: truncation_rate(ck, items), Q: truncation_rate(cq, items)}
    flags, vals = e8_flags(ctx, [Q], ("ifbench",))
    res["e8_flags"] = flags
    detail["e8_D_peer"] = vals
    res["detail"] = detail
    return res


# --------------------------------------------------------------------------
# H5 — German tokenizer
# --------------------------------------------------------------------------

def h5(ctx: Context, B: int) -> dict:
    m = ctx.margins
    tok = _require_bench(ctx, "tokenizer", "H5")
    cols = tokenizer_docs(tok)
    for name in ("kolibri", "gemma4", "qwen3_6"):
        if name not in cols["tokens"]:
            raise NotRun(f"H5: no {name} token counts in the tokenizer record")
    nb, tk, tg, tq = cols["bytes"], cols["tokens"]["kolibri"], cols["tokens"]["gemma4"], cols["tokens"]["qwen3_6"]
    de = tok.get("de", tok)
    ident = (de.get("tokenizer_identity") or {}).get("qwen3_8_vs_qwen3_6", tok.get("qwen3_8_identical_to_qwen3_6"))
    # bytes/token(K) / bytes/token(X) = sum tok_X / sum tok_K (same documents, same bytes).
    num = np.stack([tg, tq, nb], axis=1)
    den = np.stack([tk, tk, tk], axis=1)
    samples = stats.boot_docs_ratio(num, den, B, stats.rng("H5"))
    thr = float(m["H5"])
    rg, rq = float(tg.sum() / tk.sum()), float(tq.sum() / tk.sum())
    p_g = stats.p_one_sided(samples[:, 0], thr, "le")
    p_q = stats.p_one_sided(samples[:, 1], thr, "le")
    pr_g = stats.p_one_sided(samples[:, 0], thr, "ge")
    pr_q = stats.p_one_sided(samples[:, 1], thr, "ge")
    p = max(p_g, p_q)                        # intersection-union test
    p_rev = min(1.0, 2.0 * min(pr_g, pr_q))  # Bonferroni over the two
    level = float(m["ci_level"])
    bpt = float(nb.sum() / tk.sum())
    lo_r, hi_r = m["H5_report_band"]
    lo_c, hi_c = m["H5_card_band"]
    sub = ("report-consistent" if lo_r <= bpt <= hi_r else
           "card-consistent" if lo_c <= bpt < hi_c else "neither")
    detail = {
        "n_docs": int(nb.size),
        "ratio_vs_gemma4": rg, "ratio_vs_gemma4_ci95": list(stats.percentile_ci(samples[:, 0], level)),
        "ratio_vs_qwen3_6": rq, "ratio_vs_qwen3_6_ci95": list(stats.percentile_ci(samples[:, 1], level)),
        "p_gemma4": p_g, "p_qwen3_6": p_q, "p_rev_gemma4": pr_g, "p_rev_qwen3_6": pr_q,
        "bytes_per_token": {"kolibri": bpt, "gemma4": float(nb.sum() / tg.sum()), "qwen3_6": float(nb.sum() / tq.sum())},
        "kolibri_bytes_per_token_ci95": list(stats.percentile_ci(samples[:, 2], level)),
        "sub_verdict": sub,
        "vendor": ctx.vendor["tokenizer"]["de_fineweb2"],
        "qwen3_8_vs_qwen3_6": ident,
        "en_descriptive": tok.get("en_descriptive", tok.get("en")),
    }
    if cols["digits"] is not None:
        share = float(m["H5_digit_dense_min_share"])
        ch = cols["chars"]
        dense = np.flatnonzero((ch > 0) & (cols["digits"] >= share * ch))
        if dense.size:
            idx = dense
            detail["digit_dense"] = {"n_docs": int(dense.size),
                                     "ratio_vs_gemma4": float(tg[idx].sum() / tk[idx].sum()),
                                     "ratio_vs_qwen3_6": float(tq[idx].sum() / tk[idx].sum())}
    lo_g, hi_g = detail["ratio_vs_gemma4_ci95"]
    lo_q, hi_q = detail["ratio_vs_qwen3_6_ci95"]
    s0, s1 = samples[:, 0], samples[:, 1]
    return {
        "status": "RUN", "estimate": min(rg, rq), "estimate_note": "the smaller of the two ratios",
        "ci95": [min(lo_g, lo_q), min(hi_g, hi_q)], "p": p, "p_rev": p_rev, "B": B,
        "mc_se_p": stats.mc_se(p, B), "mc_se_p_rev": stats.mc_se(p_rev, B), "detail": detail,
        "_bound": lambda a, side: min(stats.one_sided_bound(s0, a, side), stats.one_sided_bound(s1, a, side)),
    }


# --------------------------------------------------------------------------
# H6 — closed-book correct-answer deficit
# --------------------------------------------------------------------------

def _paired_vs_peers(ctx: Context, task: str, exclude_truncated: bool):
    K, peers = ctx.K, ctx.peers
    if not peers:
        raise NotRun("no MoE peer left")
    cells = {a: ctx.cell(a, task) for a in [K] + peers}
    items = ctx.common_items([K] + peers, task)
    n_excl = 0
    if exclude_truncated:
        keep = [i for i in items if not any(cells[a][i].truncated for a in cells)]
        n_excl = len(items) - len(keep)
        items = keep
    if not items:
        raise NotRun(f"{task}: no items left")
    d = _scores_vec(cells[K], items) - np.mean([_scores_vec(cells[a], items) for a in peers], axis=0)
    return cells, items, d, n_excl


def h6(ctx: Context, B: int, exclude_truncated: bool = False, h_id: str = "H6") -> dict:
    m = ctx.margins
    cells, items, d, n_excl = _paired_vs_peers(ctx, "rgb_cb", exclude_truncated)
    samples = stats.boot_paired({"all": d}, B, stats.rng(h_id))
    res = _boot_result(samples, float(np.mean(d)), float(m["H6"]), "ge", float(m["H6"]), "le",
                       B, float(m["ci_level"]))
    res["n_excluded_truncated"] = n_excl
    if exclude_truncated:
        return res
    K, peers = ctx.K, ctx.peers
    detail: dict = {"arm": K, "peers": peers, "n_items": len(items),
                    "correct_rate": {a: float(np.mean(_scores_vec(cells[a], items))) for a in cells},
                    "vendor": {a: vendor_value(ctx.vendor, "rgb_cb", a) for a in cells},
                    "vendor_gap": ctx.vendor["vendor_gaps"]["H6_rgb_cb_kolibri_minus_peer_mean"],
                    "truncation_rate": {a: truncation_rate(cells[a], items) for a in cells}}
    split = {}
    for a in cells:
        counts = {"correct": 0, "abstain": 0, "wrong": 0, "truncated": 0}
        for i in items:
            it = cells[a][i]
            if it.truncated:
                counts["truncated"] += 1
            elif it.score == 1.0:
                counts["correct"] += 1
            elif it.category == "abstain":
                counts["abstain"] += 1
            else:
                counts["wrong"] += 1
        split[a] = counts
    detail["four_way_split"] = split
    fact = [i for i in items if cells[K][i].subset == "en_fact"]
    if fact:
        detail["en_fact_subset"] = {"n": len(fact), "correct_rate": {
            a: float(np.mean(_scores_vec(cells[a], fact))) for a in cells}}
    # Wording state (C22): "knows less" needs the forced-answer deficit's 95 % upper bound < -10 pp.
    forced = {"ran": False}
    if ctx.plan["forced_cells"]:
        try:
            _, fi, fd, _ = _paired_vs_peers(ctx, "rgb_forced", False)
            fs = stats.boot_paired({"all": fd}, B, stats.rng("H6|forced"))
            lo, hi = stats.percentile_ci(fs, float(m["ci_level"]))
            forced = {"ran": True, "n_items": len(fi), "deficit": float(np.mean(fd)), "ci95": [lo, hi],
                      "upper_below_margin": hi < float(m["E2_knows_less_upper"])}
        except NotRun as e:
            forced = {"ran": False, "reason": str(e)}
    detail["forced"] = forced
    detail["wording"] = ("knows-less allowed if CONFIRMED" if forced.get("upper_below_margin")
                         else "closed-book correct-answer rate (no 'recall', 'knowledge', 'knows less')")
    flags, vals = e8_flags(ctx, peers, ("rgb_cb",))
    res["e8_flags"] = flags
    detail["e8_D_peer"] = vals
    res["detail"] = detail
    return res


# --------------------------------------------------------------------------
# H7 — 4-bit task cost
# --------------------------------------------------------------------------

def h7(ctx: Context, B: int, exclude_truncated: bool = False, h_id: str = "H7") -> dict:
    m = ctx.margins
    if ctx.K != "K8":
        raise NotRun("K8 cannot run; K4 is primary (HYPOTHESIS: H7 is NOT RUN)")
    rows = list(ctx.plan["h7_rows"])
    # The B8 rows (GPQA-D EN, DE) count where both K4 and K8 completed them: H7 is "averaged over the rows K4 runs"
    # and a Tier-B cell that never ran (or was aborted) drops its row, recorded, instead of voiding H7. A missing
    # Tier-A row still makes H7 NOT RUN (review fix 2026-10-03).
    dropped = {}
    for row in [r for r in rows if r in GPQA_ROWS]:
        if not (ctx.has_cell("K4", row) and ctx.has_cell("K8", row)):
            dropped[row] = "the B8 row did not run to completion for both K4 and K8"
    rows = [r for r in rows if r not in dropped]
    gen = stats.rng(h_id)
    row_samples, row_points, row_n, n_excl = [], {}, {}, 0
    for row in rows:
        c4, c8 = ctx.cell("K4", row), ctx.cell("K8", row)
        items = ctx.common_items(["K4", "K8"], row)
        if exclude_truncated:
            keep = [i for i in items if not (c4[i].truncated or c8[i].truncated)]
            n_excl += len(items) - len(keep)
            items = keep
        if not items:
            raise NotRun(f"H7 {row}: no items left")
        d = _scores_vec(c4, items) - _scores_vec(c8, items)
        st = _strata(c8, items, d, _is_mmlu(row))
        row_samples.append(stats.boot_strata(st, B, gen))
        row_points[row] = float(np.mean(d))
        row_n[row] = len(items)
    samples = np.mean(np.stack(row_samples, axis=0), axis=0)
    point = float(np.mean(list(row_points.values())))
    res = _boot_result(samples, point, float(m["H7"]), "le", float(m["H7"]), "ge", B, float(m["ci_level"]))
    res["n_excluded_truncated"] = n_excl
    if exclude_truncated:
        return res
    detail: dict = {"rows": {r: {"D": row_points[r], "n": row_n[r]} for r in rows},
                    "rows_dropped": dropped,
                    "truncation_rate": {a: {r: truncation_rate(ctx.cell(a, r), sorted(ctx.cell(a, r))) for r in rows}
                                        for a in ("K4", "K8")}}
    # Descriptive, confounded (model and bits): K4 vs G8 on the Tier-A rows.
    k4g8 = {}
    for row in H7_ROWS_BASE:
        try:
            it = ctx.common_items(["K4", "G8"], row)
            k4g8[row] = row_point(ctx, "K4", row, it) - row_point(ctx, "G8", row, it)
        except (NotRun, ConfigError):
            k4g8[row] = None
    detail["k4_vs_g8_confounded"] = {"label": "confounds model with bits", "rows": k4g8}
    res["detail"] = detail
    return res


# --------------------------------------------------------------------------
# H8 — 4-bit logit fidelity against the peers
# --------------------------------------------------------------------------

H8_FAMILY_ARMS = {"gemma4": ("G8", "G4"), "qwen3_6": ("Q36-8", "Q36-4"), "qwen3_8": ("Q38-8", "Q38-4")}


def _h8_arrays(models_blob: dict, models: list[str], texts: list[str]):
    num, den_b, den_t = [], [], []
    for t in texts:
        rows_kl, rows_b, rows_t = [], [], []
        for mdl in models:
            blocks = models_blob.get(mdl, {}).get(t)
            if not blocks:
                raise NotRun(f"H8: no KL blocks for {mdl} {t}")
            rows_kl.append([float(b["kl"]) for b in blocks])
            rows_b.append([float(b["bytes"]) for b in blocks])
            rows_t.append([float(b["tokens"]) for b in blocks])
        if len({len(r) for r in rows_kl}) != 1:
            raise ConfigError(f"H8: models disagree on the block count of {t}")
        b_arr = np.array(rows_b)
        if not np.all(b_arr == b_arr[0]):
            raise ConfigError(f"H8: models disagree on the bytes per block of {t} (blocks must be byte-aligned)")
        num.append(np.array(rows_kl))
        den_b.append(b_arr)
        den_t.append(np.array(rows_t))
    return num, den_b, den_t


def _ratio(per_model: np.ndarray) -> np.ndarray:
    """Kolibri (column 0) over the median of the peers (columns 1..)."""
    return per_model[..., 0] / np.median(per_model[..., 1:], axis=-1)


def h8(ctx: Context, B: int) -> dict:
    m = ctx.margins
    kl = _require_bench(ctx, "kl", "H8")
    texts = list(ctx.plan["h8_texts"])
    # A peer family leaves the median when either of its builds was dropped by the peer
    # check (plan "excluded_arms"); H8 is NOT RUN if no peer family remains (HYPOTHESIS H8,
    # resolution added 2026-10-04 before the pre-registration push).
    excluded = set((ctx.plan.get("excluded_arms") or {}).keys())
    peers = [f for f in ctx.plan["h8_peers"] if not (set(H8_FAMILY_ARMS.get(f, ())) & excluded)]
    if not peers:
        raise NotRun("H8: every peer family was dropped by the peer check")
    models = ["kolibri"] + peers
    num, den_b, den_t = _h8_arrays(kl_models(kl), models, texts)
    M = len(models)
    bytes_t = np.array([d[0].sum() for d in den_b])
    w_bytes = np.tile(bytes_t / bytes_t.sum(), (M, 1))
    point_models = stats.blocks_point(num, den_b, w_bytes)
    ratio = float(_ratio(point_models))
    samples_models = stats.boot_blocks_stratified(num, den_b, w_bytes, B, stats.rng("H8"))
    samples = _ratio(samples_models)
    thr = float(m["H8"])
    res = _boot_result(samples, ratio, thr, "ge", thr, "le", B, float(m["ci_level"]))
    per_text = {}
    for j, t in enumerate(texts):
        kpb = num[j].sum(axis=1) / den_b[j].sum(axis=1)
        per_text[t] = float(_ratio(kpb))
    n_ok = sum(1 for v in per_text.values() if v <= thr)
    res["confirm_condition"] = n_ok >= int(m["H8_texts_min"])
    # Per-token sensitivity (pre-registered, published next to it).
    tok_t = np.stack([d.sum(axis=1) for d in den_t], axis=1)          # (M, T)
    w_tok = tok_t / tok_t.sum(axis=1, keepdims=True)
    pt_tok = stats.blocks_point(num, den_t, w_tok)
    r_tok = float(_ratio(pt_tok))
    s_tok = _ratio(stats.boot_blocks_stratified(num, den_t, w_tok, B, stats.rng("H8|per_token")))
    detail = {
        "models": models, "texts": texts,
        "kl_per_byte": {mdl: float(v) for mdl, v in zip(models, point_models)},
        "per_text_ratio": per_text, "texts_at_or_below_threshold": n_ok,
        "per_text_condition": res["confirm_condition"],
        "per_token": {"ratio": r_tok, "ci95": list(stats.percentile_ci(s_tok, float(m["ci_level"]))),
                      "kl_per_token": {mdl: float(v) for mdl, v in zip(models, pt_tok)},
                      "exceeds_threshold": r_tok > thr},
        "estimand": "these six gate texts (2 EN, 4 DE), not English or German text in general",
    }
    fp32 = kl_models(kl, fp32=True)
    if fp32.get("kolibri"):
        try:
            nf, dbf, _ = _h8_arrays(fp32, ["kolibri"], texts)
            kpb32 = sum(n.sum() for n in nf) / sum(d.sum() for d in dbf)
            detail["kolibri_fp32_logits"] = {"kl_per_byte": float(kpb32),
                                             "ratio": float(kpb32 / np.median(point_models[1:]))}
        except (NotRun, ConfigError) as e:
            detail["kolibri_fp32_logits"] = {"status": NOT_RUN, "reason": str(e)}
    res["detail"] = detail
    return res


# --------------------------------------------------------------------------
# Family, Holm, decision
# --------------------------------------------------------------------------

COMPUTE = {"H1": h1, "H2": h2, "H3": h3, "H4": h4, "H5": h5, "H6": h6, "H7": h7, "H8": h8}
SENSITIVITY = {"H3": h3, "H4": h4, "H6": h6, "H7": h7}


def _run_one(ctx: Context, h: str, B: int) -> dict:
    if h in ctx.plan["not_run"]:
        return not_run(str(ctx.plan["not_run"][h]))
    try:
        return COMPUTE[h](ctx, B)
    except NotRun as e:
        return not_run(str(e))


def decide(results: Mapping[str, dict], family: list[str], margins: Mapping) -> dict:
    """Two Holm families and the verdict words (pure function of the p-values).

    results[h] needs "status", "p", "p_rev" and optionally "confirm_condition".
    """
    alpha = float(margins["alpha"])
    leans = float(margins["leans_refuted_p"])
    pc = {h: (float(results[h]["p"]) if results[h]["status"] != NOT_RUN else 1.0) for h in family}
    pr = {h: (float(results[h]["p_rev"]) if results[h]["status"] != NOT_RUN else 1.0) for h in family}
    hc = stats.holm(pc, alpha)
    hr = stats.holm(pr, alpha)
    out = {}
    for h in family:
        r = results[h]
        if r["status"] == NOT_RUN:
            word = NOT_RUN
        elif hc[h]["adjusted"] <= alpha and r.get("confirm_condition", True):
            word = CONFIRMED
        elif hr[h]["adjusted"] <= alpha:
            word = REFUTED
        else:
            word = INCONCLUSIVE
        out[h] = {
            "verdict": word,
            "leans_refuted": word == INCONCLUSIVE and pr[h] <= leans,
            "confirm": hc[h], "refute": hr[h],
        }
    return out


def _escalate_all(ctx: Context, results: dict, family: list[str]) -> list[str]:
    """Recompute with B_escalated every H whose p or p_rev is within a factor of 2 of its threshold."""
    m = ctx.margins
    escalated: list[str] = []
    factor = float(m["escalation_factor"])
    B_esc = int(m["B_escalated"])
    while True:
        dec = decide(results, family, m)
        todo = []
        for h in family:
            r = results[h]
            if h in escalated or r["status"] == NOT_RUN or r.get("B", 0) <= 0 or r["B"] >= B_esc:
                continue
            if (stats.near_threshold(r["p"], dec[h]["confirm"]["threshold"], factor)
                    or stats.near_threshold(r["p_rev"], dec[h]["refute"]["threshold"], factor)):
                todo.append(h)
        if not todo:
            return escalated
        for h in todo:
            old_B = results[h]["B"]
            results[h] = _run_one(ctx, h, B_esc)
            results[h]["escalated_from_B"] = old_B
            escalated.append(h)


def _label(word: str, leans: bool, flags: list[dict]) -> str:
    label = LEANS_REFUTED if (word == INCONCLUSIVE and leans) else word
    if flags and word != NOT_RUN:
        label += " — peer control failed (" + "; ".join(f"{f['arm']}, {f['row']}" for f in flags) + ")"
    return label


def _clean_result(r: dict) -> dict:
    return {k: v for k, v in r.items() if not k.startswith("_")}


def evaluate_family(ctx: Context) -> dict:
    """Compute H1-H8, escalate, run Holm, apply the sensitivity, flag and headline rules."""
    family = list(ctx.plan["family"])
    B = ctx.B()
    results = {h: _run_one(ctx, h, B) for h in family}
    escalated = _escalate_all(ctx, results, family)
    dec = decide(results, family, ctx.margins)
    alpha = float(ctx.margins["alpha"])
    out = {}
    for h in family:
        r, d = results[h], dec[h]
        flags = r.get("e8_flags", [])
        entry = _clean_result(r)
        entry.update({
            "verdict": d["verdict"],
            "label": _label(d["verdict"], d["leans_refuted"], flags),
            "p_adjusted": d["confirm"]["adjusted"], "p_rev_adjusted": d["refute"]["adjusted"],
            "holm_step": d["confirm"]["step"], "holm_step_rev": d["refute"]["step"],
            "holm_threshold": d["confirm"]["threshold"], "holm_threshold_rev": d["refute"]["threshold"],
            "leans_refuted": d["leans_refuted"], "e8_flags": flags,
            "seed": stats.seed_of(h),
        })
        if r["status"] != NOT_RUN and "_bound" in r and h in BOUND_SIDES:
            sc, sr = BOUND_SIDES[h]
            # alpha at the Holm step, i.e. the level the adjusted p is compared with.
            entry["bound_confirm"] = {"level": d["confirm"]["threshold"], "side": sc,
                                      "value": r["_bound"](d["confirm"]["threshold"], sc)}
            entry["bound_refute"] = {"level": d["refute"]["threshold"], "side": sr,
                                     "value": r["_bound"](d["refute"]["threshold"], sr)}
        # Paired truncation sensitivity (H3, H4, H6, H7).
        sens_word = None
        sens_failed = False
        if h in SENSITIVITY and r["status"] != NOT_RUN:
            try:
                s = SENSITIVITY[h](ctx, max(B, int(r.get("B", B))), exclude_truncated=True,
                                   h_id=f"{h}|truncation")
                sub = dict(results)
                sub[h] = s
                sd = decide(sub, family, ctx.margins)[h]
                sens_word = sd["verdict"]
                entry["truncation_sensitivity"] = {
                    "n_excluded": s["n_excluded_truncated"], "estimate": s["estimate"], "ci95": s["ci95"],
                    "p": s["p"], "p_rev": s["p_rev"], "p_adjusted": sd["confirm"]["adjusted"],
                    "p_rev_adjusted": sd["refute"]["adjusted"], "verdict": sens_word,
                    "agrees": sens_word == d["verdict"]}
            except NotRun as e:
                # Not computable (e.g. every item of a compared cell truncated): the two verdicts cannot be shown
                # to agree, so the headline rule treats it as a disagreement (review fix 2026-10-03).
                sens_failed = True
                entry["truncation_sensitivity"] = {"status": NOT_RUN, "reason": str(e), "agrees": False}
        eligible = (d["verdict"] != NOT_RUN and not flags and not sens_failed
                    and (sens_word is None or sens_word == d["verdict"]))
        entry["headline_eligible"] = eligible
        if sens_failed or (sens_word is not None and sens_word != d["verdict"]):
            entry["headline_verdict"] = TRUNCATION_SENSITIVE
        elif d["verdict"] == INCONCLUSIVE and d["leans_refuted"]:
            entry["headline_verdict"] = LEANS_REFUTED
        else:
            entry["headline_verdict"] = d["verdict"]
        out[h] = entry
    holm_tables = {
        "alpha": alpha, "m": len(family), "family": family,
        "confirm": {h: dec[h]["confirm"] for h in family},
        "refute": {h: dec[h]["refute"] for h in family},
        "escalated": escalated,
    }
    return {"hypotheses": out, "holm": holm_tables}


# --------------------------------------------------------------------------
# Plain answers (HYPOTHESIS "Plain answers"): the state per question
# --------------------------------------------------------------------------

def plain_answer_states(hyp: Mapping[str, dict], d1_res: Mapping, plan: Mapping, ctx: "Context | None" = None) -> dict:
    """Which pre-registered sentence applies to each question; tables.py fills the numbers.

    With ctx, Q3's German basis and the E1 label come from the cells that ran (review fix 2026-10-03); without it,
    from the plan's Tier-B list."""
    def word(h):
        return hyp[h]["verdict"] if h in hyp else NOT_RUN

    def nr(h):
        return {"id": "not_run", "H": h, "reason": hyp.get(h, {}).get("reason", "not in the family")}

    q1 = [{"id": "Q1_always"}]
    q1.append(nr("H1") if word("H1") == NOT_RUN else {"id": "Q1_H1", "state": word("H1")})
    d1w = d1_res.get("verdict", NOT_RUN)
    q1.append({"id": "not_run", "H": "D1", "reason": d1_res.get("reason")} if d1w == NOT_RUN
              else {"id": "Q1_D1", "state": d1w})

    h2 = hyp.get("H2", {})
    pc = (h2.get("detail") or {}).get("protocol_control") or {}
    band_known = "peers_within_band" in pc
    if word("H2") == NOT_RUN:
        q2 = nr("H2")
    elif band_known and not pc["peers_within_band"]:
        q2 = {"id": "Q2_peers_outside", "direction": "lower" if pc["peers_mean_D_shared"] < 0 else "higher"}
    elif not band_known and word("H2") in (CONFIRMED, REFUTED):
        # The CONFIRMED and REFUTED sentences are pre-registered only with the peers' mean D within +/- 2 pp; with
        # the protocol control not computed no pre-registered sentence applies (review fix 2026-10-03).
        q2 = {"id": "Q2_peer_control_not_run", "state": word("H2"), "reason": pc.get("reason")}
    else:
        q2 = {"id": {CONFIRMED: "Q2_confirmed", REFUTED: "Q2_refuted", INCONCLUSIVE: "Q2_inconclusive"}[word("H2")]}
        if q2["id"] == "Q2_confirmed":
            q2["named_rows"] = (h2.get("detail") or {}).get("rows_upper_bound_below_minus_8pp", [])
    q2["peer_band_known"] = band_known
    q2["headline_eligible"] = bool(h2.get("headline_eligible")) and q2["id"] != "Q2_peer_control_not_run"

    tier_b = set(plan.get("tier_b", []))
    german = ["MMLU-ProX DE"]
    if ctx is not None:
        arms = [ctx.K] + ctx.peers

        def ran(task):
            return bool(ctx.peers) and all(ctx.has_cell(a, task) for a in arms)

        if ran("gpqa_de"):
            german.append("GPQA-D DE")
        if ran("aime_de"):
            german.append("AIME DE")
        # E1: the label applies unless the rows every compared model ran include an AIME and a GPQA row.
        e1_applies = not (any(ran(t) for t in ("aime_en", "aime_de")) and any(ran(t) for t in GPQA_ROWS))
    else:
        if "B6" in tier_b and plan.get("plan") != "P10":
            german.append("GPQA-D DE")
        if "B2" in tier_b and "B1" in tier_b:
            german.append("AIME DE")
        e1_applies = not ("B2" in tier_b and bool({"B3", "B6"} & tier_b))
    q3 = {"id": "Q3_always", "verdicts": {h: hyp.get(h, {}).get("label", NOT_RUN) for h in ("H3", "H4", "H5")},
          "e1_label_applies": e1_applies, "german_basis": german}
    q3["not_run"] = [nr(h) for h in ("H3", "H4", "H5") if word(h) == NOT_RUN]

    h6 = hyp.get("H6", {})
    forced = (h6.get("detail") or {}).get("forced", {})
    if word("H6") == NOT_RUN:
        q4 = nr("H6")
    elif word("H6") == CONFIRMED and forced.get("ran") and forced.get("upper_below_margin"):
        q4 = {"id": "Q4_knows_less"}
    elif word("H6") == CONFIRMED:
        q4 = {"id": "Q4_confirmed_rate"}
    elif word("H6") == REFUTED:
        q4 = {"id": "Q4_refuted"}
    else:
        q4 = {"id": "Q4_inconclusive"}

    if word("H7") == NOT_RUN or word("H8") == NOT_RUN:
        missing = [h for h in ("H7", "H8") if word(h) == NOT_RUN]
        q5 = {"id": "Q5_not_run", "missing": missing,
              "reason": "; ".join(f"{h}: {hyp.get(h, {}).get('reason')}" for h in missing)}
    else:
        q5 = {"id": "Q5_both", "verdicts": {h: hyp[h]["label"] for h in ("H7", "H8")}}
        if word("H8") == CONFIRMED and (hyp["H8"].get("detail") or {}).get("per_token", {}).get("exceeds_threshold"):
            q5["per_token_note"] = "H8 is CONFIRMED per byte but the per-token ratio exceeds 1.5; the post says so"
    return {"Q1": q1, "Q2": q2, "Q3": q3, "Q4": q4, "Q5": q5}


# --------------------------------------------------------------------------
# Summary of cells
# --------------------------------------------------------------------------

def cell_summary(scores: Scores) -> dict:
    out = {}
    for (arm, task, effort, pass_) in scores.keys():
        c = scores.cell(arm, task, effort, pass_)
        n = len(c)
        out[f"{arm}/{task}/{effort}/pass{pass_}"] = {
            "n": n,
            "accuracy": sum(i.score for i in c.values()) / n if n else None,
            "truncation_rate": sum(i.truncated for i in c.values()) / n if n else None,
            "parse_failure_rate": sum((not i.parse_ok) and not i.truncated for i in c.values()) / n if n else None,
        }
    return out


# --------------------------------------------------------------------------
# compute / write
# --------------------------------------------------------------------------

def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        f = float(x)
        return f if math.isfinite(f) else None
    if callable(x):
        return None
    return x


def dumps(obj) -> str:
    """Sorted keys, UTF-8, no NaN: the same input gives the same bytes."""
    return json.dumps(_jsonable(obj), sort_keys=True, indent=1, ensure_ascii=False, allow_nan=False) + "\n"


def analysis_file_shas() -> dict:
    tier2 = {"exploratory.py", "tables.py"}
    out = {}
    for p in sorted(ANALYSIS_DIR.iterdir()):
        if p.is_file() and p.suffix in (".py", ".json"):
            out[p.name] = {"sha256": sha256_file(p), "tier": 2 if p.name in tier2 else 1}
    return out


def compute(results_dir, plan, scores: Scores | None = None, *, bench: dict | None = None,
            margins: dict | None = None, vendor: dict | None = None, now: datetime | None = None,
            require_rescore: bool = True, exploratory: bool = True) -> dict:
    """All verdicts in the exp_023 shape: {ts, hardware, config, manifests, summary, verdicts}.

    results_dir: the experiment's results/ (scores and bench files are read from it
    unless passed in). plan: the plan_fixed dict. Refuses (RefuseError) without a
    matching mini re-score unless require_rescore is False (tests only).
    """
    results_dir = Path(results_dir) if results_dir is not None else None
    inputs: dict[str, str] = {}
    rescore = None
    if require_rescore:
        if results_dir is None:
            raise RefuseError("results_dir is required for the rescore check")
        rescore = check_rescore(results_dir)
    if scores is None:
        scores, shas = Scores.from_dir(results_dir / "scores")
        inputs.update(shas)
    if bench is None:
        bench, shas = load_bench(results_dir) if results_dir is not None else ({}, {})
        inputs.update(shas)
    margins = margins if margins is not None else load_margins()
    vendor = vendor if vendor is not None else load_vendor()
    ctx = Context(results_dir, plan, scores, bench, margins, vendor)

    fam = evaluate_family(ctx)
    hyp = fam["hypotheses"]
    d1_res = d1(ctx)
    verdicts: dict = {}
    for h in ctx.plan["family"]:
        e = hyp[h]
        verdicts[h] = e["label"]
        verdicts[f"{h}_detail"] = e
    verdicts["D1"] = d1_res.get("label", d1_res["verdict"])
    verdicts["D1_detail"] = d1_res

    exploratory_out = None
    if exploratory:
        try:
            from analysis import exploratory as expl  # Tier 2, frozen by its own amendment
        except ModuleNotFoundError as e:
            if e.name not in ("analysis.exploratory",):
                raise
            exploratory_out = {"status": "not computed: analysis/exploratory.py absent"}
        else:
            exploratory_out = expl.compute_all(ctx, hyp)
    if isinstance(exploratory_out, dict):
        for k, v in exploratory_out.items():
            verdicts[k] = v

    ts = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    out = {
        "ts": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "hardware": HOST_LABEL,
        "config": {
            "version": VERSION,
            "plan": ctx.plan.get("plan"), "n_M": ctx.plan.get("n_M"),
            "m": len(ctx.plan["family"]), "family": ctx.plan["family"],
            "kolibri_primary": ctx.K, "peers": ctx.peers, "tier_b": ctx.plan["tier_b"],
            "h2_rows": ctx.plan["h2_rows"], "h7_rows": ctx.plan["h7_rows"],
            "forced_cells": ctx.plan["forced_cells"], "b10": ctx.plan["b10"],
            "alpha": margins["alpha"], "B": margins["B"], "B_escalated": margins["B_escalated"],
            "margins": {k: v for k, v in margins.items() if k != "notes"},
            "seeds": {h: stats.seed_of(h) for h in ctx.plan["family"]},
            "analysis_files": analysis_file_shas(),
        },
        "manifests": {"inputs": inputs, "rescore": rescore,
                      "plan_sha256": hashlib.sha256(dumps(plan or {}).encode()).hexdigest()},
        "summary": {
            "holm": fam["holm"],
            "plain_answers": plain_answer_states(hyp, d1_res, ctx.plan, ctx),
            "cells": cell_summary(scores),
            "duplicates_ignored": scores.duplicates,
            "verdict_words": {h: hyp[h]["verdict"] for h in ctx.plan["family"]} | {"D1": d1_res["verdict"]},
        },
        "verdicts": verdicts,
    }
    return _jsonable(out)


def verdict_table_md(v: dict) -> str:
    """The Tier-1 verdict table: one row per H, with p, Holm and the guards."""
    def f(x, nd=4):
        return "—" if x is None else (f"{x:.{nd}g}" if isinstance(x, float) else str(x))

    lines = [
        f"# exp_036 verdicts ({v['ts']})", "",
        f"Plan {v['config'].get('plan')}, n_M {v['config'].get('n_M')}, m = {v['config']['m']}, "
        f"alpha = {v['config']['alpha']}, primary arm {v['config']['kolibri_primary']}, "
        f"peers {', '.join(v['config']['peers'])}. Hardware: {v['hardware']}.", "",
        "| H | Verdict | Estimate | 95 % CI | p | p (Holm) | p_rev | p_rev (Holm) | Step | B | Truncation sensitivity | Headline |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for h in v["config"]["family"]:
        d = v["verdicts"][f"{h}_detail"]
        ts = d.get("truncation_sensitivity") or {}
        sens = ts.get("verdict", "—" if not ts else ts.get("status", "—"))
        ci = d.get("ci95")
        ci_s = f"[{f(ci[0])}, {f(ci[1])}]" if ci else "—"
        lines.append(
            f"| {h} | {v['verdicts'][h]} | {f(d.get('estimate'))} | {ci_s} | {f(d.get('p'))} | "
            f"{f(d.get('p_adjusted'))} | {f(d.get('p_rev'))} | {f(d.get('p_rev_adjusted'))} | "
            f"{d.get('holm_step')} | {d.get('B')} | {sens} | "
            f"{d.get('headline_verdict') if d.get('headline_eligible') else 'not eligible'} |")
    d1d = v["verdicts"].get("D1_detail", {})
    lines += ["", f"D1 (outside Holm): {v['verdicts'].get('D1')}; median peak at 64k "
              f"{f(d1d.get('median_peak_gib_64k'))} GiB (M5 Max, MLX 0.31.2)."]
    nr = [f"- {h}: {v['verdicts'][f'{h}_detail'].get('reason')}" for h in v["config"]["family"]
          if v["verdicts"][f"{h}_detail"].get("verdict") == NOT_RUN]
    if nr:
        lines += ["", "NOT RUN (p = p_rev = 1, m unchanged):"] + nr
    return "\n".join(lines) + "\n"


def write(results_dir, verdicts: dict) -> tuple[Path, Path]:
    """Write results/verdicts_<UTC>.json and .md (UTC from verdicts["ts"]); never overwrite."""
    results_dir = Path(results_dir)
    stamp = verdicts["ts"].replace("-", "").replace(":", "")
    jp = results_dir / f"verdicts_{stamp}.json"
    mp = results_dir / f"verdicts_{stamp}.md"
    for p in (jp, mp):
        if p.exists():
            raise FileExistsError(f"{p} exists; verdicts are never overwritten")
    md = verdict_table_md(verdicts)
    try:
        from analysis import tables  # Tier 2
    except ModuleNotFoundError as e:
        if e.name != "analysis.tables":
            raise
    else:
        md += "\n" + tables.render_markdown(verdicts)
    with open(jp, "x", encoding="utf-8") as f:
        f.write(dumps(verdicts))
    with open(mp, "x", encoding="utf-8") as f:
        f.write(md)
    return jp, mp


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="exp_036 verdicts (HYPOTHESIS H1-H8, D1; E1-E12)")
    ap.add_argument("--results", default=str(EXP_DIR / "results"))
    ap.add_argument("--plan", default=None, help="plan_fixed_<UTC>.json (default: the newest in results/)")
    ap.add_argument("--no-exploratory", action="store_true")
    a = ap.parse_args(argv)
    results = Path(a.results)
    try:
        from runner import guard
    except ModuleNotFoundError:
        print("runner/guard.py is missing: cannot check the git identity before writing results/",
              file=sys.stderr)
        return 3
    guard.require_identity()
    plan_path = Path(a.plan) if a.plan else newest(results.glob("plan_fixed_*.json"))
    if plan_path is None:
        print("no results/plan_fixed_<UTC>.json", file=sys.stderr)
        return 1
    try:
        v = compute(results, read_json(plan_path), exploratory=not a.no_exploratory)
    except RefuseError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 1
    v["manifests"]["plan_file"] = {"path": plan_path.name, "sha256": sha256_file(plan_path)}
    jp, mp = write(results, v)
    print(jp)
    print(mp)
    return 0


if __name__ == "__main__":
    # Run the package module, not this __main__ copy: analysis/exploratory.py and tables.py import
    # analysis.verdicts and catch its NotRun / ConfigError; a second module object would carry different
    # exception classes and an E-analysis that is NOT RUN would crash the CLI (integration finding, BUILD_LOG.md).
    from analysis import verdicts as _verdicts

    sys.exit(_verdicts.main())
