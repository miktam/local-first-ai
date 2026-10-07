"""Build the item manifests from local data by the frozen rule (BUILD_SPEC §5.5; RUNBOOK step 7).

    "$PY" tasks/build_manifests.py                  # every set; on the mbp in S1
    "$PY" tasks/build_manifests.py --sets ifbench   # only the named sets (no shingles, no results record)

Writes:
- tasks/manifests/<name>.json: one per set. Public sets (MMLU-ProX, AIME EN, IFBench) hold
  {id, item_sha256, prompt_sha256} plus category, gold, cat_rank and n_options where they apply. Withheld sets
  (GPQA EN/DE, AIME-DE, RGB) hold {id, item_sha256, prompt_sha256}, plus the RGB document indices and the GPQA
  `in_primary` flag. No manifest holds item text, and no withheld manifest holds gold (a hash of a letter or an
  integer would be trivially reversible). The GPQA EN manifest also records `primary_n` and `overlong_excluded`,
  the number of Diamond rows eval-framework's over-long filter removed: 0 by Amendment 3, asserted at every build.
  The MMLU-ProX-Lite manifests list all of Lite in the Webster seat order (Amendment 4: the n_M set is the first
  n_M entries) and record `nM_allocation`, the items per category at every ladder n_M. The four full-pool MMLU-ProX
  manifests (pilot EN/DE, C1, peer check) record `gold_inconsistent_excluded`, the ids per language whose answer
  letter disagrees with answer_index and which the pool leaves out (Amendment 4: ["3787"] in each, asserted at
  every build).
- tasks/manifests/mmlu_prox_category_counts.json: full test split counts per language and category (H2 weights;
  every row, a gold-inconsistent one included).
- $EXP036_PRIVATE/manifests/<name>.jsonl (withheld sets only): the same items with messages and gold (for RGB
  `gold` is the answer as RGB stores it and `gold_fake` the counterfactual answer); scorers/score_all.py reads
  gold from here for the withheld sets and from tasks/manifests/ for the public ones.
- tools/withheld_shingles.sha256 (all sets only): tools/shingles.py `write_shingle_file` (one normalisation
  for writer and leak check) over the sources BUILD_SPEC §5.5 names: GPQA EN/DE questions and options
  (options also as full option hashes), AIME-DE problems, RGB queries and answers, plus RGB's instruction.yaml
  strings.
- results/manifests_<UTC>.json: sha256 of every file above and the `checks` (the GPQA EN over-long count; the
  MMLU-ProX full-split gold-inconsistent ids) (all sets only; needs the git identity).

The same data in gives byte-identical manifests out: no timestamp is written into a manifest, every JSON file
has sorted keys, and every order is a frozen rule. An existing output with different content is never
overwritten: the build stops and names it.

`items_for(name)` is the reverse direction for the runner: it re-renders a manifest's items from the local
data (RGB from the recorded document indices) and refuses on any item or prompt sha256 mismatch.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

TASKS_DIR = Path(__file__).resolve().parent
EXP_DIR = TASKS_DIR.parent
if __name__ == "__main__" and str(EXP_DIR) not in sys.path:  # run as a script: make `tasks` importable
    sys.path.insert(0, str(EXP_DIR))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from tasks import aime, assets, gpqa, ifbench, mmlu_prox, rgb  # noqa: E402
from tasks.common import Item, canonical_json, sha256_hex  # noqa: E402

RULES_PATH = TASKS_DIR / "selection_rules.json"
MANIFESTS_DIR = TASKS_DIR / "manifests"
SHINGLES_PATH = EXP_DIR / "tools" / "withheld_shingles.sha256"
SCHEMA = "exp036 item manifest v1"

# Keys a manifest entry may carry (HYPOTHESIS "Sources & Rights": ids, hashes, and for MMLU-ProX and AIME EN
# only, the gold label and category). Anything else is a build error.
BASE_FIELDS = ("id", "item_sha256", "prompt_sha256")
ALLOWED_WITHHELD = set(BASE_FIELDS) | {"in_primary", "doc_indices", "subset"}
ALLOWED_PUBLIC = ALLOWED_WITHHELD | {"category", "gold", "cat_rank", "n_options"}
MAX_PUBLIC_STRING = 64


@functools.lru_cache(maxsize=1)
def rules() -> dict:
    return json.loads(RULES_PATH.read_text(encoding="utf-8"))


def rules_sha256() -> str:
    return sha256_hex(RULES_PATH.read_bytes())


# --------------------------------------------------------------------------------------------------
# Sources and the per-set builders
# --------------------------------------------------------------------------------------------------

ASSET_NAMES = {
    "gpqa": "Idavidrein/gpqa",
    "gpqa_de": "ellamind/gpqa-multilingual",
    "mmlu_lite": "li-lab/MMLU-ProX-Lite",
    "mmlu_full": "li-lab/MMLU-ProX",
    "aime_en": "math-ai/aime26",
    "aime_de": "ellamind/aime26-multilingual",
    "ifbench": "allenai/IFBench_test",
    "rgb": "chen700564/RGB",
}


class Context:
    """Paths of the local sources plus per-build caches (each dataset is read once)."""

    def __init__(self, data: Path | None = None, check_revisions: bool = True):
        self.data = Path(data).expanduser() if data else None
        self.check_revisions = check_revisions
        self._cache: dict = {}

    def path(self, key: str) -> Path:
        a = assets.asset(ASSET_NAMES[key])
        if self.data is None:
            return a.path
        rel = a.rel[len("data/"):] if a.rel.startswith("data/") else a.rel
        return self.data / rel

    def source_record(self, key: str) -> dict:
        a = assets.asset(ASSET_NAMES[key])
        found = None
        if self.check_revisions:
            ok, found = assets.verify_revision(assets.Asset(a.name, self.path(key), a.revision, a.kind, a.rel))
            if not ok:
                raise RuntimeError(f"{a.name}: revision on disk {found!r} is not the pinned {a.revision}")
        return {"repo": a.name, "revision": a.revision, "revision_found": found}

    def cached(self, key: str, make: Callable):
        if key not in self._cache:
            self._cache[key] = make()
        return self._cache[key]

    # shared intermediate results
    def gpqa_diamond(self) -> list[Item]:
        return self.cached("gpqa_diamond", lambda: gpqa.load_diamond_en(self.path("gpqa")))

    def gpqa_diamond_ids(self) -> set[str]:
        return {it.id for it in self.gpqa_diamond()}

    def mmlu_parallel(self) -> dict[str, list[str]]:
        seed = rules()["selection_seed"]
        return self.cached("mmlu_parallel", lambda: mmlu_prox.parallel_ids(self.path("mmlu_lite"), seed))

    def mmlu_pool(self) -> list[str]:
        seed = rules()["selection_seed"]
        return self.cached("mmlu_pool", lambda: mmlu_prox.full_pool_ids(self.path("mmlu_full"), self.path("mmlu_lite"), seed))

    def mmlu_gold_mismatch(self) -> dict[str, list[str]]:
        """{lang: ids of the full test split whose answer letter disagrees with answer_index} (Amendment 4)."""
        return self.cached("mmlu_gold_mismatch", lambda: mmlu_prox.gold_mismatch_ids(self.path("mmlu_full")))

    def rgb_instruction(self) -> dict:
        return self.cached("rgb_instr", lambda: rgb.load_instruction(self.path("rgb")))

    def aime_en(self) -> list[Item]:
        return self.cached("aime_en", lambda: aime.load_en(self.path("aime_en")))

    def ifbench(self) -> list[Item]:
        return self.cached("ifbench", lambda: ifbench.load(self.path("ifbench")))


def _gpqa_diamond_en(ctx: Context, enforce: bool) -> list[Item]:
    """Every Diamond row, flagged in_primary. Amendment 3 (2026-10-04): the vendored over-long filter stays in
    force and must exclude exactly `overlong_excluded` (0) rows; that is asserted on every build, counts enforced
    or not (the dry run, items_for on the run host), so a change in the data cannot slip through."""
    items = ctx.gpqa_diamond()
    rule = rules()["gpqa"]["diamond_en"]
    primary = {it.id for it in gpqa.primary_en(items, expect_excluded=rule["overlong_excluded"])}
    if enforce and len(primary) != rule["primary_n"]:
        raise ValueError(f"expected a primary set of {rule['primary_n']} items, got {len(primary)}")
    for it in items:
        it.public["in_primary"] = it.id in primary
    return items


def _gpqa_diamond_de(ctx: Context, enforce: bool) -> list[Item]:
    return gpqa.load_diamond_de(ctx.path("gpqa_de"))


def _gpqa_pilot(lang: str):
    def build(ctx: Context, enforce: bool) -> list[Item]:
        r = rules()
        key = "gpqa" if lang == "en" else "gpqa_de"
        return gpqa.load_pilot_main(ctx.path(key), ctx.gpqa_diamond_ids(), n=r["gpqa"][f"pilot_{lang}"]["n"],
                                    seed=r["selection_seed"], lang=lang)
    return build


def _mmlu_lite(lang: str):
    """Every Lite item, listed in the Webster seat order (Amendment 4, selection_rules.json mmlu_prox.nM_rule), so
    the runner's n_M set is a prefix. With counts enforced, the parallel ids per category must equal the registered
    lite_category_counts; on every build, each category must have an item at every ladder n_M the data allows."""
    def build(ctx: Context, enforce: bool) -> list[Item]:
        m = rules()["mmlu_prox"]
        per_cat = ctx.mmlu_parallel()
        counts = {c: len(v) for c, v in per_cat.items()}
        if enforce and counts != m["lite_category_counts"]:
            bad = {c: (n, m["lite_category_counts"].get(c)) for c, n in counts.items() if n != m["lite_category_counts"].get(c)}
            raise ValueError(f"MMLU-ProX-Lite: parallel ids per category differ from the registered "
                             f"lite_category_counts (Amendment 4), (found, registered): {bad}")
        total = sum(counts.values())
        for n in m["nM_ladder"]:
            if n <= total:
                empty = [c for c, k in mmlu_prox.allocation(counts, n).items() if k == 0]
                if empty:
                    raise ValueError(f"MMLU-ProX-Lite: n_M={n} leaves categories without an item: {empty}")
        ids = mmlu_prox.select_ids(per_cat, total)
        rank = {i: r for ids_c in per_cat.values() for r, i in enumerate(ids_c)}
        items = mmlu_prox.load_lite(lang, ids, ctx.path("mmlu_lite"))
        for it in items:
            it.public.update(category=it.category, gold=it.gold, cat_rank=rank[it.id], n_options=it.n_options)
        return items
    return build


def _mmlu_full_slice(lang: str, part: str):
    """A slice of the full pool. Amendment 4: rows whose answer letter disagrees with answer_index are not in the
    pool, and on every build (counts enforced or not, as items_for on the run host) their ids must equal the
    registered gold_inconsistent_expected, so a change in the data cannot slip through."""
    def build(ctx: Context, enforce: bool) -> list[Item]:
        m = rules()["mmlu_prox"]
        found, want = ctx.mmlu_gold_mismatch(), m["gold_inconsistent_expected"]
        if found != want:
            raise ValueError(f"MMLU-ProX full: rows whose answer letter disagrees with answer_index {found}, "
                             f"expected {want} (Amendment 4)")
        lo, hi = m["full_layout"][part]
        items = mmlu_prox.load_full(lang, ctx.mmlu_pool()[lo:hi], ctx.path("mmlu_full"))
        for it in items:
            it.public.update(category=it.category, gold=it.gold, n_options=it.n_options)
        return items
    return build


def _aime_en(ctx: Context, enforce: bool) -> list[Item]:
    items = ctx.aime_en()
    for it in items:
        it.public["gold"] = it.gold
    return items


def _aime_pilot(ctx: Context, enforce: bool) -> list[Item]:
    r = rules()
    items = aime.pilot_en(ctx.aime_en(), n=r["aime"]["pilot_n"], seed=r["selection_seed"])
    for it in items:
        it.public["gold"] = it.gold
    return items


def _aime_de(ctx: Context, enforce: bool) -> list[Item]:
    return aime.load_de(ctx.path("aime_de"))


def _ifbench(ctx: Context, enforce: bool) -> list[Item]:
    return ctx.ifbench()


def _ifbench_pilot(ctx: Context, enforce: bool) -> list[Item]:
    r = rules()
    return ifbench.pilot(ctx.ifbench(), n=r["ifbench"]["pilot_n"], seed=r["selection_seed"])


def _rgb(kind: str):
    def build(ctx: Context, enforce: bool, indices: dict | None = None) -> list[Item]:
        d, instr = ctx.path("rgb"), ctx.rgb_instruction()
        r = rules()["rgb"]
        if kind == "cb":
            return rgb.cb_items(d, instr)
        if kind == "forced":
            return rgb.forced_items(d)
        if kind == "neg":
            return rgb.negative_items(d, instr, indices)
        if kind == "fact":
            return rgb.fact_items(d, instr, indices)
        cb, fo = rgb.pilot_items(d, r["pilot"]["n_cb"], r["pilot"]["n_forced"], rules()["selection_seed"], instr)
        return cb if kind == "pilot_cb" else fo
    return build


@dataclass(frozen=True)
class SetDef:
    name: str
    task: str
    withheld: bool
    sources: tuple[str, ...]
    build: Callable
    expected_n: int | None
    rule: str


def set_defs() -> list[SetDef]:
    r = rules()
    g, m, rg = r["gpqa"], r["mmlu_prox"], r["rgb"]
    lite_n = sum(m["lite_category_counts"].values())  # 588 (Amendment 4: not 42 x 14)
    return [
        SetDef("gpqa_diamond_en", "gpqa_en", True, ("gpqa",), _gpqa_diamond_en, g["diamond_en"]["n"], g["diamond_en"]["primary_rule"]),
        SetDef("gpqa_diamond_de", "gpqa_de", True, ("gpqa_de",), _gpqa_diamond_de, g["diamond_de"]["n"], "deu config, is_diamond rows, file order"),
        SetDef("gpqa_pilot_en", "gpqa_en", True, ("gpqa",), _gpqa_pilot("en"), g["pilot_en"]["n"], g["pilot_en"]["exclude"]),
        SetDef("gpqa_pilot_de", "gpqa_de", True, ("gpqa_de", "gpqa"), _gpqa_pilot("de"), g["pilot_de"]["n"], g["pilot_de"]["exclude"]),
        SetDef("mmlu_prox_lite_en", "mmlu_en", False, ("mmlu_lite",), _mmlu_lite("en"), lite_n, m["nM_rule"]),
        SetDef("mmlu_prox_lite_de", "mmlu_de", False, ("mmlu_lite",), _mmlu_lite("de"), lite_n, m["nM_rule"]),
        SetDef("mmlu_prox_full_pilot_en", "mmlu_en", False, ("mmlu_full", "mmlu_lite"), _mmlu_full_slice("en", "pilot"), 8, m["full_pool_rule"]),
        SetDef("mmlu_prox_full_pilot_de", "mmlu_de", False, ("mmlu_full", "mmlu_lite"), _mmlu_full_slice("de", "pilot"), 8, m["full_pool_rule"]),
        SetDef("mmlu_prox_c1_en", "mmlu_en", False, ("mmlu_full", "mmlu_lite"), _mmlu_full_slice("en", "c1"), 100, m["full_pool_rule"]),
        SetDef("mmlu_prox_peercheck_en", "mmlu_en", False, ("mmlu_full", "mmlu_lite"), _mmlu_full_slice("en", "peer_check"), 30, m["full_pool_rule"]),
        SetDef("aime_en", "aime_en", False, ("aime_en",), _aime_en, r["aime"]["n"], "file order"),
        SetDef("aime_pilot_en", "aime_en", False, ("aime_en",), _aime_pilot, r["aime"]["pilot_n"], "seeded order, pool aime_pilot_en"),
        SetDef("aime_de", "aime_de", True, ("aime_de",), _aime_de, r["aime"]["n"], "deu config, file order"),
        SetDef("ifbench", "ifbench", False, ("ifbench",), _ifbench, r["ifbench"]["n"], "file order"),
        SetDef("ifbench_pilot", "ifbench", False, ("ifbench",), _ifbench_pilot, r["ifbench"]["pilot_n"], "seeded order, pool ifbench_pilot"),
        SetDef("rgb_cb", "rgb_cb", True, ("rgb",), _rgb("cb"), rg["cb"]["n"], "en.json then en_fact.json, file order; passage_num 0"),
        SetDef("rgb_forced", "rgb_forced", True, ("rgb",), _rgb("forced"), rg["forced"]["n"], "the closed-book items, forced-answer prompt"),
        SetDef("rgb_negative", "rgb_neg", True, ("rgb",), _rgb("neg"), rg["negative"]["n"], "en.json; noise 1.0, 5 passages, seed 2333"),
        SetDef("rgb_fact", "rgb_fact", True, ("rgb",), _rgb("fact"), rg["fact"]["n"], rg["fact"]["prompt"]),
        SetDef("rgb_pilot_cb", "rgb_cb", True, ("rgb",), _rgb("pilot_cb"), rg["pilot"]["n_cb"], "en_int, seeded order, pool rgb_pilot"),
        SetDef("rgb_pilot_forced", "rgb_forced", True, ("rgb",), _rgb("pilot_forced"), rg["pilot"]["n_forced"], rg["pilot"]["forced_rule"]),
    ]


def set_def(name: str) -> SetDef:
    for s in set_defs():
        if s.name == name:
            return s
    raise KeyError(f"unknown manifest set {name!r}")


# --------------------------------------------------------------------------------------------------
# Serialisation
# --------------------------------------------------------------------------------------------------


def public_entry(item: Item, withheld: bool) -> dict:
    entry = {"id": item.id, "item_sha256": item.item_sha256, "prompt_sha256": item.prompt_sha256}
    entry.update(item.public)
    entry.pop("id_source", None)  # manifest-level
    entry.pop("n_instructions", None)
    entry.pop("doc_source", None)  # manifest-level
    allowed = ALLOWED_WITHHELD if withheld else ALLOWED_PUBLIC
    extra = set(entry) - allowed
    if extra:
        raise ValueError(f"{item.task} {item.id}: fields not allowed in a {'withheld' if withheld else 'public'} manifest: {sorted(extra)}")
    for k, v in entry.items():
        if isinstance(v, str) and len(v) > MAX_PUBLIC_STRING:
            raise ValueError(f"{item.task} {item.id}: field {k} is a long string; manifests carry no text")
    return entry


def private_entry(item: Item) -> dict:
    entry = {"id": item.id, "item_sha256": item.item_sha256, "prompt_sha256": item.prompt_sha256}
    entry.update(item.public)
    entry.update(item.private)
    entry.update(messages=item.messages, gold=item.gold, category=item.category, n_options=item.n_options)
    return entry


def json_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1) + "\n").encode("utf-8")


def jsonl_bytes(records: list[dict]) -> bytes:
    return b"".join(canonical_json(r) + b"\n" for r in records)


MMLU_LITE_SETS = ("mmlu_prox_lite_en", "mmlu_prox_lite_de")
MMLU_FULL_SETS = ("mmlu_prox_full_pilot_en", "mmlu_prox_full_pilot_de", "mmlu_prox_c1_en", "mmlu_prox_peercheck_en")


def nm_allocation(items: list[Item], ladder: list[int]) -> dict[str, dict[str, int]]:
    """{str(n_M): {category: items}} of the first n_M listed items, for every ladder n_M the listing holds."""
    out = {}
    for n in ladder:
        if n <= len(items):
            cats: dict[str, int] = {}
            for it in items[:n]:
                cats[it.category] = cats.get(it.category, 0) + 1
            out[str(n)] = cats
    return out


def manifest_doc(sd: SetDef, items: list[Item], sources: list[dict], extra: dict | None = None) -> dict:
    id_sources = sorted({it.public.get("id_source") for it in items if it.public.get("id_source")})
    doc = {
        "schema": SCHEMA,
        "name": sd.name,
        "task": sd.task,
        "withheld": sd.withheld,
        "rule": sd.rule,
        "selection_rules_sha256": rules_sha256(),
        "sources": sources,
        "n": len(items),
        "items": [public_entry(it, sd.withheld) for it in items],
    }
    if id_sources:
        doc["id_source"] = id_sources[0] if len(id_sources) == 1 else id_sources
    flags = [it.public["in_primary"] for it in items if "in_primary" in it.public]
    if flags:  # GPQA EN: the recorded over-long count (Amendment 3: 0)
        doc["primary_n"] = sum(1 for f in flags if f)
        doc["overlong_excluded"] = len(flags) - doc["primary_n"]
    if sd.name in ("rgb_negative", "rgb_fact"):
        doc["doc_source"] = "negative" if sd.name == "rgb_negative" else "positive_wrong"
    if sd.name in MMLU_LITE_SETS:  # Amendment 4: the per-category n at every ladder n_M (a prefix of this listing)
        doc["nM_allocation"] = nm_allocation(items, rules()["mmlu_prox"]["nM_ladder"])
    doc.update(extra or {})
    return doc


def _write_once(path: Path, data: bytes) -> str:
    """Write `data` unless an identical file exists; refuse to replace a different one."""
    if path.exists():
        if path.read_bytes() == data:
            return "unchanged"
        raise FileExistsError(f"{path.name} exists with different content; manifests are frozen once built")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_bytes(data)
    os.replace(tmp, path)
    return "written"


# --------------------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------------------


def withheld_texts(ctx: Context) -> tuple[list[str], list[str]]:
    """(texts, option strings) for the shingle list, exactly the BUILD_SPEC §5.5 sources.

    Texts (8-word shingles): GPQA EN questions and options of gpqa_diamond.csv and gpqa_main.csv; German GPQA
    questions and options of every deu row; AIME-DE problems; RGB queries and answers (en, en_fact, en_int);
    the RGB instruction.yaml strings. Options (also a full hash when >= 3 words): the GPQA EN/DE answer
    options, the RGB instruction string, and every RGB query, GPQA EN/DE question and AIME-DE problem of 3–8
    normalised words, so that quoting a short withheld question alone is a finding (review fix 2026-10-03;
    tools/shingles.py scans 3–8-word windows). Not included, to keep the leak check's false positives on
    published outputs low: GPQA explanation and metadata columns, and RGB answers as option hashes (common
    named entities). The RGB documents and the FineWeb-2 rows are checked by the leak check's local-corpus
    pass from $EXP036_DATA (≈ 47 MB of hashes, too large to commit). Shingles of the kit's own public text
    (tools/shingles.py public_shingle_strings) are left out of the file."""
    texts: list[str] = []
    options: list[str] = []
    for name in ("gpqa_diamond.csv", "gpqa_main.csv"):
        for row in gpqa.en_rows(ctx.path("gpqa"), name):
            texts.append(row["Question"])
            options += [row[k] for k in gpqa.EN_FIELDS[1:]]
    de_rows, _ = gpqa.de_rows(ctx.path("gpqa_de"))
    for row in de_rows:
        texts.append(row["question"])
        options += [row["correct_answer"], *row["incorrect_answers"]]
    texts += [it.source["problem"] for it in aime.load_de(ctx.path("aime_de"))]
    d = ctx.path("rgb")
    for row in rgb.load_en(d) + rgb.load_fact(d) + rgb.load_int(d):
        texts += [row["query"], *_flatten(row["answer"]), *_flatten(row.get("fakeanswer"))]
    instr = ctx.rgb_instruction()
    texts += [instr["system"], instr["instruction"]]
    options.append(instr["instruction"])
    from tools.shingles import MIN_OPTION_WORDS, SHINGLE_WORDS, normalise_words

    questions = [r["Question"] for name in ("gpqa_diamond.csv", "gpqa_main.csv") for r in gpqa.en_rows(ctx.path("gpqa"), name)]
    questions += [r["question"] for r in de_rows]
    questions += [it.source["problem"] for it in aime.load_de(ctx.path("aime_de"))]
    questions += [row["query"] for row in rgb.load_en(d) + rgb.load_fact(d) + rgb.load_int(d)]
    options += [q for q in questions if isinstance(q, str) and MIN_OPTION_WORDS <= len(normalise_words(q)) <= SHINGLE_WORDS]
    return [t for t in texts if isinstance(t, str)], [o for o in options if isinstance(o, str)]


def _flatten(x) -> list[str]:
    if isinstance(x, str):
        return [x]
    if isinstance(x, (list, tuple)):
        out = []
        for y in x:
            out += _flatten(y)
        return out
    return []  # numbers, booleans, None


def build_all(out_dir: Path = MANIFESTS_DIR, private: Path | None = None, data: Path | None = None,
              sets: list[str] | None = None, results_dir: Path | None = None, shingles_out: Path | None = SHINGLES_PATH,
              enforce_counts: bool = True, check_revisions: bool = True) -> dict:
    """Build the named sets (default: all) and return {"manifests", "private", "shingles", "counts"}.

    results_dir and shingles_out are honoured only when every set is built. The caller (main) checks the
    git identity before passing results_dir.
    """
    out_dir, private = Path(out_dir), Path(private or assets.private_dir())
    ctx = Context(data, check_revisions)
    defs = set_defs() if sets is None else [set_def(n) for n in sets]
    every_set = sets is None
    summary: dict = {"manifests": {}, "private": {}, "counts": {}, "status": {}, "checks": {}}
    source_cache: dict[str, dict] = {}
    for sd in defs:
        try:
            items = sd.build(ctx, enforce_counts)
            if enforce_counts and sd.expected_n is not None and len(items) != sd.expected_n:
                raise ValueError(f"expected {sd.expected_n} items, got {len(items)}")
            for k in sd.sources:
                if k not in source_cache:
                    source_cache[k] = ctx.source_record(k)
            extra = {}
            if sd.name in MMLU_FULL_SETS:  # Amendment 4: the ids the full pool leaves out, per language
                extra["gold_inconsistent_excluded"] = ctx.mmlu_gold_mismatch()
                summary["checks"]["mmlu_prox_full_gold_inconsistent_excluded"] = ctx.mmlu_gold_mismatch()
            doc = manifest_doc(sd, items, [source_cache[k] for k in sd.sources], extra)
            if "overlong_excluded" in doc:
                summary["checks"][f"{sd.name}_overlong_excluded"] = doc["overlong_excluded"]
            pub_path = out_dir / f"{sd.name}.json"
            summary["status"][pub_path.name] = _write_once(pub_path, json_bytes(doc))
            summary["manifests"][f"tasks/manifests/{pub_path.name}"] = sha256_hex(pub_path.read_bytes())
            if sd.withheld:
                header = {"type": "header", "name": sd.name, "task": sd.task, "withheld": True,
                          "public_manifest_sha256": summary["manifests"][f"tasks/manifests/{pub_path.name}"]}
                priv_path = private / "manifests" / f"{sd.name}.jsonl"
                body = jsonl_bytes([header] + [dict(private_entry(it), type="item") for it in items])
                summary["status"][f"private/{priv_path.name}"] = _write_once(priv_path, body)
                summary["private"][f"manifests/{priv_path.name}"] = sha256_hex(body)
            summary["counts"][sd.name] = len(items)
        except Exception as e:
            raise RuntimeError(f"manifest {sd.name}: {type(e).__name__}: {e}") from e

    if every_set or (sets and any(s.startswith("mmlu_prox") for s in sets)):
        counts = {lang: mmlu_prox.category_counts(ctx.path("mmlu_full"), lang) for lang in mmlu_prox.LANGS}
        # {"en": {category: n}, "de": {...}} at the top level, as analysis/verdicts.py reads it; metadata in "_meta".
        doc = dict(counts)
        doc["_meta"] = {"schema": "exp036 MMLU-ProX category counts v1", "split": "test",
                        "source": ctx.source_record("mmlu_full"), "rule": rules()["mmlu_prox"]["category_counts"],
                        "totals": {lang: sum(c.values()) for lang, c in counts.items()},
                        "gold_inconsistent_counted": ctx.mmlu_gold_mismatch()}
        p = out_dir / "mmlu_prox_category_counts.json"
        summary["status"][p.name] = _write_once(p, json_bytes(doc))
        summary["manifests"][f"tasks/manifests/{p.name}"] = sha256_hex(p.read_bytes())

    if every_set and shingles_out is not None:
        from tools import shingles as tool_shingles  # the leak check's own module: one normalisation

        texts, options = withheld_texts(ctx)
        with tempfile.TemporaryDirectory() as tmp:
            staged = Path(tmp) / "withheld_shingles.sha256"
            public = [f.read_text(encoding="utf-8", errors="replace") for f in
                      [EXP_DIR / n for n in tool_shingles.PUBLIC_FILES]
                      + [q for g in tool_shingles.PUBLIC_GLOBS for q in sorted(EXP_DIR.glob(g))] if f.is_file()]
            counts = tool_shingles.write_shingle_file(staged, texts, options, note="built by tasks/build_manifests.py",
                                                      public_texts=public)
            data_bytes = staged.read_bytes()
        summary["status"][Path(shingles_out).name] = _write_once(Path(shingles_out), data_bytes)
        summary["shingles"] = {"tools/withheld_shingles.sha256": sha256_hex(data_bytes), **counts}

    if every_set and results_dir is not None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        record = {"ts": ts, "selection_rules_sha256": rules_sha256(),
                  "build_manifests_sha256": sha256_hex(Path(__file__).read_bytes()),
                  "manifests": summary["manifests"], "private": summary["private"],
                  "shingles": summary.get("shingles"), "counts": summary["counts"], "checks": summary["checks"]}
        p = Path(results_dir) / f"manifests_{ts}.json"
        if p.exists():
            raise FileExistsError(p.name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(json_bytes(record))
        summary["results"] = f"results/{p.name}"
    return summary


# --------------------------------------------------------------------------------------------------
# Reverse direction: manifest -> items on the run host
# --------------------------------------------------------------------------------------------------


class ManifestMismatch(RuntimeError):
    pass


def load_manifest(name: str, manifests_dir: Path = MANIFESTS_DIR) -> dict:
    return json.loads((Path(manifests_dir) / f"{name}.json").read_text(encoding="utf-8"))


def items_for(name: str, data: Path | None = None, manifests_dir: Path = MANIFESTS_DIR,
              check_revisions: bool = True) -> list[Item]:
    """The manifest's items, re-rendered from local data, in manifest order; every item_sha256 and
    prompt_sha256 must equal the manifest's (ManifestMismatch otherwise, BUILD_SPEC §5.5 rgb.py)."""
    doc = load_manifest(name, manifests_dir)
    sd = set_def(name)
    ctx = Context(data, check_revisions)
    if name in ("rgb_negative", "rgb_fact"):
        indices = {e["id"]: e["doc_indices"] for e in doc["items"]}
        items = sd.build(ctx, False, indices)
    else:
        items = sd.build(ctx, False)
    by_id = {it.id: it for it in items}
    out = []
    for e in doc["items"]:
        it = by_id.get(e["id"])
        if it is None:
            raise ManifestMismatch(f"{name}: item {e['id']} not found in the local data")
        if it.item_sha256 != e["item_sha256"] or it.prompt_sha256 != e["prompt_sha256"]:
            raise ManifestMismatch(f"{name}: item {e['id']} differs from the manifest (item or prompt sha256)")
        out.append(it)
    return out


def runner_items(name: str, data: Path | None = None, manifests_dir: Path = MANIFESTS_DIR,
                 n: int | None = None) -> list[dict]:
    """items_for() as runner/generate.py item dicts; `n` keeps a prefix (n_M for the MMLU-ProX-Lite sets, listed
    in the Webster seat order, so every n_M set is a prefix; Amendment 4)."""
    items = items_for(name, data, manifests_dir)
    return [it.to_dict() for it in (items if n is None else items[:n])]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build exp_036 item manifests (BUILD_SPEC §5.5).")
    ap.add_argument("--sets", help="comma-separated set names (default: all; then shingles + results record)")
    ap.add_argument("--data", type=Path, help="dataset root (default $EXP036_DATA)")
    ap.add_argument("--out", type=Path, default=MANIFESTS_DIR)
    ap.add_argument("--private", type=Path, help="default $EXP036_PRIVATE")
    ap.add_argument("--list", action="store_true", help="list the set names and exit")
    args = ap.parse_args(argv)
    if args.list:
        for sd in set_defs():
            print(f"{sd.name}\t{sd.task}\t{'withheld' if sd.withheld else 'public'}\t{sd.expected_n}")
        return 0
    sets = [s.strip() for s in args.sets.split(",")] if args.sets else None
    results_dir = None
    if sets is None:
        try:  # BUILD_SPEC §2 Identity: checked before anything is written under results/
            from runner.guard import require_identity
        except ModuleNotFoundError as e:
            print(json.dumps({"ok": False, "error": f"runner.guard is required for a full build ({e})"}))
            return 1
        require_identity()
        results_dir = EXP_DIR / "results"
    try:
        summary = build_all(args.out, args.private, args.data, sets, results_dir)
    except Exception as e:
        print(json.dumps({"ok": False, "error": str(e)}, ensure_ascii=False))
        return 1
    print(json.dumps({"ok": True, **{k: summary[k] for k in ("counts", "status", "checks")}, "results": summary.get("results")},
                     sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
