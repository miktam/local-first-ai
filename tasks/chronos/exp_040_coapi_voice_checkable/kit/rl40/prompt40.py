"""
exp_040 — the one prompt renderer (train = filter = serve). Every exp_040 prompt is exp_039's "prompt" mode
(glue's language-rendered system prompt, the advice suffix when glue's _ADVICE fires, the retrieved material
inside the user turn, k = glue.K, budget = glue.BUDGET, ES retrieves from EN) plus reward40.FINAL_LINE[lang]
appended last to the system prompt. A13 = A12's glue with the same sentence; it must import system40 from here.

Categorical and yes/no items show their options after the question (options_line): retrieval runs on the bare
question, as at serve; the options go only into the user turn. Without them the policy cannot know the closed set
and a right answer in other words scores "missing" (filter run 2026-10-15: 344 of 427 categorical questions named
no candidate). The v3 harness renders v3 rows through the same function.

Abstention rows withhold chunks as v3 does (split_manifest_v3 "retrieval"): the withheld chunks are removed from
the index and BM25 is rebuilt over the rest (index_without), so idf and the vocab tier see the reduced guide.
Abstention rows show no options (v3's abstention rows carry no candidates).
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CV = HERE.parent
sys.path[:0] = [str(CV), str(HERE)]
import glue as G          # noqa: E402
import guide_bm25 as g    # noqa: E402
import reward40 as W      # noqa: E402

OPTIONS_LABEL = {"en": "Options", "pl": "Opcje", "es": "Opciones"}


def system40(q: str, lang: str) -> str:
    return G.SYSTEM[lang] + (G.ADVICE_SUFFIX[lang] if G._ADVICE.search(q) else "") + W.FINAL_LINE[lang]


def options_line(candidates: dict, lang: str, key: str) -> str:
    """'Options: seller / buyer / both.' — each candidate's first form in lang, order shuffled by key (pass the
    twin root so both twins show the same order)."""
    names = sorted(candidates)
    random.Random(f"40:opts:{key}").shuffle(names)
    return f"{OPTIONS_LABEL[lang]}: " + " / ".join((candidates[n].get(lang) or [n])[0] for n in names) + "."


_WITHOUT: dict = {}


def index_without(idx: dict, lang: str, withhold) -> g.BM25:
    """The language's index rebuilt without the withheld chunk ids (cached). A withhold list may name chunks of both
    indexes (v3's abstention rows list en: and pl: ids); only this index's ids apply. Refuses an id of this index
    that it cannot find, and a list with no id for this index (nothing would be withheld)."""
    ilang = "en" if lang == "es" else lang
    base = idx[ilang]
    drop = frozenset(i for i in withhold if i.split(":", 1)[0] == ilang)
    key = (id(base), drop)
    if key not in _WITHOUT:
        if not drop:
            raise ValueError(f"no withheld id belongs to the {ilang} index")
        missing = drop - {c["id"] for c in base.chunks}
        if missing:
            raise ValueError(f"withheld ids not in the {ilang} index: {sorted(missing)[:3]}")
        _WITHOUT[key] = g.BM25([c for c in base.chunks if c["id"] not in drop])
    return _WITHOUT[key]


def pl_section(chunk: str, idx: dict) -> list[str]:
    """Every PL chunk id of an EN chunk's section (module, tema, kind) — what a PL abstention row withholds."""
    section = "pl:" + ":".join(chunk.split(":")[1:4]) + ":"
    return [c["id"] for c in idx["pl"].chunks if c["id"].startswith(section)]


def render40(q: str, lang: str, idx: dict, options: str | None = None, withhold=()) -> dict:
    """{system, user, context, chunk_ids, shown_q} for question q rendered in lang; options (if any) follow q;
    withhold = chunk ids removed from the index before retrieval (abstention rows)."""
    index = index_without(idx, lang, withhold) if withhold else idx["en" if lang == "es" else lang]
    chunks = index.retrieve(q, k=G.K, budget=G.BUDGET)
    context = g.context_block(chunks)
    shown = f"{q}\n{options}" if options else q
    user = f"{G.CONTEXT_LABEL[lang]}:\n{context}\n\n{G.QUESTION_LABEL[lang]}:\n{shown}"
    return {"system": system40(q, lang), "user": user, "context": context, "chunk_ids": [c["id"] for c in chunks],
            "shown_q": shown}


def source_retrieved(chunk: str, lang: str, chunk_ids: list[str]) -> bool:
    """Is the item's source chunk (an EN id) in the material? EN and ES: the exact EN chunk. PL: any PL chunk
    of the same section (module, tema, kind) — EN↔PL chunk ids are positional and misalign in 14 sections."""
    if lang != "pl":
        return chunk in chunk_ids
    section = "pl:" + ":".join(chunk.split(":")[1:4]) + ":"
    return any(c.startswith(section) for c in chunk_ids)
