"""
Cross-lingual retrieval for Spanish questions. The guide index exists in EN and PL only, so a Spanish question is
searched against the EN index; lexical BM25 then misses the source chunk on about half of them (bank: ES 0.489 vs
EN 0.782, 2026-10-17). Fix: translate the Spanish question to English with the deployed model (temperature 0,
thinking off, Spanish legal terms kept in parentheses) and search the EN index with the translation.
Measured on the frozen bank (734 ES questions) and the held-out validation items (29), source chunk retrieved:
  original ES 0.489 / 0.621 · translation 0.703 / 0.931 · translation + original 0.666 / 0.897 ·
  ceiling (the bank's own EN question) 0.782 / 1.000. The model still answers the Spanish question.

  query_for_index(q, lang, model) → the string to pass to BM25.retrieve
"""
from __future__ import annotations

import functools
import json
import urllib.request

import run_eval as ev

TRANSLATE_SYSTEM = ("You translate questions about Spanish property law from Spanish into English. Keep every "
                    "Spanish legal term, tax name, acronym, law name and number exactly as written, and add it in "
                    "parentheses after its English rendering when there is one. Output only the translated question.")
OPTIONS = {"temperature": 0.0, "seed": 42, "num_predict": 200, "num_ctx": 2048}


@functools.lru_cache(maxsize=4096)
def translate_es_en(q: str, model: str) -> str:
    body = json.dumps({"model": model, "stream": False, "think": False, "options": OPTIONS,
                       "messages": [{"role": "system", "content": TRANSLATE_SYSTEM},
                                    {"role": "user", "content": q}]}).encode()
    req = urllib.request.Request(f"{ev.OLLAMA}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["message"]["content"].strip()


def query_for_index(q: str, lang: str, model: str) -> str:
    """ES: the English translation; EN / PL: unchanged."""
    return translate_es_en(q, model) if lang == "es" else q
