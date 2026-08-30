#!/usr/bin/env python3
"""
Exp 026 — Contextual retrieval on the COAPI corpus, fully local.

Reuses CasaSol's actual production chunking (index_coapi.py::clean_text/
chunk_text) and embedding function (embedder.py::get_ef) so this tests the
real pipeline, not an invented one. Never touches the real coapi_knowledge_en
collection — uses ephemeral in-memory ChromaDB collections only.
"""
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
import chromadb

CASASOL_ROOT = Path("/Users/miktam02/REPOS/casasol")
sys.path.insert(0, str(CASASOL_ROOT))
sys.path.insert(0, str(CASASOL_ROOT / "scripts"))

from scripts.index_coapi import clean_text, chunk_text
from scripts.embedder import get_ef, MODEL_NAME

OLLAMA_BASE_URL = "http://localhost:11434"
MODULE_FILE = CASASOL_ROOT / "tasks/coapi/materials/text/en/10.modulo_6_el_contrato.txt"
CHUNK_WORDS = 500
OVERLAP_WORDS = 50
TOP_K = 3

QUERY_GEN_MODEL = "gemma4:26b"          # ground truth — capability, not the thing under test
CONTEXT_MODELS = ["gemma4:e4b", "gemma4:26b"]  # both tested for H2

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TS = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

CONTEXT_PROMPT_TEMPLATE = """Here is a chunk from Module 6 of a Spanish real estate professional certification course:

<chunk>
{chunk}
</chunk>

Here is the full module for context:

<module>
{module}
</module>

Please give a short, succinct context (1-2 sentences) situating this chunk within the module, to improve search retrieval of the chunk. Answer with only the context, nothing else."""

QUERY_GEN_PROMPT_TEMPLATE = """Here is a chunk from a Spanish real estate professional certification course (Module 6, The Contract):

<chunk>
{chunk}
</chunk>

Write one realistic question a course trainee might ask, for which this chunk is the best answer. Be specific enough that the question is clearly about content in THIS chunk, not the module in general. Answer with only the question, nothing else."""


def ollama_generate(model: str, prompt: str, num_predict: int = 150) -> tuple[str, dict]:
    t0 = time.time()
    r = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model, "prompt": prompt, "stream": False, "think": False,
            "options": {"temperature": 0.3, "num_predict": num_predict},
        },
        timeout=300,
    )
    r.raise_for_status()
    data = r.json()
    meta = {
        "wall_clock_s": round(time.time() - t0, 2),
        "prompt_tokens": data.get("prompt_eval_count", 0),
        "output_tokens": data.get("eval_count", 0),
        "gen_tok_s": round(data.get("eval_count", 0) / max(data.get("eval_duration", 1) / 1e9, 0.001), 2),
    }
    return data["response"].strip(), meta


def main():
    raw = MODULE_FILE.read_text(encoding="utf-8")
    cleaned = clean_text(raw)
    chunks = [c for c in chunk_text(cleaned, CHUNK_WORDS, OVERLAP_WORDS) if len(c.split()) >= 30]
    print(f"Module: {MODULE_FILE.name} — {len(cleaned.split())} words cleaned, {len(chunks)} chunks")

    ef = get_ef()

    # ── Step 1: ground-truth query per chunk ────────────────────────────
    print("\n=== Generating ground-truth queries (gemma4:26b) ===")
    queries = []
    for i, chunk in enumerate(chunks):
        q, meta = ollama_generate(QUERY_GEN_MODEL, QUERY_GEN_PROMPT_TEMPLATE.format(chunk=chunk), num_predict=60)
        queries.append(q)
        print(f"  [{i}] {q[:80]!r} ({meta['wall_clock_s']}s)")

    # ── Step 2: contextual prefixes, both models ────────────────────────
    prefixes = {m: [] for m in CONTEXT_MODELS}
    context_meta = {m: [] for m in CONTEXT_MODELS}
    for model in CONTEXT_MODELS:
        print(f"\n=== Generating contextual prefixes ({model}) ===")
        for i, chunk in enumerate(chunks):
            prefix, meta = ollama_generate(
                model, CONTEXT_PROMPT_TEMPLATE.format(chunk=chunk, module=cleaned), num_predict=100
            )
            prefixes[model].append(prefix)
            context_meta[model].append(meta)
            print(f"  [{i}] {prefix[:80]!r} ({meta['wall_clock_s']}s, {meta['prompt_tokens']} prompt tok)")

    # ── Step 3: build three collections, embed ──────────────────────────
    client = chromadb.EphemeralClient()
    collections = {}
    collections["unprefixed"] = client.create_collection("unprefixed", embedding_function=ef)
    collections["unprefixed"].add(
        ids=[f"chunk_{i}" for i in range(len(chunks))], documents=chunks
    )
    for model in CONTEXT_MODELS:
        key = f"prefixed_{model}"
        collections[key] = client.create_collection(key.replace(":", "_").replace(".", "_"), embedding_function=ef)
        prefixed_docs = [f"{prefixes[model][i]}\n\n{chunks[i]}" for i in range(len(chunks))]
        collections[key].add(ids=[f"chunk_{i}" for i in range(len(chunks))], documents=prefixed_docs)

    # ── Step 4: retrieval, per variant ───────────────────────────────────
    print("\n=== Retrieval accuracy ===")
    hit_results = {name: [] for name in collections}
    for i, query in enumerate(queries):
        target = f"chunk_{i}"
        neighbors = {f"chunk_{i-1}", target, f"chunk_{i+1}"}
        for name, col in collections.items():
            res = col.query(query_texts=[query], n_results=TOP_K)
            retrieved = set(res["ids"][0])
            hit = bool(retrieved & neighbors)
            exact_hit = target in retrieved
            hit_results[name].append({"query_idx": i, "hit_with_neighbor_tolerance": hit, "exact_hit": exact_hit,
                                       "retrieved": res["ids"][0]})

    summary = {}
    for name, results in hit_results.items():
        n = len(results)
        summary[name] = {
            "top3_hit_rate_with_neighbor_tolerance": round(sum(r["hit_with_neighbor_tolerance"] for r in results) / n, 3),
            "top3_exact_hit_rate": round(sum(r["exact_hit"] for r in results) / n, 3),
        }
        print(f"  {name:25s} neighbor-tolerant={summary[name]['top3_hit_rate_with_neighbor_tolerance']:.3f} "
              f"exact={summary[name]['top3_exact_hit_rate']:.3f}")

    # ── Step 5: cost extrapolation (H3) ──────────────────────────────────
    import config as casasol_config
    real_col = chromadb.PersistentClient(path=str(casasol_config.CHROMA_DIR)).get_collection("coapi_knowledge_en")
    full_corpus_size = real_col.count()

    e4b_times = [m["wall_clock_s"] for m in context_meta["gemma4:e4b"]]
    mean_e4b_time = sum(e4b_times) / len(e4b_times)
    extrapolated_hours = round(mean_e4b_time * full_corpus_size / 3600, 2)

    result = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "module": MODULE_FILE.name,
        "n_chunks_tested": len(chunks),
        "queries": queries,
        "hit_results": hit_results,
        "summary": summary,
        "context_generation_meta": context_meta,
        "h3_extrapolation": {
            "full_corpus_size_coapi_en": full_corpus_size,
            "mean_seconds_per_chunk_gemma4_e4b": round(mean_e4b_time, 2),
            "extrapolated_full_corpus_hours": extrapolated_hours,
        },
    }
    out = RESULTS_DIR / f"run_{TS}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {out}")
    print(f"\nH3: full corpus ({full_corpus_size} chunks) at gemma4:e4b's "
          f"{mean_e4b_time:.1f}s/chunk ≈ {extrapolated_hours}h one-time cost")


if __name__ == "__main__":
    main()
