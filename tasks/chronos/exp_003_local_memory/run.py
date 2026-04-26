"""
run.py — Orchestrator and CLI.

Defaults come from paths.py — the manifest. You can override any path on
the command line, but if you don't, the layout under the project root is
used as-is.

Subcommands:
  setup       Create the standard directory tree.
  anonymize   Substitute entities, flag fingerprints.
  pre-flight  Verify the LLM no longer recognises the source.
  ingest      Load anonymized chunks into Layer 1.
  summarise   Build today's Layer 2 summary.
  query       Ask a question; Layer 2 -> Layer 1 fallback.
  decay       Compress Layer 2 summaries older than N days.
  restore     Restore an archived summary back into Layer 2.
  archive     List archived versions for a given day.
  canary      Probe for identity leaks via pseudonyms.

Each test command writes a structured JSON result file under results/
and exits non-zero on failure so this can slot into a Makefile.
"""

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date, datetime
from pathlib import Path
from typing import List

import ollama_client
import paths
import tests
from anonymizer import (
    Entity, Fingerprint,
    anonymize, find_fingerprints, write_review_report,
)
from memory import Memory
from vocab_store import VocabStore


# --- helpers --------------------------------------------------------------

def _load_entities(path: Path) -> List[Entity]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Entity(**e) for e in raw]


def _load_fingerprints(path: Path) -> List[Fingerprint]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return [Fingerprint(**fp) for fp in raw]


def _write_results(run, kind: str) -> Path:
    """Write a test run to results/ and return the path."""
    out_path = paths.results_path(kind)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(tests.run_to_dict(run), f, indent=2, ensure_ascii=False)
    return out_path


# --- commands -------------------------------------------------------------

def cmd_setup(args):
    paths.ensure_directories()
    print(f"[setup] created standard directories under {paths.ROOT}")
    for d in [paths.SOURCE_DIR, paths.INTERMEDIATE_DIR, paths.VAULT_DIR,
              paths.MEMORY_DIR, paths.LAYER2_ARCHIVE_DIR, paths.RESULTS_DIR]:
        print(f"        {d.relative_to(paths.ROOT)}/")


def cmd_anonymize(args):
    paths.ensure_directories()
    text = Path(args.input).read_text(encoding="utf-8")
    entities = _load_entities(Path(args.entities))
    fingerprints = (
        _load_fingerprints(Path(args.fingerprints)) if args.fingerprints else []
    )

    vocab_path = Path(args.vocab)
    vocab = VocabStore.load(vocab_path) if vocab_path.exists() else VocabStore()

    anon = anonymize(text, entities, vocab)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(anon, encoding="utf-8")
    vocab.save(vocab_path)

    print(f"[anonymize] wrote {args.output}  "
          f"({len(anon)} chars, {len(vocab.items())} vault entries)")
    print(f"[anonymize] vault: {vocab_path}  (chmod 600)")

    if fingerprints:
        hits = find_fingerprints(anon, fingerprints)
        write_review_report(hits, Path(args.review))
        print(f"[anonymize] {len(hits)} fingerprint hit(s) -> {args.review}")
        if hits:
            print("[anonymize] review and paraphrase before pre-flight.")


def cmd_preflight(args):
    if not ollama_client.healthcheck():
        print("[pre-flight] Ollama not reachable", file=sys.stderr)
        sys.exit(1)

    text = Path(args.text).read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sample = paragraphs[: args.chunks]
    print(f"[pre-flight] testing {len(sample)} chunk(s)...")

    run = tests.recognition_test(
        sample, model=args.model, text_source=str(args.text),
    )
    out_path = _write_results(run, "pre-flight")

    for r in run.results:
        marker = "RECOGNISED" if r.likely_recognised else "ok"
        print(f"  [{marker:10s}] chunk {r.chunk_index}: {r.chunk_excerpt!r}")
        if r.likely_recognised:
            print(f"             matched: {r.matched_terms}")
            print(f"             response: {r.response[:200]}")
    print()
    print(f"[pre-flight] result: {run.chunks_recognised}/{run.chunks_tested} recognised")
    print(f"[pre-flight] saved:  {out_path}")
    if not run.passed:
        print("[pre-flight] FAIL: anonymization too shallow. Abort.")
        sys.exit(2)
    print("[pre-flight] PASS.")


def cmd_ingest(args):
    text = Path(args.text).read_text(encoding="utf-8")
    mem = Memory(model=args.model)
    chunks = [c.strip() for c in text.split("\n\n") if c.strip()]
    for c in chunks:
        mem.ingest(c, source=args.source)
    print(f"[ingest] {len(chunks)} chunk(s) -> Layer 1 ({mem.layer1_path})")


def cmd_summarise(args):
    if not ollama_client.healthcheck():
        print("[summarise] Ollama not reachable", file=sys.stderr)
        sys.exit(1)
    mem = Memory(model=args.model)
    day = date.fromisoformat(args.day) if args.day else date.today()
    entry = mem.build_daily_summary(day, force=args.force)
    print(f"[summarise] {entry.date} v{entry.version}: "
          f"{len(entry.turn_ids)} turns -> {entry.word_count} words")
    print()
    print(entry.summary)


def cmd_query(args):
    if not ollama_client.healthcheck():
        print("[query] Ollama not reachable", file=sys.stderr)
        sys.exit(1)
    mem = Memory(model=args.model)
    print(mem.query(args.q))


def cmd_decay(args):
    if not ollama_client.healthcheck():
        print("[decay] Ollama not reachable", file=sys.stderr)
        sys.exit(1)
    mem = Memory(model=args.model)
    decayed = mem.decay_old_summaries(
        older_than_days=args.older_than,
        target_words=args.target_words,
    )
    if not decayed:
        print(f"[decay] no summaries older than {args.older_than} days "
              f"need compression.")
        return
    print(f"[decay] compressed {len(decayed)} summary(ies):")
    for d in decayed:
        archives = mem.list_archive(d)
        print(f"        {d}  -> archive now has {len(archives)} version(s)")


def cmd_archive(args):
    mem = Memory(model=args.model)
    archives = mem.list_archive(args.day)
    if not archives:
        print(f"[archive] no archived versions for {args.day}")
        return
    print(f"[archive] {len(archives)} version(s) for {args.day}:")
    for path in archives:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  {path.name}")
        print(f"      version: v{data['version']}")
        print(f"      reason:  {data.get('archive_reason', '?')}")
        print(f"      words:   {data['word_count']}")
        print(f"      created: {data['created_at']}")
        print(f"      archived:{data.get('archived_at', '?')}")


def cmd_restore(args):
    mem = Memory(model=args.model)
    archive_path = Path(args.archive)
    if not archive_path.exists():
        print(f"[restore] no such archive file: {archive_path}", file=sys.stderr)
        sys.exit(1)
    restored = mem.restore_archive(args.day, archive_path)
    print(f"[restore] {args.day}: restored to v{restored.version} "
          f"({restored.word_count} words)")


def cmd_canary(args):
    if not ollama_client.healthcheck():
        print("[canary] Ollama not reachable", file=sys.stderr)
        sys.exit(1)
    vocab = VocabStore.load(Path(args.vocab))
    extra = []
    if args.extra:
        extra = [
            line.strip()
            for line in Path(args.extra).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    run = tests.canary_test(
        vocab, model=args.model, extra_questions=extra,
        vault_path=str(args.vocab),
    )
    out_path = _write_results(run, "canary")

    for r in run.results:
        marker = "LEAK" if r.leaked else "ok"
        print(f"  [{marker:4s}] {r.question}")
        print(f"         response: {r.response[:200]}")
        print(f"         note: {r.note}")
    print()
    print(f"[canary] result: {run.leaks_count}/{run.probes_count} leaked")
    print(f"[canary] saved:  {out_path}")
    if not run.passed:
        print("[canary] FAIL.")
        sys.exit(2)
    print("[canary] PASS.")


# --- main -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Local-first memory validator.")
    p.add_argument("--model", default=ollama_client.DEFAULT_MODEL,
                   help=f"Ollama model name (default: {ollama_client.DEFAULT_MODEL})")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("setup", help="Create the standard directory tree.")
    s.set_defaults(func=cmd_setup)

    s = sub.add_parser("anonymize", help="Substitute entities + flag fingerprints.")
    s.add_argument("--input",        default=str(paths.DEFAULT_SOURCE_TEXT))
    s.add_argument("--entities",     default=str(paths.DEFAULT_ENTITIES))
    s.add_argument("--fingerprints", default=str(paths.DEFAULT_FINGERPRINTS))
    s.add_argument("--output",       default=str(paths.DEFAULT_ANON_TEXT))
    s.add_argument("--vocab",        default=str(paths.DEFAULT_VAULT))
    s.add_argument("--review",       default=str(paths.DEFAULT_REVIEW))
    s.set_defaults(func=cmd_anonymize)

    s = sub.add_parser("pre-flight", help="Probe whether the LLM still recognises the source.")
    s.add_argument("--text", default=str(paths.DEFAULT_ANON_FINAL))
    s.add_argument("--chunks", type=int, default=10)
    s.set_defaults(func=cmd_preflight)

    s = sub.add_parser("ingest", help="Load anonymized text into Layer 1.")
    s.add_argument("--text", default=str(paths.DEFAULT_ANON_FINAL))
    s.add_argument("--source", default="anonymized_text")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("summarise", help="Build today's Layer 2 summary.")
    s.add_argument("--day", help="YYYY-MM-DD (default: today UTC)")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_summarise)

    s = sub.add_parser("query", help="Ask a question; Layer 2 -> Layer 1 fallback.")
    s.add_argument("--q", required=True)
    s.set_defaults(func=cmd_query)

    s = sub.add_parser("decay", help="Compress Layer 2 summaries older than N days.")
    s.add_argument("--older-than", type=int, default=30,
                   help="Compress summaries created more than N days ago.")
    s.add_argument("--target-words", type=int, default=50,
                   help="Target word count for compressed summary.")
    s.set_defaults(func=cmd_decay)

    s = sub.add_parser("archive", help="List archived versions for a day.")
    s.add_argument("--day", required=True, help="YYYY-MM-DD")
    s.set_defaults(func=cmd_archive)

    s = sub.add_parser("restore", help="Restore an archived summary into Layer 2.")
    s.add_argument("--day", required=True, help="YYYY-MM-DD")
    s.add_argument("--archive", required=True, help="path to archive file")
    s.set_defaults(func=cmd_restore)

    s = sub.add_parser("canary", help="Probe for identity leaks via pseudonyms.")
    s.add_argument("--vocab", default=str(paths.DEFAULT_VAULT))
    s.add_argument("--extra", help="optional file: extra probe questions, one per line.")
    s.set_defaults(func=cmd_canary)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
