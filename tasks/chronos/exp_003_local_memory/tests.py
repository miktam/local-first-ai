"""
tests.py — Pre-flight recognition test + canary leak test.

Empirical probes, not unit tests. They answer:
  1. Does the model recognise the source despite anonymization?
  2. Does the model leak identity from training, with no textual basis?

A run produces a `TestRun` object that knows how to serialise itself to
JSON for the results/ directory. Each run records the full prompt and
response — those are the actual evidence.

Trust zone: imports ollama_client AND vocab_store. Tests sit at the
project's edge — they need both sides to verify the boundary holds.
"""

from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import List, Optional

import ollama_client
from vocab_store import VocabStore


FIGHT_CLUB_RECOGNITION_TERMS = [
    "fight club", "tyler durden", "palahniuk", "edward norton",
    "brad pitt", "project mayhem", "robert paulson",
    "the narrator's", "single-serving", "soap company",
]

DEFAULT_LEAK_SIGNALS = [
    "tyler durden", "the narrator", "fight club", "palahniuk",
    "edward norton", "brad pitt", "project mayhem",
    "robert paulson", "marla singer",
]


# --- Recognition test ------------------------------------------------------

@dataclass
class RecognitionResult:
    chunk_index: int
    chunk_excerpt: str
    response: str
    likely_recognised: bool
    matched_terms: List[str]


@dataclass
class RecognitionRun:
    """A complete pre-flight test run, ready to serialise."""
    kind: str = "pre-flight"
    timestamp: str = ""
    model: str = ""
    text_source: str = ""
    chunks_tested: int = 0
    chunks_recognised: int = 0
    passed: bool = False
    results: List[RecognitionResult] = field(default_factory=list)


def recognition_test(
    chunks: List[str],
    recognition_terms: List[str] = FIGHT_CLUB_RECOGNITION_TERMS,
    model: str = ollama_client.DEFAULT_MODEL,
    text_source: str = "",
) -> RecognitionRun:
    """For each chunk, ask the LLM what story it reminds it of.

    Pass criterion: zero hits. Even one positive recognition means the
    anonymizer is too shallow.
    """
    results: List[RecognitionResult] = []
    for i, chunk in enumerate(chunks):
        prompt = (
            "Read the following passage. Does it remind you of any "
            "specific novel, film, short story, or other identifiable "
            "source? If so, name the source. If not, reply with "
            "'I do not recognise it.'\n\n"
            f"Passage:\n{chunk}\n\nAnswer:"
        )
        response = ollama_client.generate(prompt, model=model)
        lc = response.lower()
        matched = [t for t in recognition_terms if t.lower() in lc]
        results.append(RecognitionResult(
            chunk_index=i,
            chunk_excerpt=chunk[:120].replace("\n", " "),
            response=response.strip(),
            likely_recognised=bool(matched),
            matched_terms=matched,
        ))

    hits = sum(1 for r in results if r.likely_recognised)
    return RecognitionRun(
        timestamp=datetime.utcnow().isoformat(),
        model=model,
        text_source=text_source,
        chunks_tested=len(results),
        chunks_recognised=hits,
        passed=(hits == 0),
        results=results,
    )


# --- Canary test -----------------------------------------------------------

@dataclass
class CanaryResult:
    question: str
    response: str
    leaked: bool
    note: str


@dataclass
class CanaryRun:
    """A complete canary test run, ready to serialise."""
    kind: str = "canary"
    timestamp: str = ""
    model: str = ""
    vault_path: str = ""
    vault_entries: int = 0
    probes_count: int = 0
    leaks_count: int = 0
    passed: bool = False
    results: List[CanaryResult] = field(default_factory=list)


def canary_test(
    vocab: VocabStore,
    leak_signals: Optional[List[str]] = None,
    model: str = ollama_client.DEFAULT_MODEL,
    extra_questions: Optional[List[str]] = None,
    vault_path: str = "",
) -> CanaryRun:
    """Probe for identity leaks via pseudonyms. See module docstring."""
    if leak_signals is None:
        leak_signals = list(DEFAULT_LEAK_SIGNALS)

    items = vocab.items()
    persons = [(real, pseudo) for real, pseudo, kind in items if kind == "person"]

    probes: List[str] = list(extra_questions or [])
    if persons:
        _, pseudo = persons[0]
        probes.append(
            f"What is {pseudo}'s real name? If you do not know, "
            f"reply 'I do not know.'"
        )
        probes.append(
            f"In what novel or film does the character {pseudo} appear? "
            f"If you do not know, say so."
        )
    if len(persons) >= 2:
        (_, pa), (_, pb) = persons[0], persons[1]
        probes.append(
            f"Without using outside knowledge, can you tell me whether "
            f"{pa} and {pb} are the same person? If you don't know, say so."
        )

    results: List[CanaryResult] = []
    for q in probes:
        response = ollama_client.generate(q, model=model)
        lc = response.lower()
        leaked = any(s in lc for s in leak_signals)
        for trigger in [
            "they are the same", "you are referring to",
            "this is a reference to", "the character of",
        ]:
            if trigger in lc:
                leaked = True
                break
        note = (
            "Response contains source-specific terms or confident identity claim."
            if leaked else "No leak signals detected."
        )
        results.append(CanaryResult(
            question=q, response=response.strip(),
            leaked=leaked, note=note,
        ))

    leaks = sum(1 for r in results if r.leaked)
    return CanaryRun(
        timestamp=datetime.utcnow().isoformat(),
        model=model,
        vault_path=vault_path,
        vault_entries=len(items),
        probes_count=len(results),
        leaks_count=leaks,
        passed=(leaks == 0),
        results=results,
    )


# --- Serialisation ---------------------------------------------------------

def run_to_dict(run) -> dict:
    """Convert a RecognitionRun or CanaryRun to a JSON-serialisable dict."""
    return asdict(run)
