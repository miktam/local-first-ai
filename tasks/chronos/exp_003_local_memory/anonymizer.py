"""
anonymizer.py — Pure substitution. No LLM in this module.

Takes raw text + an entity list, produces anonymized text. Also flags
"fingerprint" passages — phrases distinctive enough that the LLM might
recognise the source even after names are gone.

Trust zone: imports vocab_store; does NOT import the LLM client. The
vocab store is consulted only to register/lookup pseudonyms and is never
exposed downstream.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

from vocab_store import VocabStore


@dataclass
class Entity:
    """An entity to anonymize.

    `surface_forms` includes every spelling/variant that should be replaced
    with the same pseudonym. For "Tyler Durden":
        surface_forms = ["Tyler Durden", "Mr. Durden", "Tyler"]

    The anonymizer sorts forms by length (descending) before substituting,
    so longer forms are replaced first. Otherwise "Tyler Durden" would
    become "<pseudo> Durden" with "Durden" left dangling.
    """
    canonical: str            # canonical real name (key in the vocab store)
    kind: str                 # "person" | "org" | "place"
    surface_forms: List[str]  # all spellings to substitute


def anonymize(text: str, entities: List[Entity], vocab: VocabStore) -> str:
    """Replace every surface form with its (stable) pseudonym.

    Implementation notes:
    - We sort all (form, pseudonym) pairs by form length, descending. The
      longest match runs first so "Tyler Durden" wins over "Tyler" alone.
    - Word boundaries (\\b) prevent accidental substring replacement
      (so "Marla" doesn't eat "Marlal" if such a word existed). Falls
      back to plain replace for forms containing punctuation.
    """
    pairs: List[Tuple[str, str]] = []
    for ent in entities:
        pseudo = vocab.register(ent.canonical, ent.kind)
        for form in ent.surface_forms:
            pairs.append((form, pseudo))

    pairs.sort(key=lambda p: len(p[0]), reverse=True)

    out = text
    for form, pseudo in pairs:
        # If the form is purely letters + spaces, use word boundaries.
        if re.fullmatch(r"[A-Za-z][A-Za-z\s]*", form):
            out = re.sub(r"\b" + re.escape(form) + r"\b", pseudo, out)
        else:
            out = out.replace(form, pseudo)
    return out


# --- Fingerprint detection -------------------------------------------------

@dataclass
class Fingerprint:
    """A canonical phrase or pattern that may give away the source."""
    label: str
    pattern: str  # regex; matched case-insensitively
    note: str     # why this is risky / hint for the human paraphraser

def find_fingerprints(
    text: str,
    fingerprints: List[Fingerprint],
) -> List[Tuple[Fingerprint, str]]:
    """Return list of (fingerprint, matched_excerpt) tuples.

    Run this AFTER substitution. Anything matched here should be
    paraphrased by hand. Do NOT auto-paraphrase by feeding the original
    to the LLM — that defeats the whole point.
    """
    hits: List[Tuple[Fingerprint, str]] = []
    for fp in fingerprints:
        for m in re.finditer(fp.pattern, text, flags=re.IGNORECASE):
            # Pull a small window of context around the match for review.
            start = max(0, m.start() - 40)
            end = min(len(text), m.end() + 40)
            excerpt = text[start:end].replace("\n", " ").strip()
            hits.append((fp, excerpt))
    return hits


def write_review_report(
    hits: List[Tuple[Fingerprint, str]],
    path: Path,
) -> None:
    """Write a human-review report listing fingerprints needing paraphrase.

    The report contains anonymized excerpts — safe to keep alongside the
    anonymized corpus. It still narrows attention to specific lines, so
    treat it with the same care as the corpus itself.
    """
    path = Path(path)
    with path.open("w", encoding="utf-8") as f:
        if not hits:
            f.write("No fingerprints detected.\n\n")
            f.write("Either the text is clean, or your fingerprint list is "
                    "incomplete. If the source is well-known, prefer the "
                    "second explanation.\n")
            return
        f.write(f"# Fingerprint review — {len(hits)} item(s)\n\n")
        f.write("These passages may give the source away even after\n")
        f.write("anonymization. Paraphrase by hand. Do NOT use the LLM\n")
        f.write("to paraphrase the original.\n\n")
        for fp, excerpt in hits:
            f.write(f"## [{fp.label}]\n")
            f.write(f"_Why it's risky: {fp.note}_\n\n")
            f.write(f"> …{excerpt}…\n\n")
