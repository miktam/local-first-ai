"""
paths.py — Single source of truth for the project's data layout.

This module IS the orchestration manifest. Every script reads paths from
here. If you want to know where a piece of data lives, you look here.
If you want to relocate something, you change it here and only here.

Project data flow
=================

  ┌──────────────────────────────────────────────────────────────┐
  │  INPUTS  (you provide)                          source/      │
  │    fight_club.txt              raw source text               │
  │    entities.json               which names to substitute     │
  │    fingerprints.json           which phrases to flag         │
  └──────────────────────┬───────────────────────────────────────┘
                         │  python run.py anonymize
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  INTERMEDIATE                                  intermediate/ │
  │    fight_club.anon.txt         machine-substituted           │
  │    fingerprint_review.md       human paraphrase checklist    │
  │    fight_club.anon.final.txt   you produce this by hand      │
  └──────────────────────┬───────────────────────────────────────┘
                         │
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  VAULT  (crown jewels; chmod 600; encrypt at rest)   vault/  │
  │    vocab.json                  reversible identity           │
  └──────────────────────────────────────────────────────────────┘
                         │  python run.py ingest / summarise / decay
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  MEMORY  (runtime state)                       memory_data/  │
  │    layer1.jsonl                raw turns, append-only        │
  │    layer2.json                 CURRENT daily summaries       │
  │    layer2_archive/             DECAYED summaries             │
  │      YYYY-MM-DD/                                             │
  │        v1_<archived_at>.json   first version (longest)       │
  │        v2_<archived_at>.json   compressed (shorter)          │
  │        v3_<archived_at>.json   compressed again              │
  └──────────────────────────────────────────────────────────────┘
                         │  python run.py pre-flight / canary
                         ▼
  ┌──────────────────────────────────────────────────────────────┐
  │  RESULTS  (evidence for Experiment 003)            results/  │
  │    pre-flight_YYYYMMDDTHHMMSS.json                           │
  │    canary_YYYYMMDDTHHMMSS.json                               │
  └──────────────────────────────────────────────────────────────┘

Trust zones (which directories the LLM may see):
  - source/           : LLM may NEVER see (contains real names)
  - intermediate/     : LLM may see (anonymized only)
  - vault/            : LLM may NEVER see
  - memory_data/      : LLM may see (anonymized only)
  - results/          : LLM may see (already public-grade evidence)
"""

from datetime import datetime
from pathlib import Path
from typing import Optional


# Project root: directory containing this file.
ROOT = Path(__file__).parent.resolve()

# --- Inputs (user-provided) -------------------------------------------------
SOURCE_DIR              = ROOT / "source"
DEFAULT_SOURCE_TEXT     = SOURCE_DIR / "fight_club.txt"
DEFAULT_ENTITIES        = SOURCE_DIR / "entities.json"
DEFAULT_FINGERPRINTS    = SOURCE_DIR / "fingerprints.json"

# --- Intermediate artefacts -------------------------------------------------
INTERMEDIATE_DIR        = ROOT / "intermediate"
DEFAULT_ANON_TEXT       = INTERMEDIATE_DIR / "fight_club.anon.txt"
DEFAULT_ANON_FINAL      = INTERMEDIATE_DIR / "fight_club.anon.final.txt"
DEFAULT_REVIEW          = INTERMEDIATE_DIR / "fingerprint_review.md"

# --- Vault ------------------------------------------------------------------
VAULT_DIR               = ROOT / "vault"
DEFAULT_VAULT           = VAULT_DIR / "vocab.json"

# --- Runtime memory ---------------------------------------------------------
MEMORY_DIR              = ROOT / "memory_data"
LAYER1_FILE             = MEMORY_DIR / "layer1.jsonl"
LAYER2_FILE             = MEMORY_DIR / "layer2.json"
LAYER2_ARCHIVE_DIR      = MEMORY_DIR / "layer2_archive"

# --- Results / evidence -----------------------------------------------------
RESULTS_DIR             = ROOT / "results"


def ensure_directories() -> None:
    """Create all standard directories. Safe to call repeatedly."""
    for d in (
        SOURCE_DIR, INTERMEDIATE_DIR, VAULT_DIR,
        MEMORY_DIR, LAYER2_ARCHIVE_DIR, RESULTS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)


def _safe_ts(ts_iso: str) -> str:
    """Make an ISO timestamp safe for filenames on every OS."""
    return ts_iso.replace(":", "-").replace(".", "-")


def archive_path(day_iso: str, version: int, archived_at_iso: str) -> Path:
    """Path for a specific archived version of a day's summary."""
    folder = LAYER2_ARCHIVE_DIR / day_iso
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"v{version}_{_safe_ts(archived_at_iso)}.json"


def results_path(kind: str, ts: Optional[datetime] = None) -> Path:
    """Path for a results JSON file. kind: 'pre-flight' or 'canary'."""
    ts = ts or datetime.utcnow()
    stamp = ts.strftime("%Y%m%dT%H%M%S")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR / f"{kind}_{stamp}.json"
