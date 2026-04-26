"""
memory.py — Dual-track local memory with archived decay.

  Layer 1 (Raw):    every chunk ever ingested, append-only. Never deleted.
  Layer 2 (Daily):  CURRENT daily summaries (full, ~300 words).
  Archive:          every PRIOR version of every Layer 2 summary, with
                    timestamps and reason. Nothing is destroyed; older
                    interpretations are themselves a kind of signal.

Decay rule:
  - Layer 1 never decays.
  - A Layer 2 summary can be COMPRESSED into a shorter version. The old
    longer version is moved to Layer 2 Archive before the shorter one
    replaces it. Repeat as needed.
  - When a summary is force-rebuilt from Layer 1, the prior version is
    archived first.
  - When Layer 2 disagrees with Layer 1, Layer 1 wins.
  - Any archived version can be restored back into Layer 2 on demand.

Trust zone: imports ollama_client. MUST NOT import vocab_store. The LLM
sees only anonymized text.
"""

# Note the deliberate absence: no `from vocab_store import ...`.
# Verify with:  grep '^import\|^from' memory.py | grep vocab_store
# (should print nothing)

import json
from dataclasses import dataclass, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import List, Optional

import ollama_client
import paths


@dataclass
class Turn:
    """A single ingested chunk. The atom of Layer 1."""
    id: int
    timestamp: str  # ISO format
    text: str       # ANONYMIZED text only
    source: str     # tag, e.g. "fight_club_chapter_03"


@dataclass
class DailySummary:
    """A Layer 2 entry: an LLM-generated summary of one day's turns."""
    date: str              # YYYY-MM-DD
    summary: str           # the summary text
    turn_ids: List[int]    # which Layer 1 turns this summary covers
    created_at: str        # ISO timestamp of creation
    last_accessed: str     # ISO timestamp of last read
    version: int = 1       # bumps each time the summary is replaced
    word_count: int = 0    # cached, for decay decisions

    def __post_init__(self):
        if not self.word_count:
            self.word_count = len(self.summary.split())


class Memory:
    """
    Append-only Layer 1 + cached Layer 2 + Layer 2 Archive.

    Defaults to the paths module; pass an explicit memory_dir to override
    (useful in tests).
    """

    def __init__(
        self,
        memory_dir: Optional[Path] = None,
        model: str = ollama_client.DEFAULT_MODEL,
    ):
        self.dir = Path(memory_dir) if memory_dir else paths.MEMORY_DIR
        self.dir.mkdir(parents=True, exist_ok=True)
        self.layer1_path = self.dir / "layer1.jsonl"
        self.layer2_path = self.dir / "layer2.json"
        self.archive_dir = self.dir / "layer2_archive"
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        self.model = model
        self._next_id = self._load_next_id()
        self._layer2 = self._load_layer2()

    # --- Persistence helpers ----------------------------------------------

    def _load_next_id(self) -> int:
        if not self.layer1_path.exists():
            return 0
        max_id = -1
        with self.layer1_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    max_id = max(max_id, json.loads(line)["id"])
        return max_id + 1

    def _load_layer2(self) -> dict:
        if not self.layer2_path.exists():
            return {}
        with self.layer2_path.open("r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: DailySummary(**v) for k, v in raw.items()}

    def _save_layer2(self) -> None:
        with self.layer2_path.open("w", encoding="utf-8") as f:
            json.dump(
                {k: asdict(v) for k, v in self._layer2.items()},
                f, indent=2, ensure_ascii=False,
            )

    # --- Layer 1: ingest --------------------------------------------------

    def ingest(
        self,
        text: str,
        source: str,
        timestamp: Optional[datetime] = None,
    ) -> Turn:
        """Append a chunk to Layer 1. Text MUST be anonymized already."""
        ts = (timestamp or datetime.utcnow()).isoformat()
        turn = Turn(id=self._next_id, timestamp=ts, text=text, source=source)
        self._next_id += 1
        with self.layer1_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(turn), ensure_ascii=False) + "\n")
        return turn

    def all_turns(self) -> List[Turn]:
        if not self.layer1_path.exists():
            return []
        out = []
        with self.layer1_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(Turn(**json.loads(line)))
        return out

    def turns_on(self, day: date) -> List[Turn]:
        prefix = day.isoformat()
        return [t for t in self.all_turns() if t.timestamp.startswith(prefix)]

    # --- Layer 2 Archive --------------------------------------------------

    def _archive(self, entry: DailySummary, reason: str) -> Path:
        """Move a summary to the archive. Returns the path written."""
        archived_at = datetime.utcnow().isoformat()
        record = {
            **asdict(entry),
            "archived_at": archived_at,
            "archive_reason": reason,
        }
        # Write under THIS Memory instance's archive dir (which may differ
        # from paths.LAYER2_ARCHIVE_DIR if a custom memory_dir was passed).
        folder = self.archive_dir / entry.date
        folder.mkdir(parents=True, exist_ok=True)
        safe_ts = archived_at.replace(":", "-").replace(".", "-")
        out_path = folder / f"v{entry.version}_{safe_ts}.json"
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        return out_path

    def list_archive(self, day_iso: str) -> List[Path]:
        """All archived versions for a given day, oldest first."""
        folder = self.archive_dir / day_iso
        if not folder.exists():
            return []
        return sorted(folder.glob("v*.json"))

    def restore_archive(self, day_iso: str, archive_path: Path) -> DailySummary:
        """Restore a specific archived version back into Layer 2.

        The CURRENT Layer 2 entry (if any) is itself archived first, so
        nothing is overwritten without a record.
        """
        with archive_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        # Drop archive-only fields before reconstructing the dataclass.
        data.pop("archived_at", None)
        data.pop("archive_reason", None)
        restored = DailySummary(**data)

        if day_iso in self._layer2:
            old = self._layer2[day_iso]
            self._archive(old, reason="replaced_by_restore")
            restored.version = old.version + 1

        restored.last_accessed = datetime.utcnow().isoformat()
        self._layer2[day_iso] = restored
        self._save_layer2()
        return restored

    # --- Layer 2: build, compress, decay ---------------------------------

    def build_daily_summary(self, day: date, force: bool = False) -> DailySummary:
        """Generate or rebuild Layer 2 for a single day.

        force=True regenerates from Layer 1 even if a cached summary
        exists. The previous summary is archived before replacement.
        """
        key = day.isoformat()
        if not force and key in self._layer2:
            entry = self._layer2[key]
            entry.last_accessed = datetime.utcnow().isoformat()
            self._save_layer2()
            return entry

        turns = self.turns_on(day)
        if not turns:
            raise ValueError(f"No Layer 1 turns on {key}")

        joined = "\n\n".join(f"[#{t.id}] {t.text}" for t in turns)
        system = (
            "You are a careful summariser. Given a sequence of text "
            "chunks, produce a concise factual summary covering events, "
            "people, places, and decisions. Quote exact phrases sparingly. "
            "Do not speculate about source material."
        )
        prompt = (
            "Summarise the following day's content in 200-300 words:\n\n"
            f"{joined}"
        )
        summary_text = ollama_client.generate(
            prompt, model=self.model, system=system,
        ).strip()

        now = datetime.utcnow().isoformat()
        if key in self._layer2:
            old = self._layer2[key]
            self._archive(old, reason="rebuilt_from_layer1")
            new_version = old.version + 1
        else:
            new_version = 1

        entry = DailySummary(
            date=key,
            summary=summary_text,
            turn_ids=[t.id for t in turns],
            created_at=now,
            last_accessed=now,
            version=new_version,
            word_count=len(summary_text.split()),
        )
        self._layer2[key] = entry
        self._save_layer2()
        return entry

    def compress_summary(
        self,
        day: date,
        target_words: int = 50,
    ) -> DailySummary:
        """
        Shorten an existing daily summary to ~target_words. Archives the
        longer version first. The compressed version replaces the entry
        in Layer 2.

        This is the "prune and shorten" decay path. Does NOT touch Layer 1.
        """
        key = day.isoformat()
        if key not in self._layer2:
            raise ValueError(f"No Layer 2 entry for {key}; nothing to compress")

        old = self._layer2[key]
        if old.word_count <= target_words:
            return old  # already small enough

        prompt = (
            f"Compress the following summary to roughly {target_words} "
            f"words. Keep concrete facts (people, places, decisions). "
            f"Drop framing, atmosphere, and elaboration.\n\n"
            f"Summary:\n{old.summary}\n\nCompressed:"
        )
        compressed = ollama_client.generate(prompt, model=self.model).strip()

        self._archive(old, reason="compressed")

        now = datetime.utcnow().isoformat()
        new_entry = DailySummary(
            date=key,
            summary=compressed,
            turn_ids=old.turn_ids,
            created_at=now,
            last_accessed=now,
            version=old.version + 1,
            word_count=len(compressed.split()),
        )
        self._layer2[key] = new_entry
        self._save_layer2()
        return new_entry

    def decay_old_summaries(
        self,
        older_than_days: int = 30,
        target_words: int = 50,
    ) -> List[str]:
        """Compress every Layer 2 summary older than N days. Returns the
        list of dates that were decayed."""
        cutoff = datetime.utcnow() - timedelta(days=older_than_days)
        decayed = []
        for key, entry in list(self._layer2.items()):
            created = datetime.fromisoformat(entry.created_at)
            if created < cutoff and entry.word_count > target_words:
                self.compress_summary(date.fromisoformat(key), target_words)
                decayed.append(key)
        return decayed

    def stale_summaries(self, max_age_days: int = 30) -> List[str]:
        """Dates of Layer 2 summaries not accessed in N days."""
        cutoff = datetime.utcnow() - timedelta(days=max_age_days)
        return [
            day for day, entry in self._layer2.items()
            if datetime.fromisoformat(entry.last_accessed) < cutoff
        ]

    # --- Query ------------------------------------------------------------

    def query(self, question: str) -> str:
        """Layer 2 first; fall back to Layer 1 on INSUFFICIENT DETAIL."""
        if self._layer2:
            layer2_context = "\n\n".join(
                f"[{e.date}] {e.summary}"
                for e in sorted(self._layer2.values(), key=lambda x: x.date)
            )
            prompt = (
                "Using only the following summaries, answer the question. "
                "If the summaries don't contain enough detail, reply with "
                "exactly 'INSUFFICIENT DETAIL'.\n\n"
                f"Summaries:\n{layer2_context}\n\n"
                f"Question: {question}\n\nAnswer:"
            )
            answer = ollama_client.generate(prompt, model=self.model)
            if "INSUFFICIENT DETAIL" not in answer.upper():
                # Bump access timestamps on summaries that were used.
                now = datetime.utcnow().isoformat()
                for e in self._layer2.values():
                    e.last_accessed = now
                self._save_layer2()
                return answer

        all_turns = self.all_turns()
        if not all_turns:
            return "(no memory yet)"
        raw_context = "\n\n".join(f"[#{t.id}] {t.text}" for t in all_turns)
        prompt = (
            "Answer the question using the following raw text.\n\n"
            f"{raw_context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        return ollama_client.generate(prompt, model=self.model)
