"""
vocab_store.py — The Crown Jewels.

Holds the mapping from real names to pseudonyms. This is the only file in
the project that contains reversible identity. The trust contract:

  - This module MUST NEVER be imported by any module that calls the LLM.
  - The persisted vault file MUST NEVER be sent to any inference endpoint.
  - Anonymized text is safe to leave around; the vault is what you encrypt.

If you find yourself wanting to import this module from `memory.py` or
anywhere downstream of the LLM boundary, stop. The whole architecture
lives in the fact that you can't.
"""

import hashlib
import json
import random
from pathlib import Path
from typing import Dict, List, Optional


# --- Pseudonym pool ---------------------------------------------------------
# Bland names. Avoid culturally loaded names, names from famous works, or
# anything that re-introduces association leaks. If you anonymize a tech
# memoir, don't pick "Linus" as a pseudonym. Adjust the pool to your domain.

FIRST_NAMES = [
    "Marcus", "Daniel", "Owen", "Iain", "Theo", "Felix", "Adrian", "Joel",
    "Hugo", "Reed", "Wren", "Otis", "Caleb", "Mason", "Errol", "Quinn",
    "Sarah", "Naomi", "Iris", "Helen", "Clare", "Mara", "Linnea", "Esme",
    "Tess", "Vera", "Maeve", "Petra", "Elena", "Greta", "Cora", "Nadia",
]

SURNAMES = [
    "Walsh", "Patel", "Holloway", "Bramwell", "Reeves", "Castro", "Wexford",
    "Halverson", "Trent", "Okafor", "Marsh", "Pemberton", "Ashby",
    "Fenwick", "Crane", "Mossman", "Yardley", "Tindall", "Harlow", "Vance",
]

ORG_LEFTS = [
    "North", "Linden", "Bridgehead", "Vale", "Ashford", "Pemberton",
    "Beacon", "Cascade", "Foundry", "Marlow", "Underhill", "Westgate",
]
ORG_RIGHTS = [
    "Collective", "Workshop", "Cooperative", "Initiative", "Society",
    "Trust", "Branch", "Forum", "Circle", "Assembly", "Group", "Guild",
]

PLACE_LEFTS = [
    "Maple", "Linden", "Ashgrove", "Brookline", "Highmoor", "Foxholt",
    "Riverbend", "Coppice", "Stillwater", "Grantham",
]
PLACE_RIGHTS = ["Lane", "Street", "Crescent", "Square", "Row", "Avenue"]


class VocabStore:
    """
    The vault. Maps real entities to stable pseudonyms.

    Why deterministic mapping (not random per run):
      - Reproducibility: same input always produces the same anonymized
        output. Matters for benchmarks and regression tests.
      - Consistency within a corpus: "Tyler" appearing 200 times gets ONE
        pseudonym, not 200 different ones.

    Why salted hash (not naive hash):
      - Without salt, anyone holding a pseudonym could brute-force the
        original by hashing common name lists. The salt lives in the vault
        only and never leaves it.
    """

    KINDS = {"person", "org", "place"}

    def __init__(self, salt: Optional[str] = None):
        # Salt is generated on first use, persisted with the vault.
        # When loading an existing vault, the salt comes from disk.
        self._salt = salt or self._generate_salt()
        self._forward: Dict[str, str] = {}   # real -> pseudonym
        self._reverse: Dict[str, str] = {}   # pseudonym -> real
        self._kinds: Dict[str, str] = {}     # real -> kind

    @staticmethod
    def _generate_salt() -> str:
        # 128 bits of entropy. Good enough to defeat dictionary attacks
        # against the pseudonym pool. Not a cryptographic guarantee.
        return hashlib.sha256(str(random.getrandbits(256)).encode()).hexdigest()[:32]

    def _index(self, real: str, kind: str, pool_size: int, suffix: str = "") -> int:
        # Hash(real | salt | kind | suffix) modulo pool size.
        # Including `kind` means a person and an org with the same string
        # name get different pseudonyms (rare, but safer).
        h = hashlib.sha256(
            f"{self._salt}|{kind}|{real}|{suffix}".encode()
        ).hexdigest()
        return int(h, 16) % pool_size

    def register(self, real: str, kind: str) -> str:
        """Return the stable pseudonym for `real`. Generates one if new."""
        if kind not in self.KINDS:
            raise ValueError(f"unknown kind: {kind}; expected {self.KINDS}")

        if real in self._forward:
            return self._forward[real]

        # Deterministic candidate. Resolve any collision (rare — happens
        # only if two distinct reals hash to the same pseudonym slot) by
        # adding a disambiguator and re-hashing.
        candidate = self._generate(real, kind)
        i = 0
        while candidate in self._reverse and i < 1000:
            candidate = self._generate(real, kind, suffix=f"#{i}")
            i += 1
        if candidate in self._reverse:
            raise RuntimeError("pseudonym pool exhausted; expand it")

        self._forward[real] = candidate
        self._reverse[candidate] = real
        self._kinds[real] = kind
        return candidate

    def _generate(self, real: str, kind: str, suffix: str = "") -> str:
        if kind == "person":
            first = FIRST_NAMES[self._index(real, kind, len(FIRST_NAMES), suffix + ":first")]
            last  = SURNAMES[self._index(real, kind, len(SURNAMES), suffix + ":last")]
            return f"{first} {last}"
        if kind == "org":
            left  = ORG_LEFTS[self._index(real, kind, len(ORG_LEFTS), suffix + ":l")]
            right = ORG_RIGHTS[self._index(real, kind, len(ORG_RIGHTS), suffix + ":r")]
            return f"the {left} {right}"
        if kind == "place":
            left  = PLACE_LEFTS[self._index(real, kind, len(PLACE_LEFTS), suffix + ":l")]
            right = PLACE_RIGHTS[self._index(real, kind, len(PLACE_RIGHTS), suffix + ":r")]
            return f"{left} {right}"
        raise ValueError(kind)

    def reverse_lookup(self, pseudonym: str) -> Optional[str]:
        """Map pseudonym -> real. Use only at display time, never at inference."""
        return self._reverse.get(pseudonym)

    def items(self) -> List[tuple]:
        """For audit/debug only. Treat the result as sensitive."""
        return [(r, p, self._kinds[r]) for r, p in self._forward.items()]

    # --- Persistence -------------------------------------------------------

    def save(self, path: Path) -> None:
        """
        Write the vault to disk. In production: encrypt at rest, restrictive
        mode, and store on a partition that is NEVER mounted into any
        process that calls the LLM.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump({
                "salt": self._salt,
                "forward": self._forward,
                "kinds": self._kinds,
            }, f, indent=2, ensure_ascii=False)
        try:
            path.chmod(0o600)  # owner-only
        except OSError:
            pass  # non-POSIX

    @classmethod
    def load(cls, path: Path) -> "VocabStore":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        v = cls(salt=data["salt"])
        v._forward = dict(data["forward"])
        v._reverse = {p: r for r, p in v._forward.items()}
        v._kinds = dict(data["kinds"])
        return v
