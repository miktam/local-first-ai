"""Cell seeds (HYPOTHESIS "Fixed before any run", Sampling; BUILD_SPEC §5.4).

The seed of a cell is H("exp037", arm, task, effort, pass): the first 4 bytes,
big-endian, of sha256 over "exp037|<arm>|<task>|<effort>|<pass>". The runner
calls mx.random.seed(cell_seed) once per cell (and again when a cell resumes);
the RNG is global, so exact replay of a batched run is not claimed (item 27).

exp_037 draws its own cell seeds (DESIGN §2.10): the rule is exp_036's, with
the prefix "exp037" instead of "exp036".
"""

from __future__ import annotations

import hashlib

SEED_PREFIX = "exp037"


def seed_string(arm: str, task: str, effort: str, pass_: int = 0) -> str:
    for name, v in (("arm", arm), ("task", task), ("effort", effort)):
        if not v or "|" in v:
            raise ValueError(f"{name} must be a non-empty string without '|', got {v!r}")
    if int(pass_) < 0:
        raise ValueError("pass must be >= 0")
    return f"{SEED_PREFIX}|{arm}|{task}|{effort}|{int(pass_)}"


def cell_seed(arm: str, task: str, effort: str, pass_: int = 0) -> int:
    """Unsigned 32-bit seed of one cell."""
    digest = hashlib.sha256(seed_string(arm, task, effort, pass_).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")
