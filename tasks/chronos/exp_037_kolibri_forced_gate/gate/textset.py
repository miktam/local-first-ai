# SPDX-License-Identifier: MIT
"""The token sequences the gate consumes, for a real run or a tiny run.

Real (HYPOTHESIS Phase 0 "Gate text"): T1-T4, T7, T8 and the G5 prompts from
gate/texts/ (committed ids); T5, T6, T9 from $EXP036_WORK/gate_texts/ (built
by gate/build_gate_text.py --work-only, checked against MANIFEST.json).

Tiny (`--tiny`, the weight-free tests): the same structure with seeded random
ids below the tiny vocabulary's stop tokens, shorter lengths and positions
scaled to the tiny sliding windows (TinyProfile).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from gate import common
from gate.common import TEXTS_DIR, ids_sha256

PLAIN_IDS = ("T1", "T2", "T3", "T4", "T5", "T6")
ALL8 = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
LANG = {"T1": "en", "T2": "en", "T3": "de", "T4": "de", "T5": "de", "T6": "de", "T7": "en", "T8": "de"}
FILES = {
    "T1": "T1_exp035_post.ids.json", "T2": "T2_malaga_ai_post.ids.json",
    "T3": "T3_de_prose_claude.ids.json", "T4": "T4_grundgesetz_art1_19.ids.json",
    "T7": "T7_chat_en_high.json", "T8": "T8_chat_de_none.json",
}


@dataclass
class Profile:
    """Lengths and positions of one gate run (real values from HYPOTHESIS)."""
    name: str = "real"
    text_len: int = 1536
    t9_len: int = 16384
    t9_buckets: tuple = ((0, 2048), (2048, 8192), (8192, 16384))
    decode_ranges: dict = field(default_factory=lambda: {"T1": (520, 1100), "T3": (520, 1100), "T9": (15000, 15300)})
    g5_greedy_tokens: int = 256
    batch_lengths: tuple = (37, 300, 700, 1100, 64, 520, 900, 150, 37, 700, 300, 1100)
    batch_max_tokens: tuple = (48, 8, 32, 16, 40, 24, 56, 12, 20, 36, 28, 44)
    batch_B: int = 8
    behaviour_cap_high: int = 8192
    behaviour_cap_none: int = 8192
    prefill_chunk: int = 2048
    window_position: int = 513
    real_weight_mutant_layers: tuple = (0, 3, 4, 49)


def real_profile(th: dict) -> Profile:
    """The real run's lengths and positions, from the frozen gate/thresholds.json."""
    g2, g5 = th["G2"], th["G5"]
    (lo1, hi1), (lo9, hi9) = g5["decode_positions"]
    return Profile(
        t9_buckets=tuple(tuple(b) for b in g2["t9_buckets"]),
        decode_ranges={"T1": (lo1, hi1), "T3": (lo1, hi1), "T9": (lo9, hi9)},
        g5_greedy_tokens=g5["greedy_tokens"],
        batch_B=g5["batch_B"],
        behaviour_cap_high=g5["behaviour_effort_high_cap"],
        behaviour_cap_none=g5["behaviour_effort_high_cap"],  # HYPOTHESIS gives one cap; effort none ends early
        real_weight_mutant_layers=tuple(g2["real_weight_mutant_layers"]),
    )


def tiny_profile(window: int, n_layers: int) -> Profile:
    """Scaled to a tiny checkpoint: every text crosses the window at least
    twice, T9 has three buckets, decode ranges start past the window."""
    L = max(160, 2 * window + 30)
    t9 = 4 * L
    return Profile(
        name="tiny",
        text_len=L,
        t9_len=t9,
        t9_buckets=((0, L), (L, 2 * L), (2 * L, t9)),
        decode_ranges={"T1": (window + 7, L - 1), "T3": (window + 7, L - 1), "T9": (t9 - 3 * window, t9 - 1)},
        g5_greedy_tokens=24,
        batch_lengths=(7, 30, 70, 110, 12, 52, 90, 15, 7, 70, 30, 110),
        batch_max_tokens=(12, 3, 9, 5, 11, 7, 14, 4, 6, 10, 8, 13),
        batch_B=8,
        behaviour_cap_high=48,
        behaviour_cap_none=48,
        prefill_chunk=64,
        window_position=window,
        real_weight_mutant_layers=tuple(sorted({0, 3, 4, n_layers - 1} & set(range(n_layers)))),
    )


@dataclass
class TextSet:
    profile: Profile
    ids: dict            # T1..T8 -> list[int] (each profile.text_len long)
    t9: list
    g5: list             # [{"id", "lang", "effort", "ids"}]
    token_bytes: dict    # T1..T6 -> list[int] UTF-8 bytes per token (bpb)
    key: str             # cache key of the reference dumps
    sources: dict

    def lang(self, tid: str) -> str:
        return LANG[tid]

    def packed8(self) -> list:
        return [self.ids[t] for t in ALL8]


def _token_byte_table(tok_dir: Path) -> dict:
    """UTF-8 byte length of every token id of a byte-level BPE vocabulary
    (GPT-2 bytes_to_unicode inverse); added tokens count their content."""
    tok = common.load_raw_tokenizer(tok_dir)
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    inv = {chr(c): b for b, c in zip(bs, cs)}
    table = {}
    for piece, i in tok.get_vocab(with_added_tokens=False).items():
        # Each character of a byte-level piece stands for one byte.
        table[i] = len(piece) if all(ch in inv for ch in piece) else len(piece.encode("utf-8"))
    for i, added in tok.get_added_tokens_decoder().items():
        table[i] = len(added.content.encode("utf-8"))
    return table


def load_real(tok_dir: Path | None = None, work: Path | None = None, profile: Profile | None = None) -> TextSet:
    tok_dir = Path(tok_dir or common.tokenizer_dir())
    work = Path(work or common.work_dir() / "gate_texts")
    manifest = common.read_json(TEXTS_DIR / "MANIFEST.json")
    ids = {t: common.read_json(TEXTS_DIR / f)["ids"] for t, f in FILES.items()}
    web = {}
    for tid in ("T5", "T6", "T9"):
        path = work / f"{tid}.ids.json"
        if not path.is_file():
            raise FileNotFoundError(f"{common.redact_path(path)} missing: run gate/build_gate_text.py --work-only (RUNBOOK step 7)")
        web[tid] = common.read_json(path)["ids"]
        want = manifest.get("web", {}).get(tid, {}).get("ids_sha256")
        if want and want != ids_sha256(web[tid]):
            raise ValueError(f"{tid} in the work dir does not match MANIFEST.json")
    ids["T5"], ids["T6"] = web["T5"], web["T6"]
    g5 = [{k: p[k] for k in ("id", "lang", "effort", "ids")} for p in common.read_json(TEXTS_DIR / "G5_prompts.json")["prompts"]]
    table = _token_byte_table(tok_dir)
    token_bytes = {t: [table[i] for i in ids[t]] for t in PLAIN_IDS}
    key = common.sha256_bytes("|".join([common.gate_text_sha256()] + [ids_sha256(web[t]) for t in ("T5", "T6", "T9")]).encode())
    return TextSet(profile or real_profile(common.load_thresholds()), ids, web["T9"], g5, token_bytes, key,
                   {"manifest_sha256": common.gate_text_sha256(), "work_dir": common.redact_path(work)})


def make_tiny(window: int, n_layers: int, vocab: int, seed: int = 36, reserved_top: int = 16) -> TextSet:
    """Seeded random ids in [0, vocab - reserved_top) (stop tokens excluded)."""
    prof = tiny_profile(window, n_layers)
    rng = np.random.Generator(np.random.PCG64(common.seed_from("tiny-gate-texts", seed, prefix="exp036")))  # registered fixture seed, not an exp_037 draw
    high = vocab - reserved_top

    def draw(n):
        return [int(x) for x in rng.integers(0, high, size=n)]

    ids = {t: draw(prof.text_len) for t in ALL8}
    t9 = draw(prof.t9_len)
    g5 = [{"id": f"G5_tiny_{j}", "lang": "en" if j < 4 else "de", "effort": "none" if j % 2 == 0 else "high",
           "ids": draw(prof.text_len // 2 + 7 * j)} for j in range(8)]
    token_bytes = {t: [4] * prof.text_len for t in PLAIN_IDS}
    key = common.sha256_bytes(common.dumps({"ids": ids, "t9": t9, "g5": g5}).encode())
    return TextSet(prof, ids, t9, g5, token_bytes, key, {"tiny": True, "seed": seed})
