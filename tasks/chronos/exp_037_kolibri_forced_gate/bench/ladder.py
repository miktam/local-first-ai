# SPDX-License-Identifier: MIT
"""E6 — context ladder and prefill cliff (HYPOTHESIS E6, Tier B item B4;
BUILD_SPEC §5.7 ladder.py). TIER 2: frozen by the "Tier-2 analysis"
amendment, not by the pre-registration hash of bench/ (HYPOTHESIS hash table:
"bench/ tree minus ladder.py"). Refuses to run unless that amendment is at
HEAD (runner.guard.require_tier2). Output: results/bench/ladder_<UTC>.jsonl.

K4, K8 and G4, one resident at a time. Rungs:
  4k    the whole of pad_4k.txt
  15k   the whole of pad_15k.txt
  32k   exact-length prefix of pad_120k.txt, 32,768 tokens
  64k   exact-length prefix of pad_120k.txt, 65,536 tokens
  120k  K4 only: the whole of pad_120k.txt. It holds 117,610 Kolibri tokens,
        fewer than 120k, so no exact-length 120k prefix exists; the rung is
        the whole file and its actual N is recorded.
3 reps per rung (2 at 120k), fresh cache each, prefill (step 2048) then a
64-token greedy decode (EOS masked); prompt_tps, prefill ms/token and peak
memory per run; 60 s idle between sizes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import common, genutil  # noqa: E402

SPEC = "HYPOTHESIS E6 (Tier 2); BUILD_SPEC §5.7 ladder.py"
LADDER_ARMS = ("K4", "K8", "G4")
# (label, fixture, exact token count or None for the whole file, arms or None for all)
RUNGS = (
    ("4k", "pad_4k.txt", None, None),
    ("15k", "pad_15k.txt", None, None),
    ("32k", "pad_120k.txt", 32768, None),
    ("64k", "pad_120k.txt", 65536, None),
    ("120k", "pad_120k.txt", None, ("K4",)),
)
REPS = {"120k": 2}
DEFAULT_REPS = 3
GEN_TOKENS = 64
IDLE_BETWEEN_SIZES_S = 60


def required_fixtures(rungs=RUNGS) -> list[str]:
    """Fixture names the ladder needs; preflight asserts each exists by name."""
    return sorted({r[1] for r in rungs})


def assert_fixtures(rungs=RUNGS) -> dict:
    return {name: common.sha256_file(common.fixture_path(name)) for name in required_fixtures(rungs)}


def run_cell(
    results_dir: Optional[Path] = None,
    *,
    loader: Callable[[str], tuple] = genutil.load_arm,
    arms=LADDER_ARMS,
    rungs=RUNGS,
    gen_tokens: int = GEN_TOKENS,
    idle_s: float = IDLE_BETWEEN_SIZES_S,
    reps: Optional[dict] = None,
) -> Path:
    import mlx.core as mx

    results_dir = Path(results_dir or common.default_results_dir())
    common.require_identity()
    common.require_tier2()
    fixtures = assert_fixtures(rungs)
    common.require_gates(arms)
    reps = {**REPS, **(reps or {})}

    header = {
        **common.base_header("ladder", SPEC, results_dir),
        "fixtures_sha256": fixtures,
        "protocol": {
            "arms": list(arms),
            "rungs": [{"label": r[0], "fixture": r[1], "tokens": r[2], "arms": list(r[3]) if r[3] else None} for r in rungs],
            "reps": {r[0]: reps.get(r[0], DEFAULT_REPS) for r in rungs},
            "generated_tokens": gen_tokens,
            "prefill_step_size": genutil.PREFILL_STEP,
            "idle_s_between_sizes": idle_s,
        },
    }
    path = common.new_output_path(results_dir, "ladder")
    with common.JsonlCell(path, header) as w:
        for arm in arms:
            genutil.release()
            model, tok, d = loader(arm)
            eos = genutil.eos_ids(d, tok)
            w.append(
                {
                    "kind": "arm",
                    "arm": arm,
                    "model_dir": common.redact_path(d),
                    "fingerprint": common.model_fingerprint(arm, d),
                    "eos_ids": eos,
                    "memory_after_load": genutil.memory(),
                }
            )
            for label, fixture, n, only in rungs:
                if only and arm not in only:
                    continue
                ids, info = genutil.fixture_ids(tok, fixture, n)
                for rep in range(reps.get(label, DEFAULT_REPS)):
                    mx.clear_cache()
                    mx.reset_peak_memory()
                    before = genutil.memory()
                    snap = common.snapshot()
                    r = genutil.timed_generate(model, tok, ids, gen_tokens, eos)
                    peak = int(mx.get_peak_memory())
                    w.append(
                        {
                            "kind": "run",
                            "arm": arm,
                            "rung": label,
                            "rep": rep,
                            "prompt": info,
                            "n_context": len(ids),
                            "prefill_ms_per_token": 1000.0 / r["prompt_tps"],
                            "peak_bytes": peak,
                            "memory_before": before,
                            "system": snap,
                            **r,
                        }
                    )
                w.append({"kind": "idle", "arm": arm, "after_rung": label, **common.idle(idle_s)})
            del model, tok
            genutil.release()
        w.finish({"note": "descriptive (E6); analysed by analysis/exploratory.py"})
    return path
