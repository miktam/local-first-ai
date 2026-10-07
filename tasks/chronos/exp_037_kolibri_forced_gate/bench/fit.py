# SPDX-License-Identifier: MIT
"""D1 — does Kolibri 4-bit fit a 64 GB node at 64k context (HYPOTHESIS D1;
BUILD_SPEC §5.7 fit.py). Output: results/bench/fit_<UTC>.jsonl.

K4 is the only resident model. Per run:
  mx.clear_cache(); mx.reset_peak_memory();
  prefill exactly N Kolibri tokens of an exact-length prefix of pad_120k.txt
  (real name) with prefill_step_size 2048, then 512 greedy tokens (EOS masked,
  so exactly 512), then mx.get_peak_memory(); prefill tok/s is recorded too.
N in {32,768; 65,536}, 3 reps each, in that order.

The rule (median peak at 64k <= 46.66 GiB CONFIRMED, > 51.84 GiB REFUTED) is
analysis/verdicts.py's, read from analysis/margins.json; nothing is decided here.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import common, genutil  # noqa: E402

SPEC = "HYPOTHESIS D1; BUILD_SPEC §5.7 fit.py"
D1_ARM = "K4"
D1_SIZES = (32768, 65536)
D1_REPS = 3
D1_GEN_TOKENS = 512
D1_FIXTURE = "pad_120k.txt"
GIB = float(2**30)


def run_cell(
    results_dir: Optional[Path] = None,
    *,
    loader: Callable[[str], tuple] = genutil.load_arm,
    sizes=D1_SIZES,
    reps: int = D1_REPS,
    gen_tokens: int = D1_GEN_TOKENS,
) -> Path:
    import mlx.core as mx

    results_dir = Path(results_dir or common.default_results_dir())
    common.require_identity()
    common.require_gates([D1_ARM])

    genutil.release()
    model, tok, d = loader(D1_ARM)
    eos = genutil.eos_ids(d, tok)
    ids, info = genutil.fixture_ids(tok, D1_FIXTURE, max(sizes))
    header = {
        **common.base_header("fit", SPEC, results_dir),
        "arm": D1_ARM,
        "model_dir": common.redact_path(d),
        "fingerprint": common.model_fingerprint(D1_ARM, d),
        "weights_bytes": common.weights_bytes(d),
        "eos_ids": eos,
        "prompt_source": info,
        "memory_after_load": genutil.memory(),
        "protocol": {
            "sizes": list(sizes),
            "reps": reps,
            "generated_tokens": gen_tokens,
            "prefill_step_size": genutil.PREFILL_STEP,
            "per_run": "mx.clear_cache, mx.reset_peak_memory, stream_generate (fresh cache, greedy, EOS masked), mx.get_peak_memory",
            "peak_unit": "bytes; peak_gib = bytes / 2**30",
        },
    }
    path = common.new_output_path(results_dir, "fit")
    peaks: dict[str, list[float]] = {str(n): [] for n in sizes}
    with common.JsonlCell(path, header) as w:
        for n in sizes:
            prompt = ids[:n]
            for rep in range(reps):
                mx.clear_cache()
                mx.reset_peak_memory()
                before = genutil.memory()
                snap = common.snapshot()
                r = genutil.timed_generate(model, tok, prompt, gen_tokens, eos)
                peak = int(mx.get_peak_memory())
                peaks[str(n)].append(peak / GIB)
                w.append(
                    {
                        "kind": "run",
                        "n_context": n,
                        "rep": rep,
                        "prompt_ids_sha256": common.sha256_ids(prompt),
                        "peak_bytes": peak,
                        "peak_gib": peak / GIB,
                        "memory_before": before,
                        "memory_after": genutil.memory(),
                        "system": snap,
                        **r,
                    }
                )
        w.finish(
            {
                "median_peak_gib": {k: float(np.median(v)) for k, v in peaks.items()},
                "note": "descriptive; the D1 verdict is computed by analysis/verdicts.py",
            }
        )
    del model, tok
    genutil.release()
    return path
