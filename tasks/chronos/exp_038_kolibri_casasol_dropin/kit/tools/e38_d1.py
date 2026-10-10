"""E38-D1 (DESIGN rev 2 R2, (a)): does Kolibri 4-bit fit a 64 GB node? Measured on the mini (M4 Pro, 64 GB).

Runs exp_037's own D1 bench (E37 bench/fit.py run_cell, unchanged): K4 as the only resident model; per run
mx.clear_cache, mx.reset_peak_memory, prefill exactly N tokens of pad_120k.txt (prefill_step_size 2048), then 512
greedy tokens with EOS masked, then mx.get_peak_memory; N in {32,768; 65,536}, 3 reps each. Memory only: no
generated text is kept, and no K4 quality row runs on the mini (the gate certified the M5 Max only).

Rule (exp_037 D1's, registered for E38-D1): CONFIRMED if the median peak at 64k <= 46.66 GiB (0.9 x the mini's
51.84 GiB Metal working-set limit); REFUTED if > 51.84 GiB; INCONCLUSIVE otherwise. The mini's own limit is read
and recorded; if it is not 51.84 GiB the run stops (the thresholds were derived from it).

Preconditions checked here: Ollama is not running (it holds gemma4:26b resident); the K4 copy's weights bytes and
convert manifest equal exp_037's K4 (the fingerprint in E37's fit record); the port file passes check_port_file.

    ~/models/exp037-mini/venv312/bin/python tools/e38_d1.py --model-dir ~/models/exp038/Kolibri-1-MLX-4bit-g64
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
E37 = Path.home() / "REPOS/local-first-ai/tasks/chronos/exp_037_kolibri_forced_gate"
GIB = float(2**30)
LIMIT_GIB, CONFIRM_GIB = 51.84, 46.66
K4_WEIGHTS_BYTES = 44_102_991_060  # exp_037 results/kl_8v4_20261008T075810Z.json, fingerprints.K4.weights_bytes


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", type=Path, default=Path.home() / "models/exp038/Kolibri-1-MLX-4bit-g64")
    a = ap.parse_args(argv)
    if subprocess.run(["pgrep", "-x", "ollama"], capture_output=True).returncode == 0 or \
       subprocess.run(["pgrep", "-f", "ollama serve"], capture_output=True).returncode == 0:
        sys.exit("REFUSED: Ollama is running; stop it first (it keeps gemma4:26b resident)")
    if not a.model_dir.is_dir():
        sys.exit(f"REFUSED: {a.model_dir} missing (copy the K4 build from the mbp first)")
    for p in (str(E37), str(E37 / "tools")):
        sys.path.insert(0, p)
    import mlx.core as mx
    from bench import common, fit, genutil
    from port import convert

    limit = mx.device_info().get("max_recommended_working_set_size")
    limit_gib = round(limit / GIB, 2) if limit else None
    if limit_gib != LIMIT_GIB:
        sys.exit(f"REFUSED: this host's Metal working-set limit is {limit_gib} GiB, the D1 thresholds assume {LIMIT_GIB}")
    port_sha = convert.check_port_file(a.model_dir)
    wb = common.weights_bytes(a.model_dir)
    if wb != K4_WEIGHTS_BYTES:
        sys.exit(f"REFUSED: K4 copy has {wb} weight bytes, exp_037's K4 has {K4_WEIGHTS_BYTES}")
    e37_fit = sorted((E37 / "results/bench").glob("fit_*.jsonl"))
    ref_fp = json.loads(e37_fit[-1].read_text(encoding="utf-8").splitlines()[0]).get("fingerprint") if e37_fit else None
    fp = common.model_fingerprint("K4", a.model_dir)
    if ref_fp is not None and fp != ref_fp:
        sys.exit(f"REFUSED: K4 copy fingerprint differs from exp_037's fit record ({e37_fit[-1].name})")

    out_dir = P38 / "runs" / "e38_d1"
    path = fit.run_cell(out_dir, loader=lambda arm: genutil.load_arm(arm, model_dir=a.model_dir))
    rows = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines()]
    peaks = {}
    for r in rows:
        if r.get("kind") == "run":
            peaks.setdefault(int(r["n_context"]), []).append(float(r["peak_gib"]))
    med64 = statistics.median(peaks[65536])
    state = "CONFIRMED" if med64 <= CONFIRM_GIB else "REFUTED" if med64 > LIMIT_GIB else "INCONCLUSIVE"
    res = {"check": "E38-D1", "host_limit_gib": limit_gib, "port_sha256": port_sha, "weights_bytes": wb,
           "fingerprint_matches_exp037": ref_fp is None or fp == ref_fp, "record": Path(path).name,
           "peaks_gib": peaks, "median_peak_64k_gib": med64, "median_peak_32k_gib": statistics.median(peaks[32768]),
           "rule": f"CONFIRMED <= {CONFIRM_GIB}; REFUTED > {LIMIT_GIB}; INCONCLUSIVE otherwise", "state": state}
    (out_dir / "e38_d1_result.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
