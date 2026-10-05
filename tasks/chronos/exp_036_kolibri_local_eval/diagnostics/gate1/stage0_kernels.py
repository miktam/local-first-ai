# SPDX-License-Identifier: MIT
"""Stage 0 (DIAGNOSIS §4): kernel probe at Kolibri's real decode shapes, B = 8 rows
against B = 1 rows, against an exact float64 reference, on this GPU. No model
weights (random weights at the real shapes); about 1 minute.

Runs diagnostics/gate1/kprobe2.py unchanged (sha256 88c1ce6b..., checked here)
with the M4 Pro baseline kprobe2_m4pro.json, and records its JSON. The output's
baseline.verdict is the rule input kprobe_verdict (FROZEN_RULES.md §2, §3 G5-D sign 5).
bf16 probes run with the production dtypes; MLX_ENABLE_TF32=0 (Amendment 1)
only affects the fp32 probes (router GEMM, fp32-input head), as in production.

usage (kit root, env sourced): caffeinate -i "$PY" diagnostics/gate1/stage0_kernels.py
"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parents[2]), str(Path(__file__).resolve().parent)]
from tools.precision import ensure_exact_fp32  # noqa: E402

PREC = ensure_exact_fp32()

import contextlib  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import runpy  # noqa: E402

import diaglib as D  # noqa: E402

KPROBE_SHA256 = "88c1ce6bb2fa1b819c5790e5476f9a1cab37cd4b8410fa1ad0206d51c8c61d43"
BASELINE_SHA256 = "86bf6d7af34f81c019bc99a500a5278aad3da2cc86ed8afe4772eca57aee7a58"


def main() -> int:
    ap = D.parser("stage0_kernels", __doc__)
    args = ap.parse_args()
    run = D.Run("stage0_kernels", args, PREC)
    kp, base = D.HERE / "kprobe2.py", D.HERE / "kprobe2_m4pro.json"
    sha = {"kprobe2.py": D.sha256_file(kp), "kprobe2_m4pro.json": D.sha256_file(base)}
    if sha["kprobe2.py"] != KPROBE_SHA256 or sha["kprobe2_m4pro.json"] != BASELINE_SHA256:
        raise SystemExit(f"stage0: kprobe2 files differ from the frozen copies: {sha}")
    buf = io.StringIO()
    argv = sys.argv
    sys.argv = [str(kp), str(base)]
    try:
        with contextlib.redirect_stdout(buf):
            runpy.run_path(str(kp), run_name="__main__")
    finally:
        sys.argv = argv
    out = json.loads(buf.getvalue())
    out["baseline"]["file"] = "diagnostics/gate1/kprobe2_m4pro.json"
    verdict = out["baseline"]["verdict"]
    record = {
        "rule_inputs": {"kprobe_verdict": verdict, "kprobe_defects": out["baseline"]["chip_specific_defects"]},
        "kprobe2": out, "files_sha256": sha,
    }
    D.log(f"[stage0] {out['device']}: {verdict}; flagged {out['flagged']}")
    run.finish(record, {})
    return 0


if __name__ == "__main__":
    sys.exit(main())
