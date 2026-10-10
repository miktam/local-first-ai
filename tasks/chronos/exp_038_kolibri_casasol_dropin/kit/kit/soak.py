"""exp_038 soak (DESIGN rev 2 §0b.2): the backend, settle active, on E37's 50-layer / 10-NoPE tiny build, with
resource_limit - 15,000 Metal buffers held.

  --mode call     one call of 2 x (reasoning cap + answer cap + 2) decode steps: twice the longest single call
                  any exp_038 arm can make (the reasoning cap is clamped to 16,384 by R1's rule)
  --mode session  2 x the K4 arm's session: 2 x 177 rows x (gate 4 + answer 512) decode steps as consecutive calls

Usage (mini, exp_037's mini venv): python kit/soak.py --mode call|session --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

KIT = Path(__file__).resolve().parent
sys.path.insert(0, str(KIT))
import mlx_backend as mb  # noqa: E402

for p in (str(mb.E37), str(mb.E37 / "tests"), str(mb.E37 / "tools")):
    sys.path.insert(0, p)

REASONING_CAP_MAX = 16_384
ANSWER_CAP = 512
ROWS_K4 = 177  # v1 60 + v2 30 + DE 36 + twins 36 + probes 10 + R6 5
MARGIN = 15_000


class StubTok:
    unk_token_id = None
    eos_token_ids = {5000}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(f"<{int(i)}>" for i in ids)

    def convert_tokens_to_ids(self, t):
        return {"<think>": 1000, "</think>": 1001}.get(t)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("call", "session"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import mlx.core as mx
    import port_harness
    import test_runner_offset_leak as t2
    import tiny_checkpoint as tc

    ckpt = t2._write_real50(Path(tempfile.mkdtemp(prefix="exp038_soak_")))
    model = port_harness.load_port(ckpt)
    mx.eval(model.parameters())
    nope = [i for i, layer in enumerate(model.layers) if not layer.use_sliding]
    limit = int(mx.device_info()["resource_limit"])
    mx.clear_cache()
    hold = [mx.array(i) for i in range(limit - MARGIN)]
    mx.eval(hold)
    b = mb.MLXGatedBackend.tiny(model, StubTok(), [5000])
    b._render = lambda f, t, m, e: ("user turn", tc.random_ids(1700 if e == "none" else 16, seed=901), "0" * 64)
    res = {"mode": a.mode, "nope_layers": nope, "resource_limit": limit, "held": len(hold), "error": None,
           "settle_active": True, "calls": 0, "decode_steps": 0}
    if a.mode == "call":
        plan = [2 * (REASONING_CAP_MAX + ANSWER_CAP + 2)]
    else:
        plan = [4, ANSWER_CAP] * (2 * ROWS_K4)
    t0 = time.time()
    try:
        for n in plan:
            opts = {"temperature": 0.0, "num_predict": n, "num_ctx": 4096,
                    "repeat_penalty": 1.05 if n != 4 else 1.0, "repeat_last_n": 256}
            r = b.chat("K4", [], opts)
            assert r["exp038"]["completion_tokens"] == n, r["exp038"]
            res["calls"] += 1
            res["decode_steps"] += n
    except Exception as e:
        res["error"] = f"{type(e).__name__}: {e}"
    res["planned_steps"] = sum(plan)
    res["seconds"] = round(time.time() - t0, 1)
    res["verdict"] = "PASS" if res["error"] is None and res["decode_steps"] == res["planned_steps"] else "FAIL"
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res))
    return 0 if res["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
