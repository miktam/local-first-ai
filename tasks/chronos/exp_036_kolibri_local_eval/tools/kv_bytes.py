"""KV-cache bytes per token from a model config (BUILD_SPEC §5.9 `tools/kv_bytes.py`;
HYPOTHESIS "Pilot and the rule that fixes n", rule 1).

`kv_bytes_per_token(config) -> (growing, fixed_window)`:
- `growing`: bytes added per token per sequence by the layers whose cache grows
  without bound (full attention: K and V, bf16).
- `fixed_window`: bytes per sequence that do not grow with length: the
  sliding-window layers at their full window (K and V, bf16), and for Qwen
  DeltaNet layers the conv state (bf16) plus the recurrent state (fp32, as
  mlx_lm 0.31.3 `gated_delta.py` allocates it).

The cache layout mirrors mlx_lm 0.31.3 `make_cache()` of each family and
port/kolibri1.py (spec items 10, 20, 24): Gemma 4 global layers use
`num_global_key_value_heads` × `global_head_dim` and store K and V even with
`attention_k_eq_v`; KV-shared layers have no cache of their own.

PREDICTIONS are the HYPOTHESIS / BUILD_SPEC values; `check(config)` compares
and records a mismatch without failing (BUILD_SPEC: "A mismatch is recorded,
not fatal").
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

BF16 = 2
FP32 = 4

# family -> (growing B/token, fixed bytes per sequence or None if not predicted)
PREDICTIONS = {
    "kolibri1": (20480, 42_024_960),   # 10 full layers; 40 sliding × 513 (≈ 42 MB)
    "gemma4": (20480, 209_715_200),    # 5 full layers; 25 sliding × 1024 (≈ 210 MB)
    "qwen3_5_moe": (20480, None),      # Qwen3.6: 10 full layers; DeltaNet state
    "qwen3_5": (65536, None),          # Qwen3.8: 16 full layers; DeltaNet state
}


def _text_config(config: dict) -> dict:
    tc = config.get("text_config")
    if isinstance(tc, dict) and tc:
        merged = dict(tc)
        merged.setdefault("model_type", config.get("model_type"))
        return merged
    return config


def family(config: dict) -> str:
    mt = config.get("model_type") or ""
    if mt.startswith("gemma4"):
        return "gemma4"
    return mt


def _layer_types(tc: dict) -> list[str]:
    n = int(tc["num_hidden_layers"])
    lt = tc.get("layer_types")
    if lt:
        return list(lt)[:n]
    if "full_attention_interval" in tc:
        k = int(tc["full_attention_interval"])
        return ["full_attention" if (i + 1) % k == 0 else "linear_attention" for i in range(n)]
    return ["full_attention"] * n


def kv_bytes_per_token(config: dict) -> tuple[int, int]:
    fam = family(config)
    tc = _text_config(config)
    types = _layer_types(tc)
    growing = 0
    fixed = 0
    if fam == "gemma4":
        n_cached = len(types) - int(tc.get("num_kv_shared_layers") or 0)
        for t in types[:n_cached]:
            if t == "full_attention":
                k_eq_v = bool(tc.get("attention_k_eq_v"))
                heads = tc.get("num_global_key_value_heads") if k_eq_v else None
                heads = int(heads or tc["num_key_value_heads"])
                hd = int(tc.get("global_head_dim") or tc["head_dim"])
                growing += 2 * heads * hd * BF16
            else:
                fixed += int(tc["sliding_window"]) * 2 * int(tc["num_key_value_heads"]) * int(tc["head_dim"]) * BF16
        return growing, fixed
    heads = int(tc.get("num_key_value_heads") or tc["num_attention_heads"])
    hd = int(tc.get("head_dim") or tc["hidden_size"] // tc["num_attention_heads"])
    for t in types:
        if t == "full_attention":
            growing += 2 * heads * hd * BF16
        elif t == "sliding_attention":
            fixed += int(tc["sliding_window"]) * 2 * heads * hd * BF16
        elif t == "linear_attention":
            nk, dk = int(tc["linear_num_key_heads"]), int(tc["linear_key_head_dim"])
            nv, dv = int(tc["linear_num_value_heads"]), int(tc["linear_value_head_dim"])
            conv_dim = 2 * nk * dk + nv * dv
            conv = (int(tc["linear_conv_kernel_dim"]) - 1) * conv_dim * BF16
            state = nv * dv * dk * FP32
            fixed += conv + state
        else:
            raise ValueError(f"unknown layer type {t!r} in {fam} config")
    return growing, fixed


def check(config: dict) -> dict:
    fam = family(config)
    growing, fixed = kv_bytes_per_token(config)
    pred = PREDICTIONS.get(fam)
    out = {"family": fam, "growing_bytes_per_token": growing, "fixed_window_bytes": fixed}
    if pred:
        out["predicted_growing"] = pred[0]
        out["predicted_fixed"] = pred[1]
        mism = []
        if growing != pred[0]:
            mism.append(f"growing {growing} != predicted {pred[0]}")
        if pred[1] is not None and fixed != pred[1]:
            mism.append(f"fixed {fixed} != predicted {pred[1]}")
        out["mismatch"] = mism
    else:
        out["mismatch"] = [f"no prediction for family {fam!r}"]
    return out


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model_dirs", nargs="+", help="directories holding config.json")
    args = ap.parse_args(argv)
    for d in args.model_dirs:
        cfg = json.loads((Path(d) / "config.json").read_text())
        print(json.dumps({"dir": Path(d).name, **check(cfg)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
