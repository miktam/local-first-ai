"""Exact fp32 on the GPU: MLX_ENABLE_TF32=0 for every exp_036 process.

mlx 0.31.2 reads MLX_ENABLE_TF32 (default 1) once per process, at the first
GPU matmul. On the M5 GPU, TF32 truncates the fp32 operands of matmul with
M >= 2, sorted gather_mm and multi-query SDPA to 10 mantissa bits (relative L2
error about 7.7e-4 against float64, instead of about 3e-7). Prefill then runs
in TF32 while decode (M = 1) stays exact, and the gate's fp32 checks measure
the GPU instead of the port (aborted/20261004T044821Z-tests/NOTE.md). The
vendor's numbers come from exact products with fp32 accumulation (vLLM on
CUDA, where PyTorch leaves TF32 off for matmul), so exp_036 turns TF32 off
everywhere: tests, gate, peer check, bench and scored runs alike (Amendment 1,
2026-10-04). bf16 and quantised matmuls are not affected by the variable.

Every entry point that runs GPU work calls ensure_exact_fp32() as its first
statement, before any GPU operation. Setting the variable later in a process
has no effect, so the probe checks that the setting took.
"""

from __future__ import annotations

import os

ENV = "MLX_ENABLE_TF32"
# Exact fp32 GEMM: ~3e-7 relative L2 against float64; TF32: ~7.7e-4.
PROBE_REL_L2_MAX = 1e-5


class PrecisionError(RuntimeError):
    """The GPU does not compute fp32 matmuls exactly in this process."""


def ensure_exact_fp32(probe: bool = True) -> dict:
    """Set MLX_ENABLE_TF32=0 unless it is already set, refuse any other value,
    and (probe=True) check on the GPU that an fp32 matmul with M >= 2 is exact.

    Returns {"MLX_ENABLE_TF32": "0", "probe_rel_l2": float} for version records."""
    value = os.environ.setdefault(ENV, "0")
    if value != "0":
        raise PrecisionError(
            f"exp_036 needs exact fp32 on the GPU, but {ENV}={value!r}. "
            f"Unset it, or set {ENV}=0, before starting Python.")
    record: dict = {ENV: value}
    if probe:
        record["probe_rel_l2"] = probe_rel_l2()
        if record["probe_rel_l2"] > PROBE_REL_L2_MAX:
            raise PrecisionError(
                f"GPU fp32 matmul is not exact in this process (relative L2 {record['probe_rel_l2']:.2e} > "
                f"{PROBE_REL_L2_MAX:.0e}): TF32 is active. A GPU matmul ran before {ENV}=0 was set; "
                f"set it in the environment before Python starts.")
    return record


def probe_rel_l2() -> float:
    """Relative L2 error of a 64 x 512 x 256 fp32 GPU matmul against float64."""
    import mlx.core as mx
    import numpy as np

    rng = np.random.default_rng(36)
    a = rng.standard_normal((64, 512)).astype(np.float32)
    b = rng.standard_normal((512, 256)).astype(np.float32)
    ref = a.astype(np.float64) @ b.astype(np.float64)
    out = mx.matmul(mx.array(a), mx.array(b), stream=mx.gpu)
    mx.eval(out)
    got = np.array(out).astype(np.float64)
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))
