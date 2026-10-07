# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06). Lifted in part from exp_036's
# diagnostics/gate1/confirm/mlx_repro.py (make_weights, reference, call, rel,
# flatten_sorted, route_uniform) and mlx_repro_min.py (case, its 5 % bad-row
# statistic and its boundary sweep).
"""G0k: the kernel probe at the run's own shapes (DESIGN §3.1 P3; decision F2).

MLX 0.31.2's sorted `gather_qmm` overflowed above 32,768 rows on the M5's NAX
path (exp_036 BUGHUNT §6.4-§6.9). MLX 0.32.3 carries the fix (mlx#3922), and
exp_037 keeps no row limit. G0k is the insurance: at gate start it calls the
single op that SwitchGLU's expert projections call, `mx.gather_qmm` with
sorted_indices=True, at every row count the run issues, and compares it with
the same call with sorted_indices=False on identical inputs, each against a
float64 reference.

Weights: Kolibri's expert projections, E = 384 experts, affine group 64, at 8
and 4 bits, seeded per expert (gate/up exactly as mlx_repro.py's make_weights),
bf16-rounded before quantising; activations bf16. Seed 37.

Row counts: the judged set of §3.1 P3 (P3_TABLE below). At gate time they come
from gate/thresholds.json "P3_G0k" through config_from_thresholds(), whose row
groups carry the routing; probe(**config_from_thresholds(block)). Each count is
probed with the routing of the call it stands for:
  tokens  Kolibri calls: T = rows / 6 tokens, each routed to 6 distinct experts
          (route_uniform), rows per (token, expert) stably sorted by expert
          (flatten_sorted, as SwitchGLU builds them). gate/up sees each token's
          input once per expert; down sees one input per row;
  rows    peers' counts and the boundary sweep: one uniformly drawn expert per
          row, sorted (mlx_repro_min.py's case);
  hot     the 2,048 rows of one prefill chunk on one expert.
The routing of a shape is the same for both projections and both bit widths;
the inputs are the same for both bit widths.

Per (projection, bits, shape) the probe measures (no verdict here; the rule is
gate/rules.p3_g0k, applied by gate/preconditions.p3):
  rel_f64_sorted     ||sorted - ref|| / ||ref||, ref the float64 product of the
                     same bf16 inputs with the float32-dequantised weights;
  rel_f64_unsorted   the same for the sorted_indices=False call;
  bad_rows           rows where ||sorted - unsorted|| > 0.05 ||unsorted||
                     (mlx_repro_min.py:l.40);
  repeat_bitwise     a second sorted call is bitwise equal to the first.

Nothing is downloaded and no model is loaded. mlx is imported lazily, so
measure() and the shape tables work without it.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SEED = 37
E = 384             # experts
GROUP_SIZE = 64     # affine quantisation group
K = 6               # Kolibri top-k
HOT_EXPERT = 3      # mlx_repro.py's hot expert (test E)
BITS = (8, 4)
# name -> (n_out, d_in): mx.gather_qmm(..., transpose=True) multiplies x [rows, d_in]
# with W[e] [n_out, d_in]^T, as SwitchGLU's gate_proj/up_proj and down_proj do.
PROJECTIONS = {"gate_up": (512, 2560), "down": (2560, 512)}
BAD_ROW_FRACTION = 0.05   # mlx_repro_min.py:l.40
ROUTINGS = ("tokens", "rows", "hot")
_PROJ_CODE = {"gate_up": 0, "down": 1}
_ROUTING_CODE = {"tokens": 0, "rows": 1, "hot": 2}
_CHUNK_ELEMS = 1 << 24    # input generation chunk (64 MB of float32)

# mlx_repro_min.py's sweep around 2^15 rows (its first config, kolibri_gate_proj).
BOUNDARY_SWEEP = (16384, 24576, 30000, 32000, 32704, 32760, 32767, 32768, 32769, 32770,
                  32776, 32832, 33000, 34000, 40000, 49152, 65536)

# DESIGN §3.1 P3 "Judged row counts", one entry per (rows, routing, origin).
P3_TABLE: tuple[tuple[int, str, str], ...] = (
    (6, "tokens", "decode, Kolibri B=1"),
    (12, "tokens", "decode, Kolibri B=2"),
    (24, "tokens", "decode, Kolibri B=4"),
    (48, "tokens", "decode, Kolibri B=8"),
    (96, "tokens", "decode, Kolibri B=16"),
    (8, "rows", "decode, peers B=1 (top-8)"),
    (16, "rows", "decode, peers B=2 (top-8)"),
    (32, "rows", "decode, peers B=4 (top-8)"),
    (64, "rows", "decode, peers B=8 (top-8)"),
    (128, "rows", "decode, peers B=16 (top-8)"),
    (12288, "tokens", "one prefill chunk, Kolibri (2,048 tokens, uniform routing)"),
    (2048, "hot", "one prefill chunk, a hot expert (2,048 rows on one expert)"),
    (16384, "rows", "one prefill chunk, peers (2,048 tokens x 8)"),
    (24576, "tokens", "batched prefill, Kolibri 2 x 2,048 x 6"),
    (49152, "tokens", "batched prefill, Kolibri 4 x 2,048 x 6"),
    (98304, "tokens", "batched prefill, Kolibri 8 x 2,048 x 6 (the largest Kolibri call)"),
    (52752, "tokens", "exp_036's first wave, 8 x 1,099 x 6 (mlx_repro.py test A count)"),
    (32768, "rows", "batched prefill, peers 2 x 2,048 x 8"),
    (65536, "rows", "batched prefill, peers 4 x 2,048 x 8"),
    (131072, "rows", "batched prefill, peers 8 x 2,048 x 8"),
    (73728, "tokens", "G2's registered T1-8 call, 8 x 1,536 x 6"),
    (196608, "tokens", "the B = 16 bound, 16 x 2,048 x 6 (headroom; never issued)"),
) + tuple((n, "rows", "mlx_repro_min.py boundary sweep") for n in BOUNDARY_SWEEP)


@dataclass(frozen=True)
class Shape:
    rows: int
    routing: str
    sources: tuple = field(default=(), compare=False)

    @property
    def key(self) -> str:
        return f"{self.routing}:{self.rows}"

    def as_dict(self) -> dict:
        return {"rows": self.rows, "routing": self.routing, "sources": list(self.sources)}


def merge_shapes(entries) -> tuple[Shape, ...]:
    """(rows, routing, source) entries -> Shapes, one per (rows, routing), sources
    merged, sorted by (rows, routing). Refuses an unknown routing, a non-positive
    count, or a tokens count that is not a multiple of K."""
    merged: dict[tuple[int, str], list[str]] = {}
    for rows, routing, source in entries:
        rows = int(rows)
        if routing not in ROUTINGS:
            raise ValueError(f"G0k: unknown routing {routing!r}")
        if rows <= 0:
            raise ValueError(f"G0k: row count {rows} is not positive")
        if routing == "tokens" and rows % K:
            raise ValueError(f"G0k: a tokens-routed count must be a multiple of {K}, got {rows}")
        srcs = merged.setdefault((rows, routing), [])
        if source and source not in srcs:
            srcs.append(source)
    return tuple(Shape(r, g, tuple(s)) for (r, g), s in sorted(merged.items(), key=lambda kv: (kv[0][0], kv[0][1])))


JUDGED: tuple[Shape, ...] = merge_shapes(P3_TABLE)


def p3_row_counts() -> tuple[int, ...]:
    """The sorted distinct row counts of §3.1 P3 (what thresholds.json P3_G0k holds)."""
    return tuple(sorted({s.rows for s in JUDGED}))


def judged_shapes(counts) -> tuple[Shape, ...]:
    """Shapes for a list of judged row counts (thresholds.json "P3_G0k"): each
    count gets every routing P3_TABLE gives it (24,576 and 49,152 are both
    Kolibri calls and boundary counts, so both routings); a count P3_TABLE
    does not know is probed with "rows" routing (any count is a valid call)."""
    known: dict[int, list[tuple[int, str, str]]] = {}
    for rows, routing, source in P3_TABLE:
        known.setdefault(rows, []).append((rows, routing, source))
    entries = []
    for n in counts:
        n = int(n)
        entries += known.get(n, [(n, "rows", "thresholds.json P3_G0k")])
    return merge_shapes(entries)


def result_key(projection: str, bits: int, shape: Shape) -> str:
    return f"{projection}|{int(bits)}|{shape.key}"


def expected_keys(shapes=None, bits=BITS, projections=None) -> list[str]:
    shapes = JUDGED if shapes is None else shapes
    projections = list(PROJECTIONS) if projections is None else list(projections)
    return [result_key(p, b, s) for p in projections for b in bits for s in shapes]


# thresholds.json P3_G0k "judged_rows" group -> the routing of the calls it stands for
GROUP_ROUTING = {
    "decode_kolibri": "tokens", "prefill_chunk_kolibri": "tokens", "batched_prefill_kolibri": "tokens",
    "first_wave_exp036": "tokens", "g2_single_call": "tokens", "b16_bound": "tokens",
    "decode_peers": "rows", "prefill_chunk_peers": "rows", "batched_prefill_peers": "rows",
    "boundary_mlx_repro_min": "rows",
}


def config_from_thresholds(block: dict) -> dict:
    """thresholds.json "P3_G0k" -> probe()'s keyword arguments {shapes, bits,
    projections, seed, bad_row_fraction}: every count of "judged_rows" with
    its group's routing, and "judged_rows_one_expert" as hot rows. Refuses a
    block whose weight layout differs from this module's (E, group size,
    affine, bf16 activations, the two projections), a row group it does not
    know, or a non-empty "recorded_only" (F2 judges every count)."""
    if (block.get("experts"), block.get("group_size"), block.get("mode"), block.get("activation_dtype")) != \
            (E, GROUP_SIZE, "affine", "bfloat16"):
        raise ValueError("P3_G0k: the weight layout differs from gate/checks/g0k_kernels.py's")
    projections = block.get("projections") or {}
    for name, dims in projections.items():
        if name not in PROJECTIONS or (dims.get("n_out"), dims.get("d_in")) != PROJECTIONS[name]:
            raise ValueError(f"P3_G0k: projection {name} {dims} differs from PROJECTIONS")
    if block.get("recorded_only"):
        raise ValueError("P3_G0k: recorded-only counts are not probed (decision F2 judges every count)")
    entries = []
    for group, counts in (block.get("judged_rows") or {}).items():
        if group not in GROUP_ROUTING:
            raise ValueError(f"P3_G0k: unknown row group {group!r}")
        entries += [(n, GROUP_ROUTING[group], group) for n in counts]
    entries += [(n, "hot", "judged_rows_one_expert") for n in block.get("judged_rows_one_expert", [])]
    if not entries:
        raise ValueError("P3_G0k: no judged row counts")
    return {"shapes": merge_shapes(entries), "bits": tuple(int(b) for b in block["bits"]),
            "projections": tuple(projections) or tuple(PROJECTIONS), "seed": int(block["seed"]),
            "bad_row_fraction": float(block["bad_row_rel"])}


# ---------------------------------------------------------------------------
# Measurement (numpy only)
# ---------------------------------------------------------------------------


def rel(a, ref) -> float:
    """mlx_repro.py's rel: ||a - ref|| / ||ref||, float64."""
    a = np.asarray(a, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.linalg.norm(a - ref) / max(np.linalg.norm(ref), 1e-300))


class Accumulator:
    """The G0k statistics over row blocks: squared-error sums against the
    float64 reference, and the per-row bad-row count of mlx_repro_min.py
    (||sorted - unsorted|| > fraction x ||unsorted||, row by row)."""

    def __init__(self, bad_row_fraction: float = BAD_ROW_FRACTION):
        self.fraction = float(bad_row_fraction)
        self.rows = 0
        self.ss_ref = 0.0
        self.ss_sorted = 0.0
        self.ss_unsorted = 0.0
        self.bad_rows = 0
        self.first_bad_row = None
        self.max_row_rel = 0.0

    def add(self, y_sorted, y_unsorted, ref) -> None:
        ys = np.asarray(y_sorted, dtype=np.float64)
        yu = np.asarray(y_unsorted, dtype=np.float64)
        rf = np.asarray(ref, dtype=np.float64)
        if not (ys.shape == yu.shape == rf.shape):
            raise ValueError(f"G0k: shapes differ: {ys.shape}, {yu.shape}, {rf.shape}")
        ys, yu, rf = (a.reshape(-1, a.shape[-1]) for a in (ys, yu, rf))
        self.ss_ref += float(np.sum(rf * rf))
        self.ss_sorted += float(np.sum((ys - rf) ** 2))
        self.ss_unsorted += float(np.sum((yu - rf) ** 2))
        d = np.linalg.norm(ys - yu, axis=1)
        u = np.linalg.norm(yu, axis=1)
        bad = d > self.fraction * u
        if bad.any():
            if self.first_bad_row is None:
                self.first_bad_row = self.rows + int(np.argmax(bad))
            self.bad_rows += int(bad.sum())
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(u > 0, d / np.where(u > 0, u, 1.0), np.where(d > 0, np.inf, 0.0))
        if r.size:
            self.max_row_rel = max(self.max_row_rel, float(r.max()))
        self.rows += ys.shape[0]

    @staticmethod
    def _ratio(num: float, den: float) -> float:
        if den > 0:
            return float(np.sqrt(num / den))
        return 0.0 if num == 0 else float("inf")

    def result(self) -> dict:
        rs = self._ratio(self.ss_sorted, self.ss_ref)
        ru = self._ratio(self.ss_unsorted, self.ss_ref)
        return {
            "rows": self.rows,
            "rel_f64_sorted": rs,
            "rel_f64_unsorted": ru,
            "ratio_sorted_to_unsorted": (rs / ru) if ru > 0 else (0.0 if rs == 0 else float("inf")),
            "bad_rows": self.bad_rows,
            "first_bad_row": self.first_bad_row,
            "max_row_rel_sorted_vs_unsorted": self.max_row_rel,
            "bad_row_fraction": self.fraction,
        }


def bitwise_equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def measure(y_sorted, y_sorted_repeat, y_unsorted, ref, bad_row_fraction: float = BAD_ROW_FRACTION) -> dict:
    """The G0k measurement of one call on whole arrays [rows, n_out] (or
    [rows, 1, n_out]): the probe's own statistics, usable on synthetic arrays."""
    acc = Accumulator(bad_row_fraction)
    acc.add(y_sorted, y_unsorted, ref)
    return {**acc.result(), "repeat_bitwise": bitwise_equal(y_sorted, y_sorted_repeat)}


# ---------------------------------------------------------------------------
# Inputs, weights and the op (mlx imported lazily)
# ---------------------------------------------------------------------------


def bf16_bits_to_f64(bits) -> np.ndarray:
    """uint16 bfloat16 bit patterns -> float64 (exact)."""
    return (np.asarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def _bf16_bits(rng, rows: int, cols: int) -> np.ndarray:
    """Standard-normal float32 [rows, cols] rounded to bfloat16 by mlx (as
    mlx_repro.py's bf16_round), returned as uint16 bit patterns; in chunks, so
    no full float32 copy exists."""
    import mlx.core as mx

    out = np.empty((rows, cols), dtype=np.uint16)
    step = max(1, _CHUNK_ELEMS // max(cols, 1))
    for a in range(0, rows, step):
        b = min(rows, a + step)
        x = rng.standard_normal((b - a, cols), dtype=np.float32)
        y = mx.array(x).astype(mx.bfloat16).view(mx.uint16)
        out[a:b] = np.array(y)
    return out


def route_uniform(rng, T: int) -> np.ndarray:
    """mlx_repro.py: each of T tokens routed to K distinct uniformly drawn experts."""
    return np.argsort(rng.random((T, E)), axis=1)[:, :K].astype(np.int64)


def flatten_sorted(tok_idx: np.ndarray):
    """mlx_repro.py (index part): rows per (token, expert), stably sorted by
    expert, as SwitchGLU builds them. Returns (expert per row, token per row)."""
    flat = tok_idx.reshape(-1)
    order = np.argsort(flat, kind="stable")
    return flat[order], order // tok_idx.shape[1]


def make_inputs(shape: Shape, projection: str, d_in: int, seed: int = SEED):
    """(x as bf16 bit patterns [rows, d_in], sorted expert ids [rows]) for one
    shape. The routing depends on (seed, routing, rows) only, so both
    projections see the same expert ids; the inputs on the projection too."""
    n = shape.rows
    rng_route = np.random.default_rng([seed, _ROUTING_CODE[shape.routing], n])
    rng_x = np.random.default_rng([seed, _ROUTING_CODE[shape.routing], n, 1 + _PROJ_CODE[projection]])
    if shape.routing == "tokens":
        T = n // K
        idx, tok = flatten_sorted(route_uniform(rng_route, T))
        if projection == "gate_up":  # gate/up multiply each token's hidden state once per expert
            x_bits = _bf16_bits(rng_x, T, d_in)[tok]
        else:  # down_proj's input is one activation per (token, expert) row
            x_bits = _bf16_bits(rng_x, n, d_in)
    elif shape.routing == "rows":  # mlx_repro_min.py's case
        idx = np.sort(rng_route.integers(0, E, n))
        x_bits = _bf16_bits(rng_x, n, d_in)
    else:  # hot: every row on one expert
        idx = np.full(n, HOT_EXPERT, dtype=np.int64)
        x_bits = _bf16_bits(rng_x, n, d_in)
    return x_bits, idx.astype(np.int64)


def make_weights(projection: str, bits: int, seed: int = SEED):
    """mlx_repro.py's make_weights for one projection: per-expert seeded weights
    [E, n_out, d_in] x 0.02, bf16-rounded, quantised (affine, group 64);
    returns the mlx parts (q, scales, biases) and the dequantised weights
    [E, n_out, d_in] float32 (dequantised with float32 scales and biases, as
    mlx_repro.py; reference_blocks multiplies them in float64).
    gate_up draws default_rng([seed, e]), as mlx_repro.py; down draws
    default_rng([seed, 1, e])."""
    import mlx.core as mx

    n_out, d_in = PROJECTIONS[projection]
    wq, sc, bi = [], [], []
    deq = np.empty((E, n_out, d_in), dtype=np.float32)
    prefix = [seed] if projection == "gate_up" else [seed, _PROJ_CODE[projection]]
    for e in range(E):
        r = np.random.default_rng(prefix + [e])
        w = mx.array(r.standard_normal((n_out, d_in), dtype=np.float32) * 0.02).astype(mx.bfloat16)
        q, s, b = mx.quantize(w, group_size=GROUP_SIZE, bits=bits)
        d = mx.dequantize(q, s.astype(mx.float32), b.astype(mx.float32), group_size=GROUP_SIZE, bits=bits)
        mx.eval(q, s, b, d)
        wq.append(q)
        sc.append(s)
        bi.append(b)
        deq[e] = np.array(d)
    W = (mx.stack(wq), mx.stack(sc), mx.stack(bi))
    mx.eval(*W)
    del wq, sc, bi, w, q, s, b, d
    mx.clear_cache()  # freed per-expert buffers stay in mlx's cache otherwise
    return W, deq


def call(x, idx, W, sorted_flag: bool, bits: int):
    """mlx_repro.py's call, bf16 path, kept on the device: x [rows, 1, d_in]
    bf16, idx [rows] uint32; returns the evaluated bf16 output [rows, 1, n_out]."""
    import mlx.core as mx

    q, s, b = W
    y = mx.gather_qmm(x, q, s, b, rhs_indices=idx, transpose=True, group_size=GROUP_SIZE, bits=bits,
                      mode="affine", sorted_indices=sorted_flag)
    mx.eval(y)
    return y


def reference_blocks(x_bits: np.ndarray, idx: np.ndarray, deq: np.ndarray):
    """mlx_repro.py's reference, one expert segment at a time (idx is sorted):
    yields (a, b, float64 x[a:b] @ deq[e]^T)."""
    n = idx.shape[0]
    if n == 0:
        return
    starts = np.flatnonzero(np.r_[True, idx[1:] != idx[:-1]])
    ends = np.r_[starts[1:], n]
    for a, b in zip(starts.tolist(), ends.tolist()):
        e = int(idx[a])
        yield a, b, bf16_bits_to_f64(x_bits[a:b]) @ deq[e].astype(np.float64).T


def probe_one(shape: Shape, projection: str, bits: int, W, deq, seed: int = SEED,
              bad_row_fraction: float = BAD_ROW_FRACTION) -> dict:
    """Measure one (projection, bits, shape)."""
    import mlx.core as mx

    t0 = time.time()
    n_out, d_in = PROJECTIONS[projection]
    x_bits, idx = make_inputs(shape, projection, d_in, seed)
    x = mx.array(x_bits).view(mx.bfloat16)[:, None, :]
    ridx = mx.array(idx.astype(np.uint32))
    ys = call(x, ridx, W, True, bits)
    yr = call(x, ridx, W, True, bits)
    repeat = bool(mx.array_equal(ys.view(mx.uint16), yr.view(mx.uint16)).item())
    del yr
    ys_bits = np.array(ys.view(mx.uint16)).reshape(shape.rows, n_out)
    del ys
    yu = call(x, ridx, W, False, bits)
    yu_bits = np.array(yu.view(mx.uint16)).reshape(shape.rows, n_out)
    del x, yu
    mx.clear_cache()  # the largest shapes' device buffers are not reused
    acc = Accumulator(bad_row_fraction)
    for a, b, ref in reference_blocks(x_bits, idx, deq):
        acc.add(bf16_bits_to_f64(ys_bits[a:b]), bf16_bits_to_f64(yu_bits[a:b]), ref)
    out = {
        "projection": projection, "bits": int(bits), **shape.as_dict(),
        **acc.result(), "repeat_bitwise": repeat,
        "experts_touched": int(np.unique(idx).size),
        "max_rows_per_expert": int(np.bincount(idx, minlength=E).max()),
        "seconds": round(time.time() - t0, 3),
    }
    return out


def env_record(seed: int = SEED) -> dict:
    """mlx_repro.py's env_record."""
    import mlx.core as mx

    info = mx.device_info()
    try:
        import mlx_lm

        mlx_lm_v = mlx_lm.__version__
    except Exception:
        mlx_lm_v = None
    return {"mlx": mx.__version__, "mlx_lm": mlx_lm_v, "device": info.get("device_name"),
            "arch": info.get("architecture"), "macos": platform.mac_ver()[0],
            "python": platform.python_version(), "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32"),
            "seed": seed}


def probe(shapes=None, bits=BITS, projections=None, seed: int = SEED,
          bad_row_fraction: float = BAD_ROW_FRACTION, log=None) -> dict:
    """G0k: every (projection, bits, shape), measured. `shapes` defaults to the
    judged set of §3.1 P3 (JUDGED); a list of ints is read as judged row
    counts (judged_shapes). Returns
        {"results": {result_key: {rel_f64_sorted, rel_f64_unsorted, bad_rows,
                                  repeat_bitwise, ...}}, "shapes", "env", ...}
    and never a verdict (gate/preconditions.p3 judges)."""
    import mlx.core as mx

    if shapes is None:
        shapes = JUDGED
    shapes = tuple(shapes)
    if shapes and not isinstance(shapes[0], Shape):
        shapes = judged_shapes(shapes)
    projections = list(PROJECTIONS) if projections is None else list(projections)
    for p in projections:
        if p not in PROJECTIONS:
            raise ValueError(f"G0k: unknown projection {p!r}")
    bits = tuple(int(b) for b in bits)
    t0 = time.time()
    results = {}
    for projection in projections:
        for b in bits:
            tw = time.time()
            W, deq = make_weights(projection, b, seed)
            if log:
                log(f"[g0k] {projection} {b}-bit weights in {time.time() - tw:.1f}s")
            for shape in shapes:
                r = probe_one(shape, projection, b, W, deq, seed, bad_row_fraction)
                results[result_key(projection, b, shape)] = r
                if log:
                    log(f"[g0k] {projection} {b}-bit {shape.routing:6s} rows {shape.rows:6d}  "
                        f"rel sorted {r['rel_f64_sorted']:.5f} unsorted {r['rel_f64_unsorted']:.5f}  "
                        f"bad {r['bad_rows']}  repeat {r['repeat_bitwise']}  {r['seconds']:.2f}s")
            del W, deq
            mx.clear_cache()
    return {
        "what": "G0k: sorted vs unsorted mx.gather_qmm against float64 at the run's shapes (DESIGN §3.1 P3)",
        "env": env_record(seed),
        "seed": seed,
        "weights": {"E": E, "group_size": GROUP_SIZE, "mode": "affine", "activations": "bfloat16",
                    "projections": {p: {"n_out": PROJECTIONS[p][0], "d_in": PROJECTIONS[p][1]} for p in projections},
                    "bits": list(bits)},
        "bad_row_fraction": bad_row_fraction,
        "shapes": [s.as_dict() for s in shapes],
        "results": results,
        "seconds": round(time.time() - t0, 1),
    }


def default_out_dir() -> Path:
    """G0k outputs are exp_037 work files (DESIGN §2.9): under work37_dir()."""
    from gate import common

    return common.work37_dir() / "g0k"


def main(argv=None) -> int:
    """Diagnostic run (a gate run calls probe() from its phase 0). Without --rows
    it probes thresholds.json's P3_G0k set."""
    ap = argparse.ArgumentParser(description="exp_037 G0k kernel probe (DESIGN §3.1 P3); measures, decides nothing")
    ap.add_argument("--out", default=None, help="output JSON (default: $EXP036_WORK/exp037/g0k/g0k_<UTC>.json)")
    ap.add_argument("--rows", type=int, nargs="*", default=None, help="row counts (default: the judged set)")
    ap.add_argument("--bits", type=int, nargs="*", default=list(BITS))
    ap.add_argument("--projection", nargs="*", default=list(PROJECTIONS))
    a = ap.parse_args(argv)
    from tools.precision import ensure_exact_fp32  # before any GPU work

    ensure_exact_fp32()
    from gate import common

    if a.rows is None:
        cfg = config_from_thresholds(common.load_thresholds()["P3_G0k"])
        shapes, seed, frac = cfg["shapes"], cfg["seed"], cfg["bad_row_fraction"]
    else:
        shapes, seed, frac = judged_shapes(a.rows), SEED, BAD_ROW_FRACTION
    res = probe(shapes, a.bits, a.projection, seed=seed, bad_row_fraction=frac,
                log=lambda m: print(m, flush=True))
    out = Path(a.out) if a.out else default_out_dir() / f"g0k_{common.utc_stamp()}.json"
    common.write_new_json(out, res)
    print(f"[g0k] wrote {common.redact_path(out)} ({res['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
