# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06; build task W5b).
"""G4-F32 and G4-F16: the port forced onto I end to end, with their calibrators
(DESIGN §3.0, §3.7, §3.8, §4.2, §5.1-§5.5; build task W5b).

Measurement only. Every function here returns measured values (numpy series,
or JSON-ready dicts); every decision is gate/rules.py's (g4_f32, g4_f16,
control_result), applied by run_gate.py (W7) to the dicts built here. The
dict contracts are the ones rules.py's docstring states.

The port and its comparators (DESIGN §3.0):
  I          I[l, n] = R1's layerNN.top6.npy[n], sorted ascending (pack order
             T1..T8 then T9; ref_drivers.i_from_r1). Forcing ids never come
             from the port.
  R2F, R3    the fp32 reference on the dequantised K8 weights forced onto I,
             and its bf16 emulation (ref_drivers.forced_ref_pass; ForcedDump
             directories with logits, the shadow top-6 and the 6th-7th gap).
  S(t)       KL(R2F||R3)(t): the sensitivity profile, reference-only (§5.5).
  port       forced onto I by harness.ForcedTrunk ("force" mode: the ids are
             handed to the port ascending; the weights still come from the
             port's own route(), so weight-path mutants stay visible), with
             its shadow (the port's own selection on the unboosted scores).
             G4-F32: the K8 fp32 port (make_fp32_port: set_dtype(float32),
             then fp32_on_dequantised_weights, lifted verbatim from exp_036
             diagnostics/gate1/diaglib.py l.247-279, file sha256 2897597e...
             in COPY_RECORD.json). G4-F16: the K8 build as loaded (bf16).
  Csort      the port forced onto its own free-run ids sorted ascending,
             against that free run, with identical chunking (§3.0, §5.2).

Per position (forced_series): KL(R2F||port), R2F's lead (top-1 minus top-2
logit, which equals the log-probability lead), top-1 agreement, S, R3's top-1
agreement; per (layer, position): the port's shadow set against R2F's, R2F's
6th-7th biased gap, R3's shadow set against R2F's. Every KL is
common.kl_rows(ref_logits, q_logits): a float64 log-softmax of fp32 logits
over the full vocabulary (DESIGN §3.0; never a KL of fp32 log-probabilities).

Reductions (rules.py's contracts):
  g4_f32_measured  {"sets": {"T1-8": S, "T9": S}, "t9_buckets": {"lo-hi": B},
                   "descriptive": {...}, "meta": {...}}; S and B as rules.py
                   states (csort_* only on the unmutated run, which carries
                   Csort; mutants are judged with its tau and Csort).
  g4_f16_measured  {"T1-8": {"mean_kl_port", "mean_kl_r3", descriptive...},
                   "T9": {...}, "meta": {...}}.
Descriptive only (never a rule): F4's +-32 windows around 2,048 k and the T9
document starts (§1.3: F4 is not a rule), Csort's maximum and its largest
position's share, per-text and per-bucket values, shadow rates; for G4-F16
the p99 ratio, decisive misses of port and R3, shadow rates and the G4-D5
sparse-spike count.

Mutants (controls.json's lists; §4.2): mutant_loop loads each required mutant
once (bf16 as loaded), runs its bf16 checks, converts it in place to the fp32
port for its fp32 checks (mutant 6 serves G4-F16 and G4-F32), then releases
it (mx.clear_cache) before the next load. Memory order around the unmutated
model is run_gate.py's (W7).

mlx is imported lazily.
"""

from __future__ import annotations

import gc
import math
import time
from pathlib import Path

import numpy as np

from gate import common, harness, rules
from gate.checks import ref_pass
from gate.checks.ref_drivers import ForcedDump
from gate.harness import ForcedTrunk, HarnessError

ORDER8 = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
STANDARD_SETS = {"T1-8": ORDER8, "T9": ("T9",)}
KL_ESTIMATOR = ("common.kl_rows(ref_logits, q_logits): float64 log-softmax of fp32 logits over the full "
                "vocabulary, KL(ref || q) in nats per position")
G4_CONTROL_CHECKS = ("g4_f16_K8", "g4_f32_K8")
PRECISIONS = ("bf16", "fp32")  # the order a mutant runs in: bf16 as loaded, then converted in place
FP32_PORT_ATTR = "exp037_fp32_port"  # set on a model by make_fp32_port (a bool: kept in __dict__)


# ---------------------------------------------------------------------------
# The K8 fp32 port (DESIGN §3.0)
# ---------------------------------------------------------------------------


def fp32_on_dequantised_weights(model) -> None:
    """After model.set_dtype(float32): make the port multiply exactly the
    dequantised K8 weights the reference's dequantised mode reads.

    Linear layers already do: quantized_matmul dequantises in fp32 whatever the
    scales' dtype, and the reference reads mx.dequantize with fp32 scales
    (tests/test_convert.py test_dequantised_mode_reads_the_mx_dequantize_values).
    The quantised embedding does not: QuantizedEmbedding dequantises in its
    scales' dtype, so set_dtype(float32) turns its stored bf16 dequantisation
    (the value the reference reads) into an fp32 one, a different weight that
    differs by up to a bf16 rounding. Measured on the tiny K8 build: mean
    KL(ref_deq || port fp32) 1.8e-3 with set_dtype alone (above the bug test's
    1e-3 bound with no defect anywhere), 8e-13 with the embedding restored.
    So the embedding keeps its stored bf16 scales and biases (an exact round
    trip), and its output is cast to fp32 before the trunk."""
    import mlx.core as mx
    import mlx.nn as nn

    emb = model.model.embed_tokens
    if not hasattr(emb, "scales"):
        return

    class _Fp32Out(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def __call__(self, x):
            return self.inner(x).astype(mx.float32)

    emb.scales = emb.scales.astype(mx.bfloat16)
    emb.biases = emb.biases.astype(mx.bfloat16)
    model.model.embed_tokens = _Fp32Out(emb)


def is_fp32_port(model) -> bool:
    return bool(getattr(model, "__dict__", {}).get(FP32_PORT_ATTR, False))


def make_fp32_port(model):
    """The K8 fp32 port, in place (DESIGN §3.0): model.set_dtype(float32), then
    fp32_on_dequantised_weights(model). Applied once: a second set_dtype would
    cast the embedding's restored bf16 scales to fp32 again, so a model already
    converted is returned unchanged."""
    import mlx.core as mx

    if is_fp32_port(model):
        return model
    model.set_dtype(mx.float32)
    fp32_on_dequantised_weights(model)
    model.__dict__[FP32_PORT_ATTR] = True
    return model


def release_memory() -> None:
    """Drop unreferenced arrays and MLX's buffer cache (before every mutant load)."""
    gc.collect()
    try:
        import mlx.core as mx
    except ImportError:
        return
    clear = getattr(mx, "clear_cache", None) or getattr(getattr(mx, "metal", None), "clear_cache", None)
    if clear is not None:
        clear()


# ---------------------------------------------------------------------------
# Layout: texts in pack order, sets, spans
# ---------------------------------------------------------------------------


def text_layout(ts) -> list:
    """[(name, ids int64)] in R1's pack order: T1..T8, then T9 (DESIGN §3.0)."""
    return [(t, np.asarray(ts.ids[t], dtype=np.int64).reshape(-1)) for t in ORDER8] + \
        [("T9", np.asarray(ts.t9, dtype=np.int64).reshape(-1))]


def _texts(texts) -> list:
    out = [(str(n), np.asarray(ids, dtype=np.int64).reshape(-1)) for n, ids in texts]
    names = [n for n, _ in out]
    if not out or len(set(names)) != len(names) or any(ids.size == 0 for _, ids in out):
        raise ValueError(f"texts must be non-empty, with distinct names: {names}")
    return out


def pack_spans(texts) -> dict:
    """{name: (start, end)}: each text's pack rows, in the order given."""
    out, p = {}, 0
    for n, ids in texts:
        out[n] = (p, p + int(np.asarray(ids).size))
        p += int(np.asarray(ids).size)
    return out


def default_sets(names) -> dict:
    """The gate's sets when the texts are T1..T8 and T9, else one set per text."""
    names = list(names)
    if all(n in names for n in ORDER8 + ("T9",)):
        return {k: tuple(v) for k, v in STANDARD_SETS.items()}
    return {n: (n,) for n in names}


def chunk_boundaries(n: int, chunk) -> list:
    """Positions in (0, n) where a forward starts: k x chunk, or the starts of
    explicit spans."""
    return [a for a, _ in harness.chunk_spans(int(n), chunk) if a > 0]


def block_means(values, block: int = 2048) -> list:
    """Means of consecutive `block`-position blocks (the last may be shorter)."""
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    return [float(v[a:a + block].mean()) for a in range(0, v.size, int(block))]


def _dump(x):
    """A forced dump: a ForcedDump, a directory (wrapped), or any object with
    logits() (and, for shadows, shadow(i) and gap(i))."""
    if x is None:
        return None
    if isinstance(x, (str, Path)):
        d = ForcedDump(x)
        if not d.complete:
            raise HarnessError(f"{common.redact_path(x)}: incomplete forced dump (no record.json)")
        return d
    return x


def _has_shadows(d) -> bool:
    return d is not None and hasattr(d, "shadow") and hasattr(d, "gap")


def _check_dump(d, n_rows: int, lengths: list, what: str) -> None:
    rows = int(d.logits().shape[0])
    if rows != n_rows:
        raise HarnessError(f"{what} has {rows} logit rows, the pack {n_rows}")
    rec = d.record() if hasattr(d, "record") and getattr(d, "complete", False) else None
    if rec is not None and rec.get("seq_lengths") is not None and list(rec["seq_lengths"]) != list(lengths):
        raise HarnessError(f"{what}'s sequences {rec['seq_lengths']} are not the pack's {lengths}")


# ---------------------------------------------------------------------------
# Reference-only columns: R2F's lead, top-1 and shadows; S (§5.5); R3's top-1
# and shadows. Computed once per gate run and reused by every forced pass.
# ---------------------------------------------------------------------------


def s_profile(R2F, R3, start: int, n: int, rows_chunk: int = 512) -> np.ndarray:
    """S(t) = KL(R2F||R3)(t) for pack rows [start, start + n) (DESIGN §3.0, §5.5)."""
    a2, a3 = _dump(R2F).logits(), _dump(R3).logits()
    out = np.empty(n, dtype=np.float64)
    for a in range(0, n, rows_chunk):
        b = min(n, a + rows_chunk)
        out[a:b] = common.kl_rows(a2[start + a:start + b], a3[start + a:start + b])
    return out


def reference_profile(texts, R2F, R3=None, *, num_layers: int, rows_chunk: int = 512, log=print) -> dict:
    """The reference-only columns of every text, from the forced dumps:
    {"names", "pack", "num_layers", "top_k", "has_r3", "has_shadow",
     "cols": {name: {"lead", "top1", "S", "r3_top1", "shadow_sorted" [L, n, k],
                     "gap" [L, n], "r3_shadow_dis" [L, n]}}} (None where a dump
    lacks the input)."""
    texts = _texts(texts)
    pack = pack_spans(texts)
    n_rows = sum(ids.size for _, ids in texts)
    lengths = [int(ids.size) for _, ids in texts]
    R2F, R3 = _dump(R2F), _dump(R3)
    _check_dump(R2F, n_rows, lengths, "R2F")
    if R3 is not None:
        _check_dump(R3, n_rows, lengths, "R3")
    shadows = _has_shadows(R2F)
    r3_shadows = shadows and _has_shadows(R3)
    logits2 = R2F.logits()
    logits3 = R3.logits() if R3 is not None else None
    t0 = time.time()
    top_k = None
    cols = {}
    for name, ids in texts:
        a0, b0 = pack[name]
        n = b0 - a0
        lead = np.empty(n, dtype=np.float64)
        top1 = np.empty(n, dtype=np.int64)
        S = np.empty(n, dtype=np.float64) if R3 is not None else None
        r3_top1 = np.empty(n, dtype=np.int64) if R3 is not None else None
        for a in range(0, n, rows_chunk):
            b = min(n, a + rows_chunk)
            ref = np.asarray(logits2[a0 + a:a0 + b], dtype=np.float32)
            lead[a:b] = common.top_lead(ref)
            top1[a:b] = ref.argmax(axis=-1)
            if R3 is not None:
                r3 = np.asarray(logits3[a0 + a:a0 + b], dtype=np.float32)
                S[a:b] = common.kl_rows(ref, r3)
                r3_top1[a:b] = r3.argmax(axis=-1)
        c = {"lead": lead, "top1": top1, "S": S, "r3_top1": r3_top1,
             "shadow_sorted": None, "gap": None, "r3_shadow_dis": None}
        if shadows:
            sh = [np.sort(np.asarray(R2F.shadow(i)[a0:b0], dtype=np.int64), axis=-1) for i in range(num_layers)]
            c["shadow_sorted"] = np.stack(sh)
            c["gap"] = np.stack([np.asarray(R2F.gap(i)[a0:b0], dtype=np.float32) for i in range(num_layers)])
            top_k = int(c["shadow_sorted"].shape[-1])
            if r3_shadows:
                c["r3_shadow_dis"] = np.stack([ref_pass.set_disagree(np.asarray(R3.shadow(i)[a0:b0]), sh[i])
                                               for i in range(num_layers)])
        cols[name] = c
    log(f"[gate] G4 forced: reference profile of {n_rows} rows ({time.time() - t0:.1f}s)")
    return {"kind": "g4_reference_profile", "names": [n for n, _ in texts], "pack": pack,
            "num_layers": int(num_layers), "top_k": top_k, "has_r3": R3 is not None, "has_shadow": shadows,
            "cols": cols}


def _check_profile(ref: dict, texts, num_layers: int) -> None:
    names = [n for n, _ in texts]
    if ref.get("names") != names or ref.get("pack") != pack_spans(texts) or ref.get("num_layers") != num_layers:
        raise HarnessError("the reference profile was built for another layout or layer count")


# ---------------------------------------------------------------------------
# The forced series (G4-F32, G4-F16, F_tiny): the port forced onto I
# ---------------------------------------------------------------------------


def _check_I(I, num_layers: int, n_rows: int) -> int:
    if len(I) != num_layers:
        raise HarnessError(f"I has {len(I)} layers, the model {num_layers}")
    shape = tuple(np.shape(I[0]))
    if len(shape) != 2 or shape[0] != n_rows:
        raise HarnessError(f"I[0] has shape {shape}; the pack has {n_rows} rows")
    return int(shape[1])


def forced_texts(model, texts, I, R2F, R3=None, chunk=2048, *, module, ref: dict | None = None,
                 sets: dict | None = None, log=print) -> dict:
    """The port forced onto I over `texts` ([(name, ids)] in pack order, each
    an independent sequence from position 0, fed in `chunk`-position forwards
    through one cache), against R2F (and R3 for S). module: the port module the
    model was built from (ForcedTrunk patches its route). ref: a
    reference_profile of the same layout (computed here when None).

    Returns the series: {"kind", "names", "sets", "pack", "num_layers",
    "top_k", "chunk", "seconds", "cols": {name: {"kl", "lead", "agree", "S",
    "r3_agree", "shadow_dis" [L, n], "gap" [L, n], "r3_shadow_dis" [L, n]}}}.
    HarnessError when the forcing, row or layer bookkeeping does not hold."""
    texts = _texts(texts)
    L = harness.num_layers_of(model)
    pack = pack_spans(texts)
    n_rows = sum(ids.size for _, ids in texts)
    k = _check_I(I, L, n_rows)
    R2F, R3 = _dump(R2F), _dump(R3)
    if ref is None:
        ref = reference_profile(texts, R2F, R3, num_layers=L, log=log)
    else:
        _check_profile(ref, texts, L)
        _check_dump(R2F, n_rows, [int(ids.size) for _, ids in texts], "R2F")
    if R3 is not None and not ref["has_r3"]:
        raise HarnessError("R3 given, but the reference profile was built without it")
    logits = R2F.logits()
    t0 = time.time()
    cols = {}
    for name, ids in texts:
        a0, b0 = pack[name]
        rows = np.arange(a0, b0, dtype=np.int64)
        n = rows.size
        ft = ForcedTrunk(module, L, "force", table=I)
        kl = np.empty(n, dtype=np.float64)
        top1 = np.empty(n, dtype=np.int64)
        for a, blk in harness.forced_stream(model, ft, rows, ids, chunk):
            b = a + blk.shape[0]
            kl[a:b] = common.kl_rows(logits[a0 + a:a0 + b], blk)
            top1[a:b] = blk.argmax(axis=-1)
        rec = ft.records()
        if not np.array_equal(rec["rows"], rows):
            raise HarnessError(f"{name}: the forced pass recorded other rows than the text's pack rows")
        rc = ref["cols"][name]
        shadow_dis = None
        if rc["shadow_sorted"] is not None:
            if rec["shadow"].shape != rc["shadow_sorted"].shape:
                raise HarnessError(f"{name}: port shadow {rec['shadow'].shape} vs R2F shadow {rc['shadow_sorted'].shape}")
            shadow_dis = np.any(np.sort(rec["shadow"], axis=-1) != rc["shadow_sorted"], axis=-1)
        cols[name] = {"kl": kl, "lead": rc["lead"], "agree": top1 == rc["top1"], "S": rc["S"],
                      "r3_agree": None if rc["r3_top1"] is None else rc["r3_top1"] == rc["top1"],
                      "shadow_dis": shadow_dis, "gap": rc["gap"], "r3_shadow_dis": rc["r3_shadow_dis"]}
        log(f"[gate] G4 forced {name}: mean KL(R2F||port) {kl.mean():.3e}")
    return {"kind": "g4_forced_series", "names": [n for n, _ in texts],
            "sets": {s: tuple(v) for s, v in (sets or default_sets(n for n, _ in texts)).items()},
            "pack": pack, "num_layers": L, "top_k": k,
            "chunk": int(chunk) if isinstance(chunk, (int, np.integer)) else [list(s) for s in chunk],
            "seconds": time.time() - t0, "cols": cols}


def forced_series(model, ts, I, R2F, R3=None, chunk=2048, *, module, ref: dict | None = None, log=print) -> dict:
    """forced_texts over a gate text set (T1..T8, each one forward of
    text_len <= chunk positions, then T9 in `chunk`-position forwards through
    one cache): per-position KL(R2F||port), R2F's lead, top-1 agreement, S and
    the shadow comparisons (DESIGN §3.7, §3.8)."""
    return forced_texts(model, text_layout(ts), I, R2F, R3, chunk, module=module, ref=ref, log=log)


# ---------------------------------------------------------------------------
# Csort (§3.0, §5.2): forced onto its own free-run ids, sorted, vs the free run
# ---------------------------------------------------------------------------


def csort_sequence(model, ids, chunk=2048, *, module, positions=None) -> dict:
    """One sequence from position 0: (i) the free run, recording the ids of
    every forward, and (ii) the port forced onto (i)'s ids sorted ascending
    ("force" mode), with identical forwards (`chunk`: a size or explicit
    spans), each run through its own cache. The two runs advance forward by
    forward, so no [T, V] matrix is held.

    Returns {"kl": KL((i)||(ii)) per position [T] float64, "table": (i)'s ids
    [L, T, k] (route()'s order; force them sorted, as (ii) did), "spans"}, plus
    "free_at" and "sorted_at" (fp32 logits rows at `positions`, in the order
    given) when positions are given (G5-D32's comparator (ii))."""
    ids = np.asarray(ids, dtype=np.int64).reshape(-1)
    T = ids.size
    if T == 0:
        raise ValueError("csort_sequence needs a non-empty sequence")
    L = harness.num_layers_of(model)
    spans = harness.chunk_spans(T, chunk)
    pos = None if positions is None else np.asarray(positions, dtype=np.int64).reshape(-1)
    if pos is not None and pos.size and (pos.min() < 0 or pos.max() >= T):
        raise ValueError("positions outside the sequence")
    cache_free, cache_sorted = model.make_cache(), model.make_cache()
    table, ft_sorted = None, None
    kl = np.empty(T, dtype=np.float64)
    free_at = sorted_at = None
    for a, b in spans:
        rows = np.arange(a, b, dtype=np.int64)
        ft_free = ForcedTrunk(module, L, "free")
        free = harness.forced_logits(model, ft_free, rows, ids[a:b], b - a, cache=cache_free)
        rec = ft_free.records()
        if table is None:
            table = np.full((L, T, rec["ids"].shape[-1]), -1, dtype=np.int64)
            ft_sorted = ForcedTrunk(module, L, "force", table=table, shadow=False)
        table[:, a:b] = rec["ids"]
        srt = harness.forced_logits(model, ft_sorted, rows, ids[a:b], b - a, cache=cache_sorted)
        kl[a:b] = common.kl_rows(free, srt)
        if pos is not None:
            if free_at is None:
                free_at = np.empty((pos.size, free.shape[-1]), dtype=np.float32)
                sorted_at = np.empty_like(free_at)
            sel = np.flatnonzero((pos >= a) & (pos < b))
            free_at[sel] = free[pos[sel] - a]
            sorted_at[sel] = srt[pos[sel] - a]
    used = ft_sorted.records()["ids"]
    if not np.array_equal(used, np.sort(table, axis=-1)):
        raise HarnessError("Csort: the sorted run did not use the free run's ids in ascending order")
    out = {"kl": kl, "table": table, "spans": [list(s) for s in spans]}
    if pos is not None:
        out["free_at"], out["sorted_at"] = free_at, sorted_at
    return out


def csort_texts(model, texts, chunk=2048, *, module, log=print) -> dict:
    """Csort per position over `texts` (each from position 0, as forced_texts
    feeds them): {"kind", "names", "pack", "chunk", "seconds", "cols":
    {name: {"kl"}}}."""
    texts = _texts(texts)
    t0 = time.time()
    cols = {}
    for name, ids in texts:
        cols[name] = {"kl": csort_sequence(model, ids, chunk, module=module)["kl"]}
        log(f"[gate] Csort {name}: mean {cols[name]['kl'].mean():.3e}")
    return {"kind": "csort_series", "names": [n for n, _ in texts], "pack": pack_spans(texts),
            "chunk": int(chunk) if isinstance(chunk, (int, np.integer)) else [list(s) for s in chunk],
            "seconds": time.time() - t0, "cols": cols}


def csort_series(model, ts, chunk=2048, *, module, log=print) -> dict:
    """csort_texts over a gate text set (T1..T8, T9), with the chunking of
    forced_series."""
    return csort_texts(model, text_layout(ts), chunk, module=module, log=log)


# ---------------------------------------------------------------------------
# Reductions to rules.py's measured dicts
# ---------------------------------------------------------------------------


def _cols(series: dict, names) -> dict:
    """The columns of `names` concatenated (position axis last)."""
    parts = [series["cols"][n] for n in names]
    out = {}
    for key in parts[0]:
        vals = [p[key] for p in parts]
        out[key] = None if any(v is None for v in vals) else np.concatenate(vals, axis=-1)
    return out


def _csort(csort: dict | None, names, series: dict):
    if csort is None:
        return None
    if csort.get("pack") != series.get("pack") or csort.get("chunk") != series.get("chunk"):
        raise HarnessError("Csort was measured on another layout or chunking than the forced series")
    return np.concatenate([csort["cols"][n]["kl"] for n in names])


def _mean(x) -> float:
    return float(np.mean(x)) if np.size(x) else math.nan


def _max(x):
    return float(np.max(x)) if np.size(x) else None


def _quantile(x, q: float):
    return float(np.quantile(x, q)) if np.size(x) else None


def _ratio(num, den):
    return None if num is None or den is None or not den > 0 else float(num / den)


def _csort_fields(cs: np.ndarray) -> dict:
    total = float(np.sum(cs))
    return {"csort_mean": _mean(cs), "csort_max": _max(cs),
            "csort_max_share": (float(np.max(cs)) / total) if cs.size and total > 0 else None}


def _f32_set(c: dict, cs, decisive: float) -> dict:
    """One set's S dict for rules.g4_f32 (plus descriptive values)."""
    kl, lead, agree = c["kl"], c["lead"], c["agree"]
    changed = ~agree
    dec = lead >= decisive
    out = {"n": int(kl.size), "mean_kl": _mean(kl), "max_kl": _max(kl), "p99_kl": _quantile(kl, 0.99),
           "kl": rules.tail(kl), "top1_change_leads": rules.tail(lead[changed]),
           "n_top1_changes": int(changed.sum()), "n_decisive": int(dec.sum()),
           "n_top1_changes_decisive": int((changed & dec).sum())}
    if c["shadow_dis"] is not None:
        dis = c["shadow_dis"]
        out.update({"shadow_gaps": rules.tail(c["gap"][dis]), "shadow_pairs": int(dis.size),
                    "n_shadow_disagree": int(dis.sum()), "shadow_disagree_rate": _ratio(float(dis.sum()), dis.size)})
    if c["S"] is not None:
        out["mean_S"] = _mean(c["S"])
    if cs is not None:
        out.update(_csort_fields(cs))
    return out


def t9_bucket_rows(c: dict, cs, buckets) -> dict:
    """{"lo-hi": B} over T9's columns: {"lo", "hi", "n", "mean_kl", "mean_S",
    "csort_mean", ...} (csort_* only with Csort). An empty bucket is a layout
    error (ValueError)."""
    T = c["kl"].size
    out = {}
    for lo, hi in buckets:
        lo, hi = int(lo), int(hi)
        a, b = max(0, lo), min(hi, T)
        if b <= a:
            raise ValueError(f"T9 bucket [{lo}, {hi}) holds no position of a {T}-position T9")
        kl = c["kl"][a:b]
        row = {"lo": lo, "hi": hi, "n": int(b - a), "mean_kl": _mean(kl), "max_kl": _max(kl),
               "n_top1_changes": int((~c["agree"][a:b]).sum())}
        if c["S"] is not None:
            row["mean_S"] = _mean(c["S"][a:b])
        if cs is not None:
            row["csort_mean"], row["csort_max"] = _mean(cs[a:b]), _max(cs[a:b])
        out[f"{lo}-{hi}"] = row
    return out


def f4_windows(c: dict, cs, boundaries, doc_starts=(), chunk=None, half: int = 32, decisive: float = 2.0) -> dict:
    """F4, descriptive (DESIGN §3.7; §1.3: not a rule): for each window of
    +-`half` positions around a T9 boundary (a chunk start or a document
    start), the window's mean KL, mean S, rho = mean KL / mean S, the median
    rho of the sequence's other complete 2*half-position windows (the tiling
    [0, 2h), [2h, 4h), ... minus those overlapping the window), their ratio,
    Csort's window mean and the KL / Csort ratio, the window's max KL and its
    top-1 changes at R2F-decisive positions."""
    kl, S = c["kl"], c["S"]
    T = kl.size
    width = 2 * int(half)
    docs = sorted({int(d) for d in doc_starts if 0 < int(d) < T})
    chunks = sorted({int(b) for b in (chunk_boundaries(T, chunk) if chunk is not None else []) if 0 < b < T})
    bounds = sorted(set(docs) | set(chunks) | {int(b) for b in boundaries if 0 < int(b) < T})

    def rho(a, b):
        if S is None:
            return None
        ms = float(S[a:b].mean())
        return float(kl[a:b].mean()) / ms if ms > 0 else None

    tiles = [(s, rho(s, s + width)) for s in range(0, T - width + 1, width)]
    rows = []
    for bd in bounds:
        a, b = max(0, bd - half), min(T, bd + half)
        others = [r for s, r in tiles if (s + width <= a or s >= b) and r is not None and math.isfinite(r)]
        med = float(np.median(others)) if others else None
        r = rho(a, b)
        cm = _mean(cs[a:b]) if cs is not None else None
        rows.append({"boundary": bd, "kinds": (["chunk"] if bd in chunks else []) + (["document"] if bd in docs else []),
                     "lo": a, "hi": b, "mean_kl": _mean(kl[a:b]), "max_kl": _max(kl[a:b]),
                     "mean_S": None if S is None else _mean(S[a:b]), "rho": r, "rho_others_median": med,
                     "rho_ratio": _ratio(r, med), "csort_mean": cm, "kl_over_csort": _ratio(_mean(kl[a:b]), cm),
                     "decisive_top1_changes": int((~c["agree"][a:b] & (c["lead"][a:b] >= decisive)).sum())})
    return {"half_window": int(half), "chunk_starts": chunks, "document_starts": docs, "n_other_windows": len(tiles),
            "windows": rows,
            "note": "descriptive only (DESIGN §1.3, §3.7): F4 is not a rule; no value here enters a verdict"}


def g4_f32_measured(series: dict, csort: dict | None = None, *, buckets, doc_starts=(), th: dict | None = None) -> dict:
    """rules.g4_f32's measured dict from a forced series (the K8 fp32 port) and,
    for the unmutated run, its Csort series (None for a mutant: mutants are
    judged with the unmutated run's tau and Csort). buckets: T9's buckets
    (profile.t9_buckets); doc_starts: T9's document starts (F4)."""
    th = th if th is not None else common.load_thresholds()
    t = th["G4F32"]
    decisive = float(t["f1_decisive_lead_nats"])
    sets = {}
    for s in t["sets"]:
        names = series["sets"][s]
        sets[s] = _f32_set(_cols(series, names), _csort(csort, names, series), decisive)
    t9 = series["sets"]["T9"]
    c9, cs9 = _cols(series, t9), _csort(csort, t9, series)
    per_text = {}
    for n in series["names"]:
        c = series["cols"][n]
        row = {"n": int(c["kl"].size), "mean_kl": _mean(c["kl"]), "max_kl": _max(c["kl"]),
               "n_top1_changes": int((~c["agree"]).sum()),
               "n_top1_changes_decisive": int((~c["agree"] & (c["lead"] >= decisive)).sum())}
        if c["S"] is not None:
            row["mean_S"] = _mean(c["S"])
        if c["shadow_dis"] is not None:
            row["shadow_disagree_rate"] = _ratio(float(c["shadow_dis"].sum()), c["shadow_dis"].size)
        if csort is not None:
            row.update(_csort_fields(csort["cols"][n]["kl"]))
        per_text[n] = row
    desc = {"per_text": per_text,
            "shadow_disagree_rate": {s: v.get("shadow_disagree_rate") for s, v in sets.items()},
            "csort_max": {s: v.get("csort_max") for s, v in sets.items()},
            "csort_max_share": {s: v.get("csort_max_share") for s, v in sets.items()}}
    if csort is not None and c9["S"] is not None:
        desc["F4"] = f4_windows(c9, cs9, (), doc_starts, series["chunk"], int(t["descriptive_f4_window"]), decisive)
    return {"sets": sets, "t9_buckets": t9_bucket_rows(c9, cs9, buckets), "descriptive": desc,
            "meta": {"kind": "g4_f32", "port": "K8 fp32 (make_fp32_port) forced onto I by ForcedTrunk",
                     "comparator": "R2F", "S": "KL(R2F||R3)", "estimator": KL_ESTIMATOR,
                     "csort": csort is not None, "chunk": series["chunk"], "num_layers": series["num_layers"],
                     "seconds": {"forced": series["seconds"], "csort": None if csort is None else csort["seconds"]}}}


def g4_f16_measured(series: dict, *, th: dict | None = None) -> dict:
    """rules.g4_f16's measured dict from a forced series of the K8 build as
    loaded (bf16) with R3: per set mean KL(R2F||port) and mean KL(R2F||R3), and
    the descriptive rows of DESIGN §3.8."""
    th = th if th is not None else common.load_thresholds()
    t = th["G4F16"]
    dec_lead = float(t["descriptive_decisive_lead_nats"])
    out = {}
    for s in t["sets"]:
        c = _cols(series, series["sets"][s])
        if c["S"] is None or c["r3_agree"] is None:
            raise HarnessError("G4-F16 needs R3 (S and R3's top-1)")
        kl, S = c["kl"], c["S"]
        dec = c["lead"] >= dec_lead
        row = {"n": int(kl.size), "mean_kl_port": _mean(kl), "mean_kl_r3": _mean(S),
               "port_over_r3": _ratio(_mean(kl), _mean(S)),
               "p99_kl_port": _quantile(kl, 0.99), "p99_kl_r3": _quantile(S, 0.99),
               "p99_ratio": _ratio(_quantile(kl, 0.99), _quantile(S, 0.99)),
               "max_kl_port": _max(kl), "n_decisive": int(dec.sum()),
               "decisive_miss_port": int((~c["agree"] & dec).sum()),
               "decisive_miss_r3": int((~c["r3_agree"] & dec).sum()),
               "n_sparse_spikes": int(((kl > t["descriptive_spike_port_kl_min"]) & (S < t["descriptive_spike_r3_kl_max"])).sum())}
        if c["shadow_dis"] is not None:
            dis = c["shadow_dis"]
            row.update({"shadow_pairs": int(dis.size), "shadow_disagree_port": int(dis.sum()),
                        "shadow_rate_port": _ratio(float(dis.sum()), dis.size)})
            if c["r3_shadow_dis"] is not None:
                r3d = c["r3_shadow_dis"]
                row.update({"shadow_disagree_r3": int(r3d.sum()), "shadow_rate_r3": _ratio(float(r3d.sum()), r3d.size)})
        out[s] = row
    out["meta"] = {"kind": "g4_f16", "port": "K8 as loaded (bf16) forced onto I by ForcedTrunk", "comparator": "R2F",
                   "calibrator": "R3 (KL(R2F||R3))", "estimator": KL_ESTIMATOR, "chunk": series["chunk"],
                   "num_layers": series["num_layers"], "seconds": series["seconds"],
                   "descriptive": {"spike": f"KL(R2F||port) > {t['descriptive_spike_port_kl_min']} and "
                                            f"KL(R2F||R3) < {t['descriptive_spike_r3_kl_max']} (G4-D5)",
                                   "decisive_lead_nats": dec_lead}}
    return out


# ---------------------------------------------------------------------------
# T9's document starts (F4, descriptive)
# ---------------------------------------------------------------------------


def t9_document_starts(ts, *, manifest: dict | None = None, sep_tokens: int | None = None, tok_dir=None) -> list:
    """T9's document starts after the first (gate/texts/MANIFEST.json's T9
    components, each followed by the separator's ids), as exp_036's
    diagnostics/gate1/stage3_g4.py doc_boundaries; [] unless ts.t9 is the
    manifest's T9 (tiny and synthetic sets have no documents)."""
    man = manifest if manifest is not None else common.read_json(common.TEXTS_DIR / "MANIFEST.json")["web"]["T9"]
    t9 = np.asarray(ts.t9).reshape(-1)
    if man.get("ids_sha256") != common.ids_sha256(t9):
        return []
    if sep_tokens is None:
        tok = common.load_raw_tokenizer(Path(tok_dir) if tok_dir is not None else common.tokenizer_dir())
        sep_tokens = len(tok.encode(man["separator"], add_special_tokens=False).ids)
    starts, p = [], 0
    for comp in man["components"]:
        if p >= t9.size:
            break
        starts.append(p)
        p += int(comp["n_tokens"]) + int(sep_tokens)
    return starts[1:]


# ---------------------------------------------------------------------------
# One check's measurement (what run_gate.py calls)
# ---------------------------------------------------------------------------


def g4_f32_run(model, ts, I, R2F, R3, chunk=2048, *, module, ref: dict | None = None, csort: bool = True,
               doc_starts=None, th: dict | None = None, with_series: bool = False, log=print):
    """G4-F32's measured dict for one model, which must already be the K8 fp32
    port (make_fp32_port). csort=True for the unmutated run (Csort and F4 are
    measured with it); False for a mutant. with_series: return (measured,
    series, csort series) instead."""
    if not is_fp32_port(model):
        raise HarnessError("G4-F32 needs the K8 fp32 port: call make_fp32_port(model) first")
    series = forced_series(model, ts, I, R2F, R3, chunk, module=module, ref=ref, log=log)
    cs = csort_series(model, ts, chunk, module=module, log=log) if csort else None
    if cs is not None and doc_starts is None:
        doc_starts = t9_document_starts(ts)
    m = g4_f32_measured(series, cs, buckets=ts.profile.t9_buckets, doc_starts=doc_starts or (), th=th)
    return (m, series, cs) if with_series else m


def g4_f16_run(model, ts, I, R2F, R3, chunk=2048, *, module, ref: dict | None = None, th: dict | None = None,
               with_series: bool = False, log=print):
    """G4-F16's measured dict for one model: the K8 build as loaded (bf16),
    never the fp32 port."""
    if is_fp32_port(model):
        raise HarnessError("G4-F16 runs the K8 build as loaded (bf16), not the fp32 port")
    if R3 is None:
        raise HarnessError("G4-F16 needs R3")
    series = forced_series(model, ts, I, R2F, R3, chunk, module=module, ref=ref, log=log)
    m = g4_f16_measured(series, th=th)
    return (m, series) if with_series else m


# ---------------------------------------------------------------------------
# The mutant loop over controls.json's lists (§4.2)
# ---------------------------------------------------------------------------


def port_loader(model_dir):
    """load(name) -> (model, module): the build at model_dir with port mutant
    `name`, as loaded (gate.common.load_port)."""
    def load(name: str):
        model, _, module = common.load_port(Path(model_dir), mutant=name)
        return model, module

    return load


def control_plan(check_ids, controls: dict | None = None) -> list:
    """The required mutants of `check_ids` from controls.json, each once:
    [{"mutant": id, "name", "runs": [{"check", "precision", "control", "leg"}]}]
    in mutant-id order; a mutant's bf16 runs come before its fp32 runs (the
    conversion is one way and in place)."""
    from gate import port_mutants

    controls = controls if controls is not None else rules.load_controls()
    plan = {}
    for cid in check_ids:
        req = rules.required_controls(cid, controls)
        if not req:
            raise ValueError(f"controls.json requires no control for {cid!r}")
        for c in req:
            mid, name = int(c["mutant"]), str(c["mutant_name"])
            if port_mutants.MUTANT_IDS.get(mid) != name:
                raise ValueError(f"control {c['id']}: mutant {mid} is {port_mutants.MUTANT_IDS.get(mid)!r} in "
                                 f"gate/port_mutants.py, not {name!r}")
            if c["precision"] not in PRECISIONS:
                raise ValueError(f"control {c['id']}: precision {c['precision']!r}; known: {PRECISIONS}")
            e = plan.setdefault(mid, {"mutant": mid, "name": name, "runs": []})
            e["runs"].append({"check": cid, "precision": c["precision"], "control": c["id"], "leg": c["leg"]})
    for e in plan.values():
        e["runs"].sort(key=lambda r: PRECISIONS.index(r["precision"]))
    return [plan[m] for m in sorted(plan)]


def mutant_loop(check_ids, load, measures: dict, *, controls: dict | None = None, to_fp32=None, release=None,
                log=print) -> dict:
    """{check: {mutant id: measured}} for controls.json's required mutants of
    `check_ids`. Each mutant is loaded once by load(name) -> (model, module)
    (the build as loaded, bf16), run through measures[check](model, module)
    for its bf16 checks, converted in place by to_fp32 (make_fp32_port) before
    its first fp32 check, then dropped; release() (gc and mx.clear_cache) runs
    before every load and after every mutant, so two mutants never coexist."""
    to_fp32 = to_fp32 or make_fp32_port
    release = release or release_memory
    check_ids = list(check_ids)
    absent = [c for c in check_ids if c not in measures]
    if absent:
        raise ValueError(f"no measure for {absent}")
    plan = control_plan(check_ids, controls)
    out = {c: {} for c in check_ids}
    for e in plan:
        release()
        log(f"[gate] control mutant {e['mutant']} ({e['name']}): "
            + ", ".join(f"{r['check']} {r['precision']}" for r in e["runs"]))
        model, module = load(e["name"])
        try:
            converted = False
            for r in e["runs"]:
                if r["precision"] == "fp32" and not converted:
                    to_fp32(model)
                    converted = True
                m = measures[r["check"]](model, module)
                if isinstance(m, dict):
                    m = {**m, "meta": {**(m.get("meta") or {}), "mutant": e["name"], "mutant_id": e["mutant"],
                                       "control": r["control"], "precision": r["precision"]}}
                out[r["check"]][e["mutant"]] = m
        finally:
            del model, module
            release()
    return out


def g4_mutants(load, ts, I, R2F, R3, chunk=2048, *, ref: dict | None = None, checks=G4_CONTROL_CHECKS,
               controls: dict | None = None, th: dict | None = None, log=print) -> dict:
    """The G4 control mutants (controls.json: G4-F16 mutant 6 in bf16; G4-F32
    mutants 1, 6 and 19 as the fp32 port), each forced onto I as the unmutated
    run is: {"g4_f16_K8": {6: measured}, "g4_f32_K8": {1: ..., 6: ..., 19: ...}}
    (only the checks asked). load: port_loader(K8 build dir) or equivalent. The
    reference profile is computed once and shared."""
    if ref is None:
        ref = reference_profile(text_layout(ts), R2F, R3, num_layers=len(I), log=log)
    known = {
        "g4_f16_K8": lambda model, module: g4_f16_run(model, ts, I, R2F, R3, chunk, module=module, ref=ref, th=th, log=log),
        "g4_f32_K8": lambda model, module: g4_f32_run(model, ts, I, R2F, R3, chunk, module=module, ref=ref,
                                                      csort=False, th=th, log=log),
    }
    unknown = [c for c in checks if c not in known]
    if unknown:
        raise ValueError(f"not a G4 forced check: {unknown}")
    return mutant_loop(checks, load, {c: known[c] for c in checks}, controls=controls, log=log)
