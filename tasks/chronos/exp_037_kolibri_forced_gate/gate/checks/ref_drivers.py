# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06; build task W3).
"""Reference drivers of exp_037 (DESIGN §3.0, §2.9, §3.1 P5a, §3.13).

Everything numerical is the numpy reference's (reference/kolibri_ref.py,
unedited); this module only drives it, one layer at a time, and measures.

  forced_ref_pass  R2F (mode "fp32") and R3 (mode "emu"): KolibriReference on
                   the BF16 config with the dequantised K8 weights
                   (dequant_dir), every layer forced onto I[l] through
                   layer_forward(l, h, segments, force_ids=I[l]), over the pack
                   with its segments. Stores per layer the shadow (top-6 of
                   info["biased"], by descending biased score, ties to the lower
                   id, as route_ref orders) and the 6th-7th biased gap
                   (ref_pass.top_gap), then fp32 logits [N, V]. Keyed and
                   reused under work37_dir()/ref_forced/<key16>/<mode>/.
  r1_rows_pass     R1 (fp32, BF16 weights, natural routing) on arbitrary
                   sequences, layer by layer, with the final norm and lm_head
                   applied only to the wanted rows (phase 6; the [N, V] matrix
                   of every anchor sequence would be about 57 GB).
  p5a              P5a: layer_forward on R1's own pack, forced onto the dump's
                   top-6, reproduces the dump's next h_in bitwise.

I (DESIGN §3.0): I[l, n] = the ascending sort of R1's layerNN.top6.npy[n]
(i_from_r1). Forcing ids never come from the port.

Writers: every directory written here must resolve under
gate.common.work37_dir() ($EXP036_WORK/exp037); anything else is refused
before a byte is written or removed (DESIGN §2.9).
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import numpy as np

from gate import common
from gate.checks import ref_pass
from gate.harness import HarnessError, bitwise_equal

FORCED_SUBDIR = "ref_forced"
MODES = ("fp32", "emu")  # R2F, R3
DRIVER = "KolibriReference(bf16_dir, dequant_dir=K8[, emulate_bf16=True]).layer_forward(l, h, segments, force_ids=I[l])"
I_SHA256_RULE = ("sha256 over the lines 'layerNN.top6.npy\\t<sha256 of that file>\\n' of the R1 dump, layers in "
                 "order (the files that define I)")
KEY_RULE = ("sha256 of gate.common.dumps({reference_tree_sha256, k8_shards: sorted [shard name, sha256] pairs, "
            "text_key, i_sha256, mode})")
P5A_LAYERS = (0, 4)  # one sliding, one full (DESIGN §3.1 P5a)


def _kr():
    from reference import kolibri_ref

    return kolibri_ref


# ---------------------------------------------------------------------------
# Paths and keys
# ---------------------------------------------------------------------------


def require_under_work37(path) -> Path:
    """`path`, resolved, if it lies strictly inside work37_dir(); else ValueError.
    The guard every writer here calls before it creates or removes anything."""
    root = common.work37_dir().expanduser().resolve()
    p = Path(path).expanduser().resolve()
    if p == root or root not in p.parents:
        raise ValueError(f"refusing to write {common.redact_path(p)}: exp_037 writers stay under "
                         f"{common.redact_path(root)} (DESIGN §2.9)")
    return p


def i_from_r1(r1_dir, num_layers: int | None = None) -> tuple[np.ndarray, str]:
    """(I [L, N, k] int64, i_sha256) from an R1 dump: each layer's top6.npy sorted
    ascending along k. i_sha256 follows I_SHA256_RULE; when the dump's
    record.json lists a file's sha256, the file must still hash to it."""
    d = Path(r1_dir)
    rec_files = {}
    if (d / "record.json").is_file():
        rec = common.read_json(d / "record.json")
        rec_files = rec.get("files", {})
        num_layers = num_layers if num_layers is not None else rec.get("num_layers")
    if num_layers is None:
        num_layers = len(sorted(d.glob("layer*.top6.npy")))
    if num_layers <= 0:
        raise HarnessError(f"{common.redact_path(d)}: no layerNN.top6.npy files")
    lines, layers = [], []
    for i in range(num_layers):
        name = f"layer{i:02d}.top6.npy"
        sha = common.sha256_file(d / name)
        want = rec_files.get(name, {}).get("sha256")
        if want is not None and want != sha:
            raise HarnessError(f"{name}: sha256 {sha[:16]} differs from the dump record's {want[:16]}")
        lines.append(f"{name}\t{sha}\n")
        layers.append(np.sort(np.load(d / name).astype(np.int64), axis=-1))
    return np.stack(layers), common.sha256_bytes("".join(lines).encode("utf-8"))


def build_shards(build_dir) -> list:
    """Sorted [[shard name, sha256], ...] of a build's model*.safetensors files.
    The gate may pass P4's verified values to forced_dump_key instead."""
    files = sorted(Path(build_dir).glob("model*.safetensors"), key=lambda p: p.name.encode("utf-8"))
    if not files:
        raise HarnessError(f"{common.redact_path(build_dir)}: no model*.safetensors")
    return [[p.name, common.sha256_file(p)] for p in files]


def forced_dump_key(*, reference_tree_sha256: str, k8_shards, text_key: str, i_sha256: str, mode: str) -> dict:
    """The forced-dump key (DESIGN §2.9; KEY_RULE): {"components": ..., "sha256": hex}.
    k8_shards: [(name, sha256), ...] or {name: sha256}."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}; known: {MODES}")
    pairs = k8_shards.items() if isinstance(k8_shards, dict) else k8_shards
    shards = sorted(([str(n), str(s)] for n, s in pairs), key=lambda t: t[0].encode("utf-8"))
    components = {"reference_tree_sha256": str(reference_tree_sha256), "k8_shards": shards,
                  "text_key": str(text_key), "i_sha256": str(i_sha256), "mode": mode}
    return {"components": components, "sha256": common.sha256_bytes(common.dumps(components).encode("utf-8"))}


def forced_dump_dir(key: dict, work: Path | None = None) -> Path:
    """work37_dir()/ref_forced/<key sha256[:16]>/<mode>/ (work: a work37 directory
    to use instead, for tests)."""
    base = Path(work) if work is not None else common.work37_dir()
    return base / FORCED_SUBDIR / key["sha256"][:16] / key["components"]["mode"]


def derived_key(dequant_dir, pack_ids, segments, I, mode: str) -> dict:
    """A key from the inputs alone, when the caller has no R1 dump or gate-text
    key to name (tiny tests, diagnostics): the text key is the sha256 of the pack
    ids and segment lengths, i_sha256 the sha256 of I's int64 bytes."""
    lengths = [int(b - a) for a, b in segments]
    text = common.sha256_bytes(f"pack|{common.ids_sha256(np.asarray(pack_ids).reshape(-1))}|{lengths}".encode())
    i_sha = common.sha256_bytes(np.ascontiguousarray(np.asarray(I, dtype=np.int64)).tobytes())
    return forced_dump_key(reference_tree_sha256=common.reference_tree_sha256(), k8_shards=build_shards(dequant_dir),
                           text_key="derived:" + text, i_sha256="derived:" + i_sha, mode=mode)


# ---------------------------------------------------------------------------
# R2F / R3
# ---------------------------------------------------------------------------


def _check_segments(segments, n: int) -> list:
    segs = [(int(a), int(b)) for a, b in segments]
    pos = 0
    for a, b in segs:
        if a != pos or b <= a:
            raise ValueError(f"segments must tile [0, {n}) in order; got {(a, b)} after {pos}")
        pos = b
    if pos != n:
        raise ValueError(f"segments end at {pos}; the pack has {n} rows")
    return segs


def shadow_top(biased: np.ndarray, k: int) -> np.ndarray:
    """Top-k of the biased scores per row, by descending score, ties to the lower
    id (route_ref's order): the reference's own selection on these scores."""
    return np.argsort(-np.asarray(biased, dtype=np.float32), axis=-1, kind="stable")[:, :k].astype(np.int64)


def _complete(d: Path, rec: dict) -> bool:
    for name, entry in rec.get("files", {}).items():
        p = d / name
        if not p.is_file() or p.stat().st_size != entry.get("bytes"):
            return False
    return True


def forced_ref_pass(bf16_dir, dequant_dir, pack_ids, segments, I, mode: str, out_dir, *, key: dict | None = None,
                    attn_chunk: int | None = None, log=print) -> dict:
    """R2F (mode "fp32") or R3 (mode "emu") over the pack (DESIGN §3.0).

    bf16_dir: the BF16 checkpoint (config); dequant_dir: the K8 build whose
    dequantised weights the reference reads; pack_ids [N]; segments [(a, b)]
    tiling [0, N), each an independent sequence from position 0; I [L, N, k],
    the forcing ids (ascending; i_from_r1); out_dir: under work37_dir(),
    normally forced_dump_dir(key). key: forced_dump_key(...) (the gate's: R1's
    i_sha256 and the gate-text key); None derives one from the inputs
    (derived_key).

    Writes into out_dir: layerNN.shadow6.npy [N, k] int64, layerNN.gap.npy [N]
    fp32 (6th-7th biased gap), logits.npy [N, V] fp32, ids.npy, seq_lengths.npy,
    and record.json last (a directory without it is incomplete and rebuilt). A
    complete directory whose record carries the same key is reused. Returns the
    record (plus "reused": True when reused)."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r}; known: {MODES}")
    out = require_under_work37(out_dir)
    pack_ids = np.asarray(pack_ids, dtype=np.int64).reshape(-1)
    segs = _check_segments(segments, pack_ids.size)
    kr = _kr()
    cfg = kr.load_config(str(bf16_dir))
    L, k = cfg.num_hidden_layers, cfg.num_experts_per_tok
    if len(I) != L:
        raise HarnessError(f"I has {len(I)} layers, the model {L}")
    for i in range(L):
        if tuple(np.shape(I[i])) != (pack_ids.size, k):
            raise HarnessError(f"I[{i}] has shape {tuple(np.shape(I[i]))}, expected {(pack_ids.size, k)}")
    if key is None:
        key = derived_key(dequant_dir, pack_ids, segs, I, mode)
    if key["components"]["mode"] != mode:
        raise ValueError(f"key is for mode {key['components']['mode']!r}, the pass for {mode!r}")
    if (out / "record.json").is_file():
        rec = common.read_json(out / "record.json")
        if rec.get("key") == key and _complete(out, rec):
            log(f"[gate] forced reference {mode} reused: {common.redact_path(out)}")
            return rec | {"reused": True}
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    kw = {"dequant_dir": str(dequant_dir), "emulate_bf16": mode == "emu"}
    if attn_chunk is not None:
        kw["attn_chunk"] = attn_chunk
    ref = kr.KolibriReference(str(bf16_dir), **kw)
    t0, utc0 = time.time(), common.utc_iso()
    dump = ref_pass.Dump(out)
    lengths = np.array([b - a for a, b in segs], dtype=np.int64)
    h = ref.embed(pack_ids)
    layer_seconds = []
    for i in range(L):
        t = time.time()
        force = np.asarray(I[i], dtype=np.int64)
        h = ref.layer_forward(i, h, segs, force_ids=force)
        info = ref.last_layer_info
        if not info.get("forced") or not np.array_equal(np.asarray(info["top6"]), force):
            raise HarnessError(f"layer {i}: the reference did not use the forced ids")
        dump.save(f"layer{i:02d}.shadow6.npy", shadow_top(info["biased"], k))
        dump.save(f"layer{i:02d}.gap.npy", ref_pass.top_gap(info["biased"], k))
        layer_seconds.append(time.time() - t)
        log(f"[gate] ref forced {mode} layer {i + 1}/{L} {layer_seconds[-1]:.1f}s "
            f"RSS {common.peak_rss_bytes() / 2**30:.1f} GiB")
    dump.save("logits.npy", ref.lm_head(ref.final_norm(h)))
    dump.save("ids.npy", pack_ids)
    dump.save("seq_lengths.npy", lengths)
    return dump.finish({"kind": f"forced_{mode}", "check": "R2F" if mode == "fp32" else "R3", "mode": mode,
                        "key": key, "key_rule": KEY_RULE, "i_sha256_rule": I_SHA256_RULE, "driver": DRIVER,
                        "ref_version": kr.REF_VERSION, "num_layers": L, "num_tokens": int(pack_ids.size),
                        "seq_lengths": lengths.tolist(), "shadow_order": "descending biased score, ties to the lower id",
                        "bf16_dir": common.redact_path(bf16_dir), "dequant_dir": common.redact_path(dequant_dir),
                        "attn_chunk": attn_chunk, "layer_seconds": layer_seconds, "t_start": utc0,
                        "t_end": common.utc_iso(), "wall_s": time.time() - t0,
                        "peak_rss_bytes": common.peak_rss_bytes()})


class ForcedDump:
    """Reader of one forced_ref_pass directory."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self._dump = ref_pass.Dump(self.dir)

    @property
    def complete(self) -> bool:
        return self._dump.complete

    def record(self) -> dict:
        return self._dump.record()

    def logits(self, mmap: bool = True) -> np.ndarray:
        return self._dump.load("logits.npy", mmap)

    def shadow(self, i: int, mmap: bool = True) -> np.ndarray:
        return self._dump.layer(i, "shadow6", mmap)

    def gap(self, i: int, mmap: bool = True) -> np.ndarray:
        return self._dump.layer(i, "gap", mmap)


# ---------------------------------------------------------------------------
# R1 at wanted rows (phase 6)
# ---------------------------------------------------------------------------


def _packed_rows(rows_wanted, lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(packed rows, sequence index, position) from per-sequence positions (a list
    with one array per sequence) or from packed row indices (one 1-D int array)."""
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    if isinstance(rows_wanted, np.ndarray) and rows_wanted.ndim == 1 and np.issubdtype(rows_wanted.dtype, np.integer):
        rows = rows_wanted.astype(np.int64)
        if rows.size and (rows.min() < 0 or rows.max() >= bounds[-1]):
            raise ValueError(f"packed rows outside [0, {bounds[-1]})")
        seq = np.searchsorted(bounds, rows, side="right") - 1
        return rows, seq, rows - bounds[seq]
    per = list(rows_wanted)
    if len(per) != lengths.size:
        raise ValueError(f"rows_wanted has {len(per)} entries for {lengths.size} sequences")
    rows, seq, pos = [], [], []
    for s, p in enumerate(per):
        p = np.asarray(p, dtype=np.int64).reshape(-1)
        if p.size and (p.min() < 0 or p.max() >= lengths[s]):
            raise ValueError(f"sequence {s}: positions outside [0, {lengths[s]})")
        rows.append(bounds[s] + p)
        seq.append(np.full(p.size, s, dtype=np.int64))
        pos.append(p)
    return np.concatenate(rows), np.concatenate(seq), np.concatenate(pos)


def r1_rows_pass(bf16_dir, seqs, rows_wanted, *, attn_chunk: int | None = None, out_dir=None, log=print) -> dict:
    """R1 (fp32 reference, BF16 weights, natural routing) on independent
    sequences `seqs` (each from position 0, packed as KolibriReference packs
    them), layer by layer; the final norm and lm_head run only on the wanted
    rows. rows_wanted: one position array per sequence, or packed row indices.

    Returns {"logits": [n, V] fp32 (rows in the order asked), "rows": packed
    rows [n], "seq_index": [n], "positions": [n], "seq_lengths": [S],
    "record": {...}}. With out_dir (under work37_dir()), also writes
    logits.npy, rows.npy and record.json there."""
    out = require_under_work37(out_dir) if out_dir is not None else None
    kr = _kr()
    ref = kr.KolibriReference(str(bf16_dir), **({"attn_chunk": attn_chunk} if attn_chunk is not None else {}))
    vocab = ref.cfg.vocab_size
    seqs = [np.asarray(s, dtype=np.int64).reshape(-1) for s in seqs]
    if not seqs or any(s.size == 0 for s in seqs):
        raise ValueError("r1_rows_pass needs non-empty sequences")
    pack = np.concatenate(seqs)
    if pack.min() < 0 or pack.max() >= vocab:
        raise ValueError(f"token id outside [0, {vocab})")
    lengths = np.array([s.size for s in seqs], dtype=np.int64)
    segs = ref_pass.segments_of(lengths)
    rows, seq_index, positions = _packed_rows(rows_wanted, lengths)
    t0, utc0 = time.time(), common.utc_iso()
    h = ref.embed(pack)
    for i in range(ref.cfg.num_hidden_layers):
        h = ref.layer_forward(i, h, segs)
        log(f"[gate] R1 rows layer {i + 1}/{ref.cfg.num_hidden_layers}")
    logits = ref.lm_head(ref.final_norm(h[rows])) if rows.size else np.zeros((0, vocab), dtype=np.float32)
    record = {"kind": "r1_rows", "driver": "KolibriReference(bf16_dir).layer_forward, natural; "
                                          "final_norm and lm_head on the wanted rows only",
              "ref_version": kr.REF_VERSION, "num_layers": ref.cfg.num_hidden_layers, "num_tokens": int(pack.size),
              "num_rows": int(rows.size), "seq_lengths": lengths.tolist(), "pack_ids_sha256": common.ids_sha256(pack),
              "bf16_dir": common.redact_path(bf16_dir), "attn_chunk": attn_chunk, "t_start": utc0,
              "t_end": common.utc_iso(), "wall_s": time.time() - t0, "peak_rss_bytes": common.peak_rss_bytes()}
    if out is not None:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        dump = ref_pass.Dump(out)
        dump.save("logits.npy", logits)
        dump.save("rows.npy", rows)
        record = dump.finish(record)
    return {"logits": logits, "rows": rows, "seq_index": seq_index, "positions": positions,
            "seq_lengths": lengths, "record": record}


# ---------------------------------------------------------------------------
# P5a
# ---------------------------------------------------------------------------


def p5a(bf16_dir, r1_dir, layers=P5A_LAYERS, *, attn_chunk: int | None = None, log=print) -> dict:
    """P5a (DESIGN §3.1): for each layer l in `layers`, layer_forward(l, h_in_l,
    segments, force_ids=top6_l) on R1's full pack reproduces the dump's
    h_in_{l+1} (h_last.npy after the last layer) bitwise. attn_chunk must be
    the one R1 was made with (the gate's: None). Returns {ok, values, reason}."""
    d = ref_pass.Dump(r1_dir)
    if not d.complete:
        raise HarnessError(f"{common.redact_path(r1_dir)}: incomplete R1 dump (no record.json)")
    rec = d.record()
    lengths = np.asarray(d.load("seq_lengths.npy", mmap=False), dtype=np.int64)
    segs = ref_pass.segments_of(lengths)
    kr = _kr()
    ref = kr.KolibriReference(str(bf16_dir), **({"attn_chunk": attn_chunk} if attn_chunk is not None else {}))
    L = int(rec.get("num_layers", ref.cfg.num_hidden_layers))
    values = {}
    for i in layers:
        if not 0 <= i < L:
            raise ValueError(f"P5a layer {i} outside [0, {L})")
        t = time.time()
        h = np.array(d.layer(i, "h_in"))
        top6 = np.array(d.layer(i, "top6"))
        got = ref.layer_forward(i, h, segs, force_ids=top6)
        want = np.array(d.layer(i + 1, "h_in")) if i + 1 < L else np.array(d.load("h_last.npy"))
        equal = bitwise_equal(got.astype(np.float32, copy=False), want)
        diff = np.abs(got.astype(np.float64) - want.astype(np.float64)) if got.shape == want.shape else None
        values[f"layer{i:02d}"] = {"layer": int(i), "layer_type": ref.cfg.layer_types[i], "n_rows": int(h.shape[0]),
                                   "bitwise_equal": bool(equal),
                                   "max_abs_diff": float(diff.max()) if diff is not None and diff.size else None,
                                   "rows_differing": int(np.any(got != want, axis=-1).sum())
                                   if got.shape == want.shape else None,
                                   "seconds": time.time() - t}
        log(f"[gate] P5a layer {i}: {'bitwise' if equal else 'DIFFERS'}")
    bad = [k for k, v in values.items() if not v["bitwise_equal"]]
    return {"ok": not bad, "values": {"layers": values, "dump_record_sha256": common.sha256_file(Path(r1_dir) / "record.json")},
            "reason": None if not bad else f"reference forced onto R1's own top-6 does not reproduce the dump at {bad}"}
