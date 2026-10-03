# SPDX-License-Identifier: MIT
"""Reference passes of the gate (phases 2 and 5; HYPOTHESIS Phase 0 "Reference").

Everything numerical here is the numpy reference's (reference/kolibri_ref.py:
layer_branches, forward_streams; reference/mutants.py); this module only
drives it, one layer at a time, and writes what the checks need:

  fp32 dump   per layer: h_in, r_attn, r_moe (item 29, before the residual
              adds), router logits, top-6 ids and the 6th-7th biased gap;
              then h_last, final_norm and fp32 logits. T1-T8 and T9 in one
              weight pass (independent sequences, each from position 0).
  emu stats   the bf16-emulated reference fed the fp32 reference's h_l, forced
              onto its experts and natural, per layer: the calibration of
              the bf16-mode G2 rows.
  mutant NLL  reference/mutants.py on T3 (G3).

Dumps live under $EXP036_WORK/ref/<reference tree sha>/<checkpoint
fingerprint>/<gate-text key>/ and are reused when all three match (record.json
is written last; a directory without it is incomplete and is rebuilt).
"""

from __future__ import annotations

import inspect
import shutil
import time
from pathlib import Path

import numpy as np

from gate import common


class CapabilityMissing(RuntimeError):
    """A reference capability the gate needs is not in reference/ (yet)."""


def _kr():
    from reference import kolibri_ref

    return kolibri_ref


class RefRunner:
    """A KolibriReference in one mode: "fp32" (the reference proper), "emu"
    (bf16 emulation, BUILD_SPEC §5.1b-3) or "dequant" (weights unpacked from
    a converted MLX directory, §5.1b-4)."""

    def __init__(self, model_dir: Path, mode: str = "fp32", dequant_dir: Path | None = None,
                 attn_chunk: int | None = None, cls=None):
        kr = _kr()
        cls = cls or kr.KolibriReference
        params = inspect.signature(cls.__init__).parameters
        kw = {}
        if attn_chunk is not None:
            kw["attn_chunk"] = attn_chunk
        if mode == "emu":
            if "emulate_bf16" not in params:
                raise CapabilityMissing("KolibriReference(emulate_bf16=True) (BUILD_SPEC §5.1b-3) is not in reference/ yet")
            kw["emulate_bf16"] = True
        elif mode == "dequant":
            if "dequant_dir" not in params:
                raise CapabilityMissing("KolibriReference(dequant_dir=...) (BUILD_SPEC §5.1b-4) is not in reference/ yet")
            kw["dequant_dir"] = str(dequant_dir)
        elif mode != "fp32":
            raise ValueError(mode)
        self.mode = mode
        self.ref = cls(str(model_dir), **kw)
        self.cfg = self.ref.cfg
        self.k = self.cfg.num_experts_per_tok
        if not hasattr(self.ref, "layer_branches"):
            raise CapabilityMissing("KolibriReference.layer_branches (BUILD_SPEC §5.1b-1/2) is not in reference/ yet")

    def branches(self, i: int, h: np.ndarray, segments=None, force_ids: np.ndarray | None = None) -> dict:
        """{r_attn, r_moe, h_out, logits, biased, top6} for layer i on h [N, H]
        (KolibriReference.layer_branches; force_ids fixes the selection)."""
        out = self.ref.layer_branches(i, h, segments, force_ids)
        info = out["info"]
        return {"r_attn": out["r_attn"], "r_moe": out["r_moe"], "h_out": out["h_out"],
                "logits": info["logits"], "biased": info["biased"], "top6": info.get("top6", info.get("ids"))}


def top_gap(biased: np.ndarray, k: int) -> np.ndarray:
    """k-th minus (k+1)-th biased score per row (fp32, exact by Sterbenz)."""
    if biased.shape[-1] <= k:
        return np.full(biased.shape[0], np.inf, dtype=np.float32)
    s = -np.sort(-np.asarray(biased, dtype=np.float32), axis=-1)
    return (s[:, k - 1] - s[:, k]).astype(np.float32)


def set_disagree(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per row: do the two id sets differ?"""
    return np.any(np.sort(a, axis=-1) != np.sort(b, axis=-1), axis=-1)


def segments_of(lengths) -> list:
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    return [(int(bounds[j]), int(bounds[j + 1])) for j in range(len(lengths))]


# ---------------------------------------------------------------------------
# Dump directory
# ---------------------------------------------------------------------------


class Dump:
    """Reader/writer of one dump directory (np.save files + record.json)."""

    def __init__(self, directory: Path):
        self.dir = Path(directory)
        self.files: dict = {}

    @property
    def complete(self) -> bool:
        return (self.dir / "record.json").is_file()

    def record(self) -> dict:
        return common.read_json(self.dir / "record.json")

    def save(self, name: str, arr: np.ndarray) -> None:
        path = self.dir / name
        np.save(path, arr)
        self.files[name] = {"sha256": common.sha256_file(path), "bytes": path.stat().st_size,
                            "shape": list(arr.shape), "dtype": str(arr.dtype)}

    def load(self, name: str, mmap: bool = True) -> np.ndarray:
        return np.load(self.dir / name, mmap_mode="r" if mmap else None)

    def layer(self, i: int, what: str, mmap: bool = True) -> np.ndarray:
        return self.load(f"layer{i:02d}.{what}.npy", mmap)

    def finish(self, record: dict) -> dict:
        record = dict(record, files=self.files)
        common.write_json(self.dir / "record.json", record)
        return record


def cache_dir(kind: str, text_key: str, ckpt_dir: Path, work: Path | None = None) -> Path:
    work = Path(work or common.work_dir())
    return (work / "ref" / common.reference_tree_sha256()[:16] / common.checkpoint_fingerprint(ckpt_dir)[:16]
            / text_key[:16] / kind)


def _keys(ckpt_dir: Path, text_key: str) -> dict:
    return {"reference_tree_sha256": common.reference_tree_sha256(),
            "checkpoint_fingerprint": common.checkpoint_fingerprint(ckpt_dir),
            "text_key": text_key}


def reusable(dump: Dump, ckpt_dir: Path, text_key: str) -> bool:
    if not dump.complete:
        return False
    rec = dump.record()
    return all(rec.get(k) == v for k, v in _keys(ckpt_dir, text_key).items())


def fp32_dump(ckpt_dir: Path, seqs: list, out_dir: Path, text_key: str, attn_chunk: int | None = None,
              log=print) -> dict:
    """The fp32 reference pass over `seqs` (one pass over the weights)."""
    dump = Dump(out_dir)
    if reusable(dump, ckpt_dir, text_key):
        log(f"[gate] reference fp32 dump reused: {out_dir.name} ({common.redact_path(out_dir)})")
        return dump.record() | {"reused": True}
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    t0, utc0 = time.time(), common.utc_iso()
    rr = RefRunner(ckpt_dir, "fp32", attn_chunk=attn_chunk)
    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    segments = segments_of(lengths)
    ids = np.concatenate([np.asarray(s, dtype=np.int64) for s in seqs])
    h = rr.ref.embed(ids)
    layer_seconds = []
    for i in range(rr.cfg.num_hidden_layers):
        t = time.time()
        br = rr.branches(i, h, segments)
        dump.save(f"layer{i:02d}.h_in.npy", h)
        dump.save(f"layer{i:02d}.r_attn.npy", br["r_attn"].astype(np.float32))
        dump.save(f"layer{i:02d}.r_moe.npy", br["r_moe"].astype(np.float32))
        dump.save(f"layer{i:02d}.router_logits.npy", br["logits"].astype(np.float32))
        dump.save(f"layer{i:02d}.top6.npy", np.asarray(br["top6"], dtype=np.int64))
        dump.save(f"layer{i:02d}.top6_gap.npy", top_gap(br["biased"], rr.k))
        h = br["h_out"]
        layer_seconds.append(time.time() - t)
        log(f"[gate] ref fp32 layer {i + 1}/{rr.cfg.num_hidden_layers} {layer_seconds[-1]:.1f}s "
            f"RSS {common.peak_rss_bytes() / 2**30:.1f} GiB")
    final = rr.ref.final_norm(h)
    dump.save("h_last.npy", h)
    dump.save("final_norm.npy", final)
    dump.save("logits.npy", rr.ref.lm_head(final))
    dump.save("ids.npy", ids)
    dump.save("seq_lengths.npy", lengths)
    return dump.finish({**_keys(ckpt_dir, text_key), "kind": "fp32", "ref_version": _kr().REF_VERSION,
                        "num_layers": rr.cfg.num_hidden_layers, "num_tokens": int(ids.size),
                        "seq_lengths": lengths.tolist(), "layer_seconds": layer_seconds,
                        "t_start": utc0, "t_end": common.utc_iso(), "wall_s": time.time() - t0,
                        "peak_rss_bytes": common.peak_rss_bytes(),
                        "branches": "KolibriReference.layer_branches"})


def emu_stats(ckpt_dir: Path, dump: Dump, n_rows: int, lengths, attn_chunk: int | None = None, log=print) -> dict:
    """Per layer: the emulated reference vs the fp32 reference on the same
    input h_l (rows [0, n_rows), i.e. T1-T8), forced and natural."""
    t0, utc0 = time.time(), common.utc_iso()
    emu = RefRunner(ckpt_dir, "emu", attn_chunk=attn_chunk)
    segments = segments_of(lengths)
    layers = []
    for i in range(emu.cfg.num_hidden_layers):
        h = np.array(dump.layer(i, "h_in")[:n_rows])
        top6 = np.array(dump.layer(i, "top6")[:n_rows])
        r_attn = dump.layer(i, "r_attn")[:n_rows]
        r_moe = dump.layer(i, "r_moe")[:n_rows]
        forced = emu.branches(i, h, segments, force_ids=top6)
        natural = emu.branches(i, h, segments)
        dis = set_disagree(np.asarray(natural["top6"]), top6)
        layers.append({
            "layer": i,
            "attn": common.stats(common.rel_err_rows(forced["r_attn"], r_attn)),
            "moe": common.stats(common.rel_err_rows(forced["r_moe"], r_moe)),
            "natural_disagree_frac": float(dis.mean()),
        })
        log(f"[gate] ref emu layer {i + 1}/{emu.cfg.num_hidden_layers}")
    return {"layers": layers, "t_start": utc0, "t_end": common.utc_iso(), "wall_s": time.time() - t0}


def logprobs_rows(logits: np.ndarray) -> np.ndarray:
    return common.log_softmax64(logits).astype(np.float64)


def next_token_nll(logits: np.ndarray, ids) -> np.ndarray:
    """NLL (nats) of ids[t+1] given positions 0..t, for t = 0..T-2."""
    ids = np.asarray(ids)
    out = np.empty(ids.size - 1, dtype=np.float64)
    for a in range(0, ids.size - 1, 256):
        b = min(ids.size - 1, a + 256)
        lp = common.log_softmax64(np.asarray(logits[a:b]))
        out[a:b] = -lp[np.arange(b - a), ids[a + 1:b + 1]]
    return out


def mutant_nll(ckpt_dir: Path, ids, attn_chunk: int | None = None) -> tuple[dict, str]:
    """({"ref": nll, <mutant>: nll, ...}, path): next-token NLL on one sequence
    for the reference and each of reference/mutants.py's MUTANTS, as parallel
    streams of one layer-streamed pass (KolibriReference.forward_streams), so
    k logit matrices are never held at once."""
    try:
        from reference.mutants import MUTANTS
    except ImportError as e:
        raise CapabilityMissing(f"reference/mutants.py (BUILD_SPEC §5.2) is not importable: {e}") from e
    kr = _kr()
    if not hasattr(kr.KolibriReference, "forward_streams"):
        raise CapabilityMissing("KolibriReference.forward_streams (BUILD_SPEC §5.1b-6) is not in reference/ yet")
    ids = np.asarray(ids, dtype=np.int64)
    ref = kr.KolibriReference(str(ckpt_dir), **({"attn_chunk": attn_chunk} if attn_chunk else {}))
    out = ref.forward_streams(ids, MUTANTS, include_base=True, on_logits=lambda name, logits: next_token_nll(logits, ids))
    return {k: v for k, v in out.items() if k != "_lengths"}, "KolibriReference.forward_streams (one pass, base stream 'ref')"
