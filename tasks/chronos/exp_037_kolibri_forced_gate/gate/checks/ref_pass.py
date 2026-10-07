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

exp_037 (DESIGN §2.9, §3.5; build task W5a):
* The emulation statistics gain, per layer on T1-T8, sigma_emu,l = RMS(emu -
  R1 router logits), the per-pair z_emu = R1's 6th-7th biased gap / sigma_emu,l
  at every pair where the emulation's natural selection differs from R1's,
  and the flagged pair's flags (layer 20, T4 position 19: whether the
  emulation disagrees there, z_emu, sigma_t(emu)). M_emu is the maximum z_emu
  over layers and pairs: the calibrator of leg (p) of G2 bf16 natural
  selection, which gate/rules.py judges. natural_vs_r1 is the one per-pair z
  computation, shared with the port side (g2_layers.bf16_selection).
* emu_stats_exp037 keeps them in emu_stats_exp037.json under work37_dir()
  (emu_stats_path), reused only for the same R1 dump record and settings.
* A dump outside work37_dir() is only ever read (R1 under its registered key
  is exp_036's dump): fp32_dump creates, replaces or removes a directory only
  under work37_dir() (DESIGN §2.9: no writer, fp32_dump's rmtree included,
  targets a path outside it).
Nothing here judges: no verdict and no "pass" (gate/rules.py decides).
"""

from __future__ import annotations

import inspect
import shutil
import time
from pathlib import Path

import numpy as np

from gate import common

T18 = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")  # the pack order of rows [0, 8 x text_len)

# exp_036's flagged pair (DESIGN §3.5 "Descriptive"; dg1 stage 2): the layer-20
# bf16 disagreement with the largest gap / sigma_l (6.078), at T4 position 19.
FLAGGED_PAIR = {"layer": 20, "text": "T4", "pos": 19}

EMU_STATS_NAME = "emu_stats_exp037.json"
EMU_STATS_VERSION = "exp037-emu-stats-1"
Z_RULE = ("z = R1's 6th-7th biased gap / sigma_l at each (token, layer) pair where the natural selection's top-6 "
          "set differs from R1's; sigma_l = RMS over the T1-T8 rows and all experts of (router logits - R1 router "
          "logits) in layer l, float64 (DESIGN §3.5); sigma_t = the same RMS over the experts of one token; with "
          "sigma_l = 0 a pair's z is +inf if its gap > 0, else 0")


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
# Natural selection against R1: the per-pair z (DESIGN §3.5; shared by the port
# side, g2_layers.bf16_selection, and the emulation side, emu_stats)
# ---------------------------------------------------------------------------


def text_pos(row: int, text_len: int) -> tuple[str, int]:
    """(text id, position) of a T1-T8 pack row."""
    return T18[int(row) // int(text_len)], int(row) % int(text_len)


def flagged_site(n_layers: int, text_len: int) -> dict:
    """The flagged pair's layer, text, position and T1-T8 pack row.

    With layer 20 present this is the registered site. A tiny checkpoint has no
    layer 20; it gets a stand-in at its last layer, same text and position
    ("stand_in": true), so the report's code runs on tiny."""
    reg = FLAGGED_PAIR
    stand_in = not (reg["layer"] < n_layers and reg["pos"] < text_len)
    layer = min(reg["layer"], n_layers - 1) if stand_in else reg["layer"]
    pos = min(reg["pos"], text_len - 1) if stand_in else reg["pos"]
    return {"layer": int(layer), "text": reg["text"], "pos": int(pos),
            "row": T18.index(reg["text"]) * int(text_len) + int(pos), "stand_in": bool(stand_in)}


def _z(gaps, sigma: float) -> np.ndarray:
    gaps = np.asarray(gaps, dtype=np.float64)
    if sigma > 0:
        return gaps / sigma
    return np.where(gaps > 0, np.inf, 0.0)


def natural_vs_r1(sel_ids, ref_top6, ref_gap, logits, ref_logits, at_row: int | None = None) -> dict:
    """One layer's natural selection against R1's on the same rows, with the
    per-pair z (Z_RULE). sel_ids [N, k] (any leading shape), ref_top6 [N, k],
    ref_gap [N] (R1's 6th-7th biased gap), logits and ref_logits [N, E] (the
    selection's and R1's router logits).

    Returns {"n", "n_disagree", "sigma" (sigma_l), "max_gap_over_sigma" (the
    layer's maximum z; 0.0 without a disagreement)} and the arrays "_dis" [N]
    bool, "_rows" (the disagreeing rows, ascending), "_z" (their z, aligned)
    and "_gaps" (their gaps); with at_row, also "_at": that row's pair
    (whether the sets differ, gap, z, sigma_l, sigma_t, both sets, the swap)."""
    ref_top6 = np.asarray(ref_top6)
    k = ref_top6.shape[-1]
    ref_top6 = ref_top6.reshape(-1, k)
    sel = np.asarray(sel_ids).reshape(-1, k)
    dis = set_disagree(sel, ref_top6)
    ref64 = np.asarray(ref_logits, dtype=np.float64).reshape(dis.size, -1)
    diff = np.asarray(logits, dtype=np.float64).reshape(ref64.shape) - ref64
    sigma = float(np.sqrt(np.mean(diff * diff)))
    gap_all = np.asarray(ref_gap, dtype=np.float64).reshape(-1)
    rows = np.flatnonzero(dis)
    gaps = gap_all[rows]
    z = _z(gaps, sigma)
    out = {"n": int(dis.size), "n_disagree": int(rows.size), "sigma": sigma,
           "max_gap_over_sigma": float(z.max()) if z.size else 0.0,
           "_dis": dis, "_rows": rows, "_z": z, "_gaps": gaps}
    if at_row is not None:
        r = int(at_row)
        ref_set, sel_set = sorted(int(x) for x in ref_top6[r]), sorted(int(x) for x in sel[r])
        out["_at"] = {"row": r, "disagrees": bool(dis[r]), "gap": float(gap_all[r]),
                      "z": float(_z(gap_all[r:r + 1], sigma)[0]), "sigma_l": sigma,
                      "sigma_t": float(np.sqrt(np.mean(diff[r] * diff[r]))),
                      "ref_top6": ref_set, "top6": sel_set,
                      "dropped": sorted(set(ref_set) - set(sel_set)), "added": sorted(set(sel_set) - set(ref_set))}
    return out


def z_record(sel: dict, text_len: int, gap_sigma: float) -> dict:
    """The JSON part of natural_vs_r1's result: counts, sigma_l, the maximum z
    and where it is, the descriptive count of pairs with gap >= gap_sigma x
    sigma_l (exp_036's retired 6-sigma rule, counted as exp_036 counted it),
    and every pair's z ("z_pairs": rows ascending, z aligned)."""
    rows, z = sel["_rows"], sel["_z"]
    at = None
    if z.size:
        r = int(rows[int(np.argmax(z))])
        tid, pos = text_pos(r, text_len)
        at = {"row": r, "text": tid, "pos": pos}
    return {"n": sel["n"], "n_disagree": sel["n_disagree"], "sigma": sel["sigma"],
            "max_gap_over_sigma": sel["max_gap_over_sigma"], "max_at": at,
            "n_beyond_6sigma": int(np.sum(sel["_gaps"] >= gap_sigma * sel["sigma"])),
            "z_pairs": {"rows": [int(x) for x in rows], "z": [float(x) for x in z]}}


def gap_sigma_default() -> float:
    """thresholds.json G2 bf16_selection_gap_sigma (6.0; a descriptive count in exp_037)."""
    return float(common.load_thresholds()["G2"]["bf16_selection_gap_sigma"])


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


def _require_under_work37(path) -> Path:
    """The writer guard of DESIGN §2.9 (gate/checks/ref_drivers.py's, imported
    lazily: ref_drivers imports this module)."""
    from gate.checks.ref_drivers import require_under_work37

    return require_under_work37(path)


def fp32_dump(ckpt_dir: Path, seqs: list, out_dir: Path, text_key: str, attn_chunk: int | None = None,
              log=print) -> dict:
    """The fp32 reference pass over `seqs` (one pass over the weights).

    A complete dump with matching keys is reused wherever it is (R1 under its
    registered key is read in exp_036's work directory). Anything else is
    (re)computed, and only under work37_dir(): out_dir is checked before any
    directory is removed or created."""
    out_dir = Path(out_dir)
    dump = Dump(out_dir)
    if reusable(dump, ckpt_dir, text_key):
        log(f"[gate] reference fp32 dump reused: {out_dir.name} ({common.redact_path(out_dir)})")
        return dump.record() | {"reused": True}
    _require_under_work37(out_dir)
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


# ---------------------------------------------------------------------------
# Emulation statistics (G2 bf16 rows; exp_037: z_emu, M_emu, the flagged pair)
# ---------------------------------------------------------------------------


def emu_stats(ckpt_dir: Path, dump: Dump, n_rows: int, lengths, attn_chunk: int | None = None, log=print, *,
              flagged: dict | None = None, gap_sigma: float | None = None) -> dict:
    """Per layer: the emulated reference vs the fp32 reference on the same
    input h_l (rows [0, n_rows), i.e. T1-T8), forced and natural.

    Per layer: "attn" and "moe" (forced branch error statistics, as
    registered), "natural_disagree_frac", and z_record's fields for the
    emulation's natural selection: "sigma" (sigma_emu,l), "n_disagree",
    "max_gap_over_sigma" (the layer's maximum z_emu), "max_at",
    "n_beyond_6sigma" and "z_pairs". Top level: "M_emu" (the maximum z_emu
    over layers and pairs, 0.0 without a disagreement) and "M_emu_at",
    "n_beyond_6sigma" (summed), and "flagged_pair" (flagged_site's pair, by
    default: whether the emulation disagrees there, z_emu, sigma_t(emu),
    sigma_emu,l, the gap, both sets and the swap)."""
    t0, utc0 = time.time(), common.utc_iso()
    lengths = [int(x) for x in lengths]
    text_len = lengths[0]
    if any(x != text_len for x in lengths) or sum(lengths) != n_rows:
        raise ValueError(f"emu_stats: T1-T8 must be {len(lengths)} texts of one length summing to n_rows "
                         f"({lengths}, n_rows {n_rows})")
    gap_sigma = gap_sigma_default() if gap_sigma is None else float(gap_sigma)
    emu = RefRunner(ckpt_dir, "emu", attn_chunk=attn_chunk)
    n_layers = emu.cfg.num_hidden_layers
    site = dict(flagged) if flagged is not None else flagged_site(n_layers, text_len)
    segments = segments_of(lengths)
    layers, flag = [], None
    for i in range(n_layers):
        h = np.array(dump.layer(i, "h_in")[:n_rows])
        top6 = np.array(dump.layer(i, "top6")[:n_rows])
        gap = np.array(dump.layer(i, "top6_gap")[:n_rows])
        ref_logits = np.array(dump.layer(i, "router_logits")[:n_rows])
        r_attn = dump.layer(i, "r_attn")[:n_rows]
        r_moe = dump.layer(i, "r_moe")[:n_rows]
        forced = emu.branches(i, h, segments, force_ids=top6)
        natural = emu.branches(i, h, segments)
        sel = natural_vs_r1(natural["top6"], top6, gap, natural["logits"], ref_logits,
                            at_row=site["row"] if i == site["layer"] else None)
        layers.append({
            "layer": i,
            "attn": common.stats(common.rel_err_rows(forced["r_attn"], r_attn)),
            "moe": common.stats(common.rel_err_rows(forced["r_moe"], r_moe)),
            "natural_disagree_frac": float(sel["_dis"].mean()),
            **z_record(sel, text_len, gap_sigma),
        })
        if "_at" in sel:
            a = sel["_at"]
            flag = {**site, "emu_disagrees": a["disagrees"], "gap": a["gap"], "z_emu": a["z"],
                    "sigma_l_emu": a["sigma_l"], "sigma_t_emu": a["sigma_t"], "ref_top6": a["ref_top6"],
                    "emu_top6": a["top6"], "emu_dropped": a["dropped"], "emu_added": a["added"]}
        log(f"[gate] ref emu layer {i + 1}/{n_layers}")
    m_layer = max(range(n_layers), key=lambda j: layers[j]["max_gap_over_sigma"])
    m_emu = layers[m_layer]["max_gap_over_sigma"]
    m_at = {"layer": m_layer, **layers[m_layer]["max_at"]} if layers[m_layer]["max_at"] else None
    return {"version": EMU_STATS_VERSION, "layers": layers, "M_emu": m_emu, "M_emu_at": m_at,
            "n_disagree": sum(r["n_disagree"] for r in layers),
            "n_beyond_6sigma": sum(r["n_beyond_6sigma"] for r in layers), "gap_sigma": gap_sigma,
            "flagged_pair": flag, "z_rule": Z_RULE, "n_rows": int(n_rows), "lengths": lengths,
            "t_start": utc0, "t_end": common.utc_iso(), "wall_s": time.time() - t0}


def emu_stats_path(ckpt_dir: Path, text_key: str, work37: Path | None = None) -> Path:
    """work37_dir()/ref/<reference tree sha16>/<checkpoint fingerprint16>/<text
    key16>/emu_stats_exp037.json: beside R1's registered key, in exp_037's own
    work directory (DESIGN §2.9)."""
    base = Path(work37) if work37 is not None else common.work37_dir()
    return cache_dir("fp32", text_key, ckpt_dir, work=base).parent / EMU_STATS_NAME


def emu_stats_exp037(ckpt_dir: Path, dump: Dump, n_rows: int, lengths, path: Path, attn_chunk: int | None = None,
                     log=print, *, flagged: dict | None = None, gap_sigma: float | None = None) -> dict:
    """emu_stats, kept in `path` (emu_stats_path; under work37_dir()).

    Reused when the file exists with this version and the same R1 dump record
    sha256, rows, lengths, gap_sigma and flagged site; otherwise computed and
    (over)written. Returns the statistics with "reused" (bool) and "path"
    (redacted) added; the file holds the statistics with
    "dump_record_sha256"."""
    path = Path(path)
    lengths = [int(x) for x in lengths]
    gap_sigma = gap_sigma_default() if gap_sigma is None else float(gap_sigma)
    dump_sha = common.sha256_file(Path(dump.dir) / "record.json")
    if path.is_file():
        old = common.read_json(path)
        old_site = old.get("flagged_pair") or {}
        want_site = dict(flagged) if flagged is not None else old_site  # the default site is fixed by the dump
        same_site = bool(old_site) and all(old_site.get(k) == want_site.get(k) for k in ("layer", "text", "pos", "row"))
        if (old.get("version") == EMU_STATS_VERSION and old.get("dump_record_sha256") == dump_sha
                and old.get("n_rows") == int(n_rows) and old.get("lengths") == lengths
                and old.get("gap_sigma") == gap_sigma and same_site):
            log(f"[gate] emulation statistics reused: {common.redact_path(path)}")
            return old | {"reused": True, "path": common.redact_path(path)}
    _require_under_work37(path)
    st = emu_stats(ckpt_dir, dump, n_rows, lengths, attn_chunk=attn_chunk, log=log, flagged=flagged,
                   gap_sigma=gap_sigma)
    st["dump_record_sha256"] = dump_sha
    common.write_json(path, st)
    return st | {"reused": False, "path": common.redact_path(path)}


# ---------------------------------------------------------------------------
# NLL (G3)
# ---------------------------------------------------------------------------


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
