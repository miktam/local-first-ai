# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06; build task W5d).
"""F_tiny: the tau floor of G4-F32, its calibration and its out-of-sample
validation (DESIGN §5.1, §3.7, §4.2). gate/calibration.json is its record.

F_tiny = 10 x max_b mean_{t in block b} KL(R2F_tiny || port_tiny)(t), over the
2,048-position blocks b of one 16,384-position sequence, with KL by
gate.common.kl_rows (a float64 log-softmax of fp32 logits over the full
vocabulary). The series comes from the exact G4-F32 code path
(gate/checks/g4_forced.py: make_fp32_port, forced_texts, block_means), driven
from here:
  * port_tiny: the tiny K8 build loaded through the port and made the "K8 fp32
    port" (§3.0: set_dtype(float32), then fp32_on_dequantised_weights), forced
    by ForcedTrunk onto I_tiny, prefilled in 2,048-token chunks;
  * R2F_tiny: the reference on the tiny K8 build's dequantised weights,
    forced onto I_tiny (ref_drivers.forced_ref_pass);
  * I_tiny: the ascending sort of the tiny R1's natural top-6 (the fp32
    reference on the tiny BF16 checkpoint, ref_pass.fp32_dump;
    ref_drivers.i_from_r1). Forcing ids never come from the port.

Calibration: tests/tiny_real_layout.py at seed 23 (the real layer and head
layout at a tiny width), token ids np.random.default_rng(5).integers(0, 1008,
16384). The cap: F_tiny <= 1e-5 (thresholds.json G4F32 f_tiny_cap). Above it,
stop for Andrei before the freeze; the cap is not raised.

Validation (out of sample): the same builder at seed 29, token ids
np.random.default_rng(6).integers(0, 1008, 28672) laid out as the real sets in
pack order: 8 independent 1,536-token sequences (T1-8, each from position 0 in
one forward) and one 16,384-token sequence (T9, 2,048-token forwards; buckets
[0, 2,048), [2,048, 8,192), [8,192, 16,384)). The full G4-F32 code runs
(g4_forced.g4_f32_run and g4_mutants) with tau from F_tiny and that run's own
Csort, and gate/rules.py judges it (rules.g4_f32). The criteria, all of which
must be met (else stop for Andrei before the freeze):
  * F1, F3, F5 and F6 do not fire;
  * every F2 set and bucket mean <= tau / 10 (a >= 10x margin);
  * mutants 1, 6 and 19 each fire their bound leg (controls.json) with the
    margin of §4.2 (rules.control_result).
Because the seed differs, the margin can fail: it is not a tautology.

Every pass/fail decision of the gate is gate/rules.py's. This module measures,
and states whether the build-time criteria above are "met"; it never runs in
the gate (run_gate reads only F_tiny from gate/calibration.json, load_f_tiny).
Everything it writes is keyed and lies under work37_dir()/ftiny/ (DESIGN §2.9),
except gate/calibration.json (main(--write)).

CLI (the mini, before the freeze; about 15 minutes):
    python -m gate.checks.ftiny --root <scratch dir> [--write]
builds the two tiny sets under <root> if absent, runs both stages and, with
--write and every criterion met, writes gate/calibration.json. Exit 0 when
every criterion is met, 2 otherwise (stop for Andrei), 3 on an error.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gate import common
from gate.checks import ref_drivers, ref_pass

CALIBRATION_PATH = common.GATE_DIR / "calibration.json"
VERSION = "exp037-calibration-1"
WORK_SUBDIR = "ftiny"
DRAW_HIGH = 1008  # token ids from [0, 1008): below the tiny vocabulary's reserved top ids (tests/tiny_checkpoint.py)

CAL_CHECKPOINT_SEED = 23
VAL_CHECKPOINT_SEED = 29
CAL_TOKEN_SEED = 5
VAL_TOKEN_SEED = 6
T9_LEN = 16384
TEXT_LEN = 1536
N_TEXTS = 8
CHUNK = 2048  # every forward of T9 (and of the calibration sequence); T1-8 are one forward each
BLOCK = 2048  # F_tiny's block
F_TINY_FACTOR = 10.0  # F_tiny = 10 x the largest block mean
VALIDATION_MARGIN = 10.0  # every F2 set and bucket mean <= tau / 10
SILENT_LEGS = ("F1", "F3", "F5", "F6")  # must not fire on the correct port
G4F32_CHECK = "g4_f32_K8"
BUGHUNT_CONTEXT = ("exp_036's bughunt number 6.4e-7 (a block max of the tiny series, computed with fp32 log-prob "
                   "inputs) is context only and is not F_tiny (DESIGN §5.1)")
ESTIMATOR = ("gate.common.kl_rows(R2F_logits, port_logits): KL(R2F || port) per position in nats, a float64 "
             "log-softmax of fp32 logits over the full vocabulary (DESIGN §3.0); never a KL of fp32 log-probabilities")
F_TINY_RULE = ("F_tiny = 10 x max over the 2,048-position blocks b of one 16,384-position sequence of "
               "mean_{t in b} KL(R2F_tiny || port_tiny)(t) (DESIGN §5.1)")
CODE_PATH = ("gate/checks/g4_forced.py: make_fp32_port (set_dtype(float32), fp32_on_dequantised_weights), "
             "forced_texts (ForcedTrunk 'force' onto I, chunked through one cache, kl_rows against R2F), "
             "block_means; validation: g4_f32_run (with Csort), g4_mutants, then gate/rules.py g4_f32")
REFERENCE_PATH = ("I: gate/checks/ref_pass.fp32_dump (R1: KolibriReference on the BF16 checkpoint, natural) and "
                  "ref_drivers.i_from_r1; R2F / R3: ref_drivers.forced_ref_pass (KolibriReference on the BF16 config "
                  "with the K8 build's dequantised weights, emulate_bf16 for R3, forced onto I)")


@dataclass(frozen=True)
class Layout:
    """A tiny token layout: independent sequences, each from position 0, in pack order."""

    name: str
    token_seed: int
    lengths: tuple
    labels: tuple

    @property
    def n(self) -> int:
        return int(sum(self.lengths))

    def pack(self) -> np.ndarray:
        """np.random.default_rng(token_seed).integers(0, DRAW_HIGH, n): one draw, in pack order."""
        return np.random.default_rng(self.token_seed).integers(0, DRAW_HIGH, self.n).astype(np.int64)

    def bounds(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.lengths)]).astype(np.int64)

    def texts(self) -> list:
        """[(label, ids)] in pack order (g4_forced's texts)."""
        pack, b = self.pack(), self.bounds()
        return [(lab, pack[b[j]:b[j + 1]]) for j, lab in enumerate(self.labels)]

    def seqs(self) -> list:
        return [ids for _, ids in self.texts()]

    def segments(self) -> list:
        return ref_pass.segments_of(self.lengths)

    def rule(self) -> str:
        return (f"np.random.default_rng({self.token_seed}).integers(0, {DRAW_HIGH}, {self.n}), split in pack order "
                f"into {list(self.labels)} of {list(self.lengths)} tokens")

    def describe(self) -> dict:
        return {"name": self.name, "token_rng": self.rule(), "token_seed": self.token_seed, "draw_high": DRAW_HIGH,
                "lengths": list(self.lengths), "labels": list(self.labels),
                "pack_ids_sha256": common.ids_sha256(self.pack()), "ids_sha256_rule": common.IDS_SHA256_RULE}

    def text_key(self) -> str:
        """The key of this layout's reference dumps (in place of the gate-text key)."""
        return common.sha256_bytes(common.dumps({
            "kind": "exp037-ftiny", "layout": self.name, "token_seed": self.token_seed, "draw_high": DRAW_HIGH,
            "lengths": list(self.lengths), "pack_ids_sha256": common.ids_sha256(self.pack())}).encode("utf-8"))


CAL_LAYOUT = Layout("calibration", CAL_TOKEN_SEED, (T9_LEN,), ("T9",))
VAL_LAYOUT = Layout("validation", VAL_TOKEN_SEED, (TEXT_LEN,) * N_TEXTS + (T9_LEN,),
                    tuple(f"T{j + 1}" for j in range(N_TEXTS)) + ("T9",))


def work_root() -> Path:
    """work37_dir()/ftiny: every work file this module writes (DESIGN §2.9)."""
    return common.work37_dir() / WORK_SUBDIR


# ---------------------------------------------------------------------------
# Reference side: R1 -> I, R2F, R3 (the gate's own drivers)
# ---------------------------------------------------------------------------


def reference_dumps(bf16_dir, k8_dir, layout: Layout, modes=("fp32", "emu"), log=print) -> dict:
    """R1 (fp32 reference, BF16 weights, natural; ref_pass.fp32_dump), I (R1's
    top-6, ascending; ref_drivers.i_from_r1), then R2F ("fp32") and R3 ("emu")
    forced onto I on the K8 build's dequantised weights
    (ref_drivers.forced_ref_pass), each keyed and reused under work_root().
    Returns {"r1": dir, "I", "i_sha256", "text_key", "forced": {mode: dir},
    "keys": {mode: forced-dump key}}."""
    root = work_root()
    text_key = layout.text_key()
    r1_dir = ref_drivers.require_under_work37(ref_pass.cache_dir("fp32", text_key, Path(bf16_dir), work=root))
    ref_pass.fp32_dump(Path(bf16_dir), layout.seqs(), r1_dir, text_key, log=log)
    I, i_sha = ref_drivers.i_from_r1(r1_dir)
    shards = ref_drivers.build_shards(k8_dir)
    pack, segs = layout.pack(), layout.segments()
    forced, keys = {}, {}
    for mode in modes:
        key = ref_drivers.forced_dump_key(reference_tree_sha256=common.reference_tree_sha256(), k8_shards=shards,
                                          text_key=text_key, i_sha256=i_sha, mode=mode)
        out = ref_drivers.forced_dump_dir(key, root)
        ref_drivers.forced_ref_pass(bf16_dir, k8_dir, pack, segs, I, mode, out, key=key, log=log)
        forced[mode], keys[mode] = out, key
    return {"r1": r1_dir, "I": I, "i_sha256": i_sha, "text_key": text_key, "forced": forced, "keys": keys}


def _ref_summary(refs: dict) -> dict:
    return {"text_key": refs["text_key"], "i_sha256": refs["i_sha256"],
            "r1_record_sha256": common.sha256_file(Path(refs["r1"]) / "record.json"),
            "forced_keys": {m: k["sha256"] for m, k in refs["keys"].items()}}


# ---------------------------------------------------------------------------
# F_tiny
# ---------------------------------------------------------------------------


def f_tiny_of(means) -> float:
    """F_TINY_FACTOR x the largest block mean (ValueError unless all are finite)."""
    means = [float(m) for m in means]
    if not means or not all(np.isfinite(means)):
        raise ValueError(f"block means must be finite and non-empty: {means}")
    return F_TINY_FACTOR * max(means)


def _load_fp32_port(k8_dir):
    """The K8 fp32 port (DESIGN §3.0) of a build: (model, module)."""
    from gate.checks import g4_forced

    model, _, module = common.load_port(Path(k8_dir))
    g4_forced.make_fp32_port(model)
    return model, module


def calibrate(bf16_dir, k8_dir, *, cap: float | None = None, log=print) -> dict:
    """The calibration (DESIGN §5.1): R1 and R2F of CAL_LAYOUT on this set, the
    K8 fp32 port forced onto I_tiny by g4_forced.forced_texts (2,048-token
    forwards), the per-position KL's 2,048-position block means and F_tiny.
    cap: thresholds.json G4F32 f_tiny_cap when None. Returns the measured dict,
    with "cap_met" (F_tiny <= cap)."""
    from gate.checks import g4_forced

    th = common.load_thresholds()
    cap = float(th["G4F32"]["f_tiny_cap"] if cap is None else cap)
    t0 = time.time()
    refs = reference_dumps(bf16_dir, k8_dir, CAL_LAYOUT, modes=("fp32",), log=log)
    model, module = _load_fp32_port(k8_dir)
    series = g4_forced.forced_texts(model, CAL_LAYOUT.texts(), refs["I"], refs["forced"]["fp32"], None, CHUNK,
                                    module=module, log=log)
    del model, module
    g4_forced.release_memory()
    col = series["cols"]["T9"]
    kl = col["kl"]
    means = g4_forced.block_means(kl, BLOCK)
    f_tiny = f_tiny_of(means)
    changed = ~col["agree"]
    out = {
        "f_tiny": f_tiny,
        "block_means": means,
        "block": BLOCK,
        "factor": F_TINY_FACTOR,
        "argmax_block": int(np.argmax(means)),
        "n_positions": int(kl.size),
        "chunk": CHUNK,
        "series": {"mean_kl": float(kl.mean()), "max_kl": float(kl.max()), "p99_kl": float(np.quantile(kl, 0.99)),
                   "min_kl": float(kl.min()), "n_top1_changes": int(changed.sum()),
                   "n_top1_changes_decisive": int((changed & (col["lead"] >= th["G4F32"]["f1_decisive_lead_nats"])).sum()),
                   "shadow_disagree_rate": (float(col["shadow_dis"].mean()) if col["shadow_dis"] is not None else None)},
        "cap": cap,
        "cap_met": bool(f_tiny <= cap),
        "references": _ref_summary(refs),
        "seconds": time.time() - t0,
    }
    log(f"[ftiny] calibration: F_tiny {f_tiny:.4e} (block means max {max(means):.4e}); cap {cap:g}: "
        f"{'met' if out['cap_met'] else 'NOT MET (stop for Andrei)'}")
    return out


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validation_textset(th: dict | None = None, layout: Layout = VAL_LAYOUT):
    """A gate.textset.TextSet of the validation layout: T1..T8 of 1,536 tokens,
    T9 of 16,384, T9's buckets from thresholds.json (G2 t9_buckets)."""
    from gate import textset

    th = th if th is not None else common.load_thresholds()
    texts = dict(layout.texts())
    prof = textset.Profile(name="ftiny-validation", text_len=TEXT_LEN, t9_len=T9_LEN,
                           t9_buckets=tuple(tuple(int(x) for x in b) for b in th["G2"]["t9_buckets"]))
    ids = {t: [int(x) for x in texts[t]] for t in textset.ALL8}
    return textset.TextSet(profile=prof, ids=ids, t9=[int(x) for x in texts["T9"]], g5=[], token_bytes={},
                           key=layout.text_key(), sources={"ftiny": layout.rule()})


RULES_FIELDS = ("state", "reason", "measured", "threshold", "legs", "controls")


def _rules_fields(c: dict) -> dict:
    """The fields of a rules.py check dict that the record keeps (its state,
    reason, values, legs and controls; not its exp_036-style boolean)."""
    return {k: c[k] for k in RULES_FIELDS if k in c}


def validation_criteria(check: dict, margin: float = VALIDATION_MARGIN) -> dict:
    """The §5.1 criteria on rules.g4_f32's result for the validation run:
    {"criteria": [{"name", "met", ...}], "all_met"}."""
    legs, controls = check.get("legs") or {}, check.get("controls") or {}
    crit = [{"name": "state", "met": check.get("state") == "PASS", "state": check.get("state"),
             "reason": check.get("reason")}]
    for leg in SILENT_LEGS:
        fired = (legs.get(leg) or {}).get("fired")
        crit.append({"name": f"{leg} silent", "met": fired is False, "fired": fired})
    for where, v in sorted(((legs.get("F2") or {}).get("per") or {}).items()):
        bound = v["tau"] / margin
        crit.append({"name": f"F2 {where} mean <= tau / {margin:g}", "met": bool(v["mean_kl"] <= bound),
                     "mean_kl": v["mean_kl"], "tau": v["tau"], "bound": bound,
                     "headroom": (bound / v["mean_kl"]) if v["mean_kl"] > 0 else None})
    for cid in sorted(controls):
        r = controls[cid]
        m = r.get("margin") or {}
        crit.append({"name": f"control {cid} caught with the §4.2 margin", "met": bool(r.get("caught") and m.get("met")),
                     "mutant": r.get("mutant"), "leg": r.get("leg"), "caught": r.get("caught"), "margin": m})
    if not controls:
        crit.append({"name": "controls present", "met": False})
    return {"criteria": crit, "all_met": all(c["met"] for c in crit)}


def validate(bf16_dir, k8_dir, f_tiny: float, *, log=print) -> dict:
    """The out-of-sample validation (DESIGN §5.1) on this set: R1, R2F and R3 of
    VAL_LAYOUT; the K8 fp32 port through g4_forced.g4_f32_run (forced series,
    Csort, S); mutants 1, 6 and 19 through g4_forced.g4_mutants; then
    rules.g4_f32 with tau from f_tiny and this run's Csort, and the criteria.
    Returns {"f_tiny", "tau", "csort", "summary", "check" (rules.g4_f32 without
    its boolean), "criteria", "all_met", ...}."""
    from gate import rules
    from gate.checks import g4_forced

    th = common.load_thresholds()
    controls = rules.load_controls()
    t0 = time.time()
    refs = reference_dumps(bf16_dir, k8_dir, VAL_LAYOUT, modes=("fp32", "emu"), log=log)
    I, R2F, R3 = refs["I"], refs["forced"]["fp32"], refs["forced"]["emu"]
    ts = validation_textset(th)
    ref = g4_forced.reference_profile(g4_forced.text_layout(ts), R2F, R3, num_layers=len(I), log=log)
    model, module = _load_fp32_port(k8_dir)
    base = g4_forced.g4_f32_run(model, ts, I, R2F, R3, CHUNK, module=module, ref=ref, csort=True, doc_starts=(),
                                th=th, log=log)
    del model, module
    g4_forced.release_memory()
    muts = g4_forced.g4_mutants(g4_forced.port_loader(k8_dir), ts, I, R2F, R3, CHUNK, ref=ref,
                                checks=(G4F32_CHECK,), controls=controls, th=th, log=log)[G4F32_CHECK]
    check = rules.g4_f32(base, muts, f_tiny, th, controls)
    crit = validation_criteria(check)
    summary = {
        "sets": {s: {k: base["sets"][s].get(k) for k in ("n", "mean_kl", "max_kl", "csort_mean", "csort_max", "mean_S",
                                                          "n_top1_changes", "n_top1_changes_decisive",
                                                          "shadow_disagree_rate")}
                 for s in th["G4F32"]["sets"]},
        "t9_buckets": {lab: {k: b.get(k) for k in ("lo", "hi", "n", "mean_kl", "max_kl", "mean_S", "csort_mean")}
                       for lab, b in base["t9_buckets"].items()},
        "mutants": {str(m): {"sets": {s: {k: v["sets"][s].get(k) for k in ("mean_kl", "max_kl", "shadow_disagree_rate")}
                                      for s in th["G4F32"]["sets"]},
                             "t9_buckets": {lab: b.get("mean_kl") for lab, b in v["t9_buckets"].items()}}
                    for m, v in sorted(muts.items())},
    }
    out = {"f_tiny": float(f_tiny), "tau": (check.get("measured") or {}).get("tau"),
           "csort": (check.get("measured") or {}).get("csort"), "summary": summary,
           "check": _rules_fields(check), "criteria": crit["criteria"], "all_met": crit["all_met"],
           "margin": VALIDATION_MARGIN, "references": _ref_summary(refs), "seconds": time.time() - t0}
    log(f"[ftiny] validation: rules.g4_f32 {check.get('state')}; criteria "
        f"{'all met' if crit['all_met'] else 'NOT MET (stop for Andrei)'}")
    return out


# ---------------------------------------------------------------------------
# The record (gate/calibration.json)
# ---------------------------------------------------------------------------


def _sha_files(d: Path, names) -> dict:
    return {n: common.sha256_file(Path(d) / n) for n in names}


def checkpoint_record(bf16_dir, k8_dir, seed: int, sharpen: float = 3.0) -> dict:
    """Config and shard sha256 of a tiny set's BF16 checkpoint and its K8 build,
    with the builder's parameters."""
    bf16, k8 = Path(bf16_dir), Path(k8_dir)
    cfg = common.read_json(bf16 / "config.json")
    q = common.read_json(k8 / "config.json")["quantization"]
    return {
        "seed": int(seed),
        "builder": ("tests/tiny_real_layout.py build_real_layout_set (exp_036 diagnostics/gate1/mini/bughunt/verify/"
                    "build_ck.py's recipe); the K8 build by port/convert.py, 8 bits, group 64"),
        "preset": "w513",
        "sharpen_qk_norms": float(sharpen),
        "layout": {k: cfg.get(k) for k in ("num_hidden_layers", "layer_types", "sliding_window", "num_attention_heads",
                                          "num_key_value_heads", "head_dim", "hidden_size", "num_experts",
                                          "num_experts_per_tok", "moe_intermediate_size",
                                          "shared_expert_intermediate_size", "vocab_size")},
        "bf16": {"config_sha256": common.sha256_file(bf16 / "config.json"),
                 "index_sha256": common.sha256_file(bf16 / "model.safetensors.index.json"),
                 "shards_sha256": _sha_files(bf16, sorted(p.name for p in bf16.glob("model*.safetensors")))},
        "k8": {"bits": int(q["bits"]), "group_size": int(q["group_size"]),
               "config_sha256": common.sha256_file(k8 / "config.json"),
               "shards_sha256": _sha_files(k8, sorted(p.name for p in k8.glob("model*.safetensors")))},
    }


def _sysctl(name: str):
    try:
        p = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=10)
        return p.stdout.strip() or None if p.returncode == 0 else None
    except Exception:
        return None


CODE_FILES = ("gate/checks/ftiny.py", "gate/checks/g4_forced.py", "gate/harness.py", "gate/checks/ref_drivers.py",
              "gate/checks/ref_pass.py", "gate/common.py", "gate/rules.py", "gate/port_mutants.py")


def code_sha256() -> dict:
    """sha256 of the measurement and rule code behind the record (experiment-relative paths)."""
    return {rel: common.sha256_file(common.EXP_DIR / rel) for rel in CODE_FILES}


def environment() -> dict:
    """Software and host of this run: a host label, never a host name or a path."""
    import importlib.metadata as md

    import mlx.core as mx
    import mlx_lm

    try:
        metal = md.version("mlx-metal")
    except md.PackageNotFoundError:
        metal = None
    try:
        from tools import redact

        mem = _sysctl("hw.memsize")
        host = redact.host_label(_sysctl("machdep.cpu.brand_string"), int(mem) if mem else None)
    except Exception:
        host = None
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return {"host": host, "python": platform.python_version(), "macos": platform.mac_ver()[0],
            "packages": {"mlx": mx.__version__, "mlx-metal": metal, "mlx-lm": mlx_lm.__version__,
                         "numpy": np.__version__},
            "architecture": info.get("architecture"), "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32")}


def build_record(cal: dict, val: dict, cal_ckpt: dict, val_ckpt: dict, env: dict | None = None) -> dict:
    """gate/calibration.json's content (DESIGN §5.1 "Record")."""
    return {
        "version": VERSION,
        "source": "DESIGN §5.1 (F_tiny, fix 4); written by gate/checks/ftiny.py (build task W5d); frozen with "
                  "GATE_RULES_SHA256 (tools/hash_tree.py scope gate_rules)",
        "f_tiny": cal["f_tiny"],
        "f_tiny_rule": F_TINY_RULE,
        "block_means": cal["block_means"],
        "block": cal["block"],
        "factor": cal["factor"],
        "cap": cal["cap"],
        "cap_rule": "F_tiny <= 1e-5 (= 10 x the Csort mean ceiling 1e-6; thresholds.json G4F32 f_tiny_cap); above it, "
                    "stop for Andrei before the freeze; the cap is not raised",
        "estimator": ESTIMATOR,
        "code_path": CODE_PATH,
        "reference_path": REFERENCE_PATH,
        "calibration": {
            "checkpoint": cal_ckpt,
            "tokens": CAL_LAYOUT.describe(),
            "chunk": CHUNK,
            "I": "the ascending sort of the tiny R1's natural top-6 (R1: the fp32 reference on the tiny BF16 "
                 "checkpoint)",
            "measured": {k: cal[k] for k in ("n_positions", "argmax_block", "series", "references", "seconds")},
            "cap_met": cal["cap_met"],
        },
        "validation": {
            "checkpoint": val_ckpt,
            "tokens": VAL_LAYOUT.describe(),
            "layout": {"T1-8": f"{N_TEXTS} independent {TEXT_LEN}-token sequences, each from position 0, one forward "
                               "each", "T9": f"one {T9_LEN}-token sequence in {CHUNK}-token forwards through one cache",
                       "t9_buckets": [[0, 2048], [2048, 8192], [8192, 16384]], "pack_order": list(VAL_LAYOUT.labels)},
            "rule": "the full G4-F32 code with tau = max(F_tiny, 100 x that run's mean Csort) per set: F1, F3, F5 and F6 "
                    "do not fire; every F2 set and bucket mean <= tau / 10; mutants 1, 6 and 19 each fire their bound "
                    "leg with the §4.2 margin. Any failure: stop for Andrei before the freeze",
            "tau": val["tau"],
            "csort": val["csort"],
            "summary": val["summary"],
            "criteria": val["criteria"],
            "all_met": val["all_met"],
            "rules_g4_f32": val["check"],
            "references": val["references"],
            "seconds": val["seconds"],
        },
        "context": BUGHUNT_CONTEXT,
        "provenance": {"port_sha256": common.sha256_file(common.PORT_FILE),
                       "reference_tree_sha256": common.reference_tree_sha256(),
                       "code_sha256": code_sha256(),
                       "thresholds_sha256": common.sha256_file(common.THRESHOLDS_PATH),
                       "controls_sha256": common.sha256_file(common.GATE_DIR / "controls.json"),
                       "environment": env if env is not None else environment(),
                       "utc": common.utc_iso()},
    }


def load(path: Path = CALIBRATION_PATH) -> dict:
    """gate/calibration.json."""
    return common.read_json(path)


def load_f_tiny(path: Path = CALIBRATION_PATH) -> float:
    """F_tiny from gate/calibration.json, the value rules.g4_f32 takes."""
    v = load(path).get("f_tiny")
    if not isinstance(v, (int, float)) or isinstance(v, bool) or not np.isfinite(v) or v <= 0:
        raise ValueError(f"calibration.json f_tiny is {v!r}, not a positive finite number")
    return float(v)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _set_dirs(root: Path, seed: int):
    """(bf16, k8) of the tiny real-layout set for `seed` under root/set<seed>,
    built by tests/tiny_real_layout.py when absent."""
    tests_dir = common.EXP_DIR / "tests"
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    import tiny_real_layout as trl

    d = Path(root) / f"set{seed}"
    k4 = d / trl.build_dir_name(4)
    if not (k4 / "config.json").is_file():
        trl.build_real_layout_set(d, seed)
    return d / trl.BF16_NAME, d / trl.build_dir_name(8)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="exp_037 F_tiny: calibration (seed 23) and validation (seed 29), "
                                             "DESIGN §5.1")
    ap.add_argument("--root", required=True, help="directory for the two tiny sets (built when absent)")
    ap.add_argument("--write", action="store_true", help="write gate/calibration.json when every criterion is met")
    ap.add_argument("--out", default=str(CALIBRATION_PATH), help="where --write writes (default gate/calibration.json)")
    ap.add_argument("--report", help="also write the record (met or not) to this file")
    args = ap.parse_args(argv)
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    from tools.precision import ensure_exact_fp32

    ensure_exact_fp32()
    root = Path(args.root).expanduser()
    try:
        cal_bf16, cal_k8 = _set_dirs(root, CAL_CHECKPOINT_SEED)
        cal = calibrate(cal_bf16, cal_k8)
        if not cal["cap_met"]:
            print(json.dumps({"stage": "calibration", "f_tiny": cal["f_tiny"], "cap": cal["cap"], "cap_met": False,
                              "action": "stop for Andrei before the freeze (DESIGN §5.1)"}))
            return 2
        val_bf16, val_k8 = _set_dirs(root, VAL_CHECKPOINT_SEED)
        val = validate(val_bf16, val_k8, cal["f_tiny"])
        record = build_record(cal, val, checkpoint_record(cal_bf16, cal_k8, CAL_CHECKPOINT_SEED),
                              checkpoint_record(val_bf16, val_k8, VAL_CHECKPOINT_SEED))
    except Exception as e:  # noqa: BLE001 - reported, exit 3
        print(json.dumps({"error": f"{type(e).__name__}: {e}"}))
        return 3
    if args.report:
        common.write_json(Path(args.report), record)
    if not val["all_met"]:
        print(json.dumps({"stage": "validation", "f_tiny": cal["f_tiny"], "all_met": False,
                          "unmet": [c["name"] for c in val["criteria"] if not c["met"]],
                          "action": "stop for Andrei before the freeze (DESIGN §5.1)"}))
        return 2
    if args.write:
        common.write_json(Path(args.out), record)
    print(json.dumps({"f_tiny": cal["f_tiny"], "cap_met": True, "all_met": True,
                      "written": common.redact_path(args.out) if args.write else None}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
