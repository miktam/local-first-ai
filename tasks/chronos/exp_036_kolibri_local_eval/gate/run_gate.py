# SPDX-License-Identifier: MIT
"""Phase 0 driver: the port-correctness gate (HYPOTHESIS Phase 0; BUILD_SPEC §5.3).

Phases (each a separate process in a real run, writing results/gate/<UTC>/phaseN.json):
  1  G0 static (census, strict loads, template and tokenizer parity, converted
     configs, same port sha) and G1 (the weight-free suite)
  2  reference passes on CPU: fp32 on T1-T8 and T9 (dumps), bf16 emulation,
     the seven reference mutants in one streamed pass; G3
  3  G2 per layer: fp32, bf16, T9, router margin, G2q (K8, K4), real-weight mutants
  4  K8: G0 value-level dtypes, G4, G5 (greedy continuations, noise floor,
     decode vs prefill, batch parity, behaviour)
  5  reference pass on the G5 continuations; the G5 greedy comparison
  6  K4: G0 value-level dtypes, G4 (descriptive), batch parity (reported), behaviour

Verdict (computed in code from gate/thresholds.json):
  K8 PASS iff every blocking check for K8 passes (G0-G5);
  K4 PASS iff G0 for K4, the same port sha256 as K8, G2q at 4-bit and the K4
  behaviour check pass. A partial run is never PASS.
Exit codes (incident_003 contract; the last stdout line is JSON):
  0 both PASS, 4 K8 PASS and K4 FAIL, 1 K8 FAIL, 3 could not run.
Record: results/gate/gate_<UTC>.json (+ _layers.csv, _mutants.json) with every
measured value next to its threshold and the sha256 of the port, the converted
manifests, the thresholds, the reference tree, the gate code and the gate text.

require_pass(arm) is the refusal the runner calls (runner/guard.py require_gate).

CLI:
  run_gate.py --all                                   every check, arms K8, K4
  run_gate.py [--checks g0,g1,...] [--arms K8,K4]
  run_gate.py --tiny DIR [...]                        the same drivers on a tiny
      checkpoint (DIR/Kolibri-1-BF16 and its DIR/Kolibri-1-MLX-{8,4}bit-g64
      conversions); thresholds are reported, the verdict is "TINY", and the
      exit code follows the tiny outcome (used by the tests)
  run_gate.py --require-pass K8                       exit 1 unless the latest record allows K8
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gate import common

GROUPS = ("g0", "g1", "g2", "g3", "g4", "g5")
PHASES_FOR = {"g0": {1, 4, 6}, "g1": {1}, "g2": {2, 3}, "g3": {2}, "g4": {2, 4, 6}, "g5": {4, 5, 6}}
PHASE_TITLES = {
    1: "G0 static, G1",
    2: "reference passes (fp32 dumps, bf16 emulation, mutants), G3",
    3: "G2 per layer (fp32, bf16, T9, router margin, G2q, real-weight mutants)",
    4: "K8: G0 dtypes, G4, G5",
    5: "reference pass on the G5 continuations",
    6: "K4: G0 dtypes, G4, G5 batch and behaviour",
}
SHARED_G0 = ["g0_census", "g0_strict_load_source", "g0_template_parity", "g0_tokenizer_parity"]
BLOCKING = {
    "K8": SHARED_G0 + [
        "g0_strict_load_K8", "g0_converted_config_K8", "g0_router_bf16_exact_K8", "g0_head_bf16_exact_K8",
        "g1",
        "g2_fp32_forced_branch", "g2_fp32_natural_selection", "g2_embedding", "g2_head",
        "g2_bf16_forced_branch", "g2_bf16_natural_selection", "g2_router_margin", "g2_t9_buckets", "g2q_K8",
        "g2_real_weight_mutants",
        "g3_bpb_per_text", "g3_bpb_vs_peers", "g3_ref_mutants",
        "g4_K8_backstop",
        "g5_greedy", "g5_decode_vs_prefill", "g5_batch_parity_K8", "g5_behaviour_K8",
    ],
    # thresholds.json K4_blocking: ["G0", "same_port_sha", "G2q", "G5_behaviour"]
    "K4": SHARED_G0 + [
        "g0_strict_load_K4", "g0_converted_config_K4", "g0_router_bf16_exact_K4", "g0_head_bf16_exact_K4",
        "same_port_sha", "g2q_K4", "g5_behaviour_K4",
    ],
}
GROUP_OF = lambda cid: cid.split("_", 1)[0] if cid != "same_port_sha" else "g0"  # noqa: E731


class GateRefused(SystemExit):
    """require_pass refuses (an exit-1 SystemExit carrying the reason)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class CouldNotRun(RuntimeError):
    """A precondition of the gate is not met (exit 3)."""


@dataclass
class GateContext:
    """BUILD_SPEC §5.3 GateContext(models_dir, work_dir, thresholds, arms)."""
    models_dir: Path
    work_dir: Path
    thresholds: dict
    arms: tuple = ("K8", "K4")
    checks: tuple = GROUPS
    tiny: bool = False
    results_dir: Path = common.DEFAULT_RESULTS_DIR
    tok_dir: Path | None = None
    mutant: str | None = None            # tiny only: run the gate on a mutated port
    head_policy: str | None = None       # tiny only; real runs read the sign-off
    g1_select: list | None = None        # tiny only: a subset of the weight-free suite
    subprocess_phases: bool = True
    utc: str = field(default_factory=common.utc_stamp)
    log_stream: object = None

    @property
    def bf16_dir(self) -> Path:
        return Path(self.models_dir) / common.BF16_DIR

    def arm_dir(self, arm: str) -> Path:
        return Path(self.models_dir) / common.ARM_DIRS[arm]

    @property
    def gate_dir(self) -> Path:
        return Path(self.results_dir) / "gate"

    @property
    def run_dir(self) -> Path:
        return self.gate_dir / self.utc

    def want(self, group: str) -> bool:
        return group in self.checks

    def log(self, msg: str) -> None:
        print(msg, file=self.log_stream or sys.stderr, flush=True)


@dataclass
class GateResult:
    record: dict
    path: Path | None
    exit_code: int


@dataclass
class GateRecord:
    """What require_pass returns: the record that allows `arm`."""
    path: Path
    arm: str
    record: dict

    @property
    def verdict(self) -> str:
        return self.record["verdict"][self.arm]

    def get(self, key, default=None):
        return self.record.get(key, default)


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def signed_off_policy(hypothesis: Path = common.EXP_DIR / "HYPOTHESIS.md") -> tuple[str, str]:
    """The head policy of the sign-off block: "vendor_faithful" if Andrei's
    entry mentions vendor-faithful, "quantised_head" otherwise (the
    pre-registered default, also when the entry is still empty)."""
    for line in hypothesis.read_text(encoding="utf-8").splitlines():
        if line.startswith("- Kolibri `embed_tokens` / `lm_head` policy"):
            answer = line.rsplit("):", 1)[-1].strip() if "):" in line else ""
            if not answer:
                return "quantised_head", "pre-registered default (sign-off entry empty)"
            if "vendor" in answer.lower():
                return "vendor_faithful", f"sign-off: {answer[:120]}"
            return "quantised_head", f"sign-off: {answer[:120]}"
    return "quantised_head", "pre-registered default (no sign-off line found)"


def head_policy(ctx: GateContext) -> tuple[str, str]:
    if ctx.tiny and ctx.head_policy:
        return ctx.head_policy, "--head-policy (tiny run)"
    return signed_off_policy()


def textset_for(ctx: GateContext):
    from gate import textset

    if ctx.tiny:
        cfg = common.read_json(ctx.bf16_dir / "config.json")
        return textset.make_tiny(cfg.get("sliding_window") or 0, cfg["num_hidden_layers"], cfg["vocab_size"])
    ts = textset.load_real(ctx.tok_dir, Path(ctx.work_dir) / "gate_texts", textset.real_profile(ctx.thresholds))
    th5 = ctx.thresholds["G5"]
    if len(ts.g5) != th5["greedy_prompts"] or min(len(p["ids"]) for p in ts.g5) < th5["greedy_prompt_min_tokens"]:
        raise CouldNotRun("gate/texts/G5_prompts.json does not match thresholds G5 (8 prompts, >= 600 tokens)")
    return ts


def dump_dir(ctx: GateContext, ts) -> Path:
    from gate.checks.ref_pass import cache_dir

    return cache_dir("fp32", ts.key, ctx.bf16_dir, ctx.work_dir)


def planned_dump_bytes(ctx: GateContext, ts) -> int:
    cfg = common.read_json(ctx.bf16_dir / "config.json")
    N = 8 * ts.profile.text_len + ts.profile.t9_len
    H, E, k, V, L = cfg["hidden_size"], cfg["num_experts"], cfg["num_experts_per_tok"], cfg["vocab_size"], cfg["num_hidden_layers"]
    per_layer = N * (3 * H * 4 + E * 4 + k * 8 + 4)
    return L * per_layer + N * V * 4 + 2 * N * H * 4


def preconditions(ctx: GateContext) -> dict:
    """Everything that makes a run impossible rather than failed (exit 3)."""
    info = {}
    if not ctx.bf16_dir.is_dir():
        raise CouldNotRun(f"{common.redact_path(ctx.bf16_dir)} is missing")
    need_arms = [a for a in ctx.arms if any(ctx.want(g) for g in ("g0", "g2", "g4", "g5"))]
    port = common.port_sha256()
    for arm in need_arms:
        d = ctx.arm_dir(arm)
        if not d.is_dir():
            raise CouldNotRun(f"{common.redact_path(d)} is missing (RUNBOOK step 8)")
        from port.convert import StalePortFile, check_port_file

        try:  # the build runs its own kolibri1.py copy: it must be the gated file
            check_port_file(d)
        except StalePortFile as e:
            raise CouldNotRun(str(e)) from e
        info[f"{arm}_port_sha256"] = common.read_json(d / common.CONVERT_RECORD).get("port_sha256")
    info["port_sha256"] = port
    if not ctx.tiny:
        tok = Path(ctx.tok_dir or common.tokenizer_dir())
        if not (tok / "tokenizer.json").is_file():
            raise CouldNotRun(f"no Kolibri tokenizer in {common.redact_path(tok)}")
        for tid in ("T5", "T6", "T9"):
            if not (Path(ctx.work_dir) / "gate_texts" / f"{tid}.ids.json").is_file():
                raise CouldNotRun(f"$EXP036_WORK/gate_texts/{tid}.ids.json missing (RUNBOOK step 7)")
        common.require_identity()
    return info


def check_disk(ctx: GateContext, ts) -> dict:
    from gate.checks.ref_pass import Dump, reusable

    d = dump_dir(ctx, ts)
    if reusable(Dump(d), ctx.bf16_dir, ts.key):
        return {"dump_reusable": True}
    need = planned_dump_bytes(ctx, ts)
    Path(ctx.work_dir).mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(ctx.work_dir).free
    if free < need:
        raise CouldNotRun(f"$EXP036_WORK has {free / 1e9:.1f} GB free; the reference dumps need about {need / 1e9:.1f} GB")
    return {"dump_reusable": False, "planned_bytes": need, "free_bytes": free}


def _check(passed, measured=None, threshold=None, applicable=True, **extra) -> dict:
    return {"pass": None if passed is None else bool(passed), "measured": measured, "threshold": threshold,
            "applicable": applicable, **extra}


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


def phase1(ctx: GateContext) -> dict:
    from gate import tokenizer_lines
    from gate.checks import g0_static, g1_synthetic

    th0 = ctx.thresholds["G0"]
    checks, data = {}, {}
    if ctx.want("g0"):
        if ctx.tiny:
            from reference.kolibri_ref import expected_shapes, load_config, param_count

            cfg = load_config(str(ctx.bf16_dir))
            expected = {"tensor_count": len(expected_shapes(cfg)), "param_count": param_count(cfg),
                        "full_attention_layers": [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]}
        else:
            expected = {k: th0[k] for k in ("tensor_count", "param_count", "full_attention_layers")}
        cen = g0_static.census(ctx.bf16_dir, expected, ctx.results_dir, hash_shards=not ctx.tiny,
                               oids=None if ctx.tiny else g0_static.lfs_oids())
        sha_ok = cen["shard_sha"]["status"] == "checked" and not cen["shard_sha"].get("mismatches")
        ok = (cen["tensors"] == expected["tensor_count"] and cen["params"] == expected["param_count"]
              and cen["full_attention_layers"] == expected["full_attention_layers"] and (sha_ok or ctx.tiny))
        checks["g0_census"] = _check(ok, cen, expected, note="tiny: counts from the tiny config; shard sha not available" if ctx.tiny else None)
        for name, d in [("source", ctx.bf16_dir)] + [(a, ctx.arm_dir(a)) for a in ctx.arms]:
            sl = g0_static.strict_load_check(d, ctx.mutant)
            ok = (sl["n_unused_ckpt_tensors"] == 0 and sl["n_missing_params"] == 0 and not sl["shape_mismatch"]
                  and sl["load_error"] is None and (sl["numel_equal"] or sl["quantized"]))
            checks[f"g0_strict_load_{name}"] = _check(ok, sl, "every tensor consumed once, no parameter absent")
        tok = ctx.tok_dir if ctx.tiny else (ctx.tok_dir or common.tokenizer_dir())
        if tok is not None and (Path(tok) / "tokenizer.json").is_file():
            tp = g0_static.template_parity(tok)
            n_want = th0["template_conversations"] * th0["template_settings"]
            ok = (tp["n"] == n_want and not tp["mismatches"] and not tp["runner_chat"]["mismatches"])
            checks["g0_template_parity"] = _check(ok, tp, {"n": n_want, "mismatches": 0})
            groups = tokenizer_lines.committed_rule_lines()
            mmlu, mmlu_files = tokenizer_lines.mmlu_lines() if not ctx.tiny else ([], [])
            if mmlu:
                groups["mmlu"] = mmlu
            gate_texts = _gate_texts_for_parity()
            tk = g0_static.tokenizer_parity(tok, groups, gate_texts)
            tk["mmlu_files"] = mmlu_files or "none found under $EXP036_DATA (committed lines only)"
            ok = (tk["n_lines"] >= th0["tokenizer_lines_min"] and tk["mismatches"] <= th0["tokenizer_mismatches_max"]
                  and tk["roundtrip_fail"] == 0 and tk["fix_mistral_regex"] is not True)
            checks["g0_tokenizer_parity"] = _check(ok, tk, {"lines_min": th0["tokenizer_lines_min"],
                                                            "mismatches_max": th0["tokenizer_mismatches_max"]})
        else:
            for cid in ("g0_template_parity", "g0_tokenizer_parity"):
                checks[cid] = _check(None, {"note": "no tokenizer directory given"}, None, applicable=False)
        policy, source = head_policy(ctx)
        data["head_policy"] = {"policy": policy, "source": source}
        for arm in ctx.arms:
            cc = g0_static.converted_config_check(ctx.arm_dir(arm), common.ARM_BITS[arm], policy, th0)
            checks[f"g0_converted_config_{arm}"] = _check(cc["n_problems"] == 0, cc, {"policy": policy})
        port = common.port_sha256()
        shas = {a: common.read_json(ctx.arm_dir(a) / common.CONVERT_RECORD).get("port_sha256") for a in ctx.arms}
        checks["same_port_sha"] = _check(all(v == port for v in shas.values()), {"port": port, **shas},
                                         "K8 and K4 converted with the current port/kolibri1.py")
    if ctx.want("g1"):
        sel = ctx.g1_select if ctx.tiny else None
        # junit XML carries the host name and absolute paths: it stays in the
        # work dir; the record keeps only its counts and sha256 (BUILD_SPEC §2 Privacy).
        g1 = g1_synthetic.run(Path(ctx.work_dir) / "gate_g1" / f"junit_{ctx.utc}.xml", sel, require_all=not ctx.tiny)
        # A "build-host only:" skip (files only the mini holds; passed there before the push) is not a G1 skip.
        skipped = g1["skipped"] - g1.get("skipped_build_host_only", 0)
        ok = g1["tests"] > 0 and g1["failures"] == 0 and g1["errors"] == 0 and (ctx.tiny or skipped == 0)
        checks["g1"] = _check(ok, g1, {"failures": 0, "errors": 0, "skipped": 0, "skipped_except_build_host_only": 0,
                                       "source": ctx.thresholds["G1"]["source"]})
    return {"checks": checks, "data": data}


def _gate_texts_for_parity() -> dict:
    from gate.textset import FILES

    out = {}
    for tid, f in FILES.items():
        rec = common.read_json(common.TEXTS_DIR / f)
        if tid in ("T7", "T8"):
            out[tid] = (rec["text"], rec["ids"])
        else:
            out[tid] = ((common.TEXTS_DIR / f.replace(".ids.json", ".txt")).read_text(encoding="utf-8"), rec["ids"])
    return out


def phase2(ctx: GateContext) -> dict:
    from gate.checks import g3_oracle, ref_pass

    ts = textset_for(ctx)
    data = {"disk": check_disk(ctx, ts)}
    seqs = ts.packed8() + [ts.t9]
    d = dump_dir(ctx, ts)
    rec = ref_pass.fp32_dump(ctx.bf16_dir, seqs, d, ts.key, log=ctx.log)
    data["dump"] = {"dir": "$EXP036_WORK/" + d.relative_to(ctx.work_dir).as_posix(), "reused": bool(rec.get("reused")),
                    "record_sha256": common.sha256_file(d / "record.json"),
                    "files": {k: v["sha256"] for k, v in rec["files"].items()},
                    "summary": {k: rec[k] for k in ("num_tokens", "seq_lengths", "ref_version", "wall_s", "peak_rss_bytes")
                                if k in rec}}
    checks = {}
    dump = ref_pass.Dump(d)
    per_pass = common.checkpoint_bytes(ctx.bf16_dir)
    passes = [{"what": "fp32 reference, T1-T8 and T9 in one pass", "bytes": 0 if rec.get("reused") else per_pass}]
    if ctx.want("g2"):
        emu_path = d.parent / "emu_stats.json"
        if emu_path.is_file() and common.read_json(emu_path).get("dump_record_sha256") == data["dump"]["record_sha256"]:
            emu = common.read_json(emu_path)
            data["emu_reused"] = True
        else:
            try:
                emu = ref_pass.emu_stats(ctx.bf16_dir, dump, 8 * ts.profile.text_len, [ts.profile.text_len] * 8, log=ctx.log)
                emu["dump_record_sha256"] = data["dump"]["record_sha256"]
                common.write_json(emu_path, emu)
            except ref_pass.CapabilityMissing as e:
                emu = {"capability_missing": str(e)}
            passes.append({"what": "bf16-emulated reference fed the fp32 h_l, forced and natural",
                           "bytes": 0 if "capability_missing" in emu else per_pass})
        data["emu"] = emu
    if ctx.want("g3"):
        th3 = ctx.thresholds["G3"]
        bpb = g3_oracle.ref_bpb(dump, ts)
        peers = g3_oracle.peer_bpb(ctx.results_dir) if not ctx.tiny else None
        try:
            nll, path = ref_pass.mutant_nll(ctx.bf16_dir, ts.ids[th3["ref_mutant_text"]])
            muts = g3_oracle.ref_mutant_nll(nll, th3["ref_mutant_blocks"], th3["ref_mutant_dnll_quantile"])
            data["g3_mutant_path"] = path
            passes.append({"what": "7 reference mutants + base on T3, parallel streams", "bytes": per_pass})
        except ref_pass.CapabilityMissing as e:
            muts = None
            data["g3_mutants_missing"] = str(e)
        checks.update(g3_oracle.evaluate(bpb, peers, muts, th3, ctx.tiny))
        if muts is None:
            checks["g3_ref_mutants"] = _check(None, {"capability_missing": data["g3_mutants_missing"]}, None,
                                              applicable=not ctx.tiny)
    data["reference_passes"] = passes
    return {"checks": checks, "data": data}


def phase3(ctx: GateContext) -> dict:
    from gate.checks import g2_layers, ref_pass

    ts = textset_for(ctx)
    p2 = common.read_json(ctx.run_dir / "phase2.json")
    emu = p2["data"].get("emu")
    if emu is not None and "capability_missing" in emu:
        emu = None
    dump = ref_pass.Dump(dump_dir(ctx, ts))
    res = g2_layers.run(ctx, dump, emu, ts, log=ctx.log)
    checks = g2_layers.evaluate(res, emu, ctx.thresholds["G2"])
    if emu is None:
        reason = p2["data"].get("emu", {}).get("capability_missing", "no emulation statistics")
        for cid in ("g2_bf16_forced_branch", "g2_bf16_natural_selection", "g2_t9_buckets", "g2_router_margin"):
            checks.setdefault(cid, _check(None, {"capability_missing": reason}, None))
    for arm in ctx.arms:
        g = res["layers_g2q"].get(arm, {})
        if f"g2q_{arm}" not in checks:
            checks[f"g2q_{arm}"] = _check(None, {"capability_missing": g.get("capability_missing", "not run")}, None)
    if ctx.mutant is not None:
        checks.pop("g2_real_weight_mutants", None)
        checks["g2_real_weight_mutants"] = _check(None, {"note": "not run on a mutated port"}, None, applicable=False)
    rows = g2_layers.layer_csv_rows(res, emu)
    mutants = {"real_weight_mutants": res["real_weight_mutants"], "router_margin": res.get("router_margin"),
               "forcing_crosscheck": res.get("forcing_crosscheck")}
    passes = [{"what": f"dequantised reference on {arm}", "bytes": common.checkpoint_bytes(ctx.arm_dir(arm))}
              for arm in ctx.arms if "capability_missing" not in res["layers_g2q"].get(arm, {"capability_missing": 1})]
    return {"checks": checks, "data": {"g2": _strip(res), "layer_rows": rows, "mutants": mutants,
                                       "reference_passes": passes}}


def _strip(res: dict) -> dict:
    """The phase JSON keeps summaries; per-layer rows go to the CSV."""
    out = {k: v for k, v in res.items() if k not in ("layers_fp32", "layers_bf16", "layers_t9", "layers_g2q")}
    out["layers_g2q"] = {a: {k: v for k, v in g.items() if k != "layers"} for a, g in res.get("layers_g2q", {}).items()}
    return out


def _arm_phase(ctx: GateContext, arm: str) -> dict:
    import mlx.core as mx

    from gate.checks import g0_static, g4_e2e, g5_generation, ref_pass

    ts = textset_for(ctx)
    prof = ts.profile
    th0, th5 = ctx.thresholds["G0"], ctx.thresholds["G5"]
    checks, data = {}, {}
    model, _, module = common.load_port(ctx.arm_dir(arm), mutant=ctx.mutant)
    if ctx.want("g0"):
        n = 256 if not ctx.tiny else 64
        dt = g0_static.dtype_asserts(model, module, ts.ids["T1"][:n])
        checks[f"g0_router_bf16_exact_{arm}"] = _check(
            dt["router_logits_dtype"] == th0["router_logits_dtype"] and dt["router_bf16_exact_frac"] <= th0["router_bf16_exact_frac_max"],
            dt, {k: th0[k] for k in ("router_logits_dtype", "router_bf16_exact_frac_max")})
        checks[f"g0_head_bf16_exact_{arm}"] = _check(
            dt["head_dtype"] == th0["head_output_dtype"] and dt["head_bf16_exact_frac"] <= th0["head_bf16_exact_frac_max"],
            dt, {k: th0[k] for k in ("head_output_dtype", "head_bf16_exact_frac_max")})
    if ctx.tiny:
        # Tiny random weights: bf16 noise is dominated by routing near-ties
        # (tests/INTEGRATION_LOG.md entries 1-3), so the tiny run exercises the
        # G4/G5 drivers with fp32 activations on the quantised weights.
        model.set_dtype(mx.float32)
        data["tiny_precision"] = "G4/G5 with fp32 activations (model.set_dtype(float32)); G0 dtypes as loaded"
    if ctx.want("g4"):
        dump = ref_pass.Dump(dump_dir(ctx, ts))
        res = g4_e2e.e2e(model, ts, dump, ctx.thresholds["G4"]["K8_backstop_decisive_lead_nats"], log=ctx.log)
        checks.update(g4_e2e.evaluate(res, arm, ctx.thresholds["G4"]))
    if ctx.want("g5"):
        if arm == "K8":
            conts = g5_generation.greedy_continuations(model, ts.g5, prof.g5_greedy_tokens)
            common.write_json(ctx.run_dir / "g5_continuations.json", {"continuations": conts})
            data["g5_continuations_sha256"] = common.sha256_file(ctx.run_dir / "g5_continuations.json")
            fd = g5_generation.floor_and_decode(model, {"T1": ts.ids["T1"], "T3": ts.ids["T3"], "T9": ts.t9},
                                                prof.decode_ranges, tuple(th5["noise_floor_prefill_steps"]), log=ctx.log)
            bound = g5_generation.parity_bound(fd, th5)
            data["noise_floor"] = fd
            checks["g5_decode_vs_prefill"] = _check(fd["decode_kl"] <= bound["kl_max"] and fd["decode_dis"] <= bound["dis_max"],
                                                    fd, bound)
        else:
            k8 = common.read_json(ctx.run_dir / "phase4.json") if (ctx.run_dir / "phase4.json").is_file() else {}
            fd = k8.get("data", {}).get("noise_floor")
            bound = g5_generation.parity_bound(fd, th5) if fd else None
        order = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
        prompts = [ts.ids[order[j % 8]][: prof.batch_lengths[j]] for j in range(len(prof.batch_lengths))]
        bp = g5_generation.batch_parity(model, prompts, list(prof.batch_max_tokens), th5["batch_B"])
        if arm == "K8":
            ok = bp["mean_kl"] <= bound["kl_max"] and bp["top1_dis"] <= bound["dis_max"] and bp["admitted_mid_run"] > 0 \
                and bp["max_live"] <= th5["batch_B"]
            checks["g5_batch_parity_K8"] = _check(ok, bp, {**bound, "admitted_mid_run_min": 1, "B": th5["batch_B"]})
        else:
            checks[f"g5_batch_parity_{arm}"] = _check(None, bp, bound, descriptive=True)
        checks[f"g5_behaviour_{arm}"] = _behaviour(ctx, model, arm, ts, th5)
    del model
    mx.clear_cache()
    return {"checks": checks, "data": data}


def _behaviour(ctx: GateContext, model, arm: str, ts, th5: dict) -> dict:
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    from gate.checks import g5_generation

    tokenizer = load_tokenizer(ctx.arm_dir(arm))
    if ctx.tiny:
        rng = np.random.Generator(np.random.PCG64(common.seed_from("tiny-behaviour")))
        vocab = common.read_json(ctx.bf16_dir / "config.json")["vocab_size"]
        items = [{"id": f"B{j:02d}", "lang": "en" if j < 10 else "de",
                  "prompt_ids": [int(x) for x in rng.integers(0, vocab - 16, size=12)], "prompt_text": ""}
                 for j in range(th5["behaviour_prompts"])]
        lang_tag = None
    else:
        prompts = common.read_json(common.BEHAVIOUR_PROMPTS)["prompts"]
        if len(prompts) != th5["behaviour_prompts"]:
            raise CouldNotRun(f"gate/behaviour_prompts.json holds {len(prompts)} prompts, thresholds say {th5['behaviour_prompts']}")
        items = [{"id": p["id"], "lang": p["lang"], "messages": [{"role": "user", "content": p["text"]}]} for p in prompts]
        from scorers.lang_tag import tag as lang_tag
    prof = ts.profile
    cells = {}
    for effort, cap in (("high", prof.behaviour_cap_high), ("none", prof.behaviour_cap_none)):
        cells[effort] = g5_generation.behaviour(model, tokenizer, items, arm, effort, cap, th5["batch_B"],
                                                tuple(th5["eos_ids"]), lang_tag, th5["loop_span_tokens"], th5["loop_repeats"])
    k = th5["behaviour_block_count"]
    blocked, summary = g5_generation.behaviour_verdict(cells, k)
    return _check(not blocked, {"cells": summary, "blocked_efforts": blocked, "rows": {e: c["rows"] for e, c in cells.items()}},
                  {"block_count": k, "loop": f"{th5['loop_span_tokens']} x {th5['loop_repeats']}", "eos_ids": th5["eos_ids"]},
                  applicable=not ctx.tiny)


def phase4(ctx: GateContext) -> dict:
    return _arm_phase(ctx, "K8")


def phase5(ctx: GateContext) -> dict:
    from gate.checks import g5_generation, ref_pass

    conts = common.read_json(ctx.run_dir / "g5_continuations.json")["continuations"]
    ts = textset_for(ctx)
    prompts = {p["id"]: p for p in ts.g5}
    seqs = [list(prompts[c["id"]]["ids"]) + list(c["continuation"]) for c in conts]
    rr = ref_pass.RefRunner(ctx.bf16_dir, "fp32")
    logits, _, _, lengths = rr.ref.forward_packed(seqs)
    passes = [{"what": "fp32 reference on the 8 G5 prompts + continuations", "bytes": common.checkpoint_bytes(ctx.bf16_dir)}]
    bounds = np.concatenate([[0], np.cumsum(lengths)])
    by_id = {c["id"]: logits[bounds[j]:bounds[j + 1]] for j, c in enumerate(conts)}
    th5 = ctx.thresholds["G5"]
    res = g5_generation.greedy_vs_ref(conts, prompts, by_id, th5["greedy_decisive_lead_nats"])
    ok = (res["decisive_top1"] is None or res["decisive_top1"] >= th5["greedy_decisive_top1_min"]) and \
        res["in_top5"] is not None and res["in_top5"] >= th5["greedy_ref_top5_min"]
    note = ("tiny: flat random logits leave no decisive positions, and 8-bit noise alone moves the greedy token "
            "out of the reference top-5 (tests/INTEGRATION_LOG.md entry 3); the driver runs, the verdict is not applied") if ctx.tiny else None
    return {"checks": {"g5_greedy": _check(ok, res, {k: th5[k] for k in ("greedy_decisive_lead_nats", "greedy_decisive_top1_min",
                                                                          "greedy_ref_top5_min", "greedy_tokens")},
                                           applicable=not ctx.tiny, note=note)},
            "data": {"reference_passes": passes}}


def phase6(ctx: GateContext) -> dict:
    return _arm_phase(ctx, "K4")


PHASE_FN = {1: phase1, 2: phase2, 3: phase3, 4: phase4, 5: phase5, 6: phase6}


def phases_to_run(ctx: GateContext) -> list[int]:
    ps = set()
    for g in ctx.checks:
        ps |= PHASES_FOR[g]
    if "K8" not in ctx.arms:
        ps -= {4, 5}
    if "K4" not in ctx.arms:
        ps -= {6}
    if 5 in ps and not ctx.want("g5"):
        ps.discard(5)
    return sorted(ps)


def run_phase(n: int, ctx: GateContext) -> dict:
    t0, utc0 = time.time(), common.utc_iso()
    out = PHASE_FN[n](ctx)
    out.update(phase=n, title=PHASE_TITLES[n], t_start=utc0, t_end=common.utc_iso(), wall_s=time.time() - t0,
               peak_rss_bytes=common.peak_rss_bytes())
    path = ctx.run_dir / f"phase{n}.json"
    common.write_new_json(path, out)
    return out


def _phase_subprocess(n: int, ctx: GateContext, argv: list[str]) -> None:
    cmd = [sys.executable, str(Path(__file__).resolve()), *argv, "--phase", str(n), "--utc", ctx.utc]
    # The child's stdout goes to our stderr: our last stdout line is the JSON verdict.
    p = subprocess.run(cmd, cwd=str(common.EXP_DIR), stdout=sys.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"phase {n} exited {p.returncode}")


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------


def arm_verdict(arm: str, checks: dict, tiny: bool) -> tuple[str, list, list]:
    """("PASS" | "FAIL" | "INCOMPLETE", failing ids, missing ids)."""
    failing, missing = [], []
    for cid in BLOCKING[arm]:
        c = checks.get(cid)
        if c is not None and tiny and c.get("applicable") is False:
            continue
        if c is None or c.get("pass") is None:
            missing.append(cid)
        elif c["pass"] is False:
            failing.append(cid)
    if failing:
        return "FAIL", failing, missing
    return ("INCOMPLETE" if missing else "PASS"), failing, missing


def exit_code_for(verdicts: dict) -> int:
    """0 both PASS, 4 K8 PASS and K4 FAIL, 1 K8 FAIL, 3 could not run."""
    k8, k4 = verdicts.get("K8"), verdicts.get("K4")
    if k8 == "FAIL":
        return 1
    if k8 != "PASS":
        return 3
    if k4 == "PASS":
        return 0
    if k4 == "FAIL":
        return 4
    return 3


def _hashes(ctx: GateContext, ts_key: str | None) -> dict:
    out = {
        "port": common.port_sha256(),
        "thresholds": common.thresholds_sha256(),
        "reference_tree": common.reference_tree_sha256(),
        "gate_code": common.gate_code_sha256(),
        "gate_text": common.gate_text_sha256(),
        "text_key": ts_key,
        "converted_manifest": {},
    }
    for arm in common.ARMS:
        d = ctx.arm_dir(arm)
        if (d / common.CONVERT_RECORD).is_file():
            out["converted_manifest"][arm] = common.converted_manifest_sha256(d)
    return out


def aggregate(ctx: GateContext, phases: list[dict], pre: dict, error: str | None, t0: str) -> dict:
    checks = {}
    for ph in phases:
        checks.update(ph.get("checks", {}))
    for cid, c in checks.items():
        c.setdefault("applicable", True)
        c.setdefault("measured", None)
        c.setdefault("threshold", None)
        c.setdefault("arms", [a for a in ("K8", "K4") if cid in BLOCKING[a]])
        c["blocking"] = bool(c["arms"]) and not c.get("descriptive", False)
    verdicts, failing, missing = {}, {}, {}
    for arm in ("K8", "K4"):
        if arm not in ctx.arms:
            verdicts[arm], failing[arm], missing[arm] = "INCOMPLETE", [], ["arm not run"]
            continue
        v, f, m = arm_verdict(arm, checks, ctx.tiny)
        if error is not None and v != "FAIL":
            v = "INCOMPLETE"
        verdicts[arm], failing[arm], missing[arm] = v, f, m
    try:
        ts_key = textset_for(ctx).key
    except Exception:
        ts_key = None
    rec = {
        "experiment": "exp_036",
        "gate_version": ctx.thresholds.get("version"),
        "mode": "tiny" if ctx.tiny else "real",
        "utc": ctx.utc,
        "t_start": t0,
        "t_end": common.utc_iso(),
        "host": common.host_label() if not ctx.tiny else "tiny (weight-free test)",
        "git": {"head": common.git_head(), "dirty": common.git_dirty()},
        "arms": list(ctx.arms),
        "checks_requested": list(ctx.checks),
        "verdict": {a: ("TINY" if ctx.tiny else v) for a, v in verdicts.items()},
        "failing": failing,
        "missing": missing,
        "checks": checks,
        "sha256": _hashes(ctx, ts_key),
        "tree_sha256_rule": common.TREE_SHA256_RULE,
        "preconditions": pre,
        "phases": [{k: ph.get(k) for k in ("phase", "title", "t_start", "t_end", "wall_s", "peak_rss_bytes")} for ph in phases],
        "error": error,
        "thresholds_version": ctx.thresholds.get("version"),
        "head_policy": next((ph["data"]["head_policy"] for ph in phases if "head_policy" in ph.get("data", {})), None),
        "fix_cycles_max": ctx.thresholds.get("fix_cycles_max"),
        "reference_passes": [p for ph in phases for p in ph.get("data", {}).get("reference_passes", [])],
    }
    rec["reference_streamed_bytes"] = sum(p["bytes"] for p in rec["reference_passes"])
    if ctx.tiny:
        rec["tiny_outcome"] = verdicts
        rec["mutant"] = ctx.mutant
    # A crash or an unmet precondition is never a PASS: exit 3 unless K8 already FAILed.
    rec["exit_code"] = 3 if error is not None and verdicts.get("K8") != "FAIL" else exit_code_for(verdicts)
    return rec


def _write_records(ctx: GateContext, rec: dict, phases: list[dict]) -> Path:
    stem = f"gate_{ctx.utc}"
    rows = []
    mutants = {}
    for ph in phases:
        rows += ph.get("data", {}).pop("layer_rows", []) if ph.get("phase") == 3 else []
        if ph.get("phase") == 3:
            mutants.update(ph["data"].get("mutants", {}))
        if ph.get("phase") == 2:
            mutants["g3_reference_mutants"] = (ph["checks"].get("g3_ref_mutants") or {}).get("measured")
    files = {}
    if rows:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
        p = ctx.gate_dir / f"{stem}_layers.csv"
        with open(p, "x", encoding="utf-8") as f:
            f.write(buf.getvalue())
        files["layers_csv"] = p.name
    if mutants:
        p = common.write_new_json(ctx.gate_dir / f"{stem}_mutants.json", mutants)
        files["mutants_json"] = p.name
    rec["files"] = files
    rec["phase_files"] = {f"phase{ph['phase']}": common.sha256_file(ctx.run_dir / f"phase{ph['phase']}.json")
                          for ph in phases if (ctx.run_dir / f"phase{ph['phase']}.json").is_file()}
    return common.write_new_json(ctx.gate_dir / f"{stem}.json", rec)


def run_all(ctx: GateContext, argv: list[str] | None = None) -> GateResult:
    """Run the requested phases and write the gate record (BUILD_SPEC §5.3)."""
    common.set_offline_env()
    if not ctx.tiny:
        try:  # BUILD_SPEC §2 Identity: before anything is written under results/
            common.require_identity()
        except (Exception, SystemExit) as e:
            rec = {"verdict": {"K8": "INCOMPLETE", "K4": "INCOMPLETE"}, "failing": {}, "exit_code": 3,
                   "error": f"could not run: git identity: {e}"}
            return GateResult(rec, None, 3)
    while ctx.run_dir.exists() or (ctx.gate_dir / f"gate_{ctx.utc}.json").exists():
        time.sleep(0.25)  # one record per UTC second; never overwrite (BUILD_SPEC §2)
        ctx.utc = common.utc_stamp()
    t0 = common.utc_iso()
    ctx.run_dir.mkdir(parents=True, exist_ok=False)
    phases, error, pre, n = [], None, {}, None
    try:
        pre = preconditions(ctx)
        for n in phases_to_run(ctx):
            ctx.log(f"[gate] phase {n}: {PHASE_TITLES[n]}")
            if ctx.subprocess_phases and argv is not None:
                _phase_subprocess(n, ctx, argv)
                phases.append(common.read_json(ctx.run_dir / f"phase{n}.json"))
            else:
                phases.append(run_phase(n, ctx))
            if not ctx.tiny:
                try:
                    import mlx.core as mx

                    mx.clear_cache()
                except Exception:
                    pass
    except CouldNotRun as e:
        error = f"could not run: {e}"
    except Exception as e:  # a crash is never a PASS; it is recorded and exits 3
        error = f"phase {n}: {type(e).__name__}: {e}" if n is not None else f"{type(e).__name__}: {e}"
        ctx.log(traceback.format_exc())
    rec = aggregate(ctx, phases, pre, error, t0)
    path = _write_records(ctx, rec, phases)
    return GateResult(rec, path, rec["exit_code"])


# ---------------------------------------------------------------------------
# Refusal (runner/guard.py require_gate calls this)
# ---------------------------------------------------------------------------


def latest_record(results_dir: Path = common.DEFAULT_RESULTS_DIR) -> Path | None:
    recs = sorted(p for p in (Path(results_dir) / "gate").glob("gate_*.json")
                  if p.name.count("_") == 1 and p.suffix == ".json")
    return recs[-1] if recs else None


def require_pass(arm: str, results_dir: Path = common.DEFAULT_RESULTS_DIR, models_dir: Path | None = None) -> GateRecord:
    """HYPOTHESIS Phase 0 "Verdict and refusal": refuse every Kolibri pilot,
    bench or scored run unless the latest results/gate/gate_<UTC>.json says
    PASS for `arm` and its sha256 of port/kolibri1.py, of the arm's converted
    manifest, of gate/thresholds.json and of reference/ equal the current
    ones; the converted directory's own kolibri1.py must be current too.
    Raises GateRefused (an exit-1 SystemExit) with the reason."""
    if arm not in common.ARMS:
        raise GateRefused(f"{arm} is not a Kolibri arm")
    path = latest_record(results_dir)
    if path is None:
        raise GateRefused(f"no gate record under {common.redact_path(Path(results_dir) / 'gate')}")
    rec = common.read_json(path)
    if rec.get("mode") != "real":
        raise GateRefused(f"{path.name} is a {rec.get('mode')} run, not a real gate run")
    if rec.get("verdict", {}).get(arm) != "PASS":
        raise GateRefused(f"{path.name}: {arm} verdict is {rec.get('verdict', {}).get(arm)!r}, not PASS")
    sha = rec.get("sha256", {})
    models = Path(models_dir) if models_dir is not None else common.models_dir()
    arm_dir = models / common.ARM_DIRS[arm]
    current = {
        "port": common.port_sha256(),
        "thresholds": common.thresholds_sha256(),
        "reference_tree": common.reference_tree_sha256(),
    }
    for key, value in current.items():
        if sha.get(key) != value:
            raise GateRefused(f"{path.name}: recorded {key} sha256 {str(sha.get(key))[:12]} != current {value[:12]}; "
                              "the gate must be re-run (HYPOTHESIS Phase 0)")
    if not (arm_dir / common.CONVERT_RECORD).is_file():
        raise GateRefused(f"{common.redact_path(arm_dir)}: no converted directory")
    manifest = common.converted_manifest_sha256(arm_dir)
    if sha.get("converted_manifest", {}).get(arm) != manifest:
        raise GateRefused(f"{path.name}: recorded {arm} manifest sha256 differs from {common.redact_path(arm_dir)}")
    from port.convert import StalePortFile, check_port_file

    try:
        check_port_file(arm_dir)
    except StalePortFile as e:
        raise GateRefused(str(e)) from e
    return GateRecord(path=path, arm=arm, record=rec)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_context(args) -> GateContext:
    thresholds = common.load_thresholds()
    checks = GROUPS if args.all or not args.checks else tuple(c.strip() for c in args.checks.split(",") if c.strip())
    bad = [c for c in checks if c not in GROUPS]
    if bad:
        raise SystemExit(f"unknown checks {bad}; use {','.join(GROUPS)}")
    arms = ("K8", "K4") if args.all or not args.arms else tuple(a.strip() for a in args.arms.split(","))
    if args.tiny:
        tiny = Path(args.tiny).expanduser().resolve()
        ctx = GateContext(models_dir=tiny, work_dir=tiny / "work", thresholds=thresholds, arms=arms, checks=checks,
                          tiny=True, results_dir=Path(args.results_dir or tiny / "results"),
                          tok_dir=Path(args.tok) if args.tok else None, mutant=args.port_mutant,
                          head_policy=args.head_policy, g1_select=args.g1_select or ["test_routing.py"],
                          subprocess_phases=args.subprocess_phases)
    else:
        if args.port_mutant or args.head_policy or args.results_dir:
            raise SystemExit("--port-mutant, --head-policy and --results-dir are for --tiny runs only")
        ctx = GateContext(models_dir=common.models_dir(), work_dir=common.work_dir(), thresholds=thresholds, arms=arms,
                          checks=checks, tok_dir=Path(args.tok) if args.tok else None, subprocess_phases=True)
    if args.utc:
        ctx.utc = args.utc
    return ctx


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="exp_036 Phase 0 gate")
    ap.add_argument("--all", action="store_true", help="every check, arms K8 and K4")
    ap.add_argument("--checks", help="comma list of g0,g1,g2,g3,g4,g5")
    ap.add_argument("--arms", help="comma list of K8,K4")
    ap.add_argument("--tiny", help="run on a tiny checkpoint directory (tests)")
    ap.add_argument("--tok", help="Kolibri tokenizer directory (default $EXP036_TOK or the BF16 snapshot)")
    ap.add_argument("--results-dir", help="tiny only: where results/gate goes (default DIR/results)")
    ap.add_argument("--port-mutant", help="tiny only: run the gate on gate/port_mutants.py's mutant")
    ap.add_argument("--head-policy", choices=("quantised_head", "vendor_faithful"), help="tiny only")
    ap.add_argument("--g1-select", nargs="*", help="tiny only: test files for G1")
    ap.add_argument("--subprocess-phases", action="store_true", help="tiny only: phases as subprocesses (real runs always)")
    ap.add_argument("--phase", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--utc", help=argparse.SUPPRESS)
    ap.add_argument("--require-pass", metavar="ARM", help="exit 1 unless the latest gate record allows ARM")
    args = ap.parse_args(argv)
    common.set_offline_env()

    if args.require_pass:
        try:
            rec = require_pass(args.require_pass)
        except GateRefused as e:
            print(json.dumps({"require_pass": args.require_pass, "allowed": False, "reason": str(e)}))
            return 1
        print(json.dumps({"require_pass": args.require_pass, "allowed": True, "record": rec.path.name}))
        return 0

    ctx = build_context(args)
    if args.phase is not None:  # a phase subprocess of run_all
        run_phase(args.phase, ctx)
        return 0
    child_argv = [a for a in argv if a not in ("--subprocess-phases",)]
    res = run_all(ctx, child_argv if ctx.subprocess_phases else None)
    print(json.dumps({"verdict": res.record["verdict"], "exit_code": res.exit_code,
                      "record": common.redact_path(res.path) if res.path else None,
                      "failing": res.record["failing"], "error": res.record["error"]}))
    return res.exit_code


if __name__ == "__main__":
    sys.exit(main())
