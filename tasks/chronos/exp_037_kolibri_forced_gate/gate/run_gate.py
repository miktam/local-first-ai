# SPDX-License-Identifier: MIT
"""Phase 0 driver: the exp_037 port-correctness gate (DESIGN §3, §4.2, §9; BUILD_SPEC §5.3; build task W7).

Phases (each a separate process in a real run, writing results/gate/<UTC>/phase<N>.json):
  0   preconditions P1-P4: hashes and freeze (real mode), the environment binding of
      decision F2 (P2), G0k at the §3.1 P3 shapes (P3), build identity (P4); and
      exp_036's could-not-run checks (directories, port copies, tokenizer, T5/T6/T9)
  1   G0 static (census, strict loads, template and tokenizer parity, converted
      configs, same port sha) and G1 (the weight-free suite minus g1_synthetic.EXCLUDE)
  2a  R1: exp_036's fp32 dump under its registered key, reused (or recomputed under
      $EXP036_WORK/exp037); the emulation statistics (emu_stats_exp037.json); G3
  2b  P5, harness integrity: P5a reference driver bitwise on layers {0, 4}, P5b
      ForcedTrunk exactness and P5c forcing active on the K8 fp32 port, T1. No
      real-weight port-against-reference value is computed before P5 passes
  2c  R2F and R3: the fp32 reference and its bf16 emulation on the dequantised K8,
      forced onto I (keyed under $EXP036_WORK/exp037/ref_forced, reused by key)
  3   G2 per layer (the registered single call over T1-T8; T9; router margin; G2q;
      real-weight mutants)
  4   K8, bf16 as loaded: G0 dtypes, G4-N(i), noise floor and decode vs prefill,
      greedy continuations, G4-F16, G5-R1 (the runner at B = 1), G5-BP-lean at
      B 8 and 16, behaviour; then in place set_dtype(float32) +
      fp32_on_dequantised_weights: Csort, G4-F32, G5-D32. Then the required
      control mutants of gate/controls.json (1, 6, 19, 20, 26, 27; 27 is
      G5-BP-lean's at K8 B = 8, decision (b)) and, in tiny mode only, its probes
      (22 at K8 B = 8; DESIGN §4.2): the unmutated model is deleted and
      mx.clear_cache() runs before every mutant load; mutant 6 is loaded once,
      run in bf16 (G4-F16) and converted in place (G4-F32). The bf16 and the
      fp32 K8 never coexist
  5   K4: G0 dtypes, G4 descriptive, G5-R1, G5-BP-lean at B 8 and 16, behaviour
  6   the R1 anchor over every sequence phases 4 and 5 generated (both arms: greedy,
      G5-R1 K8 and K4, G5-BP-lean K8 and K4, mutant 27; probe 22 in tiny mode) and
      the R1-dependent reductions (gate/checks/g5_runner.reduce_with_r1)
  7   gate/rules.py's evaluation of every measured value, and the record (in process)
Phase 4 is dropped without K8, phase 5 without K4, phase 6 when neither ran or G5 is
not requested. The phases measure; every pass/fail decision is gate/rules.py's
(DESIGN §9.2), frozen with GATE_RULES_SHA256. Phase files keep non-finite floats
exactly ({"$nonfinite": "nan"}), so a NaN that a port produces reaches the rules as
a NaN and fails there.

Verdicts and exits (DESIGN §3.16; gate/rules.py arm_verdict, exit_code_for), first match wins:
  a precondition (P1-P5) failed: the run stops there              exit 3, not a gate run
  K8 FAIL                                                          exit 1
  an exception, or a blocking value missing (and K8 not FAIL)     exit 3, not a gate run
  K8 INCOMPLETE or K4 INCOMPLETE ("power not shown", "VOID")       exit 5
  K8 PASS, K4 FAIL                                                 exit 4
  K8 PASS, K4 PASS                                                 exit 0
On exit 3 the record is written and the run's phase files move to
aborted/<UTC>-gate/ with a NOTE.md (DESIGN §3.1, §9.4 item 3).
Interrupts (final review W15-03): Ctrl-C (KeyboardInterrupt), SIGTERM and SIGHUP
(GateInterrupted) and SystemExit during the phases are recorded like a crash
("interrupted in phase N: ..."; exit 3, or exit 1 if a K8 check had already failed),
the record is written with blind_phase_reached, an exit-3 run moves to aborted/, and
only then is the interrupt re-raised (the CLI prints the verdict line and exits with
the record's code). Signals that arrive while phase 7 writes the record are deferred
until it exists. A run killed outright (SIGKILL, an out-of-memory kill, a power cut)
leaves results/gate/<UTC>/ without a record: P1(d) then refuses every gate run until
`run_gate.py --close-orphans` has written its record from its phase files (and its
run.json) and moved it to aborted/.

Record: results/gate/gate_<UTC>.json (+ _layers.csv with rules.py's per-row verdicts,
_mutants.json), with exp_036's fields plus experiment "exp_037", allowed_B,
gate_rules_sha256, preconditions (P1-P5), run_counts, blind_phase_reached, controls,
probes (tiny mode: each probe's leg values and whether it fires, without a margin;
never an input to a verdict, an INCOMPLETE, allowed_B or the exit code; {} in real
mode), calibrators and g5_logprob_files (DESIGN §3.16), and code_hashes (P1(d)).

require_pass(arm) is the refusal the runner calls (runner/guard.py require_gate): the
latest record must say PASS for the arm, and its port, thresholds, reference, gate
rules and builds-manifest sha256 must equal the current ones; it returns the arm's
allowed_B with the record.

CLI:
  run_gate.py --all                                   every check, arms K8, K4
  run_gate.py [--checks g0,g1,...] [--arms K8,K4]
  run_gate.py --tiny DIR [...]                        the same drivers on a tiny
      checkpoint (DIR/Kolibri-1-BF16 and its DIR/Kolibri-1-MLX-{8,4}bit-g64
      conversions; work files under DIR/work/exp037): every rule is applied, the
      record says verdict "TINY" with the outcome in tiny_outcome, and the exit code
      follows the tiny outcome (the tests and tools/dry_run.py)
  run_gate.py --require-pass K8                       exit 1 unless the latest record allows K8
  run_gate.py --close-orphans [--tiny DIR]            record and move to aborted/ every run directory
      results/gate/<UTC>/ that has no record (a run killed before phase 7; W15-03)
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, replace
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from gate import common, rules

GROUPS = ("g0", "g1", "g2", "g3", "g4", "g5")
PHASES = ("0", "1", "2a", "2b", "2c", "3", "4", "5", "6", "7")
# Which phases each check group needs (phase 0 and phase 7 always run).
PHASES_FOR = {
    "g0": {"1", "4", "5"},
    "g1": {"1"},
    "g2": {"2a", "2b", "3"},
    "g3": {"2a"},
    "g4": {"2a", "2b", "2c", "4", "5"},
    "g5": {"2a", "2b", "4", "5", "6"},
}
PHASE_TITLES = {
    "0": "P1-P4: hashes and freeze, environment binding, G0k, build identity",
    "1": "G0 static, G1",
    "2a": "R1 (reuse), emulation statistics, G3",
    "2b": "P5: harness integrity",
    "2c": "R2F and R3 (forced references on the dequantised K8)",
    "3": "G2 per layer (fp32, bf16, T9, router margin, G2q, real-weight mutants)",
    "4": "K8: bf16 checks, then fp32 in place (Csort, G4-F32, G5-D32), then the control mutants",
    "5": "K4: G0 dtypes, G4 descriptive, G5-R1, G5-BP-lean, behaviour",
    "6": "R1 anchor over the G5 sequences of both arms, and the R1-dependent reductions",
    "7": "gate/rules.py evaluation and the record",
}
SUBPROCESS_PHASES = PHASES[:-1]  # phase 7 runs in the parent (no mlx)
# Re-exports: the one definition is gate/rules.py's (DESIGN §3.16, §9.2).
BLOCKING = rules.BLOCKING
SHARED_G0 = rules.SHARED_G0
arm_verdict = rules.arm_verdict
exit_code_for = rules.exit_code_for

EXPERIMENT = "exp_037"
NONFINITE = "$nonfinite"
TINY_NOT_APPLICABLE = {
    # Tiny random weights cannot exercise these (exp_036 tests/INTEGRATION_LOG.md entries 1-3; exp_036's tiny
    # run exempted G3, G5 greedy and behaviour). G5-R1 K8 carries R-greedy, the registered greedy rule re-applied
    # to the runner's tokens (DESIGN §3.13), so it is exempt as G5 greedy is; its R-anchor and R-parity legs and
    # mutant 26's control are still computed and recorded (and asserted by the tiny tests).
    "g3_bpb_per_text": "tiny: bits per byte of random weights on random ids mean nothing",
    "g3_bpb_vs_peers": "tiny: no peers",
    "g3_ref_mutants": "tiny: random weights cannot rank the reference mutants on NLL",
    "g5_greedy": ("tiny: flat random logits leave few decisive positions, and 8-bit noise alone moves the greedy "
                  "token out of the reference top-5 (tests/INTEGRATION_LOG.md entry 3); the driver runs, the "
                  "verdict is not applied"),
    "g5_r1_K8": ("tiny: its R-greedy leg is the G5 greedy rule, which tiny weights cannot exercise (as g5_greedy); "
                 "R-anchor, R-parity and mutant 26's control are recorded in its legs and controls"),
    "g5_behaviour_K8": "tiny: random weights neither stop nor speak a language",
    "g5_behaviour_K4": "tiny: random weights neither stop nor speak a language",
}


class GateRefused(SystemExit):
    """require_pass refuses (an exit-1 SystemExit carrying the reason)."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

    def __str__(self) -> str:
        return self.message


class CouldNotRun(RuntimeError):
    """The gate cannot run at all (exit 3): a directory, a port copy, a tokenizer, disk."""


class GateInterrupted(BaseException):
    """SIGTERM or SIGHUP during a gate run (final review W15-03). Raised in the main
    thread, like KeyboardInterrupt for SIGINT, so that run_all writes the record and
    moves an exit-3 run to aborted/ before the process ends; run_all then re-raises
    it with the result attached (`gate_result`)."""

    def __init__(self, signame: str):
        super().__init__(signame)
        self.signame = signame


INTERRUPT_SIGNALS = ("SIGTERM", "SIGHUP")   # SIGINT already raises KeyboardInterrupt
RUN_FILE = "run.json"                       # results/gate/<UTC>/run.json: what --close-orphans needs


def _install(names, handler) -> dict:
    saved = {}
    for name in names:
        sig = getattr(signal, name, None)
        if sig is None:
            continue
        try:
            saved[sig] = signal.signal(sig, handler)
        except ValueError:  # not the main thread: signals stay as they are
            break
    return saved


@contextlib.contextmanager
def _signals_raise():
    """For the phases: SIGTERM and SIGHUP raise GateInterrupted (W15-03)."""
    def handler(signum, frame):
        raise GateInterrupted(signal.Signals(signum).name)

    saved = _install(INTERRUPT_SIGNALS, handler)
    try:
        yield
    finally:
        for sig, h in saved.items():
            signal.signal(sig, h)


@contextlib.contextmanager
def _signals_deferred():
    """While phase 7 writes the record: SIGINT, SIGTERM and SIGHUP are noted, not acted
    on (W15-03); run_all raises the first one noted once the record exists. Handlers
    are functions, never SIG_IGN, so no phase subprocess could inherit an ignore."""
    pending: list[str] = []

    def handler(signum, frame):
        pending.append(signal.Signals(signum).name)

    saved = _install(("SIGINT",) + INTERRUPT_SIGNALS, handler)
    try:
        yield pending
    finally:
        for sig, h in saved.items():
            signal.signal(sig, h)


def _interrupt_name(e: BaseException) -> str:
    return e.signame if isinstance(e, GateInterrupted) else type(e).__name__


@dataclass
class GateContext:
    """BUILD_SPEC §5.3 GateContext(models_dir, work_dir, thresholds, arms); exp_037 adds
    builds_dir (DESIGN §2.8) and the tiny-only g0k_rows."""
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
    builds_dir: Path | None = None       # the Kolibri arm directories (None: models_dir)
    g0k_rows: tuple | None = None        # tiny only: G0k at these row counts (() = not run); None = §3.1 P3
    controls: bool = True                # run the control mutants of controls.json (phase 4)
    tiny_precision: str = "fp32"         # tiny only: "fp32" activations for the free-routing checks, or "bf16"

    @property
    def bf16_dir(self) -> Path:
        return Path(self.models_dir) / common.BF16_DIR

    @property
    def builds(self) -> Path:
        return Path(self.builds_dir) if self.builds_dir is not None else Path(self.models_dir)

    def arm_dir(self, arm: str) -> Path:
        """An arm's converted build, under builds_dir() (DESIGN §2.8)."""
        return self.builds / common.ARM_DIRS[arm]

    @property
    def work37(self) -> Path:
        """exp_037's own work files (DESIGN §2.9): $EXP036_WORK/exp037."""
        return Path(self.work_dir) / "exp037"

    @property
    def gate_dir(self) -> Path:
        return Path(self.results_dir) / "gate"

    @property
    def run_dir(self) -> Path:
        return self.gate_dir / self.utc

    @property
    def aborted_dir(self) -> Path:
        """aborted/<UTC>-gate/ beside results/ (DESIGN §3.1, §9.4 item 3)."""
        return Path(self.results_dir).parent / "aborted" / f"{self.utc}-gate"

    @property
    def mode(self) -> str:
        return "tiny" if self.tiny else "real"

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
    """What require_pass returns: the record that allows `arm`, with its allowed_B."""
    path: Path
    arm: str
    record: dict

    @property
    def verdict(self) -> str:
        return self.record["verdict"][self.arm]

    @property
    def allowed_B(self) -> list:
        """The arm's allowed_B (DESIGN §3.14); a record without one gives [1]."""
        v = (self.record.get("allowed_B") or {}).get(self.arm)
        return sorted(int(b) for b in v) if v else [1]

    def get(self, key, default=None):
        return self.record.get(key, default)


# ---------------------------------------------------------------------------
# Phase files: exact non-finite floats
# ---------------------------------------------------------------------------


def _wire(obj):
    """JSON-ready, with every non-finite float kept as {"$nonfinite": "nan" | "inf" | "-inf"}."""
    if isinstance(obj, dict):
        return {str(k): _wire(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_wire(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _wire(obj.tolist())
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else {NONFINITE: repr(f)}
    if isinstance(obj, Path):
        return common.redact_path(obj)
    return obj


def _unwire(obj):
    if isinstance(obj, dict):
        if len(obj) == 1 and NONFINITE in obj:
            return float(obj[NONFINITE])
        return {k: _unwire(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwire(v) for v in obj]
    return obj


def phase_path(ctx: GateContext, n: str) -> Path:
    return ctx.run_dir / f"phase{n}.json"


def read_phase(ctx: GateContext, n: str) -> dict | None:
    p = phase_path(ctx, n)
    return _unwire(common.read_json(p)) if p.is_file() else None


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------


def signed_off_policy(hypothesis: Path = common.EXP_DIR / "HYPOTHESIS.md") -> tuple[str, str]:
    """The head policy of the sign-off block: "vendor_faithful" if Andrei's
    entry mentions vendor-faithful, "quantised_head" otherwise (the
    pre-registered default, also when the entry is still empty or the file is
    not written yet; a real run without HYPOTHESIS.md fails P1 anyway)."""
    hypothesis = Path(hypothesis)
    if not hypothesis.is_file():
        return "quantised_head", "pre-registered default (no HYPOTHESIS.md)"
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


def needed_arms(ctx: GateContext) -> list:
    """The arms whose builds the requested checks load."""
    return [a for a in ctx.arms if any(ctx.want(g) for g in ("g0", "g2", "g4", "g5"))]


def p5_arm(ctx: GateContext) -> str:
    """P5b/P5c run on the K8 fp32 port (DESIGN §3.1); a K4-only run uses K4."""
    return "K8" if "K8" in ctx.arms else ctx.arms[0]


def registered_dump_dir(ctx: GateContext, ts) -> Path:
    """R1 under its registered key (DESIGN §2.9): $EXP036_WORK/ref/<key>/fp32, exp_036's dump."""
    from gate.checks.ref_pass import cache_dir

    return cache_dir("fp32", ts.key, ctx.bf16_dir, ctx.work_dir)


def dump_dir(ctx: GateContext, ts) -> Path:
    """R1's directory: the registered dump when it is complete under its key, else
    the recomputed one under work37 ($EXP036_WORK/exp037/ref/<key>/fp32)."""
    from gate.checks.ref_pass import Dump, cache_dir, reusable

    reg = registered_dump_dir(ctx, ts)
    if reusable(Dump(reg), ctx.bf16_dir, ts.key):
        return reg
    return cache_dir("fp32", ts.key, ctx.bf16_dir, ctx.work37)


def planned_dump_bytes(ctx: GateContext, ts) -> int:
    cfg = common.read_json(ctx.bf16_dir / "config.json")
    N = 8 * ts.profile.text_len + ts.profile.t9_len
    H, E, k, V, L = cfg["hidden_size"], cfg["num_experts"], cfg["num_experts_per_tok"], cfg["vocab_size"], cfg["num_hidden_layers"]
    per_layer = N * (3 * H * 4 + E * 4 + k * 8 + 4)
    return L * per_layer + N * V * 4 + 2 * N * H * 4


def planned_forced_bytes(ctx: GateContext, ts) -> int:
    """R2F and R3: per mode fp32 logits [N, V] and per layer the shadow [N, k] int64 and the gap [N] fp32."""
    cfg = common.read_json(ctx.bf16_dir / "config.json")
    N = 8 * ts.profile.text_len + ts.profile.t9_len
    k, V, L = cfg["num_experts_per_tok"], cfg["vocab_size"], cfg["num_hidden_layers"]
    return 2 * (N * V * 4 + L * N * (k * 8 + 4))


def _free_bytes(ctx: GateContext) -> int:
    Path(ctx.work_dir).mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(ctx.work_dir).free


def check_disk(ctx: GateContext, ts) -> dict:
    from gate.checks.ref_pass import Dump, reusable

    d = dump_dir(ctx, ts)
    if reusable(Dump(d), ctx.bf16_dir, ts.key):
        return {"dump_reusable": True}
    need = planned_dump_bytes(ctx, ts)
    free = _free_bytes(ctx)
    if free < need:
        raise CouldNotRun(f"$EXP036_WORK has {free / 1e9:.1f} GB free; the reference dumps need about {need / 1e9:.1f} GB")
    return {"dump_reusable": False, "planned_bytes": need, "free_bytes": free}


def work_label(ctx: GateContext, p: Path) -> str:
    """A work path as "$EXP036_WORK/<rel>" (no home path enters a record)."""
    try:
        return "$EXP036_WORK/" + Path(p).resolve().relative_to(Path(ctx.work_dir).resolve()).as_posix()
    except ValueError:
        return common.redact_path(p)


def release_memory() -> None:
    from gate.checks import g4_forced

    g4_forced.release_memory()


@contextlib.contextmanager
def tiny_env(ctx: GateContext):
    """A tiny run's exp_037 writers (work37_dir(): R1 if recomputed, R2F/R3, the
    G0k and G5 files) and builds_dir() resolve inside the tiny directory, and its
    paths are redacted as $EXP036_MODELS / $EXP036_WORK in what it writes: for the
    duration of the run $EXP036_MODELS is the tiny directory, $EXP036_WORK its
    work dir and $EXP037_BUILDS its builds dir (phase subprocesses inherit them).
    Real runs are untouched."""
    if not ctx.tiny:
        yield
        return
    want = {"EXP036_MODELS": str(Path(ctx.models_dir)), "EXP036_WORK": str(Path(ctx.work_dir)),
            "EXP037_BUILDS": str(ctx.builds)}
    saved = {k: os.environ.get(k) for k in want}
    os.environ.update(want)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ---------------------------------------------------------------------------
# Phase 0: preconditions P1-P4 (DESIGN §3.1)
# ---------------------------------------------------------------------------


def could_not_run_checks(ctx: GateContext) -> dict:
    """exp_036's preconditions: everything that makes a run impossible rather than failed (exit 3)."""
    info = {}
    if not ctx.bf16_dir.is_dir():
        raise CouldNotRun(f"{common.redact_path(ctx.bf16_dir)} is missing")
    port = common.port_sha256()
    for arm in needed_arms(ctx):
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


def g0k_thresholds(ctx: GateContext) -> tuple[dict, str | None]:
    """(thresholds for P3, note): the §3.1 P3 set; a tiny run with g0k_rows probes and
    judges only those counts ("rows" routing), and says so."""
    th = ctx.thresholds
    if not (ctx.tiny and ctx.g0k_rows):
        return th, None
    rows = sorted({int(r) for r in ctx.g0k_rows})
    block = dict(th["P3_G0k"], judged_rows={"boundary_mlx_repro_min": rows}, judged_rows_one_expert=[])
    return dict(th, P3_G0k=block), f"tiny run: G0k at the subset {rows} of the §3.1 P3 rows (g0k_rows)"


def run_g0k(ctx: GateContext) -> dict:
    """P3: gate/checks/g0k_kernels.probe at the judged shapes, judged by
    gate/preconditions.p3 (gate/rules.p3_g0k). The probe's full output is a work
    file ($EXP036_WORK/exp037/run/<UTC>/g0k.json); the phase keeps the judgement."""
    from gate import preconditions
    from gate.checks import g0k_kernels, ref_drivers

    if ctx.tiny and ctx.g0k_rows is not None and len(ctx.g0k_rows) == 0:
        return {"ok": True, "values": {"judged": False, "note": "tiny run with g0k_rows = (): G0k not run"}, "reason": None}
    th, note = g0k_thresholds(ctx)
    cfg = g0k_kernels.config_from_thresholds(th["P3_G0k"])
    ctx.log(f"[gate] G0k: {len(cfg['shapes'])} shapes x {len(cfg['projections'])} projections x {len(cfg['bits'])} bit widths")
    probe = g0k_kernels.probe(**cfg, log=ctx.log)
    path = ref_drivers.require_under_work37(ctx.work37 / "run" / ctx.utc / "g0k.json")
    common.write_new_json(path, probe)
    res = preconditions.p3(probe, th)
    res["values"]["file"] = {"path": work_label(ctx, path), "sha256": common.sha256_file(path)}
    if note:
        res["values"]["tiny_subset"] = note
    return res


def phase0(ctx: GateContext) -> dict:
    from gate import preconditions

    th = ctx.thresholds
    basic = could_not_run_checks(ctx)
    arms = needed_arms(ctx)
    pre = {"P1": preconditions.p1(results_dir=ctx.results_dir, builds=ctx.builds, mode=ctx.mode, current_utc=ctx.utc),
           "P2": preconditions.p2(th["P2"], mode=ctx.mode, results_dir=ctx.results_dir)}
    pre["P4"] = (preconditions.p4(th["P4"], builds=ctx.builds, mode=ctx.mode, arms=arms) if arms else
                 {"ok": True, "values": {"judged": False, "note": "no requested check loads a build"}, "reason": None})
    pre["P3"] = run_g0k(ctx)
    failed = [k for k in ("P1", "P2", "P3", "P4") if not pre[k]["ok"]]
    return {"preconditions": pre, "precondition_failed": bool(failed), "failed_preconditions": failed,
            "data": {"could_not_run_checks": basic,
                     "code_hashes": preconditions.code_hashes(common.EXP_DIR, ctx.builds)}}


# ---------------------------------------------------------------------------
# Phase 1: G0 static, G1 (measurement; rules in phase 7)
# ---------------------------------------------------------------------------


def phase1(ctx: GateContext) -> dict:
    from gate import tokenizer_lines
    from gate.checks import g0_static, g1_synthetic

    th0 = ctx.thresholds["G0"]
    data = {}
    if ctx.want("g0"):
        g0 = {}
        if ctx.tiny:
            from reference.kolibri_ref import expected_shapes, load_config, param_count

            cfg = load_config(str(ctx.bf16_dir))
            expected = {"tensor_count": len(expected_shapes(cfg)), "param_count": param_count(cfg),
                        "full_attention_layers": [i for i, t in enumerate(cfg.layer_types) if t == "full_attention"]}
        else:
            expected = {k: th0[k] for k in ("tensor_count", "param_count", "full_attention_layers")}
        g0["census"] = g0_static.census(ctx.bf16_dir, expected, ctx.results_dir, hash_shards=not ctx.tiny,
                                        oids=None if ctx.tiny else g0_static.lfs_oids())
        g0["expected"] = expected
        g0["strict_load"] = {name: g0_static.strict_load_check(d, ctx.mutant)
                             for name, d in [("source", ctx.bf16_dir)] + [(a, ctx.arm_dir(a)) for a in ctx.arms]}
        tok = ctx.tok_dir if ctx.tiny else (ctx.tok_dir or common.tokenizer_dir())
        if tok is not None and (Path(tok) / "tokenizer.json").is_file():
            g0["template"] = g0_static.template_parity(tok)
            groups = tokenizer_lines.committed_rule_lines()
            mmlu, mmlu_files = tokenizer_lines.mmlu_lines() if not ctx.tiny else ([], [])
            if mmlu:
                groups["mmlu"] = mmlu
            tk = g0_static.tokenizer_parity(tok, groups, _gate_texts_for_parity())
            tk["mmlu_files"] = mmlu_files or "none found under $EXP036_DATA (committed lines only)"
            g0["tokenizer"] = tk
        else:
            g0["template"] = g0["tokenizer"] = None
        policy, source = head_policy(ctx)
        data["head_policy"] = {"policy": policy, "source": source}
        g0["converted_config"] = {arm: g0_static.converted_config_check(ctx.arm_dir(arm), common.ARM_BITS[arm], policy, th0)
                                  for arm in ctx.arms}
        g0["port_shas"] = {"port": common.port_sha256(),
                           **{a: common.read_json(ctx.arm_dir(a) / common.CONVERT_RECORD).get("port_sha256")
                              for a in ctx.arms}}
        data["g0"] = g0
    if ctx.want("g1"):
        sel = ctx.g1_select if ctx.tiny else None
        # junit XML carries the host name and absolute paths: it stays a work file under
        # $EXP036_WORK/exp037 (DESIGN §2.9); the record keeps its counts and sha256 (BUILD_SPEC §2 Privacy).
        data["g1"] = g1_synthetic.run(ctx.work37 / "gate_g1" / f"junit_{ctx.utc}.xml", sel, require_all=not ctx.tiny)
    return {"data": data}


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


# ---------------------------------------------------------------------------
# Phase 2a: R1, emulation statistics, G3
# ---------------------------------------------------------------------------


def _strip_emu(emu: dict) -> dict:
    """The emulation statistics as the rules and the CSV read them (per-pair z lists stay in the work file)."""
    out = {k: v for k, v in emu.items() if k != "layers"}
    out["layers"] = [{k: v for k, v in lay.items() if k != "z_pairs"} for lay in emu["layers"]]
    return out


def phase2a(ctx: GateContext) -> dict:
    from gate.checks import g3_oracle, ref_pass

    ts = textset_for(ctx)
    data = {"disk": check_disk(ctx, ts)}
    seqs = ts.packed8() + [ts.t9]
    d = dump_dir(ctx, ts)
    rec = ref_pass.fp32_dump(ctx.bf16_dir, seqs, d, ts.key, log=ctx.log)
    data["dump"] = {"dir": work_label(ctx, d), "registered_key": d == registered_dump_dir(ctx, ts),
                    "reused": bool(rec.get("reused")), "record_sha256": common.sha256_file(d / "record.json"),
                    "files": {k: v["sha256"] for k, v in rec["files"].items()},
                    "summary": {k: rec[k] for k in ("num_tokens", "seq_lengths", "ref_version", "wall_s", "peak_rss_bytes")
                                if k in rec}}
    dump = ref_pass.Dump(d)
    per_pass = common.checkpoint_bytes(ctx.bf16_dir)
    passes = [{"what": "R1: fp32 reference, T1-T8 and T9 in one pass", "bytes": 0 if rec.get("reused") else per_pass}]
    if ctx.want("g2"):
        L = ts.profile.text_len
        path = ref_pass.emu_stats_path(ctx.bf16_dir, ts.key, work37=ctx.work37)
        try:
            emu = ref_pass.emu_stats_exp037(ctx.bf16_dir, dump, 8 * L, [L] * 8, path, log=ctx.log)
            data["emu"] = _strip_emu(emu)
            data["emu_file"] = {"path": work_label(ctx, path), "sha256": common.sha256_file(path)}
            passes.append({"what": "bf16-emulated reference fed R1's h_l, forced and natural (emu_stats_exp037)",
                           "bytes": 0 if emu.get("reused") else per_pass})
        except ref_pass.CapabilityMissing as e:
            data["emu"] = {"capability_missing": str(e)}
    if ctx.want("g3"):
        th3 = ctx.thresholds["G3"]
        g3 = {"bpb": g3_oracle.ref_bpb(dump, ts), "peers": g3_oracle.peer_bpb(ctx.results_dir) if not ctx.tiny else None}
        try:
            nll, path = ref_pass.mutant_nll(ctx.bf16_dir, ts.ids[th3["ref_mutant_text"]])
            g3["mutants"] = g3_oracle.ref_mutant_nll(nll, th3["ref_mutant_blocks"], th3["ref_mutant_dnll_quantile"])
            g3["mutant_path"] = path
            passes.append({"what": "7 reference mutants + base on T3, parallel streams", "bytes": per_pass})
        except ref_pass.CapabilityMissing as e:
            g3["mutants"] = None
            g3["mutants_missing"] = str(e)
        data["g3"] = g3
    data["reference_passes"] = passes
    return {"data": data}


# ---------------------------------------------------------------------------
# Phase 2b: P5 (DESIGN §3.1)
# ---------------------------------------------------------------------------


def phase2b(ctx: GateContext) -> dict:
    """P5a on R1's own pack (reference only), then P5b and P5c on the unmutated
    K8 fp32 port, T1 in one chunk. P5 tests the harness, never a port: a tiny
    mutant run checks it on the unmutated port too."""
    from gate import harness
    from gate.checks import g4_forced, ref_drivers

    th = ctx.thresholds
    t = th["P5"]
    ts = textset_for(ctx)
    d = dump_dir(ctx, ts)
    a = ref_drivers.p5a(ctx.bf16_dir, d, tuple(t["p5a_layers"]), log=ctx.log)
    arm = p5_arm(ctx)
    model, _, module = common.load_port(ctx.arm_dir(arm))
    try:
        g4_forced.make_fp32_port(model)
        b = harness.p5b_exactness(model, module, ts.ids[t["p5b_text"]])
        c = harness.p5c_forcing_active(model, module, ts.ids[t["p5c_text"]], min_mean_kl=t["p5c_mean_kl_min"])
    finally:
        del model, module
        release_memory()
    measured = {"p5a": {"bitwise": {str(v["layer"]): v["bitwise_equal"] for v in a["values"]["layers"].values()}},
                "p5b": {"bitwise": b["values"]["bitwise_equal"]},
                "p5c": {"mean_kl": c["values"]["mean_kl"]}}
    r = rules.p5(measured, th)
    res = {"ok": r["ok"], "values": {"rule": r, "measured": measured, "arm": arm, "port": "K8 fp32 port (unmutated)",
                                     "p5a": a["values"], "p5b": b["values"], "p5c": c["values"]},
           "reason": None if r["ok"] else "P5 failed: " + ", ".join(k for k, v in r["parts"].items() if not v["ok"])}
    passes = [{"what": f"P5a: reference layers {list(t['p5a_layers'])} on R1's pack",
               "bytes": int(common.checkpoint_bytes(ctx.bf16_dir) * len(t["p5a_layers"]) / max(1, _num_layers(ctx)))}]
    return {"preconditions": {"P5": res}, "precondition_failed": not r["ok"],
            "failed_preconditions": [] if r["ok"] else ["P5"], "data": {"reference_passes": passes}}


def _num_layers(ctx: GateContext) -> int:
    return int(common.read_json(ctx.bf16_dir / "config.json")["num_hidden_layers"])


# ---------------------------------------------------------------------------
# Phase 2c: R2F and R3 (DESIGN §3.0, §2.9)
# ---------------------------------------------------------------------------


def k8_shards(ctx: GateContext) -> dict:
    """{shard name: sha256} of the K8 build, from the convert record P4 checked the
    build against (real: exp_036's pinned record; tiny: the build's own), so the
    forced-dump key needs no second hash of the shards."""
    import fnmatch

    p4 = ctx.thresholds["P4"]
    if ctx.tiny:
        rec_path = ctx.arm_dir("K8") / common.CONVERT_RECORD
    else:
        rec_path = (common.EXP_DIR / p4["builds"]["K8"]["convert_record"]).resolve()
    files = common.read_json(rec_path).get("files") or {}
    out = {n: e["sha256"] for n, e in files.items() if fnmatch.fnmatch(n, p4.get("shards", "model*.safetensors"))}
    if not out:
        raise CouldNotRun(f"{common.redact_path(rec_path)} lists no K8 shard")
    return out


def pack_layout(ts) -> tuple[np.ndarray, list]:
    """(pack ids [N], segments) in R1's pack order: T1..T8, then T9."""
    from gate.checks import g4_forced
    from gate.checks.ref_pass import segments_of

    texts = g4_forced.text_layout(ts)
    return np.concatenate([ids for _, ids in texts]), segments_of([ids.size for _, ids in texts])


def forced_refs(ctx: GateContext, ts) -> dict:
    """I (from R1), and the keys and directories of R2F ("fp32") and R3 ("emu")."""
    from gate.checks import ref_drivers

    d = dump_dir(ctx, ts)
    I, i_sha = ref_drivers.i_from_r1(d, _num_layers(ctx))
    shards = k8_shards(ctx)
    out = {"I": I, "i_sha256": i_sha, "modes": {}}
    for mode in ref_drivers.MODES:
        key = ref_drivers.forced_dump_key(reference_tree_sha256=common.reference_tree_sha256(), k8_shards=shards,
                                          text_key=ts.key, i_sha256=i_sha, mode=mode)
        out["modes"][mode] = {"key": key, "dir": ref_drivers.forced_dump_dir(key, work=ctx.work37)}
    return out


def phase2c(ctx: GateContext) -> dict:
    from gate.checks import ref_drivers

    ts = textset_for(ctx)
    fr = forced_refs(ctx, ts)
    pack, segs = pack_layout(ts)
    data, passes = {"i_sha256": fr["i_sha256"]}, []
    pending = [m for m, e in fr["modes"].items() if not (e["dir"] / "record.json").is_file()]
    if pending:
        need = planned_forced_bytes(ctx, ts) * len(pending) // 2
        free = _free_bytes(ctx)
        if free < need:
            raise CouldNotRun(f"$EXP036_WORK has {free / 1e9:.1f} GB free; R2F and R3 need about {need / 1e9:.1f} GB")
    per_pass = common.checkpoint_bytes(ctx.arm_dir("K8"))
    for mode, e in fr["modes"].items():
        rec = ref_drivers.forced_ref_pass(ctx.bf16_dir, ctx.arm_dir("K8"), pack, segs, fr["I"], mode, e["dir"],
                                          key=e["key"], log=ctx.log)
        name = "R2F" if mode == "fp32" else "R3"
        data[name] = {"dir": work_label(ctx, e["dir"]), "key_sha256": e["key"]["sha256"], "key": e["key"]["components"],
                      "reused": bool(rec.get("reused")), "record_sha256": common.sha256_file(e["dir"] / "record.json"),
                      "wall_s": rec.get("wall_s")}
        passes.append({"what": f"{name}: forced reference ({mode}) on the dequantised K8, T1-T8 and T9",
                       "bytes": 0 if rec.get("reused") else per_pass})
    data["reference_passes"] = passes
    return {"data": data}


# ---------------------------------------------------------------------------
# Phase 3: G2
# ---------------------------------------------------------------------------


def phase3(ctx: GateContext) -> dict:
    from gate.checks import g2_layers, ref_pass

    ts = textset_for(ctx)
    p2 = read_phase(ctx, "2a") or {}
    emu = (p2.get("data") or {}).get("emu")
    if emu is not None and "capability_missing" in emu:
        emu = None
    dump = ref_pass.Dump(dump_dir(ctx, ts))
    res = g2_layers.run(ctx, dump, emu, ts, log=ctx.log)
    passes = [{"what": f"dequantised reference on {arm}", "bytes": common.checkpoint_bytes(ctx.arm_dir(arm))}
              for arm in ctx.arms if "capability_missing" not in res["layers_g2q"].get(arm, {"capability_missing": 1})]
    return {"data": {"g2": res, "reference_passes": passes}}


# ---------------------------------------------------------------------------
# Phases 4 and 5: the arms
# ---------------------------------------------------------------------------


def g5_settings(ctx: GateContext, ts) -> dict:
    """Generation settings of G5-R1 and G5-BP-lean: thresholds G5R1 (256 tokens,
    prefill step 2,048) and the runner's PREFILL_STEP_SIZE on a real run; the tiny
    profile's (greedy tokens, 64-token prefill chunk; DESIGN §5.1) on a tiny run."""
    from runner import generate

    t = ctx.thresholds["G5R1"]
    if ctx.tiny:
        p = ts.profile
        return {"r1_max_tokens": p.g5_greedy_tokens, "r1_step": p.prefill_chunk, "bp_step": p.prefill_chunk}
    return {"r1_max_tokens": int(t["max_tokens"]), "r1_step": int(t["prefill_step_size"]),
            "bp_step": int(generate.PREFILL_STEP_SIZE)}


def _behaviour_cells(ctx: GateContext, model, arm: str, ts, th5: dict) -> dict:
    from mlx_lm.tokenizer_utils import load as load_tokenizer

    from gate.checks import g5_generation

    tokenizer = load_tokenizer(ctx.arm_dir(arm))
    if ctx.tiny:
        rng = np.random.Generator(np.random.PCG64(common.seed_from("tiny-behaviour", prefix="exp036")))  # registered fixture seed, not an exp_037 draw
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
    return cells


def _load_arm(ctx: GateContext, arm: str, mutant: str | None = None):
    model, _, module = common.load_port(ctx.arm_dir(arm), mutant=mutant)
    return model, module


def _g5_dir(ctx: GateContext, arm: str, check_dir: str) -> Path:
    from gate.checks import g5_runner

    return g5_runner.g5_out_dir(ctx.utc, arm, check_dir)


def tiny_free_routing_fp32(ctx: GateContext, model, data: dict) -> None:
    """A tiny run feeds its free-routing checks (G4-N(i) and G4 K4, the noise floor and
    decode vs prefill, greedy, G5-R1, G5-BP-lean, behaviour) fp32 activations on the
    quantised weights, as exp_036's tiny run did: on tiny random weights bf16 noise is
    dominated by routing near-ties (tests/INTEGRATION_LOG.md entries 1-3). The forced
    checks are not affected: G4-F16 runs before this, as loaded, and G4-F32 and G5-D32
    run on the fp32 port. A real run never calls it (its arms run bf16 as loaded)."""
    import mlx.core as mx

    if not ctx.tiny or ctx.tiny_precision != "fp32":
        return
    model.set_dtype(mx.float32)
    data["tiny_precision"] = ("free-routing checks with fp32 activations (model.set_dtype(float32)); G0 dtypes and "
                              "G4-F16 as loaded (bf16)")


def arm_bf16(ctx: GateContext, arm: str, model, module, ts, data: dict, floor: dict | None,
             forced: "_ForcedK8 | None" = None) -> dict | None:
    """The bf16 checks of one arm, as loaded (DESIGN §12 W7 phases 4 and 5); forced:
    the K8's forced inputs (G4-F16). Returns the noise floor (K8) or `floor`."""
    from gate.checks import g0_static, g4_e2e, g5_generation, g5_runner, ref_pass

    th = ctx.thresholds
    th5 = th["G5"]
    prof = ts.profile
    if ctx.want("g0"):
        n = 256 if not ctx.tiny else 64
        data["g0_dtypes"] = g0_static.dtype_asserts(model, module, ts.ids["T1"][:n])
    if forced is not None:
        ctx.log("[gate] G4-F16: K8 bf16 forced onto I")
        data["g4_f16"] = forced.f16(model, module)
    tiny_free_routing_fp32(ctx, model, data)
    if ctx.want("g4"):
        dump = ref_pass.Dump(dump_dir(ctx, ts))
        data["g4_free"] = g4_e2e.e2e(model, ts, dump, th["G4"]["K8_backstop_decisive_lead_nats"], log=ctx.log)
    if ctx.want("g5") and arm == "K8":
        fd = g5_generation.floor_and_decode(model, {"T1": ts.ids["T1"], "T3": ts.ids["T3"], "T9": ts.t9},
                                            prof.decode_ranges, tuple(th5["noise_floor_prefill_steps"]), log=ctx.log)
        data["noise_floor"] = fd
        floor = {"floor_kl": fd["floor_kl"], "floor_dis": fd["floor_dis"]}
        conts = g5_generation.greedy_continuations(model, ts.g5, prof.g5_greedy_tokens)
        data["greedy"] = g5_runner.greedy_block(conts, ts.g5, arm="K8")
    if ctx.want("g5"):
        s = g5_settings(ctx, ts)
        data["g5_r1"] = g5_runner.g5_r1(model, ts.g5, floor, _g5_dir(ctx, arm, "g5_r1"), arm=arm,
                                        max_tokens=s["r1_max_tokens"], prefill_step_size=s["r1_step"], th=th, log=ctx.log)
        p26 = g5_runner.prompts26(ts, th)
        data["g5_bp"] = {str(B): g5_runner.g5_bp(model, p26, B, _g5_dir(ctx, arm, f"g5_bp_B{B}"), arm=arm,
                                                 prefill_step_size=s["bp_step"], log=ctx.log)
                         for B in th["G5BP"]["B"]}
        data["behaviour"] = _behaviour_cells(ctx, model, arm, ts, th5)
    return floor


class _ForcedK8:
    """The forced inputs shared by G4-F16, G4-F32 and their control mutants in
    phase 4: I, R2F, R3 (ForcedDumps) and the reference profile, computed once."""

    def __init__(self, ctx: GateContext, ts):
        from gate.checks import g4_forced
        from gate.checks.ref_drivers import ForcedDump

        self.ctx, self.ts = ctx, ts
        fr = forced_refs(ctx, ts)
        self.I = fr["I"]
        self.R2F, self.R3 = (ForcedDump(fr["modes"][m]["dir"]) for m in ("fp32", "emu"))
        for name, d in (("R2F", self.R2F), ("R3", self.R3)):
            if not d.complete:
                raise CouldNotRun(f"{name} is not complete under {work_label(ctx, d.dir)} (phase 2c)")
        self.chunk = ts.profile.prefill_chunk
        self.ref = g4_forced.reference_profile(g4_forced.text_layout(ts), self.R2F, self.R3, num_layers=len(self.I),
                                               log=ctx.log)
        self.doc_starts = g4_forced.t9_document_starts(ts, tok_dir=ctx.tok_dir) if not ctx.tiny else []

    def f16(self, model, module) -> dict:
        from gate.checks import g4_forced

        return g4_forced.g4_f16_run(model, self.ts, self.I, self.R2F, self.R3, self.chunk, module=module, ref=self.ref,
                                    th=self.ctx.thresholds, log=self.ctx.log)

    def f32(self, model, module, csort: bool) -> dict:
        from gate.checks import g4_forced

        return g4_forced.g4_f32_run(model, self.ts, self.I, self.R2F, self.R3, self.chunk, module=module, ref=self.ref,
                                    csort=csort, doc_starts=self.doc_starts, th=self.ctx.thresholds, log=self.ctx.log)


def control_checks(ctx: GateContext) -> list:
    """The K8 checks whose required control mutants run in phase 4 (controls.json)."""
    out = []
    if ctx.want("g4"):
        out += ["g4_f16_K8", "g4_f32_K8"]
    if ctx.want("g5"):
        out += ["g5_r1_K8", "g5_bp_K8_B8", "g5_d32_K8"]
    return out


def phase4(ctx: GateContext) -> dict:
    """K8 (DESIGN §12 W7 phase 4): bf16 as loaded, then the fp32 port in place, then
    the control mutants, one model resident at a time."""
    from gate.checks import g4_forced, g5_runner

    ts = textset_for(ctx)
    th = ctx.thresholds
    data = {"arm": "K8"}
    forced = _ForcedK8(ctx, ts) if ctx.want("g4") else None
    model, module = _load_arm(ctx, "K8", ctx.mutant)
    try:
        arm_bf16(ctx, "K8", model, module, ts, data, None, forced)
        if ctx.want("g4") or ctx.want("g5"):
            g4_forced.make_fp32_port(model)  # in place: the bf16 K8 is gone from here on
            data["fp32_port"] = "model.set_dtype(float32) + fp32_on_dequantised_weights (in place)"
        if ctx.want("g4"):
            ctx.log("[gate] Csort and G4-F32: the K8 fp32 port forced onto I")
            data["g4_f32"] = forced.f32(model, module, csort=True)
        if ctx.want("g5"):
            texts, ranges = g5_runner.d32_inputs(ts, th)
            data["g5_d32"] = g5_runner.g5_d32(model, module, texts, ranges, th=th, log=ctx.log)
    finally:
        del model, module
        release_memory()
    checks = control_checks(ctx)
    if checks and ctx.controls and ctx.mutant is None:
        data["mutants"] = run_controls(ctx, ts, checks, data, forced)
        probes = probes_to_run(ctx, checks)
        if probes:  # tiny mode only (controls.json "modes"); measured and reported, never required (DESIGN §4.2)
            try:
                data["probes"] = run_controls(ctx, ts, sorted({p["check"] for p in probes}), data, forced,
                                              controls={"required": probes})
            except Exception as e:  # a probe never decides the run, its exit code included: recorded, not raised
                ctx.log(traceback.format_exc())
                data["probes_error"] = f"{type(e).__name__}: {e}"
                release_memory()
    elif checks:
        data["mutants_not_run"] = ("a tiny run on a mutated port" if ctx.mutant is not None
                                   else "controls switched off (tiny run)")
    return {"data": data}


def probes_to_run(ctx: GateContext, checks: list, controls: dict | None = None) -> list:
    """controls.json's probes of `checks` that run in this run's mode (DESIGN §4.2:
    G5BP/22, mutant 22 on G5-BP-lean (d) at K8 B = 8, tiny mode only). A probe is
    measured and reported (the record's "probes"), never required: it enters no
    verdict, no INCOMPLETE, no allowed_B and no exit code."""
    controls = controls if controls is not None else rules.load_controls()
    return [p for p in controls.get("probes") or [] if p["check"] in checks and ctx.mode in (p.get("modes") or ())]


def run_controls(ctx: GateContext, ts, checks: list, data: dict, forced: "_ForcedK8 | None",
                 controls: dict | None = None) -> dict:
    """{check: {mutant id: measured}}: controls.json's required mutants of `checks`
    (or, with controls={"required": probes}, those probes), through
    gate/checks/g4_forced.mutant_loop: each loaded once, bf16 runs first, then
    converted in place for its fp32 runs; gc and mx.clear_cache() before every
    load. The G5 measures are tagged with the mutant actually loaded (27 for
    G5-BP-lean's control, 22 for its probe, 26 for G5-R1's), so each writes its
    own log-prob directory (g5_bp_m27, g5_bp_m22, g5_r1_m26) and block."""
    from gate import port_mutants
    from gate.checks import g4_forced, g5_runner

    th = ctx.thresholds
    floor = ({"floor_kl": data["noise_floor"]["floor_kl"], "floor_dis": data["noise_floor"]["floor_dis"]}
             if "noise_floor" in data else None)
    loaded = {}

    def load(name):
        loaded["id"] = port_mutants.NAME_TO_ID[name]
        return _load_arm(ctx, "K8", name)

    measures = {}
    if "g4_f16_K8" in checks:
        measures["g4_f16_K8"] = lambda model, module: forced.f16(model, module)
    if "g4_f32_K8" in checks:
        measures["g4_f32_K8"] = lambda model, module: forced.f32(model, module, csort=False)
    if "g5_r1_K8" in checks or "g5_bp_K8_B8" in checks:
        s = g5_settings(ctx, ts)
    if "g5_r1_K8" in checks:
        def m_r1(model, module):
            mid = loaded["id"]
            tiny_free_routing_fp32(ctx, model, {})  # as the unmutated K8's G5-R1 ran
            return g5_runner.g5_r1(model, ts.g5, floor, _g5_dir(ctx, "K8", g5_runner.check_dir_name("g5_r1", mutant=mid)),
                                   arm="K8", mutant=mid, max_tokens=s["r1_max_tokens"], prefill_step_size=s["r1_step"],
                                   th=th, log=ctx.log)
        measures["g5_r1_K8"] = m_r1
    if "g5_bp_K8_B8" in checks:
        p26 = g5_runner.prompts26(ts, th)

        def m_bp(model, module):
            mid = loaded["id"]
            tiny_free_routing_fp32(ctx, model, {})  # as the unmutated K8's G5-BP-lean ran
            return g5_runner.g5_bp(model, p26, 8, _g5_dir(ctx, "K8", g5_runner.check_dir_name("g5_bp", mutant=mid)),
                                   arm="K8", mutant=mid, prefill_step_size=s["bp_step"], log=ctx.log)
        measures["g5_bp_K8_B8"] = m_bp
    if "g5_d32_K8" in checks:
        texts, ranges = g5_runner.d32_inputs(ts, th)
        measures["g5_d32_K8"] = lambda model, module: g5_runner.g5_d32(model, module, texts, ranges,
                                                                       comparisons=("decode",), th=th, log=ctx.log)
    return g4_forced.mutant_loop(checks, load, measures, controls=controls, log=ctx.log)


def phase5(ctx: GateContext) -> dict:
    """K4: G0 dtypes, G4 descriptive, G5-R1 (R-anchor), G5-BP-lean, behaviour (bf16 as loaded)."""
    ts = textset_for(ctx)
    p4 = read_phase(ctx, "4") or {}
    nf = (p4.get("data") or {}).get("noise_floor")
    floor = {"floor_kl": nf["floor_kl"], "floor_dis": nf["floor_dis"]} if nf else None
    data = {"arm": "K4"}
    model, module = _load_arm(ctx, "K4", ctx.mutant)
    try:
        arm_bf16(ctx, "K4", model, module, ts, data, floor)
    finally:
        del model, module
        release_memory()
    return {"data": data}


# ---------------------------------------------------------------------------
# Phase 6: the R1 anchor (DESIGN §3.13-§3.14, G1 item 8)
# ---------------------------------------------------------------------------


def generating_phases(ctx: GateContext) -> list:
    """The phase files whose G5 blocks phase 6 scores: phase 4 (K8) and phase 5 (K4)."""
    return [p for p in (read_phase(ctx, "4"), read_phase(ctx, "5")) if p is not None]


def phase6(ctx: GateContext) -> dict:
    from gate.checks import g5_runner, ref_drivers

    pj = generating_phases(ctx)
    anchor = g5_runner.anchor_sequences(pj)
    data = {"anchor": {k: anchor[k] for k in ("n_sequences", "n_rows", "n_tokens")},
            "logprob_files": g5_runner.verify_logprob_files(pj)}
    if data["logprob_files"]["problems"]:
        raise g5_runner.G5DataError(f"log-prob files: {data['logprob_files']['problems'][:5]}")
    if anchor["n_sequences"] == 0:
        data["note"] = "no G5 sequence to score"
        data["reference_passes"] = []
        return {"data": data}
    out = ctx.work37 / "run" / ctx.utc / "r1_anchor"
    r1 = ref_drivers.r1_rows_pass(ctx.bf16_dir, anchor["seqs"], anchor["rows"], out_dir=out, log=ctx.log)
    data["r1"] = {k: r1["record"].get(k) for k in ("num_tokens", "num_rows", "pack_ids_sha256", "wall_s", "ref_version")}
    data["r1"]["dir"] = work_label(ctx, out)
    data["r1"]["record_sha256"] = common.sha256_file(out / "record.json")
    data["reduced"] = g5_runner.reduce_with_r1(pj, r1, anchor)
    data["reference_passes"] = [{"what": f"R1 anchor: fp32 reference on {anchor['n_sequences']} G5 sequences "
                                         f"({anchor['n_tokens']} tokens)", "bytes": common.checkpoint_bytes(ctx.bf16_dir)}]
    return {"data": data}


PHASE_FN = {"0": phase0, "1": phase1, "2a": phase2a, "2b": phase2b, "2c": phase2c, "3": phase3, "4": phase4,
            "5": phase5, "6": phase6}


def phases_to_run(ctx: GateContext) -> list[str]:
    """The phase table of DESIGN §12 W7: phase 4 is dropped without K8, phase 5
    without K4, phase 6 when neither ran or G5 is not requested; 2c needs K8."""
    ps = {"0", "7"}
    for g in ctx.checks:
        ps |= PHASES_FOR[g]
    if "K8" not in ctx.arms:
        ps -= {"4", "2c"}
    if "K4" not in ctx.arms:
        ps -= {"5"}
    if not ctx.want("g5") or not ({"4", "5"} & ps):
        ps.discard("6")
    return [p for p in PHASES if p in ps]


def run_phase(n: str, ctx: GateContext) -> dict:
    """Run phase n and write results/gate/<UTC>/phase<n>.json (a new file). A
    could-not-run condition is written too ({"could_not_run": reason}) and re-raised."""
    t0, utc0 = time.time(), common.utc_iso()
    meta = {"phase": n, "title": PHASE_TITLES[n], "t_start": utc0}
    try:
        out = PHASE_FN[n](ctx)
    except CouldNotRun as e:
        common.write_new_json(phase_path(ctx, n), _wire({**meta, "could_not_run": str(e), "t_end": common.utc_iso(),
                                                         "wall_s": time.time() - t0}))
        raise
    out.update(meta, t_end=common.utc_iso(), wall_s=time.time() - t0, peak_rss_bytes=common.peak_rss_bytes())
    common.write_new_json(phase_path(ctx, n), _wire(out))
    return out


def _phase_subprocess(n: str, ctx: GateContext, argv: list[str]) -> None:
    cmd = [sys.executable, str(Path(__file__).resolve()), *argv, "--phase", str(n), "--utc", ctx.utc]
    # The child's stdout goes to our stderr: our last stdout line is the JSON verdict.
    p = subprocess.run(cmd, cwd=str(common.EXP_DIR), stdout=sys.stderr)
    if p.returncode != 0:
        ph = read_phase(ctx, n)
        if ph is not None and "could_not_run" in ph:
            raise CouldNotRun(ph["could_not_run"])
        raise RuntimeError(f"phase {n} exited {p.returncode}")


# ---------------------------------------------------------------------------
# Phase 7: the rules (gate/rules.py) over the measured values
# ---------------------------------------------------------------------------


def _d(ph: dict | None) -> dict:
    return (ph or {}).get("data") or {}


def evaluate(ctx: GateContext, phases: dict) -> dict:
    """Every check of DESIGN §3 from the phase files, decided by gate/rules.py:
    {"checks", "layer_rows", "g5bp", "notes"}."""
    from gate.checks import ftiny, g2_layers

    th = ctx.thresholds
    th0, th5 = th["G0"], th["G5"]
    checks, notes, layer_rows, g5bp = {}, {}, [], None
    p1, p2a, p3, p4, p5, p6 = (_d(phases.get(n)) for n in ("1", "2a", "3", "4", "5", "6"))

    # G0 static and G1 (phase 1)
    if "g0" in p1:
        g0 = p1["g0"]
        checks["g0_census"] = rules.g0_census(g0["census"], g0["expected"], ctx.tiny)
        if ctx.tiny:
            checks["g0_census"]["note"] = "tiny: counts from the tiny config; shard sha not available"
        for name, sl in g0["strict_load"].items():
            checks[f"g0_strict_load_{name}"] = rules.g0_strict_load(sl)
        if g0["template"] is not None:
            checks["g0_template_parity"] = rules.g0_template_parity(g0["template"], th0)
            checks["g0_tokenizer_parity"] = rules.g0_tokenizer_parity(g0["tokenizer"], th0)
        else:
            for cid in ("g0_template_parity", "g0_tokenizer_parity"):
                checks[cid] = rules.missing("no tokenizer directory given", applicable=False)
        policy = p1["head_policy"]["policy"]
        for arm, cc in g0["converted_config"].items():
            checks[f"g0_converted_config_{arm}"] = rules.g0_converted_config(cc, policy)
        shas = dict(g0["port_shas"])
        port = shas.pop("port")
        checks["same_port_sha"] = rules.same_port_sha(port, shas)
    if "g1" in p1:
        checks["g1"] = rules.g1(p1["g1"], th["G1"], ctx.tiny)

    # G0 value-level dtypes (phases 4 and 5)
    for arm, pd in (("K8", p4), ("K4", p5)):
        if "g0_dtypes" in pd:
            checks[f"g0_router_bf16_exact_{arm}"], checks[f"g0_head_bf16_exact_{arm}"] = rules.g0_dtypes(pd["g0_dtypes"], th0)

    # G2 (phase 3) and its layers CSV
    if "g2" in p3:
        res = p3["g2"]
        emu = p2a.get("emu")
        if emu is not None and "capability_missing" in emu:
            reason = emu["capability_missing"]
            emu = None
        else:
            reason = "no emulation statistics"
        checks.update(rules.g2_checks(res, emu, th["G2"]))
        if emu is None:
            for cid in ("g2_bf16_forced_branch", "g2_bf16_natural_selection", "g2_t9_buckets", "g2_router_margin"):
                checks.setdefault(cid, rules.missing(reason))
        if "g2_router_margin" not in checks:
            checks["g2_router_margin"] = rules.missing(
                "the router-margin run is not made on a mutated port" if ctx.mutant else "no router-margin run")
        for arm in ctx.arms:
            if f"g2q_{arm}" not in checks:
                g = (res.get("layers_g2q") or {}).get(arm, {})
                checks[f"g2q_{arm}"] = rules.missing(g.get("capability_missing", "not run"))
        if ctx.mutant is not None:
            checks["g2_real_weight_mutants"] = rules.missing("not run on a mutated port", applicable=False)
        layer_rows = g2_layers.layer_csv_rows(res, emu, rules.g2_row_verdicts(res, emu, th["G2"]))

    # G3 (phase 2a)
    if "g3" in p2a:
        g3 = p2a["g3"]
        checks.update(rules.g3_checks(g3["bpb"], g3["peers"], g3["mutants"], th["G3"]))
        if g3["mutants"] is None:
            checks["g3_ref_mutants"] = rules.missing(g3.get("mutants_missing", "no mutant NLL pass"))

    # G4 (phases 4 and 5)
    if "g4_free" in p4:
        checks["g4_n_K8"] = rules.g4_n(p4["g4_free"], th)
    if "g4_free" in p5:
        checks["g4_K4"] = rules.g4_k4(p5["g4_free"], th)
    muts = p4.get("mutants") or {}
    if ctx.want("g4") and "K8" in ctx.arms and phases.get("4") is not None:
        checks["g4_f32_K8"] = rules.g4_f32(p4.get("g4_f32"), muts.get("g4_f32_K8"), ftiny.load_f_tiny(), th)
        checks["g4_f16_K8"] = rules.g4_f16(p4.get("g4_f16"), muts.get("g4_f16_K8"), th)

    # G5 (phases 4, 5, 6)
    red = p6.get("reduced") or {}
    if "noise_floor" in p4:
        checks["g5_decode_vs_prefill"] = rules.g5_decode_vs_prefill(p4["noise_floor"], th5)
    if red.get("g5_greedy") is not None:
        checks["g5_greedy"] = rules.g5_greedy(red["g5_greedy"], th5)
    if "g5_d32" in p4:
        checks["g5_d32_K8"] = rules.g5_d32(p4["g5_d32"], muts.get("g5_d32_K8"), th)
    if ctx.want("g5"):
        nf = p4.get("noise_floor")
        floor = {"floor_kl": nf["floor_kl"], "floor_dis": nf["floor_dis"]} if nf else None
        r1 = rules.g5_r1(red.get("g5_r1") or None, floor, red.get("g5_r1_mutants") or None, th)
        for arm in ctx.arms:
            checks[f"g5_r1_{arm}"] = r1[f"g5_r1_{arm}"]
        g5bp = rules.g5_bp(red.get("g5_bp") or None, red.get("g5_bp_mutants") or None, th)
        for cid, c in g5bp["checks"].items():
            if cid.split("_")[2] in ctx.arms:
                checks[cid] = c
        for arm, pd in (("K8", p4), ("K4", p5)):
            if "behaviour" in pd:
                cells = pd["behaviour"]
                checks[f"g5_behaviour_{arm}"] = rules.g5_behaviour(cells, th5, {e: c["rows"] for e, c in cells.items()})
    if g5bp is None:
        g5bp = rules.g5_bp(None, None, th)
        notes["allowed_B"] = "G5-BP-lean not run: allowed_B = {1} for both arms"

    # Tiny applicability (exp_036's precedent; tests/INTEGRATION_LOG.md entries 1-3)
    if ctx.tiny:
        for cid, why in TINY_NOT_APPLICABLE.items():
            if cid in checks:
                checks[cid]["applicable"] = False
                checks[cid]["note"] = why
    return {"checks": checks, "layer_rows": layer_rows, "g5bp": g5bp, "notes": notes}


# The phase-6 reduction that carries a probe's series, by the probe's check (controls.json "probes").
PROBE_SERIES = {"g5_bp_K8_B8": "g5_bp_mutants"}


def probes_outcome(ctx: GateContext, phases: dict, controls: dict | None = None) -> dict:
    """The record's "probes" (DESIGN §3.16, §4.2): every probe of controls.json that
    runs in this mode ({} in real mode, where none does), with its bound leg's values
    and whether that leg fires, as gate/rules.py's control_result computes them for
    the same leg of a required control; for G5BP/22 that is (d)'s ratio per subset at
    K8 B = 8. Reported without a margin. A probe enters no verdict, no INCOMPLETE, no
    allowed_B and no exit code: this runs after them, apart from evaluate(), and an
    error here is recorded in the probe's entry, never as the run's error."""
    controls = controls if controls is not None else rules.load_controls()
    th = ctx.thresholds
    factor = float(controls["margin_factor"])
    p4, p6 = _d(phases.get("4")), _d(phases.get("6"))
    out = {}
    for p in controls.get("probes") or []:
        if ctx.mode not in (p.get("modes") or ()):
            continue
        entry = {k: p.get(k) for k in ("id", "check", "arm", "B", "mutant", "mutant_name", "leg", "leg_type", "where",
                                       "modes", "reported", "enters")}
        entry.update(measured=False, fires=None, ratio=None, values=None, margin=None)
        src = (p6.get("reduced") or {}).get(PROBE_SERIES.get(p["check"], ""), {}) or {}
        m = src.get(str(p["mutant"]), src.get(int(p["mutant"])))
        if m is None:
            if "probes_error" in p4:
                entry["error"] = p4["probes_error"]
            entry["note"] = ("not reduced: phase 6 has no series for this probe" if "probes" in p4 else
                             "not run: " + (p4.get("mutants_not_run") or p4.get("probes_error") or "no K8 control run"))
            out[p["id"]] = entry
            continue
        entry["measured"] = True
        try:
            r = rules.control_result(p, m, {}, th, factor)
            entry.update(fires=r["caught"], values=r["values"],
                         ratio={sub: v.get("ratio") for sub, v in (r["values"] or {}).items()})
        except Exception as e:  # recorded with the probe; never the run's error (a probe is never required)
            entry["error"] = f"{type(e).__name__}: {e}"
        out[p["id"]] = entry
    return out


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def gate_rules_sha256() -> str:
    """GATE_RULES_SHA256: the frozen decision scope (DESIGN §9.2; tools/hash_tree.py "gate_rules")."""
    from tools import hash_tree as ht

    return ht.scope_value(ht.BY_NAME["gate_rules"])


def _hashes(ctx: GateContext, ts_key: str | None) -> dict:
    out = {
        "port": common.port_sha256(),
        "thresholds": common.thresholds_sha256(),
        "reference_tree": common.reference_tree_sha256(),
        "gate_code": common.gate_code_sha256(),
        "gate_text": common.gate_text_sha256(),
        "gate_rules": gate_rules_sha256(),
        "text_key": ts_key,
        "converted_manifest": {},
    }
    for arm in common.ARMS:
        d = ctx.arm_dir(arm)
        if (d / common.CONVERT_RECORD).is_file():
            out["converted_manifest"][arm] = common.converted_manifest_sha256(d)
    return out


def run_counts_for(ctx: GateContext, exit_code: int) -> dict:
    """The record's run_counts (DESIGN §3.16, §9.4): the real-mode exp_037 gate
    records so far, plus this run when it is a real one."""
    from gate import preconditions

    recs = [r for _, r in preconditions.gate_records(ctx.results_dir)]
    if not ctx.tiny:
        recs.append({"mode": "real", "experiment": EXPERIMENT, "exit_code": exit_code, "utc": ctx.utc})
    return preconditions.run_counts(recs, int(ctx.thresholds.get("fix_cycles_max", 2)))


def _preconditions_of(phases: dict) -> tuple[dict, list]:
    pre, failed = {}, []
    for n in ("0", "2b"):
        ph = phases.get(n) or {}
        pre.update(ph.get("preconditions") or {})
        failed += ph.get("failed_preconditions") or []
    return pre, failed


def aggregate(ctx: GateContext, phases: dict, error: str | None, t0: str) -> tuple[dict, list]:
    """Phase 7: (record, layer CSV rows)."""
    from gate.checks import ftiny, g5_runner

    pre, failed = _preconditions_of(phases)
    precondition_failed = bool(failed)
    ev = {"checks": {}, "layer_rows": [], "g5bp": None, "notes": {}}
    try:  # the values measured before a stop are judged and recorded too; the verdicts stay MISSING
        ev = evaluate(ctx, phases)
    except Exception as e:  # a rule that cannot read its input is a gate bug: exit 3
        ctx.log(traceback.format_exc())
        error = error or f"phase 7: {type(e).__name__}: {e}"
    checks = ev["checks"]
    for cid, c in checks.items():
        c.setdefault("applicable", True)
        c.setdefault("measured", None)
        c.setdefault("threshold", None)
        c["arms"] = rules.arms_of(cid)
        c["blocking"] = bool(c["arms"]) and c.get("state") != rules.DESCRIPTIVE and cid not in rules.NOT_BLOCKING
    verdicts, failing, missing, incomplete, reasons = {}, {}, {}, {}, {}
    for arm in rules.ARMS:
        if arm not in ctx.arms or precondition_failed:
            verdicts[arm], failing[arm], incomplete[arm] = rules.MISSING, [], []
            missing[arm] = ["arm not run"] if arm not in ctx.arms else ["precondition failed: " + ", ".join(failed)]
            reasons[arm] = {}
            continue
        v = rules.arm_verdict(arm, checks, ctx.tiny)
        verdict = v["verdict"]
        if error is not None and verdict != rules.FAIL:
            verdict = rules.MISSING  # a crash is never a PASS, nor an INCOMPLETE gate run
        verdicts[arm], failing[arm], missing[arm], incomplete[arm], reasons[arm] = (
            verdict, v["failing"], v["missing"], v["incomplete"], v["reasons"])
    exit_code = rules.exit_code_for(verdicts, precondition_failed=precondition_failed, error=error)
    try:
        ts_key = textset_for(ctx).key
    except Exception:
        ts_key = None
    g5bp = ev["g5bp"] or rules.g5_bp(None, None, ctx.thresholds)
    p0, p2b = phases.get("0") or {}, phases.get("2b") or {}
    blind = bool(((p2b.get("preconditions") or {}).get("P5") or {}).get("ok"))
    gen = [p for p in (phases.get("4"), phases.get("5")) if p is not None]
    try:
        lp_files = g5_runner.logprob_files(gen)
    except Exception as e:
        lp_files = [{"error": f"{type(e).__name__}: {e}"}]
    try:
        cal = rules.calibrators(checks)
        cal["f_tiny_source"] = {"path": "gate/calibration.json", "sha256": common.sha256_file(ftiny.CALIBRATION_PATH)}
    except Exception as e:
        cal = {"error": f"{type(e).__name__}: {e}"}
    try:  # after the verdicts, allowed_B and the exit code, which a probe never enters
        probes = probes_outcome(ctx, phases)
    except Exception as e:
        probes = {"error": f"{type(e).__name__}: {e}"}
    rec = {
        "experiment": EXPERIMENT,
        "gate_version": ctx.thresholds.get("version"),
        "rules_version": rules.RULES_VERSION,
        "mode": ctx.mode,
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
        "incomplete": incomplete,
        "reasons": reasons,
        "checks": checks,
        "allowed_B": g5bp["allowed_B"],
        "allowed_B_power_note": g5bp.get("power_note"),
        "sha256": _hashes(ctx, ts_key),
        "gate_rules_sha256": gate_rules_sha256(),
        "code_hashes": (p0.get("data") or {}).get("code_hashes"),
        "tree_sha256_rule": common.TREE_SHA256_RULE,
        "preconditions": pre,
        "precondition_failed": precondition_failed,
        "failed_preconditions": failed,
        "blind_phase_reached": blind,
        "controls": rules.controls_outcome(checks, g5bp),
        "probes": probes,
        "calibrators": cal,
        "g5_logprob_files": {"root": g5_runner.LOGPROB_ROOT, "files": lp_files},
        "phases": [{k: ph.get(k) for k in ("phase", "title", "t_start", "t_end", "wall_s", "peak_rss_bytes")}
                   for n, ph in phases.items() if ph is not None],
        "error": error,
        "exit_code": exit_code,
        "thresholds_version": ctx.thresholds.get("version"),
        "head_policy": _d(phases.get("1")).get("head_policy"),
        "fix_cycles_max": ctx.thresholds.get("fix_cycles_max"),
        "notes": ev["notes"],
        "reference_passes": [p for ph in phases.values() if ph is not None for p in _d(ph).get("reference_passes", [])],
    }
    rec["reference_streamed_bytes"] = sum(p["bytes"] for p in rec["reference_passes"])
    rec["run_counts"] = run_counts_for(ctx, exit_code)
    if ctx.tiny:
        rec["tiny_outcome"] = verdicts
        rec["mutant"] = ctx.mutant
    return rec, ev["layer_rows"]


def _mutants_file(phases: dict, checks: dict) -> dict:
    """gate_<UTC>_mutants.json: G2's real-weight mutants, router margin and forcing
    cross-check, and G3's reference mutants (exp_036's content)."""
    out = {}
    g2 = _d(phases.get("3")).get("g2")
    if g2 is not None:
        out.update({"real_weight_mutants": g2.get("real_weight_mutants"), "router_margin": g2.get("router_margin"),
                    "forcing_crosscheck": g2.get("forcing_crosscheck")})
        reg = (checks.get("g2_real_weight_mutants") or {}).get("measured") or {}
        detected = set(reg.get("detected") or [])
        for name, m in ((out["real_weight_mutants"] or {}).get("mutants") or {}).items():
            m["detected"] = name in detected
    if "g3" in _d(phases.get("2a")):
        out["g3_reference_mutants"] = (checks.get("g3_ref_mutants") or {}).get("measured")
    return out


def _note_md(ctx: GateContext, rec: dict) -> str:
    lines = [f"# Aborted gate run {ctx.utc}", "",
             "Exit 3: not a gate run and never a fix cycle (DESIGN §3.1, §9.4 item 3).", "",
             f"- Record: `results/gate/gate_{ctx.utc}.json`",
             f"- Mode: {rec['mode']}; arms {', '.join(rec['arms'])}; checks {', '.join(rec['checks_requested'])}",
             f"- Failed preconditions: {', '.join(rec['failed_preconditions']) or 'none'}",
             f"- Error: {rec['error'] or 'none'}",
             f"- Missing blocking values: {json.dumps(rec['missing'], sort_keys=True)}",
             f"- blind_phase_reached: {str(rec['blind_phase_reached']).lower()}"]
    if rec["blind_phase_reached"]:
        done = [p["phase"] for p in rec["phases"] if p.get("phase") in ("2c", "3", "4", "5", "6")]
        lines.append(f"- P5 had passed: port-against-reference values were computed in phases {', '.join(done) or 'none'}; "
                     "a gate fix after this run names them in its first line (DESIGN §9.4 item 3)")
    lines += ["", "The phase files of the run are in this directory."]
    return "\n".join(lines) + "\n"


def _write_records(ctx: GateContext, rec: dict, layer_rows: list, phases: dict) -> Path:
    """The record (results/gate/gate_<UTC>.json) with its CSV and mutants file; on
    exit 3 the run's phase files (and the CSV and mutants file) go to
    aborted/<UTC>-gate/ with a NOTE.md."""
    stem = f"gate_{ctx.utc}"
    aborted = rec["exit_code"] == 3
    side_dir = ctx.run_dir if aborted else ctx.gate_dir
    files = {}
    if layer_rows:
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(layer_rows[0]))
        w.writeheader()
        for r in layer_rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})
        p = side_dir / f"{stem}_layers.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "x", encoding="utf-8") as f:
            f.write(buf.getvalue())
        files["layers_csv"] = p.name
    mutants = _mutants_file(phases, rec["checks"])
    if mutants:
        p = common.write_new_json(side_dir / f"{stem}_mutants.json", mutants)
        files["mutants_json"] = p.name
    rec["files"] = files
    rec["phase_files"] = {f"phase{n}": common.sha256_file(phase_path(ctx, n)) for n in PHASES
                          if phase_path(ctx, n).is_file()}
    base = Path(ctx.results_dir).parent
    if aborted:
        dest = ctx.aborted_dir
        dest.parent.mkdir(parents=True, exist_ok=True)
        if ctx.run_dir.is_dir():
            shutil.move(str(ctx.run_dir), str(dest))
        dest.mkdir(parents=True, exist_ok=True)
        rec["phase_dir"] = dest.relative_to(base).as_posix()
        (dest / "NOTE.md").write_text(_note_md(ctx, rec), encoding="utf-8")
    else:
        rec["phase_dir"] = ctx.run_dir.relative_to(base).as_posix()
    return common.write_new_json(ctx.gate_dir / f"{stem}.json", rec)


def run_all(ctx: GateContext, argv: list[str] | None = None) -> GateResult:
    """Run the requested phases and write the gate record (BUILD_SPEC §5.3; DESIGN §12 W7)."""
    common.set_offline_env()
    if not ctx.tiny:
        try:  # BUILD_SPEC §2 Identity: before anything is written under results/
            common.require_identity()
        except (Exception, SystemExit) as e:
            rec = {"verdict": {"K8": rules.MISSING, "K4": rules.MISSING}, "failing": {}, "exit_code": 3,
                   "error": f"could not run: git identity: {e}"}
            return GateResult(rec, None, 3)
    while ctx.run_dir.exists() or (ctx.gate_dir / f"gate_{ctx.utc}.json").exists() or ctx.aborted_dir.exists():
        time.sleep(0.25)  # one record per UTC second; never overwrite (BUILD_SPEC §2)
        ctx.utc = common.utc_stamp()
    t0 = common.utc_iso()
    ctx.run_dir.mkdir(parents=True, exist_ok=False)
    common.write_new_json(ctx.run_dir / RUN_FILE, {"utc": ctx.utc, "t_start": t0, "mode": ctx.mode,
                                                   "arms": list(ctx.arms), "checks": list(ctx.checks)})
    phases, error, n, interrupt = {}, None, None, None
    with tiny_env(ctx):
        with _signals_raise():
            try:
                for n in phases_to_run(ctx):
                    if n == "7":
                        continue
                    ctx.log(f"[gate] phase {n}: {PHASE_TITLES[n]}")
                    if ctx.subprocess_phases and argv is not None:
                        _phase_subprocess(n, ctx, argv)
                    else:
                        run_phase(n, ctx)
                    phases[n] = read_phase(ctx, n)
                    try:
                        import mlx.core as mx

                        mx.clear_cache()
                    except Exception:
                        pass
                    if phases[n].get("precondition_failed"):
                        ctx.log(f"[gate] precondition failed: {', '.join(phases[n].get('failed_preconditions', []))}; "
                                "the run stops here (exit 3)")
                        break
            except CouldNotRun as e:
                error = f"could not run: {e}"
            except Exception as e:  # a crash is never a PASS; it is recorded and exits 3
                error = f"phase {n}: {type(e).__name__}: {e}" if n is not None else f"{type(e).__name__}: {e}"
                ctx.log(traceback.format_exc())
            except BaseException as e:  # noqa: B036 -- W15-03: Ctrl-C, SIGTERM, SIGHUP, SystemExit: recorded first
                interrupt = e
                error = (f"interrupted in phase {n}: {_interrupt_name(e)}" if n is not None
                         else f"interrupted: {_interrupt_name(e)}")
                ctx.log(f"[gate] {error}; the record is written before the process ends")
        with _signals_deferred() as pending:
            t7, t7_iso = time.time(), common.utc_iso()
            rec, layer_rows = aggregate(ctx, phases, error, t0)
            rec["phases"].append({"phase": "7", "title": PHASE_TITLES["7"], "t_start": t7_iso,
                                  "t_end": common.utc_iso(), "wall_s": time.time() - t7,
                                  "peak_rss_bytes": common.peak_rss_bytes()})
            path = _write_records(ctx, rec, layer_rows, phases)
    res = GateResult(rec, path, rec["exit_code"])
    if interrupt is None and pending:  # a signal while the record was written: the record stands
        ctx.log(f"[gate] {pending[0]} while the record was written; the record stands")
        interrupt = KeyboardInterrupt() if pending[0] == "SIGINT" else GateInterrupted(pending[0])
    if interrupt is not None:
        interrupt.gate_result = res
        raise interrupt
    return res


def close_orphans(ctx: GateContext) -> list[GateResult]:
    """Close every run directory results/gate/<UTC>/ that has no record (final review
    W15-03): a run killed before phase 7 (SIGKILL, an out-of-memory kill, a power
    cut, a kernel panic). Its phase files are judged by phase 7 as they stand, with
    the error "interrupted", so it is exit 3 (or exit 1, a gate run, if a K8 check
    had already failed), with blind_phase_reached from its own P5 result; an exit-3
    run moves to aborted/<UTC>-gate/ with its NOTE.md. P1(d) refuses every real
    gate run while such a directory exists. `ctx` gives the mode and the results
    directory; arms and checks come from the run's run.json."""
    from gate import preconditions

    if not ctx.tiny:
        common.require_identity()
    out = []
    for utc in preconditions.orphaned_runs(ctx.results_dir):
        c = replace(ctx, utc=utc)
        try:
            man = common.read_json(c.run_dir / RUN_FILE)
        except Exception:
            man = {}
        if man.get("mode") not in (None, c.mode):
            ctx.log(f"[gate] {utc}: a {man['mode']} run; not closed by this {c.mode}-mode call")
            continue
        if man.get("arms"):
            c.arms = tuple(man["arms"])
        if man.get("checks"):
            c.checks = tuple(man["checks"])
        phases = {}
        for n in SUBPROCESS_PHASES:
            try:
                ph = read_phase(c, n)
            except Exception:  # a phase file cut short by the kill is not a measurement
                ph = None
            if ph is not None:
                phases[n] = ph
        error = (f"interrupted: the run wrote no record (killed before phase 7); closed by --close-orphans at "
                 f"{common.utc_iso()}")
        t7, t7_iso = time.time(), common.utc_iso()
        rec, layer_rows = aggregate(c, phases, error, man.get("t_start"))
        rec["phases"].append({"phase": "7", "title": PHASE_TITLES["7"], "t_start": t7_iso, "t_end": common.utc_iso(),
                              "wall_s": time.time() - t7, "peak_rss_bytes": common.peak_rss_bytes()})
        path = _write_records(c, rec, layer_rows, phases)
        ctx.log(f"[gate] closed {utc}: exit {rec['exit_code']}, phase files in {rec['phase_dir']}")
        out.append(GateResult(rec, path, rec["exit_code"]))
    return out


# ---------------------------------------------------------------------------
# Refusal (runner/guard.py require_gate calls this)
# ---------------------------------------------------------------------------


def latest_record(results_dir: Path = common.DEFAULT_RESULTS_DIR) -> Path | None:
    recs = sorted(p for p in (Path(results_dir) / "gate").glob("gate_*.json")
                  if p.name.count("_") == 1 and p.suffix == ".json")
    return recs[-1] if recs else None


def require_pass(arm: str, results_dir: Path = common.DEFAULT_RESULTS_DIR, models_dir: Path | None = None,
                 builds_dir: Path | None = None) -> GateRecord:
    """HYPOTHESIS Phase 0 "Verdict and refusal" (DESIGN §3.16): refuse every
    Kolibri pilot, bench or scored run unless the latest
    results/gate/gate_<UTC>.json is a real-mode exp_037 record that says PASS
    for `arm`, and its sha256 of port/kolibri1.py, gate/thresholds.json,
    reference/, the gate rules (GATE_RULES_SHA256) and the arm's converted
    manifest under builds_dir() equal the current ones; the build's own
    kolibri1.py must be current too. Returns the record with the arm's
    allowed_B. Raises GateRefused (an exit-1 SystemExit) with the reason.
    models_dir is the older name of builds_dir (where the arm directories are)."""
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
    if rec.get("experiment") != EXPERIMENT:
        raise GateRefused(f"{path.name} is an {rec.get('experiment')!r} record, not an {EXPERIMENT} gate record")
    sha = rec.get("sha256", {})
    builds = Path(builds_dir or models_dir) if (builds_dir or models_dir) is not None else common.builds_dir()
    arm_dir = builds / common.ARM_DIRS[arm]
    current = {
        "port": common.port_sha256(),
        "thresholds": common.thresholds_sha256(),
        "reference_tree": common.reference_tree_sha256(),
    }
    for key, value in current.items():
        if sha.get(key) != value:
            raise GateRefused(f"{path.name}: recorded {key} sha256 {str(sha.get(key))[:12]} != current {value[:12]}; "
                              "the gate must be re-run (HYPOTHESIS Phase 0)")
    rules_now = gate_rules_sha256()
    if rec.get("gate_rules_sha256") != rules_now:
        raise GateRefused(f"{path.name}: recorded gate_rules sha256 {str(rec.get('gate_rules_sha256'))[:12]} != current "
                          f"{rules_now[:12]} (GATE_RULES_SHA256); the gate must be re-run")
    if not (arm_dir / common.CONVERT_RECORD).is_file():
        raise GateRefused(f"{common.redact_path(arm_dir)}: no converted directory")
    manifest = common.converted_manifest_sha256(arm_dir)
    if sha.get("converted_manifest", {}).get(arm) != manifest:
        raise GateRefused(f"{path.name}: recorded {arm} builds manifest sha256 differs from {common.redact_path(arm_dir)}")
    from port.convert import StalePortFile, check_port_file

    try:
        check_port_file(arm_dir)
    except StalePortFile as e:
        raise GateRefused(str(e)) from e
    return GateRecord(path=path, arm=arm, record=rec)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _g0k_rows_arg(values) -> tuple | None:
    if values is None:
        return None
    if values == ["none"]:
        return ()
    return tuple(int(v) for v in values)


def build_context(args) -> GateContext:
    thresholds = common.load_thresholds()
    checks = GROUPS if args.all or not args.checks else tuple(c.strip() for c in args.checks.split(",") if c.strip())
    bad = [c for c in checks if c not in GROUPS]
    if bad:
        raise SystemExit(f"unknown checks {bad}; use {','.join(GROUPS)}")
    arms = ("K8", "K4") if args.all or not args.arms else tuple(a.strip() for a in args.arms.split(","))
    bad = [a for a in arms if a not in common.ARMS]
    if bad:
        raise SystemExit(f"unknown arms {bad}; use K8,K4")
    if args.tiny:
        tiny = Path(args.tiny).expanduser().resolve()
        ctx = GateContext(models_dir=tiny, work_dir=tiny / "work", thresholds=thresholds, arms=arms, checks=checks,
                          tiny=True, results_dir=Path(args.results_dir or tiny / "results"),
                          tok_dir=Path(args.tok) if args.tok else None, mutant=args.port_mutant,
                          head_policy=args.head_policy, g1_select=args.g1_select or ["test_routing.py"],
                          subprocess_phases=args.subprocess_phases, builds_dir=tiny,
                          g0k_rows=_g0k_rows_arg(args.g0k_rows), controls=not args.no_controls,
                          tiny_precision=args.tiny_precision or "fp32")
    else:
        if (args.port_mutant or args.head_policy or args.results_dir or args.g0k_rows is not None or args.no_controls
                or args.tiny_precision):
            raise SystemExit("--port-mutant, --head-policy, --results-dir, --g0k-rows, --no-controls and "
                             "--tiny-precision are for --tiny runs only")
        ctx = GateContext(models_dir=common.models_dir(), work_dir=common.work_dir(), thresholds=thresholds, arms=arms,
                          checks=checks, tok_dir=Path(args.tok) if args.tok else None, subprocess_phases=True,
                          builds_dir=common.builds_dir())
    if args.utc:
        ctx.utc = args.utc
    return ctx


def main(argv=None) -> int:
    from tools.precision import ensure_exact_fp32
    ensure_exact_fp32()  # MLX_ENABLE_TF32=0 before any GPU work (tools/precision.py; Amendment 1)
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description="exp_037 Phase 0 gate")
    ap.add_argument("--all", action="store_true", help="every check, arms K8 and K4")
    ap.add_argument("--checks", help="comma list of g0,g1,g2,g3,g4,g5")
    ap.add_argument("--arms", help="comma list of K8,K4")
    ap.add_argument("--tiny", help="run on a tiny checkpoint directory (tests, the dry run)")
    ap.add_argument("--tok", help="Kolibri tokenizer directory (default $EXP036_TOK or the BF16 snapshot)")
    ap.add_argument("--results-dir", help="tiny only: where results/gate goes (default DIR/results)")
    ap.add_argument("--port-mutant", help="tiny only: run the gate on gate/port_mutants.py's mutant")
    ap.add_argument("--head-policy", choices=("quantised_head", "vendor_faithful"), help="tiny only")
    ap.add_argument("--g1-select", nargs="*", help="tiny only: test files for G1")
    ap.add_argument("--g0k-rows", nargs="*", help="tiny only: G0k at these row counts ('none': not run); "
                                                  "default the §3.1 P3 set")
    ap.add_argument("--no-controls", action="store_true", help="tiny only: skip the control mutants of phase 4")
    ap.add_argument("--tiny-precision", choices=("fp32", "bf16"), help="tiny only: activations of the free-routing "
                                                                      "checks (default fp32, exp_036's tiny run)")
    ap.add_argument("--subprocess-phases", action="store_true", help="tiny only: phases as subprocesses (real runs always)")
    ap.add_argument("--phase", help=argparse.SUPPRESS)
    ap.add_argument("--utc", help=argparse.SUPPRESS)
    ap.add_argument("--require-pass", metavar="ARM", help="exit 1 unless the latest gate record allows ARM")
    ap.add_argument("--close-orphans", action="store_true",
                    help="write the record of every run directory results/gate/<UTC>/ that has none (a run killed "
                         "before phase 7) and move an exit-3 one to aborted/; P1(d) refuses a gate run until then")
    args = ap.parse_args(argv)
    common.set_offline_env()

    if args.require_pass:
        try:
            rec = require_pass(args.require_pass)
        except GateRefused as e:
            print(json.dumps({"require_pass": args.require_pass, "allowed": False, "reason": str(e)}))
            return 1
        print(json.dumps({"require_pass": args.require_pass, "allowed": True, "record": rec.path.name,
                          "allowed_B": rec.allowed_B}))
        return 0

    ctx = build_context(args)
    if args.phase is not None:  # a phase subprocess of run_all
        if args.phase not in SUBPROCESS_PHASES:
            raise SystemExit(f"unknown phase {args.phase!r}; phases {', '.join(SUBPROCESS_PHASES)}")
        with tiny_env(ctx):
            try:
                run_phase(args.phase, ctx)
            except CouldNotRun as e:
                print(f"could not run: {e}", file=sys.stderr)
                return 3
        return 0
    if args.close_orphans:
        closed = close_orphans(ctx)
        print(json.dumps({"closed": [{"utc": r.record["utc"], "exit_code": r.exit_code,
                                      "record": common.redact_path(r.path), "phase_dir": r.record["phase_dir"],
                                      "blind_phase_reached": r.record["blind_phase_reached"]} for r in closed]}))
        return 0
    child_argv = [a for a in argv if a not in ("--subprocess-phases",)]
    interrupted = None
    try:
        res = run_all(ctx, child_argv if ctx.subprocess_phases else None)
    except (KeyboardInterrupt, GateInterrupted) as e:  # W15-03: the record exists; report it and exit with its code
        res = getattr(e, "gate_result", None)
        if res is None:
            raise
        interrupted = _interrupt_name(e)
    print(json.dumps({"verdict": res.record["verdict"], "exit_code": res.exit_code,
                      "record": common.redact_path(res.path) if res.path else None,
                      "failing": res.record["failing"], "error": res.record["error"],
                      **({"interrupted": interrupted} if interrupted else {})}))
    return res.exit_code


if __name__ == "__main__":
    sys.exit(main())
