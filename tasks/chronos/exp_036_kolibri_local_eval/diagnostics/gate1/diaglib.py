# SPDX-License-Identifier: MIT
"""Shared plumbing of the gate-1 diagnostic suite (exp_036, gate 20261005T050112Z).

Not a gate check and not a verdict. Read-only on the kit: it imports gate/,
reference/, runner/, bench/ and tools/, never edits them, and never invokes
gate/run_gate.py's record writer. Nothing here touches the network.

Every stage script calls tools.precision.ensure_exact_fp32() as its first
statement (before this module), then builds a Run:

    run = Run("stage1_g5", args, PREC)
    ...
    run.finish(record, work_only)

finish() writes two JSON files and a completion marker:
  * $EXP036_WORK/diag_gate1/<stage>_<UTC>.json   the full record (record + work_only)
  * diagnostics/gate1/out/<stage>_<UTC>.json     the restricted copy (record only),
    checked by leak_scan(): no token ids, no text, no host paths, and no key
    starting with "k8abs_" (absolute K8 quality numbers stay in the work file,
    DIAGNOSIS §4 "What may be committed")
  * $EXP036_WORK/diag_gate1/<stage>.done.json    what run_stages.sh resumes on

--tiny ROOT runs the same code on a tiny checkpoint root (ROOT/Kolibri-1-BF16,
ROOT/Kolibri-1-MLX-{8,4}bit-g64, ROOT/work with a reference dump): the smoke
test. Its outputs go under ROOT, never into the repository.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[1]
for p in (str(KIT), str(KIT / "tests"), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

from gate import common  # noqa: E402

common.set_offline_env()

GATE_UTC = "20261005T050112Z"
GATE_RECORD = KIT / "results" / "gate" / f"gate_{GATE_UTC}.json"
GATE_PHASES = KIT / "results" / "gate" / GATE_UTC
REPO_OUT = HERE / "out"
ORDER8 = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
FROZEN_RULES_CANDIDATES = (HERE / "FROZEN_RULES.md", KIT / "diagnostics" / "FROZEN_RULES.md")

# Values of the gate record the diagnostics reproduce (gate_20261005T050112Z.json).
RECORD = {
    "g5_batch_parity_K8_mean_kl": 0.3073773544462904,
    "g5_batch_parity_K8_top1_dis": 0.14285714285714285,
    "g5_batch_parity_K8_n_positions": 364,
    "g5_kl_max": 0.15129245844781955,
    "g5_dis_max": 0.19885577580314423,
    "g2_layer20": {"n_disagree": 225, "sigma": 0.0011611128129882606, "max_gap_over_sigma": 6.077697277069092},
    "g4_T9_top1_decisive": 0.98889449772842,
    "g4_T9_n_decisive": 5943,
}


def log(*x) -> None:
    print(time.strftime("%H:%M:%S"), *x, file=sys.stderr, flush=True)


def sha256_file(p) -> str | None:
    p = Path(p)
    return common.sha256_file(p) if p.is_file() else None


def frozen_rules_path() -> Path | None:
    return next((p for p in FROZEN_RULES_CANDIDATES if p.is_file()), None)


# ---------------------------------------------------------------------------
# Arguments and context
# ---------------------------------------------------------------------------


def parser(stage: str, doc: str) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=f"{stage}.py", description=doc.strip().split("\n\n")[0])
    ap.add_argument("--tiny", metavar="ROOT", help="smoke test: a tiny checkpoint root (outputs stay under ROOT)")
    ap.add_argument("--work-out", metavar="DIR", help="default $EXP036_WORK/diag_gate1 (tiny: ROOT/work/diag_gate1)")
    ap.add_argument("--repo-out", metavar="DIR", help="default diagnostics/gate1/out (tiny: ROOT/repo_out)")
    return ap


def gate_context(tiny: str | None):
    from gate import run_gate

    th = common.load_thresholds()
    if tiny:
        root = Path(tiny).expanduser().resolve()
        return run_gate.GateContext(models_dir=root, work_dir=root / "work", thresholds=th, tiny=True,
                                    results_dir=root / "results", head_policy="quantised_head",
                                    subprocess_phases=False)
    return run_gate.GateContext(models_dir=common.models_dir(), work_dir=common.work_dir(), thresholds=th,
                                subprocess_phases=False)


class Run:
    def __init__(self, stage: str, args, precision: dict):
        self.stage = stage
        self.args = args
        self.t0 = time.time()
        self.utc = common.utc_stamp()
        self.utc_start = common.utc_iso()
        self.tiny = bool(getattr(args, "tiny", None))
        self.ctx = gate_context(getattr(args, "tiny", None))
        if getattr(args, "work_out", None):
            self.work_root = Path(args.work_out).expanduser()
        else:
            self.work_root = Path(self.ctx.work_dir) / "diag_gate1"
        if getattr(args, "repo_out", None):
            self.repo_out = Path(args.repo_out).expanduser()
        elif self.tiny:
            self.repo_out = Path(self.ctx.models_dir) / "repo_out"
        else:
            self.repo_out = REPO_OUT
        self.scratch = self.work_root / stage
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.repo_out.mkdir(parents=True, exist_ok=True)
        self.precision = precision
        self._ts = None
        self._dump = None

    # -- inputs -------------------------------------------------------------

    def textset(self):
        if self._ts is None:
            from gate import run_gate

            self._ts = run_gate.textset_for(self.ctx)
        return self._ts

    def dump(self):
        """The gate's fp32 reference dump (T1-T8 and T9), required complete and reusable."""
        if self._dump is None:
            from gate import run_gate
            from gate.checks import ref_pass

            ts = self.textset()
            d = run_gate.dump_dir(self.ctx, ts)
            dump = ref_pass.Dump(d)
            if not ref_pass.reusable(dump, self.ctx.bf16_dir, ts.key):
                raise SystemExit(f"[{self.stage}] no complete reusable fp32 reference dump at {common.redact_path(d)}")
            self._dump = dump
        return self._dump

    def load_arm(self, arm: str = "K8", fp32: bool = False, dequantised_embedding: bool = False):
        """(model, module) as the gate loads it (common.load_port; the converted
        directory's own kolibri1.py must equal port/kolibri1.py). fp32: fp32
        activations, model.set_dtype(float32). dequantised_embedding (with fp32):
        see fp32_on_dequantised_weights."""
        import mlx.core as mx

        d = self.ctx.arm_dir(arm)
        if not self.tiny:
            from port.convert import check_port_file

            check_port_file(d)
        model, _, module = common.load_port(d)
        if fp32:
            model.set_dtype(mx.float32)
            if dequantised_embedding:
                fp32_on_dequantised_weights(model)
        mx.eval(model.parameters())
        return model, module

    @staticmethod
    def release() -> None:
        gc.collect()
        try:
            import mlx.core as mx

            mx.clear_cache()
        except Exception:
            pass

    def gate_record(self) -> dict | None:
        return None if self.tiny else common.read_json(GATE_RECORD)

    def phase(self, n: int) -> dict | None:
        return None if self.tiny else common.read_json(GATE_PHASES / f"phase{n}.json")

    # -- outputs ------------------------------------------------------------

    def header(self) -> dict:
        import mlx.core as mx

        try:
            import mlx_lm

            mlx_lm_v = mlx_lm.__version__
        except Exception:  # pragma: no cover
            mlx_lm_v = None
        rules = frozen_rules_path()
        info = mx.device_info()
        return {
            "experiment": "exp_036", "what": "gate-1 diagnostic (not a gate run, no gate record, not a fix cycle)",
            "stage": self.stage, "mode": "tiny" if self.tiny else "real", "utc": self.utc,
            "t_start": self.utc_start, "t_end": common.utc_iso(), "wall_s": time.time() - self.t0,
            "peak_rss_bytes": common.peak_rss_bytes(),
            "host": common.host_label() if not self.tiny else "tiny (smoke test)",
            "device": {"name": info.get("device_name"), "architecture": info.get("architecture"),
                       "memory_bytes": info.get("memory_size")},
            "mlx": mx.__version__, "mlx_lm": mlx_lm_v, "precision": self.precision,
            "git": {"head": common.git_head(), "dirty": common.git_dirty()},
            "gate_record": {"file": f"results/gate/gate_{GATE_UTC}.json", "sha256": sha256_file(GATE_RECORD)},
            "frozen_rules": {"file": rules.relative_to(KIT).as_posix() if rules else None,
                             "sha256": sha256_file(rules) if rules else None},
            "scripts_sha256": {p.name: sha256_file(p) for p in sorted(HERE.glob("*.py"))}
                              | {"run_stages.sh": sha256_file(HERE / "run_stages.sh")},
            "argv": ["<path>" if ("/" in a or "~" in a) else a for a in sys.argv[1:]],
        }

    def finish(self, record: dict, work_only: dict | None = None) -> dict:
        """Write the work file, the restricted repo copy and the marker."""
        head = self.header()
        restricted = common.jsonable({"header": head, **record})
        leak_scan(restricted)
        full = common.jsonable({"header": head, **record, "work_only": work_only or {}})
        name = f"{self.stage}_{self.utc}.json"
        wp = self.work_root / name
        rp = self.repo_out / name
        write_new(wp, full)
        write_new(rp, restricted)
        marker = {"stage": self.stage, "utc": self.utc, "work_json": common.redact_path(wp),
                  "repo_json": common.redact_path(rp), "repo_json_sha256": common.sha256_file(rp)}
        common.write_json(self.work_root / f"{self.stage}.done.json", marker)
        log(f"[{self.stage}] wrote {common.redact_path(wp)} and {common.redact_path(rp)}")
        print(json.dumps({"stage": self.stage, "ok": True, "repo_json": common.redact_path(rp)}), flush=True)
        return marker


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


def write_new(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n")
    return path


# ---------------------------------------------------------------------------
# Restricted-copy guard
# ---------------------------------------------------------------------------

FORBIDDEN_KEYS = {"token_id", "prev_token_id", "token_ids", "ids", "prompt_ids", "input_ids", "continuation",
                  "completion_ids", "tokens", "top_ref", "top_got", "top_tokens", "greedy_text", "decoded",
                  "wrapper_ids", "text_ids", "logits", "logprobs"}
_HOME = re.compile(r"/(Users|home)/[^/\s]+")


def _user() -> str | None:
    try:
        import getpass

        u = getpass.getuser()
        return u if u and len(u) >= 3 else None
    except Exception:
        return None


_USER = _user()


class LeakError(RuntimeError):
    pass


def leak_scan(obj, where: str = "", max_len: int = 400) -> None:
    """Refuse token ids, text, host paths and k8abs_* keys in a restricted record
    (strings must be ASCII and at most max_len characters: the scripts' own labels)."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in FORBIDDEN_KEYS or kl.startswith("k8abs_"):
                raise LeakError(f"restricted record: forbidden key {where}/{k}")
            leak_scan(v, f"{where}/{k}", max_len)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            leak_scan(v, f"{where}[{i}]", max_len)
    elif isinstance(obj, str):
        if len(obj) > max_len or not obj.isascii() or _HOME.search(obj) or (_USER and _USER in obj):
            raise LeakError(f"restricted record: string at {where} is long, non-ASCII or a host path")


# ---------------------------------------------------------------------------
# Numerics shared by the stages
# ---------------------------------------------------------------------------


def lsm(x) -> np.ndarray:
    return common.log_softmax64(np.asarray(x, dtype=np.float64))


def kl_lp(lp_p: np.ndarray, lp_q: np.ndarray) -> np.ndarray:
    """KL(p || q) per row from log-probs (float64)."""
    return (np.exp(lp_p) * (lp_p - lp_q)).sum(axis=-1)


def lead_lp(lp: np.ndarray) -> np.ndarray:
    part = np.partition(lp, -2, axis=-1)
    return part[:, -1] - part[:, -2]


def entropy_lp(lp: np.ndarray) -> np.ndarray:
    return -(np.exp(lp) * lp).sum(axis=-1)


def moving_block_boot(x: np.ndarray, y: np.ndarray | None = None, block: int = 64, B: int = 4000,
                      seed: int = 36, q: tuple = (0.025, 0.975)) -> dict:
    """Moving-block bootstrap of mean(x) (and of mean(x)/mean(y) when y is given),
    positions resampled in contiguous blocks of `block` (e1_analyse.py's scheme)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        return {"mean": float(x.mean()) if n else None, "ci": [None, None], "ratio": None, "ratio_ci": [None, None]}
    rng = np.random.Generator(np.random.PCG64(seed))
    blk = min(block, n)
    nb = int(math.ceil(n / blk))
    starts = rng.integers(0, max(1, n - blk + 1), size=(B, nb))
    idx = (starts[:, :, None] + np.arange(blk)[None, None, :]).reshape(B, -1)[:, :n]
    xm = x[idx].mean(axis=1)
    out = {"mean": float(x.mean()), "ci": [float(np.quantile(xm, q[0])), float(np.quantile(xm, q[1]))],
           "block": blk, "B": B, "q": list(q)}
    if y is not None:
        y = np.asarray(y, dtype=np.float64)
        ym = y[idx].mean(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = xm / ym
        r = r[np.isfinite(r)]
        out["ratio"] = float(x.mean() / y.mean()) if y.mean() != 0 else None
        out["ratio_ci"] = [float(np.quantile(r, q[0])), float(np.quantile(r, q[1]))] if r.size else [None, None]
    return out


def mcnemar_one_sided(n10: int, n01: int) -> float:
    """Exact one-sided McNemar p-value: P(X >= n10), X ~ Binomial(n10 + n01, 1/2)."""
    n = int(n10) + int(n01)
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(int(n10), n + 1))
    return float(tail / (2 ** n))


def pos_class(pos: int, window: int, chunk: int) -> str:
    """DIAGNOSIS §4 G2-D sign 2 position classes."""
    if pos <= 1:
        return "0-1"
    if window - 2 <= pos <= window + 2:
        return f"{window - 2}-{window + 2}"
    if chunk - 1 <= pos <= chunk + 1:
        return f"{chunk - 1}-{chunk + 1}"
    return "other"


def text_pos(row: int, L: int) -> tuple[str, int]:
    return ORDER8[row // L], int(row % L)


def write_npz(path: Path, **arrays) -> Path:
    """Work-dir arrays (token ids allowed here; never committed): written to a
    temporary name, then renamed, so a partial file is never taken as done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return path


def done_marker(path: Path, info: dict | None = None) -> None:
    common.write_json(Path(str(path) + ".done.json"), {"utc": common.utc_iso(), **(info or {})})


def is_done(path: Path) -> bool:
    return Path(str(path) + ".done.json").is_file() and Path(path).exists()


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(common.jsonable(obj), sort_keys=True).encode()).hexdigest()
