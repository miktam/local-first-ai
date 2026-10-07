"""Version record (BUILD_SPEC §5.9 `tools/version_record.py`; HYPOTHESIS
"Evidence layout": version_record_<UTC>.json, the exp_035 shape).

`record()` collects: OS, Python, mlx, mlx-metal, mlx-lm, numpy, tokenizers,
safetensors, transformers (and pyarrow, jinja2), the pip-freeze sha256, git
commit and dirty flag, the sha256 of every tree of the HYPOTHESIS hash table
compared with its pre-registered value or a numbered amendment
(tools/hash_tree.py), the converted Kolibri manifests, the asset revisions,
the Metal limits, and each arm's total and active parameters from its config
(tools/params.py) with its KV bytes per token (tools/kv_bytes.py).

`write()` puts it in results/version_record_<UTC>.json; it refuses unless the
git identity is Miktam <hello@localfirstai.eu>. Every phase calls it (the
preflight, the gate, the bench, the sessions, scoring).

exp_037 (decision F2; DESIGN §3.1 P2, §6.5): the schema is "exp037 version
record v1", the record exp_037's gate P2 binds to (RUNBOOK step 1 writes one
under the exp_037 venv; exp_036's "exp036 …" records never satisfy P2). Besides
the package versions it carries `versions`, the comparison with the core pins
of E37's own env/versions.json (MLX 0.32.3, mlx-metal 0.32.3, mlx-lm 0.32.0),
and `mx_device_info` (with its `architecture`, applegpu_g17* on the M5 Max).
The Kolibri builds are read under builds_dir() ($EXP037_BUILDS, else
$EXP036_MODELS); the default interpreter is the exp_037 venv ($EXP037_VENV).

    version_record.py [--out-dir DIR] [--python PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import assetcheck, common, hash_tree
from tools.redact import HOST_LABEL, redact_path

SCHEMA = "exp037 version record v1"
RECORDED_PACKAGES = ("mlx", "mlx-metal", "mlx-lm", "numpy", "tokenizers", "safetensors", "transformers",
                     "jinja2", "pyarrow")
# HYPOTHESIS "Arms" → directory name (same as runner/common.py ARMS): K8 and K4 under builds_dir()
# (DESIGN §2.8), the peers under $EXP036_MODELS.
KOLIBRI_ARMS = ("K8", "K4")
ARM_DIRS = {
    "K8": "Kolibri-1-MLX-8bit-g64", "K4": "Kolibri-1-MLX-4bit-g64",
    "G8": "gemma-4-26b-a4b-it-8bit", "G4": "gemma-4-26b-a4b-it-4bit",
    "Q36-8": "Qwen3.6-35B-A3B-8bit", "Q36-4": "Qwen3.6-35B-A3B-4bit",
    "Q38-8": "Qwen3.8-27B-8bit", "Q38-4": "Qwen3.8-27B-4bit",
}
KOLIBRI_SOURCE_DIR = "Kolibri-1-BF16"
CONVERT_RECORD = "exp036_convert_record.json"


def _arm_dir(arm: str) -> Path:
    return (common.builds_dir() if arm in KOLIBRI_ARMS else common.models_dir()) / ARM_DIRS[arm]


def _converted() -> dict:
    out = {}
    for arm in KOLIBRI_ARMS:
        d = _arm_dir(arm)
        rec = d / CONVERT_RECORD
        if not rec.is_file():
            out[arm] = {"present": d.is_dir(), "record": None}
            continue
        r = json.loads(rec.read_text(encoding="utf-8"))
        out[arm] = {
            "present": True, "path": redact_path(d), "record_sha256": common.sha256_file(rec),
            "manifest_sha256": r.get("manifest_sha256"), "port_sha256": r.get("port_sha256"),
            "bits": (r.get("quantization") or {}).get("bits", r.get("bits")),
            "spec_version": r.get("spec_version"),
        }
    return out


def _arm_params() -> dict:
    """Config arithmetic per arm; Kolibri arms from the BF16 config if not converted."""
    from tools import kv_bytes, params

    out = {}
    for arm in ARM_DIRS:
        d = _arm_dir(arm)
        if not (d / "config.json").is_file() and arm in KOLIBRI_ARMS:
            d = common.models_dir() / KOLIBRI_SOURCE_DIR
        if not (d / "config.json").is_file():
            out[arm] = {"present": False}
            continue
        try:
            p = params.config_params(d)
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            g, f = kv_bytes.kv_bytes_per_token(cfg)
            out[arm] = {"present": True, "config_dir": redact_path(d), **p,
                        "kv_growing_bytes_per_token": g, "kv_fixed_bytes": f}
        except Exception as e:  # recorded, not fatal
            out[arm] = {"present": True, "error": f"{type(e).__name__}: {e}"}
    return out


def _device_info() -> dict | None:
    """mx.device_info() of this process (the exp_037 venv when run as "$PY"), as the
    preflight records it; None without mlx."""
    from tools.preflight import System

    return System().device_info()


def _metal(di: dict | None = None) -> dict:
    from tools.preflight import System

    s = System()
    di = (di if di is not None else s.device_info()) or {}
    raw = s.sysctl("iogpu.wired_limit_mb")
    mb = int(raw) if raw and raw.lstrip("-").isdigit() else 0
    rec = int(di.get("max_recommended_working_set_size", 0) or 0)
    return {"max_recommended_working_set_size": rec, "iogpu_wired_limit_mb": mb,
            "L_bytes": max(rec, mb * 2**20), "device_name": di.get("device_name"),
            "architecture": di.get("architecture")}


def _pins(found: dict) -> dict:
    """The recorded venv against the core pins of E37's env/versions.json (decision F2), as the
    preflight compares them; `core_ok` is what the gate's P2 needs of the packages."""
    from tools import preflight

    try:
        cmp = preflight.compare_versions(found, preflight.load_versions())
    except (OSError, ValueError) as e:
        return {"error": f"{type(e).__name__}: {e}", "core_ok": False}
    return {"source": "env/versions.json", "versions_file_sha256": common.sha256_file(preflight.VERSIONS_FILE),
            "core_expected": cmp["core_expected"], "core_found": cmp["core_found"],
            "core_mismatch": cmp["core_mismatch"], "python_expected": cmp["python_expected"],
            "python_ok": cmp["python_ok"], "core_ok": not cmp["core_mismatch"] and cmp["python_ok"],
            "n_other_mismatch": len(cmp["other_mismatch"]), "extras": cmp["extras"]}


def _fp32_precision() -> dict:
    from tools import precision

    try:
        return precision.ensure_exact_fp32()
    except precision.PrecisionError as e:
        return {precision.ENV: os.environ.get(precision.ENV), "error": str(e)}


def record(venv_python: str | None = None, assets_result: dict | None = None,
           with_params: bool = True, with_assets: bool = True) -> dict:
    common.set_offline_env()
    t0 = common.utc_iso()
    if venv_python is None:
        cand = common.exp037_venv() / "bin" / "python"   # decision F2: the exp_037 venv
        venv_python = str(cand) if cand.exists() else None
    fr = common.freeze(venv_python)
    pkgs = fr["packages"]
    from tools.preflight import System

    sv = System().sw_vers()
    trees = hash_tree.check(common.EXP_DIR / "HYPOTHESIS.md")
    di = _device_info()
    rec = {
        "schema": SCHEMA, "utc": t0, "host": HOST_LABEL,
        "os": {"macos": sv.get("product_version"), "build": sv.get("build"), "platform": platform.platform()},
        "python": fr["python"],
        "python_executable": redact_path(venv_python) if venv_python else "current interpreter",
        "packages": {k: pkgs.get(k) for k in RECORDED_PACKAGES},
        "versions": _pins(fr),
        "pip_freeze_sha256": common.sha256_bytes("\n".join(common.freeze_lines(pkgs)).encode()),
        "pip_freeze_n": len(pkgs),
        "git": common.git_state(),
        "trees": {"all_match": trees["ok"], "scopes": trees["scopes"]},
        "converted": _converted(),
        # mx.device_info() of this process, whole, under the preflight record's key; P2 reads
        # mx_device_info.architecture (applegpu_g17*) from it (DESIGN §3.1 P2).
        "mx_device_info": di,
        "metal": _metal(di),
        # Exact fp32 on the GPU (tools/precision.py; Amendment 1): the environment value
        # and the probe's relative L2 error in this process.
        "fp32_precision": _fp32_precision(),
    }
    if with_assets:
        res = assets_result or assetcheck.check_all(deep=False)
        rec["assets"] = assetcheck.revisions_summary(res)
    if with_params:
        try:
            rec["params"] = _arm_params()
        except Exception as e:
            rec["params"] = {"error": f"{type(e).__name__}: {e}"}
    rec["t_end"] = common.utc_iso()
    return rec


def write(out_dir=None, venv_python: str | None = None, assets_result: dict | None = None) -> Path:
    common.require_identity()
    rec = record(venv_python=venv_python, assets_result=assets_result)
    out_dir = Path(out_dir or common.EXP_DIR / "results")
    stamp = rec["utc"].replace("-", "").replace(":", "")
    return common.write_new_json(out_dir / f"version_record_{stamp}.json", rec)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="exp_037 version record (exp_036 BUILD_SPEC §5.9; DESIGN §3.1 P2)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--python", default=None,
                    help="interpreter whose packages are recorded (default: $EXP037_VENV, the exp_037 venv)")
    args = ap.parse_args(argv)
    try:
        path = write(args.out_dir, args.python)
    except common.IdentityError as e:
        print(f"[version_record] refusing to write: {e}", file=sys.stderr)
        return 1
    print(f"[version_record] wrote {redact_path(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
