"""Preflight of the run host (exp_036 BUILD_SPEC §5.9 `tools/preflight.py`; RUNBOOK step 4;
HYPOTHESIS "K8 working-set rule"; exp_037 DESIGN §2.8, §3.1 P2, §6.5).

exp_037: the schema is "exp037 preflight v1"; the venv compared by default is
the exp_037 venv ($EXP037_VENV, tools.common.exp037_venv()), against E37's own
env/versions.json (decision F2: MLX 0.32.3, mlx-metal 0.32.3, mlx-lm 0.32.0 are
core pins); the Kolibri builds are looked up under builds_dir() ($EXP037_BUILDS,
else $EXP036_MODELS). mx.device_info() is recorded whole, so the gate's P2 reads
the architecture (applegpu_g17*) from the newest exp_037 preflight record.

    preflight.py --quick                     (< 1 min) everything except hashing the weights
    preflight.py --deep                      adds the sha256 of every model shard and dataset
                                             file against its etag (Kolibri shards also against
                                             tools/kolibri_lfs_oids.json), parallel workers
    preflight.py --quick --not-run-host      on the mini or any other build host: nothing is
                                             required, nothing is written unless --out is given,
                                             and no sysctl line is ever printed

Records only whitelisted fields: CPU brand string, hw.model, hw.memsize, GPU
core count (ioreg), macOS version, mx.device_info(), iogpu.wired_limit_mb,
power source, Low Power Mode and powermode (pmset -g), thermalState, free
disk on the $EXP036_MODELS volume, swap usage, venv versions against
env/versions.json, git identity / HEAD / upstream / dirty flag, every
asset's presence and revision, the exp_007 fixtures by real name, the K8
memory-rule B per cell with any sysctl advice, and the C1 B. It never calls
socket.gethostname, scutil or system_profiler, and records no serial number,
hardware UUID, user name or IP address; every path goes through redact_path.

Output on the run host: results/preflight_<UTC>.json, then
results/version_record_<UTC>.json (tools/version_record.py). Writing refuses
unless git identity is Miktam <hello@localfirstai.eu>.

Exit codes: 0 ok, 1 unmet prerequisite, 2 warnings only.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import assetcheck, common
from tools.redact import HOST_LABEL, host_label, redact_path

common.set_offline_env()

SCHEMA = "exp037 preflight v1"
CORE_PACKAGES = ("mlx", "mlx-metal", "mlx-lm", "numpy", "safetensors", "tokenizers", "transformers")
VERSIONS_FILE = common.EXP_DIR / "env" / "versions.json"   # E37's own: the F2 pins (decision F)
PLAN_RULES_FILE = common.EXP_DIR / "runner" / "plan_rules.json"

GB = 1000**3
GiB = 2**30
MiB = 2**20
DISK_BEFORE_CONVERSION = 260 * GB   # BUILD_SPEC §5.9
DISK_AFTER_CONVERSION = 120 * GB
K8_DIR, K4_DIR = "Kolibri-1-MLX-8bit-g64", "Kolibri-1-MLX-4bit-g64"   # under common.builds_dir() (DESIGN §2.8)
PREDICTED_WEIGHT_BYTES = {"K8": 83_100_000_000, "K4": 44_000_000_000}   # spec item 20
EXP007_FIXTURES = ("pad_4k.txt", "pad_15k.txt", "pad_120k.txt")
EXP007_DIR = common.EXP_DIR.parent / "exp_007_hardware_comparison" / "fixtures" / "padding"

# Defaults of runner/plan_rules.json (BUILD_SPEC §7.2), used only if that file
# or runner/memory.py cannot be used. prompt_max: a conservative 2,048 tokens.
DEFAULT_RULES = {
    "B_choices": [16, 8, 4, 2, 1], "B_max_aime": 8, "memory_overhead_gib": 4.0, "limit_fraction": 0.9,
    "kv_transient_factor": 2.0, "sysctl_advice_mb": 114688,
    "kv_bytes_per_token": {"kolibri": 20480}, "fixed_window_bytes": {"kolibri": 42_024_960},
    "caps": {"gpqa": 32768, "mmlu": 32768, "aime": 65536, "ifbench": 16384, "rgb": 16384},
    "prompt_max_tokens": {"gpqa": 2048, "mmlu": 2048, "aime": 2048, "ifbench": 2048, "rgb": 2048},
}
POWERMODE_LABELS = {"0": "automatic", "1": "low power", "2": "high power"}
THERMAL_LABELS = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


# --------------------------------------------------------------------------
# Every OS call goes through System (monkeypatched in the tests)
# --------------------------------------------------------------------------

class System:
    def sysctl(self, name: str) -> str | None:
        rc, out, _ = common.run(["sysctl", "-n", name], timeout=10)
        return out.strip() if rc == 0 and out.strip() else None

    def sw_vers(self) -> dict:
        out = {}
        for key, flag in (("product_version", "-productVersion"), ("build", "-buildVersion")):
            rc, o, _ = common.run(["sw_vers", flag], timeout=10)
            out[key] = o.strip() if rc == 0 else None
        return out

    def gpu_cores(self) -> int | None:
        rc, out, _ = common.run(["ioreg", "-rc", "AGXAccelerator", "-d", "1"], timeout=15)
        m = re.search(r'"gpu-core-count"\s*=\s*(\d+)', out) if rc == 0 else None
        return int(m.group(1)) if m else None

    def pmset(self) -> dict:
        settings = {}
        rc, out, _ = common.run(["pmset", "-g"], timeout=10)
        if rc == 0:
            for line in out.splitlines():
                parts = line.split()
                if len(parts) >= 2 and parts[0] in ("lowpowermode", "powermode"):
                    settings[parts[0]] = parts[1]
        rc, out, _ = common.run(["pmset", "-g", "ps"], timeout=10)
        source = None
        if rc == 0:
            if "'AC Power'" in out:
                source = "AC"
            elif "'Battery Power'" in out:
                source = "battery"
        return {"lowpowermode": settings.get("lowpowermode"), "powermode": settings.get("powermode"),
                "source": source}

    def thermal_state(self) -> int | None:
        rc, out, _ = common.run(
            ["osascript", "-l", "JavaScript", "-e",
             'ObjC.import("Foundation"); $.NSProcessInfo.processInfo.thermalState'], timeout=15)
        try:
            return int(out.strip()) if rc == 0 else None
        except ValueError:
            return None

    def device_info(self) -> dict | None:
        try:
            import mlx.core as mx
        except Exception:
            return None
        try:
            info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        except Exception:
            return None
        return {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in dict(info).items()}

    def disk_free(self, path: Path) -> int:
        p = Path(path)
        while not p.exists() and p != p.parent:
            p = p.parent
        return shutil.disk_usage(p).free

    def freeze(self, python: str | None) -> dict:
        return common.freeze(python)


# --------------------------------------------------------------------------
# Pieces
# --------------------------------------------------------------------------

def load_versions(path=VERSIONS_FILE) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare_versions(found: dict, expected: dict) -> dict:
    """Core mismatch → problem; any other pin mismatch or extra → recorded."""
    pkgs = found.get("packages", {})
    pins = expected.get("pins", {})
    core = expected.get("core", {})
    core_mismatch = {k: {"expected": v, "found": pkgs.get(k)} for k, v in core.items() if pkgs.get(k) != v}
    other_mismatch = {k: {"expected": v, "found": pkgs.get(k)} for k, v in pins.items()
                      if k not in core and pkgs.get(k) != v}
    extras = sorted(k for k in pkgs if k not in pins)
    want_py = str(expected.get("python", "3.12"))
    py_ok = str(found.get("python", "")).startswith(want_py + ".") or found.get("python") == want_py
    return {
        "python": found.get("python"), "python_expected": want_py, "python_ok": py_ok,
        "core_expected": core, "core_found": {k: pkgs.get(k) for k in core},
        "core_mismatch": core_mismatch, "other_mismatch": other_mismatch, "extras": extras,
        "n_packages": len(pkgs),
        "freeze_sha256": common.sha256_bytes("\n".join(common.freeze_lines(pkgs)).encode()),
    }


def fixtures() -> dict:
    out = {}
    for name in EXP007_FIXTURES:
        p = EXP007_DIR / name
        out[name] = {"present": p.is_file(), "path": redact_path(p)}
        if p.is_file():
            out[name].update(bytes=p.stat().st_size, sha256=common.sha256_file(p))
    return out


def _load_rules() -> dict:
    rules = dict(DEFAULT_RULES)
    if PLAN_RULES_FILE.is_file():
        try:
            rules.update(json.loads(PLAN_RULES_FILE.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            pass
    return rules


def fallback_need(weight: int, B: int, prompt_max: int, cap: int, rules: dict) -> int:
    """HYPOTHESIS rule 1: need(B) = W + 2·B·(prompt_max + cap)·kv + B·fixed + 4 GiB."""
    kv = int(rules["kv_bytes_per_token"]["kolibri"])
    fixed = int(rules.get("fixed_window_bytes", {}).get("kolibri", DEFAULT_RULES["fixed_window_bytes"]["kolibri"]))
    return int(weight + float(rules["kv_transient_factor"]) * B * (prompt_max + cap) * kv
               + B * fixed + float(rules["memory_overhead_gib"]) * GiB)


def fallback_choose_B(weight: int, task: str, cap: int, L: int, rules: dict) -> int:
    pm = int(rules.get("prompt_max_tokens", DEFAULT_RULES["prompt_max_tokens"]).get(task, 2048))
    for B in sorted(rules["B_choices"], reverse=True):
        if task == "aime" and B > int(rules["B_max_aime"]):
            continue
        if fallback_need(weight, B, pm, cap, rules) <= float(rules["limit_fraction"]) * L:
            return B
    return 0


def memory_rule(L: int, weight: int, weight_source: str) -> dict:
    """K8's memory-rule B per cell at L and at L' = 112 GiB, the sysctl advice
    and the C1 B (HYPOTHESIS "K8 working-set rule"; Control C1)."""
    rules = _load_rules()
    caps = rules.get("caps", DEFAULT_RULES["caps"])
    tasks = [t for t in ("gpqa", "mmlu", "ifbench", "rgb", "aime") if t in caps]
    L2 = int(rules["sysctl_advice_mb"]) * MiB
    source, err = "runner.memory", None
    try:
        import importlib

        rm = importlib.import_module("runner.memory")  # sibling area; lazy

        cells = []
        for t in tasks:
            b = rm.choose_B("K8", t, int(caps[t]), L, weight_bytes=weight)
            b2 = rm.choose_B("K8", t, int(caps[t]), L2, weight_bytes=weight) if L2 > L else b
            cells.append({"task": t, "cap": int(caps[t]), "B": b, "B_at_112GiB": b2})
        advice = rm.sysctl_advice([(t, int(caps[t])) for t in tasks], L, weight_bytes=weight)
    except Exception as e:  # missing module, missing plan_rules.json, ...
        source, err = "tools.preflight fallback", f"{type(e).__name__}: {e}"
        cells = []
        for t in tasks:
            b = fallback_choose_B(weight, t, int(caps[t]), L, rules)
            b2 = fallback_choose_B(weight, t, int(caps[t]), L2, rules) if L2 > L else b
            cells.append({"task": t, "cap": int(caps[t]), "B": b, "B_at_112GiB": b2})
        gains = [f"K8 {c['task']} cap {c['cap']}: B {c['B']} -> {c['B_at_112GiB']}" for c in cells
                 if c["B"] and c["B_at_112GiB"] > c["B"]]
        unfit = [f"K8 {c['task']} cap {c['cap']}: does not fit at B = 1" for c in cells if c["B"] == 0]
        advice = None
        if gains or unfit:
            advice = (f"sudo sysctl iogpu.wired_limit_mb={int(rules['sysctl_advice_mb'])}    "
                      f"# 112 GiB, leaves ~16 GiB for macOS; resets at reboot. " + "; ".join(unfit + gains))
    c1 = next((c["B"] for c in cells if c["task"] == "mmlu"), None)
    return {"source": source, "source_error": err, "L_bytes": L, "L_gib": round(L / GiB, 2),
            "L_prime_bytes": L2, "weight_bytes": weight, "weight_source": weight_source,
            "cells": cells, "advice": advice, "c1_B": c1,
            "k8_fits": all(c["B"] > 0 for c in cells)}


def _weights(models: Path, arm_dir: str, arm: str) -> tuple[int, str]:
    d = models / arm_dir
    files = sorted(d.glob("*.safetensors")) if d.is_dir() else []
    if files:
        return sum(f.stat().st_size for f in files), "measured (converted directory)"
    return PREDICTED_WEIGHT_BYTES[arm], "predicted (BUILD_SPEC item 20; not converted yet)"


def next_step(exp: Path, state: dict) -> str:
    """The next RUNBOOK step, from what exists (printed with the phase plan)."""
    if not state.get("identity_ok"):
        return "RUNBOOK step 2: set git identity and install the pre-push hook"
    if not state.get("versions_ok"):
        return "RUNBOOK step 3: venv sync (bash env/setup.sh) and the unit tests"
    if not state.get("assets_ok"):
        return "RUNBOOK step 0b/0c: finish the downloads, then re-run this preflight"
    if state.get("advice"):
        return "RUNBOOK step 5: Andrei runs the printed sudo sysctl line, then re-run step 4a"
    hyp = exp / "HYPOTHESIS.md"
    if hyp.is_file() and not re.search(r"^- Signed off by: Andrei \(.+\)$", hyp.read_text(encoding="utf-8"), re.M):
        return "RUNBOOK step 6: sign-off (Andrei types it)"
    if not (exp / "tools" / "withheld_shingles.sha256").is_file():
        return "RUNBOOK step 7: verify the inherited item manifests, withheld shingles and T5/T6/T9 against their records"
    builds = common.builds_dir()
    res = exp / "results"
    refreshed = all(list((res / "convert").glob(f"refresh_{b}bit_*.json")) for b in (8, 4))
    if not ((builds / K8_DIR).is_dir() and (builds / K4_DIR).is_dir() and refreshed):
        return "RUNBOOK step 8: clone and refresh the K8 and K4 builds (tools/refresh_builds.py)"
    if not list(res.glob("peers_*.json")):
        return "RUNBOOK step 9: peer check"
    if not list((res / "gate").glob("gate_*.json")):
        return "RUNBOOK step 10: Phase 0 gate"
    if not list((res / "bench").glob("*.jsonl")):
        return "RUNBOOK step 11: bench cells"
    if not list(res.glob("pilot_summary_*.json")):
        return "RUNBOOK step 12: pilot"
    if not list(res.glob("plan_fixed_*.json")):
        return "RUNBOOK step 13: plan amendment"
    return "RUNBOOK step 14 onward: sessions (runner/run.py session --name S2|S3)"


PHASE_PLAN = (  # exp_037 DESIGN §8.2
    "S1 (machine time, nominal / pessimistic h): preflight --deep 0.1/0.3 · port refresh, clones 0.1/0.2 · "
    "peer check 0.4/0.6 · gate 3.0/4.4 · bench (D1, H1, speed, C1) 0.8/1.1 · H5 0.1 · H8 0.3/0.5 · "
    "pilot 1.3/1.9 · plan-fix 0.1 — total 6.2/9.2 h; then S2 and S3 (≤ 16 h each)"
)


# --------------------------------------------------------------------------
# Gather
# --------------------------------------------------------------------------

def gather(mode: str = "quick", run_host: bool = True, system: System | None = None,
           venv_python: str | None = None, workers: int = 8) -> dict:
    sysc = system or System()
    t_start = common.utc_iso()
    problems: list[str] = []
    warnings: list[str] = []
    models = common.models_dir()
    builds = common.builds_dir()   # the Kolibri arm directories (DESIGN §2.8)

    chip = sysc.sysctl("machdep.cpu.brand_string")
    memsize = sysc.sysctl("hw.memsize")
    mem = int(memsize) if memsize and memsize.isdigit() else None
    machine = {"chip": chip, "hw_model": sysc.sysctl("hw.model"), "memory_bytes": mem,
               "gpu_cores": sysc.gpu_cores(), "macos": sysc.sw_vers()}

    di = sysc.device_info()
    wired_raw = sysc.sysctl("iogpu.wired_limit_mb")
    wired_mb = int(wired_raw) if wired_raw and wired_raw.lstrip("-").isdigit() else 0
    rec_ws = int((di or {}).get("max_recommended_working_set_size", 0) or 0)
    L = max(rec_ws, wired_mb * MiB)
    metal = {"max_recommended_working_set_size": rec_ws, "iogpu_wired_limit_mb": wired_mb,
             "L_bytes": L, "L_gib": round(L / GiB, 2),
             "L_source": "iogpu.wired_limit_mb" if wired_mb * MiB > rec_ws else "max_recommended_working_set_size"}
    if di is None:
        (problems if run_host else warnings).append("mx.device_info() unavailable (mlx not importable)")

    pm = sysc.pmset()
    ts = sysc.thermal_state()
    power = {"source": pm.get("source"), "low_power_mode": pm.get("lowpowermode"),
             "powermode": pm.get("powermode"), "powermode_label": POWERMODE_LABELS.get(str(pm.get("powermode"))),
             "thermal_state": ts, "thermal_label": THERMAL_LABELS.get(ts)}
    if run_host:
        if power["source"] != "AC":
            warnings.append(f"power source is {power['source']}, not AC")
        if power["low_power_mode"] not in (None, "0"):
            warnings.append("Low Power Mode is on")
        if power["powermode"] is not None and str(power["powermode"]) != "2":
            warnings.append(f"powermode {power['powermode']} ({power['powermode_label']}), not High Power")
        if ts not in (None, 0):
            warnings.append(f"thermal state {THERMAL_LABELS.get(ts, ts)}")

    swap_raw = sysc.sysctl("vm.swapusage") or ""
    swap = {}
    for key in ("total", "used", "free"):
        m = re.search(key + r"\s*=\s*([\d.]+)M", swap_raw)
        if m:
            swap[f"{key}_mb"] = float(m.group(1))

    # Versions of the run venv: the exp_037 venv (decision F2), never exp_036's.
    if venv_python is None:
        cand = common.exp037_venv() / "bin" / "python"
        venv_python = str(cand) if (run_host and cand.exists()) else None
    try:
        found = sysc.freeze(venv_python)
        expected = load_versions()
        versions = compare_versions(found, expected)
        versions["python_executable"] = redact_path(venv_python) if venv_python else "current interpreter"
        versions["versions_file_sha256"] = common.sha256_file(VERSIONS_FILE)
    except Exception as e:
        versions = {"error": f"{type(e).__name__}: {e}"}
        (problems if run_host else warnings).append(f"cannot compare versions: {versions['error']}")
    if "core_mismatch" in versions:
        msgs = [f"{k} {v['found']} != {v['expected']}" for k, v in versions["core_mismatch"].items()]
        if not versions["python_ok"]:
            msgs.append(f"python {versions['python']} != {versions['python_expected']}")
        if msgs:
            (problems if run_host else warnings).append("core versions: " + ", ".join(msgs))
        if versions["other_mismatch"]:
            warnings.append(f"{len(versions['other_mismatch'])} non-core pins differ from env/versions.json")

    ident = common.git_identity()
    gstate = common.git_state()
    git = {"identity": ident, "identity_ok": common.identity_ok(ident), **gstate}
    if not git["identity_ok"]:
        problems.append(f"git identity {ident.get('name')!r} <{ident.get('email')!r}> is not "
                        f"{common.REQUIRED_GIT_NAME} <{common.REQUIRED_GIT_EMAIL}>")
    if gstate["dirty"]:
        warnings.append(f"working tree has {gstate['n_dirty_paths']} changed paths in the experiment dir")

    assets = assetcheck.check_all(deep=(mode == "deep"), workers=workers)
    if not assets["complete"]:
        (problems if run_host else warnings).extend(f"asset: {p}" for p in assets["problems"])

    fx = fixtures()
    for name, f in fx.items():
        if not f["present"]:
            (problems if run_host else warnings).append(f"exp_007 fixture {name} missing")

    # Disk on the $EXP036_MODELS volume (the clones under $EXP037_BUILDS default to it).
    converted = {"K8": (builds / K8_DIR).is_dir(), "K4": (builds / K4_DIR).is_dir()}
    missing_dl = sum(max(0, int(r.get("bytes_manifest", 0)) - int(r.get("bytes_on_disk", 0)))
                     for r in assets["models"] if r.get("bytes_manifest"))
    need = (DISK_AFTER_CONVERSION if all(converted.values()) else DISK_BEFORE_CONVERSION) + missing_dl
    free = sysc.disk_free(models)
    disk = {"volume_of": "$EXP036_MODELS", "free_bytes": free, "required_bytes": need,
            "rule": "≥ 260 GB free beyond the downloads before conversion, ≥ 120 GB after it",
            "conversions_present": converted, "missing_download_bytes": missing_dl, "ok": free >= need}
    if not disk["ok"]:
        (problems if run_host else warnings).append(
            f"disk: {free / GB:.0f} GB free < {need / GB:.0f} GB required")

    # Memory rule for K8 and the C1 B.
    weight, wsrc = _weights(builds, K8_DIR, "K8")
    mem_rule = memory_rule(L, weight, wsrc) if L else {"error": "no Metal limit (mlx unavailable)"}
    if run_host and mem_rule.get("advice"):
        warnings.append("K8 working-set rule: sysctl advice printed")
    if not run_host and "advice" in mem_rule:
        mem_rule["advice_not_applicable"] = "not the run host: no sysctl line is printed here"
        mem_rule["advice"] = None

    record = {
        "schema": SCHEMA, "mode": mode, "role": "run host" if run_host else "not run host",
        "utc": t_start, "t_start": t_start,
        "host_label": HOST_LABEL if run_host else host_label(chip, mem),
        "machine": machine, "mx_device_info": di, "metal": metal, "power": power, "swap": swap,
        "versions": versions, "git": git, "assets": assets, "exp007_fixtures": fx, "disk": disk,
        "memory_rule": mem_rule,
    }
    state = {"identity_ok": git["identity_ok"],
             "versions_ok": "core_mismatch" in versions and not versions["core_mismatch"] and versions["python_ok"],
             "assets_ok": assets["complete"], "advice": mem_rule.get("advice")}
    record["next_step"] = next_step(common.EXP_DIR, state)
    record["phase_plan"] = PHASE_PLAN
    record["problems"] = problems
    record["warnings"] = warnings
    record["t_end"] = common.utc_iso()
    return record


def exit_code(record: dict) -> int:
    if record["problems"]:
        return 1
    return 2 if record["warnings"] else 0


def write_record(record: dict, out_dir: Path | None = None) -> Path:
    common.require_identity()
    out_dir = Path(out_dir or common.EXP_DIR / "results")
    stamp = record["utc"].replace("-", "").replace(":", "")
    return common.write_new_json(out_dir / f"preflight_{stamp}.json", record)


def _print_summary(record: dict) -> None:
    m, mr = record["machine"], record.get("memory_rule", {})
    print(f"[preflight] {record['mode']} · {record['role']} · {record['host_label']}")
    print(f"  chip {m['chip']} · memory {(m['memory_bytes'] or 0) / GiB:.0f} GiB · GPU cores {m['gpu_cores']} · "
          f"macOS {m['macos'].get('product_version')}")
    print(f"  Metal limit L = {record['metal']['L_gib']} GiB ({record['metal']['L_source']}); "
          f"iogpu.wired_limit_mb = {record['metal']['iogpu_wired_limit_mb']}")
    if mr.get("cells"):
        cells = ", ".join(f"{c['task']} {c['cap']}: B={c['B']}" for c in mr["cells"])
        print(f"  K8 working set ({mr['weight_source']}): {cells}; C1 B = {mr['c1_B']}")
        if mr.get("advice"):
            print(f"  >>> {mr['advice']}")
        elif record["role"] == "run host":
            print("  K8 working set: no sysctl needed")
    for p in record["problems"]:
        print(f"  PROBLEM: {p}")
    for w in record["warnings"]:
        print(f"  warning: {w}")
    print(f"  {record['phase_plan']}")
    print(f"  next: {record['next_step']}")


def main(argv=None) -> int:
    from tools.precision import ensure_exact_fp32
    ensure_exact_fp32()  # MLX_ENABLE_TF32=0 before any GPU work (tools/precision.py; Amendment 1)
    ap = argparse.ArgumentParser(description="exp_037 preflight (exp_036 BUILD_SPEC §5.9)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--quick", action="store_true", help="(default)")
    g.add_argument("--deep", action="store_true")
    ap.add_argument("--not-run-host", action="store_true",
                    help="build host: record only, nothing required, nothing written unless --out")
    ap.add_argument("--out", default=None, help="write the record to this file (not-run-host mode)")
    ap.add_argument("--out-dir", default=None, help="run host: directory for the records (default results/)")
    ap.add_argument("--python", default=None,
                    help="interpreter whose packages are compared (default: $EXP037_VENV, the exp_037 venv)")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 4))
    ap.add_argument("--no-version-record", action="store_true")
    args = ap.parse_args(argv)
    mode = "deep" if args.deep else "quick"
    run_host = not args.not_run_host

    record = gather(mode, run_host=run_host, venv_python=args.python, workers=args.workers)
    _print_summary(record)
    code = exit_code(record)
    if run_host:
        try:
            path = write_record(record, Path(args.out_dir) if args.out_dir else None)
        except common.IdentityError as e:
            print(f"[preflight] refusing to write results: {e}", file=sys.stderr)
            return 1
        print(f"[preflight] wrote {redact_path(path)}")
        if not args.no_version_record:
            try:
                from tools import version_record

                vr = version_record.write(out_dir=Path(args.out_dir) if args.out_dir else None,
                                          venv_python=args.python, assets_result=record["assets"])
                print(f"[preflight] wrote {redact_path(vr)}")
            except Exception as e:  # recorded, not fatal for the preflight itself
                print(f"[preflight] version record not written: {type(e).__name__}: {e}", file=sys.stderr)
                code = max(code, 2)
    elif args.out:
        if not common.identity_ok(record["git"]["identity"]):
            print("[preflight] refusing to write: git identity", file=sys.stderr)
            return 1
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(common.dumps(record), encoding="utf-8")
        print(f"[preflight] wrote {redact_path(args.out)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
