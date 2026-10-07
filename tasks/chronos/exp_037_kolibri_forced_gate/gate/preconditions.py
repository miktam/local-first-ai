# SPDX-License-Identifier: MIT
# Part of Chronos exp_037 (Miktam, 2026-10-06).
"""Gate preconditions P1-P4 (DESIGN §3.1, §9.1-§9.4; decision F2).

A precondition failure is exit 3: the run stops there, it is not a gate run
and never a fix cycle. P1-P4 run at phase 0; P5 (phase 2b) is built from the
helpers of gate/harness.py and gate/checks/ref_drivers.py.

Every function returns {"ok": bool, "values": dict, "reason": str | None}
(reason is None when ok). Nothing here writes a file.

P1  hashes and freeze (real mode only):
    (a) tools/hash_tree.check(HYPOTHESIS.md): every one of the 20 scopes "match"
        (exp_036's 17, gate_rules, and the tests and env scopes added by the
        final review, W15-02);
    (b) GATE_RULES_SHA256, THRESHOLDS_SHA256 and GATE_TEXT_SHA256 are registered
        by the pre-registration: no amendment line names them;
    (c) the commit on the line "Pre-registration commit: <sha>" is an ancestor
        of HEAD and of the upstream tracking branch (the freeze push happened);
    (d) re-run admissibility (§9.4): gate/rules.run_counts allows another
        gate run (at most fix_cycles_max cycles, so fix_cycles_max + 1 runs),
        and after a gate run that did not exit 0 at least one of port,
        reference tree, runner tree, gate-code tree and builds manifest has
        changed. Two crash rules (final review W15-03): no earlier run
        directory results/gate/<UTC>/ may lack its record (a run killed before
        it could write one; `gate/run_gate.py --close-orphans` writes its
        exit-3 record and moves it to aborted/); and every exit-3 run that had
        passed P5 (blind_phase_reached) and whose recorded code hashes differ
        from the current ones must be named, by its UTC, in the first line of
        a numbered amendment (§9.4 item 3: that line names the blind values
        the crashed run computed, which are published from aborted/);
    (e) the "Fixed before any run" table rows of HEAD's (and the working
        copy's) HYPOTHESIS.md equal those of the pre-registration commit.
P2  environment binding: the process's macOS version and build, mlx,
    mlx-metal, mlx-lm, GPU architecture prefix and MLX_ENABLE_TF32 equal
    thresholds.json "P2"; in real mode the newest exp_037 version record (OS,
    packages) and the newest exp_037 preflight record (architecture) too.
    Tiny mode records the values and does not judge them.
P3  G0k: gate/rules.p3_g0k (the fix-5 rule, frozen with the gate_rules scope)
    over gate/checks/g0k_kernels.probe's measurements: at every judged
    (projection, bits, rows) rel_f64(sorted) <= 3 x rel_f64(unsorted), zero
    bad rows, and a bitwise repeat. The probe's arguments come from
    g0k_kernels.config_from_thresholds(thresholds["P3_G0k"]).
P4  build identity: per arm, exp_036's convert record (its sha256 checked
    first) lists exactly the build's model*.safetensors with their sha256,
    and config.json; convert.check_port_file(dir) passes.

Gate runs and cycles (§9.4), used by P1(d) and by the gate record's
`run_counts`: the exit codes of the real-mode exp_037 records
results/gate/gate_<UTC>.json (dry-run promotions excluded), counted by
gate/rules.run_counts: exit 0, 1, 4 or 5 is a gate run; each gate run after
the first that follows one which exited 1, 4 or 5 is a fix cycle; exit 3 is
never a gate run. Every pass/fail decision stays in gate/rules.py (§9.2);
P1, P2 and P4 are checks of identity and environment, not gate rules.
The gate record should carry `code_hashes` (code_hashes() below) so that P1(d)
can compare a later run with it; `sha256` with exp_036's key names is read
as a fallback.
"""

from __future__ import annotations

import fnmatch
import importlib.metadata
import os
import platform
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gate import common
from tools import hash_tree as ht

EXP_DIR = common.EXP_DIR
DEFAULT_RESULTS_DIR = common.DEFAULT_RESULTS_DIR
HYPOTHESIS_NAME = "HYPOTHESIS.md"

# DESIGN §9.2: the 20 rows of exp_037's table (hash_tree.SCOPES: W10's 18, plus tests and env from the final
# review, W15-02) and the three scopes no amendment may change.
REQUIRED_SCOPES = 20
FROZEN_SCOPES = tuple(getattr(ht, "NO_AMENDMENT_SCOPES", ("gate_rules", "thresholds", "gate_text")))

PREREG_LINE = re.compile(r"^\W*Pre-registration commit:\W*([0-9a-f]{7,40})\b", re.M)

EXPERIMENT = "exp_037"
# P1(d)'s code identity of a run; the record keys of exp_036 are aliases.
CODE_HASH_KEYS = ("port", "reference_tree", "runner_tree", "gate_code", "builds_manifest")
_CODE_ALIASES = {
    "port": ("port",),
    "reference_tree": ("reference_tree", "reference"),
    "runner_tree": ("runner_tree", "runner"),
    "gate_code": ("gate_code", "gate"),
    "builds_manifest": ("builds_manifest", "converted_manifest"),
}


def _result(ok: bool, values: dict, reason: str | None = None) -> dict:
    return {"ok": bool(ok), "values": values, "reason": None if ok else (reason or "failed")}


def _combine(parts: dict, label: str) -> dict:
    """{name: result} -> one result; reasons of the failed parts, in order."""
    bad = [f"{label}({k}): {r['reason']}" for k, r in parts.items() if not r["ok"]]
    return _result(not bad, {k: r for k, r in parts.items()}, "; ".join(bad))


def _git(args, cwd) -> tuple[int, str, str]:
    try:
        p = subprocess.run(["git", "-c", "core.quotepath=off", *args], cwd=str(cwd), capture_output=True,
                           text=True, timeout=60)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError as e:
        return 127, "", str(e)
    except subprocess.TimeoutExpired:
        return 124, "", "git timed out"


# ---------------------------------------------------------------------------
# P1: hashes and freeze
# ---------------------------------------------------------------------------


def p1a_hashes(hypothesis: Path, root: Path, use_git: bool = True) -> dict:
    """(a) every scope of the 20-row table matches its registered value."""
    names = [s.name for s in ht.SCOPES]
    if "gate_rules" not in names or len(names) != REQUIRED_SCOPES:
        return _result(False, {"scopes": names},
                       f"tools/hash_tree.py has {len(names)} scopes{'' if 'gate_rules' in names else ' and no gate_rules'}; "
                       f"exp_037's table has {REQUIRED_SCOPES} (DESIGN §9.2)")
    res = ht.check(hypothesis, root, use_git=use_git)
    status = {k: v["status"] for k, v in res["scopes"].items()}
    bad = {k: v for k, v in status.items() if v != "match"}
    return _result(res["ok"] and not bad, {"status": status, "scopes": res["scopes"]},
                   "not matching: " + ", ".join(f"{k} {v}" for k, v in bad.items()))


def amendment_lines(text: str) -> list[dict]:
    """Every "hash_tree: NAME = hex" line inside a "## Amendment k" section, as
    hash_tree.registered_values reads them, with the scope it names (or None)."""
    out, amendment = [], None
    for n, line in enumerate(text.splitlines(), 1):
        h = ht.AMEND_HEADING.match(line)
        if h:
            amendment = int(h.group(1))
            continue
        if line.startswith("## "):
            amendment = None
        if amendment is None:
            continue
        for m in ht.AMEND_LINE.finditer(line):
            key = m.group(1)
            scope = ht.BY_NAME.get(key) or ht.BY_PLACEHOLDER.get(key.upper())
            out.append({"amendment": amendment, "line": n, "key": key, "value": m.group(2),
                        "scope": scope.name if scope else None})
    return out


def p1b_frozen(hypothesis: Path) -> dict:
    """(b) gate_rules, thresholds and gate_text: source "pre-registration", and
    no amendment line names any of them."""
    hypothesis = Path(hypothesis)
    if not hypothesis.is_file():
        return _result(False, {}, f"{hypothesis.name} not found")
    reg = ht.registered_values(hypothesis)
    sources = {name: reg.get(name, {}).get("source", "unknown scope") for name in FROZEN_SCOPES}
    lines = [a for a in amendment_lines(hypothesis.read_text(encoding="utf-8")) if a["scope"] in FROZEN_SCOPES]
    problems = [f"{n} source {s!r}" for n, s in sources.items() if s != "pre-registration"]
    problems += [f"Amendment {a['amendment']} (line {a['line']}) names {a['key']}" for a in lines]
    return _result(not problems, {"sources": sources, "frozen_amendment_lines": lines},
                   "frozen without amendment (DESIGN §9.2): " + "; ".join(problems))


def prereg_commit(text: str) -> tuple[str | None, str | None]:
    """(sha, problem) from the line "Pre-registration commit: <sha>"."""
    found = PREREG_LINE.findall(text)
    if not found:
        return None, "no line 'Pre-registration commit: <sha>'"
    if len(set(found)) > 1:
        return None, f"{len(set(found))} different 'Pre-registration commit' lines"
    return found[0], None


def p1c_pushed(exp_dir: Path, hypothesis: Path) -> dict:
    """(c) the pre-registration commit is an ancestor of HEAD and of @{u}."""
    sha, problem = prereg_commit(Path(hypothesis).read_text(encoding="utf-8")) if Path(hypothesis).is_file() \
        else (None, f"{Path(hypothesis).name} not found")
    if sha is None:
        return _result(False, {"prereg_commit": None}, problem)
    rc, out, err = _git(["rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"], exp_dir)
    if rc != 0:
        return _result(False, {"prereg_commit": sha}, f"{sha} is not a commit of this repository")
    full = out.strip()
    values = {"prereg_commit": full}
    rc_h, _, _ = _git(["merge-base", "--is-ancestor", full, "HEAD"], exp_dir)
    values["ancestor_of_head"] = rc_h == 0
    rc_u, out_u, _ = _git(["rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"], exp_dir)
    values["upstream"] = out_u.strip() if rc_u == 0 else None
    if rc_u != 0:
        values["ancestor_of_upstream"] = False
        return _result(False, values, "no upstream tracking branch: the freeze push cannot be shown")
    rc_a, _, _ = _git(["merge-base", "--is-ancestor", full, "@{u}"], exp_dir)
    values["ancestor_of_upstream"] = rc_a == 0
    problems = []
    if not values["ancestor_of_head"]:
        problems.append(f"{full[:12]} is not an ancestor of HEAD")
    if not values["ancestor_of_upstream"]:
        problems.append(f"{full[:12]} is not an ancestor of {values['upstream']} (the pre-registration is not pushed)")
    return _result(not problems, values, "; ".join(problems))


def _show(exp_dir: Path, rev: str, name: str = HYPOTHESIS_NAME) -> str | None:
    rc, out, _ = _git(["show", f"{rev}:./{name}"], exp_dir)
    return out if rc == 0 else None


def p1e_table(exp_dir: Path, hypothesis: Path, sha: str | None) -> dict:
    """(e) the hash table rows of HEAD's and the working copy's HYPOTHESIS.md
    equal those of the pre-registration commit (amendment lines supersede
    values; the table itself is never edited)."""
    if sha is None:
        return _result(False, {}, "no pre-registration commit")
    pre = _show(exp_dir, sha)
    if pre is None:
        return _result(False, {"prereg_commit": sha}, f"{HYPOTHESIS_NAME} not in {sha[:12]}")
    rows_pre = ht._table_rows(pre)
    texts = {"HEAD": _show(exp_dir, "HEAD"),
             "worktree": Path(hypothesis).read_text(encoding="utf-8") if Path(hypothesis).is_file() else None}
    values = {"prereg_commit": sha, "rows": len(rows_pre),
              "unfilled_in_prereg": sorted(r for r, line in rows_pre.items() if "{{" in line)}
    problems = []
    if not rows_pre:
        problems.append("the pre-registration has no 'Fixed before any run' table")
    if values["unfilled_in_prereg"]:
        problems.append(f"the pre-registration's table has unfilled rows: {values['unfilled_in_prereg']}")
    for where, text in texts.items():
        if text is None:
            values[f"differs_{where}"] = None
            problems.append(f"{HYPOTHESIS_NAME} missing at {where}")
            continue
        rows = ht._table_rows(text)
        diff = sorted((set(rows) ^ set(rows_pre)) | {r for r in set(rows) & set(rows_pre) if rows[r] != rows_pre[r]})
        values[f"differs_{where}"] = diff
        if diff:
            problems.append(f"table rows edited since the pre-registration ({where}): {diff}")
    return _result(not problems, values, "; ".join(problems))


def gate_records(results_dir: Path | None = None) -> list[tuple[Path, dict]]:
    """(path, record) of results/gate/gate_<UTC>.json, oldest first."""
    d = Path(results_dir or DEFAULT_RESULTS_DIR) / "gate"
    out = []
    for p in sorted(d.glob("gate_*.json")):
        if p.name.count("_") != 1:
            continue  # gate_<UTC>_mutants.json and the like
        try:
            out.append((p, common.read_json(p)))
        except Exception:
            out.append((p, {"unreadable": True}))
    return out


def is_real_record(rec: dict) -> bool:
    """A real-mode exp_037 gate record (not a dry-run promotion): its exit code
    enters the run count (§9.4 item 1; exit 3 is never a gate run)."""
    return (rec.get("mode") == "real" and rec.get("experiment") == EXPERIMENT
            and not rec.get("dry_run_promoted_from") and isinstance(rec.get("exit_code"), int))


def run_counts(records, fix_cycles_max: int = 2) -> dict:
    """§9.4 items 1-3 over records (oldest first; (path, record) pairs or
    records): gate/rules.run_counts over the exit codes of the real-mode
    exp_037 records, plus the number of exit-3 records and the latest run's UTC.
    The gate record's `run_counts` field."""
    from gate import rules

    recs = [r for _, r in records] if records and isinstance(records[0], tuple) else list(records or [])
    real = [r for r in recs if is_real_record(r)]
    out = rules.run_counts([r["exit_code"] for r in real], {"fix_cycles_max": fix_cycles_max})
    runs = [r for r in real if rules.is_gate_run(r["exit_code"])]
    out.update({"exit3_records": sum(1 for r in real if r["exit_code"] == 3),
                "exits": [r["exit_code"] for r in runs],
                "latest_utc": runs[-1].get("utc") if runs else None})
    return out


def code_hashes(exp_dir: Path = EXP_DIR, builds: Path | None = None, use_git: bool = True) -> dict:
    """The code identity of a run (§9.4 item 4): port, reference tree, runner
    tree, gate-code tree (hash_tree scopes) and the builds manifest (each arm's
    converted manifest under builds_dir())."""
    exp_dir = Path(exp_dir)
    out = {}
    for key, scope in (("port", "port"), ("reference_tree", "reference"), ("runner_tree", "runner"),
                       ("gate_code", "gate")):
        try:
            out[key] = ht.scope_value(ht.BY_NAME[scope], exp_dir, use_git=use_git)
        except ht.MissingScope:
            out[key] = None
    builds = Path(builds) if builds is not None else common.builds_dir()
    manifests = {}
    for arm in common.ARMS:
        d = builds / common.ARM_DIRS[arm]
        manifests[arm] = common.converted_manifest_sha256(d) if (d / common.CONVERT_RECORD).is_file() else None
    out["builds_manifest"] = manifests
    return out


def _recorded_codes(rec: dict) -> dict:
    src = rec.get("code_hashes") or {}
    sha = rec.get("sha256") or {}
    out = {}
    for key, names in _CODE_ALIASES.items():
        for n in names:
            if n in src:
                out[key] = src[n]
                break
            if n in sha:
                out[key] = sha[n]
                break
    return out


RUN_DIR_NAME = re.compile(r"\d{8}T\d{6}Z")


def orphaned_runs(results_dir: Path | None = None, exclude_utc: str | None = None) -> list[str]:
    """UTC names of the run directories results/gate/<UTC>/ that have no record
    results/gate/gate_<UTC>.json (final review W15-03): a run killed before it
    could write its record (SIGKILL, an out-of-memory kill, a power cut, a kernel
    panic). `exclude_utc` is the running gate's own directory."""
    d = Path(results_dir or DEFAULT_RESULTS_DIR) / "gate"
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir() and RUN_DIR_NAME.fullmatch(p.name)
                  and p.name != exclude_utc and not (d / f"gate_{p.name}.json").is_file())


def amendment_first_lines(text: str) -> dict[int, str]:
    """{k: the first non-empty line after the heading "## Amendment k ..."}."""
    out, current = {}, None
    for line in text.splitlines():
        h = ht.AMEND_HEADING.match(line)
        if h:
            current = int(h.group(1))
            continue
        if line.startswith("## "):
            current = None
            continue
        if current is not None and current not in out and line.strip():
            out[current] = line.strip()
    return out


def blind_crashes(records, current: dict) -> list[dict]:
    """The real-mode exit-3 records that had passed P5 (blind_phase_reached) and
    whose code hashes (CODE_HASH_KEYS) differ from `current`, or were not
    recorded: a code change followed a crash after blind values were computed
    (§9.4 item 3)."""
    out = []
    for path, rec in records:
        if not (is_real_record(rec) and rec["exit_code"] == 3 and rec.get("blind_phase_reached") is True):
            continue
        recorded = _recorded_codes(rec)
        changed = ([k for k in CODE_HASH_KEYS if recorded.get(k) != current.get(k)] if recorded
                   else ["no code hashes recorded"])
        if changed:
            out.append({"record": Path(path).name, "utc": rec.get("utc"), "changed": changed})
    return out


def p1d_rerun(results_dir: Path | None, current: dict, fix_cycles_max: int = 2, hypothesis: Path | None = None,
              current_utc: str | None = None) -> dict:
    """(d) re-run admissibility (§9.4 items 2-4): gate/rules.run_counts must
    allow another gate run, and after a gate run that did not exit 0 at least
    one of CODE_HASH_KEYS must differ from what that run recorded.

    Crash rules (final review W15-03): no run directory other than the running
    gate's own (`current_utc`) may lack its record; and each blind crash
    (blind_crashes) must be named by its UTC in the first line of a numbered
    amendment of `hypothesis`."""
    from gate import rules

    records = gate_records(results_dir)
    counts = run_counts(records, fix_cycles_max)
    values = {"run_counts": counts}
    problems = []
    orphans = orphaned_runs(results_dir, exclude_utc=current_utc)
    values["orphaned_runs"] = orphans
    if orphans:
        problems.append(f"run directories without a record: {', '.join('results/gate/' + o for o in orphans)} "
                        "(an interrupted run; `gate/run_gate.py --close-orphans` writes its exit-3 record and moves "
                        "it to aborted/ first)")
    crashes = blind_crashes(records, current)
    text = Path(hypothesis).read_text(encoding="utf-8") if hypothesis and Path(hypothesis).is_file() else ""
    firsts = amendment_first_lines(text)
    for c in crashes:
        c["named_by"] = sorted(k for k, line in firsts.items() if c["utc"] and c["utc"] in line)
    values["blind_crashes"] = crashes
    unnamed = [c for c in crashes if not c["named_by"]]
    if unnamed:
        problems.append("a code change followed an exit-3 run that had passed P5, and no amendment's first line names "
                        "it (§9.4 item 3): " + "; ".join(f"{c['record']} (changed: {', '.join(c['changed'])})"
                                                          for c in unnamed))
    if not counts["may_run_again"]:
        if counts["gate_runs"] >= counts["gate_runs_max"]:
            problems.append(f"{counts['gate_runs']} gate runs exist (at most {counts['gate_runs_max']})")
        if counts["next_run_is_cycle"] and counts["cycles_used"] >= counts["cycles_max"]:
            problems.append(f"{counts['cycles_used']} fix cycles used (at most {counts['cycles_max']})")
        if not problems:
            problems.append(f"no further gate run is allowed ({counts})")
    runs = [(p, r) for p, r in records if is_real_record(r) and rules.is_gate_run(r["exit_code"])]
    if runs and runs[-1][1]["exit_code"] != 0:
        path, latest = runs[-1]
        recorded = _recorded_codes(latest)
        changed = [k for k in CODE_HASH_KEYS if k in recorded and recorded[k] != current.get(k)]
        values.update({"latest_record": path.name, "latest_exit": latest["exit_code"],
                       "recorded_code_hashes": recorded, "current_code_hashes": current, "changed": changed})
        if not recorded:
            problems.append(f"{path.name} recorded no code hashes; a change cannot be shown")
        elif not changed:
            problems.append(f"the latest gate run ({path.name}) exited {latest['exit_code']} and none of "
                            f"{', '.join(CODE_HASH_KEYS)} has changed since")
    return _result(not problems, values, "; ".join(problems))


def p1(exp_dir: Path = EXP_DIR, results_dir: Path | None = None, current_codes: dict | None = None,
       builds: Path | None = None, fix_cycles_max: int | None = None, mode: str = "real",
       use_git: bool = True, current_utc: str | None = None) -> dict:
    """P1 (a)-(e). Real mode only: in tiny mode nothing is judged. `current_utc`
    is the running gate's own UTC (its run directory exists during phase 0)."""
    if mode != "real":
        return _result(True, {"judged": False, "mode": mode})
    exp_dir = Path(exp_dir)
    hyp = exp_dir / HYPOTHESIS_NAME
    results_dir = Path(results_dir) if results_dir else exp_dir / "results"
    if fix_cycles_max is None:
        th = exp_dir / "gate" / "thresholds.json"
        fix_cycles_max = int(common.read_json(th).get("fix_cycles_max", 2)) if th.is_file() else 2
    if current_codes is None:
        current_codes = code_hashes(exp_dir, builds, use_git=use_git)
    parts = {"a": p1a_hashes(hyp, exp_dir, use_git=use_git), "b": p1b_frozen(hyp)}
    parts["c"] = p1c_pushed(exp_dir, hyp)
    parts["d"] = p1d_rerun(results_dir, current_codes, fix_cycles_max, hypothesis=hyp, current_utc=current_utc)
    sha = parts["c"]["values"].get("prereg_commit")
    parts["e"] = p1e_table(exp_dir, hyp, sha)
    return _combine(parts, "P1")


# ---------------------------------------------------------------------------
# P2: environment binding (decision F2)
# ---------------------------------------------------------------------------


def _mac_ver() -> str:
    return platform.mac_ver()[0]


def _os_build() -> str | None:
    try:
        p = subprocess.run(["sysctl", "-n", "kern.osversion"], capture_output=True, text=True, timeout=10)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def _package_versions() -> dict:
    import mlx.core as mx
    import mlx_lm

    try:
        metal = importlib.metadata.version("mlx-metal")
    except importlib.metadata.PackageNotFoundError:
        metal = None
    return {"mlx": mx.__version__, "mlx-metal": metal, "mlx-lm": mlx_lm.__version__}


def _architecture() -> str | None:
    import mlx.core as mx

    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return info.get("architecture")


def _tf32() -> dict:
    from tools import precision

    try:
        return precision.ensure_exact_fp32()
    except precision.PrecisionError as e:
        return {precision.ENV: os.environ.get(precision.ENV), "error": str(e)}


def observe_environment() -> dict:
    """The values P2 binds, from this process."""
    return {"macos": _mac_ver(), "macos_build": _os_build(), "packages": _package_versions(),
            "architecture": _architecture(), "fp32": _tf32()}


def pin_problems(observed: dict, pins: dict) -> list[str]:
    """Each difference between observed values (observe_environment's shape)
    and thresholds.json "P2"."""
    probs = []
    if observed.get("macos") != pins["macos"]:
        probs.append(f"macOS {observed.get('macos')!r} != {pins['macos']!r}")
    if observed.get("macos_build") != pins["macos_build"]:
        probs.append(f"macOS build {observed.get('macos_build')!r} != {pins['macos_build']!r}")
    for pkg, want in pins["packages"].items():
        got = (observed.get("packages") or {}).get(pkg)
        if got != want:
            probs.append(f"{pkg} {got!r} != {want!r}")
    arch = observed.get("architecture")
    if not (isinstance(arch, str) and arch.startswith(pins["architecture_prefix"])):
        probs.append(f"architecture {arch!r} does not start with {pins['architecture_prefix']!r}")
    fp32 = observed.get("fp32") or {}
    if "error" in fp32 or fp32.get("MLX_ENABLE_TF32") != pins["MLX_ENABLE_TF32"]:
        probs.append(f"MLX_ENABLE_TF32 {fp32.get('MLX_ENABLE_TF32')!r} != {pins['MLX_ENABLE_TF32']!r}"
                     + (f" ({fp32['error']})" if "error" in fp32 else ""))
    return probs


def newest_record(results_dir: Path, prefix: str) -> Path | None:
    """results/<prefix>_<UTC>.json with the latest UTC stamp."""
    recs = sorted(p for p in Path(results_dir).glob(f"{prefix}_*.json") if re.fullmatch(
        rf"{re.escape(prefix)}_\d{{8}}T\d{{6}}Z\.json", p.name))
    return recs[-1] if recs else None


def _arch_of(rec: dict) -> str | None:
    return (rec.get("mx_device_info") or {}).get("architecture")


def version_record_problems(rec: dict, pins: dict) -> list[str]:
    probs = []
    if rec.get("schema") != pins["version_record_schema"]:
        probs.append(f"schema {rec.get('schema')!r} != {pins['version_record_schema']!r}")
    os_ = rec.get("os") or {}
    if os_.get("macos") != pins["macos"]:
        probs.append(f"os.macos {os_.get('macos')!r} != {pins['macos']!r}")
    if os_.get("build") != pins["macos_build"]:
        probs.append(f"os.build {os_.get('build')!r} != {pins['macos_build']!r}")
    for pkg, want in pins["packages"].items():
        got = (rec.get("packages") or {}).get(pkg)
        if got != want:
            probs.append(f"packages.{pkg} {got!r} != {want!r}")
    if "mx_device_info" in rec:  # tools/version_record.py (W10) records it; checked when present
        arch = _arch_of(rec)
        if not (isinstance(arch, str) and arch.startswith(pins["architecture_prefix"])):
            probs.append(f"mx_device_info.architecture {arch!r} does not start with {pins['architecture_prefix']!r}")
    return probs


def preflight_record_problems(rec: dict, pins: dict) -> list[str]:
    probs = []
    if rec.get("schema") != pins["preflight_schema"]:
        probs.append(f"schema {rec.get('schema')!r} != {pins['preflight_schema']!r}")
    arch = _arch_of(rec)
    if not (isinstance(arch, str) and arch.startswith(pins["architecture_prefix"])):
        probs.append(f"mx_device_info.architecture {arch!r} does not start with {pins['architecture_prefix']!r}")
    return probs


def _record_check(results_dir: Path, prefix: str, pins: dict, check) -> dict:
    p = newest_record(results_dir, prefix)
    if p is None:
        return {"file": None, "sha256": None, "problems": [f"no results/{prefix}_<UTC>.json"]}
    try:
        rec = common.read_json(p)
    except Exception as e:
        return {"file": p.name, "sha256": common.sha256_file(p), "problems": [f"unreadable: {e}"]}
    return {"file": p.name, "sha256": common.sha256_file(p), "problems": check(rec, pins)}


def p2(pins: dict, mode: str = "real", results_dir: Path | None = None, observed: dict | None = None) -> dict:
    """P2. `pins` is thresholds.json "P2". In real mode the process's values
    and the newest exp_037 version and preflight records must equal the pins;
    in tiny mode the observed values are recorded, not judged."""
    observed = observe_environment() if observed is None else observed
    values = {"mode": mode, "observed": observed,
              "pins": {k: pins[k] for k in ("macos", "macos_build", "packages", "architecture_prefix",
                                            "MLX_ENABLE_TF32") if k in pins}}
    if mode != "real":
        values["judged"] = False
        return _result(True, values)
    problems = pin_problems(observed, pins)
    results_dir = Path(results_dir or DEFAULT_RESULTS_DIR)
    vr = _record_check(results_dir, "version_record", pins, version_record_problems)
    pf = _record_check(results_dir, "preflight", pins, preflight_record_problems)
    values.update({"judged": True, "process_problems": problems, "version_record": vr, "preflight_record": pf})
    problems = problems + [f"version record {vr['file']}: {x}" for x in vr["problems"]] \
        + [f"preflight record {pf['file']}: {x}" for x in pf["problems"]]
    return _result(not problems, values, "; ".join(problems))


# ---------------------------------------------------------------------------
# P3: G0k's rule (the measurements are gate/checks/g0k_kernels.probe's)
# ---------------------------------------------------------------------------


def p3(probe_result: dict, th: dict) -> dict:
    """P3: gate/rules.p3_g0k (the fix-5 rule; frozen with gate_rules) over
    gate/checks/g0k_kernels.probe's measurements, at every (projection, bits,
    rows) thresholds.json "P3_G0k" judges. `th` is the whole thresholds dict."""
    from gate import rules

    entries = list((probe_result.get("results") or {}).values())
    try:
        r = rules.p3_g0k(entries, th)
    except rules.RuleInputError as e:
        return _result(False, {"error": str(e)}, f"G0k: {e}")
    ratios = [(e["rel_f64_sorted"] / e["rel_f64_unsorted"], e) for e in entries
              if isinstance(e.get("rel_f64_sorted"), float) and e.get("rel_f64_unsorted")]
    worst = max(ratios, key=lambda t: t[0], default=(None, {}))
    values = {**r, "measured": len(entries),
              "worst_ratio": {"ratio": worst[0], **{k: worst[1].get(k) for k in ("projection", "bits", "rows", "routing")}},
              "env": probe_result.get("env"), "seconds": probe_result.get("seconds")}
    bad = [f"{f['shape']} ({', '.join(f['failed'])})" for f in r["failing"]] + [f"{a} (not measured)" for a in r["absent"]]
    return _result(r["ok"], values, f"G0k failed at {len(bad)} judged shapes: " + "; ".join(bad[:5])
                   + (" ..." if len(bad) > 5 else ""))


# ---------------------------------------------------------------------------
# P4: build identity
# ---------------------------------------------------------------------------


def _hash_files(paths, workers: int = 4) -> dict:
    paths = list(paths)
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(paths) or 1))) as ex:
        return dict(zip((p.name for p in paths), ex.map(common.sha256_file, paths)))


def check_build(build: Path, record_path: Path, record_sha256: str | None, shards: str = "model*.safetensors",
                config: str = "config.json", port_file: Path | None = None, workers: int = 4) -> dict:
    """P4 for one build directory against one convert record."""
    from port.convert import StalePortFile, check_port_file

    build, record_path = Path(build), Path(record_path)
    values = {"dir": common.redact_path(build), "record": common.redact_path(record_path),
              "record_sha256_pinned": record_sha256}
    if not record_path.is_file():
        return _result(False, values, f"convert record {common.redact_path(record_path)} not found")
    got = common.sha256_file(record_path)
    values["record_sha256"] = got
    if record_sha256 is not None and got != record_sha256:  # checked before anything is hashed
        return _result(False, values, f"convert record sha256 {got[:16]} != pinned {record_sha256[:16]}")
    if not build.is_dir():
        return _result(False, values, f"{common.redact_path(build)} is not a directory")
    files = common.read_json(record_path).get("files") or {}
    want = {name: e for name, e in files.items() if fnmatch.fnmatch(name, shards)}
    have = {p.name: p for p in build.glob(shards) if p.is_file()}
    problems = []
    missing, extra = sorted(set(want) - set(have)), sorted(set(have) - set(want))
    values.update({"shards": len(want), "missing": missing, "extra": extra})
    if not want:
        problems.append(f"the record lists no {shards}")
    if missing:
        problems.append(f"missing shards {missing}")
    if extra:
        problems.append(f"shards not in the record {extra}")
    common_names = sorted(set(want) & set(have))
    size_bad = [n for n in common_names if "bytes" in want[n] and have[n].stat().st_size != want[n]["bytes"]]
    shas = _hash_files([have[n] for n in common_names if n not in size_bad], workers)
    sha_bad = sorted(size_bad + [n for n, s in shas.items() if s != want[n].get("sha256")])
    values["shard_mismatches"] = sha_bad
    if sha_bad:
        problems.append(f"shard sha256 differs from the record: {sha_bad}")
    cfg = build / config
    cfg_want = (files.get(config) or {}).get("sha256")
    cfg_got = common.sha256_file(cfg) if cfg.is_file() else None
    values["config_sha256"] = cfg_got
    if cfg_want is None or cfg_got != cfg_want:
        problems.append(f"{config} sha256 {str(cfg_got)[:16]} != record {str(cfg_want)[:16]}")
    try:
        values["port_sha256"] = check_port_file(build, **({"port_file": port_file} if port_file else {}))
    except StalePortFile as e:
        values["port_sha256"] = None
        problems.append(f"stale port file: {str(e).splitlines()[0]}")
    except (FileNotFoundError, ValueError, KeyError) as e:
        values["port_sha256"] = None
        problems.append(f"port file check failed: {type(e).__name__}: {e}")
    return _result(not problems, values, "; ".join(problems))


def p4(block: dict, builds: Path | None = None, exp_dir: Path = EXP_DIR, port_file: Path | None = None,
       mode: str = "real", arms=None, workers: int = 4) -> dict:
    """P4 per arm. `block` is thresholds.json "P4": {"builds": {arm: {dir,
    convert_record (relative to the experiment dir), convert_record_sha256}},
    "shards", "config"}. Real mode reads exp_036's pinned convert records; tiny
    mode (no exp_036 record exists for a tiny build) reads each build's own
    exp036_convert_record.json, unpinned, and judges the rest the same way."""
    builds = Path(builds) if builds is not None else common.builds_dir()
    shards, config = block.get("shards", "model*.safetensors"), block.get("config", "config.json")
    parts = {}
    for arm, spec in block["builds"].items():
        if arms is not None and arm not in arms:
            continue
        d = builds / spec["dir"]
        if mode == "real":
            rec, pinned = (Path(exp_dir) / spec["convert_record"]).resolve(), spec["convert_record_sha256"]
        else:
            rec, pinned = d / common.CONVERT_RECORD, None
        parts[arm] = check_build(d, rec, pinned, shards, config, port_file, workers)
    out = _combine(parts, "P4")
    out["values"] = {"mode": mode, "arms": out["values"]}
    return out
