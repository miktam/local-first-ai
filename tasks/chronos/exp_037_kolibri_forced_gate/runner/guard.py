"""Refusal checks (BUILD_SPEC §5.4 runner/guard.py; §2 Identity; HYPOTHESIS
"Verdict and refusal", rule 5 "Freeze", "K8 working-set rule").

Every function either returns (possibly a record) or prints one clear line
to stderr and exits non-zero (GuardError, a SystemExit with code 1). Nothing
here writes anything.

exp_037 (DESIGN §6.3): require_gate returns the gate record's allowed_B for a
Kolibri arm (G5-BP-lean, §3.14), and require_environment binds every pilot,
session and cell (and bench/run_bench.py) to P2's pinned runtime when the
newest gate record is a real-mode one.
"""

from __future__ import annotations

import glob
import importlib
import json
import os
import platform
import re
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

from runner.common import EXP_DIR, KOLIBRI_ARMS, git, sha256_bytes, sha256_file, sysctl

IDENTITY = ("Miktam", "hello@localfirstai.eu")
SIGNOFF_RE = re.compile(r"^- Signed off by: Andrei \(.+\)$", re.M)
AMEND_RE = re.compile(r"^## Amendment (\d+) [—–-] (.+?) \((.+?)\)\s*$", re.M)
PLAN_FILE_RE = re.compile(r"plan_fixed: `(results/plan_fixed_[0-9TZ]+\.json)`, sha256 `([0-9a-f]{64})`")
SCORERS_RE = re.compile(r"Scorers tree sha256: `([0-9a-f]{64})`")
# The peer-check verdict of an arm that runs only H1, the speed cells and the B4 ladder (Amendment 6;
# tools/peer_check.py SPEED_ONLY).
SPEED_ONLY = "speed-only"


class GuardError(SystemExit):
    def __init__(self, message: str, code: int = 1):
        super().__init__(code)
        self.message = message

    def __str__(self) -> str:
        return self.message


def refuse(message: str, code: int = 1):
    print(f"REFUSED: {message}", file=sys.stderr)
    raise GuardError(message, code)


def _rel_exp(exp_dir: Path) -> tuple[Path, str]:
    top = git(["rev-parse", "--show-toplevel"], exp_dir)
    if not top:
        refuse(f"{exp_dir} is not inside a git repository")
    top_p = Path(top).resolve()
    return top_p, Path(exp_dir).resolve().relative_to(top_p).as_posix()


def _head_file(exp_dir: Path, rel_in_exp: str) -> bytes | None:
    import subprocess

    top, rel = _rel_exp(exp_dir)
    try:
        out = subprocess.run(["git", "show", f"HEAD:{rel}/{rel_in_exp}"], cwd=top, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout if out.returncode == 0 else None


# ---------------------------------------------------------------- identity


def require_identity(exp_dir: Path | None = None) -> None:
    d = Path(exp_dir or EXP_DIR)
    name = git(["config", "user.name"], d)
    email = git(["config", "user.email"], d)
    if (name, email) != IDENTITY:
        refuse(
            f'git identity is "{name} <{email}>", expected "{IDENTITY[0]} <{IDENTITY[1]}>": '
            f'run git config user.name "{IDENTITY[0]}" && git config user.email "{IDENTITY[1]}" (RUNBOOK step 2)'
        )


def require_clean_tree(allow: Iterable[str] = ("results/", "aborted/", "evidence/"), exp_dir: Path | None = None) -> None:
    d = Path(exp_dir or EXP_DIR)
    top, rel = _rel_exp(d)
    out = git(["status", "--porcelain", "--untracked-files=all"], top)
    if out is None:
        refuse("git status failed")
    allowed = tuple(f"{rel}/{a}" for a in allow)
    bad = []
    for line in out.splitlines():
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        path = path.strip().strip('"')
        if not path.startswith(allowed):
            bad.append(path)
    if bad:
        refuse(f"working tree has changes outside {', '.join(allow)}: {bad[:10]}{' …' if len(bad) > 10 else ''}")


def require_signoff(exp_dir: Path | None = None) -> None:
    d = Path(exp_dir or EXP_DIR)
    text = _head_file(d, "HYPOTHESIS.md")
    if text is None:
        refuse("HYPOTHESIS.md is not committed at HEAD")
    if not SIGNOFF_RE.search(text.decode("utf-8")):
        refuse('HYPOTHESIS.md at HEAD has no line "- Signed off by: Andrei (…)": Andrei signs off first (RUNBOOK step 6)')


# -------------------------------------------------------------------- gate


GATE_RECORD_RE = re.compile(r"gate_\d{8}T\d{6}Z\.json")
# The only allowed_B sets G5-BP-lean can produce (DESIGN §3.14): {1} ∪ ({2, 4, 8} if pass(arm, 8)) ∪ ({16} if
# pass(arm, 8) and pass(arm, 16)).
ALLOWED_B_SETS = (frozenset({1}), frozenset({1, 2, 4, 8}), frozenset({1, 2, 4, 8, 16}))


def require_gate(arm: str, results: Path | None = None) -> list[int] | None:
    """Kolibri arms only: the latest gate record says PASS for `arm`, bound to
    the current port, converted manifest, thresholds and reference shas.
    Delegates to gate/run_gate.py:require_pass (the one implementation);
    `results` overrides its results directory (tests).

    Returns the arm's allowed_B (exp_037 DESIGN §3.14, §6.3) as a sorted list;
    None for a peer arm (no gate)."""
    if arm not in KOLIBRI_ARMS:
        return None
    try:
        run_gate = importlib.import_module("gate.run_gate")
    except ModuleNotFoundError as e:
        refuse(f"cannot verify a gate PASS for {arm}: gate/run_gate.py is not importable ({e.name})")
    fn = getattr(run_gate, "require_pass", None)
    if fn is None:
        refuse("gate/run_gate.py has no require_pass(arm)")
    try:
        rec = fn(arm) if results is None else fn(arm, results_dir=Path(results))
    except SystemExit as e:
        refuse(f"no gate PASS for {arm}: {e}")
    except Exception as e:  # a missing or unreadable record is a refusal, not a crash
        refuse(f"no gate PASS for {arm}: {type(e).__name__}: {e}")
    if rec is None or rec is False:
        refuse(f"no gate PASS for {arm}")
    if isinstance(rec, dict):
        verdict = rec.get("verdict")
    else:
        verdict = getattr(rec, "verdict", None)
        if verdict is None and isinstance(getattr(rec, "record", None), dict):
            verdict = rec.record.get("verdict")
    if isinstance(verdict, dict):
        verdict = verdict.get(arm)
    if verdict not in ("PASS", None):
        refuse(f"gate verdict for {arm} is {verdict!r}, not PASS")
    return allowed_B_of(rec, arm)


def allowed_B_of(rec: Any, arm: str) -> list[int]:
    """The arm's allowed_B from what gate/run_gate.py:require_pass returns, or from a gate record (DESIGN §3.16:
    the record carries allowed_B, and require_pass returns it). Accepted forms: a list or set of ints (require_pass
    returning allowed_B itself), an object with .allowed_B or with .record (a GateRecord), or a record dict; the
    allowed_B value itself may be per arm ({"K8": [...], "K4": [...]}) or one list.

    A record without allowed_B shows no batched-path pass and gives {1}, the base set of §3.14 (B = 1 only, the
    conservative reading). Anything that is not one of the three sets G5-BP-lean can produce is refused."""
    value: Any = None
    if isinstance(rec, (list, tuple, set, frozenset)):
        value = rec
    elif rec is not None:
        if isinstance(rec, dict):
            value = rec.get("allowed_B")
        else:
            value = getattr(rec, "allowed_B", None)
            if value is None and isinstance(getattr(rec, "record", None), dict):
                value = rec.record.get("allowed_B")
    if isinstance(value, dict):
        value = value.get(arm)
    if value is None:
        return [1]
    try:
        got = frozenset(int(b) for b in value if not isinstance(b, bool) and int(b) == b)
        n = len(list(value))
    except (TypeError, ValueError):
        refuse(f"gate allowed_B for {arm} is not a list of batch sizes: {value!r}")
    if len(got) != n or got not in ALLOWED_B_SETS:
        refuse(f"gate allowed_B for {arm} is {value!r}: not one of the sets G5-BP-lean can give "
               f"({', '.join(str(sorted(s)) for s in ALLOWED_B_SETS)})")
    return sorted(got)


def newest_gate_record(results: Path | None = None) -> dict | None:
    """The newest results/gate/gate_<UTC>.json as a dict (with "_file" set to its name), or None without one."""
    results = Path(results or EXP_DIR / "results")
    gates = sorted(p for p in (results / "gate").glob("gate_*.json") if GATE_RECORD_RE.fullmatch(p.name))
    if not gates:
        return None
    try:
        rec = json.loads(gates[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        refuse(f"{gates[-1].name} is not readable: {type(e).__name__}: {e}")
    if not isinstance(rec, dict):
        refuse(f"{gates[-1].name} is not a gate record")
    rec["_file"] = gates[-1].name
    return rec


# ------------------------------------------------------------- environment

# P2's pins (exp_037 DESIGN §3.1 P2, decision F2, 2026-10-06T02:48:30Z): the run host's macOS and build, the MLX
# runtime and the GPU family, with TF32 off. gate/thresholds.json "P2" holds the frozen copy the gate judges at
# phase 0; the runner binds every session to the same values (tests/test_runner_guard.py checks this table against
# env/versions.json and against thresholds.json's P2 block). No macOS, MLX or mlx-lm update is made from the
# pre-registration until S3 ends (§6.4).
P2_PINS = {
    "macos": "27.0",
    "os_build": "26A428",
    "mlx": "0.32.3",
    "mlx-metal": "0.32.3",
    "mlx-lm": "0.32.0",
    "architecture_prefix": "applegpu_g17",
    "MLX_ENABLE_TF32": "0",
}


def _dist_version(dist: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(dist)
    except Exception:
        return None


def observe_environment() -> dict:
    """The values P2 judges, read from this process (§3.1 P2): platform.mac_ver(), sysctl kern.osversion,
    mlx.core.__version__, the mlx-metal distribution, mlx_lm.__version__, mx.device_info()["architecture"] and
    MLX_ENABLE_TF32. Nothing identifying the machine (no device name, no hostname)."""
    obs: dict[str, Any] = {"macos": platform.mac_ver()[0] or None, "os_build": sysctl("kern.osversion"),
                           "mlx": None, "mlx-metal": _dist_version("mlx-metal"), "mlx-lm": None,
                           "architecture": None, "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32")}
    try:
        import mlx.core as mx

        obs["mlx"] = getattr(mx, "__version__", None) or _dist_version("mlx")
        obs["architecture"] = str(mx.device_info().get("architecture"))
    except Exception as e:  # recorded; a real-mode binding then refuses on the missing value
        obs["mlx"] = obs["mlx"] or _dist_version("mlx")
        obs["observe_error"] = f"{type(e).__name__}: {e}"
    try:
        import mlx_lm

        obs["mlx-lm"] = getattr(mlx_lm, "__version__", None) or _dist_version("mlx-lm")
    except Exception:
        obs["mlx-lm"] = _dist_version("mlx-lm")
    return obs


def environment_mismatches(observed: dict, pins: dict | None = None) -> list[str]:
    """One line per P2 value that differs from its pin (empty when all match)."""
    pins = dict(pins or P2_PINS)
    bad = []
    for key in ("macos", "os_build", "mlx", "mlx-metal", "mlx-lm", "MLX_ENABLE_TF32"):
        if str(observed.get(key)) != str(pins[key]):
            bad.append(f"{key} {observed.get(key)!r} != {pins[key]!r}")
    arch = observed.get("architecture")
    if not (isinstance(arch, str) and arch.startswith(pins["architecture_prefix"])):
        bad.append(f"architecture {arch!r} does not start with {pins['architecture_prefix']!r}")
    return bad


def require_environment(rec: dict | None, *, observed: dict | None = None,
                        observe: Callable[[], dict] = observe_environment) -> dict:
    """Version binding at session start (exp_037 DESIGN §6.3; G1 item 14), with `rec` the newest gate record
    (newest_gate_record()):

    - a real-mode record without dry_run_promoted_from: every P2 value must equal its pin, else the session is
      refused (a pre-registered consequence, §9.6);
    - otherwise (a tiny record, a dry run's promoted record, or no record): the values are recorded, never judged.

    Returns {"binding": "judged" | "recorded", "gate_record": <file name or None>, "observed": {...},
    "pins": P2_PINS}, which the caller writes into its session or pilot record."""
    obs = dict(observed if observed is not None else observe())
    is_dict = isinstance(rec, dict)
    real = is_dict and rec.get("mode") == "real" and not rec.get("dry_run_promoted_from")
    out = {"binding": "judged" if real else "recorded", "gate_record": rec.get("_file") if is_dict else None,
           "gate_mode": rec.get("mode") if is_dict else None, "observed": obs, "pins": dict(P2_PINS)}
    if real:
        bad = environment_mismatches(obs)
        if bad:
            refuse(f"the runtime differs from P2's pins under the real-mode gate record "
                   f"{out['gate_record'] or '(newest)'}: {'; '.join(bad)}. No session runs on another runtime "
                   f"(exp_037 DESIGN §6.3-§6.4, a pre-registered consequence)")
    return out


# ------------------------------------------------------------------- peers


def newest(results: Path, pattern: str) -> Path | None:
    files = sorted(glob.glob(str(Path(results) / pattern)))
    return Path(files[-1]) if files else None


def require_peers(arm: str, results: Path | None = None) -> dict | None:
    """Peer arms only: the newest results/peers_<UTC>.json marks `arm` ok."""
    if arm in KOLIBRI_ARMS:
        return None
    results = Path(results or EXP_DIR / "results")
    p = newest(results, "peers_*.json")
    if p is None:
        refuse(f"no results/peers_<UTC>.json: run the peer check first (RUNBOOK step 9)")
    rec = json.loads(p.read_text(encoding="utf-8"))
    arms = rec.get("arms", rec)
    entry = arms.get(arm) if isinstance(arms, dict) else None
    if not isinstance(entry, dict):
        refuse(f"{p.name} has no entry for {arm}")
    # tools/peer_check.py writes verdict "ok" | "B=1" | "speed-only" | "fail" per arm; "B=1"
    # (only the batched path failed) runs at B = 1 by the plan rule. Every caller of this guard runs quality
    # cells (pilot, session, cell); a "speed-only" arm (Amendment 6) runs none: only H1, the speed cells and the
    # B4 ladder, which bench/ runs without this guard.
    if str(entry.get("verdict", "")).lower() == SPEED_ONLY:
        refuse(f"{arm} is speed-only by the peer check ({p.name}; Amendment 6): it runs no quality cell, "
               f"only H1, the speed cells and the B4 ladder")
    if "verdict" in entry:
        ok = str(entry["verdict"]).lower() in ("ok", "b=1")
    else:
        ok = entry.get("ok")
        if ok is None:
            ok = str(entry.get("status", "")).lower() in ("ok", "pass", "b=1")
    if not ok:
        refuse(f"peer check did not pass for {arm} ({p.name})")
    return entry


def excluded_arms(results: Path | None = None) -> dict[str, str]:
    """{arm: reason} for the arms the records exclude from the run (review fix 2026-10-03):

    - K4, when the latest real gate record says K8 PASS and K4 FAIL: the run continues without K4 once the fix
      cycles are used up, and H1, H7, H8 and D1 are NOT RUN (HYPOTHESIS Phase 0 "Verdict and refusal", exit 4);
    - every arm whose newest peer-check verdict is "fail": dropped by amendment before any scored run (HYPOTHESIS
      "Peers are verified, not gated"); "B=1" is not a drop, and "speed-only" is speed_only_arms()'s.

    Read by `runner/run.py pilot --without` (which refuses any other exclusion) and by runner/plan_fix.py."""
    results = Path(results or EXP_DIR / "results")
    out: dict[str, str] = {}
    gates = sorted(p for p in (results / "gate").glob("gate_*.json") if re.fullmatch(r"gate_\d{8}T\d{6}Z\.json", p.name))
    if gates:
        try:
            rec = json.loads(gates[-1].read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rec = {}
        v = rec.get("verdict") if isinstance(rec.get("verdict"), dict) else {}
        if rec.get("mode") == "real" and v.get("K8") == "PASS" and v.get("K4") == "FAIL":
            out["K4"] = f"gate K4 FAIL ({gates[-1].name})"
    out.update(_peer_verdict_arms(results, "fail"))
    return out


def speed_only_arms(results: Path | None = None) -> dict[str, str]:
    """{arm: reason} for every arm whose newest peer-check verdict is "speed-only" (Amendment 6): its family's
    NLL(8) / KL(8‖4) rule failed and the pinned bf16 reference put the failure on this build (tools/peer_check.py).
    Such an arm runs no quality cell (no pilot, Tier-A or Tier-B task cell) and its family leaves H8's peer median,
    but it is not excluded: H1, the descriptive speed cells and the B4 ladder keep it (bench/ does not read the peer
    record). Read by `runner/run.py pilot --without` and by
    runner/plan_fix.py, which records the arms under "speed_only_arms"."""
    return _peer_verdict_arms(Path(results or EXP_DIR / "results"), SPEED_ONLY)


def _peer_verdict_arms(results: Path, verdict: str) -> dict[str, str]:
    """{arm: reason} for the arms whose verdict in the newest results/peers_<UTC>.json is `verdict`."""
    out: dict[str, str] = {}
    p = newest(results, "peers_*.json")
    if p is not None:
        try:
            rec = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            rec = {}
        arms = rec.get("arms", rec) if isinstance(rec, dict) else {}
        for a, e in sorted(arms.items()) if isinstance(arms, dict) else []:
            if isinstance(e, dict) and str(e.get("verdict", "")).lower() == verdict:
                out[a] = f"peer check {verdict} ({p.name})"
    return out


# -------------------------------------------------------------------- plan


def amendments(text: str) -> list[dict]:
    """[{k, type, utc, body}] for every "## Amendment k — type (UTC)" block."""
    out = []
    ms = list(AMEND_RE.finditer(text))
    for i, m in enumerate(ms):
        end = ms[i + 1].start() if i + 1 < len(ms) else len(text)
        nxt = re.search(r"^## ", text[m.end():end], re.M)
        body_end = m.end() + nxt.start() if nxt else end
        out.append({"k": int(m.group(1)), "type": m.group(2).strip(), "utc": m.group(3), "body": text[m.end():body_end]})
    return out


def scorers_tree_sha256(exp_dir: Path | None = None) -> str:
    root = Path(exp_dir or EXP_DIR)
    try:
        ht = importlib.import_module("tools.hash_tree")
        return ht.scope_value(ht.BY_NAME["scorers"], root)
    except ModuleNotFoundError:
        from runner.common import tree_sha256

        return tree_sha256(root / "scorers")


def require_plan(exp_dir: Path | None = None, scorers_sha: Callable[[Path], str] = scorers_tree_sha256) -> dict:
    """The newest "plan" amendment at HEAD, its plan_fixed JSON committed at
    HEAD and unchanged, HEAD == @{u}, the scorers/ tree unchanged, and the
    plan's status FIXED. Returns the plan dict."""
    d = Path(exp_dir or EXP_DIR)
    text = _head_file(d, "HYPOTHESIS.md")
    if text is None:
        refuse("HYPOTHESIS.md is not committed at HEAD")
    plans = [a for a in amendments(text.decode("utf-8")) if a["type"] == "plan"]
    if not plans:
        refuse("no plan amendment at HEAD: run runner/plan_fix.py and commit its amendment (RUNBOOK step 13)")
    a = max(plans, key=lambda x: x["k"])
    m = PLAN_FILE_RE.search(a["body"])
    s = SCORERS_RE.search(a["body"])
    if not m or not s:
        refuse(f"Amendment {a['k']} does not name its plan_fixed file, sha256 and scorers tree sha256")
    rel, want = m.group(1), m.group(2)
    committed = _head_file(d, rel)
    if committed is None:
        refuse(f"{rel} is not committed at HEAD")
    if sha256_bytes(committed) != want:
        refuse(f"{rel} at HEAD does not match the sha256 in Amendment {a['k']}")
    if not (d / rel).is_file() or sha256_file(d / rel) != want:
        refuse(f"{rel} in the working tree differs from Amendment {a['k']}")
    head = git(["rev-parse", "HEAD"], d)
    up = git(["rev-parse", "@{u}"], d)
    if up is None:
        refuse("the branch has no upstream: push the plan amendment first")
    if head != up:
        refuse("HEAD is not equal to its upstream: pull or push so that the plan amendment is the pushed HEAD")
    cur = scorers_sha(d)
    if cur != s.group(1):
        refuse(f"scorers/ tree sha256 {cur[:12]}… differs from Amendment {a['k']} ({s.group(1)[:12]}…)")
    plan = json.loads(committed.decode("utf-8"))
    if plan.get("status") != "FIXED":
        refuse(f"Amendment {a['k']} is {plan.get('status')}, not a fixed plan: no scored run")
    plan["_amendment_k"] = a["k"]
    plan["_plan_file"] = rel
    plan["_plan_sha256"] = want
    return plan


def require_metal_limit(plan: dict, limit_fn: Callable | None = None, k8_queued: bool = True) -> int:
    """Refuse when any K8 cell is queued and the current L is below the L the
    plan used (the sysctl resets at reboot)."""
    from runner import memory

    L, _ = (limit_fn or memory.effective_limit)()
    used = int(plan.get("L_used") or 0)
    if k8_queued and L < used:
        mb = plan.get("sysctl_advice_mb", 114688)
        refuse(
            f"Metal limit {L / 2**30:.2f} GiB is below the {used / 2**30:.2f} GiB the plan used for K8. "
            f"Andrei runs: sudo sysctl iogpu.wired_limit_mb={mb}"
        )
    return L


def _head_amendment_files(exp_dir: Path) -> list[bytes]:
    """Contents of every amendments/*.md committed at HEAD (the mini pushes its
    amendments as such files; RUNBOOK "HYPOTHESIS.md has one writer")."""
    import subprocess

    top, rel = _rel_exp(exp_dir)
    try:
        out = subprocess.run(["git", "-c", "core.quotepath=off", "ls-tree", "--name-only", "HEAD", f"{rel}/amendments/"],
                             cwd=top, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return []
    blobs = []
    for path in out.stdout.splitlines() if out.returncode == 0 else []:
        if path.endswith(".md"):
            b = _head_file(exp_dir, "amendments/" + path.rsplit("/", 1)[-1])
            if b is not None:
                blobs.append(b)
    return blobs


def tier2_at_head(exp_dir: Path | None = None) -> bool:
    """A "Tier-2 analysis" amendment is committed at HEAD: appended to
    HYPOTHESIS.md, or as the mini's amendments/<k>_tier2_<UTC>.md file (HYPOTHESIS
    "What counts as evidence": frozen by its own amendment, pushed from the mini;
    the mbp appends the file verbatim at its next commit). Review fix 2026-10-03:
    the file alone counts, so a pull is enough before B4 and scoring."""
    d = Path(exp_dir or EXP_DIR)
    text = _head_file(d, "HYPOTHESIS.md")
    texts = [text] if text is not None else []
    texts += _head_amendment_files(d)
    return any(a["type"] == "Tier-2 analysis" for t in texts for a in amendments(t.decode("utf-8")))


def require_tier2(exp_dir: Path | None = None) -> None:
    if not tier2_at_head(exp_dir):
        refuse('no "Tier-2 analysis" amendment at HEAD, neither in HYPOTHESIS.md nor as a committed '
               'amendments/ file (BUILD_SPEC "Build tiers"): B4 cells and scoring wait for it; pull first')


def require_assets(name: str) -> None:
    """The named asset (arm folder or dataset) is present at its pinned
    revision, via tasks/assets.py."""
    try:
        assets = importlib.import_module("tasks.assets")
    except ModuleNotFoundError as e:
        refuse(f"cannot verify asset {name}: tasks/assets.py is not importable ({e.name})")
    try:
        a = assets.asset(name)
        ok, found = assets.verify_revision(a)
    except Exception as e:
        refuse(f"asset {name}: {type(e).__name__}: {e}")
    if not ok:
        refuse(f"asset {name} is not at its pinned revision (found {found})")
