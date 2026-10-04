"""Status, run-record blocks and amendment sync for HYPOTHESIS.md
(BUILD_SPEC §5.9 `tools/status.py`; RUNBOOK "HYPOTHESIS.md has one writer").

    status.py                              runner/run.py status (per-cell progress, no accuracy)
    status.py --record-block <phase> [--files PATH ...] [--status TEXT]
        appends "## <phase> — run record (<UTC>)" to HYPOTHESIS.md: the phase,
        UTC from the clock, the commit, the sha256 of the result files (given,
        or the phase's newest files) and, for the gate, the verdict per arm;
        then rewrites only the Status line
    status.py --sync-amendments
        appends, verbatim and in number order, every amendments/*.md whose
        first line ("## Amendment k — <type> (<UTC>)") is not yet in
        HYPOTHESIS.md; a no-op when there is none; refused, with nothing
        appended, if one of them still holds a {{PLACEHOLDER}}
    status.py --set-status TEXT            rewrites only the Status line
    status.py --append-amendment FILE      appends one amendment file verbatim (e.g. plan_fix's
        results/AMENDMENT_<k>_<UTC>.md, RUNBOOK step 13); refused if its heading is malformed or
        already in HYPOTHESIS.md (review fix 2026-10-03: no `cat … >>` in a RUNBOOK block), or if it
        still holds a {{PLACEHOLDER}}
    status.py --tier2                      "Tier-2 analysis amendment at HEAD: yes|no" (in HYPOTHESIS.md
        or as a committed amendments/ file); exit 0 yes, 1 no (RUNBOOK steps 14, 16, 17)
    status.py --verify-private [--private DIR]
        on the mini after the mirror (RUNBOOK step 18): every private file that
        evidence/withheld_manifest.jsonl lists, at its latest sha256 (a moved file at
        its new path), exists under $EXP036_PRIVATE with that sha256; manifests/ and
        aborted/ files the list does not name are reported as expected extras
        (review fix 2026-10-03); exit 0 ok, 1 otherwise

Edits in place: the Status line only (BUILD_SPEC §2 Append-only). Everything
else is appended.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import common

STATUS_RE = re.compile(r"^(\*Pre-registered: .*? · Status: )(.*)(\*)\s*$")
AMEND_HEAD_RE = re.compile(r"^## Amendment (\d+) — .+\(.+\)\s*$")
# An amendment still holding a {{PLACEHOLDER}} (e.g. {{ANDREI_DECISION}}, a draft awaiting Andrei's own words) is
# never appended: HYPOTHESIS.md is append-only, so an unfilled draft could not be taken back.
PLACEHOLDER_RE = re.compile(r"\{\{[A-Z][A-Z0-9_]*\}\}")

# Default result files per phase (globs relative to the experiment dir; newest match).
PHASE_FILES = {
    "preflight": ["results/preflight_*.json", "results/version_record_*.json"],
    "sign-off": [],
    "manifests": ["results/manifests_*.json"],
    "convert": ["results/convert/convert_8bit_*.json", "results/convert/convert_4bit_*.json"],
    "peers": ["results/peers_*.json"],
    "gate": ["results/gate/gate_*Z.json"],
    "bench": ["results/bench/*.jsonl", "results/tokenizer_*.json", "results/kl_8v4_*.json"],
    "plan": ["results/plan_fixed_*.json", "results/AMENDMENT_*_*.md"],
}
DEFAULT_STATUS = {
    "sign-off": "signed off; Session 1 in progress; no scored run",
    "preflight": "preflight recorded; Session 1 in progress; no scored run",
    "manifests": "item manifests built; Session 1 in progress; no scored run",
    "convert": "K8/K4 converted; Session 1 in progress; no scored run",
    "peers": "peer check recorded; Session 1 in progress; no scored run",
    "bench": "S1 bench cells recorded; Session 1 in progress; no scored run",
    "plan": "plan fixed by rule; Session 2 next",
    "S2": "Session 2 recorded; Session 3 next",
}


def _newest(exp: Path, pattern: str) -> list[Path]:
    """Every match for *.jsonl patterns, otherwise the newest match (mtime, then name)."""
    hits = sorted(exp.glob(pattern))
    if not hits:
        return []
    if pattern.endswith("*.jsonl"):
        return hits
    return [max(hits, key=lambda h: (h.stat().st_mtime, h.name))]


def _phase_files(exp: Path, phase: str) -> list[Path]:
    key = phase.split(",")[0].strip()
    if key in PHASE_FILES:
        out = []
        for pat in PHASE_FILES[key]:
            out.extend(_newest(exp, pat))
        return out
    m = re.match(r"^(S\d+b?)\b", key)
    if m:   # a session: its raw files (and scores if present)
        sess = m.group(1)
        out = sorted((exp / "results" / "raw" / sess).rglob("*.jsonl"))
        if "score" in phase:
            out += sorted((exp / "results" / "scores").rglob("*.jsonl"))
        return out
    return []


def _gate_verdict(path: Path) -> str | None:
    try:
        g = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    v = g.get("verdict") or g.get("verdicts")
    if isinstance(v, dict):
        return ", ".join(f"{k} {v[k]}" for k in sorted(v))
    return str(v) if v is not None else None


def set_status(hypothesis, text: str) -> str:
    """Rewrite only the Status line; returns the old status text."""
    path = Path(hypothesis)
    lines = path.read_text(encoding="utf-8").split("\n")
    for i, line in enumerate(lines):
        m = STATUS_RE.match(line)
        if m:
            old = m.group(2)
            lines[i] = f"{m.group(1)}{text}{m.group(3)}"
            tmp = path.with_name(path.name + ".status.tmp")
            tmp.write_text("\n".join(lines), encoding="utf-8")
            tmp.replace(path)
            return old
    raise ValueError(f"{path.name}: no '*Pre-registered: … · Status: …*' line")


def _append(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    sep = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    with open(path, "a", encoding="utf-8") as f:
        f.write(sep + block.rstrip("\n") + "\n")


def record_block(phase: str, hypothesis=None, files=None, status: str | None = None,
                 exp_dir=None, when=None) -> str:
    """Append the run-record block and update the Status line. Returns the block."""
    exp = Path(exp_dir or common.EXP_DIR)
    hyp = Path(hypothesis or exp / "HYPOTHESIS.md")
    utc = common.utc_iso(when)
    gstate = common.git_state(cwd=exp)
    paths = [Path(f) for f in files] if files is not None else _phase_files(exp, phase)
    paths = [p if p.is_absolute() else exp / p for p in paths]
    lines = [f"## {phase} — run record ({utc})", "",
             f"- Phase: {phase}",
             f"- UTC: {utc} (from the clock)",
             f"- Commit: {gstate['head'] or 'none'} (uncommitted changes: {'yes' if gstate['dirty'] else 'no'})"]
    if paths:
        lines.append("- Result files (sha256):")
        for p in paths:
            try:
                rel = p.resolve().relative_to(exp.resolve()).as_posix()
            except ValueError:
                from tools.redact import redact_path

                rel = redact_path(p)
            lines.append(f"  - `{rel}` `{common.sha256_file(p)}`")
    else:
        lines.append("- Result files: none")
    if phase.split(",")[0].strip() == "gate":
        gate_files = [p for p in paths if p.name.startswith("gate_") and p.suffix == ".json"]
        verdict = _gate_verdict(gate_files[-1]) if gate_files else None
        lines.append(f"- Gate verdict: {verdict or 'not found'}")
        if status is None and verdict:
            status = f"gate {verdict}; Session 1 in progress; no scored run"
    block = "\n".join(lines) + "\n"
    _append(hyp, block)
    new_status = status or DEFAULT_STATUS.get(phase.split(",")[0].strip()) or f"{phase} recorded ({utc})"
    set_status(hyp, new_status)
    return block


def sync_amendments(hypothesis=None, exp_dir=None) -> list[str]:
    """Append missing amendments/*.md verbatim in number order; returns the
    headings appended (empty list = no-op). Raises ValueError, appending
    nothing, if a missing one still holds a {{PLACEHOLDER}}."""
    exp = Path(exp_dir or common.EXP_DIR)
    hyp = Path(hypothesis or exp / "HYPOTHESIS.md")
    adir = exp / "amendments"
    if not adir.is_dir():
        return []
    present = set(hyp.read_text(encoding="utf-8").splitlines())
    todo = []
    for f in adir.glob("*.md"):
        text = f.read_text(encoding="utf-8")
        first = text.splitlines()[0].rstrip() if text.strip() else ""
        m = AMEND_HEAD_RE.match(first)
        if not m:
            raise ValueError(f"{f.name}: first line is not '## Amendment k — <type> (<UTC>)'")
        if first not in present:
            _refuse_placeholder(f.name, text)
            todo.append((int(m.group(1)), f.name, first, text))
    appended = []
    for _, _, first, text in sorted(todo):
        _append(hyp, text)
        appended.append(first)
    return appended


def append_amendment(path, hypothesis=None, exp_dir=None) -> str:
    """Append one amendment file verbatim; returns its heading. Refuses a file whose first line is not
    "## Amendment k — <type> (<UTC>)", whose heading HYPOTHESIS.md already holds, or that still holds a
    {{PLACEHOLDER}}."""
    exp = Path(exp_dir or common.EXP_DIR)
    hyp = Path(hypothesis or exp / "HYPOTHESIS.md")
    text = Path(path).read_text(encoding="utf-8")
    first = text.splitlines()[0].rstrip() if text.strip() else ""
    if not AMEND_HEAD_RE.match(first):
        raise ValueError(f"{Path(path).name}: first line is not '## Amendment k — <type> (<UTC>)'")
    if first in set(hyp.read_text(encoding="utf-8").splitlines()):
        raise ValueError(f"{hyp.name} already holds {first!r}")
    _refuse_placeholder(Path(path).name, text)
    _append(hyp, text)
    return first


def _refuse_placeholder(name: str, text: str) -> None:
    m = PLACEHOLDER_RE.search(text)
    if m:
        raise ValueError(f"{name}: {m.group(0)} is not filled in; nothing appended (the main session fills it "
                         f"and pushes again)")


def _resolve_private(redacted: str, private: Path) -> Path | None:
    prefix = "$EXP036_PRIVATE"
    if redacted == prefix:
        return private
    if redacted.startswith(prefix + "/"):
        return private / redacted[len(prefix) + 1:]
    return None


def verify_private(exp_dir=None, private_dir=None) -> dict:
    """Mirror check for RUNBOOK step 18 (run on the mini). The latest record per path in
    evidence/withheld_manifest.jsonl wins; a "private_moved" record points to the new path."""
    exp = Path(exp_dir or common.EXP_DIR)
    private = Path(private_dir or common.private_dir()).expanduser()
    man = exp / "evidence" / "withheld_manifest.jsonl"
    latest: dict[str, dict] = {}
    if man.is_file():
        for line in man.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("type") in ("private_file", "private_moved") and r.get("path"):
                latest[r["path"]] = r
    expected: dict[str, str] = {}
    for path, r in latest.items():
        target = r["to"] if r.get("type") == "private_moved" else path
        expected[target] = r["sha256"]
    missing, mismatch, unresolved, ok = [], [], [], 0
    for target, sha in sorted(expected.items()):
        f = _resolve_private(target, private)
        if f is None:
            unresolved.append(target)
        elif not f.is_file():
            missing.append(target)
        elif common.sha256_file(f) != sha:
            mismatch.append(target)
        else:
            ok += 1
    listed = {t for t in expected}
    extras_expected, extras_other = [], []
    if private.is_dir():
        for f in sorted(x for x in private.rglob("*") if x.is_file() and x.name != ".DS_Store"):
            rel = "$EXP036_PRIVATE/" + f.relative_to(private).as_posix()
            if rel in listed:
                continue
            (extras_expected if rel.split("/")[1] in ("manifests", "aborted") else extras_other).append(rel)
    return {"ok": not (missing or mismatch or unresolved) and private.is_dir(), "files_listed": len(expected),
            "verified": ok, "missing": missing, "sha256_mismatch": mismatch, "unresolved": unresolved,
            "expected_extras": len(extras_expected), "other_extras": extras_other,
            "private_dir_exists": private.is_dir()}


def run_status(exp_dir=None) -> int:
    exp = Path(exp_dir or common.EXP_DIR)
    run_py = exp / "runner" / "run.py"
    if not run_py.is_file():
        print("status: runner/run.py is not present", file=sys.stderr)
        return 3
    return subprocess.call([sys.executable, str(run_py), "status"], cwd=str(exp))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="exp_036 status and HYPOTHESIS run records (BUILD_SPEC §5.9)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--record-block", metavar="PHASE")
    g.add_argument("--sync-amendments", action="store_true")
    g.add_argument("--set-status", metavar="TEXT")
    g.add_argument("--append-amendment", metavar="FILE")
    g.add_argument("--verify-private", action="store_true")
    g.add_argument("--tier2", action="store_true")
    ap.add_argument("--private", default=None, help="private directory for --verify-private (default $EXP036_PRIVATE)")
    ap.add_argument("--files", nargs="*", default=None, help="result files for --record-block")
    ap.add_argument("--status", default=None, help="Status text for --record-block")
    ap.add_argument("--hypothesis", default=None)
    args = ap.parse_args(argv)
    try:
        if args.record_block:
            block = record_block(args.record_block, args.hypothesis, args.files, args.status)
            print(block)
            return 0
        if args.sync_amendments:
            done = sync_amendments(args.hypothesis)
            print("\n".join(f"appended: {h}" for h in done) if done else "no amendments to append")
            return 0
        if args.set_status:
            old = set_status(args.hypothesis or common.EXP_DIR / "HYPOTHESIS.md", args.set_status)
            print(f"Status: {old!r} -> {args.set_status!r}")
            return 0
        if args.append_amendment:
            print(f"appended: {append_amendment(args.append_amendment, args.hypothesis)}")
            return 0
        if args.tier2:
            from runner import guard

            ok = guard.tier2_at_head(common.EXP_DIR)
            print(f"Tier-2 analysis amendment at HEAD: {'yes' if ok else 'no'}")
            return 0 if ok else 1
        if args.verify_private:
            res = verify_private(private_dir=args.private)
            print(json.dumps(res, indent=1, sort_keys=True))
            return 0 if res["ok"] else 1
    except (ValueError, OSError) as e:
        print(f"status: {e}", file=sys.stderr)
        return 1
    return run_status()


if __name__ == "__main__":
    sys.exit(main())
