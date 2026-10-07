"""Tree and file hashes of the HYPOTHESIS "Fixed before any run" table
(exp_036 BUILD_SPEC §5.9 `tools/hash_tree.py`; exp_037 DESIGN §9.1-§9.2).

exp_037 adds three scopes to exp_036's 17. `gate_rules` (GATE_RULES_SHA256, row
"Gate rules (frozen)"): the tree of gate/ restricted to calibration.json,
controls.json, port_mutants.py, rules.py and thresholds.json, i.e. every
pass/fail decision, threshold, control and mutant of the gate. `tests`
(TESTS_SHA256, row "Tests"): the tests/ tree, i.e. G1's weight-free suite, its
skip policy (tests/conftest.py), the heavy tiny tests and the fixtures; and
`env` (ENV_SHA256, row "Environment scripts"): env/exp037.settings.sh,
env/setup.sh and env/versions.json, the three env/ files outside the Runtime
row (final review W15-02, 2026-10-06: before it, G1 could be relaxed after the
freeze, by editing or skipping a test, without an amendment line). Both are
amendable like gate code: a gate fix carries a test, so its amendment gives a
`hash_tree: TESTS_SHA256` line. The table has 20 rows. No amendment may change
`gate_rules`, `thresholds` or `gate_text` (NO_AMENDMENT_SCOPES):
gate/preconditions.py P1(b) refuses an amendment line for any of them, and
`--amend-line` refuses to print one.

Definitions (frozen with the TOOLS tree):

- A **file** value is the sha256 of the file's bytes.
- A **tree** value is the sha256 of the UTF-8 text made of one line
  "relpath\\tsha256\\n" per file, sorted by relpath (bytewise, '/' separators),
  where relpath is relative to the tree's root directory and sha256 is the
  file value. Files that git ignores are left out (so a tree hashes the same
  before and after a clone), and so are `__pycache__/`, `.pytest_cache/`,
  `*.pyc`, `*.pyo` and `.DS_Store` everywhere.
- The scopes are exactly the rows of the HYPOTHESIS table, in `SCOPES` below:
  each names its placeholder, the table row it sits in, its position among the
  64-hex values of that row, and its include / exclude sets.

CLI:
    hash_tree.py --tree <name>             print one scope's value
    hash_tree.py --amend-line <name>       print "hash_tree: <PLACEHOLDER> = <hex>"
                                           for an amendment that changes a frozen tree
                                           (refused for gate_rules, thresholds, gate_text)
    hash_tree.py --list                    print every scope's current value
    hash_tree.py --fill HYPOTHESIS.md      fill every hash placeholder (all or nothing)
    hash_tree.py --stamp HYPOTHESIS.md     write the UTC time into {{PREREG_UTC}}
    hash_tree.py --check [HYPOTHESIS.md]   compare current values with the registered
                                           ones (the filled table, or the latest
                                           "hash_tree: NAME = hex" line of a numbered
                                           amendment); exit 1 on any difference, or
                                           when HYPOTHESIS.md does not exist (every
                                           scope is then "unfilled")

Exit codes: 0 ok, 1 refused / mismatch, 2 usage.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import common

GLOBAL_EXCLUDE_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git"}
GLOBAL_EXCLUDE_SUFFIXES = (".pyc", ".pyo")
GLOBAL_EXCLUDE_NAMES = {".DS_Store"}

HEX64 = re.compile(r"\b[0-9a-f]{64}\b")
PLACEHOLDER = re.compile(r"\{\{([A-Z0-9_]+)\}\}")
STAMP_PLACEHOLDER = "PREREG_UTC"
AMEND_LINE = re.compile(r"hash_tree:\s*([A-Za-z0-9_]+)\s*=\s*`?([0-9a-f]{64})`?")
AMEND_HEADING = re.compile(r"^## Amendment (\d+)\b")


@dataclass(frozen=True)
class Scope:
    name: str                 # --tree name
    placeholder: str          # {{...}} in HYPOTHESIS.md
    row: str                  # first cell of the HYPOTHESIS table row
    ordinal: int              # position among the 64-hex values of that row
    kind: str                 # "file" | "tree"
    path: str                 # file path, or tree root, relative to the experiment dir
    include: tuple = field(default=())   # tree: only these relpaths (empty = all)
    exclude: tuple = field(default=())   # tree: relpaths; a trailing "/" excludes a directory


# HYPOTHESIS.md "Fixed before any run (hashes)", row by row.
SCOPES: tuple[Scope, ...] = (
    Scope("assets", "ASSETS_JSON_SHA256", "Asset manifest", 0, "file", "assets.json"),
    Scope("evalfw", "EVALFW_MANIFEST_SHA256", "eval-framework", 0, "file",
          "tasks/vendored_evalfw/MANIFEST.json"),
    Scope("requirements", "REQUIREMENTS_SHA256", "Runtime", 0, "file", "env/requirements-mbp.txt"),
    Scope("port", "PORT_SHA256", "Port", 0, "file", "port/kolibri1.py"),
    Scope("convert", "CONVERT_SHA256", "Port", 1, "file", "port/convert.py"),
    Scope("reference", "REFERENCE_SHA256", "Reference", 0, "tree", "reference"),
    Scope("gate", "GATE_CODE_SHA256", "Gate code", 0, "tree", "gate",
          exclude=("thresholds.json", "texts/")),
    Scope("thresholds", "THRESHOLDS_SHA256", "Gate thresholds", 0, "file", "gate/thresholds.json"),
    # exp_037 (DESIGN §9.2): the frozen decision scope. Overlaps "gate" and "thresholds" by design.
    Scope("gate_rules", "GATE_RULES_SHA256", "Gate rules (frozen)", 0, "tree", "gate",
          include=("calibration.json", "controls.json", "port_mutants.py", "rules.py", "thresholds.json")),
    Scope("gate_text", "GATE_TEXT_SHA256", "Gate text", 0, "file", "gate/texts/MANIFEST.json"),
    Scope("runner", "RUNNER_SHA256", "Runner", 0, "tree", "runner"),
    Scope("tasks", "TASKS_CODE_SHA256", "Tasks code", 0, "tree", "tasks", exclude=("manifests/",)),
    Scope("manifest_rule", "MANIFEST_RULE_SHA256", "Manifest rule", 0, "tree", "tasks",
          include=("build_manifests.py", "selection_rules.json")),
    Scope("plan_rules", "PLAN_RULES_SHA256", "Plan rules", 0, "file", "runner/plan_rules.json"),
    Scope("scorers", "SCORERS_SHA256", "Scorers", 0, "tree", "scorers"),
    Scope("analysis", "ANALYSIS_SHA256", "Analysis / verdict code (Tier 1)", 0, "tree", "analysis",
          exclude=("exploratory.py", "tables.py")),
    Scope("bench", "BENCH_SHA256", "Bench", 0, "tree", "bench", exclude=("ladder.py",)),
    Scope("tools", "TOOLS_SHA256", "Tools", 0, "tree", "tools", exclude=("withheld_shingles.sha256",)),
    # exp_037, final review W15-02: G1's suite (and every other test) and the environment scripts. Amendable.
    Scope("tests", "TESTS_SHA256", "Tests", 0, "tree", "tests"),
    # Exactly these three files (requirements-mbp.txt is the Runtime row's); nothing else under env/ is read.
    Scope("env", "ENV_SHA256", "Environment scripts", 0, "tree", "env",
          include=("exp037.settings.sh", "setup.sh", "versions.json")),
)
BY_NAME = {s.name: s for s in SCOPES}
BY_PLACEHOLDER = {s.placeholder: s for s in SCOPES}
# DESIGN §9.2: no amendment may change these (P1(b) refuses an amendment line naming one).
NO_AMENDMENT_SCOPES = ("gate_rules", "thresholds", "gate_text")


class MissingScope(FileNotFoundError):
    pass


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------

def _globally_excluded(rel: str) -> bool:
    parts = rel.split("/")
    if any(p in GLOBAL_EXCLUDE_DIRS for p in parts[:-1]):
        return True
    name = parts[-1]
    return name in GLOBAL_EXCLUDE_NAMES or name.endswith(GLOBAL_EXCLUDE_SUFFIXES)


def _git_listing(root: Path) -> list[str] | None:
    """Files under root that git would commit (tracked or untracked, not
    ignored), relative to root; None if root is not in a git work tree."""
    rc, out, _ = common.run(
        ["git", "-c", "core.quotepath=off", "ls-files", "-z", "-c", "-o", "--exclude-standard", "--", "."],
        cwd=str(root),
    )
    if rc != 0:
        return None
    rels = sorted({r for r in out.split("\0") if r})
    return [r for r in rels if (root / r).is_file()]


def list_files(root, use_git: bool = True) -> list[str]:
    """Sorted relpaths of the files of a tree (see the module docstring)."""
    root = Path(root)
    rels = _git_listing(root) if use_git else None
    if rels is None:
        rels = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames if d not in GLOBAL_EXCLUDE_DIRS)
            for fn in filenames:
                rels.append(Path(dirpath, fn).relative_to(root).as_posix())
    rels = [r for r in rels if not _globally_excluded(r)]
    return sorted(rels, key=lambda r: r.encode("utf-8"))


def _selected(rel: str, include: tuple, exclude: tuple) -> bool:
    """include: exact relpaths. exclude: a relpath excludes that file and, as a
    directory, everything below it ("texts/" and "texts" both exclude texts/)."""
    if include and rel not in include:
        return False
    for e in exclude:
        d = e.rstrip("/")
        if rel == d or rel.startswith(d + "/"):
            return False
    return True


def tree_lines(root, include=(), exclude=(), use_git: bool = True) -> list[str]:
    root = Path(root)
    lines = []
    for rel in list_files(root, use_git=use_git):
        if _selected(rel, tuple(include), tuple(exclude)):
            lines.append(f"{rel}\t{common.sha256_file(root / rel)}\n")
    return lines


def tree_sha256(root, include=(), exclude=(), use_git: bool = True) -> str:
    """sha256 over sorted "relpath\\tsha256\\n" lines (BUILD_SPEC §5.9)."""
    root = Path(root)
    if not root.is_dir():
        raise MissingScope(f"{root} is not a directory")
    lines = tree_lines(root, include, exclude, use_git=use_git)
    if include:
        found = {ln.split("\t", 1)[0] for ln in lines}
        missing = sorted(set(include) - found)
        if missing:
            raise MissingScope(f"{root}: missing {missing}")
    if not lines:
        raise MissingScope(f"{root} has no files")
    return common.sha256_bytes("".join(lines).encode("utf-8"))


def manifest_sha256(files: dict) -> str:
    """The same rule for a {relpath: sha256} mapping (e.g. a converted
    model directory's per-file hashes)."""
    lines = sorted((f"{k}\t{v}\n" for k, v in files.items()), key=lambda s: s.encode("utf-8"))
    return common.sha256_bytes("".join(lines).encode("utf-8"))


def scope_value(scope: Scope, root=None, use_git: bool = True) -> str:
    root = Path(root or common.EXP_DIR)
    target = root / scope.path
    if scope.kind == "file":
        if not target.is_file():
            raise MissingScope(f"{scope.path} does not exist")
        return common.sha256_file(target)
    try:
        return tree_sha256(target, scope.include, scope.exclude, use_git=use_git)
    except MissingScope as e:   # experiment-relative, so a record that carries the message names no home path
        raise MissingScope(str(e).replace(str(target), scope.path)) from None


def all_values(root=None, use_git: bool = True) -> tuple[dict, dict]:
    """({name: hex}, {name: error}) for every scope."""
    values, errors = {}, {}
    for s in SCOPES:
        try:
            values[s.name] = scope_value(s, root, use_git=use_git)
        except MissingScope as e:
            errors[s.name] = str(e)
    return values, errors


# --------------------------------------------------------------------------
# HYPOTHESIS.md: fill, stamp, registered values
# --------------------------------------------------------------------------

def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".hash_tree.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def fill(hypothesis, root=None, use_git: bool = True) -> dict:
    """Replace every hash placeholder; all or nothing. Returns {placeholder: hex}.

    Raises MissingScope if any placeholder's scope cannot be hashed, and
    ValueError for an unknown placeholder. {{PREREG_UTC}} is left for --stamp.
    A file without hash placeholders is left unchanged (idempotent)."""
    path = Path(hypothesis)
    text = path.read_text(encoding="utf-8")
    present = [m.group(1) for m in PLACEHOLDER.finditer(text)]
    unknown = sorted({p for p in present if p not in BY_PLACEHOLDER and p != STAMP_PLACEHOLDER})
    if unknown:
        raise ValueError(f"unknown placeholders in {path.name}: {unknown}")
    wanted = [p for p in dict.fromkeys(present) if p in BY_PLACEHOLDER]
    if not wanted:
        return {}
    values, errors = {}, []
    for p in wanted:
        try:
            values[p] = scope_value(BY_PLACEHOLDER[p], root, use_git=use_git)
        except MissingScope as e:
            errors.append(f"{p}: {e}")
    if errors:
        raise MissingScope("cannot fill, nothing written:\n  " + "\n  ".join(errors))
    new = PLACEHOLDER.sub(lambda m: values.get(m.group(1), m.group(0)), text)
    _atomic_write(path, new)
    return values


def stamp(hypothesis, when=None) -> str | None:
    """Write the UTC time (YYYY-MM-DDTHH:MM:SSZ) into {{PREREG_UTC}}. Returns
    the value written, or None if there is no placeholder left."""
    path = Path(hypothesis)
    text = path.read_text(encoding="utf-8")
    token = "{{" + STAMP_PLACEHOLDER + "}}"
    if token not in text:
        return None
    value = common.utc_iso(when)
    _atomic_write(path, text.replace(token, value))
    return value


STAMP_LINE = re.compile(r"^(\*Pre-registered: )(?!\{\{)([^ ·*]+)( · Status: )", re.M)


def unfill(text: str) -> str:
    """The pre-registration template of a filled HYPOTHESIS.md text (a pure function).

    Every scope's 64-hex value in the "Fixed before any run" table goes back to
    its {{PLACEHOLDER}}, the stamp goes back to {{PREREG_UTC}}, and every
    amendment line "hash_tree: NAME = hex" is neutralised, so that fill() and
    check() on a copy over another tree start from a clean slate. Used by the
    round-trip test and by tools/dry_run.py on a copy of an already filled kit
    (review fix 2026-10-03); never on the real file."""
    by_row: dict[str, list[Scope]] = {}
    for s in SCOPES:
        by_row.setdefault(s.row, []).append(s)
    rows = _table_rows(text)
    lines = text.split("\n")
    for row, scopes in by_row.items():
        line = rows.get(row)
        if line is None:
            continue
        hexes = list(HEX64.finditer(line))
        new = line
        for s in sorted(scopes, key=lambda x: x.ordinal, reverse=True):
            if "{{" + s.placeholder + "}}" in line or len(hexes) <= s.ordinal:
                continue
            m = hexes[s.ordinal]
            new = new[:m.start()] + "{{" + s.placeholder + "}}" + new[m.end():]
        if new != line:
            lines[lines.index(line)] = new
    text = "\n".join(lines)
    text = STAMP_LINE.sub(lambda m: m.group(1) + "{{" + STAMP_PLACEHOLDER + "}}" + m.group(3), text, count=1)
    return AMEND_LINE.sub(lambda m: f"hash_tree value of {m.group(1)} (neutralised in this copy)", text)


def _table_rows(text: str) -> dict[str, str]:
    """First cell -> whole line, for the rows of the "Fixed before any run" table."""
    rows: dict[str, str] = {}
    in_section = False
    for line in text.splitlines():
        if line.startswith("## "):
            in_section = line.startswith("## Fixed before any run")
            continue
        if in_section and line.startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if cells and cells[0] not in rows:
                rows[cells[0]] = line
    return rows


def registered_values(hypothesis) -> dict:
    """{name: {"value": hex | None, "source": str}}.

    The table gives the pre-registered value; a later "hash_tree: NAME = hex"
    line inside a "## Amendment k" section replaces it (the last one wins).
    A HYPOTHESIS.md that does not exist (yet) registers nothing: every value is
    None with the source "HYPOTHESIS.md not found"."""
    path = Path(hypothesis)
    if not path.is_file():
        return {s.name: {"value": None, "source": f"{path.name} not found"} for s in SCOPES}
    text = path.read_text(encoding="utf-8")
    rows = _table_rows(text)
    out = {}
    for s in SCOPES:
        line = rows.get(s.row)
        if line is None:
            out[s.name] = {"value": None, "source": f"row {s.row!r} not found"}
            continue
        if "{{" + s.placeholder + "}}" in line:
            out[s.name] = {"value": None, "source": "unfilled"}
            continue
        hexes = HEX64.findall(line)
        if len(hexes) > s.ordinal:
            out[s.name] = {"value": hexes[s.ordinal], "source": "pre-registration"}
        else:
            out[s.name] = {"value": None, "source": f"no value at position {s.ordinal} of row {s.row!r}"}
    amendment = None
    for line in text.splitlines():
        h = AMEND_HEADING.match(line)
        if h:
            amendment = int(h.group(1))
            continue
        if line.startswith("## ") and not h:
            amendment = None
        if amendment is None:
            continue
        for m in AMEND_LINE.finditer(line):
            key = m.group(1)
            scope = BY_NAME.get(key) or BY_PLACEHOLDER.get(key.upper())
            if scope:
                out[scope.name] = {"value": m.group(2), "source": f"amendment {amendment}"}
    return out


def check(hypothesis=None, root=None, use_git: bool = True) -> dict:
    """{"ok": bool, "scopes": {name: {current, registered, source, status}}}.
    status: match | MISMATCH | unfilled | missing."""
    hypothesis = Path(hypothesis or (Path(root or common.EXP_DIR) / "HYPOTHESIS.md"))
    reg = registered_values(hypothesis)
    current, errors = all_values(root, use_git=use_git)
    scopes = {}
    ok = True
    for s in SCOPES:
        r = reg[s.name]
        cur = current.get(s.name)
        if cur is None:
            status = "missing"
        elif r["value"] is None:
            status = "unfilled"
        elif r["value"] == cur:
            status = "match"
        else:
            status = "MISMATCH"
        ok = ok and status == "match"
        scopes[s.name] = {
            "current": cur, "registered": r["value"], "source": r["source"], "status": status,
            **({"error": errors[s.name]} if s.name in errors else {}),
        }
    return {"ok": ok, "scopes": scopes}


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="exp_037 tree hashes (exp_036 BUILD_SPEC §5.9; DESIGN §9.2)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tree", choices=sorted(BY_NAME))
    g.add_argument("--amend-line", choices=sorted(BY_NAME))
    g.add_argument("--list", action="store_true")
    g.add_argument("--fill", metavar="HYPOTHESIS.md")
    g.add_argument("--stamp", metavar="HYPOTHESIS.md")
    g.add_argument("--check", nargs="?", const="", metavar="HYPOTHESIS.md")
    ap.add_argument("--root", default=None, help="experiment directory (default: this kit's)")
    args = ap.parse_args(argv)
    root = Path(args.root) if args.root else common.EXP_DIR

    try:
        if args.amend_line in NO_AMENDMENT_SCOPES:
            print(f"hash_tree: {BY_NAME[args.amend_line].placeholder} is frozen without amendment "
                  f"(DESIGN §9.2: no amendment may change {', '.join(NO_AMENDMENT_SCOPES)}); no line printed",
                  file=sys.stderr)
            return 1
        if args.tree or args.amend_line:
            name = args.tree or args.amend_line
            value = scope_value(BY_NAME[name], root)
            print(value if args.tree else f"hash_tree: {BY_NAME[name].placeholder} = {value}")
            return 0
        if args.list:
            values, errors = all_values(root)
            for s in SCOPES:
                print(f"{s.name:14s} {values.get(s.name) or 'MISSING: ' + errors[s.name]}")
            return 0 if not errors else 1
        if args.fill:
            filled = fill(args.fill, root)
            print(f"filled {len(filled)} placeholders" if filled else "nothing to fill")
            return 0
        if args.stamp:
            v = stamp(args.stamp)
            print(f"stamped {v}" if v else "no {{PREREG_UTC}} placeholder; nothing stamped")
            return 0
        res = check(args.check or None, root)
        for name, r in res["scopes"].items():
            print(f"{r['status']:9s} {name:14s} current {str(r['current'])[:16]}  "
                  f"registered {str(r['registered'])[:16]} ({r['source']})")
        return 0 if res["ok"] else 1
    except (MissingScope, ValueError) as e:
        print(f"hash_tree: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
