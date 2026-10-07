#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""exp_037: diagnostics of gate run 20261007T110355Z's G1 failure (pre-registered in PLAN.md, same folder).

G1 failed on one test, tests/test_runner_tiny.py::test_resume_after_simulated_crash_gives_identical_records
(the assertion at line 222: content(records(p)) == content(records(ref))). This script runs the experiments that
PLAN.md registers and applies PLAN.md's outcome-to-action table to their outputs. The table and every rule are in
PLAN.md; this file implements them and decides nothing on its own.

  --experiment cachestate (iv)  the tokenizer-check cache probe: state that kit code leaves in
                                scorers.reasoning._CHECKED and that runner.generate._scorer_split reads
  --experiment isolated   (i)   the single test, -vv, in N separate pytest processes
  --experiment module     (i')  the whole tests/test_runner_tiny.py module, G1's flags, N processes
  --experiment pair       (i'') tests/test_bench_batch_flip.py then tests/test_runner_tiny.py, G1's flags, N processes
  --experiment g1ctx      (ii)  G1's own selection and invocation (gate/checks/g1_synthetic.py), K runs
  --experiment inproc     (iii) one process; the model loaded once, as the test's fixture loads it; N iterations
                                of A (uninterrupted), C (stop after three, torn tail, resume), B (uninterrupted)
  --experiment fresh      L2    inproc with --n 1 in M separate processes (a follow-up: PLAN.md section 6.2)
  --classify                    the outcome-to-action table (PLAN.md section 5) on out/*.json
  --selftest                    stub models only: imports nothing from the kit, no MLX, no port, no tiny checkpoint

It writes no gate record and nothing under results/. Outputs: out/<host>_<experiment>_<UTC>.json (redacted; the
leak check runs before they are committed). Work files (junit XML, pytest logs, JSONL copies, tiny checkpoints of
(iii) and (iv)) go to --work, outside the repository. Nothing is ever deleted.
"""

from __future__ import annotations

import argparse
import ast
import gc
import getpass
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import weakref
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.dont_write_bytecode = True

# ----------------------------------------------------------------------------------------------- the package

PKG_DIR = Path(__file__).resolve().parent
KIT_DIR = PKG_DIR.parent.parent
TESTS_DIR = KIT_DIR / "tests"
REPO_DIR = KIT_DIR.parents[2]
OFFICIAL_OUT = PKG_DIR / "out"
SCHEMA = "exp037 g1-resume diagnostics v2"
PACKAGE = "diagnostics/g1_resume_20261007"
PACKAGE_FILES = ("PLAN.md", "diag_resume.py")  # the pre-registration; README.md is descriptive

# The gate run this package diagnoses (results/gate/gate_20261007T110355Z.json, committed in 1efbadc).
GATE_UTC = "20261007T110355Z"
GATE_RECORD = "results/gate/gate_20261007T110355Z.json"
GATE_RECORD_SHA256 = "cb21c2dfa461c0cd17b68f47e9577cca248062bf3a7c6becbbfbd90dc9bdd809"
GATE_CODE_COMMIT = "8dea1dc934e9c8cf1796e2e59a2b1f16213100ca"  # the record's git.head: the code G1 ran
# The record's code_hashes, by tools/hash_tree.py scope (recomputed before every run: PLAN.md 3.0).
GATE_CODE_HASHES = {
    "port": ("port", "2c153357862f15b61f3cadaea4f436de2939567180bf5a01a304f3fa60e3f182"),
    "runner": ("runner_tree", "439590de6a5c7971fc167e8918f9570f142446bd7046e49fac8c38cac8b45d74"),
    "gate": ("gate_code", "1d41328359b9c6af3c58b51894cfd1f38ac028f79c6866cfcf246d293ac8edbf"),
    "reference": ("reference_tree", "85337ed7efeef6b0463f3560fff6da6496661a2fa7a14de5e50a1bce9e251bbd"),
}
# Drafted against this commit (the mini's HEAD when the package was written).
WRITTEN_AGAINST = "1efbadca1c4eaf70f0399950a570069149287e22"
# Andrei's choice after the gate run, as relayed to the main session. The mbp session records it in HYPOTHESIS.md.
# CHOICE_COMMIT is set when the package is committed: the commit that holds that entry (PLAN.md header and 9.1).
# Until it is set, and while that commit's HYPOTHESIS.md lacks CHOICE_UTC and CHOICE_LABEL, every run is refused.
CHOICE_UTC = "2026-10-07T14:49:52Z"
CHOICE_LABEL = "b · diagnostics"
CHOICE_COMMIT = "d60f3b6ec857b021a540d847a6594ef14a8d554d"

# Since GATE_CODE_COMMIT only these kit paths may differ (git diff and git status), or the run is refused.
ALLOWED_PREFIXES = ("results/", "diagnostics/", "aborted/", "evidence/", "amendments/")
ALLOWED_FILES = ("HYPOTHESIS.md", "BUILD_LOG.md")
# Read and pinned when this package was written (all unchanged since GATE_CODE_COMMIT).
PINNED_SHA256 = {
    "tests/test_runner_tiny.py": "35d1d5ff78958e23d126986d09064c42d385c92a20d8ce0818cfcac16f44878f",
    "tests/conftest.py": "ba84336d32d95b9d2761fad3b47329ca8bed842d8082a72d714251bd5d61d507",
    "tests/tiny_checkpoint.py": "8f339b6f564c62b18241c7a3fe9d92ee15ff7374083f8ddc48950eeaf0fef78c",
    "tests/port_harness.py": "8c0f09d1ff4807fc49a53fedb9094a691fdf0cb1bb1c64f84e4804fedfc11830",
    "tests/test_bench_batch_flip.py": "3e3cd8fe4ce9bc8a38cde3cbf73620de04489a5e30e497272bcdb1ee3d6434e3",
    "gate/checks/g1_synthetic.py": "eb378cceb2312e230d7c6dcb8302d597f8992c9929cc34cecaebe4c1baa2fa8a",
    "runner/generate.py": "e1cce3fb76de8c0fffe360aa8127c70efac93ee44d5ff0da1afd9035632addc5",
    "runner/jsonl.py": "6bd53bae0e130a462fb4f8962892a0c17da6d58fcbe95bafe38a25b10d40c49d",
    "scorers/reasoning.py": "820defab2247d38aa988fbb44f258c1574e6b74555ed87a04b480c724dd704e5",
    "bench/batch_flip.py": "fac57ee716addc7a4a6274d5d8ed91c84ebf7d78da5f8928c823fe4db2122f7f",
    "port/kolibri1.py": "2c153357862f15b61f3cadaea4f436de2939567180bf5a01a304f3fa60e3f182",
}

# The failing test (tests/test_runner_tiny.py lines 40, 43-51, 201-223).
TEST_REL = "tests/test_runner_tiny.py"
TEST_NAME = "test_resume_after_simulated_crash_gives_identical_records"
TEST_LINE = 222
C_LINES = (213, 217, 219, 221, 223)  # the test's other assertions on the stop / torn tail / resume path
TMP_PREFIX = TEST_NAME[:30]  # pytest's tmp_path directory name (_pytest/tmpdir.py _mk_tmp: MAXVAL 30)
CELL_FILE = Path("raw/S2/K8/mmlu_en_high.jsonl")
TORN = b'{"completion_ids": [1, 2, 3], "key": {"it'
TIMING = ("t_submit", "t_first_token", "t_done", "wall_s", "batch_id")
# The record schema (tests/test_runner_tiny.py SCHEMA; exp_036 BUILD_SPEC section 5.4).
SCHEMA_FIELDS = (
    "type", "key", "item_sha256", "prompt_sha256", "rendered_sha256", "rendered_prompt_tokens", "prompt_prefill",
    "completion_ids", "completion_tokens", "reasoning_tokens", "answer_tokens", "text", "reasoning_text",
    "answer_text", "finish_reason", "truncated", "reasoning_status", "split_by", "stop_token", "t_submit",
    "t_first_token", "t_done", "wall_s", "batch_id", "batch_size", "cell_seed", "max_tokens", "sampler", "eos_ids")
# Fields that depend on process state (PLAN.md section 4): the reasoning split that runner.generate._scorer_split
# selects (it reads scorers.reasoning._CHECKED, keyed by id(tokenizer)), and the finish reason mlx-lm reports.
SPLIT_FIELDS = ("split_by", "reasoning_tokens", "answer_tokens", "reasoning_text", "answer_text", "reasoning_status")
MLXLM_FIELDS = ("finish_reason", "truncated", "stop_token")

# G1 (gate/checks/g1_synthetic.py lines 29, 36-45).
G1_EXCLUDE = ("test_gate_drivers_tiny.py", "test_gate_end_to_end_tiny.py", "test_gate_ftiny.py")
PAIR_FIRST = "test_bench_batch_flip.py"  # runs before test_runner_tiny.py in G1's order; it checks a real tokenizer
TIMEOUT_S = {"isolated": 900.0, "module": 1800.0, "pair": 2700.0, "g1ctx": 3600.0, "fresh": 1800.0}

HOSTS = ("mbp", "mini")
EXPERIMENTS = ("cachestate", "isolated", "module", "pair", "g1ctx", "inproc", "fresh")
PYTEST_ARMS = ("isolated", "module", "pair", "g1ctx")
CONTEXT_ARMS = ("module", "pair", "g1ctx")  # the smallest selection first
REQUIRED = {"mbp": ("cachestate", "isolated", "module", "pair", "g1ctx", "inproc"),
            "mini": ("cachestate", "isolated", "g1ctx", "inproc")}
REGISTERED_N = {("mbp", "cachestate"): 1, ("mini", "cachestate"): 1,
                ("mbp", "isolated"): 50, ("mini", "isolated"): 50,
                ("mbp", "module"): 30, ("mbp", "pair"): 30,
                ("mbp", "g1ctx"): 4, ("mini", "g1ctx"): 1,
                ("mbp", "inproc"): 50, ("mini", "inproc"): 50,
                ("mbp", "fresh"): 100}
PAIRS_UNINTERRUPTED = ("AB", "A_vs_A0", "B_vs_A0")
STUB_KINDS = ("deterministic", "noisy", "resume", "fieldonly")
CS_ONE = 20_000  # (iv) phase A: StubTok objects created one at a time, each dropped before the next
CS_HELD = 1_000_000  # (iv) phase B: StubTok objects created and held


def attempts_cap(n: int) -> int:
    """PLAN.md 3.0: invalid runs are replaced within the same output, up to this many attempts in all."""
    return n + max(3, n // 5)


# Runtime pins (HYPOTHESIS P2): every host for the packages; the mbp also for the OS and the GPU. The mini is told
# apart by its model identifier, so that neither host's data can be filed under the other's label.
PKG_PINS = {"mlx": "0.32.3", "mlx-metal": "0.32.3", "mlx-lm": "0.32.0"}
MBP_OS_PINS = {"macos": "27.0", "kern_osversion": "26A428", "arch_prefix": "applegpu_g17"}
MINI_PINS = {"hw_model": "Mac16,11"}

ENV_LABELS = ("EXP036_PRIVATE", "EXP036_WORK", "EXP037_BUILDS", "EXP037_VENV", "EXP036_MODELS", "EXP036_DATA",
              "EXP036_TOK", "LFA")
ABSENT = "<absent>"
KIND_RANK = {"equal": 0, "field_only": 1, "c_assert": 2, "other": 3, "generation": 4}


class Refused(RuntimeError):
    """A precondition of a valid diagnostic run failed: no output is data."""


# --------------------------------------------------------------------------------------------- small helpers


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_under(p: Path, root: Path) -> bool:
    try:
        Path(p).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def worst(*kinds: str) -> str:
    return max(kinds, key=lambda k: KIND_RANK[k])


def cap_text(s: str | None, limit: int = 65536) -> str | None:
    if s is None or len(s) <= limit:
        return s
    half = limit // 2
    return s[:half] + f"\n[... {len(s) - limit} characters omitted ...]\n" + s[-half:]


def check_out_dir(out: Path) -> Path:
    """--out: this folder's out/, or a directory outside the repository (PLAN.md section 7: never under results/)."""
    out = Path(out).expanduser().resolve()
    if out != OFFICIAL_OUT.resolve() and is_under(out, REPO_DIR):
        raise Refused("--out must be this folder's out/ or a directory outside the repository")
    return out


# ------------------------------------------------------------------------------------------------- redaction


class Redactor:
    """Outputs are committed: paths become labels ($EXP036_WORK, <kit>, <work>, <tmpdir>, ~) and the user and host
    names become <name>. leaks() is checked on the final text before any output is written."""

    def __init__(self, extra: tuple = ()):
        pairs: list[tuple[str, str]] = []

        def add(real: str | None, label: str) -> None:
            if not real:
                return
            for v in {str(real), os.path.realpath(str(real))}:
                v = v.rstrip("/")
                if len(v) > 1:
                    pairs.append((v, label))

        for var in ENV_LABELS:
            if os.environ.get(var):
                add(os.path.expanduser(os.environ[var]), "$" + var)
        for real, label in extra:
            add(real, label)
        add(str(KIT_DIR), "<kit>")
        add(tempfile.gettempdir(), "<tmpdir>")
        self.home = str(Path.home())
        add(self.home, "~")
        self.pairs = sorted(set(pairs), key=lambda x: -len(x[0]))
        names: set[str] = set()
        try:
            names.add(getpass.getuser())
        except Exception:
            pass
        try:
            h = socket.gethostname()
            names.update({h, h.split(".")[0]})
        except Exception:
            pass
        try:
            p = subprocess.run(["scutil", "--get", "LocalHostName"], capture_output=True, text=True, timeout=5)
            if p.returncode == 0:
                names.add(p.stdout.strip())
        except Exception:
            pass
        self.names = sorted({n for n in names if n and len(n) >= 5 and n not in HOSTS}, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        for real, label in self.pairs:
            text = text.replace(real, label)
        for n in self.names:
            text = text.replace(n, "<name>")
        return text

    def leaks(self, text: str) -> list[str]:
        found = [n for n in self.names if n in text]
        if self.home and len(self.home) > 1 and self.home in text:
            found.append("home directory")
        if re.search(r"/Users/[A-Za-z0-9._-]+", text):
            found.append("/Users/<name> path")
        return found


def write_output(obj: dict, path: Path, redactor: Redactor) -> None:
    text = redactor(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False) + "\n")
    leaks = redactor.leaks(text)
    if leaks:
        raise Refused(f"output would carry {leaks}; not written")
    json.loads(text)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial-write")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------- records: transcriptions of runner/jsonl.py and the test's helpers


def scan_jsonl(path: Path) -> tuple[list[dict], int]:
    """runner/jsonl.py scan() (lines 168-186)."""
    records: list[dict] = []
    bad = 0
    with open(path, "rb") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                bad += 1
                continue
            if isinstance(obj, dict):
                records.append(obj)
            else:
                bad += 1
    return records, bad


def first_records(records: list[dict]) -> list[dict]:
    """runner/jsonl.py is_complete() and first_records() (lines 194-210)."""
    seen: set = set()
    out = []
    for r in records:
        if not (r.get("type") == "record" and isinstance(r.get("key"), dict) and "finish_reason" in r):
            continue
        k = (str(r["key"]["item"]), int(r["key"].get("pass", 0)))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def records_by_item(path: Path) -> dict:
    """tests/test_runner_tiny.py records() (line 88), transcribed; the test file is pinned by its sha256."""
    return {r["key"]["item"]: r for r in first_records(scan_jsonl(path)[0])}


def content(recs: dict) -> dict:
    """tests/test_runner_tiny.py content() (line 92), transcribed: every field except TIMING."""
    return {k: {f: v for f, v in r.items() if f not in TIMING} for k, r in recs.items()}


def completed_keys_local(root: Path) -> set:
    """runner/jsonl.py completed_keys("K8", "mmlu_en", "high", root=root) (lines 221-234)."""
    done: set = set()
    for p in sorted(Path(root).glob("*/K8/mmlu_en_high.jsonl")):
        recs, _ = scan_jsonl(p)
        done.update((str(r["key"]["item"]), int(r["key"].get("pass", 0))) for r in first_records(recs))
    return done


def diff_pair(ca: dict, cb: dict) -> dict:
    """The test's comparison content(a) == content(b), itemised: per differing item every differing field with both
    values. kind: equal | field_only (completion_ids equal in every differing item) | generation."""
    items: dict = {}
    for k in sorted(set(ca) | set(cb)):
        if k not in ca or k not in cb:
            items[k] = {"missing_in": "a" if k not in ca else "b", "completion_ids_equal": False}
            continue
        ra, rb = ca[k], cb[k]
        if ra == rb:
            continue
        fields = {}
        for f in sorted(set(ra) | set(rb)):
            if f not in ra or f not in rb or ra[f] != rb[f]:
                fields[f] = {"a": ra.get(f, ABSENT), "b": rb.get(f, ABSENT)}
        ia, ib = ra.get("completion_ids"), rb.get("completion_ids")
        entry: dict = {"fields": fields, "completion_ids_equal": isinstance(ia, list) and ia == ib}
        if isinstance(ia, list) and isinstance(ib, list) and ia != ib:
            j = next((t for t, (x, y) in enumerate(zip(ia, ib)) if x != y), min(len(ia), len(ib)))
            entry["first_divergence"] = {"index": j, "a": ia[j] if j < len(ia) else None,
                                         "b": ib[j] if j < len(ib) else None}
        items[k] = entry
    if not items:
        return {"kind": "equal"}
    kind = "field_only" if all(e["completion_ids_equal"] for e in items.values()) else "generation"
    return {"kind": kind, "items": items}


def pair_fields(pair: dict | None) -> list[str]:
    if not pair:
        return []
    return sorted({f for e in pair.get("items", {}).values() for f in e.get("fields", {})})


OUT_BUDGET = 4_500_000  # bytes; the leak check refuses files over 5 MB outside results/raw, results/pilot, aborted
KEEP_FULL = 10


def _compact_pair(pair: dict | None) -> dict | None:
    if not pair or pair.get("kind") == "equal":
        return pair
    items = {k: {"completion_ids_equal": e.get("completion_ids_equal"), "first_divergence": e.get("first_divergence"),
                 "missing_in": e.get("missing_in"), "fields": sorted(e.get("fields", {}))}
             for k, e in pair.get("items", {}).items()}
    return {"kind": pair["kind"], "items": items, "compacted": True}


def compact_runs(runs: list[dict]) -> list[dict]:
    """Only when an output would exceed OUT_BUDGET: the first KEEP_FULL runs with a difference keep every value;
    later ones keep, per differing item, the field names, completion_ids_equal and the first divergence. The full
    output is always written to the work directory (its sha256 is in the compacted output)."""
    out, kept = [], 0
    for r in runs:
        pairs = [k for k in ("pair",) + PAIRS_UNINTERRUPTED + ("AC", "BC") if isinstance(r.get(k), dict)]
        differs = any(r[k].get("kind") != "equal" for k in pairs)
        if differs and kept < KEEP_FULL:
            kept += 1
            out.append(r)
            continue
        c = dict(r)
        for k in pairs:
            c[k] = _compact_pair(r[k])
        if isinstance(c.get("target"), dict):
            c["target"] = dict(c["target"], text=cap_text(c["target"].get("text"), 4096))
        out.append(c)
    return out


# ---------------------------------------------------------------------- the scenario (tests/test_runner_tiny.py)


def run_uninterrupted(S, d: Path, its: list, tok=None) -> tuple[dict, float]:
    """The test's reference run (lines 203-204): run(model, ref, its, B=1, cap=20). tok: (iv) only, the tokenizer
    object the run receives (None: the test's own StubTok())."""
    ref = d / "ref.jsonl"
    t0 = time.perf_counter()
    S.run(ref, its, tok=tok)
    return S.records(ref), round(time.perf_counter() - t0, 3)


def run_stop_resume(S, d: Path, its: list, stop_tok=None, resume_tok=None) -> tuple[dict, dict, float]:
    """The test's lines 206-223, step by step. Each assertion is recorded (by its line) instead of raised; the
    line-222 comparison is the caller's. stop_tok / resume_tok: (iv) only."""
    p = d / CELL_FILE
    t0 = time.perf_counter()

    def stop_after_three():
        recs = [r for r in S.read_jsonl(p) if r.get("type") == "record"] if p.exists() else []
        return len(recs) >= 3

    summ = S.run(p, its, stop=stop_after_three, tok=stop_tok)
    l213 = bool(summ["stopped"] and summ["n_done"] == 3)
    with open(p, "ab") as f:  # killed while writing the 4th line
        f.write(TORN)
    done = S.completed_keys(d / "raw")
    l217 = len(done) == 3
    summ = S.run(p, its, done=done, tok=resume_tok)
    l219 = summ["n_todo"] == 3 and summ["n_done"] == 3
    recs, bad = S.scan(p)
    try:  # the test indexes r["type"]: a parsed object without "type" makes it error, so the check is false
        l221 = bad == 1 and [r["type"] for r in recs if r["type"] != "record"] == ["header", "resume"]
    except (KeyError, TypeError):
        l221 = False
    out = S.records(p)
    l223 = S.completed_keys(d / "raw") == {(i["id"], 0) for i in its}
    checks = {"l213": l213, "l217": l217, "l219": l219, "l221": l221, "l223": l223}
    return out, checks, round(time.perf_counter() - t0, 3)


def inproc_iterations(S, n: int, work: Path, progress=None) -> tuple[list[dict], dict]:
    """(iii): per iteration, in this order, A (uninterrupted), C (the test's stop / torn tail / resume, compared with
    A as the test compares it with ref), B (uninterrupted). Uninterrupted runs are compared with each other and
    with A of iteration 0. checked_len (descriptive): the size of scorers.reasoning._CHECKED after the iteration."""
    runs: list[dict] = []
    a0 = None
    for i in range(n):
        d = work / f"iter_{i:03d}"
        its = S.items()
        rec_a, w_a = run_uninterrupted(S, d / "A", its)
        rec_c, checks, w_c = run_stop_resume(S, d / "C", its)
        rec_b, w_b = run_uninterrupted(S, d / "B", its)
        ca, cb, cc = S.content(rec_a), S.content(rec_b), S.content(rec_c)
        if a0 is None:
            a0 = ca
        run = {"i": i, "order": ["A", "C", "B"],
               "AB": diff_pair(ca, cb), "A_vs_A0": diff_pair(a0, ca), "B_vs_A0": diff_pair(a0, cb),
               "AC": diff_pair(ca, cc), "BC": diff_pair(cb, cc),
               "C_checks": checks, "C_checks_ok": all(checks.values()),
               "c_assert": [k for k, v in checks.items() if not v],
               "wall_s": {"A": w_a, "C": w_c, "B": w_b}, "checked_len": S.checked_len()}
        run["uninterrupted_kind"] = worst(*(run[p]["kind"] for p in PAIRS_UNINTERRUPTED))
        run["ac_kind"] = run["AC"]["kind"]
        runs.append(run)
        if progress is not None:
            progress(runs)
    return runs, a0 or {}


def summarise_inproc(runs: list[dict]) -> dict:
    ok = [r for r in runs if r.get("valid", True)]
    first = None
    for r in ok:
        for name in PAIRS_UNINTERRUPTED + ("AC",):
            if r[name]["kind"] != "equal":
                first = {"i": r["i"], "pair": name, "kind": r[name]["kind"], "items": sorted(r[name]["items"])}
                break
        if first:
            break
    return {"iterations": len(ok), "invalid": len(runs) - len(ok),
            "uninterrupted": dict(Counter(r["uninterrupted_kind"] for r in ok)),
            "ac": dict(Counter(r["ac_kind"] for r in ok)),
            "bc": dict(Counter(r["BC"]["kind"] for r in ok)),
            "c_checks_failed": sum(1 for r in ok if r.get("c_assert")),
            "checked_len_max": max((r.get("checked_len") or 0 for r in ok), default=None),
            "first_difference": first}


class StubWorld:
    """Selftest stand-in for scorers.reasoning's check cache, runner.generate._scorer_split, the real tokenizer and
    the test's StubTok, with the same object layouts (plain classes) and the same id()-keyed cache."""

    family = "kolibri"

    class _Reasoning:
        def __init__(self):
            self._CHECKED: set = set()

        def check_tokenizer(self, family, tok):
            key = (family, id(tok))
            if key in self._CHECKED:
                return
            if getattr(tok, "real", False) is not True:
                raise ValueError("tokenizer does not match the family table")
            self._CHECKED.add(key)

    class StubTok:
        def decode(self, ids, skip_special_tokens=False):
            return "".join(f"<{int(i)}>" for i in ids)

    class RealTok:
        real = True

        def decode(self, ids, skip_special_tokens=False):
            return "".join(f"[{int(i)}]" for i in ids)

    def __init__(self):
        self.reasoning = self._Reasoning()

    def scorer_split(self, family, tok):
        try:
            self.reasoning.check_tokenizer(family, tok)
        except Exception:
            return None
        return lambda *a: None

    def load_tok(self):
        return self.RealTok()

    def exercise(self, tok):
        self.reasoning.check_tokenizer(self.family, tok)
        return {"checked": True}

    def after_free(self):
        return None


class StubScenario:
    """Selftest stand-in for the port + run_cell: the test's items, JSONL layout, record fields and summaries, in
    pure Python. kind: deterministic | noisy (injected non-determinism: a generation call changes one token with
    probability 0.25, seeded) | resume (only the first item of a resumed run changes) | fieldonly (a resumed run
    writes a different split_by; completion_ids unchanged). With a StubWorld, split_by follows the world's
    scorer_split for the run's tokenizer, as run_cell's does."""

    def __init__(self, kind: str, seed: int = 37, world: StubWorld | None = None):
        if kind not in STUB_KINDS:
            raise ValueError(kind)
        self.kind = kind
        self.rng = random.Random(seed)
        self.calls = 0
        self.world = world

    def items(self) -> list[dict]:
        out = []
        for k in range(6):
            r = random.Random(1000 + k)
            out.append({"id": f"q{k:03d}", "item_sha256": f"{k:064x}", "prompt_sha256": f"{k + 100:064x}",
                        "prompt_ids": [r.randrange(1008) for _ in range(12 + 9 * k)], "prompt_text": "user turn",
                        "max_tokens": 8 + 2 * k})
        return out

    def _completion(self, it: dict, resumed: bool, first: bool) -> list[int]:
        n = min(int(it["max_tokens"]), 20)
        h = hashlib.sha256(json.dumps(it["prompt_ids"]).encode()).digest()
        ids = [(h[i % 32] * 7 + 13 * i) % 1000 for i in range(n)]
        if self.kind == "noisy" and self.rng.random() < 0.25:
            j = self.rng.randrange(n)
            ids[j] = (ids[j] + 1) % 1000
        if self.kind == "resume" and resumed and first:
            ids[-1] = (ids[-1] + 1) % 1000
        return ids

    def _record(self, it: dict, ids: list[int], split_by: str) -> dict:
        self.calls += 1
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        text = "".join(f"<{i}>" for i in ids)
        return {"type": "record", "key": {"arm": "K8", "task": "mmlu_en", "effort": "high", "item": it["id"],
                                          "pass": 0},
                "item_sha256": it["item_sha256"], "prompt_sha256": it["prompt_sha256"],
                "rendered_sha256": sha256_bytes(it["prompt_text"].encode()),
                "rendered_prompt_tokens": len(it["prompt_ids"]), "prompt_prefill": "none",
                "completion_ids": ids, "completion_tokens": len(ids), "reasoning_tokens": 0,
                "answer_tokens": len(ids), "text": text, "reasoning_text": "", "answer_text": text,
                "finish_reason": "length", "truncated": True, "reasoning_status": "none", "split_by": split_by,
                "stop_token": None, "t_submit": now, "t_first_token": now, "t_done": now,
                "wall_s": round(random.random(), 3), "batch_id": self.calls, "batch_size": 1, "cell_seed": 36,
                "max_tokens": min(int(it["max_tokens"]), 20),
                "sampler": {"order": "vllm", "temperature": 0.0, "top_p": 1.0, "top_k": 0}, "eos_ids": [1022, 1021]}

    def run(self, path: Path, its: list, stop=None, done=None, tok=None) -> dict:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = path.exists() and path.stat().st_size > 0
        if existing and not path.read_bytes().endswith(b"\n"):
            with open(path, "ab") as f:  # runner/jsonl.py open_run: a torn tail gets its newline
                f.write(b"\n")
        done = set(done or ())
        todo = [it for it in its if (it["id"], 0) not in done]
        summary = {"stopped": False, "n_done": 0, "n_todo": len(todo)}
        resumed = bool(done)
        if self.world is not None:
            t = tok if tok is not None else self.world.StubTok()
            world_split = ("scorers.reasoning.split_ids" if self.world.scorer_split(self.world.family, t) is not None
                           else "runner.generate.split_ids")
        with open(path, "ab") as f:
            f.write((json.dumps({"type": "resume" if existing else "header", "cell": "t", "B": 1},
                                sort_keys=True) + "\n").encode())
            f.flush()
            for j, it in enumerate(todo):
                if stop is not None and stop():
                    summary["stopped"] = True
                    break
                ids = self._completion(it, resumed, j == 0)
                if self.world is not None:
                    split_by = world_split
                else:
                    split_by = "stub.resumed" if (self.kind == "fieldonly" and resumed) else "stub.split"
                f.write((json.dumps(self._record(it, ids, split_by), sort_keys=True) + "\n").encode())
                f.flush()
                summary["n_done"] += 1
        return summary

    def records(self, path: Path) -> dict:
        return records_by_item(path)

    def content(self, recs: dict) -> dict:
        return content(recs)

    def read_jsonl(self, path: Path) -> list[dict]:
        return scan_jsonl(path)[0]

    def scan(self, path: Path):
        return scan_jsonl(path)

    def completed_keys(self, root: Path) -> set:
        return completed_keys_local(root)

    def checked_len(self):
        return len(self.world.reasoning._CHECKED) if self.world is not None else None


class RealScenario:
    """The test's own helpers (imported from tests/test_runner_tiny.py, unchanged) on the fixture's model."""

    def __init__(self, T, model, stub_cls=None):
        self.T, self.model = T, model
        self.stub_cls = stub_cls or T.StubTok

    def items(self) -> list[dict]:
        return self.T.items(6, max_tokens=lambda k: 8 + 2 * k)  # line 202

    def run(self, path, its, tok=None, **kw):
        if tok is None:
            return self.T.run(self.model, path, its, B=1, cap=20, **kw)  # lines 204, 212, 218
        # (iv) only: the test's run() calls StubTok() (line 79); it receives the given object instead.
        self.T.StubTok = lambda: tok
        try:
            return self.T.run(self.model, path, its, B=1, cap=20, **kw)
        finally:
            self.T.StubTok = self.stub_cls

    def records(self, path):
        return self.T.records(path)

    def content(self, recs):
        return self.T.content(recs)

    def read_jsonl(self, path):
        return self.T.jsonl.read_jsonl(path)

    def scan(self, path):
        return self.T.jsonl.scan(path)

    def completed_keys(self, root):
        return self.T.jsonl.completed_keys("K8", "mmlu_en", "high", root=root)

    def checked_len(self):
        m = sys.modules.get("scorers.reasoning")
        c = getattr(m, "_CHECKED", None) if m is not None else None
        return len(c) if c is not None else None


def kit_sys_path() -> None:
    """tests/conftest.py lines 27-45: no bytecode, MLX_ENABLE_TF32=0 before MLX, the kit and tests/ on sys.path
    (tests/ first), the Hub offline."""
    sys.dont_write_bytecode = True
    os.environ.setdefault("MLX_ENABLE_TF32", "0")
    for p in (str(KIT_DIR), str(TESTS_DIR)):
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def load_test_module():
    """The test's setting, in conftest's order: the TF32 probe (pytest_sessionstart), conftest's imports and the
    test module, imported unchanged."""
    kit_sys_path()
    from tools.precision import ensure_exact_fp32

    precision = ensure_exact_fp32()
    import tiny_checkpoint  # noqa: F401  (conftest imports both at collection)
    import tiny_real_layout  # noqa: F401

    spec = importlib.util.spec_from_file_location("exp037_diag_test_runner_tiny", KIT_DIR / TEST_REL)
    T = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(T)
    return T, precision


def load_model(work: Path):
    """The session fixture tiny_vendor_dir (conftest _write: seed 0, preset vendor) and the module fixture model
    (port_harness.load_port(tiny_vendor_dir, float32=True))."""
    import tiny_checkpoint

    ckpt = tiny_checkpoint.write_tiny_checkpoint(work / "tiny_vendor0", seed=0, preset="vendor",
                                                 copy_port=tiny_checkpoint.PORT_MODEL_FILE.exists())
    from exp036_helpers import import_sibling

    harness = import_sibling("port_harness")
    return harness.load_port(ckpt, float32=True)


def load_real(work: Path):
    T, precision = load_test_module()
    return T, load_model(work), precision


# ------------------------------------------------------------- (iv) the tokenizer-check cache (PLAN.md 3.6)


class RealProbe:
    """(iv) on the kit: scorers.reasoning's cache, runner.generate._scorer_split, the real Kolibri tokenizer as
    tests/test_bench_batch_flip.py loads it, and the test's StubTok. Imports happen in load_test_module's setting."""

    def __init__(self, T, work: Path):
        from bench import batch_flip
        from runner import generate
        from scorers import reasoning
        import test_bench_support as tbs

        self.T, self.work, self.model = T, work, None
        self.reasoning, self.batch_flip, self.tbs = reasoning, batch_flip, tbs
        self.scorer_split = generate._scorer_split
        self.StubTok = T.StubTok
        self.family = reasoning.family_spec("kolibri").name

    def load_tok(self):
        """tests/test_bench_batch_flip.py _kolibri_tokenizer() (lines 215-218)."""
        from mlx_lm.tokenizer_utils import load as load_tok

        try:
            d = self.tbs.kolibri_tok_dir()
        except BaseException as e:  # a pytest skip outside pytest
            if isinstance(e, KeyboardInterrupt):
                raise
            raise Refused(f"the Kolibri tokenizer directory is not available ({type(e).__name__})")
        return load_tok(d)

    def exercise(self, tok) -> dict:
        """What tests/test_bench_batch_flip.py lines 241-244 do with the tokenizer: bench.batch_flip.default_extract
        and two extractions, each through scorers.reasoning.split_reasoning (which checks the tokenizer and caches
        the check by id). The item preparation (default_prepare, chat.render) does not touch the cache."""
        extract = self.batch_flip.default_extract(tok)
        prompt = tok.encode("Synthetic question 0? (A) x (B) y", add_special_tokens=False)
        a = extract(prompt, tok.encode("Short reasoning. The answer is (B).<|im_end|>", add_special_tokens=False))
        b = extract(prompt, tok.encode("No letter here.<|im_end|>", add_special_tokens=False))
        return {"extract": a, "extract_no_letter": b}

    def after_free(self):
        """G1 loads the module fixture's model after the batch-flip test; so does the probe."""
        self.model = load_model(self.work)


def _clean_stub(P, C) -> object:
    for _ in range(10_000):
        s = P.StubTok()
        if (P.family, id(s)) not in C:
            return s
    raise RuntimeError("no StubTok outside the check cache in 10,000 tries")


def scenario_pair(S, d: Path, its: list, ref_tok, stop_tok, resume_tok) -> dict:
    """The failing test's scenario with given tokenizer objects: the reference run (ref_tok), the stopped run
    (stop_tok), the torn tail, the resumed run (resume_tok); the line-222 comparison, itemised."""
    rec_ref, _ = run_uninterrupted(S, d / "ref", its, tok=ref_tok)
    rec_c, checks, _ = run_stop_resume(S, d / "C", its, stop_tok=stop_tok, resume_tok=resume_tok)
    return {"pair": diff_pair(S.content(rec_ref), S.content(rec_c)), "C_checks": checks,
            "split_by": {"ref": sorted({str(r.get("split_by")) for r in rec_ref.values()}),
                         "resumed_file": sorted({str(r.get("split_by")) for r in rec_c.values()})}}


def cachestate_outcome(r: dict) -> str:
    """hit | stale_only | none (PLAN.md 3.6)."""
    seeded = r.get("seeded") or {}
    sp = seeded.get("pair") or {}
    seeded_ok = bool(seeded.get("scorer_split")) and sp.get("kind") == "field_only" and "split_by" in pair_fields(sp)
    control_ok = ((r.get("control") or {}).get("pair") or {}).get("kind") == "equal"
    if not (r.get("stale_keys_added") and r.get("tokenizer_freed") and seeded_ok and control_ok):
        return "none"
    h = r.get("hit") or {}
    hp = h.get("pair") or {}
    if h.get("found") and h.get("scorer_split") and hp.get("kind") == "field_only" and "split_by" in pair_fields(hp):
        return "hit"
    return "stale_only"


def cachestate_probe(P, make_scenario, work: Path) -> dict:
    """(iv), PLAN.md 3.6. 1: the batch-flip test's calls with a real tokenizer, which is then freed; 2: StubTok
    objects one at a time (A), then held (B), each checked against the cache; 3: the failing test's scenario with a
    StubTok that landed on a stale id as the resumed run's tokenizer; 4 (always): the same with a clean StubTok seeded
    into the cache (symbolic); a control with clean StubToks only."""
    fam = P.family
    C = P.reasoning._CHECKED
    res: dict = {"family": fam, "checked_before": len(C), "m_one": CS_ONE, "m_held": CS_HELD}
    before = set(C)
    tok = P.load_tok()
    tcls, scls = type(tok), P.StubTok
    res["layout"] = {
        "tokenizer_class": f"{tcls.__module__}.{tcls.__qualname__}", "stub_class": scls.__qualname__,
        "tokenizer_basicsize": tcls.__basicsize__, "stub_basicsize": scls.__basicsize__,
        "tokenizer_dictoffset": tcls.__dictoffset__, "stub_dictoffset": scls.__dictoffset__,
        "tokenizer_slots": "__slots__" in tcls.__dict__, "stub_slots": "__slots__" in scls.__dict__}
    try:
        wr = weakref.ref(tok)
    except TypeError:
        wr = None
    res["exercise"] = P.exercise(tok)
    stale = set(C) - before
    res["stale_keys_added"] = len(stale)
    res["stale_key_is_tokenizer"] = (fam, id(tok)) in stale
    del tok
    gc.collect()
    res["tokenizer_freed"] = wr is not None and wr() is None
    P.after_free()
    S = make_scenario()
    hit_obj, hit = None, {"found": False}
    for j in range(CS_ONE):  # phase A
        s = P.StubTok()
        if (fam, id(s)) in C:
            hit_obj, hit = s, {"found": True, "phase": "one_at_a_time", "index": j}
            break
        del s
    res["held_allocated"] = 0
    if hit_obj is None:  # phase B
        hold = []
        for j in range(CS_HELD):
            s = P.StubTok()
            if (fam, id(s)) in C:
                hit_obj, hit = s, {"found": True, "phase": "held", "index": j}
                break
            hold.append(s)
        res["held_allocated"] = len(hold)
        s = None
        del hold
        gc.collect()
    res["stale_keys_now"] = len(set(C) & stale)
    its = S.items()
    c1, c2, c3 = _clean_stub(P, C), _clean_stub(P, C), _clean_stub(P, C)
    while len({id(c1), id(c2), id(c3)}) < 3:  # held together, so distinct; guard anyway
        c3 = _clean_stub(P, C)
    res["control"] = scenario_pair(S, work / "control", its, c1, c2, c3)
    if hit_obj is not None:
        hit["scorer_split"] = P.scorer_split(fam, hit_obj) is not None
        hit.update(scenario_pair(S, work / "hit", its, c1, c2, hit_obj))
    res["hit"] = hit
    key = (fam, id(c3))
    C.add(key)
    try:
        seeded: dict = {"scorer_split": P.scorer_split(fam, c3) is not None}
        seeded.update(scenario_pair(S, work / "seeded", its, c1, c2, c3))
    finally:
        C.discard(key)
    res["seeded"] = seeded
    res["outcome"] = cachestate_outcome(res)
    return res


# --------------------------------------------------------------------------------- margins (descriptive only)


def teacher_forced_margins(model, prompt_ids: list[int], completion_ids: list[int]) -> dict:
    """Per generated step: top-1 and top-2 ids and the top-1 minus top-2 log-prob margin, from one teacher-forced
    pass (runner.generate.teacher_force_logprobs, unchanged) over prompt + completion. A proxy for the decode
    path's values (one forward pass, not incremental decode): it shows near-ties, not the exact decode numbers."""
    import numpy as np

    from runner.generate import teacher_force_logprobs

    ids = [int(t) for t in list(prompt_ids) + list(completion_ids)]
    lp = teacher_force_logprobs(model, ids)
    P = len(prompt_ids)
    steps = []
    for s, tok in enumerate(completion_ids):
        row = lp[P - 1 + s]
        order = np.argsort(-row, kind="stable")
        t1, t2 = int(order[0]), int(order[1])
        steps.append({"step": s, "token": int(tok), "top1": t1, "top2": t2, "margin": float(row[t1] - row[t2]),
                      "token_is_top1": int(tok) == t1, "token_minus_top1": float(row[int(tok)] - row[t1])})
    m = min(steps, key=lambda x: x["margin"]) if steps else None
    return {"steps": steps, "min_margin": m["margin"] if m else None, "min_margin_step": m["step"] if m else None}


def first_generation_difference(runs: list[dict]):
    for r in runs:
        for name in PAIRS_UNINTERRUPTED + ("AC",):
            for item, e in sorted(r[name].get("items", {}).items()):
                f = e.get("fields", {}).get("completion_ids")
                if f and isinstance(f["a"], list) and isinstance(f["b"], list):
                    return r["i"], name, item, f["a"], f["b"]
    return None


def inproc_margins(S, model, runs: list[dict], a0: dict) -> dict:
    its = {it["id"]: it for it in S.items()}
    out: dict = {"method": "teacher-forced: runner.generate.teacher_force_logprobs over prompt + completion; "
                           "a proxy for the decode path (PLAN.md section 3.7)",
                 "a0_path": {}, "first_difference": None}
    for item in ("q003", "q004"):  # the items the gate's visible output named
        if item in a0:
            out["a0_path"][item] = teacher_forced_margins(model, its[item]["prompt_ids"], a0[item]["completion_ids"])
    first = first_generation_difference(runs)
    if first:
        i, name, item, ia, ib = first
        out["first_difference"] = {"i": i, "pair": name, "item": item,
                                   "a": teacher_forced_margins(model, its[item]["prompt_ids"], ia),
                                   "b": teacher_forced_margins(model, its[item]["prompt_ids"], ib)}
    return out


# --------------------------------------------------------------------------------------- pytest experiments


def pytest_isolated_cmd(py: str, junit: Path, basetemp: Path, test_file: Path, test_name: str) -> list[str]:
    """(i): the single test, -vv (untruncated assertion diff), G1's -rs and -p no:cacheprovider, a junit per run,
    and --basetemp so that the test's two JSONL files can be read afterwards."""
    return [py, "-m", "pytest", "-vv", "-rs", "-p", "no:cacheprovider", f"--junitxml={junit}",
            f"--basetemp={basetemp}", f"{test_file}::{test_name}"]


def pytest_selection_cmd(py: str, junit: Path, basetemp: Path, targets: list) -> list[str]:
    """(i') and (i''): G1's flags (-q -rs -p no:cacheprovider) with the two reporting additions of (ii), on the
    given files in the given order."""
    return [py, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", f"--junitxml={junit}",
            "-o", "verbosity_assertions=2", f"--basetemp={basetemp}"] + [str(t) for t in targets]


def pytest_g1ctx_cmd(py: str, junit: Path, basetemp: Path, tests_dir: Path, exclude=G1_EXCLUDE) -> list[str]:
    """(ii): gate/checks/g1_synthetic.run's command (lines 36-38) with two additions after --junitxml:
    -o verbosity_assertions=2 (the -vv assertion diff only; -q reporting kept) and --basetemp."""
    cmd = [py, "-m", "pytest", "-q", "-rs", "-p", "no:cacheprovider", f"--junitxml={junit}",
           "-o", "verbosity_assertions=2", f"--basetemp={basetemp}"]
    cmd += [f"--ignore={tests_dir / e}" for e in exclude] + [str(tests_dir)]
    return cmd


def g1_env(base) -> dict:
    """gate/checks/g1_synthetic.run's environment (lines 39-45, require_all=True)."""
    env = dict(base, EXP036_GATE_G1="1", PYTHONDONTWRITEBYTECODE="1")
    env["EXP036_REQUIRE_ALL"] = "run-host"
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    return env


def parse_junit(path: Path, test_name: str = TEST_NAME) -> dict:
    """Counts, the target test and every other failing test. The target's outcome: passed (the call passed; a
    teardown error is flagged) | failed (a <failure>: the call phase) | setup_error (an <error> that is not a
    teardown error) | skipped | missing."""
    root = ET.parse(path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    counts = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    for s in suites:
        for k in counts:
            counts[k] += int(s.get(k, 0))
    counts["skipped_build_host_only"] = sum(
        1 for sk in root.iter("skipped") if "build-host only:" in (sk.get("message") or "") + (sk.text or ""))
    target: dict = {"outcome": "missing"}
    others = []
    for tc in root.iter("testcase"):
        name, cls = tc.get("name") or "", tc.get("classname") or ""
        fail, errs, skip = tc.find("failure"), tc.findall("error"), tc.find("skipped")
        teardown = [e for e in errs if "teardown" in (e.get("message") or "")]
        setup = [e for e in errs if e not in teardown]
        if fail is not None:
            res, node = "failed", fail
        elif setup:
            res, node = "setup_error", setup[0]
        elif skip is not None:
            res, node = "skipped", skip
        else:
            res, node = "passed", (teardown[0] if teardown else None)
        if name == test_name and cls.endswith("test_runner_tiny"):
            text = node.text if node is not None else None
            lines = re.findall(r"test_runner_tiny\.py:(\d+)", text or "")
            target = {"outcome": res, "time_s": float(tc.get("time") or 0.0),
                      "line": int(lines[-1]) if lines and res == "failed" else None,
                      "teardown_error": bool(teardown),
                      "message": cap_text(node.get("message") if node is not None else None, 4096),
                      "text": cap_text(text)}
        elif res in ("failed", "setup_error") or teardown:
            others.append({"test": f"{cls}::{name}", "outcome": res if not (res == "passed" and teardown)
                           else "teardown_error",
                           "message": cap_text((node.get("message") or "") if node is not None else "", 500)})
    return {"counts": counts, "target": target, "others": others}


def find_record_files(basetemp: Path) -> dict:
    # pytest adds a "<name>current" symlink to the newest numbered directory: candidates are deduplicated by target.
    cands = (sorted({p.resolve() for p in Path(basetemp).glob(TMP_PREFIX + "*") if p.is_dir()})
             if Path(basetemp).is_dir() else [])
    good = [c for c in cands if (c / "ref.jsonl").is_file() and (c / CELL_FILE).is_file()]
    if len(good) == 1:
        return {"ref": good[0] / "ref.jsonl", "resumed": good[0] / CELL_FILE, "n_candidates": len(cands)}
    return {"ref": None, "resumed": None, "n_candidates": len(cands)}


def analyse_pytest_run(k: int, junit: Path, basetemp: Path, rc, wall: float, outtxt: str,
                       test_name: str = TEST_NAME, target_line: int = TEST_LINE,
                       c_lines=C_LINES) -> tuple[dict, dict]:
    """kind (PLAN.md section 4): equal | field_only | generation | c_assert | other, or invalid (not data: the
    target missing, a setup error, a skip, or files that contradict a pass)."""
    j = parse_junit(junit, test_name) if Path(junit).is_file() else None
    files = find_record_files(basetemp)
    pair = None
    if files["ref"] and files["resumed"]:
        pair = diff_pair(content(records_by_item(files["ref"])), content(records_by_item(files["resumed"])))
    t = j["target"] if j else {"outcome": "missing"}
    o = t["outcome"]
    reason = None
    if o in ("missing", "setup_error", "skipped"):
        kind, reason = "invalid", o
    elif o == "passed":
        kind = "equal" if pair is None or pair["kind"] == "equal" else "invalid"
        reason = None if kind == "equal" else "the files differ but the test passed"
    elif t.get("line") == target_line and pair is not None and pair["kind"] != "equal":
        kind = pair["kind"]
    elif t.get("line") in c_lines:
        kind = "c_assert"
    else:
        kind = "other"
    tail = [ln for ln in outtxt.splitlines() if ln.strip()][-1:] or [""]
    run = {"k": k, "valid": kind != "invalid", "kind": kind, "invalid_reason": reason, "returncode": rc,
           "wall_s": round(wall, 1), "target": j["target"] if j else None, "counts": j["counts"] if j else None,
           "other_failures": j["others"] if j else [], "record_files_found": pair is not None,
           "record_dir_candidates": files["n_candidates"], "pair": pair,
           "junit_sha256": sha256_file(junit) if Path(junit).is_file() else None,
           "log_sha256": sha256_bytes(outtxt.encode("utf-8", "replace")), "summary_line": tail[0][:300]}
    return run, files


def summarise_pytest(runs: list[dict]) -> dict:
    valid = [r for r in runs if r["valid"]]
    kinds = Counter(r["kind"] for r in valid)
    lines = Counter(str((r.get("target") or {}).get("line")) for r in valid if r["kind"] != "equal")
    first = next((r["k"] for r in valid if r["kind"] != "equal"), None)
    others = Counter(o["test"] for r in valid for o in r.get("other_failures", []))
    return {"n_valid": len(valid), "n_invalid": len(runs) - len(valid), "kinds": dict(kinds),
            "invalid_reasons": dict(Counter(r.get("invalid_reason") for r in runs if not r["valid"])),
            "reproductions": sum(v for k, v in kinds.items() if k != "equal"), "failing_lines": dict(lines),
            "first_reproduction": first, "other_failing_tests": dict(others)}


def run_pytest_series(kind: str, n: int, run_dir: Path, bt_root: Path, cwd: Path, env: dict, make_cmd,
                      timeout: float, progress, test_name: str = TEST_NAME, target_line: int = TEST_LINE,
                      c_lines=C_LINES) -> tuple[list[dict], bool]:
    """Runs until n valid runs exist or attempts_cap(n) attempts were made (PLAN.md 3.0). Each run's two JSONL
    files are copied into the work directory. Returns (runs, reached)."""
    runs: list[dict] = []
    cap = attempts_cap(n)
    k = 0
    while sum(1 for r in runs if r["valid"]) < n and k < cap:
        junit = run_dir / f"junit_{k:03d}.xml"
        log = run_dir / f"pytest_{k:03d}.log"
        basetemp = bt_root / f"{kind}_{k:03d}"
        if basetemp.exists():
            raise Refused(f"{basetemp} exists (pytest removes an existing --basetemp)")
        cmd = make_cmd(junit, basetemp)
        t0 = time.time()
        try:
            p = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True, timeout=timeout)
            rc, outtxt = p.returncode, (p.stdout or "") + (p.stderr or "")
        except subprocess.TimeoutExpired as e:
            partial = e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
            rc, outtxt = None, f"timeout after {timeout} s\n{partial}"
        wall = time.time() - t0
        log.write_text(outtxt if isinstance(outtxt, str) else str(outtxt), encoding="utf-8")
        run, files = analyse_pytest_run(k, junit, basetemp, rc, wall, outtxt, test_name, target_line, c_lines)
        if files["ref"] and files["resumed"]:
            keep = run_dir / f"records_{k:03d}"
            keep.mkdir(parents=True, exist_ok=False)
            shutil.copy2(files["ref"], keep / "ref.jsonl")
            shutil.copy2(files["resumed"], keep / "resumed.jsonl")
            run["records_copy"] = str(keep)
        runs.append(run)
        progress(runs)
        k += 1
    return runs, sum(1 for r in runs if r["valid"]) >= n


# -------------------------------------------------------------------------------------------- preconditions


def _git(*args: str) -> str:
    p = subprocess.run(["git", "-C", str(KIT_DIR), *args], capture_output=True, text=True, timeout=60)
    if p.returncode != 0:
        raise Refused(f"git {' '.join(args)}: {p.stderr.strip()[:300]}")
    return p.stdout


def _allowed(rel: str) -> bool:
    return rel in ALLOWED_FILES or rel.startswith(ALLOWED_PREFIXES)


def package_identity() -> dict:
    """The package files' sha256, the last commit that touched them, whether they are clean and whether that commit
    is in the upstream branch (pushed; read from the local ref, no network)."""
    rels = [f"{PACKAGE}/{n}" for n in PACKAGE_FILES]
    ident: dict = {"sha256": {n: sha256_file(PKG_DIR / n) for n in PACKAGE_FILES if (PKG_DIR / n).is_file()}}
    try:
        ident["commit"] = _git("log", "-1", "--format=%H", "--", *rels).strip() or None
        ident["clean"] = not _git("status", "--porcelain=v1", "--untracked-files=all", "--", *rels).strip()
        up = subprocess.run(["git", "-C", str(KIT_DIR), "merge-base", "--is-ancestor", ident["commit"] or "HEAD",
                             "@{upstream}"], capture_output=True, timeout=60)
        ident["pushed"] = bool(ident["commit"]) and up.returncode == 0
    except Refused as e:
        ident.update(commit=None, clean=False, pushed=False, error=str(e))
    return ident


def package_problems(outputs: list[dict], ident: dict | None = None) -> list[str]:
    """PLAN.md 3.0: every output was written by one package version, the committed one."""
    problems = []
    shas = {json.dumps(o.get("package_sha256"), sort_keys=True) for o in outputs}
    if len(shas) > 1:
        problems.append("the outputs were written by different package versions")
    if ident is not None:
        if not ident.get("clean"):
            problems.append("the package files have uncommitted changes")
        for o in outputs:
            if o.get("package_sha256") != ident.get("sha256"):
                problems.append(f"{o.get('_file')}: package_sha256 differs from the committed package")
    return problems


def official_guard(existing: list[dict], host: str, exp: str, approved: str | None) -> list[str]:
    """PLAN.md 3.0: one complete official output per host and experiment; after a crash one unchanged repeat; a
    further repeat only on Andrei's recorded decision."""
    mine = [o for o in existing if o.get("host") == host and o.get("experiment") == exp and not o.get("stub")]
    done = [o for o in mine if o.get("complete")]
    if done:
        return [f"out/ already holds a complete {host} {exp} output ({done[0].get('_file')})"]
    if len(mine) >= 2 and not approved:
        return [f"out/ holds {len(mine)} incomplete {host} {exp} outputs: a further repeat needs Andrei's decision "
                f"(--repeat-approved '<UTC> <label>')"]
    return []


def _hash_tree(*args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(KIT_DIR / "tools" / "hash_tree.py"), *args], cwd=str(KIT_DIR),
                          env=env, capture_output=True, text=True, timeout=600)


def kit_binding() -> tuple[dict, list[str]]:
    problems: list[str] = []
    info: dict = {"gate_code_commit": GATE_CODE_COMMIT, "written_against": WRITTEN_AGAINST,
                  "choice": {"utc": CHOICE_UTC, "label": CHOICE_LABEL, "commit": CHOICE_COMMIT or None}}
    rec = KIT_DIR / GATE_RECORD
    if not rec.is_file() or sha256_file(rec) != GATE_RECORD_SHA256:
        problems.append(f"{GATE_RECORD} missing or not sha256 {GATE_RECORD_SHA256[:12]}...")
    for rel, want in PINNED_SHA256.items():
        got = sha256_file(KIT_DIR / rel) if (KIT_DIR / rel).is_file() else None
        if got != want:
            problems.append(f"{rel} sha256 {got} != pinned {want}")
    try:
        info["head"] = _git("rev-parse", "HEAD").strip()
        prefix = _git("rev-parse", "--show-prefix").strip()
        anc = subprocess.run(["git", "-C", str(KIT_DIR), "merge-base", "--is-ancestor", GATE_CODE_COMMIT, "HEAD"],
                             capture_output=True, timeout=60)
        if anc.returncode != 0:
            problems.append(f"{GATE_CODE_COMMIT[:12]} is not an ancestor of HEAD")
        changed = [x for x in _git("diff", "--relative", "--name-only", GATE_CODE_COMMIT, "HEAD", "--").splitlines()
                   if x]
        info["changed_since_gate_code_commit"] = changed
        bad = [x for x in changed if not _allowed(x)]
        if bad:
            problems.append(f"kit files changed since {GATE_CODE_COMMIT[:12]}: {bad[:10]}")
        status = []
        for line in _git("status", "--porcelain=v1", "--untracked-files=all", "--", ".").splitlines():
            for part in line[3:].split(" -> "):
                part = part.strip().strip('"')
                status.append(part[len(prefix):] if part.startswith(prefix) else part)
        info["uncommitted"] = status
        bad = [x for x in status if not _allowed(x)]
        if bad:
            problems.append(f"uncommitted kit files outside the allowed paths: {bad[:10]}")
        # Andrei's choice is recorded in HYPOTHESIS.md at CHOICE_COMMIT (PLAN.md header).
        if not CHOICE_COMMIT:
            problems.append("CHOICE_COMMIT is not set: the package is not committed against the choice entry")
        else:
            a2 = subprocess.run(["git", "-C", str(KIT_DIR), "merge-base", "--is-ancestor", CHOICE_COMMIT, "HEAD"],
                                capture_output=True, timeout=60)
            hyp = _git("show", f"{CHOICE_COMMIT}:./HYPOTHESIS.md") if a2.returncode == 0 else ""
            if a2.returncode != 0 or CHOICE_UTC not in hyp or CHOICE_LABEL not in hyp:
                problems.append(f"HYPOTHESIS.md at {CHOICE_COMMIT[:12]} does not record {CHOICE_UTC} "
                                f"'{CHOICE_LABEL}' (or the commit is not an ancestor of HEAD)")
    except Refused as e:
        problems.append(str(e))
    # The package is committed and pushed before any run (HYPOTHESIS "Diagnostics between runs").
    ident = package_identity()
    info["package"] = {k: ident.get(k) for k in ("commit", "clean", "pushed")}
    if not (ident.get("commit") and ident.get("clean") and ident.get("pushed")):
        problems.append(f"the package is not committed, clean and pushed: {info['package']}")
    # The frozen hash table (all 20 scopes) and the gate record's code hashes, recomputed.
    chk = _hash_tree("--check", str(KIT_DIR / "HYPOTHESIS.md"))
    info["hash_tree_check"] = {"returncode": chk.returncode,
                               "lines": [ln for ln in chk.stdout.splitlines() if ln.strip()][:40]}
    if chk.returncode != 0:
        problems.append("tools/hash_tree.py --check HYPOTHESIS.md failed (a scope does not match the frozen table)")
    info["code_hashes"] = {}
    for scope, (key, want) in GATE_CODE_HASHES.items():
        p = _hash_tree("--tree", scope)
        got = p.stdout.strip() if p.returncode == 0 else None
        info["code_hashes"][key] = got
        if got != want:
            problems.append(f"code hash {key} {got} != the gate record's {want[:12]}...")
    src = (KIT_DIR / "gate/checks/g1_synthetic.py").read_text(encoding="utf-8")
    excl = None
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "EXCLUDE" for t in node.targets):
            excl = tuple(ast.literal_eval(node.value))
    if excl != G1_EXCLUDE:
        problems.append(f"gate/checks/g1_synthetic.py EXCLUDE {excl} != {G1_EXCLUDE}")
    tsrc = (KIT_DIR / TEST_REL).read_text(encoding="utf-8")
    timing = None
    for node in ast.walk(ast.parse(tsrc)):
        if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "TIMING" for t in node.targets):
            timing = tuple(ast.literal_eval(node.value))
    if timing != TIMING:
        problems.append(f"{TEST_REL} TIMING {timing} != {TIMING}")
    if f"def {TEST_NAME}(" not in tsrc:
        problems.append(f"{TEST_NAME} not in {TEST_REL}")
    return info, problems


def host_pin_problems(host: str, info: dict) -> list[str]:
    problems = []
    arch = str(info.get("architecture") or "")
    if host == "mbp":
        if info.get("macos") != MBP_OS_PINS["macos"]:
            problems.append(f"macOS {info.get('macos')} != {MBP_OS_PINS['macos']}")
        if info.get("kern_osversion") != MBP_OS_PINS["kern_osversion"]:
            problems.append(f"kern.osversion {info.get('kern_osversion')} != {MBP_OS_PINS['kern_osversion']}")
        if not arch.startswith(MBP_OS_PINS["arch_prefix"]):
            problems.append(f"GPU architecture {info.get('architecture')} is not {MBP_OS_PINS['arch_prefix']}*")
        if info.get("hw_model") == MINI_PINS["hw_model"]:
            problems.append(f"hw.model {info.get('hw_model')} is the mini's")
    elif host == "mini":
        if info.get("hw_model") != MINI_PINS["hw_model"]:
            problems.append(f"hw.model {info.get('hw_model')} != {MINI_PINS['hw_model']} (the mini)")
        if arch.startswith(MBP_OS_PINS["arch_prefix"]):
            problems.append(f"GPU architecture {arch} is the mbp's")
    return problems


def runtime_env(host: str) -> tuple[dict, list[str]]:
    import importlib.metadata as md

    problems: list[str] = []
    v = os.environ.setdefault("MLX_ENABLE_TF32", "0")
    if v != "0":
        problems.append(f"MLX_ENABLE_TF32={v!r} (must be 0)")
    if os.environ.get("EXP036_PORT_FILE"):
        problems.append("EXP036_PORT_FILE is set (the mutation-testing hook)")
    info: dict = {"python": platform.python_version(), "MLX_ENABLE_TF32": os.environ.get("MLX_ENABLE_TF32")}
    for pkg in ("mlx", "mlx-metal", "mlx-lm", "pytest", "numpy"):
        try:
            info[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            info[pkg] = None
    for pkg, want in PKG_PINS.items():
        if info.get(pkg) != want:
            problems.append(f"{pkg} {info.get(pkg)} != {want}")
    info["macos"] = platform.mac_ver()[0]
    for key, name in (("kern_osversion", "kern.osversion"), ("hw_model", "hw.model")):
        try:
            info[key] = subprocess.run(["sysctl", "-n", name], capture_output=True, text=True,
                                       timeout=10).stdout.strip()
        except Exception:
            info[key] = None
    try:
        import mlx.core as mx

        di = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        info["architecture"] = di.get("architecture")
    except Exception as e:  # recorded; judged by the host pins
        info["architecture"] = None
        info["architecture_error"] = str(e)[:200]
    problems += host_pin_problems(host, info)
    return info, problems


def resolve_work(arg: Path | None, host: str) -> Path:
    if arg is not None:
        w = Path(arg).expanduser()
    elif os.environ.get("EXP036_WORK"):
        w = Path(os.environ["EXP036_WORK"]).expanduser() / "exp037" / "diag_g1_resume_20261007"
    elif host == "mini":
        w = Path.home() / "models" / "exp037-mini" / "diag_g1_resume_20261007"
    else:
        raise Refused("no --work and no $EXP036_WORK")
    w = w.resolve()
    if is_under(w, REPO_DIR):
        raise Refused("--work must be outside the repository")
    w.mkdir(parents=True, exist_ok=True)
    return w


def basetemp_root(utc: str) -> Path:
    """pytest's --basetemp parent for the pytest arms: the system temp directory, as pytest's own default, so that
    no test's tmp_path lies under $HOME or an $EXP036_* / $EXP037_* path (path-redaction tests read those)."""
    root = Path(tempfile.gettempdir()).resolve() / f"exp037_g1diag_{utc}"
    guards = [Path.home()] + [Path(os.path.expanduser(os.environ[v])) for v in ENV_LABELS if os.environ.get(v)]
    if any(is_under(root, g) for g in guards) or is_under(root, REPO_DIR):
        raise Refused("the temp directory lies under $HOME or an experiment path; set TMPDIR elsewhere")
    root.mkdir(parents=True, exist_ok=False)
    return root


# -------------------------------------------------------------------------------------------- the experiments


def _pytest_maker(exp: str, py: str):
    test_file = KIT_DIR / TEST_REL
    if exp == "isolated":
        return lambda j, b: pytest_isolated_cmd(py, j, b, test_file, TEST_NAME)
    if exp == "module":
        return lambda j, b: pytest_selection_cmd(py, j, b, [test_file])
    if exp == "pair":
        return lambda j, b: pytest_selection_cmd(py, j, b, [TESTS_DIR / PAIR_FIRST, test_file])
    return lambda j, b: pytest_g1ctx_cmd(py, j, b, TESTS_DIR)


def run_experiment(args) -> int:
    utc = utc_stamp()
    host, exp = args.host, args.experiment
    if not host or not exp:
        raise SystemExit("--host and --experiment are required")
    out_dir = check_out_dir(Path(args.out))
    official = out_dir == OFFICIAL_OUT.resolve()
    reg = REGISTERED_N.get((host, exp))
    n = args.n if args.n is not None else reg
    if n is None:
        raise SystemExit(f"PLAN.md registers no {exp} run on the {host}; give --n and a --out outside out/")
    if args.stub and (official or exp not in ("inproc", "fresh", "cachestate")):
        raise SystemExit("--stub is for the selftest: inproc, fresh or cachestate only, never into out/")
    if official and n != reg:
        raise SystemExit(f"PLAN.md registers N = {reg} for {host} {exp}; out/ takes registered runs only")
    existing = load_outputs(out_dir) if official else []
    if official:
        g = official_guard(existing, host, exp, args.repeat_approved)
        if g:
            raise Refused("; ".join(g))
    t_start = utc_iso()
    t0 = time.time()
    work = resolve_work(args.work, host)
    run_dir = work / f"{exp}_{utc}"
    run_dir.mkdir(parents=True, exist_ok=False)
    header = {"schema": SCHEMA, "package": PACKAGE, "experiment": exp, "host": host, "utc": utc,
              "t_start": t_start, "n": n, "n_registered": reg, "stub": args.stub or False,
              "gate_run": GATE_UTC, "gate_record_sha256": GATE_RECORD_SHA256, "test": f"{TEST_REL}::{TEST_NAME}",
              "package_sha256": {name: sha256_file(PKG_DIR / name) for name in PACKAGE_FILES
                                 if (PKG_DIR / name).is_file()},
              "earlier_outputs": [o.get("_file") for o in existing
                                  if o.get("host") == host and o.get("experiment") == exp],
              "repeat_approved": args.repeat_approved}
    extra = [(str(work), "<work>")]
    if not args.stub:
        kit, p1 = kit_binding()
        env, p2 = runtime_env(host)
        if p1 + p2:
            print(json.dumps({"ok": False, "refused": p1 + p2}))
            return 3
        header.update(kit=kit, env=env)
    bt_root = None
    if exp in PYTEST_ARMS:
        bt_root = basetemp_root(utc)
        extra.append((str(bt_root), "<basetemp>"))
    redactor = Redactor(extra=tuple(extra))
    out_path = out_dir / f"{host}_{exp}_{utc}.json"

    def summary_of(runs):
        if exp in PYTEST_ARMS:
            return summarise_pytest(runs)
        if exp == "cachestate":
            return {"outcome": runs[0].get("outcome") if runs else None}
        return summarise_inproc(runs)

    def emit(runs, complete, more=None):
        obj = dict(header, runs=runs, complete=complete, n_done=len(runs),
                   n_valid=sum(1 for r in runs if r.get("valid", True)), wall_s=round(time.time() - t0, 1),
                   t_end=utc_iso())
        obj["summary"] = summary_of(runs)
        if more:
            obj.update(more)
        full = run_dir / f"full_{out_path.name}"
        write_output(obj, full, redactor)  # the uncompacted output, always, in the work directory
        if len(json.dumps(obj, indent=1, sort_keys=True, ensure_ascii=False)) > OUT_BUDGET:
            obj = dict(obj, runs=compact_runs(runs),
                       compacted={"budget_bytes": OUT_BUDGET, "keep_full": KEEP_FULL,
                                  "full_copy": redactor(str(full)), "full_copy_sha256": sha256_file(full)})
        write_output(obj, out_path, redactor)

    reached = True
    if exp in PYTEST_ARMS:
        env = g1_env(os.environ)
        header["command"] = _pytest_maker(exp, "$PY")(Path("<work>/junit_<k>.xml"), Path(f"<basetemp>/{exp}_<k>"))
        runs, reached = run_pytest_series(exp, n, run_dir, bt_root, KIT_DIR, env, _pytest_maker(exp, sys.executable),
                                          TIMEOUT_S[exp], lambda r: emit(r, False))
        emit(runs, reached)
    elif exp == "inproc":
        if args.stub:
            S, model = StubScenario(args.stub), None
        else:
            T, model, precision = load_real(run_dir)
            header["env"]["tf32_probe"] = precision
            S = RealScenario(T, model)
        runs, a0 = inproc_iterations(S, n, run_dir, progress=lambda r: emit(r, False))
        emit(runs, True)  # complete before the descriptive margins, which decide nothing
        if model is not None and not args.no_margin:
            try:
                more = {"margins": inproc_margins(S, model, runs, a0)}
            except Exception as e:  # recorded; the output stays complete
                more = {"margins_error": f"{type(e).__name__}: {str(e)[:500]}"}
            emit(runs, True, more)
    elif exp == "cachestate":
        if args.stub:
            W = StubWorld()
            P, make_S = W, (lambda: StubScenario("deterministic", world=W))
        else:
            T, precision = load_test_module()
            header["env"]["tf32_probe"] = precision
            P = RealProbe(T, run_dir)
            make_S = lambda: RealScenario(T, P.model, stub_cls=P.StubTok)  # noqa: E731
        runs = [dict(cachestate_probe(P, make_S, run_dir / "probe"), i=0, valid=True)]
        emit(runs, True)
    elif exp == "fresh":
        runs, reached = fresh_series(host, n, run_dir, args.stub, lambda r: emit(r, False))
        emit(runs, reached)
    else:
        raise SystemExit(f"unknown experiment {exp}")
    print(json.dumps({"ok": True, "complete": reached, "output": redactor(str(out_path)),
                      "summary": json.loads(redactor(json.dumps(summary_of(runs))))}))
    if not reached:
        print(json.dumps({"note": "the invalid-run cap was reached before N valid runs: the output is incomplete "
                                  "and counts as a crashed run (PLAN.md 3.0)"}))
        return 4
    return 0


def fresh_series(host: str, n: int, run_dir: Path, stub: str | None, progress) -> tuple[list[dict], bool]:
    """L2: (iii) with one iteration per process, until n valid processes or attempts_cap(n) attempts (each process
    loads the model as the fixture does). A process refused by a precondition stops the series (exit 3)."""
    runs: list[dict] = []
    cap = attempts_cap(n)
    k = 0
    while sum(1 for r in runs if r.get("valid", True)) < n and k < cap:
        sub_out, sub_work = run_dir / f"proc_{k:03d}_out", run_dir / f"proc_{k:03d}_work"
        cmd = [sys.executable, str(Path(__file__).resolve()), "--host", host, "--experiment", "inproc", "--n", "1",
               "--out", str(sub_out), "--work", str(sub_work), "--no-margin"] + (["--stub", stub] if stub else [])
        try:
            p = subprocess.run(cmd, cwd=str(KIT_DIR), env=dict(os.environ), capture_output=True, text=True,
                               timeout=TIMEOUT_S["fresh"])
            rc, err = p.returncode, (p.stderr or "")[-2000:]
        except subprocess.TimeoutExpired:
            rc, err = None, "timeout"
        if rc == 3:
            raise Refused(f"L2 process {k} was refused by a precondition: {(p.stdout or '')[-500:]}")
        files = sorted(sub_out.glob(f"{host}_inproc_*.json"))
        if rc == 0 and len(files) == 1:
            sub = json.loads(files[0].read_text(encoding="utf-8"))
            runs.append(dict(sub["runs"][0], process=k, i=k, valid=True))
        else:
            runs.append({"process": k, "i": k, "valid": False, "returncode": rc, "stderr_tail": err})
        progress(runs)
        k += 1
    return runs, sum(1 for r in runs if r.get("valid", True)) >= n


# ------------------------------------------------------------------------------- classification (PLAN.md s5)


ACTIONS = {
    "O1": "Generation is non-deterministic within one process -> L1 localisation on the uninterrupted pair "
          "(PLAN.md 6.1); its class decides through the scope classes of section 4 (K-in / K-sel in port/ or the "
          "generation path -> gate fix there, 6.8; M -> 6.5's listed remedies, else stop and publish; N -> stop and "
          "publish).",
    "O4-split": "Only split fields differ and (iv) shows the stale tokenizer-check state on the run host -> gate fix "
                "at the reading site, runner.generate._scorer_split (PLAN.md 6.6).",
    "O4-trace": "Only non-generation fields differ, seen in a fresh process ((i) or (iii)) -> L1f field trace "
                "(PLAN.md 6.4) in that setting; the input it names decides through the scope classes of section 4.",
    "O4-order": "Only non-generation fields differ, seen only in a context selection -> L3 (PLAN.md 6.3) on the "
                "smallest selection that showed it; the state it names decides through the scope classes.",
    "O2": "The resume path changes generation while uninterrupted runs agree -> L1 on the A-C pair (PLAN.md 6.1); "
          "its class decides through the scope classes of section 4.",
    "O2c-line": "A C-assertion failed in (iii) or (i) -> gate fix in the producing code of that line (PLAN.md 6.7, "
                "all in the generation path), with the failing test's scenario as the test (6.8).",
    "O2c-order": "A C-assertion failed only in a context selection -> L3 (PLAN.md 6.3) on the smallest selection.",
    "O6-pending": "(i) reproduces with a generation difference or another failure -> run L2 (--experiment fresh "
                  "--n 100 on the mbp; its rules and code are in this package), commit its output, classify again.",
    "O6-O1": "L2: a generation difference in an uninterrupted pair -> O1's consequence, L1 in fresh processes.",
    "O6-O2": "L2: an A-C generation difference -> O2's consequence, L1 in fresh processes.",
    "O6-O2c": "L2: a C-assertion failure -> O2c-line's consequence.",
    "O6-O4-split": "L2: only split fields differ and (iv) shows the stale state -> O4-split's consequence.",
    "O6-O4": "L2: only non-generation fields differ -> L1f in fresh processes (PLAN.md 6.4).",
    "O6-stop": "L2 reproduces nothing in 100 fresh processes -> the trigger lies in the pytest harness or test code, "
               "not in kit runtime code -> stop and publish.",
    "O3": "A context selection ((i'), (i'') or (ii)) reproduces with a generation difference or another failure, "
          "(i) does not -> L3 (PLAN.md 6.3) on the smallest selection that reproduced.",
    "O7": "(iv) shows on the run host that a StubTok created after the batch-flip test's calls lands on a stale "
          "check-cache id and gets the scorers split in the failing test's scenario -> gate fix at "
          "runner.generate._scorer_split (PLAN.md 6.6).",
    "O1-mini": "Mini fallback: uninterrupted generation differs on the mini -> L1 on the mini; every fix criterion "
               "must also hold on the mbp (PLAN.md 6.5, 6.8).",
    "O7-mini": "Mini fallback: (iv) hit on the mini -> gate fix at runner.generate._scorer_split (PLAN.md 6.6); "
               "its test must also fail before and pass after on the mbp.",
    "O5": "Nothing reproduces on the mbp and no mini fallback applies -> stop and publish (G1 keeps its failure; "
          "the flakiness and the detection power disclosed; no code change exists, so P1(d) refuses a re-run).",
}
FINAL = {"stop": "stop and publish: at least one pursued rule ends in stop and publish (PLAN.md section 5)",
         "pending": "follow-ups pending: run each named follow-up after its code is committed and pushed, then "
                    "classify again",
         "fix": "gate fix: one amendment lists every fix named (PLAN.md 6.8); writing it and any re-run need "
                "Andrei's go"}


def _official(o: dict) -> bool:
    return o.get("schema") == SCHEMA and o.get("n") == REGISTERED_N.get((o.get("host"), o.get("experiment")))


def observe(outputs: list[dict], host: str) -> dict:
    """Every recorded valid run or iteration counts (complete outputs and partial ones of crashed runs);
    complete = exactly one complete official output; duplicate = more than one."""
    obs: dict = {}
    for exp in EXPERIMENTS:
        outs = [o for o in outputs if o.get("host") == host and o.get("experiment") == exp]
        comp = [o for o in outs if o.get("complete") and _official(o)]
        runs = [r for o in outs for r in o.get("runs", []) if r.get("valid", True) is not False]
        e: dict = {"files": len(outs), "complete_outputs": len(comp), "complete": len(comp) == 1,
                   "duplicate": len(comp) > 1, "n_valid": len(runs)}
        if exp in PYTEST_ARMS:
            e["kinds"] = dict(Counter(r["kind"] for r in runs))
            e["field_only"] = e["kinds"].get("field_only", 0)
            e["field_only_fields"] = sorted({f for r in runs if r["kind"] == "field_only"
                                             for f in pair_fields(r.get("pair"))})
            e["c_assert"] = e["kinds"].get("c_assert", 0)
            e["c_assert_lines"] = sorted({(r.get("target") or {}).get("line") for r in runs
                                          if r["kind"] == "c_assert"} - {None})
            e["gen_or_other"] = e["kinds"].get("generation", 0) + e["kinds"].get("other", 0)
        elif exp in ("inproc", "fresh"):
            e["uninterrupted"] = dict(Counter(r["uninterrupted_kind"] for r in runs))
            e["ac"] = dict(Counter(r["ac_kind"] for r in runs))
            e["c_assert"] = sum(1 for r in runs if r.get("c_assert"))
            e["c_assert_checks"] = sorted({c for r in runs for c in r.get("c_assert", [])})
            fo: set = set()
            n_fo = 0
            for r in runs:
                hit = False
                for name in PAIRS_UNINTERRUPTED + ("AC",):
                    p = r.get(name)
                    if isinstance(p, dict) and p.get("kind") == "field_only":
                        fo.update(pair_fields(p))
                        hit = True
                n_fo += hit
            e["field_only"], e["field_only_fields"] = n_fo, sorted(fo)
        elif exp == "cachestate":
            oc = [r.get("outcome") for r in runs]
            e["outcome"] = ("hit" if "hit" in oc else "stale_only" if "stale_only" in oc else "none" if oc else None)
        obs[exp] = e
    rep = []
    for exp in PYTEST_ARMS:
        rep += [f"{exp}:{k}" for k, c in obs[exp]["kinds"].items() if k != "equal" for _ in range(c)]
    for exp in ("inproc",):
        rep += [f"{exp}:uninterrupted:{k}" for k, c in obs[exp]["uninterrupted"].items() if k != "equal"
                for _ in range(c)]
        rep += [f"{exp}:AC:{k}" for k, c in obs[exp]["ac"].items() if k != "equal" for _ in range(c)]
        rep += [f"{exp}:c_assert"] * obs[exp]["c_assert"]
    obs["reproductions"] = rep
    return obs


def _rule(rule: int, outcome: str, status: str, key: str, **kw) -> dict:
    return dict(rule=rule, outcome=outcome, sub=key, status=status, consequence=ACTIONS[key], **kw)


def split_case(fields: list[str], cs_outcome: str | None) -> bool:
    return bool(fields) and set(fields) <= set(SPLIT_FIELDS) and "split_by" in fields \
        and cs_outcome in ("hit", "stale_only")


def l2_rules(M: dict) -> list[dict]:
    F = M["fresh"]
    if not F["complete"]:
        return [_rule(5, "O6", "pending", "O6-pending", follow_up="L2: --experiment fresh --n 100 on the mbp")]
    res = []
    if F["uninterrupted"].get("generation"):
        res.append(_rule(5, "O6", "pending", "O6-O1", follow_up="L1, uninterrupted pair, fresh processes, mbp"))
    if F["field_only"]:
        if split_case(F["field_only_fields"], M["cachestate"]["outcome"]):
            res.append(_rule(5, "O6", "fix", "O6-O4-split", fields=F["field_only_fields"], fix="6.6"))
        else:
            res.append(_rule(5, "O6", "pending", "O6-O4", fields=F["field_only_fields"],
                             follow_up="L1f, fresh processes, mbp"))
    if F["ac"].get("generation"):
        res.append(_rule(5, "O6", "pending", "O6-O2", follow_up="L1, A-C pair, fresh processes, mbp"))
    if F["c_assert"]:
        res.append(_rule(5, "O6", "fix", "O6-O2c", checks=F["c_assert_checks"], fix="6.7"))
    if not res:
        res.append(_rule(5, "O6", "stop", "O6-stop"))
    return res


def mbp_rules(M: dict) -> list[dict]:
    """Rules 1-7 on the mbp (governing host). Each condition stands alone; every matching rule is pursued."""
    out = []
    arms = [a for a in REQUIRED["mbp"] if a != "cachestate"]
    if M["inproc"]["uninterrupted"].get("generation"):
        out.append(_rule(1, "O1", "pending", "O1", follow_up="L1, uninterrupted pair, mbp"))
    fo_arms = [a for a in arms if M[a].get("field_only")]
    if fo_arms:
        F = sorted(set().union(*(M[a]["field_only_fields"] for a in fo_arms)))
        if split_case(F, M["cachestate"]["outcome"]):
            out.append(_rule(2, "O4", "fix", "O4-split", fields=F, arms=fo_arms, fix="6.6"))
        elif set(fo_arms) & {"isolated", "inproc"}:
            where = "in-process ((iii) iterations)" if "inproc" in fo_arms else "fresh processes (as L2)"
            out.append(_rule(2, "O4", "pending", "O4-trace", fields=F, arms=fo_arms, follow_up=f"L1f, {where}, mbp"))
        else:
            first = next(a for a in CONTEXT_ARMS if a in fo_arms)
            out.append(_rule(2, "O4", "pending", "O4-order", fields=F, arms=fo_arms, follow_up=f"L3 on {first}"))
    if M["inproc"]["ac"].get("generation"):
        out.append(_rule(3, "O2", "pending", "O2", follow_up="L1, A-C pair, mbp"))
    ca_arms = [a for a in ("inproc",) + PYTEST_ARMS if M[a].get("c_assert")]
    if ca_arms:
        lines = sorted(set(M["inproc"].get("c_assert_checks", []))
                       | {f"l{x}" for a in PYTEST_ARMS for x in M[a].get("c_assert_lines", [])})
        if set(ca_arms) & {"inproc", "isolated"}:
            out.append(_rule(4, "O2c", "fix", "O2c-line", arms=ca_arms, checks=lines, fix="6.7"))
        else:
            first = next(a for a in CONTEXT_ARMS if a in ca_arms)
            out.append(_rule(4, "O2c", "pending", "O2c-order", arms=ca_arms, checks=lines,
                             follow_up=f"L3 on {first}"))
    if M["isolated"].get("gen_or_other"):
        out += l2_rules(M)
    ctx = [a for a in CONTEXT_ARMS if M[a].get("gen_or_other")]
    if ctx:
        out.append(_rule(6, "O3", "pending", "O3", arms=ctx, follow_up=f"L3 on {ctx[0]}"))
    if M["cachestate"]["outcome"] == "hit":
        out.append(_rule(7, "O7", "fix", "O7", fix="6.6"))
    return out


def mini_rules(N: dict) -> list[dict]:
    """Rule 8: the mini's evidence, used only when no mbp rule matches."""
    out = []
    if N["inproc"]["uninterrupted"].get("generation"):
        out.append(_rule(8, "O1", "pending", "O1-mini", follow_up="L1, uninterrupted pair, mini"))
    if N["cachestate"]["outcome"] == "hit":
        out.append(_rule(8, "O7", "fix", "O7-mini", fix="6.6"))
    return out


def classify(outputs: list[dict], allow_stub: bool = False, ident: dict | None = None) -> dict:
    outputs = [o for o in outputs if o.get("schema") == SCHEMA and (allow_stub or not o.get("stub"))]
    obs = {h: observe(outputs, h) for h in HOSTS}
    base = {"schema": SCHEMA, "package": PACKAGE, "governing_host": "mbp", "observations": obs,
            "inputs": sorted({f"{o.get('host')}_{o.get('experiment')}_{o.get('utc')}" for o in outputs}),
            "mini_reproduced": bool(obs["mini"]["reproductions"])}
    problems = package_problems(outputs, ident)
    problems += [f"{h}/{e}: more than one complete official output" for h in HOSTS for e in EXPERIMENTS
                 if obs[h][e]["duplicate"]]
    if problems:
        return dict(base, status="refused", problems=problems, outcome=None, decision=None, matched=[])
    missing = [f"{h}/{e}" for h in HOSTS for e in REQUIRED[h] if not obs[h][e]["complete"]]
    if missing:
        return dict(base, status="incomplete", missing=missing, outcome=None, decision=None, matched=[])
    matched = mbp_rules(obs["mbp"])
    mini = mini_rules(obs["mini"])
    fallback = False
    if not matched and mini:
        matched, fallback = mini, True
    if not matched:
        matched = [_rule(9, "O5", "stop", "O5")]
    st = {m["status"] for m in matched}
    final = "stop" if "stop" in st else "pending" if "pending" in st else "fix"
    fixes = sorted({m["fix"] for m in matched if m.get("fix")})
    return dict(base, status="applied" if final != "pending" else "follow-up pending",
                outcome=matched[0]["outcome"], sub=matched[0]["sub"], matched=matched,
                mini_matches_descriptive=[] if fallback else mini, mini_fallback=fallback,
                decision=final, decision_text=FINAL[final], fixes=fixes if final == "fix" else [],
                follow_ups=[m["follow_up"] for m in matched if m.get("follow_up")])


OUTPUT_RE = re.compile(r"^(mbp|mini)_(cachestate|isolated|module|pair|g1ctx|inproc|fresh)_\d{8}T\d{6}Z\.json$")


def load_outputs(out_dir: Path) -> list[dict]:
    outs = []
    for p in sorted(Path(out_dir).glob("*.json")):
        if OUTPUT_RE.match(p.name):
            o = json.loads(p.read_text(encoding="utf-8"))
            o["_file"], o["_sha256"] = p.name, sha256_file(p)
            outs.append(o)
    return outs


def run_classify(args) -> int:
    out_dir = check_out_dir(Path(args.out))
    outs = load_outputs(out_dir)
    res = classify(outs, ident=package_identity())
    res["inputs"] = [{"file": o["_file"], "sha256": o["_sha256"]} for o in outs]
    res["utc"] = utc_stamp()
    if args.write:
        if res["status"] in ("incomplete", "refused"):
            raise SystemExit(f"{res['status']}: the table is not applied; nothing written")
        write_output(res, out_dir / f"classification_{res['utc']}.json", Redactor())
    print(json.dumps({k: res.get(k) for k in ("status", "problems", "missing", "outcome", "sub", "decision",
                                               "decision_text", "fixes", "follow_ups", "matched", "mini_fallback",
                                               "mini_matches_descriptive", "mini_reproduced")}, indent=1))
    return 0


# ------------------------------------------------------------------------------------------------- selftest

STUB_TEST = '''
import json

TIMING = ("t_submit", "t_first_token", "t_done", "wall_s", "batch_id")


def _rec(item, ids, t):
    return {"type": "record", "key": {"arm": "K8", "task": "mmlu_en", "effort": "high", "item": item, "pass": 0},
            "completion_ids": ids, "text": "".join(f"<{i}>" for i in ids), "answer_text": "".join(f"<{i}>" for i in ids),
            "finish_reason": "length", "t_done": t, "batch_id": t}


def _write(p, recs, torn=False):
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        f.write(json.dumps({"type": "header"}) + "\\n")
        for r in recs:
            f.write(json.dumps(r, sort_keys=True) + "\\n")


def _records(p):
    out = {}
    for line in open(p):
        r = json.loads(line)
        if r.get("type") == "record" and r["key"]["item"] not in out:
            out[r["key"]["item"]] = r
    return out


def content(recs):
    return {k: {f: v for f, v in r.items() if f not in TIMING} for k, r in recs.items()}


def test_resume_after_simulated_crash_gives_identical_records(tmp_path):
    ref = tmp_path / "ref.jsonl"
    p = tmp_path / "raw/S2/K8/mmlu_en_high.jsonl"
    _write(ref, [_rec(f"q{k:03d}", [k, k + 1, k + 2], 1) for k in range(6)])
    _write(p, [_rec(f"q{k:03d}", [k, k + 1, k + 2 + (k == 3)], 2) for k in range(6)])
    assert content(_records(p)) == content(_records(ref))


def test_an_unrelated_failure():
    assert 1 == 2
'''

STUB_SETUP_ERROR = '''
import pytest


@pytest.fixture
def broken():
    raise RuntimeError("setup fails")


def test_resume_after_simulated_crash_gives_identical_records(broken, tmp_path):
    assert True
'''

STUB_C_ASSERT = '''
def test_resume_after_simulated_crash_gives_identical_records(tmp_path):
    done = {1, 2}
    assert len(done) == 3
'''

STUB_FIRST = '''
def test_runs_first():
    assert True
'''

PKG_STUB = {"PLAN.md": "a" * 64, "diag_resume.py": "c" * 64}


def _synthetic(host: str, exp: str, kinds: list, fields=("completion_ids",)) -> dict:
    """A complete output with the registered N whose runs have the given kinds (classifier tests only). pytest
    arms: equal | field_only | generation | c_assert | other. inproc / fresh: (uninterrupted, AC) with AC also
    "c_assert" (a C-assertion failure, AC equal). cachestate: [outcome]."""
    n = REGISTERED_N[(host, exp)]

    def pair(kind):
        if kind == "equal":
            return {"kind": "equal"}
        return {"kind": kind, "items": {"q003": {"fields": {f: {"a": 1, "b": 2} for f in fields},
                                                 "completion_ids_equal": kind == "field_only"}}}

    runs = []
    if exp == "cachestate":
        runs = [{"i": 0, "valid": True, "outcome": kinds[0] if kinds else "none"}]
    else:
        kinds = list(kinds) + ["equal"] * (n - len(kinds))
        for i, k in enumerate(kinds):
            if exp in PYTEST_ARMS:
                runs.append({"k": i, "valid": True, "kind": k,
                             "target": {"line": 219} if k == "c_assert" else {"line": TEST_LINE},
                             "pair": pair(k) if k in ("field_only", "generation") else None})
            else:
                u, a = k if isinstance(k, tuple) else (k, "equal")
                runs.append({"i": i, "AB": pair(u), "A_vs_A0": {"kind": "equal"}, "B_vs_A0": {"kind": "equal"},
                             "AC": pair(a if a != "c_assert" else "equal"), "BC": {"kind": "equal"},
                             "c_assert": ["l221"] if a == "c_assert" else [], "uninterrupted_kind": u,
                             "ac_kind": a if a != "c_assert" else "equal"})
    return {"schema": SCHEMA, "host": host, "experiment": exp, "n": n, "complete": True, "stub": False,
            "utc": "20261007T000000Z", "runs": runs, "package_sha256": dict(PKG_STUB)}


def _quiet(skip=()) -> list[dict]:
    return [_synthetic(h, e, []) for h in HOSTS for e in REQUIRED[h] if (h, e) not in skip]


def selftest(work: Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    done: list[str] = []

    def ok(name, cond, detail=""):
        if not cond:
            raise AssertionError(f"selftest {name} failed {detail}")
        done.append(name)

    # 1. the comparison
    a = {"q000": {"completion_ids": [1, 2], "text": "<1><2>", "split_by": "x"}}
    ok("diff.equal", diff_pair(a, json.loads(json.dumps(a)))["kind"] == "equal")
    b = {"q000": {"completion_ids": [1, 3], "text": "<1><3>", "split_by": "x"}}
    d = diff_pair(a, b)
    ok("diff.generation", d["kind"] == "generation" and d["items"]["q000"]["first_divergence"]["index"] == 1
       and set(d["items"]["q000"]["fields"]) == {"completion_ids", "text"})
    c = {"q000": {"completion_ids": [1, 2], "text": "<1><2>", "split_by": "y"}}
    ok("diff.field_only", diff_pair(a, c)["kind"] == "field_only" and pair_fields(diff_pair(a, c)) == ["split_by"])
    ok("diff.missing", diff_pair(a, {})["kind"] == "generation")
    ok("content.timing", content({"q": {"t_done": 1, "wall_s": 2, "batch_id": 3, "x": 4}}) == {"q": {"x": 4}})

    # 2. (iii) on the stub models; the classifier on their outputs (the mbp, then the mini only)
    results = {}
    for kind, n in (("deterministic", 6), ("noisy", 8), ("resume", 5), ("fieldonly", 5)):
        S = StubScenario(kind)
        runs, a0 = inproc_iterations(S, n, work / f"inproc_{kind}")
        summ = summarise_inproc(runs)
        ok(f"inproc.{kind}.c_checks", summ["c_checks_failed"] == 0, summ)
        out = dict(_synthetic("mbp", "inproc", []), runs=runs, stub=True)
        res = classify(_quiet(skip={("mbp", "inproc")}) + [out], allow_stub=True)
        results[kind] = res["sub"]
        if kind == "deterministic":
            ok("stub.deterministic.no_difference", summ["uninterrupted"] == {"equal": n} and summ["ac"] == {"equal": n})
            ok("stub.deterministic.O5", res["outcome"] == "O5" and res["decision"] == "stop"
               and res["mini_reproduced"] is False, res.get("matched"))
        elif kind == "noisy":
            ok("stub.noisy.uninterrupted_differs", summ["uninterrupted"].get("generation", 0) >= 1, summ)
            ok("stub.noisy.O1", res["outcome"] == "O1" and res["decision"] == "pending", res.get("matched"))
        elif kind == "resume":
            ok("stub.resume.A_equals_B", summ["uninterrupted"] == {"equal": n}, summ)
            ok("stub.resume.C_differs", summ["ac"] == {"generation": n}
               and summ["first_difference"]["items"] == ["q003"], summ)
            ok("stub.resume.O2", res["outcome"] == "O2" and [m["sub"] for m in res["matched"]] == ["O2"],
               res.get("matched"))
        else:
            ok("stub.fieldonly.only_split_by", summ["ac"] == {"field_only": n}, summ)
            ok("stub.fieldonly.O4_trace", res["outcome"] == "O4" and res["sub"] == "O4-trace"
               and res["matched"][0]["fields"] == ["split_by"], res.get("matched"))
            cs = _synthetic("mbp", "cachestate", ["stale_only"])
            res2 = classify(_quiet(skip={("mbp", "inproc"), ("mbp", "cachestate")}) + [out, cs], allow_stub=True)
            ok("stub.fieldonly.O4_split", res2["sub"] == "O4-split" and res2["decision"] == "fix"
               and res2["fixes"] == ["6.6"], res2.get("matched"))
        out_mini = dict(out, host="mini")
        res_m = classify(_quiet(skip={("mini", "inproc")}) + [out_mini], allow_stub=True)
        want = {"deterministic": "O5", "noisy": "O1-mini", "resume": "O5", "fieldonly": "O5"}[kind]
        ok(f"stub.{kind}.mini_only.{want}", res_m["sub"] == want
           and res_m["mini_reproduced"] is (kind != "deterministic"), res_m.get("matched"))

    # 3. the classifier on synthetic outputs
    cls = classify
    ok("classify.incomplete", cls(_quiet(skip={("mini", "g1ctx")}))["status"] == "incomplete")
    ok("classify.mini_module_not_required", cls(_quiet())["status"] == "applied")
    ok("classify.quiet.O5", cls(_quiet())["outcome"] == "O5")
    iso = [_synthetic("mbp", "isolated", ["generation"])]
    r = cls(_quiet(skip={("mbp", "isolated")}) + iso)
    ok("classify.isolated_only.O6_pending", r["sub"] == "O6-pending" and r["decision"] == "pending")
    base6 = _quiet(skip={("mbp", "isolated")}) + iso
    ok("classify.L2.stop", cls(base6 + [_synthetic("mbp", "fresh", [])])["decision"] == "stop")
    ok("classify.L2.O2", cls(base6 + [_synthetic("mbp", "fresh", [("equal", "generation")])])["sub"] == "O6-O2")
    ok("classify.L2.O1", cls(base6 + [_synthetic("mbp", "fresh", [("generation", "equal")])])["sub"] == "O6-O1")
    ok("classify.L2.O2c", cls(base6 + [_synthetic("mbp", "fresh", [("equal", "c_assert")])])["sub"] == "O6-O2c")
    fr_split = _synthetic("mbp", "fresh", [("equal", "field_only")], fields=("split_by",))
    r = cls(base6 + [fr_split])
    ok("classify.L2.O4_trace", r["sub"] == "O6-O4" and r["decision"] == "pending")
    r = cls([o for o in base6 if not (o["host"] == "mbp" and o["experiment"] == "cachestate")]
            + [fr_split, _synthetic("mbp", "cachestate", ["hit"])])
    ok("classify.L2.O4_split_and_O7", [m["sub"] for m in r["matched"]] == ["O6-O4-split", "O7"]
       and r["decision"] == "fix" and r["fixes"] == ["6.6"], r.get("matched"))
    r = cls(_quiet(skip={("mbp", "g1ctx")}) + [_synthetic("mbp", "g1ctx", ["generation"])])
    ok("classify.g1ctx_only.O3", r["outcome"] == "O3" and r["matched"][0]["follow_up"] == "L3 on g1ctx")
    r = cls(_quiet(skip={("mbp", "g1ctx")}) + [_synthetic("mbp", "g1ctx", ["other"])])
    ok("classify.g1ctx_other.O3", r["outcome"] == "O3")
    r = cls(_quiet(skip={("mbp", "g1ctx"), ("mbp", "pair")})
            + [_synthetic("mbp", "g1ctx", ["other"]), _synthetic("mbp", "pair", ["generation"])])
    ok("classify.smallest_selection", r["matched"][0]["follow_up"] == "L3 on pair")
    r = cls(_quiet(skip={("mbp", "g1ctx")}) + [_synthetic("mbp", "g1ctx", ["field_only"], fields=("stop_token",))])
    ok("classify.g1ctx_field_only.O4_order", r["sub"] == "O4-order" and r["matched"][0]["fields"] == ["stop_token"])
    r = cls(_quiet(skip={("mbp", "pair"), ("mbp", "cachestate")})
            + [_synthetic("mbp", "pair", ["field_only"], fields=("split_by",)),
               _synthetic("mbp", "cachestate", ["stale_only"])])
    ok("classify.pair_split.O4_split", r["sub"] == "O4-split" and r["decision"] == "fix")
    r = cls(_quiet(skip={("mbp", "pair")}) + [_synthetic("mbp", "pair", ["field_only"], fields=("split_by",))])
    ok("classify.pair_split_no_state.O4_order", r["sub"] == "O4-order" and r["decision"] == "pending")
    r = cls(_quiet(skip={("mbp", "inproc")}) + [_synthetic("mbp", "inproc", [("equal", "c_assert")])])
    ok("classify.c_assertion.O2c_line", r["sub"] == "O2c-line" and r["decision"] == "fix"
       and r["matched"][0]["checks"] == ["l221"])
    r = cls(_quiet(skip={("mbp", "module")}) + [_synthetic("mbp", "module", ["c_assert"])])
    ok("classify.c_assertion_context.O2c_order", r["sub"] == "O2c-order" and r["matched"][0]["checks"] == ["l219"])
    r = cls(_quiet(skip={("mbp", "cachestate")}) + [_synthetic("mbp", "cachestate", ["hit"])])
    ok("classify.cachestate_hit.O7", r["outcome"] == "O7" and r["decision"] == "fix")
    r = cls(_quiet(skip={("mbp", "cachestate")}) + [_synthetic("mbp", "cachestate", ["stale_only"])])
    ok("classify.cachestate_stale_only_alone.O5", r["outcome"] == "O5")
    r = cls([o for o in base6 if not (o["host"] == "mbp" and o["experiment"] == "cachestate")]
            + [_synthetic("mbp", "fresh", []), _synthetic("mbp", "cachestate", ["hit"])])
    ok("classify.any_stop_wins", [m["sub"] for m in r["matched"]] == ["O6-stop", "O7"] and r["decision"] == "stop")
    r = cls(_quiet(skip={("mbp", "inproc"), ("mbp", "isolated")})
            + [_synthetic("mbp", "inproc", [("equal", "field_only")], fields=("split_by",)), iso[0]])
    ok("classify.cooccurrence", [m["sub"] for m in r["matched"]] == ["O4-trace", "O6-pending"])
    r = cls(_quiet(skip={("mini", "cachestate")}) + [_synthetic("mini", "cachestate", ["hit"])])
    ok("classify.mini_cachestate_hit.O7_mini", r["sub"] == "O7-mini" and r["mini_fallback"] is True)
    r = cls(_quiet(skip={("mini", "isolated")}) + [_synthetic("mini", "isolated", ["generation"])])
    ok("classify.mini_isolated_only.O5", r["outcome"] == "O5" and r["mini_reproduced"] is True)
    r = cls(_quiet(skip={("mbp", "cachestate"), ("mini", "cachestate")})
            + [_synthetic("mbp", "cachestate", ["hit"]), _synthetic("mini", "cachestate", ["hit"])])
    ok("classify.mbp_first", [m["sub"] for m in r["matched"]] == ["O7"] and r["mini_fallback"] is False
       and [m["sub"] for m in r["mini_matches_descriptive"]] == ["O7-mini"])
    short = _synthetic("mbp", "inproc", [])
    short["n"] = 10
    ok("classify.unregistered_n.incomplete", cls(_quiet(skip={("mbp", "inproc")}) + [short])["status"] == "incomplete")
    ok("classify.stub_refused", cls(_quiet(skip={("mbp", "inproc")}) + [dict(_synthetic("mbp", "inproc", []),
                                                                          stub=True)])["status"] == "incomplete")
    ok("classify.duplicate_refused", cls(_quiet() + [_synthetic("mbp", "inproc", [])])["status"] == "refused")
    mixed = _quiet(skip={("mbp", "inproc")}) + [dict(_synthetic("mbp", "inproc", []),
                                                     package_sha256=dict(PKG_STUB, **{"PLAN.md": "d" * 64}))]
    ok("classify.mixed_package_refused", cls(mixed)["status"] == "refused")
    ok("classify.package_ident", package_problems(_quiet(), {"sha256": dict(PKG_STUB), "clean": True}) == []
       and len(package_problems(_quiet(), {"sha256": {"PLAN.md": "e"}, "clean": True})) == len(_quiet())
       and package_problems(_quiet(), {"sha256": dict(PKG_STUB), "clean": False}) != [])
    partial = dict(_synthetic("mbp", "isolated", ["generation"]), complete=False)
    r = cls(_quiet() + [partial])
    ok("classify.partial_counts", r["sub"] == "O6-pending" and r["observations"]["mbp"]["isolated"]["complete"])

    # 3b. the official-output guard, the --out guard and the host pins
    ex = [{"host": "mbp", "experiment": "inproc", "complete": True, "_file": "x.json"}]
    ok("guard.complete_refused", official_guard(ex, "mbp", "inproc", None) != [])
    ok("guard.other_host_free", official_guard(ex, "mini", "inproc", None) == [])
    inc = [{"host": "mbp", "experiment": "g1ctx", "complete": False}]
    ok("guard.one_repeat", official_guard(inc, "mbp", "g1ctx", None) == []
       and official_guard(inc * 2, "mbp", "g1ctx", None) != []
       and official_guard(inc * 2, "mbp", "g1ctx", "2026-10-08T00:00:00Z a") == [])
    try:
        check_out_dir(KIT_DIR / "results")
        ok("guard.out_in_repo_refused", False)
    except Refused:
        ok("guard.out_in_repo_refused", True)
    ok("guard.out_official_and_outside", check_out_dir(OFFICIAL_OUT) == OFFICIAL_OUT.resolve()
       and check_out_dir(work / "elsewhere") == (work / "elsewhere").resolve())
    mbp_info = {"macos": "27.0", "kern_osversion": "26A428", "architecture": "applegpu_g17s", "hw_model": "MacX,1"}
    mini_info = {"macos": "26.5.1", "kern_osversion": "25F80", "architecture": "applegpu_g16s", "hw_model": "Mac16,11"}
    ok("pins.mbp", host_pin_problems("mbp", mbp_info) == [] and host_pin_problems("mbp", mini_info) != [])
    ok("pins.mini", host_pin_problems("mini", mini_info) == [] and host_pin_problems("mini", mbp_info) != [])
    ok("attempts_cap", attempts_cap(50) == 60 and attempts_cap(4) == 7 and attempts_cap(1) == 4)

    # 4. (iv) on the stub world: the batch-flip calls leave a stale id; the probe finds what the runner reads
    W = StubWorld()
    cs = cachestate_probe(W, lambda: StubScenario("deterministic", world=W), work / "cachestate")
    ok("cachestate.stale_key", cs["stale_keys_added"] == 1 and cs["stale_key_is_tokenizer"] and cs["tokenizer_freed"])
    ok("cachestate.layout", cs["layout"]["tokenizer_basicsize"] == cs["layout"]["stub_basicsize"])
    ok("cachestate.control_equal", cs["control"]["pair"]["kind"] == "equal")
    ok("cachestate.seeded", cs["seeded"]["scorer_split"] and cs["seeded"]["pair"]["kind"] == "field_only"
       and pair_fields(cs["seeded"]["pair"]) == ["split_by"]
       and sorted(cs["seeded"]["pair"]["items"]) == ["q003", "q004", "q005"], cs["seeded"])
    ok("cachestate.outcome", cs["outcome"] in ("hit", "stale_only")
       and (cs["outcome"] == "hit") == bool(cs["hit"].get("found")), {k: cs[k] for k in ("outcome", "hit")})
    ok("cachestate.cache_restored", len(W.reasoning._CHECKED) == 1)
    none_case = dict(cs, stale_keys_added=0)
    ok("cachestate.outcome_none", cachestate_outcome(none_case) == "none"
       and cachestate_outcome(dict(cs, control={"pair": {"kind": "field_only"}})) == "none")
    forced = dict(cs, hit={"found": True, "scorer_split": True, "pair": cs["seeded"]["pair"]})
    ok("cachestate.outcome_hit", cachestate_outcome(forced) == "hit"
       and cachestate_outcome(dict(cs, hit={"found": False})) == "stale_only")

    # 5. the pytest harness of the pytest arms on synthetic stub tests (no kit, no conftest, no model)
    stub_dir = work / "pytest_stub"
    stub_dir.mkdir(parents=True, exist_ok=True)
    (stub_dir / "test_runner_tiny.py").write_text(STUB_TEST, encoding="utf-8")
    (stub_dir / "test_bench_batch_flip.py").write_text(STUB_FIRST, encoding="utf-8")
    want_line = next(i for i, ln in enumerate(STUB_TEST.splitlines(), 1) if "assert content(_records(p))" in ln)
    env = g1_env(os.environ)
    env.update(TMPDIR=str(work / "tmp"), PYTHONDONTWRITEBYTECODE="1")
    (work / "tmp").mkdir(exist_ok=True)
    bt = work / "basetemp"
    bt.mkdir(exist_ok=True)
    test_file = stub_dir / "test_runner_tiny.py"
    makers = {
        "isolated": lambda j, b: pytest_isolated_cmd(sys.executable, j, b, test_file, TEST_NAME),
        "module": lambda j, b: pytest_selection_cmd(sys.executable, j, b, [test_file]),
        "pair": lambda j, b: pytest_selection_cmd(sys.executable, j, b, [stub_dir / "test_bench_batch_flip.py",
                                                                         test_file]),
        "g1ctx": lambda j, b: pytest_g1ctx_cmd(sys.executable, j, b, stub_dir),
    }
    got = {}
    for arm, mk in makers.items():
        (work / f"pt_{arm}").mkdir(exist_ok=True)
        rs, reached = run_pytest_series(arm, 1, work / f"pt_{arm}", bt, stub_dir, env, mk, 120, lambda r: None,
                                        target_line=want_line)
        got[arm] = rs[0]
        r0 = rs[0]
        ok(f"pytest.{arm}.failed_at_line", reached and len(rs) == 1 and r0["valid"]
           and r0["target"]["outcome"] == "failed" and r0["target"]["line"] == want_line, r0.get("target"))
        ok(f"pytest.{arm}.records", r0["kind"] == "generation" and r0["record_files_found"]
           and sorted(r0["pair"]["items"]) == ["q003"]
           and r0["pair"]["items"]["q003"]["first_divergence"] == {"index": 2, "a": 5, "b": 6}
           and (Path(r0["records_copy"]) / "resumed.jsonl").is_file(), r0.get("pair"))
        ok(f"pytest.{arm}.untruncated", "Full output truncated" not in (r0["target"]["text"] or "")
           and "use -vv to show" not in (r0["target"]["text"] or ""))
    ok("pytest.g1ctx.other_failure", got["g1ctx"]["counts"]["tests"] == 3
       and [o["test"] for o in got["g1ctx"]["other_failures"]] == ["test_runner_tiny::test_an_unrelated_failure"],
       got["g1ctx"].get("other_failures"))
    ok("pytest.pair.order", got["pair"]["counts"]["tests"] == 3)
    ok("pytest.summary", summarise_pytest(list(got.values()))["reproductions"] == 4)
    err_dir = work / "pytest_stub_setup_error"
    err_dir.mkdir(parents=True, exist_ok=True)
    (err_dir / "test_runner_tiny.py").write_text(STUB_SETUP_ERROR, encoding="utf-8")
    (work / "pt_err").mkdir(exist_ok=True)
    rs, reached = run_pytest_series("err", 1, work / "pt_err", bt, err_dir, env,
                                    lambda j, b: pytest_isolated_cmd(sys.executable, j, b,
                                                                     err_dir / "test_runner_tiny.py", TEST_NAME),
                                    120, lambda r: None)
    ok("pytest.setup_error_invalid_and_replaced", not reached and len(rs) == attempts_cap(1)
       and all(r["kind"] == "invalid" and r["invalid_reason"] == "setup_error" for r in rs), rs[0])
    c_dir = work / "pytest_stub_c_assert"
    c_dir.mkdir(parents=True, exist_ok=True)
    (c_dir / "test_runner_tiny.py").write_text(STUB_C_ASSERT, encoding="utf-8")
    c_line = next(i for i, ln in enumerate(STUB_C_ASSERT.splitlines(), 1) if "assert len(done)" in ln)
    (work / "pt_c").mkdir(exist_ok=True)
    rs, _ = run_pytest_series("c", 1, work / "pt_c", bt, c_dir, env,
                              lambda j, b: pytest_isolated_cmd(sys.executable, j, b, c_dir / "test_runner_tiny.py",
                                                               TEST_NAME),
                              120, lambda r: None, target_line=999, c_lines=(c_line,))
    ok("pytest.c_assert", rs[0]["kind"] == "c_assert" and rs[0]["target"]["line"] == c_line, rs[0].get("target"))

    # 6. L2's driver: this script in separate processes, on the stub
    fruns, freached = fresh_series("mbp", 3, work / "fresh", "resume", lambda r: None)
    ok("fresh.processes", freached and len(fruns) == 3 and all(r.get("valid", True) for r in fruns)
       and [r["ac_kind"] for r in fruns] == ["generation"] * 3, [r.get("stderr_tail") for r in fruns])
    fout = dict(_synthetic("mbp", "fresh", []), runs=fruns, stub=True)
    r = classify(base6 + [fout], allow_stub=True)
    ok("fresh.L2.O2", r["sub"] == "O6-O2", r.get("matched"))

    # 6b. compaction (outputs over the size budget) keeps everything the table reads
    runs_n, _ = inproc_iterations(StubScenario("noisy", seed=5), 14, work / "inproc_compact")
    comp = compact_runs(runs_n)
    ok("compact.kinds_kept", [(r["uninterrupted_kind"], r["ac_kind"]) for r in comp]
       == [(r["uninterrupted_kind"], r["ac_kind"]) for r in runs_n])
    full_o = dict(_synthetic("mbp", "inproc", []), runs=runs_n)
    comp_o = dict(_synthetic("mbp", "inproc", []), runs=comp)
    ok("compact.observation_kept", observe([full_o], "mbp")["inproc"] == observe([comp_o], "mbp")["inproc"]
       and any(r["AB"].get("compacted") or r["AC"].get("compacted") for r in comp))

    # 7. redaction and output writing
    red = Redactor(extra=((str(work), "<work>"),))
    leaky = json.dumps({"p": str(Path.home() / "models" / "x"), "w": str(work / "a"), "u": getpass.getuser() + "-x"})
    clean = red(leaky)
    ok("redact.clean", not red.leaks(clean) and "<work>/a" in clean and "~/models/x" in clean, clean)
    target = work / "out_selftest" / "mbp_inproc_20261007T000000Z.json"
    write_output({"schema": SCHEMA, "where": str(work / "z")}, target, red)
    ok("write.roundtrip", json.loads(target.read_text())["where"] == "<work>/z")
    try:
        planted = "/".join(("", "Users", "someone", "y"))  # built at run time: the leak check scans this file
        write_output({"x": planted}, work / "out_selftest" / "leak.json", red)
        ok("write.refuses_leak", False)
    except Refused:
        ok("write.refuses_leak", True)

    # 8. nothing from the kit, no MLX, no port
    kit_mods = sorted(name for name, m in list(sys.modules.items())
                      if getattr(m, "__file__", None) and is_under(Path(m.__file__), KIT_DIR)
                      and Path(m.__file__).resolve() != Path(__file__).resolve())
    ok("imports.no_kit", not kit_mods, kit_mods)
    ok("imports.no_mlx", not any(n == "mlx" or n.startswith(("mlx.", "mlx_lm")) for n in sys.modules))
    return {"selftest": "ok", "checks": len(done), "names": done, "stub_outcomes": results,
            "cachestate_stub_outcome": cs["outcome"]}


# ----------------------------------------------------------------------------------------------------- main


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--host", choices=HOSTS)
    ap.add_argument("--experiment", choices=EXPERIMENTS)
    ap.add_argument("--n", type=int, help="runs or iterations (default: PLAN.md's registered N)")
    ap.add_argument("--out", type=Path, default=OFFICIAL_OUT, help="output directory (default: this folder's out/; "
                                                                   "otherwise outside the repository)")
    ap.add_argument("--work", type=Path, help="work directory outside the repository (default: "
                                              "$EXP036_WORK/exp037/diag_g1_resume_20261007; on the mini "
                                              "~/models/exp037-mini/diag_g1_resume_20261007)")
    ap.add_argument("--no-margin", action="store_true", help="inproc: skip the descriptive margins")
    ap.add_argument("--stub", choices=STUB_KINDS, help="selftest only: a stub model for inproc / fresh / cachestate")
    ap.add_argument("--repeat-approved", metavar="'<UTC> <label>'",
                    help="a further repeat after two incomplete outputs: Andrei's recorded decision (PLAN.md 3.0)")
    ap.add_argument("--classify", action="store_true", help="apply PLAN.md's table to the outputs in --out")
    ap.add_argument("--write", action="store_true", help="with --classify: write out/classification_<UTC>.json")
    ap.add_argument("--selftest", action="store_true", help="stub models only; writes under --work")
    args = ap.parse_args(argv)
    try:
        if args.selftest:
            w = Path(args.work).expanduser().resolve() if args.work else Path(tempfile.mkdtemp(prefix="g1diag_selftest_"))
            if is_under(w, REPO_DIR):
                raise Refused("--work must be outside the repository")
            print(json.dumps(selftest(w / f"selftest_{utc_stamp()}"), indent=1))
            return 0
        if args.classify:
            return run_classify(args)
        return run_experiment(args)
    except Refused as e:
        print(json.dumps({"ok": False, "refused": str(e)}))
        return 3


if __name__ == "__main__":
    sys.exit(main())
