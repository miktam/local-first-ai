"""Re-create or check tasks/vendored_evalfw/ from an eval-framework v0.14.2 checkout (BUILD_SPEC §5.5).

    python tasks/vendored_evalfw/vendor.py --upstream DIR            # write the vendored tree + MANIFEST.json
    python tasks/vendored_evalfw/vendor.py --upstream DIR --check    # verify only (exit 1 on any difference)

DIR is a local clone of github.com/Aleph-Alpha-Research/eval-framework at tag v0.14.2 (commit 3532e9f1).
Every file in VENDORED_FILES is copied byte for byte, keeping its upstream relative path, except
src/eval_framework/benchmarks/gpqa.py, which gets exactly one modification (Apache-2.0 §4(b)):

- the `_OVERLONG_QUESTION = (...)` string literal, the full text of one GPQA question, is replaced by
  `_OVERLONG_QUESTION_SHA256 = "<sha256 of that string, UTF-8>"`;
- the dataset filter compares `sha256(row["Question"])` with that constant instead of the text
  (the same predicate: upstream compared the raw `row["Question"]` for equality);
- three header lines are added: the SPDX identifier, the upstream copyright and the
  "Modified by Miktam for Chronos exp_036" line.

The patch is applied in memory; the question text is never written anywhere. No network access.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

UPSTREAM_REPO = "https://github.com/Aleph-Alpha-Research/eval-framework"
UPSTREAM_TAG = "v0.14.2"
UPSTREAM_COMMIT = "3532e9f16e19e396fc2b4d4543878cdccc4d04da"

# Upstream relative paths. The benchmark files named in BUILD_SPEC §5.5 plus the helpers they need at
# import time or call (choices.ChoiceFields, eval_kind.assemble_messages, task_style.shuffle_correct_
# with_distractors, utils.get_n_letters, minerva_math_utils' _fix_* helpers used by
# _strip_string_with_bug). eval-framework v0.14.2 ships no NOTICE file; its LICENSE is copied.
VENDORED_FILES = [
    "LICENSE",
    "src/eval_framework/answer.py",
    "src/eval_framework/choices.py",
    "src/eval_framework/eval_kind.py",
    "src/eval_framework/benchmarks/cot.py",
    "src/eval_framework/benchmarks/gpqa.py",
    "src/eval_framework/benchmarks/gpqa_ellamind.py",
    "src/eval_framework/benchmarks/mmlu_pro.py",
    "src/eval_framework/benchmarks/math_reasoning.py",
    "src/eval_framework/tasks/task_style.py",
    "src/eval_framework/tasks/utils.py",
    "src/eval_framework/metrics/completion/minerva_math_utils.py",
]
PATCHED_FILE = "src/eval_framework/benchmarks/gpqa.py"

_UPSTREAM_FILTER = 'lambda row: row["Question"] != _OVERLONG_QUESTION, description="excluding one over-long question"'
_PATCHED_FILTER = (
    'lambda row: hashlib.sha256(row["Question"].encode()).hexdigest() != _OVERLONG_QUESTION_SHA256,\n'
    '            description="excluding one over-long question",'
)
_HEADER = (
    "# SPDX-License-Identifier: Apache-2.0\n"
    "# Copyright 2025 Aleph Alpha Research GmbH (eval-framework v0.14.2, src/eval_framework/benchmarks/gpqa.py)\n"
    "# Modified by Miktam for Chronos exp_036, 2026-10-03: the _OVERLONG_QUESTION text is replaced by its\n"
    "# sha256 (_OVERLONG_QUESTION_SHA256) and the filter compares hashes; nothing else is changed.\n"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def overlong_literal(upstream_gpqa_source: str) -> tuple[str, int, int]:
    """(value, first line, last line) of the upstream `_OVERLONG_QUESTION = (...)` assignment."""
    tree = ast.parse(upstream_gpqa_source)
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "_OVERLONG_QUESTION"
        ):
            value = ast.literal_eval(node.value)
            if not isinstance(value, str):
                raise ValueError("_OVERLONG_QUESTION is not a string literal")
            return value, node.lineno, node.end_lineno
    raise ValueError("no _OVERLONG_QUESTION assignment in upstream gpqa.py")


def patch_gpqa(upstream_source: str) -> tuple[str, str]:
    """Return (patched source, sha256 of the overlong question). Raises if upstream is not as expected."""
    value, first, last = overlong_literal(upstream_source)
    digest = sha256_bytes(value.encode("utf-8"))
    lines = upstream_source.split("\n")
    # lines[first-1 .. last-1] hold the assignment; replace them by one constant.
    lines[first - 1 : last] = [f'_OVERLONG_QUESTION_SHA256 = "{digest}"']
    patched = "\n".join(lines)
    if patched.count(_UPSTREAM_FILTER) != 1:
        raise ValueError("upstream gpqa.py filter line not found exactly once")
    patched = patched.replace(_UPSTREAM_FILTER, _PATCHED_FILTER)
    if re.search(r"_OVERLONG_QUESTION(?!_SHA256)", patched):
        raise ValueError("a reference to the removed _OVERLONG_QUESTION literal remains")
    return _HEADER + patched, digest


def build(upstream: Path) -> tuple[dict[str, bytes], dict]:
    """Return ({relpath: bytes to vendor}, manifest dict) from an upstream checkout, without writing."""
    contents: dict[str, bytes] = {}
    files: dict[str, dict] = {}
    overlong_sha = None
    for rel in VENDORED_FILES:
        raw = (upstream / rel).read_bytes()
        entry = {"upstream_sha256": sha256_bytes(raw), "modified": False}
        if rel == PATCHED_FILE:
            patched, overlong_sha = patch_gpqa(raw.decode("utf-8"))
            data = patched.encode("utf-8")
            entry["modified"] = True
            entry["modification"] = (
                "_OVERLONG_QUESTION string literal replaced by _OVERLONG_QUESTION_SHA256 (sha256 of the "
                "string, UTF-8); the filter compares sha256(row['Question']) with it; SPDX, copyright and "
                "'Modified by Miktam for Chronos exp_036' header lines added"
            )
        else:
            data = raw
        entry["sha256"] = sha256_bytes(data)
        contents[rel] = data
        files[rel] = entry
    manifest = {
        "schema": "exp036 vendored eval-framework manifest v1",
        "source": {"repo": UPSTREAM_REPO, "tag": UPSTREAM_TAG, "commit": UPSTREAM_COMMIT, "licence": "Apache-2.0"},
        "overlong_question_sha256": overlong_sha,
        "files": files,
    }
    return contents, manifest


def manifest_bytes(manifest: dict) -> bytes:
    return (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")


def check_local(root: Path = HERE) -> list[str]:
    """Problems found comparing the vendored files with MANIFEST.json (no upstream needed)."""
    problems = []
    manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
    for rel, entry in sorted(manifest["files"].items()):
        p = root / rel
        if not p.is_file():
            problems.append(f"missing: {rel}")
            continue
        got = sha256_bytes(p.read_bytes())
        if got != entry["sha256"]:
            problems.append(f"sha256 mismatch: {rel} ({got} != {entry['sha256']})")
        if not entry["modified"] and entry["sha256"] != entry["upstream_sha256"]:
            problems.append(f"unmodified file whose sha256 differs from upstream: {rel}")
    listed = set(manifest["files"])
    if listed != set(VENDORED_FILES):
        problems.append(f"MANIFEST.json lists {sorted(listed ^ set(VENDORED_FILES))} differently from VENDORED_FILES")
    return problems


def check_against_upstream(upstream: Path, root: Path = HERE) -> list[str]:
    """Problems found re-deriving the vendored tree from an upstream checkout."""
    contents, manifest = build(upstream)
    problems = check_local(root)
    for rel, data in contents.items():
        p = root / rel
        if p.is_file() and p.read_bytes() != data:
            problems.append(f"differs from the re-derived upstream copy: {rel}")
    if (root / "MANIFEST.json").read_bytes() != manifest_bytes(manifest):
        problems.append("MANIFEST.json differs from the re-derived manifest")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--upstream", type=Path, required=True, help="eval-framework checkout at v0.14.2")
    ap.add_argument("--check", action="store_true", help="verify only; write nothing")
    args = ap.parse_args(argv)
    upstream = args.upstream.expanduser()
    if args.check:
        problems = check_against_upstream(upstream)
        for p in problems:
            print(p)
        print(json.dumps({"ok": not problems, "problems": len(problems)}))
        return 1 if problems else 0
    contents, manifest = build(upstream)
    for rel, data in contents.items():
        out = HERE / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
    (HERE / "MANIFEST.json").write_bytes(manifest_bytes(manifest))
    print(json.dumps({"ok": True, "files": len(contents), "overlong_question_sha256": manifest["overlong_question_sha256"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
