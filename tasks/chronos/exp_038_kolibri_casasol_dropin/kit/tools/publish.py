"""exp_038: assemble the public directory and leak-check it (DESIGN §11). Code only; never pushes.

  python3 tools/publish.py stage <public_dir>   copy the public kit into <public_dir>, write PRIVATE_SHA256.json
  python3 tools/publish.py check <public_dir>   leak check: 8-word shingles of every private text file (glue,
                                                 prompts, strings, sets, golds, raw outputs, judge material, rights)
                                                 must not occur in any public file; exits 1 on any hit

Public: the MLX backend and its tests, the harness skeleton (run_arm, R1, R2, stage 1, analysis, judge kit), the
tools, HYPOTHESIS.md. Private (sha256 only): vendor/ (the glue), harness/de_path.py and its tests (German strings and
regexes), private/ (sets, golds, writer inputs, strings, rubric), runs/ (raw outputs), rights/, blindness/ notes.
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

P38 = Path(__file__).resolve().parents[1]
PUBLIC = ["kit/mlx_backend.py", "kit/soak.py", "kit/tests/conftest.py", "kit/tests/test_mlx_backend.py",
          "harness/run_arm.py", "harness/r1.py", "harness/r2.py", "harness/stage1.py", "harness/analysis.py",
          "harness/judge_kit.py", "harness/with_side_server.sh", "harness/run_r1.sh", "harness/tests/test_analysis.py",
          "harness/run_analysis.py", "harness/r3.py", "harness/r7.py", "harness/mlx_entry.py", "harness/pilot_cap.py",
          "tools/export_results.py",
          "tools/audit_agents.py", "tools/de_gates.py", "tools/e38_d1.py", "tools/m0_shingle_check.py",
          "tools/make_writer_inputs.py", "tools/r4_freeze.py", "tools/redact_rubric.py", "tools/seal.py",
          "tools/slice_steps.py", "tools/publish.py"]
PRIVATE_ROOTS = ["vendor", "private", "rights", "runs", "blindness", "aborted", "harness/de_path.py", "harness/tests/test_de_path.py"]
TEXT = {".py", ".md", ".json", ".jsonl", ".txt", ".sh"}
N = 8
# The registered analysis output is public by design (its strings are the verdict and state labels); hashed, not forbidden
PUBLIC_DERIVED = ["runs/results.json"]
# Score and check field names that the public exports repeat; windows made only of these, booleans, numbers and file-path
# components are structure (field lists, file lists), not private content
STRUCT_WORDS = {"true", "false", "null", "none", "correctness", "usefulness", "language_ok", "language_native",
                "tail_degradation", "from_weights", "n_unsupported_claims", "n_fabricated_citations_kept",
                "unsupported_claims", "fabricated_citations_kept"}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def private_files() -> list[Path]:
    out = []
    for r in PRIVATE_ROOTS:
        p = P38 / r
        out += [p] if p.is_file() else sorted(x for x in p.rglob("*") if x.is_file() and "__pycache__" not in x.parts)
    return out


def words(t: str) -> list[str]:
    return re.findall(r"\w+", t.lower())


def shingles(t: str, struct: frozenset = frozenset()) -> set[str]:
    """8-word shingles; with `struct`, a window made only of structure words and numbers is skipped."""
    w = words(t)
    return {hashlib.sha1(" ".join(w[i:i + N]).encode()).hexdigest() for i in range(len(w) - N + 1)
            if not struct or not all(x in struct or x.isdigit() for x in w[i:i + N])}


def struct_words() -> frozenset:
    """STRUCT_WORDS plus every component of every P38 file path (file lists are public by registration)."""
    comps = {x for p in P38.rglob("*") if p.is_file() and ".git" not in p.parts for x in words(str(p.relative_to(P38)))}
    return frozenset(STRUCT_WORDS | comps | {"private", "json", "jsonl", "md", "py"})


def stage(public: Path) -> None:
    public.mkdir(parents=True, exist_ok=True)
    for rel in PUBLIC:
        dst = public / "kit" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(P38 / rel, dst)
    hyp = P38 / "public_draft/HYPOTHESIS.md"
    shutil.copyfile(hyp, public / "HYPOTHESIS.md")
    priv = {str(p.relative_to(P38)): sha(p) for p in private_files()}
    (public / "PRIVATE_SHA256.json").write_text(json.dumps({"note": "sha256 of every private exp_038 file (P38, local only)",
                                                            "files": priv}, indent=1) + "\n")
    print(json.dumps({"public_files": len(PUBLIC) + 2, "private_hashed": len(priv)}))


def strings(x) -> list[str]:
    """Every string value in a parsed JSON document (keys and numbers are structure, not private content)."""
    if isinstance(x, dict):
        return [s for v in x.values() for s in strings(v)]
    if isinstance(x, list):
        return [s for v in x for s in strings(v)]
    return [x] if isinstance(x, str) else []


def private_shingles(p: Path, struct: frozenset = frozenset()) -> set[str]:
    """JSON and JSONL: the shingles of each string value on its own; other text files: the whole text."""
    t = p.read_text(encoding="utf-8", errors="replace")
    try:
        docs = [json.loads(t)] if p.suffix == ".json" else [json.loads(x) for x in t.splitlines() if x.strip()] \
            if p.suffix == ".jsonl" else None
    except json.JSONDecodeError:
        docs = None
    if docs is None:
        return shingles(t, struct)
    return set().union(*(shingles(s, struct) for d in docs for s in strings(d)))


def check(public: Path) -> int:
    forbidden = set()
    # public code legitimately shares lines with private copies of itself; only private *content* counts
    shared_code = {(P38 / r).read_text(encoding="utf-8", errors="replace") for r in PUBLIC}
    shared = set().union(*(shingles(t) for t in shared_code)) if shared_code else set()
    struct = struct_words()
    for p in private_files():
        if p.suffix in TEXT and str(p.relative_to(P38)) not in PUBLIC_DERIVED:
            forbidden |= private_shingles(p, struct)
    forbidden -= shared
    hits = {}
    for p in sorted(x for x in public.rglob("*") if x.is_file()):
        if p.suffix not in TEXT or p.name == "PRIVATE_SHA256.json":
            continue
        n = len(shingles(p.read_text(encoding="utf-8", errors="replace")) & forbidden)
        if n:
            hits[str(p.relative_to(public))] = n
    print(json.dumps({"forbidden_shingles": len(forbidden), "hits": hits}))
    return 1 if hits else 0


if __name__ == "__main__":
    if sys.argv[1] == "stage":
        stage(Path(sys.argv[2]))
    else:
        sys.exit(check(Path(sys.argv[2])))
