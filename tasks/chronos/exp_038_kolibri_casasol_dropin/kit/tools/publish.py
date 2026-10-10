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
          "tools/audit_agents.py", "tools/de_gates.py", "tools/e38_d1.py", "tools/m0_shingle_check.py",
          "tools/make_writer_inputs.py", "tools/r4_freeze.py", "tools/redact_rubric.py", "tools/seal.py",
          "tools/slice_steps.py", "tools/publish.py"]
PRIVATE_ROOTS = ["vendor", "private", "rights", "runs", "harness/de_path.py", "harness/tests/test_de_path.py"]
TEXT = {".py", ".md", ".json", ".jsonl", ".txt", ".sh"}
N = 8


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


def shingles(t: str) -> set[str]:
    w = words(t)
    return {hashlib.sha1(" ".join(w[i:i + N]).encode()).hexdigest() for i in range(len(w) - N + 1)}


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


def check(public: Path) -> int:
    forbidden = set()
    # public code legitimately shares lines with private copies of itself; only private *content* counts
    shared_code = {(P38 / r).read_text(encoding="utf-8", errors="replace") for r in PUBLIC}
    shared = set().union(*(shingles(t) for t in shared_code)) if shared_code else set()
    for p in private_files():
        if p.suffix in TEXT:
            forbidden |= shingles(p.read_text(encoding="utf-8", errors="replace"))
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
