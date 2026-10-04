#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""The exp_036 dry run: the whole pipeline end to end on tiny checkpoints (RUNBOOK step 3b; BUILD_LOG.md).

    "$PY" tools/dry_run.py [--tok DIR] [--peers DIR] [--data DIR] [--workdir DIR] [--keep]

Weight-free and offline. Everything happens in a temporary directory that is deleted at the end (unless --keep):

  repo/        a git repository holding a copy of this kit at tasks/chronos/exp_036_kolibri_local_eval (plus the
               exp_007 padding fixtures), with the identity Miktam <hello@localfirstai.eu>, a bare remote
               (remote.git) so HEAD == @{u} can hold, and the kit's pre-push hook installed
  models/      $EXP036_MODELS: a tiny Kolibri BF16 checkpoint (the gate's pattern5 layout, the real vocabulary
               of 128,000 and the real Kolibri tokenizer), its K8/K4 conversions by port/convert.py, and tiny
               stand-ins for the six peers: Kolibri-architecture weights at each peer's vocabulary size with
               that peer's real tokenizer, template and generation_config, converted to 8 and 4 bits
  data/        $EXP036_DATA: synthetic GPQA / MMLU-ProX / AIME / RGB in the real on-disk layout (tests/
               tasks_synthetic.py; no real item text), revision metadata at the pinned revisions; the public
               IFBench_test, IFBench-src and FineWeb-2 shard from --data when present (else synthetic)
  private/, work/   $EXP036_PRIVATE, $EXP036_WORK

Stages (RUNBOOK order; each must pass):
  preregister   hash_tree --fill and --stamp, the pre-registration commit and push (pre-push hook)
  preflight     tools/preflight.py --quick --not-run-host --out ...      (the mini's mode)
  signoff       a placeholder sign-off line (this temporary copy only), status --record-block, commit, push
  manifests     tasks/build_manifests.build_all on the synthetic data (counts not enforced: the sets are small,
                and the GPQA over-long question cannot be synthesised); gate/build_gate_text.py --work-only
  convert       port/convert.py for K8 and K4, exactly as RUNBOOK step 8
  peers         tools/peer_check.py --checks load,nll,kl,batch on the stand-ins (the batched path through the
                gate's G5 functions), then a dry-run peers record that marks every stand-in ok
  gate          gate/run_gate.py --tiny on $EXP036_MODELS; its PASS record is re-labelled as a real-mode record
                (dry_run_promoted_from) so the runner's guards accept it, bound to the same shas
  tier2         the Tier-2 amendment as an amendments/ file, status --sync-amendments, commit, push
  bench         every bench cell through its Python entry point at seconds scale (H1, descriptive speed, D1, H8,
                H5, C1, E6 ladder)
  pilot         runner/run.py pilot
  plan          runner/plan_fix.py, the plan amendment appended, commit, push
  sessions      runner/run.py session --name S2, then S3 (and S3b if it asks), status, commits and pushes
  score         scorers/score_all.py (mbp), tools/version_record.py; then the mini's steps:
                score_all.py --ifbench (when an IFBench venv exists), score_all.py --rescore-compare
  verdicts      analysis/verdicts.py
  hash_check    tools/hash_tree.py --check HYPOTHESIS.md
  leak          tools/leak_check.py --all in the copy (with the withheld shingles), and over this kit's working
                tree with --no-gpqa-source

The runner's frozen plan_rules.json is scaled down in the copy only (cell sizes, caps, the B4 estimate), so the
whole queue runs in minutes; every line of code that runs is the kit's own. The last stdout line is JSON
({"ok": ..., "stages": [...]}); exit 0 when every stage passed, 1 otherwise.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

EXP_DIR = Path(__file__).resolve().parents[1]
if str(EXP_DIR) not in sys.path:  # so main() can import tools.precision when run as a script
    sys.path.insert(0, str(EXP_DIR))
REL_EXP = "tasks/chronos/exp_036_kolibri_local_eval"
REL_PADDING = "tasks/chronos/exp_007_hardware_comparison/fixtures/padding"
IDENTITY = ("Miktam", "hello@localfirstai.eu")
COPY_SKIP_DIRS = {"__pycache__", ".pytest_cache", "results", "aborted", "evidence", "amendments"}
COPY_SKIP_FILES = {".DS_Store", "exp036.local.env"}

# Peer stand-ins: (family, the folder whose tokenizer files are borrowed, the arms built from it)
PEER_FAMILIES = {
    "gemma4": ("gemma-4-26b-a4b-it-8bit", {"G8": "gemma-4-26b-a4b-it-8bit", "G4": "gemma-4-26b-a4b-it-4bit"}),
    "qwen3_6": ("Qwen3.6-35B-A3B-8bit", {"Q36-8": "Qwen3.6-35B-A3B-8bit", "Q36-4": "Qwen3.6-35B-A3B-4bit"}),
    "qwen3_8": ("Qwen3.8-27B-8bit", {"Q38-8": "Qwen3.8-27B-8bit", "Q38-4": "Qwen3.8-27B-4bit"}),
}
PEER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json")
KOLIBRI_FILES = ("tokenizer.json", "tokenizer_config.json", "generation_config.json")
SIGNOFF_LINE = ("- Signed off by: Andrei (DRY RUN placeholder written by tools/dry_run.py into a temporary copy; "
                "not a sign-off)")


# ================================================================================ helpers


def utc_stamp(dt: datetime | None = None) -> str:
    return (dt or datetime.now(timezone.utc)).strftime("%Y%m%dT%H%M%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def first_env(*names: str) -> str | None:
    for n in names:
        if os.environ.get(n):
            return os.path.expanduser(os.environ[n])
    return None


def default_tok() -> Path | None:
    cands = [first_env("EXP036_TOK", "EXP036_TOKENIZER_DIR")]
    if os.environ.get("EXP036_MODELS"):
        cands.append(str(Path(os.environ["EXP036_MODELS"]).expanduser() / "Kolibri-1-BF16"))
    cands.append(str(Path("~/models/exp036-mini/kolibri/Kolibri-1-BF16").expanduser()))
    for c in cands:
        if c and (Path(c) / "tokenizer.json").is_file():
            return Path(c)
    return None


def default_peers() -> Path | None:
    cands = []
    if os.environ.get("EXP036_MODELS"):
        cands.append(Path(os.environ["EXP036_MODELS"]).expanduser())
    cands.append(Path("~/models/exp036-mini/mlx-community").expanduser())
    for c in cands:
        if all((c / src / "tokenizer.json").is_file() for src, _ in PEER_FAMILIES.values()):
            return c
    return None


def default_data() -> Path | None:
    cands = [first_env("EXP036_DATA")]
    if os.environ.get("EXP036_MODELS"):
        cands.append(str(Path(os.environ["EXP036_MODELS"]).expanduser() / "data"))
    cands.append(str(Path("~/models/exp036-mini/data").expanduser()))
    for c in cands:
        if c and Path(c).is_dir():
            return Path(c)
    return None


def default_ifbench() -> tuple[Path | None, Path | None]:
    py = first_env("EXP036_IFBENCH_PY")
    nltk = first_env("EXP036_NLTK_DATA")
    roots = []
    if os.environ.get("EXP036_MODELS"):
        roots.append(Path(os.environ["EXP036_MODELS"]).expanduser())
    roots.append(Path("~/models/exp036-mini").expanduser())
    if py is None:
        py = next((str(r / "ifbench-venv" / "bin" / "python") for r in roots
                   if (r / "ifbench-venv" / "bin" / "python").exists()), None)
    if nltk is None:
        nltk = next((str(r / "nltk_data") for r in roots if (r / "nltk_data").is_dir()), None)
    return (Path(py) if py else None), (Path(nltk) if nltk else None)


class World:
    """Paths of one dry run."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.repo = self.root / "repo"
        self.remote = self.root / "remote.git"
        self.exp = self.repo / REL_EXP
        self.models = self.root / "models"
        self.data = self.root / "data"
        self.private = self.root / "private"
        self.work = self.root / "work"
        self.src = self.root / "src"
        self.logs = self.root / "logs"
        self.side = self.root / "side"
        self.state = self.root / "state.json"

    def env(self, extra: dict | None = None) -> dict:
        e = {k: v for k, v in os.environ.items() if not k.startswith("EXP036_")}
        e.update(EXP036_MODELS=str(self.models), EXP036_DATA=str(self.data), EXP036_PRIVATE=str(self.private),
                 EXP036_WORK=str(self.work), EXP036_TOK=str(self.models / "Kolibri-1-BF16"),
                 HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONDONTWRITEBYTECODE="1", PY=sys.executable,
                 GIT_TERMINAL_PROMPT="0")
        st = self.load_state()
        if st.get("ifbench_py"):
            e["EXP036_IFBENCH_PY"] = st["ifbench_py"]
        if st.get("nltk_data"):
            e["EXP036_NLTK_DATA"] = st["nltk_data"]
        e.update(extra or {})
        return e

    def load_state(self) -> dict:
        return json.loads(self.state.read_text()) if self.state.is_file() else {}

    def save_state(self, **kw) -> None:
        st = self.load_state()
        st.update(kw)
        self.state.write_text(json.dumps(st, indent=1, sort_keys=True))


class StageError(RuntimeError):
    pass


class Runner:
    """Runs commands for a stage, logs their output, and keeps the stage record."""

    def __init__(self, w: World, verbose: bool):
        self.w, self.verbose = w, verbose
        self.stages: list[dict] = []
        self.n = 0

    def run(self, args: list, *, cwd: Path | None = None, ok_codes=(0,), env: dict | None = None,
            timeout: float = 3600, stdin: str | None = None) -> subprocess.CompletedProcess:
        self.n += 1
        cmd = [str(a) for a in args]
        log = self.w.logs / f"{self.n:03d}.log"
        t0 = time.time()
        p = subprocess.run(cmd, cwd=str(cwd or self.w.exp), env=env or self.w.env(), capture_output=True,
                           text=True, timeout=timeout, input=stdin)
        log.write_text(f"$ {' '.join(cmd)}\n# rc={p.returncode} {time.time() - t0:.1f}s\n--- stdout\n{p.stdout}"
                       f"\n--- stderr\n{p.stderr}\n", encoding="utf-8")
        if self.verbose:
            print(f"    [{p.returncode}] {' '.join(cmd[:4])} … ({time.time() - t0:.1f}s)", file=sys.stderr)
        if p.returncode not in ok_codes:
            tail = "\n".join((p.stdout + "\n" + p.stderr).strip().splitlines()[-25:])
            raise StageError(f"{' '.join(cmd[:6])} exited {p.returncode} (log {log.name}):\n{tail}")
        return p

    def py(self, script: str, *args, **kw) -> subprocess.CompletedProcess:
        return self.run([sys.executable, script, *args], **kw)

    def git(self, *args, **kw) -> subprocess.CompletedProcess:
        return self.run(["git", *args], cwd=self.w.repo, **kw)

    def stage(self, name: str, fn) -> bool:
        t0 = time.time()
        print(f"[dry-run] {name} …", file=sys.stderr, flush=True)
        rec = {"stage": name}
        try:
            note = fn()
            rec.update(ok=True, note=note)
        except Exception as e:  # a stage failure names the broken stage (deterministic glue, BUILD_SPEC §2)
            rec.update(ok=False, error=f"{type(e).__name__}: {e}"[:4000])
        rec["seconds"] = round(time.time() - t0, 1)
        self.stages.append(rec)
        print(f"[dry-run] {name}: {'ok' if rec['ok'] else 'FAILED'} ({rec['seconds']} s)"
              + (f"\n{rec['error']}" if not rec["ok"] else ""), file=sys.stderr, flush=True)
        return rec["ok"]

    def commit(self, message: str, add: str = REL_EXP) -> None:
        self.git("add", "-A", add)
        if self.git("status", "--porcelain").stdout.strip():
            self.git("commit", "-q", "-m", message, "-m", "dry run")
        self.git("push", "-q", "origin", "HEAD")


# ================================================================================ world


def copy_kit(w: World) -> None:
    for p in sorted(EXP_DIR.rglob("*")):
        rel = p.relative_to(EXP_DIR)
        if COPY_SKIP_DIRS & set(rel.parts) or p.name in COPY_SKIP_FILES or p.is_dir():
            continue
        dst = w.exp / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    pad = EXP_DIR.parents[2] / REL_PADDING
    if not pad.is_dir():
        raise StageError(f"exp_007 padding fixtures not found next to the kit ({REL_PADDING})")
    shutil.copytree(pad, w.repo / REL_PADDING)
    for d in ("results", "aborted", "evidence", "amendments"):
        (w.exp / d).mkdir(parents=True, exist_ok=True)
        (w.exp / d / ".gitkeep").write_text("")
    reset_run_state(w.exp)


def reset_run_state(exp: Path) -> list[str]:
    """Turn a copy of a kit that is already in its run (on the mbp after the pre-registration push, or after RUNBOOK
    step 7) back into its pre-registration state, in the copy only (review fix 2026-10-03):
    - HYPOTHESIS.md: the hash table and the stamp back to their placeholders, and every "hash_tree: NAME = hex"
      amendment line neutralised (tools/hash_tree.py unfill), so the copy fills its own values over its own,
      scaled runner/plan_rules.json and hash_check compares like with like;
    - the item manifests built from real data at step 7 (everything in tasks/manifests/ except the IFBench sets,
      which are built from public data at kit build time) and tools/withheld_shingles.sha256: the dry run builds
      synthetic ones, which would otherwise collide with the real files (build_manifests never overwrites).
    Returns what was reset."""
    sys.path.insert(0, str(EXP_DIR))
    from tools import hash_tree

    done = []
    hyp = exp / "HYPOTHESIS.md"
    text = hyp.read_text(encoding="utf-8")
    back = hash_tree.unfill(text)
    if back != text:
        hyp.write_text(back, encoding="utf-8")
        done.append("HYPOTHESIS.md hash table, stamp and hash_tree amendment lines")
    for p in sorted((exp / "tasks" / "manifests").glob("*.json")):
        if not p.name.startswith("ifbench"):
            p.unlink()
            done.append(f"tasks/manifests/{p.name}")
    sh = exp / "tools" / "withheld_shingles.sha256"
    if sh.exists():
        sh.unlink()
        done.append("tools/withheld_shingles.sha256")
    return done


def scale_plan_rules(rules: dict) -> dict:
    """runner/plan_rules.json at dry-run scale (the copy only): every cell small, every cap small, the B4 ladder
    never queued (the bench stage runs it). Semantics untouched: the same plans, queue order, rules and B rule."""
    r = json.loads(json.dumps(rules))
    small = {198: 4, 300: 4, 400: 6, 100: 4, 30: 3}

    def n_of(n):
        return n if n == "nM" else small.get(int(n), min(int(n), 4))

    for c in r["tier_a_queue"]:
        c["n"] = n_of(c["n"])
    for cells in r["tier_b_cells"].values():
        for c in cells:
            if c["arm"] != "bench":
                c["n"] = n_of(c["n"])
    for p, d in r["plan_defs"].items():
        d["nM"] = 28 if p in ("P0", "P1") else 14
    r["nM_ladder"] = [28, 28, 14, 14, 14, 14, 14, 14]
    r["caps"] = {k: 24 for k in r["caps"]}
    r["cap_raise"]["ceilings"] = {k: 32 for k in r["cap_raise"]["ceilings"]}
    r["ladder_estimate_h"] = {"B4": 1000.0}
    r["dry_run"] = "scaled by tools/dry_run.py in a temporary copy; never committed"
    return r


def write_hf_metadata(d: Path, revision: str) -> None:
    """hf download --local-dir metadata: line 1 the commit, line 2 the etag (sha256 here)."""
    for p in sorted(d.rglob("*")):
        rel = p.relative_to(d)
        if not p.is_file() or rel.parts[0] == ".cache" or rel.parts[0] == ".git":
            continue
        m = d / ".cache" / "huggingface" / "download" / (rel.as_posix() + ".metadata")
        m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(f"{revision}\n{sha256_file(p)}\n1791051258.0\n")


def assets() -> dict:
    return json.loads((EXP_DIR / "assets.json").read_text(encoding="utf-8"))


def pinned(repo: str) -> str:
    a = assets()
    for group in ("models", "datasets"):
        for e in a[group]:
            if e["repo"] == repo:
                return e["revision"]
    for e in a["code"]:
        if e["repo"].rstrip("/").endswith(repo):
            return e["commit"]
    raise KeyError(repo)


def build_models(r: Runner, tok: Path, peers: Path) -> str:
    """The tiny Kolibri source, the peer stand-in sources (written in-process by tests/tiny_checkpoint.py from
    the copy), the K8/K4 conversions happen in the convert stage; the peer conversions here."""
    w = r.w
    r.run([sys.executable, w.exp / "tools" / "dry_run.py", "--stage", "write-tiny", "--root", w.root,
           "--tok", tok, "--peers", peers])
    for fam, (src_folder, arms) in PEER_FAMILIES.items():
        for arm, folder in arms.items():
            bits = 8 if arm.endswith("8") else 4
            r.py("port/convert.py", "--src", w.src / fam, "--out", w.models / folder, "--bits", bits,
                 "--group-size", 64, "--results-dir", "none")
            write_hf_metadata(w.models / folder, pinned(f"mlx-community/{folder}"))
    return "tiny Kolibri BF16 (pattern5, vocab 128000) + 6 peer stand-ins converted"


def stage_write_tiny(root: Path, tok: Path, peers: Path) -> None:
    """In the copy: tiny checkpoints through tests/tiny_checkpoint.py (the kit's own writer)."""
    w = World(root)
    sys.path[:0] = [str(EXP_DIR), str(EXP_DIR / "tests")]
    import tiny_checkpoint as tc

    def hf_layout(d: Path, files_from: Path, files) -> None:
        cfg = json.loads((d / "config.json").read_text())
        cfg.pop("model_file", None)  # an HF snapshot carries no model_file; convert.py adds it
        (d / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        for f in files:
            if (files_from / f).is_file():
                shutil.copyfile(files_from / f, d / f)

    k = tc.write_tiny_checkpoint(w.models / "Kolibri-1-BF16", seed=0, preset="pattern5", copy_port=False,
                                 overrides={"vocab_size": 128000, "eos_token_id": 127906, "pad_token_id": 127901})
    hf_layout(k, tok, KOLIBRI_FILES)
    write_hf_metadata(k, pinned("Aleph-Alpha/Kolibri-1-BF16"))
    for fam, (src_folder, _arms) in PEER_FAMILIES.items():
        pdir = peers / src_folder
        cfg = json.loads((pdir / "config.json").read_text())
        tcfg = cfg.get("text_config", cfg)
        gen = json.loads((pdir / "generation_config.json").read_text())
        eos = gen.get("eos_token_id")
        eos = eos if isinstance(eos, list) else [eos]
        pad = gen.get("pad_token_id", tcfg.get("pad_token_id", eos[0]))
        d = tc.write_tiny_checkpoint(w.src / fam, seed=11, preset="vendor", copy_port=False,
                                     overrides={"vocab_size": int(tcfg["vocab_size"]), "eos_token_id": int(eos[0]),
                                                "pad_token_id": int(pad)})
        hf_layout(d, pdir, PEER_FILES)


def build_data(w: World, data: Path | None) -> dict:
    sys.path[:0] = [str(EXP_DIR / "tests")]
    import tasks_synthetic as syn

    w.data.mkdir(parents=True, exist_ok=True)
    syn.write_gpqa_en(w.data, n_diamond=6, n_main_extra=10)
    syn.write_gpqa_de(w.data, n_diamond=6, n_other=10)
    syn.write_mmlu(w.data, lite_per_cat=3, full_extra_per_cat=11)
    syn.write_aime(w.data, n=5)
    syn.write_rgb(w.data, n_en=12, n_fact=6, n_int=20)
    reword_synthetic(w.data)
    for name, repo in (("gpqa", "Idavidrein/gpqa"), ("gpqa-multilingual", "ellamind/gpqa-multilingual"),
                       ("MMLU-ProX-Lite", "li-lab/MMLU-ProX-Lite"), ("MMLU-ProX", "li-lab/MMLU-ProX"),
                       ("aime26", "math-ai/aime26"), ("aime26-multilingual", "ellamind/aime26-multilingual")):
        write_hf_metadata(w.data / name, pinned(repo))
    (w.data / "RGB-src" / ".git").mkdir(parents=True, exist_ok=True)
    (w.data / "RGB-src" / ".git" / "HEAD").write_text(pinned("chen700564/RGB") + "\n")
    info = {"synthetic": ["gpqa", "gpqa-multilingual", "MMLU-ProX(-Lite)", "aime26(-multilingual)", "RGB-src"]}
    real_ifb = data is not None and (data / "IFBench_test").is_dir() and (data / "IFBench-src" / ".git").exists()
    if real_ifb:
        (w.data / "IFBench_test").symlink_to(data / "IFBench_test")
        (w.data / "IFBench-src").symlink_to(data / "IFBench-src")
        info["ifbench"] = "public IFBench_test and IFBench-src from --data"
    else:
        syn.write_ifbench(w.data, n=12)
        write_hf_metadata(w.data / "IFBench_test", pinned("allenai/IFBench_test"))
        info["ifbench"] = "synthetic (IFBench scoring skipped)"
    pq = data / "fineweb-2" / "data" / "deu_Latn" / "test" / "000_00000.parquet" if data else None
    if pq is not None and pq.is_file():
        (w.data / "fineweb-2" / "data" / "deu_Latn" / "test").mkdir(parents=True, exist_ok=True)
        (w.data / "fineweb-2" / "data" / "deu_Latn" / "test" / "000_00000.parquet").symlink_to(pq)
        info["fineweb"] = "the pinned FineWeb-2 shard from --data (text stays in the temporary directory)"
    else:
        write_synthetic_parquet(w.data / "fineweb-2" / "data" / "deu_Latn" / "test" / "000_00000.parquet")
        info["fineweb"] = "synthetic German parquet (gate texts T5/T6/T9 synthetic)"
    info["real_ifbench"] = real_ifb
    info["real_fineweb"] = pq is not None and pq.is_file()
    return info


# The synthetic rows of tests/tasks_synthetic.py also appear as literals in the kit's tests. Here they play the
# withheld sets, so the leak check (pre-push hook) would rightly find them in the kit; the dry run rewords them
# so that every 8-word window and every option string differs from the test literals.
_REWORD = (("Synthetische", "Probelauf"), ("Synthetic", "Dryrun"), ("synthetic", "dryrun"),
           ("Bauteilwert", "Zahnradwert"), ("Bauteils", "Zahnrads"), ("Kalibrierung", "Ausrichtung"),
           ("Aufgabe", "Uebungsaufgabe"), ("Summe", "Gesamtsumme"), ("widget", "sprocket"), ("Widget", "Sprocket"),
           ("calibration", "alignment"), ("gadget", "gizmo"), ("Gadget", "Gizmo"), ("Option", "Choice"),
           ("Antwort", "Auswahl"), (" value ", " reading "),
           # RGB's instruction strings are withheld options since the review fix of 2026-10-03; the synthetic
           # instruction.yaml of tests/tasks_synthetic.py is a test literal too
           ("STUB", "Placeholder"))  # text only: JSON keys and CSV headers stay


def reword_synthetic(root: Path) -> None:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in (".csv", ".jsonl", ".json", ".yaml"):
            s = p.read_text(encoding="utf-8")
            for a, b in _REWORD:
                s = s.replace(a, b)
            p.write_text(s, encoding="utf-8")


_DE_WORDS = ("Die Stadt plant im kommenden Jahr eine neue Brücke über den Fluss, damit der Verkehr am Morgen "
             "schneller fließt. Viele Bürger begrüßen das Vorhaben, andere fürchten Lärm und höhere Kosten. "
             "Im Rathaus wurde lange über die Finanzierung gestritten, doch am Ende stimmte eine Mehrheit zu. "
             "Die Bauarbeiten sollen im Frühjahr beginnen und zwei Jahre dauern. ").split()


def synthetic_doc(i: int, n_words: int) -> str:
    """Our own German filler (no web text), varied by index."""
    out = []
    for j in range(n_words):
        out.append(_DE_WORDS[(i * 7 + j * 3) % len(_DE_WORDS)])
        if j % 23 == 22:
            out.append(str(1000 + i * 13 + j))
    return " ".join(out)


def write_synthetic_parquet(path: Path, n: int = 5100) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    path.parent.mkdir(parents=True, exist_ok=True)
    texts = [synthetic_doc(i, 60 + (i % 9) * 40 + (1600 if i in (5000, 5019) else 0)) for i in range(n)]
    pq.write_table(pa.table({"text": texts, "id": [f"syn-{i}" for i in range(n)]}), str(path))


def write_synthetic_gate_texts(w: World) -> None:
    """T5, T6 (text + ids) and T9 (ids) for $EXP036_WORK/gate_texts when the real shard is absent."""
    sys.path[:0] = [str(EXP_DIR)]
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(str(w.models / "Kolibri-1-BF16" / "tokenizer.json"))
    out = w.work / "gate_texts"
    out.mkdir(parents=True, exist_ok=True)
    t9 = []
    for k, tid in enumerate(("T5", "T6")):
        text = synthetic_doc(5000 + 19 * k, 2400)
        ids = tok.encode(text, add_special_tokens=False).ids[:1536]
        text = tok.decode(ids)
        ids = tok.encode(text, add_special_tokens=False).ids
        (out / f"{tid}.txt").write_text(text, encoding="utf-8")
        (out / f"{tid}.ids.json").write_text(json.dumps({"id": tid, "ids": ids}))
        t9 += ids
    while len(t9) < 16384:
        t9 += tok.encode(synthetic_doc(len(t9), 400), add_special_tokens=False).ids
    (out / "T9.ids.json").write_text(json.dumps({"id": "T9", "ids": t9[:16384]}))


# ================================================================================ in-copy stages


def stage_manifests() -> dict:
    """tasks/build_manifests.build_all on the synthetic data. Counts are not enforced: the synthetic sets are
    small, and the GPQA over-long question (eval-framework's, excluded by hash) cannot be synthesised. Revisions
    are checked (the synthetic snapshots carry the pinned revision metadata)."""
    sys.path.insert(0, str(EXP_DIR))
    from runner.guard import require_identity
    from tasks import build_manifests as bm

    require_identity()
    s = bm.build_all(results_dir=EXP_DIR / "results", enforce_counts=False, check_revisions=True)
    return {"counts": s["counts"], "results": s.get("results"), "shingles": bool(s.get("shingles"))}


def stage_peers_record() -> str:
    """results/peers_<UTC>.json for the runner's guard: every stand-in ok. The stand-ins are Kolibri-architecture
    weights under the peers' tokenizers, so tools/peer_check.py's static checks (config arithmetic, parity
    record) cannot pass on them by construction; its weight checks ran in the peers stage."""
    sys.path.insert(0, str(EXP_DIR))
    from runner.guard import require_identity

    require_identity()
    now = datetime.now(timezone.utc)
    rec = {"schema": "exp036 peer check v1", "dry_run": "tools/dry_run.py: tiny stand-ins, marked ok",
           "t_start": now.isoformat().replace("+00:00", "Z"), "t_end": now.isoformat().replace("+00:00", "Z"),
           "arms": {a: {"verdict": "ok", "problems": [], "dry_run": True}
                    for _f, (_s, arms) in PEER_FAMILIES.items() for a in arms}}
    p = EXP_DIR / "results" / f"peers_{utc_stamp(now)}.json"
    with open(p, "x", encoding="utf-8") as f:
        f.write(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    return p.name


def stage_promote_gate() -> str:
    """Re-label the tiny gate PASS as a real-mode record for the guards (dry run only). Everything the guard
    checks (port, thresholds, reference tree, converted manifests) is the tiny run's own, unchanged."""
    sys.path.insert(0, str(EXP_DIR))
    from gate import run_gate

    path = run_gate.latest_record(EXP_DIR / "results")
    rec = json.loads(path.read_text(encoding="utf-8"))
    if rec.get("mode") != "tiny" or rec.get("tiny_outcome") != {"K8": "PASS", "K4": "PASS"} or rec["exit_code"] != 0:
        raise StageError(f"{path.name}: tiny outcome {rec.get('tiny_outcome')}, exit {rec.get('exit_code')}")
    t = datetime.strptime(rec["utc"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    stamp = utc_stamp(max(datetime.now(timezone.utc), t + timedelta(seconds=1)))
    rec.update(mode="real", verdict=dict(rec["tiny_outcome"]), utc=stamp,
               dry_run_promoted_from=path.name,
               dry_run_note="tools/dry_run.py: the tiny gate's PASS re-labelled so the runner's guards accept it")
    out = path.parent / f"gate_{stamp}.json"
    with open(out, "x", encoding="utf-8") as f:
        f.write(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    for arm in ("K8", "K4"):
        run_gate.require_pass(arm, results_dir=EXP_DIR / "results")
    return out.name


def stage_bench() -> dict:
    """Every bench cell through its Python entry point, at seconds scale (sizes passed as arguments)."""
    sys.path.insert(0, str(EXP_DIR))
    from bench import batch_flip, fit, kl_8v4, ladder, speed, tokenizer_ratio

    res = EXP_DIR / "results"
    cool = {"dry_run": "no cool-down: tiny models"}
    out = {}
    out["speed"] = speed.run_h1(res, n_blocks=2, idle_s=0, prompt_tokens=64, gen_tokens=16, prefill_tokens=256,
                                prefill_reps=1, cool=cool)
    out["speed_desc"] = speed.run_desc(res, reps=1, idle_s=0, prompt_tokens=64, gen_tokens=16, prefill_tokens=256,
                                       batch_sizes=(1, 2), cool=cool)
    out["fit"] = fit.run_cell(res, sizes=(512, 1024), reps=1, gen_tokens=8)
    out["kl"] = kl_8v4.run_cell(res)
    st = World(Path(os.environ["DRY_RUN_ROOT"])).load_state()
    out["tokenizer"] = tokenizer_ratio.run_cell(res, n=60, verify_parquet_sha=bool(st.get("real_fineweb")))
    out["c1"] = batch_flip.run_cell(res, B=4, max_tokens=16)
    out["ladder"] = ladder.run_cell(res, rungs=(("4k", "pad_4k.txt", 256, None), ("32k", "pad_120k.txt", 512, None)),
                                    gen_tokens=8, idle_s=0, reps={"4k": 1, "32k": 1})
    return {k: Path(v).relative_to(EXP_DIR).as_posix() for k, v in out.items()}


# ================================================================================ orchestration


def tier2_amendment(w: World, k: int) -> Path:
    files = ["analysis/exploratory.py", "analysis/tables.py", "bench/ladder.py"]
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    lines = [f"## Amendment {k} — Tier-2 analysis ({now})", "",
             "*Dry run (tools/dry_run.py): the Tier-2 files frozen by sha256.*", ""]
    lines += [f"- `{f}` sha256 `{sha256_file(w.exp / f)}`" for f in files]
    p = w.exp / "amendments" / f"{k}_tier2_{utc_stamp()}.md"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def fill_signoff(hyp: Path) -> None:
    text = hyp.read_text(encoding="utf-8").splitlines()
    out = []
    for ln in text:
        if ln.strip() == "- Signed off by:":
            ln = SIGNOFF_LINE
        elif ln.startswith("- Kolibri `embed_tokens` / `lm_head` policy") and ln.rstrip().endswith("):"):
            ln = ln.rstrip() + " quantised at the arm's bits with fp32 logits (pre-registered default; dry run)"
        out.append(ln)
    hyp.write_text("\n".join(out) + "\n", encoding="utf-8")


def newest(d: Path, pattern: str) -> Path:
    fs = sorted(d.glob(pattern))
    if not fs:
        raise StageError(f"no {pattern} in {d.name}/")
    return fs[-1]


def orchestrate(args) -> int:
    tok = Path(args.tok).expanduser() if args.tok else default_tok()
    peers = Path(args.peers).expanduser() if args.peers else default_peers()
    data = Path(args.data).expanduser() if args.data else default_data()
    if tok is None or not (tok / "tokenizer.json").is_file():
        print(json.dumps({"ok": False, "error": "no Kolibri tokenizer: pass --tok"}))
        return 1
    if peers is None or not all((peers / s / "tokenizer.json").is_file() for s, _ in PEER_FAMILIES.values()):
        print(json.dumps({"ok": False, "error": "no peer tokenizer folders: pass --peers"}))
        return 1
    root = Path(tempfile.mkdtemp(prefix="exp036_dry_run_", dir=args.workdir))
    w = World(root)
    for d in (w.logs, w.models, w.private, w.work, w.side):
        d.mkdir(parents=True, exist_ok=True)
    ifb_py, nltk = default_ifbench()
    r = Runner(w, args.verbose)
    info: dict = {}
    env_root = {"DRY_RUN_ROOT": str(root)}

    def in_copy(stage: str, *extra) -> subprocess.CompletedProcess:
        return r.run([sys.executable, w.exp / "tools" / "dry_run.py", "--stage", stage, "--root", root, *extra],
                     env=w.env(env_root))

    def world():
        copy_kit(w)
        rules_p = w.exp / "runner" / "plan_rules.json"
        rules_p.write_text(json.dumps(scale_plan_rules(json.loads(rules_p.read_text())), indent=2) + "\n")
        info.update(build_data(w, data))
        w.save_state(real_fineweb=info["real_fineweb"], real_ifbench=info["real_ifbench"],
                     ifbench_py=str(ifb_py) if (ifb_py and info["real_ifbench"]) else None,
                     nltk_data=str(nltk) if (nltk and info["real_ifbench"]) else None)
        if not info["real_ifbench"]:  # the committed IFBench manifests come from the real data
            for p in (w.exp / "tasks" / "manifests").glob("ifbench*.json"):
                p.unlink()
        build_models(r, tok, peers)
        r.run(["git", "init", "-q", "--bare", w.remote], cwd=root)
        r.run(["git", "init", "-q", "-b", "main"], cwd=w.repo)
        r.git("config", "user.name", IDENTITY[0])
        r.git("config", "user.email", IDENTITY[1])
        r.git("config", "core.hooksPath", ".git/hooks")
        r.git("config", "commit.gpgsign", "false")
        r.git("remote", "add", "origin", w.remote)
        hooks = w.repo / ".git" / "hooks"
        shutil.copy2(w.exp / "tools" / "hooks" / "pre-push", hooks / "pre-push")
        os.chmod(hooks / "pre-push", 0o755)
        return {k: v for k, v in info.items() if not k.startswith("real_")}

    def preregister():
        r.py("tools/hash_tree.py", "--fill", "HYPOTHESIS.md")
        r.py("tools/hash_tree.py", "--stamp", "HYPOTHESIS.md")
        left = (w.exp / "HYPOTHESIS.md").read_text(encoding="utf-8").count("{{")
        if left:
            raise StageError(f"{left} placeholders left after --fill/--stamp")
        r.git("add", "-A")
        r.git("commit", "-q", "-m", "chronos/exp_036: pre-registration (dry run)", "-m", "dry run")
        r.git("push", "-q", "-u", "origin", "main")
        return "filled, stamped, committed, pushed through the pre-push hook"

    def preflight():
        out = w.side / "preflight_mini.json"
        p = r.py("tools/preflight.py", "--quick", "--not-run-host", "--out", out, ok_codes=(0, 1, 2))
        rec = json.loads(out.read_text())
        return {"exit": p.returncode, "next_step": rec.get("next_step"), "memory_rule": bool(rec.get("memory_rule"))}

    def signoff():
        fill_signoff(w.exp / "HYPOTHESIS.md")
        r.py("tools/status.py", "--record-block", "sign-off")
        r.commit("chronos/exp_036: sign-off placeholder (dry run)")
        return "placeholder sign-off line (temporary copy only)"

    def manifests():
        p = in_copy("manifests")
        out = json.loads(p.stdout.strip().splitlines()[-1])
        st = w.load_state()
        if st.get("real_fineweb"):
            r.py("gate/build_gate_text.py", "--work-only", "--fineweb",
                 w.data / "fineweb-2" / "data" / "deu_Latn" / "test" / "000_00000.parquet")
        else:
            in_copy("synthetic-gate-texts")
        r.commit("chronos/exp_036: item manifests (dry run)")
        return out

    def convert():
        for bits in (8, 4):
            r.py("port/convert.py", "--src", w.models / "Kolibri-1-BF16",
                 "--out", w.models / f"Kolibri-1-MLX-{bits}bit-g64", "--bits", bits, "--group-size", 64)
        r.commit("chronos/exp_036: K8/K4 conversion records (dry run)")
        return sorted(p.name for p in (w.exp / "results" / "convert").glob("*.json"))

    def peers_stage():
        p = r.py("tools/peer_check.py", "--arms", "G8,G4,Q36-8,Q36-4,Q38-8,Q38-4", "--checks", "load,nll,kl,batch",
                 "--out-dir", w.side, ok_codes=(0, 1, 2))
        rec = json.loads(newest(w.side, "peers_*.json").read_text())
        bad = []
        for arm in ("G8", "Q36-8"):
            bp = rec["arms"][arm].get("batched_path") or {}
            if "error" in bp or not bp.get("parity", {}).get("admitted_mid_run"):
                bad.append((arm, bp.get("error")))
        for fam in ("gemma4", "qwen3_6", "qwen3_8"):
            f = rec["families"].get(fam, {})
            if "nll_error" in f or "fidelity" not in f:
                bad.append((fam, f.get("nll_error")))
        if bad:
            raise StageError(f"peer_check weight checks did not run: {bad}")
        name = in_copy("peers-record").stdout.strip().splitlines()[-1]
        r.commit("chronos/exp_036: peer check (dry run)")
        return {"peer_check_exit": p.returncode, "batched_path_ok": {a: rec["arms"][a]["batched_path"]["ok"]
                                                                     for a in ("G8", "Q36-8")},
                "record": name}

    def gate():
        p = r.py("gate/run_gate.py", "--tiny", w.models, "--tok", w.models / "Kolibri-1-BF16",
                 "--results-dir", w.exp / "results", "--head-policy", "quantised_head", timeout=3600)
        last = json.loads(p.stdout.strip().splitlines()[-1])
        promoted = in_copy("promote-gate").stdout.strip().splitlines()[-1]
        for arm in ("K8", "K4"):  # the runner's refusal, through the CLI as well
            r.py("gate/run_gate.py", "--require-pass", arm)
        r.py("tools/status.py", "--sync-amendments")
        r.py("tools/status.py", "--record-block", "gate")
        r.commit("chronos/exp_036: gate (dry run, tiny) — PASS")
        return {"exit": p.returncode, "verdict": last["verdict"], "promoted": promoted}

    def tier2():
        # the next free number, as runner/plan_fix.py next_amendment_k counts (Amendment 0 is the sign-off's)
        import re

        texts = [(w.exp / "HYPOTHESIS.md").read_text(encoding="utf-8")]
        texts += [p.read_text(encoding="utf-8") for p in sorted((w.exp / "amendments").glob("*.md"))]
        ks = [int(m) for t in texts for m in re.findall(r"^## Amendment (\d+) ", t, re.M)]
        k = max([0] + ks) + 1
        tier2_amendment(w, k)
        r.git("add", "-A", REL_EXP)
        r.git("commit", "-q", "-m", f"chronos/exp_036: amendment {k} — Tier-2 analysis (dry run)", "-m", "dry run")
        r.git("push", "-q", "origin", "HEAD")
        r.py("tools/status.py", "--sync-amendments")
        r.commit("chronos/exp_036: sync amendments (dry run)")
        return f"Amendment {k}"

    def bench():
        p = in_copy("bench")
        out = json.loads(p.stdout.strip().splitlines()[-1])
        r.commit("chronos/exp_036: S1 bench cells (dry run)")
        return out

    def pilot():
        r.py("runner/run.py", "pilot", timeout=3600)
        s = newest(w.exp / "results", "pilot_summary_*.json")
        rec = json.loads(s.read_text())
        return {"summary": s.name, "cells": len(rec["cells"]),
                "truncated": sum(c["n_truncated"] for c in rec["cells"]),
                "items": sum(c["n"] for c in rec["cells"])}

    def plan():
        s = newest(w.exp / "results", "pilot_summary_*.json")
        p = r.py("runner/plan_fix.py", "--pilot", s.relative_to(w.exp))
        out = json.loads(p.stdout.strip().splitlines()[-1])
        am = newest(w.exp / "results", "AMENDMENT_*.md")
        r.py("tools/status.py", "--sync-amendments")
        with open(w.exp / "HYPOTHESIS.md", "a", encoding="utf-8") as f:
            f.write(am.read_text(encoding="utf-8"))
        r.py("tools/status.py", "--record-block", "plan")
        r.commit(f"chronos/exp_036: {am.stem} — plan fixed by rule (dry run)")
        return out

    def sessions():
        out = {}
        for name in ("S2", "S3"):
            p = r.py("runner/run.py", "session", "--name", name, timeout=7200)
            out[name] = sum(1 for ln in p.stdout.splitlines() if ln.startswith("{"))
            r.py("tools/status.py", "--sync-amendments")
            r.py("tools/status.py", "--record-block", name)
            r.commit(f"chronos/exp_036: {name} — raw outputs (dry run)")
            if name == "S3" and "Tier-A outstanding" in p.stdout:
                r.py("runner/run.py", "session", "--name", "S3b", timeout=7200)
                r.commit("chronos/exp_036: S3b — raw outputs (dry run)")
                out["S3b"] = True
        st = r.py("runner/run.py", "status")
        out["status_tail"] = st.stdout.strip().splitlines()[-1][:200]
        nr = sorted((w.exp / "results" / "raw").glob("*/not_run.jsonl"))
        out["not_run_files"] = [p.relative_to(w.exp).as_posix() for p in nr]
        r.commit("chronos/exp_036: status (dry run)")
        return out

    def score():
        r.py("scorers/score_all.py")
        r.py("tools/version_record.py")
        r.py("tools/status.py", "--record-block", "S3, mbp scores")
        r.commit("chronos/exp_036: S3 — raw outputs; mbp scores (dry run)")
        st = w.load_state()
        ifb = bool(st.get("ifbench_py"))
        if ifb:
            r.py("scorers/score_all.py", "--ifbench", timeout=3600)
        r.py("scorers/score_all.py", "--rescore-compare", *(["--ifbench"] if ifb else []), timeout=3600)
        rc = json.loads(newest(w.exp / "results", "rescore_mini_*.json").read_text())
        files = sorted(p.relative_to(w.exp / "results").as_posix() for p in (w.exp / "results" / "scores").rglob("*.jsonl"))
        return {"score_files": len(files), "ifbench_scored": ifb, "rescore_all_identical": rc["all_identical"]}

    def verdicts():
        r.py("analysis/verdicts.py", timeout=3600)
        v = json.loads(newest(w.exp / "results", "verdicts_*.json").read_text())
        out = {}
        for h in ("H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "D1"):
            x = v["verdicts"].get(h)
            out[h] = x.get("verdict", x.get("state")) if isinstance(x, dict) else x
        r.py("tools/status.py", "--record-block", "verdicts")
        r.commit("chronos/exp_036: verdicts (dry run)")
        return out

    def hash_check():
        p = r.py("tools/hash_tree.py", "--check", "HYPOTHESIS.md")
        lines = [ln.split() for ln in p.stdout.splitlines() if ln.strip()]
        n = sum(1 for ln in lines if ln[0] == "match")
        if n == 0 or n != len(lines):
            raise StageError(f"hash_tree --check: {n} of {len(lines)} scopes match")
        return f"{n} scopes match the filled table"

    def leak():
        r.py("tools/leak_check.py", "--all")
        r.py("tools/leak_check.py", "--range", "@{u}..HEAD", "--staged")
        r.run([sys.executable, EXP_DIR / "tools" / "leak_check.py", "--all", "--no-gpqa-source"], cwd=EXP_DIR,
              env={k: v for k, v in os.environ.items()} | {"HF_HUB_OFFLINE": "1"})
        return "the copy (--all, --range) and this kit's working tree (--all --no-gpqa-source) are clean"

    plan_ = [("world", world), ("preregister", preregister), ("preflight", preflight), ("signoff", signoff),
             ("manifests", manifests), ("convert", convert), ("peers", peers_stage), ("gate", gate),
             ("tier2", tier2), ("bench", bench), ("pilot", pilot), ("plan", plan), ("sessions", sessions),
             ("score", score), ("verdicts", verdicts), ("hash_check", hash_check), ("leak", leak)]
    t0 = time.time()
    ok = True
    for name, fn in plan_:
        if not r.stage(name, fn):
            ok = False
            break
    result = {"ok": ok, "minutes": round((time.time() - t0) / 60, 1), "stages": r.stages}
    if args.keep or not ok:
        result["kept"] = str(root)
    else:
        shutil.rmtree(root, ignore_errors=True)
    print(json.dumps(result, sort_keys=True, default=str))
    return 0 if ok else 1


def main(argv=None) -> int:
    from tools.precision import ensure_exact_fp32
    ensure_exact_fp32()  # MLX_ENABLE_TF32=0 before any GPU work (tools/precision.py; Amendment 1)
    ap = argparse.ArgumentParser(description="exp_036 dry run on tiny checkpoints (RUNBOOK step 3b)")
    ap.add_argument("--tok", help="Kolibri tokenizer dir (default $EXP036_TOK, $EXP036_MODELS/Kolibri-1-BF16, "
                                  "~/models/exp036-mini/kolibri/Kolibri-1-BF16)")
    ap.add_argument("--peers", help="folder holding the mlx-community peer folders (tokenizer files are borrowed; "
                                    "default $EXP036_MODELS, ~/models/exp036-mini/mlx-community)")
    ap.add_argument("--data", help="dataset root for the public IFBench and FineWeb-2 files (default $EXP036_DATA, "
                                   "$EXP036_MODELS/data, ~/models/exp036-mini/data); synthetic if absent")
    ap.add_argument("--workdir", help="parent of the temporary directory (default: the system temp dir)")
    ap.add_argument("--keep", action="store_true", help="keep the temporary directory")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--stage", help=argparse.SUPPRESS)
    ap.add_argument("--root", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.stage:
        os.environ.setdefault("DRY_RUN_ROOT", args.root or "")
        if args.stage == "write-tiny":
            stage_write_tiny(Path(args.root), Path(args.tok), Path(args.peers))
            print(json.dumps({"ok": True}))
            return 0
        fns = {"manifests": stage_manifests, "peers-record": stage_peers_record,
               "promote-gate": stage_promote_gate, "bench": stage_bench,
               "synthetic-gate-texts": lambda: write_synthetic_gate_texts(World(Path(args.root))) or "written"}
        out = fns[args.stage]()
        print(json.dumps(out, sort_keys=True, default=str) if not isinstance(out, str) else out)
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
