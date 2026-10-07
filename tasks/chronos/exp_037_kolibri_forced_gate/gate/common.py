# SPDX-License-Identifier: MIT
"""Shared helpers for the exp_036 gate (BUILD_SPEC §2, §5.3).

Paths come from the environment (BUILD_SPEC §1; the same defaults as
tools/common.py, which is used when present):

    EXP036_MODELS   default ~/models/exp036   (Kolibri-1-BF16, Kolibri-1-MLX-{8,4}bit-g64)
    EXP036_DATA     default $EXP036_MODELS/data
    EXP036_WORK     default $EXP036_MODELS/work   (reference dumps, T5/T6/T9)
    EXP036_TOK / EXP036_TOKENIZER_DIR   a directory with the Kolibri tokenizer (tests)

exp_037 (DESIGN §2.8-§2.10, §6.5) adds:

    EXP037_BUILDS   default $EXP036_MODELS   builds_dir(): the Kolibri-1-MLX-{8,4}bit-g64 arm
                    directories (on the mbp, env/exp037.settings.sh sets $EXP036_MODELS/exp037-builds)
    work37_dir()    $EXP036_WORK/exp037      exp_037's own work files; nothing of exp_037 is written
                                             elsewhere under $EXP036_WORK

seed_from() draws under the "exp037" prefix; the registered exp_036 fixture
seeds pass prefix="exp036" explicitly. Port modules are named
exp037_gate_port_<n>, and load_port() passes trust_remote_code=True to
mlx_lm explicitly (mlx-lm 0.32.0, #1385).

Nothing here touches the network. mlx, mlx_lm and tokenizers are imported
lazily, so the pure-Python parts (hashing, records, refusal) work without them.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.util
import itertools
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

GATE_DIR = Path(__file__).resolve().parent
EXP_DIR = GATE_DIR.parent
TEXTS_DIR = GATE_DIR / "texts"
THRESHOLDS_PATH = GATE_DIR / "thresholds.json"
BEHAVIOUR_PROMPTS = GATE_DIR / "behaviour_prompts.json"
TEMPLATE_CASES = GATE_DIR / "template_cases.json"
PORT_FILE = EXP_DIR / "port" / "kolibri1.py"
REFERENCE_DIR = EXP_DIR / "reference"
VENDOR_JINJA = EXP_DIR / "tests" / "fixtures" / "kolibri1_chat_template.vendor.jinja"
DEFAULT_RESULTS_DIR = EXP_DIR / "results"

if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

# Kolibri-1 special ids (spec item 15).
EOS_IDS = (127906, 127901)
THINK_START_ID = 127907
THINK_END_ID = 127908

ARMS = ("K8", "K4")
ARM_BITS = {"K8": 8, "K4": 4}
ARM_DIRS = {"K8": "Kolibri-1-MLX-8bit-g64", "K4": "Kolibri-1-MLX-4bit-g64"}  # under builds_dir() (exp_037)
BF16_DIR = "Kolibri-1-BF16"

# The serialisation behind every "ids_sha256" in the gate (MANIFEST.json, records).
IDS_SHA256_RULE = "sha256 of the decimal token ids joined by ',' (no spaces, no brackets), UTF-8"
TREE_SHA256_RULE = (
    "tools/hash_tree.py: sha256 over the bytewise-sorted lines 'relpath\\tsha256(file)\\n' of every "
    "file git would commit under the tree, excluding __pycache__/, .pytest_cache/, *.pyc, *.pyo, .DS_Store"
)


def set_offline_env() -> None:
    """BUILD_SPEC §1: set before importing mlx_lm or transformers."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


# ---------------------------------------------------------------------------
# Environment paths (tools/common.py when present; identical defaults otherwise)
# ---------------------------------------------------------------------------


def _tools_common():
    try:
        from tools import common as tc  # sibling; stdlib only

        return tc
    except Exception:  # pragma: no cover - only while tools/ is absent
        return None


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def models_dir() -> Path:
    tc = _tools_common()
    return tc.models_dir() if tc else _env_path("EXP036_MODELS", Path.home() / "models" / "exp036")


def data_dir() -> Path:
    tc = _tools_common()
    return tc.data_dir() if tc else _env_path("EXP036_DATA", models_dir() / "data")


def work_dir() -> Path:
    tc = _tools_common()
    return tc.work_dir() if tc else _env_path("EXP036_WORK", models_dir() / "work")


def builds_dir() -> Path:
    """Where the Kolibri arm directories (ARM_DIRS) resolve: $EXP037_BUILDS if
    set and non-empty, else models_dir() (DESIGN §2.8). On the mbp
    env/exp037.settings.sh sets it to the refreshed clones; without it (the
    tiny tests, the dry run) the arms stay where exp_036 had them. The BF16
    source and the peers are always under models_dir()."""
    return _env_path("EXP037_BUILDS", models_dir())


def work37_dir() -> Path:
    """exp_037's own work files: $EXP036_WORK/exp037, i.e. under work_dir()
    and its fallbacks (DESIGN §2.9). The rule: every exp_037 work-file writer
    (R1 if recomputed, R2F/R3, the G0k outputs, the G5 log-probs, the G1 junit
    files) targets a path under it, so none of them, an rmtree included, can
    reach exp_036's work files. Not created here; writers make their own
    subdirectories."""
    return work_dir() / "exp037"


def tokenizer_dir() -> Path:
    """$EXP036_TOK, else $EXP036_TOKENIZER_DIR, else the BF16 snapshot."""
    for name in ("EXP036_TOK", "EXP036_TOKENIZER_DIR"):
        if os.environ.get(name):
            return Path(os.environ[name]).expanduser()
    return models_dir() / BF16_DIR


def redact_path(p) -> str:
    """tools.redact.redact_path when present; else $EXP036_* / ~ prefixes."""
    try:
        from tools.redact import redact_path as rp

        return rp(p)
    except Exception:  # pragma: no cover - only while tools/ is absent
        s = str(Path(p).expanduser())
        pairs = [(work_dir(), "$EXP036_WORK"), (data_dir(), "$EXP036_DATA"),
                 (models_dir(), "$EXP036_MODELS"), (EXP_DIR, "$EXP"), (Path.home(), "~")]
        for real, label in sorted(pairs, key=lambda t: -len(str(t[0]))):
            r = str(real).rstrip("/")
            if s == r or s.startswith(r + "/"):
                return label + s[len(r):]
        return s


def host_label() -> str:
    try:
        from tools.redact import HOST_LABEL

        return HOST_LABEL
    except Exception:  # pragma: no cover
        return "mbp (M5 Max, 128 GB)"


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def ids_sha256(ids) -> str:
    """See IDS_SHA256_RULE."""
    return sha256_bytes(",".join(str(int(i)) for i in ids).encode("utf-8"))


_TREE_SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".git"}
_TREE_SKIP_FILES = {".DS_Store"}


def _hash_tree():
    try:
        from tools import hash_tree  # sibling: the rule of the HYPOTHESIS hash table

        return hash_tree
    except Exception:  # pragma: no cover - only while tools/ is absent
        return None


def tree_files(root: Path, exclude: tuple[str, ...] = ()) -> list[Path]:
    """Local fallback of tools/hash_tree.list_files (without git): files under
    root sorted bytewise by relpath, minus `exclude` and the cache entries."""
    root = Path(root)
    rels = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(root).as_posix()
        parts = rel.split("/")
        if any(part in _TREE_SKIP_DIRS for part in parts[:-1]) or p.name in _TREE_SKIP_FILES \
                or p.suffix in (".pyc", ".pyo"):
            continue
        if any(rel == e.rstrip("/") or rel.startswith(e.rstrip("/") + "/") for e in exclude):
            continue
        rels.append(rel)
    return [root / r for r in sorted(rels, key=lambda r: r.encode("utf-8"))]


def tree_sha256(root: Path, exclude: tuple[str, ...] = ()) -> str:
    """See TREE_SHA256_RULE (tools/hash_tree.tree_sha256 when present)."""
    ht = _hash_tree()
    if ht is not None:
        return ht.tree_sha256(root, exclude=tuple(exclude))
    root = Path(root)
    lines = "".join(f"{p.relative_to(root).as_posix()}\t{sha256_file(p)}\n" for p in tree_files(root, exclude))
    return sha256_bytes(lines.encode("utf-8"))


def _scope(name: str, fallback):
    ht = _hash_tree()
    if ht is not None:
        return ht.scope_value(ht.BY_NAME[name])
    return fallback()


def port_sha256(port_file: Path = PORT_FILE) -> str:
    if Path(port_file) == PORT_FILE:
        return _scope("port", lambda: sha256_file(PORT_FILE))
    return sha256_file(port_file)


def reference_tree_sha256() -> str:
    return _scope("reference", lambda: tree_sha256(REFERENCE_DIR))


def gate_code_sha256() -> str:
    """HYPOTHESIS hash table: `gate/` tree minus thresholds.json and texts/."""
    return _scope("gate", lambda: tree_sha256(GATE_DIR, exclude=("thresholds.json", "texts/")))


def thresholds_sha256(path: Path = THRESHOLDS_PATH) -> str:
    if Path(path) == THRESHOLDS_PATH:
        return _scope("thresholds", lambda: sha256_file(THRESHOLDS_PATH))
    return sha256_file(path)


def gate_text_sha256() -> str:
    """HYPOTHESIS hash table: sha256 of gate/texts/MANIFEST.json."""
    return _scope("gate_text", lambda: sha256_file(TEXTS_DIR / "MANIFEST.json"))


CONVERT_RECORD = "exp036_convert_record.json"


def converted_manifest_sha256(model_dir: Path) -> str:
    """The converted directory's manifest sha256 (BUILD_SPEC §5.1 port/convert.py
    "Record"): the record's `manifest_sha256` when present, otherwise sha256
    over the sorted lines 'relpath\\tsha256\\n' of the record's file list."""
    record = json.loads((Path(model_dir) / CONVERT_RECORD).read_text(encoding="utf-8"))
    if record.get("manifest_sha256"):
        return record["manifest_sha256"]
    files = {name: entry["sha256"] for name, entry in record["files"].items()}
    ht = _hash_tree()
    if ht is not None:
        return ht.manifest_sha256(files)
    lines = "".join(sorted((f"{k}\t{v}\n" for k, v in files.items()), key=lambda x: x.encode("utf-8")))
    return sha256_bytes(lines.encode("utf-8"))


def checkpoint_fingerprint(model_dir: Path) -> str:
    """sha256 of model.safetensors.index.json (the reference's own fingerprint rule)."""
    index = Path(model_dir) / "model.safetensors.index.json"
    if index.is_file():
        return sha256_file(index)
    return sha256_file(Path(model_dir) / "config.json")


# ---------------------------------------------------------------------------
# Time, JSON, git
# ---------------------------------------------------------------------------


def utc_now() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0)


def utc_iso(t: _dt.datetime | None = None) -> str:
    return (t or utc_now()).strftime("%Y-%m-%dT%H:%M:%SZ")


def utc_stamp(t: _dt.datetime | None = None) -> str:
    """20261003T170152Z, the file-name form of host/ and results/."""
    return (t or utc_now()).strftime("%Y%m%dT%H%M%SZ")


def jsonable(obj):
    """numpy scalars/arrays to Python; non-finite floats to None (strict JSON)."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return jsonable(obj.tolist())
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, Path):
        return redact_path(obj)
    return obj


_NUM = r"-?[0-9][0-9eE.+\-]*"
# A multi-line JSON array of plain numbers (JSON strings never hold a raw newline).
_NUM_ARRAY = re.compile(r"\[\n\s*(" + _NUM + r"(?:,\n\s*" + _NUM + r")*)\n\s*\]")


def dumps(obj) -> str:
    """BUILD_SPEC §2 Determinism: UTF-8 JSON, sorted keys. Arrays of plain
    numbers (token ids, per-layer values) are written on one line."""
    text = json.dumps(jsonable(obj), indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
    text = _NUM_ARRAY.sub(lambda m: "[" + ", ".join(x.strip() for x in m.group(1).split(",")) + "]", text)
    return text + "\n"


def write_new_json(path: Path, obj) -> Path:
    """Write a new file; never overwrite (BUILD_SPEC §2 Append-only)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        f.write(dumps(obj))
    return path


def write_json(path: Path, obj) -> Path:
    """For build products under gate/texts and work files (overwrite allowed)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(obj), encoding="utf-8")
    return path


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def git_head() -> str | None:
    try:
        p = subprocess.run(["git", "rev-parse", "HEAD"], cwd=EXP_DIR, capture_output=True, text=True, timeout=30)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def git_dirty() -> bool | None:
    try:
        p = subprocess.run(["git", "status", "--porcelain", "--", "."], cwd=EXP_DIR,
                           capture_output=True, text=True, timeout=30)
        return bool(p.stdout.strip()) if p.returncode == 0 else None
    except Exception:
        return None


def require_identity() -> None:
    """BUILD_SPEC §2 Identity, before anything is written under results/.
    runner.guard.require_identity when present, else tools.common's rule."""
    try:
        from runner.guard import require_identity as rg  # sibling

        rg()
        return
    except ImportError:
        pass
    tc = _tools_common()
    if tc is not None:
        tc.require_identity(cwd=EXP_DIR)
        return
    raise RuntimeError("no identity check available (runner/guard.py and tools/common.py absent)")


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict:
    return read_json(path)


# ---------------------------------------------------------------------------
# Numerics shared by the checks (numpy only)
# ---------------------------------------------------------------------------


def bf16_exact_fraction(x) -> float:
    """Fraction of finite values exactly representable in bf16, i.e.
    x == float32(bfloat16(x)): the low 16 bits of the fp32 pattern are zero.
    fp32 compute gives about 2**-16; bf16 compute gives 1 (HYPOTHESIS G0)."""
    a = np.ascontiguousarray(np.asarray(x, dtype=np.float32)).reshape(-1)
    finite = np.isfinite(a)
    if not finite.any():
        return float("nan")
    bits = a[finite].view(np.uint32)
    return float(np.mean((bits & np.uint32(0xFFFF)) == 0))


def log_softmax64(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def kl_rows(ref_logits: np.ndarray, q_logits: np.ndarray, chunk: int = 256) -> np.ndarray:
    """KL(ref || q) per row in nats, float64, over the full vocabulary."""
    out = np.empty(ref_logits.shape[0], dtype=np.float64)
    for a in range(0, ref_logits.shape[0], chunk):
        lp = log_softmax64(ref_logits[a:a + chunk])
        lq = log_softmax64(q_logits[a:a + chunk])
        out[a:a + chunk] = (np.exp(lp) * (lp - lq)).sum(axis=-1)
    return out


def top_lead(logits: np.ndarray) -> np.ndarray:
    """Top-1 minus top-2 per row (for log-probs this is the lead in nats)."""
    part = np.partition(np.asarray(logits, dtype=np.float64), -2, axis=-1)
    return part[:, -1] - part[:, -2]


def rel_err_rows(got: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """||got - ref|| / ||ref|| per row (item 29: each branch relative to its own
    reference norm), float64. A zero reference row gives inf unless got is 0 too."""
    got = np.asarray(got, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    num = np.linalg.norm(got - ref, axis=-1)
    den = np.linalg.norm(ref, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.where(num > 0, np.inf, 0.0))
    return e


def stats(e: np.ndarray) -> dict:
    e = np.asarray(e, dtype=np.float64).reshape(-1)
    if e.size == 0:
        return {"n": 0, "median": None, "p99": None, "max": None, "mean": None}
    return {
        "n": int(e.size),
        "median": float(np.median(e)),
        "p99": float(np.quantile(e, 0.99)),
        "max": float(e.max()),
        "mean": float(e.mean()),
    }


def seed_from(*parts, prefix: str = "exp037") -> int:
    """A 64-bit seed: the first 8 bytes, big-endian, of
    sha256(prefix + "|" + '|'.join(parts)) (the analysis/stats.py rule).

    exp_037 draws under "exp037" (DESIGN §2.10): the G3b bootstrap, the
    router-margin bootstrap and the behaviour cells call it unqualified. The
    registered fixture seeds of exp_036 (the tiny gate texts, the G0 digit
    lines, the tiny behaviour prompts) pass prefix="exp036", so they stay the
    registered ones byte for byte. With prefix="exp036" this is exp_036's
    seed_from(*parts) exactly."""
    return int.from_bytes(hashlib.sha256((prefix + "|" + "|".join(map(str, parts))).encode()).digest()[:8], "big")


# ---------------------------------------------------------------------------
# Port loading (mlx imported lazily)
# ---------------------------------------------------------------------------

_counter = itertools.count()
PORT_MODULE_PREFIX = "exp037_gate_port_"  # + a counter: the __name__ of each module exec_port_module() runs


def exec_port_module(port_file: Path = PORT_FILE):
    """A fresh module object executed from port_file. Its globals are private
    to the returned module, so patches (gate/port_mutants.py) affect only the
    models built from it, exactly as mlx_lm's own model_file loading does."""
    sys.dont_write_bytecode = True
    name = f"{PORT_MODULE_PREFIX}{next(_counter)}"
    spec = importlib.util.spec_from_file_location(name, str(port_file))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def port_module(mutant: str | None = None, port_file: Path = PORT_FILE):
    if mutant is None:
        return exec_port_module(port_file)
    from gate import port_mutants

    return port_mutants.mutant_module(mutant, port_file)


def load_port(model_dir: Path, mutant: str | None = None, lazy: bool = False, strict: bool = True,
              port_file: Path = PORT_FILE, module=None):
    """(model, config, module): model_dir loaded through mlx_lm.utils.load_model
    with the classes of `port_file` (optionally mutated). A converted directory's
    own kolibri1.py copy is bypassed; run_gate first requires that copy to equal
    port_file (convert.check_port_file), so the code is the same bytes."""
    set_offline_env()
    from mlx_lm.utils import load_model

    module = module or port_module(mutant, port_file)
    model, config = load_model(
        Path(model_dir),
        lazy=lazy,
        strict=strict,
        model_config={"model_file": None},
        get_model_classes=lambda config: (module.Model, module.ModelArgs),
        # mlx-lm 0.32.0 (#1385, CVE-2026-5843) executes a config's model_file only
        # with trust_remote_code=True. Inert here: model_config={"model_file": None}
        # routes construction through `module` (exp_037's port, DESIGN §6.5), so the
        # directory's kolibri1.py never runs. Passed so no kit load relies on
        # mlx-lm's default.
        trust_remote_code=True,
    )
    return model, config, module


def mx_to_np(a) -> np.ndarray:
    import mlx.core as mx

    if a.dtype == mx.bfloat16:
        a = a.astype(mx.float32)
    return np.array(a)


def load_raw_tokenizer(tok_dir: Path):
    from tokenizers import Tokenizer

    return Tokenizer.from_file(str(Path(tok_dir) / "tokenizer.json"))


def peak_rss_bytes() -> int:
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) * 1024


def safetensors_header(path) -> dict:
    """{name: {"dtype", "shape", "data_offsets"}} of one .safetensors file
    (the 8-byte little-endian length, then the JSON header); no tensor data."""
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header


def checkpoint_bytes(model_dir) -> int:
    """Bytes of a directory's model*.safetensors shards: what one layer-streamed
    reference pass reads (about 156 GB for Kolibri-1-BF16)."""
    return sum(p.stat().st_size for p in Path(model_dir).glob("model*.safetensors"))
