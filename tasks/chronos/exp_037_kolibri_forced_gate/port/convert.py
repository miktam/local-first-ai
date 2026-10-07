#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Part of Chronos exp_036 (Miktam, 2026-10-03). Copies port/kolibri1.py
# (Apache-2.0, see that file's header) into every converted checkpoint.
# Modified by Miktam for Chronos exp_037, 2026-10-06: mlx-lm 0.32.0 loads (trust_remote_code, model_file_trust).
"""Convert the Kolibri-1 BF16 checkpoint into an affine-quantised MLX checkpoint.

    python port/convert.py --src $EXP036_MODELS/Kolibri-1-BF16 \
        --out $EXP036_MODELS/Kolibri-1-MLX-8bit-g64 --bits 8 --group-size 64

Policy (BUILD_SPEC 5.1, HYPOTHESIS "Quantisation"): attention, routed and shared
experts, embed_tokens and lm_head are quantised (affine, --bits, --group-size,
bf16 scales and biases); the router weight and expert_bias stay fp32 and the
norms bf16. --no-quantize-embeddings / --no-quantize-lm-head give the
vendor-faithful variant (embedding bf16, head an exact fp32 upcast).

No network. The source directory is never modified. Steps:
  1. check the source: model_type kolibri1, BF16 (FP8 refused), every shard
     named by the index present and complete;
  2. build `<out>.staging`: symlinks to the shards, index, tokenizer files and
     generation_config.json, a copy of port/kolibri1.py, and config.json with
     "model_file": "kolibri1.py" plus the two policy keys
     "exp036_quantize_embeddings" / "exp036_quantize_lm_head" (read by the
     port's sanitize, cast_predicate and quant_predicate; nothing else changed);
  3. convert into `<out>.partial`: mlx_lm.convert.convert with the port's own
     quant predicate (affine, --bits, --group-size, dtype bfloat16, no recipe),
     or, with --streaming, convert_streaming.py layer by layer (same tensors);
     a memory guard samples swap and the process footprint every 10 s and
     stops the conversion when swap grows by more than 2 GB or the footprint
     passes 0.8 x RAM, removes `.partial` and prints the --streaming command;
  4. put the source tokenizer files back byte for byte (mlx_lm's
     save_pretrained under transformers 5 rewrites them and renumbers the
     reserved special tokens), verify the output against the policy, write
     exp036_convert_record.json, rename `<out>.partial` to `<out>`;
  5. remove the staging dir; the CLI also writes a copy of the record to
     results/convert/convert_<bits>bit_<UTC>.json (after the git identity check).
An interrupted run leaves `<out>.staging` / `<out>.partial`; the next run
removes them (after checking they contain only what this script writes).

mlx_lm executes the converted directory's own copy of kolibri1.py, not
port/kolibri1.py. A later fix to the port therefore never reaches an existing
build by itself:

    python port/convert.py --check-port-file --out DIR     # exit 1 if stale
    python port/convert.py --refresh-port-file --out DIR   # recopy + record

check_port_file() is what every run and gate entry point must call before a
measured run. The refresh is for forward-code fixes only: the weights do not
depend on kolibri1.py unless sanitize or the quant predicate changed, and
then the build must be converted again.

mlx-lm 0.32.0 (exp_037, decision F2) executes a config's model_file only when
the loader passes trust_remote_code=True (mlx-lm #1385, CVE-2026-5843).
convert() and refresh_port_file() pass it for the directory they have just
written themselves (the staging dir and the refreshed build). Every other
load of a converted directory passes **model_file_trust(dir), which grants
the flag only for model_file "kolibri1.py" after check_port_file() has
passed, returns {} for a directory without a model_file, and raises for
anything else, before any code runs. A source whose config.json or
tokenizer_config.json declares an "auto_map" (code of its own) is refused,
because mlx_lm hands the same flag to transformers' tokenizer loader.
"""

import _thread
import argparse
import ctypes
import datetime
import fnmatch
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import os
import resource
import shlex
import shutil
import struct
import subprocess
import sys
import threading
from pathlib import Path

# No network, ever: huggingface_hub / transformers read these when first
# imported (inside convert()), which also stops transformers' Hub lookups.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

PORT_DIR = Path(__file__).resolve().parent
EXP_DIR = PORT_DIR.parent
PORT_FILE = PORT_DIR / "kolibri1.py"
STREAMING_FILE = PORT_DIR / "convert_streaming.py"
MODEL_FILE_NAME = "kolibri1.py"
RECORD_NAME = "exp036_convert_record.json"
INDEX_NAME = "model.safetensors.index.json"
RECORD_SCHEMA = "exp036-convert-record-2"

# Config keys convert.py adds to the staging config (read by kolibri1.ModelArgs).
POLICY_KEYS = ("exp036_quantize_embeddings", "exp036_quantize_lm_head")

TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
)
# What an MLX output directory written by this script may contain; --force and
# the stale `.partial` clean-up refuse to delete anything else.
OUTPUT_PATTERNS = (
    "model*.safetensors",
    "*.json",
    "*.py",
    "*.jinja",
    "*.txt",
    "*.md",
    "tokenizer.model",
)

# Memory guard (BUILD_SPEC 5.1 "Self-monitoring").
MONITOR_INTERVAL_S = 10.0
SWAP_GROWTH_LIMIT_BYTES = 2 * 10**9
FOOTPRINT_LIMIT_FRACTION = 0.8
# After asking the main thread to stop, how long the guard waits before it
# removes the partial output itself and ends the process.
MONITOR_GRACE_S = 120.0


def utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def utc_iso(t: datetime.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def read_safetensors_header(path: Path) -> dict:
    """(tensor header dict, byte offset where the data starts) of a .safetensors
    file: 8-byte little-endian header length, then the JSON header."""
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
    header.pop("__metadata__", None)
    return header, 8 + n


def read_all_headers(model_dir: Path) -> dict:
    """name -> {dtype, shape, data_offsets} over every model*.safetensors."""
    tensors = {}
    for shard in sorted(Path(model_dir).glob("model*.safetensors")):
        header, _ = read_safetensors_header(shard)
        tensors.update(header)
    return tensors


# ---------------------------------------------------------------------------
# Paths in records
# ---------------------------------------------------------------------------


def _kit_module(name: str):
    """A module of the kit (e.g. "tools.redact", "runner.guard"), imported as
    a package module with the experiment directory on sys.path; None if that
    file does not exist (yet)."""
    if not (EXP_DIR / (name.replace(".", "/") + ".py")).is_file():
        return None
    if str(EXP_DIR) not in sys.path:
        sys.path.insert(0, str(EXP_DIR))
    return importlib.import_module(name)


def _load_redactor():
    """tools/redact.py:redact_path when the kit has it (BUILD_SPEC 2,
    "Privacy"); otherwise the local fallback in display_path, which applies
    the same rewrites."""
    try:
        module = _kit_module("tools.redact")
    except Exception:  # a broken sibling must not break a conversion record
        return None
    return getattr(module, "redact_path", None) if module is not None else None


def display_path(path) -> str:
    """A path for a record: $EXP036_WORK, $EXP036_PRIVATE, $EXP036_DATA,
    $EXP036_MODELS and $HOME prefixes are written as the variable, so records
    carry no home path or user name."""
    redact = _load_redactor()
    if redact is not None:
        return str(redact(str(path)))
    path = Path(path).expanduser().absolute()
    # Most specific first: DATA, PRIVATE and WORK default to folders under MODELS.
    for var in ("EXP036_WORK", "EXP036_PRIVATE", "EXP036_DATA", "EXP036_MODELS", "HOME"):
        root = os.environ.get(var)
        if not root:
            continue
        root = Path(root).expanduser().absolute()
        if path == root:
            return f"${var}"
        if root in path.parents:
            return f"${var}/" + str(path.relative_to(root))
    return str(path)


# ---------------------------------------------------------------------------
# Directory hygiene
# ---------------------------------------------------------------------------


def _looks_like(directory: Path, patterns, allow_symlinks: bool) -> bool:
    for entry in directory.iterdir():
        if entry.name == "__pycache__" and not entry.is_symlink() and entry.is_dir():
            # Python's bytecode cache for kolibri1.py, written whenever mlx_lm
            # loads the model from this directory.
            if all(p.is_file() and p.suffix == ".pyc" for p in entry.iterdir()):
                continue
            return False
        if entry.is_symlink():
            if not allow_symlinks:
                return False
        elif not entry.is_file():
            return False
        if not any(fnmatch.fnmatch(entry.name, p) for p in patterns):
            return False
    return True


def _remove_own_dir(directory: Path, patterns, allow_symlinks: bool, what: str) -> None:
    if not directory.exists():
        return
    if not directory.is_dir() or not _looks_like(directory, patterns, allow_symlinks):
        raise SystemExit(
            f"refusing to delete {directory}: it is not a {what} written by this script"
        )
    shutil.rmtree(directory)


# ---------------------------------------------------------------------------
# Source checks and staging
# ---------------------------------------------------------------------------


def check_source(src: Path) -> dict:
    """Validate the BF16 source; return its config, shard list and tensor shapes."""
    config_path = src / "config.json"
    if not config_path.is_file():
        raise SystemExit(f"{src}: no config.json")
    config = json.loads(config_path.read_text())
    if config.get("model_type") != "kolibri1":
        raise SystemExit(f"{src}: model_type is {config.get('model_type')!r}, expected 'kolibri1'")
    if "quantization_config" in config or "quantization" in config:
        raise SystemExit(
            f"{src}: the config has a quantization block. This is the FP8 repo "
            "(Aleph-Alpha/Kolibri-1) or an already converted model; convert from "
            "Aleph-Alpha/Kolibri-1-BF16 only."
        )
    dtype = config.get("dtype", config.get("torch_dtype"))
    if dtype not in (None, "bfloat16"):
        raise SystemExit(f"{src}: dtype {dtype!r}, expected bfloat16")
    # convert() loads the staging dir with trust_remote_code=True (mlx-lm 0.32.0),
    # and mlx_lm passes the same flag to transformers' AutoTokenizer. Our own
    # kolibri1.py is the only code that flag may reach.
    for name in ("config.json", "tokenizer_config.json"):
        path = src / name
        if path.is_file() and "auto_map" in json.loads(path.read_text()):
            raise SystemExit(
                f"{src}: {name} declares an auto_map (code shipped with the checkpoint). The "
                "conversion loads with trust_remote_code=True for kolibri1.py and would run it; "
                "convert from Aleph-Alpha/Kolibri-1-BF16, which ships no code."
            )

    shards = sorted(p.name for p in src.glob("model*.safetensors"))
    if not shards:
        raise SystemExit(f"{src}: no model*.safetensors")

    index = None
    if (src / INDEX_NAME).is_file():
        index = json.loads((src / INDEX_NAME).read_text())
        named = sorted(set(index["weight_map"].values()))
        if named != shards:
            missing = sorted(set(named) - set(shards))
            extra = sorted(set(shards) - set(named))
            raise SystemExit(
                f"{src}: shards do not match {INDEX_NAME} "
                f"(missing {missing[:3]}{'...' if len(missing) > 3 else ''}, extra {extra[:3]}). "
                "Is the download complete?"
            )
    elif len(shards) > 1:
        raise SystemExit(f"{src}: {len(shards)} shards but no {INDEX_NAME}")

    # Every tensor BF16 (F32 tolerated), every shard complete on disk.
    shapes = {}
    for shard in shards:
        path = src / shard
        header, data_start = read_safetensors_header(path)
        end = 0
        for name, info in header.items():
            if name.endswith("weight_scale_inv") or info["dtype"].startswith("F8"):
                raise SystemExit(
                    f"{path}: {name} is {info['dtype']}. FP8 checkpoints are not "
                    "supported; use Aleph-Alpha/Kolibri-1-BF16."
                )
            if info["dtype"] not in ("BF16", "F32"):
                raise SystemExit(f"{path}: {name} has dtype {info['dtype']}, expected BF16")
            end = max(end, info["data_offsets"][1])
            shapes[name] = tuple(info["shape"])
        if path.stat().st_size != data_start + end:
            raise SystemExit(
                f"{path}: {path.stat().st_size} bytes on disk, header says "
                f"{data_start + end}. Truncated or still downloading?"
            )
    if index is not None and set(shapes) != set(index["weight_map"]):
        raise SystemExit(f"{src}: shard headers and {INDEX_NAME} name different tensors")

    return {"config": config, "shards": shards, "has_index": index is not None, "shapes": shapes}


def source_revision(src: Path) -> dict:
    """The HF commit the source was downloaded at, from the metadata files
    `hf download --local-dir` leaves in <src>/.cache/huggingface/download/
    (first line: commit hash; second: etag). No network."""
    meta_dir = src / ".cache" / "huggingface" / "download"
    commits = {}
    if meta_dir.is_dir():
        for meta in sorted(meta_dir.rglob("*.metadata")):
            lines = meta.read_text(errors="replace").splitlines()
            if lines and lines[0].strip():
                rel = str(meta.relative_to(meta_dir))[: -len(".metadata")]
                commits[rel] = lines[0].strip()
    distinct = sorted(set(commits.values()))
    return {
        "commit": distinct[0] if len(distinct) == 1 else None,
        "commits_distinct": distinct,
        "files_with_metadata": len(commits),
        "source": ".cache/huggingface/download/*.metadata" if commits else "none found",
    }


def build_staging(src: Path, staging: Path, config: dict, shards: list, has_index: bool,
                  quantize_embeddings: bool = True, quantize_lm_head: bool = True) -> None:
    staging.mkdir(parents=True)
    links = list(shards)
    if has_index:
        links.append(INDEX_NAME)
    links += [n for n in TOKENIZER_FILES if (src / n).is_file()]
    if (src / "generation_config.json").is_file():
        links.append("generation_config.json")
    for name in links:
        os.symlink(src / name, staging / name)
    shutil.copyfile(PORT_FILE, staging / MODEL_FILE_NAME)
    # The config edits: register the port (spec item 17) and carry the head /
    # embedding policy to the port's sanitize, cast and quant predicates.
    edited = dict(config)
    edited["model_file"] = MODEL_FILE_NAME
    edited["exp036_quantize_embeddings"] = bool(quantize_embeddings)
    edited["exp036_quantize_lm_head"] = bool(quantize_lm_head)
    (staging / "config.json").write_text(json.dumps(edited, indent=2) + "\n")


def load_port_module(port_file=PORT_FILE, name: str = "exp036_kolibri1_port"):
    spec = importlib.util.spec_from_file_location(name, port_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_streaming_module():
    spec = importlib.util.spec_from_file_location("exp036_convert_streaming", STREAMING_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def restore_tokenizer_files(src: Path, work: Path) -> None:
    """Replace what tokenizer.save_pretrained wrote with the source files."""
    for name in TOKENIZER_FILES:
        if (work / name).exists():
            (work / name).unlink()
    for name in TOKENIZER_FILES:
        if (src / name).is_file():
            shutil.copyfile(src / name, work / name)


# ---------------------------------------------------------------------------
# Tensor policy
# ---------------------------------------------------------------------------


def head_policy_name(quantize_embeddings: bool, quantize_lm_head: bool) -> str:
    """The names gate/thresholds.json uses for the two signed-off variants."""
    if quantize_embeddings and quantize_lm_head:
        return "quantised_head"
    if not quantize_embeddings and not quantize_lm_head:
        return "vendor_faithful"
    return "custom"


def expected_tensor_policy(quantize_embeddings: bool, quantize_lm_head: bool) -> dict:
    """What each class of tensor must be. "arm_bits" = affine-quantised at the
    build's bits and group size with bf16 scales and biases; otherwise the
    stored dtype. Vocabulary as in gate/thresholds.json G0.tensor_policy."""
    return {
        "attention": "arm_bits",
        "routed_experts": "arm_bits",
        "shared_experts": "arm_bits",
        "embed_tokens": "arm_bits" if quantize_embeddings else "bf16",
        "lm_head": "arm_bits" if quantize_lm_head else "fp32",
        "router_weight": "fp32",
        "expert_bias": "fp32",
        "norms": "bf16",
    }


_DTYPE_NAMES = {"F32": "fp32", "BF16": "bf16", "F16": "fp16", "U32": "u32"}


def observed_tensor_policy(tensors: dict, num_layers: int) -> dict:
    """The tensor policy read back from saved safetensors headers (tensors:
    name -> {dtype, shape}), in the vocabulary of expected_tensor_policy.
    A class whose members disagree is reported as "mixed"."""

    def kind(module: str) -> str:
        if f"{module}.scales" in tensors:
            return "arm_bits"
        info = tensors.get(f"{module}.weight")
        if info is None:
            return "missing"
        return _DTYPE_NAMES.get(info["dtype"], info["dtype"])

    def one(kinds) -> str:
        kinds = set(kinds)
        return kinds.pop() if len(kinds) == 1 else "mixed"

    layers = [f"model.layers.{i}" for i in range(num_layers)]
    norms = []
    for p in layers:
        for n in ("input_layernorm", "post_attn_norm", "post_attention_layernorm",
                  "post_ffn_norm", "self_attn.q_norm", "self_attn.k_norm"):
            norms.append(kind(f"{p}.{n}"))
    norms.append(kind("model.norm"))
    bias = [
        _DTYPE_NAMES.get(tensors[f"{p}.mlp.gate.expert_bias"]["dtype"], "?")
        if f"{p}.mlp.gate.expert_bias" in tensors else "missing"
        for p in layers
    ]
    return {
        "attention": one(kind(f"{p}.self_attn.{m}") for p in layers
                         for m in ("q_proj", "k_proj", "v_proj", "o_proj")),
        "routed_experts": one(kind(f"{p}.mlp.switch_mlp.{m}") for p in layers
                              for m in ("gate_proj", "up_proj", "down_proj")),
        "shared_experts": one(kind(f"{p}.mlp.shared_experts.{m}") for p in layers
                              for m in ("gate_proj", "up_proj", "down_proj")),
        "embed_tokens": kind("model.embed_tokens"),
        "lm_head": kind("lm_head"),
        "router_weight": one(kind(f"{p}.mlp.gate") for p in layers),
        "expert_bias": one(bias),
        "norms": one(norms),
    }


def verify_output(src: Path, work: Path, bits: int, group_size: int,
                  quantize_embeddings: bool, quantize_lm_head: bool) -> dict:
    """Check the converted directory against the policy; return the observed
    tensor policy (which then equals the expected one)."""
    config = json.loads((work / "config.json").read_text())
    if config.get("model_file") != MODEL_FILE_NAME:
        raise SystemExit(f"{work}: config.json lacks model_file={MODEL_FILE_NAME}")
    quant = config.get("quantization")
    if not isinstance(quant, dict) or quant.get("bits") != bits or quant.get("group_size") != group_size:
        raise SystemExit(f"{work}: config.json quantization block is {quant!r}")
    if quant.get("mode", "affine") != "affine" or "quantization_config" not in config:
        raise SystemExit(f"{work}: unexpected quantization mode or no quantization_config")
    if (config.get("exp036_quantize_embeddings"), config.get("exp036_quantize_lm_head")) != (
        quantize_embeddings, quantize_lm_head
    ):
        raise SystemExit(f"{work}: config.json policy keys do not match the requested policy")
    if sha256_file(work / MODEL_FILE_NAME) != sha256_file(PORT_FILE):
        raise SystemExit(f"{work}: {MODEL_FILE_NAME} differs from {PORT_FILE}")
    for name in TOKENIZER_FILES + ("generation_config.json",):
        if (src / name).is_file() and sha256_file(src / name) != sha256_file(work / name):
            raise SystemExit(f"{work}: {name} differs from the source")

    tensors = read_all_headers(work)
    problems = []
    observed = observed_tensor_policy(tensors, config["num_hidden_layers"])
    expected = expected_tensor_policy(quantize_embeddings, quantize_lm_head)
    for key, want in expected.items():
        if observed[key] != want:
            problems.append(f"tensor class {key}: {observed[key]}, expected {want}")
    # Every quantised tensor: uint32 packed weight, bf16 scales and biases.
    for name, info in tensors.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            if info["dtype"] != "BF16":
                problems.append(f"{name} is {info['dtype']}, expected BF16")
            module = name.rsplit(".", 1)[0]
            if tensors.get(f"{module}.weight", {}).get("dtype") != "U32":
                problems.append(f"{module}.weight is not a packed U32 tensor")
            other = ".biases" if name.endswith(".scales") else ".scales"
            if f"{module}{other}" not in tensors:
                problems.append(f"{module} has {name.rsplit('.', 1)[1]} but no {other[1:]}")
    if any(".mlp.experts." in k or ".moe.router." in k for k in tensors):
        problems.append("unsanitised checkpoint names in the output")
    if problems:
        raise SystemExit(f"{work}: output check failed:\n  " + "\n  ".join(problems[:20]))
    return observed


# ---------------------------------------------------------------------------
# Output manifest
# ---------------------------------------------------------------------------


def manifest_lines(files: dict) -> str:
    """The manifest text: one "relpath<TAB>sha256" line per file, sorted by
    path, each ending in a newline."""
    return "".join(f"{name}\t{files[name]}\n" for name in sorted(files))


def manifest_sha256(files: dict) -> str:
    """sha256 over manifest_lines(files); files maps relpath -> sha256 hex, or
    relpath -> {"sha256": ...} as in a record's "files"."""
    flat = {k: (v["sha256"] if isinstance(v, dict) else v) for k, v in files.items()}
    return hashlib.sha256(manifest_lines(flat).encode("utf-8")).hexdigest()


def output_files(model_dir) -> dict:
    """relpath -> sha256 for every regular file of a converted directory except
    the record itself and Python's __pycache__ (what manifest_sha256 covers)."""
    model_dir = Path(model_dir)
    out = {}
    for path in sorted(model_dir.rglob("*")):
        rel = path.relative_to(model_dir)
        if not path.is_file() or rel.parts[0] == "__pycache__" or str(rel) == RECORD_NAME:
            continue
        out[str(rel)] = sha256_file(path)
    return out


def directory_manifest_sha256(model_dir) -> str:
    """manifest_sha256 recomputed from the files on disk (hashes every byte:
    minutes on an 84 GB build)."""
    return manifest_sha256(output_files(model_dir))


# ---------------------------------------------------------------------------
# Memory guard
# ---------------------------------------------------------------------------


class _RusageInfoV4(ctypes.Structure):
    # struct rusage_info_v4, <sys/resource.h> (macOS). Only the footprint
    # fields are read; the struct is declared in full so the offsets are right.
    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups",
            "ri_pageins", "ri_wired_size", "ri_resident_size", "ri_phys_footprint",
            "ri_proc_start_abstime", "ri_proc_exit_abstime", "ri_child_user_time",
            "ri_child_system_time", "ri_child_pkg_idle_wkups", "ri_child_interrupt_wkups",
            "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive",
            "ri_billed_system_time", "ri_serviced_system_time", "ri_logical_writes",
            "ri_lifetime_max_phys_footprint", "ri_instructions", "ri_cycles", "ri_billed_energy",
            "ri_serviced_energy", "ri_interval_max_phys_footprint", "ri_runnable_time",
        )
    ]


def _proc_footprint():
    """(current, lifetime max) physical footprint of this process in bytes,
    or None off macOS. The footprint counts Metal buffers (MLX arrays), which
    `ps` RSS and getrusage's ru_maxrss do not."""
    if sys.platform != "darwin":
        return None
    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
        info = _RusageInfoV4()
        if libc.proc_pid_rusage(os.getpid(), 4, ctypes.byref(info)) != 0:  # RUSAGE_INFO_V4
            return None
        return int(info.ri_phys_footprint), int(info.ri_lifetime_max_phys_footprint)
    except (OSError, AttributeError):
        return None


def footprint_bytes() -> int:
    """Current memory of this process: the macOS physical footprint, or the
    peak RSS from getrusage elsewhere."""
    fp = _proc_footprint()
    if fp is not None:
        return fp[0]
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) * 1024


def peak_rss_bytes() -> int:
    """getrusage ru_maxrss in bytes (CPU-side pages only; Metal buffers are
    not included, see peak_footprint_bytes)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r) if sys.platform == "darwin" else int(r) * 1024


def peak_footprint_bytes() -> int:
    """Lifetime maximum physical footprint (macOS; includes MLX's Metal
    buffers), or the peak RSS elsewhere."""
    fp = _proc_footprint()
    return max(peak_rss_bytes(), fp[1] if fp is not None else 0)


def swap_used_bytes():
    """`sysctl -n vm.swapusage` "used = 668.06M" in bytes; None if unavailable."""
    try:
        text = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True,
                              text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for part in text.split("  "):
        part = part.strip()
        if part.startswith("used"):
            value = part.split("=", 1)[1].strip()
            scale = {"K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}.get(value[-1].upper())
            if scale is None:
                return None
            return int(float(value[:-1]) * scale)
    return None


def memsize_bytes():
    try:
        return int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                                  text=True, timeout=10).stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


class MemoryGuard:
    """Samples swap use and the process footprint every `interval` seconds on
    a daemon thread. When swap has grown by more than `swap_growth_limit`
    bytes since start, or the footprint exceeds `footprint_fraction` x RAM,
    it records the reason, interrupts the main thread (KeyboardInterrupt)
    and, if the main thread has not acknowledged within `grace` seconds,
    calls `on_hard_abort` (convert() passes a clean-up that ends the process).

    Readers are injectable for the tests."""

    def __init__(self, interval=MONITOR_INTERVAL_S, swap_growth_limit=SWAP_GROWTH_LIMIT_BYTES,
                 footprint_fraction=FOOTPRINT_LIMIT_FRACTION, read_swap=swap_used_bytes,
                 read_footprint=footprint_bytes, memsize=None, grace=MONITOR_GRACE_S,
                 on_hard_abort=None, interrupt_main=True):
        self.interval = float(interval)
        self.swap_growth_limit = int(swap_growth_limit)
        self.read_swap = read_swap
        self.read_footprint = read_footprint
        self.memsize = memsize if memsize is not None else memsize_bytes()
        self.footprint_limit = (
            int(footprint_fraction * self.memsize) if self.memsize else None
        )
        self.grace = float(grace)
        self.on_hard_abort = on_hard_abort
        self.interrupt_main = interrupt_main
        self.tripped = None  # reason string once the limit is crossed
        self.acknowledged = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self.swap_start = None
        self.swap_max = None
        self.footprint_max = 0
        self.samples = 0

    def _sample(self) -> None:
        swap = self.read_swap()
        fp = self.read_footprint()
        self.samples += 1
        if swap is not None:
            if self.swap_start is None:
                self.swap_start = swap
            self.swap_max = swap if self.swap_max is None else max(self.swap_max, swap)
        if fp is not None:
            self.footprint_max = max(self.footprint_max, int(fp))
        if self.tripped is not None:
            return
        if swap is not None and swap - self.swap_start > self.swap_growth_limit:
            self.tripped = (f"swap grew by {(swap - self.swap_start) / 1e9:.2f} GB "
                            f"(limit {self.swap_growth_limit / 1e9:.1f} GB)")
        elif fp is not None and self.footprint_limit and fp > self.footprint_limit:
            self.tripped = (f"process footprint {fp / 1e9:.1f} GB exceeds "
                            f"{self.footprint_limit / 1e9:.1f} GB (0.8 x RAM)")

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample()
            if self.tripped is not None:
                if self.interrupt_main:
                    _thread.interrupt_main()
                if not self.acknowledged.wait(self.grace) and self.on_hard_abort is not None:
                    self.on_hard_abort(self.tripped)
                return
            self._stop.wait(self.interval)

    def start(self) -> "MemoryGuard":
        self._sample()  # baseline swap before anything is allocated
        self._thread = threading.Thread(target=self._run, name="exp036-memory-guard", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> dict:
        self._stop.set()
        self.acknowledged.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        growth = None
        if self.swap_start is not None and self.swap_max is not None:
            growth = max(0, self.swap_max - self.swap_start)
        return {
            "samples": self.samples,
            "interval_s": self.interval,
            "swap_used_start_bytes": self.swap_start,
            "swap_growth_bytes": growth,
            "swap_growth_limit_bytes": self.swap_growth_limit,
            "footprint_max_sampled_bytes": self.footprint_max,
            "footprint_limit_bytes": self.footprint_limit,
            "tripped": self.tripped,
        }


def streaming_command(src: Path, out: Path, bits: int, group_size: int,
                      quantize_embeddings: bool, quantize_lm_head: bool) -> str:
    parts = ["python", "port/convert.py", "--src", display_path(src), "--out", display_path(out),
             "--bits", str(bits), "--group-size", str(group_size)]
    if not quantize_embeddings:
        parts.append("--no-quantize-embeddings")
    if not quantize_lm_head:
        parts.append("--no-quantize-lm-head")
    parts.append("--streaming")
    # $VARS stay unquoted so the shell expands them.
    return " ".join(p if p.startswith("$") else shlex.quote(p) for p in parts)


# ---------------------------------------------------------------------------
# Conversion
# ---------------------------------------------------------------------------


def convert(src, out, bits: int, group_size: int = 64, quantize_embeddings: bool = True,
            quantize_lm_head: bool = True, force: bool = False, streaming: bool = False,
            guard: "MemoryGuard | None | bool" = True) -> dict:
    """Convert `src` (Kolibri-1-BF16 directory) into `out`. Returns the record.

    guard: True (default) runs a MemoryGuard with the BUILD_SPEC limits;
    False runs without one; a MemoryGuard instance is used as given (tests)."""
    t_start = utc_now()
    src = Path(src).expanduser().resolve()
    out = Path(out).expanduser().absolute()
    if bits not in (2, 3, 4, 5, 6, 8):
        raise SystemExit(f"--bits {bits}: affine supports 2, 3, 4, 5, 6, 8")
    if group_size not in (32, 64, 128):
        raise SystemExit(f"--group-size {group_size}: use 32, 64 or 128")
    if out == src or src in out.parents:
        raise SystemExit("--out must not be the source directory or inside it")

    if out.exists() and (not out.is_dir() or any(out.iterdir())) and not force:
        raise SystemExit(f"{out} exists and is not empty; pass --force to replace it")

    staging = out.parent / (out.name + ".staging")
    work = out.parent / (out.name + ".partial")

    print(f"[exp036] checking source {src}")
    info = check_source(src)
    src_bytes = sum((src / s).stat().st_size for s in info["shards"])

    # Clean up after an interrupted run, then the previous output if --force.
    _remove_own_dir(staging, ("*",), allow_symlinks=True, what="staging dir")
    _remove_own_dir(work, OUTPUT_PATTERNS, allow_symlinks=False, what="partial output")
    if out.exists():
        if out.is_dir() and not any(out.iterdir()):
            out.rmdir()
        else:
            print(f"[exp036] --force: removing previous output {out}")
            _remove_own_dir(out, OUTPUT_PATTERNS, allow_symlinks=False, what="converted model")

    # Rough size of the result: b bits + a bf16 scale and bias per group, per
    # source weight; an unquantised head is fp32 (twice its bf16 size) and an
    # unquantised embedding stays bf16. Refuse early rather than at 90 %.
    def bf16_bytes(name):
        shape = info["shapes"].get(name, ())
        n = 1
        for d in shape:
            n *= int(d)
        return 2 * n if shape else 0

    estimate = int(src_bytes * (bits + 32 / group_size) / 16 * 1.05) + (2 << 30)
    if not quantize_lm_head:
        estimate += 2 * bf16_bytes("lm_head.weight")
    if not quantize_embeddings:
        estimate += bf16_bytes("model.embed_tokens.weight")
    out.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(out.parent).free
    if free < estimate:
        raise SystemExit(
            f"{out.parent}: {free / 1e9:.1f} GB free, the output needs about {estimate / 1e9:.1f} GB"
        )

    # mlx_lm executes kolibri1.py from the staging dir; keep it free of __pycache__.
    sys.dont_write_bytecode = True
    port = load_port_module()
    predicate = port.make_quant_predicate(
        group_size=group_size,
        quantize_embeddings=quantize_embeddings,
        quantize_lm_head=quantize_lm_head,
    )
    retry = streaming_command(src, out, bits, group_size, quantize_embeddings, quantize_lm_head)

    def hard_abort(reason):
        # The main thread did not stop within the grace period: clean up from
        # here and end the process (BUILD_SPEC 5.1 "Self-monitoring").
        print(f"[exp036] memory guard: {reason}; the conversion did not stop, ending the process",
              file=sys.stderr, flush=True)
        for d in (work, staging):
            try:
                if d.exists():
                    shutil.rmtree(d)
            except OSError:
                pass
        print(f"[exp036] run instead:\n  {retry}", file=sys.stderr, flush=True)
        os._exit(3)

    if guard is True:
        guard = MemoryGuard(on_hard_abort=hard_abort)
    elif guard is False:
        guard = None
    guard_summary = None

    try:
        build_staging(src, staging, info["config"], info["shards"], info["has_index"],
                      quantize_embeddings, quantize_lm_head)
        print(f"[exp036] converting into {work}" + (" (streaming)" if streaming else ""))
        try:
            if guard is not None:
                guard.start()
            try:
                if streaming:
                    load_streaming_module().write_streaming(
                        staging, work, bits, group_size,
                        quantize_embeddings=quantize_embeddings, quantize_lm_head=quantize_lm_head,
                        port=port,
                    )
                else:
                    from mlx_lm.convert import convert as mlx_convert  # here, so --help works without mlx

                    # mlx-lm 0.32.0 runs the staging dir's model_file only with
                    # trust_remote_code=True (#1385). That kolibri1.py is the copy of
                    # PORT_FILE build_staging() has just written, and check_source()
                    # refused any auto_map the flag would also reach.
                    mlx_convert(
                        hf_path=str(staging),
                        mlx_path=str(work),
                        quantize=True,
                        q_group_size=group_size,
                        q_bits=bits,
                        q_mode="affine",
                        dtype="bfloat16",
                        quant_predicate=predicate,
                        trust_remote_code=True,
                    )
            finally:
                if guard is not None:
                    guard.acknowledged.set()
                    guard_summary = guard.stop()
        except KeyboardInterrupt:
            # The guard's interrupt may land anywhere in the block above; a
            # Ctrl-C from the user (guard not tripped) propagates.
            if guard is None or guard.tripped is None:
                raise
            guard.acknowledged.set()
            guard_summary = guard.stop()
        if guard_summary is not None and guard_summary["tripped"] is not None:
            _remove_own_dir(work, OUTPUT_PATTERNS, allow_symlinks=False, what="partial output")
            raise SystemExit(
                f"[exp036] memory guard: {guard_summary['tripped']}. The conversion was stopped "
                f"and {work.name} removed. Run instead:\n  {retry}"
            )

        restore_tokenizer_files(src, work)
        observed = verify_output(src, work, bits, group_size, quantize_embeddings, quantize_lm_head)

        files = {}
        for path in sorted(work.iterdir()):
            if not path.is_file():
                continue
            files[path.name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        out_config = json.loads((work / "config.json").read_text())
        t_end = utc_now()
        record = {
            "experiment": "exp_036",
            "schema": RECORD_SCHEMA,
            "spec_version": port.SPEC_VERSION,
            "src": display_path(src),
            "src_revision": source_revision(src),
            "src_index_sha256": sha256_file(src / INDEX_NAME) if info["has_index"] else None,
            "src_config_sha256": sha256_file(src / "config.json"),
            "src_bytes": src_bytes,
            "out": display_path(out),
            "out_name": out.name,
            "bits": bits,
            "group_size": group_size,
            "mode": "affine",
            "dtype": "bfloat16",
            "quantization": out_config.get("quantization"),
            "quantize_embeddings": quantize_embeddings,
            "quantize_lm_head": quantize_lm_head,
            "flags": {
                "quantize_embeddings": quantize_embeddings,
                "quantize_lm_head": quantize_lm_head,
                "streaming": streaming,
                "force": force,
            },
            "head_policy": head_policy_name(quantize_embeddings, quantize_lm_head),
            "tensor_policy": observed,
            "scales_biases_dtype": "bfloat16",
            "converter": "convert_streaming.write_streaming" if streaming else "mlx_lm.convert.convert",
            "quant_predicate": "kolibri1.make_quant_predicate (no mlx_lm recipe)",
            "port_sha256": sha256_file(PORT_FILE),
            "tokenizer_files": "byte copies of the source files",
            "mlx": importlib.metadata.version("mlx"),
            "mlx_metal": _version_or_none("mlx-metal"),
            "mlx_lm": importlib.metadata.version("mlx-lm"),
            "python": sys.version.split()[0],
            "t_start": utc_iso(t_start),
            "t_end": utc_iso(t_end),
            "utc": utc_iso(t_end),
            "peak_rss_bytes": peak_rss_bytes(),
            "peak_footprint_bytes": peak_footprint_bytes(),
            "memory_guard": guard_summary,
            "swap_growth_bytes": (guard_summary or {}).get("swap_growth_bytes"),
            "files": files,
            "manifest_sha256": manifest_sha256(files),
        }
        (work / RECORD_NAME).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        os.rename(work, out)
    finally:
        _remove_own_dir(staging, ("*",), allow_symlinks=True, what="staging dir")

    print(f"[exp036] done: {out}")
    return record


def _version_or_none(dist: str):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return None


def _require_identity() -> None:
    """runner/guard.py:require_identity(), which every entry point calls before
    writing under results/ (BUILD_SPEC 2, "Identity")."""
    guard = _kit_module("runner.guard")
    if guard is None:
        raise SystemExit(
            f"{EXP_DIR / 'runner' / 'guard.py'} not found: the git identity cannot be checked before "
            "writing under results/. Pass --results-dir none to skip the results copy."
        )
    guard.require_identity()


def write_results_copy(record: dict, results_dir) -> Path:
    """results/convert/convert_<bits>bit_<UTC>.json: the record, sorted keys.
    Never overwrites an existing file (append-only results tree)."""
    _require_identity()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    stamp = record["t_end"].replace("-", "").replace(":", "")
    path = results_dir / f"convert_{record['bits']}bit_{stamp}.json"
    with open(path, "x", encoding="utf-8") as f:
        f.write(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(f"[exp036] record copy: {display_path(path)}")
    return path


# ---------------------------------------------------------------------------
# Port-file freshness
# ---------------------------------------------------------------------------


class StalePortFile(RuntimeError):
    """A converted directory's kolibri1.py is not the current port/kolibri1.py,
    or does not match its own conversion record."""


def _read_record(model_dir: Path) -> dict:
    path = model_dir / RECORD_NAME
    if not path.is_file():
        raise StalePortFile(f"{model_dir}: no {RECORD_NAME}; not a directory written by convert.py")
    return json.loads(path.read_text())


def check_port_file(model_dir, port_file=PORT_FILE) -> str:
    """Refuse a converted directory whose forward code is stale.

    Requires sha256(<model_dir>/kolibri1.py) == record["port_sha256"] ==
    record["files"]["kolibri1.py"]["sha256"] == sha256(port_file). Returns the
    common sha256; raises StalePortFile otherwise. Reads only kolibri1.py and
    the record, so it takes milliseconds on an 84 GB build."""
    model_dir = Path(model_dir).expanduser()
    record = _read_record(model_dir)
    copy = model_dir / MODEL_FILE_NAME
    if not copy.is_file():
        raise StalePortFile(f"{model_dir}: no {MODEL_FILE_NAME}")
    port_sha = sha256_file(Path(port_file))
    shas = [
        (sha256_file(copy), f"{MODEL_FILE_NAME} in the model dir"),
        (record.get("port_sha256"), "record port_sha256"),
        (record.get("files", {}).get(MODEL_FILE_NAME, {}).get("sha256"), f"record files[{MODEL_FILE_NAME}]"),
        (port_sha, str(port_file)),
    ]
    if any(sha != port_sha for sha, _ in shas):
        lines = "\n  ".join(f"{sha}  {what}" for sha, what in shas)
        raise StalePortFile(
            f"{model_dir}: the port file differs from its record or from the current port:\n  {lines}\n"
            f"Run: python port/convert.py --refresh-port-file --out {display_path(model_dir.resolve())}"
        )
    return port_sha


class UntrustedModelFile(ValueError):
    """A config.json names a model_file other than kolibri1.py: mlx_lm would
    execute code this kit did not write."""


def model_file_trust(model_dir, port_file=PORT_FILE) -> dict:
    """The keyword arguments for an mlx_lm load of `model_dir` under mlx-lm 0.32.0
    (#1385, CVE-2026-5843): `load_model(d, **model_file_trust(d))`.

    * config.json names no model_file: {} (mlx_lm builds a built-in model type;
      the peers);
    * model_file is "kolibri1.py": {"trust_remote_code": True}, but only after
      check_port_file(model_dir, port_file) has passed, i.e. the directory was
      written by convert.py and its kolibri1.py is `port_file`, byte for byte.
      A stale, edited or unrecorded copy raises StalePortFile;
    * any other model_file: UntrustedModelFile.

    Reads config.json, kolibri1.py and the record only; nothing is executed, so
    a refused directory never reaches mlx_lm."""
    model_dir = Path(model_dir).expanduser()
    config = json.loads((model_dir / "config.json").read_text())
    model_file = config.get("model_file")
    if model_file is None:
        return {}
    if model_file != MODEL_FILE_NAME:
        raise UntrustedModelFile(
            f"{display_path(model_dir)}: config.json names model_file={model_file!r}; only "
            f"{MODEL_FILE_NAME!r} written by port/convert.py is trusted"
        )
    check_port_file(model_dir, port_file=port_file)
    return {"trust_remote_code": True}


def refresh_port_file(out, port_file=PORT_FILE) -> dict:
    """Replace only the converted directory's kolibri1.py with `port_file` and
    rewrite its record (port_sha256, spec_version, the kolibri1.py entry,
    manifest_sha256, plus a port_refreshes history entry). Then load the
    directory strictly and lazily through mlx_lm (no weight data read;
    trust_remote_code=True for the file just copied, mlx-lm 0.32.0) to
    confirm the new port's parameter tree still matches the stored weights.
    Returns the new record."""
    out = Path(out).expanduser().absolute()
    port_file = Path(port_file)
    record = _read_record(out)
    config = json.loads((out / "config.json").read_text())
    if config.get("model_file") != MODEL_FILE_NAME:
        raise SystemExit(f"{out}: config.json lacks model_file={MODEL_FILE_NAME}")

    new_sha = sha256_file(port_file)
    old_sha = sha256_file(out / MODEL_FILE_NAME) if (out / MODEL_FILE_NAME).is_file() else None
    if old_sha == new_sha and record.get("port_sha256") == new_sha:
        print(f"[exp036] {out}: {MODEL_FILE_NAME} is already current ({new_sha[:12]})")
        return record

    from mlx_lm.utils import load_model  # here, so --help works without mlx

    module = load_port_module(port_file, "exp036_kolibri1_refresh")

    # Temporary names match OUTPUT_PATTERNS, so --force can still clean up
    # after an interrupted refresh.
    previous = (out / MODEL_FILE_NAME).read_bytes() if old_sha is not None else None
    tmp = out / "kolibri1.refresh.py"
    shutil.copyfile(port_file, tmp)
    os.replace(tmp, out / MODEL_FILE_NAME)
    # A bytecode cache written by an earlier load could be reused if the new
    # file had the same size and mtime second; drop it.
    pycache = out / "__pycache__"
    if pycache.is_dir():
        for pyc in pycache.glob("kolibri1.*.pyc"):
            pyc.unlink()

    # As in convert(): keep the model dir free of a fresh __pycache__.
    sys.dont_write_bytecode = True
    try:
        # mlx-lm 0.32.0 (#1385): the flag lets mlx_lm run the kolibri1.py copied
        # from port_file just above. model_file_trust() cannot vouch for it yet,
        # because the record still names the previous port until this load passes.
        load_model(out, lazy=True, strict=True, trust_remote_code=True)
    except Exception as e:
        # Put the old file back, so the directory still matches its record.
        if previous is not None:
            (out / MODEL_FILE_NAME).write_bytes(previous)
        raise SystemExit(
            f"{out}: the new {MODEL_FILE_NAME} does not load these weights ({e}). "
            "sanitize or the parameter tree changed; convert again instead."
        ) from e

    utc = utc_iso(utc_now())
    record.setdefault("port_refreshes", []).append(
        {"utc": utc, "previous_port_sha256": record.get("port_sha256"), "port_sha256": new_sha}
    )
    record["port_sha256"] = new_sha
    record["spec_version"] = module.SPEC_VERSION
    record["files"][MODEL_FILE_NAME] = {
        "sha256": new_sha,
        "bytes": (out / MODEL_FILE_NAME).stat().st_size,
    }
    record["manifest_sha256"] = manifest_sha256(record["files"])
    tmp = out / "exp036_convert_record.refresh.json"
    tmp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, out / RECORD_NAME)
    print(f"[exp036] {out}: {MODEL_FILE_NAME} {str(old_sha)[:12]} -> {new_sha[:12]}, record rewritten")
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--src", help="Kolibri-1-BF16 directory")
    parser.add_argument("--out", required=True,
                        help="output directory, e.g. $EXP036_MODELS/Kolibri-1-MLX-8bit-g64 (must not "
                             "exist or be empty); with --check-port-file or --refresh-port-file, an "
                             "existing build")
    parser.add_argument("--bits", type=int, choices=(8, 4), help="affine bits: 8 (K8) or 4 (K4)")
    parser.add_argument("--group-size", type=int, default=64, choices=(32, 64, 128),
                        help="affine group size (exp_036: 64)")
    parser.add_argument("--no-quantize-embeddings", action="store_true",
                        help="keep embed_tokens bf16 (vendor-faithful; default: quantised at --bits)")
    parser.add_argument("--no-quantize-lm-head", action="store_true",
                        help="keep lm_head unquantised, stored as an exact fp32 upcast "
                             "(vendor-faithful; default: quantised at --bits, fp32 logits either way)")
    parser.add_argument("--streaming", action="store_true",
                        help="convert layer by layer with convert_streaming.py (the fallback the "
                             "memory guard prints; same tensors)")
    parser.add_argument("--force", action="store_true",
                        help="delete a previous output in --out (before converting) and convert again")
    parser.add_argument("--results-dir", default=str(EXP_DIR / "results" / "convert"),
                        help="where the record copy convert_<bits>bit_<UTC>.json goes "
                             "(default: results/convert of this experiment; 'none' to skip)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check-port-file", action="store_true",
                      help="only check that --out's kolibri1.py is port/kolibri1.py and matches its record")
    mode.add_argument("--refresh-port-file", action="store_true",
                      help="only replace --out's kolibri1.py with port/kolibri1.py and rewrite its record")
    args = parser.parse_args(argv)
    if args.check_port_file:
        try:
            sha = check_port_file(args.out)
        except StalePortFile as e:
            raise SystemExit(str(e))
        print(f"[exp036] {args.out}: {MODEL_FILE_NAME} is current ({sha})")
        return
    if args.refresh_port_file:
        refresh_port_file(args.out)
        return
    if args.src is None or args.bits is None:
        parser.error("--src and --bits are required to convert")
    results_dir = None if args.results_dir.lower() == "none" else args.results_dir
    if results_dir is not None:
        _require_identity()  # fail before a long conversion, not after it
    record = convert(args.src, args.out, args.bits, args.group_size,
                     quantize_embeddings=not args.no_quantize_embeddings,
                     quantize_lm_head=not args.no_quantize_lm_head,
                     force=args.force, streaming=args.streaming)
    if results_dir is not None:
        write_results_copy(record, results_dir)


if __name__ == "__main__":
    main()
