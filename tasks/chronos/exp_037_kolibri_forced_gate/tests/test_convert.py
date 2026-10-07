# SPDX-License-Identifier: MIT
"""port/convert.py on the tiny vendor checkpoint.

Variants: 8-bit g64, 4-bit g64 and 4-bit g32 with the pre-registered head
policy (embed_tokens and lm_head quantised at the arm's bits, BUILD_SPEC 5.1
delta 2), plus 8-bit and 4-bit g64 with the vendor-faithful head
(--no-quantize-embeddings --no-quantize-lm-head).

Structure (spec items 17, 18, 21; BUILD_SPEC 5.1):
  * the output loads through mlx_lm.utils.load_model (config "model_file");
  * config.json has model_file, the two policy keys and the quantization block;
  * router weight and expert_bias are F32 and unquantised, every norm BF16;
    attention, routed experts and the shared expert are quantised; the
    embedding and the head follow the policy; every scale and bias is BF16;
    a quantised head or embedding was quantised from its bf16 values;
  * exp036_convert_record.json lists every output file with its sha256 and
    size, the reproducible manifest_sha256, the observed tensor policy, the
    flags, t_start/t_end and the memory-guard summary;
  * the source directory is left byte-identical;
  * --streaming writes bit-identical tensors (test_streaming_bit_identical);
  * the memory guard stops a conversion, removes .partial and prints the
    --streaming command.

Numerics, in steps (INTEGRATION_LOG.md entries 3 and 5):
  * against the reference run on the *dequantised* weights of the converted
    checkpoint (both through mx.dequantize and through the reference's own
    dequantised mode): this isolates conversion and loading (wrong group
    size, bits, tensor layout, a stray rounding) from the quantisation noise
    itself, and must agree at the bf16 level;
  * against the fp32 reference on the original weights: smoke bounds for
    tiny random weights, derived from the affine noise model. They are NOT
    the real-weight gate.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import port_harness as ph
import tiny_checkpoint as tc
from exp036_helpers import import_sibling

# (bits, group size, quantised head policy); every tiny dim (64, 256) divides by 32 and 64.
VARIANTS = [(8, 64, True), (4, 64, True), (4, 32, True), (8, 64, False), (4, 64, False)]
SEQ_LEN = 160
RECORD = "exp036_convert_record.json"
NORMS = (
    "input_layernorm",
    "post_attn_norm",
    "post_attention_layernorm",
    "post_ffn_norm",
    "self_attn.q_norm",
    "self_attn.k_norm",
)


def _variant_id(v) -> str:
    bits, gs, quantised_head = v
    return f"{bits}bit-g{gs}" + ("" if quantised_head else "-vendor")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_digest(d: Path) -> dict:
    return {p.name: _sha256(p) for p in sorted(d.iterdir()) if p.is_file()}


def _write_minimal_tokenizer(d: Path) -> None:
    """A word-level tokenizer over t0..t1023, so mlx_lm's loader (which loads a
    tokenizer during convert) has one. Offline, no Kolibri files needed; the
    tokenizer plays no part in the logits checked here."""
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {f"t{i}": i for i in range(tc.VOCAB_SIZE)}
    tok = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tok.save(str(d / "tokenizer.json"))
    (d / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "eos_token": f"t{tc.EOS_TOKEN_ID}",
                "pad_token": f"t{tc.PAD_TOKEN_ID}",
            }
        )
    )


def _read_headers(d: Path) -> dict:
    convert = import_sibling("port.convert")
    return convert.read_all_headers(d)


def _quiet(fn, *args, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return fn(*args, **kwargs)


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    """The tiny vendor checkpoint laid out like the HF source: no model_file in
    config.json, no kolibri1.py, a tokenizer next to the weights."""
    import_sibling("port.kolibri1")
    src = tc.write_tiny_checkpoint(tmp_path_factory.mktemp("convert_src"), seed=0, preset="vendor", copy_port=False)
    cfg = json.loads((src / "config.json").read_text())
    cfg.pop("model_file", None)
    (src / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    _write_minimal_tokenizer(src)
    return src


@pytest.fixture(scope="module")
def reference_on_source(source):
    kolibri_ref = import_sibling("reference.kolibri_ref")
    ids = tc.random_ids(SEQ_LEN, seed=3000)
    ref = kolibri_ref.KolibriReference(str(source))
    logits = ref.forward(np.array(ids))
    return SimpleNamespace(ids=ids, logits=logits, routing_ids=[i for _, i in ref.last_routing])


@pytest.fixture(scope="module", params=VARIANTS, ids=_variant_id)
def converted(request, source, tmp_path_factory):
    convert = import_sibling("port.convert")
    bits, group_size, quantised_head = request.param
    before = _tree_digest(source)
    out = tmp_path_factory.mktemp("converted") / f"Kolibri-tiny-MLX-{bits}bit-g{group_size}"
    record = _quiet(convert.convert, source, out, bits, group_size,
                    quantize_embeddings=quantised_head, quantize_lm_head=quantised_head)
    assert _tree_digest(source) == before, "convert.py modified the source directory"
    return SimpleNamespace(bits=bits, group_size=group_size, quantised_head=quantised_head,
                           out=out, record=record)


def test_output_config_and_files(converted):
    out = converted.out
    port_file = Path(import_sibling("port.convert").PORT_FILE)
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["model_file"] == "kolibri1.py"
    assert cfg["quantization"]["bits"] == converted.bits
    assert cfg["quantization"]["group_size"] == converted.group_size
    assert cfg["quantization"].get("mode", "affine") == "affine"
    assert "quantization_config" in cfg
    assert cfg["exp036_quantize_embeddings"] is converted.quantised_head
    assert cfg["exp036_quantize_lm_head"] is converted.quantised_head
    assert (out / "kolibri1.py").read_bytes() == port_file.read_bytes()
    assert not (out.parent / (out.name + ".staging")).exists()
    assert not (out.parent / (out.name + ".partial")).exists()


def test_record_lists_every_file_with_sha256(converted):
    convert = import_sibling("port.convert")
    out = converted.out
    text = (out / RECORD).read_text()
    record = json.loads(text)
    assert text == json.dumps(record, indent=2, sort_keys=True) + "\n"  # sorted keys (BUILD_SPEC 2)
    assert record == converted.record
    assert record["bits"] == converted.bits and record["group_size"] == converted.group_size
    on_disk = {p.name for p in out.iterdir() if p.is_file() and p.name != RECORD}
    assert set(record["files"]) == on_disk
    for name, entry in record["files"].items():
        path = out / name
        assert entry["bytes"] == path.stat().st_size, name
        assert entry["sha256"] == _sha256(path), name
    assert record["port_sha256"] == _sha256(Path(convert.PORT_FILE))
    # The manifest: sha256 over sorted "relpath\tsha256\n" lines, reproducible
    # from the files on disk.
    lines = "".join(f"{n}\t{record['files'][n]['sha256']}\n" for n in sorted(record["files"]))
    assert record["manifest_sha256"] == hashlib.sha256(lines.encode()).hexdigest()
    assert convert.directory_manifest_sha256(out) == record["manifest_sha256"]


def test_record_fields(converted):
    convert = import_sibling("port.convert")
    kolibri1 = import_sibling("port.kolibri1")
    r = converted.record
    q = converted.quantised_head
    assert r["schema"] == convert.RECORD_SCHEMA and r["spec_version"] == kolibri1.SPEC_VERSION
    # exp_037 W1 moves the port's SPEC_VERSION from exp036-port-2 to exp037-port-1 (DESIGN §2.10).
    assert kolibri1.SPEC_VERSION in ("exp036-port-2", "exp037-port-1")
    assert r["quantization"] == {"group_size": converted.group_size, "bits": converted.bits, "mode": "affine"}
    assert r["flags"] == {"quantize_embeddings": q, "quantize_lm_head": q, "streaming": False, "force": False}
    assert r["head_policy"] == ("quantised_head" if q else "vendor_faithful")
    assert r["tensor_policy"] == convert.expected_tensor_policy(q, q)
    assert r["scales_biases_dtype"] == "bfloat16" and r["dtype"] == "bfloat16"
    assert r["converter"] == "mlx_lm.convert.convert"
    assert r["t_start"] <= r["t_end"] and r["t_end"].endswith("Z") and len(r["t_start"]) == 20
    assert r["mlx"] and r["mlx_lm"] and r["python"]
    assert r["peak_footprint_bytes"] >= r["peak_rss_bytes"] > 0
    guard = r["memory_guard"]
    assert guard["tripped"] is None and guard["samples"] >= 1
    assert guard["swap_growth_limit_bytes"] == 2 * 10**9
    assert r["src_revision"]["source"] == "none found"  # the tiny source has no HF metadata
    # Privacy: no home path in the record.
    assert str(Path.home()) not in json.dumps(r)


def test_source_revision_reads_hf_download_metadata(tmp_path):
    convert = import_sibling("port.convert")
    meta = tmp_path / ".cache" / "huggingface" / "download"
    meta.mkdir(parents=True)
    commit = "7a8f290e7858825c3cf5e4c447ba68345de9f1d3"
    for name in ("config.json", "model-00001-of-00002.safetensors"):
        (meta / f"{name}.metadata").write_text(f"{commit}\n\"etag\"\n1727000000.0\n")
    rev = convert.source_revision(tmp_path)
    assert rev["commit"] == commit and rev["files_with_metadata"] == 2
    (meta / "tokenizer.json.metadata").write_text("0000000000000000000000000000000000000000\nx\n")
    assert convert.source_revision(tmp_path)["commit"] is None  # mixed revisions are not one commit


def test_fresh_build_passes_the_port_file_check(converted):
    convert = import_sibling("port.convert")
    assert convert.check_port_file(converted.out) == _sha256(Path(convert.PORT_FILE))


def test_stale_port_file_is_refused_and_refreshed(converted, tmp_path):
    """Review 2026-10-03: mlx_lm runs the build's own copy of kolibri1.py, so
    a later fix to port/kolibri1.py must be detected and copied in.
    "new_port" stands for port/kolibri1.py after such a fix (a comment
    appended; the forward code is the same)."""
    convert = import_sibling("port.convert")
    port_file = Path(convert.PORT_FILE)
    build = tmp_path / "build"
    shutil.copytree(converted.out, build)  # never touch the shared fixture output
    new_port = tmp_path / "new_port.py"
    new_port.write_text(port_file.read_text() + "\n# exp_036 test: a later port fix\n")

    # The port moved on, the build did not.
    with pytest.raises(convert.StalePortFile, match="--refresh-port-file"):
        convert.check_port_file(build, port_file=new_port)
    # The build's copy was edited by hand: it no longer matches its own record.
    copy = build / "kolibri1.py"
    original = copy.read_bytes()
    copy.write_bytes(original + b"\n")
    with pytest.raises(convert.StalePortFile):
        convert.check_port_file(build)
    copy.write_bytes(original)
    convert.check_port_file(build)

    record = _quiet(convert.refresh_port_file, build, port_file=new_port)
    new_sha = _sha256(new_port)
    assert copy.read_bytes() == new_port.read_bytes()
    assert record == json.loads((build / RECORD).read_text())
    assert record["port_sha256"] == new_sha
    assert record["files"]["kolibri1.py"] == {"sha256": new_sha, "bytes": new_port.stat().st_size}
    assert record["port_refreshes"][-1]["previous_port_sha256"] == _sha256(port_file)
    # The manifest follows the new file, and still matches the directory.
    assert record["manifest_sha256"] != converted.record["manifest_sha256"]
    assert record["manifest_sha256"] == convert.directory_manifest_sha256(build)
    # Every other file entry is untouched, and nothing else was left behind.
    assert {n: e for n, e in record["files"].items() if n != "kolibri1.py"} == {
        n: e for n, e in converted.record["files"].items() if n != "kolibri1.py"
    }
    assert {p.name for p in build.iterdir()} == {p.name for p in converted.out.iterdir()}
    assert convert.check_port_file(build, port_file=new_port) == new_sha
    with pytest.raises(convert.StalePortFile):
        convert.check_port_file(build)  # the real port now differs from the build

    # A second refresh is a no-op.
    assert _quiet(convert.refresh_port_file, build, port_file=new_port) == record

    # The CLI, with the real port: --check fails, --refresh restores it.
    with pytest.raises(SystemExit):
        convert.main(["--check-port-file", "--out", str(build)])
    _quiet(convert.main, ["--refresh-port-file", "--out", str(build)])
    _quiet(convert.main, ["--check-port-file", "--out", str(build)])
    assert len(json.loads((build / RECORD).read_text())["port_refreshes"]) == 2


def test_refresh_refuses_a_port_that_does_not_load_the_weights(converted, tmp_path):
    """A port whose parameter tree changed (here: the shared expert renamed)
    needs a new conversion, not a refresh. The old file stays in place, so
    the build still matches its record."""
    convert = import_sibling("port.convert")
    build = tmp_path / "build"
    shutil.copytree(converted.out, build)
    text = Path(convert.PORT_FILE).read_text()
    assert text.count("self.shared_experts = MLP(") == 1
    bad_port = tmp_path / "bad_port.py"
    bad_port.write_text(text.replace("self.shared_experts = MLP(", "self.shared_expert = MLP("))
    with pytest.raises(SystemExit, match="convert again"):
        _quiet(convert.refresh_port_file, build, port_file=bad_port)
    convert.check_port_file(build)
    assert json.loads((build / RECORD).read_text()) == converted.record


# --- mlx-lm 0.32.0: model_file needs trust_remote_code (#1385; exp_037 decision F2) ------


def _marked_port(text: str, marker: Path) -> str:
    """Port source that also writes `marker` when executed: shows whether any
    loader ran the file."""
    return text + f"\nimport pathlib as _exp037_p\n_exp037_p.Path({str(marker)!r}).write_text('executed')\n"


def test_model_file_trust_on_a_tiny_conversion(converted):
    """A fresh conversion is trusted: model_file kolibri1.py, the port's bytes,
    matching its record. mlx-lm 0.32.0 refuses it without the flag."""
    from mlx_lm.utils import load_model

    convert = import_sibling("port.convert")
    trust = convert.model_file_trust(converted.out)
    assert trust == {"trust_remote_code": True}
    assert trust is not convert.model_file_trust(converted.out)  # a fresh dict per call
    with pytest.raises(ValueError, match="trust_remote_code"):
        load_model(converted.out, lazy=True, strict=True)
    model, config = load_model(converted.out, lazy=True, strict=True, **trust)
    assert config["model_file"] == "kolibri1.py" and type(model).__name__ == "Model"


def test_model_file_trust_without_model_file(source, tmp_path):
    """No model_file (the HF source; every peer): nothing to trust, so {}."""
    convert = import_sibling("port.convert")
    assert "model_file" not in json.loads((source / "config.json").read_text())
    assert convert.model_file_trust(source) == {}
    d = tmp_path / "null"
    d.mkdir()
    (d / "config.json").write_text(json.dumps({"model_type": "llama", "model_file": None}))
    assert convert.model_file_trust(d) == {}


def test_model_file_trust_refuses_a_stale_port_file_before_any_load(converted, tmp_path):
    """A build whose kolibri1.py is not the current port (edited copy, or the
    port moved on), or a directory convert.py did not write, is refused by
    model_file_trust itself: the caller's load never starts, so the file never
    runs. The marker shows the file would have run had the flag been given."""
    from mlx_lm.utils import load_model

    convert = import_sibling("port.convert")
    build = tmp_path / "build"
    shutil.copytree(converted.out, build)
    marker = tmp_path / "ran"
    copy = build / "kolibri1.py"
    copy.write_text(_marked_port(copy.read_text(), marker))

    with pytest.raises(convert.StalePortFile, match="--refresh-port-file"):
        load_model(build, lazy=True, strict=True, **convert.model_file_trust(build))
    assert not marker.exists()
    with pytest.raises(ValueError, match="trust_remote_code"):  # mlx-lm's own default refuses too
        load_model(build, lazy=True, strict=True)
    assert not marker.exists()
    # Positive control: given the flag, mlx_lm runs the file.
    load_model(build, lazy=True, strict=True, trust_remote_code=True)
    assert marker.read_text() == "executed"

    # The port moved on, the build did not.
    new_port = tmp_path / "new_port.py"
    new_port.write_text(Path(convert.PORT_FILE).read_text() + "\n# exp_037 test: a later port fix\n")
    with pytest.raises(convert.StalePortFile):
        convert.model_file_trust(converted.out, port_file=new_port)
    assert convert.model_file_trust(converted.out) == {"trust_remote_code": True}

    # A checkpoint with a port copy but no conversion record (tiny_checkpoint.py's
    # writer) is not a convert.py directory: no trust; the tests load such
    # fixtures with an explicit flag (port_harness.load_port).
    written = tc.write_tiny_checkpoint(tmp_path / "tiny", seed=0, preset="vendor", copy_port=True)
    with pytest.raises(convert.StalePortFile, match="not a directory written by convert.py"):
        convert.model_file_trust(written)


@pytest.mark.parametrize("model_file", ["evil.py", "../kolibri1.py", "sub/kolibri1.py", ""])
def test_model_file_trust_refuses_a_foreign_model_file(converted, tmp_path, model_file):
    convert = import_sibling("port.convert")
    build = tmp_path / "build"
    shutil.copytree(converted.out, build)
    marker = tmp_path / "ran"
    (build / "evil.py").write_text(_marked_port("", marker))
    cfg = json.loads((build / "config.json").read_text())
    cfg["model_file"] = model_file
    (build / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    with pytest.raises(convert.UntrustedModelFile, match="only 'kolibri1.py'"):
        convert.model_file_trust(build)
    assert issubclass(convert.UntrustedModelFile, ValueError)
    assert not marker.exists()


def test_convert_and_refresh_pass_trust_remote_code(source, tmp_path, monkeypatch):
    """The two loads of a directory convert.py has just written pass the flag
    themselves (mlx_convert and refresh_port_file's verification load), and
    both succeed under mlx-lm 0.32.0."""
    import importlib

    # The modules themselves: the package attribute mlx_lm.convert is the function.
    lm_convert = importlib.import_module("mlx_lm.convert")
    lm_utils = importlib.import_module("mlx_lm.utils")
    convert = import_sibling("port.convert")
    calls = []
    real_convert, real_load = lm_convert.convert, lm_utils.load_model

    def recording_convert(*args, **kwargs):
        calls.append(("convert", kwargs.get("trust_remote_code")))
        return real_convert(*args, **kwargs)

    def recording_load(*args, **kwargs):
        calls.append(("load_model", kwargs.get("trust_remote_code")))
        return real_load(*args, **kwargs)

    monkeypatch.setattr(lm_convert, "convert", recording_convert)
    monkeypatch.setattr(lm_utils, "load_model", recording_load)
    out = tmp_path / "Kolibri-tiny-MLX-8bit-g64"
    _quiet(convert.convert, source, out, 8, 64, guard=False)
    assert calls[0] == ("convert", True)
    assert convert.model_file_trust(out) == {"trust_remote_code": True}

    calls.clear()
    new_port = tmp_path / "new_port.py"
    new_port.write_text(Path(convert.PORT_FILE).read_text() + "\n# exp_037 test: a later port fix\n")
    record = _quiet(convert.refresh_port_file, out, port_file=new_port)
    assert calls == [("load_model", True)]
    assert record["port_sha256"] == _sha256(new_port)
    assert convert.model_file_trust(out, port_file=new_port) == {"trust_remote_code": True}


@pytest.mark.parametrize("name", ["tokenizer_config.json", "config.json"])
def test_a_source_with_code_of_its_own_is_refused(source, tmp_path, name):
    """mlx_lm passes convert's trust_remote_code=True on to transformers'
    tokenizer loader, so a source that declares an auto_map is refused before
    anything is staged."""
    convert = import_sibling("port.convert")
    src = tmp_path / "src"
    shutil.copytree(source, src)
    data = json.loads((src / name).read_text())
    data["auto_map"] = {"AutoTokenizer": ["tokenization_evil.EvilTokenizer", None]}
    (src / name).write_text(json.dumps(data))
    with pytest.raises(SystemExit, match="auto_map"):
        convert.check_source(src)
    with pytest.raises(SystemExit, match="auto_map"):
        _quiet(convert.convert, src, tmp_path / "out", 8, 64, guard=False)
    assert not (tmp_path / "out").exists() and not (tmp_path / "out.staging").exists()


def test_tensor_policy(converted):
    """What is quantised and what is not, from the saved safetensors headers."""
    t = _read_headers(converted.out)
    cfg = tc.preset_config("vendor")
    q = converted.quantised_head

    def quantised(module: str) -> bool:
        return f"{module}.scales" in t or f"{module}.biases" in t

    def check_quantised(module: str) -> None:
        assert quantised(module), module
        assert t[f"{module}.weight"]["dtype"] == "U32", module
        assert t[f"{module}.scales"]["dtype"] == "BF16", module  # bf16 scales and biases
        assert t[f"{module}.biases"]["dtype"] == "BF16", module

    keep = []  # (module, expected weight dtype)
    for i in range(cfg["num_hidden_layers"]):
        p = f"model.layers.{i}"
        keep.append((f"{p}.mlp.gate", "F32"))  # router: exact fp32 upcast, items 11, 18
        keep += [(f"{p}.{n}", "BF16") for n in NORMS]
        assert t[f"{p}.mlp.gate.expert_bias"]["dtype"] == "F32"  # spec item 12
        assert not any(k.startswith(f"{p}.mlp.gate.expert_bias.") for k in t)
        for m in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"):
            check_quantised(f"{p}.{m}")
        for proj in ("gate_proj", "up_proj", "down_proj"):
            check_quantised(f"{p}.mlp.switch_mlp.{proj}")
            assert t[f"{p}.mlp.switch_mlp.{proj}.weight"]["shape"][0] == cfg["num_experts"]
            check_quantised(f"{p}.mlp.shared_experts.{proj}")
    keep.append(("model.norm", "BF16"))
    if q:
        check_quantised("model.embed_tokens")
        check_quantised("lm_head")
    else:
        keep += [("model.embed_tokens", "BF16"), ("lm_head", "F32")]

    for module, dtype in keep:
        assert f"{module}.weight" in t, module
        assert not quantised(module), f"{module} must not be quantised"
        assert t[f"{module}.weight"]["dtype"] == dtype, (module, t[f"{module}.weight"]["dtype"])
    # Every scale and bias tensor is bf16.
    assert all(v["dtype"] == "BF16" for k, v in t.items() if k.endswith((".scales", ".biases")))
    # Checkpoint names fully sanitised (spec item 21).
    assert not any(".mlp.experts." in k or ".moe.router." in k for k in t)


def test_head_and_embedding_values(converted, source):
    """BUILD_SPEC 5.1 delta 3: a quantised head or embedding is quantised from
    its bf16 values (bit-identical to mx.quantize of the stored bf16 tensor);
    the vendor-faithful head is the exact fp32 upcast and the embedding the
    stored bf16 tensor. The router weight is the exact fp32 upcast (delta 4)."""
    import mlx.core as mx

    saved = {}
    for f in sorted(converted.out.glob("model*.safetensors")):
        saved.update(mx.load(str(f)))
    src = {}
    for f in sorted(source.glob("model*.safetensors")):
        src.update(mx.load(str(f)))

    def bits_equal(a, b) -> bool:
        if a.dtype != b.dtype or a.shape != b.shape:
            return False
        view = {mx.bfloat16: mx.uint16, mx.float32: mx.uint32, mx.uint32: mx.uint32}[a.dtype]
        return bool(mx.array_equal(a.view(view), b.view(view)))

    for name, module in (("lm_head.weight", "lm_head"), ("model.embed_tokens.weight", "model.embed_tokens")):
        w_bf16 = src[name]
        assert w_bf16.dtype == mx.bfloat16
        if converted.quantised_head:
            wq, s, b = mx.quantize(w_bf16, group_size=converted.group_size, bits=converted.bits)
            assert s.dtype == mx.bfloat16
            assert bits_equal(saved[f"{module}.weight"], wq), module
            assert bits_equal(saved[f"{module}.scales"], s), module
            assert bits_equal(saved[f"{module}.biases"], b), module
        elif module == "lm_head":
            assert bits_equal(saved[name], w_bf16.astype(mx.float32))
        else:
            assert bits_equal(saved[name], w_bf16)
    router = "model.layers.3.mlp.gate.weight"
    assert bits_equal(saved[router], src[router].astype(mx.float32))


def test_output_loads_with_mlx_lm(converted):
    import mlx.core as mx
    import mlx.nn as nn

    model = ph.load_port(converted.out)
    layer = model.layers[0]
    assert type(layer.mlp.gate).__name__ == "Router" and not hasattr(layer.mlp.gate, "to_quantized")
    assert layer.mlp.gate.weight.dtype == mx.float32
    assert isinstance(layer.self_attn.q_proj, nn.QuantizedLinear)
    assert type(layer.mlp.switch_mlp.gate_proj).__name__ == "QuantizedSwitchLinear"
    if converted.quantised_head:
        assert isinstance(model.lm_head, nn.QuantizedLinear) and model.lm_head.bits == converted.bits
        assert model.lm_head.scales.dtype == mx.bfloat16
        assert isinstance(model.model.embed_tokens, nn.QuantizedEmbedding)
    else:
        assert type(model.lm_head) is nn.Linear and model.lm_head.weight.dtype == mx.float32
        assert type(model.model.embed_tokens) is nn.Embedding
        assert model.model.embed_tokens.weight.dtype == mx.bfloat16
    logits = model(mx.array([tc.random_ids(12, seed=5)]))
    assert logits.dtype == mx.float32  # fp32 logits under either head policy (item 14)
    logits = np.array(logits[0])
    assert logits.shape == (12, tc.VOCAB_SIZE) and np.isfinite(logits).all()


@pytest.mark.parametrize("bits", [8, 4])
def test_streaming_bit_identical(bits, source, tmp_path):
    """convert_streaming.py (--streaming) writes the same tensors as the
    mlx_lm path: every name, dtype, shape and bit; the same config.json,
    model card, tokenizer files and port file. Only the shard layout (and so
    the file list and manifest) differs."""
    import mlx.core as mx

    convert = import_sibling("port.convert")
    streaming = import_sibling("port.convert_streaming")
    a, b = tmp_path / "mlx", tmp_path / "streaming"
    ra = _quiet(convert.convert, source, a, bits, 64)
    rb = _quiet(streaming.convert_streaming, source, b, bits, 64)

    def load(d):
        out = {}
        for f in sorted(d.glob("model*.safetensors")):
            out.update(mx.load(str(f)))
        return out

    ta, tb = load(a), load(b)
    assert set(ta) == set(tb)
    views = {mx.bfloat16: mx.uint16, mx.float32: mx.uint32, mx.uint32: mx.uint32}
    for name in ta:
        x, y = ta[name], tb[name]
        assert x.dtype == y.dtype and x.shape == y.shape, name
        assert bool(mx.array_equal(x.view(views[x.dtype]), y.view(views[y.dtype]))), name
    for name in ("config.json", "README.md", "kolibri1.py", "tokenizer.json", "tokenizer_config.json",
                 "generation_config.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name
    ia = json.loads((a / "model.safetensors.index.json").read_text())
    ib = json.loads((b / "model.safetensors.index.json").read_text())
    assert ia["metadata"] == ib["metadata"] and set(ia["weight_map"]) == set(ib["weight_map"])
    assert rb["converter"] == "convert_streaming.write_streaming" and rb["flags"]["streaming"] is True
    assert rb["tensor_policy"] == ra["tensor_policy"]
    assert len(list(b.glob("model-*.safetensors"))) == tc.preset_config("vendor")["num_hidden_layers"] + 2


def test_streaming_casts_like_mlx_lm(source, tmp_path):
    """The --dtype cast step matters only for tensors stored in fp32 (the
    released checkpoint is all BF16, so on it the cast is a no-op): a source
    with an F32 norm weight and an F32 expert weight must come out identical
    from both converters, the norm cast to BF16 and the expert quantised from
    its bf16 cast."""
    import mlx.core as mx

    convert = import_sibling("port.convert")
    src = tmp_path / "src_f32"
    shutil.copytree(source, src)
    names = ("model.layers.0.input_layernorm.weight", "model.layers.0.mlp.experts.3.up_proj.weight")
    for shard in sorted(src.glob("model*.safetensors")):
        tensors = mx.load(str(shard))
        if any(n in tensors for n in names):
            tensors = {k: (v.astype(mx.float32) + (1e-3 if k in names else 0.0)) if k in names else v
                       for k, v in tensors.items()}
            mx.save_safetensors(str(shard), tensors, metadata={"format": "pt"})
    a = _quiet(convert.convert, src, tmp_path / "mlx", 8, 64)
    b = _quiet(convert.convert, src, tmp_path / "streaming", 8, 64, streaming=True)
    ta, tb = _read_headers(tmp_path / "mlx"), _read_headers(tmp_path / "streaming")
    assert ta["model.layers.0.input_layernorm.weight"]["dtype"] == "BF16"
    assert ta["model.layers.0.mlp.switch_mlp.up_proj.scales"]["dtype"] == "BF16"
    la, lb = {}, {}
    for f in sorted((tmp_path / "mlx").glob("model*.safetensors")):
        la.update(mx.load(str(f)))
    for f in sorted((tmp_path / "streaming").glob("model*.safetensors")):
        lb.update(mx.load(str(f)))
    views = {mx.bfloat16: mx.uint16, mx.float32: mx.uint32, mx.uint32: mx.uint32}
    for name in la:
        assert la[name].dtype == lb[name].dtype, name
        assert bool(mx.array_equal(la[name].view(views[la[name].dtype]), lb[name].view(views[lb[name].dtype]))), name
    assert a["tensor_policy"] == b["tensor_policy"]


# --- memory guard ----------------------------------------------------------------


def _readings(values):
    """A reader returning the given values in turn, then the last one forever."""
    values = list(values)

    def read():
        return values.pop(0) if len(values) > 1 else values[0]

    return read


def test_memory_guard_trips_on_swap_growth():
    convert = import_sibling("port.convert")
    aborted = []
    guard = convert.MemoryGuard(interval=0.01, read_swap=_readings([1e9, 1e9, 3.5e9]),
                                read_footprint=lambda: 1e8, memsize=10**12, grace=0.05,
                                on_hard_abort=aborted.append, interrupt_main=False)
    guard.start()
    guard._thread.join(timeout=5)
    summary = guard.stop()
    assert summary["tripped"].startswith("swap grew by 2.50 GB")
    assert summary["swap_growth_bytes"] == 2.5e9
    # Nobody acknowledged within the grace period: the hard abort ran.
    assert aborted == [summary["tripped"]]


def test_memory_guard_trips_on_footprint_and_waits_for_the_main_thread():
    convert = import_sibling("port.convert")
    aborted = []
    guard = convert.MemoryGuard(interval=0.01, read_swap=lambda: None,
                                read_footprint=_readings([10, 50, 90]), memsize=100, grace=5.0,
                                on_hard_abort=aborted.append, interrupt_main=False)
    guard.start()
    for _ in range(500):
        if guard.tripped:
            break
        guard._stop.wait(0.01)
    assert guard.tripped and "0.8 x RAM" in guard.tripped
    guard.acknowledged.set()  # the main thread stopped in time
    summary = guard.stop()
    assert aborted == [] and summary["footprint_max_sampled_bytes"] == 90


def test_memory_guard_stays_quiet_within_limits():
    convert = import_sibling("port.convert")
    guard = convert.MemoryGuard(interval=0.01, read_swap=_readings([1e9, 2e9, 2.9e9]),
                                read_footprint=lambda: 70, memsize=100, interrupt_main=False)
    guard.start()
    guard._stop.wait(0.2)
    summary = guard.stop()
    assert summary["tripped"] is None and summary["samples"] >= 3
    assert summary["swap_growth_bytes"] == pytest.approx(1.9e9)


def test_the_real_readers_work_here():
    convert = import_sibling("port.convert")
    assert convert.footprint_bytes() > 0
    assert convert.peak_footprint_bytes() >= convert.peak_rss_bytes() > 0
    swap = convert.swap_used_bytes()
    assert swap is None or swap >= 0
    assert (convert.memsize_bytes() or 1) > 0


def test_convert_stops_when_the_guard_trips(source, tmp_path):
    """The guard interrupts the conversion: .partial and .staging are removed,
    no output appears, and the message names the --streaming command."""
    convert = import_sibling("port.convert")
    guard = convert.MemoryGuard(interval=0.01, read_swap=_readings([1e9, 4e9]),
                                read_footprint=lambda: 1, memsize=10**12, grace=60.0)
    out = tmp_path / "Kolibri-tiny-MLX-8bit-g64"
    with pytest.raises(SystemExit) as e:
        _quiet(convert.convert, source, out, 8, 64, guard=guard)
    message = str(e.value)
    assert "memory guard: swap grew" in message
    assert "--streaming" in message and "--bits 8 --group-size 64" in message
    assert not out.exists()
    assert not (tmp_path / (out.name + ".partial")).exists()
    assert not (tmp_path / (out.name + ".staging")).exists()


# --- CLI -------------------------------------------------------------------------


def test_cli_flags_and_results_copy(source, tmp_path, monkeypatch):
    """--no-quantize-* invert the pre-registered default; --bits is 8 or 4;
    the record copy goes to results/convert after the identity check."""
    convert = import_sibling("port.convert")
    calls, identity = [], []
    fake_record = {"bits": 8, "t_end": "2026-10-03T17:01:52Z", "files": {}}

    def fake_convert(src, out, bits, group_size, **kw):
        calls.append((bits, group_size, kw))
        return dict(fake_record)

    monkeypatch.setattr(convert, "convert", fake_convert)
    monkeypatch.setattr(convert, "_require_identity", lambda: identity.append(1))
    results = tmp_path / "results" / "convert"
    base = ["--src", str(source), "--out", str(tmp_path / "o"), "--bits", "8"]
    _quiet(convert.main, base + ["--results-dir", str(results)])
    assert calls[-1] == (8, 64, {"quantize_embeddings": True, "quantize_lm_head": True,
                                 "force": False, "streaming": False})
    copy = results / "convert_8bit_20261003T170152Z.json"
    assert json.loads(copy.read_text()) == fake_record and identity == [1, 1]
    with pytest.raises(FileExistsError):  # append-only: never overwritten
        _quiet(convert.main, base + ["--results-dir", str(results)])
    _quiet(convert.main, base + ["--no-quantize-embeddings", "--no-quantize-lm-head", "--streaming",
                                 "--results-dir", "none"])
    assert calls[-1][2] == {"quantize_embeddings": False, "quantize_lm_head": False,
                            "force": False, "streaming": True}
    assert len(identity) == 4  # 'none' skips the identity check and the copy
    for bad in (["--bits", "6"], ["--group-size", "48"]):
        with pytest.raises(SystemExit):
            with contextlib.redirect_stderr(io.StringIO()):
                convert.main(base[:-2] + bad)


def test_convert_python_api_defaults_to_the_quantised_head():
    import inspect

    convert = import_sibling("port.convert")
    kolibri1 = import_sibling("port.kolibri1")
    params = inspect.signature(convert.convert).parameters
    assert params["quantize_embeddings"].default is True and params["quantize_lm_head"].default is True
    params = inspect.signature(kolibri1.make_quant_predicate).parameters
    assert params["quantize_embeddings"].default is True and params["quantize_lm_head"].default is True


def _write_dequantised_checkpoint(out: Path, src_config: Path, dst: Path) -> Path:
    """Dequantise the saved tensors of a converted dir (read straight from its
    files, not through the port's loader) and write them back under the
    checkpoint names as one F32 model.safetensors, for the numpy reference."""
    import mlx.core as mx

    cfg = json.loads((out / "config.json").read_text())
    gs, bits = cfg["quantization"]["group_size"], cfg["quantization"]["bits"]
    saved = {}
    for shard in sorted(out.glob("model*.safetensors")):
        saved.update(mx.load(str(shard)))
    hf = {}
    for name, value in saved.items():
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        base = name[: -len(".weight")]
        if name.endswith(".weight") and f"{base}.scales" in saved:
            w = mx.dequantize(
                value,
                saved[f"{base}.scales"].astype(mx.float32),
                saved[f"{base}.biases"].astype(mx.float32),
                group_size=gs,
                bits=bits,
            )
        else:
            w = value.astype(mx.float32)
        # Inverse of the port's sanitize (spec item 21).
        if ".mlp.switch_mlp." in name:
            prefix, rest = name.split(".mlp.switch_mlp.")
            proj = rest.split(".")[0]
            for e in range(w.shape[0]):
                hf[f"{prefix}.mlp.experts.{e}.{proj}.weight"] = w[e]
        elif name.endswith(".mlp.gate.expert_bias"):
            hf[name.replace(".mlp.gate.expert_bias", ".moe.router.expert_bias")] = w
        else:
            hf[name] = w
    dst.mkdir(parents=True)
    mx.save_safetensors(str(dst / "model.safetensors"), hf)
    shutil.copyfile(src_config, dst / "config.json")
    return dst


def test_quantised_port_vs_reference_on_dequantised_weights(converted, source, tmp_path, monkeypatch):
    """Conversion and loading are exact up to bf16 arithmetic.

    The reference runs in fp32 on the dequantised weights, which are the
    weights the quantised port actually multiplies with. With the port forced
    onto the reference's experts (as in test_port_vs_ref b1), only bf16
    activation rounding remains, plus the rounding of dequantised weights
    inside the quantised matmul. Same bounds as b1: mean KL <= 1e-3,
    max|d| / max|ref| <= 5e-2, top-1 >= 0.98 on decisive positions.
    Measured (8-bit g64, 4-bit g64, 4-bit g32): KL 8.4e-5 to 9.7e-5, rel
    1.7e-2 to 1.9e-2, decisive top-1 1.000. That is the b1 level, so the
    conversion adds nothing beyond the quantisation itself.
    """
    kolibri_ref = import_sibling("reference.kolibri_ref")
    deq = _write_dequantised_checkpoint(converted.out, source / "config.json", tmp_path / "deq")
    ids = tc.random_ids(SEQ_LEN, seed=3000)
    ref = kolibri_ref.KolibriReference(str(deq))
    ref_logits = ref.forward(np.array(ids))

    model = ph.load_port(converted.out)
    ph.force_routing(monkeypatch, model, [i for _, i in ref.last_routing])
    stats = ph.compare(ref_logits, ph.port_logits(model, ids))
    assert stats["kl_mean"] <= 1e-3, ph.fmt(stats)
    assert stats["rel_max"] <= 5e-2, ph.fmt(stats)
    assert stats["top1_decisive"] >= 0.98, ph.fmt(stats)


def test_quantised_port_vs_fp32_reference(converted, reference_on_source, monkeypatch):
    """Smoke bounds against the fp32 reference on the original weights.

    These hold for tiny random weights only and are not the real-weight gate.
    The requested 0.95 (8-bit) and 0.85 (4-bit) top-1 cannot be met by a
    correct conversion of these weights (INTEGRATION_LOG entry 3).

    Affine noise model: the 64 weights of a group span about 4.8 standard
    deviations, so b bits give a step of 4.8 sigma / (2^b - 1) and a
    per-weight error of step / sqrt(12).

    * 8-bit: 0.54% of sigma, about twice the bf16 activation rounding
      (u / sqrt(3) = 0.23%). The bounds of the bf16 tests carry over:
      - natural routing, with the b2 worst-case bounds: top-1 >= 0.60 and
        mean KL <= 3e-2. Those bounds assume every token is flip-affected,
        and 8-bit noise flips more routes than bf16 does. Measured: top-1
        0.775 (vendor-faithful head) and 0.787 (quantised head), KL 1.2e-2
        and 1.4e-2;
      - forced onto the reference's experts, with the b1 bounds: top-1
        >= 0.98 on decisive positions and mean KL <= 1e-3. Logit noise is
        about 1.7-2x the bf16 level, still well under the 0.1 decisive lead,
        and the KL grows by the square of that factor, 3-5x b1's 6e-5 to
        1.1e-4. Measured: decisive 1.000, KL 3.0e-4 (vendor-faithful) and
        3.6e-4 (quantised head and embedding).
    * 4-bit: 9.2% of sigma per weight. Every quantised matmul output is
      about 9% off, with 2-3 of them in series per sublayer. That leaves the
      residual stream about 15% off and the logit noise at sigma ~0.2-0.35
      (from KL ~ sigma^2 / 2 = 0.03-0.06). The top logits of 1024 Gaussian
      logits with std 0.8 are spaced about 0.8 / sqrt(2 ln 1024) = 0.21 apart,
      so 3-4 candidates sit within one noise sigma of the maximum. Even a
      uniform choice among them keeps about 0.3. Bound: top-1 >= 0.25, still
      250x chance (1/1008). A 4-bit head adds a logit error of about 9% of
      the logit std (0.07), small next to the trunk's 0.2-0.35. Measured:
      0.37 (g64, vendor-faithful head), 0.32 (g64) and 0.39 (g32) with the
      head and embedding at 4 bits. This bound cannot tell a conversion bug
      from 4-bit noise; the dequantised-reference tests above do that.
    """
    ref = reference_on_source
    model = ph.load_port(converted.out)
    natural = ph.compare(ref.logits, ph.port_logits(model, ref.ids))
    assert np.isfinite(natural["kl_mean"])
    if converted.bits == 8:
        assert natural["top1"] >= 0.60, ph.fmt(natural)
        assert natural["kl_mean"] <= 3e-2, ph.fmt(natural)
        forced_model = ph.load_port(converted.out)
        ph.force_routing(monkeypatch, forced_model, ref.routing_ids)
        forced = ph.compare(ref.logits, ph.port_logits(forced_model, ref.ids))
        assert forced["top1_decisive"] >= 0.98, ph.fmt(forced)
        assert forced["kl_mean"] <= 1e-3, ph.fmt(forced)
    else:
        assert natural["top1"] >= 0.25, ph.fmt(natural)


def test_dequantised_mode_reads_the_mx_dequantize_values(converted):
    """BUILD_SPEC 5.1b item 4: the reference's own unpacker (mlx_affine_np,
    no mlx) returns, under the checkpoint names, exactly what mlx computes
    from the saved tensors: mx.dequantize with fp32 scales for every linear
    weight (quantized_matmul's fp32 dequantisation) and mx.dequantize with
    the stored bf16 scales for the embedding (QuantizedEmbedding). 0 ulp."""
    import mlx.core as mx

    affine = import_sibling("reference.mlx_affine_np")
    view = affine.ConvertedCheckpoint(str(converted.out))
    saved = {}
    for shard in sorted(converted.out.glob("model*.safetensors")):
        saved.update(mx.load(str(shard)))
    gs, bits = converted.group_size, converted.bits

    def mlx_value(module, row=None, embedding=False):
        w, s, b = (saved[f"{module}.{p}"] for p in ("weight", "scales", "biases"))
        if row is not None:
            w, s, b = w[row], s[row], b[row]
        if embedding:
            return np.array(mx.dequantize(w, s, b, group_size=gs, bits=bits).astype(mx.float32))
        return np.array(mx.dequantize(w, s.astype(mx.float32), b.astype(mx.float32), group_size=gs, bits=bits))

    checked = 0
    for i in (0, 4):
        p = f"model.layers.{i}"
        for e in (0, 5):
            name = f"{p}.mlp.experts.{e}.down_proj.weight"
            np.testing.assert_array_equal(view.get(name), mlx_value(f"{p}.mlp.switch_mlp.down_proj", row=e))
            checked += 1
        np.testing.assert_array_equal(view.get(f"{p}.self_attn.q_proj.weight"), mlx_value(f"{p}.self_attn.q_proj"))
        np.testing.assert_array_equal(view.get(f"{p}.moe.router.expert_bias"),
                                      np.array(saved[f"{p}.mlp.gate.expert_bias"]))
        np.testing.assert_array_equal(view.get(f"{p}.mlp.gate.weight"), np.array(saved[f"{p}.mlp.gate.weight"]))
        checked += 3
    rows = np.array([3, 17, 900])
    if converted.quantised_head:
        np.testing.assert_array_equal(view.get_rows("model.embed_tokens.weight", rows),
                                      mlx_value("model.embed_tokens", embedding=True)[rows])
        np.testing.assert_array_equal(view.get_rows("lm_head.weight", slice(10, 20)), mlx_value("lm_head")[10:20])
    else:
        np.testing.assert_array_equal(view.get_rows("model.embed_tokens.weight", rows),
                                      np.array(saved["model.embed_tokens.weight"].astype(mx.float32))[rows])
        np.testing.assert_array_equal(view.get("lm_head.weight"), np.array(saved["lm_head.weight"]))
    assert checked == 10
    assert view.shape_of("model.layers.0.mlp.experts.3.gate_proj.weight") == (256, 256)


def test_reference_dequantised_mode_vs_quantised_port(converted, monkeypatch):
    """The reference in dequantised mode (its own unpacker, BUILD_SPEC 5.1b
    item 4) against the quantised port, forced onto the reference's experts:
    the (b1) bounds, as for the mx.dequantize checkpoint above, in both
    reference modes. The bf16 emulation does not land closer to the port:
    it rounds at the same places but its roundings are independent of the
    port's (different accumulation order), so the two errors add. That is
    what makes it a calibration of the error size, not a bit-level twin.
    Measured (8/4-bit, both head policies): fp32 mode KL 8.4e-5 to 9.6e-5,
    rel 1.7e-2 to 2.1e-2; emulation KL 1.05e-4 to 1.22e-4, rel 2.2e-2 to
    2.5e-2; decisive top-1 1.000 throughout."""
    kolibri_ref = import_sibling("reference.kolibri_ref")
    ids = tc.random_ids(SEQ_LEN, seed=3000)
    for emulate, kl_max in ((False, 1e-3), (True, 1e-3)):
        ref = kolibri_ref.KolibriReference(str(converted.out), dequant_dir=str(converted.out), emulate_bf16=emulate)
        ref_logits = ref.forward(np.array(ids))
        model = ph.load_port(converted.out)
        with monkeypatch.context() as mp:
            ph.force_routing(mp, model, [i for _, i in ref.last_routing])
            stats = ph.compare(ref_logits, ph.port_logits(model, ids))
        assert stats["kl_mean"] <= kl_max, (emulate, ph.fmt(stats))
        assert stats["rel_max"] <= 5e-2, (emulate, ph.fmt(stats))
        assert stats["top1_decisive"] >= 0.98, (emulate, ph.fmt(stats))


def test_g2q_head_and_embedding_on_dequantised_weights(converted):
    """The G2q head and embedding checks on the tiny build. Both sides get
    the reference's normed hidden state (head) or the same ids (embedding).
    * head: the port's compute_logits (fp32 input into a quantised or fp32
      head) vs the reference's fp32 head on the dequantised weight: relative
      L2 per position <= 1e-5 (gate threshold 1e-3; fp32 accumulation over
      256 products is ~1e-7). Measured: <= 3.3e-7 (quantised head), 0
      (vendor-faithful fp32 head).
    * embedding: identical values (both round the same fp32 dequantised
      value to bf16 once; gate threshold 4e-3 relative)."""
    import mlx.core as mx

    kolibri_ref = import_sibling("reference.kolibri_ref")
    ids = np.array(tc.random_ids(64, seed=3001))
    ref = kolibri_ref.KolibriReference(str(converted.out), dequant_dir=str(converted.out))
    _, _, hn = ref.forward(ids, return_hidden=True)
    ref_logits = ref.lm_head(hn)
    model = ph.load_port(converted.out)
    port_logits = model.compute_logits(mx.array(hn)[None])
    assert port_logits.dtype == mx.float32
    port_logits = np.array(port_logits[0])
    rel = np.linalg.norm(port_logits - ref_logits, axis=-1) / np.linalg.norm(ref_logits, axis=-1)
    assert rel.max() <= 1e-5, rel.max()
    emb = np.array(model.model.embed_tokens(mx.array(ids)).astype(mx.float32))
    np.testing.assert_array_equal(emb, ref.embed(ids))


def test_g2q_layer_branches_on_dequantised_weights(converted):
    """A G2q row on the tiny build: each port layer (quantised weights, bf16),
    fed the reference's input h_l rounded to bf16 and forced onto the
    reference's experts, against the reference layer on the dequantised
    weights in bf16 emulation with the same input and experts. Branch errors
    e = |r_port - r_ref| / |r_ref| (item 29), median over tokens per layer:
    <= 3e-2. A rounding to bf16 has a relative RMS error of u / sqrt(3)
    (u = 2^-8); a branch passes about 12 stored values in series (two norms
    with two roundings each, q/k/v, q/k norms, RoPE, attention, o_proj;
    gate/up, silu, product, down, sum), so each implementation is about
    sqrt(12) * u / sqrt(3) = 2u = 7.8e-3 off, and two independent ones
    differ by sqrt(2) * 2u = 1.1e-2. The bound is 2.7x that.
    Measured: medians 5.4e-3 to 1.5e-2, p99 <= 2.2e-2 (all five builds)."""
    import mlx.core as mx

    kolibri_ref = import_sibling("reference.kolibri_ref")
    ids = np.array(tc.random_ids(96, seed=3002))
    fp32 = kolibri_ref.KolibriReference(str(converted.out), dequant_dir=str(converted.out))
    emu = kolibri_ref.KolibriReference(str(converted.out), dequant_dir=str(converted.out), emulate_bf16=True)
    model = ph.load_port(converted.out)
    h = fp32.embed(ids)
    for i in range(fp32.cfg.num_hidden_layers):
        ref_out = fp32.layer_branches(i, h)
        top6 = ref_out["info"]["top6"]
        emu_out = emu.layer_branches(i, kolibri_ref.round_bf16(h), force_ids=top6)
        x = mx.array(h)[None].astype(mx.bfloat16)
        fa, swa = model.model.make_masks(x)
        layer = model.layers[i]
        r_attn, h_mid, r_moe, h_out, logits, got_ids = layer.branches(
            x, swa if layer.use_sliding else fa, None, force_ids=mx.array(top6.astype(np.uint32))[None])
        assert logits.dtype == mx.float32
        np.testing.assert_array_equal(np.sort(np.array(got_ids[0]), -1), np.sort(top6, -1))
        for name, port_r in (("r_attn", r_attn), ("r_moe", r_moe)):
            ref_r = emu_out[name]
            e = np.linalg.norm(np.array(port_r[0].astype(mx.float32)) - ref_r, axis=-1) / np.linalg.norm(ref_r, axis=-1)
            assert np.median(e) <= 3e-2, (i, name, float(np.median(e)))
        h = ref_out["h_out"]
