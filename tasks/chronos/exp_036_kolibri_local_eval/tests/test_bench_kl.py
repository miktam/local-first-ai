# SPDX-License-Identifier: MIT
"""bench/kl_8v4.py (H8): byte blocks, token first bytes, the context-start
token per family (real tokenizers), the KL arithmetic, and the whole cell on
tiny checkpoints, where the Kolibri dump path and the both-loaded peer path
must agree."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from test_bench_support import (  # noqa: F401  (fixtures)
    bench_env,
    kolibri_tok_dir,
    mini_dir,
    tiny_models,
    tiny_loader,
)

from bench import common, kl_8v4

# Synthetic stand-ins for the gate texts (our own sentences; no gate or web text).
_EN = (
    "Local inference keeps every byte on the machine. We measure what fits, how fast it runs, "
    "and whether a smaller grid costs fidelity. Numbers such as 3.14, 2026 and 128 GB appear "
    "in the text so that digit handling is exercised. "
)
_DE = (
    "Die Messung läuft vollständig auf dem eigenen Rechner. Größere Kontexte kosten Speicher, "
    "kleinere Gitter kosten Genauigkeit – beides wird gezählt: 1.234,56 € und 7 % 🙂. "
)
TEXTS = {
    "T1": _EN * 3,
    "T2": (_EN * 2)[::-1].swapcase()[:600] + " end.",
    "T3": _DE * 3,
    "T4": "Artikel 1 (Würde). " + _DE * 2,
    "T5": "Straße, Übergröße, Ärger: " + _DE * 2 + " 漢字 test.",
    "T6": _DE * 2 + " Zeile\nNeue Zeile\tmit Tab.",
}


@pytest.fixture
def text_dirs(tmp_path):
    pub, work = tmp_path / "gate_texts_public", tmp_path / "work" / "gate_texts"
    pub.mkdir()
    work.mkdir(parents=True)
    for t, s in TEXTS.items():
        d = pub if t in kl_8v4.PUBLIC_TEXTS else work
        (d / f"{t}_synthetic.txt").write_text(s, encoding="utf-8")
    return pub, work


# --------------------------------------------------------------------------
# Blocks and first bytes
# --------------------------------------------------------------------------


@pytest.mark.parametrize("t", list(TEXTS))
def test_byte_blocks_cut_at_whitespace(t):
    data = TEXTS[t].encode("utf-8")
    b = kl_8v4.byte_blocks(data)
    assert len(b) == 9 and b[0] == 0 and b[-1] == len(data)
    assert all(x < y for x, y in zip(b, b[1:]))
    assert all(data[i] in kl_8v4.WHITESPACE for i in b[1:-1])


def test_byte_blocks_without_whitespace_respect_utf8():
    data = ("ä" * 50).encode("utf-8")
    b = kl_8v4.byte_blocks(data)
    assert all((data[i] & 0xC0) != 0x80 for i in b[1:-1])
    with pytest.raises(common.BenchError):
        kl_8v4.byte_blocks(b"abc")


def test_first_bytes_map_split_characters_to_their_first_byte(tiny_models):
    text = "Grüße 🙂 x"
    ids, fb = kl_8v4.tokenize(tiny_models["a"] / "tokenizer.json", text)
    assert ids == list(text.encode("utf-8"))  # the byte tokenizer: one token per byte
    # ü (2 bytes) and 🙂 (4 bytes) are split; every piece maps to the first byte
    assert fb.tolist() == [0, 1, 2, 2, 4, 4, 6, 7, 8, 8, 8, 8, 12, 13]


def test_token_blocks_partition_tokens_by_first_byte():
    bounds = [0, 10, 20, 30]
    fb = np.array([0, 3, 9, 10, 15, 29])
    assert kl_8v4.token_blocks(fb, bounds).tolist() == [0, 0, 0, 1, 1, 2]


def _real_dirs() -> dict[str, Path]:
    return {
        "kolibri": kolibri_tok_dir(),
        "gemma4": mini_dir("mlx-community", "gemma-4-26b-a4b-it-8bit"),
        "qwen3_6": mini_dir("mlx-community", "Qwen3.6-35B-A3B-8bit"),
        "qwen3_8": mini_dir("mlx-community", "Qwen3.8-27B-8bit"),
    }


def test_context_prefix_per_family_with_the_real_tokenizers():
    got = {name: kl_8v4.context_prefix(d) for name, d in _real_dirs().items()}
    assert got["kolibri"] == {"id": 127901, "token": "<|endoftext|>", "rule": "<|endoftext|> (no BOS defined)"}
    assert got["gemma4"] == {"id": 2, "token": "<bos>", "rule": "tokenizer_config bos_token"}
    for q in ("qwen3_6", "qwen3_8"):
        assert got[q] == {"id": 248044, "token": "<|endoftext|>", "rule": "generation_config bos_token_id"}
    for name in ("gemma4", "qwen3_6", "qwen3_8"):
        four = mini_dir("mlx-community", _real_dirs()[name].name.replace("8bit", "4bit"))
        assert kl_8v4.context_prefix(four) == got[name]


@pytest.mark.parametrize("t", list(TEXTS))
def test_real_tokenizer_first_bytes_and_blocks(t):
    text = TEXTS[t]
    data = text.encode("utf-8")
    bounds = kl_8v4.byte_blocks(data)
    for name, d in _real_dirs().items():
        ids, fb = kl_8v4.tokenize(d / "tokenizer.json", text)
        assert fb[0] == 0 and np.all(np.diff(fb) >= 0) and fb[-1] < len(data), name
        blocks = kl_8v4.token_blocks(fb, bounds)
        assert blocks.min() == 0 and blocks.max() == 7, name
        # a token's first byte lies in its block
        lo, hi = np.asarray(bounds[:-1])[blocks], np.asarray(bounds[1:])[blocks]
        assert np.all((lo <= fb) & (fb < hi)), name


# --------------------------------------------------------------------------
# KL arithmetic
# --------------------------------------------------------------------------


def _np_kl(a, b):
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    la = a - np.log(np.exp(a - a.max(-1, keepdims=True)).sum(-1, keepdims=True)) - a.max(-1, keepdims=True)
    lb = b - np.log(np.exp(b - b.max(-1, keepdims=True)).sum(-1, keepdims=True)) - b.max(-1, keepdims=True)
    return (np.exp(la) * (la - lb)).sum(-1)


def test_kl_rows_match_float64_numpy():
    import mlx.core as mx

    rng = np.random.default_rng(0)
    a = rng.normal(0, 3, (64, 5000)).astype(np.float32)
    b = (a + rng.normal(0, 0.3, a.shape)).astype(np.float32)
    got = np.array(kl_8v4.kl_rows(kl_8v4.logprobs(mx.array(a), False), kl_8v4.logprobs(mx.array(b), False)))
    np.testing.assert_allclose(got, _np_kl(a, b), rtol=2e-3, atol=1e-6)
    same = np.array(kl_8v4.kl_rows(kl_8v4.logprobs(mx.array(a), True), kl_8v4.logprobs(mx.array(a), True)))
    assert np.all(same == 0)


def test_bf16_rounding_is_applied_to_the_logits():
    import mlx.core as mx

    x = mx.array(np.array([[1.0 + 2**-12, 0.0, -1.0]], dtype=np.float32))
    r = np.array(kl_8v4.logprobs(x, True))
    f = np.array(kl_8v4.logprobs(x, False))
    assert not np.array_equal(r, f)
    xr = np.array(x.astype(mx.bfloat16).astype(mx.float32))
    np.testing.assert_allclose(r, xr - np.log(np.exp(xr).sum()), rtol=1e-6)


# --------------------------------------------------------------------------
# The cell on tiny checkpoints
# --------------------------------------------------------------------------


def _run(bench_env, text_dirs, mapping, families=("kolibri", "gemma4"), chunk=2048):
    pub, work = text_dirs
    path = kl_8v4.run_cell(
        bench_env.results,
        loader=tiny_loader(mapping),
        families=families,
        public_dir=pub,
        work_texts_dir=work,
        chunk=chunk,
        arm_dirs=lambda arm: mapping[arm],
    )
    return path, json.loads(path.read_text())


def test_kl_cell_dump_path_equals_loaded_path(bench_env, text_dirs, tiny_models):
    a, b = tiny_models["a"], tiny_models["b"]
    mapping = {"K8": a, "K4": b, "G8": a, "G4": b}
    path, out = _run(bench_env, text_dirs, mapping, chunk=300)  # several chunks per text
    assert out["complete"] is True and path.name.startswith("kl_8v4_")
    assert path.read_text() == common.dumps(out) + "\n"  # sorted keys
    assert {c for c in bench_env.calls if c[0] == "gate"} == {("gate", "K8"), ("gate", "K4")}
    kol, gem = out["families"]["kolibri"], out["families"]["gemma4"]
    for t, s in TEXTS.items():
        nbytes = len(s.encode("utf-8"))
        assert out["texts"][t]["bytes"] == nbytes and sum(out["texts"][t]["block_bytes"]) == nbytes
        kb, gb = kol["texts"][t]["blocks"], gem["texts"][t]["blocks"]
        assert len(kb) == 8 and sum(x["tokens"] for x in kb) == kol["texts"][t]["tokens"] == nbytes
        for x, y in zip(kb, gb):
            assert x["tokens"] == y["tokens"] and x["bytes"] == y["bytes"]
            for k in ("kl_sum", "nll8_sum", "nll4_sum", "agree_count"):
                assert x[k] == pytest.approx(y[k], rel=1e-6, abs=1e-9), (t, k)
            assert "kl_fp32_sum" in x and "kl_fp32_sum" not in y
        assert kol["texts"][t]["logits_dtype_native"] == {"8bit": "float32", "4bit": "float32"}
    assert kol["totals"]["kl_sum"] > 0
    assert kol["totals"]["kl_per_byte"] == pytest.approx(kol["totals"]["kl_sum"] / kol["totals"]["bytes"])
    assert kol["prefix"]["id"] == 1021
    # the K8 dump: two fp32 arrays per text, hashed in the record, in $EXP036_WORK/kl/
    files = out["k8_dump"]["files"]
    assert len(files) == 12 and all(len(v) == 64 for v in files.values())
    dumped = sorted((bench_env.root / "work" / "kl").rglob("*.npy"))
    assert len(dumped) == 12
    arr = np.load(dumped[0], mmap_mode="r")
    assert arr.dtype == np.float32 and arr.shape[1] == 1024
    # no T5/T6 (work-only) text in the record
    assert "Übergröße" not in path.read_text() and "Neue Zeile" not in path.read_text()


def test_kl_cell_same_model_gives_zero_kl_and_full_agreement(bench_env, text_dirs, tiny_models):
    a = tiny_models["a"]
    _, out = _run(bench_env, text_dirs, {"K8": a, "K4": a, "G8": a, "G4": a})
    for fam in ("kolibri", "gemma4"):
        for t, rec in out["families"][fam]["texts"].items():
            for blk in rec["blocks"]:
                assert blk["kl_sum"] == 0.0 and blk["agree_count"] == blk["tokens"]
                assert blk["nll8_sum"] == blk["nll4_sum"]


def test_gate_texts_must_be_unique_and_present(tmp_path, text_dirs, bench_env):
    pub, work = text_dirs
    (work / "T5_synthetic.txt").unlink()
    with pytest.raises(common.BenchError, match="gate text T5"):
        kl_8v4.gate_text_paths(pub, work)
    (work / "T5_a.txt").write_text("x y")
    (work / "T5_b.txt").write_text("x y")
    with pytest.raises(common.BenchError, match="found 2"):
        kl_8v4.gate_text_paths(pub, work)


def test_invalid_utf8_gate_text_is_refused(tmp_path, bench_env):
    p = tmp_path / "T1_bad.txt"
    p.write_bytes(b"abc \xff def")
    with pytest.raises(UnicodeDecodeError):
        kl_8v4.load_texts({"T1": p})


def test_texts_must_match_the_gate_manifest(bench_env, text_dirs, tiny_models):
    pub, work = text_dirs
    texts = kl_8v4.load_texts(kl_8v4.gate_text_paths(pub, work))
    good = {t: r["sha256"] for t, r in texts.items()}
    (pub / "MANIFEST.json").write_text(
        json.dumps({"texts": {t: {"text_sha256": good[t]} for t in ("T1", "T2", "T3", "T4")},
                    "web": {"T5": {"text_sha256": good["T5"]}, "T6": {"text_sha256": "0" * 64}}})
    )
    assert kl_8v4.manifest_text_sha256(pub)["T6"] == "0" * 64
    with pytest.raises(common.BenchError, match="gate text T6: sha256"):
        kl_8v4.verify_texts(texts, kl_8v4.manifest_text_sha256(pub))
    a = tiny_models["a"]
    with pytest.raises(common.BenchError, match="MANIFEST"):
        _run(bench_env, text_dirs, {"K8": a, "K4": a, "G8": a, "G4": a}, families=("gemma4",))
    assert not list(bench_env.results.glob("kl_8v4_*"))


def test_committed_gate_texts_verify_and_tokenise_for_every_family():
    """The real T1-T4 (gate/texts/) hash to the MANIFEST, Kolibri's
    re-tokenisation of each equals the committed 1,536 ids, and every family's
    tokenizer covers all 8 blocks."""
    pub = common.EXP_DIR / "gate" / "texts"
    if not (pub / "MANIFEST.json").is_file():
        pytest.skip("gate/texts/MANIFEST.json not built yet (gate area)")
    found = {t: sorted(pub.glob(f"{t}_*.txt")) for t in kl_8v4.PUBLIC_TEXTS}
    assert all(len(v) == 1 for v in found.values()), found
    texts = kl_8v4.load_texts({t: v[0] for t, v in found.items()})
    assert kl_8v4.verify_texts(texts, kl_8v4.manifest_text_sha256(pub)) == {t: True for t in kl_8v4.PUBLIC_TEXTS}
    dirs = _real_dirs()
    for t, rec in texts.items():
        bounds = kl_8v4.byte_blocks(rec["text"].encode("utf-8"))
        for name, d in dirs.items():
            ids, fb = kl_8v4.tokenize(d / "tokenizer.json", rec["text"])
            assert sorted(set(kl_8v4.token_blocks(fb, bounds).tolist())) == list(range(8)), (t, name)
            if name == "kolibri":
                assert ids == kl_8v4._committed_ids(pub, t) and len(ids) == 1536
