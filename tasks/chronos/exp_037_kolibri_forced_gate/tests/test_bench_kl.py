# SPDX-License-Identifier: MIT
"""bench/kl_8v4.py (H8): byte blocks, token first bytes, the context-start
token per family (real tokenizers), the Amendment 5 chat wrapper per family
(real tokenizers and templates), the KL arithmetic, scoring only the text's
tokens after the wrapper, and the whole cell on tiny checkpoints, where the
Kolibri dump path and the both-loaded peer path must agree and the output
still feeds analysis/verdicts.py (kl_models, H8 arrays)."""

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


# A ChatML stand-in for Kolibri's template on the tiny byte tokenizer (the tiny checkpoints ship none). It
# prefills an empty think block at reasoning_effort "none", like Kolibri's own template.
TINY_CHATML = (
    "{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if reasoning_effort is defined and reasoning_effort == 'none' %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)


def _with_chat_template(src: Path, dst: Path) -> Path:
    import shutil

    shutil.copytree(src, dst)
    tc = json.loads((dst / "tokenizer_config.json").read_text())
    tc["chat_template"] = TINY_CHATML
    (dst / "tokenizer_config.json").write_text(json.dumps(tc))
    return dst


@pytest.fixture(scope="module")
def chat_models(tiny_models, tmp_path_factory):
    """The two tiny models ("a", "b") with a chat template, so every family can be wrapped (Amendment 5)."""
    root = tmp_path_factory.mktemp("bench_tiny_chat")
    return {k: _with_chat_template(d, root / k) for k, d in tiny_models.items()}


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


# Amendment 5: the user turn and the assistant header each family renders with thinking off (runner.chat).
QWEN_WRAP = [248045, 846, 198, 7734, 264, 1414, 13, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
EXPECTED_WRAP = {
    "gemma4": {"kwargs": {"enable_thinking": False}, "prefill": "empty_channel",
               "ids": [2, 105, 2364, 107, 6974, 496, 1816, 236761, 106, 107, 105, 4368, 107, 100, 45518, 107, 101]},
    "qwen3_6": {"kwargs": {"enable_thinking": False}, "prefill": "empty_think", "ids": QWEN_WRAP},
    "qwen3_8": {"kwargs": {"enable_thinking": False}, "prefill": "empty_think", "ids": QWEN_WRAP},
    "kolibri": {"kwargs": {"reasoning_effort": "none"}, "prefill": "empty_think", "ids": None},
}


def test_chat_wrapper_per_family_with_the_real_tokenizers():
    """Each family's wrapper is runner.chat.render of the one user message at effort "none": the explicit
    thinking-off kwargs, the committed peer template (or Kolibri's own), the template's own BOS only (Gemma 4
    <bos> once; none for Kolibri and Qwen), and the same ids from the 4-bit directory."""
    from runner import chat

    manifest = json.loads((common.EXP_DIR / "runner" / "templates" / "MANIFEST.json").read_text())["files"]
    msgs = [{"role": "user", "content": "Write a text."}]
    for name, d in _real_dirs().items():
        w = kl_8v4.chat_wrapper(name, d)
        rec, exp = w["record"], EXPECTED_WRAP[name]
        assert rec["kwargs"] == exp["kwargs"] == chat.template_kwargs(name, "none"), name
        assert rec["prefill"] == exp["prefill"] and rec["add_generation_prompt"] is True, name
        assert rec["messages"] == msgs and rec["effort"] == "none" and rec["amendment"] == "Amendment 5"
        tok = kl_8v4.load_chat_tokenizer(d)
        text, ids, rsha = chat.render(name, tok, msgs, "none")        # the scored runs' rendering, verbatim
        assert w["prompt_ids"] == ids == rec["prompt_ids"] and rec["rendered_sha256"] == rsha
        assert rec["prompt_tokens"] == len(ids) and rec["prompt_ids_sha256"] == common.sha256_ids(ids)
        assert chat.render(name, tok, msgs, "high")[1] != ids, name    # the kwargs matter: thinking on differs
        if exp["ids"] is not None:
            assert ids == exp["ids"], (name, ids)
        bos = kl_8v4.context_prefix(d)["id"]
        assert ids.count(bos) == (1 if name == "gemma4" else 0), name  # no extra BOS: the template's own only
        if name in ("gemma4", "qwen3_6"):
            assert len(ids) < 37      # G8 / Q36-8: fits the batched path's shortest prompt (37 tokens)
        if name == "kolibri":
            assert rec["template_sha256"] == common.sha256_bytes(chat.local_template_bytes(d))
            assert text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
            continue
        assert rec["template"] == f"runner/templates/{chat.TEMPLATE_FILES[name]}"
        assert rec["template_sha256"] == manifest[chat.TEMPLATE_FILES[name]]["sha256"]
        assert rec["template_equals_model_dir"] is True
        four = mini_dir("mlx-community", d.name.replace("8bit", "4bit"))
        assert kl_8v4.chat_wrapper(name, four)["prompt_ids"] == ids, name


def test_scored_input_is_prompt_then_text_and_scores_only_the_text():
    inputs, sf = kl_8v4.scored_input([7, 8, 9], [1, 2, 3, 4])
    assert inputs == [7, 8, 9, 1, 2, 3] and sf == 2
    assert len(inputs) - sf == 4             # one scored row per text token; row sf + j predicts ids[j]
    assert kl_8v4.scored_input([5], [1, 2]) == ([5, 1], 0)   # the raw-text layout: [context start] + ids[:-1]
    with pytest.raises(common.BenchError):
        kl_8v4.scored_input([], [1, 2])
    with pytest.raises(common.BenchError):
        kl_8v4.scored_input([1], [])


@pytest.mark.parametrize("t", ["T1", "T3", "T5"])
def test_wrapped_sequence_with_the_real_tokenizers(t):
    """prompt ids + the text tokenised alone; the text's ids (and so its first bytes and blocks) are those of
    the raw measurement. On these synthetic texts, which start with a letter, the joint encoding of rendered
    prompt + text gives the same ids for every family. That does not hold for every text: a text that starts
    with a newline merges with the trailing "\\n\\n" of Kolibri's and Qwen's wrappers in the joint encoding
    (the real T1, T2 and T6; test_wrapper_text_boundary_on_the_real_gate_texts)."""
    from runner import chat

    text = TEXTS[t]
    for name, d in _real_dirs().items():
        w = kl_8v4.chat_wrapper(name, d)
        ids, fb = kl_8v4.tokenize(d / "tokenizer.json", text)
        inputs, sf = kl_8v4.scored_input(w["prompt_ids"], ids)
        P = len(w["prompt_ids"])
        assert inputs[:P] == w["prompt_ids"] and inputs[P:] == ids[:-1] and sf == P - 1, name
        hf = chat.hf_tokenizer(kl_8v4.load_chat_tokenizer(d))
        assert list(hf.encode(text, add_special_tokens=False)) == ids, name
        rendered = chat.render(name, kl_8v4.load_chat_tokenizer(d), w["record"]["messages"], "none")[0]
        assert list(hf.encode(rendered + text, add_special_tokens=False)) == w["prompt_ids"] + ids, name


# The wrapper/text boundary on the real gate texts (review of Amendment 5). The scored ids are the wrapper's
# then the text's tokenised alone (Andrei's decision); where the joint encoding of rendered prompt + text differs,
# it differs only at the wrapper's last token and the text's first: (scored ids, joint ids) per family and text.
_NL2_SPLIT = {"kolibri": ([263, 10], [120038]), "qwen": ([271, 198], [1358])}       # "\n\n" + "\n" vs "\n\n\n"
_NL4_SPLIT = {"kolibri": ([263, 263], [120724]), "qwen": ([271, 271], [987])}       # "\n\n" + "\n\n" vs "\n\n\n\n"


def _real_gate_texts() -> dict[str, str]:
    """T1-T4 from gate/texts/, T5 and T6 rebuilt from the pinned FineWeb-2 parquet (as RUNBOOK step 7's
    --work-only does), each checked against MANIFEST.json; nothing is written."""
    from tokenizers import Tokenizer

    from gate import build_gate_text
    from test_bench_support import mini_parquet

    pub = common.EXP_DIR / "gate" / "texts"
    texts = {t: next(iter(sorted(pub.glob(f"{t}_*.txt")))).read_text(encoding="utf-8") for t in kl_8v4.PUBLIC_TEXTS}
    tok = Tokenizer.from_file(str(kolibri_tok_dir() / "tokenizer.json"))
    _, work = build_gate_text.build_web(tok, mini_parquet())
    texts.update({t: work[t]["text"] for t in ("T5", "T6")})
    want = kl_8v4.manifest_text_sha256(pub)
    assert {t: common.sha256_bytes(s.encode("utf-8")) for t, s in texts.items()} == {t: want[t] for t in texts}
    return texts


def test_wrapper_text_boundary_on_the_real_gate_texts():
    """Pins the known boundary. Kolibri's, Qwen3.6's and Qwen3.8's wrappers end on the token "\\n\\n"; T1 and T2
    start with "\\n" and T6 with "\\n\\n", so the scored sequence has a separate newline token where the joint
    encoding has one merged token ("\\n\\n\\n" or "\\n\\n\\n\\n"); every other id is the same. T3-T5 start with a
    letter and are unaffected, and so is Gemma 4 on all six (its wrapper ends on the special token <channel|>).
    The H8 record carries the same result per text (families.<f>.texts.<T>.wrapper_boundary)."""
    from tokenizers import Tokenizer

    texts = _real_gate_texts()
    assert [t for t, s in texts.items() if s.startswith("\n")] == ["T1", "T2", "T6"]
    assert texts["T6"].startswith("\n\n") and not any(texts[t].startswith("\n\n") for t in ("T1", "T2"))
    for name, d in _real_dirs().items():
        w = kl_8v4.chat_wrapper(name, d)
        P = w["prompt_ids"]
        raw = Tokenizer.from_file(str(d / "tokenizer.json"))
        key = "kolibri" if name == "kolibri" else "qwen"
        for t, s in texts.items():
            ids, _ = kl_8v4.tokenize(d / "tokenizer.json", s)
            joint = raw.encode(w["rendered"] + s, add_special_tokens=False).ids
            got = kl_8v4.wrapper_boundary(d / "tokenizer.json", w["rendered"], s, P, ids)
            if name == "gemma4" or t in ("T3", "T4", "T5"):
                assert joint == P + ids and got == {"joint_equals_scored": True}, (name, t)
                continue
            sep, merged = (_NL4_SPLIT if t == "T6" else _NL2_SPLIT)[key]
            assert P[-1] == sep[0] and ids[0] == sep[1], (name, t)     # the wrapper's "\\n\\n", then the text's newline
            assert joint == P[:-1] + merged + ids[1:], (name, t)        # one merged token; the rest identical
            assert got == {"joint_equals_scored": False, "scored_ids": sep, "joint_ids": merged,
                           "from_text_token": -1}, (name, t)
        if name == "gemma4":
            assert P[-1] == 101 and raw.id_to_token(101) == "<channel|>"


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


def _manual_nll(model, inputs, sf, ids, chunk=None):
    """NLL of ids read from rows sf.. of every logit row of the whole input (bf16-rounded logits): one
    uncached forward (chunk None), or the same chunked forward through the cache, all rows kept."""
    import mlx.core as mx

    if chunk is None:
        logits = model(mx.array(inputs, dtype=mx.int32)[None])[0].astype(mx.float32)
    else:
        logits = mx.concatenate([lg for _, lg, _ in kl_8v4.forward_chunks(model, inputs, chunk)], axis=0)
    assert logits.shape[0] == len(inputs)
    lp = np.array(kl_8v4.logprobs(logits, True), dtype=np.float64)
    return np.array([-lp[sf + j, i] for j, i in enumerate(ids)])


@pytest.mark.parametrize("chunk", [5, 64, 2048])
def test_only_the_text_tokens_are_scored_after_the_wrapper(chat_models, chunk, tmp_path):
    """Teacher forcing after the wrapper scores exactly the text's tokens: the per-position NLL equals every
    row of the same forward of prompt + text read from row len(prompt) - 1 on, for chunks smaller than the
    prompt (whole chunks of prompt rows are skipped), mid-sized and whole (then also an uncached forward); the
    K8 dump holds one row per text token; the dump path equals the loaded path."""
    from mlx_lm import load as mlx_load

    m8, _ = mlx_load(str(chat_models["a"]), trust_remote_code=True)
    m4, _ = mlx_load(str(chat_models["b"]), trust_remote_code=True)
    w = kl_8v4.chat_wrapper("kolibri", chat_models["a"])
    assert w["record"]["prefill"] == "empty_think"
    text = TEXTS["T3"]
    ids, _ = kl_8v4.tokenize(chat_models["a"] / "tokenizer.json", text)
    inputs, sf = kl_8v4.scored_input(w["prompt_ids"], ids)
    assert sf == len(w["prompt_ids"]) - 1 == 72   # 73 byte tokens: chunk 5 and 64 start inside the prompt
    st, native = kl_8v4.compare_loaded(m8, m4, inputs, ids, chunk, with_fp32=True, score_from=sf)
    assert all(v.shape == (len(ids),) for v in st.values())
    np.testing.assert_allclose(st["nll8"], _manual_nll(m8, inputs, sf, ids, chunk), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(st["nll4"], _manual_nll(m4, inputs, sf, ids, chunk), rtol=1e-6, atol=1e-6)
    if chunk >= len(inputs):
        np.testing.assert_allclose(st["nll8"], _manual_nll(m8, inputs, sf, ids), rtol=1e-5, atol=1e-5)
    files, _ = kl_8v4.dump_logprobs(m8, inputs, tmp_path, "T3", chunk, score_from=sf)
    assert np.load(files["bf16"], mmap_mode="r").shape == (len(ids), 1024)
    sd, _ = kl_8v4.compare_dump(m4, inputs, ids, files, chunk, score_from=sf)
    for k in ("kl", "nll8", "nll4", "agree", "kl_fp32"):
        np.testing.assert_allclose(sd[k], st[k], rtol=1e-6, atol=1e-7, err_msg=k)
    with pytest.raises(common.BenchError, match="do not score"):
        kl_8v4.compare_loaded(m8, m4, inputs, ids[:-1], chunk, score_from=sf)
    with pytest.raises(common.BenchError, match="score_from"):
        list(kl_8v4.scored_chunks(m8, inputs, len(inputs)))


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


def test_kl_cell_dump_path_equals_loaded_path(bench_env, text_dirs, chat_models):
    """Kolibri (K8 dump, then K4) and Qwen3.6 (both loaded) on the same tiny weights: the tiny ChatML template
    and Qwen3.6's committed template render the same wrapper there, so the two paths must agree block for
    block. Gemma 4's own template gives a different wrapper (and so different KL) over the same text tokens."""
    a, b = chat_models["a"], chat_models["b"]
    mapping = {"K8": a, "K4": b, "Q36-8": a, "Q36-4": b, "G8": a, "G4": b}
    path, out = _run(bench_env, text_dirs, mapping, families=("kolibri", "qwen3_6", "gemma4"), chunk=300)
    assert out["complete"] is True and path.name.startswith("kl_8v4_")
    assert path.read_text() == common.dumps(out) + "\n"  # sorted keys
    assert {c for c in bench_env.calls if c[0] == "gate"} == {("gate", "K8"), ("gate", "K4")}
    kol, gem = out["families"]["kolibri"], out["families"]["qwen3_6"]
    assert kol["wrapper"]["prompt_ids"] == gem["wrapper"]["prompt_ids"]
    assert out["families"]["gemma4"]["wrapper"]["prompt_ids"] != kol["wrapper"]["prompt_ids"]
    for t in TEXTS:
        gb = out["families"]["gemma4"]["texts"][t]["blocks"]
        assert [x["tokens"] for x in gb] == [x["tokens"] for x in kol["texts"][t]["blocks"]]
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
    # Amendment 5: each family's wrapper is recorded (prompt, kwargs, template sha256), and only text tokens
    # are scored: one scored row per text token, from row len(prompt) - 1 on.
    for fam, kw in (("kolibri", {"reasoning_effort": "none"}), ("qwen3_6", {"enable_thinking": False}),
                    ("gemma4", {"enable_thinking": False})):
        wr = out["families"][fam]["wrapper"]
        assert wr["kwargs"] == kw and wr["messages"] == [{"role": "user", "content": "Write a text."}]
        assert wr["add_generation_prompt"] is True and len(wr["template_sha256"]) == 64
        assert wr["prompt_ids_4bit_identical"] is True and wr["prompt_tokens"] == len(wr["prompt_ids"]) > 1
        for t in TEXTS:
            assert out["families"][fam]["texts"][t]["score_from"] == wr["prompt_tokens"] - 1
    assert out["families"]["kolibri"]["wrapper"]["prefill"] == "empty_think"
    assert out["families"]["gemma4"]["wrapper"]["prefill"] == "empty_channel"
    assert out["settings"]["wrapper_user_message"] == "Write a text." and out["settings"]["wrapper_effort"] == "none"
    assert "prefix" not in kol and "prefix_rule" not in out["settings"]
    # the wrapper/text boundary and the first scored rows, per text (review of Amendment 5): on the byte-level
    # tiny tokenizer nothing merges; first_rows holds the first two per-position values of every stat (Kolibri's
    # kl_fp32 too), within block 0, and the dump path's equal the loaded path's on the same weights and wrapper
    for fam in ("kolibri", "qwen3_6", "gemma4"):
        for t in TEXTS:
            rec = out["families"][fam]["texts"][t]
            assert rec["wrapper_boundary"] == {"joint_equals_scored": True}, (fam, t)
            fr = rec["first_rows"]
            assert fr["rows"] == 2 and fr["token_first_bytes"] == [0, 1, 2]
            assert set(fr) == {"rows", "token_first_bytes", "kl", "nll8", "nll4", "agree"} | (
                {"kl_fp32"} if fam == "kolibri" else set())
            assert all(len(v) == 2 for k, v in fr.items() if k not in ("rows", "token_first_bytes"))
            assert sum(fr["kl"]) <= rec["blocks"][0]["kl_sum"] + 1e-9
    for t in TEXTS:
        a, b = kol["texts"][t]["first_rows"], gem["texts"][t]["first_rows"]
        for k in ("kl", "nll8", "nll4", "agree"):
            np.testing.assert_allclose(a[k], b[k], rtol=1e-6, atol=1e-7, err_msg=f"{t} {k}")
    assert "wrapper_boundary" in out["settings"]
    # byte blocks over the text's own bytes, unchanged by the wrapper
    for t, s in TEXTS.items():
        assert out["texts"][t]["block_bounds"] == kl_8v4.byte_blocks(s.encode("utf-8"))
    # the K8 dump: two fp32 arrays per text (one row per text token), hashed in the record, in $EXP036_WORK/kl/
    files = out["k8_dump"]["files"]
    assert len(files) == 12 and all(len(v) == 64 for v in files.values())
    dumped = sorted((bench_env.root / "work" / "kl").rglob("*.npy"))
    assert len(dumped) == 12
    for f in dumped:
        arr = np.load(f, mmap_mode="r")
        t = f.name.split("_")[1].split(".")[0]
        assert arr.dtype == np.float32 and arr.shape == (len(TEXTS[t].encode("utf-8")), 1024)
    # no T5/T6 (work-only) text in the record
    assert "Übergröße" not in path.read_text() and "Neue Zeile" not in path.read_text()
    # the output still feeds analysis/verdicts.py: kl_models and the H8 arrays
    from analysis import verdicts as V

    models = V.kl_models(out)
    assert set(models) == {"kolibri", "qwen3_6", "gemma4"} and set(V.kl_models(out, fp32=True)) == {"kolibri"}
    num, den_b, den_t = V._h8_arrays(models, ["kolibri", "qwen3_6", "gemma4"], list(TEXTS))
    for j, t in enumerate(TEXTS):
        assert num[j].shape == (3, 8) and den_b[j][0].sum() == len(TEXTS[t].encode("utf-8"))
        assert den_t[j][0].sum() == kol["texts"][t]["tokens"] == len(TEXTS[t].encode("utf-8"))


def test_kl_cell_same_model_gives_zero_kl_and_full_agreement(bench_env, text_dirs, chat_models):
    a = chat_models["a"]
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


def test_kl_cell_refuses_differing_8bit_and_4bit_wrappers(bench_env, text_dirs, chat_models, tmp_path):
    b2 = _with_chat_template(chat_models["b"], tmp_path / "b2")
    tc = json.loads((b2 / "tokenizer_config.json").read_text())
    tc["chat_template"] = TINY_CHATML.replace("<think>", "<thinking>")
    (b2 / "tokenizer_config.json").write_text(json.dumps(tc))
    with pytest.raises(common.BenchError, match="chat wrappers render different prompt ids"):
        _run(bench_env, text_dirs, {"K8": chat_models["a"], "K4": b2}, families=("kolibri",))
    assert not list(bench_env.results.glob("kl_8v4_*"))


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
