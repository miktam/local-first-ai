# SPDX-License-Identifier: MIT
"""bench/tokenizer_ratio.py (H5).

Mechanics on synthetic parquet files (exact counts with a byte tokenizer),
then the real FineWeb-2 parquet and the real tokenizers on the mini, on a
late slice (rows 60,000+) that is disjoint from the H5 corpus (rows 0-4,999)
and from the gate texts T5/T6/T9 (from row 5,000 on). No assertion here looks
at a bytes/token value, so no test reads out the H5 statistic."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from test_bench_support import (  # noqa: F401  (fixtures)
    bench_env,
    kolibri_tok_dir,
    mini_dir,
    mini_parquet,
    tiny_models,
)

from bench import common, tokenizer_ratio as tr

SLICE_START = 60_000
SLICE_N = 100

DOCS = [
    "Ein kurzer Satz.",
    "Größe 12345 und 678 Äpfel, 9 Birnen: 2026-10-03.",
    "Straße ohne Ziffern, aber mit Umlauten: ä ö ü ß.",
    "1 2 3 4 5 6 7 8 9 0",
    "Zeile eins\nZeile zwei 🙂",
    "Noch ein Dokument mit 3 Wörtern.",
    "Letztes Dokument.",
]


def _parquet(path: Path, texts: list[str], group: int = 3) -> Path:
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pa.table({"text": texts, "id": [f"doc{i}" for i in range(len(texts))]})
    pq.write_table(table, str(path), row_group_size=group)
    return path


def _byte_tokenizers(tiny_models) -> dict[str, Path]:
    f = tiny_models["a"] / "tokenizer.json"
    return {name: f for name in tr.TOKENIZER_FOLDERS}


def test_iter_docs_file_order_across_row_groups(tmp_path):
    p = _parquet(tmp_path / "x.parquet", DOCS, group=3)
    assert [i for i, _ in tr.iter_docs(p, 0, 5)] == [0, 1, 2, 3, 4]
    got = list(tr.iter_docs(p, 2, 4))
    assert got == [(i, DOCS[i]) for i in range(2, 6)]
    assert list(tr.iter_docs(p, 6, 10)) == [(6, DOCS[6])]


def test_ratio_counts_exactly_with_a_byte_tokenizer(tmp_path, tiny_models):
    p = _parquet(tmp_path / "x.parquet", DOCS)
    res = tr.ratio(p, _byte_tokenizers(tiny_models), n=len(DOCS))
    assert res["columns"] == ["index", "bytes", "chars", "digits", "tokens_kolibri", "tokens_gemma4", "tokens_qwen3_6", "tokens_qwen3_8"]
    for row, text in zip(res["docs"], DOCS):
        b = len(text.encode("utf-8"))
        assert row[:4] == [DOCS.index(text), b, len(text), sum(c.isdigit() and c.isascii() for c in text)]
        assert row[4:] == [b] * 4
    s = res["summary"]["all"]
    assert s["bytes_per_token"]["kolibri"] == 1.0 and s["kolibri_ratio_vs"] == {"gemma4": 1.0, "qwen3_6": 1.0, "qwen3_8": 1.0}
    dense = res["summary"]["digit_dense"]
    expected = [t for t in DOCS if sum(c.isdigit() for c in t) / len(t) >= 0.05]
    assert dense["n_docs"] == len(expected) and dense["bytes"] == sum(len(t.encode()) for t in expected)
    assert res["tokenizer_identity"]["qwen3_8_vs_qwen3_6"] == {"identical": True, "n_docs_differing": 0}
    assert res["corpus_sha256"] == tr.corpus_sha256(list(enumerate(DOCS)))
    with pytest.raises(common.BenchError, match="7 documents"):
        tr.ratio(p, _byte_tokenizers(tiny_models), n=8)


def test_run_cell_writes_counts_only(bench_env, tmp_path, tiny_models, monkeypatch):
    p = _parquet(tmp_path / "x.parquet", DOCS)
    monkeypatch.setattr(tr, "_expected_parquet_sha256", lambda: common.sha256_file(p))
    path = tr.run_cell(bench_env.results, parquet=p, tokenizer_files=_byte_tokenizers(tiny_models), n=len(DOCS), lite_dir=tmp_path / "nolite")
    assert path.parent == bench_env.results and path.name.startswith("tokenizer_")
    out = json.loads(path.read_text())
    assert path.read_text() == common.dumps(out) + "\n"
    assert out["complete"] is True and out["t_end"] >= out["t_start"]
    assert out["parquet"]["sha256"] == out["parquet"]["expected_sha256"]
    assert "metal" not in out["environment"]
    assert out["en_descriptive"]["mmlu_prox_lite_en_questions"] == {"status": "missing"}
    assert out["en_descriptive"]["gate_T1"]["status"] in ("ok", "missing")
    text = path.read_text()
    for d in DOCS:
        for k in range(0, max(1, len(d) - 12), 6):
            assert d[k : k + 12] not in text
    assert bench_env.calls[0] == ("identity",)


def test_parquet_sha_mismatch_is_refused(bench_env, tmp_path, tiny_models, monkeypatch):
    p = _parquet(tmp_path / "x.parquet", DOCS)
    monkeypatch.setattr(tr, "_expected_parquet_sha256", lambda: "0" * 64)
    with pytest.raises(common.BenchError, match="assets.json"):
        tr.run_cell(bench_env.results, parquet=p, tokenizer_files=_byte_tokenizers(tiny_models), n=3)
    assert not list(bench_env.results.glob("tokenizer_*"))


def test_compare_and_cli(bench_env, tmp_path, tiny_models, monkeypatch):
    p = _parquet(tmp_path / "x.parquet", DOCS)
    monkeypatch.setattr(tr, "_expected_parquet_sha256", lambda: None)
    tokdir = tiny_models["a"]
    args = ["--parquet", str(p), "--n", "5"] + [f"--tokenizer={n}={tokdir}" for n in tr.TOKENIZER_FOLDERS]
    first = tmp_path / "a" / "tok.json"
    assert tr.main(args + ["--out", str(first)]) == 0
    assert tr.main(args + ["--out", str(tmp_path / "b.json"), "--compare", str(first)]) == 0
    assert tr.compare(first, tmp_path / "b.json")["identical"] is True
    assert tr.main(["--parquet", str(p), "--n", "4", "--out", str(tmp_path / "c.json"), "--compare", str(first)] + args[4:]) == 1
    with pytest.raises(SystemExit):
        tr.main(["--tokenizer", "llama=/x"])


def test_lite_en_questions_picks_english_test_files(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    for lang in ("en", "de"):
        (tmp_path / lang).mkdir()
        pq.write_table(pa.table({"question": [f"{lang} q1", f"{lang} q2"]}), str(tmp_path / lang / "test-00000-of-00001.parquet"))
        pq.write_table(pa.table({"question": [f"{lang} v"]}), str(tmp_path / lang / "validation-00000.parquet"))
    assert tr.lite_en_questions(tmp_path) == ["en q1", "en q2"]
    assert tr.lite_en_questions(tmp_path / "absent") is None


def test_h5_cell_never_imports_mlx(tmp_path, tiny_models):
    p = _parquet(tmp_path / "x.parquet", DOCS)
    code = (
        "import sys, json; sys.path.insert(0, %r); from bench import tokenizer_ratio as tr; "
        "tr.ratio(%r, {'kolibri': %r}, n=3); print('mlx' in sys.modules)"
    ) % (str(common.EXP_DIR), str(p), str(tiny_models["a"] / "tokenizer.json"))
    out = subprocess.run([sys.executable, "-B", "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def test_same_reader_as_tasks_fineweb2_if_present(tmp_path):
    try:
        from tasks import fineweb2
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "tasks":
            pytest.skip("tasks/fineweb2.py not built yet")
        raise
    p = _parquet(tmp_path / "x.parquet", DOCS)

    def text_of(x):  # (index, text), a dict with "text", or a bare string
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            return x["text"]
        return x[1]

    theirs = [text_of(x) for x in fineweb2.iter_docs(p, start=0, n=5)]
    assert theirs == [t for _, t in tr.iter_docs(p, 0, 5)]


# --------------------------------------------------------------------------
# Real tokenizers and the real parquet (mini build-time assets)
# --------------------------------------------------------------------------


def _real_files() -> dict[str, Path]:
    return {
        "kolibri": kolibri_tok_dir() / "tokenizer.json",
        "gemma4": mini_dir("mlx-community", "gemma-4-26b-a4b-it-8bit") / "tokenizer.json",
        "qwen3_6": mini_dir("mlx-community", "Qwen3.6-35B-A3B-8bit") / "tokenizer.json",
        "qwen3_8": mini_dir("mlx-community", "Qwen3.8-27B-8bit") / "tokenizer.json",
    }


def test_real_parquet_matches_the_assets_pin():
    p = mini_parquet()
    assert common.sha256_file(p) == tr._expected_parquet_sha256()


def test_real_slice_counts_and_tokenizer_identity():
    p = mini_parquet()
    res = tr.ratio(p, _real_files(), n=SLICE_N, start=SLICE_START)
    assert res["n_docs"] == SLICE_N and res["docs"][0][0] == SLICE_START
    for row in res["docs"]:
        assert row[1] >= row[2] > 0 and all(t > 0 for t in row[4:])
    ident = res["tokenizer_identity"]["qwen3_8_vs_qwen3_6"]
    assert set(ident) == {"identical", "n_docs_differing"}


def _slice_texts() -> list[str]:
    return [t for _, t in tr.iter_docs(mini_parquet(), SLICE_START, SLICE_N)]


def test_real_raw_tokenizers_equal_the_transformers_tokenizers():
    """The harness (mlx_lm -> transformers) and H5 (raw tokenizers) give the
    same ids for every family, on German web text."""
    from tokenizers import Tokenizer
    from transformers import AutoTokenizer

    texts = _slice_texts()
    for name, f in _real_files().items():
        raw = Tokenizer.from_file(str(f))
        hf = AutoTokenizer.from_pretrained(str(f.parent))
        bad = [i for i, t in enumerate(texts) if raw.encode(t, add_special_tokens=False).ids != hf.encode(t, add_special_tokens=False)]
        assert bad == [], (name, bad[:5])


@pytest.mark.parametrize(
    "mc,up",
    [
        ("gemma-4-26b-a4b-it-8bit", "gemma-4-26B-A4B-it@20da991a"),
        ("gemma-4-26b-a4b-it-8bit", "gemma-4-26B-A4B-it@4d7ae498"),
        ("Qwen3.6-35B-A3B-8bit", "Qwen3.6-35B-A3B@995ad96e"),
        ("Qwen3.8-27B-8bit", "Qwen3.8-27B@1d4bf0f2"),
    ],
)
def test_mlx_community_tokenizers_give_upstream_ids(mc, up):
    """The pinned peer folders and the upstream repos may differ in bytes;
    the token ids on German web text must not (HYPOTHESIS H5 uses the
    pinned folders)."""
    from tokenizers import Tokenizer

    a = Tokenizer.from_file(str(mini_dir("mlx-community", mc) / "tokenizer.json"))
    b = Tokenizer.from_file(str(mini_dir("upstream", up) / "tokenizer.json"))
    for t in _slice_texts():
        assert a.encode(t, add_special_tokens=False).ids == b.encode(t, add_special_tokens=False).ids
