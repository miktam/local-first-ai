# SPDX-License-Identifier: MIT
"""H5 — German tokenizer economy (HYPOTHESIS H5; BUILD_SPEC §5.7
tokenizer_ratio.py). CPU only: this module never imports mlx.
Output: results/tokenizer_<UTC>.json (or --out).

Corpus: the first 5,000 documents of the FineWeb-2 deu_Latn test parquet
(assets.json pin, sha256 checked) in file order, column "text".
Per document: UTF-8 bytes, characters, ASCII digits, and the token count of
each tokenizer, encoded with add_special_tokens=False, no template, no
normalisation, through raw `tokenizers` from each tokenizer.json:
  kolibri  Kolibri-1-BF16/tokenizer.json (id parity with the harness: G0)
  gemma4   gemma-4-26b-a4b-it-8bit/tokenizer.json   (pinned peer folder)
  qwen3_6  Qwen3.6-35B-A3B-8bit/tokenizer.json      (pinned peer folder)
  qwen3_8  Qwen3.8-27B-8bit/tokenizer.json          (hash check; reported
           as one with qwen3_6 when its ids equal Qwen3.6's on every document)
Bytes/token = sum bytes / sum tokens. Only counts are written: no document
text, no per-document text hash (an overall corpus sha256 binds the input).

The document bootstrap, the intersection-union test and the sub-verdict are
analysis/'s. The summary block holds descriptive point estimates only, for the
whole corpus and for the digit-dense subset (>= 5 % ASCII digits per
character). EN descriptive: our EN gate texts T1 and T2 and MMLU-ProX-Lite EN
question text, labelled "not FineWeb".

The mini re-runs this cell as a check (HYPOTHESIS H5 "The mini re-runs it"):
  python bench/tokenizer_ratio.py --tokenizer kolibri=DIR ... --out FILE --compare results/tokenizer_<UTC>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import common  # noqa: E402

SPEC = "HYPOTHESIS H5; BUILD_SPEC §5.7 tokenizer_ratio.py"
H5_N = 5000
H5_PARQUET_REL = "data/deu_Latn/test/000_00000.parquet"
TOKENIZER_FOLDERS = {
    "kolibri": common.KOLIBRI_BF16_FOLDER,
    "gemma4": "gemma-4-26b-a4b-it-8bit",
    "qwen3_6": "Qwen3.6-35B-A3B-8bit",
    "qwen3_8": "Qwen3.8-27B-8bit",
}
COMPARED = ("gemma4", "qwen3_6")  # H5: Kolibri against both
DIGIT_DENSE_MIN = 0.05
BATCH = 500
DOC_COLUMNS = ("index", "bytes", "chars", "digits")


def default_tokenizer_files() -> dict[str, Path]:
    return {name: common.models_dir() / folder / "tokenizer.json" for name, folder in TOKENIZER_FOLDERS.items()}


def iter_docs(parquet: Path, start: int = 0, n: int = H5_N) -> Iterator[tuple[int, str]]:
    """(row index, text) for rows start .. start+n-1 in file order."""
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(parquet))
    i = 0
    stop = start + n
    for batch in pf.iter_batches(columns=["text"], batch_size=1000):
        col = batch.column(0).to_pylist()
        if i + len(col) <= start:
            i += len(col)
            continue
        for text in col:
            if i >= stop:
                return
            if i >= start:
                if not isinstance(text, str):
                    raise common.BenchError(f"row {i}: text is not a string")
                yield i, text
            i += 1
        if i >= stop:
            return


def _load(files: dict[str, Path]):
    from tokenizers import Tokenizer

    toks = {}
    for name, f in files.items():
        if not Path(f).is_file():
            raise common.BenchError(f"tokenizer {name}: {common.redact_path(f)} is missing")
        toks[name] = Tokenizer.from_file(str(f))
    return toks


def count_docs(docs: list[tuple[int, str]], tokenizers: dict) -> tuple[list[list[int]], dict]:
    """Per-document rows [index, bytes, chars, digits, tokens per tokenizer]
    and, for qwen3_8 when present, the number of documents whose ids differ
    from qwen3_6's."""
    names = list(tokenizers)
    rows: list[list[int]] = []
    differ = 0
    for b0 in range(0, len(docs), BATCH):
        chunk = docs[b0 : b0 + BATCH]
        texts = [t for _, t in chunk]
        encs = {name: tok.encode_batch(texts, add_special_tokens=False) for name, tok in tokenizers.items()}
        for j, (idx, text) in enumerate(chunk):
            rows.append(
                [
                    idx,
                    len(text.encode("utf-8")),
                    len(text),
                    sum(1 for c in text if "0" <= c <= "9"),
                    *[len(encs[name][j].ids) for name in names],
                ]
            )
            if "qwen3_8" in encs and "qwen3_6" in encs and encs["qwen3_8"][j].ids != encs["qwen3_6"][j].ids:
                differ += 1
    ident = None
    if "qwen3_8" in tokenizers and "qwen3_6" in tokenizers:
        ident = {"identical": differ == 0, "n_docs_differing": differ}
    return rows, {"qwen3_8_vs_qwen3_6": ident}


def summarize(rows: list[list[int]], names: list[str]) -> dict:
    """Descriptive point estimates; no verdict."""
    if not rows:
        return {"n_docs": 0}
    col = {name: 4 + i for i, name in enumerate(names)}
    b = sum(r[1] for r in rows)
    c = sum(r[2] for r in rows)
    t = {name: sum(r[col[name]] for r in rows) for name in names}
    bpt = {name: b / t[name] for name in names if t[name]}
    out = {
        "n_docs": len(rows),
        "bytes": b,
        "chars": c,
        "tokens": t,
        "bytes_per_token": bpt,
        "chars_per_token": {name: c / t[name] for name in names if t[name]},
    }
    if "kolibri" in bpt:
        out["kolibri_ratio_vs"] = {name: bpt["kolibri"] / bpt[name] for name in names if name != "kolibri" and name in bpt}
    return out


def corpus_sha256(docs: list[tuple[int, str]]) -> str:
    """sha256 over "index<TAB>sha256(text)" lines: binds the exact input."""
    return common.sha256_text("".join(f"{i}\t{common.sha256_text(t)}\n" for i, t in docs))


def ratio(parquet: Path, tokenizer_files: dict[str, Path], n: int = H5_N, start: int = 0) -> dict:
    """BUILD_SPEC §5.7: ratio(parquet, tokenizers, n=5000) -> per-doc
    (bytes, chars, tokens) plus the checks and descriptive summaries."""
    toks = _load(tokenizer_files)
    names = list(toks)
    docs = list(iter_docs(parquet, start, n))
    if len(docs) != n:
        raise common.BenchError(f"the parquet holds {len(docs)} documents from row {start}; {n} are needed")
    rows, ident = count_docs(docs, toks)
    dense = [r for r in rows if r[2] and r[3] / r[2] >= DIGIT_DENSE_MIN]
    return {
        "start": start,
        "n_docs": len(rows),
        "columns": list(DOC_COLUMNS) + [f"tokens_{name}" for name in names],
        "docs": rows,
        "corpus_sha256": corpus_sha256(docs),
        "tokenizer_identity": ident,
        "summary": {"all": summarize(rows, names), "digit_dense": summarize(dense, names)},
    }


# ---------------------------------------------------------------------------
# EN descriptive (not FineWeb)
# ---------------------------------------------------------------------------


def _gate_text(public_dir: Path, t: str) -> Optional[Path]:
    found = sorted(Path(public_dir).glob(f"{t}_*.txt"))
    return found[0] if len(found) == 1 else None


def lite_en_questions(lite_dir: Path) -> Optional[list[str]]:
    """The "question" column of MMLU-ProX-Lite's English test parquet(s):
    files under lite_dir whose relative path has a component "en" (or a stem
    starting "en") and mentions "test". None when no such file exists."""
    import pyarrow.parquet as pq

    lite_dir = Path(lite_dir)
    if not lite_dir.is_dir():
        return None
    files = []
    for p in sorted(lite_dir.rglob("*.parquet")):
        rel = p.relative_to(lite_dir)
        parts = [x.lower() for x in rel.parts]
        is_en = "en" in parts[:-1] or parts[-1].startswith("en")
        if is_en and "test" in rel.as_posix().lower():
            files.append(p)
    qs: list[str] = []
    for f in files:
        t = pq.read_table(str(f))
        if "question" in t.column_names:
            qs.extend(str(q) for q in t.column("question").to_pylist() if q)
    return qs or None


def en_descriptive(tokenizers: dict, public_dir: Path, lite_dir: Path) -> dict:
    names = list(tokenizers)
    out: dict[str, dict] = {}
    sources: dict[str, Optional[list[str]]] = {}
    for t in ("T1", "T2"):
        p = _gate_text(public_dir, t)
        sources[f"gate_{t}"] = [p.read_text(encoding="utf-8")] if p else None
    sources["mmlu_prox_lite_en_questions"] = lite_en_questions(lite_dir)
    for label, texts in sources.items():
        if not texts:
            out[label] = {"status": "missing"}
            continue
        rows, _ = count_docs(list(enumerate(texts)), tokenizers)
        s = summarize(rows, names)
        out[label] = {"status": "ok", "label": "not FineWeb", **{k: s[k] for k in ("n_docs", "bytes", "chars", "tokens", "bytes_per_token") if k in s}}
        if "kolibri_ratio_vs" in s:
            out[label]["kolibri_ratio_vs"] = s["kolibri_ratio_vs"]
    return out


# ---------------------------------------------------------------------------
# The cell
# ---------------------------------------------------------------------------


def _expected_parquet_sha256() -> Optional[str]:
    for d in common.assets_json().get("datasets", []):
        if d.get("repo") == "HuggingFaceFW/fineweb-2":
            return (d.get("sha256") or {}).get(H5_PARQUET_REL)
    return None


def run_cell(
    results_dir: Optional[Path] = None,
    *,
    parquet: Optional[Path] = None,
    tokenizer_files: Optional[dict[str, Path]] = None,
    n: int = H5_N,
    start: int = 0,
    public_dir: Optional[Path] = None,
    lite_dir: Optional[Path] = None,
    out: Optional[Path] = None,
    verify_parquet_sha: bool = True,
) -> Path:
    results_dir = Path(results_dir or common.default_results_dir())
    common.require_identity()
    t_start = common.utc_iso()
    parquet = Path(parquet or common.fineweb_parquet())
    if not parquet.is_file():
        raise common.BenchError(f"FineWeb-2 parquet missing: {common.redact_path(parquet)}")
    files = {k: Path(v) for k, v in (tokenizer_files or default_tokenizer_files()).items()}
    sha = common.sha256_file(parquet)
    expected = _expected_parquet_sha256()
    if verify_parquet_sha and expected and sha != expected:
        raise common.BenchError(f"parquet sha256 {sha} != assets.json {expected}")
    de = ratio(parquet, files, n=n, start=start)
    public_dir = Path(public_dir) if public_dir else common.EXP_DIR / "gate" / "texts"
    lite_dir = Path(lite_dir) if lite_dir else common.data_dir() / "MMLU-ProX-Lite"
    en = en_descriptive(_load(files), public_dir, lite_dir)
    import pyarrow.parquet as pq

    result = {
        "cell": "tokenizer",
        "spec": SPEC,
        "complete": True,
        "t_start": t_start,
        "t_end": common.utc_iso(),
        "environment": common.environment(include_mlx=False),
        "parquet": {
            "file": common.redact_path(parquet),
            "sha256": sha,
            "expected_sha256": expected,
            "rows_total": pq.ParquetFile(str(parquet)).metadata.num_rows,
        },
        "tokenizers": {
            name: {"file": common.redact_path(f), "sha256": common.sha256_file(f)} for name, f in files.items()
        },
        "settings": {
            "n": n,
            "start": start,
            "order": "file order",
            "encoding": "raw tokenizers, add_special_tokens=False, no template, no normalisation",
            "bytes": "UTF-8 bytes of the text column",
            "digit_dense_min_fraction": DIGIT_DENSE_MIN,
            "digits": "ASCII 0-9 per character",
        },
        "de": de,
        "en_descriptive": en,
    }
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        path = common.new_output_path(results_dir, "tokenizer")
    common.write_json_new(path, result)
    return path


def compare(a: Path, b: Path) -> dict:
    """Per-document counts and corpus hash of two tokenizer records."""
    ra, rb = json.loads(Path(a).read_text()), json.loads(Path(b).read_text())
    da, db = ra["de"], rb["de"]
    same_cols = da["columns"] == db["columns"]
    same_docs = same_cols and da["docs"] == db["docs"]
    return {
        "corpus_sha256_equal": da["corpus_sha256"] == db["corpus_sha256"],
        "columns_equal": same_cols,
        "docs_equal": same_docs,
        "identical": same_docs and da["corpus_sha256"] == db["corpus_sha256"],
    }


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--parquet", type=Path, default=None)
    ap.add_argument("--n", type=int, default=H5_N)
    ap.add_argument("--tokenizer", action="append", default=[], metavar="NAME=DIR_OR_JSON", help="override a tokenizer (kolibri, gemma4, qwen3_6, qwen3_8)")
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="write here instead of results/tokenizer_<UTC>.json")
    ap.add_argument("--compare", type=Path, default=None, help="an earlier record; exit 1 unless the counts are identical")
    args = ap.parse_args(argv)
    files = default_tokenizer_files()
    for spec in args.tokenizer:
        name, _, val = spec.partition("=")
        if name not in TOKENIZER_FOLDERS or not val:
            ap.error(f"bad --tokenizer {spec!r}")
        p = Path(val).expanduser()
        files[name] = p / "tokenizer.json" if p.is_dir() else p
    path = run_cell(args.results_dir, parquet=args.parquet, tokenizer_files=files, n=args.n, out=args.out)
    res = {"file": path.name}
    code = 0
    if args.compare:
        res["compare"] = compare(path, args.compare)
        code = 0 if res["compare"]["identical"] else 1
    print(common.dumps(res))
    return code


if __name__ == "__main__":
    sys.exit(main())
