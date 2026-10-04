"""Synthetic, dataset-shaped fixtures for the tasks/ tests (BUILD_SPEC §2 Rights: no real item text).

Every string here is made up ("Synthetic question 3 about widget calibration?"). The files mirror the layout
of the real local snapshots under $EXP036_DATA (assets.json paths), so the loaders and the manifest builder
run on them unchanged.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

MMLU_CATS = [
    "engineering", "physics", "psychology", "chemistry", "biology", "law", "philosophy",
    "computer science", "other", "economics", "business", "history", "math", "health",
]
RGB_SYSTEM = "STUB SYSTEM PROMPT FOR TESTS."
RGB_INSTRUCTION = "STUB-DOCS:\n{DOCS}\n\nSTUB-QUERY: {QUERY}"


def _jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return path


GPQA_COLUMNS = ["Record ID", "Question", "Correct Answer", "Incorrect Answer 1", "Incorrect Answer 2",
                "Incorrect Answer 3", "Subdomain"]


def gpqa_row(i: int) -> dict:
    return {
        "Record ID": f"recSYN{i:04d}",
        "Question": f"Synthetic question {i} about widget calibration under load {i * 7} units?",
        "Correct Answer": f"Widget value {i}A",
        "Incorrect Answer 1": f"Widget value {i}B [title] extra",
        "Incorrect Answer 2": f"Widget  value {i}C",
        "Incorrect Answer 3": f"Widget value {i}D [note]",
        "Subdomain": "Synthetic",
    }


# The synthetic stand-in for eval-framework's over-long question. As on the real data (Amendment 3, 2026-10-04:
# by hash, 0 of 198 gpqa_diamond.csv rows, 0 of 448 gpqa_main.csv rows, 1 gpqa_extended.csv row), it is a row of
# gpqa_extended.csv only. The real _OVERLONG_QUESTION_SHA256 matches no synthetic row; tests that need the filter
# to fire pass OVERLONG_SHA256 (or patch the vendored constant) instead.
OVERLONG_ROW_INDEX = 900
OVERLONG_SHA256 = hashlib.sha256(gpqa_row(OVERLONG_ROW_INDEX)["Question"].encode()).hexdigest()


def write_gpqa_en(root: Path, n_diamond: int = 6, n_main_extra: int = 10, n_extended_extra: int = 3) -> Path:
    """gpqa_diamond.csv ⊂ gpqa_main.csv ⊂ gpqa_extended.csv, as in Idavidrein/gpqa; the over-long stand-in
    (OVERLONG_ROW_INDEX) is the last row of gpqa_extended.csv, in neither of the files the kit reads."""
    d = root / "gpqa"
    d.mkdir(parents=True, exist_ok=True)
    diamond = [gpqa_row(i) for i in range(n_diamond)]
    main = diamond + [gpqa_row(i) for i in range(100, 100 + n_main_extra)]
    extended = main + [gpqa_row(i) for i in range(500, 500 + n_extended_extra)] + [gpqa_row(OVERLONG_ROW_INDEX)]
    for name, rows in (("gpqa_diamond.csv", diamond), ("gpqa_main.csv", main), ("gpqa_extended.csv", extended)):
        with (d / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=GPQA_COLUMNS)
            w.writeheader()
            w.writerows(rows)
    (d / "README.md").write_text("synthetic\n", encoding="utf-8")
    return d


def gpqa_de_row(i: int, diamond: bool, record_id: str | None = None) -> dict:
    row = {
        "question": f"Synthetische Frage {i} zur Kalibrierung eines Bauteils unter hoher Last?",
        "correct_answer": f"Bauteilwert {i}A",
        "incorrect_answers": [f"Bauteilwert {i}B", f"Bauteilwert {i}C", f"Bauteilwert {i}D"],
        "is_diamond": diamond,
    }
    if record_id is not None:
        row["Record ID"] = record_id
    return row


def write_gpqa_de(root: Path, n_diamond: int = 6, n_other: int = 10, with_record_ids: bool = True) -> Path:
    rows = [gpqa_de_row(i, True, f"recSYN{i:04d}" if with_record_ids else None) for i in range(n_diamond)]
    rows += [gpqa_de_row(i, False, f"recSYN{i:04d}" if with_record_ids else None) for i in range(100, 100 + n_other)]
    d = root / "gpqa-multilingual"
    _jsonl(d / "deu" / "train-00000-of-00001.jsonl", rows)
    _jsonl(d / "fra" / "train-00000-of-00001.jsonl", [gpqa_de_row(999, True)])  # another config, ignored
    return d


def mmlu_row(qid: int, cat: str, lang: str, n_opt: int = 10, style: str = "columns") -> dict:
    opts = [f"{'Option' if lang == 'en' else 'Antwort'} {qid}-{k}" for k in range(n_opt)]
    idx = qid % n_opt
    row = {"question_id": qid, "question": f"  Synthetic {lang} question {qid} in {cat}?  ",
           "answer": "ABCDEFGHIJ"[idx], "answer_index": idx, "cot_content": "", "category": cat,
           "src": "synthetic"}
    if style == "list":
        row["options"] = opts
    else:
        for k in range(10):
            row[f"option_{k}"] = opts[k] if k < n_opt else "N/A"
    return row


def write_mmlu(root: Path, lite_per_cat: int = 3, full_extra_per_cat: int = 11) -> tuple[Path, Path]:
    lite, full = root / "MMLU-ProX-Lite", root / "MMLU-ProX"
    lite_rows, full_rows = [], []
    qid = 1000
    for c, cat in enumerate(MMLU_CATS):
        for _ in range(lite_per_cat):
            lite_rows.append((qid, cat, 10 if qid % 3 else 7))
            qid += 1
        for _ in range(full_extra_per_cat):
            full_rows.append((qid, cat, 10))
            qid += 1
    for lang in ("en", "de"):
        _jsonl(lite / lang / "test-00000-of-00001.jsonl", [mmlu_row(q, c, lang, n) for q, c, n in lite_rows])
        _jsonl(lite / lang / "validation-00000-of-00001.jsonl", [mmlu_row(1, "law", lang)])
        _jsonl(full / lang / "test-00000-of-00001.jsonl",
               [mmlu_row(q, c, lang, n, style="list") for q, c, n in lite_rows + full_rows])
    # an id only in EN Lite (not parallel) must be ignored by parallel_ids
    with (lite / "en" / "test-00000-of-00001.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(mmlu_row(9999, "law", "en")) + "\n")
    return lite, full


def write_aime(root: Path, n: int = 5) -> tuple[Path, Path]:
    en = _jsonl(root / "aime26" / "test.jsonl",
                [{"id": i, "problem": f"Synthetic problem {i}: compute the sum of {i} and {i} exactly.", "answer": str(2 * i)} for i in range(n)])
    de = _jsonl(root / "aime26-multilingual" / "deu" / "test-00000-of-00001.jsonl",
                [{"id": i, "problem": f"Synthetische Aufgabe {i}: berechne die Summe von {i} und {i} genau.", "answer": str(2 * i)} for i in range(n)])
    return en.parent, de.parent.parent


def ifbench_rows(n: int = 3) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({"key": str(i), "prompt": f"Write a synthetic note number {i}. Use exactly {i + 2} bullet points.",
                     "instruction_id_list": ["count:bullets", "format:title"][: 1 + i % 2],
                     "kwargs": [{"N": i + 2, "keyword": None}, {"title": None}][: 1 + i % 2]})
    return rows


def write_ifbench(root: Path, n: int = 12) -> Path:
    d = root / "IFBench_test"
    _jsonl(d / "data" / "train-00000-of-00001.jsonl", ifbench_rows(n))
    (d / "README.md").write_text("synthetic\n", encoding="utf-8")
    return d


def rgb_rows(kind: str, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        base = {"id": i, "query": f"Synthetic RGB query {i} about the gadget fair?"}
        if kind == "en":
            base.update(answer=[[f"gadget {i}", f"Gadget-{i}"]],
                        positive=[f"POS{i}-{k} the gadget fair answer gadget {i}" for k in range(4)],
                        negative=[f"NEG{i}-{k} unrelated filler text" for k in range(3 + i % 6)])
        elif kind == "en_fact":
            npos = 1 + i % 5
            base.update(answer=f"gadget {i}", fakeanswer=f"widget {i}",
                        positive=[f"POS{i}-{k} gadget {i}" for k in range(npos)],
                        positive_wrong=[f"WRONG{i}-{k} widget {i}" for k in range(npos)],
                        negative=[f"NEG{i}-{k} filler" for k in range(2)])
        else:  # en_int
            base.update(answer=[[f"gadget {i}"], f"part {i}"], asnwer1=[], answer2=[],
                        positive=[[f"POSA{i}-{k}" for k in range(2)], [f"POSB{i}-{k}" for k in range(2)]],
                        negative=[f"NEG{i}-{k}" for k in range(4)])
        rows.append(base)
    return rows


def write_rgb(root: Path, n_en: int = 9, n_fact: int = 5, n_int: int = 20) -> Path:
    d = root / "RGB-src"
    for name, kind, n in (("en.json", "en", n_en), ("en_fact.json", "en_fact", n_fact), ("en_int.json", "en_int", n_int)):
        _jsonl(d / "data" / name, rgb_rows(kind, n))
    (d / "config").mkdir(parents=True, exist_ok=True)
    (d / "config" / "instruction.yaml").write_text(
        "en:\n"
        f"  system: {json.dumps(RGB_SYSTEM)}\n"
        f"  instruction: {json.dumps(RGB_INSTRUCTION)}\n"
        "zh:\n  system: \"stub\"\n  instruction: \"stub {QUERY} {DOCS}\"\n",
        encoding="utf-8",
    )
    return d


def write_all(root: Path) -> Path:
    """A complete synthetic $EXP036_DATA tree for every set build_manifests knows."""
    write_gpqa_en(root)
    write_gpqa_de(root)
    write_mmlu(root)
    write_aime(root)
    write_ifbench(root)
    write_rgb(root)
    return root
