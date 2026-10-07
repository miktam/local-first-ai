# SPDX-License-Identifier: MIT
"""gate/texts (HYPOTHESIS Phase 0 "Gate text"; BUILD_SPEC §5.3 build_gate_text.py).

  * T1-T4, T7, T8 are exactly 1,536 Kolibri tokens; ids and text agree;
  * the committed directory reproduces byte for byte from the committed sources
    (MANIFEST sha reproducible);
  * T5, T6, T9 rebuilt from the pinned FineWeb-2 parquet match MANIFEST.json,
    and no web text is anywhere in the experiment directory;
  * no GPQA canary; provenance recorded (T3 label, Grundgesetz URL and zip sha).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

EXP = Path(__file__).resolve().parents[1]
TEXTS = EXP / "gate" / "texts"


def _manifest() -> dict:
    return json.loads((TEXTS / "MANIFEST.json").read_text(encoding="utf-8"))


def _ids_sha(ids) -> str:
    return hashlib.sha256(",".join(str(int(i)) for i in ids).encode()).hexdigest()


@pytest.fixture(scope="module")
def raw_tok(tokenizer_dir):
    tokenizers = pytest.importorskip("tokenizers")
    return tokenizers.Tokenizer.from_file(str(tokenizer_dir / "tokenizer.json"))


def test_manifest_shape_and_provenance():
    m = _manifest()
    assert set(m["texts"]) == {"T1", "T2", "T3", "T4", "T7", "T8"}
    for tid, t in m["texts"].items():
        assert t["n_tokens"] == 1536, tid
    assert m["texts"]["T3"]["origin"]["label"] == "written by Claude for exp_036"
    t4 = m["texts"]["T4"]["origin"]
    assert t4["origin"] == "https://www.gesetze-im-internet.de/gg/xml.zip"
    assert t4["origin_sha256"].startswith("b0dbbb71")
    assert t4["articles"][0] == "Art 1" and t4["articles"][-1] == "Art 19"
    for tid, name in (("T1", "2026-09-22-we-trained-it-three-times-then-stopped.md"),
                      ("T2", "2026-09-30-the-part-that-looked-fine.md")):
        assert m["texts"][tid]["origin"]["origin"].endswith(name)
        assert len(m["texts"][tid]["origin"]["origin_sha256"]) == 64
    # The sources the texts were cut from are committed and hash as recorded.
    for tid in ("T1", "T2", "T3", "T4"):
        src = TEXTS / "src" / f"{tid}_source.txt"
        assert hashlib.sha256(src.read_bytes()).hexdigest() == m["texts"][tid]["source_sha256"]
    # T5, T6, T9: rule, rows, counts and hashes only.
    for tid in ("T5", "T6", "T9"):
        entry = m["web"][tid]
        assert "text" not in entry and "ids" not in entry
        assert len(entry["ids_sha256"]) == 64
    assert m["web"]["T5"]["row"] >= 5000 and m["web"]["T6"]["row"] > m["web"]["T5"]["row"]
    assert m["web"]["T9"]["n_tokens"] == 16384
    assert [c["what"] for c in m["web"]["T9"]["components"][:3]] == ["T1 source in full", "T4 source in full", "T3 source in full"]


def test_committed_ids_and_text_agree():
    m = _manifest()
    for tid in ("T1", "T2", "T3", "T4"):
        stem = m["texts"][tid]["files"][0][: -len(".ids.json")]
        rec = json.loads((TEXTS / f"{stem}.ids.json").read_text(encoding="utf-8"))
        text = (TEXTS / f"{stem}.txt").read_text(encoding="utf-8")
        assert len(rec["ids"]) == 1536
        assert rec["ids_sha256"] == _ids_sha(rec["ids"]) == m["texts"][tid]["ids_sha256"]
        assert rec["text_sha256"] == hashlib.sha256(text.encode()).hexdigest()
        assert "�" not in text
        # The cut is a prefix of the source (after any one-word advances).
        src = (TEXTS / "src" / f"{tid}_source.txt").read_text(encoding="utf-8")
        assert text in src[: len(text) + 2000]


def test_exactly_1536_kolibri_tokens(raw_tok):
    m = _manifest()
    for tid in ("T1", "T2", "T3", "T4"):
        stem = m["texts"][tid]["files"][0][: -len(".ids.json")]
        rec = json.loads((TEXTS / f"{stem}.ids.json").read_text(encoding="utf-8"))
        text = (TEXTS / f"{stem}.txt").read_text(encoding="utf-8")
        assert raw_tok.decode(rec["ids"], skip_special_tokens=False) == text
        src = (TEXTS / "src" / f"{tid}_source.txt").read_text(encoding="utf-8")
        start = src.index(text)
        assert raw_tok.encode(src[start:], add_special_tokens=False).ids[:1536] == rec["ids"]
    t3_src = (TEXTS / "src" / "T3_source.txt").read_text(encoding="utf-8")
    assert len(raw_tok.encode(t3_src, add_special_tokens=False).ids) >= 1600  # HYPOTHESIS: >= 1,600 before the cut


def test_chat_texts_render_and_tokenise(raw_tok):
    from gate import vendor_template

    for name, effort in (("T7_chat_en_high.json", "high"), ("T8_chat_de_none.json", "none")):
        rec = json.loads((TEXTS / name).read_text(encoding="utf-8"))
        assert rec["effort"] == effort and len(rec["ids"]) == 1536
        text = vendor_template.render(rec["messages"], add_generation_prompt=False, reasoning_effort=effort)
        assert text == rec["text"]
        assert raw_tok.encode(text, add_special_tokens=False).ids == rec["ids"]
        assert rec["ids"].count(127907) == 1 and rec["ids"].count(127908) == 1  # one closed think block
        assert rec["text"].endswith("<|im_end|>\n") and rec["ids"][-2] == 127906  # the closed assistant turn
    t7 = json.loads((TEXTS / "T7_chat_en_high.json").read_text(encoding="utf-8"))
    assert t7["messages"][-1]["reasoning"] and "<think>\n" + t7["messages"][-1]["reasoning"] in t7["text"]
    t8 = json.loads((TEXTS / "T8_chat_de_none.json").read_text(encoding="utf-8"))
    assert "<|im_start|>assistant\n<think>\n\n</think>\n\n" in t8["text"]


def test_g5_prompts(raw_tok):
    from gate import vendor_template

    prompts = json.loads((TEXTS / "G5_prompts.json").read_text(encoding="utf-8"))["prompts"]
    assert len(prompts) == 8
    assert sorted(p["lang"] for p in prompts) == ["de"] * 4 + ["en"] * 4
    for lang in ("en", "de"):
        assert sorted(p["effort"] for p in prompts if p["lang"] == lang) == ["high", "high", "none", "none"]
    for p in prompts:
        assert len(p["ids"]) >= 600
        text = vendor_template.render(p["messages"], add_generation_prompt=True, reasoning_effort=p["effort"])
        assert text == p["text"]
        assert raw_tok.encode(text, add_special_tokens=False).ids == p["ids"]


def test_committed_texts_reproduce(tokenizer_dir):
    """Rebuilding from the committed sources gives the committed files byte
    for byte (so MANIFEST.json's sha256 is reproducible)."""
    pytest.importorskip("jinja2")
    from gate import build_gate_text

    manifest, files, _ = build_gate_text.build(tokenizer_dir, None, _manifest())
    for name, content in files.items():
        assert (TEXTS / name).read_text(encoding="utf-8") == content, name
    assert set(files) == {p.name for p in TEXTS.iterdir() if p.is_file()}


def _parquet() -> Path:
    from gate import build_gate_text, common

    path = common.data_dir() / build_gate_text.FINEWEB_REL
    if not path.is_file():
        pytest.skip(f"FineWeb-2 parquet not found under $EXP036_DATA ({path.name}); set EXP036_DATA")
    return path


def test_web_texts_match_manifest(raw_tok):
    from gate import build_gate_text

    public, work = build_gate_text.build_web(raw_tok, _parquet())
    assert build_gate_text.compare_web(_manifest(), public) == []
    assert len(work["T5"]["ids"]) == len(work["T6"]["ids"]) == 1536 and len(work["T9"]["ids"]) == 16384


def test_no_web_text_in_the_repo_tree(raw_tok):
    """Windows of T5/T6 and of T9's web part appear in no file of the experiment directory."""
    from gate import build_gate_text

    _, work = build_gate_text.build_web(raw_tok, _parquet())
    needles = []
    for tid in ("T5", "T6"):
        t = work[tid]["text"]
        needles += [t[i:i + 60] for i in (200, len(t) // 2, len(t) - 300)]
    t9_tail = raw_tok.decode(work["T9"]["ids"][-600:], skip_special_tokens=False)
    needles.append(t9_tail[100:160])
    needles = [n for n in needles if len(n.strip()) >= 40]
    for p in EXP.rglob("*"):
        if not p.is_file() or p.stat().st_size > 5_000_000 or "__pycache__" in p.parts or ".pytest_cache" in p.parts:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n in needles:
            assert n not in text, f"FineWeb text found in {p.relative_to(EXP)}"


def test_no_gpqa_canary():
    """The benchmark canary (tools/leak_check.py CANARY patterns) is in no gate file."""
    from exp036_helpers import import_sibling

    leak = import_sibling("tools.leak_check")
    for p in (EXP / "gate").rglob("*"):
        if p.is_file() and p.suffix in (".json", ".txt", ".py", ".md", ".jinja"):
            text = p.read_text(encoding="utf-8")
            assert not any(rx.search(text) for rx in leak.CANARY), p


def test_tokenizer_lines_rule():
    from gate import tokenizer_lines

    groups = tokenizer_lines.committed_rule_lines()
    assert sum(len(v) for v in groups.values()) >= 2000  # thresholds G0.tokenizer_lines_min
    assert len(groups["digits"]) == 600 and groups["digits"] == tokenizer_lines.digit_lines()  # seeded
    assert all(line.strip() for line in groups["sources"])


def test_work_only_build_and_real_textset(raw_tok, tokenizer_dir, tmp_path, monkeypatch):
    """RUNBOOK step 7: --work-only writes T5, T6, T9 to $EXP036_WORK/gate_texts
    and checks them against MANIFEST.json; the gate's real text set then loads."""
    from gate import build_gate_text, textset

    parquet = _parquet()
    monkeypatch.setenv("EXP036_WORK", str(tmp_path / "work"))
    assert build_gate_text.main(["--work-only", "--fineweb", str(parquet), "--tok", str(tokenizer_dir)]) == 0
    work = tmp_path / "work" / "gate_texts"
    assert {p.name for p in work.iterdir()} == {"T5.ids.json", "T5.txt", "T6.ids.json", "T6.txt", "T9.ids.json"}
    ts = textset.load_real(tokenizer_dir, work)
    assert all(len(ts.ids[t]) == 1536 for t in textset.ALL8) and len(ts.t9) == 16384 and len(ts.g5) == 8
    for tid in ("T1", "T3", "T5"):  # bytes per token add up to the decoded text's UTF-8 length
        text = raw_tok.decode(ts.ids[tid], skip_special_tokens=False)
        assert sum(ts.token_bytes[tid]) == len(text.encode("utf-8"))
    # A work file that disagrees with MANIFEST.json is refused.
    rec = json.loads((work / "T5.ids.json").read_text())
    rec["ids"] = rec["ids"][::-1]
    (work / "T5.ids.json").write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="T5"):
        textset.load_real(tokenizer_dir, work)
