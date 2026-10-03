"""scorers/score_all.py: raw JSONL -> score JSONL round trip on a synthetic results tree (BUILD_SPEC §5.6).

Synthetic completions only (invented facts, stub prompts). Checks: one score per item, the rules per task,
truncation and parse statuses, byte-identical output on a rerun, refusal to rewrite a different file, the
private mirror for withheld sets (hash-checked), no free text in withheld score records, IFBench skipped
without --ifbench, and the mini re-score comparison.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path

import pytest

from scorers import score_all as S

HEADER = {"type": "header", "host": "synthetic", "seed": 1}


def _need_shim():
    try:
        importlib.import_module("tasks.vendored_evalfw.shim")
    except ModuleNotFoundError as e:
        if (e.name or "").startswith("tasks"):
            pytest.skip(f"tasks/vendored_evalfw/shim.py is not there yet ({e.name})")
        raise


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _rec(arm, task, item, text, *, finish="stop", withheld=False, effort="high", item_sha=None):
    r = {
        "type": "record",
        "key": {"arm": arm, "task": task, "effort": effort, "item": item, "pass": 0},
        "item_sha256": item_sha or _sha(f"{task}|{item}"),
        "prompt_sha256": _sha(f"prompt|{task}|{item}"),
        "finish_reason": finish,
        "truncated": finish == "length",
    }
    if withheld:
        r["text_sha256"] = _sha(text)
    else:
        r["text"] = text
    return r


def _write(path: Path, objs, torn: str | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
        if torn:
            f.write(torn)


K_THINK = "<think>\nSynthetic reasoning.\n</think>\n\n"
G_THINK = "<|channel>thought\nSynthetic reasoning.<channel|>"


def _tree(tmp_path: Path) -> dict:
    res, priv, man = tmp_path / "results", tmp_path / "private", tmp_path / "manifests"
    raw = res / "raw"
    # MMLU EN (public), Kolibri effort high: model opens <think> itself.
    mmlu = [
        ("1", K_THINK + "Therefore, the answer is (B)<|im_end|>", "stop"),
        ("2", K_THINK + "Therefore, the answer is (C)<|im_end|>", "stop"),
        ("3", K_THINK + "I cannot decide.<|im_end|>", "stop"),
        ("4", "<think>\nStill thinking when the cap hit", "length"),
        ("5", "<think>\nStopped inside the block<|im_end|>", "stop"),
        ("6", "Therefore, the answer is (A)<|im_end|>", "stop"),  # no think block at effort high: a defect
    ]
    recs = [_rec("K8", "mmlu_en", i, t, finish=f) for i, t, f in mmlu]
    recs.append(_rec("K8", "mmlu_en", "1", K_THINK + "Therefore, the answer is (D)<|im_end|>"))  # duplicate key
    _write(raw / "S2" / "K8" / "mmlu_en_high.jsonl", [HEADER] + recs, torn='{"type": "record", "key": {"ar')
    _write(man / "mmlu_en.jsonl", [{"id": i, "item_sha256": _sha(f"mmlu_en|{i}"), "gold": g, "category": c}
                                   for i, g, c in [("1", "B", "law"), ("2", "A", "law"), ("3", "C", "math"),
                                                   ("4", "D", "math"), ("5", "E", "math"), ("6", "A", "law")]])
    # GPQA DE (withheld), Kolibri.
    g_texts = {"1": "<think>\nWir rechnen das nach und es ist nicht B.\n</think>\n\nDaher ist die Antwort (C)<|im_end|>",
               "2": "<think>\nWe check it and it is not B.\n</think>\n\nDaher ist die Antwort (A)<|im_end|>"}
    _write(raw / "S2" / "K8" / "gpqa_de_high.jsonl",
           [HEADER] + [_rec("K8", "gpqa_de", i, t, withheld=True) for i, t in g_texts.items()])
    _write(priv / "raw" / "S2" / "K8" / "gpqa_de_high.jsonl",
           [HEADER] + [_rec("K8", "gpqa_de", i, t) for i, t in g_texts.items()])
    _write(priv / "manifests" / "gpqa_de.jsonl", [{"id": "1", "item_sha256": _sha("gpqa_de|1"), "gold": "C"},
                                                 {"id": "2", "item_sha256": _sha("gpqa_de|2"), "gold": "B"}])
    # RGB closed-book (withheld), Gemma 4: one answer opens the channel, one does not ("none" is fine).
    cb = {"1": G_THINK + "It is Lunaport.<turn|>", "2": "I am not sure.<turn|>", "3": "Sandmere.<turn|>",
          "4": G_THINK + "Lunaport, but insufficient information.<turn|>", "5": "<|channel>thought\nlong"}
    fin = {"5": "length"}
    _write(raw / "S2" / "G8" / "rgb_cb_high.jsonl",
           [HEADER] + [_rec("G8", "rgb_cb", i, t, withheld=True, finish=fin.get(i, "stop")) for i, t in cb.items()])
    _write(priv / "raw" / "S2" / "G8" / "rgb_cb_high.jsonl",
           [HEADER] + [_rec("G8", "rgb_cb", i, t, finish=fin.get(i, "stop")) for i, t in cb.items()])
    _write(priv / "manifests" / "rgb_cb.jsonl",
           [{"id": i, "item_sha256": _sha(f"rgb_cb|{i}"), "gold": [["Lunaport", "Luna Port"]],
             "subset": "en" if i in "123" else "en_fact"} for i in cb])
    # RGB negative and fact-check (withheld), Qwen3.6: the prompt prefills <think>, so the text starts inside it.
    neg = {"1": "Thinking.</think>\n\nThere is insufficient information.<|im_end|>",
           "2": "Thinking.</think>\n\nLunaport.<|im_end|>"}
    _write(raw / "S3" / "Q36-8" / "rgb_neg_high.jsonl",
           [HEADER] + [_rec("Q36-8", "rgb_neg", i, t, withheld=True) for i, t in neg.items()])
    _write(priv / "raw" / "S3" / "Q36-8" / "rgb_neg_high.jsonl",
           [HEADER] + [_rec("Q36-8", "rgb_neg", i, t) for i, t in neg.items()])
    _write(priv / "manifests" / "rgb_neg.jsonl", [{"id": i, "item_sha256": _sha(f"rgb_neg|{i}")} for i in neg])
    fact = {"1": "T.</think>The documents contain factual errors; it is Lunaport.<|im_end|>",
            "2": "T.</think>It is Sandmere.<|im_end|>"}
    _write(raw / "S3" / "Q36-8" / "rgb_fact_high.jsonl",
           [HEADER] + [_rec("Q36-8", "rgb_fact", i, t, withheld=True) for i, t in fact.items()])
    _write(priv / "raw" / "S3" / "Q36-8" / "rgb_fact_high.jsonl",
           [HEADER] + [_rec("Q36-8", "rgb_fact", i, t) for i, t in fact.items()])
    _write(priv / "manifests" / "rgb_fact.jsonl",
           [{"id": i, "item_sha256": _sha(f"rgb_fact|{i}"), "gold": "Lunaport", "gold_fake": "Sandmere"} for i in fact])
    # AIME EN (public), Kolibri.
    _write(raw / "S3" / "K8" / "aime_en_high.jsonl",
           [HEADER, _rec("K8", "aime_en", "1", K_THINK + "Thus $\\boxed{042}$.<|im_end|>"),
            _rec("K8", "aime_en", "2", K_THINK + "\\boxed{\\frac{1}{2}}<|im_end|>")])
    _write(man / "aime_en.jsonl", [{"id": "1", "item_sha256": _sha("aime_en|1"), "gold": 42},
                                   {"id": "2", "item_sha256": _sha("aime_en|2"), "gold": 7}])
    # IFBench (public), Kolibri: scored only with ifbench=True.
    _write(raw / "S3" / "K8" / "ifbench_high.jsonl",
           [HEADER, _rec("K8", "ifbench", "17", K_THINK + "I have 2 apples and 3 pears.<|im_end|>"),
            _rec("K8", "ifbench", "290", K_THINK + "The Quick Brown Fox<|im_end|>", finish="length")])
    _write(man / "ifbench.jsonl", [{"id": "17", "item_sha256": _sha("ifbench|17")},
                                   {"id": "290", "item_sha256": _sha("ifbench|290")}])
    return {"results": res, "private": priv, "manifests": man}


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _by_item(path: Path) -> dict[str, dict]:
    return {s["key"]["item"]: s for s in _load(path)}


def test_round_trip_rules_and_rerun(tmp_path):
    _need_shim()
    t = _tree(tmp_path)
    rep = S.score_all(t["results"], t["private"], manifests_dir=t["manifests"], require_guards=False)
    assert rep["K8/ifbench_high.jsonl"]["status"].startswith("skipped")
    assert rep["K8/mmlu_en_high.jsonl"]["torn_lines"] == 1
    scores = t["results"] / "scores"

    m = _by_item(scores / "K8" / "mmlu_en_high.jsonl")
    assert list(m) == ["1", "2", "3", "4", "5", "6"]
    assert m["1"]["extracted"] == "B" and m["1"]["correct"]  # first record for a key counts, not the duplicate
    assert m["2"]["extracted"] == "C" and not m["2"]["correct"] and m["2"]["category"] == "law"
    assert m["3"]["parse_status"] == "no_match" and not m["3"]["correct"]
    assert m["4"]["truncated"] and m["4"]["parse_status"] == "truncated" and m["4"]["reasoning_status"] == "unclosed"
    assert m["5"]["parse_status"] == "unclosed_reasoning" and not m["5"]["truncated"]
    assert m["6"]["reasoning_status"] == "none" and m["6"]["correct"]  # scored, flagged as a defect by parse_record
    assert set(m["1"]) == {"key", "item_sha256", "extracted", "category", "correct", "truncated", "parse_status",
                           "reasoning_status", "reasoning_lang"}
    assert m["1"]["reasoning_lang"] == "unknown" and m["6"]["reasoning_lang"] == "unknown"

    g = _by_item(scores / "K8" / "gpqa_de_high.jsonl")
    assert (g["1"]["extracted"], g["1"]["correct"], g["2"]["extracted"], g["2"]["correct"]) == ("C", True, "A", False)
    assert g["1"]["reasoning_lang"] == "de" and g["2"]["reasoning_lang"] == "en"

    cb = _by_item(scores / "G8" / "rgb_cb_high.jsonl")
    assert {k: v["category"] for k, v in cb.items()} == {
        "1": "correct", "2": "abstain", "3": "wrong", "4": "abstain", "5": "truncated"}
    assert [cb[k]["subset"] for k in "12345"] == ["en", "en", "en", "en_fact", "en_fact"]
    assert cb["2"]["reasoning_status"] == "none" and cb["1"]["reasoning_status"] == "closed"
    assert [cb[k]["correct"] for k in "12345"] == [True, False, False, False, False]

    neg = _by_item(scores / "Q36-8" / "rgb_neg_high.jsonl")
    assert (neg["1"]["category"], neg["1"]["correct"], neg["2"]["category"]) == ("rejected", True, "not")
    fact = _by_item(scores / "Q36-8" / "rgb_fact_high.jsonl")
    assert (fact["1"]["category"], fact["2"]["category"]) == ("corrected", "deferred")

    a = _by_item(scores / "K8" / "aime_en_high.jsonl")
    assert (a["1"]["extracted"], a["1"]["correct"]) == (42, True)
    assert (a["2"]["extracted"], a["2"]["parse_status"]) == (None, "non_integer")

    before = {p: p.read_bytes() for p in scores.rglob("*.jsonl")}
    rep2 = S.score_all(t["results"], t["private"], manifests_dir=t["manifests"], require_guards=False)
    assert all(v.get("file") == "unchanged" for v in rep2.values() if v["status"] == "scored")
    assert {p: p.read_bytes() for p in scores.rglob("*.jsonl")} == before
    other = tmp_path / "again"
    S.score_all(t["results"], t["private"], manifests_dir=t["manifests"], out_dir=other, require_guards=False)
    assert S.compare_trees(scores, other / "scores") == {p.relative_to(scores).as_posix(): "identical" for p in before}


def test_withheld_scores_carry_no_free_text(tmp_path):
    _need_shim()
    t = _tree(tmp_path)
    S.score_all(t["results"], t["private"], manifests_dir=t["manifests"], require_guards=False)
    for arm, name in (("K8", "gpqa_de_high"), ("G8", "rgb_cb_high"), ("Q36-8", "rgb_neg_high"),
                      ("Q36-8", "rgb_fact_high")):
        raw = (t["results"] / "scores" / arm / f"{name}.jsonl").read_text(encoding="utf-8")
        for needle in ("Lunaport", "Sandmere", "Daher", "insufficient", "Synthetic reasoning", "<"):
            assert needle not in raw, (name, needle)
        for s in _load(t["results"] / "scores" / arm / f"{name}.jsonl"):
            assert s["extracted"] is None or (isinstance(s["extracted"], str) and len(s["extracted"]) == 1)


def test_existing_different_scores_are_never_rewritten(tmp_path):
    _need_shim()
    t = _tree(tmp_path)
    S.score_all(t["results"], t["private"], manifests_dir=t["manifests"], require_guards=False)
    target = t["results"] / "scores" / "K8" / "aime_en_high.jsonl"
    target.write_text(target.read_text().replace('"correct": true', '"correct": false'))
    with pytest.raises(RuntimeError, match="amendment"):
        S.score_all(t["results"], t["private"], manifests_dir=t["manifests"], require_guards=False)


def test_private_mirror_is_required_and_hash_checked(tmp_path):
    _need_shim()
    t = _tree(tmp_path)
    with pytest.raises(FileNotFoundError):
        S.score_all(t["results"], None, manifests_dir=t["manifests"], require_guards=False)
    p = t["private"] / "raw" / "S2" / "K8" / "gpqa_de_high.jsonl"
    p.write_text(p.read_text(encoding="utf-8").replace("(C)", "(D)"), encoding="utf-8")
    with pytest.raises(ValueError, match="text_sha256"):
        S.score_all(t["results"], t["private"], manifests_dir=t["manifests"], require_guards=False)


def test_item_hash_mismatch_is_refused(tmp_path):
    _need_shim()
    t = _tree(tmp_path)
    m = t["manifests"] / "aime_en.jsonl"
    m.write_text(m.read_text().replace(_sha("aime_en|1"), _sha("other")))
    with pytest.raises(ValueError, match="item_sha256"):
        S.score_all(t["results"], t["private"], manifests_dir=t["manifests"], require_guards=False)


def test_parse_record_for_the_pilot():
    _need_shim()
    rec = _rec("K8", "gpqa_en", "1", "Therefore, the answer is (A)<|im_end|>")
    assert S.parse_record("gpqa_en", rec, rec["text"]) == {
        "reasoning_status": "none", "truncated": False, "parse_status": "ok", "defect": True}
    rec = _rec("G8", "gpqa_en", "1", "The answer is (A)<turn|>")
    assert S.parse_record("gpqa_en", rec, rec["text"])["defect"] is False  # Gemma 4 "none" is legitimate
    rec = _rec("K8", "aime_en", "1", K_THINK + "no box<|im_end|>")
    assert S.parse_record("aime_en", rec, rec["text"])["parse_status"] == "no_match"
    rec = _rec("K8", "rgb_cb", "1", K_THINK + "   <|im_end|>")
    assert S.parse_record("rgb_cb", rec, rec["text"])["parse_status"] == "empty_answer"
    rec = _rec("K8", "mmlu_de", "1", "<think>x", finish="length")
    assert S.parse_record("mmlu_de", rec, rec["text"])["parse_status"] == "truncated"


def test_pass_two_files(tmp_path):
    _need_shim()
    t = _tree(tmp_path)
    recs = [_rec("K8", "aime_en", "1", K_THINK + "\\boxed{42}<|im_end|>")]
    recs[0]["key"]["pass"] = 1
    _write(t["results"] / "raw" / "S3" / "K8" / "aime_en_high_p1.jsonl", [HEADER] + recs)
    rep = S.score_all(t["results"], t["private"], manifests_dir=t["manifests"], require_guards=False)
    assert rep["K8/aime_en_high_p1.jsonl"]["records"] == 1
    s = _load(t["results"] / "scores" / "K8" / "aime_en_high_p1.jsonl")
    assert s[0]["key"]["pass"] == 1 and s[0]["correct"]
    assert S.parse_stem("gpqa_en_high_p1") == ("gpqa_en", "high", 1)
    assert S.parse_stem("rgb_cb_none") == ("rgb_cb", "none", 0)


def test_task_names():
    assert S.canonical_task("gpqa_diamond_de") == "gpqa_de"
    assert S.canonical_task("rgb_negative") == "rgb_neg"
    with pytest.raises(ValueError):
        S.canonical_task("hellaswag")


def _ifbench_env():
    from test_scorers_ifbench import _data_dirs  # same discovery as the adapter test

    for d in _data_dirs():
        src, py, nltk = d / "IFBench-src", d.parent / "ifbench-venv" / "bin" / "python", d.parent / "nltk_data"
        py = Path(os.environ.get("EXP036_IFBENCH_PY", py)).expanduser()
        nltk = Path(os.environ.get("EXP036_NLTK_DATA", nltk)).expanduser()
        if (src / "evaluation_lib.py").is_file() and py.exists() and nltk.is_dir():
            return {"ifbench_src": src, "ifbench_test": src / "ifbench" / "data" / "IFBench_test.jsonl",
                    "python": py, "nltk_data": nltk}
    pytest.skip("IFBench checkout / venv / NLTK data not found")


def test_ifbench_cells_with_the_adapter_and_rescore(tmp_path):
    _need_shim()
    kw = _ifbench_env()
    t = _tree(tmp_path)
    S.score_all(t["results"], t["private"], manifests_dir=t["manifests"], ifbench=True, ifbench_kwargs=kw,
                require_guards=False)
    s = _by_item(t["results"] / "scores" / "K8" / "ifbench_high.jsonl")
    assert s["17"]["correct"] and s["17"]["extracted"] == {
        "instruction_id_list": ["count:numbers"], "loose": [True], "strict": [True]}
    assert s["17"]["correct_loose"] and s["17"]["correct_strict"]
    assert s["290"]["extracted"]["loose"] == [True] and not s["290"]["correct"] and s["290"]["truncated"]
    assert not s["290"]["correct_loose"] and not s["290"]["correct_strict"]

    # The mini re-score (with IFBench) reproduces every file byte for byte, in verdicts.py's shape.
    rescore = S.rescore_compare(t["results"], t["private"], manifests_dir=t["manifests"], ifbench=True,
                                ifbench_kwargs=kw, require_guards=False)
    payload = json.loads(rescore.read_text())
    assert payload["all_identical"] is True
    files = {e["path"]: e for e in payload["files"]}
    ifb = files["scores/K8/ifbench_high.jsonl"]
    assert ifb["mini_only"] and ifb["sha256_mbp"] is None and ifb["match"]
    other = files["scores/K8/aime_en_high.jsonl"]
    assert not other["mini_only"] and other["sha256_mbp"] == other["sha256_mini"]


def test_rescore_compare_identical_without_ifbench(tmp_path):
    _need_shim()
    t = _tree(tmp_path)
    S.score_all(t["results"], t["private"], manifests_dir=t["manifests"], require_guards=False)
    path = S.rescore_compare(t["results"], t["private"], manifests_dir=t["manifests"], require_guards=False)
    payload = json.loads(path.read_text())
    assert payload["all_identical"] is True and len(payload["files"]) == 6
    assert all(e["match"] and e["sha256_mbp"] == e["sha256_mini"] for e in payload["files"])


def test_rescore_compare_flags_differences(tmp_path):
    _need_shim()
    t = _tree(tmp_path)
    S.score_all(t["results"], t["private"], manifests_dir=t["manifests"], require_guards=False)
    target = t["results"] / "scores" / "K8" / "aime_en_high.jsonl"
    target.write_text(target.read_text().replace('"correct": true', '"correct": false'))
    # A committed IFBench file that this re-score (no --ifbench) does not reproduce.
    (t["results"] / "scores" / "K8" / "ifbench_high.jsonl").write_text("{}\n")
    path = S.rescore_compare(t["results"], t["private"], manifests_dir=t["manifests"], require_guards=False)
    payload = json.loads(path.read_text())
    files = {e["path"]: e for e in payload["files"]}
    assert payload["all_identical"] is False
    assert not files["scores/K8/aime_en_high.jsonl"]["match"]
    assert not files["scores/K8/ifbench_high.jsonl"]["match"] and files["scores/K8/ifbench_high.jsonl"]["mini_only"]
    assert files["scores/K8/gpqa_de_high.jsonl"]["match"]


def test_prompt_prefill_from_the_record_is_used():
    _need_shim()
    # A Qwen record whose render prefilled an open <think> (open_think): the text starts inside the block.
    rec = _rec("Q36-8", "mmlu_en", "1", "Reasoning.</think>Therefore, the answer is (B)<|im_end|>")
    rec["prompt_prefill"] = "open_think"
    assert S.split_record("mmlu_en", rec, rec["text"])[:2] == ("Therefore, the answer is (B)", "closed")
    rec["prompt_prefill"] = "empty_think"  # thinking off: the whole completion is the answer
    assert S.split_record("mmlu_en", rec, rec["text"])[1] == "none"
    rec["prompt_prefill"] = "bogus"
    with pytest.raises(ValueError):
        S.split_record("mmlu_en", rec, rec["text"])


def test_rescore_compare_implies_ifbench_when_ifbench_scores_exist(tmp_path, monkeypatch):
    """Review fix 2026-10-03: RUNBOOK step 19's --rescore-compare re-scores the committed IFBench files too."""
    from scorers import score_all as sa

    seen = {}

    def fake(results, private, manifests_dir=None, ifbench=False, **kw):
        seen["ifbench"] = ifbench
        p = tmp_path / "rescore_mini_x.json"
        p.write_text('{"all_identical": true, "files": []}')
        return p

    monkeypatch.setattr(sa, "rescore_compare", fake)
    (tmp_path / "scores" / "K8").mkdir(parents=True)
    assert sa.main(["--rescore-compare", "--results", str(tmp_path)]) == 0
    assert seen["ifbench"] is False
    (tmp_path / "scores" / "K8" / "ifbench_high.jsonl").write_text("{}\n")
    assert sa.main(["--rescore-compare", "--results", str(tmp_path)]) == 0
    assert seen["ifbench"] is True
