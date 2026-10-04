"""tasks/build_manifests.py and selection_rules.json on a complete synthetic data tree (BUILD_SPEC §5.5)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import tasks_synthetic as syn
from tasks import build_manifests as bm
from tasks.common import sha256_hex

EXP = Path(__file__).resolve().parent.parent
TEXT_KEYS = {"text", "messages", "question", "prompt", "problem", "query", "answer", "options", "content"}
# Amendment 4: the full pool leaves out the full split's gold-mismatch row (planted in the synthetic world, as on the
# real data) and every build records it; together with Amendment 3's over-long count these are the build's checks.
GOLD = {"en": [str(syn.MMLU_GOLD_MISMATCH_ID)], "de": [str(syn.MMLU_GOLD_MISMATCH_ID)]}
CHECKS = {"gpqa_diamond_en_overlong_excluded": 0, "mmlu_prox_full_gold_inconsistent_excluded": GOLD}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic_data")
    syn.write_all(root)
    out = tmp_path_factory.mktemp("manifests")
    priv = tmp_path_factory.mktemp("private")
    res = tmp_path_factory.mktemp("results")
    sh = tmp_path_factory.mktemp("tools") / "withheld_shingles.sha256"
    summary = bm.build_all(out, priv, root, None, res, sh, enforce_counts=False, check_revisions=False)
    return {"root": root, "out": out, "priv": priv, "res": res, "sh": sh, "summary": summary}


def _docs(out: Path):
    return {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(out.glob("*.json"))}


def test_every_set_built(built):
    docs = _docs(built["out"])
    names = {sd.name for sd in bm.set_defs()}
    assert names | {"mmlu_prox_category_counts"} == set(docs)
    assert built["summary"]["counts"]["rgb_cb"] == 14 and built["summary"]["counts"]["gpqa_diamond_en"] == 6


def test_withheld_manifests_carry_no_text_or_gold(built):
    for name, doc in _docs(built["out"]).items():
        if name == "mmlu_prox_category_counts":
            continue
        for e in doc["items"]:
            assert not (set(e) & TEXT_KEYS)
            assert set(e) <= (bm.ALLOWED_WITHHELD if doc["withheld"] else bm.ALLOWED_PUBLIC)
            assert all(not isinstance(v, str) or len(v) <= bm.MAX_PUBLIC_STRING for v in e.values())
            if doc["withheld"]:
                assert "gold" not in e and "category" not in e
    raw = b"".join(p.read_bytes() for p in built["out"].glob("*.json"))
    for marker in (b"Synthetic question", b"Synthetische", b"Synthetic RGB query", b"STUB", b"NEG0-", b"Bauteilwert"):
        assert marker not in raw


def test_public_sets_carry_gold_and_category(built):
    docs = _docs(built["out"])
    lite = docs["mmlu_prox_lite_en"]["items"]
    assert all({"category", "gold", "cat_rank", "n_options"} <= set(e) for e in lite)
    # Amendment 4: all of Lite (the real composition here), listed in the Webster seat order, so the n_M set is a
    # prefix; each category's items come in cat_rank order 0, 1, 2, ... (a prefix of its seed-36 order)
    assert len(lite) == 588 and [e["category"] for e in lite] == bm.mmlu_prox.webster_order(syn.MMLU_LITE_COUNTS)
    for c in syn.MMLU_CATS:
        assert [e["cat_rank"] for e in lite if e["category"] == c] == list(range(syn.MMLU_LITE_COUNTS[c]))
    alloc = docs["mmlu_prox_lite_en"]["nM_allocation"]
    assert set(alloc) == {str(n) for n in bm.rules()["mmlu_prox"]["nM_ladder"]}
    for n, per in alloc.items():
        assert per == bm.mmlu_prox.allocation(syn.MMLU_LITE_COUNTS, int(n))
        assert sum(per.values()) == int(n) and set(per) == set(syn.MMLU_CATS) and min(per.values()) >= 5
    # EN and DE list the same ids, in the same order, with the same categories (parallel)
    lite_de = docs["mmlu_prox_lite_de"]
    assert [(e["id"], e["category"]) for e in lite_de["items"]] == [(e["id"], e["category"]) for e in lite]
    assert lite_de["nM_allocation"] == alloc
    # the full-pool sets list the left-out gold-mismatch ids and never hold them
    for name in bm.MMLU_FULL_SETS:
        assert docs[name]["gold_inconsistent_excluded"] == GOLD
        assert str(syn.MMLU_GOLD_MISMATCH_ID) not in {e["id"] for e in docs[name]["items"]}
    assert all("gold_inconsistent_excluded" not in d for n, d in docs.items() if n not in bm.MMLU_FULL_SETS)
    assert all(isinstance(e["gold"], int) for e in docs["aime_en"]["items"])
    assert all(set(e) == set(bm.BASE_FIELDS) for e in docs["ifbench"]["items"])
    assert {e["id"] for e in docs["gpqa_diamond_en"]["items"]} >= {"recSYN0000"}
    # Amendment 3: every Diamond item is primary, and the over-long count (0) is recorded in the manifest
    assert all(e["in_primary"] is True for e in docs["gpqa_diamond_en"]["items"])
    g = docs["gpqa_diamond_en"]
    assert g["primary_n"] == g["n"] == 6 and g["overlong_excluded"] == 0
    assert built["summary"]["checks"] == CHECKS
    assert all("overlong_excluded" not in d for n, d in docs.items() if n != "gpqa_diamond_en")
    assert all(len(e["doc_indices"]) >= 1 for e in docs["rgb_negative"]["items"])
    assert docs["rgb_negative"]["doc_source"] == "negative" and docs["rgb_fact"]["doc_source"] == "positive_wrong"
    counts = docs["mmlu_prox_category_counts"]
    # every full-split row counts (Lite's 588, 11 more per category, and the gold-mismatch row: Amendment 4)
    assert set(counts) == {"en", "de", "_meta"} and counts["_meta"]["totals"]["en"] == 588 + 14 * 11 + 1
    assert counts["en"] == {c: syn.MMLU_LITE_COUNTS[c] + 11 + (c == syn.MMLU_GOLD_MISMATCH_CATEGORY)
                            for c in syn.MMLU_CATS} == counts["de"]
    assert counts["_meta"]["gold_inconsistent_counted"] == GOLD


def test_private_manifests_hold_text_and_gold(built):
    p = built["priv"] / "manifests" / "gpqa_diamond_en.jsonl"
    lines = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert lines[0]["type"] == "header" and lines[0]["withheld"] is True
    assert lines[0]["public_manifest_sha256"] == sha256_hex((built["out"] / "gpqa_diamond_en.json").read_bytes())
    assert all(l["gold"] in "ABCD" and l["messages"][0]["role"] == "user" for l in lines[1:])
    assert built["summary"]["private"]["manifests/gpqa_diamond_en.jsonl"] == sha256_hex(p.read_bytes())
    # private manifests exist for the withheld sets only
    written = {q.stem for q in (built["priv"] / "manifests").glob("*.jsonl")}
    assert written == {sd.name for sd in bm.set_defs() if sd.withheld}
    fact = [json.loads(l) for l in (built["priv"] / "manifests" / "rgb_fact.jsonl").read_text(encoding="utf-8").splitlines()][1:]
    assert all(isinstance(l["gold"], str) and l["gold_fake"].startswith("widget") for l in fact)
    cb = [json.loads(l) for l in (built["priv"] / "manifests" / "rgb_cb.jsonl").read_text(encoding="utf-8").splitlines()][1:]
    assert cb[0]["gold"] == [["gadget 0", "Gadget-0"]] and "gold_fake" not in cb[0]


def test_rebuild_is_byte_identical_and_refuses_changes(built, tmp_path):
    out2, priv2 = tmp_path / "m2", tmp_path / "p2"
    s2 = bm.build_all(out2, priv2, built["root"], None, None, None, enforce_counts=False, check_revisions=False)
    assert s2["manifests"] == built["summary"]["manifests"] and s2["private"] == built["summary"]["private"]
    s3 = bm.build_all(out2, priv2, built["root"], ["ifbench"], None, None, enforce_counts=False, check_revisions=False)
    assert s3["status"]["ifbench.json"] == "unchanged"
    p = out2 / "ifbench.json"
    p.write_bytes(p.read_bytes().replace(b'"n": 12', b'"n": 13'))
    with pytest.raises(RuntimeError, match="different content"):
        bm.build_all(out2, priv2, built["root"], ["ifbench"], None, None, enforce_counts=False, check_revisions=False)


def test_results_record_and_shingles(built):
    [rec_path] = list(built["res"].glob("manifests_*.json"))
    rec = json.loads(rec_path.read_text(encoding="utf-8"))
    assert rec["manifests"] == built["summary"]["manifests"] and rec["private"] == built["summary"]["private"]
    assert rec["selection_rules_sha256"] == bm.rules_sha256()
    assert rec["checks"] == CHECKS
    tool_shingles = pytest.importorskip("tools.shingles", reason="tools/shingles.py (tools area) not present")
    listed = tool_shingles.read_shingle_file(built["sh"])
    assert listed.prefixes and listed.options and built["summary"]["shingles"]["prefixes"] == len(listed.prefixes)
    assert listed.matches(syn.gpqa_row(1)["Question"])  # withheld GPQA question text is covered
    assert listed.matches("so it is " + syn.gpqa_row(4)["Incorrect Answer 1"] + " then")  # a 5-word option
    # 1-3-word options are not hashed (Amendment 1, 2026-10-04): too generic to be evidence of a leak.
    assert not listed.matches("so it is " + syn.gpqa_row(4)["Correct Answer"] + " then")  # a 3-word option
    assert listed.matches("Synthetic RGB query 3 about the gadget fair? extra words here")
    assert listed.matches("Synthetische Frage 2 zur Kalibrierung eines Bauteils unter hoher Last?")  # German GPQA
    assert listed.matches("Synthetische Aufgabe 1: berechne die Summe von 1 und 1 genau.")  # AIME-DE
    assert not listed.matches("Write a synthetic note number 1. Use exactly 3 bullet points.")  # public IFBench
    assert not listed.matches("Synthetic problem 1: compute the sum of 1 and 1 exactly.")  # public AIME EN


def test_counts_enforced_by_default(built, tmp_path):
    with pytest.raises(RuntimeError, match="expected 300 items"):
        bm.build_all(tmp_path / "m", tmp_path / "p", built["root"], ["ifbench"], None, None,
                     enforce_counts=True, check_revisions=False)
    with pytest.raises(RuntimeError, match="expected a primary set of 198 items, got 6"):
        bm.build_all(tmp_path / "m", tmp_path / "p", built["root"], ["gpqa_diamond_en"], None, None,
                     enforce_counts=True, check_revisions=False)
    # Amendment 4: the synthetic Lite has the registered composition, so the enforced Lite builds pass (588)...
    s = bm.build_all(tmp_path / "m", tmp_path / "p", built["root"], ["mmlu_prox_lite_en", "mmlu_prox_lite_de"], None,
                     None, enforce_counts=True, check_revisions=False)
    assert s["counts"] == {"mmlu_prox_lite_en": 588, "mmlu_prox_lite_de": 588}
    # ... and the 42 x 14 the pre-registration assumed is refused when counts are enforced
    bal = tmp_path / "balanced"
    syn.write_mmlu(bal, lite_per_cat=42, full_extra_per_cat=11)
    with pytest.raises(RuntimeError, match="differ from the registered lite_category_counts"):
        bm.build_all(tmp_path / "m2", tmp_path / "p2", bal, ["mmlu_prox_lite_en"], None, None,
                     enforce_counts=True, check_revisions=False)


def _rewrite_jsonl(p: Path, fn) -> None:
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    p.write_text("".join(json.dumps(fn(r)) + "\n" for r in rows), encoding="utf-8")


def test_unexpected_full_gold_mismatch_stops_the_build(built, tmp_path):
    """Amendment 4: the full pool leaves out gold-mismatch rows, and every build (counts enforced or not, as items_for
    on the run host) requires their ids to equal the registered ["3787"] per language: a second one stops it."""
    import shutil

    root = tmp_path / "data"
    shutil.copytree(built["root"], root)
    victim = str(bm.mmlu_prox.full_pool_ids(root / "MMLU-ProX", root / "MMLU-ProX-Lite")[0])
    for lang in ("en", "de"):
        _rewrite_jsonl(root / "MMLU-ProX" / lang / "test-00000-of-00001.jsonl",
                       lambda r: dict(r, answer="ABCDEFGHIJ"[(r["answer_index"] + 1) % 10]) if str(r["question_id"]) == victim else r)
    for name in bm.MMLU_FULL_SETS:
        with pytest.raises(RuntimeError, match="expected .*3787.*Amendment 4"):
            bm.build_all(tmp_path / "m", tmp_path / "p", root, [name], None, None, enforce_counts=False, check_revisions=False)
    with pytest.raises(ValueError, match="Amendment 4"):
        bm.items_for("mmlu_prox_c1_en", root, built["out"], check_revisions=False)
    # the Lite sets do not read the full split: they still build, unchanged
    s = bm.build_all(tmp_path / "m", tmp_path / "p", root, ["mmlu_prox_lite_en"], None, None,
                     enforce_counts=False, check_revisions=False)
    assert (tmp_path / "m" / "mmlu_prox_lite_en.json").read_bytes() == (built["out"] / "mmlu_prox_lite_en.json").read_bytes()
    assert s["counts"] == {"mmlu_prox_lite_en": 588}


def test_missing_registered_full_gold_mismatch_stops_the_build(built, tmp_path):
    """The registered id must be found: a full split without the gold-mismatch row is a data change too."""
    root = tmp_path / "data"
    syn.write_all(root)
    syn.write_mmlu(root, gold_mismatch=False)
    with pytest.raises(RuntimeError, match="expected .*3787"):
        bm.build_all(tmp_path / "m", tmp_path / "p", root, ["mmlu_prox_full_pilot_en"], None, None,
                     enforce_counts=False, check_revisions=False)


def test_lite_gold_mismatch_stops_the_build(built, tmp_path):
    """Lite stays strict (Amendment 4): a Lite row whose answer letter disagrees with answer_index stops the build."""
    import shutil

    root = tmp_path / "data"
    shutil.copytree(built["root"], root)
    victim = bm.mmlu_prox.parallel_ids(root / "MMLU-ProX-Lite")["law"][0]
    _rewrite_jsonl(root / "MMLU-ProX-Lite" / "de" / "test-00000-of-00001.jsonl",
                   lambda r: dict(r, answer="ABCDEFGHIJ"[(r["answer_index"] + 1) % 10]) if str(r["question_id"]) == victim else r)
    for name in ("mmlu_prox_lite_en", "mmlu_prox_lite_de"):
        with pytest.raises(RuntimeError, match=f"MMLU-ProX {victim}: answer .* disagrees with answer_index"):
            bm.build_all(tmp_path / "m", tmp_path / "p", root, [name], None, None, enforce_counts=False, check_revisions=False)


def _plant_overlong(monkeypatch, sha):
    """Point the vendored over-long filter at a synthetic question (the real text never enters the kit)."""
    from tasks.vendored_evalfw import shim

    monkeypatch.setattr(shim, "overlong_question_sha256", lambda: sha)
    monkeypatch.setattr(shim, "gpqa_overlong_filter",
                        lambda: (lambda row: hashlib.sha256(row["Question"].encode()).hexdigest() != sha))


def test_overlong_question_outside_diamond_excludes_nothing(built, tmp_path, monkeypatch):
    """Amendment 3, as on the real data: the over-long question is a gpqa_extended.csv row, so the filter keeps
    every Diamond item and the build records 0 (counts enforced or not)."""
    _plant_overlong(monkeypatch, syn.OVERLONG_SHA256)
    s = bm.build_all(tmp_path / "m", tmp_path / "p", built["root"], ["gpqa_diamond_en", "gpqa_pilot_en"], None, None,
                     enforce_counts=False, check_revisions=False)
    assert s["checks"] == {"gpqa_diamond_en_overlong_excluded": 0}  # no MMLU-ProX full set built: no gold check
    assert (tmp_path / "m" / "gpqa_diamond_en.json").read_bytes() == (built["out"] / "gpqa_diamond_en.json").read_bytes()
    assert (tmp_path / "m" / "gpqa_pilot_en.json").read_bytes() == (built["out"] / "gpqa_pilot_en.json").read_bytes()


def test_overlong_question_inside_diamond_stops_the_build(built, tmp_path, monkeypatch):
    """A data change that put the over-long question into Diamond cannot slip through: the build stops (also with
    counts not enforced, as in the dry run and in items_for on the run host)."""
    _plant_overlong(monkeypatch, hashlib.sha256(syn.gpqa_row(3)["Question"].encode()).hexdigest())
    with pytest.raises(RuntimeError, match="expected 0 over-long item"):
        bm.build_all(tmp_path / "m", tmp_path / "p", built["root"], ["gpqa_diamond_en"], None, None,
                     enforce_counts=False, check_revisions=False)
    with pytest.raises(ValueError, match="expected 0 over-long item"):
        bm.items_for("gpqa_diamond_en", built["root"], built["out"], check_revisions=False)


def test_items_for_round_trip_and_mismatch(built, tmp_path):
    for name in ("gpqa_diamond_en", "mmlu_prox_lite_de", "mmlu_prox_c1_en", "rgb_negative", "rgb_fact", "aime_de",
                 "ifbench_pilot"):
        items = bm.items_for(name, built["root"], built["out"], check_revisions=False)
        doc = json.loads((built["out"] / f"{name}.json").read_text(encoding="utf-8"))
        assert [it.id for it in items] == [e["id"] for e in doc["items"]]
    # a recorded document index that no longer reproduces the prompt is refused
    doc = json.loads((built["out"] / "rgb_negative.json").read_text(encoding="utf-8"))
    doc["items"][0]["doc_indices"] = list(reversed(doc["items"][0]["doc_indices"]))
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "rgb_negative.json").write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(bm.ManifestMismatch):
        bm.items_for("rgb_negative", built["root"], bad, check_revisions=False)


def test_public_entry_guards():
    from tasks.common import make_item

    it = make_item("gpqa_en", "rec1", {"q": "x"}, [{"role": "user", "content": "x"}], gold="B")
    it.public["category"] = "physics"
    with pytest.raises(ValueError, match="not allowed"):
        bm.public_entry(it, withheld=True)
    it2 = make_item("mmlu_en", "1", {"q": "x"}, [{"role": "user", "content": "x"}])
    it2.public["category"] = "y" * 200
    with pytest.raises(ValueError, match="long string"):
        bm.public_entry(it2, withheld=False)


def test_selection_rules_consistent():
    r = bm.rules()
    assert r["selection_seed"] == 36 and r["rgb_seed"] == 2333
    assert r["mmlu_prox"]["nM_ladder"] == [588, 504, 406, 350, 294, 252, 196, 154]  # BUILD_SPEC §7.2
    # Amendment 4: the ladder values are unchanged, but n_M need not be a multiple of 14 any more; the registered Lite
    # counts (not 42 per category) total 588, name every subject, and give every category an item at every n_M.
    m = r["mmlu_prox"]
    assert "per_category_lite" not in m and sum(m["lite_category_counts"].values()) == 588
    assert list(m["lite_category_counts"]) == bm.mmlu_prox.subjects()
    assert all(min(bm.mmlu_prox.allocation(m["lite_category_counts"], n).values()) >= 1 for n in m["nM_ladder"])
    assert m["gold_inconsistent_expected"] == {"en": ["3787"], "de": ["3787"]}
    assert bm.set_def("mmlu_prox_lite_en").expected_n == bm.set_def("mmlu_prox_lite_de").expected_n == 588
    names = {sd.name for sd in bm.set_defs()}
    for arm, cells in r["pilot"].items():
        if arm == "rule":
            continue
        assert {c["manifest"] for c in cells} <= names and {c["task"] for c in cells} <= set(r["tasks"])
    assert {c["task"] for c in r["pilot"]["K8"]} - {c["task"] for c in r["pilot"]["G8"]} == {"aime_en"}
    assert {c["task"] for c in r["pilot"]["G8"]} - {c["task"] for c in r["pilot"]["K4"]} == {"rgb_forced"}
    assert {sd.task for sd in bm.set_defs()} <= set(r["tasks"])
    assert {t["cap_family"] for t in r["tasks"].values()} == {"gpqa", "mmlu", "aime", "ifbench", "rgb"}
    plan_rules = EXP / "runner" / "plan_rules.json"
    if plan_rules.is_file():
        assert json.loads(plan_rules.read_text(encoding="utf-8"))["nM_ladder"] == r["mmlu_prox"]["nM_ladder"]
    lo, hi = r["mmlu_prox"]["full_layout"]["c1"]
    assert hi - lo == r["c1"]["n"] == 100
