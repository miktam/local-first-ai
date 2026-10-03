"""tasks/build_manifests.py and selection_rules.json on a complete synthetic data tree (BUILD_SPEC §5.5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import tasks_synthetic as syn
from tasks import build_manifests as bm
from tasks.common import sha256_hex

EXP = Path(__file__).resolve().parent.parent
TEXT_KEYS = {"text", "messages", "question", "prompt", "problem", "query", "answer", "options", "content"}


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
    # interleaved by rank: the first 14 entries are rank 0 of every category, so n_M is a prefix
    assert [e["cat_rank"] for e in lite[:14]] == [0] * 14 and len({e["category"] for e in lite[:14]}) == 14
    assert all(isinstance(e["gold"], int) for e in docs["aime_en"]["items"])
    assert all(set(e) == set(bm.BASE_FIELDS) for e in docs["ifbench"]["items"])
    assert {e["id"] for e in docs["gpqa_diamond_en"]["items"]} >= {"recSYN0000"}
    assert all(e["in_primary"] is True for e in docs["gpqa_diamond_en"]["items"])
    assert all(len(e["doc_indices"]) >= 1 for e in docs["rgb_negative"]["items"])
    assert docs["rgb_negative"]["doc_source"] == "negative" and docs["rgb_fact"]["doc_source"] == "positive_wrong"
    counts = docs["mmlu_prox_category_counts"]
    assert set(counts) == {"en", "de", "_meta"} and counts["_meta"]["totals"]["en"] == 14 * 14
    assert set(counts["en"]) == set(syn.MMLU_CATS) and set(counts["en"].values()) == {14}


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
    tool_shingles = pytest.importorskip("tools.shingles", reason="tools/shingles.py (tools area) not present")
    listed = tool_shingles.read_shingle_file(built["sh"])
    assert listed.prefixes and listed.options and built["summary"]["shingles"]["prefixes"] == len(listed.prefixes)
    assert listed.matches(syn.gpqa_row(1)["Question"])  # withheld GPQA question text is covered
    assert listed.matches("so it is " + syn.gpqa_row(4)["Correct Answer"] + " then")  # a 3-word option
    assert listed.matches("Synthetic RGB query 3 about the gadget fair? extra words here")
    assert listed.matches("Synthetische Frage 2 zur Kalibrierung eines Bauteils unter hoher Last?")  # German GPQA
    assert listed.matches("Synthetische Aufgabe 1: berechne die Summe von 1 und 1 genau.")  # AIME-DE
    assert not listed.matches("Write a synthetic note number 1. Use exactly 3 bullet points.")  # public IFBench
    assert not listed.matches("Synthetic problem 1: compute the sum of 1 and 1 exactly.")  # public AIME EN


def test_counts_enforced_by_default(built, tmp_path):
    with pytest.raises(RuntimeError, match="expected 300 items"):
        bm.build_all(tmp_path / "m", tmp_path / "p", built["root"], ["ifbench"], None, None,
                     enforce_counts=True, check_revisions=False)


def test_items_for_round_trip_and_mismatch(built, tmp_path):
    for name in ("gpqa_diamond_en", "mmlu_prox_lite_de", "rgb_negative", "rgb_fact", "aime_de", "ifbench_pilot"):
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
    assert all(n % 14 == 0 for n in r["mmlu_prox"]["nM_ladder"])
    assert r["mmlu_prox"]["per_category_lite"] * 14 == 588
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
