"""tasks/aime.py and tasks/prompts/aime_de.txt on synthetic AIME-shaped rows (BUILD_SPEC §5.5)."""

from __future__ import annotations

import pytest

import tasks_synthetic as syn
from tasks import aime
from tasks.common import json_sha256
from tasks.vendored_evalfw import shim

AIME_DE_EXPECTED = (
    "Löse die folgende Mathematikaufgabe effizient und klar:\n\n"
    "    - Bei einfachen Aufgaben (höchstens 2 Schritte):\n"
    "    Gib eine knappe Lösung mit minimaler Erklärung an.\n\n"
    "    - Bei komplexen Aufgaben (3 oder mehr Schritte):\n"
    "    Verwende dieses schrittweise Format:\n\n"
    "    ## Schritt 1: [Kurze Beschreibung]\n"
    "    [Knappe Erklärung und Berechnungen]\n\n"
    "    ## Schritt 2: [Kurze Beschreibung]\n"
    "    [Knappe Erklärung und Berechnungen]\n\n"
    "    ...\n\n"
    "    Schließe unabhängig vom Vorgehen immer mit:\n\n"
    "    Daher ist die endgültige Antwort: $\\boxed{answer}$. Ich hoffe, die Antwort ist korrekt.\n\n"
    "    Dabei ist [answer] nur die endgültige Zahl oder der Ausdruck, der die Aufgabe löst.\n\n"
    "    Aufgabe: Synthetische Aufgabe 1: berechne die Summe von 1 und 1 genau."
)


@pytest.fixture()
def dirs(tmp_path):
    return syn.write_aime(tmp_path, n=6)


def test_en_items_use_the_vendored_prompt(dirs):
    en_dir, _ = dirs
    items = aime.load_en(en_dir)
    assert [it.id for it in items] == [str(i) for i in range(6)]
    for it in items:
        assert it.gold == 2 * int(it.id) and it.task == "aime_en"
        expected = shim.aime_query_template().format(Question=it.source["problem"])
        assert it.messages == [{"role": "user", "content": expected}]
        assert it.prompt_sha256 == json_sha256(it.messages)
        assert "$\\boxed{answer}$" in it.prompt_text


def test_de_template_known_good(dirs):
    _, de_dir = dirs
    items = aime.load_de(de_dir)
    assert len(items) == 6 and items[1].messages == [{"role": "user", "content": AIME_DE_EXPECTED}]
    assert not aime.AIME_DE_TEMPLATE.read_text(encoding="utf-8").endswith("\n")
    # the German wrapper keeps the English template's structure line for line
    en_lines = shim.aime_query_template().split("\n")
    de_lines = aime.de_template().split("\n")
    assert len(en_lines) == len(de_lines)
    assert [len(l) - len(l.lstrip(" ")) for l in en_lines] == [len(l) - len(l.lstrip(" ")) for l in de_lines]
    assert "$\\boxed{{answer}}$" in aime.de_template() and de_lines[-1].endswith("{Question}")


def test_gold_must_be_an_aime_integer():
    assert aime.gold_int(" 042 ") == 42
    for bad in ("1000", "-3", "3/4", "x"):
        with pytest.raises(ValueError):
            aime.gold_int(bad)


def test_pilot_is_seeded_subset(dirs):
    en_dir, _ = dirs
    items = aime.load_en(en_dir)
    p = aime.pilot_en(items, n=4)
    assert len(p) == 4 and [it.id for it in p] == [it.id for it in aime.pilot_en(items, n=4)]
    assert {it.id for it in p} <= {it.id for it in items}


def test_ids_fall_back_to_row_index(tmp_path):
    (tmp_path / "aime26").mkdir()
    (tmp_path / "aime26" / "test.jsonl").write_text(
        '{"problem": "Synthetic problem A.", "answer": "1"}\n{"problem": "Synthetic problem B.", "answer": "2"}\n', encoding="utf-8")
    items = aime.load_en(tmp_path / "aime26")
    assert [it.id for it in items] == ["0", "1"] and items[0].public["id_source"] == "_exp036_row_index"
