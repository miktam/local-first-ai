#!/usr/bin/env python3
"""exp_062 Addendum C: writes the made-up texts the speed and same-text parts use. No model is called.

    python3.14 -I kit/build62c.py

  kit/text62c.txt      the test text, seed 62: made-up prose, cut by the driver to 2,048 and to 128 tokens
  kit/smoke62c.txt     another text, seed 900, for the smoke test only
  kit/prompts62c.json  eight short openings for Part 3, and two others for the smoke test

Everything is invented: a harbour town that does not exist, no person's name, no address, no number that could
be a telephone.
"""
import json
import random
from pathlib import Path

KIT = Path(__file__).resolve().parent

PLACES = ["the lower quay", "the salt store", "the rope walk", "the tide gate", "the lamp room", "the fish stairs",
          "the customs shed", "the dry dock", "the upper terrace", "the net loft", "the chandler's yard",
          "the breakwater", "the signal mast", "the ice house", "the ferry slip"]
PEOPLE = ["the harbour clerk", "the night pilot", "the lamp keeper", "the ropemaker", "the ferryman",
          "the cooper", "the tide reader", "the net mender", "the customs weigher", "the ship's carpenter",
          "the baker from the hill", "the salt merchant", "the school mistress", "the bell ringer"]
THINGS = ["a ledger bound in sailcloth", "three coils of tarred rope", "a brass sounding weight",
          "a crate of dried figs", "the tide tables for the coming month", "a cracked signal lantern",
          "two barrels of lamp oil", "a bundle of mended nets", "the keys to the weighing room",
          "a chart with the shoals marked in red", "a sack of coarse salt", "the list of berths",
          "a kettle of pitch", "a ladder with one rung missing"]
VERBS = ["carried", "counted", "left", "found", "checked", "returned", "set down", "locked away", "asked about",
         "wrote down", "repaired", "weighed", "handed over", "looked for"]
TIMES = ["before the morning bell", "at slack water", "when the fog lifted", "after the second ferry",
         "during the long ebb", "as the lamps were lit", "while the wind was still in the north",
         "on the day of the spring tide", "an hour after noon", "late in the watch"]
REASONS = ["because the ledger did not agree with the berths", "so that the pilot could leave on the tide",
           "since nobody else had the key", "although the clerk had said it could wait",
           "as the rule of the harbour required", "because the weather glass was falling",
           "so the count would be right by evening", "though the store was already full"]
LINKS = ["Then", "Later", "By then", "Meanwhile", "After that", "In the end", "As usual", "Once more"]


def paragraph(rng: random.Random, sentences: int) -> str:
    out = []
    for i in range(sentences):
        who, what, where = rng.choice(PEOPLE), rng.choice(THINGS), rng.choice(PLACES)
        s = f"{who} {rng.choice(VERBS)} {what} at {where} {rng.choice(TIMES)}"
        if rng.random() < 0.6:
            s += f", {rng.choice(REASONS)}"
        if i and rng.random() < 0.5:
            s = f"{rng.choice(LINKS)} {s}"
        out.append(s[0].upper() + s[1:] + ".")
    return " ".join(out)


def text(seed: int, paragraphs: int = 60) -> str:
    rng = random.Random(seed)
    head = "Notes on the working day of a small harbour town, as they were kept in the quay book.\n\n"
    return head + "\n\n".join(paragraph(rng, rng.randint(4, 7)) for _ in range(paragraphs)) + "\n"


PROMPTS = [
    ("P1", "The ferry had been late every morning that week, and by Thursday the harbour clerk"),
    ("P2", "To mend a torn net you need three things. First,"),
    ("P3", "Question: A barrel holds forty measures of lamp oil and the lamp room uses three measures a night. "
           "After how many nights is the barrel less than half full?\nAnswer:"),
    ("P4", "def tide_height(hours_after_high: float, tidal_range: float) -> float:\n"
           '    """Height of the water above low tide, by the rule of twelfths."""\n'),
    ("P5", "Dear cousin,\n\nYou asked what the winter is like here. I will tell you plainly:"),
    ("P6", "The five rules of the weighing room are as follows.\n1."),
    ("P7", "El farero subió la escalera despacio, porque"),
    ("P8", "Translate into plain English: \"The said vessel shall not depart the berth until the dues are discharged.\"\n"
           "Plain English:"),
]
SMOKE_PROMPTS = [
    ("S1", "The ropemaker opened the shop late that day because"),
    ("S2", "A short list of things to check before a boat leaves the slip:\n-"),
]


def main():
    (KIT / "text62c.txt").write_text(text(62), encoding="utf-8")
    (KIT / "smoke62c.txt").write_text(text(900), encoding="utf-8")
    (KIT / "prompts62c.json").write_text(json.dumps(
        {"test": [{"id": i, "text": t} for i, t in PROMPTS],
         "smoke": [{"id": i, "text": t} for i, t in SMOKE_PROMPTS]}, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8")
    for f in ("text62c.txt", "smoke62c.txt", "prompts62c.json"):
        print(f, (KIT / f).stat().st_size, "bytes,", len((KIT / f).read_text(encoding="utf-8").split()), "words")


if __name__ == "__main__":
    main()
