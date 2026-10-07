"""RGB (chen700564/RGB @ 65ec39e4): loaders and the four prompt conditions, rendered at run time.

BUILD_SPEC §5.5 `tasks/rgb.py`; HYPOTHESIS H6, H7, E2, E3, C21, C22, "Task sets".

RGB's code and data are CC BY-NC-SA 4.0 (README, Licence). Nothing of RGB (no query, answer, document or
instruction string) is committed: every prompt is rendered on the run host from the local RGB files, and
the manifests carry ids, hashes and document indices only. The prompt construction mirrors RGB's evalue.py
at the pinned commit, read and re-expressed here, not copied:

- `closed_book`: evalue.py with `passage_num = 0`: `docs = []`, the text is
  `instruction.format(QUERY=query, DOCS='')`, and the model is called without the system prompt.
  (RGB's own OpenAI wrapper would add its default system string; HYPOTHESIS C21 fixes "no system prompt".)
- `forced`: our frozen text tasks/prompts/rgb_forced_en.txt, no system message (E2, C22).
- `negative`: evalue.py with `noise_rate = 1.0`, `passage_num = 5`: `random.seed(2333)` per instance, then
  `processdata`: the first 5 entries of `negative` (fewer if the item has fewer), shuffled; the documents
  joined by "\\n" fill `{DOCS}`; the system prompt is RGB's `system`.
- `fact`: RGB has no config/instruction_fact.yaml at the pin (evalue.py names it only under
  `--factchecking`). Its README evaluates counterfactual robustness by running evalue.py on `en_fact` with
  the defaults (noise_rate 0, correct_rate 0, passage_num 5) and the standard instruction.yaml prompt, then
  scoring "factual errors" mentions. That is reproduced: `processdata`'s `_fact` branch draws
  `random.sample(range(len(positive)), min(len(positive), 5))`, takes `positive_wrong` at those indices, and
  shuffles them; system + instruction as for `negative`.

RNG draws are reproduced with `random.Random(2333)`, which is the state `random.seed(2333)` gives RGB's
module-level calls. The chosen document indices are recorded in the manifest; the run host renders from the
recorded indices (not the RNG) and checks the prompt sha256 against the manifest.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

from tasks.common import Item, default_path, json_sha256, make_item, seeded_order, user_message

REPO = "chen700564/RGB"
PROMPTS = Path(__file__).resolve().parent / "prompts"
FORCED_TEMPLATE = PROMPTS / "rgb_forced_en.txt"
RGB_SEED = 2333
PASSAGE_NUM = 5


def _dir(rgb_dir: Path | None) -> Path:
    return Path(rgb_dir) if rgb_dir is not None else default_path(REPO)


def _data(rgb_dir: Path | None, name: str) -> Path:
    rgb_dir = _dir(rgb_dir)
    p = rgb_dir / "data" / name
    if not p.is_file():
        raise FileNotFoundError(f"RGB file data/{name} not found under {Path(rgb_dir).name}/")
    return p


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_en(rgb_dir: Path | None = None) -> list[dict]:
    return _read_jsonl(_data(rgb_dir, "en.json"))


def load_fact(rgb_dir: Path | None = None) -> list[dict]:
    return _read_jsonl(_data(rgb_dir, "en_fact.json"))


def load_int(rgb_dir: Path | None = None) -> list[dict]:
    return _read_jsonl(_data(rgb_dir, "en_int.json"))


def load_instruction(rgb_dir: Path | None = None, lang: str = "en") -> dict:
    """{"system", "instruction"} from config/instruction.yaml (RGB's prompt for every condition but forced)."""
    import yaml  # lazy: only the run host's rendering needs it

    p = _dir(rgb_dir) / "config" / "instruction.yaml"
    doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    entry = doc[lang]
    if "{QUERY}" not in entry["instruction"] or "{DOCS}" not in entry["instruction"]:
        raise ValueError("RGB instruction lacks the {QUERY}/{DOCS} placeholders")
    return {"system": entry["system"], "instruction": entry["instruction"]}


def item_id(source: str, row: dict) -> str:
    """Ids are unique only within a file, so they carry the file: en-17, en_fact-17, en_int-17."""
    return f"{source}-{row['id']}"


def _source_fields(row: dict) -> dict:
    out = {"id": row["id"], "query": row["query"], "answer": row["answer"]}
    if "fakeanswer" in row:
        out["fakeanswer"] = row["fakeanswer"]
    return out


# --------------------------------------------------------------------------------------------------
# Document choice (RGB processdata, reproduced) and rendering
# --------------------------------------------------------------------------------------------------


def negative_indices(row: dict, seed: int = RGB_SEED, passage_num: int = PASSAGE_NUM) -> list[int]:
    """Indices into row["negative"], in prompt order (noise_rate 1.0)."""
    rng = random.Random(seed)
    neg_num = passage_num  # noise_rate == 1: neg_num = passage_num, pos_num = 0
    idx = list(range(len(row["negative"][:neg_num])))
    rng.shuffle(idx)
    return idx


def fact_indices(row: dict, seed: int = RGB_SEED, passage_num: int = PASSAGE_NUM) -> list[int]:
    """Indices into row["positive_wrong"], in prompt order (noise_rate 0, correct_rate 0)."""
    rng = random.Random(seed)
    neg_num = math.ceil(passage_num * 0.0)
    correct_num = math.ceil(passage_num * 0.0)
    pos_num = passage_num - neg_num - correct_num
    indexs = list(range(len(row["positive"])))
    selected = rng.sample(indexs, min(len(indexs), pos_num))
    # correct_num == 0 and neg_num == 0: no further documents; then one shuffle of the selection.
    rng.shuffle(selected)
    return selected


def closed_book(row: dict, instr: dict | None = None) -> list[dict]:
    instr = instr or load_instruction()
    return user_message(instr["instruction"].format(QUERY=row["query"], DOCS=""))


def forced_template() -> str:
    text = FORCED_TEMPLATE.read_text(encoding="utf-8")
    return text[:-1] if text.endswith("\n") else text


def forced(row: dict) -> list[dict]:
    return user_message(forced_template().format(QUERY=row["query"]))


def with_docs(row: dict, instr: dict, docs: list[str]) -> list[dict]:
    text = instr["instruction"].format(QUERY=row["query"], DOCS="\n".join(docs))
    return [{"role": "system", "content": instr["system"]}, {"role": "user", "content": text}]


def negative(row: dict, instr: dict | None = None, indices: list[int] | None = None,
             seed: int = RGB_SEED) -> tuple[list[dict], list[int]]:
    """(messages, document indices into row["negative"]); recorded indices, when given, replace the RNG."""
    instr = instr or load_instruction()
    idx = negative_indices(row, seed) if indices is None else list(indices)
    return with_docs(row, instr, [row["negative"][i] for i in idx]), idx


def fact(row: dict, instr: dict | None = None, indices: list[int] | None = None,
         seed: int = RGB_SEED) -> tuple[list[dict], list[int]]:
    """(messages, document indices into row["positive_wrong"]); recorded indices replace the RNG."""
    instr = instr or load_instruction()
    idx = fact_indices(row, seed) if indices is None else list(indices)
    return with_docs(row, instr, [row["positive_wrong"][i] for i in idx]), idx


# --------------------------------------------------------------------------------------------------
# Items per condition (HYPOTHESIS "Task sets")
# --------------------------------------------------------------------------------------------------


def _item(task: str, source: str, row: dict, messages: list[dict], **public) -> Item:
    # gold = the answer as RGB stores it (str, list of slots, a slot may be a list of aliases); the
    # counterfactual answer of en_fact goes to the private manifest as gold_fake (scorers/score_all.py).
    it = make_item(task, item_id(source, row), _source_fields(row), messages, gold=row["answer"])
    if "fakeanswer" in row:
        it.private["gold_fake"] = row["fakeanswer"]
    it.public.update(public)
    it.source = dict(row)
    return it


def cb_items(rgb_dir: Path | None = None, instr: dict | None = None) -> list[Item]:
    """Closed-book, 400: en.json (300) then en_fact.json (100), file order."""
    instr = instr or load_instruction(rgb_dir)
    rows = [("en", r) for r in load_en(rgb_dir)] + [("en_fact", r) for r in load_fact(rgb_dir)]
    return [_item("rgb_cb", s, r, closed_book(r, instr), subset=s) for s, r in rows]


def forced_items(rgb_dir: Path | None = None) -> list[Item]:
    """Forced-answer closed-book, the same 400 items."""
    rows = [("en", r) for r in load_en(rgb_dir)] + [("en_fact", r) for r in load_fact(rgb_dir)]
    return [_item("rgb_forced", s, r, forced(r), subset=s) for s, r in rows]


def negative_items(rgb_dir: Path | None = None, instr: dict | None = None, indices: dict[str, list[int]] | None = None) -> list[Item]:
    """RGB Negative, 300 (en.json; noise rate 1.0, 5 negative passages)."""
    instr = instr or load_instruction(rgb_dir)
    out = []
    for r in load_en(rgb_dir):
        iid = item_id("en", r)
        msgs, idx = negative(r, instr, None if indices is None else indices[iid])
        out.append(_item("rgb_neg", "en", r, msgs, doc_source="negative", doc_indices=idx))
    return out


def fact_items(rgb_dir: Path | None = None, instr: dict | None = None, indices: dict[str, list[int]] | None = None) -> list[Item]:
    """RGB Fact-Check, 100 (en_fact.json; counterfactual positive_wrong passages)."""
    instr = instr or load_instruction(rgb_dir)
    out = []
    for r in load_fact(rgb_dir):
        iid = item_id("en_fact", r)
        msgs, idx = fact(r, instr, None if indices is None else indices[iid])
        out.append(_item("rgb_fact", "en_fact", r, msgs, doc_source="positive_wrong", doc_indices=idx))
    return out


def pilot_ids(rgb_dir: Path | None = None, n: int = 16, seed: int = 36) -> list[str]:
    """RGB pilot pool: en_int ids in the seed-`seed` order (C19: disjoint from en and en_fact)."""
    return seeded_order([item_id("en_int", r) for r in load_int(rgb_dir)], seed, "rgb_pilot")[:n]


def pilot_items(rgb_dir: Path | None = None, n_cb: int = 16, n_forced: int = 8, seed: int = 36,
                instr: dict | None = None) -> tuple[list[Item], list[Item]]:
    """(closed-book pilot items, forced-answer pilot items); the forced ones are the first n_forced of the
    closed-book selection."""
    instr = instr or load_instruction(rgb_dir)
    rows = {item_id("en_int", r): r for r in load_int(rgb_dir)}
    ids = pilot_ids(rgb_dir, n_cb, seed)
    cb = [_item("rgb_cb", "en_int", rows[i], closed_book(rows[i], instr), subset="en_int") for i in ids]
    fo = [_item("rgb_forced", "en_int", rows[i], forced(rows[i]), subset="en_int") for i in ids[:n_forced]]
    return cb, fo


def instruction_sha256(instr: dict) -> str:
    """Binds the RGB prompt strings in use without publishing them."""
    return json_sha256(instr)
