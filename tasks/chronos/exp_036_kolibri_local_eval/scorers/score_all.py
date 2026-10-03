"""Raw JSONL -> score JSONL (BUILD_SPEC §5.6; HYPOTHESIS "What counts as evidence", RUNBOOK steps 17-19).

    mbp  (step 17):  python scorers/score_all.py
    mini (step 19):  python scorers/score_all.py --ifbench                      # adds the IFBench scores
                     python scorers/score_all.py --rescore-compare --ifbench    # rescore_mini_<UTC>.json

Reads every `results/raw/<session>/<arm>/<task>_<effort>.jsonl` (all sessions; the first complete record
per key counts) and writes one score record per item to `results/scores/<arm>/<task>_<effort>[_p<k>].jsonl`,
sorted by (pass, item):

    {key, item_sha256, extracted, category, correct, truncated, parse_status, reasoning_status,
     reasoning_lang [, subset] [, correct_loose, correct_strict (IFBench)]}

- extracted: a letter (GPQA, MMLU), an integer (AIME), the per-instruction checker results
  {instruction_id_list, loose, strict} (IFBench), null (RGB: the category carries the result).
- category: the MMLU-ProX category; for RGB the split (closed-book and forced: correct | abstain | wrong
  | truncated; negative: rejected | not | truncated; fact-check: corrected | deferred | detected | other
  | truncated); null otherwise. `subset` (RGB: en | en_fact) is copied from the manifest when present.
- correct: the task's rule, and never for a truncated item, unclosed reasoning or an empty answer (C7).
  IFBench: prompt-level loose (as the vendor reports); correct_strict alongside.
- parse_status: ok | truncated | unclosed_reasoning | empty_answer | no_match | non_integer. Parse
  failures (pilot rule 3) are `PARSE_FAILURES` among non-truncated items.
- reasoning_status: closed | unclosed | none (scorers/reasoning.py); reasoning_lang: scorers/lang_tag.py
  on the reasoning segment (E4).

The reasoning split uses the record's `prompt_prefill` (runner/chat.py, from the actual render), else the
arm's family and effort by the pinned templates. Only the text after the reasoning segment is scored.

Withheld sets (GPQA EN/DE, AIME-DE, RGB): the repo record carries only `text_sha256`; the full record is
read from `$EXP036_PRIVATE/raw/<same relative path>` and its text must hash to that value. Their score
records carry no free text. Gold comes from the item manifests (tasks/build_manifests.py): public sets
from tasks/manifests/<set>.json, withheld sets from $EXP036_PRIVATE/manifests/<set>.jsonl.

IFBench is scored only with --ifbench (the mini, where the IFBench venv lives); without it its cells are
listed as skipped.

Guards: score_all() calls runner.guard.require_identity() and runner.guard.require_tier2() before
writing anything (BUILD_SPEC §1 Tier 2, §2 Identity). A score file is never rewritten: an identical one is
left as it is, a different one stops the run (a changed score is an amendment).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    # Run as a script: make the experiment directory importable (scorers, runner, tasks).
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scorers import aime as aime_scorer  # noqa: E402
from scorers import lang_tag  # noqa: E402
from scorers import mc as mc_scorer  # noqa: E402
from scorers import reasoning  # noqa: E402
from scorers import rgb as rgb_scorer  # noqa: E402

EXP_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TaskSpec:
    kind: str  # mc | aime | ifbench | rgb_cb | rgb_neg | rgb_fact
    withheld: bool


TASKS: dict[str, TaskSpec] = {
    "gpqa_en": TaskSpec("mc", True),
    "gpqa_de": TaskSpec("mc", True),
    "mmlu_en": TaskSpec("mc", False),
    "mmlu_de": TaskSpec("mc", False),
    "aime_en": TaskSpec("aime", False),
    "aime_de": TaskSpec("aime", True),
    "ifbench": TaskSpec("ifbench", False),
    "rgb_cb": TaskSpec("rgb_cb", True),
    "rgb_forced": TaskSpec("rgb_cb", True),
    "rgb_neg": TaskSpec("rgb_neg", True),
    "rgb_fact": TaskSpec("rgb_fact", True),
}

# Accepted spellings of the task names (the runner and the manifests may use the longer ones).
TASK_ALIASES = {
    "gpqa_diamond_en": "gpqa_en", "gpqa_d_en": "gpqa_en",
    "gpqa_diamond_de": "gpqa_de", "gpqa_d_de": "gpqa_de",
    "mmlu_prox_en": "mmlu_en", "mmlu_prox_lite_en": "mmlu_en",
    "mmlu_prox_de": "mmlu_de", "mmlu_prox_lite_de": "mmlu_de",
    "aime26_en": "aime_en", "aime26_de": "aime_de",
    "rgb_closed_book": "rgb_cb", "rgb_closedbook": "rgb_cb",
    "rgb_negative": "rgb_neg", "rgb_fact_check": "rgb_fact", "rgb_factcheck": "rgb_fact",
    "rgb_forced_cb": "rgb_forced",
}

PARSE_FAILURES = frozenset({"unclosed_reasoning", "empty_answer", "no_match"})


def canonical_task(task: str) -> str:
    t = TASK_ALIASES.get(task, task)
    if t not in TASKS:
        raise ValueError(f"unknown task {task!r}; known: {sorted(TASKS)} (+ aliases {sorted(TASK_ALIASES)})")
    return t


# ---------------------------------------------------------------------------------------------------
# Reading raw files


def read_jsonl(path: Path) -> tuple[list[dict], int]:
    """(parsed objects, number of unparsable lines). Torn lines are skipped and counted, never repaired."""
    objs: list[dict] = []
    bad = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                objs.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return objs, bad


def records_of(objs: Iterable[dict]) -> list[dict]:
    """The `type: "record"` lines; the first complete record per key counts (BUILD_SPEC §5.4)."""
    seen: set[str] = set()
    out: list[dict] = []
    for o in objs:
        if o.get("type", "record") != "record" or "key" not in o:
            continue
        k = key_id(o["key"])
        if k in seen:
            continue
        seen.add(k)
        out.append(o)
    return out


def key_id(key: dict) -> str:
    return json.dumps(key, sort_keys=True, ensure_ascii=False)


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def completion_text(record: dict, private_index: dict[str, dict] | None) -> str:
    """The raw completion text: from the record (public sets) or from its private mirror (withheld sets)."""
    if "text" in record and record["text"] is not None:
        return record["text"]
    want = record.get("text_sha256")
    if want is None:
        raise ValueError(f"record {record.get('key')} has neither text nor text_sha256")
    if private_index is None:
        raise FileNotFoundError(f"withheld record {record.get('key')} needs the private mirror ($EXP036_PRIVATE)")
    priv = private_index.get(key_id(record["key"]))
    if priv is None or priv.get("text") is None:
        raise KeyError(f"withheld record {record.get('key')} is missing from the private mirror")
    if text_sha256(priv["text"]) != want:
        raise ValueError(f"private text of {record.get('key')} does not match text_sha256")
    return priv["text"]


# ---------------------------------------------------------------------------------------------------
# Items and gold


def load_items(task: str, manifests_dir: Path | None, private_dir: Path | None) -> dict[str, dict]:
    """item id -> manifest row, from `<task>.jsonl` (or .json) in the public or private manifests.

    Public sets (MMLU-ProX, AIME EN, IFBench) are read from `manifests_dir` (tasks/manifests); withheld
    sets from `private_dir/manifests`. Rows are {id, item_sha256, gold?, category?, gold_fake?}; for RGB
    `gold` is the answer list as stored by RGB and `gold_fake` the counterfactual answer (Fact-Check).
    """
    spec = TASKS[canonical_task(task)]
    base = (private_dir / "manifests") if spec.withheld else manifests_dir
    if base is None:
        raise FileNotFoundError(f"no manifest directory for {task} (withheld={spec.withheld})")
    names = [task] + [a for a, t in TASK_ALIASES.items() if t == task]
    for name in names:
        for ext in (".jsonl", ".json"):
            p = Path(base) / f"{name}{ext}"
            if p.is_file():
                rows = _read_manifest(p)
                return {str(r["id"]): r for r in rows}
    raise FileNotFoundError(f"no manifest for {task} in {base} (tried {names} as .jsonl/.json)")


def _read_manifest(p: Path) -> list[dict]:
    if p.suffix == ".jsonl":
        objs, bad = read_jsonl(p)
        if bad:
            raise ValueError(f"{p}: {bad} unparsable lines")
        return [o for o in objs if "id" in o]
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("items", data.get("rows", []))
    return list(data)


# ---------------------------------------------------------------------------------------------------
# Scoring one record


def prompt_state(record: dict, family: str) -> str:
    """The prompt state of a record: from the runner's `prompt_prefill` label (the actual render), else
    from the arm's family and effort by the pinned templates (reasoning.expected_prompt_state)."""
    if record.get("prompt_prefill") is not None:
        return reasoning.prompt_state_from_prefill(record["prompt_prefill"])
    if record.get("prompt_state") is not None:
        return record["prompt_state"]
    return reasoning.expected_prompt_state(family, effort=record["key"].get("effort"))


def split_record(task: str, record: dict, text: str) -> tuple[str, str, str, bool]:
    """(answer, reasoning_status, family, truncated) for a raw record and its completion text."""
    answer, status, family, truncated, _reasoning = _split(record, text)
    return answer, status, family, truncated


def _split(record: dict, text: str) -> tuple[str, str, str, bool, str]:
    family = reasoning.family_of_arm(record["key"]["arm"])
    reasoning_text, answer, status = reasoning.split_text(family, prompt_state(record, family), text)
    truncated = bool(record.get("truncated")) or record.get("finish_reason") == "length"
    return answer, status, family, truncated, reasoning_text


def parse_record(task: str, record: dict, text: str) -> dict:
    """Gold-free parse view for the pilot (plan_fix): {reasoning_status, truncated, parse_status, defect}."""
    task = canonical_task(task)
    answer, status, family, truncated = split_record(task, record, text)
    parse_status = _base_status(status, truncated, answer)
    if parse_status is None:
        kind = TASKS[task].kind
        if kind == "mc":
            parse_status = "ok" if mc_scorer.EXTRACT[task](answer) is not None else "no_match"
        elif kind == "aime":
            raw = aime_scorer.extract(answer)
            parse_status = "no_match" if (raw is None or not raw.strip()) else (
                "ok" if aime_scorer.to_int(raw) is not None else "non_integer")
        else:
            parse_status = "ok"
    defect = reasoning.reasoning_defect(family, status, effort=record["key"].get("effort"))
    return {"reasoning_status": status, "truncated": truncated, "parse_status": parse_status, "defect": defect}


def _base_status(status: str, truncated: bool, answer: str) -> str | None:
    if truncated:
        return "truncated"
    if status == "unclosed":
        return "unclosed_reasoning"
    if not answer.strip():
        return "empty_answer"
    return None


def score_record(task: str, record: dict, text: str, item: dict) -> dict:
    """One score record (IFBench excluded: it needs the checker subprocess, see score_ifbench_records)."""
    task = canonical_task(task)
    kind = TASKS[task].kind
    if kind == "ifbench":
        raise ValueError("IFBench records are scored in bulk by score_ifbench_records")
    if record.get("item_sha256") != item.get("item_sha256"):
        raise ValueError(f"item_sha256 mismatch for {record['key']}: record vs manifest")
    answer, status, _family, truncated, reasoning_text = _split(record, text)
    base = _base_status(status, truncated, answer)
    gold = item.get("gold")
    category = item.get("category") if kind == "mc" else None

    if kind == "mc":
        res = mc_scorer.score(task, answer, gold)
        extracted = res["extracted"]
        ok = res["correct"]
        parse = res["parse_status"]
    elif kind == "aime":
        res = aime_scorer.score(answer, gold)
        extracted = res["extracted"]
        ok = res["correct"]
        parse = res["parse_status"]
    else:  # RGB: the category is the result; no free text is kept
        extracted = None
        parse = "ok" if answer.strip() else "empty_answer"
        if kind == "rgb_cb":
            category = rgb_scorer.classify_cb(answer, gold)
            ok = category == "correct"
        elif kind == "rgb_neg":
            category = rgb_scorer.classify_neg(answer)
            ok = category == "rejected"
        else:
            category = rgb_scorer.classify_fact(answer, gold, item.get("gold_fake"))
            ok = category in ("corrected", "detected")
        if truncated:
            category = "truncated"

    if base is not None:
        parse = base
    out = {
        "key": record["key"],
        "item_sha256": record.get("item_sha256"),
        "extracted": extracted,
        "category": category,
        "correct": bool(ok) and base is None,
        "truncated": truncated,
        "parse_status": parse,
        "reasoning_status": status,
        "reasoning_lang": lang_tag.tag(reasoning_text),
    }
    if item.get("subset") is not None:
        out["subset"] = item["subset"]
    return out


def score_ifbench_records(records: list[dict], texts: list[str], items: dict[str, dict], **adapter_kwargs) -> list[dict]:
    """Score IFBench records in one checker run (scorers/ifbench_adapter.py)."""
    from scorers import ifbench_adapter

    answers: dict[str, str] = {}
    meta: dict[str, tuple] = {}
    for rec, text in zip(records, texts):
        item_id = str(rec["key"]["item"])
        if item_id in answers:
            raise ValueError(f"IFBench item {item_id} appears twice in one cell (pass > 0 is not scored here)")
        if item_id in items and rec.get("item_sha256") != items[item_id].get("item_sha256"):
            raise ValueError(f"item_sha256 mismatch for IFBench item {item_id}")
        answer, status, _family, truncated, reasoning_text = _split(rec, text)
        answers[item_id] = answer
        meta[item_id] = (rec, status, truncated, lang_tag.tag(reasoning_text))
    scored = ifbench_adapter.score_answers(
        answers, truncated={k: m[2] for k, m in meta.items()}, **adapter_kwargs
    )
    out = []
    for item_id, (rec, status, truncated, lang) in meta.items():
        s = scored[item_id]
        parse = _base_status(status, truncated, answers[item_id]) or s["parse_status"]
        out.append({
            "key": rec["key"],
            "item_sha256": rec.get("item_sha256"),
            "extracted": s["extracted"],
            "category": None,
            "correct": bool(s["correct"]) and parse == "ok",
            "correct_loose": bool(s["correct"]) and parse == "ok",
            "correct_strict": bool(s["correct_strict"]) and parse == "ok",
            "truncated": truncated,
            "parse_status": parse,
            "reasoning_status": status,
            "reasoning_lang": lang,
        })
    return out


def score_ifbench_file(raw_jsonl, ifbench_src=None, ifbench_test=None, private_dir=None,
                       manifests_dir=None, **adapter_kwargs) -> list[dict]:
    """ifbench_adapter.score(): one raw IFBench file -> score records (sorted)."""
    objs, _bad = read_jsonl(Path(raw_jsonl))
    records = records_of(objs)
    texts = [completion_text(r, None) for r in records]
    items: dict[str, dict] = {}
    if manifests_dir is not None:
        items = load_items("ifbench", Path(manifests_dir), None)
    return sorted(
        score_ifbench_records(records, texts, items, ifbench_src=ifbench_src, ifbench_test=ifbench_test,
                              **adapter_kwargs),
        key=_sort_key,
    )


# ---------------------------------------------------------------------------------------------------
# Whole results tree


def _sort_key(score: dict):
    k = score["key"]
    item = str(k.get("item"))
    return (int(k.get("pass", 0)), (0, int(item)) if item.isdigit() else (1, item))


def parse_stem(stem: str) -> tuple[str, str, int]:
    """`<task>_<effort>` or `<task>_<effort>_p<k>` (runner/common.py cell_file_stem) -> (task, effort, pass)."""
    m = re.fullmatch(r"(?P<rest>.+)_p(?P<pass>[0-9]+)", stem)
    rest, pass_ = (m["rest"], int(m["pass"])) if m else (stem, 0)
    if "_" not in rest:
        raise ValueError(f"unrecognised raw file name {stem}.jsonl (expected <task>_<effort>[_p<k>].jsonl)")
    task, effort = rest.rsplit("_", 1)
    return canonical_task(task), effort, pass_


def score_file_name(task: str, effort: str, pass_: int) -> str:
    return f"{task}_{effort}.jsonl" if pass_ == 0 else f"{task}_{effort}_p{pass_}.jsonl"


def discover(results_dir: Path) -> dict[tuple[str, str, str, int], list[Path]]:
    """(arm, task, effort, pass) -> raw files across sessions, in session-name order."""
    cells: dict[tuple[str, str, str, int], list[Path]] = {}
    raw = results_dir / "raw"
    if not raw.is_dir():
        return cells
    for p in sorted(raw.glob("*/*/*.jsonl")):
        if p.name.endswith(".steps.jsonl"):
            continue
        task, effort, pass_ = parse_stem(p.stem)
        cells.setdefault((p.parent.name, task, effort, pass_), []).append(p)
    return cells


def _private_index(raw_path: Path, results_dir: Path, private_dir: Path | None) -> dict[str, dict] | None:
    if private_dir is None:
        return None
    rel = raw_path.relative_to(results_dir / "raw")
    p = private_dir / "raw" / rel
    if not p.is_file():
        return None
    objs, _bad = read_jsonl(p)
    return {key_id(r["key"]): r for r in records_of(objs)}


def dumps(score: dict) -> str:
    return json.dumps(score, sort_keys=True, ensure_ascii=False)


def score_cell(arm: str, task: str, effort: str, raw_files: list[Path], results_dir: Path,
               private_dir: Path | None, manifests_dir: Path | None, ifbench: bool,
               ifbench_kwargs: dict | None = None) -> tuple[list[dict], dict]:
    records: list[dict] = []
    texts: list[str] = []
    seen: set[str] = set()
    torn = 0
    for path in raw_files:
        objs, bad = read_jsonl(path)
        torn += bad
        priv = _private_index(path, results_dir, private_dir) if TASKS[task].withheld else None
        for rec in records_of(objs):
            k = key_id(rec["key"])
            if k in seen:
                continue
            seen.add(k)
            records.append(rec)
            texts.append(completion_text(rec, priv))
    if TASKS[task].kind == "ifbench":
        if not ifbench:
            return [], {"status": "skipped (IFBench is scored on the mini with --ifbench)", "records": len(records)}
        items = load_items(task, manifests_dir, private_dir) if manifests_dir is not None else {}
        scores = score_ifbench_records(records, texts, items, **(ifbench_kwargs or {}))
    else:
        items = load_items(task, manifests_dir, private_dir)
        scores = []
        for rec, text in zip(records, texts):
            item = items.get(str(rec["key"]["item"]))
            if item is None:
                raise KeyError(f"item {rec['key']['item']} of {arm}/{task}_{effort} is not in the manifest")
            scores.append(score_record(task, rec, text, item))
    scores.sort(key=_sort_key)
    summary = {
        "status": "scored",
        "records": len(scores),
        "torn_lines": torn,
        "truncated": sum(s["truncated"] for s in scores),
        "parse_failures": sum(s["parse_status"] in PARSE_FAILURES for s in scores),
    }
    return scores, summary


def write_scores(path: Path, scores: list[dict]) -> str:
    """Write a score file; never rewrite one. Returns "written" | "unchanged"; raises on a difference."""
    data = "".join(dumps(s) + "\n" for s in scores).encode("utf-8")
    if path.exists():
        if path.read_bytes() == data:
            return "unchanged"
        raise RuntimeError(f"{path} exists with different content; a changed score needs an amendment")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return "written"


def score_all(results_dir, private_dir=None, *, manifests_dir=None, out_dir=None, ifbench: bool = False,
              ifbench_kwargs: dict | None = None, require_guards: bool = True) -> dict:
    """BUILD_SPEC §5.6: score every raw cell under results_dir into <out_dir or results_dir>/scores/."""
    results_dir = Path(results_dir)
    private_dir = Path(private_dir).expanduser() if private_dir is not None else None
    manifests_dir = Path(manifests_dir) if manifests_dir is not None else EXP_DIR / "tasks" / "manifests"
    out_root = Path(out_dir) if out_dir is not None else results_dir
    if require_guards:
        from runner import guard  # lazy: the runner owns identity and the Tier-2 amendment check

        guard.require_identity()
        guard.require_tier2()
    report: dict[str, dict] = {}
    for (arm, task, effort, pass_), files in sorted(discover(results_dir).items()):
        scores, summary = score_cell(arm, task, effort, files, results_dir, private_dir, manifests_dir,
                                     ifbench, ifbench_kwargs)
        fname = score_file_name(task, effort, pass_)
        if summary["status"] == "scored":
            summary["file"] = write_scores(out_root / "scores" / arm / fname, scores)
        report[f"{arm}/{fname}"] = summary
    return report


def compare_trees(a: Path, b: Path) -> dict:
    """Byte-for-byte comparison of two scores/ trees: {relpath: "identical" | "different" | "only_in_a" | "only_in_b"}."""
    fa = {p.relative_to(a).as_posix(): p for p in a.rglob("*.jsonl")} if a.is_dir() else {}
    fb = {p.relative_to(b).as_posix(): p for p in b.rglob("*.jsonl")} if b.is_dir() else {}
    out = {}
    for rel in sorted(set(fa) | set(fb)):
        if rel not in fb:
            out[rel] = "only_in_a"
        elif rel not in fa:
            out[rel] = "only_in_b"
        else:
            out[rel] = "identical" if fa[rel].read_bytes() == fb[rel].read_bytes() else "different"
    return out


def rescore_compare(results_dir, private_dir=None, *, manifests_dir=None, ifbench: bool = False,
                    ifbench_kwargs: dict | None = None, require_guards: bool = True) -> Path:
    """The mini re-score (RUNBOOK step 19): score every set into a temporary tree, compare each file with
    results/scores byte for byte, and write results/rescore_mini_<UTC>.json in the shape
    analysis/verdicts.py:check_rescore reads:

        {"utc", "all_identical", "files": [{"path": "scores/<arm>/<task>_<effort>.jsonl",
          "sha256_mbp": hex | null, "sha256_mini": hex | null, "mini_only": bool, "match": bool}, ...]}

    IFBench is scored only on the mini: its files are mini_only with sha256_mbp null, and they match when
    the committed file (written by `score_all.py --ifbench` on the mini) equals the re-score. Run with
    --ifbench so they are re-scored too.
    """
    results_dir = Path(results_dir)
    repo_root = results_dir / "scores"
    with tempfile.TemporaryDirectory(prefix="exp036_rescore_") as tmp:
        score_all(results_dir, private_dir, manifests_dir=manifests_dir, out_dir=tmp, ifbench=ifbench,
                  ifbench_kwargs=ifbench_kwargs, require_guards=require_guards)
        mini_root = Path(tmp) / "scores"
        rels = sorted({p.relative_to(r).as_posix() for r in (repo_root, mini_root) if r.is_dir()
                       for p in r.rglob("*.jsonl")})
        entries = []
        for rel in rels:
            repo_sha = _file_sha256(repo_root / rel)
            mini_sha = _file_sha256(mini_root / rel)
            mini_only = Path(rel).name.startswith("ifbench_")
            entries.append({
                "path": f"scores/{rel}",
                "sha256_mbp": None if mini_only else repo_sha,
                "sha256_mini": mini_sha,
                "mini_only": mini_only,
                "match": repo_sha is not None and repo_sha == mini_sha,
            })
    utc = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = results_dir / f"rescore_mini_{utc}.json"
    payload = {"utc": utc, "files": entries, "all_identical": bool(entries) and all(e["match"] for e in entries)}
    if out.exists():
        raise RuntimeError(f"{out} exists")
    out.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    return out


def _file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="exp_036: raw JSONL -> results/scores (BUILD_SPEC §5.6)")
    ap.add_argument("--results", default=str(EXP_DIR / "results"))
    ap.add_argument("--private", default=os.environ.get("EXP036_PRIVATE"))
    ap.add_argument("--manifests", default=str(EXP_DIR / "tasks" / "manifests"))
    ap.add_argument("--ifbench", action="store_true", help="also score IFBench (the mini, IFBench venv)")
    ap.add_argument("--rescore-compare", action="store_true", help="the mini re-score and comparison")
    args = ap.parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.rescore_compare:
        # Committed IFBench score files can only match a re-score that includes IFBench: --rescore-compare implies
        # --ifbench whenever they exist (review fix 2026-10-03; RUNBOOK step 19 also passes it).
        ifb = args.ifbench or any(Path(args.results, "scores").glob("*/ifbench_*.jsonl"))
        if ifb and not args.ifbench:
            print("score_all: IFBench score files exist; --rescore-compare re-scores them too (--ifbench implied)",
                  file=sys.stderr)
        path = rescore_compare(args.results, args.private, manifests_dir=args.manifests, ifbench=ifb)
        print(path)
        return 0 if json.loads(path.read_text())["all_identical"] else 1
    report = score_all(args.results, args.private, manifests_dir=args.manifests, ifbench=args.ifbench)
    for name, s in report.items():
        print(f"{name}: {s['status']}, {s['records']} records"
              + (f", truncated {s['truncated']}, parse failures {s['parse_failures']}, torn {s['torn_lines']},"
                 f" file {s['file']}" if s["status"] == "scored" else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
