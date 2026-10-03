# SPDX-License-Identifier: MIT
"""Control C1 — greedy answer flips from batching (HYPOTHESIS "Exploratory and
descriptive", Control C1; C15; BUILD_SPEC §5.7 batch_flip.py).
Output: results/bench/c1_<UTC>.jsonl (hashed records) and the full records,
completion ids included, in $EXP036_WORK/c1/c1_<UTC>.full.jsonl (MMLU-ProX
full is published as ids, hashes and counts only: HYPOTHESIS "Task sets").

K8 at effort none, greedy, on the 100 C1 items (MMLU-ProX full, non-Lite, EN),
twice through mlx-lm's BatchGenerator built as BUILD_SPEC item 27 says:
  1. at B = the memory-rule B of the K8 MMLU cell at the initial cap, as the
     newest preflight record computed it (memory_rule.c1_B); without one, the
     same rule now (runner.memory.choose_B("K8", "mmlu", cap, L)); both are
     recorded;
  2. at B = 1.
Items are inserted in manifest order and slots are refilled as sequences end.
Per item: the extracted answer in each run (post-reasoning text through
scorers.reasoning.split_reasoning and scorers.mc.extract_mmlu_en; a truncated
completion counts as no answer), whether it flipped, and the first position
where the two completions' token ids diverge.

Siblings used (imported lazily): tasks.assets.asset, tasks.mmlu_prox.c1_items
(items carry their rendered MMLU_PRO_COT_V2 messages), runner.chat.render,
runner.memory.choose_B, runner.memory.effective_limit,
scorers.reasoning.split_reasoning, scorers.mc.extract_mmlu_en. Tests inject
`prepare` and `extract` instead.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bench import common, genutil  # noqa: E402

SPEC = "HYPOTHESIS Control C1; BUILD_SPEC §5.7 batch_flip.py"
C1_ARM = "K8"
C1_TASK = "mmlu"
C1_EFFORT = "none"
C1_N = 100
C1_INITIAL_CAP = 32768  # runner/plan_rules.json caps.mmlu (BUILD_SPEC §7.2)
# Completion budget per item. Not fixed by HYPOTHESIS or BUILD_SPEC; effort
# none answers are short (planning assumption: 0.08 x the effort-high MMLU
# length, ~160 tokens), and a greedy loop would otherwise run to 32k tokens.
C1_MAX_TOKENS = 4096


def first_divergence(a: Sequence[int], b: Sequence[int]) -> Optional[int]:
    """Index of the first differing token id; the shorter length when one is a
    proper prefix of the other; None when identical."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def initial_cap() -> int:
    p = common.EXP_DIR / "runner" / "plan_rules.json"
    if p.is_file():
        return int(json.loads(p.read_text())["caps"][C1_TASK])
    return C1_INITIAL_CAP


def preflight_c1_B(results_dir: Path) -> Optional[dict]:
    """The C1 B the newest preflight record computed (tools/preflight.py
    writes it as record["memory_rule"]["c1_B"])."""
    files = sorted(Path(results_dir).glob("preflight_*.json"))
    if not files:
        return None
    rec = json.loads(files[-1].read_text())
    b = (rec.get("memory_rule") or {}).get("c1_B")
    return {"file": files[-1].name, "B": b if isinstance(b, int) else None}


def memory_rule_B(results_dir: Path) -> dict:
    """The C1 B: the value the newest preflight record computed (BUILD_SPEC
    §5.7 "computed at preflight and recorded in the C1 file"), else the same
    rule computed now. Both values are recorded, with whether they agree."""
    from runner import memory

    L, sources = memory.effective_limit()
    cap = initial_cap()
    now = int(memory.choose_B(C1_ARM, C1_TASK, cap, L))
    pre = preflight_c1_B(results_dir)
    use_pre = pre is not None and isinstance(pre.get("B"), int)
    return {
        "B": pre["B"] if use_pre else now,
        "source": "preflight memory_rule.c1_B" if use_pre else "runner.memory.choose_B at bench time",
        "preflight": pre,
        "B_now": now,
        "agree": (pre["B"] == now) if use_pre else None,
        "L_bytes_now": int(L),
        "L_sources_now": sources,
        "cap": cap,
    }


def default_prepare(tokenizer) -> list[dict]:
    """The 100 C1 items (tasks.mmlu_prox.c1_items: EN, non-Lite pool, after the
    pilot items), rendered for Kolibri at effort none through runner.chat:
    [{id, item_sha256, prompt_sha256, rendered_prompt_sha256, prompt_ids}]."""
    from runner import chat
    from tasks import assets, mmlu_prox

    full = assets.asset("li-lab/MMLU-ProX").path
    lite = assets.asset("li-lab/MMLU-ProX-Lite").path
    out = []
    for item in mmlu_prox.c1_items(full, lite, n=C1_N):
        _text, ids, rendered_sha = chat.render("kolibri", tokenizer, item.messages, C1_EFFORT)
        out.append(
            {
                "id": str(item.id),
                "item_sha256": item.item_sha256,
                "prompt_sha256": item.prompt_sha256,
                "rendered_prompt_sha256": rendered_sha,
                "prompt_ids": [int(i) for i in ids],
            }
        )
    if len(out) != C1_N:
        raise common.BenchError(f"C1 needs {C1_N} items, got {len(out)}")
    return out


def default_extract(tokenizer) -> Callable[[list[int], list[int]], Optional[str]]:
    from scorers import mc, reasoning

    def extract(prompt_ids: list[int], completion_ids: list[int]) -> Optional[str]:
        _r, answer, _status = reasoning.split_reasoning("kolibri", prompt_ids, completion_ids, tokenizer)
        return mc.extract_mmlu_en(answer)

    return extract


def run_cell(
    results_dir: Optional[Path] = None,
    *,
    loader: Callable[[str], tuple] = genutil.load_arm,
    prepare: Optional[Callable] = None,
    extract: Optional[Callable] = None,
    B: Optional[int] = None,
    max_tokens: int = C1_MAX_TOKENS,
    full_dir: Optional[Path] = None,
) -> Path:
    results_dir = Path(results_dir or common.default_results_dir())
    t_cell = common.utc_iso()  # the header's t_start: generation runs before the file opens
    common.require_identity()
    common.require_gates([C1_ARM])
    b_rule = memory_rule_B(results_dir) if B is None else {"B": int(B), "source": "argument"}
    if b_rule["B"] < 1:
        raise common.CouldNotRun("the memory rule gives B = 0 for the K8 MMLU cell: K8 does not fit (HYPOTHESIS K8 working-set rule)")

    genutil.release()
    model, tok, d = loader(C1_ARM)
    stop = genutil.eos_ids(d, tok)
    items = (prepare or default_prepare)(tok)
    extract = extract or default_extract(tok)
    prompts = [it["prompt_ids"] for it in items]

    runs = {}
    timing = {}
    for label, b in (("B", b_rule["B"]), ("B1", 1)):
        t0 = common.utc_iso()
        runs[label] = genutil.batch_completions(model, prompts, b, max_tokens, stop)
        timing[label] = {"B": b, "t_start": t0, "t_end": common.utc_iso()}

    header = {
        **common.base_header("c1", SPEC, results_dir),
        "arm": C1_ARM,
        "model_dir": common.redact_path(d),
        "fingerprint": common.model_fingerprint(C1_ARM, d),
        "effort": C1_EFFORT,
        "task": "MMLU-ProX full (non-Lite) EN, C1 items",
        "n_items": len(items),
        "B_rule": b_rule,
        "max_tokens": max_tokens,
        "stop_ids": stop,
        "sampler": "greedy (argmax)",
        "generation": "mlx_lm BatchGenerator as BUILD_SPEC item 27 (prefill_batch_size min(B, 8), prefill_step_size 2048, max_kv_size None)",
        "runs": timing,
        "t_start": t_cell,
    }
    stamp_path = common.new_output_path(results_dir, "c1")
    fdir = Path(full_dir or common.work_dir() / "c1")
    fdir.mkdir(parents=True, exist_ok=True)
    full_path = fdir / (stamp_path.stem + ".full.jsonl")

    n_flip = n_div = n_parsed_both = 0
    divergences = []
    with common.JsonlCell(stamp_path, header) as w, common.JsonlCell(full_path, {**header, "note": "full records; not committed"}) as wf:
        for it, rb, r1 in zip(items, runs["B"], runs["B1"]):
            per = {}
            for label, r in (("B", rb), ("B1", r1)):
                truncated = r["finish_reason"] == "length"
                ex = None if truncated else extract(it["prompt_ids"], r["completion_ids"])
                per[label] = {
                    "completion_ids_sha256": common.sha256_ids(r["completion_ids"]),
                    "completion_tokens": len(r["completion_ids"]),
                    "finish_reason": r["finish_reason"],
                    "truncated": truncated,
                    "extracted": ex,
                    "parse_status": "truncated" if truncated else ("ok" if ex is not None else "parse_failure"),
                }
            flipped = per["B"]["extracted"] != per["B1"]["extracted"]
            div = first_divergence(rb["completion_ids"], r1["completion_ids"])
            n_flip += flipped
            n_div += div is not None
            n_parsed_both += per["B"]["extracted"] is not None and per["B1"]["extracted"] is not None
            if div is not None:
                divergences.append(div)
            rec = {
                "key": {"arm": C1_ARM, "task": "c1", "effort": C1_EFFORT, "item": it["id"]},
                "item_sha256": it.get("item_sha256"),
                "prompt_sha256": it.get("prompt_sha256"),
                "rendered_prompt_sha256": it.get("rendered_prompt_sha256"),
                "rendered_prompt_tokens": len(it["prompt_ids"]),
                "runs": per,
                "flipped": flipped,
                "first_divergence": div,
            }
            w.append(rec)
            wf.append({**rec, "prompt_ids": it["prompt_ids"], "completion_ids": {"B": rb["completion_ids"], "B1": r1["completion_ids"]}})
        summary = {
            "n_items": len(items),
            "B": b_rule["B"],
            "n_flips": n_flip,
            "flip_rate": n_flip / len(items) if items else None,
            "n_parsed_both": n_parsed_both,
            "n_diverged": n_div,
            "first_divergence_positions": sorted(divergences),
            "full_records": {"file": common.redact_path(full_path)},
        }
        wf.finish(summary)
        summary["full_records"]["sha256"] = common.sha256_file(full_path)
        w.finish(summary)
    del model, tok
    genutil.release()
    return stamp_path
