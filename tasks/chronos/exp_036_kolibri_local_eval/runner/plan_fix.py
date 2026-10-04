"""Fix n and max_tokens by rule from the pilot (HYPOTHESIS "Pilot and the rule
that fixes n and max_tokens", rules 1-6; BUILD_SPEC §5.4 runner/plan_fix.py;
runner/plan_rules.json, frozen).

fix() is a pure function of its inputs (pilot summary, decode steps, S1
records, plan rules, existing amendments, and a context of file-derived
facts: the Metal limit, weight bytes, manifest and record sha256s, the
scorers tree sha256, the UTC time). Same input, byte-identical output. It
applies, in order:

  1. the cap-raise rule (per task family, for all models, at most once);
  2. the parse and reasoning gate (BLOCKED only with >= 2 parse failures and
     a diagnosed delimiter or extractor mismatch; IFBench and AIME pilots
     never block);
  3. the memory rule for B per cell (runner/memory.py), with the Metal
     limit L the plan uses;
  4. the continuous-batching projection of every cell (runner/simulate.py),
     x 1.15;
  5. P0 ... P10 against B_main = min(31, 40 - S1 hours), with the ordered
     queue split at cell boundaries into S2 and S3, each <= 16 h;
  6. Tier B in order, each item added whole if the total and the split still
     fit;
  7. the ordered queue and its split point;
  8. the row sets (H2: 5 rows, 4 under P10; the AIME secondary iff B1; H7
     adds GPQA iff B8, GPQA-D DE only where K8 runs it, i.e. not under P10);
  9. power at the fixed n (analysis/power.py; the registered assumptions, so
     the H2 figures are conservative under Amendment 4's proportional MMLU
     allocation), and the MMLU-ProX-Lite items per category at n_M (the first
     n_M entries of the Lite manifest, listed in the Webster seat order;
     Amendment 4: not n_M/14 each).

Arms the records exclude (runner/guard.py excluded_arms; review fix
2026-10-03) have no cell in any queue: K4 after a final gate K4 FAIL (H1, H7,
H8 and D1 NOT RUN), a peer arm whose peer check says "fail" (dropped by this
amendment; H3 and H6 use the remaining MoE peer, H4 is NOT RUN without
Qwen3.6). The plan names them under "excluded_arms", the remaining MoE peers
under "peers" and the hypotheses made NOT RUN under "not_run". A peer arm
whose peer check says "B=1" (its batched-path check failed: the parity or the
greedy flip rate; Amendment 5) is not excluded: every cell of it, Tier A and
Tier B, gets B = 1 and is projected at B = 1; the plan lists such arms under
"peer_b1" and the amendment names them.

A peer arm whose peer check says "speed-only" (Amendment 6, runner/guard.py
speed_only_arms: its family's fidelity rule failed and the pinned bf16
reference put the failure on this build) has no quality cell (Tier A or Tier
B task cell; a MoE peer among them is handled as a dropped one for H3, H4, H6
and the H2 protocol control) but is not excluded: the B4 ladder (a bench cell)
keeps it, as do H1 and the speed cells, which bench/ runs outside the plan.
The plan lists such arms under "speed_only_arms". Every family with a build
excluded or speed-only leaves H8's peer median (edge case 7): the plan records
the remaining families as "h8_peers" and why the others left in
"h8_families_left", and puts H8 under "not_run" when none remains;
analysis/verdicts.py applies the same rule.

The CLI (`plan_fix.py --pilot results/pilot_summary_<UTC>.json [...]`)
reads the files, calls fix(), and writes results/plan_fixed_<UTC>.json and
results/AMENDMENT_<k>_<UTC>.md. It never opens an existing file for writing
and never runs a model.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):  # run as a script: put the experiment dir on sys.path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from runner import memory, simulate  # noqa: E402
from runner.common import (  # noqa: E402
    ARMS,
    EXP_DIR,
    KOLIBRI_ARMS,
    arm_dir,
    cell_id,
    dumps,
    load_rules,
    parse_utc,
    sha256_bytes,
    sha256_file,
    task_kind,
    utc_iso,
    utc_stamp,
    write_new_text,
)

MISMATCH_CODES = ("extractor_mismatch", "delimiter_mismatch")
INF = float("inf")


# =========================================================== 2. diagnose()

# Frozen with the runner tree. Deterministic; reads no file.
# Closing delimiters of a reasoning segment. If the id-level split says
# "unclosed" but the text holds one of these, the model closed its reasoning
# in a way the splitter does not see (another delimiter, or the family's own
# delimiter spelled out as plain-text tokens): a delimiter mismatch.
_CLOSE_DELIMS = ("</think>", "</thinking>", "<channel|>", "</reasoning>", "◁/think▷", "<|end_of_thought|>")
_LETTER_PATTERNS = (
    # "answer is (C)", "Answer: C", "Antwort: C", "die richtige Antwort ist C", "Lösung: C"
    r"(?i)\b(?:answer|antwort|lösung|loesung|option|choice|auswahl)\b[^A-Za-z0-9\n]{0,12}"
    r"(?:(?:is|ist|lautet|wäre|would be|:)\s*)?[^A-Za-z0-9\n]{0,6}\(?\b([{L}])\b\)?",
    r"\\boxed\{\s*\(?\\?(?:text\{)?([{L}])\}?\)?\s*\}",       # \boxed{C}
    r"\*\*\s*\(?([{L}])\)?\s*\*\*",                             # **C**
    r"(?m)^\s*\(?([{L}])\)?[.)]?\s*$",                          # a line that is only the letter
)


def _letter_range(task: str) -> str:
    kind = task_kind(task)
    return "A-D" if kind == "gpqa" else "A-J"


def diagnose(task: str, answer_text: str, full_text: str, status: str, finish_reason: str) -> str | None:
    """The pre-written diagnostic of pilot rule 3 for one parse failure.

    Returns "delimiter_mismatch" when the reasoning was not closed by the
    splitter's delimiter but the completion ended by EOS and contains another
    reasoning delimiter (the splitter does not know it), "extractor_mismatch"
    when an answer is visibly present after the reasoning but the vendored
    extractor did not match it, else None (a model failure, not ours)."""
    answer_text = answer_text or ""
    full_text = full_text or ""
    if status == "unclosed" and finish_reason == "stop":
        if any(d in full_text for d in _CLOSE_DELIMS):
            return "delimiter_mismatch"
        return None
    kind = task_kind(task)
    if kind in ("gpqa", "mmlu"):
        L = _letter_range(task)
        for pat in _LETTER_PATTERNS:
            if re.search(pat.replace("{L}", L), answer_text):
                return "extractor_mismatch"
        return None
    if kind == "aime":
        if re.search(r"\\boxed\s*\{", answer_text) or re.search(
            r"(?i)\b(?:answer|antwort)\b[^0-9\n]{0,20}\b\d{1,3}\b", answer_text
        ):
            return "extractor_mismatch"
        return None
    # IFBench / RGB: a parse failure is an empty post-reasoning answer.
    if answer_text.strip() == "" and any(d in full_text for d in _CLOSE_DELIMS):
        return "delimiter_mismatch"
    return None


# ============================================================ pilot inputs


def merge_summaries(summaries: list[dict]) -> dict:
    """Later summaries (a re-pilot) replace the (arm, task, effort) cells they
    re-ran. An arm's timing (its wall-time window, hence the step model, and
    its prefill rate) is replaced only when the later summary re-ran every
    pilot cell of that arm; a re-pilot of one BLOCKED cell keeps the arm's
    original timing and changes only that cell's statistics."""
    cells: dict[tuple, dict] = {}
    arms: dict[str, dict] = {}
    caps = None
    for s in summaries:
        caps = caps or s.get("caps")
        new_cells = {(c["arm"], c["task"], c.get("effort", "high")): c for c in s.get("cells", [])}
        for a, info in s.get("arms", {}).items():
            before = {k for k in cells if k[0] == a}
            if a not in arms or before <= {k for k in new_cells if k[0] == a}:
                arms[a] = info
        cells.update(new_cells)
    return {"cells": [cells[k] for k in sorted(cells)], "arms": {k: arms[k] for k in sorted(arms)}, "caps": caps}


def _cell_index(pilot: dict) -> dict[tuple[str, str], dict]:
    return {(c["arm"], c["task"]): c for c in pilot["cells"]}


def s1_hours(s1_records: Iterable[dict]) -> float:
    """S1 hours = sum of (t_end - t_start) over the S1 records (rule 4)."""
    total = 0.0
    for r in s1_records:
        if r.get("t_start") and r.get("t_end"):
            total += (parse_utc(r["t_end"]) - parse_utc(r["t_start"])).total_seconds()
    return total / 3600.0


# ============================================================ 1. cap raise


def cap_raise(pilot: dict, rules: dict) -> tuple[dict, dict]:
    """(final caps per task family, trigger record). Trigger per family: in
    any model's pilot cell, >= 1 truncation or any completion longer than
    0.5 x the cap (0.4 x for GPQA). Effect: the ceiling, for all models."""
    cr = rules["cap_raise"]
    caps = dict(rules["caps"])
    triggers: dict[str, list] = {}
    for c in pilot["cells"]:
        kind = task_kind(c["task"])
        cap0 = rules["caps"][kind]
        frac = cr["trigger_len_fraction_gpqa"] if kind == "gpqa" else cr["trigger_len_fraction"]
        n_trunc = int(c.get("n_truncated", sum(bool(t) for t in c.get("truncated", []))))
        longest = max(c.get("lengths") or [0])
        why = []
        if n_trunc >= int(cr["trigger_truncations"]):
            why.append(f"{n_trunc} truncated")
        if longest > frac * cap0:
            why.append(f"longest {longest} > {frac} x {cap0}")
        if why:
            triggers.setdefault(kind, []).append({"arm": c["arm"], "task": c["task"], "why": "; ".join(why)})
    for kind in triggers:
        caps[kind] = int(cr["ceilings"][kind])
    return caps, {k: triggers[k] for k in sorted(triggers)}


# =============================================== 2. parse / reasoning gate


def parse_gate(pilot: dict, rules: dict) -> dict:
    g = rules["pilot_gates"]
    gate_tasks = set(rules["pilot"]["parse_gate_tasks"])
    out = {"blocked": [], "cells": {}}
    for c in pilot["cells"]:
        diags = c.get("diagnoses") or {}
        n_mis = sum(int(diags.get(code, 0)) for code in MISMATCH_CODES)
        n_pf = int(c.get("n_parse_fail", 0))
        eligible = c["task"] in gate_tasks
        blocked = eligible and n_pf >= int(g["block_min_failures"]) and (n_mis >= 1 or not g["require_diagnosed_mismatch"])
        key = cell_id(c["arm"], c["task"], c.get("effort", "high"))
        out["cells"][key] = {
            "n": int(c.get("n", len(c.get("lengths", [])))),
            "n_parse_fail": n_pf,
            "n_diagnosed_mismatch": n_mis,
            "n_truncated": int(c.get("n_truncated", 0)),
            "status_counts": c.get("status_counts", {}),
            "reasoning_defects": int(c.get("n_defect", 0)),
            "gate_applies": eligible,
            "blocked": blocked,
        }
        if blocked:
            out["blocked"].append(key)
    return out


# ============================================================== 4. projection


class Projector:
    """Projected hours (x margin) of a cell, from the pilot by rule 4."""

    def __init__(self, pilot: dict, steps: dict, rules: dict, caps: dict, context: dict):
        self.rules = rules
        self.caps = caps
        self.cells = _cell_index(pilot)
        self.arms = pilot.get("arms", {})
        self.margin = float(rules["budget"]["margin"])
        self.seed = int(rules["projection"]["resample_seed"])
        self.notes: dict[str, str] = {}
        self.step_models: dict[str, tuple] = {}
        self.fit_info: dict[str, dict] = {}
        for arm, st in sorted(steps.items()):
            info = self.arms.get(arm, {})
            sel = simulate.select_last_third(st, info.get("t_start"), info.get("t_end"))
            try:
                self.step_models[arm] = simulate.fit_step_model(sel)
                self.fit_info[arm] = {"n_steps": len(sel), "a": self.step_models[arm][0],
                                      "b": self.step_models[arm][1], "c": self.step_models[arm][2]}
            except ValueError as e:
                self.fit_info[arm] = {"error": str(e)}
        for arm, abc in sorted((context.get("step_models_extra") or {}).items()):
            self.step_models[arm] = tuple(float(x) for x in abc)
            self.fit_info[arm] = {"source": "context (descriptive speed cells)", "a": abc[0], "b": abc[1], "c": abc[2]}
        self.prefill: dict[str, float] = {}
        for arm, info in self.arms.items():
            if info.get("prefill_seconds"):
                self.prefill[arm] = float(info["prefill_tokens"]) / float(info["prefill_seconds"])
        for arm, v in (context.get("prefill_tps_extra") or {}).items():
            self.prefill[arm] = float(v)
        self._memo: dict[tuple, float] = {}

    # -- length sources --------------------------------------------------
    def _pilot(self, arm: str, ptask: str) -> dict | None:
        return self.cells.get((arm, ptask))

    def _adjusted_lengths(self, c: dict, kind: str) -> list[float]:
        """Pilot lengths; items truncated at the initial cap count at the post-raise cap."""
        lens = [float(x) for x in c.get("lengths", [])]
        trunc = c.get("truncated") or [False] * len(lens)
        cap = float(self.caps[kind])
        return [cap if t else v for v, t in zip(lens, trunc)]

    def source(self, arm: str, task: str, effort: str) -> tuple[list[float], list[float], float, str]:
        """(lengths, prompt_lengths, scale, note) for a cell."""
        R = self.rules
        kind = task_kind(task)
        ptask = R["pilot_source"].get(task)
        if ptask is None:
            raise KeyError(f"no pilot source for task {task!r}")
        prompt_default = float(R["prompt_tokens_default"].get(task, R["prompt_tokens_default"].get(kind, 300)))
        note = []
        scale = 1.0
        len_arm = arm
        if ARMS[arm]["family"] == "qwen3_8":
            len_arm = "Q36-8"
            note.append("Q38-8 lengths from Q36-8's pilot")

        # Peer AIME: the arm's GPQA pilot lengths x K8's AIME/GPQA length ratio.
        if kind == "aime" and arm not in KOLIBRI_ARMS:
            k_aime = self._pilot("K8", "aime_en_pilot")
            k_gpqa = self._pilot("K8", "gpqa_main_en")
            lang = "de" if task.endswith("_de") else "en"
            own = self._pilot(len_arm, f"gpqa_main_{lang}")
            if not (k_aime and k_gpqa and own):
                raise KeyError(f"{arm} {task}: needs K8 AIME and GPQA pilots and the arm's GPQA {lang} pilot")
            ka = self._adjusted_lengths(k_aime, "aime")
            kg = self._adjusted_lengths(k_gpqa, "gpqa")
            ratio = (sum(ka) / len(ka)) / (sum(kg) / len(kg))
            lens = self._adjusted_lengths(own, "gpqa")
            note.append(f"GPQA {lang} pilot x K8 AIME/GPQA ratio {ratio:.4f}")
            return lens, [prompt_default], ratio * simulate.length_scale(lens), "; ".join(note)

        c = self._pilot(len_arm, ptask)
        own_cell = c is not None
        if c is None and kind == "rgb":
            c = self._pilot(len_arm, "rgb_int_cb")
            if c is not None:
                note.append(f"no {ptask} pilot for {len_arm}: closed-book pilot lengths")
        if c is None:
            raise KeyError(f"no pilot cell {len_arm}:{ptask} for {arm}:{task}")
        lens = self._adjusted_lengths(c, kind)
        scale *= simulate.length_scale(lens)
        if kind == "gpqa":
            scale *= float(R["projection"]["gpqa_diamond_factor"])
        if ARMS[arm]["family"] == "kolibri" and effort != "high":
            r = float(R["derived_lengths"]["kolibri_effort_ratio"][effort])
            scale *= r
            note.append(f"effort {effort}: {r} x effort-high lengths")
        own_prompts = own_cell and task not in ("rgb_neg", "rgb_fact") and bool(c.get("prompt_tokens"))
        prompts = [float(x) for x in c["prompt_tokens"]] if own_prompts else [prompt_default]
        if task in ("rgb_neg", "rgb_fact"):
            note.append("lengths from the closed-book pilot; prompt tokens from prompt_tokens_default")
        return lens, prompts, scale, "; ".join(note)

    # -- one cell --------------------------------------------------------
    def hours(self, cell: dict) -> float:
        """Projected hours of one cell, margin included (inf if unprojectable)."""
        if cell["arm"] == "bench":
            return float(self.rules["ladder_estimate_h"][cell["fixed_h_key"]]) * self.margin
        key = (cell["arm"], cell["task"], cell["effort"], cell.get("pass", 0), cell["n"], cell["cap"], cell["B"])
        if key in self._memo:
            return self._memo[key]
        arm = cell["arm"]
        cid = cell_id(arm, cell["task"], cell["effort"], cell.get("pass", 0))
        try:
            if cell["B"] < 1:
                raise KeyError("does not fit in memory at B = 1")
            lens, prompts, scale, note = self.source(arm, cell["task"], cell["effort"])
            model = self.step_models.get(arm)
            rate = self.prefill.get(arm)
            if model is None or rate is None:
                raise KeyError(f"no step model / prefill rate for {arm}")
            h = simulate.simulate_cell(
                int(cell["n"]), lens, prompts, int(cell["B"]), model, rate,
                seed=self.seed, scale=scale, cap=int(cell["cap"]),
            ) * self.margin
            if note:
                self.notes[cid] = note
        except (KeyError, ValueError) as e:
            self.notes[cid] = f"unprojectable: {e}"
            h = INF
        self._memo[key] = h
        return h


# ===================================================== 5-7. plans and queue


# HYPOTHESIS Tier B "B4 | context ladder for K4, K8, G4": the arms the bench ladder cell loads.
LADDER_ARMS = ("K4", "K8", "G4")
MOE_PEERS = ("G8", "Q36-8")
# H8's peer families (HYPOTHESIS H8: Gemma 4, Qwen3.6, Qwen3.8), in ARMS order; edge case 7 removes a family when a
# build of it is excluded, Amendment 6 when a build of it is speed-only.
H8_PEER_FAMILIES = tuple(dict.fromkeys(s["family"] for a, s in ARMS.items() if a not in KOLIBRI_ARMS))


def tier_a_cells(rules: dict, plan: str, k8_runs: bool = True, excluded: Iterable[str] = ()) -> list[dict]:
    d = rules["plan_defs"][plan]
    excluded = set(excluded)
    out = []
    for c in rules["tier_a_queue"]:
        if c["arm"] in excluded:
            continue
        g = c.get("group", "core")
        if g == "k8_gpqa_de" and not d["k8_gpqa_de"]:
            continue
        if g == "forced" and not d["forced"]:
            continue
        if g == "neg_fc" and not d["neg_fc"]:
            continue
        cell = {k: v for k, v in c.items() if k != "group"}
        cell["n"] = d["nM"] if c["n"] == "nM" else int(c["n"])
        cell["tier"] = "A"
        out.append(cell)
    return _substitute_k8(out, k8_runs)


def tier_b_cells(rules: dict, item: str, plan: str, k8_runs: bool = True, excluded: Iterable[str] = (),
                 speed_only: Iterable[str] = ()) -> list[dict]:
    """excluded: arms with no cell at all (the B4 ladder needs K4, K8 and G4); speed_only (Amendment 6): arms
    with no task cell, which the B4 ladder (a speed and memory bench cell) keeps."""
    d = rules["plan_defs"][plan]
    excluded = set(excluded)
    no_task_cells = excluded | set(speed_only)
    out = []
    for c in rules["tier_b_cells"][item]:
        if c.get("group") == "k8_gpqa_de" and not d["k8_gpqa_de"]:
            continue
        if c["arm"] in no_task_cells or (c["arm"] == "bench" and excluded & set(LADDER_ARMS)):
            continue
        cell = {k: v for k, v in c.items() if k != "group"}
        if cell["arm"] != "bench":
            cell["n"] = d["nM"] if c["n"] == "nM" else int(c["n"])
        cell["tier"] = "B"
        cell["tier_b_item"] = item
        out.append(cell)
    return _substitute_k8(out, k8_runs)


def _substitute_k8(cells: list[dict], k8_runs: bool) -> list[dict]:
    """K8 not runnable: K4 replaces K8 (HYPOTHESIS rule 6); duplicates dropped."""
    if k8_runs:
        return cells
    seen = set()
    out = []
    for c in cells:
        c = dict(c)
        if c["arm"] == "K8":
            c["arm"] = "K4"
            c["replaces"] = "K8"
        k = (c["arm"], c["task"], c["effort"], c.get("pass", 0))
        if k in seen:
            continue
        seen.add(k)
        out.append(c)
    return out


def split_two(hours: list[float], cap: float) -> tuple[bool, int, float, float]:
    """Split the ordered queue at a cell boundary into S2 (the longest prefix
    <= cap) and S3 (the rest); ok iff S3 <= cap too. Returns (ok, index of
    the first S3 cell, S2 hours, S3 hours)."""
    s2 = 0.0
    i = 0
    while i < len(hours) and s2 + hours[i] <= cap:
        s2 += hours[i]
        i += 1
    s3 = sum(hours[i:])
    return (s3 <= cap and all(math.isfinite(h) for h in hours)), i, s2, s3


def choose_plan(plan_hours: dict[str, list[float]], order: list[str], B_main: float, cap: float) -> dict:
    """The first plan whose total <= B_main and whose two-session split works."""
    tried = []
    for p in order:
        hs = plan_hours[p]
        total = sum(hs)
        ok, idx, s2, s3 = split_two(hs, cap)
        fits = total <= B_main
        tried.append({"plan": p, "total_h": total, "fits_budget": fits, "split_ok": ok, "S2_h": s2, "S3_h": s3})
        if fits and ok:
            return {"plan": p, "tried": tried}
    return {"plan": None, "tried": tried}


# ================================================================ fix()


def _r(x: float, nd: int = 4) -> float | None:
    return None if x is None or not math.isfinite(x) else round(float(x), nd)


def next_amendment_k(existing: Iterable[dict]) -> int:
    ks = [int(a["k"]) for a in existing]
    return max([0] + ks) + 1


def _power(n_M: int, h2_rows: int) -> dict:
    try:
        power = importlib.import_module("analysis.power")
    except ModuleNotFoundError:
        return {"status": "analysis/power.py not available"}
    try:
        res = power.power_at_plan(n_M, h2_rows=h2_rows)
    except Exception as e:  # recorded, never fatal: power is descriptive
        return {"status": f"error: {type(e).__name__}: {e}"}
    return json.loads(json.dumps(res, default=float))


def fix(
    pilot_summary: dict,
    steps: dict,
    s1_records: list[dict],
    rules: dict,
    existing_amendments: list[dict],
    *,
    context: dict | None = None,
) -> tuple[dict, str, int]:
    """(plan_fixed, amendment_md, k). context keys: now_utc (ISO), L_bytes,
    L_sources, weight_bytes {arm: bytes}, peer_b1 [arms], excluded_arms
    {arm: reason}, speed_only_arms {arm: reason}, step_models_extra,
    prefill_tps_extra, manifests, category_counts, mmlu_lite_categories (the
    categories of tasks/manifests/mmlu_prox_lite_en.json in listing order),
    gate_record, peers_record, scorers_tree_sha256, inputs."""
    ctx = dict(context or {})
    now = ctx.get("now_utc") or utc_iso()
    k = next_amendment_k(existing_amendments)
    prior_plans = sorted(int(a["k"]) for a in existing_amendments if a.get("type") == "plan")
    supersedes = prior_plans[-1] if prior_plans else None
    B = rules["budget"]
    pilot = merge_summaries([pilot_summary]) if "cells" in pilot_summary else pilot_summary

    # 1. caps
    caps, cap_triggers = cap_raise(pilot, rules)
    # 2. parse / reasoning gate
    gate = parse_gate(pilot, rules)
    S1 = s1_hours(s1_records)
    B_main = min(float(B["main_cap_h"]), float(B["total_h"]) - S1)

    plan: dict[str, Any] = {
        "type": "plan_fixed",
        "rules_version": rules["version"],
        "utc": now,
        "amendment_k": k,
        "supersedes": supersedes,
        "S1_hours": _r(S1),
        "B_main_h": _r(B_main),
        "caps": caps,
        "cap_raised": {kk: caps[kk] != rules["caps"][kk] for kk in sorted(caps)},
        "cap_triggers": cap_triggers,
        "pilot_gate": gate,
        "L_used": ctx.get("L_bytes"),
        "L_sources": ctx.get("L_sources"),
        "sysctl_advice_mb": rules["sysctl_advice_mb"],
        "inputs": ctx.get("inputs", {}),
        "manifests": ctx.get("manifests", {}),
        "category_counts": ctx.get("category_counts", {}),
        "gate_record": ctx.get("gate_record"),
        "peers_record": ctx.get("peers_record"),
        "scorers_tree_sha256": ctx.get("scorers_tree_sha256"),
    }

    if gate["blocked"]:
        plan.update({"status": "BLOCKED", "plan": None, "queue": [],
                     "reason": "pilot cells BLOCKED (>= 2 parse failures with a diagnosed mismatch): "
                               "an extractor or delimiter amendment, then a re-pilot of those cells"})
        return _finish(plan, rules, k, now)

    # 3. memory rule for B
    L = int(ctx.get("L_bytes") or 0)
    wb = ctx.get("weight_bytes", {})
    peer_b1 = set(ctx.get("peer_b1", []))
    plan["peer_b1"] = sorted(peer_b1)
    B_cache: dict[tuple, int] = {}

    def B_for(arm: str, task: str) -> int:
        if arm == "bench":
            return 1
        cap = caps[task_kind(task)]
        key = (arm, task, cap)
        if key not in B_cache:
            if arm not in wb:
                B_cache[key] = 0
            else:
                b = memory.choose_B(arm, task, cap, L, weight_bytes=int(wb[arm]), rules=rules,
                                    kv_bytes_per_token=int(rules["kv_bytes_per_token"][ARMS[arm]["family"]]),
                                    fixed_window_bytes=int(rules["fixed_window_bytes"][ARMS[arm]["family"]]))
                B_cache[key] = min(b, 1) if (arm in peer_b1 and b >= 1) else b
        return B_cache[key]

    k8_tasks = {c["task"] for c in rules["tier_a_queue"] if c["arm"] == "K8"}
    k8_runs = all(B_for("K8", t) >= 1 for t in k8_tasks)
    plan["k8_runs"] = k8_runs
    if not k8_runs:
        plan["k8_note"] = "need(1) > 0.9 L for a K8 cell: K8 NOT RUN; K4 replaces K8 in H2, H3, H4, H6; H7 NOT RUN"

    # Arms the records exclude (gate K4 FAIL; peer-check fail): no cell of theirs is queued (review fix 2026-10-03).
    excluded = {a: str(r) for a, r in sorted((ctx.get("excluded_arms") or {}).items())}
    # Speed-only arms (peer-check "speed-only", Amendment 6): no task cell, but not excluded (the B4 ladder keeps
    # them; H1 and the speed cells run in bench/, outside the plan).
    speed_only = {a: str(r) for a, r in sorted((ctx.get("speed_only_arms") or {}).items()) if a not in excluded}
    no_quality = set(excluded) | set(speed_only)
    not_run: dict[str, str] = {}
    if "K4" in excluded:
        for h in ("H1", "H7", "H8", "D1"):
            not_run[h] = f"K4 excluded: {excluded['K4']} (HYPOTHESIS Phase 0, exit 4)"
    peers = [a for a in MOE_PEERS if a not in no_quality]
    for a in sorted(no_quality & set(MOE_PEERS)):
        why = (f"{a} dropped: {excluded[a]} (HYPOTHESIS \"Peers are verified, not gated\")" if a in excluded
               else f"{a} speed-only: {speed_only[a]} (Amendment 6: no quality cell)")
        if a == "Q36-8":
            not_run["H4"] = why
        if not peers:
            not_run.setdefault("H3", why)
            not_run.setdefault("H6", why)
    # H8's peer median: a family leaves it when either build is excluded or speed-only (edge case 7; Amendment 6);
    # H8 is NOT RUN if no family remains. analysis/verdicts.py applies the same rule.
    h8_peers, h8_left = [], {}
    for fam in H8_PEER_FAMILIES:
        hit = [a for a in ARMS if ARMS[a]["family"] == fam and a in no_quality]
        if hit:
            h8_left[fam] = "; ".join(f"{a} excluded: {excluded[a]}" if a in excluded
                                     else f"{a} speed-only: {speed_only[a]}" for a in hit)
        else:
            h8_peers.append(fam)
    if not h8_peers:
        not_run.setdefault("H8", "every peer family left H8's peer median (edge case 7; Amendment 6): "
                           + "; ".join(f"{f} ({r})" for f, r in sorted(h8_left.items())))
    # What a peer-check fail of G4 does to H1 is left to an amendment (RUNBOOK step 9): the plan lists the
    # exclusion and makes nothing else NOT RUN. A speed-only G4 keeps H1 (Amendment 6).
    plan["excluded_arms"] = excluded
    plan["speed_only_arms"] = speed_only
    plan["peers"] = peers
    plan["h8_peers"] = h8_peers
    plan["h8_families_left"] = h8_left
    plan["not_run"] = dict(sorted(not_run.items()))
    if not k8_runs and "K4" in excluded:
        plan.update({"status": "STOP", "plan": None, "queue": [],
                     "reason": "K8 does not fit at B = 1 and K4 is excluded: no Kolibri arm can run"})
        return _finish(plan, rules, k, now)

    def resolve(cells: list[dict]) -> list[dict]:
        out = []
        for c in cells:
            c = dict(c)
            c.setdefault("pass", 0)
            if c["arm"] != "bench":
                c["cap"] = caps[task_kind(c["task"])]
                c["B"] = B_for(c["arm"], c["task"])
            out.append(c)
        return out

    # 4. projection
    proj = Projector(pilot, steps, rules, caps, ctx)

    def hours_of(cells: list[dict]) -> list[float]:
        return [proj.hours(c) for c in cells]

    # 5. ladder
    cap_s = float(B["session_cap_h"])
    plan_cells = {p: resolve(tier_a_cells(rules, p, k8_runs, no_quality)) for p in rules["plans"]}
    plan_hours = {p: hours_of(cs) for p, cs in plan_cells.items()}
    choice = choose_plan(plan_hours, rules["plans"], B_main, cap_s)
    plan["ladder"] = [{**t, "total_h": _r(t["total_h"]), "S2_h": _r(t["S2_h"]), "S3_h": _r(t["S3_h"])}
                      for t in choice["tried"]]
    if choice["plan"] is None:
        plan.update({"status": "STOP", "plan": None, "queue": [],
                     "reason": "even P10 does not fit B_main with two sessions <= 16 h: no scored run "
                               "until Andrei decides on a 4th session by amendment"})
        plan["projection_notes"] = dict(sorted(proj.notes.items()))
        plan["step_models"] = proj.fit_info
        plan["prefill_tps"] = {a: _r(v, 2) for a, v in sorted(proj.prefill.items())}
        return _finish(plan, rules, k, now)

    P = choice["plan"]
    queue = list(plan_cells[P])
    qh = list(plan_hours[P])

    # 6. Tier B
    tb_log = []
    added = []
    for item in rules["tier_b_order"]:
        cells = resolve(tier_b_cells(rules, item, P, k8_runs, excluded, speed_only))
        # K8 replaced by K4: drop Tier-B cells already queued (e.g. B8 duplicates).
        have = {(c["arm"], c["task"], c["effort"], c["pass"]) for c in queue}
        cells = [c for c in cells if (c["arm"], c["task"], c["effort"], c["pass"]) not in have]
        hs = hours_of(cells)
        total = sum(qh) + sum(hs)
        ok, _, _, s3 = split_two(qh + hs, cap_s)
        fits = total <= B_main and ok and bool(cells)
        reason = ("added" if fits else
                  "no cells" if not cells else
                  "unprojectable" if not all(math.isfinite(h) for h in hs) else
                  f"total {total:.2f} h > B_main {B_main:.2f} h" if total > B_main else
                  f"split fails (S3 {s3:.2f} h > {cap_s} h)")
        tb_log.append({"item": item, "projected_h": _r(sum(hs)), "added": fits, "reason": reason})
        if fits:
            queue += cells
            qh += hs
            added.append(item)

    # 7. queue and split point
    ok, idx, s2, s3 = split_two(qh, cap_s)
    q_out = []
    for i, (c, h) in enumerate(zip(queue, qh)):
        e = {
            "cell": "B4:ladder" if c["arm"] == "bench" else cell_id(c["arm"], c["task"], c["effort"], c["pass"]),
            "arm": c["arm"], "task": c["task"], "effort": c["effort"], "pass": c["pass"],
            "n": c["n"], "tier": c["tier"], "projected_h": _r(h), "session": "S2" if i < idx else "S3",
        }
        if c["arm"] != "bench":
            e.update({"cap": c["cap"], "B": c["B"]})
        else:
            e["needs"] = c.get("needs")
        for opt in ("tier_b_item", "replaces"):
            if opt in c:
                e[opt] = c[opt]
        q_out.append(e)

    d = rules["plan_defs"][P]
    # 8. row sets
    rows = rules["rows"]
    h2 = [r for r in rows["H2"] if not (P == "P10" and r == rows["H2_P10_drop"])]
    # B8 adds GPQA-D EN and DE to H7, paired K4 vs K8 on identical items: only the rows K8 runs (P10 has no K8
    # GPQA-D DE cell; review fix 2026-10-03).
    h7_b8 = [r for r in rows["H7_if_B8"] if d["k8_gpqa_de"] or r != rows["H2_P10_drop"]]
    h7 = list(rows["H7"]) + (h7_b8 if "B8" in added else [])
    h7_ok = k8_runs and "K4" not in excluded
    main_h = sum(qh)
    plan.update({
        "status": "FIXED",
        "plan": P,
        "n_M": d["nM"],
        "forced_cells": d["forced"],
        "neg_fc_cells": d["neg_fc"],
        "k8_gpqa_de": d["k8_gpqa_de"],
        "queue": q_out,
        "split_index": idx,
        "S2_h": _r(s2),
        "S3_h": _r(s3),
        "main_h": _r(main_h),
        "run_total_h": _r(S1 + main_h),
        "tier_b": added,
        "tier_b_considered": tb_log,
        "rows": {
            "H2": h2,
            "H2_arm": "K8" if k8_runs else "K4",
            "H2_secondary_aime": list(rows["H2_secondary_if_B1"]) if "B1" in added else [],
            "H7": h7 if h7_ok else [],
            "H7_status": ("runs" if h7_ok else "NOT RUN (K8 cannot run)" if not k8_runs
                          else f"NOT RUN ({not_run.get('H7', 'K4 excluded')})"),
        },
        "projection_notes": dict(sorted(proj.notes.items())),
        "step_models": proj.fit_info,
        "prefill_tps": {a: _r(v, 2) for a, v in sorted(proj.prefill.items())},
        # 9. power, and the items per category at n_M (Amendment 4)
        "power": _power(int(d["nM"]), len(h2)),
        "mmlu_allocation": mmlu_allocation(ctx.get("mmlu_lite_categories"), int(d["nM"])),
    })
    return _finish(plan, rules, k, now)


def mmlu_allocation(categories: list | None, n_M: int) -> dict | None:
    """{category: items} among the first n_M entries of the MMLU-ProX-Lite listing (Amendment 4: the Webster seat
    order, so this is the per-category n of the run), or None without the manifest."""
    if not categories:
        return None
    if len(categories) < n_M:
        return {"error": f"the Lite manifest lists {len(categories)} items, fewer than n_M = {n_M}"}
    out: dict[str, int] = {}
    for c in categories[:n_M]:
        out[str(c)] = out.get(str(c), 0) + 1
    return dict(sorted(out.items()))


def plan_file_name(now: str) -> str:
    return f"results/plan_fixed_{utc_stamp(parse_utc(now))}.json"


def plan_bytes(plan: dict) -> bytes:
    """The exact bytes written to results/plan_fixed_<UTC>.json."""
    return (dumps(plan, indent=2) + "\n").encode("utf-8")


def _finish(plan: dict, rules: dict, k: int, now: str) -> tuple[dict, str, int]:
    plan = json.loads(dumps(plan))  # normalise (sorted keys, plain JSON types)
    md = amendment_md(plan, k, now, (plan_file_name(now), sha256_bytes(plan_bytes(plan))))
    return plan, md, k


# ============================================================== amendment


def amendment_md(plan: dict, k: int, now: str, plan_file: tuple[str, str] | None) -> str:
    """The plan amendment, appended verbatim to HYPOTHESIS.md by the mbp.
    plan_file = (relpath, sha256) of the written plan_fixed JSON."""
    L = []
    L.append(f"## Amendment {k} — plan ({now})")
    L.append("")
    L.append(f"*Written by `runner/plan_fix.py` from the pilot, by the frozen rule (HYPOTHESIS \"Pilot and the rule "
             f"that fixes n and max_tokens\"; `runner/plan_rules.json` {plan['rules_version']}).*")
    L.append("")
    L.append(f"- Status: {plan['status']}")
    if plan_file:
        L.append(f"- plan_fixed: `{plan_file[0]}`, sha256 `{plan_file[1]}`")
    L.append(f"- Scorers tree sha256: `{plan.get('scorers_tree_sha256') or 'n/a'}`")
    L.append(f"- Supersedes: {('Amendment ' + str(plan['supersedes'])) if plan.get('supersedes') else 'none'}")
    L.append(f"- S1 hours: {plan['S1_hours']}; B_main = min(31, 40 − S1) = {plan['B_main_h']} h")
    caps = plan["caps"]
    raised = [kk for kk, v in plan["cap_raised"].items() if v]
    L.append("- Caps (completion tokens): " + ", ".join(f"{kk} {caps[kk]:,}" for kk in sorted(caps))
             + (f"; raised to the ceiling: {', '.join(raised)}" if raised else "; no raise"))
    if plan.get("L_used"):
        L.append(f"- Metal limit L used: {plan['L_used']:,} B ({plan['L_used'] / 2**30:.2f} GiB)")
    if plan["status"] != "FIXED":
        L.append(f"- Reason: {plan.get('reason', '')}")
        if plan["pilot_gate"]["blocked"]:
            L.append(f"- BLOCKED cells: {', '.join(plan['pilot_gate']['blocked'])}")
        L.append("")
        return "\n".join(L) + "\n"
    L.append(f"- Plan: **{plan['plan']}** (n_M = {plan['n_M']}; forced-answer cells: "
             f"{'yes' if plan['forced_cells'] else 'no'}; RGB Negative and Fact-Check: "
             f"{'yes' if plan['neg_fc_cells'] else 'no'}; K8 GPQA-D DE: {'yes' if plan['k8_gpqa_de'] else 'no'})")
    L.append(f"- K8 runs: {'yes' if plan['k8_runs'] else 'no — ' + plan.get('k8_note', '')}")
    ex = plan.get("excluded_arms") or {}
    if ex:
        L.append("- Excluded arms (no cell queued): " + "; ".join(f"{a} — {r}" for a, r in sorted(ex.items())))
    so = plan.get("speed_only_arms") or {}
    if so:
        L.append("- Speed-only arms (peer-check verdict \"speed-only\", Amendment 6: no quality cell and not in H8's peer "
                 "median; H1, the speed cells and the B4 ladder keep them): "
                 + "; ".join(f"{a} — {r}" for a, r in sorted(so.items())))
    if plan.get("peer_b1"):
        L.append("- Peers at B = 1 (peer-check verdict \"B=1\": the batched-path check failed; every cell of the arm "
                 "runs at B = 1): " + ", ".join(plan["peer_b1"]))
    L.append(f"- MoE peers for H3, H6 and the H2 protocol control: {', '.join(plan.get('peers') or []) or 'none'}")
    if "h8_peers" in plan:
        left = plan.get("h8_families_left") or {}
        L.append(f"- H8 peer families: {', '.join(plan['h8_peers']) or 'none'}"
                 + ("; left H8's peer median (edge case 7, Amendment 6): "
                    + "; ".join(f"{f} ({r})" for f, r in sorted(left.items())) if left else ""))
    if plan.get("not_run"):
        L.append("- NOT RUN by this amendment (p = p_rev = 1, m unchanged): "
                 + "; ".join(f"{h}: {r}" for h, r in sorted(plan["not_run"].items())))
    L.append(f"- Tier B: {', '.join(plan['tier_b']) or 'none'}")
    L.append(f"- Projected main {plan['main_h']} h (S2 {plan['S2_h']} h, S3 {plan['S3_h']} h); "
             f"run total {plan['run_total_h']} h")
    r = plan["rows"]
    L.append(f"- H2 rows ({r['H2_arm']}): {', '.join(r['H2'])}"
             + (f"; AIME secondary: {', '.join(r['H2_secondary_aime'])}" if r["H2_secondary_aime"] else ""))
    L.append(f"- H7 rows: {', '.join(r['H7']) or r['H7_status']}")
    alloc = plan.get("mmlu_allocation")
    if isinstance(alloc, dict) and alloc:
        L.append("- MMLU-ProX-Lite items per category at n_M (Amendment 4, the first n_M of the Webster listing): "
                 + ("; ".join(f"{c} {v}" for c, v in alloc.items()) if "error" not in alloc else alloc["error"]))
    for name, key in (("Gate record", "gate_record"), ("Peer-check record", "peers_record")):
        v = plan.get(key)
        if isinstance(v, dict):
            L.append(f"- {name}: `{v.get('path')}`, sha256 `{v.get('sha256')}`")
    if plan.get("manifests"):
        L.append("- Item manifests (sha256):")
        for name, sha in sorted(plan["manifests"].items()):
            L.append(f"  - `{name}`: `{sha}`")
    L.append("")
    L.append("| # | Cell | n | Cap | B | Tier | Projected h | Session |")
    L.append("|---|---|---|---|---|---|---|---|")
    for i, q in enumerate(plan["queue"], 1):
        L.append(f"| {i} | {q['cell']} | {q['n']} | {q.get('cap', '—')} | {q.get('B', '—')} | "
                 f"{q['tier']}{(' ' + q['tier_b_item']) if q.get('tier_b_item') else ''} | {q['projected_h']} | {q['session']} |")
    L.append("")
    L.append("Tier B considered: " + "; ".join(f"{t['item']} {t['projected_h']} h — {t['reason']}"
                                               for t in plan["tier_b_considered"]) + ".")
    pw = plan.get("power") or {}
    if "H2_D0" in pw:
        L.append(f"Power at n_M = {plan['n_M']} (α/8 / α/3): H2 at D = 0 {pw['H2_D0']['a8']:.2f} / {pw['H2_D0']['a3']:.2f}; "
                 f"H3 at the vendor gap {pw['H3_vendor_gap']['a8']:.2f} / {pw['H3_vendor_gap']['a3']:.2f}; "
                 f"H7 at 0 {pw['H7_0']['a8']:.2f} / {pw['H7_0']['a3']:.2f}. Registered assumptions (H2: MMLU design "
                 f"effect 1.11 of a category-balanced sample; Amendment 4's proportional allocation brings it near "
                 f"1.0, so the H2 figures are conservative).")
    L.append("")
    return "\n".join(L) + "\n"


# ==================================================================== CLI


def existing_amendments(exp_dir: Path) -> list[dict]:
    """Amendment numbers in HYPOTHESIS.md, amendments/ and results/AMENDMENT_*."""
    from runner.guard import amendments as parse

    out = []
    hyp = exp_dir / "HYPOTHESIS.md"
    if hyp.is_file():
        out += [{"k": a["k"], "type": a["type"], "where": "HYPOTHESIS.md"} for a in parse(hyp.read_text(encoding="utf-8"))]
    for p in sorted((exp_dir / "amendments").glob("*.md")) + sorted((exp_dir / "results").glob("AMENDMENT_*.md")):
        for a in parse(p.read_text(encoding="utf-8")):
            out.append({"k": a["k"], "type": a["type"], "where": p.name})
    return out


def load_steps(exp_dir: Path, pilot: dict) -> dict:
    """Every arm's decode steps from the pilot step logs."""
    steps: dict[str, list] = {}
    for c in pilot["cells"]:
        p = c.get("steps_file")
        if not p:
            continue
        fp = exp_dir / p
        if not fp.is_file():
            raise FileNotFoundError(f"pilot step log {p} is missing (fixtures by real name)")
        for line in fp.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") == "step":
                steps.setdefault(c["arm"], []).append(r)
    for a in steps:
        steps[a].sort(key=lambda r: r["t"])
    return steps


S1_GLOBS = ("preflight_*.json", "convert/*.json", "peers_*.json", "gate/gate_*.json", "bench/*.jsonl",
            "bench/*.json", "tokenizer_*.json", "kl_8v4_*.json", "pilot_summary_*.json")


def load_s1_records(results: Path) -> list[dict]:
    """t_start / t_end of every S1 block (rule 4): the top level of each JSON
    record, and the closing {"type": "end"} line of each bench JSONL cell
    file (its inner run lines are not added again). Only preflight --deep
    counts."""
    recs = []
    for pat in S1_GLOBS:
        for p in sorted(results.glob(pat)):
            if p.suffix == ".json":
                try:
                    cands = [json.loads(p.read_text(encoding="utf-8"))]
                except json.JSONDecodeError:
                    continue
            else:
                cands = []
                for line in p.read_text(encoding="utf-8").splitlines():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(r, dict) and r.get("type") == "end":
                        cands.append(r)
            for r in cands:
                if not isinstance(r, dict) or not (r.get("t_start") and r.get("t_end")):
                    continue
                if p.name.startswith("preflight_") and r.get("mode") not in ("deep", "--deep"):
                    continue
                recs.append({"phase": p.relative_to(results).as_posix(), "t_start": r["t_start"], "t_end": r["t_end"]})
    return recs


def q38_from_speed_desc(records: list[dict]) -> tuple[dict, dict]:
    """Q38-8's step model and prefill rate from the descriptive speed cells
    (bench/speed.py speed_desc records; rule 4 "a step model fitted on Q38-8's
    descriptive speed cells"). A "batch" or "decode_b1" record gives one
    point: n_live = B, step_seconds = B / generation_tps, padded_len = the
    mean row length during decode (prompt + half the generated tokens). The
    descriptive cells all use the same prompt length, so c (the padded-length
    term) is not identifiable from them: when the padded lengths vary by less
    than 10 % the fit is a + b·n_live with c = 0, and the plan says so."""
    pts, ptps = [], []
    for r in records:
        if r.get("arm") != "Q38-8":
            continue
        if r.get("kind") in ("batch", "decode_b1") and r.get("generation_tps"):
            B = int(r.get("B", 1))
            gen_row = float(r.get("generation_tokens", 0)) / B
            prompt_row = float(r.get("prompt_tokens", 0)) / B
            pts.append({"n_live": B, "padded_len": prompt_row + gen_row / 2.0,
                        "step_seconds": B / float(r["generation_tps"])})
        if r.get("kind") == "prefill" and r.get("prompt_tps"):
            ptps.append(float(r["prompt_tps"]))
    sm, pf, note = {}, {}, {}
    if len(pts) >= 2:
        pls = [p["padded_len"] for p in pts]
        if max(pls) - min(pls) < 0.1 * max(pls) or len(pts) < 3:
            import numpy as np

            A = np.array([[1.0, p["n_live"]] for p in pts])
            y = np.array([p["step_seconds"] for p in pts])
            (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
            sm["Q38-8"] = [float(a), float(b), 0.0]
            note["Q38-8"] = "a + b*n_live from the descriptive speed cells; c = 0 (not identifiable there)"
        else:
            sm["Q38-8"] = list(simulate.fit_step_model(pts))
    if ptps:
        pf["Q38-8"] = float(sorted(ptps)[len(ptps) // 2])
    return sm, pf | ({"_note": note} if note else {})


def _q38_from_speed_desc(results: Path) -> tuple[dict, dict]:
    recs = []
    for p in sorted(results.glob("bench/speed_desc_*.jsonl")):
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    sm, pf = q38_from_speed_desc(recs)
    pf.pop("_note", None)
    return sm, pf


def build_context(exp_dir: Path, pilot_paths: list[Path]) -> dict:
    results = exp_dir / "results"
    from runner.guard import scorers_tree_sha256

    L, sources = memory.effective_limit()
    wb = {}
    for arm in ARMS:
        d = arm_dir(arm)
        if d.is_dir():
            try:
                wb[arm] = memory.arm_weight_bytes(d)
            except FileNotFoundError:
                pass
    ctx: dict[str, Any] = {"now_utc": utc_iso(), "L_bytes": L, "L_sources": sources, "weight_bytes": wb,
                           "scorers_tree_sha256": scorers_tree_sha256(exp_dir)}
    ctx["inputs"] = {"pilot_summaries": [{"path": p.relative_to(exp_dir).as_posix(), "sha256": sha256_file(p)}
                                         for p in pilot_paths],
                     "plan_rules_sha256": sha256_file(exp_dir / "runner" / "plan_rules.json")}
    mf = sorted(results.glob("manifests_*.json"))
    if mf:
        m = json.loads(mf[-1].read_text(encoding="utf-8"))
        # Public manifests by repo path, withheld (private) ones by their $EXP036_PRIVATE path.
        ctx["manifests"] = {**m.get("manifests", {}),
                            **{f"$EXP036_PRIVATE/{k}": v for k, v in m.get("private", {}).items()}}
        ctx["inputs"]["manifests_record"] = {"path": mf[-1].relative_to(exp_dir).as_posix(), "sha256": sha256_file(mf[-1])}
    cc = exp_dir / "tasks" / "manifests" / "mmlu_prox_category_counts.json"
    if cc.is_file():
        ctx["category_counts"] = json.loads(cc.read_text(encoding="utf-8"))
    lite = exp_dir / "tasks" / "manifests" / "mmlu_prox_lite_en.json"
    if lite.is_file():  # Amendment 4: the plan records the per-category n at n_M
        ctx["mmlu_lite_categories"] = [e.get("category") for e in json.loads(lite.read_text(encoding="utf-8"))["items"]]
    for key, pat in (("gate_record", "gate/gate_*.json"), ("peers_record", "peers_*.json")):
        fs = sorted(results.glob(pat))
        if fs:
            ctx[key] = {"path": fs[-1].relative_to(exp_dir).as_posix(), "sha256": sha256_file(fs[-1])}
            if key == "peers_record":
                rec = json.loads(fs[-1].read_text(encoding="utf-8"))
                arms = rec.get("arms", rec)
                ctx["peer_b1"] = sorted(a for a, e in arms.items() if isinstance(e, dict)
                                        and str(e.get("verdict", e.get("status", ""))).upper() == "B=1")
    sm, pf = _q38_from_speed_desc(results)
    ctx["step_models_extra"], ctx["prefill_tps_extra"] = sm, pf
    from runner.guard import excluded_arms, speed_only_arms

    ctx["excluded_arms"] = excluded_arms(results)
    ctx["speed_only_arms"] = speed_only_arms(results)
    return ctx


def check_pilot_inputs(summaries: list[dict], rules: dict, excluded: dict) -> list[str]:
    """Problems that make the pilot input unusable (review fix 2026-10-03): a summary that is not marked complete
    (an interrupted pilot writes none; an older one lacks the flag), and any pilot cell of an arm that runs which
    neither a summary holds nor a summary skipped by rule (does not fit at B = 1). A re-pilot summary only replaces
    cells, so it must follow the full pilot's summary on the command line."""
    problems = []
    for i, sm in enumerate(summaries, 1):
        if sm.get("complete") is not True:
            problems.append(f"--pilot #{i} is not a complete pilot summary (no \"complete\": true)")
    have = {(c["arm"], c["task"]) for sm in summaries for c in sm.get("cells", [])}
    have |= {(x["arm"], x["task"]) for sm in summaries for x in sm.get("skipped", [])}
    want = {(a, c["task"]) for a in rules["pilot"]["arms"] if a not in excluded for c in rules["pilot"]["cells"][a]}
    missing = sorted(want - have)
    if missing:
        problems.append("pilot cells missing from the summaries: " + ", ".join(f"{a}:{t}" for a, t in missing)
                        + " (pass the full pilot's summary first, then any re-pilot's)")
    extra = sorted({a for a, _ in have} & set(excluded))
    if extra:
        problems.append(f"the summaries hold cells of excluded arms {extra}")
    return problems


def write_outputs(exp_dir: Path, plan: dict, md: str, k: int, now: str) -> tuple[Path, Path]:
    """Write the two new files; refuses to overwrite (open mode "x")."""
    from runner.jsonl import write_new_bytes

    pf = exp_dir / plan_file_name(now)
    am = exp_dir / "results" / f"AMENDMENT_{k}_{utc_stamp(parse_utc(now))}.md"
    write_new_bytes(pf, plan_bytes(plan))
    write_new_text(am, md)
    return pf, am


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fix n and max_tokens by rule from the pilot (BUILD_SPEC §5.4).")
    ap.add_argument("--pilot", action="append", required=True, type=Path,
                    help="results/pilot_summary_<UTC>.json (repeat for a re-pilot; later ones replace cells)")
    ap.add_argument("--exp-dir", type=Path, default=EXP_DIR)
    args = ap.parse_args(argv)
    exp_dir = args.exp_dir.resolve()

    from runner.guard import require_identity

    require_identity(exp_dir)
    paths = [p if p.is_absolute() else (exp_dir / p) for p in args.pilot]
    for p in paths:
        if not p.is_file():
            print(f"missing pilot summary {p}", file=sys.stderr)
            return 1
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    rules = load_rules(exp_dir / "runner" / "plan_rules.json")
    ctx = build_context(exp_dir, paths)
    # A speed-only arm (Amendment 6) has no pilot cell either.
    problems = check_pilot_inputs(summaries, rules, {**(ctx.get("excluded_arms") or {}),
                                                     **(ctx.get("speed_only_arms") or {})})
    if problems:
        print("REFUSED: " + "; ".join(problems), file=sys.stderr)
        return 1
    pilot = merge_summaries(summaries)
    steps = load_steps(exp_dir, pilot)
    s1 = load_s1_records(exp_dir / "results")
    plan, md, k = fix(pilot, steps, s1, rules, existing_amendments(exp_dir), context=ctx)
    pf, am = write_outputs(exp_dir, plan, md, k, ctx["now_utc"])
    print(f"plan_fixed: {pf.relative_to(exp_dir)}")
    print(f"amendment:  {am.relative_to(exp_dir)}")
    print(json.dumps({"status": plan["status"], "plan": plan.get("plan"), "n_M": plan.get("n_M"),
                      "tier_b": plan.get("tier_b"), "main_h": plan.get("main_h")}, sort_keys=True))
    return 0 if plan["status"] == "FIXED" else 2


if __name__ == "__main__":
    sys.exit(main())
