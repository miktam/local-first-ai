"""exp_036 runner CLI: pilot | session | cell | status (BUILD_SPEC §5.4
runner/run.py; HYPOTHESIS "Pilot …" rules 5-6; RUNBOOK steps 12, 14, 16).

  run.py pilot [--cells ARM:TASK,…]   the pilot (never scored for accuracy)
  run.py session --name S2|S3|S3b     the frozen queue from the plan amendment
  run.py cell --arm A --task T [--effort E] [--pass P] [--session S]
  run.py status                       progress, truncation and parse counters, ETA

Every subcommand calls the guards first (runner/guard.py). Long jobs are run
by Andrei under `caffeinate -i`; Ctrl-C stops cleanly after the records
already written (live sequences are discarded and re-run on resume); a second
Ctrl-C aborts at once. Rerunning the same command resumes from the missing
keys.

Layout (BUILD_SPEC §5.4, HYPOTHESIS "Evidence layout"):
  results/raw/<session>/<arm>/<task>_<effort>.jsonl       repo records (hashes for withheld sets)
  results/raw/<session>/<arm>/<task>_<effort>.steps.jsonl per-step decode log
  results/raw/<session>/.heartbeat                        cell, B, last record time, state
  results/raw/<session>/session.jsonl                     start / stop events (machine time)
  results/raw/<session>/not_run.jsonl                     NOT RUN cells with the reason
  $EXP036_PRIVATE/raw/<session>/<arm>/<task>_<effort>.jsonl full withheld records
  evidence/withheld_manifest.jsonl                        sha256 of every private file, appended
A cell resumed in a later session keeps appending to the file of the session
where it started, with a "resume" header.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from runner import chat, guard, jsonl, memory, plan_fix, seeds  # noqa: E402
from runner.common import (  # noqa: E402
    ARMS,
    EXP_DIR,
    KOLIBRI_ARMS,
    arm_dir,
    cell_file_stem,
    cell_id,
    env_record,
    is_withheld,
    load_rules,
    parse_utc,
    private_dir,
    redact_path,
    sha256_file,
    task_kind,
    utc_iso,
    utc_now,
    utc_stamp,
    write_new_json,
    write_new_text,
)
from runner.generate import read_heartbeat, run_cell, write_heartbeat  # noqa: E402

SESSIONS = ("S2", "S3", "S3b")

# Task key -> tasks/build_manifests.py set name (the item source of a cell).
MANIFEST_SET = {
    "gpqa_en": "gpqa_diamond_en", "gpqa_de": "gpqa_diamond_de",
    "mmlu_en": "mmlu_prox_lite_en", "mmlu_de": "mmlu_prox_lite_de",
    "aime_en": "aime_en", "aime_de": "aime_de", "ifbench": "ifbench",
    "rgb_cb": "rgb_cb", "rgb_forced": "rgb_forced", "rgb_neg": "rgb_negative", "rgb_fact": "rgb_fact",
    # pilot cells
    "gpqa_main_en": "gpqa_pilot_en", "gpqa_main_de": "gpqa_pilot_de",
    "mmlu_full_en": "mmlu_prox_full_pilot_en", "mmlu_full_de": "mmlu_prox_full_pilot_de",
    "ifbench_pilot": "ifbench_pilot", "aime_en_pilot": "aime_pilot_en",
    "rgb_int_cb": "rgb_pilot_cb", "rgb_int_forced": "rgb_pilot_forced",
}
# Pilot task -> the scored task whose split and extractor apply (pilot rule 3).
PILOT_TASK = {"gpqa_main_en": "gpqa_en", "gpqa_main_de": "gpqa_de", "mmlu_full_en": "mmlu_en",
              "mmlu_full_de": "mmlu_de", "ifbench_pilot": "ifbench", "rgb_int_cb": "rgb_cb",
              "rgb_int_forced": "rgb_forced", "aime_en_pilot": "aime_en"}


# ================================================================== state


class Stop:
    """SIGINT: first press stops after the current records, second aborts."""

    def __init__(self):
        self.flag = False

    def __call__(self) -> bool:
        return self.flag

    def install(self):
        def handler(signum, frame):
            if self.flag:
                raise KeyboardInterrupt
            self.flag = True
            print("\nStopping after the records already written (Ctrl-C again to abort now) …", file=sys.stderr)

        signal.signal(signal.SIGINT, handler)
        return self


class Ctx:
    def __init__(self, exp_dir: Path | None = None):
        self.exp = Path(exp_dir or EXP_DIR)
        self.results = self.exp / "results"
        self.raw = self.results / "raw"
        self.aborted = self.exp / "aborted"
        self.evidence = self.exp / "evidence"
        self.private = private_dir()
        self.rules = load_rules()  # the frozen runner/plan_rules.json next to this code


# ================================================================== items


def items_for_cell(task: str, n: int | None) -> list[dict]:
    """The cell's items in manifest order (tasks/build_manifests.py:items_for
    re-renders them from local data and checks every item and prompt sha256
    against the manifest). MMLU-ProX-Lite manifests list ids in the Webster
    seat order (Amendment 4: per-category n proportional to Lite's own
    category counts, nested), so the first n_M are the n_M set."""
    try:
        from tasks import build_manifests
    except ModuleNotFoundError as e:
        guard.refuse(f"tasks/build_manifests.py is not importable ({e.name})")
    items = build_manifests.items_for(MANIFEST_SET[task])
    if n is not None:
        if len(items) < n:
            guard.refuse(f"{task}: manifest has {len(items)} items, the plan needs {n}")
        items = items[:n]
    return [{"id": it.id, "item_sha256": it.item_sha256, "prompt_sha256": it.prompt_sha256,
             "messages": it.messages} for it in items]


# ============================================================ model loading


class Loaded:
    def __init__(self):
        self.arm = None
        self.model = None
        self.tokenizer = None
        self.meta: dict = {}

    def get(self, arm: str):
        if self.arm == arm:
            return self.model, self.tokenizer, self.meta
        self.unload()
        import mlx_lm

        d = arm_dir(arm)
        if not d.is_dir():
            guard.refuse(f"{arm}: {redact_path(d)} does not exist")
        family = chat.family_of(arm)
        meta: dict[str, Any] = {"model_dir": redact_path(d), "family": family}
        if arm in KOLIBRI_ARMS:
            try:
                from port import convert

                meta["port_sha256"] = convert.check_port_file(d)
            except ModuleNotFoundError:
                guard.refuse("port/convert.py is not importable: cannot check the converted port file")
            except Exception as e:
                guard.refuse(f"{arm}: stale or missing port file in the converted directory: {e}")
            rec = d / "exp036_convert_record.json"
            if rec.is_file():
                meta["model_manifest_sha256"] = json.loads(rec.read_text(encoding="utf-8")).get("manifest_sha256")
        samp = chat.check_sampling(family, d)
        if not samp["match"]:
            guard.refuse(f"{arm}: generation_config sampling {samp['generation_config']} != frozen {samp['frozen']}")
        meta["template_parity"] = chat.template_parity(family, d)
        meta["eos_ids"] = chat.eos_ids(d)
        model, tok = mlx_lm.load(str(d))
        self.arm, self.model, self.tokenizer, self.meta = arm, model, tok, meta
        return model, tok, meta

    def unload(self):
        if self.model is not None:
            import mlx.core as mx

            self.model = self.tokenizer = None
            self.arm = None
            gc.collect()
            mx.clear_cache()


# ================================================================== headers


def cell_header(ctx: Ctx, cell: dict, B: int, seed: int, meta: dict, plan: dict | None, extra: dict) -> dict:
    fam = chat.family_of(cell["arm"])
    h = env_record(ctx.exp)
    h.update({
        "cell": cell_id(cell["arm"], cell["task"], cell["effort"], cell.get("pass", 0)),
        "arm": cell["arm"], "task": cell["task"], "effort": cell["effort"], "pass": cell.get("pass", 0),
        "family": fam, "cap": cell["cap"], "B": B, "seed": seed, "n": cell.get("n"),
        "sampler": chat.sampling_for(fam),
        "template_kwargs": chat.template_kwargs(fam, cell["effort"]),
        "template_sha256": chat.template_sha256(fam),
        "port_sha256": meta.get("port_sha256"),
        "model_manifest_sha256": meta.get("model_manifest_sha256"),
        "model_dir": meta.get("model_dir"),
        "template_parity": meta.get("template_parity"),
        "eos_ids": meta.get("eos_ids"),
        "item_manifest_sha256": (plan or {}).get("manifests", {}).get(
            f"tasks/manifests/{MANIFEST_SET.get(cell['task'], '')}.json"),
        "plan_amendment": (plan or {}).get("_amendment_k"),
        "plan_sha256": (plan or {}).get("_plan_sha256"),
        "gate_record_sha256": ((plan or {}).get("gate_record") or {}).get("sha256"),
        "b_fallback_from": None,
    })
    try:
        from tasks import assets

        a = assets.asset(ARMS[cell["arm"]]["dir"])
        h["asset_revision"] = a.revision
    except Exception:
        h["asset_revision"] = None
    h.update(extra)
    return h


# ============================================================ cell running


def cell_paths(ctx: Ctx, session: str, cell: dict) -> tuple[Path, Path, Path]:
    """(repo file, private file, steps file). An existing file in any session
    (the session where the cell started) wins."""
    stem = cell_file_stem(cell["task"], cell["effort"], cell.get("pass", 0))
    existing = jsonl.cell_files(cell["arm"], cell["task"], cell["effort"], cell.get("pass", 0), root=ctx.raw)
    home = existing[0].parent.parent.name if existing else session
    repo = ctx.raw / home / cell["arm"] / f"{stem}.jsonl"
    priv = ctx.private / "raw" / home / cell["arm"] / f"{stem}.jsonl"
    steps = ctx.raw / home / cell["arm"] / f"{stem}.steps.jsonl"
    return repo, priv, steps


def next_lower_B(B: int, choices: list[int]) -> int:
    lower = [b for b in sorted(choices, reverse=True) if b < B]
    return lower[0] if lower else 0


def execute_cell(ctx: Ctx, session: str, cell: dict, loaded: Loaded, plan: dict | None, stop: Stop,
                 fallback: dict | None = None) -> dict:
    """Run (or resume) one queued cell to completion, or until Ctrl-C."""
    arm, task, effort, pass_ = cell["arm"], cell["task"], cell["effort"], cell.get("pass", 0)
    cid = cell_id(arm, task, effort, pass_)
    repo, priv, steps = cell_paths(ctx, session, cell)
    withheld = is_withheld(task)
    last = jsonl.last_header(repo)
    B = int(last["B"]) if last else int(cell["B"])
    # After a crash fallback every later header and record of the cell keeps
    # b_fallback_from; crash_fallback marks the header that made the step down.
    extra = {"b_fallback_from": (last or {}).get("b_fallback_from"), "crash_fallback": False}
    if fallback:
        extra = {"b_fallback_from": fallback["from"], "crash_fallback": True}
        B = fallback["to"]
    items = items_for_cell(task, cell["n"])
    done = jsonl.completed_keys(arm, task, effort, pass_, root=ctx.raw)
    if all((str(i["id"]), int(pass_)) in done for i in items):
        return {"cell": cid, "complete": True, "n_done": 0}
    model, tok, meta = loaded.get(arm)
    seed = seeds.cell_seed(arm, task, effort, pass_)
    header = cell_header(ctx, cell, B, seed, meta, plan, extra)
    header["session"] = session
    w = jsonl.open_run(repo, header, aborted_root=ctx.aborted, private_root=ctx.private)
    pw = jsonl.open_run(priv, header, aborted_root=ctx.aborted, private_root=ctx.private) if withheld else None
    try:
        summ = run_cell(
            model, tok, items, arm, task, effort, int(cell["cap"]), B, seed, w, chat.family_of(arm),
            pass_=pass_, eos=meta["eos_ids"], private_writer=pw, steps_path=steps,
            heartbeat_path=ctx.raw / session / ".heartbeat",
            heartbeat_extra={"session": session, "home": repo.parent.parent.name},
            b_fallback_from=extra.get("b_fallback_from"), done=done, stop=stop,
        )
    finally:
        w.close()
        if pw is not None:
            pw.close()
    done = jsonl.completed_keys(arm, task, effort, pass_, root=ctx.raw)
    summ["complete"] = all((str(i["id"]), int(pass_)) in done for i in items)
    return summ


# ============================================================ crash fallback


def detect_crash(ctx: Ctx, rules: dict) -> dict | None:
    """A heartbeat in state "running" whose process is gone or which is older
    than heartbeat_stale_min means the previous run crashed in that cell."""
    stale_s = 60.0 * float(rules["crash_fallback"]["heartbeat_stale_min"])
    for hb_path in sorted(ctx.raw.glob("*/.heartbeat")):
        hb = read_heartbeat(hb_path)
        if not hb or hb.get("state") not in ("running", "failed") or "cell" not in hb:
            continue
        if hb["state"] == "failed":  # the previous run ended on an exception inside the cell
            hb["_path"] = str(hb_path)
            return hb
        age = (utc_now() - parse_utc(hb["utc"])).total_seconds() if hb.get("utc") else stale_s + 1
        alive = _pid_alive(int(hb.get("pid", -1)))
        if alive and age <= stale_s:
            guard.refuse(f"another run (pid {hb.get('pid')}) is active on {hb.get('cell')} ({redact_path(hb_path)})")
        hb["_path"] = str(hb_path)
        return hb
    return None


def _pid_alive(pid: int) -> bool:
    if pid <= 0 or pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def plan_fallback(ctx: Ctx, rules: dict, hb: dict) -> dict:
    """{cell, from, to} for the crashed cell, or {cell, abort: reason}."""
    arm, task, effort, pass_ = hb["arm"], hb["task"], hb["effort"], int(hb.get("pass", 0))
    files = jsonl.cell_files(arm, task, effort, pass_, root=ctx.raw)
    used = jsonl.fallback_count(files[0]) if files else 0
    B = int(hb["B"])
    to = next_lower_B(B, rules["B_choices"])
    cid = cell_id(arm, task, effort, pass_)
    if used >= int(rules["crash_fallback"]["max_fallbacks"]):
        return {"cell": cid, "abort": f"crashed again after {used} fallbacks"}
    if to == 0:
        return {"cell": cid, "abort": f"crashed at B = {B}; no lower B"}
    return {"cell": cid, "from": B, "to": to}


def abort_cell(ctx: Ctx, session: str, cell: dict, reason: str) -> Path:
    """Move a cell's files to aborted/ (private ones to $EXP036_PRIVATE/aborted/)
    with a NOTE.md, and record it NOT RUN. Nothing is deleted."""
    arm, task, effort, pass_ = cell["arm"], cell["task"], cell["effort"], cell.get("pass", 0)
    cid = cell_id(arm, task, effort, pass_)
    stamp = utc_stamp()
    tag = f"{stamp}-{arm}-{cell_file_stem(task, effort, pass_)}"
    dest = ctx.aborted / tag
    dest.mkdir(parents=True, exist_ok=False)
    lines = [f"# Cell {cid} aborted ({stamp})", "", f"- Reason: {reason}", "- Its hypotheses are NOT RUN (p = 1).", ""]
    stem = cell_file_stem(task, effort, pass_)
    for p in sorted(ctx.raw.glob(f"*/{arm}/{stem}.jsonl")) + sorted(ctx.raw.glob(f"*/{arm}/{stem}.steps.jsonl")):
        target = dest / f"{p.parent.parent.name}-{p.name}"
        shutil.move(str(p), str(target))
        lines.append(f"- Moved `{redact_path(p)}` here as `{target.name}`")
    moves = []
    for p in sorted((ctx.private / "raw").glob(f"*/{arm}/{stem}.jsonl")):
        pdest = ctx.private / "aborted" / tag
        pdest.mkdir(parents=True, exist_ok=True)
        size, digest = p.stat().st_size, sha256_file(p)
        target = pdest / f"{p.parent.parent.name}-{p.name}"
        shutil.move(str(p), str(target))
        moves.append((p, target, digest, size))
        lines.append(f"- Private file moved to `{redact_path(pdest)}`: {size} bytes, sha256 `{digest}`")
    write_new_text(dest / "NOTE.md", "\n".join(lines) + "\n")
    record_private_moves(ctx, session, moves)
    record_not_run(ctx, session, cell, reason)
    return dest


def record_private_moves(ctx: Ctx, session: str, moves: list[tuple]) -> None:
    """evidence/withheld_manifest.jsonl: a private file moved to $EXP036_PRIVATE/aborted/ keeps its sha256 under
    its new path, so the mirror check (tools/status.py --verify-private, RUNBOOK step 18) finds it (review fix
    2026-10-03)."""
    if not moves:
        return
    man = ctx.evidence / "withheld_manifest.jsonl"
    man.parent.mkdir(parents=True, exist_ok=True)
    w = jsonl.Writer(man)
    try:
        for old, new, digest, size in moves:
            w.append({"type": "private_moved", "path": redact_path(old), "to": redact_path(new), "sha256": digest,
                      "bytes": size, "session": session, "utc": utc_iso()})
    finally:
        w.close()


def record_not_run(ctx: Ctx, session: str, cell: dict, reason: str) -> None:
    p = ctx.raw / session / "not_run.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    w = jsonl.Writer(p)
    # The queue's own cell id when known ("B4:ladder" for the bench cell; review fix 2026-10-03), so that
    # outstanding() and status find the record.
    cid = cell.get("cell") or cell_id(cell["arm"], cell["task"], cell["effort"], cell.get("pass", 0))
    w.append({"type": "not_run", "cell": cid,
              "tier": cell.get("tier"), "reason": reason, "utc": utc_iso(), "session": session})
    w.close()


def not_run_cells(ctx: Ctx) -> dict[str, dict]:
    out = {}
    for p in sorted(ctx.raw.glob("*/not_run.jsonl")):
        for r in jsonl.read_jsonl(p):
            out.setdefault(r["cell"], r)
    return out


# ============================================================ session time


def session_hours(ctx: Ctx, session: str) -> float:
    """Machine hours of a session: start/stop pairs; a start without a stop
    (a crash) ends at the session's last heartbeat."""
    p = ctx.raw / session / "session.jsonl"
    if not p.is_file():
        return 0.0
    total, start = 0.0, None
    for r in jsonl.read_jsonl(p):
        if r.get("type") == "start":
            if start is not None:
                total += _crash_end(ctx, session, start)
            start = parse_utc(r["utc"])
        elif r.get("type") == "stop" and start is not None:
            total += (parse_utc(r["utc"]) - start).total_seconds()
            start = None
    if start is not None:
        total += (utc_now() - start).total_seconds()  # the running session
    return total / 3600.0


def _crash_end(ctx: Ctx, session: str, start) -> float:
    hb = read_heartbeat(ctx.raw / session / ".heartbeat")
    if hb and hb.get("utc"):
        end = parse_utc(hb["utc"])
        if end > start:
            return (end - start).total_seconds()
    return 0.0


def session_event(ctx: Ctx, session: str, kind: str, **kw) -> None:
    p = ctx.raw / session / "session.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    w = jsonl.Writer(p)
    w.append({"type": kind, "utc": utc_iso(), "pid": os.getpid(), **kw})
    w.close()


# ============================================================ evidence


def update_withheld_manifest(ctx: Ctx, session: str) -> int:
    """Append sha256 + bytes of every private raw file whose content changed."""
    man = ctx.evidence / "withheld_manifest.jsonl"
    last: dict[str, str | None] = {}
    if man.is_file():
        for r in jsonl.read_jsonl(man):
            # A moved file no longer lives at its old path: a new file there is recorded again.
            last[r["path"]] = None if r.get("type") == "private_moved" else r["sha256"]
    new = 0
    files = sorted((ctx.private / "raw").glob("*/*/*.jsonl")) + sorted((ctx.private / "pilot").glob("*/*/*.jsonl"))
    if not files:
        return 0
    man.parent.mkdir(parents=True, exist_ok=True)
    w = jsonl.Writer(man)
    try:
        for p in files:
            rp = redact_path(p)
            d = sha256_file(p)
            if last.get(rp) != d:
                w.append({"type": "private_file", "path": rp, "sha256": d, "bytes": p.stat().st_size,
                          "session": session, "utc": utc_iso()})
                new += 1
    finally:
        w.close()
    return new


# ============================================================ subcommands


def _cell_from_queue(q: dict) -> dict:
    return {k: q[k] for k in ("cell", "arm", "task", "effort", "pass", "n", "cap", "B", "tier", "projected_h") if k in q} | (
        {"needs": q["needs"]} if "needs" in q else {})


def cmd_session(args, ctx: Ctx) -> int:
    session = args.name
    rules = ctx.rules
    guard.require_identity(ctx.exp)
    guard.require_clean_tree(exp_dir=ctx.exp)
    guard.require_signoff(ctx.exp)
    plan = guard.require_plan(ctx.exp)
    queue = [q for q in plan["queue"]]
    nr = not_run_cells(ctx)

    def outstanding(q):
        if q["cell"] in nr:
            return False
        if q["arm"] == "bench":
            return not sorted(ctx.results.glob("bench/ladder_*.jsonl"))
        done = jsonl.completed_keys(q["arm"], q["task"], q["effort"], q.get("pass", 0), root=ctx.raw)
        return len(done) < int(q["n"])

    if any(q["arm"] == "K8" and outstanding(q) for q in queue):
        guard.require_metal_limit(plan)

    stop = Stop().install()
    hb = detect_crash(ctx, rules)
    pending_fb: dict[str, dict] = {}
    crashed_q = next((q for q in queue if hb is not None and q["cell"] == hb.get("cell")), None)
    if hb is not None and (crashed_q is None or not outstanding(crashed_q)):
        # The heartbeat names a finished cell (the crash came between cells) or
        # a cell outside this queue: nothing to fall back.
        write_heartbeat(Path(hb["_path"]), {**{k: v for k, v in hb.items() if k != "_path"},
                                            "state": "crashed", "seen_utc": utc_iso(), "fallback": "none needed"})
        hb = None
    if hb is not None:
        fb = plan_fallback(ctx, rules, hb)
        print(f"Crash detected in {fb['cell']} ({redact_path(hb['_path'])}): {fb}", file=sys.stderr)
        write_heartbeat(Path(hb["_path"]), {**{k: v for k, v in hb.items() if k != "_path"},
                                            "state": "crashed", "seen_utc": utc_iso()})
        if "abort" in fb:
            q = next((q for q in queue if q["cell"] == fb["cell"]), None)
            if q is not None:
                abort_cell(ctx, session, _cell_from_queue(q), fb["abort"])
                nr = not_run_cells(ctx)
        else:
            pending_fb[fb["cell"]] = fb

    session_event(ctx, session, "start", plan_amendment=plan.get("_amendment_k"))
    S1 = float(plan.get("S1_hours") or 0.0)
    cap_h = float(rules["budget"]["session_cap_h"])
    ceiling = float(rules["budget"]["overrun_ceiling_h"])
    loaded = Loaded()
    checked: set[str] = set()
    tier2_skipped = False
    reason = "queue complete"
    state = "stopped"
    in_cell = {"now": False}
    try:
        for q in queue:
            if stop():
                reason = "SIGINT"
                break
            if not outstanding(q):
                continue
            cell = _cell_from_queue(q)
            started = bool(jsonl.cell_files(q["arm"], q["task"], q["effort"], q.get("pass", 0), root=ctx.raw)) \
                if q["arm"] != "bench" else False
            if not started:
                proj = float(q.get("projected_h") or 0.0)
                if session == "S3b":
                    if q["tier"] != "A":
                        continue
                    run_total = S1 + sum(session_hours(ctx, s) for s in SESSIONS)
                    if run_total + proj > ceiling:
                        record_not_run(ctx, session, cell, f"S3b: projected run total {run_total + proj:.2f} h > "
                                                           f"overrun ceiling {ceiling} h")
                        continue
                elif session_hours(ctx, session) + proj > cap_h:
                    reason = f"next cell {q['cell']} would take {session} past {cap_h} h"
                    print(f"{session}: stops admitting new cells; the next session starts at {q['cell']}.")
                    break
            if q["arm"] == "bench":
                if not guard.tier2_at_head(ctx.exp):
                    # Review fix 2026-10-03: a missing Tier-2 amendment no longer ends the session. The ladder
                    # stays outstanding (a later session runs it once the amendment is pulled); the rest of the
                    # queue continues; at the end of S3 an unstarted B4 is NOT RUN with this reason.
                    tier2_skipped = True
                    print(f"{session}: {q['cell']} skipped: no Tier-2 analysis amendment at HEAD (pull, then rerun "
                          f"the session to run it); the queue continues.", file=sys.stderr)
                    continue
                rc = subprocess.run([sys.executable, str(ctx.exp / "bench" / "run_bench.py"), "--cells", "ladder"],
                                    cwd=ctx.exp).returncode
                if rc != 0:
                    record_not_run(ctx, session, cell, f"bench/run_bench.py --cells ladder exited {rc}")
                continue
            if q["arm"] not in checked:
                guard.require_gate(q["arm"], ctx.results)
                guard.require_peers(q["arm"], ctx.results)
                if q["arm"] not in KOLIBRI_ARMS:
                    guard.require_assets(ARMS[q["arm"]]["dir"])
                checked.add(q["arm"])
            in_cell["now"] = True
            summ = execute_cell(ctx, session, cell, loaded, plan, stop, pending_fb.pop(q["cell"], None))
            in_cell["now"] = False
            update_withheld_manifest(ctx, session)
            print(json.dumps({"cell": q["cell"], **{k: summ.get(k) for k in ("n_done", "n_truncated", "complete", "stopped")}},
                             sort_keys=True))
            if summ.get("stopped"):
                reason = "SIGINT"
                break
    except KeyboardInterrupt:
        reason = "SIGINT (aborted)"
        raise
    except BaseException as e:
        # An exception while a cell generates (e.g. a Metal out-of-memory
        # error) is a crash: the heartbeat keeps the cell and B, and the next
        # start applies the crash fallback to it. A refusal is not a crash.
        reason = f"exception: {type(e).__name__}: {e}"
        if in_cell["now"] and not isinstance(e, SystemExit):
            state = "failed"
        raise
    finally:
        loaded.unload()
        hbp = ctx.raw / session / ".heartbeat"
        hb_now = read_heartbeat(hbp) or {}
        write_heartbeat(hbp, {**hb_now, "state": state, "utc": utc_iso(), "reason": reason, "pid": os.getpid()})
        session_event(ctx, session, "stop", reason=reason)
        update_withheld_manifest(ctx, session)

    if session == "S3" and reason != "SIGINT":
        # End of S3: Tier-B cells not started are NOT RUN; outstanding Tier-A cells go to S3b.
        nr = not_run_cells(ctx)
        for q in queue:
            if q["tier"] == "B" and q["cell"] not in nr and outstanding(q) and (
                    q["arm"] == "bench" or not jsonl.cell_files(q["arm"], q["task"], q["effort"], q.get("pass", 0),
                                                                root=ctx.raw)):
                why = "Tier B not started by the end of S3"
                if q["arm"] == "bench" and tier2_skipped:
                    why += " (no Tier-2 analysis amendment at HEAD)"
                record_not_run(ctx, session, _cell_from_queue(q), why)
        if any(q["tier"] == "A" and outstanding(q) for q in queue):
            print("Tier-A outstanding — run S3b now")
    return 0


def cmd_cell(args, ctx: Ctx) -> int:
    guard.require_identity(ctx.exp)
    guard.require_clean_tree(exp_dir=ctx.exp)
    guard.require_signoff(ctx.exp)
    plan = guard.require_plan(ctx.exp)
    cid = cell_id(args.arm, args.task, args.effort, args.pass_)
    q = next((q for q in plan["queue"] if q["cell"] == cid), None)
    if q is None:
        guard.refuse(f"{cid} is not in the plan's queue")
    if args.arm == "K8":
        guard.require_metal_limit(plan)
    guard.require_gate(args.arm, ctx.results)
    guard.require_peers(args.arm, ctx.results)
    stop = Stop().install()
    session = args.session or next((s for s in reversed(SESSIONS) if (ctx.raw / s).is_dir()), "S2")
    session_event(ctx, session, "start", cell=cid)
    loaded = Loaded()
    try:
        summ = execute_cell(ctx, session, _cell_from_queue(q), loaded, plan, stop)
    finally:
        loaded.unload()
        session_event(ctx, session, "stop", cell=cid)
        update_withheld_manifest(ctx, session)
    print(json.dumps(summ, sort_keys=True, default=str))
    return 0


# ------------------------------------------------------------------ pilot


def _parse_view(task: str, rec: dict) -> dict | None:
    """scorers/score_all.py:parse_record (the scorers' one definition of a
    parse failure, gold-free) for a pilot record; None if that module is absent."""
    try:
        from scorers import score_all
    except ModuleNotFoundError:
        return None
    v = score_all.parse_record(PILOT_TASK.get(task, task), rec, rec.get("text", ""))
    v["failed"] = v["parse_status"] in score_all.PARSE_FAILURES
    return v


def _own_parse_view(task: str, rec: dict, fam: str, effort: str) -> dict:
    """Fallback when scorers/ is absent: an extractor miss (GPQA, MMLU, AIME)
    or an empty post-reasoning answer (IFBench, RGB)."""
    kind = task_kind(task)
    ans = rec.get("answer_text", "")
    if kind in ("gpqa", "mmlu"):
        from scorers import mc

        failed = getattr(mc, f"extract_{PILOT_TASK[task]}")(ans) is None
    elif kind == "aime":
        from scorers import aime

        failed = aime.extract(ans) is None
    else:
        failed = ans.strip() == ""
    st = rec["reasoning_status"]
    defect = st == "none" and fam in ("kolibri", "qwen3_6", "qwen3_8") and effort != "none"
    return {"reasoning_status": st, "failed": failed or st == "unclosed", "defect": defect}


def summarise_pilot_cell(records: list[dict], arm: str, task: str, effort: str, cap: int, B: int) -> dict:
    """Length, truncation, status and parse counts; no accuracy (rule 3).

    Truncated = finish_reason "length", or a completion that ends at its cap
    while still inside its reasoning segment (HYPOTHESIS, Pilot "Measured").
    A truncation is never a parse failure. Parse failures carry the frozen
    diagnose() code; status "none" for Kolibri / Qwen with thinking on is a
    defect, recorded, not a failure; for Gemma 4 it is legitimate."""
    fam = chat.family_of(arm)
    lengths, prompts, trunc, statuses = [], [], [], {}
    n_pf, diags, n_def = 0, {}, 0
    for r in records:
        L = int(r["completion_tokens"])
        view = _parse_view(task, r) or _own_parse_view(task, r, fam, effort)
        st = view["reasoning_status"]
        t = r["finish_reason"] == "length" or bool(r.get("truncated")) or (
            st == "unclosed" and L >= int(r.get("max_tokens", cap)))
        lengths.append(L)
        prompts.append(int(r["rendered_prompt_tokens"]))
        trunc.append(bool(t))
        statuses[st] = statuses.get(st, 0) + 1
        n_def += int(bool(view["defect"]))
        if t or not view["failed"]:
            continue
        n_pf += 1
        code = plan_fix.diagnose(task, r.get("answer_text", ""), r.get("text", ""), st, r["finish_reason"]) or "model"
        diags[code] = diags.get(code, 0) + 1
    import statistics

    mean = statistics.fmean(lengths) if lengths else 0.0
    se = statistics.stdev(lengths) / len(lengths) ** 0.5 if len(lengths) > 1 else 0.0
    return {"arm": arm, "task": task, "effort": effort, "n": len(records), "B": B, "cap": cap,
            "lengths": lengths, "prompt_tokens": prompts, "truncated": trunc, "n_truncated": sum(trunc),
            "status_counts": statuses, "n_parse_fail": n_pf, "diagnoses": diags, "n_defect": n_def,
            "len_mean": round(mean, 2), "len_se": round(se, 2), "len_max": max(lengths) if lengths else 0}


def cmd_pilot(args, ctx: Ctx) -> int:
    rules = ctx.rules
    guard.require_identity(ctx.exp)
    guard.require_clean_tree(exp_dir=ctx.exp)
    guard.require_signoff(ctx.exp)
    wanted = parse_pilot_cells(args.cells, rules) if args.cells else None
    without = pilot_without(args.without, ctx) if getattr(args, "without", None) else {}
    arms = [a for a in rules["pilot"]["arms"]
            if a not in without and (wanted is None or any(w[0] == a for w in wanted))]
    if wanted is not None and any(w[0] in without for w in wanted):
        guard.refuse(f"--cells names an arm that --without excludes: {sorted(w for w in wanted if w[0] in without)}")
    for a in arms:
        guard.require_gate(a, ctx.results)
        guard.require_peers(a, ctx.results)
    L, L_src = memory.effective_limit()
    stamp = utc_stamp()
    pdir = ctx.results / "pilot" / stamp
    ppriv = ctx.private / "pilot" / stamp
    stop = Stop().install()
    effort = rules["pilot"]["effort"]
    loaded = Loaded()
    summary = {"type": "pilot_summary", "utc": utc_iso(), "pilot_dir": pdir.relative_to(ctx.exp).as_posix(),
               "rules_version": rules["version"], "caps": rules["caps"], "effort": effort,
               "L_bytes": L, "L_sources": L_src, "cells": [], "arms": {},
               # Review fix 2026-10-03: what was asked for, what was skipped by rule, which arms the records
               # exclude, and whether the pilot ran to the end (runner/plan_fix.py refuses an incomplete one).
               "requested": sorted([a, c["task"]] for a in arms for c in rules["pilot"]["cells"][a]
                                   if wanted is None or (a, c["task"]) in wanted),
               "repilot": wanted is not None, "skipped": [], "without": without, "complete": False}
    interrupted = False
    try:
        for arm in arms:
            t_arm0 = time.time()
            for c in rules["pilot"]["cells"][arm]:
                task = c["task"]
                if wanted is not None and (arm, task) not in wanted:
                    continue
                if stop():
                    interrupted = True
                    break
                cap = int(rules["caps"][task_kind(task)])
                model, tok, meta = loaded.get(arm)
                B = memory.choose_B(arm, task, cap, L, rules=rules)
                if B < 1:
                    print(f"{arm} {task}: does not fit at B = 1 (L = {L / 2**30:.2f} GiB); skipped", file=sys.stderr)
                    summary["skipped"].append({"arm": arm, "task": task,
                                               "reason": f"does not fit at B = 1 (L = {L} B)"})
                    continue
                items = items_for_cell(task, int(c["n"]))
                stem = cell_file_stem(task, effort)
                repo = pdir / arm / f"{stem}.jsonl"
                priv = ppriv / arm / f"{stem}.jsonl"
                steps = pdir / arm / f"{stem}.steps.jsonl"
                seed = seeds.cell_seed(arm, task, effort, 0)
                cell = {"arm": arm, "task": task, "effort": effort, "pass": 0, "n": int(c["n"]), "cap": cap}
                header = cell_header(ctx, cell, B, seed, meta, None, {"pilot": True})
                withheld = is_withheld(task)
                t0 = utc_iso()
                w = jsonl.open_run(repo, header, aborted_root=ctx.aborted, private_root=ctx.private)
                pw = jsonl.open_run(priv, header, aborted_root=ctx.aborted, private_root=ctx.private) if withheld else None
                try:
                    run_cell(model, tok, items, arm, task, effort, cap, B, seed, w, chat.family_of(arm),
                             eos=meta["eos_ids"], private_writer=pw, steps_path=steps,
                             heartbeat_path=pdir / ".heartbeat", heartbeat_extra={"session": "pilot"}, stop=stop)
                finally:
                    w.close()
                    if pw is not None:
                        pw.close()
                full = jsonl.first_records(jsonl.read_jsonl(priv if withheld else repo))
                if stop() or len(full) < int(c["n"]):
                    interrupted = True  # a cell cut short by Ctrl-C: the pilot is incomplete
                    break
                cs = summarise_pilot_cell(full, arm, task, effort, cap, B)
                cs.update({"t_start": t0, "t_end": utc_iso(), "steps_file": steps.relative_to(ctx.exp).as_posix(),
                           "file": repo.relative_to(ctx.exp).as_posix()})
                summary["cells"].append(cs)
            if interrupted:
                break
            ptok = psec = 0.0
            for sf in sorted((pdir / arm).glob("*.steps.jsonl")):
                for r in jsonl.read_jsonl(sf):
                    if r.get("type") == "step":
                        ptok += r.get("prompt_tokens", 0)
                        psec += r.get("prompt_seconds", 0.0)
            summary["arms"][arm] = {"t_start": t_arm0, "t_end": time.time(), "prefill_tokens": ptok,
                                    "prefill_seconds": psec}
            loaded.unload()
    finally:
        loaded.unload()
    if interrupted or stop():
        # Review fix 2026-10-03: no summary for an interrupted pilot (plan_fix must never see a partial one).
        update_withheld_manifest(ctx, "pilot")
        print(f"pilot interrupted: no summary written. Move the partial pilot to aborted/ with "
              f"runner/run.py abort-pilot --stamp {stamp}, then rerun the pilot in full (RUNBOOK step 12).",
              file=sys.stderr)
        return 1
    summary["complete"] = True
    summary["t_start"] = summary["utc"]
    summary["t_end"] = utc_iso()
    out = ctx.results / f"pilot_summary_{stamp}.json"
    write_new_json(out, summary)
    update_withheld_manifest(ctx, "pilot")
    print(f"pilot summary: {out.relative_to(ctx.exp)}")
    return 0


def parse_pilot_cells(spec: str, rules: dict) -> set[tuple[str, str]]:
    """--cells for a re-pilot: ARM:TASK, or the ARM:TASK:EFFORT cell id that plan_fix prints for a BLOCKED cell
    (review fix 2026-10-03). Every cell must be a pilot cell of the frozen rules: an unknown one is refused, so a
    re-pilot can never run nothing and still write a summary."""
    effort = rules["pilot"]["effort"]
    known = {(a, c["task"]) for a, cs in rules["pilot"]["cells"].items() for c in cs}
    out = set()
    for x in (y.strip() for y in spec.split(",")):
        if not x:
            continue
        parts = x.split(":")
        if len(parts) not in (2, 3) or (len(parts) == 3 and parts[2] != effort):
            guard.refuse(f"--cells {x!r}: use ARM:TASK or ARM:TASK:{effort}")
        if (parts[0], parts[1]) not in known:
            guard.refuse(f"--cells {x!r} is not a pilot cell of runner/plan_rules.json")
        out.add((parts[0], parts[1]))
    if not out:
        guard.refuse("--cells is empty")
    return out


def pilot_without(spec: str, ctx: Ctx) -> dict[str, str]:
    """--without ARM[,ARM]: arms the records exclude (a gate K4 FAIL; a peer-check "fail"). Anything else is
    refused, so the pilot never drops an arm on the operator's word alone (review fix 2026-10-03)."""
    ex = guard.excluded_arms(ctx.results)
    out = {}
    for a in (y.strip() for y in spec.split(",")):
        if not a:
            continue
        if a not in ex:
            guard.refuse(f"--without {a}: the records do not exclude {a} (only a gate K4 FAIL with K8 PASS, or a "
                         f"peer-check verdict 'fail', does)")
        out[a] = ex[a]
    return out


def cmd_abort_pilot(args, ctx: Ctx) -> int:
    """Move an interrupted or crashed pilot (no summary) to aborted/<UTC>-pilot/ with a NOTE.md; its private files to
    $EXP036_PRIVATE/aborted/<UTC>-pilot/, with only their sha256 and byte counts in the note (RUNBOOK step 12;
    HYPOTHESIS "What counts as evidence": nothing is deleted; review fix 2026-10-03)."""
    import re as _re

    guard.require_identity(ctx.exp)
    stamp = args.stamp
    if not _re.fullmatch(r"\d{8}T\d{6}Z", stamp or ""):
        guard.refuse(f"--stamp {stamp!r} is not a <UTC> stamp like 20261004T120000Z")
    pdir = ctx.results / "pilot" / stamp
    ppriv = ctx.private / "pilot" / stamp
    if (ctx.results / f"pilot_summary_{stamp}.json").exists():
        guard.refuse(f"pilot {stamp} has a summary: it is complete and is never aborted")
    if not pdir.is_dir() and not ppriv.is_dir():
        guard.refuse(f"no pilot {stamp} in results/pilot/ or $EXP036_PRIVATE/pilot/")
    tag = f"{stamp}-pilot"
    dest = ctx.aborted / tag
    dest.mkdir(parents=True, exist_ok=False)
    lines = [f"# Pilot {stamp} aborted ({utc_stamp()})", "",
             "- Reason: interrupted or crashed before its summary was written; rerun in full (RUNBOOK step 12).", ""]
    if pdir.is_dir():
        files = sorted(f for f in pdir.rglob("*") if f.is_file())
        shutil.move(str(pdir), str(dest / "pilot"))
        lines += [f"- Moved `results/pilot/{stamp}/{f.relative_to(pdir).as_posix()}` here" for f in files]
    moves = []
    if ppriv.is_dir():
        pdest = ctx.private / "aborted" / tag
        pdest.mkdir(parents=True, exist_ok=False)
        for f in sorted(x for x in ppriv.rglob("*") if x.is_file()):
            target = pdest / f.relative_to(ppriv)
            target.parent.mkdir(parents=True, exist_ok=True)
            size, digest = f.stat().st_size, sha256_file(f)
            shutil.move(str(f), str(target))
            moves.append((f, target, digest, size))
            lines.append(f"- Private file moved to `{redact_path(target)}`: {size} bytes, sha256 `{digest}`")
    write_new_text(dest / "NOTE.md", "\n".join(lines) + "\n")
    record_private_moves(ctx, "pilot", moves)
    print(f"aborted: {dest.relative_to(ctx.exp)}")
    return 0


# ------------------------------------------------------------------ status


def cmd_status(args, ctx: Ctx) -> int:
    plan = None
    plans = sorted(ctx.results.glob("plan_fixed_*.json"))
    if plans:
        plan = json.loads(plans[-1].read_text(encoding="utf-8"))
    nr = not_run_cells(ctx)
    rows = []
    for q in (plan or {}).get("queue", []):
        if q["arm"] == "bench":
            rows.append({"cell": q["cell"], "n": 0, "done": int(bool(sorted(ctx.results.glob("bench/ladder_*.jsonl")))),
                         "not_run": nr.get(q["cell"], {}).get("reason")})
            continue
        recs = []
        for p in jsonl.cell_files(q["arm"], q["task"], q["effort"], q.get("pass", 0), root=ctx.raw):
            recs += jsonl.first_records(jsonl.read_jsonl(p))
            if is_withheld(q["task"]):  # the text lives in the private mirror (mbp only)
                pp = ctx.private / "raw" / p.relative_to(ctx.raw)
                full = {jsonl.record_key(r): r for r in jsonl.first_records(jsonl.read_jsonl(pp))} if pp.is_file() else {}
                recs = [full.get(jsonl.record_key(r), r) for r in recs]
        n_done = len({jsonl.record_key(r) for r in recs})
        st: dict[str, int] = {}
        n_pf = 0 if recs else None
        for r in recs:
            st[r["reasoning_status"]] = st.get(r["reasoning_status"], 0) + 1
            if "text" in r and not r["truncated"]:
                v = _parse_view(q["task"], r)
                if v is None:
                    n_pf = None
                elif n_pf is not None and v["failed"]:
                    n_pf += 1
        remaining = max(int(q["n"]) - n_done, 0)
        eta = (q.get("projected_h") or 0.0) * remaining / max(int(q["n"]), 1)
        rows.append({"cell": q["cell"], "tier": q["tier"], "session": q["session"], "n": q["n"], "done": n_done,
                     "truncated": sum(1 for r in recs if r["truncated"]), "status_counts": st, "parse_fail": n_pf,
                     "eta_h": round(eta, 2), "not_run": nr.get(q["cell"], {}).get("reason")})
    hbs = {p.parent.name: read_heartbeat(p) for p in sorted(ctx.raw.glob("*/.heartbeat"))}
    sess = {s: round(session_hours(ctx, s), 3) for s in SESSIONS if (ctx.raw / s).is_dir()}
    rec = {"type": "status", "utc": utc_iso(), "plan": (plan or {}).get("plan"), "cells": rows,
           "sessions_h": sess, "heartbeats": hbs,
           "eta_h": round(sum(r.get("eta_h", 0.0) for r in rows if not r.get("not_run")), 2)}
    for r in rows:
        flag = f"NOT RUN ({r['not_run']})" if r.get("not_run") else f"{r['done']}/{r['n']}"
        print(f"{r['cell']:32s} {flag:>14s}  trunc {r.get('truncated', 0):>4}  parse-fail {str(r.get('parse_fail')):>4}"
              f"  eta {r.get('eta_h', 0):>6} h")
    print(f"sessions: {sess}; remaining ≈ {rec['eta_h']} h")
    write_new_json(ctx.results / f"status_{utc_stamp()}.json", rec)
    return 0


def main(argv: list[str] | None = None) -> int:
    from tools.precision import ensure_exact_fp32
    ensure_exact_fp32()  # MLX_ENABLE_TF32=0 before any GPU work (tools/precision.py; Amendment 1)
    ap = argparse.ArgumentParser(description="exp_036 runner (BUILD_SPEC §5.4)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pilot")
    p.add_argument("--cells", help="ARM:TASK[:EFFORT],… (re-pilot only these cells)")
    p.add_argument("--without", help="ARM,… arms the records exclude (gate K4 FAIL; peer-check fail)")
    ap_abort = sub.add_parser("abort-pilot")
    ap_abort.add_argument("--stamp", required=True, help="the <UTC> of the interrupted results/pilot/<UTC>/")
    s = sub.add_parser("session")
    s.add_argument("--name", required=True, choices=SESSIONS)
    c = sub.add_parser("cell")
    c.add_argument("--arm", required=True, choices=sorted(ARMS))
    c.add_argument("--task", required=True)
    c.add_argument("--effort", default="high")
    c.add_argument("--pass", dest="pass_", type=int, default=0)
    c.add_argument("--session", choices=SESSIONS, help="session log to use (default: the latest session started)")
    sub.add_parser("status")
    args = ap.parse_args(argv)
    ctx = Ctx()
    return {"pilot": cmd_pilot, "session": cmd_session, "cell": cmd_cell, "status": cmd_status,
            "abort-pilot": cmd_abort_pilot}[args.cmd](args, ctx)


if __name__ == "__main__":
    sys.exit(main())
