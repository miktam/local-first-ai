"""Batched generation for one cell (BUILD_SPEC §4 items 26-28, §5.4
runner/generate.py; HYPOTHESIS "Arms", C15).

One item per sequence, through mlx_lm 0.32.0 BatchGenerator, constructed in
one place, make_batch_generator (exp_037 DESIGN §6.3; BUILD_SPEC item 27):

  BatchGenerator(fp32_logits(model), completion_batch_size=B,
                 prefill_batch_size=min(B, 8), prefill_step_size=2048,
                 max_kv_size=None, stop_tokens=[[eos] for eos in eos_ids],
                 sampler=<cell sampler>, max_tokens=<cap>)

run_cell, the gate's G5 checks and the bench all call it. Prompts admitted
together are prefilled together: right-padded, then finalised to left padding
(PromptProcessingBatch.prompt), as under 0.31.3. mlx_lm 0.32.0 has no
_make_cache any more: each sequence's cache comes from
models.cache.make_prompt_cache (the port's make_cache) and the batch caches
from each cache class's merge (KVCache -> BatchKVCache, the port's
SlidingKVCache -> BatchRotatingKVCache with max_size = sliding_window). No
logits processor is passed; the sampler is runner/sampler.py's own.

mlx_lm admits new sequences whenever the decode batch is below
completion_batch_size, without counting sequences still in prefill, so on its
own it can hold more than B live sequences. run_cell therefore owns the
slots: it inserts an item only while fewer than B items are in flight
(queued + prefilling + decoding), refilling a slot as soon as a sequence
finishes. Items go in manifest order; a cell never mixes tasks. Given the
gate's allowed_B (runner/run.py passes it for the Kolibri arms), run_cell
refuses any other B.

Each record is written (and fsync'ed) as soon as its sequence finishes.
Withheld sets write the full record to the private writer first and the
hashed record to the repo writer second (a crash in between re-runs the item;
the private record that counts is the one whose hashes the repo record
carries). Every decode step is logged (n_live, padded_len, step seconds,
prompt tokens and seconds of admissions in the same call, read from a
BatchGenerator.stats() window around that call) to the steps file;
the heartbeat is refreshed at every record and at least once a minute;
thermalState and the power mode are logged every 10 minutes.

No stop strings other than EOS; truncation means finish_reason == "length"
(the sequence reached its max_tokens). The RNG is global: mx.random.seed(seed)
once per cell run, so exact replay of a batched cell is not claimed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Iterable

from runner.common import (
    cell_id,
    ids_sha256,
    power_mode,
    sha256_text,
    thermal_state,
    utc_iso,
)

PREFILL_STEP_SIZE = 2048
CACHE_LIMIT_BYTES = 4 << 30
HEARTBEAT_EVERY_S = 60.0
ENV_EVERY_S = 600.0
STEPS_FLUSH_EVERY_S = 5.0

# Delimiters of the reasoning segment per family (item 28).
THINK_TOKENS = {
    "kolibri": ("<think>", "</think>"),
    "qwen3_6": ("<think>", "</think>"),
    "qwen3_8": ("<think>", "</think>"),
    "gemma4": ("<|channel>", "<channel|>"),
}


# ------------------------------------------------------------- small writers


class StepLog:
    """Buffered JSONL for the per-step decode log (flushed every few seconds,
    fsync'ed on close): a lost tail of a step log only loses timing samples."""

    def __init__(self, path: Path | None):
        self.path = Path(path) if path else None
        self._f = None
        self._last_flush = time.monotonic()
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._f = open(self.path, "a", encoding="utf-8")

    def append(self, rec: dict) -> None:
        if not self._f:
            return
        self._f.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        if time.monotonic() - self._last_flush > STEPS_FLUSH_EVERY_S:
            self._f.flush()
            self._last_flush = time.monotonic()

    def close(self) -> None:
        if self._f and not self._f.closed:
            self._f.flush()
            os.fsync(self._f.fileno())
            self._f.close()


def write_heartbeat(path: Path | None, state: dict) -> None:
    """Atomically replace the heartbeat file (a state file, not a result)."""
    if not path:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(state, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_heartbeat(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ----------------------------------------------------------- reasoning split


def special_ids(tokenizer, family: str) -> tuple[int | None, int | None]:
    from runner.chat import hf_tokenizer

    hf = hf_tokenizer(tokenizer)
    open_t, close_t = THINK_TOKENS[family]
    out = []
    for t in (open_t, close_t):
        i = hf.convert_tokens_to_ids(t)
        unk = getattr(hf, "unk_token_id", None)
        out.append(int(i) if i is not None and i != unk else None)
    return out[0], out[1]


def split_ids(
    family: str,
    prompt_prefill: str,
    completion_ids: list[int],
    open_id: int | None,
    close_id: int | None,
    eos: Iterable[int] = (),
    is_blank: Callable[[int], bool] | None = None,
) -> tuple[list[int], list[int], str]:
    """Runner-side split (item 28) -> (reasoning_ids, answer_ids, status).

    status: closed | unclosed | none. A trailing EOS id is not part of the
    reasoning or the answer. The scorers' split (scorers/reasoning.py) is the
    one scores use; this one fills the descriptive record fields when that
    module is absent."""
    ids = list(completion_ids)
    eos = set(int(e) for e in eos)
    while ids and ids[-1] in eos:
        ids.pop()
    if prompt_prefill in ("empty_think", "empty_channel"):
        return [], ids, "none"
    start = 0
    if prompt_prefill != "open_think":
        # The model may open the segment itself (Kolibri at effort high, whose
        # template does not prefill <think>; Gemma 4 with thinking on).
        j = 0
        while j < len(ids) and is_blank is not None and is_blank(ids[j]):
            j += 1
        if open_id is None or j >= len(ids) or ids[j] != open_id:
            return [], ids, "none"
        start = j + 1
    if close_id is not None and close_id in ids[start:]:
        k = ids.index(close_id, start)
        return ids[start:k], ids[k + 1:], "closed"
    return ids[start:], [], "unclosed"


def _scorer_split(family: str, tokenizer):
    """scorers/reasoning.py's id-level split (the one scoring uses), bound to
    this family and tokenizer, or None when that module is absent or its
    delimiter table does not match this tokenizer (e.g. a test tokenizer)."""
    try:
        from scorers import reasoning  # sibling area; lazy
    except ModuleNotFoundError:
        return None
    try:
        reasoning.check_tokenizer(family, tokenizer)
    except Exception:
        return None

    def split(prompt_ids, completion, eos, is_blank):
        state = reasoning.prompt_state_ids(family, prompt_ids)
        return reasoning.split_ids(family, state, completion, is_blank, eos)

    return split


# ------------------------------------------------------- the one construction


def make_batch_generator(model, B: int, eos: Iterable[int], max_tokens: int, sampler,
                         prefill_step_size: int = PREFILL_STEP_SIZE):
    """The kit's only BatchGenerator construction (exp_037 DESIGN §6.3; BUILD_SPEC item 27, decision F2):
    completion_batch_size = B, the registered prefill_batch_size = min(B, 8), prefill_step_size 2,048,
    max_kv_size None, the fp32-logit wrapper, EOS stop ids and the given sampler. No logits processor.

    run_cell, gate/checks/g5_generation.py, gate/checks/g5_runner.py and bench/genutil.py call it. mlx_lm 0.32.0
    keeps these arguments and still sets completion_batch_size = max(completion_batch_size, prefill_batch_size),
    which is B here. An empty `eos` means no stop ids (generation runs to max_tokens)."""
    from mlx_lm.generate import BatchGenerator

    from runner.sampler import fp32_logits

    B = int(B)
    if B < 1:
        raise ValueError("B must be >= 1")
    return BatchGenerator(
        fp32_logits(model),
        max_tokens=int(max_tokens),
        stop_tokens=[[int(e)] for e in eos],
        sampler=sampler,
        completion_batch_size=B,
        prefill_batch_size=min(B, 8),
        prefill_step_size=int(prefill_step_size),
        max_kv_size=None,
    )


# ------------------------------------------------------------------- run_cell


def _padded_len(gen) -> int:
    """Length of the batch KV of the decode batch (the longest live row)."""
    caches = getattr(gen._generation_batch, "prompt_cache", None) or []
    best = 0
    for c in caches:
        name = type(c).__name__
        if name == "BatchKVCache":
            return int(getattr(c, "_idx", 0))
        if name == "BatchRotatingKVCache":
            off = getattr(c, "_offset", 0)
            best = max(best, int(off) if isinstance(off, int) else 0)
    return best


def _decode(tokenizer, ids: list[int]) -> str:
    from runner.chat import hf_tokenizer

    return hf_tokenizer(tokenizer).decode(ids, skip_special_tokens=False)


def _hashed(rec: dict) -> dict:
    """Repo copy of a withheld-set record (BUILD_SPEC §5.4 record schema):
    completion ids and text replaced by their sha256. The reasoning and
    answer texts are dropped, not hashed: an answer is often a few words, and
    the hash of a short string can be reversed by guessing; the counts stay."""
    out = dict(rec)
    out["completion_ids_sha256"] = ids_sha256(out.pop("completion_ids"))
    out["text_sha256"] = sha256_text(out.pop("text"))
    out.pop("reasoning_text")
    out.pop("answer_text")
    return out


def run_cell(
    model,
    tokenizer,
    items: list[dict],
    arm: str,
    task: str,
    effort: str,
    cap: int,
    B: int,
    seed: int,
    writer,
    family: str,
    *,
    pass_: int = 0,
    sampler_cfg: dict | None = None,
    eos: list[int] | None = None,
    private_writer=None,
    steps_path: Path | None = None,
    heartbeat_path: Path | None = None,
    heartbeat_extra: dict | None = None,
    b_fallback_from: int | None = None,
    done: set | None = None,
    stop: Callable[[], bool] | None = None,
    prefill_step_size: int = PREFILL_STEP_SIZE,
    render_fn: Callable | None = None,
    on_step: Callable[[Any], None] | None = None,
    allowed_B: Iterable[int] | None = None,
) -> dict:
    """Generate every item of one cell not in `done`; returns a summary.

    items: dicts with "id", "item_sha256", "prompt_sha256" and either
    "prompt_ids" + "prompt_text" (already rendered) or "messages" (rendered
    here with runner.chat.render at `effort`); optional "max_tokens".
    writer / private_writer: runner.jsonl.Writer objects (private_writer set
    means a withheld set: the repo writer gets hashes only).
    allowed_B: the gate's allowed_B for a Kolibri arm (runner/run.py passes
    it; the gate's own calls pass none, because the gate sets it). A B outside
    it is refused before anything runs (DESIGN §6.3)."""
    import mlx.core as mx

    from runner import chat
    from runner.sampler import make_vllm_sampler

    if B < 1:
        raise ValueError("B must be >= 1")
    if allowed_B is not None and int(B) not in {int(b) for b in allowed_B}:
        raise ValueError(f"B = {B} is not in the gate's allowed_B {sorted(int(b) for b in allowed_B)}")
    cfg = dict(sampler_cfg or chat.sampling_for(family))
    sampler = make_vllm_sampler(cfg["temperature"], cfg["top_p"], cfg["top_k"])
    eos = [int(e) for e in (eos if eos is not None else sorted(getattr(tokenizer, "eos_token_ids", [])))]
    if not eos:
        raise ValueError("no EOS ids")
    open_id, close_id = special_ids(tokenizer, family)
    blank_cache: dict[int, bool] = {}

    def is_blank(t: int) -> bool:
        if t not in blank_cache:
            blank_cache[t] = _decode(tokenizer, [t]).strip() == ""
        return blank_cache[t]

    scorer_split = _scorer_split(family, tokenizer)
    done = set(done or ())
    render = render_fn or chat.render

    todo = []
    for it in items:
        if (str(it["id"]), int(pass_)) in done:
            continue
        if "prompt_ids" not in it:
            text, ids, rsha = render(family, tokenizer, it["messages"], effort)
            it = {**it, "prompt_text": text, "prompt_ids": ids, "rendered_sha256": rsha}
        elif "rendered_sha256" not in it:
            it = {**it, "rendered_sha256": sha256_text(it["prompt_text"])}
        todo.append(it)

    summary = {"cell": cell_id(arm, task, effort, pass_), "n_todo": len(todo), "n_done": 0,
               "n_truncated": 0, "status_counts": {}, "stopped": False, "B": B}
    if not todo:
        return summary

    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    mx.random.seed(int(seed))
    gen = make_batch_generator(model, B, eos, cap, sampler, prefill_step_size=prefill_step_size)
    steps = StepLog(steps_path)
    hb = {"state": "running", "pid": os.getpid(), "cell": summary["cell"], "arm": arm, "task": task,
          "effort": effort, "pass": pass_, "B": B, "seed": int(seed), **(heartbeat_extra or {})}
    t_hb = 0.0
    t_env = 0.0

    def env_line():
        steps.append({"type": "env", "t": round(time.time(), 3), "utc": utc_iso(),
                      "thermal_state": thermal_state(), "power_mode": power_mode()})

    meta: dict[int, dict] = {}
    out: dict[int, list[int]] = {}
    queue = list(todo)
    in_flight = 0
    batch_id = 0
    step_i = 0
    last_record_utc = None
    try:
        write_heartbeat(heartbeat_path, {**hb, "utc": utc_iso(), "last_record_utc": None, "n_done": 0})
        t_hb = time.monotonic()
        env_line()
        t_env = time.monotonic()
        while queue or in_flight:
            if stop is not None and stop():
                summary["stopped"] = True
                break
            free = B - in_flight
            if free > 0 and queue:
                group = queue[:free]
                del queue[:free]
                uids = gen.insert(
                    [g["prompt_ids"] for g in group],
                    max_tokens=[min(int(g.get("max_tokens", cap)), int(cap)) for g in group],
                )
                now = utc_iso()
                for uid, g in zip(uids, group):
                    meta[uid] = {"item": g, "t_submit": now, "t0": time.perf_counter(), "batch_id": batch_id,
                                 "t_first": None}
                    out[uid] = []
                in_flight += len(group)
                batch_id += 1
                admitted = len(group)
            else:
                admitted = 0

            # mlx_lm 0.32.0 (#1829): the public stats() window over the generator's monotonic counters replaces
            # the private _prompt_tokens_counter / _prompt_time_counter of 0.31.3. The window is filled on exit.
            with gen.stats() as st:
                t0 = time.perf_counter()
                prompt_resps, gen_resps = gen.next()
                dt = time.perf_counter() - t0
            p_tok = int(st.prompt_tokens)
            p_sec = float(st.prompt_time)
            if on_step is not None:
                on_step(gen)
            if gen_resps or p_tok:
                steps.append({
                    "type": "step", "step": step_i, "t": round(time.time(), 3),
                    "n_live": len(gen_resps), "n_prefill": len(gen._prompt_batch),
                    "padded_len": _padded_len(gen), "step_seconds": round(max(dt - p_sec, 0.0), 6),
                    "call_seconds": round(dt, 6), "prompt_tokens": int(p_tok), "prompt_seconds": round(p_sec, 6),
                    "admitted": admitted,
                })
                step_i += 1

            for r in gen_resps:
                m = meta[r.uid]
                if m["t_first"] is None:
                    m["t_first"] = utc_iso()
                out[r.uid].append(int(r.token))
                if r.finish_reason is None:
                    continue
                rec = _make_record(
                    m, out.pop(r.uid), r.finish_reason, arm, task, effort, pass_, family, B, seed, cfg, eos,
                    tokenizer, open_id, close_id, is_blank, scorer_split, b_fallback_from, cap,
                )
                if private_writer is not None:
                    private_writer.append(rec)
                    writer.append(_hashed(rec))
                else:
                    writer.append(rec)
                del meta[r.uid]
                in_flight -= 1
                summary["n_done"] += 1
                summary["n_truncated"] += int(rec["truncated"])
                st = rec["reasoning_status"]
                summary["status_counts"][st] = summary["status_counts"].get(st, 0) + 1
                last_record_utc = rec["t_done"]
                write_heartbeat(heartbeat_path, {**hb, "utc": utc_iso(), "last_record_utc": last_record_utc,
                                                 "n_done": summary["n_done"]})
                t_hb = time.monotonic()

            if time.monotonic() - t_hb > HEARTBEAT_EVERY_S:
                write_heartbeat(heartbeat_path, {**hb, "utc": utc_iso(), "last_record_utc": last_record_utc,
                                                 "n_done": summary["n_done"]})
                t_hb = time.monotonic()
            if time.monotonic() - t_env > ENV_EVERY_S:
                env_line()
                t_env = time.monotonic()
    finally:
        gen.close()
        steps.close()
    summary["n_steps"] = step_i
    return summary


def _make_record(m, completion, finish_reason, arm, task, effort, pass_, family, B, seed, cfg, eos,
                 tokenizer, open_id, close_id, is_blank, scorer_split, b_fallback_from, cap) -> dict:
    from runner import chat

    it = m["item"]
    prompt_ids = it["prompt_ids"]
    prefill = chat.prompt_prefill(family, it["prompt_text"])
    if scorer_split is not None:
        r_ids, a_ids, status = scorer_split(prompt_ids, completion, eos, is_blank)
        split_by = "scorers.reasoning.split_ids"
    else:
        r_ids, a_ids, status = split_ids(family, prefill, completion, open_id, close_id, eos, is_blank)
        split_by = "runner.generate.split_ids"
    t_done = utc_iso()
    wall = time.perf_counter() - m["t0"]
    rec = {
        "type": "record",
        "key": {"arm": arm, "task": task, "effort": effort, "item": str(it["id"]), "pass": int(pass_)},
        "item_sha256": it.get("item_sha256"),
        "prompt_sha256": it.get("prompt_sha256"),
        "rendered_sha256": it.get("rendered_sha256"),
        "rendered_prompt_tokens": len(prompt_ids),
        "prompt_prefill": prefill,
        "completion_ids": list(completion),
        "completion_tokens": len(completion),
        "reasoning_tokens": len(r_ids),
        "answer_tokens": len(a_ids),
        "text": _decode(tokenizer, completion),
        "reasoning_text": _decode(tokenizer, r_ids),
        "answer_text": _decode(tokenizer, a_ids),
        "finish_reason": finish_reason,
        "truncated": finish_reason == "length",
        "reasoning_status": status,
        "split_by": split_by,
        "stop_token": int(completion[-1]) if finish_reason == "stop" and completion else None,
        "t_submit": m["t_submit"],
        "t_first_token": m["t_first"],
        "t_done": t_done,
        "wall_s": round(wall, 3),
        "batch_id": m["batch_id"],
        "batch_size": B,
        "cell_seed": int(seed),
        "max_tokens": min(int(it.get("max_tokens", cap)), int(cap)),
        "sampler": dict(cfg),
        "eos_ids": list(eos),
    }
    if b_fallback_from is not None:
        rec["b_fallback_from"] = int(b_fallback_from)
    return rec


# ------------------------------------------------------------ teacher forcing


def iter_teacher_force_logprobs(model, ids: list[int], chunk: int = 2048):
    """Yield (start, logprobs[n, V] float32 numpy) chunk by chunk; row j of a
    chunk is the next-token distribution after ids[start + j]. Uses the
    model's own cache (model.make_cache()), so long texts never build a
    dense [T, T] mask (port README, Limitations)."""
    import mlx.core as mx
    import numpy as np
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    x = mx.array([list(ids)])
    for s in range(0, len(ids), chunk):
        logits = model(x[:, s:s + chunk], cache=cache).astype(mx.float32)[0]
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        mx.eval(lp)
        yield s, np.array(lp, dtype=np.float32)


def teacher_force_logprobs(model, ids: list[int], chunk: int = 2048):
    """Full [T, V] fp32 log-probs (fits for gate-text lengths; for long texts
    iterate with iter_teacher_force_logprobs)."""
    import numpy as np

    parts = [lp for _, lp in iter_teacher_force_logprobs(model, ids, chunk)]
    return np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), np.float32)
