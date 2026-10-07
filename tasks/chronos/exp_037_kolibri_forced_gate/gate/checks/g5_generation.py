# SPDX-License-Identifier: MIT
# Modified by Miktam for Chronos exp_037, 2026-10-06 (build task W5c: the one BatchGenerator
# construction, the shared admission loop, measurement only).
"""G5: the generation path (HYPOTHESIS Phase 0 G5; BUILD_SPEC §5.3 g5_generation.py).

  greedy continuations   K8 greedy 256-token continuations of the 8 G5 prompts
                         (4 EN, 4 DE; efforts none and high; >= 600 tokens),
                         teacher-forced through the reference: at positions
                         where the reference leads by >= 2 nats the generated
                         token is the reference top-1 at >= 99.5 %; it is within
                         the reference top-5 at >= 99 % of all positions
  noise floor            prefill with prefill_step_size 2048 vs 64 on the same
                         positions: floor_KL, floor_dis
  decode vs prefill      positions 520-1,100 of T1 and T3 and 15,000-15,300 of
                         T9: mean KL <= max(1e-4, 3 floor_KL), top-1
                         disagreement <= 3 floor_dis + 0.2 pp (pooled)
  batch parity           exp_036's batched-vs-single comparison (the peer check,
                         tools/peer_check.py, still uses it): the runner's
                         BatchGenerator, left-padded, staggered max_tokens so
                         that sequences are admitted mid-run, vs each sequence
                         alone, teacher-forced
  behaviour              the 20 frozen prompts through runner.generate.run_cell
                         (the scored runs' cell path, vendor sampling), effort
                         high capped at 8,192 and effort none: blocks if, for an
                         arm, >= 2 of 20 at one effort loop, end without EOS, or
                         get lang_tag "unknown"

exp_037 (DESIGN §6.3, §9.2; W5c):
* Measurement only. Every pass/fail decision is gate/rules.py's: parity_bound
  (exp_036 l.158-160) and behaviour_verdict (l.312-320) moved there. The two
  names below are re-exports of rules.py's functions, kept so that callers
  written against exp_036 (tools/peer_check.py; run_gate.py until W7's phases)
  read the one definition; nothing here decides.
* Every BatchGenerator comes from runner.generate.make_batch_generator, the
  kit's one construction (BUILD_SPEC item 27: completion_batch_size B, the
  registered prefill_batch_size = min(B, 8), prefill_step_size 2,048,
  max_kv_size None, fp32 logits). No row assertion is made (decision F2).
* run_admission is run_cell's admission loop (at most B - in_flight new
  sequences before each step), shared by batch_parity and by
  gate/checks/g5_runner.py (G5-R1 at B = 1, G5-BP-lean at B = 8 and 16). Its
  max_live reads BatchGenerator._generation_batch and ._prompt_batch, which
  mlx-lm 0.32.0 keeps (generate.py l.1553-1559).
* greedy_rows_vs_ref is the greedy measure on reference rows taken at the
  generated positions only (phase 6's R1 rows; G5 greedy and G5-R1 R-greedy).

A loop is a window of 4 x 32 = 128 tokens with a period of at most 32 tokens
(a 32-token span repeated 4 times consecutively, or a shorter cycle).
"""

from __future__ import annotations

import numpy as np

from gate import common
from gate.rules import behaviour_verdict, parity_bound  # noqa: F401  (re-exports; see the module docstring)

CONSTRUCTION = "runner.generate.make_batch_generator (BUILD_SPEC item 27) + runner.sampler.fp32_logits"


def _lp(x: np.ndarray) -> np.ndarray:
    return common.log_softmax64(np.asarray(x, dtype=np.float64))


def kl_lp(lp_p: np.ndarray, lp_q: np.ndarray) -> np.ndarray:
    return (np.exp(lp_p) * (lp_p - lp_q)).sum(axis=-1)


# ---------------------------------------------------------------------------
# Greedy continuations (phase 4) and their reference comparison (phase 6)
# ---------------------------------------------------------------------------


def greedy_continuations(model, prompts: list, n_tokens: int, eos=common.EOS_IDS) -> list:
    import mlx.core as mx
    from mlx_lm.generate import generate_step

    out = []
    for p in prompts:
        gen, finished = [], "length"
        for tok, _ in generate_step(mx.array(p["ids"]), model, max_tokens=n_tokens,
                                    sampler=lambda x: mx.argmax(x, axis=-1), prefill_step_size=2048):
            t = int(tok)
            gen.append(t)
            if t in eos:
                finished = "eos"
                break
        out.append({"id": p["id"], "lang": p["lang"], "effort": p["effort"], "prompt_tokens": len(p["ids"]),
                    "prompt_ids_sha256": common.ids_sha256(p["ids"]), "continuation": gen, "finished": finished})
    return out


def greedy_rows_vs_ref(items: list, decisive_lead: float) -> dict:
    """The greedy measure on reference rows at the generated positions.

    items: [(id, ref_rows [n, V], tokens [n], finished)], where ref_rows[j] is
    the reference's next-token distribution (logits or log-probs) for
    generated token tokens[j]. Decisive: the reference's top-1 leads its top-2
    by >= decisive_lead nats."""
    hit_dec, n_dec, in5, n_all = 0, 0, 0, 0
    per = {}
    for sid, ref_rows, tokens, finished in items:
        g = np.asarray(tokens)
        if g.size == 0:
            continue
        rows = np.asarray(ref_rows, dtype=np.float64)
        if rows.shape[0] != g.size:
            raise ValueError(f"{sid}: {rows.shape[0]} reference rows for {g.size} generated tokens")
        lp = _lp(rows)
        top1 = lp.argmax(axis=-1)
        lead = lp.max(axis=-1) - np.partition(lp, -2, axis=-1)[:, -2]
        top5 = np.argpartition(-lp, 5, axis=-1)[:, :5]
        dec = lead >= decisive_lead
        is5 = np.any(top5 == g[:, None], axis=-1)
        hit_dec += int((top1 == g)[dec].sum())
        n_dec += int(dec.sum())
        in5 += int(is5.sum())
        n_all += int(g.size)
        per[sid] = {"n": int(g.size), "n_decisive": int(dec.sum()),
                    "decisive_top1": float((top1 == g)[dec].mean()) if dec.any() else None,
                    "in_top5": float(is5.mean()), "finished": finished}
    return {"n_positions": n_all, "n_decisive": n_dec,
            "decisive_top1": hit_dec / n_dec if n_dec else None,
            "in_top5": in5 / n_all if n_all else None, "per_prompt": per}


def greedy_vs_ref(conts: list, prompts: dict, ref_logits_by_id: dict, decisive_lead: float) -> dict:
    """ref_logits_by_id[id]: reference logits for prompt + continuation (row
    P - 1 + j predicts generated token j)."""
    items = []
    for c in conts:
        P, g = c["prompt_tokens"], np.asarray(c["continuation"])
        if g.size == 0:
            continue
        items.append((c["id"], ref_logits_by_id[c["id"]][P - 1:P - 1 + g.size], g, c["finished"]))
    return greedy_rows_vs_ref(items, decisive_lead)


# ---------------------------------------------------------------------------
# Noise floor and decode vs prefill
# ---------------------------------------------------------------------------


def prefill_rows(model, ids, lo: int, hi: int, step: int) -> np.ndarray:
    """Logit rows lo..hi (inclusive) of a teacher-forced prefill of ids[:hi+1]
    fed in chunks of `step` through the model's own cache."""
    import mlx.core as mx

    cache = model.make_cache()
    ids = np.asarray(ids[:hi + 1], dtype=np.int32)
    rows = []
    for a in range(0, ids.size, step):
        b = min(ids.size, a + step)
        out = model(mx.array(ids[a:b])[None], cache=cache)
        if b > lo:
            sel = out[0, max(lo, a) - a:b - a].astype(mx.float32)
            mx.eval(sel)
            rows.append(np.array(sel))
        else:
            mx.eval(out)
    return np.concatenate(rows, axis=0)


def decode_rows(model, ids, lo: int, hi: int, step: int = 2048) -> np.ndarray:
    """Prefill ids[:lo] (chunks of `step`), then feed ids[lo..hi] one token at
    a time; the row of each decode step."""
    import mlx.core as mx

    cache = model.make_cache()
    ids = np.asarray(ids, dtype=np.int32)
    for a in range(0, lo, step):
        mx.eval(model(mx.array(ids[a:min(lo, a + step)])[None], cache=cache))
    rows = []
    for p in range(lo, hi + 1):
        out = model(mx.array(ids[p:p + 1])[None], cache=cache)[0, -1].astype(mx.float32)
        mx.eval(out)
        rows.append(np.array(out))
    return np.stack(rows)


def floor_and_decode(model, texts: dict, ranges: dict, steps=(2048, 64), log=print) -> dict:
    """texts: {name: ids}; ranges: {name: (lo, hi)}. Pooled over the ranges."""
    kl_floor, dis_floor, kl_dec, dis_dec, per = [], [], [], [], {}
    big, small = steps
    for name, (lo, hi) in ranges.items():
        ids = texts[name]
        a = _lp(prefill_rows(model, ids, lo, hi, big))
        b = _lp(prefill_rows(model, ids, lo, hi, small))
        d = _lp(decode_rows(model, ids, lo, hi, big))
        kf, kd = kl_lp(a, b), kl_lp(a, d)
        df, dd = a.argmax(-1) != b.argmax(-1), a.argmax(-1) != d.argmax(-1)
        kl_floor.append(kf), dis_floor.append(df), kl_dec.append(kd), dis_dec.append(dd)
        per[name] = {"positions": [lo, hi], "floor_kl": float(kf.mean()), "floor_dis": float(df.mean()),
                     "decode_kl": float(kd.mean()), "decode_dis": float(dd.mean())}
        log(f"[gate] G5 decode vs prefill {name}")
    return {"floor_kl": float(np.concatenate(kl_floor).mean()), "floor_dis": float(np.concatenate(dis_floor).mean()),
            "decode_kl": float(np.concatenate(kl_dec).mean()), "decode_dis": float(np.concatenate(dis_dec).mean()),
            "prefill_steps": list(steps), "per_text": per}


# ---------------------------------------------------------------------------
# The batched path: one construction, one admission loop
# ---------------------------------------------------------------------------


def greedy_sampler():
    """Argmax of the fp32 log-probs BatchGenerator hands its sampler."""
    import mlx.core as mx

    return lambda logprobs: mx.argmax(logprobs, axis=-1)


def _batch_generator(model, B: int, eos, max_tokens: int, prefill_step_size: int | None = None):
    """(generator, construction label): runner.generate.make_batch_generator,
    the kit's one BatchGenerator construction (DESIGN §6.3), with a greedy
    sampler. Looked up on the module at call time, so a test can watch it."""
    import mlx.core as mx

    from runner import generate

    mx.set_cache_limit(generate.CACHE_LIMIT_BYTES)
    step = generate.PREFILL_STEP_SIZE if prefill_step_size is None else int(prefill_step_size)
    gen = generate.make_batch_generator(model, B, eos, max_tokens, greedy_sampler(), prefill_step_size=step)
    return gen, CONSTRUCTION


def max_live(gen) -> int:
    """Sequences the generator holds live: in prefill plus in decode (exp_036
    l.215). Queued, not yet prefilled sequences are not counted."""
    return len(gen._generation_batch) + len(gen._prompt_batch)


def _to_np(a) -> np.ndarray:
    if isinstance(a, np.ndarray):
        return a.astype(np.float32, copy=False)
    import mlx.core as mx

    a = a.astype(mx.float32)
    mx.eval(a)
    return np.array(a)


def run_admission(gen, prompts: list, max_tokens: list, B: int, *, on_finish=None, keep_logprobs: bool = True,
                  max_steps: int = 100_000) -> dict:
    """run_cell's admission loop (runner/generate.py) on `gen`: before each
    gen.next(), at most B - in_flight queued prompts are inserted, in queue
    order, so later prompts join a running batch as earlier ones finish.

    Returns {"seqs": [per prompt, in queue order: {"tokens", "logprobs" ([n, V]
    fp32, or None when on_finish took it or keep_logprobs is False),
    "finish_reason", "insert_order", "insert_step", "finish_step",
    "first_wave", "mid_run"}], "max_live", "admitted_mid_run", "steps",
    "first_finish_step", "B"}.

    first_wave: among the first B sequences inserted. mid_run: inserted after
    the first finish (DESIGN §3.14; consistency-14). on_finish(j, tokens,
    logprobs) is called once per sequence when it finishes, and its return
    value is stored as seqs[j]["on_finish"]; the log-probs are then dropped."""
    n = len(prompts)
    if len(max_tokens) != n:
        raise ValueError(f"{len(max_tokens)} max_tokens for {n} prompts")
    if B < 1:
        raise ValueError("B must be >= 1")
    seqs = [{"tokens": [], "logprobs": [], "finish_reason": None, "insert_order": None, "insert_step": None,
             "finish_step": None, "first_wave": False, "mid_run": False} for _ in range(n)]
    queue = list(range(n))
    uid_of = {}
    in_flight = step = live = admitted_mid_run = order = 0
    first_finish = None
    try:
        while queue or in_flight:
            free = B - in_flight
            if free > 0 and queue:
                group = queue[:free]
                del queue[:free]
                uids = gen.insert([list(prompts[j]) for j in group], max_tokens=[int(max_tokens[j]) for j in group])
                if len(uids) != len(group):
                    raise RuntimeError(f"insert returned {len(uids)} uids for {len(group)} prompts")
                for u, j in zip(uids, group):
                    uid_of[u] = j
                    s = seqs[j]
                    s.update(insert_order=order, insert_step=step, first_wave=order < B, mid_run=first_finish is not None)
                    order += 1
                in_flight += len(group)
                if first_finish is not None:
                    admitted_mid_run += len(group)
            _, resps = gen.next()
            live = max(live, max_live(gen))
            for r in resps:
                j = uid_of[r.uid]
                s = seqs[j]
                if s["finish_reason"] is not None:
                    raise RuntimeError(f"a response for prompt {j} after it finished")
                s["tokens"].append(int(r.token))
                if keep_logprobs or on_finish is not None:
                    s["logprobs"].append(_to_np(r.logprobs))
                if r.finish_reason is not None:
                    s["finish_reason"] = r.finish_reason
                    s["finish_step"] = step
                    in_flight -= 1
                    if first_finish is None:
                        first_finish = step
                    lps = np.stack(s["logprobs"]) if s["logprobs"] else None
                    if on_finish is not None:
                        s["on_finish"] = on_finish(j, list(s["tokens"]), lps)
                        s["logprobs"] = None
                    else:
                        s["logprobs"] = lps if keep_logprobs else None
            step += 1
            if step > max_steps:
                raise RuntimeError(f"the batch did not finish in {max_steps} steps")
    finally:
        gen.close()
    return {"seqs": seqs, "max_live": live, "admitted_mid_run": admitted_mid_run, "steps": step,
            "first_finish_step": first_finish, "B": int(B)}


def batch_parity(model, prompts: list, max_tokens: list, B: int, eos=common.EOS_IDS) -> dict:
    """prompts: list of id lists (more than B). Admission follows
    runner/generate.run_cell (run_admission): at most B - in_flight new
    sequences are inserted before each step, so later prompts join a running
    batch as earlier ones finish (mid-run admission, left padding). Each
    sequence's batched log-probs are compared with the same sequence alone,
    teacher-forced through a fresh cache."""
    from gate.harness import logits_at

    gen, path = _batch_generator(model, B, eos, max(max_tokens))
    res = run_admission(gen, prompts, max_tokens, B)
    kls, dis = [], []
    for j, s in enumerate(res["seqs"]):
        p, toks = prompts[j], s["tokens"]
        seq = list(p) + toks[:-1]
        P = len(p)
        single = _lp(logits_at(model, seq, np.arange(P - 1, P - 1 + len(toks))))
        batched = _lp(s["logprobs"])
        kls.append(kl_lp(single, batched))
        dis.append(single.argmax(-1) != batched.argmax(-1))
    kl, d = np.concatenate(kls), np.concatenate(dis)
    return {"path": path, "B": B, "n_sequences": len(prompts), "n_positions": int(kl.size),
            "mean_kl": float(kl.mean()), "top1_dis": float(d.mean()), "max_live": res["max_live"],
            "admitted_mid_run": res["admitted_mid_run"], "prompt_lengths": [len(p) for p in prompts],
            "max_tokens": list(max_tokens), "steps": res["steps"]}


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


def has_loop(ids, span: int = 32, repeats: int = 4) -> bool:
    """A window of span * repeats tokens with a period of at most `span`."""
    a = np.asarray(ids)
    w = span * repeats
    if a.size < w:
        return False
    for p in range(1, span + 1):
        eq = (a[p:] == a[:-p]).astype(np.int32)  # eq[t]: a[t] == a[t + p]
        need = w - p
        if eq.size >= need:
            run = np.convolve(eq, np.ones(need, dtype=np.int32), mode="valid")
            if (run == need).any():
                return True
    return False


class _ListWriter:
    def __init__(self):
        self.records = []

    def append(self, rec: dict) -> None:
        self.records.append(rec)


def behaviour(model, tokenizer, items: list, arm: str, effort: str, cap: int, B: int, eos=common.EOS_IDS,
              lang_tag=None, span: int = 32, repeats: int = 4) -> dict:
    """items: [{"id", "lang", "messages"} or {"id", "lang", "prompt_ids", "prompt_text"}],
    generated through runner.generate.run_cell with the vendor sampling."""
    from runner.generate import run_cell

    w = _ListWriter()
    cell_items = []
    for it in items:
        d = {"id": it["id"], "item_sha256": common.sha256_bytes(common.dumps(it.get("messages") or it["prompt_ids"]).encode()),
             "prompt_sha256": None, "max_tokens": cap}
        if "messages" in it:
            d["messages"] = it["messages"]
        else:
            d["prompt_ids"], d["prompt_text"] = it["prompt_ids"], it.get("prompt_text", "")
        cell_items.append(d)
    seed = common.seed_from("gate", "behaviour", arm, effort) % (2**32)
    run_cell(model, tokenizer, cell_items, arm, "gate_behaviour", effort, cap, B, seed, w, "kolibri", eos=list(eos))
    by_id = {r["key"]["item"]: r for r in w.records}
    rows = []
    for it in items:
        r = by_id[it["id"]]
        lang = lang_tag(r.get("answer_text") or r.get("text") or "") if lang_tag else None
        rows.append({"id": it["id"], "lang": it["lang"], "completion_tokens": r["completion_tokens"],
                     "finish_reason": r["finish_reason"], "eos": r["finish_reason"] == "stop",
                     "loop": has_loop(r["completion_ids"], span, repeats), "lang_tag": lang,
                     "lang_match": (lang == it["lang"]) if lang else None,
                     "reasoning_status": r.get("reasoning_status")})
    n = len(rows)
    return {"arm": arm, "effort": effort, "cap": cap, "B": B, "seed": seed, "n": n,
            "n_loop": sum(r["loop"] for r in rows), "n_no_eos": sum(not r["eos"] for r in rows),
            "n_unknown_lang": sum(r["lang_tag"] == "unknown" for r in rows),
            "think_closed_rate": sum(r["reasoning_status"] == "closed" for r in rows) / n,
            "lang_match_rate": (sum(bool(r["lang_match"]) for r in rows) / n) if lang_tag else None,
            "rows": rows}
