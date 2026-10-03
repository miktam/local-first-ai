# SPDX-License-Identifier: MIT
"""G5: the generation path (HYPOTHESIS Phase 0 G5; BUILD_SPEC §5.3 g5_generation.py).

  greedy continuations   K8 greedy 256-token continuations of the 8 G5 prompts
                         (4 EN, 4 DE; efforts none and high; >= 600 tokens),
                         teacher-forced through the reference (phase 5): at
                         positions where the reference leads by >= 2 nats the
                         generated token is the reference top-1 at >= 99.5 %;
                         it is within the reference top-5 at >= 99 % of all positions
  noise floor            prefill with prefill_step_size 2048 vs 64 on the same
                         positions: floor_KL, floor_dis
  decode vs prefill      positions 520-1,100 of T1 and T3 and 15,000-15,300 of
                         T9: mean KL <= max(1e-4, 3 floor_KL), top-1
                         disagreement <= 3 floor_dis + 0.2 pp (pooled)
  batch parity           B = 8 through the runner's BatchGenerator construction
                         (item 27), left-padded, lengths 37 ... 1,100, staggered
                         max_tokens so that sequences are admitted mid-run, vs
                         each sequence alone, teacher-forced; K8 must meet the
                         decode-vs-prefill bound, K4 is reported
  behaviour              the 20 frozen prompts through runner.generate.run_cell
                         (the scored runs' cell path, vendor sampling), effort
                         high capped at 8,192 and effort none: blocks if, for an
                         arm, >= 2 of 20 at one effort loop, end without EOS, or
                         get lang_tag "unknown"

A loop is a window of 4 x 32 = 128 tokens with a period of at most 32 tokens
(a 32-token span repeated 4 times consecutively, or a shorter cycle).
"""

from __future__ import annotations

import numpy as np

from gate import common


def _lp(x: np.ndarray) -> np.ndarray:
    return common.log_softmax64(np.asarray(x, dtype=np.float64))


def kl_lp(lp_p: np.ndarray, lp_q: np.ndarray) -> np.ndarray:
    return (np.exp(lp_p) * (lp_p - lp_q)).sum(axis=-1)


# ---------------------------------------------------------------------------
# Greedy continuations (phase 4) and their reference comparison (phase 5)
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


def greedy_vs_ref(conts: list, prompts: dict, ref_logits_by_id: dict, decisive_lead: float) -> dict:
    """ref_logits_by_id[id]: reference logits for prompt + continuation."""
    hit_dec, n_dec, in5, n_all = 0, 0, 0, 0
    per = {}
    for c in conts:
        P, g = c["prompt_tokens"], np.asarray(c["continuation"])
        if g.size == 0:
            continue
        rows = np.asarray(ref_logits_by_id[c["id"]][P - 1:P - 1 + g.size], dtype=np.float64)
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
        per[c["id"]] = {"n": int(g.size), "n_decisive": int(dec.sum()),
                        "decisive_top1": float((top1 == g)[dec].mean()) if dec.any() else None,
                        "in_top5": float(is5.mean()), "finished": c["finished"]}
    return {"n_positions": n_all, "n_decisive": n_dec,
            "decisive_top1": hit_dec / n_dec if n_dec else None,
            "in_top5": in5 / n_all if n_all else None, "per_prompt": per}


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


def parity_bound(floor: dict, th: dict) -> dict:
    return {"kl_max": max(th["parity_kl_floor"], th["parity_kl_factor"] * floor["floor_kl"]),
            "dis_max": th["parity_dis_factor"] * floor["floor_dis"] + th["parity_dis_add"]}


# ---------------------------------------------------------------------------
# Batch parity
# ---------------------------------------------------------------------------


def _batch_generator(model, B: int, eos, max_tokens: int):
    """mlx_lm BatchGenerator built as runner/generate.run_cell builds it
    (BUILD_SPEC item 27), with a greedy sampler for the parity check."""
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator

    try:
        from runner.generate import CACHE_LIMIT_BYTES, PREFILL_STEP_SIZE
        from runner.sampler import fp32_logits
        path = "runner.generate (item 27 construction) + runner.sampler.fp32_logits"
    except ImportError:  # tiny tests before runner/ existed
        CACHE_LIMIT_BYTES, PREFILL_STEP_SIZE, fp32_logits = 4 << 30, 2048, (lambda m: m)
        path = "mlx_lm BatchGenerator (runner/ absent)"
    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    gen = BatchGenerator(fp32_logits(model), max_tokens=max_tokens, stop_tokens=[[e] for e in eos],
                         sampler=lambda x: mx.argmax(x, axis=-1), completion_batch_size=B,
                         prefill_batch_size=min(B, 8), prefill_step_size=PREFILL_STEP_SIZE, max_kv_size=None)
    return gen, path


def batch_parity(model, prompts: list, max_tokens: list, B: int, eos=common.EOS_IDS) -> dict:
    """prompts: list of id lists (more than B). Admission follows
    runner/generate.run_cell: at most B - in_flight new sequences are inserted
    before each step, so later prompts join a running batch as earlier ones
    finish (mid-run admission, left padding). Each sequence's batched log-probs
    are compared with the same sequence alone, teacher-forced through a fresh
    cache."""
    import mlx.core as mx

    from gate.harness import logits_at

    gen, path = _batch_generator(model, B, eos, max(max_tokens))
    got, queue = {}, list(range(len(prompts)))
    uid_of, in_flight, step, max_live, admitted_mid_run, any_finished = {}, 0, 0, 0, 0, False
    try:
        while queue or in_flight:
            free = B - in_flight
            if free > 0 and queue:
                group = queue[:free]
                del queue[:free]
                uids = gen.insert([list(prompts[j]) for j in group], max_tokens=[int(max_tokens[j]) for j in group])
                for u, j in zip(uids, group):
                    uid_of[u] = j
                    got[u] = {"tokens": [], "lp": []}
                in_flight += len(group)
                admitted_mid_run += len(group) if any_finished else 0
            _, resps = gen.next()
            max_live = max(max_live, len(gen._generation_batch) + len(gen._prompt_batch))
            for r in resps:
                g = got[r.uid]
                g["tokens"].append(int(r.token))
                lp = r.logprobs.astype(mx.float32)
                mx.eval(lp)
                g["lp"].append(np.array(lp))
                if r.finish_reason is not None:
                    in_flight -= 1
                    any_finished = True
            step += 1
            if step > 100_000:
                raise RuntimeError("batch parity did not finish")
    finally:
        gen.close()
    kls, dis = [], []
    for u, j in sorted(uid_of.items(), key=lambda t: t[1]):
        g, p = got[u], prompts[j]
        seq = list(p) + g["tokens"][:-1]
        P = len(p)
        single = _lp(logits_at(model, seq, np.arange(P - 1, P - 1 + len(g["tokens"]))))
        batched = _lp(np.stack(g["lp"]))
        kls.append(kl_lp(single, batched))
        dis.append(single.argmax(-1) != batched.argmax(-1))
    kl, d = np.concatenate(kls), np.concatenate(dis)
    return {"path": path, "B": B, "n_sequences": len(prompts), "n_positions": int(kl.size),
            "mean_kl": float(kl.mean()), "top1_dis": float(d.mean()), "max_live": max_live,
            "admitted_mid_run": admitted_mid_run, "prompt_lengths": [len(p) for p in prompts],
            "max_tokens": list(max_tokens), "steps": step}


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


def behaviour_verdict(cells: dict, block_count: int) -> tuple[list, dict]:
    """(blocked efforts, summary). An (arm, effort) cell of 20 blocks when
    >= block_count completions loop, or end without EOS, or get lang_tag
    "unknown" (HYPOTHESIS G5 behaviour row)."""
    blocked = [e for e, c in cells.items()
               if c["n_loop"] >= block_count or c["n_no_eos"] >= block_count or c["n_unknown_lang"] >= block_count]
    summary = {e: {k: c[k] for k in ("n", "n_loop", "n_no_eos", "n_unknown_lang", "think_closed_rate",
                                     "lang_match_rate", "cap", "seed")} for e, c in cells.items()}
    return blocked, summary
