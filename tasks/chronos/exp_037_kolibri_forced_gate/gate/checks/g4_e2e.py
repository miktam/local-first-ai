# SPDX-License-Identifier: MIT
"""G4: end to end, free routing (HYPOTHESIS Phase 0 G4; BUILD_SPEC §5.3 g4_e2e.py).

A converted build (K8 or K4, normal load) teacher-forced over T1-T8 (12,288
positions) and over T9 in chunks through its own cache (the runner's
teacher_force path), against the fp32 reference's logits. KL(ref || port)
is computed in float64 from the full vocabulary, chunked by position.

exp_037 (DESIGN §3.9, §3.10, §9.2; build task W5b): measurement only. The
rules moved to gate/rules.py, unchanged in substance:
  K8, G4-N(i) (rules.g4_n; blocking, as registered): mean KL(R1 || K8) <= 0.10
     nats/token on T1-T8 and on T9, each pooled; K8 bf16 as loaded, free
     routing, chunk 2,048. Decisive top-1 (the registered 99 %) is now
     descriptive.
  K4 (rules.g4_k4): descriptive only (feeds H8 and E11).
e2e() returns the measured values only: per set mean KL, top-1, decisive
top-1, dNLL, and KL by window position, by T9 bucket, by language and per
text (the G4 descriptive rows, §3.10). The forced checks G4-F32 and G4-F16
are gate/checks/g4_forced.py.
"""

from __future__ import annotations

import numpy as np

from gate import common


def _teacher_force():
    """runner.generate.iter_teacher_force_logprobs (the teacher forcing H8
    uses; BUILD_SPEC §5.4) when present, else the gate harness's own."""
    try:
        from runner.generate import iter_teacher_force_logprobs

        return iter_teacher_force_logprobs, "runner.generate.iter_teacher_force_logprobs"
    except ImportError:
        from gate.harness import stream_logits

        return stream_logits, "gate.harness.stream_logits"

ORDER8 = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")


def _compare(ref: np.ndarray, got: np.ndarray, ids_next: np.ndarray) -> dict:
    """Per row: KL(ref || got), the reference's top-1 lead (nats), top-1
    agreement and dNLL of the next token (NaN where there is none)."""
    kl = common.kl_rows(ref, got)
    lp_ref = common.log_softmax64(ref)
    lead = lp_ref.max(axis=-1) - np.partition(lp_ref, -2, axis=-1)[:, -2]
    agree = ref.argmax(axis=-1) == got.argmax(axis=-1)
    dnll = np.full(ref.shape[0], np.nan)
    n = ids_next.size
    if n:
        lp_got = common.log_softmax64(got[:n])
        r = np.arange(n)
        dnll[:n] = -lp_got[r, ids_next] + lp_ref[r, ids_next]
    return {"kl": kl, "lead": lead, "agree": agree, "dnll": dnll}


def _cat(parts: list[dict]) -> dict:
    return {k: np.concatenate([p[k] for p in parts]) for k in ("kl", "lead", "agree", "dnll")}


def _summary(parts: list[dict], decisive_lead: float) -> dict:
    c = _cat(parts)
    dec = c["lead"] >= decisive_lead
    return {"n": int(c["kl"].size), "mean_kl": float(c["kl"].mean()), "max_kl": float(c["kl"].max()),
            "top1": float(c["agree"].mean()), "n_decisive": int(dec.sum()),
            "top1_decisive": float(c["agree"][dec].mean()) if dec.any() else None,
            "dnll_mean": float(np.nanmean(c["dnll"])) if np.isfinite(c["dnll"]).any() else None}


def _masked(c: dict, m: np.ndarray) -> dict:
    return {k: v[m] for k, v in c.items()}


def e2e(model, textset, dump, decisive_lead: float, log=print) -> dict:
    """Per-position comparison over T1-T8 and T9; returns summaries only."""
    prof = textset.profile
    L, chunk, w = prof.text_len, prof.prefill_chunk, prof.window_position
    ref_logits = dump.load("logits.npy")
    stream_logits, path = _teacher_force()
    per_text, parts8 = {}, []
    by_lang = {"en": [], "de": []}
    before, after = [], []
    for j, tid in enumerate(ORDER8):
        ids = np.asarray(textset.ids[tid])
        text_parts = []
        for a, block in stream_logits(model, list(map(int, ids)), chunk):
            b = a + block.shape[0]
            ref = np.asarray(ref_logits[j * L + a:j * L + b], dtype=np.float32)
            c = _compare(ref, block, ids[a + 1:b + 1])
            text_parts.append(c)
            pos = np.arange(a, b)
            before.append(_masked(c, pos < w))
            after.append(_masked(c, pos >= w))
        per_text[tid] = _summary(text_parts, decisive_lead)
        parts8 += text_parts
        by_lang[textset.lang(tid)] += text_parts
        log(f"[gate] G4 {tid}")
    t9 = np.asarray(textset.t9)
    n8 = 8 * L
    parts9 = []
    buckets = {tuple(bk): [] for bk in prof.t9_buckets}
    for a, block in stream_logits(model, list(map(int, t9)), chunk):
        b = a + block.shape[0]
        ref = np.asarray(ref_logits[n8 + a:n8 + b], dtype=np.float32)
        c = _compare(ref, block, t9[a + 1:b + 1])
        parts9.append(c)
        pos = np.arange(a, b)
        for lo, hi in buckets:
            m = (pos >= lo) & (pos < hi)
            if m.any():
                buckets[(lo, hi)].append(_masked(c, m))
    log("[gate] G4 T9")
    return {
        "teacher_forcing": path,
        "T1-8": _summary(parts8, decisive_lead),
        "T9": _summary(parts9, decisive_lead),
        "per_text": per_text,
        "by_lang": {k: _summary(v, decisive_lead) for k, v in by_lang.items() if v},
        "kl_by_window": {f"pos<{w}": _summary(before, decisive_lead)["mean_kl"],
                         f"pos>={w}": _summary(after, decisive_lead)["mean_kl"]},
        "kl_by_bucket": {f"{lo}-{hi}": _summary(v, decisive_lead) for (lo, hi), v in buckets.items() if v},
    }
