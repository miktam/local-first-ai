# SPDX-License-Identifier: MIT
"""exp_036 G3 diagnostic (descriptive only; not a gate run, writes no gate record).

Run on the mbp from the kit root, after `git pull`, with the kit's Python:
    MLX_ENABLE_TF32=0 caffeinate -i "$PY" /path/to/g3_diag_mbp.py --out diagnostics_g3_<UTC>.json
(~10-15 min; peak RSS ~ K8's ~80 GB, then Q36-8 ~40 GB, loaded one after the other.)

Reads the EXISTING fp32 reference dump of gate 20261005T050112Z (no reference pass)
and runs K8 and Q36-8 teacher-forced on T1-T6. Reports, pre-registered in the
investigation note:
  D1  K8 no-prefix bpb == reference bpb (sanity; G4 dnll was -0.001 nats/token)
  D2  K8 bpb with <|endoftext|> before the text, and Amendment 5 chat-wrapped
      (what a chat-wrapped G3 would show), vs the gate's no-prefix rule
  D3  where the Kolibri - Q36-8 gap lives: per line, per T4 article, digits vs not,
      and the share of tokens in verbatim recall (NLL < 0.05)
  D4  T4 greedy recall probe (48 tokens after each article label), K8 and Q36-8
  F   bpb of the reference excluding its first 16 / 64 tokens
Stand-in mode for testing on the mini: --k-dir/--q-dir point at any two mlx_lm
models and --no-dump skips the reference dump.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

KIT = Path.cwd()
sys.path.insert(0, str(KIT))

LN2 = math.log(2)
TEXTS = ("T1", "T2", "T3", "T4", "T5", "T6")


def tf_nll(model, prompt_ids, ids, chunk=2048):
    """Per-token NLL of ids[j] for every j (prompt_ids non-empty), or of ids[1:]
    when prompt_ids is empty (the gate's no-prefix rule). fp32 log-softmax."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    inputs = list(prompt_ids) + list(ids[:-1]) if prompt_ids else list(ids[:-1])
    targets = list(ids) if prompt_ids else list(ids[1:])
    first = len(prompt_ids) - 1 if prompt_ids else 0
    cache = make_prompt_cache(model)
    rows = []
    for s in range(0, len(inputs), chunk):
        x = mx.array(inputs[s:s + chunk], dtype=mx.int32)[None]
        lg = model(x, cache=cache)[0].astype(mx.float32)
        lp = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
        lo = max(first - s, 0)
        n = lp.shape[0]
        if lo >= n:
            continue
        t0 = s + lo - first
        tg = mx.array(np.asarray(targets[t0:t0 + n - lo], dtype=np.int32))[:, None]
        v = -mx.take_along_axis(lp[lo:], tg, axis=-1)[:, 0]
        mx.eval(v)
        rows.append(np.array(v, dtype=np.float64))
    out = np.concatenate(rows)
    assert out.size == len(targets), (out.size, len(targets))
    return out


def greedy(model, prompt_ids, n):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    cache = make_prompt_cache(model)
    lg = model(mx.array(prompt_ids, dtype=mx.int32)[None], cache=cache)[0, -1]
    out = []
    for _ in range(n):
        t = int(mx.argmax(lg).item())
        out.append(t)
        lg = model(mx.array([[t]], dtype=mx.int32), cache=cache)[0, -1]
    return out


def tokenize(tok_json: Path, text: str):
    from bench.kl_8v4 import tokenize as tk

    ids, fb = tk(tok_json, text)
    return ids, fb


def line_of(text_bytes: bytes):
    """byte offset -> line index (lines split at b'\\n')."""
    starts = [0] + [m.end() for m in re.finditer(b"\n", text_bytes)]
    return np.asarray(starts, dtype=np.int64)


def per_line(nll, first_bytes, starts):
    idx = np.searchsorted(starts, first_bytes, side="right") - 1
    return np.bincount(idx, weights=nll, minlength=len(starts))


def article_spans(text: str):
    b = text.encode("utf-8")
    heads = [(m.start(), m.group(1)) for m in re.finditer(rb"(?m)^Art (\d+[a-z]?)$", b)]
    spans = []
    for k, (s, name) in enumerate(heads):
        e = heads[k + 1][0] if k + 1 < len(heads) else len(b)
        spans.append((name.decode(), s, e))
    return spans


def probe_points(text, ids, fb, names=("2", "3", "5", "8", "12a", "16a"), n=48):
    """(name, prompt ids, true next ids) split inside the JOINT tokenisation of
    T4, right after each article's label line 'Art N\\n'."""
    out = []
    for name, s, e in article_spans(text):
        if name not in names:
            continue
        cut = s + len(f"Art {name}\n".encode())
        j = int(np.searchsorted(fb, cut, side="left"))
        if j + n <= len(ids):
            out.append((name, list(ids[:j]), list(ids[j:j + n])))
    return out


def byte_bits(nll, fb, nbytes, skip_first=False):
    """Spread each token's bits evenly over its bytes [fb_j, fb_j+1)."""
    bounds = np.append(np.asarray(fb, dtype=np.int64), nbytes)
    out = np.zeros(nbytes)
    vals = np.asarray(nll, dtype=np.float64) / LN2
    toks = range(1, len(fb)) if skip_first else range(len(fb))
    for v, j in zip(vals, toks):
        a, b = bounds[j], bounds[j + 1]
        if b > a:
            out[a:b] += v / (b - a)
        else:
            out[max(a - 1, 0)] += v
    if skip_first:
        out[:bounds[1]] = np.nan
    return out


def stats_block(nll_bits_lines_k, nll_bits_lines_q, line_bytes):
    gap = nll_bits_lines_k - nll_bits_lines_q
    tot = gap.sum()
    order = np.argsort(-gap)
    top10 = order[:max(1, len(order) // 10)]
    rho = float(np.corrcoef(np.argsort(np.argsort(gap)), np.argsort(np.argsort(nll_bits_lines_q)))[0, 1]) \
        if len(gap) > 2 else None
    return {"gap_bits": float(tot), "lines": int(len(gap)),
            "top10pct_lines_share_of_gap": float(gap[top10].sum() / tot) if tot else None,
            "lines_where_kolibri_better": int((gap < 0).sum()),
            "spearman_gap_vs_q_bits": rho}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k-dir")
    ap.add_argument("--q-dir")
    ap.add_argument("--k-family", default="kolibri")
    ap.add_argument("--no-dump", action="store_true")
    ap.add_argument("--texts-dir", help="dir with T5.txt/T6.txt (default $EXP036_WORK/gate_texts)")
    a = ap.parse_args()

    from tools.precision import ensure_exact_fp32
    ensure_exact_fp32()
    from bench import common as bc, genutil
    from bench import kl_8v4 as K
    from gate import common as gc

    k_dir = Path(a.k_dir) if a.k_dir else bc.arm_dir("K8")
    q_dir = Path(a.q_dir) if a.q_dir else bc.arm_dir("Q36-8")
    paths = K.gate_text_paths(work_texts_dir=Path(a.texts_dir) if a.texts_dir else None)
    texts = {t: paths[t].read_text(encoding="utf-8") for t in TEXTS}
    rec = {"what": "exp_036 G3 diagnostic, descriptive", "k_dir": gc.redact_path(k_dir), "q_dir": gc.redact_path(q_dir),
           "texts": {t: {"bytes": len(texts[t].encode())} for t in TEXTS}}

    # Kolibri ids: the gate's committed / work-dir ids (identical to tokenising the .txt alone; checked)
    k_tok = k_dir / "tokenizer.json"
    k_ids, k_fb = {}, {}
    for t in TEXTS:
        k_ids[t], k_fb[t] = tokenize(k_tok, texts[t])
    if not a.no_dump:
        from gate import textset
        from gate.checks import ref_pass
        ts = textset.load_real()
        for t in TEXTS:
            if ts.ids[t] != k_ids[t]:
                raise SystemExit(f"{t}: tokenising the .txt does not give the gate ids")
        d = ref_pass.cache_dir("fp32", ts.key, gc.models_dir() / gc.BF16_DIR)
        dump = ref_pass.Dump(d)
        if not ref_pass.reusable(dump, gc.models_dir() / gc.BF16_DIR, ts.key):
            raise SystemExit(f"no complete reference dump at {gc.redact_path(d)}")
        rec["dump_record_sha256"] = gc.sha256_file(d / "record.json")
        logits = dump.load("logits.npy")
        L = ts.profile.text_len
        order = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
        ref_nll = {t: ref_pass.next_token_nll(logits[order.index(t) * L:(order.index(t) + 1) * L], ts.ids[t]) for t in TEXTS}
    else:
        ref_nll = None

    # ---- K8 (or stand-in) ----
    model, tok, _ = genutil.load_arm("K8", k_dir) if not a.k_dir else (*__import__("mlx_lm").load(str(k_dir)), k_dir)
    eot = K.context_prefix(k_dir)["id"]
    wrap = K.chat_wrapper(a.k_family, k_dir)["prompt_ids"]
    k_res = {}
    for t in TEXTS:
        ids = k_ids[t]
        nb = len(texts[t].encode())
        nb1 = nb - int(k_fb[t][1])  # bytes of tokens 1..T-1
        n0 = tf_nll(model, [], ids)
        ne = tf_nll(model, [eot], ids)
        nc = tf_nll(model, wrap, ids)
        k_res[t] = {"noprefix": n0, "eot": ne, "chat": nc}
        r = {"bpb_noprefix_gate_rule": n0.sum() / LN2 / nb1, "bpb_eot_all_tokens": ne.sum() / LN2 / nb,
             "bpb_eot_tokens1plus": ne[1:].sum() / LN2 / nb1, "bpb_chat_A5": nc.sum() / LN2 / nb,
             "frac_tokens_nll_lt_0.05_noprefix": float((n0 < 0.05).mean()),
             "top_first16_share_noprefix": float(n0[:16].sum() / n0.sum())}
        if ref_nll is not None:
            r["bpb_reference"] = ref_nll[t].sum() / LN2 / nb1
            r["k8_minus_ref_nats_per_token"] = float((n0 - ref_nll[t]).mean())
            fb = k_fb[t]
            for k in (16, 64):
                r[f"bpb_reference_from_token{k}"] = ref_nll[t][k - 1:].sum() / LN2 / (nb - int(fb[k]))
        rec["texts"][t]["kolibri"] = {k: float(v) for k, v in r.items()}
        print(t, "K", json.dumps(rec["texts"][t]["kolibri"]), flush=True)
    for name in ("noprefix", "eot", "chat"):
        tot = sum(k_res[t][name].sum() for t in TEXTS)
        byt = sum(len(texts[t].encode()) - (int(k_fb[t][1]) if name == "noprefix" else 0) for t in TEXTS)
        rec.setdefault("kolibri_pooled_bpb", {})[name] = tot / LN2 / byt
    # D4 recall probe on T4
    t4 = texts["T4"]
    probes = {}
    for name, p_ids, true in probe_points(t4, k_ids["T4"], k_fb["T4"]):
        g = greedy(model, p_ids, len(true))
        probes[name] = {"k_match_frac": float(np.mean([x == y for x, y in zip(g, true)])),
                        "k_prefix_exact_tokens": next((i for i, (x, y) in enumerate(zip(g, true)) if x != y), len(true)),
                        "k_greedy_text": tok.decode(g)}
    del model
    genutil.release()

    # ---- Q36-8 (or stand-in) ----
    from mlx_lm import load
    qm, _ = load(str(q_dir))
    q_eot = K.context_prefix(q_dir)["id"]
    q_tok = q_dir / "tokenizer.json"
    q_res = {}
    for t in TEXTS:
        ids, fb = tokenize(q_tok, texts[t])
        ne = tf_nll(qm, [q_eot], ids)
        n0 = tf_nll(qm, [], ids)
        nb = len(texts[t].encode())
        q_res[t] = (ids, fb, ne)
        rec["texts"][t]["q36_8"] = {"bpb_eot_all_tokens": ne.sum() / LN2 / nb,
                                     "bpb_noprefix_tokens1plus": n0.sum() / LN2 / (nb - int(fb[1])),
                                     "frac_tokens_nll_lt_0.05": float((ne < 0.05).mean())}
        print(t, "Q", json.dumps(rec["texts"][t]["q36_8"]), flush=True)
    q4_ids, q4_fb = tokenize(q_tok, t4)
    for name, p_ids, true in probe_points(t4, q4_ids, q4_fb):
        if name not in probes:
            continue
        g = greedy(qm, [q_eot] + p_ids, len(true))
        probes[name].update({"q_match_frac": float(np.mean([x == y for x, y in zip(g, true)])),
                             "q_prefix_exact_tokens": next((i for i, (x, y) in enumerate(zip(g, true)) if x != y), len(true))})
    rec["t4_recall_probe"] = probes
    del qm
    genutil.release()

    # ---- D3: where the gap lives (Kolibri reference/no-prefix vs Q36-8 eot), byte-aligned ----
    for t in TEXTS:
        b = texts[t].encode()
        nb = len(b)
        kn = k_res[t]["noprefix"] if ref_nll is None else ref_nll[t]
        bk = byte_bits(kn, k_fb[t], nb, skip_first=True)
        ids, fb, qn = q_res[t]
        bq = byte_bits(qn, fb, nb)
        ok = ~np.isnan(bk)
        W = 128
        wk = np.array([np.nansum(bk[i:i + W]) for i in range(0, nb, W)])
        wq = np.array([bq[i:i + W][ok[i:i + W]].sum() for i in range(0, nb, W)])
        gap = wk - wq
        order = np.argsort(-gap)
        top = order[:max(1, len(gap) // 10)]
        ratio = wk / np.maximum(wq, 1e-9)
        dig = np.frombuffer(b, dtype=np.uint8)
        is_dig = (dig >= 48) & (dig <= 57)
        recall = np.zeros(nb, dtype=bool)   # bytes inside 128-byte windows where Q36-8 is < 0.1 bpb
        for k_, i in enumerate(range(0, nb, W)):
            if wq[k_] / max(min(W, nb - i), 1) < 0.1:
                recall[i:i + W] = True
        g_all = np.nansum(bk) - bq[ok].sum()
        blk = {"gap_bits": float(g_all), "windows_128B": int(len(gap)),
               "top10pct_windows_share_of_gap": float(gap[top].sum() / gap.sum()) if gap.sum() else None,
               "windows_kolibri_better": int((gap < 0).sum()),
               "window_ratio_k_over_q_p10_p50_p90": [float(x) for x in np.percentile(ratio, [10, 50, 90])],
               "gap_share_digit_bytes": float((np.nansum(bk[is_dig & ok]) - bq[is_dig & ok].sum()) / g_all) if g_all else None,
               "digit_byte_share": float(is_dig.mean()),
               "gap_share_q_recall_windows": float((np.nansum(bk[recall & ok]) - bq[recall & ok].sum()) / g_all) if g_all else None,
               "q_recall_window_byte_share": float(recall.mean())}
        if t == "T4":
            arts = {}
            for name, s0, e0 in article_spans(texts[t]):
                m = ok.copy()
                m[:s0] = False
                m[e0:] = False
                by = m.sum()
                if by:
                    arts[name] = {"k_bpb": float(bk[m].sum() / by), "q_bpb": float(bq[m].sum() / by)}
            blk["per_article"] = arts
        rec["texts"][t]["gap"] = blk
        print(t, "gap", json.dumps({k: v for k, v in blk.items() if k != "per_article"}), flush=True)
    Path(a.out).write_text(json.dumps(rec, indent=1, default=float))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
