"""E2: batch-parity null on REAL weights on the M4 Pro, mlx_lm models (correct implementations),
through the gate's own batch loop (gate/checks/g5_generation.batch_parity, instrumented copy).

Per position of the 12 registered sequences (lengths 37..1,100, max_tokens 8..56, B=8,
mid-run admission):
  KL(single||batched)            the gate statistic (single = teacher-forced prefill, chunk 2048)
  KL(single||decode_tf)          decode-vs-prefill on these positions (B=1, plain cache)
  KL(decode_tf||batched)         the batching part
  KL(single2048||single64)       the chunking floor on these positions
  KL(truth||single), KL(truth||batched)   truth = same weights, fp32 activations, prefill
  decisive (truth lead >= 2) top-1 of single and batched vs truth; flips where single leads >= 2
Plus the identical-prompt localiser (B=8 copies of one prompt, no padding) vs B=1 BatchGenerator.
usage: e2_batch_null.py <model dir> <tag> [quant:none|q8] [wrap:0|1]"""
import os, sys, json, time
os.environ["MLX_ENABLE_TF32"] = "0"
K = "$KIT"
sys.path.insert(0, K)
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.generate import BatchGenerator
from runner.sampler import fp32_logits
from runner.generate import CACHE_LIMIT_BYTES, PREFILL_STEP_SIZE
from gate.checks import g5_generation as g5

MP, TAG = sys.argv[1], sys.argv[2]
QUANT = sys.argv[3] if len(sys.argv) > 3 else "none"
WRAP = len(sys.argv) > 4 and sys.argv[4] == "1"
TEXTS = json.load(open("texts.json"))
LENGTHS = (37, 300, 700, 1100, 64, 520, 900, 150, 37, 700, 300, 1100)
MAXTOK = (48, 8, 32, 16, 40, 24, 56, 12, 20, 36, 28, 44)
B = 8


class W(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.inner = m

    @property
    def layers(self):
        return self.inner.layers

    def make_cache(self):
        return make_prompt_cache(self.inner)

    def __call__(self, *a, **k):
        return self.inner(*a, **k)


def get(fp32):
    m, tok = load(MP)
    if QUANT == "q8":
        nn.quantize(m, group_size=64, bits=8)
    if fp32:
        m.set_dtype(mx.float32)
    mx.eval(m.parameters())
    return W(m), tok


model, tok = get(False)
truth, _ = get(True)
eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
wrap = []
if WRAP:
    from runner import chat
    from bench import kl_8v4
    _, wrap, _ = chat.render("gemma4", tok, [{"role": "user", "content": kl_8v4.WRAP_USER_MESSAGE}], kl_8v4.WRAP_EFFORT)
    wrap = list(wrap)
order = ("T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8")
streams = {t: wrap + tok.encode(TEXTS[t], add_special_tokens=False) for t in order}
prompts = [streams[order[j % 8]][:LENGTHS[j]] for j in range(12)]
lp, kl = g5._lp, g5.kl_lp


def gen_new(m, Bc, mt):
    mx.set_cache_limit(CACHE_LIMIT_BYTES)
    return BatchGenerator(fp32_logits(m), max_tokens=mt, stop_tokens=[[e] for e in eos],
                          sampler=lambda x: mx.argmax(x, axis=-1), completion_batch_size=Bc,
                          prefill_batch_size=min(Bc, 8), prefill_step_size=PREFILL_STEP_SIZE, max_kv_size=None)


def batched(m, prompts, max_tokens, Bc):
    gen = gen_new(m, Bc, max(max_tokens))
    got, queue, uid_of, in_flight = {}, list(range(len(prompts))), {}, 0
    try:
        while queue or in_flight:
            free = Bc - in_flight
            if free > 0 and queue:
                group = queue[:free]; del queue[:free]
                uids = gen.insert([list(prompts[j]) for j in group], max_tokens=[int(max_tokens[j]) for j in group])
                for u, j in zip(uids, group):
                    uid_of[u] = j; got[u] = {"tokens": [], "lp": []}
                in_flight += len(group)
            _, resps = gen.next()
            for r in resps:
                g = got[r.uid]; g["tokens"].append(int(r.token))
                x = r.logprobs.astype(mx.float32); mx.eval(x); g["lp"].append(np.array(x))
                if r.finish_reason is not None:
                    in_flight -= 1
    finally:
        gen.close()
    return [got[u] for u, j in sorted(uid_of.items(), key=lambda t: t[1])]


def prefill_lp(m, seq, pos, step):
    from gate.harness import logits_at
    return lp(logits_at(m, seq, pos, chunk=step))


def decode_tf(m, prompt, toks):
    cache = m.make_cache()
    ids = np.asarray(prompt, dtype=np.int32)
    if ids.size > 1:
        for a in range(0, ids.size - 1, 2048):
            mx.eval(m(mx.array(ids[a:min(ids.size - 1, a + 2048)])[None], cache=cache))
    rows, cur = [], int(ids[-1])
    for t in toks:
        o = m(mx.array([[cur]], dtype=mx.int32), cache=cache)[0, -1].astype(mx.float32); mx.eval(o)
        rows.append(np.array(o)); cur = t
    return lp(np.stack(rows))


t0 = time.time()
res = batched(model, prompts, MAXTOK, B)
rows = []
for j, g in enumerate(res):
    p = prompts[j]; toks = g["tokens"]; seq = list(p) + toks[:-1]
    pos = np.arange(len(p) - 1, len(p) - 1 + len(toks))
    s = prefill_lp(model, seq, pos, 2048); s64 = prefill_lp(model, seq, pos, 64)
    tr = prefill_lp(truth, seq, pos, 2048)
    b = lp(np.stack(g["lp"])); d = decode_tf(model, p, toks)
    lead_t = np.sort(tr, -1)[:, -1] - np.sort(tr, -1)[:, -2]
    lead_s = np.sort(s, -1)[:, -1] - np.sort(s, -1)[:, -2]
    for i in range(len(toks)):
        rows.append({"seq": j, "t": i, "first_wave": j < 8, "plen": len(p),
                     "kl_sb": float(kl(s[i:i+1], b[i:i+1])[0]), "kl_sd": float(kl(s[i:i+1], d[i:i+1])[0]),
                     "kl_db": float(kl(d[i:i+1], b[i:i+1])[0]), "kl_floor": float(kl(s[i:i+1], s64[i:i+1])[0]),
                     "kl_ts": float(kl(tr[i:i+1], s[i:i+1])[0]), "kl_tb": float(kl(tr[i:i+1], b[i:i+1])[0]),
                     "lead_t": float(lead_t[i]), "lead_s": float(lead_s[i]),
                     "s_ok": bool(s[i].argmax() == tr[i].argmax()), "b_ok": bool(b[i].argmax() == tr[i].argmax()),
                     "sb_flip": bool(s[i].argmax() != b[i].argmax())})
A = {k: np.array([r[k] for r in rows]) for k in rows[0]}
dec = A["lead_t"] >= 2.0; sdec = A["lead_s"] >= 2.0
out = {"model": MP, "tag": TAG, "quant": QUANT, "wrap": WRAP, "device": mx.device_info()["device_name"],
       "n_positions": int(A["kl_sb"].size), "wall_s": time.time() - t0,
       "mean": {k: float(A[k].mean()) for k in ("kl_sb", "kl_sd", "kl_db", "kl_floor", "kl_ts", "kl_tb")},
       "median": {k: float(np.median(A[k])) for k in ("kl_sb", "kl_sd", "kl_db", "kl_floor", "kl_ts", "kl_tb")},
       "top1_dis_sb": float(A["sb_flip"].mean()),
       "n_dec_truth": int(dec.sum()), "single_dec_top1": float(A["s_ok"][dec].mean()), "batched_dec_top1": float(A["b_ok"][dec].mean()),
       "flips_where_single_leads_2": int((A["sb_flip"] & sdec).sum()), "n_single_leads_2": int(sdec.sum()),
       "frac_decisive_truth": float(dec.mean()),
       "top5_share_kl_sb": float(np.sort(A["kl_sb"])[-5:].sum() / A["kl_sb"].sum()),
       "by_class": {}}
for name, msk in (("first_wave", A["first_wave"]), ("mid_run", ~A["first_wave"]), ("t0", A["t"] == 0),
                  ("t1_2", (A["t"] >= 1) & (A["t"] <= 2)), ("t3plus", A["t"] >= 3), ("plen_ge_513", A["plen"] >= 513)):
    out["by_class"][name] = {"n": int(msk.sum()), **{k: float(A[k][msk].mean()) for k in ("kl_sb", "kl_sd", "kl_db", "kl_ts", "kl_tb")}}
# identical-prompt localiser: B=8 copies of prompts[2] (no padding) vs B=1 BatchGenerator
p = prompts[2]
r8 = batched(model, [p] * 8, [16] * 8, 8)
r1 = batched(model, [p], [16], 1)
n = min(len(r1[0]["lp"]), min(len(r["lp"]) for r in r8))
b1 = lp(np.stack(r1[0]["lp"][:n]))
out["identical_B8_vs_B1_mean_kl"] = float(np.mean([kl(b1, lp(np.stack(r["lp"][:n]))).mean() for r in r8]))
out["identical_B8_rows_bitwise_equal"] = bool(all(np.array_equal(np.stack(r8[0]["lp"][:n]), np.stack(r["lp"][:n])) for r in r8))
d1 = decode_tf(model, p, r1[0]["tokens"][:n])
out["B1_batchgen_vs_decode_tf_mean_kl"] = float(kl(d1, b1).mean())
json.dump(out, open(f"e2_{TAG}.json", "w"), indent=1)
print(json.dumps(out, indent=1))
