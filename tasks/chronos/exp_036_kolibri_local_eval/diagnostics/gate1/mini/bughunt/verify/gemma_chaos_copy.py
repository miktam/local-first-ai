"""Is fp32 long-context MoE inference in MLX chaotic at the level stage 3 saw?
A correct, mature MLX MoE (mlx_lm gemma4 26B-A4B, OptiQ 4-bit weights,
activations fp32 via set_dtype, MLX_ENABLE_TF32=0) on the real T9 text
(re-tokenised), three runs in lockstep:
  A  prefill step 2048
  B  prefill step 64                       (stage 3's chunk-64 test, same code)
  C  prefill step 2048 with one fp32 weight of layer 0's input norm changed by
     2^-22 relative (a pure rounding-size perturbation, no code path change)
Per position: KL(A||B), KL(A||C), top-1 changes, A's top-1 lead; document starts.
Nothing here writes text or ids anywhere but this scratch directory."""
import os, sys, json, time
from pathlib import Path
K = "$KIT"
sys.path.insert(0, K)
from tools.precision import ensure_exact_fp32
print(ensure_exact_fp32(), flush=True)
import numpy as np
import mlx.core as mx
from mlx_lm import load

HERE = Path(__file__).resolve().parent
T = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
MP = [p for p in Path.home().glob(".cache/huggingface/hub/models--mlx-community--gemma-4-26B-A4B-it-OptiQ-4bit/snapshots/*")][0]
model, tok = load(str(MP))
model.set_dtype(mx.float32)
mx.eval(model.parameters())

text = json.load(open(HERE.parent / "gatediag_rev/texts.json"))["T9"]
enc = tok._tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
ids = np.array(enc["input_ids"], dtype=np.int32)[:T]
offs = enc["offset_mapping"][:T]
# document starts: char offsets of the Kolibri '\n\n' separators (id 263) in the decoded text
from tokenizers import Tokenizer
ktok = Tokenizer.from_file("$EXP036_MODELS/kolibri/Kolibri-1-BF16/tokenizer.json")
k9 = json.load(open(HERE.parent / "work/gate_texts/T9.ids.json"))["ids"]
char_starts = [len(ktok.decode(k9[:p + 1])) for p in np.flatnonzero(np.array(k9) == 263)]
tok_starts = []
for c in char_starts:
    j = next((i for i, (a, b) in enumerate(offs) if a >= c), None)
    if j is not None and j < T:
        tok_starts.append(int(j))
print("T", ids.size, "doc starts (gemma tokens)", tok_starts, flush=True)

lm = model.language_model.model
w = lm.layers[0].input_layernorm.weight
w_orig = mx.array(w)
w_pert = mx.array(np.array(w_orig))
w_pert[7] = w_pert[7] * (1.0 + 2.0 ** -22)
mx.eval(w_pert)
print("perturbed weight delta", float((w_pert - w_orig).abs().max()), flush=True)


def set_w(x):
    lm.layers[0].input_layernorm.weight = x


def lsm(x):
    x = x.astype(mx.float32)
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def kl_rows(lp, lq):
    return (mx.exp(lp) * (lp - lq)).sum(-1)


cA, cB, cC = model.make_cache(), model.make_cache(), model.make_cache()
klAB = np.zeros(ids.size); klAC = np.zeros(ids.size)
chAB = np.zeros(ids.size, bool); chAC = np.zeros(ids.size, bool); lead = np.zeros(ids.size)
BIG, SMALL = 2048, 64
t0 = time.time()
for a in range(0, ids.size, BIG):
    b = min(ids.size, a + BIG)
    x = mx.array(ids[a:b])[None]
    set_w(w_orig)
    lA = lsm(model(x, cache=cA)[0]); mx.eval(lA)
    top2 = mx.topk(lA, 2, axis=-1)
    lead[a:b] = np.abs(np.array(top2[:, 1] - top2[:, 0]))
    aA = np.array(lA.argmax(-1))
    for s in range(a, b, SMALL):
        e = min(b, s + SMALL)
        lB = lsm(model(mx.array(ids[s:e])[None], cache=cB)[0])
        k = kl_rows(lA[s - a:e - a], lB); am = lB.argmax(-1); mx.eval(k, am)
        klAB[s:e] = np.array(k); chAB[s:e] = np.array(am) != aA[s - a:e - a]
    set_w(w_pert)
    lC = lsm(model(x, cache=cC)[0])
    k = kl_rows(lA, lC); am = lC.argmax(-1); mx.eval(k, am)
    klAC[a:b] = np.array(k); chAC[a:b] = np.array(am) != aA
    set_w(w_orig)
    del lA, lC
    print(f"[{a}:{b}] {time.time() - t0:.0f}s  KL(A||B) mean {klAB[a:b].mean():.3e} max {klAB[a:b].max():.3e}  "
          f"KL(A||C) mean {klAC[a:b].mean():.3e} max {klAC[a:b].max():.3e}", flush=True)

np.savez(HERE / f"gemma_chaos_{T}.npz", klAB=klAB, klAC=klAC, chAB=chAB, chAC=chAC, lead=lead, starts=np.array(tok_starts))
dec = lead >= 2.0


def summ(k, ch, m):
    if not m.any():
        return None
    return {"mean_kl": float(k[m].mean()), "max_kl": float(k[m].max()), "top1_changes": int(ch[m].sum()),
            "decisive_top1_changes": int((ch & dec)[m].sum()), "n": int(m.sum())}


pos = np.arange(ids.size)
rep = {"model": str(MP.name), "T": int(ids.size), "doc_starts": tok_starts}
for name, (lo, hi) in {"0-2048": (0, 2048), "520-1100": (520, 1101), "2048-8192": (2048, 8192),
                       "8192-16384": (8192, 16384), "15000-15300": (15000, 15301)}.items():
    m = (pos >= lo) & (pos < hi)
    if m.any():
        rep[name] = {"A_vs_B_prefill64": summ(klAB, chAB, m), "A_vs_C_perturbed": summ(klAC, chAC, m)}
win = np.zeros(ids.size, bool)
for s in tok_starts:
    win[max(0, s - 32):s + 32] = True
far = ~win & (pos >= 2048)
rep["doc_windows"] = {"A_vs_B_prefill64": summ(klAB, chAB, win), "A_vs_C_perturbed": summ(klAC, chAC, win)}
rep["non_doc_ge2048"] = {"A_vs_B_prefill64": summ(klAB, chAB, far), "A_vs_C_perturbed": summ(klAC, chAC, far)}
rep["per_doc_start_window_max_kl_AC"] = {int(s): float(klAC[max(0, s - 32):s + 32].max()) for s in tok_starts}
rep["per_doc_start_window_max_kl_AB"] = {int(s): float(klAB[max(0, s - 32):s + 32].max()) for s in tok_starts}
print(json.dumps(rep, indent=1))
(HERE / f"gemma_chaos_{T}.json").write_text(json.dumps(rep, indent=1))
