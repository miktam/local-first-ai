"""Does a CORRECT port/reference pair show findings (1) and (2) at long context
when the MoE router has many near-ties (as Kolibri's 384-expert top-6 router
does)? And does exact RoPE remove them?

Tiny head_dim-128, window-513 model, fp32, with: E experts top-k, small
expert_bias, sharper attention (q/k norm weights x QK), confident head (x HEAD).
Runs (all fp32, same weights):
  port prefill 2048, port prefill 64, port decode (15000..15300),
  reference fp32 (numpy), and forced-routing replays.
Routing ids are recorded per layer and position.

usage: r3_chaos.py out.json E K QK HEAD BIASSTD [random|realtok] [exact]
"""
import json
import sys
import time
from pathlib import Path

KIT = "$KIT"
sys.path.insert(0, KIT)
sys.path.insert(0, KIT + "/tests")
from tools.precision import ensure_exact_fp32

print(ensure_exact_fp32(), flush=True)
import mlx.core as mx
import mlx.nn as nn
import numpy as np

from reference import kolibri_ref as KR
from tiny_checkpoint import write_tiny_checkpoint, random_ids, _layer_types
from port_harness import load_port

OUT = Path(sys.argv[1])
E, K, QK, HEAD, BSTD = int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
SEQ = sys.argv[7] if len(sys.argv) > 7 else "random"
ROPE = sys.argv[8] if len(sys.argv) > 8 else "mlx"  # mlx | exact (port+ref float64 cos/sin) | freqs (port fast.rope freqs=1/inv_ref; ref unchanged)
EXACT = ROPE == "exact"
WSEED = int(sys.argv[9]) if len(sys.argv) > 9 else 11
SSEED = int(sys.argv[10]) if len(sys.argv) > 10 else 3
NL = 10
HERE = Path(__file__).resolve().parent
T = 15301
WINS = {"early": (520, 1100), "T9": (15000, 15300)}
THETA, D = 10000.0, 128

ck = HERE / f"ckpt_chaos_E{E}_K{K}_qk{QK}_h{HEAD}_b{BSTD}_w{WSEED}"


def edit(t):
    for name in list(t):
        if name.endswith("q_norm.weight") or name.endswith("k_norm.weight"):
            t[name] = t[name] * QK
        if name.endswith("expert_bias"):
            t[name] = t[name] * (BSTD / 2.0)
        if name == "lm_head.weight":
            t[name] = t[name] * HEAD


if not (ck / "config.json").exists():
    write_tiny_checkpoint(ck, seed=WSEED, preset="w513", edit_tensors=edit, overrides={
        "head_dim": 128, "num_attention_heads": 12, "num_key_value_heads": 4,
        "num_hidden_layers": NL, "layer_types": _layer_types(NL, 5),
        "num_experts": E, "num_experts_per_tok": K, "moe_intermediate_size": 64,
        "max_position_embeddings": 32768})

if SEQ == "random":
    ids = np.array(random_ids(T, seed=SSEED), dtype=np.int32)
else:
    ids = np.load(HERE / "realtok_ids.npy")[:T].astype(np.int32)


def lsm(x):
    x = np.asarray(x, np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def kl(p, q):
    return (np.exp(p) * (p - q)).sum(-1)


def pair(p, q, lo):
    k = kl(p, q)
    s = np.sort(p, -1)
    ld = s[:, -1] - s[:, -2]
    ch = p.argmax(-1) != q.argmax(-1)
    return {"mean_kl": float(k.mean()), "max_kl": float(k.max()), "argmax_pos": int(lo + k.argmax()),
            "n_gt_1e-6": int((k > 1e-6).sum()), "top1_changes": int(ch.sum()),
            "top1_changes_lead_ge_2": int((ch & (ld >= 2.0)).sum())}


inv64 = THETA ** (-np.arange(0, D, 2, dtype=np.float64) / D)


def exact_cs(positions):
    a = np.asarray(positions, np.float64)[:, None] * inv64[None, :]
    return np.cos(a).astype(np.float32), np.sin(a).astype(np.float32)


class ExactRoPE(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def __call__(self, x, offset=0):
        c, s = exact_cs(np.arange(int(offset), int(offset) + x.shape[-2]))
        c, s = mx.array(c), mx.array(s)
        h = self.dims // 2
        x1, x2 = x[..., :h], x[..., h:]
        return mx.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], axis=-1).astype(x.dtype)


class FreqsRoPE(nn.Module):
    """Candidate port fix: mx.fast.rope with explicit freqs = 1/inv_freq(vLLM),
    so the kernel's angle is fp32(pos * inv_freq) with vLLM's inv_freq."""

    def __init__(self, dims, theta):
        super().__init__()
        self.dims = dims
        self._freqs = mx.array((1.0 / KR.rope_inv_freq(dims, theta).astype(np.float64)).astype(np.float32))

    def __call__(self, x, offset=0):
        return mx.fast.rope(x, self.dims, traditional=False, base=None, scale=1.0, offset=offset, freqs=self._freqs)


class RefExact(KR.KolibriReference):
    def rope(self, x, positions):
        c, s = exact_cs(positions)
        c, s = c[:, None, :], s[:, None, :]
        h = x.shape[-1] // 2
        x1, x2 = x[..., :h], x[..., h:]
        return self._store(np.concatenate([x1 * c - x2 * s, x2 * c + x1 * s], -1).astype(np.float32))


class Recorder:
    """Wraps each layer's mlp.forward_routed: records ids, optionally forces them."""

    def __init__(self, model, force=None):
        self.ids = [[] for _ in model.layers]
        self.pos = [0] * len(model.layers)
        self.force = force
        for i, layer in enumerate(model.layers):
            orig = type(layer.mlp).forward_routed.__get__(layer.mlp)

            def fr(x, force_ids=None, i=i, orig=orig):
                L = x.shape[1]
                if self.force is not None:
                    f = self.force[i][self.pos[i]:self.pos[i] + L]
                    force_ids = mx.array(f[None].astype(np.uint32))
                y, lg, sel = orig(x, force_ids)
                self.ids[i].append(sel)
                self.pos[i] += L
                return y, lg, sel

            layer.mlp.forward_routed = fr

    def routing(self):
        return np.stack([np.concatenate([np.array(a[0]) for a in L], 0) for L in self.ids])


def port_prefill(model, step, force=None):
    rec = Recorder(model, force)
    cache = model.make_cache()
    rows = []
    for a in range(0, T, step):
        out = model(mx.array(ids[a:a + step])[None], cache=cache)[0].astype(mx.float32)
        mx.eval(out, *[x for L in rec.ids for x in L[-1:]])
        rows.append(np.array(out))
    for layer in model.layers:  # unwrap
        del layer.mlp.forward_routed
    return np.concatenate(rows, 0), rec.routing()


def port_decode(model, lo, hi):
    cache = model.make_cache()
    for a in range(0, lo, 2048):
        mx.eval(model(mx.array(ids[a:min(lo, a + 2048)])[None], cache=cache))
    rows = []
    for p in range(lo, hi + 1):
        out = model(mx.array(ids[p:p + 1])[None], cache=cache)[0, -1].astype(mx.float32)
        mx.eval(out)
        rows.append(np.array(out))
    return np.stack(rows)


res = {"E": E, "K": K, "QK": QK, "HEAD": HEAD, "bias_std": BSTD, "seq": SEQ, "rope": ROPE, "wseed": WSEED, "sseed": SSEED, "ckpt": str(ck)}
model = load_port(ck, float32=True)
if ROPE != "mlx":
    for layer in model.layers:
        if layer.self_attn.rope is not None:
            layer.self_attn.rope = ExactRoPE(D) if EXACT else FreqsRoPE(D, THETA)
t0 = time.time()
L2048, R2048 = port_prefill(model, 2048)
L64, R64 = port_prefill(model, 64)
lo9, hi9 = WINS["T9"]
Ldec = port_decode(model, lo9, hi9)
print(f"port {time.time() - t0:.0f}s", flush=True)
ref = (RefExact if EXACT else KR.KolibriReference)(str(ck))
t0 = time.time()
Lref = ref.forward(ids)
Rref = np.stack([np.asarray(r[1]).reshape(T, -1) for r in ref.last_routing])
print(f"ref {time.time() - t0:.0f}s", flush=True)
# forced replays: port prefill 64 forced to the prefill-2048 routing; port prefill 2048 forced to ref routing
L64f, _ = port_prefill(model, 64, force=R2048)
L2048fr, _ = port_prefill(model, 2048, force=Rref)

lp = {k: lsm(v) for k, v in {"p2048": L2048, "p64": L64, "ref": Lref, "p64_forced_to_p2048_routing": L64f,
                              "p2048_forced_to_ref_routing": L2048fr}.items()}
pairs = {}
for w, (lo, hi) in WINS.items():
    s = slice(lo, hi + 1)
    pairs[w] = {
        "p2048_vs_p64": pair(lp["p2048"][s], lp["p64"][s], lo),
        "ref_vs_p2048": pair(lp["ref"][s], lp["p2048"][s], lo),
        "ref_vs_p64": pair(lp["ref"][s], lp["p64"][s], lo),
        "p2048_vs_p64_forced_to_p2048_routing": pair(lp["p2048"][s], lp["p64_forced_to_p2048_routing"][s], lo),
        "ref_vs_p2048_forced_to_ref_routing": pair(lp["ref"][s], lp["p2048_forced_to_ref_routing"][s], lo),
    }
pairs["T9"]["p2048_vs_decode"] = pair(lp["p2048"][lo9:hi9 + 1], lsm(Ldec), lo9)


def routing_diff(A, B):
    d = (np.sort(A, -1) != np.sort(B, -1)).any(-1)  # [layers, T]
    first = [int(np.argmax(r)) if r.any() else None for r in d]
    anyl = d.any(0)
    return {"n_flipped_layer_positions": int(d.sum()), "per_layer": [int(x) for x in d.sum(1)],
            "first_flip_pos_per_layer": first, "first_flip_any_layer": int(np.argmax(anyl)) if anyl.any() else None,
            "flipped_positions_lt_1101": int(anyl[:1101].sum()), "flipped_positions_ge_14000": int(anyl[14000:].sum())}


res["routing_p2048_vs_p64"] = routing_diff(R2048, R64)
res["routing_ref_vs_p2048"] = routing_diff(Rref, R2048)
res["pairs"] = pairs
OUT.write_text(json.dumps(res, indent=1))
print(json.dumps({k: res[k] for k in ("routing_p2048_vs_p64", "routing_ref_vs_p2048")}, indent=0)[:1500])
for w, pp in pairs.items():
    for k, v in pp.items():
        print(f"{w:6s} {k:40s} mean {v['mean_kl']:.3e} max {v['max_kl']:.3e} @{v['argmax_pos']} n>1e-6 {v['n_gt_1e-6']} top1ch {v['top1_changes']} (lead>=2 {v['top1_changes_lead_ge_2']})")
