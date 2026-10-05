"""Exact visibility check of the G5 batched path (the gate's own _batch_generator + batch_parity admission loop,
mlx_lm BatchGenerator, the port's make_cache and make_masks): every token id encodes (prompt, position); a probe
model replaces attention with a recorder. For every query of every row, at every prefill and decode step, the
visible keys must be exactly the row's own tokens in [i-512, i] (sliding) / [0, i] (full), never a pad."""
from common_v import *
from gate import common
from gate.checks import g5_generation as g5
mod = common.port_module()
args = mod.ModelArgs(model_type="kolibri1", hidden_size=8, num_hidden_layers=2, num_attention_heads=1,
                     num_key_value_heads=1, head_dim=8, vocab_size=65536, num_experts=2, num_experts_per_tok=1,
                     moe_intermediate_size=8, shared_expert_intermediate_size=8, sliding_window=513,
                     layer_types=["sliding_attention", "full_attention"], rope_theta=10000.0, use_sliding_window=True)
port = mod.Model(args)
BASE = 5000
STATS = {"queries": 0, "bad": 0, "examples": [], "calls": 0, "max_B": 0}
class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = port.layers
    def make_cache(self):
        return port.make_cache()
    def __call__(self, inputs, cache=None, **kw):
        STATS["calls"] += 1
        B, N = inputs.shape; STATS["max_B"] = max(STATS["max_B"], B)
        h = mx.zeros((B, N, 8))
        fa, swa = port.model.make_masks(h, cache)
        x = inputs.astype(mx.float32)[:, None, :, None]  # [B,1,N,1] position codes
        ids_np = np.array(inputs)
        for li, (c, mask, win) in enumerate(((cache[0], swa, 513), (cache[1], fa, None))):
            k, _ = c.update_and_fetch(x, x)
            kk = np.array(k)[:, 0, :, 0].astype(np.int64)  # [B, kL]
            kL = kk.shape[1]
            if mask is None:
                al = np.ones((B, N, kL), bool)
            elif isinstance(mask, str):
                al = np.broadcast_to(np.arange(kL)[None, None] <= (kL - N) + np.arange(N)[None, :, None], (B, N, kL))
            else:
                a = np.array(mask); a = a.reshape((a.shape[0] if a.ndim == 4 else 1,) + a.shape[-2:])
                al = np.broadcast_to(a, (B, N, kL))
            for b in range(B):
                for q in range(N):
                    tid = int(ids_np[b, q])
                    if tid == 0:
                        continue  # right-pad query (output unused)
                    base = (tid - 1) // BASE * BASE + 1; pos = tid - base
                    lo = 0 if win is None else max(0, pos - win + 1)
                    got = np.sort(kk[b][al[b, q]]); exp = np.arange(base + lo, base + pos + 1)
                    STATS["queries"] += 1
                    if got.shape != exp.shape or (got != exp).any():
                        STATS["bad"] += 1
                        if len(STATS["examples"]) < 5:
                            STATS["examples"].append({"layer": li, "B": B, "N": N, "row": b, "pos": pos, "n_got": int(got.size),
                                                      "n_exp": int(exp.size), "extra": [int(v) for v in np.setdiff1d(got, exp)[:5]],
                                                      "missing": [int(v) for v in np.setdiff1d(exp, got)[:5]]})
        nxt = ids_np[:, -1] + 1
        lg = np.full((B, 1, 65536), -1e9, np.float32); lg[np.arange(B), 0, nxt] = 0.0
        return mx.array(lg)
LEN = [37, 300, 700, 1100, 64, 520, 900, 150, 37, 700, 300, 1100]
MT = [48, 8, 32, 16, 40, 24, 56, 12, 20, 36, 28, 44]
prompts = [list(range(j * BASE + 1, j * BASE + 1 + n)) for j, n in enumerate(LEN)]

# the gate's batch_parity admission loop (gate/checks/g5_generation.py l.199-229), without the single comparison
gen, path = g5._batch_generator(Probe(), 8, [65535], max(MT))
got, queue, uid_of, in_flight, step = {}, list(range(len(prompts))), {}, 0, 0
try:
    while queue or in_flight:
        free = 8 - in_flight
        if free > 0 and queue:
            group = queue[:free]; del queue[:free]
            uids = gen.insert([list(prompts[j]) for j in group], max_tokens=[int(MT[j]) for j in group])
            for u, j in zip(uids, group):
                uid_of[u] = j; got[u] = []
            in_flight += len(group)
        _, resps = gen.next()
        for r in resps:
            got[r.uid].append(int(r.token))
            if r.finish_reason is not None:
                in_flight -= 1
        step += 1
finally:
    gen.close()
# generated tokens must continue each prompt's own code (id + 1 each step)
cont_ok = all(got[u] == list(range(prompts[j][-1] + 1, prompts[j][-1] + 1 + len(got[u]))) and len(got[u]) == MT[j] for u, j in uid_of.items())
print(f"path: {path}; steps {step}; model calls {STATS['calls']} (max B {STATS['max_B']}); queries checked {STATS['queries']}; "
      f"wrong visible sets {STATS['bad']}; continuations follow own prompt: {cont_ok}")
print(STATS["examples"])
