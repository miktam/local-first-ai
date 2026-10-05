"""Valid rows of the first-wave prefill must not depend on the pad token id (mlx_lm pads with 0).
Prefill the 8 first-wave prompts right-padded with id 0 and with id 7; compare the valid KV and
the t0 logprobs bitwise."""
import sys; sys.path.insert(0, ".")
import mk
a = mk.parser().parse_args()
mx, np = mk.mx, mk.np
import importlib; G = importlib.import_module("mlx_lm.generate")
from mlx_lm.generate import BatchGenerator
model, module = mk.build(a)
rng = np.random.default_rng(1)
lens = [37, 300, 700, 1100, 64, 520, 900, 150]
prompts = [rng.integers(1, a.V - 16, size=n).tolist() for n in lens]
orig = G._right_pad_prompts
def run(pad_id):
    G._right_pad_prompts = lambda p, max_length=None: mx.array([q + [pad_id] * ((max_length or max(len(r) for r in p)) - len(q)) for q in p])
    try:
        gen = BatchGenerator(model, max_tokens=4, stop_tokens=[[a.V - 2]], sampler=lambda x: mx.argmax(x, -1),
                             completion_batch_size=8, prefill_batch_size=8, prefill_step_size=2048)
        gen.insert(prompts, max_tokens=[4] * 8)
        lps = {}
        for _ in range(6):
            _, rs = gen.next()
            for r in rs:
                x = r.logprobs.astype(mx.float32); mx.eval(x); lps.setdefault(r.uid, []).append(np.array(x))
        gen.close()
        return lps
    finally:
        G._right_pad_prompts = orig
A, B = run(0), run(7)
for u in sorted(A):
    a_, b_ = np.stack(A[u]), np.stack(B[u])
    print("row", u, "len", lens[u], "steps", len(a_), "bitwise", bool(np.array_equal(a_, b_)), "max|d|", float(np.abs(a_ - b_).max()))
