"""The gate's batch_parity (B = 8, profile lengths, mid-run admission) in fp32 on the tiny head_dim-128 model,
kit port vs the vLLM-angle RoPE scratch port: the RoPE change must keep the batched path exact (array offsets)."""
from common_v import *
from pathlib import Path
from gate import common
from gate.checks import g5_generation as g5
V = Path(__file__).resolve().parent
LEN = [37, 300, 700, 1100, 64, 520, 900, 150, 37, 700, 300, 1100]; MT = [48, 8, 32, 16, 40, 24, 56, 12, 20, 36, 28, 44]
rng = np.random.default_rng(1); texts = [rng.integers(1, 1000, size=1100).tolist() for _ in range(8)]
prompts = [texts[j % 8][:LEN[j]] for j in range(12)]
for variant in ("kit_base", "kit_rope"):
    pf = V / variant / "port" / "kolibri1.py"
    m, _, _ = common.load_port(V / "ckv_s1", port_file=pf, module=common.port_module(None, pf))
    for dt in ("fp32", "bf16"):
        if dt == "fp32": m.set_dtype(mx.float32)
        mx.eval(m.parameters())
        r = g5.batch_parity(m, prompts, MT, 8, eos=[1022])
        print(f"{variant} {dt}: batch parity mean KL {r['mean_kl']:.2e} top1_dis {r['top1_dis']:.3f} n {r['n_positions']} admitted mid-run {r['admitted_mid_run']}", flush=True)
        if dt == "fp32":
            m, _, _ = common.load_port(V / "ckv_s1", port_file=pf, module=common.port_module(None, pf))
