import os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K, K + "/tests"]
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.nn as nn
from gate import common
from gate.checks import g5_generation as g5
from gate.textset import Profile
prof = Profile()
for ck in ("ckpt_w513h128", "ckpt_p5h128"):
    for bits in (8, 4):
        m, _, _ = common.load_port(Path(ck))
        nn.quantize(m, group_size=64, bits=bits, class_predicate=m.quant_predicate)
        m.set_dtype(mx.float32)
        rng = np.random.default_rng(1000)
        texts = [rng.integers(0, m.args.vocab_size - 16, size=1536).tolist() for _ in range(8)]
        prompts = [texts[j % 8][: prof.batch_lengths[j]] for j in range(12)]
        bp = g5.batch_parity(m, prompts, list(prof.batch_max_tokens), 8)
        print(ck, f"q{bits} fp32-act", "lm_head", type(m.lm_head).__name__, m.lm_head.scales.dtype, "mean_kl", bp["mean_kl"], "dis", bp["top1_dis"], "mid", bp["admitted_mid_run"])
