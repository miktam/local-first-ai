import os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K, K + "/tests"]
from tools.precision import ensure_exact_fp32
print(ensure_exact_fp32())
from pathlib import Path
import numpy as np, mlx.core as mx, mlx.nn as nn
from gate import common
G5 = Path("$SCRATCH/gatediag_g5")

def load(name="ckpt_w513h128"):
    m, cfg, mod = common.load_port(G5 / name)
    m.set_dtype(mx.float32)
    mx.eval(m.parameters())
    return m, mod

def prefill_hidden(m, ids, step, lo=0, hi=None):
    hi = len(ids) - 1 if hi is None else hi
    cache = m.make_cache()
    ids = np.asarray(ids[:hi + 1], dtype=np.int32)
    rows = []
    for a in range(0, ids.size, step):
        b = min(ids.size, a + step)
        out = m.forward_hidden(mx.array(ids[a:b])[None], cache=cache)
        if b > lo:
            sel = out[0, max(lo, a) - a:b - a].astype(mx.float32)
            mx.eval(sel); rows.append(np.array(sel))
        else:
            mx.eval(out)
    return np.concatenate(rows, 0)

def decode_hidden(m, ids, lo, hi, step=2048):
    cache = m.make_cache()
    ids = np.asarray(ids, dtype=np.int32)
    for a in range(0, lo, step):
        mx.eval(m.forward_hidden(mx.array(ids[a:min(lo, a + step)])[None], cache=cache))
    rows = []
    for p in range(lo, hi + 1):
        out = m.forward_hidden(mx.array(ids[p:p + 1])[None], cache=cache)[0, -1].astype(mx.float32)
        mx.eval(out); rows.append(np.array(out))
    return np.stack(rows)

def rel(a, b):
    return np.linalg.norm(a - b, axis=-1) / (np.linalg.norm(b, axis=-1) + 1e-30)
