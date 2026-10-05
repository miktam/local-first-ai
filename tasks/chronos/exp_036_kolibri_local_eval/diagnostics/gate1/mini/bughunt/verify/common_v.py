import os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K, K + "/tests"]
from tools.precision import ensure_exact_fp32
PREC = ensure_exact_fp32()
import numpy as np, mlx.core as mx, mlx.nn as nn
