"""K8 quantized matmul and gather_qmm with fp32 activations (the bug test's arm) at M = 1, 64, 965, 2048 rows:
precision vs float64 of the dequantised weights, and whether rows are bitwise equal across M."""
from common_v import *
from mlx_lm.models.switch_layers import SwitchGLU
rng = np.random.default_rng(3)
def rel(a, b): return float(np.linalg.norm(a - b) / np.linalg.norm(b))
for (N, Kd, nm) in ((6144, 2560, "q_proj"), (2560, 6144, "o_proj"), (512, 2560, "shared_gate")):
    w = mx.array((rng.standard_normal((N, Kd)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
    wq, s, b = mx.quantize(w, group_size=64, bits=8)
    deq = np.array(mx.dequantize(wq, s, b, group_size=64, bits=8).astype(mx.float32)).astype(np.float64)
    X = mx.array(rng.standard_normal((2048, Kd)).astype(np.float32))
    t = np.array(X).astype(np.float64) @ deq.T
    ys = {}
    for M in (1, 64, 965, 2048):
        ys[M] = np.concatenate([np.array(mx.quantized_matmul(X[a:a + M], wq, s, b, transpose=True, group_size=64, bits=8)) for a in range(0, 2048 if M > 1 else 64, M)])
    print(nm, " ".join(f"M={M}: rel {rel(y, t[:y.shape[0]]):.1e}" for M, y in ys.items()),
          "| M=64 rows == M=2048 rows:", np.array_equal(ys[64], ys[2048]), "| M=965 == M=2048:", np.array_equal(ys[965][:965], ys[2048][:965]))
# SwitchGLU (sorted gather_qmm at prefill sizes), fp32 activations, 8-bit
E, K, H, I = 384, 6, 2560, 512
sg = SwitchGLU(H, I, E); sg.set_dtype(mx.bfloat16); nn.quantize(sg, group_size=64, bits=8); sg.set_dtype(mx.float32)
# keep scales bf16 as the K8 model: cast scales/biases back
def fix(m):
    for name in ("gate_proj", "up_proj", "down_proj"):
        l = getattr(m, name); l.scales = l.scales.astype(mx.bfloat16); l.biases = l.biases.astype(mx.bfloat16)
fix(sg); mx.eval(sg.parameters())
X = mx.array(rng.standard_normal((1, 2048, H)).astype(np.float32))
idx = mx.array(np.stack([rng.choice(E, K, replace=False) for _ in range(2048)]).astype(np.uint32))[None]
full = np.array(sg(X, idx))
for M in (1, 64, 965):
    n = 64 if M == 1 else 2048
    y = np.concatenate([np.array(sg(X[:, a:a + M], idx[:, a:a + M])) for a in range(0, n, M)], axis=1)
    print(f"SwitchGLU K8 fp32 act: chunk {M} vs 2048 rows max|d| {np.abs(y - full[:, :y.shape[1]]).max():.2e} rel {rel(y, full[:, :y.shape[1]]):.1e} bitwise {np.array_equal(y, full[:, :y.shape[1]])}")
