import os; os.environ.setdefault("MLX_ENABLE_TF32","0")
import mlx.core as mx, numpy as np, json
rng=np.random.default_rng(1)
H,E=2560,384
W=mx.array((rng.standard_normal((E,H))*0.02).astype(np.float32)).astype(mx.bfloat16).astype(mx.float32)
xb=mx.array(rng.standard_normal((8,1,H)).astype(np.float32)).astype(mx.bfloat16)
f=lambda x: x.astype(mx.float32) @ W.T
y8=f(xb); y1=mx.concatenate([f(xb[i:i+1]) for i in range(8)])
ref=np.array(xb.astype(mx.float32)).astype(np.float64) @ np.array(W).astype(np.float64).T
mx.eval(y8,y1)
a,b=np.array(y8),np.array(y1)
# gap statistics: how often does the M=8 vs M=1 difference flip a top-6 selection with a bias?
bias=rng.standard_normal(E).astype(np.float32)*1.0
def top6(z): return np.sort(np.argpartition(-(z+bias),5,axis=-1)[...,:6],axis=-1)
print(json.dumps({"bitwise_3d_B8_eq_B1": bool((a==b).all()), "maxabs": float(np.abs(a-b).max()),
 "rel8": float(np.linalg.norm(a-ref)/np.linalg.norm(ref)), "rel1": float(np.linalg.norm(b-ref)/np.linalg.norm(ref))}))
