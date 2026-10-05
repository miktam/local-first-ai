import os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K, K + "/tests"]
from pathlib import Path
import mlx.core as mx
from gate import common
from gate.checks import g5_generation as g5
model, _, _ = common.load_port(Path("ckpt_w513h128"))
gen, path = g5._batch_generator(model, 8, common.EOS_IDS, 8)
print("path:", path, "| wrapper has make_cache:", hasattr(gen.model, "make_cache"))
c = gen._make_new_cache()
print("per-sequence caches:", sorted({(type(x).__name__, getattr(x, 'max_size', None), getattr(x, 'keep', None)) for x in c}))
uids = gen.insert([[1,2,3,4,5,6,7,8,9,10], [5,6,7]], max_tokens=[4,4])
for _ in range(3):
    gen.next()
    gb = gen._generation_batch
    if len(gb):
        print("generation batch caches:", sorted({(type(x).__name__, getattr(x, 'max_size', None)) for x in gb.prompt_cache}))
        rc = [x for x in gb.prompt_cache if type(x).__name__ == 'BatchRotatingKVCache'][0]
        print("  rotating: offset", rc.offset.tolist(), "left_padding", rc.left_padding.tolist())
gen.close()
