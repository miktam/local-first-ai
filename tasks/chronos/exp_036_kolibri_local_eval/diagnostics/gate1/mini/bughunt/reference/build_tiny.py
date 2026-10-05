"""Tiny Kolibri checkpoint with the real attention geometry: 48 q / 4 kv heads,
head_dim 128, sliding_window 513, full attention at i % 5 == 4 (NoPE),
max_position_embeddings 262144; small hidden/experts/vocab."""
import sys
from pathlib import Path
K = "$KIT"
sys.path.insert(0, K)
sys.path.insert(0, K + "/tests")
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
from tiny_checkpoint import write_tiny_checkpoint, _layer_types

HERE = Path(__file__).resolve().parent
L = int(sys.argv[1]) if len(sys.argv) > 1 else 10
out = HERE / f"ckpt_k16_L{L}"
write_tiny_checkpoint(out, seed=36, preset="pattern5", overrides={
    "num_hidden_layers": L,
    "layer_types": _layer_types(L, 5),
    "num_attention_heads": 48,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "sliding_window": 513,
    "max_position_embeddings": 262144,
    "num_experts": 16,
    "num_experts_per_tok": 6,
    "moe_intermediate_size": 128,
    "shared_expert_intermediate_size": 128,
})
print(out)
