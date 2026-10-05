from common_v import *
from tiny_checkpoint import write_tiny_checkpoint, _layer_types
def sharpen(f):
    def edit(t):
        for k in list(t):
            if k.endswith("q_norm.weight") or k.endswith("k_norm.weight"):
                t[k] = t[k] * f
    return edit
base = {"num_hidden_layers": 10, "layer_types": _layer_types(10, 5), "sliding_window": 513,
        "max_position_embeddings": 32768, "num_attention_heads": 48, "num_key_value_heads": 4, "head_dim": 128,
        "hidden_size": 256, "num_experts": 64, "num_experts_per_tok": 6, "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64}
for name, seed, f in (("ckv_s1", 23, 1.0), ("ckv_s3", 23, 3.0)):
    write_tiny_checkpoint(name, seed=seed, preset="w513", overrides=base, edit_tensors=sharpen(f))
    print(name)
