"""Build the spike's tiny checkpoints and token ids (W13, exp_037 DESIGN.md §10).

Uses exp_036's tests/tiny_checkpoint.py read-only (imported from its path; no
bytecode written). Runs under $HOME/models/exp036-mini/venv312 (needs mlx).

Checkpoints written under vendor_spike/ck/:
  vendor_s0   tiny vendor preset ("vendor"), seed 0  (window 65; 4 sliding + 2 full)
  vendor_s1   tiny vendor preset, seed 1
  w513_s0     "w513" preset, seed 0 (vendor layout, window 513)
  rl_s23      scratch rebuild of the §5.1 real-layout recipe (bughunt build_ck.py
              base overrides + q/k-norm x3, preset w513), seed 23
  rl_s29      the same recipe, seed 29
Token ids: random_ids(64, seed=101) and random_ids(600, seed=102) from the same
module (uniform over [0, 1008)).
"""
import json
import sys

sys.dont_write_bytecode = True
E36 = str(__import__("pathlib").Path(__file__).resolve().parents[4] / "exp_036_kolibri_local_eval")  # exp_037 copy: was the absolute path of exp_036's directory on the mini
sys.path.insert(0, E36 + "/tests")
from tiny_checkpoint import _layer_types, random_ids, write_tiny_checkpoint  # noqa: E402

V = sys.argv[1]

REAL = {"num_hidden_layers": 10, "layer_types": _layer_types(10, 5), "sliding_window": 513,
        "max_position_embeddings": 32768, "num_attention_heads": 48, "num_key_value_heads": 4, "head_dim": 128,
        "hidden_size": 256, "num_experts": 64, "num_experts_per_tok": 6, "moe_intermediate_size": 64,
        "shared_expert_intermediate_size": 64}


def sharpen(f):
    def edit(t):
        for k in list(t):
            if k.endswith("q_norm.weight") or k.endswith("k_norm.weight"):
                t[k] = t[k] * f
    return edit


write_tiny_checkpoint(f"{V}/ck/vendor_s0", seed=0, preset="vendor", copy_port=False)
write_tiny_checkpoint(f"{V}/ck/vendor_s1", seed=1, preset="vendor", copy_port=False)
write_tiny_checkpoint(f"{V}/ck/w513_s0", seed=0, preset="w513", copy_port=False)
write_tiny_checkpoint(f"{V}/ck/rl_s23", seed=23, preset="w513", overrides=REAL, edit_tensors=sharpen(3.0), copy_port=False)
write_tiny_checkpoint(f"{V}/ck/rl_s29", seed=29, preset="w513", overrides=REAL, edit_tensors=sharpen(3.0), copy_port=False)
ids = {"ids64": random_ids(64, seed=101), "ids600": random_ids(600, seed=102)}
with open(f"{V}/ck/ids.json", "w") as f:
    json.dump(ids, f)
print("ok")
