"""Model builder shared by the bughunt scripts (in-memory Kolibri port, random weights)."""
import os, sys
os.environ.setdefault("MLX_ENABLE_TF32", "0")
K = "$KIT"
sys.path[:0] = [K, K + "/tests"]
from tools.precision import ensure_exact_fp32
ensure_exact_fp32()
import argparse
import mlx.core as mx, mlx.nn as nn, numpy as np
from gate import common

def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=2560); ap.add_argument("--heads", type=int, default=48)
    ap.add_argument("--kv", type=int, default=4); ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--E", type=int, default=64); ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--I", type=int, default=512); ap.add_argument("--layers", type=int, default=5)
    ap.add_argument("--V", type=int, default=4096); ap.add_argument("--W", type=int, default=513)
    ap.add_argument("--prec", default="bf16"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emb_std", type=float, default=1.0); ap.add_argument("--proj_std", type=float, default=0.02)
    ap.add_argument("--head_std", type=float, default=0.08); ap.add_argument("--router_std", type=float, default=0.02)
    ap.add_argument("--bias_std", type=float, default=2.0); ap.add_argument("--postnorm_scale", type=float, default=1.0)
    ap.add_argument("--outlier", type=float, default=0.0)
    return ap

def build(a):
    module = common.port_module()
    lt = ["full_attention" if i % 5 == 4 else "sliding_attention" for i in range(a.layers)]
    args = module.ModelArgs(model_type="kolibri1", hidden_size=a.H, num_hidden_layers=a.layers,
                            num_attention_heads=a.heads, num_key_value_heads=a.kv, head_dim=a.D, vocab_size=a.V,
                            num_experts=a.E, num_experts_per_tok=a.k, moe_intermediate_size=a.I,
                            shared_expert_intermediate_size=a.I, sliding_window=a.W, layer_types=lt,
                            rope_theta=10000.0, use_sliding_window=True)
    model = module.Model(args)
    mx.random.seed(a.seed)
    def nrm(shape, std, dt=mx.bfloat16): return (mx.random.normal(shape) * std).astype(dt)
    W = {}
    W["model.embed_tokens.weight"] = nrm((a.V, a.H), a.emb_std)
    if a.outlier:
        e = W["model.embed_tokens.weight"].astype(mx.float32)
        e[:, 7] = a.outlier
        W["model.embed_tokens.weight"] = e.astype(mx.bfloat16)
    W["model.norm.weight"] = (1 + nrm((a.H,), 0.1, mx.float32)).astype(mx.bfloat16)
    W["lm_head.weight"] = nrm((a.V, a.H), a.head_std)
    for i in range(a.layers):
        p = f"model.layers.{i}"
        W[f"{p}.self_attn.q_proj.weight"] = nrm((a.heads * a.D, a.H), a.proj_std)
        W[f"{p}.self_attn.k_proj.weight"] = nrm((a.kv * a.D, a.H), a.proj_std)
        W[f"{p}.self_attn.v_proj.weight"] = nrm((a.kv * a.D, a.H), a.proj_std)
        W[f"{p}.self_attn.o_proj.weight"] = nrm((a.H, a.heads * a.D), a.proj_std)
        W[f"{p}.self_attn.q_norm.weight"] = (1 + nrm((a.D,), 0.1, mx.float32)).astype(mx.bfloat16)
        W[f"{p}.self_attn.k_norm.weight"] = (1 + nrm((a.D,), 0.1, mx.float32)).astype(mx.bfloat16)
        for n in ("input_layernorm", "post_attention_layernorm"):
            W[f"{p}.{n}.weight"] = (1 + nrm((a.H,), 0.1, mx.float32)).astype(mx.bfloat16)
        for n in ("post_attn_norm", "post_ffn_norm"):
            W[f"{p}.{n}.weight"] = ((1 + nrm((a.H,), 0.1, mx.float32)) * a.postnorm_scale).astype(mx.bfloat16)
        W[f"{p}.mlp.gate.weight"] = nrm((a.E, a.H), a.router_std).astype(mx.float32)
        W[f"{p}.mlp.gate.expert_bias"] = nrm((a.E,), a.bias_std).astype(mx.float32)
        W[f"{p}.mlp.switch_mlp.gate_proj.weight"] = nrm((a.E, a.I, a.H), a.proj_std)
        W[f"{p}.mlp.switch_mlp.up_proj.weight"] = nrm((a.E, a.I, a.H), a.proj_std)
        W[f"{p}.mlp.switch_mlp.down_proj.weight"] = nrm((a.E, a.H, a.I), a.proj_std)
        W[f"{p}.mlp.shared_experts.gate_proj.weight"] = nrm((a.I, a.H), a.proj_std)
        W[f"{p}.mlp.shared_experts.up_proj.weight"] = nrm((a.I, a.H), a.proj_std)
        W[f"{p}.mlp.shared_experts.down_proj.weight"] = nrm((a.H, a.I), a.proj_std)
        mx.eval([v for k_, v in W.items() if k_.startswith(p + ".")])
    model.load_weights(list(W.items()))
    del W
    if a.prec in ("q8", "q4"):
        pred = module.make_quant_predicate(quantize_embeddings=False, quantize_lm_head=True)
        nn.quantize(model, group_size=64, bits=8 if a.prec == "q8" else 4, class_predicate=pred)
    elif a.prec == "fp32":
        model.set_dtype(mx.float32)
    mx.eval(model.parameters())

    return model, module
