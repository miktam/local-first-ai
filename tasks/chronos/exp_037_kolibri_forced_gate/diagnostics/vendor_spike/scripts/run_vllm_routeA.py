"""Route A of the W13 spike: vLLM 0.29.0 (built from source, macOS CPU) with
the vendor plugin aleph-alpha-inference @049a6a7 (installed unmodified; it
registers Kolibri1ForCausalLM through vLLM's general_plugins entry point).

One LLM engine per checkpoint, in-process (VLLM_ENABLE_V1_MULTIPROCESSING=0),
dtype float32, eager. Each sequence is one prompt-only request (max_tokens=1),
prefilled in a single step (max_num_batched_tokens >= 600, prefix caching off).
Forward hooks on the vendor's decoder layers record the residual stream after
each layer (hidden_states + residual, the value the next layer's fused
add-norm forms) and the final model.norm output; the logits are the vendor
model's own compute_logits (LogitsProcessor + ParallelLMHead) on that output.
"""
import json
import os
import sys

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
os.environ.setdefault("VLLM_CPU_KVCACHE_SPACE", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import torch  # noqa: E402

V = sys.argv[1]
ck = sys.argv[2]
ids = json.load(open(f"{V}/ck/ids.json"))

CAP = {"layers": {}, "final": None, "n_calls": 0, "router": {}, "emb": None}


def install_hooks(model):
    import aleph_alpha_inference.kolibri1 as k1

    assert type(model).__name__ == "Kolibri1ForCausalLM", type(model)
    assert type(model).__module__ == k1.__name__, type(model).__module__
    layers = model.model.layers

    def mk(i):
        def hook(mod, inp, out):
            hs, res = out
            CAP["layers"][i] = (hs.detach().float() + res.detach().float()).clone()
        return hook

    for i, layer in enumerate(layers):
        assert type(layer).__name__ == "Kolibri1DecoderLayer"
        layer.register_forward_hook(mk(i))

    def fhook(mod, inp, out):
        x = out[0] if isinstance(out, tuple) else out
        CAP["final"] = x.detach().clone()
        CAP["n_calls"] += 1

    model.model.norm.register_forward_hook(fhook)

    # Diagnostics (recording only): router logits (GateLinear output, fp32) per
    # layer, and the embedding output (layer 0's input).
    def mkg(i):
        def ghook(mod, inp, out):
            x = out[0] if isinstance(out, tuple) else out
            CAP["router"][i] = x.detach().float().clone()
        return ghook

    for i, layer in enumerate(layers):
        layer.mlp.gate.register_forward_hook(mkg(i))

    def ehook(mod, inp, out):
        CAP["emb"] = out.detach().float().clone()

    model.model.embed_tokens.register_forward_hook(ehook)
    info = {
        "model_cls": f"{type(model).__module__}.{type(model).__name__}",
        "param_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
        "n_layers": len(layers),
        "layer0_attn_window": [getattr(l.self_attn.attn, "sliding_window", None) for l in layers],
        "impl_window": [getattr(l.self_attn.attn.impl, "sliding_window", None) for l in layers],
        "rope": [None if l.self_attn.rotary_emb is None else
                 {"cls": type(l.self_attn.rotary_emb).__name__, "base": float(l.self_attn.rotary_emb.base),
                  "neox": bool(l.self_attn.rotary_emb.is_neox_style),
                  "rotary_dim": int(l.self_attn.rotary_emb.rotary_dim)} for l in layers],
        "bias_l0_first4": model.model.layers[0].mlp.gate.e_score_correction_bias.detach().float()[:4].tolist(),
        "attn_backend": type(layers[0].self_attn.attn.impl).__name__,
        "moe_cls": type(layers[0].mlp.experts).__name__,
    }
    return info


def get_bias(model):
    return [l.mlp.gate.e_score_correction_bias.detach().float().clone().numpy() for l in model.model.layers]


def logits_of(model):
    with torch.no_grad():
        return model.compute_logits(CAP["final"]).detach().float().clone()


def main():
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    llm = LLM(
        model=f"{V}/ck/{ck}",
        skip_tokenizer_init=True,
        dtype="float32",
        enforce_eager=True,
        max_model_len=1024,
        max_num_batched_tokens=2048,
        max_num_seqs=1,
        enable_prefix_caching=False,
        seed=0,
    )
    info = llm.apply_model(install_hooks)[0]
    print("INFO", json.dumps(info))
    sp = SamplingParams(max_tokens=1, temperature=0.0, detokenize=False)
    for name in ("ids64", "ids600"):
        x = ids[name]
        CAP["layers"].clear()
        CAP["final"] = None
        CAP["n_calls"] = 0
        CAP["router"].clear()
        CAP["emb"] = None
        out = llm.generate([TokensPrompt(prompt_token_ids=x)], sp, use_tqdm=False)
        assert CAP["n_calls"] == 1, f"model forwards during {name}: {CAP['n_calls']} (expected one prefill)"
        T = len(x)
        hidden = torch.stack([CAP["layers"][i] for i in range(len(CAP["layers"]))])
        assert hidden.shape[1] >= T, hidden.shape
        logits = llm.apply_model(logits_of)[0]
        np.savez(
            f"{V}/out/{os.environ.get('SPIKE_PREFIX', 'vllmA')}_{ck}_{name}.npz",
            hidden=hidden[:, :T].numpy(),
            final=CAP["final"][:T].float().numpy(),
            logits=logits[:T].numpy(),
            ids=np.array(x),
            captured_rows=np.array(hidden.shape[1]),
            sampled=np.array(out[0].outputs[0].token_ids),
            router_logits=torch.stack([CAP["router"][i] for i in range(len(CAP["router"]))])[:, :T].numpy(),
            emb=CAP["emb"][:T].numpy(),
            bias=np.stack(llm.apply_model(get_bias)[0]),
        )
        print(ck, name, "rows", hidden.shape[1], "logits", tuple(logits.shape), "sampled", out[0].outputs[0].token_ids)


if __name__ == "__main__":
    main()
