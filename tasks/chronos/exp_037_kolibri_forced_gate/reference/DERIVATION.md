# Where each part of the reference comes from

Each block of `kolibri_ref.py` is mapped to the vendor plugin line it implements and to the vLLM line that fixes its semantics.

- **Vendor:** `aleph_alpha_inference/kolibri1.py` at commit `049a6a7bd2405b27d6d280d256bd3d585191c7ae` (329 lines), plus `aleph_alpha_inference/config.py` where noted.
- **vLLM:** tag v0.29.0, the version the plugin pins (`vllm>=0.29.0,<0.30.0`).
- **Spec:** "Spec" is the item number of the exp_036 forward-pass specification, which `kolibri_ref.py` cites in its comments.

Line numbers were read from copies of these files saved on 2026-10-03 (the vLLM ones under flattened names). Every cited line was checked against the copy when this table was written. One limit applies: two semantics are kernel conventions with no single Python line, and they are marked as such.

## Path key for vLLM v0.29.0

| Short name used below | Repository path | Saved copy |
|---|---|---|
| `ir/layernorm` | `vllm/ir/ops/layernorm.py` | `ir_layernorm.py` |
| `layers/layernorm` | `vllm/model_executor/layers/layernorm.py` | `layernorm.py` |
| `rope/__init__`, `rope/base`, `rope/common` | `vllm/model_executor/layers/rotary_embedding/{__init__,base,common}.py` | `rope_init.py`, `rope_base.py`, `rope_common.py` |
| `flash_attn`, `triton_attn` | `vllm/v1/attention/backends/{flash_attn,triton_attn}.py` | `flash_attn.py`, `triton_attn.py` |
| `attention` | the `Attention` layer the plugin imports from `vllm.model_executor.layers.attention` | `attention.py` |
| `arg_utils` | `vllm/engine/arg_utils.py` | `arg_utils.py` |
| `gate_linear`, `custom_routing_router` | `vllm/model_executor/layers/fused_moe/router/{gate_linear,custom_routing_router}.py` | `fm_router_gate_linear.py`, `fm_router_custom_routing_router.py` |
| `fused_moe/layer` | the fused-MoE module that defines `FusedMoEFactory` (exported by `vllm.model_executor.layers.fused_moe`) | `fm_layer.py` |
| `moe_runner` | `vllm/model_executor/layers/fused_moe/runner/moe_runner.py` | `fm_runner_moe_runner.py` |
| `qwen3_moe` | `vllm/model_executor/models/qwen3_moe.py` | `qwen3_moe.py` |
| `vpe` | `vllm/model_executor/layers/vocab_parallel_embedding.py` | `vpe.py` |
| `logits_processor` | `vllm/model_executor/layers/logits_processor.py` | `logits_processor.py` |
| `config/model` | `vllm/config/model.py` | `model.py` |

## Map

| Reference (`kolibri_ref.py`) | What it does | Vendor `kolibri1.py` | vLLM v0.29.0 | Spec |
|---|---|---|---|---|
| `KolibriConfig.from_dict`: window check | Sliding layers need `use_sliding_window: true` (a missing key counts as false) and a positive `sliding_window`. | 84–90 (refusal). `config.py`:14 `Kolibri1Config(Qwen3MoeConfig)`. | Not vLLM: transformers 5.18 `configuration_qwen3_moe.py`:99 (`use_sliding_window: bool = False`) and :115 (`sliding_window = ... if use_sliding_window else None`). | 2 |
| `KolibriConfig.from_dict`: RoPE parameters | theta from `rope_theta` or `rope_parameters`; only rope type `default`; partial rotary factor 1. | 91–95 (`get_rope(..., rope_parameters=config.rope_parameters)`) | `rope/__init__`:64 (`base = rope_parameters.get("rope_theta", 10000)`), :65 (`rope_type` default), :69–72 (`partial_rotary_factor`) | 2, 8 |
| `expected_shapes` | Tensor names and shapes; `.moe.router.expert_bias` is the router bias. | 64–79 (q/k/v/o without bias), 107–108, 171–188, 229–234, 258–265 (weight-name mapper) | — | 3, 21 |
| `KolibriReference.embed` | Plain row lookup, no scaling. | Inherited from `Qwen3MoeModel` (257) | `qwen3_moe`:459–464, 475–476, 489; `vpe`:78–79 (`F.embedding(input_, layer.weight)`) | 4 |
| `layer_forward` | `h += post_attn_norm(attn(input_layernorm(h)))`; `h += post_ffn_norm(moe(post_attention_layernorm(h)))`. | 229–234 (the four norms), 242–253 (forward) | `qwen3_moe`:499–503 (residual threaded through the layers), `layers/layernorm`:74–94 (`RMSNorm.forward_native`: plain norm, or fused add + norm when a residual is passed), `ir/layernorm`:44–62 (`fused_add_rms_norm`) | 5 |
| `rms_norm` | `w * x / sqrt(mean(x²) + eps)`, with plain `w` (not `1 + w`). | 107–108, 229–234 (`RMSNorm`, not `GemmaRMSNorm`) | `ir/layernorm`:10–21 (`rms_norm`) | 6 |
| `attention_block`: projections | q, k, v, o without bias. | 64–79, 113–114, 122 | — | 7 |
| `attention_block`: qk-norm | Per-head RMSNorm over `head_dim` on q and k, before RoPE, in every layer. | 107–108 (built for every layer), 115–118 (applied), 119–120 (RoPE after) | — | 7 |
| `attention_block`: scale | `head_dim ** -0.5` | 100 | — | 7 |
| `sdpa_ref`: GQA | Query head `h` reads kv head `h // (n_heads / n_kv)`. | 97–101 (`Attention(num_heads, head_dim, scale, num_kv_heads=...)`) | Kernel convention (FlashAttention / Triton grouping); no single Python line. | 7 |
| `rope_inv_freq` | `1 / theta^(2j / D)` in fp32. | 91–95 | `rope/base`:80–92 (`_compute_inv_freq`) | 8 |
| `rope_neox` | Half-rotation `[x1·cos − x2·sin, x2·cos + x1·sin]`, fp32 angles. | 119–120 | `rope/__init__`:36 (`is_neox_style: bool = True`), `rope/base`:94–103 (fp32 cos/sin cache), `rope/common`:169–179 (neox split and concatenation), :19–22 (`rotate_neox`) | 8 |
| `attention_block`: RoPE on sliding layers only | Full layers: no positional encoding. | 81–83 (`rotary_emb = None` on full layers), 84–95 | — | 8, 9 |
| `attention_mask`: sliding | `i − (W − 1) ≤ j ≤ i`: W keys including the query. | 85, 104 (`per_layer_sliding_window=sliding_window`) | `attention`:256–258 (per-layer window wins), `flash_attn`:857–862 (`(sliding_window - 1, 0)` for decoders), `triton_attn`:471–476 (same) | 9, 10 |
| `attention_mask`: full | Plain causal, no window. | 81–82 (`sliding_window = None`) | `arg_utils`:2061–2067 (`CacheConfig.sliding_window` set only when every layer is sliding) | 9 |
| `moe_block`: router logits | `x @ W_gate^T` in fp32. | 171–177 (`GateLinear(..., out_dtype=torch.float32)`), 206 | `gate_linear`:157–199 (`forward`: every tier returns fp32 when `out_dtype` is fp32; tier 4 at :190–192 is `torch.mm(x, W.T, out_dtype=torch.float32)`) | 11 |
| `route_ref` | ids = top-k of `logits + bias`; weights = `sigmoid(logits[ids])`; renormalise only if `norm_topk_prob`. | 126–142 (`sigmoid_logit_add_routing`), 178–180 (fp32 bias), 196 (`renormalize=config.norm_topk_prob`), 202–205 | `custom_routing_router`:46–64 (calls the custom function; weights cast to fp32 at :62) | 11, 12 |
| `expert_mlp` | `down(silu(gate(x)) * up(x))` | 181–188 (shared expert is `Qwen3MoeMLP` with no `expert_gate`) | `qwen3_moe`:85–127 (`Qwen3MoeMLP`: `gate_up_proj`, `SiluAndMul`, `down_proj`; the sigmoid gate at :124–125 applies only when `expert_gate` is given) | 13 |
| `moe_block`: combine | `Σ_k w_k · y_{e_k} + shared(x)`, unscaled. | 181–207 (no `routed_scaling_factor` passed) | `fused_moe/layer`:107 (`FusedMoEFactory(..., routed_scaling_factor: float = 1.0)`), `moe_runner`:418–423 (scaling only when ≠ 1), :779–780 (`shared_output + fused_output`). The weighted sum over the k experts is inside the fused MoE kernel; no single Python line. | 13 |
| `final_norm` | `model.norm` on the last residual. | Inherited (257) | `qwen3_moe`:470, 512 | 5, 14 |
| `lm_head` | fp32 logits `h.f32 @ W.f32^T`; untied head. | 302–310 | `qwen3_moe`:641–645 (`compute_logits`), `logits_processor`:96 (`head_dtype` from the model config), :136–178 (`_apply_head`: fp32 projection), `config/model`:1957–1985 and 2360–2381 (`head_dtype: "float32"` from config.json) | 14 |

## Precision modes and gate support (exp036-ref-2)

| Reference | What it does | Vendor `kolibri1.py` | vLLM v0.29.0 | Spec |
|---|---|---|---|---|
| `rope_neox` (cos/sin precision) | cos/sin computed and kept in fp32, in both modes. vLLM computes them in fp32 and stores the cache in the model dtype (bf16) unless flashinfer is used; keeping fp32 is a precision deviation in the reference's favour, not a semantic one. | 91–95 | `rope/base`:59–61 (`cache = self._compute_cos_sin_cache()`; `if not self.use_flashinfer: cache = cache.to(dtype)`), :94–103 (fp32 computation) | 8 |
| `KolibriReference.norm` (emulation) | Normalise in fp32, round to bf16, multiply by the bf16 weight, round. | 107–108, 229–234 | `ir/layernorm`:14–21 (`x.to(float32)`, `x * rsqrt(var + eps)`, `x.to(weight.dtype) * weight`, `.to(orig_dtype)`) | 6 |
| `layer_branches` (emulation residual) | Sum in fp32; the stored residual is the sum rounded to bf16; the next norm normalises the unrounded sum. | 242–253 | `ir/layernorm`:44–62 (`fused_add_rms_norm`: :53–54 add in fp32, :55 `x_residual = x.to(orig_dtype)`, :57–59 normalise the unrounded `x`), `layers/layernorm`:87–94 (the residual path) | 5 |
| `_proj`, `attention_block`, `expert` (emulation) | GEMM outputs (q/k/v, o_proj, gate/up, down) and the attention output rounded to bf16; logits of the router and the head not. | 64–79, 113–122, 171–188 | GEMM output dtype = activation dtype (bf16) except `GateLinear(out_dtype=fp32)` (`gate_linear`:157–199) and the fp32 head (`logits_processor`:136–178). `silu(gate)` and its product with `up` are rounded as the bf16 `SiluAndMul` kernel writes them: a kernel convention, no single Python line in the saved files. | 11, 13, 14 |
| `moe_block` (emulation combine) | Routed sum rounded to bf16, then `routed + shared` rounded. | 181–207 | `moe_runner`:779–780 (`shared_output + fused_output`, both bf16 tensors) | 13 |
| `moe_block(force_ids=...)`, `forced_route_ref` | A fixed selection weighted by `sigmoid(logits[ids])` of the layer's own fp32 logits. | 126–142 (the weights of `sigmoid_logit_add_routing`) | — (a gate device: the "forced" comparisons of HYPOTHESIS G2) | 12 |
| `mlx_affine_np.dequantize` | `w = scale · q + bias` per group of 64, codes packed LSB-first in uint32 words; fp32 for linear weights, rounded to bf16 for the embedding. | — | — (MLX's storage format, mlx 0.31.2 `mx.quantize(mode="affine")`; checked to 0 ulp against `mx.dequantize`) | 18 |

## The G3 mutants (`mutants.py`)

Each is the misreading a port could make, written as a one-method subclass. The right-hand column is the line that rules it out.

| Mutant | Overrides | The misreading | Ruled out by |
|---|---|---|---|
| `sigmoid_bias_select` | `route` | select on `sigmoid(logits) + bias` (afmoe, DeepSeek-V3, llama.cpp) | vendor 126–142 (`topk(logits + bias)`) |
| `rope_on_full` | `uses_rope` | RoPE in the full-attention layers | vendor 81–83 (`rotary_emb = None` on full layers) |
| `one_plus_w_norm` | `norm` | Gemma's `(1 + w)` RMSNorm | vendor 107–108, 229–234 (`RMSNorm`); `ir/layernorm`:20 |
| `renorm_topk` | `route` | top-k weights renormalised | vendor 196 (`renormalize=config.norm_topk_prob`, false) |
| `swap_sandwich_norms` | `layer_norm_names` | `post_attn_norm` and `post_attention_layernorm` exchanged | vendor 229–234, 242–253 |
| `rope_traditional` | `rope` | interleaved (GPT-J) pairs | `rope/__init__`:36 (`is_neox_style=True`), `rope/common`:19–22 |
| `qknorm_after_rope` | `qk_norm_rope` | q/k norm after RoPE | vendor 115–120 (norm, then rotary) |

## Checked against this table

- `tests/test_reference_invariants.py` tests the window boundary, full-layer NoPE with no window, causality, the `w · x` norm, the plain embedding and the zero sandwich norms on tiny checkpoints.
- `tests/test_routing.py` tests the routing rule against the vendor's own routing test setup.
- `tests/test_config_refusals.py` tests the `use_sliding_window` refusal against the installed transformers.
- `tests/test_port_ref_reference_modes.py` tests the emulation's rounding points, its agreement with INTEGRATION_LOG entry 1, the unpacker against `mx.dequantize`, `force_ids` and `forward_streams`.
- `tests/test_port_ref_reference_mutants.py` tests that each mutant overrides one hook, does what its name says and moves the logits.
