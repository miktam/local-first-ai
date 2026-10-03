# Kolibri-1 numpy reference (exp_036)

`reference/` holds an independent float32 numpy implementation of the Kolibri-1 forward pass. It is the numerical oracle that gates the experiment's MLX port: the port passes only if its logits, hidden states and routing agree with this reference within the gate's tolerances.

Version string: `REF_VERSION = "exp036-ref-2"`. Every output file records it. exp036-ref-2 added the BUILD_SPEC §5.1b deltas: the bf16-emulation and dequantised modes, `force_ids`, the branch dumps, `forward_streams` and the seven G3 mutants. In fp32 mode its logits, hidden states and routing are bit-identical to exp036-ref-1 on both tiny presets (checked 2026-10-03).

## Modes

| Mode | Constructor | What it is for |
|---|---|---|
| fp32 | `KolibriReference(model_dir)` | the reference proper: every activation and matmul fp32, bf16 weights upcast exactly |
| bf16 emulation | `KolibriReference(model_dir, emulate_bf16=True)` | the gate's "emu": what a correct bf16 implementation achieves on these weights |
| dequantised | `KolibriReference(model_dir, dequant_dir=K8)` | G2q: the weights of a converted MLX directory, unpacked by `mlx_affine_np.py`; combines with either precision mode |

**bf16 emulation** rounds to bf16 (nearest even, `round_bf16`) wherever vLLM stores bf16, and nowhere else:

- every norm output, rounded twice as vLLM's `rms_norm` does (normalise in fp32, round, multiply by the bf16 weight, round);
- q, k, v (GEMM outputs), q and k after RoPE, the attention output, `o_proj`;
- in every expert (routed and shared): the gate and up outputs, `silu(gate)`, the product, the down output; then the routed sum and `routed + shared`;
- the residual: the add is done in fp32, the stored residual is the sum rounded to bf16, and the next norm normalises the **unrounded** sum, as vLLM's `fused_add_rms_norm` does (spec item 5). The stream passed between layers (and dumped as `h_in`, `h_mid`, `h_out`) is therefore the unrounded fp32 sum; vLLM's stored residual is `round_bf16` of it.

Router logits, routing weights and the head stay fp32. RoPE's cos/sin stay fp32 (vLLM stores its cos/sin cache in bf16, `rope_base.py:59–61`; spec item 8 keeps fp32 in the reference's favour). On the tiny checkpoints the mode reproduces INTEGRATION_LOG entry 1's independent emulation within its sampling noise (pooled top-1 0.927 against 0.924; `tests/test_port_ref_reference_modes.py`).

**Dequantised** reads a directory written by `port/convert.py` through `mlx_affine_np.ConvertedCheckpoint`, which presents it under the BF16 checkpoint's tensor names (`switch_mlp` rows as `experts.E`, `mlp.gate.expert_bias` as `moe.router.expert_bias`). Each quantised tensor is dequantised the way the MLX op that consumes it does: linear weights (`quantized_matmul`, `gather_qmm`) as `scale · q + bias` in fp32, the embedding (`QuantizedEmbedding`) rounded once more to bf16. Both equal `mx.dequantize` to 0 ulp on the GPU. The unpacker is written from the storage format and imports neither mlx nor the port.

| File | What it is | Licence |
|---|---|---|
| `kolibri_ref.py` | config parsing, pure numerical functions, layer-streaming model (three modes), CLI | Apache-2.0, derived from the vendor plugin (header in the file) |
| `mutants.py` | the seven G3 reference mutants, one overridden method each | Apache-2.0, derived from the vendor plugin (header in the file) |
| `mlx_affine_np.py` | `round_bf16`, the MLX affine unpacker and dequantiser, `ConvertedCheckpoint` | MIT (repo licence) |
| `safetensors_np.py` | minimal safetensors reader, numpy only, written from the format description | MIT (repo licence) |
| `__init__.py` | makes `reference` importable as a package | MIT |
| `DERIVATION.md` | each reference block mapped to the vendor `kolibri1.py` line and the vLLM v0.29.0 file:line it implements | MIT |

## How it was derived

The reference was written from these sources only:

1. **The vendor vLLM plugin**: `aleph_alpha_inference/kolibri1.py` at commit `049a6a7` (Aleph Alpha, Apache-2.0), and its `tests/test_kolibri1.py`, whose only numerical test fixes the routing semantics.
2. **vLLM v0.29.0**, the version the plugin pins (`vllm>=0.29.0,<0.30.0`). Each behaviour below comes from the source code, not from documentation:
   - `ir/ops/layernorm.py`: RMSNorm is `w * x / sqrt(mean(x²) + eps)`.
   - `rotary_embedding` (`get_rope`, `base.py`, `common.py`): `is_neox_style` defaults to True; `inv_freq` and the cos/sin cache are fp32.
   - `flash_attn.py` / `triton_attn.py`: `sliding_window=W` becomes `window_size=(W-1, 0)`.
   - `arg_utils.py`: full-attention layers get no window.
   - `fused_moe/router/gate_linear.py`: router logits are fp32.
   - `custom_routing_router.py`: `renormalize = norm_topk_prob`, and the weights are cast to fp32.
   - `moe_runner.py`: shared and routed outputs are added unscaled.
   - `logits_processor.py` and `config/model.py`: `head_dtype: float32` gives fp32 logits.
3. **The exp_036 forward-pass specification**, items 1–25. It was resolved against (1) and (2), and the code cites it as "spec item N".

Nothing was taken from the MLX port, which was written separately. Its directory was not read while this reference was written.

### What the reference deliberately does not share with the port

| Concern | Port (MLX) | This reference |
|---|---|---|
| Weight loading | mlx / safetensors loaders, sanitize, experts stacked into `switch_mlp` | own `safetensors_np.py`; reads the HF tensor names directly, one tensor at a time |
| Precision | bf16 activations, quantised weights | fp32 everywhere; bf16 weights upcast exactly |
| RoPE | `nn.RoPE(traditional=False)` / `mx.fast.rope` | own `rope_neox` from vLLM's formula |
| Attention | fused SDPA, `RotatingKVCache` / `KVCache`, MLX mask helpers | own masked softmax over the whole sequence, explicit boolean masks, no cache |
| MoE | `SwitchGLU` / `gather_mm` over stacked experts | Python loop over the selected experts, one matmul per expert over its token group |
| Routing | MLX top-k / argpartition | `np.argsort` on `logits + bias` |
| Norms | `mx.fast.rms_norm` | own `rms_norm` |

What they do share: the checkpoint, `config.json` and the specification.

## Semantics implemented

All tensors are float32. In the table, `x` is a layer's input and `h` the residual stream.

| Step | Implementation | Spec |
|---|---|---|
| Embedding | `h = embed_tokens[ids]`, no scaling | 4 |
| Layer | `h += post_attn_norm(attn(input_layernorm(h)))`; `h += post_ffn_norm(moe(post_attention_layernorm(h)))`. `post_attention_layernorm` is the **pre**-MoE norm (Qwen naming). | 5 |
| RMSNorm | `w * x / sqrt(mean(x²) + 1e-6)`; plain `w`, not `1 + w` | 6 |
| Attention | q, k, v projections without bias. Per-head RMSNorm on q and k (`q_norm`, `k_norm`) **before** RoPE, in every layer. Scale is `head_dim**-0.5`. GQA: query head `h` reads kv head `h // (n_heads / n_kv)`. No softcap, sinks or output gate. | 7 |
| RoPE | NeoX half-rotation: `[x1·cos − x2·sin, x2·cos + x1·sin]`. `theta` comes from the config (1e4). Positions are absolute, 0-based, and restart at 0 for each sequence. Angles, cos and sin are fp32, as in vLLM's cache. **Sliding layers only.** | 8 |
| Sliding layers | Query `i` attends keys `i-(W-1) … i`, so W keys including itself. W is `sliding_window` taken verbatim (513 gives 512 previous + current). | 9 |
| Full layers | Plain causal mask, no positional encoding (NoPE) | 9 |
| Router | `logits = x @ W_gate^T` in fp32. Experts are chosen as the top-k of `logits + expert_bias`, where the bias is BF16 in the checkpoint and upcast exactly. Weights are `sigmoid(logits[ids])`: unbiased, **not** renormalised (`norm_topk_prob: false`), scale 1.0. | 11–12 |
| Experts | `down(silu(gate(x)) * up(x))`. MoE output is `Σ_k w_k·y_k + shared(x)`; the shared expert is an ungated SwiGLU of the same form. | 13 |
| Head | `model.norm`, then `logits = h @ lm_head^T` in fp32. `lm_head` is untied (`tie_word_embeddings: true` is also supported). | 14 |

`KolibriConfig.from_dict` rejects configurations it does not implement, rather than computing something else. It raises on rope scaling, a partial rotary factor other than 1, an activation other than silu, attention bias, missing `layer_types`, or sliding layers without a positive window. It also raises on sliding layers without `use_sliding_window: true`. A missing key counts as false, as in transformers' `Qwen3MoeConfig`, which the vendor config subclasses and which then nulls the window.

## Numerics and determinism

- **fp32 throughout.** All matmuls use fp32 inputs, so the results are fp32. Checkpoint BF16 values are decoded exactly: the float32 bit pattern is the bf16 pattern shifted left by 16 (`uint16 << 16`). Nothing is rounded back to bf16. The reference is therefore more precise than both the vendor's serving path (bf16 residual stream, FP8 weights) and the port. Disagreements at the bf16 level are expected and belong to the port, not to the reference.
- **RoPE angles are fp32**, mirroring vLLM rather than being "more exact". At position p the angle carries an error of about one fp32 ulp of p radians. The self-check measured 6e-4 near position 5000, against a float64 RoPE.
- **Exactly causal and batch-invariant on one machine.** Changing tokens after position t leaves logits `0..t` bit-identical. A sequence run inside a batch gives bit-identical results to the same sequence run alone. Two things make this hold:
  - Masked attention scores are `-inf`, so they contribute exact zeros.
  - Every weight product goes through `linear()`, which never sends a single row to BLAS. A one-row product takes a gemv path with a different summation order (observed with Apple Accelerate). Products of two or more rows were observed to give each row bit-identical results whatever the row count.
- **Not bit-identical across sequence lengths, machines or BLAS builds.** A run on `ids[:t+1]` agrees with the first `t+1` rows of a longer run only to about 4e-6 on the tiny test model, because the attention shapes differ. Gates should use tolerances, not bit equality.

## Python API

```python
from reference.kolibri_ref import KolibriReference, REF_VERSION

ref = KolibriReference(model_dir)            # header-only shape check of all 58,353 tensors
logits = ref.forward(ids)                    # [T, V] float32
logits, layers, final = ref.forward(ids, return_hidden=True)
#   layers: list of num_hidden_layers arrays [T, H], residual stream after each layer
#   final:  model.norm output [T, H]
results = ref.forward_batch([ids_a, ids_b])  # one pass over the weights, list like forward()
logits, hidden, final, lengths = ref.forward_packed([ids_a, ids_b], return_hidden=True)
#   packed: logits [N, V], hidden [L, N, H], final [N, H], sequences concatenated in order
ref.forward(ids, num_layers=4)               # partial: first 4 layers, then norm + head
ref.last_routing                             # per layer (weights [N, k] f32, ids [N, k] i64)
ref.last_stats                               # per layer seconds and experts read
ref.layer_forward(i, h, force_ids=None)      # one decoder layer on a residual stream [T, H]
out = ref.layer_branches(i, h, force_ids=None)
#   out: r_attn, h_mid, r_moe, h_out [N, H] (spec item 29) and info (moe_block's)
y, info = ref.moe_block(i, x, force_ids=None)  # x = post_attention_layernorm output [N, H]
#   info: logits [N, E] f32 (raw router logits), biased [N, E] f32 (logits + expert_bias),
#         top6 [N, k] i64 (by descending biased score, or force_ids; also as 'ids'),
#         weights [N, k] f32, forced (bool), experts_loaded
ref.last_layer_info                          # info of the latest layer, plus r_attn, h_mid, r_moe
ref.forward_packed(seqs, on_layer=f)         # f(i, h_in, h_out, info) after each layer
out = ref.forward_streams(ids, {"name": MutantClass, ...}, include_base=True, on_logits=None)
#   one layer-streamed pass carrying one hidden stream per variant; out[name] = logits [N, V]
#   (or on_logits(name, logits)); out["ref"] is this instance's own stream; each stream is
#   bit-identical to that variant's own forward, and each layer tensor is read once
ref.mode                                     # "fp32" or "bf16_emulation"
```

`force_ids` [N, k] fixes the MoE selection; the weights are still `sigmoid(logits[force_ids])` from the layer's own fp32 logits (`forced_route_ref`). Forcing a layer's own natural selection changes nothing, bit for bit.

Hooks. Each semantic choice the G3 mutants vary is one method: `norm`, `rope`, `uses_rope`, `qk_norm_rope`, `route`, `layer_norm_names`. A mutant in `mutants.py` is a subclass that overrides exactly one of them; `MUTANTS` maps the seven names of HYPOTHESIS G3 to the classes (`sigmoid_bias_select`, `rope_on_full`, `one_plus_w_norm`, `renorm_topk`, `swap_sandwich_norms`, `rope_traditional`, `qknorm_after_rope`). On the tiny checkpoints each moves the logits by 1.1–7.6 (max-abs).

Pure functions, for probing the semantics in tests:

```python
rms_norm(x, w, eps)                         # y = w * x / sqrt(mean(x^2) + eps)
rope_neox(x, positions, theta)              # x [T, n_heads, D], positions [T]
attention_mask(q_pos, k_pos, window)        # bool [Tq, Tk]; window=None means causal
sdpa_ref(q, k, v, scale, window, q_chunk=None)  # q [T, nh, D], k/v [T, nkv, D]
route_ref(logits_f32, bias_f32, k, renormalize=False)  # -> (weights [T,k] f32, ids [T,k] i64)
expert_mlp(x, w_gate, w_up, w_down)         # SwiGLU, weights in checkpoint layout [out, in]
linear(x, w)                                # x @ w.T, never as a one-row BLAS call
forced_route_ref(logits_f32, ids, renormalize=False)  # weights of a fixed selection
round_bf16(x)                               # fp32 -> nearest-even bf16 -> fp32 (IEEE; no flush of subnormals)
next_token_nll(logits, ids, lengths)        # [N] f32, NaN at each sequence's last position
reference_tree_sha256()                     # tools/hash_tree.py's tree rule over reference/
load_config(model_dir), KolibriConfig.from_dict(d), expected_shapes(cfg), param_count(cfg)
```

`safetensors_np.open_checkpoint(model_dir)` opens `model.safetensors.index.json` or a single `model.safetensors`. Its methods:

- `.names()`, `.dtype_of(name)` and `.shape_of(name)`
- `.get(name)`: a float32 copy; I64 tensors come back as int64
- `.get_rows(name, rows)`: only the touched rows are read
- `.get_raw(name)`: the stored bits, uint16 for BF16

It reads BF16, F16, F32, I64 and U32 (the packed codes of converted checkpoints, returned as uint32). Any other dtype raises `TypeError`, including the FP8 repo's `F8_E4M3`; use the BF16 repo. Each tensor is memory-mapped on its own and unmapped once it has been decoded.

## CLI

Run it from the experiment directory:

```bash
python -m reference.kolibri_ref --model-dir "$EXP036_MODELS/Kolibri-1-BF16" \
    --ids-file ids.json [--out ref_logits.npz] [--dump DIR] [--hidden] [--num-layers N] [--attn-chunk N] \
    [--emulate-bf16] [--dequant-dir DIR] [--mutants all|NAME,...] [--gate-text-sha HEX] [--quiet]
```

- `--emulate-bf16` and `--dequant-dir DIR` select the modes above.
- `--mutants` (with `--dump`) carries the named G3 mutants through the same pass as the reference's own stream and writes `stream_<name>.nll.npy` for `ref` and each mutant: the next-token NLL per position, NaN at each sequence's last position. It cannot be combined with `--hidden`.
- `--gate-text-sha` is recorded as `gate_text_sha256` (default: the sha256 of the ids file).

At least one of `--out` and `--dump` is required. `--out` gets `.npz` appended when it lacks the suffix, as `np.savez` would, and the path printed at the end is the file actually written.

- `--model-dir` defaults to `$EXP036_MODELS/Kolibri-1-BF16`.
- `ids.json` holds one sequence `[1, 2, 3]` or several `[[...], [...]]`, optionally wrapped as `{"ids": ...}`. Several sequences run in **one pass over the weights**, which is the efficient way to build a gate set.
- Progress goes to stderr, one line per layer: time, experts read, peak RSS. The final line on stdout gives the wall time and peak RSS.

The `.npz` output contains:

| Key | Shape / type | Notes |
|---|---|---|
| `logits` | `[N, V]` float32 | all sequences concatenated along tokens |
| `ids` | `[N]` int64 | |
| `seq_lengths` | `[S]` int64 | split with `np.split(x, np.cumsum(seq_lengths)[:-1])` |
| `hidden` | `[L, N, H]` float32 | `--hidden` only; residual stream after each layer |
| `final_norm` | `[N, H]` float32 | `--hidden` only |
| `router_ids`, `router_weights` | `[L, N, k]` | `--hidden` only |
| `ref_version`, `config_sha256`, `checkpoint_fingerprint`, `model_dir_name` | str | the fingerprint is the sha256 of the shard index (or of the single file's header) |
| `num_layers`, `layer_seconds`, `experts_loaded`, `wall_seconds`, `peak_rss_bytes` | | run record; `peak_rss_bytes` is taken before saving |

No absolute paths are written into the output.

`--dump DIR` writes the gate's per-layer files (BUILD_SPEC §5.2) into `DIR`, which must not exist or must be empty. The gate passes `$EXP036_WORK/ref/<gate_text_sha>`. The files are written layer by layer as the forward runs:

| File | Shape / type | Notes |
|---|---|---|
| `layerNN.h_in.npy` | `[N, H]` float32 | residual stream entering layer NN (`layer00.h_in` is the embedding) |
| `layerNN.r_attn.npy` | `[N, H]` float32 | attention branch after `post_attn_norm`, before the residual add (spec item 29) |
| `layerNN.h_mid.npy` | `[N, H]` float32 | `h_in + r_attn` (emulation: `round_bf16(h_in) + r_attn`) |
| `layerNN.r_moe.npy` | `[N, H]` float32 | MoE branch after `post_ffn_norm`, before the residual add |
| `layerNN.h_out.npy` | `[N, H]` float32 | residual stream leaving layer NN; equals `layer(NN+1).h_in` |
| `layerNN.router_logits.npy` | `[N, E]` float32 | raw router logits |
| `layerNN.biased.npy` | `[N, E]` float32 | router logits + `expert_bias`: the selection scores |
| `layerNN.top6.npy` | `[N, k]` int64 | selected expert ids, by descending biased score |
| `layerNN.top6_gap.npy` | `[N]` float32 | k-th minus (k+1)-th biased score: the G2 near-tie margin (exemption below 1e-4) |
| `logits.npy`, `logprobs.npy` | `[N, V]` float32 | fp32 logits; log-softmax computed in float64 and stored as float32 |
| `final_norm.npy`, `ids.npy`, `seq_lengths.npy` | | as in the `.npz` |
| `stream_<name>.nll.npy` | `[N]` float32 | with `--mutants` only |
| `ref_record.json` | JSON, sorted keys | sha256, bytes, shape and dtype of every file; `ref_version`, `mode`, `dequantised_from`, `dequantised_manifest_sha256`, `config_sha256`, `checkpoint_fingerprint`, `reference_tree_sha256`, `gate_text_sha256`, `ids_file_sha256`, `ids_sha256`, `streams`, `num_layers`, `num_tokens`, `seq_lengths`, per-layer seconds and experts read, wall time, peak RSS, `t_start`, `t_end` |

`ref_record.json` is written last, so a directory without it is an incomplete dump. `logprobs.npy` is written in row blocks, so it never exists whole in memory next to the logits. On the real model a 1,536-token text dumps about 5.8 GB: 3.9 GB of per-layer `[N, H]` states (five per layer), 0.2 GB of router logits and selection scores, and 1.6 GB of logits and log-probs. T1–T8 (12,288 tokens) come to about 46 GB and T9 (16,384 tokens) to about 62 GB, so the gate plans for ≈ 100 GB under `$EXP036_WORK`.

## Memory and time on the real model

The real checkpoint has 50 layers. Each layer is 1,548,954,240 parameters, 3.10 GB in BF16, of which 3.02 GB are the 384 routed experts. `embed_tokens` and `lm_head` are 0.66 GB each.

The reference reads one tensor at a time and converts it to fp32. It runs the layer and then frees the weights. For the MoE it reads only the experts that at least one token selects. The peak RAM therefore does **not** scale with the checkpoint:

- Weights in RAM at any moment: at most about 130 MB (the fp32 q/o projections).
- Activations: about 150 KB per token inside a layer (q, its head-major copy, the attention output and the MoE buffers).
- Attention score buffer: up to 128 MB for full layers and about 100 MB for sliding layers. It is reused for every query chunk.
- Logits: 0.5 MB per token (`N × 128000 × 4 B`). With `--dump`, the log-probs are written in blocks and add no second copy, so T9 (16,384 tokens) peaks at about 9 GB of logits plus the per-layer buffers, within the ≈ 20 GB BUILD_SPEC allows.
- `forward_streams` keeps one layer's weights for all streams: up to about 6.2 GB fp32 on the real model when every expert is selected.
- With `--hidden`, add 10 KB per token per layer, which is 0.5 MB per token over 50 layers.

**Measurements.** These come from miktam-mini (M4 Pro, 64 GB, numpy 2.5.3 on Accelerate). The test checkpoint had real Kolibri dimensions and random weights: two layers (one sliding, one full), the full vocabulary, and small router biases, so tokens spread over most experts (the worst case for reads). The page cache was warm.

| Tokens per pass | Experts read, sliding / full layer | Seconds per layer, sliding / full | Peak RSS of the whole run |
|---|---|---|---|
| 64 | 177 / 136 | 0.24 / 0.20 | 0.36 GiB |
| 512 | 344 / 301 | 0.51 / 0.47 | 0.89 GiB |
| 2,048 | 383 / 372 | 0.93 / 1.04 | 2.9 GiB |
| 8 × 512 (one batch, `--hidden`) | 384 / 384 | 1.13 / 1.05 | 3.2 GiB |
| 8,192 | 384 / 382 | 2.26 / 5.47 | 6.0 GiB |

**Expectation for the 50-layer model.** On the run host (M5 Max, 128 GB), a pass of up to a few thousand tokens should cost about 25–55 s of compute: 40 sliding and 10 full layers at the rates above. On top of that comes reading up to 3.1 GB per layer, about 156 GB per pass. The checkpoint is larger than the run host's RAM, so most of every pass is a cold read from SSD. Budget **about 1–3 minutes per pass** at ≤ 4k tokens, and about 5 minutes at 8k. Peak RAM stays in single-digit GiB, dominated by the logits.

Batch the gate prompts into one `ids.json`. Eight prompts cost one pass, not eight. Run long jobs yourself from a terminal; the per-layer progress lines show where the run is.

## Verification done when it was written (2026-10-03)

These checks used throwaway scripts that are not part of the experiment's tests. The tiny checkpoint had the real tensor names, hidden size 256, 6 layers, 8 q / 2 kv heads, head_dim 32, 8 experts with top-2 routing, intermediate size 64, sliding window 9, full layers at 4 and 5, and vocab 512. It was written by our own numpy BF16 writer with round-to-nearest-even, as one sharded copy with an index and one single-file copy.

- **Reader.** BF16 decoding matched `mlx.core.load` bit for bit, including -0 and subnormals. F16, F32, I64 and scalar tensors, row reads and raw bits matched the written data. The sharded and single-file copies gave identical logits.
- **Against a naive float64 model.** A deliberately naive implementation (per-token loops, per-pair RoPE) agreed with the reference to 1.5e-6 relative on the logits, with identical routing and argmax.
- **Mutation tests.** Each plausible porting mistake moved the logits by 0.56–1.6 relative, against a baseline of 1.5e-6:
  - a `(1+w)` RMSNorm
  - GPT-J interleaved RoPE
  - selecting experts on `sigmoid + bias`
  - renormalised routing weights
  - a window of W+1 keys
  - a window of W-1 keys
- **(a) Causality.** Changing tokens 12–23 left logits 0–11 bit-identical. Changing token 10 moved logits[11] by 2.35.
- **(b) Window, on a single sliding layer (W = 9, query i = 30).**
  - Perturbing input row 21 (= i−W) or row 16 changed output row 30 by exactly 0.
  - Perturbing row 22 (= i−(W−1)) changed it by 1.6.
  - A future row changed it by exactly 0.
  - In full layer 4, perturbing row 0 changed it.
  - The same held at token level with `num_layers=1`.
- **(c) Routing.** The vendor test setup was 64 tokens × 384 experts, logits `randn×3`, bias `randn×5`, with numpy seed 0 because torch is not installed.
  - `route_ref`'s ordered ids equalled an independent top-6 of `logits + bias`.
  - Its weights equalled `sigmoid(logits[ids])` to rtol 1e-6.
  - The selected expert sets differed from `sigmoid + bias` selection in 64 of 64 rows.
  - The weights were not renormalised: row sums ran from 4.7 to 6.0.
- **(d) Real config and parameter count.**
  - The shipped `config.json` parses: 50 layers, full attention at `i % 5 == 4`, window 513, theta 1e4, 384 experts top-6.
  - `param_count` gives **78,103,074,560**. This equals the BF16 index's `total_size` (156,206,149,120 B) divided by 2.
  - It also equals 50 × layer 0 + embed + lm_head + norm, using the 1,167 layer-0 tensors in the real shard-1 safetensors header.
  - The shard-1 header parses with `parse_header_bytes`, and all of its 1,375 names, shapes and BF16 dtypes match `expected_shapes`.
- **Pure functions against MLX kernels.**
  - `rope_neox` matches `mx.fast.rope(traditional=False)` to 4.7e-6 and differs from `traditional=True`.
  - `rms_norm` matches `mx.fast.rms_norm` to 4.8e-7.
  - `sdpa_ref` with GQA and a boolean mask matches `mx.fast.scaled_dot_product_attention` to 6.6e-7, both causal and with a window.
- **Batch and chunking.** `forward_batch` over sequences of 24, 17 and 1 tokens was bit-identical to separate forwards. Using `attn_chunk=3` instead of the default changed the logits by at most 4e-6. The CLI output (`--hidden`, multiple sequences) was bit-identical to the API.
