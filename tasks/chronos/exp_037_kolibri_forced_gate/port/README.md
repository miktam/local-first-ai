# Kolibri-1 MLX port (exp_036, exp_037)

No local runtime supports Aleph Alpha's Kolibri-1 (`model_type: kolibri1`: 78.1B total / 3.46B active MoE, 50 layers, 384 routed experts top-6 plus one ungated shared expert, sliding-window RoPE layers mixed with full-attention NoPE layers). This folder is the port that exp_036 ran on and exp_037 runs on, under MLX 0.32.3 / mlx-lm 0.32.0 (exp_037 decision F2).

| File | What it is |
|---|---|
| `kolibri1.py` | `ModelArgs` and `Model` for mlx_lm 0.32.0, plus `VllmRoPE`, `route()`, `forced_route()`, `make_quant_predicate()` and `SPEC_VERSION = "exp037-port-1"` |
| `convert.py` | BF16 checkpoint → affine-quantised MLX checkpoint `Kolibri-1-MLX-{8,4}bit-g64`, plus `exp036_convert_record.json`; memory guard; `--check-port-file` / `--refresh-port-file`; `model_file_trust()` for mlx-lm 0.32.0 loads |
| `convert_streaming.py` | the fallback converter (`--streaming`): the same tensors, written one layer at a time |

## What changed in exp037-port-1 (exp_037 DESIGN §6.1)

1. **vLLM-angle RoPE.** The sliding layers rotate with `VllmRoPE(head_dim, rope_theta)` instead of `nn.RoPE(head_dim, traditional=False, base=rope_theta)`. Both are NeoX (half-rotation) RoPE through `mx.fast.rope`; they differ only in the angle. `VllmRoPE` computes `inv_freq[j] = 1 / theta^(2j / 128)` in fp32, exactly as vLLM's `RotaryEmbedding._compute_inv_freq` and the reference's `rope_inv_freq` do, and passes the periods `1 / inv_freq` as `freqs=` (`base=None`). The fp32 reciprocal of each period is `inv_freq` bit for bit, so the kernel's angle `pos × (1 / freqs[j])` is vLLM's fp32 angle `pos × inv_freq[j]`. MLX's `base=` path computes the frequencies in an exp2 form that is −2 to +8 fp32 ulps off, an angle error that grows with position. `_freqs` is a private attribute, so it is not a parameter and strict loads of exp_036's shards are unaffected.
2. **Two attention hooks.** `Attention.__call__` reads the RoPE offset from `rope_offset(cache, L)` and the attention mask from `attn_mask(mask, cache, L)` (below). The gate's new port mutants (19, 20, 22, 26) override only these hooks, never `__call__`, so a later fix to `__call__` reaches them too. The refactor changes no output bit for bit.
3. **mlx-lm 0.32.0.** The docstring targets mlx_lm 0.32.0, which executes a `model_file` only with `trust_remote_code=True` (next section).

`tests/test_port_rope_vllm.py` checks all three (exp_037 G1 item 1): the frequencies bit for bit; the rotated vectors against the reference's `rope_neox` within 1e-6 relative L2 per vector at every position 0–20,479 and every 97th position up to 262,143, with the per-octave maxima committed in `tests/fixtures/rope_angle_262143.json`; array offsets (one per sequence, as batch caches pass them) bit for bit against integer offsets; and the port with its hooks bit for bit against the same port running exp_036's `Attention.__call__`, without a cache, through single-sequence caches and through left-padded batch caches.

exp_036's port was `cd6153b8b00b81f7058e786f33672a07b46c494be0203ad75b04372179dc5584`; exp_037's sha256 is recorded at the pre-registration.

## What changed in exp036-port-2 (BUILD_SPEC §5.1)

1. **Head call.** The quantised head gets fp32 activations: `logits = lm_head(h.astype(float32))`. mlx 0.31.2's `quantized_matmul` with an fp32 input and bf16 scales returns fp32, computed from `scale · q + bias` in fp32. exp036-port-1 cast `h` to the scales' dtype (bf16), which gave bf16 logits (port mutant 11).
2. **Defaults.** `make_quant_predicate` and `convert.py` quantise `embed_tokens` and `lm_head` by default (the pre-registered "quantised_head" policy). `--no-quantize-embeddings --no-quantize-lm-head` give the vendor-faithful variant.
3. **Quantising from bf16.** Under the default policy, `sanitize` and `cast_predicate` leave `embed_tokens` and `lm_head` in bf16, so their scales are computed from the bf16 values, exactly as for the peers (bf16 scales and biases).
4. **Router weight in fp32.** `sanitize` stores `mlp.gate.weight` as an exact fp32 upcast; `cast_predicate` keeps it out of the `--dtype` cast; `Router.__call__` casts only `x`. The per-call cast of exp036-port-1 cost about 0.5 GB of memory traffic per token.
5. **Inspection hooks** for the gate, below.

The port follows the exp_036 forward-pass specification (items 1–25). Spec item numbers are cited in the code comments. An independent numpy reference, written separately, checks it before any run counts.

## How it registers

`config.json` carries `"model_file": "kolibri1.py"`. `mlx_lm.utils.load_model` executes that file from the model directory and uses its `Model` and `ModelArgs`. Nothing in site-packages is edited. mlx_lm's `save` copies `*.py` files, so every converted directory is self-contained. Anything that loads through `mlx_lm.load` (`mlx_lm.generate`, `mlx_lm.server`, `mlx_lm.evaluate`) picks the port up (under mlx-lm 0.32.0 with `--trust-remote-code` on the command line). The self-check exercised `mlx_lm.load`, `generate` and `generate_step`.

**Loading under mlx-lm 0.32.0.** mlx-lm 0.32.0 executes a `model_file` only when the caller passes `trust_remote_code=True` (mlx-lm #1385, CVE-2026-5843); without it the load refuses. The kit passes the flag only for its own port, and only after `convert.check_port_file(model_dir)` has passed. `convert.model_file_trust(model_dir)` returns the load's keyword arguments: `{}` when `config.json` names no `model_file` (the peers, built-in model types), `{"trust_remote_code": True}` when it names `kolibri1.py` and the copy is the current `port/kolibri1.py` (otherwise `StalePortFile`, before anything is executed), and `UntrustedModelFile` for any other `model_file`. Every kit load goes through it (`load_model(d, **convert.model_file_trust(d))`), except `convert.py`'s own two loads of a directory it has just written and the gate's `load_port`, which builds the model from this experiment's port module and passes the flag explicitly. The tests pass the flag directly, because they load fixtures written by the kit's own writers.

**The copy runs, not `port/kolibri1.py`.** Each converted directory executes its own copy. A later fix to `port/kolibri1.py` does not reach an existing build by itself; exp_037's builds are APFS clones of exp_036's whose `kolibri1.py` `tools/refresh_builds.py` refreshes to this port (exp_037 DESIGN §2.8). The record's `port_sha256` describes the conversion, not the run. Before any measured run, the run and gate entry points must call `convert.check_port_file(model_dir)`. It requires the copy, the record's `port_sha256`, the record's file entry for `kolibri1.py` and the current `port/kolibri1.py` to have the same sha256, and raises `StalePortFile` otherwise. From a shell:

```bash
python port/convert.py --check-port-file   --out $EXP036_MODELS/Kolibri-1-8bit   # exit 1 if stale
python port/convert.py --refresh-port-file --out $EXP036_MODELS/Kolibri-1-8bit   # seconds, not a re-conversion
```

`--refresh-port-file` copies `port/kolibri1.py` in and rewrites the record: `port_sha256`, `spec_version`, the file entry, `manifest_sha256`, and a `port_refreshes` history entry. Before it rewrites the record, it loads the directory strictly and lazily through mlx_lm. If the new port's parameter tree no longer matches the stored weights, it restores the old file and refuses. A change to `sanitize` or to the quant predicate needs a full re-conversion.

## Inspection hooks (BUILD_SPEC §5.1 delta 5)

| Hook | Returns |
|---|---|
| `Model.forward_hidden(inputs, cache=None)` | the `model.norm` output `[B, T, H]`, the head's input (E11, the gate) |
| `Model.compute_logits(h)` | fp32 logits from final-norm hidden states, under either head policy |
| `Kolibri1Model.make_masks(h, cache=None)` | `(full-attention mask, sliding-window mask)`, as the forward builds them |
| `DecoderLayer.branches(h, mask, cache, force_ids=None)` | `(r_attn, h_mid, r_moe, h_out, router_logits, ids)`: the branch outputs before the residual adds (spec item 29), the stream, the fp32 router logits and the selected ids |
| `SparseMoeBlock.forward_routed(x, force_ids=None)` | `(output, router_logits, ids)` |
| `forced_route(logits_f32, ids, renormalize=False)` | the weights of a fixed selection: sigmoid of the layer's own fp32 logits at `ids` |
| `Attention.rope_offset(cache, L)` (exp_037) | the RoPE position offset for the call's `L` new tokens: `cache.offset` (an int, or one offset per sequence for mlx-lm's batch caches), 0 without a cache. Called on RoPE (sliding) layers only, before the cache update |
| `Attention.attn_mask(mask, cache, L)` (exp_037) | the mask passed to `scaled_dot_product_attention`: `mask` unchanged. Called on every layer, after `cache.update_and_fetch` |

`force_ids` fixes the MoE selection; the routing weights still come from the port's own fp32 logits (the gate's "forced" comparisons). `DecoderLayer.__call__` is `branches(...)[3]` and `Model.__call__` is `compute_logits(forward_hidden(...))`, so one implementation serves the forward, the per-layer harness and generation. A mutant of the layer (`gate/port_mutants.py`) patches what `branches` uses, or `branches` itself; a head mutant patches `compute_logits`. exp_037's position and mask mutants override only `rope_offset` or `attn_mask`.

## Convert

exp_036 ran this on the run host with Python 3.12, mlx 0.31.2 and mlx_lm 0.31.3; exp_037 does not convert again, it clones those builds and refreshes their port file (exp_037 DESIGN §2.8). Run it from this experiment's directory, with the environment of RUNBOOK step 8 (`$PY`, `$EXP036_MODELS`). Nothing is downloaded. The source directory is only read.

```bash
caffeinate -i "$PY" port/convert.py --src "$EXP036_MODELS/Kolibri-1-BF16" \
    --out "$EXP036_MODELS/Kolibri-1-MLX-8bit-g64" --bits 8 --group-size 64
caffeinate -i "$PY" port/convert.py --src "$EXP036_MODELS/Kolibri-1-BF16" \
    --out "$EXP036_MODELS/Kolibri-1-MLX-4bit-g64" --bits 4 --group-size 64
```

Add `--no-quantize-embeddings --no-quantize-lm-head` to both only if the sign-off chose the vendor-faithful head. `--bits` takes 8 or 4.

The outputs come to about 83.1 GB (K8) and 44.0 GB (K4) under the default policy (spec item 20); the vendor-faithful head adds about 1.6 GB.

What `convert.py` does:

1. **Checks the source.** `model_type` must be `kolibri1`. A config with a quantization block (the FP8 repo, or an already converted model) is refused. Every tensor must be BF16, and anything F8 or `weight_scale_inv` is refused. Every shard named in the index must be present, and each shard's size must match its header, which catches unfinished downloads.
2. **Stages the model.** It builds `<out>.staging`, which holds symlinks to the shards, the index, the tokenizer files and `generation_config.json`. It also holds a copy of `kolibri1.py` and a `config.json` with three added keys: `model_file`, and the policy keys `exp036_quantize_embeddings` and `exp036_quantize_lm_head`, which the port's `sanitize`, `cast_predicate` and `quant_predicate` read. They stay in the output's `config.json`.
3. **Converts.** It runs `mlx_lm.convert.convert(quantize=True, q_bits, q_group_size, q_mode="affine", dtype="bfloat16")` into `<out>.partial`, using the port's own quant predicate and no mlx_lm recipe. With `--streaming` it calls `convert_streaming.write_streaming` instead.
   - **Memory guard.** Every 10 s a thread samples `sysctl vm.swapusage` and the process's physical footprint (`proc_pid_rusage`, which counts MLX's Metal buffers; `ps` RSS does not). If swap has grown by more than 2 GB, or the footprint passes 0.8 × `hw.memsize`, it interrupts the conversion. `convert.py` removes `<out>.partial` and the staging directory and exits with the `--streaming` command to run instead. If the conversion does not stop within 120 s, the guard removes them itself and ends the process.
4. **Restores the tokenizer files byte for byte.** Under transformers 5.18, mlx_lm's `save_pretrained` rewrites `tokenizer_config.json`, drops `added_tokens_decoder` and `additional_special_tokens`, and renumbers the reserved special tokens (`<|reserved-token-2|>` moves from 127925 to 127923, and so on).
5. **Verifies the output and records it.** It reads every saved tensor header back and checks the config, the policy keys and the tensor policy (below), including bf16 scales and biases on every quantised tensor. It writes `exp036_convert_record.json` (sorted keys) and renames `<out>.partial` to `<out>`. The record holds:
   - the source: path, revision (from `.cache/huggingface/download/*.metadata`), index and config sha256, size;
   - bits, group size, mode, dtype, the output's `quantization` block, the flags, `head_policy` (`quantised_head` or `vendor_faithful`) and `tensor_policy` per tensor class, as read back from the output (`arm_bits`, `bf16` or `fp32`, the vocabulary of `gate/thresholds.json`);
   - the port's sha256 and spec version, the converter, mlx, mlx-metal, mlx-lm and Python versions;
   - `t_start`, `t_end` (UTC), peak RSS, peak physical footprint and the memory-guard summary (swap growth, samples, limits);
   - the sha256 and size of every output file, and `manifest_sha256`: sha256 over the sorted lines `relpath<TAB>sha256<LF>` (the record itself excluded). `convert.directory_manifest_sha256(dir)` recomputes it from the files.

   Paths go through `tools/redact.py:redact_path`.
6. **Copies the record** to `results/convert/convert_<bits>bit_<UTC>.json`, after `runner.guard.require_identity()`, never overwriting. `--results-dir none` skips the copy.

An interrupted run leaves `<out>.staging` or `<out>.partial` behind, and the next run removes them. A non-empty `--out` is refused unless you pass `--force`, which deletes the previous output before converting. It deletes only directories whose contents look like a converted model.

Quantisation policy (`make_quant_predicate`; `tensor_policy` in the record):

| Tensor | Default (`quantised_head`) | `--no-quantize-embeddings --no-quantize-lm-head` (`vendor_faithful`) |
|---|---|---|
| attention q/k/v/o, routed experts (`switch_mlp`), shared expert | affine, `--bits`, `--group-size`, bf16 scales and biases | same |
| `embed_tokens` | affine at `--bits` from its bf16 values (`QuantizedEmbedding`) | BF16 |
| `lm_head` | affine at `--bits` from its bf16 values; fed fp32 activations, so the logits are fp32 | FP32, an exact upcast; logits `h.f32 @ W.f32ᵀ` |
| router `mlp.gate.weight` | FP32, an exact upcast; a custom module with no `to_quantized`, so no recipe can quantise it | same |
| `mlp.gate.expert_bias` | FP32, an exact upcast of the BF16 checkpoint value | same |
| all RMSNorm weights | BF16 | same |

A BF16 source loaded directly (tests, the gate's layer-wise checks) keeps a bf16 `lm_head` and upcasts it per call in `compute_logits`; no measured run takes that path.

**`--streaming`** (`convert_streaming.py`) applies the steps mlx_lm takes (sanitize, the `--dtype` cast, `module.to_quantized`) to one group of tensors at a time: the embedding, each decoder layer, then the final norm with the head. It writes one shard per group, so the file list and the manifest differ from an mlx_lm conversion, but every tensor's name, dtype, shape and bits are the same (`test_streaming_bit_identical`, `test_streaming_casts_like_mlx_lm`). Peak memory is about one layer.

Don't use `mlx_lm.convert --quant-predicate mixed_*` on this model. Those recipes put `lm_head` at 6 bits and give the embedding low bits.

`--kv-bits` runs: the sliding-window caches (513 tokens) stay BF16 and only the 10 full-attention caches are quantised. The port's `SlidingKVCache.to_quantized` returns itself, where mlx_lm's `RotatingKVCache` raises. exp_036 does not use it; every arm keeps a BF16 KV cache.

**`--kv-bits` costs memory during long prefills.** With a quantised cache, every prefill chunk on the full-attention layers goes through mlx_lm's unfused `quantized_scaled_dot_product_attention`. That function materialises the score matrix for all 48 heads × the chunk × the whole context. The transient is about 100 B × `--prefill-step-size` × context length (measured 96–103 B on the real head geometry):

| Context | Step 2048 (default) | Step 256 |
|---|---|---|
| 16k | 3.3 GB (measured) | 0.4 GB |
| 256k | ≈ 52 GB | ≈ 6.5 GB |
| 1M | ≈ 209 GB | ≈ 26 GB |

Without `--kv-bits`, the fused kernel keeps the transient flat (0.02 GB measured at 16k). So `--kv-bits` saves cache memory but needs a small `--prefill-step-size` at long context. With the default step, a 256k prefill of the 8-bit build exceeds 128 GB.

## Known deviations from the vLLM serving path

- **Residual add.** vLLM adds the residual in fp32 inside `fused_add_rms_norm` and normalises the unrounded sum. MLX adds in BF16 and then normalises. The norm input differs by at most 1 BF16 ulp. The RMSNorm itself matches: `mx.fast.rms_norm` rounds the normalised value to BF16 before the weight multiply, as vLLM does.
- **Quantised head and embedding (default policy).** vLLM serves Kolibri with an unquantised head and fp32 logits (`head_dtype: float32`). The pre-registered default quantises `lm_head` and `embed_tokens` at the arm's bits, like the peers, and still produces fp32 logits from fp32 activations. This is HYPOTHESIS C1's stated deviation; E11 measures what it costs. The vendor-faithful build is one flag pair away.
- **No FP8.** The model card's scores were produced with FP8 E4M3 128×128 block weights, dynamic FP8 activations and FP8 KV. The port starts from the BF16 repo and quantises to affine 8-bit or 4-bit (group 64), which has a different error model. MLX has no FP8 matmul.
- **Rounding inside the attention and MoE kernels.**
  - **RoPE:** the angle is vLLM's fp32 angle (`VllmRoPE`, exp037-port-1). MLX computes cos and sin in fp32 and rotates in fp32, rounding once; vLLM stores its cos/sin cache in BF16. Neither the port nor the reference is bit-faithful to that BF16 rounding, which is flat in position.
  - **Router logits:** fp32 matmul of upcast BF16 operands, which matches vLLM's fp32-accumulating GEMM up to summation order.
  - **MoE weighted sum:** computed in fp32 and then cast to BF16.
  - **Expert ties:** `argpartition` and `torch.topk` may break exact ties differently.
- **Batch-size dependence.** MLX's BF16 and quantised GEMMs round differently for different numbers of rows. Chunked prefill, single prefill and decode can therefore differ at the ulp level, and that difference can flip near-tied routes. On a tiny random checkpoint in fp32, prefill and decode agree to within 5e-6, and the port agrees with an fp64 numpy forward to within 1e-6 relative.

## Limitations

- **Saved prompt caches.** `mlx_lm.models.cache.load_prompt_cache` restores caches by class name, so a prompt cache saved with `SlidingKVCache` cannot be reloaded. In-process reuse (`mlx_lm.server`) is unaffected.
- **Batched generation with `--kv-bits`.** mlx_lm's `BatchRotatingKVCache` raises on quantisation.
- **Scoring long sequences without a cache.** Called as `model(ids)` with no cache and more tokens than the window (513), the port gets a dense [T, T] boolean window mask from mlx_lm's `create_attention_mask`, plus intermediates of the same size. Memory therefore grows with T²: the mask helper alone peaks at 0.4 GB at 16k tokens, and the mask is 17 GB at 128k. `mlx_lm.perplexity` (default sequence length 512) and any harness that scores whole sequences in one call take this path. `generate`, `server` and `evaluate` pass a cache, and their masks are only [chunk, chunk + 512]. Score long texts through `model.make_cache()` in chunks: concatenate `model(ids[:, i:i+2048], cache=c)` over i. That is the teacher-forcing pattern the gate and H8 use (BUILD_SPEC `teacher_force_logprobs(model, ids, chunk=2048)`).
- **FP8 repo.** `Aleph-Alpha/Kolibri-1` is not supported. Use `Aleph-Alpha/Kolibri-1-BF16`.
- **Out-of-scope architecture options.** Tied embeddings, rope scaling and a `head_dtype` other than float32 are refused.
- **`use_sliding_window`.** It defaults to false, as in transformers' `Qwen3MoeConfig`, the base class of the vendor's config. A config with sliding layers must set it to true, or the port refuses it, as the vendor plugin does. The released config sets it.
- **Config changes made by mlx_lm.** The output `config.json` gets `eos_token_id: [127906, 127901]` because mlx_lm merges `generation_config.json` into it. mlx_lm merges the same list at load time anyway.
- **Tokenizer warning.** The "incorrect regex pattern … fix_mistral_regex" warning from transformers 5 is spurious for Kolibri. It fires because `config.json` has no `transformers_version`, and the regex is changed only if `fix_mistral_regex=True` is passed, which this kit does not do.
- **Per-token cost** at batch 1 (spec item 20): the fp32 router weights are 197 MB over 50 layers (0.2 GB read per token, 98 MB more than bf16, with no per-call cast). The quantised head reads about 0.35 GB (8-bit) or 0.18 GB (4-bit) per token; the vendor-faithful fp32 head 1.31 GB. Decode reads about 4.0 GB per token (K8) and 2.2 GB (K4) in all.

## Licence and attribution

`kolibri1.py` is derived from [aleph-alpha-inference](https://github.com/Aleph-Alpha/aleph-alpha-inference) (Copyright 2026 Aleph Alpha GmbH, Apache-2.0, commit 049a6a7bd2405b27d6d280d256bd3d585191c7ae) and was modified for MLX by Miktam for Chronos exp_036 on 2026-10-03 and for Chronos exp_037 on 2026-10-06 (vLLM-angle RoPE, attention hooks). It is licensed under Apache-2.0; see `../LICENSE-APACHE-2.0` and `../NOTICE`. It builds on mlx_lm (MIT, Apple Inc.). `convert.py`, `convert_streaming.py` and this README fall under the repository's MIT licence. The Kolibri-1 weights are Aleph Alpha's, released under Apache-2.0.
