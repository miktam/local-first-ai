# exp_036 integration log

*2026-10-03. Port `exp036-port-1` (port/kolibri1.py sha256 `11eeae75…8b24c`), reference `exp036-ref-1`, mlx 0.31.2, mlx_lm 0.31.3, numpy 2.5.3, Python 3.12, on miktam-mini (M4 Pro). The run host is the M5 Max.*

## Outcome

- The port and the reference, written independently, agree in fp32 on both tiny presets and both weight seeds:
  - logits to 1.3e-6–1.8e-6 relative, with the same argmax at every one of 160 positions;
  - the residual stream after every layer to ≤ 2.0e-6;
  - the same expert set at every layer for every token.
- The KV-cache paths reproduce the full forward to ≤ 1.6e-6 in fp32: prefill below and above the window, token-by-token decode wrapping the rotating buffer 3–9 times, and chunked prefill below and above the window.
- Converted 8-bit and 4-bit checkpoints match the reference run on their own dequantised weights at bf16 level (KL ≤ 9.7e-5).
- **No defect was found in `port/` or `reference/`, and neither was edited.** Every failure during integration traced to a test threshold that assumed well-separated logits and routing. Tiny random weights have neither. Entries 1–3 hold the diagnoses.
- Final suite:
  - with the exact command: `157 passed, 11 skipped`. The 11 skips are the tokenizer tests, which have no `EXP036_TOKENIZER_DIR`;
  - with the tokenizer: 168 passed;
  - with `port/` and `reference/` absent: no errors.
- One coverage gap was closed: entry 4.

Spot check against the vendor semantics, since two independent implementations that agree could share a misreading (vLLM v0.29 sources):
- `flash_attn.py` and `triton_attn.py` set the window to `(sliding_window - 1, 0)`: 513 keys including the query, as both sides implement.
- `arg_utils.py` sets `CacheConfig.sliding_window` only when every layer is sliding, so full layers have no window. Both sides agree.
- `get_rope` defaults to `is_neox_style=True`, and `rotate_neox = cat(-x2, x1)`: half rotation, as both sides implement.
- The vendor `sigmoid_logit_add_routing` selects `topk(logits + bias)`, weights by `sigmoid(logits[ids])` and does not renormalise. Both sides match.
- The vendor decoder threads the residual through the norms in the same order as the plain loop both sides use (spec item 5).

---

## Entry 1: bf16 port vs fp32 reference misses top-1 ≥ 0.98 and mean KL ≤ 1e-3

**Symptom.** In `test_port_vs_ref.py` (b), the port in bf16 as loaded against the fp32 reference on a 160-token sequence:
- top-1 agreement 0.81–0.94;
- mean KL(ref‖port) 1.6e-3–7.5e-3, with a single position up to 0.12.

The fp32 port on the same inputs is exact (1.8e-6).

**Diagnosis.**
1. **Per-op precision.** Each port module was fed the same bf16 input and its bf16 output compared with the same module in fp32 (layer 0, vendor preset):

   | Module | Max relative error |
   |---|---|
   | `input_layernorm` | 3.6e-3 |
   | `q_proj` | 2.2e-3 |
   | `q_norm` / `k_norm` | ≤ 4.3e-3 |
   | RoPE | 2.1e-3 |
   | attention | 4.4e-3 |
   | shared expert | 4.4e-3 |
   | MoE | 6.3e-3 |
   | router logits | 0 (bit-identical, fp32) |

   All are within about one bf16 unit roundoff (u = 2⁻⁸ = 3.9e-3). No operation loses extra precision.
2. **Routing flips.** Recording the port's expert ids per layer and comparing them with the reference's showed:
   - In the first layer, where only one layer of bf16 noise has built up, flips happen only at reference selection margins of 0.0008–0.02. That is the size of the bf16 router-logit noise: the router input is about 1% off, on a router-logit scale of about 1.6.
   - Each flip moves that token's residual by up to about 40%. The change cascades into later layers and, through attention, into other tokens.
   - On pattern5 (16 experts, top-6, 10 layers), 40–52% of tokens flip in at least one layer.
   - The reference's 1st-percentile selection margin is 0.002–0.04.
3. **Independent control.** A numpy bf16 emulation of the reference rounds activations to bf16 wherever the port stores bf16, keeps the router logits, routing weights and LM head in fp32, and uses no MLX code. Against the fp32 reference it scores top-1 0.863–0.975 and KL 2.0e-4–3.9e-3. That is the same range as the port. A correct bf16 implementation of these weights cannot meet the original bounds.
4. **Forced routing.** With the port forced onto the reference's expert ids (weights from its own fp32 logits, per spec item 12):
   - KL 5.7e-5–1.1e-4 and relative error 1.6e-2–2.3e-2;
   - top-1 1.000 on every position where the reference leads by ≥ 0.1;
   - every disagreement at a lead ≤ 0.031.

   These logits are flat (std 0.8 over 1024 ids), and 6–11% of positions lead by less than 0.02.
5. **Resolving power of the end-to-end bf16 comparison.**
   - A gross routing bug (expert_bias zeroed) gives top-1 0.18–0.24 and KL 0.13–0.18.
   - A semantic bug under forced routing (traditional RoPE) gives KL 0.16–0.19.
   - ulp-level deviations are invisible end to end: router logits rounded to bf16 and routing weights in bf16 score the same as correct bf16.

**Root cause.** Neither side is wrong. bf16 precision on tiny random weights is dominated by routing near-ties and flat logits. The port follows spec item 23 (bf16 activations) and items 11–12 (fp32 router logits and routing weights). vLLM serving in bf16 is subject to the same near-tie flips.

**Fix (tests only).** (b) is split in two, with thresholds derived in the docstrings.
- **(b1) forced routing:**
  - mean KL ≤ 1e-3, the original bound, kept;
  - max relative error ≤ 5e-2, which is √n·u for n ≈ 160 sequential roundings;
  - top-1 ≥ 0.98 on decisive positions (lead ≥ 0.1).
- **(b2) natural routing:** worst-case bounds that assume every position is flip-affected at a per-position KL of ≤ 0.03 (δ_rms 0.24).
  - mean KL ≤ 3e-2;
  - top-1 ≥ 0.60. That is the expected loss averaged over the reference's own leads (25–29%) plus 3 binomial standard deviations over 160 positions.

  Both bounds still separate the gross-bug control.

I first set (b2) at top-1 ≥ 0.70, on the assumption that at most half the tokens flip. The measurements then showed 52% of tokens flipped on pattern5, and 0.775 for the 8-bit build (entry 3). No test had failed, but the assumption behind the bound was false, so I re-derived the bound from the worst case. It was not loosened to make anything pass.

## Entry 2: bf16 cached decode vs full forward misses argmax ≥ 0.98

**Symptom.** In `test_cache_decode.py`, bf16 with natural routing, the cached run against the full forward gives top-1 0.92–0.98. All 8 pattern5 cases fall below 0.98. In fp32 every plan agrees to ≤ 1.6e-6.

**Diagnosis.**
- The mechanism is entry 1's. MLX bf16 GEMMs round differently for different row counts: one row per decode step against T rows in one forward. The port's own report saw the same (chunked 16/16/8: top-1 96%).
- Forcing the cached run onto the full forward's own expert ids gives top-1 1.000 on decisive positions in all 16 cases, with relative error 1.1e-2–1.8e-2.
- fp32 rules out any mechanical cache error:
  - RoPE offsets, mask and rotating-buffer trim are exact;
  - the buffers end at W keys after decode and W−1+S after a prefill chunk;
  - the caches are `SlidingKVCache(max_size=W, keep=0)` and `KVCache` as spec item 24 requires.

**Root cause.** Neither side; bf16 precision, as in entry 1. Spec items 10 and 24.

**Fix (tests only).** The same split:
- forced routing: top-1 ≥ 0.98 on decisive positions and relative error ≤ 5e-2;
- natural routing: the (b2) worst-case bounds.

Separately, my first vendor sequence (300 tokens) left the long-prefill plan only 167 decode steps, fewer than three wraps of the 65-slot buffer. The test's own precondition caught it, and the vendor sequence is now 340 tokens. That was a design error in the test, not a finding.

## Entry 3: quantised build vs fp32 reference misses top-1 ≥ 0.95 (8-bit) and ≥ 0.85 (4-bit)

**Symptom.** In `test_convert.py`, against the fp32 reference on the original weights:

| Build | Routing | Top-1 | Mean KL |
|---|---|---|---|
| 8-bit g64 | natural | 0.775 (0.856–0.875 at two probe seeds) | 1.2e-2 |
| 4-bit g64 | natural | 0.37 | 6.2e-2 |
| 4-bit g32 | natural | 0.46 | 5.3e-2 |

**Diagnosis.**
- **Separating conversion from quantisation noise.** The test writes the converted tensors back as an F32 checkpoint, dequantised with `mx.dequantize` from the saved files (not through the port's loader), and runs the reference on it. The quantised port, forced onto that run's experts, agrees with it at the bf16 level of entry 1 for all three builds:
  - KL 8.4e-5–9.7e-5;
  - relative error 1.7e-2–1.9e-2;
  - top-1 1.000 on decisive positions.

  So convert.py and the quantised load path add nothing beyond the quantisation itself.
- **What remains is affine noise.** On Gaussian weights, a group of 64 spans about 4.8σ, so the per-weight error is 4.8σ/(2ᵇ−1)/√12:
  - 8-bit: 0.54% of σ, about twice the bf16 activation rounding. Forced onto the reference's experts it scores decisive top-1 1.000 and KL 3.0e-4.
  - 4-bit: 9.2% of σ, which puts the logit noise at about 0.2–0.35. That is the same size as the median top-1 lead (0.13–0.18).

**Root cause.** Neither side; the requested thresholds don't fit tiny random weights. Spec items 18 and 23.

**Fix (tests only).**
- New `test_quantised_port_vs_reference_on_dequantised_weights`, a sharp check with the (b1) bounds, for all builds.
- The comparison against the original weights keeps derived smoke bounds:
  - 8-bit: the (b2) worst-case bounds (top-1 ≥ 0.60, KL ≤ 3e-2), plus forced-routing top-1 ≥ 0.98 on decisive positions and KL ≤ 1e-3;
  - 4-bit: top-1 ≥ 0.25. About 3–4 candidates sit within one noise σ of the maximum; even a uniform choice among them keeps about 0.3.
- The docstring states that the 4-bit bound cannot tell a bug from 4-bit noise; the dequantised-reference test does that.

None of this is the real-weight gate.

## Entry 4: coverage gap for spec item 11 (router logits in fp32)

**Symptom.** A port mutant that computes the router logits in bf16 (`(x @ W.T).astype(float32)`) passed every test.

**Cause.**
- test_routing feeds `route()` fp32 logits that it builds itself.
- The end-to-end bf16 tests cannot see a change of u·|logit| ≈ 6e-3, which is below the routing-flip noise of entry 1.

**Fix.** `test_router_logits_are_fp32_of_bf16_operands` checks the `Router` module directly. Its weight must be bf16 and its bias fp32. Its output must be fp32 and within 1e-5 of a float64 product of the same bf16 operands. Measured: 9.9e-8; bf16 rounding would be 2.0e-3 off. The mutant now fails this test and only this one.

## Mutation check of the new tests

Each mutant was applied to a scratch copy of `port/kolibri1.py`, and the three new modules were run on it.

| Mutant | Caught by |
|---|---|
| window W−1 | (a), (b1), (b2), (c), both convert numerics tests |
| traditional RoPE | the above, plus fp32 cache |
| RoPE offset ignored with cache | all three cache tests |
| 2 sink tokens kept | all three cache tests |
| W−1 cache slots | all three cache tests |
| full layers windowed | everything numeric |
| select on sigmoid(logits)+bias | (a), (b2), (c), convert vs fp32. Forced routing is blind to the selection rule by design. |
| shared expert ×0.5 | (a), (b1), (b2), (c), both convert numerics tests |
| embed quantised by default | convert: verify_output refuses, so every convert test errors |
| pre-MoE and post-FFN norms swapped | (a), (b1), (b2), (c), both convert numerics tests |
| router logits in bf16 | the router unit test (entry 4) |

As expected, a window mutant passes the cache tests, because they compare the port with itself. The port-vs-reference tests catch it.

## Files added or changed in tests/

- **New:**
  - `port_harness.py`: load, forced and recorded routing, per-layer capture, comparison statistics;
  - `test_port_vs_ref.py` (17 tests);
  - `test_cache_decode.py` (49);
  - `test_convert.py` (18);
  - this log.
- **`conftest.py`:** added the session fixture `tiny_checkpoint_factory(preset, seed)`.
- **`test_convert.py`** writes a word-level tokenizer offline, because mlx_lm's convert loads one. The Kolibri tokenizer is not needed.

## Still open

- **The real-weight gate**, KL and top-1 of the 8-bit and 4-bit builds against the reference on real prompts, has not run. These tiny-model thresholds must not be reused for it.
- **Peak memory and time for convert.py** on the 156 GB source are untested (port report, spec item 25b).
- **Metal kernels on the M5 Max** may round differently from the M4 Pro. Only the natural-routing bounds depend on that, and they are worst-case derived. All fp32 assertions have ≥ 50× headroom.

---

# Review fixes (2026-10-03, after the mutation pass and the code review)

Two things happened after the entries above.

**The mutation pass.** It added two things:
- `tests/test_mutation_guards.py::test_bf16_model_logits_are_fp32_of_bf16_operands`, parametrised over both presets. It kills M18a, logits in bf16, which was the one survivor out of 29 mutants.
- The `$EXP036_PORT_FILE` hook in `tests/tiny_checkpoint.py`.

**The review** returned six findings. Each one was checked before any change; the results are below.

Unlike the entries above, **`port/` and `reference/` were edited**. The new files:

| File | sha256 |
|---|---|
| `port/kolibri1.py` | `1a5f79d4…ac425` |
| `port/convert.py` | `bf35281a…9fb35` |
| `reference/kolibri_ref.py` | `0e269ef3…bbe1` |

No threshold was changed. **This supersedes the Outcome bullet "neither was edited".**

## Finding R1 (minor): `--kv-bits` prefill memory grows with context

- **Finding.** With `--kv-bits`, full-layer prefill goes through mlx_lm's unfused `quantized_scaled_dot_product_attention`. The port README said only "supported".
- **Verdict: confirmed.**
  - The code path is `mlx_lm/models/base.py:64-105`: `scaled_dot_product_attention` dispatches there when `hasattr(cache, "bits")`, and it materialises the scores.
  - I measured one attention call with the real head geometry (48 q / 4 kv, head_dim 128), a 2048-row chunk and an 8-bit `QuantizedKVCache`. The transient was 0.78 / 1.53 / 3.04 GiB at 4k / 8k / 16k context. With a bf16 `KVCache` it stayed at 0.02 GiB. That is about 97 B per (query, key) pair; the reviewer measured about 103 B.
- **Change: documentation only.**
  - `SlidingKVCache` docstring and the port README now say that `--kv-bits` needs a small `--prefill-step-size` at long context.
  - The README gives the formula (about 100 B × step × context) and a table: 16k 3.3 GB measured; 256k ≈ 52 GB at step 2048 and ≈ 6.5 GB at step 256; 1M ≈ 209 GB and ≈ 26 GB.
  - Both now state that exp_036 does not use `--kv-bits`. HYPOTHESIS: "KV cache is bf16 for every model; no --kv-bits". BUILD_SPEC item 19 says the same.
- **Not done:** the reviewer's optional rewrite, which would dequantise the full-layer K/V for prefill chunks and use the fused kernel. It would add a numerics path that no exp_036 run exercises.

## Finding R2 (minor): a call without a cache builds a dense [T, T] window mask

- **Finding.** `model(ids)` without a cache and with T > 513 gets a dense window mask from `create_attention_mask`.
- **Verdict: confirmed.** `base.py:44-55` takes `create_causal_mask(N, window_size=W)` when `cache` is None and `N > window_size`, and that allocates [T, T] booleans. The mask helper alone peaked at 0.03 / 0.11 / 0.38 GiB at 4k / 8k / 16k, which is quadratic.
- **Change: documentation only.**
  - `Kolibri1Model.__call__` docstring, a port README limitation, and the `tests/port_harness.port_logits` docstring.
  - They give the figures (0.4 GB at 16k, 17 GB at 128k for the mask alone) and the chunked teacher-forcing pattern. That pattern is the one BUILD_SPEC already prescribes for the gate and H8: `teacher_force_logprobs(model, ids, chunk=2048)`.
- **Not done:** the optional internal chunking in `Kolibri1Model.__call__`. No exp_036 path calls the model without a cache on long input. The gate texts are 1,536 tokens, so their mask is 2.4 MB. `mlx_lm.perplexity` defaults to 512-token sequences, below the window. Whole-sequence scoring also already pays 0.5 MB per token for fp32 [T, V] logits.

## Finding R3 (minor): converted builds run a stale copy of `kolibri1.py`

- **Finding.** Converted builds run their own copy of `kolibri1.py`, and nothing compares it with `port/` at load time.
- **Verdict: confirmed.**
  - `mlx_lm/utils.py:325-331` executes `<model_dir>/kolibri1.py`.
  - `convert.py` checked the copy's hash only at conversion time.
- **Change: `port/convert.py`.** BUILD_SPEC §5.1 already names `--refresh-port-file`. The additions:
  - `check_port_file(model_dir, port_file=PORT_FILE)` raises `StalePortFile` unless four hashes are equal: the copy, `record["port_sha256"]`, `record["files"]["kolibri1.py"]["sha256"]` and `port/kolibri1.py`.
  - `refresh_port_file(out)` copies the port in and drops any stale `kolibri1.*.pyc`. It then loads the build strictly with `lazy=True`, so no weight data is read. On success it rewrites the record: `port_sha256`, `spec_version`, the file entry, and a `port_refreshes` history list. On a load failure it restores the old file and exits with "convert again".
  - The CLI gets `--check-port-file` and `--refresh-port-file`, which are mutually exclusive. `--src` and `--bits` are now required only to convert.
  - The port README explains the copy semantics and requires every run and gate entry point to call `check_port_file`.
- **Tests,** in `tests/test_convert.py`, each run on the 3 conversion variants, always on a copy of the shared output:
  - `test_fresh_build_passes_the_port_file_check`;
  - `test_stale_port_file_is_refused_and_refreshed`, which covers a port that moved on, a hand-edited copy, refresh, the no-op second refresh, and the CLI check and refresh with the real port;
  - `test_refresh_refuses_a_port_that_does_not_load_the_weights`: renaming `shared_experts` fails the strict load, the old file comes back, and the record is unchanged.
- **Open:** `runner/guard.py` and `gate/run_gate.py` do not exist yet. When they are built, they must call `check_port_file`.

## Finding R4 (major): the reference CLI and `moe_block` API fall short of BUILD_SPEC §5.2, and G2 cannot get its near-tie gap

- **Finding.** The CLI and `moe_block` don't provide what BUILD_SPEC §5.2 asks for. G2's near-tie gap can't be rebuilt from the saved outputs, and `DERIVATION.md` is missing.
- **Verdict: confirmed.**
  - `moe_block` returned only `{weights, ids, experts_loaded}`.
  - The CLI had no `--dump`.
  - The 7th biased score was never saved, and the pre-MoE input was not saved either.
  - `reference/DERIVATION.md` did not exist, although HYPOTHESIS cites it at lines 112 and 165.
- **Change: option (a), made before the pre-registration freeze.**
  - `moe_block` info now also carries `logits` (raw fp32), `biased` (logits + expert_bias, the same fp32 sum `route_ref` selects on) and `top6`. `ids` stays as an alias.
  - `forward_packed(..., on_layer=None)` calls `on_layer(i, h_in, h_out, info)` after each layer, and `self.last_layer_info` holds the latest info.
  - The new CLI flag `--dump DIR` writes into an empty or new `DIR`:
    - per layer: `layerNN.{h_in,h_out,router_logits,top6,top6_gap}.npy`, where `top6_gap` is the k-th minus (k+1)-th biased score, the G2 exemption quantity;
    - once: `logits.npy`, `logprobs.npy` (float64 log-softmax stored as fp32), `final_norm.npy`, `ids.npy` and `seq_lengths.npy`;
    - last: `ref_record.json` with sorted keys, holding the sha256, bytes, shape and dtype of every file, the checkpoint fingerprint, config sha, version, seq_lengths, per-layer seconds and experts, wall time, peak RSS and UTC.
  - `--out` is now optional when `--dump` is given.
  - New `reference/DERIVATION.md`: every reference block mapped to vendor `kolibri1.py` lines at 049a6a7 and to vLLM v0.29.0 file:line, each line checked against the saved sources. Two kernel conventions, GQA grouping and the weighted sum inside the fused MoE kernel, are marked as having no single Python line.
  - The reference README documents the dump, the new info keys and the new file.
- **Numerics unchanged.** `forward_packed` before and after the edit gives bit-identical logits, hidden states, final norm and lengths on both presets: seed 1, sequences of 150 and 40 tokens, compared with `np.array_equal`.
- **Tests,** in the new `tests/test_reference_cli.py`:
  - info keys and their values (`biased == logits + bias`, `top6` and `weights` equal `route_ref`'s output);
  - `test_dump_matches_the_run_and_the_checkpoint`, for both presets: the record's sha256 and size match every file, and the dump agrees with the same run's `.npz`. In detail, `h_out` equals `hidden`, `layer00.h_in` equals the checkpoint embedding rows, `h_in[i]` equals `h_out[i-1]`, and the dumped router logits plus the checkpoint bias reproduce `top6` and the weights exactly. `top6_gap` matches `port_harness.selection_margin` to rtol 1e-6, and `logprobs` matches a float64 log-softmax to 1e-6 relative;
  - `--num-layers 2` writes only two layers;
  - a non-empty `--dump` directory is refused;
  - neither `--out` nor `--dump` is an error.
- **Open:** the prereg drafts are outside this agent's scope. HYPOTHESIS and BUILD_SPEC should name the dump file layout above. The dump writes `top6` as int64 ids rather than "fp32", and adds `top6_gap.npy`, `logits.npy`, `final_norm.npy`, `ids.npy` and `seq_lengths.npy`.

## Finding R5 (minor): `use_sliding_window` defaults to True, the vendor to False

- **Finding.** Both of ours default `use_sliding_window` to True, while the vendor defaults it to False.
- **Verdict: confirmed.**
  - The vendor's `config.py:14` is `Kolibri1Config(Qwen3MoeConfig)`.
  - transformers 5.18 `configuration_qwen3_moe.py:99` has `use_sliding_window: bool = False`, and `:115` nulls the window.
  - Checked live: `Qwen3MoeConfig(sliding_window=513).sliding_window is None`.
  - The released config sets `true`, so this changes nothing today.
- **Change.**
  - Port `ModelArgs.use_sliding_window` now defaults to False. The refusal message names the key.
  - Reference: `d.get("use_sliding_window", False)`, with the message "is false or missing".
  - The port README lists the refusal; the reference README lists it in the rejection list.
- **Tests,** in the new `tests/test_config_refusals.py`:
  - port and reference both refuse a config whose key is missing or false;
  - both accept one with it true;
  - a config with only full layers needs no key;
  - the transformers default itself is checked.

  A mutant that reverts either default to True makes the matching `missing` case fail; checked with a mutated in-memory copy.

## Finding R6 (minor): `--out` without `.npz` prints a path that doesn't exist

- **Verdict: confirmed.** `np.savez` appends `.npz`.
- **Change.** `--out` gets `.npz` appended when it lacks the suffix. The final line prints every path actually written, the npz and/or `ref_record.json`.
- **Test.** `test_out_without_npz_suffix_names_the_written_file`.

## Mutation check after the fixes

- The 30 mutants were regenerated from the edited `port/kolibri1.py`. Every exact replacement still matched its expected count, and the null copy is byte-identical to the port.
- They were rerun through the full suite (`$EXP036_PORT_FILE`, `-x`). All 29 non-null mutants are killed, by the same first killers as in the mutation pass. M18a is still killed by `test_mutation_guards`.
- The null control: 184 passed, 11 skipped.
- The new R3–R6 tests import `port.kolibri1`, `port.convert` and `reference.kolibri_ref` directly. They are therefore not reachable through `$EXP036_PORT_FILE`. Their discriminating power was checked separately, as noted under each finding.

## Suite

- With the exact command: `184 passed, 11 skipped`. There are 25 new tests; the 11 skips are the tokenizer tests.
- With `EXP036_TOKENIZER_DIR` set: 195 passed.


---

# BUILD_SPEC deltas for port and reference (2026-10-03, area "port-ref")

The port moved to `exp036-port-2` (BUILD_SPEC §5.1) and the reference to `exp036-ref-2` (§5.1b, §5.2). Files at the end of this round:

| File | sha256 | |
|---|---|---|
| `port/kolibri1.py` | `cd6153b8…79dc5584` | changed |
| `port/convert.py` | `6c16fc4b…a44c29a2` | changed |
| `port/convert_streaming.py` | `be9a72f1…c5cdfb40` | new |
| `reference/kolibri_ref.py` | `f621b094…78e88f6f` | changed |
| `reference/mlx_affine_np.py` | `c6fe955f…2fdc26bd` | new |
| `reference/mutants.py` | `92632f4d…c960b748` | new (Apache-derived; NOTICE.addendum.port-ref.md) |
| `reference/safetensors_np.py` | `c11c63fb…ce5db2f5` | U32 added |
| `reference/` tree | `85337ed7…9e251bbd` | equals `tools/hash_tree.py --tree reference` and `gate.common.reference_tree_sha256()` |

No threshold of entries 1–4 was changed. Every pre-existing test still passes; three were extended to the new behaviour (below).

## Entry 5: the port deltas

1. **Head call.** `Model.compute_logits` feeds the head `h.astype(float32)`. For a quantised head, mlx 0.31.2's `quantized_matmul` with an fp32 input returns fp32 logits computed from `scale · q + bias` in fp32. Measured on a 512 × 256 tensor: 6.5e-7 against the fp32 dequantised weight, and 5.4e-3 against the bf16-rounded one. exp036-port-1 cast `h` to bf16 here, which is port mutant 11.
2. **Defaults.** `make_quant_predicate(quantize_embeddings=True, quantize_lm_head=True)`; `convert.py` takes `--no-quantize-embeddings` / `--no-quantize-lm-head`.
3. **Quantising from bf16.** `sanitize` runs inside `mlx_lm.convert`'s load, before quantisation, so it cannot see whether the head will be quantised. `convert.py` therefore writes two policy keys into the staging `config.json`: `exp036_quantize_embeddings` and `exp036_quantize_lm_head`. `ModelArgs` reads them, defaulting to true. `sanitize` upcasts `lm_head` only when the key is false, and `cast_predicate` then keeps it fp32; otherwise both stay bf16 and are quantised from bf16. `test_head_and_embedding_values` checks that the saved head and embedding are bit-identical to `mx.quantize` of the stored bf16 tensor, with bf16 scales.
4. **Router weight fp32.** `sanitize` upcasts `mlp.gate.weight` exactly, `cast_predicate` excludes it, and `Router.__call__` computes `x.astype(float32) @ W.T`. `test_router_logits_are_fp32_of_bf16_operands` now asserts an fp32 weight equal to the bf16 checkpoint read by mlx's plain loader. Its output assertions are unchanged.
5. **Hooks.** `Model.forward_hidden`, `Model.compute_logits`, `Kolibri1Model.make_masks`, `DecoderLayer.branches(h, mask, cache, force_ids)`, `SparseMoeBlock.forward_routed` and `forced_route`. `__call__` delegates to `branches` and `compute_logits`, so one implementation serves the forward, the per-layer harness and generation. The gate's `port_mutants.py` already patches against these (its mutants 10 and 11 are cross-checked in `test_port_ref_head_dtype.py`).

**`convert.py`.**
- The CLI takes `--bits {8,4}` and `--streaming`.
- The memory guard samples swap and the process footprint every 10 s. The footprint comes from `proc_pid_rusage`, because MLX's Metal buffers do not show in `ps` RSS or `ru_maxrss`: a 2 GiB MLX array moved RSS by 0.01 GiB.
- The record carries the source revision, `manifest_sha256`, the tensor policy read back from the headers, the flags, `head_policy`, `t_start` and `t_end`, peak RSS and footprint, and the guard summary. It is written with sorted keys and paths through `tools/redact.py`.
- The CLI copies the record to `results/convert/` after `runner.guard.require_identity()`.
- `refresh_port_file` also rewrites `manifest_sha256`.
- The manifest rule equals `tools/hash_tree.manifest_sha256`.

**`convert_streaming.py`.** It applies the steps mlx_lm takes to one group at a time: sanitize, the `--dtype` cast, then `module.to_quantized`.
- 8-bit and 4-bit: every tensor's name, dtype, shape and bits match the mlx_lm path, as do `config.json`, the model card and the index metadata.
- Only the shard layout differs: the streaming path writes `L + 2` shards.
- A source with F32 tensors converts identically too (`test_streaming_casts_like_mlx_lm`). That test was added after the mutation pass below showed the cast step is a no-op on an all-BF16 source.

## Entry 6: the reference deltas

1. **`force_ids`** on `moe_block`, `layer_branches` and `layer_forward`. Forcing a layer's own selection is bit-identical to not forcing it.
2. **Dumps.** The dump adds `r_attn`, `h_mid`, `r_moe` and `biased` per layer. `ref_record.json` adds `mode`, `reference_tree_sha256`, `gate_text_sha256` (from `--gate-text-sha`, defaulting to the ids file's sha256), `ids_sha256`, `t_start`, `t_end` and the dequantised build's manifest. `logprobs.npy` is written in row blocks.
3. **bf16 emulation** rounds where vLLM stores bf16 and threads the residual as `fused_add_rms_norm` does: it normalises the unrounded fp32 sum and stores it rounded (`ir/layernorm.py`:53–59). `test_emulation_rounds_exactly_where_it_says` checks every rounding point through the GEMM operands and the q/k/v that reach attention.

   **Against entry 1.** On entry 1's own setup (presets × seeds 0/1, 160 tokens, ids seed 100 + seed):

   | | Top-1, per case | Pooled top-1 | KL, per case |
   |---|---|---|---|
   | entry 1, re-run with its original diagnostic script | 0.969, 0.975, 0.887, 0.863 (the logged band end points) | 0.924 | 1.3e-3, 2.0e-4, 3.7e-3, 3.9e-3 |
   | this mode | 0.969, 0.981, 0.900, 0.856 | 0.927 | 8.8e-4, 2.7e-4, 2.7e-3, 3.5e-3 |

   - The two schedules differ only in what the norm reads, a difference of 1 bf16 ulp (spec item 5). Their flips land on different tokens, so single cases move by binomial noise (σ = 0.027 per case).
   - The test therefore checks the pooled top-1 within 3σ, the pooled KL within a factor of 2, and each case inside entry 1's band widened by 3σ.
   - An exact per-case reproduction would require emulating MLX rather than vLLM, and HYPOTHESIS defines the mode as "wherever vLLM stores bf16".
4. **Dequantised mode.** `mlx_affine_np.ConvertedCheckpoint` presents a `convert.py` output under the checkpoint names. Linear weights come back as fp32 `scale · q + bias` (what `quantized_matmul` uses) and the embedding rounded to bf16 (what `QuantizedEmbedding` returns).
   - Both match `mx.dequantize` on the GPU to 0 ulp, at 8, 4 and 2 bits and groups 32, 64 and 128, and on the five tiny builds.
   - mlx's CPU kernel rounds the product and the sum separately in bf16, which differs in about 3% of values. The experiment runs on the GPU.
   - **The dequantised reference against the quantised port, forced onto the reference's experts:** KL 8.4e-5–9.6e-5 in fp32 mode and 1.05e-4–1.22e-4 in emulation, with decisive top-1 1.000.
   - **Why emulation is not closer.** It rounds at the same places as the port, but independently of the port's accumulation order, so the two errors add. It calibrates the size of the bf16 error, not its value.
   - **G2q-style rows on tiny** (all five builds):

     | Check | Measured | Test bound |
     |---|---|---|
     | Head relative L2 per position | ≤ 3.3e-7 | 1e-5 (gate threshold 1e-3) |
     | Embedding | identical | — |
     | Per-layer branch medians | 5.4e-3–1.5e-2 | 3e-2 |

     The test bound for the branch medians is derived in the test's docstring.
5. **Long sequences.** Real hidden size, head geometry and vocabulary, 2 layers (1 sliding, 1 full) and 16 experts, on one 16,384-token sequence: peak RSS 13.4 GiB in fp32 and 13.8 GiB in emulation, of which 7.8 GiB are the logits. The peak is per layer, so 50 layers do not add to it. BUILD_SPEC asks for about 20 GB. This was a scratch measurement, not a test. `round_bf16` works in uint32 to keep its temporaries at the input's size.
6. **`forward_streams`** carries k variants through one pass. Each layer tensor is read once and cached read-only for the layer; the head is read once per stream. Every stream is bit-identical to that variant's own pass. The CLI's `--mutants all|names` writes `stream_<name>.nll.npy`.

**`mutants.py`.** It holds the seven G3 mutants, each overriding exactly one of the hooks `norm`, `rope`, `uses_rope`, `qk_norm_rope`, `route` and `layer_norm_names`. On both presets each moves the logits by 1.1–7.6 (max-abs). The hooks were factored out of `layer_forward` and `attention_block`. fp32 logits, hidden states, final norm and lengths are bit-identical to `exp036-ref-1` on both presets at seeds 0 and 1 (`np.array_equal`).

## Entry 7: mutation pass over the new code

There were 16 mutants: 6 port, 6 reference and 4 converter. Each was applied to a scratch copy of `port/`, `reference/` and the relevant tests and run with `-x`.

| Mutant | First killer |
|---|---|
| quantised head fed bf16 activations | `test_g2q_head_and_embedding_on_dequantised_weights` |
| router weight left bf16 | `verify_output` (tensor class `router_weight`) |
| head always upcast to fp32 | `test_policy_keys_drive_sanitize_cast_and_quant_predicates` |
| predicate defaults false | `test_convert_python_api_defaults_to_the_quantised_head` |
| `force_ids` ignored | `test_g2q_layer_branches_on_dequantised_weights` |
| `branches` returns the pre-norm attention branch | `test_g2q_layer_branches_on_dequantised_weights` |
| emulation: residual not rounded | `test_dump_in_bf16_emulation` |
| emulation: norm rounded once | `test_emulation_rounds_exactly_where_it_says` |
| dequantised embedding in fp32 | `test_dequantised_mode_reads_the_mx_dequantize_values` |
| dequantised linear weights in bf16 | `test_dequantised_mode_reads_the_mx_dequantize_values` |
| streams without the weight cache | `test_forward_streams_equal_separate_passes` |
| forced weights from biased scores | `test_g2q_layer_branches_on_dequantised_weights` |
| no policy keys in the staging config | `verify_output` |
| manifest lines without the final newline | `test_record_lists_every_file_with_sha256` |
| memory guard ignored | `test_convert_stops_when_the_guard_trips` |
| streaming skips the `--dtype` cast | survived, then `test_streaming_casts_like_mlx_lm` was added (killed) |

## Other test-side changes

- `tests/tiny_checkpoint.py` has a `w513` preset: the vendor preset with the real window and `max_position_embeddings` 8192. The vocabulary stays 1,024.
- In `conftest.py`, `EXP036_TOK` is the tokenizer variable BUILD_SPEC names; `EXP036_TOKENIZER_DIR` still works. `EXP036_REQUIRE_ALL=1` turns every skip, at collection or at run time, into a failure.
- `test_convert.py` covers five builds (8/4-bit g64 and 4-bit g32 with the quantised head; 8/4-bit g64 vendor-faithful), plus record, streaming, memory-guard and CLI tests.
  - The 4-bit smoke bound (top-1 ≥ 0.25 against the fp32 reference on the original weights) still holds with a 4-bit head: measured 0.32 (g64) and 0.39 (g32).
- `test_reference_cli.py` covers the new dump files and record fields, the emulation dump and `--mutants`.

**New files, named `test_port_ref_*` for this build round.** BUILD_SPEC §6 names them `test_head_dtype.py`, `test_cache_batching.py`, `test_reference_mutants.py`, `test_safetensors_np.py` and `test_reference_modes.py`; `test_port_ref_hooks.py` has no BUILD_SPEC counterpart.
- **`test_port_ref_cache_batching`** runs `BatchGenerator` directly:
  - batched against B = 1: ≤ 4.3e-6;
  - B = 1 against a one-shot forward: ≤ 4.8e-6;
  - no greedy divergence;
  - 4, 8 and 7 admissions while another sequence is past 2W, at `completion_batch_size` 8, 4 and 3;
  - the same at W = 65 and W = 513.

## Suite

- My area (the pre-existing files and `test_port_ref_*`), with `EXP036_TOKENIZER_DIR` set: `311 passed` in 37 s.
- The whole `tests/` directory, including the other areas' files written in parallel: `1126 passed, 2 skipped`, plus one failure in `test_gate_checks.py::test_greedy_rule`. That file belongs to the gate area and was being edited at the time; it passed when rerun alone.
