# W13: vendor-code spike (exp_037 DESIGN.md §10)

**Route A: vLLM 0.29.0's own CPU path, built from source on the mini, running the vendor plugin's `Kolibri1ForCausalLM` unmodified.** Route B (the stub transcription) was not needed and was not run.

- **Time box:** started 2026-10-06T04:07:20Z; all runs finished by 04:18:22Z; README written about 04:20Z. About 13 minutes of wall time against the 2 h box, downloads included.
- **Route A installed and imported in 3 min 13 s:** 04:07:20Z to 04:10:33Z, against the 30 min limit (`routeA/install.log`).

## What ran

1. **Downloads into `downloads/`.** No git was used: both are tarballs, addressed by commit and tag.
   - aleph-alpha-inference at commit `049a6a7bd2405b27d6d280d256bd3d585191c7ae` (`https://github.com/Aleph-Alpha/aleph-alpha-inference/archive/<sha>.tar.gz`). Tarball sha256 `c51e23dccd0ab598d951eab0696cd658b038fafa28d47f621eb9d05b713c5c61`.
   - vLLM v0.29.0 source (`https://github.com/vllm-project/vllm/archive/refs/tags/v0.29.0.tar.gz`). Tarball sha256 `bb7152b97726e98f8fe224ac255e9f882c6a92b84e2d0ead0e887e4110befca0`.
2. **Venv `venvA`.** Built with `uv venv --python 3.12` (Python 3.12.13). The uv cache is `uv-cache/`. Everything stays inside this folder: about 2.7 GB in total.
   - Wheels came from PyPI: torch 2.13.0 (the macOS arm64 wheel, a CPU build with Accelerate BLAS), cmake, ninja, setuptools-rust, setuptools-scm, and vLLM's `requirements/cpu.txt` (148 packages, among them transformers 5.18.0 and numpy 2.3.5).
   - vLLM was then built with `--no-build-isolation`, `VLLM_TARGET_DEVICE=cpu` and `SETUPTOOLS_SCM_PRETEND_VERSION=0.29.0`, giving `vllm==0.29.0+cpu` with `_C.abi3.so`. The optional Rust frontend was skipped, so no Rust toolchain was installed.
   - The vendor package was installed `--no-deps` from the tarball, unmodified: `aleph-alpha-inference==1.0.0`.
   - **Disclosure:** vLLM's dependency set and the build tools go beyond the three named downloads. They are what installing vLLM 0.29.0 requires.
3. **Inputs.** `scripts/build_inputs.py` builds them with exp_036's `tests/tiny_checkpoint.py`, read-only, under `~/models/exp036-mini/venv312` (mlx 0.31.2).
   - `vendor_s0` and `vendor_s1`: the tiny vendor preset (`vendor`, window 65, 4 sliding + 2 full layers, 8 experts, top-2) at seeds 0 and 1.
   - `w513_s0`: the same layout with window 513.
   - `rl_s23` and `rl_s29`: **scratch rebuilds of the §5.1 real-layout recipe.** They use bughunt `build_ck.py`'s overrides: preset w513, 10 layers with full attention at i % 5 = 4, 48 q / 4 kv heads, head_dim 128, E 64 top-6, hidden 256, q/k-norm × 3. Seeds 23 and 29, BF16.
   - Token ids: `random_ids(64, seed=101)` and `random_ids(600, seed=102)` (`ck/ids.json`).
   - **The exp_037 W0b checkpoints are pending.** `tasks/chronos/exp_037_kolibri_forced_gate` did not exist at 04:18Z.
4. **Reference** (`scripts/run_reference.py`): exp_036 `reference/kolibri_ref.py`, read-only, fp32, free run from position 0, no bytecode written.
   - Tree sha `85337ed7efeef6b0463f3560fff6da6496661a2fa7a14de5e50a1bce9e251bbd`, which is the pinned value.
   - Version `exp036-ref-2`.
5. **Vendor side** (`scripts/run_vllm_routeA.py`): `LLM(dtype="float32", enforce_eager=True, skip_tokenizer_init=True, enable_prefix_caching=False, max_num_batched_tokens=2048)`, in-process.
   - The plugin registers through vLLM's `general_plugins` entry point.
   - Each sequence is one prompt-only request, prefilled in exactly one forward (asserted).
   - Forward hooks only record:
     - each `Kolibri1DecoderLayer`'s `hidden_states + residual`, which is the residual stream after the layer;
     - the `model.norm` output;
     - the GateLinear router logits;
     - the embedding.
   - Logits are the vendor model's own `compute_logits`.
   - Backends in use: `CPUAttentionBackendImpl` (C++ `_C`), `CPUUnquantizedExperts` MoE, native RMSNorm and vLLM `RotaryEmbedding` (NeoX, base 10000, rotary_dim = head_dim) on sliding layers only. The impl windows were 65/513 on sliding layers and -1 on full layers.
   - Two runs per checkpoint were bitwise identical.
6. **Compare** (`scripts/compare.py`): relative L2 error ‖vendor − ref‖₂ / ‖ref‖₂ over each layer's [T, 256] output and over the logits [T, 1024]. This is the §10.5 statistic; a mismatch is any value > 1e-4. The script also reports the worst single position and max|Δ| / max|ref|.

## Results

| checkpoint | T | max layer rel L2 | logits rel L2 | worst position rel L2 | max\|Δ\|/max\|ref\| | argmax agree | §10.5 literal |
|---|---|---|---|---|---|---|---|
| vendor_s0 | 64 | 9.74e-07 | 1.04e-06 | 1.29e-06 | 1.23e-06 | 1.0000 | clean |
| vendor_s0 | 600 | 9.01e-07 | 9.77e-07 | 1.70e-06 | 1.27e-06 | 1.0000 | clean |
| vendor_s1 | 64 | 1.01e-06 | 1.08e-06 | 1.35e-06 | 1.19e-06 | 1.0000 | clean |
| vendor_s1 | 600 | 9.21e-07 | 1.00e-06 | 1.40e-06 | 1.08e-06 | 1.0000 | clean |
| w513_s0 | 64 | 9.74e-07 | 1.04e-06 | 1.29e-06 | 1.23e-06 | 1.0000 | clean |
| w513_s0 | 600 | 9.10e-07 | 9.88e-07 | 1.43e-06 | 1.06e-06 | 1.0000 | clean |
| rl_s23 (scratch §5.1 recipe) | 64 | 4.85e-05 | 4.89e-05 | 9.77e-05 | 7.33e-05 | 1.0000 | clean |
| rl_s23 (scratch §5.1 recipe) | 600 | 2.21e-02 | 2.21e-02 | 2.75e-01 | 2.00e-01 | 0.9917 | **> 1e-4** |
| rl_s29 (scratch §5.1 recipe) | 64 | 4.38e-05 | 4.40e-05 | 7.83e-05 | 6.76e-05 | 1.0000 | clean |
| rl_s29 (scratch §5.1 recipe) | 600 | 1.71e-04 | 1.72e-04 | 3.57e-04 | 2.68e-04 | 1.0000 | **> 1e-4** |

**Tiny vendor preset (the §10.4 input): clean.** The maximum relative L2 is 1.08e-6 over every layer and the logits, on both sequences and both seeds, with window 65 and window 513 crossed. The worst single position is 1.7e-6.

**Real-layout recipe (scratch rebuilds) at 600 tokens: the literal rule fires on the free-running pass.** Localisation (`scripts/localise.py`, `out/localise.json`) finds no wiring difference:

- **Teacher-forced per layer:** the reference's layer l, run on the vendor's own input h_{l-1}, matches the vendor's h_l to at most 1.94e-6 relative L2 (worst position 3.4e-6) at every layer of every checkpoint.
  - The expert sets are identical at every (layer, token) under both natural and forced routing.
  - Router logits agree to at most 1.7e-6.
  - The embedding and the loaded `expert_bias` are bitwise equal to the checkpoint.
  - The final norm agrees to 5e-8; the LM head is exact (0).
- **The free-run growth is fp32 rounding amplified by the sharpened tiny model.** The error grows about 1.5× per layer from layer 0 (1.9e-6).
  - In rl_s23/600 the one routing flip is at layer 7, token 244: vendor's 6th-vs-7th biased-score gap 5.6e-5, a near-tie. It cascades into the later layers.
  - **Chaos control:** the reference against itself, with h_0 perturbed by Gaussian noise of relative size 1.5e-6, diverges the same way or worse:
    - rl_s23/600: 2.7e-1 by layer 9 (logits 2.7e-1);
    - rl_s29/600: 1.3e-4 by layer 8 (logits 4.8e-3);
    - rl_s23/64: logits 6.0e-5; rl_s29/64: logits 5.6e-5;
    - vendor presets: about 1.9e-6.

**Discrimination control** (`scripts/discrimination.py`, `out/discrimination.json`): the test can see a wiring misreading.

- vendor_s0/600, vendor against each of exp_036's seven G3 reference mutants, as logits relative L2:

  | mutant | logits rel L2 |
  |---|---|
  | sigmoid_bias_select | 0.50 |
  | rope_on_full | 0.15 |
  | one_plus_w_norm | 0.98 |
  | renorm_topk | 0.28 |
  | swap_sandwich_norms | 0.38 |
  | rope_traditional | 0.79 |
  | qknorm_after_rope | 0.22 |

- Window off by one: W−1 gives 0.22 and W+1 gives 0.20.
- On rl_s29/64, every mutant is ≥ 0.71.
- The smallest of all is 1.5e3 × the bound.

## The §10.5 sentence (vendor preset)

> A tiny-checkpoint comparison of the vendor's own model code, through vLLM 0.29.0, against our reference agreed to 1.1e-6 at every layer; it excludes a misreading in the wiring it exercises, not one in vLLM's kernels, in FP8 serving, or in behaviour tiny random weights cannot show.

**Open for the main session or Andrei.**

- **How §10.5 applies to the real-layout checkpoints.** On those checkpoints, a free-running 600-token pass exceeds 1e-4 for any fp32 implementation, including the reference against itself. Layer by layer from the same input, the vendor agrees to 1.9e-6.
- **This spike changed nothing.** There was no reference change and no port change.
- **The formal run on the exp_037 W0b seed-23/29 builds is still to do.** Their shard sha256 can be checked against the scratch rebuilds below.

## Inputs (sha256)

```
313312e5ab3ef60824a93779d92aaf77ada9e22ad46591983383ca4c24d6c3ae  vendor_s0/model-00001-of-00002.safetensors
be8dc93b425476a533e2f25ad0ae0f0e34ac7b5da7fa28fec0775cc728c73569  vendor_s0/model-00002-of-00002.safetensors
49eaeadc517391f05ae3b905a38370a14636e84962604c670e14a596f3ee45fe  vendor_s0/config.json
66eb56bcddecb7dc9e16336df557b87aab66e39e8b8c3c335bd94265acef9c39  vendor_s1/model-00001-of-00002.safetensors
43a425583e60e4c419db89f977d1a0415baf5a86d5cda177d7490c454280ef5c  vendor_s1/model-00002-of-00002.safetensors
49eaeadc517391f05ae3b905a38370a14636e84962604c670e14a596f3ee45fe  vendor_s1/config.json
313312e5ab3ef60824a93779d92aaf77ada9e22ad46591983383ca4c24d6c3ae  w513_s0/model-00001-of-00002.safetensors   (same weights as vendor_s0)
be8dc93b425476a533e2f25ad0ae0f0e34ac7b5da7fa28fec0775cc728c73569  w513_s0/model-00002-of-00002.safetensors
55985963b5ddde8b12d5c6052776e0a61b448528ff049b051f67aa2068da895e  w513_s0/config.json
52b676b4b53ae081d3aa3ef86144ed4b97341f1c024a1c733279cebb866412a3  rl_s23/model-00001-of-00002.safetensors
ed2f74e5adeb4e89103bc2b1ac060870614bff3a620c93eeef1443f10f36299c  rl_s23/model-00002-of-00002.safetensors
3ee351c7403329386518e3f0ad36f1b73c4597052190829c5bf281363622e7f3  rl_s23/config.json
057123e65808633602ae9359f4b600f688d719fd8c6315fd265e83b9ed4e549f  rl_s29/model-00001-of-00002.safetensors
be34b2e072cc099d0cf6a4d6089822d06b47848e46a466518f41d33a6e4a01c1  rl_s29/model-00002-of-00002.safetensors
3ee351c7403329386518e3f0ad36f1b73c4597052190829c5bf281363622e7f3  rl_s29/config.json
24696faf87907b71e6d5420e5f9a0b45bd45720e9322dc4c235b02c58a002f34  ids.json
```

Vendor and vLLM source files exercised (sha256):

```
f93635dd3ebeed3c12ca2b3a00ab1b92fdc95b5290eaa672420861ba5a1bd320  aleph_alpha_inference/kolibri1.py (installed copy identical)
38e8ef7f742da2d0722a796667d8cd0795de779409eca24e0a9ca4f4c70358b1  aleph_alpha_inference/__init__.py
8770a22e4615ce46708146ea605ab37965c3c064c6905f9e6a6a5dd2c526d1ec  aleph_alpha_inference/config.py
3a0dae7a1b0140bcbf1f7903135c72e5e44a8174c7fade026b77659c8377d7ee  vllm/model_executor/models/qwen3_moe.py (installed copy identical)
65d33dcb96404ddde273acf84ef901151a8155a2cffc144bdd0c49fe1d576a22  vllm/ir/ops/layernorm.py
c81804498f08f072356a26b41967f211475df2694dabf5da9f488c074d1fdef5  vllm/model_executor/layers/rotary_embedding/base.py
76fb02a6bd725125565d84bf04f627cf16da5f7ff8f31f87c01ed6a1439398c7  vllm/v1/attention/backends/cpu_attn.py
8a0396676391ca0faa67ca68af49a775271719ea3f2c8bf1a605cf5955d52716  vllm/model_executor/layers/fused_moe/router/gate_linear.py
```

## Files

- `scripts/`:
  - `routeA_install.sh` and `routeA_watchdog.sh`: the install, with a 30-min deadline;
  - `build_inputs.py`;
  - `run_reference.py`;
  - `run_vllm_routeA.py`;
  - `compare.py`;
  - `localise.py`;
  - `discrimination.py`.
- `out/`:
  - `ref_*.npz`, `vllmA_*.npz`, and `vllmAd_*.npz` (the vLLM run with router/embedding capture);
  - `compare_vllmA.json`, `localise.json`, `discrimination.json`, `inputs_sha256.txt`.
- `routeA/`: the install and run logs.
- `ck/`: the checkpoints. `vendor_s0_w64` and `vendor_s0_w66` are config-only copies used by the window control.
- **Not for the repo:** `downloads/`, `venvA/` and `uv-cache/`, as §10.6 requires.
- **For the main session:** per §10.6, the scripts, this README and the `out/*.json` files go into E37 `diagnostics/vendor_spike/`. This task did not write into E37.

## Copied into exp_037 (W12, 2026-10-06)

- **What was copied** from the spike's scratch folder: this README, `scripts/` and the four small outputs `out/compare_vllmA.json`, `out/localise.json`, `out/discrimination.json` and `out/inputs_sha256.txt` (byte for byte). `out/arrays_sha256.txt` (new) lists the sha256 and size of the 30 `.npz` arrays of the scratch `out/` that those JSON files were computed from; the arrays themselves are not copied (the kit's `.gitignore` excludes `*.npz`, and the leak check refuses them). Not copied, as decision E requires: the downloaded vendor and vLLM sources (`downloads/`), the venv (`venvA/`), the uv cache, the install and run logs (`routeA/`) and the checkpoints (`ck/`).
- **What changed in the copy** (no host name, user name or home path in a kit file): line 3 of this README named the mini by its host name and now says "the mini"; 4 Python scripts held the absolute path of exp_036's directory and now derive the same directory from their own location (`Path(__file__).resolve().parents[4] / "exp_036_kolibri_local_eval"`); 2 shell scripts held the absolute scratch path and now read it from `$SPIKE_DIR`. Each changed line carries the comment "exp_037 copy". Nothing else differs from the scratch originals, whose sha256 are listed below.
- **The real-layout inputs are the kit's builds.** `rl_s23` and `rl_s29`, the scratch rebuilds of the real-layout recipe (q/k norms × 3), have the same shard and config sha256 as this kit's `tests/tiny_real_layout.write_real_layout(out, 23)` and `(out, 29)`: `52b676b4…` / `ed2f74e5…` and `057123e6…` / `be34b2e0…`, config `3ee351c7…` (checked again on 2026-10-06 by rebuilding both). The spike therefore ran on weights byte-identical to the kit's sharpened seed-23 and seed-29 builds; the unsharpened seed-29 build of decision (a) was not part of it. This sha256 identity replaces the "formal run on the exp_037 W0b seed-23/29 builds" listed as open above.
- **Classification.** Andrei's E-verdict (2026-10-06T05:17:51Z, "ok, use your recommendatino", to the recommendation "record as no wiring mismatch, with the disclosure"): no wiring mismatch. The literal end-to-end exceedance of the 1e-4 rule on the two real-layout 600-token runs, its cause (routing chaos, shown by the reference-against-itself control) and the fact that the reading was settled after the result are disclosed in HYPOTHESIS.md ("Disclosures", "The vendor-code spike"). The sentence of the section "The §10.5 sentence" is appended to HYPOTHESIS.md's blind-spot paragraph.
- **Status.** A pre-freeze diagnostic, not an exp_037 result. `diagnostics/` is outside every hash scope.

Scratch originals (sha256):

```
ea79f47b3f9e63bcd9e4cadbddb0dc3d9b63003f4ca47c4c8948260f4ee0fbbc  README.md
f6700c4681f35f2021b29bf691a9340f496f95ecd7ed390b219e223409aa3946  out/compare_vllmA.json
edcb94baebea7c914e4477b22b52a4fd47e492c81308495ee5af5d5a704ead9c  out/discrimination.json
38f5f09a2d36350eb773bcffb0ecade1e418a7ba395fa5b8b940e3d172744c12  out/inputs_sha256.txt
e7eb98f3b9160ca940097023fe13d7b219a10ff9eaf2c0211a678c96d29fe6e8  out/localise.json
c679ea80505873912d88a419b4999c72da0e8e0268d73c6b78d1c36f9cbf20bd  scripts/build_inputs.py
224d120925505d5d2757e3b4e5a6fb164617be014bb196196ea864f3ea4d9629  scripts/compare.py
c77a6d3407db0f26a23b5ae7d0c206036b98100e3ab3ed5abbf438206dc5a6ff  scripts/discrimination.py
60237af9fc2d3862739a9ed85a65dddaf6eff85a327e87bec3af556f64c3470b  scripts/localise.py
ec167d89e591181e453dc23df242cb1ba03e91d06de4a8197b2ec1a616c2285c  scripts/routeA_install.sh
28f29b79ceafe94b73006144d07b1cb3e877b9b0538b0550966c7f868865f35a  scripts/routeA_watchdog.sh
16ba48451498d4be7d7a6e6a55aba75d25ff70dc131336e27513ec214fdd9d22  scripts/run_reference.py
7c6c4ef2b3d4934e1569294a59ef4033bfb740ac861161552db88e57d0a9bb28  scripts/run_vllm_routeA.py
```
