# Experiment 036 — build specification for the kit

*For the build agents on miktam-mini. It specifies what the kit consists of, file by file. [`HYPOTHESIS.md`](./HYPOTHESIS.md) is the pre-registration; where the two disagree, HYPOTHESIS.md wins and this file is fixed. [`RUNBOOK.md`](./RUNBOOK.md) is how Andrei and the mbp session run it. Precedence: `scientific_log.md` > HYPOTHESIS.md > BUILD_SPEC.md > RUNBOOK.md.*

**Hosts**

| Host | Role |
|---|---|
| miktam-mini | Builds the kit. Runs only weight-free unit tests on tiny random checkpoints, and later the analysis. Persistent environments (never in a session scratchpad): `~/models/exp036-mini/venv312` (Python 3.12, `pip install -r env/requirements-mbp.txt`, mirrors the mbp) and `~/models/exp036-mini/venv314` (`python3 -m venv --system-site-packages` over Homebrew Python 3.14 with its mlx 0.31.2 / mlx-lm 0.31.3, plus pytest and pyarrow). Build-time data under `~/models/exp036-mini/` (§8). |
| mbp | Runs everything that touches real weights, from `$EXP036_MODELS/venv` (Python 3.12). |

**Hand-off:** through GitHub, `miktam/local-first-ai`.

**Already present (untracked) in the directory. These files are binding: keep them, extend them, never weaken their calibrated bounds.**
- `port/kolibri1.py`, `port/convert.py`, `port/README.md`
- `reference/{__init__,kolibri_ref,safetensors_np}.py`, `reference/README.md`
- `tests/{conftest,exp036_helpers,tiny_checkpoint,port_harness}.py`
- `tests/{test_routing,test_reference_invariants,test_chat_template,test_tiny_checkpoint,test_port_vs_ref,test_cache_decode,test_convert,test_mutation_guards}.py`
- `tests/INTEGRATION_LOG.md` (the calibrated (b1) forced-routing and (b2) natural-routing bounds and their derivations)
- `tests/fixtures/README.md`, `tests/fixtures/kolibri1_chat_template.{vendor,tokenizer_config}.jinja`
- `NOTICE`, `LICENSE-APACHE-2.0`
- `host/mbp_preflight_*.json` (pushed; the mbp's measured Metal limit)

The required deltas are listed in §5.1 and §5.1b.

**Build tiers.**
- **Tier 1 blocks the pre-registration push:** port deltas; reference deltas and `reference/mutants.py`; gate G0–G5; runner; tasks (loaders, renderers, manifests builder, vendored eval-framework); scorers incl. the IFBench adapter and its test; tools (preflight, leak_check, hash_tree, peer_check, version_record, status, kv_bytes); bench H1, D1, H5, H8, C1; `analysis/stats.py`, the H1–H8 and D1 verdict code, `margins.json`, `vendor_values.json`, `power.py`.
- **Tier 2** (exploratory analyses E1–E12 in `analysis/exploratory.py`, `analysis/tables.py`, `bench/ladder.py`) is frozen by its own numbered amendment ("Tier-2 analysis"), pushed from the mini as `amendments/<k>_tier2_<UTC>.md` plus the code (the mini never edits HYPOTHESIS.md during the run; the mbp appends the file), before `scorers/score_all.py` first runs and before any B4 cell starts. The amendment lists the sha256 of every Tier-2 file. It counts as at HEAD once either HYPOTHESIS.md or the committed `amendments/` file holds it (review fix 2026-10-03). `score_all.py` refuses otherwise; a session reaching the B4 cell without it skips that cell and runs the rest of the queue (an unstarted B4 is NOT RUN at the end of S3).

---

## 1. Directory tree

```
exp_036_kolibri_local_eval/
├── HYPOTHESIS.md  BUILD_SPEC.md  RUNBOOK.md  README.md (short index + links)
├── ASSETS.md  assets.json                     (existing; the main session appends New assets before the push)
├── NOTICE                                     (every Apache-derived or copied file; IFBench_test ODC-BY; FineWeb-2 ODC-By; Grundgesetz note)
├── LICENSE-APACHE-2.0                         (existing; verbatim from the plugin's LICENSE)
├── .gitignore                                 (__pycache__/, .pytest_cache/, *.npy, *.npz, *.safetensors, *.parquet, *.pdf, work/, private/, env/exp036.local.env)
├── host/                                      (existing mbp preflight records)
├── env/
│   ├── requirements-mbp.txt                   exact pins (§3)
│   ├── versions.json                          the same pins, machine-readable, for preflight
│   ├── setup.sh                               creates or syncs $EXP036_MODELS/venv from the pins (uv; RUNBOOK step 3)
│   └── exp036.env                             EXP036_* defaults, LFA, EXP, PY, HF_HUB_OFFLINE; sourced by every RUNBOOK block
├── port/
│   ├── kolibri1.py                            MLX model file (model_type kolibri1), loaded through config.json "model_file"
│   ├── convert.py                             BF16 → MLX affine g64 8/4-bit, explicit quant predicate, self-monitored memory
│   ├── convert_streaming.py                   fallback converter, layer by layer, bit-identical on tiny
│   └── README.md                              (existing; updated to the new defaults and output names)
├── reference/
│   ├── __init__.py  safetensors_np.py  kolibri_ref.py  README.md
│   ├── mlx_affine_np.py                       own numpy unpacker for MLX affine packing (G2q)
│   ├── mutants.py                             7 semantic mutants of the reference (G3); subclasses only
│   └── DERIVATION.md                          each reference block → vendor kolibri1.py line / vLLM v0.29.0 file:line
├── gate/
│   ├── run_gate.py                            phase driver; writes results/gate/gate_<UTC>.json; exit 0 / 4 / 1 / 3
│   ├── common.py  harness.py  textset.py  vendor_template.py   shared helpers (paths, port loading, gate texts, vendor jinja)
│   ├── template_cases.json                    the 12 conversations × 10 settings of the G0 template parity
│   ├── thresholds.json                        frozen (§7.1)
│   ├── checks/{g0_static,g1_synthetic,g2_layers,g3_oracle,g4_e2e,g5_generation,ref_pass}.py
│   ├── port_mutants.py                        the 15 port mutants (§5.3)
│   ├── build_gate_text.py                     builds and verifies gate/texts; T5, T6, T9 by rule into $EXP036_WORK
│   ├── tokenizer_lines.py                     committed rule for the G0 tokenizer-parity lines
│   ├── texts/T{1,2,3,4}_*.ids.json + .txt  T7_chat_en_high.json  T8_chat_de_none.json  G5_prompts.json  MANIFEST.json  src/
│   └── behaviour_prompts.json                 20 frozen short factual / one-step prompts (10 EN, 10 DE)
├── runner/
│   ├── run.py                                 CLI: pilot | session | cell | status
│   ├── generate.py                            BatchGenerator wrapper; per-step log; heartbeat
│   ├── sampler.py                             batched vLLM-order sampler (item 26)
│   ├── chat.py                                per-family chat kwargs, templates and EOS ids
│   ├── common.py                              directories, arms, hashing, UTC stamps, the header environment record
│   ├── templates/{gemma4,qwen3_6,qwen3_8}.jinja   committed peer templates (upstream if they differ from mlx-community); MANIFEST.json, SOURCE.md
│   ├── memory.py                              memory rule, B choice, Metal limit
│   ├── jsonl.py                               append-only, fsync, header, resume, torn-line handling
│   ├── guard.py                               refusal checks
│   ├── seeds.py                               cell seeds
│   ├── simulate.py                            continuous-batching projection (plan rule 4)
│   ├── plan_fix.py                            pilot → plan_fixed_<UTC>.json + AMENDMENT_<k>_<UTC>.md
│   └── plan_rules.json                        frozen (§7.2)
├── tasks/
│   ├── assets.py  common.py
│   ├── gpqa.py  mmlu_prox.py  aime.py  ifbench.py  rgb.py  fineweb2.py
│   ├── build_manifests.py                     manifests (ids + sha256 [+ category, public gold]) from local data; text → $EXP036_PRIVATE
│   ├── selection_rules.json                   seeds, pilot pools, ladder, RGB seeds: frozen
│   ├── prompts/aime_de.txt  prompts/rgb_forced_en.txt  prompts/mmlu_prox_de_note.md
│   ├── vendored_evalfw/                       eval-framework v0.14.2 files verbatim at their upstream paths under src/eval_framework/ (gpqa.py patched, §5.5) + LICENSE, MANIFEST.json, SOURCE.md, shim.py, vendor.py
│   └── manifests/                             built on the mbp in S1 (never by hand)
├── scorers/
│   ├── reasoning.py  mc.py  aime.py  rgb.py  lang_tag.py  abstain_lexicon.json
│   ├── golden/*_golden.json                   the scorers' golden fixtures (synthetic text only; hashed with scorers/)
│   ├── ifbench_adapter.py  ifbench_driver.py  calls the IFBench checkers from data/IFBench-src inside the IFBench venv
│   └── score_all.py                           raw JSONL → scores/*.jsonl
├── bench/
│   ├── common.py  genutil.py                  bench helpers (files, guards, cool-down; model loading, timed generation)
│   ├── fit.py (D1)  speed.py (H1 + descriptive)  tokenizer_ratio.py (H5)  kl_8v4.py (H8)  batch_flip.py (C1)
│   ├── ladder.py (E6; Tier 2)
│   └── run_bench.py
├── analysis/
│   ├── stats.py                               bootstrap, t-test, Holm (two families), post-stratification
│   ├── verdicts.py                            H1–H8, D1 (Tier 1) + E1–E12 (Tier 2) → results/verdicts_<UTC>.json
│   ├── margins.json                           every decision threshold of H1–H8 and D1
│   ├── vendor_values.json                     vendor rows: value, page, N_v, n_v; peer values; MMLU-Pro category counts
│   ├── power.py                               power at the fixed n (plan amendment) and the HYPOTHESIS tables
│   ├── exploratory.py                         E1–E12 (Tier 2)
│   └── tables.py                              key-numbers table and plain-answer sentences (Tier 2)
├── tools/
│   ├── preflight.py  version_record.py  leak_check.py  hash_tree.py  peer_check.py  status.py  kv_bytes.py  redact.py
│   ├── common.py  assetcheck.py  params.py  shingles.py   helpers (identity, git state, asset revisions, parameter counts, shingles)
│   ├── kolibri_lfs_oids.json  peer_parity_build.json   the BF16 shards' LFS oids; the build-time peer parity record
│   ├── dry_run.py                             the whole pipeline on tiny checkpoints in a temporary copy (RUNBOOK step 3b)
│   ├── hooks/pre-push                         runs leak_check over @{u}..HEAD plus the index
│   └── withheld_shingles.sha256               built on the mbp at RUNBOOK step 7 and committed there
├── tests/                                     pytest; weight-free; must pass on the mini before push (§6); includes the IFBench
│                                              adapter test (test_scorers_ifbench.py), which starts the IFBench venv itself
├── amendments/ .gitkeep                       the mini's amendment inbox during the run (RUNBOOK: HYPOTHESIS.md has one writer)
├── results/  .gitkeep + README.md (layout)
├── evidence/ .gitkeep
└── aborted/  .gitkeep
```

**Environment variables** (no absolute paths anywhere in the kit; `env/exp036.env` holds the defaults):
- `EXP036_MODELS` (default `~/models/exp036`).
- `EXP036_DATA` (default `$EXP036_MODELS/data`).
- `EXP036_PRIVATE` (default `$EXP036_MODELS/private`): withheld raw text, text manifests and private torn lines.
- `EXP036_WORK` (default `$EXP036_MODELS/work`): reference dumps, gate texts T5/T6/T9, K8 log-probs for H8, conversion staging.
- `EXP036_TOK` (tests and the gate texts): a directory with the Kolibri `tokenizer.json` and `tokenizer_config.json`. On the mini, `~/models/exp036-mini/kolibri/Kolibri-1-BF16`; on the mbp, the BF16 snapshot (the tests fall back to `$EXP036_MODELS/Kolibri-1-BF16`). `EXP036_TOKENIZER_DIR` is an older alias, still accepted.
- `EXP036_REQUIRE_ALL=1`: conftest turns every skip into a failure (the pre-push run on the mini). `EXP036_REQUIRE_ALL=run-host` (the mbp at RUNBOOK step 3, and the gate's G1) does the same except for a skip whose reason starts with `build-host only:`: the tests that compare with files only the mini holds (upstream peer files, the eval-framework checkout).
- Test-only asset roots: `EXP036_MINI_ASSETS` (or `EXP036_MINI`; default `~/models/exp036-mini`), `EXP036_PEER_TOK_ROOT`, `EXP036_UPSTREAM_DIR`, `EXP036_EVALFW_SRC`. IFBench scoring: `EXP036_IFBENCH_PY` (default `<models>/ifbench-venv/bin/python`) and `EXP036_NLTK_DATA` (default `<models>/nltk_data`).
- `EXP036_PRIVATE_NAMES` (never committed): extra strings the leak check refuses (surnames).

Every entry point sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` before importing mlx_lm or transformers, defaults `PY` to `sys.executable` and the repo root to `git rev-parse --show-toplevel`.

---

## 2. Invariants (whole kit)

**Deterministic glue** (library `patterns/deterministic-glue-pipeline.md`):
1. Models sit only at leaves: one item per sequence. No model output is input to another model. Every score is code.
2. Joins are code. Item ↔ gold matching, aggregation, comparisons and verdicts are Python, never a prompt.
3. Every stage writes its state to disk: manifests, raw JSONL including reasoning and completion ids, scores, summaries, verdicts, version record. A failure names the broken stage, and partial outputs move to `aborted/` with a `NOTE.md` (private text never; see §5.4).
4. Context budget. Prompts are short by design. The long reasoning caps (16k–98k) and the context ladder (E6) are declared exceptions in HYPOTHESIS C7 and E6.
5. Output contracts are strict. Reasoning stripping and answer extraction are deterministic. A failure is scored as a failure with its reason and is **never** re-asked. Its rate is reported.
6. The topology is fixed and tested. Prompts, scorers and verdict code are frozen and hashed in HYPOTHESIS.md before any scored run (every tree in its hash table), with unit tests on hand-made fixtures. Any change is a numbered amendment.
7. There is no cloud and no network in kit code. Downloads are the explicit commands in ASSETS.md and RUNBOOK.md, and the build-time fetches of §8.

**Kit rules:**
- **Rights.** No withheld item text (GPQA, AIME-DE, RGB), no withheld gold, no withheld raw output, no FineWeb web text and no RGB text (instructions included) ever enters the repo or the test fixtures. Withheld sets write `text_sha256` to the repo JSONL and the full text to `$EXP036_PRIVATE`. Fixtures are synthetic, with stub instruction strings.
- **Privacy.** Records carry a fixed host label (`"mbp (M5 Max, 128 GB)"`), never a hostname, computer name, serial number or hardware UUID. Every path goes through `tools/redact.py:redact_path()`, which rewrites `$EXP036_MODELS`, `$EXP036_DATA`, `$EXP036_PRIVATE`, `$EXP036_WORK` and `$HOME`.
- **Determinism.** Every random choice uses a named seed from `tasks/selection_rules.json`, `runner/seeds.py` or `analysis/stats.py`. Every file the kit writes is UTF-8 JSON or JSONL with sorted keys (headers excepted), and every timestamp is UTC from the clock.
- **Append-only.** No kit tool opens an existing results, evidence or amendment file for writing except `jsonl.Writer` (append) and `tools/status.py` (the Status line of HYPOTHESIS.md only).
- **Identity.** Before writing anything under `results/`, every entry point calls `runner.guard.require_identity()`, which demands git `user.name == "Miktam"` and `user.email == "hello@localfirstai.eu"` as git resolves them in the repo.
- **No sudo.** The kit never calls sudo. It may *print* `sudo sysctl …` for Andrei.
- **Fixtures by real file name.** A missing fixture is a hard failure (the exp_011 lesson).
- **Machine-time records.** Every S1 block writes `t_start` and `t_end` (UTC) into its record; `plan_fix` sums them for S1 hours.

---

## 3. Environment

`env/requirements-mbp.txt` pins exact versions; the first block matches the mbp's preflight of 2026-10-03 (`host/mbp_preflight_20261003T170152Z.json`). The build agent adds `pyarrow` and `pytest` to `venv312` and pins the versions it resolves:

```
mlx==0.31.2
mlx-metal==0.31.2
mlx-lm==0.31.3
numpy==2.5.3
safetensors==0.8.0
tokenizers==0.23.2
transformers==5.18.0
jinja2==3.1.6
pyyaml==6.0.3
pyarrow==25.0.1
pytest==9.1.1
```

`tools/preflight.py` compares the installed distributions of `$EXP036_MODELS/venv` (read with `importlib.metadata`: the uv-built venv has no pip) against `env/versions.json`:
- a mismatch on mlx, mlx-metal, mlx-lm, numpy, safetensors, tokenizers or transformers → exit 1;
- any other extra package → recorded.

There is no torch anywhere. IFBench checker dependencies live only in `~/models/exp036-mini/ifbench-venv` on the mini, created at build time (§8), because the adapter's test must pass before the pre-registration push.

---

## 4. Forward-pass specification (authoritative for port and reference)

Items 1–25 are the arch-port reader's numbered spec (`reader_claims-arch-port.json` → `port_relevant_details`). Code comments cite them as "item N", so the numbering is kept. **[CORRECTED]** marks changes made after review. Items 26–29 are new.

1. **Sources and pinning.** Vendor plugin `aleph_alpha_inference/kolibri1.py` (329 lines, Apache-2.0) at v1.0.0 `049a6a7bd2`. Semantics are resolved against vLLM tag v0.29.0 (the plugin pins `vllm>=0.29,<0.30`): `qwen3_moe.py`, `layernorm.py`, `ir/ops/layernorm.py`, the rotary embedding `__init__.py` and `base.py`, `attention.py`, `flash_attn.py`, `triton_attn.py`, `flex_attention.py`, the fused_moe layer, router and runner, `gate_linear.py`, `logits_processor.py`, `config/model.py`, `arg_utils.py`, `cache.py`. vLLM main agrees on every point used here.
2. **Config.**

   | Field | Value |
   |---|---|
   | hidden size | 2560 |
   | layers | 50 |
   | query heads / kv heads / head_dim | 48 / 4 / 128 (q dim 6144, kv dim 512) |
   | vocab | 128000 |
   | rms_norm_eps | 1e-6 |
   | rope_theta | 10000 (default type) |
   | max_position_embeddings | 262144 |
   | use_sliding_window / sliding_window | true / **513** |
   | layer_types | `full_attention` at indices {4, 9, …, 49} (i % 5 == 4); all others `sliding_attention` |
   | experts | num_experts 384, num_experts_per_tok 6, moe_intermediate_size 512 |
   | shared expert | shared_expert_intermediate_size 512 |
   | other flags | norm_topk_prob false, hidden_act silu, attention_bias false, tie_word_embeddings false |
   | dtypes | head_dtype `"float32"`; `dtype` `"bfloat16"` (the key is `dtype`, not `torch_dtype`) |
   | special ids | bos None, eos 127906, pad 127901 |

3. **Checkpoint tensors** (BF16 repo, 32 shards, all BF16):
   - `model.embed_tokens.weight` [128000, 2560]; `lm_head.weight` [128000, 2560]; `model.norm.weight` [2560].
   - Per layer N, attention: `self_attn.{q_proj [6144,2560], k_proj [512,2560], v_proj [512,2560], o_proj [2560,6144]}.weight`; `self_attn.{q_norm,k_norm}.weight` [128].
   - Per layer N, norms: `{input_layernorm, post_attn_norm, post_attention_layernorm, post_ffn_norm}.weight` [2560].
   - Per layer N, MoE: `mlp.gate.weight` [384, 2560] (router); `moe.router.expert_bias` [384] (BF16); `mlp.experts.E.{gate_proj [512,2560], up_proj [512,2560], down_proj [2560,512]}.weight` for E = 0…383; `mlp.shared_experts.{gate_proj,up_proj,down_proj}.weight` (same shapes).
   - Count: 50 × 1167 + 3 = **58,353** tensors, **78,103,074,560** parameters. There is no attention output gate tensor.
4. **Embedding.** `h = embed_tokens[ids]`, with no scaling. Do not copy afmoe's muP √d scaling or Gemma's normaliser.
5. **Decoder layer order.** The plain-loop equivalent of the vLLM residual threading:

   ```
   a = Attn(RMSNorm_input_layernorm(h))
   a = RMSNorm_post_attn_norm(a)          # r_attn
   h = h + a                              # h_mid
   m = MoE(RMSNorm_post_attention_layernorm(h))
   m = RMSNorm_post_ffn_norm(m)           # r_moe
   h = h + m
   ```

   After the last layer, `h = RMSNorm_model.norm(h)`.
   - **Naming trap:** HF `post_attention_layernorm` is the **pre-MoE** norm, `post_attn_norm` is the sandwich norm on the attention output, and `post_ffn_norm` is the sandwich norm on the MoE output. Mirror the HF names in the MLX module; do not rename to afmoe's names.
   - Precision: vLLM keeps the residual in bf16. Its fused add + norm (`ir/ops/layernorm.py`) adds in fp32, stores the residual rounded to bf16 and normalises the unrounded sum. MLX bf16 add-then-norm differs by ≤ 1 bf16 ulp at the norm input. This is acceptable and is noted in the methods.
6. **RMSNorm** = `y = (x32 · rsqrt(mean(x32²) + 1e-6)).to(bf16) · w_bf16`, i.e. **w·x**, not (1+w)·x. `mx.fast.rms_norm` is correct. The same formula applies to q_norm and k_norm (dim 128), the four layer norms and the final norm (dim 2560).
7. **Attention.**
   - `q,k,v = x @ [Wq;Wk;Wv]ᵀ`, with no bias. View q as [..., 48, 128] and k as [..., 4, 128].
   - `q = q_norm(q)` and `k = k_norm(k)` per head, in **every** layer, **before** RoPE. v is not normalised.
   - RoPE is applied only in sliding layers. Scale is 128^-0.5 = 0.08838834764831845 in all layers.
   - GQA 12:1: query head h uses kv head h // 12.
   - `o_proj` 6144 → 2560, no bias.
   - There is **no** attention output gate, logit softcap, sinks, ALiBi or attention bias.
8. **RoPE.** NeoX half-rotation:
   - `rotate(x) = cat(−x[..., 64:], x[..., :64])`; `x·cos + rotate(x)·sin`.
   - rotary_dim 128; `inv_freq = 1 / 10000^(arange(0, 128, 2) / 128)`.
   - Positions are absolute 0-based token indices, not relative to the window. Keys are cached after RoPE.
   - In MLX: `nn.RoPE(128, traditional=False, base=10000.0)` with `offset=cache.offset`. `RotatingKVCache.offset` grows without bound, so positions stay correct past the window.
   - vLLM computes the cos/sin cache in fp32 and stores it in the model dtype (bf16) unless flashinfer is used (`rope_base.py:59–61`). The reference keeps fp32 cos/sin: a precision deviation in the reference's favour, not a semantic one. `DERIVATION.md` and `reference/README.md` state it so.
9. **Sliding window, exact.** vLLM FlashAttention uses window (513 − 1, 0) = (512, 0). The query at position i attends keys j ∈ [i − 512, i]: 513 keys including itself. Full layers use a pure causal mask with **no** positional encoding (NoPE), no window and no chunking. `cache_config.sliding_window` is None because not all layers are sliding.
10. **MLX window mapping** (mlx_lm 0.31.3 `models/base.py`, `models/cache.py`):
    - `create_causal_mask(N, offset, window_size=513)` keeps keys in (q − 513, q]. Use **513**, the config value verbatim.
    - Sliding layers use `RotatingKVCache(max_size=513, keep=0)`; keep must be 0 because there are no sinks. Full layers use `KVCache()` with a `"causal"` mask.
    - Build two masks per forward: the swa mask from the first sliding layer's cache, with window 513, and the fa mask from layer 4's cache.
    - At decode, once the cache is full it returns exactly 513 keys and the mask is None.
11. **Router.** **[CORRECTED: binding]**
    - logits = fp32(x) @ W_gateᵀ, where x is the bf16 output of `post_attention_layernorm` and **W_gate is stored as an exact fp32 upcast of the bf16 checkpoint** (sanitize upcasts; `cast_predicate` excludes `mlp.gate.weight`). `Router.__call__` casts only x. No per-call weight cast (it cost ≈ 0.5 GB of memory traffic per token).
    - The logits are never rounded to bf16. vLLM's GateLinear has `out_dtype=float32` and `quant_config=None` (`gate_linear.py`: `ll_bf16_gemm` or `torch.mm(out_dtype=fp32)`).
    - In MLX the router is a custom module with no `to_quantized`, so no predicate or recipe can quantise it.
    - Bias magnitudes reach ≈ 20, which makes top-k sensitive to logit rounding.
12. **Routing function.**
    - `ids = topk(logits + expert_bias, k=6)`; `w = sigmoid(logits[ids])`.
    - No renormalisation; routed scale 1.0; no grouped top-k.
    - `expert_bias` comes from checkpoint key `model.layers.N.moe.router.expert_bias` (BF16) and is upcast exactly to fp32.
    - **Trap:** afmoe and deepseek_v3 select on sigmoid(logits) + bias, which is wrong here. The order within the 6 selected experts is irrelevant.
13. **Experts.**
    - Each routed expert computes `y_e = down(silu(gate(x)) * up(x))`. The routed output is Σ_{e ∈ top6} w_e · y_e, with fp32 weights.
    - The shared expert is the same SwiGLU (2560 → 512 → 2560), **ungated and unscaled**: `out = routed + shared(x)`.
    - All 50 layers are MoE. In MLX: `SwitchGLU(2560, 512, 384)` plus a plain SwiGLU MLP for `shared_experts`.
14. **Final norm and LM head.** **[CORRECTED]**
    - vLLM honours `head_dtype float32`: logits = mm(bf16 h, bf16 Wᵀ, out = fp32), with no softcap, no bias and an untied head.
    - **exp_036 default policy** (HYPOTHESIS "Fixed before any run", subject to Andrei's sign-off): `lm_head` is **quantised at the arm's bits** (affine g64, from its bf16 values, scales and biases bf16) like the peers' heads. The forward pass feeds **fp32 activations**: `logits = lm_head(h.astype(mx.float32))`. mlx 0.31.2 `quantized_matmul` with an fp32 input and bf16 scales returns fp32 (verified on the mini: max diff 4.2e-5 against fp32 dequant on a 1024 × 256 test). Logits must **never** be computed from bf16 activations.
    - The vendor's unquantised head is a stated deviation (HYPOTHESIS C1), measured by E11.
    - If the sign-off chooses "vendor-faithful", both K8 and K4 are converted with `--no-quantize-embeddings --no-quantize-lm-head`; the head is then an exact fp32 upcast of the bf16 weight.
15. **Tokens and EOS.**
    - eos `<|im_end|>` 127906; pad `<|endoftext|>` 127901. generation_config eos is [127906, 127901], and both stop generation. No BOS.
    - Added tokens: `<think>` 127907, `</think>` 127908, `<tool_call>` 127909, `</tool_call>` 127910, `<tool_response>` 127911, `</tool_response>` 127912.
    - Byte-level BPE with a GPT-4 regex using `\p{N}{1}`, so single digits. ByteLevel without a prefix space; no normaliser.
    - The chat template lives in `tokenizer_config.json`.
    - Never pass `fix_mistral_regex=True`. The transformers warning is a false positive caused by the missing `transformers_version`, and following its advice changes the ids.
16. **Closest mlx templates** (mini: `/opt/homebrew/lib/python3.14/site-packages/mlx_lm`).
    - Skeleton: `models/afmoe.py`. Changes needed: drop the attention gate_proj, drop muP scaling, num_dense_layers = 0, select on logits + bias, route_norm False, route_scale 1.0, fp32 router logits, router path `mlp.gate.weight`, remap `moe.router.expert_bias`, norm names per item 5, replace afmoe's quant_predicate.
    - Other sources: `qwen3_moe.py` for attention naming and sanitize stacking; `gpt_oss.py` and `olmo3.py` for layer_types and mask construction; `gemma3_text.py` for the rotating cache, but **not** its (1+w) norm.
17. **Registration without editing site-packages.** `config.json` gets `"model_file": "kolibri1.py"`. mlx_lm `load_model` execs `<model_dir>/kolibri1.py`. `convert` copies `*.py` into the output, so converted directories are self-contained. A staging directory of symlinks plus the edited config is used, and the HF snapshot is never edited in place.
18. **Quantisation via `mlx_lm.convert`.** **[CORRECTED]**
    - Only modules with `to_quantized` and a last dim % 64 == 0 are quantised. RMSNorm weights and raw arrays (`expert_bias`) never are.
    - The model's `quant_predicate` is used only when no `--quant-predicate` recipe is given. **The kit never passes a recipe.**
    - Default policy: quantise attention projections, routed and shared experts, **`embed_tokens` (QuantizedEmbedding) and `lm_head`**. Never quantise the router, `expert_bias` or norms. Scales and biases are bf16 on every quantised tensor.
    - `--dtype bfloat16` is passed explicitly, because Kolibri's config has `dtype`, not `torch_dtype`.
19. **KV quantisation gotcha.** `RotatingKVCache.to_quantized` raises. The port's sliding cache subclass overrides it to return self. exp_036 never uses `--kv-bits`.
20. **Memory.** **[CORRECTED]**
    - Weights: K8 ≈ 83.1 GB (77.4 GiB); K4 ≈ 44.0 GB (41.0 GiB). The router is stored fp32: 197 MB (+98 MB over bf16). The vendor-faithful head adds ≈ 1.6 GB.
    - KV: 20,480 B/token (10 full layers × 2 × 4 × 128 × 2 B), plus ≈ 42 MB fixed per sequence for the sliding layers (40 × 513 × 2 × 4 × 128 × 2 B).
    - Batch-1 decode reads ≈ 4.0 GB/token (K8) and ≈ 2.2 GB/token (K4), of which 0.2 GB is the fp32 router.
21. **Sanitize map.**
    - Pop `model.layers.N.moe.router.expert_bias` → `model.layers.N.mlp.gate.expert_bias` (fp32).
    - Keep `mlp.gate.weight`, upcast exactly to fp32 (item 11).
    - Stack `mlp.experts.{0..383}.{gate,up,down}_proj.weight` → `mlp.switch_mlp.{gate,up,down}_proj.weight` [384, out, in].
    - Keep `shared_experts`, attention, q_norm and k_norm, and the four norm names. Keep `lm_head` (untied). Drop nothing else.
    - Sanitize must be idempotent on an already converted checkpoint.
22. **Validation reality.** The vendor tests have no end-to-end numerical reference; `test_routing_semantics` is the only numeric test. The fidelity reference is therefore the layer-streamed numpy forward (`reference/`), written separately from the same spec, plus the tiny checkpoint reproduced from the vendor's `tests/checkpoints.py` in numpy.
23. **Numerics that cannot match official serving.** The vendor serves FP8 e4m3 128 × 128 weights with dynamic FP8 activations (1 × 128) and an FP8 KV cache. MLX has none of these. bf16 activations (not fp16) are used.
24. **Caches.** `make_cache()` returns `[SlidingKVCache(513, keep=0) if layer_types[i] == "sliding_attention" else KVCache() for i in range(50)]`. The Model has a `.layers` property. Eval harnesses use fresh caches, with no prompt-cache reuse.
25. **Unverified list, as of now.**
    - (a) The `fix_mistral_regex` effect is **resolved**: a false positive, guarded by G0.
    - (b) Convert peak memory on 128 GB is unverified. `convert.py` monitors itself; the fallback is `convert_streaming.py`.
    - (c) Numerics at long positions: checked by gate T9 (per-layer at 0–16k and decode-vs-prefill at 15,000–15,300). MLX fused SDPA at 64k–120k keys is exercised by D1 and E6 for memory and speed only.
    - (d) RoPE fp32 angle parity near 1M is out of scope.
    - (e) Layer 0–1 routed-expert use after post-training is measured by E11.
    - (f) M5 Max bandwidth: measured, not assumed.
    - (g) Template parity is gate G0.
    - (h) External tokenizers are out of scope.
26. **Sampler (new).** vLLM v0.29 order, for every model, one batched sampler per cell (all rows share parameters):
    - The input logits are fp32 for every model: peer models are wrapped in a thin module that returns `logits.astype(mx.float32)` (and exposes `layers` and `make_cache`), so BatchGenerator's log-softmax normalisation runs in fp32.
    - Temperature first (all models use T = 1.0).
    - Then **top-k** via `mx.argpartition`: threshold = the k-th largest value; mask values < threshold, so ties at the threshold are kept.
    - Then **top-p over the renormalised top-k distribution**, sorting only the kept values: `probs = softmax(masked)`, `cum = cumsum(probs ascending)`, mask where `cum ≤ 1 − p`, always keeping the largest.
    - Then categorical sampling from the masked fp32 log-probs.
    - Transcribed from vLLM v0.29.0 `vllm/v1/sample/ops/topk_topp_sampler.py::apply_top_k_top_p`. The build agent checks it against that tagged source if available; the semantics stated here are binding either way.
    - Stock `mlx_lm.sample_utils.make_sampler` applies top-p (on the full distribution) **before** top-k. It is never used.
27. **Batching (new).**
    - `BatchGenerator(model, completion_batch_size=B, prefill_batch_size=min(B, 8), prefill_step_size=2048, max_kv_size=None, stop_tokens=[[127906], [127901]] (or the peer's EOS ids), sampler=<cell sampler>)`. mlx_lm 0.31.3 sets `completion_batch_size = max(completion_batch_size, prefill_batch_size)` with a default prefill batch of 8, so passing only `completion_batch_size` would run up to 8 sequences at B < 8. `max_kv_size` stays None: setting it would turn every `KVCache` into a `RotatingKVCache`.
    - mlx_lm 0.31.3 `generate._make_cache` maps `KVCache` → `BatchKVCache(left_padding)` and any `RotatingKVCache` subclass with keep = 0 → `BatchRotatingKVCache(max_size, left_padding)` (generate.py lines 838–866, checked). It maps `ArraysCache` (Qwen DeltaNet) to left-padded batching.
    - `BatchKVCache` left-pads every newcomer to the longest live sequence and grows by concatenation, so memory and full-attention cost scale with B × the longest live sequence, with a transient copy on growth. Batches are never mixed across tasks, and B follows the memory rule (`runner/memory.py`). `generate.py` calls `mx.set_cache_limit(4 << 30)`.
    - Per-sequence `max_tokens` through `insert`. The RNG is global, so a seed is set **per cell**, and exact replay is not claimed.
28. **Reasoning split (new).** For each model family, `split_reasoning(family, rendered_prompt_ids, completion_ids, tokenizer) → (reasoning, answer, status)`, with status ∈ {closed, unclosed, none}. It works on token ids, so it can be recomputed from the stored record:
    - **Kolibri.** If the prompt ends inside an open `<think>` (127907; effort ≠ none), the reasoning runs to the first `</think>` (127908). At effort none the prompt already holds `<think>\n\n</think>\n\n`, and the whole completion is the answer. At effort high, "none" is impossible and is recorded as a defect.
    - **Qwen.** The same `<think>…</think>` logic, using the rendered prompt to tell whether `<think>` was prefilled; "none" at thinking on is a defect.
    - **Gemma 4.** Thought channel `<|channel>thought\n … <channel|>`; the answer follows `<channel|>`. With `enable_thinking=True` the template does not prefill the channel (jinja line 359 prefills an empty closed channel only when thinking is off), so the model may not open it: status "none" is legitimate, the whole completion is the answer, it is scored normally and it does not count against the pilot reasoning gate.
    - Unclosed reasoning at the cap counts as truncated and wrong. Fixtures are built from each pinned template's actual render.
29. **What the per-layer harness compares (new).** The branch outputs `r_attn` and `r_moe` of item 5, before the residual add, each relative to its own reference norm. Measuring the layer output relative to the layer update ‖h_{l+1} − h_l‖ would scale bf16 rounding of the residual by ‖h‖/‖Δh‖ (10–30 or more in deep layers and on massive-activation tokens) and fail a correct bf16 layer.

---

## 5. File-by-file specification

Signatures are binding in name and meaning; the build agents choose the internals. "Tests" lists the unit tests (§6 has the full list) that must pass on the mini before push.

### 5.1 port/

**`port/kolibri1.py`** (exists; Apache-2.0 header and "Modified by" line already present)
- *Purpose:* the MLX model for mlx_lm 0.31.3, items 4–24.
- *API:*
  - `ModelArgs` (dataclass from `BaseModelArgs`; validates the config, rejects `tie_word_embeddings`, `attention_bias` and `head_dtype ∉ {None, "float32"}`).
  - `route(logits_f32, expert_bias_f32, top_k, renormalize=False) -> (weights_f32, ids)`.
  - `Router` (not an `nn.Linear`).
  - `MLP`, `SparseMoeBlock`, `Attention(args, use_rope)`, `DecoderLayer(args, layer_type)`, `Kolibri1Model`.
  - `SlidingKVCache(RotatingKVCache)` whose `to_quantized` returns self.
  - `Model` with `.sanitize(weights)`, `.layers`, `.make_cache()`, `.cast_predicate`, `.quant_predicate`.
  - `make_quant_predicate(group_size=None, quantize_embeddings=True, quantize_lm_head=True)`.
- *Required deltas:*
  1. **Head call.** The quantised-head branch must call `self.lm_head(h.astype(mx.float32))`. The current code casts h to the scales dtype (bf16), which yields bf16 logits: that is mutant 11.
  2. **Defaults.** Default `quantize_embeddings` and `quantize_lm_head` to **True**, per the HYPOTHESIS policy.
  3. **Quantising from bf16.** When the head or embedding is to be quantised, sanitize and cast_predicate must leave them in **bf16** (no fp32 upcast before quantisation), so their scales are computed exactly as for the peers.
  4. **Router weight in fp32** (item 11): sanitize upcasts `mlp.gate.weight` exactly; `cast_predicate` excludes it; `Router.__call__` does `x.astype(mx.float32) @ self.weight.T`. `test_router_logits_are_fp32_of_bf16_operands` (INTEGRATION_LOG entry 4) changes its weight assertion from "bf16" to "fp32 and equal to the bf16 checkpoint upcast"; the output assertions stay.
  5. **Inspection hooks.** `Model.forward_hidden(inputs, cache=None) -> h_final_normed` (E11 and the gate) and `DecoderLayer.branches(h, mask, cache, force_ids=None) -> (r_attn, h_mid, r_moe, h_out, router_logits, ids)` (G2/G2q; `force_ids` forces the selection, weights still from the port's own logits).
- *Invariants:* no import of reference/; no mutation flags in this file (mutants live in `gate/port_mutants.py`); `SPEC_VERSION` constant bumped to `exp036-port-2`.
- *Tests:* `test_port_vs_ref.py`, `test_routing.py`, `test_cache_decode.py`, `test_cache_batching.py`, `test_head_dtype.py`, `test_mutants.py`, `test_mutation_guards.py`, `test_tiny_checkpoint.py::test_mlx_lm_loads_through_model_file`.

**`port/convert.py`** (exists)
- *Purpose:* `$EXP036_MODELS/Kolibri-1-BF16` → `Kolibri-1-MLX-{8,4}bit-g64`. Staging dir of symlinks plus `config.json` with `"model_file"`; `mlx_lm.convert` with the port's predicate (no recipe), `--dtype bfloat16`; tokenizer files restored byte for byte; `exp036_convert_record.json`.
- *CLI:* `convert.py --src DIR --out DIR --bits {8,4} --group-size 64 [--no-quantize-embeddings] [--no-quantize-lm-head] [--streaming] [--force]`, plus `convert.py --refresh-port-file --out DIR`.
  - The defaults quantise both embedding and head, which is a **delta**: the flags are inverted relative to the current `--quantize-*` flags.
  - `--streaming` delegates to `convert_streaming.py`.
  - `--refresh-port-file` replaces only the copied `kolibri1.py` in an existing output and rewrites its record. It is for a gate fix that touches forward code but not sanitize or quantisation.
- *Self-monitoring:* samples `sysctl vm.swapusage` and its own RSS every 10 s. It aborts itself when swap grows by more than 2 GB or RSS exceeds 0.8 × `hw.memsize`, removes `.partial`, and prints the `--streaming` command.
- *Config keys:* the staging and output `config.json` also carry `exp036_quantize_embeddings` and `exp036_quantize_lm_head` (the head policy sanitize needs at load time: a head that will be quantised stays bf16, a vendor-faithful one becomes fp32).
- *Record:* source revision (from `.cache/huggingface/download` metadata); per-file sha256 of the output; `manifest_sha256` = sha256 over the sorted lines "relpath\tsha256"; quant config; the tensor policy (which tensors are quantised, at which bits); flags; mlx and mlx-lm versions; `t_start`, `t_end`; peak RSS and swap growth. Paths through `redact_path`.
- *Invariants:* never writes into `--src`; refuses FP8 input; refuses an existing `--out` without `--force`; and `--force` deletes only files matching its own output patterns.
- *Tests:* `test_convert.py` (exists; extended): tiny 8-bit and 4-bit conversion; quant config `{64, bits, affine}`; no `.scales` for `mlp.gate`, `expert_bias` or norms; `.scales` and bf16 biases present for `embed_tokens` and `lm_head`; router weight fp32; the round trip loads through `model_file`; `manifest_sha256` reproducible; `test_streaming_bit_identical` (8 and 4 bits); `test_quantised_port_vs_reference_on_dequantised_weights` kept.

**`port/convert_streaming.py`** (new; fallback)
- *Purpose:* same output, built layer by layer with `mx.quantize`.
- *API:* `convert_streaming(src, out, bits, group_size, quantize_embeddings, quantize_lm_head) -> dict`.

**`port/README.md`**: updated to the new defaults, the fp32 router, the output names `Kolibri-1-MLX-{8,4}bit-g64` and the per-token cost.

### 5.1b reference/ deltas (before the reference is frozen)

Each lands with a unit test, all before `{{REFERENCE_SHA256}}` is filled:
1. `moe_block(i, x, force_ids=None) -> (y, info{logits, biased, top6, weights})`.
2. Dumps of `h_in`, `h_mid`, `h_out`, `r_attn`, `r_moe`, router logits, biased scores, top-6 and final log-probs (`--dump DIR`, writing `ref_record.json` with the sha256 of every dump, the checkpoint fingerprint, the reference tree sha, the gate-text sha, `t_start`/`t_end` and peak RSS).
3. **bf16-emulation mode** (`KolibriReference(..., emulate_bf16=True)`): round-to-nearest-even to bf16 wherever vLLM stores bf16 (norm outputs, q/k/v, RoPE output, attention output, o_proj, each expert output, the residual after the fp32 add), keeping router logits, routing weights and the head in fp32. Test: on the tiny checkpoints it reproduces INTEGRATION_LOG entry 1's emulation numbers (top-1 0.863–0.975 against fp32).
4. **Dequantised mode** via `reference/mlx_affine_np.py`: unpack uint32 words of an MLX affine-quantised tensor at 8 or 4 bits, group 64, `w = scale · q + bias`, written from the format (not importing mlx). Test: on tiny converted checkpoints it equals `mx.dequantize` to 0 ulp.
5. **Long sequences:** chunked attention (`attn_chunk`) so that T9 (16,384 tokens) runs within ≈ 20 GB RSS.
6. **Parallel streams:** `forward_streams(ids, variants)` carries k hidden streams through one layer-streamed pass, so the G3 mutants read each layer's weights once.

### 5.2 reference/

**`reference/safetensors_np.py`** (exists)
- Its own header parser: memory-mapped uint16 widened by `<< 16` to fp32, exactly.
- `Checkpoint.get`, `get_rows`, `get_raw`; `open_checkpoint(model_dir)`.
- No import of `safetensors`, mlx or the port.
- *Tests:* `test_tiny_checkpoint.py::test_reference_reader_matches_mlx`; `test_safetensors_np.py` (header parsing, bf16 widening exactness on all 65,536 bit patterns, offsets across shards).

**`reference/kolibri_ref.py`** (exists; frozen at the pre-registration commit after §5.1b)
- *Purpose:* the numpy forward pass, streaming per layer and per routed expert, in fp32, bf16-emulation or dequantised mode. Each block cites a vendor or vLLM line, listed in `DERIVATION.md`.
- *API:* `KolibriReference(model_dir, attn_chunk=None, emulate_bf16=False, dequant_dir=None, verbose=False)`; `.embed`, `.layer_forward(i, h, segments=None, force_ids=None)`, `.moe_block`, `.final_norm`, `.lm_head`, `.forward`, `.forward_packed`, `.forward_streams`; CLI `main()`.
- *Invariants:* imports only numpy and stdlib; never reads the port; checks the checkpoint census before computing; deterministic (no RNG).
- *Tests:* `test_reference_invariants.py` (exists) and `test_port_vs_ref.py` (exists).

**`reference/mutants.py`** (new)
- *Purpose:* G3 reference mutants, as subclasses of `KolibriReference` that override a single method each. The frozen reference file is never edited.
- *API:* `MUTANTS: dict[str, type[KolibriReference]]` with keys `sigmoid_bias_select`, `rope_on_full`, `one_plus_w_norm`, `renorm_topk`, `swap_sandwich_norms`, `rope_traditional`, `qknorm_after_rope`.
- *Tests:* `test_reference_mutants.py`. On the tiny checkpoint each mutant changes the logits by more than 1e-2 (max-abs). The registry has exactly these 7 keys.

**`reference/DERIVATION.md`:** a table mapping each reference function to vendor `kolibri1.py` lines and vLLM v0.29.0 file:line, incl. GateLinear's tiers (`gate_linear.py`), `fused_add_rms_norm` (`ir/ops/layernorm.py`) and the cos/sin storage (`rope_base.py:59–61`; item 8). `reference/README.md` is corrected to match.

### 5.3 gate/

**`gate/thresholds.json`:** frozen; content in §7.1.

**`gate/run_gate.py`**
- *Purpose:* run the gate in fixed phases, each a separate process writing its own JSON under `results/gate/<UTC>/`, then write `results/gate/gate_<UTC>.json` with every measured value next to its threshold, the per-layer CSV, a mutants JSON, the verdict per arm (`K8`, `K4`: PASS or FAIL), and the sha256s of the port, converted manifests, thresholds, reference tree, gate code and gate text.
- *Phases:* (1) G0, G1; (2) reference passes on CPU (fp32 on T1–T8 and T9, bf16 emulation, mutants in one streamed pass); (3) G2 fp32 / bf16 / G2q per layer; (4) K8 G4 and G5 generation, continuations written to disk, model unloaded; (5) reference pass on the G5 continuations, then the G5 comparisons; (6) K4 G0/G2q/G4/G5 behaviour. `mx.clear_cache()` between MLX phases.
- *Reuse:* reference dumps are cached under `$EXP036_WORK/ref/<reference tree sha>/<checkpoint fingerprint>/<gate-text sha>/` and reused on a rerun when all three match; port-side checks always rerun. Preflight `--deep` shard hashes are reused by G0 when the preflight record is from the same HEAD.
- *CLI:* `run_gate.py --all` (= every check, arms K8,K4) or `run_gate.py [--checks g0,g1,…] [--arms K8,K4] [--tiny DIR]`. With `--tiny`, it runs the same drivers against a tiny checkpoint and its 8-bit and 4-bit conversions; thresholds are reported but the verdict is "TINY", which is used by the tests.
- *Disk:* before phase 2 it checks that free space on the `$EXP036_WORK` volume ≥ its planned dump size (≈ 100 GB without cache hits); otherwise exit 3.
- *Invariants:* the verdict is computed in code; exit 0 = both PASS, 4 = K8 PASS and K4 FAIL, 1 = K8 FAIL, 3 = could not run (incident_003 contract; the last stdout line is JSON); a partial run is never PASS; a rerun after a fix runs **all** checks.
- *Functions:* `run_all(ctx) -> GateResult`; `GateContext(models_dir, work_dir, thresholds, arms)`; `require_pass(arm) -> GateRecord`, also used by `runner/guard.py`.

**`gate/checks/g0_static.py`**
- `census(model_dir) -> {tensors, params, shard_sha_ok}`.
- `strict_load_check(model_dir, port) -> {unused_ckpt_tensors, missing_params}`.
- `template_parity(tok_dir, vendor_jinja, cases) -> {n, mismatches[]}`: 12 conversations × 10 kwarg settings, from `gate/` fixtures.
- `tokenizer_parity(tok_dir, lines) -> {n, mismatches, roundtrip_fail}`, on the lines from `gate/tokenizer_lines.py` (committed rule: lines of T1–T4, identifiers from the kit's own `.py` files, seeded synthetic digit strings; plus MMLU-ProX EN/DE question lines read at gate time and never committed; the record holds only counts and sha256) and the gate text.
- `converted_config_check(converted_dir, bits, signed_off_policy) -> {...}`: reads the tensor policy from `exp036_convert_record.json` and asserts it equals the signed-off policy (thresholds.json carries the expected policy for both variants); bf16 scales and biases on every quantised tensor; router weight float32.
- `dtype_asserts(model, ids) -> {router_logits_dtype, head_dtype, router_bf16_exact_frac, head_bf16_exact_frac}`.

**`gate/checks/g1_synthetic.py`:** runs the weight-free suite (the same functions pytest calls) with `EXP036_REQUIRE_ALL=run-host` and records the results in the gate JSON; a `build-host only:` skip (files only the mini holds) is counted separately and is not a G1 skip.

**`gate/checks/g2_layers.py`**
- `per_layer(ref_dumps, port_model, mode: "fp32" | "bf16", forced: bool, text: "T1-8" | "T9") -> list[LayerRow]`. Each row has median, p99 and max of e_attn and e_moe, selection disagreements with their reference gaps, σ_l, router dtype; T9 rows are bucketed by position.
- `emu_stats(ref_dumps, emu_dumps) -> list[LayerRow]` (the calibration).
- `per_layer_quantised(converted_dir, ref, bits) -> list[LayerRow]` (G2q), plus `embedding_check` and `head_check` on the dequantised weights.
- `router_margin(ref_dump, port, mutant_bf16_router) -> {agree_port, agree_mutant, diff_p01, diff_p99}`.
- `real_weight_mutants(ref_dump, layers=[0, 3, 4, 49]) -> {mutant: fails}` for mutants 1–8 and 12–15.
- fp32 and bf16 modes build the port's layer from the BF16 shards, one layer at a time (≤ 7 GB).

**`gate/checks/g3_oracle.py`**
- `ref_bpb(texts) -> {per_text_bpb, mean_bpb, per_text_nll}`; reads the peers' bpb from `results/peers_<UTC>.json`.
- `ref_mutant_nll(T3, MUTANTS) -> {mutant: {dnll_p01, dnll_mean, resolvable}}` (48-block bootstrap).

**`gate/checks/g4_e2e.py`**
- `e2e(converted_dir, ref_logprobs, positions) -> {mean_kl, top1, top1_decisive, dnll, kl_by_window, kl_by_bucket, by_lang}`, for K8 and K4. KL is computed in fp32 from the full vocabulary, chunked by position.

**`gate/checks/g5_generation.py`**
- `greedy_vs_ref(...)`: decisive-position agreement and top-5 rate.
- `floor_and_decode(model, texts, ranges, steps=(2048, 64)) -> {floor_kl, floor_dis, decode_kl, decode_dis, per_text}`: the noise floor (prefill with `prefill_step_size` 2048 vs 64 on the same positions) and decode vs prefill, in one pass per text, for 520–1,100 (T1, T3) and 15,000–15,300 (T9); `parity_bound(floor, thresholds_G5) -> {kl_max, dis_max}`.
- `batch_parity(model, prompts, max_tokens, B=8, eos)`, BatchGenerator built as `runner.generate` builds it (item 27), with staggered `max_tokens` so that sequences are admitted mid-run, teacher-forced. `tools/peer_check.py` calls the same three functions for G8 and Q36-8.
- `behaviour(model, prompts_json)`: loop detector (a 32-token span repeated ≥ 4 times consecutively), EOS check, `scorers.lang_tag`, think-format rate.

**`gate/port_mutants.py`**
- `PORT_MUTANTS: dict[str, Callable[[ModelArgs], nn.Module]]`, 15 entries: 1 `sigmoid_bias_select`, 2 `window_512`, 3 `rope_on_full`, 4 `one_plus_w_norm`, 5 `swap_sandwich_norms`, 6 `renorm_topk`, 7 `route_scale_2826`, 8 `mup_embed_scale`, 9 `attn_output_gate`, 10 `bf16_router_logits`, 11 `bf16_head_logits`, 12 `rope_traditional`, 13 `qknorm_after_rope`, 14 `biased_weights`, 15 `swiglu_swapped`.
- Each is a subclass or patch of the port classes, built on a copy, never by editing `port/kolibri1.py`.
- *Tests:* `test_mutants.py`. On the tiny checkpoint each mutant fails its target check by more than 100 × tolerance (forced-routing (b1) bounds for the semantic ones; mutants 10 and 11 on the near-tie and large-logit fixtures; mutant 9 by strict load).

**`gate/build_gate_text.py`**
- `build(fineweb_parquet=None, out=gate/texts, work=$EXP036_WORK/gate_texts) -> MANIFEST`. CLI: `build_gate_text.py [--fineweb PARQUET]` on the mini at build time; `build_gate_text.py --work-only --fineweb PARQUET` on the mbp (RUNBOOK step 7), which writes only T5, T6 and T9 to `$EXP036_WORK/gate_texts/`, checks them against `MANIFEST.json` if it already holds their indices, and otherwise writes `results/gate_texts_<UTC>.json`.
- T1, T2, T3, T4: cut to exactly 1,536 Kolibri tokens; committed as `T*.ids.json` (ids + sha256) with the decoded text `T*.txt` alongside. The decoded text must be valid UTF-8 (no U+FFFD); if the 1,536th token ends inside a multi-byte character, the source start is advanced by one word and the cut repeated.
- T7, T8: message lists rendered through the vendor jinja at effort high and none, content cut so the render is exactly 1,536 tokens; ids committed.
- T5, T6: the first two documents at file index ≥ 5,000 with ≥ 1,536 Kolibri tokens; T9: T1, T4 and T3 in full, then T5, T6 and further deu_Latn test documents from index ≥ 5,000 in file order, cut at 16,384 tokens. Written to `$EXP036_WORK/gate_texts/`, **never committed**; `MANIFEST.json` records only the rule, row indices, token counts and sha256 (filled at build time if the parquet is on the mini, otherwise by the mbp at RUNBOOK step 7 into the gate record).
- *Tests:* `test_gate_text.py` (exactly 1,536 tokens with the Kolibri tokenizer; ids ↔ text consistency; manifest sha reproducible; no T5/T6/T9 text in the repo tree; no GPQA canary).

**Texts.**
- T1: `2026-09-22-we-trained-it-three-times-then-stopped.md` (3,910 tokens of body); T2: `2026-09-30-the-part-that-looked-fine.md` (3,626 tokens of body), front matter removed, from the blog repo. (The exp_021 post has only 1,483 tokens and is not used.)
- T3: German prose written by Claude for exp_036, at least 1,600 Kolibri tokens (≈ 1,250+ German words) before the cut.
- T4: Grundgesetz Art. 1–19 text, from the public source the build agent records in `MANIFEST.json`.

**`gate/behaviour_prompts.json`:** 20 short factual or one-step tasks (10 EN, 10 DE), written by the build agent, frozen in the gate code hash. Each asks for an answer in two complete sentences: `scorers/lang_tag.py` needs at least two function words, so a correct one-word or number-only answer would be tagged "unknown" and counted by the blocking rule.

### 5.4 runner/

**`runner/jsonl.py`**
- `open_run(path, header: dict) -> Writer`. On open, if the file does not end in `\n`, a single `\n` is appended (still append-only), so a torn line never corrupts the next record.
- `Writer.append(record: dict)`: write a line, flush, `os.fsync`.
- `completed_keys(arm, task, effort) -> set[Key]`: scans `results/raw/*/<arm>/<task>_<effort>.jsonl` across sessions.
- `scan(path) -> (records, n_unparsable)`: skips unparsable lines and counts them in the status.
- A resumed cell keeps appending to the file of the session where it started, with a `type: "resume"` header.
- Torn lines from withheld-set files are copied to `$EXP036_PRIVATE/aborted/`; the repo `aborted/<UTC>-<what>/NOTE.md` records only their sha256 and byte count. Torn lines from public files are copied to `aborted/<UTC>-<what>/`.
- Header record first, with `type: "header"`, holding: host label, chip, memory, macOS, python, mlx, mlx-metal, mlx-lm, git commit and dirty flag, port sha, model manifest sha, asset revision, sampler config, effort, cap, B, `b_fallback_from`, seed, item-manifest sha, plan amendment sha, gate record sha, Metal limits, iogpu, power mode.
- *Invariants:* append-only; duplicate keys are kept, and only the first complete record counts.
- *Tests:* `test_jsonl.py` (torn line then resume: the next record parses; duplicates; fsync called; S2→S3 hand-over; a torn private line leaves no `text` field for a withheld task anywhere in the repo tree).

**Record schema** (`type: "record"`):

```json
{"key": {"arm": "K8", "task": "gpqa_de", "effort": "high", "item": "<item_id>", "pass": 0},
 "item_sha256": "...", "prompt_sha256": "...", "rendered_prompt_tokens": 412,
 "completion_ids": [ ... ]  |  "completion_ids_sha256": "..." (withheld sets),
 "completion_tokens": 7310, "reasoning_tokens": 6982, "answer_tokens": 328,
 "text": "<raw completion, special tokens kept>"  |  "text_sha256": "..." (withheld sets),
 "finish_reason": "stop" | "length", "truncated": false, "reasoning_status": "closed",
 "t_submit": "...Z", "t_first_token": "...Z", "t_done": "...Z", "wall_s": 61.2,
 "batch_id": 17, "batch_size": 8, "cell_seed": 3141592653, "max_tokens": 32768,
 "sampler": {"order": "vllm", "temperature": 1.0, "top_p": 0.97, "top_k": 128}, "eos_ids": [127906, 127901]}
```

Withheld sets write the full record (with `completion_ids` and `text`) to `$EXP036_PRIVATE/raw/<session>/<arm>/…` and the hashed record to the repo.

**`runner/sampler.py`**
- `make_vllm_sampler(temperature: float, top_p: float, top_k: int) -> Callable[[mx.array], mx.array]` (batched; item 26).
- `filter_top_k_top_p(logprobs_f32, top_k, top_p) -> masked`, also exported in numpy as `filter_np` for the tests.
- `fp32_logits(model) -> nn.Module`: the thin peer wrapper of item 26.
- *Tests:* `test_sampler.py`: mask parity with a numpy transcription of vLLM `apply_top_k_top_p` on 1,000 random vectors (ties included); the kept set differs from `mlx_lm.make_sampler`'s on a constructed case; chi-square sanity on 20,000 draws (seeded); the wrapper returns fp32.

**`runner/chat.py`**
- `render(family, tokenizer, messages, effort|thinking) -> (text, ids, sha256)`. Kwargs: Kolibri `reasoning_effort`; Gemma 4 `enable_thinking=True`; Qwen `enable_thinking=True`. Peers render with the committed `runner/templates/<family>.jinja`; `chat.py` records whether it equals the local mlx-community template byte for byte.
- `eos_ids(model_dir) -> list[int]`, from `generation_config.json`.
- `family_of(arm) -> str`.
- *Tests:* `test_chat.py` (Kolibri effort sentences for none, low, medium, high; Gemma and Qwen thinking toggles against the committed templates).

**`runner/memory.py`**
- `effective_limit() -> (L_bytes, sources)`: max of `mx.device_info()["max_recommended_working_set_size"]` and `iogpu.wired_limit_mb × 2²⁰`, the latter read via `sysctl -n` without sudo.
- `need_bytes(weight_bytes, B, prompt_max, cap, kv_bytes_per_token, fixed_window_bytes) -> int` = weight_bytes + 2 × B × (prompt_max + cap) × kv_bytes_per_token + B × fixed_window_bytes + 4 GiB, with weight_bytes the summed safetensors sizes of the arm's directory.
- `choose_B(arm, task, cap, L) -> int`, the largest of {16, 8, 4, 2, 1} with need ≤ 0.9 L, AIME ≤ 8.
- `sysctl_advice(k8_cells, L) -> str | None`: prints `sudo sysctl iogpu.wired_limit_mb=114688` with the B gain whenever L' = 112 GiB would raise some K8 cell's B, or need(1) > 0.9 L.
- *Tests:* `test_memory.py`: arithmetic; Kolibri 20,480 B/token and 42 MB from config; at L = 107.5 GiB, K8 GPQA cap 32,768 → B = 8, cap 65,536 → B = 4, IFBench cap 16,384 → B = 16; at L = 112 GiB the same; no advice at those caps; advice when L = 90 GiB; AIME cap.

**`runner/generate.py`**
- `run_cell(model, tokenizer, items, arm, task, effort, cap, B, seed, writer, family)`.
- Uses `BatchGenerator` exactly as item 27; inserts items in manifest order, task-grouped, refilling slots as sequences finish. Each record is written as soon as its sequence finishes. `mx.random.seed(seed)` once per cell.
- Logs every decode step (`n_live`, `padded_len`, `step_seconds`) to `results/raw/<session>/<arm>/<task>_<effort>.steps.jsonl` (pilot: under `results/pilot/<UTC>/`).
- Writes and refreshes the heartbeat `results/raw/<session>/.heartbeat` (cell, B, last record time).
- Logs `thermalState` (`osascript -l JavaScript -e 'ObjC.import("Foundation"); $.NSProcessInfo.processInfo.thermalState'`) and the power mode every 10 minutes.
- Also `teacher_force_logprobs(model, ids, chunk=2048) -> np.ndarray`, used by the gate and H8.
- *Invariants:* one item per sequence; no stop strings other than EOS; truncation means `finish_reason == "length"`.
- *Tests:* `test_runner_tiny.py`: end-to-end tiny cell; resume after a simulated kill; the crash fallback (stale heartbeat → next lower B, `b_fallback_from` set; after two fallbacks the cell moves to `aborted/`); record schema; B = 4 vs B = 1 greedy equality on the tiny model; per-sequence `max_tokens`; `len(gen._generation_batch) + len(gen._prompt_batch) <= B` at every step for B ∈ {1, 2, 4, 8}.

**`runner/guard.py`** — every function exits non-zero with a clear message:
- `require_identity()`.
- `require_clean_tree(allow=("results/", "aborted/", "evidence/"))`.
- `require_signoff()`: HYPOTHESIS.md at HEAD has a line matching `^- Signed off by: Andrei \(.+\)$`.
- `require_gate(arm)`: PASS, with the port, manifest, thresholds and reference shas matching.
- `require_peers(arm)`.
- `require_plan()`: the newest amendment of type "plan" and its `plan_fixed_<UTC>.json` committed at HEAD, and HEAD == `@{u}`; the `scorers/` tree sha equals the value in that amendment.
- `require_metal_limit(plan)`: refuses (exit 1, printing the sysctl line) if `memory.effective_limit()` < `plan["L_used"]` while any K8 cell is queued. Called by `run.py session` and `cell` at start and on every resume.
- `require_tier2()` / `tier2_at_head()`: the "Tier-2 analysis" amendment is at HEAD, in HYPOTHESIS.md or as a committed `amendments/*.md` file (for `score_all.py` and B4 cells).
- `excluded_arms()`: {arm: reason} for K4 after a real gate record with K8 PASS and K4 FAIL, and for every arm whose newest peer-check verdict is `fail` (review fix 2026-10-03; read by `run.py pilot --without` and `plan_fix`).
- `require_assets(arm | task)`.
- *Tests:* `test_guard.py`, on a temporary git repo, with a monkeypatched sysctl reader.

**`runner/seeds.py`:** `cell_seed(arm, task, effort, pass_) -> int`, the first 4 bytes of sha256 of `"exp036|…"`, big-endian. *Tests:* stable values.

**`runner/simulate.py`**
- `fit_step_model(steps) -> (a, b, c)` for step_time = a + b·n_live + c·n_live·padded_len, least squares on the steps of the last third of the arm's pilot wall time.
- `simulate_cell(n_items, lengths, prompt_lengths, B, step_model, prefill_rate, seed=36) -> hours`: continuous batching with B slots, lengths resampled with replacement and scaled as in plan rule 4, admission prefill included, tail included.
- The fit is least squares constrained to a, b, c ≥ 0 (a step cannot get faster as sequences grow); it equals the unconstrained fit whenever that is non-negative.
- *Tests:* `test_runner_simulate.py` against a synthetic trace with known a, b, c; determinism; `test_integration_contracts.py` (non-negative on noisy steps).

**`runner/plan_fix.py`**
- `fix(pilot_summary, steps, s1_records, rules, existing_amendments) -> (plan_fixed: dict, amendment_md: str, k: int)`.
- It applies, in order: (1) the cap-raise rule; (2) the parse and reasoning gate with `diagnose()`; (3) the B rule; (4) the projection by `simulate.py`; (5) P0 … P10 against B_main = min(31, 40 − S1 hours) with the two-session split (each ≤ 16 h); (6) Tier B in order, each added whole if the total and the split still fit; (7) the ordered queue and its split point; (8) the row sets (H2: 5 rows, 4 under P10; the AIME secondary iff B1; H7 adds GPQA iff B8, GPQA-D DE only where K8 runs it, so not under P10); (9) power via `analysis/power.py`.
- Excluded arms (`guard.excluded_arms`, from the gate and peer records; review fix 2026-10-03): no Tier-A or Tier-B cell of theirs is queued (B4 needs K4, K8 and G4); the plan records `excluded_arms`, `peers` (the MoE peers left) and `not_run` (K4 excluded: H1, H7, H8, D1; Qwen3.6 dropped: H4; no MoE peer left: H3, H6), which `analysis/verdicts.py` reads. K8 not fitting and K4 excluded is STOP.
- S1 hours = Σ (t_end − t_start) over the S1 records (preflight deep, convert, peer check, gate phases, bench cells, pilot).
- *CLI:* `plan_fix.py --pilot results/pilot_summary_<UTC>.json [--pilot …]` (several summaries after a re-pilot, the full pilot's first); it reads S1 hours from the S1 records and prints the paths it wrote. It refuses (exit 1, nothing written) a summary not marked `"complete": true`, a set of summaries that misses a pilot cell of an arm that runs (a cell skipped by the memory rule counts as present), and cells of an excluded arm. Exit 0 FIXED, 2 STOP or BLOCKED.
- It writes `results/plan_fixed_<UTC>.json` and `results/AMENDMENT_<k>_<UTC>.md` with k = the next free amendment number found by scanning HYPOTHESIS.md and `amendments/`, and a `supersedes` field for a re-pilot. It never opens an existing file for writing and never runs a model.
- *Tests:* `test_plan_fix.py`: the synthetic nominal pilot gives P0 + {B1, B2, B5}; the pessimistic gives P8 and no Tier B; the adverse gives P10; slower gives STOP; a session over 16 h steps down the ladder; one truncation (or one completion > 0.5 × cap) gives a raise to the ceiling; one parse failure gives no BLOCK, two with an extractor mismatch give BLOCKED; unclosed-at-cap counts as truncation; Gemma "none" is not a reasoning failure; amendment numbering after a gate-fix amendment; a re-pilot writes a new amendment with `supersedes`; same input, byte-identical output.

**`runner/run.py`** (CLI; every subcommand calls the guards):
- `pilot [--cells ARM:TASK[:EFFORT],…] [--without ARM,…]`: identity, gate, peers and sign-off; runs the pilot items from `tasks/selection_rules.json` (or only the named cells, for a re-pilot after a BLOCKED cell; the cell ids plan_fix prints are accepted, an unknown cell is refused); `--without` leaves out arms the records exclude (`guard.excluded_arms`) and is refused for any other arm; writes `results/pilot/<UTC>/` and `pilot_summary_<UTC>.json` with `requested`, `skipped`, `without` and `"complete": true`. An interrupted pilot writes no summary (exit 1). No accuracy scoring; length, truncation, status and parse counts only.
- `abort-pilot --stamp <UTC>`: moves an interrupted pilot without a summary to `aborted/<UTC>-pilot/` with a NOTE.md, its private files to `$EXP036_PRIVATE/aborted/<UTC>-pilot/` (sha256 and bytes in the note, a `private_moved` line in `evidence/withheld_manifest.jsonl`).
- `session --name S2|S3|S3b`: requires the plan and the Metal limit; appends the sha256 of every new private file to `evidence/withheld_manifest.jsonl`; runs the queue from `plan_fixed_<UTC>.json` in order, skipping completed cells. It stops admitting new cells when elapsed + the next cell's projection > `session_cap_h`; started cells finish. `S3b` admits only Tier-A cells, and only while the projected run total ≤ `overrun_ceiling_h`; any Tier-A cell it cannot admit is recorded NOT RUN. It stops cleanly on SIGINT after the current records and is resumable. On start it checks the heartbeat and applies the crash fallback.
- `cell --arm A --task T [--effort E]`: a single cell, the same guards (for repair).
- `status`: per-cell progress, truncation and parse counters (no accuracy) and ETA from the step model; prints and writes `results/status_<UTC>.json`.

`plan_rules.json` content: §7.2.

### 5.5 tasks/

**`tasks/assets.py`**
- `models_dir()`, `data_dir()`, `private_dir()`, `work_dir()`.
- `asset(name) -> Asset(path, revision, kind)`, from assets.json.
- `verify_revision(asset) -> (ok, found)`. It reads `<dir>/.cache/huggingface/download/**/*.metadata`: the first line is the commit hash, the second the etag (the LFS sha256 for weights). For git clones it reads `.git/HEAD` and refs (no network).
- *Tests:* `test_assets.py`, against fake directories with metadata files.

**`tasks/gpqa.py`**
- `load_diamond_en(path) -> list[Item]` (198) and `primary_197(items)`, which excludes eval-framework's overlong item by `sha256(row["Question"].strip()) == OVERLONG_SHA256` (constant from the patched vendored file).
- `load_diamond_de(path)` (198, `is_diamond`) and `load_pilot_main(path, diamond_ids, n=8)`, which takes rows whose Record ID is **not** in the `gpqa_diamond.csv` Record ID set, in a seed-36 order.
- `render_en(item)` and `render_de(item)` use the vendored `tulu3_cot_prompt`, `tulu3_cot_prompt_de` and the option shuffles: EN `GpqaReader.read`, whose seed is the sha256 of the option texts; DE `shuffle_correct_with_distractors` seeded by question + correct answer.
- `Item = {id, item_sha256, prompt_text, gold_letter, n_options}`. Text and gold never leave memory except into `$EXP036_PRIVATE`.
- *Tests:* `test_gpqa.py`, on **synthetic** GPQA-shaped rows we write ourselves: shuffle determinism against the vendored function; the overlong exclusion by hash; DE `is_diamond` filtering; pilot ∩ Diamond Record IDs = ∅.

**`tasks/mmlu_prox.py`**
- `parallel_ids(lite_dir) -> dict[category, list[str]]`: ids present in both en and de, per category, each in one seed-36 permutation. n_M (a multiple of 14) takes n_M/14 ids per category as a prefix, so smaller sets are nested and category-balanced.
- `category_counts(full_dir, lang) -> dict` (the full test split, for post-stratification).
- `load_lite(lang, ids)`, `load_full_pilot(lang, n)` (non-Lite ids) and `c1_items(n=100)`.
- `render_en`: vendored `MMLU_PRO_COT_V2` prompt. `render_de`: the vendored `tulu3_cot_prompt_de` with `get_n_letters(10)`, keeping the German answer phrase.
- *Tests:* `test_mmlu.py` on small synthetic rows (per-category prefix nesting, A–J letters, parallel ids, counts).

**`tasks/aime.py`**
- `load_en(path)` and `load_de(path)`; `render_en`: vendored `_AIME_QUERY_TEMPLATE`; `render_de`: `tasks/prompts/aime_de.txt` (a frozen German translation of the NeMo-Skills wrapper, keeping `$\boxed{answer}$`; Apache-derived, in NOTICE).
- *Tests:* `test_aime.py`.

**`tasks/ifbench.py`**
- `load(path) -> 300 items (key, prompt, instruction_id_list, kwargs)`; `render(item)`: the prompt verbatim as the user message.
- *Tests:* `test_ifbench_loader.py` on a 3-row synthetic file.

**`tasks/rgb.py`** — all items are EN; every prompt is rendered at run time from the local RGB files:
- `load_en(path)`, `load_fact(path)`, `load_int(path)`.
- `closed_book(item)`: `passage_num = 0`, the instruction from `config/instruction.yaml["en"]["instruction"]` formatted with `DOCS=''`, and no system message.
- `forced(item)`: `tasks/prompts/rgb_forced_en.txt` (our text).
- `negative(item, seed=2333)`: RGB's `processdata(noise_rate=1.0, passage_num=5)` rule reproduced from its description, with the RGB system prompt.
- `fact(item)`: `config/instruction.yaml` (system + instruction; RGB at `65ec39e4` has no `instruction_fact.yaml`), with documents from `positive_wrong` by RGB's `processdata` `_fact` branch (RGB's README procedure for counterfactual robustness; `tasks/selection_rules.json` "rgb" › "fact").
- The chosen document **indices** are recorded in the manifest, so the mbp reproduces prompts by index and not by RNG. A prompt-sha mismatch against the manifest exits 1.
- *Tests:* `test_rgb_prompts.py` on synthetic RGB-shaped items with a stub instruction string: deterministic indices; the closed-book prompt carries no system message; the noise-rate-1.0 construction contains no positive passage.

**`tasks/fineweb2.py`:** `iter_docs(parquet, start=0, n=5000)` and `gate_docs(parquet, start=5000, min_tokens=1536, k=2)`; `pyarrow` is imported lazily.

**`tasks/build_manifests.py`**
- `build_all(out_dir="tasks/manifests", private=$EXP036_PRIVATE) -> dict`.
- Public sets (MMLU-ProX, AIME EN, IFBench): `{id, item_sha256, prompt_sha256}` plus `category` and `gold` for MMLU-ProX and `gold` for AIME EN. No item text.
- Withheld sets (GPQA EN/DE, AIME-DE, RGB): `{id, item_sha256, prompt_sha256}` and the RGB document indices only. No per-item gold hash (a hash of a letter or an integer is trivially reversible); text and gold go to `$EXP036_PRIVATE/manifests/`, bound by the sha256 of the whole private file in the plan amendment.
- Also writes `tasks/manifests/mmlu_prox_category_counts.json` (full test split counts per language) and `tools/withheld_shingles.sha256`: the sorted 12-hex prefixes of sha256 over normalised 8-word shingles (lower-case, whitespace collapsed, punctuation other than digits removed; all-digit shingles left out) of GPQA EN/DE questions and options (Diamond and main), AIME-DE problems, RGB queries and answers and RGB's instruction strings, plus the full sha256 of every withheld option string of ≥ 3 words and of every RGB query, GPQA EN/DE question and AIME-DE problem of 3–8 words; shingles of the kit's public text (LICENSE-APACHE-2.0, gate texts) are left out (review fix 2026-10-03). The RGB documents and the FineWeb-2 rows are not in the file (≈ 47 MB of hashes); the leak check streams them from `$EXP036_DATA`.
- Writes `results/manifests_<UTC>.json` with the sha256 of every manifest, public and private.
- *Invariant:* the same data in gives byte-identical manifests out.
- *Tests:* `test_build_manifests.py` on synthetic data: no `text` or `gold` key in withheld manifests; shingle normalisation.

**`tasks/selection_rules.json`** (frozen): seed 36 for the MMLU per-category order and the GPQA pilot pick; RGB seed 2333; the pilot pools and counts per arm (HYPOTHESIS "Pilot"); C1 items; n_M ladder [588, 504, 406, 350, 294, 252, 196, 154].

**`tasks/vendored_evalfw/`**
- Verbatim copies: `benchmarks/cot.py`, `gpqa.py` (patched), `gpqa_ellamind.py`, `mmlu_pro.py`, `math_reasoning.py`, `answer.py`, plus the helpers they need, and eval-framework's NOTICE if one exists.
- **`gpqa.py` patch:** the `_OVERLONG_QUESTION = (…)` string literal is replaced by `_OVERLONG_QUESTION_SHA256 = "<sha256 of the stripped question>"`, the filter compares hashes, and a "Modified by Miktam for Chronos exp_036" line is added. The main session computes the hash from the upstream file on the mini; the question text never enters the repo or any commit, and the upstream file is never staged.
- A thin `shim.py` exposes the functions the kit uses (`tulu3_cot_prompt`, `tulu3_cot_prompt_de`, `tulu_answer_v2`, `tulu_answer_de`, `GpqaReader.read`, `shuffle_correct_with_distractors`, `_AIME_QUERY_TEMPLATE`, `_extract_boxed`, `_strip_string_with_bug`, `get_n_letters`). The originals import the eval-framework package; the shim re-implements **nothing**, and only loads the source text with stubs for unused imports.
- `MANIFEST.json` holds the sha256 of every file (and both the upstream and the patched sha256 of `gpqa.py`); `SOURCE.md` holds the URL, tag v0.14.2, the licence and the patch note.
- *Tests:* `test_vendored_evalfw.py`: the manifest matches; prompt strings for synthetic items equal hand-written known-good renders; `gpqa.py` contains no `_OVERLONG_QUESTION = (` literal and no string of ≥ 8 words other than templates on an allowlist.

### 5.6 scorers/

All scorers are deterministic, pure functions with golden fixtures in `scorers/golden/` (inside the hashed `scorers/` tree). **No fixture contains withheld or RGB text.** Synthetic questions and stub instructions only.

| File | API | Rules |
|---|---|---|
| `reasoning.py` | `split_reasoning(family, rendered_prompt_ids, completion_ids, tokenizer) -> (reasoning, answer, status)` | Item 28. Unclosed at the cap → truncated → wrong. |
| `mc.py` | `extract_gpqa_en(answer)`, `extract_gpqa_de(answer)`, `extract_mmlu_en(answer)`, `extract_mmlu_de(answer)` | The vendored regexes (`tulu_answer_v2(4 \| 10)`, `tulu_answer_de`; the DE extractor extended to A–J for MMLU by changing the letter class only). The last match wins. None → parse failure. |
| `aime.py` | `extract(answer)`, `correct(extracted, gold)` | Vendored `_extract_boxed`, then the strip, then an integer compare. |
| `rgb.py` | `checkanswer(pred, gold)` (our implementation of RGB's rule); `classify_cb(answer, gold) -> correct\|abstain\|wrong` (truncated is set upstream); `classify_neg(answer) -> rejected\|not`; `classify_fact(answer, true, fake) -> corrected\|deferred\|detected\|other` | Correct iff checkanswer = 1 for every answer slot (lower-cased alias containment) and the case-sensitive string "insufficient information" is absent. The lexicon only splits non-correct items into abstain and wrong. Rejection = "insufficient information" or a lexicon hit. Detection = "factual errors" or the lexicon. |
| `lang_tag.py` | `tag(text) -> "de"\|"en"\|"mixed"\|"unknown"` | Frozen stopword lists (≈ 100 each); ratio rules frozen. |
| `abstain_lexicon.json` | — | Frozen at pre-registration. Additions only by a numbered amendment, and only before the first record of any main (non-pilot) cell exists; frozen at the plan amendment at the latest. |
| `score_all.py` | `score_all(results_dir, private_dir=None) -> scores/*.jsonl` | Requires the Tier-2 amendment. One score record per raw record: `{key, item_sha256, extracted, category, correct, truncated, parse_status}`. `extracted` is a letter (GPQA, MMLU), an integer (AIME), the list of instruction-id results (IFBench) or null (RGB: category only). Withheld-set score records carry no free-text field. IFBench is scored by `ifbench_adapter.py` on the mini. |
| `ifbench_adapter.py` | `score(raw_jsonl, ifbench_src, ifbench_test) -> scores` | Calls `evaluation_lib` from `IFBench-src` (loose and strict) through `ifbench_driver.py` in the IFBench venv (`$EXP036_IFBENCH_PY`, else `<models>/ifbench-venv`), with the prompts from the local IFBench_test copy. |

*Tests:* `test_scorers_mc.py`, `test_scorers_aime.py`, `test_scorers_rgb.py` (golden fixtures incl. the overlap case "gold string plus a lexicon hedge" → correct, and "gold string plus 'insufficient information'" → not correct; plus `test_checkanswer_matches_rgb_in_place`, which imports RGB's `evalue.checkanswer` from `$EXP036_DATA/RGB-src` and compares on 500 synthetic cases), `test_reasoning_split.py` (split from ids == split from text with special tokens kept; Gemma "none"), `test_lang_tag.py`, `test_score_all.py` (raw → scores round trip, byte-identical on a rerun; no free text for withheld sets), and `tests/test_scorers_ifbench.py` (synthetic responses with known loose/strict outcomes for ≥ 8 instruction ids; it starts the IFBench venv itself, so it runs in the normal suite). The test file names in this paragraph are the planned ones; the built names are listed in §6.

### 5.7 bench/

| File | Cell | Spec |
|---|---|---|
| `fit.py` | D1 | K4 alone; N ∈ {32,768; 65,536} × 3 reps of exact-length prefixes of `pad_120k.txt`; `clear_cache`, `reset_peak_memory`, prefill step 2048, 512 greedy tokens, `get_peak_memory`, prefill tok/s. Output `results/bench/fit_<UTC>.jsonl`. |
| `speed.py` | H1 | K4 and G4 resident; 1 warm-up each; 10 blocks with randomised order (seed 36); 1,024-token prompt from `pad_4k.txt` in each model's tokenisation; greedy, EOS masked (a logits processor setting EOS to −inf), exactly 256 tokens; `generation_tps` and `prompt_tps`; 30 s idle; power mode and `thermalState` before and after each block. |
| | descriptive | K8, G8, Q36-4/8, Q38-4/8: 5 reps each; prefill at 4,096; batch {1, 2, 4} aggregate. |
| `ladder.py` (Tier 2) | E6 | K4, K8, G4 at 4k (`pad_4k.txt`), 15k (`pad_15k.txt`), and 32k / 64k / 120k (exact-length prefixes of `pad_120k.txt`; 120k for K4 only), 3 reps (2 at 120k), fresh cache, `prompt_tps`, peak memory, 64-token decode; 60 s idle between sizes. Asserts each fixture exists by name at preflight. |
| `tokenizer_ratio.py` | H5 | `ratio(parquet, tokenizers, n=5000) -> per-doc (bytes, chars, tokens)` for Kolibri (raw `tokenizers`), Gemma 4 and Qwen3.6 (+ the Qwen3.8 hash); digit-dense subset; EN descriptive texts. CPU only; no mlx import. |
| `kl_8v4.py` | H8 | T1–T6 (T5–T6 from `$EXP036_WORK/gate_texts/`); per model, per position KL(p8‖p4) in fp32, from bf16-rounded logits (Kolibri also from fp32); 8 byte blocks per text; per-block sums of KL, bytes and tokens written to `kl_8v4_<UTC>.json`. Kolibri: dump K8 log-probs to `$EXP036_WORK/kl/` (fp32 .npy), then load K4. Peers: both bit-widths loaded together. |
| `batch_flip.py` | C1 | K8, effort none, greedy, 100 C1 items, at the memory-rule B for the MMLU cell at the initial cap (computed at preflight and recorded in the C1 file) vs B = 1; answer-flip rate and first-divergence position. |
| `run_bench.py` | — | `--cells fit,speed,speed_desc,c1,tokenizer,kl[,ladder]`. Fixed order: speed cells first on a cool machine (10-min idle, `thermalState` nominal), then fit, KL, tokenizer, C1. Each cell writes its own file with `t_start`/`t_end` and is skipped if already complete. |

Every bench cell requires `guard.require_gate` for Kolibri arms and `guard.require_identity`.

### 5.8 analysis/

**`analysis/stats.py`** (Tier 1)
- `rng(h_id) -> np.random.Generator(PCG64(int.from_bytes(sha256(b"exp036|" + h_id)[:8], "big")))`.
- `boot_paired(diffs_by_stratum, B) -> np.ndarray`; `boot_mean_rows(rows, vendor_terms=None, strata=None) -> np.ndarray` (H2: stratified per row, by category within MMLU rows, post-stratified weights, plus the normal vendor draw); `boot_blocks_stratified(per_text_blocks, byte_weights, B)` (H8).
- `post_stratified_mean(scores_by_category, weights) -> float`.
- `two_pass_row_var(x1, x2) -> float` = Σ(x₁ − x₂)²/(4n²) (H2 GPQA rows if B10).
- `t_test_one_sample(x, theta0) -> (p, p_rev, ci95)` (H1, df = n − 1).
- `percentile_ci(samples, level) -> (lo, hi)`.
- `p_one_sided(samples, theta0, null_side: "le" | "ge") -> float` = (1 + #null) / (B + 1), with θ* = θ₀ on the null side; `mc_se(p, B)`.
- `holm(pvals: dict) -> dict(adjusted, step, reject)`, step-down with running maximum.
- `escalate(h_id, compute, threshold) -> result`: reruns with B = 100,000 when p is within a factor of 2 of its Holm threshold.
- *Tests:* `test_stats.py` (known-answer cases; Holm on textbook examples, including a case where a later bound clears its threshold but the adjusted p does not; ties on the null side; t-test against a hand-computed case; seed per H independent of order; post-stratified mean; two-pass variance).

**`analysis/verdicts.py`**
- `compute(results_dir, plan, scores) -> verdicts dict`, written to `results/verdicts_<UTC>.json` in the exp_023 shape: `{ts, hardware, config, manifests, summary, verdicts: {H1…H8, D1, H*_detail, E1…E12, C1}}`.
- Rules exactly as in HYPOTHESIS: two Holm families; REFUTED only if not CONFIRMED; "leans refuted"; m frozen at the plan amendment with p = p_rev = 1 for NOT RUN; the paired truncation sensitivity for H3, H4, H6, H7 and the headline-eligibility flag; the E8 flag; the H2 protocol control and DiD; the H8 per-text condition and per-token sensitivity; the H6 wording state; the plain-answer state per question.
- Every threshold and margin (H1 0.75 over 10 blocks; H2 −0.04; H3 0 and −0.02; H4 0 and +0.04; H5 1.15; H6 −0.10; H7 −0.03; H8 1.5 and 5 of 6; D1 46.66 / 51.84 GiB over 3 reps; E8 flag 0.08; headline peer band 0.02; E2 bounds +0.03 and 0.40) is read from `analysis/margins.json`. The sign-off may change only the H2 margin (Amendment 0).
- Review fixes 2026-10-03: threshold comparisons carry a 1e-9 tolerance in the direction of each rule's wording (E8 "> 8 pp", the ±2 pp band, the 5 pp and −8 pp naming rules); H1 with other than 10 blocks and D1 with other than 3 reps at 64k are NOT RUN; H2 uses two passes per GPQA row only where the B10 pass-1 cell is complete (else single pass, recorded), with the truncation rate and the excluding-truncated result over both passes; H7 keeps a B8 row only where K4 and K8 both completed it (dropped rows recorded; a missing Tier-A row is still NOT RUN); a truncation sensitivity that cannot be computed makes the verdict headline-ineligible ("INCONCLUSIVE (truncation-sensitive)"); the excluding-truncated H2 number is recorded as not computable instead of crashing; the protocol control is NOT RUN without a peer; Q2 has no pre-registered sentence when the protocol control is NOT RUN and H2 is CONFIRMED or REFUTED (state `Q2_peer_control_not_run`, never a headline); Q3's German basis and E1 label come from the cells that ran; the peers come from the plan (`plan_fix` writes them).
- Refuses to run unless `results/rescore_mini_<UTC>.json` shows every scores file (public and withheld) matching the mbp's byte for byte.
- *Tests:* `test_verdicts.py`: synthetic score sets hitting CONFIRMED, REFUTED, INCONCLUSIVE, "leans refuted" and NOT RUN for every H; CONFIRMED-and-REFUTED-both-possible cases for H3 and H4 resolve to CONFIRMED; Holm ordering; D1 (64k REFUTED case); E8 flag; truncation sensitivity disagreement → not headline-eligible; H8 failing the 5-of-6 condition; m frozen when an H becomes NOT RUN; refusal without a matching rescore.

**`analysis/margins.json`**, **`analysis/vendor_values.json`** (rows with value, page, N_v, n_v, label; the peers' values for MMLU EN/DE, IFBench, RGB CB/Negative/FC, GPQA EN/DE and AIME EN/DE; the MMLU-Pro 12,032 category counts), both frozen in the analysis tree.

**`analysis/power.py`:** `power(h, n, alpha_level, assumptions) -> float`, using the HYPOTHESIS formulas, plus `budget(assumptions)`, which reproduces the HYPOTHESIS ladder and scenario tables. *Tests:* reproduces the HYPOTHESIS power tables to ±0.01 and the scenario picks exactly.

**Tier 2:** `analysis/exploratory.py` (E1–E12), `analysis/tables.py` (key-numbers table with vendor | ours | V/R | CI | truncation rate | excluding truncated; plain-answer sentences). Frozen by the Tier-2 amendment.

### 5.9 tools/

**`tools/redact.py`:** `redact_path(p) -> str` (§2 Privacy) and `HOST_LABEL`.

**`tools/preflight.py`**
- *Mode `--quick`* (< 1 min) records only whitelisted fields: `machdep.cpu.brand_string`, `hw.model`, `hw.memsize`, GPU core count, macOS version, `mx.device_info()`, `iogpu.wired_limit_mb`, power source, Low Power Mode and `powermode` (`pmset -g`), `thermalState`, free disk on the `$EXP036_MODELS` volume, swap usage, venv versions against `env/versions.json`, git identity (hard requirement), HEAD, upstream and dirty flag, every asset's presence and revision, exp_007 fixture presence by real name, the K8 memory-rule B per cell and any sysctl advice, and the C1 B. It never calls `socket.gethostname`, `scutil` or `system_profiler` hardware overviews (serial, UUID).
- Disk: ≥ 260 GB free beyond the downloads before conversion, ≥ 120 GB free after it (conversions 127 GB, gate and H8 work files ≈ 100 GB).
- *Mode `--deep`* adds the sha256 of every model shard against its etag, with parallel workers, plus dataset files, with `t_start`/`t_end`.
- *Output:* `results/preflight_<UTC>.json`. Prints the phase plan and the next RUNBOOK step.
- *Exit codes:* 0 ok, 1 unmet prerequisite, 2 warnings only.
- **Refuses to write results if the git identity is not "Miktam <hello@localfirstai.eu>".**
- *Tests:* `test_preflight.py` with monkeypatched system calls (identity refusal, disk thresholds, advice text, metadata parsing; no key matching `/serial|uuid|hostname|computername/i` in the record).

**`tools/version_record.py`:** `record() -> dict`, the exp_035 shape: OS, Python, mlx, mlx-metal, mlx-lm, numpy, tokenizers, safetensors, transformers, pip freeze sha256, git commit and dirty flag, the sha256 of every tree in the HYPOTHESIS hash table (each compared with the pre-registered value or a numbered amendment), converted manifests, asset revisions, Metal limits. Written by every phase to `results/version_record_<UTC>.json`.

**`tools/leak_check.py`**
- `check(paths | --staged | --range REV..REV | --all) -> findings`; the RUNBOOK uses `--range @{u}..HEAD --staged` (every file changed in unpushed commits plus the index).
- **Kit-authored files and run records** (everything outside `results/raw/**` and `results/pilot/**`): 100.64.0.0/10 addresses; home-directory paths (`/Users/[A-Za-z0-9_.-]+`); `hf_[A-Za-z0-9]{20,}` tokens; private keys; e-mail addresses other than `hello@localfirstai.eu` and `noreply@anthropic.com`; the GPQA canary; withheld shingles; any `text` or `gold` field in withheld-set JSONL; and runtime patterns read from the environment and never committed: `whoami`, `scutil --get LocalHostName`, `*.local`, `*.ts.net`, `EXP036_PRIVATE_NAMES`.
- **Raw and pilot outputs** (`results/raw/**`, `results/pilot/**`): withheld shingles, the canary and any `text` field in a withheld set are findings; other matches (e.g. `john.doe@example.com` written by a model) are warnings printed for the commit note.
- **Files:** refuses staged `*.pdf`, `*.parquet`, `*.npy`, `*.npz`, `*.safetensors`; any file > 5 MB outside `results/raw/**`, and > 50 MB inside it. Inside the experiment directory also archives and binary formats (`*.gz`, `*.zip`, `*.tar`, `*.tgz`, `*.bz2`, `*.xz`, `*.7z`, `*.zst`, `*.arrow`, `*.bin`, `*.pt`, `*.pth`, `*.gguf`, `*.h5`) and any file with a NUL byte or not strict UTF-8; outside it these are warnings (review fix 2026-10-03).
- **Also (review fix 2026-10-03):** the Tailscale IPv6 prefix `fd7a:115c:a1e0::/48`; JSON-escaped home paths (`\/Users\/<name>`); option windows of 3–8 words (an 8-word option is a full-hash finding); RGB-shaped and rendered-prompt field names in withheld-set records (`rendered_prompt`, `docs`, `docs_text`, `document`, `passage`, `positive`, `negative`, `positive_wrong`, `query`, `fakeanswer`).
- **Local corpora (review fix 2026-10-03):** the RGB documents (positive, negative, positive_wrong of en, en_fact, en_int, en_refine) and the FineWeb-2 rows the kit reads (0–4999, the T5/T6/T9 rows of `gate/texts/MANIFEST.json`) are streamed from `$EXP036_DATA` when present and matched against the scanned text's own 8-word shingles, minus the kit's public text and all-digit shingles: ≥ 2 hits in one unit are a finding, 1 a warning. Public raw / pilot outputs are not matched (withheld-set raw files are). Without the data a note says the check did not run.
- **Withheld text sources:** `tools/withheld_shingles.sha256` (after RUNBOOK step 7), or the GPQA/RGB/AIME-DE sources in `$EXP036_DATA` or `$EXP036_PRIVATE/manifests`. If none is present it exits 3, unless `--no-gpqa-source` is passed explicitly (only for the pre-registration push from the mini, which carries no withheld text by construction), and prints that the check was skipped.
- Also usable on a single file: `leak_check.py <post.md>` (the blog post before Andrei's deploy).
- Exit 1 on any finding. No file is allowlisted as a whole (`tools/leak_check.py` passes its own rules); `tests/fixtures/leak/` (planted synthetic leaks) is exempt only from the personal-pattern rules (addresses, paths, e-mail, host and runtime names).
- *Tests:* `test_leak_check.py` on synthetic files: a fake shingle list; an IFBench-style raw output with an `example.com` address (warning, not finding); a staged PDF refused; the range mode; exit 3 without a source.

**`tools/hooks/pre-push`:** runs `leak_check.py --range @{u}..HEAD --staged`. It is copied to the absolute hooks path `git rev-parse --path-format=absolute --git-path hooks` prints (RUNBOOK step 2 on the mbp; the pre-push checklist on the mini). `core.hooksPath` is not used, because it would apply to the whole repository.

**`tools/hash_tree.py`:** `tree_sha256(dir, include=…, exclude=…) -> hex` over sorted "relpath\tsha256" lines, with the include/exclude sets of the HYPOTHESIS hash table (e.g. `analysis/` without the Tier-2 files, `tools/` without `withheld_shingles.sha256`, `tasks/` without `manifests/`, `bench/` without `ladder.py`). `--tree <name>` prints one tree's sha256. `--fill HYPOTHESIS.md` fills every hash placeholder; `--stamp HYPOTHESIS.md` writes `datetime.now(timezone.utc)` into the `{{PREREG_UTC}}` placeholder immediately before the pre-registration commit; `--check` compares the current trees with the filled values (or a numbered amendment). `unfill(text)` turns a filled, stamped, amended text back into its template (the round-trip test and the dry run's copy; review fix 2026-10-03). *Tests:* `test_hash_tree.py` (stamp format; fill idempotent; no double-brace placeholder left after fill and stamp).

**`tools/peer_check.py`:** `check(arm) -> dict`. Covers: config quantization and its router overrides; a strict text-only load; the parameter count against config arithmetic (≤ 2 %); tokenizer and template sha256 and byte parity with the committed upstream copies; NLL and bits per byte on T1–T6; NLL(8) ≤ NLL(4) + 0.02 and KL(8‖4) < 0.2; EOS ids; and, for G8 and Q36-8, the batched-path parity (B = 8 vs B = 1 teacher-forced through `runner.generate` with mixed lengths 37–1,100 and mid-run admission, under the G5 noise-floor rule) and the greedy answer-flip rate on 30 MMLU-ProX full EN items with thinking off (≤ 2 %). Writes `results/peers_<UTC>.json` with `t_start`/`t_end`.

**`tools/status.py`:** wraps `run.py status`. `--record-block <phase>` appends a dated `## <phase> — run record (<UTC>)` block to HYPOTHESIS.md (phase, UTC from the clock, commit, result-file sha256, the gate verdict where relevant) and rewrites only the Status line. `--sync-amendments` appends, verbatim and in number order, every `amendments/*.md` file whose first line (its `## Amendment k — <type> (<UTC>)` heading) is not yet in HYPOTHESIS.md; it is a no-op when there is none. Review fix 2026-10-03: `--append-amendment FILE` appends one amendment file (plan_fix's) verbatim, refusing a malformed or repeated heading; `--tier2` says whether the Tier-2 amendment is at HEAD; `--verify-private` (on the mini, RUNBOOK step 18) checks every private file `evidence/withheld_manifest.jsonl` lists at its latest sha256, following `private_moved` lines. *Tests:* `test_status.py` (only the Status line changes in place; sync is idempotent; works with an empty `amendments/`).

**`tools/dry_run.py`:** the whole pipeline end to end in a temporary copy of the kit (its own git repository and bare remote; a tiny Kolibri BF16 at the real vocabulary with the real tokenizer, its K8/K4 conversions, tiny stand-ins for the six peers under their real tokenizers; synthetic GPQA/MMLU/AIME/RGB data, the public IFBench and FineWeb files when present): preflight (mini mode) → convert → peer check (weight checks) → gate `--tiny` (its PASS re-labelled for the guards) → Tier-2 amendment → bench cells at seconds scale → pilot → plan fix → S2/S3 → scorers (mbp and mini) → verdicts → `hash_tree --check` → leak check. `runner/plan_rules.json` is scaled (cell sizes, caps) in the copy only. The copy is first reset to the pre-registration state (`reset_run_state`: HYPOTHESIS hash table and stamp back to placeholders, `hash_tree:` amendment lines neutralised, the real step-7 manifests and shingle file removed), so the dry run is valid at any step of the run (review fix 2026-10-03). Last stdout line JSON; exit 0 when every stage passed. RUNBOOK step 3b (optional mbp self-test).

**`tools/kv_bytes.py`:** `kv_bytes_per_token(config) -> (growing, fixed_window)`. Predictions: Kolibri 20,480 / 42 MB; Gemma 4 and Qwen3.6 20,480 with their fixed parts; Qwen3.8 65,536. A mismatch is recorded, not fatal.

---

## 6. Unit tests that must pass on the mini before push

Run them in **both** mini environments with every skip turned into a failure. There is no torch and there are no weights. Expected wall time is a few minutes.

```bash
cd tasks/chronos/exp_036_kolibri_local_eval
export EXP036_TOK=~/models/exp036-mini/kolibri/Kolibri-1-BF16 EXP036_DATA=~/models/exp036-mini/data EXP036_REQUIRE_ALL=1
~/models/exp036-mini/venv314/bin/python -m pytest -q -rs tests
~/models/exp036-mini/venv312/bin/python -m pytest -q -rs tests
```

The IFBench adapter test (`tests/test_scorers_ifbench.py`) is part of `tests/`: the adapter starts `~/models/exp036-mini/ifbench-venv` itself (BUILD_SPEC's planned `tests_ifbench/` was not needed).

**Built test file names.** Each area's tests are named `test_<area>_*.py`; the table below names the planned files. The mapping: `test_sampler`, `test_chat`, `test_jsonl`, `test_runner_tiny`, `test_guard`, `test_memory`, `test_simulate`, `test_plan_fix`, `test_seeds` → `test_runner_{sampler,chat,jsonl,tiny,guard,memory,simulate,plan_fix,seeds,session}.py`; `test_cache_batching`, `test_head_dtype`, `test_safetensors_np`, `test_reference_modes`, `test_reference_mutants` → `test_port_ref_{cache_batching,head_dtype,safetensors_np,reference_modes,reference_mutants,hooks}.py`; `test_mutants` → `test_gate_port_mutants.py` (with `test_gate_{thresholds,text,refusal,checks,drivers_tiny}.py`); `test_assets`, `test_gpqa`, `test_mmlu`, `test_aime`, `test_ifbench_loader`, `test_rgb_prompts`, `test_build_manifests`, `test_vendored_evalfw` → `test_tasks_*.py` (plus `test_tasks_fineweb2.py`, `test_tasks_no_withheld_text.py`); `test_reasoning_split`, `test_lang_tag`, `test_score_all` → `test_scorers_{reasoning,lang_tag,score_all,mc,aime,rgb,ifbench}.py`; `test_stats`, `test_verdicts`, `test_power` → `test_analysis_{stats,verdicts,power,exploratory}.py`; `test_preflight`, `test_leak_check`, `test_hash_tree`, `test_status` → `test_tools_{preflight,leak_check,hash_tree,status,version_record,params,kv_bytes,redact,peer_check,env}.py`; bench → `test_bench_*.py`; `test_notice.py` as named; cross-area contracts → `test_integration_contracts.py`.

| Test file | What must hold |
|---|---|
| `tiny_checkpoint.py` (helper, exists) | numpy generator mirroring vendor `tests/checkpoints.py`: hidden 256, 6 layers, 8 q / 2 kv heads, head_dim 32, 8 experts top-2, moe and shared intermediate 256, window 65, layers 4–5 full, vocab 96,000, the exact tensor names incl. `moe.router.expert_bias`. Presets `vendor`, `pattern5`, `w513`. Deterministic per seed. |
| `port_harness.py` (helper, exists) | load, forced and recorded routing, per-layer capture, comparison statistics |
| `test_tiny_checkpoint.py` (exists) | layout, config mirror, the traps are live, writer determinism, the reference reader equals the mlx reader, mlx_lm loads through `model_file` |
| `test_port_vs_ref.py` (exists, binding) | (a) fp32: max \|Δlogit\| ≤ 1e-4 relative, identical top-1; (b1) bf16 forced routing: mean KL ≤ 1e-3, max relative error ≤ 5e-2, top-1 ≥ 0.98 at lead ≥ 0.1; (b2) bf16 natural routing: mean KL ≤ 3e-2, top-1 ≥ 0.60 (INTEGRATION_LOG entry 1) |
| `test_routing.py` (exists) | the vendor's `test_routing_semantics`; port == reference; exact bf16 bias upcast; `test_router_logits_are_fp32_of_bf16_operands` (weight now fp32 = bf16 upcast) |
| `test_cache_decode.py` (exists, binding) | fp32 cache paths ≤ 1.6e-6 class bounds; bf16 forced and natural splits (INTEGRATION_LOG entry 2); window and NoPE through `SlidingKVCache`; chunked prefill |
| `test_cache_batching.py` (new) | B = 4 left-padded batch == single within 1e-4 (fp32); `_make_cache` maps `SlidingKVCache` → `BatchRotatingKVCache`; `to_quantized` returns self; `test_refill_rotated`: tiny vendor preset in fp32 at W = 65 and W = 513, `completion_batch_size` 8 / 4 / 3, prompts up to 3W, generations up to 3W, ≥ 2 refills while other sequences are past 2W; per-sequence greedy logprobs within 1e-4 of B = 1 and of a one-shot forward |
| `test_head_dtype.py` (new) | router logits fp32; head output fp32 for unquantised **and** quantised heads; quantised-head logits from fp32 input differ from the bf16-input variant on a large-logit fixture; the bf16-exact fraction is ≈ 0 for the port and 1 for mutants 10/11 |
| `test_convert.py` (exists, extended) | §5.1 |
| `test_mutants.py` (new), `test_mutation_guards.py` (exists) | each of the 15 port mutants fails its target by > 100 × tolerance |
| `test_reference_invariants.py` (exists), `test_reference_mutants.py`, `test_safetensors_np.py`, `test_reference_modes.py` | §5.1b, §5.2 (bf16 emulation reproduces entry 1; dequantiser equals `mx.dequantize`; streams equal separate passes) |
| `test_chat_template.py` (exists) | parity of 12 conversations × 10 settings against the vendor jinja; tokenizer ids == raw `tokenizers`; single digits; special ids; `fix_mistral_regex=True` would change the ids |
| `test_chat.py`, `test_reasoning_split.py` | per-family kwargs against the committed templates; split from ids and from text agree; Gemma "none" |
| `test_sampler.py` | §5.4 |
| `test_jsonl.py`, `test_runner_tiny.py`, `test_guard.py`, `test_memory.py`, `test_simulate.py`, `test_plan_fix.py`, `test_seeds.py` | §5.4 |
| `test_assets.py`, `test_gpqa.py` (synthetic rows only), `test_mmlu.py`, `test_aime.py`, `test_ifbench_loader.py`, `test_rgb_prompts.py`, `test_build_manifests.py`, `test_vendored_evalfw.py`, `test_gate_text.py` | §5.3, §5.5 |
| `test_scorers_mc.py`, `test_scorers_aime.py`, `test_scorers_rgb.py`, `test_lang_tag.py`, `test_score_all.py`; `tests_ifbench/test_ifbench_adapter.py` (IFBench venv only) | §5.6 |
| `test_stats.py`, `test_verdicts.py`, `test_power.py` | §5.8 |
| `test_preflight.py`, `test_leak_check.py`, `test_hash_tree.py`, `test_status.py` | §5.9 |
| `test_notice.py` | every file with an Apache SPDX header, under `tasks/vendored_evalfw/`, under `runner/templates/` or matching `tests/fixtures/*.jinja` is named in NOTICE |
| `test_gate_drivers_tiny.py` | `gate/run_gate.py --tiny` runs G0 (the parts that need no real weights), G1, G2 (fp32, bf16 with a tiny emulated reference, G2q on tiny conversions), G3 (bpb and the 7 mutants on tiny), G4 and G5 (noise floor, decode vs prefill, batch parity with mid-run admission, behaviour) end to end and writes a schema-valid gate JSON with verdict "TINY"; exit codes 0 / 4 / 1 / 3 on constructed cases |

**Pre-push checklist** (main session on the mini):
1. Andrei's pre-registration decisions are recorded in HYPOTHESIS "Pre-registration decisions" with his words and the date (port publication, RGB terms, new assets and mini fetches, T3, overrun ceiling, README rows). If he declines publishing the port, `port/` and `reference/` are left out of the push and only their sha256 are committed.
2. New assets pinned in `assets.json` and `ASSETS.md` (full SHAs; ASSETS.md's disk line becomes "keep ~600 GB free"); no provisional pin left in HYPOTHESIS or RUNBOOK.
3. All tests pass in both environments, with `EXP036_REQUIRE_ALL=1` (the IFBench adapter test is part of `tests/`).
4. `tools/hash_tree.py --fill HYPOTHESIS.md`, then `--stamp HYPOTHESIS.md` immediately before the commit; `grep -c '{{' HYPOTHESIS.md` prints 0.
5. `tools/leak_check.py --all --no-gpqa-source` finds nothing over the whole directory; `.git/hooks/pre-push` installed.
6. `NOTICE` and `LICENSE-APACHE-2.0` present; every Apache-derived or copied file named (`test_notice.py`).
7. README rows with status "Pre-registered" in `README.md` (Experiments table) and `tasks/chronos/README.md` (Current contents), if Andrei agreed.
8. Commits: `scientific_log: exp_036 Kolibri pre-registration` (pointer only) and `chronos/exp_036: pre-registration — kit, gate, hypotheses` (plus `readme: exp_036 row` if item 7). As made on 2026-10-03 the second commit is 725d628, subject `chronos/exp_036: pre-registration, MLX port and kit, awaiting Andrei's sign-off`; checks go by the sha on HYPOTHESIS.md's last line, never by subject, each ending with the Co-Authored-By trailer.
9. Push (approved by Andrei for the kit and the pre-registration). Then append `Pre-registration commit: <sha>` to HYPOTHESIS.md and push that one-line commit.
10. Locally: workspace symlink and pointer bump; add `kolibri-experiment/` to the workspace chronos submodule's `.gitignore` (the vendor PDF is never committed).

---

## 7. Frozen configuration files

### 7.1 `gate/thresholds.json`

```json
{
  "version": "exp036-gate-2",
  "G0": {"tensor_count": 58353, "param_count": 78103074560,
         "full_attention_layers": [4,9,14,19,24,29,34,39,44,49],
         "template_conversations": 12, "template_settings": 10,
         "tokenizer_lines_min": 2000, "tokenizer_mismatches_max": 0,
         "router_logits_dtype": "float32", "head_output_dtype": "float32", "router_weight_dtype": "float32",
         "router_bf16_exact_frac_max": 0.01, "head_bf16_exact_frac_max": 0.01,
         "scales_biases_dtype": "bfloat16",
         "tensor_policy": {"quantised_head": {"embed_tokens": "arm_bits", "lm_head": "arm_bits"},
                           "vendor_faithful": {"embed_tokens": "bf16", "lm_head": "fp32"}}},
  "G1": {"source": "tests/INTEGRATION_LOG.md bounds (a), (b1), (b2) and cache entries",
         "mutant_fail_factor": 100, "sampler_vectors": 1000, "port_mutants": 15},
  "G2": {"fp32_forced_rel_err_median_max": 1e-4, "fp32_forced_rel_err_max_max": 1e-3,
         "fp32_selection_near_tie_gap": 1e-4, "head_fp32_logit_maxabs": 1e-3,
         "bf16_forced_median_factor_vs_emu": 3.0,
         "bf16_forced_p99_floor": 5e-2, "bf16_forced_p99_factor_vs_emu": 3.0,
         "bf16_selection_disagree_floor": 0.002, "bf16_selection_disagree_factor_vs_emu": 2.0,
         "bf16_selection_gap_sigma": 6.0,
         "router_mutant_block_if_diff_p99_below": 0.0,
         "g2q_embedding_rel_err_max": 4e-3, "g2q_head_rel_l2_max": 1e-3,
         "t9_buckets": [[0, 2048], [2048, 8192], [8192, 16384]],
         "real_weight_mutant_layers": [0, 3, 4, 49], "real_weight_mutants": [1,2,3,4,5,6,7,8,12,13,14,15]},
  "G3": {"ref_bpb_per_text_max": 1.2, "ref_bpb_mean_vs_best_peer_max": 1.25,
         "ref_mutant_text": "T3", "ref_mutant_blocks": 48, "ref_mutant_dnll_quantile": 0.01,
         "ref_mutants": 7},
  "G4": {"K8_backstop_mean_kl_max": 0.10, "K8_backstop_decisive_lead_nats": 2.0,
         "K8_backstop_decisive_top1_min": 0.99, "K4": "descriptive"},
  "G5": {"greedy_tokens": 256, "greedy_prompts": 8, "greedy_prompt_min_tokens": 600,
         "greedy_decisive_lead_nats": 2.0, "greedy_decisive_top1_min": 0.995, "greedy_ref_top5_min": 0.99,
         "noise_floor_prefill_steps": [2048, 64],
         "parity_kl_floor": 1e-4, "parity_kl_factor": 3.0,
         "parity_dis_factor": 3.0, "parity_dis_add": 0.002,
         "decode_positions": [[520, 1100], [15000, 15300]],
         "batch_B": 8, "batch_staggered_max_tokens": true,
         "behaviour_prompts": 20, "behaviour_effort_high_cap": 8192,
         "behaviour_block_count": 2, "loop_span_tokens": 32, "loop_repeats": 4,
         "eos_ids": [127906, 127901]},
  "K4_blocking": ["G0", "same_port_sha", "G2q", "G5_behaviour"],
  "fix_cycles_max": 2
}
```

### 7.2 `runner/plan_rules.json`

```json
{
  "version": "exp036-plan-2",
  "budget": {"total_h": 40.0, "main_cap_h": 31.0, "margin": 1.15, "session_cap_h": 16.0,
             "overrun_ceiling_h": 44.0},
  "B_choices": [16, 8, 4, 2, 1], "B_max_aime": 8, "memory_overhead_gib": 4.0, "limit_fraction": 0.9,
  "kv_transient_factor": 2.0, "cache_limit_gib": 4.0,
  "kv_bytes_per_token": {"kolibri": 20480, "gemma4": 20480, "qwen3_6": 20480, "qwen3_8": 65536},
  "sysctl_advice_mb": 114688,
  "caps": {"gpqa": 32768, "mmlu": 32768, "aime": 65536, "ifbench": 16384, "rgb": 16384},
  "cap_raise": {"trigger_truncations": 1, "trigger_len_fraction": 0.5, "trigger_len_fraction_gpqa": 0.4,
                "ceilings": {"gpqa": 65536, "mmlu": 65536, "aime": 98304, "ifbench": 32768, "rgb": 32768}},
  "pilot_gates": {"block_min_failures": 2, "require_diagnosed_mismatch": true},
  "projection": {"length_scale": "mean_plus_1se", "gpqa_diamond_factor": 1.25,
                 "step_model": "a + b*n_live + c*n_live*padded_len", "fit_on": "last_third_of_arm_pilot",
                 "resample_seed": 36},
  "nM_ladder": [588, 504, 406, 350, 294, 252, 196, 154],
  "plans": ["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"],
  "plan_defs": "as HYPOTHESIS.md 'Pilot and the rule that fixes n'",
  "tier_a_queue": "as HYPOTHESIS.md 'Tier A cells, in queue order'",
  "tier_b_order": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B10"],
  "ladder_estimate_h": {"B4": 1.4},
  "crash_fallback": {"heartbeat_stale_min": 30, "max_fallbacks": 2}
}
```

`plan_defs`, `tier_a_queue` and the Tier-B cell lists are written out explicitly in the JSON by the build agent. The strings above mark the source, and `test_plan_fix.py` checks the expansion against HYPOTHESIS.

---

## 8. New assets and build-time fetches

**mbp downloads by Andrei** (HYPOTHESIS "New assets needed"; the main session pins them in assets.json first):

| Asset | Local path under `$EXP036_MODELS` | Needed by |
|---|---|---|
| RGB @ pin | `data/RGB-src` | `tasks/rgb.py` (rendering at run time); the mini re-score |
| FineWeb-2 deu_Latn test 000_00000 @ `af9c1333` | `data/fineweb-2/data/deu_Latn/test/000_00000.parquet` | `bench/tokenizer_ratio.py`, `gate/build_gate_text.py` (T5, T6, T9), `bench/kl_8v4.py` |

**Build-time fetches on the mini** (main session, with Andrei's OK, before the pre-registration push; into `~/models/exp036-mini/`; none is committed except where stated):

| What | Exact source | Size | Why |
|---|---|---|---|
| FineWeb-2 deu_Latn test parquet | as above | 104 MB | T5/T6/T9 indices and sha256 in `MANIFEST.json` at pre-registration; the mini's H5 re-run |
| RGB | as above | ≈ 25 MB | `test_checkanswer_matches_rgb_in_place`; the mini re-score of RGB |
| Kolibri tokenizer files | Aleph-Alpha/Kolibri-1-BF16 @ `7a8f290e`: `tokenizer.json`, `tokenizer_config.json`, `config.json`, `generation_config.json` (a copy of the scratch `kolibri_tok/` is acceptable if its sha256 match) | ≈ 20 MB | `EXP036_TOK` for the tests and the gate texts |
| mlx-community peer tokenizer and template files | the six peer repos at their assets.json revisions: `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`, `config.json` | ≈ 60 MB | `test_chat.py`, `test_reasoning_split.py`, H5 re-run, parity check |
| Upstream peer tokenizer and template files | google/gemma-4-26B-A4B-it @ `20da991a`; Qwen3.6-35B-A3B and Qwen3.8-27B base repos at pins chosen by the main session | < 30 MB | byte parity; `runner/templates/*.jinja` committed (Apache-2.0, NOTICE) |
| eval-framework v0.14.2 helpers and NOTICE | Aleph-Alpha-Research/eval-framework tag v0.14.2: `task_style.py`, `utils.py`, any NOTICE | < 1 MB | vendoring |
| Grundgesetz Art. 1–19 | gesetze-im-internet.de/gg (the URL recorded in `MANIFEST.json`) | < 1 MB | T4 (committed; public domain) |
| IFBench_test | allenai/IFBench_test @ `2e8a48de` (already pinned) | 0.1 MB | the IFBench adapter on the mini |
| IFBench checker environment | allenai/IFBench @ `1c40f0c1` + `uv sync` (spacy `en_core_web_sm`, nltk `punkt`, langdetect, emoji, syllapy) → `ifbench-venv` | ≈ 0.5 GB | `tests_ifbench/` before the push; H4 scoring |
| aleph-alpha-inference v1.0.0 | already on the mini | — | port derivation, tests, NOTICE |

If Andrei declines the FineWeb-2 fetch on the mini, RUNBOOK step 7 builds T5, T6 and T9 on the mbp and the gate record carries their indices and sha256. If he declines the RGB fetch on the mini, `test_checkanswer_matches_rgb_in_place` runs on the mbp at RUNBOOK step 3 instead.
