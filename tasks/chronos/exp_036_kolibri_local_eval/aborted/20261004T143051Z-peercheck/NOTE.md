# RUNBOOK step 9: the peer check fails three arms, and two look like kit issues (2026-10-04)

Step 8 is done (`99047b8`): K8 83.1 GB and K4 44.1 GB, quantisation `{64, 8|4, affine}`, router and `expert_bias` fp32, `quantised_head`, and the tokenizer files and `kolibri1.py` byte-identical. Andrei then ran step 9 as written. Its record, `results/peers_20261004T143051Z.json`, is committed with this note.

```text
G8     fail NLL(8) / KL(8‖4) rule failed; greedy flip rate 0.033> 0.02
G4     fail NLL(8) / KL(8‖4) rule failed
Q36-8  fail greedy flip rate 0.033 > 0.02
Q36-4  ok
Q38-8  ok
Q38-4  ok
```

## 1. Gemma 4: near-uniform predictions in the teacher-forced path

| family | NLL/token 8-bit | NLL/token 4-bit | mean bpb (8-bit) | KL(8‖4)/token |
|---|---|---|---|---|
| gemma4 | 10.21 | 10.66 | 3.57 | **4.57** (`kl_ok` false; `nll_ok` true) |
| qwen3_6 | 2.04 | 2.18 | 0.71 | 0.177 |
| qwen3_8 | 1.93 | 1.95 | 0.67 | 0.100 |

About 10 nats/token is close to uniform over Gemma's 262,144-token vocabulary (ln V ≈ 12.5). This holds on every text, T1–T6 (7.2–13.1 nats/token). G8's batched-path noise floor is also large: `floor_kl` 0.276 and `floor_dis` 0.177, against 0.009 and 0.043 for Q36-8.

**The weights look fine.** In the mbp preflight smoke test on 2026-10-03, `mlx_lm.generate` on the 4-bit build (400-token run) produced a German answer after its thinking channel. That build was at revision `0d77464e`, the one checked here.

**Ruled out (read from the code, not run):**
- A missing BOS. `bench/kl_8v4.py` prefixes `tokenizer_config.bos_token`, which is `<bos>` for Gemma 4.
- A missing final-logit softcap. mlx_lm 0.31.3 `gemma4.Model.__call__` returns `language_model(...)`, which applies `logit_softcap(30.0)` after the tied `as_linear` head. So `model(x, cache=cache)[0]` in `kl_8v4.py:215` returns softcapped `[T, V]` logits.

**Left to check (needs the model loaded, so the main session's diagnosis):**
- The cache that `kl_8v4.py` passes for Gemma 4's mix of sliding-window and global layers.
- The strict text-only load path for `Gemma4ForConditionalGeneration` (`model_type` gemma4, `text_config.model_type` gemma4_text), including any `per_layer_inputs`.
- Any chunking or offset handling specific to this architecture.

## 2. Q36-8: the verdict does not follow the documented rule

- Q36-8's only problem is the greedy answer-flip test: **1 flip in n = 30** (rate 0.033). With `FLIP_MAX = 0.02` and `FLIP_ITEMS = 30`, any single flip fails.
- The `tools/peer_check.py` docstring (lines 33–39) and RUNBOOK step 9 treat the batched-path parity and the greedy flip (B = 8 vs B = 1) as the G8/Q36-8 batched checks, with "`B=1` (only the batched path failed …)".
- `_verdict()` instead returns `fail` whenever `problems` is non-empty, and the flip failure is filed under `problems`. Only `batched_path.ok is False` yields `B=1`.
- So under the documented rule, Q36-8 is `B=1`, not a drop. G8's flip is also 1 in 30.

Per the RUNBOOK, the mbp stops here. Before a peer-drop amendment, the main session decides whether these are kit fixes followed by a re-run of the peer check. Whatever a failed G4 or a dropped Gemma family means for H1 or H8 is Andrei's decision.
