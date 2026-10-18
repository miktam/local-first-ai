# exp_041 smoke step (2026-10-17): which gates run locally

*Plumbing only. Eight rows per gate, from exp_040's training-side abstention variants: EN and PL, 2 judged undecidable and 2 decidable by the blind Fable reader each. No v3 row was touched. These numbers are not results.*

## Runtime

- **llama.cpp b11711** (commit `9d1e036`, 2026-10-17, the latest release on the day): only `llama-server` built, Metal, in `~/models/llama.cpp-b11711`.
  - `/v1/systemone` was merged on 2026-10-02 (PR 29818).
  - b11577 stays the frozen quantiser for exp_040.
- Server flags for every decision model: `-c 16384 -b 8192 -ub 8192`. Clef needs the whole prompt in one micro-batch.
- Gate question for every decision model: one `noul`:
  - instructions: "Does the state contain the information needed to answer this question? Question: …";
  - `true` described as "the state states the rule or figure that decides the answer";
  - `false` described as "the state does not settle it; answering would need facts not in the state";
  - state = the retrieved material.

## Candidates

| Arm | Model | File (sha256) | License | Status |
|---|---|---|---|---|
| G1 | Qwen3-4B, imatrix Q4 (exp_040's B0i GGUF) | `38623d2a…` | Apache-2.0 | **runs.** Text model on the same llama-server; P(yes) from the first token's logprobs, thinking off |
| G2 | Kev-4B (`ggml-org/Kev-4B-GGUF`) | Q8_0 `7c2ebed9…` | Apache-2.0 | **runs**, native `kev` decision type. Card: English only; training included missing-fact abstention |
| G3 | Clef-Flash 9B (`bartowski/Cloudflare_clef-flash-GGUF`) | Q8_0 `55c222df…` | Apache-2.0 | **runs**, native `clef` |
| G4 | **Clef 27B** (`bartowski/Cloudflare_clef-GGUF`) | Q8_0 `9fe953f0…` | Apache-2.0 | **runs**, native `clef`. Replaces the rev 0 G4 |
| — | pplx-decider-v1.1-27b | — | Apache-2.0 | dropped: the GGUF needs a third-party compiled bridge (`pplx_bridge.cpp`) |
| — | Decision-2.0-Vega-27B | — | Apache-2.0 | dropped: fails to load in b11711 ("wrong number of tensors; expected 860, got 850"); needs a fork (`Xunzhuo/llama.cpp`, branch `decision2-gguf`). Download deleted |
| — | OpenJev | — | **CC-BY-NC-4.0** | dropped: non-commercial license |

## Smoke values: P(answerable), plumbing only

| Rows | G1 Qwen3-4B | G2 Kev-4B | G3 Clef-Flash | G4 Clef 27B |
|---|---|---|---|---|
| EN undecidable | 0.002, 0.023 | 0.306, 0.362 | 0.233, **0.832** | 0.253, 0.212 |
| EN decidable | 1.000, 1.000 | 0.956, 0.775 | 0.937, 0.940 | 0.970, 0.932 |
| PL undecidable | 0.019, 0.000 | 0.331, 0.366 | 0.577, **0.769** | 0.214, 0.255 |
| PL decidable | 0.995, 1.000 | **0.591**, 0.950 | 0.903, 0.963 | 0.812, 0.972 |
| Seconds per call | 1–2 | 1.6–2.4 | 2.8–3.9 | 9.5–14 |

**Observation to test, not a finding:** asked as a separate question, the untrained Qwen3-4B separates these 8 rows almost with certainty. When it answers the question itself (exp_040, B0i), it abstains on only 0.30 of withheld-source rows. If that holds on the dev set, the failure is not that the model cannot tell. It is that answering and judging are different tasks, and a separate judging call recovers the second.
