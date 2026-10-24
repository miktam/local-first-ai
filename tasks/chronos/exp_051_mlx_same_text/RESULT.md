# exp_051 result: MLX reads long prompts faster than Ollama, and generates slower

**Run:** `runs/20261024T085524Z`, 2026-10-24 08:55–10:07Z, on the Mac mini.
- The run passed the protocol:
  - 99 calls, all clean, with no outside Ollama request during any call.
  - Every call ended on its own stop token.
  - Drift: Ollama at 15k was 1.683 ms/tok first and 1.648 last (−2.1 %, inside the ±5 % check).
- The scorer was registered beforehand (`kit/score51.py`), and its output is in `score.json`.

## Verdicts

| | Claim | Result | Verdict |
|---|---|---|---|
| H1 | MLX has no prefill cliff up to 40K on the fixtures | 40k is 1.09 × its 15k value (cliff rule: 2 ×) | **pass** |
| H2 | MLX prefill no slower than Ollama, fixtures | MLX ÷ Ollama 0.884 at 15k, 0.781 at 35k | **pass** |
| H3 | MLX generates ≥ 10 % faster at 4k | 0.991: level | **fail** |
| H4 | Today's MLX reproduces exp_011 on exp_011's input | 1.448 against 1.650 at 15k (−12 %), 1.548 against 2.161 at 35k (−28 %): faster than June | **fail** |
| H5 | The repeated sentence did not change MLX prefill speed | at matched length −0.01 % (15k) and +0.3 % (35k) | **pass** |
| H6 | MLX prefill no slower than Ollama, natural text | 0.825 at 15k, 0.782 at 35k | **pass** |

## The numbers (median of 3; ms per prompt token, generated tokens per second)

| Size | Ollama prefill | MLX prefill | MLX ÷ Ollama | Ollama gen | MLX gen | MLX ÷ Ollama |
|---|---|---|---|---|---|---|
| 4k | 1.454 | 1.248 | 0.86 | 47.5 | 47.1 | 0.99 |
| 8k | 1.511 | 1.321 | 0.87 | 43.2 | 41.4 | 0.96 |
| 15k | 1.683 | 1.488 | 0.88 | 38.3 | 34.0 | 0.89 |
| 20k | 1.744 | 1.469 | 0.84 | 34.9 | 30.5 | 0.87 |
| 25k | 1.828 | 1.506 | 0.82 | 33.1 | 27.5 | 0.83 |
| 30k | 1.914 | 1.530 | 0.80 | 32.2 | 25.1 | 0.78 |
| 35k | 2.039 | 1.592 | 0.78 | 30.6 | 25.2 | 0.82 |
| 40k | 2.141 | 1.628 | 0.76 | 28.6 | 23.4 | 0.82 |

**Natural text** (916 published synthetic enquiries, none repeated):
- Prefill: MLX ÷ Ollama is 0.85 at 4k, 0.83 at 15k, 0.83 at 25k and 0.78 at 35k.
- Generation: 1.01, 0.91, 0.82 and 0.82.
- Ollama on natural text is within 2.5 % of Ollama on the fixtures at every size.

**Neither runtime has a cliff.** Ollama's 40k is 1.27 × its 15k. MLX peaked at 19.3–20.2 GB.

## What it says

1. **exp_011's "no cliff on MLX" holds on real text.** It also holds on the fixtures and on non-repeating enquiries.
2. **The repeated sentence did not distort MLX's speed** (H5). Matched token for token, the "quick brown fox" runs at the same per-token speed as the fixtures. What the June bug changed was the prompt length at each size label, not the speed. So exp_011's side-by-side numbers compared different lengths, but its MLX curve itself was not an artefact of the text.
3. **Both runtimes are faster than in June.**
   - MLX on exp_011's own input is 12–28 % faster than June (H4 fails in the good direction).
   - Ollama at 15k went from 1.774 (exp_008, June) to 1.683, and at 35k from 2.241 to 2.039.
   - The record cannot say which change did it. Both runtimes and macOS changed since June, and June's MLX version was never recorded.
4. **June's generation lead is gone** (H3). Today MLX generates at the same speed as Ollama at 4k, and 4–22 % slower from 8k up. So June's 52 against 42 t/s at 4k has no explanation in the record and does not reproduce.

## SO WHAT

- **For CasaSol's answering model:** moving from Ollama to an MLX server would save prompt-reading time but cost generation time at long context.
  - At a typical 8k prompt with a 300-token answer, prefill saves about 1.5 s (8,000 × 0.19 ms) and generation costs about 0.3 s, so a net 1.2 s per answer.
  - The weights also differ: the MLX build is 5.0 bits per weight against Q4_K_M, so answer quality would need its own test.
  - Per the registered decision table (H1, H2 and H6 pass), the move is now a candidate, worth its own registered test with quality checks. It is not a clear win, because generation got slower.
- **For the record:** exp_011's post already carries the 24 October correction. Its box says the 4k generation gap is not explained. exp_051 adds that the gap does not reproduce today, and that the prefill comparison on the same text favours MLX by 12–24 %, not 3 %. Pre-registered rule: H5 passed, so no further note on exp_011's speed numbers is required; a pointer to exp_051 is optional.
- **Not tested:** Ollama 0.40.2 also lists a `ggml` runner for `gemma4:26b` besides llama.cpp. It is a third candidate, not measured here.

## Limits

- **Generation length.** Generation speed was measured on one-sentence summaries: 23–54 tokens.
- **Weights.** The two runtimes ran different quantisations of the same model.
- **Timing boundary.** MLX's prefill timing includes the first token's sampling.
- **Scope.** One machine, one day, three repetitions per size.
