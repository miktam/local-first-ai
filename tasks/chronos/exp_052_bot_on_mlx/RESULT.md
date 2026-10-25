# exp_052 result: on MLX the bot is slower, and no better on its rules

**Run:** `runs/20261024T122416Z`, 2026-10-24, 12:24–14:39Z, on the Mac mini.
- The run was clean: 120 questions per arm, no contaminated answer, no outside request.
- Drift: the first 10 Ollama questions took a median 34.1 s at the start and 34.1 s at the end.
- `score.json` comes from the registered scorer.

## Verdicts

| | Claim | Result | Verdict |
|---|---|---|---|
| H1 | MLX is ≥ 10 % faster for the user | median wait **30.0 s against 27.3 s**: MLX is 10 % *slower*. p90 55.3 s against 45.1 s. MLX was faster on 45 of 120 questions | **fail** |
| H2 | MLX breaks the bot's rules no more often | flagged answers: 13 against 12 (2 flagged only on Ollama, 3 only on MLX; McNemar p 1.0) | **pass** |
| H3 (Amendment 1: blind Opus judges, all 120 pairs) | Opus does not find MLX worse | prefers **MLX 39**, Ollama 22, no difference 59 (sign test p 0.04); answers marked wrong: **MLX 7, Ollama 13** | **pass**: MLX is *better* |

**The registered decision:** the bot moves to MLX only if all three pass. H1 failed, so **the bot stays on Ollama** by the rule. But the quality result points the other way, which reopens the trade-off (SO WHAT).

## Why MLX was slower

- **Generation time decides the wait.** Every answer thinks first: a median of about 1,200 generated tokens for an answer of about 500 characters. So the wait is set by generation, where exp_051 found MLX no faster than Ollama.
- **MLX was a little slower to generate.** Counting from the user's side, MLX produced 43.0 tokens per second of wait against Ollama's 45.3.
- **It also wrote slightly more:** median 1,254 tokens against 1,224, and answers of 559 characters against 511.
- **Its slow tail is longer.** One MLX answer hit the bot's 120 s timeout and succeeded on the retry, with a 150 s wait. On Ollama, one answer stopped at the 8,192-token context and counts as a failure.

## The code checks

| Check | Ollama | MLX |
|---|---|---|
| V2 price not in context or question | 9 | 9 |
| V3 place not in context or question | 3 | 5 |
| V5 failure | 1 (stop for length) | 0 |
| V1 language, V4 contact | 0 | 0 |

- **The price check over-flags.** Most V2 flags are the same amount flagged in both arms. They come from the checker, not the model:
  - It does not read amounts written as "300 y 400 mil", "280 tys" or the low end of "400-450k".
  - It counts a computed tax (7 % of 300k = 21,000) as a price missing from the context.
  
  The error is identical in both arms. It is noted here, and the registered checker is unchanged.
- **The place flags are mostly places near the one asked about.** Coín, Istán, Benalmádena and Fuengirola were named from the model's own knowledge. Both arms do it.

## The blind Opus judgments (Amendment 1)

Four Opus 5.5 judges each saw 30 pairs, blind and with the retrieved context. After unblinding:

| Routed intent | prefers Ollama | prefers MLX | no difference |
|---|---|---|---|
| area | 2 | **12** | 10 |
| process | 5 | **10** | 9 |
| property search | 1 | 4 | 24 |
| legal and tax | 8 | 8 | 9 |
| market | 6 | 5 | 7 |
| **all** | **22** | **39** | **59** |

**What the judges marked wrong.** 13 Ollama answers against 7 MLX answers.
- **One Ollama error accounts for 4 of its 13.** It presented a community witness report as its own visit ("I was at the entrance", "lors de ma visite"). That breaks the bot's rule to say how each property was seen. MLX made that error 0 times.
- **Ollama's other errors:** a garbled place name ("Estepna"), a claim that the material does not say who pays the plusvalía when it does, and an empty answer, where thinking used up the 8,192-token context.
- **MLX's errors:** "cadastral value" for "cadastral reference value", "housing is inelastic" applied to an investment flat, and IRPF for a non-resident.
- **Errors both made:**
  - non-resident gains taxed under IRPF at 19–30 % instead of IRNR at a flat 19 %;
  - the first-registration procedure presented as the normal purchase steps;
  - a rate range attributed to the ECB;
  - a residential deposit of "one or two months" (legally one).

**Where the ties come from.** The 59 ties are mostly both answers honestly saying nothing matches, often where retrieval returned unrelated text. The judges add that "=" there often means *equally unhelpful*: neither answer gave the general fact the user needed.

**Limits.** One judge per pair, and one sampled answer per arm at temperature 0.3, so part of the gap may be sampling noise. The sign test (p 0.04) treats each pair as independent.

## SO WHAT

- **The trade-off.** MLX answers were judged better (39 against 22) and wrong less often (7 against 13), but users would wait about 3 s longer at the median and 10 s longer at p90. The registered rule keeps the bot on Ollama, and a switch for quality would be Andrei's call, after confirmation. MLX runs a 5.0-bit mixed-precision build against Ollama's Q4_K_M, so **the quality gain may come from the weights, not the runtime**. Seed: a higher-precision GGUF of gemma4:26b in Ollama (Q5_K_M, Q6_K or Q8_0) against today's Q4_K_M, judged the same way, with speed measured. If the gain comes with the weights, the bot keeps Ollama's speed and gains the quality.
- **The bot stays on Ollama for now.**
  - On the bot's real workload, which is short prompts and long thinking, MLX's faster prompt reading (exp_051) is outweighed by slightly slower generation.
  - Users would wait about 3 s longer at the median and 10 s longer at p90.
  - By the code checks, the bot breaks its rules equally often on either runtime. The judges found fewer wrong answers on MLX.
- **The wait is driven by thinking, not by the runtime.** About 1,200 thinking-and-answer tokens per reply puts the median wait at 27 s. If the bot needs to be faster, the lever is how much it thinks, not where it runs. Seed: the live answer call with thinking off, or with a capped thinking budget, against thinking on, judged the same way. exp_050 found thinking made gemma4:26b far stronger at the front desk, so this needs its own quality test.
- **Bot fixes that do not depend on the runtime:**
  - The answer prompt should tell the model that a community report is not the bot's own visit.
  - The retrieval or the course notes should cover IRNR against IRPF for non-residents, and the 1-month residential deposit.
  - The bot should give the general fact when retrieval has nothing.
- **The checker needs work.** V2 should read "mil", "tys" and ranges, and should exempt amounts computed from the context. Fix this before it is reused.
