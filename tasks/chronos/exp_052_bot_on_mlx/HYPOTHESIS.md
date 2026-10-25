# exp_052: the live bot's answering model on MLX against Ollama

**Registered 2026-10-24, before any measured answer.** Andrei asked for it ("let's test if bot on mlx is better") and chose how answers are judged: code checks on every answer, plus his own blind read of 30 pairs. Kit: `kit/`. The inputs are frozen by the hashes at the end.

## Why

exp_051 found that on this Mac mini, MLX reads long prompts 12–24 % faster than Ollama but generates no faster: level at 4K, slower beyond. The registered next step was a separate test on the bot itself, with quality checks, before any switch. This is that test.

The live bot (@CasaSolGuideBot, `scripts/telegram_bot.py`) answers with `gemma4:26b` through Ollama's `/api/chat`:
- its system prompt, the retrieved context and the question;
- temperature 0.3;
- no think flag. Ollama's default for this model is to think, so every live answer is preceded by hundreds to thousands of thinking tokens.

On the two dev questions, a live-style answer took 35–42 s, almost all of it generation.

## What is held fixed: the frozen inputs

- **The questions.** 120 bot messages written by Fable 5.1 (`data/questions.jsonl`) in the bot's seven languages: en 24, es 20, de 16, fr 16, pl 16, ru 14, sv 14. Their written intents:
  - property search 30, area 24, legal and tax 24, process 24, market 18;
  - every language has at least two of every intent.

  The messages contain no personal data.
- **The freeze.** Each message was run once through the bot's own code, as a first message in a fresh session in its language (`kit/freeze52.py`):
  - the branches before routing;
  - the router (`gemma4:e4b` on Ollama, as live);
  - the broad and holiday checks;
  - retrieval, with the empty-property cascade and the double-miss stop.

  Where the bot would call `gemma4:26b`, the call was recorded instead of made. So both arms receive **exactly** the messages the live bot would send.
- **What does not reach the model.** Messages the bot answers without the model (out of scope, broad listing query, holiday redirect, no listings) are recorded with their reason and are outside the comparison.
- **Privacy.** The frozen file holds text retrieved from the bot's collections, so it stays on the mini. Its sha256 is below.

## Arms

| Arm | Backend | Call |
|---|---|---|
| **G** (live) | Ollama 0.40.2, `gemma4:26b` (GGUF Q4_K_M, `777b927e3b6b`, llama.cpp runner), the daemon's production settings (flash attention off, default fp16 KV cache, context 8192) | `/api/chat` with the frozen messages, `{"temperature": 0.3, "seed": 52}`, no think flag: the bot's call plus a fixed seed |
| **M** (candidate) | mlx 0.32.4 and mlx-lm 0.32.2 server on 127.0.0.1:8090, `mlx-community/gemma-4-26B-A4B-it-OptiQ-4bit` (exp_051's build, 5.0 bits per weight, unquantised KV cache) | `/v1/chat/completions` with the same messages, temperature 0.3, `enable_thinking: true`, `max_tokens` 6144; the server returns the thinking apart from the answer, as Ollama does |

**Both arms:**
- **Failure behaviour.** Both keep the bot's: a 120 s timeout, one retry after 3 s, then the bot's error message.
- **Order.** Every question that reaches the model is asked once per arm, in one fixed shuffled order (seed 52), the same for both arms.
- **The router.** `gemma4:e4b` stays loaded in Ollama throughout, as it would if only the answering model moved.
- **Memory.** During M, `gemma4:26b` is unloaded from Ollama.
- **Sequence.** G first, then M, then G again on the first 10 questions as a drift check.
- **Outside requests.** After every question the Ollama log is read. A POST other than the question's own (Hermes and the live bot share this Ollama) marks it contaminated, and it is asked again once. Contaminated answers are left out of the timing and kept in the checks.

## Measures

- **Wait.** The time a user waits for the answer: from the first request to the final reply, retries included. Also recorded: generated tokens, thinking length, and the finish reason.
- **Code checks** (`kit/checks52.py`), from the bot's own rules. An answer is **flagged** if any of these fires:
  - **V1, language.** Not in the question's language. The detector, `kit/langcheck.py`, scored 913 of 916 on the published labelled enquiries.
  - **V2, price.** A euro amount ≥ 1,000 that is in neither the retrieved context nor the question (±1 %). Amounts the model computes count too, read the same way in both arms.
  - **V3, place.** A place from the registered list that is named in the answer but is in neither the context nor the question.
  - **V4, contact.** A phone number or e-mail address.
  - **V5, failure.** The bot's error message, an empty answer, a stop for length, or thinking markup in the answer.
- **Andrei's blind read** (`kit/read52.py`).
  - 30 questions that reached the model: 6 per routed intent, filled at random if an intent has fewer, seed 52.
  - In each pair the two answers are labelled A and B in a random order per pair. The key is kept in a separate file.
  - He marks which answer he would rather the bot send (A, B or =), and whether either answer is wrong. "Wrong" means a false statement or a broken rule.

## Hypotheses

| | Claim | Rule |
|---|---|---|
| **H1** | M is meaningfully faster for the user | M's median wait ≤ 0.90 × G's, on questions clean in both arms |
| **H2** | M breaks the bot's rules no more often | flagged answers: M ≤ G + 3 |
| **H3** | Andrei does not find M worse | pairs where he prefers M ≥ pairs where he prefers G − 2, **and** answers marked wrong: M ≤ G + 1 |

**Decision, registered:** the bot's answering model moves to MLX only if **H1, H2 and H3 all pass**. If H2 or H3 fails, it stays. If only H1 fails, it stays too, because there is no reason to move.

**Descriptive only.**
- p90 wait, and the number of pairs in which M was faster.
- Generated tokens and thinking length per arm.
- Each check by arm, with McNemar on flagged answers.
- Results by intent and by language.
- Questions that never reached the model, with their reasons.
- The drift check (the first 10 G questions again).

## Known differences and limits

- **Weights.** M's weights are 5.0 bits per weight against G's Q4_K_M, so a quality difference can come from the quantisation, not from the runtime.
- **Generation limits.** G runs inside the daemon's 8192-token context, as live. M has a 6,144-token generation cap.
- **Sampling.** Both sample at 0.3, so one answer per arm per question carries sampling noise. Thinking length varies from answer to answer, and it drives the wait.
- **What the checks see.** V2 and V3 cannot tell an invented figure or place from a legitimate one the model knew or computed. They are read as the same check on both arms. V3 does not see Cyrillic place names.
- **The questions are synthetic**, written by Fable. Real enquiries are the later test.

## Frozen inputs

| File | sha256 |
|---|---|
| `data/questions.jsonl` (120 messages) | `3071db37a27a459e7ef521670a7e069e7cfacd39d2a4da742463a05ac3b59701` |
| `data/frozen.jsonl` (private: holds retrieved text) | `3722079799f58118c79a18e5592019049a3fd6007cd4eed0ed2533d3c8c84ac9` |

**The freeze** (2026-10-24, before registration):
- All 120 messages reach the answering model; none is answered without it.
- The routed intents: property search 29, legal and tax 25, area 24, process 24, market 18. The router agrees with the written intent on 115.
- Every message got retrieved context: Home History 29, area and witnessed notes 24, the course collections in the question's language 67.

**Dev before registration.** Two made-up questions (not in the set) ran once through each arm to check the kit:
- G took 35.1 and 42.0 s; M took 45.3 and 40.4 s.
- None of the four answers was flagged.

Nothing from the set was answered before registration.

## Amendment 1 (2026-10-24, after H1 and H2 were scored, before any judging)

H1 failed, so the registered decision is already fixed: the bot stays on Ollama, and H3 cannot change it. Andrei chose to have the quality question answered anyway by **blind Opus judges on all 120 pairs**, in place of his own read of 30. He accepted that the judges see the retrieved context. The judges are Opus 5.5 subagents of this session, so nothing leaves the session, and nothing is published.

- **Blinding.** Each pair gets a new random A/B order (seed 5201), and the key stays in a separate file the judges are told not to open.
- **Batches.** Four judges take 30 pairs each. Each pair carries the question, its language, the bot's rules (its system prompt), the retrieved context, and the two answers.
- **What the judge marks:**
  - which answer the bot should send (A, B or =);
  - whether either answer is **wrong**: a false statement against the context or plain fact, an invented property, price or place, the wrong language, or legal advice presented as binding.

**H3 (Opus), scaled from the registered 30-pair rule (−2 and +1 per 30):** pairs where Opus prefers MLX ≥ pairs where it prefers Ollama − 8, **and** answers marked wrong: MLX ≤ Ollama + 4. A two-sided sign test on the preferences is reported. The verdict is descriptive: it cannot reverse the decision fixed by H1.
