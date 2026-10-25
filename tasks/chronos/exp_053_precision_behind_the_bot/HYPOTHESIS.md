# exp_053: higher-precision gemma builds behind the live bot

**Registered 2026-10-24, before any measured answer.** Andrei asked for it ("register higher precision ollama test, make 3 bot fixes, and let's run 53"). Judging is as in exp_052 Amendment 1, which Andrei chose: blind Opus judges who see the retrieved context, inside this session, with nothing published. Kit: `kit/`. The inputs are frozen by the hashes at the end.

## Why

In exp_052, blind Opus judges preferred the bot's answers from MLX over the live Ollama build: 39 to 22, with 59 ties (sign test p 0.04), and 7 answers wrong against 13. MLX was 10 % slower.

The MLX build is mixed precision at 5.0 bits per weight, against the live Q4_K_M, so the gain may come from the weights, not the runtime. If it does, the bot can keep Ollama and gain the quality.

## The bot under test

The live bot after the three fixes of 2026-10-24 (CasaSol commit in "Frozen inputs"):
1. The answer prompt says the bot has seen nothing itself: a "Community witness report" is a community member's, never "I saw".
2. CasaSol's own fact-checked notes on non-resident tax (IRNR, not IRPF) and the residential rental deposit (LAU art. 36) are added beside the course chunks in all seven language collections.
3. The answer prompt says that when the context does not answer, the bot should say so, and give the general rule it is sure of for law, tax and process questions, labelled as general information. It must never do this for properties, prices or places.

All arms run on the fixed bot, so the fixes are held constant across arms.

## Frozen inputs: what the bot would send

- **The questions.** 120 **fresh** messages written by Fable 5.1 (`data/questions.jsonl`):
  - the same spread as exp_052: en 24, es 20, de 16, fr 16, pl 16, ru 14, sv 14; property search 30, area 24, legal and tax 24, process 24, market 18;
  - none repeats or paraphrases an exp_052 message (similarity below 0.6 against all 120).
  
  So the fixes, which were motivated by exp_052's answers, are not judged on the questions they came from.
- **The freeze.** Each message goes once through the fixed bot's own code (`kit/freeze53.py`, as in exp_052): routing, retrieval, then the exact messages `generate_response` builds.
- **Privacy.** The frozen file holds retrieved text and stays on the mini.

## Arms (all Ollama 0.40.2, the bot's exact call)

| Arm | Build | Note |
|---|---|---|
| **G4** | `gemma4:26b` (GGUF Q4_K_M, `777b927e3b6b`) | the live build |
| **GQ** | `gemma4:26b-a4b-it-qat` (16 GB) | Google's quantisation-aware-trained 4-bit: the deployable candidate at 4-bit size |
| **G8** | `gemma4:26b-a4b-it-q8_0` (28 GB) | 8-bit: tests whether precision is the cause |

**The call.**
- `/api/chat` with the frozen messages, `{"temperature": 0.3, "seed": 53}`, no think flag (Ollama's default for this model: thinking on), the daemon's context (8192).
- The bot's failure behaviour: a 120 s timeout, one retry after 3 s, then its error message.

**The run.**
- **Order.** One fixed shuffled order (seed 53) in every arm. Arms run G4, GQ, G8, then G4 again on the first 10 questions as a drift check.
- **Memory.** Only the build under test and the router (`gemma4:e4b`) are loaded.
- **Outside requests.** After every question the Ollama log is read. A POST that is not the question's own marks it contaminated, and it is asked again once. Contaminated answers are left out of the timing.

## Measures

- **Wait.** What the user waits, retries included.
- **Code checks** (`kit/checks53.py`): exp_052's five, with the price check fixed before registration. It now reads "mil", "tys", "тыс" and ranges ("400-450k", "entre 300 y 400 mil"), and it does not read "121 m" as millions. It still counts amounts the model computes, the same way in every arm.
- **Blind Opus judges** (`kit/judge53.py`). Two comparisons against the live build, G8 against G4 and GQ against G4, each with 120 pairs.
  - Each comparison has its own random A/B order per pair. The 240 pairs are shuffled together into 8 batches of 30, one Opus 5.5 judge each.
  - A judge sees the question, the bot's rules, what the bot retrieved and the two answers. It does not see which build wrote which, or which comparison a pair belongs to.
  - It marks which answer it would rather the bot send (A, B or =), and whether either answer is wrong: false against the context or plain fact, an invented property, price or place, the wrong language, or legal advice presented as binding.

## Hypotheses

| | Claim | Rule |
|---|---|---|
| **H1** | Higher precision writes better answers | in G8 against G4: judges prefer G8 more often, sign test p < 0.05, **and** answers wrong G8 ≤ G4 |
| **H2** | QAT writes better answers at 4-bit size | in GQ against G4: judges prefer GQ more often, sign test p < 0.05, **and** answers wrong GQ ≤ G4 |
| **H3** | Speed cost, per candidate | median wait ≤ 1.10 × G4's |
| **H4** | Code checks, per candidate | flagged answers ≤ G4's + 3 |

**Decision, registered:**
- A candidate is adopted in the bot only if its quality hypothesis (H1 or H2), H3 and H4 all pass.
- If both candidates qualify, the faster one is adopted.
- If neither qualifies, the bot keeps `gemma4:26b` Q4_K_M.
- If H1 passes and G8 fails H3, precision is shown to be the cause, but too slow to deploy. Andrei decides.

**Descriptive only.**
- p90 wait and generated tokens per arm.
- Each check by arm, with McNemar on flagged answers.
- Preferences by intent.
- How often answers still claim a visit (the fix-1 error).
- How often answers use the new notes on IRNR and the deposit, where asked.
- The drift check.

## Known differences and limits

- **Sampling.** One sampled answer per arm per question, at temperature 0.3, and one judge per pair.
- **QAT's lineage.** QAT is not just a different quantiser: its weights were further trained by Google, so a gain from it can come from that training.
- **The questions.** They are synthetic and written by Fable; real enquiries come later.

## Frozen inputs

| Input | Identifier |
|---|---|
| `data/questions.jsonl` (120 messages) | sha256 `f53ea3b7ee7aac1a40b581682de5384e1c43b23c0a97600340ebfb870a906000` |
| `data/frozen.jsonl` (private: holds retrieved text) | sha256 `870d15e42c0a90ba3be42a7ba583a291e0aa1c885dfeed425cd79a9ad0d23111` |
| The bot frozen through | CasaSol `f37b8a7` (the three fixes, deployed 2026-10-24) |
| Builds | `gemma4:26b` `777b927e3b6b`; `gemma4:26b-a4b-it-qat` (Q4_0, QAT); `gemma4:26b-a4b-it-q8_0`; all pulled 2026-10-24 except the live one |

**The freeze** (2026-10-24, before registration):
- **Reach.** 118 of 120 messages reach the answering model. The router sent 2 out of scope, and those are outside the comparison.
- **Routing.** Routed intents: property search 29, area 21, legal and tax 32, process 17, market 19. The router agrees with the written intent on 102.
- **Prompt.** Every frozen request carries the new prompt rules.
- **Notes.** A CasaSol note is among the retrieved chunks for 29 of the 118 questions. That is more often than their two topics alone would explain, so whether the notes crowd out course chunks is reported, descriptively.

**Dev before registration.** None on this set. The kit is exp_052's: `arms53.py` changes only the model per arm, and the price parser was unit-tested on made-up strings.
