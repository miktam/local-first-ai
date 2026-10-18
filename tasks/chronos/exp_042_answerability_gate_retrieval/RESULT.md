# exp_042 result: the answerability gate with fixed retrieval

*Scored 2026-10-18 on the mini, under HYPOTHESIS.md (signed 2026-10-18 04:31 CEST, text = c142c76). Thresholds were frozen from dev before any v3 gate call (casasol 4f61704). The voice and each gate made one v3 pass on the stored rendered rows (`v3_rendered`, `6e344e12`). Files: [results/v3/](results/v3/), [results/dev/](results/dev/).*

## Verdict: **Gate adopted for the glue (on the mini)**

H1, H2 and H3 pass for the primary gate, G4 (Clef 27B, τ 0.602). H4 fails and is reported; it is not part of the verdict map.

| v3 (150 rows), today's retrieval | S0 (untrained voice) | S0 + G4 Clef 27B | Rule | |
|---|---|---|---|---|
| **H1:** abstains on the 38 abstention rows | 0.474 | **0.921** | rise ≥ +0.20, p < 0.05 | +0.447, CI [+0.29, +0.61], p < 0.001: **pass** |
| **H2:** false abstention on the 101 answerable rows (source in material) | 0 / 101 | **0 / 101** | ≤ 0.05 | **pass** |
| **H3:** exact match | 0.753 | **0.833** | 95 % lower bound > −0.02 | +0.080, CI [+0.033, +0.120]: **pass** |
| **H4:** dev ECE | — | 0.174 | ≤ 0.10 | **fail** |

## G2 Kev-4B (reported)

| | S0 + G2 Kev-4B |
|---|---|
| Abstention | 0.921 (+0.447) |
| False abstention | 1 / 101 |
| Exact | 0.820 (+0.067, CI [+0.020, +0.113]) |
| Dev ECE | 0.193 |
| Seconds per question | ~2, against ~12 for Clef 27B |

## The 11 source-lost rows (reported apart, as registered)

These rows have a source in the guide, but retrieval did not return it (10 ES, 1 PL).
- **G4 refused 8 of them.** S0 had 5 of those 8 right, from memory. So the gate refuses answers that happen to be right but are not grounded in the material, which is what the glue's "never fill the gap from memory" asks for.
- Exact match counts these refusals as wrong, and still rises +0.080.

## Compared with exp_041 (descriptive; different retrieval and labels)

| | exp_041 (old retrieval) | exp_042 (fixed retrieval) |
|---|---|---|
| answerable rows whose source was lost | 22 / 112 | 11 / 112 |
| S0 exact | 0.707 | 0.753 |
| S0 + Clef 27B exact | 0.820 | 0.833 |
| Clef 27B false abstention | 12 / 112 (all source-lost) | 0 / 101 (source in material) |

The retrieval fix alone moved the untrained voice from 0.707 to 0.753 exact on the same 150 questions. This is descriptive: the rows are the same, but this is not a registered comparison.

## Known limits (as written before the run, and one more)

- **v3 is reused.** Every exp_041 gate had a pass on it, so this is weaker evidence than a fresh set.
- **Labels are code-only:** "answerable" means the source chunk is in the material.
- **The probabilities are poorly calibrated** (H4: ECE 0.17–0.19). The threshold works; the raw probability should not be read as a confidence.
- **Latency:** Clef 27B adds about 12 s per question on the mini; Kev-4B adds about 2 s for nearly the same result (1 false abstention, exact +0.067).

## What adoption needs (not done; Andrei's call)

- **Wire the gate into `glue.answer`.** The gate needs to run before the voice, on the same material; a block returns the "not in the material" reply.
- **Choose the gate:**
  - Clef 27B is the registered winner: 0 false abstentions, about 12 s, and 29 GB that stays loaded.
  - Kev-4B has 1 false abstention, about 2 s and 4.5 GB.
- **Serve it:** a llama.cpp b11711 `llama-server` beside Ollama on the mini.

---

*Added 2026-10-18, after publication.*

## After adoption (2026-10-18): Kev-4B wired, per-language thresholds

These are configuration changes, not a new measurement.
- **Wired:** Kev-4B is wired into `glue.answer` (casasol 3b5b2b7) and served by `ai.casasol.gate-kev4b` on :18442, which Andrei loaded from the miktam account. If the gate is unreachable, the glue answers as before (fail open).
- **Per-language thresholds (Andrei):** the same rule per language on this experiment's dev probabilities. EN 0.600, PL 0.448, ES 0.576.
  - Dev: PL wrong refusals 4 → 2 of 80, PL correct refusals 0.730 → 0.607; EN 0.719 → 0.744, wrong 1 → 2.
  - Each value rests on 80 answerable rows, so it is provisional.
- **Live smoke, 6 questions (anecdotes):**
  - EN boiler repair: passes, answered right.
  - EN obscure fine: blocked, right.
  - ES boiler: blocked. Spanish retrieval missed the chunk with the rule.
  - ES inheritance deadline: passes on a near-miss rule (the transfer-tax deadline) and is answered wrong ("2 months").
  - PL resale transfer tax: blocked at the global threshold; under the PL threshold it passes and is answered wrong (1.2 %, the AJD rate; the material says ITP 7 %), with the question echoed first.
- **Lesson:** the gate judges whether the material can answer. Whether the 4B voice answers correctly is a separate, still-open weakness.

## SO WHAT (2026-10-18)

**Adopted (true today):**
- The COAPI Voice glue runs the Kev-4B gate (casasol 3b5b2b7, per-language thresholds 671eb3d), served by `ai.casasol.gate-kev4b` on the mini.
- The retrieval fixes this experiment depended on are in the glue: Spanish via translation, two chunks from the leading tema, budget 1,350.

**Not adopted yet: CasaSolGuide.** The live bot (`scripts/telegram_bot.py`) does not use the glue. Its legal and process answers come from its own chromadb collection (`coapi_knowledge_{lang}`, top 3) and gemma4:26b, in up to 7 languages. The gate's thresholds were measured on the glue's material, not the bot's, so they do not transfer as they are.

**Cascade: CasaSolGuide runs the gate in SHADOW MODE since 2026-10-18** (casasol 3b84827, deployed with `deploy.sh bot`, Andrei). For legal, fiscal, process and market answers, Kev-4B scores the bot's own retrieved chunks in a background thread and logs P(answerable) to `logs/gate_shadow.jsonl`. It blocks nothing and stores no text. The first live check shows the glue's thresholds do not transfer: on the bot's material a basic plusvalía question scored 0.33, below EN's 0.600.

**Originally proposed:** put the gate into CasaSolGuide's legal, fiscal, process and market intents.
- First in **shadow mode**: log P(answerable) per answer, block nothing.
- Then set a bot-specific threshold from the logged or replayed traffic, and switch to blocking.

**Seeds for experiments:**
1. **The voice confuses neighbouring rules** even when the material is right: ITP 7 % answered as AJD 1.2 %; one tax's deadline for another's. Two candidates:
   - a deterministic check that the answer's figure sits beside the matching tax or term in the material;
   - a stronger voice behind the same glue and gate.
2. **Spanish retrieval still misses 23 %** of sources, and the gate then refuses those questions: correct about the material, frustrating for the user.
3. **The gate can be fooled by a near-miss rule** (a deadline for another tax). A second `noul` asking "does the material state *this* rule?" is a candidate.

*Added 2026-10-18, later the same day: shadow-mode check.* We replayed 60 guide-answerable questions (20 per language) through CasaSol Guide's own retrieval, which returns three 500-word chunks of the course text. At the glue's thresholds the gate would refuse 29 of 60 (EN 11, PL 11, ES 7); below 0.30, 12; below 0.10, 1. The bot's model (gemma4:26b) also abstains by itself where the untrained 4B did not. The gate therefore stays in shadow mode. Whether it adds anything in front of gemma4:26b, and at what threshold, is the next experiment.
