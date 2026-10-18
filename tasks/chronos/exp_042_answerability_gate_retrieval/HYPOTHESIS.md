# exp_042 pre-registration: the answerability gate with fixed retrieval

*Written 2026-10-18 on the mini, before any exp_042 run. **Signed by Andrei: 2026-10-18 04:31 CEST** ("sign"); unchanged from the draft committed in c142c76. Design: [DESIGN.md](DESIGN.md) rev 1. Code and data: casasol `coapi_voice/rl42/`, `data/gate42/` (b3b29fc).*

## The question

With today's retrieval, does a gate make the system say "not in the material" when the rule is missing, without blocking questions whose rule is in the material?

## Fixed before any run

- **Retrieval** (glue as of casasol a43e26b): k 5, budget 1,350, `LEAD_DEPTH` 2; Spanish questions are searched with their English translation (Unsloth Qwen3-4B, temperature 0).
- **Rendered sets** (byte-identical for the voice and the gates):
  - `dev.jsonl` (sha `d130fd7d`, 610 rows): 240 answerable (bank, source in the material, 80 per language) and 370 not answerable. The 370 are 120 bank rows whose source retrieval lost, the 163 reader-confirmed undecidable variants and 87 off-topic prompts.
  - `v3_rendered.jsonl` (sha `6e344e12`; v3 sealed `2c6cbad1`): 101 answerable, plus 49 not answerable. The 49 are the 38 abstention and out-of-scope rows and 11 rows whose source was lost (10 ES, 1 PL).
- **Labels (code only):** a row is answerable iff it is not an abstention row **and** its source chunk is in the material.
- **Voice S0:** the untrained Qwen3-4B, imatrix Q4 (`qwen3-4b-b11577:q4km-imat`, digest `92bfae6dd5dd`), on the Ollama 0.33.0 side server. It runs on the stored system and user turns with `run_eval.OPTIONS` (greedy), thinking off, scored by reward40 rev 3.
- **Gates (no selection):** G4 Clef 27B (primary) and G2 Kev-4B, on llama.cpp b11711 (9d1e036), with exp_041's prompts (`rl41/gate41.py`).
- **Thresholds:** a gate blocks iff P(answerable) < τ, where τ is the largest value with dev false abstention ≤ 0.03. They are set from dev, committed **before** any v3 gate call; the runner refuses otherwise.
- **Gated system:** a blocked row becomes "not in the material", which is right on the 38 abstention rows and wrong elsewhere. Every other row keeps S0's verdict.

## Hypotheses (G4; G2 reported)

- **H1:** abstention on the 38 abstention rows rises ≥ 0.20 over S0, exact one-sided McNemar p < 0.05.
- **H2:** false abstention on the 101 answerable rows ≤ 0.05.
- **H3:** exact match over all 150 rows, gated − S0, paired bootstrap 95 % lower bound > −0.02 (strata language × kind, seed 40).
- **H4:** dev ECE ≤ 0.10.
- **Reported apart:** how many of the 11 source-lost rows the gate blocks, and how many of those S0 had right (answers from memory).

## Verdict map (first that fires)

1. H1 or H2 fails → **No gate earns its place.**
2. H3 fails → **Abstains, but costs answers.**
3. Otherwise → **Gate adopted for the glue (on the mini)**, with H4 and seconds per call reported.

## Known limits, written now

- **v3 is reused** (Andrei's decision 1). Every exp_041 gate had one pass on it, and this session read exp_041's analysis of its blocked rows. A pass here is weaker evidence than one on a fresh set.
- **Labels are code-only.** A rule repeated in another retrieved chunk can make a source-lost row answerable in fact; such rows count as not answerable here.
- **H2 rests on small counts:** 101 rows, so at most 5 false abstentions.

---

**Signature:** Andrei, 2026-10-18 04:31 CEST — "sign". The text above is frozen at c142c76; any change from here is a dated amendment below.
