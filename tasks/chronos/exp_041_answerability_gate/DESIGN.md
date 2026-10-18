# exp_041: an answerability gate — can a local categorizer tell when the material cannot answer?

*Rev 1, 2026-10-17: decisions 1–4 answered as recommended (Andrei: "Go"). Rev 0 → rev 1: G4 is now Clef 27B, after the smoke step ([SMOKE.md](SMOKE.md)). Andrei decided: local models only (no Jev API). No dev or v3 run yet.*

## 1. Why

exp_040 left one failure that training did not fix. When the question is on topic but its rule is missing from the retrieved material, the model answers anyway:

| exp_040 arm (v3) | Abstains on the 38 abstention rows | False abstention on the 112 answerable rows |
|---|---|---|
| B0i (untrained, prompt mode) | 0.447 | 0.027 |
| R3 (trained) | 0.579 | 0.027 |
| A13 (deployed glue) | 0.605 | 0.080 |

The hard case is right topic, wrong branch: the off-topic rows are mostly declined, but the withheld-source rows are not (B0i 0.30). GRPO could not help where the base never abstains (79 of 163 training variants were 0/6).

**The idea:** take the decision away from the voice. Before the voice answers, a separate **categorizer** looks at material + question and returns one label with a probability. If the label is `not_answerable`, the glue answers "not in the material". This is the decision-model row in the library pattern (`library/patterns/facts-in-context-form-in-weights.md`, third row): closed-set judgments about an input. That class is reported strong on classification and verification, and overconfident on reasoning. Our case sits on that boundary, so it is measured, not assumed.

## 2. Arms: the gate model (all local, on the mini)

| Arm | Gate | Size | Why |
|---|---|---|---|
| G0 | none (B0i as scored in exp_040) | — | the bar |
| G1 | **Qwen3-4B prompted** (`qwen3-4b-b11577:q4km-imat`), one yes/no call, probability from the label token | 4B, already served | what a decision call costs with no new model |
| G2 | **Kev-4B** (`ggml-org/Kev-4B-GGUF`) | 4B | a Jev-style decision model at our base's size |
| G3 | **Clef-Flash** (Cloudflare, from Qwen3.5-9B), Q8_0 | 9B | a mid-size decision model |
| G4 | **Clef 27B** (Cloudflare), Q8_0 | 27B (28.8 GB) | the largest decision model llama.cpp serves natively; replaces pplx-decider (needs a third-party bridge) and Vega-27B (needs a fork) |

**The system scored in each arm** is gate + B0i, where B0i is the untrained voice in prompt mode, the same answers exp_040 scored:
- where the gate says `not_answerable` (probability ≥ the threshold θ), the row's answer becomes "not in the material";
- elsewhere B0i's exp_040 answer stands.

So no voice answer is regenerated, and the gate is the only variable. **D1 (descriptive):** the best gate in place of the glue's own scope gate in A13.

All four run on llama.cpp b11711 `llama-server`: G2–G4 through `/v1/systemone` (one `noul` question), G1 through chat completions with P(yes) from the first token. Smoke results, dropped models and reasons are in [SMOKE.md](SMOKE.md).

## 3. Data (none of it new; no Fable needed)

**Dev, for prompt wording and the threshold θ.**
- From exp_040's abstention variants, 330 rows, each labelled by a blind Fable reader:
  - `not_answerable`: 163 confirmed undecidable.
  - `answerable`: 161 the reader could still decide from the remaining material.
  - 6 dropped (doubt).
- From the training prompts (`data/rl40/prompts`), bank rows whose source chunk was retrieved: `answerable` by construction, about 1,500 to sample from.
- The 87 off-topic prompts: `not_answerable`.

**Test, used once per arm:** v3's 150 rows (38 `not_answerable`, 112 `answerable`). v3 was built blind to all of this and has not been used for development.

## 4. Threshold rule (fixed before v3)

- On dev, each gate's θ is the lowest probability at which its false-abstention rate on dev `answerable` rows is ≤ 3 %. That is stricter than the test bar, to leave room.
- θ and the prompt are frozen per arm, then v3 is run once.

## 5. Hypotheses (v3)

- **H1 (primary):** for the best pre-registered gate, chosen on dev before v3, gate + B0i raises abstention on the 38 rows by ≥ 0.20 over B0i (0.447 → ≥ 0.647), paired McNemar p < 0.05.
- **H2 (guard):** false abstention on the 112 answerable rows ≤ 0.05. At most about 3 more than B0i's 3.
- **H3 (net):** exact match over all 150 rows does not fall (paired bootstrap lower bound > −0.02).
- **H4 (calibration):** expected calibration error of the gate's probability on dev ≤ 0.10. Overconfidence is the reported weakness of this model class.
- **D1:** the gate in place of A13's scope gate; descriptive.

**Verdict map:**
1. No gate meets H1 + H2 → **No gate earns its place.**
2. H1 + H2 pass, H3 fails → **Abstains, but costs answers.**
3. H1 + H2 + H3 pass → **Gate adopted for the glue**, with H4 and the latency per call reported.

## 6. Cost

| Stage | Time | Model use |
|---|---|---|
| Smoke and serving for G2–G4 (downloads ~4 + 6 + 17 GB) | ~2 h, session | none |
| Dev runs: ~1,000 rows × 4 gates | ~1–2 h on the mini, Andrei starts | local only |
| v3 runs: 150 × 4 | ~20 min | local only |
| Scoring | minutes | code only |

## 7. Decisions for Andrei

1. **Arms:** G1–G4 as above, or drop G4 (the 17 GB download, slower)? *Recommendation: keep it; it's the ceiling for "local".*
2. **Primary system:** gate + B0i (the gate is the only variable), with the glue variant as D1. *Recommendation: yes.*
3. **Dev labels:** the Fable reader's verdicts from exp_040 stand as ground truth, with no new Fable pass. *Recommendation: yes. They were blind and doubt-filtered.*
4. **False-abstention bar:** ≤ 0.05 on v3, ≤ 0.03 on dev for θ. *Recommendation: as written.*
