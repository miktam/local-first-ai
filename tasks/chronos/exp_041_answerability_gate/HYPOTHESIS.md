# exp_041 pre-registration: an answerability gate

*Written 2026-10-17 on the mini, before any dev run beyond a 10-row plumbing check and before any gate has seen a v3 row. **Signed by Andrei: 2026-10-17 16:09 CEST** ("sign"); unchanged from the draft committed in d4e7778. Design: [DESIGN.md](DESIGN.md) rev 1. Smoke: [SMOKE.md](SMOKE.md). Code: casasol `coapi_voice/rl41/` (d206e17).*

## The question

Can a separate local categorizer, asked "can this material answer this question?", make the system say "not in the material" when the rule is missing, without blocking answerable questions?

## Arms (all on llama.cpp b11711, commit 9d1e036, one at a time on the mini)

| Gate | Model (sha256) | How |
|---|---|---|
| G1 | Qwen3-4B imatrix Q4 (`38623d2a`), exp_040's B0i weights | chat, thinking off, P(yes) from the first token |
| G2 | Kev-4B Q8_0 (`7c2ebed9`) | `/v1/systemone`, one `noul` |
| G3 | Clef-Flash 9B Q8_0 (`55c222df`) | same |
| G4 | Clef 27B Q8_0 (`9fe953f0`) | same |

- **Prompts:** fixed in `rl41/gate41.py`.
- **Input:** the material and the question exactly as the voice saw them: the same retrieval, withheld chunks removed from the index, and the options line on categorical and yes/no rows.

## Procedure

1. **Dev runs:** all four gates on `data/gate41/dev.jsonl` (615 rows, sha `f0153ba9`).
   - 288 exp_040 abstention variants, labelled by the blind Fable reader with doubt-flagged verdicts dropped: 163 `not_answerable`, 125 `answerable`.
   - 87 off-topic prompts: `not_answerable`.
   - 240 bank prompts whose source was retrieved, 80 per language: `answerable`.
2. **Thresholds (`rl41/select41.py`):** a gate blocks a row iff P(answerable) < τ.
   - τ = the largest value at which dev false abstention ≤ 0.03.
   - **Primary gate** = the highest dev abstention on `not_answerable` rows at its τ; ties go to the faster gate.
   - `thresholds.json` is committed **before** any v3 call. The runner refuses v3 without it, and refuses from uncommitted code.
3. **v3:** one pass per gate (150 rows; the runner refuses a second).
   - Gated system = gate + exp_040's B0i v3 answers: a blocked row becomes "not in the material"; every other row keeps B0i's verdict. No voice answer is regenerated.
4. **Scoring (`rl41/score41.py`)** with exp_040's paired statistics (stratified bootstrap, seed 40; exact one-sided McNemar).

## Hypotheses (primary gate; the other gates are reported)

- **H1, abstention:** on v3's 38 abstention rows, abstention rises ≥ 0.20 over B0i (0.447 → ≥ 0.647) **and** McNemar p < 0.05.
- **H2, no over-blocking:** false abstention on v3's 112 answerable rows ≤ 0.05. B0i is at 0.027 (3 rows), so the gate may add at most 2.
- **H3, no net loss:** exact match over the 150 rows, gated − B0i, paired bootstrap 95 % lower bound > −0.02.
- **H4, calibration:** expected calibration error of the primary gate's P(answerable) on dev ≤ 0.10.
- **D1 (descriptive):** each gate over A13, the deployed glue, instead of over B0i.

## Verdict map (first that fires)

1. H1 or H2 fails → **No gate earns its place.**
2. H3 fails → **Abstains, but costs answers.**
3. Otherwise → **Gate adopted for the glue**, with H4 and seconds per call reported.

## Known limits, written now

- **The labels are one model's judgement.** Dev labels come from one blind Fable reader per row.
- **H2 rests on very few rows.** 112 answerable rows make it a count of single digits; the interval is reported.
- **Kev-4B's card says English only.** Its PL and ES results are reported separately on dev.
- **G1 is the voice's own model.** If G1 wins, the cheapest deployment is one extra call to the model already served.

---

**Signature:** Andrei, 2026-10-17 16:09 CEST — "sign". The text above is frozen at d4e7778; any change from here is a dated amendment below.
