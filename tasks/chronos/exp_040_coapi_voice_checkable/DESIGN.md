# exp_040: COAPI Voice — make correctness checkable (design, rev 0)

*Rev 1, 2026-10-15: decisions answered (§12). Drafted 2026-10-15, after exp_039 closed: "no training gain; the Q4 +0.20 was quantisation; form not learned". This is a design only. Nothing is built, nothing has run, no value exists. It becomes `HYPOTHESIS.md` once Andrei has answered §12 and the new test set is sealed.*

**Blindness record.** The session that drafted this read judge notes on exp_039's v1 and v2.1 rows. So it must not write exp_040's training items or its test set. Both are built in fresh sessions (§4–§5).

---

## 0. What exp_039 taught, in one table

| Finding (exp_039) | Evidence | What exp_040 changes |
|---|---|---|
| The reward measured form; GRPO raised form and traded away the one correctness term | run 2 traces: citation 0.09→0.42, arithmetic 0.57→0.07 | a correctness term on **every** item, dominant over form |
| GRPO learns only what the model already does sometimes | hand-off −0.76→−0.89, out-of-scope decline −1.1→−1.2 (no variance within a group of 6) | keep only items the base gets right 1–5 times in 6 (offline dynamic sampling) |
| The model's real error is an inverted or misapplied rule, which code couldn't see | judge notes, exp_035 and exp_039 alike | every item carries a checkable short answer, and half come as **contrast pairs** whose answers flip on one fact |
| A third of the errors were retrieval misses | 17 of 57 in-scope v1 rows | **abstention** items: the source chunk is withheld, and the correct answer is "not in the material" |
| Our Q4_K_M recipe cost the untrained base ~0.2 | Q8 control: base 1.425 at Q8 vs 1.20 at our Q4. A12 (glue over Unsloth's Q4) also scores 1.425, but there glue and quantiser are confounded | **step 0 fixes the quantiser** (importance matrix); all arms are served from one recipe, with Q8 as a robustness arm |
| 40 scored rows cannot see effects below ~0.2 | review | a new sealed **v3 with ≥150 rows**, and a primary metric that needs no judge |

## 1. The question

Does GRPO against a reward that checks the **answer itself** (a short label code can verify) make Qwen3-4B more correct on held-out property-law questions than its untrained base? This is measured by exact match on a fresh sealed set, at a quantisation that does not penalise the base.

**The one variable against exp_039 is the reward.** It moves from form checks to a correctness check, with form terms kept small. Unchanged:
- base snapshot `1cfa9a72`;
- adapter (LoRA r16 on q/k/v/o, 11,796,480 parameters);
- trainer (mlx-lm-lora 5.8.6 with exp_039's five guards);
- retrieval, and the mbp platform.

The training **items** change too, as they must: a correctness reward needs labelled items. That is recorded, not hidden.

## 2. The answer format

Every answer ends with one line the code reads:

| Item type | Final line | Label check |
|---|---|---|
| numeric | `Answer: 26 256 €` | within 0.5 % (exp_039's `_arith_hit`, signed) |
| categorical | `Answer: ITP` / `Answer: 2 months` / `Answer: seller` / `Answer: art. 1454 CC` | normalised match against a closed list of accepted forms, in EN/PL/ES |
| yes/no | `Answer: No` / `Nie` / `No` | exact, by language |
| abstention | `Answer: not in the material` (EN/PL/ES forms) | exact |

The prose above the line is the explanation the user reads. The glue already post-processes, and it can keep the line as a closing "In short: …" or drop it. That choice is decision 4.

**Train = serve:** the system prompt for every arm (base and trained) asks for the final line. So the reference becomes **A13**: the A12 glue plus that one prompt sentence over the untrained base. A13 is scored like everything else.

## 3. Arms

| Arm | Model | Post-processing | Role |
|---|---|---|---|
| B0-i | untrained 4B, **imatrix Q4_K_M** (step 0 recipe) | none | the base, quantised like the trained arm |
| R3 | trained 4B, same imatrix Q4_K_M | none | the test of the reward |
| B0-q8 / R3-q8 | the same two at Q8_0 | none | robustness: the effect must survive quantisation |
| A13 | Unsloth base + glue + final-line prompt | glue | the deployed reference |
| R3+G | R3 + the same glue | glue | the deployable candidate |

## 4. Training items (fresh sessions, blind to v1, v2.1 and v3)

**Source.** The guide, which is public at casasol.ai/guide, chunk by chunk, in the training temas only (Módulos 1–4, M5 T39–41; Módulo 6 never).

**Construction:**
1. **Draft.** A Fable writer reads one chunk and drafts 3–5 questions a buyer would ask. For each it writes a short label (§2 types), the label's accepted forms in EN/PL/ES, and the chunk id it rests on.
   - The writer drafts **labels, not answers to imitate**, so exp_039's no-teacher-text rule holds: the model still never sees a target answer.
   - The label is the matured outcome Coinbase's reward used.
2. **Contrast twin.** For every second item the writer adds a twin that changes one fact so the label flips (resale ↔ new build; buyer under 35 ↔ over; habitual residence ↔ second home).
3. **Adversarial check.** A second Fable instance, blind to the writer, answers each item from the chunk alone. Only items where it reaches the same label are kept. Disagreements go to a third reader, and kept items must reach 2 of 3. Items resting on rules marked provisional in the guide (RDL 29/2026) are dropped.
4. **Arithmetic.** Reuse exp_039's 15 checked templates and extend them with multi-step forms (e.g. ITP plus notary plus registry on one price). Gold is computed in code, as before.
5. **Abstention items.** For about 15 % of items, the retrieved context is built **without** the source chunk (it is removed from retrieval). The label becomes "not in the material". Code can verify this because the source chunk id is known.
6. **Offline dynamic sampling.** The untrained base (imatrix Q4) answers every item 6 times at temperature 0.9. Items it gets right 0 or 6 times are dropped: they carry no gradient. This is DAPO's filter, done once before training.

**Target:** about 1,700 kept items. That is one pass at 8 per step, about 212 steps, which fits the 12 h mbp cap at exp_039's ~200 s per step.

**Balance:** EN/PL/ES 1:1:1; numeric, categorical, yes/no and abstention each at least 15 %; contrast twins about 40 %.

**Manifest:** item counts, chunk coverage, sha256 of every file, and the base's pass-rate histogram before and after filtering.

## 5. The test set: v3 (fresh sessions, sealed before training)

- **≥150 rows,** built like the training items but from all temas including Módulo 6. Rule-fresh against v1, v2.1 and every training item (exp_035 amendment 9's criterion: no v3 key rule stated in any training label or eval gold). Trigram < 0.45 against every training question.
- **Composition:** 40 % contrast pairs, both twins in v3; 15 % abstention rows; 10 % out of scope; the rest single items. EN/PL/ES 1:1:1.
- **Each row carries:** a label with accepted forms, a gold prose answer for the judged subset, and the source chunk id.
- **Andrei reads a 30-row sample before the seal** (his COAPI diploma is the only non-model check).
- **Use policy:** a test set; one scored pass per arm; never a development set.

## 6. The reward (deterministic, no model call)

| Component | Value |
|---|---|
| `answer` | final line matches the label: **+3**; wrong: **−2**; missing or unparseable: **−1** |
| `abstain` | abstention item answered "not in the material": +3 (it is the `answer` term). A non-abstention item answered "not in the material": −2 |
| `lang` | ±0.5 |
| `plain` | ±0.25 |
| `cite` | capped at +0.25 if every citation is in context; −0.5 per citation not in context, floor −1 |
| `num` | −0.5 per unsupported number, floor −1; no positive term |
| `length` | −1 at the cap; no floor and no band (exp_039 run 1's lesson) |
| `repeat`, `copy` | −1 each, as exp_039 |

The `answer` term's range (−2 to +3) is larger than all form terms combined (about 1.5), so a well-cited wrong answer always scores below a plain right one.

**What the reward still cannot see:** whether the prose explains the right rule, given a right final line. That is guarded by H2, the judged prose, and by H5, the audit.

## 7. Training recipe

Exactly exp_039's (§7 there) except:
- items: §4;
- reward: §6;
- iterations: one pass, about 212 steps.

β 0.02, group 6, ε 0.2, temperature 0.9, 512 tokens, thinking off, seed 40. Checkpoints every 25 steps; selection by exact match on the validation temas (M5 T42–43), greedy.

The 12 h stop rule applies, with a pilot of 10 steps. Platform: the mbp.

## 8. Step 0: the quantiser (before anything else; its own result)

Build an importance matrix with `llama-imatrix` from guide text (the same public guide, chunked as retrieval sees it). Quantise the untrained base to Q4_K_M with it. Score on the 40 with-substance v1 rows, with the same Fable judge protocol:
- B0-i (our imatrix Q4) against B0-q8 (1.425 in exp_039's Q8 control).
- Unsloth's Q4_K_M **without** the glue. This separates the quantiser from the glue in A12's +0.225, which exp_039 left confounded.
- B0-q8's exp_039 replies are **re-judged in the same batches** as the two new arms, so the rule compares answers judged side by side, not across judging sessions. Exact procedure: [STEP0.md](STEP0.md).

**Rule (fixed in advance):** the imatrix recipe is adopted for all exp_040 arms if B0-i scores within 0.10 of B0-q8. If it does not, Q8_0 is the primary quantisation.

**Result (2026-10-15): adopted.** B0-i 1.375 against B0-q8 1.40 (d −0.025, CI −0.20 to +0.15); `qwen3-4b-b11577:q4km-imat` (`92bfae6dd5dd`) is every exp_040 arm's Q4. See [STEP0_RESULT.md](STEP0_RESULT.md).

This step stands alone: it improves the deployed base even if exp_040 stops. It needs 120 judgments (40 rows × 3 arms) and an hour on the mini.

## 9. Hypotheses (thresholds for the pre-registration)

- **H1 (primary): correctness gained.** R3 exact match on v3 minus B0-i, paired over rows.
  - **Gain shown** iff the stratified paired bootstrap 95 % lower bound is above 0 **and** the exact one-sided McNemar test gives p < 0.05.
  - The same must hold at Q8_0 (R3-q8 vs B0-q8), or the verdict is "not robust".
  - Registered target: +0.10 accuracy.
  - Power: at 150 rows with about 25 % discordant pairs, a +0.10 difference gives McNemar power of about 0.8.
- **H2: no loss in judged substance.** Fable judges the prose on a 60-row subset (A12 rubric verbatim). R3 ≥ B0-i − 0.10 (non-inferiority). This guards against a right final line on top of a wrong explanation.
- **H3: inversions.** The share of contrast pairs with both twins right rises by ≥ 0.10 (paired, same test as H1).
- **H4: honest abstention.**
  - On v3's abstention rows, R3's "not in the material" rate rises by ≥ 0.20.
  - Its false-abstention rate on answerable rows stays ≤ 5 %.
- **H5 (gate):** the blind audit of 40 top-reward rollouts by two readers (Fable and Opus), more than 4 gamed by **either** → STOP. Gaming here means a correct final line on an answer that doesn't explain it, or a guessed label.
- **D1:** R3+G vs A13 on exact match and judged substance; descriptive, the deployment question.

**Verdict map (first that fires):** H5 fails → *Stopped — verifier*. H1 not robust → *Not shown*. H1 null → *No gain*. H1 pass and H2 fail → *Right labels, worse prose*. H1 and H2 pass → *Correctness from a checkable reward*, with H3 and H4 reported.

## 10. Cost

| Stage | Who | Time | Model use |
|---|---|---|---|
| Step 0, quantiser | session + mini | 2 h | 120 Fable judgments |
| Training items (~1,700 kept from ~2,800 drafted) | fresh Fable sessions | 1 day | drafting + 2–3 readers per item: the largest Fable cost; spread over several days of usage limit |
| v3 set (150+) | fresh Fable sessions + Andrei's 30-row read | 1 day + 30 min Andrei | as above, smaller |
| Offline sampling (6 × ~2,800 on the mini) | mini | ~10 h, Andrei starts it | none |
| Kit changes (reward, final-line parser, launcher) | session | 0.5 day | none |
| Training | mbp | ≤ 12 h | none |
| Scoring | mini | 2 h | exact match by code; ~120 judgments for H2 (60 rows × 2) + 80 for the audit |

## 11. What this does not do

- No MLP adapter, no bigger rank, no new base. Those are later cells, one variable at a time.
- No RL on the glue's closed decisions (scope gate, advice detection). That is exp_041, where Coinbase's single-label outcome fits exactly; it can reuse §4's label machinery.

## 12. Decisions (answered 2026-10-15, Andrei: "yes, ok, let's go" — all five as recommended)

1. **Step 0 runs now, on its own.** Pre-registration: [STEP0.md](STEP0.md).
2. **Item writer:** Fable writes labels, a second Fable checks blind, a third breaks ties.
3. **v3 size:** 150.
4. **Final line at serve:** kept, rendered as "In short: …".
5. **Stop rules:** 12 h mbp cap; H5 with two readers (Fable and Opus), either > 4 gamed → STOP.
