# exp_040: how the training bank and v3 get built (plan, rev 0)

> **Status 2026-10-15: training bank FROZEN.**
> - `BANK_FROZEN.jsonl` has 734 items (sha256 `6de249dd…`): categorical 427, numeric 237, yes/no 70; 131 twin links; 261 of 264 train chunks.
> - `VALID_FROZEN.jsonl` has 29 items (`67934fb5…`).
> - Waves 1–3, the pilot and the validation pool cost about 4.3 M Fable tokens.
> - The checker flagged 12 items ambiguous in all; all dropped.
> - Code dropped 20 near-duplicates, 1 option-collision and 1 trigram-fresh failure.
> - Manifest: `data/bank40/FREEZE_MANIFEST.json`.
> - Next: the v3 session (`bank/V3_BRIEF.md`), abstention variants (§3.4), kit changes, offline filter.
>
> **Kit, 2026-10-15** (casasol `5ca5bbd`, `84a76b5`, `d7206ee`), in `coapi_voice/rl40/`:
> - `reward40.py` (rev 3, 32 tests) and `prompt40.py` (one renderer for train, filter and serve; A13 imports `system40`).
> - `build_prompts40.py` → `data/rl40/prompts/`: 2,104 train rows and 79 valid rows.
>   - The train rows break down as: bank × EN/PL/ES 1,567; OOS-abstain 87; arith 450.
>   - Drops: bank rows whose source chunk is not retrieved (EN 153, **ES 386**, PL 82). ES retrieves from the EN index, and half the ES items lose their rule. This is serve parity, recorded and not changed.
> - `dapo_filter.py` and `.sh` run on the 0.33.0 side server with GRPO rollout sampling.
> - Smoke run (4 rows × 6):
>   - **parser fix:** Qwen3-4B writes one paragraph and puts "Answer: …" last, so the key now also counts after a sentence end;
>   - after the fix the base was right on 23 of 24, so **yield may be far below half**;
>   - throughput 0.41 samples/s, so the full run takes about 8.6 h. Andrei runs a 60-row yield probe first.
>
> **Filter run 1, 2026-10-15** (12,624 samples):
> - 579 of 2,104 rows kept: 1,017 were 6/6 right and 508 were 0/6. The base was right on 63 % of samples.
> - Arithmetic and numeric zeros are genuine.
> - **Categorical and yes/no zeros were mostly the matcher.**
>   - 344 of 427 categorical questions name no candidate, so right answers in other words scored "missing".
>   - On yes/no questions, the final-line sentence ("the short name of the rule…") steered the model to give a citation instead of yes/no.
> - **Decision (Andrei, 2026-10-16): show the options.** Categorical and yes/no prompts end with "Options: a / b / c." Retrieval runs on the bare question, and the v3 harness renders options the same way (`prompt40.options_line`).
> - Prompts rebuilt (casasol `c4fb850`, `04fa634`). Numeric, arith and OOS samples are reused through per-row prompt hashes. 1,016 rows are redrawn: 6,096 samples, about 4.1 h.
> - Run 1's select output is in `runs/exp040_dapo/select_pre_options/`.
>
> **Filter run 2, 2026-10-16** (6,096 samples redrawn, 18,720 in total):
> - **553 of 2,104 rows kept** (casasol `6f93af8`, `data/rl40/train/`, sha `376cf2e2…`). 1,281 were 6/6 right and 270 were 0/6. The base was right on 75 % of samples.
> - "missing" fell from 1,736 to 303.
> - **With options shown, the base picks the right one 6/6 on 614 of 870 categorical rows.** The labelled task is mostly within reach already.
> - The variance sits in:
>   - PL categorical (111 kept) and PL yes/no (37);
>   - arithmetic (144);
>   - numeric (127).
> - Kept twin pairs: 33. Module mix: M1 288, M4 121, M3 64, M2 44, M5 26.
> - At batch 8 that is about 70 steps per pass.
> - Headroom risk for H1: if v3 categorical rows also run at about 90 % for the base, the gain can only show on PL, arithmetic and abstention.
>
> **v3 SEALED 2026-10-16** (in the "[exp40] V3 test" session):
> - casasol `c93b451`, sha256 `2c6cbad123adff6d9e1b55ed91cd61c71259a3800cd0a5b25d83537a061da454`.
> - This session pinned the sha, read counts only, and never opened the rows.
> - 150 rows: pairs 60, singles 52, abstain 23, out of scope 15. EN, PL and ES are 50 each.
> - By type: numeric 76, categorical 32, yes/no 4, abstain 38.
> - Inputs: the frozen bank `6de249dd`, index `f1e28120` / `9a422b0e`.
> - Build: 285 drafted; 28 dropped on rule overlap with the training bank.
> - Retrieval (the manifest's method note): withheld chunks are removed from the index. v3 abstention rows carry no candidates (a count by code), so no options are shown on them.
>
> **Abstention variants (§3.4), set up 2026-10-16:** `bank/abstain.py`, `bank/abstain_reader_prompt.md`.
> - 110 items (15 %, one per twin group) × EN/PL/ES = 330 rows, in 11 reader batches of 30.
> - Material is about 340k tokens. One blind Fable reader per row; only "undecidable" becomes ABSTAIN.
>   - **Read on 2026-10-16** in two chunks (batches 1–4, then 5–11). Fable cost about 1.31M tokens (about 120k per reader); Andrei's Fable meter moved about 1 point per 0.5M.
>   - One id repair (batch 3, `#1`→`#3` suffix, verified by position), logged in `abstain/REPAIRS.md`.
>   - **Frozen:** 163 of 330 confirmed undecidable. EN 51, PL 53, ES 59; categorical 107, numeric 40, yes/no 16.
>   - Dropped: 161 the reader could decide, and 6 "undecidable" verdicts flagged with doubt.
>   - `ABSTAIN_FROZEN.jsonl` sha `53421523…` (casasol `b9f7b07`).
>   - Prompts rebuilt to 2,267 train rows. The other 2,104 rows are byte-identical, so the filter draws only 163 × 6 = 978 samples (about 40 min).
>
> **Filter run 3, 2026-10-16** (978 new samples):
> - **622 train rows kept** (sha `9fe67900…`), 79 of them abstention: 69 withheld-source rows and 10 OOS.
> - Of the 163 variants: 69 have variance, 79 never abstain (0/6) and 15 always abstain.
> - **Reward fix (rev 3, before registration):** on an abstention row, a final line that does not abstain is now wrong (−2), not missing (−1). 708 such answers had been paying the policy to guess. The selection is unchanged, because it counts "right" only.
>
> **R3 trained on the mbp, 2026-10-16/17** (HYPOTHESIS signed 21:31 CEST):
> - First attempt: out of GPU memory ("[METAL] … Insufficient Memory", in step-1 rollout generation). The pilot and the full run had been started 6 s apart and ran at the same time.
>   - exp_040 prompts are about 12 % longer than exp_039's (mean 1,714 vs 1,529 tokens; max 2,310 vs 2,170).
> - **Environment change only:** `sudo sysctl iogpu.wired_limit_mb=120000` (default 0 = about 75 % of 128 GB; resets at reboot). No registered value changed.
> - Pilot alone: 10/10 steps in about 2,198 s, projecting about 4 h 45 m for 78 steps, within the 12 h rule.
> - Full run: 78/78 steps, 22:21:35 → 03:05:37, no errors. Trainable 11,796,480; effective = config. Checkpoints 25/50/75 + final.
>   - Peak footprint about 116 GB; MLX peak about 44 GB.
>   - Copied to the mini: `runs_rl/mbp_20261016_r3/`, sha256 verified on the mbp.
> - **Training reward was flat**, unlike exp_039's 2.2→3.6. Block means of 13 steps:
>   - total 1.77, 1.64, 1.76, 1.61, 2.21, 1.58;
>   - answer term 1.09, 0.94, 1.02, 0.87, 1.47, 0.87;
>   - KL 0.008 → 0.090 → 0.040.
>
>   Recorded before selection or any v3 answer.

*2026-10-15. This turns DESIGN.md §4–§5 into steps. Nothing here is run until the step 0 result is in and a 3-chunk pilot has been read.*

## 1. Chunk pools: fixed by code before any question is written

Pools are built from the live EN index (`f1e28120`) by `bank/pools.py` (seed 40). Only chunks of kind `concepts`, `citations` or `formulas` are used; `vocab` and `queries` are not rule text.

| Pool | Chunks | Who may open it |
|---|---|---|
| **train** | M1–M4 and M5 T39–41, minus a 25 % holdout per tema: **264** (254 with a PL twin chunk) | the writer and checker subagents |
| **valid** | M5 T42–43: **11** | the writer and checker subagents (items for checkpoint selection only) |
| **holdout** | the 25 % per tema: **92** | **the v3 session only** |
| **m6** | Módulo 6: **35** | **the v3 session only** |

Pool membership is written to `bank/pools.json`, with each chunk's text sha256. The holdout and m6 chunk **texts** are never shown to this session or to any training subagent. The guide repeats rules across chapters, so the chunk split is necessary but not sufficient; §4 adds a rule-overlap check.

## 2. Item schema

Every item carries a **closed candidate set**, so that agreement and reward are exact matches, not fuzzy ones:

```json
{"item": "M2-T12-c3-01", "chunk": "en:M2:T12:concepts:3", "type": "categorical|yesno|numeric",
 "q": {"en": "...", "pl": "...", "es": "..."},
 "candidates": {"ITP": {"en": ["ITP", "transfer tax"], "pl": ["ITP", "podatek od przeniesienia"], "es": ["ITP", "transmisiones patrimoniales"]},
                "IVA": {...}, "AJD": {...}},
 "label": "ITP",
 "numeric": null,
 "rule": "one sentence: the rule in the chunk that decides the label",
 "twin_of": null, "flip": null}
```

- **yes/no:** candidates `yes` and `no`, with forms in each language.
- **numeric:** `numeric: {"value": 26256, "unit": "€", "tol": 0.005}` and no candidates. The **first** number on the final line is the answer.
- **Reward parse (refines DESIGN §2):**
  - The final `Answer:` line must contain the forms of **exactly one** candidate, and it must be the label, for +3.
  - Forms of two or more candidates count as a hedge: −2.
  - No candidate found: −1.

  This closes the "ITP or IVA" hedge.

## 3. Training bank: writer → blind checker → tie-break

1. **Writer** (Fable; 10 train chunks per subagent; prompt `bank/writer_prompt.md`):
   - Per chunk, up to 3 items where **the chunk alone decides the label**. Zero is acceptable.
   - Where the chunk itself states both branches of a rule, a contrast twin is added (`twin_of`, `flip`).
   - The writer gives a question in EN, PL and ES, the candidates with their forms, the label and the rule.
   - No items rest on a rule the guide marks provisional (RDL 29/2026).
2. **Checker** (Fable, a fresh instance; about 40 items per subagent; prompt `bank/checker_prompt.md`):
   - It sees the EN question, the chunk and the candidate **names** (in shuffled order), never the label.
   - It picks one candidate (or the number), or `"undecidable"`, and it flags ambiguity.
3. **Agreement:**
   - If the checker matches the label, the item is kept.
   - On a mismatch, a third Fable reader answers blind and the item is kept on 2 of 3.
   - `undecidable` or an ambiguity flag drops the item.
4. **Abstention variants** (about 15 % of kept items, chosen by seed):
   - Retrieval runs with the source chunk removed. For a **PL** item, **every PL chunk of the same section** (module, tema, kind) is removed, not only the PL chunk with the same id.
     - EN↔PL ids are positional, and 14 sections split into different chunk counts.
     - A shared-number check on the 275 train and valid chunks gives 229 aligned, 18 weak, 18 misaligned and 10 with no PL chunk (2026-10-15).
   - A fresh Fable reader sees the question and **that** context and confirms that the candidates can't be decided from it. Confirmed items get label `ABSTAIN`; the rest are dropped.
5. **Arithmetic:** exp_039's 15 templates and new multi-step ones, with gold computed in code, about 20 % of the bank.
6. **Freshness by code:** trigram < 0.45 against v1, v2.1 and every other item's question. Exact-duplicate labels with near-duplicate questions are collapsed.

**Yield estimate:**
- 264 chunks × about 4 drafted items (with twins) gives about 1,050, and about 900 survive checking.
- Rendering each in EN, PL and ES gives about 2,700 prompts, and the offline filter keeps about half, so roughly 1,350.
- Adding arithmetic gives about 1,700.
- Module 1 carries 164 of the 264 chunks. The bank's module balance is recorded and not forced.

## 4. v3: a separate session, sealed

- **Run by Andrei in a fresh Claude Code session.** It is given `bank/V3_BRIEF.md` (written here), the holdout and m6 pools, and the **frozen** training bank as an exclusion list.
- **Rule-overlap check:** for each v3 row, a Fable reader compares its `rule` against the training items' `rule` lines from the same tema and drops it on a match. Code first narrows the comparison by tema and label.
- **Composition (DESIGN §5):** 150 rows; 40 % contrast pairs; 15 % abstention; 10 % out of scope; EN/PL/ES 1:1:1. Each row also carries a gold prose answer for the H2 subset.
- **Andrei reads 30 rows**, then the session writes `coapi_eval_v3.jsonl` and its sha256 and stops.
- **This session receives only the sha256 and the counts.**

## 5. Offline filter (DAPO), on the mini

- The base at the quantisation chosen in step 0 answers every bank prompt 6 times at T 0.9, with the exp_040 system prompt (glue system prompt + final-line sentence).
- Items right 0 or 6 times are dropped.
- This is about 16,000 generations, too long to run in-session, so Andrei starts it, with the command given here.

## 6. Order

1. Step 0 result → quantisation fixed.
2. `bank/pools.py`, the writer prompt and the checker prompt → **pilot: 3 chunks, 1 writer, 1 checker; Andrei reads the pilot items.**
3. Full bank (about 27 writer and about 25 checker subagents, plus tie-breaks and abstention readers): Fable, over several usage windows.
4. Bank frozen (sha256) → v3 session (Andrei starts it) → sealed.
5. Kit changes (final-line prompt, reward rev 3, parser and tests) → offline filter on the mini.
6. HYPOTHESIS.md signed → pilot of 10 steps on the mbp → full run.
