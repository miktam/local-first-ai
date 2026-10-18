# COAPI Voice: fixes after exp_041 (2026-10-17 night)

*Andrei: "fix it before starting a new experiment", "don't start any new experiments yet", "proceed with fixing, don't stop … stop only if something is really broken". No experiment was started and no test set was touched. Every fix was measured on training-side data: the frozen bank of 734 questions, with the 29 held-out validation items as the check. Commits are in casasol, local; nothing was pushed or published.*

## 1. Spanish retrieval (casasol `1257a46`)

Spanish questions are searched against the English index, and BM25 missed the source chunk on about half of them.
- **Fix:** `coapi_voice/xling.py` translates the Spanish question to English with the deployed model (Qwen3-4B, temperature 0) and searches with the translation. The model still sees and answers the Spanish question.

| ES query | Bank | Validation |
|---|---|---|
| original Spanish (before) | 0.489 | 0.621 |
| **translation (adopted)** | **0.703** | **0.931** |
| translation + original | 0.666 | 0.897 |
| translation + glossary terms of the original | — (tested after #2: 0.759 vs 0.771) | 0.966 |
| translation by gemma4:26b (tested after #2) | 0.792 vs 0.771 | 1.000 |

gemma4:26b translates slightly better (+0.02) but needs a second 17 GB model loaded next to the voice, so it was not adopted. It is recorded as an option.

## 2. Retrieval depth and budget (casasol `0f8f600`)

Misses mostly had the right tema but the wrong chunk: retrieval took one chunk per tema.
- **Fix:** `guide_bm25.LEAD_DEPTH = 2` takes two rule chunks from the leading tema first. The glue budget goes from 1,200 to 1,350 tokens; k stays 5.

| Source chunk retrieved (k 5) | EN | ES (translated) | PL | Validation EN / ES / PL | Context |
|---|---|---|---|---|---|
| before | 0.782 | 0.703 | 0.888 | 1.00 / 0.93 / 1.00 | ~1,130 |
| depth 2, budget 1200 | 0.843 | 0.755 | **0.864** | **0.97** / 0.97 / 1.00 | ~1,130 |
| depth 2 on the EN index only | 0.843 | 0.755 | 0.888 | **0.97** / 0.97 / 1.00 | ~1,130 |
| **depth 2, budget 1350 (adopted)** | **0.871** | **0.771** | **0.899** | **1.00 / 1.00 / 1.00** | ~1,240 |

Only the adopted variant improves every language with no regression. Its cost is about 110 more tokens per prompt. A larger budget (k 7, 1,600 tokens: about +35 % context) helps more, but it changes what the voice reads, so it belongs to an experiment.

## 3. glue.clean() cut complete answers (casasol `a43e26b`)

`clean()` dropped every final line without a full stop, as if the reply had been cut off. On exp_040's stored glue answers it removed the last line of 33 of 135 A13 replies and 12 of 139 R3G replies, **none of them at the token cap**.
- **Fix:** an unpunctuated ending is trimmed only when the reply hit the token cap. `clean()`'s default keeps the old behaviour for other callers.

## End-to-end check

One question each in EN, PL and ES through `glue.answer`: all in scope, the right rule (art. 21 LAU), the Spanish one now in the right tema. Full casasol suite: 3,131 passed, 1 skipped.

## Left as is, on purpose

- **`test_chunk_sizes_fit_a_phone_context` fails, pre-existing** (since the 2026-10-13 index rebuild). `en:M1:T10:concepts:5` is 530 tokens against the 500 limit. Fixing it means re-chunking and rebuilding the index, which renumbers chunk ids that the bank and v3 refer to. The chunk fits the 1,350 budget.
- **The exp_039 / exp_040 runners now refuse.** `glue.py` and `guide_bm25.py` moved past their pin b1a648c, by design; reproduce them from that commit. exp_039's prompt-pin test now skips with that reason.
- **The exp_041 harness renders v3 rows with today's retrieval.** A re-run would not reproduce the scored pass. exp_041 is closed; a gate re-test needs its own pre-registration and a fresh test set.
- **A remaining quirk:** on a genuinely truncated reply, `clean()` splits sentences at "art.", so it can keep "…art." and drop the number. This only applies at the cap now.
- **Not pushed:** casasol is ahead of Codeberg by these commits plus earlier exp_041 work. Pushing is Andrei's call.

## Open, for when experiments resume

- About 13 % (EN) to 23 % (ES) of bank questions still miss their source chunk.
- The exp_041 gate (Clef 27B) was not wired into the glue: its verdict was "No gate earns its place". A re-test should use today's retrieval, so its false-abstention rate is measured on rows whose source is actually present.
