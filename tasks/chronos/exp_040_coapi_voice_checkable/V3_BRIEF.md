# exp_040 v3 test set: brief for a fresh session

*Andrei starts this in a **new** Claude Code session (not the exp_040 design session, which is not blind to v1/v2.1 judge notes). The fresh session builds, checks and seals v3, and hands back only its sha256 and counts. Written 2026-10-15.*

## What v3 is for

v3 is the sealed test set for exp_040. It measures whether a model trained against a correctness-checkable reward answers held-out Spanish property-law questions more correctly than its untrained base. The primary metric is **exact match on a final `Answer:` line, by code**. The secondary metric is Fable-judged prose on a subset.

**Use policy:** a test set; one scored pass per arm; never a development set.

## Inputs (read these; nothing else in `tasks/exp039/` or `tasks/exp040/`)

| Input | Path | Use |
|---|---|---|
| Chunk pools | `~/REPOS/casasol/coapi_voice/data/bank40/pools.json` | v3 rows come **only** from the `holdout` (92) and `m6` (35) pools |
| Guide index (live) | `~/REPOS/casasol/coapi_voice/index/guide_en.json` (`f1e28120`), `guide_pl.json` (`9a422b0e`) | chunk texts by id |
| Frozen training bank | `~/REPOS/casasol/coapi_voice/data/bank40/BANK_FROZEN.jsonl` (sha256 in `BANK_FROZEN.sha256`) | exclusion list: read only the `q.en`, `label`/`numeric` and `rule` fields |
| Item schema and parse rules | `~/.openclaw/workspace/tasks/exp040/bank/BUILD.md` §2 | v3 rows use the same schema |
| Writer and checker prompts | `~/.openclaw/workspace/tasks/exp040/bank/writer_prompt.md`, `checker_prompt.md` | reuse; v3 adds the fields below |

**Do not read:**
- v1 or v2.1 eval rows (beyond the trigram check by code below);
- any `judge/` folder, `runs/` folder, `RESULT*.md` or `REVIEW*.md`;
- exp_039 or exp_040 transcripts.

## Composition: 150 rows

| Share | Rows | Kind | Label |
|---|---|---|---|
| 40 % | 60 (30 pairs) | contrast pairs: both twins in v3, labels differ | candidates or numeric |
| 35 % | 52 | single items | candidates or numeric |
| 15 % | 23 | **abstention**: a normal question whose source chunk (and, for PL, its whole PL section) is withheld from retrieval at scoring time | `ABSTAIN` |
| 10 % | 15 | out of scope: a question a buyer might ask that the guide does not cover (e.g. Portuguese tax, a car purchase, Moroccan inheritance law) | `ABSTAIN` |

- **Language:** EN/PL/ES 1:1:1 (50 each). Each row is **one** language; twins share a language.
- **Type:** at most 25 % yes/no; at least 25 % numeric.
- **Spread:** across holdout temas and M6; no tema above 12 rows.

## Row fields

```json
{"id": "V3-001", "lang": "en|pl|es", "q": "...", "type": "categorical|yesno|numeric|abstain",
 "candidates": {...} | null, "label": "...", "numeric": {...} | null,
 "source_chunk": "en:M6:T46:concepts:2" | null, "withhold": ["en:...", "pl:..."] | [],
 "rule": "one sentence", "twin_of": "V3-0xx" | null, "flip": "..." | null,
 "gold": "a short prose answer a careful agent would give (2–5 sentences)"}
```

- `ABSTAIN`'s accepted forms per language: EN "not in the material", PL "brak w materiale", ES "no está en el material".
- `gold` is needed for the 60-row judged subset. Write it for every row anyway.

## Build steps

1. **Draft:** Fable writer subagents work over the holdout and m6 chunks, using the bank's writer prompt plus this brief's fields. Draft about 220 rows.
2. **Check:** a blind Fable checker per row, using the bank's checker prompt. Abstention rows are checked against the context retrieved **with the withheld chunks removed**, and must be undecidable there. Out-of-scope rows are checked as out of scope.
3. **Rule overlap with the training bank:** for each row, a Fable reader compares its `rule` against the `rule` lines of training items from the same tema. Code first narrows by tema and same label. A row whose key rule a training item states is dropped.
4. **Trigram freshness:** by code, < 0.45 against every training-bank question, v1 and v2.1.
5. **Select 150** to the composition above (seeded), then write `coapi_eval_v3_draft.jsonl`.
6. **Andrei reads 30 rows**, sampled by seed and stratified by kind. Fix or replace any row he rejects (re-check it), then re-sample until a sample of 30 passes.
7. **Seal:**
   - write `~/REPOS/casasol/coapi_voice/data/eval/coapi_eval_v3.jsonl` and its sha256;
   - commit in casasol: `coapi_voice/data/eval: v3 sealed (<sha256[:8]>)`;
   - write a manifest with counts by kind, language, type and tema.

## Hand-back (the only thing that leaves this session)

The sha256, the counts table, and the path. **No rows.** The exp_040 design session pins the sha256 and never opens the file.
