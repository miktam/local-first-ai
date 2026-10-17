# exp_040 bank writer prompt (rev 1: yes/no restricted after the pilot, 2026-10-15)

You write training questions for a small model that answers Spanish property-law questions from retrieved study-guide text. In training, the model writes an explanation and then a final line `Answer: …`. Code rewards it only when that final line matches **your label**. So a label must be unambiguous, and **decided by the chunk text alone**.

Your batch file (JSON lines) gives several guide chunks, each with `id`, `title`, `text_en` and, when there is one, `text_pl` (the Polish version of the same chunk). Read only that file.

## Per chunk

Write **up to 3 items**. Zero is a fine answer: write a skip line when the chunk is a definition list, a history note, or anything that does not decide a concrete answer.

Item types:
- **categorical:** which tax, deadline, article, party, body, document, register, procedure or regime applies.
- **yesno:** whether something is required, allowed, valid or due. A yes/no answer can be guessed half the time, so use it only when no categorical or numeric framing fits, and preferably as a contrast pair whose labels flip.
- **numeric:** a number the chunk states, or that follows from it in one step (a rate, a count of days or months, a threshold, an amount).

### Questions

- Ask the way a buyer, seller, owner or tenant would: a concrete situation in one to three sentences.
- Do not quote the chunk. Do not name the answer. Do not refer to "the text" or "the guide".
- The question alone must contain every fact the label depends on. If the answer would change with an unstated fact (age, residence, new build or resale, region), state that fact.
- Write the same question in natural English (`q.en`), Polish (`q.pl`) and Spanish (`q.es`), not as literal translations. The Polish may use the Polish chunk's terms.

### Candidates

- Give 2–5 named alternatives a layperson might pick, with the label among them. For yesno, use `yes` and `no`.
- For each candidate, give its **forms** in en, pl and es: the short strings that would appear in a one-line answer, including acronyms and the Spanish legal term.
- A form must not be a substring of another candidate's form.
- numeric items have no candidates. Give `numeric: {"value": <number>, "unit": "<unit>", "tol": 0.005}` instead.

### Contrast twins

When the chunk itself states a rule with explicit branches (resale vs new build, under vs over an age, habitual residence vs not), add a **twin**: the same question with one fact changed, so the label changes. Set `twin_of` to the first item's id and `flip` to a short description of the change. The twin's label must also be decided by this chunk.

### Never

- A rule the chunk marks provisional, pending, or tied to RDL 29/2026.
- Advice or opinion ("should I…", "is it worth…").
- An answer that needs another chunk or outside legal knowledge.
- Two items in one chunk that test the same fact (a twin is not a repeat).

## Output

JSON lines only, nothing else. One line per item, ids `<chunk id>#<n>` (n = 1, 2, …):

```
{"item": "en:M2:T12:concepts:3#1", "chunk": "en:M2:T12:concepts:3", "type": "categorical",
 "q": {"en": "…", "pl": "…", "es": "…"},
 "candidates": {"ITP": {"en": ["ITP", "transfer tax"], "pl": ["ITP", "podatek od czynności cywilnoprawnych"], "es": ["ITP", "Transmisiones Patrimoniales"]}, "IVA": {…}},
 "label": "ITP", "numeric": null,
 "rule": "<one sentence: the rule in the chunk that decides the label>",
 "twin_of": null, "flip": null}
```

A chunk with no item gives one line: `{"chunk": "<id>", "skip": "<short reason>"}`.
