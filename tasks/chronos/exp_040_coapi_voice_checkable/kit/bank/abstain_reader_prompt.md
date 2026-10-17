# exp_040 abstention reader prompt (rev 0)

You check whether a question can be answered from a given piece of study material. A small Spanish property-law model will be trained to say "not in the material" when, and only when, the material does not settle the answer. Your verdicts decide which questions it learns that on, so a wrong "undecidable" teaches it to give up on answerable questions.

Your batch file (JSON lines) gives one row per line:
- `id`: the row id. Copy it exactly.
- `q`: the question, in English, Polish or Spanish.
- `material`: the retrieved study-guide text, the only source allowed.
- Either `candidates`, the possible answers (names only, random order), or `numeric_unit`, when the answer is a number.

Read only that file.

## Per row

1. Read the material, then the question.
2. Decide from the material alone. Your own legal knowledge never counts, even when you are sure of the law.
3. If the material states, or lets you compute, which candidate is right (or the number) for exactly the situation in the question, give that pick.
4. If it does not, answer `"undecidable"`. This covers three cases:
   - the material doesn't cover the rule;
   - it covers a neighbouring rule but not the branch the question needs;
   - you would have to assume a fact it doesn't give.
5. When in doubt between a pick and `"undecidable"`, give the pick and set `doubt: true`.

## Output

JSON lines only, one per row, nothing else:

```
{"id": "<id>", "pick": "<candidate name>" | <number> | "undecidable", "doubt": true|false, "why": "<one sentence: the sentence of the material you relied on, or what is missing>"}
```
