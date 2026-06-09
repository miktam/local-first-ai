# Intervention Ledger — exp_013

Every change that could affect the score goes here, classed before the run it's tested in.
This is what keeps the final number interpretable. If a change isn't logged, its result
isn't publishable.

## Classification

- **generic-scaffolding** — structural help that names no rubric content: decomposition,
  present/absent checklists over *property types*, article-matching over supplied rule
  text, output schemas, sampling/context settings. Fair. Publishable.
- **fair-evidence** — changes the inputs the model sees by adding a source that was
  legitimately part of the system but missing from the bundle (e.g. the VLM file behind
  A2). Fair, but the result must be reported as "with expanded context", not compared
  head-to-head with the original bundle.
- **rubric-leakage** — any prompt, checklist item, or example that names or paraphrases a
  specific rubric answer (e.g. "verify a model *hash* exists", "check for a *DPIA*").
  Forbidden. Voids the score for that run.

Rule of thumb: if removing the rubric from the room would make the change impossible to
write, it's leakage.

## Ledger

| id | date | stage targeted | change | class | diagnosis before | score before | score after | publishable | notes |
|----|------|----------------|--------|-------|-------------------|--------------|-------------|-------------|-------|
| 000 | YYYY-MM-DD | baseline | none — first localization run | generic-scaffolding | — | 0/8 (exp_012) | TBD | y | establishes the drop-off point before any intervention |
| 001 | | stage1 | _example:_ open extraction → present/absent checklist over property types | generic-scaffolding | extraction_empty | | | | only if diagnosis says extraction_empty/vague |
| 002 | | stage2 | _example:_ bridge becomes matching task over supplied article text | generic-scaffolding | no_bridge | | | | feed rule text, do not name which rule |
| 003 | | stage3 | _example:_ loosen verifier after calibration check | generic-scaffolding | over_pruned | | | | run calibration first; record accept-rate on known-true items |

## Per-run checklist

- [ ] `rubric.md` pre-registered and unchanged since before this run
- [ ] num_ctx confirmed active via `ollama show` (not 4096 default)
- [ ] canary passed (Stage 0 did not abort)
- [ ] diagnosis label recorded from the run's stderr / trace
- [ ] every prompt change since last run added above with a class
- [ ] no row classed `rubric-leakage` is part of a reported score
- [ ] trace.jsonl committed alongside the result

## Verifier calibration sub-experiment (for `over_pruned`)

Before loosening the verifier, feed it each known-true rubric item as a candidate and
record whether it accepts. A verifier that rejects real gaps is the bug; a verifier that
accepts them means the problem is upstream (extraction/bridge), not strictness.

| rubric item | verifier verdict (supported?) | expected | match |
|-------------|-------------------------------|----------|-------|
| | | true | |
