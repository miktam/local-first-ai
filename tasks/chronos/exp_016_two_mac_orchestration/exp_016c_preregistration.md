# Exp 016C — Orchestrated vs Flat: Sea View Feature Verification

**Pre-registered:** 2026-06-13  
**Parent experiment:** Exp 016 (two-mac orchestration)  
**Status:** pre-registered

---

## Hypothesis

**H6:** The orchestrated spot→verify pipeline produces zero false positives on the adversarial candidate set; the flat approach (SQL filter, no verification) produces ≥1 false positive.

Specifically: for a query "4-bed villa with sea views under €2.5M" against the CasaSol corpus, the flat pipeline returns all SQL-matching listings including those with only mountain views or beach proximity — properties that match the SQL filters but do not have sea views. The orchestrated pipeline produces only verified sea-view listings, with an evidence quote per property.

---

## Design

### What varies

| Approach | Spot stage | Verify stage | Join |
|---|---|---|---|
| Flat | SQLite filter | None | All candidates returned |
| Orchestrated | SQLite filter | leaf call per candidate → `{verified, evidence}` | Code filters to `verified=True` |

Both approaches use the same SQL filter. The verify stage is the only difference.

### Leaf call contract (verify)

- **System:** fact extractor, JSON output only: `{"verified": bool, "evidence": str}`
- **User:** one description, one question about one feature
- **Model:** gemma4:26b on localhost (Ollama)
- **Retry:** one retry on parse failure; fail-safe returns `verified=False`

### Adversarial candidate set (pre-adjudicated, 2026-06-13)

SQL filter: `property_type=villa AND bedrooms>=4 AND price<=2500000 AND listing_status='active'`

| casasol_id | Expected | Adversarial role |
|---|---|---|
| seed-20260405-144103 | True | TP — explicit "panoramic sea views" |
| seed-20260503-192259 | True | TP — explicit "panoramic sea and mountain views" |
| seed-20260405-145924 | False | FP candidate — "minutes from beaches" (beach ≠ sea view) |
| seed-20260405-155130 | False | Key adversarial FP — "panoramic mountain views", no sea mention |
| witness-kdteam-20260606-191505-img-6130 | False | FP candidate — no views mentioned |
| witness-kdteam-20260606-191557-img-6131 | False | FP candidate — no views mentioned |
| witness-kdteam-20260606-202627 | False | FP candidate — no views mentioned |

Flat approach returns all 7; orchestrated should return only the 2 True entries.

### Gate

`H6_CONFIRMED` if `orch_fp < flat_fp`  
`H6_INCONCLUSIVE` if `orch_fp == flat_fp`  
`H6_FALSIFIED` if `orch_fp > flat_fp`

Expected: H6_CONFIRMED (orch_fp=0, flat_fp=5).

---

## Connection to evidence chain

- **Exp 012:** gemma4:26b scores 0/8 flat cross-document reasoning. Frontier model: 5/8.  
- **Exp 013:** Same model scores 2/3 when join is externalized to code.  
- **Exp 016B (2026-06-13):** 6 bounded leaf calls, 0 interventions, 10/10 tests. One leaf = one symbol = one output.  
- **Exp 016C (this):** Applies the same leaf pattern to property feature verification. Replaces cross-document reasoning with one-document leaf calls + code join.

**Pitch line:** *"We didn't upgrade the model. We upgraded the question."*

---

## Artefacts

- Runner: `casasol/scripts/run_exp_016c.py`
- Pipeline: `casasol/scripts/client_match.py`
- Results: `measurements/exp_016c_results.json` (written after run)
