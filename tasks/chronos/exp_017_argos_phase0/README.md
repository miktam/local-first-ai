# exp_017 — Argos Phase 0: Feed Reconnaissance

**Status:** PRE-REGISTERED — 2026-06-12  
**Registered by:** Andrei / Nestor  
**Chronos path:** `local-first-ai/tasks/chronos/exp_017_argos_phase0/`

---

## Hypothesis

The DGT National Access Point (DATEX II) carries sufficient dynamic data (operational status
+ timestamps) to detect EV charger faults and stale reporting without any scraping of
operator-facing apps. Real update cadence is unknown; we assume 15–60 min for dynamic data
and ~24 h for static data. The OpenChargeMap API will reveal chargers absent from the
official feed ("ghost" category).

## Bounding box

**Scope confirmed:** Málaga–Gibraltar corridor

```
lat:  36.00 – 36.90
lon:  -5.40 – -4.40
```

Includes: Málaga city, Torremolinos, Benalmádena, Fuengirola, Marbella,
Estepona, San Roque, La Línea de la Concepción.

## Success criteria

All five must be true before Argos Phase 1 code is written:

1. At least one full DGT NAP feed snapshot has been downloaded and parsed.
2. Status enum values observed in the wild are enumerated (the state machine alphabet).
3. Real update cadence is measured over ≥ 24 h (not assumed).
4. OpenChargeMap bounding box pulled; overlap count vs NAP recorded.
5. `findings.md` committed with: feed access notes, auth requirements, point counts,
   field inventory, cadence table, status vocabulary, REVE verdict, OCM overlap.

## Evidence contract

Every artefact in `raw/` must include:
- ISO 8601 timestamp in the filename or in a companion `.meta.json`
- Source URL
- SHA-256 hash of the payload (recorded in `findings.md`)

Aborted pulls go to `raw/aborted/` — nothing deleted.

## Open questions this experiment must answer

| Question | Why it matters |
|---|---|
| Auth required for DGT NAP? | Gating: do we need a registered account before collector can run? |
| Format: DATEX II XML or JSON? | Parser design in Phase 1 |
| Field: does the feed carry `lastUpdated` per charger? | Needed for STALE_DATA detection |
| Status vocabulary in the wild | State machine alphabet for Phase 2 detector |
| Real update cadence (dynamic data) | Sets production polling interval; must be measured, not assumed |
| REVE: public JSON backend usable? | Determines whether REVE is a second source or skip |
| OCM overlap in scope | Baseline for GHOST category size |

## Negative results are results

If DGT NAP has no per-charger timestamp field → log it; STALE_DATA rule design changes.  
If REVE terms forbid API use → log it; REVE is dropped from Phase 1.  
If OCM has zero overlap → log it; GHOST category starts with a count of 0.
