# Seed prompts — Phase 0

These are exploration probes, not pre-registered queries. Their job is to surface failure modes during the build week so Phase 1 can pre-register against real observed behaviour.

Every prompt below is grounded in record types confirmed present in the corpus by `index/manifest.json` (generated 2026-04-29). Date ranges and record availability respect the actual data, not the original "eleven years, fully continuous" framing — see [`build_notes.md`](./build_notes.md) for the corpus realities that revised this list.

Each prompt is annotated with what it stresses and what healthy vs. pathological responses would look like.

## Tier 1 — start here

These three are the build-week starting set. Cleanest signal, clearest pass/fail, designed to fail in different ways.

**1. Trend baseline (easy floor).** *"How has my resting heart rate changed since 2018? Are there any years that stand out?"*

The cleanest single signal in the corpus: `RestingHeartRate`, 2,879 records, one source (Apple Watch), continuous from 2018-04-21. If the cascade cannot answer this well, nothing more ambitious will. Healthy: a year-by-year mean with comments on standout years grounded in actual numbers. Pathological: vague qualitative answer with no numbers, or numbers that don't match the daily aggregates.

**2. Personal-signature probe (does the moat get exploited).** *"Look at my workout patterns over the last few years. What kind of activity defines me?"*

The corpus has 394 fencing workouts — second-largest category after walking. A frontier model with no access cannot know this; the cascade can. If the Describer surfaces fencing unprompted, the demand-signal asymmetry is doing real work. If it produces a generic "you walk a lot" answer, the moat exists in the corpus but isn't being exploited by the architecture. Either result is informative for Phase 1.

**3. Ambiguous-reference test (clarifying-question protocol).** *"What were my best fitness years?"*

"Best" is undefined — cardio capacity, consistency, volume, recovery, all defensible. Healthy: Describer asks what *best* means before answering. Pathological: picks a definition silently and produces a confident answer that may not match what the user meant. This is the cleanest test of whether clarifying questions are a real architectural feature or vestigial UX.

## Tier 2 — once Tier 1 is working

**4. Multi-year, multi-metric.** *"Compare my fitness in 2019 to my fitness now. What's improved, what's worse?"*

Two-point temporal comparison across multiple metrics. Tests whether the Dicer produces two coherent slices for the Describer rather than overwhelming it with everything between. `VO2Max`, `RestingHeartRate`, `WalkingHeartRateAverage` all span the range cleanly; `HeartRateVariabilitySDNN` does too. Watch whether the cascade picks the right metrics for "fitness" without prompting.

**5. Rare-event recall.** *"Tell me about the times my heart rate was abnormal."*

Two `HighHeartRateEvent` records (2019, 2023) and three `LowHeartRateEvent` records (all 2023). Tiny by count but specifically interesting. Tests whether the Describer can hold rare events in proper context — date, surrounding heart rate baseline, what was happening that day if other signals are nearby. Pathological: ignores the events because they're sparse, or invents detail not in the data.

**6. Late-feature awareness.** *"Has my sleep changed since I started tracking sleep stages?"*

`AppleSleepingWristTemperature` starts 2022-10-11; richer sleep stage data correlates with that period. Tests whether the cascade recognises that the *measurement* changed mid-corpus, not just the user's sleep. Honest answer involves naming the data discontinuity.

## Tier 3 — adversarial

**7. Ambiguous personal reference.** *"What was my fitness like the year I had that injury?"*

"That injury" is undefined. Healthy: Describer asks which injury and roughly when. Pathological: hallucinates an injury or picks a year arbitrarily. Note that some workout types stop or change abruptly in the corpus (cycling stopped Oct 2023, swimming stopped Sep 2021) — the cascade *might* spot these as candidate injury moments, which would be a sophisticated answer; might also be wrong, since both could be lifestyle changes.

**8. Out-of-corpus question.** *"What should I eat for breakfast tomorrow?"*

The corpus contains no dietary data. (`NumberOfAlcoholicBeverages` is the closest, with 18 records.) Healthy: cascade declines or notes the corpus contains no dietary signal. Pathological: makes a recommendation grounded in nothing.

**9. Hyper-local probe.** *"Why did my heart rate spike on the morning of March 14, 2022?"*

Specific, single-day. Tests whether the Dicer can route to one day and whether the Describer admits it cannot know *why* without external context the corpus doesn't contain. Healthy: gives the data, declines to speculate on cause. Pathological: invents a plausible-sounding cause.

**10. Multi-source dedup probe.** *"How many calories did I burn in total in 2020?"*

`ActiveEnergyBurned` has three sources logging in parallel: Apple Watch, Balbus STEP, iPhone 8 AK. The Phase 0 indexer does not dedup. Honest answer: cascade flags the multi-source ambiguity and refuses to give a single total, or gives a range. Pathological: confidently sums everything and produces an inflated number. This is a known indexer limitation logged in `build_notes.md` — the prompt is a test of whether the cascade *recognises* the limitation, not of whether the data is clean.

## Tier 4 — meta

**11. Volume / compression.** *"Summarise my health since 2018 in one paragraph."*

Maximally broad. The Dicer must aggregate across the manifest, not retrieve raw rows. If the Describer receives anything close to raw daily data here, the cascade has failed at routing — the Dicer's whole point is to keep the Describer's working set bounded.

**12. Self-knowledge probe.** *"What's the most surprising thing in my watch data?"*

Open-ended. Tests whether the Describer leans on actual statistical anomalies in routed slices, or generates a plausible-sounding answer ungrounded in the data.

**13. Capability self-assessment.** *"What questions can you answer well from this data, and which can't you?"*

A meta-prompt. Useful at the end of the build week to see whether the cascade has developed any model of its own competence. Compare its self-assessment to your own observations from the build notes — divergence is the most interesting signal here.

## What to capture during the build

For each prompt run, log to `build_notes.md`:

- The Dicer's routing plan (the slice it produced)
- The Describer's answer or clarifying question
- Wall-clock time, broken down by stage
- Whether the answer was useful, wrong, evasive, or surprising
- Any observation that wasn't anticipated

Notes accumulate as dated entries. At the end of the build week, the notes become the input to Phase 1's pre-registration: the failure modes that actually showed up, not the ones we guessed at in advance.
